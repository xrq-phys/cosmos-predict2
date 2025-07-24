import tempfile
import numpy as np
import torch
import onnx

import tensorrt as trt
import onnx_graphsurgeon as gs

from imaginaire.utils import log
from cosmos_predict2.utils.ext import trt_inference


def is_default_domain(node):
    return node.domain is None or node.domain == "" or "onnx" in node.domain

def build_dit_block_from_onnx(
    onnx_path, num_heads, head_size,
    trt_builder=None,
    optimization_level=3,
    lo_res=False,
    cp_size=1,
    cp_rank=0,
    cp_group=None,
    oss_sageattn=False,
    fix_B=None,
    fix_T=None,
    fix_H=None,
    fix_W=None,
    use_fp8_context_fmha=True,
):
    if trt_builder is None:
        trt_builder = trt.Builder(trt_inference._trt_logger)
    trt_explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    trt_strongly_typed = 1 << (int)(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    if cp_group is None:
        cp_group = list(range(cp_size))

    # Set SageAttention block size according to device architecture
    cc_major, cc_minor = torch.cuda.get_device_capability()
    if cc_major == 9:
        # Hopper
        SAGE_ATTN_Q_BLOCK_SIZE = 64
        SAGE_ATTN_K_BLOCK_SIZE = 64
        SAGE_ATTN_V_BLOCK_SIZE = 256
    elif (cc_major, cc_minor) in {(8, 9), (12, 0)}:
        # Ada (& Blackwell)
        SAGE_ATTN_Q_BLOCK_SIZE = 64
        SAGE_ATTN_K_BLOCK_SIZE = 32
        SAGE_ATTN_V_BLOCK_SIZE = 32
    else:
        SAGE_ATTN_Q_BLOCK_SIZE = 1
        SAGE_ATTN_K_BLOCK_SIZE = 1
        SAGE_ATTN_V_BLOCK_SIZE = 1

    # SelfAttn head count & dimension of all heads
    HS, D = num_heads, head_size

    # CrossAttn seq len & head count
    N, HX = 512, 8

    # Sequence
    B, T, H, W = 1, 24, 44, 80
    T_MIN = 1
    H_MIN = 27 # Min 480p
    W_MIN = 27
    T_MAX = T
    H_MAX = 80 # Max 720p
    W_MAX = 80 # Max 720p (rotated)

    # Limit engine size for low-memory (consumer-grade) devices
    device_memory = torch.cuda.get_device_properties().total_memory
    if device_memory < 33 * 1024**3:
        H_MAX = H
        W_MAX = W

    if lo_res:
        H = H_MIN
        W = 48 # 480p : 768
        H_MAX = H_MIN
        W_MAX = W

    # If building for a specific size (used in JIT engines)
    if fix_B is not None:
        B = fix_B
    if fix_T is not None:
        T = fix_T
        T_MAX = T * cp_size
        T_MIN = T
    if fix_H is not None:
        H = fix_H
        H_MAX = H
        H_MIN = H
    if fix_W is not None:
        W = fix_W
        W_MAX = W
        W_MIN = W

    # Preprocess the ONNX graph
    log.info(f"Preprocessing ONNX graph from {onnx_path}")
    onnx_graph = gs.import_onnx(onnx.load(onnx_path))

    # BUG: SageAttention will give NaN if the model doesn't contain quantization
    has_quant_nodes = len([node for node in onnx_graph.nodes if "quantize" in str(node.op).lower() or "nunchaku" in str(node.op).lower()]) > 0

    # GPTAttention node presence check
    has_gpt_attention = len([node for node in onnx_graph.nodes if node.op == "GPTAttention"]) > 0

    # NATTEN node presence check
    has_natten = len([node for node in onnx_graph.nodes if node.op == "NeighborhoodAttention"]) > 0

    # Additional sequence length tensor for TRTLLM attention ops
    context_lengths = gs.Variable(name="context_lengths", shape=(B,), dtype=np.int32)

    # Create host-side tensors for GPTAttention
    host_max_attention_window = gs.Variable(name="host_max_attention_window", shape=(1,), dtype=np.int32) # Full context, or unused
    host_sink_token_length = gs.Variable(name="host_sink_token_length", shape=(1,), dtype=np.int32)       # Unsure what's it for. Shape is (1,)
    host_request_types = gs.Variable(name="host_request_types", shape=(1,), dtype=np.int32)               # Please supply 0 (CONTEXT). Shape is (batch, ) (currently 1).
    host_context_length = gs.Variable(name="host_context_length", shape=(1,), dtype=np.int32)             # Context lengths (T*H*W). Shape is (batch, ) (currently 1).
    host_runtime_perf_knobs = gs.Variable(name="host_runtime_perf_knobs", shape=(1,), dtype=np.int64)     # Optional. Not sure about shape. Safe to supply nullptr.
    host_context_progress = gs.Variable(name="host_context_progress", shape=(1,), dtype=np.int64)         # Optional. Not sure about shape. Safe to supply nullptr.

    # Create host-side tensors for RingSageAttentionFusedQKV
    host_cp_size = gs.Variable(name="host_cp_size", shape=(1,), dtype=np.int32)
    host_cp_rank = gs.Variable(name="host_cp_rank", shape=(1,), dtype=np.int32)
    host_cp_group = gs.Variable(name="host_cp_group", shape=(72,), dtype=np.int32)

    # Create host-side tensors for NeighborhoodAttention
    host_video_size = gs.Variable(name="host_video_size", shape=(3,), dtype=np.int32)

    onnx_graph.inputs.append(context_lengths)
    if has_gpt_attention:
        # NOTE: Host addresses can only be supplied from inputs
        onnx_graph.inputs.append(host_max_attention_window)
        onnx_graph.inputs.append(host_sink_token_length)
        onnx_graph.inputs.append(host_request_types)
        onnx_graph.inputs.append(host_context_length)
        onnx_graph.inputs.append(host_runtime_perf_knobs)
        onnx_graph.inputs.append(host_context_progress)
    if has_natten:
        onnx_graph.inputs.append(host_video_size)
    if has_natten or oss_sageattn:
        onnx_graph.inputs.append(host_cp_size)
        onnx_graph.inputs.append(host_cp_rank)
        onnx_graph.inputs.append(host_cp_group)

    for inode in range(len(onnx_graph.nodes)):
        node = onnx_graph.nodes[inode]

        if node.domain == "tensorrt_llm":

            # Setups for BertAttention
            if node.op == "BertAttention":
                # Sequence length tensor
                node.inputs.append(context_lengths)

                # CP metadata are attributes
                node.attrs["cp_size"] = cp_size
                node.attrs["cp_rank"] = cp_rank
                node.attrs["cp_group"] = cp_group

                # Enable SageAttn with architecture-specific block sizes
                node.attrs["sage_attn"] = int(all([
                    SAGE_ATTN_Q_BLOCK_SIZE != 1, # Kernel is available
                    has_quant_nodes,             # Has at least one FP8 node (bug otherwise)
                ]))
                node.attrs["sage_attn_q_block_size"] = SAGE_ATTN_Q_BLOCK_SIZE
                node.attrs["sage_attn_k_block_size"] = SAGE_ATTN_K_BLOCK_SIZE
                node.attrs["sage_attn_v_block_size"] = SAGE_ATTN_V_BLOCK_SIZE
                if oss_sageattn:
                    op_attrs = { "num_heads": node.attrs["num_heads"], "head_size": node.attrs["head_size"] }
                    node.attrs = op_attrs
                    node.domain = "Cosmos"
                    node.op = "RingSageAttentionFusedQKV"
                    node.name = node.name.replace("BertAttention", "RingSageAttentionFusedQKV")
                    node.inputs.append(host_cp_size)
                    node.inputs.append(host_cp_rank)
                    node.inputs.append(host_cp_group)

            # Setups for GPTAttention
            if node.op == "GPTAttention":
                # Before GPTAttention:
                # First, find the subsequent quantization layer to get the output scale
                if use_fp8_context_fmha:
                    max_dig_down = 10
                    subsequent_node = node
                    for _ in range(max_dig_down):
                        log.info(f"Searching for quantization node from: {subsequent_node.op}")
                        subsequent_nodes = [ n for n in onnx_graph.nodes if subsequent_node.outputs[0] in n.inputs ]
                        if len(subsequent_nodes) > 1:
                            raise ValueError("Multiple subsequent nodes found for GPTAttention output without any quantization. "
                                             "Was this graph exported with --modelopt_state?")
                        if len(subsequent_nodes) == 0:
                            raise ValueError("Output of GPTAttention is not consumed by any node. Graph is invalid.")
                        subsequent_node = subsequent_nodes[0]
                        if subsequent_node.op == "TRT_FP8QuantizeLinear":
                            break
                    if subsequent_node.op != "TRT_FP8QuantizeLinear":
                        raise ValueError(f"No quantization layer found within {max_dig_down} nodes after GPTAttention. "
                                         "Was this graph exported with --modelopt_state?")
                    oproj_quant_scale_v = subsequent_node.inputs[1]
                    oproj_quant_const_node = [ n for n in onnx_graph.nodes if n.outputs[0] == oproj_quant_scale_v ][0]
                    assert oproj_quant_const_node.op == "Constant"

                    # Add that scale to GPTAttention's input
                    attn_out_dequant_scale_val = oproj_quant_const_node.attrs["value"].values
                    log.info(f"Quantization scale: {attn_out_dequant_scale_val}")
                    attn_out_quant_scale_c = gs.Constant(name=f"GPTAttention{inode}/out/quant/const",
                                                         values=np.array(np.reciprocal(attn_out_dequant_scale_val)))
                    attn_out_quant_scale_v = gs.Variable(name=f"GPTAttention{inode}/out/quant/var")
                    onnx_graph.nodes.append(gs.Node(op="Constant",
                                                    name=f"GPTAttention{inode}/out/quant/node",
                                                    outputs=[attn_out_quant_scale_v],
                                                    attrs={"value": attn_out_quant_scale_c}))

                    # If CP is enabled, try to do quantization from TRT instead of GPTAttention
                    if cp_size > 1:
                        attn_in_cast_out = gs.Variable(name=f"GPTAttention{inode}/in/cast/out")
                        onnx_graph.nodes.append(gs.Node(op="Cast",
                                                        name=f"GPTAttention{inode}/in/cast/op",
                                                        inputs=[node.inputs[0]],
                                                        outputs=[attn_in_cast_out],
                                                        attrs={"to": onnx.TensorProto.FLOAT}))

                        unit_scale = np.array(1.0, dtype=np.float32)
                        attn_in_quant_scale_c = gs.Constant(name=f"GPTAttention{inode}/in/quant/const",
                                                            values=unit_scale)
                        attn_in_quant_scale_v = gs.Variable(name=f"GPTAttention{inode}/in/quant/var")
                        onnx_graph.nodes.append(gs.Node(op="Constant",
                                                        name=f"GPTAttention{inode}/in/quant/node",
                                                        outputs=[attn_in_quant_scale_v],
                                                        attrs={"value": attn_in_quant_scale_c}))
                        attn_in_quant_out = gs.Variable(name=f"GPTAttention{inode}/in/quant/out")
                        onnx_graph.nodes.append(gs.Node(op="TRT_FP8QuantizeLinear",
                                                        name=f"GPTAttention{inode}/in/quant/op",
                                                        domain="trt",
                                                        inputs=[attn_in_cast_out, attn_in_quant_scale_v],
                                                        outputs=[attn_in_quant_out]))
                        node.inputs[0] = attn_in_quant_out

                # At GPTAttention:
                # Supplimentary tensors in order
                node.inputs.append(host_max_attention_window)    # IdxEntry::HOST_MAX_ATTENTION_WINDOW
                node.inputs.append(host_sink_token_length)       # IdxEntry::HOST_SINK_TOKEN_LENGTH
                node.inputs.append(context_lengths)              # IdxEntry::CONTEXT_LENGTHS
                node.inputs.append(host_request_types)           # IdxEntry::REQUEST_TYPES
                if use_fp8_context_fmha:
                    node.inputs.append(attn_out_quant_scale_v)   # IdxEntry::ATTENTION_OUTPUT_QUANTIZATION_SCALE
                node.inputs.append(host_context_length)          # IdxEntry::HOST_CONTEXT_LENGTH
                node.inputs.append(host_runtime_perf_knobs)      # IdxEntry::HOST_RUNTIME_PERF_KNOBS: Judged `USED` from attributes, but is optional.
                node.inputs.append(host_context_progress)        # IdxEntry::HOST_CONTEXT_PROGRESS: Judged `USED` from attributes, but is optional.

                # Enable FP8 quantization
                node.attrs["use_fp8_context_fmha"] = use_fp8_context_fmha
                node.attrs["skip_preprocess"] = use_fp8_context_fmha and cp_size > 1

                # CP metadata are attributes
                node.attrs["cp_size"] = cp_size
                node.attrs["cp_rank"] = cp_rank
                node.attrs["cp_group"] = cp_group

                # After GPTAttention:
                # Add dequantization layer
                if use_fp8_context_fmha:
                    attn_out_dequant_scale_c = gs.Constant(name=f"GPTAttention{inode}/out/dequant/const",
                                                           values=attn_out_dequant_scale_val)
                    attn_out_dequant_scale_v = gs.Variable(name=f"GPTAttention{inode}/out/dequant/var")
                    onnx_graph.nodes.append(gs.Node(op="Constant",
                                                    name=f"GPTAttention{inode}/out/dequant/node",
                                                    outputs=[attn_out_dequant_scale_v],
                                                    attrs={"value": attn_out_dequant_scale_c}))
                    attn_out_dequant_out = gs.Variable(name=f"GPTAttention{inode}/out/dequant/out")
                    attn_out_dequant = gs.Node(op="TRT_FP8DequantizeLinear",
                                               name=f"GPTAttention{inode}/out/dequant/op",
                                               domain="trt",
                                               inputs=[node.outputs[0], attn_out_dequant_scale_v],
                                               outputs=[attn_out_dequant_out])
                    for n in onnx_graph.nodes:
                        for i in range(len(n.inputs)):
                            if n.inputs[i] == node.outputs[0]:
                                n.inputs[i] = attn_out_dequant_out
                    onnx_graph.nodes.append(attn_out_dequant)

        if node.domain == "Cosmos" and node.op == "NeighborhoodAttention":
            assert len(node.inputs) == 3, "Where are our Q, K, and V?"
            node.inputs.append(context_lengths)
            node.inputs.append(host_video_size)
            node.inputs.append(host_cp_size)
            node.inputs.append(host_cp_rank)
            node.inputs.append(host_cp_group)

        if node.domain == "Cosmos" and node.op in ["QSmoothFactor", "KSmoothFactor"]:
            node.attrs["cp_size"] = cp_size
            node.attrs["cp_rank"] = cp_rank
            node.attrs["cp_group"] = cp_group

        if is_default_domain(node) and node.op == "Gelu":
            # Fuse GELU into its subsequent NunchakuGemm (for FFN)

            # Find nodes that consume this GeLU's output
            subsequent_gemms = [
                n
                for n in onnx_graph.nodes if all([
                    node.outputs[0] in n.inputs,
                    n.domain == "tensorrt_llm",
                    n.op == "NunchakuGemm"
                ])
            ]
            for gemm in subsequent_gemms:
                assert gemm.inputs.index(node.outputs[0]) == 0, "GeLU not feeding into NunchakuGemm's activation input. Why?"
                gemm.inputs[0] = node.inputs[0]
                gemm.attrs["fuse_lu"] = 1

    onnx_graph.cleanup().toposort()

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        # Dump the attribute-corrected ONNX to a temporary file
        onnx.save(gs.export_onnx(onnx_graph), tmp.name)

        # Create TRT network from ONNX
        network = trt_builder.create_network(trt_explicit_batch | trt_strongly_typed)
        parser = trt.OnnxParser(network, trt_builder.logger)
        log.info(f"Loading ONNX model from {tmp.name}")
        status = parser.parse_from_file(tmp.name)

    if not status:
        log.error("Failed to read ONNX")
        for ierr in range(parser.num_errors):
            log.error(f'{parser.get_error(ierr)}')

    config = trt_builder.create_builder_config()
    # No precision flags for strongly typed networks
    if False:
        config.set_flag(trt.BuilderFlag.BF16)
        config.set_flag(trt.BuilderFlag.FP8)
        config.set_flag(trt.BuilderFlag.FP4)
    config.builder_optimization_level = optimization_level

    profile = trt_builder.create_optimization_profile()
    profile.set_shape('x_B_T_H_W_D',          min=[B, T_MIN, H_MIN, W_MIN, HS*D], opt=[B, T, H, W, HS*D], max=[B, T_MAX, H_MAX, W_MAX, HS*D])
    profile.set_shape('emb_B_T_D',            min=[B, T_MIN, HS*D],               opt=[B, T, HS*D],       max=[B, T_MAX, HS*D])
    profile.set_shape('crossattn_emb',        min=[B, T_MIN, HX*D],               opt=[B, N, HX*D],       max=[B, N, HX*D])
    profile.set_shape('rope_emb_T_H_W_1_1_D', min=[T_MIN, H_MIN, W_MIN, 1, 1, D], opt=[T, H, W, 1, 1, D], max=[T_MAX, H_MAX, W_MAX, 1, 1, D])
    profile.set_shape('adaln_lora_B_T_3D',    min=[B, T_MIN, 3*HS*D],             opt=[B, T, 3*HS*D],     max=[B, T_MAX, 3*HS*D])
    config.add_optimization_profile(profile)

    log.info("Building TRT engine from ONNX")
    engine_serialized = trt_builder.build_serialized_network(network, config)
    log.info(f"Built TRT engine")

    return engine_serialized
