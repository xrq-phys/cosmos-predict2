import glob
import os
import gc
import argparse
import torch

import tensorrt as trt
from imaginaire.utils import log
from cosmos_predict2.utils.ext import trt_inference
from cosmos_predict2.utils.ext.dit_block_onnx2trt import build_dit_block_from_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_prefix", type=str, default="output/trt",
                        help="Directory to load ONNX files from.")
    parser.add_argument("--optimize", type=int, default=3, help="TRT optimization level")
    parser.add_argument("--skip_testrun", action="store_true", help="Skip testrun")
    parser.add_argument("--ring_attn", action="store_true", help="Ring SageAttn")
    parser.add_argument("--model_size", choices=["2B", "14B"], default="2B", type=str,
                        help="Size of the model to use for video-to-world generation")
    parser.add_argument("--resolution", choices=["480", "720"], default="720", type=str,
                        help="Resolution of the model to use for video-to-world generation")
    parser.add_argument("--auto", action="store_true",
                        help="Automatic attention backend & save to separate dir.")
    args = parser.parse_args()

    convert_dit_onnx2trt(args)

def is_default_domain(node):
    return node.domain is None or node.domain == "" or "onnx" in node.domain

def convert_dit_onnx2trt(args):
    pyt_stream = trt_inference._pyt_stream
    trt_stream = trt_inference._trt_stream
    trt_builder = trt.Builder(trt_inference._trt_logger)
    trt_runtime = trt.Runtime(trt_inference._trt_logger)

    # SelfAttn head count & dimension of all heads
    if args.model_size == "2B":
        HS, D = 16, 128
    elif args.model_size == "14B":
        HS, D = 40, 128
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")

    N, HX = 512, 8 # CrossAttn seq len & head count

    B, T = 1, 24 # Batch & frame count

    if args.resolution == "720":
        H, W = 44, 80 # 720p : 1280x704
    elif args.resolution == "480":
        H, W = 27, 48 # 480p : 768x480
    else:
        raise ValueError(f"Unsupported resolution: {args.resolution}")

    for onnx_file in glob.glob(f"{args.onnx_prefix}/cosmos_predict2_dit_net_block*.onnx"):

        # Args

        engine_file = '.'.join(onnx_file.split('.')[:-1] + ['engine'])
        ring_attn = args.ring_attn
        if args.auto:
            cc_major, cc_minor = torch.cuda.get_device_capability()

            # Attn backend
            if cc_major == 12 or (cc_major, cc_minor) == (8, 9) or cc_major == 9:
                ring_attn = True
            else:
                raise NotImplementedError("TODO: Add NATTEN PluginV3")

            # Engine file
            engine_file_dirs = engine_file.split('/')
            engine_file_dirs[-2] += f'.sm{cc_major}{cc_minor}'
            os.makedirs('/'.join(engine_file_dirs[:-1]), exist_ok=True)
            engine_file = '/'.join(engine_file_dirs)

        # Build TRT engine

        engine_serialized = build_dit_block_from_onnx(onnx_file, HS, D,
                                                      trt_builder=trt_builder,
                                                      optimization_level=args.optimize,
                                                      lo_res=args.resolution == "480",
                                                      oss_sageattn=ring_attn)
        with open(engine_file, "wb") as f:
            f.write(engine_serialized)
        log.info(f"Saved TRT engine {engine_file}")

        if args.skip_testrun:
            continue

        # Test execute.

        engine = trt_runtime.deserialize_cuda_engine(engine_serialized)
        context = trt_inference.create_execution_context_from_pool(engine)

        inspector = engine.create_engine_inspector()
        contains_layer = lambda name: any([
            name in inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE) for i in range(engine.num_layers)
        ])
        has_gpt_attention = contains_layer("GPTAttention")
        has_natten = contains_layer("NeighborhoodAttention")

        _alloc = lambda shape: torch.randn(shape, requires_grad=False, device="cuda", dtype=torch.bfloat16)
        _regin = lambda name, tensor: trt_inference.trt_set_tensor_check(context, name, tensor, check_shape=True)
        _regout = lambda name, tensor: trt_inference.trt_set_tensor_check(context, name, tensor, check_shape=False)
        x_B_T_H_W_D = _alloc((B, T, H, W, HS*D))
        emb_B_T_D = _alloc((B, T, HS*D))
        crossattn_emb = _alloc((B, N, HX*D)) # ?
        rope_emb_T_H_W_1_1_D = _alloc((T, H, W, 1, 1, D)).float()
        adaln_lora_B_T_3D = _alloc((B, T, 3*HS*D))
        context_lengths_B = torch.tensor([T*H*W], dtype=torch.int32, device="cuda") # Sequence length = T*H*W
        out = torch.empty_like(x_B_T_H_W_D)

        _regin('x_B_T_H_W_D', x_B_T_H_W_D)
        _regin('emb_B_T_D', emb_B_T_D)
        _regin('crossattn_emb', crossattn_emb)
        _regin('rope_emb_T_H_W_1_1_D', rope_emb_T_H_W_1_1_D)
        _regin('adaln_lora_B_T_3D', adaln_lora_B_T_3D)
        _regin('context_lengths', context_lengths_B)
        _regout('output', out)
        if has_gpt_attention:
            host_max_attention_window_1 = torch.tensor([0], dtype=torch.int32)
            host_sink_token_length_1 = torch.tensor([0], dtype=torch.int32)
            host_request_types_B = torch.tensor([0], dtype=torch.int32) # 0 for context phase.
            host_context_length_B = context_lengths_B.cpu()
            host_runtime_perf_knobs_1 = torch.tensor([0], dtype=torch.int64) # FIXME: How to set to nullptr?
            host_context_progress_1 = torch.tensor([0], dtype=torch.int64) # FIXME: How to set to nullptr?
            context.set_tensor_address("host_max_attention_window", host_max_attention_window_1.data_ptr())
            context.set_tensor_address("host_sink_token_length", host_sink_token_length_1.data_ptr())
            context.set_tensor_address("host_request_types", host_request_types_B.data_ptr())
            context.set_tensor_address("host_context_length", host_context_length_B.data_ptr())
            context.set_tensor_address("host_runtime_perf_knobs", host_runtime_perf_knobs_1.data_ptr())
            context.set_tensor_address("host_context_progress", host_context_progress_1.data_ptr())
        if has_natten:
            host_video_size_3 = torch.tensor([T, H, W], dtype=torch.int32)
            context.set_tensor_address("host_video_size", host_video_size_3.data_ptr())
        if ring_attn:
            host_cp_size_1 = torch.tensor([1], dtype=torch.int32)
            host_cp_rank_1 = torch.tensor([0], dtype=torch.int32)
            host_cp_group_72 = torch.tensor([0], dtype=torch.int32) # Only [:cp_size] will be used
            context.set_tensor_address("host_cp_size", host_cp_size_1.data_ptr())
            context.set_tensor_address("host_cp_rank", host_cp_rank_1.data_ptr())
            context.set_tensor_address("host_cp_group", host_cp_group_72.data_ptr())

        trt_stream.wait_stream(pyt_stream)
        context.execute_async_v3(trt_stream.cuda_stream)
        pyt_stream.wait_stream(trt_stream)

        del context
        del engine
        del x_B_T_H_W_D
        del rope_emb_T_H_W_1_1_D
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
