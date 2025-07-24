import ctypes
from typing import Tuple
import threading
import torch
import numpy as np
from numpy.typing import NDArray

import tensorrt as trt
import tensorrt.plugin as trtp
from einops import rearrange

from imaginaire.utils import log

# Determine device mem -> scratch size
_trt_logger = trt.Logger(trt.Logger.WARNING)
_trt_runtime = trt.Runtime(_trt_logger)
_pyt_stream = torch.cuda.current_stream()
_trt_stream = torch.cuda.Stream()
_trt_existing_context = threading.local()
_trt_existing_context.device_memory = None
_trt_existing_context.execution_contexts = []


_trt2pt_dtype = {
    # TODO: Merge with export_attn.py
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF:  torch.float16,
    trt.DataType.INT8:  torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL:  torch.bool,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.FP8:   torch.float8_e4m3fn,
    trt.DataType.BF16:  torch.bfloat16,
    trt.DataType.INT64: torch.int64,
}

def create_execution_context_from_pool(engine):
    # Currently each engine only has one profile
    req_size = engine.get_device_memory_size_for_profile(0)
    if _trt_existing_context.device_memory is None or _trt_existing_context.device_memory.numel() < req_size:
        log.info(f"Reallocating {req_size/1024**3:.2f}G of scratch space")
        # Reallocate new scratch space
        _trt_existing_context.device_memory = torch.empty(req_size, dtype=torch.int8, device='cuda')
        # Update the created contexts
        for context in _trt_existing_context.execution_contexts:
            context.device_memory = _trt_existing_context.device_memory.data_ptr()
        # Clear CUDA caches
        torch.cuda.empty_cache()
    # Create new context & attach the current scratch space
    context = engine.create_execution_context(strategy=trt.tensorrt.ExecutionContextAllocationStrategy.USER_MANAGED)
    context.device_memory = _trt_existing_context.device_memory.data_ptr()
    # Make a record for the context created
    _trt_existing_context.execution_contexts.append(context)
    return context

def trt_set_tensor_check(context, name, tensor, check_shape=True):
    assert tensor.is_contiguous(), f"contiguous tensor expected: {name}"
    assert _trt2pt_dtype[context.engine.get_tensor_dtype(name)] == tensor.dtype, "incompatible dtype"
    if check_shape:
        assert context.set_input_shape(name, tensor.shape), "incompatible shape"
    context.set_tensor_address(name, tensor.data_ptr())

# Utils for attention plugins

def _recast(t):
    t_ = t
    if t.dtype == trt.DataType.BF16:
        # torch.as_tensor would fail inferring BF16 from __cuda_array_interface__ due to lack of such support in NumPy
        # have to manually workaround dtype
        t_._immutable = False
        t_.dtype = trt.DataType.HALF
        t_._immutable = True
        return torch.as_tensor(t_, device='cuda').view(torch.bfloat16)
    else:
        return torch.as_tensor(t_, device='cuda')

# Attention plugin #1: Ring attention with SageAttention backend (PluginV3)

try:
    from cosmos_predict2.utils.ext.sageattn_ring import ring_sage_attention_exec

    @trtp.register("Cosmos::RingSageAttentionFusedQKV")
    def ring_sage_attention_plugin_v3(
        input: trtp.TensorDesc,
        seq_len: trtp.TensorDesc, # Ignored
        host_cp_size : trtp.TensorDesc,
        host_cp_rank : trtp.TensorDesc,
        host_cp_group : trtp.TensorDesc,
        num_heads : int,
        head_size : int,
    ) -> trtp.TensorDesc:
        out_desc = input.like()
        batch, max_seq, _, hidden_dim = input.shape_expr
        out_desc.shape_expr = [batch, max_seq, hidden_dim]
        return out_desc

    @trtp.impl("Cosmos::RingSageAttentionFusedQKV")
    def ring_sage_attention_plugin_v3_impl(
        input: trtp.Tensor,
        seq_len: trtp.Tensor, # Ignored
        host_cp_size : trtp.Tensor,
        host_cp_rank : trtp.Tensor,
        host_cp_group : trtp.Tensor,
        num_heads : int,
        head_size : int,
        outputs: Tuple[trtp.Tensor],
        stream: int
    ) -> None:
        cp_size1 = np.ctypeslib.as_array(ctypes.c_int.from_address(host_cp_size.data_ptr)).item()
        cp_rank1 = np.ctypeslib.as_array(ctypes.c_int.from_address(host_cp_rank.data_ptr)).item()
        cp_group72 = np.ctypeslib.as_array((ctypes.c_int * cp_size1).from_address(host_cp_group.data_ptr)).tolist()
        ext_stream = torch.cuda.ExternalStream(stream)

        fused_qkv_t, out_t = map(_recast, (input, outputs[0]))
        fused_qkv_t = rearrange(fused_qkv_t, "b s t (h d) -> b s t h d", h=num_heads, d=head_size)
        q_t = fused_qkv_t[:, :, 0]
        k_t = fused_qkv_t[:, :, 1]
        v_t = fused_qkv_t[:, :, 2]

        out1 = ring_sage_attention_exec(q_t, k_t, v_t, ext_stream, cp_rank1, cp_size1, cp_group72)

        with torch.cuda.stream(ext_stream):
            out_t.copy_(out1)

except Exception as e:
    log.warning(f"Cannot create RingSageAttentionFusedQKV plugin: {e}.")

# Attention plugin #2: NATTEN (PluginV3)

try:
    import natten
    from cosmos_predict2.utils.ext.natten_interface import NeighborhoodAttentionConfigs
    from cosmos_predict2.utils.ext.natten_a2a import a2a_netten_exec

    natten.use_kv_parallelism_in_fused_na(True)
    natten.set_memory_usage_preference("unrestricted")

    @trtp.register("Cosmos::NeighborhoodAttention")
    def natten_plugin_v3(
        q : trtp.TensorDesc,
        k : trtp.TensorDesc,
        v : trtp.TensorDesc,
        seq_len: trtp.TensorDesc, # Ignored
        host_video_size : trtp.TensorDesc,
        host_cp_size : trtp.TensorDesc,
        host_cp_rank : trtp.TensorDesc,
        host_cp_group : trtp.TensorDesc,
        window_size : NDArray[np.int32],
        stride : NDArray[np.int32],
        dilation : NDArray[np.int32],
        base_size : NDArray[np.int32],
    ) -> trtp.TensorDesc:
        out_desc = q.like()
        batch, max_seq, num_heads, head_dim = q.shape_expr
        out_desc.shape_expr = [batch, max_seq, num_heads * head_dim]
        return out_desc

    @trtp.impl("Cosmos::NeighborhoodAttention")
    def natten_plugin_v3_impl(
        q : trtp.Tensor,
        k : trtp.Tensor,
        v : trtp.Tensor,
        seq_len: trtp.Tensor, # Ignored
        host_video_size : trtp.Tensor,
        host_cp_size : trtp.Tensor,
        host_cp_rank : trtp.Tensor,
        host_cp_group : trtp.Tensor,
        window_size : NDArray[np.int32],
        stride : NDArray[np.int32],
        dilation : NDArray[np.int32],
        base_size : NDArray[np.int32],
        outputs : Tuple[trtp.Tensor],
        stream : int
    ):
        T, H, W = np.ctypeslib.as_array((ctypes.c_int * 3).from_address(host_video_size.data_ptr)).tolist()
        cp_size1 = np.ctypeslib.as_array(ctypes.c_int.from_address(host_cp_size.data_ptr)).item()
        cp_rank1 = np.ctypeslib.as_array(ctypes.c_int.from_address(host_cp_rank.data_ptr)).item()
        cp_group72 = np.ctypeslib.as_array((ctypes.c_int * cp_size1).from_address(host_cp_group.data_ptr)).tolist()
        ext_stream = torch.cuda.ExternalStream(stream)

        def _convert_natten_args(arg: NDArray[np.int32]):
            arg_l = arg.tolist()
            if len(arg_l) > 1:
                return tuple(arg_l)
            else:
                if isinstance(arg_l, list):
                    return arg_l[0]
                else:
                    return arg_l

        q_t, k_t, v_t = map(
            lambda t: _recast(t).unflatten(1, (T, H, W)),
            (q, k, v)
        )
        out_t = _recast(outputs[0])
        window_size_arg, stride_arg, dilation_arg, base_size_arg = map(
            _convert_natten_args,
            (window_size, stride, dilation, base_size)
        )

        natten_configuration = NeighborhoodAttentionConfigs.get_configuration(q_t)
        window_size_arg, stride_arg, dilation_arg, is_causal = NeighborhoodAttentionConfigs.get_adaptive_parameters(
            window_size=window_size_arg,
            stride=stride_arg,
            dilation=dilation_arg,
            is_causal=False,
            input_shape=(T*cp_size1, H, W),
            base_size=base_size_arg,
        )

        out1 = a2a_netten_exec(
            query=q_t,
            key=k_t,
            value=v_t,
            cp_size=cp_size1,
            cp_rank=cp_rank1,
            cp_group=cp_group72,
            stream=ext_stream,
            kernel_size=window_size_arg,
            stride=stride_arg,
            dilation=dilation_arg,
            is_causal=is_causal,
            **natten_configuration,
        )
        out1 = out1.flatten(1, 3).flatten(-2)

        with torch.cuda.stream(ext_stream):
            out_t.copy_(out1)

except Exception as e:
    log.warning(f"Cannot create NeighborhoodAttention plugin: {e}")
