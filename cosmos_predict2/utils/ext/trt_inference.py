import threading
import torch

import tensorrt as trt
import tensorrt_llm as _ # registers attention plugins
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
