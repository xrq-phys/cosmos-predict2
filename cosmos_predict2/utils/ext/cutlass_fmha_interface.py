from cutlass_blackwell_fmha import TorchFmhaRunner, sage_quant
from types import SimpleNamespace
from typing import Optional
import torch
from .cp_allocators import BaseAllocator

_cutlass_device_apis = SimpleNamespace()
_cutlass_device_apis.runners = {}

@torch.compiler.disable
def cutlass_fmha_fp8_quantize_all(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    tensor_layout: str = "NHD",
    alloc_kv: Optional[BaseAllocator] = None,
    sm_version: int = 100,
):
    if tensor_layout.upper() == "HND":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    elif tensor_layout.upper() == "NHD":
        pass
    else:
        raise RuntimeError(f"Unknown tensor layout: {tensor_layout}")

    assert sm_version >= 100 and sm_version < 120, "Unsupported SM version"

    dtype = q.dtype
    channels = q.shape[-1]
    if (dtype, channels) not in _cutlass_device_apis.runners.keys():
        runner = TorchFmhaRunner.torch_fmha_runner(torch.float8_e4m3fn, dtype, channels)
        _cutlass_device_apis.runners[(dtype, channels)] = runner
    else:
        runner = _cutlass_device_apis.runners[(dtype, channels)]


    q_quant, k_quant, v_quant, q_scale, k_scale, v_scale, k_mean = \
        sage_quant.sage_quant(runner, q, k, v, smooth=True, use_v_scale=False)

    return (
        torch.empty([0], dtype=dtype, device=q.device), # CUTLASS backend doesn't require pre-allocated output
        {
            'q_int8': q_quant,
            'q_scale': q_scale,
            'km': k_mean,
            'k_int8': k_quant,
            'k_scale': k_scale,
            'v_fp8': v_quant,
            'v_scale': v_scale,
        }
    )

@torch.compiler.disable
def cutlass_fmha_fp8_from_quantized(
    q: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    k_mean: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    o_dummy: torch.Tensor,
    channels: int,
    tensor_layout: str = "NHD",
    sm_version: int = 100,
):
    if tensor_layout.upper() == "NHD":
        orig_seq_len = q.shape[1]
        lse_correction = torch.matmul(q.transpose(1, 2).to(torch.float32), k_mean.unsqueeze(-1)).squeeze(-1)
    elif tensor_layout.upper() == "HND":
        orig_seq_len = q.shape[2]
        lse_correction = torch.matmul(q.to(torch.float32), k_mean.unsqueeze(-1)).squeeze(-1)
    else:
        raise RuntimeError(f"Unknown tensor layout: {tensor_layout}")

    assert sm_version >= 100 and sm_version < 120, "Unsupported SM version"

    runner = _cutlass_device_apis.runners[(q.dtype, channels)]
    v_scale_nil = torch.empty((0), device=q.device)
    out, lse = runner(q_quant, k_quant, v_quant, q_scale, k_scale, v_scale_nil, torch.cuda.current_stream().cuda_stream)

    # The same default value was applied in cutlass::device
    softmax_scale = channels**-0.5

    # For padded sequence
    out = out[:, :orig_seq_len]
    lse = lse[..., :orig_seq_len]

    if tensor_layout.upper() == "HND":
        out = out.transpose(1, 2)

    return (
        out,
        lse + lse_correction * softmax_scale,
    )