from natten._libnatten import blackwell_fmha_forward
from natten.backends.blackwell_fmha import block_quantize_tensor
from typing import Optional
import torch
from .cp_allocators import BaseAllocator


Q_TILE = 256
KV_TILE = 128


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

    q_quant, q_scale = block_quantize_tensor(q, Q_TILE)
    k_quant, k_scale, k_mean = block_quantize_tensor(k, KV_TILE, smooth=True)
    v_quant, v_scale = block_quantize_tensor(v, KV_TILE)

    return (
        torch.empty([0], dtype=q.dtype, device=q.device), # CUTLASS backend doesn't require pre-allocated output
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

@torch.compile
def compute_lse_correction_NHD(q, k_mean):
    return torch.matmul(q.transpose(1, 2).to(torch.float32), k_mean.unsqueeze(-1)).squeeze(-1)

@torch.compile
def compute_lse_correction_HND(q, k_mean):
    return torch.matmul(q.to(torch.float32), k_mean.unsqueeze(-1)).squeeze(-1)

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
        lse_correction = compute_lse_correction_NHD(q, k_mean)
    elif tensor_layout.upper() == "HND":
        orig_seq_len = q.shape[2]
        lse_correction = compute_lse_correction_HND(q, k_mean)
    else:
        raise RuntimeError(f"Unknown tensor layout: {tensor_layout}")

    assert sm_version >= 100 and sm_version < 120, "Unsupported SM version"

    out = torch.empty_like(q_quant)
    lse = torch.empty(
        q_quant.shape[:-1], dtype=torch.float32, device=q_quant.device
    )

    # Other FMHA kernel parameters
    run_persistent_kernel = False
    softmax_scale = channels**-0.5
    inv_scale_o = 1.0 # 1.0 is fine for cosmos. Need to correct in the future

    blackwell_fmha_forward(
        out,
        q_quant, k_quant, v_quant,
        lse,
        softmax_scale,
        inv_scale_o,
        q_scale, k_scale, v_scale,
        Q_TILE, KV_TILE,
        run_persistent_kernel,
    )

    # For padded sequence
    out = out[:, :orig_seq_len]
    lse = lse.transpose(-2, -1)[..., :orig_seq_len] # B H S

    if tensor_layout.upper() == "HND":
        out = out.transpose(1, 2)

    return (
        out.to(q.dtype),
        lse + lse_correction * softmax_scale,
    )