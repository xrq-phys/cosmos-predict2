from typing import Dict, Optional, Union

import torch
from torch import Tensor

from natten.functional import attention
from natten.types import (
    CausalArgTypeOrDed,
    DimensionType,
    DimensionTypeOrDed,
    KernelSchedule,
)
from natten.utils.checks import (
    additional_kv_tensor_checks,
    check_all_args,
    check_args_against_input,
    check_kernel_schedule,
    is_self_attention,
    na_tensor_checks,
)
from natten.backends import (
    choose_backend,
    cutlass_blackwell_fna_generic,
    cutlass_hopper_fna_generic,
    cutlass_fna_generic,
    flex_fna_generic,
)

from imaginaire.utils import log

def get_device_cc(device) -> int:
    """
    Returns the compute capability of a given torch device if it's a CUDA device, otherwise returns 0.

    Args:
        device: torch device.

    Returns:
        device_cc (int): compute capability in the SmXXX format (i.e. 90 for Hopper).
    """
    if torch.cuda.is_available() and torch.version.cuda and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor
    return 0

class NeighborhoodAttentionConfigs:
    # Configurations
    # Tuned for 720p and window sizes (24, 12, 24), (16, 12, 24), and stride (1, 4, 8).
    default_config = {
        "backend": "flex-fna",
        "q_tile_shape": (4, 4, 4),
        "kv_tile_shape": (4, 4, 4),
        "torch_compile": False,
    }
    default_config_cuda = {
        "backend": "cutlass-fna",
        "q_tile_shape": (4, 4, 4),
        "kv_tile_shape": (4, 4, 8),
        "backward_q_tile_shape": (4, 4, 8),
        "backward_kv_tile_shape": (4, 4, 8),
        "backward_use_pt_reduction": False,
    }
    inference_configs = {
        # Hopper (SM90)
        90: {
            "backend": "hopper-fna",
            "q_tile_shape": (4, 4, 8),
            "kv_tile_shape": (4, 4, 8),
        },
        # Blackwell (SM100)
        100: {
            "backend": "blackwell-fna",
            "q_tile_shape": (8, 4, 8),
            "kv_tile_shape": (4, 4, 8),
            "run_persistent_kernel": True,
        },
    }

    @classmethod
    def get_adaptive_parameters(cls, window_size, stride, dilation, is_causal, input_shape, base_size=None):
        window_size = tuple(w if w > 1 else x for x, w in zip(input_shape, window_size))
        stride = tuple(stride for _ in range(3)) if isinstance(stride, int) else tuple(x for x in stride)
        dilation = tuple(dilation for _ in range(3)) if isinstance(dilation, int) else tuple(x for x in dilation)
        is_causal = tuple(is_causal for _ in range(3)) if isinstance(is_causal, bool) else tuple(x for x in is_causal)

        # Scale window size and stride according to some base input size
        # For example, if window size is (8, 8, 8), stride is (1, 2, 2), for a base
        # input/feature map size of (16, 16, 16); then if the input feat map in this iteration
        # has shape (8, 8, 8), we should use window size (4, 4, 4), and stride (1, 1, 1).
        if base_size is not None:
            base_shape = tuple(b if b > 0 else x for x, b in zip(input_shape, base_size))

            scale = tuple(x / b for x, b in zip(input_shape, base_shape))

            scaled_window_size = tuple(min(max(2, round(w * s)), x) for w, s, x in zip(window_size, scale, input_shape))
            scaled_stride = tuple(min(max(1, round(st * s)), w) for w, s, st in zip(scaled_window_size, scale, stride))

            max_dilation = tuple(x // w for x, w in zip(input_shape, scaled_window_size))
            scaled_dilation = tuple(
                min(max(1, round(d * s)), max_d) for d, s, max_d in zip(dilation, scale, max_dilation)
            )

            window_size = scaled_window_size
            stride = scaled_stride
            dilation = scaled_dilation

        assert all(x >= w * d for x, w, d in zip(input_shape, window_size, dilation))
        assert all(w >= s for w, s in zip(window_size, stride))
        assert all(isinstance(c, bool) for c in is_causal)

        return window_size, stride, dilation, is_causal

    @classmethod
    def get_configuration(cls, q):
        device = q.device
        compute_cap = get_device_cc(device)
        requires_grad = False
        is_cuda = torch.cuda.is_available() and torch.version.cuda and device.type == "cuda"

        natten_configuration = cls.default_config
        if requires_grad and is_cuda:
            natten_configuration = cls.default_config_cuda
        elif is_cuda and compute_cap in cls.inference_configs.keys():
            natten_configuration = cls.inference_configs[compute_cap]

        return natten_configuration

class NeighborhoodAttentionRunner:
    @staticmethod
    def run(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size: DimensionTypeOrDed,
        stride: DimensionTypeOrDed = 1,
        dilation: DimensionTypeOrDed = 1,
        is_causal: Optional[CausalArgTypeOrDed] = False,
        scale: Optional[float] = None,
        inv_scale_o: float = 1.0,
        block_scaled_quantize: bool = False,
        attention_kwargs: Optional[Dict] = None,
        return_lse: bool = False,
        # Perf-related args
        backend: Optional[str] = None,
        q_tile_shape: Optional[DimensionType] = None,
        kv_tile_shape: Optional[DimensionType] = None,
        backward_q_tile_shape: Optional[DimensionType] = None,
        backward_kv_tile_shape: Optional[DimensionType] = None,
        backward_kv_splits: Optional[DimensionType] = None,
        backward_use_pt_reduction: bool = False,
        run_persistent_kernel: bool = True,
        kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
        torch_compile: bool = False,
    ) -> Tensor:
        """Modified from natten.functional.neighborhood_attention_generic"""

        na_tensor_checks(query, key, value)
        additional_kv_tensor_checks(query, key, value, None, None)
        kernel_schedule = check_kernel_schedule(kernel_schedule)

        na_dim = query.dim() - 3  # batch, heads, head_dim

        assert na_dim in [1, 2, 3]

        kernel_size, stride, dilation, is_causal = check_all_args(
            na_dim, kernel_size, stride, dilation, is_causal
        )

        check_args_against_input(
            query,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
        )

        if is_self_attention(query, kernel_size=kernel_size, is_causal=is_causal):
            log.debug(
                f"{query.shape=} with {kernel_size=} and {is_causal=} is self attention. "
                "Calling attention instead of neighborhood attention directly."
            )

            query_shape = query.shape
            query = query.flatten(1, na_dim)
            key = key.flatten(1, na_dim)
            value = value.flatten(1, na_dim)

            attn_kwargs = attention_kwargs or {}
            out: Tensor = attention(  # type: ignore[assignment]
                query,
                key,
                value,
                scale=scale,
                return_lse=return_lse,
                **attn_kwargs,
            )
            output_shape = [s for s in query_shape[:-1]] + [value.shape[-1]]
            if not return_lse:
                return out.reshape(*output_shape)
            else:
                return (out[0].reshape(*output_shape), out[1])

        scale = scale or query.shape[-1] ** -0.5

        backend = backend or choose_backend(query, key, value, torch_compile=torch_compile)

        if backend == "blackwell-fna":
            outputs = cutlass_blackwell_fna_generic(
                query=query,
                key=key,
                value=value,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                scale=scale,
                inv_scale_o=inv_scale_o,
                block_scaled_quantize=block_scaled_quantize,
                q_tile_shape=q_tile_shape,
                kv_tile_shape=kv_tile_shape,
                backward_q_tile_shape=backward_q_tile_shape,
                backward_kv_tile_shape=backward_kv_tile_shape,
                run_persistent_kernel=run_persistent_kernel,
                return_lse=return_lse,
            )

        elif backend == "hopper-fna":
            assert inv_scale_o == 1.0, "Hopper FNA doesn't support output-scaling"
            outputs = cutlass_hopper_fna_generic(
                query=query,
                key=key,
                value=value,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                scale=scale,
                q_tile_shape=q_tile_shape,
                kv_tile_shape=kv_tile_shape,
                backward_q_tile_shape=backward_q_tile_shape,
                backward_kv_tile_shape=backward_kv_tile_shape,
                kernel_schedule=kernel_schedule,
                return_lse=return_lse,
            )

        elif backend == "cutlass-fna":
            assert inv_scale_o == 1.0, "Generic CUTLASS FNA doesn't support output-scaling"
            outputs = cutlass_fna_generic(
                query=query,
                key=key,
                value=value,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                scale=scale,
                q_tile_shape=q_tile_shape,
                kv_tile_shape=kv_tile_shape,
                backward_q_tile_shape=backward_q_tile_shape,
                backward_kv_tile_shape=backward_kv_tile_shape,
                backward_kv_splits=backward_kv_splits,
                backward_use_pt_reduction=backward_use_pt_reduction,
                return_lse=return_lse,
            )

        elif backend == "flex-fna":
            assert inv_scale_o == 1.0, "Flex FNA doesn't support output-scaling"
            outputs = flex_fna_generic(
                query=query,
                key=key,
                value=value,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                scale=scale,
                q_tile_shape=q_tile_shape,
                kv_tile_shape=kv_tile_shape,
                torch_compile=torch_compile,
                return_lse=return_lse,
            )

        else:
            raise NotImplementedError(f"Unrecognized NATTEN backend {backend}.")

        return outputs