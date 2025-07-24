import torch

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
