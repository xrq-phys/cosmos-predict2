# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Dict

import numpy as np
import torch
import transformer_engine as te
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
from torchvision import transforms
from transformer_engine.pytorch.attention import DotProductAttention, apply_rotary_pos_emb

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.networks.a2a_cp import MinimalA2AAttnOp
from cosmos_predict2.networks.model_weights_stats import WeightTrainingStat
from cosmos_predict2.networks.selective_activation_checkpoint import SACConfig as _SACConfig
from cosmos_predict2.utils.context_parallel import split_inputs_cp
from imaginaire.utils import log
from imaginaire.utils.graph import create_cuda_graph


# selective activation checkpoint; only apply to the minimal v4 model. if there are change in the networks, some policy will not work as we expect.
def predict2_2B_720_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            # from imaginaire.utils import log
            # log.info(f"op_count_key: {op_count_key}, op_count[op_count_key]: {op_count[op_count_key]}, {args[0].shape}, {args[1].shape}")
            # there are totally 6 + 4 + 4 + 2 = 16 block
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:  # recompute self attn first 3 linear layers
                return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_14B_720_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            # from imaginaire.utils import log
            # log.info(f"op_count_key: {op_count_key}, op_count[op_count_key]: {op_count[op_count_key]}, {args[0].shape}, {args[1].shape}")
            # there are totally 6 + 4 + 4 + 2 = 16 block
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:  # recompute self attn first 1 linear layers
                return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def linear_selfattn_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


class CheckpointMode(str, Enum):
    NONE = "none"
    MM_ONLY = "mm_only"
    BLOCK_WISE = "block_wise"
    LINEAR_SELFATTN = "linear_selfattn"
    PREDICT2_2B_720 = "predict2_2b_720"
    PREDICT2_14B_720 = "predict2_14b_720"

    def __str__(self) -> str:
        return self.value


@dataclass
class SACConfig(_SACConfig):
    def get_context_fn(self):
        if self.mode == CheckpointMode.LINEAR_SELFATTN:
            return linear_selfattn_context_fn
        elif self.mode == CheckpointMode.PREDICT2_2B_720:
            return predict2_2B_720_context_fn
        elif self.mode == CheckpointMode.PREDICT2_14B_720:
            return predict2_14B_720_context_fn
        else:
            # Reuse parent class implementation for other modes
            return super().get_context_fn()


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------- Feed Forward Network -----------------------
class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)

        x = self.activation(x)
        x = self.layer2(x)
        return x


def torch_attention_op(q_B_S_H_D: torch.Tensor, k_B_S_H_D: torch.Tensor, v_B_S_H_D: torch.Tensor) -> torch.Tensor:
    """Computes multi-head attention using PyTorch's native implementation.

    This function provides a PyTorch backend alternative to Transformer Engine's attention operation.
    It rearranges the input tensors to match PyTorch's expected format, computes scaled dot-product
    attention, and rearranges the output back to the original format.

    The input tensor names use the following dimension conventions:

    - B: batch size
    - S: sequence length
    - H: number of attention heads
    - D: head dimension

    Args:
        q_B_S_H_D: Query tensor with shape (batch, seq_len, n_heads, head_dim)
        k_B_S_H_D: Key tensor with shape (batch, seq_len, n_heads, head_dim)
        v_B_S_H_D: Value tensor with shape (batch, seq_len, n_heads, head_dim)

    Returns:
        Attention output tensor with shape (batch, seq_len, n_heads * head_dim)
    """
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    result_B_S_HD = rearrange(
        torch.nn.functional.scaled_dot_product_attention(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D), "b h ... l -> b ... (h l)"
    )

    return result_B_S_HD


class Attention(nn.Module):
    """
    A flexible attention module supporting both self-attention and cross-attention mechanisms.

    This module implements a multi-head attention layer that can operate in either self-attention
    or cross-attention mode. The mode is determined by whether a context dimension is provided.
    The implementation uses scaled dot-product attention and supports optional bias terms and
    dropout regularization.

    Args:
        query_dim (int): The dimensionality of the query vectors.
        context_dim (int, optional): The dimensionality of the context (key/value) vectors.
            If None, the module operates in self-attention mode using query_dim. Default: None
        n_heads (int, optional): Number of attention heads for multi-head attention. Default: 8
        head_dim (int, optional): The dimension of each attention head. Default: 64
        dropout (float, optional): Dropout probability applied to the output. Default: 0.0
        qkv_format (str, optional): Format specification for QKV tensors. Default: "bshd"
        backend (str, optional): Backend to use for the attention operation. Default: "transformer_engine"

    Examples:
        >>> # Self-attention with 512 dimensions and 8 heads
        >>> self_attn = Attention(query_dim=512)
        >>> x = torch.randn(32, 16, 512)  # (batch_size, seq_len, dim)
        >>> out = self_attn(x)  # (32, 16, 512)

        >>> # Cross-attention
        >>> cross_attn = Attention(query_dim=512, context_dim=256)
        >>> query = torch.randn(32, 16, 512)
        >>> context = torch.randn(32, 8, 256)
        >>> out = cross_attn(query, context)  # (32, 16, 512)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
        backend: str = "transformer_engine",
    ) -> None:
        super().__init__()
        log.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{n_heads} heads with a dimension of {head_dim}."
        )
        self.is_selfattn = context_dim is None  # self attention

        assert backend in ["transformer_engine", "torch", "minimal_a2a"], f"Invalid backend: {backend}"
        self.backend = backend

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        if self.backend == "transformer_engine":
            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
            )
        elif self.backend == "minimal_a2a":
            self.attn_op = MinimalA2AAttnOp()
        elif self.backend == "torch":
            self.attn_op = torch_attention_op

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope_emb: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            return q, k, v

        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

        return q, k, v

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        result = self.attn_op(q, k, v)  # [B, S, H, D]
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
        """
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v)

    def set_context_parallel_group(
        self, process_group: ProcessGroup, ranks: List[int], stream: torch.cuda.Stream
    ) -> None:
        self.attn_op.set_context_parallel_group(process_group, ranks, stream)


class VideoPositionEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cp_group = None

    def enable_context_parallel(self, process_group: ProcessGroup) -> None:
        self._cp_group = process_group

    def disable_context_parallel(self) -> None:
        self._cp_group = None

    @property
    def seq_dim(self) -> int:
        return 1

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor]) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split.
        It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        if self._cp_group is not None:
            cp_ranks = get_process_group_ranks(self._cp_group)
            cp_size = len(cp_ranks)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = torch.Size((B, T * cp_size, H, W, C))
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)

        return self._split_for_context_parallel(embeddings)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor]) -> Any:
        raise NotImplementedError

    def _split_for_context_parallel(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self._cp_group is not None:
            embeddings = split_inputs_cp(x=embeddings, seq_dim=self.seq_dim, cp_group=self._cp_group)
        return embeddings


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            Not specified in the original code snippet.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor  # type: ignore
        w_theta = 10000.0 * w_ntk_factor  # type: ignore
        t_theta = 10000.0 * t_ntk_factor  # type: ignore

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert (
                uniform_fps or B == 1 or T == 1
            ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"

            # apply sequence scaling in temporal dimension
            if fps is None:  # image case
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self) -> int:
        return 0


class LearnablePosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,
    ):
        """
        Args:
            interpolation (str): we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"
        self.model_channels = model_channels

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.model_channels)
        torch.nn.init.trunc_normal_(self.pos_emb_h, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_w, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_t, std=std, a=-3 * std, b=3 * std)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, _ = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = (
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
                + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
                + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
        else:
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(1e-6, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return emb / norm.to(emb.dtype)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        in_dype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        log.debug(
            f"Using AdaLN LoRA Flag:  {use_adaln_lora}. We enable bias if no AdaLN LoRA for backward compatibility."
        )
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb

        return emb_B_T_D, adaln_lora_B_T_3D


class FourierFeatures(nn.Module):
    """
    Implements a layer that generates Fourier features from input tensors, based on randomly sampled
    frequencies and phases. This can help in learning high-frequency functions in low-dimensional problems.

    [B] -> [B, D]

    Parameters:
        num_channels (int): The number of Fourier features to generate.
        bandwidth (float, optional): The scaling factor for the frequency of the Fourier features. Defaults to 1.
        normalize (bool, optional): If set to True, the outputs are scaled by sqrt(2), usually to normalize
                                    the variance of the features. Defaults to False.

    Example:
        >>> layer = FourierFeatures(num_channels=256, bandwidth=0.5, normalize=True)
        >>> x = torch.randn(10, 256)  # Example input tensor
        >>> output = layer(x)
        >>> print(output.shape)  # Expected shape: (10, 256)
    """

    def __init__(self, num_channels: int, bandwidth: int = 1, normalize: bool = False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1
        self.bandwidth = bandwidth
        self.num_channels = num_channels

        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator()
        generator.manual_seed(0)
        self.freqs = (
            2 * np.pi * self.bandwidth * torch.randn(self.num_channels, generator=generator).to(self.freqs.device)
        )
        self.phases = 2 * np.pi * torch.rand(self.num_channels, generator=generator).to(self.freqs.device)

    def forward(self, x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """
        Apply the Fourier feature transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            gain (float, optional): An additional gain factor applied during the forward pass. Defaults to 1.

        Returns:
            torch.Tensor: The transformed tensor, with Fourier features applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x


class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert (
            H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        ), f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift_B_T_1_1_D, scale_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d"), rearrange(
            scale_B_T_D, "b t d -> b t 1 1 d"
        )

        def _fn(
            _x_B_T_H_W_D: torch.Tensor,
            _norm_layer: nn.Module,
            _scale_B_T_1_1_D: torch.Tensor,
            _shift_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        x_B_T_H_W_D = _fn(x_B_T_H_W_D, self.layer_norm, scale_B_T_1_1_D, shift_B_T_1_1_D)
        x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


class Block(nn.Module):
    """
    A transformer block that combines self-attention, cross-attention and MLP layers with AdaLN modulation.
    Each component (self-attention, cross-attention, MLP) has its own layer normalization and AdaLN modulation.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Multiplier for MLP hidden dimension. Default: 4.0
        use_adaln_lora (bool): Whether to use AdaLN-LoRA modulation. Default: False
        adaln_lora_dim (int): Hidden dimension for AdaLN-LoRA layers. Default: 256

    The block applies the following sequence:
    1. Self-attention with AdaLN modulation
    2. Cross-attention with AdaLN modulation
    3. MLP with AdaLN modulation

    Each component uses skip connections and layer normalization.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(x_dim, None, num_heads, x_dim // num_heads, qkv_format="bshd", backend=backend)

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd", backend=backend
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )
        result_B_T_H_W_D = rearrange(
            self.self_attn(
                # normalized_x_B_T_HW_D,
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        def _x_fn(
            _x_B_T_H_W_D: torch.Tensor,
            layer_norm_cross_attn: Callable,
            _scale_cross_attn_B_T_1_1_D: torch.Tensor,
            _shift_cross_attn_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D, layer_norm_cross_attn, _scale_cross_attn_B_T_1_1_D, _shift_cross_attn_B_T_1_1_D
            )
            _result_B_T_H_W_D = rearrange(
                self.cross_attn(
                    rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                ),
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )
            return _result_B_T_H_W_D

        result_B_T_H_W_D = _x_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        return x_B_T_H_W_D


class MiniTrainDIT(WeightTrainingStat):
    """
    A clean impl of DIT that can load and  reproduce the training results of the original DIT model in~(cosmos 1)
    A general implementation of adaln-modulated VIT-like~(DiT) transformer for video processing.

    Args:
        max_img_h (int): Maximum height of the input images.
        max_img_w (int): Maximum width of the input images.
        max_frames (int): Maximum number of frames in the video sequence.
        in_channels (int): Number of input channels (e.g., RGB channels for color images).
        out_channels (int): Number of output channels.
        patch_spatial (tuple): Spatial resolution of patches for input processing.
        patch_temporal (int): Temporal resolution of patches for input processing.
        concat_padding_mask (bool): If True, includes a mask channel in the input to handle padding.
        model_channels (int): Base number of channels used throughout the model.
        num_blocks (int): Number of transformer blocks.
        num_heads (int): Number of heads in the multi-head attention layers.
        mlp_ratio (float): Expansion ratio for MLP blocks.
        crossattn_emb_channels (int): Number of embedding channels for cross-attention.
        pos_emb_cls (str): Type of positional embeddings.
        pos_emb_learnable (bool): Whether positional embeddings are learnable.
        pos_emb_interpolation (str): Method for interpolating positional embeddings.
        min_fps (int): Minimum frames per second.
        max_fps (int): Maximum frames per second.
        use_adaln_lora (bool): Whether to use AdaLN-LoRA.
        adaln_lora_dim (int): Dimension for AdaLN-LoRA.
        rope_h_extrapolation_ratio (float): Height extrapolation ratio for RoPE.
        rope_w_extrapolation_ratio (float): Width extrapolation ratio for RoPE.
        rope_t_extrapolation_ratio (float): Temporal extrapolation ratio for RoPE.
        extra_per_block_abs_pos_emb (bool): Whether to use extra per-block absolute positional embeddings.
        extra_h_extrapolation_ratio (float): Height extrapolation ratio for extra embeddings.
        extra_w_extrapolation_ratio (float): Width extrapolation ratio for extra embeddings.
        extra_t_extrapolation_ratio (float): Temporal extrapolation ratio for extra embeddings.
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,  # tuple,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        atten_backend: str = "transformer_engine",
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        # positional embedding settings
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        sac_config: SACConfig = SACConfig(),
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.atten_backend = atten_backend
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.cuda_graphs = {}

        self.build_patch_embed()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=atten_backend,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

        self.t_embedding_norm = te.pytorch.RMSNorm(model_channels, eps=1e-6)
        self.init_weights()
        self.enable_selective_checkpoint(sac_config)
        self._is_context_parallel_enabled = False

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

    def build_patch_embed(self) -> None:
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )

    def build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        log.debug(f"Building positional embedding with {self.pos_emb_cls} class, impl {cls_type}")
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(
            **kwargs,  # type: ignore
        )

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(
                **kwargs,  # type: ignore
            )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the
            `self.pos_embedder` with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    class TensorRTBlock(torch.nn.Module):
        def __init__(self, engine_path: str):
            super().__init__()

            from cosmos_predict2.utils.ext import trt_inference
            import tensorrt as trt

            # Load engine
            with open(engine_path, "rb") as f:
                engine_serialized = f.read()
            self.trt_stream = trt_inference._trt_stream
            self.pyt_stream = trt_inference._pyt_stream
            self.engine = trt_inference._trt_runtime.deserialize_cuda_engine(engine_serialized)
            self.context = trt_inference.create_execution_context_from_pool(self.engine)
            self.context_set_input = lambda name, tensor: trt_inference.trt_set_tensor_check(self.context, name, tensor, check_shape=True)
            self.context_set_output = lambda tensor: trt_inference.trt_set_tensor_check(self.context, 'output', tensor, check_shape=False)
            log.info(f"TRT engine loaded: {engine_path}")

            # Check attention backend
            engine_inspector = self.engine.create_engine_inspector()
            layer_names = [ engine_inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE) for i in range(self.engine.num_layers) ]
            self.has_bert_attn = any([ "BertAttention" in name for name in layer_names ])
            self.has_gpt_attn = any([ "GPTAttention" in name for name in layer_names ])
            self.has_ring_sage_attn = any([ "RingSageAttentionFusedQKV" in name for name in layer_names ])
            self.has_natten = any([ "NeighborhoodAttention" in name for name in layer_names ])
            assert self.has_bert_attn or self.has_gpt_attn or self.has_ring_sage_attn or self.has_natten, \
                "Engine should contain one of (BertAttention, GPTAttention, RingSageAttentionFusedQKV, NeighborhoodAttention) as its self-attention backend"

            # Prepare supplementary input tensors
            self.register_buffer("host_max_attention_window", torch.tensor([0], dtype=torch.int32)) # (1, )
            self.register_buffer("host_sink_token_length", torch.tensor([0], dtype=torch.int32)) # (1, )
            self.register_buffer("host_request_types", torch.tensor([0], dtype=torch.int32)) # (B, ): 0 for context phase
            self.register_buffer("host_context_length", torch.tensor([-1], dtype=torch.int32)) # (B, ): Cached value for the observed T*H*W
            self.register_buffer("host_runtime_perf_knobs", torch.tensor([0], dtype=torch.int64)) # (Optional at C++ level): Empty
            self.register_buffer("host_context_progress", torch.tensor([0], dtype=torch.int64)) # (Optional at C++ level): Empty
            self.register_buffer("host_cp_size", torch.tensor([1], dtype=torch.int32)) # (1, )
            self.register_buffer("host_cp_rank", torch.tensor([0], dtype=torch.int32)) # (1, )
            self.register_buffer("host_cp_group", torch.tensor([0] * 72, dtype=torch.int32)) # (72, )

            # Compatibility hack
            self.self_attn = MiniTrainDIT.TensorRTBlock.Compat(self)

        class Compat:
            def __init__(self, hook):
                self.hook = hook

            def set_context_parallel_group(self, *args, **kwargs):
                return self.hook.set_context_parallel_group(*args, **kwargs)

        def set_context_parallel_group(
            self, process_group: Optional[ProcessGroup], ranks: List[int], stream: torch.cuda.Stream
        ):
            if process_group is not None:
                from cosmos_predict2.utils.ext import cp_comms
                cp_comms.set_cp_procgrp(ranks, process_group)

                self.host_cp_rank[0] = process_group.rank()
                self.host_cp_size[0] = len(ranks)
                for i, rank in enumerate(ranks):
                    self.host_cp_group[i] = rank
            else:
                self.host_cp_rank[0] = 0
                self.host_cp_size[0] = 1

        def forward(
            self,
            x_B_T_H_W_D: torch.Tensor,
            t_embedding_B_T_D: torch.Tensor,
            crossattn_emb: torch.Tensor,
            **kwargs: Dict[str, Optional[torch.Tensor]],
        ) -> torch.Tensor:

            B, T, H, W, _ = x_B_T_H_W_D.shape
            # Check the cached context length. Recompute if necessary.
            if B != self.host_context_length.shape[0] or T*H*W != self.host_context_length[0]:
                self.host_request_types = torch.tensor([0] * B, dtype=torch.int32)
                self.host_context_length = torch.tensor([T*H*W] * B, dtype=torch.int32)
                self.context_lengths = self.host_context_length.to(device=x_B_T_H_W_D.device)

            # Treat rope_emb_L_1_1_D as requried
            rope_emb_L_1_1_D = kwargs.pop('rope_emb_L_1_1_D')
            rope_emb_T_H_W_1_1_D = rope_emb_L_1_1_D.unflatten(0, (T, H, W))
            out_B_T_H_W_D = torch.empty_like(x_B_T_H_W_D)

            self.context_set_input('x_B_T_H_W_D', x_B_T_H_W_D)
            self.context_set_input('emb_B_T_D', t_embedding_B_T_D)
            self.context_set_input('crossattn_emb', crossattn_emb)
            self.context_set_input('rope_emb_T_H_W_1_1_D', rope_emb_T_H_W_1_1_D)
            self.context_set_input('context_lengths', self.context_lengths)
            for name, tensor in kwargs.items():
                if tensor is not None:
                    self.context_set_input(name, tensor)
            self.context_set_output(out_B_T_H_W_D)
            if self.has_gpt_attn:
                self.context.set_tensor_address('host_max_attention_window', self.host_max_attention_window.data_ptr())
                self.context.set_tensor_address('host_sink_token_length', self.host_sink_token_length.data_ptr())
                self.context.set_tensor_address('host_request_types', self.host_request_types.data_ptr())
                self.context.set_tensor_address('host_context_length', self.host_context_length.data_ptr())
                self.context.set_tensor_address('host_runtime_perf_knobs', self.host_runtime_perf_knobs.data_ptr())
                self.context.set_tensor_address('host_context_progress', self.host_context_progress.data_ptr())
            if self.has_natten:
                host_video_size_3 = torch.tensor([T, H, W], dtype=torch.int32)
                self.context.set_tensor_address("host_video_size", host_video_size_3.data_ptr())
            if self.has_natten or self.has_ring_sage_attn:
                self.context.set_tensor_address('host_cp_size', self.host_cp_size.data_ptr())
                self.context.set_tensor_address('host_cp_rank', self.host_cp_rank.data_ptr())
                self.context.set_tensor_address('host_cp_group', self.host_cp_group.data_ptr())

            self.trt_stream.wait_stream(self.pyt_stream)
            self.context.execute_async_v3(self.trt_stream.cuda_stream)
            self.pyt_stream.wait_stream(self.trt_stream)

            return out_B_T_H_W_D

    def load_trt(self, prefix):
        self.trt_prefix = prefix
        for iblock in range(len(self.blocks)):
            trt_engine_file = f"{prefix}{iblock}.engine"
            self.blocks[iblock] = MiniTrainDIT.TensorRTBlock(trt_engine_file)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        use_cuda_graphs: bool = False,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
        """
        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        assert not (self.training and use_cuda_graphs), "CUDA Graphs are supported only for inference"
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert (
                x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
            ), f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"

        if use_cuda_graphs:
            shapes_key = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        for block in blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp

    def enable_selective_checkpoint(self, sac_config: SACConfig):
        if sac_config.mode == CheckpointMode.NONE:
            pass
        else:
            log.debug(
                f"Enable selective checkpoint with {sac_config.mode}, for every {sac_config.every_n_blocks} blocks. Total blocks: {len(self.blocks)}"
            )
            _context_fn = sac_config.get_context_fn()
            for block_id, block in self.blocks.named_children():
                if int(block_id) % sac_config.every_n_blocks == 0:
                    log.debug(f"Enable selective checkpoint for block {block_id}")
                    block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=_context_fn,
                        preserve_rng_state=False,
                    )
                    self.blocks.register_module(block_id, block)
            self.register_module(
                "final_layer",
                ptd_checkpoint_wrapper(
                    self.final_layer,
                    context_fn=_context_fn,
                    preserve_rng_state=False,
                ),
            )

        return self

    def fully_shard(self, mesh: DeviceMesh) -> None:
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)

        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True)
        if self.extra_per_block_abs_pos_emb:
            fully_shard(self.extra_pos_embedder, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=True)

    def disable_context_parallel(self) -> None:
        # pos_embedder
        self.pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()

        # attention
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None) -> None:
        # pos_embedder
        self.pos_embedder.enable_context_parallel(process_group=process_group)
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.enable_context_parallel(process_group=process_group)

        # attention
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = True

    @property
    def is_context_parallel_enabled(self) -> bool:
        return self._is_context_parallel_enabled
