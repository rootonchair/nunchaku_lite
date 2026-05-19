"""Flux2 adapter for patching Diffusers Flux2 transformers with Nunchaku Lite modules."""

import math
import types
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2Modulation,
    Flux2ParallelSelfAttention,
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)

from ..core import PatchOptions, register_adapter
from ..ops.attention import attention_fp16_cuda
from ..ops.fused import fused_qkv_norm_rotary
from .common import (
    SVDQPatchContext,
    alloc_packed_qkv as _alloc_packed_qkv,
    apply_gated_residual as _apply_gated_residual,
    build_svdq_context,
    finalize_svdq_checkpoint,
    fuse_linears,
    pack_rotemb,
    pad_tensor,
    patch_modules_recursively,
    prepare_transformer_dtype,
    svdq_from_linear,
)


def _pack_flux2_rotary_emb(freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Pack Diffusers Flux2 rotary embeddings for native kernels.

    Args:
        freqs_cis: ``(cos, sin)`` tuple produced by Diffusers Flux2 positional
            embeddings. Both tensors must have shape ``(sequence, dim)``.

    Returns:
        Float32 packed rotary tensor accepted by the native SVDQ GEMM kernels.

    Raises:
        ValueError: If the tuple tensors do not have matching 2D shapes.
    """

    cos, sin = freqs_cis
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("Expected Flux.2 rotary embeddings as a (cos, sin) tuple with shape (seq_len, dim).")

    rotemb = torch.stack([sin[:, 0::2], cos[:, 0::2]], dim=-1).unsqueeze(0).unsqueeze(-2).contiguous()
    return pack_rotemb(pad_tensor(rotemb, 256, 1))


def _flux2_kv_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_txt_tokens: int,
    num_ref_tokens: int,
    kv_cache=None,
    backend=None,
) -> torch.Tensor:
    """Apply Flux2 reference-token causal attention with optional KV-cache reuse.

    Args:
        query: Query tensor for the full sequence.
        key: Key tensor for the full sequence.
        value: Value tensor for the full sequence.
        num_txt_tokens: Number of leading text tokens.
        num_ref_tokens: Number of reference tokens after the text tokens.
        kv_cache: Optional cache object exposing ``get`` and ``store``.
        backend: Optional Diffusers attention backend override.

    Returns:
        Attention output with the same sequence layout as ``query``.
    """

    if num_ref_tokens == 0 and kv_cache is None:
        return dispatch_attention_fn(query, key, value, backend=backend)

    if kv_cache is not None:
        k_ref, v_ref = kv_cache.get()
        k_all = torch.cat([key[:, :num_txt_tokens], k_ref, key[:, num_txt_tokens:]], dim=1)
        v_all = torch.cat([value[:, :num_txt_tokens], v_ref, value[:, num_txt_tokens:]], dim=1)
        return dispatch_attention_fn(query, k_all, v_all, backend=backend)

    ref_start = num_txt_tokens
    ref_end = num_txt_tokens + num_ref_tokens
    q_txt = query[:, :ref_start]
    q_ref = query[:, ref_start:ref_end]
    q_img = query[:, ref_end:]
    k_txt = key[:, :ref_start]
    k_ref = key[:, ref_start:ref_end]
    k_img = key[:, ref_end:]
    v_txt = value[:, :ref_start]
    v_ref = value[:, ref_start:ref_end]
    v_img = value[:, ref_end:]

    q_txt_img = torch.cat([q_txt, q_img], dim=1)
    k_all = torch.cat([k_txt, k_ref, k_img], dim=1)
    v_all = torch.cat([v_txt, v_ref, v_img], dim=1)
    attn_txt_img = dispatch_attention_fn(query=q_txt_img, key=k_all, value=v_all, backend=backend)
    attn_txt = attn_txt_img[:, :ref_start]
    attn_img = attn_txt_img[:, ref_start:]
    attn_ref = dispatch_attention_fn(query=q_ref, key=k_ref, value=v_ref, backend=backend)
    return torch.cat([attn_txt, attn_ref, attn_img], dim=1)


class NunchakuFlux2Attention(nn.Module):
    """Lite replacement for Flux2 cross-stream attention with SVDQ projections."""

    def __init__(self, other: Flux2Attention, context: SVDQPatchContext | None = None, **kwargs):
        """Copy attention metadata and replace QKV/output projections.

        Args:
            other: Source Diffusers Flux2 attention module.
            context: Shared SVDQ patch settings.
            **kwargs: Additional SVDQ constructor overrides.

        Returns:
            None.
        """

        super().__init__()
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.out_dim = other.out_dim
        self.heads = other.heads
        self.use_bias = other.use_bias
        self.dropout = other.dropout
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.added_proj_bias = other.added_proj_bias
        processor = getattr(other, "processor", None)
        self._attention_backend = getattr(processor, "_attention_backend", None)
        self._parallel_config = getattr(processor, "_parallel_config", None)

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.to_out = other.to_out
        self.to_out[0] = svdq_from_linear(self.to_out[0], context, **kwargs)
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = svdq_from_linear(to_qkv, context, **kwargs)

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = other.norm_added_q
            self.norm_added_k = other.norm_added_k
            self.to_add_out = svdq_from_linear(other.to_add_out, context, **kwargs)
            with torch.device("meta"):
                to_added_qkv = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
            self.to_added_qkv = svdq_from_linear(to_added_qkv, context, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run Flux2 attention, choosing packed native attention when possible.

        Args:
            hidden_states: Image hidden states.
            encoder_hidden_states: Optional text/context hidden states.
            attention_mask: Optional attention mask.
            image_rotary_emb: Either Diffusers rotary tuple/tensor or packed
                rotary tensors, depending on the forward path.
            **kwargs: Optional KV-cache and reference-token controls.

        Returns:
            Hidden-state output for single-stream attention, or
            ``(hidden_states, encoder_hidden_states)`` for double-stream
            attention.
        """

        kv_cache = kwargs.get("kv_cache", None)
        kv_cache_mode = kwargs.get("kv_cache_mode", None)
        num_ref_tokens = int(kwargs.get("num_ref_tokens", 0))
        use_packed_fp16 = (
            kv_cache_mode is None
            and encoder_hidden_states is not None
            and isinstance(image_rotary_emb, tuple)
            and len(image_rotary_emb) == 2
            and torch.is_tensor(image_rotary_emb[0])
            and image_rotary_emb[0].ndim == 3
            and hidden_states.is_cuda
        )
        if use_packed_fp16:
            return self._forward_packed(hidden_states, encoder_hidden_states, image_rotary_emb)

        query, key, value, encoder_seq_len = self._project_qkv(hidden_states, encoder_hidden_states, image_rotary_emb)

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = encoder_seq_len
            ref_end = encoder_seq_len + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, encoder_seq_len, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, encoder_seq_len, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_seq_len:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_seq_len, hidden_states.shape[1] - encoder_seq_len], dim=1
            )
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        if encoder_seq_len:
            return hidden_states, encoder_hidden_states
        return hidden_states

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run padded native fp16 attention for separate image and text streams.

        Args:
            hidden_states: Image hidden states.
            encoder_hidden_states: Text/context hidden states.
            image_rotary_emb: Packed ``(image_rope, text_rope)`` tuple.

        Returns:
            Tuple ``(hidden_states, encoder_hidden_states)`` after attention and
            output projections.
        """

        batch_size = hidden_states.shape[0]
        num_txt_tokens = encoder_hidden_states.shape[1]
        num_img_tokens = hidden_states.shape[1]
        num_txt_tokens_pad = math.ceil(num_txt_tokens / 256) * 256
        num_img_tokens_pad = math.ceil(num_img_tokens / 256) * 256
        num_tokens_pad = num_txt_tokens_pad + num_img_tokens_pad
        query = torch.empty(
            batch_size, self.heads, num_tokens_pad, self.head_dim, dtype=torch.float16, device=hidden_states.device
        )
        key = torch.empty_like(query)
        value = torch.empty_like(query)
        fused_qkv_norm_rotary(
            hidden_states,
            self.to_qkv,
            self.norm_q,
            self.norm_k,
            image_rotary_emb[0],
            output=(query[:, :, num_txt_tokens_pad:], key[:, :, num_txt_tokens_pad:], value[:, :, num_txt_tokens_pad:]),
            attn_tokens=num_img_tokens,
        )
        fused_qkv_norm_rotary(
            encoder_hidden_states,
            self.to_added_qkv,
            self.norm_added_q,
            self.norm_added_k,
            image_rotary_emb[1],
            output=(query[:, :, :num_txt_tokens_pad], key[:, :, :num_txt_tokens_pad], value[:, :, :num_txt_tokens_pad]),
            attn_tokens=num_txt_tokens,
        )
        attention_output = torch.empty(
            batch_size,
            num_tokens_pad,
            self.heads * self.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        attention_fp16_cuda(query, key, value, attention_output, self.head_dim ** (-0.5))
        encoder_hidden_states = attention_output[:, :num_txt_tokens]
        hidden_states = attention_output[:, num_txt_tokens_pad : num_txt_tokens_pad + num_img_tokens]
        encoder_hidden_states = self.to_add_out(encoder_hidden_states)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states, encoder_hidden_states

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Project image and optional text states to Q/K/V tensors.

        Args:
            hidden_states: Image hidden states.
            encoder_hidden_states: Optional text/context states.
            image_rotary_emb: Rotary data used by either packed native or
                Diffusers fallback paths.

        Returns:
            Tuple ``(query, key, value, encoder_seq_len)``.
        """

        batch_size = hidden_states.shape[0]
        packed_rotary = (
            isinstance(image_rotary_emb, tuple)
            and len(image_rotary_emb) == 2
            and torch.is_tensor(image_rotary_emb[0])
            and image_rotary_emb[0].ndim == 3
        )
        if encoder_hidden_states is not None and packed_rotary:
            qkv = fused_qkv_norm_rotary(hidden_states, self.to_qkv, self.norm_q, self.norm_k, image_rotary_emb[0])
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.view(batch_size, -1, self.heads, self.head_dim)
            key = key.view(batch_size, -1, self.heads, self.head_dim)
            value = value.view(batch_size, -1, self.heads, self.head_dim)

            encoder_qkv = fused_qkv_norm_rotary(
                encoder_hidden_states,
                self.to_added_qkv,
                self.norm_added_q,
                self.norm_added_k,
                image_rotary_emb[1],
            )
            encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)
            encoder_query = encoder_query.view(batch_size, -1, self.heads, self.head_dim)
            encoder_key = encoder_key.view(batch_size, -1, self.heads, self.head_dim)
            encoder_value = encoder_value.view(batch_size, -1, self.heads, self.head_dim)
            encoder_seq_len = encoder_hidden_states.shape[1]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
            return query, key, value, encoder_seq_len

        query, key, value = self.to_qkv(hidden_states).chunk(3, dim=-1)
        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))
        query = self.norm_q(query)
        key = self.norm_k(key)
        encoder_seq_len = 0

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_query, encoder_key, encoder_value = self.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
            encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.heads, -1))
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)
            encoder_seq_len = encoder_hidden_states.shape[1]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        return query, key, value, encoder_seq_len


class NunchakuFlux2FeedForward(nn.Module):
    """Quantized Flux2 feed-forward block."""

    def __init__(self, other: Flux2FeedForward, context: SVDQPatchContext | None = None, **kwargs):
        """Replace feed-forward projections with SVDQ modules.

        Args:
            other: Source Diffusers Flux2 feed-forward module.
            context: Shared SVDQ patch settings.
            **kwargs: Additional SVDQ constructor overrides.

        Returns:
            None.
        """

        super().__init__()
        self.linear_in = svdq_from_linear(other.linear_in, context, **kwargs)
        self.act_fn = other.act_fn
        self.linear_out = svdq_from_linear(other.linear_out, context, **kwargs)
        self.linear_out.act_unsigned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantized feed-forward block.

        Args:
            x: Input hidden states.

        Returns:
            Feed-forward output tensor.
        """

        x = self.linear_in(x)
        x = self.act_fn(x)
        return self.linear_out(x)


class NunchakuFlux2ParallelSelfAttention(nn.Module):
    """Lite replacement for Flux2 parallel self-attention plus MLP blocks."""

    def __init__(self, other: Flux2ParallelSelfAttention, context: SVDQPatchContext | None = None, **kwargs):
        """Split Flux2 fused parallel projections into lite SVDQ modules.

        Args:
            other: Source Diffusers parallel self-attention module.
            context: Shared SVDQ patch settings.
            **kwargs: Additional SVDQ constructor overrides.

        Returns:
            None.
        """

        super().__init__()
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.out_dim = other.out_dim
        self.heads = other.heads
        self.use_bias = other.use_bias
        self.dropout = other.dropout
        self.mlp_ratio = other.mlp_ratio
        self.mlp_hidden_dim = other.mlp_hidden_dim
        self.mlp_mult_factor = other.mlp_mult_factor
        processor = getattr(other, "processor", None)
        self._attention_backend = getattr(processor, "_attention_backend", None)
        self._parallel_config = getattr(processor, "_parallel_config", None)

        with torch.device("meta"):
            qkv_proj = nn.Linear(other.query_dim, other.inner_dim * 3, bias=other.use_bias)
            mlp_fc1 = nn.Linear(other.query_dim, other.mlp_hidden_dim * other.mlp_mult_factor, bias=other.use_bias)
            out_proj = nn.Linear(other.inner_dim, other.out_dim, bias=other.to_out.bias is not None)
            mlp_fc2 = nn.Linear(other.mlp_hidden_dim, other.out_dim, bias=other.to_out.bias is not None)
        device = other.to_qkv_mlp_proj.weight.device
        self.qkv_proj = svdq_from_linear(qkv_proj, context, device=device, **kwargs)
        self.mlp_fc1 = svdq_from_linear(mlp_fc1, context, device=device, **kwargs)
        self.mlp_act_fn = other.mlp_act_fn
        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.out_proj = svdq_from_linear(out_proj, context, device=device, **kwargs)
        self.mlp_fc2 = svdq_from_linear(mlp_fc2, context, device=device, **kwargs)
        self.mlp_fc2.act_unsigned = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run parallel self-attention and MLP.

        Args:
            hidden_states: Joint hidden states.
            attention_mask: Optional attention mask.
            image_rotary_emb: Rotary tensor for fallback paths or packed rotary
                tensor for native paths.
            **kwargs: Optional KV-cache, text-token, and reference-token
                controls.

        Returns:
            Sum of attention output projection and MLP projection.
        """

        kv_cache = kwargs.get("kv_cache", None)
        kv_cache_mode = kwargs.get("kv_cache_mode", None)
        num_txt_tokens = int(kwargs.get("num_txt_tokens", 0))
        num_ref_tokens = int(kwargs.get("num_ref_tokens", 0))
        use_packed_fp16 = (
            kv_cache_mode is None
            and torch.is_tensor(image_rotary_emb)
            and image_rotary_emb.ndim == 3
            and hidden_states.is_cuda
        )
        if use_packed_fp16:
            return self._forward_packed(hidden_states, image_rotary_emb)

        if torch.is_tensor(image_rotary_emb) and image_rotary_emb.ndim == 3:
            batch_size = hidden_states.shape[0]
            qkv = fused_qkv_norm_rotary(hidden_states, self.qkv_proj, self.norm_q, self.norm_k, image_rotary_emb)
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.view(batch_size, -1, self.heads, self.head_dim)
            key = key.view(batch_size, -1, self.heads, self.head_dim)
            value = value.view(batch_size, -1, self.heads, self.head_dim)
        else:
            qkv = self.qkv_proj(hidden_states)
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.unflatten(-1, (self.heads, -1))
            key = key.unflatten(-1, (self.heads, -1))
            value = value.unflatten(-1, (self.heads, -1))
            query = self.norm_q(query)
            key = self.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = num_txt_tokens
            ref_end = num_txt_tokens + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            attn_output = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        attn_output = attn_output.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = self.mlp_act_fn(self.mlp_fc1(hidden_states))
        return self.out_proj(attn_output) + self.mlp_fc2(mlp_hidden_states)

    def _forward_packed(self, hidden_states: torch.Tensor, image_rotary_emb: torch.Tensor) -> torch.Tensor:
        """Run padded native fp16 attention for a single packed sequence.

        Args:
            hidden_states: Joint hidden states.
            image_rotary_emb: Packed rotary tensor for the joint sequence.

        Returns:
            Combined attention and MLP output.
        """

        batch_size = hidden_states.shape[0]
        num_tokens = hidden_states.shape[1]
        query, key, value, num_tokens_pad = _alloc_packed_qkv(
            batch_size, self.heads, num_tokens, self.head_dim, hidden_states.device
        )
        fused_qkv_norm_rotary(
            hidden_states,
            self.qkv_proj,
            self.norm_q,
            self.norm_k,
            image_rotary_emb,
            output=(query, key, value),
            attn_tokens=num_tokens,
        )
        attn_output = torch.empty(
            batch_size,
            num_tokens_pad,
            self.heads * self.head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        attention_fp16_cuda(query, key, value, attn_output, self.head_dim ** (-0.5))
        attn_output = attn_output[:, :num_tokens]
        mlp_hidden_states = self.mlp_act_fn(self.mlp_fc1(hidden_states))
        return self.out_proj(attn_output) + self.mlp_fc2(mlp_hidden_states)


class NunchakuFlux2TransformerBlock(nn.Module):
    """Lite replacement for Flux2 double-stream transformer blocks."""

    def __init__(self, block: Flux2TransformerBlock, context: SVDQPatchContext | None = None, **kwargs):
        """Replace double-stream attention and feed-forward modules.

        Args:
            block: Source Diffusers Flux2 double-stream block.
            context: Shared SVDQ patch settings.
            **kwargs: Additional SVDQ constructor overrides.

        Returns:
            None.
        """

        super().__init__()
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm1 = block.norm1
        self.norm1_context = block.norm1_context
        self.attn = NunchakuFlux2Attention(block.attn, context=context, **kwargs)
        self.norm2 = block.norm2
        self.ff = NunchakuFlux2FeedForward(block.ff, context=context, **kwargs)
        self.norm2_context = block.norm2_context
        self.ff_context = NunchakuFlux2FeedForward(block.ff_context, context=context, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one Flux2 double-stream block over image and text states.

        Args:
            hidden_states: Image hidden states.
            encoder_hidden_states: Text/context hidden states.
            temb_mod_img: Image modulation tensor.
            temb_mod_txt: Text modulation tensor.
            image_rotary_emb: Packed ``(image_rope, text_rope)`` tuple.
            joint_attention_kwargs: Optional attention kwargs.

        Returns:
            Tuple ``(encoder_hidden_states, hidden_states)`` after the block.
        """

        joint_attention_kwargs = joint_attention_kwargs or {}
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(
            temb_mod_txt, 2
        )

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = _apply_gated_residual(hidden_states, gate_msa, attn_output)
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        hidden_states = _apply_gated_residual(hidden_states, gate_mlp, self.ff(norm_hidden_states))

        encoder_hidden_states = _apply_gated_residual(encoder_hidden_states, c_gate_msa, context_attn_output)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        encoder_hidden_states = _apply_gated_residual(
            encoder_hidden_states, c_gate_mlp, self.ff_context(norm_encoder_hidden_states)
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class NunchakuFlux2SingleTransformerBlock(nn.Module):
    """Lite replacement for Flux2 single-stream transformer blocks."""

    def __init__(self, block: Flux2SingleTransformerBlock, context: SVDQPatchContext | None = None, **kwargs):
        """Replace the single-stream parallel attention module.

        Args:
            block: Source Diffusers Flux2 single-stream block.
            context: Shared SVDQ patch settings.
            **kwargs: Additional SVDQ constructor overrides.

        Returns:
            None.
        """

        super().__init__()
        self.norm = block.norm
        self.attn = NunchakuFlux2ParallelSelfAttention(block.attn, context=context, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb_mod: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Run one Flux2 single-stream block and optionally split text/image states.

        Args:
            hidden_states: Joint or image hidden states.
            encoder_hidden_states: Optional text/context states to concatenate
                before the block.
            temb_mod: Single-stream modulation tensor.
            image_rotary_emb: Rotary tensor for the joint sequence.
            joint_attention_kwargs: Optional attention kwargs.
            split_hidden_states: Whether to split text and image streams before
                returning.
            text_seq_len: Text token count used when splitting a pre-concatenated
                sequence.

        Returns:
            Joint hidden states, or ``(encoder_hidden_states, hidden_states)``
            when ``split_hidden_states`` is true.
        """

        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        hidden_states = _apply_gated_residual(hidden_states, mod_gate, attn_output)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            return encoder_hidden_states, hidden_states
        return hidden_states


def lite_flux2_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
    kv_cache=None,
    kv_cache_mode: str | None = None,
    num_ref_tokens: int = 0,
    ref_fixed_timestep: float = 0.0,
) -> torch.Tensor | Transformer2DModelOutput:
    """Flux2 transformer forward wrapper that pre-packs RoPE for lite modules.

    Args:
        self: Patched Flux2 transformer instance.
        hidden_states: Input image latents.
        encoder_hidden_states: Text/context hidden states.
        timestep: Diffusion timestep tensor.
        img_ids: Image token position ids.
        txt_ids: Text token position ids.
        guidance: Optional guidance tensor.
        joint_attention_kwargs: Optional kwargs forwarded to attention blocks.
        return_dict: Whether to return ``Transformer2DModelOutput``.
        kv_cache: Optional KV cache object for fallback cached modes.
        kv_cache_mode: Optional cache mode. Non-``None`` delegates to the
            original Diffusers forward.
        num_ref_tokens: Number of reference tokens for cached modes.
        ref_fixed_timestep: Reference timestep value for cached modes.

    Returns:
        Diffusers transformer output object, or tuple when ``return_dict`` is
        false.
    """

    if kv_cache_mode is not None:
        return self._nunchaku_lite_flux2_original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=return_dict,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            num_ref_tokens=num_ref_tokens,
            ref_fixed_timestep=ref_fixed_timestep,
        )

    num_txt_tokens = encoder_hidden_states.shape[1]
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    temb = self.time_guidance_embed(timestep, guidance)
    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    rotary_emb_img = _pack_flux2_rotary_emb(image_rotary_emb)
    rotary_emb_txt = _pack_flux2_rotary_emb(text_rotary_emb)
    rotary_emb_single = _pack_flux2_rotary_emb(
        (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )
    )
    kv_attn_kwargs = joint_attention_kwargs

    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                (rotary_emb_img, rotary_emb_txt),
                kv_attn_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                joint_attention_kwargs=kv_attn_kwargs,
            )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    for block in self.single_transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                None,
                single_stream_mod,
                rotary_emb_single,
                kv_attn_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=rotary_emb_single,
                joint_attention_kwargs=kv_attn_kwargs,
            )

    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


class Flux2Adapter:
    """Adapter for Diffusers ``Flux2Transformer2DModel`` checkpoints."""

    target = "flux2"

    def matches(self, transformer: torch.nn.Module) -> bool:
        """Return whether ``transformer`` is a Diffusers Flux2 transformer.

        Args:
            transformer: Candidate module.

        Returns:
            ``True`` when the class name and module path match Diffusers Flux2.
        """

        return (
            transformer.__class__.__name__ == "Flux2Transformer2DModel"
            and "transformer_flux2" in transformer.__class__.__module__
        )

    def patch(
        self,
        transformer: torch.nn.Module,
        checkpoint_state: dict[str, torch.Tensor],
        quantization_config: dict[str, Any],
        options: PatchOptions,
    ) -> dict[str, torch.Tensor]:
        """Patch a Flux2 transformer in place and install the lite forward wrapper.

        Args:
            transformer: Diffusers Flux2 transformer to mutate.
            checkpoint_state: Checkpoint tensors to load after patching.
            quantization_config: Quantization metadata from the checkpoint.
            options: Normalized patch options.

        Returns:
            Checkpoint state dict to load into the patched transformer.
        """

        context = build_svdq_context(transformer, quantization_config, options)
        prepare_transformer_dtype(transformer, context)
        self._patch_transformer(transformer, context)

        transformer._nunchaku_lite_flux2_original_forward = transformer.forward
        transformer.forward = types.MethodType(lite_flux2_forward, transformer)
        if _flux2_state_dict_needs_conversion(checkpoint_state):
            checkpoint_state = convert_flux2_state_dict(checkpoint_state)
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        _drop_unused_zero_bias_tensors(transformer, checkpoint_state)
        from ..lora.core.runtime import bind_transformer_lora_methods
        from ..lora.flux2 import NunchakuFlux2TransformerLoraMixin

        bind_transformer_lora_methods(transformer, NunchakuFlux2TransformerLoraMixin)
        transformer._nunchaku_lite_flux2_patched = True
        return checkpoint_state

    def patch_pipeline(
        self,
        pipeline: Any,
        *,
        component_name: str = "transformer",
        component: torch.nn.Module | None = None,
    ) -> None:
        """Attach Flux2 pipeline-level runtime APIs."""

        from ..lora.core.runtime import NunchakuPipelineLoraMixin, bind_pipeline_lora_methods

        bind_pipeline_lora_methods(pipeline, NunchakuPipelineLoraMixin)

    def _patch_transformer(self, transformer: torch.nn.Module, context: SVDQPatchContext) -> None:
        """Patch Flux2 block modules through one recursive transformer traversal.

        Args:
            transformer: Flux2 transformer whose module tree should be patched.
            context: Shared SVDQ patch settings used by lite block
                replacements.

        Returns:
            None.
        """

        patch_modules_recursively(
            transformer,
            context,
            linear_filter=lambda _path, _linear: False,
            module_converters={
                Flux2TransformerBlock: lambda block: NunchakuFlux2TransformerBlock(block, context=context),
                Flux2SingleTransformerBlock: lambda block: NunchakuFlux2SingleTransformerBlock(
                    block, context=context
                ),
            },
        )


def convert_flux2_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize original Nunchaku Flux2 checkpoint keys to lite module names."""

    if not _flux2_state_dict_needs_conversion(state_dict):
        return state_dict

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("transformer_blocks.") and not any(
            marker in new_key for marker in (".attn.", ".ff.", ".ff_context.")
        ):
            replacements = (
                (".qkv_proj_context.", ".attn.to_added_qkv."),
                (".qkv_proj.", ".attn.to_qkv."),
                (".out_proj_context.", ".attn.to_add_out."),
                (".out_proj.", ".attn.to_out.0."),
                (".mlp_context_fc1.", ".ff_context.linear_in."),
                (".mlp_context_fc2.", ".ff_context.linear_out."),
                (".mlp_fc1.", ".ff.linear_in."),
                (".mlp_fc2.", ".ff.linear_out."),
                (".norm_added_q.", ".attn.norm_added_q."),
                (".norm_added_k.", ".attn.norm_added_k."),
                (".norm_q.", ".attn.norm_q."),
                (".norm_k.", ".attn.norm_k."),
            )
            for old, new in replacements:
                if old in new_key:
                    new_key = new_key.replace(old, new, 1)
                    break

        elif new_key.startswith("single_transformer_blocks.") and ".attn." not in new_key:
            replacements = (
                (".qkv_proj.", ".attn.qkv_proj."),
                (".mlp_fc1.", ".attn.mlp_fc1."),
                (".out_proj.", ".attn.out_proj."),
                (".mlp_fc2.", ".attn.mlp_fc2."),
                (".norm_q.", ".attn.norm_q."),
                (".norm_k.", ".attn.norm_k."),
            )
            for old, new in replacements:
                if old in new_key:
                    new_key = new_key.replace(old, new, 1)
                    break

        new_key = new_key.replace(".lora_down", ".proj_down")
        new_key = new_key.replace(".lora_up", ".proj_up")
        if ".smooth_orig" in new_key and ".smooth_factor_orig" not in new_key:
            new_key = new_key.replace(".smooth_orig", ".smooth_factor_orig")
        elif ".smooth" in new_key and ".smooth_factor" not in new_key:
            new_key = new_key.replace(".smooth", ".smooth_factor")

        converted[new_key] = value
    return converted


def _flux2_state_dict_needs_conversion(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(_is_uncorrected_flux2_key(key) for key in state_dict)


def _is_uncorrected_flux2_key(key: str) -> bool:
    double_block_key = key.startswith("transformer_blocks.")
    single_block_key = key.startswith("single_transformer_blocks.")
    if not double_block_key and not single_block_key:
        return False

    if key.endswith((".lora_down", ".lora_up", ".smooth", ".smooth_orig")):
        return True

    if double_block_key and not any(marker in key for marker in (".attn.", ".ff.", ".ff_context.")):
        return any(
            marker in key
            for marker in (
                ".qkv_proj_context.",
                ".qkv_proj.",
                ".out_proj_context.",
                ".out_proj.",
                ".mlp_context_fc1.",
                ".mlp_context_fc2.",
                ".mlp_fc1.",
                ".mlp_fc2.",
                ".norm_added_q.",
                ".norm_added_k.",
                ".norm_q.",
                ".norm_k.",
            )
        )

    if single_block_key and ".attn." not in key:
        return any(
            marker in key
            for marker in (
                ".qkv_proj.",
                ".mlp_fc1.",
                ".out_proj.",
                ".mlp_fc2.",
                ".norm_q.",
                ".norm_k.",
            )
        )

    return False


def _drop_unused_zero_bias_tensors(transformer: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> None:
    """Drop zero bias tensors emitted by original Flux2 checkpoints for biasless modules."""

    expected_keys = transformer.state_dict().keys()
    for key in list(checkpoint_state):
        if key in expected_keys or not key.endswith(".bias"):
            continue
        value = checkpoint_state[key]
        if torch.count_nonzero(value) == 0:
            checkpoint_state.pop(key)


register_adapter(Flux2Adapter())
