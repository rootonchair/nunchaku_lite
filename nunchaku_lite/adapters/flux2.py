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
    prepare_transformer_dtype,
    svdq_from_linear,
)


def _pack_flux2_rotary_emb(freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
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


class LiteFlux2Attention(nn.Module):
    def __init__(self, other: Flux2Attention, context: SVDQPatchContext | None = None, **kwargs):
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


class LiteFlux2FeedForward(nn.Module):
    def __init__(self, other: Flux2FeedForward, context: SVDQPatchContext | None = None, **kwargs):
        super().__init__()
        self.linear_in = svdq_from_linear(other.linear_in, context, **kwargs)
        self.act_fn = other.act_fn
        self.linear_out = svdq_from_linear(other.linear_out, context, **kwargs)
        self.linear_out.act_unsigned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        return self.linear_out(x)


class LiteFlux2ParallelSelfAttention(nn.Module):
    def __init__(self, other: Flux2ParallelSelfAttention, context: SVDQPatchContext | None = None, **kwargs):
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
        kv_cache = kwargs.get("kv_cache", None)
        kv_cache_mode = kwargs.get("kv_cache_mode", None)
        num_txt_tokens = int(kwargs.get("num_txt_tokens", 0))
        num_ref_tokens = int(kwargs.get("num_ref_tokens", 0))
        use_packed_fp16 = (
            kv_cache_mode is None and torch.is_tensor(image_rotary_emb) and image_rotary_emb.ndim == 3 and hidden_states.is_cuda
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


class LiteFlux2TransformerBlock(nn.Module):
    def __init__(self, block: Flux2TransformerBlock, context: SVDQPatchContext | None = None, **kwargs):
        super().__init__()
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm1 = block.norm1
        self.norm1_context = block.norm1_context
        self.attn = LiteFlux2Attention(block.attn, context=context, **kwargs)
        self.norm2 = block.norm2
        self.ff = LiteFlux2FeedForward(block.ff, context=context, **kwargs)
        self.norm2_context = block.norm2_context
        self.ff_context = LiteFlux2FeedForward(block.ff_context, context=context, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class LiteFlux2SingleTransformerBlock(nn.Module):
    def __init__(self, block: Flux2SingleTransformerBlock, context: SVDQPatchContext | None = None, **kwargs):
        super().__init__()
        self.norm = block.norm
        self.attn = LiteFlux2ParallelSelfAttention(block.attn, context=context, **kwargs)

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
    target = "flux2"

    def matches(self, transformer: torch.nn.Module) -> bool:
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
        context = build_svdq_context(transformer, quantization_config, options)
        prepare_transformer_dtype(transformer, context)
        for index, block in enumerate(transformer.transformer_blocks):
            transformer.transformer_blocks[index] = LiteFlux2TransformerBlock(block, context=context)
        for index, block in enumerate(transformer.single_transformer_blocks):
            transformer.single_transformer_blocks[index] = LiteFlux2SingleTransformerBlock(block, context=context)

        transformer._nunchaku_lite_flux2_original_forward = transformer.forward
        transformer.forward = types.MethodType(lite_flux2_forward, transformer)
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        transformer._nunchaku_lite_flux2_patched = True
        return checkpoint_state


register_adapter(Flux2Adapter())
