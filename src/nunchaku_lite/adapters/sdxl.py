"""SDXL UNet adapter."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
from diffusers.models.attention import AttentionModuleMixin, BasicTransformerBlock
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    UNetMidBlock2DCrossAttn,
)
from torch import nn
from torch.nn import functional as F

from nunchaku_lite.adapters.common import (
    SVDQPatchContext,
    build_svdq_context,
    finalize_svdq_checkpoint,
    fuse_linears,
    patch_modules_recursively,
    prepare_transformer_dtype,
    svdq_from_linear,
)
from nunchaku_lite.core import PatchOptions, register_adapter


def convert_sdxl_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert original nunchaku SDXL checkpoint keys to lite module names."""

    if not _sdxl_state_dict_needs_conversion(state_dict):
        return state_dict

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if ".transformer_blocks." in key:
            if ".lora_down" in key:
                new_key = key.replace(".lora_down", ".proj_down")
            elif ".lora_up" in key:
                new_key = key.replace(".lora_up", ".proj_up")
            elif ".smooth_orig" in key:
                new_key = key.replace(".smooth_orig", ".smooth_factor_orig")
            elif ".smooth" in key and ".smooth_factor" not in key:
                new_key = key.replace(".smooth", ".smooth_factor")
        converted[new_key] = value
    return converted


def _sdxl_state_dict_needs_conversion(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(
        ".transformer_blocks." in key and key.endswith((".lora_down", ".lora_up", ".smooth", ".smooth_orig"))
        for key in state_dict
    )


class NunchakuSDXLAttention(nn.Module, AttentionModuleMixin):
    """Attention module compatible with Diffusers SDXL BasicTransformerBlock."""

    def __init__(self, attention: nn.Module, context: SVDQPatchContext) -> None:
        super().__init__()
        self._copy_attention_attributes(attention)
        self.is_cross_attention = bool(getattr(attention, "is_cross_attention", False))

        if self.is_cross_attention:
            self.to_q = svdq_from_linear(attention.to_q, context)
            self.to_k = attention.to_k
            self.to_v = attention.to_v
        else:
            with torch.device("meta"):
                to_qkv = fuse_linears([attention.to_q, attention.to_k, attention.to_v])
            self.to_qkv = svdq_from_linear(to_qkv, context)

        self.to_out = attention.to_out
        self.to_out[0] = svdq_from_linear(self.to_out[0], context)
        self.processor = NunchakuSDXLAttnProcessor()

    def _copy_attention_attributes(self, attention: nn.Module) -> None:
        for name, value in attention.__dict__.items():
            if name.startswith("_") or name in {"training", "processor"}:
                continue
            if isinstance(value, (nn.Module, nn.Parameter)):
                continue
            setattr(self, name, value)

    def set_processor(self, processor: Any) -> None:
        if isinstance(processor, str):
            if processor != "flashattn2":
                raise ValueError(f"Unsupported SDXL attention processor: {processor!r}")
            processor = NunchakuSDXLAttnProcessor()
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **cross_attention_kwargs: Any,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )


class NunchakuSDXLAttnProcessor:
    """Scaled-dot-product attention processor for quantized SDXL attention projections."""

    def __call__(
        self,
        attn: NunchakuSDXLAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError("attention_mask is not supported")

        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, _, _ = hidden_states.shape
            height = width = None

        if attn.is_cross_attention:
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states must be provided for cross attention")
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


class SDXLAdapter:
    target = "sdxl"

    def matches(self, transformer: nn.Module) -> bool:
        return (
            transformer.__class__.__name__ == "UNet2DConditionModel"
            and "unet_2d_condition" in transformer.__class__.__module__
        )

    def patch(
        self,
        transformer: nn.Module,
        state_dict: MutableMapping[str, torch.Tensor],
        metadata: Mapping[str, Any],
        options: PatchOptions,
    ) -> MutableMapping[str, torch.Tensor]:
        if options.precision not in {"int4", "nvfp4"}:
            raise ValueError(f"Unsupported SDXL precision: {options.precision!r}")
        if options.precision == "nvfp4":
            raise ValueError("SDXL adapter currently supports only int4 checkpoints")

        context = build_svdq_context(transformer, dict(metadata), options)
        prepare_transformer_dtype(transformer, context)
        self._patch_unet(transformer, context)
        converted = convert_sdxl_state_dict(state_dict) if _sdxl_state_dict_needs_conversion(state_dict) else state_dict
        finalize_svdq_checkpoint(transformer, converted, context)
        transformer._nunchaku_lite_sdxl_patched = True
        return converted

    def _patch_unet(self, unet: nn.Module, context: SVDQPatchContext) -> None:
        for block in list(getattr(unet, "down_blocks", [])) + list(getattr(unet, "up_blocks", [])):
            if isinstance(block, (CrossAttnDownBlock2D, CrossAttnUpBlock2D)):
                self._patch_cross_attn_block(block, context)

        mid_block = getattr(unet, "mid_block", None)
        if isinstance(mid_block, UNetMidBlock2DCrossAttn):
            self._patch_cross_attn_block(mid_block, context)

    def _patch_cross_attn_block(self, block: nn.Module, context: SVDQPatchContext) -> None:
        for attention_container in getattr(block, "attentions", []):
            transformer_blocks = getattr(attention_container, "transformer_blocks", None)
            if transformer_blocks is None:
                raise TypeError(f"Unsupported SDXL attention container: {attention_container.__class__.__name__}")
            for transformer_block in transformer_blocks:
                self._patch_transformer_block(transformer_block, context)

    def _patch_transformer_block(self, block: nn.Module, context: SVDQPatchContext) -> None:
        if not isinstance(block, BasicTransformerBlock):
            raise TypeError(f"Unsupported SDXL transformer block: {block.__class__.__name__}")

        block.attn1 = NunchakuSDXLAttention(block.attn1, context)
        if getattr(block, "attn2", None) is not None:
            block.attn2 = NunchakuSDXLAttention(block.attn2, context)
        if getattr(block, "ff", None) is not None:
            patch_modules_recursively(
                block.ff,
                module_converters={nn.Linear: lambda linear: svdq_from_linear(linear, context)},
            )


register_adapter(SDXLAdapter())
