"""Runtime LoRA loading and composition for patched Qwen-Image transformers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
from .base import (
    NunchakuLoraMixin,
    NunchakuPipelineLoraMixin,
    bind_transformer_lora_methods,
    load_lora_state_dict,
)
from .conversion import (
    FusedProjectionSpec,
    convert_diffusers_lora_state_dict,
    is_nunchaku_lite_lora_state_dict,
    normalize_lite_lora_state_dict,
    strip_transformer_prefix,
)
from .peft import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    apply_network_alphas,
    extract_network_alphas,
    normalize_float_tensor,
)


QKV_PROJECTION_SPECS = (
    FusedProjectionSpec(target=".attn.to_qkv", branches=(".attn.to_q", ".attn.to_k", ".attn.to_v")),
    FusedProjectionSpec(
        target=".attn.add_qkv_proj",
        branches=(".attn.add_q_proj", ".attn.add_k_proj", ".attn.add_v_proj"),
    ),
)


class NunchakuQwenImageLoraMixin(NunchakuLoraMixin):
    """Mixin-style method provider for quantized Qwen-Image LoRA runtime."""

    def _convert_lora_to_lite(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return convert_qwen_image_lora_to_lite(path_or_state_dict, self)


class NunchakuQwenImagePipelineLoraMixin(NunchakuPipelineLoraMixin):
    """Mixin-style method provider for Diffusers-compatible Qwen-Image pipeline APIs."""

    def _bind_transformer_lora_methods(self, transformer: nn.Module) -> None:
        bind_transformer_lora_methods(transformer, NunchakuQwenImageLoraMixin)


def convert_qwen_image_lora_to_lite(
    state_dict_or_path: str | Path | dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert a Qwen-Image LoRA into packed lite SVDQ low-rank tensors."""

    state_dict = load_lora_state_dict(state_dict_or_path)
    if is_nunchaku_lite_lora_state_dict(state_dict):
        return normalize_lite_lora_state_dict(state_dict, transformer)
    return convert_diffusers_lora_state_dict(
        state_dict,
        transformer,
        projection_specs=QKV_PROJECTION_SPECS,
        normalize_state_dict=_to_peft_format,
        map_direct_pair=_map_direct_diffusers_pair,
        is_transformer_lora_key=_is_transformer_lora_key,
    )


def _to_peft_format(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors = {}
    for key, value in state_dict.items():
        new_key = strip_transformer_prefix(key)
        new_key = new_key.replace(".lora_down.weight", LORA_A_SUFFIX)
        new_key = new_key.replace(".lora_up.weight", LORA_B_SUFFIX)
        tensors[new_key] = normalize_float_tensor(value)
    return apply_network_alphas(tensors, extract_network_alphas(tensors))


def _map_direct_diffusers_pair(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear],
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    if base_name not in modules:
        return []
    return [(base_name, lora_a.contiguous(), lora_b.contiguous())]


def _is_transformer_lora_key(base: str) -> bool:
    return base.startswith("transformer_blocks.")
