"""Runtime LoRA loading and composition for patched Qwen-Image transformers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .core.runtime import (
    NunchakuLoraMixin,
    load_lora_state_dict,
)
from .core.convert import (
    LORA_ERROR_LABEL,
    QKV_PROJECTION_SPECS,
    fuse_projection_branches,
    group_fused_projection_pairs,
    is_nunchaku_lite_lora_state_dict,
    normalize_nunchaku_lora_state_dict,
    set_standard_converted_lora_pair,
    strip_transformer_prefix,
    validate_nunchaku_lora_state_dict,
)
from .core.peft import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    apply_network_alphas,
    extract_network_alphas,
    normalize_float_tensor,
    peft_lora_pairs,
)
from .core.layout import lora_modules


class NunchakuQwenImageLoraMixin(NunchakuLoraMixin):
    """Mixin-style method provider for quantized Qwen-Image LoRA runtime."""

    def _convert_lora_to_nunchaku(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert any supported Qwen-Image LoRA input into Nunchaku Lite tensors.

        Args:
            path_or_state_dict: Either a loaded LoRA state dict or a safetensors
                path. Inputs may be Nunchaku Lite format or Diffusers/PEFT-style
                Qwen-Image LoRA format.
        """

        state_dict = load_lora_state_dict(path_or_state_dict)
        if is_nunchaku_lite_lora_state_dict(state_dict):
            return normalize_nunchaku_lora_state_dict(state_dict, self)
        return convert_qwen_image_diffusers_lora_state_dict(state_dict, self)


def convert_qwen_image_diffusers_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert a Qwen-Image Diffusers/PEFT LoRA into lite low-rank tensors.

    Args:
        state_dict: External Qwen-Image LoRA tensors before Nunchaku packing.
        transformer: Patched transformer used to discover LoRA-capable modules
            and validate target dimensions.
    """

    diffusers_state = normalize_qwen_image_diffusers_lora_state_dict(state_dict)
    modules = lora_modules(transformer)
    pairs = peft_lora_pairs(diffusers_state)
    converted: dict[str, torch.Tensor] = {}
    handled: set[str] = set()

    grouped = group_fused_projection_pairs(pairs, QKV_PROJECTION_SPECS)
    for target_name, (spec, branches) in grouped.items():
        if target_name not in modules:
            continue
        module = modules[target_name]
        down, up = fuse_projection_branches(branches, module, pairs, spec)
        set_standard_converted_lora_pair(converted, target_name, down, up, module)
        handled.update(branches)

    for base_name, (lora_a, lora_b) in pairs.items():
        if base_name in handled or base_name not in modules:
            continue
        set_standard_converted_lora_pair(
            converted,
            base_name,
            lora_a.contiguous(),
            lora_b.contiguous(),
            modules[base_name],
        )
        handled.add(base_name)

    unsupported = sorted(base for base in pairs if base not in handled and is_qwen_image_lora_base_name(base))
    if unsupported:
        sample = ", ".join(unsupported[:5])
        raise ValueError(f"Unsupported {LORA_ERROR_LABEL} target(s) for nunchaku_lite: {sample}")
    return validate_nunchaku_lora_state_dict(converted, transformer)


def normalize_qwen_image_diffusers_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize Qwen-Image LoRA keys into PEFT ``lora_A/lora_B`` naming.

    Args:
        state_dict: Raw Qwen-Image LoRA tensors. Diffusers ``lora_down`` and
            ``lora_up`` names are rewritten to PEFT A/B names, and network
            alpha values are applied to the matching down tensors.
    """

    tensors = {}
    for key, value in state_dict.items():
        new_key = strip_transformer_prefix(key)
        new_key = new_key.replace(".lora_down.weight", LORA_A_SUFFIX)
        new_key = new_key.replace(".lora_up.weight", LORA_B_SUFFIX)
        tensors[new_key] = normalize_float_tensor(value)
    return apply_network_alphas(tensors, extract_network_alphas(tensors))


def is_qwen_image_lora_base_name(base: str) -> bool:
    """Return whether a normalized PEFT base name belongs to Qwen transformer blocks.

    Args:
        base: PEFT base key without the ``.lora_A.weight`` suffix.
    """

    return base.startswith("transformer_blocks.")
