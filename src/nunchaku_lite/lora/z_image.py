"""Runtime LoRA loading and composition for patched Z-Image transformers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .core.runtime import NunchakuLoraMixin, load_lora_state_dict
from .core.convert import (
    FusedProjectionSpec,
    LORA_ERROR_LABEL,
    fuse_projection_branches,
    group_fused_projection_pairs,
    is_nunchaku_lite_lora_state_dict,
    normalize_nunchaku_lora_state_dict,
    set_standard_converted_lora_pair,
    strip_transformer_prefix,
    validate_nunchaku_lora_state_dict,
)
from .core.peft import LORA_A_SUFFIX, LORA_B_SUFFIX, apply_network_alphas, extract_network_alphas, normalize_float_tensor
from .core.peft import peft_lora_pairs
from .core.layout import lora_modules


Z_IMAGE_QKV_PROJECTION_SPECS = (
    FusedProjectionSpec(target=".attention.to_qkv", branches=(".attention.to_q", ".attention.to_k", ".attention.to_v")),
)

Z_IMAGE_SWIGLU_PROJECTION_SPECS = (
    FusedProjectionSpec(
        target=".feed_forward.net.0.proj",
        branches=(".feed_forward.w3", ".feed_forward.w1"),
    ),
)


class NunchakuZImageTransformerLoraMixin(NunchakuLoraMixin):
    """Mixin-style method provider for quantized Z-Image LoRA runtime."""

    def _convert_lora_to_nunchaku(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert any supported Z-Image LoRA input into Nunchaku Lite tensors.

        Args:
            path_or_state_dict: Either a loaded LoRA state dict or a safetensors
                path. Inputs may be Nunchaku Lite format or Diffusers/PEFT-style
                Z-Image LoRA format.
        """

        state_dict = load_lora_state_dict(path_or_state_dict)
        if is_nunchaku_lite_lora_state_dict(state_dict):
            return normalize_nunchaku_lora_state_dict(state_dict, self)
        return convert_z_image_diffusers_lora_state_dict(state_dict, self)


def convert_z_image_diffusers_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert a Z-Image Diffusers/PEFT LoRA into runtime low-rank tensors.

    Args:
        state_dict: External Z-Image LoRA tensors before Nunchaku packing.
        transformer: Patched transformer used to discover LoRA-capable modules
            and validate target dimensions.
    """

    diffusers_state = normalize_z_image_diffusers_lora_state_dict(state_dict)
    modules = lora_modules(transformer)
    pairs = peft_lora_pairs(diffusers_state)
    converted: dict[str, torch.Tensor] = {}
    handled: set[str] = set()

    for specs in (Z_IMAGE_QKV_PROJECTION_SPECS, Z_IMAGE_SWIGLU_PROJECTION_SPECS):
        grouped = group_fused_projection_pairs(pairs, specs)
        for target_name, (spec, branches) in grouped.items():
            if target_name not in modules:
                continue
            module = modules[target_name]
            down, up = fuse_projection_branches(branches, module, pairs, spec)
            set_standard_converted_lora_pair(converted, target_name, down, up, module)
            handled.update(branches)

    for base_name, (lora_a, lora_b) in pairs.items():
        if base_name in handled:
            continue
        target_name = z_image_direct_target_name(base_name)
        if target_name not in modules:
            continue
        set_standard_converted_lora_pair(
            converted,
            target_name,
            lora_a.contiguous(),
            lora_b.contiguous(),
            modules[target_name],
        )
        handled.add(base_name)

    unsupported = sorted(base for base in pairs if base not in handled and is_z_image_lora_base_name(base))
    if unsupported:
        sample = ", ".join(unsupported[:5])
        raise ValueError(f"Unsupported {LORA_ERROR_LABEL} target(s) for nunchaku_lite: {sample}")
    return validate_nunchaku_lora_state_dict(converted, transformer)


def normalize_z_image_diffusers_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize Z-Image LoRA keys into PEFT ``lora_A/lora_B`` naming.

    Args:
        state_dict: Raw Z-Image LoRA tensors. Diffusers ``diffusion_model`` or
            ``transformer`` component prefixes are removed, Diffusers
            ``lora_down/up`` names are rewritten to PEFT A/B names, and network
            alpha values are applied to matching down tensors.
    """

    tensors = {}
    for key, value in state_dict.items():
        new_key = strip_transformer_prefix(key)
        if new_key.startswith("diffusion_model."):
            new_key = new_key[len("diffusion_model.") :]
        new_key = new_key.replace(".lora_down.weight", LORA_A_SUFFIX)
        new_key = new_key.replace(".lora_up.weight", LORA_B_SUFFIX)
        tensors[new_key] = normalize_float_tensor(value)
    return apply_network_alphas(tensors, extract_network_alphas(tensors))


def z_image_direct_target_name(base_name: str) -> str:
    """Return the patched Z-Image module name for a directly mapped LoRA key.

    Args:
        base_name: Normalized PEFT base name before patched module renaming.
    """

    if ".feed_forward.w2" in base_name:
        return base_name.replace(".feed_forward.w2", ".feed_forward.net.2")
    return base_name


def is_z_image_lora_base_name(base: str) -> bool:
    """Return whether a normalized PEFT base name belongs to Z-Image blocks.

    Args:
        base: PEFT base key without the ``.lora_A.weight`` suffix.
    """

    return base.startswith(("layers.", "noise_refiner.", "context_refiner."))
