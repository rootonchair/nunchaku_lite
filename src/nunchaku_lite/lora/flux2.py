"""Runtime LoRA loading and composition for patched Flux2 transformers."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn

from ..linear import AWQW4A16Linear, SVDQW4A4Linear
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


FLUX2_COMFYUI_DOUBLE_BLOCK_REPLACEMENTS = (
    ("img_attn.qkv", "attn.to_qkv"),
    ("txt_attn.qkv", "attn.to_added_qkv"),
    ("img_attn.proj", "attn.to_out.0"),
    ("txt_attn.proj", "attn.to_add_out"),
    ("img_mlp.0", "ff.linear_in"),
    ("img_mlp.2", "ff.linear_out"),
    ("txt_mlp.0", "ff_context.linear_in"),
    ("txt_mlp.2", "ff_context.linear_out"),
)

FLUX2_QKV_PROJECTION_SPECS = (
    FusedProjectionSpec(target=".attn.to_qkv", branches=(".attn.to_q", ".attn.to_k", ".attn.to_v")),
    FusedProjectionSpec(
        target=".attn.to_added_qkv",
        branches=(".attn.add_q_proj", ".attn.add_k_proj", ".attn.add_v_proj"),
    ),
)


class NunchakuFlux2TransformerLoraMixin(NunchakuLoraMixin):
    """Mixin-style method provider for quantized Flux2 transformer LoRA runtime."""

    def _convert_lora_to_nunchaku(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert any supported Flux2 LoRA input into Nunchaku Lite tensors.

        Args:
            path_or_state_dict: Either a loaded LoRA state dict or a safetensors
                path. Inputs may be Nunchaku format or Flux2 ComfyUI/PEFT LoRA
                format.
        """

        state_dict = load_lora_state_dict(path_or_state_dict)
        if is_nunchaku_lite_lora_state_dict(state_dict):
            return normalize_nunchaku_lora_state_dict(state_dict, self)
        return convert_flux2_diffusers_lora_state_dict(state_dict, self)


def convert_flux2_diffusers_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert a Flux2 ComfyUI/PEFT LoRA state dict into Nunchaku tensors.

    Args:
        state_dict: External Flux2 LoRA tensors before Nunchaku packing.
        transformer: Patched Flux2 transformer used to discover LoRA-capable
            modules and validate target dimensions.
    """

    diffusers_state = normalize_flux2_diffusers_lora_state_dict(state_dict)
    modules = lora_modules(transformer)
    pairs = peft_lora_pairs(diffusers_state)
    converted: dict[str, torch.Tensor] = {}
    handled: set[str] = set()

    grouped = group_fused_projection_pairs(pairs, FLUX2_QKV_PROJECTION_SPECS)
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
        for target_name, down, up in flux2_lora_targets(base_name, lora_a, lora_b, modules):
            set_standard_converted_lora_pair(converted, target_name, down, up, modules[target_name])
            handled.add(base_name)

    unsupported = sorted(base for base in pairs if base not in handled and is_flux2_lora_base_name(base))
    if unsupported:
        sample = ", ".join(unsupported[:5])
        raise ValueError(f"Unsupported {LORA_ERROR_LABEL} target(s) for nunchaku_lite: {sample}")
    return validate_nunchaku_lora_state_dict(converted, transformer)


def normalize_flux2_diffusers_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize Flux2 LoRA keys into PEFT ``lora_A/lora_B`` naming.

    Args:
        state_dict: Raw Flux2 LoRA tensors, including ComfyUI
            ``diffusion_model.*`` keys or PEFT-style transformer keys.
    """

    tensors = {}
    for key, value in normalize_flux2_comfyui_lora_keys(state_dict).items():
        new_key = strip_transformer_prefix(key)
        new_key = new_key.replace(".lora_down.weight", LORA_A_SUFFIX)
        new_key = new_key.replace(".lora_up.weight", LORA_B_SUFFIX)
        tensors[new_key] = normalize_float_tensor(value)
    return apply_network_alphas(tensors, extract_network_alphas(tensors))


def normalize_flux2_comfyui_lora_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite ComfyUI Flux2 LoRA keys into patched transformer key form.

    Args:
        state_dict: Raw Flux2 LoRA tensors whose keys usually start with
            ``diffusion_model.`` in ComfyUI checkpoints.
    """

    if not state_dict or not all(key.startswith("diffusion_model.") for key in state_dict):
        return dict(state_dict)
    if not any(".double_blocks." in key or ".single_blocks." in key for key in state_dict):
        return dict(state_dict)

    converted = {}
    for key, value in state_dict.items():
        new_key = key[len("diffusion_model.") :]
        new_key = _normalize_flux2_comfyui_double_block_key(new_key)
        new_key = _normalize_flux2_comfyui_single_block_key(new_key)
        converted[new_key] = value
    return converted


def _normalize_flux2_comfyui_double_block_key(key: str) -> str:
    """Rewrite one ComfyUI double-block Flux2 key if it targets a known module."""

    match = re.match(r"double_blocks\.(\d+)\.(.+?)(\.lora_[AB]\.weight|\.alpha)$", key)
    if match is None:
        return key
    block_index, target, suffix = match.groups()
    for source, replacement in FLUX2_COMFYUI_DOUBLE_BLOCK_REPLACEMENTS:
        if target == source:
            return f"transformer_blocks.{block_index}.{replacement}{suffix}"
    return key


def _normalize_flux2_comfyui_single_block_key(key: str) -> str:
    """Rewrite one ComfyUI single-block Flux2 key if it targets linear1/linear2."""

    match = re.match(r"single_blocks\.(\d+)\.(linear[12])(\.lora_[AB]\.weight|\.alpha)$", key)
    if match is None:
        return key
    block_index, target, suffix = match.groups()
    return f"single_transformer_blocks.{block_index}.attn.{target}{suffix}"


def flux2_lora_targets(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear],
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Map one normalized Flux2 PEFT LoRA pair to Nunchaku target pairs.

    Args:
        base_name: PEFT base key without the ``.lora_A.weight`` suffix.
        lora_a: LoRA down tensor from PEFT format.
        lora_b: LoRA up tensor from PEFT format.
        modules: LoRA-capable modules on the patched Flux2 transformer.
    """

    if is_flux2_single_linear1_lora_base_name(base_name):
        return flux2_single_linear1_lora_targets(base_name, lora_a, lora_b, modules)
    if is_flux2_single_linear2_lora_base_name(base_name):
        return flux2_single_linear2_lora_targets(base_name, lora_a, lora_b, modules)
    if base_name in modules:
        return [(base_name, lora_a.contiguous(), lora_b.contiguous())]
    return []


def flux2_single_linear1_lora_targets(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear],
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Split a ComfyUI Flux2 single-block ``linear1`` LoRA into qkv/mlp targets."""

    suffix = ".to_qkv_mlp_proj" if base_name.endswith(".to_qkv_mlp_proj") else ".linear1"
    prefix = base_name[: -len(suffix)]
    qkv_name = f"{prefix}.qkv_proj"
    mlp_name = f"{prefix}.mlp_fc1"
    if qkv_name not in modules or mlp_name not in modules:
        return []
    qkv_out = modules[qkv_name].out_features
    mlp_out = modules[mlp_name].out_features
    if lora_b.shape[0] < qkv_out + mlp_out:
        return []
    qkv_up = lora_b[:qkv_out].contiguous()
    mlp_up = lora_b[qkv_out : qkv_out + mlp_out].contiguous()
    return [(qkv_name, lora_a.contiguous(), qkv_up), (mlp_name, lora_a.contiguous(), mlp_up)]


def flux2_single_linear2_lora_targets(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear],
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Split a ComfyUI Flux2 single-block ``linear2`` LoRA into out/mlp targets."""

    suffix = ".to_out" if base_name.endswith(".to_out") else ".linear2"
    prefix = base_name[: -len(suffix)]
    out_name = f"{prefix}.out_proj"
    mlp_name = f"{prefix}.mlp_fc2"
    if out_name not in modules or mlp_name not in modules:
        return []
    out_in = modules[out_name].in_features
    mlp_in = modules[mlp_name].in_features
    if lora_a.shape[1] < out_in + mlp_in:
        return []
    out_down = lora_a[:, :out_in].contiguous()
    mlp_down = lora_a[:, out_in : out_in + mlp_in].contiguous()
    return [(out_name, out_down, lora_b.contiguous()), (mlp_name, mlp_down, lora_b.contiguous())]


def is_flux2_lora_base_name(base: str) -> bool:
    """Return whether a normalized PEFT base name belongs to Flux2 blocks."""

    return base.startswith(("transformer_blocks.", "single_transformer_blocks.", "double_blocks.", "single_blocks."))


def is_flux2_single_linear1_lora_base_name(base: str) -> bool:
    """Return whether a Flux2 single-stream LoRA base targets combined linear1."""

    return base.startswith("single_transformer_blocks.") and base.endswith((".linear1", ".to_qkv_mlp_proj"))


def is_flux2_single_linear2_lora_base_name(base: str) -> bool:
    """Return whether a Flux2 single-stream LoRA base targets combined linear2."""

    return base.startswith("single_transformer_blocks.") and base.endswith((".linear2", ".to_out"))
