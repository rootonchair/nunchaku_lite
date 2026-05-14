"""Runtime LoRA loading and composition for patched Flux transformers."""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
from torch import nn

from ..adapters.flux import convert_flux_state_dict
from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
from .base import (
    NunchakuLoraMixin,
    load_lora_state_dict,
    lora_modules,
    pack_lowrank_weight,
    pad_lora_tensor,
)
from .common import (
    LORA_ERROR_LABEL,
    QKV_PROJECTION_SPECS,
    fuse_projection_branches,
    group_fused_projection_pairs,
    is_nunchaku_lite_lora_state_dict,
    normalize_nunchaku_lora_keys_and_validate,
    strip_transformer_prefix,
    validate_nunchaku_lora_state_dict,
)
from .peft import apply_network_alphas, extract_network_alphas, normalize_float_tensor, peft_lora_pairs


FLUX_KOHYA_KEY_REPLACEMENTS = (
    ("lora_transformer_", "transformer."),
    ("single_transformer_blocks_", "single_transformer_blocks."),
    ("transformer_blocks_", "transformer_blocks."),
    ("_attn_", ".attn."),
    ("_ff_context_net_0_proj.", ".ff_context.net.0.proj."),
    ("_ff_context_net_2.", ".ff_context.net.2."),
    ("_ff_net_0_proj.", ".ff.net.0.proj."),
    ("_ff_net_2.", ".ff.net.2."),
    ("_proj_mlp.", ".proj_mlp."),
    ("_proj_out.", ".proj_out."),
    ("to_out_0.", "to_out.0."),
    (".lora_down.", ".lora_A."),
    (".lora_up.", ".lora_B."),
)

FLUX_DIRECT_TARGET_REPLACEMENTS = (
    (".proj_mlp", ".mlp_fc1"),
    (".proj_out.linears.0", ".attn.to_out"),
    (".proj_out.linears.1", ".mlp_fc2"),
)

FLUX_LORA_PREFIXES = ("transformer_blocks.", "single_transformer_blocks.")


class NunchakuFluxTransformerLoraMixin(NunchakuLoraMixin):
    """Mixin-style method provider for quantized FLUX transformer LoRA runtime."""

    def _convert_lora_to_nunchaku(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert any supported Flux LoRA input into Nunchaku Lite tensors.

        Args:
            path_or_state_dict: Either a loaded LoRA state dict or a safetensors
                path. Inputs may be Nunchaku Lite format, Diffusers/PEFT
                format, or Kohya Flux LoRA format.
        """

        state_dict = load_lora_state_dict(path_or_state_dict)
        if is_nunchaku_lite_lora_state_dict(state_dict):
            flux_converted_state_dict = convert_flux_state_dict(state_dict)
            return normalize_nunchaku_lora_keys_and_validate(flux_converted_state_dict, self)
        return convert_flux_diffusers_lora_state_dict(state_dict, self)

    def fuse_lora(self, *args, **kwargs) -> None:
        """Reject fuse requests because Flux runtime LoRAs remain low-rank branches."""

        raise NotImplementedError("nunchaku_lite FLUX runtime LoRA keeps adapters as low-rank branches.")


def convert_flux_diffusers_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert a Flux Diffusers/PEFT LoRA state dict into lite low-rank tensors.

    Args:
        state_dict: External Flux LoRA tensors before Nunchaku packing.
        transformer: Patched Flux transformer used to discover LoRA-capable
            modules and validate target dimensions.
    """

    diffusers_state = normalize_flux_diffusers_lora_state_dict(state_dict)
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
        set_flux_converted_lora_pair(converted, target_name, down, up, module)
        handled.update(branches)

    for base_name, (lora_a, lora_b) in pairs.items():
        if base_name in handled:
            continue
        for target_name, down, up in flux_direct_lora_targets(base_name, lora_a, lora_b, modules):
            set_flux_converted_lora_pair(converted, target_name, down, up, modules[target_name])
            handled.add(base_name)

    unsupported = sorted(base for base in pairs if base not in handled and is_flux_lora_base_name(base))
    if unsupported:
        sample = ", ".join(unsupported[:5])
        raise ValueError(f"Unsupported {LORA_ERROR_LABEL} target(s) for nunchaku_lite: {sample}")
    return validate_nunchaku_lora_state_dict(converted, transformer)


def normalize_flux_diffusers_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize Flux LoRA keys into PEFT ``lora_A/lora_B`` naming.

    Args:
        state_dict: Raw Flux LoRA tensors, including PEFT-style keys, Kohya
            ``lora_transformer_*`` keys, or Diffusers loader-compatible keys.
    """

    tensors = {
        key: normalize_float_tensor(value)
        for key, value in normalize_flux_kohya_lora_keys(state_dict).items()
    }
    if any("lora_A" in key or "lora_B" in key for key in tensors):
        tensors = {strip_transformer_prefix(key): value for key, value in tensors.items()}
        return apply_network_alphas(tensors, extract_network_alphas(tensors))

    converted, network_alphas = FluxLoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
    converted = convert_unet_state_dict_to_peft(converted)
    converted = {strip_transformer_prefix(key): value for key, value in converted.items()}
    alphas = {strip_transformer_prefix(key): value for key, value in (network_alphas or {}).items()}
    return apply_network_alphas(converted, alphas)


def normalize_flux_kohya_lora_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite Kohya Flux LoRA keys into Diffusers transformer key form.

    Args:
        state_dict: Raw state dict whose keys all start with
            ``lora_transformer_`` when it is a Kohya Flux LoRA.
    """

    if not state_dict or not all(key.startswith("lora_transformer_") for key in state_dict):
        return dict(state_dict)

    converted = {}
    for key, value in state_dict.items():
        new_key = key
        for source, target in FLUX_KOHYA_KEY_REPLACEMENTS:
            new_key = new_key.replace(source, target)
        converted[new_key] = value
    return converted


def set_flux_converted_lora_pair(
    converted: dict[str, torch.Tensor],
    target_name: str,
    down: torch.Tensor,
    up: torch.Tensor,
    module: SVDQW4A4Linear | AWQW4A16Linear,
) -> None:
    """Store a converted Flux LoRA pair with Flux-specific AWQ handling.

    Args:
        converted: Output lite LoRA state dict being assembled.
        target_name: Patched module name without the ``.proj_down/.proj_up``
            suffix.
        down: Logical down tensor in rank-by-input layout.
        up: Logical up tensor in output-by-rank layout.
        module: Target quantized linear module. AWQ AdaNorm targets require
            Flux-specific output-channel reordering before padding.
    """

    if isinstance(module, SVDQW4A4Linear):
        down = pack_lowrank_weight(down, down=True)
        up = pack_lowrank_weight(up, down=False)
    else:
        up = reorder_flux_adanorm_lora_up(target_name, up)
        down = pad_lora_tensor(down, divisor=16, dim=0)
        up = pad_lora_tensor(up, divisor=16, dim=1)
    converted[f"{target_name}.proj_down"] = down
    converted[f"{target_name}.proj_up"] = up


def flux_direct_lora_targets(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear],
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Map one normalized Flux PEFT LoRA pair to lite target module pairs.

    Args:
        base_name: PEFT base key without the ``.lora_A.weight`` suffix.
        lora_a: LoRA down tensor from PEFT format.
        lora_b: LoRA up tensor from PEFT format.
        modules: LoRA-capable modules on the patched Flux transformer.
    """

    target_name = flux_direct_target_name(base_name)
    if target_name in modules:
        return [(target_name, lora_a.contiguous(), lora_b.contiguous())]

    if ".proj_out" not in base_name or ".linears." in base_name:
        return []
    prefix = base_name.rsplit(".proj_out", 1)[0]
    attn_name = f"{prefix}.attn.to_out"
    mlp_name = f"{prefix}.mlp_fc2"
    if attn_name not in modules or mlp_name not in modules:
        return []
    attn_in = modules[attn_name].in_features
    mlp_in = modules[mlp_name].in_features
    if lora_a.shape[1] < mlp_in + attn_in:
        return []

    attn_down = lora_a[:, :attn_in].contiguous()
    mlp_down = lora_a[:, attn_in : attn_in + mlp_in].contiguous()
    return [(mlp_name, mlp_down, lora_b.contiguous()), (attn_name, attn_down, lora_b.contiguous())]


def flux_direct_target_name(base_name: str) -> str:
    """Return the patched Flux module name for a directly mapped LoRA key.

    Args:
        base_name: Normalized PEFT base name before Nunchaku module renaming.
    """

    for source, target in FLUX_DIRECT_TARGET_REPLACEMENTS:
        if source in base_name:
            return base_name.replace(source, target)
    return base_name


def reorder_flux_adanorm_lora_up(target_name: str, lora_up: torch.Tensor) -> torch.Tensor:
    """Reorder Flux AdaNorm AWQ LoRA-up channels into Nunchaku runtime layout.

    Args:
        target_name: Patched AdaNorm linear module name.
        lora_up: Logical LoRA up tensor in output-by-rank layout.
    """

    if target_name.startswith("single_transformer_blocks.") and target_name.endswith(".norm.linear"):
        splits = 3
    elif target_name.startswith("transformer_blocks.") and target_name.endswith(
        (".norm1.linear", ".norm1_context.linear")
    ):
        splits = 6
    else:
        raise ValueError(f"Unsupported AWQ LoRA target {target_name!r}.")

    channels, rank = lora_up.shape
    if channels % splits != 0:
        raise ValueError(f"AdaNorm LoRA output dimension {channels} is not divisible by {splits}.")
    return lora_up.view(splits, channels // splits, rank).transpose(0, 1).reshape(channels, rank).contiguous()


def is_flux_lora_base_name(base: str) -> bool:
    """Return whether a normalized PEFT base name belongs to Flux transformer blocks.

    Args:
        base: PEFT base key without the ``.lora_A.weight`` suffix.
    """

    return base.startswith(FLUX_LORA_PREFIXES)
