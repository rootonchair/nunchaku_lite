"""Runtime LoRA loading and composition for patched Flux transformers."""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
from torch import nn

from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
from ..adapters.flux import convert_flux_state_dict
from .base import (
    NunchakuLoraMixin,
    NunchakuPipelineLoraMixin,
    bind_transformer_lora_methods,
    load_lora_state_dict,
    unpack_lowrank_weight,
)
from .conversion import (
    FusedProjectionSpec,
    convert_diffusers_lora_state_dict,
    is_nunchaku_lite_lora_state_dict,
    normalize_lite_lora_state_dict,
    set_converted_lora_pair,
    strip_transformer_prefix,
)
from .peft import apply_network_alphas, extract_network_alphas, normalize_float_tensor


QKV_PROJECTION_SPECS = (
    FusedProjectionSpec(target=".attn.to_qkv", branches=(".attn.to_q", ".attn.to_k", ".attn.to_v")),
    FusedProjectionSpec(
        target=".attn.add_qkv_proj",
        branches=(".attn.add_q_proj", ".attn.add_k_proj", ".attn.add_v_proj"),
    ),
)


class NunchakuFluxTransformerLoraMixin(NunchakuLoraMixin):
    """Mixin-style method provider for quantized FLUX transformer LoRA runtime."""

    def _convert_lora_to_lite(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return convert_flux_lora_to_lite(path_or_state_dict, self)

    def fuse_lora(self, *args, **kwargs) -> None:
        raise NotImplementedError("nunchaku_lite FLUX runtime LoRA keeps adapters as low-rank branches.")


class NunchakuFluxPipelineLoraMixin(NunchakuPipelineLoraMixin):
    """Mixin-style method provider for Diffusers-compatible FLUX pipeline LoRA APIs."""

    def _bind_transformer_lora_methods(self, transformer: nn.Module) -> None:
        bind_transformer_lora_methods(transformer, NunchakuFluxTransformerLoraMixin)

    def fuse_lora(self, *args, **kwargs) -> None:
        raise NotImplementedError("nunchaku_lite FLUX runtime LoRA does not support fusing into quantized weights.")

    def unfuse_lora(self, *args, **kwargs) -> None:
        raise NotImplementedError("nunchaku_lite FLUX runtime LoRA does not support fusing into quantized weights.")


def convert_flux_lora_to_lite(
    state_dict_or_path: str | Path | dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert a Flux LoRA into packed lite SVDQ low-rank tensors."""

    state_dict = load_lora_state_dict(state_dict_or_path)
    if is_nunchaku_lite_lora_state_dict(state_dict):
        return normalize_lite_lora_state_dict(
            state_dict,
            transformer,
            key_converter=convert_flux_state_dict,
        )

    return convert_diffusers_lora_state_dict(
        state_dict,
        transformer,
        projection_specs=QKV_PROJECTION_SPECS,
        normalize_state_dict=_to_diffusers_format,
        map_direct_pair=_map_direct_diffusers_pair,
        is_transformer_lora_key=_is_transformer_lora_key,
        set_pair=_set_flux_converted_pair,
    )


def _to_diffusers_format(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors = _handle_kohya_lora(state_dict)
    tensors = {key: normalize_float_tensor(value) for key, value in tensors.items()}
    if any("lora_A" in key or "lora_B" in key for key in tensors):
        tensors = {strip_transformer_prefix(key): value for key, value in tensors.items()}
        return apply_network_alphas(tensors, extract_network_alphas(tensors))

    converted, _network_alphas = FluxLoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
    converted = convert_unet_state_dict_to_peft(converted)
    converted = {strip_transformer_prefix(key): value for key, value in converted.items()}
    alphas = {strip_transformer_prefix(key): value for key, value in (_network_alphas or {}).items()}
    return apply_network_alphas(converted, alphas)


def _set_flux_converted_pair(
    converted: dict[str, torch.Tensor],
    target_name: str,
    down: torch.Tensor,
    up: torch.Tensor,
    module: SVDQW4A4Linear | AWQW4A16Linear,
) -> None:
    set_converted_lora_pair(
        converted,
        target_name,
        down,
        up,
        module,
        awq_up_transform=_reorder_adanorm_lora_up,
    )


def _handle_kohya_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict or not all(key.startswith("lora_transformer_") for key in state_dict):
        return dict(state_dict)

    converted = {}
    for key, value in state_dict.items():
        new_key = key.replace("lora_transformer_", "transformer.")
        new_key = new_key.replace("single_transformer_blocks_", "single_transformer_blocks.")
        new_key = new_key.replace("transformer_blocks_", "transformer_blocks.")
        new_key = new_key.replace("_attn_", ".attn.")
        new_key = new_key.replace("_ff_context_net_0_proj.", ".ff_context.net.0.proj.")
        new_key = new_key.replace("_ff_context_net_2.", ".ff_context.net.2.")
        new_key = new_key.replace("_ff_net_0_proj.", ".ff.net.0.proj.")
        new_key = new_key.replace("_ff_net_2.", ".ff.net.2.")
        new_key = new_key.replace("_proj_mlp.", ".proj_mlp.")
        new_key = new_key.replace("_proj_out.", ".proj_out.")
        new_key = new_key.replace("to_out_0.", "to_out.0.")
        new_key = new_key.replace(".lora_down.", ".lora_A.")
        new_key = new_key.replace(".lora_up.", ".lora_B.")
        converted[new_key] = value
    return converted


def _map_direct_diffusers_pair(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear],
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    target_name = _direct_target_name(base_name)
    if target_name is not None and target_name in modules:
        return [(target_name, lora_a.contiguous(), lora_b.contiguous())]

    split_targets = _single_proj_out_targets(base_name, lora_a, lora_b, modules)
    if split_targets:
        return split_targets
    return []


def _reorder_adanorm_lora_up(target_name: str, lora_up: torch.Tensor) -> torch.Tensor:
    splits = _adanorm_splits(target_name)
    channels, rank = lora_up.shape
    if channels % splits != 0:
        raise ValueError(f"AdaNorm LoRA output dimension {channels} is not divisible by {splits}.")
    return lora_up.view(splits, channels // splits, rank).transpose(0, 1).reshape(channels, rank).contiguous()


def _adanorm_splits(target_name: str) -> int:
    if target_name.startswith("single_transformer_blocks.") and target_name.endswith(".norm.linear"):
        return 3
    if target_name.startswith("transformer_blocks.") and target_name.endswith((".norm1.linear", ".norm1_context.linear")):
        return 6
    raise ValueError(f"Unsupported AWQ LoRA target {target_name!r}.")


def _direct_target_name(base_name: str) -> str | None:
    replacements = {
        ".proj_mlp": ".mlp_fc1",
        ".proj_out.linears.0": ".attn.to_out",
        ".proj_out.linears.1": ".mlp_fc2",
    }
    for source, target in replacements.items():
        if source in base_name:
            return base_name.replace(source, target)
    if ".attn.to_out.0" in base_name:
        return base_name.replace(".attn.to_out.0", ".attn.to_out.0")
    return base_name


def _single_proj_out_targets(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear],
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
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


def _is_transformer_lora_key(base: str) -> bool:
    return base.startswith(("transformer_blocks.", "single_transformer_blocks."))
