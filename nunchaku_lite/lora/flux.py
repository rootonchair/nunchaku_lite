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
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    NunchakuLoraMixin,
    NunchakuPipelineLoraMixin,
    apply_network_alphas,
    bind_mixin_methods,
    diffusers_pairs,
    ensure_lora_runtime,
    extract_network_alphas,
    fit_lora_tensor,
    load_lora_state_dict,
    lora_modules,
    normalize_float_tensor,
    pack_lowrank_weight,
    pad_lora_tensor,
    unpack_lowrank_weight,
    validate_lite_lora_state_dict,
)


def bind_flux_lora_methods(transformer: nn.Module) -> None:
    """Attach runtime LoRA methods to a patched Flux transformer."""

    transformer._nunchaku_lite_lora_model_name = "FLUX"
    bind_mixin_methods(
        transformer,
        NunchakuFluxTransformerLoraMixin,
        (
            "load_lora",
            "load_lora_adapter",
            "set_lora_strength",
            "set_adapters",
            "reset_lora",
            "delete_adapters",
            "unload_lora",
            "enable_lora",
            "disable_lora",
            "get_list_adapters",
            "get_active_adapters",
            "fuse_lora",
            "_convert_lora_to_lite",
        ),
    )
    ensure_lora_runtime(transformer)


def bind_flux_pipeline_lora_methods(pipeline) -> None:
    """Attach Diffusers-compatible runtime LoRA methods to a Flux pipeline."""

    pipeline._nunchaku_lite_lora_model_name = "FLUX"
    pipeline._nunchaku_lite_lora_component_name = "transformer"
    bind_mixin_methods(
        pipeline,
        NunchakuFluxPipelineLoraMixin,
        (
            "load_lora_weights",
            "load_lora_adapter",
            "set_adapters",
            "delete_adapters",
            "unload_lora_weights",
            "enable_lora",
            "disable_lora",
            "get_list_adapters",
            "get_active_adapters",
            "fuse_lora",
            "unfuse_lora",
            "_pipeline_transformer",
            "_bind_transformer_lora_methods",
        ),
    )
    pipeline._nunchaku_lite_lora_pipeline_api_bound = True


class NunchakuFluxTransformerLoraMixin(NunchakuLoraMixin):
    """Mixin-style method provider for quantized FLUX transformer LoRA runtime."""

    _nunchaku_lite_lora_model_name = "FLUX"

    def _convert_lora_to_lite(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return convert_flux_lora_to_lite(path_or_state_dict, self)

    def fuse_lora(self, *args, **kwargs) -> None:
        raise NotImplementedError("nunchaku_lite FLUX runtime LoRA keeps adapters as low-rank branches.")


class NunchakuFluxPipelineLoraMixin(NunchakuPipelineLoraMixin):
    """Mixin-style method provider for Diffusers-compatible FLUX pipeline LoRA APIs."""

    _nunchaku_lite_lora_model_name = "FLUX"

    def _bind_transformer_lora_methods(self, transformer: nn.Module) -> None:
        bind_flux_lora_methods(transformer)

    def fuse_lora(self, *args, **kwargs) -> None:
        raise NotImplementedError("nunchaku_lite FLUX runtime LoRA does not support fusing into quantized weights.")

    def unfuse_lora(self, *args, **kwargs) -> None:
        raise NotImplementedError("nunchaku_lite FLUX runtime LoRA does not support fusing into quantized weights.")


def is_nunchaku_flux_lora(state_dict: dict[str, torch.Tensor]) -> bool:
    """Return whether a state dict already uses Nunchaku-style low-rank keys."""

    return any(
        key.endswith((".proj_down", ".proj_up", ".lora_down", ".lora_up")) for key in state_dict
    ) and not any("lora_A" in key or "lora_B" in key for key in state_dict)


def convert_flux_lora_to_lite(
    state_dict_or_path: str | Path | dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert a Flux LoRA into packed lite SVDQ low-rank tensors."""

    state_dict = load_lora_state_dict(state_dict_or_path)
    if is_nunchaku_flux_lora(state_dict):
        return _normalize_nunchaku_lora_state_dict(state_dict, transformer)
    return _convert_diffusers_lora_state_dict(state_dict, transformer)


def _normalize_nunchaku_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    converted = convert_flux_state_dict(state_dict)
    return validate_lite_lora_state_dict(converted, transformer, model_name="Flux")


def _convert_diffusers_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    diffusers_state = _to_diffusers_format(state_dict)
    modules = lora_modules(transformer)
    pairs = diffusers_pairs(diffusers_state)
    converted: dict[str, torch.Tensor] = {}

    handled: set[str] = set()
    grouped = _group_qkv_pairs(pairs)
    for target_name, branches in grouped.items():
        if target_name not in modules:
            continue
        module = modules[target_name]
        down, up = _fuse_qkv_branches(branches, module, pairs)
        _set_converted_pair(converted, target_name, down, up, module)
        handled.update(branches)

    for base_name, (lora_a, lora_b) in pairs.items():
        if base_name in handled:
            continue
        for target_name, down, up in _map_direct_diffusers_pair(base_name, lora_a, lora_b, modules):
            _set_converted_pair(converted, target_name, down, up, modules[target_name])
            handled.add(base_name)

    unsupported = sorted(base for base in pairs if base not in handled and _is_transformer_lora_key(base))
    if unsupported:
        sample = ", ".join(unsupported[:5])
        raise ValueError(f"Unsupported Flux LoRA target(s) for nunchaku_lite: {sample}")
    return validate_lite_lora_state_dict(converted, transformer, model_name="Flux")


def _to_diffusers_format(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors = _handle_kohya_lora(state_dict)
    tensors = {key: normalize_float_tensor(value) for key, value in tensors.items()}
    if any("lora_A" in key or "lora_B" in key for key in tensors):
        tensors = {_strip_transformer_prefix(key): value for key, value in tensors.items()}
        return apply_network_alphas(tensors, extract_network_alphas(tensors))

    converted, _network_alphas = FluxLoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
    converted = convert_unet_state_dict_to_peft(converted)
    converted = {_strip_transformer_prefix(key): value for key, value in converted.items()}
    alphas = {_strip_transformer_prefix(key): value for key, value in (_network_alphas or {}).items()}
    return apply_network_alphas(converted, alphas)


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


def _strip_transformer_prefix(key: str) -> str:
    for prefix in ("base_model.model.transformer.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _group_qkv_pairs(pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for base in pairs:
        if ".attn.to_qkv" in base or ".attn.add_qkv_proj" in base:
            groups[base] = [base]
        elif ".attn.to_q" in base:
            target = base.replace(".attn.to_q", ".attn.to_qkv")
            branches = [
                base,
                base.replace(".attn.to_q", ".attn.to_k"),
                base.replace(".attn.to_q", ".attn.to_v"),
            ]
            groups[target] = [branch for branch in branches if branch in pairs]
        elif ".attn.add_q_proj" in base:
            target = base.replace(".attn.add_q_proj", ".attn.add_qkv_proj")
            branches = [
                base,
                base.replace(".attn.add_q_proj", ".attn.add_k_proj"),
                base.replace(".attn.add_q_proj", ".attn.add_v_proj"),
            ]
            groups[target] = [branch for branch in branches if branch in pairs]
    return groups


def _fuse_qkv_branches(
    branch_names: list[str],
    module: SVDQW4A4Linear,
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(branch_names) == 1 and (".to_qkv" in branch_names[0] or ".add_qkv_proj" in branch_names[0]):
        lora_a, lora_b = pairs[branch_names[0]]
        return lora_a.contiguous(), lora_b.contiguous()

    branch_order = (
        ["to_q", "to_k", "to_v"]
        if ".attn.to_" in branch_names[0]
        else ["add_q_proj", "add_k_proj", "add_v_proj"]
    )
    by_suffix = {name.rsplit(".", 1)[-1]: name for name in branch_names}
    ordered = [by_suffix[suffix] for suffix in branch_order if suffix in by_suffix]
    if not ordered:
        raise ValueError("No QKV LoRA branches were provided.")

    first_a, first_b = pairs[ordered[0]]
    in_features = first_a.shape[1]
    out_per_branch = module.out_features // 3
    branch_pairs = []
    for suffix in branch_order:
        name = by_suffix.get(suffix)
        if name is None:
            branch_pairs.append((first_a.clone(), torch.zeros(out_per_branch, first_a.shape[0], dtype=first_b.dtype)))
        else:
            branch_pairs.append(pairs[name])

    if all(lora_a.equal(branch_pairs[0][0]) for lora_a, _ in branch_pairs):
        return branch_pairs[0][0].contiguous(), torch.cat([lora_b for _, lora_b in branch_pairs], dim=0).contiguous()

    total_rank = sum(lora_a.shape[0] for lora_a, _ in branch_pairs)
    down = torch.zeros(total_rank, in_features, dtype=first_a.dtype)
    up = torch.zeros(module.out_features, total_rank, dtype=first_b.dtype)

    col = 0
    row = 0
    for lora_a, lora_b in branch_pairs:
        rank = lora_a.shape[0]
        down[col : col + rank] = lora_a
        up[row : row + lora_b.shape[0], col : col + rank] = lora_b
        col += rank
        row += out_per_branch
    return down.contiguous(), up.contiguous()


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


def _set_converted_pair(
    converted: dict[str, torch.Tensor],
    target_name: str,
    down: torch.Tensor,
    up: torch.Tensor,
    module: SVDQW4A4Linear | AWQW4A16Linear,
) -> None:
    if isinstance(module, SVDQW4A4Linear):
        down = pack_lowrank_weight(down, down=True)
        up = pack_lowrank_weight(up, down=False)
    else:
        splits = _adanorm_splits(target_name)
        down = pad_lora_tensor(down, divisor=16, dim=0)
        up = pad_lora_tensor(reorder_adanorm_lora_up(up, splits=splits), divisor=16, dim=1)
    converted[f"{target_name}.proj_down"] = down
    converted[f"{target_name}.proj_up"] = up


def _adanorm_splits(target_name: str) -> int:
    if target_name.startswith("single_transformer_blocks.") and target_name.endswith(".norm.linear"):
        return 3
    if target_name.startswith("transformer_blocks.") and target_name.endswith((".norm1.linear", ".norm1_context.linear")):
        return 6
    raise ValueError(f"Unsupported AWQ LoRA target {target_name!r}.")


def reorder_adanorm_lora_up(lora_up: torch.Tensor, splits: int) -> torch.Tensor:
    """Match the AdaNorm LoRA up-projection order used by full Nunchaku."""

    channels, rank = lora_up.shape
    if channels % splits != 0:
        raise ValueError(f"AdaNorm LoRA output dimension {channels} is not divisible by {splits}.")
    return lora_up.view(splits, channels // splits, rank).transpose(0, 1).reshape(channels, rank).contiguous()


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
