"""Runtime LoRA loading and composition for patched Flux transformers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import MethodType

import torch
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
from torch import nn

from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
from ..utils import load_state_dict_in_safetensors
from ..adapters.flux import convert_flux_state_dict


LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def bind_flux_lora_methods(transformer: nn.Module) -> None:
    """Attach runtime LoRA methods to a patched Flux transformer."""

    transformer.load_lora = MethodType(_load_lora, transformer)
    transformer.set_lora_strength = MethodType(_set_lora_strength, transformer)
    transformer.reset_lora = MethodType(_reset_lora, transformer)
    transformer._nunchaku_lite_loras = OrderedDict()
    transformer._nunchaku_lite_lora_base_state = None


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

    state_dict = _load_lora_state_dict(state_dict_or_path)
    if is_nunchaku_flux_lora(state_dict):
        return _normalize_nunchaku_lora_state_dict(state_dict, transformer)
    return _convert_diffusers_lora_state_dict(state_dict, transformer)


def _load_lora(
    transformer: nn.Module,
    path_or_state_dict: str | Path | dict[str, torch.Tensor],
    *,
    strength: float = 1.0,
    name: str | None = None,
    replace: bool = False,
) -> str:
    """Load a LoRA into a patched Flux transformer and return its adapter name."""

    _ensure_lora_base_state(transformer)
    if replace:
        transformer._nunchaku_lite_loras.clear()

    lora_name = _resolve_lora_name(transformer, path_or_state_dict, name)
    converted = convert_flux_lora_to_lite(path_or_state_dict, transformer)
    converted = {key: value.detach().cpu() for key, value in converted.items()}
    transformer._nunchaku_lite_loras[lora_name] = {"state_dict": converted, "strength": float(strength)}
    _recompose_loras(transformer)
    return lora_name


def _set_lora_strength(transformer: nn.Module, strength: float = 1.0, name: str | None = None) -> None:
    """Set the strength for one active LoRA and recompose all active adapters."""

    _ensure_lora_runtime(transformer)
    if not transformer._nunchaku_lite_loras:
        return
    if name is None:
        if len(transformer._nunchaku_lite_loras) != 1:
            raise ValueError("Multiple LoRAs are active; pass name=... to set one strength.")
        name = next(iter(transformer._nunchaku_lite_loras))
    try:
        transformer._nunchaku_lite_loras[name]["strength"] = float(strength)
    except KeyError as exc:
        raise ValueError(f"No active LoRA named {name!r}.") from exc
    _recompose_loras(transformer)


def _reset_lora(transformer: nn.Module, name: str | None = None) -> None:
    """Reset all LoRAs or remove one named LoRA from a patched Flux transformer."""

    _ensure_lora_runtime(transformer)
    if name is None:
        transformer._nunchaku_lite_loras.clear()
    else:
        try:
            del transformer._nunchaku_lite_loras[name]
        except KeyError as exc:
            raise ValueError(f"No active LoRA named {name!r}.") from exc
    _recompose_loras(transformer)


def _load_lora_state_dict(path_or_state_dict: str | Path | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(path_or_state_dict, dict):
        return dict(path_or_state_dict)
    return load_state_dict_in_safetensors(path_or_state_dict)


def _ensure_lora_runtime(transformer: nn.Module) -> None:
    if not hasattr(transformer, "_nunchaku_lite_loras"):
        transformer._nunchaku_lite_loras = OrderedDict()
    if not hasattr(transformer, "_nunchaku_lite_lora_base_state"):
        transformer._nunchaku_lite_lora_base_state = None


def _ensure_lora_base_state(transformer: nn.Module) -> None:
    _ensure_lora_runtime(transformer)
    if transformer._nunchaku_lite_lora_base_state is not None:
        return

    base_state = {}
    for name, module in _lora_modules(transformer).items():
        if isinstance(module, SVDQW4A4Linear):
            base_state[f"{name}.proj_down"] = module.proj_down.detach().clone()
            base_state[f"{name}.proj_up"] = module.proj_up.detach().clone()
        else:
            device = module.qweight.device
            dtype = module.wscales.dtype
            base_state[f"{name}.proj_down"] = torch.empty(module.in_features, 0, device=device, dtype=dtype)
            base_state[f"{name}.proj_up"] = torch.empty(module.out_features, 0, device=device, dtype=dtype)
    transformer._nunchaku_lite_lora_base_state = base_state


def _resolve_lora_name(
    transformer: nn.Module,
    path_or_state_dict: str | Path | dict[str, torch.Tensor],
    name: str | None,
) -> str:
    if name is None:
        if isinstance(path_or_state_dict, (str, Path)):
            name = Path(path_or_state_dict).stem
        else:
            name = f"lora_{len(transformer._nunchaku_lite_loras) + 1}"
    if name in transformer._nunchaku_lite_loras:
        raise ValueError(f"A LoRA named {name!r} is already active. Use replace=True or choose another name.")
    return name


def _recompose_loras(transformer: nn.Module) -> None:
    _ensure_lora_base_state(transformer)
    modules = _lora_modules(transformer)

    for name, module in modules.items():
        if isinstance(module, SVDQW4A4Linear):
            base_down = transformer._nunchaku_lite_lora_base_state[f"{name}.proj_down"]
            base_up = transformer._nunchaku_lite_lora_base_state[f"{name}.proj_up"]
            if not transformer._nunchaku_lite_loras:
                down = base_down.to(device=module.proj_down.device, dtype=module.proj_down.dtype)
                up = base_up.to(device=module.proj_up.device, dtype=module.proj_up.dtype)
            else:
                logical_downs = [_svdq_down_to_logical(base_down, module, name)]
                logical_ups = [_svdq_up_to_logical(base_up, module, name)]
                for entry in transformer._nunchaku_lite_loras.values():
                    state_dict = entry["state_dict"]
                    down_key = f"{name}.proj_down"
                    up_key = f"{name}.proj_up"
                    if down_key not in state_dict and up_key not in state_dict:
                        continue
                    if down_key not in state_dict or up_key not in state_dict:
                        raise ValueError(f"Incomplete LoRA tensors for {name}.")
                    down_logical = _svdq_down_to_logical(state_dict[down_key], module, name)
                    up_logical = _svdq_up_to_logical(state_dict[up_key], module, name)
                    if down_logical.shape[0] != up_logical.shape[1]:
                        raise ValueError(
                            f"LoRA rank mismatch for {name}: "
                            f"proj_down={tuple(down_logical.shape)}, proj_up={tuple(up_logical.shape)}"
                        )
                    logical_downs.append(down_logical * float(entry["strength"]))
                    logical_ups.append(up_logical)
                down = pack_lowrank_weight(torch.cat(logical_downs, dim=0), down=True)
                up = pack_lowrank_weight(torch.cat(logical_ups, dim=1), down=False)
                down = down.to(device=module.proj_down.device, dtype=module.proj_down.dtype)
                up = up.to(device=module.proj_up.device, dtype=module.proj_up.dtype)
            module.proj_down = nn.Parameter(down, requires_grad=module.proj_down.requires_grad)
            module.proj_up = nn.Parameter(up, requires_grad=module.proj_up.requires_grad)
            module.rank = down.shape[1]
        else:
            down = transformer._nunchaku_lite_lora_base_state[f"{name}.proj_down"]
            up = transformer._nunchaku_lite_lora_base_state[f"{name}.proj_up"]
            for entry in transformer._nunchaku_lite_loras.values():
                state_dict = entry["state_dict"]
                down_key = f"{name}.proj_down"
                up_key = f"{name}.proj_up"
                if down_key not in state_dict and up_key not in state_dict:
                    continue
                if down_key not in state_dict or up_key not in state_dict:
                    raise ValueError(f"Incomplete LoRA tensors for {name}.")
                lora_down = _fit_lora_tensor(state_dict[down_key], module.in_features, down=True, module_name=name)
                lora_up = _fit_lora_tensor(state_dict[up_key], module.out_features, down=False, module_name=name)
                if lora_down.shape[1] != lora_up.shape[1]:
                    raise ValueError(
                        f"LoRA rank mismatch for {name}: "
                        f"proj_down={tuple(lora_down.shape)}, proj_up={tuple(lora_up.shape)}"
                    )
                lora_down = lora_down.to(device=down.device, dtype=down.dtype) * float(entry["strength"])
                lora_up = lora_up.to(device=up.device, dtype=up.dtype)
                down = torch.cat([down, lora_down], dim=1)
                up = torch.cat([up, lora_up], dim=1)
            down = down.to(device=module.qweight.device, dtype=module.wscales.dtype)
            up = up.to(device=module.qweight.device, dtype=module.wscales.dtype)
            module._nunchaku_lite_lora_down = down
            module._nunchaku_lite_lora_up = up


def _svdq_modules(transformer: nn.Module) -> dict[str, SVDQW4A4Linear]:
    return {name: module for name, module in transformer.named_modules() if isinstance(module, SVDQW4A4Linear)}


def _awq_modules(transformer: nn.Module) -> dict[str, AWQW4A16Linear]:
    return {name: module for name, module in transformer.named_modules() if isinstance(module, AWQW4A16Linear)}


def _lora_modules(transformer: nn.Module) -> dict[str, SVDQW4A4Linear | AWQW4A16Linear]:
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear] = {}
    modules.update(_svdq_modules(transformer))
    modules.update(_awq_modules(transformer))
    return modules


def _normalize_nunchaku_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    converted = convert_flux_state_dict(state_dict)
    return _validate_lite_lora_state_dict(converted, transformer)


def _convert_diffusers_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    diffusers_state = _to_diffusers_format(state_dict)
    modules = _lora_modules(transformer)
    pairs = _diffusers_pairs(diffusers_state)
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
    return _validate_lite_lora_state_dict(converted, transformer)


def _to_diffusers_format(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors = _handle_kohya_lora(state_dict)
    tensors = {key: _normalize_float_tensor(value) for key, value in tensors.items()}
    if any("lora_A" in key or "lora_B" in key for key in tensors):
        tensors = {_strip_transformer_prefix(key): value for key, value in tensors.items()}
        return _apply_network_alphas(tensors, _extract_network_alphas(tensors))

    converted, _network_alphas = FluxLoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
    converted = convert_unet_state_dict_to_peft(converted)
    converted = {_strip_transformer_prefix(key): value for key, value in converted.items()}
    alphas = {_strip_transformer_prefix(key): value for key, value in (_network_alphas or {}).items()}
    return _apply_network_alphas(converted, alphas)


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


def _normalize_float_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
        return value
    return value.to(torch.bfloat16)


def _strip_transformer_prefix(key: str) -> str:
    for prefix in ("base_model.model.transformer.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _extract_network_alphas(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if key.endswith(".alpha")}


def _apply_network_alphas(
    state_dict: dict[str, torch.Tensor],
    alphas: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not alphas:
        return {key: value for key, value in state_dict.items() if not key.endswith(".alpha")}

    converted = {key: value for key, value in state_dict.items() if not key.endswith(".alpha")}
    for alpha_key, alpha in alphas.items():
        base = alpha_key[: -len(".alpha")]
        a_key = f"{base}{LORA_A_SUFFIX}"
        b_key = f"{base}{LORA_B_SUFFIX}"
        if a_key not in converted or b_key not in converted:
            continue
        rank = converted[a_key].shape[0]
        alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
        converted[a_key] = converted[a_key] * (alpha_value / rank)
    return converted


def _diffusers_pairs(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    pairs = {}
    for key, value in state_dict.items():
        if not key.endswith(LORA_A_SUFFIX):
            continue
        base = key[: -len(LORA_A_SUFFIX)]
        b_key = f"{base}{LORA_B_SUFFIX}"
        if b_key not in state_dict:
            raise ValueError(f"Missing LoRA B tensor for {key!r}.")
        pairs[base] = (value, state_dict[b_key])
    return pairs


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
        down = _pad_lora_tensor(down, divisor=16, dim=0)
        up = _pad_lora_tensor(reorder_adanorm_lora_up(up, splits=splits), divisor=16, dim=1)
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


def _validate_lite_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    modules = _lora_modules(transformer)
    if not state_dict:
        raise ValueError("LoRA state dict did not contain any supported Flux SVDQ projection tensors.")

    valid = {}
    for down_key, up_key in _iter_lora_pairs(state_dict):
        module_name = down_key[: -len(".proj_down")]
        if module_name not in modules:
            raise ValueError(f"LoRA target {module_name!r} does not exist on this patched Flux transformer.")
        module = modules[module_name]
        down = _fit_lora_tensor(state_dict[down_key], module.in_features, down=True, module_name=module_name)
        up = _fit_lora_tensor(state_dict[up_key], module.out_features, down=False, module_name=module_name)
        if down.shape[1] != up.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for {module_name}: proj_down={tuple(down.shape)}, proj_up={tuple(up.shape)}"
            )
        valid[down_key] = down
        valid[up_key] = up
    return valid


def _iter_lora_pairs(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    down_keys = sorted(key for key in state_dict if key.endswith(".proj_down"))
    pairs = []
    for down_key in down_keys:
        up_key = f"{down_key[: -len('.proj_down')]}.proj_up"
        if up_key not in state_dict:
            raise ValueError(f"Missing proj_up tensor for {down_key!r}.")
        pairs.append((down_key, up_key))
    return pairs


def _fit_lora_tensor(tensor: torch.Tensor, feature_dim: int, *, down: bool, module_name: str) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"LoRA tensor for {module_name} must be 2D, got shape {tuple(tensor.shape)}.")

    if tensor.shape[0] >= feature_dim:
        return tensor[:feature_dim].contiguous()
    if tensor.shape[1] >= feature_dim:
        return tensor[:, :feature_dim].transpose(0, 1).contiguous()
    kind = "proj_down" if down else "proj_up"
    raise ValueError(
        f"{module_name}.{kind} shape {tuple(tensor.shape)} is incompatible with feature size {feature_dim}."
    )


def _svdq_down_to_logical(tensor: torch.Tensor, module: SVDQW4A4Linear, module_name: str) -> torch.Tensor:
    fitted = _fit_lora_tensor(tensor, module.in_features, down=True, module_name=module_name)
    if _looks_like_packed_lowrank(fitted, feature_dim=module.in_features):
        return unpack_lowrank_weight(fitted, down=True)
    return fitted.transpose(0, 1).contiguous()


def _svdq_up_to_logical(tensor: torch.Tensor, module: SVDQW4A4Linear, module_name: str) -> torch.Tensor:
    fitted = _fit_lora_tensor(tensor, module.out_features, down=False, module_name=module_name)
    if _looks_like_packed_lowrank(fitted, feature_dim=module.out_features):
        return unpack_lowrank_weight(fitted, down=False)
    return fitted.contiguous()


def _looks_like_packed_lowrank(tensor: torch.Tensor, feature_dim: int) -> bool:
    return tensor.shape[0] == feature_dim and feature_dim % 16 == 0 and tensor.shape[1] % 16 == 0


def pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Pack a low-rank weight for Nunchaku W4A4 kernels."""

    if weight.dtype == torch.float32:
        weight = weight.to(torch.bfloat16)
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported LoRA dtype {weight.dtype}; expected float16 or bfloat16.")

    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    weight = _pad_lora_tensor(weight, divisor=(frag_n, frag_k), dim=(0, 1))
    if down:
        rank, channels = weight.shape
        rank_frags, channel_frags = rank // frag_n, channels // frag_k
        weight = weight.view(rank_frags, frag_n, channel_frags, frag_k).permute(2, 0, 1, 3)
        channels_out, rank_out = channels, rank
    else:
        channels, rank = weight.shape
        channel_frags, rank_frags = channels // frag_n, rank // frag_k
        weight = weight.view(channel_frags, frag_n, rank_frags, frag_k).permute(0, 2, 1, 3)
        channels_out, rank_out = channels, rank
    weight = weight.reshape(channel_frags, rank_frags, n_pack_size, num_n_lanes, k_pack_size, num_k_lanes, lane_k)
    weight = weight.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return weight.view(channels_out, rank_out)


def unpack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Unpack a Nunchaku W4A4 low-rank tensor into logical rank-major layout."""

    channels, rank = weight.shape
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported LoRA dtype {weight.dtype}; expected float16 or bfloat16.")
    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    if down:
        rank_frags, channel_frags = rank // frag_n, channels // frag_k
    else:
        channel_frags, rank_frags = channels // frag_n, rank // frag_k
    weight = weight.view(channel_frags, rank_frags, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, lane_k)
    weight = weight.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    weight = weight.view(channel_frags, rank_frags, frag_n, frag_k)
    if down:
        return weight.permute(1, 2, 0, 3).contiguous().view(rank, channels)
    return weight.permute(0, 2, 1, 3).contiguous().view(channels, rank)


def _pad_lora_tensor(
    tensor: torch.Tensor,
    divisor: int | tuple[int, int],
    dim: int | tuple[int, int],
    fill_value: float = 0,
) -> torch.Tensor:
    if isinstance(divisor, int):
        divisor = (divisor,)
    if isinstance(dim, int):
        dim = (dim,)
    shape = list(tensor.shape)
    for current_dim, current_divisor in zip(dim, divisor):
        if current_divisor > 1 and shape[current_dim] % current_divisor != 0:
            shape[current_dim] = ((shape[current_dim] + current_divisor - 1) // current_divisor) * current_divisor
    if shape == list(tensor.shape):
        return tensor.contiguous()
    result = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    result.fill_(fill_value)
    result[tuple(slice(0, extent) for extent in tensor.shape)] = tensor
    return result
