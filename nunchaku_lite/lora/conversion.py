"""Shared LoRA weight conversion helpers for Nunchaku Lite adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
from .base import (
    fit_lora_tensor,
    iter_lora_pairs,
    lora_modules,
    pack_lowrank_weight,
    pad_lora_tensor,
)
from .peft import peft_lora_pairs


LORA_ERROR_LABEL = "Nunchaku LoRA"


@dataclass(frozen=True)
class FusedProjectionSpec:
    """Describe separate LoRA branches that map into one fused lite projection."""

    target: str
    branches: tuple[str, ...]

    @property
    def suffixes(self) -> tuple[str, ...]:
        return tuple(branch.rsplit(".", 1)[-1] for branch in self.branches)


def strip_transformer_prefix(key: str) -> str:
    for prefix in ("base_model.model.transformer.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def is_nunchaku_lite_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    diffusers_markers: tuple[str, ...] = ("lora_A", "lora_B", "lora_down.weight", "lora_up.weight"),
) -> bool:
    """Return whether a state dict already uses Nunchaku-style low-rank keys."""

    has_lite_key = any(key.endswith((".proj_down", ".proj_up", ".lora_down", ".lora_up")) for key in state_dict)
    has_diffusers_key = any(any(marker in key for marker in diffusers_markers) for key in state_dict)
    return has_lite_key and not has_diffusers_key


def normalize_lite_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer,
    *,
    key_converter: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    """Normalize Nunchaku-format low-rank keys and validate target modules."""

    source = key_converter(state_dict) if key_converter is not None else state_dict
    converted = {}
    for key, value in source.items():
        new_key = strip_transformer_prefix(key)
        new_key = new_key.replace(".lora_down", ".proj_down")
        new_key = new_key.replace(".lora_up", ".proj_up")
        converted[new_key] = value
    return validate_lite_lora_state_dict(converted, transformer)


def convert_diffusers_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer,
    *,
    projection_specs: tuple[FusedProjectionSpec, ...],
    normalize_state_dict: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    map_direct_pair: Callable[
        [str, torch.Tensor, torch.Tensor, dict[str, SVDQW4A4Linear | AWQW4A16Linear]],
        list[tuple[str, torch.Tensor, torch.Tensor]],
    ],
    is_transformer_lora_key: Callable[[str], bool],
    set_pair: Callable[
        [dict[str, torch.Tensor], str, torch.Tensor, torch.Tensor, SVDQW4A4Linear | AWQW4A16Linear],
        None,
    ] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert normalized Diffusers/PEFT LoRA pairs into lite low-rank tensors."""

    if set_pair is None:
        set_pair = set_converted_lora_pair
    diffusers_state = normalize_state_dict(state_dict)
    modules = lora_modules(transformer)
    pairs = peft_lora_pairs(diffusers_state)
    converted: dict[str, torch.Tensor] = {}

    handled: set[str] = set()
    grouped = group_fused_projection_pairs(pairs, projection_specs)
    for target_name, (spec, branches) in grouped.items():
        if target_name not in modules:
            continue
        module = modules[target_name]
        down, up = fuse_projection_branches(branches, module, pairs, spec)
        set_pair(converted, target_name, down, up, module)
        handled.update(branches)

    for base_name, (lora_a, lora_b) in pairs.items():
        if base_name in handled:
            continue
        for target_name, down, up in map_direct_pair(base_name, lora_a, lora_b, modules):
            set_pair(converted, target_name, down, up, modules[target_name])
            handled.add(base_name)

    unsupported = sorted(base for base in pairs if base not in handled and is_transformer_lora_key(base))
    if unsupported:
        sample = ", ".join(unsupported[:5])
        raise ValueError(f"Unsupported {LORA_ERROR_LABEL} target(s) for nunchaku_lite: {sample}")
    return validate_lite_lora_state_dict(converted, transformer)


def group_fused_projection_pairs(
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    specs: tuple[FusedProjectionSpec, ...],
) -> dict[str, tuple[FusedProjectionSpec, list[str]]]:
    groups: dict[str, tuple[FusedProjectionSpec, list[str]]] = {}
    for base in pairs:
        for spec in specs:
            if spec.target in base:
                groups[base] = (spec, [base])
                break
            first_branch = spec.branches[0]
            if first_branch in base:
                target = base.replace(first_branch, spec.target)
                branches = [base.replace(first_branch, branch) for branch in spec.branches]
                groups[target] = (spec, [branch for branch in branches if branch in pairs])
                break
    return groups


def fuse_projection_branches(
    branch_names: list[str],
    module: SVDQW4A4Linear,
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    spec: FusedProjectionSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(branch_names) == 1 and spec.target in branch_names[0]:
        lora_a, lora_b = pairs[branch_names[0]]
        return lora_a.contiguous(), lora_b.contiguous()

    branch_order = spec.suffixes
    by_suffix = {name.rsplit(".", 1)[-1]: name for name in branch_names}
    ordered = [by_suffix[suffix] for suffix in branch_order if suffix in by_suffix]
    if not ordered:
        raise ValueError("No fused projection LoRA branches were provided.")

    first_a, first_b = pairs[ordered[0]]
    in_features = first_a.shape[1]
    out_per_branch = module.out_features // len(branch_order)
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


def set_converted_lora_pair(
    converted: dict[str, torch.Tensor],
    target_name: str,
    down: torch.Tensor,
    up: torch.Tensor,
    module: SVDQW4A4Linear | AWQW4A16Linear,
    *,
    awq_up_transform: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
) -> None:
    if isinstance(module, SVDQW4A4Linear):
        down = pack_lowrank_weight(down, down=True)
        up = pack_lowrank_weight(up, down=False)
    else:
        if awq_up_transform is not None:
            up = awq_up_transform(target_name, up)
        down = pad_lora_tensor(down, divisor=16, dim=0)
        up = pad_lora_tensor(up, divisor=16, dim=1)
    converted[f"{target_name}.proj_down"] = down
    converted[f"{target_name}.proj_up"] = up


def validate_lite_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer,
) -> dict[str, torch.Tensor]:
    modules = lora_modules(transformer)
    if not state_dict:
        raise ValueError(f"LoRA state dict did not contain any supported {LORA_ERROR_LABEL} projection tensors.")

    valid = {}
    for down_key, up_key in iter_lora_pairs(state_dict):
        module_name = down_key[: -len(".proj_down")]
        if module_name not in modules:
            raise ValueError(
                f"LoRA target {module_name!r} does not exist on this patched {LORA_ERROR_LABEL} transformer."
            )
        module = modules[module_name]
        down = fit_lora_tensor(state_dict[down_key], module.in_features, down=True, module_name=module_name)
        up = fit_lora_tensor(state_dict[up_key], module.out_features, down=False, module_name=module_name)
        if down.shape[1] != up.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for {module_name}: proj_down={tuple(down.shape)}, proj_up={tuple(up.shape)}"
            )
        valid[down_key] = down
        valid[up_key] = up
    return valid
