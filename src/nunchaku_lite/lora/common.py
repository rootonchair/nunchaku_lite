"""Shared LoRA weight conversion helpers for Nunchaku Lite adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
from .base import (
    DenseRuntimeLoraLinear,
    fit_lora_tensor,
    iter_lora_pairs,
    looks_like_packed_lowrank,
    lora_modules,
    pack_lowrank_weight,
    pad_lora_tensor,
    svdq_down_to_logical,
    svdq_up_to_logical,
)


LORA_ERROR_LABEL = "Nunchaku LoRA"


@dataclass(frozen=True)
class FusedProjectionSpec:
    """Describe separate LoRA branches that map into one fused lite projection."""

    target: str
    branches: tuple[str, ...]

    @property
    def suffixes(self) -> tuple[str, ...]:
        """Return branch suffixes in the order they should be packed."""

        return tuple(branch.rsplit(".", 1)[-1] for branch in self.branches)


QKV_PROJECTION_SPECS = (
    FusedProjectionSpec(target=".attn.to_qkv", branches=(".attn.to_q", ".attn.to_k", ".attn.to_v")),
    FusedProjectionSpec(
        target=".attn.add_qkv_proj",
        branches=(".attn.add_q_proj", ".attn.add_k_proj", ".attn.add_v_proj"),
    ),
)


def strip_transformer_prefix(key: str) -> str:
    """Remove Diffusers transformer wrapper prefixes from a LoRA tensor key.

    Args:
        key: Original state-dict key, possibly prefixed with ``transformer.`` or
            ``base_model.model.transformer.``.
    """

    for prefix in ("base_model.model.transformer.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def is_nunchaku_lite_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    diffusers_markers: tuple[str, ...] = ("lora_A", "lora_B", "lora_down.weight", "lora_up.weight"),
) -> bool:
    """Return whether a state dict already uses Nunchaku-style low-rank keys.

    The converter accepts both already-packed Nunchaku Lite LoRAs and external
    Diffusers/PEFT LoRAs. This predicate routes the input before conversion:
    lite LoRAs are normalized and validated, while Diffusers/PEFT LoRAs need
    key conversion, fused-projection handling, and tensor packing.

    Args:
        state_dict: Input LoRA tensors keyed by checkpoint/state-dict names.
        diffusers_markers: Key fragments that identify Diffusers/PEFT LoRA
            tensors and should prevent treating mixed inputs as lite format.
    """

    has_lite_key = any(key.endswith((".proj_down", ".proj_up", ".lora_down", ".lora_up")) for key in state_dict)
    has_diffusers_key = any(any(marker in key for marker in diffusers_markers) for key in state_dict)
    return has_lite_key and not has_diffusers_key


def normalize_nunchaku_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer,
) -> dict[str, torch.Tensor]:
    """Normalize standard Nunchaku-format low-rank keys and validate them.

    Args:
        state_dict: LoRA tensors with keys ending in ``.proj_down/.proj_up`` or
            older ``.lora_down/.lora_up`` names.
        transformer: Patched transformer whose quantized linear modules define
            valid target names and expected input/output feature sizes.
    """

    return normalize_nunchaku_lora_keys_and_validate(state_dict, transformer)


def normalize_nunchaku_lora_keys_and_validate(
    state_dict: dict[str, torch.Tensor],
    transformer,
) -> dict[str, torch.Tensor]:
    """Canonicalize Nunchaku-format LoRA keys, then validate tensor pairs against a model.

    Args:
        state_dict: State dict already using Nunchaku-format target names.
        transformer: Patched transformer used to validate module names and
            coerce tensor orientation/shape for each target module.
    """

    converted = {}
    for key, value in state_dict.items():
        new_key = strip_transformer_prefix(key)
        new_key = new_key.replace(".lora_down", ".proj_down")
        new_key = new_key.replace(".lora_up", ".proj_up")
        converted[new_key] = value
    return validate_nunchaku_lora_state_dict(converted, transformer)


def group_fused_projection_pairs(
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    specs: tuple[FusedProjectionSpec, ...],
) -> dict[str, tuple[FusedProjectionSpec, list[str]]]:
    """Group PEFT LoRA pairs that should be merged into fused projections.

    Args:
        pairs: Mapping from PEFT base names to ``(lora_A, lora_B)`` tensors.
        specs: Fused projection descriptions, such as separate q/k/v branches
            that map to one packed qkv target.
    """

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
    """Build one low-rank pair for a fused projection from branch LoRA pairs.

    Args:
        branch_names: PEFT base names present for this fused target.
        module: Quantized target projection; its output size determines branch
            layout when q/k/v ranks differ.
        pairs: Mapping from PEFT base names to ``(lora_A, lora_B)`` tensors.
        spec: Fused projection layout describing branch order and target name.
    """

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


def set_standard_converted_lora_pair(
    converted: dict[str, torch.Tensor],
    target_name: str,
    down: torch.Tensor,
    up: torch.Tensor,
    module: SVDQW4A4Linear | AWQW4A16Linear | DenseRuntimeLoraLinear,
) -> None:
    """Store a converted low-rank pair using the packing expected by a module.

    Args:
        converted: Output state dict being assembled.
        target_name: Patched module name without the ``.proj_down/.proj_up``
            suffix.
        down: Logical LoRA down tensor in rank-by-input layout.
        up: Logical LoRA up tensor in output-by-rank layout.
        module: Target runtime LoRA module that determines whether tensors are
            packed for SVDQ W4A4, padded for AWQ W4A16, or kept dense.
    """

    if isinstance(module, SVDQW4A4Linear):
        down = pack_lowrank_weight(down, down=True)
        up = pack_lowrank_weight(up, down=False)
    elif isinstance(module, AWQW4A16Linear):
        down = pad_lora_tensor(down, divisor=16, dim=0)
        up = pad_lora_tensor(up, divisor=16, dim=1)
    else:
        down = down.contiguous()
        up = up.contiguous()
    converted[f"{target_name}.proj_down"] = down
    converted[f"{target_name}.proj_up"] = up


def validate_nunchaku_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer,
) -> dict[str, torch.Tensor]:
    """Validate Nunchaku-format LoRA tensor pairs and coerce them to module dimensions.

    Args:
        state_dict: Nunchaku-format LoRA state dict with ``.proj_down/.proj_up`` keys.
        transformer: Patched transformer containing target SVDQ/AWQ/dense
            runtime LoRA modules.
    """

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
        if isinstance(module, SVDQW4A4Linear):
            down_logical = svdq_down_to_logical(state_dict[down_key], module, module_name)
            up_logical = svdq_up_to_logical(state_dict[up_key], module, module_name)
            if down_logical.shape[0] != up_logical.shape[1]:
                raise ValueError(
                    f"LoRA rank mismatch for {module_name}: "
                    f"proj_down={tuple(down_logical.shape)}, proj_up={tuple(up_logical.shape)}"
                )
            if looks_like_packed_lowrank(state_dict[down_key], module.in_features) or looks_like_packed_lowrank(
                state_dict[up_key], module.out_features
            ):
                valid[down_key] = pack_lowrank_weight(down_logical, down=True)
                valid[up_key] = pack_lowrank_weight(up_logical, down=False)
            else:
                valid[down_key] = fit_lora_tensor(
                    state_dict[down_key], module.in_features, down=True, module_name=module_name
                )
                valid[up_key] = fit_lora_tensor(
                    state_dict[up_key], module.out_features, down=False, module_name=module_name
                )
            continue
        down = fit_lora_tensor(state_dict[down_key], module.in_features, down=True, module_name=module_name)
        up = fit_lora_tensor(state_dict[up_key], module.out_features, down=False, module_name=module_name)
        if down.shape[1] != up.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for {module_name}: proj_down={tuple(down.shape)}, proj_up={tuple(up.shape)}"
            )
        valid[down_key] = down
        valid[up_key] = up
    return valid
