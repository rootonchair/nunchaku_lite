"""Generic runtime-manifest adapter for Nunchaku Lite checkpoints."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..core import PatchOptions, register_adapter
from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..manifest import RuntimeManifest, RuntimeManifestTarget, parse_runtime_manifest
from .common import SVDQPatchContext, finalize_svdq_checkpoint, prepare_transformer_dtype


class ManifestAdapter:
    """Generic adapter driven by ``quantization_config.runtime_manifest``."""

    target = "manifest"

    def matches(self, transformer: torch.nn.Module) -> bool:
        """The manifest adapter is selected from checkpoint metadata, not model type."""

        del transformer
        return False

    def patch(
        self,
        transformer: torch.nn.Module,
        checkpoint_state: dict[str, torch.Tensor],
        quantization_config: dict[str, Any],
        options: PatchOptions,
    ) -> dict[str, torch.Tensor]:
        """Patch declared manifest targets and return the checkpoint state."""

        manifest = parse_runtime_manifest(quantization_config)
        if manifest is None:
            raise ValueError("target='manifest' requires quantization_config.runtime_manifest metadata.")

        context = _manifest_context(transformer, manifest, options)
        prepare_transformer_dtype(transformer, context)
        _apply_structural_patches(transformer, manifest)
        for target in manifest.targets:
            _replace_target(transformer, target, context)

        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        transformer._nunchaku_lite_manifest_patched = True
        return checkpoint_state


class SplitLinearInput(nn.Module):
    """Linear replacement that splits input features across child linears."""

    def __init__(self, linears: list[nn.Linear], in_features_list: list[int]) -> None:
        super().__init__()
        self.linears = nn.ModuleList(linears)
        self.in_features_list = list(in_features_list)
        self.in_features = sum(self.in_features_list)
        self.out_features = self.linears[0].out_features

    @classmethod
    def from_linear(cls, linear: nn.Linear, splits: list[int]) -> "SplitLinearInput":
        splits = _complete_splits(linear.in_features, splits, "split_linear_input")
        linears = []
        start = 0
        for index, split in enumerate(splits):
            child = nn.Linear(
                split,
                linear.out_features,
                bias=linear.bias is not None and index == len(splits) - 1,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
            if not linear.weight.is_meta:
                child.weight.data.copy_(linear.weight[:, start : start + split])
                if child.bias is not None and linear.bias is not None:
                    child.bias.data.copy_(linear.bias)
            linears.append(child)
            start += split
        return cls(linears, splits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.split(self.in_features_list, dim=-1)
        return sum(linear(chunk.contiguous()) for linear, chunk in zip(self.linears, chunks, strict=True))


class SplitLinearOutput(nn.Module):
    """Linear replacement that splits output features across child linears."""

    def __init__(self, linears: list[nn.Linear], out_features_list: list[int]) -> None:
        super().__init__()
        self.linears = nn.ModuleList(linears)
        self.out_features_list = list(out_features_list)
        self.in_features = self.linears[0].in_features
        self.out_features = sum(self.out_features_list)

    @classmethod
    def from_linear(cls, linear: nn.Linear, splits: list[int]) -> "SplitLinearOutput":
        splits = _complete_splits(linear.out_features, splits, "split_linear_output")
        linears = []
        start = 0
        for split in splits:
            child = nn.Linear(
                linear.in_features,
                split,
                bias=linear.bias is not None,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
            if not linear.weight.is_meta:
                child.weight.data.copy_(linear.weight[start : start + split])
                if child.bias is not None and linear.bias is not None:
                    child.bias.data.copy_(linear.bias[start : start + split])
            linears.append(child)
            start += split
        return cls(linears, splits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([linear(x) for linear in self.linears], dim=-1)


def _manifest_context(transformer: nn.Module, manifest: RuntimeManifest, options: PatchOptions) -> SVDQPatchContext:
    parameter = next(transformer.parameters(), None)
    torch_dtype = options.torch_dtype or (parameter.dtype if parameter is not None else torch.bfloat16)
    return SVDQPatchContext(
        precision=manifest.runtime_precision or options.precision,
        rank=manifest.rank,
        torch_dtype=torch_dtype,
        requested_torch_dtype=options.torch_dtype,
    )


def _apply_structural_patches(transformer: nn.Module, manifest: RuntimeManifest) -> None:
    for patch in manifest.structural_patches:
        patch_type = patch["type"]
        for module_name in _matched_module_names(transformer, patch["module"]):
            module = transformer.get_submodule(module_name)
            if isinstance(module, (SplitLinearInput, SplitLinearOutput)):
                continue
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"{patch_type} expected nn.Linear at {module_name!r}, got {module.__class__.__name__}."
                )
            splits = _resolve_splits(module, patch["args"].get("splits", []), input_split=patch_type == "split_linear_input")
            if patch_type == "split_linear_input":
                replacement = SplitLinearInput.from_linear(module, splits)
            elif patch_type == "split_linear_output":
                replacement = SplitLinearOutput.from_linear(module, splits)
            else:
                raise ValueError(f"Unsupported runtime_manifest structural patch type {patch_type!r}.")
            _set_submodule(transformer, module_name, replacement)


def _replace_target(transformer: nn.Module, target: RuntimeManifestTarget, context: SVDQPatchContext) -> None:
    for source_module in target.source_modules:
        try:
            transformer.get_submodule(source_module)
        except AttributeError as exc:
            raise ValueError(
                f"runtime_manifest target {target.checkpoint_prefix!r} source module {source_module!r} "
                "does not exist after structural patches."
            ) from exc

    try:
        module = transformer.get_submodule(target.checkpoint_prefix)
    except AttributeError as exc:
        raise ValueError(
            "runtime_manifest target checkpoint_prefix "
            f"{target.checkpoint_prefix!r} does not exist after structural patches."
        ) from exc

    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if not isinstance(in_features, int) or not isinstance(out_features, int):
        raise TypeError(
            f"runtime_manifest target {target.checkpoint_prefix!r} must expose integer in_features/out_features."
        )

    if target.nunchaku_op == "svdq_w4a4":
        _validate_svdq_group_size(target)
        replacement = SVDQW4A4Linear(
            in_features=in_features,
            out_features=out_features,
            rank=target.rank,
            bias=target.has_bias,
            precision=target.runtime_precision,
            torch_dtype=context.torch_dtype,
            device=_module_device(module),
        )
    elif target.nunchaku_op in {"awq_w4a16", "adanorm_awq_w4a16"}:
        if target.precision != "int4":
            raise ValueError(f"{target.nunchaku_op} target {target.checkpoint_prefix!r} requires precision='int4'.")
        replacement = AWQW4A16Linear(
            in_features=in_features,
            out_features=out_features,
            bias=target.has_bias,
            group_size=target.group_size,
            torch_dtype=context.torch_dtype,
            device=_module_device(module),
        )
    else:
        raise ValueError(f"Unsupported runtime_manifest nunchaku_op {target.nunchaku_op!r}.")

    _set_submodule(transformer, target.checkpoint_prefix, replacement)


def _validate_svdq_group_size(target: RuntimeManifestTarget) -> None:
    expected = 16 if target.precision == "fp4" else 64
    if target.group_size != expected:
        raise ValueError(
            f"svdq_w4a4 target {target.checkpoint_prefix!r} precision={target.precision!r} "
            f"requires group_size={expected}, got {target.group_size}."
        )


def _resolve_splits(linear: nn.Linear, splits: list[Any], *, input_split: bool) -> list[int]:
    resolved = []
    for split in splits:
        if split == "out_features":
            resolved.append(linear.out_features)
        elif split == "in_features":
            resolved.append(linear.in_features)
        elif isinstance(split, int):
            resolved.append(split)
        else:
            raise ValueError(f"Unsupported runtime_manifest split value {split!r}.")
    total = linear.in_features if input_split else linear.out_features
    return _complete_splits(total, resolved, "split_linear_input" if input_split else "split_linear_output")


def _complete_splits(total: int, splits: list[int], patch_type: str) -> list[int]:
    remaining = total - sum(splits)
    if remaining > 0:
        splits = [*splits, remaining]
    splits = [split for split in splits if split > 0]
    if len(splits) < 2:
        raise ValueError(f"{patch_type} requires at least two positive splits.")
    if sum(splits) != total:
        raise ValueError(f"{patch_type} splits must sum to the module feature dimension.")
    return splits


def _matched_module_names(transformer: nn.Module, pattern: str) -> list[str]:
    modules = dict(transformer.named_modules())
    matches = [name for name in modules if name and _path_matches(pattern, name)]
    if not matches:
        raise ValueError(f"runtime_manifest structural patch pattern {pattern!r} matched no modules.")
    return matches


def _path_matches(pattern: str, path: str) -> bool:
    pattern_parts = pattern.split(".")
    path_parts = path.split(".")
    if len(pattern_parts) != len(path_parts):
        return False
    return all(
        pattern_part == "*" or pattern_part == path_part
        for pattern_part, path_part in zip(pattern_parts, path_parts, strict=True)
    )


def _set_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(recurse=False), None)
    if parameter is not None:
        return parameter.device
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    return torch.device("cpu")


register_adapter(ManifestAdapter())
