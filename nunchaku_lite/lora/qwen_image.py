"""Runtime LoRA loading and composition for patched Qwen-Image transformers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
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
    load_lora_state_dict,
    lora_modules,
    normalize_float_tensor,
    pack_lowrank_weight,
    pad_lora_tensor,
    validate_lite_lora_state_dict,
)


def bind_qwen_image_lora_methods(transformer: nn.Module) -> None:
    """Attach runtime LoRA methods to a patched Qwen-Image transformer."""

    transformer._nunchaku_lite_lora_model_name = "Qwen-Image"
    bind_mixin_methods(
        transformer,
        NunchakuQwenImageLoraMixin,
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


def bind_qwen_image_pipeline_lora_methods(pipeline) -> None:
    """Attach Diffusers-compatible runtime LoRA methods to a Qwen-Image pipeline."""

    pipeline._nunchaku_lite_lora_model_name = "Qwen-Image"
    pipeline._nunchaku_lite_lora_component_name = "transformer"
    bind_mixin_methods(
        pipeline,
        NunchakuQwenImagePipelineLoraMixin,
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


class NunchakuQwenImageLoraMixin(NunchakuLoraMixin):
    """Mixin-style method provider for quantized Qwen-Image LoRA runtime."""

    _nunchaku_lite_lora_model_name = "Qwen-Image"

    def _convert_lora_to_lite(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return convert_qwen_image_lora_to_lite(path_or_state_dict, self)


class NunchakuQwenImagePipelineLoraMixin(NunchakuPipelineLoraMixin):
    """Mixin-style method provider for Diffusers-compatible Qwen-Image pipeline APIs."""

    _nunchaku_lite_lora_model_name = "Qwen-Image"

    def _bind_transformer_lora_methods(self, transformer: nn.Module) -> None:
        bind_qwen_image_lora_methods(transformer)


def is_nunchaku_qwen_image_lora(state_dict: dict[str, torch.Tensor]) -> bool:
    """Return whether a state dict already uses Nunchaku-style low-rank keys."""

    return any(
        key.endswith((".proj_down", ".proj_up", ".lora_down", ".lora_up")) for key in state_dict
    ) and not any(("lora_A" in key or "lora_B" in key or "lora_down.weight" in key or "lora_up.weight" in key) for key in state_dict)


def convert_qwen_image_lora_to_lite(
    state_dict_or_path: str | Path | dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert a Qwen-Image LoRA into packed lite SVDQ low-rank tensors."""

    state_dict = load_lora_state_dict(state_dict_or_path)
    if is_nunchaku_qwen_image_lora(state_dict):
        return _normalize_nunchaku_lora_state_dict(state_dict, transformer)
    return _convert_diffusers_lora_state_dict(state_dict, transformer)


def _normalize_nunchaku_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        new_key = _strip_transformer_prefix(key)
        new_key = new_key.replace(".lora_down", ".proj_down")
        new_key = new_key.replace(".lora_up", ".proj_up")
        converted[new_key] = value
    return validate_lite_lora_state_dict(converted, transformer, model_name="Qwen-Image")


def _convert_diffusers_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    diffusers_state = _to_peft_format(state_dict)
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
        target_name = _direct_target_name(base_name, modules)
        if target_name is None:
            continue
        _set_converted_pair(converted, target_name, lora_a.contiguous(), lora_b.contiguous(), modules[target_name])
        handled.add(base_name)

    unsupported = sorted(base for base in pairs if base not in handled and _is_transformer_lora_key(base))
    if unsupported:
        sample = ", ".join(unsupported[:5])
        raise ValueError(f"Unsupported Qwen-Image LoRA target(s) for nunchaku_lite: {sample}")
    return validate_lite_lora_state_dict(converted, transformer, model_name="Qwen-Image")


def _to_peft_format(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors = {}
    for key, value in state_dict.items():
        new_key = _strip_transformer_prefix(key)
        new_key = new_key.replace(".lora_down.weight", LORA_A_SUFFIX)
        new_key = new_key.replace(".lora_up.weight", LORA_B_SUFFIX)
        tensors[new_key] = normalize_float_tensor(value)
    return apply_network_alphas(tensors, extract_network_alphas(tensors))


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


def _direct_target_name(
    base_name: str,
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear],
) -> str | None:
    if base_name in modules:
        return base_name
    return None


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
        down = pad_lora_tensor(down, divisor=16, dim=0)
        up = pad_lora_tensor(up, divisor=16, dim=1)
    converted[f"{target_name}.proj_down"] = down
    converted[f"{target_name}.proj_up"] = up


def _is_transformer_lora_key(base: str) -> bool:
    return base.startswith("transformer_blocks.")
