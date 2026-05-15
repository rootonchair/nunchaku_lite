"""Runtime LoRA tensor layout and module discovery helpers."""

from __future__ import annotations

from typing import TypeVar

import torch
from torch import nn

from ...linear import AWQW4A16Linear, DenseRuntimeLoraLinear, SVDQW4A4Linear


ModuleT = TypeVar("ModuleT", bound=nn.Module)


def modules_by_class(transformer: nn.Module, module_cls: type[ModuleT]) -> dict[str, ModuleT]:
    """Return modules of ``module_cls`` keyed by their transformer module names.

    Args:
        transformer: Patched transformer to scan by module name.
        module_cls: Module class to collect.
    """

    return {name: module for name, module in transformer.named_modules() if isinstance(module, module_cls)}


def lora_modules(transformer: nn.Module) -> dict[str, SVDQW4A4Linear | AWQW4A16Linear | DenseRuntimeLoraLinear]:
    """Return every linear module that can receive runtime LoRA tensors.

    Args:
        transformer: Patched transformer to scan by module name.
    """

    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear | DenseRuntimeLoraLinear] = {}
    modules.update(modules_by_class(transformer, SVDQW4A4Linear))
    modules.update(modules_by_class(transformer, AWQW4A16Linear))
    modules.update(modules_by_class(transformer, DenseRuntimeLoraLinear))
    return modules


def iter_lora_pairs(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    """Return matching ``proj_down`` and ``proj_up`` key pairs.

    Args:
        state_dict: Lite LoRA state dict using ``.proj_down/.proj_up`` keys.
    """

    down_keys = sorted(key for key in state_dict if key.endswith(".proj_down"))
    pairs = []
    for down_key in down_keys:
        up_key = f"{down_key[: -len('.proj_down')]}.proj_up"
        if up_key not in state_dict:
            raise ValueError(f"Missing proj_up tensor for {down_key!r}.")
        pairs.append((down_key, up_key))
    return pairs


def fit_lora_tensor(tensor: torch.Tensor, feature_dim: int, *, down: bool, module_name: str) -> torch.Tensor:
    """Coerce a LoRA tensor to output-by-rank or input-by-rank module layout.

    Args:
        tensor: Candidate two-dimensional LoRA tensor.
        feature_dim: Expected input/output feature size for the target module.
        down: Whether the tensor is a down projection; used for error labels.
        module_name: Target module name used in validation errors.
    """

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


def svdq_down_to_logical(tensor: torch.Tensor, module: SVDQW4A4Linear, module_name: str) -> torch.Tensor:
    """Convert an SVDQ down tensor to logical rank-by-input layout.

    Args:
        tensor: Packed or unpacked down tensor.
        module: Target SVDQ module that provides feature dimensions.
        module_name: Target module name used in validation errors.
    """

    if looks_like_packed_lowrank(tensor, feature_dim=module.in_features):
        return unpack_lowrank_weight(tensor, down=True)[:, : module.in_features].contiguous()
    fitted = fit_lora_tensor(tensor, module.in_features, down=True, module_name=module_name)
    return fitted.transpose(0, 1).contiguous()


def svdq_up_to_logical(tensor: torch.Tensor, module: SVDQW4A4Linear, module_name: str) -> torch.Tensor:
    """Convert an SVDQ up tensor to logical output-by-rank layout.

    Args:
        tensor: Packed or unpacked up tensor.
        module: Target SVDQ module that provides feature dimensions.
        module_name: Target module name used in validation errors.
    """

    if looks_like_packed_lowrank(tensor, feature_dim=module.out_features):
        return unpack_lowrank_weight(tensor, down=False)[: module.out_features].contiguous()
    fitted = fit_lora_tensor(tensor, module.out_features, down=False, module_name=module_name)
    return fitted.contiguous()


def looks_like_packed_lowrank(tensor: torch.Tensor, feature_dim: int) -> bool:
    """Return whether a low-rank tensor appears to use Nunchaku packed layout.

    Args:
        tensor: Candidate low-rank tensor.
        feature_dim: Expected input/output feature dimension for the module.
    """

    return tensor.shape[0] >= feature_dim and tensor.shape[0] % 16 == 0 and tensor.shape[1] % 16 == 0


def pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Pack a logical low-rank tensor for Nunchaku W4A4 kernels.

    Args:
        weight: Logical LoRA tensor, rank-by-input for down projections or
            output-by-rank for up projections.
        down: Whether ``weight`` is a down projection.
    """

    if weight.dtype == torch.float32:
        weight = weight.to(torch.bfloat16)
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported LoRA dtype {weight.dtype}; expected float16 or bfloat16.")

    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    weight = pad_lora_tensor(weight, divisor=(frag_n, frag_k), dim=(0, 1))
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
    """Unpack a Nunchaku W4A4 low-rank tensor into logical layout.

    Args:
        weight: Packed low-rank tensor stored on an SVDQ module.
        down: Whether ``weight`` is a down projection.
    """

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


def pad_lora_tensor(
    tensor: torch.Tensor,
    divisor: int | tuple[int, int],
    dim: int | tuple[int, int],
    fill_value: float = 0,
) -> torch.Tensor:
    """Pad selected dimensions of a LoRA tensor to kernel alignment divisors.

    Args:
        tensor: Tensor to pad.
        divisor: Divisor or per-dimension divisors the resulting shape must
            satisfy.
        dim: Dimension or dimensions to pad.
        fill_value: Value used for newly allocated padded entries.
    """

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
