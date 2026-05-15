"""PEFT-style LoRA key and pair helpers."""

from __future__ import annotations

import torch


LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def normalize_float_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return a floating LoRA tensor in a supported runtime dtype.

    Args:
        value: Tensor loaded from a LoRA checkpoint. Non-floating tensors are
            converted to bfloat16 so later conversion code can operate on them.
    """

    if value.dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
        return value
    return value.to(torch.bfloat16)


def extract_network_alphas(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract PEFT/Kohya network alpha entries from a LoRA state dict.

    Args:
        state_dict: LoRA state dict that may contain keys ending in ``.alpha``.
    """

    return {key: value for key, value in state_dict.items() if key.endswith(".alpha")}


def apply_network_alphas(
    state_dict: dict[str, torch.Tensor],
    alphas: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply LoRA network alpha scaling to matching PEFT down tensors.

    Args:
        state_dict: PEFT-style LoRA tensors keyed by ``.lora_A.weight`` and
            ``.lora_B.weight`` names. Alpha entries are removed from the result.
        alphas: Mapping of ``<base>.alpha`` keys to scalar alpha values.
    """

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


def peft_lora_pairs(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Collect PEFT ``lora_A/lora_B`` tensors into paired base-name entries.

    Args:
        state_dict: Normalized PEFT-style LoRA state dict.
    """

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
