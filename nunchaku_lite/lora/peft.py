"""PEFT-style LoRA key and pair helpers."""

from __future__ import annotations

import torch


LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def normalize_float_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
        return value
    return value.to(torch.bfloat16)


def extract_network_alphas(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if key.endswith(".alpha")}


def apply_network_alphas(
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


def peft_lora_pairs(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
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
