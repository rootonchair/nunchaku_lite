import json
from pathlib import Path
from typing import Any

import safetensors
import torch
from huggingface_hub import hf_hub_download


def ceil_divide(x: int, divisor: int) -> int:
    return (x + divisor - 1) // divisor


def pad_tensor(tensor: torch.Tensor | None, multiples: int, dim: int, fill: Any = 0) -> torch.Tensor | None:
    if multiples <= 1 or tensor is None:
        return tensor
    shape = list(tensor.shape)
    if shape[dim] % multiples == 0:
        return tensor
    shape[dim] = ceil_divide(shape[dim], multiples) * multiples
    result = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    result.fill_(fill)
    result[tuple(slice(0, extent) for extent in tensor.shape)] = tensor
    return result


def fetch_or_download(path: str | Path, repo_type: str = "model") -> Path:
    path = Path(path)
    if path.exists():
        return path

    parts = path.parts
    if len(parts) < 3:
        raise ValueError(f"Path '{path}' is too short to extract repo_id and filename")

    repo_id = "/".join(parts[:2])
    sub_path = Path(*parts[2:])
    subfolder = str(sub_path.parent) if sub_path.parent != Path(".") else None
    return Path(hf_hub_download(repo_id=repo_id, filename=sub_path.name, subfolder=subfolder, repo_type=repo_type))


def load_state_dict_in_safetensors(
    path: str | Path,
    device: str | torch.device = "cpu",
    filter_prefix: str = "",
    return_metadata: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, str]]:
    state_dict = {}
    with safetensors.safe_open(fetch_or_download(path), framework="pt", device=device) as f:
        metadata = f.metadata()
        for key in f.keys():
            if filter_prefix and not key.startswith(filter_prefix):
                continue
            state_dict[key.removeprefix(filter_prefix)] = f.get_tensor(key)
    if return_metadata:
        return state_dict, metadata
    return state_dict


def get_precision(
    precision: str = "auto",
    device: str | torch.device = "cuda",
    pretrained_model_name_or_path: str | Path | None = None,
) -> str:
    if precision not in ("auto", "int4", "fp4"):
        raise ValueError("precision must be one of 'auto', 'int4', or 'fp4'")
    if precision != "auto":
        return precision

    if isinstance(device, str):
        device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return "int4"

    capability = torch.cuda.get_device_capability(0 if device.index is None else device.index)
    sm = capability[0] * 10 + capability[1]
    if sm >= 100:
        return "fp4"

    checkpoint_name = "" if pretrained_model_name_or_path is None else str(pretrained_model_name_or_path)
    if "fp4" in checkpoint_name:
        return "fp4"
    return "int4"


def patch_scale_key(
    transformer_from_config: torch.nn.Module, state_dict_from_checkpoint: dict[str, torch.Tensor]
) -> None:
    state_dict = transformer_from_config.state_dict()
    for key in state_dict:
        if key not in state_dict_from_checkpoint:
            if ".wcscales" not in key:
                continue
            state_dict_from_checkpoint[key] = torch.ones_like(state_dict[key])

    from .models.linear import SVDQW4A4Linear

    for name, module in transformer_from_config.named_modules():
        if isinstance(module, SVDQW4A4Linear) and module.wtscale is not None:
            module.wtscale = state_dict_from_checkpoint.pop(f"{name}.wtscale", 1.0)


def convert_fp16(transformer_from_config: torch.nn.Module, state_dict_from_checkpoint: dict[str, torch.Tensor]) -> None:
    for key, value in transformer_from_config.state_dict().items():
        checkpoint_value = state_dict_from_checkpoint.get(key)
        if checkpoint_value is None or value.dtype == checkpoint_value.dtype:
            continue
        if value.dtype != torch.float16 or checkpoint_value.dtype != torch.bfloat16:
            raise ValueError(
                f"Unexpected dtype difference for key {key}: model={value.dtype}, checkpoint={checkpoint_value.dtype}"
            )
        state_dict_from_checkpoint[key] = torch.nan_to_num(
            checkpoint_value.to(torch.float16), nan=0.0, posinf=65504, neginf=-65504
        )


def parse_config_metadata(metadata: dict[str, str] | None) -> dict[str, Any]:
    if not metadata or "config" not in metadata:
        return {}
    parsed = json.loads(metadata["config"])
    if not isinstance(parsed, dict):
        raise ValueError("Checkpoint config metadata must decode to a JSON object.")
    return parsed
