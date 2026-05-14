"""General checkpoint, tensor-shape, and precision helpers for Nunchaku Lite."""

import json
from pathlib import Path
from typing import Any

import safetensors
import torch
from huggingface_hub import hf_hub_download


def ceil_divide(x: int, divisor: int) -> int:
    """Return integer ceiling division.

    Args:
        x: Dividend.
        divisor: Divisor.

    Returns:
        Smallest integer greater than or equal to ``x / divisor``.
    """

    return (x + divisor - 1) // divisor


def pad_tensor(tensor: torch.Tensor | None, multiples: int, dim: int, fill: Any = 0) -> torch.Tensor | None:
    """Pad a tensor along one dimension to a requested multiple.

    Args:
        tensor: Tensor to pad. ``None`` is accepted and returned unchanged.
        multiples: Required multiple for the selected dimension.
        dim: Dimension to pad.
        fill: Fill value for padded elements.

    Returns:
        The original tensor if no padding is required, otherwise a new padded
        tensor. Returns ``None`` when ``tensor`` is ``None``.
    """

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
    """Resolve a local path or download a Hugging Face checkpoint path.

    Args:
        path: Local path, or a Hugging Face path formatted as
            ``org/repo/path/to/file``.
        repo_type: Hugging Face Hub repository type passed to
            :func:`hf_hub_download`.

    Returns:
        Local filesystem path to the requested file.

    Raises:
        ValueError: If a non-local path does not contain enough components to
            infer ``repo_id`` and filename.
    """

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
    """Load tensors from a safetensors checkpoint with optional prefix filtering.

    Args:
        path: Local or Hugging Face safetensors checkpoint path.
        device: Device argument passed to ``safetensors.safe_open``.
        filter_prefix: Optional key prefix to keep. The prefix is removed from
            returned keys.
        return_metadata: Whether to return safetensors metadata with tensors.

    Returns:
        State dict when ``return_metadata`` is false. Otherwise returns
        ``(state_dict, metadata)``.
    """

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
    """Resolve public precision selection.

    Args:
        precision: ``"auto"``, ``"int4"``, or ``"fp4"``.
        device: Device used to inspect CUDA capability for ``"auto"``.
        pretrained_model_name_or_path: Optional checkpoint path used as a name
            hint when hardware alone does not imply fp4.

    Returns:
        Public precision name, either ``"int4"`` or ``"fp4"``.

    Raises:
        ValueError: If ``precision`` is not one of the supported values.
    """

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
    """Normalize scale-related checkpoint keys expected by SVDQ modules.

    Args:
        transformer_from_config: Patched transformer whose state dict defines
            the expected SVDQ key set.
        state_dict_from_checkpoint: Mutable checkpoint state dict to update.

    Returns:
        ``None``. The checkpoint state is modified in place.
    """

    state_dict = transformer_from_config.state_dict()
    for key in state_dict:
        if key not in state_dict_from_checkpoint:
            if ".wcscales" not in key:
                continue
            scale = state_dict[key]
            state_dict_from_checkpoint[key] = torch.ones(scale.shape, dtype=scale.dtype, device="cpu")

    from .models.linear import SVDQW4A4Linear

    for name, module in transformer_from_config.named_modules():
        if isinstance(module, SVDQW4A4Linear) and module.wtscale is not None:
            module.wtscale = state_dict_from_checkpoint.pop(f"{name}.wtscale", 1.0)


def convert_fp16(transformer_from_config: torch.nn.Module, state_dict_from_checkpoint: dict[str, torch.Tensor]) -> None:
    """Convert bf16 checkpoint tensors to fp16 where the model expects fp16.

    Args:
        transformer_from_config: Patched transformer defining target dtypes.
        state_dict_from_checkpoint: Mutable checkpoint state dict to update.

    Raises:
        ValueError: If a dtype mismatch is not the supported bf16-to-fp16
            conversion.

    Returns:
        None.
    """

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
    """Parse optional JSON ``config`` metadata from a safetensors file.

    Args:
        metadata: Safetensors metadata dictionary, or ``None``.

    Returns:
        Parsed config object, or an empty dict when absent.

    Raises:
        ValueError: If the config value does not decode to a JSON object.
    """

    if not metadata or "config" not in metadata:
        return {}
    parsed = json.loads(metadata["config"])
    if not isinstance(parsed, dict):
        raise ValueError("Checkpoint config metadata must decode to a JSON object.")
    return parsed
