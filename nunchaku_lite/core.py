import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from .utils import get_precision, load_state_dict_in_safetensors


@dataclass
class PatchOptions:
    precision: str
    torch_dtype: torch.dtype | None = None
    device: str | torch.device | None = None
    strict: bool = True
    adapter_options: dict[str, Any] = field(default_factory=dict)


class TransformerAdapter(Protocol):
    target: str

    def matches(self, transformer: torch.nn.Module) -> bool:
        ...

    def patch(
        self,
        transformer: torch.nn.Module,
        checkpoint_state: dict[str, torch.Tensor],
        quantization_config: dict[str, Any],
        options: PatchOptions,
    ) -> dict[str, torch.Tensor] | None:
        ...


_ADAPTERS: dict[str, TransformerAdapter] = {}
_BUILTINS_LOADED = False


def register_adapter(adapter: TransformerAdapter) -> None:
    if not getattr(adapter, "target", None):
        raise ValueError("Adapter must define a non-empty 'target'.")
    _ADAPTERS[adapter.target] = adapter


def list_adapters() -> list[str]:
    _ensure_builtin_adapters()
    return sorted(_ADAPTERS)


def patch_transformer(
    transformer: torch.nn.Module,
    checkpoint: str | Path,
    *,
    target: str = "auto",
    precision: str = "auto",
    torch_dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
    adapter_options: dict[str, Any] | None = None,
) -> torch.nn.Module:
    _ensure_builtin_adapters()

    existing_target = getattr(transformer, "_nunchaku_lite_target", None)
    if getattr(transformer, "_nunchaku_lite_patched", False):
        if target == "auto" or target == existing_target:
            return transformer
        raise ValueError(f"Transformer is already patched as {existing_target!r}; requested {target!r}.")

    adapter = _select_adapter(transformer, target)
    state_dict, metadata = load_state_dict_in_safetensors(checkpoint, return_metadata=True)
    quantization_config = _parse_json_metadata(metadata, "quantization_config")
    normalized_precision = _normalize_precision(precision, device, checkpoint)
    options = PatchOptions(
        precision=normalized_precision,
        torch_dtype=torch_dtype,
        device=device,
        strict=strict,
        adapter_options=adapter_options or {},
    )

    maybe_state_dict = adapter.patch(transformer, state_dict, quantization_config, options)
    if maybe_state_dict is not None:
        state_dict = maybe_state_dict

    incompatible = transformer.load_state_dict(state_dict, strict=strict)
    if device is not None:
        transformer.to(device)

    transformer._nunchaku_lite_patched = True
    transformer._nunchaku_lite_target = adapter.target
    transformer._nunchaku_lite_quantization_config = quantization_config
    transformer._nunchaku_lite_incompatible_keys = incompatible
    return transformer


def _parse_json_metadata(metadata: dict[str, str] | None, key: str) -> dict[str, Any]:
    if not metadata:
        return {}
    raw = metadata.get(key, "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in checkpoint metadata field {key!r}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Checkpoint metadata field {key!r} must decode to a JSON object.")
    return parsed


def _normalize_precision(precision: str, device: str | torch.device | None, checkpoint: str | Path) -> str:
    selected = get_precision(precision=precision, device=device or "cuda", pretrained_model_name_or_path=checkpoint)
    if selected == "fp4":
        return "nvfp4"
    return selected


def _select_adapter(transformer: torch.nn.Module, target: str) -> TransformerAdapter:
    if target != "auto":
        try:
            return _ADAPTERS[target]
        except KeyError as exc:
            available = ", ".join(list_adapters())
            raise ValueError(f"Unsupported target {target!r}. Available adapters: {available}") from exc

    matches = [adapter for adapter in _ADAPTERS.values() if adapter.matches(transformer)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        cls = transformer.__class__
        available = ", ".join(list_adapters()) or "<none>"
        raise ValueError(
            f"No nunchaku_lite adapter matches {cls.__module__}.{cls.__name__}. Available adapters: {available}"
        )
    names = ", ".join(adapter.target for adapter in matches)
    raise ValueError(f"Multiple nunchaku_lite adapters match this transformer: {names}. Pass target=... explicitly.")


def _ensure_builtin_adapters() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    importlib.import_module("nunchaku_lite.adapters.flux")
    importlib.import_module("nunchaku_lite.adapters.z_image")
    _BUILTINS_LOADED = True
