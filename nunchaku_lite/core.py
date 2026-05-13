"""Adapter registry and public transformer patching entry points."""

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from .utils import get_precision, load_state_dict_in_safetensors


@dataclass
class PatchOptions:
    """Normalized options passed from :func:`patch_transformer` into adapters.

    Attributes:
        precision: Native quantization precision name. Public ``"fp4"`` is
            normalized to ``"nvfp4"`` before adapters see it.
        torch_dtype: Optional dtype requested for the transformer before
            quantized modules are installed.
        device: Optional target device used after the checkpoint is loaded.
        strict: Whether checkpoint loading should require an exact state dict
            match.
        adapter_options: Adapter-specific options, such as rank overrides or
            model-family toggles.
    """

    precision: str
    torch_dtype: torch.dtype | None = None
    device: str | torch.device | None = None
    strict: bool = True
    adapter_options: dict[str, Any] = field(default_factory=dict)


class TransformerAdapter(Protocol):
    """Protocol implemented by model-specific in-place transformer adapters.

    Adapters own all model-topology details. The core registry only selects an
    adapter, loads checkpoint tensors, and delegates the module rewrite.
    """

    target: str

    def matches(self, transformer: torch.nn.Module) -> bool:
        """Return whether this adapter can patch ``transformer``.

        Args:
            transformer: Candidate Diffusers transformer module.

        Returns:
            ``True`` when this adapter recognizes the transformer class and
            module path.
        """
        ...

    def patch(
        self,
        transformer: torch.nn.Module,
        checkpoint_state: dict[str, torch.Tensor],
        quantization_config: dict[str, Any],
        options: PatchOptions,
    ) -> dict[str, torch.Tensor] | None:
        """Rewrite ``transformer`` in place for a Nunchaku Lite checkpoint.

        Args:
            transformer: Transformer module to mutate.
            checkpoint_state: Tensors loaded from the safetensors checkpoint.
            quantization_config: Parsed quantization metadata from the
                checkpoint, or an empty dict when metadata is absent.
            options: Normalized patch options supplied by the public API.

        Returns:
            A checkpoint state dict to load after patching. Returning ``None``
            keeps ``checkpoint_state`` unchanged.
        """
        ...


_ADAPTERS: dict[str, TransformerAdapter] = {}
_BUILTINS_LOADED = False


def register_adapter(adapter: TransformerAdapter) -> None:
    """Register a transformer adapter under its ``target`` name.

    Args:
        adapter: Adapter instance implementing :class:`TransformerAdapter`.

    Raises:
        ValueError: If ``adapter.target`` is missing or empty.

    Returns:
        None.
    """

    if not getattr(adapter, "target", None):
        raise ValueError("Adapter must define a non-empty 'target'.")
    _ADAPTERS[adapter.target] = adapter


def list_adapters() -> list[str]:
    """Return the sorted list of registered adapter target names.

    Built-in adapters are imported before the list is returned so callers see
    the default Flux, Flux2, and Z-Image registrations.

    Returns:
        Sorted adapter target names.
    """

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
    """Patch a Diffusers transformer in place with a Nunchaku Lite checkpoint.

    ``checkpoint`` may be a local safetensors file or a Hugging Face path of
    the form ``org/repo/path/to/file.safetensors``. Adapter selection defaults
    to ``"auto"`` and succeeds when exactly one registered adapter matches the
    transformer class.

    Args:
        transformer: Diffusers transformer module to patch in place.
        checkpoint: Local safetensors path or Hugging Face checkpoint path.
        target: Adapter target name, or ``"auto"`` for adapter detection.
        precision: ``"auto"``, ``"int4"``, or ``"fp4"``. ``"fp4"`` maps to
            the native ``"nvfp4"`` kernel path.
        torch_dtype: Optional dtype to move the transformer to before patching.
        device: Optional device to move the patched transformer to.
        strict: Whether ``load_state_dict`` should enforce exact key matching.
        adapter_options: Optional adapter-specific patch settings.

    Returns:
        The same ``transformer`` object after mutation and checkpoint loading.

    Raises:
        ValueError: If the transformer is already patched for a different
            target or no adapter can be selected.
    """

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
    """Decode a JSON object from safetensors metadata.

    Args:
        metadata: Metadata dictionary returned by safetensors, or ``None``.
        key: Metadata key to decode.

    Returns:
        Parsed JSON object. Missing metadata returns an empty dict.

    Raises:
        ValueError: If the metadata value is not valid JSON or does not decode
            to an object.
    """

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
    """Resolve public precision options to native kernel precision names.

    Args:
        precision: Public precision string accepted by :func:`patch_transformer`.
        device: Device used for auto precision selection.
        checkpoint: Checkpoint path used as a fallback precision hint.

    Returns:
        ``"int4"`` or native ``"nvfp4"``.
    """

    selected = get_precision(precision=precision, device=device or "cuda", pretrained_model_name_or_path=checkpoint)
    if selected == "fp4":
        return "nvfp4"
    return selected


def _select_adapter(transformer: torch.nn.Module, target: str) -> TransformerAdapter:
    """Choose the requested adapter or auto-detect one from the registry.

    Args:
        transformer: Transformer module being patched.
        target: Explicit adapter name or ``"auto"``.

    Returns:
        Selected adapter instance.

    Raises:
        ValueError: If an explicit target is unknown, no auto match exists, or
            more than one adapter matches.
    """

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
    """Import built-in adapters once so their registration side effects run.

    The adapter modules call :func:`register_adapter` at import time. Delaying
    these imports keeps package import lightweight while still making the
    built-ins available before registry operations.

    Returns:
        None.
    """

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    importlib.import_module("nunchaku_lite.adapters.flux")
    importlib.import_module("nunchaku_lite.adapters.flux2")
    importlib.import_module("nunchaku_lite.adapters.z_image")
    _BUILTINS_LOADED = True
