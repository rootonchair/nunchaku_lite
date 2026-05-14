"""Adapter registry and public model and pipeline loading entry points."""

import importlib
import inspect
import json
import os
import warnings
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
    the default Flux, Flux2, Qwen-Image, SDXL, and Z-Image registrations.

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

    return _patch_component(
        transformer,
        checkpoint,
        target=target,
        precision=precision,
        torch_dtype=torch_dtype,
        device=device,
        strict=strict,
        adapter_options=adapter_options,
        assign=False,
    )


def load_nunchaku_pipeline(
    pretrained_model_name_or_path: str | os.PathLike,
    *,
    pipeline_cls: type,
    checkpoint: str | Path,
    target: str = "auto",
    component: str | None = None,
    precision: str = "auto",
    torch_dtype: torch.dtype | dict[str, torch.dtype] | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
    adapter_options: dict[str, Any] | None = None,
    bind_lora: bool = True,
    **pipeline_kwargs,
):
    """Load a Diffusers pipeline with its Nunchaku component installed up front.

    The selected ``transformer`` or ``unet`` is constructed on the meta device,
    patched with the Nunchaku adapter, materialized from ``checkpoint``, and
    then passed into ``pipeline_cls.from_pretrained``. Passing the patched
    component into Diffusers prevents the original dense BF16 component weights
    from being loaded only to be discarded.

    Args:
        pretrained_model_name_or_path: Diffusers pipeline model id or local path.
        pipeline_cls: Diffusers pipeline class to instantiate.
        checkpoint: Local safetensors path or Hugging Face checkpoint path.
        target: Adapter target name, or ``"auto"`` for adapter detection.
        component: Pipeline component to replace. Defaults to ``"transformer"``
            when present, otherwise ``"unet"``.
        precision: ``"auto"``, ``"int4"``, or ``"fp4"``.
        torch_dtype: Optional dtype passed to Diffusers. Dict values follow the
            Diffusers convention; the patched component uses its own entry or
            ``"default"`` when present.
        device: Optional device to move the patched component to after loading.
        strict: Whether checkpoint loading should require exact key matching.
        adapter_options: Optional adapter-specific patch settings.
        bind_lora: Whether to bind supported pipeline-level LoRA APIs.
        **pipeline_kwargs: Additional keyword arguments forwarded to
            ``pipeline_cls.from_pretrained``.

    Returns:
        The loaded Diffusers pipeline with a patched Nunchaku component.
    """

    if not hasattr(pipeline_cls, "from_pretrained"):
        raise TypeError("pipeline_cls must provide a from_pretrained classmethod.")

    pipeline_config = _load_pipeline_config(pipeline_cls, pretrained_model_name_or_path, pipeline_kwargs)
    component_name = _select_pipeline_component(pipeline_cls, pipeline_config, component)
    if component_name in pipeline_kwargs:
        raise ValueError(
            f"pipeline_kwargs already contains {component_name!r}; "
            "load_nunchaku_pipeline creates and injects that component itself."
        )

    component_cls = _resolve_pipeline_component_class(pipeline_config, component_name)
    component_dtype = _resolve_component_torch_dtype(torch_dtype, component_name)
    loaded_component = _load_nunchaku_component_from_config(
        pretrained_model_name_or_path,
        component_name=component_name,
        component_cls=component_cls,
        checkpoint=checkpoint,
        target=target,
        precision=precision,
        torch_dtype=component_dtype,
        device=device,
        strict=strict,
        adapter_options=adapter_options,
        pipeline_kwargs=pipeline_kwargs,
    )

    pipe = pipeline_cls.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        **pipeline_kwargs,
        **{component_name: loaded_component},
    )
    if bind_lora:
        _bind_pipeline_runtime_methods(pipe, getattr(loaded_component, "_nunchaku_lite_target", target))
    return pipe


def _patch_component(
    transformer: torch.nn.Module,
    checkpoint: str | Path,
    *,
    target: str,
    precision: str,
    torch_dtype: torch.dtype | None,
    device: str | torch.device | None,
    strict: bool,
    adapter_options: dict[str, Any] | None,
    assign: bool,
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
    _validate_quantization_compatibility(quantization_config, normalized_precision, device, checkpoint)
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

    if assign:
        _coerce_state_dict_for_assign(transformer, state_dict)
    incompatible = transformer.load_state_dict(state_dict, strict=strict, assign=assign)
    _materialize_known_meta_tensors(transformer)
    if device is not None:
        transformer.to(device)

    transformer._nunchaku_lite_patched = True
    transformer._nunchaku_lite_target = adapter.target
    transformer._nunchaku_lite_quantization_config = quantization_config
    transformer._nunchaku_lite_incompatible_keys = incompatible
    return transformer


_CONFIG_LOADING_KWARGS = {
    "cache_dir",
    "force_download",
    "proxies",
    "local_files_only",
    "token",
    "revision",
    "user_agent",
    "mirror",
}


def _load_pipeline_config(
    pipeline_cls: type,
    pretrained_model_name_or_path: str | os.PathLike,
    pipeline_kwargs: dict[str, Any],
) -> dict[str, Any]:
    if not hasattr(pipeline_cls, "load_config"):
        raise TypeError("pipeline_cls must provide a load_config classmethod.")
    config_kwargs = _extract_config_loading_kwargs(pipeline_kwargs)
    config = pipeline_cls.load_config(pretrained_model_name_or_path, **config_kwargs)
    if isinstance(config, tuple):
        config = config[0]
    if not isinstance(config, dict):
        raise TypeError(f"{pipeline_cls.__name__}.load_config(...) must return a configuration dictionary.")
    return config


def _select_pipeline_component(
    pipeline_cls: type,
    pipeline_config: dict[str, Any],
    component: str | None,
) -> str:
    if component is not None:
        if component not in {"transformer", "unet"}:
            raise ValueError("component must be 'transformer', 'unet', or None.")
        return component

    expected_modules = _pipeline_expected_modules(pipeline_cls)
    for candidate in ("transformer", "unet"):
        if candidate in expected_modules or candidate in pipeline_config:
            return candidate

    raise ValueError(
        "Could not auto-select a Nunchaku pipeline component. "
        "Pass component='transformer' or component='unet'."
    )


def _pipeline_expected_modules(pipeline_cls: type) -> set[str]:
    if hasattr(pipeline_cls, "_get_signature_keys"):
        expected_modules, _optional_kwargs = pipeline_cls._get_signature_keys(pipeline_cls)
        return set(expected_modules)

    parameters = inspect.signature(pipeline_cls.__init__).parameters
    return {name for name, value in parameters.items() if name != "self" and value.default == inspect._empty}


def _resolve_pipeline_component_class(pipeline_config: dict[str, Any], component_name: str) -> type[torch.nn.Module]:
    component_spec = pipeline_config.get(component_name)
    if not isinstance(component_spec, (list, tuple)) or len(component_spec) < 2:
        raise ValueError(
            f"Pipeline config does not contain a loadable class entry for component {component_name!r}."
        )

    library_name, class_name = component_spec[:2]
    if library_name is None or class_name is None:
        raise ValueError(f"Pipeline component {component_name!r} is optional or missing in the model config.")

    try:
        from diffusers.pipelines.pipeline_loading_utils import simple_get_class_obj
    except ImportError as exc:
        raise ImportError("load_nunchaku_pipeline requires diffusers pipeline loading utilities.") from exc

    return simple_get_class_obj(library_name, class_name)


def _resolve_component_torch_dtype(
    torch_dtype: torch.dtype | dict[str, torch.dtype] | None,
    component_name: str,
) -> torch.dtype | None:
    if not isinstance(torch_dtype, dict):
        return torch_dtype
    return torch_dtype.get(component_name, torch_dtype.get("default"))


def _load_nunchaku_component_from_config(
    pretrained_model_name_or_path: str | os.PathLike,
    *,
    component_name: str,
    component_cls: type,
    checkpoint: str | Path,
    target: str,
    precision: str,
    torch_dtype: torch.dtype | None,
    device: str | torch.device | None,
    strict: bool,
    adapter_options: dict[str, Any] | None,
    pipeline_kwargs: dict[str, Any],
) -> torch.nn.Module:
    if not hasattr(component_cls, "load_config") or not hasattr(component_cls, "from_config"):
        raise TypeError(f"{component_cls.__module__}.{component_cls.__name__} must support load_config/from_config.")

    component_config_kwargs = _extract_config_loading_kwargs(pipeline_kwargs)
    component_config_kwargs["subfolder"] = component_name
    component_config = component_cls.load_config(pretrained_model_name_or_path, **component_config_kwargs)
    if isinstance(component_config, tuple):
        component_config = component_config[0]

    with torch.device("meta"):
        component = component_cls.from_config(component_config)

    return _patch_component(
        component,
        checkpoint,
        target=target,
        precision=precision,
        torch_dtype=torch_dtype,
        device=device,
        strict=strict,
        adapter_options=adapter_options,
        assign=True,
    )


def _extract_config_loading_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key in _CONFIG_LOADING_KWARGS}


def _bind_pipeline_runtime_methods(pipeline: Any, target: str) -> None:
    if target == "flux":
        from .lora import bind_flux_pipeline_lora_methods

        bind_flux_pipeline_lora_methods(pipeline)
    elif target == "qwen_image":
        from .lora import bind_qwen_image_pipeline_lora_methods

        bind_qwen_image_pipeline_lora_methods(pipeline)


def _materialize_known_meta_tensors(module: torch.nn.Module) -> None:
    for child in module.modules():
        pos_freqs = getattr(child, "pos_freqs", None)
        neg_freqs = getattr(child, "neg_freqs", None)
        if not (
            isinstance(pos_freqs, torch.Tensor)
            and isinstance(neg_freqs, torch.Tensor)
            and pos_freqs.is_meta
            and neg_freqs.is_meta
            and hasattr(child, "rope_params")
            and hasattr(child, "axes_dim")
            and hasattr(child, "theta")
        ):
            continue

        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        child.pos_freqs = torch.cat(
            [child.rope_params(pos_index, axis_dim, child.theta) for axis_dim in child.axes_dim],
            dim=1,
        )
        child.neg_freqs = torch.cat(
            [child.rope_params(neg_index, axis_dim, child.theta) for axis_dim in child.axes_dim],
            dim=1,
        )


def _coerce_state_dict_for_assign(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    expected_state = module.state_dict()
    for key, value in list(state_dict.items()):
        expected = expected_state.get(key)
        if expected is None or not torch.is_tensor(value) or expected.dtype == value.dtype:
            continue
        if not (expected.is_floating_point() and value.is_floating_point()):
            continue
        state_dict[key] = value.to(dtype=expected.dtype)


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


def _validate_quantization_compatibility(
    quantization_config: dict[str, Any],
    precision: str,
    device: str | torch.device | None,
    checkpoint: str | Path,
) -> None:
    checkpoint_precision = _precision_from_quantization_config(quantization_config)
    if checkpoint_precision is None:
        return
    if checkpoint_precision != precision:
        requested = "fp4" if precision == "nvfp4" else precision
        actual = "fp4" if checkpoint_precision == "nvfp4" else checkpoint_precision
        raise ValueError(
            f"Checkpoint {checkpoint!s} is {actual}, but precision={requested!r} was selected. "
            f"Use a {requested} checkpoint or set precision={actual!r}."
        )

    if not torch.cuda.is_available():
        return
    target_device = torch.device(device or "cuda")
    if target_device.type != "cuda":
        return
    device_index = 0 if target_device.index is None else target_device.index
    capability = torch.cuda.get_device_capability(device_index)
    sm = capability[0] * 10 + capability[1]
    if sm >= 100 and checkpoint_precision != "nvfp4":
        warnings.warn(
            "INT4 quantization on Blackwell GPUs may be slower than FP4. "
            "Use an FP4 checkpoint for best performance when one is available.",
            UserWarning,
            stacklevel=3,
        )
        return
    if sm in {75, 80, 86, 89} and checkpoint_precision != "int4":
        raise ValueError('Please use "int4" quantization for Turing, Ampere, and Ada GPUs.')
    if sm not in {75, 80, 86, 89} and sm < 100:
        raise ValueError(
            f"Unsupported GPU architecture sm{sm}; Nunchaku Lite requires Turing, Ampere, Ada, or Blackwell."
        )


def _precision_from_quantization_config(quantization_config: dict[str, Any]) -> str | None:
    weight_config = quantization_config.get("weight")
    if not isinstance(weight_config, dict):
        return None
    weight_dtype = weight_config.get("dtype")
    if weight_dtype == "int4":
        return "int4"
    if weight_dtype == "fp4_e2m1_all":
        return "nvfp4"
    return None


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
    importlib.import_module("nunchaku_lite.adapters.qwen_image")
    importlib.import_module("nunchaku_lite.adapters.sdxl")
    importlib.import_module("nunchaku_lite.adapters.z_image")
    _BUILTINS_LOADED = True
