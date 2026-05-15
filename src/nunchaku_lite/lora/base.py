"""Shared runtime LoRA lifecycle helpers for Nunchaku Lite adapters."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn

from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
from ..utils import load_state_dict_in_safetensors


TRANSFORMER_LORA_METHODS = (
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
    "_convert_lora_to_nunchaku",
)

PIPELINE_LORA_METHODS = (
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
)

RUNTIME_LORA_LABEL = "Nunchaku LoRA"


def bind_mixin_methods(instance, mixin_cls: type, method_names: tuple[str, ...]) -> None:
    """Bind selected methods from a mixin class onto one runtime instance.

    Args:
        instance: Object that should receive bound methods.
        mixin_cls: Class providing unbound method implementations.
        method_names: Names to copy from ``mixin_cls`` onto ``instance``.
    """

    for method_name in method_names:
        setattr(instance, method_name, MethodType(getattr(mixin_cls, method_name), instance))


def bind_transformer_lora_methods(
    transformer: nn.Module,
    mixin_cls: type,
) -> None:
    """Attach transformer-level LoRA runtime methods to a patched module.

    Args:
        transformer: Patched transformer that will own adapter state.
        mixin_cls: Model-specific mixin implementing conversion hooks.
    """

    bind_mixin_methods(transformer, mixin_cls, TRANSFORMER_LORA_METHODS)
    ensure_lora_runtime(transformer)


def bind_pipeline_lora_methods(
    pipeline,
    mixin_cls: type,
    *,
    component_name: str = "transformer",
) -> None:
    """Attach Diffusers-compatible LoRA methods to a pipeline object.

    Args:
        pipeline: Pipeline instance whose component should receive runtime LoRA
            operations.
        mixin_cls: Model-specific pipeline mixin implementation.
        component_name: Name of the pipeline component containing the patched
            transformer or UNet.
    """

    pipeline._nunchaku_lite_lora_component_name = component_name
    bind_mixin_methods(pipeline, mixin_cls, PIPELINE_LORA_METHODS)
    pipeline._nunchaku_lite_lora_pipeline_api_bound = True


class NunchakuLoraMixin:
    """Mixin-style method provider for quantized transformer LoRA runtime."""

    def load_lora(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
        *,
        strength: float = 1.0,
        name: str | None = None,
        replace: bool = False,
    ) -> str:
        """Load a runtime LoRA adapter into this patched transformer.

        Args:
            path_or_state_dict: Safetensors path or already loaded LoRA state dict.
            strength: Initial adapter scale.
            name: Optional adapter name.
            replace: Whether to clear existing adapters before loading.
        """

        return load_lora(self, path_or_state_dict, strength=strength, name=name, replace=replace)

    def load_lora_adapter(
        self,
        pretrained_model_name_or_path_or_dict,
        prefix: str = "transformer",
        hotswap: bool = False,
        **kwargs,
    ) -> str:
        """Load a Diffusers/PEFT-style adapter dict through the transformer API.

        Args:
            pretrained_model_name_or_path_or_dict: State dict passed by the
                Diffusers loader path.
            prefix: Component prefix to strip from keys before conversion.
            hotswap: Unsupported PEFT hotswap flag.
            **kwargs: Diffusers loader options; supported adapter metadata is
                consumed and unsupported runtime options are ignored.
        """

        if hotswap:
            raise NotImplementedError(
                f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime does not support PEFT hotswap."
            )
        adapter_name = kwargs.pop("adapter_name", None)
        network_alphas = kwargs.pop("network_alphas", None)
        kwargs.pop("metadata", None)
        kwargs.pop("_pipeline", None)
        kwargs.pop("low_cpu_mem_usage", None)
        state_dict = strip_component_prefixes(dict(pretrained_model_name_or_path_or_dict), prefix=prefix)
        if network_alphas:
            state_dict.update(network_alphas)
        raise_if_text_encoder_lora(state_dict)
        return self.load_lora(state_dict, name=adapter_name)

    def set_lora_strength(self, strength: float = 1.0, name: str | None = None) -> None:
        """Set the scale for one loaded adapter.

        Args:
            strength: New adapter scale.
            name: Adapter name, required when multiple adapters are active.
        """

        set_lora_strength(self, strength=strength, name=name)

    def set_adapters(self, adapter_names: list[str] | str, weights=None) -> None:
        """Select active adapters and optional per-adapter weights.

        Args:
            adapter_names: Adapter name or names to enable.
            weights: Optional scalar/list/dict weights matching Diffusers API.
        """

        set_active_adapters(self, adapter_names, weights)

    def reset_lora(self, name: str | None = None) -> None:
        """Remove one adapter or all adapters and restore base low-rank state.

        Args:
            name: Adapter name to remove, or ``None`` to remove all adapters.
        """

        reset_lora(self, name=name)

    def delete_adapters(self, adapter_names: list[str] | str) -> None:
        """Delete one or more named adapters.

        Args:
            adapter_names: Adapter name or names to remove.
        """

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for adapter_name in adapter_names:
            self.reset_lora(adapter_name)

    def unload_lora(self) -> None:
        """Unload every runtime LoRA adapter from this transformer."""

        self.reset_lora()

    def enable_lora(self) -> None:
        """Enable runtime LoRA composition and reapply active adapters."""

        ensure_lora_runtime(self)
        self._nunchaku_lite_lora_enabled = True
        recompose_loras(self)

    def disable_lora(self) -> None:
        """Disable runtime LoRA composition while keeping loaded adapters."""

        ensure_lora_runtime(self)
        self._nunchaku_lite_lora_enabled = False
        recompose_loras(self)

    def get_list_adapters(self) -> list[str]:
        """Return loaded adapter names in insertion order."""

        ensure_lora_runtime(self)
        return list(self._nunchaku_lite_loras)

    def get_active_adapters(self) -> list[str]:
        """Return currently active adapter names."""

        return active_lora_names(self)

    def fuse_lora(self, *args, **kwargs) -> None:
        """Reject Diffusers fuse requests because runtime LoRAs stay separate."""

        raise NotImplementedError(
            f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime keeps adapters as low-rank branches."
        )

    def _convert_lora_to_nunchaku(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert model-specific LoRA input into Nunchaku Lite tensors.

        Args:
            path_or_state_dict: Safetensors path or loaded LoRA state dict.
        """

        raise NotImplementedError


class DenseRuntimeLoraLinear(nn.Linear):
    """Dense linear layer with runtime LoRA branches managed by Nunchaku Lite."""

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "DenseRuntimeLoraLinear":
        """Wrap an existing dense linear while preserving its state-dict keys.

        Args:
            linear: Source dense linear module.
        """

        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        module.weight = linear.weight
        module.bias = linear.bias
        return module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the dense projection plus any active runtime LoRA branch.

        Args:
            input: Activation tensor whose last dimension is ``in_features``.
        """

        output = F.linear(input, self.weight, self.bias)
        lora_down = getattr(self, "_nunchaku_lite_lora_down", None)
        lora_up = getattr(self, "_nunchaku_lite_lora_up", None)
        if lora_down is None or lora_up is None or lora_down.shape[1] == 0:
            return output
        if lora_down.device != input.device:
            lora_down = lora_down.to(input.device)
            self._nunchaku_lite_lora_down = lora_down
        if lora_up.device != input.device:
            lora_up = lora_up.to(input.device)
            self._nunchaku_lite_lora_up = lora_up
        lora = torch.matmul(input.to(lora_down.dtype), lora_down)
        lora = torch.matmul(lora, lora_up.transpose(0, 1))
        return output + lora.to(output.dtype)


class NunchakuPipelineLoraMixin:
    """Mixin-style provider for Diffusers-compatible pipeline LoRA APIs."""

    _nunchaku_lite_lora_component_name = "transformer"

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        adapter_name: str | None = None,
        hotswap: bool = False,
        **kwargs,
    ) -> None:
        """Load LoRA weights through a Diffusers-compatible pipeline API.

        Args:
            pretrained_model_name_or_path_or_dict: Path, repo id, or state dict
                accepted by the pipeline's ``lora_state_dict`` loader.
            adapter_name: Optional runtime adapter name.
            hotswap: Unsupported PEFT hotswap flag.
            **kwargs: Extra loader options forwarded to ``lora_state_dict``.
        """

        if hotswap:
            raise NotImplementedError(
                f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime does not support PEFT hotswap."
            )
        transformer = self._pipeline_transformer()
        ensure_lora_runtime(transformer)
        kwargs["return_lora_metadata"] = True
        result = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            return_alphas=True,
            **kwargs,
        )
        network_alphas = None
        if len(result) == 3:
            state_dict, network_alphas, _metadata = result
        else:
            state_dict, maybe_alphas_or_metadata = result
            if isinstance(maybe_alphas_or_metadata, dict) and all(
                isinstance(value, torch.Tensor) for value in maybe_alphas_or_metadata.values()
            ):
                network_alphas = maybe_alphas_or_metadata
        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        raise_if_text_encoder_lora(state_dict)
        transformer_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(
                (
                    f"{component_name}.",
                    f"base_model.model.{component_name}.",
                )
            )
        }
        if not transformer_state:
            transformer_state = state_dict
        if network_alphas:
            transformer_state = dict(transformer_state)
            transformer_state.update(network_alphas)
        transformer.load_lora(transformer_state, name=adapter_name)

    def load_lora_adapter(
        self,
        pretrained_model_name_or_path_or_dict,
        adapter_name: str | None = None,
        hotswap: bool = False,
        **kwargs,
    ) -> None:
        """Alias ``load_lora_weights`` for PEFT/Diffusers compatibility.

        Args:
            pretrained_model_name_or_path_or_dict: LoRA source accepted by
                ``load_lora_weights``.
            adapter_name: Optional runtime adapter name.
            hotswap: Unsupported PEFT hotswap flag.
            **kwargs: Extra loader options forwarded to ``load_lora_weights``.
        """

        self.load_lora_weights(
            pretrained_model_name_or_path_or_dict,
            adapter_name=adapter_name,
            hotswap=hotswap,
            **kwargs,
        )

    def set_adapters(
        self,
        adapter_names: list[str] | str,
        adapter_weights: float | dict | list[float] | list[dict] | None = None,
    ) -> None:
        """Select active pipeline adapters and optional component weights.

        Args:
            adapter_names: Adapter name or names to activate.
            adapter_weights: Scalar/list/dict weights accepted by Diffusers APIs.
        """

        transformer = self._pipeline_transformer()
        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        weights = transformer_adapter_weights(adapter_weights, component_name)
        transformer.set_adapters(adapter_names, weights)

    def delete_adapters(self, adapter_names: list[str] | str) -> None:
        """Delete one or more adapters from the patched pipeline component.

        Args:
            adapter_names: Adapter name or names to remove.
        """

        self._pipeline_transformer().delete_adapters(adapter_names)

    def unload_lora_weights(self, reset_to_overwritten_params: bool = False) -> None:
        """Unload all LoRA weights from the patched pipeline component.

        Args:
            reset_to_overwritten_params: Unsupported Diffusers option because
                this runtime never overwrites dense parameters.
        """

        if reset_to_overwritten_params:
            raise NotImplementedError(
                f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime does not overwrite dense params."
            )
        self._pipeline_transformer().unload_lora()

    def enable_lora(self) -> None:
        """Enable LoRA composition on the patched pipeline component."""

        self._pipeline_transformer().enable_lora()

    def disable_lora(self) -> None:
        """Disable LoRA composition on the patched pipeline component."""

        self._pipeline_transformer().disable_lora()

    def get_list_adapters(self) -> dict[str, list[str]]:
        """Return loaded adapter names grouped by pipeline component."""

        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        return {component_name: self._pipeline_transformer().get_list_adapters()}

    def get_active_adapters(self) -> list[str]:
        """Return active adapter names for the patched pipeline component."""

        return self._pipeline_transformer().get_active_adapters()

    def fuse_lora(self, *args, **kwargs) -> None:
        """Reject Diffusers fuse requests for quantized runtime LoRAs."""

        raise NotImplementedError(
            f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime does not support fusing into quantized weights."
        )

    def unfuse_lora(self, *args, **kwargs) -> None:
        """Reject Diffusers unfuse requests for quantized runtime LoRAs."""

        raise NotImplementedError(
            f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime does not support fusing into quantized weights."
        )

    def _pipeline_transformer(self) -> nn.Module:
        """Return the patched pipeline component with transformer LoRA APIs."""

        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        transformer = getattr(self, component_name, None)
        if transformer is None:
            transformer_name = getattr(self, "transformer_name", component_name)
            transformer = getattr(self, transformer_name)
        if not callable(getattr(transformer, "load_lora", None)):
            raise RuntimeError(
                f"Pipeline component {component_name!r} is not bound to the nunchaku_lite "
                "transformer LoRA runtime. Patch the transformer with its adapter before "
                "binding pipeline LoRA methods."
            )
        return transformer


def load_lora(
    transformer: nn.Module,
    path_or_state_dict: str | Path | dict[str, torch.Tensor],
    *,
    strength: float = 1.0,
    name: str | None = None,
    replace: bool = False,
) -> str:
    """Load a LoRA into a patched transformer and return its adapter name.

    Args:
        transformer: Patched transformer with bound Nunchaku Lite LoRA methods.
        path_or_state_dict: Safetensors path or in-memory LoRA state dict.
        strength: Initial adapter scale used when composing active LoRAs.
        name: Optional adapter name. If omitted, a path stem or generated name
            is used.
        replace: Whether to remove existing adapters before loading this one.
    """

    ensure_lora_base_state(transformer)
    if replace:
        transformer._nunchaku_lite_loras.clear()
        transformer._nunchaku_lite_active_loras.clear()

    lora_name = resolve_lora_name(transformer, path_or_state_dict, name)
    converted = transformer._convert_lora_to_nunchaku(path_or_state_dict)
    converted = {key: value.detach().cpu() for key, value in converted.items()}
    transformer._nunchaku_lite_loras[lora_name] = {"state_dict": converted, "strength": float(strength)}
    transformer._nunchaku_lite_active_loras = [
        active_name for active_name in transformer._nunchaku_lite_active_loras if active_name != lora_name
    ]
    transformer._nunchaku_lite_active_loras.append(lora_name)
    recompose_loras(transformer)
    return lora_name


def set_lora_strength(transformer: nn.Module, strength: float = 1.0, name: str | None = None) -> None:
    """Set one adapter strength and recompose all active LoRAs.

    Args:
        transformer: Patched transformer containing runtime adapter state.
        strength: New scalar multiplier for the adapter.
        name: Adapter name. Required when more than one adapter is active.
    """

    ensure_lora_runtime(transformer)
    if not transformer._nunchaku_lite_loras:
        return
    if name is None:
        active_names = active_lora_names(transformer)
        if len(active_names) != 1:
            raise ValueError("Multiple LoRAs are active; pass name=... to set one strength.")
        name = active_names[0]
    try:
        transformer._nunchaku_lite_loras[name]["strength"] = float(strength)
    except KeyError as exc:
        raise ValueError(f"No loaded LoRA named {name!r}.") from exc
    recompose_loras(transformer)


def reset_lora(transformer: nn.Module, name: str | None = None) -> None:
    """Remove adapters from a patched transformer and restore base low-rank state.

    Args:
        transformer: Patched transformer containing runtime adapter state.
        name: Optional adapter name. When omitted, all adapters are removed.
    """

    ensure_lora_runtime(transformer)
    if name is None:
        transformer._nunchaku_lite_loras.clear()
        transformer._nunchaku_lite_active_loras.clear()
    else:
        try:
            del transformer._nunchaku_lite_loras[name]
        except KeyError as exc:
            raise ValueError(f"No loaded LoRA named {name!r}.") from exc
        transformer._nunchaku_lite_active_loras = [
            active_name for active_name in transformer._nunchaku_lite_active_loras if active_name != name
        ]
    recompose_loras(transformer)


def load_lora_state_dict(path_or_state_dict: str | Path | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load a LoRA state dict from memory or a safetensors file.

    Args:
        path_or_state_dict: Existing state dict or path to a safetensors file.
    """

    if isinstance(path_or_state_dict, dict):
        return dict(path_or_state_dict)
    return load_state_dict_in_safetensors(path_or_state_dict)


def strip_component_prefixes(
    state_dict: dict[str, torch.Tensor], prefix: str = "transformer"
) -> dict[str, torch.Tensor]:
    """Remove pipeline component prefixes from LoRA state-dict keys.

    Args:
        state_dict: LoRA tensors keyed by pipeline or model component names.
        prefix: Component prefix to remove, usually ``transformer``.
    """

    prefixes = (
        f"{prefix}.",
        f"base_model.model.{prefix}.",
    )
    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        for current_prefix in prefixes:
            if new_key.startswith(current_prefix):
                new_key = new_key[len(current_prefix) :]
                break
        stripped[new_key] = value
    return stripped


def raise_if_text_encoder_lora(state_dict: dict[str, torch.Tensor]) -> None:
    """Reject LoRA state dicts that target text encoder modules.

    Args:
        state_dict: Candidate LoRA tensors to inspect before transformer-only
            runtime loading.
    """

    text_keys = [
        key
        for key in state_dict
        if key.startswith(("text_encoder.", "text_encoder_2.", "base_model.model.text_encoder."))
    ]
    if text_keys:
        sample = ", ".join(text_keys[:5])
        raise NotImplementedError(
            f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime supports transformer LoRA weights only; "
            f"text encoder LoRA keys are not supported: {sample}"
        )


def transformer_adapter_weights(adapter_weights, component_name: str = "transformer"):
    """Extract the transformer component weights from Diffusers adapter weights.

    Args:
        adapter_weights: Scalar, list, or component-name dict accepted by
            Diffusers pipeline APIs.
        component_name: Component key to read when weights are dictionaries.
    """

    if isinstance(adapter_weights, dict):
        return adapter_weights.get(component_name)
    if isinstance(adapter_weights, list):
        return [
            weight.get(component_name) if isinstance(weight, dict) else weight
            for weight in adapter_weights
        ]
    return adapter_weights


def set_active_adapters(transformer: nn.Module, adapter_names: list[str] | str, weights=None) -> None:
    """Select active adapters and update their runtime weights.

    Args:
        transformer: Patched transformer containing loaded adapters.
        adapter_names: Adapter name or ordered list of adapter names to enable.
        weights: Optional scalar/list/dict weights matching Diffusers adapter
            API conventions.
    """

    ensure_lora_runtime(transformer)
    adapter_names = [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)
    if len(adapter_names) != len(weights):
        raise ValueError(
            f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(weights)}."
        )

    missing = [name for name in adapter_names if name not in transformer._nunchaku_lite_loras]
    if missing:
        loaded = set(transformer._nunchaku_lite_loras)
        raise ValueError(f"Adapter name(s) {set(missing)} not in the list of present adapters: {loaded}.")

    transformer._nunchaku_lite_active_loras = list(adapter_names)
    for adapter_name, weight in zip(adapter_names, weights):
        if isinstance(weight, dict):
            weight = weight.get("transformer", 1.0)
        transformer._nunchaku_lite_loras[adapter_name]["strength"] = 1.0 if weight is None else float(weight)
    recompose_loras(transformer)


def active_lora_names(transformer: nn.Module) -> list[str]:
    """Return currently active adapter names, respecting the global enable flag.

    Args:
        transformer: Patched transformer containing runtime adapter state.
    """

    ensure_lora_runtime(transformer)
    if not transformer._nunchaku_lite_lora_enabled:
        return []
    return [name for name in transformer._nunchaku_lite_active_loras if name in transformer._nunchaku_lite_loras]


def active_lora_entries(transformer: nn.Module) -> list[dict]:
    """Return active adapter metadata entries in composition order.

    Args:
        transformer: Patched transformer containing runtime adapter state.
    """

    return [transformer._nunchaku_lite_loras[name] for name in active_lora_names(transformer)]


def ensure_lora_runtime(transformer: nn.Module) -> None:
    """Initialize runtime LoRA bookkeeping fields if they are missing.

    Args:
        transformer: Patched transformer that may not yet have adapter state.
    """

    if not hasattr(transformer, "_nunchaku_lite_loras"):
        transformer._nunchaku_lite_loras = OrderedDict()
    if not hasattr(transformer, "_nunchaku_lite_lora_base_state"):
        transformer._nunchaku_lite_lora_base_state = None
    if not hasattr(transformer, "_nunchaku_lite_active_loras"):
        transformer._nunchaku_lite_active_loras = list(transformer._nunchaku_lite_loras)
    if not hasattr(transformer, "_nunchaku_lite_lora_enabled"):
        transformer._nunchaku_lite_lora_enabled = True


def ensure_lora_base_state(transformer: nn.Module) -> None:
    """Snapshot base low-rank tensors before runtime LoRA composition.

    Args:
        transformer: Patched transformer whose SVDQ/AWQ modules should be
            restorable after adapters are disabled or unloaded.
    """

    ensure_lora_runtime(transformer)
    if transformer._nunchaku_lite_lora_base_state is not None:
        return

    base_state = {}
    for name, module in lora_modules(transformer).items():
        if isinstance(module, SVDQW4A4Linear):
            base_state[f"{name}.proj_down"] = module.proj_down.detach().clone()
            base_state[f"{name}.proj_up"] = module.proj_up.detach().clone()
        elif isinstance(module, AWQW4A16Linear):
            device = module.qweight.device
            dtype = module.wscales.dtype
            base_state[f"{name}.proj_down"] = torch.empty(module.in_features, 0, device=device, dtype=dtype)
            base_state[f"{name}.proj_up"] = torch.empty(module.out_features, 0, device=device, dtype=dtype)
        else:
            base_state[f"{name}.proj_down"] = torch.empty(
                module.in_features,
                0,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            base_state[f"{name}.proj_up"] = torch.empty(
                module.out_features,
                0,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
    transformer._nunchaku_lite_lora_base_state = base_state


def resolve_lora_name(
    transformer: nn.Module,
    path_or_state_dict: str | Path | dict[str, torch.Tensor],
    name: str | None,
) -> str:
    """Choose and validate the adapter name for a newly loaded LoRA.

    Args:
        transformer: Patched transformer containing existing adapter names.
        path_or_state_dict: Source used to derive a default name when needed.
        name: User-provided adapter name, or ``None`` to auto-generate one.
    """

    if name is None:
        if isinstance(path_or_state_dict, (str, Path)):
            name = Path(path_or_state_dict).stem
        else:
            name = f"lora_{len(transformer._nunchaku_lite_loras) + 1}"
    if name in transformer._nunchaku_lite_loras:
        raise ValueError(f"A LoRA named {name!r} is already active. Use replace=True or choose another name.")
    return name


def recompose_loras(transformer: nn.Module) -> None:
    """Rebuild module LoRA tensors from base state plus all active adapters.

    Args:
        transformer: Patched transformer whose quantized modules should receive
            the currently active adapter composition.
    """

    ensure_lora_base_state(transformer)
    modules = lora_modules(transformer)
    active_entries = active_lora_entries(transformer)

    for name, module in modules.items():
        if isinstance(module, SVDQW4A4Linear):
            base_down = transformer._nunchaku_lite_lora_base_state[f"{name}.proj_down"]
            base_up = transformer._nunchaku_lite_lora_base_state[f"{name}.proj_up"]
            if not active_entries:
                down = base_down.to(device=module.proj_down.device, dtype=module.proj_down.dtype)
                up = base_up.to(device=module.proj_up.device, dtype=module.proj_up.dtype)
            else:
                logical_downs = [svdq_down_to_logical(base_down, module, name)]
                logical_ups = [svdq_up_to_logical(base_up, module, name)]
                for entry in active_entries:
                    state_dict = entry["state_dict"]
                    down_key = f"{name}.proj_down"
                    up_key = f"{name}.proj_up"
                    if down_key not in state_dict and up_key not in state_dict:
                        continue
                    if down_key not in state_dict or up_key not in state_dict:
                        raise ValueError(f"Incomplete LoRA tensors for {name}.")
                    down_logical = svdq_down_to_logical(state_dict[down_key], module, name)
                    up_logical = svdq_up_to_logical(state_dict[up_key], module, name)
                    if down_logical.shape[0] != up_logical.shape[1]:
                        raise ValueError(
                            f"LoRA rank mismatch for {name}: "
                            f"proj_down={tuple(down_logical.shape)}, proj_up={tuple(up_logical.shape)}"
                        )
                    down_logical = down_logical.to(device=logical_downs[0].device, dtype=logical_downs[0].dtype)
                    up_logical = up_logical.to(device=logical_ups[0].device, dtype=logical_ups[0].dtype)
                    logical_downs.append(down_logical * float(entry["strength"]))
                    logical_ups.append(up_logical)
                down = pack_lowrank_weight(torch.cat(logical_downs, dim=0), down=True)
                up = pack_lowrank_weight(torch.cat(logical_ups, dim=1), down=False)
                down = down.to(device=module.proj_down.device, dtype=module.proj_down.dtype)
                up = up.to(device=module.proj_up.device, dtype=module.proj_up.dtype)
            module.proj_down = nn.Parameter(down, requires_grad=module.proj_down.requires_grad)
            module.proj_up = nn.Parameter(up, requires_grad=module.proj_up.requires_grad)
            module.rank = down.shape[1]
        else:
            down = transformer._nunchaku_lite_lora_base_state[f"{name}.proj_down"]
            up = transformer._nunchaku_lite_lora_base_state[f"{name}.proj_up"]
            for entry in active_entries:
                state_dict = entry["state_dict"]
                down_key = f"{name}.proj_down"
                up_key = f"{name}.proj_up"
                if down_key not in state_dict and up_key not in state_dict:
                    continue
                if down_key not in state_dict or up_key not in state_dict:
                    raise ValueError(f"Incomplete LoRA tensors for {name}.")
                lora_down = fit_lora_tensor(state_dict[down_key], module.in_features, down=True, module_name=name)
                lora_up = fit_lora_tensor(state_dict[up_key], module.out_features, down=False, module_name=name)
                if lora_down.shape[1] != lora_up.shape[1]:
                    raise ValueError(
                        f"LoRA rank mismatch for {name}: "
                        f"proj_down={tuple(lora_down.shape)}, proj_up={tuple(lora_up.shape)}"
                    )
                lora_down = lora_down.to(device=down.device, dtype=down.dtype) * float(entry["strength"])
                lora_up = lora_up.to(device=up.device, dtype=up.dtype)
                down = torch.cat([down, lora_down], dim=1)
                up = torch.cat([up, lora_up], dim=1)
            if isinstance(module, AWQW4A16Linear):
                device = module.qweight.device
                dtype = module.wscales.dtype
            else:
                device = module.weight.device
                dtype = module.weight.dtype
            down = down.to(device=device, dtype=dtype)
            up = up.to(device=device, dtype=dtype)
            module._nunchaku_lite_lora_down = down
            module._nunchaku_lite_lora_up = up


def svdq_modules(transformer: nn.Module) -> dict[str, SVDQW4A4Linear]:
    """Return SVDQ W4A4 linear modules addressable by LoRA conversion.

    Args:
        transformer: Patched transformer to scan by module name.
    """

    return {name: module for name, module in transformer.named_modules() if isinstance(module, SVDQW4A4Linear)}


def awq_modules(transformer: nn.Module) -> dict[str, AWQW4A16Linear]:
    """Return AWQ W4A16 linear modules addressable by LoRA conversion.

    Args:
        transformer: Patched transformer to scan by module name.
    """

    return {name: module for name, module in transformer.named_modules() if isinstance(module, AWQW4A16Linear)}


def dense_lora_modules(transformer: nn.Module) -> dict[str, DenseRuntimeLoraLinear]:
    """Return dense linear modules addressable by runtime LoRA conversion.

    Args:
        transformer: Patched transformer to scan by module name.
    """

    return {name: module for name, module in transformer.named_modules() if isinstance(module, DenseRuntimeLoraLinear)}


def lora_modules(transformer: nn.Module) -> dict[str, SVDQW4A4Linear | AWQW4A16Linear | DenseRuntimeLoraLinear]:
    """Return every linear module that can receive runtime LoRA tensors.

    Args:
        transformer: Patched transformer to scan by module name.
    """

    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear | DenseRuntimeLoraLinear] = {}
    modules.update(svdq_modules(transformer))
    modules.update(awq_modules(transformer))
    modules.update(dense_lora_modules(transformer))
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
