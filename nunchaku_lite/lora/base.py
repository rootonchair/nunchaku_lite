"""Shared runtime LoRA lifecycle helpers for Nunchaku Lite adapters."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import MethodType

import torch
from torch import nn

from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
from ..utils import load_state_dict_in_safetensors


LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def bind_mixin_methods(instance, mixin_cls: type, method_names: tuple[str, ...]) -> None:
    for method_name in method_names:
        setattr(instance, method_name, MethodType(getattr(mixin_cls, method_name), instance))


class NunchakuLoraMixin:
    """Mixin-style method provider for quantized transformer LoRA runtime."""

    _nunchaku_lite_lora_model_name = "transformer"

    def load_lora(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
        *,
        strength: float = 1.0,
        name: str | None = None,
        replace: bool = False,
    ) -> str:
        return load_lora(self, path_or_state_dict, strength=strength, name=name, replace=replace)

    def load_lora_adapter(
        self,
        pretrained_model_name_or_path_or_dict,
        prefix: str = "transformer",
        hotswap: bool = False,
        **kwargs,
    ) -> str:
        if hotswap:
            raise NotImplementedError(
                f"nunchaku_lite {getattr(self, '_nunchaku_lite_lora_model_name', 'transformer')} "
                "runtime LoRA does not support PEFT hotswap."
            )
        adapter_name = kwargs.pop("adapter_name", None)
        network_alphas = kwargs.pop("network_alphas", None)
        kwargs.pop("metadata", None)
        kwargs.pop("_pipeline", None)
        kwargs.pop("low_cpu_mem_usage", None)
        state_dict = strip_component_prefixes(dict(pretrained_model_name_or_path_or_dict), prefix=prefix)
        if network_alphas:
            state_dict.update(network_alphas)
        raise_if_text_encoder_lora(state_dict, getattr(self, "_nunchaku_lite_lora_model_name", "transformer"))
        return self.load_lora(state_dict, name=adapter_name)

    def set_lora_strength(self, strength: float = 1.0, name: str | None = None) -> None:
        set_lora_strength(self, strength=strength, name=name)

    def set_adapters(self, adapter_names: list[str] | str, weights=None) -> None:
        set_active_adapters(self, adapter_names, weights)

    def reset_lora(self, name: str | None = None) -> None:
        reset_lora(self, name=name)

    def delete_adapters(self, adapter_names: list[str] | str) -> None:
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for adapter_name in adapter_names:
            self.reset_lora(adapter_name)

    def unload_lora(self) -> None:
        self.reset_lora()

    def enable_lora(self) -> None:
        ensure_lora_runtime(self)
        self._nunchaku_lite_lora_enabled = True
        recompose_loras(self)

    def disable_lora(self) -> None:
        ensure_lora_runtime(self)
        self._nunchaku_lite_lora_enabled = False
        recompose_loras(self)

    def get_list_adapters(self) -> list[str]:
        ensure_lora_runtime(self)
        return list(self._nunchaku_lite_loras)

    def get_active_adapters(self) -> list[str]:
        return active_lora_names(self)

    def fuse_lora(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"nunchaku_lite {getattr(self, '_nunchaku_lite_lora_model_name', 'transformer')} "
            "runtime LoRA keeps adapters as low-rank branches."
        )

    def _convert_lora_to_lite(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class NunchakuPipelineLoraMixin:
    """Mixin-style provider for Diffusers-compatible pipeline LoRA APIs."""

    _nunchaku_lite_lora_model_name = "transformer"
    _nunchaku_lite_lora_component_name = "transformer"

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        adapter_name: str | None = None,
        hotswap: bool = False,
        **kwargs,
    ) -> None:
        if hotswap:
            raise NotImplementedError(
                f"nunchaku_lite {getattr(self, '_nunchaku_lite_lora_model_name', 'transformer')} "
                "runtime LoRA does not support PEFT hotswap."
            )
        transformer = self._pipeline_transformer()
        ensure_lora_runtime(transformer)
        kwargs["return_lora_metadata"] = True
        result = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            return_alphas=True,
            **kwargs,
        )
        if len(result) == 3:
            state_dict, network_alphas, _metadata = result
        else:
            state_dict, network_alphas = result
        model_name = getattr(self, "_nunchaku_lite_lora_model_name", "transformer")
        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        raise_if_text_encoder_lora(state_dict, model_name)
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
        transformer = self._pipeline_transformer()
        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        weights = transformer_adapter_weights(adapter_weights, component_name)
        transformer.set_adapters(adapter_names, weights)

    def delete_adapters(self, adapter_names: list[str] | str) -> None:
        self._pipeline_transformer().delete_adapters(adapter_names)

    def unload_lora_weights(self, reset_to_overwritten_params: bool = False) -> None:
        if reset_to_overwritten_params:
            raise NotImplementedError(
                f"nunchaku_lite {getattr(self, '_nunchaku_lite_lora_model_name', 'transformer')} "
                "runtime LoRA does not overwrite dense params."
            )
        self._pipeline_transformer().unload_lora()

    def enable_lora(self) -> None:
        self._pipeline_transformer().enable_lora()

    def disable_lora(self) -> None:
        self._pipeline_transformer().disable_lora()

    def get_list_adapters(self) -> dict[str, list[str]]:
        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        return {component_name: self._pipeline_transformer().get_list_adapters()}

    def get_active_adapters(self) -> list[str]:
        return self._pipeline_transformer().get_active_adapters()

    def fuse_lora(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"nunchaku_lite {getattr(self, '_nunchaku_lite_lora_model_name', 'transformer')} "
            "runtime LoRA does not support fusing into quantized weights."
        )

    def unfuse_lora(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"nunchaku_lite {getattr(self, '_nunchaku_lite_lora_model_name', 'transformer')} "
            "runtime LoRA does not support fusing into quantized weights."
        )

    def _pipeline_transformer(self) -> nn.Module:
        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        transformer = getattr(self, component_name, None)
        if transformer is None:
            transformer_name = getattr(self, "transformer_name", component_name)
            transformer = getattr(self, transformer_name)
        if not hasattr(transformer, "load_lora"):
            self._bind_transformer_lora_methods(transformer)
        return transformer

    def _bind_transformer_lora_methods(self, transformer: nn.Module) -> None:
        raise NotImplementedError


def load_lora(
    transformer: nn.Module,
    path_or_state_dict: str | Path | dict[str, torch.Tensor],
    *,
    strength: float = 1.0,
    name: str | None = None,
    replace: bool = False,
) -> str:
    """Load a LoRA into a patched transformer and return its adapter name."""

    ensure_lora_base_state(transformer)
    if replace:
        transformer._nunchaku_lite_loras.clear()
        transformer._nunchaku_lite_active_loras.clear()

    lora_name = resolve_lora_name(transformer, path_or_state_dict, name)
    converted = transformer._convert_lora_to_lite(path_or_state_dict)
    converted = {key: value.detach().cpu() for key, value in converted.items()}
    transformer._nunchaku_lite_loras[lora_name] = {"state_dict": converted, "strength": float(strength)}
    transformer._nunchaku_lite_active_loras = [
        active_name for active_name in transformer._nunchaku_lite_active_loras if active_name != lora_name
    ]
    transformer._nunchaku_lite_active_loras.append(lora_name)
    recompose_loras(transformer)
    return lora_name


def set_lora_strength(transformer: nn.Module, strength: float = 1.0, name: str | None = None) -> None:
    """Set the strength for one active LoRA and recompose all active adapters."""

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
    """Reset all LoRAs or remove one named LoRA from a patched transformer."""

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
    if isinstance(path_or_state_dict, dict):
        return dict(path_or_state_dict)
    return load_state_dict_in_safetensors(path_or_state_dict)


def strip_component_prefixes(state_dict: dict[str, torch.Tensor], prefix: str = "transformer") -> dict[str, torch.Tensor]:
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


def raise_if_text_encoder_lora(state_dict: dict[str, torch.Tensor], model_name: str) -> None:
    text_keys = [
        key
        for key in state_dict
        if key.startswith(("text_encoder.", "text_encoder_2.", "base_model.model.text_encoder."))
    ]
    if text_keys:
        sample = ", ".join(text_keys[:5])
        raise NotImplementedError(
            f"nunchaku_lite {model_name} runtime LoRA supports transformer LoRA weights only; "
            f"text encoder LoRA keys are not supported: {sample}"
        )


def transformer_adapter_weights(adapter_weights, component_name: str = "transformer"):
    if isinstance(adapter_weights, dict):
        return adapter_weights.get(component_name)
    if isinstance(adapter_weights, list):
        return [
            weight.get(component_name) if isinstance(weight, dict) else weight
            for weight in adapter_weights
        ]
    return adapter_weights


def set_active_adapters(transformer: nn.Module, adapter_names: list[str] | str, weights=None) -> None:
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
    ensure_lora_runtime(transformer)
    if not transformer._nunchaku_lite_lora_enabled:
        return []
    return [name for name in transformer._nunchaku_lite_active_loras if name in transformer._nunchaku_lite_loras]


def active_lora_entries(transformer: nn.Module) -> list[dict]:
    return [transformer._nunchaku_lite_loras[name] for name in active_lora_names(transformer)]


def ensure_lora_runtime(transformer: nn.Module) -> None:
    if not hasattr(transformer, "_nunchaku_lite_loras"):
        transformer._nunchaku_lite_loras = OrderedDict()
    if not hasattr(transformer, "_nunchaku_lite_lora_base_state"):
        transformer._nunchaku_lite_lora_base_state = None
    if not hasattr(transformer, "_nunchaku_lite_active_loras"):
        transformer._nunchaku_lite_active_loras = list(transformer._nunchaku_lite_loras)
    if not hasattr(transformer, "_nunchaku_lite_lora_enabled"):
        transformer._nunchaku_lite_lora_enabled = True


def ensure_lora_base_state(transformer: nn.Module) -> None:
    ensure_lora_runtime(transformer)
    if transformer._nunchaku_lite_lora_base_state is not None:
        return

    base_state = {}
    for name, module in lora_modules(transformer).items():
        if isinstance(module, SVDQW4A4Linear):
            base_state[f"{name}.proj_down"] = module.proj_down.detach().clone()
            base_state[f"{name}.proj_up"] = module.proj_up.detach().clone()
        else:
            device = module.qweight.device
            dtype = module.wscales.dtype
            base_state[f"{name}.proj_down"] = torch.empty(module.in_features, 0, device=device, dtype=dtype)
            base_state[f"{name}.proj_up"] = torch.empty(module.out_features, 0, device=device, dtype=dtype)
    transformer._nunchaku_lite_lora_base_state = base_state


def resolve_lora_name(
    transformer: nn.Module,
    path_or_state_dict: str | Path | dict[str, torch.Tensor],
    name: str | None,
) -> str:
    if name is None:
        if isinstance(path_or_state_dict, (str, Path)):
            name = Path(path_or_state_dict).stem
        else:
            name = f"lora_{len(transformer._nunchaku_lite_loras) + 1}"
    if name in transformer._nunchaku_lite_loras:
        raise ValueError(f"A LoRA named {name!r} is already active. Use replace=True or choose another name.")
    return name


def recompose_loras(transformer: nn.Module) -> None:
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
            down = down.to(device=module.qweight.device, dtype=module.wscales.dtype)
            up = up.to(device=module.qweight.device, dtype=module.wscales.dtype)
            module._nunchaku_lite_lora_down = down
            module._nunchaku_lite_lora_up = up


def svdq_modules(transformer: nn.Module) -> dict[str, SVDQW4A4Linear]:
    return {name: module for name, module in transformer.named_modules() if isinstance(module, SVDQW4A4Linear)}


def awq_modules(transformer: nn.Module) -> dict[str, AWQW4A16Linear]:
    return {name: module for name, module in transformer.named_modules() if isinstance(module, AWQW4A16Linear)}


def lora_modules(transformer: nn.Module) -> dict[str, SVDQW4A4Linear | AWQW4A16Linear]:
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear] = {}
    modules.update(svdq_modules(transformer))
    modules.update(awq_modules(transformer))
    return modules


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


def diffusers_pairs(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
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


def validate_lite_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
    *,
    model_name: str,
) -> dict[str, torch.Tensor]:
    modules = lora_modules(transformer)
    if not state_dict:
        raise ValueError(f"LoRA state dict did not contain any supported {model_name} projection tensors.")

    valid = {}
    for down_key, up_key in iter_lora_pairs(state_dict):
        module_name = down_key[: -len(".proj_down")]
        if module_name not in modules:
            raise ValueError(f"LoRA target {module_name!r} does not exist on this patched {model_name} transformer.")
        module = modules[module_name]
        down = fit_lora_tensor(state_dict[down_key], module.in_features, down=True, module_name=module_name)
        up = fit_lora_tensor(state_dict[up_key], module.out_features, down=False, module_name=module_name)
        if down.shape[1] != up.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for {module_name}: proj_down={tuple(down.shape)}, proj_up={tuple(up.shape)}"
            )
        valid[down_key] = down
        valid[up_key] = up
    return valid


def iter_lora_pairs(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    down_keys = sorted(key for key in state_dict if key.endswith(".proj_down"))
    pairs = []
    for down_key in down_keys:
        up_key = f"{down_key[: -len('.proj_down')]}.proj_up"
        if up_key not in state_dict:
            raise ValueError(f"Missing proj_up tensor for {down_key!r}.")
        pairs.append((down_key, up_key))
    return pairs


def fit_lora_tensor(tensor: torch.Tensor, feature_dim: int, *, down: bool, module_name: str) -> torch.Tensor:
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
    fitted = fit_lora_tensor(tensor, module.in_features, down=True, module_name=module_name)
    if looks_like_packed_lowrank(fitted, feature_dim=module.in_features):
        return unpack_lowrank_weight(fitted, down=True)
    return fitted.transpose(0, 1).contiguous()


def svdq_up_to_logical(tensor: torch.Tensor, module: SVDQW4A4Linear, module_name: str) -> torch.Tensor:
    fitted = fit_lora_tensor(tensor, module.out_features, down=False, module_name=module_name)
    if looks_like_packed_lowrank(fitted, feature_dim=module.out_features):
        return unpack_lowrank_weight(fitted, down=False)
    return fitted.contiguous()


def looks_like_packed_lowrank(tensor: torch.Tensor, feature_dim: int) -> bool:
    return tensor.shape[0] == feature_dim and feature_dim % 16 == 0 and tensor.shape[1] % 16 == 0


def pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Pack a low-rank weight for Nunchaku W4A4 kernels."""

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
    """Unpack a Nunchaku W4A4 low-rank tensor into logical rank-major layout."""

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
