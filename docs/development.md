# Development Guide

This guide covers local validation, repository layout, adapter authoring, and
runtime LoRA implementation for `nunchaku_lite`.

## Tests

Run unit tests:

```bash
pytest -q tests
```

Run the opt-in FLUX.1-dev full inference test:

```bash
NUNCHAKU_LITE_RUN_FULL_INFERENCE=1 \
PYTHONPATH=src pytest -q -m full_inference tests/test_full_inference_flux.py
```

The full inference test requires CUDA, model access, and enough VRAM or offload
memory for FLUX.1-dev. It exercises `load_nunchaku_pipeline`, baseline
generation, Diffusers-style FLUX LoRA loading, strength changes, multi-LoRA
composition with Ghibsky plus Canopus UltraRealism, delete/reset, and unload.
Generated images are written to pytest's temp directory by default; set
`NUNCHAKU_LITE_FULL_INFERENCE_OUTPUT_DIR=outputs/full_inference_flux` to keep
them.

Run the opt-in FLUX.2 Klein runtime LoRA full inference test:

```bash
NUNCHAKU_LITE_RUN_FULL_INFERENCE=1 \
PYTHONPATH=src pytest -q -m full_inference tests/test_full_inference_flux2.py
```

The FLUX.2 full inference test defaults to the INT4
`tonera/FLUX.2-klein-9B-Nunchaku` checkpoint and the ComfyUI-format
`artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B` LoRA. Override
`NUNCHAKU_LITE_FLUX2_FULL_INFERENCE_*` environment variables to use another
compatible checkpoint or LoRA.

Run the opt-in Z-Image Turbo runtime LoRA full inference test:

```bash
NUNCHAKU_LITE_RUN_FULL_INFERENCE=1 \
PYTHONPATH=src pytest -q -m full_inference tests/test_full_inference_z_image.py
```

The Z-Image full inference test defaults to the INT4
`nunchaku-ai/nunchaku-z-image-turbo` checkpoint and the
`tarn59/pixel_art_style_lora_z_image_turbo` LoRA. Override
`NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_*` environment variables to use another
compatible checkpoint or LoRA.

Build the extension in place:

```bash
python setup.py build_ext --inplace
```

Useful validation checks:

```bash
python -c "import nunchaku_lite; print(nunchaku_lite.list_adapters())"
python -c "import nunchaku_lite._C as ext; print(hasattr(ext.ops, 'gemm_w4a4'))"
```

## Repository Layout

```text
nunchaku_lite/
  src/
    nunchaku_lite/
      adapters/    Model-specific patch adapters
      csrc/        Python extension bindings
      lora/        Runtime LoRA conversion and binding
      ops/         Python wrappers for native ops
      core.py      Public loading and patching API
      linear.py    Quantized linear modules
native/            Vendored native kernel sources and headers
benchmarks/        End-to-end benchmark scripts
docs/              Project documentation
docs/models/       Model-specific quick-start guides
tests/             Unit tests
```

## Adding a New Model Adapter

New models should be added as small adapter modules under
`src/nunchaku_lite/adapters/`. The adapter should reuse the shared SVDQ helpers
in `nunchaku_lite.adapters.common` for common quantization mechanics, and keep
only model topology and forward-pass differences in the model-specific file.

Recommended structure:

```python
from nunchaku_lite import register_adapter
from nunchaku_lite.adapters.common import (
    build_svdq_context,
    finalize_svdq_checkpoint,
    patch_modules_recursively,
    prepare_transformer_dtype,
)


class MyModelAdapter:
    target = "my_model"

    def matches(self, transformer):
        return transformer.__class__.__name__ == "MyTransformer"

    def patch(self, transformer, checkpoint_state, quantization_config, options):
        context = build_svdq_context(transformer, quantization_config, options)
        prepare_transformer_dtype(transformer, context)

        # Recursively replace generic Diffusers Attention children and
        # checkpoint-backed dense linear children.
        patch_modules_recursively(
            transformer,
            context,
            attention_processor_factory=lambda path, attention: MyAttentionProcessor(),
            linear_filter=lambda path, linear: path.startswith("blocks."),
            module_converters={
                MyFeedForwardBlock: convert_my_feed_forward_block,
                MyAttentionSubclass: lambda attention: MyLiteAttention(attention, context=context),
            },
        )

        # Normalize checkpoint keys here only if this model's checkpoint layout needs it.
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        return checkpoint_state


register_adapter(MyModelAdapter())
```

Adapter responsibilities:

- Use `build_svdq_context`, `patch_modules_recursively`, `svdq_from_linear`,
  `patch_svdq_linears`, and `finalize_svdq_checkpoint` for rank, precision,
  dtype, recursive module replacement, scale-key patching, and fp16 checkpoint
  conversion.
- Keep graph-specific rewrites in the adapter, including QKV fusion, MLP fusion,
  module renaming, and any synthetic projection modules required to match
  checkpoint keys.
- Keep rotary embedding preparation, packed attention paths, KV-cache behavior,
  and custom forward wrappers model-specific.
- Add the adapter import in `nunchaku_lite.core._ensure_builtin_adapters()` if
  it should be built in.
- Add focused tests that build a tiny Diffusers transformer, patch it from a
  synthetic safetensors checkpoint, and verify expected module names and state
  dict keys.

`patch_modules_recursively` mutates the selected module tree in place and
returns a `ModulePatchReport` with replacement and skip counts. Use
`linear_filter` or narrow roots so only checkpoint-backed dense projections are
replaced. Use `skip_subtree` for branches that must remain completely untouched.
Use `attention_processor_factory` for exact Diffusers `Attention` children that
can use the shared `NunchakuAttention` wrapper. Use `module_converters` for
exact-class model blocks or Diffusers `Attention` subclasses that require a
model-specific replacement; unsupported attention subclasses raise `TypeError`
instead of being silently skipped.

Avoid adding a pipeline subclass for a new model unless the upstream Diffusers
pipeline itself requires one. The preferred integration is
`load_nunchaku_pipeline(model_id, pipeline_cls=..., checkpoint=..., target="...")`.

## Adding Runtime LoRA Support

Runtime LoRA support for a new adapter has three parts:

1. Define a transformer mixin in `src/nunchaku_lite/lora/<model>.py`.
2. Add a converter from the model's Diffusers or PEFT LoRA keys into lite
   `.proj_down` / `.proj_up` tensors.
3. Bind the transformer mixin from adapter `patch(...)` and the shared pipeline
   mixin from optional `patch_pipeline(...)`.

The transformer mixin should inherit `NunchakuLoraMixin` and implement
`_convert_lora_to_nunchaku`. Pipeline-level Diffusers APIs use the shared
`NunchakuPipelineLoraMixin`; transformer LoRA binding stays in the adapter
`patch(...)` method.

```python
from pathlib import Path

import torch

from nunchaku_lite.lora.core.convert import (
    is_nunchaku_lite_lora_state_dict,
    normalize_nunchaku_lora_state_dict,
)
from nunchaku_lite.lora.core.runtime import (
    NunchakuLoraMixin,
    load_lora_state_dict,
)


class NunchakuMyModelLoraMixin(NunchakuLoraMixin):
    def _convert_lora_to_nunchaku(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        state_dict = load_lora_state_dict(path_or_state_dict)
        if is_nunchaku_lite_lora_state_dict(state_dict):
            return normalize_nunchaku_lora_state_dict(state_dict, self)
        return convert_my_model_peft_lora_state_dict(state_dict, self)
```

The adapter owns runtime binding. Bind transformer LoRA methods after replacing
modules and normalizing checkpoint keys, then bind pipeline APIs from
`patch_pipeline`:

```python
from nunchaku_lite.lora.core.runtime import (
    NunchakuPipelineLoraMixin,
    bind_pipeline_lora_methods,
    bind_transformer_lora_methods,
)
from nunchaku_lite.lora.my_model import NunchakuMyModelLoraMixin


class MyModelAdapter:
    target = "my_model"

    def patch(self, transformer, checkpoint_state, quantization_config, options):
        # Patch quantized modules and finalize checkpoint state first.
        bind_transformer_lora_methods(transformer, NunchakuMyModelLoraMixin)
        return checkpoint_state

    def patch_pipeline(self, pipeline, *, component_name="transformer", component=None):
        bind_pipeline_lora_methods(
            pipeline,
            NunchakuPipelineLoraMixin,
            component_name=component_name,
        )
```

For a PEFT-style LoRA where incoming keys already look like
`transformer.blocks.0.attn.to_q.lora_A.weight` and
`transformer.blocks.0.attn.to_q.lora_B.weight`, the model LoRA file can look
like this:

```python
from pathlib import Path

import torch
from torch import nn

from nunchaku_lite.linear import AWQW4A16Linear, SVDQW4A4Linear
from nunchaku_lite.lora.core.convert import (
    FusedProjectionSpec,
    is_nunchaku_lite_lora_state_dict,
    normalize_nunchaku_lora_state_dict,
    strip_transformer_prefix,
)
from nunchaku_lite.lora.core.peft import apply_network_alphas, extract_network_alphas, normalize_float_tensor, peft_lora_pairs
from nunchaku_lite.lora.core.runtime import (
    NunchakuLoraMixin,
    load_lora_state_dict,
)


QKV_PROJECTION_SPECS = (
    FusedProjectionSpec(target=".attn.to_qkv", branches=(".attn.to_q", ".attn.to_k", ".attn.to_v")),
)


class NunchakuMyModelLoraMixin(NunchakuLoraMixin):
    def _convert_lora_to_nunchaku(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        state_dict = load_lora_state_dict(path_or_state_dict)
        if is_nunchaku_lite_lora_state_dict(state_dict):
            return normalize_nunchaku_lora_state_dict(state_dict, self)
        return convert_my_model_peft_lora_state_dict(state_dict, self)


def convert_my_model_peft_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    normalized = _normalize_peft_keys(state_dict)
    modules = {
        name: module
        for name, module in transformer.named_modules()
        if isinstance(module, (SVDQW4A4Linear, AWQW4A16Linear))
    }
    converted = {}
    for base_name, lora_a, lora_b in peft_lora_pairs(normalized):
        for target_name, down, up in _map_direct_pair(base_name, lora_a, lora_b, modules):
            converted[f"{target_name}.proj_down"] = down
            converted[f"{target_name}.proj_up"] = up
    return converted


def _normalize_peft_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors = {
        strip_transformer_prefix(key): normalize_float_tensor(value)
        for key, value in state_dict.items()
    }
    return apply_network_alphas(tensors, extract_network_alphas(tensors))


def _map_direct_pair(
    base_name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    modules: dict[str, SVDQW4A4Linear | AWQW4A16Linear],
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    if base_name not in modules:
        return []
    return [(base_name, lora_a.contiguous(), lora_b.contiguous())]
```

The converter can reuse shared helpers from `nunchaku_lite.lora.core.convert`
and `nunchaku_lite.lora.core.peft`. Provide model-specific projection specs for
fused QKV-style modules, a normalizer for incoming LoRA key formats, and a
direct-pair mapper for ordinary projections. Unsupported transformer LoRA keys
should fail in conversion instead of being silently ignored.
