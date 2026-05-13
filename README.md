# nunchaku_lite

`nunchaku_lite` is a small, plugin-oriented runtime package for applying Nunchaku v2 quantized transformer weights to existing Diffusers pipelines. It is designed to patch a pipeline's transformer module in place, so downstream code can keep using standard Diffusers pipeline classes without subclassing or importing the full `nunchaku` package.

The first built-in adapters target Flux, Flux2, and Z-Image transformer classes with SVDQ W4A4 checkpoints.

## Design Goals

- Keep the public integration surface model-agnostic.
- Patch existing `torch.nn.Module` transformer instances in place.
- Use a registry of small adapters for model-specific graph rewrites.
- Package only the native kernels and Python code required for the lite runtime.
- Avoid a hard dependency on the original `nunchaku` Python package.

## Status

This package is an early lite runtime. The current built-in adapter set is:

| Adapter | Target | Status |
| --- | --- | --- |
| `flux` | Diffusers `FluxTransformer2DModel` | Implemented |
| `flux2` | Diffusers `Flux2Transformer2DModel` | Implemented |
| `z_image` | Diffusers `ZImageTransformer2DModel` | Implemented |

Additional model families should be added through the common adapter registry rather than through pipeline-specific subclasses.

## Requirements

- Python 3.10 or newer
- PyTorch 2.7 or newer with CUDA
- CUDA toolkit with `nvcc`
- Ninja
- Diffusers 0.36 or newer

The build detects the local GPU architecture by default. Supported targets are `sm75`, `sm80`, `sm86`, `sm89`, `sm120a`, and `sm121a`, subject to the installed CUDA toolkit version.

## Installation

Install from source:

```bash
pip install .
```

For wheel builds:

```bash
python setup.py bdist_wheel
pip install dist/nunchaku_lite-*.whl
```

By default, the build uses `NUNCHAKU_INSTALL_MODE=FAST` and compiles for visible local CUDA devices. To build all supported architectures:

```bash
NUNCHAKU_INSTALL_MODE=ALL pip install .
```

## Quick Start: Z-Image Turbo FP4

```python
import torch
from diffusers import ZImagePipeline
from nunchaku_lite import patch_transformer

model_id = "Tongyi-MAI/Z-Image-Turbo"
checkpoint = "nunchaku-ai/nunchaku-z-image-turbo/svdq-fp4_r128-z-image-turbo.safetensors"

pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

patch_transformer(
    pipe.transformer,
    checkpoint,
    precision="fp4",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

pipe = pipe.to("cuda")

image = pipe(
    prompt="a cinematic photo of a glass greenhouse full of tropical plants during golden hour",
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=0.0,
    generator=torch.Generator(device="cuda").manual_seed(12345),
).images[0]

image.save("z_image_nunchaku_lite.png")
```

## Quick Start: FLUX.1-schnell FP4

```python
import torch
from diffusers import FluxPipeline
from nunchaku_lite import patch_transformer

model_id = "black-forest-labs/FLUX.1-schnell"
checkpoint = "nunchaku-ai/nunchaku-flux.1-schnell/svdq-fp4_r32-flux.1-schnell.safetensors"

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

patch_transformer(
    pipe.transformer,
    checkpoint,
    target="flux",
    precision="fp4",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

pipe = pipe.to("cuda")

image = pipe(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
    generator=torch.Generator(device="cuda").manual_seed(12345),
).images[0]

image.save("flux_schnell_nunchaku_lite.png")
```

## Quick Start: FLUX.2 Klein FP4

```python
import torch
from diffusers import Flux2KleinPipeline
from nunchaku_lite import patch_transformer

model_id = "tonera/FLUX.2-klein-9B-Nunchaku"
checkpoint = "tonera/FLUX.2-klein-9B-Nunchaku/svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors"

pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

patch_transformer(
    pipe.transformer,
    checkpoint,
    target="flux2",
    precision="fp4",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

pipe = pipe.to("cuda")

image = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(12345),
).images[0]

image.save("flux2_klein_nunchaku_lite.png")
```

Checkpoint paths can be local `.safetensors` files or Hugging Face paths of the form:

```text
org-or-user/repo-name/path/to/checkpoint.safetensors
```

## Public API

```python
from nunchaku_lite import (
    TransformerAdapter,
    list_adapters,
    patch_transformer,
    register_adapter,
)
```

### `patch_transformer`

```python
patch_transformer(
    transformer,
    checkpoint,
    target="auto",
    precision="auto",
    torch_dtype=None,
    device=None,
    strict=True,
    adapter_options=None,
)
```

Arguments:

- `transformer`: the Diffusers transformer module to patch.
- `checkpoint`: local or Hugging Face `.safetensors` checkpoint path.
- `target`: adapter name, or `"auto"` to select the only matching adapter.
- `precision`: `"auto"`, `"fp4"`, or `"int4"`. Internally, `"fp4"` maps to NVFP4 kernels.
- `torch_dtype`: optional model dtype, typically `torch.bfloat16` or `torch.float16`.
- `device`: optional destination device after patching.
- `strict`: forwarded to `load_state_dict`.
- `adapter_options`: model-specific adapter options.

The function is idempotent for the same target. A transformer patched once will be returned unchanged if patched again with the same target.

### Adapter Registry

Adapters implement a small protocol:

```python
class MyAdapter:
    target = "my_model"

    def matches(self, transformer):
        return transformer.__class__.__name__ == "MyTransformer"

    def patch(self, transformer, checkpoint_state, quantization_config, options):
        # Rewrite modules, install hooks, or normalize checkpoint keys.
        return checkpoint_state
```

Register an adapter before calling `patch_transformer`:

```python
from nunchaku_lite import register_adapter

register_adapter(MyAdapter())
```

Model-specific code should stay inside adapters. Pipeline construction, scheduling, prompting, and image generation should remain standard Diffusers code.

### Adding a New Model Adapter

New models should be added as small adapter modules under `nunchaku_lite/adapters/`. The adapter should reuse the shared SVDQ helpers in `nunchaku_lite.adapters.common` for common quantization mechanics, and keep only model topology and forward-pass differences in the model-specific file.

Recommended structure:

```python
from nunchaku_lite import register_adapter
from nunchaku_lite.adapters.common import (
    build_svdq_context,
    finalize_svdq_checkpoint,
    prepare_transformer_dtype,
    svdq_from_linear,
)


class MyModelAdapter:
    target = "my_model"

    def matches(self, transformer):
        return transformer.__class__.__name__ == "MyTransformer"

    def patch(self, transformer, checkpoint_state, quantization_config, options):
        context = build_svdq_context(transformer, quantization_config, options)
        prepare_transformer_dtype(transformer, context)

        # Replace or wrap the model-specific modules here.
        transformer.block.attn.to_qkv = svdq_from_linear(transformer.block.attn.to_qkv, context)

        # Normalize checkpoint keys here only if this model's checkpoint layout needs it.
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        return checkpoint_state


register_adapter(MyModelAdapter())
```

Adapter responsibilities:

- Use `build_svdq_context`, `svdq_from_linear`, `patch_svdq_linears`, and `finalize_svdq_checkpoint` for rank, precision, dtype, scale-key patching, and fp16 checkpoint conversion.
- Keep graph-specific rewrites in the adapter, including QKV fusion, MLP fusion, module renaming, and any synthetic projection modules required to match checkpoint keys.
- Keep rotary embedding preparation, packed attention paths, KV-cache behavior, and custom forward wrappers model-specific.
- Add the adapter import in `nunchaku_lite.core._ensure_builtin_adapters()` if it should be built in.
- Add focused tests that build a tiny Diffusers transformer, patch it from a synthetic safetensors checkpoint, and verify expected module names and state dict keys.

Avoid adding a pipeline subclass for a new model unless the upstream Diffusers pipeline itself requires one. The preferred integration is still `patch_transformer(pipe.transformer, checkpoint, target="...")`.

## Benchmarking

The repository includes a benchmark that compares an unmodified Diffusers Z-Image pipeline with the `nunchaku_lite` patched transformer.

```bash
python benchmarks/benchmark_z_image.py \
  --model-id Tongyi-MAI/Z-Image-Turbo \
  --checkpoint nunchaku-ai/nunchaku-z-image-turbo/svdq-fp4_r128-z-image-turbo.safetensors \
  --precision fp4 \
  --dtype bf16 \
  --runs 3 \
  --warmup-runs 1
```

Outputs are written to `outputs/benchmark_z_image/` and include generated images plus a `summary.json` file with timing and CUDA memory statistics.

Flux benchmark:

```bash
python benchmarks/benchmark_flux.py \
  --model-id black-forest-labs/FLUX.1-schnell \
  --checkpoint nunchaku-ai/nunchaku-flux.1-schnell/svdq-fp4_r32-flux.1-schnell.safetensors \
  --precision fp4 \
  --dtype bf16 \
  --runs 3 \
  --warmup-runs 1
```

Flux2 benchmark:

```bash
python benchmarks/benchmark_flux2.py \
  --model-id tonera/FLUX.2-klein-9B-Nunchaku \
  --checkpoint tonera/FLUX.2-klein-9B-Nunchaku/svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors \
  --precision fp4 \
  --dtype bf16 \
  --runs 3 \
  --warmup-runs 1
```

## Development

Run unit tests:

```bash
pytest -q tests
```

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
  adapters/        Model-specific patch adapters
  csrc/            Python extension bindings
  models/          Lite runtime modules
  ops/             Python wrappers for native ops
native/            Vendored native kernel sources and headers
benchmarks/        End-to-end benchmark scripts
tests/             Unit tests
```

## License

`nunchaku_lite` is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

## Acknowledgements

`nunchaku_lite` builds on the Nunchaku project and uses selected native kernel code and third-party header-only dependencies needed for the lite runtime, including CUTLASS, nlohmann/json, mio, and spdlog. We are grateful to the maintainers and contributors of these projects.

## Notes

- `nunchaku_lite` does not import or require the full `nunchaku` Python package.
- Generated artifacts, local outputs, compiled extensions, caches, and virtual environments are intentionally ignored by git.
