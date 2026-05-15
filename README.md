<p align="center">
  <img src="assets/logo.svg" alt="nunchaku_lite" width="640">
</p>

<p align="center">
  <a href="https://github.com/rootonchair/nunchaku_lite/actions/workflows/cpu-tests.yml">
    <img src="https://github.com/rootonchair/nunchaku_lite/actions/workflows/cpu-tests.yml/badge.svg" alt="CPU Tests">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
  </a>
</p>

## About

`nunchaku_lite` brings Nunchaku-quantized image generation models to standard
Diffusers pipelines with a lean runtime, native CUDA kernels, and drop-in
pipeline loading. It is designed for fast SVDQ W4A4 inference with a small
integration surface: load a normal Diffusers pipeline, replace only the
quantized component, and keep the rest of the workflow
unchanged.

Its core features include:

- Efficient pipeline loading: constructs the patched component up front so
  Diffusers does not first materialize dense BF16 transformer or UNet weights
  that are immediately replaced.
- Native quantized kernels: packages the CUDA kernels and Python wrappers needed
  for INT4 and FP4 Nunchaku checkpoints.
- Broad image-model coverage: includes built-in adapters for FLUX.1, FLUX.2
  Klein, Qwen-Image, Qwen-Image-Edit, SDXL, SDXL-Turbo, and Z-Image Turbo.
- Runtime LoRA support: exposes Diffusers-style LoRA loading, adapter strength
  control, multi-LoRA composition, and unload/reset flows for supported model
  families.
- Extensible adapter registry: keeps model-specific graph rewrites isolated in
  small adapters without requiring pipeline subclasses or a dependency on the
  full `nunchaku` Python package.

## Supported Models

`nunchaku_lite` currently supports Diffusers pipelines whose transformer or UNet
component matches one of the built-in adapters below. Checkpoints should be SVDQ
W4A4 Nunchaku v2 `.safetensors` files in INT4 or FP4 format. Full runnable
guides live under `examples/`.

| Model family | Coverage | Runtime LoRA | Guide | Features |
| --- | --- | --- | --- | --- |
| FLUX.1 | FLUX.1-schnell and FLUX.1-dev | Yes | [examples/flux.md](examples/flux.md) | Pipeline loading, INT4/FP4 checkpoints, Diffusers-format LoRA loading, strength control, multi-LoRA composition, reset, and unload. |
| FLUX.2 Klein | FLUX.2 Klein | Yes | [examples/flux2.md](examples/flux2.md) | Pipeline loading, INT4/FP4 checkpoints, runtime LoRA, and ComfyUI Flux2 LoRA key conversion. |
| Qwen-Image and Qwen-Image-Edit | Qwen-Image, Qwen-Image-Lightning, Qwen-Image-Edit, and Qwen-Image-Edit-2509 | Yes | [examples/qwen_image.md](examples/qwen_image.md) | Pipeline loading, INT4/FP4 checkpoints, low-VRAM examples, Lightning LoRA workflows, and edit-pipeline examples. |
| SDXL and SDXL-Turbo | SDXL and SDXL-Turbo | Not yet | [examples/sdxl.md](examples/sdxl.md) | Pipeline loading for quantized UNet checkpoints. |
| Z-Image Turbo | Z-Image Turbo | Yes | [examples/z_image.md](examples/z_image.md) | Pipeline loading, INT4/FP4 checkpoints, runtime LoRA, and dense AdaLN modulation LoRA branches. |

The Qwen low-VRAM guides use `enable_model_cpu_offload()`, which requires
`accelerate`.

Additional model families should be added through the adapter registry rather
than pipeline subclasses. See [docs/roadmap.md](docs/roadmap.md) for planned
coverage and remaining feature work.

## Requirements

- Python 3.10 or newer
- PyTorch 2.7 or newer with CUDA
- CUDA toolkit with `nvcc`
- Ninja
- Diffusers 0.36 or newer
- Transformers 4.41.2 or newer
- PEFT
- Accelerate 0.31 or newer for CPU offload examples
- Hugging Face Hub, Safetensors, and Packaging

The build detects the local GPU architecture by default. Supported targets are
`sm75`, `sm80`, `sm86`, `sm89`, `sm120a`, and `sm121a`, subject to the
installed CUDA toolkit version.

The Python package metadata installs the runtime Python dependencies. The CUDA
toolkit and a compatible PyTorch CUDA build must already be available in the
environment before building from source.

## Installation

Install from source:

```bash
pip install .
```

Build and install a wheel:

```bash
python setup.py bdist_wheel
pip install dist/nunchaku_lite-*.whl
```

By default, the build uses `NUNCHAKU_INSTALL_MODE=FAST` and compiles for visible
local CUDA devices. To build all supported architectures:

```bash
NUNCHAKU_INSTALL_MODE=ALL pip install .
```

## Quick Start

Run from the repository root:

```python
from pathlib import Path

import torch
from diffusers import FluxPipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "black-forest-labs/FLUX.1-schnell"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "nunchaku-ai/nunchaku-flux.1-schnell/svdq-int4_r32-flux.1-schnell.safetensors",
    "fp4": "nunchaku-ai/nunchaku-flux.1-schnell/svdq-fp4_r32-flux.1-schnell.safetensors",
}
output_path = Path(f"outputs/flux_schnell_nunchaku_lite_{precision}.png")

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=FluxPipeline,
    checkpoint=checkpoints[precision],
    target="flux",
    precision=precision,
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

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```

### Runtime LoRA

Pipelines loaded with `load_nunchaku_pipeline` expose Diffusers-style LoRA
methods when the selected adapter supports runtime LoRA:

```python
from pathlib import Path

import torch
from diffusers import FluxPipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "black-forest-labs/FLUX.1-dev"
precision = "fp4"
checkpoint = "nunchaku-tech/nunchaku-flux.1-dev/svdq-fp4_r32-flux.1-dev.safetensors"
output_path = Path("outputs/flux_dev_ghibsky_lora_fp4.png")

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=FluxPipeline,
    checkpoint=checkpoint,
    target="flux",
    precision=precision,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

pipe.load_lora_weights(
    "aleksa-codes/flux-ghibsky-illustration",
    weight_name="lora.safetensors",
    adapter_name="ghibsky",
)
pipe.set_adapters("ghibsky", adapter_weights=0.75)

image = pipe(
    "GHIBSKY style painting of a cozy mountain cabin beside a clear lake at sunset",
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=torch.Generator(device="cpu").manual_seed(12345),
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```

Checkpoint paths can be local `.safetensors` files or Hugging Face paths of the
form:

```text
org-or-user/repo-name/path/to/checkpoint.safetensors
```

## Documentation

| Topic | Link |
| --- | --- |
| Public API and runtime LoRA usage | [docs/api.md](docs/api.md) |
| Development, testing, and adapter authoring | [docs/development.md](docs/development.md) |
| Supported models and feature backlog | [docs/roadmap.md](docs/roadmap.md) |
| Benchmarks | [benchmarks/README.md](benchmarks/README.md) |

## Public API

```python
from nunchaku_lite import (
    TransformerAdapter,
    list_adapters,
    load_nunchaku_pipeline,
    patch_transformer,
    register_adapter,
)
```

`load_nunchaku_pipeline(...)` is the preferred entry point for normal pipeline
loading. Use `patch_transformer(...)` only when a Diffusers component has already
been constructed and needs in-place patching.

See [docs/api.md](docs/api.md) for argument details, runtime LoRA examples, and
adapter registry usage.

## Development

Run the unit tests:

```bash
pytest -q tests
```

Build the extension in place:

```bash
python setup.py build_ext --inplace
```

See [docs/development.md](docs/development.md) for full inference tests, adapter
authoring guidance, runtime LoRA implementation notes, and repository layout.

## License

`nunchaku_lite` is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE).

## Acknowledgements

`nunchaku_lite` builds on the Nunchaku project and uses selected native kernel
code for the lite runtime. The vendored native support code includes spdlog and
its bundled fmt headers. We are grateful to the maintainers and contributors of
these projects.
