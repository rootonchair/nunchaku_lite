<p align="center">
  <img src="assets/logo.svg" alt="nunchaku_lite" width="640">
</p>

`nunchaku_lite` is a small, plugin-oriented runtime package for applying Nunchaku v2 quantized transformer and UNet weights to Diffusers pipelines. The preferred loader injects the patched Nunchaku component while the pipeline is created, so Diffusers does not load unused dense BF16 transformer or UNet weights first.

The first built-in adapters target Flux, Flux2, Qwen-Image, and Z-Image transformer classes plus SDXL UNet with SVDQ W4A4 checkpoints.

## Design Goals

- Keep the public integration surface model-agnostic.
- Load pipelines without first materializing unused dense transformer or UNet weights.
- Keep low-level in-place patching available for advanced use.
- Use a registry of small adapters for model-specific graph rewrites.
- Package only the native kernels and Python code required for the lite runtime.
- Avoid a hard dependency on the original `nunchaku` Python package.

## Status

This package is an early lite runtime. The current built-in adapter set is:

| Adapter | Target | Status |
| --- | --- | --- |
| `flux` | Diffusers `FluxTransformer2DModel` | Implemented |
| `flux2` | Diffusers `Flux2Transformer2DModel` | Implemented |
| `qwen_image` | Diffusers `QwenImageTransformer2DModel` | Implemented |
| `sdxl` | Diffusers `UNet2DConditionModel` | Implemented |
| `z_image` | Diffusers `ZImageTransformer2DModel` | Implemented |

Additional model families should be added through the common adapter registry rather than through pipeline-specific subclasses.

### Feature Backlog from Original `nunchaku`

The full `nunchaku` package in this repository exposes a broader set of model-specific loaders and workflow integrations. Use this checklist as the current porting backlog for `nunchaku_lite`:

- [x] FLUX.1 transformer adapter for Diffusers `FluxTransformer2DModel`.
- [x] Flux2 transformer adapter for Diffusers `Flux2Transformer2DModel`.
- [x] Qwen-Image transformer adapter based on `NunchakuQwenImageTransformer2DModel`, covering Qwen-Image, Qwen-Image-Lightning, Qwen-Image-Edit, Qwen-Image-Edit-2509, and Qwen-Image ControlNet examples.
- [x] Z-Image transformer adapter for Diffusers `ZImageTransformer2DModel`.
- [ ] Sana transformer adapter based on `NunchakuSanaTransformer2DModel`, covering Sana 1.6B and Sana PAG examples.
- [x] SDXL UNet adapter based on `NunchakuSDXLUNet2DConditionModel`, covering SDXL and SDXL-Turbo examples.
- [ ] Quantized T5 text encoder support based on `NunchakuT5EncoderModel`.
- [x] FLUX runtime LoRA support, including Diffusers-format conversion, Nunchaku-format loading, strength control, reset, and multi-LoRA composition.
- [ ] Qwen-Image runtime LoRA support, covering Qwen-Image and Qwen-Image-Edit families after LoRA key mapping is defined.
- [ ] Flux2 runtime LoRA support, reusing FLUX conversion patterns where compatible.
- [ ] SDXL runtime LoRA support for quantized UNet attention and MLP projections.
- [ ] Z-Image runtime LoRA support for quantized transformer projections.
- [ ] FLUX IP-Adapter integration.
- [ ] FLUX PuLID pipeline or patching support.
- [ ] FLUX ControlNet workflow coverage for Canny, Depth, Fill, and ControlNet-Union variants.
- [ ] Caching integrations equivalent to TeaCache, first-block cache, double-block cache, and DiT cache examples.
- [ ] Async/offload paths for lower-VRAM inference where supported by the original implementation.

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

## Examples

Full quick-start scripts live under `examples/` so the main README stays focused on the runtime API and adapter model.

| Model | Example | Notes |
| --- | --- | --- |
| Qwen-Image INT4 / FP4 | [examples/qwen_image.md](examples/qwen_image.md) | Qwen-Image plus Qwen-Image-Edit-2509 base, 4-step distilled, and 8-step distilled examples. |
| Z-Image Turbo INT4 / FP4 | [examples/z_image.md](examples/z_image.md) | Pipeline loader flow. |
| FLUX.1-schnell INT4 / FP4 | [examples/flux.md](examples/flux.md) | Pipeline loader plus FLUX LoRA examples. |
| FLUX.2 Klein INT4 / FP4 | [examples/flux2.md](examples/flux2.md) | Pipeline loader flow. |
| SDXL / SDXL-Turbo INT4 | [examples/sdxl.md](examples/sdxl.md) | Pipeline loader flow for a quantized UNet. |

The Qwen low-VRAM examples use `enable_model_cpu_offload()`, which requires `accelerate`.

Checkpoint paths can be local `.safetensors` files or Hugging Face paths of the form:

```text
org-or-user/repo-name/path/to/checkpoint.safetensors
```

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

### `load_nunchaku_pipeline`

```python
pipe = load_nunchaku_pipeline(
    "black-forest-labs/FLUX.1-schnell",
    pipeline_cls=FluxPipeline,
    checkpoint="nunchaku-ai/nunchaku-flux.1-schnell/svdq-fp4_r32-flux.1-schnell.safetensors",
    target="flux",
    precision="fp4",
    torch_dtype=torch.bfloat16,
    device="cuda",
)
```

`load_nunchaku_pipeline` is the preferred public API. It reads the pipeline config, constructs the selected `transformer` or `unet` on the meta device, patches it with the Nunchaku adapter, loads the quantized checkpoint with `assign=True`, and passes the patched component into `pipeline_cls.from_pretrained(...)`. Diffusers then loads the rest of the pipeline normally while skipping the original dense component.

Arguments:

- `pretrained_model_name_or_path`: Diffusers pipeline model id or local path.
- `pipeline_cls`: Diffusers pipeline class, such as `FluxPipeline` or `StableDiffusionXLPipeline`.
- `checkpoint`: local or Hugging Face `.safetensors` checkpoint path.
- `target`: adapter name, or `"auto"` to select the only matching adapter.
- `component`: optional `"transformer"` or `"unet"` override. Auto-selection prefers `transformer`, then `unet`.
- `precision`, `torch_dtype`, `device`, `strict`, and `adapter_options`: same patching controls as `patch_transformer`.
- `bind_lora`: binds supported pipeline-level runtime APIs. FLUX LoRA APIs are bound by default.
- additional keyword arguments are forwarded to `pipeline_cls.from_pretrained(...)`.

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

- `transformer`: the Diffusers transformer or UNet module to patch.
- `checkpoint`: local or Hugging Face `.safetensors` checkpoint path.
- `target`: adapter name, or `"auto"` to select the only matching adapter.
- `precision`: `"auto"`, `"fp4"`, or `"int4"`. Internally, `"fp4"` maps to NVFP4 kernels.
- `torch_dtype`: optional model dtype, typically `torch.bfloat16` or `torch.float16`.
- `device`: optional destination device after patching.
- `strict`: forwarded to `load_state_dict`.
- `adapter_options`: model-specific adapter options.

This is the low-level compatibility API for callers that already constructed a component. Prefer `load_nunchaku_pipeline` for normal pipeline loading because it avoids loading dense weights that are immediately replaced. The function is idempotent for the same target. A transformer patched once will be returned unchanged if patched again with the same target.

### FLUX runtime LoRA

Patched FLUX transformers expose runtime LoRA methods:

```python
pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=FluxPipeline,
    checkpoint=checkpoint,
    target="flux",
    torch_dtype=torch.bfloat16,
)

pipe.load_lora_weights("artist-style.safetensors", adapter_name="artist")
pipe.set_adapters("artist", adapter_weights=0.5)
pipe.unload_lora_weights()
```

`load_lora` accepts Diffusers-format FLUX LoRAs and Nunchaku-format low-rank tensors. Multiple LoRAs can be active at once; they are recomposed from the original checkpoint low-rank state when strengths change or one LoRA is reset.

`load_nunchaku_pipeline` binds Diffusers-style pipeline LoRA methods for FLUX automatically. Advanced callers using `patch_transformer` directly can still bind those methods manually with `nunchaku_lite.lora.bind_flux_pipeline_lora_methods(pipe)`.

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

Register an adapter before calling `load_nunchaku_pipeline` or `patch_transformer`:

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
            },
            custom_attention_converters={
                MyAttentionSubclass: lambda attention: MyLiteAttention(attention, context=context),
            },
        )

        # Normalize checkpoint keys here only if this model's checkpoint layout needs it.
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        return checkpoint_state


register_adapter(MyModelAdapter())
```

Adapter responsibilities:

- Use `build_svdq_context`, `patch_modules_recursively`, `svdq_from_linear`, `patch_svdq_linears`, and `finalize_svdq_checkpoint` for rank, precision, dtype, recursive module replacement, scale-key patching, and fp16 checkpoint conversion.
- Keep graph-specific rewrites in the adapter, including QKV fusion, MLP fusion, module renaming, and any synthetic projection modules required to match checkpoint keys.
- Keep rotary embedding preparation, packed attention paths, KV-cache behavior, and custom forward wrappers model-specific.
- Add the adapter import in `nunchaku_lite.core._ensure_builtin_adapters()` if it should be built in.
- Add focused tests that build a tiny Diffusers transformer, patch it from a synthetic safetensors checkpoint, and verify expected module names and state dict keys.

`patch_modules_recursively` mutates the selected module tree in place and returns a `ModulePatchReport` with replacement and skip counts. Use `linear_filter` or narrow roots so only checkpoint-backed dense projections are replaced. Use `module_converters` for exact-class model blocks that can be normalized before descending. Use `custom_attention_converters` for Diffusers `Attention` subclasses that require a model-specific replacement; unsupported attention subclasses now raise `TypeError` instead of being silently skipped.

Avoid adding a pipeline subclass for a new model unless the upstream Diffusers pipeline itself requires one. The preferred integration is `load_nunchaku_pipeline(model_id, pipeline_cls=..., checkpoint=..., target="...")`.

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

Qwen-Image benchmark:

```bash
python benchmarks/benchmark_qwen_image.py \
  --model-id Qwen/Qwen-Image \
  --checkpoint nunchaku-tech/nunchaku-qwen-image/svdq-fp4_r32-qwen-image.safetensors \
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

Run the opt-in FLUX.1-dev full inference test:

```bash
NUNCHAKU_LITE_RUN_FULL_INFERENCE=1 \
PYTHONPATH=. pytest -q -m full_inference tests/test_full_inference_flux.py
```

The full inference test requires CUDA, model access, and enough VRAM or offload memory for FLUX.1-dev. It exercises `load_nunchaku_pipeline`, baseline generation, Diffusers-style FLUX LoRA loading, strength changes, multi-LoRA composition with Ghibsky plus Canopus UltraRealism, delete/reset, and unload. Generated images are written to pytest's temp directory by default; set `NUNCHAKU_LITE_FULL_INFERENCE_OUTPUT_DIR=outputs/full_inference_flux` to keep them.

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
