# API Reference

`nunchaku_lite` exposes a small public API for loading quantized Nunchaku
components into Diffusers pipelines and for registering model-specific adapters.

```python
from nunchaku_lite import (
    TransformerAdapter,
    list_adapters,
    load_nunchaku_pipeline,
    patch_transformer,
    register_adapter,
)
```

## `load_nunchaku_pipeline`

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

`load_nunchaku_pipeline` is the preferred public API. It reads the pipeline
config, constructs the selected `transformer` or `unet` on the meta device,
patches it with the Nunchaku adapter, loads the quantized checkpoint with
`assign=True`, and passes the patched component into
`pipeline_cls.from_pretrained(...)`. Diffusers then loads the rest of the
pipeline normally while skipping the original dense component. Adapters that
provide pipeline runtime APIs patch the loaded pipeline automatically.

Arguments:

- `pretrained_model_name_or_path`: Diffusers pipeline model id or local path.
- `pipeline_cls`: Diffusers pipeline class, such as `FluxPipeline` or
  `StableDiffusionXLPipeline`.
- `checkpoint`: local or Hugging Face `.safetensors` checkpoint path.
- `target`: adapter name, or `"auto"` to select the only matching adapter.
- `component`: optional `"transformer"` or `"unet"` override. Auto-selection
  prefers `transformer`, then `unet`.
- `precision`, `torch_dtype`, `device`, `strict`, and `adapter_options`: same
  patching controls as `patch_transformer`.
- Additional keyword arguments are forwarded to `pipeline_cls.from_pretrained(...)`.

## `patch_transformer`

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
- `precision`: `"auto"`, `"fp4"`, or `"int4"`. Internally, `"fp4"` maps to
  NVFP4 kernels.
- `torch_dtype`: optional model dtype, typically `torch.bfloat16` or
  `torch.float16`.
- `device`: optional destination device after patching.
- `strict`: forwarded to `load_state_dict`.
- `adapter_options`: model-specific adapter options.

This is the low-level compatibility API for callers that already constructed a
component. Prefer `load_nunchaku_pipeline` for normal pipeline loading because it
avoids loading dense weights that are immediately replaced. The function is
idempotent for the same target. A transformer patched once will be returned
unchanged if patched again with the same target.

## Runtime LoRA

Pipelines loaded through `load_nunchaku_pipeline` expose Diffusers-style runtime
LoRA methods when their adapter provides pipeline support:

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

Runtime LoRA loading accepts Diffusers-format LoRAs and Nunchaku-format low-rank
tensors for supported adapters. Use pipeline-level `load_lora_weights` on
pipelines loaded with `load_nunchaku_pipeline`, or transformer-level `load_lora`
on directly patched components. Multiple LoRAs can be active at once; they are
recomposed from the original checkpoint low-rank state when strengths change or
one LoRA is reset.

Advanced callers using `patch_transformer` directly can bind pipeline LoRA
methods manually:

```python
from nunchaku_lite.lora.core.runtime import NunchakuPipelineLoraMixin, bind_pipeline_lora_methods

bind_pipeline_lora_methods(pipe, NunchakuPipelineLoraMixin)
```

## Adapter Registry

Adapters implement a small protocol:

```python
class MyAdapter:
    target = "my_model"

    def matches(self, transformer):
        return transformer.__class__.__name__ == "MyTransformer"

    def patch(self, transformer, checkpoint_state, quantization_config, options):
        # Rewrite modules, install hooks, or normalize checkpoint keys.
        return checkpoint_state

    def patch_pipeline(self, pipeline, *, component_name="transformer", component=None):
        # Optional: attach pipeline-level runtime APIs.
        return None
```

Register an adapter before calling `load_nunchaku_pipeline` or
`patch_transformer`:

```python
from nunchaku_lite import register_adapter

register_adapter(MyAdapter())
```

Model-specific code should stay inside adapters. Pipeline construction,
scheduling, prompting, and image generation should remain standard Diffusers
code.
