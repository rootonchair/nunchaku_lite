# FLUX.1-schnell INT4 / FP4

Example for `black-forest-labs/FLUX.1-schnell`.

Set `precision = "int4"` or `precision = "fp4"` in the script.

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
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/flux_schnell_nunchaku_lite_{precision}.png")

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=FluxPipeline,
    checkpoint=checkpoint,
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

## FLUX.1-dev Runtime LoRA

Patched FLUX transformers expose `load_lora`, `set_lora_strength`, and
`reset_lora`. `load_lora` accepts Diffusers-format FLUX LoRAs and
Nunchaku-format low-rank tensors. Multiple LoRAs can be active at the same
time; they are recomposed from the original checkpoint low-rank branch when a
strength changes or one adapter is reset.

```python
from pathlib import Path

import torch
from diffusers import FluxPipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "black-forest-labs/FLUX.1-dev"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "nunchaku-tech/nunchaku-flux.1-dev/svdq-int4_r32-flux.1-dev.safetensors",
    "fp4": "nunchaku-tech/nunchaku-flux.1-dev/svdq-fp4_r32-flux.1-dev.safetensors",
}

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=FluxPipeline,
    checkpoint=checkpoints[precision],
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

output_path = Path(f"outputs/flux_dev_ghibsky_{precision}.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)

pipe.unload_lora_weights()
```

### Multiple LoRAs

Load each adapter with a stable name, then update or remove one without
disturbing the others.

```python
pipe.load_lora_weights(
    "aleksa-codes/flux-ghibsky-illustration",
    weight_name="lora.safetensors",
    adapter_name="ghibsky",
)
pipe.load_lora_weights(
    "prithivMLmods/Canopus-LoRA-Flux-UltraRealism-2.0",
    weight_name="Canopus-LoRA-Flux-UltraRealism.safetensors",
    adapter_name="realism",
)

pipe.set_adapters(["ghibsky", "realism"], adapter_weights=[0.65, 0.35])
pipe.set_adapters(["ghibsky", "realism"], adapter_weights=[0.65, 0.5])
pipe.delete_adapters("realism")  # leaves "ghibsky" loaded and active
pipe.unload_lora_weights()  # removes all runtime LoRAs
```
