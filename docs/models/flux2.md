# FLUX.2 Klein INT4 / FP4

Example for `tonera/FLUX.2-klein-9B-Nunchaku`.

Set `precision = "int4"` or `precision = "fp4"` in the script. On pre-Blackwell GPUs such as RTX A5000, prefer the INT4 checkpoint.

Run from the repository root:

```python
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "tonera/FLUX.2-klein-9B-Nunchaku"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "tonera/FLUX.2-klein-9B-Nunchaku/svdq-int4_r32-FLUX.2-klein-9B-Nunchaku.safetensors",
    "fp4": "tonera/FLUX.2-klein-9B-Nunchaku/svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors",
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/flux2_klein_nunchaku_lite_{precision}.png")

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=Flux2KleinPipeline,
    checkpoint=checkpoint,
    target="flux2",
    precision=precision,
    torch_dtype=torch.bfloat16,
    device="cuda",
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(12345),
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```

## FLUX.2 Klein Runtime LoRA

Patched FLUX.2 transformers expose the same runtime LoRA API as FLUX.1 and Qwen-Image. ComfyUI-format FLUX.2 LoRAs with `diffusion_model.double_blocks.*` and `diffusion_model.single_blocks.*` keys are converted to patched Nunchaku module names at load time.

```python
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "tonera/FLUX.2-klein-9B-Nunchaku"
precision = "int4"
checkpoint = "tonera/FLUX.2-klein-9B-Nunchaku/svdq-int4_r32-FLUX.2-klein-9B-Nunchaku.safetensors"
output_path = Path("outputs/flux2_klein_pixelart_lora_int4.png")

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=Flux2KleinPipeline,
    checkpoint=checkpoint,
    target="flux2",
    precision=precision,
    torch_dtype=torch.bfloat16,
    device="cuda",
)
pipe.enable_model_cpu_offload()

pipe.load_lora_weights(
    "artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B",
    weight_name="[FLUX.2.Klein]PixelArt_Redmond.safetensors",
    adapter_name="pixelart",
)
pipe.set_adapters("pixelart", adapter_weights=0.8)

image = pipe(
    prompt="Pixel Art, PixArFK, a tiny knight beside a glowing arcade machine in a forest",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(12345),
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```
