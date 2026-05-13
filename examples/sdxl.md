# SDXL / SDXL-Turbo INT4

Examples for `stabilityai/stable-diffusion-xl-base-1.0` and `stabilityai/sdxl-turbo`.

Run from the repository root.

## SDXL Base

```python
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from nunchaku_lite import patch_transformer


model_id = "stabilityai/stable-diffusion-xl-base-1.0"
checkpoint = "nunchaku-ai/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors"
output_path = Path("outputs/sdxl_nunchaku_lite_int4.png")

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="fp16",
)
patch_transformer(
    pipe.unet,
    checkpoint,
    target="sdxl",
    precision="int4",
    torch_dtype=torch.bfloat16,
    device="cuda",
)
pipe = pipe.to("cuda")

image = pipe(
    prompt="A cinematic shot of a glass greenhouse full of tropical plants during golden hour.",
    guidance_scale=5.0,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(12345),
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```

## SDXL-Turbo

```python
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from nunchaku_lite import patch_transformer


model_id = "stabilityai/sdxl-turbo"
checkpoint = "nunchaku-ai/nunchaku-sdxl-turbo/svdq-int4_r32-sdxl-turbo.safetensors"
output_path = Path("outputs/sdxl_turbo_nunchaku_lite_int4.png")

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    variant="fp16",
)
patch_transformer(
    pipe.unet,
    checkpoint,
    target="sdxl",
    precision="int4",
    torch_dtype=torch.bfloat16,
    device="cuda",
)
pipe = pipe.to("cuda")

image = pipe(
    prompt="A cinematic shot of a glass greenhouse full of tropical plants during golden hour.",
    guidance_scale=0.0,
    num_inference_steps=4,
    generator=torch.Generator(device="cuda").manual_seed(12345),
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```
