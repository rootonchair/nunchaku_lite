# Z-Image Turbo INT4 / FP4

Example for `Tongyi-MAI/Z-Image-Turbo`.

Set `precision = "int4"` or `precision = "fp4"` in the script.

Run from the repository root:

```python
from pathlib import Path

import torch
from diffusers import ZImagePipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "Tongyi-MAI/Z-Image-Turbo"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "nunchaku-ai/nunchaku-z-image-turbo/svdq-int4_r128-z-image-turbo.safetensors",
    "fp4": "nunchaku-ai/nunchaku-z-image-turbo/svdq-fp4_r128-z-image-turbo.safetensors",
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/z_image_nunchaku_lite_{precision}.png")

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=ZImagePipeline,
    checkpoint=checkpoint,
    precision=precision,
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

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```

## Z-Image Turbo Runtime LoRA

Patched Z-Image transformers expose the same runtime LoRA API as FLUX and Qwen-Image. The runtime supports the
quantized attention/feed-forward projections and dense AdaLN modulation LoRA branches used by Z-Image style LoRAs.

```python
from pathlib import Path

import torch
from diffusers import ZImagePipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "Tongyi-MAI/Z-Image-Turbo"
precision = "int4"
checkpoint = "nunchaku-ai/nunchaku-z-image-turbo/svdq-int4_r128-z-image-turbo.safetensors"
output_path = Path("outputs/z_image_turbo_pixelart_lora_int4.png")

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=ZImagePipeline,
    checkpoint=checkpoint,
    target="z_image",
    precision=precision,
    torch_dtype=torch.bfloat16,
    device="cuda",
)
pipe = pipe.to("cuda")

pipe.load_lora_weights(
    "tarn59/pixel_art_style_lora_z_image_turbo",
    weight_name="pixel_art_style_z_image_turbo.safetensors",
    adapter_name="pixelart",
)
pipe.set_adapters("pixelart", adapter_weights=1.0)

image = pipe(
    prompt="Pixel art style. a cozy fantasy castle village at sunset, warm windows, river, detailed 16-bit game art",
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=0.0,
    generator=torch.Generator(device="cuda").manual_seed(12345),
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```
