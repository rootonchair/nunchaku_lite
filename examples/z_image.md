# Z-Image Turbo INT4 / FP4

Example for `Tongyi-MAI/Z-Image-Turbo`.

Set `precision = "int4"` or `precision = "fp4"` in the script.

Run from the repository root:

```bash
python - <<'PY'
from pathlib import Path

import torch
from diffusers import ZImagePipeline

from nunchaku_lite import patch_transformer


model_id = "Tongyi-MAI/Z-Image-Turbo"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "nunchaku-ai/nunchaku-z-image-turbo/svdq-int4_r128-z-image-turbo.safetensors",
    "fp4": "nunchaku-ai/nunchaku-z-image-turbo/svdq-fp4_r128-z-image-turbo.safetensors",
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/z_image_nunchaku_lite_{precision}.png")

pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
patch_transformer(
    pipe.transformer,
    checkpoint,
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
PY
```
