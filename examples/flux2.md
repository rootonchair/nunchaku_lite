# FLUX.2 Klein INT4 / FP4

Example for `tonera/FLUX.2-klein-9B-Nunchaku`.

Set `precision = "int4"` or `precision = "fp4"` in the script.

Run from the repository root:

```python
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

from nunchaku_lite import patch_transformer


model_id = "tonera/FLUX.2-klein-9B-Nunchaku"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "tonera/FLUX.2-klein-9B-Nunchaku/svdq-int4_r32-FLUX.2-klein-9B-Nunchaku.safetensors",
    "fp4": "tonera/FLUX.2-klein-9B-Nunchaku/svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors",
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/flux2_klein_nunchaku_lite_{precision}.png")

pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
patch_transformer(
    pipe.transformer,
    checkpoint,
    target="flux2",
    precision=precision,
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

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```
