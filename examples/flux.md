# FLUX.1-schnell INT4 / FP4

Example for `black-forest-labs/FLUX.1-schnell`.

Set `precision = "int4"` or `precision = "fp4"` in the script.

Run from the repository root:

```python
from pathlib import Path

import torch
from diffusers import FluxPipeline

from nunchaku_lite import patch_transformer


model_id = "black-forest-labs/FLUX.1-schnell"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "nunchaku-ai/nunchaku-flux.1-schnell/svdq-int4_r32-flux.1-schnell.safetensors",
    "fp4": "nunchaku-ai/nunchaku-flux.1-schnell/svdq-fp4_r32-flux.1-schnell.safetensors",
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/flux_schnell_nunchaku_lite_{precision}.png")

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
patch_transformer(
    pipe.transformer,
    checkpoint,
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
