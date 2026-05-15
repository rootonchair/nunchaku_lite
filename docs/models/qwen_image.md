# Qwen-Image INT4 / FP4

Low-VRAM example for `Qwen/Qwen-Image`. This loads the pipeline with a patched Nunchaku transformer so Diffusers does not load dense BF16 transformer weights first.

Requires `accelerate` for `enable_model_cpu_offload()`.

Set `precision = "int4"` or `precision = "fp4"` in the script.

Run from the repository root:

```python
from pathlib import Path

import torch
from diffusers import QwenImagePipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "Qwen/Qwen-Image"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "nunchaku-tech/nunchaku-qwen-image/svdq-int4_r32-qwen-image.safetensors",
    "fp4": "nunchaku-tech/nunchaku-qwen-image/svdq-fp4_r32-qwen-image.safetensors",
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/qwen_image_{precision}_low_vram.png")

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=QwenImagePipeline,
    checkpoint=checkpoint,
    target="qwen_image",
    precision=precision,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.",
    "zh": "超清，4K，电影级构图",
}
prompt = """Bookstore window display. A sign displays “New Arrivals This Week”. Below, a shelf tag with the text “Best-Selling Novels Here”. To the side, a colorful poster advertises “Author Meet And Greet on Saturday” with a central portrait of the author. There are four books on the bookshelf, namely “The light between worlds” “When stars are scattered” “The slient patient” “The night circus”"""

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=" ",
    width=1664,
    height=928,
    num_inference_steps=50,
    true_cfg_scale=4.0,
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```

## Qwen-Image Lightning Runtime LoRA

Use the same base Nunchaku Qwen checkpoint and load the Lightning adapter at runtime. The scheduler and `true_cfg_scale=1.0` match the LightX2V Qwen-Image-Lightning Diffusers example.

```python
import math
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "Qwen/Qwen-Image"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "nunchaku-tech/nunchaku-qwen-image/svdq-int4_r32-qwen-image.safetensors",
    "fp4": "nunchaku-tech/nunchaku-qwen-image/svdq-fp4_r32-qwen-image.safetensors",
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/qwen_image_lightning_lora_{precision}.png")
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=QwenImagePipeline,
    checkpoint=checkpoint,
    target="qwen_image",
    precision=precision,
    scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config),
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
    adapter_name="lightning",
)
pipe.set_adapters("lightning", adapter_weights=1.0)

image = pipe(
    prompt="a tiny astronaut hatching from an egg on the moon, Ultra HD, 4K, cinematic composition.",
    negative_prompt=" ",
    width=1024,
    height=1024,
    num_inference_steps=4,
    true_cfg_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(0),
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
pipe.unload_lora_weights()
print(f"saved {output_path}")
```

Qwen-Image-Edit Lightning LoRAs use the same API with `QwenImageEditPlusPipeline` and the edit Lightning weights from `lightx2v/Qwen-Image-Lightning`, for example `Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`.

## Qwen-Image-Edit-2509 INT4 / FP4 Base

Low-VRAM edit example for the base INT4 or FP4 checkpoint from `nunchaku-ai/nunchaku-qwen-image-edit-2509`.

Requires `accelerate` for `enable_model_cpu_offload()`.

Set `precision = "int4"` or `precision = "fp4"` in the script.

Run from the repository root:

```python
from pathlib import Path

import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku_lite import load_nunchaku_pipeline


model_id = "Qwen/Qwen-Image-Edit-2509"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "nunchaku-ai/nunchaku-qwen-image-edit-2509/svdq-int4_r32-qwen-image-edit-2509.safetensors",
    "fp4": "nunchaku-ai/nunchaku-qwen-image-edit-2509/svdq-fp4_r32-qwen-image-edit-2509.safetensors",
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/qwen_image_edit_2509_{precision}/base_{precision}.png")
prompt = "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep."
image_urls = [
    "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/man.png",
    "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png",
    "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png",
]

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=QwenImageEditPlusPipeline,
    checkpoint=checkpoint,
    target="qwen_image",
    precision=precision,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

images = [load_image(url).convert("RGB") for url in image_urls]
image = pipe(
    image=images,
    prompt=prompt,
    negative_prompt=" ",
    true_cfg_scale=4.0,
    num_inference_steps=40,
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```

## Qwen-Image-Edit-2509 INT4 / FP4 Lightning 4-Step

Low-VRAM edit example for the 4-step distilled INT4 or FP4 checkpoint from `nunchaku-ai/nunchaku-qwen-image-edit-2509`.

Requires `accelerate` for `enable_model_cpu_offload()`.

Set `precision = "int4"` or `precision = "fp4"` in the script.

Run from the repository root:

```python
import math
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku_lite import load_nunchaku_pipeline


model_id = "Qwen/Qwen-Image-Edit-2509"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": (
        "nunchaku-ai/nunchaku-qwen-image-edit-2509/"
        "lightning-251115/svdq-int4_r32-qwen-image-edit-2509-lightning-4steps-251115.safetensors"
    ),
    "fp4": (
        "nunchaku-ai/nunchaku-qwen-image-edit-2509/"
        "lightning-251115/svdq-fp4_r32-qwen-image-edit-2509-lightning-4steps-251115.safetensors"
    ),
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/qwen_image_edit_2509_{precision}/lightning_4_{precision}.png")
prompt = "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep."
image_urls = [
    "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/man.png",
    "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png",
    "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png",
]
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=QwenImageEditPlusPipeline,
    checkpoint=checkpoint,
    target="qwen_image",
    precision=precision,
    scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config),
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

images = [load_image(url).convert("RGB") for url in image_urls]
image = pipe(
    image=images,
    prompt=prompt,
    true_cfg_scale=1.0,
    num_inference_steps=4,
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```

## Qwen-Image-Edit-2509 INT4 / FP4 Lightning 8-Step

Low-VRAM edit example for the 8-step distilled INT4 or FP4 checkpoint from `nunchaku-ai/nunchaku-qwen-image-edit-2509`.

Requires `accelerate` for `enable_model_cpu_offload()`.

Set `precision = "int4"` or `precision = "fp4"` in the script.

Run from the repository root:

```python
import math
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku_lite import load_nunchaku_pipeline


model_id = "Qwen/Qwen-Image-Edit-2509"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": (
        "nunchaku-ai/nunchaku-qwen-image-edit-2509/"
        "lightning-251115/svdq-int4_r32-qwen-image-edit-2509-lightning-8steps-251115.safetensors"
    ),
    "fp4": (
        "nunchaku-ai/nunchaku-qwen-image-edit-2509/"
        "lightning-251115/svdq-fp4_r32-qwen-image-edit-2509-lightning-8steps-251115.safetensors"
    ),
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/qwen_image_edit_2509_{precision}/lightning_8_{precision}.png")
prompt = "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep."
image_urls = [
    "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/man.png",
    "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png",
    "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png",
]
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=QwenImageEditPlusPipeline,
    checkpoint=checkpoint,
    target="qwen_image",
    precision=precision,
    scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config),
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

images = [load_image(url).convert("RGB") for url in image_urls]
image = pipe(
    image=images,
    prompt=prompt,
    true_cfg_scale=1.0,
    num_inference_steps=8,
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```
