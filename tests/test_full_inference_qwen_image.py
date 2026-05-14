import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from nunchaku_lite.utils import get_precision


pytestmark = pytest.mark.full_inference


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _image_stats(image) -> dict[str, float]:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    adjacent_x = np.abs(array[:, 1:] - array[:, :-1]).mean()
    adjacent_y = np.abs(array[1:] - array[:-1]).mean()
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "adjacent_diff": float((adjacent_x + adjacent_y) / 2),
    }


def _assert_image_has_signal(image, name: str) -> None:
    stats = _image_stats(image)
    assert image.size == (512, 512)
    assert stats["max"] > stats["min"], f"{name} is constant: {stats}"
    assert stats["std"] > 3.0, f"{name} has unexpectedly low variation: {stats}"
    assert stats["adjacent_diff"] < 35.0, f"{name} looks like high-frequency noise: {stats}"


def _scheduler_config() -> dict:
    return {
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


@pytest.mark.skipif(
    os.environ.get("NUNCHAKU_LITE_RUN_FULL_INFERENCE") != "1",
    reason="set NUNCHAKU_LITE_RUN_FULL_INFERENCE=1 to run GPU/network full inference",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="full inference requires CUDA")
def test_qwen_image_lightning_runtime_lora_full_inference(tmp_path):
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

    from nunchaku_lite import load_nunchaku_pipeline

    model_id = _env("NUNCHAKU_LITE_QWEN_FULL_INFERENCE_MODEL_ID", "Qwen/Qwen-Image")
    precision = _env("NUNCHAKU_LITE_QWEN_FULL_INFERENCE_PRECISION", get_precision(device="cuda"))
    checkpoint = _env(
        "NUNCHAKU_LITE_QWEN_FULL_INFERENCE_CHECKPOINT",
        f"nunchaku-tech/nunchaku-qwen-image/svdq-{precision}_r32-qwen-image.safetensors",
    )
    lora_repo = _env("NUNCHAKU_LITE_QWEN_FULL_INFERENCE_LORA_REPO", "lightx2v/Qwen-Image-Lightning")
    lora_weight = _env(
        "NUNCHAKU_LITE_QWEN_FULL_INFERENCE_LORA_WEIGHT",
        "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
    )
    steps = int(_env("NUNCHAKU_LITE_QWEN_FULL_INFERENCE_STEPS", "4"))
    output_dir = Path(_env("NUNCHAKU_LITE_FULL_INFERENCE_OUTPUT_DIR", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_nunchaku_pipeline(
        model_id,
        pipeline_cls=QwenImagePipeline,
        checkpoint=checkpoint,
        target="qwen_image",
        precision=precision,
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(_scheduler_config()),
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(lora_repo, weight_name=lora_weight, adapter_name="lightning")
    pipe.set_adapters("lightning", adapter_weights=float(_env("NUNCHAKU_LITE_QWEN_FULL_INFERENCE_LORA_SCALE", "1.0")))

    prompt = "a tiny astronaut hatching from an egg on the moon, Ultra HD, 4K, cinematic composition."
    image = pipe(
        prompt=prompt,
        negative_prompt=" ",
        width=512,
        height=512,
        num_inference_steps=steps,
        true_cfg_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(0),
    ).images[0]
    image.save(output_dir / "qwen_image_lightning_lora.png")

    _assert_image_has_signal(image, "qwen_image_lightning")
    assert pipe.get_active_adapters() == ["lightning"]
    pipe.unload_lora_weights()
    assert pipe.get_list_adapters() == {"transformer": []}
