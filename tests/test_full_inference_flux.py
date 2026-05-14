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
    assert stats["adjacent_diff"] < 25.0, f"{name} looks like high-frequency noise: {stats}"


def _mean_abs_diff(left, right) -> float:
    left_array = np.asarray(left.convert("RGB"), dtype=np.float32)
    right_array = np.asarray(right.convert("RGB"), dtype=np.float32)
    return float(np.abs(left_array - right_array).mean())


def _generate(pipe, prompt: str, seed: int):
    return pipe(
        prompt,
        height=512,
        width=512,
        num_inference_steps=int(_env("NUNCHAKU_LITE_FULL_INFERENCE_STEPS", "28")),
        guidance_scale=float(_env("NUNCHAKU_LITE_FULL_INFERENCE_GUIDANCE", "3.5")),
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]


@pytest.mark.skipif(
    os.environ.get("NUNCHAKU_LITE_RUN_FULL_INFERENCE") != "1",
    reason="set NUNCHAKU_LITE_RUN_FULL_INFERENCE=1 to run GPU/network full inference",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="full inference requires CUDA")
def test_flux_dev_load_nunchaku_pipeline_runtime_lora_full_inference(tmp_path):
    from diffusers import FluxPipeline

    from nunchaku_lite import load_nunchaku_pipeline

    model_id = _env("NUNCHAKU_LITE_FULL_INFERENCE_MODEL_ID", "black-forest-labs/FLUX.1-dev")
    precision = _env("NUNCHAKU_LITE_FULL_INFERENCE_PRECISION", get_precision(device="cuda"))
    checkpoint = _env(
        "NUNCHAKU_LITE_FULL_INFERENCE_CHECKPOINT",
        f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",
    )
    ghibsky_repo = _env("NUNCHAKU_LITE_FULL_INFERENCE_GHIBSKY_REPO", "aleksa-codes/flux-ghibsky-illustration")
    ghibsky_weight = _env("NUNCHAKU_LITE_FULL_INFERENCE_GHIBSKY_WEIGHT", "lora.safetensors")
    realism_repo = _env(
        "NUNCHAKU_LITE_FULL_INFERENCE_REALISM_REPO",
        "prithivMLmods/Canopus-LoRA-Flux-UltraRealism-2.0",
    )
    realism_weight = _env(
        "NUNCHAKU_LITE_FULL_INFERENCE_REALISM_WEIGHT",
        "Canopus-LoRA-Flux-UltraRealism.safetensors",
    )
    output_dir = Path(_env("NUNCHAKU_LITE_FULL_INFERENCE_OUTPUT_DIR", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_nunchaku_pipeline(
        model_id,
        pipeline_cls=FluxPipeline,
        checkpoint=checkpoint,
        target="flux",
        precision=precision,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    prompt = "A portrait photo of a ceramic tea cup on a wooden desk beside a small fern, soft window light"
    seed = 12345

    baseline = _generate(pipe, prompt, seed)
    baseline.save(output_dir / "flux_dev_baseline.png")
    _assert_image_has_signal(baseline, "baseline")

    pipe.load_lora_weights(ghibsky_repo, weight_name=ghibsky_weight, adapter_name="ghibsky")
    pipe.set_adapters("ghibsky", adapter_weights=0.75)
    ghibsky_strong = _generate(pipe, f"GHIBSKY style {prompt}", seed)
    ghibsky_strong.save(output_dir / "flux_dev_ghibsky_075.png")
    _assert_image_has_signal(ghibsky_strong, "ghibsky_strong")

    pipe.set_adapters("ghibsky", adapter_weights=0.25)
    ghibsky_weak = _generate(pipe, f"GHIBSKY style {prompt}", seed)
    ghibsky_weak.save(output_dir / "flux_dev_ghibsky_025.png")
    _assert_image_has_signal(ghibsky_weak, "ghibsky_weak")
    assert _mean_abs_diff(ghibsky_strong, ghibsky_weak) > 0.5

    pipe.load_lora_weights(realism_repo, weight_name=realism_weight, adapter_name="realism")
    pipe.set_adapters(["ghibsky", "realism"], adapter_weights=[0.55, 0.45])
    composed = _generate(pipe, f"GHIBSKY style ultra realistic {prompt}", seed)
    composed.save(output_dir / "flux_dev_ghibsky_realism.png")
    _assert_image_has_signal(composed, "composed")
    assert set(pipe.get_active_adapters()) == {"ghibsky", "realism"}

    pipe.delete_adapters("realism")
    assert pipe.get_list_adapters() == {"transformer": ["ghibsky"]}
    pipe.set_adapters("ghibsky", adapter_weights=0.75)
    after_delete = _generate(pipe, f"GHIBSKY style {prompt}", seed)
    after_delete.save(output_dir / "flux_dev_after_delete_realism.png")
    _assert_image_has_signal(after_delete, "after_delete")

    pipe.unload_lora_weights()
    assert pipe.get_list_adapters() == {"transformer": []}
    reset = _generate(pipe, prompt, seed)
    reset.save(output_dir / "flux_dev_reset.png")
    _assert_image_has_signal(reset, "reset")
    assert _mean_abs_diff(composed, reset) > 0.5
