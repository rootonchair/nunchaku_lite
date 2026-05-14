import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


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


def _mean_abs_diff(left, right) -> float:
    left_array = np.asarray(left.convert("RGB"), dtype=np.float32)
    right_array = np.asarray(right.convert("RGB"), dtype=np.float32)
    return float(np.abs(left_array - right_array).mean())


def _generate(pipe, prompt: str, seed: int):
    return pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=int(_env("NUNCHAKU_LITE_FLUX2_FULL_INFERENCE_STEPS", "4")),
        guidance_scale=float(_env("NUNCHAKU_LITE_FLUX2_FULL_INFERENCE_GUIDANCE", "1.0")),
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]


@pytest.mark.skipif(
    os.environ.get("NUNCHAKU_LITE_RUN_FULL_INFERENCE") != "1",
    reason="set NUNCHAKU_LITE_RUN_FULL_INFERENCE=1 to run GPU/network full inference",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="full inference requires CUDA")
def test_flux2_klein_runtime_lora_full_inference(tmp_path):
    from diffusers import Flux2KleinPipeline

    from nunchaku_lite import load_nunchaku_pipeline

    model_id = _env("NUNCHAKU_LITE_FLUX2_FULL_INFERENCE_MODEL_ID", "tonera/FLUX.2-klein-9B-Nunchaku")
    precision = _env("NUNCHAKU_LITE_FLUX2_FULL_INFERENCE_PRECISION", "int4")
    checkpoint = _env(
        "NUNCHAKU_LITE_FLUX2_FULL_INFERENCE_CHECKPOINT",
        "tonera/FLUX.2-klein-9B-Nunchaku/svdq-int4_r32-FLUX.2-klein-9B-Nunchaku.safetensors",
    )
    lora_repo = _env("NUNCHAKU_LITE_FLUX2_FULL_INFERENCE_LORA_REPO", "artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B")
    lora_weight = _env(
        "NUNCHAKU_LITE_FLUX2_FULL_INFERENCE_LORA_WEIGHT",
        "[FLUX.2.Klein]PixelArt_Redmond.safetensors",
    )
    output_dir = Path(_env("NUNCHAKU_LITE_FULL_INFERENCE_OUTPUT_DIR", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_nunchaku_pipeline(
        model_id,
        pipeline_cls=Flux2KleinPipeline,
        checkpoint=checkpoint,
        target="flux2",
        precision=precision,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    prompt = "a tiny knight standing beside a glowing arcade machine in a forest"
    lora_prompt = f"Pixel Art, PixArFK, {prompt}"
    seed = 12345

    baseline = _generate(pipe, prompt, seed)
    baseline.save(output_dir / "flux2_klein_baseline.png")
    _assert_image_has_signal(baseline, "baseline")

    pipe.load_lora_weights(lora_repo, weight_name=lora_weight, adapter_name="pixelart")
    pipe.set_adapters("pixelart", adapter_weights=float(_env("NUNCHAKU_LITE_FLUX2_FULL_INFERENCE_LORA_SCALE", "0.8")))
    lora_image = _generate(pipe, lora_prompt, seed)
    lora_image.save(output_dir / "flux2_klein_pixelart_lora.png")
    _assert_image_has_signal(lora_image, "lora_image")
    assert pipe.get_active_adapters() == ["pixelart"]
    assert _mean_abs_diff(baseline, lora_image) > 0.5

    pipe.unload_lora_weights()
    assert pipe.get_list_adapters() == {"transformer": []}
    reset = _generate(pipe, prompt, seed)
    reset.save(output_dir / "flux2_klein_reset.png")
    _assert_image_has_signal(reset, "reset")
    assert _mean_abs_diff(lora_image, reset) > 0.5


@pytest.mark.skipif(
    os.environ.get("NUNCHAKU_LITE_RUN_FULL_INFERENCE") != "1",
    reason="set NUNCHAKU_LITE_RUN_FULL_INFERENCE=1 to run GPU/network full inference",
)
def test_flux2_klein_4b_comfyui_lora_conversion_against_real_checkpoint():
    from diffusers import Flux2Transformer2DModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from nunchaku_lite.adapters.flux2 import Flux2Adapter

    model_id = _env("NUNCHAKU_LITE_FLUX2_4B_CONVERSION_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
    lora_repo = _env("NUNCHAKU_LITE_FLUX2_4B_CONVERSION_LORA_REPO", "Sawata97/flux2_4b_koni_animestyle")
    lora_weight = _env("NUNCHAKU_LITE_FLUX2_4B_CONVERSION_LORA_WEIGHT", "Flux_klein_4b_anime_Koni.safetensors")
    lora_path = hf_hub_download(lora_repo, lora_weight)
    lora_state = load_file(lora_path, device="cpu")

    with torch.device("meta"):
        config = Flux2Transformer2DModel.load_config(model_id, subfolder="transformer")
        transformer = Flux2Transformer2DModel.from_config(config)

    Flux2Adapter().patch(
        transformer,
        {},
        {"rank": 32},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=False,
            adapter_options={},
        ),
    )

    converted = transformer._convert_lora_to_nunchaku(lora_state)

    assert len(lora_state) == 160
    assert len(converted) == 240
    assert "single_transformer_blocks.0.attn.qkv_proj.proj_down" in converted
    assert "single_transformer_blocks.0.attn.mlp_fc1.proj_up" in converted
