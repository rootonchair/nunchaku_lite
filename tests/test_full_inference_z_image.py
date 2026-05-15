import gc
import json
import os
from pathlib import Path

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
        num_inference_steps=int(_env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_STEPS", "8")),
        guidance_scale=float(_env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_GUIDANCE", "0.0")),
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]


def _generate_with_output_type(pipe, prompt: str, seed: int, output_type: str):
    return pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=int(_env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_STEPS", "8")),
        guidance_scale=float(_env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_GUIDANCE", "0.0")),
        generator=torch.Generator(device="cuda").manual_seed(seed),
        output_type=output_type,
    ).images


def _clone_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    return value


def _move_to_cuda(value):
    if torch.is_tensor(value):
        return value.to("cuda")
    if isinstance(value, list):
        return [_move_to_cuda(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_cuda(item) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_cuda(item) for key, item in value.items()}
    return value


def _flatten_tensors(value) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, np.ndarray):
        return [torch.from_numpy(value)]
    if hasattr(value, "convert"):
        return [torch.from_numpy(np.asarray(value.convert("RGB"), dtype=np.float32))]
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def _compare_tensors(actual: torch.Tensor, expected: torch.Tensor, label: str) -> dict[str, float]:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()

    assert actual.shape == expected.shape, f"{label} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(actual).all(), f"{label} actual output contains non-finite values."
    assert torch.isfinite(expected).all(), f"{label} expected output contains non-finite values."

    diff = actual - expected
    mae = float(diff.abs().mean())
    max_abs = float(diff.abs().max())
    actual_flat = actual.flatten()
    expected_flat = expected.flatten()
    actual_norm = float(actual_flat.norm())
    expected_norm = float(expected_flat.norm())
    if actual_norm == 0 and expected_norm == 0:
        cosine = 1.0
    elif actual_norm == 0 or expected_norm == 0:
        cosine = 0.0
    else:
        cosine = float(torch.nn.functional.cosine_similarity(actual_flat, expected_flat, dim=0))
    return {"mae": mae, "max_abs": max_abs, "cosine": cosine}


@pytest.mark.skipif(
    os.environ.get("NUNCHAKU_LITE_RUN_FULL_INFERENCE") != "1",
    reason="set NUNCHAKU_LITE_RUN_FULL_INFERENCE=1 to run GPU/network full inference",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="full inference requires CUDA")
def test_z_image_turbo_runtime_lora_full_inference(tmp_path):
    from diffusers import ZImagePipeline

    from nunchaku_lite import load_nunchaku_pipeline

    model_id = _env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
    precision = _env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_PRECISION", "int4")
    checkpoint = _env(
        "NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_CHECKPOINT",
        "nunchaku-ai/nunchaku-z-image-turbo/svdq-int4_r128-z-image-turbo.safetensors",
    )
    lora_repo = _env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_LORA_REPO", "tarn59/pixel_art_style_lora_z_image_turbo")
    lora_weight = _env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_LORA_WEIGHT", "pixel_art_style_z_image_turbo.safetensors")
    output_dir = Path(_env("NUNCHAKU_LITE_FULL_INFERENCE_OUTPUT_DIR", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)

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

    prompt = "a cozy fantasy castle village at sunset, warm windows, river, detailed 16-bit game art"
    lora_prompt = f"Pixel art style. {prompt}"
    seed = 12345

    baseline = _generate(pipe, prompt, seed)
    baseline.save(output_dir / "z_image_turbo_baseline.png")
    _assert_image_has_signal(baseline, "baseline")

    pipe.load_lora_weights(lora_repo, weight_name=lora_weight, adapter_name="pixelart")
    pipe.set_adapters("pixelart", adapter_weights=float(_env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_LORA_SCALE", "1.0")))
    lora_image = _generate(pipe, lora_prompt, seed)
    lora_image.save(output_dir / "z_image_turbo_pixelart_lora.png")
    _assert_image_has_signal(lora_image, "lora_image")
    assert pipe.get_active_adapters() == ["pixelart"]
    assert _mean_abs_diff(baseline, lora_image) > 0.5

    pipe.unload_lora_weights()
    assert pipe.get_list_adapters() == {"transformer": []}
    reset = _generate(pipe, prompt, seed)
    reset.save(output_dir / "z_image_turbo_reset.png")
    _assert_image_has_signal(reset, "reset")
    assert _mean_abs_diff(lora_image, reset) > 0.5


@pytest.mark.skipif(
    os.environ.get("NUNCHAKU_LITE_RUN_FULL_INFERENCE") != "1",
    reason="set NUNCHAKU_LITE_RUN_FULL_INFERENCE=1 to run GPU/network full inference",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="full inference requires CUDA")
def test_z_image_turbo_packed_lora_matches_diffusers_lora_per_dit_run(tmp_path):
    from diffusers import ZImagePipeline

    from nunchaku_lite import load_nunchaku_pipeline

    model_id = _env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
    precision = _env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_PRECISION", "int4")
    checkpoint = _env(
        "NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_CHECKPOINT",
        "nunchaku-ai/nunchaku-z-image-turbo/svdq-int4_r128-z-image-turbo.safetensors",
    )
    lora_repo = _env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_LORA_REPO", "tarn59/pixel_art_style_lora_z_image_turbo")
    lora_weight = _env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_LORA_WEIGHT", "pixel_art_style_z_image_turbo.safetensors")
    lora_scale = float(_env("NUNCHAKU_LITE_Z_IMAGE_FULL_INFERENCE_LORA_SCALE", "1.0"))
    output_type = _env("NUNCHAKU_LITE_Z_IMAGE_FULL_PIPELINE_COMPARE_OUTPUT", "latent")
    dit_mae_atol = float(_env("NUNCHAKU_LITE_Z_IMAGE_DIT_MAE_ATOL", "0.35"))
    dit_max_atol = float(_env("NUNCHAKU_LITE_Z_IMAGE_DIT_MAX_ATOL", "8.0"))
    dit_cosine_min = float(_env("NUNCHAKU_LITE_Z_IMAGE_DIT_COSINE_MIN", "0.95"))
    final_mae_atol = float(_env("NUNCHAKU_LITE_Z_IMAGE_FINAL_MAE_ATOL", "0.5"))
    final_max_atol = float(_env("NUNCHAKU_LITE_Z_IMAGE_FINAL_MAX_ATOL", "12.0"))
    final_cosine_min = float(_env("NUNCHAKU_LITE_Z_IMAGE_FINAL_COSINE_MIN", "0.95"))
    output_dir = Path(_env("NUNCHAKU_LITE_FULL_INFERENCE_OUTPUT_DIR", str(tmp_path)))
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = "a cozy fantasy castle village at sunset, warm windows, river, detailed 16-bit game art"
    lora_prompt = f"Pixel art style. {prompt}"
    seed = 12345
    records = []

    dense_pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    dense_pipe.load_lora_weights(lora_repo, weight_name=lora_weight, adapter_name="pixelart")
    dense_pipe.set_adapters("pixelart", adapter_weights=lora_scale)
    dense_forward = dense_pipe.transformer.forward

    def recording_forward(*args, **kwargs):
        cpu_args = _clone_to_cpu(args)
        cpu_kwargs = _clone_to_cpu(kwargs)
        output = dense_forward(*args, **kwargs)
        records.append(
            {
                "args": cpu_args,
                "kwargs": cpu_kwargs,
                "output": _clone_to_cpu(output),
            }
        )
        return output

    dense_pipe.transformer.forward = recording_forward
    try:
        dense_final = _generate_with_output_type(dense_pipe, lora_prompt, seed, output_type)
    finally:
        dense_pipe.transformer.forward = dense_forward
    dense_final = _clone_to_cpu(dense_final)
    dense_forward = None
    del dense_pipe
    gc.collect()
    torch.cuda.empty_cache()

    assert records, "Dense Diffusers pipeline did not record any transformer calls."

    nunchaku_pipe = load_nunchaku_pipeline(
        model_id,
        pipeline_cls=ZImagePipeline,
        checkpoint=checkpoint,
        target="z_image",
        precision=precision,
        torch_dtype=torch.bfloat16,
        device="cuda",
    ).to("cuda")
    nunchaku_pipe.load_lora_weights(lora_repo, weight_name=lora_weight, adapter_name="pixelart")
    nunchaku_pipe.set_adapters("pixelart", adapter_weights=lora_scale)
    assert nunchaku_pipe.get_active_adapters() == ["pixelart"]

    metrics = []
    for step_index, record in enumerate(records):
        with torch.no_grad():
            actual_output = nunchaku_pipe.transformer(*_move_to_cuda(record["args"]), **_move_to_cuda(record["kwargs"]))
        actual_tensors = _flatten_tensors(actual_output)
        expected_tensors = _flatten_tensors(record["output"])

        assert len(actual_tensors) == len(expected_tensors), (
            f"DiT call {step_index} returned {len(actual_tensors)} tensors, "
            f"expected {len(expected_tensors)} tensors."
        )
        for tensor_index, (actual, expected) in enumerate(zip(actual_tensors, expected_tensors, strict=True)):
            label = f"dit_step={step_index}, tensor={tensor_index}"
            tensor_metrics = _compare_tensors(actual, expected, label)
            metrics.append({"step": step_index, "tensor": tensor_index, **tensor_metrics})
            assert tensor_metrics["mae"] <= dit_mae_atol, f"{label} MAE too high: {tensor_metrics}"
            assert tensor_metrics["max_abs"] <= dit_max_atol, f"{label} max abs diff too high: {tensor_metrics}"
            assert tensor_metrics["cosine"] >= dit_cosine_min, f"{label} cosine too low: {tensor_metrics}"

    nunchaku_final = _generate_with_output_type(nunchaku_pipe, lora_prompt, seed, output_type)
    final_metrics = []
    actual_final_tensors = _flatten_tensors(nunchaku_final)
    expected_final_tensors = _flatten_tensors(dense_final)
    assert len(actual_final_tensors) == len(
        expected_final_tensors
    ), f"Final output returned {len(actual_final_tensors)} tensors, expected {len(expected_final_tensors)} tensors."
    for tensor_index, (actual, expected) in enumerate(zip(actual_final_tensors, expected_final_tensors, strict=True)):
        label = f"final_output tensor={tensor_index}"
        tensor_metrics = _compare_tensors(actual, expected, label)
        final_metrics.append({"tensor": tensor_index, **tensor_metrics})
        assert tensor_metrics["mae"] <= final_mae_atol, f"{label} MAE too high: {tensor_metrics}"
        assert tensor_metrics["max_abs"] <= final_max_atol, f"{label} max abs diff too high: {tensor_metrics}"
        assert tensor_metrics["cosine"] >= final_cosine_min, f"{label} cosine too low: {tensor_metrics}"

    report = {
        "model_id": model_id,
        "checkpoint": checkpoint,
        "lora_repo": lora_repo,
        "lora_weight": lora_weight,
        "output_type": output_type,
        "steps": len(records),
        "dit_metrics": metrics,
        "final_metrics": final_metrics,
    }
    (output_dir / "z_image_turbo_packed_vs_diffusers_lora_metrics.json").write_text(json.dumps(report, indent=2))
