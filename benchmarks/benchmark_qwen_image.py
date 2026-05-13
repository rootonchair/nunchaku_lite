#!/usr/bin/env python3
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import torch


DEFAULT_MODEL_ID = "Qwen/Qwen-Image"
DEFAULT_CHECKPOINT = "nunchaku-tech/nunchaku-qwen-image/svdq-fp4_r32-qwen-image.safetensors"
DEFAULT_PROMPT = "a bookstore window display with legible signs and warm evening light"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark original Diffusers Qwen-Image against nunchaku_lite patched Qwen-Image."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--local-diffusers-src", default=None)
    parser.add_argument("--output-dir", default="outputs/benchmark_qwen_image")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--precision", choices=["auto", "fp4", "int4"], default="fp4")
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip-lite", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    return parser.parse_args()


def import_diffusers(local_diffusers_src: str | None):
    if local_diffusers_src:
        path = Path(local_diffusers_src)
        if path.exists():
            sys.path.insert(0, str(path))
    from diffusers import QwenImagePipeline

    return QwenImagePipeline


def dtype_from_arg(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def cuda_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def timed_cuda_call(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {"values": values, "mean": mean(values), "stdev": stdev(values) if len(values) > 1 else 0.0}


def run_generation(pipe, args: argparse.Namespace, label: str, output_dir: Path) -> dict:
    timings = []
    peaks = []
    last_image = None
    total_runs = args.warmup_runs + args.runs

    for index in range(total_runs):
        measured = index >= args.warmup_runs
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        image, elapsed = timed_cuda_call(
            lambda: pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                true_cfg_scale=args.true_cfg_scale,
                generator=generator,
            ).images[0]
        )
        peak = cuda_gb() if torch.cuda.is_available() else 0.0
        print(f"{label} run {index + 1}/{total_runs}: {elapsed:.3f}s, peak {peak:.2f} GB", flush=True)

        if measured:
            timings.append(elapsed)
            peaks.append(peak)
            last_image = image

    image_path = output_dir / f"{label}.png"
    if last_image is not None:
        last_image.save(image_path)
    return {"image": str(image_path), "seconds": summarize(timings), "peak_cuda_gb": summarize(peaks)}


def run_original(args: argparse.Namespace, output_dir: Path, pipeline_cls, torch_dtype: torch.dtype) -> dict:
    cleanup()
    print("loading original Qwen-Image pipeline", flush=True)
    pipe, load_seconds = timed_cuda_call(
        lambda: pipeline_cls.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    )
    pipe = pipe.to("cuda")
    result = run_generation(pipe, args, "original_diffusers", output_dir)
    result["load_seconds"] = load_seconds
    del pipe
    cleanup()
    return result


def run_lite(args: argparse.Namespace, output_dir: Path, pipeline_cls, torch_dtype: torch.dtype) -> dict:
    from nunchaku_lite import patch_transformer

    cleanup()
    print("loading base Qwen-Image pipeline for nunchaku_lite", flush=True)
    pipe, load_seconds = timed_cuda_call(
        lambda: pipeline_cls.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    )
    print("patching transformer with nunchaku_lite Qwen-Image", flush=True)
    _, patch_seconds = timed_cuda_call(
        lambda: patch_transformer(
            pipe.transformer,
            args.checkpoint,
            target="qwen_image",
            precision=args.precision,
            torch_dtype=torch_dtype,
            device="cuda",
        )
    )
    pipe = pipe.to("cuda")
    result = run_generation(pipe, args, "nunchaku_lite", output_dir)
    result["load_seconds"] = load_seconds
    result["patch_seconds"] = patch_seconds
    del pipe
    cleanup()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_cls = import_diffusers(args.local_diffusers_src)
    torch_dtype = dtype_from_arg(args.dtype)
    results = {
        "metadata": {
            "model_id": args.model_id,
            "checkpoint": args.checkpoint,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "true_cfg_scale": args.true_cfg_scale,
            "seed": args.seed,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "dtype": args.dtype,
            "precision": args.precision,
            "device": torch.cuda.get_device_name(0),
        }
    }

    if not args.skip_original:
        results["original_diffusers"] = run_original(args, output_dir, pipeline_cls, torch_dtype)
    if not args.skip_lite:
        results["nunchaku_lite"] = run_lite(args, output_dir, pipeline_cls, torch_dtype)
    if "original_diffusers" in results and "nunchaku_lite" in results:
        results["speedup"] = results["original_diffusers"]["seconds"]["mean"] / results["nunchaku_lite"]["seconds"][
            "mean"
        ]
        print(f"speedup: {results['speedup']:.3f}x", flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
