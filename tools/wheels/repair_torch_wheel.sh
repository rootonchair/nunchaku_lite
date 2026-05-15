#!/usr/bin/env bash
set -euo pipefail

wheel="${1:?wheel path is required}"
dest_dir="${2:?destination directory is required}"

auditwheel repair "${wheel}" \
  --wheel-dir "${dest_dir}" \
  --exclude libc10.so \
  --exclude libc10_cuda.so \
  --exclude libcudart.so.12 \
  --exclude libcublas.so.12 \
  --exclude libcublasLt.so.12 \
  --exclude libcuda.so.1 \
  --exclude libtorch.so \
  --exclude libtorch_cpu.so \
  --exclude libtorch_cuda.so \
  --exclude libtorch_python.so
