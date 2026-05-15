#!/usr/bin/env bash
set -euo pipefail

cuda_version="${1:-${NUNCHAKU_CUDA_VERSION:-}}"
if [[ -z "${cuda_version}" ]]; then
  echo "Usage: $0 <cuda-version>" >&2
  exit 2
fi

case "${cuda_version}" in
  12.8)
    cuda_package_version="12-8"
    ;;
  13.0)
    cuda_package_version="13-0"
    ;;
  *)
    echo "Unsupported CUDA version: ${cuda_version}" >&2
    exit 2
    ;;
esac

if [[ -x "/usr/local/cuda-${cuda_version}/bin/nvcc" ]]; then
  "/usr/local/cuda-${cuda_version}/bin/nvcc" --version
  exit 0
fi

if command -v dnf >/dev/null 2>&1; then
  package_manager=(dnf -y)
elif command -v yum >/dev/null 2>&1; then
  package_manager=(yum -y)
else
  echo "Expected dnf or yum in the manylinux container." >&2
  exit 1
fi

"${package_manager[@]}" install dnf-plugins-core
"${package_manager[@]}" config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
"${package_manager[@]}" clean all
"${package_manager[@]}" install \
  "cuda-cudart-devel-${cuda_package_version}" \
  "cuda-driver-devel-${cuda_package_version}" \
  "cuda-nvcc-${cuda_package_version}" \
  gcc-toolset-13-gcc \
  gcc-toolset-13-gcc-c++ \
  "libcublas-devel-${cuda_package_version}" \
  "libcusparse-devel-${cuda_package_version}"

"/opt/rh/gcc-toolset-13/root/usr/bin/g++" --version
"/usr/local/cuda-${cuda_version}/bin/nvcc" --version
