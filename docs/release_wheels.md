# Release Wheels

Release wheels are built by GitHub Actions with `cibuildwheel` when a version
tag is pushed.

Before releasing, make sure `src/nunchaku_lite/__version__.py` points at the
release series. For example, `0.1.0dev` is valid for tag `v0.1.0`.

## Trigger A Release Build

Create and push a tag from the commit you want to release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release workflow validates that the tag matches the committed package
version after allowing a trailing `dev` postfix. Tag `v0.1.0` may build from
either `0.1.0` or `0.1.0dev`. For tag builds, CI strips only the build version
by setting `NUNCHAKU_LITE_RELEASE_VERSION`; it does not edit the committed
source file.

The final wheel version still includes the existing CUDA and Torch local suffix,
for example:

```text
0.1.0+cu13.0torch2.11
```

After release, bump to the next dev version:

```bash
git add src/nunchaku_lite/__version__.py
git commit -m "Start 0.1.1 development"
```

Manual builds are available from the `Release Wheels` workflow through
`workflow_dispatch`; manual builds may use dev versions for validation.

## Build Matrix

The release workflow builds Linux `x86_64` wheels for Python 3.10 through 3.13.
Wheels are split by Python version, Torch version, and CUDA variant. They are
not split by individual GPU architecture.

| Torch | CUDA variants |
| --- | --- |
| 2.9.1 | cu128, cu130 |
| 2.10.0 | cu128, cu130 |
| 2.11.0 | cu128, cu130 |
| 2.12.0 | cu130 |

Torch 2.12 does not build a `cu128` wheel in this project matrix. CUDA 13.2
(`cu132`) is intentionally excluded until the project decides to support
experimental CUDA release wheels.

Each wheel is built with `NUNCHAKU_INSTALL_MODE=ALL`, so the compiled extension
contains every supported SM target for the selected CUDA toolkit. CUDA 12.8
builds include `sm75`, `sm80`, `sm86`, `sm89`, and `sm120a`. CUDA 13.0 builds
also include `sm121a`.

CI builds one Python wheel per job and limits CUDA compiler parallelism to keep
GitHub-hosted runners from being terminated during heavy CUDA builds.

## Local Reproduction

Install `cibuildwheel`, choose a Torch/CUDA pair, and run the Linux build. Set
`NUNCHAKU_LITE_RELEASE_VERSION` only when reproducing a tag release; omit it for
normal dev builds.

```bash
python -m pip install cibuildwheel==3.4.1

export CUDA_VISIBLE_DEVICES=""
export NUNCHAKU_BUILD_WHEELS=1
export NUNCHAKU_CUDA_VERSION=13.0
export NUNCHAKU_INSTALL_MODE=ALL
export NUNCHAKU_LITE_RELEASE_VERSION=0.1.0
export NUNCHAKU_NVCC_THREADS=2
export NUNCHAKU_TORCH_CUDA_TAG=cu130
export NUNCHAKU_TORCH_VERSION=2.11.0
export MAX_JOBS=2
export CIBW_BUILD=cp313-manylinux_x86_64

python -m cibuildwheel --platform linux --output-dir wheelhouse
```

Local builds require Docker because Linux `cibuildwheel` runs inside manylinux
containers. The workflow installs the matching CUDA toolkit inside the build
container and installs the exact PyTorch wheel before building without build
isolation.
