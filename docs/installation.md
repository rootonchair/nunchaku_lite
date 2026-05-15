# Installation

## Prerequisites

- Python 3.10 or newer
- PyTorch 2.7 or newer with CUDA
- CUDA toolkit 12.6 or newer with `nvcc`
- Diffusers 0.36 or newer

The source build compiles the native CUDA extension locally. Make sure `nvcc`
is on `PATH` and that the installed PyTorch CUDA build is compatible with the
CUDA toolkit.

!!! note "CUDA version compatibility"

    Source installs and wheel builds compile the native `nunchaku_lite._C` CUDA
    extension. Use a CUDA toolkit with `nvcc` that is compatible with your
    installed PyTorch CUDA build. CUDA 12.6 or newer is the documented minimum;
    Blackwell `sm120a` requires CUDA 12.8 or newer, and `sm121a` requires CUDA
    13.0 or newer.

## Install From Source

Clone the repository and install from the repository root:

```bash
git clone https://github.com/rootonchair/nunchaku_lite.git
cd nunchaku_lite
pip install .
```

This installs the Python dependencies from `pyproject.toml` and builds the
native extension. By default, the build uses `NUNCHAKU_INSTALL_MODE=FAST` and
compiles for visible local CUDA devices.

To build all supported GPU architectures:

```bash
NUNCHAKU_INSTALL_MODE=ALL pip install .
```

Supported targets are `sm75`, `sm80`, `sm86`, `sm89`, `sm120a`, and `sm121a`,
subject to the installed CUDA toolkit version. CUDA 12.6 or newer supports the
non-Blackwell targets, CUDA 12.8 or newer is required for `sm120a`, and CUDA
13.0 or newer is required for `sm121a`.

## Install From GitHub

```bash
pip install git+https://github.com/rootonchair/nunchaku_lite.git
```

## Build A Wheel

```bash
python setup.py bdist_wheel
pip install dist/nunchaku_lite-*.whl
```
