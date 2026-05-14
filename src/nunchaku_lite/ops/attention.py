"""Attention kernel wrappers exposed by the native extension."""

import torch


def _ops():
    """Import native ops lazily.

    Returns:
        Native extension ``ops`` namespace.
    """

    from nunchaku_lite._C import ops

    return ops


def attention_fp16_cuda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, scale: float) -> None:
    """Run native fp16 attention into a preallocated output tensor.

    Args:
        q: Query tensor in the native packed attention layout.
        k: Key tensor in the native packed attention layout.
        v: Value tensor in the native packed attention layout.
        o: Output tensor to fill.
        scale: Attention scale factor, typically ``head_dim ** -0.5``.

    Returns:
        None.
    """

    _ops().attention_fp16(q, k, v, o, scale)
