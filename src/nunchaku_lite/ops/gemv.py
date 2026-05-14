"""GEMV kernel wrappers exposed by the native extension."""

import torch


def _ops():
    """Import native ops lazily.

    Returns:
        Native extension ``ops`` namespace.
    """

    from nunchaku_lite._C import ops

    return ops


def awq_gemv_w4a16_cuda(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    """Run the native AWQ W4A16 GEMV kernel.

    Args:
        in_feats: Input activations.
        kernel: Packed AWQ weight tensor.
        scaling_factors: Per-group scaling factors.
        zeros: Per-group zero-point values.
        m: Runtime row/token count.
        n: Output feature count.
        k: Input feature count.
        group_size: AWQ group size.

    Returns:
        Projected activation tensor.
    """

    return _ops().gemv_awq(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size)
