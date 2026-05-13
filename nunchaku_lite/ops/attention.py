import torch


def _ops():
    from nunchaku_lite._C import ops

    return ops


def attention_fp16_cuda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, scale: float) -> None:
    _ops().attention_fp16(q, k, v, o, scale)
