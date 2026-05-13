"""Activation quantization wrappers exposed by the native extension."""

import torch

from ..utils import ceil_divide


def _ops():
    """Import native ops lazily.

    Returns:
        Native extension ``ops`` namespace.
    """

    from nunchaku_lite._C import ops

    return ops


def svdq_quantize_w4a4_act_fuse_lora_cuda(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize W4A4 activations and optionally fuse smooth and LoRA-down paths.

    Args:
        input: Flattened activation tensor with shape ``(tokens, channels)``.
        output: Optional preallocated quantized activation buffer.
        oscales: Optional preallocated activation-scale buffer.
        lora_down: Low-rank down-projection used to compute side activations.
        lora_act_out: Optional preallocated low-rank activation output.
        smooth: Optional smooth factor applied during quantization.
        fuse_glu: Whether the native quantizer should fuse GLU handling.
        fp4: Whether to use the NVFP4 activation scale layout.
        pad_size: Token padding multiple required by the kernel.

    Returns:
        Tuple of ``(quantized_output, output_scales, lora_activation_output)``.

    Raises:
        ValueError: If channel dimensions are incompatible with the selected
            precision layout.
    """

    batch_size, channels = input.shape
    rank = lora_down.shape[1]
    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size
    if output is None:
        output = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
    if oscales is None:
        if fp4:
            if channels % 16 != 0:
                raise ValueError("NVFP4 activation channels must be divisible by 16")
            oscales = torch.empty(channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=input.device)
        else:
            if channels % 64 != 0:
                raise ValueError("INT4 activation channels must be divisible by 64")
            oscales = torch.empty(channels // 64, batch_size_pad, dtype=input.dtype, device=input.device)
    if lora_act_out is None:
        lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)

    _ops().quantize_w4a4_act_fuse_lora(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4)
    return output, oscales, lora_act_out
