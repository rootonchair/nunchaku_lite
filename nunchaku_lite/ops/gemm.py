"""GEMM kernel wrappers exposed by the native extension."""

import math

import torch


def _ops():
    """Import native ops lazily.

    Returns:
        Native extension ``ops`` namespace.
    """

    from nunchaku_lite._C import ops

    return ops


def svdq_gemm_w4a4_cuda(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor | None = None,
    qout: torch.Tensor | None = None,
    ascales: torch.Tensor | None = None,
    wscales: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    poolout: torch.Tensor | None = None,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    norm_q: torch.Tensor | None = None,
    norm_k: torch.Tensor | None = None,
    rotary_emb: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    smooth_factor: torch.Tensor | None = None,
    out_vk: torch.Tensor | None = None,
    out_linearattn: torch.Tensor | None = None,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float | None = 1.0,
    wcscales: torch.Tensor | None = None,
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    attn_tokens: int = 0,
) -> None:
    """Run the native SVDQ W4A4 GEMM with optional fused output paths.

    Args:
        act: Quantized activation tensor.
        wgt: Packed quantized weight tensor.
        out: Optional dense output tensor.
        qout: Optional quantized output tensor for chained kernels.
        ascales: Activation scales for ``act``.
        wscales: Weight scales.
        oscales: Optional output activation scales for ``qout``.
        poolout: Optional pooled output buffer.
        lora_act_in: Low-rank activation input.
        lora_up: Low-rank up-projection.
        lora_down: Optional low-rank down-projection for fused output.
        lora_act_out: Optional low-rank activation output buffer.
        norm_q: Optional Q normalization weights.
        norm_k: Optional K normalization weights.
        rotary_emb: Optional packed rotary embedding for Q/K.
        bias: Optional output bias.
        smooth_factor: Optional smooth factor for output quantization.
        out_vk: Optional fused value/key output buffer.
        out_linearattn: Optional linear-attention output buffer.
        act_unsigned: Whether activation values use unsigned interpretation.
        lora_scales: Optional per-rank-block low-rank scales.
        fuse_silu: Whether to fuse SiLU in the native kernel.
        fp4: Whether to use NVFP4 kernel paths.
        alpha: Optional fp4 weight scale multiplier.
        wcscales: Optional fp4 weight correction scales.
        out_q: Optional preallocated Q output for attention paths.
        out_k: Optional preallocated K output for attention paths.
        out_v: Optional preallocated V output for attention paths.
        attn_tokens: Number of unpadded attention tokens when using Q/K/V
            outputs.

    Returns:
        None.
    """

    if lora_scales is None:
        rank = lora_up.shape[1]
        lora_scales = [1.0] * math.ceil(rank / 16)
    if alpha is None:
        alpha = 1.0
    _ops().gemm_w4a4(
        act,
        wgt,
        out,
        qout,
        ascales,
        wscales,
        oscales,
        poolout,
        lora_act_in,
        lora_up,
        lora_down,
        lora_act_out,
        norm_q,
        norm_k,
        rotary_emb,
        bias,
        smooth_factor,
        out_vk,
        out_linearattn,
        act_unsigned,
        lora_scales,
        fuse_silu,
        fp4,
        alpha,
        wcscales,
        out_q,
        out_k,
        out_v,
        attn_tokens,
    )
