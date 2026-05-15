"""Higher-level fused operations assembled from native quantization and GEMM kernels."""

import torch
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from torch.nn import RMSNorm

from ..linear import SVDQW4A4Linear
from ..utils import ceil_divide
from .gemm import svdq_gemm_w4a4_cuda


def fused_gelu_mlp(x: torch.Tensor, fc1: SVDQW4A4Linear, fc2: SVDQW4A4Linear, pad_size: int = 256) -> torch.Tensor:
    """Run a two-layer GELU MLP while keeping the intermediate activation quantized.

    Args:
        x: Input tensor with shape ``(batch, sequence, channels)``.
        fc1: First quantized linear projection, wrapped by Diffusers GELU.
        fc2: Second quantized linear projection.
        pad_size: Token padding multiple for the intermediate quantized buffer.

    Returns:
        MLP output with shape ``(batch, sequence, fc2.out_features)``.
    """

    batch_size, seq_len, channels = x.shape
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = fc1.quantize(x)

    batch_size_pad = ceil_divide(batch_size * seq_len, pad_size) * pad_size
    qout_act = torch.empty(batch_size_pad, fc1.out_features // 2, dtype=torch.uint8, device=x.device)
    if fc2.precision == "nvfp4":
        qout_ascales = torch.empty(fc1.out_features // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=x.device)
    else:
        qout_ascales = torch.empty(fc1.out_features // 64, batch_size_pad, dtype=x.dtype, device=x.device)
    qout_lora_act = torch.empty(batch_size_pad, fc2.proj_down.shape[1], dtype=torch.float32, device=x.device)

    svdq_gemm_w4a4_cuda(
        act=quantized_x,
        wgt=fc1.qweight,
        qout=qout_act,
        ascales=ascales,
        wscales=fc1.wscales,
        oscales=qout_ascales,
        lora_act_in=lora_act,
        lora_up=fc1.proj_up,
        lora_down=fc2.proj_down,
        lora_act_out=qout_lora_act,
        bias=fc1.bias,
        smooth_factor=fc2.smooth_factor,
        fp4=fc1.precision == "nvfp4",
        alpha=fc1.wtscale,
        wcscales=fc1.wcscales,
    )
    output = torch.empty(batch_size * seq_len, fc2.out_features, dtype=x.dtype, device=x.device)
    output = fc2.forward_quant(qout_act, qout_ascales, qout_lora_act, output=output)
    return output.view(batch_size, seq_len, -1)


def fused_qkv_norm_rotary(
    x: torch.Tensor,
    proj: SVDQW4A4Linear,
    norm_q: RMSNorm | DiffusersRMSNorm | None = None,
    norm_k: RMSNorm | DiffusersRMSNorm | None = None,
    rotary_emb: torch.Tensor | None = None,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    attn_tokens: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run QKV projection with fused Q/K normalization and packed RoPE.

    Args:
        x: Input tensor with shape ``(batch, sequence, channels)``.
        proj: Fused SVDQ QKV projection module.
        norm_q: Optional query RMSNorm module.
        norm_k: Optional key RMSNorm module.
        rotary_emb: Optional packed rotary embedding consumed by the native
            GEMM kernel.
        output: Optional dense output buffer, or a tuple of preallocated
            ``(query, key, value)`` tensors for packed attention.
        attn_tokens: Number of unpadded tokens when ``output`` is a Q/K/V
            tuple.

    Returns:
        Dense fused QKV tensor when ``output`` is not a tuple, otherwise the
        populated ``(query, key, value)`` tuple.
    """

    batch_size, seq_len, channels = x.shape
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = proj.quantize(x)

    if output is None:
        output = torch.empty(batch_size * seq_len, proj.out_features, dtype=x.dtype, device=x.device)

    norm_q_weight = norm_q.weight if norm_q is not None else None
    norm_k_weight = norm_k.weight if norm_k is not None else None

    if isinstance(output, tuple):
        out_q, out_k, out_v = output
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=proj.qweight,
            ascales=ascales,
            wscales=proj.wscales,
            lora_act_in=lora_act,
            lora_up=proj.proj_up,
            bias=proj.bias,
            fp4=proj.precision == "nvfp4",
            alpha=proj.wtscale,
            wcscales=proj.wcscales,
            norm_q=norm_q_weight,
            norm_k=norm_k_weight,
            rotary_emb=rotary_emb,
            out_q=out_q,
            out_k=out_k,
            out_v=out_v,
            attn_tokens=attn_tokens,
        )
        return out_q, out_k, out_v

    svdq_gemm_w4a4_cuda(
        act=quantized_x,
        wgt=proj.qweight,
        out=output,
        ascales=ascales,
        wscales=proj.wscales,
        lora_act_in=lora_act,
        lora_up=proj.proj_up,
        bias=proj.bias,
        fp4=proj.precision == "nvfp4",
        alpha=proj.wtscale,
        wcscales=proj.wcscales,
        norm_q=norm_q_weight,
        norm_k=norm_k_weight,
        rotary_emb=rotary_emb,
    )
    return output.view(batch_size, seq_len, -1)
