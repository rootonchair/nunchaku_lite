import math
from dataclasses import dataclass
from typing import Type

import torch
import torch.nn as nn

from ..models.linear import SVDQW4A4Linear
from ..utils import convert_fp16, patch_scale_key


@dataclass(frozen=True)
class SVDQPatchContext:
    precision: str
    rank: int
    torch_dtype: torch.dtype
    requested_torch_dtype: torch.dtype | None = None

    @property
    def linear_kwargs(self) -> dict:
        return {"precision": self.precision, "rank": self.rank, "torch_dtype": self.torch_dtype}


def build_svdq_context(transformer: nn.Module, quantization_config: dict, options) -> SVDQPatchContext:
    rank = int(options.adapter_options.get("rank", quantization_config.get("rank", 32)))
    torch_dtype = options.torch_dtype or next(transformer.parameters()).dtype
    return SVDQPatchContext(
        precision=options.precision,
        rank=rank,
        torch_dtype=torch_dtype,
        requested_torch_dtype=options.torch_dtype,
    )


def prepare_transformer_dtype(transformer: nn.Module, context: SVDQPatchContext) -> None:
    if context.requested_torch_dtype is not None:
        transformer.to(context.requested_torch_dtype)


def finalize_svdq_checkpoint(
    transformer: nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
    context: SVDQPatchContext,
) -> None:
    patch_scale_key(transformer, checkpoint_state)
    if context.torch_dtype == torch.float16:
        convert_fp16(transformer, checkpoint_state)


def _linear_kwargs(context: SVDQPatchContext | None, explicit_kwargs: dict) -> dict:
    if context is None:
        return explicit_kwargs
    linear_kwargs = context.linear_kwargs
    linear_kwargs.update(explicit_kwargs)
    return linear_kwargs


def svdq_from_linear(
    linear: nn.Linear,
    context: SVDQPatchContext | None = None,
    **kwargs,
) -> SVDQW4A4Linear:
    return SVDQW4A4Linear.from_linear(linear, **_linear_kwargs(context, kwargs))


def patch_svdq_linears(
    module: nn.Module,
    context: SVDQPatchContext | None = None,
    **kwargs,
) -> nn.Module:
    return patch_linear(module, SVDQW4A4Linear, **_linear_kwargs(context, kwargs))


def fuse_linears(linears: list[nn.Linear]) -> nn.Linear:
    if not linears:
        raise ValueError("fuse_linears requires at least one linear layer")
    if len(linears) == 1:
        return linears[0]
    if not all(linear.in_features == linears[0].in_features for linear in linears):
        raise ValueError("All linear layers must share the same input feature dimension")
    return nn.Linear(
        linears[0].in_features,
        sum(linear.out_features for linear in linears),
        bias=all(linear.bias is not None for linear in linears),
        dtype=linears[0].weight.dtype,
        device=linears[0].weight.device,
    )


def patch_linear(module: nn.Module, linear_cls: Type[nn.Module], **kwargs) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, linear_cls.from_linear(child, **kwargs))
        else:
            patch_linear(child, linear_cls, **kwargs)
    return module


def pack_rotemb(rotemb: torch.Tensor) -> torch.Tensor:
    if rotemb.dtype != torch.float32:
        raise ValueError("Packed rotary embeddings require float32 input")
    batch = rotemb.shape[0]
    seq_len = rotemb.shape[1]
    dim = rotemb.shape[2] * 2
    if rotemb.shape != (batch, seq_len, dim // 2, 1, 2):
        raise ValueError("Unexpected rotary embedding shape")
    if seq_len % 16 != 0 or dim % 8 != 0:
        raise ValueError("Rotary embedding sequence length must be divisible by 16 and dim by 8")
    rotemb = rotemb.reshape(batch, seq_len // 16, 16, dim // 8, 8)
    rotemb = rotemb.permute(0, 1, 3, 2, 4)
    rotemb = rotemb.reshape(*rotemb.shape[0:3], 2, 8, 4, 2)
    rotemb = rotemb.permute(0, 1, 2, 4, 5, 3, 6).contiguous()
    return rotemb.view(batch, seq_len, dim)


def pad_tensor(tensor: torch.Tensor, multiple: int, dim: int) -> torch.Tensor:
    if tensor.shape[dim] % multiple == 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = ((shape[dim] + multiple - 1) // multiple) * multiple
    result = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, extent) for extent in tensor.shape)
    result[slices] = tensor
    return result


def alloc_packed_qkv(
    batch_size: int,
    heads: int,
    num_tokens: int,
    head_dim: int,
    device: torch.device,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    num_tokens_pad = math.ceil(num_tokens / pad_size) * pad_size
    query = torch.empty(batch_size, heads, num_tokens_pad, head_dim, dtype=torch.float16, device=device)
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    return query, key, value, num_tokens_pad


def apply_gated_residual(residual: torch.Tensor, gate: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled():
        return residual + gate * update
    residual.addcmul_(gate, update)
    return residual
