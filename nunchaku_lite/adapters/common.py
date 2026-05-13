"""Shared adapter utilities for replacing Diffusers modules with lite SVDQ modules."""

import math
from dataclasses import dataclass
from typing import Type

import torch
import torch.nn as nn

from ..models.linear import SVDQW4A4Linear
from ..utils import convert_fp16, patch_scale_key


@dataclass(frozen=True)
class SVDQPatchContext:
    """SVDQ replacement settings shared across a patch operation.

    Attributes:
        precision: Native precision used by SVDQ kernels.
        rank: Low-rank correction rank expected by checkpoint tensors.
        torch_dtype: Runtime dtype for quantized module buffers.
        requested_torch_dtype: Explicit dtype requested by the caller, if any.
    """

    precision: str
    rank: int
    torch_dtype: torch.dtype
    requested_torch_dtype: torch.dtype | None = None

    @property
    def linear_kwargs(self) -> dict:
        """Return constructor kwargs shared by SVDQ linear replacements.

        Args:
            None.

        Returns:
            Dictionary containing ``precision``, ``rank``, and ``torch_dtype``.
        """

        return {"precision": self.precision, "rank": self.rank, "torch_dtype": self.torch_dtype}


def build_svdq_context(transformer: nn.Module, quantization_config: dict, options) -> SVDQPatchContext:
    """Build SVDQ patch settings from checkpoint metadata and patch options.

    Args:
        transformer: Transformer being patched; used to infer dtype when no
            dtype override is provided.
        quantization_config: Parsed checkpoint quantization metadata.
        options: Patch options from :func:`nunchaku_lite.patch_transformer`.

    Returns:
        Immutable context used by adapter helper functions.
    """

    rank = int(options.adapter_options.get("rank", quantization_config.get("rank", 32)))
    torch_dtype = options.torch_dtype or next(transformer.parameters()).dtype
    return SVDQPatchContext(
        precision=options.precision,
        rank=rank,
        torch_dtype=torch_dtype,
        requested_torch_dtype=options.torch_dtype,
    )


def prepare_transformer_dtype(transformer: nn.Module, context: SVDQPatchContext) -> None:
    """Move a transformer to the requested dtype before module replacement.

    Args:
        transformer: Module tree to mutate.
        context: SVDQ patch context containing the requested dtype.

    Returns:
        None.
    """

    if context.requested_torch_dtype is not None:
        transformer.to(context.requested_torch_dtype)


def finalize_svdq_checkpoint(
    transformer: nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
    context: SVDQPatchContext,
) -> None:
    """Normalize checkpoint tensors after adapter-specific module rewrites.

    Args:
        transformer: Patched transformer whose state dict defines expected keys.
        checkpoint_state: Mutable checkpoint state dict to normalize.
        context: SVDQ patch context controlling optional fp16 conversion.

    Returns:
        None.
    """

    patch_scale_key(transformer, checkpoint_state)
    if context.torch_dtype == torch.float16:
        convert_fp16(transformer, checkpoint_state)


def _linear_kwargs(context: SVDQPatchContext | None, explicit_kwargs: dict) -> dict:
    """Merge context-derived SVDQ kwargs with explicit constructor overrides.

    Args:
        context: Optional SVDQ settings object.
        explicit_kwargs: Keyword arguments supplied by the caller.

    Returns:
        Constructor kwargs where explicit values override context defaults.
    """

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
    """Create an empty SVDQ linear replacement from an ``nn.Linear``.

    Args:
        linear: Source linear module whose input/output metadata is copied.
        context: Optional SVDQ settings applied to the replacement.
        **kwargs: Additional constructor overrides for
            :class:`SVDQW4A4Linear`.

    Returns:
        SVDQ linear module ready for checkpoint loading.
    """

    return SVDQW4A4Linear.from_linear(linear, **_linear_kwargs(context, kwargs))


def patch_svdq_linears(
    module: nn.Module,
    context: SVDQPatchContext | None = None,
    **kwargs,
) -> nn.Module:
    """Recursively replace every ``nn.Linear`` child with ``SVDQW4A4Linear``.

    Args:
        module: Module tree to mutate in place.
        context: Optional SVDQ settings applied to every replacement.
        **kwargs: Additional constructor overrides for replacement modules.

    Returns:
        The same ``module`` object after mutation.
    """

    return patch_linear(module, SVDQW4A4Linear, **_linear_kwargs(context, kwargs))


def fuse_linears(linears: list[nn.Linear]) -> nn.Linear:
    """Create a placeholder linear for concatenated output projections.

    The returned layer has the shared input dimension and the sum of output
    dimensions from all source linears. It is allocated only for metadata; its
    parameters are later replaced by checkpoint tensors.

    Args:
        linears: Source linears to fuse along the output dimension.

    Returns:
        Metadata-compatible ``nn.Linear``.

    Raises:
        ValueError: If the list is empty or input dimensions differ.
    """

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
    """Recursively replace child ``nn.Linear`` modules.

    Args:
        module: Module tree to mutate in place.
        linear_cls: Replacement class exposing ``from_linear``.
        **kwargs: Constructor arguments forwarded to ``from_linear``.

    Returns:
        The same ``module`` object after mutation.
    """

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, linear_cls.from_linear(child, **kwargs))
        else:
            patch_linear(child, linear_cls, **kwargs)
    return module


def pack_rotemb(rotemb: torch.Tensor) -> torch.Tensor:
    """Pack float32 rotary embeddings into the native kernel layout.

    Args:
        rotemb: Rotary embedding tensor with shape
            ``(batch, seq_len, dim // 2, 1, 2)`` and dtype ``float32``.

    Returns:
        Packed rotary tensor with shape ``(batch, padded_seq_len, dim)``.

    Raises:
        ValueError: If dtype, shape, or divisibility constraints are not met.
    """

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
    """Pad a tensor with zeros along one dimension.

    Args:
        tensor: Tensor to pad.
        multiple: Required output-size multiple for ``dim``.
        dim: Dimension to pad.

    Returns:
        ``tensor`` unchanged when already aligned, otherwise a new zero-padded
        tensor with the original values copied into the leading slice.
    """

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
    """Allocate padded Q/K/V tensors for packed native attention kernels.

    Args:
        batch_size: Batch dimension.
        heads: Number of attention heads.
        num_tokens: Unpadded sequence length.
        head_dim: Per-head channel dimension.
        device: Device for the allocated tensors.
        pad_size: Sequence padding multiple required by the kernel.

    Returns:
        Tuple of ``(query, key, value, padded_num_tokens)``.
    """

    num_tokens_pad = math.ceil(num_tokens / pad_size) * pad_size
    query = torch.empty(batch_size, heads, num_tokens_pad, head_dim, dtype=torch.float16, device=device)
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    return query, key, value, num_tokens_pad


def apply_gated_residual(residual: torch.Tensor, gate: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Apply a gated residual update.

    Args:
        residual: Residual tensor to update.
        gate: Multiplicative gate broadcastable to ``update``.
        update: New block output.

    Returns:
        ``residual + gate * update``. The operation is functional when
        gradients are enabled and in-place during inference.
    """

    if torch.is_grad_enabled():
        return residual + gate * update
    residual.addcmul_(gate, update)
    return residual
