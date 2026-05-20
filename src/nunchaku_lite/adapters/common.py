"""Shared adapter utilities for replacing Diffusers modules with lite SVDQ modules."""

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention

from ..linear import SVDQW4A4Linear
from ..utils import convert_fp16, patch_scale_key

PATCHED_MODULE_ATTR = "_nunchaku_lite_patched_module"


def _mark_patched_module(module: nn.Module) -> nn.Module:
    setattr(module, PATCHED_MODULE_ATTR, True)
    return module


def _is_patched_module(module: nn.Module) -> bool:
    return bool(getattr(module, PATCHED_MODULE_ATTR, False))


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


@dataclass
class ModulePatchReport:
    """Counts of module replacements performed by a recursive patch pass.

    Attributes:
        converted_modules: Number of adapter-specific child modules normalized
            or replaced by a caller-provided converter before recursive
            patching.
        skipped_modules: Number of children intentionally skipped because they
            were already patched, were outside the configured patch scope, or
            belonged to a module type that the generic traversal must not
            rewrite.
    """

    converted_modules: int = 0
    skipped_modules: int = 0

    def add(self, other: "ModulePatchReport") -> None:
        """Merge counts from another report into this report.

        Args:
            other: Report produced by a nested recursive traversal.

        Returns:
            None. The current report is updated in place.
        """

        self.converted_modules += other.converted_modules
        self.skipped_modules += other.skipped_modules


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

    return _mark_patched_module(SVDQW4A4Linear.from_linear(linear, **_linear_kwargs(context, kwargs)))


class NunchakuAttention(nn.Module):
    """Generic Nunchaku Lite replacement for Diffusers attention modules.

    The class is intended for generic Diffusers attention modules whose forward
    contract can be preserved by swapping the processor and replacing dense
    projections with a fused SVDQ QKV projection. Model-specific attention
    classes with custom topology should still use rewritten lite classes.
    """

    def __init__(
        self,
        attention: nn.Module,
        processor,
        context: SVDQPatchContext | None = None,
        *,
        qkv_attr: str = "to_qkv",
        patch_output: bool = True,
        **kwargs,
    ):
        """Create a Nunchaku attention module from a generic Diffusers module.

        Args:
            attention: Source Diffusers attention module with ``to_q``,
                ``to_k``, and ``to_v`` projections.
            processor: Processor instance used by the new module.
            context: Optional SVDQ settings applied to replacement projections.
            qkv_attr: Attribute name that will hold the fused QKV projection.
            patch_output: Whether to replace ``attention.to_out[0]`` with SVDQ.
            **kwargs: Additional SVDQ constructor overrides.

        Raises:
            TypeError: If ``attention`` is not the exact Diffusers
                :class:`Attention` class or does not expose the expected
                generic Q/K/V projections.
            ValueError: If the requested output projection cannot be patched.

        Returns:
            None.
        """

        super().__init__()
        required = ("to_q", "to_k", "to_v")
        if not all(hasattr(attention, name) for name in required):
            raise TypeError("NunchakuAttention requires an attention module with to_q, to_k, and to_v projections")

        self._copy_attention_attributes(attention)
        with torch.device("meta"):
            to_qkv = fuse_linears([attention.to_q, attention.to_k, attention.to_v])
        setattr(self, qkv_attr, svdq_from_linear(to_qkv, context, **kwargs))

        if hasattr(attention, "norm_q"):
            self.norm_q = attention.norm_q
        if hasattr(attention, "norm_k"):
            self.norm_k = attention.norm_k

        if patch_output:
            if not hasattr(attention, "to_out") or len(attention.to_out) == 0:
                raise ValueError("NunchakuAttention expected a non-empty to_out projection list")
            self.to_out = attention.to_out
            self.to_out[0] = svdq_from_linear(self.to_out[0], context, **kwargs)

        self.processor = processor
        self._nunchaku_lite_attention_patched = True
        _mark_patched_module(self)

    def _copy_attention_attributes(self, attention: nn.Module) -> None:
        """Copy non-module metadata from a Diffusers attention module.

        Args:
            attention: Source attention module.

        Returns:
            None.
        """

        for name, value in attention.__dict__.items():
            if name.startswith("_") or name in {"training"}:
                continue
            if isinstance(value, (nn.Module, nn.Parameter)):
                continue
            setattr(self, name, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **cross_attention_kwargs,
    ):
        """Dispatch attention to the installed processor.

        Args:
            hidden_states: Main hidden states.
            encoder_hidden_states: Optional encoder/context states.
            attention_mask: Optional attention mask.
            **cross_attention_kwargs: Extra kwargs consumed by the processor,
                such as packed rotary embeddings.

        Returns:
            Output returned by the installed attention processor.
        """

        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )


def patch_attention_module(
    attention: nn.Module,
    processor,
    context: SVDQPatchContext | None = None,
    *,
    qkv_attr: str = "to_qkv",
    patch_output: bool = True,
    **kwargs,
) -> nn.Module:
    """Create a Nunchaku attention module for a generic Diffusers attention.

    The helper intentionally only validates/selects the generic attention path
    and initializes :class:`NunchakuAttention`. Model-specific attention classes
    with custom topology should still use rewritten lite classes directly.

    Args:
        attention: Generic Diffusers attention module with ``to_q``, ``to_k``,
            ``to_v``, and optionally ``to_out``.
        processor: Processor instance used by the patched module.
        context: Optional SVDQ settings applied to replacement projections.
        qkv_attr: Attribute name that will hold the fused QKV projection.
        patch_output: Whether to replace ``attention.to_out[0]`` with SVDQ.
        **kwargs: Additional SVDQ constructor overrides.

    Returns:
        New :class:`NunchakuAttention` instance.

    Raises:
        TypeError: If the module is not the exact Diffusers
            :class:`Attention` class.
        ValueError: If the requested output projection cannot be patched.
    """

    if isinstance(attention, NunchakuAttention):
        return _mark_patched_module(attention)
    if attention.__class__ is not Attention:
        raise TypeError(
            "patch_attention_module only supports the exact diffusers Attention class; "
            f"got {attention.__class__.__module__}.{attention.__class__.__name__}"
        )
    return NunchakuAttention(
        attention,
        processor,
        context,
        qkv_attr=qkv_attr,
        patch_output=patch_output,
        **kwargs,
    )


def patch_modules_recursively(
    module: nn.Module,
    *,
    path: str = "",
    skips: Callable[[str, nn.Module], bool] | None = None,
    module_converters: dict[type[nn.Module], Callable[[nn.Module], nn.Module]] | None = None,
) -> ModulePatchReport:
    """Recursively patch modules through explicit converters.

    The traversal mutates ``module`` in place by walking its direct children,
    then descending into unhandled child modules. Adapter code should state
    every replacement explicitly through ``module_converters`` and use
    ``skips`` to keep out-of-scope subtrees or leaf modules dense.

    Args:
        module: Root module whose children should be inspected and mutated.
            The root itself is not replaced; pass its parent when the root may
            be an attention or linear module.
        path: Dot-separated path prefix for ``module``. Recursive calls append
            child names to this value before invoking filters and factories.
        skips: Optional callable receiving ``(child_path, child)``. Returning
            ``True`` skips the child and all of its descendants. It can be used
            for both whole subtrees and leaf linears.
        module_converters: Optional dictionary keyed by exact module class.
            When a child has a matching ``child.__class__`` entry, the
            converter is called with the child and its return value replaces
            the child before recursive handling continues. Converters are
            intended for all replacement decisions, including dense
            ``nn.Linear`` replacement through :func:`svdq_from_linear`.

    Returns:
        :class:`ModulePatchReport` containing replacement and skip counts for
        this traversal.
    """

    report = ModulePatchReport()
    module_converters = module_converters or {}

    for name, child in list(module.named_children()):
        child_path = f"{path}.{name}" if path else name

        if skips is not None and skips(child_path, child):
            report.skipped_modules += 1
            continue

        if _is_patched_module(child):
            report.skipped_modules += 1
            continue

        module_converter = module_converters.get(child.__class__)
        if module_converter is not None:
            child = _mark_patched_module(module_converter(child))
            setattr(module, name, child)
            report.converted_modules += 1

        report.add(
            patch_modules_recursively(
                child,
                path=child_path,
                skips=skips,
                module_converters=module_converters,
            )
        )

    return report

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
