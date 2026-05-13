"""Z-Image adapter for patching Diffusers transformers with Nunchaku Lite modules."""

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_z_image import FeedForward as DiffusersZImageFeedForward
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor

from ..core import PatchOptions, register_adapter
from ..models.linear import SVDQW4A4Linear
from ..ops.fused import fused_qkv_norm_rotary
from .common import (
    SVDQPatchContext,
    build_svdq_context,
    finalize_svdq_checkpoint,
    fuse_linears,
    pack_rotemb,
    pad_tensor,
    patch_svdq_linears,
    prepare_transformer_dtype,
    svdq_from_linear,
)


def _pack_z_image_rotary_emb(freqs_cis: torch.Tensor) -> torch.Tensor:
    """Convert Diffusers complex Z-Image RoPE tensors into Nunchaku's packed layout.

    Args:
        freqs_cis: Complex64 rotary tensor produced by Diffusers Z-Image,
            typically shaped ``(batch, sequence, dim // 2)``.

    Returns:
        Float32 packed rotary tensor accepted by the fused SVDQ GEMM kernels.
    """

    rotemb = torch.view_as_real(freqs_cis).unsqueeze(3)
    rotemb = torch.flip(rotemb, dims=[-1])
    return pack_rotemb(pad_tensor(rotemb, 256, 1))


class ZImageAttention(nn.Module):
    """Lite replacement for Diffusers Z-Image attention.

    The module preserves the Diffusers attention interface while replacing
    dense Q/K/V and output projections with SVDQ-backed projections.
    """

    def __init__(
        self,
        orig_attn: Attention,
        processor: str = "flashattn2",
        context: SVDQPatchContext | None = None,
        **kwargs,
    ):
        """Copy attention metadata and replace projections with SVDQ modules.

        Args:
            orig_attn: Source Diffusers attention module.
            processor: Processor name. Only ``"flashattn2"`` is supported.
            context: Shared SVDQ patch settings.
            **kwargs: Additional SVDQ constructor overrides.

        Returns:
            None.
        """

        super().__init__()
        self.inner_dim = orig_attn.inner_dim
        self.query_dim = orig_attn.query_dim
        self.use_bias = orig_attn.use_bias
        self.dropout = orig_attn.dropout
        self.out_dim = orig_attn.out_dim
        self.context_pre_only = orig_attn.context_pre_only
        self.pre_only = orig_attn.pre_only
        self.heads = orig_attn.heads
        self.rescale_output_factor = orig_attn.rescale_output_factor
        self.is_cross_attention = orig_attn.is_cross_attention
        self.norm_q = orig_attn.norm_q
        self.norm_k = orig_attn.norm_k

        with torch.device("meta"):
            to_qkv = fuse_linears([orig_attn.to_q, orig_attn.to_k, orig_attn.to_v])
        self.to_qkv = svdq_from_linear(to_qkv, context, **kwargs)
        self.to_out = orig_attn.to_out
        self.to_out[0] = svdq_from_linear(self.to_out[0], context, **kwargs)
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """Dispatch attention to the installed Z-Image lite processor.

        Args:
            hidden_states: Input hidden states.
            encoder_hidden_states: Unused cross-attention states kept for
                Diffusers API compatibility.
            attention_mask: Optional mask forwarded to attention dispatch.
            **cross_attention_kwargs: Extra Diffusers attention kwargs,
                including packed ``freqs_cis``.

        Returns:
            Attention output with the same leading dimensions as
            ``hidden_states``.
        """

        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def set_processor(self, processor: str) -> None:
        """Install the Z-Image attention processor.

        Args:
            processor: Processor name. Must be ``"flashattn2"``.

        Raises:
            ValueError: If ``processor`` is unsupported.

        Returns:
            None.
        """

        if processor != "flashattn2":
            raise ValueError(f"Processor {processor} is not supported")
        self.processor = ZImageSingleStreamAttnProcessor()


class ZImageSingleStreamAttnProcessor(ZSingleStreamAttnProcessor):
    """Attention processor that consumes packed RoPE and dispatches attention."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run fused QKV projection, attention, and output projection.

        Args:
            attn: :class:`ZImageAttention` module containing SVDQ projections.
            hidden_states: Input tensor for the attention block.
            encoder_hidden_states: Unused cross-attention states kept for API
                compatibility.
            attention_mask: Optional 2D or broadcasted attention mask.
            freqs_cis: Packed rotary embedding tensor installed by the forward
                wrapper.

        Returns:
            Projected attention output.
        """

        qkv = fused_qkv_norm_rotary(hidden_states, attn.to_qkv, attn.norm_q, attn.norm_k, rotary_emb=freqs_cis)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


def _convert_z_image_ff(z_ff: DiffusersZImageFeedForward) -> FeedForward:
    """Convert Z-Image's custom SwiGLU feed-forward module.

    Args:
        z_ff: Diffusers Z-Image feed-forward module, or an already converted
            module.

    Returns:
        Standard Diffusers :class:`FeedForward` with equivalent dimensions, or
        ``z_ff`` unchanged when it is not a Z-Image feed-forward module.

    Raises:
        ValueError: If the custom feed-forward projections have unexpected
            incompatible shapes.
    """

    if not isinstance(z_ff, DiffusersZImageFeedForward):
        return z_ff
    if z_ff.w1.in_features != z_ff.w3.in_features or z_ff.w1.out_features != z_ff.w3.out_features:
        raise ValueError("Unexpected Z-Image feed-forward projection shapes")
    converted_ff = FeedForward(
        dim=z_ff.w1.in_features,
        dim_out=z_ff.w2.out_features,
        dropout=0.0,
        activation_fn="swiglu",
        inner_dim=z_ff.w2.in_features,
        bias=False,
    ).to(dtype=z_ff.w1.weight.dtype, device=z_ff.w1.weight.device)
    return converted_ff


class LiteZImageFeedForward(nn.Module):
    """Quantized Z-Image feed-forward wrapper built from standard Diffusers layers."""

    def __init__(self, ff: DiffusersZImageFeedForward, context: SVDQPatchContext | None = None, **kwargs):
        """Convert the feed-forward block and replace linears with SVDQ modules.

        Args:
            ff: Source Z-Image feed-forward module.
            context: Shared SVDQ patch settings.
            **kwargs: Additional SVDQ constructor overrides.

        Returns:
            None.
        """

        super().__init__()
        self.net = patch_svdq_linears(_convert_z_image_ff(ff).net, context, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the quantized Z-Image feed-forward network.

        Args:
            hidden_states: Input tensor.

        Returns:
            Feed-forward output tensor.
        """

        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class ZImageAdapter:
    """Adapter for Diffusers ``ZImageTransformer2DModel`` checkpoints."""

    target = "z_image"

    def matches(self, transformer: torch.nn.Module) -> bool:
        """Return whether ``transformer`` is a Diffusers Z-Image transformer.

        Args:
            transformer: Candidate module.

        Returns:
            ``True`` when the class name and module path match Diffusers
            Z-Image.
        """

        return (
            transformer.__class__.__name__ == "ZImageTransformer2DModel"
            and "transformer_z_image" in transformer.__class__.__module__
        )

    def patch(
        self,
        transformer: torch.nn.Module,
        checkpoint_state: dict[str, torch.Tensor],
        quantization_config: dict[str, Any],
        options: PatchOptions,
    ) -> dict[str, torch.Tensor]:
        """Patch a Z-Image transformer in place and install packed-RoPE handling.

        Args:
            transformer: Diffusers Z-Image transformer to mutate.
            checkpoint_state: Checkpoint tensors to normalize for the patched
                module names.
            quantization_config: Quantization metadata from the checkpoint.
            options: Normalized patch options.

        Returns:
            The checkpoint state dict to load into the patched transformer.
        """

        context = build_svdq_context(transformer, quantization_config, options)
        skip_refiners = bool(
            options.adapter_options.get("skip_refiners", quantization_config.get("skip_refiners", False))
        )
        prepare_transformer_dtype(transformer, context)

        self._patch_transformer_blocks(transformer.layers, context)
        if skip_refiners:
            self._convert_feed_forward(transformer.noise_refiner)
            self._convert_feed_forward(transformer.context_refiner)
        else:
            self._patch_transformer_blocks(transformer.noise_refiner, context)
            self._patch_transformer_blocks(transformer.context_refiner, context)

        transformer.skip_refiners = skip_refiners
        self._install_rope_forward_wrapper(transformer)
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        return checkpoint_state

    def _patch_transformer_blocks(self, block_list: list[ZImageTransformerBlock], context: SVDQPatchContext) -> None:
        """Replace attention and feed-forward modules for Z-Image blocks.

        Args:
            block_list: Mutable list of Z-Image transformer blocks.
            context: Shared SVDQ patch settings.

        Returns:
            None.
        """

        for block in block_list:
            block.attention = ZImageAttention(block.attention, context=context)
            block.feed_forward = LiteZImageFeedForward(block.feed_forward, context=context)

    def _convert_feed_forward(self, block_list: list[ZImageTransformerBlock]) -> None:
        """Convert unquantized refiner feed-forward blocks.

        Args:
            block_list: Mutable list of Z-Image transformer blocks whose
                feed-forward modules should remain dense but use standard
                Diffusers ``FeedForward`` modules.

        Returns:
            None.
        """

        for block in block_list:
            block.feed_forward = _convert_z_image_ff(block.feed_forward)

    def _install_rope_forward_wrapper(self, transformer: torch.nn.Module) -> None:
        """Wrap transformer forward to pack Z-Image RoPE once per forward call.

        Args:
            transformer: Patched Z-Image transformer whose attention modules
                expect packed rotary embeddings.

        Returns:
            None.
        """

        if getattr(transformer, "_nunchaku_lite_rope_wrapped", False):
            return

        original_forward = transformer.forward

        def register_rope_hook(rope_hook):
            """Register a pre-hook on all quantized attention modules.

            Args:
                rope_hook: Callable forward pre-hook that rewrites
                    ``freqs_cis`` in keyword arguments.

            Returns:
                Hook handles that must be removed after forward completes.
            """

            handles = []
            for layer in transformer.layers:
                handles.append(layer.attention.register_forward_pre_hook(rope_hook, with_kwargs=True))
            if not getattr(transformer, "skip_refiners", False):
                for layer in transformer.noise_refiner:
                    handles.append(layer.attention.register_forward_pre_hook(rope_hook, with_kwargs=True))
                for layer in transformer.context_refiner:
                    handles.append(layer.attention.register_forward_pre_hook(rope_hook, with_kwargs=True))
            return handles

        def forward_with_packed_rope(*args, **kwargs):
            """Run the original forward with a per-call packed-RoPE cache.

            Args:
                *args: Positional arguments for the original transformer
                    forward.
                **kwargs: Keyword arguments for the original transformer
                    forward.

            Returns:
                Whatever the original transformer forward returns.
            """

            packed_cache = {}

            def rope_hook(module: nn.Module, input_args: tuple, input_kwargs: dict):
                """Replace a complex ``freqs_cis`` kwarg with a packed tensor.

                Args:
                    module: Attention module receiving the hook.
                    input_args: Positional inputs received by the module.
                    input_kwargs: Keyword inputs received by the module.

                Returns:
                    ``None`` when no ``freqs_cis`` is present, otherwise the
                    rewritten ``(input_args, input_kwargs)`` tuple expected by
                    PyTorch forward pre-hooks.
                """

                freqs_cis = input_kwargs.get("freqs_cis")
                if freqs_cis is None:
                    return None
                cache_key = freqs_cis.data_ptr()
                packed_freqs_cis = packed_cache.get(cache_key)
                if packed_freqs_cis is None:
                    packed_freqs_cis = _pack_z_image_rotary_emb(freqs_cis)
                    packed_cache[cache_key] = packed_freqs_cis
                new_input_kwargs = input_kwargs.copy()
                new_input_kwargs["freqs_cis"] = packed_freqs_cis
                return input_args, new_input_kwargs

            handles = register_rope_hook(rope_hook)
            try:
                return original_forward(*args, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()

        transformer._nunchaku_lite_original_forward = original_forward
        transformer.forward = forward_with_packed_rope
        transformer._nunchaku_lite_rope_wrapped = True


register_adapter(ZImageAdapter())
