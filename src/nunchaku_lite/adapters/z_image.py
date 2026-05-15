"""Z-Image adapter for patching Diffusers transformers with Nunchaku Lite modules."""

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_z_image import FeedForward as DiffusersZImageFeedForward
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor

from ..core import PatchOptions, register_adapter
from ..linear import DenseRuntimeLoraLinear
from ..ops.fused import fused_qkv_norm_rotary
from .common import (
    SVDQPatchContext,
    build_svdq_context,
    finalize_svdq_checkpoint,
    pack_rotemb,
    pad_tensor,
    patch_modules_recursively,
    prepare_transformer_dtype,
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
            attn: Generic Diffusers attention module patched with a fused SVDQ
                ``to_qkv`` projection.
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


def _convert_z_image_ff(z_ff: nn.Module) -> nn.Module:
    """Convert Z-Image's custom SwiGLU feed-forward module.

    Args:
        z_ff: Diffusers Z-Image feed-forward module, or an already converted
            module.

    Returns:
        Standard Diffusers :class:`FeedForward` with equivalent dimensions, or
        the original module unchanged when it is not a Z-Image feed-forward
        module.

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

        self._patch_transformer(transformer, context, skip_refiners)
        self._patch_adaln_lora_linears(transformer, skip_refiners)

        transformer.skip_refiners = skip_refiners
        self._install_rope_forward_wrapper(transformer)
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        from ..lora.core.runtime import bind_transformer_lora_methods
        from ..lora.z_image import NunchakuZImageTransformerLoraMixin

        bind_transformer_lora_methods(transformer, NunchakuZImageTransformerLoraMixin)
        return checkpoint_state

    def patch_pipeline(
        self,
        pipeline: Any,
        *,
        component_name: str = "transformer",
        component: torch.nn.Module | None = None,
    ) -> None:
        """Attach Z-Image pipeline-level runtime LoRA APIs."""

        from ..lora.core.runtime import NunchakuPipelineLoraMixin, bind_pipeline_lora_methods

        bind_pipeline_lora_methods(pipeline, NunchakuPipelineLoraMixin)

    def _patch_transformer(
        self,
        transformer: torch.nn.Module,
        context: SVDQPatchContext,
        skip_refiners: bool,
    ) -> None:
        """Patch Z-Image attention and feed-forward modules in one traversal.

        Args:
            transformer: Z-Image transformer whose module tree should be
                traversed.
            context: Shared SVDQ patch settings.
            skip_refiners: Whether refiner attention and feed-forward linears
                should remain dense. Refiner feed-forward modules are still
                converted to the generic Diffusers layout for checkpoint key
                compatibility.

        Returns:
            None.
        """

        patch_modules_recursively(
            transformer,
            context,
            attention_processor_factory=lambda _path, _attention: ZImageSingleStreamAttnProcessor(),
            linear_filter=lambda path, _linear: self._should_patch_linear(path, skip_refiners),
            skip_subtree=lambda path, module: self._should_skip_subtree(path, module, skip_refiners),
            module_converters={DiffusersZImageFeedForward: _convert_z_image_ff},
        )

    def _patch_adaln_lora_linears(self, transformer: torch.nn.Module, skip_refiners: bool) -> None:
        """Wrap AdaLN modulation linears so runtime LoRAs can target them.

        Args:
            transformer: Z-Image transformer after block conversion.
            skip_refiners: Whether refiner blocks should stay outside the
                runtime LoRA target set.
        """

        for layer in transformer.layers:
            self._wrap_block_adaln_lora(layer)
        if not skip_refiners:
            for layer in transformer.noise_refiner:
                self._wrap_block_adaln_lora(layer)

    def _wrap_block_adaln_lora(self, block: torch.nn.Module) -> None:
        """Wrap one block's AdaLN modulation linear when present."""

        adaln = getattr(block, "adaLN_modulation", None)
        if adaln is None or not hasattr(adaln, "__getitem__"):
            return
        if not isinstance(adaln[0], nn.Linear) or isinstance(adaln[0], DenseRuntimeLoraLinear):
            return
        adaln[0] = DenseRuntimeLoraLinear.from_linear(adaln[0])

    def _should_patch_linear(self, path: str, skip_refiners: bool) -> bool:
        """Return whether a discovered dense linear should become SVDQ.

        Args:
            path: Dot-separated module path produced by the recursive
                traversal.
            skip_refiners: Whether refiner linears should remain dense.

        Returns:
            ``True`` only for feed-forward linears in active Z-Image block
            groups. Embeddings, final projections, modulation projections, and
            skipped refiners remain dense because those tensors are not laid
            out as SVDQ checkpoint entries.
        """

        if ".feed_forward." not in path:
            return False
        if path.startswith("layers."):
            return True
        if skip_refiners:
            return False
        return path.startswith(("noise_refiner.", "context_refiner."))

    def _should_skip_subtree(self, path: str, module: nn.Module, skip_refiners: bool) -> bool:
        """Return whether a child subtree should be excluded from traversal.

        Args:
            path: Dot-separated module path produced by the recursive
                traversal.
            module: Child module currently being considered by
                :func:`patch_modules_recursively`.
            skip_refiners: Whether refiner attention should remain dense.

        Returns:
            ``True`` for refiner attention modules when refiners are skipped;
            otherwise ``False`` so feed-forward conversion and active attention
            patching can proceed.
        """

        if not skip_refiners or module.__class__ is not Attention:
            return False
        return path.startswith(("noise_refiner.", "context_refiner."))

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
