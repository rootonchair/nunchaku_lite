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
from ..ops.gemm import svdq_gemm_w4a4_cuda
from ..ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
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


class ZImageRopeHook:
    def __init__(self):
        self.packed_cache = {}

    def __call__(self, module: nn.Module, input_args: tuple, input_kwargs: dict):
        freqs_cis = input_kwargs.get("freqs_cis")
        if freqs_cis is None:
            return None
        cache_key = freqs_cis.data_ptr()
        packed_freqs_cis = self.packed_cache.get(cache_key)
        if packed_freqs_cis is None:
            packed_freqs_cis = torch.view_as_real(freqs_cis).unsqueeze(3)
            packed_freqs_cis = torch.flip(packed_freqs_cis, dims=[-1])
            packed_freqs_cis = pack_rotemb(pad_tensor(packed_freqs_cis, 256, 1))
            self.packed_cache[cache_key] = packed_freqs_cis
        new_input_kwargs = input_kwargs.copy()
        new_input_kwargs["freqs_cis"] = packed_freqs_cis
        return input_args, new_input_kwargs


class ZImageFusedModule(nn.Module):
    def __init__(self, qkv: SVDQW4A4Linear, norm_q: nn.Module, norm_k: nn.Module):
        super().__init__()
        for name, param in qkv.named_parameters(prefix="qkv_"):
            setattr(self, name.replace(".", ""), param)
        self.qkv_precision = qkv.precision
        self.qkv_out_features = qkv.out_features
        for name, param in norm_q.named_parameters(prefix="norm_q_"):
            setattr(self, name.replace(".", ""), param)
        for name, param in norm_k.named_parameters(prefix="norm_k_"):
            setattr(self, name.replace(".", ""), param)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        x = x.view(batch_size * seq_len, channels)
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=self.qkv_proj_down,
            smooth=self.qkv_smooth_factor,
            fp4=self.qkv_precision == "nvfp4",
            pad_size=256,
        )
        output = torch.empty(batch_size * seq_len, self.qkv_out_features, dtype=x.dtype, device=x.device)
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qkv_qweight,
            out=output,
            ascales=ascales,
            wscales=self.qkv_wscales,
            lora_act_in=lora_act_out,
            lora_up=self.qkv_proj_up,
            bias=getattr(self, "qkv_bias", None),
            fp4=self.qkv_precision == "nvfp4",
            alpha=1.0 if self.qkv_precision == "nvfp4" else None,
            wcscales=self.qkv_wcscales if self.qkv_precision == "nvfp4" else None,
            norm_q=self.norm_q_weight,
            norm_k=self.norm_k_weight,
            rotary_emb=freqs_cis,
        )
        return output.view(batch_size, seq_len, -1)


class ZImageAttention(nn.Module):
    def __init__(
        self,
        orig_attn: Attention,
        processor: str = "flashattn2",
        context: SVDQPatchContext | None = None,
        **kwargs,
    ):
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
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def set_processor(self, processor: str) -> None:
        if processor != "flashattn2":
            raise ValueError(f"Processor {processor} is not supported")
        self.processor = ZImageSingleStreamAttnProcessor()


class ZImageSingleStreamAttnProcessor(ZSingleStreamAttnProcessor):
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = attn.fused_module(hidden_states, freqs_cis)
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
    def __init__(self, ff: DiffusersZImageFeedForward, context: SVDQPatchContext | None = None, **kwargs):
        super().__init__()
        self.net = patch_svdq_linears(_convert_z_image_ff(ff).net, context, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def replace_fused_module(module, incompatible_keys) -> None:
    module.fused_module = ZImageFusedModule(module.to_qkv, module.norm_q, module.norm_k)
    del module.to_qkv
    del module.norm_q
    del module.norm_k


class ZImageAdapter:
    target = "z_image"

    def matches(self, transformer: torch.nn.Module) -> bool:
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
        for block in block_list:
            block.attention = ZImageAttention(block.attention, context=context)
            block.attention.register_load_state_dict_post_hook(replace_fused_module)
            block.feed_forward = LiteZImageFeedForward(block.feed_forward, context=context)

    def _convert_feed_forward(self, block_list: list[ZImageTransformerBlock]) -> None:
        for block in block_list:
            block.feed_forward = _convert_z_image_ff(block.feed_forward)

    def _install_rope_forward_wrapper(self, transformer: torch.nn.Module) -> None:
        if getattr(transformer, "_nunchaku_lite_rope_wrapped", False):
            return

        original_forward = transformer.forward

        def register_rope_hook(rope_hook: ZImageRopeHook):
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
            rope_hook = ZImageRopeHook()
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
