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
from ..utils import convert_fp16, pad_tensor, patch_scale_key


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


def patch_linear(module: nn.Module, linear_cls, **kwargs) -> nn.Module:
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
    def __init__(self, orig_attn: Attention, processor: str = "flashattn2", **kwargs):
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
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_out = orig_attn.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)
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
    def __init__(self, ff: DiffusersZImageFeedForward, **kwargs):
        super().__init__()
        self.net = patch_linear(_convert_z_image_ff(ff).net, SVDQW4A4Linear, **kwargs)

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
        rank = int(options.adapter_options.get("rank", quantization_config.get("rank", 32)))
        skip_refiners = bool(
            options.adapter_options.get("skip_refiners", quantization_config.get("skip_refiners", False))
        )
        torch_dtype = options.torch_dtype or next(transformer.parameters()).dtype

        if options.torch_dtype is not None:
            transformer.to(options.torch_dtype)

        kwargs = {"precision": options.precision, "rank": rank, "torch_dtype": torch_dtype}
        self._patch_transformer_blocks(transformer.layers, **kwargs)
        if skip_refiners:
            self._convert_feed_forward(transformer.noise_refiner)
            self._convert_feed_forward(transformer.context_refiner)
        else:
            self._patch_transformer_blocks(transformer.noise_refiner, **kwargs)
            self._patch_transformer_blocks(transformer.context_refiner, **kwargs)

        transformer.skip_refiners = skip_refiners
        self._install_rope_forward_wrapper(transformer)
        patch_scale_key(transformer, checkpoint_state)
        if torch_dtype == torch.float16:
            convert_fp16(transformer, checkpoint_state)
        return checkpoint_state

    def _patch_transformer_blocks(self, block_list: list[ZImageTransformerBlock], **kwargs) -> None:
        for block in block_list:
            block.attention = ZImageAttention(block.attention, **kwargs)
            block.attention.register_load_state_dict_post_hook(replace_fused_module)
            block.feed_forward = LiteZImageFeedForward(block.feed_forward, **kwargs)

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
