from typing import Any

import torch
import torch.nn as nn
from diffusers.models.activations import GELU
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)
from packaging.version import Version
import diffusers
import torch.nn.functional as F

from ..core import PatchOptions, register_adapter
from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
from ..ops.fused import fused_gelu_mlp, fused_qkv_norm_rotary
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


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("Rotary dimension must be even.")
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    batch_size, seq_len = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.sin(out), torch.cos(out)], dim=-1)
    return out.view(batch_size, seq_len, dim // 2, 1, 2).float()


class LiteFluxPosEmbed(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: tuple[int, ...] | list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def prepare_flux_rotary(
    image_rotary_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    text_tokens: int,
    image_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if image_rotary_emb is None:
        return None
    if isinstance(image_rotary_emb, tuple):
        raise ValueError("nunchaku_lite Flux expects packed Nunchaku rotary embeddings, not Diffusers cos/sin tuples.")
    if image_rotary_emb.ndim == 6:
        image_rotary_emb = image_rotary_emb.reshape(1, text_tokens + image_tokens, *image_rotary_emb.shape[3:])
    if image_rotary_emb.shape[1] != text_tokens + image_tokens:
        raise ValueError("Unexpected Flux rotary token count")
    rotary_txt = pack_rotemb(pad_tensor(image_rotary_emb[:, :text_tokens], 256, 1))
    rotary_img = pack_rotemb(pad_tensor(image_rotary_emb[:, text_tokens:], 256, 1))
    rotary_single = pack_rotemb(pad_tensor(image_rotary_emb, 256, 1))
    return rotary_txt, rotary_img, rotary_single


class LiteAdaLayerNormZero(nn.Module):
    def __init__(self, other: AdaLayerNormZero, scale_shift: float = 1.0, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.scale_shift = scale_shift
        self.emb = other.emb
        self.silu = other.silu
        self.linear = AWQW4A16Linear.from_linear(other.linear, torch_dtype=torch_dtype)
        self.norm = other.norm

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        class_labels: torch.LongTensor | None = None,
        hidden_dtype: torch.dtype | None = None,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        emb = emb.view(emb.shape[0], -1, 6).permute(2, 0, 1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb
        norm_x = self.norm(x)
        if self.scale_shift != 0:
            scale_msa.add_(self.scale_shift)
            scale_mlp.add_(self.scale_shift)
        norm_x = norm_x * scale_msa[:, None] + shift_msa[:, None]
        return norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class LiteAdaLayerNormZeroSingle(nn.Module):
    def __init__(
        self, other: AdaLayerNormZeroSingle, scale_shift: float = 1.0, torch_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.scale_shift = scale_shift
        self.silu = other.silu
        self.linear = AWQW4A16Linear.from_linear(other.linear, torch_dtype=torch_dtype)
        self.norm = other.norm

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        emb = emb.view(emb.shape[0], -1, 3).permute(2, 0, 1)
        shift_msa, scale_msa, gate_msa = emb
        if self.scale_shift != 0:
            scale_msa.add_(self.scale_shift)
        norm_x = self.norm(x)
        norm_x = norm_x * scale_msa[:, None] + shift_msa[:, None]
        return norm_x, gate_msa


class LiteFluxAttnProcessor(nn.Module):
    _attention_backend = None
    _parallel_config = None

    def __init__(self, diffusers_processor: nn.Module | None = None):
        super().__init__()
        self.hidden_size = None
        self.cross_attention_dim = None
        self.scale = None
        self.to_k_ip = None
        self.to_v_ip = None
        if diffusers_processor is not None and diffusers_processor.__class__.__name__ == "FluxIPAdapterAttnProcessor":
            self.hidden_size = diffusers_processor.hidden_size
            self.cross_attention_dim = diffusers_processor.cross_attention_dim
            self.scale = diffusers_processor.scale
            self.to_k_ip = diffusers_processor.to_k_ip
            self.to_v_ip = diffusers_processor.to_v_ip

    @property
    def supports_ip_adapter(self) -> bool:
        return self.to_k_ip is not None and self.to_v_ip is not None

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        ip_hidden_states: list[torch.Tensor] | None = None,
        ip_adapter_masks: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rotary_hidden = image_rotary_emb[0] if isinstance(image_rotary_emb, tuple) else image_rotary_emb
        query, key, value = self._project_qkv(hidden_states, attn.to_qkv, attn.norm_q, attn.norm_k, rotary_hidden, attn)
        ip_query = query

        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            if not isinstance(image_rotary_emb, tuple):
                raise ValueError("Joint Flux attention requires image/text rotary embeddings.")
            encoder_query, encoder_key, encoder_value = self._project_qkv(
                encoder_hidden_states,
                attn.add_qkv_proj,
                attn.norm_added_q,
                attn.norm_added_k,
                image_rotary_emb[1],
                attn,
            )
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        ip_query = ip_query.transpose(1, 2)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0)
        output = output.transpose(1, 2).reshape(hidden_states.shape[0], -1, attn.heads * attn.head_dim)
        output = output.to(query.dtype)

        if encoder_hidden_states is None:
            return attn.to_out(output)

        encoder_output, hidden_output = output.split_with_sizes(
            [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1
        )
        hidden_output = attn.to_out[0](hidden_output.contiguous())
        hidden_output = attn.to_out[1](hidden_output)
        encoder_output = attn.to_add_out(encoder_output.contiguous())

        if ip_hidden_states is None:
            return hidden_output, encoder_output
        if not self.supports_ip_adapter:
            raise ValueError("Flux IP-Adapter states were provided, but this attention processor has no IP weights.")

        ip_output = torch.zeros_like(hidden_output)
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
        ):
            ip_key = to_k_ip(current_ip_hidden_states).view(
                hidden_states.shape[0], -1, attn.heads, attn.head_dim
            )
            ip_value = to_v_ip(current_ip_hidden_states).view(
                hidden_states.shape[0], -1, attn.heads, attn.head_dim
            )
            ip_key = ip_key.transpose(1, 2)
            ip_value = ip_value.transpose(1, 2)
            current_ip_output = F.scaled_dot_product_attention(
                ip_query,
                ip_key,
                ip_value,
                attn_mask=None,
                dropout_p=0.0,
            )
            current_ip_output = current_ip_output.transpose(1, 2).reshape(
                hidden_states.shape[0], -1, attn.heads * attn.head_dim
            )
            ip_output += scale * current_ip_output.to(ip_query.dtype)

        return hidden_output, encoder_output, ip_output

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        projection: SVDQW4A4Linear,
        norm_q: nn.Module,
        norm_k: nn.Module,
        rotary_emb: torch.Tensor | None,
        attn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = fused_qkv_norm_rotary(hidden_states, projection, norm_q, norm_k, rotary_emb=rotary_emb)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, attn.head_dim))
        key = key.unflatten(-1, (attn.heads, attn.head_dim))
        value = value.unflatten(-1, (attn.heads, attn.head_dim))
        return query, key, value


class LiteFluxAttention(nn.Module):
    def __init__(
        self,
        other: FluxAttention,
        processor: str | nn.Module | None = None,
        context: SVDQPatchContext | None = None,
        **kwargs,
    ):
        super().__init__()
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.use_bias = other.use_bias
        self.dropout = other.dropout
        self.out_dim = other.out_dim
        self.context_pre_only = other.context_pre_only
        self.pre_only = other.pre_only
        self.heads = other.heads
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.added_proj_bias = other.added_proj_bias

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = svdq_from_linear(to_qkv, context, **kwargs)

        if not self.pre_only:
            self.to_out = other.to_out
            self.to_out[0] = svdq_from_linear(self.to_out[0], context, **kwargs)

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = other.norm_added_q
            self.norm_added_k = other.norm_added_k
            with torch.device("meta"):
                add_qkv_proj = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
            self.add_qkv_proj = svdq_from_linear(add_qkv_proj, context, **kwargs)
            self.to_add_out = svdq_from_linear(other.to_add_out, context, **kwargs)

        self.set_processor(other.processor if processor is None else processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
            **kwargs,
        )

    def get_processor(self):
        return self.processor

    def set_processor(self, processor) -> None:
        if isinstance(processor, str):
            if processor not in ("flashattn2", "sdpa"):
                raise ValueError(f"Processor {processor} is not supported")
            self.processor = LiteFluxAttnProcessor()
            return

        name = processor.__class__.__name__
        if name in ("FluxAttnProcessor", "LiteFluxAttnProcessor"):
            self.processor = LiteFluxAttnProcessor()
        elif name == "FluxIPAdapterAttnProcessor":
            self.processor = LiteFluxAttnProcessor(processor)
        else:
            raise ValueError(f"Processor {name} is not supported")


class LiteFluxFeedForward(nn.Module):
    def __init__(self, ff: nn.Module, context: SVDQPatchContext | None = None, **kwargs):
        super().__init__()
        self.net = patch_svdq_linears(ff.net, context, **kwargs)
        if len(self.net) > 2 and isinstance(self.net[2], SVDQW4A4Linear):
            self.net[2].act_unsigned = self.net[2].precision != "nvfp4"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            len(self.net) > 2
            and isinstance(self.net[0], GELU)
            and isinstance(self.net[0].proj, SVDQW4A4Linear)
            and isinstance(self.net[2], SVDQW4A4Linear)
        ):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class LiteFluxTransformerBlock(nn.Module):
    def __init__(
        self,
        block: FluxTransformerBlock,
        scale_shift: float = 0.0,
        context: SVDQPatchContext | None = None,
        **kwargs,
    ):
        super().__init__()
        torch_dtype = context.torch_dtype if context is not None else kwargs.get("torch_dtype", torch.bfloat16)
        self.norm1 = LiteAdaLayerNormZero(block.norm1, scale_shift=scale_shift, torch_dtype=torch_dtype)
        self.norm1_context = LiteAdaLayerNormZero(
            block.norm1_context, scale_shift=scale_shift, torch_dtype=torch_dtype
        )
        self.attn = LiteFluxAttention(block.attn, context=context, **kwargs)
        self.norm2 = block.norm2
        self.norm2_context = block.norm2_context
        self.ff = LiteFluxFeedForward(block.ff, context=context, **kwargs)
        self.ff_context = LiteFluxFeedForward(block.ff_context, context=context, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        packed_rotary = prepare_flux_rotary(image_rotary_emb, encoder_hidden_states.shape[1], hidden_states.shape[1])
        attention_rotary = None if packed_rotary is None else (packed_rotary[1], packed_rotary[0])

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=attention_rotary,
            **(joint_attention_kwargs or {}),
        )
        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
            ip_attn_output = None
        else:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * scale_mlp[:, None] + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hidden_states)
        if ip_attn_output is not None:
            hidden_states = hidden_states + ip_attn_output

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * c_scale_mlp[:, None] + c_shift_mlp[:, None]
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(
            norm_encoder_hidden_states
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class LiteFluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        block: FluxSingleTransformerBlock,
        scale_shift: float = 0.0,
        context: SVDQPatchContext | None = None,
        **kwargs,
    ):
        super().__init__()
        torch_dtype = context.torch_dtype if context is not None else kwargs.get("torch_dtype", torch.bfloat16)
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm = LiteAdaLayerNormZeroSingle(block.norm, scale_shift=scale_shift, torch_dtype=torch_dtype)
        self.mlp_fc1 = svdq_from_linear(block.proj_mlp, context, **kwargs)
        self.act_mlp = block.act_mlp
        self.mlp_fc2 = svdq_from_linear(block.proj_out, context, in_features=self.mlp_hidden_dim, **kwargs)
        self.mlp_fc2.act_unsigned = self.mlp_fc2.precision != "nvfp4"
        self.attn = LiteFluxAttention(block.attn, context=context, **kwargs)
        self.attn.to_out = svdq_from_linear(block.proj_out, context, in_features=self.mlp_fc1.in_features, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        image_seq_len = hidden_states.shape[1]
        packed_rotary = prepare_flux_rotary(image_rotary_emb, text_seq_len, image_seq_len)
        attention_rotary = None if packed_rotary is None else packed_rotary[2]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        mlp_hidden_states = self.mlp_fc1(norm_hidden_states)
        mlp_hidden_states = self.act_mlp(mlp_hidden_states)
        mlp_hidden_states = self.mlp_fc2(mlp_hidden_states)
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=attention_rotary,
            **(joint_attention_kwargs or {}),
        )

        hidden_states = residual + gate.unsqueeze(1) * (attn_output + mlp_hidden_states)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


class FluxAdapter:
    target = "flux"

    def matches(self, transformer: torch.nn.Module) -> bool:
        return (
            transformer.__class__.__name__ == "FluxTransformer2DModel"
            and "transformer_flux" in transformer.__class__.__module__
        )

    def patch(
        self,
        transformer: torch.nn.Module,
        checkpoint_state: dict[str, torch.Tensor],
        quantization_config: dict[str, Any],
        options: PatchOptions,
    ) -> dict[str, torch.Tensor]:
        context = build_svdq_context(transformer, quantization_config, options)
        prepare_transformer_dtype(transformer, context)
        axes_dim = tuple(getattr(transformer.pos_embed, "axes_dim", transformer.config.axes_dims_rope))
        transformer.pos_embed = LiteFluxPosEmbed(dim=transformer.inner_dim, theta=10000, axes_dim=axes_dim)
        for index, block in enumerate(transformer.transformer_blocks):
            transformer.transformer_blocks[index] = LiteFluxTransformerBlock(block, scale_shift=0.0, context=context)
        for index, block in enumerate(transformer.single_transformer_blocks):
            transformer.single_transformer_blocks[index] = LiteFluxSingleTransformerBlock(
                block, scale_shift=0.0, context=context
            )

        checkpoint_state = convert_flux_state_dict(checkpoint_state)
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        transformer._nunchaku_lite_flux_patched = True
        return checkpoint_state


def convert_flux_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if "single_transformer_blocks." in key:
            if ".qkv_proj." in key:
                new_key = key.replace(".qkv_proj.", ".attn.to_qkv.")
            elif ".out_proj." in key:
                new_key = key.replace(".out_proj.", ".attn.to_out.")
            elif (".norm_q." in key or ".norm_k." in key) and ".attn.norm_" not in key:
                new_key = key.replace(".norm_k.", ".attn.norm_k.")
                new_key = new_key.replace(".norm_q.", ".attn.norm_q.")
        elif "transformer_blocks." in key:
            if ".mlp_context_fc1." in key:
                new_key = key.replace(".mlp_context_fc1.", ".ff_context.net.0.proj.")
            elif ".mlp_context_fc2." in key:
                new_key = key.replace(".mlp_context_fc2.", ".ff_context.net.2.")
            elif ".mlp_fc1." in key:
                new_key = key.replace(".mlp_fc1.", ".ff.net.0.proj.")
            elif ".mlp_fc2." in key:
                new_key = key.replace(".mlp_fc2.", ".ff.net.2.")
            elif ".qkv_proj_context." in key:
                new_key = key.replace(".qkv_proj_context.", ".attn.add_qkv_proj.")
            elif ".qkv_proj." in key:
                new_key = key.replace(".qkv_proj.", ".attn.to_qkv.")
            elif (".norm_q." in key or ".norm_k." in key) and ".attn.norm_" not in key:
                new_key = key.replace(".norm_k.", ".attn.norm_k.")
                new_key = new_key.replace(".norm_q.", ".attn.norm_q.")
            elif (".norm_added_q." in key or ".norm_added_k." in key) and ".attn.norm_added_" not in key:
                new_key = key.replace(".norm_added_k.", ".attn.norm_added_k.")
                new_key = new_key.replace(".norm_added_q.", ".attn.norm_added_q.")
            elif ".out_proj." in key:
                new_key = key.replace(".out_proj.", ".attn.to_out.0.")
            elif ".out_proj_context." in key:
                new_key = key.replace(".out_proj_context.", ".attn.to_add_out.")

        new_key = new_key.replace(".lora_down", ".proj_down")
        new_key = new_key.replace(".lora_up", ".proj_up")
        if ".smooth_orig" in new_key and ".smooth_factor_orig" not in new_key:
            new_key = new_key.replace(".smooth_orig", ".smooth_factor_orig")
        elif ".smooth" in new_key and ".smooth_factor" not in new_key:
            new_key = new_key.replace(".smooth", ".smooth_factor")
        new_state_dict[new_key] = value
    return new_state_dict


register_adapter(FluxAdapter())
