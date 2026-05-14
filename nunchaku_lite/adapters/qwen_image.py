"""Qwen-Image adapter for patching Diffusers transformers with Nunchaku Lite modules."""

import math
import types
from math import prod
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformerBlock,
    apply_rotary_emb_qwen,
)

from ..core import PatchOptions, register_adapter
from ..models.linear import AWQW4A16Linear, SVDQW4A4Linear
from ..ops.fused import fused_gelu_mlp
from .common import (
    SVDQPatchContext,
    build_svdq_context,
    finalize_svdq_checkpoint,
    fuse_linears,
    patch_modules_recursively,
    prepare_transformer_dtype,
    svdq_from_linear,
)


def _compute_text_seq_len_from_mask(
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor | None,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """Compute Qwen text RoPE length and normalize text mask.

    Args:
        encoder_hidden_states: Text hidden states.
        encoder_hidden_states_mask: Optional text mask.

    Returns:
        Tuple of full text sequence length, per-sample active lengths, and a
        boolean mask.
    """

    batch_size, text_seq_len = encoder_hidden_states.shape[:2]
    if encoder_hidden_states_mask is None:
        return text_seq_len, None, None
    if encoder_hidden_states_mask.shape[:2] != (batch_size, text_seq_len):
        raise ValueError(
            f"`encoder_hidden_states_mask` shape {encoder_hidden_states_mask.shape} must match "
            f"(batch_size, text_seq_len)=({batch_size}, {text_seq_len})."
        )
    if encoder_hidden_states_mask.dtype != torch.bool:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)
    position_ids = torch.arange(text_seq_len, device=encoder_hidden_states.device, dtype=torch.long)
    active_positions = torch.where(encoder_hidden_states_mask, position_ids, position_ids.new_zeros(()))
    has_active = encoder_hidden_states_mask.any(dim=1)
    per_sample_len = torch.where(
        has_active,
        active_positions.max(dim=1).values + 1,
        torch.as_tensor(text_seq_len, device=encoder_hidden_states.device),
    )
    return text_seq_len, per_sample_len, encoder_hidden_states_mask


def _patch_linear(module: nn.Module, linear_cls, **kwargs) -> nn.Module:
    """Recursively replace dense linears in ``module`` with ``linear_cls``.

    Args:
        module: Module tree to patch in place.
        linear_cls: Replacement class exposing ``from_linear``.
        **kwargs: Constructor options forwarded to ``from_linear``.

    Returns:
        The same module instance after replacement.
    """

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, linear_cls.from_linear(child, **kwargs))
        else:
            _patch_linear(child, linear_cls, **kwargs)
    return module


class NunchakuQwenFeedForward(FeedForward):
    """Quantized Qwen feed-forward block using lite SVDQ linears."""

    def __init__(self, ff: FeedForward, context: SVDQPatchContext | None = None, **kwargs):
        """Patch a Diffusers feed-forward module in place.

        Args:
            ff: Source Diffusers feed-forward module.
            context: Shared SVDQ patch settings.
            **kwargs: Additional SVDQ constructor overrides.

        Returns:
            None.
        """

        super(FeedForward, self).__init__()
        linear_kwargs = context.linear_kwargs if context is not None else {}
        linear_kwargs.update(kwargs)
        self.net = _patch_linear(ff.net, SVDQW4A4Linear, **linear_kwargs)
        if len(self.net) > 2 and isinstance(self.net[2], SVDQW4A4Linear):
            self.net[2].act_unsigned = self.net[2].precision != "nvfp4"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the quantized feed-forward block.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Feed-forward output tensor.
        """

        if len(self.net) > 2 and isinstance(self.net[0], GELU) and isinstance(self.net[0].proj, SVDQW4A4Linear):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class NunchakuQwenImageAttnProcessor:
    """Attention processor for Qwen joint text-image attention."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen joint attention over text and image streams.

        Args:
            attn: Patched Qwen attention module.
            hidden_states: Image stream hidden states.
            encoder_hidden_states: Text stream hidden states.
            encoder_hidden_states_mask: Text mask kept for API compatibility.
            attention_mask: Optional attention mask.
            image_rotary_emb: ``(image_freqs, text_freqs)`` rotary embeddings.
            **kwargs: Ignored extra attention kwargs.

        Returns:
            Tuple ``(image_output, text_output)``.

        Raises:
            ValueError: If text stream states are missing.
        """

        if encoder_hidden_states is None:
            raise ValueError("NunchakuQwenImageAttnProcessor requires encoder_hidden_states")

        seq_txt = encoder_hidden_states.shape[1]
        img_query, img_key, img_value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        txt_query, txt_key, txt_value = attn.add_qkv_proj(encoder_hidden_states).chunk(3, dim=-1)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=getattr(attn, "_attention_backend", None),
            parallel_config=getattr(attn, "_parallel_config", None),
        )
        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt]
        img_attn_output = joint_hidden_states[:, seq_txt:]
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
        return img_attn_output, txt_attn_output


class NunchakuQwenAttention(nn.Module):
    """Lite replacement for Qwen-Image joint attention."""

    def __init__(self, other: nn.Module, context: SVDQPatchContext | None = None, **kwargs):
        """Copy attention metadata and replace projections with lite modules.

        Args:
            other: Source Diffusers Qwen attention module.
            context: Shared SVDQ patch settings.
            **kwargs: Additional SVDQ constructor overrides.

        Returns:
            None.
        """

        super().__init__()
        for name, value in other.__dict__.items():
            if name.startswith("_") or name in {"training"}:
                continue
            if isinstance(value, (nn.Module, nn.Parameter)):
                continue
            setattr(self, name, value)

        self.norm_cross = other.norm_cross
        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.norm_added_q = other.norm_added_q
        self.norm_added_k = other.norm_added_k
        self._attention_backend = getattr(other, "_attention_backend", None)
        self._parallel_config = getattr(other, "_parallel_config", None)
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
            add_qkv_proj = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
        self.to_qkv = svdq_from_linear(to_qkv, context, **kwargs)
        self.to_out = other.to_out
        self.to_out[0] = svdq_from_linear(self.to_out[0], context, **kwargs)
        self.add_qkv_proj = svdq_from_linear(add_qkv_proj, context, **kwargs)
        self.to_add_out = svdq_from_linear(other.to_add_out, context, **kwargs)
        self.processor = NunchakuQwenImageAttnProcessor()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch Qwen attention to the lite processor.

        Args:
            hidden_states: Image stream hidden states.
            encoder_hidden_states: Text stream hidden states.
            encoder_hidden_states_mask: Text mask kept for API compatibility.
            attention_mask: Optional attention mask.
            image_rotary_emb: Qwen rotary embeddings.
            **kwargs: Additional attention kwargs.

        Returns:
            Tuple ``(image_output, text_output)``.
        """

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )


class NunchakuQwenImageTransformerBlock(nn.Module):
    """Lite replacement for one Qwen-Image transformer block."""

    def __init__(
        self,
        other: QwenImageTransformerBlock,
        scale_shift: float = 1.0,
        context: SVDQPatchContext | None = None,
        **kwargs,
    ):
        """Patch Qwen block attention, modulation, and MLP modules.

        Args:
            other: Source Diffusers Qwen block.
            scale_shift: Value added to modulation scale terms.
            context: Shared quantization settings.
            **kwargs: Additional constructor overrides.

        Returns:
            None.
        """

        super().__init__()
        awq_kwargs = context.linear_kwargs if context is not None else {}
        awq_kwargs.update(kwargs)
        self.dim = other.dim
        self.img_mod = other.img_mod
        self.img_mod[1] = AWQW4A16Linear.from_linear(other.img_mod[1], **awq_kwargs)
        self.img_norm1 = other.img_norm1
        self.attn = NunchakuQwenAttention(other.attn, context=context, **kwargs)
        self.img_norm2 = other.img_norm2
        self.img_mlp = NunchakuQwenFeedForward(other.img_mlp, context=context, **kwargs)
        self.txt_mod = other.txt_mod
        self.txt_mod[1] = AWQW4A16Linear.from_linear(other.txt_mod[1], **awq_kwargs)
        self.txt_norm1 = other.txt_norm1
        self.txt_norm2 = other.txt_norm2
        self.txt_mlp = NunchakuQwenFeedForward(other.txt_mlp, context=context, **kwargs)
        self.scale_shift = scale_shift
        self.zero_cond_t = getattr(other, "zero_cond_t", False)

    def _modulate(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor,
        index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Qwen modulation.

        Args:
            x: Normalized hidden states.
            mod_params: Packed ``shift, scale, gate`` tensor.
            index: Optional image-token condition selector for zero-cond-t
                variants.

        Returns:
            Tuple of modulated hidden states and gate tensor.
        """

        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if self.scale_shift != 0:
            scale.add_(self.scale_shift)
        if index is not None:
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]
            index_expanded = index.unsqueeze(-1)
            shift = torch.where(index_expanded == 0, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
            scale = torch.where(index_expanded == 0, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
            gate = torch.where(index_expanded == 0, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
        else:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
            gate = gate.unsqueeze(1)
        return x * scale + shift, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        modulate_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one Qwen transformer block over text and image streams.

        Args:
            hidden_states: Image stream hidden states.
            encoder_hidden_states: Text stream hidden states.
            encoder_hidden_states_mask: Text stream mask.
            temb: Time/text embedding.
            image_rotary_emb: Qwen rotary embeddings.
            joint_attention_kwargs: Optional attention kwargs.
            modulate_index: Optional image-token selector for zero-cond-t
                variants.

        Returns:
            Tuple ``(encoder_hidden_states, hidden_states)``.
        """

        img_mod_params = self.img_mod(temb)
        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)
        img_mod_params = (
            img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
        )
        txt_mod_params = (
            txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)
        )
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1, modulate_index)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2, modulate_index)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)
        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


def lite_qwen_image_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: list[tuple[int, int, int]] | None = None,
    guidance: torch.Tensor = None,
    attention_kwargs: dict[str, Any] | None = None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict: bool = True,
) -> torch.Tensor | Transformer2DModelOutput:
    """Forward wrapper for patched Qwen-Image transformers.

    Args:
        self: Patched Diffusers Qwen transformer.
        hidden_states: Image hidden states.
        encoder_hidden_states: Text hidden states.
        encoder_hidden_states_mask: Text mask.
        timestep: Timestep tensor.
        img_shapes: Image token shape metadata for Qwen RoPE.
        guidance: Optional guidance tensor.
        attention_kwargs: Optional kwargs forwarded to attention blocks.
        controlnet_block_samples: Optional ControlNet residual samples.
        additional_t_cond: Optional additional timestep condition.
        return_dict: Whether to return Diffusers output object.

    Returns:
        Diffusers transformer output object, or tuple when ``return_dict`` is
        false.
    """

    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)
    if getattr(self, "zero_cond_t", False):
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [[0] * prod(sample[0]) + [1] * sum(prod(s) for s in sample[1:]) for sample in img_shapes],
            device=timestep.device,
            dtype=torch.int,
        )
    else:
        modulate_index = None
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)
    text_seq_len, _, encoder_hidden_states_mask = _compute_text_seq_len_from_mask(
        encoder_hidden_states, encoder_hidden_states_mask
    )
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    temb = (
        self.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )
    image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)
    block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
    if encoder_hidden_states_mask is not None:
        batch_size, image_seq_len = hidden_states.shape[:2]
        image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
        joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
        block_attention_kwargs["attention_mask"] = joint_attention_mask[:, None, None, :]

    for block_idx, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                None,
                temb,
                image_rotary_emb,
                block_attention_kwargs,
                modulate_index,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
            )
        if controlnet_block_samples is not None:
            interval_control = math.ceil(len(self.transformer_blocks) / len(controlnet_block_samples))
            hidden_states = hidden_states + controlnet_block_samples[block_idx // interval_control]

    if getattr(self, "zero_cond_t", False):
        temb = temb.chunk(2, dim=0)[0]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


class QwenImageAdapter:
    """Adapter for Diffusers ``QwenImageTransformer2DModel`` checkpoints."""

    target = "qwen_image"

    def matches(self, transformer: torch.nn.Module) -> bool:
        """Return whether ``transformer`` is a Diffusers Qwen-Image transformer.

        Args:
            transformer: Candidate module.

        Returns:
            ``True`` when class name and module path match Diffusers Qwen.
        """

        return (
            transformer.__class__.__name__ == "QwenImageTransformer2DModel"
            and "transformer_qwenimage" in transformer.__class__.__module__
        )

    def patch(
        self,
        transformer: torch.nn.Module,
        checkpoint_state: dict[str, torch.Tensor],
        quantization_config: dict[str, Any],
        options: PatchOptions,
    ) -> dict[str, torch.Tensor]:
        """Patch a Qwen-Image transformer in place.

        Args:
            transformer: Diffusers Qwen transformer to mutate.
            checkpoint_state: Checkpoint tensors to load after patching.
            quantization_config: Quantization metadata from the checkpoint.
            options: Normalized patch options.

        Returns:
            Checkpoint state dict to load into the patched transformer.
        """

        context = build_svdq_context(transformer, quantization_config, options)
        prepare_transformer_dtype(transformer, context)
        self._patch_transformer(transformer, context)
        transformer._nunchaku_lite_qwen_image_original_forward = transformer.forward
        transformer.forward = types.MethodType(lite_qwen_image_forward, transformer)
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        from ..lora.qwen_image import bind_qwen_image_lora_methods

        bind_qwen_image_lora_methods(transformer)
        transformer._nunchaku_lite_qwen_image_patched = True
        return checkpoint_state

    def _patch_transformer(self, transformer: torch.nn.Module, context: SVDQPatchContext) -> None:
        """Patch Qwen transformer blocks through one recursive traversal.

        Args:
            transformer: Qwen transformer whose module tree should be patched.
            context: Shared SVDQ/AWQ patch settings.

        Returns:
            None.
        """

        patch_modules_recursively(
            transformer,
            context,
            linear_filter=lambda _path, _linear: False,
            module_converters={
                QwenImageTransformerBlock: lambda block: NunchakuQwenImageTransformerBlock(
                    block, scale_shift=0, context=context
                )
            },
        )


register_adapter(QwenImageAdapter())
