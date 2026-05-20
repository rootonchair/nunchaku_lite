"""Flux adapter for patching Diffusers Flux transformers with Nunchaku Lite modules."""

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxSingleTransformerBlock,
)
from packaging.version import Version
import diffusers
import torch.nn.functional as F

from ..core import PatchOptions, register_adapter
from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..ops.fused import fused_qkv_norm_rotary
from .common import (
    SVDQPatchContext,
    build_svdq_context,
    finalize_svdq_checkpoint,
    fuse_linears,
    pack_rotemb,
    pad_tensor,
    patch_modules_recursively,
    prepare_transformer_dtype,
    svdq_from_linear,
)


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """Build real-valued Flux rotary embedding pairs for one positional axis.

    Args:
        pos: Position ids for one axis with shape ``(batch, sequence)``.
        dim: Rotary dimension assigned to this axis. Must be even.
        theta: Frequency base used by Flux rotary embeddings.

    Returns:
        Float32 rotary tensor shaped ``(batch, sequence, dim // 2, 1, 2)``
        containing ``sin`` and ``cos`` pairs.

    Raises:
        ValueError: If ``dim`` is not even.
    """

    if dim % 2 != 0:
        raise ValueError("Rotary dimension must be even.")
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    batch_size, seq_len = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.sin(out), torch.cos(out)], dim=-1)
    return out.view(batch_size, seq_len, dim // 2, 1, 2).float()


class NunchakuFluxPosEmbed(nn.Module):
    """Flux positional embedder that returns RoPE tensors in the native packable layout."""

    def __init__(self, dim: int, theta: int, axes_dim: tuple[int, ...] | list[int]):
        """Store Flux rotary-axis dimensions and frequency base.

        Args:
            dim: Total transformer hidden dimension.
            theta: Rotary frequency base.
            axes_dim: Per-axis rotary dimensions.

        Returns:
            None.
        """

        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return concatenated per-axis rotary embeddings.

        Args:
            ids: Flux positional ids. The last dimension indexes rotary axes.

        Returns:
            Rotary tensor in the native packable layout expected by
            :func:`prepare_flux_rotary`.
        """

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
    """Split and pack Flux rotary embeddings for text, image, and joint streams.

    Args:
        image_rotary_emb: Diffusers Flux rotary tensor, or ``None`` when the
            caller is running without RoPE.
        text_tokens: Number of text tokens at the front of the joint sequence.
        image_tokens: Number of image tokens after the text tokens.

    Returns:
        ``None`` if no rotary embedding is provided. Otherwise returns packed
        ``(text_rope, image_rope, joint_rope)`` tensors.

    Raises:
        ValueError: If a Flux2-style tuple is provided or token counts do not
            match the rotary tensor.
    """

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


class NunchakuAdaLayerNormZero(nn.Module):
    """Flux AdaLayerNormZero variant whose projection uses AWQ W4A16 weights."""

    def __init__(
        self,
        other: AdaLayerNormZero,
        scale_shift: float = 1.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        return_scale_shift: float = 0.0,
    ):
        """Copy normalization components and replace the modulation projection.

        Args:
            other: Source Diffusers AdaLayerNormZero module.
            scale_shift: Additive offset applied to scale outputs.
            torch_dtype: Runtime dtype for AWQ modulation projection buffers.

        Returns:
            None.
        """

        super().__init__()
        self.scale_shift = scale_shift
        self.return_scale_shift = return_scale_shift
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
        """Apply adaptive normalization and return gates and MLP modulation.

        Args:
            x: Hidden states to normalize.
            timestep: Optional timestep input for the copied embedding module.
            class_labels: Optional class labels for the copied embedding module.
            hidden_dtype: Optional dtype hint for the copied embedding module.
            emb: Precomputed modulation embedding. Used when ``self.emb`` is
                absent.

        Returns:
            Tuple ``(norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)``.
        """

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
        if self.return_scale_shift != 0:
            scale_mlp = scale_mlp + self.return_scale_shift
        return norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class NunchakuAdaLayerNormZeroSingle(nn.Module):
    """Single-stream Flux AdaLayerNormZero variant using AWQ for modulation."""

    def __init__(
        self, other: AdaLayerNormZeroSingle, scale_shift: float = 1.0, torch_dtype: torch.dtype = torch.bfloat16
    ):
        """Copy single-stream normalization components and replace projection.

        Args:
            other: Source Diffusers AdaLayerNormZeroSingle module.
            scale_shift: Additive offset applied to scale outputs.
            torch_dtype: Runtime dtype for AWQ modulation projection buffers.

        Returns:
            None.
        """

        super().__init__()
        self.scale_shift = scale_shift
        self.silu = other.silu
        self.linear = AWQW4A16Linear.from_linear(other.linear, torch_dtype=torch_dtype)
        self.norm = other.norm

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply single-stream adaptive normalization.

        Args:
            x: Hidden states to normalize.
            emb: Precomputed modulation embedding.

        Returns:
            Tuple ``(norm_x, gate_msa)``.
        """

        emb = self.linear(self.silu(emb))
        emb = emb.view(emb.shape[0], -1, 3).permute(2, 0, 1)
        shift_msa, scale_msa, gate_msa = emb
        if self.scale_shift != 0:
            scale_msa.add_(self.scale_shift)
        norm_x = self.norm(x)
        norm_x = norm_x * scale_msa[:, None] + shift_msa[:, None]
        return norm_x, gate_msa


class NunchakuFluxAttnProcessor(nn.Module):
    """Attention processor for lite Flux attention modules, including optional IP-Adapter state."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, diffusers_processor: nn.Module | None = None):
        """Copy optional IP-Adapter projections from a Diffusers processor.

        Args:
            diffusers_processor: Optional Diffusers attention processor. When
                it is a ``FluxIPAdapterAttnProcessor``, IP-Adapter weights and
                scales are reused by the lite processor.

        Returns:
            None.
        """

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
        """Return whether this processor has IP-Adapter key/value projections.

        Args:
            None.

        Returns:
            ``True`` when both IP key and value projections are available.
        """

        return self.to_k_ip is not None and self.to_v_ip is not None

    def __call__(
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
        """Expose a concrete signature for Diffusers attention kwarg filtering."""

        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
            ip_hidden_states=ip_hidden_states,
            ip_adapter_masks=ip_adapter_masks,
            **kwargs,
        )

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
        """Run Flux attention using fused QKV projection.

        Args:
            attn: Lite Flux attention module that owns the projections.
            hidden_states: Image or joint hidden states.
            encoder_hidden_states: Optional text hidden states for double-stream
                attention.
            attention_mask: Optional attention mask.
            image_rotary_emb: Packed rotary tensor or ``(image, text)`` packed
                rotary tuple.
            ip_hidden_states: Optional IP-Adapter hidden states.
            ip_adapter_masks: Optional IP-Adapter masks. Currently accepted for
                API compatibility and not used by this processor.
            **kwargs: Additional Diffusers attention kwargs.

        Returns:
            Single-stream output tensor, ``(hidden, encoder)`` for double
            stream, or ``(hidden, encoder, ip)`` when IP-Adapter output is
            produced.

        Raises:
            ValueError: If double-stream rotary embeddings are not provided as
                an image/text tuple, or IP states are provided without IP
                projections.
        """

        image_rotary_emb = self._prepare_rotary(attn, image_rotary_emb, hidden_states, encoder_hidden_states)
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
            if hasattr(attn, "to_out"):
                return attn.to_out(output)
            return output

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

    def _prepare_rotary(
        self,
        attn,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None:
        """Normalize Diffusers Flux RoPE tensors to the packed native layout."""

        if image_rotary_emb is None or isinstance(image_rotary_emb, tuple):
            return image_rotary_emb
        if image_rotary_emb.ndim == 3:
            return image_rotary_emb
        if encoder_hidden_states is None:
            if image_rotary_emb.ndim == 6:
                image_rotary_emb = image_rotary_emb.reshape(1, hidden_states.shape[1], *image_rotary_emb.shape[3:])
            if image_rotary_emb.shape[1] != hidden_states.shape[1]:
                raise ValueError("Unexpected Flux rotary token count")
            return pack_rotemb(pad_tensor(image_rotary_emb, 256, 1))

        packed_rotary = prepare_flux_rotary(
            image_rotary_emb,
            text_tokens=encoder_hidden_states.shape[1],
            image_tokens=hidden_states.shape[1],
        )
        if packed_rotary is None:
            return None
        return packed_rotary[1], packed_rotary[0]

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        projection: SVDQW4A4Linear,
        norm_q: nn.Module,
        norm_k: nn.Module,
        rotary_emb: torch.Tensor | None,
        attn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to Q/K/V and apply Q/K norm plus packed RoPE.

        Args:
            hidden_states: Input states to project.
            projection: Fused SVDQ QKV projection.
            norm_q: Query normalization module.
            norm_k: Key normalization module.
            rotary_emb: Optional packed rotary embedding.
            attn: Attention module providing head metadata.

        Returns:
            Tuple of query, key, and value tensors shaped for attention.
        """

        qkv = fused_qkv_norm_rotary(hidden_states, projection, norm_q, norm_k, rotary_emb=rotary_emb)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, attn.head_dim))
        key = key.unflatten(-1, (attn.heads, attn.head_dim))
        value = value.unflatten(-1, (attn.heads, attn.head_dim))
        return query, key, value


class NunchakuFluxAttention:
    """Patch Diffusers Flux attention modules in place for Nunchaku kernels."""

    def __new__(
        cls,
        other: FluxAttention,
        processor: str | nn.Module | None = None,
        context: SVDQPatchContext | None = None,
        *,
        patch_output: bool = True,
        **kwargs,
    ) -> FluxAttention:
        attn = other
        if getattr(attn, "_nunchaku_lite_flux_attention_patched", False):
            if processor is not None:
                attn.set_processor(_flux_attention_processor(processor))
            return attn

        with torch.device("meta"):
            to_qkv = fuse_linears([attn.to_q, attn.to_k, attn.to_v])
        attn.to_qkv = svdq_from_linear(to_qkv, context, **kwargs)
        delattr(attn, "to_q")
        delattr(attn, "to_k")
        delattr(attn, "to_v")

        if patch_output and not attn.pre_only:
            attn.to_out[0] = svdq_from_linear(attn.to_out[0], context, **kwargs)

        if attn.added_kv_proj_dim is not None:
            with torch.device("meta"):
                add_qkv_proj = fuse_linears([attn.add_q_proj, attn.add_k_proj, attn.add_v_proj])
            attn.add_qkv_proj = svdq_from_linear(add_qkv_proj, context, **kwargs)
            attn.to_add_out = svdq_from_linear(attn.to_add_out, context, **kwargs)
            delattr(attn, "add_q_proj")
            delattr(attn, "add_k_proj")
            delattr(attn, "add_v_proj")

        attn.set_processor(_flux_attention_processor(attn.processor if processor is None else processor))
        attn._nunchaku_lite_flux_attention_patched = True
        return attn


def _flux_attention_processor(processor) -> NunchakuFluxAttnProcessor:
    if isinstance(processor, str):
        if processor not in ("flashattn2", "sdpa"):
            raise ValueError(f"Processor {processor} is not supported")
        return NunchakuFluxAttnProcessor()

    name = processor.__class__.__name__
    if name in ("FluxAttnProcessor", "NunchakuFluxAttnProcessor"):
        return NunchakuFluxAttnProcessor()
    if name == "FluxIPAdapterAttnProcessor":
        return NunchakuFluxAttnProcessor(processor)
    raise ValueError(f"Processor {name} is not supported")


def _patch_flux_feed_forward(ff: nn.Module, context: SVDQPatchContext, **kwargs) -> nn.Module:
    """Patch a Diffusers Flux feed-forward module in place."""

    patch_modules_recursively(
        ff.net,
        module_converters={nn.Linear: lambda linear: svdq_from_linear(linear, context, **kwargs)},
    )
    if len(ff.net) > 2 and isinstance(ff.net[2], SVDQW4A4Linear):
        ff.net[2].act_unsigned = ff.net[2].precision != "nvfp4"
    return ff


class NunchakuFluxSingleTransformerBlock(nn.Module):
    """Single-stream Flux block with split Nunchaku attention and MLP outputs."""

    def __init__(
        self,
        block: FluxSingleTransformerBlock,
        context: SVDQPatchContext,
        *,
        scale_shift: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        torch_dtype = context.torch_dtype if context is not None else kwargs.get("torch_dtype", torch.bfloat16)
        proj_mlp = block.proj_mlp
        proj_out = block.proj_out
        self.mlp_hidden_dim = block.mlp_hidden_dim
        self.norm = NunchakuAdaLayerNormZeroSingle(block.norm, scale_shift=scale_shift, torch_dtype=torch_dtype)
        self.mlp_fc1 = svdq_from_linear(proj_mlp, context, **kwargs)
        self.act_mlp = block.act_mlp
        self.mlp_fc2 = svdq_from_linear(proj_out, context, in_features=self.mlp_hidden_dim, **kwargs)
        self.mlp_fc2.act_unsigned = self.mlp_fc2.precision != "nvfp4"
        self.attn = NunchakuFluxAttention(block.attn, context=context, patch_output=False, **kwargs)
        self.attn.to_out = svdq_from_linear(proj_out, context, in_features=self.mlp_fc1.in_features, **kwargs)
        self._nunchaku_lite_flux_block_patched = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        mlp_hidden_states = self.mlp_fc1(norm_hidden_states)
        mlp_hidden_states = self.act_mlp(mlp_hidden_states)
        mlp_hidden_states = self.mlp_fc2(mlp_hidden_states)
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )

        hidden_states = residual + gate.unsqueeze(1) * (attn_output + mlp_hidden_states)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states



class FluxAdapter:
    """Adapter for Diffusers ``FluxTransformer2DModel`` checkpoints."""

    target = "flux"

    def matches(self, transformer: torch.nn.Module) -> bool:
        """Return whether ``transformer`` is a Diffusers Flux transformer.

        Args:
            transformer: Candidate module.

        Returns:
            ``True`` when the class name and module path match Diffusers Flux.
        """

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
        """Patch a Flux transformer in place and normalize checkpoint keys.

        Args:
            transformer: Diffusers Flux transformer to mutate.
            checkpoint_state: Checkpoint tensors to normalize for lite module
                names.
            quantization_config: Quantization metadata from the checkpoint.
            options: Normalized patch options.

        Returns:
            Normalized checkpoint state dict to load into the patched model.
        """

        context = build_svdq_context(transformer, quantization_config, options)
        prepare_transformer_dtype(transformer, context)
        axes_dim = tuple(getattr(transformer.pos_embed, "axes_dim", transformer.config.axes_dims_rope))
        transformer.pos_embed = NunchakuFluxPosEmbed(dim=transformer.inner_dim, theta=10000, axes_dim=axes_dim)
        self._patch_transformer(transformer, context)

        if _flux_state_dict_needs_conversion(checkpoint_state):
            checkpoint_state = convert_flux_state_dict(checkpoint_state)
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        transformer._nunchaku_lite_flux_patched = True
        from ..lora.core.runtime import bind_transformer_lora_methods
        from ..lora.flux import NunchakuFluxTransformerLoraMixin

        bind_transformer_lora_methods(transformer, NunchakuFluxTransformerLoraMixin)
        return checkpoint_state

    def patch_pipeline(
        self,
        pipeline: Any,
        *,
        component_name: str = "transformer",
        component: torch.nn.Module | None = None,
    ) -> None:
        """Attach Flux pipeline-level runtime APIs."""

        from ..lora.core.runtime import NunchakuPipelineLoraMixin, bind_pipeline_lora_methods

        bind_pipeline_lora_methods(pipeline, NunchakuPipelineLoraMixin)

    def _patch_transformer(self, transformer: torch.nn.Module, context: SVDQPatchContext) -> None:
        """Patch Flux block modules through the shared recursive traversal.

        Args:
            transformer: Flux transformer whose module tree should be patched.
            context: Shared SVDQ patch settings used by lite block
                replacements.

        Returns:
            None.
        """

        torch_dtype = context.torch_dtype
        patch_modules_recursively(
            transformer,
            skips=lambda _path, module: isinstance(module, nn.Linear),
            module_converters={
                AdaLayerNormZero: lambda norm: NunchakuAdaLayerNormZero(
                    norm,
                    scale_shift=0.0,
                    torch_dtype=torch_dtype,
                    return_scale_shift=-1.0,
                ),
                FluxAttention: lambda attn: NunchakuFluxAttention(attn, context=context),
                FeedForward: lambda ff: _patch_flux_feed_forward(ff, context),
                FluxSingleTransformerBlock: lambda block: NunchakuFluxSingleTransformerBlock(
                    block,
                    context=context,
                    scale_shift=0.0,
                ),
            },
        )


def convert_flux_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize Flux checkpoint key names to match lite module names.

    Args:
        state_dict: Raw checkpoint state dict.

    Returns:
        New state dict with projection, smooth-factor, and low-rank key names
        rewritten to match the lite module tree.
    """

    if not _flux_state_dict_needs_conversion(state_dict):
        return state_dict

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


def _flux_state_dict_needs_conversion(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(_is_uncorrected_flux_key(key) for key in state_dict)


def _is_uncorrected_flux_key(key: str) -> bool:
    double_block_key = "transformer_blocks." in key
    single_block_key = "single_transformer_blocks." in key
    if not double_block_key and not single_block_key:
        return False

    if key.endswith((".lora_down", ".lora_up", ".smooth", ".smooth_orig")):
        return True

    if single_block_key and ".attn." not in key:
        return any(marker in key for marker in (".qkv_proj.", ".out_proj.", ".norm_q.", ".norm_k."))

    if double_block_key and not any(marker in key for marker in (".attn.", ".ff.", ".ff_context.")):
        return any(
            marker in key
            for marker in (
                ".mlp_context_fc1.",
                ".mlp_context_fc2.",
                ".mlp_fc1.",
                ".mlp_fc2.",
                ".qkv_proj_context.",
                ".qkv_proj.",
                ".norm_q.",
                ".norm_k.",
                ".norm_added_q.",
                ".norm_added_k.",
                ".out_proj.",
                ".out_proj_context.",
            )
        )

    return False


register_adapter(FluxAdapter())
