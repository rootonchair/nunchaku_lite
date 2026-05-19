import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from diffusers import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxIPAdapterAttnProcessor
from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.flux import FluxAdapter, NunchakuFluxAttention, NunchakuFluxAttnProcessor, convert_flux_state_dict


def make_tiny_flux_transformer():
    return FluxTransformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 6, 6),
    )


def test_flux_adapter_matches_diffusers_transformer():
    transformer = make_tiny_flux_transformer()
    assert FluxAdapter().matches(transformer)


def test_convert_flux_state_dict_maps_original_nunchaku_keys():
    state = {
        "transformer_blocks.0.qkv_proj.qweight": torch.empty(1),
        "transformer_blocks.0.qkv_proj_context.lora_down": torch.empty(1),
        "transformer_blocks.0.out_proj_context.smooth_orig": torch.empty(1),
        "single_transformer_blocks.0.out_proj.lora_up": torch.empty(1),
    }

    converted = convert_flux_state_dict(state)

    assert "transformer_blocks.0.attn.to_qkv.qweight" in converted
    assert "transformer_blocks.0.attn.add_qkv_proj.proj_down" in converted
    assert "transformer_blocks.0.attn.to_add_out.smooth_factor_orig" in converted
    assert "single_transformer_blocks.0.attn.to_out.proj_up" in converted


def test_convert_flux_state_dict_leaves_corrected_keys_unchanged():
    state = {
        "transformer_blocks.0.attn.to_qkv.qweight": torch.empty(1),
        "transformer_blocks.0.ff_context.net.0.proj.smooth_factor": torch.empty(1),
        "single_transformer_blocks.0.attn.to_out.proj_up": torch.empty(1),
    }

    converted = convert_flux_state_dict(state)

    assert converted is state
    assert set(converted) == set(state)


def test_patch_transformer_patches_flux_from_synthetic_checkpoint(tmp_path):
    rank = 4
    source = make_tiny_flux_transformer()
    adapter = FluxAdapter()
    adapter.patch(
        source,
        {},
        {"rank": rank},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )
    state = source.state_dict()
    checkpoint = tmp_path / "flux-lite.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_flux_transformer()
    returned = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)

    assert returned is transformer
    assert transformer._nunchaku_lite_patched
    assert transformer._nunchaku_lite_target == "flux"
    assert transformer.transformer_blocks[0].__class__.__name__ == "NunchakuFluxTransformerBlock"
    assert transformer.single_transformer_blocks[0].__class__.__name__ == "NunchakuFluxSingleTransformerBlock"
    assert transformer.transformer_blocks[0].attn.__class__.__name__ == "NunchakuFluxAttention"


def test_patch_transformer_is_idempotent_for_flux(tmp_path):
    rank = 4
    source = make_tiny_flux_transformer()
    FluxAdapter().patch(
        source,
        {},
        {"rank": rank},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )
    checkpoint = tmp_path / "flux-lite.safetensors"
    save_file(source.state_dict(), checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_flux_transformer()
    first = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)
    second = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)

    assert first is second is transformer


def test_flux_attention_wraps_ip_adapter_processor():
    base = make_tiny_flux_transformer().transformer_blocks[0].attn
    ip_processor = FluxIPAdapterAttnProcessor(
        hidden_size=base.inner_dim,
        cross_attention_dim=8,
        num_tokens=(2,),
        scale=0.5,
        dtype=torch.bfloat16,
    )

    attn = NunchakuFluxAttention(base, processor=ip_processor, precision="int4", rank=4, torch_dtype=torch.bfloat16)

    assert isinstance(attn.processor, NunchakuFluxAttnProcessor)
    assert attn.processor.supports_ip_adapter
    assert len(attn.processor.to_k_ip) == 1
    assert len(attn.processor.to_v_ip) == 1
