import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from diffusers import Flux2Transformer2DModel
from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.flux2 import (
    Flux2Adapter,
    NunchakuFlux2Attention,
    _pack_flux2_rotary_emb,
    convert_flux2_state_dict,
)


def make_tiny_flux2_transformer():
    return Flux2Transformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 4, 4, 4),
        timestep_guidance_channels=32,
    )


def test_flux2_adapter_matches_diffusers_transformer():
    transformer = make_tiny_flux2_transformer()
    assert Flux2Adapter().matches(transformer)


def test_pack_flux2_rotary_emb_uses_packed_nunchaku_layout():
    cos = torch.ones(17, 32)
    sin = torch.zeros(17, 32)

    packed = _pack_flux2_rotary_emb((cos, sin))

    assert packed.shape == (1, 256, 32)
    assert packed.dtype == torch.float32


def test_convert_flux2_state_dict_maps_original_nunchaku_keys():
    state = {
        "transformer_blocks.0.qkv_proj.lora_down": torch.empty(1),
        "transformer_blocks.0.qkv_proj_context.smooth_orig": torch.empty(1),
        "transformer_blocks.0.out_proj_context.smooth": torch.empty(1),
        "transformer_blocks.0.mlp_context_fc1.lora_up": torch.empty(1),
        "single_transformer_blocks.0.qkv_proj.lora_down": torch.empty(1),
        "single_transformer_blocks.0.mlp_fc2.smooth": torch.empty(1),
    }

    converted = convert_flux2_state_dict(state)

    assert "transformer_blocks.0.attn.to_qkv.proj_down" in converted
    assert "transformer_blocks.0.attn.to_added_qkv.smooth_factor_orig" in converted
    assert "transformer_blocks.0.attn.to_add_out.smooth_factor" in converted
    assert "transformer_blocks.0.ff_context.linear_in.proj_up" in converted
    assert "single_transformer_blocks.0.attn.qkv_proj.proj_down" in converted
    assert "single_transformer_blocks.0.attn.mlp_fc2.smooth_factor" in converted


def test_convert_flux2_state_dict_leaves_corrected_keys_unchanged():
    state = {
        "transformer_blocks.0.attn.to_qkv.proj_down": torch.empty(1),
        "transformer_blocks.0.ff_context.linear_in.smooth_factor": torch.empty(1),
        "single_transformer_blocks.0.attn.qkv_proj.proj_down": torch.empty(1),
    }

    converted = convert_flux2_state_dict(state)

    assert converted is state
    assert set(converted) == set(state)


def test_patch_transformer_patches_flux2_from_synthetic_checkpoint(tmp_path):
    rank = 4
    source = make_tiny_flux2_transformer()
    Flux2Adapter().patch(
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
    checkpoint = tmp_path / "flux2-lite.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_flux2_transformer()
    returned = patch_transformer(transformer, checkpoint, target="flux2", precision="int4", torch_dtype=torch.bfloat16)

    assert returned is transformer
    assert transformer._nunchaku_lite_patched
    assert transformer._nunchaku_lite_target == "flux2"
    assert transformer.transformer_blocks[0].__class__.__name__ == "NunchakuFlux2TransformerBlock"
    assert transformer.single_transformer_blocks[0].__class__.__name__ == "NunchakuFlux2SingleTransformerBlock"
    assert isinstance(transformer.transformer_blocks[0].attn, NunchakuFlux2Attention)
    assert hasattr(transformer, "_nunchaku_lite_flux2_original_forward")


def test_flux2_checkpoint_keys_match_nunchaku_module_names():
    transformer = make_tiny_flux2_transformer()
    Flux2Adapter().patch(
        transformer,
        {},
        {"rank": 4},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )
    keys = transformer.state_dict().keys()

    assert "transformer_blocks.0.attn.to_qkv.qweight" in keys
    assert "transformer_blocks.0.attn.to_added_qkv.qweight" in keys
    assert "single_transformer_blocks.0.attn.qkv_proj.qweight" in keys
    assert "single_transformer_blocks.0.attn.mlp_fc1.qweight" in keys
    assert "single_transformer_blocks.0.attn.out_proj.qweight" in keys
    assert "single_transformer_blocks.0.attn.mlp_fc2.qweight" in keys
