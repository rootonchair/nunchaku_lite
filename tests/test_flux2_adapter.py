import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from diffusers import Flux2Transformer2DModel
from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.flux2 import Flux2Adapter, LiteFlux2Attention, _pack_flux2_rotary_emb


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
    assert transformer.transformer_blocks[0].__class__.__name__ == "LiteFlux2TransformerBlock"
    assert transformer.single_transformer_blocks[0].__class__.__name__ == "LiteFlux2SingleTransformerBlock"
    assert isinstance(transformer.transformer_blocks[0].attn, LiteFlux2Attention)
    assert hasattr(transformer, "_nunchaku_lite_flux2_original_forward")


def test_flux2_checkpoint_keys_match_lite_module_names():
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
