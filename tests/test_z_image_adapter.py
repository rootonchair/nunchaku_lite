import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.z_image import ZImageAdapter


def make_tiny_z_image_transformer():
    return ZImageTransformer2DModel(
        in_channels=4,
        dim=64,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=4,
        n_kv_heads=4,
        cap_feat_dim=8,
        axes_dims=[4, 6, 6],
        axes_lens=[16, 16, 16],
    )


def test_z_image_adapter_matches_diffusers_transformer():
    transformer = make_tiny_z_image_transformer()
    assert ZImageAdapter().matches(transformer)


def test_patch_transformer_patches_z_image_from_synthetic_checkpoint(tmp_path):
    rank = 4
    source = make_tiny_z_image_transformer()
    adapter = ZImageAdapter()
    adapter.patch(
        source,
        {},
        {"rank": rank, "skip_refiners": False},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )
    state = source.state_dict()
    checkpoint = tmp_path / "z-image-lite.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank, "skip_refiners": False})})

    transformer = make_tiny_z_image_transformer()
    returned = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)

    assert returned is transformer
    assert transformer._nunchaku_lite_patched
    assert transformer._nunchaku_lite_target == "z_image"
    assert transformer._nunchaku_lite_rope_wrapped
    assert transformer.layers[0].attention.__class__.__name__ == "ZImageAttention"
    assert hasattr(transformer.layers[0].attention, "fused_module")
    assert transformer.layers[0].feed_forward.__class__.__name__ == "LiteZImageFeedForward"


def test_patch_transformer_is_idempotent(tmp_path):
    rank = 4
    source = make_tiny_z_image_transformer()
    ZImageAdapter().patch(
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
    checkpoint = tmp_path / "z-image-lite.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_z_image_transformer()
    first = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)
    second = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)

    assert first is second is transformer
