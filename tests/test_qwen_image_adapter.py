import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from diffusers import QwenImageTransformer2DModel
from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.qwen_image import (
    NunchakuQwenAttention,
    NunchakuQwenFeedForward,
    NunchakuQwenImageTransformerBlock,
    QwenImageAdapter,
)
from nunchaku_lite.models.linear import AWQW4A16Linear, SVDQW4A4Linear


def make_tiny_qwen_image_transformer():
    return QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=4,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 6, 6),
    )


def patch_tiny_qwen_image(transformer, rank: int = 4):
    return QwenImageAdapter().patch(
        transformer,
        {},
        {"rank": rank},
        SimpleNamespace(
            precision="nvfp4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )


def test_qwen_image_adapter_matches_diffusers_transformer():
    transformer = make_tiny_qwen_image_transformer()
    assert QwenImageAdapter().matches(transformer)


def test_patch_transformer_patches_qwen_image_from_synthetic_fp4_checkpoint(tmp_path):
    rank = 4
    source = make_tiny_qwen_image_transformer()
    patch_tiny_qwen_image(source, rank=rank)
    checkpoint = tmp_path / "qwen-image-lite-fp4.safetensors"
    save_file(source.state_dict(), checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_qwen_image_transformer()
    returned = patch_transformer(
        transformer,
        checkpoint,
        target="qwen_image",
        precision="fp4",
        torch_dtype=torch.bfloat16,
    )

    assert returned is transformer
    assert transformer._nunchaku_lite_patched
    assert transformer._nunchaku_lite_target == "qwen_image"
    assert hasattr(transformer, "_nunchaku_lite_qwen_image_original_forward")

    block = transformer.transformer_blocks[0]
    assert isinstance(block, NunchakuQwenImageTransformerBlock)
    assert isinstance(block.attn, NunchakuQwenAttention)
    assert isinstance(block.img_mod[1], AWQW4A16Linear)
    assert isinstance(block.txt_mod[1], AWQW4A16Linear)
    assert isinstance(block.img_mlp, NunchakuQwenFeedForward)
    assert isinstance(block.txt_mlp, NunchakuQwenFeedForward)
    assert block.attn.to_qkv.precision == "nvfp4"
    assert block.attn.add_qkv_proj.precision == "nvfp4"
    assert block.img_mlp.net[0].proj.precision == "nvfp4"
    assert block.txt_mlp.net[0].proj.precision == "nvfp4"


def test_qwen_image_checkpoint_keys_match_nunchaku_module_names():
    transformer = make_tiny_qwen_image_transformer()
    patch_tiny_qwen_image(transformer)
    keys = transformer.state_dict().keys()

    assert "transformer_blocks.0.attn.to_qkv.qweight" in keys
    assert "transformer_blocks.0.attn.add_qkv_proj.qweight" in keys
    assert "transformer_blocks.0.attn.to_add_out.qweight" in keys
    assert "transformer_blocks.0.img_mod.1.qweight" in keys
    assert "transformer_blocks.0.txt_mod.1.qweight" in keys
    assert "transformer_blocks.0.img_mlp.net.0.proj.qweight" in keys
    assert "transformer_blocks.0.img_mlp.net.2.qweight" in keys
    assert "transformer_blocks.0.txt_mlp.net.0.proj.qweight" in keys
    assert "transformer_blocks.0.txt_mlp.net.2.qweight" in keys

    block = transformer.transformer_blocks[0]
    assert isinstance(block.attn.to_qkv, SVDQW4A4Linear)
    assert isinstance(block.attn.add_qkv_proj, SVDQW4A4Linear)
