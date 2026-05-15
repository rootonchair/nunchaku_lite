import json
from types import SimpleNamespace

import pytest
import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.sdxl import NunchakuSDXLAttention, SDXLAdapter, convert_sdxl_state_dict
from nunchaku_lite.linear import SVDQW4A4Linear


def make_tiny_sdxl_unet():
    return UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=32,
        attention_head_dim=4,
        norm_num_groups=8,
    )


def first_transformer_block(unet):
    return unet.down_blocks[0].attentions[0].transformer_blocks[0]


def test_sdxl_adapter_matches_diffusers_unet():
    assert SDXLAdapter().matches(make_tiny_sdxl_unet())


def test_convert_sdxl_state_dict_maps_original_nunchaku_keys():
    state = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_qkv.lora_down": torch.empty(1),
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.smooth_orig": torch.empty(1),
        "conv_in.weight": torch.empty(1),
    }

    converted = convert_sdxl_state_dict(state)

    assert "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_qkv.proj_down" in converted
    assert "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.smooth_factor_orig" in converted
    assert "conv_in.weight" in converted


def test_patch_transformer_patches_sdxl_from_synthetic_checkpoint(tmp_path):
    rank = 4
    source = make_tiny_sdxl_unet()
    SDXLAdapter().patch(
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
    checkpoint = tmp_path / "sdxl-lite.safetensors"
    save_file(source.state_dict(), checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    unet = make_tiny_sdxl_unet()
    returned = patch_transformer(unet, checkpoint, target="sdxl", precision="int4", torch_dtype=torch.bfloat16)
    block = first_transformer_block(unet)

    assert returned is unet
    assert unet._nunchaku_lite_patched
    assert unet._nunchaku_lite_target == "sdxl"
    assert isinstance(block.attn1, NunchakuSDXLAttention)
    assert isinstance(block.attn1.to_qkv, SVDQW4A4Linear)
    assert isinstance(block.attn2, NunchakuSDXLAttention)
    assert isinstance(block.attn2.to_q, SVDQW4A4Linear)
    assert isinstance(block.ff.net[0].proj, SVDQW4A4Linear)


def test_sdxl_adapter_rejects_fp4():
    with pytest.raises(ValueError, match="supports only int4"):
        SDXLAdapter().patch(
            make_tiny_sdxl_unet(),
            {},
            {"rank": 4},
            SimpleNamespace(
                precision="nvfp4",
                torch_dtype=torch.bfloat16,
                device=None,
                strict=True,
                adapter_options={},
            ),
        )
