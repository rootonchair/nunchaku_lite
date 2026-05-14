import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_z_image import FeedForward as DiffusersZImageFeedForward
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.common import NunchakuAttention, patch_attention_module, patch_modules_recursively
from nunchaku_lite.adapters.z_image import ZImageAdapter
from nunchaku_lite.models.linear import SVDQW4A4Linear


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
    assert isinstance(transformer.layers[0].attention, NunchakuAttention)
    assert transformer.layers[0].attention.__class__.__name__ == "NunchakuAttention"
    assert hasattr(transformer.layers[0].attention, "to_qkv")
    assert hasattr(transformer.layers[0].attention, "norm_q")
    assert hasattr(transformer.layers[0].attention, "norm_k")
    assert isinstance(transformer.layers[0].feed_forward, FeedForward)
    assert isinstance(transformer.layers[0].feed_forward.net[0].proj, SVDQW4A4Linear)
    assert isinstance(transformer.layers[0].feed_forward.net[2], SVDQW4A4Linear)


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


def test_patch_transformer_skip_refiners_converts_feed_forward_without_quantizing_refiners(tmp_path):
    rank = 4
    source = make_tiny_z_image_transformer()
    ZImageAdapter().patch(
        source,
        {},
        {"rank": rank, "skip_refiners": True},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )
    state = source.state_dict()
    checkpoint = tmp_path / "z-image-lite-skip-refiners.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank, "skip_refiners": True})})

    transformer = make_tiny_z_image_transformer()
    patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)

    assert isinstance(transformer.layers[0].attention, NunchakuAttention)
    assert isinstance(transformer.layers[0].feed_forward.net[0].proj, SVDQW4A4Linear)
    assert not isinstance(transformer.noise_refiner[0].attention, NunchakuAttention)
    assert not isinstance(transformer.context_refiner[0].attention, NunchakuAttention)
    assert isinstance(transformer.noise_refiner[0].feed_forward, FeedForward)
    assert isinstance(transformer.context_refiner[0].feed_forward, FeedForward)
    assert isinstance(transformer.noise_refiner[0].feed_forward.net[0].proj, nn.Linear)
    assert isinstance(transformer.context_refiner[0].feed_forward.net[0].proj, nn.Linear)


def test_patch_attention_module_rejects_non_diffusers_attention():
    class AttentionLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(4, 4)
            self.to_k = nn.Linear(4, 4)
            self.to_v = nn.Linear(4, 4)
            self.to_out = nn.ModuleList([nn.Linear(4, 4)])

    with pytest.raises(TypeError, match="only supports the exact diffusers Attention class"):
        patch_attention_module(AttentionLike(), object())


def test_patch_modules_recursively_replaces_exact_attention_and_filtered_linears():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention(query_dim=8, heads=2, dim_head=4, bias=False)
            self.feed_forward = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))
            self.modulation = nn.Linear(8, 8)

    module = Wrapper()
    report = patch_modules_recursively(
        module,
        attention_processor_factory=lambda _path, _attention: object(),
        linear_filter=lambda path, _linear: path.startswith("feed_forward."),
    )

    assert report.attention_modules == 1
    assert report.linear_modules == 2
    assert isinstance(module.attention, NunchakuAttention)
    assert isinstance(module.feed_forward[0], SVDQW4A4Linear)
    assert isinstance(module.feed_forward[2], SVDQW4A4Linear)
    assert isinstance(module.modulation, nn.Linear)


def test_patch_modules_recursively_applies_module_converters_before_descending():
    def convert_z_image_ff(module):
        return FeedForward(
            dim=module.w1.in_features,
            dim_out=module.w2.out_features,
            dropout=0.0,
            activation_fn="swiglu",
            inner_dim=module.w2.in_features,
            bias=False,
        )

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.feed_forward = DiffusersZImageFeedForward(dim=8, hidden_dim=16)

    module = Wrapper()
    report = patch_modules_recursively(
        module,
        module_converters={DiffusersZImageFeedForward: convert_z_image_ff},
        linear_filter=lambda path, _linear: path.startswith("feed_forward."),
    )

    assert report.converted_modules == 1
    assert report.linear_modules == 2
    assert isinstance(module.feed_forward, FeedForward)
    assert isinstance(module.feed_forward.net[0].proj, SVDQW4A4Linear)
    assert isinstance(module.feed_forward.net[2], SVDQW4A4Linear)


def test_patch_modules_recursively_raises_for_unhandled_attention_subclasses():
    class CustomAttention(Attention):
        pass

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = CustomAttention(query_dim=8, heads=2, dim_head=4, bias=False)

    module = Wrapper()

    with pytest.raises(TypeError, match="unsupported Diffusers Attention subclass without a converter"):
        patch_modules_recursively(module, attention_processor_factory=lambda _path, _attention: object())
    assert isinstance(module.attention, CustomAttention)


def test_patch_modules_recursively_replaces_attention_subclasses_with_converter():
    class CustomAttention(Attention):
        pass

    class NunchakuCustomAttention(nn.Module):
        pass

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = CustomAttention(query_dim=8, heads=2, dim_head=4, bias=False)

    module = Wrapper()
    report = patch_modules_recursively(
        module,
        module_converters={CustomAttention: lambda _attention: NunchakuCustomAttention()},
    )

    assert report.converted_modules == 1
    assert isinstance(module.attention, NunchakuCustomAttention)
