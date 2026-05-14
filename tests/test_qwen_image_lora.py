from types import MethodType, SimpleNamespace

import pytest
import torch

from nunchaku_lite.lora.base import bind_pipeline_lora_methods, unpack_lowrank_weight
from nunchaku_lite.lora.qwen_image import (
    NunchakuQwenImagePipelineLoraMixin,
    convert_qwen_image_lora_to_lite,
)

from test_qwen_image_adapter import make_tiny_qwen_image_transformer, patch_tiny_qwen_image


def make_patched_qwen_image_transformer(rank: int = 4):
    transformer = make_tiny_qwen_image_transformer()
    patch_tiny_qwen_image(transformer, rank=rank)
    with torch.no_grad():
        for index, parameter in enumerate(transformer.parameters()):
            if parameter.is_floating_point():
                parameter.fill_((index % 17) / 100)
        for index, buffer in enumerate(transformer.buffers()):
            if buffer.is_floating_point():
                buffer.fill_((index % 13) / 100)
    return transformer


def test_qwen_image_lora_methods_are_bound_after_patch():
    transformer = make_patched_qwen_image_transformer()

    assert callable(transformer.load_lora)
    assert callable(transformer.load_lora_adapter)
    assert callable(transformer.set_lora_strength)
    assert callable(transformer.reset_lora)
    assert transformer._nunchaku_lite_loras == {}
    assert transformer._nunchaku_lite_active_loras == []
    assert transformer._nunchaku_lite_lora_enabled is True


def test_convert_lightx2v_qwen_qkv_lora_to_lite_fused_projection():
    transformer = make_patched_qwen_image_transformer(rank=4)
    module = transformer.get_submodule("transformer_blocks.0.attn.to_qkv")
    rank = 2
    lora = {}
    for index, branch in enumerate(("to_q", "to_k", "to_v"), start=1):
        base = f"transformer_blocks.0.attn.{branch}"
        lora[f"{base}.lora_down.weight"] = torch.full(
            (rank, module.in_features), float(index), dtype=torch.bfloat16
        )
        lora[f"{base}.lora_up.weight"] = torch.full(
            (module.out_features // 3, rank), float(index), dtype=torch.bfloat16
        )
        lora[f"{base}.alpha"] = torch.tensor(float(rank), dtype=torch.bfloat16)

    converted = convert_qwen_image_lora_to_lite(lora, transformer)

    assert set(converted) == {
        "transformer_blocks.0.attn.to_qkv.proj_down",
        "transformer_blocks.0.attn.to_qkv.proj_up",
    }
    assert converted["transformer_blocks.0.attn.to_qkv.proj_down"].shape[0] == module.in_features
    assert converted["transformer_blocks.0.attn.to_qkv.proj_up"].shape[0] == module.out_features


def test_convert_qwen_add_qkv_and_direct_mlp_lora_to_lite():
    transformer = make_patched_qwen_image_transformer(rank=4)
    add_module = transformer.get_submodule("transformer_blocks.0.attn.add_qkv_proj")
    mlp_name = "transformer_blocks.0.img_mlp.net.2"
    mlp_module = transformer.get_submodule(mlp_name)
    rank = 2
    lora = {}
    for branch in ("add_q_proj", "add_k_proj", "add_v_proj"):
        base = f"transformer.{branch.replace('add_', 'transformer_blocks.0.attn.add_')}"
        lora[f"{base}.lora_A.weight"] = torch.ones(rank, add_module.in_features, dtype=torch.bfloat16)
        lora[f"{base}.lora_B.weight"] = torch.ones(add_module.out_features // 3, rank, dtype=torch.bfloat16)
    lora[f"transformer.{mlp_name}.lora_A.weight"] = torch.ones(rank, mlp_module.in_features, dtype=torch.bfloat16)
    lora[f"transformer.{mlp_name}.lora_B.weight"] = torch.ones(mlp_module.out_features, rank, dtype=torch.bfloat16)

    converted = convert_qwen_image_lora_to_lite(lora, transformer)

    assert "transformer_blocks.0.attn.add_qkv_proj.proj_down" in converted
    assert "transformer_blocks.0.attn.add_qkv_proj.proj_up" in converted
    assert f"{mlp_name}.proj_down" in converted
    assert f"{mlp_name}.proj_up" in converted


def test_qwen_lora_strength_reset_delete_and_unload():
    transformer = make_patched_qwen_image_transformer(rank=4)
    module_name = "transformer_blocks.0.attn.to_out.0"
    module = transformer.get_submodule(module_name)
    base_down = module.proj_down.detach().clone()
    first = {
        f"{module_name}.lora_down.weight": torch.ones(1, module.in_features, dtype=torch.bfloat16),
        f"{module_name}.lora_up.weight": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }
    second = {
        f"{module_name}.lora_down.weight": torch.full((1, module.in_features), 3.0, dtype=torch.bfloat16),
        f"{module_name}.lora_up.weight": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }

    transformer.load_lora(first, strength=1.0, name="first")
    transformer.load_lora(second, strength=0.25, name="second")
    transformer.set_adapters(["second"], weights=[0.5])

    logical_down = unpack_lowrank_weight(module.proj_down.detach(), down=True)
    assert transformer.get_list_adapters() == ["first", "second"]
    assert transformer.get_active_adapters() == ["second"]
    assert torch.allclose(logical_down[4:5], torch.full_like(logical_down[4:5], 1.5))

    transformer.disable_lora()
    assert transformer.get_active_adapters() == []
    assert torch.equal(module.proj_down, base_down)

    transformer.enable_lora()
    transformer.delete_adapters("second")
    assert transformer.get_list_adapters() == ["first"]
    assert transformer.get_active_adapters() == []
    assert torch.equal(module.proj_down, base_down)

    transformer.unload_lora()
    assert transformer.get_list_adapters() == []
    assert torch.equal(module.proj_down, base_down)


def test_qwen_pipeline_lora_mixin_maps_diffusers_api_to_transformer_runtime():
    transformer = make_patched_qwen_image_transformer(rank=4)
    module_name = "transformer_blocks.0.attn.to_out.0"
    module = transformer.get_submodule(module_name)
    lora = {
        f"transformer.{module_name}.lora_down.weight": torch.ones(1, module.in_features, dtype=torch.bfloat16),
        f"transformer.{module_name}.lora_up.weight": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }
    pipeline = SimpleNamespace(transformer=transformer)

    def lora_state_dict(self, state_dict, return_alphas=False, **kwargs):
        if return_alphas and kwargs.get("return_lora_metadata"):
            return state_dict, {}, {}
        if return_alphas:
            return state_dict, {}
        return state_dict

    pipeline.lora_state_dict = MethodType(lora_state_dict, pipeline)
    bind_pipeline_lora_methods(pipeline, NunchakuQwenImagePipelineLoraMixin)

    pipeline.load_lora_weights(lora, adapter_name="lightning")

    assert pipeline.get_list_adapters() == {"transformer": ["lightning"]}
    assert pipeline.get_active_adapters() == ["lightning"]
    pipeline.set_adapters("lightning", adapter_weights=0.25)
    logical_down = unpack_lowrank_weight(module.proj_down.detach(), down=True)
    assert torch.allclose(logical_down[4:5], torch.full_like(logical_down[4:5], 0.25))
    pipeline.unload_lora_weights()
    assert pipeline.get_list_adapters() == {"transformer": []}


def test_qwen_unsupported_lora_target_raises():
    transformer = make_patched_qwen_image_transformer()
    lora = {
        "transformer.transformer_blocks.0.not_a_module.lora_down.weight": torch.ones(2, 64),
        "transformer.transformer_blocks.0.not_a_module.lora_up.weight": torch.ones(64, 2),
    }

    with pytest.raises(ValueError, match="Unsupported Nunchaku LoRA target"):
        transformer.load_lora(lora)
