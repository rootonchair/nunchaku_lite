import json
from types import MethodType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.flux import FluxAdapter
from nunchaku_lite.lora.core.layout import unpack_lowrank_weight
from nunchaku_lite.lora.core.runtime import NunchakuPipelineLoraMixin, bind_pipeline_lora_methods

from test_flux_adapter import make_tiny_flux_transformer


def make_patched_flux_transformer(tmp_path, rank=4):
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
    state = {}
    for index, (key, value) in enumerate(source.state_dict().items()):
        if value.is_floating_point():
            state[key] = torch.full_like(value, (index % 17) / 100)
        else:
            state[key] = torch.zeros_like(value)

    checkpoint = tmp_path / "flux-lite.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})
    transformer = make_tiny_flux_transformer()
    return patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)


def test_flux_lora_methods_are_bound_after_patch(tmp_path):
    transformer = make_patched_flux_transformer(tmp_path)

    assert callable(transformer.load_lora)
    assert callable(transformer.load_lora_adapter)
    assert callable(transformer.set_lora_strength)
    assert callable(transformer.reset_lora)
    assert transformer._nunchaku_lite_loras == {}
    assert transformer._nunchaku_lite_active_loras == []
    assert transformer._nunchaku_lite_lora_enabled is True


def test_load_nunchaku_lora_composes_strength_and_reset(tmp_path):
    transformer = make_patched_flux_transformer(tmp_path, rank=4)
    module_name = "transformer_blocks.0.attn.to_out.0"
    module = transformer.get_submodule(module_name)
    base_down = module.proj_down.detach().clone()
    base_up = module.proj_up.detach().clone()
    lora = {
        f"{module_name}.proj_down": torch.ones(module.in_features, 2, dtype=torch.bfloat16),
        f"{module_name}.proj_up": torch.full((module.out_features, 2), 2.0, dtype=torch.bfloat16),
    }

    name = transformer.load_lora(lora, strength=0.5, name="style")

    assert name == "style"
    logical_down = unpack_lowrank_weight(module.proj_down.detach(), down=True)
    logical_up = unpack_lowrank_weight(module.proj_up.detach(), down=False)
    assert torch.equal(logical_down[:4], base_down.T)
    assert torch.equal(logical_up[:, :4], base_up)
    assert torch.allclose(logical_down[4:6], torch.full_like(logical_down[4:6], 0.5))
    assert torch.allclose(logical_up[:, 4:6], torch.full_like(logical_up[:, 4:6], 2.0))
    assert torch.count_nonzero(logical_down[6:]) == 0
    assert torch.count_nonzero(logical_up[:, 6:]) == 0

    transformer.reset_lora()

    assert torch.equal(module.proj_down, base_down)
    assert torch.equal(module.proj_up, base_up)
    assert transformer._nunchaku_lite_loras == {}
    assert transformer.get_active_adapters() == []


def test_set_named_lora_strength_recomposes_from_baseline(tmp_path):
    transformer = make_patched_flux_transformer(tmp_path, rank=4)
    module_name = "transformer_blocks.0.attn.to_out.0"
    module = transformer.get_submodule(module_name)
    first = {
        f"{module_name}.proj_down": torch.ones(module.in_features, 1, dtype=torch.bfloat16),
        f"{module_name}.proj_up": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }
    second = {
        f"{module_name}.proj_down": torch.full((module.in_features, 1), 3.0, dtype=torch.bfloat16),
        f"{module_name}.proj_up": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }

    transformer.load_lora(first, strength=1.0, name="first")
    transformer.load_lora(second, strength=0.5, name="second")
    transformer.set_lora_strength(1.0, name="second")

    logical_down = unpack_lowrank_weight(module.proj_down.detach(), down=True)
    assert torch.allclose(logical_down[4:5], torch.ones_like(logical_down[4:5]))
    assert torch.allclose(logical_down[5:6], torch.full_like(logical_down[5:6], 3.0))
    with pytest.raises(ValueError, match="Multiple LoRAs"):
        transformer.set_lora_strength(1.0)

    transformer.reset_lora("first")

    assert list(transformer._nunchaku_lite_loras) == ["second"]
    assert transformer.get_active_adapters() == ["second"]
    logical_down = unpack_lowrank_weight(module.proj_down.detach(), down=True)
    assert module.proj_down.shape[1] == 16
    assert torch.allclose(logical_down[4:5], torch.full_like(logical_down[4:5], 3.0))
    assert torch.count_nonzero(logical_down[5:]) == 0


def test_transformer_set_adapters_disable_enable_and_delete(tmp_path):
    transformer = make_patched_flux_transformer(tmp_path, rank=4)
    module_name = "transformer_blocks.0.attn.to_out.0"
    module = transformer.get_submodule(module_name)
    base_down = module.proj_down.detach().clone()
    first = {
        f"{module_name}.proj_down": torch.ones(module.in_features, 1, dtype=torch.bfloat16),
        f"{module_name}.proj_up": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }
    second = {
        f"{module_name}.proj_down": torch.full((module.in_features, 1), 3.0, dtype=torch.bfloat16),
        f"{module_name}.proj_up": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }

    transformer.load_lora(first, strength=1.0, name="first")
    transformer.load_lora(second, strength=1.0, name="second")
    transformer.set_adapters(["second"], weights=[0.5])

    logical_down = unpack_lowrank_weight(module.proj_down.detach(), down=True)
    assert transformer.get_list_adapters() == ["first", "second"]
    assert transformer.get_active_adapters() == ["second"]
    assert torch.allclose(logical_down[4:5], torch.full_like(logical_down[4:5], 1.5))
    assert torch.count_nonzero(logical_down[5:]) == 0

    transformer.disable_lora()
    assert transformer.get_active_adapters() == []
    assert torch.equal(module.proj_down, base_down)

    transformer.enable_lora()
    assert transformer.get_active_adapters() == ["second"]
    logical_down = unpack_lowrank_weight(module.proj_down.detach(), down=True)
    assert torch.allclose(logical_down[4:5], torch.full_like(logical_down[4:5], 1.5))

    transformer.delete_adapters("second")
    assert transformer.get_list_adapters() == ["first"]
    assert transformer.get_active_adapters() == []
    assert torch.equal(module.proj_down, base_down)

    transformer.unload_lora()
    assert transformer.get_list_adapters() == []
    assert torch.equal(module.proj_down, base_down)


def test_load_diffusers_adanorm_lora_sets_awq_side_branch(tmp_path):
    transformer = make_patched_flux_transformer(tmp_path, rank=4)
    module_name = "transformer_blocks.0.norm1.linear"
    module = transformer.get_submodule(module_name)
    lora = {
        f"transformer.{module_name}.lora_A.weight": torch.ones(2, module.in_features, dtype=torch.bfloat16),
        f"transformer.{module_name}.lora_B.weight": torch.ones(module.out_features, 2, dtype=torch.bfloat16),
    }

    transformer.load_lora(lora, strength=0.5, name="norm")

    assert module._nunchaku_lite_lora_down.shape == (module.in_features, 16)
    assert module._nunchaku_lite_lora_up.shape == (module.out_features, 16)
    assert torch.allclose(
        module._nunchaku_lite_lora_down[:, :2],
        torch.full_like(module._nunchaku_lite_lora_down[:, :2], 0.5),
    )
    assert torch.count_nonzero(module._nunchaku_lite_lora_down[:, 2:]) == 0
    transformer.reset_lora()
    assert module._nunchaku_lite_lora_down.shape == (module.in_features, 0)
    assert module._nunchaku_lite_lora_up.shape == (module.out_features, 0)


def test_convert_diffusers_qkv_lora_to_nunchaku_fused_projection(tmp_path):
    transformer = make_patched_flux_transformer(tmp_path, rank=4)
    rank = 2
    lora = {}
    for index, branch in enumerate(("to_q", "to_k", "to_v"), start=1):
        base = f"transformer.transformer_blocks.0.attn.{branch}"
        lora[f"{base}.lora_A.weight"] = torch.full((rank, 32), float(index), dtype=torch.bfloat16)
        lora[f"{base}.lora_B.weight"] = torch.full((32, rank), float(index), dtype=torch.bfloat16)

    converted = transformer._convert_lora_to_nunchaku(lora)

    assert set(converted) == {
        "transformer_blocks.0.attn.to_qkv.proj_down",
        "transformer_blocks.0.attn.to_qkv.proj_up",
    }
    assert converted["transformer_blocks.0.attn.to_qkv.proj_down"].shape == (32, 16)
    assert converted["transformer_blocks.0.attn.to_qkv.proj_up"].shape == (96, 16)


def test_convert_single_proj_out_lora_splits_attn_before_mlp(tmp_path):
    transformer = make_patched_flux_transformer(tmp_path, rank=4)
    attn_module = transformer.get_submodule("single_transformer_blocks.0.attn.to_out")
    mlp_module = transformer.get_submodule("single_transformer_blocks.0.mlp_fc2")
    rank = 2
    attn_down = torch.full((rank, attn_module.in_features), 3.0, dtype=torch.bfloat16)
    mlp_down = torch.full((rank, mlp_module.in_features), 7.0, dtype=torch.bfloat16)
    lora = {
        "transformer.single_transformer_blocks.0.proj_out.lora_A.weight": torch.cat(
            [attn_down, mlp_down], dim=1
        ),
        "transformer.single_transformer_blocks.0.proj_out.lora_B.weight": torch.ones(
            attn_module.out_features, rank, dtype=torch.bfloat16
        ),
    }

    converted = transformer._convert_lora_to_nunchaku(lora)

    converted_attn = unpack_lowrank_weight(
        converted["single_transformer_blocks.0.attn.to_out.proj_down"], down=True
    )
    converted_mlp = unpack_lowrank_weight(converted["single_transformer_blocks.0.mlp_fc2.proj_down"], down=True)
    assert torch.allclose(converted_attn[:rank], attn_down)
    assert torch.allclose(converted_mlp[:rank], mlp_down)


def test_unsupported_flux_lora_target_raises(tmp_path):
    transformer = make_patched_flux_transformer(tmp_path)
    lora = {
        "transformer.transformer_blocks.0.not_a_module.lora_A.weight": torch.ones(2, 32),
        "transformer.transformer_blocks.0.not_a_module.lora_B.weight": torch.ones(192, 2),
    }

    with pytest.raises(ValueError, match="Unsupported Nunchaku LoRA target"):
        transformer.load_lora(lora)


def test_pipeline_lora_mixin_maps_diffusers_api_to_transformer_runtime(tmp_path):
    transformer = make_patched_flux_transformer(tmp_path, rank=4)
    module_name = "transformer_blocks.0.attn.to_out.0"
    module = transformer.get_submodule(module_name)
    base_down = module.proj_down.detach().clone()
    lora = {
        f"transformer.{module_name}.lora_A.weight": torch.ones(1, module.in_features, dtype=torch.bfloat16),
        f"transformer.{module_name}.lora_B.weight": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }

    pipeline = SimpleNamespace(transformer=transformer)

    def lora_state_dict(self, state_dict, return_alphas=False, **kwargs):
        metadata = {"format": "test"}
        if return_alphas and kwargs.get("return_lora_metadata"):
            return state_dict, {}, metadata
        if return_alphas:
            return state_dict, {}
        return state_dict

    pipeline.lora_state_dict = MethodType(lora_state_dict, pipeline)
    bind_pipeline_lora_methods(pipeline, NunchakuPipelineLoraMixin)

    pipeline.load_lora_weights(lora, adapter_name="style")

    assert pipeline.get_list_adapters() == {"transformer": ["style"]}
    assert pipeline.get_active_adapters() == ["style"]
    assert module.proj_down.shape[1] == 32

    pipeline.set_adapters("style", adapter_weights=0.5)
    logical_down = unpack_lowrank_weight(module.proj_down.detach(), down=True)
    assert torch.allclose(logical_down[4:5], torch.full_like(logical_down[4:5], 0.5))

    pipeline.disable_lora()
    assert pipeline.get_active_adapters() == []
    assert torch.equal(module.proj_down, base_down)

    pipeline.enable_lora()
    assert pipeline.get_active_adapters() == ["style"]

    pipeline.delete_adapters("style")
    assert pipeline.get_list_adapters() == {"transformer": []}
    assert torch.equal(module.proj_down, base_down)

    pipeline.load_lora_weights(lora, adapter_name="style")
    pipeline.unload_lora_weights()
    assert pipeline.get_list_adapters() == {"transformer": []}
    assert torch.equal(module.proj_down, base_down)


def test_pipeline_lora_mixin_rejects_unsupported_apis_and_text_encoder_lora(tmp_path):
    transformer = make_patched_flux_transformer(tmp_path, rank=4)
    pipeline = SimpleNamespace(transformer=transformer)

    def lora_state_dict(self, state_dict, return_alphas=False, **kwargs):
        metadata = {}
        if return_alphas and kwargs.get("return_lora_metadata"):
            return state_dict, {}, metadata
        if return_alphas:
            return state_dict, {}
        return state_dict

    pipeline.lora_state_dict = MethodType(lora_state_dict, pipeline)
    bind_pipeline_lora_methods(pipeline, NunchakuPipelineLoraMixin)
    text_lora = {
        "text_encoder.encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 4),
        "text_encoder.encoder.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 1),
    }

    with pytest.raises(NotImplementedError, match="text encoder LoRA keys"):
        pipeline.load_lora_weights(text_lora, adapter_name="text")
    with pytest.raises(NotImplementedError, match="does not support fusing"):
        pipeline.fuse_lora()
    with pytest.raises(NotImplementedError, match="does not support fusing"):
        pipeline.unfuse_lora()


def test_pipeline_lora_mixin_requires_bound_transformer_lora_runtime():
    pipeline = SimpleNamespace(transformer=SimpleNamespace())
    bind_pipeline_lora_methods(pipeline, NunchakuPipelineLoraMixin)

    with pytest.raises(RuntimeError, match="not bound to the nunchaku_lite transformer LoRA runtime"):
        pipeline.get_list_adapters()
