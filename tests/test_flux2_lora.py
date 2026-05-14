import json
from types import MethodType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.flux2 import Flux2Adapter
from nunchaku_lite.lora.base import NunchakuPipelineLoraMixin, bind_pipeline_lora_methods, unpack_lowrank_weight
from nunchaku_lite.lora.flux2 import normalize_flux2_comfyui_lora_keys

from test_flux2_adapter import make_tiny_flux2_transformer


def make_patched_flux2_transformer(tmp_path, rank: int = 4):
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
    state = {}
    for index, (key, value) in enumerate(source.state_dict().items()):
        if value.is_floating_point():
            state[key] = torch.full_like(value, (index % 17) / 100)
        else:
            state[key] = torch.zeros_like(value)
    checkpoint = tmp_path / "flux2-lite.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})
    transformer = make_tiny_flux2_transformer()
    return patch_transformer(transformer, checkpoint, target="flux2", precision="int4", torch_dtype=torch.bfloat16)


def test_flux2_lora_methods_are_bound_after_patch(tmp_path):
    transformer = make_patched_flux2_transformer(tmp_path)

    assert callable(transformer.load_lora)
    assert callable(transformer.load_lora_adapter)
    assert callable(transformer.set_lora_strength)
    assert callable(transformer.reset_lora)
    assert transformer._nunchaku_lite_loras == {}
    assert transformer._nunchaku_lite_active_loras == []
    assert transformer._nunchaku_lite_lora_enabled is True


def test_normalize_flux2_comfyui_lora_keys_rewrites_double_and_single_blocks():
    state_dict = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": torch.ones(1, 4),
        "diffusion_model.double_blocks.0.txt_mlp.2.lora_B.weight": torch.ones(4, 1),
        "diffusion_model.single_blocks.1.linear1.lora_A.weight": torch.ones(1, 4),
        "diffusion_model.single_blocks.1.linear2.alpha": torch.tensor(1.0),
    }

    normalized = normalize_flux2_comfyui_lora_keys(state_dict)

    assert set(normalized) == {
        "transformer_blocks.0.attn.to_qkv.lora_A.weight",
        "transformer_blocks.0.ff_context.linear_out.lora_B.weight",
        "single_transformer_blocks.1.attn.linear1.lora_A.weight",
        "single_transformer_blocks.1.attn.linear2.alpha",
    }


def test_convert_flux2_comfyui_double_block_lora_to_nunchaku(tmp_path):
    transformer = make_patched_flux2_transformer(tmp_path, rank=4)
    qkv_module = transformer.get_submodule("transformer_blocks.0.attn.to_qkv")
    mlp_module = transformer.get_submodule("transformer_blocks.0.ff_context.linear_out")
    rank = 2
    lora = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": torch.ones(
            rank, qkv_module.in_features, dtype=torch.bfloat16
        ),
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": torch.ones(
            qkv_module.out_features, rank, dtype=torch.bfloat16
        ),
        "diffusion_model.double_blocks.0.txt_mlp.2.lora_A.weight": torch.full(
            (rank, mlp_module.in_features), 2.0, dtype=torch.bfloat16
        ),
        "diffusion_model.double_blocks.0.txt_mlp.2.lora_B.weight": torch.full(
            (mlp_module.out_features, rank), 3.0, dtype=torch.bfloat16
        ),
    }

    converted = transformer._convert_lora_to_nunchaku(lora)

    assert set(converted) == {
        "transformer_blocks.0.attn.to_qkv.proj_down",
        "transformer_blocks.0.attn.to_qkv.proj_up",
        "transformer_blocks.0.ff_context.linear_out.proj_down",
        "transformer_blocks.0.ff_context.linear_out.proj_up",
    }
    assert converted["transformer_blocks.0.attn.to_qkv.proj_down"].shape == (qkv_module.in_features, 16)
    assert converted["transformer_blocks.0.attn.to_qkv.proj_up"].shape == (qkv_module.out_features, 16)
    mlp_down = unpack_lowrank_weight(converted["transformer_blocks.0.ff_context.linear_out.proj_down"], down=True)
    assert torch.allclose(mlp_down[:rank], torch.full_like(mlp_down[:rank], 2.0))


def test_convert_flux2_diffusers_double_block_qkv_lora_to_nunchaku(tmp_path):
    transformer = make_patched_flux2_transformer(tmp_path, rank=4)
    image_module = transformer.get_submodule("transformer_blocks.0.attn.to_qkv")
    text_module = transformer.get_submodule("transformer_blocks.0.attn.to_added_qkv")
    rank = 2
    lora = {}
    for value, branch in enumerate(("to_q", "to_k", "to_v"), start=1):
        lora[f"transformer.transformer_blocks.0.attn.{branch}.lora_A.weight"] = torch.full(
            (rank, image_module.in_features), value, dtype=torch.bfloat16
        )
        lora[f"transformer.transformer_blocks.0.attn.{branch}.lora_B.weight"] = torch.full(
            (image_module.out_features // 3, rank), value + 3, dtype=torch.bfloat16
        )
    for value, branch in enumerate(("add_q_proj", "add_k_proj", "add_v_proj"), start=7):
        lora[f"transformer.transformer_blocks.0.attn.{branch}.lora_A.weight"] = torch.full(
            (rank, text_module.in_features), value, dtype=torch.bfloat16
        )
        lora[f"transformer.transformer_blocks.0.attn.{branch}.lora_B.weight"] = torch.full(
            (text_module.out_features // 3, rank), value + 3, dtype=torch.bfloat16
        )

    converted = transformer._convert_lora_to_nunchaku(lora)

    assert set(converted) == {
        "transformer_blocks.0.attn.to_qkv.proj_down",
        "transformer_blocks.0.attn.to_qkv.proj_up",
        "transformer_blocks.0.attn.to_added_qkv.proj_down",
        "transformer_blocks.0.attn.to_added_qkv.proj_up",
    }
    image_up = unpack_lowrank_weight(converted["transformer_blocks.0.attn.to_qkv.proj_up"], down=False)
    text_up = unpack_lowrank_weight(converted["transformer_blocks.0.attn.to_added_qkv.proj_up"], down=False)
    image_q_up = image_up[: image_module.out_features // 3, :rank]
    text_q_up = text_up[: text_module.out_features // 3, :rank]
    assert torch.allclose(image_q_up, torch.full_like(image_q_up, 4.0))
    assert torch.allclose(text_q_up, torch.full_like(text_q_up, 10.0))


def test_convert_flux2_comfyui_single_linear1_lora_splits_qkv_and_mlp(tmp_path):
    transformer = make_patched_flux2_transformer(tmp_path, rank=4)
    qkv_module = transformer.get_submodule("single_transformer_blocks.0.attn.qkv_proj")
    mlp_module = transformer.get_submodule("single_transformer_blocks.0.attn.mlp_fc1")
    rank = 2
    qkv_up = torch.full((qkv_module.out_features, rank), 3.0, dtype=torch.bfloat16)
    mlp_up = torch.full((mlp_module.out_features, rank), 7.0, dtype=torch.bfloat16)
    lora = {
        "diffusion_model.single_blocks.0.linear1.lora_A.weight": torch.ones(
            rank, qkv_module.in_features, dtype=torch.bfloat16
        ),
        "diffusion_model.single_blocks.0.linear1.lora_B.weight": torch.cat([qkv_up, mlp_up], dim=0),
    }

    converted = transformer._convert_lora_to_nunchaku(lora)

    assert set(converted) == {
        "single_transformer_blocks.0.attn.qkv_proj.proj_down",
        "single_transformer_blocks.0.attn.qkv_proj.proj_up",
        "single_transformer_blocks.0.attn.mlp_fc1.proj_down",
        "single_transformer_blocks.0.attn.mlp_fc1.proj_up",
    }
    qkv_logical_up = unpack_lowrank_weight(converted["single_transformer_blocks.0.attn.qkv_proj.proj_up"], down=False)
    mlp_logical_up = unpack_lowrank_weight(converted["single_transformer_blocks.0.attn.mlp_fc1.proj_up"], down=False)
    assert torch.allclose(qkv_logical_up[:, :rank], qkv_up)
    assert torch.allclose(mlp_logical_up[:, :rank], mlp_up)


def test_convert_flux2_diffusers_single_to_qkv_mlp_proj_lora_splits_qkv_and_mlp(tmp_path):
    transformer = make_patched_flux2_transformer(tmp_path, rank=4)
    qkv_module = transformer.get_submodule("single_transformer_blocks.0.attn.qkv_proj")
    mlp_module = transformer.get_submodule("single_transformer_blocks.0.attn.mlp_fc1")
    rank = 2
    qkv_up = torch.full((qkv_module.out_features, rank), 3.0, dtype=torch.bfloat16)
    mlp_up = torch.full((mlp_module.out_features, rank), 7.0, dtype=torch.bfloat16)
    lora = {
        "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight": torch.ones(
            rank, qkv_module.in_features, dtype=torch.bfloat16
        ),
        "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.weight": torch.cat(
            [qkv_up, mlp_up], dim=0
        ),
    }

    converted = transformer._convert_lora_to_nunchaku(lora)

    qkv_logical_up = unpack_lowrank_weight(converted["single_transformer_blocks.0.attn.qkv_proj.proj_up"], down=False)
    mlp_logical_up = unpack_lowrank_weight(converted["single_transformer_blocks.0.attn.mlp_fc1.proj_up"], down=False)
    assert torch.allclose(qkv_logical_up[:, :rank], qkv_up)
    assert torch.allclose(mlp_logical_up[:, :rank], mlp_up)


def test_convert_flux2_comfyui_single_linear2_lora_splits_out_and_mlp(tmp_path):
    transformer = make_patched_flux2_transformer(tmp_path, rank=4)
    out_module = transformer.get_submodule("single_transformer_blocks.0.attn.out_proj")
    mlp_module = transformer.get_submodule("single_transformer_blocks.0.attn.mlp_fc2")
    rank = 2
    out_down = torch.full((rank, out_module.in_features), 3.0, dtype=torch.bfloat16)
    mlp_down = torch.full((rank, mlp_module.in_features), 7.0, dtype=torch.bfloat16)
    lora = {
        "diffusion_model.single_blocks.0.linear2.lora_A.weight": torch.cat([out_down, mlp_down], dim=1),
        "diffusion_model.single_blocks.0.linear2.lora_B.weight": torch.ones(
            out_module.out_features, rank, dtype=torch.bfloat16
        ),
    }

    converted = transformer._convert_lora_to_nunchaku(lora)

    assert set(converted) == {
        "single_transformer_blocks.0.attn.out_proj.proj_down",
        "single_transformer_blocks.0.attn.out_proj.proj_up",
        "single_transformer_blocks.0.attn.mlp_fc2.proj_down",
        "single_transformer_blocks.0.attn.mlp_fc2.proj_up",
    }
    out_logical_down = unpack_lowrank_weight(
        converted["single_transformer_blocks.0.attn.out_proj.proj_down"], down=True
    )
    mlp_logical_down = unpack_lowrank_weight(
        converted["single_transformer_blocks.0.attn.mlp_fc2.proj_down"], down=True
    )
    assert torch.allclose(out_logical_down[:rank], out_down)
    assert torch.allclose(mlp_logical_down[:rank], mlp_down)


def test_convert_flux2_diffusers_single_to_out_lora_splits_out_and_mlp(tmp_path):
    transformer = make_patched_flux2_transformer(tmp_path, rank=4)
    out_module = transformer.get_submodule("single_transformer_blocks.0.attn.out_proj")
    mlp_module = transformer.get_submodule("single_transformer_blocks.0.attn.mlp_fc2")
    rank = 2
    out_down = torch.full((rank, out_module.in_features), 3.0, dtype=torch.bfloat16)
    mlp_down = torch.full((rank, mlp_module.in_features), 7.0, dtype=torch.bfloat16)
    lora = {
        "transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight": torch.cat(
            [out_down, mlp_down], dim=1
        ),
        "transformer.single_transformer_blocks.0.attn.to_out.lora_B.weight": torch.ones(
            out_module.out_features, rank, dtype=torch.bfloat16
        ),
    }

    converted = transformer._convert_lora_to_nunchaku(lora)

    out_logical_down = unpack_lowrank_weight(
        converted["single_transformer_blocks.0.attn.out_proj.proj_down"], down=True
    )
    mlp_logical_down = unpack_lowrank_weight(
        converted["single_transformer_blocks.0.attn.mlp_fc2.proj_down"], down=True
    )
    assert torch.allclose(out_logical_down[:rank], out_down)
    assert torch.allclose(mlp_logical_down[:rank], mlp_down)


def test_flux2_lora_strength_reset_delete_and_unload(tmp_path):
    transformer = make_patched_flux2_transformer(tmp_path, rank=4)
    module_name = "single_transformer_blocks.0.attn.out_proj"
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
    transformer.load_lora(second, strength=0.25, name="second")
    transformer.set_adapters("second", weights=0.5)

    logical_down = unpack_lowrank_weight(module.proj_down.detach(), down=True)
    assert transformer.get_list_adapters() == ["first", "second"]
    assert transformer.get_active_adapters() == ["second"]
    assert torch.allclose(logical_down[4:5], torch.full_like(logical_down[4:5], 1.5))

    transformer.disable_lora()
    assert transformer.get_active_adapters() == []
    assert torch.equal(module.proj_down, base_down)

    transformer.enable_lora()
    assert transformer.get_active_adapters() == ["second"]
    transformer.delete_adapters("second")
    assert transformer.get_list_adapters() == ["first"]
    transformer.unload_lora()
    assert transformer.get_list_adapters() == []
    assert torch.equal(module.proj_down, base_down)


def test_flux2_pipeline_lora_mixin_maps_diffusers_api_to_transformer_runtime(tmp_path):
    transformer = make_patched_flux2_transformer(tmp_path, rank=4)
    module_name = "transformer_blocks.0.attn.to_out.0"
    module = transformer.get_submodule(module_name)
    base_down = module.proj_down.detach().clone()
    lora = {
        f"transformer.{module_name}.lora_A.weight": torch.ones(1, module.in_features, dtype=torch.bfloat16),
        f"transformer.{module_name}.lora_B.weight": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }

    pipeline = SimpleNamespace(transformer=transformer)

    def lora_state_dict(self, state_dict, return_alphas=False, **kwargs):
        if return_alphas and kwargs.get("return_lora_metadata"):
            return state_dict, {}, {}
        if return_alphas:
            return state_dict, {}
        return state_dict

    pipeline.lora_state_dict = MethodType(lora_state_dict, pipeline)
    bind_pipeline_lora_methods(pipeline, NunchakuPipelineLoraMixin)

    pipeline.load_lora_weights(lora, adapter_name="style")

    assert pipeline.get_list_adapters() == {"transformer": ["style"]}
    assert pipeline.get_active_adapters() == ["style"]
    pipeline.set_adapters("style", adapter_weights=0.5)
    logical_down = unpack_lowrank_weight(module.proj_down.detach(), down=True)
    assert torch.allclose(logical_down[4:5], torch.full_like(logical_down[4:5], 0.5))
    pipeline.unload_lora_weights()
    assert pipeline.get_list_adapters() == {"transformer": []}
    assert torch.equal(module.proj_down, base_down)


def test_flux2_lora_rejects_unknown_targets(tmp_path):
    transformer = make_patched_flux2_transformer(tmp_path, rank=4)
    lora = {
        "diffusion_model.double_blocks.0.img_attn.not_a_module.lora_A.weight": torch.ones(2, 32),
        "diffusion_model.double_blocks.0.img_attn.not_a_module.lora_B.weight": torch.ones(32, 2),
    }

    with pytest.raises(ValueError, match="Unsupported Nunchaku LoRA target"):
        transformer.load_lora(lora)
