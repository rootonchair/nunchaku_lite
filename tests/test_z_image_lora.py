import os
from copy import deepcopy
from types import MethodType, SimpleNamespace

import pytest
import torch
from diffusers.loaders.lora_pipeline import ZImageLoraLoaderMixin
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

from nunchaku_lite.adapters.z_image import ZImageAdapter
from nunchaku_lite.lora.base import (
    DenseRuntimeLoraLinear,
    NunchakuPipelineLoraMixin,
    bind_pipeline_lora_methods,
    unpack_lowrank_weight,
)
from nunchaku_lite.lora.z_image import normalize_z_image_diffusers_lora_state_dict
from nunchaku_lite.models.linear import SVDQW4A4Linear

from test_z_image_adapter import make_tiny_z_image_transformer

Z_IMAGE_BLOCK_GROUPS = ("layers", "noise_refiner", "context_refiner")
Z_IMAGE_LINEAR_TARGETS = (
    "attention.to_q",
    "attention.to_k",
    "attention.to_v",
    "attention.to_out.0",
    "feed_forward.w1",
    "feed_forward.w3",
    "feed_forward.w2",
    "adaLN_modulation.0",
)


def make_patched_z_image_transformer(rank: int = 4):
    transformer = make_tiny_z_image_transformer()
    ZImageAdapter().patch(
        transformer,
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
    with torch.no_grad():
        for index, parameter in enumerate(transformer.parameters()):
            if parameter.is_floating_point():
                parameter.fill_((index % 17) / 100)
        for index, buffer in enumerate(transformer.buffers()):
            if buffer.is_floating_point():
                buffer.fill_((index % 13) / 100)
    return transformer


def make_native_kernel_z_image_transformer():
    return ZImageTransformer2DModel(
        in_channels=4,
        dim=384,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=6,
        n_kv_heads=6,
        cap_feat_dim=8,
        axes_dims=[16, 24, 24],
        axes_lens=[16, 16, 16],
    )


def make_patched_native_kernel_z_image_transformer(rank: int = 16):
    transformer = make_native_kernel_z_image_transformer()
    ZImageAdapter().patch(
        transformer,
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
    return transformer


def z_image_lora_target_modules(transformer):
    targets = {}
    for group_name in Z_IMAGE_BLOCK_GROUPS:
        for block_index, _block in enumerate(getattr(transformer, group_name)):
            for local_name in Z_IMAGE_LINEAR_TARGETS:
                module_name = f"{group_name}.{block_index}.{local_name}"
                try:
                    module = transformer.get_submodule(module_name)
                except AttributeError:
                    continue
                if hasattr(module, "in_features") and hasattr(module, "out_features"):
                    targets[module_name] = module
    return targets


def make_z_image_lora_for_all_targets(transformer, rank: int = 2, scale: float = 0.15):
    lora = {}
    for index, (name, module) in enumerate(z_image_lora_target_modules(transformer).items(), start=1):
        torch.manual_seed(index)
        lora[f"transformer.{name}.lora_A.weight"] = (torch.randn(rank, module.in_features) * scale).to(torch.bfloat16)
        torch.manual_seed(index + 1000)
        lora[f"transformer.{name}.lora_B.weight"] = (torch.randn(module.out_features, rank) * scale).to(torch.bfloat16)
    return lora


def normalize_native_lora_test_base(transformer):
    with torch.no_grad():
        for module in transformer.modules():
            if isinstance(module, SVDQW4A4Linear):
                module.qweight.zero_()
                module.wscales.fill_(1)
                module.smooth_factor.fill_(1)
                module.smooth_factor_orig.fill_(1)
                module.proj_down.zero_()
                module.proj_up.zero_()
                if module.bias is not None:
                    module.bias.zero_()
                if module.wcscales is not None:
                    module.wcscales.fill_(1)
            elif isinstance(module, DenseRuntimeLoraLinear):
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()


def diffusers_lora_delta(diffusers_transformer, module_name: str) -> torch.Tensor:
    module = diffusers_transformer.get_submodule(module_name)
    down = module.lora_A["oracle"].weight.detach().float()
    up = module.lora_B["oracle"].weight.detach().float() * module.scaling["oracle"]
    return up @ down


def expected_z_image_runtime_delta(diffusers_transformer, runtime_name: str) -> torch.Tensor:
    if runtime_name.endswith(".attention.to_qkv"):
        prefix = runtime_name[: -len(".attention.to_qkv")]
        return torch.cat(
            [
                diffusers_lora_delta(diffusers_transformer, f"{prefix}.attention.to_q"),
                diffusers_lora_delta(diffusers_transformer, f"{prefix}.attention.to_k"),
                diffusers_lora_delta(diffusers_transformer, f"{prefix}.attention.to_v"),
            ],
            dim=0,
        )
    if runtime_name.endswith(".feed_forward.net.0.proj"):
        prefix = runtime_name[: -len(".feed_forward.net.0.proj")]
        return torch.cat(
            [
                diffusers_lora_delta(diffusers_transformer, f"{prefix}.feed_forward.w3"),
                diffusers_lora_delta(diffusers_transformer, f"{prefix}.feed_forward.w1"),
            ],
            dim=0,
        )
    if runtime_name.endswith(".feed_forward.net.2"):
        prefix = runtime_name[: -len(".feed_forward.net.2")]
        return diffusers_lora_delta(diffusers_transformer, f"{prefix}.feed_forward.w2")
    return diffusers_lora_delta(diffusers_transformer, runtime_name)


def converted_z_image_runtime_delta(transformer, converted: dict[str, torch.Tensor], runtime_name: str) -> torch.Tensor:
    module = transformer.get_submodule(runtime_name)
    down = converted[f"{runtime_name}.proj_down"]
    up = converted[f"{runtime_name}.proj_up"]
    if isinstance(module, SVDQW4A4Linear):
        down = unpack_lowrank_weight(down, down=True)[:, : module.in_features].float()
        up = unpack_lowrank_weight(up, down=False)[: module.out_features].float()
        return up @ down
    return up.float() @ down.T.float()


def test_z_image_lora_methods_are_bound_after_patch():
    transformer = make_patched_z_image_transformer()

    assert callable(transformer.load_lora)
    assert callable(transformer.load_lora_adapter)
    assert callable(transformer.set_lora_strength)
    assert callable(transformer.reset_lora)
    assert transformer._nunchaku_lite_loras == {}
    assert transformer._nunchaku_lite_active_loras == []
    assert transformer._nunchaku_lite_lora_enabled is True
    assert isinstance(transformer.layers[0].adaLN_modulation[0], DenseRuntimeLoraLinear)


def test_normalize_z_image_lora_keys_rewrites_prefixes_and_lora_names():
    state_dict = {
        "diffusion_model.layers.0.attention.to_q.lora_down.weight": torch.ones(1, 4),
        "diffusion_model.layers.0.attention.to_q.lora_up.weight": torch.ones(4, 1),
        "transformer.layers.0.feed_forward.w2.lora_A.weight": torch.ones(1, 4),
        "transformer.layers.0.feed_forward.w2.alpha": torch.tensor(1.0),
    }

    normalized = normalize_z_image_diffusers_lora_state_dict(state_dict)

    assert set(normalized) == {
        "layers.0.attention.to_q.lora_A.weight",
        "layers.0.attention.to_q.lora_B.weight",
        "layers.0.feed_forward.w2.lora_A.weight",
    }


def test_convert_z_image_qkv_lora_to_nunchaku_fused_projection():
    transformer = make_patched_z_image_transformer(rank=4)
    module = transformer.get_submodule("layers.0.attention.to_qkv")
    rank = 2
    lora = {}
    for value, branch in enumerate(("to_q", "to_k", "to_v"), start=1):
        lora[f"diffusion_model.layers.0.attention.{branch}.lora_A.weight"] = torch.ones(
            rank, module.in_features, dtype=torch.bfloat16
        )
        lora[f"diffusion_model.layers.0.attention.{branch}.lora_B.weight"] = torch.full(
            (module.out_features // 3, rank), value, dtype=torch.bfloat16
        )

    converted = transformer._convert_lora_to_nunchaku(lora)

    assert set(converted) == {
        "layers.0.attention.to_qkv.proj_down",
        "layers.0.attention.to_qkv.proj_up",
    }
    logical_up = unpack_lowrank_weight(converted["layers.0.attention.to_qkv.proj_up"], down=False)
    q_up = logical_up[: module.out_features // 3, :rank]
    assert torch.allclose(q_up, torch.ones_like(q_up))


def test_convert_z_image_swiglu_lora_to_nunchaku_fused_projection():
    transformer = make_patched_z_image_transformer(rank=4)
    module = transformer.get_submodule("layers.0.feed_forward.net.0.proj")
    rank = 2
    w3_up = torch.full((module.out_features // 2, rank), 3.0, dtype=torch.bfloat16)
    w1_up = torch.full((module.out_features // 2, rank), 7.0, dtype=torch.bfloat16)
    lora = {
        "diffusion_model.layers.0.feed_forward.w3.lora_A.weight": torch.ones(
            rank, module.in_features, dtype=torch.bfloat16
        ),
        "diffusion_model.layers.0.feed_forward.w3.lora_B.weight": w3_up,
        "diffusion_model.layers.0.feed_forward.w1.lora_A.weight": torch.ones(
            rank, module.in_features, dtype=torch.bfloat16
        ),
        "diffusion_model.layers.0.feed_forward.w1.lora_B.weight": w1_up,
    }

    converted = transformer._convert_lora_to_nunchaku(lora)

    assert set(converted) == {
        "layers.0.feed_forward.net.0.proj.proj_down",
        "layers.0.feed_forward.net.0.proj.proj_up",
    }
    logical_up = unpack_lowrank_weight(converted["layers.0.feed_forward.net.0.proj.proj_up"], down=False)
    half = module.out_features // 2
    assert torch.allclose(logical_up[:half, :rank], w3_up)
    assert torch.allclose(logical_up[half : module.out_features, :rank], w1_up)


def test_convert_z_image_direct_and_dense_lora_targets():
    transformer = make_patched_z_image_transformer(rank=4)
    out_module = transformer.get_submodule("layers.0.attention.to_out.0")
    ff_module = transformer.get_submodule("layers.0.feed_forward.net.2")
    adaln_module = transformer.get_submodule("layers.0.adaLN_modulation.0")
    rank = 2
    lora = {
        "diffusion_model.layers.0.attention.to_out.0.lora_A.weight": torch.ones(
            rank, out_module.in_features, dtype=torch.bfloat16
        ),
        "diffusion_model.layers.0.attention.to_out.0.lora_B.weight": torch.ones(
            out_module.out_features, rank, dtype=torch.bfloat16
        ),
        "diffusion_model.layers.0.feed_forward.w2.lora_A.weight": torch.ones(
            rank, ff_module.in_features, dtype=torch.bfloat16
        ),
        "diffusion_model.layers.0.feed_forward.w2.lora_B.weight": torch.ones(
            ff_module.out_features, rank, dtype=torch.bfloat16
        ),
        "diffusion_model.layers.0.adaLN_modulation.0.lora_A.weight": torch.ones(
            rank, adaln_module.in_features, dtype=torch.bfloat16
        ),
        "diffusion_model.layers.0.adaLN_modulation.0.lora_B.weight": torch.ones(
            adaln_module.out_features, rank, dtype=torch.bfloat16
        ),
    }

    converted = transformer._convert_lora_to_nunchaku(lora)

    assert set(converted) == {
        "layers.0.attention.to_out.0.proj_down",
        "layers.0.attention.to_out.0.proj_up",
        "layers.0.feed_forward.net.2.proj_down",
        "layers.0.feed_forward.net.2.proj_up",
        "layers.0.adaLN_modulation.0.proj_down",
        "layers.0.adaLN_modulation.0.proj_up",
    }
    assert converted["layers.0.adaLN_modulation.0.proj_down"].shape == (adaln_module.in_features, rank)
    assert converted["layers.0.adaLN_modulation.0.proj_up"].shape == (adaln_module.out_features, rank)


def test_z_image_dense_lora_strength_reset_delete_and_unload():
    transformer = make_patched_z_image_transformer(rank=4)
    module_name = "layers.0.adaLN_modulation.0"
    module = transformer.get_submodule(module_name)
    with torch.no_grad():
        module.weight.zero_()
        module.bias.zero_()
    input_tensor = torch.ones(1, module.in_features, dtype=torch.bfloat16)
    base_output = module(input_tensor)
    first = {
        f"{module_name}.lora_A.weight": torch.ones(1, module.in_features, dtype=torch.bfloat16),
        f"{module_name}.lora_B.weight": torch.full((module.out_features, 1), 2.0, dtype=torch.bfloat16),
    }
    second = {
        f"{module_name}.lora_A.weight": torch.full((1, module.in_features), 3.0, dtype=torch.bfloat16),
        f"{module_name}.lora_B.weight": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }

    transformer.load_lora(first, strength=0.25, name="first")
    first_output = module(input_tensor)
    transformer.load_lora(second, strength=0.5, name="second")
    transformer.set_adapters("second", weights=0.25)
    second_output = module(input_tensor)

    assert transformer.get_list_adapters() == ["first", "second"]
    assert transformer.get_active_adapters() == ["second"]
    assert torch.all(first_output > base_output)
    assert torch.all(second_output > first_output)

    transformer.disable_lora()
    assert transformer.get_active_adapters() == []
    assert torch.equal(module(input_tensor), base_output)

    transformer.enable_lora()
    transformer.delete_adapters("second")
    assert transformer.get_list_adapters() == ["first"]
    transformer.unload_lora()
    assert transformer.get_list_adapters() == []
    assert torch.equal(module(input_tensor), base_output)


def test_z_image_pipeline_lora_mixin_maps_diffusers_api_to_transformer_runtime():
    transformer = make_patched_z_image_transformer(rank=4)
    module_name = "layers.0.adaLN_modulation.0"
    module = transformer.get_submodule(module_name)
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

    pipeline.load_lora_weights(lora, adapter_name="pixel")

    assert pipeline.get_list_adapters() == {"transformer": ["pixel"]}
    assert pipeline.get_active_adapters() == ["pixel"]
    pipeline.set_adapters("pixel", adapter_weights=0.5)
    assert transformer._nunchaku_lite_loras["pixel"]["strength"] == 0.5
    pipeline.unload_lora_weights()
    assert pipeline.get_list_adapters() == {"transformer": []}


def test_z_image_pipeline_lora_mixin_ignores_diffusers_metadata_tuple():
    transformer = make_patched_z_image_transformer(rank=4)
    module_name = "layers.0.adaLN_modulation.0"
    module = transformer.get_submodule(module_name)
    lora = {
        f"transformer.{module_name}.lora_A.weight": torch.ones(1, module.in_features, dtype=torch.bfloat16),
        f"transformer.{module_name}.lora_B.weight": torch.ones(module.out_features, 1, dtype=torch.bfloat16),
    }
    pipeline = SimpleNamespace(transformer=transformer)

    def lora_state_dict(self, state_dict, return_alphas=False, **kwargs):
        if kwargs.get("return_lora_metadata"):
            return state_dict, {"format": "z-image"}
        return state_dict

    pipeline.lora_state_dict = MethodType(lora_state_dict, pipeline)
    bind_pipeline_lora_methods(pipeline, NunchakuPipelineLoraMixin)

    pipeline.load_lora_weights(lora, adapter_name="pixel")

    loaded = transformer._nunchaku_lite_loras["pixel"]["state_dict"]
    assert "format" not in loaded
    assert f"{module_name}.proj_down" in loaded


def test_convert_z_image_lora_deltas_match_diffusers_peft_modules():
    base = make_tiny_z_image_transformer()
    rank = 2
    lora = {}
    target_modules = {
        "layers.0.attention.to_q": base.layers[0].attention.to_q,
        "layers.0.attention.to_k": base.layers[0].attention.to_k,
        "layers.0.attention.to_v": base.layers[0].attention.to_v,
        "layers.0.attention.to_out.0": base.layers[0].attention.to_out[0],
        "layers.0.feed_forward.w1": base.layers[0].feed_forward.w1,
        "layers.0.feed_forward.w3": base.layers[0].feed_forward.w3,
        "layers.0.feed_forward.w2": base.layers[0].feed_forward.w2,
        "layers.0.adaLN_modulation.0": base.layers[0].adaLN_modulation[0],
    }
    for index, (name, module) in enumerate(target_modules.items(), start=1):
        torch.manual_seed(index)
        lora[f"transformer.{name}.lora_A.weight"] = (torch.randn(rank, module.in_features) * 0.01).to(torch.bfloat16)
        torch.manual_seed(index + 100)
        lora[f"transformer.{name}.lora_B.weight"] = (torch.randn(module.out_features, rank) * 0.01).to(torch.bfloat16)

    diffusers_transformer = deepcopy(base)
    ZImageLoraLoaderMixin.load_lora_into_transformer(lora, diffusers_transformer, adapter_name="oracle")
    transformer = make_patched_z_image_transformer(rank=4)
    converted = transformer._convert_lora_to_nunchaku(lora)

    def diffusers_delta(name):
        module = diffusers_transformer.get_submodule(name)
        down = module.lora_A["oracle"].weight.detach().float()
        up = module.lora_B["oracle"].weight.detach().float() * module.scaling["oracle"]
        return up @ down

    qkv_down = unpack_lowrank_weight(converted["layers.0.attention.to_qkv.proj_down"], down=True)[:, :64].float()
    qkv_up = unpack_lowrank_weight(converted["layers.0.attention.to_qkv.proj_up"], down=False)[:192].float()
    qkv_expected = torch.cat(
        [
            diffusers_delta("layers.0.attention.to_q"),
            diffusers_delta("layers.0.attention.to_k"),
            diffusers_delta("layers.0.attention.to_v"),
        ],
        dim=0,
    )
    assert torch.equal(qkv_up @ qkv_down, qkv_expected)

    swiglu_down = unpack_lowrank_weight(converted["layers.0.feed_forward.net.0.proj.proj_down"], down=True)[
        :, :64
    ].float()
    swiglu_up = unpack_lowrank_weight(converted["layers.0.feed_forward.net.0.proj.proj_up"], down=False)[:340].float()
    swiglu_expected = torch.cat(
        [diffusers_delta("layers.0.feed_forward.w3"), diffusers_delta("layers.0.feed_forward.w1")],
        dim=0,
    )
    assert torch.equal(swiglu_up @ swiglu_down, swiglu_expected)

    out_down = unpack_lowrank_weight(converted["layers.0.attention.to_out.0.proj_down"], down=True)[:, :64].float()
    out_up = unpack_lowrank_weight(converted["layers.0.attention.to_out.0.proj_up"], down=False)[:64].float()
    assert torch.equal(out_up @ out_down, diffusers_delta("layers.0.attention.to_out.0"))

    w2_down = unpack_lowrank_weight(converted["layers.0.feed_forward.net.2.proj_down"], down=True)[:, :170].float()
    w2_up = unpack_lowrank_weight(converted["layers.0.feed_forward.net.2.proj_up"], down=False)[:64].float()
    assert torch.equal(w2_up @ w2_down, diffusers_delta("layers.0.feed_forward.w2"))

    adaln_down = converted["layers.0.adaLN_modulation.0.proj_down"].T.float()
    adaln_up = converted["layers.0.adaLN_modulation.0.proj_up"].float()
    assert torch.equal(adaln_up @ adaln_down, diffusers_delta("layers.0.adaLN_modulation.0"))


def test_convert_z_image_lora_deltas_match_diffusers_peft_modules_for_each_layer():
    base = make_tiny_z_image_transformer()
    lora = make_z_image_lora_for_all_targets(base, scale=0.01)
    diffusers_transformer = deepcopy(base)
    ZImageLoraLoaderMixin.load_lora_into_transformer(lora, diffusers_transformer, adapter_name="oracle")
    transformer = make_patched_z_image_transformer(rank=4)

    converted = transformer._convert_lora_to_nunchaku(lora)
    runtime_targets = sorted(key[: -len(".proj_down")] for key in converted if key.endswith(".proj_down"))

    assert runtime_targets
    for runtime_name in runtime_targets:
        actual = converted_z_image_runtime_delta(transformer, converted, runtime_name)
        expected = expected_z_image_runtime_delta(diffusers_transformer, runtime_name)

        assert actual.shape == expected.shape, runtime_name
        assert torch.equal(actual, expected), runtime_name


@pytest.mark.skipif(
    os.environ.get("NUNCHAKU_LITE_RUN_NATIVE_CUDA_TESTS") != "1",
    reason="set NUNCHAKU_LITE_RUN_NATIVE_CUDA_TESTS=1 to run CUDA native SVDQ tests",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA native SVDQ kernels")
def test_z_image_native_lora_output_deltas_match_diffusers_for_each_layer():
    base = make_native_kernel_z_image_transformer()
    lora = make_z_image_lora_for_all_targets(base)
    diffusers_transformer = deepcopy(base)
    ZImageLoraLoaderMixin.load_lora_into_transformer(lora, diffusers_transformer, adapter_name="oracle")

    transformer = make_patched_native_kernel_z_image_transformer()
    normalize_native_lora_test_base(transformer)
    transformer.load_lora(lora, name="all")
    transformer.to("cuda")

    state_dict = transformer._nunchaku_lite_loras["all"]["state_dict"]
    runtime_targets = sorted(key[: -len(".proj_down")] for key in state_dict if key.endswith(".proj_down"))
    compared_targets = set()

    for index, runtime_name in enumerate(runtime_targets, start=1):
        module = transformer.get_submodule(runtime_name)
        torch.manual_seed(index + 2000)
        input_tensor = (torch.randn(2, 3, module.in_features, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()

        with torch.no_grad():
            enabled_output = module(input_tensor)
            transformer.disable_lora()
            disabled_output = module(input_tensor)
            transformer.enable_lora()

        observed_delta = (enabled_output - disabled_output).float()
        expected_weight = expected_z_image_runtime_delta(diffusers_transformer, runtime_name).to("cuda")
        expected_delta = torch.matmul(input_tensor.float(), expected_weight.t())

        assert expected_delta.abs().max() > 1e-5, f"{runtime_name} expected LoRA delta is unexpectedly zero."
        assert torch.allclose(observed_delta, expected_delta, rtol=0.08, atol=0.03), runtime_name
        compared_targets.add(runtime_name)

    assert compared_targets == set(runtime_targets)


@pytest.mark.skipif(
    os.environ.get("NUNCHAKU_LITE_RUN_NATIVE_CUDA_TESTS") != "1",
    reason="set NUNCHAKU_LITE_RUN_NATIVE_CUDA_TESTS=1 to run CUDA native SVDQ tests",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA native SVDQ kernels")
def test_z_image_native_lora_changes_attention_and_feed_forward_blocks():
    base = make_native_kernel_z_image_transformer()
    lora = make_z_image_lora_for_all_targets(base)
    transformer = make_patched_native_kernel_z_image_transformer()
    normalize_native_lora_test_base(transformer)
    transformer.load_lora(lora, name="all")
    transformer.to("cuda")

    for index, group_name in enumerate(Z_IMAGE_BLOCK_GROUPS, start=1):
        block = getattr(transformer, group_name)[0]
        torch.manual_seed(index + 3000)
        hidden_size = block.attention.to_qkv.in_features
        hidden_states = (torch.randn(1, 4, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()

        with torch.no_grad():
            attention_enabled = block.attention(hidden_states)
            feed_forward_enabled = block.feed_forward(hidden_states)
            transformer.disable_lora()
            attention_disabled = block.attention(hidden_states)
            feed_forward_disabled = block.feed_forward(hidden_states)
            transformer.enable_lora()

        for label, enabled, disabled in (
            (f"{group_name}.0.attention", attention_enabled, attention_disabled),
            (f"{group_name}.0.feed_forward", feed_forward_enabled, feed_forward_disabled),
        ):
            assert enabled.shape == disabled.shape == hidden_states.shape
            assert torch.isfinite(enabled).all(), f"{label} enabled output contains non-finite values."
            assert torch.isfinite(disabled).all(), f"{label} disabled output contains non-finite values."
            assert (enabled - disabled).float().abs().max() > 1e-5, f"{label} did not respond to LoRA."


def test_z_image_unsupported_lora_target_raises():
    transformer = make_patched_z_image_transformer()
    lora = {
        "diffusion_model.layers.0.not_a_module.lora_A.weight": torch.ones(2, 64),
        "diffusion_model.layers.0.not_a_module.lora_B.weight": torch.ones(64, 2),
    }

    with pytest.raises(ValueError, match="Unsupported Nunchaku LoRA target"):
        transformer.load_lora(lora)
