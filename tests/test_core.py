import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


class FakePipelineComponent(torch.nn.Module):
    config_kwargs = None

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 2))

    @classmethod
    def load_config(cls, model_id, **kwargs):
        cls.config_kwargs = kwargs
        return {"model_id": str(model_id)}

    @classmethod
    def from_config(cls, config):
        return cls()


class FakePipeline:
    load_config_kwargs = None
    from_pretrained_kwargs = None

    def __init__(self, transformer):
        self.transformer = transformer

    @classmethod
    def _get_signature_keys(cls, obj):
        return ["transformer"], []

    @classmethod
    def load_config(cls, model_id, **kwargs):
        cls.load_config_kwargs = kwargs
        return {"transformer": [__name__, "FakePipelineComponent"]}

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.from_pretrained_kwargs = kwargs
        return cls(kwargs["transformer"])


class FakeUnetPipeline(FakePipeline):
    def __init__(self, unet):
        self.unet = unet

    @classmethod
    def _get_signature_keys(cls, obj):
        return ["unet"], []

    @classmethod
    def load_config(cls, model_id, **kwargs):
        cls.load_config_kwargs = kwargs
        return {"unet": [__name__, "FakePipelineComponent"]}

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.from_pretrained_kwargs = kwargs
        return cls(kwargs["unet"])


class FakeAdapter:
    target = "fake_meta"

    def matches(self, transformer):
        return isinstance(transformer, FakePipelineComponent)

    def patch(self, transformer, checkpoint_state, quantization_config, options):
        transformer.patch_context = SimpleNamespace(
            checkpoint_keys=set(checkpoint_state),
            quantization_config=quantization_config,
            options=options,
            was_meta=transformer.weight.is_meta,
        )
        return checkpoint_state

    def patch_pipeline(self, pipeline, *, component_name, component):
        pipeline.patch_pipeline_context = SimpleNamespace(
            component_name=component_name,
            component=component,
            adapter_target=self.target,
        )


class FakeAdapterWithoutPipelinePatch(FakeAdapter):
    target = "fake_meta_no_pipeline_patch"

    patch_pipeline = None


def _fake_checkpoint(tmp_path):
    checkpoint = tmp_path / "fake.safetensors"
    save_file({"weight": torch.ones(2, 2)}, checkpoint, metadata={"quantization_config": "{}"})
    return checkpoint


def _fake_quantized_checkpoint(tmp_path, quantization_config):
    import json

    checkpoint = tmp_path / "fake-quantized.safetensors"
    save_file(
        {"weight": torch.ones(2, 2)},
        checkpoint,
        metadata={"quantization_config": json.dumps(quantization_config)},
    )
    return checkpoint


def _install_fake_adapter(monkeypatch):
    import nunchaku_lite.core as core

    monkeypatch.setitem(core._ADAPTERS, FakeAdapter.target, FakeAdapter())


def _install_fake_adapter_without_pipeline_patch(monkeypatch):
    import nunchaku_lite.core as core

    monkeypatch.setitem(
        core._ADAPTERS,
        FakeAdapterWithoutPipelinePatch.target,
        FakeAdapterWithoutPipelinePatch(),
    )


def test_import_does_not_import_full_nunchaku():
    sys.modules.pop("nunchaku", None)
    import nunchaku_lite

    assert "nunchaku" not in sys.modules
    assert "flux" in nunchaku_lite.list_adapters()
    assert "flux2" in nunchaku_lite.list_adapters()
    assert "manifest" in nunchaku_lite.list_adapters()
    assert "qwen_image" in nunchaku_lite.list_adapters()
    assert "sdxl" in nunchaku_lite.list_adapters()
    assert "z_image" in nunchaku_lite.list_adapters()


def test_unsupported_transformer_error_lists_adapters(tmp_path):
    from nunchaku_lite import patch_transformer

    checkpoint = _fake_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="Available adapters: flux, flux2, manifest, qwen_image, sdxl, z_image"):
        patch_transformer(torch.nn.Linear(1, 1), checkpoint)


def test_load_nunchaku_pipeline_injects_meta_loaded_transformer(tmp_path, monkeypatch):
    from nunchaku_lite import load_nunchaku_pipeline

    _install_fake_adapter(monkeypatch)
    pipe = load_nunchaku_pipeline(
        tmp_path,
        pipeline_cls=FakePipeline,
        checkpoint=_fake_checkpoint(tmp_path),
        target=FakeAdapter.target,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        custom_arg="forwarded",
    )

    transformer = pipe.transformer
    assert transformer._nunchaku_lite_patched
    assert transformer._nunchaku_lite_target == FakeAdapter.target
    assert transformer.patch_context.was_meta
    assert not transformer.weight.is_meta
    assert torch.equal(transformer.weight, torch.ones(2, 2))
    assert FakePipeline.from_pretrained_kwargs["transformer"] is transformer
    assert FakePipeline.from_pretrained_kwargs["torch_dtype"] is torch.bfloat16
    assert FakePipeline.from_pretrained_kwargs["custom_arg"] == "forwarded"
    assert FakePipelineComponent.config_kwargs["subfolder"] == "transformer"
    assert FakePipelineComponent.config_kwargs["local_files_only"] is True
    assert pipe.patch_pipeline_context.component_name == "transformer"
    assert pipe.patch_pipeline_context.component is transformer
    assert pipe.patch_pipeline_context.adapter_target == FakeAdapter.target


def test_load_nunchaku_pipeline_allows_adapter_without_pipeline_patch(tmp_path, monkeypatch):
    from nunchaku_lite import load_nunchaku_pipeline

    _install_fake_adapter_without_pipeline_patch(monkeypatch)
    pipe = load_nunchaku_pipeline(
        tmp_path,
        pipeline_cls=FakePipeline,
        checkpoint=_fake_checkpoint(tmp_path),
        target=FakeAdapterWithoutPipelinePatch.target,
    )

    assert pipe.transformer._nunchaku_lite_patched
    assert not hasattr(pipe, "patch_pipeline_context")


def test_load_nunchaku_pipeline_rejects_removed_bind_lora_argument(tmp_path, monkeypatch):
    from nunchaku_lite import load_nunchaku_pipeline

    _install_fake_adapter(monkeypatch)
    with pytest.raises(TypeError, match="unexpected keyword argument 'bind_lora'"):
        load_nunchaku_pipeline(
            tmp_path,
            pipeline_cls=FakePipeline,
            checkpoint=_fake_checkpoint(tmp_path),
            target=FakeAdapter.target,
            bind_lora=False,
        )


def test_load_nunchaku_pipeline_auto_selects_unet(tmp_path, monkeypatch):
    from nunchaku_lite import load_nunchaku_pipeline

    _install_fake_adapter(monkeypatch)
    pipe = load_nunchaku_pipeline(
        tmp_path,
        pipeline_cls=FakeUnetPipeline,
        checkpoint=_fake_checkpoint(tmp_path),
        target=FakeAdapter.target,
        component=None,
    )

    assert pipe.unet._nunchaku_lite_patched
    assert FakeUnetPipeline.from_pretrained_kwargs["unet"] is pipe.unet
    assert FakePipelineComponent.config_kwargs["subfolder"] == "unet"


def test_patch_transformer_warns_for_int4_checkpoint_on_blackwell(tmp_path, monkeypatch):
    from nunchaku_lite import patch_transformer

    _install_fake_adapter(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (12, 0))
    checkpoint = _fake_quantized_checkpoint(
        tmp_path,
        {"weight": {"dtype": "int4", "group_size": 64}},
    )

    with pytest.warns(UserWarning, match="INT4 quantization on Blackwell GPUs may be slower than FP4"):
        transformer = patch_transformer(
            FakePipelineComponent(),
            checkpoint,
            target=FakeAdapter.target,
            precision="int4",
        )

    assert transformer._nunchaku_lite_patched


def test_patch_transformer_rejects_precision_checkpoint_mismatch(tmp_path, monkeypatch):
    from nunchaku_lite import patch_transformer

    _install_fake_adapter(monkeypatch)
    checkpoint = _fake_quantized_checkpoint(
        tmp_path,
        {"weight": {"dtype": "fp4_e2m1_all", "group_size": 16}},
    )

    with pytest.raises(ValueError, match="is fp4, but precision='int4'"):
        patch_transformer(
            FakePipelineComponent(),
            checkpoint,
            target=FakeAdapter.target,
            precision="int4",
            device="cpu",
        )


def test_patch_scale_key_materializes_missing_fp4_scales_from_meta():
    from nunchaku_lite.utils import patch_scale_key

    with torch.device("meta"):
        module = torch.nn.Module()
        module.proj = torch.nn.Module()
        module.proj.wcscales = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16), requires_grad=False)
    state = {}

    patch_scale_key(module, state)

    assert state["proj.wcscales"].device.type == "cpu"
    assert state["proj.wcscales"].dtype == torch.bfloat16
    assert torch.equal(state["proj.wcscales"], torch.ones(4, dtype=torch.bfloat16))


def test_materialize_known_meta_tensors_rebuilds_qwen_rope_attributes():
    from nunchaku_lite.core import _materialize_known_meta_tensors

    class FakeQwenRope(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = 10000
            self.axes_dim = [4, 6, 6]
            with torch.device("meta"):
                self.pos_freqs = torch.empty(4096, 8, dtype=torch.complex64)
                self.neg_freqs = torch.empty(4096, 8, dtype=torch.complex64)

        def rope_params(self, index, dim, theta=10000):
            freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
            return torch.polar(torch.ones_like(freqs), freqs)

    module = torch.nn.Module()
    module.rope = FakeQwenRope()

    _materialize_known_meta_tensors(module)

    assert module.rope.pos_freqs.device.type == "cpu"
    assert module.rope.neg_freqs.device.type == "cpu"
    assert module.rope.pos_freqs.shape == (4096, 8)
    assert module.rope.neg_freqs.shape == (4096, 8)
    assert not module.rope.pos_freqs.is_meta
    assert not module.rope.neg_freqs.is_meta


def test_coerce_state_dict_for_assign_matches_module_float_dtypes():
    from nunchaku_lite.core import _coerce_state_dict_for_assign

    module = torch.nn.Module()
    module.scale = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16), requires_grad=False)
    module.quantized = torch.nn.Parameter(torch.empty(4, dtype=torch.int8), requires_grad=False)
    state = {
        "scale": torch.ones(4, dtype=torch.float8_e4m3fn),
        "quantized": torch.ones(4, dtype=torch.int8),
    }

    _coerce_state_dict_for_assign(module, state)

    assert state["scale"].dtype == torch.bfloat16
    assert state["quantized"].dtype == torch.int8
