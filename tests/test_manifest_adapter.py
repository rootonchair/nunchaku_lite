import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.manifest import ManifestAdapter, SplitLinearInput, SplitLinearOutput
from nunchaku_lite.linear import AWQW4A16Linear, SVDQW4A4Linear
from nunchaku_lite.manifest import parse_runtime_manifest


class TinyManifestModel(nn.Module):
    def __init__(self, out_features: int = 128):
        super().__init__()
        self.proj = nn.Linear(128, out_features)

    def forward(self, x):
        return self.proj(x)


class MatchingFakeAdapter:
    target = "manifest_fake"

    def matches(self, transformer):
        return isinstance(transformer, TinyManifestModel)

    def patch(self, transformer, checkpoint_state, quantization_config, options):
        transformer.fake_adapter_used = True
        return checkpoint_state


def _manifest(*, op="svdq_w4a4", precision="int4", group_size=64, rank=4, patches=None, has_bias=True):
    return {
        "schema": "nunchaku_lite.runtime_manifest",
        "version": 1,
        "component": "transformer",
        "nunchaku_format_version": 1,
        "producer": {"name": "test", "version": "0"},
        "requirements": {
            "method": "svdquant",
            "precision": precision,
            "rank": rank,
            "weight_dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
            "activation_dtype": "int4",
            "torch_dtype": None,
        },
        "structural_patches": patches or [],
        "targets": [
            {
                "name": "proj",
                "checkpoint_prefix": "proj",
                "source_modules": ["proj"],
                "roles": [],
                "kind": "linear",
                "nunchaku_op": op,
                "precision": precision,
                "group_size": group_size,
                "rank": rank,
                "has_bias": has_bias,
                "op_options": {"adanorm_splits": 6} if op == "adanorm_awq_w4a16" else {},
                "activation": {},
            }
        ],
    }


def _quantization_config(manifest):
    return {"runtime_manifest": manifest}


def _write_manifest_checkpoint(tmp_path, model, manifest):
    quantization_config = _quantization_config(manifest)
    ManifestAdapter().patch(
        model,
        {},
        quantization_config,
        type(
            "Options",
            (),
            {
                "precision": parse_runtime_manifest(quantization_config).runtime_precision or "int4",
                "torch_dtype": torch.bfloat16,
                "device": None,
                "strict": True,
                "adapter_options": {},
            },
        )(),
    )
    checkpoint = tmp_path / "manifest.safetensors"
    save_file(
        model.state_dict(),
        checkpoint,
        metadata={"quantization_config": json.dumps(quantization_config)},
    )
    return checkpoint


def test_parse_runtime_manifest_rejects_unsupported_schema():
    quantization_config = _quantization_config({**_manifest(), "schema": "other"})

    with pytest.raises(ValueError, match="Unsupported runtime_manifest schema"):
        parse_runtime_manifest(quantization_config)


def test_patch_transformer_manifest_target_replaces_svdq_linear(tmp_path):
    manifest = _manifest()
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)

    transformer = TinyManifestModel()
    patch_transformer(
        transformer,
        checkpoint,
        target="manifest",
        precision="auto",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    assert transformer._nunchaku_lite_target == "manifest"
    assert isinstance(transformer.proj, SVDQW4A4Linear)
    assert transformer.proj.rank == 4
    assert transformer.proj.precision == "int4"


def test_patch_transformer_auto_uses_manifest_before_matching_adapter(tmp_path, monkeypatch):
    import nunchaku_lite.core as core

    manifest = _manifest()
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)
    monkeypatch.setitem(core._ADAPTERS, MatchingFakeAdapter.target, MatchingFakeAdapter())

    transformer = TinyManifestModel()
    patch_transformer(transformer, checkpoint, target="auto", torch_dtype=torch.bfloat16, device="cpu")

    assert transformer._nunchaku_lite_target == "manifest"
    assert not hasattr(transformer, "fake_adapter_used")


def test_patch_transformer_auto_falls_back_when_manifest_absent(tmp_path, monkeypatch):
    import nunchaku_lite.core as core

    checkpoint = tmp_path / "dense.safetensors"
    save_file(
        TinyManifestModel().state_dict(),
        checkpoint,
        metadata={"quantization_config": json.dumps({})},
    )
    monkeypatch.setitem(core._ADAPTERS, MatchingFakeAdapter.target, MatchingFakeAdapter())

    transformer = TinyManifestModel()
    patch_transformer(transformer, checkpoint, target="auto", torch_dtype=torch.bfloat16, device="cpu")

    assert transformer._nunchaku_lite_target == MatchingFakeAdapter.target
    assert transformer.fake_adapter_used


def test_patch_transformer_manifest_target_requires_manifest(tmp_path):
    checkpoint = tmp_path / "dense.safetensors"
    save_file(
        TinyManifestModel().state_dict(),
        checkpoint,
        metadata={"quantization_config": json.dumps({})},
    )

    with pytest.raises(ValueError, match="requires quantization_config.runtime_manifest"):
        patch_transformer(TinyManifestModel(), checkpoint, target="manifest", torch_dtype=torch.bfloat16, device="cpu")


def test_manifest_adapter_applies_split_linear_output_before_replacement(tmp_path):
    manifest = _manifest(
        patches=[{"type": "split_linear_output", "module": "proj", "args": {"splits": [64]}}],
    )
    source = TinyManifestModel(out_features=128)
    checkpoint = _write_manifest_checkpoint(tmp_path, source, manifest)
    assert isinstance(source.proj, SVDQW4A4Linear)

    transformer = TinyManifestModel(out_features=128)
    ManifestAdapter().patch(
        transformer,
        {},
        _quantization_config(manifest),
        type(
            "Options",
            (),
            {
                "precision": "int4",
                "torch_dtype": torch.bfloat16,
                "device": None,
                "strict": True,
                "adapter_options": {},
            },
        )(),
    )

    assert isinstance(transformer.proj, SVDQW4A4Linear)
    patch_transformer(
        TinyManifestModel(out_features=128),
        checkpoint,
        target="manifest",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )


def test_manifest_structural_patch_classes_preserve_linear_metadata():
    linear = nn.Linear(128, 128)

    split_input = SplitLinearInput.from_linear(linear, [64])
    split_output = SplitLinearOutput.from_linear(linear, [64])

    assert split_input.in_features == 128
    assert split_input.out_features == 128
    assert split_output.in_features == 128
    assert split_output.out_features == 128


def test_manifest_adapter_replaces_awq_target(tmp_path):
    manifest = _manifest(op="awq_w4a16", precision="int4", group_size=64, rank=0)
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)

    transformer = TinyManifestModel()
    patch_transformer(
        transformer,
        checkpoint,
        target="manifest",
        precision="int4",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    assert isinstance(transformer.proj, AWQW4A16Linear)
