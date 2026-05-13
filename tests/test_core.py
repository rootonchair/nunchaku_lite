import sys

import pytest
import torch


def test_import_does_not_import_full_nunchaku():
    sys.modules.pop("nunchaku", None)
    import nunchaku_lite

    assert "nunchaku" not in sys.modules
    assert "flux" in nunchaku_lite.list_adapters()
    assert "flux2" in nunchaku_lite.list_adapters()
    assert "z_image" in nunchaku_lite.list_adapters()


def test_unsupported_transformer_error_lists_adapters():
    from nunchaku_lite import patch_transformer

    with pytest.raises(ValueError, match="Available adapters: flux, flux2, z_image"):
        patch_transformer(torch.nn.Linear(1, 1), "missing/repo/checkpoint.safetensors")
