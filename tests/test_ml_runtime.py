"""Runtime contract for optional PyTorch segmentation inference."""
from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")


@pytest.mark.parametrize("arch", ["Unet", "UnetPlusPlus"])
def test_supported_architectures_build_for_cpu_inference(arch):
    from app.ml import model

    net = model.build(arch=arch, encoder_weights=None)
    net.eval().to("cpu")
    assert net is not None


def test_explicit_cpu_checkpoint_load_stays_on_cpu(tmp_path):
    from app.ml import model

    net = model.build(arch="Unet", encoder_weights=None)
    path = tmp_path / "unet.pt"
    model.save_checkpoint(
        net,
        path,
        meta={"arch": "Unet", "encoder": model.DEFAULT_ENCODER,
              "classes": model.N_CLASSES, "in_channels": model.IN_CHANNELS},
        metrics={},
    )

    loaded = model.load_checkpoint(path, map_location="cpu")
    assert loaded["arch"] == "Unet"
    assert loaded["device"] == "cpu"
