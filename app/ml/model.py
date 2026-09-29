"""U-Net builder and checkpoint loading.

torch and segmentation_models_pytorch are optional at runtime. Importing this
module never imports torch; the import happens inside the functions, so a
laptop without torch still starts the API and runs the baseline path.

Frozen architecture from the spec:
    UnetPlusPlus or Unet, encoder timm-efficientnet-b0 (default) or resnet34
    3 classes: 0 sea, 1 look_alike, 2 mineral_oil
    input 512x512, 2 channel VV+VH (VV repeated to 3 channels if the encoder
    insists on 3)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .. import config

DEFAULT_ARCH = "UnetPlusPlus"
DEFAULT_ENCODER = "timm-efficientnet-b0"
N_CLASSES = 3
IN_CHANNELS = 3  # VV, VH, VV repeated: keeps ImageNet encoder stems happy


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def smp_available() -> bool:
    try:
        import segmentation_models_pytorch  # noqa: F401

        return True
    except Exception:
        return False


def device_name() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build(arch: str = DEFAULT_ARCH, encoder: str = DEFAULT_ENCODER,
          encoder_weights: Optional[str] = "imagenet", classes: int = N_CLASSES,
          in_channels: int = IN_CHANNELS):
    """Construct the segmentation network. Raises if smp is not installed."""
    import segmentation_models_pytorch as smp

    factory = {
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
    }.get(arch)
    if factory is None:
        raise ValueError("unsupported architecture %r" % arch)
    return factory(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


def load_checkpoint(path: Path = None, map_location: str = None) -> Dict[str, Any]:
    """Load a checkpoint written by `train.py`.

    The checkpoint carries its own architecture metadata so inference never has
    to guess. Returns a dict with the model, its config and the reported metrics.
    """
    import torch

    path = Path(config.CHECKPOINT if path is None else path)
    if not path.exists():
        raise FileNotFoundError("checkpoint not found: %s" % path)
    map_location = map_location or device_name()
    blob = torch.load(str(path), map_location=map_location, weights_only=False)

    if isinstance(blob, dict) and "state_dict" in blob:
        meta = blob.get("meta", {})
        arch = meta.get("arch", DEFAULT_ARCH)
        encoder = meta.get("encoder", DEFAULT_ENCODER)
        in_ch = int(meta.get("in_channels", IN_CHANNELS))
        classes = int(meta.get("classes", N_CLASSES))
        model = build(arch, encoder, encoder_weights=None, classes=classes, in_channels=in_ch)
        model.load_state_dict(blob["state_dict"])
        metrics = blob.get("metrics", {})
        norm = meta.get("normalisation", {})
    else:  # a bare state_dict
        model = build(encoder_weights=None)
        model.load_state_dict(blob)
        meta, metrics, norm = {}, {}, {}
        arch, encoder, in_ch, classes = DEFAULT_ARCH, DEFAULT_ENCODER, IN_CHANNELS, N_CLASSES

    model.eval()
    model.to(map_location)
    return {
        "model": model,
        "arch": arch,
        "encoder": encoder,
        "in_channels": in_ch,
        "classes": classes,
        "device": map_location,
        "metrics": metrics,
        "normalisation": norm or {"mean_db": -18.0, "std_db": 6.0},
        "path": str(path),
        "meta": meta,
    }


def save_checkpoint(model, path: Path, meta: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta, "metrics": metrics}, str(path))


def status() -> Dict[str, Any]:
    """What /api/health reports about the model half of the system."""
    ck = Path(config.CHECKPOINT)
    return {
        "torch": torch_available(),
        "smp": smp_available(),
        "cuda": cuda_available(),
        "device": device_name(),
        "checkpoint_path": str(ck),
        "checkpoint_present": ck.exists(),
        "checkpoint_mb": round(ck.stat().st_size / 1e6, 1) if ck.exists() else None,
    }
