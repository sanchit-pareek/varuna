"""Raster overlays for the map.

Leaflet gets two PNGs per scene: a grayscale Sigma0 backdrop and a transparent
class overlay. Both are written with a small builtin PNG encoder so Pillow stays
optional, and both are placed with the same lat/lon bounds derived from the SAR
affine, which is why the polygons land exactly on the dark patch.

Large scenes are decimated before encoding. A 2048 pixel chip drawn on a laptop
screen does not need every pixel, and the browser should not be handed 8 MB of
PNG during a three minute pitch.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..geo.raster import Raster, stretch_to_uint8

MAX_SIDE = 1400

CLASS_RGBA = {
    1: (255, 214, 10, 70),     # look-alike, yellow and deliberately faint
    2: (255, 45, 200, 150),    # mineral oil, neon magenta
}


def write_png(path: Path, array: np.ndarray) -> Path:
    """Write an 8 bit PNG. Accepts (h, w) grey, (h, w, 3) RGB or (h, w, 4) RGBA."""
    a = np.asarray(array)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    if a.ndim == 2:
        color_type, channels = 0, 1
        raw = a[:, :, None]
    elif a.shape[2] == 3:
        color_type, channels = 2, 3
        raw = a
    elif a.shape[2] == 4:
        color_type, channels = 6, 4
        raw = a
    else:
        raise ValueError("unsupported array shape %r" % (a.shape,))

    h, w = raw.shape[0], raw.shape[1]
    lines = bytearray()
    stride = w * channels
    flat = raw.reshape(h, stride)
    for y in range(h):
        lines.append(0)  # filter type none
        lines.extend(flat[y].tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(lines), 6))
    png += chunk(b"IEND", b"")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def _decimate(a: np.ndarray, max_side: int = MAX_SIDE) -> Tuple[np.ndarray, int]:
    h, w = a.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / float(max_side))))
    return (a[::step, ::step] if step > 1 else a), step


def sar_backdrop(sar: Raster, out_path: Path, band: int = 0,
                 max_side: int = MAX_SIDE) -> Dict[str, Any]:
    """Percentile stretched grayscale PNG of the Sigma0 band."""
    arr = sar.array[band] if sar.array.ndim == 3 else sar.array
    small, step = _decimate(np.asarray(arr, dtype=np.float32), max_side)
    grey = stretch_to_uint8(small)
    write_png(out_path, grey)
    return {
        "path": str(out_path),
        "url": "/data/" + _rel(out_path),
        "bounds": _leaflet_bounds(sar),
        "decimation": step,
        "size": [int(grey.shape[1]), int(grey.shape[0])],
    }


def class_overlay(mask: np.ndarray, sar: Raster, out_path: Path,
                  max_side: int = MAX_SIDE) -> Dict[str, Any]:
    """Transparent RGBA overlay of the look-alike and oil classes."""
    small, step = _decimate(np.asarray(mask), max_side)
    rgba = np.zeros((small.shape[0], small.shape[1], 4), dtype=np.uint8)
    for klass, color in CLASS_RGBA.items():
        sel = small == klass
        if sel.any():
            rgba[sel] = color
    write_png(out_path, rgba)
    return {
        "path": str(out_path),
        "url": "/data/" + _rel(out_path),
        "bounds": _leaflet_bounds(sar),
        "decimation": step,
        "size": [int(rgba.shape[1]), int(rgba.shape[0])],
    }


def _leaflet_bounds(sar: Raster):
    """[[south, west], [north, east]] as Leaflet imageOverlay wants it."""
    w, s, e, n = sar.bounds_lonlat()
    return [[s, w], [n, e]]


def _rel(path: Path) -> str:
    from .. import config

    p = Path(path).resolve()
    try:
        return p.relative_to(Path(config.DATA_DIR).resolve()).as_posix()
    except ValueError:
        return p.name


def thumbnail(sar: Raster, out_path: Path, side: int = 220) -> Dict[str, Any]:
    arr = sar.array[0] if sar.array.ndim == 3 else sar.array
    small, _ = _decimate(np.asarray(arr, dtype=np.float32), side)
    write_png(out_path, stretch_to_uint8(small))
    return {"path": str(out_path), "url": "/data/" + _rel(out_path)}
