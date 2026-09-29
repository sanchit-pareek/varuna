"""Raster IO and georeferencing.

The single most important rule from the spec lives here: Zenodo masks are often
bare 2048x2048 TIFFs with no CRS and no transform. Every centroid computed from
such a mask is fiction unless the affine and CRS are copied from the matching
Sigma0 image first. `load_mask_with_georef` is the only sanctioned way to read a
mask in this codebase.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import tiffio

try:  # optional but preferred
    import rasterio
    from rasterio.transform import Affine

    _HAVE_RASTERIO = True
except Exception:  # pragma: no cover
    _HAVE_RASTERIO = False

try:
    from pyproj import Transformer

    _HAVE_PYPROJ = True
except Exception:  # pragma: no cover
    _HAVE_PYPROJ = False


class GeorefError(RuntimeError):
    """Raised when a raster cannot be placed on the Earth."""


@dataclass
class Raster:
    """A georeferenced raster stack, bands first."""

    array: np.ndarray                     # (bands, h, w) float32
    transform: Tuple[float, float, float, float, float, float]
    crs: str
    path: Optional[Path] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.array.shape[-2])

    @property
    def width(self) -> int:
        return int(self.array.shape[-1])

    @property
    def count(self) -> int:
        return int(self.array.shape[0])

    # -- coordinate helpers -------------------------------------------------
    def xy(self, col, row):
        """Pixel centre to native CRS coordinates."""
        a, b, c, d, e, f = self.transform
        col = np.asarray(col, dtype=float) + 0.5
        row = np.asarray(row, dtype=float) + 0.5
        return a * col + b * row + c, d * col + e * row + f

    def lonlat(self, col, row):
        """Pixel centre to EPSG:4326 lon/lat."""
        x, y = self.xy(col, row)
        return to_wgs84(x, y, self.crs)

    def bounds_lonlat(self) -> Tuple[float, float, float, float]:
        cols = np.array([0, self.width - 1, 0, self.width - 1], dtype=float)
        rows = np.array([0, 0, self.height - 1, self.height - 1], dtype=float)
        lon, lat = self.lonlat(cols, rows)
        return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))

    def centre_lonlat(self) -> Tuple[float, float]:
        lon, lat = self.lonlat(self.width / 2.0, self.height / 2.0)
        return float(lon), float(lat)

    def pixel_area_km2(self) -> float:
        """Ground area of one pixel, using the raster centre for scale."""
        a, b, _c, d, e, _f = self.transform
        if self.crs.upper().endswith("4326"):
            _lon0, lat0 = self.centre_lonlat()
            from .crs import meters_per_degree

            mx, my = meters_per_degree(lat0)
            sx = math.hypot(a * mx, d * my)
            sy = math.hypot(b * mx, e * my)
        else:  # projected CRS, units already metres
            sx = math.hypot(a, d)
            sy = math.hypot(b, e)
        return (sx * sy) / 1.0e6


def to_wgs84(x, y, crs: str):
    """Native coordinates to lon/lat. Identity for EPSG:4326."""
    crs_u = (crs or "EPSG:4326").upper()
    if crs_u in ("EPSG:4326", "WGS84", "OGC:CRS84"):
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not _HAVE_PYPROJ:
        raise GeorefError(
            "raster is in %s and pyproj is not installed; reproject the scene to "
            "EPSG:4326 or install pyproj" % crs
        )
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def _read_any(path: Path) -> Dict[str, Any]:
    if _HAVE_RASTERIO:
        with rasterio.open(path) as ds:
            arr = ds.read().astype(np.float32)
            t = ds.transform
            transform = (t.a, t.b, t.c, t.d, t.e, t.f)
            crs = ds.crs.to_string() if ds.crs else None
            tags = ds.tags()
        return {"array": arr, "transform": transform, "crs": crs, "datetime": tags.get("TIFFTAG_DATETIME")}
    out = tiffio.read(path)
    out["array"] = out["array"].astype(np.float32)
    return out


def load_sar(path) -> Raster:
    """Load a Sigma0 SAR chip. Must carry its own georeference."""
    path = Path(path)
    info = _read_any(path)
    if info["transform"] is None or info["crs"] is None:
        raise GeorefError(
            "SAR chip %s has no affine transform or CRS. A scene without "
            "georeference cannot produce real coordinates and must be dropped "
            "from the demo set." % path.name
        )
    return Raster(
        array=info["array"],
        transform=tuple(float(v) for v in info["transform"]),
        crs=str(info["crs"]),
        path=path,
        meta={"datetime": info.get("datetime")},
    )


def load_raw(path) -> Raster:
    """Read pixels with no georeference requirement.

    `load_sar` refuses a chip with no affine, and it is right to: a centroid
    from an ungeoreferenced raster is fiction, so such a chip must never reach
    the product. Training is the exception. The Zenodo set ships many chips with
    no CRS at all, and their pixels are still perfectly good supervision, so the
    tiler needs a way to read them that does not care about the Earth.

    The identity transform returned here is a marker, not a location. Nothing
    that computes coordinates may use this function.
    """
    path = Path(path)
    info = _read_any(path)
    arr = info["array"]
    if arr.ndim == 2:
        arr = arr[None, :, :]
    transform = info.get("transform") or (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    return Raster(
        array=arr.astype(np.float32),
        transform=tuple(float(v) for v in transform),
        crs=str(info.get("crs") or "EPSG:4326"),
        path=path,
        meta={"georeferenced": info.get("transform") is not None and info.get("crs") is not None,
              "read_as": "raw"},
    )


def load_mask_with_georef(mask_path, sar: Raster) -> Raster:
    """Load a label mask and force the SAR georeference onto it.

    Zenodo masks routinely ship with no CRS. Copying the affine from the
    matching Sigma0 image is mandatory before any lat/lon maths. If the mask
    grid does not match the SAR grid the mask is resampled with nearest
    neighbour so that class codes survive.
    """
    mask_path = Path(mask_path)
    info = _read_any(mask_path)
    arr = info["array"]
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape != (sar.height, sar.width):
        arr = _nearest_resize(arr, sar.height, sar.width)
    return Raster(
        array=arr[None, :, :],
        transform=sar.transform,
        crs=sar.crs,
        path=mask_path,
        meta={"georef_copied_from": str(sar.path)},
    )


def _nearest_resize(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = (np.arange(height) * (arr.shape[0] / float(height))).astype(int).clip(0, arr.shape[0] - 1)
    cols = (np.arange(width) * (arr.shape[1] / float(width))).astype(int).clip(0, arr.shape[1] - 1)
    return arr[np.ix_(rows, cols)]


def write_geotiff(path, array: np.ndarray, transform: Sequence[float], crs: str = "EPSG:4326") -> None:
    """Write a GeoTIFF using rasterio when available, else the builtin writer."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if _HAVE_RASTERIO:
        a, b, c, d, e, f = transform
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=arr.shape[1],
            width=arr.shape[2],
            count=arr.shape[0],
            dtype=arr.dtype,
            crs=crs,
            transform=Affine(a, b, c, d, e, f),
        ) as ds:
            ds.write(arr)
        return
    epsg = 4326
    if crs and ":" in crs:
        try:
            epsg = int(crs.split(":")[-1])
        except ValueError:
            epsg = 4326
    tiffio.write(path, arr, tuple(float(v) for v in transform), epsg=epsg)


def to_db(array: np.ndarray, already_db: bool = True) -> np.ndarray:
    """Return Sigma0 in dB.

    Zenodo Part I ships Sigma0 already in dB. If a scene arrives in linear
    power (all values positive, median well above 1) it is converted here so the
    -22 dB threshold baseline stays meaningful either way.
    """
    a = np.asarray(array, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return a
    if already_db and float(np.nanmin(finite)) < 0.0:
        return a
    if float(np.nanmedian(finite)) > 1.5:  # looks like DN or linear power
        a = np.where(a > 0, a, np.nan)
        return 10.0 * np.log10(a / max(float(np.nanmax(a)), 1e-6))
    a = np.where(a > 0, a, np.nan)
    return 10.0 * np.log10(a)


def stretch_to_uint8(array: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Percentile stretch for map display. Never used for detection maths."""
    a = np.asarray(array, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = (np.clip(a, lo, hi) - lo) / (hi - lo)
    out = np.nan_to_num(out, nan=0.0)
    return (out * 255.0).astype(np.uint8)
