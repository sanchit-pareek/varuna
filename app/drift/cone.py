"""Uncertainty cones and origin zones built from an ensemble run.

The spec is blunt about this: always show uncertainty, never a single 400 m pin.
So the drift answer is a swept envelope of the whole ensemble over time, plus
the specific envelope at the chosen origin hour, buffered outward.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from ..geo.geometry import buffer_ring_km, hull_ring, ring_area_km2
from .advection import EnsembleRun


def percentile_ring(lons: np.ndarray, lats: np.ndarray, pct: float = 90.0) -> List[Tuple[float, float]]:
    """Hull of the particles inside the `pct` distance percentile of the median.

    Trimming the tail before hulling stops one runaway particle from inflating
    the whole envelope, which is what makes a 50 percent and a 90 percent ring
    actually differ.
    """
    from ..geo.crs import haversine_km

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.size < 3:
        return hull_ring(lons, lats)
    clon, clat = float(np.median(lons)), float(np.median(lats))
    d = haversine_km(clat, clon, lats, lons)
    cut = np.percentile(d, pct)
    keep = d <= cut + 1e-12
    if keep.sum() < 3:
        keep = np.ones_like(d, dtype=bool)
    return hull_ring(lons[keep], lats[keep])


def swept_cone(run: EnsembleRun, step: int = 1, pct: float = 90.0,
               start: int = 0, end: Optional[int] = None) -> List[Tuple[float, float]]:
    """Hull of every particle position between two indices: the drift cone."""
    end = run.n_steps - 1 if end is None else int(end)
    lo = run.lon[start:end + 1:step].ravel()
    la = run.lat[start:end + 1:step].ravel()
    return percentile_ring(lo, la, pct=pct)


def cone_feature(ring: Sequence[Tuple[float, float]], name: str, **props) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[float(x), float(y)] for x, y in ring]]} if ring else None,
        "properties": {"name": name, "area_km2": round(ring_area_km2(ring), 3), **props},
    }


def origin_zone(run: EnsembleRun, index: int, buffer_km: float = None,
                pct: float = 90.0) -> Dict[str, Any]:
    """The 90 percent envelope at the origin hour, buffered outward.

    Returns the ring, its centre, and the honest radius so the UI can label the
    zone instead of implying a point fix.
    """
    buffer_km = config.ORIGIN_BUFFER_KM if buffer_km is None else float(buffer_km)
    ring = percentile_ring(run.lon[index], run.lat[index], pct=pct)
    buffered = buffer_ring_km(ring, buffer_km) if ring else []
    lon_c = float(np.median(run.lon[index]))
    lat_c = float(np.median(run.lat[index]))
    return {
        "ring": buffered,
        "core_ring": ring,
        "lon": lon_c,
        "lat": lat_c,
        "t": run.times[index].isoformat(),
        "spread_km": float(run.spread_km[index]),
        "buffer_km": buffer_km,
        "area_km2": ring_area_km2(buffered),
        "percentile": pct,
    }


def envelopes_by_hour(run: EnsembleRun, pct: float = 90.0) -> List[Dict[str, Any]]:
    """Per hour envelope, so the UI time slider can walk the cone."""
    out: List[Dict[str, Any]] = []
    for i in range(run.n_steps):
        ring = percentile_ring(run.lon[i], run.lat[i], pct=pct)
        out.append({
            "index": i,
            "t": run.times[i].isoformat(),
            "ring": [[round(float(x), 6), round(float(y), 6)] for x, y in ring],
            "lon": round(float(run.mean_lon[i]), 6),
            "lat": round(float(run.mean_lat[i]), 6),
            "spread_km": round(float(run.spread_km[i]), 3),
        })
    return out


def threatened_bbox(run: EnsembleRun) -> Dict[str, float]:
    """Bounding box the forecast ensemble can reach, for the threat readout."""
    return {
        "west": float(np.min(run.lon)),
        "south": float(np.min(run.lat)),
        "east": float(np.max(run.lon)),
        "north": float(np.max(run.lat)),
    }
