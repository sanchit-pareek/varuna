"""Optional coastline check for the forecast cone.

The spec is explicit that the land mask is optional and that the step is skipped
when it is missing. So this module never fails: with no coastline file present
it reports `available: False` and the pipeline moves on.

Supply one by dropping a GeoJSON of land or coastline polygons at
`data/land/coastline.geojson`. Natural Earth 1:10m land is a good free choice
(public domain). Only the polygons that overlap the forecast bounding box are
tested, so a global file is fine.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from ..geo.geometry import point_in_ring

_CACHE: Dict[str, Any] = {"loaded": False, "rings": None, "water": None,
                          "path": None}


def _candidate_paths() -> List[Path]:
    base = Path(config.DATA_DIR) / "land"
    return [base / "coastline.geojson", base / "land.geojson"] + sorted(base.glob("*.geojson"))


def load_rings() -> Optional[List[List[Tuple[float, float]]]]:
    """Flatten every polygon in the land file into a list of exterior rings."""
    if _CACHE["loaded"]:
        return _CACHE["rings"]
    _CACHE["loaded"] = True

    for p in _candidate_paths():
        if not p.exists():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rings: List[List[Tuple[float, float]]] = []
        water: List[List[Tuple[float, float]]] = []
        feats = doc.get("features", [doc]) if isinstance(doc, dict) else []
        for f in feats:
            geom = (f or {}).get("geometry", f)
            if not geom:
                continue
            # Features may be tagged `kind: water`. Natural Earth models inland
            # seas as holes in the surrounding landmass, and the builder emits
            # those holes as water rather than dropping them -- without that the
            # Caspian is part of Eurasia and masking would erase a whole scene.
            props = (f or {}).get("properties") or {}
            target = water if props.get("kind") == "water" else rings
            gtype = geom.get("type")
            coords = geom.get("coordinates") or []
            if gtype == "Polygon":
                if coords:
                    target.append([(float(x), float(y)) for x, y in coords[0]])
            elif gtype == "MultiPolygon":
                for poly in coords:
                    if poly:
                        target.append([(float(x), float(y)) for x, y in poly[0]])
        if rings:
            _CACHE["rings"] = rings
            _CACHE["water"] = water
            _CACHE["path"] = str(p)
            return rings
    _CACHE["rings"] = None
    _CACHE["water"] = None
    return None


def load_water_rings() -> List[List[Tuple[float, float]]]:
    """Rings that are water even though they sit inside a land polygon."""
    load_rings()
    return _CACHE.get("water") or []


def is_land(lon: float, lat: float) -> bool:
    """True when a point is on land, honouring inland seas.

    Used to keep detections off the shore. A dark patch on a hillside is radar
    shadow, not a slick, and before this existed the detector was free to report
    one: a scene over the Santa Barbara mountains produced polygons across the
    ridge line.
    """
    rings = load_rings()
    if not rings:
        return False
    if not any(point_in_ring(lon, lat, r) for r in rings):
        return False
    return not any(point_in_ring(lon, lat, r) for r in load_water_rings())


def _segments_cross(a, b, c, d) -> bool:
    """Do segments ab and cd properly intersect?"""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _rings_intersect(cone: Sequence[Tuple[float, float]],
                     land: Sequence[Tuple[float, float]]) -> bool:
    """True polygon overlap, not a bounding box near miss.

    Three cases have to be covered or the flag is wrong in a way that matters:
    the cone sits inside the landmass, the landmass sits inside the cone, or
    their boundaries cross. Testing land vertices against the cone's bounding
    box, which is what an earlier version did, reports a hit whenever a coast is
    merely nearby, and a false coastal impact warning is worse than none.
    """
    if len(cone) < 3 or len(land) < 3:
        return False

    lon_c = [p[0] for p in cone]
    lat_c = [p[1] for p in cone]
    lon_l = [p[0] for p in land]
    lat_l = [p[1] for p in land]
    if (max(lon_l) < min(lon_c) or min(lon_l) > max(lon_c)
            or max(lat_l) < min(lat_c) or min(lat_l) > max(lat_c)):
        return False

    if any(point_in_ring(x, y, land) for x, y in cone):
        return True
    if any(point_in_ring(x, y, cone) for x, y in land):
        return True

    for i in range(len(cone) - 1):
        a, b = cone[i], cone[i + 1]
        for j in range(len(land) - 1):
            if _segments_cross(a, b, land[j], land[j + 1]):
                return True
    return False


def check(cone_ring: Sequence[Tuple[float, float]],
          bbox: Dict[str, float]) -> Dict[str, Any]:
    """Does the forecast cone reach land? Returns a skip result if unavailable."""
    rings = load_rings()
    if not rings:
        return {
            "available": False,
            "coast_flag": None,
            "note": "No land mask present. Run scripts/build_land_mask.py, or drop a "
                    "GeoJSON at data/land/coastline.geojson, to enable the coast "
                    "impact flag. This step is optional by design.",
        }

    cone = [(float(x), float(y)) for x, y in cone_ring]
    if cone and cone[0] != cone[-1]:
        cone = cone + [cone[0]]

    # Only true land counts as coastline here. The water rings exist to punch
    # inland seas back out of the landmass, and treating them as shore would
    # raise a coast-impact flag in the middle of open water.
    hits = [i for i, ring in enumerate(rings) if _rings_intersect(cone, ring)]
    return {
        "available": True,
        "source": _CACHE["path"],
        "polygons_checked": len(rings),
        "coast_flag": bool(hits),
        "polygons_intersected": len(hits),
        "note": ("Forecast cone overlaps land within the forecast horizon." if hits
                 else "Forecast cone stays offshore over the forecast horizon."),
    }


def _ring_contains(lon: np.ndarray, lat: np.ndarray,
                   ring: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Vectorised even-odd ray cast for a whole grid against one ring."""
    xs = np.asarray([p[0] for p in ring], dtype=np.float64)
    ys = np.asarray([p[1] for p in ring], dtype=np.float64)
    inside = np.zeros(lon.shape, dtype=bool)
    x1, y1 = xs[-1], ys[-1]
    for x2, y2 in zip(xs, ys):
        if y1 != y2:
            straddles = (lat >= np.minimum(y1, y2)) & (lat < np.maximum(y1, y2))
            if straddles.any():
                cross_x = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
                inside ^= straddles & (lon < cross_x)
        x1, y1 = x2, y2
    return inside


def mask_for_raster(sar, step: int = 8) -> Optional[np.ndarray]:
    """Boolean land mask on a scene's own pixel grid, or None with no coastline.

    A radar chip that contains coast contains dark pixels that are not slicks:
    radar shadow behind a ridge, a sheltered harbour basin, wet ground. Dropping
    polygons whose centroid is ashore, which is what the pipeline did on its
    own, cannot catch a single polygon that straddles the shoreline and takes
    the whole coastal strip with it -- on the Santa Barbara chip that was worth
    136 km2 of reported oil against a real 0.4.

    Land is therefore removed before detection rather than after. The mask is
    evaluated on a decimated grid and expanded back, because a shoreline placed
    to within `step` pixels is far finer than the coastline data itself.
    """
    rings = load_rings()
    if not rings:
        return None
    h, w = sar.array.shape[-2:]
    step = max(1, int(step))
    rows = np.arange(0, h, step, dtype=np.float64)
    cols = np.arange(0, w, step, dtype=np.float64)
    cc, rr = np.meshgrid(cols + 0.5, rows + 0.5)

    a, b, c, d, e, f = sar.transform
    x = a * cc + b * rr + c
    y = d * cc + e * rr + f
    from ..geo import raster as raster_mod

    try:
        lon, lat = raster_mod.to_wgs84(x, y, sar.crs)
    except Exception:
        return None

    land_hit = np.zeros(lon.shape, dtype=bool)
    for ring in rings:
        if len(ring) >= 3:
            land_hit |= _ring_contains(lon, lat, ring)
    for ring in load_water_rings():          # lakes and inlets punched back out
        if len(ring) >= 3:
            land_hit &= ~_ring_contains(lon, lat, ring)
    if not land_hit.any():
        return None

    full = np.repeat(np.repeat(land_hit, step, axis=0), step, axis=1)
    return full[:h, :w]
