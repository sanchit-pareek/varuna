"""Check each SAR detection against the cached optical chip.

The problem statement lists SAR *and* EO imagery as the data to detect from. A
fused detector is not honestly available here: there is no paired labelled
SAR/optical oil dataset, and inventing supervision for one would repeat the
mistake that produced the first quarantined checkpoint.

What is available, and is real analysis rather than decoration, is
corroboration. A mineral oil film flattens the surface, so in optical it usually
reads darker than the water around it away from sun glint, and brighter within
the glint pattern. Either way it differs from its surroundings. A SAR dark patch
that is actually a low-wind cell has no optical expression at all; one that is a
rig, a sandbar or a ship reads *brighter*. So comparing inside a polygon against
a ring around it separates three cases that matter, without ever claiming to
have detected oil optically.

The load-bearing caveat, carried on every verdict: Sentinel-2 did not observe
this water at the radar acquisition time. Offsets here run from 19 hours to 17
days. A slick moves and disperses in that time, so agreement is supporting
evidence and disagreement is not a refutation. The verdicts say so, and none of
them changes a detection: they are reported alongside, never subtracted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config

# How much darker, in stretched 8-bit levels, counts as a real difference
# rather than noise in the percentile stretch.
DARKER_LEVELS = 6.0
BRIGHTER_LEVELS = 10.0

# At or below this level the stretched chip is nodata, not dark water.
NODATA_LEVELS = 2.0

# Above this obscured fraction inside a polygon, the optical says nothing.
OBSCURED_FRACTION = 0.35

# A chip more than this far from the radar pass is reported, but its verdict is
# labelled weak: a slick will have moved well beyond its own footprint.
WEAK_AFTER_HOURS = 72.0

#: Below this cloud-free fraction of a polygon, the comparison is not reported
#: as a verdict at all. A delta over a handful of pixels looks like a
#: measurement in the API and is not one.
MIN_VALID_FRACTION = 0.30


def _optical_dir() -> Path:
    return Path(config.DATA_DIR) / "optical"


def load_optical(scene_id: str) -> Optional[Dict[str, Any]]:
    """Cached metadata for a scene, or None when there is no usable chip."""
    meta_path = _optical_dir() / ("%s.json" % scene_id)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("status") != "ok":
        return None
    png = _optical_dir() / ("%s.png" % scene_id)
    if not png.exists():
        return None
    meta["_png"] = png
    return meta


def _read_png(path: Path) -> Optional[np.ndarray]:
    """Luminance plane of the cached chip. Pillow is optional, so degrade."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32)
    except Exception:
        return None


def _ring_lonlat(ring: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(ring, dtype=np.float64)
    return a[:, 0], a[:, 1]


def _to_pixels(lons, lats, bounds, shape) -> Tuple[np.ndarray, np.ndarray]:
    """Leaflet bounds are [[south, west], [north, east]]; rows run north to south."""
    (south, west), (north, east) = bounds
    h, w = shape
    if east == west or north == south:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    x = (np.asarray(lons) - west) / (east - west) * (w - 1)
    y = (north - np.asarray(lats)) / (north - south) * (h - 1)
    return x, y


def _polygon_mask(ring, bounds, shape) -> np.ndarray:
    """Even-odd fill of one ring, in image coordinates. No shapely needed."""
    lons, lats = _ring_lonlat(ring)
    xs, ys = _to_pixels(lons, lats, bounds, shape)
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if xs.size < 3:
        return mask

    y0 = max(0, int(np.floor(ys.min())))
    y1 = min(h - 1, int(np.ceil(ys.max())))
    n = xs.size
    for row in range(y0, y1 + 1):
        crossings: List[float] = []
        for i in range(n):
            j = (i + 1) % n
            yi, yj = ys[i], ys[j]
            if (yi <= row < yj) or (yj <= row < yi):
                t = (row - yi) / (yj - yi)
                crossings.append(xs[i] + t * (xs[j] - xs[i]))
        crossings.sort()
        for k in range(0, len(crossings) - 1, 2):
            a = max(0, int(np.ceil(crossings[k])))
            b = min(w - 1, int(np.floor(crossings[k + 1])))
            if b >= a:
                mask[row, a:b + 1] = True
    return mask


def _dilate(mask: np.ndarray, size: int) -> np.ndarray:
    out = mask.copy()
    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def corroborate(polygons: Sequence[Dict[str, Any]], scene_id: str) -> Dict[str, Any]:
    """Per-polygon optical verdicts plus a summary. Never raises."""
    meta = load_optical(scene_id)
    if meta is None:
        return {"available": False,
                "reason": "no cached Sentinel-2 chip for this scene",
                "verdicts": []}

    grey = _read_png(meta["_png"])
    if grey is None:
        return {"available": False,
                "reason": "the optical chip could not be read (pillow missing?)",
                "verdicts": []}

    offset_h = float(meta.get("offset_hours") or 0.0)
    weak = abs(offset_h) > WEAK_AFTER_HOURS
    bounds = meta["bounds"]
    shape = grey.shape

    # Both extremes mean "no information", and the black end is the one that
    # bites. `cut_chip` fills anything outside the granule with zero, and a
    # Sentinel-2 granule often covers only part of a scene footprint: the Gulf
    # chip is 0 across the whole area the detections sit in. Treating only the
    # white end as obscured made every one of those polygons come back "neutral,
    # no optical difference" with a delta of exactly 0.0 -- a confident-looking
    # verdict derived from an empty image.
    obscured = (grey >= 250.0) | (grey <= NODATA_LEVELS)

    verdicts: List[Dict[str, Any]] = []
    for f in polygons:
        geom = (f or {}).get("geometry") or {}
        coords = geom.get("coordinates") or []
        if not coords:
            continue
        pid = ((f.get("properties") or {}).get("polygon_id"))
        inside = _polygon_mask(coords[0], bounds, shape)
        n_in = int(inside.sum())
        if n_in < 12:
            verdicts.append({"polygon_id": pid, "verdict": "too_small",
                             "status": "indeterminate",
                             "valid_pixel_fraction": 0.0,
                             "note": "polygon covers too few optical pixels to compare"})
            continue

        ring = _dilate(inside, 6) & ~_dilate(inside, 2)
        n_ring = int(ring.sum())
        if n_ring < 12:
            verdicts.append({"polygon_id": pid, "verdict": "no_surroundings",
                             "status": "indeterminate",
                             "valid_pixel_fraction": 0.0,
                             "note": "no clear water ring around this polygon"})
            continue

        # How much of this polygon is actually observable, after cloud and
        # nodata. A delta computed over a quarter of the pixels is a different
        # claim from one computed over all of them, and the fraction has to
        # travel with the verdict.
        clear = inside & ~obscured
        frac_obscured = float(obscured[inside].mean())
        valid_fraction = float(clear.sum()) / float(n_in) if n_in else 0.0
        if valid_fraction < MIN_VALID_FRACTION:
            verdicts.append({"polygon_id": pid, "verdict": "obscured",
                             "status": "indeterminate",
                             "obscured_fraction": round(frac_obscured, 3),
                             "valid_pixel_fraction": round(valid_fraction, 3),
                             "optical_pixels": n_in,
                             "clear_optical_pixels": int(clear.sum()),
                             "time_delta_hours": round(offset_h, 2),
                             "time_is_exact": False,
                             "note": "only %.0f percent of this polygon is cloud-free optical data"
                                     % (valid_fraction * 100.0)})
            continue

        in_mean = float(np.mean(grey[clear]))
        ring_clear = ring & ~obscured
        ring_mean = float(np.mean(grey[ring_clear])) if ring_clear.any() else in_mean
        delta = in_mean - ring_mean
        delta_sd = float(np.std(grey[clear])) if int(clear.sum()) > 1 else 0.0

        if delta <= -DARKER_LEVELS:
            verdict, status, note = ("consistent", "corroborated",
                                     "darker than the surrounding water, as a surface film reads")
        elif delta >= BRIGHTER_LEVELS:
            verdict, status, note = ("inconsistent", "contradicted",
                                     "brighter than its surroundings, which a slick is not")
        else:
            verdict, status, note = ("neutral", "indeterminate",
                                     "no optical difference from the surrounding water")

        verdicts.append({
            "polygon_id": pid,
            "verdict": verdict,
            "status": status,
            "note": note,
            "levels_vs_surroundings": round(delta, 1),
            "levels_stddev_inside": round(delta_sd, 2),
            "obscured_fraction": round(frac_obscured, 3),
            "valid_pixel_fraction": round(valid_fraction, 3),
            "optical_pixels": n_in,
            "clear_optical_pixels": int(clear.sum()),
            "time_delta_hours": round(offset_h, 2),
            "time_is_exact": False,
        })

    counts: Dict[str, int] = {}
    for v in verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    statuses: Dict[str, int] = {}
    for v in verdicts:
        statuses[v["status"]] = statuses.get(v["status"], 0) + 1

    # A scene is only called corroborated when the verdict actually survived
    # both the time offset and the cloud fraction. Naming a status that the
    # verdicts do not support is the failure this whole module guards against.
    if weak or statuses.get("corroborated", 0) == 0:
        status = "indeterminate"
    elif statuses.get("corroborated", 0) > statuses.get("contradicted", 0):
        status = "corroborated"
    elif statuses.get("contradicted", 0) > 0:
        status = "contradicted"
    else:
        status = "indeterminate"

    return {
        "available": True,
        "status": status,
        "status_meaning": {
            "corroborated": "A surface film was visible where the radar saw one.",
            "contradicted": "The optical scene does not support a surface film there.",
            "indeterminate": "The optical chip cannot decide this. No claim is made.",
        },
        "source": meta.get("collection"),
        "item_id": meta.get("item_id"),
        "acquired": meta.get("acquired"),
        "offset_hours": round(offset_h, 2),
        "time_delta": {
            "hours": round(offset_h, 2),
            "sar_acquisition": meta.get("sar_acquisition"),
            "optical_acquisition": meta.get("acquired"),
            "is_exact": False,
            "note": "Sentinel-2 did not observe this water at the radar time. Any slick "
                    "will have moved and spread in this interval, so a negative optical "
                    "result is not evidence of absence.",
        },
        "offset_label": meta.get("offset_label"),
        "cloud_percent": meta.get("cloud_percent"),
        "weak": weak,
        "counts": counts,
        "status_counts": statuses,
        "verdicts": verdicts,
        "caveat": (
            "Optical corroboration only. Sentinel-2 did not observe this water at the "
            "radar acquisition time (%s), so a slick will have moved and spread in "
            "between. Agreement supports a detection; disagreement does not overturn "
            "one. No detection was added, removed or reweighted by this check."
            % (meta.get("offset_label") or "offset unknown")),
    }
