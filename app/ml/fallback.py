"""Sigma0 dark-patch baseline detector.

This is the published -22 dB threshold baseline, and it exists for exactly one
reason the spec states plainly: the pipeline must still run clauses (a), (b) and
(c) when the U-Net checkpoint is missing or CUDA is unavailable. It is a
baseline to measure the model against, not the thing being pitched.

Method, all of it standard and all of it justified by the physics:

  1. Threshold Sigma0 VV at OIL_DB_THRESHOLD, with an adaptive fallback to a
     low percentile when the scene calibration differs from the assumption.
  2. Clean speckle with a morphological open then close.
  3. Split the surviving dark patches into look-alike and mineral oil using two
     discriminators from the oil-spill literature: contrast against local sea
     backscatter, and boundary regularity. Low grazing wind areas, biogenic
     films and rain cells are darker than sea but with weak contrast and fuzzy,
     rounded outlines. Mineral oil slicks are darker still and more elongated
     with sharper edges.

The confidence returned is a calibrated-ish pseudo probability derived from
contrast, so downstream code has the same interface as the network path.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .. import config
from ..geo import geometry

CONTRAST_OIL_DB = 3.5      # dB below local sea for a patch to be called oil
CONTRAST_MIN_DB = 2.0      # below this it is speckle texture, not a patch
COMPACTNESS_LOOKALIKE = 0.62  # near-circular blobs are usually not slicks
RELATIVE_DB = 3.0          # a patch must sit at least this far below local sea
MULTILOOK = 5              # boxcar window used to suppress speckle


def multilook(img: np.ndarray, size: int = MULTILOOK) -> np.ndarray:
    """Boxcar multi-look in linear power, the standard speckle reduction step.

    Averaging must happen in power, not in dB, or the mean is biased low. Single
    look SAR intensity has unit coefficient of variation, so thresholding raw
    dB without this produces a mask made of noise.
    """
    a = np.asarray(img, dtype=np.float32)
    if size < 2:
        return a
    valid = np.isfinite(a)
    lin = np.where(valid, np.power(10.0, np.clip(a, -60.0, 30.0) / 10.0), 0.0)
    w = valid.astype(np.float32)
    try:
        from scipy import ndimage as ndi

        num = ndi.uniform_filter(lin, size=size, mode="nearest")
        den = ndi.uniform_filter(w, size=size, mode="nearest")
    except Exception:  # pragma: no cover - only on installs without scipy
        num = _box_numpy(lin, size)
        den = _box_numpy(w, size)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(out > 0, out, np.nan))
    return db.astype(np.float32)


def _box_numpy(a: np.ndarray, size: int) -> np.ndarray:
    r = size // 2
    pad = np.pad(a, r, mode="edge")
    cs = pad.cumsum(axis=0).cumsum(axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = a.shape
    total = (cs[size:size + h, size:size + w] - cs[0:h, size:size + w]
             - cs[size:size + h, 0:w] + cs[0:h, 0:w])
    return total / float(size * size)


def detect(
    sigma0_db: np.ndarray,
    threshold_db: float = None,
    min_pixels: int = 200,
    adaptive: bool = True,
    relative_db: float = RELATIVE_DB,
    multilook_size: int = MULTILOOK,
    exclude: np.ndarray = None,
) -> Dict[str, Any]:
    """Return a 3-class mask and a per-pixel oil pseudo probability.

    Classes follow the project convention: 0 sea, 1 look-alike, 2 mineral oil.

    Thresholding rule. Oil is a contrast phenomenon: it sits some dB below
    whatever the local sea happens to be, and "whatever the local sea happens to
    be" moves several dB with wind speed, incidence angle and product
    calibration. Across the shipped scenes open water spans -17.7 dB to
    -23.4 dB. So the threshold applied is

        local sea median - relative_db

    with the published -22 dB constant kept as a reported reference rather than
    a bound. The previous rule took min() of the two, which meant the absolute
    constant won on every bright scene: on the Arabian Sea chip, where the water
    sits at -17.7 dB, a -22 dB cut selected 1.6 percent of the scene and the
    detector returned nothing. Both numbers are still reported, so the choice is
    visible. Set BASELINE_RELATIVE_ONLY false to restore the strict published
    behaviour.
    """
    threshold_db = config.OIL_DB_THRESHOLD if threshold_db is None else float(threshold_db)
    raw = np.asarray(sigma0_db, dtype=np.float32)
    img = multilook(raw, multilook_size) if multilook_size > 1 else raw
    valid = np.isfinite(img)
    if exclude is not None:
        # Land is excluded from the water-level estimate, not just from the
        # result. A chip that is a quarter coastline reports its sea 1.6 dB
        # brighter than the water actually is, and every relative threshold in
        # this function is computed from that number.
        valid = valid & ~np.asarray(exclude, dtype=bool)
    if not valid.any():
        return _empty(raw.shape, threshold_db)

    vals = img[valid]
    sea_level = float(np.median(vals))
    relative_cut = sea_level - float(relative_db)
    if not adaptive:
        used_threshold = threshold_db
    elif config.BASELINE_RELATIVE_ONLY:
        used_threshold = relative_cut
    else:
        used_threshold = min(threshold_db, relative_cut)
    dark = valid & (img < used_threshold)
    frac = float(dark.sum()) / float(valid.sum())

    # Last resort guard: if that still selects almost everything the scene is
    # unusual, so fall back to a low percentile and say so.
    if adaptive and frac > 0.45:
        used_threshold = float(np.percentile(vals, 4.0))
        dark = valid & (img < used_threshold)
        frac = float(dark.sum()) / float(valid.sum())

    dark = geometry.binary_closing(dark, size=3)
    labels, n = geometry.label_components(dark)

    mask = np.zeros(img.shape, dtype=np.uint8)
    prob = np.zeros(img.shape, dtype=np.float32)
    meta = {
        "threshold_db": round(float(used_threshold), 2),
        "absolute_threshold_db": round(float(threshold_db), 2),
        "relative_threshold_db": round(float(relative_cut), 2),
        "sea_level_db": round(float(sea_level), 2),
        "dark_fraction": round(float(frac), 5),
        "multilook": int(multilook_size),
        "method": "sigma0_threshold_baseline",
    }
    if n == 0:
        return dict(meta, mask=mask, prob=prob, components=0)

    boxes = geometry.component_slices(labels, n)
    for lab in range(1, n + 1):
        box = boxes[lab - 1] if lab - 1 < len(boxes) else None
        if box is None:
            continue
        sub = labels[box] == lab
        count = int(sub.sum())
        if count < min_pixels:
            continue
        patch_db = float(np.mean(img[box][sub]))
        local_sea = _local_sea_level(img, box, sub, valid, fallback=sea_level)
        contrast = local_sea - patch_db
        if contrast < CONTRAST_MIN_DB:
            continue
        shape = _shape_stats(sub)
        is_oil = (contrast >= CONTRAST_OIL_DB) and (shape["compactness"] < COMPACTNESS_LOOKALIKE)
        mask_view = mask[box]
        mask_view[sub] = 2 if is_oil else 1
        p = 1.0 / (1.0 + np.exp(-(contrast - CONTRAST_OIL_DB)))
        prob_view = prob[box]
        prob_view[sub] = float(np.clip(p, 0.02, 0.98))

    return dict(meta, mask=mask, prob=prob, components=int(n))


def _empty(shape, threshold_db) -> Dict[str, Any]:
    return {
        "mask": np.zeros(shape, dtype=np.uint8),
        "prob": np.zeros(shape, dtype=np.float32),
        "threshold_db": float(threshold_db),
        "absolute_threshold_db": float(threshold_db),
        "relative_threshold_db": float("nan"),
        "sea_level_db": float("nan"),
        "dark_fraction": 0.0,
        "multilook": MULTILOOK,
        "method": "sigma0_threshold_baseline",
        "components": 0,
    }


def _local_sea_level(img: np.ndarray, box, sub: np.ndarray, valid: np.ndarray,
                     fallback: float, pad: int = 24) -> float:
    """Median backscatter of the annulus around a patch, not the whole scene."""
    rs, cs = box
    r0, r1 = max(0, rs.start - pad), min(img.shape[0], rs.stop + pad)
    c0, c1 = max(0, cs.start - pad), min(img.shape[1], cs.stop + pad)
    win_valid = valid[r0:r1, c0:c1].copy()
    win_valid[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0] &= ~sub
    vals = img[r0:r1, c0:c1][win_valid]
    if vals.size < 50:
        return fallback
    return float(np.median(vals))


def _shape_stats(comp: np.ndarray) -> Dict[str, float]:
    """Compactness of a component from its area and boundary length.

    The perimeter is counted from 4-connected boundary edges rather than traced
    with the Moore walk, because this runs once per dark patch and a busy scene
    has thousands of them. The edge count overestimates a diagonal boundary by
    the usual factor, so it is scaled by pi/4, which is the standard correction
    that makes a digital disc score close to 1.
    """
    area = float(comp.sum())
    if area <= 0:
        return {"compactness": 1.0, "area_px": area}
    edges = 0
    edges += int(np.count_nonzero(comp[:, :-1] != comp[:, 1:]))
    edges += int(np.count_nonzero(comp[:-1, :] != comp[1:, :]))
    edges += int(np.count_nonzero(comp[:, 0])) + int(np.count_nonzero(comp[:, -1]))
    edges += int(np.count_nonzero(comp[0, :])) + int(np.count_nonzero(comp[-1, :]))
    perim = edges * (np.pi / 4.0)
    if perim <= 0:
        return {"compactness": 1.0, "area_px": area}
    c = 4.0 * np.pi * area / (perim ** 2)
    return {"compactness": float(min(max(c, 0.0), 1.0)), "area_px": area}


def iou(pred: np.ndarray, truth: np.ndarray, klass: int) -> float:
    """Class IoU, used by the metrics card that compares model to baseline."""
    p = np.asarray(pred) == klass
    t = np.asarray(truth) == klass
    union = np.logical_or(p, t).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(p, t).sum()) / float(union)
