"""Ablation study: what does each stage of the pipeline actually buy?

An attribution system that ranks the right vessel for the right reasons is not
the same as one that ranks the right vessel for the wrong reasons. This module
answers that by re-ranking the same candidate set under progressively reduced
evidence and measuring the damage.

The ladder runs from the naive method an analyst would reach for first to the
full system:

    A  ais_nearest          nearest AIS track to the slick, nothing else
    B  +slick_geometry      distance decay to the slick centroid
    C  +slick_orientation  window-mean course against the drift bearing
    D  +drift_origin        drift-derived origin point and zone radius
    E  +ensemble_uncertainty  50/90 envelopes and the release window
    F  +ais_gap             reporting-gap and behaviour detectors
    G  full                 the production scorer

The reported statistic is agreement with G, measured as top-1 identity, top-3
overlap, and Spearman rank correlation. A stage whose removal changes nothing
is not being wasted, but it is also not carrying the result, and an analyst
describing this case should know which.

Two properties are deliberate. Variants never see a stage's output unless the
variant includes that stage, so there is no leakage from the full system. And
when a stage has not been computed in the current case, the variant that
depends on it is reported as `not_available` rather than being silently
skipped or quietly approximated.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .ais.interpolate import Track
from .geo.crs import angle_diff_deg, bearing_deg, haversine_km

#: The ladder, in order. `needs` names the run outputs a variant requires.
ABLATION_LADDER: Tuple[Dict[str, Any], ...] = (
    {"key": "ais_nearest", "step": "A", "label": "AIS nearest to slick",
     "includes": [], "excludes": ["slick geometry", "drift", "uncertainty", "behaviour"],
     "needs": [],
     "question": "Would simply finding the closest vessel have been enough?",
     "description": "Rank candidates by raw distance to the slick centroid. No decay, "
                    "no time, no type, no drift. This is the naive method."},
    {"key": "slick_geometry", "step": "B", "label": "+ slick geometry decay",
     "includes": ["slick geometry"], "excludes": ["orientation", "drift", "uncertainty", "behaviour"],
     "needs": [],
     "question": "How much does distance decay alone buy over raw distance?",
     "description": "exp(-d / L) to the slick centroid. Continuous, not a hard cut."},
    {"key": "slick_orientation", "step": "C", "label": "+ slick orientation",
     "includes": ["slick geometry", "slick orientation"],
     "excludes": ["drift", "uncertainty", "behaviour"],
     "needs": [],
     "question": "Does a coarse whole-window heading test add anything?",
     "description": "Window-mean course against the slick-to-origin bearing. Coarser "
                    "than the closest-approach trajectory term used by the full model."},
    {"key": "drift_origin", "step": "D", "label": "+ drift-derived origin",
     "includes": ["slick geometry", "slick orientation", "drift"],
     "excludes": ["uncertainty", "behaviour"],
     "needs": ["origin"],
     "question": "Is modelling transport the single biggest contributor?",
     "description": "The origin estimate replaces the slick as the spatial reference, "
                    "and the ensemble spread sets the proximity length scale."},
    {"key": "ensemble_uncertainty", "step": "E", "label": "+ ensemble uncertainty",
     "includes": ["slick geometry", "slick orientation", "drift", "uncertainty"],
     "excludes": ["behaviour"],
     "needs": ["origin", "envelopes"],
     "question": "Do the 50/90 envelopes and the release window change the order?",
     "description": "Adds 50 and 90 percent envelope containment and interval-aware "
                    "temporal scoring. Reported, and only partly folded into rank."},
    {"key": "ais_gap", "step": "F", "label": "+ AIS gap and behaviour",
     "includes": ["slick geometry", "slick orientation", "drift", "uncertainty", "behaviour"],
     "excludes": [],
     "needs": ["origin", "envelopes"],
     "question": "Is the ranking a reporting-gap detector in disguise?",
     "description": "Full evidence minus nothing; the gap and behaviour detectors are "
                    "the last stage added."},
    {"key": "full", "step": "G", "label": "Full system",
     "includes": ["slick geometry", "slick orientation", "drift", "uncertainty", "behaviour"],
     "excludes": [],
     "needs": ["origin", "envelopes"],
     "question": "Reference ranking.",
     "description": "The production scorer, unchanged. Every other variant is compared "
                    "against this."},
)

NOT_AVAILABLE = "not_available"


# ---------------------------------------------------------------------------
# Ranking utilities
# ---------------------------------------------------------------------------

def _order(scores: Dict[int, float]) -> List[int]:
    """Deterministic order: score desc, then MMSI asc."""
    return [m for m, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def spearman(a: Sequence[int], b: Sequence[int]) -> Optional[float]:
    """Spearman rank correlation of two orderings over a shared candidate set.

    Returns None when it is undefined (fewer than two shared candidates, or no
    variation), so a caller cannot render a number that means nothing.
    """
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    rank_a = {m: i for i, m in enumerate(a)}
    rank_b = {m: i for i, m in enumerate(b)}
    n = len(common)
    d2 = sum((rank_a[m] - rank_b[m]) ** 2 for m in common)
    denom = n * (n * n - 1)
    if denom <= 0:
        return None
    return round(1.0 - (6.0 * d2) / denom, 4)


def _agreement(order: Sequence[int], reference: Sequence[int],
               top: int = 3) -> Dict[str, Any]:
    ref_top = list(reference[:top])
    this_top = list(order[:top])
    shared = set(ref_top) & set(this_top)
    union = set(ref_top) | set(this_top)
    return {
        "top1_match": bool(this_top and ref_top and this_top[0] == ref_top[0]),
        "top3_overlap": len(shared),
        "top3_union": len(union),
        "top3_jaccard": round(len(shared) / len(union), 4) if union else None,
    }


# ---------------------------------------------------------------------------
# Reduced scorers
# ---------------------------------------------------------------------------

def _min_distance_to(tr: Track, lon: float, lat: float) -> Optional[float]:
    if not tr.n:
        return None
    d = np.asarray(haversine_km(lat, lon, np.asarray(tr.lat, dtype=np.float64),
                                 np.asarray(tr.lon, dtype=np.float64)), dtype=np.float64)
    return float(d.min())


def _window_mean_course(tr: Track) -> Optional[float]:
    """Circular mean of course over the whole track.

    The naive orientation test. It is intentionally coarse: it ignores when in
    the window the vessel was pointing that way, which is the whole point of
    the full model's closest-approach term.
    """
    if not tr.n:
        return None
    cog = np.asarray(tr.cog, dtype=np.float64)
    rad = np.radians(cog)
    mean = math.degrees(math.atan2(float(np.mean(np.sin(rad))), float(np.mean(np.cos(rad)))))
    return mean % 360.0


def _score_variants(tracks: Dict[int, Track], closest: Dict[int, Tuple[float, int]],
                    slick_lon: float, slick_lat: float,
                    origin_lon: Optional[float], origin_lat: Optional[float],
                    zone_radius_km: Optional[float],
                    slick_bearing_deg: Optional[float],
                    envelope_match: Optional[Dict[int, Dict[str, Any]]],
                    window_score: Optional[Dict[int, float]],
                    full_scores: Dict[int, float],
                    full_components: Dict[int, Dict[str, float]],
                    ) -> Dict[str, Dict[int, float]]:
    """Every rung of the ladder, computed from the same candidate set."""
    out: Dict[str, Dict[int, float]] = {}

    # A. Raw distance to the slick. Rank by inverse distance, no decay.
    a: Dict[int, float] = {}
    for mmsi, tr in tracks.items():
        d = _min_distance_to(tr, slick_lon, slick_lat)
        if d is not None and math.isfinite(d):
            a[mmsi] = -d
    out["ais_nearest"] = a

    # B. Exponential decay to the slick centroid.
    out["slick_geometry"] = {
        m: math.exp(-max(0.0, -v) / 3.0) for m, v in a.items()
    }

    # C. Add a coarse whole-window heading test against the slick->origin bearing.
    c: Dict[int, float] = {}
    for mmsi in out["slick_geometry"]:
        tr = tracks[mmsi]
        mean_course = _window_mean_course(tr)
        if mean_course is None or slick_bearing_deg is None:
            c[mmsi] = out["slick_geometry"][mmsi]
            continue
        diff = abs(angle_diff_deg(mean_course, slick_bearing_deg))
        c[mmsi] = 0.5 * out["slick_geometry"][mmsi] + 0.5 * math.exp(-0.5 * (diff / 90.0) ** 2)
    out["slick_orientation"] = c

    # D. The drift-derived origin replaces the slick as the spatial reference,
    #    with the ensemble spread setting the length scale. Type prior only.
    if origin_lon is None or origin_lat is None:
        out["drift_origin"] = {}
    else:
        d_scores: Dict[int, float] = {}
        for mmsi, tr in tracks.items():
            dist = _min_distance_to(tr, origin_lon, origin_lat)
            if dist is None or not math.isfinite(dist):
                continue
            scale = max(3.0, float(zone_radius_km or 0.0))
            d_scores[mmsi] = math.exp(-dist / scale)
        out["drift_origin"] = {
            mmsi: 0.7 * prox + 0.3 * float(full_components.get(mmsi, {}).get("type", 0.0))
            for mmsi, prox in d_scores.items()
        }

    # E. Add envelope containment and interval-aware temporal compatibility.
    e: Dict[int, float] = {}
    for mmsi, base in out.get("drift_origin", {}).items():
        match = (envelope_match or {}).get(mmsi) or {}
        contain = 1.0 if match.get("inside_90") else 0.0
        core = 1.0 if match.get("inside_50") else 0.0
        tscore = float((window_score or {}).get(mmsi, 0.0))
        e[mmsi] = 0.5 * base + 0.2 * contain + 0.1 * core + 0.2 * tscore
    out["ensemble_uncertainty"] = e

    # F. The full evidence score, which is the production number.
    out["ais_gap"] = dict(full_scores)
    out["full"] = dict(full_scores)
    return out


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------

def run_ablation(suspects: Sequence[Dict[str, Any]],
                 tracks: Optional[Dict[int, Track]] = None,
                 slick_lon: Optional[float] = None,
                 slick_lat: Optional[float] = None,
                 origin_lon: Optional[float] = None,
                 origin_lat: Optional[float] = None,
                 zone_radius_km: Optional[float] = None,
                 slick_bearing_deg: Optional[float] = None,
                 n_available: Optional[int] = None) -> Dict[str, Any]:
    """Run the ladder and report agreement with the full system.

    `suspects` is the production ranking, i.e. the output of
    `ais_score.rank_suspects(...).to_dict()`. `tracks` and the geometry
    arguments are the raw inputs the reduced scorers need; when they are absent
    the affected rungs report `not_available` rather than being guessed.
    """
    scores = {int(s["mmsi"]): float(s["score"]) for s in suspects or ()}
    components = {int(s["mmsi"]): dict(s.get("components") or {}) for s in suspects or ()}
    envelope_match = {int(s["mmsi"]): dict(s.get("region") or {}) for s in suspects or ()}
    window_score = {}
    for s in suspects or ():
        w = s.get("release_window") or {}
        if w.get("applicable") and w.get("score") is not None:
            window_score[int(s["mmsi"])] = float(w["score"])

    have_tracks = bool(tracks)
    if have_tracks and slick_lon is None and origin_lon is not None:
        slick_lon, slick_lat = origin_lon, origin_lat
    have_tracks = have_tracks and slick_lon is not None and slick_lat is not None
    variants = _score_variants(
        tracks or {}, {}, slick_lon, slick_lat, origin_lon, origin_lat,
        zone_radius_km, slick_bearing_deg, envelope_match, window_score,
        scores, components) if have_tracks else {}

    if not scores:
        return {
            "available": False,
            "reason": "no ranked candidates were supplied",
            "candidates": 0,
            "ladder": [],
            "statement": ("An ablation needs at least one ranked candidate. This case "
                          "produced none, which is itself the finding: there is "
                          "nothing for the ladder to re-rank."),
        }

    reference = _order(scores)
    rows: List[Dict[str, Any]] = []
    for rung in ABLATION_LADDER:
        row: Dict[str, Any] = {
            "step": rung["step"], "key": rung["key"], "label": rung["label"],
            "includes": list(rung["includes"]), "excludes": list(rung["excludes"]),
            "question": rung["question"], "description": rung["description"],
        }
        if rung["key"] not in variants or not variants[rung["key"]]:
            row["status"] = NOT_AVAILABLE
            row["reason"] = ("the raw tracks or slick geometry for this case were not "
                             "supplied to the ablation")
            rows.append(row)
            continue
        scores_k = variants[rung["key"]]
        order = _order(scores_k)
        row.update({
            "status": "computed",
            "top1_mmsi": order[0] if order else None,
            "top1_name": _name_of(suspects, order[0] if order else None),
            "order": list(order),
            "n_candidates": len(order),
            **_agreement(order, reference),
        })
        if rung["key"] != "full":
            row["spearman_vs_full"] = spearman(order, reference)
            deltas = [abs(scores_k[m] - scores[m]) for m in order if m in scores]
            row["mean_abs_score_delta"] = round(sum(deltas) / len(deltas), 3) if deltas else None
        else:
            row["spearman_vs_full"] = 1.0
            row["mean_abs_score_delta"] = 0.0
        rows.append(row)

    return {
        "available": True,
        "candidates": len(scores),
        "reference": {"key": "full", "step": "G",
                      "top1_mmsi": reference[0], "order": reference},
        "ladder": rows,
        "top_n_candidates_considered": len(scores) if n_available is None else int(n_available),
        "n_not_available": sum(1 for r in rows if r["status"] == NOT_AVAILABLE),
        "statement": (
            "Agreement with the full system, computed on the same candidate set. A "
            "step whose top-1 already matches G is not carrying the result; a step that "
            "flips the top-1 is where the argument lives."),
        "caveat": ("With fewer than about five candidates these agreements are not "
                   "statistically meaningful. The ladder shows which assumptions the "
                   "order depends on, not how reliably it would hold up."),
    }


def _name_of(suspects: Sequence[Dict[str, Any]], mmsi: Optional[int]) -> Optional[str]:
    if mmsi is None:
        return None
    for s in suspects or ():
        if int(s["mmsi"]) == int(mmsi):
            return s.get("name")
    return None


def marginal_contribution(study: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What each stage adds, read off the ladder.

    For each step, the change relative to the previous step: did the top-1
    move, and by how much did the ranking reorder. This is the table an
    analyst reads to decide which stage to defend in writing.
    """
    if not study.get("available"):
        return []
    rows = [r for r in study.get("ladder") or () if r.get("status") == "computed"]
    out: List[Dict[str, Any]] = []
    previous: Optional[Dict[str, Any]] = None
    for row in rows:
        if previous is None:
            out.append({
                "step": row["step"], "label": row["label"],
                "change": None,
                "note": "Baseline of the ladder. This is the naive method.",
            })
        else:
            out.append({
                "step": row["step"], "label": row["label"],
                "added": sorted(set(row["includes"]) - set(previous["includes"])),
                "top1_changed": previous["top1_mmsi"] != row["top1_mmsi"],
                "previous_top1": previous["top1_mmsi"],
                "new_top1": row["top1_mmsi"],
                "spearman_vs_previous": spearman(previous["order"], row["order"]),
                "note": ("Adding this stage did not change the top-ranked vessel."
                         if previous["top1_mmsi"] == row["top1_mmsi"] else
                         "Adding this stage changed the top-ranked vessel from %s to %s."
                         % (previous["top1_mmsi"], row["top1_mmsi"])),
            })
        previous = row
    return out
