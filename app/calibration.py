"""Calibration metrics, and an honest account of what the scores are.

The pipeline's headline number is an investigative triage score: a weighted sum
of spatial, temporal, trajectory, behavioural and metadata features, scaled by
track confidence. That is not a probability. It has never been fitted against
outcome labels, so it must not be reported as "87 percent likely".

This module exists so that claim can be tested rather than asserted, and so a
reader can see how far from calibrated the score currently is. It provides:

    brier_score        proper scoring rule, lower is better
    log_loss           penalises confident mistakes very heavily
    expected_calibration_error
                       mean gap between stated confidence and observed frequency,
                       over fixed bins, with a monotone reliability curve
    discrimination      AUC-like separation of positives from negatives
    skill_score        relative to a climatological base rate

The central honesty rule: with a single candidate set, calibration is not
estimable, and every function here says so and returns `estimable: false`
rather than a number computed from one or two samples. A calibration figure
from three candidates is not a measurement, and reporting it as one is the
specific dishonesty this module is built to prevent.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Below this many labelled samples, calibration statistics are noise. The
#: threshold is published rather than tuned so it cannot be adjusted to suit a
#: particular result.
MIN_SAMPLES = 20
N_BINS = 10

#: Confidence under which a score is treated as a near-abstention.
LOW_CONFIDENCE_FLOOR = 0.05

CALIBRATION_STATUS = {
    "calibrated": "Scores are close to outcome frequencies across bins.",
    "overconfident": "Scores run high: stated confidence exceeds observed frequency. "
                     "Treat the numbers as triage only.",
    "underconfident": "Scores run low: stated confidence falls below observed frequency.",
    "insufficient_samples": "Too few labelled candidates to estimate calibration. "
                            "No calibration claim is made.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip(p: float) -> float:
    return float(min(1.0 - 1e-9, max(1e-9, float(p))))


def _pairs(samples: Sequence[Tuple[float, int]]) -> List[Tuple[float, int]]:
    """Validate and normalise (score, label) pairs.

    Labels must be 0 or 1. A soft label is accepted because a partially
    corroborated candidate is genuinely between the classes, and silently
    rounding it to 0 or 1 would corrupt the measurement.
    """
    out: List[Tuple[float, int]] = []
    for item in samples or ():
        try:
            score, label = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not math.isfinite(score) or not math.isfinite(label):
            continue
        out.append((min(1.0, max(0.0, score)), label))
    return out


def _summary(samples: Sequence[Tuple[float, int]], bins: int = N_BINS) -> Optional[Dict[str, Any]]:
    """The object every function in this module returns, or None if too small."""
    if len(samples) < MIN_SAMPLES:
        return {
            "estimable": False,
            "n": len(samples),
            "min_samples": MIN_SAMPLES,
            "status": "insufficient_samples",
            "statement": CALIBRATION_STATUS["insufficient_samples"],
            "missing": (min(0, MIN_SAMPLES - len(samples))
                        if len(samples) < MIN_SAMPLES else 0),
        }
    return {"estimable": True, "n": len(samples)}


# ---------------------------------------------------------------------------
# Proper scoring rules
# ---------------------------------------------------------------------------

def brier_score(samples: Sequence[Tuple[float, int]]) -> Dict[str, Any]:
    """Mean squared error of the stated score against the outcome label.

    Range [0, 1], lower is better. A score of 1.0 on a negative scores 1.0;
    a score of 0.0 on a positive scores 1.0; both score 0.0.
    """
    pairs = _pairs(samples)
    base = _summary(pairs)
    if not base["estimable"]:
        base["brier"] = None
        return base
    total = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    base_rate = sum(y for _, y in pairs) / len(pairs)
    climatology = sum((base_rate - y) ** 2 for _, y in pairs) / len(pairs)
    base.update({
        "brier": round(total, 6),
        "base_rate": round(base_rate, 6),
        "climatology_brier": round(climatology, 6),
        "brier_skill_score": (round(1.0 - total / climatology, 6)
                              if climatology > 1e-12 else None),
        "interpretation": ("Lower is better. brier_skill_score is the improvement "
                           "over always predicting the base rate; <= 0 means the "
                           "score is no better than climatology."),
    })
    return base


def log_loss(samples: Sequence[Tuple[float, int]]) -> Dict[str, Any]:
    """Mean negative log likelihood, with an epsilon floor on the stated score.

    This rule cares much more about confident mistakes than Brier does, which is
    exactly the failure that matters here: a candidate reported as a near
    certainty that turns out to be a background vessel.
    """
    pairs = _pairs(samples)
    base = _summary(pairs)
    if not base["estimable"]:
        base["log_loss"] = None
        return base
    eps = 1e-6
    total = 0.0
    for p, y in pairs:
        p = min(1.0 - eps, max(eps, p))
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    base["log_loss"] = round(total / len(pairs), 6)
    base["interpretation"] = ("Lower is better. Heavily penalises high scores on "
                              "negatives, which is the error this pipeline is most "
                              "exposed to.")
    return base


# ---------------------------------------------------------------------------
# Expected calibration error and the reliability curve
# ---------------------------------------------------------------------------

def reliability_curve(samples: Sequence[Tuple[float, int]],
                      n_bins: int = N_BINS) -> Dict[str, Any]:
    """Stated confidence against observed frequency, per fixed-width bin.

    Bins are fixed-width on [0, 1] rather than quantile bins. Quantile bins
    always contain the same number of samples, which hides exactly the
    overconfidence this is meant to expose: a model that puts 90 percent of its
    candidates in one bucket.
    """
    pairs = _pairs(samples)
    base = _summary(pairs)
    if not base["estimable"]:
        base["bins"] = []
        return base
    n_bins = max(2, int(n_bins))
    width = 1.0 / n_bins
    rows: List[Dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = i * width, (i + 1) * width
        in_bin = [(p, y) for p, y in pairs
                  if (lo <= p < hi or (i == n_bins - 1 and p <= hi))]
        rows.append({
            "bin": i,
            "lower": round(lo, 4),
            "upper": round(hi, 4),
            "n": len(in_bin),
            "mean_score": round(sum(p for p, _ in in_bin) / len(in_bin), 6) if in_bin else None,
            "observed_frequency": (round(sum(y for _, y in in_bin) / len(in_bin), 6)
                                   if in_bin else None),
            "gap": (round(abs(sum(y for _, y in in_bin) / len(in_bin)
                              - sum(p for p, _ in in_bin) / len(in_bin)), 6)
                    if in_bin else None),
        })
    populated = [r for r in rows if r["n"]]
    ece = (round(sum(r["n"] * r["gap"] for r in populated) / len(pairs), 6)
           if populated else None)
    base["bins"] = rows
    base["n_bins"] = n_bins
    base["expected_calibration_error"] = ece
    base["interpretation"] = ("expected_calibration_error is the count-weighted mean "
                              "|stated - observed| across bins. 0 is perfect. An empty "
                              "bin means no candidate scored in that range, not a "
                              "perfect score there.")
    return base


def expected_calibration_error(samples: Sequence[Tuple[float, int]],
                               n_bins: int = N_BINS) -> Dict[str, Any]:
    """Count-weighted mean absolute gap between stated and observed frequency."""
    curve = reliability_curve(samples, n_bins=n_bins)
    curve["expected_calibration_error"] = curve.get("expected_calibration_error")
    return curve


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------

def _auc(pairs: Sequence[Tuple[float, int]]) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """Rank-based AUC with tie handling, plus the positive and negative counts."""
    pos = [p for p, y in pairs if y > 0.5]
    neg = [p for p, y in pairs if y <= 0.5]
    if not pos or not neg:
        return None, len(pos), len(neg)
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg)), len(pos), len(neg)


def discrimination(samples: Sequence[Tuple[float, int]]) -> Dict[str, Any]:
    """Can the score rank positives above negatives at all?

    This is a separate question from calibration. A score can be perfectly
    ordered and still be wildly overconfident, and it can be well calibrated
    and useless for ranking if every candidate scores the same.
    """
    pairs = _pairs(samples)
    base = _summary(pairs)
    if not base["estimable"]:
        base["auc"] = None
        return base
    auc, n_pos, n_neg = _auc(pairs)
    base["auc"] = None if auc is None else round(auc, 6)
    base["positives"] = n_pos
    base["negatives"] = n_neg
    base["mean_score_positives"] = (round(sum(p for p, y in pairs if y > 0.5) / n_pos, 6)
                                    if n_pos else None)
    base["mean_score_negatives"] = (round(sum(p for p, y in pairs if y <= 0.5) / n_neg, 6)
                                    if n_neg else None)
    base["interpretation"] = ("auc is the probability a random positive outscores a "
                              "random negative. 0.5 is chance. This measures ordering "
                              "only; it says nothing about whether the scores are "
                              "correctly scaled.")
    return base


# ---------------------------------------------------------------------------
# The report the spec asks for
# ---------------------------------------------------------------------------

def score_disclaimer() -> str:
    return ("The investigative score is a weighted, uncalibrated evidence index. "
            "It is not a probability and must not be presented as one.")


def assess(samples: Sequence[Tuple[float, int]], n_bins: int = N_BINS) -> Dict[str, Any]:
    """Full calibration report for a labelled candidate set.

    `samples` is a sequence of (score in [0, 1], label) pairs. A label of 1
    means the candidate was independently confirmed to be the source.

    Every sub-metric is reported together with `estimable`, so a caller cannot
    accidentally render a number that this module refused to compute.
    """
    pairs = _pairs(samples)
    brier = brier_score(pairs)
    ll = log_loss(pairs)
    curve = reliability_curve(pairs, n_bins=n_bins)
    disc = discrimination(pairs)

    result: Dict[str, Any] = {
        "n": len(pairs),
        "min_samples": MIN_SAMPLES,
        "estimable": bool(brier.get("estimable")),
        "status": brier.get("status", "insufficient_samples"),
        "statement": CALIBRATION_STATUS.get(brier.get("status", "insufficient_samples")),
        "brier": brier.get("brier"),
        "brier_skill_score": brier.get("brier_skill_score"),
        "base_rate": brier.get("base_rate"),
        "log_loss": ll.get("log_loss"),
        "expected_calibration_error": curve.get("expected_calibration_error"),
        "reliability_curve": curve.get("bins", []),
        "auc": disc.get("auc"),
        "positives": disc.get("positives", 0),
        "negatives": disc.get("negatives", 0),
        "score_is_a_probability": False,
        "disclaimer": score_disclaimer(),
        "how_to_supply_samples": (
            "Each sample is (score in [0, 1], label) where label 1 means the "
            "candidate was independently confirmed. Build the set from multiple "
            "cases; a single case cannot support a calibration claim."),
    }

    if not result["estimable"]:
        result["missing_samples"] = max(0, MIN_SAMPLES - len(pairs))
        return result

    ece = result["expected_calibration_error"]
    if ece is not None:
        # Compare mean score against mean observed frequency. Mean stated above
        # observed means the pipeline is systematically overconfident.
        mean_stated = sum(p for p, _ in pairs) / len(pairs)
        mean_observed = sum(y for _, y in pairs) / len(pairs)
        if mean_stated - mean_observed > 0.05:
            result["status"] = "overconfident"
        elif mean_observed - mean_stated > 0.05:
            result["status"] = "underconfident"
        else:
            result["status"] = "calibrated"
        result["statement"] = CALIBRATION_STATUS[result["status"]]
        result["mean_stated_score"] = round(mean_stated, 6)
        result["mean_observed_frequency"] = round(mean_observed, 6)
        result["mean_gap"] = round(mean_stated - mean_observed, 6)
    return result
