"""Uncertainty propagation, evidence taxonomy, and case-level confidence.

The pipeline's answer is not one number. It is a chain in which every stage is
less certain than the pixel or the row it consumed:

    SAR pixels
       -> detection uncertainty      how sure the oil class is
    geometry
       -> slick footprint uncertainty
    metocean fields
       -> transport model uncertainty
    ensemble
       -> origin region + release-time interval
    AIS reconstruction
       -> track uncertainty (observed vs dead reckoned)
    correlation
       -> candidate evidence + counter-evidence
    all of the above
       -> CASE evidence quality

Collapsing that into a single "87 percent" is the failure mode this module
exists to prevent. Every band it emits carries the measurement the band was
derived from, so an analyst can argue with the band rather than trust it.

Three rules govern everything here.

1. A band is never a probability. HIGH / MEDIUM / LOW describe the *quality of
   the evidence available at that stage*, not the odds that a vessel is guilty.
2. A case cannot be rated above its weakest material evidence source. A perfect
   vessel ranking computed from a fallback metocean cube is not a strong case.
3. Where a stage is unavailable, the state is reported as unavailable. It is
   never replaced with a plausible-looking number, and it is never silently
   treated as a pass.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

LOW, MEDIUM, HIGH = "LOW", "MEDIUM", "HIGH"
LEVEL_ORDER: Dict[str, int] = {LOW: 0, MEDIUM: 1, HIGH: 2}

# Threshold on a [0, 1] evidence quantity for each band. Deliberately fixed and
# published rather than tuned per case: a band that moves when nobody is looking
# is indistinguishable from an arbitrary number.
HIGH_AT = 0.70
MEDIUM_AT = 0.42


def level(value: Optional[float], high_at: float = HIGH_AT,
          medium_at: float = MEDIUM_AT) -> str:
    """Band an evidence quantity. Absent or non-finite evidence is LOW."""
    if value is None:
        return LOW
    try:
        v = float(value)
    except (TypeError, ValueError):
        return LOW
    if not math.isfinite(v):
        return LOW
    if v >= high_at:
        return HIGH
    if v >= medium_at:
        return MEDIUM
    return LOW


def weakest(levels: Iterable[str]) -> str:
    """The weakest level in a set. A case cannot outrank its worst input."""
    values = [LEVEL_ORDER.get(str(item).upper(), 0) for item in levels]
    if not values:
        return LOW
    floor = min(values)
    for name, rank in LEVEL_ORDER.items():
        if rank == floor:
            return name
    return LOW


def _band_from_measurement(value: Optional[float], high_at: float = HIGH_AT,
                           medium_at: float = MEDIUM_AT) -> str:
    return level(value, high_at, medium_at)


# ---------------------------------------------------------------------------
# Evidence taxonomy
# ---------------------------------------------------------------------------

# The categories the spec asks attribution to report against. Kept as data so
# the API, the report and the console all render the same vocabulary.
EVIDENCE_CATEGORIES: Tuple[Dict[str, str], ...] = (
    {"key": "spatial", "label": "Spatial evidence",
     "question": "How close did the vessel come to the probable source region?"},
    {"key": "temporal", "label": "Temporal evidence",
     "question": "Is the closest approach compatible with the release window?"},
    {"key": "trajectory", "label": "Trajectory evidence",
     "question": "Is the vessel's heading compatible with a transit of the source region?"},
    {"key": "behaviour", "label": "Behaviour evidence",
     "question": "Did the vessel do anything anomalous near the source region?"},
    {"key": "metadata", "label": "Vessel metadata evidence",
     "question": "Is the vessel's type and carriage consistent with a release?"},
    {"key": "ais_quality", "label": "AIS quality",
     "question": "How much of the track was received rather than assumed?"},
    {"key": "environmental", "label": "Environmental consistency",
     "question": "Do the cached metocean fields cover the scene and time?"},
    {"key": "counter_evidence", "label": "Counter-evidence",
     "question": "What argues against this candidate?"},
)

EVIDENCE_KINDS: Tuple[str, ...] = ("positive", "negative", "quality", "uncertainty")


def evidence_item(category: str, kind: str, text: str,
                  weight: Optional[float] = None) -> Dict[str, Any]:
    """One line of the evidence record.

    `kind` decides how the console draws it and how it is aggregated:

        positive     supports the candidate
        negative     argues against it
        quality      describes how good the underlying data is
        uncertainty  describes how wide the relevant interval is

    Quality and uncertainty are never counted as support and never counted as
    opposition. They describe the case, not the vessel.
    """
    if kind not in EVIDENCE_KINDS:
        raise ValueError("unknown evidence kind %r" % kind)
    if category not in {item["key"] for item in EVIDENCE_CATEGORIES}:
        raise ValueError("unknown evidence category %r" % category)
    return {"category": category, "kind": kind, "text": text,
            "weight": None if weight is None else round(float(weight), 4)}


def split_evidence(items: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group a flat evidence list by kind, preserving order inside each group."""
    out: Dict[str, List[Dict[str, Any]]] = {kind: [] for kind in EVIDENCE_KINDS}
    for item in items or ():
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind in out:
            out[kind].append(item)
    return out


# ---------------------------------------------------------------------------
# Safe-fail vocabulary
# ---------------------------------------------------------------------------

# The system is allowed to decline to conclude. These are the three honest
# terminal states, and each one names what is missing rather than inventing a
# weaker answer.
NO_OIL = "NO OIL DETECTED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT EVIDENCE"
NO_CANDIDATE = "NO RELIABLE VESSEL CANDIDATE"


def safe_fail_state(detection: Optional[Dict[str, Any]],
                    drift: Optional[Dict[str, Any]],
                    attribution: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Decide whether this case concludes, and say plainly if it does not.

    Returns None when the case does conclude. Otherwise a dict with the state
    label, a human sentence, and the specific missing input.
    """
    metrics = (detection or {}).get("metrics") or {}
    polygons = (detection or {}).get("polygons") or []
    if not polygons:
        clean = (detection or {}).get("clean_scene") or {}
        return {
            "state": NO_OIL,
            "headline": clean.get("headline") or NO_OIL,
            "detail": clean.get("detail") or
                      "No polygon exceeded the minimum reporting area on this scene.",
            "missing_input": "oil above the reporting area threshold",
            "stopped_before": ["drift", "attribution"],
            "lookalike_polygons_found": metrics.get("lookalike_polygons_found", 0),
        }

    if drift is None:
        return {
            "state": INSUFFICIENT_EVIDENCE,
            "headline": "Slick detected but not reconstructable.",
            "detail": ("Oil was found but no drift reconstruction exists, so there is no "
                       "origin and no release window to correlate against."),
            "missing_input": "a drift hindcast",
            "stopped_before": ["attribution"],
        }

    suspects = (attribution or {}).get("suspects") or []
    if not suspects:
        considered = ((attribution or {}).get("funnel") or {}).get("considered_vessels", 0)
        return {
            "state": NO_CANDIDATE,
            "headline": "Origin estimated. No vessel is a defensible candidate.",
            "detail": ("%d vessel tracks were considered in the origin window and time "
                       "band and %d were close enough to score. No culprit is forced."
                       % (int(considered or 0), 0)),
            "missing_input": "an AIS track intersecting the probable source region",
            "stopped_before": [],
        }
    return None


# ---------------------------------------------------------------------------
# Uncertainty chain
# ---------------------------------------------------------------------------

def uncertainty_chain(detection: Optional[Dict[str, Any]],
                      drift: Optional[Dict[str, Any]],
                      attribution: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Describe the uncertainty at every stage, with the measurement behind it.

    This is the object that stops a confident vessel ranking from implying a
    confident case. Each stage reports its quantity in the unit a reader can
    reason about (km, hours, dB, percent), never as an unexplained band.
    """
    metrics = (detection or {}).get("metrics") or {}
    radiometry = metrics.get("radiometry") or {}
    origin = (drift or {}).get("origin") or {}
    physics = (drift or {}).get("physics") or {}
    store = (attribution or {}).get("store") or {}

    lookalikes = int(metrics.get("lookalike_polygons_found") or 0)
    oil_px = int(metrics.get("oil_pixels") or 0)
    la_px = int(metrics.get("lookalike_pixels") or 0)
    total_px = oil_px + la_px

    # Detection. A trained checkpoint scores higher than the radiometric
    # baseline, and a scene with look-alikes in it is harder than one without.
    trained = str(metrics.get("detector") or "") == "unet"
    if trained:
        detect_q = 0.75
    elif metrics.get("detector"):
        detect_q = 0.45
    else:
        detect_q = 0.0
    if lookalikes and total_px:
        detect_q -= min(0.2, 0.5 * la_px / total_px)

    stages: List[Dict[str, Any]] = [
        {
            "stage": "sar_detection",
            "label": "SAR detection",
            "quantity": round(float(detect_q), 3),
            "level": _band_from_measurement(detect_q),
            "measurement": {
                "detector": metrics.get("detector"),
                "oil_pixels": oil_px,
                "lookalike_pixels": la_px,
                "lookalike_polygons_found": lookalikes,
                "dynamic_range_db": radiometry.get("dynamic_range_db"),
                "sea_level_db": radiometry.get("sea_level_db"),
            },
            "note": ("Trained detector over a scene containing look-alike targets."
                     if trained else
                     "Radiometric baseline fallback, not a benchmarked detector."),
        }
    ]

    if drift is None:
        stages.append({
            "stage": "drift_certainty", "label": "Drift certainty",
            "quantity": 0.0, "level": LOW,
            "measurement": {}, "note": "Not run: there was no slick to hindcast.",
        })
        stages.append({
            "stage": "origin_certainty", "label": "Origin certainty",
            "quantity": 0.0, "level": LOW,
            "measurement": {}, "note": "No origin estimate exists.",
        })
    else:
        metocean = (drift or {}).get("metocean") or {}
        current_ok = metocean.get("has_currents") is not False
        covers = bool(drift.get("metocean_covers_scene"))
        drift_q = 0.8 if (covers and current_ok and not metocean.get("synthetic")) else 0.35
        spread_km = float(origin.get("spread_km") or 0.0)
        # A 90 percent ensemble spread under 4 km is a tight origin; over 12 km
        # the region is larger than most of the search radius and is barely a
        # constraint on any candidate.
        origin_q = max(0.0, min(1.0, 1.0 - spread_km / 16.0))
        interval = origin.get("release_time_interval") or {}
        release_h = _interval_hours(interval.get("start"), interval.get("end"))
        release_q = max(0.0, min(1.0, 1.0 - release_h / 24.0)) if release_h is not None else 0.0

        stages.append({
            "stage": "drift_certainty", "label": "Drift certainty",
            "quantity": round(drift_q, 3), "level": _band_from_measurement(drift_q),
            "measurement": {
                "source": metocean.get("source"),
                "covers_scene": covers,
                "has_currents": current_ok,
                "synthetic": bool(metocean.get("synthetic")),
                "mean_wind_ms": metocean.get("mean_wind_ms"),
                "mean_current_ms": metocean.get("mean_current_ms"),
                "ensemble_particles": physics.get("n_particles"),
                "dt_seconds": physics.get("dt_seconds"),
            },
            "note": ("Real cached wind and currents cover the scene."
                     if drift_q >= HIGH_AT else
                     "Wind-only or partially covering environmental field."),
        })
        stages.append({
            "stage": "origin_certainty", "label": "Origin certainty",
            "quantity": round(origin_q, 3), "level": _band_from_measurement(origin_q),
            "measurement": {
                "spread_km": spread_km,
                "area_50_km2": origin.get("area_50_km2"),
                "area_90_km2": origin.get("area_90_km2"),
                "hours_back": origin.get("index_hours_back"),
            },
            "note": "90 percent ensemble envelope half-width, buffered outward.",
        })
        stages.append({
            "stage": "release_time_uncertainty", "label": "Release-time uncertainty",
            "quantity": round(release_q, 3), "level": _band_from_measurement(release_q),
            "measurement": {
                "interval_start": interval.get("start"),
                "interval_end": interval.get("end"),
                "interval_hours": release_h,
                "scenarios": len(origin.get("scenario_origins") or []),
            },
            "note": ("Width of the release interval across drift scenarios."
                     if release_h is not None else
                     "Release time is a single scenario estimate, not an interval."),
        })

    sources = (attribution or {}).get("sources_used") or {}
    simulated = sum(v for k, v in sources.items() if "simulated" in str(k))
    real = sum(v for k, v in sources.items() if "simulated" not in str(k))
    total = simulated + real
    if total == 0:
        ais_q = 0.0
    elif simulated == 0:
        ais_q = 0.8
    else:
        ais_q = 0.8 * (real / float(total))

    stages.append({
        "stage": "ais_coverage", "label": "AIS coverage",
        "quantity": round(ais_q, 3), "level": _band_from_measurement(ais_q),
        "measurement": {
            "candidate_tracks": total,
            "real_tracks": real,
            "simulated_tracks": simulated,
            "sources": sources,
            "store_rows": store.get("rows"),
            "store_vessels": store.get("vessels"),
        },
        "note": ("All candidate tracks are recorded AIS."
                 if simulated == 0 and total else
                 "%d of %d candidate tracks are simulated traffic." % (simulated, total)),
    })

    eo = (detection or {}).get("eo") or {}
    eo_available = bool(eo.get("available"))
    verdicts = eo.get("verdicts") or []
    counts: Dict[str, int] = {}
    for v in verdicts:
        key = str(v.get("verdict") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    if not eo_available:
        eo_q = 0.0
    else:
        usable = counts.get("consistent", 0) + counts.get("inconsistent", 0)
        eo_q = 0.6 if verdicts else 0.2
        if eo.get("weak"):
            eo_q -= 0.2
    stages.append({
        "stage": "optical_corroboration", "label": "Optical corroboration",
        "quantity": round(max(0.0, eo_q), 3), "level": _band_from_measurement(max(0.0, eo_q)),
        "measurement": {
            "available": eo_available,
            "counts": counts,
            "offset_hours": eo.get("offset_hours"),
            "weak": bool(eo.get("weak")),
            "reason": eo.get("reason"),
        },
        "note": (eo.get("caveat") or "Optical corroboration is reported, never applied."),
    })

    material = [item["level"] for item in stages
                if item["stage"] in ("sar_detection", "drift_certainty",
                                     "origin_certainty", "ais_coverage",
                                     "optical_corroboration")]
    return {
        "stages": stages,
        "overall": weakest(material),
        "policy": ("A case cannot be rated above its weakest material evidence source. "
                   "Look-alike risk is a hazard label, not a quality input."),
        "bands_are_not_probabilities": True,
    }


def _interval_hours(start: Optional[str], end: Optional[str]) -> Optional[float]:
    if not start or not end:
        return None
    from datetime import datetime

    def parse(value: str) -> Optional[datetime]:
        try:
            s = str(value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        except (TypeError, ValueError):
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=None)

    a, b = parse(start), parse(end)
    if a is None or b is None:
        return None
    return abs((b - a).total_seconds()) / 3600.0


# ---------------------------------------------------------------------------
# Opportunity versus evidence
# ---------------------------------------------------------------------------

OPPORTUNITY_NOTE = (
    "Opportunity answers whether this vessel could physically have caused the "
    "release: where it was, when, at what speed and of what type. It is a "
    "statement about the vessel, not about the incident."
)
EVIDENCE_NOTE = (
    "Evidence answers how much observed support there is for that opportunity. "
    "A tall tanker in open water has opportunity and no evidence; a small boat "
    "with a documented reporting gap over the origin has both. They are "
    "reported separately so vessel type alone cannot make a candidate look "
    "suspicious."
)


def split_opportunity_evidence(components: Dict[str, float],
                               weights: Dict[str, float]) -> Dict[str, Any]:
    """Separate the two dimensions the spec asks to be shown independently.

    Opportunity: prox + time + traj + type.  Could it have been there?
    Evidence:   prox + time + traj + beh.     Is there observed support?

    Both are weighted means over the same five sub-scores, so neither is a
    re-weighting of the ranking and neither changes `raw`.
    """
    opportunity_terms = ("prox", "time", "traj", "type")
    evidence_terms = ("prox", "time", "traj", "beh")

    def weighted_mean(terms: Sequence[str]) -> float:
        total = sum(weights.get(k, 0.0) for k in terms) or 1.0
        return sum(weights.get(k, 0.0) * float(components.get(k, 0.0)) for k in terms) / total

    opportunity = weighted_mean(opportunity_terms)
    evidence = weighted_mean(evidence_terms)
    return {
        "opportunity": {
            "score": round(100.0 * opportunity, 1),
            "level": _band_from_measurement(opportunity),
            "terms": list(opportunity_terms),
            "note": OPPORTUNITY_NOTE,
        },
        "evidence": {
            "score": round(100.0 * evidence, 1),
            "level": _band_from_measurement(evidence),
            "terms": list(evidence_terms),
            "note": EVIDENCE_NOTE,
        },
    }


# ---------------------------------------------------------------------------
# Case assessment
# ---------------------------------------------------------------------------

# Which stages cap the overall case band. Look-alike risk is excluded because
# LOW is favourable there and including it would invert its meaning.
CAP_STAGES = ("sar_detection", "drift_certainty", "origin_certainty",
              "ais_coverage", "optical_corroboration")


def case_assessment(detection: Optional[Dict[str, Any]],
                    drift: Optional[Dict[str, Any]],
                    attribution: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The case-level confidence layer, kept separate from any vessel ranking.

    A top candidate in a weak case is still a weak investigative lead. This
    object is what stops the console from communicating "top vessel = certain".
    """
    chain = uncertainty_chain(detection, drift, attribution)
    metrics = (detection or {}).get("metrics") or {}
    lookalikes = int(metrics.get("lookalike_polygons_found") or 0)

    factors = {stage["stage"]: stage for stage in chain["stages"]}

    # Look-alike risk is a hazard, not a quality grade: LOW risk is good.
    if lookalikes == 0:
        factors["lookalike_risk"] = {
            "stage": "lookalike_risk", "label": "Look-alike risk",
            "quantity": 0.0, "level": LOW,
            "measurement": {"lookalike_polygons_found": 0},
            "note": "No look-alike target was detected, so none had to be excluded.",
        }
    else:
        area = float(metrics.get("lookalike_area_km2") or 0.0)
        oil_area = float(metrics.get("oil_area_km2") or 0.0)
        ratio = area / oil_area if oil_area > 0 else 1.0
        # `quantity` is a RISK quantity: larger means a harder scene. Count and
        # area ratio contribute equally, so either many small look-alikes or a
        # few large ones registers. This must not be inverted: HIGH here means
        # HIGH RISK, which is the opposite sense to the quality bands above.
        risk_q = min(1.0, 0.5 * min(1.0, lookalikes / 20.0) + 0.5 * min(1.0, ratio))
        risk = _band_from_measurement(risk_q)
        factors["lookalike_risk"] = {
            "stage": "lookalike_risk", "label": "Look-alike risk",
            "quantity": round(risk_q, 3), "level": risk,
            "measurement": {"lookalike_polygons_found": lookalikes,
                            "lookalike_area_km2": round(area, 3),
                            "oil_area_km2": round(oil_area, 3),
                            "area_ratio": round(min(1.0, ratio), 3)},
            "note": ("Look-alikes are detected, drawn and excluded from attribution. "
                     "%d look-alike polygon(s) totalling %.1f km2 were found against "
                     "%.1f km2 of oil, so the detector faced hard negatives here."
                     % (lookalikes, area, oil_area)),
        }

    ordered = [factors[key] for key in
               ("sar_detection", "lookalike_risk", "drift_certainty", "origin_certainty",
                "release_time_uncertainty", "ais_coverage", "optical_corroboration")
               if key in factors]
    # `reason` is kept as an alias of `note` so the older case_quality consumers,
    # which read factors[...]["reason"], keep working against the richer object.
    for item in ordered:
        item.setdefault("reason", item.get("note"))
    overall = weakest(factors[key]["level"] for key in CAP_STAGES if key in factors)

    conclusions: List[str] = []
    if overall == LOW:
        conclusions.append("At least one material input is missing or fallback-only. "
                           "Treat any ranking as a lead, not a finding.")
    if factors.get("lookalike_risk", {}).get("level") == HIGH:
        conclusions.append("The detector faced many look-alike targets on this scene.")
    if factors.get("ais_coverage", {}).get("level") == LOW:
        conclusions.append("Candidate evidence does not rest on recorded AIS.")

    return {
        "overall": overall,
        "factors": {item["stage"]: item for item in ordered},
        "ordered_factors": [item["stage"] for item in ordered],
        "conclusions": conclusions,
        "uncertainty_chain": chain,
        "safe_fail": safe_fail_state(detection, drift, attribution),
        "label": "Case evidence quality, reported separately from candidate ranking.",
        "bands_are_not_probabilities": True,
    }
