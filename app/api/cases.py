"""Case-oriented read APIs for the investigation console.

A "case" is one completed run, and these endpoints are shaped around how an
investigator actually works: list what has been run, open one, then interrogate
the specific claims it makes. That last part is why there are separate routes
for sensitivity, ablation and calibration instead of one endpoint returning
everything. A UI that renders a top-ranked vessel without the counter-evidence
next to it can always fetch the case and simply not show the dissent, so the
dissent has an endpoint of its own.

Nothing here recomputes anything. Every figure is read from the stored job
document, so a case opened tomorrow reports exactly what it reported today.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from .. import calibration as calibration_mod
from .. import uncertainty as uncertainty_mod
from ..jobs import store as job_store

router = APIRouter()


def _load(job_id: str) -> Dict[str, Any]:
    doc = job_store.load(job_id)
    if doc is None:
        raise HTTPException(404, "unknown job_id %r" % job_id)
    return doc


# ---------------------------------------------------------------------------
# Case index and header
# ---------------------------------------------------------------------------

@router.get("/api/cases")
def cases(limit: int = 50) -> Dict[str, Any]:
    """Every case on this machine, most recent first."""
    items = job_store.case_index(limit=limit)
    return {
        "cases": items,
        "count": len(items),
        "vocabulary": {
            "case_quality": ("HIGH / MEDIUM / LOW describe the quality of the evidence "
                             "available to this case. They are not probabilities."),
            "safe_fail_state": {
                "NO OIL DETECTED": "The scene has no slick above the reporting area.",
                "INSUFFICIENT EVIDENCE": "A slick exists but the case could not be "
                                         "carried far enough to name a candidate.",
                "NO RELIABLE VESSEL CANDIDATE": "An origin was estimated and no vessel is "
                                                "a defensible candidate. No culprit is forced.",
            },
        },
    }


@router.get("/api/cases/{job_id}")
def case(job_id: str) -> Dict[str, Any]:
    """The case header: what was run, what came out, and how far it got."""
    doc = _load(job_id)
    quality = doc.get("case_quality") or {}
    return {
        "job_id": doc.get("job_id"),
        "created": doc.get("created"),
        "status": doc.get("status"),
        "mode": doc.get("mode"),
        "scene": doc.get("scene"),
        "input": doc.get("input"),
        "safe_fail": quality.get("safe_fail"),
        "case_quality": {
            "overall": quality.get("overall"),
            "label": quality.get("label"),
            "conclusions": quality.get("conclusions") or [],
            # `ordered_factors` is only a weakest-first list of stage keys, so
            # `factors` travels with it: without the second half a consumer
            # gets bare internal names with no level or measurement beside
            # them, which is the entire content of the panel.
            "ordered_factors": quality.get("ordered_factors") or [],
            "factors": quality.get("factors") or {},
        },
        "counts": {
            "oil_polygons": len((doc.get("detection") or {}).get("polygons") or []),
            "lookalike_polygons": len((doc.get("detection") or {}).get("lookalikes") or []),
            "candidates": len((doc.get("attribution") or {}).get("suspects") or []),
        },
        "detector": ((doc.get("detection") or {}).get("metrics") or {}).get("detector_metadata"),
        "provenance": doc.get("provenance"),
        "warnings": doc.get("warnings") or [],
        "total_ms": doc.get("total_ms"),
        "has_drift": doc.get("drift") is not None,
        "has_attribution": bool((doc.get("attribution") or {}).get("suspects")),
    }


# ---------------------------------------------------------------------------
# The claims, one route each
# ---------------------------------------------------------------------------

@router.get("/api/cases/{job_id}/uncertainty")
def case_uncertainty(job_id: str) -> Dict[str, Any]:
    """The uncertainty chain: what is uncertain at each stage, and by how much.

    This is the answer to "how sure is this case", kept deliberately separate
    from "which vessel is most interesting". A case can be weak and still have
    a clear top candidate, and an analyst who only sees the ranking will not
    know that.
    """
    doc = _load(job_id)
    quality = doc.get("case_quality") or {}
    chain = quality.get("uncertainty_chain")
    if chain is None:
        chain = uncertainty_mod.uncertainty_chain(
            doc.get("detection") or {}, doc.get("drift"),
            doc.get("attribution") or {})
    return {
        "job_id": job_id,
        # Named for what it is rather than for the function that made it, so a
        # caller does not have to know that `chain` came from uncertainty_chain.
        "uncertainty_chain": chain,
        "quantity_is_certainty": (
            "Each stage carries a certainty, where 1.0 means that stage's inputs "
            "are in good order and 0.0 means the stage could not be run. It is "
            "not an uncertainty magnitude and not a probability."),
        "safe_fail": quality.get("safe_fail"),
        "taxonomy": list(uncertainty_mod.EVIDENCE_CATEGORIES),
        "bands_are_not_probabilities": True,
    }


@router.get("/api/cases/{job_id}/sensitivity")
def case_sensitivity(job_id: str) -> Dict[str, Any]:
    """How much each candidate's rank depends on assumptions we chose.

    Every scenario is reported, including the ones that could not be evaluated.
    A missing scenario is reported explicitly rather than left out, because
    silence would read as "this assumption does not matter".
    """
    doc = _load(job_id)
    return {
        "job_id": job_id,
        "sensitivity": (doc.get("attribution") or {}).get("sensitivity") or [],
        "scoring": (doc.get("attribution") or {}).get("scoring") or {},
        "note": ("Each scenario removes one declared assumption or widens one declared "
                 "uncertainty, then re-ranks. They create no new evidence and are not "
                 "probabilities."),
    }


@router.get("/api/cases/{job_id}/ablation")
def case_ablation(job_id: str) -> Dict[str, Any]:
    """Which stage of the pipeline is carrying the result."""
    doc = _load(job_id)
    study = doc.get("ablation")
    if study is None:
        raise HTTPException(404, "case %r predates the ablation study; re-run it" % job_id)
    # Returned whole and unreshaped. The console reads `ladder`, `status` and
    # `spearman_vs_full` off these rows directly, so a second vocabulary here
    # would be a second place for the UI to be wrong about the shape.
    return {"job_id": job_id, **study}


@router.get("/api/cases/{job_id}/calibration")
def case_calibration(job_id: str) -> Dict[str, Any]:
    """Whether the investigative score is calibrated, and if not, say so.

    For a single case this is normally `estimable: false`, and that is the
    correct answer rather than a missing feature: a calibration figure needs
    outcome labels, and the pipeline has none for an unconfirmed scene.
    """
    doc = _load(job_id)
    report = doc.get("calibration")
    if report is None:
        report = calibration_mod.assess([])
        report["reason"] = "this case predates the calibration report"
    return {"job_id": job_id, **report}


@router.get("/api/cases/{job_id}/evidence")
def case_evidence(job_id: str) -> Dict[str, Any]:
    """Per-candidate evidence, counter-evidence, envelope and window containment.

    Returned grouped by candidate rather than as one flat list so a caller
    cannot pair a reason with the wrong vessel, which is the most likely way a
    long leaderboard gets misread.
    """
    doc = _load(job_id)
    candidates: List[Dict[str, Any]] = []
    for s in ((doc.get("attribution") or {}).get("suspects") or []):
        evidence = s.get("evidence") or {}
        candidates.append({
            "mmsi": s.get("mmsi"),
            "rank": s.get("rank"),
            "name": s.get("name"),
            "type": s.get("type"),
            "score": s.get("score"),
            "score_before_confidence": s.get("score_before_confidence"),
            "confidence": s.get("confidence"),
            "components": s.get("components"),
            "why_this_vessel": {
                "opportunity_score": evidence.get("opportunity_score"),
                "opportunity_level": evidence.get("opportunity_level"),
                "opportunity_note": evidence.get("opportunity_note"),
                "evidence_score": evidence.get("evidence_score"),
                "evidence_level": evidence.get("evidence_level"),
                "evidence_note": evidence.get("evidence_note"),
                "supporting": evidence.get("positive_evidence") or [],
            },
            "why_not": {
                "count": s.get("counter_evidence_count", 0),
                "items": s.get("counter_evidence") or evidence.get("counter_evidence") or [],
            },
            "data_quality": evidence.get("data_quality") or [],
            "uncertainty": evidence.get("uncertainty") or [],
            "origin_region": s.get("region") or {},
            "release_window": s.get("release_window") or {},
            "detail": s.get("detail") or {},
        })
    return {
        "job_id": job_id,
        "candidates": candidates,
        "taxonomy": list(uncertainty_mod.EVIDENCE_CATEGORIES),
        "vocabulary": {
            "opportunity": uncertainty_mod.OPPORTUNITY_NOTE,
            "evidence": uncertainty_mod.EVIDENCE_NOTE,
        },
    }


@router.get("/api/cases/{job_id}/timeline")
def case_timeline(job_id: str) -> Dict[str, Any]:
    """The case as an ordered set of events, for the scrubber.

    Every entry carries a kind, so the timeline can tell observed facts from
    modelled inferences. Mixing the two on one axis is how a modelled origin
    hour ends up quoted as though a satellite saw it.
    """
    doc = _load(job_id)
    events: List[Dict[str, Any]] = []
    drift = doc.get("drift") or {}
    origin = drift.get("origin") or {}
    scene = doc.get("scene") or {}
    metrics = (doc.get("detection") or {}).get("metrics") or {}
    eo = (doc.get("detection") or {}).get("eo") or {}

    def add(kind: str, t: Optional[str], label: str, detail: str, **extra) -> None:
        events.append({"kind": kind, "t": t, "label": label, "detail": detail, **extra})

    add("observed", scene.get("t_sat"), "SAR acquisition",
        "The radar observed the water at this time. Everything downstream is relative to it.",
        detector=metrics.get("detector"))

    if origin.get("t"):
        add("inferred", origin.get("t"), "Estimated release",
            "Modelled. The first hour where the ensemble spread exceeded the trigger, "
            "not an observed discharge time.",
            spread_km=origin.get("spread_km"))

    interval = origin.get("release_time_interval") or {}
    if interval.get("start") and interval.get("end"):
        add("inferred", interval.get("start"), "Release window opens",
            "Earliest scenario origin across the drift ensemble.", bound="start")
        add("inferred", interval.get("end"), "Release window closes",
            "Latest scenario origin across the drift ensemble.", bound="end")

    if eo.get("acquired"):
        add("observed", eo.get("acquired"), "Optical acquisition",
            "Sentinel-2 overpass. %+.0f h relative to the radar." % float(
                (eo.get("time_delta") or {}).get("hours") or 0.0),
            status=eo.get("status"))

    for s in ((doc.get("attribution") or {}).get("suspects") or [])[:10]:
        detail = s.get("detail") or {}
        t = detail.get("closest_approach_utc")
        if not t:
            continue
        kind = "attributed" if s.get("rank") == 1 else "inferred"
        add(kind, t, "%s closest approach" % (s.get("name") or s.get("mmsi")),
            "Rank %s. %.2f km from the origin estimate." % (
                s.get("rank"), float(detail.get("origin_distance_km") or 0.0)),
            mmsi=s.get("mmsi"), rank=s.get("rank"), score=s.get("score"))

    events.sort(key=lambda e: (e.get("t") or ""))
    return {
        "job_id": job_id,
        "events": events,
        "kinds": {
            "observed": "Measured by a sensor at the stated time.",
            "inferred": "Produced by a model. Carries its own uncertainty.",
            "attributed": "An analyst-facing conclusion. Not a sensor observation.",
        },
        "warning": ("Modelled and attributed entries are not observations. Do not quote "
                    "an inferred origin hour as a measured discharge time."),
    }


@router.get("/api/cases/{job_id}/chain")
def case_chain(job_id: str) -> Dict[str, Any]:
    """Observed, inferred, attributed: the three layers, kept apart.

    The separation is the point. Each item is tagged with the layer it belongs
    to and the artifact it came from, so a report can be assembled from this
    without the wording drifting between "we saw" and "we think".
    """
    doc = _load(job_id)
    scene = doc.get("scene") or {}
    detection = doc.get("detection") or {}
    metrics = detection.get("metrics") or {}
    drift = doc.get("drift") or {}
    origin = drift.get("origin") or {}
    attribution = doc.get("attribution") or {}
    eo = detection.get("eo") or {}

    observed: List[Dict[str, Any]] = [
        {"statement": "Sentinel-1 acquired the scene at %s." % (scene.get("t_sat") or "an unrecorded time"),
         "source": "scene metadata", "value": scene.get("t_sat")},
        {"statement": "The detector classified %d oil pixels across %d polygon(s), "
                      "totalling %s km2." % (
                          int(metrics.get("oil_pixels") or 0),
                          len(detection.get("polygons") or []),
                          metrics.get("oil_area_km2")),
         "source": "SAR classification", "value": metrics.get("oil_area_km2")},
        {"statement": "%d look-alike polygon(s) were detected and excluded from attribution."
                      % int(metrics.get("lookalike_polygons_found") or 0),
         "source": "SAR classification", "value": metrics.get("lookalike_polygons_found")},
    ]
    if eo.get("available"):
        observed.append({
            "statement": "Optical corroboration status: %s." % (eo.get("status") or "unknown"),
            "source": "cached Sentinel-2 chip", "value": eo.get("status")})

    inferred: List[Dict[str, Any]] = []
    if drift:
        inferred.extend([
            {"statement": "Modelled source region centred at %.5f, %.5f with a 90 percent "
                          "envelope half-width of %s km." % (
                              float(origin.get("lon") or 0.0), float(origin.get("lat") or 0.0),
                              origin.get("spread_km")),
             "source": "Lagrangian ensemble hindcast", "value": origin.get("spread_km")},
            {"statement": "Release time estimated at %s." % (origin.get("t") or "unknown"),
             "source": "Lagrangian ensemble hindcast", "value": origin.get("t")},
            {"statement": "Transport used %s fields; scene coverage %s." % (
                (drift.get("metocean") or {}).get("source"),
                bool(drift.get("metocean_covers_scene"))),
             "source": "cached metocean", "value": (drift.get("metocean") or {}).get("source")},
        ])

    attributed: List[Dict[str, Any]] = []
    for s in (attribution.get("suspects") or [])[:5]:
        attributed.append({
            "statement": "%s (MMSI %s) is the highest-ranked investigative lead at %.1f, "
                          "with %d piece(s) of counter-evidence recorded against it." % (
                              s.get("name") or "UNKNOWN", s.get("mmsi"),
                              float(s.get("score") or 0.0),
                              int(s.get("counter_evidence_count") or 0)),
            "source": "weighted evidence model", "value": s.get("score")})

    quality = doc.get("case_quality") or {}
    return {
        "job_id": job_id,
        "observed": observed,
        "inferred": inferred,
        "attributed": attributed,
        "case_quality": quality.get("overall"),
        "conclusions": quality.get("conclusions") or [],
        "discipline": {
            "observed": "Measured. Can be checked against the source product.",
            "inferred": "Modelled. True value unknown; carries the stated uncertainty.",
            "attributed": "Concluded. Investigative triage, not a finding of discharge.",
        },
    }
