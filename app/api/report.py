"""GET /api/report/{job_id} - the case report.

Optional by design. HTML is always available because it needs nothing beyond
the standard library. PDF is produced only if reportlab happens to be installed;
if it is not, the endpoint says so instead of failing the demo.

The report is written as seven fixed sections, in the order a reader has to
work through them: what the case found, what was observed, what was inferred,
who is involved and what is said against each of them, how fragile the result
is, what the case was built from, and what it cannot tell you. The order is the
argument. A report that opens with a ranked vessel and puts the caveats in a
footer has already decided how it will be read.

One section model feeds both renderers. A report where the PDF and the HTML are
generated separately drifts, and the two that get circulated are never the two
that were written.

Every section is labelled with the kind of claim it makes, because the report
mixes three things a reader must not confuse: what a sensor measured, what a
model produced, and what an analyst concluded. A modelled origin hour quoted
next to a radar acquisition without that distinction is how an inference
becomes an eyewitness.
"""
from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from .. import config
from .. import uncertainty as uncertainty_mod
from ..jobs import store as job_store

# Re-exported so the state labels in this module are the same strings the
# pipeline produces, rather than literals that can drift out of step with them.
NO_OIL = uncertainty_mod.NO_OIL

router = APIRouter()


# --------------------------------------------------------------------------
# Section model
# --------------------------------------------------------------------------
# A block is a dict with a "kind" the renderers understand:
#   {"kind": "p",    "text": ...}
#   {"kind": "kv",   "rows": [(label, value), ...], "layer": "observed"}
#   {"kind": "list", "items": [...], "layer": ...}
#   {"kind": "table", "head": [...], "rows": [[...]], "layer": ...}
#   {"kind": "note", "text": ..., "tone": "warn" | "bad" | "ok"}
#
# "layer" is optional and, where present, is rendered as a small tag. It is the
# whole reason the report cannot be misread.

def _n(value: Any, digits: int = 2) -> str:
    """A number, or an explicit 'not available'.

    Never an empty cell. A blank in a report reads as zero, and zero is a
    measurement.
    """
    if value is None or value == "":
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return ("%%.%df" % digits) % value
    return str(value)


def _when(value: Optional[str]) -> str:
    if not value:
        return "not available"
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    return d.strftime("%Y-%m-%d %H:%M UTC")


def _layers(doc: Dict[str, Any]) -> Dict[str, Any]:
    det = doc.get("detection") or {}
    drift = doc.get("drift") or {}
    attr = doc.get("attribution") or {}
    poly = doc.get("primary_polygon") or (det.get("polygons") or [None])[0]
    return {
        "det": det,
        "drift": drift,
        "attr": attr,
        "props": (poly or {}).get("properties", {}) or {},
        "metrics": det.get("metrics") or {},
        "eo": det.get("eo") or {},
        "origin": drift.get("origin") or {},
        "quality": doc.get("case_quality") or {},
        "suspects": attr.get("suspects") or [],
    }


# --------------------------------------------------------------------------
# The seven sections
# --------------------------------------------------------------------------

def _section_summary(doc: Dict[str, Any], L: Dict[str, Any]) -> Dict[str, Any]:
    """1. What the case found, and how far it is entitled to go."""
    quality = L["quality"]
    safe = quality.get("safe_fail") or {}
    overall = quality.get("overall")
    suspects = L["suspects"]
    top = suspects[0] if suspects else None
    scene = doc.get("scene") or {}

    lede = (
        "This case examined Sentinel-1 radar imagery of %s acquired %s and asked "
        "two questions: whether an oil slick was present, and, if so, whether any "
        "recorded vessel can be put forward as an investigative lead for it."
        % (_n(scene.get("title") or (doc.get("input") or {}).get("scene_id"), 0),
           _when(scene.get("t_sat")))
    )

    rows: List[Any] = [
        ("Case reference", doc.get("job_id")),
        ("Radar acquisition", _when(scene.get("t_sat"))),
        ("Detector", _n(L["metrics"].get("detector"), 0)),
        ("Oil polygons reported", str(len(L["det"].get("polygons") or []))),
        ("Look-alikes excluded", str(len(L["det"].get("lookalikes") or []))),
        ("Candidates scored", str(len(suspects))),
    ]
    if top:
        rows.append(("Highest ranked lead", "%s, MMSI %s" % (top.get("name") or "UNKNOWN", top.get("mmsi"))))
        rows.append(("Recorded objections to that lead", str(top.get("counter_evidence_count") or 0)))

    blocks: List[Dict[str, Any]] = [
        {"kind": "kv", "rows": rows},
        {"kind": "note", "tone": "warn", "text":
            "Evidence quality for this case: %s. This describes the quality of the "
            "evidence available to the case. It is not a probability that any "
            "vessel discharged oil, and it is not a measure of blame."
            % (overall or "not rated")},
    ]

    # A clean scene is a conclusive answer, not a failure, so it is the one
    # state that reads as "ok". The other two are the case declining to name
    # anything, and that is reported plainly rather than dressed as a result.
    if safe.get("state"):
        blocks.append({"kind": "note", "tone": "ok" if safe.get("state") == NO_OIL else "bad",
                       "text": "Case state: %s. %s" % (safe.get("state"),
                                                       safe.get("detail") or "")})
        if safe.get("stopped_before"):
            blocks.append({"kind": "p", "text":
                           "The case stopped before: %s. Nothing downstream of that "
                           "point was computed, so nothing downstream of it can be "
                           "quoted as a finding."
                           % ", ".join(str(s).replace("_", " ") for s in safe["stopped_before"])})
    for c in quality.get("conclusions") or []:
        blocks.append({"kind": "p", "text": str(c)})

    if top:
        blocks.append({"kind": "note", "tone": "bad", "text":
            "The vessel named above is an investigative lead, not a finding of "
            "discharge. The ranking is an uncalibrated evidence index produced by "
            "a weighted model, and it names the vessel whose recorded track best "
            "fits a modelled slick, not the vessel that released it."})
    elif L["det"].get("polygons"):
        blocks.append({"kind": "note", "tone": "warn", "text":
            "No vessel is put forward. A slick was detected and an origin was "
            "estimated, but no recorded track is a defensible candidate. The case "
            "reports nothing rather than naming the least unlikely vessel."})

    return {"number": 1, "title": "Case summary", "lede": lede, "blocks": blocks}


def _section_observed(doc: Dict[str, Any], L: Dict[str, Any]) -> Dict[str, Any]:
    """2. What a sensor measured."""
    props, metrics, eo = L["props"], L["metrics"], L["eo"]
    polys = L["det"].get("polygons") or []
    shape = props.get("shape_diagnostics") or {}
    meta = metrics.get("detector_metadata") or {}

    lede = ("Everything in this section is a measurement or the direct output of a "
            "classification. Nothing here is a conclusion about cause.")

    if not polys:
        return {"number": 2, "title": "Observed evidence", "layer": "observed",
                "lede": lede, "blocks": [
                    {"kind": "p", "text":
                     "No oil polygon was classified above the reporting threshold on "
                     "this scene. %s look-alike polygon(s) were found and excluded. "
                     "This is a reported result rather than a processing failure: "
                     "there is no slick on the water to characterise, so drift and "
                     "attribution were not run."
                     % _n(metrics.get("lookalike_polygons_found"), 0)},
                    {"kind": "note", "tone": "warn", "text":
                     "A clean scene is a finding. It does not by itself establish "
                     "that no discharge occurred; it establishes that none was "
                     "detectable in this acquisition under this detector."}]}

    blocks: List[Dict[str, Any]] = [
        {"kind": "kv", "layer": "observed", "rows": [
            ("Oil area", "%s km2" % _n(props.get("area_km2"), 3)),
            ("Total oil area on scene", "%s km2" % _n(metrics.get("oil_area_km2"), 3)),
            ("Length", "%s km" % _n(props.get("length_km"), 2)),
            ("Width", "%s km" % _n(props.get("width_km"), 2)),
            ("Perimeter", "%s km" % _n(props.get("perimeter_km"), 2)),
            ("Orientation", "%s degrees" % _n(props.get("orientation_deg"), 0)),
            ("Contrast against local sea", "%s dB" % _n(props.get("contrast_db"), 1)),
            ("Centroid", "%s, %s" % (_n(props.get("centroid_lat"), 4), _n(props.get("centroid_lon"), 4))),
            ("Polygon confidence", _n(props.get("confidence"), 3)),
            ("Elongation", _n(shape.get("elongation"), 2)),
            ("Solidity", _n(shape.get("solidity"), 3)),
            ("Boundary complexity", _n(props.get("boundary_complexity"), 3)),
        ]},
    ]

    # Detector provenance belongs in the observed section: a classification is
    # only as good as the thing that produced it, and a silent fallback to the
    # threshold baseline changes what the whole section means.
    det_rows = [
        ("Detector", _n(meta.get("name"), 0)),
        ("Trained detector", "yes" if meta.get("is_trained_detector") else "no"),
        ("Checkpoint", _n(meta.get("checkpoint"), 0)),
        ("Threshold", _n(meta.get("threshold"), 3)),
        ("Input size", _n(" x ".join(str(v) for v in meta.get("input_shape") or []) or None, 0)),
        ("Benchmark status", _n(meta.get("benchmark_status"), 0)),
    ]
    blocks.append({"kind": "kv", "layer": "observed", "rows": det_rows})
    if meta and not meta.get("is_trained_detector"):
        blocks.append({"kind": "note", "tone": "bad", "text":
                      "The trained detector was not used. %s The detection above was "
                      "produced by the published dark-spot radiometric threshold "
                      "baseline and has none of the trained model's capability."
                      % (meta.get("fallback_reason") or "No checkpoint was available.")})

    if eo.get("available"):
        delta = eo.get("time_delta") or {}
        counts = eo.get("counts") or {}
        blocks.append({"kind": "kv", "layer": "observed", "rows": [
            ("Optical acquisition", _when(eo.get("acquired"))),
            ("Offset from radar", "%s h" % _n(delta.get("hours"), 1)),
            ("Optical status", _n(eo.get("status"), 0)),
            ("Cloud free fraction", _n(eo.get("cloud_percent"), 1)),
            ("Valid pixel fraction", _n(delta.get("valid_pixel_fraction"), 3)),
            ("Oil pixels agreeing", _n(counts.get("consistent"), 0)),
            ("Oil pixels disagreeing", _n(counts.get("inconsistent"), 0)),
        ]})
        blocks.append({"kind": "note", "tone": "warn", "text":
                       "The optical pass is corroboration only and was never an input "
                       "to detection. It is a different sensor looking at the same "
                       "water %s hours away, so a disagreement between them is weak "
                       "evidence about anything." % _n(delta.get("hours"), 1)})
    else:
        blocks.append({"kind": "p", "text":
                       "No optical corroboration was available: %s. The detection "
                       "above rests on radar alone." % (eo.get("reason") or "no chip was cached for this scene")})

    return {"number": 2, "title": "Observed evidence", "layer": "observed",
            "lede": lede, "blocks": blocks}


def _section_inferred(doc: Dict[str, Any], L: Dict[str, Any]) -> Dict[str, Any]:
    """3. What a model produced."""
    drift, origin = L["drift"], L["origin"]
    if not drift:
        return {"number": 3, "title": "Modelled inference", "layer": "inferred",
                "lede": "The drift model was not run for this case.",
                "blocks": [{"kind": "p", "text":
                            "There was no slick to trace backwards, so no origin was "
                            "estimated and no release time was inferred."}]}

    mo = drift.get("metocean") or {}
    interval = origin.get("release_time_interval") or {}
    lede = ("Everything in this section is produced by a transport model. The values "
            "are estimates with stated uncertainty and none of them is an observation.")

    blocks: List[Dict[str, Any]] = [
        {"kind": "kv", "layer": "inferred", "rows": [
            ("Estimated release position", "%s, %s" % (_n(origin.get("lat"), 4), _n(origin.get("lon"), 4))),
            ("Estimated release time", _when(origin.get("t"))),
            ("Release time window", "%s to %s" % (_when(interval.get("start")), _when(interval.get("end")))),
            ("90 percent envelope radius", "%s km" % _n(origin.get("spread_km"), 2)),
            ("Envelope percentile", _n(origin.get("percentile"), 0)),
            ("50 percent envelope area", "%s km2" % _n(origin.get("area_50_km2"), 1)),
            ("90 percent envelope area", "%s km2" % _n(origin.get("area_90_km2"), 1)),
            ("Buffered zone radius", "%s km" % _n(origin.get("buffer_km"), 2)),
            ("Buffered zone area", "%s km2" % _n(origin.get("area_km2"), 1)),
            ("Drift time since release", "%s h" % _n(doc.get("age_hours_proxy"), 2)),
            ("Alternate origins modelled", str(len(origin.get("scenario_origins") or []))),
            ("Metocean source", _n(mo.get("source"), 0)),
            ("Mean current", "%s m/s" % _n(mo.get("mean_current_ms"), 3)),
            ("Mean 10 m wind", "%s m/s" % _n(mo.get("mean_wind_ms"), 2)),
        ]},
        {"kind": "note", "tone": "warn", "text":
         "The release time is the first hour at which the backward ensemble spread "
         "exceeded its trigger. It is a modelling artefact of where the particle "
         "cloud stopped narrowing, not a measured discharge time, and it should not "
         "be quoted as one."},
    ]
    if not drift.get("metocean_covers_scene", True) or mo.get("synthetic"):
        blocks.append({"kind": "note", "tone": "bad", "text":
                       "The cached metocean does not cover this scene and time. The "
                       "transport model ran on a synthetic substitute field, so the "
                       "origin above is weakly supported."})
    return {"number": 3, "title": "Modelled inference", "layer": "inferred",
            "lede": lede, "blocks": blocks}


def _section_attribution(doc: Dict[str, Any], L: Dict[str, Any]) -> Dict[str, Any]:
    """4. Who is involved, and what is said against each of them."""
    suspects = L["suspects"]
    lede = ("Each candidate below carries two separate numbers. Opportunity is where "
            "the vessel was and when, and says nothing about evidence. Evidence is "
            "how well the recorded track matches the modelled slick. They are "
            "reported separately because collapsing them into one score is how a "
            "vessel that merely passed through gets described as a source.")

    if not suspects:
        return {"number": 4, "title": "Attribution analysis", "layer": "attributed",
                "lede": lede, "blocks": [
                    {"kind": "p", "text":
                     "No vessel passed the spatio-temporal filter for this origin. "
                     "The case reports no lead."},
                    {"kind": "note", "tone": "ok", "text":
                     "Reporting nothing is the correct behaviour here. A forced "
                     "candidate would be worse than no candidate, because it would "
                     "carry a number and an authority it has not earned."}]}

    blocks: List[Dict[str, Any]] = []
    for s in suspects[:10]:
        ev = s.get("evidence") or {}
        region = s.get("region") or {}
        win = s.get("release_window") or {}
        detail = s.get("detail") or {}

        blocks.append({"kind": "table", "layer": "attributed",
                       "caption": "Rank %s: %s, MMSI %s, %s" % (
                           s.get("rank"), s.get("name") or "UNKNOWN",
                           s.get("mmsi"), _n(s.get("type"), 0)),
                       "head": ["Field", "Value"],
                       "rows": [
                           ["Ranked evidence index", "%.1f" % s.get("score", 0.0)],
                           ["On evidence alone, before any track-quality discount",
                            "%.1f" % s.get("score_before_confidence", s.get("score", 0.0))],
                           ["Track confidence applied", "%.2f" % s.get("confidence", 1.0)],
                           ["Opportunity", "%s (%s)" % (_n(ev.get("opportunity_level"), 0),
                                                         _n(ev.get("opportunity_score"), 1))],
                           ["Evidence", "%s (%s)" % (_n(ev.get("evidence_level"), 0),
                                                    _n(ev.get("evidence_score"), 1))],
                           ["Inside 50 percent envelope", _n(region.get("inside_50"))],
                           ["Inside 90 percent envelope", _n(region.get("inside_90"))],
                           ["Observed track fraction in zone",
                            _n(region.get("observed_track_fraction"), 3)],
                           ["Inside release window",
                            "not applicable" if win.get("applicable") is False else _n(win.get("inside"))],
                           ["Closest approach", _when(detail.get("closest_approach_utc"))],
                           ["Distance from origin estimate",
                            "%s km" % _n(detail.get("origin_distance_km"), 2)],
                           ["Recorded objections", str(s.get("counter_evidence_count") or 0)],
                       ]})

        # Both sides come from the same evidence record, and both are printed.
        # `positive_evidence` and `negative_evidence` are the itemised per
        # category findings; `counter_evidence` is the record of objections the
        # scorer raised about the candidate as a whole. Showing the supports
        # without the objections is how a weighted score gets quoted as a
        # finding, so the objections are not optional here.
        def _texts(items: Any) -> List[str]:
            return [i.get("text") if isinstance(i, dict) else str(i)
                    for i in (items or [])]

        positives = _texts(ev.get("positive_evidence")) or _texts(s.get("reasons"))
        negatives = _texts(ev.get("negative_evidence"))
        if positives:
            blocks.append({"kind": "list", "layer": "attributed",
                           "caption": "Why this vessel was put forward",
                           "items": positives})
        if negatives:
            blocks.append({"kind": "list", "layer": "attributed",
                           "caption": "Findings that argue against it",
                           "items": negatives})

        counter = s.get("counter_evidence") or ev.get("counter_evidence") or []
        if counter:
            blocks.append({"kind": "list", "layer": "attributed",
                           "caption": "Recorded objections to this candidate",
                           "items": _texts(counter)})
        if not negatives and not counter:
            blocks.append({"kind": "note", "tone": "warn", "text":
                           "No counter-evidence was recorded against this candidate. "
                           "That is the absence of a recorded objection, not evidence "
                           "that none exists."})

        dq = _texts(ev.get("data_quality"))
        if dq:
            blocks.append({"kind": "list", "layer": "observed",
                           "caption": "Track data quality",
                           "items": dq})
        uq = _texts(ev.get("uncertainty"))
        if uq:
            blocks.append({"kind": "list", "layer": "inferred",
                           "caption": "Intervals wide enough to matter",
                           "items": uq})

    blocks.append({"kind": "note", "tone": "bad", "text":
                   "The ranking above is an uncalibrated evidence index. It has not "
                   "been validated against confirmed discharge outcomes, it is not a "
                   "probability, and it does not establish that any vessel released "
                   "oil. Attribution here means investigative triage."})
    return {"number": 4, "title": "Attribution analysis", "layer": "attributed",
            "lede": lede, "blocks": blocks}


def _section_robustness(doc: Dict[str, Any], L: Dict[str, Any]) -> Dict[str, Any]:
    """5. How fragile the result is."""
    quality = L["quality"]
    chain = quality.get("uncertainty_chain") or {}
    stages = chain.get("stages") or []
    sens = L["attr"].get("sensitivity") or []
    ablation = doc.get("ablation") or {}
    calibration = doc.get("calibration") or {}

    lede = ("This section exists so that a reader can decide how much weight the "
            "rest of the report carries. A result that holds only under the "
            "assumptions we chose is a weaker result, and the reader is entitled to "
            "know which it is.")

    blocks: List[Dict[str, Any]] = []

    # Sensitivity and ablation are statements about a ranking, so they can only
    # be made when there is a ranking. On a clean scene or a no-candidate case
    # the studies describe a run that did not produce one, and printing their
    # tables would put a vessel's name in a report whose own summary says no
    # vessel can be named. That contradiction is the whole failure mode this
    # project is trying to avoid, so the tables are withheld and the reason is
    # stated instead.
    no_ranking = not L["suspects"]
    if no_ranking:
        blocks.append({"kind": "p", "text":
                       "This case produced no candidate ranking, so there is nothing "
                       "for a sensitivity or ablation study to be robust about. "
                       "The uncertainty and calibration rows below still apply, "
                       "because they describe the case rather than a vessel."})
        sens = []
        ablation = {}

    if stages:
        blocks.append({"kind": "table", "layer": "inferred",
                       "caption": "Stage certainty, 1.0 being inputs in good order",
                       "head": ["Stage", "Certainty", "Level", "Note"],
                       "rows": [[s.get("label") or s.get("stage"),
                                 _n(s.get("quantity"), 3),
                                 _n(s.get("level"), 0),
                                 s.get("note") or ""] for s in stages]})

    if sens:
        for r in sens:
            rows = []
            for s in r.get("scenarios") or []:
                if s.get("applicable") is False:
                    rows.append([s.get("label") or s.get("name"), "not evaluated",
                                 s.get("reason") or ""])
                else:
                    rows.append([
                        s.get("label") or s.get("name"),
                        "%.1f at rank %s" % (float(s.get("score") or 0.0), s.get("rank")),
                        ("%+.1f points against baseline" % float(s.get("delta") or 0.0))
                        if s.get("delta") is not None else "",
                    ])
            blocks.append({"kind": "table", "caption":
                           "Counterfactuals for %s, baseline %.1f at rank %s, %s" % (
                               r.get("candidate_name") or r.get("candidate_id"),
                               float(r.get("baseline_score") or 0.0),
                               r.get("baseline_rank"),
                               str(r.get("stability") or "not evaluated").replace("_", " ")),
                           "head": ["Scenario", "Score under it", "Effect"],
                           "rows": rows})
    else:
        blocks.append({"kind": "p", "text":
                       "No counterfactual sensitivity study was recorded for this case."})

    ladder = ablation.get("ladder") or []
    if ladder:
        rows = []
        for r in ladder:
            if r.get("status") != "computed":
                rows.append([r.get("step"), r.get("label"), "not evaluated",
                             r.get("reason") or ""])
                continue
            agree = r.get("agreement") or {}
            rows.append([
                r.get("step"), r.get("label"),
                r.get("top1_name") or ("MMSI %s" % r.get("top1_mmsi")),
                "top 1 %s, Spearman %s against the full system" % (
                    "held" if agree.get("top1_match") else "changed",
                    _n(r.get("spearman_vs_full"), 3))])
        blocks.append({"kind": "table", "caption": "Ablation ladder",
                       "head": ["Step", "Stage", "Its top-1", "Agreement with the full system"],
                       "rows": rows})
        if ablation.get("caveat"):
            blocks.append({"kind": "p", "text": ablation["caveat"]})
        blocks.append({"kind": "note", "tone": "warn", "text":
                       "A step marked not evaluated is a missing input, not a measured "
                       "insensitivity. It makes no claim either way and must not be "
                       "read as showing that the stage did not matter."})

    cal_rows = [
        ("Labelled outcome samples available", str(calibration.get("n", 0))),
        ("Minimum required before calibration is claimed",
         str(calibration.get("min_samples", 20))),
        ("Brier score", _n(calibration.get("brier"), 4)),
        ("Brier skill score", _n(calibration.get("brier_skill"), 3)),
        ("Expected calibration error", _n(calibration.get("ece"), 4)),
        ("Area under the ROC", _n(calibration.get("auc"), 3)),
    ]
    blocks.append({"kind": "kv", "caption": "Score calibration", "rows": cal_rows})
    if not calibration.get("estimable", False):
        blocks.append({"kind": "note", "tone": "bad", "text":
                       "The investigative score is not calibrated. %s labelled outcome "
                       "samples were available and at least %s are required. Until that "
                       "exists, the ranking must be read as a relative ordering for "
                       "triage and nothing more."
                       % (calibration.get("n", 0), calibration.get("min_samples", 20))})

    return {"number": 5, "title": "Uncertainty and robustness", "layer": "inferred",
            "lede": lede, "blocks": blocks}


def _section_provenance(doc: Dict[str, Any], L: Dict[str, Any]) -> Dict[str, Any]:
    """6. What the case was built from."""
    prov = doc.get("provenance") or {}
    # The completeness record is the writer's own account of what it filled in,
    # and it separates a field that is missing from one that is legitimately
    # absent. Reading `prov["inputs"]` and `prov["software"]` found nothing,
    # because the payload is a flat dict with those names spelled differently.
    complete = prov.get("completeness") or {}
    absent = complete.get("absent") or []
    nullable = complete.get("legitimately_absent") or []

    lede = ("The case is reproducible from the record below. Where an input was "
            "absent it is listed as absent, because an absent input and an input "
            "that was checked and found clean are different states and only one of "
            "them is a result.")

    scene_p = prov.get("scene") or {}
    det_p = prov.get("detector") or {}
    mo_p = prov.get("metocean") or {}
    opt_p = prov.get("optical") or {}
    ais_p = prov.get("ais") or {}

    rows = [
        ("Case identifier", prov.get("case_hash") or doc.get("job_id")),
        ("Software version", _n(prov.get("software_version"), 0)),
        ("Attribution model", _n(prov.get("attribution_model"), 0)),
        ("Scene", "%s, %s, %s" % (_n(scene_p.get("id"), 0), _n(scene_p.get("source"), 0),
                                  _when(scene_p.get("t_sat")))),
        ("Scene licence", _n(scene_p.get("license"), 0)),
        ("Detector", "%s%s" % (_n(det_p.get("name"), 0),
                               "" if det_p.get("is_trained_detector")
                               else " (threshold fallback, not the trained model)")),
        ("Metocean source", _n(mo_p.get("source"), 0)),
        ("Currents present", _n(mo_p.get("has_currents"))),
        ("Metocean is synthetic", _n(mo_p.get("synthetic"))),
        ("Optical status", _n(opt_p.get("status"), 0)),
        ("AIS release window", _when(ais_p.get("release_window"))),
        ("Declared provenance fields", str(complete.get("declared_fields") or 0)),
        ("Populated", str(complete.get("populated") or 0)),
        ("Absent", str(len(absent))),
        ("Absent by design", str(len(nullable))),
        ("Record complete", _n(complete.get("complete"))),
    ]
    blocks: List[Dict[str, Any]] = [{"kind": "kv", "rows": rows}]

    if det_p.get("fallback_reason"):
        blocks.append({"kind": "note", "tone": "bad", "text":
                       "The trained detector was not used: %s" % det_p["fallback_reason"]})

    if absent:
        blocks.append({"kind": "table", "caption": "Declared provenance fields left empty",
                       "head": ["Field", "What it records"],
                       "rows": [[a.get("field") or "field", a.get("description") or "absent"]
                                for a in absent]})
    if nullable:
        blocks.append({"kind": "table", "caption":
                       "Absent by design, because the fallback or uncached path applies",
                       "head": ["Field", "What it would record"],
                       "rows": [[a.get("field") or "field", a.get("description") or "absent"]
                                for a in nullable]})
        blocks.append({"kind": "note", "tone": "warn", "text":
                       "The fields above are empty because this case used a fallback "
                       "or had no cached asset, not because the record was truncated. "
                       "The effect of each on this case's conclusions is stated in the "
                       "section that uses it."})

    if not complete:
        blocks.append({"kind": "note", "tone": "warn", "text":
                       "This case carries no completeness record, so the report cannot "
                       "state which declared inputs were populated. Treat its "
                       "provenance as unverified."})

    trace = doc.get("trace") or []
    if trace:
        # A trace step carries a `status` and whatever detail that step passed,
        # not a fixed "note" column, so the extra keys are folded into one cell
        # rather than dropped on the floor.
        rows = []
        for t in trace:
            extra = {k: v for k, v in t.items()
                     if k not in ("step", "elapsed_ms", "status", "started_ms")}
            detail = ", ".join("%s=%s" % (k, v) for k, v in extra.items()) or t.get("note") or ""
            rows.append([t.get("step"), _n(t.get("status"), 0),
                         "%s ms" % _n(t.get("elapsed_ms"), 0), detail])
        blocks.append({"kind": "table", "caption": "Pipeline execution",
                       "head": ["Step", "Status", "Elapsed", "Detail"],
                       "rows": rows})

    return {"number": 6, "title": "Data provenance and reproducibility",
            "lede": lede, "blocks": blocks}


def _section_limits(doc: Dict[str, Any], L: Dict[str, Any]) -> Dict[str, Any]:
    """7. What this case cannot tell you."""
    quality = L["quality"]
    ordered = quality.get("ordered_factors") or []
    warnings = doc.get("warnings") or []

    lede = ("This section is not a disclaimer. Each item below is a specific way "
            "the conclusions of this report could be wrong, and each one names what "
            "would be needed to close it.")

    items = [
        "The ranking is an uncalibrated evidence index, not a probability. It has "
        "not been validated against confirmed discharge outcomes and should not be "
        "quoted as a likelihood figure.",
        "The release time and origin are model output. They depend on the cached "
        "metocean fields, on the wind drift factor and on the spread trigger, none "
        "of which are measurements of this event.",
        "Age since release is a transport-time proxy. No chemical weathering model "
        "is applied, so it is not a laboratory age and cannot date a release to a "
        "particular hour.",
        "Look-alike classifications such as low wind streaks and natural slicks are "
        "excluded from attribution. A false negative in that step removes a candidate "
        "silently, and the case cannot detect that it has done so.",
        "AIS is self-reported and freely transmittable. A vessel that was not "
        "reporting, or reported a false position, is not represented in the "
        "candidate set at all. Absence from this list is not absence from the water.",
        "Dead-reckoned track segments are inferred across reporting gaps. A position "
        "reconstructed across a long gap is a different kind of fact from a received "
        "one, and the gap fractions are reported for that reason.",
        "Optical corroboration is a different sensor at a different time and is "
        "never an input to detection. It cannot rescue or invalidate the radar result.",
        "Case quality bands describe the evidence available. They are not a "
        "conclusion about the event and not a measure of blame.",
    ]
    blocks: List[Dict[str, Any]] = [
        {"kind": "list", "caption": "Standing limitations of this method", "items": items},
    ]

    if ordered:
        # `ordered_factors` holds stage keys in weakest-first order and
        # `factors` holds the detail for each of them. Reading the list alone
        # emitted a column of bare internal keys with "not available" levels.
        factors = quality.get("factors") or {}
        blocks.append({"kind": "table", "caption": "Weakest inputs in this case, in order",
                       "head": ["Input", "Certainty", "Level", "Note"],
                       "rows": [[(factors.get(k) or {}).get("label") or k,
                                 _n((factors.get(k) or {}).get("quantity"), 3),
                                 _n((factors.get(k) or {}).get("level"), 0),
                                 (factors.get(k) or {}).get("note") or ""]
                                for k in ordered[:8]]})

    if warnings:
        blocks.append({"kind": "list", "caption": "Warnings raised by the pipeline",
                       "items": [str(w) for w in warnings]})

    blocks.append({"kind": "note", "tone": "warn", "text":
                   "To move from an investigative lead to a finding, the case needs "
                   "outcome confirmation: a discharge record, a sampling result, or "
                   "interview evidence tying a vessel to this water at this time. "
                   "The platform deliberately does not infer any of those from the "
                   "remote sensing and traffic data it holds."})

    return {"number": 7, "title": "Limitations and next steps", "lede": lede, "blocks": blocks}


def _sections(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    L = _layers(doc)
    return [
        _section_summary(doc, L),
        _section_observed(doc, L),
        _section_inferred(doc, L),
        _section_attribution(doc, L),
        _section_robustness(doc, L),
        _section_provenance(doc, L),
        _section_limits(doc, L),
    ]


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------

def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _render_block(b: Dict[str, Any]) -> str:
    kind = b.get("kind")
    out: List[str] = []

    if b.get("caption"):
        out.append('<p class="cap">%s</p>' % _esc(b["caption"]))

    if kind == "p":
        out.append("<p>%s</p>" % _esc(b.get("text")))

    elif kind == "kv":
        out.append('<dl class="kv">')
        for k, v in (b.get("rows") or []):
            out.append("<dt>%s</dt><dd>%s</dd>" % (_esc(k), _esc(v)))
        out.append("</dl>")

    elif kind == "list":
        out.append("<ul>")
        for i in b.get("items") or []:
            out.append("<li>%s</li>" % _esc(i))
        out.append("</ul>")

    elif kind == "table":
        out.append('<table><thead><tr>')
        for h in b.get("head") or []:
            out.append("<th>%s</th>" % _esc(h))
        out.append("</tr></thead><tbody>")
        for row in b.get("rows") or []:
            out.append("<tr>")
            for cell in row:
                out.append("<td>%s</td>" % _esc(cell))
            out.append("</tr>")
        out.append("</tbody></table>")

    elif kind == "note":
        out.append('<p class="note %s">%s</p>' % (b.get("tone") or "", _esc(b.get("text"))))

    if b.get("layer"):
        out.append('<p class="layer">%s claim</p>' % _esc(b["layer"]))
    return "".join(out)


def _render_html(doc: Dict[str, Any]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scene = doc.get("scene") or {}
    quality = doc.get("case_quality") or {}
    safe = (quality.get("safe_fail") or {}).get("state")

    head = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Case report %s</title>" % _esc(doc.get("job_id")),
        "<style>",
        ":root{--ink:#14212b;--ink2:#3f5464;--ink3:#5a6e7e;--line:#dbe3ea;"
        "--bg:#f4f7f9;--accent:#a4571a;--bad:#a5341f;--warn:#8a6410;--ok:#1c6b52}",
        "*{box-sizing:border-box}",
        "body{margin:0;background:var(--bg);color:var(--ink);"
        "font:14px/1.6 -apple-system,'Segoe UI',system-ui,Helvetica,Arial,sans-serif}",
        ".sheet{max-width:940px;margin:0 auto;padding:40px 28px 80px}",
        "header.doc{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:8px}",
        ".eyebrow{font:600 10px/1 -apple-system,'Segoe UI',system-ui,sans-serif;"
        "letter-spacing:.16em;text-transform:uppercase;color:var(--ink3)}",
        "h1{font-size:26px;line-height:1.2;margin:10px 0 6px;letter-spacing:-.01em}",
        ".docmeta{font:12px/1.5 ui-monospace,Consolas,monospace;color:var(--ink3)}",
        ".bandrow{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}",
        ".band{font:600 10px/1 -apple-system,'Segoe UI',system-ui,sans-serif;"
        "letter-spacing:.1em;text-transform:uppercase;padding:5px 9px;"
        "border:1px solid var(--line);border-radius:3px;color:var(--ink2)}",
        ".band.q{border-color:var(--warn);color:var(--warn)}",
        "nav.toc{background:#fff;border:1px solid var(--line);border-radius:6px;"
        "padding:14px 18px;margin:22px 0 30px;columns:2;column-gap:28px}",
        "nav.toc a{display:block;color:var(--ink2);text-decoration:none;font-size:12.5px;padding:2px 0}",
        "nav.toc a:hover{color:var(--accent)}",
        "section{margin:0 0 34px;page-break-inside:avoid}",
        "h2{display:flex;align-items:baseline;gap:10px;font-size:17px;margin:0 0 4px;"
        "padding-bottom:8px;border-bottom:1px solid var(--line)}",
        "h2 .n{font:600 12px/1 ui-monospace,Consolas,monospace;color:var(--accent)}",
        ".lede{color:var(--ink2);font-size:13px;margin:10px 0 16px}",
        "p{margin:0 0 12px}",
        ".cap{font:600 11px/1.4 -apple-system,'Segoe UI',system-ui,sans-serif;"
        "letter-spacing:.04em;color:var(--ink3);margin:0 0 6px}",
        "dl.kv{display:grid;grid-template-columns:minmax(180px,1fr) 1.2fr;gap:0;"
        "border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:0 0 14px}",
        "dl.kv dt,dl.kv dd{margin:0;padding:7px 11px;border-bottom:1px solid var(--line);"
        "font-size:12.5px}",
        "dl.kv dt{background:#fff;color:var(--ink2);font-size:11.5px}",
        "dl.kv dd{background:var(--bg);font-family:ui-monospace,Consolas,monospace;"
        "font-size:12px;overflow-wrap:anywhere}",
        "table{width:100%;border-collapse:collapse;margin:0 0 14px;"
        "border:1px solid var(--line);border-radius:6px;overflow:hidden}",
        "th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;"
        "font-size:12px;vertical-align:top}",
        "th{background:#fff;font:600 10px/1.4 -apple-system,'Segoe UI',system-ui,sans-serif;"
        "letter-spacing:.09em;text-transform:uppercase;color:var(--ink3)}",
        "tr:last-child td{border-bottom:none}",
        "td{font-family:ui-monospace,Consolas,monospace}",
        "ul{margin:0 0 14px;padding-left:20px}",
        "li{margin:0 0 5px;font-size:12.5px;color:var(--ink2)}",
        ".note{border-left:3px solid var(--warn);background:#fffdf6;padding:10px 13px;"
        "margin:0 0 14px;font-size:12.5px;color:var(--ink2)}",
        ".note.bad{border-color:var(--bad);background:#fff7f5}",
        ".note.ok{border-color:var(--ok);background:#f4fbf8}",
        ".note b{color:var(--ink)}",
        ".layer{font:600 9px/1 -apple-system,'Segoe UI',system-ui,sans-serif;"
        "letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);margin:0 0 12px}",
        "footer.doc{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);"
        "font-size:11px;color:var(--ink3)}",
        "@media print{body{background:#fff}.sheet{padding:0}}",
        "</style></head><body><div class='sheet'>",
    ]

    body: List[str] = list(head)
    body.append("<header class='doc'>")
    body.append("<p class='eyebrow'>%s &middot; %s</p>" % (_esc(config.UI_TITLE), _esc(config.SIH_ID)))
    body.append("<h1>Case report: %s</h1>" % _esc(
        (scene.get("title") or (doc.get("input") or {}).get("scene_id") or doc.get("job_id"))))
    body.append("<p class='docmeta'>Case %s &middot; radar pass %s &middot; generated %s</p>" % (
        _esc(doc.get("job_id")), _esc(_when(scene.get("t_sat"))), _esc(generated)))

    band = ["<div class='bandrow'>"]
    if quality.get("overall"):
        band.append("<span class='band q'>Evidence quality: %s</span>" % _esc(quality.get("overall")))
    if safe:
        band.append("<span class='band'>%s</span>" % _esc(safe))
    band.append("<span class='band'>Uncalibrated evidence index</span>")
    band.append("</div>")
    body.append("".join(band))
    body.append("</header>")

    sections = _sections(doc)
    body.append("<nav class='toc'>")
    for s in sections:
        body.append("<a href='#s%d'>%d. %s</a>" % (s["number"], s["number"], _esc(s["title"])))
    body.append("</nav>")

    for s in sections:
        body.append("<section id='s%d'>" % s["number"])
        body.append("<h2><span class='n'>%d</span> %s</h2>" % (s["number"], _esc(s["title"])))
        if s.get("lede"):
            body.append("<p class='lede'>%s</p>" % _esc(s["lede"]))
        for b in s.get("blocks") or []:
            body.append(_render_block(b))
        body.append("</section>")

    body.append("<footer class='doc'>")
    body.append("Generated by %s from the stored job document. Every figure in this "
                "report is read from that document and none was recomputed for "
                "presentation. The investigative score is an uncalibrated evidence "
                "index and the candidate named here is an investigative lead, not a "
                "finding of discharge." % _esc(config.UI_TITLE))
    body.append("</footer>")
    body.append("</div></body></html>")
    return "".join(body)


def _render_text(doc: Dict[str, Any]) -> List[str]:
    """A plain-text rendering of the same sections, for the PDF and for copy
    and paste. Same model, so the two cannot drift apart."""
    out: List[str] = [
        "CASE REPORT",
        "=" * 78,
        "Case: %s" % doc.get("job_id"),
        "Generated: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "",
    ]
    for s in _sections(doc):
        out.append("")
        out.append("%d. %s" % (s["number"], s["title"].upper()))
        out.append("-" * 78)
        if s.get("lede"):
            out.append(s["lede"])
            out.append("")
        for b in s.get("blocks") or []:
            if b.get("caption"):
                out.append(b["caption"])
            kind = b.get("kind")
            if kind == "p":
                out.append(str(b.get("text")))
            elif kind == "kv":
                for k, v in b.get("rows") or []:
                    out.append("  %-42s %s" % (str(k)[:42], v))
            elif kind == "list":
                for i in b.get("items") or []:
                    out.append("  - %s" % i)
            elif kind == "table":
                head = b.get("head") or []
                out.append("  " + " | ".join(str(h) for h in head))
                for row in b.get("rows") or []:
                    out.append("  " + " | ".join(str(c) for c in row))
            elif kind == "note":
                out.append("  [NOTE] %s" % b.get("text"))
            out.append("")
    return out


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@router.get("/api/report/{job_id}", response_class=HTMLResponse)
def report_html(job_id: str) -> HTMLResponse:
    doc = job_store.load(job_id)
    if doc is None:
        raise HTTPException(404, "unknown job_id %r" % job_id)
    return HTMLResponse(_render_html(doc))


@router.get("/api/report/{job_id}/text", response_class=PlainTextResponse)
def report_text(job_id: str) -> PlainTextResponse:
    """The same seven sections as plain text.

    A report that will be pasted into an email or a case file has to survive
    losing its styling, and re-typing it from the PDF is how a caveat gets
    dropped on the way to the reader who needed it.

    Served as real text/plain rather than as escaped HTML, so it is diffable,
    greppable and pasteable without a browser reflowing it.
    """
    doc = job_store.load(job_id)
    if doc is None:
        raise HTTPException(404, "unknown job_id %r" % job_id)
    return PlainTextResponse("\n".join(_render_text(doc)) + "\n")


@router.get("/api/report/{job_id}/pdf")
def report_pdf(job_id: str) -> Response:
    doc = job_store.load(job_id)
    if doc is None:
        raise HTTPException(404, "unknown job_id %r" % job_id)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise HTTPException(
            501, "reportlab is not installed, so PDF export is unavailable. "
                 "The HTML report at /api/report/%s is always available." % job_id) from exc

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 56
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Case report: %s" % (
        (doc.get("scene") or {}).get("title") or doc.get("job_id")))
    y -= 16
    c.setFont("Helvetica", 8)
    c.drawString(50, y, "%s  %s  generated %s" % (
        config.UI_TITLE, config.SIH_ID,
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")))
    y -= 20

    c.setFont("Courier", 8)
    for line in _render_text(doc):
        if y < 56:
            c.showPage()
            c.setFont("Courier", 8)
            y = height - 56
        c.drawString(50, y, line[:118])
        y -= 10.5
    c.save()
    return Response(buf.getvalue(), media_type="application/pdf", headers={
        "Content-Disposition": 'attachment; filename="case_report_%s.pdf"' % job_id})
