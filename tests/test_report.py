"""The report has to survive the cases it will actually be asked about.

A report is the artefact an investigator forwards. Two things go wrong in
practice and neither raises: a field the writer spells differently leaves a
"not available" cell where a measurement should be, and a case that concluded
early renders its missing sections as though they were empty rather than
absent. Both were real here, so both are pinned.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

SECTIONS = [
    "Case summary",
    "Observed evidence",
    "Modelled inference",
    "Attribution analysis",
    "Uncertainty and robustness",
    "Data provenance and reproducibility",
    "Limitations and next steps",
]


def _job(job_id="job_report", **over):
    doc = {
        "job_id": job_id,
        "created": "2026-01-01T00:00:00Z",
        "status": "ok",
        "input": {"scene_id": "demo_scene"},
        "scene": {"id": "demo_scene", "title": "Demo scene",
                  "t_sat": "2026-01-01T00:00:00Z",
                  "source": "Sentinel-1 GRD via Copernicus", "license": "open"},
        "age_hours_proxy": 6.4,
        "detection": {
            "polygons": [{"type": "Feature",
                          "geometry": {"type": "Polygon",
                                       "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                          "properties": {"polygon_id": "OIL01", "area_km2": 4.2,
                                         "length_km": 3.1, "width_km": 1.4,
                                         "perimeter_km": 9.8, "orientation_deg": 34,
                                         "contrast_db": -8.2, "centroid_lat": 2.5,
                                         "centroid_lon": 1.5, "confidence": 0.71,
                                         "boundary_complexity": 1.3,
                                         "shape_diagnostics": {"elongation": 3.4,
                                                               "solidity": 0.31,
                                                               "irregularity": 0.42}}}],
            "lookalikes": [{"polygon_id": "LA01", "reason": "low wind streak"}],
            "metrics": {"detector": "unet", "oil_pixels": 900, "oil_area_km2": 4.2,
                        "lookalike_polygons_found": 1,
                        "detector_metadata": {"name": "unet", "is_trained_detector": True,
                                              "checkpoint": "oil_unet_best.pt",
                                              "threshold_db": -8.0,
                                              "input_shape": [1, 1, 512, 512],
                                              "benchmark_status": "not benchmarked"}},
            "eo": {"available": False, "status": "unavailable",
                   "reason": "no chip was cached for this scene"},
        },
        "drift": {
            "origin": {"lon": 1.5, "lat": 2.5, "t": "2025-12-31T18:00:00Z",
                       "spread_km": 3.0, "buffer_km": 1.0, "area_km2": 41.2,
                       "area_50_km2": 18.0, "area_90_km2": 33.5, "percentile": 90.0,
                       "release_time_interval": {"start": "2025-12-31T17:00:00Z",
                                                 "end": "2025-12-31T19:00:00Z"}},
            "metocean": {"source": "cached cube", "synthetic": False,
                         "has_currents": True, "mean_current_ms": 0.31,
                         "mean_wind_ms": 4.2},
            "metocean_covers_scene": True,
        },
        "attribution": {
            "suspects": [{
                "mmsi": 111222333, "rank": 1, "name": "MT MERIDIAN", "type": "tanker",
                "score": 62.0, "score_before_confidence": 70.0, "confidence": 0.88,
                "counter_evidence_count": 1,
                "counter_evidence": [{"category": "counter_evidence", "kind": "negative",
                                      "text": "AIS gap of 90 min across the origin window."}],
                "evidence": {"opportunity_score": 70.0, "opportunity_level": "HIGH",
                             "evidence_score": 60.0, "evidence_level": "MEDIUM",
                             "positive_evidence": ["Present in the origin zone during the window."],
                             "negative_evidence": ["Heading differed by 61 degrees from the drift direction."],
                             "data_quality": ["41% of track dead reckoned."],
                             "uncertainty": ["Origin envelope spans 3.0 km at the release hour."]},
                "region": {"inside_50": True, "inside_90": True,
                           "observed_track_fraction": 0.59},
                "release_window": {"applicable": True, "inside": True},
                "detail": {"closest_approach_utc": "2025-12-31T18:10:00Z",
                           "origin_distance_km": 0.8},
            }],
            "sensitivity": [{
                "candidate_id": 111222333, "candidate_name": "MT MERIDIAN",
                "baseline_score": 62.0, "baseline_rank": 1,
                "scenarios": [
                    {"name": "origin_shift", "label": "Origin shifted 5 km",
                     "question": "If the origin estimate is 5 km off, does the rank hold?",
                     "applicable": True, "score": 51.0, "rank": 1, "delta": -11.0},
                    {"name": "wider_window", "label": "Release window widened to 6 h",
                     "applicable": False,
                     "reason": "The release interval came from the drift model."},
                ],
                "stability": "SENSITIVE", "max_score_delta": 11.0,
                "rank_changes": 0, "primary_dependency": "origin_accuracy",
            }],
        },
        "ablation": {
            "available": True, "candidates": 1,
            "reference": {"key": "full", "step": "G", "order": [111222333]},
            "ladder": [
                {"step": "A", "key": "naive", "label": "Distance to origin only",
                 "question": "How far from the origin was the vessel?",
                 "status": "computed", "top1_mmsi": 111222333,
                 "top1_name": "MT MERIDIAN", "order": [111222333],
                 "agreement": {"top1_match": True}, "spearman_vs_full": 1.0},
                {"step": "F", "key": "envelope", "label": "Spatial envelope containment",
                 "question": "Was the track inside the slick envelope?",
                 "status": "not_available",
                 "reason": "the raw tracks for this case were not supplied to the ablation"},
            ],
            "caveat": "With fewer than about five candidates these agreements are not "
                      "statistically meaningful.",
        },
        "calibration": {"n": 0, "min_samples": 20, "estimable": False,
                        "brier": None, "brier_skill": None, "ece": None, "auc": None},
        "case_quality": {
            "overall": "MEDIUM", "label": "Case evidence quality",
            "conclusions": ["The optical input is absent, so nothing corroborates radar."],
            "ordered_factors": ["optical"],
            "factors": {"optical": {"stage": "optical", "label": "Optical corroboration",
                                    "quantity": 0.0, "level": "LOW",
                                    "measurement": {}, "note": "No chip cached."}},
            "uncertainty_chain": {"overall": "MEDIUM", "stages": [
                {"stage": "sar_detection", "label": "SAR detection", "quantity": 0.9,
                 "level": "HIGH", "measurement": {}, "note": "Trained detector."},
                {"stage": "optical", "label": "Optical corroboration", "quantity": 0.0,
                 "level": "LOW", "measurement": {}, "note": "No chip cached."},
            ]},
            "safe_fail": None,
        },
        "provenance": {
            "case_hash": "abc123",
            "software_version": "0.9.0",
            "attribution_model": "baseline-weighted-score",
            "scene": {"id": "demo_scene", "source": "Sentinel-1", "license": "open",
                      "t_sat": "2026-01-01T00:00:00Z"},
            "detector": {"name": "unet", "is_trained_detector": True,
                         "checkpoint": "oil_unet_best.pt"},
            "metocean": {"source": "cached cube", "synthetic": False, "has_currents": True},
            "optical": {"available": False, "status": "unavailable"},
            "ais": {"release_window": "2025-12-31T17:00:00Z"},
            "completeness": {
                "declared_fields": 20, "populated": 17, "complete": False,
                "absent": [{"field": "drift.origin.t", "description": "estimated release time"}],
                "legitimately_absent": [{"field": "optical.status",
                                         "description": "optical status"}],
            },
        },
        "trace": [{"step": "DETECT", "status": "ok", "elapsed_ms": 12.0},
                  {"step": "AIS", "status": "ok", "elapsed_ms": 40.0,
                   "scored": 1, "considered": 3}],
    }
    doc.update(over)
    return doc


@pytest.fixture
def stored(app_config):
    from app.jobs import store

    store.save("job_report", _job())
    yield "job_report"
    p = store.path_for("job_report")
    if p.exists():
        p.unlink()


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_the_html_report_has_all_seven_sections_in_order(client, stored):
    r = client.get("/api/report/%s" % stored)
    assert r.status_code == 200
    body = r.text
    assert "content-type" in r.headers and "text/html" in r.headers["content-type"]
    for i, title in enumerate(SECTIONS, start=1):
        assert "<span class='n'>%d</span> %s" % (i, title) in body, (
            "section %d (%s) is missing or out of order" % (i, title))
    # And a table of contents that resolves to the sections it names.
    for i in range(1, 8):
        assert "href='#s%d'" % i in body and "id='s%d'" % i in body


def test_the_text_report_is_the_same_sections_as_plain_text(client, stored):
    """The text and PDF renderings go through the same section model, so a
    caveat cannot be dropped from one of them and kept in the other."""
    r = client.get("/api/report/%s/text" % stored)
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    for i, title in enumerate(SECTIONS, start=1):
        assert "%d. %s" % (i, title.upper()) in body, "%s missing from the text report" % title
    # The critical caveat has to survive the plain-text path.
    assert "uncalibrated evidence index" in body
    assert "not a finding of discharge" in body or "does not establish" in body


def test_the_text_report_is_not_html_wearing_a_text_content_type(client, stored):
    r = client.get("/api/report/%s/text" % stored)
    assert "<pre>" not in r.text and "&lt;" not in r.text, (
        "the text report is escaped markup, so it cannot be grepped or diffed")


def test_an_unknown_case_is_a_404_on_every_format(client):
    for suffix in ("", "/text", "/pdf"):
        r = client.get("/api/report/nope_not_here" + suffix)
        assert r.status_code == 404, "%s returned %s" % (suffix, r.status_code)


# ---------------------------------------------------------------------------
# The fields that were silently blank
# ---------------------------------------------------------------------------

def test_provenance_reads_the_record_the_pipeline_actually_writes(client, stored):
    """The section read `prov["inputs"]`, `prov["software"]` and
    `prov["seeds"]`. The payload is a flat dict spelled `software_version`, with
    the declared-field accounting under `completeness`, so every one of those
    reads returned nothing and the section printed a table of "not recorded"."""
    body = client.get("/api/report/%s" % stored).text
    assert "0.9.0" in body, "the software version is not shown"
    assert "abc123" in body, "the case hash is not shown"
    assert "cached cube" in body, "the metocean source is not shown"
    assert "not benchmarked" in body or "unet" in body
    # The two states have to stay distinguishable.
    assert "Declared provenance fields left empty" in body
    assert "Absent by design" in body
    assert "drift.origin.t" in body
    # And it must not claim to have seeds it never recorded.
    assert "Seeds" not in body


def test_the_inference_section_uses_the_keys_the_drift_writes(client, stored):
    """`index_hours_back` does not exist and rendered as "not available". The
    writer publishes both envelope areas and a percentile instead."""
    body = client.get("/api/report/%s" % stored).text
    assert "18.00" in body or "18.0" in body, "the 50 percent envelope area is missing"
    assert "33.50" in body or "33.5" in body, "the 90 percent envelope area is missing"
    assert "not available" not in body.split("3. Modelled inference")[1].split("4.")[0], (
        "the inference section has an empty cell where a measurement exists")


def test_robustness_reads_the_ladder_and_the_nested_scenarios(client, stored):
    """Both of these were read against an assumed shape: `rungs` instead of
    `ladder`, and one flat row per scenario instead of one record per candidate.
    Neither raised, and the section simply came out with nothing in it."""
    body = client.get("/api/report/%s" % stored).text
    # The ladder, with the unevaluable rung still present.
    assert "Distance to origin only" in body
    assert "Spatial envelope containment" in body
    assert "not evaluated" in body
    # The sensitivity record, per candidate, with its nested scenarios.
    assert "Origin shifted 5 km" in body
    assert "Release window widened to 6 h" in body
    assert "not evaluated" in body or "not evaluated" in body
    # The scenario that could not be evaluated is named, not dropped.
    assert "came from the drift model" in body


def test_the_uncertainty_table_is_labelled_certainty_not_uncertainty(client, stored):
    """`quantity` runs 0.0 for a stage that could not run to 1.0 for one whose
    inputs are in good order. Calling that column "uncertainty" inverts it."""
    body = client.get("/api/report/%s" % stored).text
    assert "Stage certainty" in body
    assert "certainty, 1.0 being inputs in good order" in body
    # The LOW stage's zero has to be visible, not rounded into a blank.
    assert "Optical corroboration" in body


# ---------------------------------------------------------------------------
# Semantic discipline
# ---------------------------------------------------------------------------

def test_both_sides_of_the_evidence_are_printed_for_the_top_candidate(client, stored):
    """A report that prints the supporting findings and drops the objections
    is the most likely way a weighted score turns into a stated fact."""
    body = client.get("/api/report/%s" % stored).text
    assert "Why this vessel was put forward" in body
    assert "Present in the origin zone" in body
    assert "AIS gap of 90 min" in body
    assert "Findings that argue against it" in body
    assert "61 degrees" in body


def test_the_detector_fallback_is_disclosed_rather_than_passed_off(client, stored,
                                                                   app_config):
    """A threshold-baseline detection presented like a trained-model detection
    is the single largest gap between what the report says and what produced
    the number. So it is asserted, not assumed."""
    from app.jobs import store

    doc = _job()
    doc["detection"]["metrics"]["detector_metadata"] = {
        "name": "dark_spot_threshold", "is_trained_detector": False,
        "fallback_reason": "no trained checkpoint was present"}
    store.save("job_report", doc)
    body = client.get("/api/report/%s" % stored).text
    assert "trained detector was not used" in body
    assert "no trained checkpoint was present" in body
    assert "threshold baseline" in body


def test_a_clean_scene_reads_as_a_finding_and_not_a_failure(client, app_config):
    """A clean scene is the other early exit: the pipeline stops before drift and
    attribution, so the downstream sections have to say they were never reached
    rather than render as empty tables."""
    from app.jobs import store

    doc = _job()
    doc["detection"]["polygons"] = []
    doc["drift"] = None
    doc["attribution"] = {"suspects": [], "sensitivity": []}
    doc.pop("ablation")
    doc["case_quality"] = {
        "overall": "LOW", "label": "Case evidence quality",
        "conclusions": ["No polygon exceeded the reporting area threshold."],
        "ordered_factors": [],
        "factors": {},
        "safe_fail": {"state": "NO OIL DETECTED",
                      "headline": "NO OIL DETECTED",
                      "detail": "No polygon exceeded the minimum reporting area.",
                      "missing_input": "oil above the reporting area threshold",
                      "stopped_before": ["drift", "attribution"]},
    }
    store.save("job_report", doc)
    try:
        body = client.get("/api/report/job_report").text
        assert "A clean scene is a finding" in body
        assert "NO OIL DETECTED" in body
        # And the stopped-before line, so nothing downstream is quoted.
        assert "stopped before" in body.lower()
        assert "drift, attribution" in body
        # No candidate may be named on a clean scene.
        assert "MT MERIDIAN" not in body
    finally:
        p = store.path_for("job_report")
        if p.exists():
            p.unlink()


def test_a_case_with_no_candidate_does_not_name_the_least_unlikely_vessel(client,
                                                                           app_config):
    """The live pipeline produces exactly this state on a scene with a slick but
    no AIS traffic in the origin window, so this is the common path, not an
    edge case.

    The report withholds the sensitivity and ablation tables, because those name
    a vessel and section 1 says no vessel can be named. Printing both is the
    contradiction this project exists to prevent."""
    from app.jobs import store

    doc = _job()
    doc["attribution"] = {"suspects": [], "sensitivity": doc["attribution"]["sensitivity"]}
    doc["case_quality"] = dict(doc["case_quality"], safe_fail={
        "state": "NO RELIABLE VESSEL CANDIDATE",
        "detail": "No track was close enough to score.",
        "missing_input": "an AIS track intersecting the probable source region",
        "stopped_before": []})
    store.save("job_report", doc)
    try:
        body = client.get("/api/report/job_report").text
        assert "MT MERIDIAN" not in body, (
            "a vessel is named in a report whose summary says no candidate exists")
        assert "No vessel is put forward" in body
        assert "reports nothing rather than naming" in body
        assert "no candidate ranking" in body
        # And the tables really are gone, not merely the name.
        assert "Distance to origin only" not in body
        assert "Origin shifted 5 km" not in body
    finally:
        p = store.path_for("job_report")
        if p.exists():
            p.unlink()


def test_the_score_is_never_presented_as_a_probability(client, stored):
    body = client.get("/api/report/%s" % stored).text
    assert "uncalibrated evidence index" in body
    assert "not a probability" in body
    # The calibration section has to admit the figure cannot be computed yet.
    assert "not calibrated" in body
    assert "at least 20 are required" in body


def test_the_layer_tag_marks_which_claims_are_observations(client, stored):
    body = client.get("/api/report/%s" % stored).text
    assert "observed claim" in body
    assert "inferred claim" in body
    assert "attributed claim" in body


def test_a_case_with_no_provenance_record_says_it_is_unverified(client, stored,
                                                                app_config):
    from app.jobs import store

    doc = _job()
    doc["provenance"] = {}
    store.save("job_report", doc)
    try:
        body = client.get("/api/report/%s" % stored).text
        assert "unverified" in body
    finally:
        p = store.path_for("job_report")
        if p.exists():
            p.unlink()


def test_missing_studies_degrade_to_a_stated_absence(client, app_config):
    """A case written before the studies existed must say so, not render blank
    tables that read as "no effect found"."""
    from app.jobs import store

    doc = _job()
    doc.pop("ablation")
    doc.pop("calibration")
    doc["attribution"]["sensitivity"] = []
    store.save("job_report", doc)
    try:
        body = client.get("/api/report/job_report").text
        assert "No counterfactual sensitivity study" in body
        # Still renders all seven sections.
        for title in SECTIONS:
            assert title in body
    finally:
        p = store.path_for("job_report")
        if p.exists():
            p.unlink()


def test_the_report_escapes_text_it_did_not_write(client, app_config):
    """Vessel names and scene titles come from external data. Unescaped, a
    scene named `<script>` executes in the reader's browser."""
    from app.jobs import store

    doc = _job()
    doc["attribution"]["suspects"][0]["name"] = "<script>alert(1)</script>"
    store.save("job_report", doc)
    try:
        body = client.get("/api/report/job_report").text
        assert "<script>alert(1)</script>" not in body
        assert "&lt;script&gt;" in body
    finally:
        p = store.path_for("job_report")
        if p.exists():
            p.unlink()


def test_the_report_contains_no_em_dashes(client, stored):
    assert "—" not in client.get("/api/report/%s" % stored).text
    assert "—" not in client.get("/api/report/%s/text" % stored).text
