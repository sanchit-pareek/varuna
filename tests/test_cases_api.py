"""The case API is an investigation surface, not a dump of the job document.

These tests pin the two properties that make it safe to build a console on top
of it: every endpoint resolves against a stored case, and a case that could not
be completed says so in the same place a result would appear. A panel that
quietly omits an unevaluable scenario is worse than no panel, because absence
reads as "this did not matter".
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


def _job(job_id="job_case_api", **over):
    """A minimal but realistic case document.

    Deliberately hand-written rather than produced by the pipeline: the point
    is that the API reads whatever the pipeline wrote, including a document
    from before a field existed. `missing_study` covers that case.
    """
    doc = {
        "job_id": job_id,
        "created": "2026-01-01T00:00:00Z",
        "status": "ok",
        "input": {"scene_id": "demo_scene"},
        "scene": {"id": "demo_scene", "t_sat": "2026-01-01T00:00:00Z",
                  "source": "Sentinel-1 GRD via Copernicus", "license": "open"},
        "detection": {
            "polygons": [{"type": "Feature",
                          "geometry": {"type": "Polygon",
                                       "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                          "properties": {"polygon_id": "OIL01", "area_km2": 4.2,
                                         "elongation": 3.4, "shape_diagnostics": {
                                             "elongation": 3.4, "solidity": 0.31,
                                             "irregularity": 0.42, "n_components": 1}}}],
            "lookalikes": [],
            "metrics": {"detector": "unet", "oil_pixels": 900, "oil_area_km2": 4.2,
                        "lookalike_polygons_found": 3,
                        "detector_metadata": {"name": "unet",
                                              "is_trained_detector": True}},
            "eo": {"available": True, "status": "indeterminate",
                   "acquired": "2026-01-02T00:00:00Z",
                   "time_delta": {"hours": 24.0, "valid_pixel_fraction": 0.4},
                   "counts": {"consistent": 12, "inconsistent": 3}},
        },
        "drift": {
            "origin": {"lon": 1.5, "lat": 2.5, "t": "2025-12-31T18:00:00Z",
                       "spread_km": 3.0,
                       "release_time_interval": {"start": "2025-12-31T17:00:00Z",
                                                 "end": "2025-12-31T19:00:00Z"}},
            "metocean": {"source": "cached cube"}, "metocean_covers_scene": True,
        },
        "attribution": {
            "suspects": [{
                "mmsi": 111222333, "rank": 1, "name": "MT MERIDIAN", "type": "tanker",
                "score": 62.0, "score_before_confidence": 70.0, "confidence": 0.88,
                "counter_evidence_count": 2,
                "counter_evidence": [
                    {"category": "counter_evidence", "kind": "negative",
                     "text": "AIS gap of 90 min across the origin window."},
                    {"category": "counter_evidence", "kind": "negative",
                     "text": "Track crosses 4.1 km from the origin estimate."}],
                "evidence": {"opportunity_score": 70.0, "opportunity_level": "HIGH",
                             "evidence_score": 60.0, "evidence_level": "MEDIUM",
                             "positive_evidence": [
                                 {"category": "opportunity",
                                  "text": "Present in the origin zone during the window."}],
                             "data_quality": [{"kind": "ais",
                                               "text": "41% of track dead reckoned."}]},
                "region": {"inside_50": True, "inside_90": True,
                           "observed_track_fraction": 0.59},
                "release_window": {"applicable": True, "inside": True},
                "detail": {"closest_approach_utc": "2025-12-31T18:10:00Z",
                           "origin_distance_km": 0.8},
            }],
            # The real counterfactual shape, taken from score.py: one record per
            # candidate, each carrying a baseline and a list of scenarios. The
            # scenario that could not be evaluated is the one that matters most
            # here, so it is present rather than omitted.
            "sensitivity": [{
                "candidate_id": 111222333, "candidate_name": "MT MERIDIAN",
                "baseline_score": 62.0, "baseline_rank": 1,
                "scenarios": [
                    {"name": "origin_shift", "label": "Origin shifted 5 km",
                     "question": "If the origin estimate is 5 km off, does the rank hold?",
                     "applicable": True, "score": 51.0, "rank": 1, "delta": -11.0},
                    {"name": "wider_window", "label": "Release window widened to 6 h",
                     "question": "Does the rank hold if the release window is wider?",
                     "applicable": False,
                     "reason": "The release interval came from the drift model, not "
                               "from an independent measurement."},
                ],
                "stability": "SENSITIVE", "max_score_delta": 11.0,
                "rank_changes": 0, "primary_dependency": "origin_accuracy",
                "note": "The rank held, but the score moved 11 points.",
            }],
            "scoring": {"weights": {"prox": 0.4, "time": 0.2}},
        },
        "ablation": {
            "available": True, "candidates": 1,
            "reference": {"key": "full", "step": "G",
                          "top1_mmsi": 111222333, "order": [111222333]},
            "ladder": [
                {"step": "A", "key": "naive", "label": "Distance to origin only",
                 "includes": [], "excludes": ["time", "envelope", "confidence"],
                 "question": "How far from the origin was the vessel?",
                 "description": "The naive method.",
                 "status": "computed", "top1_mmsi": 111222333,
                 "top1_name": "MT MERIDIAN", "order": [111222333], "n_candidates": 1,
                 "agreement": {"top1_match": True, "top3_overlap": 1,
                               "top3_union": 1, "top3_jaccard": 1.0},
                 "spearman_vs_full": 1.0, "mean_abs_score_delta": 0.0},
                {"step": "F", "key": "envelope", "label": "Spatial envelope containment",
                 "includes": ["time", "envelope"], "excludes": ["confidence"],
                 "question": "Was the track inside the slick envelope?",
                 "description": "Adds envelope containment.",
                 "status": "not_available",
                 "reason": "the raw tracks or slick geometry for this case were not "
                           "supplied to the ablation"},
            ],
            "top_n_candidates_considered": 1, "n_not_available": 1,
            "statement": "Agreement with the full system.",
            "caveat": "With fewer than about five candidates these agreements are not "
                      "statistically meaningful.",
        },
        "calibration": {"n": 0, "min_samples": 20, "estimable": False,
                        "brier": None, "brier_skill": None, "ece": None, "auc": None},
        "case_quality": {
            "overall": "MEDIUM",
            "label": "Case evidence quality",
            "conclusions": ["Two inputs are absent, so no origin is claimed."],
            # The real shape: `ordered_factors` is a weakest-first list of
            # stage keys and `factors` holds the detail for each. Written the
            # real way round here so the renderers are tested against the
            # document the pipeline actually writes.
            "ordered_factors": ["optical", "sar_detection"],
            "factors": {
                "optical": {"stage": "optical", "label": "Optical corroboration",
                            "quantity": 0.4, "level": "LOW",
                            "measurement": {"valid_pixel_fraction": 0.4},
                            "note": "Chip present but mostly cloudy."},
                "sar_detection": {"stage": "sar_detection", "label": "SAR detection",
                                  "quantity": 0.9, "level": "HIGH",
                                  "measurement": {}, "note": "Trained detector, no look-alike."},
            },
            "uncertainty_chain": {
                "overall": "MEDIUM",
                "stages": [
                    {"stage": "sar_detection", "label": "SAR detection",
                     "quantity": 0.9, "level": "HIGH", "measurement": {},
                     "note": "Trained detector over a scene containing look-alike targets."},
                    {"stage": "optical", "label": "Optical corroboration",
                     "quantity": 0.4, "level": "LOW",
                     "measurement": {"valid_pixel_fraction": 0.4},
                     "note": "Chip present but mostly cloudy."},
                ],
            },
            "safe_fail": {"state": "NO RELIABLE VESSEL CANDIDATE",
                          "detail": "An origin was estimated and no vessel is a "
                                    "defensible candidate."},
        },
        "provenance": {"case_hash": "abc123", "inputs": [{"name": "sar"}],
                       "inputs_missing": [{"name": "optical", "reason": "cloud"}]},
    }
    doc.update(over)
    return doc


@pytest.fixture
def stored_case(app_config):
    from app.jobs import store

    store.save("job_case_api", _job())
    yield "job_case_api"
    path = store.path_for("job_case_api")
    if path.exists():
        path.unlink()


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


def test_case_index_reports_quality_and_not_just_a_candidate_count(client, stored_case):
    """A list showing only "3 candidates" invites reading an empty case as a
    strong one, so the band and the safe-fail state travel with every row."""
    r = client.get("/api/cases")
    assert r.status_code == 200
    rows = r.json()["cases"]
    mine = [c for c in rows if c["job_id"] == stored_case]
    assert mine, "the stored case is not in the index"
    row = mine[0]
    assert row["case_quality"] == "MEDIUM"
    assert row["safe_fail_state"] == "NO RELIABLE VESSEL CANDIDATE"
    assert row["top_candidate"]["mmsi"] == 111222333
    assert row["top_candidate"]["counter_evidence_count"] == 2


def test_case_header_exposes_the_finding_and_its_state(client, stored_case):
    r = client.get("/api/cases/%s" % stored_case)
    assert r.status_code == 200
    body = r.json()
    assert body["counts"] == {"oil_polygons": 1, "lookalike_polygons": 0,
                              "candidates": 1}
    assert body["case_quality"]["overall"] == "MEDIUM"
    assert body["safe_fail"]["state"] == "NO RELIABLE VESSEL CANDIDATE"
    assert body["detector"]["is_trained_detector"] is True


def test_evidence_is_grouped_per_candidate_with_its_objections(client, stored_case):
    """Grouped, not flat: a flat list is the most likely way a long reason
    column gets paired with the wrong vessel."""
    r = client.get("/api/cases/%s/evidence" % stored_case)
    assert r.status_code == 200
    c = r.json()["candidates"][0]
    assert c["mmsi"] == 111222333
    assert c["why_this_vessel"]["opportunity_level"] == "HIGH"
    assert c["why_not"]["count"] == 2
    assert len(c["why_not"]["items"]) == 2
    assert c["origin_region"]["inside_90"] is True


def test_the_timeline_tags_observations_inferences_and_conclusions(client, stored_case):
    """The tag is the panel's reason to exist. A modelled origin hour and a
    radar acquisition are both times, and without the tag the first gets
    quoted as if it were the second."""
    r = client.get("/api/cases/%s/timeline" % stored_case)
    assert r.status_code == 200
    kinds = {e["kind"] for e in r.json()["events"]}
    assert {"observed", "inferred", "attributed"} <= kinds
    assert "not observations" in r.json()["warning"]


def test_the_chain_keeps_the_three_layers_apart(client, stored_case):
    r = client.get("/api/cases/%s/chain" % stored_case)
    assert r.status_code == 200
    body = r.json()
    assert any("Sentinel-1" in s["statement"] for s in body["observed"])
    assert any("Modelled" in s["statement"] for s in body["inferred"])
    assert any("investigation" in s["statement"] or "lead" in s["statement"]
               for s in body["attributed"])
    # The attributed layer must carry the objection count, never a bare name.
    assert "counter-evidence" in body["attributed"][0]["statement"]


def test_ablation_and_sensitivity_come_back_in_the_pipeline_shape(client, stored_case):
    """The console reads these two straight off the document, so if the API
    reshaped them the UI would be the only place the shape was known.

    Pinned against the real writer in app/ablation.py and
    app/ais/score.py: `ladder` with per-rung `status`, and per-candidate
    records with nested `scenarios`.
    """
    r = client.get("/api/cases/%s/ablation" % stored_case)
    assert r.status_code == 200
    # The study keys sit at the top level, not nested under "ablation".
    study = r.json()
    assert study["available"] is True
    assert [r["step"] for r in study["ladder"]] == ["A", "F"]
    assert study["ladder"][0]["status"] == "computed"
    assert study["ladder"][0]["spearman_vs_full"] == 1.0
    # The unevaluable rung keeps its reason, so a UI can show why it is empty.
    assert study["ladder"][1]["status"] == "not_available"
    assert "not supplied" in study["ladder"][1]["reason"]

    r = client.get("/api/cases/%s/sensitivity" % stored_case)
    assert r.status_code == 200
    rows = r.json()["sensitivity"]
    assert len(rows) == 1
    assert rows[0]["baseline_rank"] == 1
    assert rows[0]["stability"] == "SENSITIVE"
    assert [s["name"] for s in rows[0]["scenarios"]] == ["origin_shift", "wider_window"]
    # The unevaluable scenario is the one that must not go missing.
    assert rows[0]["scenarios"][1]["applicable"] is False
    assert "independent measurement" in rows[0]["scenarios"][1]["reason"]


def test_calibration_refuses_to_claim_a_score_it_cannot_support(client, stored_case):
    """No outcome labels means no calibration figure, and that is the correct
    answer rather than a gap in the feature."""
    r = client.get("/api/cases/%s/calibration" % stored_case)
    assert r.status_code == 200
    body = r.json()
    assert body["estimable"] is False
    assert body["n"] == 0
    assert body["min_samples"] == 20


def test_uncertainty_reports_certainty_not_an_uncertainty_magnitude(client, stored_case):
    """`quantity` is certainty, where 1.0 means the stage's inputs are in good
    order. A UI that labelled it "uncertainty" would draw a full bar for a
    well-run stage and empty one for a stage that could not run at all, which is
    exactly backwards."""
    r = client.get("/api/cases/%s/uncertainty" % stored_case)
    assert r.status_code == 200
    stages = {s["stage"]: s for s in r.json()["uncertainty_chain"]["stages"]}
    assert stages["sar_detection"]["quantity"] == 0.9
    assert stages["sar_detection"]["level"] == "HIGH"
    assert stages["optical"]["quantity"] == 0.4
    assert stages["optical"]["level"] == "LOW"


def test_weakest_factors_resolve_to_a_level_and_a_measurement(client, stored_case):
    """`ordered_factors` is only a list of stage keys. A consumer reading it
    alone gets bare internal names with no level beside them, which is the one
    thing the panel exists to show, so the detail has to be reachable."""
    r = client.get("/api/cases/%s" % stored_case)
    q = r.json()["case_quality"]
    assert q["ordered_factors"][0] == "optical"
    assert q["factors"]["optical"]["level"] == "LOW"
    assert q["factors"]["optical"]["measurement"]["valid_pixel_fraction"] == 0.4


def test_unknown_case_is_a_404_not_an_empty_success(client, stored_case):
    """An empty 200 is the worst possible answer here: the console would render
    a blank case with no indication that the id was wrong."""
    for suffix in ("", "/uncertainty", "/sensitivity", "/evidence", "/timeline",
                   "/chain", "/calibration", "/ablation"):
        r = client.get("/api/cases/nope_not_here" + suffix)
        assert r.status_code == 404, "%s returned %s" % (suffix, r.status_code)


def test_the_index_survives_a_corrupt_job_document(client, stored_case, app_config):
    """One unreadable file must not take the whole case index down with it."""
    from pathlib import Path

    broken = Path(app_config.JOBS_DIR) / "job_broken.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{ this is not json", encoding="utf-8")
    try:
        r = client.get("/api/cases")
        assert r.status_code == 200
        assert any(c["job_id"] == stored_case for c in r.json()["cases"])
    finally:
        broken.unlink()


def test_the_console_only_ever_reads_documents_it_did_not_rewrite(client, stored_case):
    """The API is a reader. It must not normalise, reformat or otherwise
    touch a stored case, or a second visit to a case would stop being the same
    evidence as the first."""
    from app.jobs import store

    before = json.loads(store.path_for(stored_case).read_text(encoding="utf-8"))
    for suffix in ("", "/evidence", "/timeline", "/chain", "/calibration"):
        client.get("/api/cases/%s%s" % (stored_case, suffix))
    after = json.loads(store.path_for(stored_case).read_text(encoding="utf-8"))
    assert before == after
