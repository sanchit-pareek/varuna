"""Case-level quality and provenance must remain distinct and reproducible."""
from __future__ import annotations


def _doc(eo=None):
    return {
        "job_id": "job_example",
        "scene": {"id": "scene", "source": "Sentinel-1", "license": "open", "t_sat": "2024-01-01T00:00:00Z"},
        "input": {"scene_id": "scene", "ensemble_n": 50},
        "config": {"seed": 7},
        "detection": {"metrics": {"detector": "unet", "lookalike_polygons_found": 0,
                                  "oil_pixels": 900},
                      "eo": eo if eo is not None else {"available": True,
                                                       "verdicts": [{"verdict": "consistent"}]}},
        "drift": {"metocean": {"source": "cached", "synthetic": False, "has_currents": True},
                  "metocean_covers_scene": True, "origin": {"spread_km": 5.0},
                  "physics": {"dt_seconds": 3600}},
        "attribution": {"sources_used": {"marinecadastre": 2}, "store": {"rows": 4}},
    }


def test_case_quality_is_not_candidate_confidence():
    from app.pipeline import _case_quality

    quality = _case_quality(_doc()["detection"], _doc()["drift"], _doc()["attribution"])
    assert quality["overall"] == "MEDIUM"
    assert "candidate ranking" in quality["label"]
    assert quality["factors"]["origin_certainty"]["level"] == "MEDIUM"
    # Every band must carry the measurement it was derived from, so an analyst
    # can argue with the band instead of trusting it.
    for stage, factor in quality["factors"].items():
        assert factor["level"] in ("LOW", "MEDIUM", "HIGH"), stage
        assert factor.get("reason"), stage
    assert quality["uncertainty_chain"]["bands_are_not_probabilities"] is True
    assert len(quality["uncertainty_chain"]["stages"]) >= 5


def test_case_is_capped_by_its_weakest_material_evidence():
    """Optical availability alone is not corroboration.

    A scene claiming cached optical data but producing no verdicts has
    corroborated nothing, so it must not lift the case to MEDIUM. The old
    implementation banded it MEDIUM on the strength of `available: true`.
    """
    from app.pipeline import _case_quality

    empty_eo = {"available": True, "verdicts": []}
    quality = _case_quality(_doc(eo=empty_eo)["detection"], _doc()["drift"],
                            _doc()["attribution"])
    assert quality["factors"]["optical_corroboration"]["level"] == "LOW"
    assert quality["overall"] == "LOW"

    # Every other stage that caps the case is at least MEDIUM, so the empty
    # optical verdict set is what holds the case down.
    from app.uncertainty import CAP_STAGES

    others = {k: v["level"] for k, v in quality["factors"].items() if k in CAP_STAGES}
    assert others["optical_corroboration"] == "LOW"
    assert all(level in ("MEDIUM", "HIGH")
               for stage, level in others.items() if stage != "optical_corroboration")


def test_lookalike_risk_is_a_hazard_label_not_a_quality_input():
    """Many look-alikes is a bad scene, so HIGH risk must not raise the case."""
    from app.pipeline import _case_quality

    det = _doc()["detection"]
    det["metrics"] = dict(det["metrics"], lookalike_polygons_found=25,
                          lookalike_area_km2=400.0, oil_area_km2=20.0)
    quality = _case_quality(det, _doc()["drift"], _doc()["attribution"])
    assert quality["factors"]["lookalike_risk"]["level"] == "HIGH"
    assert quality["overall"] == "MEDIUM"
    assert any("look-alike" in c for c in quality["conclusions"])


def test_safe_fail_states_are_explicit():
    from app import uncertainty

    assert uncertainty.safe_fail_state({"polygons": [], "metrics": {}}, None, None)["state"] \
        == uncertainty.NO_OIL
    assert uncertainty.safe_fail_state(
        {"polygons": [{"a": 1}], "metrics": {}}, None, None)["state"] \
        == uncertainty.INSUFFICIENT_EVIDENCE
    assert uncertainty.safe_fail_state(
        {"polygons": [{"a": 1}], "metrics": {}}, {"origin": {}}, {})["state"] \
        == uncertainty.NO_CANDIDATE
    # A case that does conclude reports no safe-fail state at all.
    assert uncertainty.safe_fail_state(
        {"polygons": [{"a": 1}], "metrics": {}}, {"origin": {}},
        {"suspects": [{"rank": 1}]}) is None


def test_case_hash_is_stable_for_identical_scientific_inputs():
    from app.pipeline import _provenance

    one = _provenance(_doc())
    two = _provenance(_doc())
    assert one["case_hash"] == two["case_hash"]
    assert len(one["case_hash"]) == 64
