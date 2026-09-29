"""Evaluation must never present simulator labels as real-world validation."""
from __future__ import annotations


def _truth(mode="simulated"):
    return {"scene_id": "case", "source_mmsi": 42, "evidence_mode": mode,
            "label_source": "test label"}


def test_ranking_metrics_report_the_known_source_position():
    from app.evaluation import evaluate_ranking

    record = evaluate_ranking([{"mmsi": 9}, {"mmsi": 42}], _truth())
    assert record["rank"] == 2
    assert record["top_1"] is False
    assert record["top_3"] is True
    assert record["mrr"] == 0.5


def test_simulated_labels_do_not_pollute_real_world_metrics():
    from app.evaluation import aggregate, evaluate_ranking

    simulated = evaluate_ranking([{"mmsi": 42}], _truth("simulated"))
    report = aggregate([simulated])
    assert report["simulation_metrics"]["top_1"] == 1.0
    assert report["real_world_metrics"]["cases"] == 0
    assert "never counted" in report["caveat"]
