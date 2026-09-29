"""Offline attribution benchmarking with strict evidence-origin separation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import config


def simulation_truth(scene_id: str) -> Optional[Dict[str, Any]]:
    """Load simulator labels only from the isolated ground-truth artefact.

    This module is used by development evaluation routes and scripts, never by
    detection, drift, candidate filtering, or scoring.
    """
    path = Path(config.AIS_DIR) / ("ground_truth_%s.json" % scene_id)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        mmsi = raw.get("scenario_vessel_mmsi")
        if mmsi is None:
            return None
        return {
            "scene_id": scene_id,
            "source_mmsi": int(mmsi),
            "evidence_mode": "simulated",
            "label_source": "synthetic AIS scenario ground truth",
        }
    except (OSError, ValueError, TypeError):
        return None


def evaluate_ranking(ranked: Iterable[Dict[str, Any]], truth: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate top-k and reciprocal-rank metrics for one labelled case."""
    ranked = list(ranked)
    source = int(truth["source_mmsi"])
    position = next((index for index, candidate in enumerate(ranked, start=1)
                     if int(candidate.get("mmsi", -1)) == source), None)
    return {
        "scene_id": truth.get("scene_id"),
        "evidence_mode": truth.get("evidence_mode", "unknown"),
        "label_source": truth.get("label_source", "unknown"),
        "source_mmsi": source,
        "rank": position,
        "top_1": position == 1,
        "top_3": bool(position and position <= 3),
        "top_5": bool(position and position <= 5),
        "mrr": 0.0 if position is None else round(1.0 / position, 6),
        "false_attribution": bool(ranked and position != 1),
        "no_evidence": not ranked,
        "eligible_for_real_world_validation": truth.get("evidence_mode") == "real",
    }


def aggregate(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate only like-with-like evidence; simulated labels stay separate."""
    groups: Dict[str, List[Dict[str, Any]]] = {"real": [], "simulated": [], "unknown": []}
    for record in records:
        groups.setdefault(str(record.get("evidence_mode", "unknown")), []).append(record)

    def metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(items)
        if not count:
            return {"cases": 0, "top_1": None, "top_3": None, "top_5": None,
                    "mrr": None, "false_attribution_rate": None, "no_evidence_rate": None}
        return {
            "cases": count,
            "top_1": round(sum(bool(item["top_1"]) for item in items) / count, 4),
            "top_3": round(sum(bool(item["top_3"]) for item in items) / count, 4),
            "top_5": round(sum(bool(item["top_5"]) for item in items) / count, 4),
            "mrr": round(sum(float(item["mrr"]) for item in items) / count, 4),
            "false_attribution_rate": round(sum(bool(item["false_attribution"]) for item in items) / count, 4),
            "no_evidence_rate": round(sum(bool(item["no_evidence"]) for item in items) / count, 4),
        }

    return {
        "simulation_metrics": metrics(groups["simulated"]),
        "real_world_metrics": metrics(groups["real"]),
        "unknown_label_metrics": metrics(groups["unknown"]),
        "caveat": "Simulated AIS labels are reported separately and never counted as real-world validation.",
    }


def evaluate_job(document: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a saved job only when it has an explicit isolated label source."""
    scene_id = str((document.get("input") or {}).get("scene_id") or "")
    truth = simulation_truth(scene_id)
    if truth is None:
        return {
            "status": "unavailable",
            "scene_id": scene_id or None,
            "reason": "No confirmed source-vessel label is available for this case.",
            "evidence_mode": "unknown",
        }
    record = evaluate_ranking((document.get("attribution") or {}).get("suspects") or [], truth)
    return {"status": "ok", "record": record, "summary": aggregate([record])}
