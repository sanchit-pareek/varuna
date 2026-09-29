"""POST /api/drift."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from .. import config, pipeline
from ..jobs import store as job_store
from ..schemas import DriftRequest

router = APIRouter()


def _ring_from_geojson(feature: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
    if not feature:
        return None
    geom = feature.get("geometry", feature)
    if not geom or geom.get("type") != "Polygon":
        return None
    coords = geom.get("coordinates") or []
    if not coords:
        return None
    return [(float(x), float(y)) for x, y in coords[0]]


@router.post("/api/drift")
def drift(req: DriftRequest) -> Dict[str, Any]:
    ring: Optional[List[Tuple[float, float]]] = None
    centroid: Optional[Tuple[float, float]] = None
    t_sat = req.t_sat
    scene_id = req.scene_id or "probe"

    if req.job_id:
        doc = job_store.load(req.job_id)
        if doc is None:
            raise HTTPException(404, "unknown job_id %r" % req.job_id)
        polys = (doc.get("detection") or {}).get("polygons") or []
        if not polys:
            raise HTTPException(409, "job %s has no oil polygon to drift" % req.job_id)
        ring = _ring_from_geojson(polys[0])
        props = polys[0]["properties"]
        centroid = (float(props["centroid_lon"]), float(props["centroid_lat"]))
        t_sat = t_sat or (doc.get("input") or {}).get("t_sat") or (doc.get("scene") or {}).get("t_sat")
        scene_id = req.scene_id or (doc.get("input") or {}).get("scene_id") or scene_id

    if req.polygon:
        ring = [(float(x), float(y)) for x, y in req.polygon]
    if req.centroid:
        centroid = (float(req.centroid[0]), float(req.centroid[1]))
    if centroid is None and ring:
        centroid = (sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring))

    if centroid is None:
        raise HTTPException(400, "provide job_id, centroid or polygon")
    if not t_sat:
        raise HTTPException(400, "t_sat is required when job_id is not given")

    out = pipeline.run_drift(
        ring, centroid, pipeline._utc(t_sat),
        hindcast_hours=int(req.hours_back),
        forecast_hours=int(req.hours_fwd),
        ensemble_n=int(req.ensemble_n),
        scene_id=scene_id,
    )
    trace = out.pop("trace")
    out.pop("_runs", None)
    out["trace"] = trace.to_list()
    out["config"] = config.as_dict()
    return out
