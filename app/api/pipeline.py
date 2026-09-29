"""POST /api/run, POST /api/demo/inject, and the job endpoints."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from .. import pipeline as core
from ..jobs import store as job_store
from ..schemas import InjectRequest, JobSummary, RunRequest

router = APIRouter()


@router.post("/api/run")
def run(req: RunRequest) -> Dict[str, Any]:
    if not req.scene_id:
        raise HTTPException(400, "scene_id is required")
    try:
        return core.run(
            scene_id=req.scene_id,
            t_sat=req.t_sat,
            hindcast_hours=req.hindcast_hours,
            forecast_hours=req.forecast_hours,
            search_radius_km=req.search_radius_km,
            origin_window_hours=req.origin_window_hours,
            ensemble_n=req.ensemble_n,
            prefer_model=req.prefer_model,
            threshold_db=req.threshold_db,
            top_n=req.top_n,
            render_overlays=req.render_overlays,
            job_id=req.job_id,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/api/jobs/{job_id}/progress")
def job_progress(job_id: str) -> Dict[str, Any]:
    """Which step a run is in right now.

    /api/run is synchronous and a full run is tens of seconds, nearly all of it
    inside DETECT. The UI names the job up front and polls this so the wait
    shows position rather than just motion. Returns running=false once the run
    has finished, whether it succeeded or not.
    """
    p = job_store.progress(job_id)
    if p is None:
        return {"job_id": job_id, "running": False,
                "finished": job_store.path_for(job_id).exists()}
    return {"running": True, "finished": False, **p}


@router.post("/api/demo/inject")
def inject(req: InjectRequest) -> Dict[str, Any]:
    return core.run_probe(
        lat=req.lat, lon=req.lon, t_sat=req.t_sat,
        slick_radius_km=req.slick_radius_km,
        hindcast_hours=req.hindcast_hours, forecast_hours=req.forecast_hours,
        radius_km=req.radius_km, window_h=req.window_h, top_n=req.top_n,
    )


@router.get("/api/jobs", response_model=List[JobSummary])
def jobs(limit: int = 50) -> List[JobSummary]:
    return [JobSummary(**j) for j in job_store.listing(limit=limit)]


@router.get("/api/jobs/{job_id}")
def job(job_id: str) -> Dict[str, Any]:
    doc = job_store.load(job_id)
    if doc is None:
        raise HTTPException(404, "unknown job_id %r" % job_id)
    return doc


@router.get("/api/jobs/{job_id}/geojson")
def job_geojson(job_id: str) -> Dict[str, Any]:
    """One FeatureCollection with every layer, for export into any GIS."""
    doc = job_store.load(job_id)
    if doc is None:
        raise HTTPException(404, "unknown job_id %r" % job_id)

    features: List[Dict[str, Any]] = []
    det = doc.get("detection") or {}
    features.extend(det.get("polygons") or [])
    features.extend(det.get("lookalikes") or [])

    drift = doc.get("drift") or {}
    for key in ("origin_zone", "cone_back", "cone_fwd"):
        f = drift.get(key)
        if f and f.get("geometry"):
            features.append(f)
    for key in ("hindcast_track", "forecast_track"):
        f = drift.get(key)
        if f and f.get("geometry"):
            features.append(f)

    origin = drift.get("origin")
    if origin:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [origin["lon"], origin["lat"]]},
            "properties": {"name": "origin_estimate", "t": origin["t"],
                           "spread_km": origin["spread_km"],
                           "note": "zone centre, not a point fix"},
        })

    for s in (doc.get("attribution") or {}).get("suspects", []):
        for f in (s.get("track") or {}).get("geojson", {}).get("features", []):
            props = dict(f.get("properties") or {})
            props.update({"rank": s["rank"], "score": s["score"], "type": s["type"],
                          "reasons": ", ".join(s["reasons"])})
            features.append({**f, "properties": props})

    return {
        "type": "FeatureCollection",
        "properties": {
            "job_id": job_id,
            "created": doc.get("created"),
            "scene_id": (doc.get("input") or {}).get("scene_id"),
            "status": doc.get("status"),
            "warnings": doc.get("warnings", []),
        },
        "features": features,
    }
