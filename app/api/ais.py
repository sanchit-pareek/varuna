"""POST /api/attribute and the AIS inspection endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query

from .. import config, pipeline
from ..ais import ingest as ais_ingest, interpolate as ais_interp, score as ais_score
from ..jobs import store as job_store
from ..schemas import AttributeRequest

router = APIRouter()


def _ring(geo: Optional[Dict[str, Any]]) -> Optional[List[Tuple[float, float]]]:
    if not geo:
        return None
    geom = geo.get("geometry", geo)
    if not geom or geom.get("type") != "Polygon":
        return None
    coords = geom.get("coordinates") or []
    return [(float(x), float(y)) for x, y in coords[0]] if coords else None


@router.post("/api/attribute")
def attribute(req: AttributeRequest) -> Dict[str, Any]:
    ring = _ring(req.origin_geojson)
    origin = tuple(req.origin) if req.origin else None
    t_origin = req.t_origin
    slick = tuple(req.slick_centroid) if req.slick_centroid else None

    if req.job_id:
        doc = job_store.load(req.job_id)
        if doc is None:
            raise HTTPException(404, "unknown job_id %r" % req.job_id)
        d = doc.get("drift") or {}
        if not d:
            raise HTTPException(409, "job %s has no drift result" % req.job_id)
        ring = ring or _ring(d.get("origin_zone"))
        origin = origin or (d["origin"]["lon"], d["origin"]["lat"])
        t_origin = t_origin or d["origin"]["t"]
        if slick is None:
            p = doc.get("primary_polygon")
            if p:
                slick = (p["properties"]["centroid_lon"], p["properties"]["centroid_lat"])

    if origin is None or t_origin is None:
        raise HTTPException(400, "provide job_id, or origin plus t_origin")
    if slick is None:
        slick = origin
    if ring is None:
        from ..geo.geometry import buffer_ring_km

        ring = buffer_ring_km([(float(origin[0]), float(origin[1]))], config.ORIGIN_BUFFER_KM)

    out = pipeline.run_attribution(
        ring, float(origin[0]), float(origin[1]), pipeline._utc(t_origin),
        float(slick[0]), float(slick[1]),
        radius_km=float(req.radius_km), window_h=float(req.window_h), top_n=int(req.top_n),
    )
    trace = out.pop("trace")
    out["trace"] = trace.to_list()
    return out


@router.get("/api/ais/track/{mmsi}")
def track(mmsi: int, t_start: str = Query(...), t_end: str = Query(...),
          step_seconds: int = Query(60, ge=10, le=3600)) -> Dict[str, Any]:
    """Full reconstructed track for one vessel over a window."""
    t0 = int(pipeline._utc(t_start).timestamp())
    t1 = int(pipeline._utc(t_end).timestamp())
    conn = ais_ingest.connect()
    try:
        rows = ais_ingest.query_tracks(conn, [int(mmsi)], t0, t1)
    finally:
        conn.close()
    if int(mmsi) not in rows:
        raise HTTPException(404, "no positions for MMSI %s in that window" % mmsi)
    tr = ais_interp.resample(rows[int(mmsi)], t0, t1, step_seconds=step_seconds)
    if tr is None:
        raise HTTPException(404, "not enough positions to reconstruct a track")
    return {
        "mmsi": tr.mmsi,
        "name": tr.name,
        "type": ais_score.vessel_types.decode(tr.vessel_type_raw),
        "raw_positions": tr.raw_count,
        "gaps": [g.to_dict() for g in tr.gaps],
        "geojson": tr.to_geojson(),
        "samples": tr.samples(stride=1),
    }


@router.get("/api/ais/window")
def window(t_start: str = Query(...), t_end: str = Query(...),
           west: float = Query(...), south: float = Query(...),
           east: float = Query(...), north: float = Query(...),
           step_seconds: int = Query(300, ge=30, le=3600)) -> Dict[str, Any]:
    """Every vessel in a box and window, for the time slider playback layer."""
    t0 = int(pipeline._utc(t_start).timestamp())
    t1 = int(pipeline._utc(t_end).timestamp())
    conn = ais_ingest.connect()
    try:
        grouped = ais_ingest.query_window(conn, (west, south, east, north), t0, t1)
    finally:
        conn.close()
    tracks = ais_interp.build_tracks(grouped, t0, t1, step_seconds=step_seconds)
    return {
        "t_start": ais_ingest.iso(t0),
        "t_end": ais_ingest.iso(t1),
        "step_seconds": step_seconds,
        "vessels": [
            {
                "mmsi": tr.mmsi,
                "name": tr.name,
                "type": ais_score.vessel_types.decode(tr.vessel_type_raw),
                "samples": tr.samples(stride=1),
                "gaps": [g.to_dict() for g in tr.gaps],
            }
            for tr in tracks.values()
        ],
    }


@router.get("/api/scoring")
def scoring() -> Dict[str, Any]:
    return ais_score.explain_weights()
