"""POST /api/detect and GET /api/scenes.

Accepts either a registered scene id or an uploaded GeoTIFF. An upload without
a usable affine and CRS is rejected with a clear message rather than quietly
producing coordinates that mean nothing.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import config, pipeline, scenes as scenes_mod
from ..geo import raster as raster_mod
from ..jobs import store as job_store
from ..schemas import DetectRequest, SceneOut

router = APIRouter()


@router.get("/api/scenes", response_model=List[SceneOut])
def list_scenes() -> List[SceneOut]:
    out: List[SceneOut] = []
    for s in scenes_mod.all_scenes():
        d = s.to_dict()
        out.append(SceneOut(
            id=d["id"], title=d["title"], t_sat=d["t_sat"], bounds=d["bounds"],
            centroid=d["centroid"], crs=d["crs"], ais_mode=d["ais_mode"],
            source=d["source"], license=d["license"], notes=d["notes"],
            has_truth_mask=bool(d["mask_path"]), thumbnail=d["thumbnail"],
            exists=d["exists"],
        ))
    return out


def _strip(det: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the heavy numpy payloads before the response is serialised."""
    return {
        "scene": det["scene"],
        "sar_bounds": det["sar_bounds"],
        "polygons": det["polygons"],
        "lookalikes": det["lookalikes"],
        "metrics": det["metrics"],
        "eo": det.get("eo"),
        "overlays": det["overlays"],
    }


@router.post("/api/detect")
def detect(req: DetectRequest) -> Dict[str, Any]:
    if not req.scene_id:
        raise HTTPException(400, "scene_id is required, or use the multipart upload endpoint")
    scene = scenes_mod.get(req.scene_id)
    if scene is None:
        raise HTTPException(404, "unknown scene_id %r" % req.scene_id)

    job_id = job_store.new_job_id("detect")
    det = pipeline.detect_scene(
        scene, prefer_model=req.prefer_model, threshold_db=req.threshold_db,
        render_overlays=req.render_overlays, job_id=job_id,
    )
    doc = {
        "job_id": job_id,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "input": {"scene_id": scene.id},
        "scene": det["scene"],
        "detection": _strip(det),
        "trace": det["trace"].to_list(),
        "config": config.as_dict(),
    }
    job_store.save(job_id, doc)
    return {"job_id": job_id, **_strip(det), "trace": doc["trace"]}


@router.post("/api/detect/upload")
async def detect_upload(
    file: UploadFile = File(...),
    t_sat: Optional[str] = Form(None),
    prefer_model: bool = Form(True),
    threshold_db: Optional[float] = Form(None),
) -> Dict[str, Any]:
    """Detect on an uploaded Sigma0 GeoTIFF."""
    name = Path(file.filename or "upload.tif").name
    dest = Path(config.SAR_DIR) / ("upload_%s_%s" % (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"), name))
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        scene = scenes_mod.describe_geotiff(dest, t_sat=t_sat, source="uploaded",
                                           notes="uploaded at runtime")
    except raster_mod.GeorefError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "could not read %s: %s" % (name, exc)) from exc

    scenes_mod.upsert([scene])

    job_id = job_store.new_job_id("detect")
    det = pipeline.detect_scene(scene, prefer_model=prefer_model,
                                threshold_db=threshold_db, job_id=job_id)
    return {"job_id": job_id, "scene_id": scene.id, **_strip(det),
            "trace": det["trace"].to_list()}
