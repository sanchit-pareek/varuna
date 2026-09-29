"""GET /api/health and GET /api/config."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

from .. import config, scenes as scenes_mod
from ..ais import ingest as ais_ingest, score as ais_score
from ..drift import fields as fields_mod
from ..jobs import store as job_store
from ..ml import infer, model as model_mod
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    warnings: List[str] = []

    status = model_mod.status()
    loaded = infer.get_model()
    model_loaded = loaded is not None
    detector = "unet" if model_loaded else "sigma0_threshold_baseline"
    if not model_loaded:
        warnings.append("U-Net checkpoint not loaded (%s). Detection runs on the "
                        "dark-patch baseline, which scores IoU 0.000 for oil on the "
                        "validation tiles: treat its polygons as an availability "
                        "guarantee, not as a measurement."
                        % (infer.model_error() or "reason not recorded"))

    try:
        st = ais_ingest.stats()
        ais_rows, ais_vessels = st.rows, st.vessels
    except Exception as exc:
        ais_rows = ais_vessels = 0
        warnings.append("AIS store unreadable: %s" % exc)
    if ais_rows == 0:
        warnings.append("AIS store is empty. Run scripts/build_synthetic_ais.py or "
                        "ingest a MarineCadastre CSV.")

    metocean = fields_mod.list_cached()
    if not metocean:
        warnings.append("No cached metocean. Run scripts/build_metocean_cache.py while "
                        "online. Clause (b) is not satisfied by the constant fallback.")

    scene_list = scenes_mod.all_scenes()
    if not scene_list:
        warnings.append("No demo scenes indexed. Run scripts/download_zenodo_subset.py "
                        "then scripts/prepare_scenes.py.")

    try:
        storage = job_store.usage()
        if storage["keep_jobs"] and storage["jobs"] > storage["keep_jobs"]:
            warnings.append("Run history is above its retention bound (%d of %d kept)."
                            % (storage["jobs"], storage["keep_jobs"]))
    except Exception as exc:
        storage = {"error": str(exc)}

    return HealthResponse(
        ok=True,
        mode="offline" if config.OFFLINE else "online",
        cuda=bool(status["cuda"]),
        model_loaded=model_loaded,
        detector=detector,
        ais_rows=ais_rows,
        ais_vessels=ais_vessels,
        metocean_scenes=len(metocean),
        scenes=len(scene_list),
        version=config.VERSION,
        warnings=warnings,
        model_detail=status,
        storage=storage,
        config=config.as_dict(),
    )


@router.get("/api/config")
def get_config() -> Dict[str, Any]:
    return {
        "config": config.as_dict(),
        "scoring": ais_score.explain_weights(),
        "ui_title": config.UI_TITLE,
        "codename": config.PROJECT_CODENAME,
        "sih_id": config.SIH_ID,
    }


@router.get("/api/metocean")
def metocean() -> Dict[str, Any]:
    return {"cached": fields_mod.list_cached()}


@router.get("/api/ais/stats")
def ais_stats() -> Dict[str, Any]:
    return ais_ingest.stats().to_dict()
