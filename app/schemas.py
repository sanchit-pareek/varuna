"""Pydantic v2 request and response models for the API contract."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from . import config


class HealthResponse(BaseModel):
    # `model_detail` is about the ML model, not about pydantic's own namespace.
    model_config = ConfigDict(protected_namespaces=())

    ok: bool
    mode: str = "offline"
    cuda: bool
    model_loaded: bool
    detector: str
    ais_rows: int
    ais_vessels: int
    metocean_scenes: int
    scenes: int
    version: str
    warnings: List[str] = Field(default_factory=list)
    model_detail: Dict[str, Any] = Field(default_factory=dict)
    storage: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class SceneOut(BaseModel):
    id: str
    title: str
    t_sat: str
    bounds: List[float]
    centroid: List[float]
    crs: str
    ais_mode: str
    source: str = ""
    license: str = ""
    notes: str = ""
    has_truth_mask: bool = False
    thumbnail: Optional[str] = None
    exists: bool = True


class DetectRequest(BaseModel):
    scene_id: Optional[str] = None
    prefer_model: bool = True
    threshold_db: Optional[float] = None
    render_overlays: bool = True


class DriftRequest(BaseModel):
    job_id: Optional[str] = None
    centroid: Optional[List[float]] = Field(
        default=None, description="[lon, lat] used when job_id is absent")
    polygon: Optional[List[List[float]]] = Field(
        default=None, description="optional slick ring as [[lon, lat], ...]")
    t_sat: Optional[str] = None
    hours_back: int = config.HINDCAST_H
    hours_fwd: int = config.FORECAST_H
    ensemble_n: int = config.ENSEMBLE_N
    scene_id: Optional[str] = None


class AttributeRequest(BaseModel):
    job_id: Optional[str] = None
    origin_geojson: Optional[Dict[str, Any]] = None
    origin: Optional[List[float]] = Field(default=None, description="[lon, lat]")
    t_origin: Optional[str] = None
    slick_centroid: Optional[List[float]] = None
    radius_km: float = config.SEARCH_RADIUS_KM
    window_h: float = config.ORIGIN_WINDOW_H
    top_n: int = 10


class RunRequest(BaseModel):
    scene_id: Optional[str] = None
    t_sat: Optional[str] = None
    hindcast_hours: int = config.HINDCAST_H
    forecast_hours: int = config.FORECAST_H
    search_radius_km: float = config.SEARCH_RADIUS_KM
    origin_window_hours: float = config.ORIGIN_WINDOW_H
    ensemble_n: int = config.ENSEMBLE_N
    prefer_model: bool = True
    threshold_db: Optional[float] = None
    top_n: int = 10
    render_overlays: bool = True
    # The caller may name the run so it can poll progress while the synchronous
    # request is still open. Constrained because it becomes a filename.
    job_id: Optional[str] = Field(default=None, pattern=r"^job_[A-Za-z0-9_-]{1,48}$")


class InjectRequest(BaseModel):
    """Optional click-to-drop probe. Not the main path, and labelled as such."""

    lat: float
    lon: float
    t_sat: Optional[str] = None
    radius_km: float = config.SEARCH_RADIUS_KM
    window_h: float = config.ORIGIN_WINDOW_H
    hindcast_hours: int = config.HINDCAST_H
    forecast_hours: int = config.FORECAST_H
    slick_radius_km: float = 1.5
    top_n: int = 10


class JobSummary(BaseModel):
    job_id: str
    scene_id: Optional[str] = None
    created: Optional[str] = None
    oil_polygons: int = 0
    suspects: int = 0
    status: str = "unknown"
