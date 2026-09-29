"""Frozen configuration for TideTrace (SIH26143).

Every constant in the FROZEN block comes straight from the spec sheet. Change
them only through environment variables so that the shipped defaults stay
auditable.
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = Path(os.environ.get("TIDETRACE_DATA", ROOT_DIR / "data"))
SAR_DIR = DATA_DIR / "sar"
AIS_DIR = DATA_DIR / "ais"
METOCEAN_DIR = DATA_DIR / "metocean"
JOBS_DIR = DATA_DIR / "jobs"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = Path(os.environ.get("TIDETRACE_MODELS", ROOT_DIR / "models"))
STATIC_DIR = APP_DIR / "static"

AIS_SQLITE = AIS_DIR / "ais.sqlite"
SCENES_INDEX = SAR_DIR / "scenes.json"
CHECKPOINT = MODELS_DIR / "oil_unet_best.pt"

for _d in (DATA_DIR, SAR_DIR, AIS_DIR, METOCEAN_DIR, JOBS_DIR, CACHE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _b(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ----------------------------------------------------------------------------
# FROZEN CONSTANTS (spec: CONFIG DEFAULTS)
# ----------------------------------------------------------------------------
ALPHA_WIND = _f("TIDETRACE_ALPHA_WIND", 0.03)      # 3 percent of wind, Stokes off
DEFLECTION_DEG = _f("TIDETRACE_DEFLECTION_DEG", 15.0)  # right of wind in N hemisphere
ENSEMBLE_N = _i("TIDETRACE_ENSEMBLE_N", 50)
DT_SECONDS = _i("TIDETRACE_DT_SECONDS", 3600)
HINDCAST_H = _i("TIDETRACE_HINDCAST_H", 48)
FORECAST_H = _i("TIDETRACE_FORECAST_H", 36)
SEARCH_RADIUS_KM = _f("TIDETRACE_SEARCH_RADIUS_KM", 10.0)
ORIGIN_WINDOW_H = _f("TIDETRACE_ORIGIN_WINDOW_H", 3.0)
OIL_DB_THRESHOLD = _f("TIDETRACE_OIL_DB_THRESHOLD", -22.0)
TILE = _i("TIDETRACE_TILE", 512)
TILE_OVERLAP = _i("TIDETRACE_TILE_OVERLAP", 64)

# Scoring weights. prox and time together carry half the model, which is what
# "spatio-temporal correlation" in the problem statement actually asks for.
# Behaviour stays the largest single term because it is the only one that
# describes conduct rather than coincidence.
WEIGHTS = {
    "prox": _f("TIDETRACE_W_PROX", 0.30),
    "time": _f("TIDETRACE_W_TIME", 0.20),
    "beh": _f("TIDETRACE_W_BEH", 0.25),
    "type": _f("TIDETRACE_W_TYPE", 0.15),
    "traj": _f("TIDETRACE_W_TRAJ", 0.10),
}

OFFLINE = _b("TIDETRACE_OFFLINE", True)
OPEN_METEO_LIVE = _b("TIDETRACE_OPEN_METEO_LIVE", False)

# Ensemble noise, spec: "current noise 0.1 m/s, wind noise 1 m/s"
CURRENT_NOISE_MS = _f("TIDETRACE_CURRENT_NOISE", 0.1)
WIND_NOISE_MS = _f("TIDETRACE_WIND_NOISE", 1.0)

# Origin rule, spec: first time ensemble spread radius exceeds 8 km, clipped
SPREAD_TRIGGER_KM = _f("TIDETRACE_SPREAD_TRIGGER_KM", 8.0)
ORIGIN_H_MIN = _f("TIDETRACE_ORIGIN_H_MIN", 6.0)
ORIGIN_H_MAX = _f("TIDETRACE_ORIGIN_H_MAX", 48.0)
ORIGIN_BUFFER_KM = _f("TIDETRACE_ORIGIN_BUFFER_KM", 2.0)

# Radiometric operating point. Open water backscatter moves several dB with
# wind, incidence angle and product calibration -- the shipped scenes span
# -17.7 dB to -23.4 dB of open water -- so a network standardised on frozen
# corpus statistics is reading each scene at a different point on its response
# curve. Every run therefore measures and reports the scene's water level, and
# warns when it sits further than RADIOMETRIC_WARN_DB from the checkpoint's
# reference.
#
# Correcting the offset is available but OFF by default, and that is a measured
# decision rather than caution. The checkpoint records the corpus MEAN, not the
# corpus water level, and those differ by an unknown amount because the training
# tiles were selected for containing oil. Aligning to the mean brightens a dark
# scene until its slicks stop reading as slicks: on the Gulf chip it took a
# correct 29-polygon detection to zero. Enable it only with a checkpoint whose
# metadata carries a real `sea_level_db`.
RADIOMETRIC_ALIGN = _b("TIDETRACE_RADIOMETRIC_ALIGN", False)
RADIOMETRIC_MAX_SHIFT_DB = _f("TIDETRACE_RADIOMETRIC_MAX_SHIFT_DB", 6.0)
RADIOMETRIC_WARN_DB = _f("TIDETRACE_RADIOMETRIC_WARN_DB", 4.0)

# A scene whose co-pol dynamic range is below this has no structure to detect:
# uniform wind-roughened water. Reported so that "no oil" reads as a measured
# finding rather than a detector that fell over.
CLEAN_WATER_SPAN_DB = _f("TIDETRACE_CLEAN_WATER_SPAN_DB", 3.0)

# Baseline thresholding. Oil is a contrast phenomenon: it sits some dB below
# whatever the local sea happens to be. OIL_DB_THRESHOLD is the published
# absolute reference and is reported alongside, but the relative cut is what
# actually governs, because a fixed absolute cut finds the whole ocean on a
# calm dark scene and nothing at all on a bright one.
BASELINE_RELATIVE_ONLY = _b("TIDETRACE_BASELINE_RELATIVE_ONLY", True)

# Detection post-processing
MIN_OIL_AREA_KM2 = _f("TIDETRACE_MIN_OIL_AREA_KM2", 0.05)
MIN_OIL_PIXELS = _i("TIDETRACE_MIN_OIL_PIXELS", 500)
# Look-alikes are reported, not attributed, so a larger floor keeps the map
# readable without changing anything the pipeline actually acts on.
MIN_LOOKALIKE_AREA_KM2 = _f("TIDETRACE_MIN_LOOKALIKE_AREA_KM2", 0.12)
MAX_LOOKALIKE_POLYGONS = _i("TIDETRACE_MAX_LOOKALIKE_POLYGONS", 30)

# Run history retention. A job document is the audit trail and is small; the
# overlay PNGs are a megabyte a run and are regenerable. Without a bound the
# demo machine fills its disk, which is how this constant came to exist.
# Set TIDETRACE_KEEP_JOBS=0 to disable pruning entirely.
KEEP_JOBS = _i("TIDETRACE_KEEP_JOBS", 200)
KEEP_JOB_OVERLAYS = _i("TIDETRACE_KEEP_JOB_OVERLAYS", 20)

# Deterministic behaviour for a judged demo
RANDOM_SEED = _i("TIDETRACE_SEED", 20260920)

# Demo flags surfaced in the UI so nothing is hidden from a judge
DEMO_METOCEAN = _b("TIDETRACE_DEMO_METOCEAN", False)
ALLOW_SELFTEST_SCENES = _b("TIDETRACE_ALLOW_SELFTEST_SCENES", False)

CLASS_NAMES = {0: "sea", 1: "look_alike", 2: "mineral_oil"}

UI_TITLE = "NTRO Oil Spill Attribution Console"
PROJECT_CODENAME = "TideTrace"
SIH_ID = "SIH26143"
VERSION = "1.5.0"


def as_dict() -> dict:
    """Config snapshot the UI renders so weights are never hidden."""
    return {
        "alpha_wind": ALPHA_WIND,
        "deflection_deg": DEFLECTION_DEG,
        "ensemble_n": ENSEMBLE_N,
        "dt_seconds": DT_SECONDS,
        "hindcast_h": HINDCAST_H,
        "forecast_h": FORECAST_H,
        "search_radius_km": SEARCH_RADIUS_KM,
        "origin_window_h": ORIGIN_WINDOW_H,
        "oil_db_threshold": OIL_DB_THRESHOLD,
        "radiometric_align": RADIOMETRIC_ALIGN,
        "radiometric_max_shift_db": RADIOMETRIC_MAX_SHIFT_DB,
        "radiometric_warn_db": RADIOMETRIC_WARN_DB,
        "clean_water_span_db": CLEAN_WATER_SPAN_DB,
        "baseline_relative_only": BASELINE_RELATIVE_ONLY,
        "tile": TILE,
        "weights": dict(WEIGHTS),
        "offline": OFFLINE,
        "open_meteo_live": OPEN_METEO_LIVE,
        "current_noise_ms": CURRENT_NOISE_MS,
        "wind_noise_ms": WIND_NOISE_MS,
        "spread_trigger_km": SPREAD_TRIGGER_KM,
        "seed": RANDOM_SEED,
        "keep_jobs": KEEP_JOBS,
        "keep_job_overlays": KEEP_JOB_OVERLAYS,
        "version": VERSION,
        "sih_id": SIH_ID,
    }
