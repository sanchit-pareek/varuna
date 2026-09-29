"""TideTrace FastAPI application.

    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Serves the static console at / and the API under /api. Cached PNG overlays and
any other artefact under data/ are served read only at /data.

Nothing here reaches the network. The only scripts that do are under scripts/,
they are run before the demo, and they write into data/.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .api import ais as ais_routes, detect as detect_routes, drift as drift_routes
from .api import health as health_routes, pipeline as pipeline_routes, report as report_routes

log = logging.getLogger("tidetrace")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Say out loud what the server can and cannot do before anyone asks."""
    from . import scenes as scenes_mod
    from .ais import ingest as ais_ingest
    from .drift import fields as fields_mod

    log.info("%s %s starting", config.UI_TITLE, config.VERSION)
    try:
        st = ais_ingest.stats()
        log.info("AIS store: %d rows, %d vessels", st.rows, st.vessels)
    except Exception as exc:
        log.warning("AIS store unreadable: %s", exc)
    log.info("metocean cubes cached: %d", len(fields_mod.list_cached()))
    log.info("scenes indexed: %d", len(scenes_mod.all_scenes()))
    log.info("mode: %s", "OFFLINE" if config.OFFLINE else "ONLINE")

    # Warm the checkpoint off the request path. Loading it takes the better part
    # of ten seconds on CPU, and the first thing any browser does is call
    # /api/health -- so without this the console sits blank while a background
    # detail finishes. get_model() is idempotent and locked, so a request that
    # arrives mid-warm simply waits for the same load rather than starting one.
    def _warm():
        try:
            from .ml import infer
            t = time.perf_counter()
            ready = infer.get_model() is not None
            log.info("detector warm: %s in %.1fs",
                     "U-Net" if ready else "baseline (%s)" % infer.model_error(),
                     time.perf_counter() - t)
        except Exception as exc:            # never let warm-up take the app down
            log.warning("detector warm-up failed: %s", exc)

    threading.Thread(target=_warm, name="detector-warm", daemon=True).start()
    yield


app = FastAPI(
    title=config.UI_TITLE,
    version=config.VERSION,
    lifespan=lifespan,
    description=(
        "Offline SAR oil slick detection, drift hindcast and forecast, and AIS "
        "based vessel attribution. SIH %s, NTRO, Space Technology."
        % config.SIH_ID
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_routes.router)
app.include_router(detect_routes.router)
app.include_router(drift_routes.router)
app.include_router(ais_routes.router)
app.include_router(pipeline_routes.router)
app.include_router(report_routes.router)

Path(config.STATIC_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(config.DATA_DIR)), name="data")
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


def _asset_stamp() -> str:
    """A cache key that changes when the assets do, not when the version does.

    Stamping `config.VERSION` into the asset URLs was not enough: editing
    app.js without bumping the version leaves the URL identical, so a browser
    that already has `app.js?v=1.5.0` serves the old file from cache and pairs
    it with a freshly rendered DOM. That is exactly the mismatch the stamp was
    supposed to prevent, and it is silent. Hashing the files themselves means
    the URL changes if and only if the bytes do.
    """
    h = hashlib.sha256()
    h.update(config.VERSION.encode("utf-8"))
    static = Path(config.STATIC_DIR)
    for name in ("app.js", "style.css", "index.html"):
        f = static / name
        try:
            h.update(f.read_bytes())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:12]


@app.get("/", include_in_schema=False)
def index():
    """Serve the console shell uncached, with content-addressed asset URLs.

    A browser that caches index.html but revalidates app.js will happily pair a
    new script with an old DOM, and the console then fails on an element that
    does not exist yet. Hashing the assets into their URLs and refusing to cache
    the shell makes that combination impossible.
    """
    page = Path(config.STATIC_DIR) / "index.html"
    if not page.exists():
        return JSONResponse({"error": "static/index.html is missing"}, status_code=500)
    html = page.read_text(encoding="utf-8").replace("__V__", _asset_stamp())
    return HTMLResponse(html, headers={"Cache-Control": "no-store, must-revalidate"})
