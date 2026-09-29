"""Job persistence.

Every run of the pipeline writes one JSON document under data/jobs/. That file
is the whole audit trail: inputs, every step with its timing, the polygons, the
drift ensemble summary, the AIS funnel counts, and the scored leaderboard. A
judge can open it without the UI and check that the numbers on screen came from
somewhere.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from .. import config

_LOCK = threading.Lock()


def new_job_id(prefix: str = "job") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return "%s_%s_%s" % (prefix, stamp, uuid.uuid4().hex[:6])


def path_for(job_id: str) -> Path:
    return Path(config.JOBS_DIR) / ("%s.json" % job_id)


def save(job_id: str, document: Dict[str, Any]) -> Path:
    p = path_for(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        tmp = p.with_suffix(".json.part")
        tmp.write_text(json.dumps(document, indent=2, default=_default), encoding="utf-8")
        tmp.replace(p)      # atomic: a reader never sees a half-written document
    prune()
    return p


def prune(keep: int = None, keep_overlays: int = None) -> Dict[str, int]:
    """Hold the run history to a fixed size.

    Every run writes a job document of a few hundred KB and a pair of overlay
    PNGs of about a megabyte. Nothing removed them, so a machine left running
    demos accumulated until the disk filled -- which is not a hypothetical, it
    happened here at 161 jobs. Documents are kept longer than overlays because
    they are the audit trail and they are small; the PNGs are regenerable by
    re-running the job.

    Set TIDETRACE_KEEP_JOBS to 0 to keep everything.
    """
    keep = int(config.KEEP_JOBS if keep is None else keep)
    keep_overlays = int(config.KEEP_JOB_OVERLAYS if keep_overlays is None else keep_overlays)
    removed = {"documents": 0, "overlays": 0}
    if keep <= 0:
        return removed

    jobs_dir = Path(config.JOBS_DIR)
    cache_dir = Path(config.CACHE_DIR)
    try:
        docs = sorted(jobs_dir.glob("job_*.json"),
                      key=lambda q: q.stat().st_mtime, reverse=True)
    except OSError:
        return removed

    live_ids = {q.stem for q in docs[:max(keep_overlays, 0)]}
    for stale in docs[keep:]:
        try:
            stale.unlink()
            removed["documents"] += 1
        except OSError:
            pass

    # Overlays belonging to a job outside the overlay window, or to a job whose
    # document is gone entirely, have nothing left that can reference them.
    try:
        overlays = list(cache_dir.glob("job_*.png"))
    except OSError:
        overlays = []
    for png in overlays:
        stem = png.name.rsplit("_", 1)[0]
        if stem not in live_ids:
            try:
                png.unlink()
                removed["overlays"] += 1
            except OSError:
                pass
    return removed


def usage() -> Dict[str, Any]:
    """What the run history currently costs on disk, for /api/health."""
    def _measure(folder: Path, pattern: str):
        try:
            files = list(folder.glob(pattern))
        except OSError:
            return 0, 0
        return len(files), sum(f.stat().st_size for f in files if f.is_file())

    n_docs, b_docs = _measure(Path(config.JOBS_DIR), "job_*.json")
    n_png, b_png = _measure(Path(config.CACHE_DIR), "job_*.png")
    return {
        "jobs": n_docs,
        "overlays": n_png,
        "bytes": b_docs + b_png,
        "megabytes": round((b_docs + b_png) / 1e6, 1),
        "keep_jobs": config.KEEP_JOBS,
        "keep_overlays": config.KEEP_JOB_OVERLAYS,
    }


def load(job_id: str) -> Optional[Dict[str, Any]]:
    p = path_for(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def listing(limit: int = 50) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    files = sorted(Path(config.JOBS_DIR).glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[:limit]:
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "job_id": doc.get("job_id", p.stem),
            "scene_id": (doc.get("input") or {}).get("scene_id"),
            "created": doc.get("created"),
            "oil_polygons": len(doc.get("detection", {}).get("polygons", [])),
            "suspects": len(doc.get("attribution", {}).get("suspects", [])),
            "status": doc.get("status", "unknown"),
        })
    return out


def _default(o: Any):
    if isinstance(o, datetime):
        return o.isoformat()
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


@dataclass
class Trace:
    """Step timings and notes, surfaced in the UI trace log.

    A trace can also publish live progress. `/api/run` is synchronous and a
    full run takes the better part of a minute on a laptop CPU, almost all of
    it inside DETECT. An opaque spinner for 45 seconds is indistinguishable
    from a hang, so a trace given a job_id records the step it is currently in
    where the progress endpoint can read it.
    """

    steps: List[Dict[str, Any]] = field(default_factory=list)
    job_id: Optional[str] = None
    _t0: float = field(default_factory=perf_counter)
    _current: Optional[Dict[str, Any]] = None

    def start(self, name: str, note: str = "") -> None:
        self._current = {
            "step": name,
            "note": note,
            "started_ms": round((perf_counter() - self._t0) * 1000.0, 1),
        }
        self._publish(name, note)

    def end(self, status: str = "ok", **detail: Any) -> None:
        if self._current is None:
            return
        now = round((perf_counter() - self._t0) * 1000.0, 1)
        self._current["elapsed_ms"] = round(now - self._current["started_ms"], 1)
        self._current["status"] = status
        self._current.update(detail)
        self.steps.append(self._current)
        self._current = None

    def note(self, name: str, **detail: Any) -> None:
        self.steps.append({
            "step": name,
            "started_ms": round((perf_counter() - self._t0) * 1000.0, 1),
            "elapsed_ms": 0.0,
            "status": "info",
            **detail,
        })

    def total_ms(self) -> float:
        return round((perf_counter() - self._t0) * 1000.0, 1)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.steps)

    # -- live progress -------------------------------------------------------

    def _publish(self, step: str, note: str) -> None:
        if not self.job_id:
            return
        with _PROGRESS_LOCK:
            _PROGRESS[self.job_id] = {
                "job_id": self.job_id,
                "step": step,
                "note": note,
                "index": len(self.steps) + 1,
                "total": len(PIPELINE_STEPS),
                "done": [s["step"] for s in self.steps],
                "elapsed_ms": round((perf_counter() - self._t0) * 1000.0, 1),
            }

    def finish(self) -> None:
        if not self.job_id:
            return
        with _PROGRESS_LOCK:
            _PROGRESS.pop(self.job_id, None)


# The steps a full run walks, in order, so the UI can show position rather than
# just motion. NO_OIL replaces everything from METOCEAN on when a scene is clean.
PIPELINE_STEPS = [
    "DETECT", "CHAR", "EO", "RENDER", "METOCEAN", "HINDCAST",
    "FORECAST", "COAST", "AGE_PROXY", "AIS", "FILTER", "SCORE",
]

_PROGRESS: Dict[str, Dict[str, Any]] = {}
_PROGRESS_LOCK = threading.Lock()


def progress(job_id: str) -> Optional[Dict[str, Any]]:
    """What a running job is doing right now, or None once it has finished."""
    with _PROGRESS_LOCK:
        p = _PROGRESS.get(job_id)
        return dict(p) if p else None
