"""Developer-only evaluation endpoints; never part of operational attribution."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from .. import evaluation
from ..jobs import store as job_store

router = APIRouter()


@router.get("/api/evaluation/attribution/{job_id}")
def attribution(job_id: str) -> Dict[str, Any]:
    document = job_store.load(job_id)
    if document is None:
        raise HTTPException(404, "unknown job_id %r" % job_id)
    return evaluation.evaluate_job(document)
