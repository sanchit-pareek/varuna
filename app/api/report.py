"""GET /api/report/{job_id} - Maritime Pollution Attribution Note.

Optional by design. HTML is always available because it needs nothing beyond the
standard library. PDF is produced only if reportlab happens to be installed; if
it is not, the endpoint says so instead of failing the demo.
"""
from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

from .. import config
from ..jobs import store as job_store

router = APIRouter()


def _rows(doc: Dict[str, Any]) -> Dict[str, Any]:
    det = doc.get("detection") or {}
    drift = doc.get("drift") or {}
    attr = doc.get("attribution") or {}
    poly = doc.get("primary_polygon") or (det.get("polygons") or [None])[0]
    props = (poly or {}).get("properties", {})
    return {"det": det, "drift": drift, "attr": attr, "props": props}


def _lines(doc: Dict[str, Any]) -> List[str]:
    r = _rows(doc)
    props, drift, attr = r["props"], r["drift"], r["attr"]
    origin = drift.get("origin") or {}
    det_metrics = r["det"].get("metrics") or {}

    out = [
        "Job: %s" % doc.get("job_id"),
        "Generated: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "Scene: %s" % ((doc.get("scene") or {}).get("title") or (doc.get("input") or {}).get("scene_id")),
        "Observation time: %s" % (doc.get("input") or {}).get("t_sat"),
        "Detector: %s" % det_metrics.get("detector"),
        "",
        "SLICK GEOMETRY",
    ]

    if not props:
        out += [
            "No oil polygon above the area threshold on this scene.",
            "Look-alike polygons found: %s." % det_metrics.get("lookalike_polygons_found", 0),
            "This is a reported result, not a failure. A clean scene has no slick to",
            "characterise and no origin to trace, so drift and attribution were not run.",
        ]
    else:
        out += [
            "Area: %s km2" % props.get("area_km2"),
            "Length: %s km, width: %s km" % (props.get("length_km"), props.get("width_km")),
            "Perimeter: %s km" % props.get("perimeter_km"),
            "Orientation: %s deg" % props.get("orientation_deg"),
            "Compactness: %s" % props.get("compactness"),
            "Contrast against local sea: %s dB" % props.get("contrast_db"),
            "Centroid: %s, %s" % (props.get("centroid_lat"), props.get("centroid_lon")),
            "Detection confidence: %s" % props.get("confidence"),
        ]

    out += ["", "DRIFT"]
    if not drift:
        out.append("Not run: there was no slick to trace back.")
    else:
        out += [
            "Origin estimate: %s, %s at %s" % (origin.get("lat"), origin.get("lon"), origin.get("t")),
            "Origin zone radius: %s km (90 percent ensemble envelope, buffered %s km)"
            % (origin.get("spread_km"), origin.get("buffer_km")),
            "Origin zone area: %s km2" % origin.get("area_km2"),
            "Age since origin: %s hours (drift proxy, not a laboratory age)" % doc.get("age_hours_proxy"),
            "Metocean source: %s" % (drift.get("metocean") or {}).get("source"),
            "Wind drift factor: %s, Stokes drift off" % config.ALPHA_WIND,
        ]

    out += ["", "RANKED SUSPECTS"]
    for s in attr.get("suspects", [])[:10]:
        out.append("%2d. %-24s MMSI %-10s %-18s %5.1f%%  %s"
                   % (s["rank"], (s["name"] or "UNKNOWN")[:24], s["mmsi"],
                      s["type"], s["score"], ", ".join(s["reasons"][:4])))
    if not attr.get("suspects"):
        out.append("No vessel passed the spatio-temporal filter.")
    out += [
        "",
        "LIMITATIONS",
        "Ranked likelihood for investigation, not legal proof of discharge.",
        "Age is a drift time proxy. No chemical weathering model is used.",
        "Look-alike class polygons are excluded from attribution.",
    ]
    for w in doc.get("warnings", []):
        out.append("Note: %s" % w)
    return out


@router.get("/api/report/{job_id}", response_class=HTMLResponse)
def report_html(job_id: str) -> HTMLResponse:
    doc = job_store.load(job_id)
    if doc is None:
        raise HTTPException(404, "unknown job_id %r" % job_id)
    body = "\n".join(html.escape(line) for line in _lines(doc))
    page = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Maritime Pollution Attribution Note</title>"
        "<style>body{font:13px/1.5 ui-monospace,Consolas,monospace;max-width:820px;"
        "margin:32px auto;padding:0 20px;color:#0d1b2a}h1{font-size:19px;letter-spacing:.04em}"
        "pre{white-space:pre-wrap}</style>"
        "<h1>Maritime Pollution Attribution Note</h1>"
        "<p>%s &middot; %s</p><pre>%s</pre>"
        % (html.escape(config.UI_TITLE), html.escape(config.SIH_ID), body)
    )
    return HTMLResponse(page)


@router.get("/api/report/{job_id}/pdf")
def report_pdf(job_id: str) -> Response:
    doc = job_store.load(job_id)
    if doc is None:
        raise HTTPException(404, "unknown job_id %r" % job_id)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise HTTPException(
            501, "reportlab is not installed, so PDF export is unavailable. "
                 "The HTML note at /api/report/%s is always available." % job_id) from exc

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 60
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Maritime Pollution Attribution Note")
    y -= 18
    c.setFont("Helvetica", 8)
    c.drawString(50, y, "%s  %s" % (config.UI_TITLE, config.SIH_ID))
    y -= 22
    c.setFont("Courier", 8.5)
    for line in _lines(doc):
        if y < 60:
            c.showPage()
            c.setFont("Courier", 8.5)
            y = height - 60
        c.drawString(50, y, line[:110])
        y -= 11
    c.save()
    return Response(buf.getvalue(), media_type="application/pdf", headers={
        "Content-Disposition": 'attachment; filename="attribution_%s.pdf"' % job_id})
