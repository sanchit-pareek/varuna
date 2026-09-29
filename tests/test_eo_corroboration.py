"""The optical cross-check, and the empty-image trap it fell into first.

EO is corroboration, never a second detector: there is no paired labelled
SAR/optical oil dataset, and inventing one would repeat the mistake that
produced the first quarantined checkpoint. These tests pin the two things that
make the check trustworthy -- that it reads the right pixels, and that it says
"I cannot tell" rather than "no difference" when there is nothing to read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.eo import corroborate as eo   # noqa: E402

BOUNDS = [[10.0, 50.0], [11.0, 51.0]]     # [[south, west], [north, east]]


def _square(west, south, east, north):
    return {"type": "Feature",
            "properties": {"polygon_id": "OIL01"},
            "geometry": {"type": "Polygon", "coordinates": [[
                [west, south], [east, south], [east, north], [west, north], [west, south]]]}}


def _write_chip(tmp_path, grey, scene_id, offset_hours=17.0, monkey=None):
    from PIL import Image

    d = Path(tmp_path) / "optical"
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grey.astype(np.uint8), mode="L").save(d / ("%s.png" % scene_id))
    (d / ("%s.json" % scene_id)).write_text(json.dumps({
        "scene": scene_id, "status": "ok", "bounds": BOUNDS,
        "offset_hours": offset_hours, "offset_label": "%.0f h after the radar pass" % offset_hours,
        "cloud_percent": 0.0, "collection": "sentinel-2-l2a", "item_id": "TEST",
        "acquired": "2023-08-29T18:39:29Z",
    }), encoding="utf-8")
    return d


def test_a_dark_patch_reads_as_consistent(tmp_path, monkeypatch):
    monkeypatch.setattr(eo.config, "DATA_DIR", str(tmp_path))
    grey = np.full((200, 200), 120, dtype=np.uint8)
    grey[80:120, 80:120] = 90          # a film: clearly darker than its ring
    _write_chip(tmp_path, grey, "s")

    poly = _square(50.4, 10.4, 50.6, 10.6)
    out = eo.corroborate([poly], "s")

    assert out["available"] is True
    assert out["verdicts"][0]["verdict"] == "consistent", out["verdicts"]
    assert out["verdicts"][0]["levels_vs_surroundings"] < 0


def test_a_bright_patch_reads_as_inconsistent(tmp_path, monkeypatch):
    """A rig, a sandbar or a ship is brighter. A slick is not."""
    monkeypatch.setattr(eo.config, "DATA_DIR", str(tmp_path))
    grey = np.full((200, 200), 100, dtype=np.uint8)
    grey[80:120, 80:120] = 160
    _write_chip(tmp_path, grey, "s")

    out = eo.corroborate([_square(50.4, 10.4, 50.6, 10.6)], "s")
    assert out["verdicts"][0]["verdict"] == "inconsistent"


def test_uniform_water_reads_as_neutral(tmp_path, monkeypatch):
    monkeypatch.setattr(eo.config, "DATA_DIR", str(tmp_path))
    grey = np.full((200, 200), 110, dtype=np.uint8)
    _write_chip(tmp_path, grey, "s")

    out = eo.corroborate([_square(50.4, 10.4, 50.6, 10.6)], "s")
    assert out["verdicts"][0]["verdict"] == "neutral"


def test_nodata_reads_as_obscured_not_as_no_difference(tmp_path, monkeypatch):
    """The trap this check fell into on its first real run.

    A Sentinel-2 granule often covers only part of a scene footprint, and the
    chip cutter fills the rest with zero. On the Gulf scene that emptiness sat
    exactly where the detections were, and because only saturated white counted
    as obscured, all 29 polygons came back "neutral, no optical difference" with
    a delta of exactly 0.0 -- a confident verdict computed from a black image.
    """
    monkeypatch.setattr(eo.config, "DATA_DIR", str(tmp_path))
    grey = np.zeros((200, 200), dtype=np.uint8)      # entirely outside the granule
    _write_chip(tmp_path, grey, "s")

    out = eo.corroborate([_square(50.4, 10.4, 50.6, 10.6)], "s")
    v = out["verdicts"][0]
    assert v["verdict"] == "obscured", v
    assert v["obscured_fraction"] == 1.0


def test_cloud_reads_as_obscured(tmp_path, monkeypatch):
    monkeypatch.setattr(eo.config, "DATA_DIR", str(tmp_path))
    grey = np.full((200, 200), 120, dtype=np.uint8)
    grey[80:120, 80:120] = 255
    _write_chip(tmp_path, grey, "s")

    out = eo.corroborate([_square(50.4, 10.4, 50.6, 10.6)], "s")
    assert out["verdicts"][0]["verdict"] == "obscured"


def test_a_stale_chip_is_flagged_weak(tmp_path, monkeypatch):
    """Days of offset means the slick has moved. Say so on the verdict."""
    monkeypatch.setattr(eo.config, "DATA_DIR", str(tmp_path))
    grey = np.full((200, 200), 120, dtype=np.uint8)
    _write_chip(tmp_path, grey, "s", offset_hours=104.0)

    out = eo.corroborate([_square(50.4, 10.4, 50.6, 10.6)], "s")
    assert out["weak"] is True
    assert "did not observe this water at the radar acquisition time" in out["caveat"]


def test_a_missing_chip_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(eo.config, "DATA_DIR", str(tmp_path))
    out = eo.corroborate([_square(50.4, 10.4, 50.6, 10.6)], "nothing_here")
    assert out["available"] is False
    assert out["verdicts"] == []


def test_the_check_never_alters_a_detection(tmp_path, monkeypatch):
    """Corroboration is reported alongside, never subtracted."""
    monkeypatch.setattr(eo.config, "DATA_DIR", str(tmp_path))
    grey = np.full((200, 200), 100, dtype=np.uint8)
    grey[80:120, 80:120] = 200                 # strongly contradicts oil
    _write_chip(tmp_path, grey, "s")

    poly = _square(50.4, 10.4, 50.6, 10.6)
    before = json.dumps(poly, sort_keys=True)
    out = eo.corroborate([poly], "s")

    assert out["verdicts"][0]["verdict"] == "inconsistent"
    assert json.dumps(poly, sort_keys=True) == before, "the polygon was mutated"
    assert "No detection was added, removed or reweighted" in out["caveat"]
