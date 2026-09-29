"""The land mask, and the two ways it was silently wrong before.

A SAR chip containing coast has dark pixels on it that are not slicks: layover
shadow behind a ridge, a calm harbour basin, a wet runway. Nothing upstream of
the mask knows the difference, so an inaccurate mask is worse than none.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.drift import land   # noqa: E402


KNOWN_POINTS = [
    ("Santa Barbara mountains", -119.75, 34.47, True),
    ("Santa Barbara channel",   -119.87, 34.32, False),
    ("Mississippi delta",       -89.25,  29.15, True),
    ("Gulf open water",         -88.97,  28.90, False),
    ("Baku, Absheron",           49.85,  40.38, True),
    ("Caspian open water",       50.60,  40.10, False),
    ("Mumbai",                   72.85,  19.08, True),
    ("Arabian Sea offshore",     71.60,  19.05, False),
]


def test_the_shipped_mask_agrees_with_known_points():
    """The file that shipped before this failed two of these, in both directions.

    It called the Santa Barbara mountains sea, which lets radar shadow on a
    hillside be reported as mineral oil, and it called open Caspian water land,
    which raises a coast-impact flag in the middle of a sea.
    """
    if not land.load_rings():
        import pytest
        pytest.skip("no land mask cached; run scripts/fetch_land_mask.py --all")

    wrong = [(n, want) for n, lon, lat, want in KNOWN_POINTS
             if land.is_land(lon, lat) != want]
    assert not wrong, "land mask disagrees at: %s" % wrong


def test_inland_seas_are_water_not_land():
    """Natural Earth models the Caspian as a hole in Eurasia.

    An earlier build read only exterior rings and dropped every hole, so the
    whole Caspian became landmass: 144 of 144 sample points across the Baku
    footprint came back land, and masking on that would have deleted the scene.
    """
    if not land.load_rings():
        import pytest
        pytest.skip("no land mask cached")

    assert land.load_water_rings(), "no water rings: holes are being discarded"
    assert land.is_land(50.60, 40.10) is False
    assert land.is_land(49.85, 40.38) is True


def test_a_missing_mask_is_not_an_error(tmp_path, monkeypatch):
    """The mask is optional by design; absence must degrade, not raise."""
    monkeypatch.setattr(land.config, "DATA_DIR", str(tmp_path))
    land._CACHE.update({"loaded": False, "rings": None, "water": None, "path": None})
    try:
        assert land.load_rings() is None
        assert land.is_land(0.0, 0.0) is False
        out = land.check([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)], {})
        assert out["available"] is False
        assert out["coast_flag"] is None
    finally:
        land._CACHE.update({"loaded": False, "rings": None, "water": None, "path": None})


def test_every_scene_keeps_some_water():
    """A mask that swallows a whole footprint is broken, however plausible.

    This is the assertion that would have caught the dropped-holes bug: the
    Caspian footprint was reading as 100 percent land.
    """
    import numpy as np

    from app import scenes as scenes_mod

    if not land.load_rings():
        import pytest
        pytest.skip("no land mask cached")

    for s in scenes_mod.all_scenes(include_selftest=False):
        w, so, e, n = [float(v) for v in s.bounds]
        lons = np.linspace(w, e, 10)
        lats = np.linspace(so, n, 10)
        land_hits = sum(land.is_land(float(x), float(y)) for x in lons for y in lats)
        assert land_hits < 100, (
            "%s reads as entirely land; the mask would erase the scene" % s.id)


def test_the_mask_file_records_its_provenance():
    path = Path(ROOT) / "data" / "land" / "coastline.geojson"
    if not path.exists():
        import pytest
        pytest.skip("no land mask cached")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert "Natural Earth" in doc.get("license", "")
    kinds = {f["properties"].get("kind") for f in doc["features"]}
    assert "land" in kinds and "water" in kinds, kinds
