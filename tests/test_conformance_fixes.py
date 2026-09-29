"""Regressions from the PS 26143 conformance review.

Each test here pins a defect that was found by running the system rather than
by reading it, and each one failed before the corresponding fix.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# The model load race
# ---------------------------------------------------------------------------

def test_concurrent_callers_all_get_the_same_model(monkeypatch):
    """The bug: the console told a judge it was running the dB baseline.

    The load cache set `tried` before the load finished, so every request that
    arrived during the several seconds a CPU checkpoint takes to load saw
    "no model, no error" and fell through to the baseline. The browser fires
    /api/health, /api/scenes and /api/scoring on page load, so this happened on
    the very first page view, and a detect issued in that window silently
    produced a baseline mask with nothing recorded to say so.
    """
    from app.ml import infer

    calls = {"n": 0}

    def slow_load():
        calls["n"] += 1
        time.sleep(0.4)                       # stand in for a real checkpoint load
        return {"model": object(), "normalisation": {"mean_db": -20.0, "std_db": 9.0}}

    infer.reset_model_cache()
    monkeypatch.setattr(infer.model_mod, "torch_available", lambda: True)
    monkeypatch.setattr(infer.model_mod, "smp_available", lambda: True)
    monkeypatch.setattr(infer.model_mod, "load_checkpoint", slow_load)
    try:
        results = {}

        def call(i):
            results[i] = infer.get_model()

        threads = [threading.Thread(target=call, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert calls["n"] == 1, "the checkpoint must be loaded exactly once"
        assert all(v is not None for v in results.values()), (
            "%d of 8 concurrent callers fell through to the baseline"
            % sum(1 for v in results.values() if v is None))
        assert len({id(v) for v in results.values()}) == 1, "every caller gets the same object"
    finally:
        infer.reset_model_cache()


def test_a_real_load_failure_is_recorded_not_silent(monkeypatch):
    """The other half: when there genuinely is no model, say why."""
    from app.ml import infer

    infer.reset_model_cache()
    monkeypatch.setattr(infer.model_mod, "torch_available", lambda: True)
    monkeypatch.setattr(infer.model_mod, "smp_available", lambda: True)

    def boom():
        raise RuntimeError("checkpoint is truncated")

    monkeypatch.setattr(infer.model_mod, "load_checkpoint", boom)
    try:
        assert infer.get_model() is None
        assert "truncated" in (infer.model_error() or "")
        assert infer.model_ready() is True
    finally:
        infer.reset_model_cache()


# ---------------------------------------------------------------------------
# Scene radiometry and the clean water verdict
# ---------------------------------------------------------------------------

def test_uniform_water_is_reported_as_clean_not_as_nothing():
    """An empty result must say which kind of empty it is.

    The Arabian Sea chip spans 1.95 dB after speckle averaging: it is
    wind-roughened open water with no slick in it. Returning zero polygons is
    correct there, but a bare empty result is indistinguishable from a detector
    that fell over, and the two need different responses from an operator.
    """
    from app.ml import infer

    rng = np.random.default_rng(7)
    flat = np.full((2, 256, 256), -17.5, dtype=np.float32)
    flat += rng.normal(0.0, 0.05, flat.shape).astype(np.float32)

    rad = infer.scene_radiometry(flat)
    assert rad["clean_water"] is True
    assert rad["dynamic_range_db"] < 3.0
    assert rad["sea_level_db"] == pytest.approx(-17.5, abs=0.5)


def test_a_scene_with_a_dark_patch_is_not_called_clean():
    from app.ml import infer

    rng = np.random.default_rng(11)
    scene = np.full((2, 256, 256), -18.0, dtype=np.float32)
    scene += rng.normal(0.0, 0.3, scene.shape).astype(np.float32)
    scene[:, 80:170, 60:200] -= 9.0            # a slick, 9 dB down

    rad = infer.scene_radiometry(scene)
    assert rad["clean_water"] is False
    assert rad["dynamic_range_db"] > 3.0


def test_the_baseline_threshold_follows_the_water_not_a_constant():
    """The bug: a fixed -22 dB cut on a scene whose water sits at -17.7 dB.

    min(absolute, relative) meant the published constant won on every bright
    scene. Oil is a contrast phenomenon; the cut has to move with the water.
    """
    from app.ml import fallback

    rng = np.random.default_rng(3)
    for sea_level in (-17.0, -23.0):
        scene = np.full((300, 300), sea_level, dtype=np.float32)
        scene += rng.normal(0.0, 0.2, scene.shape).astype(np.float32)
        # Elongated, the way a slick is. A compact blob would be classified as a
        # look-alike on boundary regularity, which is correct behaviour but not
        # what this test is about.
        scene[130:160, 50:260] = sea_level - 8.0

        out = fallback.detect(scene, min_pixels=50)
        assert out["threshold_db"] < sea_level, (
            "threshold %.1f is not below the water level %.1f"
            % (out["threshold_db"], sea_level))
        assert out["threshold_db"] > sea_level - 8.0, "the cut must sit above the slick"
        assert (out["mask"] == 2).sum() > 1000, (
            "no oil found on a scene whose water sits at %.1f dB" % sea_level)


def test_land_is_excluded_from_the_water_level_estimate():
    """A quarter of the Santa Barbara chip is coast, and it moved the water
    level 1.6 dB brighter, which moves every relative threshold with it."""
    from app.ml import infer

    rng = np.random.default_rng(5)
    scene = np.full((2, 256, 256), -20.0, dtype=np.float32)
    scene += rng.normal(0.0, 0.3, scene.shape).astype(np.float32)
    scene[:, :, 190:] = -6.0                  # bright land down one side

    land = np.zeros((256, 256), dtype=bool)
    land[:, 190:] = True

    with_land = infer.scene_radiometry(scene)
    without = infer.scene_radiometry(scene, exclude=land)
    assert without["sea_level_db"] < with_land["sea_level_db"]
    assert without["sea_level_db"] == pytest.approx(-20.0, abs=0.6)


# ---------------------------------------------------------------------------
# Run history retention
# ---------------------------------------------------------------------------

def test_job_history_is_bounded(tmp_path, monkeypatch):
    """161 job documents and 218 overlay PNGs filled the disk mid-demo."""
    from app import config
    from app.jobs import store

    jobs, cache = tmp_path / "jobs", tmp_path / "cache"
    jobs.mkdir()
    cache.mkdir()
    monkeypatch.setattr(config, "JOBS_DIR", jobs)
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "KEEP_JOBS", 5)
    monkeypatch.setattr(config, "KEEP_JOB_OVERLAYS", 2)

    for i in range(12):
        jid = "job_2026090%d_%06d" % (i % 10, i)
        store.save(jid, {"job_id": jid, "status": "ok"})
        (cache / ("%s_sar.png" % jid)).write_bytes(b"x")
        (cache / ("%s_mask.png" % jid)).write_bytes(b"x")
        time.sleep(0.01)                       # keep mtimes strictly ordered

    store.prune()
    assert len(list(jobs.glob("job_*.json"))) == 5
    assert len(list(cache.glob("job_*.png"))) == 4      # 2 jobs x 2 overlays

    usage = store.usage()
    assert usage["jobs"] == 5 and usage["overlays"] == 4

    # The newest document must be one of the survivors, and still readable.
    newest = sorted(jobs.glob("job_*.json"), key=lambda p: p.stat().st_mtime)[-1]
    assert store.load(newest.stem)["status"] == "ok"


def test_keep_jobs_zero_disables_pruning(tmp_path, monkeypatch):
    from app import config
    from app.jobs import store

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(config, "JOBS_DIR", jobs)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "KEEP_JOBS", 0)
    for i in range(8):
        store.save("job_x_%02d" % i, {"job_id": "job_x_%02d" % i})
    assert len(list(jobs.glob("job_*.json"))) == 8


# ---------------------------------------------------------------------------
# Operational mode: live metocean, and the guard that keeps it shut
# ---------------------------------------------------------------------------

def test_live_metocean_is_shut_by_both_switches(monkeypatch):
    """OPEN_METEO_LIVE was declared, reported to the UI, and read by nothing.

    It is wired now, and both switches have to agree before a socket can open:
    OFFLINE is the demo default the boundary tests assert, OPEN_METEO_LIVE is
    the operator's explicit opt-in. Neither alone is enough.
    """
    from app import config
    from app.drift import fields

    for offline, live, expected in [
        (True, False, False),     # shipped default
        (True, True, False),      # opted in but still air-gapped
        (False, False, False),    # online but not asked for
        (False, True, True),      # deployed, explicitly enabled
    ]:
        monkeypatch.setattr(config, "OFFLINE", offline)
        monkeypatch.setattr(config, "OPEN_METEO_LIVE", live)
        assert fields.live_available() is expected, (
            "OFFLINE=%s OPEN_METEO_LIVE=%s should be %s" % (offline, live, expected))


def test_a_disabled_live_fetch_never_touches_the_network(monkeypatch):
    from app import config
    from app.drift import fields

    monkeypatch.setattr(config, "OFFLINE", True)
    monkeypatch.setattr(config, "OPEN_METEO_LIVE", True)

    def explode(*_a, **_k):
        raise AssertionError("fetch_live imported the prep lane while air-gapped")

    monkeypatch.setattr("importlib.util.spec_from_file_location", explode)
    assert fields.fetch_live("any_scene", 0.0, 0.0, None) is None


def test_a_failing_live_fetch_falls_back_to_cache(monkeypatch):
    """A live field that does not answer is a reason to use cache, never a
    reason to fail the run."""
    from app import config
    from app.drift import fields

    monkeypatch.setattr(config, "OFFLINE", False)
    monkeypatch.setattr(config, "OPEN_METEO_LIVE", True)
    monkeypatch.setattr(fields, "fetch_live", lambda *a, **k: None)

    field = fields.load_for_scene("gom_mc20_chronic_slick", 29.0, -89.0)
    assert field is not None
    assert field.live is False
    assert field.describe()["live"] is False


def test_a_cached_cube_is_never_labelled_live():
    from app.drift import fields

    for described in fields.list_cached():
        assert described["live"] is False, described["scene_id"]
        assert described["fetched_utc"] is None
