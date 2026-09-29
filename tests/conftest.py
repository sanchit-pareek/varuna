"""Shared fixtures. Everything here is offline by construction."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _sandbox(tmp_path_factory):
    """Point the app at a throwaway data directory before it is imported.

    Tests must never touch the demo store: a test that quietly appends to
    data/ais/ais.sqlite would make the judged demo non reproducible.
    """
    data = tmp_path_factory.mktemp("tidetrace_data")
    os.environ["TIDETRACE_DATA"] = str(data)
    os.environ["TIDETRACE_MODELS"] = str(tmp_path_factory.mktemp("tidetrace_models"))
    os.environ["TIDETRACE_ALLOW_SELFTEST_SCENES"] = "1"
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]
    yield data


@pytest.fixture(scope="session")
def app_config(_sandbox):
    from app import config

    return config


@pytest.fixture
def uniform_field():
    """A metocean cube with a known, exactly uniform velocity.

    u_current = 1 m/s east, everything else zero. One hour of advection must
    move a particle 3600 m east, which is the spec's stated unit test.
    """
    from app.drift import fields as fields_mod

    t0 = datetime(2024, 2, 14, 0, 0, tzinfo=timezone.utc).timestamp()
    times = t0 + np.arange(97) * 3600.0
    lats = np.linspace(18.0, 20.0, 9)
    lons = np.linspace(70.5, 72.5, 9)
    shape = (times.size, lats.size, lons.size)
    data = {
        "u_current": np.ones(shape, dtype=float),
        "v_current": np.zeros(shape, dtype=float),
        "u_wind10": np.zeros(shape, dtype=float),
        "v_wind10": np.zeros(shape, dtype=float),
    }
    return fields_mod._from_arrays(times, lats, lons, data,
                                  "unit test uniform field", "n/a", "unit_test")


@pytest.fixture
def selftest_scene(_sandbox):
    """Generate the synthetic self test scene inside the sandbox."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_selftest_scene", ROOT / "scripts" / "make_selftest_scene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from app import config, scenes as scenes_mod
    from app.geo import raster as raster_mod

    out = mod.build(size=384)
    sar_path = Path(config.SAR_DIR) / "selftest_synthetic.tif"
    mask_path = Path(config.SAR_DIR) / "selftest_synthetic_mask.tif"
    raster_mod.write_geotiff(sar_path, out["sar"], out["transform"], "EPSG:4326")
    raster_mod.write_geotiff(mask_path, out["truth"].astype(np.uint8),
                             (1.0, 0.0, 0.0, 0.0, -1.0, 0.0), "EPSG:4326")

    scene = scenes_mod.describe_geotiff(
        sar_path, scene_id="selftest_synthetic",
        title="SELF TEST synthetic scene",
        t_sat="2024-02-14T05:40:00Z", mask_path=mask_path,
        source="synthetic", license_="n/a", selftest=True,
    )
    scenes_mod.save_index([scene])
    return scene


@pytest.fixture
def cached_metocean(selftest_scene, uniform_field):
    """Freeze a cube on disk for the scene so the pipeline is fully offline."""
    from app import config
    from app.drift import fields as fields_mod

    field = uniform_field
    field.scene_id = selftest_scene.id
    fields_mod.save_npz(Path(config.METOCEAN_DIR) / ("%s.npz" % selftest_scene.id), field)
    return field
