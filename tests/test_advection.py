"""Physics tests.

The headline assertion is the one the spec names: a particle in a 1 m/s eastward
current, stepped for one hour, must move about 3.6 km east. If this drifts, the
whole hindcast is wrong and every origin and every suspect is wrong with it.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest


T0 = datetime(2024, 2, 14, 6, 0, tzinfo=timezone.utc)


def test_one_metre_per_second_for_one_hour_is_3_6_km(uniform_field):
    from app.drift import advection
    from app.geo.crs import haversine_km

    lon0 = np.array([71.5])
    lat0 = np.array([19.0])
    run = advection.advect(lon0, lat0, T0, hours=1, field_src=uniform_field,
                           direction="forward", current_noise=0.0, wind_noise=0.0)

    d_km = float(haversine_km(lat0[0], lon0[0], run.lat[-1, 0], run.lon[-1, 0]))
    assert d_km == pytest.approx(3.6, abs=0.02), "expected 3.6 km, got %.4f" % d_km
    # Motion must be purely eastward.
    assert run.lon[-1, 0] > lon0[0]
    assert abs(float(run.lat[-1, 0]) - lat0[0]) < 1e-4


def test_backward_run_undoes_forward_run(uniform_field):
    from app.drift import advection

    lon0 = np.array([71.5])
    lat0 = np.array([19.0])
    fwd = advection.advect(lon0, lat0, T0, hours=12, field_src=uniform_field,
                           direction="forward", current_noise=0.0, wind_noise=0.0)
    back = advection.advect(fwd.lon[-1], fwd.lat[-1], fwd.times[-1], hours=12,
                            field_src=uniform_field, direction="backward",
                            current_noise=0.0, wind_noise=0.0)
    assert float(back.lon[-1, 0]) == pytest.approx(float(lon0[0]), abs=1e-4)
    assert float(back.lat[-1, 0]) == pytest.approx(float(lat0[0]), abs=1e-4)


def test_wind_contributes_exactly_alpha_of_its_speed():
    """V = U_current + 0.03 * U_wind, with the deflection switched off."""
    from app import config
    from app.drift import advection, fields as fields_mod
    from app.geo.crs import haversine_km

    t0 = T0.timestamp()
    times = t0 + np.arange(5) * 3600.0
    lats = np.linspace(18.5, 19.5, 5)
    lons = np.linspace(71.0, 72.0, 5)
    shape = (times.size, lats.size, lons.size)
    field = fields_mod._from_arrays(
        times, lats, lons,
        {"u_current": np.zeros(shape), "v_current": np.zeros(shape),
         "u_wind10": np.full(shape, 10.0), "v_wind10": np.zeros(shape)},
        "unit test wind only", "n/a", "wind_only")

    run = advection.advect(np.array([71.5]), np.array([19.0]), T0, hours=1,
                           field_src=field, direction="forward",
                           deflection_deg=0.0, current_noise=0.0, wind_noise=0.0)
    d_km = float(haversine_km(19.0, 71.5, run.lat[-1, 0], run.lon[-1, 0]))
    expected = config.ALPHA_WIND * 10.0 * 3600.0 / 1000.0   # 1.08 km at alpha 0.03
    assert d_km == pytest.approx(expected, rel=0.01)


def test_ensemble_spreads_and_origin_rule_is_bounded(uniform_field):
    from app import config
    from app.drift import advection

    lon0, lat0 = advection.seed_particles(None, (71.5, 19.0), 40, fallback_radius_km=0.5)
    run = advection.advect(lon0, lat0, T0, hours=48, field_src=uniform_field,
                           direction="backward")
    assert run.spread_km[0] < run.spread_km[-1], "ensemble must diverge over time"

    i = advection.pick_origin_index(run)
    dt_h = config.DT_SECONDS / 3600.0
    assert config.ORIGIN_H_MIN - 1e-6 <= i * dt_h <= config.ORIGIN_H_MAX + 1e-6


def test_seeding_stays_inside_the_polygon():
    from app.drift import advection
    from app.geo.geometry import point_in_ring

    ring = [(71.50, 19.00), (71.54, 19.00), (71.54, 19.02), (71.50, 19.02), (71.50, 19.00)]
    lon, lat = advection.seed_particles(ring, (71.52, 19.01), 60)
    inside = sum(1 for a, b in zip(lon, lat) if point_in_ring(a, b, ring))
    assert inside == 60


def test_advection_is_deterministic(uniform_field):
    """A judged demo has to give the same answer twice."""
    from app.drift import advection

    kw = dict(field_src=uniform_field, direction="backward", hours=6)
    a = advection.advect(np.array([71.5] * 20), np.array([19.0] * 20), T0, **kw)
    b = advection.advect(np.array([71.5] * 20), np.array([19.0] * 20), T0, **kw)
    assert np.allclose(a.lon, b.lon) and np.allclose(a.lat, b.lat)
