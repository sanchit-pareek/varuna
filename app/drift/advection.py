"""Lagrangian surface advection in numpy.

OpenDrift is deliberately not a runtime dependency (GPL, heavy, and the judged
demo has to install on a laptop in minutes). This module implements the same
2D surface model the spec pins down:

    V = U_current + ALPHA_WIND * U_wind10        with ALPHA_WIND = 0.03

which is the value OpenDrift's own maintainers quote for wind drift when Stokes
drift is switched off. An optional leeway deflection rotates the wind term to
the right in the northern hemisphere.

Integration is RK2 (midpoint) with a one hour step by default. Degrees are
converted to metres through a local azimuthal equidistant frame re-centred on
the ensemble each step, so nothing degrades with latitude.

Backward runs are the same integrator with a negative dt. That is what makes
the hindcast honest: it is the forward physics run in reverse, not a separate
heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from ..geo.crs import LocalAEQD
from ..geo.geometry import hull_ring, spread_radius_km
from .fields import MetoceanField


def _utc(t) -> datetime:
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    s = str(t).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class EnsembleRun:
    """Result of one ensemble integration."""

    times: List[datetime]                 # length nt
    lon: np.ndarray                       # (nt, n_particles)
    lat: np.ndarray                       # (nt, n_particles)
    direction: str                        # "backward" or "forward"
    spread_km: np.ndarray                 # (nt,) 90th percentile radius
    mean_lon: np.ndarray                  # (nt,)
    mean_lat: np.ndarray                  # (nt,)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.times)

    def envelope(self, index: int) -> List[Tuple[float, float]]:
        return hull_ring(self.lon[index], self.lat[index])

    def track_geojson(self) -> Dict[str, Any]:
        coords = [[float(a), float(b)] for a, b in zip(self.mean_lon, self.mean_lat)]
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "direction": self.direction,
                "times": [t.isoformat() for t in self.times],
                "spread_km": [round(float(v), 3) for v in self.spread_km],
            },
        }

    def hourly(self) -> List[Dict[str, Any]]:
        return [
            {
                "t": self.times[i].isoformat(),
                "lon": round(float(self.mean_lon[i]), 6),
                "lat": round(float(self.mean_lat[i]), 6),
                "spread_km": round(float(self.spread_km[i]), 3),
            }
            for i in range(self.n_steps)
        ]


def seed_particles(
    ring: Optional[Sequence[Tuple[float, float]]],
    centroid: Tuple[float, float],
    n: int,
    fallback_radius_km: float = 1.0,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Seed particles inside the slick polygon, or in a disc if it is tiny.

    Rejection sampling inside the ring keeps the initial cloud shaped like the
    real slick, which is what makes the backward envelope meaningful.
    """
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    clon, clat = float(centroid[0]), float(centroid[1])

    if ring and len(ring) >= 4:
        from ..geo.geometry import point_in_ring

        lons = np.array([p[0] for p in ring])
        lats = np.array([p[1] for p in ring])
        w, e = float(lons.min()), float(lons.max())
        s, nn = float(lats.min()), float(lats.max())
        out_lon: List[float] = []
        out_lat: List[float] = []
        guard = 0
        while len(out_lon) < n and guard < 400 * n:
            guard += 1
            lo = rng.uniform(w, e)
            la = rng.uniform(s, nn)
            if point_in_ring(lo, la, ring):
                out_lon.append(lo)
                out_lat.append(la)
        if len(out_lon) >= max(4, n // 4):
            while len(out_lon) < n:
                k = rng.integers(0, len(out_lon))
                out_lon.append(out_lon[k])
                out_lat.append(out_lat[k])
            return np.array(out_lon[:n]), np.array(out_lat[:n])

    frame = LocalAEQD(clat, clon)
    r = fallback_radius_km * 1000.0 * np.sqrt(rng.random(n))
    th = rng.uniform(0, 2 * np.pi, n)
    lon, lat = frame.to_deg(r * np.cos(th), r * np.sin(th))
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def advect(
    lon0: np.ndarray,
    lat0: np.ndarray,
    t0,
    hours: int,
    field_src: MetoceanField,
    direction: str = "forward",
    dt_seconds: int = None,
    alpha: float = None,
    deflection_deg: float = None,
    current_noise: float = None,
    wind_noise: float = None,
    seed: int = None,
) -> EnsembleRun:
    """Integrate an ensemble for `hours`, forwards or backwards.

    Per-particle velocity noise is drawn once and held for the whole run, which
    is the standard way to build a drift cone: each member is a plausible
    realisation of the field error, not a random walk that averages itself out.
    """
    dt = int(config.DT_SECONDS if dt_seconds is None else dt_seconds)
    alpha = config.ALPHA_WIND if alpha is None else float(alpha)
    deflection_deg = config.DEFLECTION_DEG if deflection_deg is None else float(deflection_deg)
    cn = config.CURRENT_NOISE_MS if current_noise is None else float(current_noise)
    wn = config.WIND_NOISE_MS if wind_noise is None else float(wind_noise)

    sign = -1.0 if direction == "backward" else 1.0
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)

    lon = np.asarray(lon0, dtype=float).copy()
    lat = np.asarray(lat0, dtype=float).copy()
    n = lon.size

    du = rng.normal(0.0, cn, n)
    dv = rng.normal(0.0, cn, n)
    dwu = rng.normal(0.0, wn, n)
    dwv = rng.normal(0.0, wn, n)

    t = _utc(t0)
    n_steps = max(1, int(round(hours * 3600.0 / dt)))

    times = [t]
    lons = [lon.copy()]
    lats = [lat.copy()]

    for _ in range(n_steps):
        u1, v1 = _velocity(field_src, lat, lon, t, alpha, deflection_deg, du, dv, dwu, dwv)
        frame = LocalAEQD(float(np.mean(lat)), float(np.mean(lon)))
        x, y = frame.to_m(lon, lat)

        # RK2 midpoint
        xm = x + sign * u1 * (dt / 2.0)
        ym = y + sign * v1 * (dt / 2.0)
        lon_m, lat_m = frame.to_deg(xm, ym)
        t_mid = t + timedelta(seconds=sign * dt / 2.0)
        u2, v2 = _velocity(field_src, lat_m, lon_m, t_mid, alpha, deflection_deg, du, dv, dwu, dwv)

        x2 = x + sign * u2 * dt
        y2 = y + sign * v2 * dt
        lon, lat = frame.to_deg(x2, y2)
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        t = t + timedelta(seconds=sign * dt)

        times.append(t)
        lons.append(lon.copy())
        lats.append(lat.copy())

    LON = np.vstack(lons)
    LAT = np.vstack(lats)
    spread = np.array([spread_radius_km(LON[i], LAT[i]) for i in range(LON.shape[0])])
    return EnsembleRun(
        times=times,
        lon=LON,
        lat=LAT,
        direction=direction,
        spread_km=spread,
        mean_lon=np.median(LON, axis=1),
        mean_lat=np.median(LAT, axis=1),
        meta={
            "alpha_wind": alpha,
            "deflection_deg": deflection_deg,
            "dt_seconds": dt,
            "n_particles": int(n),
            "current_noise_ms": cn,
            "wind_noise_ms": wn,
            "metocean_source": field_src.source,
            "metocean_synthetic": bool(field_src.synthetic),
        },
    )


def _velocity(field_src, lat, lon, t, alpha, deflection_deg, du, dv, dwu, dwv):
    u, v = field_src.velocity(lat, lon, t, alpha=alpha, deflection_deg=deflection_deg)
    u = np.asarray(u, dtype=float) + du + alpha * dwu
    v = np.asarray(v, dtype=float) + dv + alpha * dwv
    return u, v


def pick_origin_index(run: EnsembleRun, trigger_km: float = None,
                      h_min: float = None, h_max: float = None) -> int:
    """Frozen origin rule from the spec.

    Walk the backward run and take the first hour where the ensemble spread
    radius exceeds SPREAD_TRIGGER_KM, clipped to [ORIGIN_H_MIN, ORIGIN_H_MAX].
    Rationale: a slick that has drifted long enough for the plausible-origin
    cloud to reach that width is where the physics stops being informative, so
    that is the honest place to stop and hand over to AIS.
    """
    trigger_km = config.SPREAD_TRIGGER_KM if trigger_km is None else float(trigger_km)
    h_min = config.ORIGIN_H_MIN if h_min is None else float(h_min)
    h_max = config.ORIGIN_H_MAX if h_max is None else float(h_max)

    dt_h = config.DT_SECONDS / 3600.0
    i_min = int(round(h_min / dt_h))
    i_max = int(round(h_max / dt_h))
    last = run.n_steps - 1
    i_min = max(1, min(i_min, last))
    i_max = max(i_min, min(i_max, last))

    hit = None
    for i in range(i_min, i_max + 1):
        if run.spread_km[i] >= trigger_km:
            hit = i
            break
    if hit is None:
        hit = i_max
    return int(hit)
