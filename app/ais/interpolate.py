"""Track reconstruction: resampling, gap detection, dead reckoning.

AIS is irregular. Before any spatial join the spec requires each track to be
interpolated to a fixed cadence inside the search window. Two rules make the
result defensible:

  * Position is interpolated linearly in a local metre frame, not in raw
    degrees, so a track does not bend near the poles.
  * Course is interpolated on the circle, so 350 to 010 goes through north and
    not the long way round.

A gap longer than `gap_minutes` is not silently bridged as if it were data. The
bridging samples are marked `dead_reckoned=True`, and that flag is what the
scorer uses to raise a NON-REPORTING reason code. A dark segment is evidence,
so it has to stay visibly distinct from a reported one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..geo.crs import LocalAEQD


@dataclass
class Gap:
    start_ts: int
    end_ts: int
    minutes: float
    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_ts": int(self.start_ts),
            "end_ts": int(self.end_ts),
            "minutes": round(float(self.minutes), 1),
            "start": [round(self.start_lon, 6), round(self.start_lat, 6)],
            "end": [round(self.end_lon, 6), round(self.end_lat, 6)],
        }


@dataclass
class Track:
    """One vessel resampled onto a regular time base."""

    mmsi: int
    name: Optional[str]
    vessel_type_raw: Optional[str]
    ts: np.ndarray             # epoch seconds
    lon: np.ndarray
    lat: np.ndarray
    sog: np.ndarray            # knots
    cog: np.ndarray            # degrees
    dead_reckoned: np.ndarray  # bool, True where the position is bridged
    gaps: List[Gap] = field(default_factory=list)
    raw_count: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.ts.size)

    def max_gap_minutes(self) -> float:
        return max((g.minutes for g in self.gaps), default=0.0)

    def to_geojson(self, **props) -> Dict[str, Any]:
        """LineString plus the dead reckoned spans as separate features."""
        feats: List[Dict[str, Any]] = [{
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[round(float(a), 6), round(float(b), 6)]
                                         for a, b in zip(self.lon, self.lat)]},
            "properties": {"mmsi": self.mmsi, "name": self.name, "kind": "reported", **props},
        }]
        for g in self.gaps:
            m = (self.ts >= g.start_ts) & (self.ts <= g.end_ts)
            if m.sum() < 2:
                continue
            feats.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [[round(float(a), 6), round(float(b), 6)]
                                             for a, b in zip(self.lon[m], self.lat[m])]},
                "properties": {"mmsi": self.mmsi, "name": self.name, "kind": "dead_reckoned",
                               "gap_minutes": round(g.minutes, 1), **props},
            })
        return {"type": "FeatureCollection", "features": feats}

    def samples(self, stride: int = 1) -> List[Dict[str, Any]]:
        return [
            {
                "ts": int(self.ts[i]),
                "lon": round(float(self.lon[i]), 6),
                "lat": round(float(self.lat[i]), 6),
                "sog": round(float(self.sog[i]), 2),
                "cog": round(float(self.cog[i]), 1),
                "dr": bool(self.dead_reckoned[i]),
            }
            for i in range(0, self.n, max(1, stride))
        ]


def _circular_interp(t_new: np.ndarray, t: np.ndarray, deg: np.ndarray) -> np.ndarray:
    rad = np.radians(np.asarray(deg, dtype=float))
    s = np.interp(t_new, t, np.sin(rad))
    c = np.interp(t_new, t, np.cos(rad))
    return (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0


def resample(
    rows: Sequence[Any],
    t_start: int,
    t_end: int,
    step_seconds: int = 60,
    gap_minutes: float = 30.0,
) -> Optional[Track]:
    """Resample raw AIS rows onto a regular grid across [t_start, t_end]."""
    if not rows:
        return None
    ts = np.array([int(r["ts"]) for r in rows], dtype=float)
    order = np.argsort(ts)
    ts = ts[order]
    lon = np.array([float(r["lon"]) for r in rows], dtype=float)[order]
    lat = np.array([float(r["lat"]) for r in rows], dtype=float)[order]
    sog = np.array([float(r["sog"]) if r["sog"] is not None else np.nan for r in rows], dtype=float)[order]
    cog = np.array([float(r["cog"]) if r["cog"] is not None else np.nan for r in rows], dtype=float)[order]

    uniq, keep = np.unique(ts, return_index=True)
    ts, lon, lat, sog, cog = uniq, lon[keep], lat[keep], sog[keep], cog[keep]
    if ts.size < 2:
        return None

    sog = _fill_nan(sog)
    cog = _fill_nan(cog, circular=True, lon=lon, lat=lat, ts=ts)

    lo = max(float(t_start), float(ts[0]))
    hi = min(float(t_end), float(ts[-1]))
    if hi <= lo:
        return None
    grid = np.arange(lo, hi + 1e-6, float(step_seconds))
    if grid.size < 2:
        grid = np.array([lo, hi])

    frame = LocalAEQD(float(np.median(lat)), float(np.median(lon)))
    x, y = frame.to_m(lon, lat)
    xg = np.interp(grid, ts, x)
    yg = np.interp(grid, ts, y)
    lon_g, lat_g = frame.to_deg(xg, yg)

    sog_g = np.interp(grid, ts, sog)
    cog_g = _circular_interp(grid, ts, cog)

    gaps: List[Gap] = []
    dr = np.zeros(grid.shape, dtype=bool)
    dts = np.diff(ts)
    for i, d in enumerate(dts):
        minutes = d / 60.0
        if minutes >= gap_minutes:
            gaps.append(Gap(
                start_ts=int(ts[i]), end_ts=int(ts[i + 1]), minutes=float(minutes),
                start_lon=float(lon[i]), start_lat=float(lat[i]),
                end_lon=float(lon[i + 1]), end_lat=float(lat[i + 1]),
            ))
            dr |= (grid > ts[i]) & (grid < ts[i + 1])

    row0 = rows[0]
    return Track(
        mmsi=int(row0["mmsi"]),
        name=row0["vessel_name"],
        vessel_type_raw=row0["vessel_type"],
        ts=grid.astype(np.int64),
        lon=np.asarray(lon_g, dtype=float),
        lat=np.asarray(lat_g, dtype=float),
        sog=sog_g,
        cog=cog_g,
        dead_reckoned=dr,
        gaps=gaps,
        raw_count=int(ts.size),
        meta={
            "length": row0["length"],
            "width": row0["width"],
            "draft": row0["draft"],
            "imo": row0["imo"],
            "call_sign": row0["call_sign"],
            "status": row0["status"],
            "cargo": row0["cargo"],
            "source": row0["source"] if "source" in row0.keys() else None,
            "step_seconds": int(step_seconds),
        },
    )


def _fill_nan(values: np.ndarray, circular: bool = False, lon=None, lat=None, ts=None) -> np.ndarray:
    """Fill missing SOG/COG. COG falls back to the bearing implied by motion."""
    v = np.asarray(values, dtype=float).copy()
    bad = ~np.isfinite(v)
    if not bad.any():
        return v
    if bad.all():
        if circular and lon is not None and len(lon) > 1:
            return _bearing_series(lon, lat)
        return np.zeros_like(v)
    good = ~bad
    idx = np.arange(v.size)
    if circular:
        rad = np.radians(v[good])
        s = np.interp(idx, idx[good], np.sin(rad))
        c = np.interp(idx, idx[good], np.cos(rad))
        return (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
    v[bad] = np.interp(idx[bad], idx[good], v[good])
    return v


def _bearing_series(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    from ..geo.crs import bearing_deg

    out = np.zeros(len(lon))
    for i in range(len(lon)):
        j = min(i + 1, len(lon) - 1)
        k = max(i - 1, 0)
        if j == k:
            out[i] = 0.0
        else:
            out[i] = bearing_deg(lat[k], lon[k], lat[j], lon[j])
    return out


def build_tracks(
    grouped: Dict[int, List[Any]],
    t_start: int,
    t_end: int,
    step_seconds: int = 60,
    gap_minutes: float = 30.0,
) -> Dict[int, Track]:
    out: Dict[int, Track] = {}
    for mmsi, rows in grouped.items():
        tr = resample(rows, t_start, t_end, step_seconds=step_seconds, gap_minutes=gap_minutes)
        if tr is not None:
            out[mmsi] = tr
    return out
