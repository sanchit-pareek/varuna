"""EPSG:4326 <-> local metres.

Uses a local azimuthal equidistant projection centred on the feature. pyproj is
used when present because it is exact; when it is missing we fall back to the
standard spherical aeqd formulas implemented in numpy. Both paths agree to well
under a metre over the few tens of kilometres this product ever needs, and the
fallback keeps the "optional library" rule honest.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:  # optional, exact
    from pyproj import CRS, Transformer

    _HAVE_PYPROJ = True
except Exception:  # pragma: no cover - exercised only on minimal installs
    _HAVE_PYPROJ = False

R_EARTH = 6371008.8  # mean Earth radius, metres (IUGG)


class LocalAEQD:
    """Azimuthal equidistant frame pinned at (lat0, lon0)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self._fwd = None
        self._inv = None
        if _HAVE_PYPROJ:
            crs = CRS.from_proj4(
                f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} "
                f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            )
            self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # -- forward ------------------------------------------------------------
    def to_m(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if self._fwd is not None:
            x, y = self._fwd.transform(lon, lat)
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return self._to_m_sphere(lon, lat)

    def _to_m_sphere(self, lon, lat):
        p0 = math.radians(self.lat0)
        l0 = math.radians(self.lon0)
        p = np.radians(lat)
        l = np.radians(lon)
        cos_c = np.sin(p0) * np.sin(p) + np.cos(p0) * np.cos(p) * np.cos(l - l0)
        cos_c = np.clip(cos_c, -1.0, 1.0)
        c = np.arccos(cos_c)
        k = np.where(np.abs(np.sin(c)) < 1e-12, 1.0, c / np.where(np.sin(c) == 0, 1.0, np.sin(c)))
        x = R_EARTH * k * np.cos(p) * np.sin(l - l0)
        y = R_EARTH * k * (np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(l - l0))
        return x, y

    # -- inverse ------------------------------------------------------------
    def to_deg(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self._inv is not None:
            lon, lat = self._inv.transform(x, y)
            return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        return self._to_deg_sphere(x, y)

    def _to_deg_sphere(self, x, y):
        p0 = math.radians(self.lat0)
        l0 = math.radians(self.lon0)
        rho = np.hypot(x, y)
        c = rho / R_EARTH
        safe = rho > 1e-9
        sin_c = np.sin(c)
        cos_c = np.cos(c)
        lat = np.where(
            safe,
            np.arcsin(np.clip(cos_c * math.sin(p0) + np.where(safe, y * sin_c * math.cos(p0) / np.where(safe, rho, 1.0), 0.0), -1.0, 1.0)),
            p0,
        )
        lon = l0 + np.arctan2(
            x * sin_c,
            rho * math.cos(p0) * cos_c - np.where(safe, y, 0.0) * math.sin(p0) * sin_c,
        )
        lon = np.where(safe, lon, l0)
        return np.degrees(lon), np.degrees(lat)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great circle distance in km. Accepts scalars or arrays."""
    lat1, lon1, lat2, lon2 = map(lambda v: np.radians(np.asarray(v, dtype=float)), (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * (R_EARTH / 1000.0) * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two compass angles, 0 to 180."""
    d = abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
    return d


def meters_per_degree(lat: float) -> Tuple[float, float]:
    """Local metres per degree of longitude and latitude."""
    lat_r = math.radians(lat)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_r) + 1.175 * math.cos(4 * lat_r)
    m_per_deg_lon = 111412.84 * math.cos(lat_r) - 93.5 * math.cos(3 * lat_r)
    return m_per_deg_lon, m_per_deg_lat
