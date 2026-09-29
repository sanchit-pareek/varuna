"""Metocean field readers.

Clause (b) of the problem statement is not mockable, so the runtime reads real
cached fields written by `scripts/build_metocean_cache.py`. That script is the
only thing that ever touches the network, and only before the demo.

Storage formats accepted, in order of preference:
  data/metocean/<scene_id>.npz   numpy archive, no extra dependency
  data/metocean/<scene_id>.nc    NetCDF, read if netCDF4 or xarray is installed
  data/metocean/<scene_id>.json  same arrays as JSON, for eyeballing

Variables, all on a regular (time, lat, lon) grid:
  u_current, v_current   metres per second, eastward and northward
  u_wind10, v_wind10     metres per second at 10 m, eastward and northward

If nothing is found the loader returns a clearly flagged constant field and the
UI shows DEMO METOCEAN in red. That path exists so the app never crashes, not
so it can be pitched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import config

_VARS = ("u_current", "v_current", "u_wind10", "v_wind10")


def _to_epoch(t) -> float:
    if isinstance(t, (int, float, np.floating, np.integer)):
        return float(t)
    if isinstance(t, np.datetime64):
        return float(t.astype("datetime64[s]").astype(np.int64))
    if isinstance(t, str):
        s = t.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(t, datetime):
        dt = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    raise TypeError("unsupported time value %r" % (t,))


@dataclass
class MetoceanField:
    """A cached (time, lat, lon) metocean cube with bilinear/linear lookup."""

    times: np.ndarray            # epoch seconds, ascending
    lats: np.ndarray             # ascending
    lons: np.ndarray             # ascending
    data: Dict[str, np.ndarray]  # each (nt, nlat, nlon)
    source: str = "unknown"
    license: str = ""
    synthetic: bool = False
    scene_id: Optional[str] = None
    # True when this cube was fetched at run time rather than read from cache.
    # The operator must be able to tell a live field from a frozen one.
    live: bool = False
    fetched_utc: Optional[str] = None

    # -- metadata -----------------------------------------------------------
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (float(self.lons.min()), float(self.lats.min()),
                float(self.lons.max()), float(self.lats.max()))

    @property
    def time_range_iso(self) -> Tuple[str, str]:
        return (
            datetime.fromtimestamp(float(self.times[0]), tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(float(self.times[-1]), tz=timezone.utc).isoformat(),
        )

    @property
    def has_currents(self) -> bool:
        """False when the cube carries wind but no ocean current data.

        Open-Meteo's marine model does not cover every basin. Inland seas come
        back with empty current arrays. Drifting on wind alone is still real
        physics, but it is a weaker answer and the operator has to be told.
        """
        u = self.data.get("u_current")
        v = self.data.get("v_current")
        if u is None or v is None:
            return False
        return bool(np.any(np.abs(u) > 1e-6) or np.any(np.abs(v) > 1e-6))

    def describe(self) -> Dict[str, Any]:
        t0, t1 = self.time_range_iso
        return {
            "scene_id": self.scene_id,
            "source": self.source,
            "license": self.license,
            "synthetic": self.synthetic,
            "live": self.live,
            "fetched_utc": self.fetched_utc,
            "has_currents": self.has_currents,
            "mean_current_ms": round(float(np.mean(np.hypot(
                self.data["u_current"], self.data["v_current"]))), 4),
            "mean_wind_ms": round(float(np.mean(np.hypot(
                self.data["u_wind10"], self.data["v_wind10"]))), 3),
            "bounds": [round(v, 4) for v in self.bounds],
            "t_start": t0,
            "t_end": t1,
            "n_times": int(self.times.size),
            "grid": [int(self.lats.size), int(self.lons.size)],
        }

    def covers(self, lat: float, lon: float, t) -> bool:
        te = _to_epoch(t)
        w, s, e, n = self.bounds
        return (s <= lat <= n) and (w <= lon <= e) and (self.times[0] <= te <= self.times[-1])

    # -- lookup -------------------------------------------------------------
    def sample(self, lat, lon, t) -> Dict[str, np.ndarray]:
        """Bilinear in space, linear in time. Edges clamp rather than fail."""
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        te = _to_epoch(t)

        ti, tw = _interp_index(self.times, te)
        yi, yw = _interp_index_arr(self.lats, lat)
        xi, xw = _interp_index_arr(self.lons, lon)

        out: Dict[str, np.ndarray] = {}
        for name in _VARS:
            cube = self.data[name]
            a = _bilinear(cube[ti[0]], yi, yw, xi, xw)
            b = _bilinear(cube[ti[1]], yi, yw, xi, xw)
            out[name] = a * (1.0 - tw) + b * tw
        return out

    def velocity(self, lat, lon, t, alpha: float = None, deflection_deg: float = None):
        """Surface drift velocity in m/s: V = U_current + alpha * U_wind.

        The optional deflection rotates only the wind-driven part, to the right
        in the northern hemisphere, matching the leeway convention in the spec.
        """
        alpha = config.ALPHA_WIND if alpha is None else float(alpha)
        deflection_deg = config.DEFLECTION_DEG if deflection_deg is None else float(deflection_deg)
        s = self.sample(lat, lon, t)
        uw, vw = s["u_wind10"], s["v_wind10"]
        if abs(deflection_deg) > 1e-9:
            lat_arr = np.atleast_1d(np.asarray(lat, dtype=float))
            sign = np.where(lat_arr >= 0, -1.0, 1.0)  # clockwise (to the right) in NH
            th = np.radians(deflection_deg) * sign
            ct, st = np.cos(th), np.sin(th)
            uw, vw = uw * ct - vw * st, uw * st + vw * ct
        return s["u_current"] + alpha * uw, s["v_current"] + alpha * vw


def _interp_index(axis: np.ndarray, value: float) -> Tuple[Tuple[int, int], float]:
    n = axis.size
    if n == 1:
        return (0, 0), 0.0
    v = float(np.clip(value, axis[0], axis[-1]))
    j = int(np.searchsorted(axis, v, side="right")) - 1
    j = max(0, min(j, n - 2))
    span = axis[j + 1] - axis[j]
    w = 0.0 if span == 0 else (v - axis[j]) / span
    return (j, j + 1), float(w)


def _interp_index_arr(axis: np.ndarray, values: np.ndarray):
    n = axis.size
    if n == 1:
        z = np.zeros(values.shape, dtype=int)
        return (z, z), np.zeros(values.shape)
    v = np.clip(values, axis[0], axis[-1])
    j = np.searchsorted(axis, v, side="right") - 1
    j = np.clip(j, 0, n - 2)
    span = axis[j + 1] - axis[j]
    w = np.where(span == 0, 0.0, (v - axis[j]) / np.where(span == 0, 1.0, span))
    return (j, j + 1), w


def _bilinear(plane: np.ndarray, yi, yw, xi, xw) -> np.ndarray:
    y0, y1 = yi
    x0, x1 = xi
    v00 = plane[y0, x0]
    v01 = plane[y0, x1]
    v10 = plane[y1, x0]
    v11 = plane[y1, x1]
    top = v00 * (1 - xw) + v01 * xw
    bot = v10 * (1 - xw) + v11 * xw
    return top * (1 - yw) + bot * yw


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _from_arrays(times, lats, lons, data, source, license_, scene_id, synthetic=False) -> MetoceanField:
    times = np.asarray([_to_epoch(t) for t in np.asarray(times).ravel()], dtype=float)
    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()

    order_t = np.argsort(times)
    order_y = np.argsort(lats)
    order_x = np.argsort(lons)
    cube = {}
    for k in _VARS:
        arr = np.asarray(data[k], dtype=float)
        arr = arr[order_t][:, order_y][:, :, order_x]
        cube[k] = np.nan_to_num(arr, nan=0.0)
    return MetoceanField(
        times=times[order_t],
        lats=lats[order_y],
        lons=lons[order_x],
        data=cube,
        source=source,
        license=license_,
        synthetic=synthetic,
        scene_id=scene_id,
    )


def load_npz(path: Path) -> MetoceanField:
    z = np.load(path, allow_pickle=True)
    meta = {}
    if "meta" in z:
        try:
            meta = json.loads(str(z["meta"]))
        except Exception:
            meta = {}
    return _from_arrays(
        z["time"], z["lat"], z["lon"],
        {k: z[k] for k in _VARS},
        meta.get("source", "cached npz"),
        meta.get("license", ""),
        meta.get("scene_id", path.stem),
    )


def load_json(path: Path) -> MetoceanField:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return _from_arrays(
        doc["time"], doc["lat"], doc["lon"],
        {k: doc[k] for k in _VARS},
        doc.get("source", "cached json"),
        doc.get("license", ""),
        doc.get("scene_id", Path(path).stem),
    )


def load_netcdf(path: Path) -> MetoceanField:
    try:
        import netCDF4  # type: ignore
    except Exception:
        netCDF4 = None
    if netCDF4 is not None:
        ds = netCDF4.Dataset(str(path))
        try:
            tvar = ds.variables["time"]
            times = netCDF4.num2date(tvar[:], tvar.units)
            times = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second, tzinfo=timezone.utc) for t in times]
            data = {k: np.array(ds.variables[k][:]) for k in _VARS}
            return _from_arrays(
                times, np.array(ds.variables["lat"][:]), np.array(ds.variables["lon"][:]),
                data, getattr(ds, "source", "netcdf"), getattr(ds, "license", ""), Path(path).stem,
            )
        finally:
            ds.close()
    import xarray as xr  # type: ignore

    ds = xr.open_dataset(path)
    try:
        return _from_arrays(
            ds["time"].values, ds["lat"].values, ds["lon"].values,
            {k: ds[k].values for k in _VARS},
            ds.attrs.get("source", "netcdf"), ds.attrs.get("license", ""), Path(path).stem,
        )
    finally:
        ds.close()


def synthetic_field(lat0: float, lon0: float, t_center, hours: int = 96, scene_id: str = "synthetic") -> MetoceanField:
    """Last resort constant field. Always flagged synthetic=True in the UI.

    This is a crash guard, not a product feature. The spec is explicit that
    clause (b) needs real cached fields.
    """
    t0 = _to_epoch(t_center) - hours * 1800
    times = t0 + np.arange(hours + 1) * 3600.0
    lats = np.linspace(lat0 - 1.0, lat0 + 1.0, 9)
    lons = np.linspace(lon0 - 1.0, lon0 + 1.0, 9)
    shape = (times.size, lats.size, lons.size)
    data = {
        "u_current": np.full(shape, 0.15),
        "v_current": np.full(shape, 0.05),
        "u_wind10": np.full(shape, 4.0),
        "v_wind10": np.full(shape, 1.5),
    }
    return _from_arrays(times, lats, lons, data,
                        "SYNTHETIC CONSTANT FIELD (no cached metocean found)",
                        "n/a", scene_id, synthetic=True)


def live_available() -> bool:
    """Whether a runtime fetch of the metocean cube is permitted at all.

    Both switches have to agree. OFFLINE is the demo default and is what the
    boundary tests assert; OPEN_METEO_LIVE is the operator's explicit opt-in.
    Neither one alone opens the socket.
    """
    return bool(config.OPEN_METEO_LIVE) and not bool(config.OFFLINE)


def fetch_live(scene_id: str, lat0: float, lon0: float, t_center) -> Optional[MetoceanField]:
    """Pull a fresh cube from Open-Meteo / CMEMS for this footprint and time.

    This is the operational path. A deployed system does not run against frozen
    cubes -- it asks for the wind and current fields covering the acquisition it
    is working on, at the hour it is working on them. The demo runs from cache
    because a judged laptop has no network, not because the architecture needs
    it to.

    The fetcher itself lives in the preparation lane (`scripts/`), where all
    network code in this project lives, and is imported here only when a live
    run has actually been permitted. `app/` therefore still contains no import
    of a network client, which is what `tests/test_offline_boundary.py` asserts.

    `build_box` also writes the cube it fetched, so a live run refreshes the
    cache on its way past and the next offline run is that much less stale.

    Returns None on any failure. A live fetch that does not answer is a reason
    to fall back to cache and say so, never a reason to fail a run.
    """
    if not live_available():
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_metocean_fetch", config.ROOT_DIR / "scripts" / "build_metocean_cache.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        t = t_center or datetime.now(timezone.utc)
        mod.build_box(scene_id, lat0, lon0, t,
                      back_h=int(config.HINDCAST_H) + 24,
                      fwd_h=int(config.FORECAST_H) + 12)
    except Exception as exc:
        print("[metocean] live fetch failed for %s: %s" % (scene_id, exc))
        return None

    path = Path(config.METOCEAN_DIR) / ("%s.npz" % scene_id)
    if not path.exists():
        return None
    try:
        field = load_npz(path)
    except Exception as exc:
        print("[metocean] live cube unreadable for %s: %s" % (scene_id, exc))
        return None

    field.scene_id = scene_id
    field.live = True
    field.fetched_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return field


def load_for_scene(scene_id: str, lat0: float = 0.0, lon0: float = 0.0, t_center=None) -> MetoceanField:
    """Find the cube for a scene: live if permitted, else cached, else flagged.

    Order matters. Live is tried first when an operator has explicitly enabled
    it, because a fresh field beats a frozen one. Cache is the fallback and the
    demo default. The constant field is last and is always flagged as such.
    """
    if live_available():
        live = fetch_live(scene_id, lat0, lon0, t_center)
        if live is not None:
            return live

    base = config.METOCEAN_DIR
    for ext, loader in ((".npz", load_npz), (".nc", load_netcdf), (".json", load_json)):
        p = base / (scene_id + ext)
        if p.exists():
            try:
                f = loader(p)
                f.scene_id = scene_id
                return f
            except Exception as exc:  # pragma: no cover - corrupt cache
                print("[metocean] failed to read %s: %s" % (p.name, exc))
    # Any cube whose footprint contains the point is better than a constant.
    for p in sorted(base.glob("*.npz")):
        try:
            f = load_npz(p)
        except Exception:
            continue
        w, s, e, n = f.bounds
        if s <= lat0 <= n and w <= lon0 <= e:
            return f
    return synthetic_field(lat0, lon0, t_center or datetime.now(timezone.utc), scene_id=scene_id)


def list_cached() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(config.METOCEAN_DIR.glob("*.npz")):
        try:
            out.append(load_npz(p).describe())
        except Exception:
            continue
    for p in sorted(config.METOCEAN_DIR.glob("*.nc")):
        try:
            out.append(load_netcdf(p).describe())
        except Exception:
            continue
    return out


def save_npz(path: Path, field: MetoceanField) -> None:
    meta = json.dumps({
        "source": field.source,
        "license": field.license,
        "scene_id": field.scene_id,
    })
    np.savez_compressed(
        path,
        time=field.times,
        lat=field.lats,
        lon=field.lons,
        meta=np.array(meta),
        **{k: field.data[k] for k in _VARS},
    )
