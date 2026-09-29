"""Demo scene registry.

A scene is one georeferenced Sigma0 chip plus the metadata the pipeline needs:
acquisition time, an optional ground truth mask, and whether real AIS exists for
its footprint. The registry lives in `data/sar/scenes.json` and is written by
`scripts/prepare_scenes.py`; anything already on disk that is missing from the
index is picked up by a filesystem scan so a dropped-in TIFF still shows up.

The `ais_mode` field carries the decision the spec insists on making in code and
not on a slide: if the chip footprint falls inside MarineCadastre NAIS coverage
then real AIS is expected, otherwise simulated traffic over that same real
geobox and time window is the sanctioned fallback.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .geo import raster as raster_mod

# Rough MarineCadastre / US NAIS coverage boxes (west, south, east, north).
NAIS_BOXES = [
    (-98.0, 17.5, -80.0, 31.0),    # Gulf of Mexico
    (-82.0, 24.0, -64.0, 45.5),    # US East coast
    (-71.0, 40.0, -66.0, 45.0),    # Gulf of Maine
    (-126.0, 32.0, -116.0, 49.5),  # US West coast
    (-179.9, 51.0, -129.0, 72.0),  # Alaska
    (-160.5, 18.5, -154.0, 23.0),  # Hawaii
    (-68.0, 17.0, -64.0, 19.0),    # Puerto Rico and USVI
]


def in_nais_coverage(bbox: Tuple[float, float, float, float]) -> bool:
    w, s, e, n = bbox
    clon, clat = (w + e) / 2.0, (s + n) / 2.0
    for bw, bs, be, bn in NAIS_BOXES:
        if bw <= clon <= be and bs <= clat <= bn:
            return True
    return False


@dataclass
class Scene:
    id: str
    title: str
    sar_path: str
    t_sat: str
    bounds: List[float]                     # west, south, east, north
    centroid: List[float]                   # lon, lat
    crs: str = "EPSG:4326"
    mask_path: Optional[str] = None         # ground truth, optional
    thumbnail: Optional[str] = None
    backdrop: Optional[str] = None
    source: str = ""
    license: str = ""
    ais_mode: str = "simulated"             # "real" or "simulated"
    notes: str = ""
    selftest: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["exists"] = Path(self.sar_path).exists()
        return d


def _index_path() -> Path:
    return Path(config.SCENES_INDEX)


def resolve_path(value: Optional[str]) -> Optional[str]:
    """Accept absolute or repo relative paths in the index.

    Paths are stored relative to data/ so the repository can be moved, copied to
    the demo laptop, or zipped for submission without the index going stale.
    """
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    for base in (config.DATA_DIR, config.ROOT_DIR):
        cand = Path(base) / p
        if cand.exists():
            return str(cand)
    return str(Path(config.DATA_DIR) / p)


def _relativise(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    p = Path(value)
    try:
        return p.resolve().relative_to(Path(config.DATA_DIR).resolve()).as_posix()
    except (ValueError, OSError):
        return str(p)


def load_index() -> List[Scene]:
    p = _index_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: List[Scene] = []
    for item in raw.get("scenes", []):
        known = {k: v for k, v in item.items() if k in Scene.__annotations__}
        try:
            scene = Scene(**known)
        except TypeError:
            continue
        scene.sar_path = resolve_path(scene.sar_path)
        scene.mask_path = resolve_path(scene.mask_path)
        out.append(scene)
    return out


def save_index(scenes: List[Scene]) -> Path:
    p = _index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for s in scenes:
        d = asdict(s)
        d["sar_path"] = _relativise(d.get("sar_path"))
        d["mask_path"] = _relativise(d.get("mask_path"))
        items.append(d)
    p.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "scenes": items,
    }, indent=2), encoding="utf-8")
    return p


def upsert(scenes: List[Scene]) -> Path:
    """Merge scenes into the index, re-reading it first.

    Scripts run for minutes at a time and several can be in flight at once. A
    plain save from a stale in-memory copy silently drops whatever another
    script indexed in the meantime, which is how a scene quietly disappears.
    """
    current = {s.id: s for s in load_index()}
    for s in scenes:
        current[s.id] = s
    return save_index(list(current.values()))


def describe_geotiff(path: Path, scene_id: str = None, title: str = None,
                     t_sat: str = None, mask_path: Path = None,
                     source: str = "", license_: str = "",
                     selftest: bool = False, notes: str = "") -> Scene:
    """Open a chip, read its footprint, and decide the AIS mode from geography."""
    path = Path(path)
    sar = raster_mod.load_sar(path)
    bounds = sar.bounds_lonlat()
    clon, clat = sar.centre_lonlat()
    scene_id = scene_id or path.stem
    t = t_sat or _time_from_meta(sar) or datetime.now(timezone.utc).replace(
        microsecond=0).isoformat()
    return Scene(
        id=scene_id,
        title=title or scene_id.replace("_", " "),
        sar_path=str(path),
        t_sat=t,
        bounds=[round(float(v), 6) for v in bounds],
        centroid=[round(float(clon), 6), round(float(clat), 6)],
        crs=sar.crs,
        mask_path=str(mask_path) if mask_path else None,
        source=source,
        license=license_,
        ais_mode="real" if in_nais_coverage(bounds) else "simulated",
        selftest=selftest,
        notes=notes,
    )


def _time_from_meta(sar) -> Optional[str]:
    raw = (sar.meta or {}).get("datetime")
    if not raw:
        return None
    s = str(raw).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def scan_disk() -> List[Scene]:
    """Pick up any georeferenced TIFF in data/sar that is not already indexed.

    Anything whose name marks it as a self test raster is discovered as
    `selftest=True`, so a stray synthetic file can never be promoted into the
    demo list just because it lost its index entry.
    """
    known = set()
    for s in load_index():
        try:
            known.add(Path(s.sar_path).resolve())
        except OSError:
            continue
    found: List[Scene] = []
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for p in sorted(Path(config.SAR_DIR).glob(pattern)):
            if "_mask" in p.stem.lower():
                continue
            try:
                if p.resolve() in known:
                    continue
            except OSError:
                continue
            is_selftest = "selftest" in p.stem.lower() or "synthetic" in p.stem.lower()
            try:
                found.append(describe_geotiff(
                    p,
                    notes="auto discovered on disk, not in the scene index",
                    selftest=is_selftest,
                ))
            except Exception:
                continue
    return found


def all_scenes(include_selftest: bool = None) -> List[Scene]:
    if include_selftest is None:
        include_selftest = config.ALLOW_SELFTEST_SCENES
    scenes = load_index() + scan_disk()
    seen = set()
    out: List[Scene] = []
    for s in scenes:
        if s.id in seen:
            continue
        if s.selftest and not include_selftest:
            continue
        seen.add(s.id)
        out.append(s)
    return out


def get(scene_id: str) -> Optional[Scene]:
    for s in all_scenes(include_selftest=True):
        if s.id == scene_id:
            return s
    return None


def load_raster(scene: Scene):
    return raster_mod.load_sar(scene.sar_path)


def load_truth(scene: Scene, sar=None):
    """Ground truth mask with the SAR georeference forced onto it."""
    if not scene.mask_path or not Path(scene.mask_path).exists():
        return None
    sar = sar or load_raster(scene)
    return raster_mod.load_mask_with_georef(scene.mask_path, sar)
