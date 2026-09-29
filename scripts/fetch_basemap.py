"""Cache map tiles for the scene areas so the console has a real backdrop offline.

The map used to be a bare graticule outside the SAR footprint. That was an
honest consequence of the airplane-mode rule, since a live tile layer would fail
the moment the network went away, but it reads as broken and it robs the
operator of any geographic context: you cannot tell coast from open water, or
see where a forecast cone is heading.

So the tiles are cached, like everything else. Run this once while online. The
runtime then serves them from `data/basemap/` and never calls out.

Layers, all free to use with attribution:

    satellite   Esri World Imagery                    what the sea looks like
    ocean       Esri Ocean Basemap + Reference        bathymetry and place names
    seamark     OpenSeaMap seamarks (ODbL)            buoys, lights, lanes

Coverage is the indexed scene footprints plus a margin wide enough for a
forecast cone. Outside that, the map falls back to the graticule, which is the
correct behaviour: it shows you exactly where you have data and where you do not.

Usage:
    python scripts/fetch_basemap.py --all
    python scripts/fetch_basemap.py --all --layers satellite seamark --max-zoom 12
    python scripts/fetch_basemap.py --scene caspian_baku_seeps
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config, scenes as scenes_mod   # noqa: E402

UA = {"User-Agent": "TideTrace/1.0 (SIH26143 academic project; one-time offline cache)"}

LAYERS: Dict[str, Dict] = {
    "satellite": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "ext": "jpg",
        "attribution": "Esri World Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        "max_zoom": 13,
    },
    "ocean": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        "ext": "jpg",
        "attribution": "Esri Ocean Basemap: Esri, GEBCO, NOAA, National Geographic, and other contributors",
        "max_zoom": 11,
    },
    "ocean_labels": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}",
        "ext": "png",
        "attribution": "Esri Ocean Reference",
        "max_zoom": 11,
    },
    "seamark": {
        "url": "https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
        "ext": "png",
        "attribution": "OpenSeaMap contributors, ODbL",
        "min_zoom": 10,
        "max_zoom": 13,
    },
}

DEFAULT_LAYERS = ["satellite", "ocean", "ocean_labels", "seamark"]


def deg2tile(lat: float, lon: float, z: int) -> Tuple[int, int]:
    lat = max(-85.05112878, min(85.05112878, lat))
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tiles_for(bbox: Tuple[float, float, float, float], z: int) -> List[Tuple[int, int]]:
    w, s, e, n = bbox
    x0, y0 = deg2tile(n, w, z)     # north-west
    x1, y1 = deg2tile(s, e, z)     # south-east
    return [(x, y)
            for x in range(min(x0, x1), max(x0, x1) + 1)
            for y in range(min(y0, y1), max(y0, y1) + 1)]


def fetch(url: str, dest: Path, retries: int = 3, pause: float = 0.12) -> str:
    """Return 'saved', 'cached', 'missing' or 'failed'. Never raises."""
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                data = r.read()
            if not data:
                return "missing"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            time.sleep(pause)
            return "saved"
        except urllib.error.HTTPError as exc:
            # 404 simply means the provider has no tile there, which is normal
            # at high zoom over open water. Not a failure.
            if exc.code in (404, 204):
                return "missing"
            last = exc
        except Exception as exc:
            last = exc
        time.sleep(pause * (attempt + 2))
    print("    %s -> %s" % (url, last))
    return "failed"


def scene_bboxes(margin_deg: float) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    out = []
    for s in scenes_mod.all_scenes(include_selftest=False):
        w, so, e, n = [float(v) for v in s.bounds]
        out.append((s.id, (w - margin_deg, so - margin_deg, e + margin_deg, n + margin_deg)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--layers", nargs="*", default=DEFAULT_LAYERS,
                    choices=sorted(LAYERS), help="which layers to cache")
    ap.add_argument("--min-zoom", type=int, default=5)
    ap.add_argument("--max-zoom", type=int, default=13)
    ap.add_argument("--margin-deg", type=float, default=0.35,
                    help="margin around each scene, wide enough for a forecast cone")
    ap.add_argument("--limit", type=int, default=20000, help="safety cap on tile count")
    args = ap.parse_args()

    wanted = set(args.scene)
    boxes = [(sid, bb) for sid, bb in scene_bboxes(args.margin_deg)
             if not wanted or sid in wanted]
    if not boxes:
        raise SystemExit("no scenes matched; pass --all or --scene ID")
    if not (args.all or wanted):
        raise SystemExit("pass --all, or --scene ID")

    root = Path(config.DATA_DIR) / "basemap"
    root.mkdir(parents=True, exist_ok=True)

    print("caching zooms %d..%d, margin %.2f deg, layers: %s"
          % (args.min_zoom, args.max_zoom, args.margin_deg, ", ".join(args.layers)))
    for sid, bb in boxes:
        print("  %-30s %s" % (sid, [round(v, 3) for v in bb]))

    # Deduplicate: neighbouring scenes at low zoom share tiles.
    plan: Dict[str, set] = {name: set() for name in args.layers}
    for name in args.layers:
        top = min(args.max_zoom, LAYERS[name].get("max_zoom", 19))
        bottom = max(args.min_zoom, LAYERS[name].get("min_zoom", 0))
        for _sid, bb in boxes:
            for z in range(bottom, top + 1):
                for x, y in tiles_for(bb, z):
                    plan[name].add((z, x, y))

    total = sum(len(v) for v in plan.values())
    print("\n%d tiles to consider" % total)
    if total > args.limit:
        raise SystemExit("that is %d tiles, over the --limit of %d. Lower --max-zoom."
                         % (total, args.limit))

    stats = {}
    for name in args.layers:
        spec = LAYERS[name]
        saved = cached = missing = failed = 0
        items = sorted(plan[name])
        for i, (z, x, y) in enumerate(items, start=1):
            url = spec["url"].format(z=z, x=x, y=y)
            dest = root / name / str(z) / str(x) / ("%d.%s" % (y, spec["ext"]))
            result = fetch(url, dest)
            if result == "saved":
                saved += 1
            elif result == "cached":
                cached += 1
            elif result == "missing":
                missing += 1
            else:
                failed += 1
            if i % 200 == 0 or i == len(items):
                print("  %-13s %5d/%-5d  saved %d cached %d empty %d failed %d"
                      % (name, i, len(items), saved, cached, missing, failed), flush=True)
        stats[name] = {"saved": saved, "cached": cached, "missing": missing,
                       "failed": failed, "ext": spec["ext"],
                       "attribution": spec["attribution"]}

    # Merge, do not replace. Running with `--scene X` used to write a manifest
    # naming only X, so a single-scene top-up silently erased the record of
    # every scene cached before it. The tiles stayed on disk and the manifest
    # stopped mentioning them, which is the worst of both.
    previous: Dict[str, Any] = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}

    layers = dict(previous.get("layers") or {})
    layers.update({n: {"ext": LAYERS[n]["ext"], "attribution": LAYERS[n]["attribution"]}
                   for n in args.layers})
    scene_bounds = dict(previous.get("bounds") or {})
    scene_bounds.update({sid: [round(v, 5) for v in bb] for sid, bb in boxes})
    all_stats = dict(previous.get("stats") or {})
    all_stats.update(stats)

    manifest = {
        "layers": layers,
        "min_zoom": min(args.min_zoom, previous.get("min_zoom", args.min_zoom)),
        "max_zoom": max(args.max_zoom, previous.get("max_zoom", args.max_zoom)),
        "margin_deg": args.margin_deg,
        "scenes": sorted(scene_bounds),
        "bounds": scene_bounds,
        "stats": all_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print("\ncached %.1f MB in %s" % (size / 1e6, root))
    print("attribution to keep visible in the UI:")
    for name in args.layers:
        print("  %s" % LAYERS[name]["attribution"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
