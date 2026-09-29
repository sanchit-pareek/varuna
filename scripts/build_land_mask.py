"""Cache a coastline so the forecast can report a coast impact flag.

The problem statement marks the land mask optional and says to skip the step
when it is missing, which is what `app/drift/land.py` does. This script closes
that gap properly rather than leaving it open.

Source: Natural Earth 1:10m physical land polygons, public domain. The full file
is 10 MB, most of it continents nowhere near any scene, so this clips to the
union of the indexed scene footprints plus a margin wide enough to cover a
forecast cone. What lands in `data/land/coastline.geojson` is usually well under
a megabyte.

Run it once, while online. The runtime only reads the file.

Usage:
    python scripts/build_land_mask.py
    python scripts/build_land_mask.py --margin-deg 3 --resolution 50m
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config, scenes as scenes_mod   # noqa: E402

SOURCES = {
    "10m": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson",
    "50m": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson",
}
LICENSE = "Natural Earth, public domain (naturalearthdata.com)"


def scene_boxes(margin: float) -> List[Tuple[float, float, float, float]]:
    boxes = []
    for s in scenes_mod.all_scenes(include_selftest=True):
        w, so, e, n = [float(v) for v in s.bounds]
        boxes.append((w - margin, so - margin, e + margin, n + margin))
    return boxes


def ring_bbox(ring: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return min(lons), min(lats), max(lons), max(lats)


def overlaps(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def clip_ring(ring: Sequence[Sequence[float]], box) -> List[List[float]]:
    """Sutherland-Hodgman clip of a ring against an axis aligned box.

    Without this, one land feature is the whole of Eurasia at 1:10m, which is
    hundreds of thousands of vertices that the runtime would then walk on every
    forecast. Clipping to the scene box turns that into the few hundred vertices
    of nearby coastline, which is all the coast impact flag needs and is what
    keeps the cached file small enough to commit.
    """
    w, s, e, n = box
    poly = [[float(p[0]), float(p[1])] for p in ring]

    def clip(subject, inside, intersect):
        if not subject:
            return []
        out = []
        prev = subject[-1]
        for cur in subject:
            cur_in, prev_in = inside(cur), inside(prev)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
            prev = cur
        return out

    def lerp_x(a, b, x):
        t = (x - a[0]) / ((b[0] - a[0]) or 1e-12)
        return [x, a[1] + t * (b[1] - a[1])]

    def lerp_y(a, b, y):
        t = (y - a[1]) / ((b[1] - a[1]) or 1e-12)
        return [a[0] + t * (b[0] - a[0]), y]

    poly = clip(poly, lambda p: p[0] >= w, lambda a, b: lerp_x(a, b, w))
    poly = clip(poly, lambda p: p[0] <= e, lambda a, b: lerp_x(a, b, e))
    poly = clip(poly, lambda p: p[1] >= s, lambda a, b: lerp_y(a, b, s))
    poly = clip(poly, lambda p: p[1] <= n, lambda a, b: lerp_y(a, b, n))

    out, last = [], None
    for p in poly:
        q = [round(p[0], 5), round(p[1], 5)]
        if q != last:
            out.append(q)
            last = q
    if len(out) >= 3 and out[0] != out[-1]:
        out.append(out[0])
    return out if len(out) >= 4 else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", choices=sorted(SOURCES), default="10m")
    ap.add_argument("--margin-deg", type=float, default=2.5,
                    help="margin around each scene, wide enough for a forecast cone")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    boxes = scene_boxes(args.margin_deg)
    if not boxes:
        raise SystemExit("no scenes indexed; fetch a scene first")
    print("clipping to %d scene footprint(s), margin %.1f deg" % (len(boxes), args.margin_deg))
    for b in boxes:
        print("   %s" % [round(v, 2) for v in b])

    url = SOURCES[args.resolution]
    print("\ndownloading %s" % url.rsplit("/", 1)[-1])
    with urllib.request.urlopen(url, timeout=300) as r:
        raw = r.read()
    print("   %.1f MB" % (len(raw) / 1e6))
    doc = json.loads(raw.decode("utf-8"))

    kept: List[Dict[str, Any]] = []
    total_rings = 0
    for feat in doc.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        polys = []
        if gtype == "Polygon":
            polys = [geom.get("coordinates") or []]
        elif gtype == "MultiPolygon":
            polys = geom.get("coordinates") or []
        for poly in polys:
            if not poly:
                continue
            ring = poly[0]
            total_rings += 1
            rb = ring_bbox(ring)
            for b in boxes:
                if not overlaps(rb, b):
                    continue
                clipped = clip_ring(ring, b)
                if not clipped:
                    continue
                kept.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [clipped]},
                    "properties": {"source": "natural_earth_%s" % args.resolution},
                })

    out_path = Path(args.out or (Path(config.DATA_DIR) / "land" / "coastline.geojson"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "properties": {
            "source": "Natural Earth %s physical land" % args.resolution,
            "license": LICENSE,
            "clipped_to_scenes": len(boxes),
            "margin_deg": args.margin_deg,
        },
        "features": kept,
    }), encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024.0
    print("\nkept %d of %d land polygons -> %s (%.0f KB)"
          % (len(kept), total_rings, out_path, size_kb))
    if not kept:
        print("No land within the margin of any scene. That is a real answer for an "
              "open ocean chip: the coast flag will report offshore.")
    print("Cite in the README: %s" % LICENSE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
