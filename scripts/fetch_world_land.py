"""Cache a world coastline so the chart base is a map at every zoom.

The satellite tiles are downloaded for the indexed scene footprints only, which
is the right trade for an offline demo: four small boxes instead of a planet.
The consequence was that zooming out left four imagery patches floating in an
empty wash, because the canvas chart base drew ocean and a graticule and nothing
else. That reads as a broken tile pipeline rather than as a deliberate cache.

Natural Earth 1:110m land is 138 KB and public domain. Drawn under the imagery
it gives continents at every zoom, offline, with no tiles at all. 1:50m is
pulled too for closer work, and both are simplified on the way in so the browser
is not asked to path tens of thousands of vertices on every pan.

Run once while online. Part of the preparation lane, never the demo.
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

BASE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "master/geojson/")
UA = {"User-Agent": "tidetrace/1.5 (SIH26143 offline console)"}
OUT = Path("data/land")
LICENSE = ("Natural Earth 1:110m and 1:50m Physical Vectors, land. "
           "Public domain, no attribution required.")

Point = Tuple[float, float]


def _perp_distance(p: Point, a: Point, b: Point) -> float:
    """Distance from p to the segment ab, in degrees. Good enough to simplify."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def simplify(ring: Sequence[Point], tol: float) -> List[Point]:
    """Ramer-Douglas-Peucker, iterative so a long coastline cannot blow the stack."""
    if len(ring) < 3 or tol <= 0:
        return list(ring)
    keep = [False] * len(ring)
    keep[0] = keep[-1] = True
    stack = [(0, len(ring) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        worst, at = 0.0, lo
        for i in range(lo + 1, hi):
            d = _perp_distance(ring[i], ring[lo], ring[hi])
            if d > worst:
                worst, at = d, i
        if worst > tol:
            keep[at] = True
            stack.append((lo, at))
            stack.append((at, hi))
    return [p for p, k in zip(ring, keep) if k]


def _rings(geom: Dict[str, Any]) -> List[List[Point]]:
    kind = geom.get("type")
    if kind == "Polygon":
        return [list(map(tuple, r)) for r in geom["coordinates"]]
    if kind == "MultiPolygon":
        return [list(map(tuple, r)) for poly in geom["coordinates"] for r in poly]
    return []


def _ring_extent(ring: Sequence[Point]) -> float:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def build(scale: str, tol: float, min_extent: float) -> Dict[str, Any]:
    url = "%sne_%s_land.geojson" % (BASE, scale)
    print("downloading %s" % url)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        raw = json.loads(r.read().decode("utf-8"))

    rings_in = rings_out = pts_in = pts_out = 0
    out: List[List[List[float]]] = []
    for feat in raw.get("features", []):
        for ring in _rings(feat.get("geometry") or {}):
            rings_in += 1
            pts_in += len(ring)
            # Drop specks. At world zoom an islet under a few tenths of a degree
            # is sub-pixel, and there are thousands of them.
            if _ring_extent(ring) < min_extent:
                continue
            simple = simplify(ring, tol)
            if len(simple) < 4:
                continue
            rings_out += 1
            pts_out += len(simple)
            out.append([[round(x, 4), round(y, 4)] for x, y in simple])

    print("  rings %d -> %d, vertices %d -> %d (%.0f%% smaller)"
          % (rings_in, rings_out, pts_in, pts_out,
             100.0 * (1 - pts_out / max(pts_in, 1))))
    # A flat array of rings, not a FeatureCollection: the browser draws these as
    # filled paths and never needs the per-feature properties.
    return {"scale": scale, "tolerance_deg": tol, "rings": out,
            "vertices": pts_out, "license": LICENSE}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Two levels of detail. The coarse one carries the whole globe at a glance;
    # the finer one takes over once a coastline is more than a few pixels of
    # decision. Anything closer than that is covered by real imagery anyway.
    for scale, tol, min_extent, name in (
        ("110m", 0.08, 0.35, "world_land_coarse.json"),
        ("50m", 0.02, 0.10, "world_land_detail.json"),
    ):
        doc = build(scale, tol, min_extent)
        path = out_dir / name
        path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        print("  wrote %s  %.0f KB" % (path, path.stat().st_size / 1024))


if __name__ == "__main__":
    main()
