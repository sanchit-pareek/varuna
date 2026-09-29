"""Build an accurate land mask for the indexed scenes, from Natural Earth.

The coastline file this replaces was wrong in both directions, which is the
worst way for a mask to be wrong. Tested against known points it reported the
Santa Barbara mountains as sea and open Caspian water as land. A detector that
trusts that will happily call radar shadow on a hillside "mineral oil", and the
coast-impact flag on the forecast cone will fire over open water.

Natural Earth 1:10m land is public domain, is the reference the drift module's
own docstring recommends, and is authoritative enough for a coastline at the
scale a 22 km chip needs.

The global file is about 25 MB, which does not belong in the repository, so each
polygon is clipped to the scene footprints with Sutherland-Hodgman against the
bounding box. Clipping rather than filtering matters: keeping whole rings would
mean carrying the outline of North America to mask one headland.

Usage:
    python scripts/fetch_land_mask.py --all
    python scripts/fetch_land_mask.py --all --margin-deg 1.0 --force
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

SOURCE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
          "master/geojson/ne_10m_land.geojson")
# Natural Earth carries inland seas inside the LAND polygon and cuts them out
# with a separate lakes layer. Without this second layer the Caspian reads as
# land, and every detection in the Baku scene would be masked away as though
# it sat on a hillside.
LAKES = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
         "master/geojson/ne_10m_lakes.geojson")
LICENSE = ("Natural Earth 1:10m Physical Vectors, land and lakes. Public "
           "domain, made with Natural Earth.")

UA = {"User-Agent": "TideTrace/1.0 (SIH26143 academic project)"}

Point = Tuple[float, float]


def _inside(p: Point, edge: str, box: Tuple[float, float, float, float]) -> bool:
    w, s, e, n = box
    if edge == "w":
        return p[0] >= w
    if edge == "e":
        return p[0] <= e
    if edge == "s":
        return p[1] >= s
    return p[1] <= n


def _intersect(a: Point, b: Point, edge: str,
               box: Tuple[float, float, float, float]) -> Point:
    w, s, e, n = box
    ax, ay = a
    bx, by = b
    if edge in ("w", "e"):
        x = w if edge == "w" else e
        t = (x - ax) / (bx - ax) if bx != ax else 0.0
        return (x, ay + t * (by - ay))
    y = s if edge == "s" else n
    t = (y - ay) / (by - ay) if by != ay else 0.0
    return (ax + t * (bx - ax), y)


def clip_ring(ring: Sequence[Point],
              box: Tuple[float, float, float, float]) -> List[Point]:
    """Sutherland-Hodgman against an axis-aligned box.

    Exact for a convex clip region, which a bounding box is. A ring that falls
    entirely outside comes back empty and is dropped by the caller.
    """
    out = [tuple(p[:2]) for p in ring]
    for edge in ("w", "e", "s", "n"):
        if not out:
            return []
        clipped: List[Point] = []
        for i in range(len(out)):
            cur = out[i]
            prev = out[i - 1]
            cur_in = _inside(cur, edge, box)
            prev_in = _inside(prev, edge, box)
            if cur_in:
                if not prev_in:
                    clipped.append(_intersect(prev, cur, edge, box))
                clipped.append(cur)
            elif prev_in:
                clipped.append(_intersect(prev, cur, edge, box))
        out = clipped
    return out


def _rings_of(geom: Dict[str, Any]) -> List[Tuple[List[Point], bool]]:
    """Every ring in a geometry, paired with whether it is a hole.

    Taking only `coords[0]` -- the exterior -- is the obvious reading and it is
    wrong for this dataset. Natural Earth models inland seas as *holes* in the
    surrounding landmass, so discarding interior rings turns the Caspian into
    part of Eurasia: 144 of 144 sample points across the Baku footprint came
    back as land, and masking on that would have deleted the entire scene.

    Holes are returned as water and win over land in the point test.
    """
    kind = geom.get("type")
    coords = geom.get("coordinates") or []
    out: List[Tuple[List[Point], bool]] = []
    if kind == "Polygon":
        for i, ring in enumerate(coords):
            out.append((list(ring), i > 0))
    elif kind == "MultiPolygon":
        for poly in coords:
            for i, ring in enumerate(poly):
                out.append((list(ring), i > 0))
    return out


def scene_boxes(margin: float) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    out = []
    for s in scenes_mod.all_scenes(include_selftest=False):
        w, so, e, n = [float(v) for v in s.bounds]
        out.append((s.id, (w - margin, so - margin, e + margin, n + margin)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--margin-deg", type=float, default=1.5,
                    help="pad each footprint, so a forecast cone leaving the "
                         "chip still meets real coastline")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.all:
        raise SystemExit("pass --all")

    out_path = Path(config.DATA_DIR) / "land" / "coastline.geojson"
    if out_path.exists() and not args.force:
        raise SystemExit("%s exists; pass --force to rebuild" % out_path)

    boxes = scene_boxes(args.margin_deg)
    if not boxes:
        raise SystemExit("no scenes indexed")

    print("clipping Natural Earth 1:10m land to %d footprints, margin %.2f deg"
          % (len(boxes), args.margin_deg))
    for sid, b in boxes:
        print("   %-30s %s" % (sid, [round(v, 3) for v in b]))

    features: List[Dict[str, Any]] = []
    for url, kind, layer in ((SOURCE, "land", "ne_10m_land"),
                             (LAKES, "water", "ne_10m_lakes")):
        print("\ndownloading %s" % url)
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read()
        print("   %.1f MB" % (len(raw) / 1e6))
        data = json.loads(raw)

        for sid, box in boxes:
            kept = {"land": 0, "water": 0}
            for feat in data.get("features", []):
                for ring, is_hole in _rings_of(feat.get("geometry") or {}):
                    clipped = clip_ring(ring, box)
                    if len(clipped) < 4:
                        continue
                    # A hole in a land polygon is water, whichever layer it came
                    # from. A ring in the lakes layer is water either way.
                    this = "water" if (is_hole or kind == "water") else "land"
                    features.append({
                        "type": "Feature",
                        "properties": {"scene": sid, "kind": this,
                                       "source": layer},
                        "geometry": {"type": "Polygon",
                                     "coordinates": [[list(q) for q in clipped]]},
                    })
                    kept[this] += 1
            print("   %-30s %5d land, %5d water" % (sid, kept["land"], kept["water"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "license": LICENSE,
        "source": SOURCE,
        "margin_deg": args.margin_deg,
        "features": features,
    }), encoding="utf-8")

    size = out_path.stat().st_size
    n_land = sum(1 for f in features if f["properties"]["kind"] == "land")
    n_water = len(features) - n_land
    print("\nwrote %s  (%d land + %d water polygons, %.2f MB)"
          % (out_path, n_land, n_water, size / 1e6))
    print(LICENSE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
