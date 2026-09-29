"""Cut a real, georeferenced Sentinel-1 chip for the demo set.

The Zenodo archives are the training corpus, but their image parts are 10 to 40
GB 7z files, which is not something you download the night before a demo just to
get five chips on screen. This script gets real Sentinel-1 imagery the cheap way:
a windowed read out of the cloud optimised GeoTIFFs that Microsoft Planetary
Computer serves for the Sentinel-1 RTC collection. Only the requested window
crosses the network, so a 2048 pixel chip costs a few megabytes and a few
seconds instead of tens of gigabytes.

What you get is real Sentinel-1 IW GRD, radiometrically terrain corrected, in
its native UTM grid with a real affine transform, converted to dB so it matches
the Zenodo Sigma0 convention the model was trained on.

No account and no API key. Planetary Computer issues the read token anonymously.
Run this while online. The chip is then a normal file in data/sar and the
product never touches the network again.

Usage:
    python scripts/fetch_sentinel1_scene.py --preset mc20
    python scripts/fetch_sentinel1_scene.py --preset all
    python scripts/fetch_sentinel1_scene.py --lat 28.94 --lon -88.97 --id my_scene \
        --start 2023-12-01 --end 2023-12-31 --size-km 20
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, scenes as scenes_mod   # noqa: E402

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/%s"
COLLECTION = "sentinel-1-rtc"
LICENSE = ("Contains modified Copernicus Sentinel data. Sentinel-1 RTC hosted by "
           "Microsoft Planetary Computer, CC BY 4.0.")

# Sites chosen for a reason, not at random.
PRESETS: Dict[str, Dict[str, Any]] = {
    "mc20": {
        "id": "gom_mc20_chronic_slick",
        "title": "Gulf of Mexico, MC20 chronic discharge site",
        "lat": 28.938,
        "lon": -88.970,
        "start": "2023-04-01",
        "end": "2023-10-31",
        "size_km": 22.0,
        "notes": ("Long running surface expression at the MC20 site. Inside "
                  "MarineCadastre NAIS coverage, so real AIS can be ingested "
                  "for this footprint instead of simulated traffic."),
    },
    "santa_barbara": {
        "id": "santa_barbara_seeps",
        "title": "Santa Barbara Channel natural seeps",
        "lat": 34.375,
        "lon": -119.870,
        "start": "2023-06-01",
        "end": "2023-08-31",
        "size_km": 22.0,
        "notes": ("Natural hydrocarbon seeps plus heavy shipping. Inside NAIS "
                  "coverage. Useful as a hard case: real dark films that are "
                  "not a vessel discharge."),
    },
    "caspian_baku": {
        "id": "caspian_baku_seeps",
        "title": "Caspian Sea, Baku offshore field",
        "lat": 40.190,
        "lon": 50.480,
        "start": "2023-04-01",
        "end": "2023-10-31",
        "size_km": 22.0,
        "notes": ("Long established offshore production area with persistent "
                  "surface films and heavy local traffic. Outside MarineCadastre "
                  "coverage, so simulated traffic over this real geobox and time "
                  "window is the sanctioned path for clause (c)."),
    },
    "gulf_of_suez": {
        "id": "gulf_of_suez_slicks",
        "title": "Gulf of Suez, tanker route",
        "lat": 28.500,
        "lon": 33.150,
        "start": "2023-04-01",
        "end": "2023-10-31",
        "size_km": 22.0,
        "notes": ("Dense tanker traffic and a documented history of operational "
                  "discharge. Outside public AIS coverage, so traffic is simulated."),
    },
    "arabian_sea": {
        "id": "arabian_sea_mumbai_offshore",
        "title": "Arabian Sea, Mumbai offshore approaches",
        "lat": 19.050,
        "lon": 71.600,
        "start": "2023-11-01",
        "end": "2024-03-31",
        "size_km": 22.0,
        "notes": ("Busy tanker approach lanes off the Indian west coast. Outside "
                  "MarineCadastre coverage and Indian DGLL AIS is not a public "
                  "feed, so simulated traffic over this real geobox and time "
                  "window is the path the problem statement allows."),
    },
    "bay_of_bengal": {
        "id": "bay_of_bengal_paradip",
        "title": "Bay of Bengal, Paradip approaches",
        "lat": 20.100,
        "lon": 86.900,
        "start": "2023-11-01",
        "end": "2024-03-31",
        "size_km": 22.0,
        "notes": ("East coast anchorage and approach traffic. Outside public AIS "
                  "coverage, so simulated traffic applies here too."),
    },
    "gulf_open": {
        "id": "gom_open_water_control",
        "title": "Gulf of Mexico open water control",
        "lat": 27.800,
        "lon": -90.400,
        "start": "2023-12-01",
        "end": "2024-01-31",
        "size_km": 22.0,
        "notes": ("Control scene with no known slick. A pipeline that reports "
                  "suspects here is broken, so this is the true negative test."),
    },
}


def _get_json(url: str, body: Optional[dict] = None, timeout: int = 90) -> dict:
    if body is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sas_token(collection: str = COLLECTION) -> str:
    return _get_json(SAS_TOKEN % collection)["token"]


def search(lat: float, lon: float, start: str, end: str, pad_deg: float = 0.08,
           limit: int = 12) -> List[dict]:
    body = {
        "collections": [COLLECTION],
        "bbox": [lon - pad_deg, lat - pad_deg, lon + pad_deg, lat + pad_deg],
        "datetime": "%sT00:00:00Z/%sT23:59:59Z" % (start, end),
        "limit": limit,
    }
    return _get_json(STAC_SEARCH, body).get("features", [])


def _bbox_km(lat: float, lon: float, size_km: float) -> Tuple[float, float, float, float]:
    from app.geo.crs import meters_per_degree

    m_lon, m_lat = meters_per_degree(lat)
    half = size_km * 1000.0 / 2.0
    return (lon - half / m_lon, lat - half / m_lat,
            lon + half / m_lon, lat + half / m_lat)


def cut_chip(item: dict, lat: float, lon: float, size_km: float, token: str,
             max_side: int = 2048) -> Dict[str, Any]:
    """Windowed read of VV and VH around a point, returned as a dB stack."""
    import os

    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")

    wgs = _bbox_km(lat, lon, size_km)
    bands: List[np.ndarray] = []
    transform = None
    crs = None

    for pol in ("vv", "vh"):
        asset = item["assets"].get(pol)
        if asset is None:
            continue
        href = "/vsicurl/%s?%s" % (asset["href"], token)
        with rasterio.open(href) as ds:
            bounds = transform_bounds("EPSG:4326", ds.crs, *wgs)
            win = from_bounds(*bounds, transform=ds.transform)
            win = win.round_offsets().round_lengths()
            if win.width < 64 or win.height < 64:
                raise RuntimeError("requested window falls outside this scene")
            out_h = min(int(win.height), max_side)
            out_w = min(int(win.width), max_side)
            arr = ds.read(1, window=win, out_shape=(out_h, out_w),
                          boundless=True, fill_value=float("nan"))
            if transform is None:
                t = ds.window_transform(win)
                sx = float(win.width) / out_w
                sy = float(win.height) / out_h
                transform = (t.a * sx, t.b, t.c, t.d, t.e * sy, t.f)
                crs = ds.crs.to_string()
        arr = np.asarray(arr, dtype=np.float32)
        arr[arr <= 0] = np.nan          # RTC nodata is a large negative sentinel
        arr[arr < 1e-6] = np.nan
        bands.append(10.0 * np.log10(arr))   # linear gamma0 to dB

    if not bands:
        raise RuntimeError("item has no VV or VH asset")
    if len(bands) == 1:
        bands.append(bands[0].copy())

    stack = np.stack(bands, axis=0).astype(np.float32)
    finite = np.isfinite(stack)
    coverage = float(finite.mean())
    return {
        "array": stack,
        "transform": transform,
        "crs": crs,
        "coverage": coverage,
        "t_sat": item["properties"]["datetime"],
        "item_id": item["id"],
        "shape": list(stack.shape),
    }


def screen(chip: Dict[str, Any]) -> Dict[str, Any]:
    """Score a candidate acquisition on how observable a slick is in it.

    This is scene selection, not detection tuning. A SAR image taken at 9 m/s
    wind has almost no slick contrast because the wind roughens the surface
    everywhere; the same site at 3 m/s shows the filament clearly. Operational
    practice is to work the acquisitions where the phenomenon is visible, so the
    fetcher runs the published dark patch baseline over each candidate and keeps
    the one with the most coherent dark area.
    """
    from app.geo import geometry
    from app.ml import fallback

    vv = chip["array"][0]
    out = fallback.detect(vv, min_pixels=48)
    mask = out["mask"]
    dark_px = int((mask > 0).sum())
    oil_px = int((mask == 2).sum())

    finite = vv[np.isfinite(vv)]
    if finite.size == 0:
        return {"score": -1.0, "oil_px": 0, "dark_px": 0, "sea_db": float("nan"),
                "p1_db": float("nan"), "contrast_db": 0.0}
    sea = float(np.median(finite))
    p1 = float(np.percentile(finite, 1.0))
    contrast = sea - p1

    _labels, n = geometry.label_components(mask == 2)
    largest = 0
    if n:
        largest = int(max((int((_labels == k).sum()) for k in range(1, n + 1)), default=0))

    # Weight the biggest single blob, since a slick is one connected filament and
    # scattered speckle is not.
    score = largest / 1000.0 + oil_px / 20000.0 + max(0.0, contrast - 2.0)
    return {"score": float(score), "oil_px": oil_px, "dark_px": dark_px,
            "largest_blob_px": largest, "sea_db": round(sea, 2),
            "p1_db": round(p1, 2), "contrast_db": round(contrast, 2)}


def fetch(scene_id: str, title: str, lat: float, lon: float, start: str, end: str,
          size_km: float = 22.0, notes: str = "", min_coverage: float = 0.85,
          max_items: int = 6, auto_select: bool = True,
          screen_side: int = 640) -> Optional[scenes_mod.Scene]:
    """Find a usable acquisition and write the chip to data/sar."""
    from app.geo import raster as raster_mod

    print("[%s] searching %s %s .. %s" % (scene_id, COLLECTION, start, end))
    items = search(lat, lon, start, end, limit=max(max_items, 12))
    if not items:
        print("  no acquisitions found for that window")
        return None
    token = sas_token()

    ordered = items[:max_items]
    if auto_select and len(ordered) > 1:
        print("  screening %d acquisitions at %d px for slick contrast" % (len(ordered), screen_side))
        scored: List[Tuple[float, dict, Dict[str, Any]]] = []
        for item in ordered:
            try:
                small = cut_chip(item, lat, lon, size_km, token, max_side=screen_side)
            except Exception as exc:
                print("   %s  skipped: %s" % (item["properties"]["datetime"][:16], exc))
                continue
            if small["coverage"] < min_coverage:
                print("   %s  skipped: %.0f%% valid" % (item["properties"]["datetime"][:16],
                                                        100 * small["coverage"]))
                continue
            s = screen(small)
            print("   %s  sea %6.2f dB  p1 %6.2f dB  contrast %5.2f dB  dark blob %5d px  score %.2f"
                  % (item["properties"]["datetime"][:16], s["sea_db"], s["p1_db"],
                     s["contrast_db"], s["largest_blob_px"], s["score"]))
            scored.append((s["score"], item, s))
        if scored:
            scored.sort(key=lambda t: -t[0])
            ordered = [t[1] for t in scored]
            print("  selected %s" % ordered[0]["properties"]["datetime"][:16])

    for item in ordered[:max_items]:
        try:
            chip = cut_chip(item, lat, lon, size_km, token)
        except Exception as exc:
            print("  %s skipped: %s" % (item["id"][:32], exc))
            continue
        if chip["coverage"] < min_coverage:
            print("  %s skipped: only %.0f%% valid pixels"
                  % (item["id"][:32], 100 * chip["coverage"]))
            continue

        out = Path(config.SAR_DIR) / ("%s.tif" % scene_id)
        raster_mod.write_geotiff(out, chip["array"], chip["transform"], chip["crs"])
        size_mb = out.stat().st_size / 1e6
        print("  wrote %s  %s  %.1f MB  %.0f%% valid"
              % (out.name, chip["shape"], size_mb, 100 * chip["coverage"]))

        scene = scenes_mod.describe_geotiff(
            out, scene_id=scene_id, title=title, t_sat=chip["t_sat"],
            source="Sentinel-1 IW GRD RTC via Microsoft Planetary Computer (%s)" % chip["item_id"],
            license_=LICENSE,
            notes=notes,
        )
        print("  bounds %s  t_sat %s  ais_mode %s"
              % ([round(v, 3) for v in scene.bounds], scene.t_sat, scene.ais_mode))
        return scene

    print("  no acquisition met the coverage threshold")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", help="one of %s, or 'all'" % ", ".join(PRESETS))
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--id")
    ap.add_argument("--title")
    ap.add_argument("--start", default="2023-06-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--size-km", type=float, default=22.0)
    ap.add_argument("--notes", default="")
    ap.add_argument("--candidates", type=int, default=8,
                    help="how many acquisitions to screen before choosing one")
    ap.add_argument("--no-auto-select", action="store_true",
                    help="take the newest acquisition instead of screening")
    args = ap.parse_args()

    jobs: List[Dict[str, Any]] = []
    if args.preset == "all":
        jobs = list(PRESETS.values())
    elif args.preset:
        if args.preset not in PRESETS:
            ap.error("unknown preset %r" % args.preset)
        jobs = [PRESETS[args.preset]]
    elif args.lat is not None and args.lon is not None:
        jobs = [{
            "id": args.id or "scene_%.3f_%.3f" % (args.lat, args.lon),
            "title": args.title or "Sentinel-1 chip",
            "lat": args.lat, "lon": args.lon,
            "start": args.start, "end": args.end,
            "size_km": args.size_km, "notes": args.notes,
        }]
    else:
        ap.error("pass --preset NAME, --preset all, or --lat and --lon")

    made = []
    for job in jobs:
        scene = fetch(job["id"], job["title"], job["lat"], job["lon"],
                      job["start"], job["end"], size_km=job.get("size_km", 22.0),
                      notes=job.get("notes", ""), max_items=args.candidates,
                      auto_select=not args.no_auto_select)
        if scene is not None:
            # Merge immediately and re-read the index each time. A fetch runs
            # for minutes, and saving a stale in-memory copy at the end would
            # silently drop whatever another script indexed meanwhile.
            scenes_mod.upsert([scene])
            made.append(scene.id)
    if made:
        print("\nindexed %s; the index now holds %d scene(s)"
              % (", ".join(made), len(scenes_mod.load_index())))
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
