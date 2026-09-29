"""One command that turns an empty checkout into a runnable offline demo.

Run this ONCE while you have internet. Afterwards the product runs in airplane
mode, which is what the problem statement requires at judging time.

What it does, in order:

  1. Vendors Leaflet into app/static/vendor so the browser needs no CDN.
  2. Pulls real Sentinel-1 chips for the preset sites, screening several
     acquisitions each and keeping the one where a slick is actually visible.
  3. Caches real ERA5 wind and, where a model covers the basin, real
     Copernicus Marine currents for every scene footprint. Set
     COPERNICUSMARINE_SERVICE_USERNAME and ..._PASSWORD first; without
     them it falls back to Open-Meteo and says so.
  4. Caches map tiles covering the scene footprints, so the console has a real
     satellite and nautical backdrop while still running offline.
  5. For scenes outside MarineCadastre coverage, simulates traffic over that
     scene's real geobox and time window. For scenes inside coverage it tells
     you to fetch real AIS instead, and does not silently substitute.
  6. Prints a readiness table you can check against the problem statement.

Usage:
    python scripts/bootstrap_demo.py
    python scripts/bootstrap_demo.py --scenes mc20 arabian_sea
    python scripts/bootstrap_demo.py --skip-vendor --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = sys.executable


def run(args: List[str], label: str) -> bool:
    print("\n" + "=" * 74)
    print("  %s" % label)
    print("=" * 74)
    proc = subprocess.run([PY, "-u"] + args, cwd=str(ROOT))
    if proc.returncode != 0:
        print("  step failed with exit code %d" % proc.returncode)
    return proc.returncode == 0


def readiness() -> int:
    from app import config, scenes as scenes_mod
    from app.ais import ingest as ais_ingest
    from app.drift import fields as fields_mod
    from app.ml import model as model_mod

    print("\n" + "=" * 74)
    print("  READINESS")
    print("=" * 74)

    scenes = scenes_mod.all_scenes()
    cached = {c["scene_id"]: c for c in fields_mod.list_cached()}
    store = ais_ingest.stats()
    problems = 0

    print("%-34s %-9s %-9s %s" % ("scene", "metocean", "ais mode", "acquired"))
    for s in scenes:
        has_mo = "yes" if s.id in cached else "NO"
        if s.id not in cached:
            problems += 1
        print("%-34s %-9s %-9s %s" % (s.id[:34], has_mo, s.ais_mode, s.t_sat[:19]))

    print("\nAIS store       : %d rows, %d vessels, %s to %s"
          % (store.rows, store.vessels, store.t_start, store.t_end))
    if store.rows == 0:
        problems += 1
    print("Metocean cubes  : %d" % len(cached))
    print("Scenes indexed  : %d" % len(scenes))

    st = model_mod.status()
    print("Detector        : %s"
          % ("U-Net checkpoint present" if st["checkpoint_present"]
             else "dB baseline (no checkpoint; this is allowed and honest)"))
    print("torch / smp     : %s / %s" % (st["torch"], st["smp"]))
    print("CUDA            : %s" % st["cuda"])

    leaflet = Path(config.STATIC_DIR) / "vendor" / "leaflet" / "leaflet.js"
    print("Leaflet vendored: %s" % ("yes" if leaflet.exists() else "NO"))
    if not leaflet.exists():
        problems += 1

    basemap = Path(config.DATA_DIR) / "basemap" / "manifest.json"
    if basemap.exists():
        import json as _json
        m = _json.loads(basemap.read_text(encoding="utf-8"))
        tiles = sum(1 for _ in (Path(config.DATA_DIR) / "basemap").rglob("*.jpg"))
        tiles += sum(1 for _ in (Path(config.DATA_DIR) / "basemap").rglob("*.png"))
        print("Basemap cached  : %s, zoom %s-%s, %d tiles"
              % (", ".join(sorted(m.get("layers", {}))), m.get("min_zoom"),
                 m.get("max_zoom"), tiles))
    else:
        print("Basemap cached  : NO (map will show a graticule only)")
        problems += 1

    print("\n" + ("READY. Start with: python -m uvicorn app.main:app --port 8000"
                  if problems == 0 else
                  "%d item(s) still missing. See the lines marked NO above." % problems))
    return 0 if problems == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=["mc20", "arabian_sea"],
                    help="presets from scripts/fetch_sentinel1_scene.py")
    ap.add_argument("--skip-vendor", action="store_true")
    ap.add_argument("--skip-scenes", action="store_true")
    ap.add_argument("--skip-metocean", action="store_true")
    ap.add_argument("--skip-ais", action="store_true")
    ap.add_argument("--skip-basemap", action="store_true")
    ap.add_argument("--basemap-zoom", type=int, default=13,
                    help="highest zoom to cache; 13 is about 19 m per pixel")
    ap.add_argument("--currents", choices=("auto", "cmems", "open-meteo"),
                    default="auto", help="currents source; auto prefers CMEMS")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--vessels", type=int, default=48)
    args = ap.parse_args()

    if not args.skip_vendor:
        run(["scripts/fetch_vendor.py"], "1/6  vendor Leaflet locally")

    if not args.skip_scenes:
        for preset in args.scenes:
            run(["scripts/fetch_sentinel1_scene.py", "--preset", preset, "--candidates", "8"],
                "2/6  fetch real Sentinel-1 chip: %s" % preset)

    if not args.skip_metocean:
        cmd = ["scripts/build_metocean_cache.py", "--all", "--half-deg", "1.2",
               "--currents", args.currents]
        if args.force:
            cmd.append("--force")
        run(cmd, "3/6  cache real currents and 10 m wind")

    if not args.skip_basemap:
        run(["scripts/fetch_basemap.py", "--all", "--max-zoom", str(args.basemap_zoom)],
            "4/6  cache map tiles for the scene areas")

    if not args.skip_ais:
        run(["scripts/build_synthetic_ais.py", "--all", "--vessels", str(args.vessels)],
            "5/6  simulate traffic for scenes with no public AIS")
        print("\n  Scenes inside MarineCadastre coverage were skipped on purpose.")
        print("  For those, fetch the real thing:")
        print("    python scripts/fetch_marinecadastre_ais.py --scene <scene_id> --days 3")

    print("\n6/6  readiness check")
    return readiness()


if __name__ == "__main__":
    raise SystemExit(main())
