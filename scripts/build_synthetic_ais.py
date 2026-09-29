"""Generate simulated maritime traffic for scenes with no public AIS coverage.

Read `app/ais/synthetic.py` before changing anything here. The short version:
the problem statement permits synthetic AIS only when real tracks for the scene
footprint and time do not exist, and it must be a traffic simulator over the
real geobox that is then scored blindly.

So this script does not invent an origin. It runs the actual detector on the
actual scene, runs the actual backward drift on the actual cached metocean, and
only then simulates a traffic picture in which one vessel physically transits
that area at that time. The scorer is never told which vessel that was.

Usage:
    python scripts/build_synthetic_ais.py --scene gom_mc20_chronic_slick
    python scripts/build_synthetic_ais.py --all
    python scripts/build_synthetic_ais.py --scene X --no-gap --vessels 40
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, pipeline, scenes as scenes_mod   # noqa: E402
from app.ais import synthetic   # noqa: E402


def origin_for_scene(scene: scenes_mod.Scene, hindcast_hours: int) -> Dict[str, Any]:
    """Detect, then hindcast, and return the computed origin. No shortcuts."""
    print("[%s] detecting" % scene.id)
    det = pipeline.detect_scene(scene, render_overlays=False)
    oil = det["polygon_objects"]
    print("  detector: %s, oil polygons: %d, look-alikes: %d"
          % (det["metrics"]["detector"], len(oil), len(det["lookalike_objects"])))

    t_sat = pipeline._utc(scene.t_sat)
    if oil:
        primary = oil[0]
        ring = primary.ring_lonlat
        centroid = (primary.centroid_lon, primary.centroid_lat)
        print("  primary slick %.3f km2 at %.4f, %.4f"
              % (primary.area_km2, primary.centroid_lat, primary.centroid_lon))
    else:
        # A clean scene still needs traffic, otherwise the true negative test has
        # nothing to be negative about. Centre the picture on the chip.
        ring = None
        centroid = (scene.centroid[0], scene.centroid[1])
        print("  no oil detected; traffic will be generated around the chip centre")

    print("  hindcasting %d h" % hindcast_hours)
    drift = pipeline.run_drift(ring, centroid, t_sat, hindcast_hours,
                              config.FORECAST_H, config.ENSEMBLE_N, scene.id)
    origin = drift["origin"]
    print("  origin %.4f, %.4f at %s (spread %.1f km, metocean: %s)"
          % (origin["lat"], origin["lon"], origin["t"], origin["spread_km"],
             drift["metocean"]["source"][:48]))
    return {"origin": origin, "t_sat": t_sat, "drift": drift}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--vessels", type=int, default=48,
                    help="fleet size over the whole window; a busy approach lane "
                         "sees far more than a handful of ships in three days")
    ap.add_argument("--hours", type=int, default=72)
    ap.add_argument("--step-seconds", type=int, default=300)
    ap.add_argument("--hindcast-hours", type=int, default=config.HINDCAST_H)
    ap.add_argument("--no-gap", action="store_true",
                    help="do not give the scenario vessel a reporting gap")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--keep", action="store_true",
                    help="append instead of replacing previous simulated rows")
    ap.add_argument("--force", action="store_true",
                    help="generate even for scenes flagged as having real AIS coverage")
    args = ap.parse_args()

    # --all means the demo scenes. Self test rasters are only reachable by
    # naming them, so a synthetic scene can never sneak into the demo store.
    wanted = set(args.scene)
    targets: List[scenes_mod.Scene] = []
    for s in scenes_mod.all_scenes(include_selftest=bool(wanted)):
        if wanted and s.id not in wanted:
            continue
        if not wanted and not args.all:
            continue
        targets.append(s)
    if not targets:
        ap.error("no matching scenes. Pass --all or --scene ID, and make sure "
                 "scenes are indexed (scripts/fetch_sentinel1_scene.py).")

    replaced = not args.keep
    total = 0
    for scene in targets:
        if scene.ais_mode == "real" and not args.force:
            print("[%s] footprint is inside MarineCadastre coverage, so real AIS is "
                  "the correct source. Use scripts/fetch_marinecadastre_ais.py, or "
                  "pass --force to simulate anyway." % scene.id)
            continue

        info = origin_for_scene(scene, args.hindcast_hours)
        origin = info["origin"]
        print("  simulating %d vessels over %d h at %d s sampling"
              % (args.vessels, args.hours, args.step_seconds))
        # Derive the seed from the scene id so two scenes never end up sharing
        # vessel identities. Still fully deterministic per scene.
        scene_seed = (args.seed + zlib.crc32(scene.id.encode("utf-8"))) % (2 ** 31)
        built = synthetic.build(
            bbox=scene.bounds,
            t_center=info["t_sat"],
            origin_lon=origin["lon"], origin_lat=origin["lat"],
            t_origin=pipeline._utc(origin["t"]),
            hours=args.hours,
            step_seconds=args.step_seconds,
            n_vessels=args.vessels,
            seed=scene_seed,
            with_gap=not args.no_gap,
        )
        out = synthetic.write(
            built["rows"], built["ground_truth"],
            csv_path=Path(config.AIS_DIR) / ("simulated_%s.csv" % scene.id),
            truth_path=Path(config.AIS_DIR) / ("ground_truth_%s.json" % scene.id),
            replace=replaced,
        )
        replaced = False  # only clear the simulated source once per run
        total += out["rows_ingested"]
        print("  wrote %d rows -> %s" % (out["rows_ingested"], Path(out["csv"]).name))
        print("  store now: %d rows, %d vessels, %s .. %s"
              % (out["store"]["rows"], out["store"]["vessels"],
                 out["store"]["t_start"], out["store"]["t_end"]))
        print("  ground truth kept aside at %s (never read by scoring)"
              % Path(out["ground_truth"]).name)

    print("\ntotal rows ingested this run: %d" % total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
