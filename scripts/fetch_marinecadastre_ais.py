"""Ingest REAL AIS from MarineCadastre for a scene footprint.

MarineCadastre publishes one nationwide CSV per day, zipped, tens of millions of
rows. The problem statement is explicit that you do not download nationwide US
days into the repo, so this script streams the archive, clips to the scene box
and time window while it reads, ingests the survivors into SQLite, and then
deletes the raw download unless you ask it to keep it.

Coverage: US waters only (Gulf of Mexico, East and West coasts, Alaska, Hawaii,
Puerto Rico). `app/scenes.py` already decides per scene whether the footprint is
inside that coverage, and this script refuses scenes that are not, because
inventing Indian or European AIS would be exactly the kind of fabrication the
spec forbids.

Source: https://coast.noaa.gov/htdata/CMSP/AISDataHandler/<year>/AIS_<Y>_<M>_<D>.zip
Licence: US Government work, public domain. Cite MarineCadastre.gov.

Usage:
    python scripts/fetch_marinecadastre_ais.py --scene gom_mc20_chronic_slick
    python scripts/fetch_marinecadastre_ais.py --scene X --pad-km 40 --days 2
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, scenes as scenes_mod   # noqa: E402
from app.ais import ingest as ais_ingest       # noqa: E402

BASE = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"
LICENSE = "MarineCadastre.gov / NOAA + BOEM, US Government work, public domain"
SOURCE_TAG = "marinecadastre"


def url_for(day: datetime) -> str:
    return "%s/%d/AIS_%d_%02d_%02d.zip" % (BASE, day.year, day.year, day.month, day.day)


def pad_bbox(bounds: List[float], pad_km: float) -> Tuple[float, float, float, float]:
    from app.geo.crs import meters_per_degree

    w, s, e, n = [float(v) for v in bounds]
    m_lon, m_lat = meters_per_degree((s + n) / 2.0)
    return (w - pad_km * 1000 / m_lon, s - pad_km * 1000 / m_lat,
            e + pad_km * 1000 / m_lon, n + pad_km * 1000 / m_lat)


def download(day: datetime, dest_dir: Path) -> Optional[Path]:
    url = url_for(day)
    dest = dest_dir / Path(url).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print("   already downloaded: %s (%.0f MB)" % (dest.name, dest.stat().st_size / 1e6))
        return dest
    print("   GET %s" % url)
    try:
        with urllib.request.urlopen(url, timeout=180) as resp, dest.open("wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            last = -1
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if total:
                    pct = int(100 * done / total)
                    if pct != last and pct % 10 == 0:
                        print("      %3d%%  %.0f / %.0f MB" % (pct, done / 1e6, total / 1e6))
                        last = pct
    except Exception as exc:
        print("   download failed: %s" % exc)
        dest.unlink(missing_ok=True)
        return None
    print("   saved %.0f MB" % (dest.stat().st_size / 1e6))
    return dest


def clip_and_ingest(zip_path: Path, bbox: Tuple[float, float, float, float],
                    t_start: int, t_end: int, conn) -> int:
    """Stream the CSV out of the zip and keep only rows inside the box."""
    w, s, e, n = bbox
    kept: List[tuple] = []
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = [x for x in zf.namelist() if x.lower().endswith(".csv")]
        if not names:
            print("   no CSV inside the archive")
            return 0
        with zf.open(names[0]) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            for rec in reader:
                total += 1
                try:
                    lat = float(rec.get("LAT") or "nan")
                    lon = float(rec.get("LON") or "nan")
                except ValueError:
                    continue
                if not (s <= lat <= n and w <= lon <= e):
                    continue
                row = ais_ingest.row_from_csv(rec, SOURCE_TAG)
                if row is None:
                    continue
                if not (t_start <= row[1] <= t_end):
                    continue
                kept.append(row)
                if len(kept) >= 20000:
                    ais_ingest.ingest_rows(conn, kept, SOURCE_TAG)
                    kept = []
                if total % 2_000_000 == 0:
                    print("      scanned %.1fM rows" % (total / 1e6))
    if kept:
        ais_ingest.ingest_rows(conn, kept, SOURCE_TAG)
    print("   scanned %d rows in the national file" % total)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", action="append", default=[], required=False)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pad-km", type=float, default=45.0,
                    help="margin around the scene box, so approaches are captured")
    ap.add_argument("--days", type=int, default=3,
                    help="days centred on the acquisition, to cover the hindcast window")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-ingest even if rows exist")
    args = ap.parse_args()

    wanted = set(args.scene)
    targets = [s for s in scenes_mod.all_scenes(include_selftest=True)
               if (s.id in wanted) or (args.all and not wanted)]
    if not targets:
        ap.error("no matching scenes; pass --scene ID or --all")

    conn = ais_ingest.connect()
    try:
        if args.force:
            removed = ais_ingest.clear_source(conn, SOURCE_TAG)
            print("cleared %d previously ingested real AIS rows" % removed)

        for scene in targets:
            if scene.ais_mode != "real":
                print("[%s] footprint is outside MarineCadastre coverage. Real AIS does "
                      "not exist for it. Use scripts/build_synthetic_ais.py." % scene.id)
                continue

            t_sat = datetime.fromisoformat(scene.t_sat.replace("Z", "+00:00"))
            bbox = pad_bbox(scene.bounds, args.pad_km)
            half = args.days // 2
            days = [t_sat.date() - timedelta(days=half) + timedelta(days=i)
                    for i in range(args.days)]
            t_start = int((t_sat - timedelta(days=half + 1)).timestamp())
            t_end = int((t_sat + timedelta(days=half + 1)).timestamp())

            print("[%s] real AIS for %s, box %s, %d day(s)"
                  % (scene.id, t_sat.date(), [round(v, 3) for v in bbox], len(days)))
            before = ais_ingest.stats(conn).rows
            for d in days:
                day = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                zp = download(day, Path(config.CACHE_DIR) / "marinecadastre")
                if zp is None:
                    continue
                clip_and_ingest(zp, bbox, t_start, t_end, conn)
                if not args.keep_zip:
                    zp.unlink(missing_ok=True)
                    print("   removed the national archive, kept only the clipped rows")
            after = ais_ingest.stats(conn)
            print("   ingested %d rows for this scene" % (after.rows - before))
            print("   store: %d rows, %d vessels, %s .. %s"
                  % (after.rows, after.vessels, after.t_start, after.t_end))
    finally:
        conn.close()

    print("\nCite in the README: %s" % LICENSE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
