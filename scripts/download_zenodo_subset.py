"""Download a working subset of the Zenodo Sentinel-1 oil spill dataset.

Records:
    Part I    10.5281/zenodo.8346860   oil spill train/val images + masks
    Part II   10.5281/zenodo.8253899   look-alike and oil free images + masks
    Part III  10.5281/zenodo.13761290  test images and ground truth

Licence: CC BY 4.0.

Reality check before you run this. The image archives are enormous:

    01_Train_Val_Oil_Spill_images.7z    40.7 GB
    01_Train_Val_Lookalike_images.7z    23.0 GB
    01_Train_Val_No_Oil_Images.7z       22.9 GB
    02_Test_images_and_ground_truth.7z   9.9 GB

The mask archives are tiny by comparison (0.4 to 6.2 MB) and are always worth
pulling. So the default behaviour is: fetch every mask archive, list the image
archives with their sizes, and download an image archive only when you ask for
it by name. `--extract-limit N` then pulls just the first N members out of the
7z rather than unpacking tens of gigabytes, which needs py7zr.

If you only want a handful of chips on screen for the demo, do not use this
script at all. Use `scripts/fetch_sentinel1_scene.py`, which reads a window out
of a cloud optimised Sentinel-1 GeoTIFF and costs a few megabytes.

Usage:
    python scripts/download_zenodo_subset.py --list
    python scripts/download_zenodo_subset.py --masks
    python scripts/download_zenodo_subset.py --file 01_Train_Val_Oil_Spill_mask.7z --extract
    python scripts/download_zenodo_subset.py --file 02_Test_images_and_ground_truth.7z \
        --extract --extract-limit 40
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config   # noqa: E402

RECORDS = {
    "part1": ("8346860", "Part I, oil spill train and validation"),
    "part2": ("8253899", "Part II, look-alike and oil free"),
    "part3": ("13761290", "Part III, test images and ground truth"),
}
API = "https://zenodo.org/api/records/%s"
LICENSE = "Zenodo Sentinel-1 SAR oil spill dataset, CC BY 4.0"


def record_files(record_id: str) -> List[Dict[str, Any]]:
    with urllib.request.urlopen(API % record_id, timeout=90) as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    out = []
    for f in doc.get("files", []):
        out.append({
            "key": f.get("key"),
            "size": int(f.get("size") or 0),
            "url": (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download"),
            "checksum": f.get("checksum"),
        })
    return out


def list_all() -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, (rid, label) in RECORDS.items():
        print("\n%s  (%s)  record %s" % (key, label, rid))
        files = record_files(rid)
        out[key] = files
        for f in files:
            tag = "mask" if "mask" in (f["key"] or "").lower() else "IMAGES"
            print("   %9.1f MB  [%-6s] %s" % (f["size"] / 1e6, tag, f["key"]))
    return out


def download(url: str, dest: Path, expected: int = 0, chunk: int = 1 << 20) -> Path:
    """Stream to disk with HTTP range resume, because these files are huge."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if expected and have == expected:
        print("   already complete: %s" % dest.name)
        return dest

    req = urllib.request.Request(url)
    mode = "wb"
    if have > 0:
        req.add_header("Range", "bytes=%d-" % have)
        mode = "ab"
        print("   resuming at %.1f MB" % (have / 1e6))

    with urllib.request.urlopen(req, timeout=180) as resp, dest.open(mode) as fh:
        total = have + int(resp.headers.get("Content-Length") or 0)
        done = have
        last = -1
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if total:
                pct = int(100 * done / total)
                if pct != last and pct % 5 == 0:
                    print("   %3d%%  %.1f / %.1f MB" % (pct, done / 1e6, total / 1e6))
                    last = pct
    print("   saved %s (%.1f MB)" % (dest.name, dest.stat().st_size / 1e6))
    return dest


def extract(archive: Path, out_dir: Path, limit: Optional[int] = None) -> int:
    """Extract a 7z archive, optionally only the first `limit` members."""
    try:
        import py7zr
    except ImportError:
        print("   py7zr is not installed. Install it with:  pip install py7zr")
        print("   or extract %s by hand into %s" % (archive.name, out_dir))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(str(archive), mode="r") as z:
        names = [n for n in z.getnames() if not n.endswith("/")]
        if limit:
            names = names[:limit]
        print("   extracting %d of %d members" % (len(names), len(z.getnames())))
        z.extract(path=str(out_dir), targets=names)
    return len(names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show every file and its size")
    ap.add_argument("--masks", action="store_true", help="download every mask archive")
    ap.add_argument("--file", action="append", default=[], help="archive name, repeatable")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--extract-limit", type=int, default=None)
    ap.add_argument("--out", default=str(Path(config.DATA_DIR) / "zenodo"))
    args = ap.parse_args()

    out_root = Path(args.out)
    if args.list or not (args.masks or args.file):
        list_all()
        if not (args.masks or args.file):
            print("\nNothing downloaded. Pass --masks, or --file NAME.")
            print("For a quick demo set, prefer scripts/fetch_sentinel1_scene.py.")
        return 0

    wanted = set(args.file)
    for key, (rid, _label) in RECORDS.items():
        files = record_files(rid)
        for f in files:
            name = f["key"] or ""
            is_mask = "mask" in name.lower()
            if not ((args.masks and is_mask) or name in wanted):
                continue
            print("\n[%s] %s  (%.1f MB)" % (key, name, f["size"] / 1e6))
            if f["size"] > 2e9 and name in wanted:
                print("   WARNING: this archive is %.1f GB." % (f["size"] / 1e9))
            dest = out_root / key / name
            download(f["url"], dest, expected=f["size"])
            if args.extract:
                extract(dest, out_root / key / dest.stem, limit=args.extract_limit)

    print("\nLicence to cite in the README: %s" % LICENSE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
