"""Cut extracted Zenodo chips into 512 training tiles.

Run this after `download_zenodo_subset.py --file ... --extract`. It walks the
extracted tree, pairs each Sigma0 image with its mask, forces the SAR
georeference onto the mask, relabels by source folder into the project's three
classes, and writes compressed npz tiles plus an index.json that records the dB
normalisation statistics.

The hackathon subset from the spec is the default: about 400 oil, 250
look-alike, 150 empty. Upload the tile folder to Kaggle and train there.

Usage:
    python scripts/prepare_tiles.py --root data/zenodo --out data/tiles
    python scripts/prepare_tiles.py --root data/zenodo --out data/tiles \
        --oil 400 --lookalike 250 --empty 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config              # noqa: E402
from app.ml import dataset as ds    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with the extracted Zenodo tree")
    ap.add_argument("--out", default=str(Path(config.DATA_DIR) / "tiles"))
    ap.add_argument("--size", type=int, default=config.TILE)
    ap.add_argument("--overlap", type=int, default=config.TILE_OVERLAP)
    ap.add_argument("--oil", type=int, default=400, help="max oil tiles")
    ap.add_argument("--lookalike", type=int, default=250)
    ap.add_argument("--empty", type=int, default=150)
    ap.add_argument("--min-oil-pixels", type=int, default=ds.MIN_OIL_PIXELS_PER_TILE)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit("no such folder: %s" % root)

    pairs = ds.index_pairs(root)
    if not pairs:
        raise SystemExit("found no image and mask pairs under %s. Check the extraction." % root)

    by_class = {0: 0, 1: 0, 2: 0}
    for _img, _mask, k in pairs:
        by_class[k] += 1
    print("found %d images: %d oil, %d look-alike, %d oil free"
          % (len(pairs), by_class[2], by_class[1], by_class[0]))
    missing = sum(1 for _i, m, _k in pairs if m is None)
    if missing:
        print("  %d images have no matching mask and will be treated as empty" % missing)

    index = ds.build_tile_index(
        pairs, Path(args.out), size=args.size, overlap=args.overlap,
        min_oil_pixels=args.min_oil_pixels,
        limit_per_class={2: args.oil, 1: args.lookalike, 0: args.empty},
    )

    print("\nwrote %d tiles to %s" % (index["tiles"], args.out))
    print("  per class: %s" % index["counts"])
    print("  normalisation: %s" % index["normalisation"])
    print("  skipped images: %d" % index["skipped_images"])
    print("\nNext: upload %s to Kaggle and run" % args.out)
    print("  python -m app.ml.train --tiles /kaggle/input/<dataset> --epochs 40 --batch-size 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
