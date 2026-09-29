"""Tiling and 3-class relabelling for the Zenodo Sentinel-1 oil spill dataset.

Label convention for the whole project:
    0 sea / background
    1 look_alike
    2 mineral_oil

Zenodo ships each part as its own folder of binary masks, so the class a mask
pixel belongs to is decided by which folder it came from:

    oil folder        mask value 1 -> class 2
    look-alike folder mask value 1 -> class 1
    oil-free folder   everything   -> class 0

Tiles are 512 with 64 overlap, per the spec. Tiles containing fewer than 50 oil
pixels are dropped unless they are drawn from the empty class, which keeps the
sampler from drowning the oil class in background.

torch is optional here too: the tiling half runs with numpy alone, so tiles can
be prepared on the demo laptop and the Dataset class only materialises when
torch is present.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from ..geo import raster as raster_mod

CLASS_SEA, CLASS_LOOKALIKE, CLASS_OIL = 0, 1, 2

# Order matters and is the whole point. "No oil" contains the substring "oil",
# so a naive scan labels clean sea as a spill and poisons the training set in
# the one direction nobody notices: the model learns that empty water is oil.
# Most specific patterns are therefore tested first.
FOLDER_RULES = [
    (("look_alike", "lookalike", "look-alike", "look_a_like"), CLASS_LOOKALIKE),
    (("no_oil", "nooil", "no-oil", "oil_free", "oilfree", "oil-free",
      "non_oil", "clean", "empty", "background"), CLASS_SEA),
    (("oil_spill", "oilspill", "oil-spill", "spill", "oil"), CLASS_OIL),
]

# Path components that mean "this is a label, not an image".
MASK_TOKENS = ("mask", "masks", "ground_truth", "groundtruth", "gt", "label", "labels")

# Label files are rarely named exactly like their image. Zenodo Part III pairs
# Images/Oil/00000.tif with Mask/Oil/00000_segmentation.tif, so matching on the
# bare stem finds nothing and every chip silently arrives unlabelled.
MASK_SUFFIXES = ("_segmentation", "-segmentation", "_seg", "_mask", "-mask",
                 "_gt", "-gt", "_label", "-label", "_labels")


def mask_key(name: str) -> str:
    """The stem an image and its label agree on, suffixes removed."""
    stem = _norm(Path(name).stem)
    for suffix in MASK_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem

MIN_OIL_PIXELS_PER_TILE = 50


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("\\", "/")


def class_for_folder(name: str) -> int:
    """Decide the class from the source folder path, most specific first."""
    key = _norm(name)
    for tokens, klass in FOLDER_RULES:
        for token in tokens:
            if token in key:
                return klass
    return CLASS_SEA


def is_mask_path(path) -> bool:
    """True when any component of the path marks it as a label tree.

    Zenodo Part III ships `Images/Oil` beside `Mask/Oil`, so checking only the
    leaf folder name is not enough: `Mask/Oil` would be read as an image folder
    full of oil chips, and the masks would be tiled as if they were SAR.
    """
    parts = [_norm(p) for p in Path(path).parts]
    return any(p in MASK_TOKENS for p in parts)


@dataclass
class TileSpec:
    image_path: str
    mask_path: Optional[str]
    klass: int
    row: int
    col: int
    size: int
    oil_pixels: int
    lookalike_pixels: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def relabel(mask: np.ndarray, klass: int) -> np.ndarray:
    """Turn a binary folder mask into the project's 3-class encoding."""
    m = np.asarray(mask)
    out = np.zeros(m.shape, dtype=np.uint8)
    if klass == CLASS_SEA:
        return out
    out[m > 0] = klass
    return out


def order_bands(sigma0_db: np.ndarray) -> np.ndarray:
    """Put a two band dB stack in ascending median order: co-pol last.

    This exists because of a bug that produced a model with an IoU of 0.889 and
    no ability to find water. The Zenodo archive stores its two bands as
    (VH, VV) -- cross-pol first, roughly 7 dB darker. `fetch_sentinel1_scene.py`
    reads Planetary Computer assets in the order ('vv', 'vh'), which is the
    opposite. So the network trained on (dark, bright) and was asked to predict
    on (bright, dark), and it answered "oil" for almost every pixel of every
    scene while scoring near perfectly on held-out tiles from its own archive.

    Nothing in the data announces which convention a source uses, so ordering by
    name would only move the assumption somewhere else. Ordering by brightness
    is a physical fact instead: over water, co-polarised backscatter exceeds
    cross-polarised by several dB in every sensor and every sea state. Sorting on
    the median therefore reproduces the archive's order exactly, and keeps doing
    so for any future source, whatever it calls its bands.
    """
    a = np.asarray(sigma0_db, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, :, :]
    if a.shape[0] < 2:
        return a
    medians = [float(np.nanmedian(b)) if np.isfinite(b).any() else 0.0 for b in a]
    return a[np.argsort(medians, kind="stable")]


def normalise_db(stack: np.ndarray, mean_db: float = -18.0, std_db: float = 6.0) -> np.ndarray:
    """Standardise a (bands, h, w) dB stack and expand to three channels."""
    a = np.asarray(stack, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, :, :]
    # Second line of defence. A non-finite or zero sigma reaching here would
    # silently flatten every input to zero or NaN.
    if not np.isfinite(mean_db):
        mean_db = DEFAULT_STATS["mean_db"]
    if not np.isfinite(std_db) or abs(std_db) < 1e-3:
        std_db = DEFAULT_STATS["std_db"]

    vv = np.nan_to_num(a[0], nan=mean_db)
    vh = np.nan_to_num(a[1], nan=mean_db) if a.shape[0] > 1 else vv
    out = np.stack([vv, vh, vv], axis=0)
    return (out - mean_db) / std_db


def tile_grid(height: int, width: int, size: int = None, overlap: int = None) -> List[Tuple[int, int]]:
    size = int(config.TILE if size is None else size)
    overlap = int(config.TILE_OVERLAP if overlap is None else overlap)
    stride = max(1, size - overlap)
    rows = list(range(0, max(1, height - size + 1), stride))
    cols = list(range(0, max(1, width - size + 1), stride))
    if height > size and rows[-1] != height - size:
        rows.append(height - size)
    if width > size and cols[-1] != width - size:
        cols.append(width - size)
    return [(r, c) for r in rows for c in cols]


def index_pairs(root: Path, image_glob: str = "*.tif") -> List[Tuple[Path, Optional[Path], int]]:
    """Walk an extracted Zenodo tree and pair each image with its mask.

    Layout tolerated: <root>/<class folder>/{images,masks}/name.tif, or a flat
    folder where the mask sits beside the image with a _mask suffix.
    """
    out: List[Tuple[Path, Optional[Path], int]] = []
    root = Path(root)
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        rel = folder.relative_to(root)
        if is_mask_path(rel):
            continue
        klass = class_for_folder(str(rel))
        images = sorted(folder.glob(image_glob))
        if not images:
            continue
        for img in images:
            if "_mask" in img.stem.lower():
                continue
            out.append((img, _find_mask(img, folder, root), klass))
    return out


def group_archive_members(names: Sequence[str]) -> Dict[str, List[str]]:
    """Group archive member paths by their parent directory."""
    out: Dict[str, List[str]] = {}
    for n in names:
        out.setdefault(str(Path(n).parent).replace("\\", "/"), []).append(n)
    return out


def image_folders(grouped: Dict[str, List[str]]) -> List[str]:
    return sorted(d for d in grouped if not is_mask_path(d))


def mask_folders(grouped: Dict[str, List[str]]) -> List[str]:
    return sorted(d for d in grouped if is_mask_path(d))


def select_batch_targets(grouped: Dict[str, List[str]], image_folder: str,
                         stems: Sequence[str]) -> List[str]:
    """Members to extract for one batch: the images AND their masks.

    This is the step that a streaming extractor gets wrong. Pulling only the
    image folder's members leaves the labels inside the archive, every chip is
    then read as unlabelled, and the oil class disappears without any error at
    all. The masks live in a parallel tree whose leaf names match, so both sides
    have to come out together.
    """
    wanted = {mask_key(s) for s in stems}
    targets = [m for m in grouped.get(image_folder, []) if mask_key(m) in wanted]

    leaf = _norm(Path(image_folder).name)
    for d in mask_folders(grouped):
        if _norm(Path(d).name) != leaf:
            continue
        targets.extend(m for m in grouped[d] if mask_key(m) in wanted)
    return sorted(set(targets))


def _find_mask(image: Path, folder: Path, root: Optional[Path] = None) -> Optional[Path]:
    """Locate the label for one image, across the layouts this data ships in.

    Zenodo Part III is the awkward one: `Images/Oil/x.tif` has its label at
    `Mask/Oil/x.tif`, a sibling tree rather than a subfolder. Missing that is
    silent, because an image with no mask is simply labelled all-sea, and the
    oil class then vanishes from the tile set without any error at all.
    """
    candidates = [
        folder / "masks" / image.name,
        folder / "mask" / image.name,
        folder.parent / "masks" / image.name,
        folder.parent / "mask" / image.name,
    ]
    # Same folder, suffixed name: x.tif -> x_segmentation.tif, x_mask.tif, ...
    for suffix in MASK_SUFFIXES:
        candidates.append(image.with_name(image.stem + suffix + image.suffix))

    mask_dirs = []
    if root is not None:
        rel = folder.relative_to(root).parts
        # Swap whichever component names the image tree for each mask synonym,
        # so Images/Oil -> Mask/Oil, image/oil -> ground_truth/oil, and so on.
        for i, part in enumerate(rel):
            if _norm(part) not in ("images", "image", "img", "sar", "sigma0"):
                continue
            for token in ("Mask", "Masks", "mask", "masks",
                          "Ground_truth", "ground_truth", "GT", "gt",
                          "Label", "Labels", "label", "labels"):
                swapped = list(rel)
                swapped[i] = token
                mask_dirs.append(root.joinpath(*swapped))

    for d in mask_dirs:
        candidates.append(d / image.name)
        for suffix in MASK_SUFFIXES:
            candidates.append(d / (image.stem + suffix + image.suffix))

    for c in candidates:
        if c.exists():
            return c

    # Last resort: scan the mask directory for anything whose key matches. This
    # covers a suffix nobody has seen yet, and reports rather than silently
    # dropping the chip.
    key = mask_key(image.name)
    for d in mask_dirs:
        if not d.is_dir():
            continue
        for cand in sorted(d.glob("*" + image.suffix)):
            if mask_key(cand.name) == key:
                return cand
    return None



def _source_tag(image_path) -> str:
    """A short, stable prefix identifying the folder a chip came from.

    Only the immediate parent is used, lowercased and reduced to alphanumerics,
    so `Images/No oil/00000.tif` becomes `nooil_`. Empty when there is no
    meaningful parent, which keeps flat test fixtures naming tiles as before.
    """
    parent = Path(image_path).parent.name
    tag = "".join(ch for ch in parent.lower() if ch.isalnum())
    return (tag + "_") if tag else ""

def build_tile_index(
    pairs: Sequence[Tuple[Path, Optional[Path], int]],
    out_dir: Path,
    size: int = None,
    overlap: int = None,
    min_oil_pixels: int = MIN_OIL_PIXELS_PER_TILE,
    limit_per_class: Optional[Dict[int, int]] = None,
    max_hard_negatives: Optional[int] = None,
    max_scene_negatives: Optional[int] = None,
    max_plain_sea: Optional[int] = None,
) -> Dict[str, Any]:
    """Cut every pair into tiles and write them as compressed npz plus an index.

    Each npz holds the dB stack and the 3-class label for one tile, so training
    on Kaggle needs no rasterio and no georeferencing at all.
    """
    size = int(config.TILE if size is None else size)
    overlap = int(config.TILE_OVERLAP if overlap is None else overlap)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {CLASS_SEA: 0, CLASS_LOOKALIKE: 0, CLASS_OIL: 0}
    specs: List[TileSpec] = []
    skipped = 0
    hard_negatives = 0
    hard_neg_tiles = 0
    scene_neg_tiles = 0
    plain_sea_tiles = 0

    for image_path, mask_path, klass in pairs:
        # A look-alike chip may end up counted as sea, so only skip early when
        # both its nominal and its fallback budget are already full.
        cap = (limit_per_class or {}).get(klass)
        sea_cap = (limit_per_class or {}).get(CLASS_SEA)
        nominal_full = cap is not None and counts[klass] >= cap
        sea_full = sea_cap is not None and counts[CLASS_SEA] >= sea_cap
        hn_full = (max_hard_negatives is not None
                   and hard_neg_tiles >= max_hard_negatives)
        if nominal_full and (klass == CLASS_SEA or sea_full or hn_full):
            continue
        try:
            sar = raster_mod.load_sar(image_path)
        except Exception:
            # A chip with no georeference is unusable for the product, but its
            # pixels are still fine supervision. `load_raw` goes through the same
            # reader, so whatever compression rasterio understands still opens.
            # The earlier fallback used the builtin TIFF reader, which handles
            # only uncompressed and Deflate, and would silently skip an LZW chip
            # and take the oil class down with it.
            try:
                sar = raster_mod.load_raw(image_path)
            except Exception as exc:
                skipped += 1
                if skipped <= 3:
                    print("  cannot read %s: %s" % (Path(image_path).name, exc))
                continue

        db = order_bands(raster_mod.to_db(sar.array))

        if mask_path is not None:
            try:
                label = relabel(raster_mod.load_mask_with_georef(mask_path, sar).array[0], klass)
            except Exception as exc:
                skipped += 1
                if skipped <= 3:
                    print("  cannot read mask %s: %s" % (Path(mask_path).name, exc))
                continue
        else:
            label = np.zeros(db.shape[-2:], dtype=np.uint8)

        # The Zenodo ground truth segments oil and nothing else. A look-alike
        # chip's mask is entirely zero, verified against Part II: every sampled
        # 2048x2048 look-alike mask contains only the value 0. So a look-alike
        # chip carries no positive supervision, and treating it as a class-1
        # example would mean inventing annotation that does not exist.
        #
        # What it IS, and what it is worth far more as, is a hard negative:
        # dark water that is not oil. Those tiles are kept and labelled sea,
        # which is precisely the confusion the model has to learn to avoid.
        effective_klass = klass
        is_hard_negative = False
        if klass != CLASS_SEA and int(label.max()) == 0:
            effective_klass = CLASS_SEA
            is_hard_negative = True
            hard_negatives += 1

        for (r, c) in tile_grid(db.shape[-2], db.shape[-1], size, overlap):
            tile_img = db[:, r:r + size, c:c + size]
            tile_lab = label[r:r + size, c:c + size]
            if tile_img.shape[-2:] != (size, size):
                continue
            oil_px = int((tile_lab == CLASS_OIL).sum())
            la_px = int((tile_lab == CLASS_LOOKALIKE).sum())
            tile_klass = effective_klass
            is_scene_negative = False
            if (effective_klass != CLASS_SEA
                    and oil_px < min_oil_pixels and la_px < min_oil_pixels):
                # The spec drops these from the oil class, and it is right
                # to: a tile with four oil pixels is not an oil example.
                # But throwing them away entirely is what skewed the class
                # prior so far from reality. A tile of open water taken
                # from an oil scene is the best negative available -- same
                # sensor, same pass, same sea state, same incidence angle,
                # differing from the positives only in the thing being
                # learned. Keep a bounded number of them, labelled sea.
                if (max_scene_negatives is None
                        or scene_neg_tiles >= max_scene_negatives):
                    continue
                tile_klass = CLASS_SEA
                is_scene_negative = True
            # Class 0 has three independent sources -- plain sea chips,
            # look-alike hard negatives, and open water cut from oil
            # scenes -- and they are walked in a fixed folder order:
            # Lookalike, No oil, Oil. On one shared budget whichever comes
            # first takes everything, which is the exact failure that left
            # the first training set with no plain sea in it. Cap each
            # source on its own and the order stops mattering.
            if is_scene_negative:
                full = (max_scene_negatives is not None
                        and scene_neg_tiles >= max_scene_negatives)
            elif is_hard_negative:
                full = (max_hard_negatives is not None
                        and hard_neg_tiles >= max_hard_negatives)
            elif tile_klass == CLASS_SEA:
                full = max_plain_sea is not None and plain_sea_tiles >= max_plain_sea
            else:
                full = False

            cap = (limit_per_class or {}).get(tile_klass)
            if cap is not None and counts[tile_klass] >= cap:
                full = True
            if full:
                # `continue`, not `break`, for a negative. A scene negative
                # is an incidental by-product of an oil chip: breaking out
                # of the grid the moment the sea budget fills would abandon
                # the oil tiles further down that same chip, and the oil
                # class would quietly shrink toward nothing.
                if is_scene_negative:
                    continue
                break
            # Hard negatives get their own ceiling. Without one they are
            # simply sea tiles competing for the sea budget, and because
            # the look-alike folders are walked before `Images/No oil`,
            # they took all 150 of it: the first training run saw 400 oil
            # tiles, 150 look-alike hard negatives, and not one tile of
            # ordinary open water. Dark water that is not oil is valuable
            # supervision, but it is not a substitute for plain sea.
            if (is_hard_negative and max_hard_negatives is not None
                    and hard_neg_tiles >= max_hard_negatives):
                break

            # The source folder goes in the name because chip stems repeat
            # across it: Part III ships 00000.tif in Images/Oil, in
            # Images/No oil AND in Images/Lookalike. Keyed on the stem alone,
            # a sea tile cut from a No oil chip silently overwrote the sea
            # tile cut from the Lookalike chip at the same grid position. One
            # run wrote 1812 tiles and kept 1164; the 648 that disappeared
            # were mostly the negatives this rebalancing exists to add.
            src = _source_tag(image_path)
            stem = "%s%s_%s_r%04d_c%04d" % (src, Path(image_path).stem,
                                           tile_klass, r, c)
            # float16 halves the tile set on disk and costs nothing that
            # matters: Sigma0 in dB lives in roughly [-45, +5], where half
            # precision resolves better than 0.01 dB. Kaggle's working disk is
            # 20 GB and a full-precision tile set does not fit in it.
            np.savez_compressed(out_dir / (stem + ".npz"),
                                image=tile_img.astype(np.float16),
                                label=tile_lab.astype(np.uint8))
            counts[tile_klass] += 1
            if is_hard_negative:
                hard_neg_tiles += 1
            if is_scene_negative:
                scene_neg_tiles += 1
            if tile_klass == CLASS_SEA and not is_scene_negative and not is_hard_negative:
                plain_sea_tiles += 1
            specs.append(TileSpec(str(image_path), str(mask_path) if mask_path else None,
                                  tile_klass, r, c, size, oil_px, la_px))

    stats = _dataset_stats(out_dir)
    index = {
        "tiles": len(specs),
        "counts": {str(k): v for k, v in counts.items()},
        "skipped_images": skipped,
        "hard_negative_chips": hard_negatives,
        "hard_negative_tiles": hard_neg_tiles,
        "scene_negative_tiles": scene_neg_tiles,
        "plain_sea_tiles": plain_sea_tiles,
        "note": ("Look-alike chips carry no positive annotation in this dataset, "
                 "so they are kept as hard negatives labelled sea rather than "
                 "given invented class-1 labels."),
        "tile_size": size,
        "overlap": overlap,
        "normalisation": stats,
        "specs": [s.to_dict() for s in specs],
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


DEFAULT_STATS = {"mean_db": -18.0, "std_db": 6.0}


def _dataset_stats(tile_dir: Path, sample: int = 60) -> Dict[str, float]:
    """Mean and std in dB over a sample of tiles, stored in the checkpoint.

    Tiles are stored as float16 to fit Kaggle's disk, and float16 cannot hold
    the running sum: `np.std` over a single 512x512 tile already overflows to
    inf. A normalisation of inf makes every training input exactly zero, so the
    model would train on nothing and never say why. Accumulate in float64.
    """
    files = sorted(Path(tile_dir).glob("*.npz"))[:sample]
    if not files:
        return dict(DEFAULT_STATS)

    vals: List[np.ndarray] = []
    for f in files:
        a = np.asarray(np.load(f)["image"], dtype=np.float64)
        v = a[np.isfinite(a)]
        if v.size:
            vals.append(v[:: max(1, v.size // 20000)])
    if not vals:
        return dict(DEFAULT_STATS)

    allv = np.concatenate(vals)
    mean = float(np.mean(allv))
    std = float(np.std(allv))
    if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-3:
        print("  dataset statistics came out unusable (mean %r, std %r); "
              "falling back to %s" % (mean, std, DEFAULT_STATS))
        return dict(DEFAULT_STATS)
    return {"mean_db": mean, "std_db": std}


class TileDataset:
    """torch Dataset over the npz tiles. Only constructed when torch exists."""

    def __init__(self, tile_dir, files: Optional[Sequence[Path]] = None,
                 mean_db: float = -18.0, std_db: float = 6.0, augment: bool = False,
                 seed: int = 0):
        self.dir = Path(tile_dir)
        self.files = list(files) if files is not None else sorted(self.dir.glob("*.npz"))
        self.mean_db = float(mean_db)
        self.std_db = float(std_db) or 6.0
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        import torch

        z = np.load(self.files[i])
        img = normalise_db(np.asarray(z["image"], dtype=np.float32),
                           self.mean_db, self.std_db)
        lab = z["label"].astype(np.int64)

        if self.augment:
            if self.rng.random() < 0.5:
                img = img[:, :, ::-1].copy()
                lab = lab[:, ::-1].copy()
            if self.rng.random() < 0.5:
                img = img[:, ::-1, :].copy()
                lab = lab[::-1, :].copy()
            k = int(self.rng.integers(0, 4))
            if k:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                lab = np.rot90(lab, k).copy()

        return torch.from_numpy(img), torch.from_numpy(lab)


def split_files(tile_dir, val_fraction: float = 0.15, seed: int = 20260920):
    """Split by source image, never by tile, or val leaks into train."""
    files = sorted(Path(tile_dir).glob("*.npz"))
    groups: Dict[str, List[Path]] = {}
    for f in files:
        key = f.stem.split("_r")[0]
        groups.setdefault(key, []).append(f)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_fraction))
    val_keys = set(keys[:n_val])
    train, val = [], []
    for k, fs in groups.items():
        (val if k in val_keys else train).extend(fs)
    return sorted(train), sorted(val)
