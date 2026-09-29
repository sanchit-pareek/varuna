"""Tiling against the real Zenodo Part III directory layout.

These exist because a Kaggle run burned 80 minutes downloading 9.86 GB and
produced zero tiles, silently. Two things were wrong and neither raised:

  1. Part III ships `Images/Oil/x.tif` with its label at `Mask/Oil/x.tif`, a
     sibling tree. The mask lookup only searched subfolders, found nothing, and
     an image with no mask is simply labelled all-sea. The oil class vanished
     with no error.
  2. `class_for_folder("Images/No oil")` matched the substring "oil" before it
     matched "no_oil", so clean water would have been labelled as a spill.

The second is the dangerous one. It fails in the direction nobody checks: the
model learns that empty sea is oil, the IoU table still looks plausible, and
the demo confidently paints slicks on open water.

The layout below is copied from the archive listing the Kaggle run printed:

    Images/Lookalike  150      Mask/Lookalike  150
    Images/No oil     150      Mask/No oil     150
    Images/Oil        150      Mask/Oil        150
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.ml import dataset as ds


CLASSES = [("Oil", ds.CLASS_OIL), ("Lookalike", ds.CLASS_LOOKALIKE), ("No oil", ds.CLASS_SEA)]


@pytest.fixture
def zenodo_part3(tmp_path):
    """A miniature of the real archive: Images/<class> beside Mask/<class>."""
    from app.geo import raster as raster_mod

    root = Path(tmp_path) / "part3"
    rng = np.random.default_rng(3)
    transform = (0.0002, 0.0, 71.0, 0.0, -0.0002, 19.0)

    for name, _klass in CLASSES:
        (root / "Images" / name).mkdir(parents=True, exist_ok=True)
        (root / "Mask" / name).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            fname = "chip_%s_%d.tif" % (name.replace(" ", ""), i)
            sar = rng.normal(-18.0, 3.0, (2, 640, 640)).astype(np.float32)
            raster_mod.write_geotiff(root / "Images" / name / fname, sar, transform)

            mask = np.zeros((640, 640), dtype=np.uint8)
            if name != "No oil":
                mask[100:400, 120:520] = 1     # binary, as the source ships it
            # Real Part III names: the label carries a _segmentation suffix.
            mask_name = fname.replace(".tif", "_segmentation.tif")
            raster_mod.write_geotiff(root / "Mask" / name / mask_name,
                                     mask, (1.0, 0.0, 0.0, 0.0, -1.0, 0.0))
    return root


def test_no_oil_is_not_labelled_oil():
    """The substring trap. This is the assertion that matters most."""
    assert ds.class_for_folder("Images/No oil") == ds.CLASS_SEA
    assert ds.class_for_folder("Images/no_oil") == ds.CLASS_SEA
    assert ds.class_for_folder("01_Train_Val_No_Oil_Images") == ds.CLASS_SEA
    assert ds.class_for_folder("oil-free") == ds.CLASS_SEA
    # And the positive cases still resolve.
    assert ds.class_for_folder("Images/Oil") == ds.CLASS_OIL
    assert ds.class_for_folder("01_Train_Val_Oil_Spill_images") == ds.CLASS_OIL
    assert ds.class_for_folder("Images/Lookalike") == ds.CLASS_LOOKALIKE
    assert ds.class_for_folder("01_Train_Val_Lookalike_images") == ds.CLASS_LOOKALIKE


def test_mask_trees_are_not_mistaken_for_image_trees():
    assert ds.is_mask_path("Mask/Oil")
    assert ds.is_mask_path("Images/Oil/masks")
    assert ds.is_mask_path("ground_truth/Oil")
    assert not ds.is_mask_path("Images/Oil")
    assert not ds.is_mask_path("Images/No oil")


def test_sibling_mask_tree_is_paired(zenodo_part3):
    """Images/Oil/x.tif must find Mask/Oil/x.tif."""
    pairs = ds.index_pairs(zenodo_part3)
    assert pairs, "no image/mask pairs found in the Part III layout"

    # Six images, two per class, and every one of them has a mask.
    assert len(pairs) == 6, [str(p[0]) for p in pairs]
    unmatched = [str(img) for img, mask, _k in pairs if mask is None]
    assert not unmatched, "these images found no mask: %s" % unmatched

    by_class = {}
    for _img, _mask, klass in pairs:
        by_class[klass] = by_class.get(klass, 0) + 1
    assert by_class == {ds.CLASS_OIL: 2, ds.CLASS_LOOKALIKE: 2, ds.CLASS_SEA: 2}, by_class

    # Nothing from the Mask tree was picked up as an image.
    assert not any("Mask" in Path(img).parts for img, _m, _k in pairs)


def test_tiling_the_real_layout_produces_all_three_classes(zenodo_part3, tmp_path):
    """The end to end check the Kaggle run needed and did not have."""
    out = Path(tmp_path) / "tiles"
    pairs = ds.index_pairs(zenodo_part3)
    index = ds.build_tile_index(pairs, out, size=256, overlap=32)

    assert index["tiles"] > 0, "tiling produced nothing from a valid layout"
    counts = {int(k): v for k, v in index["counts"].items()}
    assert counts[ds.CLASS_OIL] > 0, "no oil tiles: %s" % counts
    assert counts[ds.CLASS_LOOKALIKE] > 0, "no look-alike tiles: %s" % counts

    # Pixel labels must carry the class, not the source mask's binary 1.
    seen = set()
    for f in sorted(out.glob("*.npz")):
        seen.update(np.unique(np.load(f)["label"]).tolist())
    assert 2 in seen, "no pixel is labelled mineral oil"
    assert 1 in seen, "no pixel is labelled look-alike"
    assert seen <= {0, 1, 2}, "unexpected label values: %s" % seen


def test_no_oil_chips_contribute_no_positive_pixels(zenodo_part3, tmp_path):
    """A clean chip must stay clean all the way through tiling."""
    out = Path(tmp_path) / "tiles_sea"
    pairs = [t for t in ds.index_pairs(zenodo_part3) if t[2] == ds.CLASS_SEA]
    assert pairs
    ds.build_tile_index(pairs, out, size=256, overlap=32)

    for f in sorted(out.glob("*.npz")):
        lab = np.load(f)["label"]
        assert lab.max() == 0, "a No oil chip produced a positive label in %s" % f.name


def test_tiles_are_float16_and_small(zenodo_part3, tmp_path):
    """Kaggle's 20 GB working disk is the constraint that forced half precision."""
    out = Path(tmp_path) / "tiles_dtype"
    ds.build_tile_index(ds.index_pairs(zenodo_part3), out, size=256, overlap=32)
    f = sorted(out.glob("*.npz"))[0]
    z = np.load(f)
    assert z["image"].dtype == np.float16
    assert z["label"].dtype == np.uint8
    # Half precision must still resolve dB to well under a tenth of a dB.
    assert np.abs(z["image"].astype(np.float32)).max() < 200


# ---------------------------------------------------------------------------
# Streaming extraction
#
# The archive is 9.86 GB and cannot be expanded whole on Kaggle, so the data
# kernel extracts a batch of chips, tiles them, deletes them, and repeats. The
# batch selection is the part that went wrong twice, so it is tested against the
# exact member list the real archive reports.
# ---------------------------------------------------------------------------

REAL_MEMBERS = (
    [f"Images/Oil/{i:05d}.tif" for i in range(150)]
    + [f"Images/Lookalike/{i:05d}.tif" for i in range(150)]
    + [f"Images/No oil/{i:05d}.tif" for i in range(150)]
    + [f"Mask/Oil/{i:05d}_segmentation.tif" for i in range(150)]
    + [f"Mask/Lookalike/{i:05d}_segmentation.tif" for i in range(150)]
    + [f"Mask/No oil/{i:05d}_segmentation.tif" for i in range(150)]
)


def test_archive_grouping_matches_the_real_listing():
    grouped = ds.group_archive_members(REAL_MEMBERS)
    assert sorted(grouped) == [
        "Images/Lookalike", "Images/No oil", "Images/Oil",
        "Mask/Lookalike", "Mask/No oil", "Mask/Oil",
    ]
    assert all(len(v) == 150 for v in grouped.values())
    assert ds.image_folders(grouped) == ["Images/Lookalike", "Images/No oil", "Images/Oil"]
    assert ds.mask_folders(grouped) == ["Mask/Lookalike", "Mask/No oil", "Mask/Oil"]


def test_a_batch_pulls_the_masks_out_with_the_images():
    """The bug that made a green kernel deliver nothing."""
    grouped = ds.group_archive_members(REAL_MEMBERS)
    stems = ["00000", "00001", "00002"]
    targets = ds.select_batch_targets(grouped, "Images/Oil", stems)

    assert targets == [
        "Images/Oil/00000.tif", "Images/Oil/00001.tif", "Images/Oil/00002.tif",
        "Mask/Oil/00000_segmentation.tif",
        "Mask/Oil/00001_segmentation.tif",
        "Mask/Oil/00002_segmentation.tif",
    ], targets

    # Only the matching class's masks come along, never another class's.
    assert not any("Lookalike" in t or "No oil" in t for t in targets)


def test_batch_targets_respect_the_class_leaf():
    grouped = ds.group_archive_members(REAL_MEMBERS)
    for folder, mask_dir in [("Images/Lookalike", "Mask/Lookalike"),
                             ("Images/No oil", "Mask/No oil")]:
        targets = ds.select_batch_targets(grouped, folder, ["00007"])
        assert targets == [f"{folder}/00007.tif",
                           f"{mask_dir}/00007_segmentation.tif"], targets


def test_extracting_a_batch_then_pairing_finds_the_masks(tmp_path):
    """Simulate one streaming round on disk, end to end."""
    from app.geo import raster as raster_mod

    stage = Path(tmp_path) / "stage"
    grouped = ds.group_archive_members(REAL_MEMBERS)
    targets = ds.select_batch_targets(grouped, "Images/Oil", ["00000", "00001"])

    transform = (0.0002, 0.0, 71.0, 0.0, -0.0002, 19.0)
    for t in targets:
        dest = stage / t
        dest.parent.mkdir(parents=True, exist_ok=True)
        if ds.is_mask_path(Path(t).parent):
            m = np.zeros((320, 320), dtype=np.uint8)
            m[80:240, 60:260] = 1
            raster_mod.write_geotiff(dest, m, (1.0, 0.0, 0.0, 0.0, -1.0, 0.0))
        else:
            raster_mod.write_geotiff(
                dest, np.full((2, 320, 320), -18.0, dtype=np.float32), transform)

    pairs = ds.index_pairs(stage)
    assert len(pairs) == 2, pairs
    assert all(mask is not None for _img, mask, _k in pairs), \
        "the streaming round extracted masks but pairing still missed them"
    assert all(k == ds.CLASS_OIL for _i, _m, k in pairs)

    out = Path(tmp_path) / "tiles"
    index = ds.build_tile_index(pairs, out, size=256, overlap=32)
    assert int(index["counts"]["2"]) > 0, "a streaming round produced no oil tiles"


def test_a_chip_with_no_georeference_is_still_tiled(tmp_path):
    """Zenodo chips often ship with no CRS. Training must not lose them.

    `load_sar` refuses them, correctly, because the product must never compute
    a coordinate from an ungeoreferenced raster. The tiler has to fall back and
    keep the pixels, or the oil class quietly shrinks.
    """
    from app.geo import raster as raster_mod

    root = Path(tmp_path) / "bare"
    (root / "Images" / "Oil").mkdir(parents=True)
    (root / "Mask" / "Oil").mkdir(parents=True)

    identity = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    img = np.full((2, 320, 320), -18.0, dtype=np.float32)
    img[:, 60:260, 80:240] -= 7.0
    raster_mod.write_geotiff(root / "Images" / "Oil" / "a.tif", img, identity)

    mask = np.zeros((320, 320), dtype=np.uint8)
    mask[60:260, 80:240] = 1
    raster_mod.write_geotiff(root / "Mask" / "Oil" / "a.tif", mask, identity)

    pairs = ds.index_pairs(root)
    assert len(pairs) == 1 and pairs[0][1] is not None

    out = Path(tmp_path) / "tiles"
    index = ds.build_tile_index(pairs, out, size=256, overlap=32)
    assert index["skipped_images"] == 0, "an ungeoreferenced chip was skipped"
    assert int(index["counts"]["2"]) > 0, "no oil tiles from an ungeoreferenced chip"


def test_load_raw_reads_what_load_sar_refuses(tmp_path):
    from app.geo import raster as raster_mod

    p = Path(tmp_path) / "bare.tif"
    raster_mod.write_geotiff(p, np.full((2, 64, 64), -12.0, dtype=np.float32),
                             (1.0, 0.0, 0.0, 0.0, -1.0, 0.0))
    raw = raster_mod.load_raw(p)
    assert raw.array.shape == (2, 64, 64)
    assert raw.meta["read_as"] == "raw"


def test_mask_key_strips_the_suffixes_this_data_actually_uses():
    """Locked to the real names, read straight off the archive header."""
    assert ds.mask_key("00000_segmentation.tif") == "00000"
    assert ds.mask_key("00000.tif") == "00000"
    assert ds.mask_key("chip_mask.tif") == "chip"
    assert ds.mask_key("chip_gt.tif") == "chip"
    # A name that merely contains the word must not be truncated.
    assert ds.mask_key("segmentation_run.tif") == "segmentation_run"


def test_suffixed_mask_in_a_sibling_tree_is_found(tmp_path):
    """Images/Oil/00000.tif -> Mask/Oil/00000_segmentation.tif."""
    from app.geo import raster as raster_mod

    root = Path(tmp_path) / "p3"
    (root / "Images" / "Oil").mkdir(parents=True)
    (root / "Mask" / "Oil").mkdir(parents=True)
    identity = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    img = np.full((2, 320, 320), -18.0, dtype=np.float32)
    img[:, 60:260, 80:240] -= 7.0
    raster_mod.write_geotiff(root / "Images" / "Oil" / "00000.tif", img, identity)
    m = np.zeros((320, 320), dtype=np.uint8)
    m[60:260, 80:240] = 1
    raster_mod.write_geotiff(root / "Mask" / "Oil" / "00000_segmentation.tif", m, identity)

    pairs = ds.index_pairs(root)
    assert len(pairs) == 1
    img_p, mask_p, klass = pairs[0]
    assert mask_p is not None, "the _segmentation mask was not found"
    assert mask_p.name == "00000_segmentation.tif"
    assert klass == ds.CLASS_OIL

    out = Path(tmp_path) / "tiles"
    index = ds.build_tile_index(pairs, out, size=256, overlap=32)
    assert int(index["counts"]["2"]) > 0, "no oil tiles despite a matched mask"


# ---------------------------------------------------------------------------
# What the Zenodo ground truth actually contains
#
# Measured, not assumed: every sampled 2048x2048 look-alike mask in Part II
# contains only the value 0. The dataset segments oil and nothing else, so a
# look-alike chip has no positive annotation at all. The spec sheet's rule
# "look-alike folder mask 1 -> class 1" cannot be applied, because there is no
# mask value 1 to map.
# ---------------------------------------------------------------------------

def test_a_lookalike_chip_with_an_empty_mask_becomes_a_hard_negative(tmp_path):
    """Dark water that is not oil is the most useful negative in the set."""
    from app.geo import raster as raster_mod

    root = Path(tmp_path) / "la"
    (root / "Images" / "Lookalike").mkdir(parents=True)
    (root / "Mask" / "Lookalike").mkdir(parents=True)
    identity = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    # A dark patch in the imagery, and an entirely empty label, as shipped.
    img = np.full((2, 320, 320), -18.0, dtype=np.float32)
    img[:, 60:260, 80:240] -= 7.0
    raster_mod.write_geotiff(root / "Images" / "Lookalike" / "00000.tif", img, identity)
    raster_mod.write_geotiff(root / "Mask" / "Lookalike" / "00000_segmentation.tif",
                             np.zeros((320, 320), dtype=np.uint8), identity)

    out = Path(tmp_path) / "tiles"
    index = ds.build_tile_index(ds.index_pairs(root), out, size=256, overlap=32)

    counts = {int(k): v for k, v in index["counts"].items()}
    assert counts[ds.CLASS_SEA] > 0, "the look-alike chip was dropped entirely"
    assert counts[ds.CLASS_LOOKALIKE] == 0, "an empty mask must not create class 1"
    assert index["hard_negative_chips"] == 1

    for f in sorted(out.glob("*.npz")):
        assert np.load(f)["label"].max() == 0


def test_an_oil_chip_is_still_labelled_oil(tmp_path):
    """The hard-negative rule must not swallow real positives."""
    from app.geo import raster as raster_mod

    root = Path(tmp_path) / "oil"
    (root / "Images" / "Oil").mkdir(parents=True)
    (root / "Mask" / "Oil").mkdir(parents=True)
    identity = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    img = np.full((2, 320, 320), -18.0, dtype=np.float32)
    img[:, 60:260, 80:240] -= 8.0
    raster_mod.write_geotiff(root / "Images" / "Oil" / "00000.tif", img, identity)
    m = np.zeros((320, 320), dtype=np.uint8)
    m[60:260, 80:240] = 1
    raster_mod.write_geotiff(root / "Mask" / "Oil" / "00000_segmentation.tif", m, identity)

    out = Path(tmp_path) / "tiles"
    index = ds.build_tile_index(ds.index_pairs(root), out, size=256, overlap=32)
    counts = {int(k): v for k, v in index["counts"].items()}
    assert counts[ds.CLASS_OIL] > 0
    assert index["hard_negative_chips"] == 0


def test_statistics_do_not_overflow_on_float16_tiles(tmp_path):
    """std over a float16 512 tile is inf, and inf normalisation zeroes the input."""
    out = Path(tmp_path) / "stats"
    out.mkdir()
    rng = np.random.default_rng(5)
    for i in range(4):
        np.savez_compressed(
            out / ("s%02d_0_r0000_c0000.npz" % i),
            image=rng.normal(-19.0, 4.0, (2, 512, 512)).astype(np.float16),
            label=np.zeros((512, 512), dtype=np.uint8))

    stats = ds._dataset_stats(out)
    assert np.isfinite(stats["mean_db"]), stats
    assert np.isfinite(stats["std_db"]), stats
    assert 2.0 < stats["std_db"] < 6.0, stats
    assert -22.0 < stats["mean_db"] < -16.0, stats


def test_normalisation_survives_a_broken_sigma():
    """Defence in depth: an inf or zero sigma must not flatten the input."""
    a = np.full((2, 8, 8), -12.0, dtype=np.float32)
    for bad in (float("inf"), float("nan"), 0.0):
        x = ds.normalise_db(a, -18.0, bad)
        assert np.isfinite(x).all(), "sigma %r produced non-finite input" % bad
        assert abs(float(x.mean())) > 1e-6, "sigma %r flattened the input to zero" % bad


def _oil_chip(root, name, oil_box):
    """One georeferenced chip plus its mask, oil confined to `oil_box`."""
    from app.geo import raster as raster_mod

    (root / "Images" / "Oil").mkdir(parents=True, exist_ok=True)
    (root / "Mask" / "Oil").mkdir(parents=True, exist_ok=True)
    identity = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    img = np.full((2, 768, 768), -18.0, dtype=np.float32)
    m = np.zeros((768, 768), dtype=np.uint8)
    r0, r1, c0, c1 = oil_box
    img[:, r0:r1, c0:c1] -= 8.0
    m[r0:r1, c0:c1] = 1
    raster_mod.write_geotiff(root / "Images" / "Oil" / (name + ".tif"), img, identity)
    raster_mod.write_geotiff(root / "Mask" / "Oil" / (name + "_segmentation.tif"),
                             m, identity)


def test_open_water_tiles_from_an_oil_scene_are_kept_as_negatives(tmp_path):
    """The fix for a model that called every pixel of the ocean oil.

    A 768x768 chip with oil in one corner tiles into several squares, only one
    of which holds any oil. Dropping the rest is what left training at 73% oil
    tiles against roughly 0.1% in a real scene. They are the best negatives in
    the set: same sensor, same pass, same sea state.
    """
    root = Path(tmp_path) / "sceneneg"
    _oil_chip(root, "00000", (0, 200, 0, 200))
    pairs = ds.index_pairs(root)

    without = ds.build_tile_index(pairs, Path(tmp_path) / "a", size=256, overlap=0)
    with_neg = ds.build_tile_index(pairs, Path(tmp_path) / "b", size=256, overlap=0,
                                   max_scene_negatives=50)

    assert without.get("scene_negative_tiles", 0) == 0
    assert with_neg["scene_negative_tiles"] > 0, "empty tiles were still discarded"

    a = {int(k): v for k, v in without["counts"].items()}
    b = {int(k): v for k, v in with_neg["counts"].items()}
    assert b[ds.CLASS_OIL] == a[ds.CLASS_OIL], "positives must be untouched"
    assert b[ds.CLASS_SEA] > a[ds.CLASS_SEA]

    # Every kept negative really is empty, and the sea tiles now outnumber the
    # oil ones, which is the ratio the first training run had inverted.
    for f in sorted((Path(tmp_path) / "b").glob("*_0_*.npz")):
        assert np.load(f)["label"].max() == 0
    assert b[ds.CLASS_SEA] > b[ds.CLASS_OIL]


def test_the_scene_negative_budget_is_respected(tmp_path):
    root = Path(tmp_path) / "cap"
    for i in range(3):
        _oil_chip(root, "%05d" % i, (0, 200, 0, 200))

    index = ds.build_tile_index(ds.index_pairs(root), Path(tmp_path) / "t",
                                size=256, overlap=0, max_scene_negatives=2)
    assert index["scene_negative_tiles"] == 2


def test_hard_negatives_cannot_starve_the_plain_sea_budget(tmp_path):
    """The precise mechanism behind the first broken checkpoint.

    Look-alike folders are walked before `Images/No oil`, and every look-alike
    chip is reclassified to sea because its mask is empty. With one shared sea
    budget they took all of it, and the training set contained no ordinary open
    water at all. Their own ceiling leaves room for the real thing.
    """
    from app.geo import raster as raster_mod

    root = Path(tmp_path) / "starve"
    (root / "Images" / "Lookalike").mkdir(parents=True)
    (root / "Mask" / "Lookalike").mkdir(parents=True)
    identity = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    for i in range(4):
        img = np.full((2, 768, 768), -25.0, dtype=np.float32)
        raster_mod.write_geotiff(root / "Images" / "Lookalike" / ("%05d.tif" % i),
                                 img, identity)
        raster_mod.write_geotiff(
            root / "Mask" / "Lookalike" / ("%05d_segmentation.tif" % i),
            np.zeros((768, 768), dtype=np.uint8), identity)

    pairs = ds.index_pairs(root)
    limits = {ds.CLASS_SEA: 100, ds.CLASS_LOOKALIKE: 100, ds.CLASS_OIL: 100}

    unbounded = ds.build_tile_index(pairs, Path(tmp_path) / "u", size=256,
                                    overlap=0, limit_per_class=limits)
    bounded = ds.build_tile_index(pairs, Path(tmp_path) / "c", size=256, overlap=0,
                                  limit_per_class=limits, max_hard_negatives=5)

    assert unbounded["hard_negative_tiles"] > 5
    assert bounded["hard_negative_tiles"] == 5
    # With the ceiling in place the sea budget still has room left over, which
    # is exactly the room `Images/No oil` needs.
    assert bounded["counts"][str(ds.CLASS_SEA)] < limits[ds.CLASS_SEA]


def test_a_full_sea_budget_does_not_abandon_the_oil_tiles_below_it(tmp_path):
    """A negative hitting its ceiling must skip that tile, not end the chip.

    Tiles are walked row-major, so an oil chip whose slick sits in the lower
    half yields several empty tiles first. Those become scene negatives. If a
    full sea budget breaks out of the grid rather than skipping the tile, every
    oil tile below the slick is abandoned and the oil class shrinks toward
    nothing with no error anywhere.
    """
    root = Path(tmp_path) / "below"
    _oil_chip(root, "00000", (520, 760, 0, 240))   # oil only in the bottom rows
    pairs = ds.index_pairs(root)

    # Budget deliberately exhausted after a single scene negative.
    index = ds.build_tile_index(pairs, Path(tmp_path) / "t", size=256, overlap=0,
                                max_scene_negatives=1)

    counts = {int(k): v for k, v in index["counts"].items()}
    assert index["scene_negative_tiles"] == 1
    assert counts[ds.CLASS_OIL] > 0, (
        "the oil tiles below the exhausted sea budget were dropped")


def test_the_three_class_zero_budgets_are_independent(tmp_path):
    """Plain sea, hard negatives and scene negatives cannot starve each other.

    Folders are walked Lookalike, then No oil, then Oil. Under one shared sea
    budget the first folder took all of it, which is how the first training set
    ended up with no ordinary open water in it at all.
    """
    from app.geo import raster as raster_mod

    root = Path(tmp_path) / "three"
    identity = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    for leaf in ("Lookalike", "No oil"):
        (root / "Images" / leaf).mkdir(parents=True)
        (root / "Mask" / leaf).mkdir(parents=True)
        for i in range(2):
            raster_mod.write_geotiff(root / "Images" / leaf / ("%05d.tif" % i),
                                     np.full((2, 768, 768), -24.0, dtype=np.float32),
                                     identity)
            raster_mod.write_geotiff(
                root / "Mask" / leaf / ("%05d_segmentation.tif" % i),
                np.zeros((768, 768), dtype=np.uint8), identity)
    _oil_chip(root, "00000", (0, 200, 0, 200))

    index = ds.build_tile_index(ds.index_pairs(root), Path(tmp_path) / "t",
                                size=256, overlap=0,
                                max_hard_negatives=3, max_scene_negatives=4,
                                max_plain_sea=5)

    assert index["hard_negative_tiles"] == 3
    assert index["scene_negative_tiles"] == 4
    assert index["plain_sea_tiles"] == 5, index["plain_sea_tiles"]
    # All three sources are represented, which is the whole point.
    assert index["counts"][str(ds.CLASS_SEA)] == 12
    assert index["counts"][str(ds.CLASS_OIL)] > 0


def test_same_stem_in_two_folders_does_not_overwrite(tmp_path):
    """Part III ships 00000.tif in all three image folders.

    Keyed on the chip stem alone, a sea tile cut from `Images/No oil` silently
    overwrote the sea tile cut from `Images/Lookalike` at the same grid
    position. One real run wrote 1812 tiles and kept 1164, and the 648 that
    vanished were mostly the negatives the rebalancing exists to add. Nothing
    raised, and the index went on reporting the number it had attempted.
    """
    from app.geo import raster as raster_mod

    root = Path(tmp_path) / "collide"
    identity = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    for leaf in ("Lookalike", "No oil"):
        (root / "Images" / leaf).mkdir(parents=True)
        (root / "Mask" / leaf).mkdir(parents=True)
        # Same stem in both folders, exactly as the archive ships it.
        raster_mod.write_geotiff(root / "Images" / leaf / "00000.tif",
                                 np.full((2, 512, 512), -24.0, dtype=np.float32),
                                 identity)
        raster_mod.write_geotiff(root / "Mask" / leaf / "00000_segmentation.tif",
                                 np.zeros((512, 512), dtype=np.uint8), identity)

    out = Path(tmp_path) / "tiles"
    index = ds.build_tile_index(ds.index_pairs(root), out, size=256, overlap=0)

    written = sorted(out.glob("*.npz"))
    attempted = sum(index["counts"].values())
    assert len(written) == attempted, (
        "%d tiles attempted but %d files on disk: names are colliding"
        % (attempted, len(written)))

    # The source folder survives in the name, so the two are distinguishable.
    names = " ".join(f.name for f in written)
    assert "lookalike_" in names and "nooil_" in names, names


def test_the_class_is_still_the_third_field_from_the_end(tmp_path):
    """The rebuild step parses class out of the filename with split('_')[-3]."""
    from app.geo import raster as raster_mod

    root = Path(tmp_path) / "parse"
    (root / "Images" / "Oil").mkdir(parents=True)
    (root / "Mask" / "Oil").mkdir(parents=True)
    identity = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    img = np.full((2, 512, 512), -18.0, dtype=np.float32)
    img[:, :256, :256] -= 8.0
    raster_mod.write_geotiff(root / "Images" / "Oil" / "00000.tif", img, identity)
    m = np.zeros((512, 512), dtype=np.uint8)
    m[:256, :256] = 1
    raster_mod.write_geotiff(root / "Mask" / "Oil" / "00000_segmentation.tif", m, identity)

    out = Path(tmp_path) / "t"
    ds.build_tile_index(ds.index_pairs(root), out, size=256, overlap=0,
                        max_scene_negatives=10)
    for f in sorted(out.glob("*.npz")):
        klass = int(f.stem.split("_")[-3])
        assert klass in (0, 1, 2), f.name
        assert int(np.load(f)["label"].max()) in (0, klass), f.name
