"""The U-Net inference path, exercised end to end.

These tests do not check that the model is *good*. They check that the path
through it works: a checkpoint on disk is found, its architecture and dB
normalisation are read back from the file rather than guessed, tiled inference
stitches without seams, and the result flows into the same polygon and
attribution code the baseline uses.

That distinction matters. Training quality is measured on Kaggle against the
dB baseline and reported in `oil_unet_best.report.json`. What can go wrong
*here* is plumbing: a checkpoint that loads into the wrong architecture, a
normalisation that drifts between training and inference, a stitching bug that
leaves tile edges in the mask. So the checkpoint used below is deliberately
untrained and its predictions are deliberately not asserted on.

Skipped when torch and segmentation_models_pytorch are absent, which is the
normal state of the demo laptop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")


@pytest.fixture
def untrained_checkpoint(tmp_path):
    """A real checkpoint with random weights, written the way training writes it."""
    from app.ml import model as model_mod

    net = model_mod.build(encoder_weights=None)
    path = Path(tmp_path) / "oil_unet_best.pt"
    model_mod.save_checkpoint(
        net, path,
        meta={"arch": model_mod.DEFAULT_ARCH, "encoder": model_mod.DEFAULT_ENCODER,
              "classes": 3, "in_channels": model_mod.IN_CHANNELS, "tile": 512,
              "normalisation": {"mean_db": -17.5, "std_db": 5.5}},
        metrics={"iou_oil": 0.0, "note": "untrained, for plumbing tests only"},
    )
    return path


def test_checkpoint_round_trips_its_own_metadata(untrained_checkpoint):
    """Inference must never guess what the weights were trained on."""
    from app.ml import model as model_mod

    loaded = model_mod.load_checkpoint(untrained_checkpoint, map_location="cpu")
    assert loaded["arch"] == model_mod.DEFAULT_ARCH
    assert loaded["encoder"] == model_mod.DEFAULT_ENCODER
    assert loaded["classes"] == 3
    assert loaded["normalisation"]["mean_db"] == pytest.approx(-17.5)
    assert loaded["normalisation"]["std_db"] == pytest.approx(5.5)


def test_checkpoint_fits_the_size_budget(untrained_checkpoint):
    """The demo laptop carries this file, so the spec caps it at 80 MB."""
    mb = untrained_checkpoint.stat().st_size / 1e6
    assert mb < 80, "checkpoint is %.1f MB, over the 80 MB budget" % mb


def test_tiled_inference_covers_every_pixel_without_seams(untrained_checkpoint):
    """Stitching must weight every pixel, including the padded right and bottom.

    A cosine taper that does not sum to a positive weight somewhere leaves a
    stripe of garbage at a tile boundary. Asserting the probabilities are a
    valid distribution everywhere catches that without needing a trained model.
    """
    from app.ml import infer, model as model_mod

    loaded = model_mod.load_checkpoint(untrained_checkpoint, map_location="cpu")
    rng = np.random.default_rng(0)
    # Deliberately not a multiple of the tile size, so padding is exercised.
    scene = rng.normal(-18.0, 4.0, (2, 600, 730)).astype(np.float32)

    out = infer.run_unet(scene, loaded, tile=256, overlap=64, batch_size=2)
    probs = out["probs"]

    assert probs.shape == (3, 600, 730)
    assert np.isfinite(probs).all(), "stitching produced non-finite probabilities"
    sums = probs.sum(axis=0)
    assert np.allclose(sums, 1.0, atol=1e-3), (
        "class probabilities do not sum to 1 everywhere; min %.4f max %.4f"
        % (sums.min(), sums.max()))
    assert out["n_tiles"] > 1, "the scene should have been split into several tiles"


def test_segment_scene_reports_unet_when_a_checkpoint_is_present(untrained_checkpoint, monkeypatch):
    """The whole point of the honesty rule: say which detector actually ran."""
    from app import config
    from app.ml import infer

    monkeypatch.setattr(config, "CHECKPOINT", untrained_checkpoint)
    infer.reset_model_cache()
    try:
        scene = np.random.default_rng(1).normal(-18.0, 4.0, (2, 512, 512)).astype(np.float32)
        out = infer.segment_scene(scene, prefer_model=True)

        assert out["method"] == "unet"
        assert out["fallback_reason"] is None
        assert out["model"]["arch"] and out["model"]["encoder"]
        assert out["mask"].shape == (512, 512)
        assert set(np.unique(out["mask"])) <= {0, 1, 2}
        assert out["oil_prob"].shape == (512, 512)
    finally:
        infer.reset_model_cache()


def test_a_broken_checkpoint_falls_back_instead_of_crashing(tmp_path, monkeypatch):
    """A corrupt file must not take the demo down. It must degrade and say so."""
    from app import config
    from app.ml import infer

    bad = Path(tmp_path) / "oil_unet_best.pt"
    bad.write_bytes(b"this is not a torch checkpoint")
    monkeypatch.setattr(config, "CHECKPOINT", bad)
    infer.reset_model_cache()
    try:
        scene = np.full((2, 256, 256), -20.0, dtype=np.float32)
        out = infer.segment_scene(scene, prefer_model=True)
        assert out["method"] == "sigma0_threshold_baseline"
        assert out["fallback_reason"], "the fallback must explain itself"
    finally:
        infer.reset_model_cache()


def test_full_pipeline_runs_through_the_unet(untrained_checkpoint, monkeypatch,
                                             selftest_scene, cached_metocean):
    """DETECT through SCORE, with the network path in place of the baseline."""
    from app import config, pipeline
    from app.ml import infer

    monkeypatch.setattr(config, "CHECKPOINT", untrained_checkpoint)
    infer.reset_model_cache()
    try:
        det = pipeline.detect_scene(selftest_scene, prefer_model=True,
                                    render_overlays=False)
        metrics = det["metrics"]
        assert metrics["detector"] == "unet"
        assert metrics["fallback_reason"] is None
        # Untrained weights say nothing useful about oil, so the count is not
        # asserted. What matters is that characterisation ran on whatever the
        # network produced and that the metrics table was computed.
        assert "accuracy_vs_truth" in metrics
        assert set(metrics["accuracy_vs_truth"]) == {
            "iou_sea", "iou_lookalike", "iou_oil", "pixel_accuracy"}
    finally:
        infer.reset_model_cache()


def test_normalisation_from_the_checkpoint_is_actually_applied():
    """A drift between training and inference statistics is silent and fatal."""
    from app.ml import infer

    scene = np.full((2, 8, 8), -10.0, dtype=np.float32)
    x = infer.prepare_input(scene, {"mean_db": -10.0, "std_db": 2.0})
    assert x.shape == (3, 8, 8)
    assert np.allclose(x, 0.0), "a value equal to the mean must standardise to zero"

    x2 = infer.prepare_input(scene, {"mean_db": -12.0, "std_db": 2.0})
    assert np.allclose(x2, 1.0), "two dB above the mean, over a 2 dB sigma, is 1.0"

    # The co-polarised band is repeated into the third channel when the encoder
    # wants three. This assertion used to read `y[0] == -10.0`, encoding the
    # assumption that bands arrive bright-first -- which is precisely the bug
    # that let a model train on (VH, VV) and predict on (VV, VH). Bands are now
    # ordered by brightness, so the darker cross-pol band is first whichever way
    # the source supplied them.
    stack = np.stack([np.full((4, 4), -10.0), np.full((4, 4), -25.0)]).astype(np.float32)
    y = infer.prepare_input(stack, {"mean_db": 0.0, "std_db": 1.0})
    assert np.allclose(y[0], -25.0) and np.allclose(y[1], -10.0)
    assert np.allclose(y[2], y[0]), "third channel must repeat the first"


def test_bands_are_ordered_by_brightness_not_by_arrival():
    """The bug that made a 0.889 IoU model unable to find water.

    Zenodo stores its two bands as (VH, VV) -- cross-pol first, several dB
    darker. `fetch_sentinel1_scene.py` reads Planetary Computer assets as
    ('vv', 'vh'), the opposite way round. So the network trained on
    (dark, bright) and was asked to predict on (bright, dark). It scored near
    perfectly on held-out tiles from its own archive and called 99.99% of a
    clean-water control scene oil.

    Nothing in the pixels announces the convention, so ordering by brightness
    is the check: over water, co-polarised backscatter exceeds cross-polarised
    in every sensor and every sea state.
    """
    import numpy as np

    from app.ml import dataset as ds

    co = np.full((8, 8), -18.0, dtype=np.float32)      # VV, bright
    cross = np.full((8, 8), -26.0, dtype=np.float32)   # VH, dark

    planetary = np.stack([co, cross])      # what the scene fetcher produces
    zenodo = np.stack([cross, co])         # what the archive ships

    a = ds.order_bands(planetary)
    b = ds.order_bands(zenodo)

    assert np.allclose(a, b), "the two conventions must normalise to one order"
    assert np.nanmedian(a[0]) < np.nanmedian(a[1]), "cross-pol must come first"
    assert np.allclose(a[0], cross) and np.allclose(a[1], co)


def test_ordering_is_stable_for_a_single_band():
    import numpy as np

    from app.ml import dataset as ds

    one = np.full((1, 4, 4), -20.0, dtype=np.float32)
    assert ds.order_bands(one).shape == (1, 4, 4)
    flat = np.full((4, 4), -20.0, dtype=np.float32)
    assert ds.order_bands(flat).shape == (1, 4, 4)


def test_prepare_input_repeats_the_co_pol_band():
    """Three channels for an ImageNet encoder: cross, co, cross."""
    import numpy as np

    from app.ml import infer

    co = np.full((6, 6), -18.0, dtype=np.float32)
    cross = np.full((6, 6), -26.0, dtype=np.float32)
    norm = {"mean_db": -22.0, "std_db": 4.0}

    out = infer.prepare_input(np.stack([co, cross]), norm)
    assert out.shape == (3, 6, 6)
    assert np.allclose(out[0], out[2]), "channel 3 repeats channel 1"
    assert out[0].mean() < out[1].mean(), "the darker band must be first"

    # Feeding the same scene in the other band order must not change anything.
    other = infer.prepare_input(np.stack([cross, co]), norm)
    assert np.allclose(out, other)
