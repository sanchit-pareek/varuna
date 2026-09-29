"""A real training run, small enough to finish on a laptop CPU.

Two epochs, a handful of tiles, random encoder weights. The point is not the
model, which will be worthless. The point is that every mechanism the Kaggle run
depends on has actually executed at least once before eight GPU hours are spent
on it:

  the loss goes through combo_loss without a shape error
  validation builds a confusion matrix and produces the three IoUs
  the best checkpoint is written, and stays under the 80 MB budget
  the resume artefact is written and can be loaded back
  --resume continues from the recorded epoch instead of starting over
  the report carries the model AND the dB baseline on the same tiles

Every one of those has a failure mode that only shows up at run time, and the
GPU is the worst place to discover them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from app.ml import dataset as ds, train as train_mod   # noqa: E402

TILE = 128          # small on purpose; the network is fully convolutional
N_SOURCES = 6       # split_files groups by source image, so several are needed


@pytest.fixture(scope="module")
def tiny_tiles(tmp_path_factory):
    """A tile set with all three classes, shaped like the real one."""
    out = tmp_path_factory.mktemp("tiles")
    rng = np.random.default_rng(11)

    for src in range(N_SOURCES):
        for klass in (ds.CLASS_OIL, ds.CLASS_LOOKALIKE, ds.CLASS_SEA):
            img = rng.normal(-18.0, 3.0, (2, TILE, TILE)).astype(np.float32)
            lab = np.zeros((TILE, TILE), dtype=np.uint8)
            if klass != ds.CLASS_SEA:
                # A dark patch with the matching label, so the loss has signal.
                img[:, 30:90, 40:100] -= 6.0
                lab[30:90, 40:100] = klass
            np.savez_compressed(
                out / ("src%02d_%d_r0000_c0000.npz" % (src, klass)),
                image=img.astype(np.float16), label=lab)

    (out / "index.json").write_text(json.dumps(
        {"normalisation": {"mean_db": -18.0, "std_db": 3.0}}), encoding="utf-8")
    return out


def test_two_epochs_produce_a_usable_checkpoint(tiny_tiles, tmp_path):
    out = Path(tmp_path) / "models" / "oil_unet_best.pt"
    report = train_mod.train(
        tile_dir=tiny_tiles,
        out_path=out,
        encoder="resnet18",          # smallest sane encoder for a CPU smoke run
        encoder_weights=None,        # no ImageNet download
        epochs=2,
        batch_size=2,
        workers=0,
        amp=False,
        val_fraction=0.34,
        # Random tiles cannot clear the water->oil gate; this test is about
        # plumbing, so disable it here rather than weakening the default.
        max_water_false_oil=1.0,
    )

    assert out.exists(), "no checkpoint was written"
    mb = out.stat().st_size / 1e6
    assert mb < 80, "checkpoint is %.1f MB, over the spec's budget" % mb

    assert report["epochs_completed"] == 2
    assert len(report["history"]) == 2
    for row in report["history"]:
        assert np.isfinite(row["train_loss"]), "loss went non-finite"
        for key in ("iou_sea", "iou_lookalike", "iou_oil", "pixel_accuracy"):
            assert key in row

    # The honesty requirement: the baseline is measured on the same tiles.
    assert "baseline" in report and "iou_oil" in report["baseline"]
    assert "delta_iou_oil" in report

    written = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert written["best"]["iou_oil"] == pytest.approx(report["best"]["iou_oil"])


def test_the_checkpoint_loads_back_for_inference(tiny_tiles, tmp_path):
    """A checkpoint training wrote must be readable by the inference path."""
    from app.ml import infer, model as model_mod

    out = Path(tmp_path) / "m" / "oil_unet_best.pt"
    train_mod.train(tile_dir=tiny_tiles, out_path=out, encoder="resnet18",
                    encoder_weights=None, epochs=1, batch_size=2, workers=0,
                    amp=False, val_fraction=0.34, max_water_false_oil=1.0)

    loaded = model_mod.load_checkpoint(out, map_location="cpu")
    assert loaded["encoder"] == "resnet18"
    assert loaded["normalisation"]["mean_db"] == pytest.approx(-18.0)

    scene = np.random.default_rng(2).normal(-18.0, 3.0, (2, 200, 260)).astype(np.float32)
    result = infer.run_unet(scene, loaded, tile=128, overlap=32, batch_size=2)
    probs = result["probs"]
    assert probs.shape == (3, 200, 260)
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-3)


def test_resume_continues_instead_of_restarting(tiny_tiles, tmp_path):
    """The mechanism that makes a Kaggle time limit survivable."""
    out = Path(tmp_path) / "r" / "oil_unet_best.pt"

    first = train_mod.train(tile_dir=tiny_tiles, out_path=out, encoder="resnet18",
                            encoder_weights=None, epochs=2, batch_size=2,
                            workers=0, amp=False, val_fraction=0.34,
                            max_water_false_oil=1.0)
    assert first["epochs_completed"] == 2

    resume_file = out.parent / train_mod.RESUME_NAME
    assert resume_file.exists(), "no resume artefact was written"

    second = train_mod.train(tile_dir=tiny_tiles, out_path=out, encoder="resnet18",
                             encoder_weights=None, epochs=4, batch_size=2,
                             workers=0, amp=False, val_fraction=0.34, resume=True,
                             max_water_false_oil=1.0)
    # It ran epochs 3 and 4 only.
    assert [r["epoch"] for r in second["history"]] == [3, 4], second["history"]
    assert second["epochs_completed"] == 4


def test_time_budget_stops_cleanly_and_flags_it(tiny_tiles, tmp_path):
    """A budget of zero must stop after one epoch, not be killed mid-run."""
    out = Path(tmp_path) / "t" / "oil_unet_best.pt"
    report = train_mod.train(tile_dir=tiny_tiles, out_path=out, encoder="resnet18",
                             encoder_weights=None, epochs=10, batch_size=2,
                             workers=0, amp=False, val_fraction=0.34,
                             time_budget_s=0.0, max_water_false_oil=1.0)
    assert report["stopped_early"] is True
    assert report["epochs_completed"] == 1
    assert out.exists(), "a budget stop must still leave the best checkpoint"


def test_metrics_maths_is_right():
    """IoU from a confusion matrix, checked by hand."""
    pred = np.array([0, 0, 1, 1, 2, 2, 2, 0])
    truth = np.array([0, 0, 1, 2, 2, 2, 1, 0])
    cm = train_mod.confusion(pred, truth)
    m = train_mod.metrics_from_confusion(cm)

    # sea: 3 predicted, 3 true, 3 correct -> IoU 1.0
    assert m["iou_sea"] == pytest.approx(1.0)
    # oil: predicted {4,5,6}, true {4,5,3} -> intersection 2, union 4
    assert m["iou_oil"] == pytest.approx(0.5)
    assert m["pixel_accuracy"] == pytest.approx(6 / 8)


def test_the_water_false_oil_gate_blocks_an_over_predicting_model(tiny_tiles, tmp_path):
    """The check that would have caught the first checkpoint.

    That model scored IoU oil 0.857 on held-out tiles and then called every
    pixel of three real scenes oil, the clean-water control included. IoU_oil
    is computed over tiles selected for containing oil, so it cannot see that
    failure: answering "oil" everywhere scores well on them by construction.

    Training on these random tiles reliably produces exactly that pathology, so
    it doubles as a fixture for it. With the gate at its real value no epoch
    qualifies and nothing is written, which is the correct outcome: refusing to
    ship beats shipping a detector that cannot find water.
    """
    out = Path(tmp_path) / "gate" / "oil_unet_best.pt"
    report = train_mod.train(tile_dir=tiny_tiles, out_path=out, encoder="resnet18",
                             encoder_weights=None, epochs=2, batch_size=2,
                             workers=0, amp=False, val_fraction=0.34)

    assert all(h["water_false_oil"] > train_mod.MAX_WATER_FALSE_OIL
               for h in report["history"]), report["history"]
    assert not out.exists(), "an over-predicting model must not be written"
    assert report["best"].get("iou_oil", -1.0) < 0


def test_metrics_report_the_water_false_oil_rate():
    """A model that answers oil everywhere scores 1.0, whatever its IoU says."""
    import numpy as np

    # cm is [truth, prediction]. Every water pixel, sea and look-alike, is
    # predicted oil; the oil pixels happen to be right.
    cm = np.array([[0, 0, 800],
                   [0, 0, 100],
                   [0, 0, 100]], dtype=np.int64)
    m = train_mod.metrics_from_confusion(cm)
    assert m["water_false_oil"] == 1.0
    assert m["sea_false_oil"] == 1.0

    # A model that never confuses water for oil scores 0.
    clean = np.array([[800, 0, 0],
                      [0, 100, 0],
                      [0, 0, 100]], dtype=np.int64)
    assert train_mod.metrics_from_confusion(clean)["water_false_oil"] == 0.0
