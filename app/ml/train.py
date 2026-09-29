"""Training entry point. Runs on Kaggle, not on the demo laptop.

Kaggle is the factory, the laptop is the product. Nothing in the served
application imports this module.

    python -m app.ml.train --tiles /kaggle/input/tidetrace-tiles --epochs 40

Frozen recipe from the spec:
    UnetPlusPlus or Unet, encoder timm-efficientnet-b0 or resnet34
    3 classes, 512 tiles, 2 channel VV+VH with VV repeated to 3
    combined weighted cross entropy and soft Dice, oil weighted higher
    AdamW 1e-4, cosine schedule, AMP, batch 8 on P100 and 4 on T4
    report IoU_oil, IoU_lookalike and pixel accuracy, save best IoU_oil

The checkpoint carries its architecture and its dB normalisation statistics, so
inference never has to guess what it was trained on.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .. import config
from . import dataset as ds_mod, model as model_mod

# sea, look-alike, mineral oil.
#
# These were [0.4, 1.0, 2.5], which is a 6.25x relative push toward oil. That
# is the right shape of correction when oil is rare in training. It was not:
# tiles are kept only when they hold at least 50 oil pixels, so oil was 19.8%
# of training pixels, and the extra push produced a model that answered "oil"
# for essentially every pixel of open water. The Dice half of the loss already
# handles per-class imbalance; sea must not be suppressed on top of it.
CLASS_WEIGHTS = [1.0, 1.0, 1.5]

# An epoch that calls more than this fraction of genuine water "oil" is not
# eligible to become the shipped checkpoint, however good its IoU looks.
MAX_WATER_FALSE_OIL = 0.02


def dice_loss(logits, target, eps: float = 1e-6):
    import torch
    import torch.nn.functional as F

    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def combo_loss(logits, target, weights):
    import torch
    import torch.nn.functional as F

    w = torch.tensor(weights, device=logits.device, dtype=logits.dtype)
    ce = F.cross_entropy(logits, target, weight=w)
    return ce + dice_loss(logits, target)


def confusion(pred: np.ndarray, truth: np.ndarray, n: int = 3) -> np.ndarray:
    k = (truth >= 0) & (truth < n)
    return np.bincount(n * truth[k].astype(int) + pred[k].astype(int),
                       minlength=n * n).reshape(n, n)


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    """`cm` is indexed [truth, prediction]."""
    inter = np.diag(cm).astype(float)
    union = cm.sum(1) + cm.sum(0) - np.diag(cm)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)

    # The rate that matters at deployment and that IoU hides completely.
    #
    # The first checkpoint scored IoU oil 0.857 here and then painted every one
    # of the three demo scenes solid oil, including the clean-water control.
    # IoU_oil could not see that, because it is computed only over tiles that
    # were selected for containing oil: a model that says "oil" everywhere
    # scores well on them by construction. What catches it is asking how often
    # a pixel that is genuinely water gets called oil. On the broken checkpoint
    # that number was ~1.0. Track it every epoch and select on it.
    water = cm[0, :].sum() + cm[1, :].sum()
    water_as_oil = float(cm[0, 2] + cm[1, 2])
    sea_total = cm[0, :].sum()

    return {
        "iou_sea": float(iou[0]),
        "iou_lookalike": float(iou[1]),
        "iou_oil": float(iou[2]),
        "miou": float(np.nanmean(iou)),
        "pixel_accuracy": float(inter.sum() / max(cm.sum(), 1)),
        "water_false_oil": float(water_as_oil / max(water, 1)),
        "sea_false_oil": float(cm[0, 2] / max(sea_total, 1)),
    }


def evaluate(model, loader, device) -> Dict[str, float]:
    import torch

    model.eval()
    cm = np.zeros((3, 3), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            cm += confusion(pred.ravel(), y.numpy().ravel())
    return metrics_from_confusion(cm)


def baseline_metrics(files: List[Path], mean_db: float, std_db: float) -> Dict[str, float]:
    """Score the published dB threshold on the same tiles.

    A model is only worth pitching if it beats the baseline, so the baseline is
    measured on identical data and printed next to it.
    """
    from . import fallback

    cm = np.zeros((3, 3), dtype=np.int64)
    for f in files:
        z = np.load(f)
        img = z["image"]
        vv = img[0] if img.ndim == 3 else img
        out = fallback.detect(vv)
        cm += confusion(out["mask"].ravel(), z["label"].ravel())
    return metrics_from_confusion(cm)


RESUME_NAME = "last_state.pt"


def _save_resume(path: Path, model, opt, sched, scaler, epoch: int, best: Dict[str, Any],
                 meta: Dict[str, Any]) -> None:
    """Full optimiser state, kept separate from the shipped checkpoint.

    `oil_unet_best.pt` must stay under 80 MB because it is the artefact the demo
    laptop carries. AdamW state roughly triples that, so it lives in its own
    file that only a resuming training session ever reads.
    """
    import torch

    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best": best,
        "meta": meta,
    }, str(path))


def _load_resume(path: Path, model, opt, sched, scaler, device: str) -> Dict[str, Any]:
    import torch

    blob = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    try:
        opt.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])
        if scaler is not None and blob.get("scaler"):
            scaler.load_state_dict(blob["scaler"])
    except Exception as exc:  # architecture or torch version drift
        print("   optimiser state could not be restored (%s); continuing with weights only" % exc)
    return {"epoch": int(blob.get("epoch", 0)), "best": blob.get("best") or {"iou_oil": -1.0}}


def train(
    tile_dir: Path,
    out_path: Path = None,
    arch: str = model_mod.DEFAULT_ARCH,
    encoder: str = model_mod.DEFAULT_ENCODER,
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.15,
    amp: bool = True,
    workers: int = 2,
    seed: int = None,
    max_train_tiles: Optional[int] = None,
    encoder_weights: Optional[str] = "imagenet",
    push_to_hub: bool = False,
    hub_repo: Optional[str] = None,
    resume: bool = False,
    time_budget_s: Optional[float] = None,
    max_water_false_oil: float = MAX_WATER_FALSE_OIL,
) -> Dict[str, Any]:
    """Train the segmenter.

    `push_to_hub` uploads the best checkpoint every time validation IoU_oil
    improves. `resume` pulls the last optimiser state back down first. Together
    they make a Kaggle session's hard time limit survivable: the session dies,
    the next one picks up where it left off. `time_budget_s` stops cleanly a
    little before that limit rather than being killed mid-epoch.

    `max_water_false_oil` is the ceiling an epoch must clear before it may
    become the shipped checkpoint. Tests that only exercise the plumbing on
    synthetic tiles pass 1.0 to disable it; real training must not.
    """
    import torch
    from torch.utils.data import DataLoader

    seed = int(config.RANDOM_SEED if seed is None else seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tile_dir = Path(tile_dir)
    out_path = Path(out_path or config.CHECKPOINT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path = out_path.parent / RESUME_NAME

    index_path = tile_dir / "index.json"
    norm = {"mean_db": -18.0, "std_db": 6.0}
    if index_path.exists():
        norm.update(json.loads(index_path.read_text(encoding="utf-8")).get("normalisation", {}))

    train_files, val_files = ds_mod.split_files(tile_dir, val_fraction, seed)
    if max_train_tiles:
        train_files = train_files[:max_train_tiles]
    if not train_files or not val_files:
        raise SystemExit("no tiles found in %s. Run scripts/prepare_tiles.py first." % tile_dir)

    print("tiles: %d train, %d val   normalisation %s" % (len(train_files), len(val_files), norm))

    train_ds = ds_mod.TileDataset(tile_dir, train_files, norm["mean_db"], norm["std_db"],
                                  augment=True, seed=seed)
    val_ds = ds_mod.TileDataset(tile_dir, val_files, norm["mean_db"], norm["std_db"])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=max(1, batch_size), shuffle=False,
                        num_workers=workers, pin_memory=True)

    device = model_mod.device_name()
    print("device: %s" % device)
    # encoder_weights=None skips the ImageNet fetch, which is what a smoke
    # test on a machine in airplane mode needs.
    model = model_mod.build(arch, encoder, encoder_weights=encoder_weights).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device == "cuda"))

    meta = {"arch": arch, "encoder": encoder, "classes": 3,
            "in_channels": model_mod.IN_CHANNELS, "tile": config.TILE,
            "normalisation": norm, "class_weights": CLASS_WEIGHTS, "seed": seed,
            "max_water_false_oil": max_water_false_oil,
            "encoder_weights": encoder_weights}

    best = {"iou_oil": -1.0}
    start_epoch = 1
    history: List[Dict[str, Any]] = []

    if resume:
        restored = _restore(resume_path, model, opt, sched, scaler, device,
                            push_to_hub, hub_repo)
        if restored:
            start_epoch = restored["epoch"] + 1
            best = restored["best"]
            print("resumed at epoch %d, best IoU_oil so far %.4f"
                  % (start_epoch, best.get("iou_oil", -1.0)))
            if start_epoch > epochs:
                print("nothing left to do: already trained %d of %d epochs"
                      % (start_epoch - 1, epochs))

    wall0 = time.time()
    stopped_early = False

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(amp and device == "cuda")):
                loss = combo_loss(model(x), y, CLASS_WEIGHTS)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
        sched.step()

        m = evaluate(model, val_dl, device)
        m["train_loss"] = total / max(1, len(train_dl))
        m["epoch"] = epoch
        m["seconds"] = round(time.time() - t0, 1)
        history.append(m)
        print("epoch %3d  loss %.4f  IoU oil %.4f  look-alike %.4f  sea %.4f  "
              "acc %.4f  water->oil %.4f  %.0fs"
              % (epoch, m["train_loss"], m["iou_oil"], m["iou_lookalike"],
                 m["iou_sea"], m["pixel_accuracy"], m["water_false_oil"],
                 m["seconds"]))

        # Selection is on IoU oil *subject to* the model not calling open water
        # oil. Ranking on IoU oil alone is what let the first checkpoint through:
        # it is monotone in "predict more oil", so the epoch it picked was close
        # to the most over-predicting one available.
        usable = m["water_false_oil"] <= max_water_false_oil
        improved = usable and m["iou_oil"] > best["iou_oil"]
        if improved:
            best = dict(m)
            model_mod.save_checkpoint(model, out_path, meta=meta, metrics=best)
            print("   saved new best to %s" % out_path)

        # Always refresh the resume artefact, improvement or not, so a session
        # killed at the time limit does not repeat an epoch it already paid for.
        _save_resume(resume_path, model, opt, sched, scaler, epoch, best, meta)

        if push_to_hub and improved:
            _push(out_path, resume_path, epoch, best, hub_repo)

        # `is not None`, not truthiness: a budget of 0 is a real instruction
        # to stop after the first epoch, and `if 0:` silently ignores it.
        if time_budget_s is not None and (time.time() - wall0) >= float(time_budget_s):
            print("\nstopping at epoch %d: time budget of %.0f s reached. "
                  "Re-run with --resume to continue." % (epoch, time_budget_s))
            stopped_early = True
            if push_to_hub:
                _push(out_path, resume_path, epoch, best, hub_repo)
            break

    # A class with no pixels anywhere in validation has a NaN IoU, and a bare
    # NaN in a metrics table invites the reader to assume a bug. Say why.
    val_pixels = np.zeros(3, dtype=np.int64)
    for f in val_files:
        val_pixels += np.bincount(np.load(f)["label"].ravel(), minlength=3)
    unsupervised = [i for i in range(3) if val_pixels[i] == 0]
    if unsupervised:
        names = {0: "sea", 1: "look-alike", 2: "mineral oil"}
        for i in unsupervised:
            print("\nNOTE: class %d (%s) has no pixels in this dataset, so its IoU"
                  % (i, names[i]))
            print("      is not defined. The Zenodo ground truth segments oil only;")
            print("      a look-alike chip's mask is empty, so those chips are used")
            print("      as hard negatives rather than given invented labels.")

    if best.get("iou_oil", -1.0) < 0:
        # Every epoch exceeded MAX_WATER_FALSE_OIL, so nothing was written.
        # Say so, instead of leaving the caller to discover an absent
        # checkpoint, and print the closest miss to steer the next run.
        floor = min(history, key=lambda h: h["water_false_oil"]) if history else None
        print("\nNo epoch met the water->oil ceiling of %.3f." % max_water_false_oil)
        if floor is not None:
            print("   closest was epoch %d at %.4f (IoU oil %.4f)."
                  % (floor["epoch"], floor["water_false_oil"], floor["iou_oil"]))
        print("   No checkpoint was saved. The training mix is still too oil")
        print("   rich; raise the sea budget in prepare-data and re-run.")

    print("\nbaseline on the same validation tiles:")
    base = baseline_metrics(val_files, norm["mean_db"], norm["std_db"])
    print("   threshold baseline  IoU oil %.4f  look-alike %.4f  acc %.4f  water->oil %.4f"
          % (base["iou_oil"], base["iou_lookalike"], base["pixel_accuracy"],
             base["water_false_oil"]))
    if best.get("iou_oil", -1.0) >= 0:
        print("   trained model       IoU oil %.4f  look-alike %.4f  acc %.4f  water->oil %.4f"
              % (best["iou_oil"], best["iou_lookalike"], best["pixel_accuracy"],
                 best["water_false_oil"]))
    else:
        print("   trained model       no eligible epoch; see the note above")

    report = {
        "best": best,
        "baseline": base,
        "delta_iou_oil": round(best.get("iou_oil", 0.0) - base.get("iou_oil", 0.0), 4),
        "history": history,
        "checkpoint": str(out_path),
        "tiles": len(train_files) + len(val_files),
        "epochs_completed": (history[-1]["epoch"] if history else start_epoch - 1),
        "epochs_requested": epochs,
        "stopped_early": stopped_early,
        "arch": arch,
        "encoder": encoder,
        "max_water_false_oil": max_water_false_oil,
        "normalisation": norm,
        "validation_pixels_per_class": [int(v) for v in val_pixels],
        "unsupervised_classes": unsupervised,
    }
    report_path = Path(out_path).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if push_to_hub:
        _push(out_path, resume_path, report["epochs_completed"], best, hub_repo, final=True)
    return report


def _push(out_path: Path, resume_path: Path, epoch: int, best: Dict[str, Any],
          hub_repo: Optional[str], final: bool = False) -> None:
    """Upload the checkpoint. A Hub failure must never kill a training run."""
    from .. import hub

    try:
        state = {
            "epoch": int(epoch),
            "iou_oil": float(best.get("iou_oil", float("nan"))),
            "iou_lookalike": float(best.get("iou_lookalike", float("nan"))),
            "pixel_accuracy": float(best.get("pixel_accuracy", float("nan"))),
            "final": bool(final),
        }
        kw = {"repo_id": hub_repo} if hub_repo else {}
        hub.push_checkpoint(out_path, report=Path(out_path).with_suffix(".report.json"),
                            state=state, **kw)
        # The resume artefact carries optimiser state and is not the demo
        # artefact, so it goes up under its own name.
        if resume_path.exists():
            api = hub.api()
            api.upload_file(path_or_fileobj=str(resume_path),
                            path_in_repo=hub.RESUME_IN_REPO,
                            repo_id=hub_repo or hub.MODEL_REPO,
                            repo_type="model",
                            commit_message="resume state, epoch %d" % epoch)
        print("   pushed to %s" % (hub_repo or hub.MODEL_REPO))
    except Exception as exc:
        print("   Hub push failed (%s). Training continues; the local checkpoint is intact."
              % exc)


def _restore(resume_path: Path, model, opt, sched, scaler, device: str,
             from_hub: bool, hub_repo: Optional[str]) -> Optional[Dict[str, Any]]:
    """Restore from disk, else from the Hub. Returns None if there is nothing."""
    if not resume_path.exists() and from_hub:
        from .. import hub

        try:
            print("no local resume state; trying the Hub")
            hub.pull_checkpoint(dest=resume_path, repo_id=hub_repo or hub.MODEL_REPO,
                                filename=hub.RESUME_IN_REPO)
        except Exception as exc:
            print("   nothing to resume from on the Hub (%s)" % exc)
    if not resume_path.exists():
        return None
    return _load_resume(resume_path, model, opt, sched, scaler, device)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", default=str(config.CHECKPOINT))
    ap.add_argument("--arch", default=model_mod.DEFAULT_ARCH,
                    choices=["Unet", "UnetPlusPlus"])
    ap.add_argument("--encoder", default=model_mod.DEFAULT_ENCODER)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--max-train-tiles", type=int, default=None)
    ap.add_argument("--push-to-hub", action="store_true",
                    help="upload the checkpoint every time validation IoU improves")
    ap.add_argument("--hub-repo", default=None,
                    help="model repo id, default N-1ACE/tidetrace-oil-unet")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last optimiser state, local or from the Hub")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="stop cleanly after this many seconds, before Kaggle kills the session")
    args = ap.parse_args(argv)

    if not (model_mod.torch_available() and model_mod.smp_available()):
        raise SystemExit("training needs torch and segmentation_models_pytorch. "
                         "Install them on Kaggle, not on the demo laptop.")

    train(Path(args.tiles), Path(args.out), arch=args.arch, encoder=args.encoder,
          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          workers=args.workers, amp=not args.no_amp,
          max_train_tiles=args.max_train_tiles,
          push_to_hub=args.push_to_hub, hub_repo=args.hub_repo,
          resume=args.resume, time_budget_s=args.time_budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
