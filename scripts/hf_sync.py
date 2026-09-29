"""Move data between the laptop and the Hugging Face Hub, deliberately.

The demo laptop holds one checkpoint and a few scene chips. Everything else,
the training tiles and the training history, lives on the Hub. This script is
the only thing that moves them, and it is never called at request time.

Commands:

    status                what exists locally and on the Hub
    init                  create both repos and write their README cards
    pull-model            fetch oil_unet_best.pt into models/
    push-model            upload a locally trained checkpoint
    push-code             snapshot app/ into the model repo, for the notebooks
    push-tiles  --dir D   upload a locally prepared tile folder
    pull-tiles  --dir D   download tiles (use --max-files for a sanity subset)

Authentication, in order of precedence:

    --token argument
    HF_TOKEN environment variable
    whatever `hf auth login` stored in your home cache

Never put a token in this repository. `.gitignore` covers the usual accidents,
but the real protection is not typing it into a file in the first place.

Usage:
    python scripts/hf_sync.py status
    python scripts/hf_sync.py init
    python scripts/hf_sync.py pull-model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config, hub   # noqa: E402

DATASET_CARD = """---
license: cc-by-4.0
task_categories:
- image-segmentation
tags:
- sar
- sentinel-1
- oil-spill
- remote-sensing
- maritime
---

# TideTrace SAR training tiles

512 x 512 tiles cut from the Zenodo Sentinel-1 SAR oil spill dataset, prepared
for the TideTrace oil spill attribution system (SIH26143, NTRO).

## Contents

Each `.npz` holds one tile:

| key | shape | meaning |
| --- | --- | --- |
| `image` | (2, 512, 512) float32 | Sigma0 VV and VH in dB |
| `label` | (512, 512) uint8 | 0 sea, 1 look-alike, 2 mineral oil |

`index.json` records the tile count per class and the dB normalisation
statistics (`mean_db`, `std_db`) that the model was trained with. Inference
reads those from the checkpoint, so they must not drift.

## Provenance

Source imagery: Zenodo Sentinel-1 SAR Oil Spill Dataset, Parts I, II and III
(`10.5281/zenodo.8346860`, `10.5281/zenodo.8253899`, `10.5281/zenodo.13761290`),
licensed CC BY 4.0.

Label convention comes from the source folder, not from the mask value: an oil
folder mask becomes class 2, a look-alike folder mask becomes class 1, and an
oil-free chip is all class 0.

Masks in the source set frequently ship with no CRS. Every mask here had the
affine transform and CRS copied from its matching Sigma0 image before tiling,
because a centroid computed from an ungeoreferenced mask is fiction.

## Licence

CC BY 4.0, inherited from the source dataset. Cite Zenodo when you use these.
"""

MODEL_CARD = """---
license: mit
tags:
- image-segmentation
- sar
- sentinel-1
- oil-spill
- unet
library_name: pytorch
---

# TideTrace oil slick segmenter

Three class semantic segmentation of Sentinel-1 SAR: sea, look-alike, mineral
oil. Trained for TideTrace, the SIH26143 oil spill attribution console.

## Files

| file | what it is |
| --- | --- |
| `oil_unet_best.pt` | the shipped checkpoint, weights and metadata, kept under 80 MB |
| `oil_unet_best.report.json` | metrics, and the dB threshold baseline on the same tiles |
| `last_state.pt` | optimiser state for resuming a Kaggle session, not needed for inference |
| `training_state.json` | epoch and best metrics, small enough to poll |
| `code/app/` | the source snapshot that produced the weights |

## Architecture

- UnetPlusPlus, encoder `timm-efficientnet-b0`, ImageNet initialised
- 3 classes, 512 tiles, VV and VH in dB with VV repeated to a third channel
- Weighted cross entropy plus soft Dice, oil class weighted 2.5
- AdamW 1e-4, cosine schedule, AMP

The checkpoint carries its own architecture and normalisation statistics, so
inference never guesses what it was trained on.

## Honest evaluation

The report file always prints the trained model next to the published -22 dB
dark patch baseline, measured on identical validation tiles. A model that does
not beat that baseline is not worth shipping, and the comparison is there so
anyone can check rather than take it on trust.

## Intended use

Detecting candidate oil slicks in SAR for investigation. Output is a ranked
likelihood, not proof that any vessel discharged anything.

## Licence

MIT for the weights and code. Training data is CC BY 4.0 from Zenodo; cite it.
"""


def cmd_status(args) -> int:
    st = hub.status(args.token)
    print("huggingface_hub installed : %s" % st.installed)
    print("authenticated             : %s%s" % (st.authenticated,
                                                (" as %s" % st.user) if st.user else ""))
    print("dataset repo              : %s  (%d tile files)" % (st.dataset_repo, st.tiles_on_hub))
    print("model repo                : %s  (checkpoint present: %s)"
          % (st.model_repo, st.checkpoint_on_hub))
    print("local checkpoint          : %s  (%s)"
          % (st.local_checkpoint, config.CHECKPOINT))
    if st.error:
        print("error                     : %s" % st.error)
    state = None
    if st.authenticated:
        state = hub.remote_state(tok=args.token)
    if state:
        print("last training state       : epoch %s, IoU_oil %.4f%s"
              % (state.get("epoch"), state.get("iou_oil", float("nan")),
                 ", final" if state.get("final") else ""))
    return 0


def cmd_init(args) -> int:
    hub.ensure_repo(hub.DATASET_REPO, "dataset", private=args.private, tok=args.token)
    hub.ensure_repo(hub.MODEL_REPO, "model", private=args.private, tok=args.token)
    a = hub.api(args.token)
    a.upload_file(path_or_fileobj=DATASET_CARD.encode("utf-8"), path_in_repo="README.md",
                  repo_id=hub.DATASET_REPO, repo_type="dataset",
                  commit_message="dataset card")
    a.upload_file(path_or_fileobj=MODEL_CARD.encode("utf-8"), path_in_repo="README.md",
                  repo_id=hub.MODEL_REPO, repo_type="model", commit_message="model card")
    print("dataset: https://huggingface.co/datasets/%s" % hub.DATASET_REPO)
    print("model  : https://huggingface.co/%s" % hub.MODEL_REPO)
    return 0


def cmd_pull_model(args) -> int:
    dest = hub.pull_checkpoint(Path(args.out) if args.out else None, tok=args.token)
    mb = dest.stat().st_size / 1e6
    print("downloaded %s (%.1f MB)" % (dest, mb))
    print("Restart the app; the top bar switches from dB BASELINE to U-NET LOADED.")
    return 0


def cmd_push_model(args) -> int:
    ck = Path(args.checkpoint or config.CHECKPOINT)
    if not ck.exists():
        raise SystemExit("no checkpoint at %s" % ck)
    url = hub.push_checkpoint(ck, report=ck.with_suffix(".report.json"), tok=args.token)
    print("pushed to %s" % url)
    return 0


def cmd_push_code(args) -> int:
    print("pushed to %s" % hub.push_code(ROOT, tok=args.token))
    return 0


def cmd_push_tiles(args) -> int:
    folder = Path(args.dir)
    n = len(list(folder.glob("*.npz")))
    print("uploading %d tiles from %s" % (n, folder))
    print("upload_folder is resumable: if it stops, run the same command again")
    print("pushed to %s" % hub.push_tiles(folder, tok=args.token))
    return 0


def cmd_pull_tiles(args) -> int:
    out = hub.pull_tiles(Path(args.dir), tok=args.token, max_files=args.max_files)
    print("downloaded %d tiles to %s" % (len(list(Path(out).glob("*.npz"))), out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=None, help="HF token; prefer HF_TOKEN in the environment")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    p = sub.add_parser("init")
    p.add_argument("--private", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("pull-model")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_pull_model)

    p = sub.add_parser("push-model")
    p.add_argument("--checkpoint", default=None)
    p.set_defaults(fn=cmd_push_model)

    sub.add_parser("push-code").set_defaults(fn=cmd_push_code)

    p = sub.add_parser("push-tiles")
    p.add_argument("--dir", required=True)
    p.set_defaults(fn=cmd_push_tiles)

    p = sub.add_parser("pull-tiles")
    p.add_argument("--dir", required=True)
    p.add_argument("--max-files", type=int, default=None)
    p.set_defaults(fn=cmd_pull_tiles)

    args = ap.parse_args()
    try:
        return args.fn(args)
    except hub.HubError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
