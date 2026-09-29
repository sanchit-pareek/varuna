"""Hugging Face Hub sync for datasets and checkpoints.

Why this exists: the demo laptop should not hold the training corpus. Zenodo's
image archives are 10 to 40 GB, and a 512 tile set built from them is several
gigabytes more. None of that belongs on the machine that runs the judged demo,
which only needs one checkpoint under 80 MB.

So the split is:

    Kaggle          downloads Zenodo, tiles it, trains, and pushes results
    Hugging Face    stores the tiles (dataset repo) and the checkpoints (model repo)
    the laptop      pulls exactly one checkpoint file, and nothing else

IMPORTANT, and enforced by a test: nothing on the request path may import this
module. `app/main.py`, `app/pipeline.py` and everything under `app/api/` stay
free of it, because the judged demo runs in airplane mode. This module is for
`app/ml/train.py` (which runs on Kaggle) and for `scripts/hf_sync.py` (which the
operator runs deliberately, while online).

Checkpointing strategy. Kaggle sessions are killed at 9 or 12 hours, and a
session that dies takes its disk with it. So training pushes to the Hub every
time validation IoU improves, and can resume from the Hub on the next session.
That turns a hard time limit into a soft one.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from . import config

DEFAULT_USER = os.environ.get("TIDETRACE_HF_USER", "N-1ACE")
DATASET_REPO = os.environ.get("TIDETRACE_HF_DATASET", "%s/tidetrace-sar-tiles" % DEFAULT_USER)
MODEL_REPO = os.environ.get("TIDETRACE_HF_MODEL", "%s/tidetrace-oil-unet" % DEFAULT_USER)

CHECKPOINT_NAME = "oil_unet_best.pt"
REPORT_NAME = "oil_unet_best.report.json"
STATE_NAME = "training_state.json"
RESUME_IN_REPO = "last_state.pt"
TILE_INDEX = "index.json"


class HubError(RuntimeError):
    """Raised when the Hub is needed but unusable. Never raised at request time."""


def available() -> bool:
    try:
        import huggingface_hub  # noqa: F401

        return True
    except Exception:
        return False


def _require():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except Exception as exc:
        raise HubError(
            "huggingface_hub is not installed. It is deliberately not a runtime "
            "dependency of the demo. Install it only where you sync: "
            "pip install huggingface_hub"
        ) from exc


def token(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve a token without ever putting one in the repository.

    Order: explicit argument, HF_TOKEN, HUGGINGFACE_HUB_TOKEN, then whatever
    `hf auth login` stored in the user's home cache.
    """
    if explicit:
        return explicit
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def api(tok: Optional[str] = None):
    _require()
    from huggingface_hub import HfApi

    return HfApi(token=token(tok))


def whoami(tok: Optional[str] = None) -> Dict[str, Any]:
    return api(tok).whoami()


def ensure_repo(repo_id: str, repo_type: str, private: bool = False,
                tok: Optional[str] = None) -> str:
    """Create the repo if it does not exist. Idempotent."""
    a = api(tok)
    a.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    return repo_id


# ---------------------------------------------------------------------------
# Checkpoints (model repo)
# ---------------------------------------------------------------------------

def push_checkpoint(
    checkpoint: Path,
    report: Optional[Path] = None,
    state: Optional[Dict[str, Any]] = None,
    repo_id: str = MODEL_REPO,
    revision: Optional[str] = None,
    tok: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> str:
    """Upload a checkpoint plus its metrics, in one commit.

    Called from the training loop every time validation IoU improves, so a
    Kaggle session that is killed at the time limit still leaves the best model
    on the Hub rather than in a deleted container.
    """
    from huggingface_hub import CommitOperationAdd

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise HubError("no checkpoint at %s" % checkpoint)

    ensure_repo(repo_id, "model", tok=tok)
    ops = [CommitOperationAdd(path_in_repo=CHECKPOINT_NAME, path_or_fileobj=str(checkpoint))]
    if report and Path(report).exists():
        ops.append(CommitOperationAdd(path_in_repo=REPORT_NAME, path_or_fileobj=str(report)))
    if state is not None:
        ops.append(CommitOperationAdd(
            path_in_repo=STATE_NAME,
            path_or_fileobj=json.dumps(state, indent=2).encode("utf-8"),
        ))

    msg = commit_message or "checkpoint"
    if state and "epoch" in state:
        msg = "epoch %s, IoU_oil %.4f" % (state.get("epoch"), state.get("iou_oil", float("nan")))

    a = api(tok)
    a.create_commit(repo_id=repo_id, repo_type="model", operations=ops,
                    commit_message=msg, revision=revision)
    return "https://huggingface.co/%s" % repo_id


def pull_checkpoint(
    dest: Optional[Path] = None,
    repo_id: str = MODEL_REPO,
    filename: str = CHECKPOINT_NAME,
    tok: Optional[str] = None,
    revision: Optional[str] = None,
) -> Path:
    """Download the current best checkpoint to `dest` (default models/)."""
    _require()
    from huggingface_hub import hf_hub_download

    dest = Path(dest or config.CHECKPOINT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model",
                             revision=revision, token=token(tok))
    data = Path(cached).read_bytes()
    dest.write_bytes(data)
    return dest


def remote_state(repo_id: str = MODEL_REPO, tok: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Read `training_state.json` from the model repo, for resuming a run."""
    _require()
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        p = hf_hub_download(repo_id=repo_id, filename=STATE_NAME, repo_type="model",
                            token=token(tok))
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tiles (dataset repo)
# ---------------------------------------------------------------------------

def push_tiles(
    folder: Path,
    repo_id: str = DATASET_REPO,
    path_in_repo: str = "tiles",
    tok: Optional[str] = None,
    commit_message: str = "tiles",
) -> str:
    """Upload a tile folder. Resumable: re-run and finished files are skipped.

    `upload_folder` streams in adaptive batches and deduplicates chunks, so an
    interrupted upload is fixed by running the same command again. The legacy
    `upload_large_folder` is deprecated and deliberately not used.
    """
    folder = Path(folder)
    if not folder.exists():
        raise HubError("no tile folder at %s" % folder)
    ensure_repo(repo_id, "dataset", tok=tok)
    a = api(tok)
    a.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        ignore_patterns=["*.tmp", "**/.ipynb_checkpoints/*", "**/__pycache__/*"],
    )
    return "https://huggingface.co/datasets/%s" % repo_id


def pull_tiles(
    dest: Path,
    repo_id: str = DATASET_REPO,
    path_in_repo: str = "tiles",
    tok: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
) -> Path:
    """Download the tile set. `max_files` keeps a sanity run small and cheap."""
    _require()
    from huggingface_hub import hf_hub_download, snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if max_files:
        a = api(tok)
        names = [f for f in a.list_repo_files(repo_id, repo_type="dataset")
                 if f.startswith(path_in_repo + "/")]
        npz = sorted(n for n in names if n.endswith(".npz"))[:max_files]
        wanted = npz + [n for n in names if n.endswith(TILE_INDEX)]
        for name in wanted:
            local = hf_hub_download(repo_id=repo_id, filename=name, repo_type="dataset",
                                    token=token(tok))
            out = dest / Path(name).name
            out.write_bytes(Path(local).read_bytes())
        return dest

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=list(allow_patterns) if allow_patterns else ["%s/**" % path_in_repo],
        token=token(tok),
    )
    inner = dest / path_in_repo
    return inner if inner.exists() else dest


def push_code(
    root: Path,
    repo_id: str = MODEL_REPO,
    tok: Optional[str] = None,
) -> str:
    """Snapshot the `app` package into the model repo, under `code/`.

    Weights without the code that produced them are not reproducible, and the
    Kaggle notebook needs the package anyway to import the training entry point.
    """
    root = Path(root)
    ensure_repo(repo_id, "model", tok=tok)
    a = api(tok)
    a.upload_folder(
        folder_path=str(root / "app"),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="code/app",
        commit_message="source snapshot",
        ignore_patterns=["**/__pycache__/*", "*.pyc", "static/vendor/**"],
    )
    return "https://huggingface.co/%s/tree/main/code" % repo_id


def pull_code(dest: Path, repo_id: str = MODEL_REPO, tok: Optional[str] = None) -> Path:
    """Fetch the source snapshot, used by the Kaggle notebooks."""
    _require()
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(dest),
                      allow_patterns=["code/**"], token=token(tok))
    return dest / "code"


@dataclass
class HubStatus:
    installed: bool
    authenticated: bool
    user: Optional[str]
    dataset_repo: str
    model_repo: str
    checkpoint_on_hub: bool
    tiles_on_hub: int
    local_checkpoint: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def status(tok: Optional[str] = None) -> HubStatus:
    """What the sync script prints. Never raises."""
    local = Path(config.CHECKPOINT).exists()
    if not available():
        return HubStatus(False, False, None, DATASET_REPO, MODEL_REPO, False, 0, local,
                         "huggingface_hub not installed")
    try:
        a = api(tok)
        user = a.whoami().get("name")
    except Exception as exc:
        return HubStatus(True, False, None, DATASET_REPO, MODEL_REPO, False, 0, local, str(exc))

    ckpt, tiles = False, 0
    try:
        ckpt = CHECKPOINT_NAME in a.list_repo_files(MODEL_REPO, repo_type="model")
    except Exception:
        pass
    try:
        tiles = sum(1 for f in a.list_repo_files(DATASET_REPO, repo_type="dataset")
                    if f.endswith(".npz"))
    except Exception:
        pass
    return HubStatus(True, True, user, DATASET_REPO, MODEL_REPO, ckpt, tiles, local)
