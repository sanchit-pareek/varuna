"""Push a TideTrace notebook to Kaggle and watch it run.

Kaggle is where the GPU is. This wraps `kaggle kernels push/status/output` so
the two notebooks under `kaggle/` can be shipped and monitored without leaving
the terminal.

Authentication uses the current Kaggle scheme, in this order:

    KAGGLE_API_TOKEN environment variable
    ~/.kaggle/access_token          (the KGAT_... token from the settings page)
    ~/.kaggle/kaggle.json           (legacy username plus key)

Before the first run, add your Hugging Face token to the Kaggle notebook as a
Secret labelled `HF_TOKEN`. Kaggle Secrets are per-notebook and set in the
editor under Add-ons > Secrets. The notebooks read it from there, so no token
is ever written into a file in this repository.

Usage:
    python scripts/kaggle_push.py prepare
    python scripts/kaggle_push.py train --accelerator NvidiaTeslaT4
    python scripts/kaggle_push.py status train
    python scripts/kaggle_push.py output train --dir models
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "kaggle"

KERNELS = {
    "prepare": "tidetrace-prepare-data",
    "train": "tidetrace-train",
}

# Accelerator ids the Kaggle CLI accepts, as of 2026.
ACCELERATORS = [
    # T4 is the default for training and P100 is effectively unusable, however
    # tempting its 16 GB looks. Kaggle ships torch 2.10+cu128, which compiles
    # sm_70 through sm_120. The P100 is Pascal, sm_60, so every forward pass
    # dies with "no kernel image is available for execution on the device"
    # after the data has already been staged.
    "NvidiaTeslaP100", "NvidiaTeslaT4", "NvidiaTeslaT4Highmem", "NvidiaTeslaA100",
    "NvidiaL4", "NvidiaL4X1", "NvidiaH100", "NvidiaRtxPro6000",
    "TpuV38", "Tpu1VmV38", "TpuV5E8", "TpuV6E8",
]


def kaggle_bin() -> str:
    """Prefer the CLI next to the interpreter running this script."""
    candidate = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    return str(candidate) if candidate.exists() else "kaggle"


def run(args, check: bool = True) -> subprocess.CompletedProcess:
    print("$ %s" % " ".join(args), flush=True)
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print(proc.stderr.rstrip(), file=sys.stderr)
    if check and proc.returncode != 0:
        raise SystemExit("kaggle command failed with exit code %d" % proc.returncode)
    return proc


def folder_for(name: str) -> Path:
    slug = KERNELS.get(name, name)
    folder = KAGGLE_DIR / slug
    if not folder.exists():
        raise SystemExit(
            "no kernel folder at %s. Run scripts/build_kaggle_kernels.py first." % folder)
    return folder


def slug_for(name: str) -> str:
    meta = json.loads((folder_for(name) / "kernel-metadata.json").read_text(encoding="utf-8"))
    return meta["id"]


def cmd_push(args) -> int:
    folder = folder_for(args.kernel)
    meta_path = folder / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if args.public:
        meta["is_private"] = False
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cmd = [kaggle_bin(), "kernels", "push", "-p", str(folder)]
    if args.accelerator:
        cmd += ["--accelerator", args.accelerator]
    if args.timeout:
        cmd += ["-t", str(args.timeout)]
    run(cmd)

    print("\nkernel: https://www.kaggle.com/code/%s" % meta["id"])
    print("Reminder: the notebook needs a Kaggle Secret named HF_TOKEN.")
    print("Set it in the editor under Add-ons > Secrets, then re-run there.")
    if args.watch:
        return watch(meta["id"], args.poll)
    print("\nWatch it with:  python scripts/kaggle_push.py status %s" % args.kernel)
    return 0


def watch(slug: str, poll: int = 60, limit_minutes: int = 720) -> int:
    """Poll until the run reaches a terminal state. Every state is reported."""
    deadline = time.time() + limit_minutes * 60
    last = None
    while time.time() < deadline:
        proc = run([kaggle_bin(), "kernels", "status", slug], check=False)
        text = (proc.stdout or "") + (proc.stderr or "")
        state = "unknown"
        for candidate in ("complete", "error", "cancelAcknowledged", "cancelRequested",
                          "running", "queued"):
            if candidate.lower() in text.lower():
                state = candidate
                break
        if state != last:
            print("[%s] %s" % (time.strftime("%H:%M:%S"), state), flush=True)
            last = state
        if state in ("complete", "error", "cancelAcknowledged"):
            return 0 if state == "complete" else 1
        time.sleep(poll)
    print("stopped watching after %d minutes; the kernel may still be running" % limit_minutes)
    return 0


def cmd_status(args) -> int:
    run([kaggle_bin(), "kernels", "status", slug_for(args.kernel)], check=False)
    return 0


def cmd_watch(args) -> int:
    return watch(slug_for(args.kernel), args.poll)


def cmd_output(args) -> int:
    dest = Path(args.dir or (ROOT / "models"))
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle_bin(), "kernels", "output", slug_for(args.kernel), "-p", str(dest)]
    if args.pattern:
        cmd += ["--file-pattern", args.pattern]
    run(cmd, check=False)
    print("\nFiles in %s:" % dest)
    for f in sorted(dest.iterdir()):
        print("  %-44s %8.1f MB" % (f.name, f.stat().st_size / 1e6))
    print("\nThe checkpoint is also on the Hub. Prefer:")
    print("    python scripts/hf_sync.py pull-model")
    return 0


def cmd_list(args) -> int:
    run([kaggle_bin(), "kernels", "list", "--mine"], check=False)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("push", help="push and optionally watch")
    p.add_argument("kernel", choices=sorted(KERNELS) + list(KERNELS.values()))
    p.add_argument("--accelerator", choices=ACCELERATORS, default=None)
    p.add_argument("--timeout", type=int, default=None, help="run time cap in seconds")
    p.add_argument("--public", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll", type=int, default=60)
    p.set_defaults(fn=cmd_push)

    for name in ("prepare", "train"):
        q = sub.add_parser(name, help="shorthand for: push %s" % name)
        q.add_argument("--accelerator", choices=ACCELERATORS,
                       default="NvidiaTeslaT4" if name == "train" else None)
        q.add_argument("--timeout", type=int, default=None)
        q.add_argument("--public", action="store_true")
        q.add_argument("--watch", action="store_true")
        q.add_argument("--poll", type=int, default=60)
        q.set_defaults(fn=cmd_push, kernel=name)

    q = sub.add_parser("status")
    q.add_argument("kernel")
    q.set_defaults(fn=cmd_status)

    q = sub.add_parser("watch")
    q.add_argument("kernel")
    q.add_argument("--poll", type=int, default=60)
    q.set_defaults(fn=cmd_watch)

    q = sub.add_parser("output")
    q.add_argument("kernel")
    q.add_argument("--dir", default=None)
    q.add_argument("--pattern", default=None)
    q.set_defaults(fn=cmd_output)

    sub.add_parser("list").set_defaults(fn=cmd_list)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
