"""Vendor Leaflet locally so the judged demo needs no CDN.

Run once while online. After this, app/static/vendor/leaflet holds everything
the console needs and the browser makes no external request.

Usage:
    python scripts/fetch_vendor.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config   # noqa: E402

VERSION = "1.9.4"
BASE = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/%s" % VERSION
FILES = [
    ("leaflet.js", "leaflet.js"),
    ("leaflet.css", "leaflet.css"),
    ("images/marker-icon.png", "images/marker-icon.png"),
    ("images/marker-icon-2x.png", "images/marker-icon-2x.png"),
    ("images/marker-shadow.png", "images/marker-shadow.png"),
    ("images/layers.png", "images/layers.png"),
    ("images/layers-2x.png", "images/layers-2x.png"),
]


def main() -> int:
    out = Path(config.STATIC_DIR) / "vendor" / "leaflet"
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for remote, local in FILES:
        url = "%s/%s" % (BASE, remote)
        dest = out / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            dest.write_bytes(data)
            print("  %-28s %6.1f KB" % (local, len(data) / 1024.0))
            ok += 1
        except Exception as exc:
            print("  %-28s FAILED: %s" % (local, exc))
    print("\nvendored %d of %d files into %s" % (ok, len(FILES), out))
    if ok < 2:
        print("Leaflet is missing. The console falls back to the CDN, which means "
              "the demo would need internet. Fix this before judging.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
