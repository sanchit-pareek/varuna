"""List a remote 7z archive's contents without downloading it.

This exists because four separate data bugs were diagnosed by guessing at a
9.86 GB archive's internal layout, one 35 minute download at a time. The fourth
turned out to be that Zenodo Part III names its labels `00000_segmentation.tif`
while its images are `00000.tif`, which no amount of reasoning was going to
reveal.

A 7z file keeps its header at the end and a pointer to it at the start, so the
whole file list can be read with three HTTP range requests and about a megabyte
of traffic. Check the layout first; download second.

Usage:
    python scripts/inspect_archive.py --record 13761290 \\
        --file 02_Test_images_and_ground_truth.7z
    python scripts/inspect_archive.py --url https://example.org/thing.7z
    python scripts/inspect_archive.py --record 8346860 --list-files
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from collections import defaultdict
from pathlib import PurePosixPath, PureWindowsPath

UA = {"User-Agent": "TideTrace/1.0 (SIH26143 research)"}


class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only file backed by HTTP range requests."""

    def __init__(self, url: str, timeout: int = 90, retries: int = 3):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.pos = 0
        self.requests = 0
        self.fetched = 0
        req = urllib.request.Request(url, headers=dict(UA, Range="bytes=0-0"))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content_range = r.headers.get("Content-Range")
            if not content_range:
                raise SystemExit("the server does not support range requests")
            self.size = int(content_range.split("/")[-1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(
            self.url, headers=dict(UA, Range="bytes=%d-%d" % (self.pos, self.pos + n - 1)))
        last = None
        for _ in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                break
            except Exception as exc:
                last = exc
        else:
            raise SystemExit("range request failed: %s" % last)
        self.pos += len(data)
        self.requests += 1
        self.fetched += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def zenodo_url(record: str, filename: str) -> str:
    return "https://zenodo.org/records/%s/files/%s?download=1" % (record, filename)


def parent_of(name: str) -> str:
    """Archive members may carry either separator; handle both."""
    if "\\" in name and "/" not in name:
        return str(PureWindowsPath(name).parent).replace("\\", "/")
    return str(PurePosixPath(name).parent)


def leaf_of(name: str) -> str:
    if "\\" in name and "/" not in name:
        return PureWindowsPath(name).name
    return PurePosixPath(name).name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--record", help="Zenodo record id")
    ap.add_argument("--file", help="file name inside the Zenodo record")
    ap.add_argument("--list-files", action="store_true", help="print every member")
    ap.add_argument("--samples", type=int, default=3, help="example names per folder")
    args = ap.parse_args()

    if args.url:
        url = args.url
    elif args.record and args.file:
        url = zenodo_url(args.record, args.file)
    else:
        ap.error("pass --url, or --record together with --file")

    try:
        import py7zr
    except ImportError as exc:
        raise SystemExit("this needs py7zr:  pip install py7zr") from exc

    print(url)
    fh = HttpRangeFile(url)
    print("archive size : %.2f GB" % (fh.size / 1e9))

    z = py7zr.SevenZipFile(io.BufferedReader(fh, buffer_size=1 << 20), mode="r")
    names = z.getnames()
    print("fetched      : %.2f MB in %d range requests" % (fh.fetched / 1e6, fh.requests))
    print("members      : %d\n" % len(names))

    grouped = defaultdict(list)
    for n in names:
        grouped[parent_of(n)].append(n)

    print("%-28s %6s  %s" % ("folder", "files", "example names"))
    print("-" * 78)
    for d in sorted(grouped):
        members = sorted(grouped[d])
        examples = [leaf_of(m) for m in members[: args.samples]]
        print("%-28s %6d  %s" % (d[:28], len(members), ", ".join(examples)))

    if args.list_files:
        print()
        for n in sorted(names):
            print(" ", n)

    print("\nRead this before committing to the download. The image and label "
          "names rarely match exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
