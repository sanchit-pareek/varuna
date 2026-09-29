"""Minimal baseline TIFF / GeoTIFF reader.

This exists only so that rasterio stays a genuinely optional dependency, which
the spec demands. It handles the flavours Sentinel-1 Sigma0 chips and their
masks actually come in: little or big endian, strip or tile layout, no
compression or Deflate, uint8 / uint16 / int16 / float32, and the GeoTIFF tags
needed for an affine transform (ModelPixelScale, ModelTiepoint,
ModelTransformation) plus the EPSG code from the projected or geographic CS key.

If rasterio is installed it is always preferred, since it is exact and handles
the long tail of compressions this module does not.
"""
from __future__ import annotations

import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
               11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 6: "b", 7: "B", 8: "h", 9: "i",
             11: "f", 12: "d", 16: "Q", 17: "q"}

_SAMPLE_DTYPE = {
    (1, 8): "u1", (1, 16): "u2", (1, 32): "u4",
    (2, 8): "i1", (2, 16): "i2", (2, 32): "i4",
    (3, 32): "f4", (3, 64): "f8",
}

TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_STRIP_OFFSETS = 273
TAG_SAMPLES_PER_PIXEL = 277
TAG_ROWS_PER_STRIP = 278
TAG_STRIP_BYTE_COUNTS = 279
TAG_PLANAR_CONFIG = 284
TAG_PREDICTOR = 317
TAG_TILE_WIDTH = 322
TAG_TILE_LENGTH = 323
TAG_TILE_OFFSETS = 324
TAG_TILE_BYTE_COUNTS = 325
TAG_SAMPLE_FORMAT = 339
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922
TAG_MODEL_TRANSFORM = 34264
TAG_GEO_KEY_DIRECTORY = 34735
TAG_DATETIME = 306


class TiffError(RuntimeError):
    pass


def _decode(data: bytes, typ: int, n: int, endian: str):
    if typ == 2:  # ASCII
        return data.split(b"\x00")[0].decode("latin-1")
    if typ == 5:  # RATIONAL
        vals = struct.unpack(endian + "I" * (2 * n), data)
        return [vals[2 * i] / vals[2 * i + 1] if vals[2 * i + 1] else 0.0 for i in range(n)]
    if typ == 10:  # SRATIONAL
        vals = struct.unpack(endian + "i" * (2 * n), data)
        return [vals[2 * i] / vals[2 * i + 1] if vals[2 * i + 1] else 0.0 for i in range(n)]
    fmt = _TYPE_FMT.get(typ)
    if fmt is None:
        return None
    out = list(struct.unpack(endian + fmt * n, data))
    return out[0] if n == 1 else out


def _read_entries(buf: bytes, off: int, endian: str, bigtiff: bool) -> Dict[int, Any]:
    if bigtiff:
        count = struct.unpack(endian + "Q", buf[off:off + 8])[0]
        off += 8
        entry_size, cnt_fmt, val_size = 20, "Q", 8
    else:
        count = struct.unpack(endian + "H", buf[off:off + 2])[0]
        off += 2
        entry_size, cnt_fmt, val_size = 12, "I", 4

    tags: Dict[int, Any] = {}
    for i in range(count):
        e = off + i * entry_size
        tag, typ = struct.unpack(endian + "HH", buf[e:e + 4])
        n = struct.unpack(endian + cnt_fmt, buf[e + 4:e + 4 + val_size])[0]
        payload = buf[e + 4 + val_size:e + entry_size]
        size = _TYPE_SIZES.get(typ, 0) * n
        if size == 0:
            continue
        if size <= val_size:
            data = payload[:size]
        else:
            ptr = struct.unpack(endian + cnt_fmt, payload)[0]
            data = buf[ptr:ptr + size]
        if len(data) < size:
            continue
        tags[tag] = _decode(data, typ, n, endian)
    return tags


def _as_list(v) -> List:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _decompress(chunk: bytes, compression: int) -> bytes:
    if compression in (0, 1):
        return chunk
    if compression in (8, 32946):
        return zlib.decompress(chunk)
    raise TiffError(
        "unsupported TIFF compression %s; install rasterio to read this file" % compression
    )


def read(path) -> Dict[str, Any]:
    """Return a dict with array (bands, h, w), transform, crs, width, height."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) < 8:
        raise TiffError("file too small to be a TIFF")

    bo = buf[:2]
    if bo == b"II":
        endian = "<"
    elif bo == b"MM":
        endian = ">"
    else:
        raise TiffError("not a TIFF (bad byte order mark)")

    magic = struct.unpack(endian + "H", buf[2:4])[0]
    if magic == 42:
        bigtiff = False
        ifd_off = struct.unpack(endian + "I", buf[4:8])[0]
    elif magic == 43:
        bigtiff = True
        ifd_off = struct.unpack(endian + "Q", buf[8:16])[0]
    else:
        raise TiffError("unsupported TIFF magic %s" % magic)

    tags = _read_entries(buf, ifd_off, endian, bigtiff)

    width = int(_as_list(tags.get(TAG_IMAGE_WIDTH))[0])
    height = int(_as_list(tags.get(TAG_IMAGE_LENGTH))[0])
    bits = _as_list(tags.get(TAG_BITS_PER_SAMPLE)) or [8]
    samples = int(_as_list(tags.get(TAG_SAMPLES_PER_PIXEL) or [1])[0])
    sample_fmt = _as_list(tags.get(TAG_SAMPLE_FORMAT)) or [1] * samples
    compression = int(_as_list(tags.get(TAG_COMPRESSION) or [1])[0])
    planar = int(_as_list(tags.get(TAG_PLANAR_CONFIG) or [1])[0])
    predictor = int(_as_list(tags.get(TAG_PREDICTOR) or [1])[0])
    if predictor != 1:
        raise TiffError("TIFF predictor not supported; install rasterio for this file")

    key = (int(sample_fmt[0]), int(bits[0]))
    dt = _SAMPLE_DTYPE.get(key)
    if dt is None:
        raise TiffError("unsupported sample format %s; install rasterio for this file" % (key,))
    dtype = np.dtype(endian + dt)

    tile_w = _as_list(tags.get(TAG_TILE_WIDTH))
    if tile_w:
        array = _read_tiled(buf, tags, endian, dtype, width, height, samples, planar, compression)
    else:
        array = _read_stripped(buf, tags, endian, dtype, width, height, samples, planar, compression)

    return {
        "array": np.ascontiguousarray(array),
        "transform": _geotransform(tags),
        "crs": _crs(tags),
        "width": width,
        "height": height,
        "count": samples,
        "datetime": tags.get(TAG_DATETIME),
    }


def _read_tiled(buf, tags, endian, dtype, width, height, samples, planar, compression):
    tw = int(_as_list(tags.get(TAG_TILE_WIDTH))[0])
    th = int(_as_list(tags.get(TAG_TILE_LENGTH))[0])
    offsets = _as_list(tags.get(TAG_TILE_OFFSETS))
    counts = _as_list(tags.get(TAG_TILE_BYTE_COUNTS))
    across = (width + tw - 1) // tw
    down = (height + th - 1) // th
    out = np.zeros((samples, down * th, across * tw), dtype=dtype)
    planes = samples if planar == 2 else 1
    per_plane = across * down
    for idx, (o, c) in enumerate(zip(offsets, counts)):
        raw = _decompress(buf[int(o):int(o) + int(c)], compression)
        plane = idx // per_plane if planes > 1 else 0
        t = idx % per_plane if planes > 1 else idx
        ty, tx = divmod(t, across)
        spp = 1 if planes > 1 else samples
        arr = np.frombuffer(raw, dtype=dtype)
        need = th * tw * spp
        if arr.size < need:
            arr = np.concatenate([arr, np.zeros(need - arr.size, dtype=dtype)])
        arr = arr[:need].reshape(th, tw, spp)
        ys, xs = ty * th, tx * tw
        if planes > 1:
            out[plane, ys:ys + th, xs:xs + tw] = arr[:, :, 0]
        else:
            out[:, ys:ys + th, xs:xs + tw] = np.transpose(arr, (2, 0, 1))
    return out[:, :height, :width]


def _read_stripped(buf, tags, endian, dtype, width, height, samples, planar, compression):
    rps = int(_as_list(tags.get(TAG_ROWS_PER_STRIP) or [height])[0])
    rps = min(rps, height) if rps > 0 else height
    offsets = _as_list(tags.get(TAG_STRIP_OFFSETS))
    counts = _as_list(tags.get(TAG_STRIP_BYTE_COUNTS))
    planes = samples if planar == 2 else 1
    strips_per_plane = (height + rps - 1) // rps
    out = np.zeros((samples, height, width), dtype=dtype)
    for idx, (o, c) in enumerate(zip(offsets, counts)):
        raw = _decompress(buf[int(o):int(o) + int(c)], compression)
        plane = idx // strips_per_plane if planes > 1 else 0
        s = idx % strips_per_plane if planes > 1 else idx
        y0 = s * rps
        rows = min(rps, height - y0)
        if rows <= 0:
            continue
        spp = 1 if planes > 1 else samples
        arr = np.frombuffer(raw, dtype=dtype)
        need = rows * width * spp
        if arr.size < need:
            arr = np.concatenate([arr, np.zeros(need - arr.size, dtype=dtype)])
        arr = arr[:need].reshape(rows, width, spp)
        if planes > 1:
            out[plane, y0:y0 + rows, :] = arr[:, :, 0]
        else:
            out[:, y0:y0 + rows, :] = np.transpose(arr, (2, 0, 1))
    return out


def _geotransform(tags: Dict[int, Any]) -> Optional[Tuple[float, ...]]:
    """Affine as (a, b, c, d, e, f): X = a*col + b*row + c, Y = d*col + e*row + f."""
    mt = _as_list(tags.get(TAG_MODEL_TRANSFORM))
    if len(mt) >= 16:
        return (float(mt[0]), float(mt[1]), float(mt[3]),
                float(mt[4]), float(mt[5]), float(mt[7]))
    scale = _as_list(tags.get(TAG_MODEL_PIXEL_SCALE))
    tie = _as_list(tags.get(TAG_MODEL_TIEPOINT))
    if len(scale) >= 2 and len(tie) >= 6:
        sx, sy = float(scale[0]), float(scale[1])
        i, j, _k, x, y, _z = [float(v) for v in tie[:6]]
        return (sx, 0.0, x - i * sx, 0.0, -sy, y + j * sy)
    return None


def _crs(tags: Dict[int, Any]) -> Optional[str]:
    keys = _as_list(tags.get(TAG_GEO_KEY_DIRECTORY))
    if len(keys) < 4:
        return None
    n = int(keys[3])
    for i in range(n):
        base = 4 + i * 4
        if base + 3 >= len(keys):
            break
        key_id = int(keys[base])
        loc = int(keys[base + 1])
        count = int(keys[base + 2])
        value = int(keys[base + 3])
        if loc != 0 or count != 1:
            continue
        if key_id in (3072, 2048) and value not in (0, 32767):
            return "EPSG:%d" % value
    return None


def write(path, array: np.ndarray, transform: Tuple[float, ...], epsg: int = 4326) -> None:
    """Write a minimal, uncompressed, strip-per-image GeoTIFF.

    Used by the data preparation scripts so that a repo without rasterio can
    still produce georeferenced rasters that this reader and rasterio both read.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    bands, height, width = arr.shape
    kind = arr.dtype.kind
    if kind == "f":
        arr = arr.astype("<f4")
        sample_format, bits = 3, 32
    elif kind == "u":
        arr = arr.astype("<u1") if arr.dtype.itemsize == 1 else arr.astype("<u2")
        sample_format, bits = 1, arr.dtype.itemsize * 8
    else:
        arr = arr.astype("<i2")
        sample_format, bits = 2, 16

    interleaved = np.transpose(arr, (1, 2, 0)).tobytes()

    a, b, c, d, e, f = transform
    pixel_scale = (abs(a), abs(e), 0.0)
    tiepoint = (0.0, 0.0, 0.0, c, f, 0.0)
    geokeys = [1, 1, 0, 3,
               1024, 0, 1, 2,      # GTModelTypeGeoKey = geographic
               1025, 0, 1, 1,      # GTRasterTypeGeoKey = PixelIsArea
               2048, 0, 1, epsg]   # GeographicTypeGeoKey

    entries = []
    extra = bytearray()
    header_len = 8

    def add(tag, typ, values):
        entries.append((tag, typ, values))

    add(TAG_IMAGE_WIDTH, 4, [width])
    add(TAG_IMAGE_LENGTH, 4, [height])
    add(TAG_BITS_PER_SAMPLE, 3, [bits] * bands)
    add(TAG_COMPRESSION, 3, [1])
    add(262, 3, [1])  # photometric min-is-black
    add(TAG_STRIP_OFFSETS, 4, [0])  # patched below
    add(TAG_SAMPLES_PER_PIXEL, 3, [bands])
    add(TAG_ROWS_PER_STRIP, 4, [height])
    add(TAG_STRIP_BYTE_COUNTS, 4, [len(interleaved)])
    add(TAG_PLANAR_CONFIG, 3, [1])
    add(TAG_SAMPLE_FORMAT, 3, [sample_format] * bands)
    add(TAG_MODEL_PIXEL_SCALE, 12, list(pixel_scale))
    add(TAG_MODEL_TIEPOINT, 12, list(tiepoint))
    add(TAG_GEO_KEY_DIRECTORY, 3, geokeys)
    entries.sort(key=lambda t: t[0])

    ifd_off = header_len
    ifd_len = 2 + 12 * len(entries) + 4
    extra_off = ifd_off + ifd_len
    fmt_for = {3: "H", 4: "I", 12: "d"}

    packed = []
    for tag, typ, values in entries:
        n = len(values)
        raw = struct.pack("<" + fmt_for[typ] * n, *values)
        if len(raw) <= 4:
            payload = raw + b"\x00" * (4 - len(raw))
            packed.append((tag, typ, n, payload, None))
        else:
            packed.append((tag, typ, n, None, len(extra)))
            extra.extend(raw)

    data_off = extra_off + len(extra)

    out = bytearray()
    out.extend(struct.pack("<2sHI", b"II", 42, ifd_off))
    out.extend(struct.pack("<H", len(packed)))
    for tag, typ, n, payload, rel in packed:
        if payload is None:
            payload = struct.pack("<I", extra_off + rel)
        if tag == TAG_STRIP_OFFSETS:
            payload = struct.pack("<I", data_off)
        out.extend(struct.pack("<HHI", tag, typ, n))
        out.extend(payload)
    out.extend(struct.pack("<I", 0))
    out.extend(extra)
    out.extend(interleaved)
    with open(path, "wb") as fh:
        fh.write(bytes(out))
