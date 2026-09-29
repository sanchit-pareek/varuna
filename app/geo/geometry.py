"""Polygon extraction and geometric characterisation of slicks.

Everything here is computed from the segmentation mask plus the SAR affine.
Nothing is looked up, assumed, or hardcoded. The metrics returned are the ones
the problem statement names: area, perimeter, length, orientation, centroid,
bounding box, plus compactness and pixel count.

scipy and shapely are optional. Pure numpy equivalents are used when they are
absent so the app never dies on a minimal install.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .crs import LocalAEQD, haversine_km
from .raster import Raster

try:
    from scipy import ndimage as ndi

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

def label_components(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """4-connected labelling. Uses scipy when available, else a union-find."""
    b = np.asarray(binary, dtype=bool)
    if _HAVE_SCIPY:
        lab, n = ndi.label(b)
        return lab.astype(np.int32), int(n)
    return _label_union_find(b)


def _label_union_find(b: np.ndarray) -> Tuple[np.ndarray, int]:
    h, w = b.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: List[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, c: int) -> None:
        ra, rc = find(a), find(c)
        if ra != rc:
            parent[max(ra, rc)] = min(ra, rc)

    nxt = 1
    for y in range(h):
        row = b[y]
        for x in range(w):
            if not row[x]:
                continue
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up and left:
                labels[y, x] = min(up, left)
                union(up, left)
            elif up or left:
                labels[y, x] = up or left
            else:
                labels[y, x] = nxt
                parent.append(nxt)
                nxt += 1
    remap: Dict[int, int] = {}
    out = np.zeros_like(labels)
    n = 0
    nz = np.argwhere(labels > 0)
    for y, x in nz:
        r = find(int(labels[y, x]))
        if r not in remap:
            n += 1
            remap[r] = n
        out[y, x] = remap[r]
    return out, n


def component_slices(labels: np.ndarray, n: int):
    """Bounding box slice for every label, so per component work stays local.

    Without this, a scene with a few thousand dark patches costs a full frame
    comparison per patch and detection takes minutes instead of seconds.
    """
    if _HAVE_SCIPY:
        return ndi.find_objects(labels, max_label=n)
    out = []
    for lab in range(1, n + 1):
        rr, cc = np.nonzero(labels == lab)
        if rr.size == 0:
            out.append(None)
        else:
            out.append((slice(int(rr.min()), int(rr.max()) + 1),
                        slice(int(cc.min()), int(cc.max()) + 1)))
    return out


def binary_closing(binary: np.ndarray, size: int = 3) -> np.ndarray:
    """Morphological closing then opening, to remove speckle salt and pepper."""
    b = np.asarray(binary, dtype=bool)
    if size < 2:
        return b
    if _HAVE_SCIPY:
        st = np.ones((size, size), dtype=bool)
        b = ndi.binary_closing(b, structure=st)
        b = ndi.binary_opening(b, structure=st)
        return b
    return _open_close_numpy(b, size)


def _shift_or(b: np.ndarray, size: int) -> np.ndarray:
    r = size // 2
    out = np.zeros_like(b)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(b, dy, axis=0), dx, axis=1)
    return out


def _shift_and(b: np.ndarray, size: int) -> np.ndarray:
    r = size // 2
    out = np.ones_like(b)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out &= np.roll(np.roll(b, dy, axis=0), dx, axis=1)
    return out


def _open_close_numpy(b: np.ndarray, size: int) -> np.ndarray:
    closed = _shift_and(_shift_or(b, size), size)
    opened = _shift_or(_shift_and(closed, size), size)
    return opened


# ---------------------------------------------------------------------------
# Contour tracing
# ---------------------------------------------------------------------------

_MOORE = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def trace_boundary(mask: np.ndarray) -> List[Tuple[float, float]]:
    """Moore neighbourhood boundary trace of the largest blob in `mask`.

    Returns a closed ring of (col, row) pixel corner coordinates.
    """
    m = np.asarray(mask, dtype=bool)
    idx = np.argwhere(m)
    if idx.size == 0:
        return []
    start = tuple(int(v) for v in idx[np.lexsort((idx[:, 1], idx[:, 0]))][0])  # topmost, then leftmost
    h, w = m.shape

    def inside(p):
        return 0 <= p[0] < h and 0 <= p[1] < w and m[p[0], p[1]]

    contour = [start]
    backtrack = (start[0], start[1] - 1)
    current = start
    guard = 0
    limit = 8 * int(m.sum()) + 64
    while guard < limit:
        guard += 1
        try:
            d0 = _MOORE.index((backtrack[0] - current[0], backtrack[1] - current[1]))
        except ValueError:
            d0 = 0
        found = None
        for k in range(1, 9):
            d = _MOORE[(d0 + k) % 8]
            cand = (current[0] + d[0], current[1] + d[1])
            if inside(cand):
                found = cand
                backtrack = (current[0] + _MOORE[(d0 + k - 1) % 8][0],
                             current[1] + _MOORE[(d0 + k - 1) % 8][1])
                break
        if found is None:
            break
        current = found
        if current == start and len(contour) > 2:
            break
        contour.append(current)
    ring = [(float(c) + 0.5, float(r) + 0.5) for r, c in contour]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def simplify(points: Sequence[Tuple[float, float]], tolerance: float = 1.0) -> List[Tuple[float, float]]:
    """Douglas-Peucker on a ring, keeping it closed."""
    pts = list(points)
    if len(pts) <= 4:
        return pts
    closed = pts[0] == pts[-1]
    core = pts[:-1] if closed else pts
    keep = _dp(core, tolerance)
    if closed and (not keep or keep[0] != keep[-1]):
        keep = keep + [keep[0]]
    return keep


def _cross2(ox: float, oy: float, ax: float, ay: float, bx: float, by: float) -> float:
    """z component of (a - o) x (b - o). numpy 2 dropped the 2D np.cross."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _dp(pts: List[Tuple[float, float]], tol: float) -> List[Tuple[float, float]]:
    """Douglas-Peucker, iterative so a long coastline-shaped ring cannot blow
    the Python recursion limit."""
    n = len(pts)
    if n < 3:
        return list(pts)
    a = np.array(pts, dtype=float)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[n - 1] = True
    stack: List[Tuple[int, int]] = [(0, n - 1)]

    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        start, end = a[i0], a[i1]
        seg = end - start
        seg_len = float(np.hypot(seg[0], seg[1]))
        chunk = a[i0:i1 + 1]
        if seg_len < 1e-12:
            d = np.hypot(chunk[:, 0] - start[0], chunk[:, 1] - start[1])
        else:
            rel = chunk - start
            d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / seg_len
        k = int(np.argmax(d))
        if d[k] > tol and 0 < k < (i1 - i0):
            idx = i0 + k
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))

    return [pts[i] for i in range(n) if keep[i]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class SlickPolygon:
    """One connected component with its geometric characterisation."""

    polygon_id: str
    klass: int
    ring_lonlat: List[Tuple[float, float]]
    pixel_count: int
    area_km2: float
    perimeter_km: float
    length_km: float
    width_km: float
    orientation_deg: float
    compactness: float
    centroid_lon: float
    centroid_lat: float
    bbox: Tuple[float, float, float, float]
    confidence: float
    mean_sigma0_db: Optional[float] = None
    contrast_db: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_feature(self) -> Dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[float(x), float(y)] for x, y in self.ring_lonlat]]},
            "properties": {
                "polygon_id": self.polygon_id,
                "class": self.klass,
                "class_name": {0: "sea", 1: "look_alike", 2: "mineral_oil"}.get(self.klass, "unknown"),
                "pixel_count": self.pixel_count,
                "area_km2": round(self.area_km2, 4),
                "perimeter_km": round(self.perimeter_km, 4),
                "length_km": round(self.length_km, 4),
                "width_km": round(self.width_km, 4),
                "orientation_deg": round(self.orientation_deg, 2),
                "compactness": round(self.compactness, 4),
                "centroid_lon": round(self.centroid_lon, 6),
                "centroid_lat": round(self.centroid_lat, 6),
                "bbox": [round(float(v), 6) for v in self.bbox],
                "confidence": round(self.confidence, 4),
                "mean_sigma0_db": None if self.mean_sigma0_db is None else round(self.mean_sigma0_db, 2),
                "contrast_db": None if self.contrast_db is None else round(self.contrast_db, 2),
            },
        }


def polygons_from_mask(
    class_mask: np.ndarray,
    sar: Raster,
    klass: int,
    prob: Optional[np.ndarray] = None,
    sigma0_db: Optional[np.ndarray] = None,
    min_area_km2: float = 0.05,
    min_pixels: int = 500,
    simplify_px: float = 1.5,
    prefix: str = "P",
) -> List[SlickPolygon]:
    """Extract and characterise every component of `klass` in `class_mask`."""
    binary = np.asarray(class_mask) == klass
    if not binary.any():
        return []
    binary = binary_closing(binary, size=3)
    labels, n = label_components(binary)
    if n == 0:
        return []

    px_area = sar.pixel_area_km2()
    sea_db = None
    if sigma0_db is not None:
        vals = sigma0_db[np.isfinite(sigma0_db) & (np.asarray(class_mask) == 0)]
        if vals.size > 100:
            sea_db = float(np.median(vals))

    boxes = component_slices(labels, n)
    out: List[SlickPolygon] = []
    for lab in range(1, n + 1):
        box = boxes[lab - 1] if lab - 1 < len(boxes) else None
        if box is None:
            continue
        sub = labels[box] == lab
        count = int(sub.sum())
        area_km2 = count * px_area
        if area_km2 < min_area_km2 and count < min_pixels:
            continue
        r0, c0 = box[0].start, box[1].start
        ring_px = trace_boundary(sub)
        if len(ring_px) < 4:
            continue
        ring_px = simplify(ring_px, tolerance=simplify_px)
        cols = np.array([p[0] for p in ring_px]) + c0
        rows = np.array([p[1] for p in ring_px]) + r0
        lon, lat = sar.lonlat(cols - 0.5, rows - 0.5)
        ring = [(float(a), float(b)) for a, b in zip(lon, lat)]

        rr, cc = np.nonzero(sub)
        clon, clat = sar.lonlat(cc.mean() + c0, rr.mean() + r0)
        clon, clat = float(clon), float(clat)

        frame = LocalAEQD(clat, clon)
        xs, ys = frame.to_m(np.array(lon), np.array(lat))
        metrics = _ring_metrics(xs, ys)

        conf = 1.0
        if prob is not None:
            pv = np.asarray(prob)[box][sub]
            pv = pv[np.isfinite(pv)]
            conf = float(pv.mean()) if pv.size else 1.0

        mean_db = None
        contrast = None
        if sigma0_db is not None:
            dv = np.asarray(sigma0_db)[box][sub]
            dv = dv[np.isfinite(dv)]
            if dv.size:
                mean_db = float(dv.mean())
                if sea_db is not None:
                    contrast = float(sea_db - mean_db)

        out.append(
            SlickPolygon(
                polygon_id="%s%02d" % (prefix, len(out) + 1),
                klass=int(klass),
                ring_lonlat=ring,
                pixel_count=count,
                area_km2=float(area_km2),
                perimeter_km=metrics["perimeter_m"] / 1000.0,
                length_km=metrics["length_m"] / 1000.0,
                width_km=metrics["width_m"] / 1000.0,
                orientation_deg=metrics["orientation_deg"],
                compactness=metrics["compactness"],
                centroid_lon=clon,
                centroid_lat=clat,
                bbox=(float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))),
                confidence=conf,
                mean_sigma0_db=mean_db,
                contrast_db=contrast,
            )
        )
    out.sort(key=lambda p: p.area_km2, reverse=True)
    for i, p in enumerate(out, start=1):
        p.polygon_id = "%s%02d" % (prefix, i)
    return out


def _ring_metrics(xs: np.ndarray, ys: np.ndarray) -> Dict[str, float]:
    """Perimeter, max Feret length, cross width, PCA orientation, compactness."""
    pts = np.column_stack([xs, ys])
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        core = pts[:-1]
    else:
        core = pts
    d = np.diff(np.vstack([core, core[:1]]), axis=0)
    perimeter = float(np.sum(np.hypot(d[:, 0], d[:, 1])))

    hull = _convex_hull(core)
    length, axis_vec = _max_feret(hull)

    centred = core - core.mean(axis=0)
    if len(centred) >= 2:
        cov = np.cov(centred.T)
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(vals))]
    else:
        major = np.array([1.0, 0.0])
    if axis_vec is not None and np.hypot(*axis_vec) > 0:
        major = axis_vec / np.hypot(*axis_vec)
    orientation = math.degrees(math.atan2(major[1], major[0])) % 180.0

    perp = np.array([-major[1], major[0]])
    proj = centred @ perp
    width = float(proj.max() - proj.min()) if len(proj) else 0.0

    area = _shoelace(core)
    compactness = (4.0 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    return {
        "perimeter_m": perimeter,
        "length_m": float(length),
        "width_m": width,
        "orientation_deg": float(orientation),
        "compactness": float(min(max(compactness, 0.0), 1.0)),
        "hull_area_m2": float(area),
    }


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew monotone chain."""
    p = np.unique(pts, axis=0)
    if len(p) <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(seq):
        out: List[np.ndarray] = []
        for q in seq:
            while len(out) >= 2 and _cross2(out[-2][0], out[-2][1],
                                            out[-1][0], out[-1][1],
                                            q[0], q[1]) <= 0:
                out.pop()
            out.append(q)
        return out

    lower = half(p)
    upper = half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _max_feret(hull: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    if len(hull) < 2:
        return 0.0, None
    best = 0.0
    vec = None
    for i in range(len(hull)):
        d = hull - hull[i]
        dist = np.hypot(d[:, 0], d[:, 1])
        j = int(np.argmax(dist))
        if dist[j] > best:
            best = float(dist[j])
            vec = hull[j] - hull[i]
    return best, vec


def feature_collection(polys: Sequence[SlickPolygon], **props) -> Dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "properties": dict(props),
        "features": [p.to_feature() for p in polys],
    }


def ring_area_km2(ring: Sequence[Tuple[float, float]]) -> float:
    """Geodesic-ish area of a lon/lat ring via a local aeqd frame."""
    if len(ring) < 3:
        return 0.0
    lon = np.array([p[0] for p in ring], dtype=float)
    lat = np.array([p[1] for p in ring], dtype=float)
    frame = LocalAEQD(float(lat.mean()), float(lon.mean()))
    x, y = frame.to_m(lon, lat)
    return _shoelace(np.column_stack([x, y])) / 1.0e6


def point_in_ring(lon: float, lat: float, ring: Sequence[Tuple[float, float]]) -> bool:
    """Ray casting in lon/lat. Adequate at the scales this product works at."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            xint = (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-15) + x1
            if lon < xint:
                inside = not inside
    return inside


def distance_to_ring_km(lon: float, lat: float, ring: Sequence[Tuple[float, float]]) -> float:
    """0 if inside, else km to the nearest edge."""
    if not ring:
        return float("inf")
    if point_in_ring(lon, lat, ring):
        return 0.0
    frame = LocalAEQD(lat, lon)
    rl = np.array([p[0] for p in ring], dtype=float)
    rt = np.array([p[1] for p in ring], dtype=float)
    rx, ry = frame.to_m(rl, rt)
    px, py = 0.0, 0.0
    best = float("inf")
    for i in range(len(rx)):
        j = (i + 1) % len(rx)
        ax, ay = rx[i], ry[i]
        bx, by = rx[j], ry[j]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        cx, cy = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(px - cx, py - cy))
    return best / 1000.0


def points_in_ring(lons, lats, ring: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Vectorised ray casting for many points against one ring."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    inside = np.zeros(lons.shape, dtype=bool)
    n = len(ring)
    if n < 3:
        return inside
    rx = np.array([p[0] for p in ring], dtype=float)
    ry = np.array([p[1] for p in ring], dtype=float)
    for i in range(n):
        j = (i + 1) % n
        x1, y1, x2, y2 = rx[i], ry[i], rx[j], ry[j]
        straddles = (y1 > lats) != (y2 > lats)
        if not straddles.any():
            continue
        xint = (x2 - x1) * (lats - y1) / ((y2 - y1) + 1e-15) + x1
        inside ^= straddles & (lons < xint)
    return inside


def ring_distances_km(lons, lats, ring: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Distance from every point to a ring, 0 inside. One local frame, no loops.

    The scalar version is fine for one query, but the AIS join asks this
    question for thousands of track samples per vessel. Doing it per point in
    Python is what turned a two second join into a thirty second one, so this
    projects everything once and works on whole arrays.
    """
    lons = np.atleast_1d(np.asarray(lons, dtype=float))
    lats = np.atleast_1d(np.asarray(lats, dtype=float))
    if not ring or len(ring) < 2:
        return np.full(lons.shape, np.inf)

    rl = np.array([p[0] for p in ring], dtype=float)
    rt = np.array([p[1] for p in ring], dtype=float)
    frame = LocalAEQD(float(rt.mean()), float(rl.mean()))
    rx, ry = frame.to_m(rl, rt)
    px, py = frame.to_m(lons, lats)
    px = np.asarray(px, dtype=float).ravel()
    py = np.asarray(py, dtype=float).ravel()

    ax, ay = rx[:-1], ry[:-1]
    bx, by = rx[1:], ry[1:]
    if len(rx) > 2 and (rx[0] != rx[-1] or ry[0] != ry[-1]):
        ax = np.append(ax, rx[-1]); ay = np.append(ay, ry[-1])
        bx = np.append(bx, rx[0]);  by = np.append(by, ry[0])

    dx = bx - ax
    dy = by - ay
    seg2 = dx * dx + dy * dy
    seg2 = np.where(seg2 < 1e-12, 1e-12, seg2)

    # (points, segments) broadcast
    wx = px[:, None] - ax[None, :]
    wy = py[:, None] - ay[None, :]
    t = np.clip((wx * dx[None, :] + wy * dy[None, :]) / seg2[None, :], 0.0, 1.0)
    cx = wx - t * dx[None, :]
    cy = wy - t * dy[None, :]
    d = np.sqrt(cx * cx + cy * cy).min(axis=1) / 1000.0

    d[points_in_ring(lons, lats, ring).ravel()] = 0.0
    return d.reshape(lons.shape)


def track_min_distance_km(track_lon, track_lat, ring: Sequence[Tuple[float, float]]) -> Tuple[float, int]:
    """Minimum distance from any track sample to a ring, and its index."""
    d = ring_distances_km(track_lon, track_lat, ring)
    if d.size == 0:
        return float("inf"), 0
    i = int(np.argmin(d))
    return float(d[i]), i


def buffer_ring_km(ring: Sequence[Tuple[float, float]], km: float, segments: int = 24) -> List[Tuple[float, float]]:
    """Approximate outward buffer: convex hull of discs placed on every vertex.

    Good enough for an origin zone that is then drawn with an explicit
    uncertainty label, and it avoids a hard shapely dependency.
    """
    if not ring:
        return []
    lon = np.array([p[0] for p in ring], dtype=float)
    lat = np.array([p[1] for p in ring], dtype=float)
    lon0, lat0 = float(lon.mean()), float(lat.mean())
    frame = LocalAEQD(lat0, lon0)
    x, y = frame.to_m(lon, lat)
    r = km * 1000.0
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    pts = []
    for cx, cy in zip(x, y):
        pts.extend(zip(cx + r * np.cos(ang), cy + r * np.sin(ang)))
    hull = _convex_hull(np.array(pts))
    hlon, hlat = frame.to_deg(hull[:, 0], hull[:, 1])
    out = [(float(a), float(b)) for a, b in zip(hlon, hlat)]
    if out and out[0] != out[-1]:
        out.append(out[0])
    return out


def hull_ring(lons, lats) -> List[Tuple[float, float]]:
    """Convex hull of a point cloud, returned as a closed lon/lat ring."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.size == 0:
        return []
    if lons.size < 3:
        return [(float(a), float(b)) for a, b in zip(lons, lats)]
    lon0, lat0 = float(lons.mean()), float(lats.mean())
    frame = LocalAEQD(lat0, lon0)
    x, y = frame.to_m(lons, lats)
    hull = _convex_hull(np.column_stack([x, y]))
    hlon, hlat = frame.to_deg(hull[:, 0], hull[:, 1])
    ring = [(float(a), float(b)) for a, b in zip(hlon, hlat)]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def spread_radius_km(lons, lats) -> float:
    """Radius that contains 90 percent of an ensemble, about its median."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.size == 0:
        return 0.0
    clon, clat = float(np.median(lons)), float(np.median(lats))
    d = haversine_km(clat, clon, lats, lons)
    return float(np.percentile(d, 90))
