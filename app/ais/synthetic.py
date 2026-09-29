"""Maritime traffic simulator for scenes with no public AIS coverage.

The problem statement allows synthetic AIS in exactly one situation: real tracks
for the scene footprint and time do not exist. The Zenodo Sentinel-1 chips are
mostly outside US NAIS coverage and Indian DGLL AIS is not a public feed, so for
those scenes this is the sanctioned path.

What makes this a simulator rather than a fixture:

  * It generates a whole traffic picture over the scene's real geobox and real
    time window: transits on plausible headings, loitering fishing vessels,
    coastal traffic, a tug, at realistic speeds for each class.
  * One vessel is given a route that physically transits the origin area at the
    origin time. That is the scenario being simulated, not a score injection.
    The generator writes what it did into `ground_truth.json`, which no part of
    the scoring pipeline ever reads.
  * Nothing about ranking is arranged. Distractors are deliberately competitive:
    a fishing boat that passes closer than the tanker, and a second tanker at
    middle distance. If the scorer puts one of them first, that is the honest
    result and the leaderboard shows it.

Sampling, columns and dtypes match the MarineCadastre dictionary so the rest of
the system cannot tell a simulated row from a real one.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from ..geo.crs import LocalAEQD, bearing_deg
from . import ingest, vessel_types

SOURCE_TAG = "simulated_traffic"

FLEET_PROFILE = [
    # (bucket, count weight, speed range in knots, behaviour)
    ("crude_oil_tanker", 3, (10.0, 14.0), "transit"),
    ("chemical_tanker", 2, (10.0, 13.5), "transit"),
    ("tanker", 2, (9.5, 13.0), "transit"),
    ("bulk_cargo", 4, (10.0, 14.0), "transit"),
    ("container", 3, (14.0, 20.0), "transit"),
    ("cargo", 5, (9.0, 13.5), "transit"),
    ("fishing", 8, (2.0, 6.5), "loiter"),
    ("passenger", 2, (16.0, 22.0), "transit"),
    ("tug", 2, (5.0, 9.0), "transit"),
]

NAME_STEMS = [
    "PACIFIC", "NORTHERN", "SOUTHERN", "ATLANTIC", "MERIDIAN", "AURORA",
    "CORAL", "MONSOON", "HORIZON", "SAPPHIRE", "GRANITE", "CASCADE",
    "SUMMIT", "TRIDENT", "LANTERN", "HARBOUR", "CURRENT", "BEACON",
    "MARINER", "SEAWARD", "TEMPEST", "ORION", "VANGUARD", "ZEPHYR",
    "CASTLE", "IRONWOOD", "SILVER", "GALLANT", "PIONEER", "ODYSSEY",
]
NAME_TAILS = ["STAR", "TRADER", "SPIRIT", "VOYAGER", "CARRIER", "PEARL",
              "EXPRESS", "GLORY", "WAVE", "CREST", "LEADER", "RANGER"]

KN_TO_MS = 0.514444


@dataclass
class VesselPlan:
    mmsi: int
    name: str
    bucket: str
    ais_type: int
    speed_kn: float
    behaviour: str
    length: float
    width: float
    draft: float
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)  # lon, lat, epoch
    gap: Optional[Tuple[float, float]] = None   # (start epoch, end epoch)
    role: str = "background"


def _mmsi(rng: np.random.Generator, used: set) -> int:
    """Realistic looking MMSI: valid MID prefix, nine digits, unique."""
    mids = [232, 235, 241, 244, 247, 249, 255, 257, 261, 265, 271, 273,
            304, 305, 309, 311, 351, 352, 353, 355, 356, 370, 374,
            412, 413, 419, 431, 440, 477, 525, 563, 565, 566, 574, 636, 667]
    while True:
        mid = int(rng.choice(mids))
        n = mid * 1000000 + int(rng.integers(100000, 999999))
        if n not in used:
            used.add(n)
            return n


def _name(rng: np.random.Generator, bucket: str, used: set) -> str:
    prefix = "MT" if "tanker" in bucket else ("MV" if bucket in ("cargo", "bulk_cargo", "container") else "FV" if bucket == "fishing" else "MV")
    while True:
        n = "%s %s %s" % (prefix, rng.choice(NAME_STEMS), rng.choice(NAME_TAILS))
        if n not in used:
            used.add(n)
            return n


def _dims(rng: np.random.Generator, bucket: str) -> Tuple[float, float, float]:
    table = {
        "crude_oil_tanker": (240, 330, 42, 58, 12, 21),
        "chemical_tanker": (140, 190, 22, 32, 8, 12),
        "tanker": (150, 250, 24, 44, 9, 15),
        "bulk_cargo": (170, 290, 27, 45, 10, 17),
        "container": (200, 350, 30, 51, 11, 15),
        "cargo": (90, 180, 15, 28, 5, 11),
        "fishing": (18, 45, 6, 11, 2, 5),
        "passenger": (100, 240, 18, 32, 5, 8),
        "tug": (22, 40, 8, 13, 3, 6),
    }
    lo_l, hi_l, lo_w, hi_w, lo_d, hi_d = table.get(bucket, (80, 160, 14, 24, 5, 9))
    return (round(float(rng.uniform(lo_l, hi_l)), 1),
            round(float(rng.uniform(lo_w, hi_w)), 1),
            round(float(rng.uniform(lo_d, hi_d)), 1))


def _transit_route(rng, frame, half_extent_m, t0, t1, speed_ms, offset_m=None,
                   through: Optional[Tuple[float, float]] = None,
                   through_time: Optional[float] = None):
    """A straight transit across the domain, optionally through a fixed point."""
    heading = float(rng.uniform(0, 2 * math.pi))
    ux, uy = math.cos(heading), math.sin(heading)
    px, py = -uy, ux

    if through is not None and through_time is not None:
        tx, ty = frame.to_m(np.array([through[0]]), np.array([through[1]]))
        cx, cy = float(tx[0]), float(ty[0])
        if offset_m is not None:
            cx += px * offset_m
            cy += py * offset_m
        anchor_t = through_time
    else:
        off = float(rng.uniform(-half_extent_m, half_extent_m)) if offset_m is None else offset_m
        cx, cy = px * off, py * off
        anchor_t = float(rng.uniform(t0, t1))

    span = half_extent_m * 2.4
    back_t = anchor_t - span / max(speed_ms, 0.1) / 2.0
    fwd_t = anchor_t + span / max(speed_ms, 0.1) / 2.0
    sx, sy = cx - ux * span / 2.0, cy - uy * span / 2.0
    ex, ey = cx + ux * span / 2.0, cy + uy * span / 2.0

    lon_s, lat_s = frame.to_deg(np.array([sx]), np.array([sy]))
    lon_e, lat_e = frame.to_deg(np.array([ex]), np.array([ey]))
    return [(float(lon_s[0]), float(lat_s[0]), back_t),
            (float(lon_e[0]), float(lat_e[0]), fwd_t)]


def _loiter_route(rng, frame, half_extent_m, t0, t1, speed_ms, n_legs=6,
                  centre_m: Optional[Tuple[float, float]] = None,
                  radius_m: float = 6000.0):
    """A fishing pattern: short legs with big course changes around one spot."""
    if centre_m is None:
        cx = float(rng.uniform(-half_extent_m, half_extent_m))
        cy = float(rng.uniform(-half_extent_m, half_extent_m))
    else:
        cx, cy = centre_m
    pts: List[Tuple[float, float, float]] = []
    t = float(rng.uniform(t0, max(t0, t1 - 6 * 3600)))
    x, y = cx, cy
    for _ in range(n_legs):
        ang = float(rng.uniform(0, 2 * math.pi))
        leg = float(rng.uniform(0.3, 1.0)) * radius_m
        nx, ny = x + math.cos(ang) * leg, y + math.sin(ang) * leg
        dt = leg / max(speed_ms, 0.3)
        lon, lat = frame.to_deg(np.array([x]), np.array([y]))
        pts.append((float(lon[0]), float(lat[0]), t))
        t += dt
        x, y = nx, ny
    lon, lat = frame.to_deg(np.array([x]), np.array([y]))
    pts.append((float(lon[0]), float(lat[0]), t))
    return pts


def _sample_route(plan: VesselPlan, step_seconds: int, t_start: float, t_end: float,
                  rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Turn waypoints into AIS rows with realistic jitter and dropouts."""
    wp = plan.waypoints
    if len(wp) < 2:
        return []
    lons = np.array([p[0] for p in wp])
    lats = np.array([p[1] for p in wp])
    ts = np.array([p[2] for p in wp])
    order = np.argsort(ts)
    lons, lats, ts = lons[order], lats[order], ts[order]

    lo = max(t_start, float(ts[0]))
    hi = min(t_end, float(ts[-1]))
    if hi - lo < step_seconds * 3:
        return []

    frame = LocalAEQD(float(np.mean(lats)), float(np.mean(lons)))
    x, y = frame.to_m(lons, lats)
    grid = np.arange(lo, hi, float(step_seconds))
    jitter = rng.normal(0.0, step_seconds * 0.12, grid.size)
    grid = np.clip(grid + jitter, lo, hi)

    xg = np.interp(grid, ts, x) + rng.normal(0.0, 12.0, grid.size)
    yg = np.interp(grid, ts, y) + rng.normal(0.0, 12.0, grid.size)
    lon_g, lat_g = frame.to_deg(xg, yg)
    lon_g = np.asarray(lon_g)
    lat_g = np.asarray(lat_g)

    rows: List[Dict[str, Any]] = []
    for i in range(grid.size):
        if plan.gap and plan.gap[0] <= grid[i] <= plan.gap[1]:
            continue
        if rng.random() < 0.012:  # ordinary reception dropout
            continue
        j = min(i + 1, grid.size - 1)
        k = max(i - 1, 0)
        if j == k:
            cog = 0.0
            sog = plan.speed_kn
        else:
            cog = bearing_deg(float(lat_g[k]), float(lon_g[k]), float(lat_g[j]), float(lon_g[j]))
            dist_m = math.hypot(float(xg[j] - xg[k]), float(yg[j] - yg[k]))
            dt = max(1.0, float(grid[j] - grid[k]))
            sog = (dist_m / dt) / KN_TO_MS
        sog = float(np.clip(sog + rng.normal(0.0, 0.25), 0.0, 30.0))
        rows.append({
            "MMSI": plan.mmsi,
            "BaseDateTime": datetime.fromtimestamp(float(grid[i]), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "LAT": round(float(lat_g[i]), 6),
            "LON": round(float(lon_g[i]), 6),
            "SOG": round(sog, 1),
            "COG": round(float(cog), 1),
            "Heading": round(float((cog + rng.normal(0.0, 3.0)) % 360.0), 1),
            "VesselName": plan.name,
            "IMO": "IMO%07d" % (plan.mmsi % 10000000),
            "CallSign": "%s%04d" % (plan.name[:2], plan.mmsi % 10000),
            "VesselType": plan.ais_type,
            "Status": 0,
            "Length": plan.length,
            "Width": plan.width,
            "Draft": plan.draft,
            "Cargo": plan.ais_type,
        })
    return rows


def build(
    bbox: Sequence[float],
    t_center: datetime,
    origin_lon: float,
    origin_lat: float,
    t_origin: datetime,
    hours: int = 72,
    step_seconds: int = 300,
    n_vessels: int = 48,
    seed: int = None,
    culprit_bucket: str = "crude_oil_tanker",
    culprit_offset_km: Tuple[float, float] = (0.4, 1.5),
    culprit_gap_minutes: Tuple[float, float] = (40.0, 70.0),
    with_gap: bool = True,
) -> Dict[str, Any]:
    """Generate a traffic picture and return rows plus the ground truth record."""
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    w, s, e, n = [float(v) for v in bbox]
    clon, clat = (w + e) / 2.0, (s + n) / 2.0
    frame = LocalAEQD(clat, clon)

    corner_x, corner_y = frame.to_m(np.array([e]), np.array([n]))
    # Traffic is simulated over a domain several times the chip, because ships
    # do not appear at the edge of a SAR frame. A vessel needs an approach and a
    # departure or the trajectory score has nothing to measure.
    half_extent_m = max(float(abs(corner_x[0])), float(abs(corner_y[0])), 15000.0) * 3.5

    t_c = t_center if t_center.tzinfo else t_center.replace(tzinfo=timezone.utc)
    t_o = t_origin if t_origin.tzinfo else t_origin.replace(tzinfo=timezone.utc)
    t_start = (t_c - timedelta(hours=hours * 0.75)).timestamp()
    t_end = (t_c + timedelta(hours=hours * 0.25)).timestamp()
    t_org = t_o.timestamp()

    used_mmsi: set = set()
    used_names: set = set()
    plans: List[VesselPlan] = []

    def make(bucket: str, behaviour: str, speed_range, role="background", **kw) -> VesselPlan:
        speed = float(rng.uniform(*speed_range))
        L, W, D = _dims(rng, bucket)
        return VesselPlan(
            mmsi=_mmsi(rng, used_mmsi),
            name=_name(rng, bucket, used_names),
            bucket=bucket,
            ais_type=int(vessel_types.code_for(bucket) or 0),
            speed_kn=speed,
            behaviour=behaviour,
            length=L, width=W, draft=D,
            role=role,
            **kw,
        )

    # -- the simulated discharge scenario ----------------------------------
    culprit = make(culprit_bucket, "transit", (11.0, 13.0), role="scenario_transit")
    offset_km = float(rng.uniform(*culprit_offset_km)) * (1.0 if rng.random() < 0.5 else -1.0)
    culprit.waypoints = _transit_route(
        rng, frame, half_extent_m, t_start, t_end, culprit.speed_kn * KN_TO_MS,
        offset_m=offset_km * 1000.0, through=(origin_lon, origin_lat), through_time=t_org,
    )
    if with_gap:
        gap_min = float(rng.uniform(*culprit_gap_minutes))
        gap_start = t_org - gap_min * 60.0 * float(rng.uniform(0.35, 0.65))
        culprit.gap = (gap_start, gap_start + gap_min * 60.0)
    plans.append(culprit)

    # -- competitive distractors, deliberately not easy --------------------
    near_fisher = make("fishing", "loiter", (2.5, 5.5), role="distractor_close_wrong_type")
    ox, oy = frame.to_m(np.array([origin_lon]), np.array([origin_lat]))
    near_fisher.waypoints = _loiter_route(
        rng, frame, half_extent_m, t_start, t_end, near_fisher.speed_kn * KN_TO_MS,
        centre_m=(float(ox[0]) + rng.uniform(-800, 800), float(oy[0]) + rng.uniform(-800, 800)),
        radius_m=2500.0,
    )
    plans.append(near_fisher)

    mid_tanker = make("tanker", "transit", (10.0, 13.0), role="distractor_mid_range_tanker")
    mid_tanker.waypoints = _transit_route(
        rng, frame, half_extent_m, t_start, t_end, mid_tanker.speed_kn * KN_TO_MS,
        offset_m=float(rng.uniform(6.0, 9.0)) * 1000.0,
        through=(origin_lon, origin_lat),
        through_time=t_org + float(rng.uniform(-1.5, 1.5)) * 3600.0,
    )
    plans.append(mid_tanker)

    far_cargo = make("bulk_cargo", "transit", (10.0, 14.0), role="distractor_far_cargo")
    far_cargo.waypoints = _transit_route(
        rng, frame, half_extent_m, t_start, t_end, far_cargo.speed_kn * KN_TO_MS,
        offset_m=float(rng.uniform(14.0, 22.0)) * 1000.0,
        through=(origin_lon, origin_lat), through_time=t_org,
    )
    plans.append(far_cargo)

    # -- ordinary background traffic ---------------------------------------
    buckets: List[Tuple[str, Tuple[float, float], str]] = []
    for bucket, weight, speeds, behaviour in FLEET_PROFILE:
        buckets.extend([(bucket, speeds, behaviour)] * weight)
    while len(plans) < max(6, int(n_vessels)):
        bucket, speeds, behaviour = buckets[int(rng.integers(0, len(buckets)))]
        v = make(bucket, behaviour, speeds)
        if behaviour == "loiter":
            v.waypoints = _loiter_route(rng, frame, half_extent_m, t_start, t_end,
                                        v.speed_kn * KN_TO_MS)
        else:
            v.waypoints = _transit_route(rng, frame, half_extent_m, t_start, t_end,
                                         v.speed_kn * KN_TO_MS)
        # A minority of background vessels also drop out, so a gap alone is not
        # a fingerprint of the scenario vessel.
        if rng.random() < 0.18:
            g0 = float(rng.uniform(t_start, t_end - 3600))
            v.gap = (g0, g0 + float(rng.uniform(25, 90)) * 60.0)
        plans.append(v)

    rows: List[Dict[str, Any]] = []
    for p in plans:
        rows.extend(_sample_route(p, step_seconds, t_start, t_end, rng))
    rows.sort(key=lambda r: (r["MMSI"], r["BaseDateTime"]))

    truth = {
        "note": "Ground truth of the simulation. Never read by detection, drift or scoring.",
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": int(config.RANDOM_SEED if seed is None else seed),
        "bbox": [w, s, e, n],
        "t_start": ingest.iso(int(t_start)),
        "t_end": ingest.iso(int(t_end)),
        "origin_used": {"lon": origin_lon, "lat": origin_lat, "t": t_o.isoformat()},
        "vessels": [
            {
                "mmsi": p.mmsi, "name": p.name, "bucket": p.bucket,
                "ais_type": p.ais_type, "role": p.role,
                "speed_kn": round(p.speed_kn, 2),
                "gap_minutes": None if not p.gap else round((p.gap[1] - p.gap[0]) / 60.0, 1),
            }
            for p in plans
        ],
        "scenario_vessel_mmsi": culprit.mmsi,
        "scenario_offset_km": round(abs(offset_km), 3),
    }
    return {"rows": rows, "ground_truth": truth, "plans": plans}


def write(rows: List[Dict[str, Any]], truth: Dict[str, Any],
          csv_path: Path = None, sqlite_path: Path = None,
          truth_path: Path = None, replace: bool = True) -> Dict[str, Any]:
    """Write the CSV, load it into SQLite, and store the ground truth aside."""
    import csv as _csv

    csv_path = Path(csv_path or (config.AIS_DIR / "simulated_ais.csv"))
    truth_path = Path(truth_path or (config.AIS_DIR / "ground_truth.json"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=ingest.COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    conn = ingest.connect(sqlite_path)
    try:
        if replace:
            ingest.clear_source(conn, SOURCE_TAG)
        n = ingest.ingest_csv(csv_path, conn=conn, source=SOURCE_TAG)
        st = ingest.stats(conn).to_dict()
    finally:
        conn.close()

    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "rows_ingested": n, "store": st,
            "ground_truth": str(truth_path)}
