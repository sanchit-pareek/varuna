"""Explainable suspicion scoring.

Five sub-scores, each in [0, 1], combined with absolute weights and reported as
a percentage of the maximum possible score. The weighting is never min-max
across the batch, because a relative scale would hand the top slot to somebody
on every scene including a clean one.

    S_proximity  exp(-d_km / R), d from closest approach to the estimated
                 ORIGIN POINT, R the origin zone radius. A vessel at the centre
                 of the zone scores 1.0, one on the zone edge 0.37, one at twice
                 the radius 0.13.
    S_time       exp(-|dt|_h / TAU), dt between closest approach and the
                 estimated origin TIME.
    S_trajectory 1.0 if the vessel course at closest approach lines up with the
                 origin to slick direction within 45 degrees, else 0.4
    S_type       the published prior for the decoded AIS vessel type
    S_behavior   max of four anomaly detectors, one of which is the AIS gap

    raw     = 0.30*prox + 0.20*time + 0.25*beh + 0.15*type + 0.10*traj
    percent = 100 * raw / sum(weights) * track_confidence

Two design decisions are worth stating, because both were bugs before.

Proximity is measured to the origin POINT, not to the zone boundary. Measuring
to the boundary and clamping to zero inside made the heaviest weight in the
model a constant: the origin zone is routinely 200 km2, every candidate passes
through it at some point in a six hour window, and so every candidate scored
exactly 1.0. Distance to the point, scaled by the zone radius, keeps the zone
meaningful and restores a gradient.

Time is scored, not merely filtered. The problem statement asks for
spatio-temporal correlation; a plus or minus three hour window with no temporal
term treats a vessel present three hours early exactly like one present at the
estimated origin minute.

Every component, every reason code and the track confidence that scaled the
result are returned, so the leaderboard can be audited line by line instead of
trusted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config
from ..geo.crs import angle_diff_deg, bearing_deg, haversine_km
from ..geo.geometry import ring_distances_km
from . import vessel_types
from .interpolate import Track

PROXIMITY_SCALE_KM = 3.0       # floor for the proximity length scale
TIME_SCALE_H = 1.5             # tau for the temporal term, half the default window
TRAJECTORY_TOLERANCE_DEG = 45.0
TRAJECTORY_MISS = 0.4

BEH_DISCHARGE = 0.70
BEH_COURSE_CHANGE = 0.60
BEH_SPEED_DROP = 0.50
BEH_AIS_GAP = 0.95
BEH_SPARSE_GAP = 0.35          # a gap we cannot distinguish from thin coverage
BEH_BASELINE = 0.10

GAP_MIN_MINUTES = 30.0
GAP_NEAR_KM = 5.0
# A vessel cannot be said to have "gone dark" on the strength of a track that
# barely exists. Below this many genuine receptions the silence is indistinct
# from ordinary sparse terrestrial AIS coverage, so the gap is reported but
# scored down instead of paying the full evasion bonus.
GAP_MIN_RAW_POINTS = 6
DISCHARGE_SOG = (8.0, 16.0)
COURSE_CHANGE_DEG = 30.0
COURSE_WINDOW_MIN = 20.0
SPEED_DROP_KN = 5.0

# Track confidence. Every reported position that was bridged by dead reckoning
# rather than received is an assumption, and a ranking that ignores this puts a
# vessel seen twice above one seen six hundred times. Confidence multiplies the
# final percentage and is reported next to it.
CONF_FLOOR = 0.35
CONF_DR_PENALTY = 0.65         # full penalty at a 100 percent dead reckoned track
CONF_MIN_POINTS = 10           # below this, confidence is additionally pro-rated


@dataclass
class Suspect:
    mmsi: int
    name: Optional[str]
    vessel_type: str
    vessel_type_raw: Optional[str]
    score: float                     # 0 to 100, after the confidence scaling
    score_raw_evidence: float        # 0 to 100, before the confidence scaling
    confidence: float                # 0 to 1, how much track there is to judge
    raw: float                       # 0 to 1
    components: Dict[str, float]
    weighted: Dict[str, float]
    reasons: List[str]
    detail: Dict[str, Any]
    rank: int = 0
    track: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, with_track: bool = True) -> Dict[str, Any]:
        d = {
            "rank": self.rank,
            "mmsi": self.mmsi,
            "name": self.name or "UNKNOWN",
            "type": self.vessel_type,
            "type_raw": self.vessel_type_raw,
            "score": round(self.score, 1),
            "score_before_confidence": round(self.score_raw_evidence, 1),
            "confidence": round(self.confidence, 3),
            "raw": round(self.raw, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "weighted": {k: round(v, 4) for k, v in self.weighted.items()},
            "reasons": list(self.reasons),
            "detail": self.detail,
        }
        if with_track:
            d["track"] = self.track
        return d


def s_proximity(d_km: float, zone_radius_km: float = None) -> float:
    """Distance decay to the estimated origin point.

    The length scale is the origin zone radius, so the score reads against the
    uncertainty the hindcast actually produced: centre 1.0, zone edge 0.37,
    twice the radius 0.13. A tight zone therefore discriminates hard and a loose
    one discriminates gently, which is the honest behaviour.
    """
    if not math.isfinite(d_km):
        return 0.0
    scale = max(PROXIMITY_SCALE_KM, float(zone_radius_km or 0.0))
    return float(math.exp(-max(0.0, d_km) / scale))


def s_time(dt_hours: float, tau_h: float = TIME_SCALE_H) -> float:
    """Temporal decay between closest approach and the estimated origin time.

    Without this the model is spatial only: the plus or minus three hour AIS
    window is a hard filter, and a vessel at the far edge of it scores exactly
    like one sitting on the origin at the origin minute.
    """
    if dt_hours is None or not math.isfinite(dt_hours):
        return 0.0
    return float(math.exp(-abs(float(dt_hours)) / max(1e-6, float(tau_h))))


def track_confidence(raw_positions: int, dead_reckoned_fraction: float) -> Tuple[float, List[str]]:
    """How much of this track was received rather than assumed.

    Returns a multiplier in [CONF_FLOOR, 1.0] and any reason codes that explain
    a reduction. Reported separately from the evidence score so an analyst can
    see both the case against a vessel and how much track that case rests on.
    """
    notes: List[str] = []
    dr = min(1.0, max(0.0, float(dead_reckoned_fraction or 0.0)))
    conf = 1.0 - CONF_DR_PENALTY * dr
    if dr >= 0.25:
        notes.append("dead_reckoned_%d_percent" % int(round(dr * 100)))
    n = int(raw_positions or 0)
    if n < CONF_MIN_POINTS:
        conf *= max(0.2, n / float(CONF_MIN_POINTS))
        notes.append("only_%d_ais_receptions" % n)
    return float(min(1.0, max(CONF_FLOOR, conf))), notes


def s_trajectory(track: Track, i_closest: int, origin: Tuple[float, float],
                 slick: Tuple[float, float]) -> Tuple[float, Dict[str, Any]]:
    """Does the vessel's heading at closest approach match origin -> slick?

    A vessel that discharged at the origin was travelling through it. The slick
    then drifted along the origin to slick vector. A course roughly aligned with
    that vector is weak corroboration, so the miss case is 0.4 rather than 0.
    """
    want = bearing_deg(origin[1], origin[0], slick[1], slick[0])
    have = float(track.cog[i_closest]) if track.n else 0.0
    diff = angle_diff_deg(have, want)
    aligned = diff <= TRAJECTORY_TOLERANCE_DEG
    return (1.0 if aligned else TRAJECTORY_MISS), {
        "course_deg": round(have, 1),
        "drift_bearing_deg": round(want, 1),
        "difference_deg": round(diff, 1),
        "aligned": bool(aligned),
    }


def s_behavior(track: Track, i_closest: int, origin_ring: Sequence[Tuple[float, float]],
               origin_lon: float, origin_lat: float) -> Tuple[float, List[str], Dict[str, Any]]:
    """max() of four independent anomaly detectors, with reasons for each hit."""
    reasons: List[str] = []
    detail: Dict[str, Any] = {}
    scores: List[float] = [BEH_BASELINE]

    sog_c = float(track.sog[i_closest]) if track.n else 0.0
    detail["sog_at_closest_kn"] = round(sog_c, 2)

    # 1. Operational discharge speed band
    if DISCHARGE_SOG[0] <= sog_c <= DISCHARGE_SOG[1]:
        scores.append(BEH_DISCHARGE)
        reasons.append("sog_%.1fkn_in_discharge_band" % sog_c)
        detail["discharge_band"] = True

    # 2. Course change around closest approach
    step = max(1, int(track.meta.get("step_seconds", 60)))
    half = max(1, int(COURSE_WINDOW_MIN * 60 / step))
    a = max(0, i_closest - half)
    b = min(track.n - 1, i_closest + half)
    if b > a:
        seg = track.cog[a:b + 1]
        spread = _max_circular_spread(seg)
        detail["course_change_deg"] = round(spread, 1)
        if spread > COURSE_CHANGE_DEG:
            scores.append(BEH_COURSE_CHANGE)
            reasons.append("course_change_%.0fdeg" % spread)

    # 3. Speed drop
    if b > a:
        seg = track.sog[a:b + 1]
        drop = float(np.max(seg) - np.min(seg))
        detail["speed_swing_kn"] = round(drop, 2)
        if drop > SPEED_DROP_KN:
            scores.append(BEH_SPEED_DROP)
            reasons.append("speed_swing_%.1fkn" % drop)

    # 4. AIS gap whose bridged segment passes near the origin zone
    gap_hit = None
    for g in track.gaps:
        if g.minutes < GAP_MIN_MINUTES:
            continue
        m = (track.ts >= g.start_ts) & (track.ts <= g.end_ts)
        if not m.any():
            continue
        if origin_ring:
            d = float(np.min(ring_distances_km(track.lon[m], track.lat[m], origin_ring)))
        else:
            d = float(np.min(haversine_km(origin_lat, origin_lon, track.lat[m], track.lon[m])))
        if d <= GAP_NEAR_KM and (gap_hit is None or g.minutes > gap_hit[0]):
            gap_hit = (g.minutes, d)
    if gap_hit is not None:
        # Gate the evasion bonus on there being a track to go dark from. On a
        # vessel seen twice in six hours the "gap" is the coverage, not the
        # conduct, and paying 0.95 for it ranks the least observed vessel first.
        sparse = int(track.raw_count or 0) < GAP_MIN_RAW_POINTS
        scores.append(BEH_SPARSE_GAP if sparse else BEH_AIS_GAP)
        if sparse:
            reasons.append("sparse_track_gap_%dmin_%d_receptions"
                           % (int(round(gap_hit[0])), int(track.raw_count or 0)))
        else:
            reasons.append("ais_gap_%dmin_within_%.1fkm"
                           % (int(round(gap_hit[0])), gap_hit[1]))
        detail["ais_gap_minutes"] = round(gap_hit[0], 1)
        detail["ais_gap_min_distance_km"] = round(gap_hit[1], 2)
        detail["ais_gap_is_sparse_coverage"] = bool(sparse)
        detail["non_reporting"] = not sparse
    else:
        detail["non_reporting"] = False
        if track.gaps:
            detail["max_gap_minutes"] = round(track.max_gap_minutes(), 1)

    return float(max(scores)), reasons, detail


def _max_circular_spread(deg: np.ndarray) -> float:
    if deg.size < 2:
        return 0.0
    ref = float(deg[0])
    rel = np.array([angle_diff_deg(float(v), ref) for v in deg])
    return float(np.max(rel))


def score_track(
    track: Track,
    d_km: float,
    i_closest: int,
    origin_ring: Sequence[Tuple[float, float]],
    origin_lon: float,
    origin_lat: float,
    slick_lon: float,
    slick_lat: float,
    weights: Dict[str, float] = None,
    track_stride: int = 5,
    t_origin_ts: Optional[int] = None,
    zone_radius_km: Optional[float] = None,
) -> Suspect:
    """Compute all five sub-scores for one vessel and assemble the reasons."""
    w = dict(config.WEIGHTS if weights is None else weights)

    bucket, prior, human = vessel_types.describe(track.vessel_type_raw)

    # Distance to the origin POINT at closest approach, which is what the
    # proximity term now decays on. d_km, the distance to the zone boundary,
    # is still reported because it is what the map shows.
    if track.n:
        d_origin_km = float(haversine_km(origin_lat, origin_lon,
                                         float(track.lat[i_closest]),
                                         float(track.lon[i_closest])))
    else:
        d_origin_km = float("inf")

    t_closest = int(track.ts[i_closest]) if track.n else 0
    if t_origin_ts is None:
        dt_hours = None
    else:
        dt_hours = (t_closest - int(t_origin_ts)) / 3600.0

    sp = s_proximity(d_origin_km, zone_radius_km)
    stime = s_time(dt_hours) if dt_hours is not None else 0.0
    st, traj_detail = s_trajectory(track, i_closest, (origin_lon, origin_lat), (slick_lon, slick_lat))
    sb, beh_reasons, beh_detail = s_behavior(track, i_closest, origin_ring, origin_lon, origin_lat)

    comps = {"prox": sp, "time": stime, "type": prior, "traj": st, "beh": sb}
    weighted = {k: w.get(k, 0.0) * v for k, v in comps.items()}
    raw = float(sum(weighted.values()))
    max_possible = float(sum(w.get(k, 0.0) for k in comps)) or 1.0
    percent = 100.0 * raw / max_possible

    dr_fraction = round(float(np.mean(track.dead_reckoned)), 3) if track.n else 0.0
    conf, conf_notes = track_confidence(track.raw_count, dr_fraction)
    percent_adjusted = percent * conf

    reasons: List[str] = ["origin_distance_%.2fkm" % d_origin_km
                          if math.isfinite(d_origin_km) else "origin_distance_unknown"]
    if dt_hours is not None:
        mins = int(round(abs(dt_hours) * 60))
        reasons.append("%s_origin_by_%dmin" % ("before" if dt_hours < 0 else "after", mins)
                       if mins else "at_origin_time")
    reasons.append("type_%s" % bucket)
    if traj_detail["aligned"]:
        reasons.append("course_aligned_%ddeg" % int(traj_detail["difference_deg"]))
    else:
        reasons.append("course_off_%ddeg" % int(traj_detail["difference_deg"]))
    reasons.extend(beh_reasons)
    reasons.extend(conf_notes)

    detail = {
        "min_distance_km": None if not math.isfinite(d_km) else round(d_km, 3),
        "origin_distance_km": None if not math.isfinite(d_origin_km) else round(d_origin_km, 3),
        "zone_radius_km": None if zone_radius_km is None else round(float(zone_radius_km), 3),
        "closest_approach_utc": _iso(t_closest),
        "origin_time_utc": None if t_origin_ts is None else _iso(int(t_origin_ts)),
        "time_offset_minutes": None if dt_hours is None else round(dt_hours * 60.0, 1),
        "closest_lon": round(float(track.lon[i_closest]), 6),
        "closest_lat": round(float(track.lat[i_closest]), 6),
        "trajectory": traj_detail,
        "behavior": beh_detail,
        "type_prior": prior,
        "type_label": human,
        "raw_positions": track.raw_count,
        "gaps": [g.to_dict() for g in track.gaps],
        "dead_reckoned_fraction": dr_fraction,
        "track_confidence": round(conf, 3),
        "confidence_notes": conf_notes,
    }

    return Suspect(
        mmsi=track.mmsi,
        name=track.name,
        vessel_type=bucket,
        vessel_type_raw=track.vessel_type_raw,
        score=percent_adjusted,
        score_raw_evidence=percent,
        confidence=conf,
        raw=raw,
        components=comps,
        weighted=weighted,
        reasons=reasons,
        detail=detail,
        track={
            "geojson": track.to_geojson(),
            "samples": track.samples(stride=track_stride),
            "closest_index": int(i_closest),
        },
    )


def _iso(ts: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rank_suspects(
    tracks: Dict[int, Track],
    closest: Dict[int, Tuple[float, int]],
    origin_ring: Sequence[Tuple[float, float]],
    origin_lon: float,
    origin_lat: float,
    slick_lon: float,
    slick_lat: float,
    weights: Dict[str, float] = None,
    top_n: int = 10,
    t_origin_ts: Optional[int] = None,
    zone_radius_km: Optional[float] = None,
) -> List[Suspect]:
    """Score every candidate and rank.

    Ranking is on the confidence-adjusted score. Ties break on distance to the
    origin point, then on how much real track there was, then on MMSI, so the
    order is total and reproducible.
    """
    out: List[Suspect] = []
    for mmsi, track in tracks.items():
        d, i = closest.get(mmsi, (float("inf"), 0))
        if not math.isfinite(d):
            continue
        out.append(score_track(track, d, i, origin_ring, origin_lon, origin_lat,
                               slick_lon, slick_lat, weights=weights,
                               t_origin_ts=t_origin_ts, zone_radius_km=zone_radius_km))
    out.sort(key=lambda s: (-s.score,
                            s.detail.get("origin_distance_km") if s.detail.get("origin_distance_km") is not None else 1e9,
                            -int(s.detail.get("raw_positions") or 0),
                            s.mmsi))
    for i, s in enumerate(out, start=1):
        s.rank = i
    return out[:top_n] if top_n else out


def explain_weights(weights: Dict[str, float] = None) -> Dict[str, Any]:
    """What the UI shows next to the leaderboard so nothing is hidden."""
    w = dict(config.WEIGHTS if weights is None else weights)
    terms = " + ".join("%.2f*%s" % (w.get(k, 0.0), k)
                       for k in ("prox", "time", "beh", "type", "traj"))
    return {
        "weights": w,
        "formula": ("raw = %s ; percent = 100 * raw / sum(weights) ; "
                    "score = percent * track_confidence" % terms),
        "proximity": ("exp(-d_km / R), d from closest approach to the estimated origin point, "
                      "R the origin zone radius (floor %.1f km). Centre 1.00, zone edge 0.37."
                      % PROXIMITY_SCALE_KM),
        "time": ("exp(-|dt| / %.1f h), dt between closest approach and the estimated origin time. "
                 "The AIS window is a filter; this is the score." % TIME_SCALE_H),
        "trajectory": "1.0 if course at closest approach is within %.0f deg of the origin to slick bearing, else %.1f"
                      % (TRAJECTORY_TOLERANCE_DEG, TRAJECTORY_MISS),
        "type_priors": vessel_types.TYPE_PRIOR,
        "behavior": {
            "operational_discharge": "%.2f if %.0f <= SOG <= %.0f kn at closest approach"
                                     % (BEH_DISCHARGE, DISCHARGE_SOG[0], DISCHARGE_SOG[1]),
            "course_change": "%.2f if course spread > %.0f deg within +/- %.0f min of closest approach"
                             % (BEH_COURSE_CHANGE, COURSE_CHANGE_DEG, COURSE_WINDOW_MIN),
            "speed_drop": "%.2f if SOG swing > %.0f kn in the same window" % (BEH_SPEED_DROP, SPEED_DROP_KN),
            "ais_gap": "%.2f if a gap >= %.0f min has a dead reckoned segment within %.0f km of the origin zone"
                       % (BEH_AIS_GAP, GAP_MIN_MINUTES, GAP_NEAR_KM),
            "sparse_gap": "%.2f instead, when the vessel has fewer than %d genuine receptions: "
                          "that silence is thin coverage, not demonstrated evasion"
                          % (BEH_SPARSE_GAP, GAP_MIN_RAW_POINTS),
            "baseline": BEH_BASELINE,
        },
        "confidence": ("multiplier in [%.2f, 1.00] applied to the final percentage: "
                       "1 - %.2f * dead_reckoned_fraction, pro-rated again below %d receptions. "
                       "A vessel seen twice cannot outrank one seen six hundred times on the "
                       "strength of the gap between those two sightings."
                       % (CONF_FLOOR, CONF_DR_PENALTY, CONF_MIN_POINTS)),
        "note": "Ranked likelihood for investigation. Not legal proof of discharge.",
    }
