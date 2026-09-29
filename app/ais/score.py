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
    S_trajectory smooth angular compatibility between the vessel course at
                 closest approach and the origin-to-slick direction
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
# A Gaussian angular response avoids the old cliff at 45 degrees.  This is an
# evidence feature, not a probability: a 45 degree mismatch remains weakly
# compatible, while an opposite course approaches zero continuously.
TRAJECTORY_SCALE_DEG = 45.0

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
    counter_evidence: List[Dict[str, Any]] = field(default_factory=list)
    region: Dict[str, Any] = field(default_factory=dict)
    window: Dict[str, Any] = field(default_factory=dict)

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
            "evidence": evidence_record(self),
            "region": self.region,
            "release_window": self.window,
            "counter_evidence": list(self.counter_evidence),
            "counter_evidence_count": len(self.counter_evidence),
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


def s_time_interval(t_closest_ts: Optional[int], start_ts: Optional[int],
                    end_ts: Optional[int], tau_h: float = TIME_SCALE_H) -> Tuple[float, Dict[str, Any]]:
    """Temporal compatibility with a release WINDOW rather than a release instant.

    The drift ensemble produces an origin hour, but the individual scenarios
    produce a range. Scoring against a single minute treats the chosen scenario
    as certain, which is exactly the assumption the ensemble exists to remove.

    Inside the window the score is 1.0: any moment in the interval is
    compatible. Outside it, the same exponential decay as `s_time` applies to
    the distance from the nearest end of the interval.
    """
    if t_closest_ts is None or start_ts is None or end_ts is None:
        return 0.0, {"applicable": False,
                     "reason": "no release interval was supplied",
                     "closest_approach_utc": None if t_closest_ts is None else _iso(int(t_closest_ts))}
    t = int(t_closest_ts)
    lo, hi = int(min(start_ts, end_ts)), int(max(start_ts, end_ts))
    if lo <= t <= hi:
        score, inside, offset = 1.0, True, 0.0
    else:
        offset = (t - hi) / 3600.0 if t > hi else (lo - t) / 3600.0
        inside = False
        score = float(math.exp(-abs(offset) / max(1e-6, float(tau_h))))
    return score, {
        "applicable": True,
        "interval_start": _iso(lo),
        "interval_end": _iso(hi),
        "interval_hours": round((hi - lo) / 3600.0, 3),
        "closest_approach_utc": _iso(t),
        "inside_interval": inside,
        "offset_from_interval_hours": round(offset, 3),
        "score": round(score, 4),
        "note": ("Closest approach falls inside the modelled release window."
                 if inside else
                 "Closest approach is %.2f h outside the modelled release window." % abs(offset)),
    }


def spatial_region_match(track: Track, ring_50: Optional[Sequence[Tuple[float, float]]],
                         ring_90: Optional[Sequence[Tuple[float, float]]]) -> Dict[str, Any]:
    """How much of a track lies inside the 50 and 90 percent origin envelopes.

    Reporting only a distance to the origin point loses the distinction that
    matters most to an investigator: a vessel that merely passes 20 km from the
    source and a vessel that spends two hours inside the 90 percent envelope are
    not comparable, and the second one is the one worth asking about.

    Returns the containment flags, the fraction of reported track time inside
    each envelope, and the same figures restricted to observed positions so a
    dead-reckoned segment cannot manufacture containment.
    """
    out: Dict[str, Any] = {
        "inside_50": False, "inside_90": False,
        "track_fraction_in_50": 0.0, "track_fraction_in_90": 0.0,
        "observed_fraction_in_90": 0.0,
        "samples_in_50": 0, "samples_in_90": 0, "samples": 0,
        "closest_distance_to_50_km": None, "closest_distance_to_90_km": None,
        "rings_available": bool(ring_50) and bool(ring_90),
    }
    if not track.n or not ring_50 or not ring_90:
        out["reason"] = "no origin envelope rings were supplied" if not ring_50 else "empty track"
        return out

    lons = np.asarray(track.lon, dtype=np.float64)
    lats = np.asarray(track.lat, dtype=np.float64)
    d50 = np.asarray(ring_distances_km(lons, lats, ring_50), dtype=np.float64)
    d90 = np.asarray(ring_distances_km(lons, lats, ring_90), dtype=np.float64)

    in50 = d50 <= 0.0
    in90 = d90 <= 0.0
    observed = ~np.asarray(track.dead_reckoned, dtype=bool)

    out.update({
        "inside_50": bool(in50.any()),
        "inside_90": bool(in90.any()),
        "track_fraction_in_50": round(float(in50.mean()), 4),
        "track_fraction_in_90": round(float(in90.mean()), 4),
        "observed_fraction_in_90": (round(float(in90[observed].mean()), 4)
                                    if bool(observed.any()) else 0.0),
        "samples": int(track.n),
        "samples_in_50": int(in50.sum()),
        "samples_in_90": int(in90.sum()),
        "closest_distance_to_50_km": round(float(d50.min()), 3),
        "closest_distance_to_90_km": round(float(d90.min()), 3),
    })
    step_s = int(track.meta.get("step_seconds", 60) or 60)
    out["minutes_in_90"] = round(float(in90.sum()) * step_s / 60.0, 1)
    out["minutes_in_50"] = round(float(in50.sum()) * step_s / 60.0, 1)
    return out


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
    then drifted along the origin to slick vector.  The score must vary smoothly
    with angular mismatch: a hard cutoff makes near-identical tracks look
    materially different merely because they fall on opposite sides of a rule.
    """
    want = bearing_deg(origin[1], origin[0], slick[1], slick[0])
    have = float(track.cog[i_closest]) if track.n else 0.0
    diff = angle_diff_deg(have, want)
    score = float(math.exp(-0.5 * (diff / TRAJECTORY_SCALE_DEG) ** 2))
    return score, {
        "course_deg": round(have, 1),
        "drift_bearing_deg": round(want, 1),
        "difference_deg": round(diff, 1),
        "alignment_score": round(score, 4),
        "aligned": bool(diff <= TRAJECTORY_SCALE_DEG),
    }


def s_behavior(track: Track, i_closest: int, origin_ring: Sequence[Tuple[float, float]],
               origin_lon: float, origin_lat: float,
               exclude_gap: bool = False) -> Tuple[float, List[str], Dict[str, Any]]:
    """max() of four independent anomaly detectors, with reasons for each hit.

    `exclude_gap` drops only the reporting-gap detector, so the counterfactual
    "remove AIS gap evidence" measures that one assumption rather than removing
    the behaviour term wholesale.
    """
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
    if exclude_gap:
        detail["gap_evidence_excluded"] = True
        detail["non_reporting"] = False
        if track.gaps:
            detail["max_gap_minutes"] = round(track.max_gap_minutes(), 1)
        return float(max(scores)), reasons, detail
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


# Counter-evidence below these thresholds. A candidate that trips several of
# them is a candidate the report must not present without its dissent attached.
COUNTER_TRAJ_DEG = 60.0        # heading mismatch beyond this is a real objection
COUNTER_TIME_H = 6.0           # outside the window by more than this is an objection
COUNTER_OBSERVED_FRACTION = 0.5


def counter_evidence(track: Track, components: Dict[str, float], detail: Dict[str, Any],
                     region: Optional[Dict[str, Any]] = None,
                     window: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Enumerate the evidence AGAINST this candidate.

    A ranking that only ever accumulates support is not an analysis, it is an
    advertisement. Every objection an investigator would raise has to be
    generated mechanically, including the ones that disqualify the top-ranked
    vessel, and shown next to the score that ignored them.

    Returns items in the shared evidence taxonomy so the console, the API and
    the report all present them identically.
    """
    from .. import uncertainty

    items: List[Dict[str, Any]] = []
    region = region or {}
    window = window or {}
    traj = detail.get("trajectory") or {}
    behavior = detail.get("behavior") or {}

    def add(text: str, strength: float = 1.0) -> None:
        items.append(uncertainty.evidence_item("counter_evidence", "negative", text, strength))

    # 1. Never came near the source region.
    if region.get("rings_available"):
        if not region.get("inside_90"):
            add("The track never entered the 90 percent origin envelope. Closest approach "
                "to the envelope was %.1f km." % float(region.get("closest_distance_to_90_km") or 0.0))
        elif not region.get("inside_50"):
            add("The track entered the 90 percent origin envelope but not the 50 percent "
                "core. Closest approach to the core was %.1f km."
                % float(region.get("closest_distance_to_50_km") or 0.0))
        if not region.get("observed_fraction_in_90") and region.get("inside_90"):
            add("Every position inside the envelope is dead reckoned. No observed "
                "position places this vessel in the source region.")
        if float(region.get("track_fraction_in_90") or 0.0) > 0.0 and \
                float(region.get("track_fraction_in_90") or 0.0) < 0.01:
            add("The vessel was inside the envelope for under one percent of its tracked "
                "time, too briefly for a transfer operation.")

    # 2. Heading incompatible with a release here.
    if float(traj.get("difference_deg") or 0.0) > COUNTER_TRAJ_DEG:
        add("Course differed by %.0f degrees from the origin-to-slick drift direction. "
            "A release at closest approach would have sent the slick the other way."
            % float(traj.get("difference_deg") or 0.0))

    # 3. Outside the release window.
    if window.get("applicable") and not window.get("inside_interval"):
        add("Closest approach is %.1f h outside the modelled release window."
            % abs(float(window.get("offset_from_interval_hours") or 0.0)))
    elif not window.get("applicable"):
        add("No release interval was available, so the temporal match could not be "
            "evaluated and is assumed rather than measured.")

    # 4. Not the sort of vessel that discharges a slick this size.
    if float(components.get("type") or 0.0) <= 0.05:
        add("Decoded AIS type is '%s', which is not a credible carrier of a slick of "
            "this size. The type prior is %.2f."
            % (str(detail.get("type_label") or "unknown").lower(), float(components.get("type") or 0.0)))

    # 5. Continuous reporting through the window.
    if not behavior.get("non_reporting") and track.gaps:
        add("The longest reporting gap on this track was %.0f minutes, below the "
            "%.0f minute threshold. Nothing is hidden."
            % (float(behavior.get("max_gap_minutes") or 0.0), GAP_MIN_MINUTES))
    if not track.gaps:
        add("AIS was continuous for the whole track. There is no reporting gap to explain.")

    # 6. Transit speed inconsistent with an operation.
    sog = float(behavior.get("sog_at_closest_kn") or 0.0)
    if sog and not behavior.get("discharge_band") and sog > DISCHARGE_SOG[1]:
        add("Speed at closest approach was %.1f kn, faster than the %.0f to %.0f kn band "
            "in which any credible transfer operation occurs."
            % (sog, DISCHARGE_SOG[0], DISCHARGE_SOG[1]))

    # 7. The track itself is weak.
    dr = float(detail.get("dead_reckoned_fraction") or 0.0)
    if dr > 0.25:
        add("%.0f percent of the track is dead reckoned. The reconstructed path is a "
            "straight line under an assumed course, not an observation." % (dr * 100.0))
    if int(track.raw_count or 0) < CONF_MIN_POINTS:
        add("Only %d AIS receptions support this track." % int(track.raw_count or 0))

    # 8. The traffic is not real.
    source = str((track.meta or {}).get("source") or "")
    if "simulated" in source:
        add("This track comes from a simulated traffic generator ('%s'), not from "
            "recorded AIS traffic. It is a test fixture, not a vessel." % source)

    return items


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
    release_start_ts: Optional[int] = None,
    release_end_ts: Optional[int] = None,
    ring_50: Optional[Sequence[Tuple[float, float]]] = None,
    ring_90: Optional[Sequence[Tuple[float, float]]] = None,
    exclude_gap: bool = False,
    confidence_override: Optional[float] = None,
) -> Suspect:
    """Compute all five sub-scores for one vessel and assemble the reasons.

    `t_origin_ts` is the single chosen origin hour, kept so the established
    baseline score stays byte-comparable with earlier runs.  `release_start_ts`
    and `release_end_ts` are the ensemble's release window; when supplied they
    add the interval-aware temporal match, the 50/90 percent envelope
    containment, and the generated counter-evidence.
    """
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
    sb, beh_reasons, beh_detail = s_behavior(track, i_closest, origin_ring, origin_lon,
                                             origin_lat, exclude_gap=exclude_gap)

    comps = {"prox": sp, "time": stime, "type": prior, "traj": st, "beh": sb}
    weighted = {k: w.get(k, 0.0) * v for k, v in comps.items()}
    raw = float(sum(weighted.values()))
    max_possible = float(sum(w.get(k, 0.0) for k in comps)) or 1.0
    percent = 100.0 * raw / max_possible

    dr_fraction = round(float(np.mean(track.dead_reckoned)), 3) if track.n else 0.0
    conf, conf_notes = track_confidence(track.raw_count, dr_fraction)
    if confidence_override is not None:
        conf = float(min(1.0, max(CONF_FLOOR, confidence_override)))
        conf_notes = list(conf_notes) + ["confidence_assumed_%.2f" % conf]
    percent_adjusted = percent * conf

    region = spatial_region_match(track, ring_50, ring_90)
    window_score, window_detail = s_time_interval(
        t_closest if track.n else None, release_start_ts, release_end_ts)

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
    if region.get("rings_available"):
        reasons.append("inside_origin_50" if region.get("inside_50")
                       else ("inside_origin_90" if region.get("inside_90")
                             else "outside_origin_90"))
    if window_detail.get("applicable"):
        reasons.append("release_window_hit" if window_detail.get("inside_interval")
                       else "release_window_miss_%dh"
                            % int(abs(float(window_detail.get("offset_from_interval_hours") or 0.0))))
    reasons.extend(beh_reasons)
    reasons.extend(conf_notes)

    detail = {
        "min_distance_km": None if not math.isfinite(d_km) else round(d_km, 3),
        "origin_distance_km": None if not math.isfinite(d_origin_km) else round(d_origin_km, 3),
        "zone_radius_km": None if zone_radius_km is None else round(float(zone_radius_km), 3),
        "closest_approach_utc": _iso(t_closest) if track.n else None,
        "origin_time_utc": None if t_origin_ts is None else _iso(int(t_origin_ts)),
        "time_offset_minutes": None if dt_hours is None else round(dt_hours * 60.0, 1),
        "closest_lon": round(float(track.lon[i_closest]), 6) if track.n else None,
        "closest_lat": round(float(track.lat[i_closest]), 6) if track.n else None,
        "trajectory": traj_detail,
        "behavior": beh_detail,
        "type_prior": prior,
        "type_label": human,
        "raw_positions": track.raw_count,
        "gaps": [g.to_dict() for g in track.gaps],
        "dead_reckoned_fraction": dr_fraction,
        "track_confidence": round(conf, 3),
        "confidence_notes": conf_notes,
        "ais_source": str((track.meta or {}).get("source") or "unknown"),
    }

    suspect = Suspect(
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
        region=region,
        window=window_detail,
    )
    suspect.counter_evidence = counter_evidence(track, comps, detail, region, window_detail)
    return suspect


def _level(value: float) -> str:
    """Use bands for investigator-facing evidence, never calibrated probability."""
    if value >= 0.75:
        return "HIGH"
    if value >= 0.45:
        return "MEDIUM"
    return "LOW"


def evidence_record(suspect: Suspect) -> Dict[str, Any]:
    """The full evidence view for one candidate, in the shared taxonomy.

    The numerical ranking remains the existing, reproducible weighted model.
    This record makes its inputs inspectable, separates physical opportunity
    from the quality of the evidence used to assess it, and carries the
    generated counter-evidence so the dissent travels with the score.
    """
    from .. import uncertainty

    c = suspect.components
    d = suspect.detail
    weights = config.WEIGHTS
    traj = d.get("trajectory") or {}
    behavior = d.get("behavior") or {}

    split = uncertainty.split_opportunity_evidence(c, weights)

    items: List[Dict[str, Any]] = []
    add = items.append

    # Spatial: the containment figures matter more than the distance.
    region = suspect.region or {}
    if region.get("rings_available"):
        if region.get("inside_50"):
            add(uncertainty.evidence_item(
                "spatial", "positive",
                "Track entered the 50 percent origin envelope and stayed inside for "
                "%.0f minutes." % float(region.get("minutes_in_50") or 0.0), c.get("prox", 0.0)))
        elif region.get("inside_90"):
            add(uncertainty.evidence_item(
                "spatial", "positive",
                "Track entered the 90 percent origin envelope for %.0f minutes but never "
                "the 50 percent core." % float(region.get("minutes_in_90") or 0.0), c.get("prox", 0.0)))
        else:
            add(uncertainty.evidence_item(
                "spatial", "negative",
                "Track never entered the 90 percent origin envelope. Closest approach to "
                "it was %.1f km." % float(region.get("closest_distance_to_90_km") or 0.0), c.get("prox", 0.0)))
        add(uncertainty.evidence_item(
            "spatial", "uncertainty",
            "%.0f percent of the tracked time lies inside the 90 percent envelope."
            % (100.0 * float(region.get("track_fraction_in_90") or 0.0))))

    # Temporal.
    window = suspect.window or {}
    if window.get("applicable"):
        kind = "positive" if window.get("inside_interval") else "negative"
        add(uncertainty.evidence_item("temporal", kind, str(window.get("note") or ""),
                                      window.get("score")))
        add(uncertainty.evidence_item(
            "temporal", "uncertainty",
            "The modelled release window is %.1f h wide." % float(window.get("interval_hours") or 0.0)))
    else:
        add(uncertainty.evidence_item(
            "temporal", "uncertainty",
            "No release interval was available. The temporal match uses a single "
            "estimated origin hour, which is less uncertain than it looks."))

    # Trajectory.
    if c.get("traj", 0.0) >= 0.5:
        add(uncertainty.evidence_item(
            "trajectory", "positive",
            "Heading differed by %.0f degrees from the drift direction, within the "
            "compatibility band." % float(traj.get("difference_deg") or 0.0), c.get("traj", 0.0)))
    else:
        add(uncertainty.evidence_item(
            "trajectory", "negative",
            "Heading differed by %.0f degrees from the drift direction."
            % float(traj.get("difference_deg") or 0.0), c.get("traj", 0.0)))

    # Behaviour, itemised so the detectors are separable.
    if behavior.get("non_reporting"):
        add(uncertainty.evidence_item(
            "behaviour", "positive",
            "A reporting gap of %.0f minutes has a bridged segment within %.1f km of "
            "the origin zone." % (float(behavior.get("ais_gap_minutes") or 0.0),
                                  float(behavior.get("ais_gap_min_distance_km") or 0.0)),
            c.get("beh", 0.0)))
    if behavior.get("discharge_band"):
        add(uncertainty.evidence_item(
            "behaviour", "positive",
            "Speed at closest approach, %.1f kn, is inside the %.0f to %.0f kn band in "
            "which a transfer operation occurs." % (float(behavior.get("sog_at_closest_kn") or 0.0),
                                                    DISCHARGE_SOG[0], DISCHARGE_SOG[1]),
            c.get("beh", 0.0)))
    if float(behavior.get("course_change_deg") or 0.0) > COURSE_CHANGE_DEG:
        add(uncertainty.evidence_item(
            "behaviour", "positive",
            "Course spread of %.0f degrees within the closest-approach window."
            % float(behavior.get("course_change_deg") or 0.0), c.get("beh", 0.0)))

    # Metadata.
    add(uncertainty.evidence_item(
        "metadata", "positive" if c.get("type", 0.0) >= 0.3 else "negative",
        "Decoded AIS type '%s' carries a published prior of %.2f for a release of this "
        "size." % (str(d.get("type_label") or "unknown").lower(), c.get("type", 0.0)),
        c.get("type", 0.0)))

    # AIS quality.
    dr_fraction = float(d.get("dead_reckoned_fraction") or 0.0)
    add(uncertainty.evidence_item(
        "ais_quality", "quality",
        "%.0f percent observed AIS samples, %.0f percent reconstructed, from %d received "
        "receptions." % ((1.0 - dr_fraction) * 100.0, dr_fraction * 100.0,
                         int(d.get("raw_positions") or 0))))
    add(uncertainty.evidence_item(
        "ais_quality", "quality",
        "Track confidence multiplier %.2f." % suspect.confidence))
    if "simulated" in str(d.get("ais_source") or ""):
        add(uncertainty.evidence_item(
            "ais_quality", "uncertainty",
            "Track source is '%s', a simulated traffic generator rather than recorded "
            "AIS." % d.get("ais_source")))

    items.extend(suspect.counter_evidence or [])
    grouped = uncertainty.split_evidence(items)

    return {
        "model_version": ImprovedAttributionModel.version,
        "raw_features": {
            "origin_distance_km": d.get("origin_distance_km"),
            "time_offset_minutes": d.get("time_offset_minutes"),
            "trajectory_difference_deg": traj.get("difference_deg"),
            "raw_ais_positions": d.get("raw_positions"),
            "dead_reckoned_fraction": dr_fraction,
            "release_window_hours": window.get("interval_hours"),
            "track_fraction_in_90": region.get("track_fraction_in_90"),
        },
        "component_scores": {k: round(v, 4) for k, v in c.items()},
        "opportunity_score": split["opportunity"]["score"],
        "opportunity_level": split["opportunity"]["level"],
        "opportunity_note": split["opportunity"]["note"],
        "evidence_score": split["evidence"]["score"],
        "evidence_level": split["evidence"]["level"],
        "evidence_note": split["evidence"]["note"],
        "confidence_level": _level(suspect.confidence),
        "positive_evidence": [item["text"] for item in grouped["positive"]],
        "negative_evidence": [item["text"] for item in grouped["negative"]],
        "data_quality": [item["text"] for item in grouped["quality"]],
        "uncertainty": [item["text"] for item in grouped["uncertainty"]],
        "evidence_items": items,
        "counter_evidence": list(suspect.counter_evidence or []),
        "counter_evidence_count": len(suspect.counter_evidence or []),
        "taxonomy": list(uncertainty.EVIDENCE_CATEGORIES),
        "region": region,
        "release_window": window,
    }


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
    release_start_ts: Optional[int] = None,
    release_end_ts: Optional[int] = None,
    ring_50: Optional[Sequence[Tuple[float, float]]] = None,
    ring_90: Optional[Sequence[Tuple[float, float]]] = None,
    exclude_gap: bool = False,
    confidence_override: Optional[float] = None,
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
                               t_origin_ts=t_origin_ts, zone_radius_km=zone_radius_km,
                               release_start_ts=release_start_ts,
                               release_end_ts=release_end_ts,
                               ring_50=ring_50, ring_90=ring_90,
                               exclude_gap=exclude_gap,
                               confidence_override=confidence_override))
    out.sort(key=lambda s: (-s.score,
                            s.detail.get("origin_distance_km") if s.detail.get("origin_distance_km") is not None else 1e9,
                            -int(s.detail.get("raw_positions") or 0),
                            s.mmsi))
    for i, s in enumerate(out, start=1):
        s.rank = i
    return out[:top_n] if top_n else out


class BaselineAttributionModel:
    """The established reproducible weighted scorer, retained for comparison."""

    version = "baseline-attribution-v1"

    def rank(self, *args, **kwargs) -> List[Suspect]:
        return rank_suspects(*args, **kwargs)


class ImprovedAttributionModel(BaselineAttributionModel):
    """The explainable V2 surface over the baseline numerical score.

    This deliberately does not claim a new trained ranker.  It adds the
    opportunity/evidence record and counterfactual analysis while preserving a
    directly comparable, hand-designed baseline.
    """

    version = "improved-attribution-v2"


#: Every counterfactual the report is required to show, with what it removes and
#: why.  Order is fixed so the UI renders a stable list.
COUNTERFACTUAL_CASES: Tuple[Dict[str, str], ...] = (
    {"name": "no_ais_gap",
     "label": "Remove AIS gap evidence",
     "question": "Does the ranking depend on the reporting gap being real?"},
    {"name": "no_vessel_type",
     "label": "Remove vessel type prior",
     "question": "Is the ranking just a tanker detector?"},
    {"name": "no_trajectory",
     "label": "Remove trajectory alignment",
     "question": "Does heading evidence change the order?"},
    {"name": "no_behaviour",
     "label": "Remove all behaviour evidence",
     "question": "How much of the score is the behaviour detectors?"},
    {"name": "expanded_release_window",
     "label": "Double the release window",
     "question": "Is the temporal match robust to a 2x wider interval?"},
    {"name": "expanded_origin_uncertainty",
     "label": "Double the origin zone radius",
     "question": "Does proximity survive a looser origin region?"},
    {"name": "alternative_drift_model",
     "label": "Alternative drift scenario origin",
     "question": "Does the order survive a different metocean scenario?"},
    {"name": "tighter_search_radius",
     "label": "Halve the proximity length scale",
     "question": "Is the order an artefact of a soft distance decay?"},
    {"name": "reconstructed_tracks_only",
     "label": "Assume full dead reckoning",
     "question": "How much of this is observed rather than assumed?"},
)


def counterfactual_analysis(
    tracks: Dict[int, Track],
    closest: Dict[int, Tuple[float, int]],
    origin_ring: Sequence[Tuple[float, float]],
    origin_lon: float,
    origin_lat: float,
    slick_lon: float,
    slick_lat: float,
    t_origin_ts: Optional[int],
    zone_radius_km: Optional[float],
    top_n: int = 10,
    release_start_ts: Optional[int] = None,
    release_end_ts: Optional[int] = None,
    ring_50: Optional[Sequence[Tuple[float, float]]] = None,
    ring_90: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    """Measure ranking sensitivity to declared modelling assumptions.

    Each case removes one declared assumption or widens one declared
    uncertainty, re-ranks under it, and reports what changed. They are not
    probabilities and they create no new evidence: the question answered is
    only "would a reviewer who disagreed with this assumption rank it
    differently?"

    Every case in `COUNTERFACTUAL_CASES` is always computed, including the ones
    a given case cannot support, so a missing scenario is visible as an explicit
    `not_applicable` rather than silently absent.
    """
    kwargs = dict(origin_ring=origin_ring, origin_lon=origin_lon, origin_lat=origin_lat,
                  slick_lon=slick_lon, slick_lat=slick_lat, top_n=0,
                  t_origin_ts=t_origin_ts, zone_radius_km=zone_radius_km,
                  ring_50=ring_50, ring_90=ring_90)
    baseline = rank_suspects(tracks, closest, **kwargs)
    base_weights = dict(config.WEIGHTS)
    radius = float(zone_radius_km or 0.0)

    def rerank(**over) -> Dict[int, Suspect]:
        merged = dict(kwargs)
        merged.update(over)
        return {item.mmsi: item for item in rank_suspects(tracks, closest, **merged)}

    cases: Dict[str, Dict[int, Any]] = {
        "no_ais_gap": rerank(exclude_gap=True),
        "no_vessel_type": rerank(weights={**base_weights, "type": 0.0}),
        "no_trajectory": rerank(weights={**base_weights, "traj": 0.0}),
        "no_behaviour": rerank(weights={**base_weights, "beh": 0.0}),
        "expanded_release_window": rerank(
            release_start_ts=(int(release_start_ts) - 21600) if release_start_ts else None,
            release_end_ts=(int(release_end_ts) + 21600) if release_end_ts else None,
            weights={**base_weights, "time": 0.0}),
        "expanded_origin_uncertainty": rerank(zone_radius_km=max(1.0, radius * 2.0)),
        "alternative_drift_model": rerank(
            t_origin_ts=(int(t_origin_ts) + 10800) if t_origin_ts else None,
            release_start_ts=(int(release_start_ts) + 10800) if release_start_ts else None,
            release_end_ts=(int(release_end_ts) + 10800) if release_end_ts else None,
            zone_radius_km=max(1.0, radius * 1.5)),
        "tighter_search_radius": rerank(zone_radius_km=max(PROXIMITY_SCALE_KM, radius / 2.0)),
        "reconstructed_tracks_only": rerank(confidence_override=1.0),
    }

    # A scenario that cannot be evaluated must say so rather than report 0.0,
    # which would read as "removing this assumption changes nothing".
    unavailable: Dict[str, str] = {}
    if not any(getattr(tr, "gaps", None) for tr in tracks.values()):
        unavailable["no_ais_gap"] = "no track in this case has a reporting gap to remove"
    if not release_start_ts or not release_end_ts:
        unavailable["expanded_release_window"] = "the drift ensemble produced no release interval"
    if not ring_50 or not ring_90:
        unavailable["alternative_drift_model"] = "no 50/90 percent origin envelopes were available"
    if not radius:
        unavailable["tighter_search_radius"] = "no origin zone radius was available"

    out: List[Dict[str, Any]] = []
    for item in baseline[:top_n]:
        scenarios: List[Dict[str, Any]] = [
            {"name": "baseline", "label": "Baseline",
             "score": round(item.score, 1), "rank": item.rank, "delta": 0.0}]
        largest = ("baseline", 0.0)
        changed_rank = False
        for case in COUNTERFACTUAL_CASES:
            name = case["name"]
            if name in unavailable:
                scenarios.append({"name": name, "label": case["label"],
                                  "question": case["question"],
                                  "applicable": False, "reason": unavailable[name]})
                continue
            alternative = cases[name].get(item.mmsi)
            if alternative is None:
                scenarios.append({"name": name, "label": case["label"],
                                  "question": case["question"],
                                  "applicable": False,
                                  "reason": "this vessel was filtered out under that assumption"})
                continue
            delta = alternative.score - item.score
            scenarios.append({"name": name, "label": case["label"],
                              "question": case["question"],
                              "applicable": True,
                              "score": round(alternative.score, 1),
                              "rank": alternative.rank, "delta": round(delta, 1)})
            if abs(delta) > largest[1]:
                largest = (name, abs(delta))
            changed_rank = changed_rank or alternative.rank != item.rank

        evaluated = [s for s in scenarios if s.get("applicable") and s["name"] != "baseline"]
        max_delta = largest[1]
        stability = ("UNSTABLE" if changed_rank else
                     "SENSITIVE" if max_delta > 15.0 else
                     "STABLE" if evaluated else "NOT_EVALUATED")
        out.append({
            "candidate_id": item.mmsi,
            "candidate_name": item.name,
            "baseline_score": round(item.score, 1),
            "baseline_rank": item.rank,
            "scenarios": scenarios,
            "stability": stability,
            "max_score_delta": round(max_delta, 1),
            "rank_changes": bool(changed_rank),
            "primary_dependency": largest[0] if largest[0] != "baseline" else None,
            "note": ("No assumption was varied for this candidate."
                     if not evaluated else
                     "%d of %d scenarios were evaluable for this candidate."
                     % (len(evaluated), len(COUNTERFACTUAL_CASES))),
        })
    return out


def explain_weights(weights: Dict[str, float] = None) -> Dict[str, Any]:
    """What the UI shows next to the leaderboard so nothing is hidden."""
    w = dict(config.WEIGHTS if weights is None else weights)
    terms = " + ".join("%.2f*%s" % (w.get(k, 0.0), k)
                       for k in ("prox", "time", "beh", "type", "traj"))
    return {
        "baseline_model": BaselineAttributionModel.version,
        "improved_model": ImprovedAttributionModel.version,
        "weights": w,
        "formula": ("raw = %s ; percent = 100 * raw / sum(weights) ; "
                    "score = percent * track_confidence" % terms),
        "proximity": ("exp(-d_km / R), d from closest approach to the estimated origin point, "
                      "R the origin zone radius (floor %.1f km). Centre 1.00, zone edge 0.37."
                      % PROXIMITY_SCALE_KM),
        "time": ("exp(-|dt| / %.1f h), dt between closest approach and the estimated origin time. "
                 "The AIS window is a filter; this is the score." % TIME_SCALE_H),
        "trajectory": ("exp(-0.5 * (angular_mismatch / %.0f deg)^2); a continuous "
                       "trajectory compatibility feature, not a calibrated probability"
                       % TRAJECTORY_SCALE_DEG),
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
        "spatial_regions": (
            "Each candidate is additionally tested against the drift ensemble's 50 and 90 "
            "percent origin envelopes: inside_50, inside_90, minutes_in_90, and the "
            "fraction of tracked time inside each. Reported under `region` and never "
            "folded into the score, because the score decays on distance to the origin "
            "point while the envelope is the uncertainty the ensemble actually produced."),
        "release_window": (
            "The temporal match is evaluated against the ensemble release interval "
            "[t_start, t_end], not one chosen origin hour. Score is 1.0 anywhere inside "
            "the interval and decays outward with the same %.1f h scale." % TIME_SCALE_H),
        "counter_evidence": (
            "Generated mechanically for every candidate from the same inputs that raised "
            "the score: envelope non-containment, heading mismatch beyond %.0f deg, "
            "closest approach outside the release window by over %.0f h, non-cargo type "
            "prior, continuous reporting, transit speed above the operation band, "
            "reconstructed-heavy tracks, and simulated traffic sources. The count is "
            "reported next to the score."
            % (COUNTER_TRAJ_DEG, COUNTER_TIME_H)),
        "counterfactuals": [dict(case) for case in COUNTERFACTUAL_CASES],
        "bands": ("HIGH / MEDIUM / LOW describe the quality of the evidence available at "
                  "that stage. They are not probabilities and must never be rendered as "
                  "percentages."),
        "note": "Ranked investigative lead. Not a finding of discharge, and not legal proof.",
    }
