"""Scoring tests.

The spec's headline requirement is that a tanker which actually transits the
origin zone at the origin time ranks first against realistic distractors. That
is asserted here, but so is the harder half: the ranking has to come out of the
scoring rule and nothing else. So the tests also check that a closer fishing
boat does not automatically lose on type alone, that a distant tanker does not
win on type alone, and that an empty candidate set produces an empty
leaderboard instead of a forced culprit.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

T_ORIGIN = datetime(2024, 2, 14, 6, 0, tzinfo=timezone.utc)
ORIGIN = (71.500, 19.000)


def _track(mmsi, name, vtype, lon_offsets, lat, sog=12.0, cog=90.0,
           step_min=1, gap_range=None, start_min=-90):
    """Build a Track directly, so scoring is tested without the database."""
    from app.ais.interpolate import Gap, Track

    n = len(lon_offsets)
    ts = np.array([int((T_ORIGIN + timedelta(minutes=start_min + i * step_min)).timestamp())
                   for i in range(n)], dtype=np.int64)
    lon = np.array([ORIGIN[0] + d for d in lon_offsets], dtype=float)
    lat_arr = np.full(n, lat, dtype=float)
    dr = np.zeros(n, dtype=bool)
    gaps = []
    if gap_range:
        a, b = gap_range
        gaps.append(Gap(start_ts=int(ts[a]), end_ts=int(ts[b]), minutes=(b - a) * step_min,
                        start_lon=float(lon[a]), start_lat=lat, end_lon=float(lon[b]), end_lat=lat))
        dr[a + 1:b] = True
    return Track(mmsi=mmsi, name=name, vessel_type_raw=vtype, ts=ts, lon=lon,
                 lat=lat_arr, sog=np.full(n, sog), cog=np.full(n, cog),
                 dead_reckoned=dr, gaps=gaps, raw_count=n,
                 meta={"step_seconds": step_min * 60})


def _ring(km=1.5):
    from app.geo.geometry import buffer_ring_km

    return buffer_ring_km([ORIGIN], km)


def _rank(tracks, slick=(71.62, 19.02), zone_radius_km=1.5, with_time=True):
    from app.ais import score as score_mod
    from app.ais.filter import _min_distance

    ring = _ring(zone_radius_km)
    closest = {}
    for mmsi, tr in tracks.items():
        closest[mmsi] = _min_distance(tr, ring, ORIGIN[0], ORIGIN[1])
    return score_mod.rank_suspects(
        tracks, closest, ring, ORIGIN[0], ORIGIN[1], slick[0], slick[1],
        t_origin_ts=int(T_ORIGIN.timestamp()) if with_time else None,
        zone_radius_km=zone_radius_km,
    )


def test_transiting_tanker_with_a_gap_ranks_first():
    """The scenario vessel wins because it scores highest, not because it is planted."""
    offsets = [-0.09 + 0.0015 * i for i in range(120)]
    tanker = _track(419000001, "MT OCEAN VOYAGER", "84", offsets, 19.0005,
                    sog=12.0, cog=90.0, gap_range=(45, 95))

    # Distractor 1: fishing boat even closer, but wrong type and loitering.
    fisher = _track(419000777, "FV NEARBY", "30",
                    [-0.004 + 0.00005 * i for i in range(120)], 19.0002,
                    sog=3.0, cog=90.0)
    # Distractor 2: cargo at about 8 km, right type-ish, no anomalies.
    far = _track(419000555, "MV DISTANT", "70",
                 [-0.09 + 0.0015 * i for i in range(120)], 19.075, sog=13.0, cog=90.0)

    tracks = {t.mmsi: t for t in (tanker, fisher, far)}
    ranked = _rank(tracks)

    assert ranked[0].mmsi == 419000001, [
        (s.rank, s.mmsi, s.name, round(s.score, 1), s.components) for s in ranked]
    assert ranked[0].score > ranked[1].score
    assert any(r.startswith("ais_gap_") for r in ranked[0].reasons)
    assert ranked[0].detail["behavior"]["non_reporting"] is True


def test_a_closer_fishing_boat_still_beats_a_distant_tanker():
    """Proximity carries the most weight, so geometry can overrule type."""
    fisher = _track(1, "FV CLOSE", "30", [0.0] * 60, 19.0, sog=4.0)
    tanker = _track(2, "MT FAR", "84", [0.45] * 60, 19.0, sog=12.0)
    ranked = _rank({1: fisher, 2: tanker})
    assert ranked[0].mmsi == 1


def test_type_prior_breaks_a_tie_at_equal_geometry():
    """Same track, different declared type: the tanker must edge ahead."""
    offsets = [-0.02 + 0.0007 * i for i in range(60)]
    tanker = _track(11, "MT A", "84", offsets, 19.0)
    passenger = _track(22, "MV B", "60", offsets, 19.0)
    ranked = _rank({11: tanker, 22: passenger})
    assert ranked[0].mmsi == 11
    assert ranked[0].components["prox"] == pytest.approx(ranked[1].components["prox"], abs=1e-6)
    assert ranked[0].components["type"] > ranked[1].components["type"]


def test_no_candidates_gives_an_empty_leaderboard():
    from app.ais import score as score_mod

    assert score_mod.rank_suspects({}, {}, _ring(), ORIGIN[0], ORIGIN[1],
                                   71.6, 19.0) == []


def test_score_is_absolute_not_relative_to_the_batch():
    """A weak field must not have its best member promoted to a high score."""
    weak = _track(5, "MV NOTHING", "60", [0.30] * 40, 19.0, sog=18.0)
    ranked = _rank({5: weak})
    assert len(ranked) == 1
    assert ranked[0].score < 25.0, "a distant passenger ship scored %.1f" % ranked[0].score


def test_proximity_score_follows_the_published_formula():
    from app.ais.score import s_proximity

    assert s_proximity(0.0) == pytest.approx(1.0)
    assert s_proximity(3.0) == pytest.approx(np.exp(-1.0), rel=1e-6)
    assert s_proximity(9.0) == pytest.approx(np.exp(-3.0), rel=1e-6)
    assert s_proximity(float("inf")) == 0.0


def test_vessel_type_decoding_covers_the_ais_ranges():
    from app.ais import vessel_types as vt

    assert vt.decode(84) == "crude_oil_tanker"
    assert vt.decode(81) == "chemical_tanker"
    assert vt.decode(80) == "tanker"
    assert vt.decode(70) == "cargo"
    assert vt.decode(30) == "fishing"
    assert vt.decode(60) == "passenger"
    assert vt.decode(52) == "tug"
    assert vt.decode(None) == "unknown"
    assert vt.decode("Crude Oil Tanker") == "crude_oil_tanker"
    assert vt.type_prior("crude_oil_tanker") > vt.type_prior("fishing")
    assert vt.type_prior("passenger") < vt.type_prior("unknown")


def test_weights_sum_to_one_so_percentages_are_meaningful():
    from app import config

    assert sum(config.WEIGHTS.values()) == pytest.approx(1.0)


def test_every_suspect_carries_its_reasons():
    offsets = [-0.02 + 0.0007 * i for i in range(60)]
    ranked = _rank({7: _track(7, "MT REASONS", "84", offsets, 19.0)})
    s = ranked[0]
    assert s.reasons and any(r.startswith("origin_distance_") for r in s.reasons)
    assert any(r.startswith("type_") for r in s.reasons)
    assert set(s.components) == {"prox", "time", "type", "traj", "beh"}
    assert s.raw == pytest.approx(sum(s.weighted.values()))


# ---------------------------------------------------------------------------
# Regressions from the PS 26143 conformance review
# ---------------------------------------------------------------------------

def test_proximity_does_not_saturate_across_a_wide_origin_zone():
    """The bug: 40 percent of the model was a constant.

    Proximity used to decay on distance to the origin zone BOUNDARY, clamped to
    zero inside. Origin zones are routinely 200 km2 and every candidate crosses
    one somewhere in a six hour window, so every candidate scored exactly 1.0
    and the heaviest weight in the ranking carried no information at all.
    """
    from app.ais.score import s_proximity

    zone_km = 8.3
    centre = s_proximity(0.0, zone_km)
    edge = s_proximity(zone_km, zone_km)
    far = s_proximity(2 * zone_km, zone_km)

    assert centre == pytest.approx(1.0)
    assert edge == pytest.approx(1 / np.e, rel=1e-6), "the zone edge must read as one sigma"
    assert far < edge < centre
    assert centre - far > 0.5, "a wide zone must still separate centre from outside"


def test_time_offset_changes_the_score():
    """Spatio-temporal correlation, not spatial correlation with a time filter.

    Two identical vessels on identical tracks, one passing the origin at the
    estimated origin time and one two hours off, must not score the same.
    """
    offsets = [-0.02 + 0.0007 * i for i in range(60)]
    on_time = _track(1, "MT PUNCTUAL", "84", offsets, 19.0, start_min=-30)
    late = _track(2, "MT LATE", "84", offsets, 19.0, start_min=90)

    ranked = {s.mmsi: s for s in _rank({1: on_time, 2: late})}
    assert ranked[1].components["time"] > ranked[2].components["time"]
    assert ranked[1].score > ranked[2].score
    assert ranked[1].rank == 1

    # And with no origin time supplied the term is simply absent, never invented.
    blind = {s.mmsi: s for s in _rank({1: on_time, 2: late}, with_time=False)}
    assert blind[1].components["time"] == 0.0


def test_a_two_ping_track_cannot_outrank_a_well_observed_one():
    """The bug that put a fishing boat with two AIS receptions at rank 1.

    A long silence pays the maximum behaviour score, so the least observed
    vessel on the scene won: its whole track was dead reckoning between two
    pings, and the gap that reckoning spanned was the evidence against it.
    """
    dense = [-0.02 + 0.0007 * i for i in range(60)]
    well_seen = _track(10, "MT WELL SEEN", "84", dense, 19.0)

    sparse = _track(11, "FV TWO PINGS", "30", [-0.004, 0.004], 19.0,
                    step_min=160, gap_range=(0, 1), start_min=-80)
    sparse.raw_count = 2
    sparse.dead_reckoned[:] = True

    ranked = _rank({10: well_seen, 11: sparse})
    by_mmsi = {s.mmsi: s for s in ranked}

    assert by_mmsi[11].confidence < 0.6, "a 2-reception track must lose confidence"
    assert by_mmsi[11].components["beh"] < 0.95, "sparse coverage is not demonstrated evasion"
    assert any("sparse_track_gap" in r for r in by_mmsi[11].reasons)
    assert by_mmsi[10].rank < by_mmsi[11].rank


def test_confidence_is_reported_and_actually_applied():
    """An analyst must see both the case and how much track it rests on."""
    offsets = [-0.02 + 0.0007 * i for i in range(60)]
    tr = _track(12, "MT HALF GUESSED", "84", offsets, 19.0)
    tr.dead_reckoned[:30] = True

    s = _rank({12: tr})[0]
    assert 0.0 < s.confidence < 1.0
    assert s.score == pytest.approx(s.score_raw_evidence * s.confidence)
    assert s.to_dict()["confidence"] == pytest.approx(round(s.confidence, 3))
    assert any(r.startswith("dead_reckoned_") for r in s.reasons)


def test_scores_are_distinct_enough_to_act_on():
    """Three vessels tied at the same percent is not a ranking."""
    dense = [-0.02 + 0.0007 * i for i in range(60)]
    tracks = {
        20: _track(20, "FV ONE", "30", dense, 19.000, start_min=-30),
        21: _track(21, "FV TWO", "30", dense, 19.004, start_min=-10),
        22: _track(22, "FV THREE", "30", dense, 19.008, start_min=20),
    }
    scores = [round(s.score, 1) for s in _rank(tracks)]
    assert len(set(scores)) == len(scores), "distinct vessels produced identical scores: %r" % scores
