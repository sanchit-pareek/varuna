"""AIS reconstruction tests.

A reporting gap is evidence, so the reconstruction has to keep it visible. The
requirement is precise: a gap of at least 30 minutes whose bridged segment
passes near the origin zone must raise the ais_gap reason, and the bridged
samples must be flagged so the map can draw them as dead reckoning rather than
as reported positions.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

T0 = datetime(2024, 2, 14, 0, 0, tzinfo=timezone.utc)


def _rows(points, mmsi=419000001, name="MT TEST", vtype="80"):
    """points: list of (minutes_from_T0, lon, lat, sog, cog)."""
    out = []
    for m, lon, lat, sog, cog in points:
        ts = int((T0 + timedelta(minutes=m)).timestamp())
        out.append({
            "mmsi": mmsi, "ts": ts,
            "base_date_time": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "lat": lat, "lon": lon, "sog": sog, "cog": cog, "heading": cog,
            "vessel_name": name, "imo": "IMO1234567", "call_sign": "TEST",
            "vessel_type": vtype, "status": "0", "length": 250.0, "width": 44.0,
            "draft": 13.0, "cargo": "80", "source": "unit_test",
        })
    return out


def _straight(minutes, lon0=71.40, lat0=19.00, dlon=0.004, skip=()):
    pts = []
    for m in minutes:
        if m in skip:
            continue
        pts.append((m, lon0 + dlon * m / 5.0, lat0, 12.0, 90.0))
    return pts


def test_gap_is_detected_and_bridged_samples_are_flagged():
    from app.ais.interpolate import resample

    every5 = list(range(0, 240, 5))
    # Drop the reports at minute 100 through 145 inclusive. The last report
    # before the silence is at 95 and the first after it is at 150, so the
    # observable gap is 55 minutes, which is what the reconstruction must say.
    skipped = set(range(100, 150, 5))
    rows = _rows(_straight(every5, skip=skipped))

    t_start = int(T0.timestamp())
    t_end = int((T0 + timedelta(minutes=240)).timestamp())
    tr = resample(rows, t_start, t_end, step_seconds=60, gap_minutes=30.0)

    assert tr is not None
    assert len(tr.gaps) == 1
    assert tr.gaps[0].minutes == pytest.approx(55.0, abs=0.1)
    assert tr.dead_reckoned.any()
    # Samples strictly inside the gap are dead reckoned; ones outside are not.
    inside = (tr.ts > tr.gaps[0].start_ts) & (tr.ts < tr.gaps[0].end_ts)
    assert tr.dead_reckoned[inside].all()
    assert not tr.dead_reckoned[~inside].any()


def test_short_dropouts_are_not_called_gaps():
    from app.ais.interpolate import resample

    every5 = list(range(0, 240, 5))
    rows = _rows(_straight(every5, skip={60, 65, 70}))   # 20 minutes missing
    tr = resample(rows, int(T0.timestamp()),
                  int((T0 + timedelta(minutes=240)).timestamp()),
                  step_seconds=60, gap_minutes=30.0)
    assert tr.gaps == []
    assert not tr.dead_reckoned.any()


def test_ais_gap_across_the_origin_raises_the_reason_code():
    from app.ais import score as score_mod
    from app.ais.interpolate import resample
    from app.geo.geometry import buffer_ring_km

    every5 = list(range(0, 240, 5))
    skipped = set(range(100, 150, 5))
    rows = _rows(_straight(every5, skip=skipped))
    tr = resample(rows, int(T0.timestamp()),
                  int((T0 + timedelta(minutes=240)).timestamp()),
                  step_seconds=60, gap_minutes=30.0)

    # Origin zone placed on the track at the midpoint of the gap.
    mid_ts = (tr.gaps[0].start_ts + tr.gaps[0].end_ts) // 2
    k = int(np.argmin(np.abs(tr.ts - mid_ts)))
    origin_lon, origin_lat = float(tr.lon[k]), float(tr.lat[k])
    ring = buffer_ring_km([(origin_lon, origin_lat)], 1.0)

    beh, reasons, detail = score_mod.s_behavior(tr, k, ring, origin_lon, origin_lat)
    assert detail["non_reporting"] is True
    assert beh == pytest.approx(score_mod.BEH_AIS_GAP)
    assert any(r.startswith("ais_gap_") for r in reasons)


def test_gap_far_from_the_origin_does_not_raise_it():
    from app.ais import score as score_mod
    from app.ais.interpolate import resample
    from app.geo.geometry import buffer_ring_km

    every5 = list(range(0, 240, 5))
    rows = _rows(_straight(every5, skip=set(range(0, 45, 5))))
    tr = resample(rows, int(T0.timestamp()),
                  int((T0 + timedelta(minutes=240)).timestamp()),
                  step_seconds=60, gap_minutes=30.0)

    # Origin 60 km away from anywhere the vessel went.
    origin_lon, origin_lat = 72.30, 19.00
    ring = buffer_ring_km([(origin_lon, origin_lat)], 1.0)
    k = tr.n - 1
    _beh, reasons, detail = score_mod.s_behavior(tr, k, ring, origin_lon, origin_lat)
    assert detail["non_reporting"] is False
    assert not any(r.startswith("ais_gap_") for r in reasons)


def test_course_interpolation_wraps_through_north():
    from app.ais.interpolate import resample

    pts = [(0, 71.4, 19.0, 10.0, 350.0), (10, 71.41, 19.02, 10.0, 10.0)]
    rows = _rows(pts)
    tr = resample(rows, int(T0.timestamp()),
                  int((T0 + timedelta(minutes=10)).timestamp()), step_seconds=60)
    mid = tr.cog[len(tr.cog) // 2]
    assert (mid > 345.0) or (mid < 15.0), "COG interpolated the long way round: %.1f" % mid


def test_sqlite_roundtrip_preserves_marinecadastre_columns(tmp_path):
    from app.ais import ingest

    conn = ingest.connect(tmp_path / "ais.sqlite")
    try:
        rows = []
        for r in _rows(_straight(list(range(0, 60, 5)))):
            rows.append((r["mmsi"], r["ts"], r["base_date_time"], r["lat"], r["lon"],
                         r["sog"], r["cog"], r["heading"], r["vessel_name"], r["imo"],
                         r["call_sign"], r["vessel_type"], r["status"], r["length"],
                         r["width"], r["draft"], r["cargo"], "unit_test"))
        n = ingest.ingest_rows(conn, rows, "unit_test")
        assert n == len(rows)

        st = ingest.stats(conn)
        assert st.rows == len(rows) and st.vessels == 1

        csv_path = tmp_path / "out.csv"
        written = ingest.export_csv(conn, csv_path)
        assert written == len(rows)
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == ingest.COLUMNS
    finally:
        conn.close()
