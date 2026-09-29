"""The live AIS recorder's boundary with the MarineCadastre store.

`row_from_csv` parses MarineCadastre CSV, so its contract is strings. aisstream
sends vessel type, navigational status and IMO as integers, and the first vessel
to report static data crashed the recorder on `.strip()`. These pin the coercion
at the boundary, so the parser is not loosened to accept whatever a feed sends.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ais import ingest as ais_ingest   # noqa: E402


def _bare_recorder():
    """A Recorder without its database connection.

    `__init__` opens the real store, and these tests are about the boundary
    between the feed and the CSV parser, not about writing. `flush_every` is set
    beyond any test's reach so nothing tries to flush.
    """
    r = rec_mod.Recorder.__new__(rec_mod.Recorder)
    r.static, r.pending, r.mmsis, r.seen = {}, [], set(), 0
    r.flush_every = 10 ** 9
    return r


def _load():
    spec = importlib.util.spec_from_file_location(
        "record_ais", ROOT / "scripts" / "record_ais.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rec_mod = _load()


def test_integer_fields_become_strings():
    assert rec_mod._text(70) == "70"
    assert rec_mod._text(0) == "0"        # status 0 is "under way", not absent
    assert rec_mod._text(None) is None
    assert rec_mod._text("") is None


def test_imo_matches_the_marinecadastre_spelling():
    assert rec_mod._imo(9260427) == "IMO9260427"
    assert rec_mod._imo(0) is None        # 0 is the "not available" sentinel
    assert rec_mod._imo(None) is None


def test_dimensions_become_length_and_beam():
    # ITU-R M.1371 reports distances to bow, stern, port, starboard.
    assert rec_mod._dims({"A": 100, "B": 50, "C": 10, "D": 12}) == (150, 22)
    assert rec_mod._dims(None) == (None, None)
    assert rec_mod._dims({"A": 100}) == (None, None)


def test_a_recorded_position_survives_the_csv_parser():
    """The end to end shape: what the recorder builds must ingest cleanly."""
    r = _bare_recorder()

    meta = {"MMSI": 563333100, "ShipName": "ANNA COSULICH",
            "time_utc": "2026-09-04 17:39:22.123456789 +0000 UTC"}
    static = {"Name": "ANNA COSULICH", "Type": 70, "ImoNumber": 9260427,
              "CallSign": "9V1234", "Dimension": {"A": 100, "B": 50, "C": 10, "D": 12},
              "MaximumStaticDraught": 8.5}
    position = {"Latitude": 1.2981, "Longitude": 103.9889, "Sog": 0.0,
                "Cog": 96.4, "TrueHeading": 124, "NavigationalStatus": 0}

    r.on_static(static, meta)
    r.on_position(position, meta)

    assert len(r.pending) == 1
    row = ais_ingest.row_from_csv(r.pending[0], "aisstream_live")
    assert row is not None, "the recorder built a record the store rejects"

    built = r.pending[0]
    assert built["BaseDateTime"] == "2026-09-04T17:39:22"
    assert built["IMO"] == "IMO9260427"
    assert built["VesselType"] == "70"
    assert built["Length"] == 150 and built["Width"] == 22


def test_not_available_sentinels_are_dropped():
    """511 heading and 102.3 knots mean 'unknown', not a real reading."""
    r = _bare_recorder()

    r.on_position({"Latitude": 1.0, "Longitude": 103.0, "TrueHeading": 511,
                   "Cog": 360.0, "Sog": 102.3}, {"MMSI": 1})
    built = r.pending[0]
    assert built["Heading"] is None
    assert built["COG"] is None
    assert built["SOG"] is None


def test_a_position_outside_the_valid_range_is_refused():
    r = _bare_recorder()

    r.on_position({"Latitude": 91.0, "Longitude": 0.0}, {"MMSI": 1})
    r.on_position({"Latitude": 0.0, "Longitude": 181.0}, {"MMSI": 2})
    r.on_position({"Latitude": None, "Longitude": 0.0}, {"MMSI": 3})
    assert r.pending == []
