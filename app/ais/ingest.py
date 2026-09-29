"""AIS store: MarineCadastre CSV in, SQLite out.

The schema is the MarineCadastre data dictionary, column for column:
MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO, CallSign,
VesselType, Status, Length, Width, Draft, Cargo.

Real CSVs and the traffic simulator both land in the same table, so nothing
downstream can tell them apart or treat them differently. That is deliberate:
the scorer must be blind to provenance.

Only the standard library is used here, so the AIS half of the product works
even on an install with no pandas.
"""
from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config

COLUMNS = [
    "MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading",
    "VesselName", "IMO", "CallSign", "VesselType", "Status",
    "Length", "Width", "Draft", "Cargo",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    mmsi        INTEGER NOT NULL,
    ts          INTEGER NOT NULL,      -- epoch seconds UTC
    base_date_time TEXT NOT NULL,
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    sog         REAL,
    cog         REAL,
    heading     REAL,
    vessel_name TEXT,
    imo         TEXT,
    call_sign   TEXT,
    vessel_type TEXT,
    status      TEXT,
    length      REAL,
    width       REAL,
    draft       REAL,
    cargo       TEXT,
    source      TEXT
);
CREATE INDEX IF NOT EXISTS idx_pos_ts       ON positions(ts);
CREATE INDEX IF NOT EXISTS idx_pos_mmsi_ts  ON positions(mmsi, ts);
CREATE INDEX IF NOT EXISTS idx_pos_bbox     ON positions(lat, lon);

CREATE TABLE IF NOT EXISTS ingest_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source    TEXT,
    rows      INTEGER,
    bbox      TEXT,
    t_start   TEXT,
    t_end     TEXT,
    ingested_at TEXT
);
"""


def parse_time(value: str) -> Optional[int]:
    """MarineCadastre BaseDateTime is naive UTC 'YYYY-MM-DDTHH:MM:SS'."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.fromisoformat(s) if fmt is None else datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


def iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def connect(path: Path = None) -> sqlite3.Connection:
    path = Path(config.AIS_SQLITE if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _num(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("na", "nan", "null", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def row_from_csv(rec: Dict[str, Any], source: str) -> Optional[Tuple]:
    lookup = {k.strip().lower(): v for k, v in rec.items() if k}
    def get(name):
        return lookup.get(name.lower())

    mmsi = _num(get("MMSI"))
    ts = parse_time(get("BaseDateTime"))
    lat = _num(get("LAT"))
    lon = _num(get("LON"))
    if mmsi is None or ts is None or lat is None or lon is None:
        return None
    return (
        int(mmsi), int(ts), iso(ts), float(lat), float(lon),
        _num(get("SOG")), _num(get("COG")), _num(get("Heading")),
        (get("VesselName") or "").strip() or None,
        (get("IMO") or "").strip() or None,
        (get("CallSign") or "").strip() or None,
        (str(get("VesselType")).strip() if get("VesselType") not in (None, "") else None),
        (str(get("Status")).strip() if get("Status") not in (None, "") else None),
        _num(get("Length")), _num(get("Width")), _num(get("Draft")),
        (str(get("Cargo")).strip() if get("Cargo") not in (None, "") else None),
        source,
    )


_INSERT = """INSERT INTO positions
 (mmsi, ts, base_date_time, lat, lon, sog, cog, heading, vessel_name, imo,
  call_sign, vessel_type, status, length, width, draft, cargo, source)
 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def ingest_rows(conn: sqlite3.Connection, rows: Iterable[Tuple], source: str) -> int:
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(_INSERT, rows)
    lats = [r[3] for r in rows]
    lons = [r[4] for r in rows]
    tss = [r[1] for r in rows]
    conn.execute(
        "INSERT INTO ingest_log (source, rows, bbox, t_start, t_end, ingested_at) VALUES (?,?,?,?,?,?)",
        (source, len(rows),
         "%.5f,%.5f,%.5f,%.5f" % (min(lons), min(lats), max(lons), max(lats)),
         iso(min(tss)), iso(max(tss)),
         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")),
    )
    conn.commit()
    return len(rows)


def ingest_csv(csv_path, conn: sqlite3.Connection = None, source: str = None,
               bbox: Optional[Sequence[float]] = None,
               t_range: Optional[Tuple[str, str]] = None,
               batch: int = 20000) -> int:
    """Load a MarineCadastre style CSV, optionally clipped to a box and window.

    The clip matters in practice: a single nationwide day of NAIS is millions of
    rows, and the spec forbids dumping that into the repo.
    """
    csv_path = Path(csv_path)
    close = conn is None
    conn = conn or connect()
    source = source or csv_path.name
    t0 = parse_time(t_range[0]) if t_range else None
    t1 = parse_time(t_range[1]) if t_range else None

    total = 0
    pending: List[Tuple] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            row = row_from_csv(rec, source)
            if row is None:
                continue
            if bbox is not None:
                w, s, e, n = bbox
                if not (s <= row[3] <= n and w <= row[4] <= e):
                    continue
            if t0 is not None and row[1] < t0:
                continue
            if t1 is not None and row[1] > t1:
                continue
            pending.append(row)
            if len(pending) >= batch:
                total += ingest_rows(conn, pending, source)
                pending = []
    total += ingest_rows(conn, pending, source)
    if close:
        conn.close()
    return total


def export_csv(conn: sqlite3.Connection, path) -> int:
    """Write the store back out in MarineCadastre column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.execute(
        "SELECT mmsi, base_date_time, lat, lon, sog, cog, heading, vessel_name,"
        " imo, call_sign, vessel_type, status, length, width, draft, cargo"
        " FROM positions ORDER BY mmsi, ts"
    )
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for row in cur:
            w.writerow(list(row))
            n += 1
    return n


@dataclass
class StoreStats:
    rows: int
    vessels: int
    t_start: Optional[str]
    t_end: Optional[str]
    bbox: Optional[List[float]]
    sources: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "vessels": self.vessels,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "bbox": self.bbox,
            "sources": self.sources,
        }


def stats(conn: sqlite3.Connection = None) -> StoreStats:
    close = conn is None
    conn = conn or connect()
    try:
        r = conn.execute(
            "SELECT COUNT(*) n, COUNT(DISTINCT mmsi) v, MIN(ts) t0, MAX(ts) t1,"
            " MIN(lon) w, MIN(lat) s, MAX(lon) e, MAX(lat) nn FROM positions"
        ).fetchone()
        srcs = [dict(x) for x in conn.execute(
            "SELECT source, SUM(rows) AS rows, MIN(t_start) AS t_start,"
            " MAX(t_end) AS t_end FROM ingest_log GROUP BY source"
        ).fetchall()]
        if not r or not r["n"]:
            return StoreStats(0, 0, None, None, None, srcs)
        return StoreStats(
            rows=int(r["n"]),
            vessels=int(r["v"]),
            t_start=iso(r["t0"]),
            t_end=iso(r["t1"]),
            bbox=[round(r["w"], 5), round(r["s"], 5), round(r["e"], 5), round(r["nn"], 5)],
            sources=srcs,
        )
    finally:
        if close:
            conn.close()


def clear_source(conn: sqlite3.Connection, source: str) -> int:
    cur = conn.execute("DELETE FROM positions WHERE source = ?", (source,))
    conn.execute("DELETE FROM ingest_log WHERE source = ?", (source,))
    conn.commit()
    return cur.rowcount


def query_window(conn: sqlite3.Connection, bbox: Sequence[float],
                 t_start: int, t_end: int) -> Dict[int, List[sqlite3.Row]]:
    """All positions in a box and time window, grouped by MMSI, time ordered."""
    w, s, e, n = bbox
    cur = conn.execute(
        "SELECT * FROM positions WHERE ts BETWEEN ? AND ?"
        " AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?"
        " ORDER BY mmsi, ts",
        (int(t_start), int(t_end), float(s), float(n), float(w), float(e)),
    )
    out: Dict[int, List[sqlite3.Row]] = {}
    for row in cur:
        out.setdefault(int(row["mmsi"]), []).append(row)
    return out


def query_tracks(conn: sqlite3.Connection, mmsis: Sequence[int],
                 t_start: int, t_end: int) -> Dict[int, List[sqlite3.Row]]:
    """Full tracks for named vessels over a window, ignoring the box.

    Needed after filtering: a vessel that clipped the search box still needs its
    whole approach and departure drawn, or the trajectory score is meaningless.
    """
    if not mmsis:
        return {}
    marks = ",".join("?" for _ in mmsis)
    cur = conn.execute(
        "SELECT * FROM positions WHERE mmsi IN (%s) AND ts BETWEEN ? AND ?"
        " ORDER BY mmsi, ts" % marks,
        list(int(m) for m in mmsis) + [int(t_start), int(t_end)],
    )
    out: Dict[int, List[sqlite3.Row]] = {}
    for row in cur:
        out.setdefault(int(row["mmsi"]), []).append(row)
    return out
