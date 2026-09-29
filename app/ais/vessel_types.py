"""AIS vessel type decoding and the type prior used in scoring.

Two things live here and they are kept apart on purpose:

1. `decode` turns a raw AIS ship-and-cargo-type code (ITU-R M.1371, which is
   what MarineCadastre carries in the VesselType column) into a readable
   bucket name.
2. `type_prior` turns that bucket into S_vessel_type, exactly the table in the
   spec. It is a prior on who plausibly discharges mineral oil, nothing more,
   and it is shown to the user next to every score so it can be argued with.
"""
from __future__ import annotations

from typing import Optional, Tuple

# Spec table, verbatim.
TYPE_PRIOR = {
    "crude_oil_tanker": 1.00,
    "chemical_tanker": 0.90,
    "tanker": 0.85,
    "bulk_cargo": 0.55,
    "cargo": 0.50,
    "container": 0.45,
    "fishing": 0.20,
    "passenger": 0.05,
    "unknown": 0.30,
}

# Buckets outside the spec table get a defensible neutral-to-low prior. They are
# still shown by name in the UI so a judge can see why they scored low.
EXTRA_PRIOR = {
    "tug": 0.35,
    "towing": 0.35,
    "dredging": 0.30,
    "military": 0.15,
    "sailing": 0.05,
    "pleasure": 0.05,
    "high_speed_craft": 0.10,
    "pilot": 0.10,
    "search_and_rescue": 0.05,
    "port_tender": 0.20,
    "anti_pollution": 0.25,
    "law_enforcement": 0.05,
    "other": 0.30,
}

# ITU-R M.1371 ship and cargo type ranges, as carried by MarineCadastre.
_EXACT = {
    30: "fishing",
    31: "towing",
    32: "towing",
    33: "dredging",
    34: "other",
    35: "military",
    36: "sailing",
    37: "pleasure",
    50: "pilot",
    51: "search_and_rescue",
    52: "tug",
    53: "port_tender",
    54: "anti_pollution",
    55: "law_enforcement",
    58: "other",
}

# MarineCadastre also publishes grouped codes in the 1000s in some products.
_GROUPS_1000 = {
    1001: "fishing", 1002: "fishing", 1003: "fishing", 1004: "fishing",
    1005: "cargo", 1006: "cargo", 1007: "passenger", 1008: "passenger",
    1009: "passenger", 1010: "passenger", 1011: "tanker", 1012: "tanker",
    1013: "tanker", 1014: "tanker", 1015: "tanker", 1016: "cargo",
    1017: "tanker", 1018: "cargo", 1019: "pleasure", 1020: "tug",
    1021: "tug", 1022: "tug", 1023: "military", 1024: "tanker",
    1025: "tug",
}

_TEXT = {
    "crude oil tanker": "crude_oil_tanker",
    "crude": "crude_oil_tanker",
    "oil tanker": "crude_oil_tanker",
    "oil/chemical tanker": "chemical_tanker",
    "chemical tanker": "chemical_tanker",
    "chemical": "chemical_tanker",
    "products tanker": "tanker",
    "tanker": "tanker",
    "lng tanker": "tanker",
    "lpg tanker": "tanker",
    "bulk carrier": "bulk_cargo",
    "bulk": "bulk_cargo",
    "bulk cargo": "bulk_cargo",
    "container ship": "container",
    "container": "container",
    "general cargo": "cargo",
    "cargo": "cargo",
    "fishing": "fishing",
    "trawler": "fishing",
    "passenger": "passenger",
    "ferry": "passenger",
    "cruise": "passenger",
    "tug": "tug",
    "towing": "towing",
    "pleasure": "pleasure",
    "sailing": "sailing",
    "military": "military",
    "dredging": "dredging",
    "pilot": "pilot",
    "unknown": "unknown",
}


def decode(value) -> str:
    """Return a bucket name for a raw VesselType value (numeric or text)."""
    if value is None:
        return "unknown"
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return "unknown"
        if s in _TEXT:
            return _TEXT[s]
        for key, bucket in _TEXT.items():
            if key in s:
                return bucket
        try:
            value = float(s)
        except ValueError:
            return "unknown"
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if code <= 0:
        return "unknown"
    if code in _GROUPS_1000:
        return _GROUPS_1000[code]
    if code in _EXACT:
        return _EXACT[code]
    if 20 <= code <= 29:
        return "high_speed_craft"
    if 40 <= code <= 49:
        return "high_speed_craft"
    if 60 <= code <= 69:
        return "passenger"
    if 70 <= code <= 79:
        return "cargo"
    if code in (81, 82):
        return "chemical_tanker"
    if code in (83, 84):
        return "crude_oil_tanker"
    if 80 <= code <= 89:
        return "tanker"
    if 90 <= code <= 99:
        return "other"
    return "unknown"


def type_prior(bucket: str) -> float:
    """S_vessel_type in [0, 1]."""
    if bucket in TYPE_PRIOR:
        return TYPE_PRIOR[bucket]
    return EXTRA_PRIOR.get(bucket, TYPE_PRIOR["unknown"])


def label(bucket: str) -> str:
    return bucket.replace("_", " ")


def describe(value) -> Tuple[str, float, str]:
    """(bucket, prior, human label) for one raw VesselType value."""
    b = decode(value)
    return b, type_prior(b), label(b)


def code_for(bucket: str) -> Optional[int]:
    """A representative AIS code for a bucket, used by the traffic simulator."""
    return {
        "crude_oil_tanker": 84,
        "chemical_tanker": 81,
        "tanker": 80,
        "bulk_cargo": 70,
        "cargo": 79,
        "container": 71,
        "fishing": 30,
        "passenger": 60,
        "tug": 52,
        "towing": 31,
    }.get(bucket)
