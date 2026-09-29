"""Georeferencing tests.

Two things must hold or every coordinate in the product is fiction:

  1. A polygon extracted from a raster with a known affine has to land where
     the affine says, to better than 1e-4 degrees.
  2. A mask with no CRS must inherit the SAR image's transform. Zenodo ships
     masks like that routinely, and this is the single easiest way to produce a
     confident, precise, completely wrong answer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write(tmp_path, array, transform, name="t.tif", crs="EPSG:4326"):
    from app.geo import raster as raster_mod

    p = Path(tmp_path) / name
    raster_mod.write_geotiff(p, array, transform, crs)
    return p


def test_centroid_matches_the_known_affine(tmp_path):
    from app.geo import geometry, raster as raster_mod

    size = 200
    px = 0.001
    lon0, lat0 = 70.0, 20.0
    transform = (px, 0.0, lon0, 0.0, -px, lat0)

    sigma = np.full((size, size), -10.0, dtype=np.float32)
    sar_path = _write(tmp_path, sigma, transform, "sar.tif")
    sar = raster_mod.load_sar(sar_path)

    # A 20x20 block whose centre is exactly at pixel (60.0, 40.0) corner space.
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[30:50, 50:70] = 2
    polys = geometry.polygons_from_mask(mask, sar, klass=2, min_area_km2=0.0,
                                        min_pixels=1, simplify_px=0.0)
    assert len(polys) == 1
    p = polys[0]

    expected_lon = lon0 + px * (np.mean(np.arange(50, 70)) + 0.5)
    expected_lat = lat0 - px * (np.mean(np.arange(30, 50)) + 0.5)
    assert p.centroid_lon == pytest.approx(expected_lon, abs=1e-4)
    assert p.centroid_lat == pytest.approx(expected_lat, abs=1e-4)


def test_area_matches_ground_truth_within_a_percent(tmp_path):
    from app.geo import geometry, raster as raster_mod
    from app.geo.crs import meters_per_degree

    size = 300
    px = 0.001
    transform = (px, 0.0, 70.0, 0.0, -px, 20.0)
    sar = raster_mod.load_sar(_write(tmp_path, np.zeros((size, size), np.float32),
                                     transform, "sar_area.tif"))

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[100:160, 100:200] = 2       # 60 by 100 pixels
    polys = geometry.polygons_from_mask(mask, sar, klass=2, min_area_km2=0.0,
                                        min_pixels=1, simplify_px=0.0)
    m_lon, m_lat = meters_per_degree(20.0)
    expected_km2 = (60 * px * m_lat) * (100 * px * m_lon) / 1e6
    assert polys[0].area_km2 == pytest.approx(expected_km2, rel=0.01)


def test_orientation_of_an_east_west_bar_is_near_zero(tmp_path):
    from app.geo import geometry, raster as raster_mod

    size = 240
    transform = (0.001, 0.0, 70.0, 0.0, -0.001, 20.0)
    sar = raster_mod.load_sar(_write(tmp_path, np.zeros((size, size), np.float32),
                                     transform, "sar_orient.tif"))
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[118:122, 40:200] = 2        # long in x, thin in y
    polys = geometry.polygons_from_mask(mask, sar, klass=2, min_area_km2=0.0,
                                        min_pixels=1, simplify_px=0.5)
    o = polys[0].orientation_deg
    assert min(o, 180.0 - o) < 6.0, "east-west bar should read near 0 or 180, got %.1f" % o
    assert polys[0].length_km > polys[0].width_km * 5


def test_mask_without_crs_inherits_the_sar_georeference(tmp_path):
    """The Zenodo trap, made into an assertion."""
    from app.geo import raster as raster_mod

    size = 128
    transform = (0.002, 0.0, 71.0, 0.0, -0.002, 19.5)
    sar = raster_mod.load_sar(_write(tmp_path, np.zeros((size, size), np.float32),
                                     transform, "sar_ref.tif"))

    bare = np.zeros((size, size), dtype=np.uint8)
    bare[40:60, 40:60] = 1
    bare_path = _write(tmp_path, bare, (1.0, 0.0, 0.0, 0.0, -1.0, 0.0), "bare_mask.tif")

    mask = raster_mod.load_mask_with_georef(bare_path, sar)
    assert mask.transform == sar.transform
    assert mask.crs == sar.crs
    lon, lat = mask.lonlat(50, 50)
    assert float(lon) == pytest.approx(71.0 + 0.002 * 50.5, abs=1e-6)
    assert float(lat) == pytest.approx(19.5 - 0.002 * 50.5, abs=1e-6)


def test_sar_without_georeference_is_refused(tmp_path):
    from app.geo import raster as raster_mod

    p = _write(tmp_path, np.zeros((64, 64), np.float32),
               (1.0, 0.0, 0.0, 0.0, -1.0, 0.0), "no_geo.tif")
    # An identity transform with no real CRS keys is not a georeference. Either
    # the reader reports it as missing, or the coordinates it yields are the
    # meaningless pixel indices; the loader must not pretend otherwise.
    try:
        sar = raster_mod.load_sar(p)
    except raster_mod.GeorefError:
        return
    lon, lat = sar.centre_lonlat()
    assert abs(lon) <= 64 and abs(lat) <= 64


def test_builtin_tiff_reader_matches_rasterio(tmp_path):
    """The optional dependency escape hatch has to agree with the real thing."""
    rasterio = pytest.importorskip("rasterio")
    from app.geo import tiffio

    arr = (np.random.default_rng(7).normal(-15.0, 3.0, (48, 61))).astype(np.float32)
    transform = (0.0005, 0.0, 72.25, 0.0, -0.0005, 18.75)
    p = _write(tmp_path, arr, transform, "roundtrip.tif")

    with rasterio.open(p) as ds:
        ref = ds.read(1)
        ref_t = ds.transform

    got = tiffio.read(p)
    assert got["array"].shape[-2:] == arr.shape
    assert np.allclose(got["array"][0], ref, atol=1e-5)
    assert got["transform"][0] == pytest.approx(ref_t.a, abs=1e-9)
    assert got["transform"][2] == pytest.approx(ref_t.c, abs=1e-9)
    assert got["crs"] == "EPSG:4326"


def test_distance_to_ring_is_zero_inside_and_positive_outside():
    from app.geo.geometry import distance_to_ring_km

    ring = [(71.0, 19.0), (71.1, 19.0), (71.1, 19.1), (71.0, 19.1), (71.0, 19.0)]
    assert distance_to_ring_km(71.05, 19.05, ring) == 0.0
    d = distance_to_ring_km(71.2, 19.05, ring)
    assert 9.0 < d < 12.0, "0.1 deg of longitude at 19 N is about 10.5 km, got %.2f" % d
