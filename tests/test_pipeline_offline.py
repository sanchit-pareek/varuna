"""End to end test with the network physically blocked.

The judged demo has to run in airplane mode, so this test makes that literal:
socket.socket is replaced with something that raises on any connection attempt,
and then /api/run is driven through the real ASGI app. If any module tries to
reach Open-Meteo, Zenodo, Planetary Computer or a tile server at request time,
the test fails.
"""
from __future__ import annotations

import socket

import pytest


LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


def _is_loopback(address) -> bool:
    """Loopback has to stay open: asyncio builds its own self pipe out of it."""
    if isinstance(address, tuple) and address:
        return str(address[0]) in LOOPBACK
    return False


@pytest.fixture
def no_network(monkeypatch):
    """Block every outbound connection except loopback.

    This is what airplane mode means for this product. The loopback exemption is
    not a loophole: nothing in app/ ever dials a local port, while asyncio's
    event loop on Windows cannot start without one.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def guard(address):
        if not _is_loopback(address):
            raise AssertionError("the pipeline tried to reach the network: %r" % (address,))

    def connect(self, address, *a, **k):
        guard(address)
        return real_connect(self, address, *a, **k)

    def connect_ex(self, address, *a, **k):
        guard(address)
        return real_connect_ex(self, address, *a, **k)

    def create_connection(address, *a, **k):
        guard(address)
        return real_create(address, *a, **k)

    def getaddrinfo(host, *a, **k):
        if str(host) not in LOOPBACK:
            raise AssertionError("the pipeline tried to resolve %r" % (host,))
        return real_getaddrinfo(host, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("urlopen called at request time")))
    return True


@pytest.fixture
def seeded_ais(selftest_scene, cached_metocean):
    """Simulate traffic for the scene, using the pipeline's own computed origin."""
    from app import pipeline
    from app.ais import synthetic

    t_sat = pipeline._utc(selftest_scene.t_sat)
    det = pipeline.detect_scene(selftest_scene, render_overlays=False)
    assert det["polygon_objects"], "self test scene must contain a detectable slick"
    primary = det["polygon_objects"][0]

    drift = pipeline.run_drift(primary.ring_lonlat,
                               (primary.centroid_lon, primary.centroid_lat),
                               t_sat, 48, 36, 30, selftest_scene.id)
    origin = drift["origin"]
    built = synthetic.build(
        bbox=selftest_scene.bounds, t_center=t_sat,
        origin_lon=origin["lon"], origin_lat=origin["lat"],
        t_origin=pipeline._utc(origin["t"]),
        hours=72, step_seconds=300, n_vessels=24,
    )
    synthetic.write(built["rows"], built["ground_truth"])
    return built


def test_full_pipeline_runs_offline(no_network, selftest_scene, cached_metocean, seeded_ais):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        r = client.post("/api/run", json={"scene_id": selftest_scene.id})
        assert r.status_code == 200, r.text
        doc = r.json()

    assert doc["status"] == "ok"
    assert doc["job_id"]

    steps = [s["step"] for s in doc["trace"]]
    for required in ("DETECT", "CHAR", "METOCEAN", "HINDCAST", "FORECAST",
                     "AGE_PROXY", "AIS", "FILTER", "SCORE"):
        assert required in steps, "missing pipeline step %s in %s" % (required, steps)

    det = doc["detection"]
    assert det["polygons"], "detection produced no oil polygon"
    props = det["polygons"][0]["properties"]
    for key in ("area_km2", "perimeter_km", "length_km", "orientation_deg",
                "centroid_lat", "centroid_lon", "compactness", "bbox", "confidence"):
        assert key in props, "missing geometric property %s" % key
    assert props["area_km2"] > 0

    drift = doc["drift"]
    assert drift["origin_zone"]["geometry"]["type"] == "Polygon"
    assert drift["cone_back"]["geometry"] and drift["cone_fwd"]["geometry"]
    assert drift["hindcast_hourly"] and drift["forecast_hourly"]
    assert doc["age_hours_proxy"] > 0
    assert "proxy" in drift["age_label"].lower()
    assert drift["metocean"]["synthetic"] is False, "clause (b) needs a real cached field"

    attr = doc["attribution"]
    assert attr["funnel"]["considered_vessels"] > 0, "no AIS traffic reached the filter"
    assert attr["scoring"]["weights"]
    for s in attr["suspects"]:
        assert 0.0 <= s["score"] <= 100.0
        assert s["reasons"]
        assert s["track"]["geojson"]["features"]


def test_run_is_reproducible(no_network, selftest_scene, cached_metocean, seeded_ais):
    """Two identical runs must agree. A judge may press the button twice."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        a = client.post("/api/run", json={"scene_id": selftest_scene.id}).json()
        b = client.post("/api/run", json={"scene_id": selftest_scene.id}).json()

    assert a["drift"]["origin"]["t"] == b["drift"]["origin"]["t"]
    assert a["drift"]["origin"]["lat"] == pytest.approx(b["drift"]["origin"]["lat"], abs=1e-9)
    assert [s["mmsi"] for s in a["attribution"]["suspects"]] == \
           [s["mmsi"] for s in b["attribution"]["suspects"]]
    assert [round(s["score"], 6) for s in a["attribution"]["suspects"]] == \
           [round(s["score"], 6) for s in b["attribution"]["suspects"]]


def test_job_document_and_geojson_export(no_network, selftest_scene, cached_metocean, seeded_ais):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        doc = client.post("/api/run", json={"scene_id": selftest_scene.id}).json()
        job_id = doc["job_id"]

        again = client.get("/api/jobs/%s" % job_id)
        assert again.status_code == 200
        assert again.json()["job_id"] == job_id

        gj = client.get("/api/jobs/%s/geojson" % job_id)
        assert gj.status_code == 200
        fc = gj.json()
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) >= 4

        note = client.get("/api/report/%s" % job_id)
        assert note.status_code == 200
        assert "Attribution Note" in note.text
        assert "not legal proof" in note.text.lower() or "not proof" in note.text.lower()


def test_health_reports_the_truth_about_missing_pieces(no_network, selftest_scene):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        h = client.get("/api/health").json()

    assert h["ok"] is True
    assert h["mode"] == "offline"
    # torch is not installed in the test environment, so the baseline must be
    # reported honestly rather than the UI implying a model ran.
    if not h["model_loaded"]:
        assert h["detector"] == "sigma0_threshold_baseline"
        assert any("baseline" in w for w in h["warnings"])


def test_empty_ais_store_yields_no_suspects_not_a_crash(no_network, selftest_scene,
                                                        cached_metocean, tmp_path):
    """A true negative must be reportable. Never invent a culprit."""
    from fastapi.testclient import TestClient

    from app.ais import ingest as ais_ingest
    from app.main import app

    conn = ais_ingest.connect()
    try:
        conn.execute("DELETE FROM positions")
        conn.commit()
    finally:
        conn.close()

    with TestClient(app) as client:
        doc = client.post("/api/run", json={"scene_id": selftest_scene.id}).json()

    assert doc["status"] == "ok"
    assert doc["attribution"]["suspects"] == []
    assert any("No vessel passed" in w for w in doc["warnings"])
