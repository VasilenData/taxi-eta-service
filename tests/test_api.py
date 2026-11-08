import pytest
from fastapi.testclient import TestClient

from tripeta.api import create_app

GOOD = {"pickup_zone": 161, "dropoff_zone": 237, "pickup_time": "2023-10-14T18:30:00", "distance_miles": 2.4}


@pytest.fixture()
def client(tiny_model):
    return TestClient(create_app(tiny_model))


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model_version"].endswith("testfp")


def test_predict_returns_ordered_range(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["low_minutes"] <= body["predicted_minutes"] <= body["high_minutes"]
    assert 1 < body["predicted_minutes"] < 120


@pytest.mark.parametrize("patch", [
    {"pickup_zone": 0}, {"pickup_zone": 266}, {"dropoff_zone": -1}, {"distance_miles": 0},
    {"distance_miles": 5000}, {"passengers": 0}, {"passengers": 9}, {"pickup_time": "yesterday"},
    {"distance_miles": "far"},
    {"pickup_time": "2023-10-14T18:30:00Z"}, {"pickup_time": "2023-10-14T18:30:00+02:00"},
])
def test_invalid_requests_are_rejected_with_422(client, patch):
    assert client.post("/predict", json={**GOOD, **patch}).status_code == 422


def test_missing_field_is_422(client):
    bad = {k: v for k, v in GOOD.items() if k != "distance_miles"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_matches_single(client):
    single = client.post("/predict", json=GOOD).json()["predicted_minutes"]
    batch = client.post("/predict/batch", json={"trips": [GOOD, GOOD]}).json()
    assert [b["predicted_minutes"] for b in batch] == [single, single]


def test_batch_size_is_capped(client):
    assert client.post("/predict/batch", json={"trips": []}).status_code == 422
    assert client.post("/predict/batch", json={"trips": [GOOD] * 1001}).status_code == 422


def test_service_without_model_is_alive_but_not_ready(tmp_path):
    c = TestClient(create_app(tmp_path / "nothing"))
    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code == 503
    assert c.post("/predict", json=GOOD).status_code == 503


def test_metrics_count_requests(client):
    client.post("/predict", json=GOOD)
    client.post("/predict", json={**GOOD, "pickup_zone": 0})
    text = client.get("/metrics").text
    assert 'tripeta_requests_total{path="/predict",status="200"} 1' in text
    assert 'tripeta_requests_total{path="/predict",status="422"} 1' in text
    assert "tripeta_predictions_total 1" in text


def test_model_card_has_metrics_fields(client):
    assert "interval_ratio" in client.get("/model").json()


def test_unmatched_urls_share_a_bounded_metric_label(client):
    for number in range(20):
        assert client.get(f"/missing-{number}").status_code == 404
    metrics = client.get("/metrics").text
    assert 'path="<unmatched>",status="404"} 20' in metrics
    assert "missing-" not in metrics
