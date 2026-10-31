import numpy as np
import pytest
from fastapi.testclient import TestClient

from tripeta import drift
from tripeta.api import create_app


def test_psi_zero_for_identical_and_large_for_shift():
    e = [0.25, 0.25, 0.25, 0.25]
    assert drift.psi(e, e) == pytest.approx(0, abs=1e-9)
    assert drift.psi(e, [0.7, 0.1, 0.1, 0.1]) > 0.25


def test_status_thresholds():
    assert [drift.status(v) for v in (0.05, 0.15, 0.4)] == ["stable", "moderate", "major"]


def test_same_distribution_is_stable_and_shifted_is_major(frames):
    train, valid = frames
    profile = drift.baseline_profile(train)
    same = drift.feature_psi(profile, valid.sample(2000, random_state=1).to_dict("records"))
    assert max(same.values()) < 0.1
    shifted = valid.assign(distance_miles=np.clip(valid.distance_miles * 4, 0.1, 99), hour=3)
    bad = drift.feature_psi(profile, shifted.sample(2000, random_state=1).to_dict("records"))
    assert bad["distance_miles"] > 0.25 and bad["hour"] > 0.25


def test_api_drift_endpoint_reports_shift(tiny_model):
    c = TestClient(create_app(tiny_model))
    assert c.get("/drift").json()["status"] == "insufficient_data"
    trips = [{"pickup_zone": 161, "dropoff_zone": 237, "pickup_time": "2023-10-14T03:30:00", "distance_miles": 40.0}] * 300
    for i in range(0, 300, 100):
        c.post("/predict/batch", json={"trips": trips[i:i + 100]})
    r = c.get("/drift").json()
    assert r["status"] == "major" and r["worst_feature"] in r["psi"]
