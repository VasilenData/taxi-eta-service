"""Small synthetic/unit fixtures; these are not the public model benchmark."""
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tripeta import booking, drift, model
from tripeta.api import create_app
from tripeta.data import BOOKING_FEATURES, load_month
from tripeta.evidence import data_fingerprint, frame_fingerprint, sha256


def phases():
    return {"train": ["2023-01"], "validation": ["2023-02"],
            "calibration": ["2023-03"], "test": ["2023-04"]}


@pytest.mark.parametrize("phase,value", [("train", []), ("validation", ["2023-01"]),
                                        ("calibration", ["2023-02"]), ("test", ["2023-03"]),
                                        ("test", ["2023-04", "2023-04"])])
def test_phase_overlap_and_duplicates_are_refused(phase, value):
    config = phases()
    config[phase] = value
    with pytest.raises(ValueError):
        booking.validate_phases(config)


def test_route_baseline_support_and_fallback_are_hand_computed():
    frame = pd.DataFrame({"pickup_zone": [1] * 20 + [3] * 2, "dropoff_zone": [2] * 20 + [4] * 2,
                          "hour": [8] * 20 + [9] * 2, "duration_min": [10.] * 20 + [30.] * 2})
    baseline = booking.RouteBaseline.fit(frame)
    request = pd.DataFrame({"pickup_zone": [1, 3, 5, 5], "dropoff_zone": [2, 4, 6, 6],
                            "hour": [9, 8, 9, 23]})
    assert baseline.predict(request).tolist() == [10., 10., 30., 10.]
    assert baseline.routes == {(1, 2): 10.}


def test_content_fingerprint_not_just_same_row_count_or_split_label():
    frame = pd.DataFrame({"x": [1., 2.], "duration_min": [10., 20.]})
    before = frame_fingerprint(frame)
    assert before == frame_fingerprint(frame.copy())
    for changed in (frame.assign(x=[1., 3.]), frame.assign(duration_min=[10., 21.]), frame.iloc[::-1]):
        assert frame_fingerprint(changed) != before
    assert data_fingerprint({"train": frame}) != data_fingerprint({"validation": frame})


def test_labels_must_be_available_at_phase_boundary_and_ids_are_source_rows(tmp_path):
    source = tmp_path / "yellow_tripdata_2023-01.parquet"
    with duckdb.connect() as con:
        con.execute("""copy (select timestamp '2023-01-31 23:59:00' as tpep_pickup_datetime,
            case when i=0 then timestamp '2023-02-01 00:01:00' else timestamp '2023-01-31 23:59:59' end
                as tpep_dropoff_datetime, 1 as PULocationID, 2 as DOLocationID, 1 as passenger_count,
            2.0 as trip_distance, 10.0 as fare_amount from range(2) t(i)) to ? (format parquet)""", [str(source)])
        # Make the second trip exactly 60 seconds so it meets existing eligibility.
        con.execute("""copy (select * replace (timestamp '2023-01-31 23:58:59' as tpep_pickup_datetime)
            from read_parquet(?)) to ? (format parquet)""", [str(source), str(tmp_path / "fixed.parquet")])
    source.write_bytes((tmp_path / "fixed.parquet").read_bytes())
    frame = load_month(source, include_identity=True, label_before="2023-02-01", expected_month="2023-01")
    assert frame.source_row.tolist() == [1]
    assert frame.duration_min.tolist() == [1.0]
    assert (frame.dropoff_at < pd.Timestamp("2023-02-01")).all()


def test_seeded_identity_sample_repeats_and_hashes_actual_file(tmp_path, frames):
    from conftest import SAMPLE

    source = tmp_path / "yellow_tripdata_2023-01.parquet"
    source.write_bytes((SAMPLE / source.name).read_bytes())
    first, sources = booking.load_phase(tmp_path, ["2023-01"], 30, 42)
    second, _ = booking.load_phase(tmp_path, ["2023-01"], 30, 42)
    assert first.record_id.tolist() == second.record_id.tolist()
    assert len(first) == 30 and not first.record_id.duplicated().any()
    assert sources[0]["sha256"] == sha256(source)
    assert sources[0]["frame_sha256"] == frame_fingerprint(first)


@pytest.fixture(scope="module")
def booking_bundle(tmp_path_factory, frames):
    training, validation = frames
    booster = model.train(training, validation, rounds=20, features=BOOKING_FEATURES, threads=2)
    destination = tmp_path_factory.mktemp("booking-registry")
    metadata = {"name": "booking-fixture", "data_fingerprint": "input-fixture", "model_sha256": "a" * 64,
                "features": BOOKING_FEATURES, "interval_ratio": [.5, 1.5],
                "drift_baseline": drift.baseline_profile(training, features=BOOKING_FEATURES)}
    model.save(booster, metadata, destination)
    return destination


def test_booking_predictions_and_drift_ignore_completed_distance(booking_bundle, frames):
    booster, metadata = model.load(booking_bundle)
    assert booster.feature_name() == BOOKING_FEATURES
    requests = frames[1].iloc[:30]
    missing = model.predict(booster, requests.drop(columns="distance_miles"))
    changed = model.predict(booster, requests.assign(distance_miles=99.0))
    np.testing.assert_array_equal(missing, changed)
    assert "distance_miles" not in metadata["drift_baseline"]


def test_booking_api_accepts_no_distance_and_uses_model_content_version(booking_bundle):
    client = TestClient(create_app(booking_bundle))
    trip = {"pickup_zone": 161, "dropoff_zone": 237, "pickup_time": "2023-06-14T18:30:00"}
    a = client.post("/predict", json=trip)
    b = client.post("/predict", json={**trip, "distance_miles": 99})
    assert a.status_code == 200 and a.json() == b.json()
    assert a.json()["model_version"].endswith("a" * 12)
    assert client.post("/predict/batch", json={"trips": [trip] * 200}).status_code == 200
    drift_report = client.get("/drift").json()
    assert "distance_miles" not in drift_report["psi"]


def test_test_targets_cannot_change_fit_selection_calibration_or_model_identity(tmp_path, monkeypatch, frames):
    events = []
    test_target = [20.]
    fixture = frames[0].iloc[:30].copy().reset_index(drop=True)
    fixture["source_row"] = range(len(fixture))
    fixture["record_id"] = [str(i) for i in range(len(fixture))]

    def load(directory, values, sample, seed):
        phase = next(name for name, value in phases().items() if value == values)
        events.append("load-" + phase)
        frame = fixture.assign(duration_min={"train": 12., "validation": 10., "calibration": 15.,
                                              "test": test_target[0]}[phase], month=values[0])
        frame.attrs["phase"] = phase
        return frame, []

    def fit(training, validation, rounds, features, threads):
        assert training.attrs["phase"] == "train" and validation.attrs["phase"] == "validation"
        assert features == BOOKING_FEATURES
        events.append("fit")
        return SimpleNamespace(best_iteration=1, model_to_string=lambda: "fixture-model")

    def predict(booster, frame):
        events.append("predict-" + frame.attrs["phase"])
        return np.full(len(frame), 10.)

    monkeypatch.setattr(booking, "load_phase", load)
    monkeypatch.setattr(booking.model, "train", fit)
    monkeypatch.setattr(booking.model, "predict", predict)
    captured = []
    monkeypatch.setattr(booking, "publish_report", lambda out, report, booster, metadata, evidence, base:
                        captured.append(metadata))
    args = Namespace(train="2023-01", valid="2023-02", calibration="2023-03", test="2023-04",
                     train_sample=30, evaluation_sample=30, rounds=1, threads=2, seed=42, data_dir="fixture",
                     output=str(tmp_path / "evidence"), registry=None)
    first = booking.run(args)
    assert events.index("fit") < events.index("load-test")
    assert events.index("predict-calibration") < events.index("load-test")
    test_target[0] = 200.
    second = booking.run(args)
    assert first["validation_gate_passed"] is second["validation_gate_passed"] is True
    for key in ("model_sha256", "data_fingerprint", "interval_ratios", "validation_mae_min"):
        assert first[key] == second[key]
    assert first["test_metrics"] != second["test_metrics"]
    assert captured[0]["interval_ratio"] == captured[1]["interval_ratio"]


def test_failed_evidence_write_preserves_prior_release_and_cleans_staging(tmp_path, monkeypatch):
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "report.json").write_text("prior complete evidence")
    output = tmp_path / "next"
    booster = SimpleNamespace(model_to_string=lambda: "test-model")
    baseline = booking.RouteBaseline({}, {8: 10.}, 10.)

    def fail(frame, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="interrupted"):
        booking.publish_report(output, {"slices": []}, booster, {}, {"test": pd.DataFrame({"a": [1]})}, baseline)
    assert not output.exists()
    assert (previous / "report.json").read_text() == "prior complete evidence"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["previous"]


def test_complete_evidence_manifest_and_gzip_pairs_roundtrip(tmp_path):
    output = tmp_path / "release"
    baseline = booking.RouteBaseline({}, {8: 10.}, 10.)
    report = {"slices": [], "phases": phases(), "sources": {name: [] for name in phases()},
              "validation_gate_passed": True,
              "test_metrics": {"lightgbm": {"mae_min": 1., "coverage": .8, "mean_interval_width_min": 5.}}}
    frame = pd.DataFrame({"record_id": ["2023-04:3", "2023-04:9"], "duration_min": [10., 20.],
                          "lightgbm_prediction": [9., 21.]})
    booster = SimpleNamespace(model_to_string=lambda: "fixture-model")
    booking.publish_report(output, report, booster, {"features": BOOKING_FEATURES}, {"test": frame}, baseline)
    manifest = json.loads((output / "manifest.json").read_text())
    for relative, identity in manifest.items():
        assert identity == {"sha256": sha256(output / relative), "bytes": (output / relative).stat().st_size}
    pd.testing.assert_frame_equal(pd.read_csv(output / "predictions_test.csv.gz"), frame)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        booking.publish_report(output, report, booster, {}, {"test": frame}, baseline)
    assert (output / "manifest.json").read_bytes() == before


def test_committed_default_bundle_matches_verified_booking_evidence():
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "reports/booking-2023-h1"
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert sha256(root / "models/current/bundle.json") == sha256(report_dir / "model/bundle.json")
    booster, metadata = model.load(root / "models/current")
    assert booster.feature_name() == metadata["features"] == BOOKING_FEATURES
    assert metadata["model_sha256"] == report["model_sha256"]
    for name, expected in report["code_sha256"].items():
        assert sha256(report_dir / "provenance/training_source" / name) == expected
    assert "\u00c2\u00b7" not in (report_dir / "index.html").read_text(encoding="utf-8")
