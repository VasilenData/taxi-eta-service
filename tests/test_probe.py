"""Probe contracts with an injected HTTP boundary; no external service or socket."""
import importlib.util
import json
from pathlib import Path
import urllib.error

import pytest

SPEC = importlib.util.spec_from_file_location("probe", Path(__file__).resolve().parents[1] / "scripts/probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
HASH = "a" * 64


def payloads():
    version = "booking-" + HASH[:12]
    return {"/health": {"status": "ok"}, "/ready": {"status": "ready", "model_version": version},
            "/model": {"name": "booking", "features": probe.FEATURES, "model_sha256": HASH},
            "/predict": {"model_version": version, "low_minutes": 10., "predicted_minutes": 15., "high_minutes": 20.}}


class Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, count):
        assert count == probe.MAX_BODY + 1
        return json.dumps(self.payload).encode()


def install(monkeypatch, responses):
    calls = []

    def get(request, timeout):
        calls.append((request.selector, request.get_method(), timeout))
        if request.selector == "/predict":
            assert "distance_miles" not in json.loads(request.data)
        result = responses[request.selector]
        if isinstance(result, Exception):
            raise result
        return Response(result)

    monkeypatch.setattr(probe.urllib.request, "urlopen", get)
    return calls


def test_probe_checks_four_bounded_requests_and_booking_identity(monkeypatch):
    calls = install(monkeypatch, payloads())
    result = probe.probe("https://service.example", HASH, timeout=2)
    assert result["ok"] and all(result["contracts"].values())
    assert calls == [("/health", "GET", 2), ("/ready", "GET", 2), ("/model", "GET", 2), ("/predict", "POST", 2)]
    assert result["observed_at_utc"] <= result["finished_at_utc"]


@pytest.mark.parametrize("fault", ["wrong_model", "distance_feature", "bad_interval", "unready", "timeout", "http503"])
def test_contract_or_transport_failures_cannot_report_success(monkeypatch, fault):
    responses = payloads()
    if fault == "wrong_model":
        responses["/model"]["model_sha256"] = "b" * 64
    elif fault == "distance_feature":
        responses["/model"]["features"] = [*probe.FEATURES, "distance_miles"]
    elif fault == "bad_interval":
        responses["/predict"]["low_minutes"] = 30
    elif fault == "unready":
        responses["/ready"]["status"] = "no model loaded"
    elif fault == "timeout":
        responses["/health"] = TimeoutError("fixture timeout")
    else:
        responses["/ready"] = urllib.error.HTTPError("fixture", 503, "unready", None, None)
    assert len(install(monkeypatch, responses)) == 0
    assert not probe.probe("https://service.example", HASH)["ok"]


def test_latency_budget_is_measured_and_failed_report_is_retained(tmp_path, monkeypatch):
    install(monkeypatch, payloads())
    ticks = iter(range(8))
    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(ticks))
    result = probe.probe("https://service.example", HASH, max_latency_ms=999)
    assert not result["ok"] and all(c["elapsed_ms"] == 1000 for c in result["checks"])
    destination = tmp_path / "probe.json"
    probe.write_report(destination, result)
    assert json.loads(destination.read_text())["ok"] is False


def test_failed_report_replace_preserves_prior_complete_observation(tmp_path, monkeypatch):
    destination = tmp_path / "probe.json"
    destination.write_text('{"old":true}')
    def fail(*args):
        raise OSError("disk error")
    monkeypatch.setattr(probe.os, "replace", fail)
    with pytest.raises(OSError, match="disk error"):
        probe.write_report(destination, {"new": True})
    assert destination.read_text() == '{"old":true}'
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("url", ["file:///secret", "https://name:password@service.example", "https://x/?token=secret"])
def test_invalid_url_is_rejected_before_requests(url):
    with pytest.raises(ValueError):
        probe.checked_url(url)
