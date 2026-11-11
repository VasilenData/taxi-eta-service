"""One bounded HTTP observation of the booking service; not an uptime monitor."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "passengers"]
MAX_BODY = 1024 * 1024
EXAMPLE = {"pickup_zone": 161, "dropoff_zone": 237, "pickup_time": "2023-06-14T18:30:00", "passengers": 1}


def checked_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError("base URL must be HTTP(S), without credentials, a query or a fragment")
    return value.rstrip("/")


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def probe(base_url: str, expected_sha256: str, *, timeout: float = 5, max_latency_ms: float = 5000) -> dict:
    base_url = checked_url(base_url)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected model SHA-256 must be 64 lowercase hexadecimal characters")
    if not _finite(timeout) or not 0 < timeout <= 10 or not _finite(max_latency_ms) or max_latency_ms <= 0:
        raise ValueError("timeout must be in (0,10] seconds and latency budget must be finite and positive")
    started_at = datetime.now(timezone.utc).isoformat()
    checks, payloads = [], {}
    for path in ("/health", "/ready", "/model", "/predict"):
        data = json.dumps(EXAMPLE).encode() if path == "/predict" else None
        request = urllib.request.Request(base_url + path, data=data, headers={"Accept": "application/json",
                                          "Content-Type": "application/json", "User-Agent": "tripeta-probe/1"})
        started = time.perf_counter()
        status, error = None, None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                body = response.read(MAX_BODY + 1)
            if status != 200 or len(body) > MAX_BODY:
                raise ValueError("expected HTTP 200 and a response no larger than 1 MiB")
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("response must be a JSON object")
            payloads[path] = payload
        except urllib.error.HTTPError as exc:
            status, error = exc.code, "HTTPError"
        except (OSError, ValueError) as exc:
            error = type(exc).__name__
        elapsed_ms = (time.perf_counter() - started) * 1000
        checks.append({"path": path, "method": request.get_method(), "status": status,
                       "elapsed_ms": round(elapsed_ms, 3), "within_latency_budget": elapsed_ms <= max_latency_ms,
                       "transport_ok": error is None, "error": error})

    health, ready, card, prediction = (payloads.get(p, {}) for p in ("/health", "/ready", "/model", "/predict"))
    identity = (card.get("model_sha256") == expected_sha256 and card.get("features") == FEATURES
                and isinstance(card.get("name"), str)
                and ready.get("model_version") == f"{card.get('name')}-{expected_sha256[:12]}"
                and prediction.get("model_version") == ready.get("model_version"))
    values = [prediction.get(k) for k in ("low_minutes", "predicted_minutes", "high_minutes")]
    prediction_ok = (all(_finite(v) for v in values) and 0 < values[0] <= values[1] <= values[2])
    contracts = {"health": health.get("status") == "ok", "ready": ready.get("status") == "ready",
                 "model_identity_and_features": identity, "booking_prediction_without_distance": prediction_ok}
    return {"observed_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": base_url, "expected_model_sha256": expected_sha256, "timeout_seconds_per_request": timeout,
            "latency_budget_ms_per_request": max_latency_ms, "checks": checks, "contracts": contracts,
            "prediction": prediction if prediction_ok else None,
            "ok": all(c["transport_ok"] and c["within_latency_budget"] for c in checks) and all(contracts.values()),
            "scope": "one sequential four-request observation; not uptime, throughput or a latency percentile"}


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".probe-", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--max-latency-ms", type=float, default=5000)
    args = parser.parse_args()
    expected = args.expected_model_sha256
    if expected is None:
        bundle = Path(__file__).resolve().parents[1] / "models/current/bundle.json"
        expected = json.loads(bundle.read_text(encoding="utf-8"))["metadata"]["model_sha256"]
    report = probe(args.base_url, expected, timeout=args.timeout, max_latency_ms=args.max_latency_ms)
    write_report(Path(args.output), report)
    print(json.dumps({"ok": report["ok"], "contracts": report["contracts"], "output": args.output}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
