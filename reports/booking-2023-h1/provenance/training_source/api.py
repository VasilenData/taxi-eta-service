"""FastAPI service. Validates every request, returns a calibrated range, exposes health, metrics and drift."""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from . import __version__, drift, model
from .data import LIMITS

MODEL_DIR = Path(os.environ.get("TRIPETA_MODEL_DIR", Path(__file__).resolve().parents[2] / "models" / "current"))
RECENT_MAX = int(os.environ.get("TRIPETA_RECENT_MAX", "5000"))
MIN_ROWS_FOR_DRIFT = 200
BATCH_MAX = 1000

zmin, zmax = LIMITS["zone"]
dmin, dmax = LIMITS["distance_miles"]
pmin, pmax = LIMITS["passengers"]


class TripRequest(BaseModel):
    pickup_zone: int = Field(ge=zmin, le=zmax, description="TLC taxi zone id of the pickup")
    dropoff_zone: int = Field(ge=zmin, le=zmax, description="TLC taxi zone id of the drop-off")
    pickup_time: datetime = Field(description="ISO 8601 local pickup time, e.g. 2023-10-14T18:30:00")
    distance_miles: float | None = Field(default=None, ge=dmin, le=dmax,
                                         description="Required by legacy distance models; ignored by booking models")
    passengers: int = Field(default=1, ge=pmin, le=pmax)

    @field_validator("distance_miles")
    @classmethod
    def finite(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("distance must be a finite number")
        return v

    @field_validator("pickup_time")
    @classmethod
    def local_time(cls, value: datetime) -> datetime:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("pickup_time must be NYC local wall time without a timezone offset")
        return value


class Prediction(BaseModel):
    predicted_minutes: float
    low_minutes: float = Field(description="10th percentile of the model's historical error range")
    high_minutes: float = Field(description="90th percentile")
    model_version: str


class BatchRequest(BaseModel):
    trips: list[TripRequest] = Field(min_length=1, max_length=BATCH_MAX)


class Metrics:
    """Tiny Prometheus-format counters, no extra dependency."""
    BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)

    def __init__(self):
        self.lock = threading.Lock()
        self.requests: dict[tuple[str, int], int] = {}
        self.latency = [0] * (len(self.BUCKETS) + 1)
        self.latency_sum = 0.0
        self.predictions = 0

    def observe(self, path: str, status: int, seconds: float):
        with self.lock:
            self.requests[(path, status)] = self.requests.get((path, status), 0) + 1
            self.latency_sum += seconds
            idx = next((i for i, b in enumerate(self.BUCKETS) if seconds <= b), len(self.BUCKETS))
            self.latency[idx] += 1

    def render(self) -> str:
        lines = ["# TYPE tripeta_requests_total counter"]
        lines += [f'tripeta_requests_total{{path="{p}",status="{s}"}} {n}' for (p, s), n in sorted(self.requests.items())]
        lines += ["# TYPE tripeta_predictions_total counter", f"tripeta_predictions_total {self.predictions}",
                  "# TYPE tripeta_request_seconds histogram"]
        cum = 0
        for b, n in zip(self.BUCKETS, self.latency):
            cum += n
            lines.append(f'tripeta_request_seconds_bucket{{le="{b}"}} {cum}')
        cum += self.latency[-1]
        lines += [f'tripeta_request_seconds_bucket{{le="+Inf"}} {cum}', f"tripeta_request_seconds_sum {self.latency_sum:.6f}",
                  f"tripeta_request_seconds_count {cum}"]
        return "\n".join(lines) + "\n"


def create_app(model_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Taxi trip duration service", version=__version__)
    state = {"booster": None, "meta": None}
    recent: deque = deque(maxlen=RECENT_MAX)
    metrics = Metrics()

    def load():
        booster, meta = model.load(Path(model_dir or MODEL_DIR))
        state["booster"], state["meta"] = booster, meta

    try:
        load()
    except Exception:  # service still starts; /ready reports 503 until a model exists
        pass

    def version() -> str:
        m = state["meta"]
        return f"{m['name']}-{m.get('model_sha256', m['data_fingerprint'])[:12]}"

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        # Arbitrary 404 URLs must not create unbounded Prometheus label series.
        path = getattr(route, "path", "<unmatched>")
        metrics.observe(path, response.status_code, time.perf_counter() - t0)
        return response

    def run(trips: list[TripRequest]) -> list[Prediction]:
        if state["booster"] is None:
            raise HTTPException(503, "model not loaded")
        if "distance_miles" in state["booster"].feature_name() and any(t.distance_miles is None for t in trips):
            raise HTTPException(422, "distance_miles is required by the loaded legacy model")
        df = pd.DataFrame([{"pickup_zone": t.pickup_zone, "dropoff_zone": t.dropoff_zone, "hour": t.pickup_time.hour,
                            "dow": t.pickup_time.weekday(), "distance_miles": t.distance_miles,
                            "passengers": t.passengers} for t in trips])
        pred = model.predict(state["booster"], df)
        lo, hi = state["meta"]["interval_ratio"]
        recent.extend(df[state["booster"].feature_name()].to_dict("records"))
        metrics.predictions += len(trips)
        return [Prediction(predicted_minutes=round(float(p), 1), low_minutes=round(float(p * lo), 1),
                           high_minutes=round(float(p * hi), 1), model_version=version()) for p in pred]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response):
        if state["booster"] is None:
            response.status_code = 503
            return {"status": "no model loaded"}
        return {"status": "ready", "model_version": version()}

    @app.post("/predict", response_model=Prediction)
    def predict(trip: TripRequest):
        return run([trip])[0]

    @app.post("/predict/batch", response_model=list[Prediction])
    def predict_batch(req: BatchRequest):
        return run(req.trips)

    @app.get("/model")
    def model_card():
        if state["meta"] is None:
            raise HTTPException(503, "model not loaded")
        m = state["meta"]
        keys = ("name", "trained_at", "train_months", "valid_months", "calibration_months", "test_months", "train_rows",
                "features", "metrics", "segments", "interval_ratio", "interval_coverage", "data_fingerprint",
                "model_sha256", "prediction_time", "validation_gate_passed")
        return {k: m[k] for k in keys if k in m}

    @app.get("/drift")
    def drift_report():
        if state["meta"] is None:
            raise HTTPException(503, "model not loaded")
        rows = list(recent)
        if len(rows) < MIN_ROWS_FOR_DRIFT:
            return {"status": "insufficient_data", "rows": len(rows), "needed": MIN_ROWS_FOR_DRIFT}
        scores = drift.feature_psi(state["meta"]["drift_baseline"], rows)
        worst = max(scores, key=scores.get)
        return {"rows": len(rows), "psi": {k: round(v, 4) for k, v in scores.items()},
                "worst_feature": worst, "status": drift.status(scores[worst])}

    @app.get("/metrics")
    def prom():
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    return app


app = create_app()
