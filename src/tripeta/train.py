"""Train, evaluate on a held-out FUTURE month, track in MLflow, and promote only if the challenger wins.

    python -m tripeta.train --data-dir /data/taxi --train 2023-01..2023-08 --valid 2023-09 --test 2023-10

Time-based split: the model is always judged on a month it has never seen, later than its training data,
which is what production looks like. A random split would flatter it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlflow
import numpy as np

from . import __version__, drift, model
from .data import FEATURES, LIMITS, load_months, validate_month

ROOT = Path(__file__).resolve().parents[2]


def month_range(spec: str) -> list[str]:
    if ".." not in spec:
        return [validate_month(spec)]
    a, b = spec.split("..")
    validate_month(a)
    validate_month(b)
    (ya, ma), (yb, mb) = (map(int, a.split("-")), map(int, b.split("-")))
    if (ya, ma) > (yb, mb):
        raise ValueError("month range must be in chronological order")
    out = []
    while (ya, ma) <= (yb, mb):
        out.append(f"{ya:04d}-{ma:02d}")
        ma += 1
        ya, ma = (ya + 1, 1) if ma == 13 else (ya, ma)
    return out


def validate_splits(train: list[str], valid: list[str], test: list[str]) -> None:
    if not train or not valid or not test or not (max(train) < min(valid) and max(valid) < min(test)):
        raise ValueError("require non-overlapping chronological splits: train < validation < test")


def segment_report(df, pred) -> dict:
    d = df.assign(err=np.abs(df.duration_min - pred))
    return {
        "by_distance_miles": {str(k): round(float(v), 3) for k, v in
                              d.groupby(np.digitize(d.distance_miles, [1, 3, 6, 12]))["err"].mean().items()},
        "by_hour_block": {k: round(float(v), 3) for k, v in d.groupby(
            np.select([d.hour < 6, d.hour < 10, d.hour < 16, d.hour < 20], ["00-05", "06-09", "10-15", "16-19"], "20-23"))
            ["err"].mean().items()},
    }


def run(args) -> dict:
    tr_m, va_m, te_m = month_range(args.train), month_range(args.valid), month_range(args.test)
    validate_splits(tr_m, va_m, te_m)
    if not 0 <= args.min_gain < 1:
        raise ValueError("min_gain must be between 0 (inclusive) and 1 (exclusive)")
    train_df = load_months(args.data_dir, tr_m, args.sample_per_month)
    valid_df = load_months(args.data_dir, va_m, args.sample_per_month)
    test_df = load_months(args.data_dir, te_m, args.test_sample)

    base = model.Baseline.fit(train_df)
    t0 = time.time()
    booster = model.train(train_df, valid_df, args.rounds)
    fit_s = time.time() - t0
    pred = model.predict(booster, test_df)
    # Error range calibrated on the VALIDATION month (never on test): 10th/90th percentile of actual / predicted.
    ratio = valid_df.duration_min / model.predict(booster, valid_df)
    interval = [float(np.quantile(ratio, 0.10)), float(np.quantile(ratio, 0.90))]
    covered = float(np.mean((test_df.duration_min >= pred * interval[0]) & (test_df.duration_min <= pred * interval[1])))
    base_pred = base.predict(test_df)

    metrics = {
        "validation_mae_min": model.mae(valid_df.duration_min, model.predict(booster, valid_df)),
        "test_mae_min": model.mae(test_df.duration_min, pred),
        "test_median_ape": model.mape_median(test_df.duration_min, pred),
        "baseline_mae_min": model.mae(test_df.duration_min, base_pred),
        "baseline_median_ape": model.mape_median(test_df.duration_min, base_pred),
        "mean_duration_min": float(test_df.duration_min.mean()),
        "best_iteration": int(booster.best_iteration),
    }
    metrics["mae_improvement_vs_baseline"] = 1 - metrics["test_mae_min"] / metrics["baseline_mae_min"]

    fingerprint = hashlib.sha256(json.dumps(
        {"train": tr_m, "valid": va_m, "rows": [len(train_df), len(valid_df)], "features": FEATURES}).encode()).hexdigest()[:12]
    meta = {
        "name": "taxi-trip-duration", "package_version": __version__,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_months": tr_m, "valid_months": va_m, "test_months": te_m,
        "train_rows": len(train_df), "test_rows": len(test_df), "data_fingerprint": fingerprint,
        "interval_ratio": interval, "interval_coverage": covered,
        "features": FEATURES, "limits": LIMITS, "metrics": metrics, "fit_seconds": round(fit_s, 1),
        "segments": segment_report(test_df, pred), "drift_baseline": drift.baseline_profile(train_df),
    }

    # Model selection uses validation only. The future test month is reporting-only.
    registry = Path(args.registry)
    promoted, reason = True, "no champion yet"
    if (registry / "model.txt").exists():
        champ, champ_meta = model.load(registry)
        champ_mae = model.mae(valid_df.duration_min, model.predict(champ, valid_df))
        promoted = metrics["validation_mae_min"] < champ_mae * (1 - args.min_gain)
        reason = (f"validation challenger {metrics['validation_mae_min']:.3f} vs champion {champ_mae:.3f} min "
                  f"(needs {args.min_gain:.1%} gain)")
    if promoted:
        model.save(booster, meta, registry)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("taxi-trip-duration")
    with mlflow.start_run(run_name=f"lgbm-{fingerprint}"):
        mlflow.log_params({"promoted": promoted, "train": args.train, "valid": args.valid, "test": args.test,
                           "rows": len(train_df), **{k: v for k, v in model.PARAMS.items() if k != "verbose"}})
        mlflow.log_metrics({k: v for k, v in metrics.items()})
        tmp = Path(args.registry).parent / "_last_run"
        model.save(booster, meta, tmp)
        mlflow.log_artifacts(str(tmp), "model")

    print(json.dumps({"promoted": promoted, "reason": reason, **metrics}, indent=2))
    return {"promoted": promoted, "metrics": metrics, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train", default="2023-01..2023-08")
    ap.add_argument("--valid", default="2023-09")
    ap.add_argument("--test", default="2023-10")
    ap.add_argument("--sample-per-month", type=int, default=400_000)
    ap.add_argument("--test-sample", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--registry", default=str(ROOT / "models" / "current"))
    ap.add_argument("--mlflow-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
