"""Booking-time duration evaluation with disjoint fit, selection, calibration and test months."""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import __version__, drift, model
from .data import BOOKING_FEATURES, LIMITS, load_month, validate_month
from .evidence import data_fingerprint, frame_fingerprint, sha256


def months(spec: str) -> list[str]:
    parts = spec.split("..")
    if len(parts) not in (1, 2):
        raise ValueError("month range must be YYYY-MM or YYYY-MM..YYYY-MM")
    start, end = validate_month(parts[0]), validate_month(parts[-1])
    if start > end:
        raise ValueError("month range must be chronological")
    return pd.period_range(start, end, freq="M").astype(str).tolist()


def validate_phases(phases: dict[str, list[str]]) -> None:
    if list(phases) != ["train", "validation", "calibration", "test"]:
        raise ValueError("four ordered phases required: train, validation, calibration, test")
    previous = None
    for values in phases.values():
        if not values or values != sorted(set(values)):
            raise ValueError("each phase requires unique chronological months")
        for value in values:
            validate_month(value)
        if previous is not None and previous >= values[0]:
            raise ValueError("phases must be strictly chronological and disjoint")
        previous = values[-1]


def phase_end(values: list[str]) -> str:
    return (pd.Timestamp(values[-1] + "-01") + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")


@dataclass
class RouteBaseline:
    routes: dict[tuple[int, int], float]
    hours: dict[int, float]
    global_median: float
    min_support: int = 20

    @classmethod
    def fit(cls, frame: pd.DataFrame, min_support: int = 20) -> RouteBaseline:
        if frame.empty or type(min_support) is not int or min_support < 1:
            raise ValueError("nonempty training data and positive route support required")
        if not np.isfinite(frame.duration_min).all() or (frame.duration_min <= 0).any():
            raise ValueError("baseline targets must be positive and finite")
        grouped = frame.groupby(["pickup_zone", "dropoff_zone"]).duration_min.agg(["median", "size"])
        routes = {tuple(map(int, key)): float(row["median"])
                  for key, row in grouped.iterrows() if row["size"] >= min_support}
        hours = {int(key): float(value) for key, value in frame.groupby("hour").duration_min.median().items()}
        return cls(routes, hours, float(frame.duration_min.median()), min_support)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([self.routes.get((int(p), int(d)), self.hours.get(int(h), self.global_median))
                         for p, d, h in frame[["pickup_zone", "dropoff_zone", "hour"]].itertuples(index=False, name=None)])

    def as_dict(self) -> dict:
        return {"routes": {f"{p}:{d}": value for (p, d), value in self.routes.items()},
                "hour_medians": self.hours, "global_median": self.global_median, "min_route_support": self.min_support}


def interval_ratio(actual, prediction) -> list[float]:
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    if len(actual) < 20 or actual.shape != prediction.shape:
        raise ValueError("at least 20 paired calibration observations required")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all() or np.any(actual <= 0) or np.any(prediction <= 0):
        raise ValueError("calibration values must be positive and finite")
    low, high = np.quantile(actual / prediction, [0.1, 0.9])
    return [float(min(low, 1)), float(max(high, 1))]


def metrics(actual, prediction, interval: list[float]) -> dict:
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    if actual.shape != prediction.shape or not len(actual) or not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise ValueError("predictions must be positive, finite and paired with nonempty targets")
    low, high = prediction * interval[0], prediction * interval[1]
    return {"rows": len(actual), "mae_min": model.mae(actual, prediction),
            "median_ape": model.mape_median(actual, prediction),
            "coverage": float(np.mean((actual >= low) & (actual <= high))),
            "mean_interval_width_min": float(np.mean(high - low))}


def load_phase(directory: Path, values: list[str], sample: int, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    cutoff = phase_end(values)
    frames, sources = [], []
    for month in values:
        path = directory / f"yellow_tripdata_{month}.parquet"
        digest = sha256(path)
        month_seed = seed + int(month[:4]) * 12 + int(month[-2:])
        frame = load_month(path, sample, month_seed, expected_month=month,
                           include_identity=True, label_before=cutoff)
        if frame.empty:
            raise ValueError(f"no valid trips in {month}")
        frame["record_id"] = month + ":" + frame.source_row.astype(str)
        if not (frame.dropoff_at < pd.Timestamp(cutoff)).all():
            raise AssertionError("a target was unavailable at the phase boundary")
        with duckdb.connect() as con:
            raw_rows = con.execute("select count(*) from read_parquet(?)", [str(path)]).fetchone()[0]
        if sha256(path) != digest:
            raise ValueError(f"source changed during loading: {month}")
        sources.append({"month": month, "filename": path.name, "sha256": digest, "bytes": path.stat().st_size,
                        "raw_rows": raw_rows, "sample_rows": len(frame), "sample_cap": sample, "seed": month_seed,
                        "label_before": cutoff, "frame_sha256": frame_fingerprint(frame)})
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.record_id.duplicated().any():
        raise ValueError("duplicate source-row identities")
    return combined, sources


def predictions(frame: pd.DataFrame, learned, baseline, intervals: dict, route_baseline: RouteBaseline) -> pd.DataFrame:
    columns = ["record_id", "month", "source_row", *BOOKING_FEATURES, "duration_min"]
    result = frame[columns].copy()
    result["seen_supported_route"] = [(int(p), int(d)) in route_baseline.routes
                                      for p, d in frame[["pickup_zone", "dropoff_zone"]].itertuples(index=False, name=None)]
    for name, values in (("lightgbm", learned), ("baseline", baseline)):
        result[f"{name}_prediction"] = values
        result[f"{name}_low"] = values * intervals[name][0]
        result[f"{name}_high"] = values * intervals[name][1]
    return result


def slices(frame: pd.DataFrame) -> list[dict]:
    result = []
    for field in ("seen_supported_route", "hour", "passengers"):
        for value, group in frame.groupby(field):
            for name in ("lightgbm", "baseline"):
                actual = group.duration_min.to_numpy()
                result.append({"slice": field, "value": str(value), "model": name, "rows": len(group),
                               "mae_min": model.mae(actual, group[f"{name}_prediction"]),
                               "coverage": float(np.mean((actual >= group[f"{name}_low"]) &
                                                          (actual <= group[f"{name}_high"])) )})
    return result


def report_html(report: dict) -> str:
    rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{m['mae_min']:.3f} min</td>"
                   f"<td>{m['coverage']:.1%}</td><td>{m['mean_interval_width_min']:.2f} min</td></tr>"
                   for name, m in report["test_metrics"].items())
    source_rows = "".join(f"<tr><td>{s['month']}</td><td>{s['raw_rows']:,}</td><td>{s['sample_rows']:,}</td></tr>"
                          for phase in report["sources"].values() for s in phase)
    verdict = "passed" if report["validation_gate_passed"] else "did not pass; baseline retained as reference"
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Taxi booking-time evaluation</title><style>body{{background:#f5f7fa;color:#172333;font:17px/1.6 system-ui;
max-width:980px;margin:50px auto;padding:0 24px}}h1{{line-height:1.15}}table{{width:100%;border-collapse:collapse;
background:white}}td,th{{padding:12px;border-bottom:1px solid #dce3eb;text-align:left}}a{{color:#00776d}}
.tag{{color:#00776d;font-weight:700}}section{{margin:32px 0}}</style>
<p class="tag">TEMPORAL EVALUATION &middot; BOOKING FEATURES ONLY</p><h1>Trip duration before the ride</h1>
<p>Uses pickup/drop-off zones, pickup hour and weekday, and passenger count. Recorded trip distance is excluded
from model inputs. Historical TLC labels are retrospective; this is not a live routing or traffic system.</p>
<section><h2>Untouched future test</h2><table><tr><th>Model</th><th>MAE</th><th>Interval coverage</th>
<th>Mean interval width</th></tr>{rows}</table><p>The validation-only learned-model gate {verdict}.
Final test rows do not decide model selection, interval calibration or registry promotion.</p></section>
<section><h2>Four distinct phases</h2><p>Train: {html.escape(', '.join(report['phases']['train']))};
validation: {html.escape(', '.join(report['phases']['validation']))}; calibration:
{html.escape(', '.join(report['phases']['calibration']))}; test: {html.escape(', '.join(report['phases']['test']))}.</p>
<p>Early stopping uses validation only. The 10th/90th ratio interval is fitted on calibration, expanded to contain
the point prediction, and assessed on test. Its nominal 80% level is not a coverage guarantee under time shift.</p></section>
<section><h2>Source and sample coverage</h2><table><tr><th>Month</th><th>Raw source rows</th><th>Sampled eligible trips</th>
</tr>{source_rows}</table><p>Source row IDs identify paired predictions. Eligibility filters use completed-trip
duration, fare and recorded distance; evaluation therefore applies to the eligible-trip cohort, not all requests.
Targets ending beyond the phase boundary are excluded before sampling. Monthly public data arrives after travel.</p></section>
<section><h2>Inspect the evidence</h2><p><a href="report.json">Protocol, metrics and source hashes</a> &middot;
<a href="predictions_test.csv.gz">Every held-out prediction (gzip CSV)</a> &middot;
<a href="slices.csv">Route/hour/passenger slices</a> &middot; <a href="manifest.json">Artifact checksums</a></p>
<p>Training-only baseline: route median for routes with at least 20 samples, otherwise hourly median, then global
median. No traffic, route planner, causal impact or production uptime claim. Historical distance-assisted scores
remain a separate experiment.</p></section></html>"""


def publish_report(output: Path, report: dict, booster, metadata: dict,
                   phase_predictions: dict[str, pd.DataFrame], baseline: RouteBaseline) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".booking-", dir=output.parent))
    try:
        model.save(booster, metadata, staging / "model")
        for phase, frame in phase_predictions.items():
            frame.to_csv(staging / f"predictions_{phase}.csv.gz", index=False, lineterminator="\n",
                         compression={"method": "gzip", "mtime": 0})
        pd.DataFrame(report["slices"]).to_csv(staging / "slices.csv", index=False, lineterminator="\n")
        (staging / "baseline.json").write_text(json.dumps(baseline.as_dict(), indent=2, allow_nan=False), encoding="utf-8")
        (staging / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        (staging / "index.html").write_text(report_html(report), encoding="utf-8")
        manifest = {str(p.relative_to(staging)).replace("\\", "/"): {"sha256": sha256(p), "bytes": p.stat().st_size}
                    for p in sorted(staging.rglob("*")) if p.is_file()}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def run(args) -> dict:
    phases = {"train": months(args.train), "validation": months(args.valid),
              "calibration": months(args.calibration), "test": months(args.test)}
    validate_phases(phases)
    if any(type(v) is not int or v < 1 for v in (args.train_sample, args.evaluation_sample, args.rounds, args.threads)):
        raise ValueError("sample caps and rounds must be positive integers")
    if type(args.seed) is not int or not 0 <= args.seed < 2**31 - 120000:
        raise ValueError("seed must be a non-negative bounded integer")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior evidence: {output}")
    frames, sources = {}, {}
    # Do not even load final test labels until fit, selection and intervals are frozen.
    for phase in ("train", "validation", "calibration"):
        frames[phase], sources[phase] = load_phase(Path(args.data_dir), phases[phase],
                                                 args.train_sample if phase == "train" else args.evaluation_sample, args.seed)
    baseline = RouteBaseline.fit(frames["train"])
    started = time.perf_counter()
    booster = model.train(frames["train"], frames["validation"], args.rounds, features=BOOKING_FEATURES, threads=args.threads)
    fit_seconds = time.perf_counter() - started
    predict = lambda frame: model.predict(booster, frame)  # noqa: E731
    validation_prediction, validation_base = predict(frames["validation"]), baseline.predict(frames["validation"])
    validation_scores = {"lightgbm": model.mae(frames["validation"].duration_min, validation_prediction),
                         "baseline": model.mae(frames["validation"].duration_min, validation_base)}
    gate_passed = validation_scores["lightgbm"] < validation_scores["baseline"]
    calibration_prediction, calibration_base = predict(frames["calibration"]), baseline.predict(frames["calibration"])
    intervals = {"lightgbm": interval_ratio(frames["calibration"].duration_min, calibration_prediction),
                 "baseline": interval_ratio(frames["calibration"].duration_min, calibration_base)}
    frames["test"], sources["test"] = load_phase(Path(args.data_dir), phases["test"], args.evaluation_sample, args.seed)
    test_prediction, test_base = predict(frames["test"]), baseline.predict(frames["test"])
    evidence = {phase: predictions(frames[phase], learned, base, intervals, baseline) for phase, learned, base in (
        ("validation", validation_prediction, validation_base),
        ("calibration", calibration_prediction, calibration_base), ("test", test_prediction, test_base))}
    test_metrics = {"lightgbm": metrics(frames["test"].duration_min, test_prediction, intervals["lightgbm"]),
                    "baseline": metrics(frames["test"].duration_min, test_base, intervals["baseline"])}
    source_dir = Path(__file__).parent
    code = {p.name: sha256(p) for p in sorted(source_dir.glob("*.py"))}
    model_hash = hashlib.sha256(booster.model_to_string().encode("utf-8")).hexdigest()
    fingerprint = data_fingerprint({phase: frames[phase] for phase in ("train", "validation", "calibration")})
    segments = slices(evidence["test"])
    flat_metrics = {"validation_mae_min": validation_scores["lightgbm"],
                    "validation_baseline_mae_min": validation_scores["baseline"],
                    "test_mae_min": test_metrics["lightgbm"]["mae_min"],
                    "baseline_mae_min": test_metrics["baseline"]["mae_min"]}
    metadata = {"name": "taxi-booking-duration", "package_version": __version__, "model_sha256": model_hash,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "features": BOOKING_FEATURES,
                "train_months": phases["train"], "valid_months": phases["validation"],
                "calibration_months": phases["calibration"], "test_months": phases["test"],
                "train_rows": len(frames["train"]), "test_rows": len(frames["test"]),
                "data_fingerprint": fingerprint, "data_fingerprint_format": "training-csv-v1",
                "interval_ratio": intervals["lightgbm"], "interval_coverage": test_metrics["lightgbm"]["coverage"],
                "metrics": flat_metrics, "segments": segments, "limits": LIMITS,
                "validation_gate_passed": gate_passed, "prediction_time": "booking; recorded distance excluded",
                "drift_baseline": drift.baseline_profile(frames["train"], features=BOOKING_FEATURES)}
    report = {"protocol_version": 1, "phases": phases, "sources": sources, "features": BOOKING_FEATURES,
              "sample_method": "per-month seeded hash(source_row,seed) top-N after eligibility and label cutoff",
              "seed": args.seed, "model_params": {**model.PARAMS, "num_threads": args.threads}, "rounds_cap": args.rounds,
              "best_iteration": booster.best_iteration, "fit_seconds": fit_seconds,
              "validation_mae_min": validation_scores, "validation_gate_passed": gate_passed,
              "interval_ratios": intervals, "nominal_coverage": 0.8, "test_metrics": test_metrics,
              "frame_sha256": {name: frame_fingerprint(frame) for name, frame in frames.items()},
              "data_fingerprint": fingerprint, "model_sha256": model_hash, "code_sha256": code,
              "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                          for name in ("duckdb", "lightgbm", "pandas", "numpy")}},
              "slices": segments, "limitations": [
                  "Retrospective monthly TLC data; no live traffic or route planning.",
                  "Completed-trip eligibility uses realized distance/fare/duration; cohort is not all booking requests.",
                  "Ratio intervals calibrated on a different month; nominal 80% is not guaranteed under shift.",
                  "Single bounded seeded sample per month; no measured production or causal business impact.",
                  "Test inspection must not be reused for tuning. Registry gate compares validation only.",
                  "Local single-writer report publication and registry update are separate atomic boundaries."]}
    publish_report(output, report, booster, metadata, evidence, baseline)
    if args.registry and gate_passed:
        model.save(booster, metadata, Path(args.registry))
    print(json.dumps({"output": str(output), "validation_gate_passed": gate_passed, "test_metrics": test_metrics}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry", help="optional separate booking registry; updated only if validation beats baseline")
    parser.add_argument("--train", default="2023-01..2023-03")
    parser.add_argument("--valid", default="2023-04")
    parser.add_argument("--calibration", default="2023-05")
    parser.add_argument("--test", default="2023-06")
    parser.add_argument("--train-sample", type=int, default=100000)
    parser.add_argument("--evaluation-sample", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
