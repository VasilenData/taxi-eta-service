"""Model wrapper: LightGBM on log-duration, plus the naive baseline it has to beat."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CATEGORICAL, FEATURES

PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, cat_smooth=20, verbose=-1, seed=7, num_threads=16)


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def mape_median(y, p) -> float:
    y = np.asarray(y)
    return float(np.median(np.abs(y - np.asarray(p)) / y))


@dataclass
class Baseline:
    """duration = distance / median speed for that hour. What a first spreadsheet would do."""
    speed_by_hour: dict[int, float]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> Baseline:
        speed = df.distance_miles / (df.duration_min / 60.0)
        return cls({int(h): float(s) for h, s in speed.groupby(df.hour).median().items()})

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        speeds = df.hour.map(self.speed_by_hour).fillna(np.median(list(self.speed_by_hour.values())))
        return (df.distance_miles / speeds * 60.0).to_numpy()


def _frame(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    selected = FEATURES if features is None else features
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(FEATURES):
        raise ValueError("unsupported model feature contract")
    x = df[selected].copy()
    for c in CATEGORICAL:
        if c not in selected:
            continue
        x[c] = pd.Categorical(x[c], categories=range(1, 266))
    return x


def train(df: pd.DataFrame, valid: pd.DataFrame, rounds: int = 600,
          features: list[str] | None = None, threads: int | None = None) -> lgb.Booster:
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError("threads must be a positive integer")
    params = PARAMS if threads is None else {**PARAMS, "num_threads": threads}
    selected = FEATURES if features is None else features
    dtr = lgb.Dataset(_frame(df, selected), np.log1p(df.duration_min),
                      categorical_feature=[c for c in CATEGORICAL if c in selected])
    dva = lgb.Dataset(_frame(valid, selected), np.log1p(valid.duration_min), reference=dtr)
    return lgb.train(params, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(40, verbose=False)])


def predict(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return np.expm1(booster.predict(_frame(df, booster.feature_name()), num_iteration=booster.best_iteration or None))


def save(booster: lgb.Booster, meta: dict, folder: Path) -> None:
    """Publish model and metadata as one atomic local-file snapshot.

    Readers see either the complete previous bundle or the complete replacement.
    Promotion decisions still require a single writer; this is not a distributed
    registry or a guarantee against filesystem/hardware failure.
    """
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"format_version": 1, "model": booster.model_to_string(), "metadata": meta},
                         sort_keys=True, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder,
                                         prefix=".bundle-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, folder / "bundle.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def exists(folder: Path) -> bool:
    """Recognize new bundles and the checked-in legacy model layout."""
    return (folder / "bundle.json").exists() or (folder / "model.txt").exists()


def load(folder: Path) -> tuple[lgb.Booster, dict]:
    bundle = folder / "bundle.json"
    if bundle.exists():
        snapshot = json.loads(bundle.read_text(encoding="utf-8"))
        if (not isinstance(snapshot, dict) or snapshot.get("format_version") != 1
                or not isinstance(snapshot.get("model"), str) or not snapshot["model"].strip()
                or not isinstance(snapshot.get("metadata"), dict)):
            raise ValueError("invalid model bundle")
        return lgb.Booster(model_str=snapshot["model"]), snapshot["metadata"]
    # LightGBM stores byte offsets between trees. Git's Windows CRLF conversion
    # changes those offsets and can terminate the native loader before Python
    # can catch an exception. Universal-newline text loading restores LF first.
    return (lgb.Booster(model_str=(folder / "model.txt").read_text(encoding="utf-8")),
            json.loads((folder / "metadata.json").read_text(encoding="utf-8")))
