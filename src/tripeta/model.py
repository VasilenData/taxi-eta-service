"""Model wrapper: LightGBM on log-duration, plus the naive baseline it has to beat."""
from __future__ import annotations

import json
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


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for c in CATEGORICAL:
        x[c] = pd.Categorical(x[c], categories=range(1, 266))
    return x


def train(df: pd.DataFrame, valid: pd.DataFrame, rounds: int = 600) -> lgb.Booster:
    dtr = lgb.Dataset(_frame(df), np.log1p(df.duration_min), categorical_feature=CATEGORICAL)
    dva = lgb.Dataset(_frame(valid), np.log1p(valid.duration_min), reference=dtr)
    return lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(40, verbose=False)])


def predict(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    return np.expm1(booster.predict(_frame(df), num_iteration=booster.best_iteration or None))


def save(booster: lgb.Booster, meta: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(folder / "model.txt"))
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def load(folder: Path) -> tuple[lgb.Booster, dict]:
    return (lgb.Booster(model_file=str(folder / "model.txt")),
            json.loads((folder / "metadata.json").read_text()))
