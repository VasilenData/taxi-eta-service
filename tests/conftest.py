import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
SAMPLE = ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def frames():
    from tripeta.data import load_month
    return (load_month(SAMPLE / "yellow_tripdata_2023-01.parquet"),
            load_month(SAMPLE / "yellow_tripdata_2023-02.parquet"))


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory, frames):
    """A real (small) model trained from the bundled sample, with calibrated interval and drift baseline."""
    import numpy as np

    from tripeta import drift, model
    train, valid = frames
    booster = model.train(train, valid, rounds=120)
    ratio = valid.duration_min / model.predict(booster, valid)
    folder = tmp_path_factory.mktemp("registry")
    meta = {"name": "taxi-trip-duration", "data_fingerprint": "testfp", "trained_at": "t", "train_months": ["2023-01"],
            "test_months": ["2023-02"], "train_rows": len(train), "features": [], "metrics": {}, "segments": {},
            "interval_ratio": [float(np.quantile(ratio, .1)), float(np.quantile(ratio, .9))], "interval_coverage": 0.8,
            "drift_baseline": drift.baseline_profile(train)}
    model.save(booster, meta, folder)
    return folder
