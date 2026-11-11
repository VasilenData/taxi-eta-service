import pytest

from tripeta import model
from tripeta.train import month_range, validate_splits


def test_month_range():
    assert month_range("2023-01..2023-03") == ["2023-01", "2023-02", "2023-03"]
    assert month_range("2022-11..2023-02") == ["2022-11", "2022-12", "2023-01", "2023-02"]
    assert month_range("2023-09") == ["2023-09"]
    assert month_range("0001-12..0002-01") == ["0001-12", "0002-01"]


def test_champion_challenger_keeps_the_better_model(tmp_path, monkeypatch, frames):
    """A worse challenger must NOT overwrite the registry; a better one must."""
    from argparse import Namespace

    from tripeta import train
    sample = train.ROOT / "data" / "sample"
    # Separate bundled-sample subsets exercise registry behavior. These synthetic
    # phase labels are a unit-test fixture, not evidence of future-month accuracy.
    training, heldout = frames
    phases = {"2023-01": training, "2023-02": heldout.iloc[::2], "2023-03": heldout.iloc[1::2]}
    monkeypatch.setattr(train, "load_months", lambda folder, months, sample: phases[months[0]])
    common = dict(data_dir=str(sample), train="2023-01", valid="2023-02", test="2023-03", sample_per_month=None,
                  test_sample=None, min_gain=0.0, registry=str(tmp_path / "reg"),
                  mlflow_uri=f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    first = train.run(Namespace(rounds=150, **common))
    assert first["promoted"]
    champion = model.load(tmp_path / "reg")[1]["metrics"]["test_mae_min"]
    weak = train.run(Namespace(rounds=3, **common))
    assert not weak["promoted"]
    assert model.load(tmp_path / "reg")[1]["metrics"]["test_mae_min"] == champion
    assert model.mae([1, 2], [1, 2]) == 0


@pytest.mark.parametrize("spec", ["2023-14..2024-01", "2023-00", "2023-1", "2023-01\n", "../2023-01", "2024-01..2023-12"])
def test_invalid_months_fail_instead_of_looping_or_reading_paths(spec):
    with pytest.raises(ValueError):
        month_range(spec)


@pytest.mark.parametrize("splits", [(["2023-01"], ["2023-02"], ["2023-02"]),
                                    (["2023-03"], ["2023-02"], ["2023-04"]),
                                    ([], ["2023-02"], ["2023-03"])])
def test_overlapping_or_reversed_splits_are_rejected(splits):
    with pytest.raises(ValueError, match="chronological"):
        validate_splits(*splits)


def test_promotion_uses_validation_even_when_test_prefers_the_champion(tmp_path, monkeypatch):
    from argparse import Namespace
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from tripeta import train

    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "model.txt").write_text("fixture")
    challenger = SimpleNamespace(best_iteration=1, model_to_string=lambda: "challenger-model-fixture")
    champion = object()

    def frame(folder, months, sample):
        result = pd.DataFrame({"duration_min": [10., 12.]})
        result.attrs["month"] = months[0]
        return result

    def predict(booster, data):
        # Validation prefers challenger; test prefers champion. Only the former
        # may influence promotion, regardless of the test report's direction.
        error = 0 if ((booster is challenger) == (data.attrs["month"] == "2023-02")) else 5
        return data.duration_min.to_numpy() + error

    monkeypatch.setattr(train, "load_months", frame)
    monkeypatch.setattr(train.model, "train", lambda *args: challenger)
    monkeypatch.setattr(train.model, "predict", predict)
    monkeypatch.setattr(train.model, "load", lambda path: (champion, {}))
    monkeypatch.setattr(train.model, "save", lambda *args: None)
    monkeypatch.setattr(train.model.Baseline, "fit", lambda df: SimpleNamespace(predict=lambda data: np.ones(len(data))))
    monkeypatch.setattr(train, "segment_report", lambda *args: {})
    monkeypatch.setattr(train.drift, "baseline_profile", lambda *args: {})
    for name in ("set_tracking_uri", "set_experiment", "log_params", "log_metrics", "log_artifacts"):
        monkeypatch.setattr(train.mlflow, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "start_run", lambda **kwargs: nullcontext())
    result = train.run(Namespace(train="2023-01", valid="2023-02", test="2023-03", data_dir="unused",
                                sample_per_month=None, test_sample=None, min_gain=0., rounds=1,
                                registry=str(registry), mlflow_uri="unused"))
    assert result["promoted"]
    assert result["metrics"]["validation_mae_min"] == 0
    assert result["metrics"]["test_mae_min"] == 5
