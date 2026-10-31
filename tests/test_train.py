import json

from tripeta import model
from tripeta.train import month_range


def test_month_range():
    assert month_range("2023-01..2023-03") == ["2023-01", "2023-02", "2023-03"]
    assert month_range("2022-11..2023-02") == ["2022-11", "2022-12", "2023-01", "2023-02"]
    assert month_range("2023-09") == ["2023-09"]


def test_champion_challenger_keeps_the_better_model(tmp_path, monkeypatch):
    """A worse challenger must NOT overwrite the registry; a better one must."""
    from argparse import Namespace

    from tripeta import train
    sample = train.ROOT / "data" / "sample"
    common = dict(data_dir=str(sample), train="2023-01", valid="2023-02", test="2023-02", sample_per_month=None,
                  test_sample=None, min_gain=0.0, registry=str(tmp_path / "reg"),
                  mlflow_uri=f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    first = train.run(Namespace(rounds=150, **common))
    assert first["promoted"]
    champion = json.loads((tmp_path / "reg" / "metadata.json").read_text())["metrics"]["test_mae_min"]
    weak = train.run(Namespace(rounds=3, **common))
    assert not weak["promoted"]
    assert json.loads((tmp_path / "reg" / "metadata.json").read_text())["metrics"]["test_mae_min"] == champion
    assert model.mae([1, 2], [1, 2]) == 0
