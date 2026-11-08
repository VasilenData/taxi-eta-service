"""Publication failures must never expose a model paired with another version's metadata."""
import json
from types import SimpleNamespace

import pytest

from tripeta import model


def candidate(version):
    return SimpleNamespace(model_to_string=lambda: version)


@pytest.fixture(autouse=True)
def fake_native_loader(monkeypatch):
    monkeypatch.setattr(model.lgb, "Booster", lambda model_str: model_str)


def test_failed_publication_preserves_complete_champion(tmp_path, monkeypatch):
    model.save(candidate("old"), {"version": "old"}, tmp_path)
    def fail_replace(*args):
        raise OSError("injected publication failure")
    monkeypatch.setattr(model.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        model.save(candidate("new"), {"version": "new"}, tmp_path)
    assert model.load(tmp_path) == ("old", {"version": "old"})
    assert not list(tmp_path.glob("*.tmp"))


def test_reader_keeps_one_snapshot_while_a_new_model_is_published(tmp_path, monkeypatch):
    model.save(candidate("old"), {"version": "old"}, tmp_path)
    def load_and_publish(model_str):
        model.save(candidate("new"), {"version": "new"}, tmp_path)
        return model_str
    monkeypatch.setattr(model.lgb, "Booster", load_and_publish)
    assert model.load(tmp_path) == ("old", {"version": "old"})
    assert model.load(tmp_path) == ("new", {"version": "new"})


def test_invalid_metadata_does_not_replace_champion(tmp_path):
    model.save(candidate("old"), {"version": "old"}, tmp_path)
    with pytest.raises(ValueError):
        model.save(candidate("new"), {"score": float("nan")}, tmp_path)
    assert model.load(tmp_path) == ("old", {"version": "old"})


def test_migration_prefers_bundle_without_overwriting_legacy_files(tmp_path):
    (tmp_path / "model.txt").write_text("legacy", encoding="utf-8")
    (tmp_path / "metadata.json").write_text('{"version": "legacy"}', encoding="utf-8")
    assert model.exists(tmp_path)
    assert model.load(tmp_path) == ("legacy", {"version": "legacy"})
    model.save(candidate("new"), {"version": "new"}, tmp_path)
    assert model.load(tmp_path) == ("new", {"version": "new"})
    assert (tmp_path / "model.txt").read_text() == "legacy"


@pytest.mark.parametrize("bundle", [{}, [], {"format_version": 1, "model": "", "metadata": {}}])
def test_invalid_current_bundle_fails_closed(tmp_path, bundle):
    (tmp_path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    (tmp_path / "model.txt").write_text("legacy", encoding="utf-8")
    (tmp_path / "metadata.json").write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid model bundle"):
        model.load(tmp_path)
