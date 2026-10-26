"""Integrity tests exercise phase boundaries without expensive model fitting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from recsys_research import run


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config = {
        "processed_dir": str(tmp_path / "processed"),
        "output_dir": str(tmp_path / "report"),
        "raw_dir": str(tmp_path / "raw"),
        "n_users": 2, "user_seed": 12, "day_seed": 13,
        "warmup_days": 1, "split_days": [10, 20, 30],
        "days_per_user": {"train": 2, "tune": 1, "select": 1, "test": 1},
        "embargo_days": 1,
    }
    events = pd.DataFrame({"uid": [1, 2], "timestamp": [100, 200]})
    queries = pd.DataFrame({
        "uid": [1, 2], "item_id": [10, 20], "day": [5, 35],
        "group_id": [1, 2], "label": [0, 1], "split": ["train", "test"],
    })
    matrix = pd.DataFrame({"example": [0.2, 0.3]})
    groups = {"manual": ["example"], "static": ["example"]}

    def fake_prepare_data(settings, output):
        output.mkdir(parents=True, exist_ok=True)
        events.to_parquet(output / "sample_events.parquet", index=False)
        queries.to_parquet(output / "queries.parquet", index=False)
        manifest = {"config": settings, "retained_events": len(events)}
        run.dump(manifest, output / "manifest.json")
        return events.copy(), queries.copy(), manifest

    monkeypatch.setattr(run, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(run, "build_features", lambda *_: (
        matrix.copy(), groups.copy(), {"seconds": 0.001}
    ))
    monkeypatch.setattr(run, "source_fingerprint", lambda: {
        "data.py": "fixed-data-source", "features.py": "fixed-feature-source"
    })
    run.prepare(config)
    return config, Path(config["processed_dir"])


def test_unchanged_preparation_roundtrips(prepared):
    config, _ = prepared
    matrix, queries, groups = run.load_matrix(config)
    assert matrix.shape == (2, 1)
    assert queries.label.tolist() == [0, 1]
    assert groups["manual"] == ["example"]


@pytest.mark.parametrize("filename,column,value,message", [
    ("queries.parquet", "label", 1, "queries"),
    ("features.parquet", "example", 9.0, "features"),
    ("sample_events.parquet", "timestamp", 999, "event history"),
])
def test_mutated_prepared_parquets_are_rejected(prepared, filename, column, value, message):
    config, root = prepared
    path = root / filename
    frame = pd.read_parquet(path)
    frame.loc[0, column] = value
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=message):
        run.load_matrix(config)


@pytest.mark.parametrize("filename,message", [
    ("manifest.json", "manifest"),
    ("feature_groups.json", "feature groups"),
])
def test_mutated_preparation_json_is_rejected(prepared, filename, message):
    config, root = prepared
    path = root / filename
    document = json.loads(path.read_text())
    document["unexpected_change"] = True
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        run.load_matrix(config)


def test_source_change_after_preparation_is_rejected(prepared, monkeypatch):
    config, _ = prepared
    monkeypatch.setattr(run, "source_fingerprint", lambda: {"features.py": "new-source"})
    with pytest.raises(ValueError, match="Source changed"):
        run.load_matrix(config)


def test_model_config_change_requires_new_preparation(prepared):
    config, _ = prepared
    changed = {**config, "embargo_days": 2}
    with pytest.raises(ValueError, match="Config differs"):
        run.load_matrix(changed)


@pytest.mark.parametrize("key,value", [
    ("n_users", 3),
    ("user_seed", 999),
    ("day_seed", 999),
    ("warmup_days", 2),
    ("split_days", [11, 21, 31]),
    ("days_per_user", {"train": 1, "tune": 1, "select": 1, "test": 1}),
    ("raw_dir", "some-other-raw-directory"),
])
def test_changed_data_settings_cannot_reuse_old_sample(prepared, monkeypatch, key, value):
    config, _ = prepared

    def forbidden(*_):
        raise AssertionError("Stale sample reached feature construction")

    monkeypatch.setattr(run, "build_features", forbidden)
    with pytest.raises(ValueError, match="Cached sample uses different data settings"):
        run.prepare({**config, key: value})


def test_final_evaluation_rejects_changed_preparation_lock_before_fitting(prepared, monkeypatch):
    config, root = prepared
    out = Path(config["output_dir"])
    preparation = root / "preparation_lock.json"
    run.dump({
        "config_sha256": run.fingerprint(config),
        "source_sha256": run.source_fingerprint(),
        "preparation_lock_sha256": run.sha256_file(preparation),
        "models": {},
    }, out / "selection_lock.json")
    old_lock = json.loads(preparation.read_text())
    old_lock["unexpected_change"] = True
    preparation.write_text(json.dumps(old_lock))

    def forbidden(*_):
        raise AssertionError("Changed preparation lock reached evaluation/fitting")

    monkeypatch.setattr(run, "load_matrix", forbidden)
    monkeypatch.setattr(run, "fit_one", forbidden)
    with pytest.raises(ValueError, match="Preparation lock changed"):
        run.final_evaluate(config)


def test_frozen_search_cannot_be_overwritten(prepared, monkeypatch):
    config, _ = prepared
    out = Path(config["output_dir"])
    run.dump({"frozen": True}, out / "selection_lock.json")
    monkeypatch.setattr(run, "load_matrix", lambda *_: pytest.fail("Frozen search loaded data"))
    with pytest.raises(FileExistsError, match="Search is frozen"):
        run.search(config)


def test_final_test_cannot_be_evaluated_twice(prepared, monkeypatch):
    config, _ = prepared
    out = Path(config["output_dir"])
    run.dump({"complete": True}, out / "test_results.json")
    monkeypatch.setattr(run, "fit_one", lambda *_: pytest.fail("Repeated test reached fitting"))
    with pytest.raises(FileExistsError, match="already evaluated"):
        run.final_evaluate(config)
