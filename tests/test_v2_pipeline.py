"""Exercise the confirmation-release barrier without fitting any model."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from recsys_history_v2 import run


@pytest.fixture
def locked_run(tmp_path, monkeypatch):
    out, root = tmp_path / "report", tmp_path / "processed"
    out.mkdir()
    root.mkdir()
    config = {"output_dir": str(out), "processed_dir": str(root),
              "bootstrap_samples": 10, "is_pilot": True}
    q = pd.DataFrame({
        "uid": [1, 1, 1, 10, 10, 11, 11],
        "item_id": [1, 2, 3, 100, 101, 102, 103],
        "day": [30, 120, 150, 180, 180, 180, 180],
        "group_id": [1, 2, 3, 1000, 1000, 1100, 1100],
        "split": ["train", "tune", "select", "test", "test", "test", "test"],
        "label": [0, 1, 0, np.nan, np.nan, np.nan, np.nan],
    })
    x = pd.DataFrame({"m": [1, 2, 3, 101, 102, 103, 104]})
    keys = ["uid", "item_id", "day", "group_id"]
    sealed = q.loc[q.split.eq("test"), keys].reset_index(drop=True)
    sealed["label"] = np.array([0, 1, 1, 0], dtype=np.uint8)
    label_path = root / "test_labels.parquet"
    sealed.to_parquet(label_path, index=False)
    preparation_path = root / "preparation_lock.json"
    preparation_path.write_text(json.dumps({"prepared": True}))
    registration_path = out / "registration.json"
    registration_path.write_text(json.dumps({"registered": True}))
    sources = {"frozen_v2.py": "fixed-source-fingerprint"}
    monkeypatch.setattr(run, "source_hashes", lambda: dict(sources))
    prediction_values = {
        "manual": [.3, .7, .6, .4],
        "all_auto": [.25, .75, .65, .35],
        "selected_auto": [.2, .8, .7, .3],
        "random_0": [.2, .8, .7, .3],
        "history_global": [.1, .9, .8, .2],
        "history_pure": [.4, .6, .55, .45],
    }
    models = {}
    for family in prediction_values:
        checkpoint = out / f"{family}.{'pt' if family.startswith('history') else 'cbm'}"
        checkpoint.write_text("frozen model " + family)
        models[family] = {"model_paths": [str(checkpoint)],
                          "model_sha256": {str(checkpoint): run.sha256_file(checkpoint)},
                          "columns": ["m"]}
        if family.startswith("history"):
            models[family]["kind"] = "history"
    lock = {"config_sha256": run.fingerprint(config), "sources": dict(sources),
            "registration_sha256": run.sha256_file(registration_path),
            "preparation_sha256": run.sha256_file(preparation_path), "models": models,
            "blend": {"auto_weight": .5}}
    (out / "selection_lock.json").write_text(json.dumps(lock))
    events = []

    def load_prepared(current_config):
        events.append(("load_prepared",))
        assert current_config == config
        assert q.loc[q.split.eq("test"), "label"].isna().all()
        return x, q, {}, {"prepared": True}

    class FakeTabular:
        def load_model(self, path):
            self.family = Path(path).stem
            return self

        def predict_proba(self, frame):
            assert frame.m.tolist() == [101, 102, 103, 104]
            events.append(("predict", self.family))
            p = np.array(prediction_values[self.family])
            return np.column_stack([1 - p, p])

    def mean_history(paths, current_root, current_q, split, current_config):
        assert split == "test" and current_root == root and current_config == config
        assert current_q.loc[current_q.split.eq("test"), "label"].isna().all()
        family = Path(paths[0]).stem
        events.append(("predict", family))
        return np.array(prediction_values[family])

    monkeypatch.setattr(run, "load_prepared", load_prepared)
    monkeypatch.setattr(run, "CatBoostClassifier", FakeTabular)
    monkeypatch.setattr(run, "mean_history", mean_history)
    original_read_parquet = pd.read_parquet

    def guarded_read_parquet(path, *args, **kwargs):
        if Path(path) == label_path:
            events.append(("read_confirmation_labels",))
            # These assertions run exactly at the first label deserialization.
            assert {e[1] for e in events if e[0] == "predict"} == set(models)
            assert len([e for e in events if e[0] == "predict"]) == len(models)
            assert (out / "blind_predictions.parquet").exists()
            assert (out / "prediction_lock.json").exists()
            assert not (out / "test_results.json").exists()
            blind = original_read_parquet(out / "blind_predictions.parquet")
            assert "label" not in blind
            assert set(blind.columns) == set(keys) | set(models) | {"hybrid"}
            for family, expected in prediction_values.items():
                np.testing.assert_allclose(blind[family], expected, rtol=0, atol=0)
            np.testing.assert_allclose(blind.hybrid, run.blend(
                np.array(prediction_values["selected_auto"]),
                np.array(prediction_values["history_global"]), .5))
            prediction_lock = json.loads((out / "prediction_lock.json").read_text())
            assert prediction_lock["labels_opened"] is False
            assert prediction_lock["selection_sha256"] == run.sha256_file(out / "selection_lock.json")
            assert prediction_lock["blind_predictions_sha256"] == run.sha256_file(out / "blind_predictions.parquet")
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(run.pd, "read_parquet", guarded_read_parquet)
    return {"config": config, "out": out, "root": root, "q": q, "sealed": sealed,
            "events": events, "lock": lock, "models": models, "label_path": label_path,
            "sources": sources, "original_read": original_read_parquet}


def test_prepare_refuses_published_run_without_prepared_cache(tmp_path):
    out = tmp_path / "report"
    out.mkdir()
    lock = out / "preparation_lock.json"
    lock.write_text("published preparation")
    root = tmp_path / "missing-cache"
    with pytest.raises(FileExistsError, match="Published run is locked"):
        run.prepare({"output_dir": str(out), "processed_dir": str(root)})
    assert lock.read_text() == "published preparation"
    assert not root.exists()


def test_changed_registration_aborts_before_loading_prepared_data(locked_run):
    fixture = locked_run
    (fixture["out"] / "registration.json").write_text("changed registration")
    with pytest.raises(ValueError, match="Registration changed after selection"):
        run.evaluate_locked(fixture["config"])
    assert fixture["events"] == []
    assert not (fixture["out"] / "prediction_lock.json").exists()


def test_all_family_predictions_and_hash_lock_exist_before_first_label_read(locked_run):
    fixture = locked_run
    run.evaluate_locked(fixture["config"])
    events, out = fixture["events"], fixture["out"]
    assert sum(e[0] == "read_confirmation_labels" for e in events) == 1
    label_position = next(i for i, e in enumerate(events) if e[0] == "read_confirmation_labels")
    assert all(i < label_position for i, e in enumerate(events) if e[0] == "predict")
    saved = fixture["original_read"](out / "test_predictions.parquet")
    assert saved.label.tolist() == [0, 1, 1, 0]
    result = json.loads((out / "test_results.json").read_text())
    assert set(result["models"]) == set(fixture["models"]) | {"hybrid"}
    assert result["primary_history_vs_autofe"]["logloss"]["delta"] < 0
    assert result["is_pilot"] is True
    assert result["predictions_sha256"] == run.sha256_file(out / "test_predictions.parquet")


def test_changed_model_bytes_abort_before_confirmation_labels(locked_run):
    fixture = locked_run
    model_path = fixture["models"]["history_global"]["model_paths"][0]
    Path(model_path).write_text("different checkpoint bytes")
    with pytest.raises(ValueError, match="Frozen model changed"):
        run.evaluate_locked(fixture["config"])
    assert not any(e[0] == "read_confirmation_labels" for e in fixture["events"])
    assert not (fixture["out"] / "prediction_lock.json").exists()
    assert not (fixture["out"] / "test_results.json").exists()


@pytest.mark.parametrize("corruption", ["replacement_path", "missing_hash", "duplicate_path"])
def test_every_prediction_checkpoint_requires_exact_unique_hash_coverage(locked_run, corruption):
    fixture = locked_run
    specification = fixture["lock"]["models"]["history_global"]
    if corruption == "replacement_path":
        replacement = fixture["out"] / "replacement.pt"
        replacement.write_text("valid-looking but unregistered checkpoint")
        specification["model_paths"] = [str(replacement)]
    elif corruption == "missing_hash":
        specification["model_sha256"] = {}
    else:
        specification["model_paths"] = specification["model_paths"] * 2
    (fixture["out"] / "selection_lock.json").write_text(json.dumps(fixture["lock"]))
    with pytest.raises(ValueError, match="not exactly covered by frozen hashes"):
        run.evaluate_locked(fixture["config"])
    assert not any(e[0] == "read_confirmation_labels" for e in fixture["events"])
    assert not any(e == ("predict", "history_global") for e in fixture["events"])
    assert not (fixture["out"] / "prediction_lock.json").exists()


@pytest.mark.parametrize("bad_values", [
    [np.nan, .9, .8, .2], [np.inf, .9, .8, .2], [-.1, .9, .8, .2],
    [.1, 1.1, .8, .2], [.1, .9, .8], [[.1], [.9], [.8], [.2]],
])
def test_invalid_family_probabilities_abort_before_label_release(locked_run, monkeypatch, bad_values):
    fixture = locked_run
    monkeypatch.setattr(run, "mean_history", lambda *args: np.asarray(bad_values))
    with pytest.raises(ValueError, match="Invalid prediction shape|Invalid probabilities"):
        run.evaluate_locked(fixture["config"])
    assert not any(e[0] == "read_confirmation_labels" for e in fixture["events"])
    assert not (fixture["out"] / "blind_predictions.parquet").exists()
    assert not (fixture["out"] / "prediction_lock.json").exists()


def test_invalid_hybrid_probabilities_are_rejected_even_when_all_models_are_valid(locked_run, monkeypatch):
    fixture = locked_run
    monkeypatch.setattr(run, "blend", lambda *args: np.full(4, np.nan))
    with pytest.raises(ValueError, match="Invalid probabilities before label release: hybrid"):
        run.evaluate_locked(fixture["config"])
    assert {e[1] for e in fixture["events"] if e[0] == "predict"} == set(fixture["models"])
    assert not any(e[0] == "read_confirmation_labels" for e in fixture["events"])
    assert not (fixture["out"] / "prediction_lock.json").exists()


@pytest.mark.parametrize("drift", ["config", "source", "preparation"])
def test_frozen_drift_rejected_before_loading_or_label_read(locked_run, monkeypatch, drift):
    fixture = locked_run
    supplied_config = dict(fixture["config"])
    if drift == "config":
        supplied_config["bootstrap_samples"] += 1
    elif drift == "source":
        monkeypatch.setattr(run, "source_hashes", lambda: {"frozen_v2.py": "changed-source"})
    else:
        (fixture["root"] / "preparation_lock.json").write_text("{\"prepared\": false}\n")
    with pytest.raises(ValueError, match="Frozen implementation/config|Preparation changed"):
        run.evaluate_locked(supplied_config)
    assert not fixture["events"]
    assert not (fixture["out"] / "blind_predictions.parquet").exists()


def test_misaligned_targets_cannot_be_attached_to_locked_predictions(locked_run):
    fixture = locked_run
    fixture["sealed"].iloc[::-1].reset_index(drop=True).to_parquet(fixture["label_path"], index=False)
    with pytest.raises(ValueError, match="do not align"):
        run.evaluate_locked(fixture["config"])
    assert any(e[0] == "read_confirmation_labels" for e in fixture["events"])
    assert (fixture["out"] / "prediction_lock.json").exists()
    assert not (fixture["out"] / "test_results.json").exists()


def test_locked_predictions_prevent_second_confirmation_release(locked_run):
    fixture = locked_run
    (fixture["out"] / "prediction_lock.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already released or predictions locked"):
        run.evaluate_locked(fixture["config"])
    assert not fixture["events"]

