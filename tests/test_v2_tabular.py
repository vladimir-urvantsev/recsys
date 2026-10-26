"""Verify causal development boundaries and the v2 selector contract."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from recsys_history_v2 import tabular


def fixture_data():
    q = pd.DataFrame({
        "split": ["train"] * 4 + ["tune"] * 2 + ["select"] * 2 + ["test"] * 2,
        "label": [0, 1, 0, 1, 0, 1, 0, 1, np.nan, np.nan],
        "uid": [1, 2, 1, 2, 1, 2, 1, 2, 9001, 9002],
        "day": [30, 30, 31, 31, 210, 210, 240, 240, 270, 270],
    })
    x = pd.DataFrame({
        "m": np.arange(10, dtype=float), "train_constant": [0] * 8 + [999, 999],
        "a": np.arange(10, dtype=float) % 3,
        "b": np.arange(10, dtype=float) % 4,
        "c": np.arange(10, dtype=float) % 5,
    })
    groups = {"manual": ["m", "train_constant"], "static": ["m"],
              "window_2": ["a"], "window_3": ["b"], "decay_3": ["c"]}
    config = {"iterations": 8, "threads": 1, "early_stopping": 2,
              "hyper_grid": [{"depth": 2}, {"depth": 3}], "model_seeds": [17, 29, 43],
              "selection_prefixes": [1, 2, 3], "selection_feature_penalty": 1e-6,
              "random_controls": 3, "random_seed": 8821}
    return x, q, groups, config


class FakeModel:
    def __init__(self, columns, params, seed, iterations):
        self.columns, self.params, self.seed = list(columns), dict(params), int(seed)
        self.tree_count_ = int(iterations)

    def save_model(self, filename):
        Path(filename).write_text(json.dumps({"columns": self.columns, "params": self.params,
                                            "seed": self.seed, "iterations": self.tree_count_}))

    def get_feature_importance(self):
        return np.array([{"a": 3, "b": 2, "c": 1}.get(c, 0) for c in self.columns])


def install_mock_learner(monkeypatch):
    calls = []
    def fit(x, q, columns, params, config, seed, iterations=None, tune=True):
        assert set(q.split) <= {"train", "tune", "select"}
        assert set(q.uid) == {1, 2}
        assert q.label.notna().all()
        assert len(x) == len(q) == 8
        assert "train_constant" not in columns
        calls.append({"columns": list(columns), "params": dict(params), "iterations": iterations,
                      "tune": tune, "seed": seed})
        return FakeModel(columns, params, seed, iterations if iterations else params["depth"] + 1), .01

    def load(path):
        value = json.loads(Path(path).read_text())
        return FakeModel(value["columns"], value["params"], value["seed"], value["iterations"])

    def predict(model, x, columns, mask, config):
        indices = np.flatnonzero(mask.to_numpy())
        assert max(indices) < 8  # no confirmation prediction is permitted
        is_tune = bool(indices[0] == 4)
        if is_tune:
            quality = (.85 if model.params["depth"] == 2 else .7) if len(columns) == 1 else (.9 if model.params["depth"] == 3 else .6)
        else:
            quality = .95 if columns == ["m", "a"] else .7
        return np.resize(np.array([1 - quality, quality]), len(indices))

    monkeypatch.setattr(tabular, "_fit_model", fit)
    monkeypatch.setattr(tabular, "_load_model", load)
    monkeypatch.setattr(tabular, "_predict_model", predict)
    return calls


def test_selection_excludes_confirmation_and_uses_fixed_manual_backbone(tmp_path, monkeypatch):
    x, q, groups, config = fixture_data()
    x.index = np.arange(100, 110)
    q.index = np.arange(200, 210)
    calls = install_mock_learner(monkeypatch)
    result = tabular.train_tabular(x, q, groups, config, tmp_path)
    assert result["metadata"]["selected_groups"] == ["window_2"]
    assert result["models"]["manual"]["params"] == {"depth": 2}
    assert result["models"]["all_auto"]["params"] == {"depth": 3}
    assert result["models"]["selected_auto"]["params"] == {"depth": 2}
    assert result["models"]["selected_auto"]["iterations"] == 3
    selector = [t for t in result["metadata"]["trials"] if t["phase"] == "select"]
    assert all(t["params"] == {"depth": 2} and t["iterations"] == 3 for t in selector)
    assert result["metadata"]["n_confirmation_rows_used"] == 0
    assert result["metadata"]["dropped_constant_columns"] == ["train_constant"]
    assert all(len(p) == 2 for p in result["select_predictions"].values())
    assert all(Path(p).exists() for s in result["models"].values() for p in s["model_paths"])
    assert len([c for c in calls if c["tune"]]) == 4
    # The all-prefix may reuse a rejected tuning trial if its exact params,
    # effective tree count and seed match the fixed manual backbone.
    all_prefix = next(t for t in selector if t["prefix"] == 3)
    saved = json.loads(Path(all_prefix["checkpoint"]).read_text())
    assert saved["columns"] == ["m", "a", "b", "c"]
    assert saved["params"] == {"depth": 2} and saved["iterations"] == 3
    assert json.loads((tmp_path / "tabular_complete.json").read_text())["models"]["selected_auto"]["n_features"] == 2


def test_unique_random_controls_include_collisions_without_reroll_and_reuse_models(tmp_path, monkeypatch):
    x, q, groups, config = fixture_data()
    calls = install_mock_learner(monkeypatch)
    result = tabular.train_tabular(x, q, groups, config, tmp_path)
    random_groups = result["metadata"]["random_groups"]
    assert len(random_groups) == 3
    assert {tuple(g) for g in random_groups} == {("window_2",), ("window_3",), ("decay_3",)}
    match = result["metadata"]["random_control_matches_selector"]
    assert len(match) == 1
    collided = result["models"][f"random_{match[0]}"]
    assert collided["columns"] == result["models"]["selected_auto"]["columns"]
    assert all(r["reused_fit"] for r in collided["seed_training"])
    selected_calls = [c for c in calls if c["columns"] == ["m", "a"] and not c["tune"]]
    assert {c["seed"] for c in selected_calls} == {17, 29, 43}
    assert len(selected_calls) == 3


def test_random_subset_edge_space_is_not_fabricated():
    groups = ["g3", "g1", "g2"]
    assert tabular.random_group_subsets(groups, 0, 3, 19) == [[]]
    assert tabular.random_group_subsets(groups, 3, 3, 19) == [["g1", "g2", "g3"]]
    assert tabular.random_group_subsets(groups, 2, 3, 19) == [["g1", "g2"], ["g1", "g3"], ["g2", "g3"]]
    assert tabular.random_group_subsets(groups, 1, 2, 19) == tabular.random_group_subsets(list(reversed(groups)), 1, 2, 19)


def test_repeating_completed_training_cannot_overwrite_model_selection(tmp_path, monkeypatch):
    x, q, groups, config = fixture_data()
    install_mock_learner(monkeypatch)
    tabular.train_tabular(x, q, groups, config, tmp_path)
    with pytest.raises(FileExistsError, match="already completed"):
        tabular.train_tabular(x, q, groups, config, tmp_path)


def test_unlabelled_development_is_rejected_but_test_labels_never_inspected():
    x, q, _, _ = fixture_data()
    q.loc[q.split.eq("test"), "label"] = [1234, -500]
    _, development = tabular._development_data(x, q)
    assert len(development) == 8 and set(development.label.unique()) == {0, 1}
    q.loc[0, "label"] = np.nan
    with pytest.raises(ValueError, match="Development labels"):
        tabular._development_data(x, q)


def test_real_cpu_catboost_checkpoints_reproduce_development_probabilities(tmp_path):
    rng = np.random.default_rng(51)
    signal = rng.normal(size=96)
    x = pd.DataFrame({"cat_uid": np.tile(["a", "b", "c", "d"], 24), "m": signal,
                      "a": signal * signal, "b": np.sin(signal)})
    q = pd.DataFrame({"split": ["train"] * 48 + ["tune"] * 20 + ["select"] * 20 + ["test"] * 8,
                      "label": (signal > 0).astype(float)})
    q.loc[q.split.eq("test"), "label"] = np.nan
    groups = {"manual": ["cat_uid", "m"], "static": ["cat_uid"],
              "window_2": ["a"], "window_3": ["b"]}
    config = {"iterations": 5, "threads": 1, "early_stopping": 2,
              "hyper_grid": [{"depth": 2, "learning_rate": .1}], "model_seeds": [17],
              "selection_prefixes": [1, 2], "selection_feature_penalty": 1e-6,
              "random_controls": 1, "random_seed": 8821}
    result = tabular.train_tabular(x, q, groups, config, tmp_path)
    select = q.split.eq("select")
    for family, specification in result["models"].items():
        model = tabular._load_model(Path(specification["model_paths"][0]))
        actual = model.predict_proba(x.loc[select, specification["columns"]], thread_count=1)[:, 1]
        np.testing.assert_allclose(actual, result["select_predictions"][family], rtol=0, atol=1e-12)
        assert model.tree_count_ == specification["iterations"]
    assert result["metadata"]["ctr_counter_policy"].startswith("SkipTest")
