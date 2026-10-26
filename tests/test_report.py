"""Synthetic integration fixture verifies artifact-driven report generation."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from recsys_research.metrics import evaluate, paired_user_bootstrap
from recsys_research.report import _conclusion, write_report


def _write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def report_artifacts(tmp_path):
    out, processed = tmp_path / "report", tmp_path / "processed"
    out.mkdir()
    processed.mkdir()
    config = {
        "output_dir": str(out), "processed_dir": str(processed), "n_users": 2,
        "user_seed": 2026, "day_seed": 1701, "days_per_user": {"train": 6, "tune": 2, "select": 2, "test": 3},
        "warmup_days": 30, "split_days": [210, 240, 270], "embargo_days": 1,
        "auto_windows": [2, 3, 10, 14, 60, 90], "auto_half_lives": [3, 10, 45],
        "iterations": 500, "early_stopping": 50, "model_seeds": [17, 29, 43],
        "hyper_grid": [{"depth": 5, "learning_rate": .06, "l2_leaf_reg": 5}],
        "selection_feature_penalty": .000001, "bootstrap_samples": 30, "threads": 1,
    }
    q = pd.DataFrame({"uid": [1, 1, 2, 2], "item_id": [10, 11, 12, 13], "day": [270]*4, "group_id": [11, 11, 21, 21], "label": [1, 0, 1, 0], "split": ["test"]*4})
    q["label"] = q["label"].astype("uint8")
    p = q.drop(columns="split").copy()
    p["manual"] = [.8, .2, .7, .3]
    p["all_auto"] = [.9, .1, .6, .4]
    p["selected_auto"] = [.85, .15, .65, .35]
    p["random_auto"] = [.6, .4, .5, .5]
    p.to_parquet(out / "test_predictions.parquet", index=False)
    q.to_parquet(processed / "queries.parquet", index=False)
    pd.DataFrame({"user__count__all": np.log1p([0, 9, 100, 1000]), "user_item__count__all": np.log1p([0, 1, 0, 2]), "user_artist__count__all": np.log1p([0, 0, 3, 3]), "item__count__all": np.log1p([0, 1, 3, 3])}).to_parquet(processed / "features.parquet", index=False)
    models = {}
    for name in ["manual", "all_auto", "selected_auto", "random_auto"]:
        metrics = evaluate(p.label, p[name], p.uid, p.group_id)
        models[name] = {"metrics": metrics, "n_features": 2 if name == "manual" else 4, "fit_seconds": .3, "prediction_seconds": .01, "model_bytes": 1024, "iterations": 5, "seeds": [{"seed": seed, "metrics": metrics} for seed in config["model_seeds"]]}
    bootstrap = {name: paired_user_bootstrap(p.label, p.manual, p[name], p.uid, p.group_id, n_boot=30) for name in ["all_auto", "selected_auto", "random_auto"]}
    results = {"models": models, "bootstrap_vs_manual": bootstrap, "selection_lock_sha256": "synthetic-test-lock", "test_prediction_sha256": "synthetic-test-predictions"}
    _write(out / "test_results.json", results)
    model_spec = {"columns": ["manual_1", "manual_2"], "params": {"depth": 5}, "iterations": 5}
    lock = {"models": {"manual": model_spec, "all_auto": {**model_spec, "columns": ["manual_1", "manual_2", "auto_1", "auto_2"]}, "selected_auto": {**model_spec, "columns": ["manual_1", "manual_2", "auto_1", "auto_2"]}}, "selected_groups": ["window_2"], "random_groups": ["window_2"], "ranked_groups": ["window_2"], "selection_candidates": [{"prefix": 0, "columns": ["manual_1", "manual_2"], "groups": [], "logloss": .3, "objective": .3, "fit_seconds": 0}, {"prefix": 1, "columns": ["manual_1", "manual_2", "auto_1", "auto_2"], "groups": ["window_2"], "logloss": .29, "objective": .290002, "fit_seconds": .1}], "search_seconds": .5, "n_test_rows_used": 0, "platform": "synthetic fixture", "python": "3.12"}
    _write(out / "selection_lock.json", lock)
    _write(out / "search_trials.json", [{"phase": "tune"}])
    _write(out / "search_importance.json", {"group_importance": {"window_2": 20}, "feature_importance": {"manual_1": 60, "manual_2": 20, "auto_1": 10, "auto_2": 10}})
    _write(processed / "feature_groups.json", {"manual": ["manual_1", "manual_2"], "static": ["manual_1"], "window_2": ["auto_1", "auto_2"]})
    _write(processed / "feature_timing.json", {"seconds": .2, "n_rows": 4, "n_features": 4, "matrix_bytes": 1024})
    split = {"day_min": 270, "day_max": 270, "users": 2, "groups": 2, "rows": 4, "positive_rate": .5, "group_size_median": 2, "group_size_p90": 2}
    _write(processed / "manifest.json", {"dataset": "synthetic integration fixture", "sources": {"events": {"revision": "fixture"}}, "sampled_users": 2, "source_users": 10, "retained_events": 12, "retained_items": 4, "history_lengths": {"min": 3, "median": 6, "p90": 9, "max": 10}, "event_type_counts": {"listen": 12}, "timestamp_max": 300*86400, "elapsed_seconds": .1, "splits": {"test": split}})
    results["selection_lock_sha256"] = hashlib.sha256((out / "selection_lock.json").read_bytes()).hexdigest()
    results["test_prediction_sha256"] = hashlib.sha256((out / "test_predictions.parquet").read_bytes()).hexdigest()
    _write(out / "test_results.json", results)
    return config, out, lock, results


def test_report_reads_numeric_artifacts_and_draws_figures(report_artifacts):
    config, out, _, results = report_artifacts
    target = write_report(config)
    text = target.read_text()
    delta = results["models"]["selected_auto"]["metrics"]["logloss"] - results["models"]["manual"]["metrics"]["logloss"]
    assert f"{abs(delta):.6f}" in text or f"{delta:.7f}" in text
    assert results["selection_lock_sha256"] in text
    assert "не p-value" in text
    assert "события более ранних test-дней" in text
    for name in ["comparison", "paired_deltas", "validation_search", "seed_variability", "kernel_importance", "calibration"]:
        assert (out / "figures" / f"{name}.png").stat().st_size > 1000
    subgroups = pd.read_csv(out / "exploratory_subgroups.csv")
    assert len(subgroups) > 0
    assert (subgroups.manual_logloss >= 0).all()
    cold_pairs = subgroups.loc[(subgroups.dimension == "user_item") & (subgroups.subgroup == "нет прошлых прослушиваний")].iloc[0]
    assert cold_pairs.manual_logloss == pytest.approx(-np.log([.8, .7]).mean())
    assert len(pd.read_csv(out / "calibration.csv")) > 0


def test_report_rejects_a_lock_that_used_test(report_artifacts):
    config, out, lock, _ = report_artifacts
    lock["n_test_rows_used"] = 1
    _write(out / "selection_lock.json", lock)
    with pytest.raises(ValueError, match="no test rows"):
        write_report(config)


def test_optional_scaling_uses_recorded_measurements_without_extrapolation(report_artifacts):
    config, out, _, _ = report_artifacts
    _write(out / "scaling.json", {"platform": "synthetic scaling fixture", "note": "Test data only", "rows": [
        {"users": 1, "history_events": 6, "query_rows": 2, "feature_seconds": .2, "load_seconds": .1, "peak_rss_bytes": 2**20, "matrix_bytes": 512},
        {"users": 2, "history_events": 12, "query_rows": 4, "feature_seconds": .35, "load_seconds": .15, "peak_rss_bytes": 2*2**20, "matrix_bytes": 1024},
    ]})
    text = write_report(config).read_text()
    assert "| 1 | 6 | 2 | 0.20 | 0.10 | 1.0 |" in text
    assert "Экстраполяция на миллиарды событий не проводится" in text
    assert (out / "figures" / "scaling.png").stat().st_size > 1000


def test_identical_random_control_is_disclosed_without_superiority_claim(report_artifacts):
    config, out, lock, results = report_artifacts
    lock["models"]["random_auto"] = lock["models"]["selected_auto"].copy()
    _write(out / "selection_lock.json", lock)
    predictions = pd.read_parquet(out / "test_predictions.parquet")
    predictions["random_auto"] = predictions["selected_auto"]
    predictions.to_parquet(out / "test_predictions.parquet", index=False)
    results["models"]["random_auto"] = results["models"]["selected_auto"]
    results["bootstrap_vs_manual"]["random_auto"] = results["bootstrap_vs_manual"]["selected_auto"]
    results["selection_lock_sha256"] = hashlib.sha256((out / "selection_lock.json").read_bytes()).hexdigest()
    results["test_prediction_sha256"] = hashlib.sha256((out / "test_predictions.parquet").read_bytes()).hexdigest()
    _write(out / "test_results.json", results)
    text = write_report(config).read_text()
    assert "этот запуск не позволяет показать преимущество отбора над случайным выбором" in text
    assert "максимальное абсолютное отличие сохранённых прогнозов: `0`" in text
    assert "не доказывает, что поиск бесполезен" in text


def test_cluster_interval_wording_does_not_imply_cold_start():
    text = _conclusion(
        {"manual": {"metrics": {"logloss": .5}}, "selected_auto": {"metrics": {"logloss": .499}}},
        {"selected_auto": {"logloss": {"ci95": [-.01, .01]}}}, ["window_14"],
    )
    assert "Надёжное улучшение над manual этим экспериментом не продемонстрировано" in text
    assert "повторной выборке пользователей этой популяции" in text
    assert "новых пользователях" not in text
