#!/usr/bin/env python3
"""Describe optimization-seed sensitivity after v2 confirmation is complete.

This standalone diagnostic performs inference from already frozen weights only.
It refuses to run before prediction_lock.json AND test_results.json exist and
verifies their hash chain before deserializing confirmation targets. It never
fits, ranks/selects seeds, changes ensemble weights, or rewrites experiment
locks. A single-seed ablation has no estimable seed standard deviation.

    PYTHONPATH=src python scripts/diagnose_v2_seeds.py --config configs/v2.yaml

The output is descriptive evidence about the predeclared runs; it is not a new
criterion for choosing a model after inspecting confirmation results.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
KEYS = ["uid", "item_id", "day", "group_id"]
METRICS = (
    "logloss", "roc_auc", "brier", "average_precision", "ndcg_mixed", "recall_mixed",
    "user_macro_ndcg_mixed", "user_macro_recall_mixed", "ndcg_all_defined", "recall_all_defined",
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v2.yaml"))
    parser.add_argument("--output", type=Path, help="New JSON output; default output_dir/seed_metrics.json")
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected, context):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen hash mismatch: {context}")
    return actual


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize_metrics(seed_records, metric_names=METRICS):
    """Summarize fixed runs, with sample SD only when at least two exist."""
    summary = {}
    for metric in metric_names:
        values = [float(record["metrics"][metric]) for record in seed_records
                  if record["metrics"].get(metric) is not None]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite per-seed metric: {metric}")
        summary[metric] = {
            "n_valid": len(values), "mean": statistics.fmean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
        }
    return summary


def seeds_for_model(specification):
    paths = specification["model_paths"]
    if specification.get("kind", "tabular") == "history":
        seeds = [int(record["seed"]) for record in specification.get("fits", [])]
        checkpoints = [record.get("checkpoint") for record in specification.get("fits", [])]
        if checkpoints != paths:
            raise ValueError("History fit records must align exactly with frozen model paths")
    else:
        seeds = [int(seed) for seed in specification.get("seeds", [])]
        records = specification.get("seed_training")
        if records is not None and (
            [int(record["seed"]) for record in records] != seeds
            or [record["checkpoint"] for record in records] != paths
        ):
            raise ValueError("Tabular seed records must align exactly with frozen model paths")
    if not seeds or len(seeds) != len(paths) or len(seeds) != len(set(seeds)):
        raise ValueError("Model paths require the same number of unique declared seeds")
    return seeds


def validate_completed_locks(config):
    """Validate completion and the full compact chain before target access."""
    from recsys_history_v2.run import fingerprint, source_hashes

    out, root = Path(config["output_dir"]), Path(config["processed_dir"])
    required = ["selection_lock.json", "prediction_lock.json", "test_results.json",
                "registration.json", "blind_predictions.parquet", "test_predictions.parquet"]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise ValueError(f"Confirmation must finish before seed diagnosis; missing {missing}")
    lock = read_json(out / "selection_lock.json")
    prediction_lock = read_json(out / "prediction_lock.json")
    result = read_json(out / "test_results.json")
    preparation = read_json(root / "preparation_lock.json")
    registration = read_json(out / "registration.json")
    config_sha = fingerprint(config)
    if lock["config"] != config or any(record["config_sha256"] != config_sha
                                        for record in (lock, preparation, registration)):
        raise ValueError("Config differs from the completed frozen experiment")
    if lock["sources"] != source_hashes() or preparation["sources"] != lock["sources"]:
        raise ValueError("Current/prepared source differs from the frozen implementation")
    if lock.get("n_confirmation_targets_used_for_selection") != 0 or lock.get("test_labels_opened") is not False:
        raise ValueError("Selection must have been frozen before confirmation-label release")
    if prediction_lock.get("labels_opened") is not False:
        raise ValueError("Prediction lock must certify predictions preceded target release")
    selection_sha = sha256(out / "selection_lock.json")
    require_hash(out / "selection_lock.json", result["selection_sha256"], "result selection lock")
    require_hash(out / "selection_lock.json", prediction_lock["selection_sha256"], "prediction selection lock")
    require_hash(out / "prediction_lock.json", result["prediction_lock_sha256"], "result prediction lock")
    require_hash(out / "blind_predictions.parquet", prediction_lock["blind_predictions_sha256"], "blind predictions")
    require_hash(out / "test_predictions.parquet", result["predictions_sha256"], "released predictions")
    require_hash(root / "preparation_lock.json", lock["preparation_sha256"], "selection preparation lock")
    require_hash(out / "registration.json", lock["registration_sha256"], "selection registration")
    require_hash(out / "registration.json", preparation["registration_sha256"], "preparation registration")
    for family, specification in lock["models"].items():
        if specification.get("kind", "tabular") not in ("history", "tabular"):
            raise ValueError(f"Unknown frozen model kind: {family}")
        paths = specification["model_paths"]
        if not paths or len(paths) != len(set(paths)) or set(paths) != set(specification["model_sha256"]):
            raise ValueError(f"Frozen model paths are not exactly covered by hashes: {family}")
        seeds_for_model(specification)
        for path in paths:
            require_hash(path, specification["model_sha256"][path], f"model {family}/{path}")
    if set(lock["models"]) - set(result["models"]):
        raise ValueError("A frozen family has no completed confirmation metrics")
    return lock, prediction_lock, result, preparation, selection_sha


def paired_seed_description(models):
    """Descriptive common-seed differences, not a bootstrap or seed chooser."""
    if not {"selected_auto", "history_global"}.issubset(models):
        return None
    baseline = {record["seed"]: record for record in models["selected_auto"]["seeds"]}
    history = {record["seed"]: record for record in models["history_global"]["seeds"]}
    common = sorted(set(baseline).intersection(history))
    records = []
    for seed in common:
        metrics = {}
        for metric in METRICS:
            left, right = baseline[seed]["metrics"].get(metric), history[seed]["metrics"].get(metric)
            metrics[metric] = None if left is None or right is None else float(right - left)
        records.append({"seed": seed, "metrics": metrics})
    return {
        "definition": "history_global minus selected_auto for matching declared seed labels",
        "caution": "Optimizers differ across methods; common seed labels do not imply common random draws. Descriptive only; no best-seed selection.",
        "seeds": records, "summary": summarize_metrics(records),
    }


def diagnose(config, output):
    import numpy as np
    import pandas as pd
    import torch
    from catboost import CatBoostClassifier

    from recsys_history_v2.run import checked_probabilities, load_prepared, mean_history
    from recsys_history_v2.trainer import resolve_device
    from recsys_research.metrics import evaluate

    started = time.perf_counter()
    lock, prediction_lock, frozen_results, preparation, selection_sha = validate_completed_locks(config)
    has_history = any(spec.get("kind", "tabular") == "history" for spec in lock["models"].values())
    if has_history and not config.get("is_pilot") and resolve_device(config).type != "cuda":
        raise RuntimeError("Full-study neural inference requires the CUDA host; no prepared data/targets opened")
    # This additionally verifies every large prepared artifact hash exactly once,
    # and confirms labels are still absent from the development query table.
    x, queries, _, _ = load_prepared(config, features=True)
    test = queries.split.eq("test")
    frozen = pd.read_parquet(Path(config["output_dir"]) / "test_predictions.parquet")
    blind = pd.read_parquet(Path(config["output_dir"]) / "blind_predictions.parquet")
    identity = queries.loc[test, KEYS].reset_index(drop=True)
    if not identity.equals(frozen[KEYS]) or not identity.equals(blind[KEYS]):
        raise ValueError("Frozen predictions do not align with prepared confirmation rows")
    # Target deserialization is permitted only after all completion/source/model
    # checks above. This script cannot be used as a pre-release evaluator.
    sealed = pd.read_parquet(Path(config["processed_dir"]) / "test_labels.parquet")
    if not identity.equals(sealed[KEYS]):
        raise ValueError("Confirmation targets do not align with locked predictions")
    y = sealed.label.to_numpy()
    if not np.isin(y, [0, 1]).all() or not np.array_equal(y, frozen.label.to_numpy()):
        raise ValueError("Released and sealed confirmation labels differ or are nonbinary")
    uid, group_id = frozen.uid.to_numpy(), frozen.group_id.to_numpy()
    models = {}
    output_summary = {
        "status": "running", "diagnostic_only": True,
        "selection_sha256": selection_sha,
        "prediction_lock_sha256": sha256(Path(config["output_dir"]) / "prediction_lock.json"),
        "test_results_sha256": sha256(Path(config["output_dir"]) / "test_results.json"),
        "predictions_sha256": sha256(Path(config["output_dir"]) / "test_predictions.parquet"),
        "script_sha256": sha256(Path(__file__)), "is_pilot": bool(config.get("is_pilot", False)),
        "fit_calls": 0, "model_selection_calls": 0, "weight_changes": 0,
        "note": "Per-seed metrics are descriptive after the single completed confirmation evaluation. No seed/model/ensemble weight may be selected using them.",
        "standard_deviation": "Sample SD (ddof=1) across fixed optimization seeds; None when fewer than two valid seed values. Not a user-sampling confidence interval.",
        "ensemble_note": "Mean of per-seed metrics generally differs from metric of mean probabilities; both are reported separately.",
        "models": models,
    }
    for family, specification in lock["models"].items():
        kind = specification.get("kind", "tabular")
        seeds = seeds_for_model(specification)
        records = []
        summed = np.zeros(len(frozen), dtype=np.float64)
        uses_cuda = kind == "history" and resolve_device(config).type == "cuda"
        tolerance = 1e-6 if uses_cuda else 1e-10
        for seed, path in zip(seeds, specification["model_paths"]):
            step_started = time.perf_counter()
            if kind == "history":
                probabilities = mean_history([path], Path(config["processed_dir"]), queries, "test", config)
            else:
                model = CatBoostClassifier().load_model(path)
                probabilities = model.predict_proba(
                    x.loc[test, specification["columns"]],
                    thread_count=int(config.get("threads", 16)),
                )[:, 1]
                del model
            probabilities = checked_probabilities(probabilities, len(frozen), f"{family}/seed{seed}")
            summed += probabilities
            record = {
                "seed": seed, "metrics": evaluate(y, probabilities, uid, group_id),
                "seconds": time.perf_counter() - step_started,
                "model_path": path, "model_sha256": specification["model_sha256"][path],
            }
            records.append(record)
            print(f"diagnostic inference {family}/seed{seed}: {record['seconds']:.1f}s", flush=True)
            del probabilities
            if uses_cuda:
                torch.cuda.empty_cache()
            gc.collect()
        ensemble = summed / len(records)
        frozen_probabilities = frozen[family].to_numpy(np.float64)
        max_error = float(np.max(np.abs(ensemble - frozen_probabilities)))
        if max_error > tolerance:
            raise ValueError(f"Reproduced ensemble differs from frozen {family}: max error {max_error} > {tolerance}")
        models[family] = {
            "kind": kind, "n_seeds": len(records), "seeds": records,
            "controlled_ablation": bool(specification.get("controlled_ablation", False)),
            "summary": summarize_metrics(records),
            "ensemble_check": {"passed": True, "max_absolute_probability_error": max_error,
                               "tolerance": tolerance, "computed_on_cuda": uses_cuda},
            "reproduced_ensemble_metrics": evaluate(y, ensemble, uid, group_id),
            "frozen_ensemble_metrics": frozen_results["models"][family],
            "diagnostic_inference_and_metric_seconds": float(sum(record["seconds"] for record in records)),
        }
        # Saving only complete family records makes long GPU diagnostics
        # reviewable without retaining a large probability matrix per seed.
        output_summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(output, output_summary)
    output_summary.update(
        status="completed", elapsed_seconds=time.perf_counter() - started,
        all_ensemble_checks_passed=True,
        common_seed_history_vs_autofe=paired_seed_description(models),
        derived_ensemble_only_families=sorted(set(frozen_results["models"]) - set(models)),
        derived_family_note="Derived hybrid predictions have no independent frozen seed models; no post-hoc per-seed hybrid definition is invented.",
    )
    write_json(output, output_summary)
    print(f"Post-confirmation seed diagnostic saved: {output}", flush=True)
    return output_summary


def main():
    args = parse_arguments()  # --help performs no model/data imports or reads.
    import yaml
    config_path = args.config if args.config.is_absolute() else REPOSITORY / args.config
    config = yaml.safe_load(config_path.read_text())
    output = args.output or Path(config["output_dir"]) / "seed_metrics.json"
    output = output if output.is_absolute() else REPOSITORY / output
    if output.exists():
        raise SystemExit(f"Diagnostic output already exists: {output}; preserve it and choose a new --output")
    os.chdir(REPOSITORY)
    try:
        diagnose(config, output)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"Seed diagnostic refused: {error}") from error


if __name__ == "__main__":
    main()
