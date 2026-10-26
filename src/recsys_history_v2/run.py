"""Registered fresh-cohort comparison, with predictions locked before labels."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from catboost import CatBoostClassifier

from recsys_research.data import sha256_file
from recsys_research.features import build_features
from recsys_research.metrics import evaluate, paired_user_bootstrap
from recsys_research.run import classification_loss
from .contexts import prepare_contexts
from .data import choose_cohorts, prepare_data
from .trainer import (GroupBatches, fit_history, load_history, predict_history,
                      prepare_global_features, resolve_device)
from .statistics import temporal_sensitivity


def dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): sha256_file(p) for directory in ("recsys_history_v2", "recsys_research")
            for p in sorted((root / directory).glob("*.py")) if p.name != "report.py"}


def preflight(config):
    return {"platform": platform.platform(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "is_pilot": config.get("is_pilot", False), "n_users": config["n_users"]}


def register(config):
    out = Path(config["output_dir"])
    target = out / "registration.json"
    if target.exists():
        record = json.loads(target.read_text())
        if record["config_sha256"] != fingerprint(config):
            raise ValueError("Registration is immutable; use a new study, not the same holdout")
        return record
    all_uids = pq.read_table(Path(config["raw_dir"]) / "multi_event.parquet", columns=["uid"])["uid"].to_numpy()
    old = json.loads(Path(config["exclude_manifest"]).read_text())
    cohorts = choose_cohorts(all_uids, old["sampled_uids"], config)
    record = {"config": config, "config_sha256": fingerprint(config),
              "cohorts": {k: v.tolist() for k, v in cohorts.items()},
              "excluded_manifest_sha256": sha256_file(config["exclude_manifest"]),
              "primary_comparison": "history_global minus selected_auto, row-weighted LogLoss",
              "primary_interval": "paired user-cluster bootstrap conditional on the observed month",
              "primary_direction": "negative is better for history",
              "freshness": "all v1 users excluded; confirmation future labels never selection targets; warm users",
              "technical_pilot_only": config.get("is_pilot", False)}
    dump(record, target)
    return record


def prepare(config):
    out = Path(config["output_dir"])
    for filename in ("preparation_lock.json", "selection_lock.json", "prediction_lock.json", "test_results.json"):
        if (out / filename).exists():
            raise FileExistsError(f"Published run is locked at {out / filename}; use new output and processed directories")
    registered = register(config)
    if config["n_users"] > 1000 and platform.system() == "Darwin":
        raise RuntimeError("Full study requires the remote RAM-rich host; use configs/v2_smoke.yaml locally")
    root = Path(config["processed_dir"])
    if (root / "preparation_lock.json").exists():
        raise FileExistsError("Preparation already locked; no silent reuse across source versions")
    started = time.perf_counter()
    events, queries, manifest = prepare_data(config, root)
    for name in ("sampled_uids", "development_uids", "confirmation_uids"):
        if manifest[name] != registered["cohorts"][name]:
            raise ValueError(f"Cohort differs from registration: {name}")
    x, groups, timing = build_features(events, queries, config)
    x.to_parquet(root / "features.parquet", index=False)
    dump(groups, root / "feature_groups.json")
    dump(timing, root / "feature_timing.json")
    prepare_global_features(x, queries, groups, root)
    del x
    gc.collect()
    context = prepare_contexts(events, queries, config, root / "contexts")
    del events
    gc.collect()
    paths = ("sample_events.parquet", "queries.parquet", "test_labels.parquet", "features.parquet",
             "feature_groups.json", "manifest.json", "global_features.npy", "global_normalizer.json")
    artifacts = {n: sha256_file(root / n) for n in paths}
    artifacts.update({"contexts/" + p.name: sha256_file(p) for p in (root / "contexts").glob("*.npy")})
    lock = {"config_sha256": fingerprint(config), "sources": source_hashes(), "artifacts": artifacts,
            "registration_sha256": sha256_file(Path(config["output_dir"]) / "registration.json"),
            "seconds": time.perf_counter() - started, "context_summary": context}
    dump(lock, root / "preparation_lock.json")
    for name in ("manifest.json", "feature_groups.json", "feature_timing.json", "global_normalizer.json", "preparation_lock.json"):
        (out / name).write_bytes((root / name).read_bytes())


def load_prepared(config, features=True):
    root = Path(config["processed_dir"])
    lock = json.loads((root / "preparation_lock.json").read_text())
    if lock["config_sha256"] != fingerprint(config) or lock["sources"] != source_hashes():
        raise ValueError("Prepared config/source has changed")
    if lock["registration_sha256"] != sha256_file(Path(config["output_dir"]) / "registration.json"):
        raise ValueError("Registration changed after preparation")
    for name, digest in lock["artifacts"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Prepared artifact changed: {name}")
    q = pd.read_parquet(root / "queries.parquet")
    if not q.loc[q.split.eq("test"), "label"].isna().all():
        raise ValueError("Confirmation labels are not sealed")
    x = pd.read_parquet(root / "features.parquet") if features else None
    groups = json.loads((root / "feature_groups.json").read_text())
    return x, q, groups, lock


def mean_history(paths, root, q, split, config):
    device = resolve_device(config)
    predictions = []
    for path in paths:
        model, saved = load_history(path, device)
        batches = GroupBatches(root, q, split, config["neural_batch_groups"], saved["global_dim"])
        predictions.append(predict_history(model, batches, config, device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.mean(predictions, axis=0)


def blend(auto, history, weight):
    def logit(p):
        p = np.clip(p, 1e-7, 1-1e-7)
        return np.log(p) - np.log1p(-p)
    values = weight * logit(auto) + (1-weight) * logit(history)
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def checked_probabilities(values, expected_rows, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ValueError(f"Invalid prediction shape before label release: {name}")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"Invalid probabilities before label release: {name}")
    return values


def search(config):
    from .tabular import train_tabular
    out, root = Path(config["output_dir"]), Path(config["processed_dir"])
    if (out / "selection_lock.json").exists():
        raise FileExistsError("Selection is frozen")
    if not config.get("is_pilot") and resolve_device(config).type != "cuda":
        raise RuntimeError("Registered full neural search requires CUDA; local technical pilot is separate")
    x, q, groups, preparation = load_prepared(config)
    tabular = train_tabular(x, q, groups, config, out)
    dump(tabular["metadata"], out / "tabular_search.json")
    del x
    gc.collect()
    neural_root = out / "neural"
    neural_root.mkdir(exist_ok=True)
    trials = []
    seed = config["model_seeds"][0]
    for i, params in enumerate(config["neural_grid"]):
        result = fit_history(root, q, params, config, seed, neural_root / f"trial_{i}.pt")
        trials.append(result)
        dump(trials, out / "neural_trials.json")
    chosen = min(trials, key=lambda r: r["tune_logloss"])
    models = dict(tabular["models"])
    for name, pure, no_encoder, seeds in (
        ("history_global", False, False, config["model_seeds"]),
        ("history_pure", True, False, config["ablation_seeds"]),
        ("history_no_encoder", False, True, config["ablation_seeds"]),
    ):
        paths, fit_logs = [], []
        for model_seed in seeds:
            path = neural_root / f"{name}_{model_seed}.pt"
            if name == "history_global" and model_seed == seed:
                path.write_bytes(Path(chosen["checkpoint"]).read_bytes())
                record = {**chosen, "checkpoint": str(path), "reused_search_winner": True}
            else:
                record = fit_history(root, q, chosen["params"], config, model_seed, path,
                                     epochs=chosen["epochs"], pure=pure, no_encoder=no_encoder)
            paths.append(str(path))
            fit_logs.append(record)
        models[name] = {"kind": "history", "model_paths": paths, "params": chosen["params"],
                        "epochs": chosen["epochs"], "fits": fit_logs,
                        "controlled_ablation": name != "history_global"}
    history_select = mean_history(models["history_global"]["model_paths"], root, q, "select", config)
    auto_select = np.asarray(tabular["select_predictions"]["selected_auto"])
    y_select = q.loc[q.split.eq("select"), "label"].to_numpy()
    blending = [{"auto_weight": w, "logloss": classification_loss(y_select, blend(auto_select, history_select, w))}
                for w in config["blend_weights"]]
    selected_blend = min(blending, key=lambda r: (r["logloss"], r["auto_weight"]))
    for specification in models.values():
        specification["model_sha256"] = {p: sha256_file(p) for p in specification["model_paths"]}
    lock = {"config": config, "config_sha256": fingerprint(config), "sources": source_hashes(),
            "preparation_sha256": sha256_file(root / "preparation_lock.json"),
            "registration_sha256": sha256_file(out / "registration.json"),
            "models": models, "blend": selected_blend, "blend_candidates": blending,
            "neural_winner": chosen, "n_confirmation_targets_used_for_selection": 0,
            "primary_comparison": "history_global minus selected_auto", "primary_metric": "logloss",
            "test_labels_opened": False, "environment": preflight(config)}
    dump(lock, out / "selection_lock.json")
    print("SELECTION LOCKED before confirmation-label release", flush=True)


def evaluate_locked(config):
    out, root = Path(config["output_dir"]), Path(config["processed_dir"])
    if (out / "test_results.json").exists() or (out / "prediction_lock.json").exists():
        raise FileExistsError("Confirmation already released or predictions locked")
    lock = json.loads((out / "selection_lock.json").read_text())
    if lock["config_sha256"] != fingerprint(config) or lock["sources"] != source_hashes():
        raise ValueError("Frozen implementation/config differs")
    if sha256_file(out / "registration.json") != lock["registration_sha256"]:
        raise ValueError("Registration changed after selection")
    if sha256_file(root / "preparation_lock.json") != lock["preparation_sha256"]:
        raise ValueError("Preparation changed after selection")
    x, q, groups, preparation = load_prepared(config)
    test = q.split.eq("test")
    predictions = q.loc[test, ["uid", "item_id", "day", "group_id"]].reset_index(drop=True)
    for name, specification in lock["models"].items():
        paths = specification["model_paths"]
        if not paths or len(set(paths)) != len(paths) or set(paths) != set(specification["model_sha256"]):
            raise ValueError(f"Model paths are not exactly covered by frozen hashes: {name}")
        for path, digest in specification["model_sha256"].items():
            if sha256_file(path) != digest:
                raise ValueError(f"Frozen model changed: {path}")
        if specification.get("kind", "tabular") == "history":
            values = mean_history(specification["model_paths"], root, q, "test", config)
        else:
            values = np.mean([CatBoostClassifier().load_model(p).predict_proba(x.loc[test, specification["columns"]])[:, 1]
                              for p in specification["model_paths"]], axis=0)
        predictions[name] = checked_probabilities(values, len(predictions), name)
    predictions["hybrid"] = checked_probabilities(
        blend(predictions.selected_auto.to_numpy(), predictions.history_global.to_numpy(), lock["blend"]["auto_weight"]),
        len(predictions), "hybrid")
    predictions.to_parquet(out / "blind_predictions.parquet", index=False)
    dump({"selection_sha256": sha256_file(out / "selection_lock.json"),
          "blind_predictions_sha256": sha256_file(out / "blind_predictions.parquet"),
          "labels_opened": False}, out / "prediction_lock.json")
    # First deserialization of confirmation targets in model development/eval.
    labels = pd.read_parquet(root / "test_labels.parquet")
    keys = ["uid", "item_id", "day", "group_id"]
    if not predictions[keys].equals(labels[keys]):
        raise ValueError("Sealed labels and locked predictions do not align")
    predictions["label"] = labels.label.to_numpy()
    y, uid, gid = predictions.label.to_numpy(), predictions.uid.to_numpy(), predictions.group_id.to_numpy()
    results = {name: evaluate(y, predictions[name].to_numpy(), uid, gid)
               for name in predictions if name not in keys + ["label"]}
    primary = paired_user_bootstrap(y, predictions.selected_auto.to_numpy(), predictions.history_global.to_numpy(),
                                    uid, gid, n_boot=config["bootstrap_samples"], seed=20261026)
    secondary = {name: paired_user_bootstrap(y, predictions.manual.to_numpy(), predictions[name].to_numpy(),
                                           uid, gid, n_boot=config["bootstrap_samples"], seed=20261026)
                 for name in ("selected_auto", "all_auto", "hybrid")}
    predictions.to_parquet(out / "test_predictions.parquet", index=False)
    dump({"models": results, "primary_history_vs_autofe": primary,
          "temporal_sensitivity": temporal_sensitivity(y, predictions.selected_auto.to_numpy(),
                predictions.history_global.to_numpy(), predictions.day.to_numpy(), config["bootstrap_samples"]),
          "secondary_vs_manual": secondary, "selection_sha256": sha256_file(out / "selection_lock.json"),
          "prediction_lock_sha256": sha256_file(out / "prediction_lock.json"),
          "predictions_sha256": sha256_file(out / "test_predictions.parquet"),
          "is_pilot": config.get("is_pilot", False)}, out / "test_results.json")
    print("Confirmation evaluated; no selection changes permitted", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--phase", choices=["register", "preflight", "prepare", "search", "evaluate", "report", "all"], default="preflight")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.phase == "preflight":
        print(json.dumps(preflight(config), indent=2))
        return
    if args.phase == "register":
        register(config)
    if args.phase in ("prepare", "all"):
        prepare(config)
    if args.phase in ("search", "all"):
        search(config)
    if args.phase in ("evaluate", "all"):
        evaluate_locked(config)
    if args.phase in ("report", "all"):
        from .report import write_report
        write_report(config)


if __name__ == "__main__":
    main()
