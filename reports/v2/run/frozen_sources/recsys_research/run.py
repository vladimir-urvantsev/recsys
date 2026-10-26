"""CLI with separate search lock and final held-out evaluation phases."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool

from .data import prepare_data, sha256_file
from .features import build_features
from .metrics import evaluate, paired_user_bootstrap


def dump(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def source_fingerprint():
    root = Path(__file__).parent
    return {p.name: sha256_file(p) for p in sorted(root.glob("*.py")) if p.name != "report.py"}


def environment():
    return {"platform": platform.platform(), "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "pyarrow", "catboost", "scikit-learn", "scipy")},
            "logical_cpu_count": os.cpu_count()}


def load_prepared(config):
    root = Path(config["processed_dir"])
    return pd.read_parquet(root / "sample_events.parquet"), pd.read_parquet(root / "queries.parquet")


def prepare(config):
    root = Path(config["processed_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "queries.parquet").exists():
        events, queries, _ = prepare_data(config, root)
    else:
        old = json.loads((root / "manifest.json").read_text())["config"]
        data_keys = ("n_users", "user_seed", "day_seed", "warmup_days", "split_days", "days_per_user", "raw_dir")
        if any(old.get(k) != config.get(k) for k in data_keys):
            raise ValueError("Cached sample uses different data settings; use a new processed_dir")
        events, queries = load_prepared(config)
    x, groups, timing = build_features(events, queries, config)
    timing["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    x.to_parquet(root / "features.parquet", index=False)
    dump(groups, root / "feature_groups.json")
    dump(timing, root / "feature_timing.json")
    dump({"config": config, "config_sha256": fingerprint(config),
          "sources": source_fingerprint(), "queries_sha256": sha256_file(root / "queries.parquet"),
          "features_sha256": sha256_file(root / "features.parquet"),
          "manifest_sha256": sha256_file(root / "manifest.json"),
          "events_sha256": sha256_file(root / "sample_events.parquet"),
          "feature_groups_sha256": sha256_file(root / "feature_groups.json")},
         root / "preparation_lock.json")
    print(f"Prepared {x.shape} matrix in {timing['seconds']:.1f}s", flush=True)


def load_matrix(config):
    root = Path(config["processed_dir"])
    lock = json.loads((root / "preparation_lock.json").read_text())
    if lock["config_sha256"] != fingerprint(config):
        raise ValueError("Config differs from prepared features; use a new processed_dir")
    if lock["queries_sha256"] != sha256_file(root / "queries.parquet"):
        raise ValueError("Prepared queries have changed")
    if lock["features_sha256"] != sha256_file(root / "features.parquet"):
        raise ValueError("Prepared features have changed")
    if lock["manifest_sha256"] != sha256_file(root / "manifest.json"):
        raise ValueError("Prepared data manifest has changed")
    if lock["events_sha256"] != sha256_file(root / "sample_events.parquet"):
        raise ValueError("Prepared event history has changed")
    if lock["feature_groups_sha256"] != sha256_file(root / "feature_groups.json"):
        raise ValueError("Prepared feature groups have changed")
    if lock["sources"] != source_fingerprint():
        raise ValueError("Source changed after preparation; rebuild features before searching")
    return (pd.read_parquet(root / "features.parquet"), pd.read_parquet(root / "queries.parquet"),
            json.loads((root / "feature_groups.json").read_text()))


def pool(x, queries, mask, columns):
    cats = [name for name in columns if name.startswith("cat_")]
    return Pool(x.loc[mask, columns], queries.loc[mask, "label"], cat_features=cats,
                feature_names=columns)


def fit_one(x, q, columns, params, config, seed, iterations=None, tune=True):
    model = CatBoostClassifier(
        loss_function="Logloss", eval_metric="Logloss", random_seed=seed,
        iterations=int(iterations or config["iterations"]), thread_count=config["threads"],
        allow_writing_files=False, verbose=False, has_time=True,
        max_ctr_complexity=1, used_ram_limit="2gb", **params)
    train_pool = pool(x, q, q.split.eq("train"), columns)
    kwargs = {}
    if tune:
        kwargs = {"eval_set": pool(x, q, q.split.eq("tune"), columns),
                  "early_stopping_rounds": config["early_stopping"], "use_best_model": True}
    started = time.perf_counter()
    model.fit(train_pool, **kwargs)
    elapsed = time.perf_counter() - started
    return model, elapsed


def predict(model, x, columns, mask):
    started = time.perf_counter()
    p = model.predict_proba(x.loc[mask, columns])[:, 1]
    return p, time.perf_counter() - started


def metrics_for(q, mask, p):
    a = q.loc[mask]
    return evaluate(a.label.to_numpy(), p, a.uid.to_numpy(), a.group_id.to_numpy())


def classification_loss(y, p):
    # Labels are stored as uint8: unary minus before conversion would wrap 1
    # to 255 and silently make the selection objective negative.
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))


def search(config):
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "selection_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"Search is frozen at {lock_path}; use a new output_dir to rerun")
    x, q, groups = load_matrix(config)
    # Test labels are removed from this process before any fitting/selection.
    development = ~q.split.eq("test")
    x, q = x.loc[development].reset_index(drop=True), q.loc[development].reset_index(drop=True)
    train = q.split.eq("train")
    # Feature filtering may inspect training inputs only.
    usable = [c for c in x if x.loc[train, c].nunique() > 1]
    manual = [c for c in groups["manual"] if c in usable]
    auto_groups = {g: [c for c in cols if c in usable] for g, cols in groups.items()
                   if g not in ("manual", "static")}
    all_columns = manual + [c for cols in auto_groups.values() for c in cols]
    seed = config["model_seeds"][0]
    logs = []
    best = {}
    importance = None
    started = time.perf_counter()
    for family, columns in (("manual", manual), ("all_auto", all_columns)):
        candidates = []
        for trial, params in enumerate(config["hyper_grid"]):
            model, elapsed = fit_one(x, q, columns, params, config, seed)
            p, _ = predict(model, x, columns, q.split.eq("tune"))
            loss = classification_loss(q.loc[q.split.eq("tune"), "label"].to_numpy(), p)
            record = {"phase": "tune", "family": family, "trial": trial, "params": params,
                      "iterations": int(model.tree_count_), "logloss": loss,
                      "n_features": len(columns), "fit_seconds": elapsed}
            candidates.append((loss, record, model))
            logs.append(record)
            dump(logs, out / "search_trials.json")
            print(f"tune {family} {trial}: loss={loss:.6f}, trees={model.tree_count_}, {elapsed:.1f}s", flush=True)
        _, record, winner = min(candidates, key=lambda z: z[0])
        best[family] = {"params": record["params"], "iterations": record["iterations"], "columns": columns}
        winner.save_model(str(out / f"search_{family}.cbm"))
        if family == "all_auto":
            importance = dict(zip(columns, winner.get_feature_importance().tolist()))
    group_scores = {g: sum(importance.get(c, 0) for c in cols) for g, cols in auto_groups.items()}
    order = sorted(auto_groups, key=lambda g: (-group_scores[g], g))
    dump({"feature_importance": importance, "group_importance": group_scores, "ranked_groups": order},
         out / "search_importance.json")
    # Generation and group ranking never use the selection slice or test.
    select_mask = q.split.eq("select")
    select_y = q.loc[select_mask, "label"].to_numpy()
    manual_model = CatBoostClassifier().load_model(str(out / "search_manual.cbm"))
    p, _ = predict(manual_model, x, manual, select_mask)
    baseline_loss = classification_loss(select_y, p)
    penalty = config["selection_feature_penalty"]
    candidates = [{"prefix": 0, "groups": [], "columns": manual, "logloss": baseline_loss,
                   "objective": baseline_loss, "params": best["manual"]["params"],
                   "iterations": best["manual"]["iterations"], "fit_seconds": 0.0}]
    for prefix in sorted(set(min(int(n), len(order)) for n in config["selection_prefixes"])):
        selected_groups = order[:prefix]
        columns = manual + [c for g in selected_groups for c in auto_groups[g]]
        if prefix == len(order):
            model = CatBoostClassifier().load_model(str(out / "search_all_auto.cbm"))
            elapsed = 0.0
        else:
            model, elapsed = fit_one(x, q, columns, best["all_auto"]["params"], config, seed,
                                      iterations=best["all_auto"]["iterations"], tune=False)
        p, _ = predict(model, x, columns, select_mask)
        loss = classification_loss(select_y, p)
        record = {"prefix": prefix, "groups": selected_groups, "columns": columns, "logloss": loss,
                  "objective": loss + penalty * (len(columns) - len(manual)),
                  "params": best["all_auto"]["params"], "iterations": int(model.tree_count_),
                  "fit_seconds": elapsed}
        candidates.append(record)
        logs.append({"phase": "select", **record, "n_features": len(columns)})
        dump(logs, out / "search_trials.json")
        print(f"select prefix={prefix} loss={loss:.6f} objective={record['objective']:.6f} features={len(columns)}", flush=True)
    chosen = min(candidates, key=lambda d: (d["objective"], len(d["columns"])))
    best["selected_auto"] = {k: chosen[k] for k in ("params", "iterations", "columns")}
    # A fixed random ordering independent of both validation labels is a control.
    random_order = np.random.default_rng(8821).permutation(order).tolist()
    n_selected = chosen["prefix"]
    # Match number of groups; every kernel group has the same relation/metric size.
    random_cols = manual + [c for g in random_order[:n_selected] for c in auto_groups[g]]
    best["random_auto"] = {**best["all_auto" if n_selected else "manual"], "columns": random_cols}
    best["static"] = {**best["manual"], "columns": [c for c in groups["static"] if c in usable]}
    # Relation ablation is predeclared, keeping user and item marginal statistics.
    best["no_pair_relations"] = {**best["manual"], "columns": [c for c in manual if not c.startswith(("user_item__", "user_artist__", "user_album__"))]}
    dump({"config": config, "config_sha256": fingerprint(config), "source_sha256": source_fingerprint(),
          "preparation_lock_sha256": sha256_file(Path(config["processed_dir"]) / "preparation_lock.json"),
          "models": best, "ranked_groups": order, "selected_groups": chosen["groups"],
          "selection_candidates": candidates, "random_groups": random_order[:n_selected],
          "objective": "selection LogLoss + penalty per additional feature", "feature_penalty": penalty,
          "baseline_selection_logloss": baseline_loss, "search_seconds": time.perf_counter() - started,
          "n_test_rows_used": 0, "train_refit_policy": "train only; tune/select are never refit",
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
          "environment": environment()}, lock_path)
    print(f"SEARCH FROZEN: {chosen['groups']}", flush=True)


def final_evaluate(config):
    out = Path(config["output_dir"])
    result_path = out / "test_results.json"
    if result_path.exists():
        raise FileExistsError("Final test already evaluated; inspect existing results")
    lock_path = out / "selection_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock["config_sha256"] != fingerprint(config):
        raise ValueError("Config changed after selection lock")
    if lock["source_sha256"] != source_fingerprint():
        raise ValueError("Source changed after selection lock")
    if lock["preparation_lock_sha256"] != sha256_file(Path(config["processed_dir"]) / "preparation_lock.json"):
        raise ValueError("Preparation lock changed after model selection")
    x, q, groups = load_matrix(config)
    mask = q.split.eq("test")
    test_q = q.loc[mask].reset_index(drop=True)
    predictions = test_q[["uid", "item_id", "day", "group_id", "label"]].copy()
    results = {}
    # A prevalence-only baseline is trained exclusively on training labels.
    constant = float(q.loc[q.split.eq("train"), "label"].mean())
    predictions["constant"] = constant
    results["constant"] = {"metrics": metrics_for(q, mask, np.full(mask.sum(), constant)), "n_features": 0,
                            "fit_seconds": 0, "prediction_seconds": 0}
    for family, specification in lock["models"].items():
        columns = specification["columns"]
        seed_results, seed_predictions = [], []
        fit_seconds, prediction_seconds, sizes = [], [], []
        for seed in config["model_seeds"]:
            model, elapsed = fit_one(x, q, columns, specification["params"], config, seed,
                                    iterations=specification["iterations"], tune=False)
            p, predict_seconds = predict(model, x, columns, mask)
            model_path = Path("artifacts") / Path(config["output_dir"]).name / f"{family}_{seed}.cbm"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_path))
            sizes.append(model_path.stat().st_size)
            fit_seconds.append(elapsed)
            prediction_seconds.append(predict_seconds)
            seed_predictions.append(p)
            seed_results.append({"seed": seed, "metrics": metrics_for(q, mask, p), "fit_seconds": elapsed})
            print(f"final fit {family} seed={seed}: {elapsed:.1f}s", flush=True)
        averaged = np.mean(seed_predictions, axis=0)
        predictions[family] = averaged
        results[family] = {"metrics": metrics_for(q, mask, averaged), "seeds": seed_results,
                           "n_features": len(columns), "iterations": specification["iterations"],
                           "fit_seconds": float(sum(fit_seconds)), "fit_seconds_per_seed": fit_seconds,
                           "prediction_seconds": float(sum(prediction_seconds)), "model_bytes": sum(sizes)}
        # Save progress but don't expose adaptive intermediate test scores.
    bootstrap = {}
    for family in ("all_auto", "selected_auto", "random_auto", "no_pair_relations"):
        bootstrap[family] = paired_user_bootstrap(
            test_q.label.to_numpy(), predictions.manual.to_numpy(), predictions[family].to_numpy(),
            test_q.uid.to_numpy(), test_q.group_id.to_numpy(),
            n_boot=config["bootstrap_samples"], seed=2026)
    predictions.to_parquet(out / "test_predictions.parquet", index=False)
    dump({"selection_lock_sha256": sha256_file(lock_path), "models": results, "bootstrap_vs_manual": bootstrap,
          "test_prediction_sha256": sha256_file(out / "test_predictions.parquet"),
          "note": "Each model uses train only. Ensemble means three fixed seed probabilities. Test is a predeclared logged-candidate conditional task."}, result_path)
    print("Final held-out evaluation saved.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--phase", choices=["prepare", "search", "evaluate", "report", "all"], default="all")
    parser.add_argument("--run-name", help="New output/cache name for a clean reproduction (e.g. reproduce-01)")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.run_name:
        if not args.run_name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("run-name must contain only letters, digits, underscores and hyphens")
        config["processed_dir"] = str(Path(config["processed_dir"]).parent / args.run_name)
        config["output_dir"] = str(Path(config["output_dir"]).parent / args.run_name)
    if args.phase in ("prepare", "all"):
        prepare(config)
    if args.phase in ("search", "all"):
        search(config)
    if args.phase in ("evaluate", "all"):
        final_evaluate(config)
    if args.phase in ("report", "all"):
        from .report import write_report
        write_report(config)


if __name__ == "__main__":
    main()
