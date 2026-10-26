"""Development-only CatBoost/AutoFE training for the fresh-user v2 study.

Confirmation rows are removed before checking labels, filtering constants,
constructing pools, fitting, importance ranking, or prediction. All returned
probabilities correspond to the select rows in their original input order.
No operation in this module predicts or evaluates the confirmation targets.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from recsys_research.run import classification_loss


def _dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def random_group_subsets(group_names, k: int, n_controls: int, seed: int) -> list[list[str]]:
    """Uniform unique subsets conditional on K, without score-based rerolls.

    The selected subset is not excluded. When fewer than n_controls subsets
    exist (notably K=0 or K=all), return the entire finite space instead.
    """
    names = tuple(sorted(group_names))
    if len(names) != len(set(names)):
        raise ValueError("Random-control group names must be unique")
    if not 0 <= int(k) <= len(names) or int(n_controls) < 0:
        raise ValueError("Invalid random-control subset size or count")
    combinations = list(itertools.combinations(names, int(k)))
    count = min(int(n_controls), len(combinations))
    if count == len(combinations):
        return [list(c) for c in combinations]
    generator = np.random.default_rng(int(seed))
    indices = generator.choice(len(combinations), size=count, replace=False)
    return [list(combinations[int(i)]) for i in indices]


def _pool(x, q, mask, columns):
    return Pool(
        x.loc[mask, columns], q.loc[mask, "label"],
        cat_features=[c for c in columns if c.startswith("cat_")],
        feature_names=columns,
    )


def _fit_model(x, q, columns, params, config, seed, iterations=None, tune=True):
    model = CatBoostClassifier(
        loss_function="Logloss", eval_metric="Logloss", random_seed=int(seed),
        iterations=int(config["iterations"] if iterations is None else iterations),
        thread_count=int(config.get("threads", 16)), allow_writing_files=False,
        verbose=False, has_time=True, max_ctr_complexity=1, task_type="CPU",
        counter_calc_method="SkipTest", used_ram_limit=config.get("tabular_used_ram_limit", "8gb"),
        **params,
    )
    train_pool = _pool(x, q, q.split.eq("train"), columns)
    kwargs = {}
    if tune:
        kwargs = {
            "eval_set": _pool(x, q, q.split.eq("tune"), columns),
            "early_stopping_rounds": int(config["early_stopping"]),
            "use_best_model": True,
        }
    started = time.perf_counter()
    model.fit(train_pool, **kwargs)
    return model, time.perf_counter() - started


def _load_model(path: Path):
    return CatBoostClassifier().load_model(str(path))


def _predict_model(model, x, columns, mask, config):
    probabilities = np.asarray(
        model.predict_proba(x.loc[mask, columns], thread_count=int(config.get("threads", 16)))[:, 1],
        dtype=np.float64,
    )
    if len(probabilities) != int(np.count_nonzero(mask)):
        raise ValueError("Prediction row count differs from development query rows")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("CatBoost returned invalid probabilities")
    return probabilities


def _model_key(columns, params, iterations, seed):
    return (tuple(columns), json.dumps(params, sort_keys=True), int(iterations), int(seed))


def _development_data(x, q):
    if len(x) != len(q):
        raise ValueError("Feature and query matrices must have the same number of rows")
    if not {"split", "label"}.issubset(q.columns):
        raise ValueError("Query data requires split and label fields")
    if q.split.isna().any():
        raise ValueError("Query split must not be missing")
    if not set(q.split.dropna().astype(str).unique()).issubset({"train", "tune", "select", "test"}):
        raise ValueError("Unexpected query split")
    # Positional filtering also works when input dataframes have different indexes.
    positions = np.flatnonzero(q.split.isin(["train", "tune", "select"]).to_numpy())
    development_x = x.iloc[positions].reset_index(drop=True)
    development_q = q.iloc[positions].reset_index(drop=True).copy()
    for split in ("train", "tune", "select"):
        if not development_q.split.eq(split).any():
            raise ValueError(f"No {split} queries available for tabular training")
    labels = development_q.label.to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Development labels must be finite binary outcomes")
    development_q["label"] = labels.astype(np.uint8)
    return development_x, development_q


def train_tabular(x: pd.DataFrame, q: pd.DataFrame, groups: dict, config: dict, output: Path) -> dict:
    """Fit manual/all-auto, matched-backbone selection and random controls.

    ``models`` is JSON-serializable. ``select_predictions`` contains numpy
    arrays for parent-level blend selection, never confirmation predictions.
    Random families can be fewer than requested when the finite subset space
    is smaller; metadata reports the actual and requested numbers explicitly.
    """
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    complete_path = output / "tabular_complete.json"
    if complete_path.exists():
        raise FileExistsError(f"Tabular training already completed at {complete_path}")
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    started_path = output / "tabular_started.json"
    if started_path.exists() and json.loads(started_path.read_text())["config_sha256"] != config_sha:
        raise ValueError("Tabular output contains a different configuration; use a new output directory")
    _dump({"config_sha256": config_sha, "confirmation_rows_used": 0}, started_path)

    x, q = _development_data(x, q)
    train_mask = q.split.eq("train")
    tune_mask = q.split.eq("tune")
    select_mask = q.split.eq("select")
    tune_y = q.loc[tune_mask, "label"].to_numpy()
    select_y = q.loc[select_mask, "label"].to_numpy()
    if not groups.get("manual"):
        raise ValueError("Manual feature group is required")
    unknown = {c for cols in groups.values() for c in cols} - set(x.columns)
    if unknown:
        raise ValueError(f"Feature groups contain unknown columns: {sorted(unknown)}")
    usable = {c for c in x.columns if x.loc[train_mask, c].nunique(dropna=False) > 1}
    manual = [c for c in groups["manual"] if c in usable]
    if not manual or len(manual) != len(set(manual)):
        raise ValueError("Manual features must contain unique non-constant training columns")
    auto_groups = {g: [c for c in cols if c in usable] for g, cols in groups.items()
                   if g not in ("manual", "static")}
    dropped_groups = [g for g, cols in auto_groups.items() if not cols]
    auto_groups = {g: cols for g, cols in auto_groups.items() if cols}
    extras = [c for cols in auto_groups.values() for c in cols]
    if len(extras) != len(set(extras)) or set(extras).intersection(manual):
        raise ValueError("Generated feature groups must be disjoint from each other and manual")
    all_columns = manual + extras

    def columns_for(chosen_groups):
        chosen = {c for group in chosen_groups for c in auto_groups[group]}
        # Shared canonical order removes feature-order/seed effects in coincident
        # selected/random sets; importance order only determines membership.
        return manual + [c for c in extras if c in chosen]

    seeds = [int(s) for s in config.get("model_seeds", [17, 29, 43])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("model_seeds must contain at least one unique seed")
    search_seed = seeds[0]
    grid = config.get("hyper_grid", [])
    if not grid:
        raise ValueError("hyper_grid must contain at least one configuration")
    model_root = output / "tabular"
    search_root = model_root / "search"
    search_root.mkdir(parents=True, exist_ok=True)
    trials = []
    registry = {}
    best = {}

    def record_trial(record):
        trials.append(record)
        _dump(trials, output / "tabular_trials.json")

    def register(columns, params, model, path, seed):
        if int(model.tree_count_) < 1:
            raise ValueError("CatBoost produced an empty model")
        model.save_model(str(path))
        registry[_model_key(columns, params, model.tree_count_, seed)] = str(path)

    for family, columns in (("manual", manual), ("all_auto", all_columns)):
        winner = None
        for trial_number, params in enumerate(grid):
            model, seconds = _fit_model(x, q, columns, params, config, search_seed, tune=True)
            prediction = _predict_model(model, x, columns, tune_mask, config)
            loss = classification_loss(tune_y, prediction)
            checkpoint = search_root / f"tune_{family}_{trial_number}.cbm"
            register(columns, params, model, checkpoint, search_seed)
            record = {
                "phase": "tune", "family": family, "trial": trial_number,
                "seed": search_seed, "params": dict(params), "iterations": int(model.tree_count_),
                "logloss": loss, "n_features": len(columns), "fit_seconds": float(seconds),
                "checkpoint": str(checkpoint),
            }
            record_trial(record)
            if winner is None or loss < winner["logloss"]:
                winner = record
            print(f"tabular tune {family}/{trial_number}: LL={loss:.6f}, trees={model.tree_count_}, {seconds:.1f}s", flush=True)
            del model
        best[family] = {"columns": columns, "params": winner["params"],
                        "iterations": winner["iterations"], "checkpoint": winner["checkpoint"],
                        "tune_logloss": winner["logloss"]}

    importance_model = _load_model(Path(best["all_auto"]["checkpoint"]))
    importance_values = np.asarray(importance_model.get_feature_importance(), dtype=float)
    if len(importance_values) != len(all_columns) or not np.isfinite(importance_values).all():
        raise ValueError("Invalid importance vector from all-auto model")
    importance = dict(zip(all_columns, importance_values.tolist()))
    group_importance = {g: float(sum(importance[c] for c in cols)) for g, cols in auto_groups.items()}
    ranked_groups = sorted(auto_groups, key=lambda g: (-group_importance[g], g))
    del importance_model
    _dump({"feature_importance": importance, "group_importance": group_importance,
           "ranked_groups": ranked_groups}, output / "tabular_importance.json")

    backbone = best["manual"]
    prefixes = sorted({0, *(min(max(int(p), 0), len(ranked_groups))
                            for p in config.get("selection_prefixes", [1, 2, 4, 6, 9]))})
    penalty = float(config.get("selection_feature_penalty", 1e-6))
    if penalty < 0:
        raise ValueError("Feature penalty must be nonnegative")
    selection_candidates = []
    for prefix in prefixes:
        chosen_groups = ranked_groups[:prefix]
        columns = columns_for(chosen_groups)
        checkpoint = search_root / f"select_prefix_{prefix}.cbm"
        key = _model_key(columns, backbone["params"], backbone["iterations"], search_seed)
        reused = key in registry
        if reused:
            source = Path(registry[key])
            if source != checkpoint:
                shutil.copyfile(source, checkpoint)
            model = _load_model(checkpoint)
            seconds = 0.0
        else:
            model, seconds = _fit_model(x, q, columns, backbone["params"], config, search_seed,
                                        iterations=backbone["iterations"], tune=False)
            register(columns, backbone["params"], model, checkpoint, search_seed)
        prediction = _predict_model(model, x, columns, select_mask, config)
        loss = classification_loss(select_y, prediction)
        candidate = {
            "phase": "select", "family": "matched_backbone", "prefix": int(prefix),
            "groups": chosen_groups, "columns": columns, "seed": search_seed,
            "params": dict(backbone["params"]), "iterations": int(backbone["iterations"]),
            "n_features": len(columns), "logloss": loss,
            "objective": loss + penalty * (len(columns) - len(manual)),
            "fit_seconds": float(seconds), "reused_fit": reused, "checkpoint": str(checkpoint),
        }
        selection_candidates.append(candidate)
        record_trial(candidate)
        print(f"tabular select prefix={prefix}: LL={loss:.6f}, features={len(columns)}", flush=True)
        del model
    chosen = min(selection_candidates, key=lambda r: (r["objective"], r["n_features"], r["prefix"]))
    selected_groups = chosen["groups"]
    best["selected_auto"] = {"columns": chosen["columns"], "params": dict(backbone["params"]),
                             "iterations": int(backbone["iterations"]), "checkpoint": chosen["checkpoint"]}
    random_groups = random_group_subsets(
        auto_groups, len(selected_groups), int(config.get("random_controls", 3)),
        int(config.get("random_seed", 8821)),
    )
    for index, subset in enumerate(random_groups):
        best[f"random_{index}"] = {"columns": columns_for(subset), "params": dict(backbone["params"]),
                                   "iterations": int(backbone["iterations"])}
    search_seconds = time.perf_counter() - started
    _dump({"selected_groups": selected_groups, "random_groups": random_groups,
           "selection_candidates": selection_candidates, "manual_backbone": backbone,
           "n_confirmation_rows_used": 0}, output / "tabular_selection.json")

    specifications = {}
    select_predictions = {}
    final_records = []
    for family, definition in best.items():
        paths, probabilities, seed_records = [], [], []
        for seed in seeds:
            checkpoint = model_root / f"{family}_{seed}.cbm"
            key = _model_key(definition["columns"], definition["params"], definition["iterations"], seed)
            reused = key in registry
            if reused:
                source = Path(registry[key])
                if source != checkpoint:
                    shutil.copyfile(source, checkpoint)
                model = _load_model(checkpoint)
                seconds = 0.0
            else:
                model, seconds = _fit_model(
                    x, q, definition["columns"], definition["params"], config, seed,
                    iterations=definition["iterations"], tune=False,
                )
                register(definition["columns"], definition["params"], model, checkpoint, seed)
            prediction = _predict_model(model, x, definition["columns"], select_mask, config)
            probabilities.append(prediction)
            paths.append(str(checkpoint))
            seed_record = {"phase": "final_train", "family": family, "seed": seed,
                           "params": definition["params"], "iterations": definition["iterations"],
                           "n_features": len(definition["columns"]), "fit_seconds": float(seconds),
                           "reused_fit": reused, "select_logloss": classification_loss(select_y, prediction),
                           "checkpoint": str(checkpoint), "model_bytes": checkpoint.stat().st_size}
            seed_records.append(seed_record)
            final_records.append(seed_record)
            record_trial(seed_record)
            print(f"tabular final {family}/{seed}: {'reused' if reused else f'{seconds:.1f}s'}", flush=True)
            del model
        ensemble = np.mean(probabilities, axis=0)
        select_predictions[family] = ensemble
        specifications[family] = {
            "columns": definition["columns"], "params": definition["params"],
            "iterations": int(definition["iterations"]), "model_paths": paths, "seeds": seeds,
            "n_features": len(definition["columns"]), "select_logloss": classification_loss(select_y, ensemble),
            "seed_training": seed_records,
        }

    metadata = {
        "config_sha256": config_sha, "search_seed": search_seed,
        "selected_groups": selected_groups, "random_groups": random_groups,
        "ranked_groups": ranked_groups, "group_importance": group_importance,
        "selection_candidates": selection_candidates, "trials": trials,
        "n_requested_random_controls": int(config.get("random_controls", 3)),
        "n_unique_random_controls": len(random_groups),
        "random_subset_space": math.comb(len(auto_groups), len(selected_groups)),
        "random_control_matches_selector": [i for i, subset in enumerate(random_groups)
                                            if set(subset) == set(selected_groups)],
        "dropped_empty_auto_groups": dropped_groups,
        "dropped_constant_columns": [c for c in x if c not in usable],
        "n_train_rows": int(train_mask.sum()), "n_tune_rows": int(tune_mask.sum()),
        "n_select_rows": int(select_mask.sum()), "n_confirmation_rows_used": 0,
        "selection_backbone": {"params": backbone["params"], "iterations": backbone["iterations"]},
        "search_seconds": float(search_seconds), "total_seconds": float(time.perf_counter() - started),
        "search_fit_seconds": float(sum(r["fit_seconds"] for r in trials if r["phase"] != "final_train")),
        "additional_final_fit_seconds": float(sum(r["fit_seconds"] for r in final_records)),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                      * (1 if platform.system() == "Darwin" else 1024)),
        "train_refit_policy": "train only; no tune/select targets added to fitting",
        "candidate_order_policy": "shared canonical feature order for identical subsets",
        "ctr_counter_policy": "SkipTest: validation feature counts excluded from Counter CTR",
        "cost_note": "Reused checkpoints are charged at their original search/family fit only. Process RSS includes caller allocations; it is not an isolated operation peak.",
    }
    _dump({"models": specifications, "metadata": metadata}, complete_path)
    return {"models": specifications, "select_predictions": select_predictions, "metadata": metadata}
