#!/usr/bin/env python3
"""Post-confirmation diagnosis of candidate-relevant history compression.

For each locked test row, compare full past LISTEN presence from the existing
numeric aggregate features with matching LISTEN events retained in its shared
bounded neural context. Bins use identities/history only, not current labels.
Frozen outcomes and probabilities are joined only after completion/hash checks.

This is an exploratory association analysis: no training, new model variant,
causal ablation, ranking on a truncated candidate set, or model selection.

    PYTHONPATH=src python scripts/diagnose_v2_memory.py --config configs/v2.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

KEYS = ["uid", "item_id", "day", "group_id"]
ENTITIES = ("item", "artist", "album")
FULL_COLUMNS = ["user_item__count__all", "user_artist__count__all", "user_album__count__all"]
BIN_NAMES = ("no_full_prior", "full_prior_retained", "full_prior_missing_context")


def retained_listen_presence(candidate_ids, group_codes, history_ids, history_mask, history_type):
    """Return (query, entity) flags using group-local np.isin, never Q×L.

    Only known raw IDs (>=0), true history masks and event_type==1 (listen)
    count. A retained like/dislike or matching PAD/missing token is insufficient.
    """
    candidates = np.asarray(candidate_ids)
    codes = np.asarray(group_codes, dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or len(codes) != len(candidates):
        raise ValueError("Candidates must be (Q,3), with one history group code per query")
    if not len(codes):
        return np.zeros((0, 3), dtype=bool)
    if codes.min() < 0 or codes.max() >= len(history_ids):
        raise ValueError("Query refers to a missing context group")
    if history_ids.shape[:2] != history_mask.shape or history_mask.shape != history_type.shape:
        raise ValueError("History ID/mask/type dimensions differ")
    if history_ids.ndim != 3 or history_ids.shape[2] != 3:
        raise ValueError("History IDs must be (G,L,3)")
    result = np.zeros((len(codes), 3), dtype=bool)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1, len(order)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        rows = order[start:stop]
        group = int(sorted_codes[start])
        valid = np.asarray(history_mask[group], bool) & (np.asarray(history_type[group]) == 1)
        if not valid.any():
            continue
        ids = np.asarray(history_ids[group])[valid]
        for entity in range(3):
            known = ids[:, entity]
            known = known[known >= 0]
            values = candidates[rows, entity]
            result[rows, entity] = (values >= 0) & np.isin(values, known)
    return result


def prior_bins(full_presence, retained_presence):
    """Assign mutually exclusive identity-based bins, rejecting contradictions."""
    full, retained = np.asarray(full_presence, bool), np.asarray(retained_presence, bool)
    if full.shape != retained.shape or full.ndim != 2 or full.shape[1] != 3:
        raise ValueError("Full/retained flags must both have shape (Q,3)")
    if np.any(retained & ~full):
        raise ValueError("A retained matching listen exists without full-history listen presence")
    return np.where(~full, 0, np.where(retained, 1, 2)).astype(np.uint8)


def classification_metrics(y, scores):
    """Classification only; never recompute ranking on diagnostic subgroups."""
    from sklearn.metrics import roc_auc_score

    targets, probabilities = np.asarray(y, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if len(targets) != len(probabilities) or targets.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Targets/probabilities must be aligned one-dimensional arrays")
    if not len(targets):
        return {"logloss": None, "auc": None}
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("Invalid binary targets or non-finite probabilities")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities outside [0,1]")
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "logloss": float(np.mean(-targets * np.log(p) - (1 - targets) * np.log1p(-p))),
        "auc": float(roc_auc_score(targets, probabilities)) if len(np.unique(targets)) == 2 else None,
    }


def quantiles(values):
    values = np.asarray(values)
    return {str(p): float(np.quantile(values, p)) for p in [0, .1, .5, .9, 1]} if len(values) else None


def diagnose(config, output):
    import pandas as pd

    from diagnose_v2_seeds import sha256, validate_completed_locks, write_json
    from recsys_history_v2.run import load_prepared

    started = time.perf_counter()
    lock, prediction_lock, frozen_results, preparation, selection_sha = validate_completed_locks(config)
    if not {"selected_auto", "history_global"}.issubset(lock["models"]):
        raise ValueError("Both predeclared primary comparison families must exist")
    root, out = Path(config["processed_dir"]), Path(config["output_dir"])
    # Verify all source/data hashes and the still-blinded query table before
    # deriving any subgroup or accessing current confirmation outcomes.
    _, queries, _, _ = load_prepared(config, features=False)
    test_rows = np.flatnonzero(queries.split.eq("test").to_numpy())
    if not len(test_rows):
        raise ValueError("No confirmation rows")
    test = queries.iloc[test_rows].reset_index(drop=True)
    identity = test[KEYS]
    model_columns = ["selected_auto", "history_global"]
    frozen = pd.read_parquet(out / "test_predictions.parquet", columns=KEYS + model_columns)
    blind = pd.read_parquet(out / "blind_predictions.parquet", columns=KEYS + model_columns)
    if not identity.equals(frozen[KEYS]) or not identity.equals(blind[KEYS]):
        raise ValueError("Locked predictions do not align with the confirmation input rows")
    for name in model_columns:
        if not np.array_equal(frozen[name].to_numpy(), blind[name].to_numpy()):
            raise ValueError(f"Released and pre-label locked probabilities differ: {name}")
    numeric = pd.read_parquet(root / "features.parquet", columns=FULL_COLUMNS)
    if len(numeric) != len(queries):
        raise ValueError("Numeric feature and query row counts differ")
    counts = numeric.iloc[test_rows].to_numpy(np.float32)
    del numeric
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Full-history log-count features must be finite and nonnegative")
    full_presence = counts > 0
    del counts
    candidates = test[["item_id", "artist_id", "album_id"]].to_numpy(np.int64)
    if np.any((candidates < 0) & full_presence):
        raise ValueError("Missing candidate metadata cannot have known full-history matching-listen presence")
    context = root / "contexts"
    arrays = {name: np.load(context / (name + ".npy"), mmap_mode="r") for name in (
        "query_group", "history_ids", "history_mask", "history_type", "group_uid", "group_day",
        "group_available_count", "group_retained_count",
    )}
    if len(arrays["query_group"]) != len(queries):
        raise ValueError("Context query-group map and query row counts differ")
    group_codes = np.asarray(arrays["query_group"][test_rows], dtype=np.int64)
    if not np.array_equal(arrays["group_uid"][group_codes], test.uid.to_numpy()):
        raise ValueError("A confirmation history group belongs to a different user")
    if not np.array_equal(arrays["group_day"][group_codes], test.day.to_numpy()):
        raise ValueError("A confirmation history group belongs to a different day")
    retained = retained_listen_presence(
        candidates, group_codes, arrays["history_ids"], arrays["history_mask"], arrays["history_type"]
    )
    bins = prior_bins(full_presence, retained)
    unique_groups = np.unique(group_codes)
    available_counts = np.asarray(arrays["group_available_count"][unique_groups], dtype=np.int64)
    retained_counts = np.asarray(arrays["group_retained_count"][unique_groups], dtype=np.int64)
    if np.any(retained_counts > available_counts):
        raise ValueError("Stored context count exceeds the complete available event history")
    actual_retained = np.array([np.count_nonzero(arrays["history_mask"][group]) for group in unique_groups])
    if not np.array_equal(actual_retained, retained_counts):
        raise ValueError("Stored retained counts do not match raw context masks")
    # Bins have now been fixed using past LISTEN/identity information only.
    # Semantic target reads happen after all completion and hash verification.
    released = pd.read_parquet(out / "test_predictions.parquet", columns=KEYS + ["label"])
    sealed = pd.read_parquet(root / "test_labels.parquet", columns=KEYS + ["label"])
    if not identity.equals(released[KEYS]) or not identity.equals(sealed[KEYS]):
        raise ValueError("Released/sealed target identities differ from the locked inputs")
    y = released.label.to_numpy()
    if not np.array_equal(y, sealed.label.to_numpy()) or not np.isin(y, [0, 1]).all():
        raise ValueError("Released/sealed targets differ or are not binary")
    uid = test.uid.to_numpy()
    predictions = {name: frozen[name].to_numpy(np.float64) for name in model_columns}
    overall = {name: classification_metrics(y, p) for name, p in predictions.items()}
    for name in model_columns:
        for metric, result_key in (("logloss", "logloss"), ("auc", "roc_auc")):
            expected = frozen_results["models"][name][result_key]
            actual = overall[name][metric]
            if (expected is None) != (actual is None) or (
                actual is not None and not np.isclose(actual, expected, rtol=0, atol=1e-12)
            ):
                raise ValueError(f"Frozen overall classification metric differs: {name}/{metric}")
    rows, entities = [], {}
    for dimension, entity in enumerate(ENTITIES):
        full_n = int(full_presence[:, dimension].sum())
        kept_n = int(retained[:, dimension].sum())
        missing_n = int(np.count_nonzero(bins[:, dimension] == 2))
        no_full_n = int(np.count_nonzero(bins[:, dimension] == 0))
        assert kept_n + missing_n == full_n and full_n + no_full_n == len(test)
        entities[entity] = {
            "full_prior_rows": full_n, "retained_prior_rows": kept_n,
            "missing_context_prior_rows": missing_n, "no_full_prior_rows": no_full_n,
            "retention_fraction_given_full_prior": kept_n / full_n if full_n else None,
            "metadata_missing_rows": int((candidates[:, dimension] < 0).sum()),
            "retained_without_full_prior_rows": 0,
        }
        for code, bin_name in enumerate(BIN_NAMES):
            mask = bins[:, dimension] == code
            scores = {name: classification_metrics(y[mask], p[mask]) for name, p in predictions.items()}
            auto_ll, history_ll = scores["selected_auto"]["logloss"], scores["history_global"]["logloss"]
            rows.append({
                "entity": entity, "bin": bin_name, "rows": int(mask.sum()),
                "users": int(len(np.unique(uid[mask]))),
                "positive_rate": float(y[mask].mean()) if mask.any() else None,
                "selected_auto_logloss": auto_ll, "history_global_logloss": history_ll,
                "delta_logloss": history_ll - auto_ll if auto_ll is not None else None,
                "selected_auto_auc": scores["selected_auto"]["auc"],
                "history_global_auc": scores["history_global"]["auc"],
            })
    csv_path = output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = csv_path.with_name(csv_path.name + ".partial")
    pd.DataFrame(rows).to_csv(temporary_csv, index=False)
    temporary_csv.replace(csv_path)
    positive_available = available_counts > 0
    fractions = retained_counts[positive_available] / available_counts[positive_available]
    diagnostic = {
        "status": "completed", "diagnostic_only": True,
        "selection_sha256": selection_sha,
        "prediction_lock_sha256": sha256(out / "prediction_lock.json"),
        "predictions_sha256": sha256(out / "test_predictions.parquet"),
        "test_results_sha256": sha256(out / "test_results.json"),
        "preparation_sha256": sha256(root / "preparation_lock.json"),
        "script_sha256": sha256(Path(__file__)),
        "lock_helper_sha256": sha256(REPOSITORY / "scripts" / "diagnose_v2_seeds.py"),
        "csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        "fit_calls": 0, "model_selection_calls": 0, "weight_changes": 0,
        "neural_inference_calls": 0, "classification_only": True,
        "is_pilot": bool(config.get("is_pilot", False)),
        "test_rows": len(test), "test_users": int(len(np.unique(uid))),
        "entities": entities, "rows": rows, "overall_classification_metrics": overall,
        "history_counts": {
            "test_groups": len(unique_groups), "empty_context_groups": int((retained_counts == 0).sum()),
            "available_events_quantiles": quantiles(available_counts),
            "retained_events_quantiles": quantiles(retained_counts),
            "retained_fraction_nonempty_quantiles": quantiles(fractions),
            "count_scope": "All event types; matching entity presence above is LISTEN-only",
        },
        "definition": {
            "full_prior": "corresponding full-past user_item/user_artist/user_album log1p LISTEN-count feature >0",
            "full_count_columns": dict(zip(ENTITIES, FULL_COLUMNS)),
            "retained_prior": "candidate raw ID>=0 matches context raw ID>=0 with history_mask=True and history_type==1",
            "retention_denominator": "test query rows with full-past matching LISTEN presence, separately per entity; not unique events or all test rows",
            "metadata_missing": "negative candidate metadata ID; a subset of no_full_prior, never a known match",
            "delta_logloss": "history_global minus selected_auto; negative favours history",
            "bins_fixed_without_current_outcomes": True,
            "ranking": "No ranking metric is recomputed on subgroup-filtered candidates",
            "interpretation": "Exploratory post-hoc association, not causal evidence that bounded memory caused a quality difference; no selection, new variants, or multiplicity-adjusted inference",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output, diagnostic)
    print(f"Post-confirmation memory diagnostic: {output}; CSV: {csv_path}", flush=True)
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v2.yaml"))
    parser.add_argument("--output", type=Path, help="New JSON output; default output_dir/memory_diagnostic.json")
    args = parser.parse_args()
    import yaml
    config_path = args.config if args.config.is_absolute() else REPOSITORY / args.config
    config = yaml.safe_load(config_path.read_text())
    output = args.output or Path(config["output_dir"]) / "memory_diagnostic.json"
    output = output if output.is_absolute() else REPOSITORY / output
    if output.exists() or output.with_suffix(".csv").exists():
        raise SystemExit(f"Diagnostic outputs already exist at {output}; choose a new --output")
    os.chdir(REPOSITORY)
    try:
        diagnose(config, output)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f"Memory diagnostic refused: {error}") from error


if __name__ == "__main__":
    main()
