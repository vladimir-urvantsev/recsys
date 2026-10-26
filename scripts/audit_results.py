#!/usr/bin/env python3
"""Independently audit frozen artifacts; never fit models or select features.

The compact checked-in report is sufficient for hashes, saved-prediction metric
recomputation, and a separate sklearn NDCG check. Local prepared data and trained
models enable additional row-alignment and model-reload checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import ndcg_score

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
from recsys_research.metrics import evaluate  # noqa: E402

TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY / path


def assert_equal(actual, expected, context="value"):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{context}: dictionary keys differ")
        for key in expected:
            assert_equal(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{context}: list lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_equal(left, right, f"{context}[{index}]")
    elif isinstance(expected, numbers.Real) and not isinstance(expected, bool):
        if actual is None or not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=TOLERANCE):
            raise AssertionError(f"{context}: {actual!r} differs from {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{context}: {actual!r} differs from {expected!r}")


class Auditor:
    def __init__(self):
        self.checks = []

    def check(self, name, function):
        try:
            details = function()
            self.checks.append({"name": name, "status": "passed", "details": details})
            print(f"PASS {name}", flush=True)
            return details
        except Exception as error:
            self.checks.append({"name": name, "status": "failed", "error": f"{type(error).__name__}: {error}"})
            print(f"FAIL {name}: {error}", flush=True)
            return None

    def skip(self, name, reason):
        self.checks.append({"name": name, "status": "skipped", "reason": reason})
        print(f"SKIP {name}: {reason}", flush=True)

    def save(self, path, config):
        failed = [c for c in self.checks if c["status"] == "failed"]
        document = {
            "passed": not failed,
            "fit_calls": 0,
            "feature_selection_calls": 0,
            "absolute_numeric_tolerance": TOLERANCE,
            "audit_script_sha256": sha256(Path(__file__)),
            "config": config,
            "counts": {status: sum(c["status"] == status for c in self.checks)
                       for status in ("passed", "skipped", "failed")},
            "checks": self.checks,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        with path.open("x", encoding="utf-8") as output:
            output.write(payload)
        print(f"Audit written: {path}", flush=True)
        return not failed


def _read_selected_feature_rows(path: Path, positions: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Decode batches but retain only requested rows, rather than the full matrix."""
    pieces = []
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=columns):
        selected = positions[(positions >= offset) & (positions < offset + len(batch))]
        if len(selected):
            pieces.append(pa.Table.from_batches([batch]).take(pa.array(selected - offset)))
        offset += len(batch)
        if offset > positions[-1]:
            break
    if not pieces:
        raise AssertionError("Requested feature rows were not found")
    return pa.concat_tables(pieces).to_pandas()


def audit(config: dict, auditor: Auditor):
    out, processed = absolute(config["output_dir"]), absolute(config["processed_dir"])

    def metadata(name):
        local = processed / name
        return local if local.exists() else out / name

    context = {}

    def load_artifacts():
        context["selection"] = read_json(out / "selection_lock.json")
        context["results"] = read_json(out / "test_results.json")
        context["predictions"] = pd.read_parquet(out / "test_predictions.parquet")
        context["preparation"] = read_json(metadata("preparation_lock.json"))
        context["manifest"] = read_json(metadata("manifest.json"))
        specifications = context["selection"]["models"]
        families = list(specifications)
        duplicates = [[left, right] for index, left in enumerate(families)
                      for right in families[index + 1:]
                      if specifications[left] == specifications[right]]
        return {"prediction_rows": len(context["predictions"]),
                "model_families": list(context["results"]["models"]),
                "identical_model_specification_pairs": duplicates,
                "identical_specification_note": "Coincident controls are reported, not rerolled; they cannot establish independent selector benefit."}

    if auditor.check("required_artifacts_present_and_readable", load_artifacts) is None:
        return
    lock, results = context["selection"], context["results"]
    predictions, preparation = context["predictions"], context["preparation"]

    def frozen_hashes():
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert_equal(lock["config_sha256"], config_hash, "selection.config_sha256")
        assert_equal(preparation["config_sha256"], config_hash, "preparation.config_sha256")
        assert_equal(lock["config"], config, "selection.config")
        assert_equal(results["selection_lock_sha256"], sha256(out / "selection_lock.json"), "results.selection_lock_sha256")
        assert_equal(results["test_prediction_sha256"], sha256(out / "test_predictions.parquet"), "results.test_prediction_sha256")
        assert_equal(lock["preparation_lock_sha256"], sha256(metadata("preparation_lock.json")), "selection.preparation_lock_sha256")
        assert_equal(preparation["manifest_sha256"], sha256(metadata("manifest.json")), "preparation.manifest_sha256")
        assert_equal(preparation["feature_groups_sha256"], sha256(metadata("feature_groups.json")), "preparation.feature_groups_sha256")
        assert_equal(lock["n_test_rows_used"], 0, "selection.n_test_rows_used")
        assert_equal(len(predictions), context["manifest"]["splits"]["test"]["rows"], "test row count")
        assert_equal(preparation["sources"], lock["source_sha256"], "preparation versus selection sources")
        return {"config_and_artifact_chain": "verified", "test_rows_used_for_selection": 0}

    auditor.check("frozen_config_and_artifact_hash_chain", frozen_hashes)

    source_root = REPOSITORY / "src" / "recsys_research"
    if source_root.exists():
        def sources():
            actual = {p.name: sha256(p) for p in sorted(source_root.glob("*.py")) if p.name != "report.py"}
            assert_equal(actual, lock["source_sha256"], "current source hashes")
            return {"source_files": len(actual), "report_module_excluded_by_experiment_design": True}
        auditor.check("current_source_matches_frozen_experiment", sources)
    else:
        auditor.skip("current_source_matches_frozen_experiment", "Source tree is not present in this lightweight artifact bundle")

    for name, key in (
        ("queries.parquet", "queries_sha256"),
        ("features.parquet", "features_sha256"),
        ("sample_events.parquet", "events_sha256"),
    ):
        path = processed / name
        if path.exists():
            def local_hash(path=path, key=key):
                assert_equal(sha256(path), preparation[key], f"local {path.name} hash")
                return {"file": str(path.relative_to(REPOSITORY)) if path.is_relative_to(REPOSITORY) else str(path)}
            auditor.check(f"local_{name}_hash", local_hash)
        else:
            auditor.skip(f"local_{name}_hash", "Large local prepared artifact is absent; frozen compact artifact hashes are still checked")

    queries_path = processed / "queries.parquet"
    queries = None
    if queries_path.exists():
        queries = pd.read_parquet(queries_path)

        def alignment():
            columns = ["uid", "item_id", "day", "group_id", "label"]
            expected = queries.loc[queries["split"].eq("test"), columns].reset_index(drop=True)
            actual = predictions[columns].reset_index(drop=True)
            for col in columns:
                np.testing.assert_array_equal(actual[col].to_numpy(), expected[col].to_numpy(), err_msg=col)
            if actual.duplicated(["uid", "item_id", "day"]).any():
                raise AssertionError("Duplicate first-play test query")
            return {"aligned_rows": len(actual), "columns": columns}
        auditor.check("saved_predictions_align_with_local_test_labels", alignment)
    else:
        auditor.skip("saved_predictions_align_with_local_test_labels", "Local query table is absent in this lightweight clone")

    y = predictions["label"].to_numpy()
    uid, group_id = predictions["uid"].to_numpy(), predictions["group_id"].to_numpy()
    for family, record in results["models"].items():
        def recompute(family=family, record=record):
            scores = predictions[family].to_numpy()
            actual = evaluate(y, scores, uid, group_id, k=record["metrics"]["ranking_k"])
            assert_equal(actual, record["metrics"], f"{family}.metrics")
            return {"metrics_compared": len(actual), "rows": len(y)}
        auditor.check(f"recomputed_saved_prediction_metrics_{family}", recompute)

    def independent_ndcg():
        grouped = predictions.groupby("group_id", sort=True)
        eligible = [key for key, frame in grouped
                    if len(frame) >= 2 and 0 < int(frame.label.sum()) < len(frame)]
        if not eligible:
            raise AssertionError("No mixed-label ranking groups available for independent NDCG check")
        rng = np.random.default_rng(2026)
        selected = rng.choice(np.array(eligible), min(50, len(eligible)), replace=False)
        max_error = 0.0
        families = list(results["models"])
        for key in selected:
            frame = grouped.get_group(key)
            labels = frame.label.to_numpy(dtype=np.float64)
            for family in families:
                scores = frame[family].to_numpy(dtype=np.float64)
                expected = ndcg_score(labels[None, :], scores[None, :], k=10, ignore_ties=False)
                actual = evaluate(labels, scores, frame.uid.to_numpy(), frame.group_id.to_numpy(), k=10)["ndcg_mixed"]
                error = abs(actual - expected)
                max_error = max(max_error, error)
                if error > TOLERANCE:
                    raise AssertionError(f"NDCG differs for {family}, group {key}: {actual} vs {expected}")
        return {"mixed_groups": len(selected), "model_families": len(families),
                "independent_api": "sklearn.metrics.ndcg_score(ignore_ties=False, k=10)",
                "max_absolute_error": max_error, "sampling_seed": 2026}

    auditor.check("independent_sklearn_tie_aware_ndcg", independent_ndcg)

    features_path = processed / "features.parquet"
    if not features_path.exists() or queries is None:
        auditor.skip("trained_model_reload_predictions", "Local features and queries are required; no training is performed to replace missing artifacts")
        return
    positions = np.flatnonzero(queries["split"].eq("test").to_numpy())[:128]
    if not len(positions):
        auditor.check("trained_model_reload_predictions", lambda: (_ for _ in ()).throw(AssertionError("No local test rows")))
        return
    union = sorted({column for spec in lock["models"].values() for column in spec["columns"]})
    selected_features = _read_selected_feature_rows(features_path, positions, union)
    from catboost import CatBoostClassifier
    for family, spec in lock["models"].items():
        seeds = [record["seed"] for record in results["models"][family]["seeds"]]
        paths = [REPOSITORY / "artifacts" / Path(config["output_dir"]).name / f"{family}_{seed}.cbm"
                 for seed in seeds]
        missing = [str(p) for p in paths if not p.exists()]
        name = f"trained_model_reload_predictions_{family}"
        if missing:
            auditor.skip(name, f"Trained model artifacts are absent: {', '.join(missing)}")
            continue

        def reloaded(family=family, spec=spec, paths=paths):
            values = []
            for path in paths:
                model = CatBoostClassifier().load_model(str(path))
                values.append(model.predict_proba(selected_features[spec["columns"]])[:, 1])
            actual = np.mean(values, axis=0)
            expected = predictions[family].to_numpy()[:len(positions)]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=TOLERANCE)
            return {"reloaded_models": len(paths), "test_rows": len(positions),
                    "max_absolute_error": float(np.max(np.abs(actual - expected)))}

        auditor.check(name, reloaded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--run-name", help="Audit the same named run used by recsys_research.run")
    parser.add_argument("--output", type=Path, help="Write a new audit JSON, preserving any existing audit")
    args = parser.parse_args()
    config = yaml.safe_load(absolute(args.config).read_text())
    if args.run_name:
        if not args.run_name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("run-name must contain only letters, digits, underscores and hyphens")
        config["processed_dir"] = str(Path(config["processed_dir"]).parent / args.run_name)
        config["output_dir"] = str(Path(config["output_dir"]).parent / args.run_name)
    destination = absolute(args.output) if args.output else absolute(config["output_dir"]) / "audit.json"
    if destination.exists():
        parser.error(f"Audit already exists at {destination}; choose a new --output path")
    auditor = Auditor()
    audit(config, auditor)
    if not auditor.save(destination, config):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
