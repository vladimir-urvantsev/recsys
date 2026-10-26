#!/usr/bin/env python3
"""Audit a frozen V2 study without fitting, selecting, or reopening a holdout.

The joined final result must already exist. Hash/metric checks work on a compact
artifact bundle. Local preparation and weights enable additional checks; absent
large files are explicitly skipped, never replaced by a new training run.
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
from recsys_research.metrics import evaluate, paired_user_bootstrap  # noqa: E402

KEYS = ["uid", "item_id", "day", "group_id"]
TOLERANCE = 1e-10
MODEL_PROBABILITY_TOLERANCE = 2e-6
GPU_MODEL_PROBABILITY_TOLERANCE = 1e-6
MAX_RELOAD_ROWS = 128


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def v1_implementation_hashes(directory):
    """Separate recognized macOS metadata from executable V1 source files.

    The full V2 attestation above still checks every registered file, including
    metadata. This narrower check compares the V1 implementation with its
    original Mac-side lock, which did not contain tar-generated AppleDouble.
    A filename alone is never sufficient to exempt a file from comparison.
    """
    implementation, metadata = {}, {}
    for path in sorted(Path(directory).glob("*.py")):
        if path.name == "report.py":
            continue
        digest = sha256(path)
        with path.open("rb") as source:
            header = source.read(8)
        if path.name.startswith("._") and header == bytes.fromhex("0005160700020000"):
            metadata[path.name] = digest
        else:
            implementation[path.name] = digest
    return implementation, metadata


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
            raise AssertionError(f"{context}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{context}: {actual!r} != {expected!r}")


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
        counts = {status: sum(c["status"] == status for c in self.checks)
                  for status in ("passed", "skipped", "failed")}
        document = {"passed": bool(counts["passed"]) and not counts["failed"],
                    "complete": not counts["failed"] and not counts["skipped"],
                    "fit_calls": 0, "refit_calls": 0, "feature_selection_calls": 0,
                    "absolute_numeric_tolerance": TOLERANCE,
                    "cpu_prediction_tolerance": MODEL_PROBABILITY_TOLERANCE,
                    "gpu_prediction_tolerance": GPU_MODEL_PROBABILITY_TOLERANCE,
                    "max_reload_rows": MAX_RELOAD_ROWS,
                    "gpu_replay_batch_policy": "Compute the entire original first test batch with original group count, history length and all candidates; compare at most max_reload_rows genuine rows.",
                    "audit_script_sha256": sha256(Path(__file__)),
                    "config": config, "counts": counts, "checks": self.checks}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with Path(path).open("x", encoding="utf-8") as output:
            output.write(payload)
        print(f"Audit written: {path}", flush=True)
        return document["passed"]


def _feature_rows(path, positions, columns):
    pieces, row_group_offset = [], 0
    parquet = pq.ParquetFile(path)
    for row_group in range(parquet.num_row_groups):
        row_group_end = row_group_offset + parquet.metadata.row_group(row_group).num_rows
        wanted = positions[(positions >= row_group_offset) & (positions < row_group_end)]
        if len(wanted):
            offset = row_group_offset
            # Earlier training row groups need not be decoded to check a small
            # confirmation prefix. Within a relevant group, retain only rows
            # requested by the bounded prediction replay.
            for batch in parquet.iter_batches(batch_size=8192, row_groups=[row_group], columns=columns):
                selected = wanted[(wanted >= offset) & (wanted < offset + len(batch))]
                if len(selected):
                    pieces.append(pa.Table.from_batches([batch]).take(pa.array(selected - offset)))
                offset += len(batch)
                if offset > wanted[-1]:
                    break
        row_group_offset = row_group_end
        if row_group_offset > positions[-1]:
            break
    if not pieces:
        raise AssertionError("Requested feature rows were not found")
    return pa.concat_tables(pieces).to_pandas()


def _first_original_history_batch(model, root, queries, config, global_dim, device):
    """Replay a full original batch, preserving GPU/BF16 operation shapes.

    No loader group, query, token or candidate is truncated. Returned row IDs
    refer to the original sealed query table, not a reduced DataFrame. The
    caller may compare a prefix of these predictions only after computation.
    This helper also accepts a CPU device for synthetic integration checks.
    """
    import torch
    from recsys_history_v2.trainer import GroupBatches, autocast_context
    loader = GroupBatches(root, queries, "test", config["neural_batch_groups"], global_dim)
    try:
        batch = next(loader.batches(shuffle=False))
    except StopIteration as error:
        raise AssertionError("Original test loader has no batch") from error
    row_indices = batch.pop("row_indices").numpy()
    shape = {"groups": int(batch["candidate_ids"].shape[0]),
             "candidate_width": int(batch["candidate_ids"].shape[1]),
             "history_length": int(batch["history_ids"].shape[1]),
             "actual_candidate_rows": int(batch["candidate_mask"].sum().item()),
             "configured_batch_groups": int(config["neural_batch_groups"])}
    batch = {key: value.to(device) for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        with autocast_context(device, config):
            logits = model(batch)
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
    valid = row_indices >= 0
    selected = probabilities[valid].astype(np.float64, copy=False)
    indices = row_indices[valid].astype(np.int64, copy=False)
    if not len(indices) or not np.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any():
        raise AssertionError("Original-batch replay has no valid or finite probability rows")
    if len(indices) != shape["actual_candidate_rows"]:
        raise AssertionError("Original batch row indices and candidate mask disagree")
    return selected, indices, shape


def audit(config, auditor):
    out, root = absolute(config["output_dir"]), absolute(config["processed_dir"])
    context = {}

    def metadata(name):
        return root / name if (root / name).exists() else out / name

    def load_required():
        for name, filename in (("registration", "registration.json"), ("selection", "selection_lock.json"),
                               ("prediction_lock", "prediction_lock.json"), ("results", "test_results.json")):
            context[name] = read_json(out / filename)
        context["preparation"] = read_json(metadata("preparation_lock.json"))
        context["manifest"] = read_json(metadata("manifest.json"))
        context["blind"] = pd.read_parquet(out / "blind_predictions.parquet")
        context["predictions"] = pd.read_parquet(out / "test_predictions.parquet")
        families = list(context["results"]["models"])
        if set(context["predictions"].columns) != set(KEYS + ["label"] + families):
            raise AssertionError("Final prediction columns differ from the evaluated model families")
        if set(context["blind"].columns) != set(KEYS + families):
            raise AssertionError("Blind predictions must contain keys and probabilities only, without labels")
        return {"rows": len(context["predictions"]), "families": families,
                "technical_pilot": bool(config.get("is_pilot", False))}

    if auditor.check("required_final_artifacts", load_required) is None:
        return
    registration, preparation, selection = (context[key] for key in ("registration", "preparation", "selection"))
    prediction_lock, results = context["prediction_lock"], context["results"]
    manifest, blind, predictions = context["manifest"], context["blind"], context["predictions"]
    families = list(results["models"])

    def hash_chain():
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        for name, record in (("registration", registration), ("preparation", preparation), ("selection", selection)):
            assert_equal(record["config_sha256"], config_hash, f"{name}.config_sha256")
        assert_equal(registration["config"], config, "registered config")
        assert_equal(selection["config"], config, "selected config")
        assert_equal(preparation["registration_sha256"], sha256(out / "registration.json"), "preparation registration link")
        assert_equal(selection["registration_sha256"], sha256(out / "registration.json"), "selection registration link")
        assert_equal(selection["preparation_sha256"], sha256(metadata("preparation_lock.json")), "selection preparation link")
        assert_equal(prediction_lock["selection_sha256"], sha256(out / "selection_lock.json"), "blind selection link")
        assert_equal(prediction_lock["blind_predictions_sha256"], sha256(out / "blind_predictions.parquet"), "blind hash")
        assert_equal(results["selection_sha256"], sha256(out / "selection_lock.json"), "final selection link")
        assert_equal(results["prediction_lock_sha256"], sha256(out / "prediction_lock.json"), "final blind-lock link")
        assert_equal(results["predictions_sha256"], sha256(out / "test_predictions.parquet"), "final prediction hash")
        assert_equal(preparation["sources"], selection["sources"], "prepared versus selected source hashes")
        assert_equal(prediction_lock["labels_opened"], False, "blind lock label declaration")
        assert_equal(selection["test_labels_opened"], False, "selection label declaration")
        assert_equal(selection["n_confirmation_targets_used_for_selection"], 0, "confirmation selection targets")
        assert_equal(results["is_pilot"], bool(config.get("is_pilot", False)), "pilot identity")
        assert_equal(len(predictions), manifest["splits"]["test"]["rows"], "test row count")
        assert_equal(registration["technical_pilot_only"], bool(config.get("is_pilot", False)), "registration pilot identity")
        return {"chain": "registration -> preparation -> selection -> blind probabilities -> joined final results",
                "confirmation_targets_used_for_selection": 0,
                "sealing_note": "These are auditable procedural guards, not cryptographic proof that source outcomes were unavailable."}
    auditor.check("registration_to_final_hash_chain", hash_chain)

    def sources():
        source_root = REPOSITORY / "src"
        current = {str(path.relative_to(source_root)): sha256(path)
                   for directory in ("recsys_history_v2", "recsys_research")
                   for path in sorted((source_root / directory).glob("*.py")) if path.name != "report.py"}
        expected = selection["sources"]
        context["live_source_matches"] = {name: current.get(name) == digest for name, digest in expected.items()}
        if current == expected:
            return {"source_files": len(current), "both_namespaces_checked": True,
                    "verification_basis": "current_source", "live_source_matches": True,
                    "report_modules_excluded": True}
        archived = out / "frozen_sources"
        snapshot = {str(path.relative_to(archived)): sha256(path)
                    for directory in ("recsys_history_v2", "recsys_research")
                    for path in sorted((archived / directory).glob("*.py")) if path.name != "report.py"}
        assert_equal(snapshot, expected, "archived execution source")
        return {"source_files": len(snapshot), "both_namespaces_checked": True,
                "verification_basis": "archived_execution_source", "live_source_matches": False,
                "archived_path": str(archived),
                "live_differences": sorted(name for name in set(current) | set(expected) if current.get(name) != expected.get(name)),
                "note": "Every archived execution-source digest matches; this does not assert that live code is unchanged.",
                "report_modules_excluded": True}
    auditor.check("frozen_execution_source_attestation", sources)

    old_lock = REPOSITORY / "reports/main/selection_lock.json"
    if old_lock.exists():
        def v1_immutable():
            prior = read_json(old_lock)["source_sha256"]
            actual, metadata = v1_implementation_hashes(REPOSITORY / "src/recsys_research")
            assert_equal(actual, prior, "original v1 frozen namespace")
            return {"files_compared_to_v1": len(actual), "appledouble_metadata": metadata,
                    "metadata_note": "Recognized binary filesystem metadata is separate from Python implementation; all V2-registered file hashes remain checked by source attestation."}
        auditor.check("v1_original_experiment_source_preserved", v1_immutable)
    else:
        auditor.skip("v1_original_experiment_source_preserved", "Original V1 lock absent in compact bundle; V2 source attestation still checked")

    def cohorts():
        declared = registration["cohorts"]
        for key in ("sampled_uids", "development_uids", "confirmation_uids"):
            assert_equal(manifest[key], declared[key], f"manifest cohort {key}")
        sampled, development, confirmation = (np.asarray(declared[key], np.int64)
                                               for key in ("sampled_uids", "development_uids", "confirmation_uids"))
        for name, values in (("sampled", sampled), ("development", development), ("confirmation", confirmation)):
            if len(np.unique(values)) != len(values):
                raise AssertionError(f"Duplicate {name} cohort IDs")
        if np.intersect1d(development, confirmation).size:
            raise AssertionError("Development and confirmation users overlap")
        np.testing.assert_array_equal(np.union1d(development, confirmation), np.sort(sampled))
        if len(sampled) != config["n_users"] or len(development) != config["development_users"]:
            raise AssertionError("Cohort sizes differ from registration")
        if not np.isin(predictions.uid.to_numpy(), confirmation).all():
            raise AssertionError("Final predictions include a non-confirmation user")
        start = int(config["split_days"][-1])
        end = int(config.get("end_day", 301))
        if not predictions.day.ge(start).all() or not predictions.day.lt(end).all():
            raise AssertionError("Final targets fall outside registered confirmation days")
        if predictions.duplicated(KEYS).any() or predictions.duplicated(["uid", "item_id", "day"]).any():
            raise AssertionError("Duplicate first-play confirmation keys")
        if predictions.groupby("group_id", sort=False).uid.nunique().gt(1).any():
            raise AssertionError("A ranking group spans more than one user")
        return {"sampled_users": len(sampled), "development_users": len(development),
                "confirmation_users": len(confirmation), "evaluated_users": int(predictions.uid.nunique()),
                "confirmation_days": [start, end], "development_confirmation_overlap": 0}
    auditor.check("registered_cohort_separation_and_test_period", cohorts)

    excluded_path = absolute(config["exclude_manifest"])
    if excluded_path.exists():
        def excluded_users():
            assert_equal(sha256(excluded_path), registration["excluded_manifest_sha256"], "excluded manifest hash")
            excluded = np.asarray(read_json(excluded_path)["sampled_uids"], np.int64)
            sampled = np.asarray(registration["cohorts"]["sampled_uids"], np.int64)
            if np.intersect1d(sampled, excluded).size:
                raise AssertionError("Fresh cohort overlaps registered excluded users")
            original = REPOSITORY / "reports/main/manifest.json"
            old_count = None
            if original.exists():
                old = np.asarray(read_json(original)["sampled_uids"], np.int64)
                if np.intersect1d(sampled, old).size:
                    raise AssertionError("Fresh cohort overlaps original V1 users")
                old_count = len(old)
            return {"excluded_users": len(excluded), "old_v1_users_checked": old_count, "overlap": 0}
        auditor.check("fresh_users_exclude_prior_experiments", excluded_users)
    else:
        auditor.skip("fresh_users_exclude_prior_experiments", "Excluded UID manifest is absent in compact bundle")

    raw = absolute(config["raw_dir"]) / "multi_event.parquet"
    if raw.exists() and excluded_path.exists():
        def raw_membership():
            all_users = pq.read_table(raw, columns=["uid"])["uid"].to_numpy().astype(np.int64)
            available = np.setdiff1d(np.unique(all_users), np.asarray(read_json(excluded_path)["sampled_uids"], np.int64))
            sampled = np.sort(np.random.default_rng(config["user_seed"]).choice(available, config["n_users"], replace=False))
            development = np.sort(np.random.default_rng(config["cohort_seed"]).choice(sampled, config["development_users"], replace=False))
            np.testing.assert_array_equal(sampled, registration["cohorts"]["sampled_uids"])
            np.testing.assert_array_equal(development, registration["cohorts"]["development_uids"])
            return {"raw_source_users": len(np.unique(all_users)), "uniform_cohort_reproduced_from_uid_column_only": True}
        auditor.check("raw_uid_sampling_reproduces_registered_cohort", raw_membership)
    else:
        auditor.skip("raw_uid_sampling_reproduces_registered_cohort", "Raw UID source/excluded manifest absent; outcome data not decoded")

    for name, digest in preparation["artifacts"].items():
        path = root / name
        if not path.exists() and "/" not in name and (out / name).exists():
            path = out / name
        if path.exists():
            def verify(path=path, digest=digest, name=name):
                assert_equal(sha256(path), digest, f"prepared {name}")
                return {"file": name, "bytes": path.stat().st_size}
            auditor.check("prepared_hash:" + name, verify)
        else:
            auditor.skip("prepared_hash:" + name, "Large local prepared artifact is absent")

    def blind_final():
        if len(blind) != len(predictions):
            raise AssertionError("Blind/final row counts differ")
        for column in KEYS + families:
            np.testing.assert_array_equal(blind[column].to_numpy(), predictions[column].to_numpy(), err_msg=column)
        if not np.isin(predictions.label.to_numpy(), [0, 1]).all():
            raise AssertionError("Final targets are not binary")
        for family in families:
            p = predictions[family].to_numpy()
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise AssertionError(f"Invalid probabilities for {family}")
        return {"all_blind_probabilities_unchanged_after_label_join": True, "rows": len(predictions)}
    auditor.check("blind_probabilities_unchanged_after_outcome_join", blind_final)

    queries = None
    if (root / "queries.parquet").exists():
        queries = pd.read_parquet(root / "queries.parquet")
        def sealed_queries():
            test = queries.split.eq("test")
            if not queries.loc[test, "label"].isna().all():
                raise AssertionError("Stored confirmation query labels are not sealed")
            if {"played_ratio_pct", "total_dur", "track_length_seconds", "available_at"}.intersection(queries.columns):
                raise AssertionError("Query table contains a current-outcome-derived field")
            for key in KEYS:
                np.testing.assert_array_equal(queries.loc[test, key].to_numpy(), predictions[key].to_numpy(), err_msg=key)
            dev = registration["cohorts"]["development_uids"]
            confirm = registration["cohorts"]["confirmation_uids"]
            if not queries.loc[queries.split.isin(["tune", "select"]), "uid"].isin(dev).all():
                raise AssertionError("Validation contains a confirmation user")
            if not queries.loc[test, "uid"].isin(confirm).all():
                raise AssertionError("Test contains a development user")
            boundaries = [config["warmup_days"], *config["split_days"], config.get("end_day", 301)]
            for split, left, right in zip(("train", "tune", "select", "test"), boundaries[:-1], boundaries[1:]):
                rows = queries.loc[queries.split.eq(split)]
                if not rows.day.ge(left).all() or not rows.day.lt(right).all():
                    raise AssertionError(f"Query dates outside registered {split} interval")
                if split != "test" and not np.isin(rows.label.to_numpy(), [0, 1]).all():
                    raise AssertionError(f"Invalid development labels in {split}")
            train_users = queries.loc[queries.split.eq("train"), "uid"].unique()
            test_users = queries.loc[test, "uid"].unique()
            return {"sealed_confirmation_rows": int(test.sum()), "aligned_rows": int(test.sum()),
                    "evaluated_users_with_early_training_queries": int(np.isin(test_users, train_users).sum()),
                    "evaluated_users": len(test_users), "warm_user_scope_not_a_guarantee_every_user_has_supervised_training_rows": True}
        auditor.check("stored_query_label_seal_alignment_and_cohorts", sealed_queries)
    else:
        auditor.skip("stored_query_label_seal_alignment_and_cohorts", "Local query table absent in compact bundle")

    if (root / "test_labels.parquet").exists():
        def final_labels():
            labels = pd.read_parquet(root / "test_labels.parquet")
            for column in KEYS + ["label"]:
                np.testing.assert_array_equal(labels[column].to_numpy(), predictions[column].to_numpy(), err_msg=column)
            return {"separate_labels_join_matches": True, "rows": len(labels)}
        auditor.check("separately_stored_outcomes_match_final_join", final_labels)
    else:
        auditor.skip("separately_stored_outcomes_match_final_join", "Sealed outcome file absent; final joined result still checked")

    y, users, groups = (predictions[column].to_numpy() for column in ("label", "uid", "group_id"))
    for family, record in results["models"].items():
        def recompute(family=family, record=record):
            actual = evaluate(y, predictions[family].to_numpy(), users, groups, k=record["ranking_k"])
            assert_equal(actual, record, family + ".metrics")
            return {"metrics_compared": len(actual), "rows": len(y)}
        auditor.check("saved_metric_recomputation:" + family, recompute)

    def primary_interval():
        actual = paired_user_bootstrap(y, predictions.selected_auto.to_numpy(), predictions.history_global.to_numpy(),
                                       users, groups, n_boot=config["bootstrap_samples"], seed=20261026)
        assert_equal(actual, results["primary_history_vs_autofe"], "primary bootstrap")
        return {"resampling_unit": "user", "seed": 20261026, "n_boot": config["bootstrap_samples"],
                "baseline": "selected_auto", "proposed": "history_global", "conditional_on_observed_month": True}
    auditor.check("primary_paired_user_bootstrap_recomputed", primary_interval)
    for family, expected in results.get("secondary_vs_manual", {}).items():
        def secondary_interval(family=family, expected=expected):
            actual = paired_user_bootstrap(y, predictions.manual.to_numpy(), predictions[family].to_numpy(),
                                           users, groups, n_boot=config["bootstrap_samples"], seed=20261026)
            assert_equal(actual, expected, f"secondary {family} bootstrap")
            return {"baseline": "manual", "proposed": family, "seed": 20261026}
        auditor.check("secondary_paired_user_bootstrap_recomputed:" + family, secondary_interval)
    if "temporal_sensitivity" in results:
        def temporal_interval():
            from recsys_history_v2.statistics import temporal_sensitivity
            actual = temporal_sensitivity(y, predictions.selected_auto.to_numpy(), predictions.history_global.to_numpy(),
                                          predictions.day.to_numpy(), config["bootstrap_samples"])
            assert_equal(actual, results["temporal_sensitivity"], "temporal sensitivity")
            return {"secondary_circular_day_blocks": [3, 7], "seed": 20261026,
                    "not_joint_user_time_inference": True}
        if context.get("live_source_matches", {}).get("recsys_history_v2/statistics.py", False):
            auditor.check("secondary_temporal_sensitivity_recomputed", temporal_interval)
        else:
            auditor.skip("secondary_temporal_sensitivity_recomputed", "Live temporal-statistics implementation differs from archived execution source")

    def independent_ranking():
        grouped = predictions.groupby("group_id", sort=True)
        mixed = [key for key, frame in grouped if len(frame) >= 2 and 0 < int(frame.label.sum()) < len(frame)]
        if not mixed:
            raise AssertionError("No mixed groups available for an independent mixed-group NDCG comparison")
        selected = np.random.default_rng(20261026).choice(np.asarray(mixed), min(20, len(mixed)), replace=False)
        max_error = 0.
        for key in selected:
            frame = grouped.get_group(key)
            labels = frame.label.to_numpy(np.float64)
            for family in families:
                scores = frame[family].to_numpy(np.float64)
                reference = ndcg_score(labels[None], scores[None], k=10, ignore_ties=False)
                actual = evaluate(labels, scores, frame.uid.to_numpy(), frame.group_id.to_numpy(), k=10)["ndcg_mixed"]
                error = abs(actual - reference)
                max_error = max(max_error, error)
                if error > TOLERANCE:
                    raise AssertionError(f"Independent NDCG differs for {family}, group {key}")
        return {"mixed_groups": len(selected), "families": len(families), "comparisons": len(selected) * len(families),
                "api": "sklearn.metrics.ndcg_score(ignore_ties=False,k=10)", "max_absolute_error": max_error}
    eligibility = predictions.groupby("group_id", sort=False).label.agg(["size", "sum"])
    if ((eligibility["size"] >= 2) & (eligibility["sum"] > 0) & (eligibility["sum"] < eligibility["size"])).any():
        auditor.check("independent_sklearn_tie_aware_ndcg", independent_ranking)
    else:
        auditor.skip("independent_sklearn_tie_aware_ndcg", "No mixed-label ranking groups; all-defined ranking metrics may still exist")

    def hybrid():
        weight = selection["blend"]["auto_weight"]
        auto, history = (np.clip(predictions[column].to_numpy(), 1e-7, 1-1e-7)
                         for column in ("selected_auto", "history_global"))
        logits = weight * (np.log(auto) - np.log1p(-auto)) + (1-weight) * (np.log(history) - np.log1p(-history))
        expected = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        np.testing.assert_allclose(expected, predictions.hybrid.to_numpy(), rtol=0, atol=TOLERANCE)
        return {"auto_weight_chosen_before_confirmation": weight, "logit_blend_verified": True}
    auditor.check("frozen_hybrid_composition", hybrid)

    test_positions = np.flatnonzero(queries.split.eq("test").to_numpy()) if queries is not None else None
    test_position_to_prediction = ({int(position): index for index, position in enumerate(test_positions)}
                                   if test_positions is not None else {})
    for family, specification in selection["models"].items():
        def path_coverage(specification=specification):
            recorded = specification["model_paths"]
            if not recorded or len(recorded) != len(set(recorded)) or set(recorded) != set(specification["model_sha256"]):
                raise AssertionError("Frozen model paths are not exactly covered by unique SHA256 entries")
            return {"checkpoints": len(recorded), "exact_hash_coverage": True}
        auditor.check("model_specification_hash_coverage:" + family, path_coverage)
        paths = [absolute(path) for path in specification["model_paths"]]
        for recorded_path, expected in specification["model_sha256"].items():
            path = absolute(recorded_path)
            if path.exists():
                def model_hash(path=path, expected=expected):
                    assert_equal(sha256(path), expected, "frozen model hash")
                    return {"bytes": path.stat().st_size}
                auditor.check(f"model_hash:{family}:{path.name}", model_hash)
            else:
                auditor.skip(f"model_hash:{family}:{path.name}", "Trained weights absent in compact bundle")
        if not all(path.exists() for path in paths):
            auditor.skip("model_reload:" + family, "At least one frozen checkpoint is absent")
            continue
        if specification.get("kind", "tabular") != "history":
            feature_path = root / "features.parquet"
            if queries is None or not feature_path.exists():
                auditor.skip("model_reload:" + family, "Local queries/features needed for bounded CatBoost prediction replay")
                continue
            def tabular_reload(family=family, specification=specification, paths=paths):
                from catboost import CatBoostClassifier
                positions = test_positions[:MAX_RELOAD_ROWS]
                if not len(positions):
                    raise AssertionError("No local test rows")
                x = _feature_rows(feature_path, positions, specification["columns"])
                output = []
                for path in paths:
                    model = CatBoostClassifier().load_model(str(path))
                    if not np.isfinite(model.get_leaf_values()).all():
                        raise AssertionError("Non-finite CatBoost leaf weights")
                    output.append(model.predict_proba(x[specification["columns"]], thread_count=1)[:, 1])
                actual = np.mean(output, axis=0)
                expected = predictions[family].to_numpy()[:len(positions)]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=TOLERANCE)
                return {"checkpoints": len(paths), "rows": len(positions),
                        "max_absolute_error": float(np.max(np.abs(actual - expected))), "fit_calls": 0}
            auditor.check("model_reload:" + family, tabular_reload)
            continue

        history_state = {}
        def history_checkpoints(family=family, paths=paths):
            import torch
            torch.set_num_threads(1)
            summary = []
            for path in paths:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                state = saved["state"]
                if not state or any(not bool(torch.isfinite(value).all()) for value in state.values()):
                    raise AssertionError("Empty or non-finite Torch state")
                summary.append({"file": path.name, "parameters_and_buffers": int(sum(value.numel() for value in state.values())),
                                "epoch": int(saved["epoch"]), "global_dim": int(saved["global_dim"]),
                                "d_model": int(saved["model_config"]["d_model"]),
                                "n_layers": int(saved["model_config"].get("n_layers", 2))})
                del saved, state
            devices = [fit.get("device", "cpu") for fit in specification.get("fits", [])]
            history_state["cpu_fitted"] = bool(devices) and all(device == "cpu" for device in devices)
            history_state["gpu_fitted"] = bool(devices) and all(str(device).startswith("cuda") for device in devices)
            return {"checkpoints": summary, "weights_only_deserialization": True, "all_state_finite": True}
        auditor.check("torch_checkpoint_state:" + family, history_checkpoints)
        if (queries is None or not (root / "contexts").exists()
                or not (root / "global_features.npy").exists()):
            auditor.skip("history_prediction_replay:" + family, "Local sealed queries/contexts/global inputs absent")
            continue
        if not all(context.get("live_source_matches", {}).get(name, False)
                   for name in ("recsys_history_v2/model.py", "recsys_history_v2/trainer.py")):
            auditor.skip("history_prediction_replay:" + family,
                         "Live model/trainer differs from archived execution source; checkpoint hashes and finite state remain checked")
            continue
        if history_state.get("gpu_fitted", False):
            import torch
            if not torch.cuda.is_available():
                auditor.skip("history_prediction_replay:" + family,
                             "Frozen CUDA run and no CUDA available: CPU checkpoint hashes/finite state checked; no CPU/BF16 equivalence is claimed")
                continue

            def gpu_history_replay(family=family, paths=paths):
                import torch
                from recsys_history_v2.trainer import load_history
                device = torch.device("cuda")
                result, row_order, original_shapes = [], None, None
                for path in paths:
                    model, saved = load_history(path, device)
                    probabilities, indices, shape = _first_original_history_batch(
                        model, root, queries, config, saved["global_dim"], device)
                    if row_order is None:
                        row_order, original_shapes = indices, shape
                    else:
                        np.testing.assert_array_equal(indices, row_order, err_msg="Original GPU-batch row order changed between checkpoints")
                        assert_equal(shape, original_shapes, "Original GPU-batch shape")
                    # Frozen mean_history averages float64 placement arrays;
                    # keep that ensemble arithmetic after float32 sigmoid.
                    result.append(probabilities)
                    torch.cuda.synchronize(device)
                    del model, saved
                    torch.cuda.empty_cache()
                compared = min(MAX_RELOAD_ROWS, len(row_order))
                actual = np.mean(result, axis=0)[:compared]
                expected_indices = [test_position_to_prediction[int(position)] for position in row_order[:compared]]
                expected = predictions[family].to_numpy()[expected_indices]
                maximum_error = float(np.max(np.abs(actual - expected)))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=GPU_MODEL_PROBABILITY_TOLERANCE)
                hardware = torch.cuda.get_device_name(device)
                frozen_hardware = selection.get("environment", {}).get("gpu")
                return {"checkpoints": len(paths), "rows_compared": compared,
                        "rows_computed_before_comparison": len(row_order),
                        "original_full_batch_shape": original_shapes,
                        "batch_policy": "Unmodified first original GroupBatches batch, including every candidate; only comparison is capped.",
                        "device": str(device), "gpu": hardware,
                        "registered_gpu": frozen_hardware,
                        "gpu_name_matches_registration": hardware == frozen_hardware,
                        "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                        "mixed_precision": config.get("mixed_precision", "fp32"),
                        "max_absolute_error": maximum_error,
                        "absolute_tolerance": GPU_MODEL_PROBABILITY_TOLERANCE,
                        "tolerance_note": "Strict absolute probability tolerance 1e-6 with original GPU batch geometry and registered autocast policy; no BF16 tolerance widening.",
                        "fit_calls": 0, "refit_calls": 0}
            auditor.check("history_prediction_replay:" + family, gpu_history_replay)
            continue
        if not history_state.get("cpu_fitted", False):
            auditor.skip("history_prediction_replay:" + family,
                         "Checkpoint training device is missing or mixed; hashes and finite state checked without guessing an inference precision")
            continue
        def history_replay(family=family, paths=paths):
            import torch
            from recsys_history_v2.trainer import GroupBatches, load_history
            result = []
            row_order = None
            for path in paths:
                model, saved = load_history(path, torch.device("cpu"))
                loader = GroupBatches(root, queries, "test", config["neural_batch_groups"], saved["global_dim"])
                chosen_groups, chosen_codes, remaining = [], [], MAX_RELOAD_ROWS
                for rows, code in zip(loader.groups, loader.group_codes):
                    chosen = rows[:remaining]
                    if len(chosen):
                        chosen_groups.append(chosen)
                        chosen_codes.append(code)
                        remaining -= len(chosen)
                    if not remaining:
                        break
                loader.groups = chosen_groups
                loader.group_codes = np.asarray(chosen_codes, np.int64)
                loader.rows = np.concatenate(chosen_groups)
                probabilities, indices = [], []
                with torch.no_grad():
                    for batch in loader.batches():
                        rows = batch.pop("row_indices").numpy()
                        p = torch.sigmoid(model(batch).float()).numpy()
                        valid = rows >= 0
                        probabilities.extend(p[valid].tolist())
                        indices.extend(rows[valid].tolist())
                if row_order is None:
                    row_order = indices
                elif row_order != indices:
                    raise AssertionError("Checkpoint replay row order changed")
                result.append(probabilities)
                del model, saved, loader
            actual = np.mean(result, axis=0)
            expected_indices = [test_position_to_prediction[int(position)] for position in row_order]
            expected = predictions[family].to_numpy()[expected_indices]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=MODEL_PROBABILITY_TOLERANCE)
            return {"checkpoints": len(paths), "rows": len(row_order), "device": "cpu",
                    "max_absolute_error": float(np.max(np.abs(actual - expected))),
                    "absolute_tolerance": MODEL_PROBABILITY_TOLERANCE, "fit_calls": 0}
        auditor.check("history_prediction_replay:" + family, history_replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--output", type=Path, help="Write a new audit JSON, preserving any existing audit")
    arguments = parser.parse_args()
    config = yaml.safe_load(absolute(arguments.config).read_text())
    destination = absolute(arguments.output) if arguments.output else absolute(config["output_dir"]) / "audit.json"
    if destination.exists():
        parser.error(f"Audit already exists at {destination}; choose a new --output path")
    auditor = Auditor()
    audit(config, auditor)
    if not auditor.save(destination, config):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
