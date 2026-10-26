"""Fresh-user-cohort preparation for the larger prospective v2 experiment.

The old experiment's users are excluded before any event data are expanded.
Development and confirmation users are then sampled without looking at labels.
All sampled users contribute early training examples, so confirmation evaluates
future behaviour of warm users, not unseen-user cold start. Their later labels
are kept outside the development query table.

This separation is an auditable workflow guard, not cryptographic concealment:
the source event stream must retain every event, including earlier test days,
for the declared rolling-history evaluation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from recsys_research.data import (
    SPLIT_NAMES,
    _source_manifest,
    add_availability,
    make_queries,
    read_relation_mapping,
    read_user_histories,
    sha256_file,
)

IDENTITY_COLUMNS = ["uid", "item_id", "day", "group_id"]
INPUT_COLUMNS = [
    "uid", "item_id", "artist_id", "album_id", "timestamp", "day",
    "group_id", "split", "label", "cohort",
]


def _json_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def choose_cohorts(
    all_user_ids: np.ndarray, excluded_user_ids: np.ndarray, config: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Uniformly draw a fresh cohort, then its development subset.

    Duplicated IDs in an event table do not alter sampling weights. The two
    independent seeds make the population draw separate from its allocation.
    """
    all_users = np.unique(np.asarray(all_user_ids, dtype=np.uint32))
    excluded = np.unique(np.asarray(excluded_user_ids, dtype=np.uint32))
    available = np.setdiff1d(all_users, excluded, assume_unique=True)
    n_users = int(config.get("n_users", 9000))
    n_development = int(config.get("development_users", 1000))
    if not 2 <= n_users <= len(available):
        raise ValueError(f"n_users must be in [2, {len(available)}] after excluding v1 users")
    if not 1 <= n_development < n_users:
        raise ValueError("development_users must leave nonempty development and confirmation cohorts")
    sampled = np.sort(np.random.default_rng(int(config.get("user_seed", 20261026))).choice(
        available, n_users, replace=False
    ))
    development = np.sort(np.random.default_rng(int(config.get("cohort_seed", 371))).choice(
        sampled, n_development, replace=False
    ))
    confirmation = np.setdiff1d(sampled, development, assume_unique=True)
    assert not np.intersect1d(sampled, excluded).size
    assert not np.intersect1d(development, confirmation).size
    return {
        "sampled_uids": sampled,
        "development_uids": development,
        "confirmation_uids": confirmation,
    }


def make_v2_queries(
    events: pd.DataFrame,
    config: dict[str, Any],
    development_uids: np.ndarray,
    confirmation_uids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Make common candidate groups and separate confirmation outcomes.

    Reuses the tested v1 first-play, >=50 threshold, day sampling and outcome
    availability purge. Per-user sampling seeds are independent of other users,
    so filtering cohorts after day sampling does not change their chosen days.
    """
    development = np.unique(np.asarray(development_uids, dtype=np.uint32))
    confirmation = np.unique(np.asarray(confirmation_uids, dtype=np.uint32))
    if not len(development) or not len(confirmation):
        raise ValueError("Both development and confirmation cohorts must be nonempty")
    if np.intersect1d(development, confirmation).size:
        raise ValueError("Development and confirmation users must be disjoint")
    source_users = np.unique(events["uid"].to_numpy(dtype=np.uint32))
    allocated = np.union1d(development, confirmation)
    if not np.isin(source_users, allocated).all():
        raise ValueError("Every event user must be allocated to a declared cohort")
    end_day = int(config.get("end_day", 301))
    boundaries = config.get("split_days", [210, 240, 270])
    if end_day <= int(boundaries[-1]):
        raise ValueError("end_day must be strictly after the final split boundary")
    # Restrict only the temporary source of supervised queries before v1's
    # per-user day sampler, so pilot budgets are spent inside their own period.
    # The returned/persisted events remain the untruncated complete histories.
    query_source = events
    if len(events) and int(events["day"].max()) >= end_day:
        query_source = events.loc[events["day"].lt(end_day)]
    queries = make_queries(query_source, config)
    purged = dict(queries.attrs.get("purged_label_rows_by_split", {}))
    # Restrict supervised queries only: every original event remains available
    # in the full history archive, subject to the later causal context cutoff.
    queries = queries.loc[queries["day"].lt(end_day)].copy()
    is_development = queries["uid"].isin(development).to_numpy()
    keep = (
        queries["split"].eq("train").to_numpy()
        | (queries["split"].isin(["tune", "select"]).to_numpy() & is_development)
        | (queries["split"].eq("test").to_numpy() & ~is_development)
    )
    queries = queries.loc[keep].copy().reset_index(drop=True)
    queries["cohort"] = np.where(queries["uid"].isin(development), "development", "confirmation")
    queries["cohort"] = queries["cohort"].astype("category")
    is_test = queries["split"].eq("test")
    sealed = queries.loc[is_test, IDENTITY_COLUMNS + ["label"]].copy().reset_index(drop=True)
    sealed["label"] = sealed["label"].astype("uint8")
    if queries.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("First-play queries must be unique by uid/item/day/group")
    # Diagnostic ratio in the v1 table would reveal y even after blanking label.
    # Do not persist it, duration, or an outcome-derived availability timestamp.
    queries["label"] = queries["label"].astype("float32")
    queries.loc[is_test, "label"] = np.nan
    queries = queries[INPUT_COLUMNS]
    stats: dict[str, Any] = {
        "purged_label_rows_by_split_before_cohort_filter": purged,
        "query_end_day_exclusive": end_day,
        "splits": {},
    }
    for split in SPLIT_NAMES:
        frame = queries.loc[queries["split"].eq(split)]
        group_sizes = frame.groupby("group_id", sort=False, observed=True).size()
        record: dict[str, Any] = {
            "rows": len(frame), "users": int(frame["uid"].nunique()),
            "groups": len(group_sizes),
            "day_min": int(frame["day"].min()) if len(frame) else None,
            "day_max": int(frame["day"].max()) if len(frame) else None,
            "group_size_median": float(group_sizes.median()) if len(frame) else None,
            "group_size_p90": float(group_sizes.quantile(.9)) if len(frame) else None,
            "labels_sealed": split == "test",
        }
        if split != "test":
            record["positive_rows"] = int(frame["label"].sum())
            record["positive_rate"] = float(frame["label"].mean()) if len(frame) else None
        stats["splits"][split] = record
    return queries, sealed, stats


def prepare_data(
    config: dict[str, Any], output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Persist complete fresh histories, blinded queries, and separate labels.

    The preparation refuses to overwrite existing outputs. A new run directory
    is required after changing data code/config; stale caches cannot silently
    acquire a new preparation hash.
    """
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ["sample_events.parquet", "queries.parquet", "test_labels.parquet", "manifest.json"]
    existing = [name for name in names if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Prepared outputs already exist: {existing}; use a new output directory")
    raw_dir = Path(config.get("raw_dir", "data/raw"))
    excluded_path = Path(config.get("exclude_manifest", "reports/main/manifest.json"))
    excluded_manifest = json.loads(excluded_path.read_text())
    if "sampled_uids" not in excluded_manifest:
        raise ValueError("exclude_manifest must declare the old experiment's sampled_uids")
    excluded = np.asarray(excluded_manifest["sampled_uids"], dtype=np.uint32)
    sources = _source_manifest(raw_dir)
    source_path = raw_dir / "multi_event.parquet"
    source_users = pq.read_table(source_path, columns=["uid"]).column("uid").to_numpy()
    cohorts = choose_cohorts(source_users, excluded, config)
    # Cohort assignment has now finished; no source outcomes have been decoded.
    events = add_availability(read_user_histories(source_path, cohorts["sampled_uids"]))
    relations = {}
    for relation in ("artist_id", "album_id"):
        filename = relation.replace("_id", "_item_mapping.parquet")
        mapping, relations[relation] = read_relation_mapping(
            raw_dir / filename, events["item_id"].unique(), relation
        )
        events[relation] = events["item_id"].map(mapping).fillna(-1).astype("int32")
        relations[relation]["sample_items_missing_mapping"] = int(
            events.loc[events[relation].eq(-1), "item_id"].nunique()
        )
    queries, sealed, stats = make_v2_queries(
        events, config, cohorts["development_uids"], cohorts["confirmation_uids"]
    )
    frames = {"sample_events.parquet": events, "queries.parquet": queries, "test_labels.parquet": sealed}
    for filename, frame in frames.items():
        temporary = output / (filename + ".partial")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(output / filename)
    history_lengths = events.groupby("uid", sort=False).size()
    manifest: dict[str, Any] = {
        "experiment": "v2 fresh-user prospective warm-user confirmation",
        "dataset": "yandex/yambda sequential/50m", "config": config,
        "config_sha256": _json_fingerprint(config),
        "data_module_sha256": sha256_file(Path(__file__)),
        "v1_data_module_sha256": sha256_file(Path(__file__).parents[1] / "recsys_research" / "data.py"),
        "sources": sources,
        "excluded_manifest": {"path": str(excluded_path), "sha256": sha256_file(excluded_path),
                              "declared_users": len(np.unique(excluded)),
                              "source_users_excluded": len(np.intersect1d(source_users, excluded))},
        "source_users": len(np.unique(source_users)),
        "available_fresh_users": len(np.setdiff1d(source_users, excluded)),
        **{name: values.tolist() for name, values in cohorts.items()},
        "sampled_users": len(cohorts["sampled_uids"]),
        "development_users": len(cohorts["development_uids"]),
        "confirmation_users": len(cohorts["confirmation_uids"]),
        "old_user_overlap": 0, "development_confirmation_overlap": 0,
        "retained_events": len(events), "retained_items": int(events["item_id"].nunique()),
        "event_type_counts": {str(k): int(v) for k, v in events["event_type"].value_counts().items()},
        "events_per_sampled_user": {str(uid): int(n) for uid, n in history_lengths.items()},
        "history_lengths": {"min": int(history_lengths.min()), "median": float(history_lengths.median()),
                            "p90": float(history_lengths.quantile(.9)), "max": int(history_lengths.max())},
        "timestamp_min": int(events["timestamp"].min()), "timestamp_max": int(events["timestamp"].max()),
        "relations": relations,
        **stats,
        "rules": {
            "sample": "uniform users after excluding every v1 sampled UID; full available histories",
            "cohort": "uniform development subset before reading outcomes; remaining users confirmation",
            "training": "early training queries from both cohorts; not a cold-user experiment",
            "validation": "tune/select queries from development users only",
            "confirmation": "test queries from confirmation users only, query.day<end_day; their tune/select rows are not supervised fitting/evaluation targets, while causal earlier events remain available history",
            "label": "first timestamp/source-order listen per user-item-day across sources, then is_organic==0; ratio>=50",
            "availability": "listen timestamp+ceil(length*uncapped_ratio/100)+5; other timestamp+5",
            "feature_cutoff": "available_day < query.day - embargo_days",
            "sealing": "test label absent from queries and test target ratio/duration absent from all queries; separate test_labels.parquet",
            "sealing_limit": "procedural guard only: complete raw histories retain later outcomes for declared rolling evaluation",
        },
        "artifacts": {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
                      for name in frames},
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary = output / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "manifest.json")
    return events, queries, manifest
