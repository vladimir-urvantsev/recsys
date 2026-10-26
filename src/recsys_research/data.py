"""User-preserving Yambda sampling and an explicit logged-play evaluation task.

All source events of sampled users are retained. Only supervised query DAYS are
subsampled; outcomes do not enter this sampling decision. Source timestamps are
relative seconds (already rounded to multiples of five), not Unix dates.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DAY_SECONDS = 86_400
SPLIT_NAMES = ("train", "tune", "select", "test")
DEFAULT_REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
EVENT_SHA256 = "2eaf0619bcfadd31a907e2bc88aab7ec34e36cbafca41f4e6ff34188985d990d"
SOURCE_SHA256 = {
    "multi_event.parquet": EVENT_SHA256,
    "artist_item_mapping.parquet": "cf1110aa2413ed70008ec9232d395660b2181ffda731002fb74da54e87b6f25d",
    "album_item_mapping.parquet": "559b6576063d96280414574b32292eee7b47ee0b9f2952cb3ae670a5d19125e4",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_users(user_ids: np.ndarray, n_users: int, seed: int) -> np.ndarray:
    """Uniformly sample unique users, independent of their activity or labels."""
    users = np.unique(np.asarray(user_ids, dtype=np.uint32))
    if not 1 <= n_users <= len(users):
        raise ValueError(f"n_users must be in [1, {len(users)}], got {n_users}")
    return np.sort(np.random.default_rng(seed).choice(users, n_users, replace=False))


def _nullable_integer_dtype(dtype: pa.DataType) -> Any:
    if pa.types.is_uint16(dtype):
        return pd.UInt16Dtype()
    if pa.types.is_uint32(dtype):
        return pd.UInt32Dtype()
    return None


def read_user_histories(path: str | Path, user_ids: np.ndarray) -> pd.DataFrame:
    """Filter nested rows in Arrow before expanding their event lists.

    This avoids materializing 48M Python event objects. It also works with flat
    input files, although the official pinned input is the sequential file.
    """
    parquet = pq.ParquetFile(path)
    wanted = pa.array(user_ids, type=pa.uint32())
    required = [
        "uid", "timestamp", "item_id", "is_organic", "played_ratio_pct",
        "track_length_seconds", "event_type",
    ]
    missing = set(required) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing source fields: {sorted(missing)}")
    timestamp_type = parquet.schema_arrow.field("timestamp").type
    nested = pa.types.is_list(timestamp_type) or pa.types.is_large_list(timestamp_type)
    tables: list[pa.Table] = []
    for batch in parquet.iter_batches(
        batch_size=64 if nested else 131_072, columns=required, use_threads=False
    ):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table["uid"], value_set=wanted))
        if not len(table):
            continue
        if nested:
            lengths = pc.list_value_length(table["timestamp"]).to_numpy()
            arrays: dict[str, Any] = {
                "uid": pa.array(np.repeat(table["uid"].to_numpy(), lengths)),
            }
            for name in required[1:]:
                actual_lengths = pc.list_value_length(table[name]).to_numpy()
                if not np.array_equal(lengths, actual_lengths):
                    raise ValueError(f"Nonparallel event lists in field {name}")
                arrays[name] = pc.list_flatten(table[name])
            table = pa.table(arrays)
        tables.append(table)
    if not tables:
        raise ValueError("No events found for selected users")
    events = pa.concat_tables(tables).to_pandas(types_mapper=_nullable_integer_dtype)
    for column in ("uid", "item_id", "timestamp"):
        if events[column].isna().any():
            raise ValueError(f"Unexpected null {column}")
        events[column] = events[column].astype("uint32")
    events["is_organic"] = events["is_organic"].astype("uint8")
    events["event_type"] = events["event_type"].astype("category")
    unknown = set(events["event_type"].dropna().unique()) - {
        "listen", "like", "dislike", "unlike", "undislike"
    }
    if unknown or events["event_type"].isna().any():
        raise ValueError(f"Unexpected event types: {unknown}")
    events["source_order"] = np.arange(len(events), dtype=np.uint32)
    return events


def add_availability(events: pd.DataFrame) -> pd.DataFrame:
    """Conservatively delay play-outcome visibility until playback can finish.

    A listen with ratio >100 is deliberately NOT capped: replays take time.
    Five seconds cover source timestamp quantization. The experiment adds a
    separate daily embargo before allowing these records into feature history.
    """
    result = events.copy()
    listen = result["event_type"].eq("listen").to_numpy()
    if result.loc[listen, ["played_ratio_pct", "track_length_seconds"]].isna().any().any():
        raise ValueError("Listen events require a play ratio and track duration")
    ratio = result["played_ratio_pct"].fillna(0).to_numpy(dtype=np.uint64)
    length = result["track_length_seconds"].fillna(0).to_numpy(dtype=np.uint64)
    # Exact integer ceiling avoids floating-point off-by-one errors.
    playback_seconds = np.where(listen, (ratio * length + 99) // 100, 0)
    result["available_at"] = (
        result["timestamp"].to_numpy(dtype=np.uint64) + playback_seconds + 5
    ).astype("uint64")
    result["day"] = (result["timestamp"].to_numpy() // DAY_SECONDS).astype("uint16")
    result["available_day"] = (
        result["available_at"].to_numpy() // DAY_SECONDS
    ).astype("uint32")
    return result


def read_relation_mapping(
    path: str | Path, item_ids: np.ndarray, relation: str
) -> tuple[pd.Series, dict[str, int]]:
    """Use the smallest relation ID when an item has multiple mappings.

    This is a fixed, outcome-independent reduction of many-to-many metadata,
    rather than selecting a main artist/album using future popularity.
    """
    wanted = pa.array(np.unique(item_ids), type=pa.uint32())
    fragments: list[pa.Table] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["item_id", relation], batch_size=262_144, use_threads=False
    ):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table["item_id"], value_set=wanted))
        if len(table):
            fragments.append(table)
    if fragments:
        mapping = pa.concat_tables(fragments).to_pandas()
        mapping = mapping.dropna().drop_duplicates()
        counts = mapping.groupby("item_id", sort=False)[relation].nunique()
        selected = mapping.groupby("item_id", sort=False)[relation].min()
    else:
        selected = pd.Series(dtype="uint32", name=relation)
        counts = pd.Series(dtype="int64")
    selected.index.name = "item_id"
    return selected, {
        "source_rows": int(parquet.metadata.num_rows),
        "sample_items_with_mapping": int(len(selected)),
        "sample_items_with_multiple_values": int(counts.gt(1).sum()),
        "max_values_per_sample_item": int(counts.max()) if len(counts) else 0,
    }


def assign_splits(days: pd.Series, config: dict[str, Any]) -> np.ndarray:
    boundaries = config.get("split_days", [210, 240, 270])
    if len(boundaries) != 3 or not all(
        int(a) < int(b) for a, b in zip(boundaries, boundaries[1:])
    ):
        raise ValueError("split_days must contain three increasing day boundaries")
    warmup = int(config.get("warmup_days", 30))
    if not 0 <= warmup < int(boundaries[0]):
        raise ValueError("warmup_days must precede the training boundary")
    values = days.to_numpy()
    result = np.full(len(values), "warmup", dtype=object)
    for name, start, end in zip(
        SPLIT_NAMES, [warmup, *boundaries], [*boundaries, np.inf]
    ):
        result[(values >= start) & (values < end)] = name
    return result


def make_queries(events: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Logged first-play completion, with uniformly sampled user/split days.

    First play is chosen across ALL listens before filtering recommendation
    traffic. An earlier organic play therefore prevents a later same-day
    recommendation play from entering this task. Candidate sets are the logged
    played tracks; they are not a claimed full-catalog retrieval benchmark.
    Development labels must finish becoming available before their next stage
    begins; this eligibility filter runs before outcome-independent day sampling.
    """
    listens = events.loc[events["event_type"].eq("listen")].copy()
    listens = listens.sort_values(["timestamp", "source_order"], kind="stable")
    listens = listens.drop_duplicates(["uid", "item_id", "day"], keep="first")
    queries = listens.loc[listens["is_organic"].eq(0)].copy()
    queries["label"] = queries["played_ratio_pct"].ge(50).astype("uint8")
    queries["split"] = assign_splits(queries["day"], config)
    queries = queries.loc[queries["split"].ne("warmup")]
    boundaries = config.get("split_days", [210, 240, 270])
    eligible = np.ones(len(queries), dtype=bool)
    purged_counts = {split: 0 for split in SPLIT_NAMES}
    for split, boundary in zip(SPLIT_NAMES[:3], boundaries):
        late = queries["split"].eq(split) & queries["available_at"].ge(int(boundary) * DAY_SECONDS)
        purged_counts[split] = int(late.sum())
        eligible &= ~late.to_numpy()
    queries = queries.loc[eligible]
    budgets = config.get("days_per_user", {"train": 6, "tune": 2, "select": 2, "test": 3})
    for split in SPLIT_NAMES:
        if split not in budgets or int(budgets[split]) < 0:
            raise ValueError(f"days_per_user requires a nonnegative {split} budget")
    day_seed = int(config.get("day_seed", 1701))
    selected: list[pd.DataFrame] = []
    for (uid, split), frame in queries.groupby(["uid", "split"], sort=True, observed=True):
        days = np.sort(frame["day"].unique())
        budget = min(int(budgets[split]), len(days))
        if not budget:
            continue
        generator = np.random.default_rng(
            np.random.SeedSequence([day_seed, int(uid), SPLIT_NAMES.index(split)])
        )
        chosen = generator.choice(days, budget, replace=False)
        selected.append(frame.loc[frame["day"].isin(chosen)])
    columns = [
        "uid", "item_id", "artist_id", "album_id", "timestamp", "day", "label",
        "group_id", "split", "track_length_seconds", "played_ratio_pct", "is_organic",
    ]
    if not selected:
        result = pd.DataFrame(columns=columns)
        result.attrs["purged_label_rows_by_split"] = purged_counts
        return result
    result = pd.concat(selected, ignore_index=True)
    # Collision-free stable ID independent of which other user/days were sampled.
    result["group_id"] = (
        result["uid"].to_numpy(dtype=np.uint64) * np.uint64(1 << 32)
        + result["day"].to_numpy(dtype=np.uint64)
    )
    result["split"] = pd.Categorical(result["split"], categories=SPLIT_NAMES, ordered=True)
    result = result[columns].sort_values(
        ["day", "uid", "item_id"], kind="stable"
    ).reset_index(drop=True)
    result.attrs["purged_label_rows_by_split"] = purged_counts
    return result


def _source_manifest(raw_dir: Path) -> dict[str, Any]:
    metadata_path = raw_dir / "huggingface_revision.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    revision = metadata.get("sha", DEFAULT_REVISION)
    sources: dict[str, Any] = {}
    for name, remote in (
        ("multi_event.parquet", "sequential/50m/multi_event.parquet"),
        ("artist_item_mapping.parquet", "artist_item_mapping.parquet"),
        ("album_item_mapping.parquet", "album_item_mapping.parquet"),
    ):
        path = raw_dir / name
        digest = sha256_file(path)
        if digest != SOURCE_SHA256[name]:
            raise ValueError(f"Source checksum mismatch for {path}: {digest}")
        sources[name] = {
            "url": f"https://huggingface.co/datasets/yandex/yambda/resolve/{revision}/{remote}",
            "revision": revision, "sha256": digest, "bytes": path.stat().st_size,
        }
    return sources


def prepare_data(
    config: dict[str, Any], output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prepare and persist complete sampled histories and supervised queries."""
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(config.get("raw_dir", "data/raw"))
    source = raw_dir / "multi_event.parquet"
    sources = _source_manifest(raw_dir)
    all_users = pq.read_table(source, columns=["uid"]).column("uid").to_numpy()
    selected_users = choose_users(
        all_users, int(config.get("n_users", 1000)), int(config.get("user_seed", 2026))
    )
    events = add_availability(read_user_histories(source, selected_users))
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
    queries = make_queries(events, config)
    history_lengths = events.groupby("uid", sort=False).size()
    splits: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        frame = queries.loc[queries["split"].eq(name)]
        group_sizes = frame.groupby("group_id", sort=False).size()
        splits[name] = {
            "rows": int(len(frame)), "users": int(frame["uid"].nunique()),
            "groups": int(len(group_sizes)), "positive_rows": int(frame["label"].sum()),
            "positive_rate": float(frame["label"].mean()) if len(frame) else None,
            "day_min": int(frame["day"].min()) if len(frame) else None,
            "day_max": int(frame["day"].max()) if len(frame) else None,
            "group_size_median": float(group_sizes.median()) if len(frame) else None,
            "group_size_p90": float(group_sizes.quantile(.9)) if len(frame) else None,
        }
    # Source-order IDs are persisted to make the first-play tie rule auditable.
    events.to_parquet(output / "sample_events.parquet", index=False, compression="zstd")
    queries.to_parquet(output / "queries.parquet", index=False, compression="zstd")
    manifest: dict[str, Any] = {
        "dataset": "yandex/yambda sequential/50m", "config": config,
        "sources": sources, "source_users": int(len(np.unique(all_users))),
        "sampled_users": int(len(selected_users)), "sampled_uids": selected_users.tolist(),
        "retained_events": int(len(events)), "retained_items": int(events["item_id"].nunique()),
        "event_type_counts": {str(k): int(v) for k, v in events["event_type"].value_counts().items()},
        "history_lengths": {
            "min": int(history_lengths.min()), "median": float(history_lengths.median()),
            "p90": float(history_lengths.quantile(.9)), "max": int(history_lengths.max()),
        },
        "events_per_sampled_user": {str(uid): int(n) for uid, n in history_lengths.items()},
        "timestamp_min": int(events["timestamp"].min()),
        "timestamp_max": int(events["timestamp"].max()),
        "timestamp_units": "relative seconds, already quantized to multiples of five",
        "availability_rule": "listen: timestamp+ceil(length*ratio/100)+5; other events: timestamp+5",
        "relations": relations, "splits": splits,
        "first_play_rule": "first timestamp then source order across all listens, then is_organic==0",
        "label_rule": "played_ratio_pct>=50",
        "label_eligibility_rule": "train/tune/select outcomes available strictly before the next split boundary; purge before day sampling",
        "purged_label_rows_by_split": queries.attrs["purged_label_rows_by_split"],
        "candidate_rule": "all logged first recommendation plays in each sampled user-day",
        "sampling_rule": "uniform users; complete available histories; uniform eligible days per user/split",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    return events, queries, manifest
