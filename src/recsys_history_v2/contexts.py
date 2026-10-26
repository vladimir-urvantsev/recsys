"""Bounded, reusable raw-event memories for each user-day query group.

All events remain in the prepared archive. This stage compresses the information
supplied to the neural model: recent events plus deterministically spaced older
events. Sampling depends on time/order only, never on candidate labels or on
whether an event had a favourable outcome.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys_research.data import DAY_SECONDS, sha256_file

EVENT_TYPES = {"listen": 1, "like": 2, "dislike": 3, "unlike": 4, "undislike": 5}
PAD_ID = -2
MISSING_ID = -1
EVENT_COLUMNS = [
    "uid", "item_id", "artist_id", "album_id", "timestamp", "source_order",
    "available_day", "event_type", "played_ratio_pct", "is_organic",
]
QUERY_COLUMNS = ["uid", "item_id", "artist_id", "album_id", "day", "group_id"]


def select_history_indices(eligible_indices: np.ndarray, recent: int, older: int) -> np.ndarray:
    """Select an ordered, duplicate-free recent tail plus spaced older memory.

    ``eligible_indices`` must already follow original event chronology. When
    older history is short it is retained in full. With at least two older
    slots linspace includes its earliest and latest event; a single older slot
    keeps the earliest event. Positions are fixed without consulting outcomes.
    """
    if recent < 0 or older < 0 or recent + older == 0:
        raise ValueError("Context recent/older sizes must be nonnegative with positive total")
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    n_recent = min(recent, len(eligible))
    prefix_length = len(eligible) - n_recent
    n_older = min(older, prefix_length)
    if n_older:
        positions = np.linspace(0, prefix_length - 1, num=n_older, dtype=np.int64)
        prefix = eligible[positions]
    else:
        prefix = eligible[:0]
    tail = eligible[len(eligible) - n_recent:] if n_recent else eligible[:0]
    return np.concatenate([prefix, tail])


def _query_fingerprint(queries: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for column in QUERY_COLUMNS:
        values = queries[column].to_numpy(dtype=np.int64)
        digest.update(column.encode())
        digest.update(values.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _groups(queries: pd.DataFrame):
    if queries.empty:
        raise ValueError("At least one query is required to prepare contexts")
    if queries[QUERY_COLUMNS].isna().any().any():
        raise ValueError("Query identity columns cannot contain null values")
    codes, group_ids = pd.factorize(queries["group_id"], sort=False)
    codes = codes.astype(np.int64, copy=False)
    n_groups = len(group_ids)
    first = np.full(n_groups, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, codes, np.arange(len(queries)))
    query_uid = queries["uid"].to_numpy(dtype=np.int64)
    query_day = queries["day"].to_numpy(dtype=np.int64)
    group_uid, group_day = query_uid[first], query_day[first]
    if not np.all(group_uid[codes] == query_uid) or not np.all(group_day[codes] == query_day):
        raise ValueError("Each group_id must identify exactly one user and day")
    identities = np.rec.fromarrays([group_uid, group_day], names="uid,day")
    if len(np.unique(identities)) != n_groups:
        raise ValueError("A user-day must have a single reusable group_id")
    return codes, np.asarray(group_ids, dtype=np.uint64), group_uid, group_day


def prepare_contexts(
    events: pd.DataFrame, queries: pd.DataFrame, config: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Write memory-mappable histories without consulting query outcomes.

    Input event eligibility is ``available_day < query.day - embargo_days``.
    Only after applying that filter are events ordered by original timestamp
    and source order. Delayed long plays are never admitted by assuming an
    original-timestamp prefix is already fully available.

    Valid events are left-padded. Raw item/artist/album IDs use -2 for padding
    and -1 for missing metadata; ``history_mask`` unambiguously identifies real
    tokens. Missing play ratios also use -1; raw replay ratios above100 remain.
    """
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    missing = set(EVENT_COLUMNS) - set(events.columns)
    missing_queries = set(QUERY_COLUMNS) - set(queries.columns)
    if missing or missing_queries:
        raise ValueError(f"Missing event/query fields: {sorted(missing)} / {sorted(missing_queries)}")
    recent, older = int(config.get("context_recent", 256)), int(config.get("context_older", 256))
    if recent < 0 or older < 0 or recent + older == 0:
        raise ValueError("Context recent/older sizes must be nonnegative with positive total")
    embargo = int(config.get("embargo_days", 1))
    if embargo < 0:
        raise ValueError("embargo_days must be nonnegative")
    codes, group_ids, group_uid, group_day = _groups(queries)
    n_groups, length = len(group_ids), recent + older
    shapes = {
        "history_ids.npy": ((n_groups, length, 3), "int32", PAD_ID),
        "history_type.npy": ((n_groups, length), "uint8", 0),
        "history_ratio.npy": ((n_groups, length), "int16", -1),
        "history_organic.npy": ((n_groups, length), "uint8", 0),
        "history_age.npy": ((n_groups, length), "float32", 0),
        "history_mask.npy": ((n_groups, length), "bool", False),
        "query_group.npy": ((len(queries),), "int64", 0),
        "group_uid.npy": ((n_groups,), "int64", 0),
        "group_day.npy": ((n_groups,), "int32", 0),
        "group_id.npy": ((n_groups,), "uint64", 0),
        "group_available_count.npy": ((n_groups,), "uint32", 0),
        "group_retained_count.npy": ((n_groups,), "uint32", 0),
    }
    existing = [name for name in [*shapes, "context_manifest.json"] if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Context outputs already exist: {existing}; use a fresh directory")

    uid = events["uid"].to_numpy(dtype=np.int64)
    timestamp = events["timestamp"].to_numpy(dtype=np.int64)
    source_order = events["source_order"].to_numpy(dtype=np.int64)
    available_day = events["available_day"].to_numpy(dtype=np.int64)
    raw_ids = events[["item_id", "artist_id", "album_id"]].to_numpy(dtype=np.int64)
    if raw_ids.size and (raw_ids.min() < MISSING_ID or raw_ids.max() > np.iinfo(np.int32).max):
        raise ValueError("Raw IDs must fit signed int32 with -1 reserved for missing metadata")
    raw_ids = raw_ids.astype(np.int32)
    ratio = events["played_ratio_pct"].to_numpy(dtype=np.int64, na_value=-1)
    if ratio.size and (ratio.min() < -1 or ratio.max() > np.iinfo(np.int16).max):
        raise ValueError("Raw ratios must fit int16 without clipping or overflow")
    ratio = ratio.astype(np.int16)
    organic = events["is_organic"].to_numpy(dtype=np.int64)
    if not np.isin(organic, [0, 1]).all():
        raise ValueError("is_organic must contain only 0 and 1")
    organic = organic.astype(np.uint8)
    # Mapping the five categorical values avoids expanding tens of millions of
    # event-type strings into an unnecessary object/string array.
    mapped_types = events["event_type"].map(EVENT_TYPES)
    if mapped_types.isna().any():
        raise ValueError("Unknown event type in history")
    event_type = mapped_types.to_numpy(dtype=np.uint8)
    if np.any((timestamp < 0) | (available_day < 0)):
        raise ValueError("Event times must be nonnegative relative times")
    # One global numeric sort replaces repeated DataFrame sorting per query.
    chronological = np.lexsort((source_order, timestamp, uid))
    sorted_uid = uid[chronological]
    owners, first, counts = np.unique(sorted_uid, return_index=True, return_counts=True)
    user_slices = {int(user): (int(start), int(start + count))
                   for user, start, count in zip(owners, first, counts)}

    arrays: dict[str, np.memmap] = {}
    for name, (shape, dtype, fill) in shapes.items():
        array = np.lib.format.open_memmap(output / (name + ".partial"), mode="w+", dtype=dtype, shape=shape)
        array[...] = fill
        arrays[name] = array
    arrays["query_group.npy"][:] = codes
    arrays["group_uid.npy"][:] = group_uid
    arrays["group_day.npy"][:] = group_day
    arrays["group_id.npy"][:] = group_ids
    by_user = np.argsort(group_uid, kind="stable")
    group_owners, group_first, group_counts = np.unique(group_uid[by_user], return_index=True, return_counts=True)
    for user, start, count in zip(group_owners, group_first, group_counts):
        interval = user_slices.get(int(user))
        user_rows = chronological[interval[0]:interval[1]] if interval else np.empty(0, dtype=np.int64)
        availability = available_day[user_rows]
        for group in by_user[start:start + count]:
            cutoff = int(group_day[group]) - embargo
            eligible = user_rows[availability < cutoff]
            chosen = select_history_indices(eligible, recent, older)
            n_selected = len(chosen)
            arrays["group_available_count.npy"][group] = len(eligible)
            arrays["group_retained_count.npy"][group] = n_selected
            if not n_selected:
                continue
            tail = slice(length - n_selected, length)
            arrays["history_ids.npy"][group, tail] = raw_ids[chosen]
            arrays["history_type.npy"][group, tail] = event_type[chosen]
            arrays["history_ratio.npy"][group, tail] = ratio[chosen]
            arrays["history_organic.npy"][group, tail] = organic[chosen]
            elapsed = (cutoff * DAY_SECONDS - timestamp[chosen]) / DAY_SECONDS
            if np.any(elapsed < 0):
                raise ValueError("Available history cannot have an original timestamp after its cutoff")
            arrays["history_age.npy"][group, tail] = np.log1p(elapsed).astype(np.float32)
            arrays["history_mask.npy"][group, tail] = True

    available_counts = np.asarray(arrays["group_available_count.npy"]).copy()
    retained_counts = np.asarray(arrays["group_retained_count.npy"]).copy()
    observed = available_counts > 0
    coverage = retained_counts[observed] / available_counts[observed]
    for array in arrays.values():
        array.flush()
    arrays.clear()
    artifacts = {}
    for name, (shape, dtype, _) in shapes.items():
        (output / (name + ".partial")).replace(output / name)
        artifacts[name] = {"shape": list(shape), "dtype": dtype, "bytes": (output / name).stat().st_size,
                           "sha256": sha256_file(output / name)}
    quantiles = [0, .1, .5, .9, 1]
    manifest = {
        "groups": n_groups, "queries": len(queries), "context_length": length,
        "recent_capacity": recent, "older_capacity": older, "embargo_days": embargo,
        "query_identity_sha256": _query_fingerprint(queries),
        "contexts_module_sha256": sha256_file(Path(__file__)),
        "event_type_mapping": EVENT_TYPES,
        "pad_id": PAD_ID, "missing_id": MISSING_ID,
        "empty_groups": int((~observed).sum()),
        "available_events_quantiles": dict(zip(map(str, quantiles), np.quantile(available_counts, quantiles).tolist())),
        "retained_events_quantiles": dict(zip(map(str, quantiles), np.quantile(retained_counts, quantiles).tolist())),
        "coverage_nonempty_quantiles": dict(zip(map(str, quantiles), np.quantile(coverage, quantiles).tolist())) if len(coverage) else None,
        "total_available_token_occurrences": int(available_counts.sum()),
        "total_retained_token_occurrences": int(retained_counts.sum()),
        "query_labels_accessed": False,
        "rules": {
            "eligibility": "available_day < query.day - embargo_days",
            "event_order": "original timestamp, then source_order, within user",
            "selection": "most recent R eligible events, plus up to O evenly spaced older chronological indices; no duplication",
            "padding": "left pad; True mask marks real tokens; raw missing metadata -1, padding -2",
            "age": "log1p((cutoff_day*86400-original_timestamp)/86400)",
            "sharing": "one context per user-day group, reused for all candidate queries",
            "information_limit": "complete histories stay in archive; neural input is a bounded compression, while AutoFE can aggregate the full eligible history",
            "sealing_limit": "earlier confirmation outcomes can enter later histories after availability/embargo, as declared rolling evaluation",
        },
        "artifacts": artifacts, "elapsed_seconds": time.perf_counter() - started,
    }
    temporary = output / "context_manifest.json.partial"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "context_manifest.json")
    return manifest
