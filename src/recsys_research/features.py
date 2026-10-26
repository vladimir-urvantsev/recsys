"""Strict-cutoff daily relational aggregates with an explicit feature grammar.

All event outcomes enter history only on ``available_day``; a query on day d
uses available_day < d - embargo_days. No current outcome is an input.
The engine computes each relationship once; window endpoints are prefix lookups.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

DAY = 86400
ID_BASE = 1_000_000_000
RELATIONS = {
    "user": ("uid",), "item": ("item_id",), "artist": ("artist_id",),
    "album": ("album_id",), "user_item": ("uid", "item_id"),
    "user_artist": ("uid", "artist_id"), "user_album": ("uid", "album_id"),
}
NUMERIC_STATS = ["n", "positive", "ratio", "organic", "like", "dislike"]


def relation_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    arrays = [frame[c].to_numpy(dtype=np.int64) + 1 for c in columns]
    if any(np.any((a < 0) | (a >= ID_BASE)) for a in arrays):
        raise ValueError("Relationship ID outside supported [-1, 1e9-2] range")
    return arrays[0] if len(arrays) == 1 else arrays[0] * ID_BASE + arrays[1]


@dataclass
class PrefixIndex:
    keys: np.ndarray
    days: np.ndarray
    values: np.ndarray
    search: np.ndarray

    @classmethod
    def from_aggregates(cls, daily: pd.DataFrame, weighted: np.ndarray | None = None):
        keys = daily["key"].to_numpy(np.int64)
        days = daily["available_day"].to_numpy(np.int64)
        raw = daily[NUMERIC_STATS].to_numpy(np.float64) if weighted is None else weighted
        # Group cumsums avoid cancellation across large unrelated entity groups.
        cumulative = pd.DataFrame(raw).groupby(keys, sort=False).cumsum().to_numpy()
        search = np.rec.fromarrays([keys, days], names="key,day")
        return cls(keys, days, cumulative, search)

    def lookup(self, keys: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.keys):
            return np.zeros((len(keys), len(NUMERIC_STATS)), np.float64), np.full(len(keys), -10000, np.int64)
        target = np.rec.fromarrays([keys.astype(np.int64), days.astype(np.int64)], names="key,day")
        positions = np.searchsorted(self.search, target, side="right") - 1
        valid = positions >= 0
        safe = np.maximum(positions, 0)
        valid &= self.keys[safe] == keys
        values = np.zeros((len(keys), len(NUMERIC_STATS)), np.float64)
        last = np.full(len(keys), -10000, np.int64)
        values[valid] = self.values[safe[valid]]
        last[valid] = self.days[safe[valid]]
        return values, last


def daily_events(events: pd.DataFrame) -> pd.DataFrame:
    out = events[["uid", "item_id", "artist_id", "album_id", "available_day"]].copy()
    listen = events.event_type.astype(str).eq("listen").to_numpy()
    ratios = events.played_ratio_pct.fillna(0).to_numpy(np.float64)
    out["n"] = listen.astype(np.float64)
    out["positive"] = (listen & (ratios >= 50)).astype(np.float64)
    out["ratio"] = np.where(listen, np.minimum(ratios, 100) / 100, 0)
    out["organic"] = listen * events.is_organic.to_numpy(np.float64)
    out["like"] = events.event_type.astype(str).eq("like").to_numpy(np.float64)
    out["dislike"] = events.event_type.astype(str).eq("dislike").to_numpy(np.float64)
    return out


def build_features(events: pd.DataFrame, queries: pd.DataFrame, config: dict):
    """Return numeric/categorical matrix, feature groups and timing metadata.

    Missing artist/album (-1) is *not* a shared relation: its aggregate is zero.
    This prevents unrelated tracks with missing metadata acquiring affinity.
    """
    start = time.perf_counter()
    q = queries.reset_index(drop=True)
    cutoff = q.day.to_numpy(np.int64) - config.get("embargo_days", 1)
    features: dict[str, np.ndarray] = {}
    groups: dict[str, list[str]] = {"manual": [], "static": []}
    for col in ("uid", "item_id", "artist_id", "album_id"):
        name = "cat_" + col
        features[name] = q[col].to_numpy(np.int64).astype(str)
        groups["manual"].append(name)
        groups["static"].append(name)
    base = daily_events(events)
    global_daily = base.groupby("available_day", as_index=False)[NUMERIC_STATS].sum()
    global_daily["key"] = 0
    global_index = PrefixIndex.from_aggregates(global_daily)
    global_values, _ = global_index.lookup(np.zeros(len(q), np.int64), cutoff - 1)
    prior = (global_values[:, 1] + 1) / (global_values[:, 0] + 2)
    features["global_positive_prior"] = prior.astype(np.float32)
    groups["manual"].append("global_positive_prior")
    groups["static"].append("global_positive_prior")
    relation_timings = {}
    relation_values = {}

    def add(name, value, group):
        features[name] = np.asarray(value, dtype=np.float32)
        groups.setdefault(group, []).append(name)

    def count_rate(values, rel, suffix, group):
        add(f"{rel}__count__{suffix}", np.log1p(np.maximum(values[:, 0], 0)), group)
        add(f"{rel}__rate__{suffix}", (values[:, 1] + 5 * prior) / (values[:, 0] + 5), group)

    for rel, cols in RELATIONS.items():
        t0 = time.perf_counter()
        keys = relation_keys(base, cols)
        work = base[["available_day"] + NUMERIC_STATS].copy()
        work["key"] = keys
        # Exclude absent metadata from aggregate state.
        valid = np.ones(len(base), bool)
        for col in cols:
            if col in ("artist_id", "album_id"):
                valid &= base[col].to_numpy() >= 0
        daily = work.loc[valid].groupby(["key", "available_day"], as_index=False, sort=True)[NUMERIC_STATS].sum()
        index = PrefixIndex.from_aggregates(daily)
        qkeys = relation_keys(q, cols)
        all_values, last = index.lookup(qkeys, cutoff - 1)
        for window in config.get("manual_windows", [7, 30, None]):
            suffix = "all" if window is None else f"w{window}"
            values = all_values
            if window is not None:
                before, _ = index.lookup(qkeys, cutoff - window - 1)
                values = np.maximum(all_values - before, 0)
            count_rate(values, rel, suffix, "manual")
            relation_values[(rel, suffix)] = values[:, 0]
        add(f"{rel}__recency", np.minimum(q.day.to_numpy() - last, 999), "manual")
        add(f"{rel}__mean_ratio", (all_values[:, 2] + 5 * prior) / (all_values[:, 0] + 5), "manual")
        add(f"{rel}__organic_share", (all_values[:, 3] + 1) / (all_values[:, 0] + 2), "manual")
        add(f"{rel}__likes", np.log1p(all_values[:, 4]), "manual")
        add(f"{rel}__dislikes", np.log1p(all_values[:, 5]), "manual")
        for window in config.get("auto_windows", [2, 3, 10, 14, 60, 90]):
            before, _ = index.lookup(qkeys, cutoff - window - 1)
            values = np.maximum(all_values - before, 0)
            count_rate(values, rel, f"w{window}", f"window_{window}")
        raw = daily[NUMERIC_STATS].to_numpy(np.float64)
        for half_life in config.get("auto_half_lives", [3, 10, 45]):
            rate = np.log(2) / half_life
            # Stable recurrence: S_j = raw_j + exp(-delta/h) S_(j-1).
            # Reset at each entity boundary. A compiled bidiagonal solve avoids
            # Python loops and an exponent origin depending on future records.
            if len(daily):
                delta = np.r_[0, np.diff(index.days)]
                same = np.r_[False, index.keys[1:] == index.keys[:-1]]
                factor = np.zeros(len(daily))
                factor[same] = np.exp(-delta[same] * rate)
                band = np.zeros((2, len(daily)))
                band[0] = 1
                band[1, :-1] = -factor[1:]
                state = solve_banded((1, 0), band, raw, overwrite_ab=True, check_finite=False)
                decay_index = PrefixIndex(index.keys, index.days, state, index.search)
            else:
                decay_index = index
            weighted, last_day = decay_index.lookup(qkeys, cutoff - 1)
            values = weighted * np.exp(-np.maximum(cutoff - last_day, 0) * rate)[:, None]
            count_rate(values, rel, f"h{half_life}", f"decay_{half_life}")
        relation_timings[rel] = time.perf_counter() - t0
        print(f"features {rel}: {len(daily):,} daily keys, {relation_timings[rel]:.1f}s", flush=True)

    for rel in ("user_item", "user_artist", "user_album"):
        for suffix in ("w7", "w30", "all"):
            if (rel, suffix) in relation_values:
                add(f"{rel}__user_share__{suffix}", relation_values[(rel, suffix)] / (relation_values[("user", suffix)] + 1), "manual")
    for rel in RELATIONS:
        if (rel, "w7") in relation_values and (rel, "w30") in relation_values:
            add(f"{rel}__trend7_30", (relation_values[(rel, "w7")] + 1) / (relation_values[(rel, "w30")] + 4), "manual")
    matrix = pd.DataFrame(features)
    for col in matrix.select_dtypes(include="number"):
        if not np.isfinite(matrix[col].to_numpy()).all():
            raise ValueError(f"Non-finite feature {col}")
    return matrix, groups, {"seconds": time.perf_counter() - start, "relations_seconds": relation_timings,
                             "n_features": len(matrix.columns), "n_rows": len(matrix),
                             "matrix_bytes": int(matrix.memory_usage(deep=True).sum()),
                             "embargo_days": config.get("embargo_days", 1)}
