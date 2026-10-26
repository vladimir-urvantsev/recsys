"""Compare optimized aggregates with an independent raw-event oracle."""

import numpy as np
import pandas as pd

from recsys_research.features import RELATIONS, build_features


def _events():
    rng = np.random.default_rng(815)
    size = 90
    items = rng.choice([10, 20, 30], size)
    events = pd.DataFrame({
        "uid": rng.choice([1, 2, 3], size),
        "item_id": items,
        "artist_id": np.where(items == 30, -1, items // 10),
        "album_id": np.where(items == 20, -1, items // 10 + 5),
        "available_day": rng.integers(0, 45, size),
        "event_type": rng.choice(["listen", "like", "dislike", "unlike"], size),
        "played_ratio_pct": rng.choice([0, 49, 50, 99, 100, 175], size),
        "is_organic": rng.integers(0, 2, size),
    })
    events.loc[events.event_type.ne("listen"), "played_ratio_pct"] = np.nan
    queries = pd.DataFrame({
        "uid": [1, 2, 1, 3, 99], "item_id": [10, 20, 30, 10, 99],
        "artist_id": [1, 2, -1, 1, 99], "album_id": [6, -1, 8, 6, 99],
        "day": [1, 8, 22, 40, 50],
    })
    return events, queries


def _summary(raw, weights=None):
    """Direct row filtering and summation, with no cumulative-index operations."""
    listen = raw.event_type.eq("listen").to_numpy()
    ratio = raw.played_ratio_pct.fillna(0).to_numpy()
    w = np.ones(len(raw)) if weights is None else np.asarray(weights)
    return {
        "n": float(w[listen].sum()),
        "positive": float(w[listen & (ratio >= 50)].sum()),
        "ratio": float((w[listen] * np.clip(ratio[listen], 0, 100) / 100).sum()),
        "organic": float((w[listen] * raw.is_organic.to_numpy()[listen]).sum()),
        "like": float(w[raw.event_type.eq("like")].sum()),
        "dislike": float(w[raw.event_type.eq("dislike")].sum()),
    }


def test_every_relation_matches_raw_event_oracle():
    events, queries = _events()
    cfg = {"embargo_days": 1, "manual_windows": [7, 30, None],
           "auto_windows": [3, 10], "auto_half_lives": [3, 10]}
    matrix, groups, _ = build_features(events, queries, cfg)
    assert len(set(groups["manual"])) == len(groups["manual"])
    for row_number, query in queries.iterrows():
        cutoff = query.day - 1
        past = events.loc[events.available_day.lt(cutoff)]
        global_summary = _summary(past)
        prior = (global_summary["positive"] + 1) / (global_summary["n"] + 2)
        np.testing.assert_allclose(matrix.loc[row_number, "global_positive_prior"], prior, rtol=1e-6)
        for relation, columns in RELATIONS.items():
            matching = np.ones(len(past), dtype=bool)
            for column in columns:
                matching &= past[column].eq(query[column]).to_numpy()
                if column in ("artist_id", "album_id") and query[column] < 0:
                    matching[:] = False
            history = past.loc[matching]
            for window in [None, 3, 7, 10, 30]:
                selected = history if window is None else history.loc[history.available_day.ge(cutoff - window)]
                stats = _summary(selected)
                suffix = "all" if window is None else f"w{window}"
                np.testing.assert_allclose(matrix.loc[row_number, f"{relation}__count__{suffix}"],
                                           np.log1p(stats["n"]), rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(matrix.loc[row_number, f"{relation}__rate__{suffix}"],
                                           (stats["positive"] + 5 * prior) / (stats["n"] + 5), rtol=1e-6)
            for half_life in [3, 10]:
                weights = 2.0 ** ((history.available_day.to_numpy() - cutoff) / half_life)
                stats = _summary(history, weights)
                np.testing.assert_allclose(matrix.loc[row_number, f"{relation}__count__h{half_life}"],
                                           np.log1p(stats["n"]), rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(matrix.loc[row_number, f"{relation}__rate__h{half_life}"],
                                           (stats["positive"] + 5 * prior) / (stats["n"] + 5), rtol=1e-6)
            stats = _summary(history)
            expected_recency = min(query.day - history.available_day.max(), 999) if len(history) else 999
            np.testing.assert_allclose(matrix.loc[row_number, f"{relation}__recency"], expected_recency)
            np.testing.assert_allclose(matrix.loc[row_number, f"{relation}__mean_ratio"],
                                       (stats["ratio"] + 5 * prior) / (stats["n"] + 5), rtol=1e-6)
            np.testing.assert_allclose(matrix.loc[row_number, f"{relation}__organic_share"],
                                       (stats["organic"] + 1) / (stats["n"] + 2), rtol=1e-6)
            for event in ["like", "dislike"]:
                np.testing.assert_allclose(matrix.loc[row_number, f"{relation}__{event}s"],
                                           np.log1p(stats[event]), rtol=1e-6, atol=1e-7)


def test_query_outcome_source_and_duration_are_not_inputs():
    events, queries = _events()
    cfg = {"auto_windows": [3], "auto_half_lives": [3]}
    first, _, _ = build_features(events, queries, cfg)
    perturbed = queries.copy()
    perturbed["played_ratio_pct"] = [0, 100, 150, 50, 49]
    perturbed["track_length_seconds"] = [900, 10, 30, 100, 450]
    perturbed["label"] = [0, 1, 1, 1, 0]
    perturbed["is_organic"] = [0, 1, 0, 1, 0]
    second, _, _ = build_features(events, perturbed, cfg)
    pd.testing.assert_frame_equal(first, second)


def test_input_row_order_does_not_change_features():
    events, queries = _events()
    cfg = {"auto_windows": [3], "auto_half_lives": [3]}
    ordered, _, _ = build_features(events, queries, cfg)
    shuffled, _, _ = build_features(events.sample(frac=1, random_state=77), queries, cfg)
    pd.testing.assert_frame_equal(ordered, shuffled)


def test_distant_future_events_and_query_batch_cannot_rescale_past_decay():
    events, queries = _events()
    query = queries.iloc[[3]].copy()
    cfg = {"auto_windows": [3], "auto_half_lives": [0.5, 3, 45]}
    original, _, _ = build_features(events, query, cfg)
    distant_event = events.iloc[[0]].copy()
    distant_event["available_day"] = 100_000
    distant_event["played_ratio_pct"] = 100
    expanded = pd.concat([events, distant_event], ignore_index=True)
    distant_query = query.copy()
    distant_query["day"] = 100_002
    batched_queries = pd.concat([query, distant_query], ignore_index=True)
    batched, _, _ = build_features(expanded, batched_queries, cfg)
    pd.testing.assert_frame_equal(original.reset_index(drop=True), batched.iloc[[0]].reset_index(drop=True))
