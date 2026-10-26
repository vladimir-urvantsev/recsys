from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recsys_research.data import (
    DAY_SECONDS,
    add_availability,
    assign_splits,
    choose_users,
    make_queries,
    read_relation_mapping,
    read_user_histories,
)


def _events(rows):
    records = []
    for n, (uid, item, day, seconds, ratio, organic, kind) in enumerate(rows):
        records.append({
            "uid": uid, "item_id": item, "day": day,
            "timestamp": day * DAY_SECONDS + seconds,
            "played_ratio_pct": ratio, "track_length_seconds": 200 if kind == "listen" else None,
            "is_organic": organic, "event_type": kind,
            "source_order": n, "artist_id": item + 100, "album_id": item + 1000,
        })
    return add_availability(pd.DataFrame(records))


def _config(**extra):
    return {
        "warmup_days": 1, "split_days": [10, 20, 30], "day_seed": 13,
        "days_per_user": {"train": 20, "tune": 20, "select": 20, "test": 20},
        **extra,
    }


def test_user_sample_is_deterministic_unique_and_not_activity_weighted():
    users = np.repeat(np.arange(20), np.arange(1, 21))
    actual = choose_users(users, 7, 2026)
    np.testing.assert_array_equal(actual, choose_users(np.arange(20), 7, 2026))
    assert len(actual) == len(np.unique(actual)) == 7
    with pytest.raises(ValueError):
        choose_users(users, 21, 2026)


def test_streaming_filter_preserves_every_event_of_selected_users(tmp_path):
    table = pa.table({
        "uid": pa.array([10, 20], type=pa.uint32()),
        "timestamp": pa.array([[5, 10, 15], [5, 8]], type=pa.large_list(pa.uint32())),
        "item_id": pa.array([[1, 2, 1], [2, 3]], type=pa.large_list(pa.uint32())),
        "is_organic": pa.array([[0, 1, 0], [0, 0]], type=pa.large_list(pa.uint8())),
        "played_ratio_pct": pa.array([[100, None, 10], [80, 20]], type=pa.large_list(pa.uint16())),
        "track_length_seconds": pa.array([[100, None, 200], [100, 200]], type=pa.large_list(pa.uint32())),
        "event_type": pa.array([["listen", "like", "listen"], ["listen", "listen"]]),
    })
    path = tmp_path / "source.parquet"
    pq.write_table(table, path)
    frame = read_user_histories(path, np.array([10], dtype=np.uint32))
    assert frame["uid"].tolist() == [10, 10, 10]
    assert frame["timestamp"].tolist() == [5, 10, 15]
    assert frame["event_type"].tolist() == ["listen", "like", "listen"]
    assert pd.isna(frame.loc[1, "played_ratio_pct"])


def test_available_at_uses_uncapped_replay_time_and_midnight_boundary():
    events = _events([
        (1, 1, 1, DAY_SECONDS - 30, 150, 0, "listen"),
        (1, 2, 1, 10, None, 0, "like"),
    ])
    actual = add_availability(events)
    assert actual["available_at"].tolist() == [2 * DAY_SECONDS + 275, DAY_SECONDS + 15]
    assert actual["day"].tolist() == [1, 1]
    assert actual["available_day"].tolist() == [2, 1]


def test_first_play_is_selected_before_organic_filter_and_threshold_is_inclusive():
    events = _events([
        (1, 1, 2, 10, 100, 1, "listen"),  # Earlier organic blocks later recommendation.
        (1, 1, 2, 20, 10, 0, "listen"),
        (1, 2, 2, 10, 50, 0, "listen"),
        (1, 2, 2, 20, 0, 0, "listen"),  # Repeated same-day play never changes label.
        (1, 3, 2, 10, 49, 0, "listen"),
        (1, 4, 2, 10, None, 0, "like"),
    ])
    queries = make_queries(events, _config())
    assert queries["item_id"].tolist() == [2, 3]
    assert queries["label"].tolist() == [1, 0]
    assert queries["group_id"].nunique() == 1


def test_split_boundaries_keep_global_time_disjoint():
    days = pd.Series([0, 1, 9, 10, 19, 20, 29, 30, 300])
    assert assign_splits(days, _config()).tolist() == [
        "warmup", "train", "train", "tune", "tune", "select", "select", "test", "test"
    ]
    with pytest.raises(ValueError):
        assign_splits(days, _config(split_days=[20, 10, 30]))


def test_day_sample_ignores_outcomes_and_keeps_complete_candidate_groups():
    events = _events([
        (uid, item, day, 10, 20 if item == 1 else 90, 0, "listen")
        for uid in [1, 2] for day in range(1, 40) for item in [1, 2, 3]
    ])
    config = _config(days_per_user={name: 2 for name in ["train", "tune", "select", "test"]})
    left = make_queries(events, config)
    events["played_ratio_pct"] = 100 - events["played_ratio_pct"]
    right = make_queries(events, config)
    assert left[["uid", "day", "item_id"]].equals(right[["uid", "day", "item_id"]])
    assert left.groupby("group_id").size().eq(3).all()
    assert left.groupby(["uid", "split"], observed=True)["day"].nunique().eq(2).all()
    # An unrelated user must not change another user's chosen days or group IDs.
    one = make_queries(events.loc[events.uid.eq(1)], config)
    assert one[["group_id", "item_id"]].reset_index(drop=True).equals(
        right.loc[right.uid.eq(1), ["group_id", "item_id"]].reset_index(drop=True)
    )


def test_mapping_reduction_is_deterministic_smallest_id(tmp_path):
    path = tmp_path / "mapping.parquet"
    pq.write_table(pa.table({
        "item_id": pa.array([1, 1, 1, 2, 3], type=pa.uint32()),
        "artist_id": pa.array([5, 2, 5, 8, 9], type=pa.uint32()),
    }), path)
    mapping, manifest = read_relation_mapping(path, np.array([1, 2]), "artist_id")
    assert mapping.to_dict() == {1: 2, 2: 8}
    assert manifest["sample_items_with_multiple_values"] == 1
    assert manifest["max_values_per_sample_item"] == 2


def test_labels_finishing_in_later_stages_are_purged_before_day_sampling():
    events = _events([
        (1, 1, 9, DAY_SECONDS - 200, 100, 0, "listen"),
        (1, 2, 9, DAY_SECONDS - 400, 150, 0, "listen"),
        (1, 3, 19, DAY_SECONDS - 100, 100, 0, "listen"),
        (1, 4, 29, DAY_SECONDS - 100, 200, 0, "listen"),
        (1, 5, 30, DAY_SECONDS - 100, 200, 0, "listen"),
    ])
    queries = make_queries(events, _config())
    assert queries["item_id"].tolist() == [2, 5]
    assert queries.attrs["purged_label_rows_by_split"] == {
        "train": 1, "tune": 1, "select": 1, "test": 0,
    }


def test_replay_outcome_at_exact_next_stage_boundary_is_not_available():
    # A 200-second track replayed for 150% takes 300s, plus the 5s guard.
    events = _events([
        (1, 1, 9, DAY_SECONDS - 305, 150, 0, "listen"),
        (1, 2, 9, DAY_SECONDS - 306, 150, 0, "listen"),
    ])
    queries = make_queries(events, _config())
    assert queries["item_id"].tolist() == [2]
    assert queries.attrs["purged_label_rows_by_split"]["train"] == 1


def test_all_late_labels_return_empty_queries_with_purge_diagnostics():
    events = _events([(1, 1, 9, DAY_SECONDS - 10, 150, 0, "listen")])
    queries = make_queries(events, _config())
    assert queries.empty
    assert queries.attrs["purged_label_rows_by_split"]["train"] == 1
