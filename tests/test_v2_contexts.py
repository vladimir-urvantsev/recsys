from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys_history_v2.contexts import prepare_contexts, select_history_indices


def events(rows):
    return pd.DataFrame([
        {"uid": uid, "item_id": item, "artist_id": item + 100, "album_id": item + 1000,
         "timestamp": timestamp, "source_order": order, "available_day": available,
         "event_type": kind, "played_ratio_pct": ratio, "is_organic": organic}
        for order, (uid, item, timestamp, available, ratio, kind, organic) in enumerate(rows)
    ])


def queries(rows):
    return pd.DataFrame([
        {"uid": uid, "item_id": item, "artist_id": item + 100, "album_id": item + 1000,
         "day": day, "group_id": group, "label": label}
        for uid, item, day, group, label in rows
    ])


def load(output, name):
    return np.load(output / (name + ".npy"), mmap_mode="r")


def test_available_filter_before_chronological_tail_including_delay_boundary(tmp_path):
    frame = events([(1, 10, 10, 1, 100, "listen", 0),
                    (1, 11, 20, 4, 150, "listen", 1),  # Earlier time, delayed outcome.
                    (1, 12, 30, 2, 49, "listen", 0),
                    (1, 13, 40, 3, 100, "listen", 0)])  # Exact excluded availability day.
    q = queries([(1, 90, 4, 8, np.nan)])
    manifest = prepare_contexts(frame, q, {"context_recent": 3, "context_older": 0, "embargo_days": 1}, tmp_path)
    assert load(tmp_path, "history_ids")[0, :, 0].tolist() == [-2, 10, 12]
    assert load(tmp_path, "history_mask")[0].tolist() == [False, True, True]
    assert manifest["available_events_quantiles"]["1"] == 2
    np.testing.assert_allclose(load(tmp_path, "history_age")[0, -1], np.log1p(3 - 30 / 86400), rtol=1e-6)


def test_old_long_range_memory_and_recent_tail_are_distinct_and_ordered(tmp_path):
    eligible = np.arange(12)
    assert select_history_indices(eligible, 3, 4).tolist() == [0, 2, 5, 8, 9, 10, 11]
    frame = events([(1, item, item * 10, 1, 100 + item, "listen", 0) for item in range(12)])
    q = queries([(1, 100, 4, 8, np.nan)])
    m = prepare_contexts(frame, q, {"context_recent": 3, "context_older": 4}, tmp_path)
    assert load(tmp_path, "history_ids")[0, :, 0].tolist() == [0, 2, 5, 8, 9, 10, 11]
    assert load(tmp_path, "history_ratio")[0].tolist() == [100, 102, 105, 108, 109, 110, 111]
    assert m["total_available_token_occurrences"] == 12
    assert m["total_retained_token_occurrences"] == 7


def test_empty_history_is_fully_left_padded_and_groups_reuse_same_context(tmp_path):
    frame = events([(1, 10, 10, 3, None, "like", 1)])
    q = queries([(1, 90, 2, 8, np.nan), (1, 91, 2, 8, 1), (2, 92, 2, 9, 0)])
    m = prepare_contexts(frame, q, {"context_recent": 2, "context_older": 2}, tmp_path)
    assert load(tmp_path, "query_group").tolist() == [0, 0, 1]
    assert load(tmp_path, "group_uid").tolist() == [1, 2]
    assert load(tmp_path, "history_ids").shape == (2, 4, 3)
    assert (load(tmp_path, "history_ids") == -2).all()
    assert not load(tmp_path, "history_mask").any()
    assert m["empty_groups"] == 2


def test_query_outcomes_and_current_metadata_do_not_change_history(tmp_path):
    frame = events([(1, 10, 10, 1, None, "like", 1), (1, 11, 20, 1, 150, "listen", 0)])
    left_q = queries([(1, 90, 4, 8, np.nan), (1, 91, 4, 8, 1)])
    right_q = left_q.copy()
    right_q["label"] = [0, 0]
    right_q["played_ratio_pct"] = [50_000, 0]
    right_q["track_length_seconds"] = 900_000
    right_q["timestamp"] = 10**12
    cfg = {"context_recent": 3, "context_older": 1}
    prepare_contexts(frame, left_q, cfg, tmp_path / "left")
    prepare_contexts(frame, right_q, cfg, tmp_path / "right")
    for name in ["history_ids", "history_type", "history_ratio", "history_organic", "history_age", "history_mask", "query_group"]:
        np.testing.assert_array_equal(load(tmp_path / "left", name), load(tmp_path / "right", name))
    assert load(tmp_path / "left", "history_type")[0, -2:].tolist() == [2, 1]
    assert load(tmp_path / "left", "history_ratio")[0, -2:].tolist() == [-1, 150]


def test_future_event_mutation_and_source_row_permutation_preserve_past_context(tmp_path):
    frame = events([(1, 10, 10, 1, 20, "listen", 0), (1, 11, 20, 2, 80, "listen", 0),
                    (1, 12, 30, 50, 150, "listen", 1), (2, 15, 10, 1, 50, "listen", 0)])
    q = queries([(1, 90, 4, 8, np.nan), (2, 91, 4, 9, np.nan)])
    cfg = {"context_recent": 2, "context_older": 1}
    prepare_contexts(frame, q, cfg, tmp_path / "left")
    changed = frame.copy()
    changed.loc[2, ["item_id", "artist_id", "album_id", "played_ratio_pct"]] = [1000, 1001, 1002, 5000]
    changed = changed.sample(frac=1, random_state=12)
    prepare_contexts(changed, q, cfg, tmp_path / "right")
    for name in ["history_ids", "history_type", "history_ratio", "history_organic", "history_age", "history_mask"]:
        np.testing.assert_array_equal(load(tmp_path / "left", name), load(tmp_path / "right", name))


def test_missing_metadata_and_replay_ratio_keep_distinct_padding_semantics(tmp_path):
    frame = events([(1, 10, 10, 1, 159, "listen", 1)])
    frame["artist_id"] = -1
    q = queries([(1, 90, 4, 8, np.nan)])
    prepare_contexts(frame, q, {"context_recent": 3, "context_older": 0}, tmp_path)
    assert load(tmp_path, "history_ids")[0, :, 1].tolist() == [-2, -2, -1]
    assert load(tmp_path, "history_ratio")[0].tolist() == [-1, -1, 159]
    assert load(tmp_path, "history_mask")[0].tolist() == [False, False, True]


def test_invalid_group_ownership_or_duplicate_userday_groups_are_rejected(tmp_path):
    frame = events([(1, 10, 10, 1, 100, "listen", 0)])
    q = queries([(1, 90, 4, 8, np.nan), (2, 91, 4, 8, np.nan)])
    with pytest.raises(ValueError, match="exactly one"):
        prepare_contexts(frame, q, {}, tmp_path / "a")
    q = queries([(1, 90, 4, 8, np.nan), (1, 91, 4, 9, np.nan)])
    with pytest.raises(ValueError, match="single reusable"):
        prepare_contexts(frame, q, {}, tmp_path / "b")


def test_no_silent_ratio_overflow_and_no_context_overwrite(tmp_path):
    frame = events([(1, 10, 10, 1, 40000, "listen", 0)])
    q = queries([(1, 90, 4, 8, np.nan)])
    with pytest.raises(ValueError, match="int16"):
        prepare_contexts(frame, q, {}, tmp_path)
    frame["played_ratio_pct"] = 100
    prepare_contexts(frame, q, {"context_recent": 1, "context_older": 0}, tmp_path)
    with pytest.raises(FileExistsError):
        prepare_contexts(frame, q, {}, tmp_path)


def test_selection_helpers_cover_pure_older_and_short_histories():
    assert select_history_indices(np.arange(4), 0, 3).tolist() == [0, 1, 3]
    assert select_history_indices(np.arange(4), 10, 10).tolist() == [0, 1, 2, 3]
    assert select_history_indices(np.array([], dtype=int), 1, 1).tolist() == []
    with pytest.raises(ValueError):
        select_history_indices(np.arange(4), 0, 0)
