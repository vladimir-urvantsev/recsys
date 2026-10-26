from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recsys_history_v2.data import choose_cohorts, make_v2_queries, prepare_data
from recsys_research.data import DAY_SECONDS, add_availability


def config(**extra):
    return {"n_users": 6, "development_users": 2, "user_seed": 333, "cohort_seed": 77,
            "warmup_days": 1, "split_days": [10, 20, 30], "day_seed": 9,
            "days_per_user": {name: 30 for name in ("train", "tune", "select", "test")}, **extra}


def events(rows):
    values = []
    for position, (uid, item, day, ratio, organic, kind, second) in enumerate(rows):
        values.append({"uid": uid, "item_id": item, "timestamp": day * DAY_SECONDS + second,
                       "played_ratio_pct": ratio, "track_length_seconds": 200 if kind == "listen" else None,
                       "is_organic": organic, "event_type": kind, "source_order": position,
                       "artist_id": item + 100, "album_id": item + 1000})
    return add_availability(pd.DataFrame(values))


def test_fresh_cohort_excludes_all_old_users_and_is_not_activity_weighted():
    population = np.repeat(np.arange(20), np.arange(1, 21))
    old = np.array([0, 2, 4, 6, 8])
    left = choose_cohorts(population, old, config())
    right = choose_cohorts(np.arange(20), old, config())
    for name in left:
        np.testing.assert_array_equal(left[name], right[name])
    assert len(left["sampled_uids"]) == 6
    assert len(left["development_uids"]) == 2
    assert len(left["confirmation_uids"]) == 4
    assert not np.intersect1d(left["sampled_uids"], old).size
    assert not np.intersect1d(left["development_uids"], left["confirmation_uids"]).size
    np.testing.assert_array_equal(np.union1d(left["development_uids"], left["confirmation_uids"]), left["sampled_uids"])


def test_cohort_sizes_fail_before_reading_events():
    with pytest.raises(ValueError, match="after excluding"):
        choose_cohorts(np.arange(10), np.arange(5), config(n_users=6))
    with pytest.raises(ValueError, match="nonempty"):
        choose_cohorts(np.arange(10), [], config(development_users=6))


def test_user_time_allocation_and_no_test_outcome_surrogate_in_inputs():
    frame = events([(uid, item, day, ratio, 0, "listen", 10)
                    for uid in [1, 2] for day in [2, 12, 22, 32]
                    for item, ratio in [(10, 50), (11, 49)]])
    queries, labels, stats = make_v2_queries(frame, config(), np.array([1]), np.array([2]))
    assert set(queries.loc[queries.split.eq("train"), "uid"]) == {1, 2}
    assert set(queries.loc[queries.split.isin(["tune", "select"]), "uid"]) == {1}
    assert set(queries.loc[queries.split.eq("test"), "uid"]) == {2}
    assert queries.loc[queries.split.eq("test"), "label"].isna().all()
    assert queries.loc[~queries.split.eq("test"), "label"].notna().all()
    assert not {"played_ratio_pct", "available_at", "available_day", "track_length_seconds"}.intersection(queries)
    assert labels.label.tolist() == [1, 0]
    assert queries.loc[queries.split.eq("test"), ["uid", "item_id", "day", "group_id"]].reset_index(drop=True).equals(
        labels.drop(columns="label"))
    assert "positive_rows" not in stats["splits"]["test"]
    assert "positive_rate" not in stats["splits"]["test"]


def test_first_all_source_play_blocks_later_recommended_play_without_truncating_history():
    frame = events([(2, 10, 32, 10, 1, "listen", 1),
                    (2, 10, 32, 100, 0, "listen", 100),
                    (2, 11, 32, 50, 0, "listen", 10),
                    (1, 20, 2, 100, 0, "listen", 10)])
    original = frame.copy(deep=True)
    queries, labels, _ = make_v2_queries(frame, config(), [1], [2])
    assert labels.item_id.tolist() == [11]
    pd.testing.assert_frame_equal(frame, original)
    assert len(frame) == 4


def test_next_stage_availability_purge_is_kept_and_raw_history_remains_complete():
    frame = events([(1, 1, 9, 150, 0, "listen", DAY_SECONDS - 300),
                    (1, 2, 9, 150, 0, "listen", DAY_SECONDS - 400),
                    (2, 3, 32, 100, 0, "listen", 10)])
    queries, _, stats = make_v2_queries(frame, config(), [1], [2])
    assert queries.loc[queries.split.eq("train"), "item_id"].tolist() == [2]
    assert stats["purged_label_rows_by_split_before_cohort_filter"]["train"] == 1
    assert len(frame) == 3


def test_day_selection_independent_of_outcomes_and_other_cohort_members():
    frame = events([(uid, item, day, 20 if item == 1 else 90, 0, "listen", 10)
                    for uid in [1, 2, 3] for day in range(1, 40) for item in [1, 2]])
    limited = config(days_per_user={name: 2 for name in ("train", "tune", "select", "test")})
    left, sealed_left, _ = make_v2_queries(frame, limited, [1], [2, 3])
    changed = frame.copy()
    changed["played_ratio_pct"] = 100 - changed.played_ratio_pct
    right, sealed_right, _ = make_v2_queries(changed, limited, [1], [2, 3])
    cols = ["uid", "item_id", "day", "group_id", "split", "cohort"]
    pd.testing.assert_frame_equal(left[cols], right[cols])
    assert not sealed_left.label.equals(sealed_right.label)
    assert left.groupby("group_id").size().eq(2).all()


def test_unallocated_or_overlapping_user_cohorts_rejected():
    frame = events([(1, 1, 2, 100, 0, "listen", 10), (3, 3, 32, 20, 0, "listen", 10)])
    with pytest.raises(ValueError, match="disjoint"):
        make_v2_queries(frame, config(), [1], [1, 3])
    with pytest.raises(ValueError, match="allocated"):
        make_v2_queries(frame, config(), [1], [2])


def test_query_end_day_truncates_supervision_without_removing_history():
    frame = events([(uid, item, day, ratio, 0, "listen", 10)
                    for uid in [1, 2] for day in [2, 12, 22, 32, 35, 42]
                    for item, ratio in [(10, 50), (11, 49)]])
    original = frame.copy(deep=True)
    queries, labels, stats = make_v2_queries(frame, config(
        end_day=35, days_per_user={"train": 30, "tune": 30, "select": 30, "test": 1}
    ), [1], [2])
    assert queries.day.max() == 32
    assert labels.day.tolist() == [32, 32]
    assert stats["query_end_day_exclusive"] == 35
    assert stats["splits"]["test"]["day_max"] == 32
    pd.testing.assert_frame_equal(frame, original)
    assert set(frame.day) == {2, 12, 22, 32, 35, 42}
    with pytest.raises(ValueError, match="strictly after"):
        make_v2_queries(frame, config(end_day=30), [1], [2])


def test_prepare_persists_every_sampled_event_and_seals_confirmation_labels(tmp_path, monkeypatch):
    import recsys_history_v2.data as module
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "multi_event.parquet"
    pq.write_table(pa.table({"uid": pa.array([1, 2, 3, 4], type=pa.uint32())}), source)
    excluded = tmp_path / "old.json"
    excluded.write_text(json.dumps({"sampled_uids": [4]}))
    original = events([(uid, item, day, ratio, 0, "listen", 10)
                       for uid in [1, 2, 3] for item, ratio in [(10, 100), (11, 10)]
                       for day in [0, 2, 12, 22, 32, 42]])
    monkeypatch.setattr(module, "_source_manifest", lambda _: {"source": {"sha256": "fixture"}})
    monkeypatch.setattr(module, "read_user_histories", lambda path, selected: original.loc[original.uid.isin(selected)].copy())
    monkeypatch.setattr(module, "read_relation_mapping", lambda path, items, relation: (
        pd.Series({10: 7, 11: 8}), {"sample_items_with_mapping": 2}))
    cfg = config(n_users=3, development_users=1, raw_dir=str(raw), exclude_manifest=str(excluded), end_day=35)
    output = tmp_path / "prepared"
    full, queries, manifest = prepare_data(cfg, output)
    assert len(full) == len(original)
    assert manifest["old_user_overlap"] == 0
    assert set(manifest["sampled_uids"]) == {1, 2, 3}
    saved = pd.read_parquet(output / "queries.parquet")
    assert saved.loc[saved.split.eq("test"), "label"].isna().all()
    assert "played_ratio_pct" not in saved
    assert len(pd.read_parquet(output / "sample_events.parquet")) == len(original)
    assert pd.read_parquet(output / "sample_events.parquet").day.max() == 42
    sealed = pd.read_parquet(output / "test_labels.parquet")
    assert len(sealed) == manifest["splits"]["test"]["rows"]
    assert sealed.day.max() == 32
    assert set(sealed.uid).isdisjoint(manifest["development_uids"])
    assert "positive_rate" not in manifest["splits"]["test"]
    assert set(manifest["artifacts"]) == {"sample_events.parquet", "queries.parquet", "test_labels.parquet"}
    with pytest.raises(FileExistsError):
        prepare_data(cfg, output)
