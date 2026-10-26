import numpy as np
import pandas as pd
from recsys_research.features import build_features


def fixture():
    events = pd.DataFrame({
        "uid": [1, 1, 2], "item_id": [10, 10, 20], "artist_id": [3, 3, 4],
        "album_id": [5, 5, 6], "available_day": [1, 2, 2],
        "event_type": ["listen"] * 3, "played_ratio_pct": [100, 0, 100],
        "is_organic": [0, 0, 0],
    })
    queries = pd.DataFrame({"uid": [1, 1], "item_id": [10, 10], "artist_id": [3, 3],
                            "album_id": [5, 5], "day": [3, 4]})
    return events, queries


def test_strict_cutoff_and_embargo():
    e, q = fixture()
    x, _, _ = build_features(e, q, {"embargo_days": 1, "auto_windows": [], "auto_half_lives": []})
    np.testing.assert_allclose(x.user__count__all, np.log1p([1, 2]))
    np.testing.assert_allclose(x.user_item__count__w7, np.log1p([1, 2]))
    assert x.user__recency.tolist() == [2, 2]


def test_future_events_do_not_change_past_features():
    e, q = fixture()
    cfg = {"embargo_days": 1, "auto_windows": [2], "auto_half_lives": [3]}
    x, _, _ = build_features(e, q.iloc[:1], cfg)
    future = e.iloc[[0]].copy()
    future["available_day"] = 100
    future["played_ratio_pct"] = 0
    y, _, _ = build_features(pd.concat([e, future], ignore_index=True), q.iloc[:1], cfg)
    for col in x.select_dtypes(include="number"):
        np.testing.assert_allclose(x[col], y[col], rtol=1e-6, atol=1e-6, err_msg=col)


def test_unknown_metadata_not_shared_affinity():
    e, q = fixture()
    e["artist_id"] = -1
    q["artist_id"] = -1
    # Empty relation is permitted and must produce zero counts.
    x, _, _ = build_features(e, q, {"auto_windows": [], "auto_half_lives": []})
    assert (x.artist__count__all == 0).all()


def test_remote_future_cannot_destabilize_decay():
    e, q = fixture()
    cfg = {"embargo_days": 1, "auto_windows": [2], "auto_half_lives": [3]}
    x, _, _ = build_features(e, q, cfg)
    future = e.iloc[[0]].copy()
    future["available_day"] = 1_000_000
    y, _, _ = build_features(pd.concat([e, future], ignore_index=True), q, cfg)
    pd.testing.assert_frame_equal(x, y)
