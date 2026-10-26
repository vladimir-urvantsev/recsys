"""Known-result checks for paired circular day-block sensitivity."""
import numpy as np
import pytest

from recsys_history_v2.statistics import temporal_sensitivity


def test_known_constant_improvement_has_exact_point_and_interval_with_uint8_labels():
    y = np.tile(np.array([0, 1], dtype=np.uint8), 12)
    baseline = np.full(len(y), .5)
    proposed = np.where(y == 1, .8, .2)
    days = np.repeat(np.arange(270, 282), 2)
    result = temporal_sensitivity(y, baseline, proposed, days, n_boot=100, seed=81)
    expected = np.log(.5 / .8)
    assert result["delta_logloss"] == pytest.approx(expected)
    assert result["delta_logloss"] < 0  # proposed is better
    assert result["n_observation_days"] == 12
    for block in ("3", "7"):
        assert result["blocks_days"][block]["ci95"] == pytest.approx([expected, expected])
        assert result["blocks_days"][block]["n_boot"] == 100


def test_identical_predictions_give_zero_for_every_temporal_resample():
    y = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
    scores = np.array([.2, .7, .3, .9, .6, .4])
    days = np.array([270, 270, 271, 271, 272, 272])
    result = temporal_sensitivity(y, scores, scores, days, n_boot=50)
    assert result["delta_logloss"] == 0
    assert all(v["ci95"] == [0.0, 0.0] for v in result["blocks_days"].values())


def test_day_gaps_remain_calendar_bins_and_daily_rows_recover_the_weighted_point():
    y = np.ones(10, dtype=np.uint8)
    baseline = np.full(10, .5)
    proposed = np.r_[.8, np.full(9, .4)]
    days = np.r_[270, np.full(9, 277)]
    result = temporal_sensitivity(y, baseline, proposed, days, n_boot=200, seed=29)
    assert result["n_observation_days"] == 8
    records = result["daily_delta"]
    assert [r["day"] for r in records] == list(range(270, 278))
    assert [r["rows"] for r in records] == [1, 0, 0, 0, 0, 0, 0, 9]
    assert all(r["delta_logloss"] is None for r in records[1:-1])
    expected = (np.log(.5 / .8) + 9 * np.log(.5 / .4)) / 10
    recovered = sum(r["rows"] * r["delta_logloss"] for r in records if r["rows"]) / len(y)
    assert result["delta_logloss"] == pytest.approx(expected)
    assert recovered == pytest.approx(expected)
    assert result["delta_logloss"] > 0  # many worsened rows outweigh one improved row


def test_seed_is_reproducible_and_sign_reversal_reverses_interval_endpoints():
    y = np.tile([0, 1, 1, 0], 8)
    baseline = np.linspace(.15, .85, len(y))
    proposed = np.linspace(.8, .2, len(y))
    days = np.repeat(np.arange(270, 286), 2)
    first = temporal_sensitivity(y, baseline, proposed, days, n_boot=300, seed=31)
    repeat = temporal_sensitivity(y, baseline, proposed, days, n_boot=300, seed=31)
    reverse = temporal_sensitivity(y, proposed, baseline, days, n_boot=300, seed=31)
    assert first == repeat
    assert reverse["delta_logloss"] == pytest.approx(-first["delta_logloss"])
    for block in ("3", "7"):
        assert reverse["blocks_days"][block]["ci95"] == pytest.approx(
            [-first["blocks_days"][block]["ci95"][1], -first["blocks_days"][block]["ci95"][0]])


def test_translation_of_relative_origin_does_not_change_inference():
    y = np.array([0, 1, 1, 0, 1, 0])
    baseline = np.array([.3, .6, .7, .2, .4, .8])
    proposed = np.array([.2, .7, .6, .3, .5, .7])
    days = np.array([270, 271, 274, 274, 277, 280])
    left = temporal_sensitivity(y, baseline, proposed, days, n_boot=100, seed=11)
    right = temporal_sensitivity(y, baseline, proposed, days + 1000, n_boot=100, seed=11)
    assert left["delta_logloss"] == right["delta_logloss"]
    assert left["blocks_days"] == right["blocks_days"]
    assert "not a joint user/time interval" in left["note"]
    assert "or evidence for a new calendar regime" in left["note"]

