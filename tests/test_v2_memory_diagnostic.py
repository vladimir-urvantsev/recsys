"""Independent tiny oracles for the optional post-confirmation diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_v2_memory.py"
SPEC = importlib.util.spec_from_file_location("diagnose_v2_memory_test_module", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_retained_presence_uses_listens_and_known_ids_only():
    history = np.array([[
        [7, 70, 700], [8, 80, 800], [9, -1, -1], [77, 770, 7700],
    ]], dtype=np.int32)
    mask = np.array([[True, True, True, False]])
    kinds = np.array([[1, 2, 1, 1]], np.uint8)
    candidates = np.array([[7, 70, 700], [8, 80, 800], [9, -1, -1], [77, 770, 7700]])
    actual = MODULE.retained_listen_presence(candidates, np.zeros(4, np.int64), history, mask, kinds)
    np.testing.assert_array_equal(actual, [
        [True, True, True], [False, False, False], [True, False, False], [False, False, False],
    ])


def test_groupwise_membership_matches_an_independent_row_token_oracle():
    history = np.array([
        [[1, 10, 100], [2, 20, 200], [-2, -2, -2]],
        [[3, 30, 300], [4, 40, 400], [5, -1, 500]],
    ], dtype=np.int32)
    mask = np.array([[True, True, False], [True, True, True]])
    kinds = np.array([[1, 3, 0], [2, 1, 1]], np.uint8)
    candidates = np.array([[4, 30, 300], [1, 20, 100], [5, -1, 500], [3, 40, 400]])
    codes = np.array([1, 0, 1, 1])
    expected = np.zeros((len(codes), 3), bool)
    for row, code in enumerate(codes):
        for entity in range(3):
            expected[row, entity] = any(
                bool(mask[code, token]) and kinds[code, token] == 1
                and candidates[row, entity] >= 0 and history[code, token, entity] >= 0
                and candidates[row, entity] == history[code, token, entity]
                for token in range(history.shape[1])
            )
    actual = MODULE.retained_listen_presence(candidates, codes, history, mask, kinds)
    np.testing.assert_array_equal(actual, expected)


def test_memory_bins_distinguish_no_history_retained_and_lost_listens():
    full = np.array([[False, True, True], [True, True, False]])
    retained = np.array([[False, True, False], [False, True, False]])
    bins = MODULE.prior_bins(full, retained)
    np.testing.assert_array_equal(bins, [[0, 1, 2], [2, 1, 0]])
    retained[0, 0] = True
    with pytest.raises(ValueError, match="without full-history"):
        MODULE.prior_bins(full, retained)


def test_unsigned_labels_have_correct_nonnegative_classification_loss():
    result = MODULE.classification_metrics(np.array([1, 0], np.uint8), np.array([.8, .2]))
    assert result["logloss"] == pytest.approx(-np.log(.8))
    assert result["auc"] == 1
    assert MODULE.classification_metrics(np.array([1, 1], np.uint8), np.array([.8, .9]))["auc"] is None
    assert MODULE.classification_metrics(np.array([], np.uint8), np.array([])) == {"logloss": None, "auc": None}
