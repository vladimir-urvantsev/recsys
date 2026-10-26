import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from recsys_research.metrics import evaluate, paired_user_bootstrap


def test_known_ndcg_and_recall():
    result = evaluate([1, 0, 1], [0.9, 0.8, 0.7], [1, 1, 1], [10, 10, 10], k=2)
    assert result["ndcg_mixed"] == pytest.approx(1 / (1 + 1 / np.log2(3)))
    assert result["recall_mixed"] == pytest.approx(0.5)
    assert result["n_mixed_groups"] == 1


def test_ties_have_expected_gain_independent_of_input_order():
    y = np.array([1, 0, 1, 0])
    scores = np.full(4, 0.5)
    first = evaluate(y, scores, np.ones(4), np.ones(4), k=2)
    second = evaluate(y[::-1], scores, np.ones(4), np.ones(4), k=2)
    assert first["ndcg_mixed"] == pytest.approx(0.5)
    assert first["recall_mixed"] == pytest.approx(0.5)
    assert first == second


def test_partial_tie_at_cutoff_matches_reference():
    y = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.7, 0.7, 0.7, 0.2])
    result = evaluate(y, scores, np.ones(5), np.ones(5), k=3)
    assert result["ndcg_mixed"] == pytest.approx(ndcg_score(y[None], scores[None], k=3))


def test_undefined_and_trivial_groups_are_reported():
    result = evaluate(
        [0, 0, 1, 1, 1],
        [0.2, 0.1, 0.9, 0.8, 0.5],
        [1, 1, 2, 2, 3],
        [10, 10, 20, 20, 30],
        k=1,
    )
    assert result["n_no_positive_groups"] == 1
    assert result["n_all_positive_groups"] == 1
    assert result["n_singleton_groups"] == 1
    assert result["n_mixed_groups"] == 0
    assert result["ndcg_mixed"] is None
    assert result["ndcg_all_defined"] == 1
    assert result["recall_all_defined"] == 0.5


def test_user_macro_differs_from_group_macro():
    # User 1 has two perfect days; user 2 has one imperfect day.
    result = evaluate(
        [1, 0, 1, 0, 0, 1], [0.9, 0.1, 0.8, 0.2, 0.8, 0.2],
        [1, 1, 1, 1, 2, 2], [10, 10, 11, 11, 20, 20], k=1,
    )
    assert result["ndcg_mixed"] == pytest.approx(2 / 3)
    assert result["user_macro_ndcg_mixed"] == pytest.approx(0.5)


def test_logloss_is_finite_for_extreme_correct_and_incorrect_probabilities():
    result = evaluate([0, 1], [1, 0], [1, 1], [1, 1])
    assert np.isfinite(result["logloss"])
    assert result["logloss"] == pytest.approx(-np.log(1e-7))
    assert result["roc_auc"] == 0


def test_single_class_auc_is_undefined():
    result = evaluate([1, 1], [0.1, 0.9], [1, 1], [1, 1])
    assert result["roc_auc"] is None
    assert result["average_precision"] == 1


def test_group_cannot_cross_users():
    with pytest.raises(ValueError, match="exactly one user"):
        evaluate([0, 1], [0.2, 0.8], [1, 2], [1, 1])


@pytest.mark.parametrize("scores", [[np.nan, 0.5], [-0.1, 0.8], [0.1, 1.1]])
def test_invalid_probabilities_rejected(scores):
    with pytest.raises(ValueError, match="probabilities"):
        evaluate([0, 1], scores, [1, 1], [1, 1])


def test_cluster_bootstrap_reproducible_and_perfect_improvement():
    args = (
        np.array([0, 1, 0, 1, 0, 1]),
        np.array([0.8, 0.2, 0.8, 0.2, 0.8, 0.2]),
        np.array([0.2, 0.8, 0.2, 0.8, 0.2, 0.8]),
        np.array([1, 1, 1, 1, 2, 2]),
        np.array([10, 10, 11, 11, 20, 20]),
    )
    first = paired_user_bootstrap(*args, k=1, n_boot=100, seed=17)
    second = paired_user_bootstrap(*args, k=1, n_boot=100, seed=17)
    assert first == second
    assert first["logloss"]["delta"] < 0
    assert first["logloss"]["bootstrap_fraction_improved"] == 1
    assert first["user_macro_ndcg_mixed"]["delta"] == 1
    assert first["user_macro_ndcg_mixed"]["ci95"] == [1, 1]
    assert first["n_users"] == 2


def test_bootstrap_matches_row_weighted_logloss_and_user_macro_difference():
    y = np.array([1, 0, 1, 0, 0, 1])
    base = np.array([0.9, 0.1, 0.8, 0.2, 0.8, 0.2])
    prop = 1 - base
    users = np.array([1, 1, 1, 1, 2, 2])
    groups = np.array([10, 10, 11, 11, 20, 20])
    result = paired_user_bootstrap(y, base, prop, users, groups, k=1, n_boot=100)
    b = evaluate(y, base, users, groups, k=1)
    p = evaluate(y, prop, users, groups, k=1)
    assert result["logloss"]["delta"] == pytest.approx(p["logloss"] - b["logloss"])
    assert result["user_macro_ndcg_mixed"]["delta"] == pytest.approx(
        p["user_macro_ndcg_mixed"] - b["user_macro_ndcg_mixed"]
    )


def test_identical_models_have_zero_bootstrap_delta():
    result = paired_user_bootstrap(
        [0, 1], [0.2, 0.8], [0.2, 0.8], [1, 1], [1, 1], n_boot=20
    )
    assert result["logloss"]["ci95"] == [0, 0]
    assert result["user_macro_ndcg_mixed"]["ci95"] == [0, 0]


def test_bootstrap_with_no_mixed_group_is_defined_for_loss_only():
    result = paired_user_bootstrap(
        [1, 1], [0.5, 0.5], [0.8, 0.8], [1, 1], [1, 1], n_boot=20
    )
    assert result["logloss"]["delta"] < 0
    assert result["user_macro_ndcg_mixed"]["delta"] is None
    assert result["user_macro_ndcg_mixed"]["ci95"] is None


def test_one_user_bootstrap_keeps_the_entire_history_together():
    # Loss differences vary by row. Resampling rows would create variation;
    # resampling one complete user must reproduce the same point each time.
    result = paired_user_bootstrap(
        [0, 1, 0, 1], [0.1, 0.8, 0.7, 0.9], [0.2, 0.7, 0.3, 0.6],
        [1, 1, 1, 1], [10, 10, 11, 11], k=1, n_boot=100,
    )
    assert result["logloss"]["ci95"] == pytest.approx([result["logloss"]["delta"]] * 2)
    assert result["user_macro_ndcg_mixed"]["ci95"] == pytest.approx(
        [result["user_macro_ndcg_mixed"]["delta"]] * 2
    )


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_tied_ndcg_matches_sklearn_across_cutoffs(k):
    rng = np.random.default_rng(11)
    for _ in range(10):
        y = rng.integers(0, 2, size=8)
        y[0], y[1] = 0, 1
        scores = rng.integers(0, 3, size=8) / 2
        result = evaluate(y, scores, np.ones(8), np.ones(8), k=k)
        reference = ndcg_score(y[None], scores[None], k=k)
        assert result["ndcg_mixed"] == pytest.approx(reference)
