import numpy as np
import pytest

from recsys_research.run import classification_loss


def test_unsigned_labels_cannot_wrap_selection_objective():
    y = np.array([1, 0], dtype=np.uint8)
    p = np.array([0.8, 0.2])
    assert classification_loss(y, p) == pytest.approx(-np.log(0.8))
    assert classification_loss(y, p) == classification_loss(y.astype(float), p)


def test_loss_rewards_correct_probabilities():
    y = np.array([0, 1, 1, 0], dtype=np.uint8)
    good = classification_loss(y, np.array([0.1, 0.9, 0.9, 0.1]))
    constant = classification_loss(y, np.full(4, 0.5))
    wrong = classification_loss(y, np.array([0.9, 0.1, 0.1, 0.9]))
    assert 0 <= good < constant < wrong


def test_extreme_probabilities_have_finite_nonnegative_loss():
    assert np.isfinite(classification_loss(np.array([1, 0], np.uint8), np.array([0., 1.])))
    assert classification_loss(np.array([1, 0], np.uint8), np.array([1., 0.])) >= 0
