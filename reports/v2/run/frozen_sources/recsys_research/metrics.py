"""Evaluation for conditional logged-play prediction and reranking.

Classification metrics weight recorded plays equally. Ranking metrics weight
user-day groups equally, or users equally where explicitly named ``user_macro``.
Ranking is evaluated only for groups with at least two candidates. No-positive
groups have undefined recall/NDCG and are counted rather than silently scored 0.
Equal-score blocks use their *expected* gain under a uniform random tie break.
Thus a constant predictor cannot profit from the input order or target labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7


def _validated(y, scores, user_ids, group_ids):
    arrays = tuple(np.asarray(a) for a in (y, scores, user_ids, group_ids))
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("Inputs must be one-dimensional arrays")
    if not len(arrays[0]) or any(len(a) != len(arrays[0]) for a in arrays):
        raise ValueError("Inputs must be nonempty and have equal length")
    y, scores, user_ids, group_ids = arrays
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Labels must be binary (0 or 1)")
    scores = scores.astype(np.float64)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    return y.astype(np.float64), scores, user_ids, group_ids


def _ranking_groups(y, scores, user_ids, group_ids, k):
    """Return group owner/count/positives and tie-aware ranking values."""
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    _, inverse = np.unique(group_ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    lengths = np.bincount(inverse)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    owners = user_ids[order[starts]]
    positives = np.zeros(len(lengths), dtype=np.int64)
    ndcg = np.full(len(lengths), np.nan)
    recall = np.full(len(lengths), np.nan)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for g, (start, length) in enumerate(zip(starts, lengths)):
        indices = order[start : start + length]
        if not np.all(user_ids[indices] == owners[g]):
            raise ValueError("Each group_id must belong to exactly one user")
        gains = y[indices]
        n_positive = int(gains.sum())
        positives[g] = n_positive
        if length < 2 or not n_positive:
            continue
        ranked = np.argsort(-scores[indices], kind="stable")
        gains, values = gains[ranked], scores[indices][ranked]
        # A tie block crossing k contributes its mean gain times the discounts
        # of the occupied top-k slots, including the whole block in that mean.
        breaks = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, length]
        dcg, hits = 0.0, 0.0
        for left, right in zip(breaks[:-1], breaks[1:]):
            if left >= k:
                break
            occupied_end = min(int(right), k)
            mean_gain = float(gains[left:right].mean())
            dcg += mean_gain * float(discounts[left:occupied_end].sum())
            hits += mean_gain * (occupied_end - left)
        ideal = float(discounts[: min(n_positive, k)].sum())
        ndcg[g] = dcg / ideal
        recall[g] = hits / n_positive
    return owners, lengths, positives, ndcg, recall


def _mean_or_none(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _user_macro(values, owners, mask):
    if not np.any(mask):
        return None
    _, codes = np.unique(owners[mask], return_inverse=True)
    averages = np.bincount(codes, weights=values[mask]) / np.bincount(codes)
    return float(averages.mean())


def evaluate(y, scores, user_ids, group_ids, k=10) -> dict:
    """Compute natural logged-play classification and user-day ranking metrics.

    ``ndcg_mixed`` is a group macro average over groups containing both labels;
    ``user_macro_ndcg_mixed`` first averages eligible days within each user.
    All-positive groups are reported separately because their NDCG is always 1.
    All-defined metrics include these groups, but exclude undefined groups.
    Predictions are clipped to [1e-7, 1-1e-7] only for logarithmic loss.
    """
    y, scores, user_ids, group_ids = _validated(y, scores, user_ids, group_ids)
    p = np.clip(scores, EPS, 1.0 - EPS)
    loss = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    owners, lengths, positives, ndcg, recall = _ranking_groups(
        y, scores, user_ids, group_ids, k
    )
    eligible = lengths >= 2
    mixed = eligible & (positives > 0) & (positives < lengths)
    defined = eligible & (positives > 0)
    return {
        "logloss": float(loss.mean()),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "brier": float(np.mean((scores - y) ** 2)),
        "average_precision": float(average_precision_score(y, scores)) if y.sum() else None,
        "ndcg_mixed": _mean_or_none(ndcg[mixed]),
        "recall_mixed": _mean_or_none(recall[mixed]),
        "user_macro_ndcg_mixed": _user_macro(ndcg, owners, mixed),
        "user_macro_recall_mixed": _user_macro(recall, owners, mixed),
        "ndcg_all_defined": _mean_or_none(ndcg[defined]),
        "recall_all_defined": _mean_or_none(recall[defined]),
        "n_rows": int(len(y)),
        "n_users": int(len(np.unique(user_ids))),
        "positive_rate": float(y.mean()),
        "n_groups": int(len(lengths)),
        "n_groups_ge2": int(eligible.sum()),
        "n_mixed_groups": int(mixed.sum()),
        "n_mixed_users": int(len(np.unique(owners[mixed]))),
        "n_all_positive_groups": int((eligible & (positives == lengths)).sum()),
        "n_no_positive_groups": int((eligible & (positives == 0)).sum()),
        "n_singleton_groups": int((lengths == 1).sum()),
        "ranking_k": int(k),
        "tie_handling": "expected gain under a uniform random tie break",
    }


def _bootstrap_summary(point, samples, lower_is_better=False):
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return {"delta": point, "ci95": None, "bootstrap_fraction_improved": None, "n_valid": 0}
    return {
        "delta": point,
        "ci95": [float(v) for v in np.quantile(samples, [0.025, 0.975])],
        "bootstrap_fraction_improved": float(np.mean(samples < 0 if lower_is_better else samples > 0)),
        "n_valid": int(len(samples)),
    }


def paired_user_bootstrap(
    y, baseline, proposed, user_ids, group_ids, k=10, n_boot=1000, seed=2026
) -> dict:
    """Paired percentile intervals by resampling complete users with replacement.

    Deltas always mean proposed minus baseline. For logloss, row weights within
    each sampled user are retained. User-macro NDCG weights each eligible user
    copy equally, after averaging their mixed-label days. The reported bootstrap
    fraction improved describes the bootstrap distribution; it is not a p-value
    or a posterior probability. AUC is intentionally not bootstrapped: its
    pairwise denominator does not admit the same cheap per-user aggregation.
    """
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    y, baseline, user_ids, group_ids = _validated(y, baseline, user_ids, group_ids)
    _, proposed, _, _ = _validated(y, proposed, user_ids, group_ids)
    users, codes = np.unique(user_ids, return_inverse=True)
    n_users = len(users)
    row_counts = np.bincount(codes, minlength=n_users)
    losses = []
    ranks = []
    eligible_users = None
    for scores in (baseline, proposed):
        p = np.clip(scores, EPS, 1 - EPS)
        per_row = -(y * np.log(p) + (1 - y) * np.log1p(-p))
        losses.append(np.bincount(codes, weights=per_row, minlength=n_users))
        owners, lengths, positives, ndcg, _ = _ranking_groups(y, scores, user_ids, group_ids, k)
        mixed = (lengths >= 2) & (positives > 0) & (positives < lengths)
        owner_codes = np.searchsorted(users, owners[mixed])
        group_counts = np.bincount(owner_codes, minlength=n_users)
        sums = np.bincount(owner_codes, weights=ndcg[mixed], minlength=n_users)
        per_user = np.divide(sums, group_counts, out=np.zeros(n_users), where=group_counts > 0)
        ranks.append(per_user)
        eligible_users = group_counts > 0
    loss_delta = losses[1] - losses[0]
    rank_delta = ranks[1] - ranks[0]
    loss_point = float(loss_delta.sum() / row_counts.sum())
    rank_point = float(rank_delta[eligible_users].mean()) if np.any(eligible_users) else None
    loss_samples = np.empty(n_boot)
    rank_samples = np.full(n_boot, np.nan)
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        sampled = rng.integers(0, n_users, size=n_users)
        loss_samples[b] = loss_delta[sampled].sum() / row_counts[sampled].sum()
        eligible_sample = sampled[eligible_users[sampled]]
        if len(eligible_sample):
            rank_samples[b] = rank_delta[eligible_sample].mean()
    return {
        "resampling_unit": "user (all rows and user-day groups retained together)",
        "delta_definition": "proposed minus baseline",
        "interval_method": "paired user-cluster percentile bootstrap, 95%",
        "bootstrap_fraction_note": "Descriptive fraction of resamples improving; not a p-value or posterior probability.",
        "n_users": int(n_users),
        "n_mixed_users": int(eligible_users.sum()),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "logloss": _bootstrap_summary(loss_point, loss_samples, lower_is_better=True),
        "user_macro_ndcg_mixed": _bootstrap_summary(rank_point, rank_samples),
    }
