"""Temporal sensitivity intervals; primary inference remains user-clustered."""
import numpy as np


def temporal_sensitivity(y, baseline, proposed, days, n_boot=5000, seed=20261026):
    """Paired circular moving-day-block bootstrap for row-weighted LogLoss.

    This is a secondary sensitivity analysis to common temporal shocks, not a
    substitute for primary user-cluster inference or a joint two-way interval.
    """
    y = np.asarray(y, np.float64)
    days = np.asarray(days, np.int64)
    if n_boot < 1:
        raise ValueError("Positive bootstrap count required")
    losses = []
    for scores in (baseline, proposed):
        p = np.clip(np.asarray(scores, np.float64), 1e-7, 1-1e-7)
        losses.append(-y*np.log(p) - (1-y)*np.log1p(-p))
    codes = days - days.min()
    n_days = int(codes.max()) + 1
    sums = np.bincount(codes, weights=losses[1]-losses[0], minlength=n_days)
    counts = np.bincount(codes, minlength=n_days)
    rng = np.random.default_rng(seed)
    intervals = {}
    for length in (3, 7):
        values = []
        for _ in range(n_boot):
            starts = rng.integers(n_days, size=int(np.ceil(n_days/length)))
            selected = ((starts[:, None]+np.arange(length)) % n_days).ravel()[:n_days]
            denominator = counts[selected].sum()
            if denominator:
                values.append(sums[selected].sum()/denominator)
        intervals[str(length)] = {"ci95": np.quantile(values, [.025, .975]).tolist(),
                                   "n_boot": len(values)}
    return {"delta_logloss": float((losses[1]-losses[0]).mean()), "blocks_days": intervals,
            "n_observation_days": n_days, "seed": seed,
            "note": "Secondary circular day-block bootstrap; no multiplicity adjustment; not a joint user/time interval or evidence for a new calendar regime.",
            "daily_delta": [{"day": int(days.min()+d), "rows": int(counts[d]),
                             "delta_logloss": float(sums[d]/counts[d]) if counts[d] else None}
                            for d in range(n_days)]}
