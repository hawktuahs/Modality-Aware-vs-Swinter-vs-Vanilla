"""Statistical tests for paired model comparisons on per-case metrics."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def wilcoxon_signed_rank(a: Iterable[float], b: Iterable[float]
                         ) -> Dict[str, float]:
    """Paired two-sided Wilcoxon signed-rank test.

    Returns {'stat', 'p', 'n', 'mean_delta', 'median_delta'}.
    Requires scipy; returns NaNs if scipy is missing.
    """
    a = np.asarray(list(a), dtype=np.float64)
    b = np.asarray(list(b), dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = int(len(a))
    out = {"n": n, "mean_delta": float(np.mean(a - b)) if n else float("nan"),
           "median_delta": float(np.median(a - b)) if n else float("nan"),
           "stat": float("nan"), "p": float("nan")}
    if n < 2 or np.allclose(a, b):
        return out
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        out["stat"] = float(stat)
        out["p"] = float(p)
    except Exception:
        pass
    return out


def bootstrap_ci(values: Iterable[float], n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Returns (mean, ci_low, ci_high).
    """
    x = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(x.mean()), lo, hi


def significance_star(p: float) -> str:
    if np.isnan(p):
        return "?"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
