"""Multiple-testing correction."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from statsmodels.stats.multitest import multipletests


_ALLOWED = {"bonferroni", "holm", "sidak", "fdr_bh", "fdr_by", "fdr_tsbh"}


def adjust_pvalues(p_values: List[float], method: str = "fdr_bh",
                   alpha: float = 0.05) -> Dict[str, Any]:
    """Apply multiple-testing correction.

    Supported methods: ``bonferroni``, ``holm``, ``sidak``, ``fdr_bh``
    (Benjamini-Hochberg), ``fdr_by`` (Benjamini-Yekutieli), ``fdr_tsbh``
    (two-stage Benjamini-Hochberg).

    A NaN p-value (a test that could not be computed) stays NaN, is never rejected, and
    does not count towards the number of tests; the others are adjusted among themselves.
    """
    if method not in _ALLOWED:
        raise ValueError(f"method must be one of {sorted(_ALLOWED)}, got {method!r}")
    p = np.asarray(p_values, dtype=float)
    ok = ~np.isnan(p)
    adjusted = np.full(p.shape, np.nan)
    rejected = np.zeros(p.shape, dtype=bool)
    if ok.any():
        reject, p_adj, _, _ = multipletests(p[ok], alpha=alpha, method=method)
        adjusted[ok] = p_adj
        rejected[ok] = reject
    return {
        "method": method,
        "alpha": alpha,
        "n_tests": int(ok.sum()),
        "original": list(p_values),
        "adjusted": [float(x) for x in adjusted],
        "rejected": [bool(r) for r in rejected],
    }
