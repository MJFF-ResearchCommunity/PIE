"""Multiple-testing correction."""
from __future__ import annotations

from typing import Any, Dict, List

from statsmodels.stats.multitest import multipletests


_ALLOWED = {"bonferroni", "holm", "sidak", "fdr_bh", "fdr_by", "fdr_tsbh"}


def adjust_pvalues(p_values: List[float], method: str = "fdr_bh",
                   alpha: float = 0.05) -> Dict[str, Any]:
    """Apply multiple-testing correction.

    Supported methods: ``bonferroni``, ``holm``, ``sidak``, ``fdr_bh``
    (Benjamini-Hochberg), ``fdr_by`` (Benjamini-Yekutieli), ``fdr_tsbh``
    (two-stage Benjamini-Hochberg).
    """
    if method not in _ALLOWED:
        raise ValueError(f"method must be one of {sorted(_ALLOWED)}, got {method!r}")
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return {
        "method": method,
        "alpha": alpha,
        "original": list(p_values),
        "adjusted": [float(p) for p in p_adj],
        "rejected": [bool(r) for r in reject],
    }
