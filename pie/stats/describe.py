"""Descriptive statistics: summaries, normality, missingness."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats as _sps


def summary_statistics(df: pd.DataFrame, variables: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Per-variable summary: n, mean, median, std, quantiles, skew, kurtosis, missing.

    Only accepts numeric columns; raises ValueError on categorical input so the
    caller fails fast instead of silently getting nonsense back.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for v in variables:
        if v not in df.columns:
            raise KeyError(v)
        s = df[v]
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"{v!r} is not numeric")
        n_missing = int(s.isna().sum())
        total = len(s)
        clean = s.dropna()
        n = int(len(clean))
        entry: Dict[str, Any] = {
            "n": n,
            "n_missing": n_missing,
            "pct_missing": float(100.0 * n_missing / total) if total else 0.0,
        }
        if n > 0:
            q1, q3 = clean.quantile([0.25, 0.75])
            entry.update({
                "mean": float(clean.mean()),
                "median": float(clean.median()),
                "std": float(clean.std(ddof=1)) if n > 1 else float("nan"),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "skew": float(_sps.skew(clean, bias=False, nan_policy="omit")) if n > 2 else float("nan"),
                "kurtosis": float(_sps.kurtosis(clean, bias=False, nan_policy="omit")) if n > 3 else float("nan"),
            })
        out[v] = entry
    return out


def normality_test(series: pd.Series, test: str = "shapiro", alpha: float = 0.05) -> Dict[str, Any]:
    """Shapiro-Wilk (n ≤ 5000) or Kolmogorov-Smirnov against a fitted normal.

    Shapiro is more powerful at small-to-moderate n; KS is the practical fallback
    when n > 5000 because Shapiro's reliability degrades there.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("normality_test requires numeric input")
    clean = series.dropna()
    n = int(len(clean))
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")
    if test == "shapiro":
        stat, p = _sps.shapiro(clean)
    elif test == "ks":
        stat, p = _sps.kstest(clean, "norm", args=(clean.mean(), clean.std(ddof=1)))
    else:
        raise ValueError(f"unknown test {test!r}; use 'shapiro' or 'ks'")
    return {
        "test": test,
        "statistic": float(stat),
        "p_value": float(p),
        "n": n,
        "is_normal": bool(p > alpha),
        "alpha": alpha,
    }


def missingness_report(df: pd.DataFrame, variables: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Per-column counts plus a chi-square pattern test across selected columns.

    A full Little's MCAR test requires an EM routine that isn't ergonomic to
    ship here; instead we compute a pragmatic chi-square on per-column missing
    counts against a uniform expectation. A very small p-value suggests
    missingness is structured (i.e. not MCAR) and warrants investigation.
    """
    from scipy.stats import chi2

    cols = list(variables) if variables is not None else list(df.columns)
    per_col: Dict[str, Dict[str, float]] = {}
    for c in cols:
        n_miss = int(df[c].isna().sum())
        per_col[c] = {
            "n_missing": n_miss,
            "pct_missing": float(100.0 * n_miss / len(df)) if len(df) else 0.0,
        }

    mcar: Optional[Dict[str, float]] = None
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        miss_counts = df[numeric_cols].isna().sum()
        expected = float(miss_counts.mean())
        if expected > 0:
            chi2_stat = float(((miss_counts - expected) ** 2 / max(expected, 1e-9)).sum())
            dof = max(len(numeric_cols) - 1, 1)
            mcar = {
                "statistic": chi2_stat,
                "p_value": float(1 - chi2.cdf(chi2_stat, dof)),
                "dof": dof,
                "interpretation": (
                    "Missingness pattern is consistent with MCAR (p > 0.05)"
                    if float(1 - chi2.cdf(chi2_stat, dof)) > 0.05
                    else "Missingness pattern varies across columns (p ≤ 0.05) — consider MAR/MNAR mechanisms"
                ),
            }

    return {
        "n_rows": int(len(df)),
        "per_column": per_col,
        "little_mcar": mcar,
    }
