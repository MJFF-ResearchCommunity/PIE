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


def _em_normal(x: np.ndarray, max_iter: int = 10_000, tol: float = 1e-10):
    """Maximum-likelihood mean and covariance of multivariate-normal data with gaps (EM).

    Rows are grouped by missingness pattern; the E-step fills each pattern's missing
    values with their conditional expectation and adds the conditional covariance.
    """
    observed = ~np.isnan(x)
    patterns, inverse = np.unique(observed, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n, p = x.shape
    mu = np.nanmean(x, axis=0)
    sigma = np.diag(np.nanvar(x, axis=0))
    for _ in range(max_iter):
        filled = np.where(observed, x, 0.0)
        extra = np.zeros((p, p))
        for k, obs in enumerate(patterns):
            miss = ~obs
            if not miss.any():
                continue
            rows = inverse == k
            s_mo = sigma[np.ix_(miss, obs)]
            coef = np.linalg.solve(sigma[np.ix_(obs, obs)], s_mo.T).T
            filled[np.ix_(rows, miss)] = mu[miss] + (x[np.ix_(rows, obs)] - mu[obs]) @ coef.T
            extra[np.ix_(miss, miss)] += rows.sum() * (sigma[np.ix_(miss, miss)] - coef @ s_mo.T)
        new_mu = filled.mean(axis=0)
        centred = filled - new_mu
        new_sigma = (centred.T @ centred + extra) / n
        done = max(np.abs(new_mu - mu).max(), np.abs(new_sigma - sigma).max()) < tol
        mu, sigma = new_mu, new_sigma
        if done:
            break
    return mu, sigma, patterns, inverse


def _little_mcar(frame: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Little's (1988) chi-square test that numeric data are missing completely at random.

    EM gives the ML mean μ and covariance Σ under multivariate normality, then
    d² = Σ_j n_j (ȳ_j − μ_j)ᵀ Σ_j⁻¹ (ȳ_j − μ_j) over missingness patterns j, on Σ p_j − p
    degrees of freedom. Reproduces naniar::mcar_test on R's airquality (d² = 35.1, df = 14).
    None when the test is undefined: nothing missing, fewer than two usable columns, no
    degrees of freedom, or a singular covariance (e.g. a constant column).
    """
    x = frame.to_numpy(dtype=float, na_value=np.nan)
    x = x[:, ~np.isnan(x).all(axis=0)]            # an all-missing column carries no information
    x = x[~np.isnan(x).all(axis=1)]               # nor does an all-missing row
    if x.shape[1] < 2 or not np.isnan(x).any():
        return None
    try:
        mu, sigma, patterns, inverse = _em_normal(x)
        d2, dof = 0.0, -x.shape[1]
        for k, obs in enumerate(patterns):
            rows = x[inverse == k][:, obs]
            diff = rows.mean(axis=0) - mu[obs]
            d2 += len(rows) * float(diff @ np.linalg.solve(sigma[np.ix_(obs, obs)], diff))
            dof += int(obs.sum())
    except np.linalg.LinAlgError:
        return None
    if dof < 1:
        return None
    p = float(_sps.chi2.sf(d2, dof))
    return {
        "statistic": d2,
        "p_value": p,
        "dof": dof,
        "n_patterns": int(len(patterns)),
        "n_rows": int(len(x)),
        "interpretation": (
            "No evidence that missingness depends on the observed values (p > 0.05); "
            "this does not show the data are MCAR"
            if p > 0.05
            else "Missingness depends on the observed values (p ≤ 0.05): not MCAR, "
                 "so complete-case estimates may be biased"
        ),
    }


def missingness_report(df: pd.DataFrame, variables: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Per-column missing counts plus Little's MCAR test across the numeric columns.

    ``little_mcar`` is Little's (1988) test: EM estimates of the mean and covariance, then
    the d² statistic across missingness patterns. It assumes the numeric columns are
    jointly normal and its p-value is asymptotic. A non-significant result only fails to
    show that missingness depends on observed values; it cannot rule out MNAR.
    """
    cols = list(variables) if variables is not None else list(df.columns)
    per_col: Dict[str, Dict[str, float]] = {}
    for c in cols:
        n_miss = int(df[c].isna().sum())
        per_col[c] = {
            "n_missing": n_miss,
            "pct_missing": float(100.0 * n_miss / len(df)) if len(df) else 0.0,
        }
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return {
        "n_rows": int(len(df)),
        "per_column": per_col,
        "little_mcar": _little_mcar(df[numeric_cols]),
    }
