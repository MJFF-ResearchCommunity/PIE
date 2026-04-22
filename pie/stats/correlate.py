"""Correlation: pairwise, partial, and matrix-wide with FDR adjustment."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats as _sps


def correlate_pair(a: pd.Series, b: pd.Series, method: str = "pearson") -> Dict[str, Any]:
    """Pearson / Spearman / Kendall correlation for two Series."""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 3:
        raise ValueError(f"need at least 3 complete pairs, got {len(df)}")
    x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
    if method == "pearson":
        r, p = _sps.pearsonr(x, y)
    elif method == "spearman":
        r, p = _sps.spearmanr(x, y)
    elif method == "kendall":
        r, p = _sps.kendalltau(x, y)
    else:
        raise ValueError(f"unknown method {method!r}; use pearson/spearman/kendall")
    return {
        "method": method,
        "r": float(r),
        "p_value": float(p),
        "n": int(len(x)),
    }


def partial_correlation(df: pd.DataFrame, x: str, y: str,
                        covariates: Iterable[str], method: str = "pearson") -> Dict[str, Any]:
    """Partial correlation of x and y after regressing out ``covariates``."""
    import pingouin as pg
    covars = list(covariates)
    clean = df[[x, y, *covars]].dropna()
    result = pg.partial_corr(data=clean, x=x, y=y, covar=covars, method=method)
    row = result.iloc[0]
    # pingouin naming varies by version: older uses 'p-val' and 'CI95%',
    # newer uses 'p_val' and 'CI95'. Probe both.
    p_key = "p_val" if "p_val" in row.index else "p-val"
    ci_key = "CI95" if "CI95" in row.index else "CI95%"
    ci = row[ci_key] if ci_key in row.index else None
    return {
        "method": f"partial_{method}",
        "r": float(row["r"]),
        "p_value": float(row[p_key]),
        "n": int(row["n"]),
        "covariates": covars,
        "ci_lower": float(ci[0]) if ci is not None and hasattr(ci, "__getitem__") else None,
        "ci_upper": float(ci[1]) if ci is not None and hasattr(ci, "__getitem__") else None,
    }


def correlation_matrix(df: pd.DataFrame, variables: List[str],
                       method: str = "pearson", fdr_method: str = "fdr_bh") -> Dict[str, Any]:
    """Full correlation matrix with FDR-adjusted p-values for off-diagonal entries."""
    from statsmodels.stats.multitest import multipletests
    sub = df[variables].dropna()
    n = len(sub)
    mat = {v: {w: 1.0 for w in variables} for v in variables}
    p_raw: Dict[str, Dict[str, float]] = {v: {w: 1.0 for w in variables} for v in variables}
    flat_p: List[float] = []
    flat_keys: List[tuple] = []
    for i, v in enumerate(variables):
        for w in variables[i + 1:]:
            r = correlate_pair(sub[v], sub[w], method=method)
            mat[v][w] = mat[w][v] = r["r"]
            p_raw[v][w] = p_raw[w][v] = r["p_value"]
            flat_p.append(r["p_value"])
            flat_keys.append((v, w))
    p_adj_mat: Dict[str, Dict[str, float]] = {v: {w: 1.0 for w in variables} for v in variables}
    if flat_p:
        _, p_adj, _, _ = multipletests(flat_p, method=fdr_method)
        for (v, w), adj in zip(flat_keys, p_adj):
            p_adj_mat[v][w] = p_adj_mat[w][v] = float(adj)
    return {
        "method": method,
        "n": int(n),
        "matrix": mat,
        "p_values": p_raw,
        "p_values_adjusted": p_adj_mat,
        "fdr_method": fdr_method,
    }
