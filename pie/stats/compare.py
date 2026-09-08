"""Group comparison tests and effect sizes."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats as _sps


_ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


def _clean(*arrays: _ArrayLike) -> tuple:
    """Convert to float ndarrays with NaNs dropped (per-array)."""
    out = []
    for a in arrays:
        arr = np.asarray(a, dtype=float)
        out.append(arr[~np.isnan(arr)])
    return tuple(out)


# ---------------------------------------------------------------------------
# Two-group tests
# ---------------------------------------------------------------------------

def independent_ttest(a: _ArrayLike, b: _ArrayLike) -> Dict[str, Any]:
    """Student's independent two-sample t-test (equal variance)."""
    x, y = _clean(a, b)
    t, p = _sps.ttest_ind(x, y, equal_var=True)
    return {
        "test": "independent_t",
        "statistic": float(t),
        "p_value": float(p),
        "df": int(len(x) + len(y) - 2),
        "n1": int(len(x)),
        "n2": int(len(y)),
        "mean1": float(np.mean(x)),
        "mean2": float(np.mean(y)),
        "cohens_d": cohens_d(x, y),
        "hedges_g": hedges_g(x, y),
    }


def welch_ttest(a: _ArrayLike, b: _ArrayLike) -> Dict[str, Any]:
    """Welch's t-test (unequal variance)."""
    x, y = _clean(a, b)
    result = _sps.ttest_ind(x, y, equal_var=False)
    return {
        "test": "welch_t",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "df": float(result.df),
        "n1": int(len(x)),
        "n2": int(len(y)),
        "mean1": float(np.mean(x)),
        "mean2": float(np.mean(y)),
        "cohens_d": cohens_d(x, y),
    }


def paired_ttest(a: _ArrayLike, b: _ArrayLike) -> Dict[str, Any]:
    """Paired t-test. Arrays must be equal length; NaN pairs are dropped."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"paired_ttest requires equal lengths, got {len(x)} vs {len(y)}")
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    t, p = _sps.ttest_rel(x, y)
    return {
        "test": "paired_t",
        "statistic": float(t),
        "p_value": float(p),
        "df": int(len(x) - 1),
        "n_pairs": int(len(x)),
        "mean_diff": float(np.mean(x - y)),
    }


def mann_whitney(a: _ArrayLike, b: _ArrayLike, alternative: str = "two-sided") -> Dict[str, Any]:
    x, y = _clean(a, b)
    u, p = _sps.mannwhitneyu(x, y, alternative=alternative)
    return {
        "test": "mann_whitney_u",
        "u_statistic": float(u),
        "p_value": float(p),
        "n1": int(len(x)),
        "n2": int(len(y)),
        "median1": float(np.median(x)) if len(x) else float("nan"),
        "median2": float(np.median(y)) if len(y) else float("nan"),
    }


def wilcoxon_signed_rank(a: _ArrayLike, b: _ArrayLike) -> Dict[str, Any]:
    """Wilcoxon signed-rank test for paired samples."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) != len(y):
        raise ValueError("wilcoxon requires equal lengths")
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    w, p = _sps.wilcoxon(x, y)
    return {
        "test": "wilcoxon_signed_rank",
        "statistic": float(w),
        "p_value": float(p),
        "n_pairs": int(len(x)),
    }


# ---------------------------------------------------------------------------
# Multi-group tests
# ---------------------------------------------------------------------------

def _clean_groups(groups: Dict[str, _ArrayLike]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in groups.items():
        arr = np.asarray(v, dtype=float)
        out[k] = arr[~np.isnan(arr)]
    return out


def one_way_anova(groups: Dict[str, _ArrayLike]) -> Dict[str, Any]:
    """One-way ANOVA with η² effect size."""
    cleaned = _clean_groups(groups)
    if len(cleaned) < 2:
        raise ValueError("need at least 2 groups")
    f, p = _sps.f_oneway(*cleaned.values())
    grand = np.concatenate(list(cleaned.values()))
    grand_mean = grand.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in cleaned.values())
    ss_total = ((grand - grand_mean) ** 2).sum()
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else float("nan")
    return {
        "test": "one_way_anova",
        "statistic": float(f),
        "p_value": float(p),
        "df_between": len(cleaned) - 1,
        "df_within": int(sum(len(g) for g in cleaned.values()) - len(cleaned)),
        "eta_squared": eta_sq,
        "n_per_group": {k: int(len(v)) for k, v in cleaned.items()},
        "mean_per_group": {k: float(np.mean(v)) for k, v in cleaned.items()},
    }


def kruskal_wallis(groups: Dict[str, _ArrayLike]) -> Dict[str, Any]:
    """Kruskal-Wallis rank-based multi-group test."""
    cleaned = _clean_groups(groups)
    h, p = _sps.kruskal(*cleaned.values())
    return {
        "test": "kruskal_wallis",
        "statistic": float(h),
        "p_value": float(p),
        "df": len(cleaned) - 1,
        "n_per_group": {k: int(len(v)) for k, v in cleaned.items()},
        "median_per_group": {k: float(np.median(v)) if len(v) else float("nan") for k, v in cleaned.items()},
    }


def tukey_hsd(groups: Dict[str, _ArrayLike]) -> Dict[str, Any]:
    """Tukey HSD post-hoc — all pairwise comparisons with family-wise error control."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    values: List[float] = []
    labels: List[str] = []
    for k, v in groups.items():
        arr = np.asarray(v, dtype=float)
        arr = arr[~np.isnan(arr)]
        values.extend(arr.tolist())
        labels.extend([k] * len(arr))
    res = pairwise_tukeyhsd(values, labels)
    pairs: List[Dict[str, Any]] = []
    for row in res.summary().data[1:]:
        pairs.append({
            "group1": str(row[0]),
            "group2": str(row[1]),
            "mean_diff": float(row[2]),
            "p_adj": float(row[3]),
            "lower": float(row[4]),
            "upper": float(row[5]),
            "reject": bool(row[6]),
        })
    return {"method": "tukey_hsd", "pairwise": pairs}


def dunn_posthoc(groups: Dict[str, _ArrayLike], p_adjust: str = "bonferroni") -> Dict[str, Any]:
    """Dunn's post-hoc with adjustable multiple-testing correction."""
    import scikit_posthocs as sp
    rows = []
    for k, v in groups.items():
        for x in np.asarray(v, dtype=float):
            if not np.isnan(x):
                rows.append({"group": k, "value": float(x)})
    df = pd.DataFrame(rows)
    mat = sp.posthoc_dunn(df, val_col="value", group_col="group", p_adjust=p_adjust)
    pairs: List[Dict[str, Any]] = []
    names = mat.columns.tolist()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append({
                "group1": str(a),
                "group2": str(b),
                "p_adj": float(mat.loc[a, b]),
                "reject": bool(mat.loc[a, b] < 0.05),
            })
    return {"method": f"dunn_{p_adjust}", "pairwise": pairs}


def eta_squared(groups: Dict[str, _ArrayLike]) -> float:
    """Effect size for one-way ANOVA."""
    return one_way_anova(groups)["eta_squared"]


# ---------------------------------------------------------------------------
# Categorical tests
# ---------------------------------------------------------------------------

def chi_square(table: List[List[int]]) -> Dict[str, Any]:
    """Chi-square test of independence on a contingency table."""
    chi2, p, dof, expected = _sps.chi2_contingency(np.asarray(table))
    return {
        "test": "chi_square",
        "statistic": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "expected": expected.tolist(),
    }


def fisher_exact(table: List[List[int]]) -> Dict[str, Any]:
    """Fisher's exact test for a 2×2 contingency table."""
    arr = np.asarray(table)
    if arr.shape != (2, 2):
        raise ValueError(f"Fisher's exact requires a 2×2 table, got {arr.shape}")
    odds, p = _sps.fisher_exact(arr)
    return {"test": "fisher_exact", "odds_ratio": float(odds), "p_value": float(p)}


def mcnemar(b: int, c: int, exact: bool = True) -> Dict[str, Any]:
    """McNemar's paired-binary test. b/c are off-diagonal discordant counts."""
    from statsmodels.stats.contingency_tables import mcnemar as _mc
    table = [[0, int(b)], [int(c), 0]]
    res = _mc(table, exact=exact)
    return {
        "test": "mcnemar",
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "b": int(b),
        "c": int(c),
    }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(a: _ArrayLike, b: _ArrayLike) -> float:
    """Cohen's d — standardized mean difference with pooled SD."""
    x, y = _clean(a, b)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = math.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def hedges_g(a: _ArrayLike, b: _ArrayLike) -> float:
    """Hedges' g — small-sample-bias-corrected Cohen's d."""
    d = cohens_d(a, b)
    if math.isnan(d):
        return float("nan")
    n = len(a) + len(b)
    return d * (1 - 3 / (4 * n - 9))
