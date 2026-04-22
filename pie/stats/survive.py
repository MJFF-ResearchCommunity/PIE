"""Survival analysis: KM, log-rank, Cox PH."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def kaplan_meier(df: pd.DataFrame, time: str, event: str,
                 group: Optional[str] = None) -> Dict[str, Any]:
    """Kaplan-Meier survival estimates, optionally stratified by ``group``."""
    from lifelines import KaplanMeierFitter

    cols = [time, event] + ([group] if group else [])
    clean = df[cols].dropna()
    timeline = np.linspace(0, float(clean[time].max()), 100)

    curves: Dict[str, List[float]] = {}
    ci_lo: Dict[str, List[float]] = {}
    ci_hi: Dict[str, List[float]] = {}

    if group:
        for lvl, g in clean.groupby(group):
            km = KaplanMeierFitter().fit(g[time], g[event], timeline=timeline)
            curves[str(lvl)] = km.survival_function_.iloc[:, 0].tolist()
            ci = km.confidence_interval_
            ci_lo[str(lvl)] = ci.iloc[:, 0].tolist()
            ci_hi[str(lvl)] = ci.iloc[:, 1].tolist()
    else:
        km = KaplanMeierFitter().fit(clean[time], clean[event], timeline=timeline)
        curves["_overall"] = km.survival_function_.iloc[:, 0].tolist()
        ci = km.confidence_interval_
        ci_lo["_overall"] = ci.iloc[:, 0].tolist()
        ci_hi["_overall"] = ci.iloc[:, 1].tolist()

    return {
        "timeline": timeline.tolist(),
        "survival": curves,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
    }


def logrank_test(df: pd.DataFrame, time: str, event: str, group: str) -> Dict[str, Any]:
    """Log-rank test comparing survival across groups."""
    from lifelines.statistics import logrank_test as _lr, multivariate_logrank_test as _mlr
    clean = df[[time, event, group]].dropna()
    levels = clean[group].unique()
    if len(levels) == 2:
        a = clean[clean[group] == levels[0]]
        b = clean[clean[group] == levels[1]]
        res = _lr(a[time], b[time], a[event], b[event])
    else:
        res = _mlr(clean[time], clean[group], event_observed=clean[event])
    return {
        "test": "logrank",
        "statistic": float(res.test_statistic),
        "p_value": float(res.p_value),
        "n_groups": int(len(levels)),
    }


def cox_regression(df: pd.DataFrame, time: str, event: str,
                   covariates: List[str]) -> Dict[str, Any]:
    """Cox proportional-hazards regression with Schoenfeld PH diagnostics."""
    from lifelines import CoxPHFitter

    clean = df[[time, event, *covariates]].dropna().copy()
    cph = CoxPHFitter().fit(clean, duration_col=time, event_col=event)

    rows: List[Dict[str, Any]] = []
    for cov in covariates:
        beta = float(cph.params_[cov])
        se = float(cph.standard_errors_[cov])
        rows.append({
            "predictor": cov,
            "coef": beta,
            "hazard_ratio": float(np.exp(beta)),
            "se": se,
            "z_statistic": beta / se if se > 0 else float("nan"),
            "p_value": float(cph.summary.loc[cov, "p"]),
            "hr_ci_lower": float(np.exp(cph.confidence_intervals_.loc[cov, "95% lower-bound"])),
            "hr_ci_upper": float(np.exp(cph.confidence_intervals_.loc[cov, "95% upper-bound"])),
        })

    # Schoenfeld PH test per covariate. lifelines' check_assumptions returns a
    # list of StatisticalResult objects; catch errors because it can be finicky
    # with perfectly-separable predictors.
    ph_rows: List[Dict[str, Any]] = []
    try:
        from lifelines.statistics import proportional_hazard_test
        ph_res = proportional_hazard_test(cph, clean, time_transform="rank")
        for cov in covariates:
            if cov in ph_res.summary.index:
                ph_rows.append({
                    "predictor": cov,
                    "test_statistic": float(ph_res.summary.loc[cov, "test_statistic"]),
                    "p_value": float(ph_res.summary.loc[cov, "p"]),
                    "violates_ph": bool(ph_res.summary.loc[cov, "p"] < 0.05),
                })
    except Exception:
        pass

    return {
        "model": "cox_ph",
        "n": int(len(clean)),
        "n_events": int(clean[event].sum()),
        "coefficients": rows,
        "concordance": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_),
        "ph_test": ph_rows,
    }
