"""Regression: linear, logistic, ANCOVA with diagnostics."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


def _fit_summary(predictors: List[str], params, bse, tvals, pvals, conf_int) -> List[Dict[str, Any]]:
    rows = []
    for p in predictors:
        rows.append({
            "predictor": p,
            "estimate": float(params[p]),
            "std_error": float(bse[p]),
            "t_statistic": float(tvals[p]),
            "p_value": float(pvals[p]),
            "ci_lower": float(conf_int.loc[p, 0]),
            "ci_upper": float(conf_int.loc[p, 1]),
        })
    return rows


def linear_regression(df: pd.DataFrame, outcome: str, predictors: List[str],
                      standardize: bool = False) -> Dict[str, Any]:
    """OLS linear regression with VIF, Durbin-Watson, fitted / residual arrays."""
    clean = df[[outcome, *predictors]].dropna().copy()
    X = clean[predictors].copy()
    y = clean[outcome]
    if standardize:
        X = (X - X.mean()) / X.std(ddof=1)
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const).fit()

    vif: Dict[str, float] = {}
    if len(predictors) >= 2:
        for i, p in enumerate(predictors):
            vif[p] = float(variance_inflation_factor(X_const.values, i + 1))
    else:
        vif = {predictors[0]: float("nan")}

    return {
        "model": "ols",
        "n": int(len(clean)),
        "coefficients": _fit_summary(predictors, model.params, model.bse,
                                     model.tvalues, model.pvalues, model.conf_int()),
        "intercept": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_p_value": float(model.f_pvalue),
        "diagnostics": {
            "vif": vif,
            "durbin_watson": float(durbin_watson(model.resid)),
            "residuals": model.resid.tolist(),
            "fitted": model.fittedvalues.tolist(),
            "standardized_residuals": (model.resid / model.resid.std(ddof=1)).tolist(),
        },
    }


def logistic_regression(df: pd.DataFrame, outcome: str, predictors: List[str]) -> Dict[str, Any]:
    """Binary logistic regression with ORs, 95% CIs, and training-set ROC."""
    from sklearn.metrics import roc_auc_score, roc_curve
    clean = df[[outcome, *predictors]].dropna().copy()
    y = clean[outcome].astype(int)
    X = sm.add_constant(clean[predictors], has_constant="add")
    model = sm.Logit(y, X).fit(disp=False)
    conf = model.conf_int()

    rows: List[Dict[str, Any]] = []
    for p in predictors:
        beta = float(model.params[p])
        lo, hi = float(conf.loc[p, 0]), float(conf.loc[p, 1])
        rows.append({
            "predictor": p,
            "estimate": beta,
            "std_error": float(model.bse[p]),
            "z_statistic": float(model.tvalues[p]),
            "p_value": float(model.pvalues[p]),
            "odds_ratio": float(np.exp(beta)),
            "or_ci_lower": float(np.exp(lo)),
            "or_ci_upper": float(np.exp(hi)),
        })

    probs = model.predict(X)
    fpr, tpr, _ = roc_curve(y, probs)
    # Downsample ROC points for display — full 500-row training sets generate
    # hundreds of knots, bloating the JSON response with no visual gain.
    idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
    return {
        "model": "logit",
        "n": int(len(clean)),
        "coefficients": rows,
        "intercept": float(model.params["const"]),
        "pseudo_r2": float(model.prsquared),
        "log_likelihood": float(model.llf),
        "auc": float(roc_auc_score(y, probs)),
        "roc_curve": {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()},
    }


def ancova(df: pd.DataFrame, outcome: str, group: str,
           covariates: List[str]) -> Dict[str, Any]:
    """One-way ANCOVA via statsmodels; returns type-II ANOVA table."""
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    clean = df[[outcome, group, *covariates]].dropna().copy()
    # Quote column names so e.g. "age.at.visit" or column names with spaces work.
    formula = f"Q('{outcome}') ~ C(Q('{group}'))"
    for c in covariates:
        formula += f" + Q('{c}')"
    model = ols(formula, data=clean).fit()
    table = anova_lm(model, typ=2)
    effects: List[Dict[str, Any]] = []
    for source, row in table.iterrows():
        source_str = str(source)
        if source_str.startswith("C("):
            clean_name = "group"
        else:
            clean_name = source_str.replace("Q('", "").replace("')", "")
        effects.append({
            "source": clean_name,
            "sum_sq": float(row["sum_sq"]),
            "df": float(row["df"]),
            "f_statistic": float(row["F"]) if not np.isnan(row["F"]) else None,
            "p_value": float(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else None,
        })
    return {
        "model": "ancova",
        "n": int(len(clean)),
        "effects": effects,
        "r_squared": float(model.rsquared),
    }
