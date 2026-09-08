"""Longitudinal / repeated-measures analysis."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def linear_mixed_model(df: pd.DataFrame, outcome: str,
                       fixed_effects: List[str], group: str,
                       random_slopes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fit a linear mixed-effects model with random intercept per ``group``.

    ``random_slopes`` adds per-group random slopes for the listed predictors.
    Fixed effects reported with β, SE, z, p, and 95% CI.
    """
    clean = df[[outcome, group, *fixed_effects]].dropna().copy()
    formula = f"Q('{outcome}') ~ " + " + ".join(f"Q('{fe}')" for fe in fixed_effects)
    re_formula = None
    if random_slopes:
        re_formula = "~" + " + ".join(f"Q('{rs}')" for rs in random_slopes)
    model = smf.mixedlm(formula, data=clean, groups=clean[group], re_formula=re_formula).fit(disp=False)

    rows: List[Dict[str, Any]] = []
    for fe in fixed_effects:
        # statsmodels mangles Q('...') into the param name; pick the matching key
        key = next((k for k in model.params.index if fe in k), None)
        if key is None:
            continue
        conf = model.conf_int().loc[key]
        rows.append({
            "predictor": fe,
            "estimate": float(model.params[key]),
            "std_error": float(model.bse[key]),
            "z_statistic": float(model.tvalues[key]),
            "p_value": float(model.pvalues[key]),
            "ci_lower": float(conf[0]),
            "ci_upper": float(conf[1]),
        })

    re_var = float(model.cov_re.iloc[0, 0]) if model.cov_re.size else float("nan")
    return {
        "model": "lmm",
        "n_obs": int(len(clean)),
        "n_groups": int(clean[group].nunique()),
        "fixed_effects": rows,
        "random_effect_variance": re_var,
        "residual_variance": float(model.scale),
        "log_likelihood": float(model.llf),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }


def change_from_baseline(df: pd.DataFrame, subject: str, time: str,
                         outcome: str, baseline_time: Any = 0) -> Dict[str, Any]:
    """Compute per-subject change from baseline and summarize by time point.

    Returns a per-subject tidy table plus summary stats at each follow-up visit.
    """
    clean = df[[subject, time, outcome]].dropna().copy()
    baselines = clean[clean[time] == baseline_time].set_index(subject)[outcome]
    clean = clean.join(baselines.rename("baseline"), on=subject)
    clean["change"] = clean[outcome] - clean["baseline"]
    clean["pct_change"] = 100.0 * clean["change"] / clean["baseline"].replace(0, np.nan)

    summary: Dict[Any, Dict[str, float]] = {}
    for t, g in clean.groupby(time):
        if t == baseline_time:
            continue
        # Convert numpy ints to plain Python ints for clean JSON serialization
        key = int(t) if isinstance(t, (np.integer,)) else t
        summary[key] = {
            "n": int(len(g)),
            "mean_change": float(g["change"].mean()),
            "sd_change": float(g["change"].std(ddof=1)) if len(g) > 1 else float("nan"),
            "mean_pct_change": float(g["pct_change"].mean(skipna=True)),
        }
    per_subject = clean[[subject, time, outcome, "change", "pct_change"]].to_dict(orient="records")
    return {
        "n_subjects": int(clean[subject].nunique()),
        "baseline_time": baseline_time,
        "changes": {int(t) if isinstance(t, (np.integer,)) else t: None for t in clean[time].unique()},
        "summary_by_time": summary,
        "per_subject": per_subject,
    }
