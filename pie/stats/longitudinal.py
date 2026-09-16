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

    ``random_slopes`` adds per-group random slopes. Each must also be a fixed effect: a
    random slope is a group's deviation around the fixed slope.

    Fixed effects are REML estimates reported with β, SE, z, p and 95% CI; a categorical
    predictor gets one row per non-reference level (``"cohort[T.PD]"``). REML likelihoods
    cannot compare models whose fixed effects differ, so ``aic``, ``bic`` and
    ``ml_log_likelihood`` come from a maximum-likelihood refit of the same model;
    ``log_likelihood`` is the REML value.
    """
    slopes = list(random_slopes or [])
    unfixed = [s for s in slopes if s not in fixed_effects]
    if unfixed:
        raise ValueError(f"random_slopes {unfixed} must also be in fixed_effects: a random slope "
                         "is a per-group deviation around the fixed slope")
    clean = df[[outcome, group, *fixed_effects]].dropna().copy()
    formula = f"Q('{outcome}') ~ " + " + ".join(f"Q('{fe}')" for fe in fixed_effects)
    re_formula = "~" + " + ".join(f"Q('{rs}')" for rs in slopes) if slopes else None

    def fit(reml):
        return smf.mixedlm(formula, data=clean, groups=clean[group],
                           re_formula=re_formula).fit(reml=reml, disp=False)
    model, ml = fit(True), fit(False)

    conf = model.conf_int()
    rows: List[Dict[str, Any]] = []
    for fe in fixed_effects:
        term = f"Q('{fe}')"
        for key in model.fe_params.index:
            # Exact term, or one of its levels (Q('cohort')[T.PD]); never a substring match.
            if key != term and not key.startswith(term + "["):
                continue
            rows.append({
                "predictor": fe + key[len(term):],
                "estimate": float(model.params[key]),
                "std_error": float(model.bse[key]),
                "z_statistic": float(model.tvalues[key]),
                "p_value": float(model.pvalues[key]),
                "ci_lower": float(conf.loc[key, 0]),
                "ci_upper": float(conf.loc[key, 1]),
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
        "ml_log_likelihood": float(ml.llf),
        "aic": float(ml.aic),
        "bic": float(ml.bic),
    }


def change_from_baseline(df: pd.DataFrame, subject: str, time: str,
                         outcome: str, baseline_time: Any = 0) -> Dict[str, Any]:
    """Compute per-subject change from baseline and summarize by time point.

    Needs one row per (subject, time): duplicates raise, because which of two baselines to
    subtract is an analysis decision, not something to settle by row order. Subjects with
    no baseline row are counted in ``n_without_baseline`` and left out of
    ``summary_by_time``; their rows stay in ``per_subject`` with NaN change.
    """
    clean = df[[subject, time, outcome]].dropna().copy()
    duplicated = clean.duplicated([subject, time], keep=False)
    if duplicated.any():
        raise ValueError(f"{clean.loc[duplicated, subject].nunique()} subject(s) have more than one "
                         f"row at the same {time!r}; deduplicate before computing change")
    baselines = clean[clean[time] == baseline_time].set_index(subject)[outcome]
    clean = clean.join(baselines.rename("baseline"), on=subject)
    clean["change"] = clean[outcome] - clean["baseline"]
    clean["pct_change"] = 100.0 * clean["change"] / clean["baseline"].replace(0, np.nan)
    has_baseline = clean["baseline"].notna()

    summary: Dict[Any, Dict[str, float]] = {}
    for t, g in clean[has_baseline].groupby(time):
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
        "n_without_baseline": int(clean.loc[~has_baseline, subject].nunique()),
        "baseline_time": baseline_time,
        "summary_by_time": summary,
        "per_subject": per_subject,
    }
