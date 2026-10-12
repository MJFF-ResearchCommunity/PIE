"""harmonize.py — scanner/site harmonisation of imaging features, and the batch x label confounding report.

Two modes, and they must not be mixed:
- inference: fit on every scan with the outcome protected (`label=`). The batch x label confounding report is
  logged and kept in the model. The harmonised values depend on every participant's label, so they must not feed
  a classifier that is then judged on those participants.
- prediction: inside each cross-validation fold, fit on the training rows without the label and apply the model
  to the held-out rows with `combat_apply`.

    out, model = combat(frame, features, "site", ["age", "sex"], label="saa")      # inference
    train_h, model = combat(train, features, "site", ["age", "sex"])                # prediction, in-fold
    test_h = combat_apply(model, test)
    long_h, est = long_combat(visits, features, "scanner", "PATNO", "years", ["age_bl", "sex"], label="saa")
    report = confounding_report(frame, "site", "saa", ["age", "sex"])

No method separates a batch effect from a label effect when label is (nearly) constant within batches; the report
says where that is. ComBat: Johnson 2007 (doi:10.1093/biostatistics/kxj037), Fortin 2018
(doi:10.1016/j.neuroimage.2017.11.024); the numerics match neuroCombat 0.2.12. Spline covariates: after ComBat-GAM
(Pomponio 2020, doi:10.1016/j.neuroimage.2019.116450), but an unpenalised cubic B-spline in the design rather than a
penalised GAM. Longitudinal: Beer 2020 (doi:10.1016/j.neuroimage.2020.117129), ported from the R package longCombat.
"""
import logging

import numpy as np
import pandas as pd
import patsy
from scipy.stats.contingency import association
from statsmodels.regression.mixed_linear_model import MixedLM

log = logging.getLogger(__name__)


def _values(df, features):
    y = df[list(features)].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("Missing or infinite feature values: impute or drop them before harmonising")
    return y


def _batches(df, batch, levels=None):
    """Sorted batch levels and each row's index into them; unseen levels are an error."""
    if df[batch].isna().any():
        raise ValueError(f"{int(df[batch].isna().sum())} rows have no {batch!r}: label them (or drop them) before harmonising")
    values = df[batch].astype(str).to_numpy()
    levels = np.unique(values) if levels is None else levels
    unseen = sorted(set(values) - set(levels))
    if unseen:
        raise ValueError(f"Batches not in the fitted model: {unseen}. Their batch effects were never estimated; "
                         "refit with them in the training data or drop those rows")
    return levels, np.searchsorted(levels, values)


def _fit_checks(x, idx):
    if np.bincount(idx).min() < 2:
        raise ValueError("Every batch needs at least 2 rows to estimate its variance")
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("Batch is collinear with the protected covariates (e.g. a label constant within every "
                         "batch): no harmonisation can separate them")


def _report(df, batch, label, covariates, min_per_class):
    """Inference mode: compute, log and return the confounding report."""
    report = confounding_report(df, batch, label, [c for c in covariates if c in df], min_per_class)
    log.warning("Harmonisation protecting %r (inference mode). Batch x label Cramér's V = %.2f; batches where "
                "batch and label cannot be separated (single-label or < %d of a class): %s\n%s",
                label, report["cramers_v"], min_per_class, report["flagged"] or "none", report["table"].to_string())
    return report


def _eb(z, g_hat, d_hat, conv=1e-4):
    """Parametric empirical Bayes for one batch (sva / neuroCombat `it_sol`), priors pooled over features."""
    n = len(z)
    g_bar, t2 = g_hat.mean(), g_hat.var(ddof=1)
    m, s2 = d_hat.mean(), d_hat.var(ddof=1)
    a, b = (2 * s2 + m ** 2) / s2, (m * s2 + m ** 3) / s2
    g_old, d_old = g_hat, d_hat
    while True:
        g = (t2 * n * g_hat + d_old * g_bar) / (t2 * n + d_old)
        d = (0.5 * ((z - g) ** 2).sum(0) + b) / (n / 2 + a - 1)
        change = max((abs(g - g_old) / g_old).max(), (abs(d - d_old) / d_old).max())  # as sva: signed denominator
        g_old, d_old = g, d
        if change <= conv:
            return g, d


def combat(df, features, batch, covariates=(), nonlinear=("age",), label=None, eb=True, spline_df=4,
           min_per_class=5):
    """ComBat with protected covariates; those in `nonlinear` enter as a cubic B-spline with `spline_df` df.

    Returns `(copy of df with harmonised features, model)`. `combat_apply(model, new)` harmonises new rows with
    the same estimates. `covariates` are column names or patsy terms; strings and booleans are categorical, so wrap
    numerically coded multi-level categories in `C()`. `label` (a column name, always categorical) is protected
    too, which is inference mode: the confounding report is logged and kept in `model["confounding"]`.
    Feature values must be complete.
    """
    covariates = list(covariates)
    missing = sorted(set(nonlinear) - set(covariates))
    if missing:
        raise ValueError(f"nonlinear {missing} not among covariates; pass nonlinear=() for a linear model")
    terms = [f"bs({c}, df={spline_df})" if c in nonlinear else c for c in covariates]
    design = patsy.dmatrix(" + ".join(["1", *terms] + ([f"C({label})"] if label else [])), df,
                           NA_action="raise", return_type="dataframe")
    y = _values(df, features)
    levels, idx = _batches(df, batch)
    k = len(levels)
    x = np.column_stack([np.eye(k)[idx], design.to_numpy()[:, 1:]])   # batch one-hot replaces the intercept
    _fit_checks(x, idx)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    grand = np.bincount(idx) / len(y) @ beta[:k]
    var = ((y - x @ beta) ** 2).mean(0)
    z = (y - grand - x[:, k:] @ beta[k:]) / np.sqrt(var)
    gamma = np.array([z[idx == i].mean(0) for i in range(k)])
    delta = np.array([z[idx == i].var(0, ddof=1) for i in range(k)])
    if eb:
        gamma, delta = map(np.array, zip(*[_eb(z[idx == i], gamma[i], delta[i]) for i in range(k)]))
    model = {"features": list(features), "batch": batch, "levels": levels, "design": design.design_info,
             "ranges": {c: (df[c].min(), df[c].max()) for c in nonlinear}, "beta": beta[k:], "grand": grand,
             "var": var, "gamma": gamma, "delta": delta, "label": label}
    if label:
        model["confounding"] = _report(df, batch, label, covariates, min_per_class)
    return _apply(model, df), model


def _apply(model, df):
    _, idx = _batches(df, model["batch"], model["levels"])
    # Spline covariates are clipped to the fitted range: constant extrapolation beyond it.
    clipped = df.assign(**{c: df[c].clip(lo, hi) for c, (lo, hi) in model["ranges"].items()})
    mod = patsy.build_design_matrices([model["design"]], clipped, NA_action="raise")[0][:, 1:] @ model["beta"]
    sd = np.sqrt(model["var"])
    z = (_values(df, model["features"]) - model["grand"] - mod) / sd
    out = df.copy()
    out[model["features"]] = (z - model["gamma"][idx]) / np.sqrt(model["delta"][idx]) * sd + model["grand"] + mod
    return out


def combat_apply(model, df):
    """Harmonise new rows with a model from `combat` (prediction mode: fit on the training fold, apply to the
    held-out fold). Their covariates are needed, their outcome is not. Unseen batches are an error."""
    if model["label"]:
        raise ValueError(f"Model protects the outcome {model['label']!r} (inference mode), so applying it would "
                         "need the held-out labels. Refit inside the training fold without label=")
    return _apply(model, df)


def long_combat(df, features, batch, subject, time, covariates=(), label=None, ranef="1", niter=30,
                min_per_class=5):
    """Longitudinal ComBat (Beer 2020), a port of the R package longCombat with its default method="REML".

    Per feature, a REML linear mixed model `feature ~ time + covariates [+ C(label)] + batch` with random effects
    `ranef` ("1": subject intercept; "1 + years": intercept and slope) by `subject` is the location model. Its
    residuals plus the batch effect, centred so that sum(n_batch * effect) = 0 and divided by the residual SD, are
    the standardised data on which batch location and scale are estimated by empirical Bayes (`niter` fixed
    iterations, as in R). A participant's batch may change between visits. Interactions go in `covariates` as
    patsy terms, e.g. "C(saa):years". `label` is inference mode, as in `combat`. Returns `(copy of df with
    harmonised features, estimates)`; estimates holds R's gammahat, delta2hat, gammastarhat, delta2starhat.
    """
    covariates = list(covariates)
    y = _values(df, features)
    levels, idx = _batches(df, batch)
    m, n_i = len(levels), np.bincount(idx)
    fixed = patsy.dmatrix(" + ".join(["1", time, *covariates] + ([f"C({label})"] if label else [])), df,
                          NA_action="raise")
    exog = np.column_stack([fixed, np.eye(m)[idx][:, 1:]])
    _fit_checks(exog, idx)
    exog_re = patsy.dmatrix(ranef, df, NA_action="raise")
    sigma, predicted, effects = [], [], []
    for v in range(y.shape[1]):
        fit = MixedLM(y[:, v], exog, df[subject].to_numpy(), exog_re).fit(reml=True)
        sigma.append(np.sqrt(fit.scale))           # residual SD, lme4 VarCorr "Residual"
        predicted.append(fit.fittedvalues)         # fixed effects + predicted random effects, as lme4 fitted()
        effects.append(np.r_[0, fit.fe_params[fixed.shape[1]:]])
    sigma, predicted, effects = np.array(sigma), np.array(predicted).T, np.array(effects).T
    effects -= n_i @ effects / len(y)
    z = (y - predicted + effects[idx]) / sigma
    g_hat = np.array([z[idx == i].mean(0) for i in range(m)])
    d_hat = np.array([z[idx == i].var(0, ddof=1) for i in range(m)])
    d_bar, s2 = d_hat.mean(1), d_hat.var(1, ddof=1)
    n, g_bar, t2, lam, theta = (a[:, None] for a in (n_i, g_hat.mean(1), g_hat.var(1, ddof=1),
                                                     (d_bar ** 2 + 2 * s2) / s2, (d_bar ** 3 + d_bar * s2) / s2))

    def delta_star(g):
        ss = np.array([((z[idx == i] - g[i]) ** 2).sum(0) for i in range(m)])
        return (theta + 0.5 * ss) / (n / 2 + lam - 1)

    g = (n * t2 * g_hat + d_hat * g_bar) / (n * t2 + d_hat)
    d = delta_star(g)
    for _ in range(niter):   # as R: each delta update uses the previous iteration's gamma
        g, d = (n * t2 * g_hat + d * g_bar) / (n * t2 + d), delta_star(g)
    out = df.copy()
    out[list(features)] = sigma / np.sqrt(d[idx]) * (z - g[idx]) + predicted - effects[idx]
    frame = lambda a: pd.DataFrame(a, index=levels, columns=list(features))
    estimates = {"gammahat": frame(g_hat), "delta2hat": frame(d_hat), "gammastarhat": frame(g),
                 "delta2starhat": frame(d)}
    if label:
        estimates["confounding"] = _report(df, batch, label, covariates, min_per_class)
    return out, estimates


def confounding_report(df, batch, label, covariates=(), min_per_class=5):
    """How entangled batch and label are: read it before trusting any harmonisation that protects the label.

    Returns a dict: `table`, the batch x label counts; `sites`, per batch: n, label proportions `p_<class>`,
    `single_label`, `sparse` (a class with fewer than `min_per_class` rows) and covariate balance (means of
    numeric covariates, proportions of each level of categorical ones); `cramers_v` (0: independent, 1: label
    determined by batch); and `flagged`, the batches where a batch effect cannot be told from a label effect.
    """
    table = pd.crosstab(df[batch], df[label])
    sites = pd.DataFrame({"n": table.sum(1)}).join(table.div(table.sum(1), axis=0).add_prefix("p_"))
    sites["single_label"] = (table > 0).sum(1) == 1
    sites["sparse"] = (table < min_per_class).any(axis=1)
    if covariates:
        sites = sites.join(pd.get_dummies(df[list(covariates)], dtype=float).groupby(df[batch]).mean())
    v = association(table.to_numpy(), method="cramer") if min(table.shape) > 1 else np.nan
    flagged = sites.index[sites["single_label"] | sites["sparse"]].tolist()
    return {"table": table, "sites": sites, "cramers_v": v, "flagged": flagged}
