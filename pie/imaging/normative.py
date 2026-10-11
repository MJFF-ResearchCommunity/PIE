"""normative.py — deviation (z) scores against a reference group, e.g. PPMI healthy controls.

Per feature, a linear model (intercept, the caller's numeric covariates, and an age term when an ``age`` covariate is
given: age^2 by default, a natural cubic spline with ``age_terms="spline"``) is fitted on the reference rows only;
z = (observed - predicted) / residual SD of the reference. Encode categorical covariates (sex) as numbers first.

Options for a multi-site reference (the consortium, PPMI's ~50 sites):
- ``site``: a column of scanner/site labels. Each site's offset is its reference rows' mean residual, shrunk toward 0
  by empirical Bayes (n / (n + sigma^2 / tau^2)); a site without reference rows gets no offset. This is the site
  adaptation of hierarchical normative models (Kia et al. 2021) in its simplest form.
- ``heteroscedastic``: the residual SD varies with age (log|residual| linear in age), as brain-volume variance does.
- ``crossfit_zscores``: reference rows get out-of-fold z (k folds), so controls are never scored by a model that
  saw them; their z is then a fair null for patients' z.

Fit it on the training controls inside each cross-validation fold so no evaluation row informs it. Pretrained
lifespan models (Rutherford et al. 2022, doi:10.7554/eLife.72904; 57k scans) need pcntoolkit and are not wrapped.

    model = fit(frame, ["vol_Left_Putamen"], ["age", "sex", "tiv_mm3"], reference=frame.COHORT.eq("Healthy Control"))
    z = zscores(frame, model)            # z_vol_Left_Putamen
    z = crossfit_zscores(frame, features, ["age", "sex", "tiv_mm3"], ref, site="scanner_id", age_terms="spline", heteroscedastic=True)
"""
import numpy as np
import pandas as pd

KNOTS = (55.0, 65.0, 75.0)          # natural cubic spline knots for age (years); boundary at the data's range


def _ncs(age, knots, lo, hi):
    """Natural cubic spline basis (truncated-power form, Hastie et al. 2009 eq. 5.4-5.5) beyond the linear term."""
    k = np.r_[lo, list(knots), hi]
    d = lambda j: (np.clip(age - k[j], 0, None) ** 3 - np.clip(age - k[-1], 0, None) ** 3) / (k[-1] - k[j])
    return np.column_stack([d(j) - d(len(k) - 2) for j in range(len(k) - 2)])


def _design(frame, covariates, age_terms="quadratic", age_range=None):
    columns = [np.ones(len(frame))] + [frame[c].to_numpy(float) for c in covariates]
    if "age" in covariates:
        age = frame["age"].to_numpy(float)
        if age_terms == "spline":
            lo, hi = age_range
            columns += list(_ncs(np.clip(age, lo, hi), KNOTS, lo, hi).T)
        else:
            columns.append((age - 65.0) ** 2)
    return np.column_stack(columns)


def fit(frame, features, covariates, reference, site=None, age_terms="quadratic", heteroscedastic=False):
    """{"covariates", "age_terms", "age_range", "site", "features": {feature: {...}}}, fitted on ``reference`` rows."""
    ref = np.asarray(reference, bool)
    age_range = (float(np.nanmin(frame.loc[ref, "age"])), float(np.nanmax(frame.loc[ref, "age"]))) if "age" in covariates else None
    X = _design(frame, covariates, age_terms, age_range)
    model = {"covariates": list(covariates), "age_terms": age_terms, "age_range": age_range, "site": site, "features": {}}
    for f in features:
        y = frame[f].to_numpy(float)
        ok = ref & np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() <= X.shape[1] + 2:
            raise ValueError(f"{f}: too few reference rows ({int(ok.sum())}) for {X.shape[1]} parameters")
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        resid = y[ok] - X[ok] @ beta
        offsets = {}
        if site is not None:       # empirical-Bayes site intercepts from the reference residuals
            s = frame.loc[ok, site].astype(str).to_numpy()
            g = pd.Series(resid).groupby(s)
            within = float(np.var(resid - g.transform("mean").to_numpy(), ddof=1))
            tau2 = max(float(g.mean().var(ddof=1)) - within / max(g.size().mean(), 1), 0.0) if g.ngroups > 1 else 0.0
            offsets = {k: float(m * n / (n + within / tau2)) if tau2 > 0 else 0.0 for k, m, n in zip(g.mean().index, g.mean(), g.size())}
            resid = resid - pd.Series(s).map(offsets).to_numpy()
        entry = {"beta": beta, "sd": float(resid.std(ddof=X.shape[1])), "site_offsets": offsets, "sd_age": None}
        if heteroscedastic and "age" in covariates:
            a = frame.loc[ok, "age"].to_numpy(float)
            lab = np.log(np.abs(resid) + 1e-12)
            slope, icpt = np.polyfit(a, lab, 1)
            c = float(np.sqrt(np.mean(resid ** 2 / np.exp(2 * (icpt + slope * a)))))   # calibrates exp(.) to an SD
            entry["sd_age"] = (float(icpt), float(slope), c)
        model["features"][f] = entry
    return model


def _sd(frame, model, entry):
    if entry["sd_age"] is None:
        return entry["sd"]
    icpt, slope, c = entry["sd_age"]
    lo, hi = model["age_range"]
    return c * np.exp(icpt + slope * np.clip(frame["age"].to_numpy(float), lo, hi))


def zscores(frame, model):
    """z_<feature> for every row of ``frame`` (NaN where a covariate or the feature is missing)."""
    X = _design(frame, model["covariates"], model.get("age_terms", "quadratic"), model.get("age_range"))
    out = {}
    for f, e in model["features"].items():
        if not isinstance(e, dict):        # models from before October 2026: (beta, sd)
            e = {"beta": e[0], "sd": e[1], "site_offsets": {}, "sd_age": None}
        pred = X @ e["beta"]
        if model.get("site") is not None:
            pred = pred + frame[model["site"]].astype(str).map(e["site_offsets"]).fillna(0.0).to_numpy()
        out[f"z_{f}"] = (frame[f].to_numpy(float) - pred) / _sd(frame, model, e)
    return pd.DataFrame(out, index=frame.index)


def crossfit_zscores(frame, features, covariates, reference, k=5, seed=0, **options):
    """z for every row: reference rows out-of-fold (``k`` folds, each scored by a model fitted on the other reference
    rows), all other rows by the model fitted on every reference row. ``options`` go to ``fit``."""
    ref = np.asarray(reference, bool)
    z = zscores(frame, fit(frame, features, covariates, ref, **options))
    idx = np.flatnonzero(ref)
    folds = np.random.default_rng(seed).permutation(len(idx)) % k
    for j in range(k):
        held = np.zeros(len(frame), bool)
        held[idx[folds == j]] = True
        zj = zscores(frame.loc[held], fit(frame, features, covariates, ref & ~held, **options))
        z.loc[frame.index[held]] = zj.to_numpy()
    return z


def mtl_preservation(z, threshold=-1.0):
    """Relative preservation of the medial temporal lobe, the DLB-supportive imaging biomarker (McKeith et al. 2017):
    ``mtl_z`` = mean normative z of the hippocampi and amygdalae (columns ``z_*Hippocampus``/``z_*Amygdala`` from
    ``zscores``; adjust for head size, age and sex in the model) and ``mtl_preserved`` = mtl_z above ``threshold``."""
    cols = [c for c in z.columns if c.startswith("z_") and ("Hippocampus" in c or "Amygdala" in c)]
    if not cols:
        raise ValueError("no hippocampus/amygdala z columns")
    m = z[cols].mean(axis=1, skipna=False)
    return pd.DataFrame({"mtl_z": m, "mtl_preserved": (m > threshold).where(m.notna())}, index=z.index)
