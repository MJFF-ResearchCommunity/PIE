"""normative.py — deviation (z) scores against a reference group, e.g. PPMI healthy controls.

Per feature, a linear model (intercept, the caller's numeric covariates, and age^2 when an ``age`` covariate is given)
is fitted on the reference rows only; z = (observed - predicted) / residual SD of the reference. Fit it on the training
controls inside each cross-validation fold so that no evaluation row informs it. Encode categorical covariates (sex,
scanner batch) as numbers or one-hot columns before calling. Pretrained lifespan models (Rutherford et al. 2022,
doi:10.7554/eLife.72904) need FreeSurfer Destrieux thickness, which PIE only has if the surface stream runs.

    model = fit(frame, ["vol_Left_Putamen"], ["age", "sex", "tiv_mm3"], reference=frame.COHORT.eq("Healthy Control"))
    z = zscores(frame, model)            # z_vol_Left_Putamen
"""
import numpy as np
import pandas as pd


def _design(frame, covariates):
    columns = [np.ones(len(frame))] + [frame[c].to_numpy(float) for c in covariates]
    if "age" in covariates:
        columns.append((frame["age"].to_numpy(float) - 65.0) ** 2)
    return np.column_stack(columns)


def fit(frame, features, covariates, reference):
    """{"covariates": [...], "features": {feature: (coefficients, residual SD)}}, fitted on ``reference`` rows."""
    ref = np.asarray(reference, bool)
    X = _design(frame, covariates)
    model = {"covariates": list(covariates), "features": {}}
    for f in features:
        y = frame[f].to_numpy(float)
        ok = ref & np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() <= X.shape[1] + 2:
            raise ValueError(f"{f}: too few reference rows ({int(ok.sum())}) for {X.shape[1]} parameters")
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        model["features"][f] = (beta, float((y[ok] - X[ok] @ beta).std(ddof=X.shape[1])))
    return model


def zscores(frame, model):
    """z_<feature> for every row of ``frame`` (NaN where a covariate or the feature is missing)."""
    X = _design(frame, model["covariates"])
    return pd.DataFrame({f"z_{f}": (frame[f].to_numpy(float) - X @ beta) / sd for f, (beta, sd) in model["features"].items()},
                        index=frame.index)
