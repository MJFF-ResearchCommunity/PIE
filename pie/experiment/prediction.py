"""Nested model selection whose preprocessing never sees a participant it will be judged on.

The failure this module exists to prevent is the ordinary one: a scaler, imputer,
feature-eligibility rule, PCA basis or nuisance regression fitted on all the data, so
that every later split is contaminated and the reported discrimination is partly a
memory of the test set. Here every transformation is a `fit`/`transform` pair fitted on
one partition and applied to the others, and the outer test outcome is not read until
the predictions are already saved.

The model search is a small, declared, finite grid — covariates alone versus covariates
plus one block of imaging or assay features, each optionally residualised for nuisance
and compressed by PCA, under logistic regression or gradient boosting. Winners are chosen
on inner-fold AUC with deterministic tie-breaking, so the same data give the same choice.

    from pie.experiment import prediction as pred

    predictions, audit = pred.nested_fold(frame, train_rows, test_rows,
                                          families={"t1": t1_columns},
                                          batch_cols=["scanner_batch"], icv="MaskVol",
                                          adjust_baseline=False, seed=20260909,
                                          outcome="saa_prodromal")
    table = pred.paired_metrics(out_of_fold_predictions)

`families` maps a name to its feature columns; `nested_fold` returns one prediction
vector per family plus `baseline` (covariates only) and `selected` (the grid winner),
which is what an increment-over-covariates comparison needs. `paired_metrics` then gives
the participant-clustered paired bootstrap of each family against that baseline.

Extracted from the PPMI prodromal-synucleinopathy study in `documentation/experiment.md`;
the numerics are unchanged from the frozen study engine.
"""

from dataclasses import dataclass, asdict
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer

SEEDS = [20260909, 20260919, 20260929]
COVARIATES = ["age_at_scan", "sex_male", "LRRK2_carrier", "GBA_carrier", "SNCA_carrier", "APOE_e4"]
C_VALUES = [0.01, 0.1, 1.0, 10.0]
DEFAULT_OUTCOME = "saa_prodromal"
PARTICIPANT = "PATNO"
# Measures of the nigra itself are not proportional to head size, so intracranial volume
# is a nuisance covariate for cortical and subcortical morphometry but not for these.
NON_ICV_PREFIXES = ("new_nm_", "new_dti_", "nm_", "dwi_")


def finite_numeric(frame, cols):
    """Requested columns as floats, with missing columns and infinities alike as NaN."""
    return (frame.reindex(columns=list(cols)).apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).to_numpy(float))


class CovariateDesign:
    """Demographic and genetic design: median imputation, standardisation, an age spline.

    `nuisance=True` narrows it to age and sex, which is what a residualisation regression
    should adjust for; the full set is the prediction baseline. Batch columns enter as
    one-hot indicators with rare levels pooled, so a scanner seen a handful of times
    cannot become a participant-identifying feature.
    """

    def __init__(self, nuisance=False, batch_cols=(), icv=None, covariates=None):
        base = list(covariates if covariates is not None else COVARIATES)
        self.cols = base[:2] if nuisance else base
        if icv and icv not in self.cols:
            self.cols.append(icv)
        self.batch_cols = list(batch_cols)

    def fit(self, frame):
        raw = finite_numeric(frame, self.cols)
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True).fit(raw)
        x = self.imp.transform(raw)
        self.scale = StandardScaler().fit(x)
        self.spline = SplineTransformer(n_knots=4, degree=2, include_bias=False,
                                        extrapolation="linear").fit(x[:, [0]])
        if self.batch_cols:
            self.onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=8,
                                        sparse_output=False).fit(self.categories(frame))
        self.fit_ids = tuple(frame[PARTICIPANT])
        return self

    def categories(self, frame):
        return frame.reindex(columns=self.batch_cols).fillna("unknown").astype(str)

    def transform(self, frame):
        x = self.imp.transform(finite_numeric(frame, self.cols))
        blocks = [self.scale.transform(x), self.spline.transform(x[:, [0]])]
        if self.batch_cols:
            blocks.append(self.onehot.transform(self.categories(frame)))
        return np.column_stack(blocks)


class ImageDesign:
    """One block of measured features, made comparable across participants and scanners.

    Fitting learns, from this partition alone: which features are present often enough
    and vary at all, the imputation medians, the scaling, optionally a ridge
    residualisation on nuisance covariates and batch, and optionally a PCA basis. A
    feature present only in the partition being predicted is therefore dropped, not
    imputed from it.
    """

    def __init__(self, features, residualize=False, pca=False, batch_cols=(), icv=None, covariates=None):
        self.features = list(features)
        self.residualize = residualize
        self.pca_requested = pca
        self.batch_cols = batch_cols
        self.icv = icv
        self.covariates = covariates

    def fit(self, frame):
        raw = finite_numeric(frame, self.features)
        # Eligibility and all moments are learned only from this fit partition.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sd = np.nanstd(raw, axis=0)
        self.keep = (np.isfinite(raw).mean(axis=0) >= 0.7) & (sd > 1e-12)
        self.cols = [c for c, k in zip(self.features, self.keep) if k]
        self.fit_ids = tuple(frame[PARTICIPANT])
        if not self.cols:
            return self
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(raw[:, self.keep])
        x = self.imp.transform(raw[:, self.keep])
        self.initial = StandardScaler().fit(x)
        x = self.initial.transform(x)
        if self.residualize:
            self.regressions = []
            t1 = np.array([not c.startswith(NON_ICV_PREFIXES) for c in self.cols])
            for use_icv, which in [(False, ~t1), (True, t1)]:
                if not which.any():
                    continue
                nuisance = CovariateDesign(True, self.batch_cols, self.icv if use_icv else None,
                                           self.covariates).fit(frame)
                reg = Ridge(alpha=1.0).fit(nuisance.transform(frame), x[:, which])
                x[:, which] -= reg.predict(nuisance.transform(frame)).reshape(len(frame), -1)
                self.regressions.append((which, nuisance, reg))
        self.scale = StandardScaler().fit(x)
        x = self.scale.transform(x)
        self.pca = None
        if self.pca_requested:
            n = min(32, len(frame) - 2, len(self.cols))
            self.pca = PCA(n_components=n, svd_solver="randomized", random_state=SEEDS[0]).fit(x)
            # Unit-variance PCs make the image-block penalty comparable across representations.
            self.pc_scale = StandardScaler().fit(self.pca.transform(x))
        return self

    def transform(self, frame):
        if not self.cols:
            return np.zeros((len(frame), 0))
        x = self.initial.transform(self.imp.transform(finite_numeric(frame, self.cols)))
        if self.residualize:
            for which, nuisance, reg in self.regressions:
                x[:, which] -= reg.predict(nuisance.transform(frame)).reshape(len(frame), -1)
        x = self.scale.transform(x)
        return self.pc_scale.transform(self.pca.transform(x)) if self.pca is not None else x


@dataclass(frozen=True)
class Candidate:
    """One point of the declared search space. `strength` is C for logistic, depth for boosted."""

    family: str
    residualize: bool = False
    pca: bool = False
    model: str = "logistic"
    strength: float = 0.1
    image_weight: float = 1.0


def candidate_grid(families):
    """The whole search space, in a fixed order that also breaks ties deterministically."""
    grid = [Candidate("baseline", strength=c) for c in C_VALUES]
    grid += [Candidate("baseline", model="boosted", strength=depth) for depth in [2, 3]]
    for family, features in families.items():
        for residualize in [False, True]:
            for pca in ([False, True] if len(features) >= 64 else [False]):
                grid += [Candidate(family, residualize, pca, "logistic", c, w)
                         for c in C_VALUES for w in [0.2, 1.0]]
                if pca or len(features) < 64:
                    grid += [Candidate(family, residualize, pca, "boosted", depth) for depth in [2, 3]]
    return grid


def estimator(candidate, seed):
    if candidate.model == "logistic":
        return LogisticRegression(C=candidate.strength, solver="lbfgs", max_iter=3000, tol=1e-6,
                                  random_state=seed)
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                          max_depth=int(candidate.strength), min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)


def split_design(train, apply_frames, families, batch_cols, icv, adjust_baseline, covariates=None):
    """Fit every design on `train` and return the design matrices for train and each apply frame."""
    cov = CovariateDesign(False, batch_cols if adjust_baseline else (),
                          icv if adjust_baseline else None, covariates).fit(train)
    allframes = [train, *apply_frames]
    bases = [cov.transform(f) for f in allframes]
    cache = {("baseline", False, False): bases}
    for name, features in families.items():
        for residualize in [False, True]:
            for pca in ([False, True] if len(features) >= 64 else [False]):
                design = ImageDesign(features, residualize, pca, batch_cols, icv, covariates).fit(train)
                cache[(name, residualize, pca)] = [np.column_stack([b, design.transform(f)])
                                                   for b, f in zip(bases, allframes)]
    return cache, bases[0].shape[1]


def candidate_arrays(cache, ncov, candidate):
    """Design matrices for one candidate, with its feature block down-weighted if asked."""
    arrays = cache[(candidate.family, candidate.residualize, candidate.pca)]
    if candidate.family == "baseline" or candidate.image_weight == 1:
        return arrays
    result = []
    for arr in arrays:
        arr = arr.copy()
        arr[:, ncov:] *= candidate.image_weight
        result.append(arr)
    return result


def fit_checked(candidate, x, y, seed):
    """Fit, treating non-convergence as failure: an unconverged fit is not a model choice."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return estimator(candidate, seed).fit(x, y)


def nested_fold(frame, train_ids, test_ids, families, batch_cols, icv, adjust_baseline, seed,
                outcome=DEFAULT_OUTCOME, covariates=None):
    """Select and fit inside `train_ids`, predict `test_ids`, never reading their outcome.

    Returns `(predictions, audit)`. `predictions` holds one probability vector per family
    plus `baseline` and `selected`. `audit` records the winning candidate and its inner
    score for each, every candidate's inner-fold AUC and log loss, the participants in
    each inner split, and any candidate that failed to fit.
    """
    if set(train_ids) & set(test_ids):
        raise ValueError("Outer partition overlap")
    train = frame.iloc[train_ids]
    test = frame.iloc[test_ids]
    if set(train[PARTICIPANT]) & set(test[PARTICIPANT]):
        raise ValueError("Participant leakage")
    y = train[outcome].to_numpy(int)
    candidates = candidate_grid(families)
    scores = np.full((len(candidates), 3), np.nan)
    losses = scores.copy()
    failures = []
    inner = StratifiedKFold(3, shuffle=True, random_state=seed + 1000)
    audit = []
    for k, (a, b) in enumerate(inner.split(train, y)):
        innertrain = train.iloc[a]
        validation = train.iloc[b]
        cache, ncov = split_design(innertrain, [validation], families, batch_cols, icv,
                                   adjust_baseline, covariates)
        audit.append({"inner_fold": k,
                      "fit_patnos": innertrain[PARTICIPANT].astype(int).tolist(),
                      "validation_patnos": validation[PARTICIPANT].astype(int).tolist()})
        for i, c in enumerate(candidates):
            try:
                xa, xb = candidate_arrays(cache, ncov, c)
                model = fit_checked(c, xa, y[a], seed + k)
                p = model.predict_proba(xb)[:, 1]
                scores[i, k] = roc_auc_score(y[b], p)
                losses[i, k] = log_loss(y[b], p)
            except (ValueError, FloatingPointError, ConvergenceWarning) as exc:
                failures.append({"candidate": i, "inner_fold": k, "error": str(exc)})
    means = np.where(np.isfinite(scores).all(axis=1), np.nan_to_num(scores).mean(axis=1), -np.inf)
    meanloss = np.where(np.isfinite(losses).all(axis=1), np.nan_to_num(losses).mean(axis=1), np.inf)

    def winner(indices):
        eligible = [i for i in indices if np.isfinite(means[i])]
        if not eligible:
            raise ValueError("All candidates failed")
        # Deterministic: largest mean inner AUC, then log loss, then declared grid order.
        return min(eligible, key=lambda i: (-means[i], meanloss[i], i))

    baseline = winner([i for i, c in enumerate(candidates) if c.family == "baseline"])
    winners = {"baseline": baseline, "selected": winner(range(len(candidates)))}
    for family in families:
        winners[family] = winner([i for i, c in enumerate(candidates) if c.family == family])
    needed = {candidates[i].family for i in winners.values()} - {"baseline"}
    cache, ncov = split_design(train, [test], {k: v for k, v in families.items() if k in needed},
                               batch_cols, icv, adjust_baseline, covariates)
    fitted = {}
    predictions = {}
    selected = {}
    for name, i in winners.items():
        c = candidates[i]
        if i not in fitted:
            xa, xb = candidate_arrays(cache, ncov, c)
            model = fit_checked(c, xa, y, seed)
            fitted[i] = model.predict_proba(xb)[:, 1]
        predictions[name] = fitted[i]
        selected[name] = {"candidate": asdict(c), "mean_inner_auc": float(means[i]),
                          "mean_inner_logloss": float(meanloss[i])}
    # No access to test labels above. Caller joins them after obtaining predictions.
    return predictions, {"selections": selected, "inner_audit": audit,
                         "candidate_scores": [{"candidate": asdict(c), "auc": scores[i].tolist(),
                                               "logloss": losses[i].tolist()}
                                              for i, c in enumerate(candidates)],
                         "failures": failures}


def paired_metrics(oof, n_boot=2000, seed=SEEDS[0], baseline="p_baseline"):
    """Each model's out-of-fold discrimination and its paired increment over the baseline.

    `oof` holds one row per participant per repeat with the true outcome in `y`, a repeat
    label in `repeat`, and one `p_<model>` column per model. Predictions are averaged
    within participant first, then resampled by participant, stratified on outcome — a
    participant appearing in several repeats moves in and out of a resample together, so
    the interval reflects the number of people, not the number of rows.

    The interval is conditional on the fitted models: it covers sampling variation in the
    evaluated participants, not the variability of the model search or of external
    validation. The returned `uncertainty` column carries that caveat with the numbers.
    """
    predcols = [c for c in oof if c.startswith("p_")]
    average = oof.groupby(PARTICIPANT, sort=True)[["y", *predcols]].mean()
    if not np.isin(average.y, [0, 1]).all():
        raise ValueError("Participant outcome changed between folds")
    y = average.y.to_numpy(int)
    base = average[baseline].to_numpy()
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    samples = [np.r_[rng.choice(positives, len(positives), replace=True),
                     rng.choice(negatives, len(negatives), replace=True)] for _ in range(n_boot)]
    base_auc = roc_auc_score(y, base)
    base_boot = np.array([roc_auc_score(y[i], base[i]) for i in samples])
    rows = []
    for c in predcols:
        p = average[c].to_numpy()
        auc = roc_auc_score(y, p)
        boot = np.array([roc_auc_score(y[i], p[i]) for i in samples])
        delta = boot - base_boot
        repeat_delta, repeat_auc = [], []
        for _, g in oof.groupby("repeat"):
            repeat_auc.append(roc_auc_score(g.y, g[c]))
            repeat_delta.append(repeat_auc[-1] - roc_auc_score(g.y, g[baseline]))
        rows.append({
            "model": c[2:], "n": len(y), "n_positive": int(y.sum()),
            "auc_averaged_oof": auc, "baseline_auc_averaged_oof": base_auc,
            "delta_auc": auc - base_auc,
            "delta_ci95_low": float(np.quantile(delta, 0.025)),
            "delta_ci95_high": float(np.quantile(delta, 0.975)),
            "delta_ci98_75_low": float(np.quantile(delta, 0.00625)),
            "delta_ci98_75_high": float(np.quantile(delta, 0.99375)),
            "auc_ci95_low": float(np.quantile(boot, 0.025)),
            "auc_ci95_high": float(np.quantile(boot, 0.975)),
            "mean_repeat_auc": float(np.mean(repeat_auc)),
            "mean_repeat_delta": float(np.mean(repeat_delta)),
            "repeat_deltas": repeat_delta,
            "positive_in_all_repeats": all(d > 0 for d in repeat_delta),
            "logloss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "delta_logloss": log_loss(y, p) - log_loss(y, base),
            "delta_brier": brier_score_loss(y, p) - brier_score_loss(y, base),
            "uncertainty": "Stratified paired participant bootstrap of repeat-averaged OOF; "
                           "conditional on fitted models, not model-search or external-validation uncertainty"})
    return pd.DataFrame(rows)
