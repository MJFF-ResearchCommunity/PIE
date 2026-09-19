"""Small-sample inference and classification that keep selection inside validation.

Feature screening and feature-subset search are common in small imaging samples. When they use the same
participants as the reported validation, the estimate is optimistically biased (Varma & Simon 2006; Cawley &
Talbot 2010). These helpers run the whole search inside cross-validation, and can also report the resubstitution
design and its permutation distribution so the size of that bias can be shown for a given sample.

    bootstrap_partial_correlation(df, x, y, covars)   partial r with a percentile bootstrap interval (no pingouin)
    naive_subset_search(X, y)                          best subset chosen on the evaluation folds themselves
                                                       (reproduces the optimistic design, for comparison)
    nested_subset_search(X, y)                         same search repeated inside every outer training fold;
                                                       the reported score is on held-out participants only
    subset_search_null(X, y, n_permutations)           the naive best score under shuffled labels: how high a
                                                       noise-only result can go with this design and sample
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _residual(values, covariates):
    design = np.column_stack([np.ones(len(values)), covariates]) if covariates.size else np.ones((len(values), 1))
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def bootstrap_partial_correlation(df: pd.DataFrame, x: str, y: str, covars=(), method="pearson",
                                  n_boot=1000, seed=0, ci=0.95):
    """Partial correlation of x and y given covars, with an analytic P and a percentile bootstrap interval."""
    data = df[[x, y, *covars]].dropna()
    n, k = len(data), len(covars)
    if n < k + 4:
        raise ValueError("too few complete rows")
    values = data.to_numpy(float)
    if method == "spearman":
        values = np.column_stack([stats.rankdata(c) for c in values.T])
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")

    def partial(rows):
        cov = values[rows, 2:]
        rx, ry = _residual(values[rows, 0], cov), _residual(values[rows, 1], cov)
        return float(np.corrcoef(rx, ry)[0, 1])

    r = partial(np.arange(n))
    dof = n - 2 - k
    t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
    p = float(2 * stats.t.sf(abs(t), dof))
    rng = np.random.default_rng(seed)
    boots = np.array([partial(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.nanquantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"r": r, "p": p, "n": n, "covariates": list(covars), "method": method,
            "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def _svm():
    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))


def _cv_predictions(X, y, cv):
    """Held-out decision scores and labels for every row under ``cv``."""
    scores, labels = np.zeros(len(y)), np.zeros(len(y), int)
    for train, test in cv.split(X, y):
        model = _svm().fit(X[train], y[train])
        scores[test] = model.decision_function(X[test])
        labels[test] = model.predict(X[test])
    return scores, labels


def _metric(y, scores, labels, metric):
    if metric == "precision":
        return precision_score(y, labels, zero_division=0)
    if metric == "roc_auc":
        return roc_auc_score(y, scores)
    raise ValueError("metric must be 'precision' or 'roc_auc'")


def _subsets(p, max_size):
    for size in range(1, (max_size or p) + 1):
        yield from combinations(range(p), size)


def _best_subset(X, y, metric, max_size, cv):
    best, best_score = None, -np.inf
    for subset in _subsets(X.shape[1], max_size):
        s, l = _cv_predictions(X[:, subset], y, cv)
        score = _metric(y, s, l, metric)
        if score > best_score + 1e-12:
            best, best_score = subset, score
    return best, best_score


def naive_subset_search(X, y, metric="precision", max_size=None):
    """Pick the subset with the best leave-one-out score on all participants and report that score.

    This is the design whose optimism ``nested_subset_search`` and ``subset_search_null`` quantify; the
    returned AUC is the leave-one-out AUC of the chosen subset on the same participants that chose it.
    """
    X, y = np.asarray(X, float), np.asarray(y, int)
    subset, score = _best_subset(X, y, metric, max_size, LeaveOneOut())
    s, _ = _cv_predictions(X[:, subset], y, LeaveOneOut())
    return {"subset": list(subset), metric: float(score), "roc_auc": float(roc_auc_score(y, s)),
            "n_subsets_searched": sum(1 for _ in _subsets(X.shape[1], max_size))}


def nested_subset_search(X, y, metric="precision", max_size=None, inner_folds=5, seed=0):
    """Leave-one-out outer loop; the whole subset search runs inside each outer training fold.

    Returns the held-out AUC and precision and how often each feature was selected (selection stability).
    """
    X, y = np.asarray(X, float), np.asarray(y, int)
    scores, labels = np.zeros(len(y)), np.zeros(len(y), int)
    chosen = []
    for train, test in LeaveOneOut().split(X):
        folds = min(inner_folds, int(np.bincount(y[train]).min()))
        if folds < 2:
            raise ValueError("a class has fewer than two training rows")
        inner = StratifiedKFold(folds, shuffle=True, random_state=seed)
        subset, _ = _best_subset(X[train], y[train], metric, max_size, inner)
        chosen.append(subset)
        model = _svm().fit(X[train][:, subset], y[train])
        scores[test] = model.decision_function(X[test][:, subset])
        labels[test] = model.predict(X[test][:, subset])
    freq = np.zeros(X.shape[1])
    for subset in chosen:
        freq[list(subset)] += 1
    return {"roc_auc": float(roc_auc_score(y, scores)), "precision": float(precision_score(y, labels, zero_division=0)),
            "selection_frequency": (freq / len(y)).tolist(), "held_out_scores": scores.tolist()}


def subset_search_null(X, y, n_permutations=100, metric="precision", max_size=None, seed=0):
    """Naive best-subset score under shuffled labels: the chance level of the optimistic design."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    rng = np.random.default_rng(seed)
    null = np.array([naive_subset_search(X, rng.permutation(y), metric, max_size)[metric] for _ in range(n_permutations)])
    observed = naive_subset_search(X, y, metric, max_size)[metric]
    return {"observed": float(observed), "null_median": float(np.median(null)), "null_95th": float(np.quantile(null, 0.95)),
            "p": float((1 + (null >= observed).sum()) / (1 + n_permutations)), "n_permutations": n_permutations}
