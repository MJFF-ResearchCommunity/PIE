"""Nested selection is honest where the naive design is optimistic; bootstrap partial correlation is correct."""
import numpy as np
import pandas as pd
import pytest

from pie.stats import small_sample as ss


def test_partial_correlation_removes_a_shared_confound():
    rng = np.random.default_rng(0)
    age = rng.normal(size=300)
    df = pd.DataFrame({"age": age, "x": age + rng.normal(0, 0.5, 300), "y": age + rng.normal(0, 0.5, 300)})
    raw = ss.bootstrap_partial_correlation(df, "x", "y", n_boot=200)
    part = ss.bootstrap_partial_correlation(df, "x", "y", ["age"], n_boot=200)
    assert raw["r"] > 0.7 and abs(part["r"]) < 0.15 and part["ci_low"] < 0 < part["ci_high"]


def test_naive_subset_search_is_optimistic_on_noise_and_nested_is_not():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 5))
    y = np.array([0] * 10 + [1] * 20)
    naive = ss.naive_subset_search(X, y, metric="roc_auc")
    nested = ss.nested_subset_search(X, y, metric="roc_auc", inner_folds=3)
    assert naive["roc_auc"] > nested["roc_auc"] + 0.1
    assert nested["roc_auc"] < 0.7


def test_null_of_the_naive_design_sits_far_above_chance():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(24, 4))
    y = np.array([0] * 8 + [1] * 16)
    null = ss.subset_search_null(X, y, n_permutations=20, metric="roc_auc")
    assert null["null_median"] > 0.55
