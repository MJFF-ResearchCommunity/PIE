import numpy as np
import pandas as pd
import pytest
from pie.stats.compare import (
    independent_ttest, paired_ttest, welch_ttest,
    mann_whitney, wilcoxon_signed_rank,
    one_way_anova, kruskal_wallis, tukey_hsd, dunn_posthoc,
    chi_square, fisher_exact, mcnemar,
    cohens_d, hedges_g,
)


@pytest.fixture
def two_groups():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 50)
    b = rng.normal(12, 2, 50)
    return a, b


def test_independent_ttest_detects_effect(two_groups):
    a, b = two_groups
    r = independent_ttest(a, b)
    assert r["p_value"] < 0.001
    assert r["statistic"] < 0
    assert r["df"] == 98
    assert "cohens_d" in r
    assert r["cohens_d"] < 0


def test_welch_ttest_unequal_variance():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 50)
    b = rng.normal(0.5, 5, 50)
    r = welch_ttest(a, b)
    # Welch df is non-integer
    assert not float(r["df"]).is_integer()


def test_paired_ttest_requires_equal_length(two_groups):
    a, b = two_groups
    with pytest.raises(ValueError):
        paired_ttest(a[:10], b)


def test_mann_whitney(two_groups):
    a, b = two_groups
    r = mann_whitney(a, b)
    assert r["p_value"] < 0.001
    assert "u_statistic" in r


def test_wilcoxon_signed_rank():
    rng = np.random.default_rng(2)
    a = rng.normal(5, 1, 30)
    b = a + rng.normal(0.5, 0.5, 30)
    r = wilcoxon_signed_rank(a, b)
    assert r["p_value"] < 0.05


def test_cohens_d_sign_and_magnitude():
    a = np.array([0.0] * 30)
    b = np.array([1.0] * 30)
    d = cohens_d(a, b)
    assert np.isnan(d) or d == pytest.approx(-float("inf"))


def test_hedges_g_matches_d_for_large_n(two_groups):
    a, b = two_groups
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    assert g == pytest.approx(d * correction, rel=1e-6)


# Multi-group

def test_one_way_anova_detects_difference():
    rng = np.random.default_rng(0)
    groups = {"A": rng.normal(0, 1, 30), "B": rng.normal(1, 1, 30), "C": rng.normal(2, 1, 30)}
    r = one_way_anova(groups)
    assert r["p_value"] < 0.001
    assert "eta_squared" in r
    assert r["eta_squared"] > 0


def test_kruskal_wallis():
    rng = np.random.default_rng(1)
    groups = {"A": rng.exponential(1, 30), "B": rng.exponential(2, 30)}
    r = kruskal_wallis(groups)
    assert "statistic" in r and "p_value" in r


def test_tukey_hsd_returns_pairwise():
    rng = np.random.default_rng(0)
    groups = {"A": rng.normal(0, 1, 30), "B": rng.normal(3, 1, 30), "C": rng.normal(5, 1, 30)}
    r = tukey_hsd(groups)
    pairs = r["pairwise"]
    # C(3,2) = 3 comparisons
    assert len(pairs) == 3
    assert {"group1", "group2", "mean_diff", "p_adj", "reject"} <= set(pairs[0].keys())


def test_dunn_posthoc_returns_pairwise():
    rng = np.random.default_rng(0)
    groups = {"A": rng.exponential(1, 30), "B": rng.exponential(2, 30), "C": rng.exponential(3, 30)}
    r = dunn_posthoc(groups)
    assert len(r["pairwise"]) == 3


# Categorical

def test_chi_square_independence():
    table = [[10, 20], [30, 15]]
    r = chi_square(table)
    assert {"statistic", "p_value", "dof", "expected"} <= r.keys()
    assert r["dof"] == 1


def test_fisher_exact_2x2():
    r = fisher_exact([[1, 9], [11, 3]])
    assert "odds_ratio" in r and "p_value" in r


def test_mcnemar_paired_binary():
    r = mcnemar(b=3, c=15)
    assert r["p_value"] < 0.05
