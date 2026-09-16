import numpy as np
import pandas as pd
import pytest
from pie.stats.longitudinal import linear_mixed_model, change_from_baseline


@pytest.fixture
def long_df():
    """30 subjects × 4 visits, with linear time effect + subject random intercept."""
    rng = np.random.default_rng(0)
    subs, rows = range(30), []
    for s in subs:
        intercept = rng.normal(20, 5)
        for t in range(4):
            y = intercept + 2.0 * t + rng.normal(0, 1)
            rows.append({"patno": s, "visit": t, "updrs": y,
                         "cohort": "PD" if s % 2 == 0 else "HC"})
    return pd.DataFrame(rows)


def test_linear_mixed_model_recovers_slope(long_df):
    r = linear_mixed_model(long_df, outcome="updrs", fixed_effects=["visit"],
                           group="patno")
    coefs = {c["predictor"]: c for c in r["fixed_effects"]}
    assert coefs["visit"]["estimate"] == pytest.approx(2.0, abs=0.3)
    assert r["n_groups"] == 30
    assert r["n_obs"] == 120


def test_change_from_baseline(long_df):
    r = change_from_baseline(long_df, subject="patno", time="visit",
                             outcome="updrs", baseline_time=0)
    assert r["n_subjects"] == 30
    v3 = r["summary_by_time"][3]
    assert v3["mean_change"] == pytest.approx(6.0, abs=0.5)


def test_lmm_matches_each_fixed_effect_by_its_exact_term(long_df):
    # "visit" is a substring of "visit_noise"; each must get its own coefficient.
    d = long_df.assign(visit_noise=np.random.default_rng(1).normal(size=len(long_df)))
    r = linear_mixed_model(d, outcome="updrs", fixed_effects=["visit_noise", "visit"], group="patno")
    coefs = {c["predictor"]: c["estimate"] for c in r["fixed_effects"]}
    assert coefs["visit"] == pytest.approx(2.0, abs=0.3)
    assert coefs["visit_noise"] == pytest.approx(0.0, abs=0.3)


def test_lmm_reports_each_level_of_a_categorical_fixed_effect(long_df):
    r = linear_mixed_model(long_df, outcome="updrs", fixed_effects=["visit", "cohort"], group="patno")
    assert [c["predictor"] for c in r["fixed_effects"]] == ["visit", "cohort[T.PD]"]


def test_lmm_random_slope_must_also_be_a_fixed_effect(long_df):
    with pytest.raises(ValueError, match="random_slopes"):
        linear_mixed_model(long_df, outcome="updrs", fixed_effects=["cohort"], group="patno",
                           random_slopes=["visit"])


def test_lmm_information_criteria_come_from_a_maximum_likelihood_fit(long_df):
    r = linear_mixed_model(long_df, outcome="updrs", fixed_effects=["visit"], group="patno")
    assert np.isfinite(r["aic"]) and np.isfinite(r["bic"]) and r["bic"] > r["aic"]


def test_change_from_baseline_counts_only_subjects_with_a_baseline():
    d = pd.DataFrame({"PATNO": [1, 1, 2, 2, 3], "visit": [0, 1, 0, 1, 1], "y": [10, 12, 20, 25, 30.]})
    r = change_from_baseline(d, "PATNO", "visit", "y")
    assert r["summary_by_time"][1]["n"] == 2
    assert r["summary_by_time"][1]["mean_change"] == pytest.approx(3.5)
    assert r["n_without_baseline"] == 1
    assert "changes" not in r


def test_change_from_baseline_refuses_duplicate_visits():
    d = pd.DataFrame({"PATNO": [1, 1, 1], "visit": [0, 0, 1], "y": [10, 11, 12.]})
    with pytest.raises(ValueError, match="more than one row"):
        change_from_baseline(d, "PATNO", "visit", "y")
