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
