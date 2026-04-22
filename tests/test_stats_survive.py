import numpy as np
import pandas as pd
import pytest
from pie.stats.survive import kaplan_meier, logrank_test, cox_regression


@pytest.fixture
def surv_df():
    rng = np.random.default_rng(0)
    n = 300
    group = rng.choice([0, 1], n)
    scale = np.where(group == 0, 1 / 0.05, 1 / 0.15)
    time = rng.exponential(scale)
    event = (time < 40).astype(int)
    time = np.minimum(time, 40)
    age = rng.normal(65, 10, n)
    return pd.DataFrame({"time": time, "event": event, "group": group, "age": age})


def test_kaplan_meier(surv_df):
    r = kaplan_meier(surv_df, time="time", event="event", group=None)
    assert "timeline" in r and "survival" in r
    surv = r["survival"]["_overall"]
    # Monotonically non-increasing
    assert all(surv[i] >= surv[i + 1] - 1e-9 for i in range(len(surv) - 1))


def test_kaplan_meier_grouped(surv_df):
    r = kaplan_meier(surv_df, time="time", event="event", group="group")
    assert set(r["survival"].keys()) == {"0", "1"}


def test_logrank_test(surv_df):
    r = logrank_test(surv_df, time="time", event="event", group="group")
    assert r["p_value"] < 1e-5


def test_cox_regression(surv_df):
    r = cox_regression(surv_df, time="time", event="event",
                       covariates=["group", "age"])
    coefs = {c["predictor"]: c for c in r["coefficients"]}
    assert 2.0 < coefs["group"]["hazard_ratio"] < 5.0
    assert "concordance" in r
    assert "ph_test" in r
