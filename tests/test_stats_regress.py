import numpy as np
import pandas as pd
import pytest
from pie.stats.regress import linear_regression, logistic_regression, ancova


@pytest.fixture
def lin_df():
    rng = np.random.default_rng(0)
    age = rng.normal(65, 10, 200)
    dd = rng.normal(5, 3, 200)
    y = 1.0 + 0.5 * age + 2.0 * dd + rng.normal(0, 2, 200)
    return pd.DataFrame({"age": age, "dd": dd, "y": y})


def test_linear_regression_recovers_coefs(lin_df):
    r = linear_regression(lin_df, outcome="y", predictors=["age", "dd"])
    coefs = {c["predictor"]: c for c in r["coefficients"]}
    assert coefs["age"]["estimate"] == pytest.approx(0.5, abs=0.05)
    assert coefs["dd"]["estimate"] == pytest.approx(2.0, abs=0.1)
    assert "r_squared" in r and "adj_r_squared" in r
    assert r["r_squared"] > 0.9


def test_linear_regression_diagnostics_present(lin_df):
    r = linear_regression(lin_df, outcome="y", predictors=["age", "dd"])
    assert "diagnostics" in r
    d = r["diagnostics"]
    assert "vif" in d and set(d["vif"].keys()) == {"age", "dd"}
    assert "durbin_watson" in d
    assert "fitted" in d and "residuals" in d


def test_logistic_regression_recovers_coefs():
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logits = -0.5 + 1.2 * x1 - 0.8 * x2
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(0, 1, n) < probs).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    r = logistic_regression(df, outcome="y", predictors=["x1", "x2"])
    coefs = {c["predictor"]: c for c in r["coefficients"]}
    assert 2.5 < coefs["x1"]["odds_ratio"] < 4.2
    assert 0.3 < coefs["x2"]["odds_ratio"] < 0.6
    assert "auc" in r


def test_ancova_detects_group_effect_after_covariate():
    rng = np.random.default_rng(0)
    n = 200
    age = rng.normal(65, 10, n)
    group = rng.choice(["A", "B"], n)
    y = 0.3 * age + np.where(group == "B", 3.0, 0.0) + rng.normal(0, 1, n)
    df = pd.DataFrame({"age": age, "group": group, "y": y})
    r = ancova(df, outcome="y", group="group", covariates=["age"])
    assert any(row["source"] == "group" and row["p_value"] < 0.001 for row in r["effects"])
