import numpy as np
import pandas as pd
import pytest
from pie.stats.describe import summary_statistics, normality_test, missingness_report


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.normal(65, 10, 100),
        "updrs": rng.normal(30, 15, 100),
        "sex": rng.choice(["M", "F"], 100),
        "missing_col": [np.nan] * 30 + list(rng.normal(0, 1, 70)),
    })


def test_summary_statistics_numeric(df):
    result = summary_statistics(df, ["age", "updrs"])
    assert set(result.keys()) == {"age", "updrs"}
    age = result["age"]
    for key in ("n", "mean", "median", "std", "min", "max",
                "q1", "q3", "iqr", "skew", "kurtosis", "n_missing", "pct_missing"):
        assert key in age, f"missing {key}"
    assert age["n"] == 100
    assert age["n_missing"] == 0
    assert 55 < age["mean"] < 75


def test_summary_statistics_handles_missing(df):
    result = summary_statistics(df, ["missing_col"])
    assert result["missing_col"]["n"] == 70
    assert result["missing_col"]["n_missing"] == 30
    assert result["missing_col"]["pct_missing"] == pytest.approx(30.0)


def test_summary_statistics_rejects_non_numeric(df):
    with pytest.raises(ValueError, match="not numeric"):
        summary_statistics(df, ["sex"])


def test_normality_test_shapiro(df):
    r = normality_test(df["age"], test="shapiro")
    assert {"test", "statistic", "p_value", "n", "is_normal"} <= r.keys()
    assert r["test"] == "shapiro"
    assert r["n"] == 100


def test_normality_test_ks(df):
    r = normality_test(df["age"], test="ks")
    assert r["test"] == "ks"


def test_normality_test_rejects_unknown(df):
    with pytest.raises(ValueError, match="unknown test"):
        normality_test(df["age"], test="made_up")


def test_missingness_report(df):
    r = missingness_report(df)
    assert r["n_rows"] == 100
    assert "per_column" in r
    assert r["per_column"]["missing_col"]["n_missing"] == 30
    assert "little_mcar" in r
