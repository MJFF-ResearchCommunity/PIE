import numpy as np
import pandas as pd
import pytest
from pie.stats.correlate import correlate_pair, partial_correlation, correlation_matrix


@pytest.fixture
def corr_df():
    rng = np.random.default_rng(0)
    age = rng.normal(65, 10, 200)
    disease_dur = age - 40 + rng.normal(0, 3, 200)
    updrs = 0.5 * disease_dur + rng.normal(0, 5, 200)
    noise = rng.normal(0, 1, 200)
    return pd.DataFrame({"age": age, "disease_dur": disease_dur, "updrs": updrs, "noise": noise})


def test_correlate_pair_pearson(corr_df):
    r = correlate_pair(corr_df["age"], corr_df["disease_dur"], method="pearson")
    assert r["method"] == "pearson"
    assert r["r"] > 0.9
    assert r["p_value"] < 1e-50


def test_correlate_pair_spearman(corr_df):
    r = correlate_pair(corr_df["age"], corr_df["disease_dur"], method="spearman")
    assert r["method"] == "spearman"


def test_correlate_pair_kendall(corr_df):
    r = correlate_pair(corr_df["age"], corr_df["disease_dur"], method="kendall")
    assert r["method"] == "kendall"


def test_correlate_pair_rejects_unknown(corr_df):
    with pytest.raises(ValueError):
        correlate_pair(corr_df["age"], corr_df["disease_dur"], method="made_up")


def test_partial_correlation_removes_confounder(corr_df):
    direct = correlate_pair(corr_df["age"], corr_df["updrs"])
    partial = partial_correlation(corr_df, "age", "updrs", covariates=["disease_dur"])
    assert abs(partial["r"]) < abs(direct["r"])


def test_correlation_matrix_shape(corr_df):
    r = correlation_matrix(corr_df, ["age", "disease_dur", "updrs"])
    assert set(r["matrix"].keys()) == {"age", "disease_dur", "updrs"}
    assert r["matrix"]["age"]["age"] == pytest.approx(1.0)
    assert "p_values_adjusted" in r
