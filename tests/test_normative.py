"""Reference-group deviation scores (pie.imaging.normative) on synthetic data."""
import numpy as np
import pandas as pd

from pie.imaging import normative


def _cohort(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"age": rng.uniform(50, 80, n), "sex": rng.integers(0, 2, n), "tiv_mm3": rng.normal(1.5e6, 1e5, n)})
    df["vol"] = 5000 - 20 * (df.age - 65) - 0.8 * (df.age - 65) ** 2 + 300 * df.sex + 0.002 * (df.tiv_mm3 - 1.5e6) + rng.normal(0, 100, n)
    return df


def test_normative_z_is_standard_in_controls_and_detects_a_shift():
    df = _cohort()
    ref = np.arange(len(df)) < 400
    df.loc[~ref, "vol"] -= 200                                   # patients: 2 SD smaller
    z = normative.zscores(df, normative.fit(df, ["vol"], ["age", "sex", "tiv_mm3"], ref))["z_vol"]
    assert abs(z[ref].mean()) < 0.05 and abs(z[ref].std() - 1) < 0.1 and abs(z[~ref].mean() + 2) < 0.2
    assert abs(np.corrcoef(z[ref], df.age[ref])[0, 1]) < 0.05 and abs(np.corrcoef(z[ref], (df.age[ref] - 65) ** 2)[0, 1]) < 0.05


def test_normative_fit_uses_only_reference_rows_and_refuses_too_few():
    df = _cohort(50)
    ref = np.zeros(len(df), bool)
    ref[:4] = True
    try:
        normative.fit(df, ["vol"], ["age", "sex", "tiv_mm3"], ref)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "vol" in str(e)
    ref[:40] = True
    df2 = df.copy()
    df2.loc[40:, "vol"] += 1e6                                   # non-reference rows must not move the model
    m1, m2 = normative.fit(df, ["vol"], ["age"], ref), normative.fit(df2, ["vol"], ["age"], ref)
    assert np.allclose(m1["features"]["vol"]["beta"], m2["features"]["vol"]["beta"])


def test_site_offsets_heteroscedastic_sd_and_out_of_fold_controls():
    rng = np.random.default_rng(4)
    n = 600
    age = rng.uniform(45, 85, n)
    site = rng.choice(["A", "B", "C"], n)
    shift = pd.Series(site).map({"A": 0.0, "B": 300.0, "C": -300.0}).to_numpy()
    sd = 50 + 5 * (age - 45)                                  # variance grows with age
    vol = 5000 - 20 * (age - 65) + shift + sd * rng.normal(size=n)
    df = pd.DataFrame({"age": age, "sex": rng.integers(0, 2, n).astype(float), "site": site, "vol": vol})
    ref = np.r_[np.ones(500, bool), np.zeros(100, bool)]
    plain = normative.zscores(df, normative.fit(df, ["vol"], ["age", "sex"], ref))["z_vol"]
    m = normative.fit(df, ["vol"], ["age", "sex"], ref, site="site", age_terms="spline", heteroscedastic=True)
    z = normative.zscores(df, m)["z_vol"]
    assert abs(m["features"]["vol"]["site_offsets"]["B"] - 300) < 60 and abs(z[ref].std() - 1) < 0.1
    young, old = df.age < 55, df.age > 75
    assert abs(z[ref & young].std() - z[ref & old].std()) < 0.3 < abs(plain[ref & young].std() - plain[ref & old].std())
    cz = normative.crossfit_zscores(df, ["vol"], ["age", "sex"], ref, site="site", age_terms="spline", heteroscedastic=True)["z_vol"]
    assert abs(cz[ref].mean()) < 0.1 and 0.95 < cz[ref].std() < 1.15 and np.allclose(cz[~ref], z[~ref])


def test_mtl_preservation_reads_hippocampal_and_amygdala_z():
    z = pd.DataFrame({"z_vol_Left_Hippocampus": [0.5, -2.0, np.nan], "z_vol_Right_Hippocampus": [0.1, -1.6, 0.0],
                      "z_vol_Left_Amygdala": [0.0, -1.0, 0.0], "z_vol_Left_Putamen": [-3.0, 0.0, 0.0]})
    m = normative.mtl_preservation(z)
    assert m["mtl_preserved"].tolist()[:2] == [True, False] and pd.isna(m["mtl_preserved"].iloc[2]) and abs(m["mtl_z"].iloc[1] + 1.533) < 1e-3


def test_a_missing_site_label_is_its_own_site_not_a_nan_model():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({"age": rng.uniform(50, 80, n), "sex": rng.integers(0, 2, n).astype(float), "vol": rng.normal(5000, 300, n),
                       "site": pd.Series(rng.choice(["A", "B"], n), dtype=object)})
    df.loc[:7, "site"] = np.nan
    z = normative.zscores(df, normative.fit(df, ["vol"], ["age", "sex"], np.ones(n, bool), site="site"))["z_vol"]
    assert z.notna().all() and abs(z.std() - 1) < 0.1
