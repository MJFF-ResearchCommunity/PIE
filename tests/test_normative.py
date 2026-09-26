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
    assert np.allclose(m1["features"]["vol"][0], m2["features"]["vol"][0])
