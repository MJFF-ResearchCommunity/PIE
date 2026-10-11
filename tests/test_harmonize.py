"""Harmonisation on simulated data with known batch effects (no participant data)."""
import contextlib
import io
import logging

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from pie.imaging import harmonize as h

SHIFT = {"A": 0.0, "B": 1.5, "C": -1.0}
SCALE = {"A": 1.0, "B": 2.0, "C": 0.5}


def cross_sectional(seed=3):
    """Three sites with overlapping age ranges, a curved age effect, a protected SAA effect, and additive and
    multiplicative site effects. Returns the frame, feature names and the site-free truth per feature."""
    rng = np.random.default_rng(seed)
    df = pd.concat([pd.DataFrame({"site": s, "age": np.round(rng.uniform(lo, lo + 16, n) * 4) / 4,  # float32-exact
                                  "sex": rng.choice(["F", "M"], n), "saa": rng.integers(0, 2, n)})
                    for s, lo, n in [("A", 50, 120), ("B", 60, 100), ("C", 70, 80)]], ignore_index=True)
    feats, truth = [f"f{i}" for i in range(20)], {}
    for i, c in enumerate(feats):
        truth[c] = 0.01 * (df.age - 50) ** 2 + 0.8 * df.saa + 0.3 * (df.sex == "M")
        df[c] = truth[c] + df.site.map(SHIFT) * (1 + 0.05 * i) + rng.normal(0, 0.5, len(df)) * df.site.map(SCALE)
    return df, feats, truth


def site_spread(out, rows, feats, truth, by):
    """Mean over features of (max - min) site mean and max / min site SD of out - truth."""
    r = pd.DataFrame({c: out[c] - truth[c][rows] for c in feats}).groupby(by[rows])
    return (r.mean().max() - r.mean().min()).mean(), (r.std().max() / r.std().min()).mean()


@pytest.mark.parametrize("eb", [True, False])
def test_linear_combat_matches_neurocombat(eb):
    from neuroCombat import neuroCombat
    df, feats, _ = cross_sectional()
    df["saa"] = df.saa.map({0: "neg", 1: "pos"})
    with contextlib.redirect_stdout(io.StringIO()):
        ref = neuroCombat(df[feats].to_numpy().T, df[["site", "age", "sex", "saa"]], "site",
                          categorical_cols=["sex", "saa"], continuous_cols=["age"], eb=eb)
    out, model = h.combat(df, feats, "site", ["age", "sex"], nonlinear=(), label="saa", eb=eb)
    np.testing.assert_allclose(out[feats].to_numpy(), ref["data"].T, rtol=0, atol=1e-9)
    np.testing.assert_allclose(model["gamma"], ref["estimates"]["gamma.star"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(model["delta"], ref["estimates"]["delta.star"], rtol=0, atol=1e-9)


def test_spline_age_recovers_site_free_values_where_linear_does_not():
    df, feats, truth = cross_sectional()
    raw, _ = site_spread(df, df.index, feats, truth, df.site)
    lin, _ = h.combat(df, feats, "site", ["age", "sex"], nonlinear=(), label="saa")
    gam, _ = h.combat(df, feats, "site", ["age", "sex"], label="saa")
    lin_mean, lin_sd = site_spread(lin, df.index, feats, truth, df.site)
    gam_mean, gam_sd = site_spread(gam, df.index, feats, truth, df.site)
    assert raw > 3 and lin_mean > 0.8 and lin_sd > 3          # site and curvature entangled under a linear age term
    assert gam_mean < 0.25 and gam_sd < 1.2
    x = np.column_stack([np.ones(len(df)), 0.01 * (df.age - 50) ** 2, df.saa, df.sex == "M"]).astype(float)
    saa = [np.linalg.lstsq(x, gam[c], rcond=None)[0][2] for c in feats]
    assert abs(np.mean(saa) - 0.8) < 0.05                    # protected label effect survives


def test_prediction_mode_fits_in_fold_without_the_label():
    df, feats, truth = cross_sectional()
    train = df.sample(frac=0.7, random_state=0)
    test = df.drop(train.index).drop(columns="saa")          # held-out labels are not needed
    test.loc[test.index[0], "age"] = 99.0                    # beyond the fitted spline range: clipped
    fitted, model = h.combat(train, feats, "site", ["age", "sex"])
    pd.testing.assert_frame_equal(h.combat_apply(model, train), fitted)
    out = h.combat_apply(model, test)
    raw, _ = site_spread(test, test.index, feats, truth, df.site)
    harmonised, _ = site_spread(out, test.index, feats, truth, df.site)
    assert harmonised < 0.1 * raw
    with pytest.raises(ValueError, match="'D'"):
        h.combat_apply(model, test.assign(site="D"))
    _, inference = h.combat(train, feats, "site", ["age", "sex"], label="saa")
    with pytest.raises(ValueError, match="inference mode"):
        h.combat_apply(inference, test)


def test_inference_mode_logs_the_confounding_report(caplog):
    df, feats, _ = cross_sectional()
    with caplog.at_level(logging.WARNING, logger="pie.imaging.harmonize"):
        _, model = h.combat(df, feats, "site", ["age", "sex"], label="saa")
    assert "Cramér's V" in caplog.text and "inference mode" in caplog.text
    assert model["confounding"]["table"].to_numpy().sum() == len(df)
    with pytest.raises(ValueError, match="nonlinear"):
        h.combat(df, feats, "site", ["sex"])                 # default spline term "age" not among covariates


def test_complete_confounding_is_refused():
    df, feats, _ = cross_sectional()
    df["saa"] = (df.site == "B").astype(int)                 # label constant within every site
    with pytest.raises(ValueError, match="collinear"):
        h.combat(df, feats, "site", ["age", "sex"], label="saa")


def test_confounding_report_flags_entangled_cohorts():
    # cohort: (n, SAA-positive): PPMI PD ~88 %, ADNI ~22 %, a single-label site and one with 3 negatives
    cohorts = {"PPMI": (200, 176), "ADNI": (150, 33), "PPP": (60, 30), "BEAT-PD": (12, 9), "Stanford": (30, 30)}
    df = pd.concat([pd.DataFrame({"cohort": c, "saa": np.r_[np.ones(k), np.zeros(n - k)].astype(int),
                                  "age": 60 + i + np.arange(n) % 10, "sex": np.where(np.arange(n) % 3, "M", "F")})
                    for i, (c, (n, k)) in enumerate(cohorts.items())], ignore_index=True)
    rep = h.confounding_report(df, "cohort", "saa", ["age", "sex"])
    sites = rep["sites"]
    assert rep["table"].loc["PPMI", 1] == 176 and sites.loc["ADNI", "n"] == 150
    assert sites.loc["PPMI", "p_1"] == 0.88 and sites.loc["ADNI", "p_1"] == 0.22
    assert sites.index[sites.single_label].tolist() == ["Stanford"]
    assert rep["flagged"] == ["BEAT-PD", "Stanford"]
    chi2 = chi2_contingency(rep["table"], correction=False)[0]
    assert rep["cramers_v"] == pytest.approx(np.sqrt(chi2 / len(df)))
    adni = df[df.cohort == "ADNI"]
    assert sites.loc["ADNI", "age"] == pytest.approx(adni.age.mean())
    assert sites.loc["ADNI", "sex_F"] == pytest.approx((adni.sex == "F").mean())


def test_long_combat_removes_scanner_effects_and_keeps_the_slope():
    # 80 participants x 4 yearly visits; 60 % move from scanner A or B to C at year 2, so scanner and time are
    # entangled and the raw within-subject slope is biased.
    rng = np.random.default_rng(5)
    ns, nv, slope = 80, 4, -0.5
    d = pd.DataFrame({"PATNO": np.repeat(np.arange(ns), nv), "years": np.tile(np.arange(nv, dtype=float), ns),
                      "sex": np.repeat(rng.choice(["F", "M"], ns), nv),
                      "age_bl": np.repeat(rng.uniform(55, 75, ns), nv)})
    moved = np.repeat(rng.random(ns) < 0.6, nv) & (d.years >= 2)
    d["scanner"] = np.where(moved, "C", np.repeat(rng.choice(["A", "B"], ns), nv))
    shift, scale = {"A": 0.0, "B": 2.0, "C": -1.5}, {"A": 1.0, "B": 2.0, "C": 0.5}
    feats, truth = [f"v{i}" for i in range(15)], {}
    for i, c in enumerate(feats):
        truth[c] = 20 + np.repeat(rng.normal(0, 2, ns), nv) + slope * d.years + 0.05 * d.age_bl + 0.3 * (d.sex == "M")
        d[c] = truth[c] + d.scanner.map(shift) * (1 + 0.05 * i) + rng.normal(0, 0.5, len(d)) * d.scanner.map(scale)
    d["saa"] = np.repeat(rng.integers(0, 2, ns), nv)        # protected, no true effect
    out, est = h.long_combat(d, feats, "scanner", "PATNO", "years", ["age_bl", "sex"], label="saa")
    assert est["confounding"]["table"].to_numpy().sum() == len(d)

    def within_slope(y):
        t = d.years - d.groupby("PATNO").years.transform("mean")
        return (t * (y - y.groupby(d.PATNO).transform("mean"))).sum() / (t * t).sum()

    raw_slope = np.mean([within_slope(d[c]) for c in feats])
    assert abs(raw_slope - slope) > 0.5
    assert abs(np.mean([within_slope(out[c]) for c in feats]) - slope) < 0.05
    raw_mean, raw_sd = site_spread(d, d.index, feats, truth, d.scanner)
    mean, sd = site_spread(out, d.index, feats, truth, d.scanner)
    assert mean < 0.05 * raw_mean and sd < 0.5 * raw_sd
    np.testing.assert_allclose(out[feats].mean(), d[feats].mean(), atol=0.05)  # sum(n_batch * effect) = 0
    g, dd = est["gammastarhat"], est["delta2starhat"]
    assert ((g.loc["B"] > g.loc["A"]) & (g.loc["A"] > g.loc["C"])).all()      # additive effects, ordered
    assert dd.loc["B"].mean() > dd.loc["A"].mean() > dd.loc["C"].mean()      # multiplicative effects, ordered
