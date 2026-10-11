import numpy as np
import pandas as pd
import pytest
from pie.stats.survive import (
    kaplan_meier, logrank_test, cox_regression, turnbull, interval_censored_regression,
    visit_intervals, cumulative_incidence, cause_specific_cox, fine_gray, illness_death,
)


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
    assert r["ph_test_error"] is None


def test_cox_regression_reports_a_failed_ph_test(surv_df, monkeypatch):
    import lifelines.statistics

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic PH failure")
    monkeypatch.setattr(lifelines.statistics, "proportional_hazard_test", broken)
    with pytest.warns(RuntimeWarning, match="synthetic PH failure"):
        r = cox_regression(surv_df, time="time", event="event", covariates=["group", "age"])
    assert r["ph_test"] == [] and "synthetic PH failure" in r["ph_test_error"]


# --- interval censoring -----------------------------------------------------------------

def _visit_grid(t, visits):
    """(L, R] from true event times seen only at `visits`; right-censored after the last."""
    L = np.array([visits[visits < x].max() for x in t])
    R = np.array([visits[visits >= x].min() if x <= visits[-1] else np.inf for x in t])
    return np.where(np.isinf(R), visits[-1], L), R


def test_turnbull_equals_the_empirical_distribution_on_a_shared_visit_grid():
    rng = np.random.default_rng(1)
    L, R = _visit_grid(rng.exponential(5, 500), np.arange(0, 11.0))
    r = turnbull(pd.DataFrame({"L": L, "R": R}), "L", "R")
    assert r["converged"]
    S = dict(zip(r["upper"], r["survival"]))
    for k in range(1, 11):  # with every interval on one grid the NPMLE is the empirical cdf
        assert S[float(k)] == pytest.approx((R > k).mean(), abs=1e-6)
    # lifelines' closed-interval NPMLE gets S(1) = 1.0 on this data; the truth is 0.80
    assert S[1.0] == pytest.approx(0.804, abs=1e-3)


def test_turnbull_equals_kaplan_meier_for_exact_and_right_censored_times():
    from lifelines import KaplanMeierFitter
    rng = np.random.default_rng(2)
    t, c = rng.exponential(5, 200).round(2), rng.uniform(0, 10, 200).round(2)
    obs = np.minimum(t, c)
    df = pd.DataFrame({"L": obs, "R": np.where(t <= c, obs, np.inf)})
    r = turnbull(df, "L", "R")
    km = KaplanMeierFitter().fit(obs, t <= c)
    for up, s in zip(r["upper"], r["survival"]):
        if np.isfinite(up):
            assert s == pytest.approx(float(km.predict(up)), abs=1e-5)


def test_turnbull_recovers_the_truth_with_irregular_visits():
    rng = np.random.default_rng(3)
    n = 1000
    t = rng.weibull(1.5, n) * 8
    L, R = np.zeros(n), np.full(n, np.inf)
    for i in range(n):  # each subject has their own irregular visit schedule
        v = np.r_[0, np.cumsum(rng.uniform(0.5, 2.5, 12))]
        v = v[v <= 15]
        j = np.searchsorted(v, t[i])  # first visit at or after the event
        L[i], R[i] = (v[j - 1], v[j]) if j < len(v) else (v[-1], np.inf)
    r = turnbull(pd.DataFrame({"L": L, "R": R}), "L", "R")
    assert r["converged"]
    up, S = np.array(r["upper"]), np.array(r["survival"])
    for x in (2.0, 5.0, 10.0):
        assert S[up <= x][-1] == pytest.approx(np.exp(-(x / 8) ** 1.5), abs=0.04)


@pytest.fixture(scope="module")
def weibull_visits():
    """Weibull PH (shape 1.5, scale 8, HR 2 per unit x) seen at yearly visits for 15 years."""
    rng = np.random.default_rng(4)
    n = 1000
    x = rng.normal(size=n)
    t = 8 * (rng.exponential(size=n) * np.exp(-np.log(2) * x)) ** (1 / 1.5)
    L, R = _visit_grid(t, np.arange(0, 16.0))
    return pd.DataFrame({"L": L, "R": R, "x": x, "noise": rng.normal(size=n)})


def test_interval_censored_regression_recovers_the_weibull_hazard_ratio(weibull_visits):
    r = interval_censored_regression(weibull_visits, "L", "R", ["x", "noise"])
    c = {row["predictor"]: row for row in r["coefficients"]}
    assert c["x"]["hr_ci_lower"] < 2.0 < c["x"]["hr_ci_upper"]
    assert c["x"]["hazard_ratio"] == pytest.approx(2.0, rel=0.12)
    assert c["noise"]["hr_ci_lower"] < 1.0 < c["noise"]["hr_ci_upper"]
    # AFT and PH are one model for the Weibull: time ratio = HR^(-1/shape), shape 1.5
    assert c["x"]["time_ratio"] == pytest.approx(2.0 ** (-1 / 1.5), rel=0.1)
    assert r["n_right_censored"] == int(np.isinf(weibull_visits["R"]).sum())


def test_interval_censored_regression_other_families(weibull_visits):
    r = interval_censored_regression(weibull_visits, "L", "R", ["x"], dist="lognormal")
    row = r["coefficients"][0]
    assert row["time_ratio"] < 1 and "hazard_ratio" not in row


def test_visit_intervals_builds_left_right_from_a_long_visit_table():
    v = pd.DataFrame([
        ("a", 0, 0), ("a", 2, 0), ("a", 4, 1), ("a", 6, 0),       # reverted after diagnosis
        ("b", 0, 0), ("b", 3, 0), ("b", np.nan, 0),               # never diagnosed, one undated visit
        ("c", 0, 1), ("c", 1, 1),                                 # prevalent
        ("d", 0, 0), ("d", np.nan, 1),                            # only diagnosis is undated
    ], columns=["id", "t", "dx"])
    r = visit_intervals(v, "id", "t", "dx").set_index("id")
    assert (r.loc["a", "left"], r.loc["a", "right"], r.loc["a", "reverted"]) == (2, 4, True)
    assert (r.loc["b", "left"], r.loc["b", "right"], r.loc["b", "n_visits"]) == (3, np.inf, 2)
    assert r.loc["c", "prevalent"] and np.isnan(r.loc["c", "left"])
    assert r.loc["d", "undated_diagnosis"] and r.loc["d", "right"] == np.inf
    assert not r.loc["a", "undated_diagnosis"] and not r.loc["a", "prevalent"]


def test_visit_intervals_widens_masked_dates_and_converts_datetimes():
    v = pd.DataFrame({
        "id": ["a"] * 3,
        "lo": pd.to_datetime(["2012-03-01", "2014-03-01", "2016-03-01"]),
        "hi": pd.to_datetime(["2012-03-31", "2014-03-31", "2016-03-31"]),  # month-masked
        "dx": [0, 0, 1],
    })
    r = visit_intervals(v, "id", "lo", "dx", time_upper="hi").iloc[0]
    assert r["left"] == pytest.approx(730 / 365.25)          # start of the undiagnosed month
    assert r["right"] == pytest.approx((1461 + 30) / 365.25)  # end of the diagnosed month


# --- competing risks --------------------------------------------------------------------

def _simulate_fine_gray(n, rng, beta1, p=0.3, beta2=0.5, cens_max=None):
    """Fine & Gray (1999) / Beyersmann et al. (2009) scheme: subdistribution HR exp(beta1).

    F1(t|x) = 1 - (1 - p(1 - e^-t))^exp(beta1 x); otherwise cause 2 ~ Exp(rate exp(beta2 x)).
    """
    x = rng.normal(size=n)
    u = rng.uniform(size=n)
    cause1 = u < 1 - (1 - p) ** np.exp(beta1 * x)
    with np.errstate(invalid="ignore"):
        t1 = -np.log(1 - (1 - (1 - u) ** np.exp(-beta1 * x)) / p)
    t = np.where(cause1, t1, rng.exponential(1 / np.exp(beta2 * x)))
    ev = np.where(cause1, 1, 2)
    if cens_max:
        c = rng.uniform(0, cens_max, n)
        ev, t = np.where(t <= c, ev, 0), np.minimum(t, c)
    return pd.DataFrame({"t": t, "ev": ev, "x": x})


def test_cumulative_incidence_with_one_cause_is_one_minus_km(surv_df):
    r = cumulative_incidence(surv_df, "time", "event")
    km = kaplan_meier(surv_df, "time", "event")
    np.testing.assert_allclose(r["cif"]["_overall"]["1"], 1 - np.array(km["survival"]["_overall"]),
                               atol=1e-12)


def test_cumulative_incidence_recovers_truth_and_sums_to_one():
    rng = np.random.default_rng(5)
    df = _simulate_fine_gray(3000, rng, beta1=0.0, beta2=0.0, cens_max=4.0)
    r = cumulative_incidence(df, "t", "ev")
    tl, F1, F2 = (np.array(v) for v in (r["timeline"], r["cif"]["_overall"]["1"],
                                         r["cif"]["_overall"]["2"]))
    for x in (0.5, 1.0, 2.0):
        i = np.argmin(abs(tl - x))
        assert F1[i] == pytest.approx(0.3 * (1 - np.exp(-tl[i])), abs=0.02)
        assert F2[i] == pytest.approx(0.7 * (1 - np.exp(-tl[i])), abs=0.03)
    S = np.array(kaplan_meier(df.assign(any=df.ev > 0), "t", "any")["survival"]["_overall"])
    np.testing.assert_allclose(F1 + F2 + S, 1.0, atol=1e-9)
    g = cumulative_incidence(df.assign(grp=df.x > 0), "t", "ev", group="grp")
    assert set(g["cif"]) == {"False", "True"} and g["causes"] == [1, 2]


def test_cumulative_incidence_ci_has_nominal_coverage():
    rng = np.random.default_rng(6)
    hits = []
    for _ in range(300):
        df = _simulate_fine_gray(200, rng, beta1=0.0, cens_max=4.0)
        r = cumulative_incidence(df, "t", "ev")
        i = int(np.argmin(abs(np.array(r["timeline"]) - 1.0)))
        truth = 0.3 * (1 - np.exp(-r["timeline"][i]))
        hits.append(r["ci_lower"]["_overall"]["1"][i] <= truth <= r["ci_upper"]["_overall"]["1"][i])
    assert 0.91 <= np.mean(hits) <= 0.985


def test_cause_specific_cox_recovers_the_cause_specific_hazard_ratio():
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.normal(size=n)
    t1, t2 = rng.exponential(1 / (0.1 * np.exp(0.7 * x))), rng.exponential(1 / (0.05 * np.exp(-0.5 * x)))
    c = rng.uniform(0, 15, n)
    t = np.minimum.reduce([t1, t2, c])
    ev = np.select([t == t1, t == t2], [1, 2], 0).astype(float)
    ev[:10] = np.nan  # unknown cause: dropped, not counted as censored
    r = cause_specific_cox(pd.DataFrame({"t": t, "ev": ev, "x": x}), "t", "ev", 1, ["x"])
    hr = r["coefficients"][0]
    assert hr["hr_ci_lower"] < np.exp(0.7) < hr["hr_ci_upper"]
    assert r["n"] == n - 10 and r["cause"] == 1


def test_fine_gray_without_censoring_is_cox_with_competing_events_kept_at_risk():
    # With no censoring every weight is 1, and Fine-Gray reduces to an ordinary Cox model in
    # which subjects with a competing event stay in the risk set to the end: same estimate,
    # and its robust SE is lifelines' robust (sandwich) SE for that model.
    from lifelines import CoxPHFitter
    df = _simulate_fine_gray(400, np.random.default_rng(8), beta1=0.5)
    df["z"] = np.random.default_rng(80).normal(size=len(df))
    fg = fine_gray(df, "t", "ev", 1, ["x", "z"])
    kept = df.assign(t=np.where(df.ev == 2, df.t.max() + 1, df.t), e1=(df.ev == 1).astype(int))
    cph = CoxPHFitter().fit(kept[["t", "e1", "x", "z"]], "t", "e1", robust=True)
    for row in fg["coefficients"]:
        assert row["coef"] == pytest.approx(cph.params_[row["predictor"]], abs=1e-6)
        assert row["se"] == pytest.approx(cph.standard_errors_[row["predictor"]], rel=1e-4)


def test_fine_gray_recovers_the_subdistribution_hazard_ratio():
    df = _simulate_fine_gray(2000, np.random.default_rng(9), beta1=0.5, cens_max=3.0)
    assert 0.2 < (df.ev == 0).mean() < 0.5
    r = fine_gray(df, "t", "ev", 1, ["x"])
    c = r["coefficients"][0]
    assert c["hr_ci_lower"] < np.exp(0.5) < c["hr_ci_upper"]
    assert c["coef"] == pytest.approx(0.5, abs=0.15)
    assert r["n_competing"] == int((df.ev == 2).sum())
    # the cause-specific HR answers a different question and is not exp(0.5) here
    cs = cause_specific_cox(df, "t", "ev", 1, ["x"])["coefficients"][0]["coef"]
    assert abs(cs - 0.5) > 0.1


# --- illness-death ----------------------------------------------------------------------

_K, _S = (1.5, 2.0, 1.5), (20.0, 25.0, 6.0)  # Weibull shape/scale: healthy->PD, healthy->dead, PD->dead


def _simulate_illness_death(n, rng, b01, b12, k=_K, s=_S, gap=3.0, follow=15.0):
    """Markov illness-death, Weibull intensities on time since baseline, binary x acting on
    healthy->PD (b01) and PD->dead (b12). PD is only seen at visits every `gap` years."""
    x = rng.integers(0, 2, n).astype(float)

    def inv(E, j, A0=0.0):  # t with (t / s)^k = A0 + E
        return s[j] * (A0 + E) ** (1 / k[j])

    t01 = inv(rng.exponential(size=n) * np.exp(-b01 * x), 0)
    t02 = inv(rng.exponential(size=n), 1)
    ill = t01 < t02
    t12 = inv(rng.exponential(size=n) * np.exp(-b12 * x), 2, (t01 / s[2]) ** k[2])
    death = np.where(ill, t12, t02)
    T, onset = np.minimum(death, follow), np.where(ill, t01, np.inf)
    v = np.arange(0, follow + 1e-9, gap)
    seen = (v >= onset[:, None]) & (v <= T[:, None])
    R = np.where(seen.any(1), v[np.argmax(seen, 1)], np.inf)
    L = np.where((v < onset[:, None]) & (v <= T[:, None]), v, 0).max(1)
    return pd.DataFrame({"L": L, "R": R, "T": T, "dead": (death <= follow).astype(int), "x": x,
                         "onset": onset})


@pytest.fixture(scope="module")
def idm_fit():
    """x doubles PD incidence and triples mortality after PD (e.g. a marker of a rapidly fatal
    synucleinopathy). Visits every 3 years: many PD cases die before they can be diagnosed."""
    df = _simulate_illness_death(1500, np.random.default_rng(10), b01=np.log(2), b12=np.log(3))
    fit = illness_death(df, "L", "R", "T", "dead", covariates=["x"], death_covariates=["x"])
    return df, fit


def test_illness_death_recovers_transition_parameters(idm_fit):
    df, r = idm_fit
    assert r["converged"]
    c = {(row["transition"], row["predictor"]): row for row in r["coefficients"]}
    for key, truth in ((("01", "x"), 2.0), (("02", "x"), 1.0), (("12", "x"), 3.0)):
        assert c[key]["hr_ci_lower"] < truth < c[key]["hr_ci_upper"], key
    assert np.log(c[("01", "x")]["hazard_ratio"]) == pytest.approx(np.log(2), abs=0.25)
    assert np.log(c[("12", "x")]["hazard_ratio"]) == pytest.approx(np.log(3), abs=0.25)
    for j, tr in enumerate(("01", "02", "12")):
        assert r["baseline"][tr]["shape"] == pytest.approx(_K[j], rel=0.2), tr
        assert r["baseline"][tr]["scale"] == pytest.approx(_S[j], rel=0.2), tr


def test_illness_death_absolute_risk_accounts_for_death(idm_fit):
    from scipy.integrate import quad
    _, r = idm_fit
    k, s = _K, _S

    def onset_density(u):  # x = 0: healthy until u, then PD
        return np.exp(-(u / s[0]) ** k[0] - (u / s[1]) ** k[1]) * k[0] / s[0] * (u / s[0]) ** (k[0] - 1)

    tl = np.array(r["timeline"])
    for x in (5.0, 10.0, 14.0):
        i = int(np.argmin(abs(tl - x)))
        truth = quad(onset_density, 0, tl[i])[0]
        # Over 30 replicates of this design the error at t = 10 averaged -0.001 (SD 0.018)
        assert r["risk_ci_lower"][i] < truth < r["risk_ci_upper"][i]
        assert r["absolute_risk"][i] == pytest.approx(truth, abs=0.05)


def test_naive_cox_on_visit_dated_diagnoses_is_biased_where_illness_death_is_not(idm_fit):
    df, r = idm_fit
    diag = np.isfinite(df.R)
    naive = df.assign(t=np.where(diag, df.R, df["T"]), pd_dx=diag.astype(int))  # deaths censored
    hr_naive = cox_regression(naive, "t", "pd_dx", ["x"])["coefficients"][0]
    hr_idm = next(c for c in r["coefficients"] if c["transition"] == "01")
    assert hr_naive["hr_ci_upper"] < 2.0  # true HR 2 excluded: high-x cases die undiagnosed
    assert hr_idm["hr_ci_lower"] < 2.0 < hr_idm["hr_ci_upper"]
    # the bias is the missed cases: cases (true onset in follow-up) never diagnosed
    missed = (df.onset <= df["T"]) & ~diag
    assert missed[df.x == 1].mean() > 2 * missed[df.x == 0].mean()


def test_illness_death_piecewise_constant_with_exact_dates():
    # 40 replicates of this design (Weibull fit): mean log HR 0.679 vs 0.693, coverage 0.925.
    # Seed 11 drew HR 2.43 under all four fits (Weibull/piecewise x interval/exact).
    rng = np.random.default_rng(12)
    df = _simulate_illness_death(800, rng, b01=np.log(2), b12=0.0, k=(1.0, 1.0, 1.0),
                                 s=(15.0, 30.0, 8.0))  # constant hazards 1/15, 1/30, 1/8
    exact = np.isfinite(df.R) & (rng.uniform(size=len(df)) < 0.3)  # some onset dates are known
    df.loc[exact, "L"] = df.loc[exact, "R"] = df.loc[exact, "onset"]
    r = illness_death(df, "L", "R", "T", "dead", covariates=["x"], hazard="piecewise",
                      breaks=[0, 5, 10])
    for tr, rate in (("01", 1 / 15), ("02", 1 / 30), ("12", 1 / 8)):
        np.testing.assert_allclose(r["baseline"][tr]["rates"], rate, rtol=0.35, err_msg=tr)
    hr = r["coefficients"][0]
    assert hr["hr_ci_lower"] < 2.0 < hr["hr_ci_upper"]


def test_illness_death_rejects_impossible_times():
    df = pd.DataFrame({"L": [0.0, 1.0], "R": [2.0, np.inf], "T": [1.5, 3.0], "dead": [0, 0]})
    with pytest.raises(ValueError, match="must be >="):
        illness_death(df, "L", "R", "T", "dead")
    with pytest.raises(ValueError, match="breaks"):
        illness_death(df.assign(T=3.0), "L", "R", "T", "dead", hazard="piecewise", breaks=[1, 5])
