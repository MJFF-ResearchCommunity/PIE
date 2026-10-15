"""Pre-registered positive control for PIE's survival layer (documentation/preregistration_survival_positive_control.md):
baseline DaT (PPMI core-lab lowest putamen SBR, % of age/sex-expected) predicts phenoconversion to PD or DLB in PPMI
prodromal participants. Interval-censored illness-death model with death competing (``survive.illness_death``);
cause-specific Cox as the comparator.

Primary: healthy -> PD/DLB hazard ratio per SD lower DaT, adjusted for age and sex; pass if the fit converged and the
95 % CI lower bound > 1.

    PYTHONPATH=. venv_py311/bin/python scripts/analyse_survival_positive_control.py --ppmi PPMI --out <dir>
    PYTHONPATH=. venv_py311/bin/python scripts/analyse_survival_positive_control.py --synthetic --out <dir>   # dry run: all fake
"""
import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pie.stats.survive import (cause_specific_cox, cumulative_incidence, fine_gray, illness_death,  # noqa: E402
                               interval_censored_regression, visit_intervals)

CONVERSION, WITH_MSA = (1, 5), (1, 5, 11)      # PRIMDIAG: idiopathic PD, DLB (+ MSA in a sensitivity analysis)
DAT_MONTHS, TAU, YEARS = 6, 5.0, (2, 4, 6)
COVS, DEATH_COVS = ["dat_low_sd", "age10", "sex"], ["age10"]


def _labels():
    """pie.imaging.labels without pie/imaging/__init__.py, which imports nibabel: absent from venv_py311, the one
    environment with lifelines (it needs pandas < 3)."""
    if "pie.imaging" not in sys.modules:
        pkg = types.ModuleType("pie.imaging")
        pkg.__path__ = [str(ROOT / "pie" / "imaging")]
        sys.modules["pie.imaging"] = pkg
    return importlib.import_module("pie.imaging.labels")


def load(ppmi):
    """Raw inputs from PPMI's tables: diagnosis visits, deaths, core-lab DaT scans, covariates (prodromal cohort only)."""
    L = _labels()
    ps = L._latest(ppmi, "_Subject_Characteristics", "Participant_Status")
    ids = set(ps.loc[ps["COHORT_DEFINITION"].eq("Prodromal"), "PATNO"])
    pr = L._latest(ppmi, "Medical_History", "Primary_Research_Diagnosis")
    pr = pr[pr["PATNO"].isin(ids)]
    visits = pd.DataFrame({"PATNO": pr["PATNO"], "date": L._month(pr["INFODT"]), "PRIMDIAG": pr["PRIMDIAG"]})
    visits["date_hi"] = visits["date"] + pd.offsets.MonthEnd(0)
    fo = L._latest(ppmi, "Study_Enrollment", "FOUND_Enrollment_Status")
    cl = L._latest(ppmi, "Study_Enrollment", "Conclusion_of_Study_Participation")
    fo, cl, st = fo[fo["FOVITAL"].eq(1)], cl[cl["WDDEATH"].eq(1)], ps[ps["ENROLL_STATUS"].eq("Withdraw Deceased")]
    first = lambda s: s.groupby(level=0).first()
    mid = pd.Timedelta(days=14)                    # month-only dates: mid-month
    deaths = (first(pd.to_datetime(fo.set_index("PATNO")["FODOD"], errors="coerce"))       # FOUND: full date
              .combine_first(first(L._month(cl.set_index("PATNO")["WDDT"]) + mid))         # conclusion form, death box
              .combine_first(first(L._month(st.set_index("PATNO")["STATUS_DATE"]) + mid)))  # enrolment status
    deaths = deaths.reindex(sorted((set(fo["PATNO"]) | set(cl["PATNO"]) | set(st["PATNO"])) & ids))
    dat = []
    for rank, src in enumerate(("xing", "invicro")):       # Xing first, Invicro where Xing has no scan in the window
        s = L.dat_sbr_scored(ppmi, src)
        dat.append(s.loc[s["PATNO"].isin(ids), ["PATNO", "DATSCAN_DATE", "sbr_pct_expected", "sbr_source"]].assign(rank=rank))
    cov = L.covariates(ppmi)
    cov = pd.DataFrame({"PATNO": cov["PATNO"], "BIRTHDT": cov["BIRTHDT"], "SEX": cov["SEX"], "ENROLL_DATE": L._month(cov["ENROLL_DATE"])})
    return visits, deaths, pd.concat(dat, ignore_index=True), cov[cov["PATNO"].isin(ids)]


def frame(visits, deaths, dat, cov, conversion=CONVERSION, reverted_censored=False):
    """One row per participant: (left, right] holding conversion, death or last-contact time, baseline DaT, age, sex;
    and the participant flow. Time 0 is the first diagnosis record; times in years."""
    v = visits.assign(dx=visits["PRIMDIAG"].isin(conversion).astype(float).where(visits["PRIMDIAG"].notna()))
    iv = visit_intervals(v, "PATNO", "date", "dx", time_upper="date_hi")
    flow = {"prodromal_with_diagnosis_record": len(iv)}
    iv["t0"] = iv["PATNO"].map(v.dropna(subset=["date", "dx"]).groupby("PATNO")["date"].min())
    flow["excluded_converted_at_first_record"] = int(iv["prevalent"].sum())
    iv = iv[~iv["prevalent"]].copy()
    flow["reverted_after_conversion"] = int(iv["reverted"].sum())
    if reverted_censored:
        r = iv["reverted"].to_numpy()
        iv.loc[r, "left"], iv.loc[r, "right"] = iv.loc[r, "last_visit"], np.inf
    dead = iv["PATNO"].isin(deaths.index).to_numpy()
    dt = ((iv["PATNO"].map(deaths) - iv["t0"]).dt.days / 365.25).to_numpy()
    iv["died"] = dead & np.isfinite(dt)
    flow["death_undated_censored"] = int((dead & ~np.isfinite(dt)).sum())
    conv = np.isfinite(iv["right"].to_numpy())
    floor = np.fmax(iv["left"].to_numpy(), np.where(conv, iv["right"], 0))
    t = np.where(iv["died"], dt, np.fmax(iv["last_visit"].to_numpy(), floor))
    flow["death_before_last_record_clipped"] = int((iv["died"] & (t < floor)).sum())
    iv["time"] = np.fmax(t, floor)
    no_fu = (~iv["died"] & (iv["n_visits"] < 2)).to_numpy()
    flow["excluded_no_follow_up"] = int(no_fu.sum())
    iv = iv[~no_fu]
    m = iv[["PATNO", "t0"]].merge(dat.dropna(subset=["DATSCAN_DATE", "sbr_pct_expected"]), on="PATNO")
    m["gap"] = (m["DATSCAN_DATE"] - m["t0"]).dt.days.abs() / 30.44
    m = m[m["gap"] <= DAT_MONTHS].sort_values(["PATNO", "rank", "gap"]).drop_duplicates("PATNO")
    iv = iv.merge(m[["PATNO", "sbr_pct_expected", "sbr_source"]], on="PATNO", how="left")
    flow["excluded_no_baseline_dat"] = int(iv["sbr_pct_expected"].isna().sum())
    c = cov.drop_duplicates("PATNO").set_index("PATNO")
    iv["age10"] = ((iv["t0"] - iv["PATNO"].map(c["BIRTHDT"])).dt.days / 365.25 - 65) / 10
    iv["sex"] = pd.to_numeric(iv["PATNO"].map(c["SEX"]), errors="coerce")
    iv["enrolled_before_2020"] = iv["PATNO"].map(c["ENROLL_DATE"]).dt.year < 2020
    flow["excluded_no_age_or_sex"] = int((iv["sbr_pct_expected"].notna() & iv[["age10", "sex"]].isna().any(axis=1)).sum())
    d = iv.dropna(subset=["sbr_pct_expected", "age10", "sex"]).copy()
    pct = d.groupby("sbr_source")["sbr_pct_expected"]
    d["dat_low_sd"] = -(d["sbr_pct_expected"] - pct.transform("mean")) / pct.transform("std")   # per SD lower, within core lab
    d["dat_d_nsdiss"] = (d["sbr_pct_expected"] <= 0.75).astype(int)
    conv = np.isfinite(d["right"].to_numpy())
    d["time_cs"] = np.where(conv, d["right"], d["time"])            # comparator: first diagnosed visit as the event time
    d["event_type"] = np.where(conv, 1, np.where(d["died"], 2, 0))
    return d.reset_index(drop=True), flow


def uno_c(time, event, risk, tau=TAU):
    """Uno's C (Uno et al. 2011, Stat Med 30:1105): IPCW concordance truncated at ``tau``; G = censoring KM at T-."""
    t, e, r = (np.asarray(x, float) for x in (time, event, risk))
    e = e.astype(bool)
    ct, dc = np.unique(t[~e], return_counts=True)
    G = np.r_[1.0, np.cumprod(1 - dc / (len(t) - np.searchsorted(np.sort(t), ct)))]
    i = e & (t < tau)
    w = 1 / G[np.searchsorted(ct, t[i])] ** 2
    comp = t[i][:, None] < t[None, :]
    conc = (r[i][:, None] > r[None, :]) + 0.5 * (r[i][:, None] == r[None, :])
    return float((w[:, None] * comp * conc).sum() / (w[:, None] * comp).sum())


def _hr(fit, pred="dat_low_sd", tr="01"):
    row = next(c for c in fit["coefficients"] if c["predictor"] == pred and c.get("transition", tr) == tr)
    return {k: float(row[k]) for k in ("hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value")}


def _idm(d, covs=COVS):
    """The illness-death fit, with the pre-specified fallback (no death covariates) if it does not converge."""
    fit = illness_death(d, "left", "right", "time", "died", covariates=covs, death_covariates=DEATH_COVS)
    if fit["converged"]:
        return fit, False
    return illness_death(d, "left", "right", "time", "died", covariates=covs), True


def _counts(d):
    conv = np.isfinite(d["right"])
    return {"n": int(len(d)), "conversions": int(conv.sum()), "deaths": int(d["died"].sum()),
            "deaths_after_conversion": int((d["died"] & conv).sum()), "median_follow_up_years": float(d["time"].median())}


def analyse(raw):
    d, flow = frame(*raw)
    res = {"flow": {**flow, "analysed": _counts(d), "by_dat_source": d["sbr_source"].value_counts().to_dict(),
                    "enrolled_before_2020": int(d["enrolled_before_2020"].sum())}}
    fit, fallback = _idm(d)
    tl, risk = np.array(fit["timeline"]), np.array(fit["absolute_risk"])
    res["primary"] = {**_hr(fit), "converged": bool(fit["converged"]), "fallback_no_death_covariates": fallback,
                      "coefficients": fit["coefficients"], "baseline": fit["baseline"],
                      "absolute_risk_mean_dat_age65_female": {y: float(np.interp(y, tl, risk)) for y in YEARS if y <= tl[-1]}}
    res["pass"] = bool(fit["converged"] and res["primary"]["hr_ci_lower"] > 1)
    cs = cause_specific_cox(d, "time_cs", "event_type", 1, COVS)
    res["comparator_cause_specific_cox"] = {**_hr(cs), "harrell_c_model": float(cs["concordance"]), "ph_test": cs["ph_test"],
                                            "ph_test_error": cs["ph_test_error"]}
    from lifelines.utils import concordance_index
    ev = d["event_type"].eq(1)
    res["discrimination_dat_alone"] = {"harrell_c": float(concordance_index(d["time_cs"], -d["dat_low_sd"], ev)),
                                       f"uno_c_tau{TAU:g}": uno_c(d["time_cs"], ev, d["dat_low_sd"])}
    cif = cumulative_incidence(d, "time_cs", "event_type", group="dat_d_nsdiss")
    at = lambda y: int(np.searchsorted(cif["timeline"], y, "right") - 1)
    res["secondary"] = {
        "unadjusted": _hr(_idm(d, ["dat_low_sd"])[0]),
        "fine_gray": _hr(fine_gray(d, "time_cs", "event_type", 1, COVS)),
        "interval_weibull_deaths_censored": _hr(interval_censored_regression(d, "left", "right", COVS)),
        "aalen_johansen_conversion_by_nsdiss_deficit": {g: {y: cif["cif"][g]["1"][at(y)] for y in YEARS if y <= cif["timeline"][-1]}
                                                        for g in cif["cif"] if "1" in cif["cif"][g]}}
    sens = {"msa_counts_as_conversion": frame(*raw, conversion=WITH_MSA)[0],
            "reverted_not_converted": frame(*raw, reverted_censored=True)[0],
            "xing_only": d[d["sbr_source"].eq("xing_cwm")],
            "enrolled_before_2020": d[d["enrolled_before_2020"]]}
    res["sensitivity"] = {k: {**_counts(x), **(_hr(_idm(x)[0]) if np.isfinite(x["right"]).sum() >= 10 else {"skipped": "< 10 conversions"})}
                          for k, x in sens.items()}
    return res


def synthetic(n=1200, hr=2.5, seed=1):
    """Fully synthetic raw inputs (no real value, label, date or identifier): annual visits, Weibull onset with
    ``hr`` per SD lower DaT, deaths before and after onset, month-resolution dates as PPMI's."""
    rng = np.random.default_rng(seed)
    z, age, sex = rng.normal(size=n), rng.normal(66, 7, n), rng.integers(0, 2, n)
    onset = 28 * (rng.exponential(size=n) / np.exp(np.log(hr) * z)) ** (1 / 1.3)     # Weibull, shape 1.3
    death = rng.exponential(1 / (0.01 * np.exp(0.07 * (age - 66))))
    death = np.where(onset < death, np.minimum(death, onset + rng.exponential(20, n)), death)   # faster after onset
    end = rng.uniform(1, 10, n)
    t0 = pd.Timestamp("2014-01-01") + pd.to_timedelta(rng.integers(0, 3000, n), "D")
    rows = []
    for i in range(n):
        last = int(min(end[i], death[i]))
        for k in range(last + 1):
            if k and rng.random() < 0.1:                                                      # missed visit
                continue
            code = (5 if i % 17 == 0 else 11 if i % 23 == 0 else 1) if onset[i] < k else 25   # DLB, MSA, PD
            if i % 11 == 0 and k == last and onset[i] < k - 1:                                # a reverted diagnosis
                code = 25
            rows.append((i, (t0[i] + pd.DateOffset(months=12 * k)).to_period("M").to_timestamp(), code))
    visits = pd.DataFrame(rows, columns=["PATNO", "date", "PRIMDIAG"])
    visits["date_hi"] = visits["date"] + pd.offsets.MonthEnd(0)
    dead = death < end
    deaths = pd.Series(t0[dead] + pd.to_timedelta(death[dead] * 365.25, "D"), index=np.flatnonzero(dead))
    src = np.where(rng.random(n) < 0.8, "xing_cwm", "invicro_occipital")
    dat = pd.DataFrame({"PATNO": np.arange(n), "DATSCAN_DATE": (t0 - pd.Timedelta(days=30)).to_period("M").to_timestamp(),
                        "sbr_pct_expected": 0.95 - 0.15 * z + np.where(src == "xing_cwm", 0, 0.05), "sbr_source": src,
                        "rank": (src != "xing_cwm").astype(int)})
    cov = pd.DataFrame({"PATNO": np.arange(n), "BIRTHDT": t0 - pd.to_timedelta(age * 365.25, "D"), "SEX": sex, "ENROLL_DATE": t0})
    return visits, deaths, dat, cov


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="dry run on fully synthetic data (reads nothing real)")
    a = ap.parse_args(argv)
    res = analyse(synthetic() if a.synthetic else load(a.ppmi))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("synthetic_result.json" if a.synthetic else "survival_positive_control_result.json")).write_text(json.dumps(res, indent=1, default=float))
    p, f = res["primary"], res["flow"]["analysed"]
    print(f"{'synthetic dry run: ' if a.synthetic else ''}HR per SD lower DaT {p['hazard_ratio']:.2f} [{p['hr_ci_lower']:.2f}, "
          f"{p['hr_ci_upper']:.2f}] (n {f['n']}, conversions {f['conversions']}, deaths {f['deaths']}; converged {p['converged']}) "
          f"-> {'PASS' if res['pass'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    main()
