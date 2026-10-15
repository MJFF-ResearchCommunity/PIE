"""The pre-registered longitudinal-stream pilot analysis on fully synthetic data (no real value, label or identifier)."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("analyse_t1_longitudinal_pilot",
                                              Path(__file__).resolve().parent.parent / "scripts" / "analyse_t1_longitudinal_pilot.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)


def test_known_change_and_zero_change_pass_and_a_failed_template_is_excluded():
    s, flow = lp.synthetic()
    s.loc[s["tid"].eq("p0") & s["stream"].eq("long") & s["IMAGEID"].str.endswith("_1"), "vol_Left_Putamen"] = np.nan
    res = lp.analyse(s, flow)
    assert res["pass"] and res["flow"]["excluded_long_failure"] == {"pilot": 1}
    assert res["zero_change"]["n_values"] == 29 * 15 and res["zero_change"]["structures_long_mdc95_lower"] >= 12
    h = res["plausibility_hc"]["Hippocampus"]
    assert h["ci95"][0] < -0.9 < h["ci95"][1] and res["pd_minus_hc"]["Hippocampus"]["pd_minus_hc_median"] < 0
    sd = res["sd_annual_change_long_vs_cross"]["Hippocampus"]
    assert sd["long"] < sd["cross"]


def test_noisy_long_stream_or_no_atrophy_fails():
    s, flow = lp.synthetic()
    noisy = s.copy()
    m = noisy["role"].eq("sameday") & noisy["stream"].eq("long")
    noisy.loc[m, lp.KEYS] *= np.random.default_rng(0).normal(1, 0.05, (m.sum(), len(lp.KEYS)))
    assert not lp.analyse(noisy, flow)["gates"]["zero_change"]
    flat = s.copy()
    hc = flat["role"].eq("pilot") & flat["cohort"].eq("HC") & flat["stream"].eq("long")
    flat.loc[hc, lp.KEYS] = flat[hc].groupby("tid")[lp.KEYS].transform("first").to_numpy()      # no change at all
    assert not lp.analyse(flat, flow)["gates"]["plausibility"]
