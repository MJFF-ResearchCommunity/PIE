"""The pre-registered survival positive control on fully synthetic data (no real value, label, date or identifier).
Needs lifelines: run with venv_py311."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("analyse_survival_positive_control",
                                              Path(__file__).resolve().parent.parent / "scripts" / "analyse_survival_positive_control.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def test_known_dat_effect_is_recovered_with_death_competing_and_sensitivities_move_the_right_way():
    res = pc.analyse(pc.synthetic(n=700, hr=2.5, seed=2))
    p, f = res["primary"], res["flow"]
    assert res["pass"] and p["converged"] and p["hr_ci_lower"] < 2.5 < p["hr_ci_upper"]
    assert f["analysed"]["deaths"] > 0 and f["analysed"]["deaths_after_conversion"] > 0 and f["excluded_no_follow_up"] > 0
    assert res["comparator_cause_specific_cox"]["hazard_ratio"] > 1.5
    disc = res["discrimination_dat_alone"]
    assert 0.65 < disc["harrell_c"] < 0.9 and 0.65 < disc["uno_c_tau5"] < 0.9
    s = res["sensitivity"]
    assert s["msa_counts_as_conversion"]["conversions"] > f["analysed"]["conversions"] > s["reverted_not_converted"]["conversions"]
    assert s["xing_only"]["n"] < f["analysed"]["n"]


def test_null_effect_is_not_found():
    d, _ = pc.frame(*pc.synthetic(n=700, hr=1.0, seed=3))
    fit, _ = pc._idm(d)
    hr = pc._hr(fit)
    assert 0.75 < hr["hazard_ratio"] < 1.33 and hr["hr_ci_lower"] < 1 < hr["hr_ci_upper"]


def test_uno_c_ranks_and_ignores_events_after_tau():
    t, e = np.array([1.0, 2, 3, 4, 6]), np.array([1, 1, 0, 1, 1])
    assert pc.uno_c(t, e, -t) == 1.0 and pc.uno_c(t, e, t) == 0.0
    t, e, r = np.r_[t, 7], np.r_[e, 1], np.array([6.0, 5, 4, 3, 1, 2])        # the pair (6, 7) is misranked
    assert pc.uno_c(t, e, r, tau=5) == 1.0 and pc.uno_c(t, e, r, tau=10) < 1.0
