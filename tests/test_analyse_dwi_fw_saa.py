"""The pre-registered free-water x SAA analysis on fully synthetic data (no real value, label or identifier)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
spec = importlib.util.spec_from_file_location("analyse_dwi_fw_saa", Path(__file__).resolve().parent.parent / "scripts" / "analyse_dwi_fw_saa.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)


def fake_increment(frame):
    assert {"PATNO", "saa_prodromal", "manufacturer", "model", "tier", fs.FEATURE, *fs.CLINICAL} <= set(frame)
    assert frame["PATNO"].is_monotonic_increasing
    row = {"n": len(frame), "delta_auc": 0.01, "delta_ci95_low": -0.01, "delta_ci95_high": 0.03, "delta_ci98_75_low": -0.02,
           "delta_ci98_75_high": 0.04, "positive_in_all_repeats": True}
    return {"primary": {"selected": row, "fw": row}, "scanner_adjusted": {"selected": row, "fw": row}}


def test_age_confounded_saa_difference_is_removed_by_adjustment_and_a_real_one_survives():
    m, pro, flow = fs.synthetic(saa_shift=0.0)
    res = fs.analyse(m, pro, flow, n_boot=60, increment=fake_increment)
    assert res["secondary"]["unadjusted"]["auroc"] > 0.58                    # SAA+ are older, free water rises with age
    assert abs(res["primary"]["auroc"] - 0.5) < 0.08 and not res["associated"]
    real = fs.analyse(*fs.synthetic(saa_shift=0.8), n_boot=60, increment=fake_increment)
    assert real["primary"]["auroc"] > 0.62 and real["associated"]
    assert set(real["secondary"]) >= {"unadjusted", "mwu_one_sided_p", "odds_ratio_per_sd", "dat_normal_only", "saa_by_dat",
                                      "within_6_months", "anterior_sn_fw"}
    assert real["increment"]["demonstrated"] is False                         # 98.75 % lower bound below zero
    by_role = real["secondary"]["by_role"]
    assert set(by_role) == {"test", "development_calibration"} and all(v["auroc"] > 0.55 for v in by_role.values())
    assert sum(sum(v.values()) for v in real["flow"]["analysed_by_role"].values()) == 300


def test_increment_rule_needs_the_9875_bound_all_repeats_and_no_reversal():
    good = {"delta_auc": 0.03, "delta_ci98_75_low": 0.001, "positive_in_all_repeats": True}
    assert fs.demonstrated({"primary": {"selected": good}, "scanner_adjusted": {"selected": good}})
    assert not fs.demonstrated({"primary": {"selected": {**good, "positive_in_all_repeats": False}}, "scanner_adjusted": {"selected": good}})
    assert not fs.demonstrated({"primary": {"selected": good}, "scanner_adjusted": {"selected": {**good, "delta_auc": -0.001}}})


def test_join_keeps_clinical_cohort_prodromal_within_18_months_of_the_csf_month():
    d = pd.DataFrame({"patno": [1, 2, 3, 4], "PATNO": [1, 2, 3, 4], "group": ["prodromal"] * 3 + ["PD"],
                      "date": pd.to_datetime(["2022-01-10", "2024-01-10", "2022-01-10", "2022-01-10"])})
    cl = pd.DataFrame({"PATNO": [1, 2, 4], "saa_prodromal": [1, 0, 1], "anchor_start": ["2021-06-01"] * 3, "anchor_end": ["2021-06-30"] * 3})
    m, flow = fs.join(d, cl)
    assert list(m["patno"]) == [1] and flow == {"prodromal_qc_pass": 3, "in_clinical_cohort": 2, "within_18_months": 1}
    assert 6 < m["csf_gap_months"].iloc[0] < 7


def test_real_run_stops_unless_the_frozen_dat_test_passed(tmp_path):
    dat = tmp_path / "fw_result.json"
    dat.write_text(json.dumps({"pass": False}))
    res = fs.main(["--tables", "x.csv", "--study", "s", "--dat-result", str(dat), "--out", str(tmp_path / "o")])
    assert res == {"ran": False, "reason": "DWI_FW_VALIDATED not reached: the frozen DaT test did not pass"}
    assert json.loads((tmp_path / "o" / "fw_saa_result.json").read_text())["ran"] is False


def test_synthetic_is_fully_fake():
    m, pro, flow = fs.synthetic()
    assert flow == {"synthetic": True} and m["patno"].max() < 1000 and np.isin(m["saa_prodromal"], [0, 1]).all()
