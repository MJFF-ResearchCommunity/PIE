"""The pre-registered free-water analysis on fully synthetic data (no real value, label or identifier)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("analyse_dwi_fw", Path(__file__).resolve().parent.parent / "scripts" / "analyse_dwi_fw.py")
fw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fw)


def test_synthetic_dat_separation_is_recovered_within_strata_and_null_does_not_pass():
    d, flow = fw.synthetic(auc_true=0.70)
    res = fw.analyse(d, flow, n_boot=60)
    p = res["primary"]
    assert 0.6 < p["auroc"] < 0.8 and p["ci95"][0] < p["auroc"] < p["ci95"][1] and res["pass"]
    assert p["n_pos"] + p["n_neg"] == 240 and set(p["strata_used"]) <= {"Siemens|topup_eddy", "Siemens|eddy_only", "GE|eddy_only", "Philips|eddy_only"}
    assert res["secondary"]["spearman_vs_sbr_pct_expected"]["rho"] < 0 and res["secondary"]["pd_vs_dat_normal_prodromal"]["auroc"] > 0.6
    null = fw.analyse(*fw.synthetic(auc_true=0.5), n_boot=60)
    assert abs(null["primary"]["auroc"] - 0.5) < 0.1
