"""The pre-registered free-water analysis on fully synthetic data (no real value, label or identifier)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("analyse_dwi_fw", Path(__file__).resolve().parent.parent / "scripts" / "analyse_dwi_fw.py")
fw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fw)


def test_synthetic_effect_is_recovered_and_read_against_the_anchor():
    d, flow = fw.synthetic(d_true=0.45)
    res = fw.analyse(d, flow, n_boot=60)
    assert 0.15 < res["d"] < 0.75 and res["d_ci95"][0] < res["d"] < res["d_ci95"][1]
    assert res["hc_age_rho"] > 0 and set(res["batches"]) <= {"Siemens|topup_eddy", "Siemens|eddy_only", "GE|topup_eddy", "GE|eddy_only",
                                                             "Philips|topup_eddy", "Philips|eddy_only", "Siemens|any", "GE|any", "Philips|any"}
    assert fw.verdict(0.2, 0.7) == "consistent with the published PPMI effect" and fw.verdict(0.05, 0.3) == "present but smaller than published"
    assert fw.verdict(-0.2, 0.3) == "not detected; excludes the published effect" and fw.verdict(-0.1, 0.6) == "inconclusive"
    assert fw.verdict(0.6, 1.0) == "larger than published"
