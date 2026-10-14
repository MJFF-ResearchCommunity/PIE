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


def test_index_keeps_only_each_participants_earliest_three_shell_session_with_its_b0s():
    import pandas as pd
    spec2 = importlib.util.spec_from_file_location("dwi_fw_index", Path(__file__).resolve().parent.parent / "scripts" / "dwi_fw_index.py")
    ix = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ix)
    rows = [(1, "2012-01-01", "DTI_LR"), (1, "2023-02-01", "DTI_B700_64dir_PA"), (1, "2023-02-01", "DTI_B1000_64dir_PA"),
            (1, "2023-02-01", "DTI_revB0_AP"), (1, "2024-02-01", "DTI_B700_64dir_PA"), (1, "2024-02-01", "DTI_b2000_64dir_PA"),
            (2, "2023-05-01", "DTI_B1000_64dir_PA"), (2, "2023-05-01", "DTI_revB0_AP")]          # 2: one shell only
    idx = pd.DataFrame(rows, columns=["patno", "date", "desc"])
    out = ix.earliest_three_shell(idx)
    assert set(out["patno"]) == {1} and set(out["date"]) == {"2023-02-01"} and len(out) == 3
