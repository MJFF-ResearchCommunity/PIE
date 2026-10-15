"""The pre-registered QSIPrep agreement analysis on fully synthetic data (no real value, label or identifier)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("qa", Path(__file__).resolve().parent.parent / "scripts" / "analyse_dwi_qsiprep_agreement.py")
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


def test_good_agreement_passes_and_poor_agreement_fails():
    res = qa.analyse(*qa.synthetic(icc=0.92), n_boot=50)
    p = res["primary"]
    assert res["pass"] and p["icc21"] >= 0.75 and p["icc21_ci95"][0] < p["icc21"] < p["icc21_ci95"][1] and p["bias"] > 0
    assert set(res["primary_by_tier"]) == {"topup_eddy", "eddy_only"} and set(res["secondary"]) == set(qa.FW_SECONDARY + qa.TENSOR)
    assert res["flow"]["both_fw_qc_pass"] < 40 and sum(res["flow"]["qsiprep_fw_qc_fail_by_check"].values()) == 40 - res["flow"]["both_fw_qc_pass"]
    assert not qa.analyse(*qa.synthetic(icc=0.4), n_boot=20)["pass"]


def test_selection_reads_no_measured_value_and_is_seeded_stratified_with_replacement(tmp_path):
    n = 60
    rows = pd.DataFrame({"patno": np.arange(n), "acquisition_date": ["2023-01-01"] * 55 + ["2015-01-01"] * 5, "error": "",
                         "dwisf_qc_pass": [True] * 50 + [False] * 10, "dwisf_fw_sn_posterior_mean": "not a number",
                         "dwisf_tier": ["topup_eddy"] * 44 + ["eddy_no_reverse_PE"] * 6 + ["topup_eddy"] * 5 + ["rigid"] * 5})
    rows.to_csv(tmp_path / "t.csv", index=False)
    c = qa.candidates([tmp_path / "t.csv"])
    assert "dwisf_fw_sn_posterior_mean" not in c and c["tier"].value_counts().to_dict() == {"topup_eddy": 44, "eddy_only": 6}
    order = qa.draw(c)
    assert order.equals(qa.draw(c.sample(frac=1, random_state=1))) and qa.quotas(order) == {"eddy_only": 6, "topup_eddy": 34}
    first = order[order["tier"].eq("topup_eddy")]["patno"].tolist()
    dwiq = pd.DataFrame({"patno": order["patno"], "error": ""})
    dwiq.loc[dwiq["patno"].eq(first[0]), "error"] = "RuntimeError: container failed"
    used, failed = qa.analysable(order, dwiq)
    assert len(used) == 40 and first[0] not in used and first[34] in used and failed == {"RuntimeError": 1}
