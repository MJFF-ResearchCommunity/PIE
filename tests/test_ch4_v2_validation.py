import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ch4_v2_validation as v2  # noqa: E402


def test_ixi_file_names_give_the_id_and_site(tmp_path):
    for name in ("IXI002-Guys-0828-T1.nii.gz", "IXI012-HH-1211-T1.nii.gz", "IXI013-HH-1212-T2.nii.gz", "notes.nii.gz"):
        (tmp_path / name).write_bytes(b"")
    s = v2.ixi_scans(tmp_path)
    assert s[["IXI_ID", "site"]].values.tolist() == [["IXI002", "Guys"], ["IXI012", "HH"]]


def test_fresh_pd_excludes_everyone_the_13_oct_test_selected(monkeypatch):
    import pie.imaging.labels as labels
    b = pd.DataFrame({"PATNO": range(1, 11), "IMAGEID": [f"I{i}" for i in range(1, 11)]})
    monkeypatch.setattr(v2, "baseline", lambda: b)
    monkeypatch.setattr(labels, "covariates", lambda ppmi: pd.DataFrame({"PATNO": range(1, 11), "COHORT": ["Parkinson's Disease"] * 8 + ["Healthy Control"] * 2}))
    got = v2.fresh_pd(pd.DataFrame({"PATNO": [1, 2, 3]}), "PPMI", n=10)
    assert set(got["PATNO"]) == {4, 5, 6, 7, 8}


def test_symmetry_gate_needs_the_left_right_ratio_in_range():
    ixi, pdm, rep = v2.synthetic()
    assert v2.analyse(ixi, pdm, rep, n_boot=20)["gates"]["symmetry"]
    lopsided = ixi.assign(ch4_p_lh=ixi["ch4_p_lh"] * 2)
    assert not v2.analyse(lopsided, pdm, rep, n_boot=20)["gates"]["symmetry"]


def test_convergence_is_gated_only_when_a_bar_is_set(monkeypatch):
    ixi, pdm, rep = v2.synthetic()
    assert "convergent" not in v2.analyse(ixi, pdm, rep, n_boot=20)["gates"]
    monkeypatch.setattr(v2, "GATE_RHO", 0.30)
    assert "convergent" in v2.analyse(ixi, pdm, rep, n_boot=20)["gates"]


def test_dry_run_uses_fake_data_only(tmp_path):
    res = v2.main(["analyse", "--synthetic", "--out", str(tmp_path), "--n-boot", "50"])
    saved = json.loads((tmp_path / "synthetic_ch4_v2.json").read_text())
    assert saved["config"] == "julich303-1mm" and set(res["gates"]) == {"symmetry", "age", "reliability"}
    assert saved["flow"]["ixi_selected"] == 280 and "sensitivity_iqr_60" in saved["descriptive"]


def test_large_scans_are_scheduled_separately(tmp_path):
    import nibabel as nib
    import numpy as np
    small, big = tmp_path / "s.nii.gz", tmp_path / "b.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.int8), np.eye(4)), small)
    nib.save(nib.Nifti1Image(np.zeros((400, 400, 300), np.int8), np.eye(4)), big)     # 4.8e7 voxels, like the 0.5 mm scans
    regular, large = v2.split_jobs([("a", "x", str(small)), ("b", "x", str(tmp_path / "missing.nii.gz")), ("c", "x", str(big))])
    assert [j[0] for j in regular] == ["a", "b"] and [j[0] for j in large] == ["c"]
