"""Manifest / feature assembly on a synthetic derived directory."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging.manifest import assemble_features, build_manifest, feature_blocks


def test_manifest_and_assembly(tmp_path):
    d = tmp_path
    pd.DataFrame({"PATNO": [1, 2, 3], "IMAGEID": ["I1", "I2", "I3"], "SCAN_DATE": ["2022-01-01"] * 3, "vol_Left_Putamen": [5000.0, 5100.0, 0.0], "MaskVol": [1.4e6] * 3}).to_csv(d / "fastsurfer_idps.csv", index=False)
    (d / "dwi").mkdir()
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "motion_mm_max": [1.0, 9.0], "motion_mm_mean": [0.5, 4.0], "sn_brain_mask_fraction": [1., 1.], "sn_physical_fraction": [.95, .95], "sn_posterior_l_fa": [.5, .5], "sn_posterior_r_fa": [.5, .5], "n_sn_l": [40, 40], "n_sn_r": [40, 40], "fa_wm_median": [0.4, 0.4],
                  "manufacturer": ["Siemens", "GE"], "shells": ["1000", "700 1000 2000"], "fw_method": ["singleshell_prior", "multishell_nls"],
                  "sn_posterior_mean_fw": [0.3, 0.4], "putamen_mean_fa": [0.2, 0.2], "n_putamen_l": [500, 500]}).to_csv(d / "dwi" / "dwi_features.csv", index=False)
    pd.DataFrame({"zip": ["z"] * 2, "prefix": ["p"] * 2, "patno": [1, 2], "desc": ["DTI_gated"] * 2, "date": ["2022-01-03", "2022-02-01"], "image_id": ["a", "b"], "n_files": [65, 65], "selected": [True, True]}).to_csv(d / "dwi" / "dwi_index.csv", index=False)
    man = build_manifest(d)
    assert len(man) == 3 and man.loc[man.PATNO == 1, "dwi_qc_pass"].item() and not man.loc[man.PATNO == 2, "dwi_qc_pass"].item()
    assert man.loc[man.PATNO == 1, "dwi_days_from_t1"].item() == 2 and man["dwi_batch"].nunique() == 2
    f = assemble_features(d, single_shell_fw=True)     # patient 1 is single-shell; the quarantine has its own test
    assert f.loc[f.PATNO == 1, "dwi_sn_posterior_mean_fw"].item() == 0.3
    assert np.isnan(f.loc[f.PATNO == 2, "dwi_sn_posterior_mean_fw"].item())     # QC-failed values blanked
    assert np.isnan(f.loc[f.PATNO == 3, "dwi_sn_posterior_mean_fw"].item())     # no DWI at all
    assert "dwi_n_putamen_l" not in f.columns and "vol_Left_Putamen" in f.columns
    assert f.loc[f.PATNO == 1, "t1_qc_pass"].item() and not f.loc[f.PATNO == 3, "t1_qc_pass"].item()   # empty label = failed segmentation
    assert np.isnan(f.loc[f.PATNO == 3, "MaskVol"].item()) and f.loc[f.PATNO == 3, "vol_Left_Putamen"].isna().item()
    assert set(feature_blocks(f.columns)["dwi"]) == {"dwi_sn_posterior_mean_fw", "dwi_putamen_mean_fa", "dwi_sn_posterior_l_fa", "dwi_sn_posterior_r_fa"}


def test_study_nigral_dwi_is_assembled_under_its_own_qc_and_failed_t1_keeps_identifiers(tmp_path):
    d = tmp_path
    pd.DataFrame({"PATNO": [1, 2, 3], "IMAGEID": ["I1", "I2", "I3"], "SCAN_DATE": ["2022-01-01"] * 3, "EVENT_ID": ["BL"] * 3,
                  "Manufacturer": ["GE"] * 3, "vol_Left_Putamen": [5000.0, 5100.0, 0.0]}).to_csv(d / "fastsurfer_idps.csv", index=False)
    (d / "dwi").mkdir()
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "motion_mm_mean": [0.5, 4.0], "sn_brain_mask_fraction": [1., 1.],
                  "sn_physical_fraction": [.95, .95], "sn_posterior_l_fa": [.5, .5], "sn_posterior_r_fa": [.5, .5], "n_sn_l": [40, 40],
                  "n_sn_r": [40, 40], "fa_wm_median": [0.4, 0.4], "manufacturer": ["GE", "GE"], "shells": ["1000"] * 2,
                  "fw_method": ["singleshell_prior"] * 2, "dwis_md_sn_posterior_mean": [8e-4, 9e-4], "dwis_n_sn": [70, 70],
                  "dwis_qc_pass": [False, True]}).to_csv(d / "dwi" / "dwi_features.csv", index=False)
    f = assemble_features(d)
    md = f.set_index("PATNO")["dwis_md_sn_posterior_mean"]
    assert np.isnan(md[1]) and md[2] == 9e-4 and np.isnan(md[3])      # own QC, not the generic ROI path's
    assert "dwis_n_sn" not in f.columns and "dwis_md_sn_posterior_mean" in feature_blocks(f.columns)["dwi"]
    assert not f["dwi_nigral_validated"].any()
    row3 = f.set_index("PATNO").loc[3]
    assert not row3["t1_qc_pass"] and np.isnan(row3["vol_Left_Putamen"]) and row3["EVENT_ID"] == "BL" and row3["Manufacturer"] == "GE"


def test_dat_qc_rejects_failed_and_anticorrelated_registrations():
    from pie.imaging.manifest import QC

    d = pd.DataFrame({"reg_metric": [-0.8, np.inf, 0.6, -0.2], "n_label_voxels": [500] * 4})
    assert QC["dat"](d).tolist() == [True, False, False, False]


def test_core_lab_sbr_is_assembled_for_the_scan_pie_reconstructed_xing_first(tmp_path):
    d, ppmi = tmp_path / "derived", tmp_path / "ppmi"
    (d / "datscan_full").mkdir(parents=True)
    (ppmi / "Imaging").mkdir(parents=True)
    pd.DataFrame({"PATNO": [1, 2, 3], "IMAGEID": ["I1", "I2", "I3"], "SCAN_DATE": ["2022-01-01"] * 3,
                  "vol_Left_Putamen": [5000.0] * 3}).to_csv(d / "fastsurfer_idps.csv", index=False)
    pd.DataFrame({"patno": [1, 2, 3], "image_id": ["S1", "S2", "S3"], "reg_metric": [-0.8, -0.8, -0.1], "n_label_voxels": [500] * 3,
                  "hdr_manufacturer": ["GE"] * 3, "hdr_model": ["Infinia"] * 3, "sbr_putamen_l": [1.0, 2.0, 3.0]}).to_csv(d / "datscan_full" / "datscan_sbr.csv", index=False)
    pd.DataFrame({"image_id": ["S1", "S2", "S3"], "member": [f"PPMI/{p}/DaTSCAN/2022-0{p}-15_10_00_00.0/S{p}/x.dcm" for p in (1, 2, 3)]}).to_csv(d / "spect_index.csv", index=False)
    sbr = lambda patnos, months, v: {"PATNO": patnos, "DATSCAN_DATE": months, "DATSCAN_ANALYZED": ["Yes"] * len(patnos),
                                     **{f"DATSCAN_{r}_{s}": [v] * len(patnos) for r in ("CAUDATE", "PUTAMEN") for s in "RL"}}
    x = sbr([1], ["01/2022"], 1.5)
    x.update({f"{r}_{s}_REF_CWM": x[f"DATSCAN_{r}_{s}"] for r in ("CAUDATE", "PUTAMEN") for s in "RL"})
    pd.DataFrame(x).to_csv(ppmi / "Imaging" / "Xing_Core_Lab_-_Quant_SBR_08Sep2026.csv", index=False)
    pd.DataFrame(sbr([1, 2, 3], ["01/2022", "02/2022", "07/2022"], 2.5)).to_csv(ppmi / "Imaging" / "DaTScan_SBR_Analysis_18Mar2025.csv", index=False)
    for stem in ("Xing_Core_Lab_-_Visual_Read_08Sep2026", "DaTScan_Visual_Interpretation_Results_18Mar2025"):
        pd.DataFrame({"PATNO": [1], "DATSCAN_DATE": ["01/2022"], "DATSCAN_VISINTRP": ["Positive"]}).to_csv(ppmi / "Imaging" / f"{stem}.csv", index=False)
    f = assemble_features(d, ppmi_dir=ppmi).set_index("PATNO")
    assert f.loc[1, "dat_core_source"] == "xing_cwm" and f.loc[1, "dat_core_putamen_l"] == 1.5       # Xing wins
    assert f.loc[2, "dat_core_source"] == "invicro_occipital" and f.loc[2, "dat_core_putamen_l"] == 2.5
    assert pd.isna(f.loc[3, "dat_core_putamen_l"])          # the core lab's scan is from another month
    assert np.isnan(f.loc[3, "dat_sbr_putamen_l"])          # PIE's failed registration is blanked, the core lab never is
    assert f.loc[1, "dat_core_pc_ratio_l"] == 1.0
    assert "dat_core_putamen_l" in feature_blocks(f.columns)["dat_core"] and "dat_core_source" not in feature_blocks(f.columns)["dat_core"]


def test_fw_endpoint_is_assembled_under_its_own_qc(tmp_path):
    d = tmp_path
    pd.DataFrame({"PATNO": [1, 2], "IMAGEID": ["I1", "I2"], "SCAN_DATE": ["2022-01-01"] * 2, "vol_Left_Putamen": [5000.0] * 2}).to_csv(d / "fastsurfer_idps.csv", index=False)
    (d / "dwi").mkdir()
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "motion_mm_mean": [0.5] * 2, "sn_brain_mask_fraction": [1.] * 2, "sn_physical_fraction": [.95] * 2,
                  "sn_posterior_l_fa": [.5] * 2, "sn_posterior_r_fa": [.5] * 2, "n_sn_l": [40] * 2, "n_sn_r": [40] * 2, "fa_wm_median": [0.4] * 2,
                  "manufacturer": ["Siemens"] * 2, "shells": ["1000 2000"] * 2, "fw_method": ["multishell_nls"] * 2,
                  "dwisf_fw_sn_posterior_mean": [0.2, 0.3], "dwisf_n_sn": [80, 80], "dwisf_qc_pass": [True, False]}).to_csv(d / "dwi" / "dwi_features.csv", index=False)
    f = assemble_features(d).set_index("PATNO")
    assert f.loc[1, "dwisf_fw_sn_posterior_mean"] == 0.2 and np.isnan(f.loc[2, "dwisf_fw_sn_posterior_mean"])
    assert "dwisf_n_sn" not in f.columns and "dwisf_fw_sn_posterior_mean" in feature_blocks(f.columns)["dwi"]


def test_t1_outliers_are_flagged_small_brains_and_visual_fails_fail_qc(tmp_path):
    from pie.imaging.manifest import t1_outliers
    rng = np.random.default_rng(0)
    n = 30
    vols = {c: rng.normal(5000, 200, n) for c in ("vol_Left_Putamen", "vol_Right_Putamen", "vol_Left_Caudate", "vol_Right_Caudate",
                                                  "vol_Left_Thalamus", "vol_Right_Thalamus")}
    vols["vol_Left_Putamen"][3] = 1500                                  # a gross segmentation error
    d = pd.DataFrame({"PATNO": range(n), "IMAGEID": [f"I{i}" for i in range(n)], "SCAN_DATE": ["2022-01-01"] * n,
                      "Manufacturer": ["Siemens"] * n, "MagneticFieldStrength": [3] * n, "BrainSegVol": [1.1e6] * (n - 1) + [6e5], **vols})
    o = t1_outliers(d)
    assert o["t1_outlier"].tolist() == [i == 3 for i in range(n)]
    d.to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    pd.DataFrame({"IMAGEID": ["I5"], "rating": ["fail"]}).to_csv(tmp_path / "t1_visual_qc.csv", index=False)
    f = assemble_features(tmp_path).set_index("PATNO")
    assert not f.loc[29, "t1_qc_pass"] and not f.loc[5, "t1_qc_pass"] and f.loc[3, "t1_qc_pass"]   # outliers stay, for review
    assert f.loc[3, "t1_outlier"] and np.isnan(f.loc[5, "vol_Left_Putamen"]) and f.loc[5, "t1_visual_qc"] == "fail"
