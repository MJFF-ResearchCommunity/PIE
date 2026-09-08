"""Manifest / feature assembly on a synthetic derived directory."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging.manifest import assemble_features, build_manifest, feature_blocks


def test_manifest_and_assembly(tmp_path):
    d = tmp_path
    pd.DataFrame({"PATNO": [1, 2, 3], "IMAGEID": ["I1", "I2", "I3"], "SCAN_DATE": ["2022-01-01"] * 3, "vol_Left_Putamen": [5000.0, 5100.0, 4900.0], "MaskVol": [1.4e6] * 3}).to_csv(d / "fastsurfer_idps.csv", index=False)
    (d / "dwi").mkdir()
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "motion_mm_max": [1.0, 9.0], "n_sn_l": [40, 40], "n_sn_r": [40, 40], "fa_wm_median": [0.4, 0.4],
                  "manufacturer": ["Siemens", "GE"], "shells": ["1000", "700 1000 2000"], "fw_method": ["singleshell_prior", "multishell_nls"],
                  "sn_posterior_mean_fw": [0.3, 0.4], "putamen_mean_fa": [0.2, 0.2], "n_putamen_l": [500, 500]}).to_csv(d / "dwi" / "dwi_features.csv", index=False)
    pd.DataFrame({"zip": ["z"] * 2, "prefix": ["p"] * 2, "patno": [1, 2], "desc": ["DTI_gated"] * 2, "date": ["2022-01-03", "2022-02-01"], "image_id": ["a", "b"], "n_files": [65, 65], "selected": [True, True]}).to_csv(d / "dwi" / "dwi_index.csv", index=False)
    man = build_manifest(d)
    assert len(man) == 3 and man.loc[man.PATNO == 1, "dwi_qc_pass"].item() and not man.loc[man.PATNO == 2, "dwi_qc_pass"].item()
    assert man.loc[man.PATNO == 1, "dwi_days_from_t1"].item() == 2 and man["dwi_batch"].nunique() == 2
    f = assemble_features(d)
    assert f.loc[f.PATNO == 1, "dwi_sn_posterior_mean_fw"].item() == 0.3
    assert np.isnan(f.loc[f.PATNO == 2, "dwi_sn_posterior_mean_fw"].item())     # QC-failed values blanked
    assert np.isnan(f.loc[f.PATNO == 3, "dwi_sn_posterior_mean_fw"].item())     # no DWI at all
    assert "dwi_n_putamen_l" not in f.columns and "vol_Left_Putamen" in f.columns
    assert feature_blocks(f.columns)["dwi"] == ["dwi_sn_posterior_mean_fw", "dwi_putamen_mean_fa"]
