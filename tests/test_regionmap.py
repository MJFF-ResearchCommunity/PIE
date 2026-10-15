import nibabel as nib
import numpy as np
import pandas as pd

from pie.imaging import regionmap

LUT = pd.DataFrame({"ID": [0, 12, 51, 1024], "LabelName": ["Background", "Left-Putamen", "Right-Putamen", "ctx-lh-precentral"]})


def _atlas():
    a = np.zeros((4, 4, 4), np.int16)
    a[0, :, :], a[1, :, :], a[2, :, :] = 12, 51, 1024
    return nib.Nifti1Image(a, np.eye(4))


def test_feature_names_match_atlas_labels_across_prefixes():
    got, unmatched = regionmap.match_features(
        ["vol_Left_Putamen", "vol_ctx_lh_precentral", "cth_lh_precentral", "eTIV", "vol_Left_WM_hypointensities"], LUT)
    assert got == {"vol_Left_Putamen": 12, "vol_ctx_lh_precentral": 1024, "cth_lh_precentral": 1024}
    assert unmatched == ["eTIV", "vol_Left_WM_hypointensities"]


def test_paint_fills_regions_and_leaves_the_rest_zero():
    out = np.asarray(regionmap.paint(_atlas(), {12: -0.4, 1024: 0.7}).dataobj)
    assert out.dtype == np.float32 and out[0].max() == np.float32(-0.4) and out[2].min() == np.float32(0.7)
    assert not out[1].any() and not out[3].any()


def test_region_map_thresholds_on_p_and_records_the_sidecar(tmp_path):
    p = tmp_path / "atlas.nii.gz"
    nib.save(_atlas(), p)
    t = pd.DataFrame({"feature": ["vol_Left_Putamen", "vol_Right_Putamen", "eTIV"], "d": [-0.5, 0.2, 0.1], "p": [0.001, 0.4, 0.01]})
    img, side = regionmap.region_map(t, p, value="d", p="p", alpha=0.05, statistic="Cohen's d (PD - HC)", units="SD")
    out = np.asarray(img.dataobj)
    assert out[0].max() == np.float32(-0.5) and not out[1].any()           # right putamen fails p < 0.05
    assert side["n_matched"] == 2 and side["unmatched"] == ["eTIV"] and side["n_shown"] == 1
    assert side["vmax"] == 0.5 and side["threshold"] == {"p_column": "p", "alpha": 0.05}
    assert len(side["source_sha256"]) == 64


def test_all_subthreshold_map_is_empty_but_valid(tmp_path):
    p = tmp_path / "atlas.nii.gz"
    nib.save(_atlas(), p)
    t = pd.DataFrame({"feature": ["vol_Left_Putamen"], "d": [0.3], "p": [0.9]})
    img, side = regionmap.region_map(t, p, value="d", p="p", alpha=0.05)
    assert not np.asarray(img.dataobj).any() and side["n_shown"] == 0 and side["vmax"] == 0.0
