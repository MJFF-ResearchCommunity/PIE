"""The pre-registered FLAIR WMH analysis on tiny arrays and fully synthetic tables (no real value, label or identifier)."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("analyse_flair_wmh", Path(__file__).resolve().parent.parent / "scripts" / "analyse_flair_wmh.py")
wmh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wmh)


def test_mask_metrics_follow_the_challenge_definitions():
    ref = np.zeros((10, 10, 10), int)
    ref[1:3, 1:3, 1:3] = 1                  # lesion A, 8 voxels
    ref[6:8, 6:8, 6:8] = 1                  # lesion B, 8 voxels
    ref[4, 4, 4] = 2                        # other pathology
    same = wmh.mask_metrics(ref, ref == 1, vox_mm3=2.0)
    assert same["dice"] == 1 and same["lesion_recall"] == 1 and same["lesion_f1"] == 1 and same["avd_pct"] == 0
    assert same["vol_ref_mm3"] == same["vol_pred_mm3"] == 32
    disjoint = np.zeros_like(ref, bool)
    disjoint[0, 9, 9] = True
    d = wmh.mask_metrics(ref, disjoint)
    assert d["dice"] == 0 and d["lesion_recall"] == 0 and d["lesion_f1"] == 0
    half = ref == 1
    half[6:8, 6:8, 6:8] = False             # B missed
    half[4, 4, 4] = True                    # a detection on other pathology is removed, not a false positive
    h = wmh.mask_metrics(ref, half)
    assert h["lesion_recall"] == 0.5 and abs(h["lesion_f1"] - 2 / 3) < 1e-9 and abs(h["dice"] - 2 / 3) < 1e-9 and h["avd_pct"] == 50
    diag = np.zeros_like(ref, bool)
    diag[1, 1, 1] = diag[2, 2, 2] = True    # corner-touching voxels are one lesion (26-connectivity)
    assert wmh.mask_metrics(diag.astype(int), diag)["lesion_f1"] == 1


def test_g1_passes_good_segmentations_and_fails_poor_ones():
    good = wmh.g1(wmh.synthetic_g1(), n_boot=50)
    assert good["pass"] and good["n"] == 110 and good["n_failed"] == 1 and good["volume_ml"]["n"] == 109
    assert set(good["by_scanner"]) == set(wmh.SCANNERS) and good["unseen_scanners"]["n"] == 20
    assert not wmh.g1(wmh.synthetic_g1(dice_mean=0.60), n_boot=50)["pass"]          # Dice below 0.70
    assert not wmh.g1(wmh.synthetic_g1(vol_sd=1.5), n_boot=50)["pass"]              # volumes disagree


def test_g2_recovers_age_and_vascular_associations_and_null_fails():
    res = wmh.analyse_g2(*wmh.synthetic_g2(), n_boot=50)
    assert res["pass"] and res["g2a_age"]["r"] > 0.35 and res["g2b_vascular"]["ci_low"] > 0
    assert res["secondary"]["vascular_items"]["hypertension"]["r"] > 0 and res["secondary"]["smoking_ever"]["n"] < 500
    null = wmh.analyse_g2(*wmh.synthetic_g2(age_slope=0.0, vrs_slope=0.0), n_boot=50)
    assert not null["g2a_age"]["pass"] and not null["g2b_vascular"]["pass"] and not null["pass"]
    weak_age = wmh.analyse_g2(*wmh.synthetic_g2(age_slope=0.02), n_boot=50)       # positive but under 0.35
    assert 0 < weak_age["g2a_age"]["r"] < 0.35 and not weak_age["g2a_age"]["pass"]
