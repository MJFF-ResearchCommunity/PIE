"""FLAIR white-matter hyperintensities: the threshold method on a phantom, and the TrUE-Net path with FSL/TrUE-Net mocked."""

import sys
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging import flair, freesurfer, manifest


def test_wmh_threshold_recovers_lesions_and_splits_periventricular():
    shape = (40, 60, 60)
    aseg = np.zeros(shape, dtype=np.int32)
    aseg[5:35, 5:55, 5:55] = 2                 # white matter block
    aseg[15:25, 25:35, 25:35] = 4              # a lateral ventricle in the middle
    rng = np.random.default_rng(0)
    img = np.where(aseg > 0, 100.0, 0.0) + rng.normal(0, 3, shape)
    img[aseg == 4] = 20.0                      # CSF dark on FLAIR
    img[18:22, 36:40, 28:32] = 160.0           # periventricular lesion (touches the ventricle, 64 mm3)
    img[8:11, 45:50, 45:50] = 150.0            # deep lesion (75 mm3, ~20 mm from the ventricle)
    img[30, 50, 50] = 200.0                    # single bright voxel: below the minimum size, must be dropped
    les, out = flair.wmh(img.astype(np.float32), aseg, vox_mm=1.0)
    assert 120 <= out["wmh_mm3"] <= 150, out["wmh_mm3"]
    assert 55 <= out["wmh_pv_mm3"] <= 70 and 65 <= out["wmh_deep_mm3"] <= 80
    assert out["wmh_n_lesions"] == 2
    assert not les[30, 50, 50]
    assert abs(out["flair_wm_median"] - 100) < 2 and out["wmh_threshold"] > 105


# A 1 mm FastSurfer T1 grid and a 2D FLAIR slab (1 x 1 x 5 mm, 13 slices = 65 mm S-I) in the same scanner space:
# voxel i of either grid sits at world -32 + i * spacing on each axis.
T1_AFF = np.array([[1.0, 0, 0, -32], [0, 1, 0, -32], [0, 0, 1, -32], [0, 0, 0, 1]])
FL_AFF = np.array([[1.0, 0, 0, -32], [0, 1, 0, -32], [0, 0, 5, -32], [0, 0, 0, 1]])


def _subject(tmp_path):
    aseg = np.zeros((64, 64, 64), np.int32)
    aseg[8:56, 8:56, 8:56] = 2
    aseg[28:36, 28:36, 28:36] = 4                   # lateral ventricle, world -4..3 mm on each axis
    mri = tmp_path / "fs" / "I000001" / "mri"
    mri.mkdir(parents=True)
    nib.save(nib.MGHImage(np.where(aseg > 0, 100, 0).astype(np.uint8), T1_AFF), mri / "orig.mgz")
    nib.save(nib.MGHImage((aseg > 0).astype(np.uint8), T1_AFF), mri / "mask.mgz")
    nib.save(nib.MGHImage(aseg, T1_AFF), mri / "aparc.DKTatlas+aseg.deep.mgz")
    fl = np.zeros((64, 64, 13), np.float32)
    fl[4:60, 4:60, :] = 100 + np.random.default_rng(0).normal(0, 3, (56, 56, 13))
    nii = tmp_path / "flair.nii.gz"
    nib.save(nib.Nifti1Image(fl, FL_AFF), nii)
    return mri.parent, nii


def _fake_fsl(tmp_path, monkeypatch):
    fsl = tmp_path / "fsl"
    (fsl / "bin").mkdir(parents=True)
    (fsl / "bin" / "fsl_anat").touch()
    (fsl / "etc").mkdir()
    (fsl / "etc" / "fslversion").write_text("6.0.7.23\n")
    (tmp_path / "models" / "mwsc").mkdir(parents=True)
    (tmp_path / "venv_bin").mkdir()
    (tmp_path / "venv_bin" / "prepare_truenet_data").touch()
    monkeypatch.setattr(flair, "BIN", tmp_path / "venv_bin")
    monkeypatch.setattr(flair, "FSLDIR", str(fsl))
    monkeypatch.setattr(flair, "TRUENET_MODELS", tmp_path / "models")


def test_truenet_path_masks_thresholds_and_measures_on_the_flair_grid(tmp_path, monkeypatch):
    import SimpleITK as sitk

    fs_dir, nii = _subject(tmp_path)
    _fake_fsl(tmp_path, monkeypatch)
    prob = np.zeros((64, 64, 13), np.float32)
    prob[37:40, 30:34, 6] = 0.9      # 2-4 mm from the ventricle: periventricular, 12 voxels x 5 mm3
    prob[50:53, 30:34, 6] = 0.8      # 15-17 mm away: deep, 60 mm3
    prob[1:3, 30:34, 6] = 0.95       # outside the WM mask (cortex): removed
    prob[20:23, 20:23, 3] = 0.4      # below the 0.5 threshold
    wm_mask = np.zeros_like(prob)
    wm_mask[6:58, 6:58, :] = 1
    calls = []

    def fake_run(cmd, log, cwd, env=None, timeout=3600):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        assert env["FSLDIR"] == flair.FSLDIR and env["TRUENET_PRETRAINED_MODEL_PATH"] == str(tmp_path / "models")
        if cmd[0].endswith("prepare_truenet_data"):
            src = nib.load(Path(cwd) / "flair.nii.gz")
            for name, data in (("sub_FLAIR", src.get_fdata()), ("sub_T1", src.get_fdata()), ("sub_WMmask", wm_mask)):
                nib.save(nib.Nifti1Image(data.astype(np.float32), src.affine), Path(cwd) / f"{name}.nii.gz")
        else:
            nib.save(nib.Nifti1Image(prob, FL_AFF), Path(cwd) / "pred" / "Predicted_probmap_truenet_sub.nii.gz")

    monkeypatch.setattr(freesurfer, "run", fake_run)
    monkeypatch.setattr(flair, "convert", lambda zip_path, prefix, out_dir: [str(nii)])
    monkeypatch.setattr(flair, "n4", lambda img, shrink=2: img)
    monkeypatch.setattr(flair, "register_flair_to_t1", lambda f, t1, m: (sitk.Transform(), -0.5))
    rows = [{"zip": "z", "prefix": "p/", "desc": "Ax T2 FLAIR", "flair_3d": False, "n_files": 13, "date": "2000-01-01"}]
    row = flair.process_subject(1, rows, str(fs_dir), str(tmp_path / "work"))

    out = tmp_path / "work" / "1" / "truenet"
    assert calls[0] == [str(flair.BIN / "prepare_truenet_data"), f"--FLAIR={out}/flair", f"--T1={out}/t1", f"--outname={out}/sub", "--nodistmaps"]
    assert calls[1] == [str(flair.BIN / "truenet"), "evaluate", "-i", str(out), "-m", "mwsc", "-o", f"{out}/pred"]
    assert not out.exists()                                         # intermediates removed without --keep-nifti
    assert row["wmh_mm3"] == 120.0 and row["wmh_pv_mm3"] == 60.0 and row["wmh_deep_mm3"] == 60.0
    assert row["wmh_n_lesions"] == 2 and row["wmh_log_mm3"] == pytest.approx(np.log1p(120.0))
    assert row["wmh_frac_wm"] == pytest.approx(120.0 / row["wm_mm3"]) and row["wm_mm3"] > 0 and row["flair_wm_mad"] > 0
    assert (row["wmh_method"], row["truenet_model"], row["wmh_prob_threshold"]) == ("truenet", "mwsc", 0.5)
    assert row["truenet_version"] == version("truenet") and row["fsl_version"] == "6.0.7.23"
    assert row["flair_si_mm"] == 65.0 and row["flair_3d"] is False
    assert not manifest.QC["flair"](pd.DataFrame([row])).iloc[0]   # registration fine, but the slab is too short


def test_threshold_method_is_recorded(tmp_path, monkeypatch):
    import SimpleITK as sitk

    fs_dir, nii = _subject(tmp_path)
    monkeypatch.setattr(flair, "convert", lambda zip_path, prefix, out_dir: [str(nii)])
    monkeypatch.setattr(flair, "n4", lambda img, shrink=2: img)
    monkeypatch.setattr(flair, "register_flair_to_t1", lambda f, t1, m: (sitk.Transform(), -0.5))
    rows = [{"zip": "z", "prefix": "p/", "desc": "Ax T2 FLAIR", "flair_3d": False, "n_files": 13, "date": "2000-01-01"}]
    row = flair.process_subject(1, rows, str(fs_dir), str(tmp_path / "work"), method="threshold")
    assert row["wmh_method"] == "threshold" and "truenet_model" not in row and row["flair_si_mm"] == 65.0


def test_flair_qc_flags_short_si_coverage_and_passes_tables_without_it():
    d = pd.DataFrame({"reg_flair_t1_mi": [-0.5] * 3, "wm_mm3": [300000.0] * 3, "flair_wm_mad": [3.0] * 3,
                      "flair_si_mm": [120.0, 150.0, np.nan]})
    assert manifest.QC["flair"](d).tolist() == [False, True, True]
    assert manifest.QC["flair"](d.drop(columns="flair_si_mm")).all()


def test_si_coverage_uses_the_axis_nearest_superior_inferior():
    assert flair.si_coverage_mm(nib.Nifti1Image(np.zeros((64, 64, 13), np.float32), FL_AFF)) == 65.0
    sag = np.array([[0, 0, 1.2, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1]])   # 3D sagittal: S-I is the second voxel axis
    assert flair.si_coverage_mm(nib.Nifti1Image(np.zeros((256, 240, 160), np.float32), sag)) == 240.0


def test_resuming_refuses_to_mix_methods(tmp_path):
    pd.DataFrame({"patno": [1], "error": [""], "wmh_mm3": [1.0]}).to_csv(tmp_path / "flair_features.csv", index=False)
    with pytest.raises(SystemExit, match="threshold rows"):       # rows from before wmh_method are threshold rows
        flair.main(["--zips", "z", "--sessions", "s", "--fastsurfer-dir", "f", "--work-dir", str(tmp_path)])


def _truenet_ready():
    try:
        flair.truenet_env()
    except RuntimeError:
        return False
    return (flair.TRUENET_MODELS / flair.TRUENET_MODEL / "Truenet_MWSC_FLAIR_T1_axial.pth").exists()


@pytest.mark.skipif(not _truenet_ready(), reason="FSL or the TrUE-Net weights are absent")
def test_truenet_evaluate_smoke_on_cpu(tmp_path, monkeypatch):
    """The real ``truenet evaluate`` with the installed weights (prepare_truenet_data faked: fsl_anat takes ~20 min)."""
    real = freesurfer.run

    def prepare_faked(cmd, log, cwd, env=None, timeout=3600):
        if not str(cmd[0]).endswith("prepare_truenet_data"):
            return real(cmd, log, cwd, env=env, timeout=timeout)
        src = nib.load(Path(cwd) / "flair.nii.gz")
        for name in ("sub_FLAIR", "sub_T1", "sub_WMmask"):
            nib.save(src, Path(cwd) / f"{name}.nii.gz")

    monkeypatch.setattr(freesurfer, "run", prepare_faked)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    x, y, z = np.ogrid[:96, :112, :40]
    brain = ((x - 48) / 40) ** 2 + ((y - 56) / 48) ** 2 + ((z - 20) / 18) ** 2 < 1
    fl = np.where(brain, 100.0, 0.0).astype(np.float32)
    nib.save(nib.Nifti1Image(fl, np.diag([1.0, 1.0, 3.0, 1.0])), tmp_path / "f.nii.gz")
    prob = flair.run_truenet(tmp_path / "f.nii.gz", nib.Nifti1Image(fl, np.eye(4)), tmp_path / "tn")
    p = np.asanyarray(prob.dataobj)
    assert p.shape == fl.shape and 0 <= p.min() and p.max() <= 1
