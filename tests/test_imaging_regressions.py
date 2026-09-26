"""Regression tests for imaging-layer bugs found during the documentation audit (September 2026).

Synthetic data only; no FastSurfer, FSL, MRtrix, ANTs or GPU. Each test failed before its fix.
"""
import argparse
import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pie.imaging
from pie.imaging import (cnn, datscan, dwi, dwi_refine, embed, fba, features, flair, labels, manifest, nm, nm_template, qc,
                         run)


# ------------------------------------------------------------------------------------------ DaTscan table header
def test_datscan_table_keeps_all_columns_when_the_first_series_fails(tmp_path, monkeypatch):
    def fake(zip_path, member, out_dir, fs_dir=None, flip_lr=False, attenuation=False):
        image_id = member.split("/")[4]
        if image_id == "I000001":
            raise ValueError("synthetic failure")
        return {"image_id": image_id, "patno": int(member.split("/")[1]), "series_desc": "SPECT",
                "hdr_model": "synthetic", "sbr_putamen_l": 2.0, "reg_metric": -0.6}

    monkeypatch.setattr(datscan, "process_series", fake)
    pd.DataFrame({"zip": "none.zip", "patno": [1, 2, 3], "image_id": ["I000001", "I000002", "I000003"],
                  "member": [f"PPMI/{p}/SPECT/2000-01-01_00_00_00.0/I00000{p}/f.dcm" for p in (1, 2, 3)],
                  "kind": "TOMO", "frames": [300, 200, 100]}).to_csv(tmp_path / "idx.csv", index=False)
    pd.DataFrame({"patno": [9], "image_id": ["I000009"], "session_date": ["2000-01-01"]}).to_csv(tmp_path / "s.csv", index=False)
    (tmp_path / "fs").mkdir()
    datscan.main(["--index", str(tmp_path / "idx.csv"), "--sessions", str(tmp_path / "s.csv"), "--fastsurfer-dir",
                  str(tmp_path / "fs"), "--out-dir", str(tmp_path / "out"), "--workers", "1"])
    out = pd.read_csv(tmp_path / "out" / "datscan_sbr.csv", dtype={"image_id": str})
    ok = out[out["error"].fillna("") == ""]
    assert len(out) == 3 and len(ok) == 2
    assert ok["hdr_model"].eq("synthetic").all() and ok["reg_metric"].eq(-0.6).all()


def _two_finished_t1s(tmp_path):
    fs = tmp_path / "fs"
    for iid in ("I000001", "I000002"):
        (fs / iid / "stats").mkdir(parents=True)
        (fs / iid / "stats" / "aseg+DKT.stats").touch()
        (fs / iid / "mri").mkdir()
        (fs / iid / "mri" / "mask.mgz").touch()
    pd.DataFrame({"patno": [1, 1], "image_id": ["I000001", "I000002"],
                  "session_date": ["2000-01-01", "2001-01-01"]}).to_csv(tmp_path / "s.csv", index=False)
    return fs


def test_datscan_quantifies_against_the_earliest_finished_t1(tmp_path, monkeypatch):
    monkeypatch.setattr(datscan, "process_series", lambda zip_path, member, out_dir, fs_dir=None, flip_lr=False, attenuation=False:
                        {"image_id": "I000011", "patno": 1, "series_desc": "SPECT", "fs": Path(fs_dir).name})
    pd.DataFrame({"zip": ["none.zip"], "patno": [1], "image_id": ["I000011"], "kind": ["TOMO"], "frames": [120],
                  "member": ["PPMI/1/SPECT/2000-01-01_00_00_00.0/I000011/f.dcm"]}).to_csv(tmp_path / "idx.csv", index=False)
    fs = _two_finished_t1s(tmp_path)
    datscan.main(["--index", str(tmp_path / "idx.csv"), "--sessions", str(tmp_path / "s.csv"), "--fastsurfer-dir",
                  str(fs), "--out-dir", str(tmp_path / "out"), "--workers", "1"])
    assert pd.read_csv(tmp_path / "out" / "datscan_sbr.csv")["fs"].tolist() == ["I000001"]


def test_dwi_refine_uses_the_earliest_finished_t1(tmp_path, monkeypatch):
    monkeypatch.setattr(dwi_refine, "refine_subject", lambda subj_dir, fs_dir: {"fs": Path(fs_dir).name})
    work = tmp_path / "dwi"
    (work / "1").mkdir(parents=True)
    (work / "1" / "fw.nii.gz").touch()
    fs = _two_finished_t1s(tmp_path)
    dwi_refine.main(["--work-dir", str(work), "--sessions", str(tmp_path / "s.csv"), "--fastsurfer-dir", str(fs), "--workers", "1"])
    assert pd.read_csv(work / "dwi_features_syn.csv")["fs"].tolist() == ["I000001"]


# ------------------------------------------------------------------------------------------ manifest
def _idps(d):
    pd.DataFrame({"PATNO": [1, 2], "IMAGEID": ["I000001", "I000002"], "SCAN_DATE": ["2000-01-01"] * 2,
                  "vol_Left_Putamen": [5000.0] * 2, "vol_Right_Putamen": [4900.0] * 2}).to_csv(d / "fastsurfer_idps.csv", index=False)


def test_manifest_dates_come_from_the_selected_series_for_nm_and_flair(tmp_path):
    _idps(tmp_path)
    feats = {"nm": {"n_sn_l": 40, "n_sn_r": 40, "sn_slab_coverage": 0.9, "repeat_motion_mm_max": 0.5, "nm_ref_l_sd": 5.0,
                    "nm_ref_l_mean": 100.0, "nm_ref_r_sd": 5.0, "nm_ref_r_mean": 100.0, "voxel_mm": "0.5x0.5x1.5"},
             "flair": {"reg_flair_t1_mi": -0.5, "wm_mm3": 300000.0, "flair_wm_mad": 3.0, "flair_3d": True}}
    for mod, cols in feats.items():
        (tmp_path / mod).mkdir()
        pd.DataFrame({"patno": [1], "error": [""], "manufacturer": ["Siemens"], **{k: [v] for k, v in cols.items()}})\
            .to_csv(tmp_path / mod / f"{mod}_features.csv", index=False)
        # the unselected series has more files on another date and must not decide the session date
        pd.DataFrame({"zip": "z", "prefix": ["a/", "b/"], "patno": [1, 1], "desc": ["x", "y"], "date": ["2000-01-10", "2000-02-10"],
                      "image_id": ["I000011", "I000012"], "n_files": [16, 200], "selected": [True, False]})\
            .to_csv(tmp_path / mod / f"{mod}_index.csv", index=False)
    row = manifest.build_manifest(tmp_path).set_index("PATNO").loc[1]
    assert str(row["nm_date"])[:10] == "2000-01-10" and str(row["flair_date"])[:10] == "2000-01-10"


def test_flair_rows_record_acquisition_date_and_t1(tmp_path, monkeypatch):
    import SimpleITK as sitk

    arr = np.zeros((20, 20, 20), np.float32)
    arr[5:15, 5:15, 5:15] = 100.0
    nii = tmp_path / "flair.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), nii)
    mri = tmp_path / "fs" / "I000001" / "mri"
    mri.mkdir(parents=True)
    nib.save(nib.MGHImage(arr, np.eye(4)), mri / "orig.mgz")
    nib.save(nib.MGHImage((arr > 0).astype(np.uint8), np.eye(4)), mri / "mask.mgz")
    nib.save(nib.MGHImage(np.where(arr > 0, 2, 0).astype(np.int32), np.eye(4)), mri / "aparc.DKTatlas+aseg.deep.mgz")
    monkeypatch.setattr(flair, "convert", lambda zip_path, prefix, out_dir: [str(nii)])
    monkeypatch.setattr(flair, "n4", lambda img, shrink=2: img)
    monkeypatch.setattr(flair, "register_flair_to_t1", lambda f, t1, m: (sitk.Transform(), -0.5))
    monkeypatch.setattr(flair, "wmh", lambda a, b, vox_mm=1.0: (np.zeros(a.shape, bool), {"wmh_mm3": 0.0}))
    rows = [{"zip": "z", "prefix": "p/", "desc": "3D FLAIR", "flair_3d": True, "n_files": 176, "date": "2000-01-01"}]
    row = flair.process_subject(1, rows, str(mri.parent), str(tmp_path / "work"))
    assert row["acquisition_date"] == "2000-01-01" and row["fs_image_id"] == "I000001"


def test_manifest_finds_datscan_via_modality_dirs_and_flags_t1_mismatch(tmp_path):
    _idps(tmp_path)
    (tmp_path / "spect").mkdir()
    pd.DataFrame({"patno": [1, 2], "image_id": ["I000021", "I000022"], "error": ["", ""], "reg_metric": [-0.6, -0.6],
                  "n_label_voxels": [500, 500], "hdr_manufacturer": ["GE"] * 2, "hdr_model": ["synthetic"] * 2,
                  "hdr_scale_fit": [False] * 2, "sbr_putamen_l": [2.0, 2.1], "sbr_putamen_r": [2.0, 2.1],
                  "sbr_caudate_l": [2.5, 2.5], "sbr_caudate_r": [2.5, 2.5], "fs_image_id": ["I000001", "I000099"]})\
        .to_csv(tmp_path / "spect" / "datscan_sbr.csv", index=False)
    dirs = {"dat": tmp_path / "spect"}
    m = manifest.build_manifest(tmp_path, modality_dirs=dirs).set_index("PATNO")
    assert m.loc[1, "dat_qc_pass"] and not m.loc[2, "dat_qc_pass"] and m.loc[2, "dat_t1_mismatch"]
    f = manifest.assemble_features(tmp_path, modality_dirs=dirs).set_index("PATNO")
    assert f.loc[1, "dat_sbr_putamen_l"] == 2.0 and np.isnan(f.loc[2, "dat_sbr_putamen_l"])
    (tmp_path / "datscan_full").mkdir()
    shutil.copy(tmp_path / "spect" / "datscan_sbr.csv", tmp_path / "datscan_full")
    assert "dat_qc_pass" in manifest.build_manifest(tmp_path)          # the default location still works


# ------------------------------------------------------------------------------------------ nm --refeature
SLAB = np.diag([0.5, 0.5, 1.5, 1.0])
SLAB[:3, 3] = (-20.0, -40.0, -24.0)          # 40 x 40 x 24 mm slab around the MNI midbrain
SLAB_SHAPE = (80, 80, 16)


def _nm_subject(tmp_path, pauli_xyz):
    d = tmp_path / "nm" / "1"
    d.mkdir(parents=True)
    save = lambda a, name, dt: nib.save(nib.Nifti1Image(np.asarray(a).astype(dt), SLAB), d / name)
    save(100 + np.random.default_rng(0).normal(0, 2, SLAB_SHAPE), "nm_mean.nii.gz", np.float32)
    save(pauli_xyz, "pauli_nm.nii.gz", np.int16)
    save(np.full(SLAB_SHAPE, 16), "aseg_nm.nii.gz", np.int16)
    x = np.arange(SLAB_SHAPE[0])[:, None, None] * 0.5 - 20
    save(np.broadcast_to(x < 0, SLAB_SHAPE), "left_nm.nii.gz", np.uint8)
    return d


def _box_sn():
    p = np.zeros(SLAB_SHAPE, np.int16)
    p[10:20, 30:46, 4:9] = 7
    p[60:70, 30:46, 4:9] = 9
    return p


def _current_sha():
    from pie.imaging.atlases import cit168_provenance
    return cit168_provenance()["atlas_sha256"]


def test_nm_refeature_labels_rows_with_the_current_version(tmp_path):
    from pie.imaging.atlases import cit168_provenance
    _nm_subject(tmp_path, _box_sn())
    row = {"patno": 1, "error": "", **cit168_provenance(), "nm_sn_l_cnr": 9.9, "processing_version": "old"}
    out = nm._refeature_job((str(tmp_path / "nm"), row))
    assert out["error"] == "" and out["nm_sn_l_cnr"] != 9.9
    assert out["processing_version"] == nm.PROCESSING_VERSION + "-refeatured"


def test_nm_refeature_refuses_a_stale_atlas_it_cannot_regenerate(tmp_path):
    _nm_subject(tmp_path, _box_sn())
    out = nm._refeature_job((str(tmp_path / "nm"), {"patno": 1, "error": ""}))   # legacy row: no atlas provenance
    assert "atlas" in out["error"]


def test_nm_refeature_regenerates_a_stale_atlas_from_saved_transforms(tmp_path):
    import SimpleITK as sitk

    d = _nm_subject(tmp_path, np.zeros(SLAB_SHAPE, np.int16))
    sitk.WriteTransform(sitk.Euler3DTransform(), str(d / "slab_to_t1.tfm"))
    fs = tmp_path / "fs" / "I000001"
    (fs / "mri" / "transforms").mkdir(parents=True)
    t1_aff = np.eye(4)
    t1_aff[:3, 3] = (-20.0, -40.0, -30.0)
    nib.save(nib.MGHImage(np.ones((40, 40, 30), np.float32), t1_aff), fs / "mri" / "orig.mgz")
    sitk.WriteTransform(sitk.AffineTransform(3), str(dwi.mni_cache_path(fs)))
    dwi._write_mni_cache_provenance(dwi.mni_cache_path(fs), {})
    out = nm._refeature_job((str(tmp_path / "nm"), {"patno": 1, "error": ""}, str(fs)))
    assert out["error"] == "" and out["atlas_sha256"] == _current_sha()
    assert out["n_sn_l"] > 0 and out["n_sn_r"] > 0 and 0 < out["sn_slab_coverage"] <= 1.5
    saved = np.asanyarray(nib.load(d / "pauli_nm.nii.gz").dataobj)
    assert np.isin(saved, [7, 9]).sum() == out["n_sn_l"] + out["n_sn_r"]


# ------------------------------------------------------------------------------------------ qc datscan
def test_qc_datscan_overlays_the_t1_used_for_quantification(tmp_path, monkeypatch):
    fs = tmp_path / "fs"
    for iid in ("I000001", "I000002", "I000003"):          # I000004 (the latest session of PATNO 2) never finished
        (fs / iid / "stats").mkdir(parents=True)
        (fs / iid / "stats" / "aseg+DKT.stats").touch()
    pd.DataFrame({"patno": [1, 1, 2, 2], "image_id": ["I000001", "I000002", "I000003", "I000004"],
                  "session_date": ["2000-01-01", "2001-01-01", "2000-01-01", "2001-01-01"]}).to_csv(tmp_path / "s.csv", index=False)
    work = tmp_path / "dat"
    work.mkdir()
    pd.DataFrame({"patno": [1, 2], "image_id": ["I000011", "I000012"], "error": ["", ""], "reg_metric": [-0.6, -0.5],
                  "fs_image_id": ["I000001", None]}).to_csv(work / "datscan_sbr.csv", index=False)
    seen = {}
    monkeypatch.setattr(qc, "render_subject", lambda modality, subj, png, fastsurfer_dir=None, row=None:
                        seen.__setitem__(int(row["patno"]), Path(fastsurfer_dir).name))
    qc.main(["--work-dir", str(work), "--modality", "datscan", "--out", str(tmp_path / "qc"),
             "--sessions", str(tmp_path / "s.csv"), "--fastsurfer-dir", str(fs)])
    assert seen == {1: "I000001", 2: "I000003"}     # the row's recorded T1, else the earliest finished session


# ------------------------------------------------------------------------------------------ carrier coding
@pytest.mark.parametrize("values, expected", [
    (pd.Series([0, 1, 0], dtype="int64"), [0, 1, 0]),
    (pd.Series([0.0, 1.0, np.nan]), [0, 1, np.nan]),
    (pd.Series(["0", "G2019S", None], dtype=object), [0, 1, np.nan]),
    (pd.Series(["0.0", "1.0", np.nan], dtype=object), [0, 1, np.nan]),
    (pd.Series(["0", "R1441G", pd.NA], dtype="string"), [0, 1, np.nan]),
    (pd.Series([0, "G2019S", "", None], dtype=object), [0, 1, np.nan, np.nan]),
])
def test_carrier_coding_is_dtype_robust(values, expected):
    np.testing.assert_array_equal(labels._carrier(values), np.array(expected, float))


def test_covariates_reads_float_coded_genotypes_as_non_carriers(tmp_path):
    sc = tmp_path / "_Subject_Characteristics"
    sc.mkdir()
    pd.DataFrame({"PATNO": [1, 2, 3], "COHORT_DEFINITION": ["Healthy Control"] * 3, "ENROLL_DATE": ["01/2000"] * 3,
                  "ENROLL_AGE": [60.0] * 3}).to_csv(sc / "Participant_Status_01Jan2000.csv", index=False)
    pd.DataFrame({"PATNO": [1, 2, 3], "LAST_UPDATE": ["2000-01-01"] * 3, "SEX": [0, 1, 0], "BIRTHDT": ["01/1940"] * 3,
                  "HANDED": [1] * 3}).to_csv(sc / "Demographics_01Jan2000.csv", index=False)
    pd.DataFrame({"PATNO": [1, 2, 3], "LRRK2": [0, 1, None], "GBA": ["0", "N370S", "0"], "SNCA": [0, 0, 0],
                  "APOE": ["E3/E4", "E3/E3", None], "PATHVAR_COUNT": [0, 1, 0]}).to_csv(sc / "iu_genetic_consensus_01Jan2000.csv", index=False)
    cov = labels.covariates(tmp_path).set_index("PATNO")
    assert cov.loc[1, "LRRK2_carrier"] == 0 and cov.loc[2, "LRRK2_carrier"] == 1 and np.isnan(cov.loc[3, "LRRK2_carrier"])
    assert cov["GBA_carrier"].tolist() == [0, 1, 0] and cov["SNCA_carrier"].tolist() == [0, 0, 0]


# ------------------------------------------------------------------------------------------ stale docstrings
class _Parsed(Exception):
    pass


def test_module_docstrings_only_mention_real_cli_flags(monkeypatch):
    def grab(self, args=None, namespace=None):
        raise _Parsed(set(self._option_string_actions))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    mods = [run, dwi, nm, flair, datscan, qc, fba, dwi_refine, nm_template, cnn, embed]
    flags = set()
    for m in mods:
        with pytest.raises(_Parsed) as e:
            m.main([])
        flags |= e.value.args[0]
    for m in mods:
        for flag in re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", (m.__doc__ or "") + (m.main.__doc__ or "")):
            assert flag in flags, (m.__name__, flag)


def test_imaging_sources_do_not_cite_the_native_space_pauli_2017_atlas():
    for p in Path(pie.imaging.__file__).parent.glob("*.py"):
        if not p.name.startswith("fmri"):
            assert "Pauli 2017" not in p.read_text(), p.name


# ------------------------------------------------------------------------------------------ traps
def test_reconstruct_attenuation_default_matches_the_pipeline():
    assert inspect.signature(datscan.reconstruct).parameters["attenuation"].default is False
    assert inspect.signature(datscan.process_series).parameters["attenuation"].default is False


def test_datscan_registration_has_no_unused_mask_argument():
    for f in (datscan.register_to_t1, datscan.quantify):
        assert "mask_img" not in inspect.signature(f).parameters


def test_run_cli_passes_the_requested_device(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(run, "prepare_sessions", lambda *a, **k: pd.DataFrame({"patno": [1], "image_id": ["I000001"]}))
    monkeypatch.setattr(run, "process_all", lambda *a, device="cuda", **k: seen.update(device=device))
    monkeypatch.setattr(run, "build_idp_table", lambda sessions, d: pd.DataFrame())
    run.main(["--zips", "none.zip", "--work-dir", str(tmp_path), "--device", "cpu"])
    assert seen["device"] == "cpu"


def test_single_scan_retry_uses_the_requested_device(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(run, "run_fastsurfer", lambda *a, **k: seen.update(k))
    run.retry_one({"patno": 1, "image_id": "I000001"}, tmp_path, 2, device="cpu")
    assert seen["device"] == "cpu"


def test_cpu_segmentation_gets_a_longer_time_budget(tmp_path, monkeypatch):
    from pie.imaging import fastsurfer
    calls = []

    def fake_run(cmd, log, cwd, timeout=1800, stall=None, stall_seconds=fastsurfer.STALL_SECONDS):
        calls.append((timeout, stall_seconds))
        if "--asegdkt_segfile" in cmd:                       # single-scan inference must leave its segmentation
            Path(cmd[cmd.index("--asegdkt_segfile") + 1]).touch()

    monkeypatch.setattr(fastsurfer, "_run", fake_run)
    budgets = {}
    for device in ("cuda", "cpu"):
        shutil.rmtree(tmp_path / "fs", ignore_errors=True)
        calls.clear()
        fastsurfer.segment(tmp_path / "S1_T1w.nii.gz", tmp_path / "fs", "S1", device=device)
        fastsurfer.segment_batch([tmp_path / "S2_T1w.nii.gz"], tmp_path / "fs", device=device)
        budgets[device] = list(calls)
    (gpu_seg, gpu_batch), (cpu_seg, cpu_batch) = budgets["cuda"], budgets["cpu"]
    assert gpu_seg[0] == 900 and gpu_batch == (90 + 300, fastsurfer.STALL_SECONDS)       # GPU budgets unchanged
    assert cpu_seg[0] > gpu_seg[0] and cpu_batch[0] > gpu_batch[0] and cpu_batch[1] > gpu_batch[1]


def test_tool_locations_can_be_overridden_by_environment():
    code = ("from pie.imaging import convert, batch, fastsurfer, embed; "
            "print(convert.DCM2NIIX, batch.DCM2NIIX, fastsurfer.FASTSURFER_HOME, fastsurfer.PYTHON, embed.WEIGHTS['sfcn'])")
    env = {**os.environ, "PIE_DCM2NIIX": "/opt/x/dcm2niix", "PIE_FASTSURFER_HOME": "/opt/fs",
           "PIE_FASTSURFER_PYTHON": "/opt/py/python", "PIE_WEIGHTS_DIR": "/opt/w"}
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert out == ["/opt/x/dcm2niix", "/opt/x/dcm2niix", "/opt/fs", "/opt/py/python", "/opt/w/sfcn/run_20190719_00_epoch_best_mae.p"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIE_")}
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == str(ROOT / "venv_imaging" / "bin" / "dcm2niix") and out[2] == str(ROOT / "third_party" / "FastSurfer")


# ------------------------------------------------------------------------------------------ PPMI working-group audit
def _ppmi(tmp_path, xing=True):
    """Minimal PPMI download: 20 visually-negative controls whose putamen SBR falls with age, one PD, one TRODAT scan."""
    sc, im = tmp_path / "PPMI/_Subject_Characteristics", tmp_path / "PPMI/Imaging"
    sc.mkdir(parents=True), im.mkdir(parents=True)
    n = 22
    patno, age = np.arange(1, n + 1), np.r_[np.linspace(50, 80, 20), 60, 60]
    pd.DataFrame({"PATNO": patno, "COHORT_DEFINITION": ["Healthy Control"] * 20 + ["Parkinson's Disease"] * 2,
                  "ENROLL_DATE": "01/2011", "ENROLL_AGE": age}).to_csv(sc / "Participant_Status_01Jan2020.csv", index=False)
    pd.DataFrame({"PATNO": patno, "SEX": np.arange(n) % 2, "BIRTHDT": [f"01/{2011 - int(a)}" for a in age], "HANDED": 1,
                  "LAST_UPDATE": "2020-01-01"}).to_csv(sc / "Demographics_01Jan2020.csv", index=False)
    pd.DataFrame({"PATNO": patno, "LRRK2": 0, "GBA": 0, "SNCA": 0, "APOE": "E3/E3", "PATHVAR_COUNT": 0})\
        .to_csv(sc / "iu_genetic_consensus_20251025_01Jan2020.csv", index=False)
    put = np.r_[3.0 - 0.02 * (age[:20] - 50), 0.9, 1.3]              # PD 21 at 0.9 / 2.4 = 38 %, PD 22 at 54 % of expected
    rows = pd.DataFrame({"PATNO": patno, "EVENT_ID": "SC", "DATSCAN_DATE": "01/2011", "DATSCAN_ANALYZED": "Yes",
                         "DATSCAN_LIGAND": [None] * 10 + ["123I-DaTscan"] * 11 + ["99mTc-TRODAT-1"]})   # PPMI-1 rows: blank
    vis = pd.DataFrame({"PATNO": patno, "DATSCAN_DATE": "01/2011", "DATSCAN_VISINTRP": ["negative"] * 20 + ["positive"] * 2,
                        "DATSCAN_LIGAND": "123I-DaTscan"})
    if xing:
        for r, v in (("PUTAMEN", put), ("CAUDATE", put + 0.5)):
            rows[f"{r}_L_REF_CWM"], rows[f"{r}_R_REF_CWM"] = v, v * 1.1
        rows.to_csv(im / "Xing_Core_Lab_-_Quant_SBR_01Jan2020.csv", index=False)
        vis.to_csv(im / "Xing_Core_Lab_-_Visual_Read_01Jan2020.csv", index=False)
    rows.assign(DATSCAN_PUTAMEN_L=put + 1, DATSCAN_PUTAMEN_R=put + 1, DATSCAN_CAUDATE_L=put + 2, DATSCAN_CAUDATE_R=put + 2)\
        .to_csv(im / "DaTScan_SBR_Analysis_01Jan2019.csv", index=False)
    vis.to_csv(im / "DaTScan_Visual_Interpretation_Results_01Jan2019.csv", index=False)
    return tmp_path / "PPMI", pd.DataFrame({"patno": patno, "image_id": [f"I{p:06d}" for p in patno], "session_date": "2011-01-15"})


def test_dat_labels_use_ppmi_primary_sbr_table_and_named_deficit_rules(tmp_path):
    ppmi, sessions = _ppmi(tmp_path)
    d = labels.dat_labels(ppmi, sessions).set_index("PATNO")
    assert set(d["sbr_source"]) == {"xing_cwm"} and 22 not in d.index and 1 in d.index   # TRODAT out, blank ligand in
    assert d.loc[21, "dat_deficit_sbr"] == 1 and d.loc[21, "dat_d_nsdiss"] == 1
    assert abs(d.loc[21, "sbr_pct_expected"] - 0.9 / 2.8) < 0.02 and d.loc[21, "sbr_putamen_min_z"] < -10
    assert d.loc[1:20, "dat_deficit_sbr"].eq(0).all() and abs(d.loc[1, "sbr_ai_putamen"] - 0.3 / 3.15 * 100) < 1e-6
    assert abs(d.loc[1, "sbr_pc_ratio_l"] - 3.0 / 3.5) < 1e-6
    (ppmi / "Imaging" / "Xing_Core_Lab_-_Quant_SBR_01Jan2020.csv").unlink()
    assert set(labels.dat_labels(ppmi, sessions)["sbr_source"]) == {"invicro_occipital"}   # archived table as fallback
    with pytest.raises(FileNotFoundError):
        labels.dat_labels(ppmi, sessions, source="xing")


def test_dat_deficit_rule_includes_the_threshold_itself():
    x = pd.Series([0.65, 0.6500001, 0.75, 0.76])
    assert ((x <= 0.65).astype(float).tolist(), (x <= 0.75).astype(float).tolist()) == ([1, 0, 0, 0], [1, 1, 1, 0])


def test_assembly_quarantines_single_shell_free_water_and_masked_dates(tmp_path):
    pd.DataFrame({"PATNO": [1, 2], "IMAGEID": ["I1", "I2"], "SCAN_DATE": ["2022-01-01", "9999-01-01"],
                  "vol_Left_Putamen": [1., 1.]}).to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    (tmp_path / "dwi").mkdir()
    pd.DataFrame({"patno": [1, 2], "motion_mm_max": [1., 1.], "motion_mm_mean": [.5, .5], "sn_brain_mask_fraction": [1., 1.], "sn_physical_fraction": [.95, .95], "sn_posterior_l_fa": [.5, .5], "sn_posterior_r_fa": [.5, .5], "n_sn_l": [15, 15], "n_sn_r": [15, 15],
                  "fa_wm_median": [.4, .4], "manufacturer": ["Siemens"] * 2, "shells": ["1000", "700 1000 2000"],
                  "fw_method": ["singleshell_prior", "multishell_nls"],
                  "acquisition_date": ["2022-01-03"] * 2, "sn_l_fw": [.2, .3], "sn_l_fat": [.5, .6], "sn_l_fa": [.4, .4],
                  "sn_l_ad": [1e-3, 1e-3], "sn_l_mk": [np.nan, .8]}).to_csv(tmp_path / "dwi" / "dwi_features.csv", index=False)
    f = manifest.assemble_features(tmp_path).set_index("PATNO")
    assert np.isnan(f.loc[1, "dwi_sn_l_fw"]) and np.isnan(f.loc[1, "dwi_sn_l_fat"]) and f.loc[1, "dwi_sn_l_fa"] == .4
    assert f.loc[2, "dwi_sn_l_fw"] == .3 and f.loc[2, "dwi_sn_l_mk"] == .8 and f.loc[1, "dwi_sn_l_ad"] == 1e-3
    assert f.loc[1, "dwi_days_from_t1"] == 2 and pd.isna(f.loc[2, "t1_date"]) and pd.isna(f.loc[2, "dwi_days_from_t1"])
    assert manifest.assemble_features(tmp_path, single_shell_fw=True).set_index("PATNO").loc[1, "dwi_sn_l_fw"] == .2


def test_assembled_dat_block_carries_both_references_and_indices(tmp_path):
    _idps(tmp_path)
    (tmp_path / "datscan_full").mkdir()
    pd.DataFrame({"patno": [1], "image_id": ["I000021"], "error": [""], "reg_metric": [-0.6], "n_label_voxels": [500],
                  "hdr_manufacturer": ["GE"], "hdr_model": ["synthetic"], "sbr_putamen_l": [1.0], "sbr_putamen_r": [2.0],
                  "sbr_caudate_l": [2.0], "sbr_caudate_r": [2.0], "sbr_putamen_l_post": [0.8], "sbrwm_putamen_l": [0.5],
                  "sbrwm_putamen_r": [1.0], "sbrwm_caudate_l": [1.0], "sbrwm_caudate_r": [-0.1], "mean_occipital": [10.]})\
        .to_csv(tmp_path / "datscan_full" / "datscan_sbr.csv", index=False)
    f = manifest.assemble_features(tmp_path).set_index("PATNO")
    assert f.loc[1, "dat_sbr_pc_ratio_l"] == 0.5 and abs(f.loc[1, "dat_sbr_ai_putamen"] - 100 / 1.5) < 1e-9
    assert f.loc[1, "dat_sbr_putamen_l_post"] == 0.8 and f.loc[1, "dat_sbrwm_pc_ratio_l"] == 0.5
    assert np.isnan(f.loc[1, "dat_sbrwm_pc_ratio_r"]) and "dat_mean_occipital" not in f      # non-positive SBR -> NaN
    assert "dat_sbrwm_ai_caudate" in manifest.feature_blocks(f.columns)["dat"]


def test_tensor_fit_reports_axial_radial_and_multishell_kurtosis():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(3, 30))
    g /= np.linalg.norm(g, axis=0)
    D = np.diag([1.5e-3, 0.4e-3, 0.4e-3])
    for bvals in (np.r_[0, 0, np.full(30, 1000.)], np.r_[0, 0, np.full(30, 700.), np.full(30, 1000.), np.full(30, 2000.)]):
        bvecs = np.c_[np.zeros((3, 2)), np.tile(g, len(bvals) // 30)]
        sig = 1000 * (0.8 * np.exp(-bvals * np.einsum("in,ij,jn->n", bvecs, D, bvecs)) + 0.2 * np.exp(-bvals * dwi.D_WATER))
        ds = {"bvals": bvals, "bvecs": bvecs, "data": np.tile(sig, (2, 2, 2, 1)).astype(np.float32), "mask": np.ones((2, 2, 2), bool)}
        maps = dwi.fit_models(ds)
        assert maps["ad"][0, 0, 0] > maps["md"][0, 0, 0] > maps["rd"][0, 0, 0] > 0
        assert ("mk" in maps) == (len(bvals) > 32)
    assert abs(maps["mdt"][0, 0, 0] - np.trace(D) / 3) < 2e-5 and 0 < maps["mk"][0, 0, 0] < 3


def test_brain_age_gap_removes_regression_to_the_mean():
    rng = np.random.default_rng(0)
    age = rng.uniform(45, 80, 400)
    predicted = 0.6 * age + 25 + rng.normal(0, 3, 400)          # a typically shrunk brain-age model
    predicted[:50] += 5                                          # an older-looking group
    raw, gap = predicted - age, embed.brain_age_gap(predicted, age, reference=np.arange(400) >= 50)
    assert np.corrcoef(raw, age)[0, 1] < -0.5 and abs(np.corrcoef(gap[50:], age[50:])[0, 1]) < 0.1
    assert abs(gap[:50].mean() - 5) < 1


def test_templateflow_files_are_checked_against_pinned_hashes(tmp_path):
    from pie.imaging import atlases
    name = "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
    (tmp_path / name).write_bytes(b"not the template")
    with pytest.raises(ValueError, match="checksum"):
        atlases.templateflow_file(name, cache_dir=tmp_path)
    assert "2009cAsym" in nm_template.syn_paths(tmp_path)["fwd"][0].name     # legacy 2009a SyN caches are never reused


def test_ants_registrations_leave_no_transform_files_behind(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))            # where ANTsPy's mktemp() puts its transforms
    base = np.zeros((32, 32, 32), np.float32)
    base[8:24, 8:24, 8:24], base[12:20, 10:22, 10:22] = 0.3, 0.7
    img = nib.Nifti1Image(base, np.eye(4))
    _, tx = dwi.map_labels_to_subject(nib.Nifti1Image(np.roll(base, 2, axis=0), np.eye(4)), img,
                                      nib.Nifti1Image((base > 0.5).astype(np.uint8), np.eye(4)), syn=False)
    assert not list(tmp_path.iterdir()) and "fwdtransforms" not in tx


# ------------------------------------------------------------------------------------------ weak-point plan (26 Sep 2026)
def test_freesurfer_env_names_missing_settings_and_builds_path(tmp_path, monkeypatch):
    from pie.imaging import freesurfer
    monkeypatch.setattr(freesurfer, "HOME", None)
    with pytest.raises(RuntimeError, match="PIE_FREESURFER_HOME"):
        freesurfer.fs_env(home=None, license=str(tmp_path / "license.txt"))
    home = tmp_path / "fs"
    (home / "bin").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="PIE_FS_LICENSE"):
        freesurfer.fs_env(home=str(home), license=str(tmp_path / "absent.txt"))
    (tmp_path / "license.txt").write_text("x\n")
    env = freesurfer.fs_env(home=str(home), license=str(tmp_path / "license.txt"))
    assert env["FREESURFER_HOME"] == str(home) and env["FS_LICENSE"] == str(tmp_path / "license.txt")
    assert env["PATH"].split(os.pathsep)[0] == str(home / "bin") and env["SUBJECTS_DIR"]


def test_fs7_tables_attach_only_to_the_same_visit(tmp_path):
    from pie.imaging import features
    im = tmp_path / "PPMI/Imaging"
    im.mkdir(parents=True)
    pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": "BL", "lh_cuneus": [2.1, 2.2]}).to_csv(im / "FS7_APARC_CTH_01Jan2020.csv", index=False)
    pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": "BL", "lh_cuneus": [900., 950.]}).to_csv(im / "FS7_APARC_SA_01Jan2020.csv", index=False)
    pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": "BL", "EstimatedTotalIntraCranialVol": [1.5e6, 1.6e6]}).to_csv(im / "FS7_ASEG_VOL_01Jan2020.csv", index=False)
    pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": "BL", "cnr": [3.0, 3.1]}).to_csv(im / "MRIQC_01Jan2020.csv", index=False)
    t = features.fs7_tables(tmp_path / "PPMI").set_index("PATNO")
    assert t.loc[1, "fs7_cth_lh_cuneus"] == 2.1 and t.loc[1, "fs7_sa_lh_cuneus"] == 900 and t.loc[2, "mriqc_cnr"] == 3.1
    pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": ["BL", "V04"], "IMAGEID": ["I1", "I2"], "SCAN_DATE": ["2020-01-01"] * 2,
                  "vol_Left_Putamen": [1., 1.]}).to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    f = manifest.assemble_features(tmp_path, ppmi_dir=tmp_path / "PPMI").set_index("PATNO")
    assert f.loc[1, "fs7_EstimatedTotalIntraCranialVol"] == 1.5e6 and np.isnan(f.loc[2, "fs7_EstimatedTotalIntraCranialVol"])
    assert f.loc[1, "fs7_cth_lh_cuneus"] == 2.1 and np.isnan(f.loc[2, "mriqc_cnr"])


def test_etiv_follows_freesurfer_atlas_scaling_and_backfill_resumes(tmp_path, monkeypatch):
    from pie.imaging import fastsurfer
    xfm = tmp_path / "talairach.xfm"
    xfm.write_text("MNI Transform File\nTransform_Type = Linear;\nLinear_Transform =\n"
                   "1.1 0 0 1.0\n0 1.1 0 2.0\n0 0 1.1 3.0;\n")
    assert abs(fastsurfer.etiv_from_xfm(xfm) - 1948106 / 1.1 ** 3) < 1e-6
    calls, afd_ok = [], {"S1": True, "S2": False}

    def fake_run(cmd, log, cwd, env=None, timeout=3600):
        cmd = [str(c) for c in cmd]
        calls.append((cmd, Path(cwd)))
        sid = Path(cwd).parent.name
        if cmd[0].endswith("talairach-reg.sh"):
            assert Path(cmd[1]).exists()                            # talairach-reg.sh refuses a log path that does not exist
            (Path(cwd) / "transforms").mkdir(exist_ok=True)
            shutil.copy(xfm, Path(cwd) / "transforms" / "talairach.xfm")
            (Path(cwd) / "nu.mgz").write_bytes(b"x" * 100)
        elif cmd[0] == "talairach_afd" and not afd_ok[sid]:
            raise RuntimeError("talairach_afd exited 1")

    monkeypatch.setattr(fastsurfer.freesurfer, "run", fake_run)
    for sid in ("S1", "S2"):
        mri = tmp_path / "fs" / sid / "mri"
        mri.mkdir(parents=True)
        for f in ("orig.mgz", "orig_nu.mgz", "aparc.DKTatlas+aseg.deep.mgz"):
            (mri / f).touch()
    v = fastsurfer.talairach_etiv(tmp_path / "fs", "S1", env={})
    mri = (tmp_path / "fs" / "S1" / "mri").resolve()
    assert abs(v - 1948106 / 1.331) < 1e-3 and calls[0][0][0].endswith("talairach-reg.sh") and calls[0][1] == mri
    assert calls[1][0][:3] == ["talairach_afd", "-T", "0.005"] and not (mri / "nu.mgz").exists()   # 7 MB each, unused
    fastsurfer.talairach_etiv(tmp_path / "fs", "S1", env={})
    assert len(calls) == 2                                                  # resumes: registration and check are reused
    assert np.isnan(fastsurfer.talairach_etiv(tmp_path / "fs", "S2", env={}))      # failed registration check: no eTIV
    sessions = pd.DataFrame({"patno": [1, 2], "image_id": ["S1", "S2"], "session_date": ["2000-01-01"] * 2, "EVENT_ID": ["BL"] * 2,
                             "protocol_phase": [1, 1]})
    for sid in ("S1", "S2"):
        (tmp_path / "fs" / sid / "stats").mkdir()
        (tmp_path / "fs" / sid / "stats" / "aseg+DKT.stats").write_text("# Measure Mask, MaskVol, Mask Volume, 1500000.0, mm^3\n")
    t = features.build_idp_table(sessions, tmp_path / "fs").set_index("IMAGEID")
    assert abs(t.loc["S1", "eTIV"] - 1948106 / 1.331) < 1e-3 and np.isnan(t.loc["S2", "eTIV"])
    (tmp_path / "fs" / "S1" / "mri" / "transforms" / "talairach.xfm").write_text("truncated")
    assert np.isnan(features.build_idp_table(sessions, tmp_path / "fs").set_index("IMAGEID").loc["S1", "eTIV"])   # never aborts


def test_run_etiv_backfills_finished_subjects_and_rebuilds_the_table(tmp_path, monkeypatch):
    work = tmp_path / "derived"
    for sid in ("S1", "S2"):
        (work / "fastsurfer" / sid / "stats").mkdir(parents=True)
        (work / "fastsurfer" / sid / "stats" / "aseg+DKT.stats").write_text("# Measure Mask, MaskVol, Mask Volume, 1500000.0, mm^3\n")
    pd.DataFrame({"patno": [1, 2], "image_id": ["S1", "S2"], "session_date": ["2000-01-01"] * 2, "EVENT_ID": ["BL"] * 2,
                  "protocol_phase": [1, 1]}).to_csv(work / "sessions.csv", index=False)

    def fake_etiv(fs_dir, sid, **kw):
        if sid == "S2":
            raise RuntimeError("talairach_avi failed")
        t = Path(fs_dir) / sid / "mri" / "transforms"
        t.mkdir(parents=True, exist_ok=True)
        (t / "talairach.xfm").write_text("Linear_Transform =\n1 0 0 0\n0 1 0 0\n0 0 1 0;\n")
        (t / "talairach.afd").write_text("pass\n")
        return 1948106.0

    monkeypatch.setattr(run, "talairach_etiv", fake_etiv)
    monkeypatch.setattr(run, "_freesurfer_ready", lambda tcsh=False: None)
    run.main(["--zips", "unused.zip", "--work-dir", str(work), "--etiv", "--workers", "1"])
    t = pd.read_csv(work / "fastsurfer_idps.csv").set_index("IMAGEID")
    assert t.loc["S1", "eTIV"] == 1948106.0 and np.isnan(t.loc["S2", "eTIV"])     # a failure is missing, not fatal


def test_tiv_prefers_pie_etiv_then_same_visit_fs7(tmp_path):
    pd.DataFrame({"PATNO": [1, 2, 3], "EVENT_ID": ["BL", "BL", "V04"], "IMAGEID": ["I1", "I2", "I3"], "SCAN_DATE": ["2020-01-01"] * 3,
                  "vol_Left_Putamen": [1.] * 3, "eTIV": [1.4e6, np.nan, np.nan]}).to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    im = tmp_path / "PPMI/Imaging"
    im.mkdir(parents=True)
    pd.DataFrame({"PATNO": [1, 2, 3], "EVENT_ID": "BL", "EstimatedTotalIntraCranialVol": [1.5e6, 1.6e6, 1.7e6]})\
        .to_csv(im / "FS7_ASEG_VOL_01Jan2020.csv", index=False)
    f = manifest.assemble_features(tmp_path, ppmi_dir=tmp_path / "PPMI").set_index("PATNO")
    assert (f.loc[1, "tiv_mm3"], f.loc[1, "tiv_source"]) == (1.4e6, "pie_talairach")
    assert (f.loc[2, "tiv_mm3"], f.loc[2, "tiv_source"]) == (1.6e6, "ppmi_fs7")
    assert np.isnan(f.loc[3, "tiv_mm3"]) and f.loc[3, "tiv_source"] == "none"
    assert manifest.assemble_features(tmp_path).set_index("PATNO").loc[2, "tiv_source"] == "none"   # no PPMI dir: PIE only


def test_surface_stream_runs_the_cc_step_first_and_resumes_on_the_done_marker(tmp_path, monkeypatch):
    from pie.imaging import fastsurfer
    calls = []

    def fake_run(cmd, log, cwd, env=None, timeout=3600):
        cmd = [str(c) for c in cmd]
        calls.append((cmd, Path(cwd), env))
        if cmd[1].endswith("paint_cc_into_pred.py"):
            (tmp_path / "S1" / "mri" / "aseg.auto.mgz").touch()
        if cmd[0].endswith("recon-surf.sh"):
            assert (tmp_path / "S1" / "mri" / "aseg.auto.mgz").exists()               # recon-surf's prerequisite
            assert not (tmp_path / "S1" / "mri" / "wm.mgz").exists()                  # a killed run's leftovers are gone
            (tmp_path / "S1" / "scripts" / "recon-surf.done").touch()

    monkeypatch.setattr(fastsurfer.freesurfer, "run", fake_run)
    (tmp_path / "S1" / "mri").mkdir(parents=True)
    (tmp_path / "S1" / "mri" / "wm.mgz").touch()                                     # left by a killed recon-surf
    (tmp_path / "S1" / "stats").mkdir()
    (tmp_path / "S1" / "stats" / "lh.aparc.stats").touch()                           # written mid-run: not completion
    fastsurfer.surfaces(tmp_path, "S1", env={"FS_LICENSE": "/lic.txt"})
    fastsurfer.surfaces(tmp_path, "S1", env={"FS_LICENSE": "/lic.txt"})
    names = [Path(c[0][1] if c[0][0].endswith("python") else c[0][0]).name for c in calls]
    assert names == ["fastsurfer_cc.py", "paint_cc_into_pred.py", "recon-surf.sh"]
    surf, cwd, env = calls[2]
    assert {"--fsaparc", "--parallel", "/lic.txt", "--mask_name"} <= set(surf) and "--ignore_fs_version" not in surf
    assert cwd.name == "recon_surf" and env["PYTHONPATH"].endswith("FastSurfer")       # recon-surf imports FastSurferCNN
    assert calls[0][2]["PYTHONPATH"] == env["PYTHONPATH"]


def test_assembly_reads_nm_template_features_with_their_own_qc(tmp_path):
    _idps(tmp_path)
    (tmp_path / "nm").mkdir()
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "nmt_sn_mean_cnr": [.2, .3], "nmt_sn_cov_l": [1., .5], "nmt_sn_cov_r": [1., 1.],
                  "nmt_crus_cv_l": [.1, .1], "nmt_crus_cv_r": [.1, .1], "nmt_crus_mode_l": [100., 100.]})\
        .to_csv(tmp_path / "nm" / "nm_template_features.csv", index=False)
    f = manifest.assemble_features(tmp_path).set_index("PATNO")
    assert f.loc[1, "nmt_sn_mean_cnr"] == .2 and np.isnan(f.loc[2, "nmt_sn_mean_cnr"]) and not f.loc[2, "nmt_qc_pass"]
    assert "nmt_crus_mode_l" not in f and "nmt_sn_mean_cnr" in manifest.feature_blocks(f.columns)["nm"]


def test_ppmi_manual_dti_rois_become_one_row_per_scan(tmp_path):
    im = tmp_path / "PPMI/Imaging"
    im.mkdir(parents=True)
    rows = []
    for m, v in (("FA", .3), ("E1", 1.2e-3), ("E2", .7e-3), ("E3", .5e-3)):
        rows.append({"PATNO": 1, "PAG_NAME": "DTIROI", "INFODT": "01/2011", "Measure": m, "Tissue": "SN",
                     **{f"ROI{i}": v + (0.01 if i in (3, 6) and m == "FA" else 0) for i in range(1, 7)},
                     "REF1": .6, "REF2": .62, "RUNDATE": "2015-01-01"})
    pd.DataFrame(rows).to_csv(im / "DTI_Regions_of_Interest_01Jan2020.csv", index=False)
    t = labels.ppmi_dti_roi_table(tmp_path / "PPMI").iloc[0]
    assert abs(t.sn_fa - (0.3 + 0.02 / 6)) < 1e-9 and abs(t.sn_caudal_fa - 0.31) < 1e-9 and abs(t.sn_md - 0.8e-3) < 1e-12
    assert abs(t.peduncle_fa - 0.61) < 1e-9 and t.DTI_DATE == pd.Timestamp("2011-01-01") and t.PATNO == 1


def test_pe_restricted_sdc_recovers_a_phase_encoding_shift():
    from scipy.ndimage import map_coordinates
    shape = (40, 48, 40)                                                   # (z, y, x) arrays
    zz, yy, xx = np.indices(shape)
    t1 = np.zeros(shape, np.float32)
    t1[8:32, 8:40, 8:32] = 1.0
    t1[16:24, 18:30, 14:26] = 2.0                                          # a "nucleus"
    lab = np.zeros(shape, np.int16)
    lab[16:24, 18:30, 14:26] = 7
    shift = 3.0 * np.exp(-((xx - 20) ** 2 + (zz - 20) ** 2) / 200.0)       # voxels, along y only
    b0 = map_coordinates(np.where(t1 > 0, 3.0 - t1, 0.0), [zz, yy - shift, xx], order=1).astype(np.float32)   # inverted contrast
    to_img = lambda a: nib.Nifti1Image(np.ascontiguousarray(np.transpose(a, (2, 1, 0))), np.eye(4))           # stored (x, y, z)
    fn, info = dwi.register_b0_to_t1_sdc(to_img(b0), to_img(t1), to_img((t1 > 0).astype(np.float32)), pe_axis="j")
    try:
        got = fn(to_img(lab)) == 7
        truth = map_coordinates((lab == 7).astype(float), [zz, yy - shift, xx], order=0) > 0.5
        dice = 2 * (got & truth).sum() / (got.sum() + truth.sum())
        assert dice > 0.8 and 0.5 < info["sdc_max_displacement_mm"] < 8 and info["sdc_offaxis_max_mm"] < 0.1
    finally:
        shutil.rmtree(info["transform_dir"])


def test_eddy_correct_consumes_eddy_outputs_once_and_cleans_up(tmp_path, monkeypatch):
    fsl = tmp_path / "fsl" / "bin"
    fsl.mkdir(parents=True)
    (fsl / "topup").write_text("#!/bin/sh\nfor a in \"$@\"; do case $a in --out=*) o=${a#--out=};; esac; done\n"
                               "cp b0_pair.nii.gz ${o}_fieldcoef.nii.gz; printf '0 0 0 0 0 0\\n0 0 0 0 0 0\\n' > ${o}_movpar.txt\n")
    (fsl / "eddy_cuda").write_text(
        "#!/bin/sh\ncp raw.nii.gz eddy.nii.gz\n"
        f"{sys.executable} -c \"import numpy as n; v=n.loadtxt('bvecs'); v[:, 2:] = n.roll(v[:, 2:], 1, axis=0); n.savetxt('eddy.eddy_rotated_bvecs', v)\"\n"
        "printf 'hdr\\n0 1\\n0 0\\n0 0\\n0 0\\n' > eddy.eddy_outlier_map\nprintf '0 0.5\\n0.4 0.7\\n0.2 0.2\\n0.1 0.1\\n' > eddy.eddy_movement_rms\n")
    for f in fsl.iterdir():
        f.chmod(0o755)
    monkeypatch.setattr(dwi, "FSLDIR", str(tmp_path / "fsl"))
    g = np.eye(3)[:, [0, 1]]
    ds = {"data": np.ones((4, 4, 4, 4), np.float32), "affine": np.eye(4), "bvals": np.array([0., 0., 1000., 1000.]),
          "bvecs": np.c_[np.zeros((3, 2)), g], "meta": {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05},
          "rev_b0": np.ones((4, 4, 4, 1), np.float32), "rev_meta": {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05}}
    out = dwi.eddy_correct(ds, tmp_path / "work", cuda=True)
    assert out["eddy"] and out["topup"] and out["bvecs_rotated"] and abs(out["eddy_outlier_fraction"] - 1 / 8) < 1e-9
    assert np.allclose(out["bvecs"][:, 2:], np.roll(g, 1, axis=0)) and out["data"].shape == ds["data"].shape
    assert out["motion_mm_max"] == 0.4 and not (tmp_path / "work" / "eddy").exists()
    with pytest.raises(ValueError, match="PhaseEncodingDirection"):
        dwi.eddy_correct({**ds, "rev_meta": {}}, tmp_path / "work", cuda=True)       # unverified reverse PE: no eddy


def test_eddy_path_replaces_rigid_motion_only_when_a_reverse_b0_exists(monkeypatch):
    calls = []
    monkeypatch.setattr(dwi, "eddy_correct", lambda ds, work, cuda=True: (calls.append("eddy"), dict(ds, eddy=True, topup=True, bvecs_rotated=True,
                        motion_mm_max=0.3, motion_mm_mean=0.1, eddy_outlier_fraction=0.01))[1])
    monkeypatch.setattr(dwi, "preprocess", lambda ds, sampling_seed=0: (calls.append("rigid"), dict(ds, mask=np.ones((4, 4, 4), bool),
                        b0=np.ones((4, 4, 4)), motion_mm_max=1., motion_mm_mean=.5, rotation_deg_max=1., bvecs_rotated=True))[1])
    base = {"data": np.ones((4, 4, 4, 3), np.float32), "affine": np.eye(4), "bvals": np.array([0., 1000., 1000.]), "meta": {}}
    out = dwi.correct(dict(base, rev_b0=np.ones((4, 4, 4, 1))), "w", eddy=True)
    assert calls == ["eddy"] and out["eddy"] and out["mask"].shape == (4, 4, 4) and out["b0"].shape == (4, 4, 4)
    assert out["motion_metric"] == "eddy_rms_mm"                             # eddy's RMS displacement, not a translation norm
    calls.clear()
    out = dwi.correct(dict(base, rev_b0=None), "w", eddy=True)
    assert calls == ["rigid"] and not out.get("eddy", False)                 # no reverse b0: the rigid path, recorded as such
    assert out["motion_metric"] == "rigid_translation_mm"


def test_dwi_batch_separates_eddy_from_rigid_correction(tmp_path):
    pd.DataFrame({"PATNO": [1, 2], "IMAGEID": ["I1", "I2"], "SCAN_DATE": ["2022-01-01"] * 2, "vol_Left_Putamen": [1., 1.]})\
        .to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    (tmp_path / "dwi").mkdir()
    pd.DataFrame({"patno": [1, 2], "motion_mm_max": [1., 1.], "n_sn_l": [4, 4], "n_sn_r": [4, 4], "fa_wm_median": [.4, .4],
                  "manufacturer": ["Siemens"] * 2, "shells": ["700 1000 2000"] * 2, "fw_method": ["multishell_nls"] * 2,
                  "eddy": [True, False], "sn_l_fa": [.4, .4]}).to_csv(tmp_path / "dwi" / "dwi_features.csv", index=False)
    m = manifest.build_manifest(tmp_path).set_index("PATNO")
    assert m["dwi_batch"].nunique() == 2 and m.loc[1, "dwi_batch"].endswith("_eddy") and bool(m.loc[1, "dwi_eddy"])


def test_long_timepoints_need_two_real_dated_finished_sessions(tmp_path):
    for sid in ("A", "B", "C", "D"):
        (tmp_path / sid / "stats").mkdir(parents=True)
        (tmp_path / sid / "stats" / "aseg+DKT.stats").touch()
    sessions = pd.DataFrame({"patno": [1, 1, 1, 2], "image_id": ["B", "A", "C", "D"],
                             "session_date": ["2012-01-01", "2010-01-01", "9999-01-01", "2010-01-01"]})
    assert run.long_timepoints(sessions, tmp_path) == {1: ["A", "B"]}


def test_longitudinal_stream_command_and_resume(tmp_path, monkeypatch):
    from pie.imaging import fastsurfer
    calls = []

    def fake_run(cmd, log, cwd, env=None, timeout=3600):
        calls.append([str(c) for c in cmd])
        for tp in ("A", "B"):
            (tmp_path / "long" / tp / "scripts").mkdir(parents=True, exist_ok=True)
            (tmp_path / "long" / tp / "scripts" / "recon-surf.done").touch()

    monkeypatch.setattr(fastsurfer.freesurfer, "run", fake_run)
    (tmp_path / "long" / "1" / "mri").mkdir(parents=True)                  # partial template from a killed run
    (tmp_path / "long" / "A" / "stats").mkdir(parents=True)
    (tmp_path / "long" / "A" / "stats" / "aseg.stats").touch()              # written early: not a completion marker
    args = ("1", ["/n/A_T1w.nii.gz", "/n/B_T1w.nii.gz"], ["A", "B"], tmp_path / "long")
    fastsurfer.longitudinal(*args, env={"FS_LICENSE": "/lic.txt"}, device="cpu")
    assert not (tmp_path / "long" / "A" / "stats" / "aseg.stats").exists()       # partial outputs cleared before the retry
    fastsurfer.longitudinal(*args, env={"FS_LICENSE": "/lic.txt"}, device="cpu")
    c = calls[0]
    assert len(calls) == 1 and c[0].endswith("long_fastsurfer.sh") and c[1:3] == ["--tid", "1"]
    assert c[c.index("--t1s") + 1:c.index("--t1s") + 3] == ["/n/A_T1w.nii.gz", "/n/B_T1w.nii.gz"]
    assert c[c.index("--tpids") + 1:c.index("--tpids") + 3] == ["A", "B"] and c[c.index("--device") + 1] == "cpu"
    assert "--fs_license" not in c        # long_fastsurfer.sh lowercases pass-through values; FS_LICENSE comes from the environment


def test_run_long_passes_date_ordered_niftis_and_continues_after_a_failure(tmp_path, monkeypatch):
    work = tmp_path / "derived"
    for sid in ("A", "B", "C", "D"):
        (work / "fastsurfer" / sid / "stats").mkdir(parents=True)
        (work / "fastsurfer" / sid / "stats" / "aseg+DKT.stats").touch()
    pd.DataFrame({"patno": [1, 1, 2, 2], "image_id": ["B", "A", "C", "D"], "session_date": ["2012-01-01", "2010-01-01", "2010-01-01", "2011-01-01"],
                  "EVENT_ID": ["V04", "BL", "BL", "V04"], "protocol_phase": [1] * 4}).to_csv(work / "sessions.csv", index=False)
    seen = []

    def fake_long(tid, t1s, tpids, sd, device="cuda", threads=4):
        seen.append((tid, [Path(t).name for t in t1s], tpids, Path(sd).name))
        if tid == "1":
            raise RuntimeError("recon-surf failed")

    monkeypatch.setattr(run, "longitudinal", fake_long)
    monkeypatch.setattr(run, "_freesurfer_ready", lambda tcsh=False: None)
    run.main(["--zips", "unused.zip", "--work-dir", str(work), "--long", "--device", "cpu"])
    assert seen == [("1", ["A_T1w.nii.gz", "B_T1w.nii.gz"], ["A", "B"], "fastsurfer_long"),
                    ("2", ["C_T1w.nii.gz", "D_T1w.nii.gz"], ["C", "D"], "fastsurfer_long")]


def _flair_phantom():
    """T1 head (2 mm) and a same-session 2D FLAIR: FLAIR contrast, 5 mm slices covering only the upper head, and an
    8 degree / 7 mm head movement between the scans. Returns (flair_sitk, t1_img, mask_img, true FLAIR->T1 transform)."""
    import SimpleITK as sitk
    from pie.imaging.dwi import _sitk_from_nib
    shape = (80, 96, 90)
    x, y, z = np.meshgrid(*[(np.arange(n) - n / 2) * 2.0 for n in shape], indexing="ij")
    r = np.sqrt((x / 70) ** 2 + (y / 88) ** 2 + (z / 80) ** 2)
    wm = np.sqrt((x / 45) ** 2 + (y / 60) ** 2 + ((z - 10) / 45) ** 2) < 1
    vent = np.sqrt((x / 8) ** 2 + (y / 25) ** 2 + ((z - 10) / 12) ** 2) < 1
    brain, scalp = r < 0.85, (r > 0.93) & (r < 1.0)
    eyes = np.sqrt(((np.abs(x) - 30) / 12) ** 2 + ((y - 70) / 12) ** 2 + ((z + 25) / 12) ** 2) < 1
    t1 = np.where(scalp | eyes, 0.9, 0.0) + np.where(brain, 0.5, 0) + np.where(wm, 0.3, 0) - np.where(vent, 0.6, 0)
    flair = np.where(scalp, 0.8, 0.0) + np.where(eyes, 0.2, 0) + np.where(brain, 0.6, 0) - np.where(wm, 0.15, 0) - np.where(vent, 0.5, 0)
    from scipy import ndimage
    texture = ndimage.gaussian_filter(np.random.default_rng(0).normal(size=shape), 2.0)       # smooth tissue texture
    texture /= np.abs(texture).max()
    t1 = ndimage.gaussian_filter(t1 * (1 + 0.3 * texture), 1.0)                            # and partial-volume blur
    flair = ndimage.gaussian_filter(flair * (1 + 0.3 * texture), 1.0)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -np.array(shape) + 1.0
    t1_img, mask_img = nib.Nifti1Image(t1.astype(np.float32), aff), nib.Nifti1Image(brain.astype(np.uint8), aff)
    true = sitk.Euler3DTransform((0.0, 0.0, 0.0), np.deg2rad(8), 0.0, 0.0, (0.0, 4.0, 6.0))      # FLAIR point -> T1 point
    grid = sitk.Image([80, 96, 18], sitk.sitkFloat32)
    grid.SetSpacing((2.0, 2.0, 5.0))
    grid.SetOrigin((79.0, 95.0, -5.0))                         # LPS; covers z = -5 .. +80 mm: the upper head only
    grid.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    flair_sitk = sitk.Resample(_sitk_from_nib(nib.Nifti1Image(flair.astype(np.float32), aff)), grid, true, sitk.sitkLinear, 0.0)
    return flair_sitk, t1_img, mask_img, true


def test_flair_registration_recovers_partial_coverage_2d_flair():
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)       # as flair.process_subject runs it; accuracy varies with threads
    flair_sitk, t1_img, mask_img, true = _flair_phantom()
    tx, metric = flair.register_flair_to_t1(flair_sitk, t1_img, mask_img)
    inv = true.GetInverse()
    pts = [(x, y, z) for x in (-40.0, 0.0, 40.0) for y in (-50.0, 0.0, 50.0) for z in (0.0, 30.0, 60.0)]
    err = [np.linalg.norm(np.subtract(tx.TransformPoint(p), inv.TransformPoint(p))) for p in pts]
    assert np.median(err) < 4.0 and max(err) < 8.0 and metric < 0          # old code: 18 mm; new 2.3-3.4 mm by threads


def test_freesurfer_stages_stop_once_when_freesurfer_is_not_set_up(tmp_path, monkeypatch):
    from pie.imaging import freesurfer
    work = tmp_path / "derived"
    (work / "fastsurfer" / "S1" / "stats").mkdir(parents=True)
    (work / "fastsurfer" / "S1" / "stats" / "aseg+DKT.stats").write_text("# Measure Mask, MaskVol, Mask Volume, 1.0, mm^3\n")
    pd.DataFrame({"patno": [1], "image_id": ["S1"], "session_date": ["2000-01-01"], "EVENT_ID": ["BL"], "protocol_phase": [1]})\
        .to_csv(work / "sessions.csv", index=False)
    monkeypatch.setattr(freesurfer, "HOME", None)
    monkeypatch.setattr(run, "talairach_etiv", lambda *a, **k: pytest.fail("must not reach any subject"))
    for flag in ("--etiv", "--long"):
        with pytest.raises(SystemExit, match="PIE_FREESURFER_HOME"):
            run.main(["--zips", "unused.zip", "--work-dir", str(work), flag])
    assert not (work / "fastsurfer_idps.csv").exists()                  # no table rewritten with an all-NaN eTIV column


def test_sdc_keeps_its_transforms_in_the_callers_folder_and_cleans_up_on_failure(tmp_path, monkeypatch):
    import ants
    monkeypatch.setattr(ants, "registration", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("registration failed")))
    img = nib.Nifti1Image(np.ones((8, 8, 8), np.float32), np.eye(4))
    with pytest.raises(RuntimeError, match="registration failed"):
        dwi.register_b0_to_t1_sdc(img, img, img, pe_axis="j", work_dir=tmp_path)
    assert not list(tmp_path.iterdir())                                   # nothing left behind in the caller's folder


def test_eddy_cpu_is_multithreaded_and_a_failed_eddy_leaves_no_working_folder(tmp_path, monkeypatch):
    from pie.imaging.dwi_correction import build_eddy_command
    meta = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05}
    assert "--nthr=8" in build_eddy_command("/fsl/bin", meta, 60, raw_is_uncorrected=True, cuda=False, nthr=8)[0]
    fsl = tmp_path / "fsl" / "bin"
    fsl.mkdir(parents=True)
    (fsl / "topup").write_text("#!/bin/sh\nfor a in \"$@\"; do case $a in --out=*) o=${a#--out=};; esac; done\n"
                               "cp b0_pair.nii.gz ${o}_fieldcoef.nii.gz\n")
    (fsl / "eddy_cpu").write_text("#!/bin/sh\necho \"$@\" > ../eddy_args.txt\nexit 1\n")
    for f in fsl.iterdir():
        f.chmod(0o755)
    monkeypatch.setattr(dwi, "FSLDIR", str(tmp_path / "fsl"))
    ds = {"data": np.ones((4, 4, 4, 4), np.float32), "affine": np.eye(4), "bvals": np.array([0., 0., 1000., 1000.]),
          "bvecs": np.c_[np.zeros((3, 2)), np.eye(3)[:, [0, 1]]], "meta": meta,
          "rev_b0": np.ones((4, 4, 4, 1), np.float32), "rev_meta": {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05}}
    with pytest.raises(RuntimeError, match="eddy failed"):
        dwi.eddy_correct(ds, tmp_path / "work", cuda=False)
    assert not (tmp_path / "work" / "eddy").exists()
    assert "--nthr=1" not in (tmp_path / "work" / "eddy_args.txt").read_text()


def test_manual_roi_duplicates_are_averaged_per_scan_not_mixed(tmp_path):
    im = tmp_path / "PPMI/Imaging"
    im.mkdir(parents=True)
    rows = []
    for fa, e in ((.30, 1.0e-3), (.40, 0.4e-3)):                      # the same scan month read twice
        for m, v in (("FA", fa), ("E1", e), ("E2", e), ("E3", e)):
            rows.append({"PATNO": 1, "PAG_NAME": "DTIROI", "INFODT": "01/2011", "Measure": m, "Tissue": "SN",
                         **{f"ROI{i}": v for i in range(1, 7)}, "REF1": .6, "REF2": .6, "RUNDATE": "2015-01-01"})
    pd.DataFrame(rows).to_csv(im / "DTI_Regions_of_Interest_01Jan2020.csv", index=False)
    t = labels.ppmi_dti_roi_table(tmp_path / "PPMI").iloc[0]
    assert abs(t.sn_fa - 0.35) < 1e-9 and abs(t.sn_md - 0.7e-3) < 1e-12 and t.n_rows == 2


def test_empty_syn_cache_files_are_not_treated_as_finished(tmp_path):
    fwd = nm_template.syn_paths(tmp_path)["fwd"]
    fwd[0].parent.mkdir(parents=True)
    for f in fwd:
        f.touch()                                                        # left by a copy onto a full disk
    assert not nm_template.syn_cached(tmp_path)
    for f in fwd:
        f.write_bytes(b"x")
    assert nm_template.syn_cached(tmp_path)


def test_assembled_nm_features_carry_their_validation_status(tmp_path):
    _idps(tmp_path)
    (tmp_path / "nm").mkdir()
    pd.DataFrame({"patno": [1], "error": [""], "n_sn_l": [40], "n_sn_r": [40], "sn_slab_coverage": [1.], "repeat_motion_mm_max": [.5],
                  "nm_ref_l_sd": [1.], "nm_ref_l_mean": [10.], "nm_ref_r_sd": [1.], "nm_ref_r_mean": [10.], "manufacturer": ["GE"],
                  "voxel_mm": ["0.5x0.5x1.5"], "nm_sn_mean_cnr": [.1]}).to_csv(tmp_path / "nm" / "nm_features.csv", index=False)
    f = manifest.assemble_features(tmp_path).set_index("PATNO")
    assert f.loc[1, "nm_sn_mean_cnr"] == .1 and not f["nm_validated"].any() and not manifest.NM_VALIDATED


def test_biondetti_files_are_checked_against_pinned_hashes(tmp_path):
    from pie.imaging import atlases
    (tmp_path / "BND_ROI.nii.gz").write_bytes(b"not the atlas")
    with pytest.raises(ValueError, match="checksum"):
        atlases.biondetti_file("BND_ROI.nii.gz", cache_dir=tmp_path)


def test_nigral_bridge_qc_rejects_misplaced_masks():
    from pie.imaging import atlases
    cit = atlases.cit168_mni2009c()
    sn = np.asarray(cit.dataobj) == 7                      # neuromelanin-MRI nigra is the SNc (CIT168 SNr lies ventrolateral)
    lab = np.where(sn, 3, 0).astype(np.int16)
    lab[np.roll(sn, -12, axis=1) & ~sn] = 4                 # background 12 mm posterior, off the nigra
    good = atlases.nigral_bridge_qc(nib.Nifti1Image(lab, cit.affine))
    assert good["pass"] and good["centroid_mm_l"] < 1 and good["bnd_in_sn"] == 0
    bad = atlases.nigral_bridge_qc(nib.Nifti1Image(np.roll(lab, 6, axis=0), cit.affine))   # 6 mm to the right
    assert not bad["pass"] and bad["centroid_mm_l"] > 4


def test_assembly_reads_published_nm_features_under_the_slab_qc(tmp_path):
    _idps(tmp_path)
    (tmp_path / "nm").mkdir()
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "n_sn_l": [40, 40], "n_sn_r": [40, 40], "sn_slab_coverage": [1., 1.],
                  "repeat_motion_mm_max": [.5, 5.], "nm_ref_l_sd": [1., 1.], "nm_ref_l_mean": [10., 10.], "nm_ref_r_sd": [1., 1.],
                  "nm_ref_r_mean": [10., 10.], "manufacturer": ["GE", "GE"], "voxel_mm": ["0.5x0.5x1.5"] * 2,
                  "nm_sn_mean_cnr": [.1, .1]}).to_csv(tmp_path / "nm" / "nm_features.csv", index=False)
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "nmb_sensorimotor_mean_cnr": [.2, .3], "nmb_bnd_cov": [1., 1.]})\
        .to_csv(tmp_path / "nm" / "nm_published_features.csv", index=False)
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "nmt_sn_mean_cnr": [.2, .3], "nmt_sn_cov_l": [1., 1.], "nmt_sn_cov_r": [1., 1.],
                  "nmt_crus_cv_l": [.1, .1], "nmt_crus_cv_r": [.1, .1]}).to_csv(tmp_path / "nm" / "nm_template_features.csv", index=False)
    f = manifest.assemble_features(tmp_path).set_index("PATNO")
    assert f.loc[1, "nmb_sensorimotor_mean_cnr"] == .2 and "nmb_bnd_cov" not in f
    assert np.isnan(f.loc[2, "nmb_sensorimotor_mean_cnr"]) and np.isnan(f.loc[2, "nmt_sn_mean_cnr"])   # 5 mm motion: slab fails
    assert "nmb_sensorimotor_mean_cnr" in manifest.feature_blocks(f.columns)["nm"]


def test_interrupted_syn_cache_is_not_taken_as_cached(tmp_path, monkeypatch):
    import types
    warp, aff = tmp_path / "w.nii.gz", tmp_path / "a.mat"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 1, 3), np.float32), np.eye(4)), warp)
    aff.write_text("affine")
    fake = types.SimpleNamespace(image_read=lambda p: 1.0, threshold_image=lambda *a: 1.0,
                                 registration=lambda **k: {"fwdtransforms": [str(warp), str(aff)], "invtransforms": []})
    monkeypatch.setitem(sys.modules, "ants", fake)
    monkeypatch.setattr(nm_template, "mni_brain_path", lambda: "mni.nii.gz")
    monkeypatch.setattr(nm_template, "crop_warp", lambda p: (_ for _ in ()).throw(KeyboardInterrupt))   # killed mid-crop
    with pytest.raises(KeyboardInterrupt):
        nm_template.syn_cache(tmp_path)
    assert not nm_template.syn_cached(tmp_path)


def test_freesurfer_run_timeout_kills_the_whole_process_tree(tmp_path):
    import time
    from pie.imaging import freesurfer
    script = tmp_path / "parent.sh"
    script.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {tmp_path}/child.pid\nwait\n")     # recon-all style: work in children
    script.chmod(0o755)
    with pytest.raises(subprocess.TimeoutExpired):
        freesurfer.run([script], tmp_path / "log", tmp_path, env=dict(os.environ), timeout=1)
    pid = int((tmp_path / "child.pid").read_text())
    def gone():
        try:
            return "State:\tZ" in Path(f"/proc/{pid}/status").read_text()     # a zombie is dead, awaiting its reaper
        except OSError:                                                   # vanished (ENOENT) or vanishing (ESRCH) mid-read
            return True

    for _ in range(40):
        if gone():
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        pytest.fail("the child outlived the timeout")


def test_idp_table_rewrite_is_atomic(tmp_path, monkeypatch):
    work = tmp_path / "derived"
    (work / "fastsurfer" / "S1" / "stats").mkdir(parents=True)
    (work / "fastsurfer" / "S1" / "stats" / "aseg+DKT.stats").write_text("# Measure Mask, MaskVol, Mask Volume, 1500000.0, mm^3\n")
    pd.DataFrame({"patno": [1], "image_id": ["S1"], "session_date": ["2000-01-01"], "EVENT_ID": ["BL"],
                  "protocol_phase": [1]}).to_csv(work / "sessions.csv", index=False)
    (work / "fastsurfer_idps.csv").write_text("the previous table\n")
    real = pd.DataFrame.to_csv

    def disk_full(self, path=None, *a, **k):
        if "fastsurfer_idps" in str(path):
            Path(path).write_text("PATNO,IMA")
            raise OSError(28, "No space left on device")
        return real(self, path, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", disk_full)
    with pytest.raises(OSError):
        run.main(["--zips", "unused.zip", "--work-dir", str(work), "--features-only"])
    assert (work / "fastsurfer_idps.csv").read_text() == "the previous table\n"


def test_sdc_records_how_oblique_the_phase_encoding_axis_is():
    assert dwi._pe_physical_axis(np.diag([2., 2., 2., 1.]), "j-") == (1, 0.0)
    c, s_ = np.cos(np.radians(20)), np.sin(np.radians(20))
    tilted = np.array([[2., 0, 0, 0], [0, 2 * c, -2 * s_, 0], [0, 2 * s_, 2 * c, 0], [0, 0, 0, 1]])   # 20 degrees about x
    axis, deg = dwi._pe_physical_axis(tilted, "j")
    assert axis == 1 and abs(deg - 20) < 1e-6


def test_registration_seeds_reach_ants_registration(monkeypatch):
    import importlib
    reg_module = importlib.import_module("ants.registration.registration")
    real, seen = reg_module.get_lib_fn, []

    def spy(name):
        fn = real(name)
        return (lambda args: (seen.append(list(map(str, args))), fn(args))[1]) if name == "antsRegistration" else fn

    monkeypatch.setattr(reg_module, "get_lib_fn", spy)
    img = np.zeros((24, 24, 24), np.float32)
    img[6:18, 5:19, 7:17] = 1
    head = nib.Nifti1Image(img, np.eye(4))
    features.tiv_from_registration(head, head, head, seed=7)      # ANTsPy 0.6 swallows random_seed= in **kwargs
    assert seen and all("--random-seed" in a and a[a.index("--random-seed") + 1] == "7" for a in seen)


def test_age_is_in_julian_years():
    cov = pd.DataFrame({"PATNO": [1], "BIRTHDT": pd.to_datetime(["1950-01-01"])})
    assert abs(labels._age_at(cov, pd.Series([1]), pd.Series(["2020-01-01"]))[0] - 25567 / 365.25) < 1e-9


def test_dat_control_norm_counts_each_control_once(tmp_path):
    ppmi, sessions = _ppmi(tmp_path)
    im = ppmi / "Imaging"
    q, v = pd.read_csv(im / "Xing_Core_Lab_-_Quant_SBR_01Jan2020.csv"), pd.read_csv(im / "Xing_Core_Lab_-_Visual_Read_01Jan2020.csv")
    low = {c: 1.8 for c in q if c.startswith("PUTAMEN")}                 # one control re-scanned nine times, all low
    pd.concat([q] + [q[q.PATNO == 1].assign(DATSCAN_DATE=f"0{m}/2012", EVENT_ID=f"V0{m}", **low) for m in range(1, 10)])\
        .to_csv(im / "Xing_Core_Lab_-_Quant_SBR_01Jan2020.csv", index=False)
    pd.concat([v] + [v[v.PATNO == 1].assign(DATSCAN_DATE=f"0{m}/2012") for m in range(1, 10)])\
        .to_csv(im / "Xing_Core_Lab_-_Visual_Read_01Jan2020.csv", index=False)
    d = labels.dat_labels(ppmi, sessions).set_index("PATNO")
    assert abs(d.loc[21, "sbr_pct_expected"] - 0.9 / 2.8) < 0.02         # the repeat scans must not drag the norm down


def test_acquisition_batches_carry_the_protocol(tmp_path):
    pd.DataFrame({"PATNO": [1, 2], "IMAGEID": ["I1", "I2"], "SCAN_DATE": ["2022-01-01"] * 2, "vol_Left_Putamen": [1., 1.]})\
        .to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    (tmp_path / "dwi").mkdir(), (tmp_path / "nm").mkdir()
    pd.DataFrame({"patno": [1, 2], "motion_mm_max": [1., 1.], "n_sn_l": [4, 4], "n_sn_r": [4, 4], "fa_wm_median": [.4, .4],
                  "manufacturer": ["Siemens"] * 2, "shells": ["1000"] * 2, "fw_method": ["singleshell_prior"] * 2, "eddy": [False] * 2,
                  "voxel_mm": [2.0, 2.0], "n_volumes": [65, 33], "topup": [True, False]}).to_csv(tmp_path / "dwi" / "dwi_features.csv", index=False)
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "n_sn_l": [40, 40], "n_sn_r": [40, 40], "sn_slab_coverage": [1., 1.],
                  "repeat_motion_mm_max": [.5, .5], "nm_ref_l_sd": [1., 1.], "nm_ref_l_mean": [10., 10.], "nm_ref_r_sd": [1., 1.],
                  "nm_ref_r_mean": [10., 10.], "manufacturer": ["GE", "GE"], "voxel_mm": ["0.5x0.5x1.5"] * 2, "tr_s": [.6, .6],
                  "te_s": [.004, .003], "mt_flag": [1, 1], "model": ["MR750"] * 2, "flip_angle": [40, 40]})\
        .to_csv(tmp_path / "nm" / "nm_features.csv", index=False)
    m = manifest.build_manifest(tmp_path).set_index("PATNO")
    assert m["nm_acquisition_batch"].nunique() == 2 and m["nm_batch"].nunique() == 1       # TE differs; vendor/voxel batch kept
    assert m["dwi_acquisition_batch"].nunique() == 2 and "65" in m.loc[1, "dwi_acquisition_batch"]
    assert m.loc[1, "dwi_correction"] == "topup" and m.loc[2, "dwi_correction"] == "rigid"


def test_nonphysical_tensor_fits_are_rejected_not_clipped():
    from dipy.core.sphere import HemiSphere, disperse_charges
    rng = np.random.default_rng(0)
    hs, _ = disperse_charges(HemiSphere(theta=np.arccos(rng.uniform(-1, 1, 30)), phi=rng.uniform(0, 2 * np.pi, 30)), 500)
    v, b = np.vstack([[0, 0, 0], hs.vertices]).T, np.r_[0, np.full(30, 1000.)]
    sig = lambda D: 1000 * np.exp(-b * np.einsum("in,ij,jn->n", v, D, v))
    data = np.zeros((2, 1, 1, 31), np.float32)
    data[0, 0, 0] = sig(np.diag([1.5e-3, 0.4e-3, 0.3e-3]))              # physical, FA 0.67
    data[1, 0, 0] = sig(np.diag([1.5e-3, -0.4e-3, 0.3e-3]))             # negative diffusivity: dipy clips it to FA 0.90
    maps = dwi.fit_models({"bvals": b, "bvecs": v, "data": data, "mask": np.ones((2, 1, 1), bool)})
    assert abs(maps["md"][0, 0, 0] - 0.733e-3) < 1e-5 and np.isnan(maps["fa"][1, 0, 0]) and np.isnan(maps["md"][1, 0, 0])
    assert maps["tensor_physical"][0, 0, 0] and not maps["tensor_physical"][1, 0, 0]


def test_dwi_qc_is_the_study_rule(tmp_path):
    _idps(tmp_path)
    (tmp_path / "dwi").mkdir()
    ok = {"n_sn_l": 30, "n_sn_r": 30, "sn_brain_mask_fraction": 1.0, "sn_physical_fraction": 0.95, "motion_mm_mean": 0.5,
          "motion_mm_max": 1.0, "fa_wm_median": 0.4, "sn_posterior_l_fa": 0.5, "sn_posterior_r_fa": 0.5}
    rows = [dict(ok), dict(ok, n_sn_l=4, n_sn_r=4), dict(ok, sn_physical_fraction=0.7), dict(ok, sn_brain_mask_fraction=0.9),
            dict(ok, motion_mm_mean=2.5), dict(ok, sn_posterior_r_fa=np.nan)]
    d = pd.DataFrame(rows).assign(patno=range(1, 7))
    assert manifest.QC["dwi"](d).tolist() == [True, False, False, False, False, False]


def test_dat_calibration_is_per_vendor_with_a_pooled_fallback():
    from pie.imaging import datscan
    rng = np.random.default_rng(0)
    true = rng.uniform(0.5, 3.0, 40)
    d = pd.DataFrame({"vendor": ["GE"] * 30 + ["PHILIPS"] * 10, "published": true})
    d["sbrwm_putamen_l"] = np.where(d.vendor == "GE", 0.5 * true + 0.2, 0.8 * true)
    d.loc[[5, 35], "published"] = np.nan                                       # e.g. prodromal: PPMI withholds the value
    out = datscan.calibrate(d, {"sbrwm_putamen_l": "published"}, min_ref=15)
    ge = out.vendor == "GE"
    assert np.allclose(out.loc[ge, "sbrwm_putamen_l_cal"], true[ge.to_numpy()])  # includes the row without a published value
    assert (out.loc[ge, "calibration"] == "GE").all() and (out.loc[~ge, "calibration"] == "POOLED").all()   # 9 < 15 reference scans


def test_dat_labels_fall_back_to_invicro_for_scans_xing_never_analysed(tmp_path):
    ppmi, sessions = _ppmi(tmp_path)
    f = ppmi / "Imaging" / "Xing_Core_Lab_-_Quant_SBR_01Jan2020.csv"
    x = pd.read_csv(f)
    x[x.PATNO != 21].to_csv(f, index=False)                      # a PPMI-1 participant only Invicro analysed
    d = labels.dat_labels(ppmi, sessions).set_index("PATNO")
    assert d.loc[21, "sbr_source"] == "invicro_occipital" and d.loc[21, "dat_deficit_sbr"] == 1 and d.loc[21, "dat_visual"] == 1
    assert d.loc[1, "sbr_source"] == "xing_cwm" and d.index.is_unique  # Xing stays primary; one row per session


def test_dat_labels_keep_every_participant_even_when_image_ids_repeat(tmp_path):
    ppmi, sessions = _ppmi(tmp_path)
    d = labels.dat_labels(ppmi, sessions.assign(image_id="x"))                # callers without real image IDs
    assert d["PATNO"].nunique() == 21                                          # 22 minus the TRODAT scan


def test_assembly_reads_native_nm_measures_under_the_slab_qc(tmp_path):
    _idps(tmp_path)
    (tmp_path / "nm").mkdir()
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "n_sn_l": [40, 40], "n_sn_r": [40, 40], "sn_slab_coverage": [1., 1.],
                  "repeat_motion_mm_max": [.5, 5.], "nm_ref_l_sd": [1., 1.], "nm_ref_l_mean": [10., 10.], "nm_ref_r_sd": [1., 1.],
                  "nm_ref_r_mean": [10., 10.], "manufacturer": ["SIEMENS"] * 2, "voxel_mm": ["0.5x0.5x2.0"] * 2,
                  "nm_sn_mean_cnr": [.1, .1]}).to_csv(tmp_path / "nm" / "nm_features.csv", index=False)
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "nml_sn_volume_mm3": [400., 300.], "nml_threshold": [1500., 1500.],
                  "nms_sn_mean_cr": [.18, .15], "nms_sn_volume_mm3": [600., 500.], "native_cov_search_l": [1., 1.]})\
        .to_csv(tmp_path / "nm" / "nm_native_features.csv", index=False)
    f = manifest.assemble_features(tmp_path).set_index("PATNO")
    assert f.loc[1, "nml_sn_volume_mm3"] == 400 and f.loc[1, "nms_sn_mean_cr"] == .18 and "nml_threshold" not in f
    assert np.isnan(f.loc[2, "nml_sn_volume_mm3"]) and np.isnan(f.loc[2, "nms_sn_mean_cr"])      # 5 mm motion: slab fails
    blocks = manifest.feature_blocks(f.columns)["nm"]
    assert {"nml_sn_volume_mm3", "nms_sn_mean_cr", "nms_sn_volume_mm3"} <= set(blocks)


def test_series_indexes_skip_phantom_scans_with_non_numeric_subject_ids(tmp_path):
    import zipfile

    from pie.imaging import batch, index

    zp = tmp_path / "loni.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("PPMI/1/MPRAGE/2000-01-01_09_00_00.0/I000001/a.dcm", b"x")
        z.writestr("PPMI/00000JAN00/MPRAGE/2011-04-27_09_00_00.0/I000002/a.dcm", b"x")   # LONI phantom subject
    assert batch.index_series([zp])["patno"].tolist() == [1]
    assert index.index_zips([zp])["patno"].tolist() == [1]


def test_ida_metadata_reads_namespaced_records_and_skips_phantoms(tmp_path):
    import zipfile

    from pie.imaging.index import read_ida_metadata

    def xml(subject, image):
        return (f'<?xml version="1.0"?><idaxs xmlns="http://ida.loni.usc.edu"><project xmlns=""><subject>'
                f'<subjectIdentifier>{subject}</subjectIdentifier><researchGroup>PD</researchGroup>'
                f'<visit><visitIdentifier>Baseline</visitIdentifier></visit><study><subjectAge>60.0</subjectAge>'
                f'<series><dateAcquired>2000-01-01</dateAcquired></series><imagingProtocol><imageUID>{image}</imageUID>'
                f'<description>MPRAGE</description><protocolTerm><protocol term="Manufacturer">SIEMENS</protocol>'
                f'</protocolTerm></imagingProtocol></study></subject></project></idaxs>')

    zp = tmp_path / "meta.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("PPMI/PPMI_1_MPRAGE_S1_I000001.xml", xml("1", "000001"))
        z.writestr("PPMI/PPMI_00000JAN00_MPRAGE_S2_I000002.xml", xml("00000JAN00", "000002"))
    d = read_ida_metadata(zp)
    assert d["image_id"].tolist() == ["I000001"]
    assert d[["patno", "ida_visit", "ida_manufacturer"]].iloc[0].tolist() == [1, "Baseline", "SIEMENS"]
