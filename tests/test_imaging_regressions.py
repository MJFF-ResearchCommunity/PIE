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
from pie.imaging import (cnn, datscan, dwi, dwi_refine, embed, fba, flair, labels, manifest, nm, nm_template, qc,
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
    pd.DataFrame({"patno": [1, 2], "motion_mm_max": [1., 1.], "n_sn_l": [4, 4], "n_sn_r": [4, 4], "fa_wm_median": [.4, .4],
                  "manufacturer": ["Siemens"] * 2, "shells": ["1000", "700 1000 2000"], "fw_method": ["singleshell_prior", "multishell_nls"],
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
