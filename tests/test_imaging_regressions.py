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
                  "ENROLL_AGE": [60.0] * 3}).to_csv(sc / "Participant_Status_2000.csv", index=False)
    pd.DataFrame({"PATNO": [1, 2, 3], "LAST_UPDATE": ["2000-01-01"] * 3, "SEX": [0, 1, 0], "BIRTHDT": ["01/1940"] * 3,
                  "HANDED": [1] * 3}).to_csv(sc / "Demographics_2000.csv", index=False)
    pd.DataFrame({"PATNO": [1, 2, 3], "LRRK2": [0, 1, None], "GBA": ["0", "N370S", "0"], "SNCA": [0, 0, 0],
                  "APOE": ["E3/E4", "E3/E3", None], "PATHVAR_COUNT": [0, 1, 0]}).to_csv(sc / "iu_genetic_consensus_2000.csv", index=False)
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


# ------------------------------------------------------------------------------------------ misc FSL scripts
def _stub(bindir, name, body):
    p = bindir / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(0o755)


def _bids(tmp_path):
    for s in ("sub-01", "sub-02"):
        (tmp_path / "data" / s / "anat").mkdir(parents=True)
        (tmp_path / "data" / s / "anat" / f"{s}_T1w.nii.gz").touch()
    b = tmp_path / "bin"
    b.mkdir()
    return b, {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "FSL_LOG": str(tmp_path / "calls.log")}


def test_run_first_reads_volumes_from_the_combined_segmentation(tmp_path):
    b, env = _bids(tmp_path)
    _stub(b, "run_first_all", 'while getopts "i:o:" f; do case $f in o) out=$OPTARG;; esac; done\n'
                              'touch "${out}_all_fast_firstseg.nii.gz"\n')
    _stub(b, "fslstats", 'echo "$@" >> "$FSL_LOG"\necho "10 12.5"\n')
    subprocess.run(["bash", str(ROOT / "misc" / "run_first.sh"), str(tmp_path / "data"), str(tmp_path / "out")],
                   env=env, cwd=tmp_path, check=True, capture_output=True)
    vols = pd.read_csv(tmp_path / "out" / "first_volumes.csv")
    assert len(vols) == 2 * 15 and vols["Volume_mm3"].eq(12.5).all()
    assert vols.set_index(["Subject", "Structure"]).loc[("sub-01", "L_Hipp"), "Label"] == 17
    assert "sub-01_first_all_fast_firstseg.nii.gz -l 16.5 -u 17.5 -V" in (tmp_path / "calls.log").read_text()


def test_run_sienax_runs_every_subject_with_the_bet_options(tmp_path):
    b, env = _bids(tmp_path)
    _stub(b, "sienax", 'echo "$@" >> "$FSL_LOG"\nwhile [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done\n'
                       'mkdir -p "$out" && echo "BRAIN 1 2" > "$out/report.sienax"\n')
    subprocess.run(["bash", str(ROOT / "misc" / "run_sienax.sh"), str(tmp_path / "data"), str(tmp_path / "out")],
                   env=env, cwd=tmp_path, check=True, capture_output=True)
    calls = (tmp_path / "calls.log").read_text().splitlines()
    assert len(calls) == 2 and all("-B -f 0.2 -g 0.02" in c for c in calls)
    assert (tmp_path / "out" / "sub-02" / "report.sienax").exists()
