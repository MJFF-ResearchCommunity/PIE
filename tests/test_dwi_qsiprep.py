"""QSIPrep/QSIRecon comparator contracts on synthetic data: no containers, no participant data."""

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging import dwi
from pie.imaging import dwi_qsiprep as dq
from pie.imaging.dwi_tensor_qc import tensor_measurements

AFF = np.diag([2.0, 2.0, 2.0, 1.0])


def _dirs(n, seed):
    g = np.random.default_rng(seed).normal(size=(3, n))
    return g / np.linalg.norm(g, axis=0)


def _run(tmp, name, bvals, meta, shape=(8, 8, 6)):
    b = np.asarray(bvals, float)
    nib.save(nib.Nifti1Image(np.random.default_rng(len(b)).random(shape + (len(b),)).astype(np.float32) + 1, AFF), tmp / f"{name}.nii.gz")
    np.savetxt(tmp / f"{name}.bval", b[None])
    v = _dirs(len(b), 1)
    v[:, b <= 50] = 0
    np.savetxt(tmp / f"{name}.bvec", v)
    (tmp / f"{name}.json").write_text(json.dumps(meta))
    return tuple(str(tmp / f"{name}{e}") for e in (".nii.gz", ".bval", ".bvec", ".json"))


def test_export_writes_qsiprep_bids_from_pies_selection(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    pe = {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05, "EchoTime": 0.09, "PartialFourier": 0.75}
    runs = {"p/b700/": [_run(src, "b700", [0] + [700] * 12, pe)], "p/b1000/": [_run(src, "b1000", [0] + [1000] * 12, pe)],
            "p/rev/": [_run(src, "rev", [0, 0], dict(pe, PhaseEncodingDirection="j-"))],
            "p/lr/": [_run(src, "lr", [0] + [1000] * 8, dict(pe, PhaseEncodingDirection="i"))]}   # not PIE's choice
    calls = []
    monkeypatch.setattr(dwi, "convert", lambda zip_path, prefix, out_dir, archive_cache=None: calls.append(prefix) or runs[prefix])
    derived = tmp_path / "derived"
    (derived / "nifti" / "1").mkdir(parents=True)
    for image_id, value in (("T1A", 1.0), ("T1B", 2.0)):
        nib.save(nib.Nifti1Image(np.full((6, 6, 6), value, np.float32), np.eye(4)), derived / "nifti" / "1" / f"{image_id}_T1w.nii.gz")
        (derived / "nifti" / "1" / f"{image_id}_T1w.json").write_text("{}")
    pd.DataFrame({"zip": ["z", "z"], "patno": [1, 1], "image_id": ["T1B", "T1A"], "member_prefix": ["q/", "p/"],
                  "session_date": ["2019-01-01", "2020-01-02"]}).to_csv(derived / "sessions.csv", index=False)
    rows = [{"zip": "z", "prefix": k, "date": "2020-01-02", "desc": "DTI"} for k in runs]
    bids = tmp_path / "bids"
    rec = dq.export_subject(bids, 1, rows, derived / "sessions.csv", tmp_path / "work")

    ses, base = bids / "sub-1" / "ses-20200102", "sub-1_ses-20200102"
    assert sorted(str(p.relative_to(ses)) for p in ses.rglob("*") if p.is_file()) == sorted(
        [f"anat/{base}_T1w.nii.gz", f"anat/{base}_T1w.json", f"fmap/{base}_dir-reverse_epi.nii.gz", f"fmap/{base}_dir-reverse_epi.json"]
        + [f"dwi/{base}_run-{k}_dwi{e}" for k in (1, 2) for e in (".nii.gz", ".bval", ".bvec", ".json")])
    assert np.loadtxt(ses / "dwi" / f"{base}_run-1_dwi.bval").max() == 700        # PIE's order: b700 then b1000
    assert (ses / "dwi" / f"{base}_run-2_dwi.nii.gz").read_bytes() == Path(runs["p/b1000/"][0][0]).read_bytes()   # raw copy
    meta = json.loads((ses / "dwi" / f"{base}_run-1_dwi.json").read_text())
    assert meta["PhaseEncodingDirection"] == "j" and meta["TotalReadoutTime"] == 0.05
    fmap = json.loads((ses / "fmap" / f"{base}_dir-reverse_epi.json").read_text())
    assert fmap["PhaseEncodingDirection"] == "j-" and fmap["IntendedFor"] == [f"ses-20200102/dwi/{base}_run-{k}_dwi.nii.gz" for k in (1, 2)]
    epi = nib.load(ses / "fmap" / f"{base}_dir-reverse_epi.nii.gz")
    assert np.allclose(epi.get_fdata(), nib.load(runs["p/rev/"][0][0]).get_fdata()) and np.allclose(epi.affine, AFF)
    assert nib.load(ses / "anat" / f"{base}_T1w.nii.gz").get_fdata().max() == 1.0          # the same-session T1
    assert rec["t1_gap_days"] == 0 and rec["output_resolution"] == 2.0 and rec["unringing_method"] == "rpg"
    assert rec["pe_known"] and rec["reverse_pe_fmap"] and rec["shells"] == [700, 1000]
    assert json.loads((bids / "sourcedata" / "sub-1.json").read_text()) == rec and (bids / "dataset_description.json").is_file()

    assert dq.export_subject(bids, 1, rows, derived / "sessions.csv", tmp_path / "work") == rec and len(calls) == 4   # reused
    (ses / "dwi" / f"{base}_run-1_dwi.bval").write_text("0\n")
    with pytest.raises(ValueError, match="differs"):
        dq.export_subject(bids, 1, rows, derived / "sessions.csv", tmp_path / "work")
    (bids / "sourcedata" / "sub-1.json").unlink()
    with pytest.raises(FileExistsError):
        dq.export_subject(bids, 1, rows, derived / "sessions.csv", tmp_path / "work")


def test_bids_sidecar_keeps_only_phase_encoding_pie_knows():
    siemens = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05, "EffectiveEchoSpacing": 5e-4}
    assert dq.bids_sidecar(siemens) == siemens
    philips = {"PhaseEncodingAxis": "j", "EstimatedTotalReadoutTime": 0.04, "WaterFatShift": 10.0, "SeriesDescription": "DTI_LR"}
    named = dq.bids_sidecar(philips)        # polarity from the PPMI series name, as dwi.assemble uses it
    assert named["PhaseEncodingDirection"] == "j-" and named["TotalReadoutTime"] == 0.04 and named["pe_from_description"]
    unknown = dq.bids_sidecar(dict(philips, SeriesDescription="AX DTI", EffectiveEchoSpacing=5e-4))
    assert not set(dq.GUESSABLE) & set(unknown) and unknown["PhaseEncodingAxis"] == "j"
    assert "PhaseEncodingDirection" not in dq.bids_sidecar({"PhaseEncodingDirection": "j"})      # no readout time


def test_qsiprep_command_pins_image_and_ppmi_settings(tmp_path):
    lic = tmp_path / "license.txt"
    lic.write_text("x")
    bids, out, work = (tmp_path / d for d in ("bids", "out", "work"))
    rec = {"output_resolution": 2.0, "unringing_method": "rpg"}
    cmd = dq.qsiprep_command(bids, out, work, "1", rec, lic)
    flags = dict(zip(cmd, cmd[1:]))
    assert cmd[:5] == ["docker", "run", "--rm", "--pull", "never"] and "--gpus" not in cmd
    assert dq.QSIPREP_IMAGE in cmd and "26.0.0@sha256:" in dq.QSIPREP_IMAGE
    assert flags["--output-resolution"] == "2.0" and flags["--unringing-method"] == "rpg" and flags["--hmc-model"] == "eddy"
    assert flags["--pepolar-method"] == "TOPUP" and flags["--use-syn-sdc"] == "warn" and flags["--denoise-method"] == "dwidenoise"
    assert flags["--eddy-config"] == "/work/pie_eddy_params.json" and flags["--participant-label"] == "1"
    assert {f"{bids.resolve()}:/data:ro", f"{out.resolve()}:/out", f"{(work / 'sub-1').resolve()}:/work",
            f"{lic.resolve()}:/license.txt:ro"} <= set(cmd)
    assert "--gpus" in dq.qsiprep_command(bids, out, work, "1", rec, lic, cuda=True)
    with pytest.raises(ValueError, match="separate"):
        dq.qsiprep_command(bids, bids / "derivatives", work, "1", rec, lic)
    with pytest.raises(ValueError, match="license"):
        dq.qsiprep_command(bids, out, work, "1", rec, tmp_path / "missing.txt")


def test_qsirecon_command_and_spec(tmp_path):
    lic = tmp_path / "license.txt"
    lic.write_text("x")
    prep, out, work = (tmp_path / d for d in ("qsiprep", "recon", "work"))
    cmd = dq.qsirecon_command(prep, out, work, "1", lic)
    assert dict(zip(cmd, cmd[1:]))["--recon-spec"] == "/work/pie_recon_spec.yaml" and dq.QSIRECON_IMAGE in cmd
    assert f"{prep.resolve()}:/qsiprep:ro" in cmd and "26.0.1@sha256:" in dq.QSIRECON_IMAGE
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(json.dumps(dq.RECON_SPEC, indent=2))      # QSIRecon reads the JSON file as YAML
    assert spec == dq.RECON_SPEC and {n["software"]: n["action"] for n in spec["nodes"]} == {"TORTOISE": "estimate", "AMICO": "fit_noddi"}


def test_run_container_records_provenance_and_never_pulls(tmp_path, monkeypatch):
    out = tmp_path / "out"
    state = {"inspect": 1, "rc": 0, "version": "QSIPrep v26.0.0\n"}
    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        if args[1] == "image":
            return subprocess.CompletedProcess(args, state["inspect"], stdout='sha256:abc ["pennlinc/qsiprep@sha256:b57"]\n', stderr="")
        (out / "sub-1.html").write_text("report")
        return subprocess.CompletedProcess(args, state["rc"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: state["version"])
    cmd, files, work = ["docker", "run", "participant-1"], {"pie_eddy_params.json": dq.EDDY_PARAMS}, tmp_path / "work" / "sub-1"
    with pytest.raises(RuntimeError, match="docker pull"):
        dq.run_container(cmd, dq.QSIPREP_IMAGE, out, work, files)
    state.update(inspect=0, version="QSIPrep v26.0.01")
    with pytest.raises(RuntimeError, match="version"):
        dq.run_container(cmd, dq.QSIPREP_IMAGE, out, work, files)
    state["version"] = "QSIPrep v26.0.0\n"
    record = dq.run_container(cmd, dq.QSIPREP_IMAGE, out, work, files, expected=[out / "sub-1.html"])
    config = json.loads(next(out.rglob("configuration.json")).read_text())
    assert config["argv"] == cmd and config["version"] == "QSIPrep v26.0.0" and config["image"] == dq.QSIPREP_IMAGE
    assert config["image_inspect"].startswith("sha256:abc") and json.loads((work / "pie_eddy_params.json").read_text()) == dq.EDDY_PARAMS
    assert record["returncode"] == 0 and record["scientific_qc_pass"] is False
    runs = sum(c[1] == "run" for c in calls)
    assert dq.run_container(cmd, dq.QSIPREP_IMAGE, out, work, files) == record and sum(c[1] == "run" for c in calls) == runs
    with pytest.raises(ValueError, match="different"):
        dq.run_container(cmd + ["--nprocs", "2"], dq.QSIPREP_IMAGE, out, work, files)
    state["rc"] = 2
    with pytest.raises(RuntimeError, match="failed"):
        dq.run_container(cmd, dq.QSIPREP_IMAGE, out, tmp_path / "work" / "sub-2", {})
    assert len(list(out.rglob("execution_completed.json"))) == 1


def test_recon_maps_and_run_bounds_parse_qsirecon_and_qsiprep_outputs(tmp_path):
    def touch(rel):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        return p

    d = "derivatives/qsirecon-TORTOISE/sub-1/ses-20200102/dwi/sub-1_ses-20200102_space-"
    fa, md = touch(d + "ACPC_model-tensor_param-fa_dwimap.nii.gz"), touch(d + "ACPC_model-tensor_param-md_dwimap.nii.gz")
    touch(d + "ACPC_model-tensor_param-li_dwimap.nii.gz")                       # not compared
    touch(d + "MNI152NLin2009cAsym_model-tensor_param-fa_dwimap.nii.gz")         # template space
    iso = touch("derivatives/qsirecon-NODDI/sub-1/dwi/sub-1_space-ACPC_model-noddi_param-isovf_desc-x_dwimap.nii.gz")
    touch("derivatives/qsirecon-NODDI/sub-10/dwi/sub-10_space-ACPC_model-noddi_param-isovf_dwimap.nii.gz")   # another participant
    assert dq.recon_maps(tmp_path, "1") == {("tensor", "fa"): fa, ("tensor", "md"): md, ("noddi", "isovf"): iso}
    touch("derivatives/qsirecon-NODDI/sub-1/dwi/sub-1_space-ACPC_model-tensor_param-fa_dwimap.nii.gz")
    with pytest.raises(ValueError, match="two"):
        dq.recon_maps(tmp_path, "1")
    assert dq.run_bounds(["a"] * 3 + ["b"] * 2) == [(0, 3), (3, 5)]
    with pytest.raises(ValueError, match="contiguous"):
        dq.run_bounds(["a", "b", "a"])


def _codes(shape):
    codes = np.zeros(shape, np.int16)       # 10 * CIT168 label (7 SNc, 9 SNr) + side (1 left, 2 right)
    codes[5:7, 5:9, 7:9], codes[5:7, 9:11, 7:9] = 71, 91
    codes[9:11, 5:9, 7:9], codes[9:11, 9:11, 7:9] = 72, 92
    return codes


def test_shared_roi_rule_reproduces_study_features():
    shape, b = (16, 16, 16), np.r_[0.0, 0.0, np.full(15, 1000.0)]
    v = np.c_[np.zeros((3, 2)), _dirs(15, 3)]
    md = 0.5e-3 + 0.05e-3 * np.indices(shape)[1]          # MD rises along y, so the SN halves differ
    data = (1000 * np.exp(-b * md[..., None])).astype(np.float32)
    data[6, 6, 7, 3] = -1                                  # one non-physical voxel
    codes = _codes(shape)
    study = dwi.study_features(data, b, v, AFF, [(0, len(b))], codes, np.ones(shape, bool), 0.5)
    xyz, rois = dwi.study_rois(codes, AFF)                # the shared ROI split and region rule the comparator uses
    t = tensor_measurements(data[tuple(xyz.T)], b, v)
    mine = dwi.roi_means(np.where(t["accepted"], t["md"], np.nan), rois, "dwis_md")
    assert all(np.isclose(study[k], mine[k], equal_nan=True) for k in mine) and len(mine) == 15
    assert study["dwis_md_sn_posterior_l"] != study["dwis_md_sn_anterior_l"]


def test_compare_subject_on_synthetic_qsiprep_and_qsirecon_trees(tmp_path, monkeypatch):
    shape, D = (16, 16, 16), np.diag([1.5e-3, 0.4e-3, 0.4e-3])
    b, v, files = [], [], []
    for k, shell in enumerate((700, 1000, 2000), 1):
        b += [0] + [shell] * 15
        v.append(np.c_[np.zeros(3), _dirs(15, k)])
        files += [f"sub-1/ses-1/dwi/sub-1_ses-1_run-{k}_dwi.nii.gz"] * 16
    b, v = np.array(b, float), np.concatenate(v, axis=1)
    signal = 1000 * (0.8 * np.exp(-b * np.einsum("in,ij,jn->n", v, D, v)) + 0.2 * np.exp(-b * 3e-3))   # free water 0.2
    sphere = ((np.indices(shape) - 7.5) ** 2).sum(axis=0) < 49
    prep = tmp_path / "qsiprep" / "sub-1"
    (prep / "ses-1" / "dwi").mkdir(parents=True)
    (prep / "anat").mkdir()
    stem = prep / "ses-1" / "dwi" / "sub-1_ses-1_space-ACPC_desc-preproc_dwi"
    nib.save(nib.Nifti1Image(np.where(sphere[..., None], signal, 0).astype(np.float32), AFF), f"{stem}.nii.gz")
    np.savetxt(f"{stem}.bval", b[None])
    np.savetxt(f"{stem}.bvec", v)
    pd.DataFrame({"original_file": files, "framewise_displacement": [np.nan] + [0.2] * (len(b) - 1)}).to_csv(
        prep / "ses-1" / "dwi" / "sub-1_ses-1_desc-confounds_timeseries.tsv", sep="\t", index=False)
    nib.save(nib.Nifti1Image(sphere.astype(np.float32), AFF), prep / "anat" / "sub-1_space-ACPC_desc-preproc_T1w.nii.gz")
    nib.save(nib.Nifti1Image(sphere.astype(np.uint8), AFF), prep / "anat" / "sub-1_space-ACPC_desc-brain_mask.nii.gz")
    for suffix, model, param, value in (("TORTOISE", "tensor", "fa", 0.5), ("TORTOISE", "tensor", "md", 8e-4), ("NODDI", "noddi", "isovf", 0.3)):
        d = tmp_path / "recon" / "derivatives" / f"qsirecon-{suffix}" / "sub-1" / "ses-1" / "dwi"
        d.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.full(shape, value, np.float32), AFF), d / f"sub-1_ses-1_space-ACPC_model-{model}_param-{param}_dwimap.nii.gz")
    (tmp_path / "fs" / "mri").mkdir(parents=True)
    for name in ("orig.mgz", "mask.mgz"):
        nib.save(nib.MGHImage(np.ones((4, 4, 4), np.float32), np.eye(4)), tmp_path / "fs" / "mri" / name)
    seen = {}
    monkeypatch.setattr(dwi, "register_b0_to_t1", lambda moving, t1, mask, sampling_seed: seen.update(seed=sampling_seed, moving=moving) or ("tx", -0.6))
    monkeypatch.setattr(dwi, "study_nigra_labels", lambda grid, fs_dir, tx: seen.update(grid=grid, tx=tx) or _codes(shape))
    row = dq.compare_subject(1, tmp_path / "qsiprep", tmp_path / "fs", tmp_path / "recon")
    assert seen["seed"] == dwi.STUDY_SEED and seen["tx"] == "tx" and seen["grid"].shape == shape and np.allclose(seen["grid"].affine, AFF)
    assert np.asarray(seen["moving"].dataobj)[~sphere].max() == 0                 # brain-masked QSIPrep T1w
    assert row["dwiq_shell"] == 1000 and row["dwiq_n_sn"] == 48 and row["dwiq_motion_for_qc_mm"] == pytest.approx(0.2)
    assert row["dwiq_fw_sn_posterior_mean"] == pytest.approx(0.2, abs=0.01) and np.isfinite(row["dwiq_fa_sn_posterior_mean"])
    assert row["dwiq_tensor_fa_sn_l"] == pytest.approx(0.5) and row["dwiq_noddi_isovf_snr_mean"] == pytest.approx(0.3)
    assert "dwiq_qc_pass" in row and not any(k.startswith("dwis_") for k in row)


def test_agreement_stats_match_closed_forms():
    x, c = np.array([0.6, 0.7, 0.8, 0.75, 0.65, 0.9]), 0.05
    s = dq.agreement_stats(x, x + c)       # a pure offset: CCC uses population variances, ICC(2,1) sample variances
    assert s["ccc"] == pytest.approx(2 * x.var() / (2 * x.var() + c ** 2))
    assert s["icc21"] == pytest.approx(2 * x.var(ddof=1) / (2 * x.var(ddof=1) + c ** 2))
    assert s["ba_bias"] == pytest.approx(c) and s["ba_loa_low"] == pytest.approx(c) and s["ba_loa_high"] == pytest.approx(c)
    assert s["pearson_r"] == pytest.approx(1)
    same = dq.agreement_stats(x, x)
    assert same["ccc"] == pytest.approx(1) and same["icc21"] == pytest.approx(1)
    assert np.isnan(dq.agreement_stats(x[:2], x[:2])["ccc"])


def test_agreement_pairs_dwiq_with_dwis_under_both_qc_rules():
    x = np.random.default_rng(0).normal(0.8e-3, 0.1e-3, 8)
    pie = pd.DataFrame({"patno": range(8), "dwis_md_sn_posterior_mean": x, "dwis_qc_pass": [True] * 7 + [False]})
    qsi = pd.DataFrame({"patno": range(8), "dwiq_md_sn_posterior_mean": x + 1e-5, "dwiq_tensor_md_sn_posterior_mean": x * 1.1,
                        "dwiq_fw_sn_posterior_mean": 0.2, "dwiq_qc_pass": ["True"] * 8})
    out = dq.agreement(pie, qsi).set_index("qsiprep")
    assert set(out.index) == {"dwiq_md_sn_posterior_mean", "dwiq_tensor_md_sn_posterior_mean"}
    assert (out["pie"] == "dwis_md_sn_posterior_mean").all() and (out["n"] == 7).all()
    assert out.loc["dwiq_md_sn_posterior_mean", "ba_bias"] == pytest.approx(1e-5)
    with pytest.raises(ValueError, match="--study"):
        dq.agreement(pie.drop(columns="dwis_qc_pass"), qsi)
