"""The rs-fMRI validation run step on synthetic NIfTIs, with fMRIPrep and XCP-D replaced by fakes that write their
output layout: no container, no PPMI file."""
import json
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fmri_validation as rf  # noqa: E402
from pie.imaging.fmri import inspect_nifti  # noqa: E402
from test_xcpd import xcpd_tree  # noqa: E402


def run(folder, name, volumes, tr, pe="j", te=0.03, acq="120000.00"):
    folder.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.random.default_rng(0).normal(100, 1, (6, 6, 4, volumes)).astype(np.float32), np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header.set_zooms((1, 1, 1, tr))
    nii, js = folder / f"{name}.nii.gz", folder / f"{name}.json"
    nib.save(img, nii)
    js.write_text(json.dumps({"RepetitionTime": tr, "PhaseEncodingDirection": pe, "TotalReadoutTime": 0.05, "EchoTime": te,
                              "AcquisitionTime": acq, "Manufacturer": "Siemens", "ManufacturersModelName": "Prisma"}))
    return dict(inspect_nifti(nii, js), nifti=str(nii), sidecar=str(js), image_id="I" + name)


def t1(folder):
    folder.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.float32), np.eye(4)), folder / "t1.nii.gz")
    (folder / "t1.json").write_text("{}")
    return str(folder / "t1.nii.gz"), str(folder / "t1.json"), "sessions"


def fake_fmriprep(config):
    out, lab = Path(config.output_dir), config.participants[0]
    tr = json.loads(next(Path(config.bids_dir).rglob("*_bold.json")).read_text())["RepetitionTime"]
    func, anat = out / f"sub-{lab}" / "func", out / f"sub-{lab}" / "anat"
    func.mkdir(parents=True)
    anat.mkdir()
    (out / "dataset_description.json").write_text('{"Name": "fMRIPrep", "DatasetType": "derivative"}')
    stem = f"sub-{lab}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"
    (func / f"{stem}.json").write_text(json.dumps({"RepetitionTime": tr}))
    (func / f"{stem}.nii.gz").write_bytes(b"x")
    (anat / f"sub-{lab}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5").write_bytes(b"x")
    (out / f"sub-{lab}.html").write_text("report")


class FakeXCPD:
    def __init__(self, bold=True, fail=None):
        self.bold, self.fail, self.motion = bold, fail, []

    def __call__(self, config):
        self.motion.append(config.motion_filter)
        out, lab = Path(config.output_dir), config.participants[0]
        if self.fail:
            (out / "pie_provenance" / "x").mkdir(parents=True)
            (out / "pie_provenance" / "x" / "attempt-1.log").write_text(self.fail)
            raise RuntimeError("XCP-D failed (1)")
        root, _ = xcpd_tree(Path(tempfile.mkdtemp()))
        for p in sorted(root.rglob("*")):
            target = out / str(p.relative_to(root)).replace("A01", lab)
            if p.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif self.bold or "_bold.nii.gz" not in p.name:
                for t in {target, Path(str(target).replace("4S456", "4S156"))}:          # XCP-D writes both atlases
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, t)
        (out / f"sub-{lab}.html").write_text("report")


def setup(monkeypatch, tmp_path, outputs, stratum="P2_TR1", have_t1=True, fmriprep=fake_fmriprep, xcpd_run=None):
    xcpd_run = xcpd_run or FakeXCPD()
    monkeypatch.setattr(rf, "convert_session", lambda row, series, work: outputs)
    monkeypatch.setattr(rf, "find_t1", lambda row, series, sessions, work: t1(tmp_path / "anat") if have_t1 else None)
    monkeypatch.setattr(rf.fmriprep, "run_fmriprep", fmriprep)
    monkeypatch.setattr(rf.xcpd, "run_xcpd", xcpd_run)
    row = pd.Series({"patno": 1001, "date": "2022-03-04", "image_ids": "I1;I2", "stratum": stratum})
    a = Namespace(work=str(tmp_path / "work"), sessions="sessions.csv", paired_p1=False, apptainer="apptainer", nprocs=1,
                  omp=1, mem_mb=1000, license=None, fmriprep_sif="f.sif", xcpd_sif="x.sif", keep=None)
    return row, a, xcpd_run


def test_the_longest_rest_run_is_chosen_with_a_verified_reverse_pe_reference(tmp_path):
    long, short = run(tmp_path, "long", 130, 2.5), run(tmp_path, "short", 125, 2.5, acq="110000.00")
    ref, same_pe = run(tmp_path, "ref", 10, 2.5, pe="j-"), run(tmp_path, "same", 10, 2.5, pe="j")
    assert rf.choose([short, same_pe, long, ref], paired=True) == (long, ref)
    assert rf.choose([short, same_pe, long], paired=True) == (long, None)
    assert rf.choose([long, ref], paired=False) == (long, None)
    assert rf.choose([run(tmp_path, "brief", 80, 2.5), ref], paired=True) == (None, None)
    other_te = run(tmp_path, "te", 10, 2.5, pe="j-", te=0.04)
    assert rf.choose([long, other_te], paired=True) == (long, None)              # export_subject needs equal echo times


@pytest.mark.parametrize("case, expected", [
    ("brief", "too_few_volumes"), ("no_ref", "no_verified_reverse_pe_pair"), ("tr", "header_tr_outside_stratum"),
    ("no_t1", "no_t1"), ("fmriprep", "fmriprep_error"), ("xcpd", "xcpd_error"), ("min_time_log", "xcpd_min_time"),
    ("no_bold", "xcpd_min_time")])
def test_each_scan_ends_with_one_exclusion(monkeypatch, tmp_path, case, expected):
    tr = 2.5 if case == "tr" else 1.0
    outputs = [run(tmp_path / "c", "bold", 80 if case == "brief" else 310, tr)] + ([] if case == "no_ref" else [run(tmp_path / "c", "ref", 10, tr, pe="j-")])

    def broken(config):
        raise RuntimeError("fMRIPrep failed (1)")
    xcpd_run = FakeXCPD(bold=case != "no_bold", fail={"xcpd": "Traceback: segfault",
                                                       "min_time_log": "Less than 300 seconds survive High-Motion Outlier Scrubbing"}.get(case))
    row, a, _ = setup(monkeypatch, tmp_path, outputs, have_t1=case != "no_t1", fmriprep=broken if case == "fmriprep" else fake_fmriprep,
                      xcpd_run=xcpd_run)
    rows = rf.process(row, None, a)
    assert rows["P2_TR1"]["exclusion"] == expected and rows["P2_TR1"]["scan"] == "1001P2TR1"


def test_a_feature_row_has_split_half_striatal_columns_and_p2_tr1_a_no_notch_twin(monkeypatch, tmp_path):
    outputs = [run(tmp_path / "c", "bold", 310, 1.0), run(tmp_path / "c", "ref", 10, 1.0, pe="j-")]
    row, a, xcpd_run = setup(monkeypatch, tmp_path, outputs)
    rows = rf.process(row, None, a)
    assert set(rows) == {"P2_TR1", "P2_TR1:no_notch"} and xcpd_run.motion == [None, False]
    r = rows["P2_TR1"]
    assert r["exclusion"] == "" and r["distortion_correction"] and r["fmri_xcpd_available"] and r["header_tr"] == 1.0
    k = "fmri_xcpd_striatal_putamen_posterior_l__SomMot"
    assert {k, k + "__h1", k + "__h2"} <= set(r) and r[k + "__h1"] != r[k + "__h2"]
    work = Path(a.work)
    assert not list(work.rglob("*_bold.nii.gz")) and not list(work.rglob("*_xfm.h5")) and not (work / "bids" / "1001P2TR1").exists()
    assert Path(r["fmriprep_report"]).is_file() and Path(r["xcpd_dir"], "sub-1001P2TR1").is_dir()


def test_ppmi1_without_a_reverse_pe_reference_runs_uncorrected(monkeypatch, tmp_path):
    row, a, xcpd_run = setup(monkeypatch, tmp_path, [run(tmp_path / "c", "bold", 130, 2.5)], stratum="P1_TR2.5")
    rows = rf.process(row, None, a)
    assert set(rows) == {"P1_TR2.5"} and rows["P1_TR2.5"]["exclusion"] == "" and not rows["P1_TR2.5"]["distortion_correction"]
    assert xcpd_run.motion == [None]


def test_main_resumes_records_hashes_and_applies_the_review(monkeypatch, tmp_path):
    sel = tmp_path / "fmri_selection.csv"
    pd.DataFrame({"patno": [1001, 1002], "date": ["2022-03-04"] * 2, "image_ids": ["I1", "I2"], "stratum": ["P2_TR2.5"] * 2}).to_csv(sel, index=False)
    calls = []

    def fake(row, series, a):
        calls.append(int(row["patno"]))
        return {"P2_TR2.5": {"scan": rf.label(row), "patno": int(row["patno"]), "session_date": "2022-03-04", "protocol": "P2_TR2.5",
                             "exclusion": "", "detail": "", "xcpd_dir": "x"}}
    import pie.imaging.batch as batch
    monkeypatch.setattr(batch, "index_series", lambda zips: pd.DataFrame({"patno": [1001, 1002]}))
    monkeypatch.setattr(rf, "process", fake)
    args = ["--selection", str(sel), "--zips", "z.zip", "--work", str(tmp_path / "w"), "--license", "lic", "--workers", "1"]
    monkeypatch.setattr(rf, "ProcessPoolExecutor", _Serial)
    rf.main(args)
    assert sorted(calls) == [1001, 1002]
    rf.main(args)
    assert len(calls) == 2                                                              # finished scans are not rerun
    settings = json.loads((tmp_path / "w" / "run_settings.json").read_text())
    assert set(settings["sha256"]) == set(rf.FROZEN) and "license" not in settings
    review = tmp_path / "review.csv"
    pd.DataFrame({"scan": ["1001P2TR25", "1002P2TR25"], "protocol": ["P2_TR2.5"] * 2, "pass": ["True", "False"]}).to_csv(review, index=False)
    rf.main(args + ["--review", str(review)])
    s = pd.read_csv(tmp_path / "w" / "scans.csv", keep_default_na=False)
    assert dict(zip(s["scan"], s["exclusion"])) == {"1001P2TR25": "", "1002P2TR25": "visual_review_fail"}


class _Serial:
    """ProcessPoolExecutor stand-in: the fakes above are not picklable."""
    def __init__(self, max_workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        from concurrent.futures import Future
        f = Future()
        f.set_result(fn(*args))
        return f
