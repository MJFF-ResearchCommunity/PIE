"""XCP-D invocation contract and output parsing on synthetic trees; no XCP-D, container or participant data."""
import json
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from pie.imaging.xcpd import XCPDConfig, build_command, run_xcpd, xcpd_edges, xcpd_features, xcpd_striatal_features


def config(tmp_path, tr=1.0, **kwargs):
    fmriprep = tmp_path / "fmriprep out"
    func = fmriprep / "sub-A01" / "func"
    func.mkdir(parents=True)
    (fmriprep / "dataset_description.json").write_text('{"Name": "fMRIPrep", "DatasetType": "derivative"}')
    (func / "sub-A01_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json").write_text(
        json.dumps({"RepetitionTime": tr}))
    return XCPDConfig(fmriprep_dir=str(fmriprep), output_dir=str(tmp_path / "xcp_d"), work_dir=str(tmp_path / "work"),
                      cache_dir=str(tmp_path / "cache"), participants=["A01"], repetition_time=tr, **kwargs)


def flag(argv, name):
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("tr, motion", [(1.0, ["notch", "12", "28"]), (2.5, ["none"])])
def test_settings_are_explicit_and_the_motion_filter_follows_the_protocol(tmp_path, tr, motion):
    argv, env = build_command(config(tmp_path, tr=tr))
    assert argv[:4] == ["xcp_d", str(Path(tmp_path / "fmriprep out")), str(tmp_path / "xcp_d"), "participant"]
    assert {name: flag(argv, name) for name in ("--mode", "--file-format", "--nuisance-regressors", "--fd-thresh",
                                                "--censor-between", "--min-time", "--output-type", "--despike",
                                                "--smoothing", "--combine-runs")} == {
        "--mode": "linc", "--file-format": "nifti", "--nuisance-regressors": "36P", "--fd-thresh": "0.3",
        "--censor-between": "4", "--min-time": "300", "--output-type": "censored", "--despike": "y",
        "--smoothing": "6", "--combine-runs": "n"}
    assert flag(argv, "--motion-filter-type") == motion[0]
    if len(motion) > 1:
        assert [flag(argv, "--band-stop-min"), flag(argv, "--band-stop-max")] == motion[1:]
    else:
        assert "--band-stop-min" not in argv
    assert argv[argv.index("--atlases") + 1:argv.index("--min-coverage")] == ["4S456Parcels", "4S156Parcels"]
    assert not (tmp_path / "work").exists()


def test_container_keeps_the_images_own_templates_and_atlases(tmp_path):
    image = tmp_path / "xcp_d-26.2.0.sif"
    image.write_bytes(b"synthetic")
    c = config(tmp_path, backend="apptainer", executable="apptainer", container_image=str(image))
    argv, env = build_command(c)
    assert str(Path(c.fmriprep_dir).resolve()) + ":/data:ro" in argv
    assert any(a.startswith("TEMPLATEFLOW_HOME=/home/xcp_d/.cache/templateflow") for a in argv)
    assert argv[argv.index(str(image)) + 1:][:4] == ["xcp_d", "/data", "/out", "participant"]
    assert build_command(c, version_only=True)[0][-1] == "--version"


@pytest.mark.parametrize("change, message", [
    ({"repetition_time": 2.5}, "RepetitionTime"), ({"extra_args": ["--fd_thresh=0.5"]}, "configuration"),
    ({"participants": ["missing"]}, "No fMRIPrep"), ({"censor_between": -1}, "Invalid"),
    ({"backend": "apptainer"}, "image"),
])
def test_invalid_settings_rejected(tmp_path, change, message):
    c = config(tmp_path)
    for key, value in change.items():
        setattr(c, key, value)
    with pytest.raises(ValueError, match=message):
        build_command(c)


def test_output_nested_in_input_rejected(tmp_path):
    c = config(tmp_path)
    c.output_dir = str(Path(c.fmriprep_dir) / "xcp_d")
    with pytest.raises(ValueError, match="separate"):
        build_command(c)


def test_completion_is_recorded_once_and_a_changed_fmriprep_input_needs_a_new_work_dir(tmp_path, monkeypatch):
    c = config(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "XCP-D v26.2.0\n")
    def execute(args, **kwargs):
        calls.append(args)
        (Path(c.output_dir) / "sub-A01.html").write_text("report")
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, "run", execute)
    result = run_xcpd(c)
    assert result["returncode"] == 0 and result["scientific_qc_pass"] is False
    assert run_xcpd(c) == result and len(calls) == 1
    identity = json.loads((Path(c.work_dir) / "pie_xcpd_identity.json").read_text())
    assert identity["version"] == "XCP-D v26.2.0" and list(identity["fmriprep_hashes"]["A01"]) == [
        "func/sub-A01_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json"]
    (Path(c.fmriprep_dir) / "sub-A01" / "func" / "sub-A01_task-rest_desc-confounds_timeseries.tsv").write_text("x")
    with pytest.raises(ValueError, match="different XCP-D"):
        run_xcpd(c)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "XCP-D v26.1.1")
    with pytest.raises(RuntimeError, match="version"):
        run_xcpd(c)


AFFINE = np.array([[4., 0, 0, -40], [0, 4, 0, -30], [0, 0, 4, -20], [0, 0, 0, 1]])   # MNI mm around the striatum
PARCELS = pd.DataFrame({"index": [1, 2, 3, 4, 5, 6], "name": ["SM1", "SM2", "V1", "V2", "LH-Pu", "RH-Pu"],
                        "network_label": ["SomMot", "SomMot", "Vis", "Vis", None, None],
                        "atlas_name": ["Schaefer"] * 4 + ["CIT168Subcortical"] * 2})


def xcpd_tree(tmp_path, frames=150):
    """XCP-D 26.2.0 NIfTI layout: drive shared by the left posterior striatum and the somatomotor parcels."""
    root = tmp_path / "xcp_d"
    atlases = root / "atlases" / "atlas-4S456Parcels"
    atlases.mkdir(parents=True)
    PARCELS.to_csv(atlases / "atlas-4S456Parcels_dseg.tsv", sep="\t", index=False)
    func = root / "sub-A01" / "func"
    func.mkdir(parents=True)
    rng = np.random.default_rng(0)
    drive = rng.normal(size=frames)
    shape = (21, 16, 12)
    xyz = nib.affines.apply_affine(AFFINE, np.indices(shape).reshape(3, -1).T).reshape(shape + (3,))
    bold = rng.normal(size=shape + (frames,))
    bold[(xyz[..., 0] < 0) & (xyz[..., 1] < 0)] += 2 * drive
    stem = "sub-A01_task-rest_space-MNI152NLin2009cAsym"
    image = nib.Nifti1Image(bold.astype(np.float32), AFFINE)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, func / f"{stem}_res-2_desc-denoised_bold.nii.gz")
    nib.save(image, func / f"{stem}_res-2_desc-denoisedSmoothed_bold.nii.gz")     # --smoothing copy, not parsed
    (func / f"{stem}_res-2_desc-denoised_bold.json").write_text('{"RepetitionTime": 1.0}')
    series = pd.DataFrame({"SM1": drive + rng.normal(size=frames), "SM2": drive + rng.normal(size=frames),
                           "V1": rng.normal(size=frames), "V2": np.nan,                  # V2 below --min-coverage
                           "LH-Pu": rng.normal(size=frames), "RH-Pu": rng.normal(size=frames)})
    series.to_csv(func / f"{stem}_atlas-4S456Parcels_stat-mean_timeseries.tsv", sep="\t", index=False, na_rep="n/a")
    series.corr().to_csv(func / f"{stem}_atlas-4S456Parcels_stat-pearsoncorrelation_relmat.tsv", sep="\t",
                         index_label="Node", na_rep="n/a")
    pd.DataFrame({"coverage": [1, 1, .9, .3, 1, 1]}, index=PARCELS["name"]).to_csv(
        func / f"{stem}_atlas-4S456Parcels_stat-coverage_bold.tsv", sep="\t", index_label="Node")
    pd.DataFrame([[400, 600, 500, np.nan, 300, 200]], columns=PARCELS["name"]).to_csv(
        func / f"{stem}_atlas-4S456Parcels_stat-alff_bold.tsv", sep="\t", index=False, na_rep="n/a")
    pd.DataFrame([[.3, .5, .2, np.nan, .4, .4]], columns=PARCELS["name"]).to_csv(
        func / f"{stem}_atlas-4S456Parcels_stat-reho_bold.tsv", sep="\t", index=False, na_rep="n/a")
    pd.DataFrame({"sub": ["A01"], "task": ["rest"], "run": [1], "space": ["MNI152NLin2009cAsym"], "mean_fd": [.21],
                  "mean_fd_post_censoring": [.12], "num_censored_volumes": [30],
                  "num_retained_volumes": [frames]}).to_csv(func / f"{stem}_desc-linc_qc.tsv", sep="\t", index=False)
    return root, series


def test_features_are_prefixed_and_uncovered_parcels_left_out(tmp_path):
    root, series = xcpd_tree(tmp_path)
    row = xcpd_features(root, "A01")
    assert row["fmri_xcpd_available"] and row["fmri_xcpd_qc_mean_fd"] == .21 and row["fmri_xcpd_qc_retained_seconds"] == 150
    assert not {"fmri_xcpd_qc_sub", "fmri_xcpd_qc_run", "fmri_xcpd_qc_space"} & set(row)          # identifiers are not features
    z = np.arctanh(series.corr().clip(-1 + 1e-7, 1 - 1e-7))
    assert row["fmri_xcpd_fc_4S456Parcels_SomMot__SomMot"] == pytest.approx(z.loc["SM1", "SM2"])
    assert row["fmri_xcpd_fc_4S456Parcels_SomMot__Vis"] == pytest.approx(z.loc[["SM1", "SM2"], "V1"].mean())
    assert np.isnan(row["fmri_xcpd_fc_4S456Parcels_Vis__Vis"])                             # its only edge touches V2
    assert row["fmri_xcpd_fc_4S456Parcels_CIT168Subcortical__CIT168Subcortical"] == pytest.approx(z.loc["LH-Pu", "RH-Pu"])
    assert row["fmri_xcpd_coverage_min_4S456Parcels"] == .3 and row["fmri_xcpd_parcels_nan_4S456Parcels"] == 1
    assert row["fmri_xcpd_malff_4S456Parcels_SomMot"] == pytest.approx(500 / 400)          # over the mean across parcels
    assert row["fmri_xcpd_reho_4S456Parcels_Vis"] == pytest.approx(.2)
    names, edges = xcpd_edges(root, "A01")
    assert names == list(PARCELS["name"]) and len(edges) == 15
    assert xcpd_features(root, "B02") == {"fmri_xcpd_available": False} and xcpd_striatal_features(root, "B02") == {}


def test_striatal_features_are_recomputed_on_the_denoised_bold(tmp_path):
    root, _ = xcpd_tree(tmp_path)
    z = xcpd_striatal_features(root, "A01")
    assert z["fmri_xcpd_striatal_putamen_posterior_l__SomMot"] > 0.5
    assert abs(z["fmri_xcpd_striatal_putamen_posterior_l__Vis"]) < 0.3
    assert abs(z["fmri_xcpd_striatal_putamen_anterior_l__SomMot"]) < 0.3 and abs(z["fmri_xcpd_striatal_putamen_posterior_r__SomMot"]) < 0.3
    assert not any("CIT168" in k for k in z)                                         # cortical targets only


def test_motion_filter_can_be_switched_off_for_the_no_notch_variant(tmp_path):
    argv, _ = build_command(config(tmp_path, tr=1.0, motion_filter=False))
    assert flag(argv, "--motion-filter-type") == "none" and "--band-stop-min" not in argv
    with pytest.raises(ValueError, match="motion filter"):
        build_command(config(tmp_path / "b", tr=2.5, motion_filter=True))      # no notch above TR 1.5 s


def test_striatal_features_on_each_half_of_the_retained_frames(tmp_path):
    root, _ = xcpd_tree(tmp_path)
    whole = xcpd_striatal_features(root, "A01")
    assert xcpd_striatal_features(root, "A01", frames=slice(0, 150)) == whole
    h1, h2 = (xcpd_striatal_features(root, "A01", frames=f) for f in (slice(0, 75), slice(75, None)))
    k = "fmri_xcpd_striatal_putamen_posterior_l__SomMot"
    assert h1[k] > 0.4 and h2[k] > 0.4 and h1[k] != whole[k] != h2[k]
