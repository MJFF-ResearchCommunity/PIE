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


def test_cli_writes_map_sidecar_and_one_manifest_entry(tmp_path):
    import json

    from pie.imaging.viewer.catalog import Catalog

    mri = tmp_path / "fs" / "mri"
    mri.mkdir(parents=True)
    nib.save(nib.MGHImage(np.asarray(_atlas().dataobj).astype(np.int32), np.eye(4)), mri / "aparc.DKTatlas+aseg.deep.mgz")
    nib.save(nib.MGHImage(np.ones((4, 4, 4), np.float32), np.eye(4)), mri / "orig_nu.mgz")
    pd.DataFrame({"feature": ["vol_Left_Putamen", "vol_ctx_lh_precentral"], "d": [-0.5, 0.3], "p": [0.01, 0.02]}).to_csv(tmp_path / "t.csv", index=False)
    m = tmp_path / "viewer.json"
    m.write_text(json.dumps({"version": 1, "subjects": [{"id": "tpl"}], "scans": [
        {"id": "tpl-t1", "subject": "tpl", "modality": "MRI", "path": "fs/mri/orig_nu.mgz", "space": "tpl:T1-conformed"}]}))
    args = ["--table", str(tmp_path / "t.csv"), "--value", "d", "--p", "p", "--alpha", "0.05", "--fastsurfer-dir", str(tmp_path / "fs"),
            "--out", str(tmp_path / "maps" / "pd_vs_hc"), "--statistic", "Cohen's d (PD - HC)", "--units", "SD",
            "--manifest", str(m), "--reference-id", "tpl-t1"]
    regionmap.main(args)
    regionmap.main(args)
    assert (tmp_path / "maps" / "pd_vs_hc.nii.gz").is_file() and json.loads((tmp_path / "maps" / "pd_vs_hc.json").read_text())["vmax"] == 0.5
    stats = [s for s in json.loads(m.read_text())["scans"] if s.get("kind") == "statistic"]
    assert len(stats) == 1 and stats[0]["space"] == "tpl:T1-conformed" and stats[0]["metadata"]["n_shown"] == 2
    scan = Catalog(tmp_path / "repo", manifest=m).scans[stats[0]["id"]]
    assert scan.registration == "verified" and scan.reference_id == "tpl-t1"


def test_figures_write_slices_for_participant_maps_even_when_empty(tmp_path):
    import json

    import matplotlib
    matplotlib.use("Agg")
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(np.random.default_rng(0).random((20, 20, 20)).astype(np.float32), affine), tmp_path / "bg.nii.gz")
    for name, vmax in (("effect", 0.5), ("none", 0.0)):
        m = np.zeros((20, 20, 20), np.float32)
        m[5:9, 5:9, 5:9], m[12:16, 12:16, 12:16] = -vmax, vmax
        nib.save(nib.Nifti1Image(m, affine), tmp_path / f"{name}.nii.gz")
        (tmp_path / f"{name}.json").write_text(json.dumps({"statistic": "Cohen's d", "vmax": vmax, "threshold": {"p_column": "p", "alpha": 0.05}}))
        out = regionmap.figures(tmp_path / f"{name}.nii.gz", tmp_path / f"{name}.json", tmp_path / "bg.nii.gz", tmp_path / "fig")
        assert sorted(p.name for p in out) == [f"{name}_slices.png", f"{name}_slices.svg"] and all(p.stat().st_size for p in out)
    assert "no region passes the threshold" in (tmp_path / "fig" / "none_slices.svg").read_text()
    assert "no region passes" not in (tmp_path / "fig" / "effect_slices.svg").read_text()


def test_surface_projection_keeps_only_that_hemispheres_cortex():
    a = np.zeros((4, 4, 4), np.int16)
    a[0], a[1], a[2] = 4, 1024, 2024                  # lateral ventricle, ctx-lh-precentral, ctx-rh-precentral
    m = nib.Nifti1Image(np.where(a > 0, 0.5, 0).astype(np.float32), np.eye(4))
    left = np.asarray(regionmap.cortex_only(m, nib.Nifti1Image(a, np.eye(4)), "left").dataobj)
    assert left[1].min() == np.float32(0.5) and not left[0].any() and not left[2].any()
    right = np.asarray(regionmap.cortex_only(m, nib.Nifti1Image(a, np.eye(4)), "right").dataobj)
    assert right[2].min() == np.float32(0.5) and not right[:2].any()


def test_surface_view_keeps_positive_effects_warm(tmp_path, monkeypatch):
    """An all-positive cortical map must not borrow the blue (negative) half of the colour scale on the surface."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.image import imread
    from nilearn import datasets
    from scipy.spatial import ConvexHull

    a = np.zeros((12, 12, 12), np.int16)
    a[2:10, 2:10, 2:10] = 1024
    m = np.where(a > 0, np.random.default_rng(1).uniform(0.05, 0.6, a.shape), 0).astype(np.float32)
    nib.save(nib.Nifti1Image(a, np.eye(4)), tmp_path / "atlas.nii.gz")
    nib.save(nib.Nifti1Image(m, np.eye(4)), tmp_path / "pos.nii.gz")
    (tmp_path / "pos.json").write_text(json.dumps({"statistic": "d", "units": "SD", "vmax": 0.6, "threshold": None,
                                                   "atlas": str(tmp_path / "atlas.nii.gz")}))
    pts = np.random.default_rng(2).uniform(3, 8, (200, 3))
    mesh = (pts, ConvexHull(pts).simplices)
    fake = {f"{k}_{h}": mesh for k in ("pial", "infl") for h in ("left", "right")}
    fake |= {f"sulc_{h}": np.zeros(len(pts)) for h in ("left", "right")}
    monkeypatch.setattr(datasets, "fetch_surf_fsaverage", lambda *args, **kwargs: fake)
    out = regionmap.figures(tmp_path / "pos.nii.gz", tmp_path / "pos.json", tmp_path / "pos.nii.gz", tmp_path / "fig", template=True)
    img = imread(next(p for p in out if p.name.endswith("_surface.png")))
    brains = img[:, : int(0.75 * img.shape[1]), :3]                # the four views, not the colour bar
    blue = (brains[..., 2] > brains[..., 0] + 0.3) & (brains[..., 2] > 0.5)
    assert not blue.any()


def test_two_measures_for_one_region_are_refused_not_overwritten(tmp_path):
    import pytest
    p = tmp_path / "atlas.nii.gz"
    nib.save(_atlas(), p)
    t = pd.DataFrame({"feature": ["vol_ctx_lh_precentral", "cth_lh_precentral"], "d": [0.8, -0.3], "p": [0.01, 0.01]})
    with pytest.raises(ValueError, match="vol_ctx_lh_precentral / cth_lh_precentral"):
        regionmap.region_map(t, p, value="d")


def _fs(root, affine=np.eye(4)):
    mri = root / "mri"
    mri.mkdir(parents=True)
    nib.save(nib.MGHImage(np.asarray(_atlas().dataobj).astype(np.int32), affine), mri / "aparc.DKTatlas+aseg.deep.mgz")
    nib.save(nib.MGHImage(np.ones((4, 4, 4), np.float32), affine), mri / "orig_nu.mgz")
    return root


def test_cli_attaches_a_map_to_a_pie_discovered_t1(tmp_path):
    import json

    from pie.imaging.viewer.catalog import Catalog
    repo = tmp_path / "repo"
    _fs(repo / "Imaging/derived/fastsurfer/A")
    (repo / "Imaging/derived/sessions.csv").write_text("patno,image_id,series_desc\n1,A,T1\n")
    pd.DataFrame({"feature": ["vol_Left_Putamen"], "z": [-2.5]}).to_csv(tmp_path / "t.csv", index=False)
    m = tmp_path / "viewer.json"
    m.write_text(json.dumps({"version": 1, "scans": []}))
    regionmap.main(["--table", str(tmp_path / "t.csv"), "--value", "z", "--fastsurfer-dir", str(repo / "Imaging/derived/fastsurfer/A"),
                    "--out", str(tmp_path / "z"), "--manifest", str(m), "--reference-id", "mri-A", "--repo", str(repo)])
    scan = Catalog(repo, manifest=m).scans["stat-z"]
    assert scan.subject == "1" and scan.space == "sub-1:T1:A" and scan.reference_id == "mri-A"


def test_cli_refuses_a_reference_on_another_grid_and_leaves_the_manifest_alone(tmp_path):
    import json

    import pytest
    _fs(tmp_path / "fs")
    _fs(tmp_path / "other", affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    pd.DataFrame({"feature": ["vol_Left_Putamen"], "d": [0.4]}).to_csv(tmp_path / "t.csv", index=False)
    m = tmp_path / "viewer.json"
    before = json.dumps({"version": 1, "scans": [{"id": "t1", "subject": "s", "modality": "MRI", "path": "other/mri/orig_nu.mgz", "space": "s:T1"}]})
    m.write_text(before)
    with pytest.raises(SystemExit):
        regionmap.main(["--table", str(tmp_path / "t.csv"), "--value", "d", "--fastsurfer-dir", str(tmp_path / "fs"),
                        "--out", str(tmp_path / "x"), "--manifest", str(m), "--reference-id", "t1"])
    assert m.read_text() == before


def test_empty_map_figure_still_shows_the_background(tmp_path):
    import json

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.image import imread
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    yy = np.indices((20, 20, 20))[1].astype(np.float32)
    nib.save(nib.Nifti1Image(yy, affine), tmp_path / "bg.nii.gz")                 # a gradient: visible structure
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20), np.float32), affine), tmp_path / "none.nii.gz")
    (tmp_path / "none.json").write_text(json.dumps({"statistic": "d", "vmax": 0.0, "threshold": None}))
    png = next(p for p in regionmap.figures(tmp_path / "none.nii.gz", tmp_path / "none.json", tmp_path / "bg.nii.gz", tmp_path / "f")
               if p.suffix == ".png")
    img = imread(png)[..., :3].mean(axis=2)
    h, w = img.shape
    inside = img[int(0.35 * h): int(0.7 * h), int(0.05 * w): int(0.25 * w)]       # inside the first panel, clear of labels
    assert inside.std() > 0.05                                                      # the background shows, not a uniform dark box
