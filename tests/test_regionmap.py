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


def test_surface_values_paint_whole_parcels_of_that_hemisphere():
    a = np.zeros((4, 4, 4), np.int16)
    a[0], a[1], a[2] = 4, 1024, 2024                  # lateral ventricle, ctx-lh-precentral, ctx-rh-precentral
    m = nib.Nifti1Image(np.where(a == 1024, 0.5, np.where(a == 2024, -0.4, np.where(a == 4, 0.9, 0))).astype(np.float32), np.eye(4))
    labels = np.array([4.0, 1024.0, 2024.0, 0.0, 1024.0])        # per-vertex labels sampled from the atlas
    atlas = nib.Nifti1Image(a, np.eye(4))
    assert regionmap.surface_values(m, atlas, labels, "left").tolist() == [0, 0.5, 0, 0, 0.5]
    assert regionmap.surface_values(m, atlas, labels, "right").tolist() == [0, 0, np.float32(-0.4), 0, 0]


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
    d = np.random.default_rng(2).normal(size=(200, 3))
    pts = 5.5 + 2.5 * d / np.linalg.norm(d, axis=1, keepdims=True)        # a small sphere: every point is a mesh vertex
    faces = ConvexHull(pts).simplices
    # one array per surface, as fsaverage's files are (nilearn's plotting recentres a mesh in place)
    fake = {f"{k}_{h}": (pts.copy(), faces.copy()) for k in ("pial", "white", "infl") for h in ("left", "right")}
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


def test_cli_sidecar_hashes_the_csv_file_itself(tmp_path):
    import hashlib
    _fs(tmp_path / "fs")
    (tmp_path / "t.csv").write_text("feature,d\nvol_Left_Putamen,0.40\n")
    side = regionmap.main(["--table", str(tmp_path / "t.csv"), "--value", "d", "--fastsurfer-dir", str(tmp_path / "fs"), "--out", str(tmp_path / "x")])
    assert side["source_sha256"] == hashlib.sha256((tmp_path / "t.csv").read_bytes()).hexdigest()


def test_regions_absent_from_the_atlas_image_are_reported_not_shown(tmp_path):
    p = tmp_path / "atlas.nii.gz"
    nib.save(_atlas(), p)                              # labels 12, 51, 1024 only
    t = pd.DataFrame({"feature": ["vol_Left_Putamen", "vol_Left_Caudate"], "d": [-0.5, 0.9]})
    _, side = regionmap.region_map(t, p, value="d")
    assert side["absent_from_atlas"] == ["vol_Left_Caudate"] and side["n_shown"] == 1 and side["vmax"] == 0.5


def test_sidecar_records_the_absolute_atlas_path_and_its_hash(tmp_path, monkeypatch):
    import hashlib
    nib.save(_atlas(), tmp_path / "atlas.nii.gz")
    monkeypatch.chdir(tmp_path)
    _, side = regionmap.region_map(pd.DataFrame({"feature": ["vol_Left_Putamen"], "d": [0.3]}), "atlas.nii.gz", value="d")
    assert side["atlas"] == str(tmp_path / "atlas.nii.gz")
    assert side["atlas_sha256"] == hashlib.sha256((tmp_path / "atlas.nii.gz").read_bytes()).hexdigest()


def test_cli_p_without_alpha_is_an_error(tmp_path):
    import pytest
    _fs(tmp_path / "fs")
    pd.DataFrame({"feature": ["vol_Left_Putamen"], "d": [0.4], "p": [0.2]}).to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(SystemExit):
        regionmap.main(["--table", str(tmp_path / "t.csv"), "--value", "d", "--p", "p", "--fastsurfer-dir", str(tmp_path / "fs"), "--out", str(tmp_path / "x")])


def _two_refs(tmp_path, second_atlas_differs=False):
    import json
    _fs(tmp_path / "fs")
    other = _fs(tmp_path / "fs2")
    if second_atlas_differs:
        nib.save(nib.MGHImage(np.full((4, 4, 4), 17, np.int32), np.eye(4)), other / "mri" / "aparc.DKTatlas+aseg.deep.mgz")
    pd.DataFrame({"feature": ["vol_Left_Putamen"], "d": [0.4]}).to_csv(tmp_path / "t.csv", index=False)
    m = tmp_path / "viewer.json"
    m.write_text(json.dumps({"version": 1, "scans": [
        {"id": "a", "subject": "s", "modality": "MRI", "path": "fs/mri/orig_nu.mgz", "atlas": "fs/mri/aparc.DKTatlas+aseg.deep.mgz", "space": "s:T1"},
        {"id": "b", "subject": "s", "modality": "MRI", "path": "fs2/mri/orig_nu.mgz", "atlas": "fs2/mri/aparc.DKTatlas+aseg.deep.mgz", "space": "s:T1b"}]}))
    return m, ["--table", str(tmp_path / "t.csv"), "--value", "d", "--fastsurfer-dir", str(tmp_path / "fs"), "--out", str(tmp_path / "x"), "--manifest", str(m)]


def test_cli_will_not_take_over_a_same_named_map_of_another_reference(tmp_path):
    import pytest
    m, args = _two_refs(tmp_path)
    regionmap.main(args + ["--reference-id", "a"])
    before = m.read_text()
    with pytest.raises(SystemExit):
        regionmap.main(args[:4] + ["--fastsurfer-dir", str(tmp_path / "fs2")] + args[6:] + ["--reference-id", "b"])
    assert m.read_text() == before


def test_cli_refuses_a_reference_whose_segmentation_differs(tmp_path):
    import pytest
    m, args = _two_refs(tmp_path, second_atlas_differs=True)
    before = m.read_text()
    with pytest.raises(SystemExit):                    # same grid, different segmentation: not the canvas it was painted on
        regionmap.main(args + ["--reference-id", "b"])
    assert m.read_text() == before


def test_figure_colours_match_the_viewers_warm_and_cool_scales():
    c = regionmap.STAT_CMAP
    assert np.allclose(c(1.0)[:3], (1, 254 / 255, 0), atol=0.01) and np.allclose(c(0.0)[:3], (0, 254 / 255, 1), atol=0.01)
    assert np.allclose(c(0.499)[:3], (0, 127 / 255, 1), atol=0.02) and np.allclose(c(0.501)[:3], (1, 127 / 255, 0), atol=0.02)


def test_z_prefixed_normative_columns_match_their_regions():
    got, unmatched = regionmap.match_features(["z_vol_Left_Putamen", "z_cth_lh_precentral"], LUT)
    assert got == {"z_vol_Left_Putamen": 12, "z_cth_lh_precentral": 1024} and unmatched == []


def _pie_repo(tmp_path):
    repo = tmp_path / "repo"
    _fs(repo / "Imaging/derived/fastsurfer/A")
    (repo / "Imaging/derived/sessions.csv").write_text("patno,image_id,series_desc\n1,A,T1\n")
    pd.DataFrame({"IMAGEID": ["A", "B"], "PATNO": [1, 2], "z_vol_Left_Putamen": [-2.4, 0.3],
                  "z_vol_Right_Putamen": [-1.1, 0.2], "z_cth_lh_precentral": [0.9, -0.5]}).to_csv(tmp_path / "z.csv", index=False)
    return repo


def test_participant_mode_paints_that_scans_row_of_a_wide_table(tmp_path):
    import json

    from pie.imaging.viewer.catalog import Catalog
    repo = _pie_repo(tmp_path)
    m = tmp_path / "viewer.json"
    m.write_text(json.dumps({"version": 1, "scans": []}))
    side = regionmap.main(["--table", str(tmp_path / "z.csv"), "--image-id", "A", "--out", str(tmp_path / "A_z"),
                           "--manifest", str(m), "--repo", str(repo)])
    out = np.asarray(nib.load(tmp_path / "A_z.nii.gz").dataobj)
    assert out[0].min() == np.float32(-2.4) and out[1].min() == np.float32(-1.1) and not out[2].any()   # vol only
    assert side["n_shown"] == 2 and side["unmatched"] == []
    scan = Catalog(repo, manifest=m).scans["stat-A_z"]
    assert scan.reference_id == "mri-A" and scan.subject == "1"


def test_participant_mode_measure_picks_thickness(tmp_path):
    repo = _pie_repo(tmp_path)
    side = regionmap.main(["--table", str(tmp_path / "z.csv"), "--image-id", "A", "--measure", "cth", "--out", str(tmp_path / "A_cth"),
                           "--repo", str(repo)])
    assert side["n_shown"] == 1 and side["vmax"] == np.float32(0.9)


def test_participant_mode_unknown_image_id_is_an_error(tmp_path, capsys):
    import pytest
    repo = _pie_repo(tmp_path)
    with pytest.raises(SystemExit):
        regionmap.main(["--table", str(tmp_path / "z.csv"), "--image-id", "C", "--out", str(tmp_path / "C"), "--repo", str(repo)])
    assert "no IMAGEID C" in capsys.readouterr().err
