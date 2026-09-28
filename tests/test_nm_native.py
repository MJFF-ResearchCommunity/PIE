"""Native-space neuromelanin measures: MNI regions pulled onto the slab in one resampling, the published nigral
threshold volume (Langley et al.) and the snceg segmentation contract (Lillebostad et al.)."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging import nm_native as N


def _dice(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    """A small 'MNI' volume, a smoothly deformed and shifted 'T1', and a real ANTs SyN between them."""
    import ants
    import SimpleITK as sitk
    from scipy import ndimage

    d = tmp_path_factory.mktemp("syn")
    rng = np.random.default_rng(0)
    shape = (44, 44, 44)
    zz, yy, xx = np.indices(shape)
    head = (((zz - 22) / 18) ** 2 + ((yy - 22) / 16) ** 2 + ((xx - 22) / 17) ** 2) < 1
    mni = (head * (1 + 0.5 * ndimage.gaussian_filter(rng.normal(size=shape), 2))).astype(np.float32)
    label = np.zeros(shape, np.int16)
    label[18:24, 20:27, 12:18] = 101                      # a left structure
    label[18:24, 20:27, 27:33] = 202                      # its right partner
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    aff[:3, 3] = -33
    mni_img, lab_img = nib.Nifti1Image(mni, aff), nib.Nifti1Image(label, aff)
    # deform: smooth displacement (up to ~3 mm) plus a 2 mm shift, applied with SimpleITK in physical space
    ref = sitk.GetImageFromArray(np.transpose(mni, (2, 1, 0)))
    ref.SetSpacing((1.5,) * 3)
    ref.SetOrigin((33.0, 33.0, -33.0))                   # nibabel RAS (-33,-33,-33) in ITK's LPS
    ref.SetDirection((-1.0, 0, 0, 0, -1.0, 0, 0, 0, 1.0))
    field = np.stack([ndimage.gaussian_filter(rng.normal(size=shape), 6) * 60 for _ in range(3)], -1)
    disp = sitk.GetImageFromArray(np.transpose(field, (2, 1, 0, 3)).astype(np.float64), isVector=True)
    disp.CopyInformation(ref)
    tx = sitk.CompositeTransform([sitk.TranslationTransform(3, (2.0, -1.0, 1.5)), sitk.DisplacementFieldTransform(disp)])
    t1 = np.transpose(sitk.GetArrayFromImage(sitk.Resample(ref, ref, tx, sitk.sitkLinear, 0.0)), (2, 1, 0))
    t1_img = nib.Nifti1Image(t1.astype(np.float32), aff)
    for name, img in (("mni", mni_img), ("t1", t1_img), ("lab", lab_img)):
        nib.save(img, d / f"{name}.nii.gz")
    from pie.imaging.features import _seed_ants
    _seed_ants(0)
    reg = ants.registration(ants.image_read(str(d / "mni.nii.gz")), ants.image_read(str(d / "t1.nii.gz")),
                            type_of_transform="SyN", outprefix=str(d / "reg_"))
    expected = ants.apply_transforms(ants.image_read(str(d / "t1.nii.gz")), ants.image_read(str(d / "lab.nii.gz")),
                                     reg["invtransforms"], whichtoinvert=[True, False], interpolator="genericLabel").numpy()
    cache = d / "cache"                                      # PIE's cache holds only the forward pair (no ANTs inverse to find)
    cache.mkdir()
    warp, affine = cache / "t1_to_MNI152NLin2009cAsym_syn_1Warp.nii.gz", cache / "t1_to_MNI152NLin2009cAsym_syn_0GenericAffine.mat"
    Path(reg["fwdtransforms"][0]).replace(warp), Path(reg["fwdtransforms"][1]).replace(affine)
    return {"dir": d, "t1": t1_img, "lab": lab_img, "warp": str(warp), "affine": str(affine),
            "expected_t1": np.rint(expected).astype(int)}


def test_labels_reach_the_t1_grid_through_the_inverted_forward_warp(syn):
    import SimpleITK as sitk
    got = N.pull_labels(syn["lab"], syn["t1"], sitk.Euler3DTransform(), syn["warp"], syn["affine"])[0]
    for code in (101, 202):
        assert _dice(got == code, syn["expected_t1"] == code) > 0.95, code


def test_labels_reach_an_oblique_slab_through_affine_and_rigid_in_one_step(syn, tmp_path):
    """Deterministic chain check: zero warp, a known affine (MNI point -> T1 point, as ANTs' forward affine) and an
    oblique rigid slab. PIE's single ANTs pull must equal an independent SimpleITK composite (slab -> T1 -> MNI)."""
    import ants
    import SimpleITK as sitk
    zero = tmp_path / "t1_to_MNI152NLin2009cAsym_syn_1Warp.nii.gz"
    ants.image_write(ants.image_read(syn["warp"]) * 0, str(zero))
    aff = sitk.AffineTransform(3)
    aff.SetMatrix(tuple(sitk.Euler3DTransform((0.0, 0.0, 0.0), 0.05, -0.03, 0.08).GetMatrix()))
    aff.SetTranslation((2.0, -1.5, 1.0))
    sitk.WriteTransform(aff, str(tmp_path / "affine.mat"))
    slab_aff = np.diag([0.75, 0.75, 1.5, 1.0])
    slab_aff[:3, 3] = (-20, -20, -12)
    slab = nib.Nifti1Image(np.zeros((54, 54, 16), np.float32), slab_aff)
    tx_t1_slab = sitk.Euler3DTransform((0.0, 0.0, 0.0), 0.0, 0.0, np.radians(8), (1.5, -2.0, 0.5))   # T1 point -> slab point
    got = N.pull_labels(syn["lab"], slab, tx_t1_slab, str(zero), str(tmp_path / "affine.mat"))[0]
    lab = sitk.GetImageFromArray(np.transpose(np.asarray(syn["lab"].dataobj), (2, 1, 0)).astype(np.float32))
    lab.SetSpacing((1.5,) * 3), lab.SetOrigin((33.0, 33.0, -33.0)), lab.SetDirection((-1.0, 0, 0, 0, -1.0, 0, 0, 0, 1.0))
    grid = sitk.Image(54, 54, 16, sitk.sitkFloat32)
    grid.SetSpacing((0.75, 0.75, 1.5)), grid.SetOrigin((20.0, 20.0, -12.0)), grid.SetDirection((-1.0, 0, 0, 0, -1.0, 0, 0, 0, 1.0))
    chain = sitk.CompositeTransform(3)
    chain.AddTransform(aff.GetInverse())                     # applied second: T1 point -> MNI point
    chain.AddTransform(tx_t1_slab.GetInverse())              # applied first: slab point -> T1 point
    ref = np.transpose(sitk.GetArrayFromImage(sitk.Resample(lab, grid, chain, sitk.sitkNearestNeighbor, 0.0)), (2, 1, 0))
    for code in (101, 202):
        assert (ref == code).sum() > 100 and _dice(got == code, ref == code) > 0.97, code


def test_coverage_is_the_share_of_a_region_the_slab_contains(syn):
    import SimpleITK as sitk
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    aff[:3, 3] = (-33, -33, -12.75)                          # 3 slices through the middle of structure 101 (z -15..-7.5 mm)
    thin = nib.Nifti1Image(np.zeros((44, 44, 3), np.float32), aff)
    lab, cov = N.pull_labels(syn["lab"], thin, sitk.Euler3DTransform(), syn["warp"], syn["affine"], codes={"l": [101], "r": [202]})
    assert lab.shape == (44, 44, 3) and 0.3 < cov["l"] < 0.7 and cov["r"] < 0.05
    full = N.pull_labels(syn["lab"], syn["t1"], sitk.Euler3DTransform(), syn["warp"], syn["affine"], codes={"l": [101]})[1]
    assert full["l"] > 0.99


def test_region_codes_carry_side_territory_search_ring_and_crus():
    aff = np.eye(4)
    aff[:3, 3] = (-20, -30, -25)
    lab = np.zeros((41, 30, 20), np.int16)                   # MNI x = -20..20
    lab[10:14, 10:14, 8:12] = 3                              # left sensorimotor (x -10..-7)
    lab[27:31, 10:14, 8:12] = 1                              # right associative (x 7..10)
    lab[5:9, 16:20, 8:12] = 4                                # left crus (x -15..-12)
    lab[31:35, 16:20, 8:12] = 4                              # right crus
    lab[18:23, 2:5, 8:12] = 4                                # midline tegmental part
    codes = np.asarray(N.region_labels(nib.Nifti1Image(lab, aff)).dataobj)
    region, side = codes % 100, codes // 100
    assert (region[10:14, 10:14, 8:12] == 3).all() and (side[10:14, 10:14, 8:12] == 1).all()
    assert (region[27:31, 10:14, 8:12] == 1).all() and (side[27:31, 10:14, 8:12] == 2).all()
    assert (region[5:9, 16:20, 8:12] == N.CRUS).all() and (region[31:35, 16:20, 8:12] == N.CRUS).all()
    assert (region[18:23, 2:5, 8:12] == N.TEGMENTUM).all()
    ring = region == N.RING
    assert ring[9, 11, 9] and ring[14, 11, 9] and not ring[10:14, 10:14, 8:12].any()   # 1 mm around the SN, not in it


def test_langley_volume_counts_search_voxels_above_the_crus_threshold():
    rng = np.random.default_rng(1)
    shape = (40, 40, 8)
    codes = np.full(shape, 100, np.int16)                    # left hemisphere background
    codes[20:] += 100                                        # x >= 20: right
    codes[5:15, 5:15, 2:6] = 100 + 3                         # left SN search
    codes[25:35, 5:15, 2:6] = 200 + 3
    codes[5:15, 25:35, 2:6] = 100 + N.CRUS
    codes[25:35, 25:35, 2:6] = 200 + N.CRUS
    nm = rng.normal(100, 5, shape)
    nm[5:10, 5:15, 2:6] = 150                                # 200 bright voxels left (half the search region)
    nm[25:28, 5:15, 2:6] = 150                               # 120 bright voxels right
    aff = np.diag([0.5, 0.5, 2.0, 1.0])                      # 0.5 mm^3 voxels
    out = N.langley_features(nib.Nifti1Image(nm, aff), codes)
    assert abs(out["nml_sn_volume_l_mm3"] - 100) < 5 and abs(out["nml_sn_volume_r_mm3"] - 60) < 5
    assert abs(out["nml_sn_volume_mm3"] - 160) < 8 and abs(out["nml_threshold"] - (100 + 2.8 * 5)) < 1.5


def test_snceg_features_measure_the_segmented_nigra_against_the_crus(tmp_path):
    shape = (40, 40, 8)
    codes = np.full(shape, 100, np.int16)
    codes[20:] += 100
    codes[5:15, 25:35, 2:6] = 100 + N.CRUS
    codes[25:35, 25:35, 2:6] = 200 + N.CRUS
    nm = np.full(shape, 100.0)
    for x0 in (5, 25):                                                    # both crus parts: mean 100, SD 10
        nm[x0:x0 + 10, 25:35, 2:6] += np.tile([-10, 10], 200).reshape(10, 10, 4)
    sn = np.zeros(shape, bool)
    sn[5:10, 5:15, 2:6] = True
    sn[25:30, 5:15, 2:6] = True
    nm[sn] = 130
    out = N.snceg_features(nib.Nifti1Image(nm, np.diag([0.5, 0.5, 2.0, 1.0])), sn, codes)
    assert abs(out["nms_sn_volume_mm3"] - 400 * 0.5) < 1e-6 and abs(out["nms_sn_l_cr"] - 0.3) < 1e-6
    assert abs(out["nms_sn_l_cnr"] - 3.0) < 1e-6 and abs(out["nms_sn_mean_cr"] - 0.3) < 1e-6


def test_snceg_runs_in_its_own_environment_and_must_return_the_input_grid(tmp_path, monkeypatch):
    img = nib.Nifti1Image(np.random.default_rng(0).random((20, 20, 6)).astype(np.float32), np.diag([0.5, 0.5, 2.0, 1.0]))
    nib.save(img, tmp_path / "nm_mean.nii.gz")
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\n"                            # stands in for the snceg environment's python
                    f"{sys.executable} - \"$@\" <<'EOF'\n"
                    "import sys, nibabel as nib, numpy as np\n"
                    "a = sys.argv[1:]; i = nib.load(a[a.index('--input') + 1]); m = np.zeros(i.shape, np.uint8); m[5:8, 5:8, 2:4] = 1\n"
                    "nib.save(nib.Nifti1Image(m, i.affine), a[a.index('--output') + 1])\nEOF\n")
    fake.chmod(0o755)
    monkeypatch.setattr(N, "snceg_model", lambda cache_dir=None: tmp_path)
    mask = N.snceg_mask(tmp_path / "nm_mean.nii.gz", tmp_path / "sn.nii.gz", python=str(fake))
    assert mask.shape == (20, 20, 6) and mask.sum() == 18


def test_tse_ingestion_keeps_the_requested_echo_and_its_rigid_transform(tmp_path, monkeypatch):
    import json

    import SimpleITK as sitk
    from pie.imaging import nm

    def fake_convert(zip_path, prefix, out_dir, **kw):              # PPMI stores the two echoes as separate images
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        te = {"a/": 0.101, "b/": 0.012}[prefix]
        p = out / f"{prefix[0]}_e.nii.gz"
        nib.save(nib.Nifti1Image(np.full((8, 8, 4), 1000 * te, np.float32), np.diag([0.9, 0.9, 3.0, 1.0])), p)
        p.with_suffix("").with_suffix(".json").write_text(json.dumps({"EchoTime": te, "Manufacturer": "Siemens"}))
        return [str(p)]

    monkeypatch.setattr(nm, "convert", fake_convert)
    monkeypatch.setattr(nm, "register_slab", lambda img, fs, sampling_seed=0: (sitk.Euler3DTransform(), -0.5, "header", None, np.nan, 0))
    rows = [{"zip": "z", "prefix": p, "desc": "Axial PD-T2 TSE FS", "date": "2011-01-01"} for p in ("a/", "b/")]
    row = N.tse_subject(7, rows, str(tmp_path / "fs"), tmp_path, echo=1)
    assert np.asarray(nib.load(tmp_path / "7" / "nm_mean.nii.gz").dataobj).mean() == pytest.approx(12)   # short TE = echo 1
    assert (tmp_path / "7" / "slab_to_t1.tfm").exists() and row["tse_echo_te_s"] == 0.012 and row["manufacturer"] == "Siemens"
    N.tse_subject(8, rows, str(tmp_path / "fs"), tmp_path, echo=2)
    assert np.asarray(nib.load(tmp_path / "8" / "nm_mean.nii.gz").dataobj).mean() == pytest.approx(101)
    assert N.flag_tse(pd.DataFrame({"desc": ["Axial_PD-T2_TSE_FS", "AX_DUAL_TSE", "3D_T2_FLAIR", "2D_GRE-MT"]})).tolist() == [True, True, False, False]


def test_prepare_tse_creates_its_work_dir(tmp_path):
    import zipfile

    zp = tmp_path / "loni.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("PPMI/1/MPRAGE/2000-01-01_09_00_00.0/I000001/a.dcm", b"x")    # no TSE series: nothing to process
    pd.DataFrame({"patno": [1], "image_id": ["I000001"], "session_date": ["2000-01-01"]}).to_csv(tmp_path / "s.csv", index=False)
    work = tmp_path / "new" / "tse"
    N.main(["--prepare-tse", "--zips", str(zp), "--sessions", str(tmp_path / "s.csv"), "--fastsurfer-dir", str(tmp_path / "fs"),
            "--work-dir", str(work)])
    assert (work / "tse_index.csv").exists()


# ------------------------------------------------------------------------------------------ the study's measure (Methods 4.13)
def _bion():
    """A small Biondetti-like MNI image: territories 1-3 on each side, a background ROI of three parts, one touching the SN."""
    aff = np.eye(4)
    aff[:3, 3] = (-20, -30, -25)
    lab = np.zeros((41, 30, 20), np.int16)                   # MNI x = -20..20
    lab[8:14, 10:14, 8:12] = [[[1]], [[1]], [[2]], [[2]], [[3]], [[3]]]   # left: associative, limbic, sensorimotor along x
    lab[27:33, 10:14, 8:12] = [[[3]], [[3]], [[2]], [[2]], [[1]], [[1]]]
    lab[4:8, 10:14, 8:12] = 4                                # left crus, touching the left SN
    lab[33:37, 10:14, 8:12] = 4                              # right crus, touching the right SN
    lab[18:23, 2:5, 8:12] = 4                                # midline tegmental part
    return nib.Nifti1Image(lab, aff)


def test_study_labels_keep_the_complete_background_as_three_components():
    """The study's reference is the whole published background ROI: no ring is cut from it (unlike region_labels)."""
    codes = np.asarray(N.study_labels(_bion()).dataobj)
    region, side = codes % 100, codes // 100
    assert (region[8:10, 10:14, 8:12] == 1).all() and (side[8:14, 10:14, 8:12] == 1).all() and (side[27:33, 10:14, 8:12] == 2).all()
    assert (region[12:14, 10:14, 8:12] == 3).all() and (region[27:29, 10:14, 8:12] == 3).all()
    assert (codes[4:8, 10:14, 8:12] == N.STUDY_REF + 1).all()           # left crus, including the voxels next to the SN
    assert (codes[18:23, 2:5, 8:12] == N.STUDY_REF + 2).all()           # midline
    assert (codes[33:37, 10:14, 8:12] == N.STUDY_REF + 3).all()         # right crus
    assert set(np.unique(codes)) == {0, 101, 102, 103, 201, 202, 203, 11, 12, 13}


def _study_slab():
    """Codes on a slab grid (x, y, z) as study_labels + pull_labels give them, with a planted signal."""
    codes = np.zeros((40, 20, 8), np.int16)
    for x0, side in ((2, 100), (28, 200)):                   # each territory on each side, 4 x 10 x 4 voxels
        codes[x0:x0 + 4, 5:15, 2:6], codes[x0 + 4:x0 + 7, 5:15, 2:6], codes[x0 + 7:x0 + 10, 5:15, 2:6] = side + 1, side + 2, side + 3
    codes[14:18, 5:15, 2:6], codes[18:22, 5:15, 2:6], codes[22:26, 5:15, 2:6] = 11, 12, 13
    nm = np.where(codes // 100 == 1, 120.0, np.where(codes // 100 == 2, 110.0, 100.0))
    nm = nm + np.random.default_rng(0).normal(0, 1, nm.shape)
    cov = {k: 1.0 for k in ("whole_l", "whole_r", "associative_l", "associative_r", "limbic_l", "limbic_r",
                            "sensorimotor_l", "sensorimotor_r", "ref_1", "ref_2", "ref_3")}
    return nm, codes, np.ones(nm.shape), cov


def test_study_measure_is_the_contrast_ratio_against_the_whole_background():
    nm, codes, support, cov = _study_slab()
    out = N.study_features(nm, codes, support, cov, acquisition_ok=True)
    ref = nm[(codes > 10) & (codes < 20)].mean()
    assert out["nm_bion_qc_pass"] and out["nm_bion_qc_failed_checks"] == ""
    assert out["nm_bion_whole_l_cr"] == pytest.approx(nm[codes // 100 == 1].mean() / ref - 1)
    assert out["nm_bion_whole_r_cr"] == pytest.approx(nm[codes // 100 == 2].mean() / ref - 1)
    assert abs(out["nm_bion_whole_l_cr"] - 0.20) < 0.01 and abs(out["nm_bion_whole_mean_cr"] - 0.15) < 0.01
    assert out["nm_bion_associative_l_cr"] == pytest.approx(nm[codes == 101].mean() / ref - 1)
    assert out["nm_bion_sensorimotor_r_cr"] == pytest.approx(nm[codes == 203].mean() / ref - 1)
    codes[codes == 201] = 0                                   # no right associative voxels: that territory fails on both
    one_sided = N.study_features(nm, codes, support, cov, acquisition_ok=True)   # sides (the study never promotes a unilateral
    assert np.isnan(one_sided["nm_bion_associative_l_cr"]) and one_sided["nm_bion_qc_pass"]   # result); the whole nigra stands


@pytest.mark.parametrize("spoil, check", [
    (lambda nm, codes, sup, cov: sup.__setitem__(codes == 101, 0.8), "l_repeat_support"),
    (lambda nm, codes, sup, cov: nm.__setitem__((codes == 202) & (np.indices(nm.shape)[0] % 2 == 0), 0.0), "r_invalid_signal"),
    (lambda nm, codes, sup, cov: cov.__setitem__("whole_r", 0.85), "r_slab_coverage"),
    (lambda nm, codes, sup, cov: cov.__setitem__("ref_2", 0.85), "reference_component_slab_coverage"),
    (lambda nm, codes, sup, cov: nm.__setitem__(codes == 12, 400.0), "reference_cv"),
])
def test_study_measure_fails_the_studys_checks(spoil, check):
    nm, codes, support, cov = _study_slab()
    spoil(nm, codes, support, cov)
    out = N.study_features(nm, codes, support, cov, acquisition_ok=True)
    assert not out["nm_bion_qc_pass"] and check in out["nm_bion_qc_failed_checks"]
    assert np.isnan(out["nm_bion_whole_l_cr"]) and np.isnan(out["nm_bion_whole_mean_cr"])


def test_study_measure_needs_an_acceptable_acquisition():
    nm, codes, support, cov = _study_slab()
    out = N.study_features(nm, codes, support, cov, acquisition_ok=False)
    assert not out["nm_bion_qc_pass"] and "acquisition" in out["nm_bion_qc_failed_checks"]


def test_nigral_repeat_motion_is_the_largest_rms_displacement_of_the_nigral_points():
    import SimpleITK as sitk
    pts = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 5]])
    moves = [sitk.TranslationTransform(3, (1.0, 0, 0)), sitk.TranslationTransform(3, (0, 3.0, 4.0))]
    assert N.repeat_nigral_rms(pts, moves) == pytest.approx(5.0)
    assert N.repeat_nigral_rms(np.empty((0, 3)), moves) == np.inf


def test_syn_numerical_qc_measures_round_trip_jacobian_and_affine_determinant(syn, tmp_path):
    """The study's transform QC: round trip MNI -> T1 -> MNI at the nigral points, Jacobian of the warp there, and the
    affine's determinant; a real SyN passes, a reflected affine fails."""
    import ants
    import SimpleITK as sitk
    pts = nib.affines.apply_affine(syn["lab"].affine, np.argwhere(np.asarray(syn["lab"].dataobj) > 0)) * [-1, -1, 1]   # LPS mm
    ok = N.syn_numerical_qc(syn["warp"], syn["affine"], pts)
    assert ok["syn_roundtrip_mm_max"] < 0.5 and ok["syn_jacobian_min"] > 0 and ok["syn_affine_det"] > 0 and ok["syn_numerical_qc_pass"]
    zero = tmp_path / "t1_to_MNI152NLin2009cAsym_syn_1Warp.nii.gz"
    ants.image_write(ants.image_read(syn["warp"]) * 0, str(zero))
    sitk.WriteTransform(sitk.AffineTransform((-1.0, 0, 0, 0, 1, 0, 0, 0, 1), (0.0, 0, 0)), str(tmp_path / "flip.mat"))
    bad = N.syn_numerical_qc(str(zero), str(tmp_path / "flip.mat"), pts)
    assert bad["syn_roundtrip_mm_max"] < 0.01 and bad["syn_jacobian_min"] == pytest.approx(1.0) and bad["syn_affine_det"] < 0
    assert not bad["syn_numerical_qc_pass"]


def _study_subject(tmp_path, left_support=1.0):
    """A synthetic ``nm --keep-nifti`` subject whose slab (identity rigid, zero SyN) covers _bion()'s regions."""
    import json
    import SimpleITK as sitk

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nm import _zero_syn_cache

    fs, d = tmp_path / "fs", tmp_path / "work" / "1"
    _zero_syn_cache(fs)
    (d / "repeat_transforms").mkdir(parents=True)
    aff = np.diag([0.5, 0.5, 1.5, 1.0])
    aff[:3, 3] = (-20, -30, -25)                             # RAS x -20..19.5, y -30..-0.5, z -25..-5.5
    shape = (80, 60, 14)
    nib.save(nib.Nifti1Image((100 + np.random.default_rng(0).normal(0, 2, shape)).astype(np.float32), aff), d / "nm_mean.nii.gz")
    support = np.ones(shape, np.float32)
    support[:40] = left_support                              # RAS x < 0: the left hemisphere
    nib.save(nib.Nifti1Image(support, aff), d / "repeat_transforms" / "repeat_support_fraction.nii.gz")
    sitk.WriteTransform(sitk.Euler3DTransform(), str(d / "slab_to_t1.tfm"))
    for i in (1, 2):
        sitk.WriteTransform(sitk.Euler3DTransform((0, 0, 0), 0, 0, 0, (0.5, 0, 0)), str(d / "repeat_transforms" / f"repeat_{i}_to_anchor.tfm"))
    (d / "repeat_selection.json").write_text(json.dumps({"volumes": [{}, {}, {}], "n_missing_acquisition_identity": 0}))
    return tmp_path / "work", fs


def test_study_subject_joins_regions_support_repeats_and_transform_checks(tmp_path):
    work, fs = _study_subject(tmp_path)
    out = N.study_subject(work, 1, fs, labels=N.study_labels(_bion()))
    assert out["nm_bion_qc_pass"], out["nm_bion_qc_failed_checks"]
    assert abs(out["nm_bion_whole_l_cr"]) < 0.02 and out["nm_bion_n_sn_l"] >= 20 and out["nm_bion_cov_ref_2"] > 0.99
    assert out["syn_numerical_qc_pass"] and out["repeat_nigral_rms_mm_max"] == pytest.approx(0.5)
    assert (work / "1" / "study_regions_nm.nii.gz").exists()
    work, fs = _study_subject(tmp_path / "b", left_support=0.8)     # the support map's left half, read in the slab's own order
    out = N.study_subject(work, 1, fs, labels=N.study_labels(_bion()))
    assert "l_repeat_support" in out["nm_bion_qc_failed_checks"] and "r_repeat_support" not in out["nm_bion_qc_failed_checks"]



def test_study_subject_refuses_a_denoised_slab(tmp_path):
    # the study's quantity is defined without MP-PCA; nm records its denoising in nm_features.csv
    import pandas as pd
    work, fs = _study_subject(tmp_path)
    pd.DataFrame({"patno": [1], "denoise": ["mppca"]}).to_csv(work / "nm_features.csv", index=False)
    with pytest.raises(ValueError, match="without --denoise"):
        N.study_subject(work, 1, fs, labels=N.study_labels(_bion()))
    pd.DataFrame({"patno": [1], "denoise": ["none"]}).to_csv(work / "nm_features.csv", index=False)
    assert N.study_subject(work, 1, fs, labels=N.study_labels(_bion()))["nm_bion_qc_pass"]

# ------------------------------------------------------------------------------------------ Langley version 2 (peduncle reference)
def _peduncle(like):
    """A JHU-like cerebral peduncle on _bion()'s grid: lateral to each nigra, touching it, over its crus part."""
    ped = np.zeros(like.shape, np.int16)
    ped[3:8, 10:14, 6:14] = 16                               # left peduncle, up to the left nigra (x index 8)
    ped[33:38, 10:14, 6:14] = 15                             # right, from the right nigra's edge (x index 32)
    return nib.Nifti1Image(ped, like.affine)


def test_langley_labels_swap_the_crus_reference_for_the_peduncle_outside_the_search_region():
    bion = _bion()
    codes = np.asarray(N.langley_labels(bion, _peduncle(bion)).dataobj)
    region, side = codes % 100, codes // 100
    assert (region[8:14, 10:14, 8:12] > 0).all() and (region[8:14, 10:14, 8:12] <= 3).all()   # the nigra keeps its territories
    assert (region[3:7, 10:14, 6:14] == N.PEDUNCLE).all() and (side[3:7, 10:14, 6:14] == 1).all()
    assert (region[34:38, 10:14, 6:14] == N.PEDUNCLE).all() and (side[34:38, 10:14, 6:14] == 2).all()
    assert not np.isin(region, [N.CRUS, N.TEGMENTUM]).any()                                    # no Biondetti reference left
    assert region[7, 11, 9] == N.RING and region[33, 11, 9] == N.RING   # the search ring wins over the peduncle next to the nigra


def test_langley_volume_can_use_the_peduncle_as_its_reference():
    codes = np.zeros((30, 20, 6), np.int16)
    codes[5:10, 5:15, 1:5] = 101                             # left nigra: 200 voxels
    codes[20:25, 5:15, 1:5] = 201
    codes[0:4, 5:15, 1:5] = 100 + N.PEDUNCLE                 # peduncle reference: 160 voxels
    codes[26:30, 5:15, 1:5] = 200 + N.PEDUNCLE
    codes[12:18, 5:15, 1:5] = 100 + N.CRUS                   # a bright "crus" that must not be the reference now
    rng = np.random.default_rng(0)
    nm = np.where(codes % 100 == N.PEDUNCLE, 100 + rng.normal(0, 5, codes.shape), 100.0)
    nm[codes % 100 == N.CRUS] = 300.0
    nm[(codes == 101) & (np.indices(codes.shape)[1] < 10)] = 200.0     # half of the left nigra is bright
    img = nib.Nifti1Image(nm.astype(np.float32), np.diag([0.5, 0.5, 1.0, 1.0]))
    out = N.langley_features(img, codes, ref_region=N.PEDUNCLE, prefix="nmlp_")
    assert out["nmlp_sn_volume_l_mm3"] == pytest.approx(100 * 0.25) and out["nmlp_sn_volume_r_mm3"] == 0
    assert abs(out["nmlp_ref_mean"] - 100) < 1.5 and abs(out["nmlp_threshold"] - (out["nmlp_ref_mean"] + 2.8 * out["nmlp_ref_sd"])) < 1e-6


def test_peduncle_bridge_qc_wants_the_peduncle_lateral_to_the_nigra_and_apart_from_it():
    from pie.imaging.atlases import peduncle_bridge_qc
    bion = _bion()
    lab = np.asarray(bion.dataobj)
    ok = peduncle_bridge_qc(np.asarray(_peduncle(bion).dataobj), lab, bion.affine)
    assert ok["pass"], ok
    medial = np.zeros(lab.shape, np.int16)
    medial[16:20, 10:14, 6:14], medial[21:25, 10:14, 6:14] = 16, 15     # medial to the nigra: the wrong structure
    assert not peduncle_bridge_qc(medial, lab, bion.affine)["pass"]
    off = np.zeros(lab.shape, np.int16)
    off[3:8, 0:4, 6:14], off[33:38, 0:4, 6:14] = 16, 15     # lateral, but nowhere near the crus: a misplaced bridge
    assert not peduncle_bridge_qc(off, lab, bion.affine)["pass"]
    over = np.asarray(_peduncle(bion).dataobj).copy()
    over[8:10, 10:14, 8:12] = 16                             # JHU's coarse label spills over the nigra's lateral edge:
    q = peduncle_bridge_qc(over, lab, bion.affine)           # reported, not a failure (the search region wins in langley_labels)
    assert q["pass"] and q["l_sn_overlap"] > 0.3


def test_process_subject_adds_the_peduncle_version_beside_version_1(tmp_path, monkeypatch):
    work, fs = _study_subject(tmp_path)
    bion = _bion()
    monkeypatch.setattr(N, "region_labels", lambda *a: _real_region(bion))    # the synthetic atlas, not the real one
    monkeypatch.setattr(N, "langley_labels", lambda *a: _real_langley(bion, _peduncle(bion)))
    row = N.process_subject(work, 1, fs, langley_peduncle=True)
    assert {"nml_sn_volume_mm3", "nmlp_sn_volume_mm3", "nmlp_threshold", "native_cov_peduncle_l"} <= set(row)
    assert row["native_cov_peduncle_l"] > 0.99 and np.isfinite(row["nmlp_sn_volume_l_mm3"]) and np.isfinite(row["nmlp_ref_sd"])
    assert (work / "1" / "langley_regions_nm.nii.gz").exists()


_real_region, _real_langley = N.region_labels, N.langley_labels
