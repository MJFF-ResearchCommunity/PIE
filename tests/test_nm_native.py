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

    def fake_convert(zip_path, prefix, out_dir):              # PPMI stores the two echoes as separate images
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
