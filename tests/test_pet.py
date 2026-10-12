"""Synthetic tests for the AV-133 PET module: frame timing from DICOM headers, time weighting, choice of series, LONI
zip -> dcm2niix ingest, an end-to-end phantom (two frames, known uptake, a known rigid misalignment and inter-frame
motion) to recovered SBRs, PPMI labels kept apart by tracer and pipeline, and the agreement statistics. No PPMI data."""

import io
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging import datscan, pet
from pie.imaging.archives import ZipArchiveCache
from pie.imaging.convert import DCM2NIIX
from pie.imaging.labels import pet_labels


# ------------------------------------------------------------------------------------------ DICOM helpers
def _hdr(series="1.2.3", acq="092000", frt=150000.0, dur=300000, inj="080000", z=0, image_index=1, rows=4, slope=1.0):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"           # PET Image Storage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.Manufacturer, ds.SeriesInstanceUID, ds.StudyInstanceUID = "PT", "SIEMENS", series, "1.2.3.9"
    ds.FrameOfReferenceUID, ds.PatientID, ds.PatientWeight = "1.2.3.8", "0", 70
    ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = "20200101"
    ds.SeriesTime, ds.AcquisitionTime = "092000", acq
    ds.FrameReferenceTime, ds.ActualFrameDuration, ds.ImageIndex, ds.InstanceNumber = frt, dur, image_index, image_index
    ds.ImagePositionPatient, ds.ImageOrientationPatient = [-rows, -rows, 2.0 * z], [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing, ds.SliceThickness, ds.Rows, ds.Columns = [2.0, 2.0], 2.0, rows, rows
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.RescaleSlope, ds.RescaleIntercept = slope, 0
    ds.Units, ds.DecayCorrection, ds.SeriesType, ds.CorrectedImage = "BQML", "START", ["DYNAMIC", "IMAGE"], ["ATTN", "SCAT", "DECY", "RAN"]
    ds.ReconstructionMethod = "OSEM3D 4i8s"
    if inj:
        r = Dataset()
        r.Radiopharmaceutical, r.RadiopharmaceuticalStartTime = "18F-AV-133", inj
        r.RadionuclideTotalDose, r.RadionuclideHalfLife = 222e6, 6586.2
        ds.RadiopharmaceuticalInformationSequence = [r]
    ds.PixelData = np.full((rows, rows), 100 + 10 * z, np.uint16).tobytes()
    return ds


def _dynamic(series="1.2.3", nz=3, rows=4, **kw):
    """Two 5-min frames from 80 min after an 08:00 injection, ``nz`` slices each, shuffled."""
    hs = [_hdr(series, acq=acq, frt=frt, z=z, image_index=f * nz + z + 1, rows=rows, slope=f + 1.0, **kw)
          for f, (acq, frt) in enumerate((("092000", 150000.0), ("092500", 450000.0))) for z in range(nz)]
    return hs[::-1]


# ------------------------------------------------------------------------------------------ frames
def test_frame_times_from_headers_across_vendor_conventions():
    f = pet.series_frames(_dynamic())
    assert f["start_s"].tolist() == [4800.0, 5100.0] and f["duration_s"].tolist() == [300.0, 300.0] and (f["n_slices"] == 3).all()
    same_acq = [_hdr(acq="092000", frt=frt, z=z) for frt in (150000.0, 450000.0) for z in range(2)]    # series start on every frame
    assert pet.series_frames(same_acq)["start_s"].tolist() == [4800.0, 5100.0]
    statics = [_hdr(series="1.1", acq="092000"), _hdr(series="1.2", acq="092500")]                          # one series per frame
    assert pet.series_frames(statics)["start_s"].tolist() == [4800.0, 5100.0]
    assert pet.series_frames([_hdr(inj=None)])["start_s"].isna().all()
    with pytest.raises(ValueError, match="overlap"):                                                     # a 10-min sum next to its frames
        pet.series_frames(statics + [_hdr(series="1.3", acq="092000", dur=600000)])


def test_time_weighted_mean_and_analysis_frames():
    data = np.stack([np.full((4, 4, 4), 1.0), np.full((4, 4, 4), 4.0)], axis=-1)
    assert np.allclose(pet.time_weighted_mean(data, [60, 240]), (60 * 1 + 240 * 4) / 300)
    assert np.allclose(pet.time_weighted_mean(data, [np.nan, 240]), 2.5)        # unknown duration: equal weights
    early = pd.DataFrame({"start_s": [600.0, 4800.0], "duration_s": [300.0, 300.0]})   # an early frame is not an analysis frame
    mean, info = pet.mean_image(data, early, np.eye(4))
    assert np.allclose(mean, 4.0) and info["n_frames"] == 2 and info["n_frames_used"] == 1 and info["start_min"] == 80
    assert info["timing_ok"] is True and np.isnan(info["motion_mm"])


# ------------------------------------------------------------------------------------------ ingest
def _index_rows(**over):
    base = dict(zip="z.zip", prefix="p/", patno=1, desc="AV133", date="2020-01-01", image_id="I1", n_files=150,
                modality="PT", tracer="18F-AV-133", n_frames=2, units="BQML", kernel="", image_type="ORIGINAL")
    return {**base, **over}


def test_select_series_prefers_unfiltered_multiframe_and_stacks_statics():
    idx = pd.DataFrame([
        _index_rows(image_id="I1", desc="AV133 Gauss 4mm"), _index_rows(image_id="I2", desc="AV133 no filter"),
        _index_rows(image_id="I3", modality="CT", desc="CTAC"), _index_rows(image_id="I4", tracer="DPA-714", n_frames=4),
        _index_rows(image_id="I5", date="2021-01-01", n_frames=1, n_files=75, desc="AV133 frame 1"),
        _index_rows(image_id="I6", date="2021-01-01", n_frames=1, n_files=75, desc="AV133 frame 2"),
        _index_rows(image_id="I7", date="2021-01-01", n_frames=1, n_files=75, kernel="XYZ Gauss2.00")])
    sel = pet.select_series(idx)
    assert set(sel.loc[sel["selected"], "image_id"]) == {"I2", "I5", "I6"}
    assert set(pet.select_series(idx, prefer="filtered").loc[lambda d: d["selected"], "image_id"]) == {"I1", "I7"}


def _loni_zip(path, patno=9000001, date="2020-01-01_09_20_00.0"):
    """A LONI-layout zip: an unfiltered and a post-filtered 2-frame AV-133 series and a CT, 8 x 8 x 4 voxels."""
    with zipfile.ZipFile(path, "w") as z:
        for image_id, desc, kernel in (("I0000001", "AV133_NoFilter", "All-pass"), ("I0000002", "AV133_Gauss", "XYZ Gauss4.00")):
            for k, ds in enumerate(_dynamic(series=generate_uid(), nz=4, rows=8)):
                ds.SeriesDescription, ds.ConvolutionKernel = desc, kernel
                buf = io.BytesIO()
                pydicom.dcmwrite(buf, ds, enforce_file_format=True)
                z.writestr(f"PPMI/{patno}/{desc}/{date}/{image_id}/{k:03d}.dcm", buf.getvalue())
        ct = _hdr(series=generate_uid())
        ct.Modality = "CT"
        buf = io.BytesIO()
        pydicom.dcmwrite(buf, ct, enforce_file_format=True)
        z.writestr(f"PPMI/{patno}/CTAC/{date}/I0000003/000.dcm", buf.getvalue())


@pytest.mark.skipif(not Path(DCM2NIIX).exists(), reason="dcm2niix not installed")
def test_loni_zip_to_frames_and_pet_bids_sidecar(tmp_path):
    _loni_zip(tmp_path / "pet.zip")
    idx = pet.select_series(pet.index_pet([tmp_path / "pet.zip"], tmp_path / "pet_index.csv"))
    assert (tmp_path / "pet_index.csv").exists() and len(idx) == 3
    rows = idx[idx["selected"]].to_dict("records")
    assert [r["image_id"] for r in rows] == ["I0000001"] and rows[0]["postfilter"] == "unfiltered"
    with ZipArchiveCache() as cache:
        data, frames, affine, side, _ = pet.load_scan(rows, tmp_path / "work", cache)
    assert data.shape == (8, 8, 4, 2) and frames["start_s"].tolist() == [4800.0, 5100.0]
    assert np.isclose(data[..., 1].mean() / data[..., 0].mean(), 2.0)        # per-frame rescale slope kept
    assert side["TracerName"] == "18F-AV-133" and side["Units"] == "Bq/mL" and side["DecayCorrection"] == "START"
    assert side["InjectedRadioactivity"] == 222 and side["FrameTimesStart"] == [4800.0, 5100.0] and side["FrameDuration"] == [300.0, 300.0]
    assert side["TimeZero"] == "08:00:00" and side["ScanStart"] == 4800.0 and side["ReconMethodName"]


@pytest.mark.skipif(not Path(DCM2NIIX).exists(), reason="dcm2niix not installed")
def test_an_early_static_is_left_out_and_another_grid_is_resampled(tmp_path):
    """2012-13 visits: separate ~50 and ~80 min acquisitions, the participant repositioned in between."""
    with zipfile.ZipFile(tmp_path / "pet.zip", "w") as z:
        for image_id, desc, early in (("I0000001", "EM50", True), ("I0000002", "EM80", False)):
            for k, ds in enumerate(_dynamic(series=generate_uid(), nz=4, rows=8)):
                if early:
                    ds.AcquisitionTime = {"092000": "085000", "092500": "085500"}[ds.AcquisitionTime]
                    ds.ImagePositionPatient = [v + 10.0 for v in ds.ImagePositionPatient]
                ds.SeriesDescription = desc
                buf = io.BytesIO()
                pydicom.dcmwrite(buf, ds, enforce_file_format=True)
                z.writestr(f"PPMI/9000001/{desc}/2013-01-01_09_20_00.0/{image_id}/{k:03d}.dcm", buf.getvalue())
    rows = pet.index_pet([tmp_path / "pet.zip"]).sort_values("image_id").to_dict("records")
    with ZipArchiveCache() as cache:
        data, frames, _, side, _ = pet.load_scan(rows, tmp_path / "late", cache, from_min=pet.ANALYSIS_FROM_MIN)
        assert data.shape == (8, 8, 4, 2) and frames["start_s"].tolist() == [4800.0, 5100.0] and side["PIERegriddedSeries"] == 0
        data, frames, _, side, _ = pet.load_scan(rows, tmp_path / "all", cache)          # no timing rule: both, on the first grid
    assert data.shape == (8, 8, 4, 4) and frames["start_s"].tolist() == [3000.0, 3300.0, 4800.0, 5100.0] and side["PIERegriddedSeries"] == 1
    assert data[..., :2].min() > 0 and data[..., 2:].min() == 0           # the later series, moved 10 mm, covers only part of the first grid


# ------------------------------------------------------------------------------------------ phantom end to end
TRUTH = {"caudate_l": 3.0, "caudate_r": 2.6, "putamen_l_ant": 3.5, "putamen_l_post": 2.0, "putamen_r_ant": 2.5, "putamen_r_post": 1.4}
OCC, CB = 1.0, 1.25


def _phantom(shape=(72, 88, 72), vox=2.0):
    """T1-like image, FastSurfer-style labels and the true AV-133 uptake on one centred RAS grid."""
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = -np.array(shape) * vox / 2
    x, y, z = nib.affines.apply_affine(aff, np.indices(shape).reshape(3, -1).T).T.reshape(3, *shape)
    ell = lambda c, r: ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2 <= 1
    left = x < 0
    head, brain, inner = ell((0, 0, 0), (66, 82, 66)), ell((0, 0, 4), (58, 74, 54)), ell((0, 0, 4), (50, 66, 46))
    lab = np.zeros(shape, np.int16)
    lab[brain & ~inner] = np.where(left, 1028, 2028)[brain & ~inner]
    occ = brain & ~inner & (y < -40)
    lab[occ] = np.where(left, datscan.OCCIPITAL[0], datscan.OCCIPITAL[4])[occ]
    lab[inner] = np.where(left, 2, 41)[inner]
    cb = ell((-22, -48, -36), (18, 16, 12)) | ell((22, -48, -36), (18, 16, 12))
    lab[cb] = np.where(left, 8, 47)[cb]
    up = np.where(head, 0.3, 0.0).astype(np.float32)
    up[np.isin(lab, [1028, 2028])], up[occ], up[np.isin(lab, [2, 41])], up[cb] = 1.1, OCC, 0.7, CB
    t1 = np.where(head, 0.8, 0.0).astype(np.float32)
    t1[lab > 0], t1[np.isin(lab, [2, 41])], t1[cb] = 0.6, 1.0, 0.65
    yi = np.indices(shape)[1]
    for side, sx in (("l", -1), ("r", 1)):
        caud = (abs(x - sx * 14) < 8) & (y > 8) & (y < 30) & (abs(z - 14) < 10)
        put = (abs(x - sx * 28) < 7) & (y > -16) & (y < 24) & (abs(z - 2) < 10)
        lab[caud], lab[put] = (11, 12) if side == "l" else (50, 51)
        ant = put & (yi >= np.median(yi[put]))                               # the split rule of datscan._ap_split
        up[caud], up[put & ~ant], up[ant] = TRUTH[f"caudate_{side}"], TRUTH[f"putamen_{side}_post"], TRUTH[f"putamen_{side}_ant"]
        t1[caud | put] = 0.7
    return nib.Nifti1Image(t1, aff), nib.Nifti1Image(lab, aff), nib.Nifti1Image(up, aff)


def _acquire(uptake, rotation, translation, shape=(60, 72, 60), vox=2.5):
    """The uptake seen on a PET grid whose points map to phantom points through a known rigid transform."""
    import SimpleITK as sitk

    from pie.imaging.dwi import _sitk_native

    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = -np.array(shape) * vox / 2
    tx = sitk.Euler3DTransform()
    tx.SetRotation(*np.deg2rad(rotation))
    tx.SetTranslation(translation)
    grid = _sitk_native(np.zeros(shape, np.float32), aff)
    return np.transpose(sitk.GetArrayFromImage(sitk.Resample(datscan._sitk_from_nib(uptake), grid, tx, sitk.sitkLinear, 0.0)), (2, 1, 0)), aff


def test_phantom_recovers_sbr_through_misalignment_and_motion():
    t1, lab, up = _phantom()
    f1, aff = _acquire(up, (5.0, -4.0, 6.0), (8.0, -6.0, 10.0))
    f2, _ = _acquire(up, (5.0, -4.0, 6.0), (11.0, -6.0, 10.0))                 # the head moves 3 mm between frames
    frames = pd.DataFrame({"start_s": [4800.0, 4900.0], "duration_s": [100.0, 500.0]})
    mean, info = pet.mean_image(np.stack([f1, f2], axis=-1), frames, aff)
    assert info["n_frames_used"] == 2 and abs(info["motion_mm"] - 3.0) < 0.75 and info["motion_corrected"]
    out, _ = pet.quantify(nib.Nifti1Image(mean, aff), t1, lab)
    assert out["reg_template_r"] > 0.4 and out["cov_striatum"] == 1.0 and out["n_cerebellum"] > 1000
    for region, v in TRUTH.items():                                          # within 10 % (+0.05): interpolation blurs ROI edges
        for col, truth in ((f"sbr_{region}", v / OCC - 1), (f"sbrcb_{region}", v / CB - 1)):
            assert abs(out[col] - truth) < 0.1 * truth + 0.05, (col, out[col], truth)
    assert abs(out["cb_occ"] - (CB / OCC - 1)) < 0.03
    assert out["sbr_putamen_l_post"] < out["sbr_putamen_l_ant"] and out["sbr_putamen_r"] < out["sbr_putamen_l"]
    assert not any(k.startswith("sbrwm_") for k in out)


# ------------------------------------------------------------------------------------------ PPMI labels / agreement
def _ppmi_tables(root):
    """PET_SBR_Analysis-like rows: legacy IND (blank ligand), protocol-004 six VOIs, sub-regional AV-133, DPA-714."""
    img = root / "Imaging"
    img.mkdir(parents=True)
    sbr = lambda v: {c: v for c in pet.PPMI_COLUMNS.values()}
    rows = [dict(PROTOCOL="001", PATNO=1, EVENT_ID="BL", PET_LIGAND=None, PET_SCAN_DATE="03/2013", PET_ANALYZED="Yes", **sbr(1.5)),
            dict(PROTOCOL="004", PATNO=2, EVENT_ID="BL", PET_LIGAND="18F-AV-133", PET_SCAN_DATE="05/2021", PET_ANALYZED="Yes", **sbr(1.4)),
            dict(PROTOCOL="004", PATNO=2, EVENT_ID="V04", PET_LIGAND="18F-AV-133", PET_SCAN_DATE="05/2022", PET_ANALYZED="No"),
            dict(PROTOCOL="015", PATNO=3, EVENT_ID="BL", PET_LIGAND="18F-AV-133", PET_SCAN_DATE="01/2024", PET_ANALYZED="Yes",
                 PET_CBM=0.25, PET_OCCIP=0.0, **sbr(2.0)),
            dict(PROTOCOL="015", PATNO=4, EVENT_ID="BL", PET_LIGAND="18F-AV-133", PET_SCAN_DATE="02/2024", PET_ANALYZED="No"),
            dict(PROTOCOL="019", PATNO=5, EVENT_ID="BL", PET_LIGAND="18F-DPA-714", PET_SCAN_DATE="06/2024", PET_ANALYZED="Yes",
                 PET_CBM=0.05, PET_OCCIP=0.0, **sbr(0.1))]
    pd.DataFrame(rows).reindex(columns=["PROTOCOL", "PATNO", "EVENT_ID", "PET_LIGAND", "PET_SCAN_DATE", *pet.PPMI_COLUMNS.values(),
                                        "PET_CBM", "PET_OCCIP", "PET_ANALYZED"]).to_csv(img / "PET_SBR_Analysis_01Jan2026.csv", index=False)
    pd.DataFrame([dict(PROTOCOL=r["PROTOCOL"], PATNO=r["PATNO"], EVENT_ID=r["EVENT_ID"], PET_SCAN_DATE=r["PET_SCAN_DATE"],
                       PET_LIGAND=r["PET_LIGAND"] or "18F-AV-133") for r in rows]).to_csv(img / "PET_Acquisition_Metadata_01Jan2026.csv", index=False)


def test_pet_labels_keep_tracer_and_pipeline_apart(tmp_path):
    _ppmi_tables(tmp_path)
    t = pet_labels(tmp_path).set_index(["PATNO", "EVENT_ID"])
    assert t.loc[(1, "BL"), "tracer"] == "AV-133" and t.loc[(1, "BL"), "pipeline"] == "ind_2014"        # ligand from the acquisition table
    assert t.loc[(2, "BL"), "pipeline"] == "core_lab_6voi" and not t.loc[(2, "V04"), "analysed"]
    assert t.loc[(3, "BL"), "pipeline"] == t.loc[(4, "BL"), "pipeline"] == "subregional"              # the unanalysed 015 row too
    assert t.loc[(5, "BL"), "tracer"] == "DPA-714" and t.loc[(5, "BL"), "pipeline"] == "subregional"
    assert t.loc[(3, "BL"), "PET_SCAN_DATE"] == pd.Timestamp("2024-01-01") and t.loc[(3, "BL"), "sbr_putamen_l_post"] == 2.0
    assert np.isclose(t.loc[(3, "BL"), "sbrcb_putamen_l_post"], 3.0 / 1.25 - 1) and np.isnan(t.loc[(2, "BL"), "sbrcb_caudate_l"])
    (tmp_path / "Imaging" / "PET_Acquisition_Metadata_01Jan2026.csv").unlink()
    assert pd.isna(pet_labels(tmp_path).set_index("PATNO").loc[1, "tracer"])                         # unknown, never assumed


def test_agreement_statistics_on_known_data():
    # Shrout & Fleiss (1979), 6 targets x 4 judges: ICC(2,1) = 0.29
    sf = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float)
    assert round(pet._icc21(sf), 2) == 0.29
    x = np.array([1.0, 2.0, 3.0, 4.0])
    a = pet.agreement(x, x)
    assert a["n"] == 4 and np.isclose(a["icc21"], 1) and np.isclose(a["ccc"], 1) and a["bias"] == 0 and a["loa_low"] == a["loa_high"] == 0
    b = pet.agreement(x + 1, x)                                                # PIE reads 1 higher
    assert np.isclose(b["pearson_r"], 1) and np.isclose(b["ccc"], 2.5 / 3.5) and b["bias"] == 1 and b["icc21"] < 1
    rng = np.random.default_rng(0)
    u = rng.normal(2, 0.5, 4000)
    c = pet.agreement(0.9 * u + rng.normal(0, 0.2, 4000) + 0.1, u)
    assert abs(c["icc21"] - c["ccc"]) < 0.01                                    # the two coincide for large n
    assert pet.agreement([1, np.nan], [1, 2]) == {"n": 1}


def test_agreement_table_matches_scan_months_and_never_pools(tmp_path):
    _ppmi_tables(tmp_path)
    labels = pet_labels(tmp_path)
    base = {"error": "", "units_ok": True, "reg_template_r": 0.8, "cov_striatum": 1.0, "cov_occipital": 1.0, "cov_cerebellum": 1.0,
            "motion_mm": 1.0, **{f"sbr_{r}": 1.0 for r in pet.REGIONS}, **{f"sbrcb_{r}": 0.5 for r in pet.REGIONS}, "cb_occ": 0.3}
    pie = pd.DataFrame([dict(base, patno=p, acquisition_date=d) for p, d in
                        [(1, "2013-03-14"), (1, "2014-03-14"), (2, "2021-05-02"), (3, "2024-01-20"), (5, "2024-06-03")]] * 3)
    pie.loc[pie.index % 3 == 1, [f"sbr_{r}" for r in pet.REGIONS]] = 1.2         # vary values so statistics exist
    pie.loc[pie.index % 3 == 2, [f"sbr_{r}" for r in pet.REGIONS]] = 0.8
    t = pet.agreement_table(pie, labels)
    occ = t[(t["reference"] == "occipital") & (t["region"] == "putamen_l_post")].set_index("pipeline")
    assert set(occ.index) == {"ind_2014", "core_lab_6voi", "subregional"} and (occ["n"] == 3).all()      # DPA-714 never joins
    assert set(t.loc[t["reference"] == "cerebellum", "pipeline"]) == {"subregional"}
    assert ((t["pipeline"] == "subregional") & (t["region"] == "cerebellum")).sum() == 1
    pie.loc[0, "reg_template_r"] = 0.1                                          # a failed registration leaves the comparison
    assert pet.agreement_table(pie, labels).set_index(["pipeline", "reference", "region"]).loc[("ind_2014", "occipital", "caudate_l"), "n"] == 2


def _write_frames(z, prefix, frames, aff):
    """PET frames (x, y, z, t; RAS affine with positive diagonal) as one LONI-layout DICOM series, Bq/mL x 1000."""
    nx, ny, nz, nt = frames.shape
    vox, (x0, y0, z0) = aff[0, 0], aff[:3, 3]
    series = generate_uid()
    for t, (acq, frt) in enumerate((("092000", 150000.0), ("092500", 450000.0))[:nt]):
        for k in range(nz):
            ds = _hdr(series, acq=acq, frt=frt, z=k, image_index=t * nz + k + 1, rows=nx)
            ds.Rows, ds.Columns, ds.PixelSpacing, ds.SliceThickness = ny, nx, [vox, vox], vox
            ds.ImagePositionPatient = [-(x0 + vox * (nx - 1)), -(y0 + vox * (ny - 1)), z0 + vox * k]    # LPS of pixel (0, 0)
            ds.SeriesDescription, ds.NumberOfSlices, ds.NumberOfTimeSlices = "AV133 Brain Dynamic", nz, nt
            ds.PixelData = np.ascontiguousarray(np.round(frames[::-1, ::-1, k, t].T * 1000)).astype(np.uint16).tobytes()
            buf = io.BytesIO()
            pydicom.dcmwrite(buf, ds, enforce_file_format=True)
            z.writestr(f"{prefix}{t * nz + k:04d}.dcm", buf.getvalue())


@pytest.mark.skipif(not Path(DCM2NIIX).exists(), reason="dcm2niix not installed")
def test_cli_end_to_end_on_phantom_dicom_with_resume_and_agreement(tmp_path):
    t1, lab, up = _phantom()
    f1, aff = _acquire(up, (5.0, -4.0, 6.0), (8.0, -6.0, 10.0))
    f2, _ = _acquire(up, (5.0, -4.0, 6.0), (10.0, -6.0, 10.0))
    with zipfile.ZipFile(tmp_path / "av133.zip", "w") as z:
        _write_frames(z, "PPMI/9000001/AV133_Brain_Dynamic/2020-01-01_09_20_00.0/I0000001/", np.stack([f1, f2], axis=-1), aff)
    subject = tmp_path / "fastsurfer" / "I0000009"
    (subject / "mri").mkdir(parents=True)
    (subject / "stats").mkdir()
    (subject / "stats" / "aseg+DKT.stats").write_text("# synthetic\n")
    nib.save(nib.MGHImage(t1.get_fdata().astype(np.float32), t1.affine), subject / "mri" / "orig.mgz")
    nib.save(nib.MGHImage(np.asarray(lab.dataobj).astype(np.int32), lab.affine), subject / "mri" / "aparc.DKTatlas+aseg.deep.mgz")
    pd.DataFrame({"patno": [9000001], "image_id": ["I0000009"], "session_date": ["2019-12-01"]}).to_csv(tmp_path / "sessions.csv", index=False)
    args = ["--zips", str(tmp_path / "av133.zip"), "--sessions", str(tmp_path / "sessions.csv"), "--fastsurfer-dir",
            str(tmp_path / "fastsurfer"), "--work-dir", str(tmp_path / "pet"), "--workers", "1"]
    pet.main(args)
    out = pd.read_csv(tmp_path / "pet" / "pet_av133.csv")
    r = out.iloc[0]
    assert len(out) == 1 and pd.isna(r["error"]) and r["image_id"] == "I0000001" and r["acquisition_date"] == "2020-01-01"
    assert r["start_min"] == 80 and r["n_frames_used"] == 2 and r["units"] == "Bq/mL" and r["tracer_av133"] and r["fs_image_id"] == "I0000009"
    assert abs(r["sbr_putamen_l_post"] - 1.0) < 0.15 and abs(r["sbr_caudate_r"] - 1.6) < 0.25 and pet.qc_pass(out).all()
    assert not (tmp_path / "pet" / "9000001" / "2020-01-01").exists()          # no --keep-nifti: no images left behind
    pet.main(args)                                                             # resume: the finished scan is not redone
    assert len(pd.read_csv(tmp_path / "pet" / "pet_av133.csv")) == 1
    _ppmi_tables(tmp_path / "PPMI")
    table = tmp_path / "PPMI" / "Imaging" / "PET_SBR_Analysis_01Jan2026.csv"
    t = pd.read_csv(table)
    pd.concat([t, t.iloc[[1]].assign(PATNO=9000001, PET_SCAN_DATE="01/2020")]).to_csv(table, index=False)
    pet.main(args + ["--agreement", str(tmp_path / "PPMI")])
    agree = pd.read_csv(tmp_path / "pet" / "pet_agreement.csv")
    assert set(agree["pipeline"]) == {"core_lab_6voi"} and (agree["n"] == 1).all()


def test_sv2a_and_tau_suvr_against_their_reference_regions(tmp_path):
    t1, lab, up = _phantom()
    f, aff = _acquire(up, (4.0, -3.0, 5.0), (6.0, -4.0, 8.0))
    out, _ = pet.quantify_suvr(nib.Nifti1Image(f, aff), t1, lab, "cs")              # SV2A: centrum semiovale (deep WM 0.7)
    assert out["cov_reference"] == 1.0 and out["n_reference"] > 1000
    assert abs(out["suvr_superiorfrontal_l"] - 1.1 / 0.7) < 0.12 and abs(out["suvr_caudate_l"] - TRUTH["caudate_l"] / 0.7) < 0.15 * TRUTH["caudate_l"] / 0.7
    assert np.isnan(out["suvr_braak6"])                                                # no such labels in the phantom
    tau, _ = pet.quantify_suvr(nib.Nifti1Image(f, aff), t1, lab, "cerebellum_ventral")
    assert abs(tau["suvr_superiorfrontal_l"] - 1.1 / CB) < 0.12
    sel = pet.select_series(pd.DataFrame([_index_rows(image_id="I1", desc="SV2A_PET_DYNAMIC"), _index_rows(image_id="I2", desc="AV133")]), tracer="sv2a")
    assert sel.loc[sel["selected"], "image_id"].tolist() == ["I1"]


def test_select_series_reads_loni_names_and_drops_transmission_scans():
    """LONI turns "6x5_PPMI_TAU (AC)" into "6x5_PPMI_TAU__AC_"; HR+ transmission scans are modality PT too."""
    idx = pd.DataFrame([_index_rows(image_id="I1", desc="6x5_PPMI_TAU__AC_", tracer="", n_frames=6),
                        _index_rows(image_id="I2", date="2021-01-01", desc="EM_80", tracer="F-18 Solution"),
                        _index_rows(image_id="I3", date="2021-01-01", desc="TX_Scan_4min_4i_16s", tracer="Ge-68 Solid Solution", n_frames=3),
                        _index_rows(image_id="I4", date="2022-01-01", desc="measured atten", tracer="", n_frames=3)])
    pick = lambda tr: sorted(pet.select_series(idx, tracer=tr).loc[lambda d: d["selected"], "image_id"])
    assert pick("tau") == ["I1"] and pick("av133") == ["I2"] and pick("sv2a") == []
    both = pd.DataFrame([_index_rows(image_id="I1", desc="[BR-DY_NAC]_VMAT_Brain_Dyn", n_frames=3),
                         _index_rows(image_id="I2", desc="[BR-DY_CTAC]_VMAT_Brain_Dyn")])
    assert pet.select_series(both).loc[lambda d: d["selected"], "image_id"].tolist() == ["I2"]   # AC wins despite fewer frames


def test_suvr_labels_pivot_regions_per_scan(tmp_path):
    from pie.imaging.labels import suvr_labels
    (tmp_path / "Imaging").mkdir()
    pd.DataFrame({"PROTOCOL": [14] * 3, "PATNO": [1, 1, 2], "EVENT_ID": ["BL"] * 3, "SV2A_SCAN_DATE": ["01/2023"] * 3,
                  "SV2A_REGION": ["PUTAMEN_L", "CS", "PUTAMEN_L"], "SV2A_SUVR_CS": [5.0, 1.0, 4.5], "SV2A_ANALYZED": ["Yes", "Yes", "No"]}
                 ).to_csv(tmp_path / "Imaging" / "SV2A_PET_SUVR_Analysis_08Sep2026.csv", index=False)
    w = suvr_labels(tmp_path, "sv2a")
    assert len(w) == 1 and w["PUTAMEN_L"].item() == 5.0 and w["PET_SCAN_DATE"].dt.year.item() == 2023
