"""Cohort-neutral input: ADNI and other LONI projects, BIDS datasets, ADNI visit codes (synthetic data only)."""
import json
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd

from pie.imaging.cohorts import OFFSET, cohort_of, subject_key


def test_keys_are_deterministic_and_never_overlap():
    assert subject_key("PPMI", "3001") == 3001 and subject_key("PPMI", "00000JAN00") is None
    assert subject_key("ADNI", "002_S_0295") == OFFSET["ADNI"] + 295 and cohort_of(OFFSET["ADNI"] + 295) == "ADNI"
    k = subject_key("BIDS:ppp", "POM1001")
    assert k == subject_key("BIDS:ppp", "POM1001") and k >= OFFSET["HASHED"] and cohort_of(k) == "OTHER"
    assert k != subject_key("BIDS:stanford", "POM1001")


def test_loni_zips_of_any_project_and_bids_datasets_are_indexed(tmp_path):
    from pie.imaging.batch import index_series
    from pie.imaging.convert import convert_series
    from pie.imaging.index import index_zips, select_t1_series

    zp = tmp_path / "adni.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for subj in ("002_S_0295", "011_S_0002"):
            for i in range(3):
                z.writestr(f"ADNI/{subj}/MPRAGE/2010-05-01_10_00_00.0/I{subj[-4:]}/{i}.dcm", b"x")
    idx = index_zips([zp])
    assert sorted(idx["patno"]) == [OFFSET["ADNI"] + 2, OFFSET["ADNI"] + 295] and idx["n_files"].tolist() == [3, 3]
    assert set(idx["cohort"]) == {"ADNI"} and set(idx["participant_id"]) == {"002_S_0295", "011_S_0002"}
    assert set(index_series([zp])["patno"]) == {OFFSET["ADNI"] + 295, OFFSET["ADNI"] + 2}
    bids = tmp_path / "ppp"
    for sub, ses, when in (("01", "1", "2015-03-02T09:30:00"), ("01", "2", None)):
        d = bids / f"sub-{sub}" / f"ses-{ses}" / "anat"
        d.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((50, 50, 50), np.float32), np.eye(4)), d / f"sub-{sub}_ses-{ses}_T1w.nii.gz")
        (d / f"sub-{sub}_ses-{ses}_T1w.json").write_text(json.dumps({"Manufacturer": "Siemens", **({"AcquisitionDateTime": when} if when else {})}))
    sel = select_t1_series(index_zips([bids], cache_csv=tmp_path / "index.csv"))
    assert len(sel) == 2 and sel["patno"].nunique() == 1 and set(sel["cohort"]) == {"BIDS:ppp"}
    assert sorted(sel["session"].str[:10]) == ["2015-03-02", "9999-01-01"]
    row = sel.iloc[0]
    meta = convert_series(row["zip"], row["member_prefix"], row["patno"], row["image_id"], tmp_path / "nifti")
    assert meta["Manufacturer"] == "Siemens" and nib.load(meta["nifti"]).shape == (50, 50, 50)


def test_adni_sessions_take_adnimerge_visit_codes(tmp_path):
    from pie.imaging.link import link_adni, link_sessions_to_events

    pd.DataFrame({"RID": [295, 295], "VISCODE": ["bl", "m12"], "EXAMDATE": ["2010-05-03", "2011-05-10"]}).to_csv(tmp_path / "ADNIMERGE.csv", index=False)
    s = pd.DataFrame({"patno": [OFFSET["ADNI"] + 295] * 3, "session_date": ["2010-05-01", "2011-04-20", "2013-01-01"]})
    s = link_adni(link_sessions_to_events(s, tmp_path / "no_ppmi"), tmp_path / "ADNIMERGE.csv")
    assert s["EVENT_ID"].tolist() == ["bl", "m12", "UNK"]


def test_bids_modality_images_are_indexed_for_the_modality_selectors_and_copied(tmp_path):
    from pie.imaging.batch import convert_series, index_series
    from pie.imaging.dwi import _flag_dwi, convert as dwi_convert
    from pie.imaging.flair import flag_flair, flag_flair_3d
    from pie.imaging.nm import flag_nm

    root = tmp_path / "stanford"
    when = json.dumps({"AcquisitionDateTime": "2019-04-01T10:00:00"})
    for folder, stem, zoom in (("dwi", "sub-7_dwi", 2.0), ("anat", "sub-7_FLAIR", 1.0), ("anat", "sub-7_acq-NM_T1w", 0.7), ("anat", "sub-7_T1w", 1.0)):
        d = root / "sub-7" / folder
        d.mkdir(parents=True, exist_ok=True)
        shape = (8, 8, 6, 4) if folder == "dwi" else (8, 8, 6)
        nib.save(nib.Nifti1Image(np.ones(shape, np.float32), np.diag([zoom, zoom, zoom, 1.0])), d / f"{stem}.nii.gz")
        (d / f"{stem}.json").write_text(when)
    np.savetxt(root / "sub-7" / "dwi" / "sub-7_dwi.bval", [[0, 1000, 1000, 1000]])
    np.savetxt(root / "sub-7" / "dwi" / "sub-7_dwi.bvec", np.c_[np.zeros(3), np.eye(3)])
    idx = index_series([root]).set_index("desc")
    assert set(idx.index) == {"BIDS_dwi", "BIDS_FLAIR_3D", "NM - BIDS sub-7_acq-NM_T1w", "BIDS_T1w"} and set(idx["date"]) == {"2019-04-01"}
    i = idx.reset_index()
    assert i.loc[flag_nm(i), "desc"].tolist() == ["NM - BIDS sub-7_acq-NM_T1w"] and i.loc[flag_flair(i) & flag_flair_3d(i), "desc"].tolist() == ["BIDS_FLAIR_3D"]
    assert _flag_dwi(i).all()
    row = idx.loc["BIDS_dwi"]
    runs = dwi_convert(row["zip"], row["prefix"], tmp_path / "nii")
    assert len(runs) == 1 and np.loadtxt(runs[0][1]).tolist() == [0, 1000, 1000, 1000]
    assert convert_series(row["zip"], idx.loc["BIDS_FLAIR_3D", "prefix"], tmp_path / "fl")[0].endswith("sub-7_FLAIR.nii.gz")
