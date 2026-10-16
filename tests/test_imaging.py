"""Unit tests for the imaging layer that need no data, GPU or FastSurfer."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging.index import select_t1_series
from pie.imaging.fastsurfer import parse_stats
from pie.imaging.features import build_idp_table


def _series(patno, desc, session, image_id, n_files, nbytes):
    return dict(zip="z", patno=patno, series_desc=desc, session=session, image_id=image_id,
                n_files=n_files, bytes=nbytes, member_prefix="p/")


def test_select_t1_series_prefers_non_repeat_largest_t1():
    idx = pd.DataFrame([
        _series(1, "MPRAGE", "2011-01-01_10_00_00.0", "I1", 170, 100),
        _series(1, "MPRAGE_Repeat", "2011-01-01_10_00_00.0", "I2", 176, 120),
        _series(1, "Coronal", "2011-01-01_10_00_00.0", "I3", 1, 1),
        _series(1, "AX_T2_FLAIR", "2011-01-01_10_00_00.0", "I5", 200, 500),
        _series(2, "3D_T1-weighted", "2000-01-01_09_00_00.0", "I4", 1, 25e6),
        _series(2, "3D_T1-weighted", "2001-01-01_09_00_00.0", "I6", 1, 25e6),
        _series(2, "Transverse", "2001-01-01_09_00_00.0", "I7", 1, 60e6),   # 2D axial, bigger file: must lose
    ])
    idx["session_date"] = pd.to_datetime(idx["session"].str[:10])
    sel = select_t1_series(idx)
    assert sel["image_id"].tolist() == ["I1", "I4", "I6"]


def test_select_t1_series_keeps_one_t1_per_visit_day():
    # LONI timestamps each series separately: an MPRAGE and its repeat 7 minutes later are one visit, not two
    idx = pd.DataFrame([
        _series(1, "MPRAGE", "2011-01-01_10_00_00.0", "I1", 170, 100),
        _series(1, "MPRAGE_RPT", "2011-01-01_10_07_30.0", "I2", 176, 120),
        _series(1, "MPRAGE", "2012-01-01_10_00_00.0", "I3", 170, 100),
        _series(2, "MPRAGE", "9999-01-01_10_00_00.0", "I4", 170, 100),   # LONI-masked dates are not one day
        _series(2, "MPRAGE", "9999-01-01_11_00_00.0", "I5", 170, 100),
    ])
    idx["session_date"] = pd.to_datetime(idx["session"].str[:10], errors="coerce")
    assert select_t1_series(idx)["image_id"].tolist() == ["I1", "I3", "I4", "I5"]


def test_parse_stats_and_idp_table(tmp_path):
    sid = "I1"
    stats = tmp_path / sid / "stats"
    stats.mkdir(parents=True)
    (stats / "aseg+DKT.stats").write_text(
        "# FastSurfer_version 2.6.0\n"
        "# Measure Mask, MaskVol, Mask Volume, 1500000.0, mm^3\n"
        "# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange\n"
        "  1  12  4000 4100.0  Left-Putamen  90 5 60 120 60\n"
        "  2  51  3900 3900.0  Right-Putamen 90 5 60 120 60\n"
        "  3   4  9000 9000.0  Left-Lateral-Ventricle 20 5 0 60 60\n"
    )
    d = parse_stats(stats / "aseg+DKT.stats")
    assert d["MaskVol"] == 1500000.0 and d["Left-Putamen"] == 4100.0
    sessions = pd.DataFrame([dict(patno=7, image_id=sid, session_date="2020-01-01", EVENT_ID="BL", protocol_phase=1,
                                  Manufacturer="Siemens", MagneticFieldStrength=15000)])
    idp = build_idp_table(sessions, tmp_path)
    assert len(idp) == 1
    row = idp.iloc[0]
    assert row["vol_Left_Putamen"] == 4100.0 and row["sum_Putamen"] == 8000.0
    assert abs(row["asym_Putamen"] - (4100 - 3900) / 8000) < 1e-9 and abs(row["ai_Putamen"] - (4100 - 3900) / 4000) < 1e-9
    assert row["sum_Ventricles"] == 9000.0 and row["Manufacturer"] == "Siemens"
    assert row["fastsurfer_version"] == "2.6.0" and row["MagneticFieldStrength"] == 1.5     # gauss in old GE headers
    assert "vol_FastSurfer_version" not in idp.columns


def test_read_ida_metadata_parses_full_records_and_skips_stubs(tmp_path):
    from pie.imaging.index import read_ida_metadata
    (tmp_path / "stub.xml").write_text('<?xml version="1.0"?><metadata version="1.0"><subject id="1"/><image uid="I1"/></metadata>')
    (tmp_path / "full.xml").write_text(
        '<?xml version="1.0"?><idaxs><project><subject><subjectIdentifier>1</subjectIdentifier>'      # synthetic record
        '<researchGroup>PD</researchGroup><visit><visitIdentifier>Baseline</visitIdentifier></visit>'
        '<study><subjectAge>70.0</subjectAge><series><dateAcquired>2000-01-01</dateAcquired></series>'
        '<imagingProtocol><imageUID>000001</imageUID><description>SAG 3D T1</description><protocolTerm>'
        '<protocol term="Manufacturer">Philips Medical Systems</protocol><protocol term="Field Strength">1.5</protocol>'
        '</protocolTerm></imagingProtocol></study></subject></project></idaxs>')
    df = read_ida_metadata(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["image_id"] == "I000001" and row["patno"] == 1 and row["ida_visit"] == "Baseline"
    assert row["ida_manufacturer"] == "Philips Medical Systems" and row["ida_field"] == "1.5"


def test_read_loni_collection_csv(tmp_path):
    from pie.imaging.index import read_loni_collection_csv
    (tmp_path / "c.csv").write_text('"Image Data ID","Subject","Group","Sex","Age","Visit","Modality","Description","Type","Acq Date","Format","Downloaded"\n'
                                    '"I000002","2","PD","F","70","BL","MRI","MPRAGE","Original","1/1/2000","DCM","Yes"\n')   # synthetic row
    df = read_loni_collection_csv(tmp_path / "c.csv")
    r = df.iloc[0]
    assert r["image_id"] == "I000002" and r["loni_visit"] == "BL" and r["loni_age"] == 70 and r["loni_group"] == "PD"
    assert str(r["loni_acq_date"])[:10] == "2000-01-01"


def test_t1_choice_ranks_original_3d_over_a_bigger_interpolated_reformat_and_phase_follows_the_date():
    import pandas as pd

    from pie.imaging.index import protocol_phase, select_t1_series, t1_candidates

    idx = pd.DataFrame([dict(zip="z", patno=1, series_desc="MPRAGE", session="2021-03-01_10_00_00.0", image_id="I1", n_files=176, bytes=40e6, member_prefix="a/"),
                        dict(zip="z", patno=1, series_desc="MPRAGE_MPR", session="2021-03-01_10_20_00.0", image_id="I2", n_files=176, bytes=90e6, member_prefix="b/"),
                        dict(zip="z", patno=2, series_desc="MPRAGE", session="2012-03-01_10_00_00.0", image_id="I3", n_files=176, bytes=40e6, member_prefix="c/")])
    assert set(t1_candidates(idx)["image_id"]) == {"I1", "I2"}
    assert select_t1_series(idx)["image_id"].tolist() == ["I2", "I3"]              # bytes alone favour the reformat
    hdr = pd.DataFrame({"image_id": ["I1", "I2"], "ImageType": ["ORIGINAL;PRIMARY;M;NORM", "DERIVED;SECONDARY;MPR"],
                        "MRAcquisitionType": ["3D", "3D"], "InversionTime": [900, 900], "PixelSpacing": ["1.0;1.0", "0.5;0.5"]})
    assert select_t1_series(idx, hdr)["image_id"].tolist() == ["I1", "I3"]
    assert protocol_phase(pd.Series(["2012-03-01", "2021-03-01", "9999-01-01"])).tolist()[:2] == [1.0, 2.0]
    assert protocol_phase(pd.Series(["9999-01-01"])).isna().all()


def test_thickness_long_stream_table_and_visit_codes(tmp_path):
    from pie.imaging.index import visit_to_event
    from pie.imaging.run import write_long_idps

    for root, sid in ((tmp_path / "fastsurfer_long", "I9"),):
        stats = root / sid / "stats"
        stats.mkdir(parents=True)
        (stats / "aseg+DKT.stats").write_text("  1  12  4000 4100.0  Left-Putamen  90 5 60 120 60\n")
        (stats / "lh.aparc.stats").write_text(          # synthetic FreeSurfer aparc rows
            "# Measure Cortex, MeanThickness, Mean Thickness, 2.4500, mm\n"
            "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd\n"
            "precuneus 9000 6100 15000 2.310 0.5 0.1 0.02 100 9\n")
    sessions = pd.DataFrame([dict(patno=7, image_id="I9", session_date="2022-01-01", EVENT_ID="V04", protocol_phase=2),
                             dict(patno=7, image_id="I8", session_date="2021-01-01", EVENT_ID="BL", protocol_phase=2)])
    idp = write_long_idps(sessions, tmp_path, {7: ["I8", "I9"]})       # I8's long time point is unfinished: no row
    row = idp.iloc[0]
    assert len(idp) == 1 and row["stream"] == "long" and row["long_template"] == "7" and row["vol_Left_Putamen"] == 4100.0
    assert row["cth_lh_precuneus"] == 2.31 and row["area_lh_precuneus"] == 6100 and row["cth_lh_mean"] == 2.45
    assert (tmp_path / "fastsurfer_long_idps.csv").exists()
    assert [visit_to_event(x) for x in ("V06", "Month 48", "Unscheduled Visit 02", "Screening", "Month 7")] == ["V06", "V10", "U02", "SC", None]


def test_optional_freesurfer_stages_are_parsed_into_the_idp_row(tmp_path):
    from pie.imaging.features import extra_t1_columns

    subj = tmp_path / "I1"
    (subj / "stats").mkdir(parents=True)
    (subj / "mri").mkdir()
    (subj / "synthseg").mkdir()
    (subj / "stats" / "sclimbic.stats").write_text("# ColHeaders Index SegId NVoxels Volume_mm3 StructName\n"
                                                   "9  865  343  359.6  Left-Basal-Forebrain\n10 866 364 369.8 Right-Basal-Forebrain\n")
    pd.DataFrame({"case": ["x"], "Left-Basal-Forebrain": [0.8], "Right-Basal-Forebrain": [-2.5]}).to_csv(subj / "stats" / "sclimbic.zqa_scores_all.csv", index=False)
    (subj / "mri" / "lh.hippoSfVolumes.txt").write_text("CA1 600.5\nWhole_hippocampus 3400.0\n")
    pd.DataFrame({"subject": ["x"], "left hippocampus": [3500.0], "total intracranial": [1.5e6]}).to_csv(subj / "synthseg" / "volumes.csv", index=False)
    pd.DataFrame({"subject": ["x"], "general white matter": [0.9], "hippocampus": [0.6]}).to_csv(subj / "synthseg" / "qc.csv", index=False)
    from pie.imaging.cat12 import DEFAULT_CONFIG, result_name
    (subj / "stats" / result_name(DEFAULT_CONFIG)).write_text('{"ch4_lh_gm_mm3": 140.0, "ch4_lh_pgm_mm3": 170.0, "cat12_tiv_mm3": 1.5e6, "nmm_lh_basal_forebrain_mm3": 400.0}')
    row = extra_t1_columns(subj)
    assert row["vol_ch4_lh_gm"] == 140.0 and row["vol_ch4_lh_pgm"] == 170.0 and row["cat12_tiv_mm3"] == 1.5e6
    assert row["vol_nmm_lh_basal_forebrain"] == 400.0
    assert row["vol_sclimbic_Left_Basal_Forebrain"] == 359.6 and row["sclimbic_qa_z_max"] == 2.5
    assert row["vol_hsf_lh_CA1"] == 600.5 and row["synthseg_left_hippocampus"] == 3500.0 and row["synthseg_qc_min"] == 0.6


def test_idp_row_takes_the_default_ch4_configuration_and_names_it(tmp_path, monkeypatch):
    from pie.imaging import cat12
    from pie.imaging.features import extra_t1_columns

    subj = tmp_path / "I1"
    (subj / "stats").mkdir(parents=True)
    (subj / "stats" / "cat12_ch4.json").write_text('{"ch4_lh_gm_mm3": 140.0}')          # 13 Oct configuration, no config key
    monkeypatch.setattr(cat12, "DEFAULT_CONFIG", "julich31-1.5mm")
    assert extra_t1_columns(subj)["cat12_ch4_config"] == "julich31-1.5mm"
    monkeypatch.setitem(cat12.CONFIGS, "x-1mm", {"map": "x", "vox": 1.0})
    monkeypatch.setattr(cat12, "DEFAULT_CONFIG", "x-1mm")
    assert "vol_ch4_lh_gm" not in extra_t1_columns(subj)                                  # never another configuration's value
    (subj / "stats" / "cat12_ch4_x-1mm.json").write_text('{"ch4_lh_gm_mm3": 200.0, "cat12_ch4_config": "x-1mm", "cat12_ch4_map": "x"}')
    row = extra_t1_columns(subj)
    assert row["vol_ch4_lh_gm"] == 200.0 and row["cat12_ch4_config"] == "x-1mm" and "vol_ch4_config" not in row
