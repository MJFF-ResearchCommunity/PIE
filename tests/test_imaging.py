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
        _series(2, "3D_T1-weighted", "2021-03-23_09_05_05.0", "I4", 1, 25e6),
        _series(2, "3D_T1-weighted", "2022-03-23_09_05_05.0", "I6", 1, 25e6),
        _series(2, "Transverse", "2022-03-23_09_05_05.0", "I7", 1, 60e6),   # 2D axial, bigger file: must lose
    ])
    idx["session_date"] = pd.to_datetime(idx["session"].str[:10])
    sel = select_t1_series(idx)
    assert sel["image_id"].tolist() == ["I1", "I4", "I6"]


def test_parse_stats_and_idp_table(tmp_path):
    sid = "I1"
    stats = tmp_path / sid / "stats"
    stats.mkdir(parents=True)
    (stats / "aseg+DKT.stats").write_text(
        "# Measure Mask, MaskVol, Mask Volume, 1500000.0, mm^3\n"
        "# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange\n"
        "  1  12  4000 4100.0  Left-Putamen  90 5 60 120 60\n"
        "  2  51  3900 3900.0  Right-Putamen 90 5 60 120 60\n"
        "  3   4  9000 9000.0  Left-Lateral-Ventricle 20 5 0 60 60\n"
    )
    d = parse_stats(stats / "aseg+DKT.stats")
    assert d["MaskVol"] == 1500000.0 and d["Left-Putamen"] == 4100.0
    sessions = pd.DataFrame([dict(patno=7, image_id=sid, session_date="2020-01-01", EVENT_ID="BL", protocol_phase=1,
                                  Manufacturer="Siemens", MagneticFieldStrength=3.0)])
    idp = build_idp_table(sessions, tmp_path)
    assert len(idp) == 1
    row = idp.iloc[0]
    assert row["vol_Left_Putamen"] == 4100.0 and row["sum_Putamen"] == 8000.0
    assert abs(row["asym_Putamen"] - (4100 - 3900) / 8000) < 1e-9
    assert row["sum_Ventricles"] == 9000.0 and row["Manufacturer"] == "Siemens"


def test_read_ida_metadata_parses_full_records_and_skips_stubs(tmp_path):
    from pie.imaging.index import read_ida_metadata
    (tmp_path / "stub.xml").write_text('<?xml version="1.0"?><metadata version="1.0"><subject id="1"/><image uid="I1"/></metadata>')
    (tmp_path / "full.xml").write_text(
        '<?xml version="1.0"?><idaxs><project><subject><subjectIdentifier>3051</subjectIdentifier>'
        '<researchGroup>PD</researchGroup><visit><visitIdentifier>Baseline</visitIdentifier></visit>'
        '<study><subjectAge>71.2</subjectAge><series><dateAcquired>2010-10-26</dateAcquired></series>'
        '<imagingProtocol><imageUID>223766</imageUID><description>SAG 3D T1</description><protocolTerm>'
        '<protocol term="Manufacturer">Philips Medical Systems</protocol><protocol term="Field Strength">1.5</protocol>'
        '</protocolTerm></imagingProtocol></study></subject></project></idaxs>')
    df = read_ida_metadata(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["image_id"] == "I223766" and row["patno"] == 3051 and row["ida_visit"] == "Baseline"
    assert row["ida_manufacturer"] == "Philips Medical Systems" and row["ida_field"] == "1.5"


def test_read_loni_collection_csv(tmp_path):
    from pie.imaging.index import read_loni_collection_csv
    (tmp_path / "c.csv").write_text('"Image Data ID","Subject","Group","Sex","Age","Visit","Modality","Description","Type","Acq Date","Format","Downloaded"\n'
                                    '"I495208","92834","Prodromal","M","66","BL","MRI","MPRAGE GRAPPA2","Original","3/20/2015","DCM","Yes"\n')
    df = read_loni_collection_csv(tmp_path / "c.csv")
    r = df.iloc[0]
    assert r["image_id"] == "I495208" and r["loni_visit"] == "BL" and r["loni_age"] == 66 and r["loni_group"] == "Prodromal"
    assert str(r["loni_acq_date"])[:10] == "2015-03-20"
