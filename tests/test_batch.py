"""run_batch keeps every column when later rows carry keys the first rows lacked (optional stages, per-side features)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging.batch import run_batch


def _job(j):
    patno, extra = j
    row = {"patno": patno, "fa": 0.5, "error": ""}
    if extra:
        row.update({"nst_afd_l": 1.0, "nst_afd_r": 2.0})
    return row


def test_run_batch_widens_header(tmp_path):
    out = tmp_path / "f.csv"
    run_batch([(1, False), (2, True), (3, False)], _job, out, workers=1)
    d = pd.read_csv(out).set_index("patno").sort_index()
    assert set(d.columns) >= {"fa", "error", "nst_afd_l", "nst_afd_r"} and len(d) == 3
    assert d.loc[2, "nst_afd_r"] == 2.0 and pd.isna(d.loc[1, "nst_afd_l"])
    run_batch([(4, True)], _job, out, workers=1)                      # resume appends with the widened header
    d = pd.read_csv(out)
    assert len(d) == 4 and d.loc[d.patno == 4, "nst_afd_l"].item() == 1.0


def test_modality_session_is_the_one_nearest_the_t1_session():
    # a later, larger 3D FLAIR must not be measured on the baseline T1's anatomy
    import pandas as pd

    from pie.imaging.batch import session_rows

    g = pd.DataFrame({"date": ["2010-01-05", "2012-03-01", "9999-01-01"], "n_files": [40, 176, 500], "image_id": ["a", "b", "c"]})
    assert [r["image_id"] for r in session_rows(g)] == ["c"]                       # no T1 date: most files, as before
    assert [r["image_id"] for r in session_rows(g, "2010-01-01")] == ["a"]
    assert [r["image_id"] for r in session_rows(g, "2012-02-20")] == ["b"]


def test_convert_series_returns_only_this_conversion_and_keeps_a_failure_log(tmp_path, monkeypatch):
    import zipfile

    from pie.imaging import batch

    zp = tmp_path / "loni.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("PPMI/1/DTI/2000-01-01_09_00_00.0/I000001/a.dcm", b"x")
    fake = tmp_path / "dcm2niix"
    fake.write_text('#!/bin/sh\nwhile [ "$1" != "-o" ]; do shift; done\n'
                    '[ -n "$FAIL" ] && { echo "no gradient tags" >&2; exit 2; }\ntouch "$2/I000001_5.nii.gz" "$2/I000001_5.bval"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(batch, "DCM2NIIX", str(fake))
    out = tmp_path / "nii"
    out.mkdir()
    (out / "I000001_5a.nii.gz").touch()                       # an earlier attempt's dcm2niix-suffixed duplicate
    got = batch.convert_series(zp, "PPMI/1/DTI/2000-01-01_09_00_00.0/I000001/", out)
    assert [p.rsplit("/", 1)[1] for p in got] == ["I000001_5.nii.gz"] and (out / "I000001_5.bval").exists()
    monkeypatch.setenv("FAIL", "1")
    assert batch.convert_series(zp, "PPMI/1/DTI/2000-01-01_09_00_00.0/I000001/", tmp_path / "nii2") == []
    assert "no gradient tags" in (tmp_path / "nii2" / "I000001_dcm2niix.log").read_text()
