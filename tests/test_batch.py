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
