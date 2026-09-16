"""pie_clean.DataLoader as PIE uses it.

Nothing is loaded at import or collection time. The synthetic test builds a fake
two-folder PPMI tree; the real-data test is marked ``ppmi`` and skipped without ./PPMI.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie_clean import DataLoader, SUBJECT_CHARACTERISTICS, MEDICAL_HISTORY, MOTOR_ASSESSMENTS

PPMI_DATA_PATH = Path(__file__).resolve().parent.parent / "PPMI"


def test_data_loader_synthetic(tmp_path):
    (tmp_path / "_Subject_Characteristics").mkdir()
    (tmp_path / "Motor___MDS-UPDRS").mkdir()
    pd.DataFrame({"PATNO": [1, 2, 3], "EVENT_ID": ["BL"] * 3, "SEX": [1, 0, 1]}) \
        .to_csv(tmp_path / "_Subject_Characteristics/Demographics_2000-01-01.csv", index=False)
    pd.DataFrame({"PATNO": [1, 1, 2], "EVENT_ID": ["BL", "V04", "BL"], "NP3TOT": [10, 14, 3]}) \
        .to_csv(tmp_path / "Motor___MDS-UPDRS/MDS-UPDRS_Part_III_2000-01-01.csv", index=False)

    mods = [SUBJECT_CHARACTERISTICS, MOTOR_ASSESSMENTS]
    d = DataLoader.load(str(tmp_path), modalities=mods)
    assert d[MOTOR_ASSESSMENTS].columns.tolist() == ["PATNO", "EVENT_ID", "NP3TOT"]
    wide = DataLoader.load(str(tmp_path), modalities=mods, merge_output=True)
    assert wide.shape == (4, 4)  # union of the four visits


@pytest.mark.ppmi
@pytest.mark.skipif(not PPMI_DATA_PATH.exists(), reason=f"PPMI data not found at {PPMI_DATA_PATH}")
def test_data_loader_real_data():
    data = DataLoader.load(data_path=str(PPMI_DATA_PATH), modalities=[SUBJECT_CHARACTERISTICS, MEDICAL_HISTORY])
    assert not data[SUBJECT_CHARACTERISTICS].empty
    assert data[MEDICAL_HISTORY]
    assert all(isinstance(df, pd.DataFrame) for df in data[MEDICAL_HISTORY].values())
