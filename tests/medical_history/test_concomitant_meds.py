import pytest
from pie.clinical_loader import load_clinical_data
from pie.data_loader import DataLoader
from pie.med_hist_loader import load_ppmi_medical_history

@pytest.fixture(scope="function")
def meds():
    data = load_ppmi_medical_history("./PPMI/Medical History/Medical")
    print(data.shape)
    return data

def test_load_concom_meds(meds):
    # Load directly as medical history
    assert "CMTRT" in meds.columns
    assert "CMINDC" in meds.columns
    assert "CMDOSE" in meds.columns

    # Load as part of med hist in clinical data
    data = load_clinical_data("./PPMI", "PPMI")
    assert isinstance(data, dict)
    assert "med_hist" in data
    assert "CMTRT" in data["med_hist"].columns

    # Load as med hist, in clin data, in DataLoader
    data = DataLoader.load("./PPMI", "PPMI")
    assert isinstance(data, dict)
    assert "clinical" in data
    assert "med_hist" in data["clinical"]
    assert "CMTRT" in data["clinical"]["med_hist"].columns
