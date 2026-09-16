# DataPreprocessor (`pie_clean.DataPreprocessor`)

`DataPreprocessor` lives in the companion package **PIE-clean** (`pie_clean/data_preprocessor.py`),
not in this repository. It is a set of static methods that rewrite values in individual PPMI
medical-history tables using knowledge of the forms: "2 = uncertain" codes, free-text medication
indications, dose strings. `DataReducer` drops columns on statistics; `DataPreprocessor` fixes what a
column *means*.

`DataLoader.load(clean_data=True)` (the default, and what the pipeline uses) already runs
`clean_medical_history`, so PIE code rarely calls these directly.

| Method | Input | What it does |
|---|---|---|
| `clean(data_dict)` | loader dict | Replaces `data_dict["medical_history"]` with `clean_medical_history(...)`. Raises `KeyError` if that key is absent. No other modality is touched. |
| `clean_medical_history(med_hist_dict)` | `data_dict["medical_history"]` | Runs the five table cleaners below on whichever tables are present; returns the same dict. |
| `clean_ledd_meds(df)` | `LEDD_Concomitant_Medication` | Parses dates, drops non-LEDD drugs, fills missing `LEDD`. |
| `clean_concomitant_meds(df)` | `Concomitant_Medication` | Parses dates, gives every row an indication code. |
| `clean_vital_signs(df)` | `Vital_Signs` | Adds blood-pressure bands. |
| `clean_features_of_parkinsonism(df, uncertain=0.5)` | `Features_of_Parkinsonism` | `FEATBRADY`, `FEATPOSINS`, `FEATRIGID`, `FEATTREMOR`: 2 (uncertain) → `uncertain`. All four columns must exist. |
| `clean_gen_physical_exam(df, uncertain=0.5)` | `General_Physical_Exam` | `ABNORM`: 2 (cannot assess) → `uncertain`. |
| `event_id_to_months(eid)` | visit code | Months from baseline via `EVENT_TIMES`; `NaN` for unscheduled codes. |
| `dt_to_datetime(ser)` | Series of `"MM/YYYY"` | `pd.to_datetime(ser, format="%m/%Y")`; other formats raise, missing → `NaT`. |
| `create_concomitant_meds(df, output_path=None)` | raw `Concomitant_Medication` | Maintenance only: rebuilds the indication-mapping JSON for a new data cut. Writes `output_path`, by default the package's `concomitant_meds_indications.json` (the file `clean_concomitant_meds` reads). |

Every table cleaner works on a copy and returns it.

## Why each cleaner exists

**Uncertain codes.** PPMI codes several yes/no findings as 0 = no, 1 = yes, 2 = uncertain / cannot
assess. Left alone, a model reads 2 as "more than yes". `uncertain=0.5` places it between the two;
pass `uncertain=np.nan` to treat it as missing instead.

**Vital signs.** Adds `Sup BP code`/`Sup BP label` from `SYSSUP`/`DIASUP` and `Stnd BP code`/`Stnd BP label`
from `SYSSTND`/`DIASTND`, using American Heart Association bands. The most severe band is tested
first and either reading alone is enough to reach a band:

| Test (first match wins) | Code | Label |
|---|---|---|
| systolic or diastolic missing | `NaN` | `NaN` |
| systolic ≥ 180 or diastolic ≥ 120 | 4 | Hypertensive crisis |
| systolic ≥ 140 or diastolic ≥ 90 | 3 | Stage 2 HTN |
| systolic ≥ 130 or diastolic ≥ 80 | 2 | Stage 1 HTN |
| systolic ≥ 120 | 1 | Elevated |
| otherwise | 0 | Normal |

**Concomitant medications.** Most rows carry a numeric indication code in `CMINDC`; the rest carry
only free text in `CMINDC_TEXT`, full of typos and synonyms. Resolution per row:

1. `CMINDC` present → keep it.
2. Neither code nor text → map by drug name (`ASPIRIN` → 17 Pain, `GINKOBIL` → 22 Supplements,
   `HUMULIN NPH` → 11 Diabetes), otherwise 25 Other.
3. Text present → look it up (lower-cased, stripped) in `pie_clean/concomitant_meds_indications.json`;
   unmatched text stays `NaN` and is logged.

`CMINDC_TEXT` is then overwritten with the standard label for the code (`"UNKNOWN"` for `NaN`), and
`CMINDC` becomes `int` when nothing is left unmapped. `STARTDT`/`STOPDT` become datetimes; missing
dates are kept as `NaT` (no start date: assume before enrolment; no stop date: assume ongoing).

**LEDD medications.** Anticholinergics (benztropine, biperden, budipin and brand names) are sometimes
entered on the LEDD form; those rows are removed. Where `LEDD` is missing it is computed from the
drug name and `LEDDSTRMG × LEDDOSE × LEDDOSFRQ` with standard conversion factors (e.g. ×1 for
carbidopa/levodopa, ×20 ropinirole, ×100 pramipexole/rasagiline). COMT inhibitors and istradefylline
depend on the levodopa dose, so they get a string such as `"LD x 0.33"`; the `LEDD` column is
therefore mixed-type, and unrecognised drugs stay `NaN`.

**Visit months.** `EVENT_TIMES` maps scheduled visits to months: `SC` −3 (screening can fall up to
3 months before baseline), `BL` 0, `V01`–`V12` at 3, 6, 9, 12, 18, 24 … 60, then `V13`–`V21` yearly
to 168; phone visits `R01`–`R20` fall between. Unscheduled codes return `NaN` on purpose: they have
no fixed time.

## Example

Runnable with synthetic tables:

```python
import numpy as np
import pandas as pd
from pie_clean import DataPreprocessor as DP

vs = pd.DataFrame({"SYSSUP": [118, 150], "DIASUP": [76, 95], "SYSSTND": [125, 185], "DIASTND": [79, 100]})
DP.clean_vital_signs(vs)[["Sup BP label", "Stnd BP label"]]
#   Sup BP label        Stnd BP label
# 0       Normal             Elevated
# 1  Stage 2 HTN  Hypertensive crisis

fop = pd.DataFrame({"FEATBRADY": [0, 1, 2], "FEATPOSINS": [2, 0, 0], "FEATRIGID": [1, 1, 1], "FEATTREMOR": [0, 2, 1]})
DP.clean_features_of_parkinsonism(fop)["FEATBRADY"].tolist()          # [0.0, 1.0, 0.5]
DP.clean_gen_physical_exam(pd.DataFrame({"ABNORM": [0, 1, 2]}), uncertain=np.nan)["ABNORM"].tolist()
                                                                      # [0.0, 1.0, nan]
[DP.event_id_to_months(e) for e in ["SC", "BL", "V04", "U01"]]        # [-3, 0, 12, nan]

cm = pd.DataFrame({"CMTRT": ["DRUG A", "ASPIRIN", "DRUG B"], "CMINDC": [14, np.nan, np.nan],
                   "CMINDC_TEXT": [np.nan, np.nan, "high blood pressure"],
                   "STARTDT": ["01/2000", "02/2000", np.nan], "STOPDT": [np.nan] * 3})
DP.clean_concomitant_meds(cm)[["CMINDC", "CMINDC_TEXT"]]
#    CMINDC   CMINDC_TEXT
# 0      14  Hypertension
# 1      17          Pain    <- no code, no text: mapped from the drug name
# 2      14  Hypertension    <- free text mapped through the JSON table

ledd = pd.DataFrame({"LEDTRT": ["CARBIDOPA/LEVODOPA", "ROPINIROLE", "BENZTROPINE"], "LEDD": [np.nan] * 3,
                     "LEDDSTRMG": [100, 2, 1], "LEDDOSE": [1, 1, 1], "LEDDOSFRQ": [3, 3, 1],
                     "LEDDOSSTR": ["", "", ""], "STARTDT": ["01/2000"] * 3, "STOPDT": [np.nan] * 3})
DP.clean_ledd_meds(ledd)[["LEDTRT", "LEDD"]]
#                LEDTRT  LEDD
# 0  CARBIDOPA/LEVODOPA   300    <- 100 mg x 1 x 3/day
# 1          ROPINIROLE   120    <- 2 mg x 1 x 3/day x 20; the benztropine row is dropped
```

On a real download, load without cleaning and apply the cleaners yourself:

```python
from pie_clean import DataLoader, DataPreprocessor, MEDICAL_HISTORY

raw = DataLoader.load("./PPMI", modalities=[MEDICAL_HISTORY], clean_data=False)
vitals = DataPreprocessor.clean_vital_signs(raw[MEDICAL_HISTORY]["Vital_Signs"])
cleaned = DataPreprocessor.clean(raw)          # all medical-history cleaners at once
```

## Tests

`tests/test_pie_clean.py::test_data_preprocessor` loads everything from `./PPMI` with
`clean_data=False`, runs `DataPreprocessor.clean` and asserts the result is a dict containing every
modality in `ALL_MODALITIES`. It is marked `ppmi` and skipped without the download.

The cleaners' unit tests live in PIE-clean (`tests/test_data_preprocessor.py`, on committed synthetic
fixtures), including `test_clean_vital_signs_band_order` and `test_create_concomitant_meds`.

```bash
pytest tests/test_pie_clean.py -m ppmi -q          # in PIE
pytest tests/test_data_preprocessor.py -q          # in PIE-clean
```
