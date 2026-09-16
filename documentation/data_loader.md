# DataLoader (`pie_clean.DataLoader`)

`DataLoader` lives in the companion package **PIE-clean** (`pie_clean`), not in this repository. PIE
installs it from `requirements.txt` (`pie-clean @ git+https://github.com/MJFF-ResearchCommunity/PIE-clean@main`)
and calls it in step 1 of the [pipeline](pipeline.md). This page covers what PIE relies on; the
per-modality loaders are documented in
[PIE-clean/documentation](https://github.com/MJFF-ResearchCommunity/PIE-clean/tree/main/documentation).

```
./PPMI/<study-data folders> ──DataLoader.load──► {modality: DataFrame | {table: DataFrame}} ──► DataReducer
                                               └─► one wide DataFrame        (merge_output=True)
```

## `DataLoader.load`

```python
DataLoader.load(
    data_path: str = "./PPMI",
    modalities: list[str] | None = None,
    source: str = "PPMI",
    merge_output: bool = False,
    output_file: str | None = None,
    clean_data: bool = True,
    biospec_exclude: list[str] | None = None,
) -> dict | pd.DataFrame
```

| Argument | Meaning |
|---|---|
| `data_path` | Root of the PPMI study-data download. Each modality reads one sub-folder (`FOLDER_PATHS`). |
| `modalities` | Names to load. `None` loads the five core modalities (`ALL_MODALITIES`). Names outside `KNOWN_MODALITIES` are logged and skipped. |
| `source` | Passed to the biospecimen loader. Only `"PPMI"` is supported. |
| `merge_output` | `False`: dict keyed by modality. `True`: one DataFrame merged on `PATNO`/`EVENT_ID`. |
| `output_file` | Also write the result to disk (see [Saving](#saving)). |
| `clean_data` | Run `DataPreprocessor.clean_medical_history` on the medical-history tables ([data_preprocessor.md](data_preprocessor.md)). No other modality is cleaned. |
| `biospec_exclude` | Biospecimen source keys to skip, e.g. `project_9000`, `project_222`, `project_196`, `standard_files`. The first three are the large proteomics sets; excluding them is what lets a full load fit in desktop RAM. |

### Return value (`merge_output=False`)

| Key (constant) | Folder | Value |
|---|---|---|
| `subject_characteristics` (`SUBJECT_CHARACTERISTICS`) | `_Subject_Characteristics` | DataFrame, one row per `PATNO`/`EVENT_ID` |
| `medical_history` (`MEDICAL_HISTORY`) | `Medical_History` | dict of tables |
| `motor_assessments` (`MOTOR_ASSESSMENTS`) | `Motor___MDS-UPDRS` | DataFrame |
| `non_motor_assessments` (`NON_MOTOR_ASSESSMENTS`) | `Non-motor_Assessments` | DataFrame |
| `biospecimen` (`BIOSPECIMEN`) | `Biospecimen` | DataFrame, all non-excluded sources merged |

- Medical history stays a dict because many of its tables are logs (adverse events, medications)
  with several rows per visit, or run on their own timeline; forcing them to one row per visit loses
  information.
- `PATNO` is read as a string and a leading `PPMI-` is stripped, so IDs from different files join.
- Table keys inside a dict are the CSV file name minus its trailing `_<date>.csv`.
- For the core modalities only files matching the loader's prefix list are read (`FILE_PREFIXES` in
  `sub_char_loader.py`, `motor_loader.py`, `non_motor_loader.py`; `MEDICAL_HISTORY_PREFIXES` in
  `med_hist_loader.py`). A new PPMI form is ignored until its prefix is added there.
- `load_biospecimen_data` skips `standard_files` by default when called directly, but `DataLoader.load`
  passes an empty exclusion list, so `standard_files` *is* loaded unless you exclude it.

### `merge_output=True`

Rows are the union of `PATNO`/`EVENT_ID` pairs across every loaded modality, biospecimen included,
and each table is left-joined onto them. Only colliding column names get a prefix (`<table>_` or `<modality>_`). `DataReducer.merge_reduced_data`
prefixes *every* column instead, which is why the pipeline loads the dict form and merges there.

### Saving

| `merge_output` | `output_file="out/x.csv"` writes |
|---|---|
| `True` | `out/x.csv` |
| `False` | `out/<modality>.csv` for each DataFrame modality and `out/<modality>/<table>.csv` for each table-dict modality. Only the directory part of `output_file` is used. |

## Extended modalities

The five core modalities are what `modalities=None` loads. The other folders of a full PPMI study-data
download are loaded generically when named explicitly: every CSV in the folder, `patno`/`event_id`
upper-cased (the FOUND tables are lower-case), files without a `PATNO` column (codebooks) skipped.

| Constant | Folder | Returns |
|---|---|---|
| `STUDY_ENROLLMENT` | `Study_Enrollment` | dict of tables (consent, eligibility incl. `INSAA`, screen fail, visit type, ...) |
| `IMAGING` | `Imaging` | dict of tables (Xing core-lab DaTscan SBR / visual reads, FreeSurfer-7 IDPs, MRIQC, DTI ROIs, PET) |
| `PPMI_ONLINE` | `PPMI_Online` | dict of tables; `EVENT_ID`s are online visits, not clinic visits |
| `REMOTE_SCREENING` | `PPMI_Remote_Screening` | one table merged on `PATNO`/`EVENT_ID` |
| `FOUND` | `Follow_Up_persons_w_Neurologic_Disease` | one table merged on `PATNO` (risk-factor questionnaires) |
| `ROCHE_APP` | `Roche_Smartphone_App` | dict with the long-format app table (one row per test result) |

`EXTENDED_MODALITIES` maps each constant to `True` (merge into one table on `PATNO`, plus `EVENT_ID`
where present) or `False` (dict keyed by file stem, because the tables are long-format, per-substudy
or on their own visit scheme). `KNOWN_MODALITIES = ALL_MODALITIES + list(EXTENDED_MODALITIES)`. With
`merge_output=True` the table-dict modalities are merged table by table like medical history; with
`output_file` each gets its own subdirectory.

The pipeline's `--modalities` flag accepts every name in `KNOWN_MODALITIES` ([pipeline.md](pipeline.md#cli)).

Helpers exported from `pie_clean`:

| Function | Returns |
|---|---|
| `load_ppmi_folder(folder_path, modality, merge=True)` | Every CSV of one folder: a wide DataFrame (`merge=True`) or a dict keyed by file stem. The generic loader behind the extended modalities. |
| `load_data_dictionary(data_path)` | The PPMI *Data & Databases* tables found in `data_path` or `data_path/Data___Databases`, latest file each: keys `dictionary`, `code_list`, `deprecated`. Use it to decode coded columns. |

Core-folder tables added in the 2026 data cuts are in the prefix lists: polygenic risk scores,
race/ethnicity and ST-Direct demographics (subject characteristics); `Primary_Research_Diagnosis` and
the newer PET/tau/FD4 substudy forms (medical history); CANTAB cognitive activities,
`Smell_and_Genetic_Testing` and the renamed REM sleep behavior disorder screening questionnaire
(non-motor); the underscore-named MDS-UPDRS Part II file (motor); and in the biospecimen
`standard_files` group the CSF alpha-synuclein SAA results, SynOne and whole-blood substudy forms,
and neuropathology and pathology-core tables.

## Examples

Runnable without PPMI access: build a two-folder stand-in with fake participants.

```python
from pathlib import Path
import pandas as pd
from pie_clean import DataLoader, SUBJECT_CHARACTERISTICS, MOTOR_ASSESSMENTS

root = Path("demo_ppmi")
(root / "_Subject_Characteristics").mkdir(parents=True, exist_ok=True)
(root / "Motor___MDS-UPDRS").mkdir(parents=True, exist_ok=True)
pd.DataFrame({"PATNO": [1, 2, 3], "EVENT_ID": ["BL"] * 3, "SEX": [1, 0, 1]}) \
    .to_csv(root / "_Subject_Characteristics/Demographics_2000-01-01.csv", index=False)
pd.DataFrame({"PATNO": [1, 1, 2], "EVENT_ID": ["BL", "V04", "BL"], "NP3TOT": [10, 14, 3]}) \
    .to_csv(root / "Motor___MDS-UPDRS/MDS-UPDRS_Part_III_2000-01-01.csv", index=False)

mods = [SUBJECT_CHARACTERISTICS, MOTOR_ASSESSMENTS]
d = DataLoader.load(str(root), modalities=mods)
d[MOTOR_ASSESSMENTS].columns.tolist()        # ['PATNO', 'EVENT_ID', 'NP3TOT']

wide = DataLoader.load(str(root), modalities=mods, merge_output=True)
wide.shape                                   # (4, 4): union of the four visits; missing cells are NaN
```

Against a real download:

```python
from pie_clean import DataLoader, load_data_dictionary
from pie_clean.constants import STUDY_ENROLLMENT, IMAGING

data = DataLoader.load("./PPMI", biospec_exclude=["project_9000", "project_222", "project_196"])

DataLoader.load("./PPMI", modalities=["subject_characteristics", "motor_assessments"],
                merge_output=True, output_file="./output/merged.csv")

ext = DataLoader.load("./PPMI", modalities=[STUDY_ENROLLMENT, IMAGING])
list(ext[IMAGING])                           # table names (file stems)
ext[IMAGING]["Xing_Core_Lab_-_Quant_SBR"].head()   # core-lab DaTscan striatal binding ratios
codes = load_data_dictionary("./PPMI")["code_list"]
```

## Tests

| Test | Data | What it checks |
|---|---|---|
| `tests/test_data_loader.py::test_data_loader_synthetic` | fake two-folder tree in `tmp_path` | The dict and merged shapes of the example above. |
| `tests/test_data_loader.py::test_data_loader_real_data` | `./PPMI`, marked `ppmi` | Subject characteristics and medical-history tables load. |
| `tests/test_pie_clean.py::test_data_loader` | `./PPMI`, marked `ppmi` | Full load excluding `project_9000`, `project_222`, `project_196`; `biospecimen` is a DataFrame and no excluded project appears in `PROJ_ID`. |

Real-data tests are skipped when `./PPMI` is missing; nothing is loaded at import. PIE-clean's own
suite runs on committed synthetic fixtures, including
`test_merged_output_keeps_biospecimen_only_visits`.

```bash
pytest tests/test_data_loader.py -m "not ppmi" -q
pytest tests/test_data_loader.py tests/test_pie_clean.py -m ppmi -q
```
