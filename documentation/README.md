# PIE documentation

PIE has four layers. Each takes plain tables or image files in and writes plain tables or
image files out, so you can use one without the others.

```
PPMI tabular download ──pie_clean──► cleaned tables ──pipeline──► reduced → engineered → selected → classified
PPMI LONI imaging     ──pie.imaging──► imaging-derived phenotypes (PATNO, EVENT_ID, …) ──┘ (--imaging-features)
analysis frame        ──pie.stats / pie.experiment──► tests, models, manifests
derived images        ──Brain Explorer──► interactive 3-D / slice review in the browser
```

## Where to read

| Layer | Document | Code |
|---|---|---|
| End-to-end tabular pipeline, CLI flags, leakage config | [pipeline.md](pipeline.md) | `pie/pipeline.py`, `config/` |
| Walkthrough: a basic `COHORT` classification, stage by stage | [basic_classification.ipynb](../walkthroughs/basic_classification.ipynb) | — |
| Walkthrough: LONI download → imaging-derived phenotypes | [imaging_features.ipynb](../walkthroughs/imaging_features.ipynb) | — |
| Walkthrough: Brain Explorer, on open data and on your PPMI outputs | [brain_viewer.ipynb](../walkthroughs/brain_viewer.ipynb) | — |
| Loading raw PPMI tables (PIE-clean) | [data_loader.md](data_loader.md) | `pie_clean` (companion package) |
| Cleaning raw PPMI tables (PIE-clean) | [data_preprocessor.md](data_preprocessor.md) | `pie_clean` |
| Dropping low-value columns, merging tables | [data_reducer.md](data_reducer.md) | `pie/data_reducer.py` |
| Encoding, scaling, derived features | [feature_engineer.md](feature_engineer.md) | `pie/feature_engineer.py` |
| Train/test split and feature selection | [feature_selector.md](feature_selector.md) | `pie/feature_selector.py` |
| Model comparison, tuning, HTML reports | [classifier.md](classifier.md) | `pie/classifier.py`, `pie/classification_report.py`, `pie/reporting.py` |
| Classical statistics (tests, regression, survival, LEDD, UPDRS) | [stats.md](stats.md) | `pie/stats/` |
| Cohort rules, nested prediction, provenance manifests | [experiment.md](experiment.md) | `pie/experiment/` |
| Imaging hub: T1/FastSurfer IDPs, FLAIR, manifests, QC, CNN/embeddings, setup | [imaging.md](imaging.md) | `pie/imaging/` |
| Diffusion MRI: free water, tensors, FBA, correction | [imaging_dwi.md](imaging_dwi.md) | `pie/imaging/dwi*.py`, `fba.py` |
| Neuromelanin MRI and DaTscan SPECT | [imaging_nm_datscan.md](imaging_nm_datscan.md) | `pie/imaging/nm*.py`, `datscan.py` |
| Measures matched to the imaging literature: JHU tracts, neuromelanin volume, tissue volumes, basal ganglia network, small-sample statistics | [imaging_literature_parity.md](imaging_literature_parity.md) | `pie/imaging/dwi_tracts.py`, `nm_volume.py`, `volumes.py`, `fmri_striatal.py`, `pie/stats/small_sample.py` |
| Resting-state fMRI: BIDS, fMRIPrep, QC, connectivity | [fmriprep.md](fmriprep.md) | `pie/imaging/fmri*.py` |
| Brain Explorer viewer | [brain_viewer.md](brain_viewer.md) | `pie/imaging/viewer/`, `brain-viewer/` |
| fMRI in the viewer | [fmri_viewer.md](fmri_viewer.md) | `pie/imaging/viewer/fmri.py` |

## Environments

The tabular and imaging layers pin different scientific stacks (the imaging venv follows
FastSurfer's torch/numpy requirements), so they live in separate environments.

| Environment | Used for | Create |
|---|---|---|
| Tabular (Python ≥ 3.10) | `pipeline`, `pie_clean`, `pie.stats`, `pie.experiment` | `pip install -r requirements.txt && pip install -e .` |
| `venv_imaging` (Python 3.12) | `pie.imaging`, fMRI, Brain Explorer backend, `pie.experiment` | `bash scripts/setup_imaging.sh [cu128\|cpu]`, then `venv_imaging/bin/python -m pip install -r pie/imaging/viewer/requirements.txt` for the viewer |
| Node | Brain Explorer frontend | `npm --prefix brain-viewer ci` |

Python ≥ 3.10 is the floor because `endgame-ml` requires it. `setup.py` still says 3.8, but
that isn't enough to run the classifier.

`pie.stats` pulls in `pingouin`, `scikit-posthocs` and `lifelines` only in the functions that
use them. They are in `requirements.txt`, but if you install PIE any other way, the posthoc,
partial-correlation and survival functions raise `ModuleNotFoundError` until you install them.

## Tests

```bash
# tabular pipeline (needs pie_clean)
python -m pytest tests/test_pipeline.py tests/test_data_loader.py tests/test_pie_clean.py \
    tests/test_data_reducer.py tests/test_feature_engineer.py tests/test_feature_selector.py \
    tests/test_classifier.py tests/test_from_fs.py

# statistics
python -m pytest tests/test_stats_*.py

# imaging, fMRI, experiment, viewer backend: everything the imaging venv can import
venv_imaging/bin/python -m pytest tests -q \
    --ignore=tests/test_pipeline.py --ignore=tests/test_data_loader.py \
    --ignore=tests/test_data_reducer.py --ignore=tests/test_pie_clean.py --ignore=tests/test_from_fs.py

# viewer frontend
npm --prefix brain-viewer test
```

The five ignored files import `pie_clean`, which the imaging environment does not have. Ignoring
those rather than listing the files to run means a new test file is picked up automatically: the
previous hand-written list had stopped covering `test_volumes.py`, `test_qc.py`, `test_datscan.py`
and eleven others.

Tests that need the real download are marked `ppmi` and are skipped when `./PPMI` is missing.
`pytest -m "not ppmi"` runs only the synthetic tests, which finish in minutes. `pytest -m ppmi`
runs the real-data integration tests, which load the whole download and write under `output/`.

## Data in documentation and examples

PPMI data is released under a data use agreement. This repository may describe the data, but it
must never identify anyone in it.

- **Never hardcoded:** participant IDs (`PATNO` values), LONI image or series IDs, or any
  participant-level record (one person's values, scan dates or visit history) written into docs,
  code, docstrings, UI text or test fixtures.
- **On screen is different:** the Brain Explorer names the participants in your download, because
  that is the point of it. It reads every one of them from the local catalogue and sample plan at
  runtime, so the identifiers live on the machine that holds the data, never in the repository.
- **Fine:** summary statistics (cohort counts, model metrics, validation results), and PPMI file
  and table names. They help people find their way around the download. Label counts with the
  download they came from, because they change with every release.

Examples in these documents follow one convention:

- synthetic data built inside the snippet (`numpy` with a fixed seed);
- `PATNO` values of 1, 2, 3 … or strings such as `"P001"`;
- dates such as `2000-01-01`, `<IMAGE_ID>` in place of image IDs, and `./PPMI` for the download.

Column names from the PPMI data dictionary (`PATNO`, `EVENT_ID`, `COHORT`, `NP3TOT`, …) are
schema, not data, so they may appear anywhere. Follow the same convention in issues, pull
requests and test fixtures.
