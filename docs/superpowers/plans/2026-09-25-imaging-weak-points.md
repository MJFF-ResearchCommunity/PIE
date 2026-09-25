# PIE Imaging Weak-Point Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every PIE imaging measure shown to the PPMI Imaging Working Group either passes a validity gate stated here in advance on PPMI data, or is labelled "not validated" in the docs and the assembled features.

**Architecture:** One workstream per weak point. Each has three parts: code changes (test-driven, synthetic data), a bounded real-data experiment on a fixed sample with a pre-registered acceptance gate, and a cohort re-run that you execute in your own terminal. The experiments decide which method becomes the default. Each gate is written down before its data are seen, so no gate is tuned to the outcome.

**Tech Stack:**
- Python 3.12 `venv_imaging` (DIPY 1.12, ANTsPy, SimpleITK, nibabel, scipy, PyTorch).
- FSL 6.0.7.23 at `~/fsl`: topup, eddy_cuda / eddy_cpu.
- MRtrix3, and FastSurfer 2.6.0-dev in `third_party/FastSurfer`.
- FreeSurfer 7.3.2 binaries from the fMRIPrep 25.2.5 rootfs (`~/Parkinsons/fmri_local_20260911/runtime/rootfs/opt/freesurfer`, no TensorFlow, no models), with the license at `~/Parkinsons/license.txt`.

**Spec:** the audit of 25 September 2026, recorded in `documentation/imaging.md`, sections "Coverage against the gold standard and the state of the art", "Real-data validity checks (September 2026)" and "Limitations". The weak points are:

| # | Weak point (measured 25 Sep 2026) | Workstream |
|---|---|---|
| W1 | Head size: registration TIV r = 0.81 vs FreeSurfer eTIV, no better than `MaskVol` (0.85) | Phase 1 |
| W2 | No cortical thickness in PIE; PPMI FS7 tables unused | Phase 1 |
| W3 | NM atlas-ROI CNR: PD vs HC AUROC 0.54; template method never completed | Phase 2 |
| W4 | Nigral DWI: FA ρ 0.39 and MD ρ −0.01 vs PPMI manual SN ROIs; no distortion correction on PPMI-1; no eddy by default | Phase 3 |
| W5 | FLAIR WMH (threshold): age ρ +0.18 vs +0.46 for FastSurfer WM-hypointensities; 13 % registration QC failures | Phase 4 |
| W6 | No ALFF / fALFF / ReHo | Phase 5 |
| W7 | No normative (deviation) scores | Phase 6 |
| W8 | No longitudinal FastSurfer stream | Phase 7 |
| W9 | Every table in `Imaging/derived` predates the 9–21 September fixes | Phase 8 |

## Decisions (your call; defaults in bold)

| ID | Question | Default | Consequence |
|---|---|---|---|
| D1 | Who runs jobs longer than 10 minutes? | **You, in your terminal, with commands from this plan** | The harness's low-memory guard kills background runs (memory note `long-runs-outside-harness`) |
| D2 | Reboot to fix the NVIDIA driver/library mismatch (`nvidia-smi`: "Driver/library version mismatch")? | **Yes, before Phase 3 Task 3.6** | eddy_cuda takes ~26 min per three-shell subject; eddy_cpu takes several hours |
| D3 | Cortical thickness beyond PPMI's FS7 tables? | **Pilot FastSurfer surfaces on 5 scans (Task 1.5); run the rest only if the pilot passes** | A full run is ~1 CPU-hour per scan, ~1,850 scans |
| D4 | Normative model | **PPMI-control reference z-scores (Phase 6)**; Brain Charts / PCNtoolkit only if D3 produces Destrieux thickness | Pretrained Brain Charts models need `aparc.a2009s` thickness |
| D5 | Install TrUE-Net (FSL's WMH U-Net, PyTorch) into `venv_imaging` and download its pretrained weights? | **Yes (Task 4.2)** | Adds ~200 MB of weights; no other WMH deep-learning segmenter runs here without a new FreeSurfer install |
| D6 | `sudo apt install tcsh` (FreeSurfer's `talairach_avi` is a tcsh script; `/bin/tcsh` is absent) | **Yes, before Task 1.2** | Without it there is no eTIV path |

## Global Constraints

- No real PPMI identifiers in code, docs, tests or plan files. That means no PATNO, LONI image IDs or participant rows. Sample lists live outside the repo, under `~/pie_validation/`.
- Commands that run more than 10 minutes are run by you (D1). Wrap each one in `systemd-run --user --scope -p MemoryMax=24G …`, and run `df -h /` first: the root disk had 24 GB free on 25 Sep 2026.
- Every `ants.registration` call is followed by `features._drop_transforms(reg)`. No temp files are left behind.
- Template and atlas space is `MNI152NLin2009cAsym` only, through `pie.imaging.atlases` (pinned hashes). Never use nilearn's default MNI templates.
- Tests: run only the affected test files. The full imaging suite takes more than 45 CPU-minutes.
- Nothing is committed unless you ask, and nothing is ever force-pushed. Each task's "Commit" step reads: *stage only; commit when the user says so.*
- Pre-registered gates are copied into the docs with the measured value, pass or fail.
- Existing derived tables are never overwritten. Re-runs write to new directories dated `…_20260927`.

## Review Focus

1. **FreeSurfer runtime absent, or the license path unset.** Every FreeSurfer-backed function must raise one error that names `PIE_FREESURFER_HOME` or `PIE_FS_LICENSE`. It must not emit NaN rows. Test in Task 0.1.
2. **Joining PPMI or FreeSurfer tables to a subject whose T1 session is not the table's visit** (FS7 is baseline only), including LONI-masked 9999 dates. Values must come out NaN, never another visit's values. Test in Task 1.1.
3. **Single-shell and multi-shell diffusion rows mixed in a new metric** (SDC, eddy). They are never pooled silently, and `dwi_batch` still separates them. Test in Task 3.5.
4. **A cohort run killed mid-subject.** Resumable runners must skip finished rows and never write half rows. This covers the new CLIs in Tasks 1.2, 4.2 and 7.1 (each test asserts resume).
5. **External tools writing into `/tmp`** (talairach, TrUE-Net, eddy, recon-surf). Each runs in a per-subject working directory, removed on success. Tests in Tasks 1.2 and 4.2.

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `pie/imaging/freesurfer.py` (new) | Locate FreeSurfer + license, build its environment, run a command | 0.1 |
| `pie/imaging/fastsurfer.py` | + `talairach_etiv`, `etiv_from_xfm`, `surfaces` | 1.2, 1.5 |
| `pie/imaging/features.py` | + `fs7_tables`, eTIV column in `build_idp_table` | 1.1, 1.2 |
| `pie/imaging/run.py` | + `--etiv`, `--surf`, `--long` modes | 1.2, 1.5, 7.1 |
| `pie/imaging/manifest.py` | + FS7 block, `tiv_mm3`/`tiv_source`, NM-template block, FLAIR method choice | 1.1, 1.4, 2.4, 4.4 |
| `pie/imaging/labels.py` | + `ppmi_dti_roi_table` (manual nigral ROIs as a reference) | 3.1 |
| `pie/imaging/dwi.py` | + `register_b0_to_t1_sdc`, `eddy_correct`, the default mapping switch | 3.3, 3.5, 3.6 |
| `pie/imaging/flair.py` | registration fix, `truenet_wmh` | 4.1, 4.2 |
| `pie/imaging/fmri_connectivity.py` | + `alff_falff`, `reho`, `local_measures` | 5.1–5.3 |
| `pie/imaging/normative.py` (new) | Reference-group deviation scores | 6.1 |
| `tests/test_imaging_regressions.py`, `tests/test_fmri_connectivity.py`, `tests/test_normative.py` (new) | Tests | all |
| `documentation/imaging*.md`, `documentation/fmriprep.md` | Results and usage | every task's last step |

## Sequencing

```
Phase 0 (prereqs) ─► Phase 1 (eTIV, FS7, surfaces pilot) ─► Phase 6 (normative; needs TIV)
                 ├─► Phase 2 (NM)          ─┐
                 ├─► Phase 3 (DWI) [3.6 needs D2 reboot]
                 ├─► Phase 4 (FLAIR) [4.2 needs D5]
                 ├─► Phase 5 (fMRI local measures)
                 └─► Phase 7 (longitudinal; needs 1.5 pass) ─► Phase 8 (cohort re-runs + docs)
```
Phases 2–5 are independent of each other and can proceed in any order once Phase 0 is done.

---

## Phase 0 — Prerequisites

### Task 0.0: Machine preparation (you)

- [ ] `sudo apt install tcsh` (D6).
- [ ] Reboot when convenient (D2); afterwards `nvidia-smi -L` lists the GPU.
- [ ] `df -h /` shows ≥ 20 GB free.

### Task 0.1: FreeSurfer runtime helper

**Files:** Create `pie/imaging/freesurfer.py`; Test `tests/test_imaging_regressions.py`

**Interfaces:**
- Produces: `freesurfer.fs_env(home=None, license=None) -> dict` (environment for subprocess); `freesurfer.run(cmd, log, cwd, env=None, timeout=3600) -> None` (raises `RuntimeError` on nonzero exit); module constants `HOME`, `LICENSE` read from `PIE_FREESURFER_HOME`, `PIE_FS_LICENSE` (falling back to `FS_LICENSE`).

- [ ] **Step 1: Write the failing test**

```python
def test_freesurfer_env_names_missing_settings_and_builds_path(tmp_path, monkeypatch):
    from pie.imaging import freesurfer
    with pytest.raises(RuntimeError, match="PIE_FREESURFER_HOME"):
        freesurfer.fs_env(home=None, license=str(tmp_path / "license.txt"))
    home = tmp_path / "fs"; (home / "bin").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="PIE_FS_LICENSE"):
        freesurfer.fs_env(home=str(home), license=str(tmp_path / "absent.txt"))
    (tmp_path / "license.txt").write_text("x\n")
    env = freesurfer.fs_env(home=str(home), license=str(tmp_path / "license.txt"))
    assert env["FREESURFER_HOME"] == str(home) and env["FS_LICENSE"] == str(tmp_path / "license.txt")
    assert env["PATH"].split(os.pathsep)[0] == str(home / "bin") and env["SUBJECTS_DIR"]
```

- [ ] **Step 2: Run to verify it fails.** Run: `venv_imaging/bin/python -m pytest -q tests/test_imaging_regressions.py -k freesurfer_env`. Expected: FAIL, `ModuleNotFoundError: pie.imaging.freesurfer`.

- [ ] **Step 3: Implement**

```python
"""freesurfer.py — run FreeSurfer 7.x command-line tools from a local install or an extracted container rootfs.

PIE_FREESURFER_HOME: the FreeSurfer tree (e.g. the fMRIPrep 25.2.5 rootfs's opt/freesurfer, 7.3.2; it has no
TensorFlow or models, so SynthSeg and SAMSEG lesion mode do not run from it). PIE_FS_LICENSE (else FS_LICENSE): the
license file. Nothing here is imported by default pipelines; callers opt in.
"""
import os
import subprocess
from pathlib import Path

HOME = os.environ.get("PIE_FREESURFER_HOME")
LICENSE = os.environ.get("PIE_FS_LICENSE") or os.environ.get("FS_LICENSE")


def fs_env(home=None, license=None):
    home, license = home or HOME, license or LICENSE
    if not home or not (Path(home) / "bin").is_dir():
        raise RuntimeError(f"FreeSurfer not found: set PIE_FREESURFER_HOME (got {home!r})")
    if not license or not Path(license).is_file():
        raise RuntimeError(f"FreeSurfer license not found: set PIE_FS_LICENSE (got {license!r})")
    h = Path(home)
    path = os.pathsep.join([str(h / "bin"), str(h / "mni" / "bin"), os.environ.get("PATH", "")])
    return {**os.environ, "FREESURFER_HOME": str(h), "FREESURFER": str(h), "FS_LICENSE": str(license), "PATH": path,
            "SUBJECTS_DIR": str(h / "subjects"), "MNI_DIR": str(h / "mni"), "MINC_BIN_DIR": str(h / "mni" / "bin"),
            "MINC_LIB_DIR": str(h / "mni" / "lib"), "MNI_DATAPATH": str(h / "mni" / "data"),
            "PERL5LIB": str(h / "mni" / "lib" / "perl5" / "5.8.5"), "FS_OVERRIDE": "0", "OMP_NUM_THREADS": "1"}


def run(cmd, log, cwd, env=None, timeout=3600):
    """Run one FreeSurfer command, appending its output to ``log``; RuntimeError on a nonzero exit."""
    with open(log, "a") as fh:
        fh.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env or fs_env(), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{Path(str(cmd[0])).name} exited {r.returncode}; see {log}")
```

- [ ] **Step 4: Run the test.** Expected: PASS.
- [ ] **Step 5: Stage** `pie/imaging/freesurfer.py tests/test_imaging_regressions.py`.

---

## Phase 1 — Head size (W1) and cortical thickness (W2)

### Task 1.1: PPMI FreeSurfer 7 / MRIQC tables as features

**Files:** Modify `pie/imaging/features.py`, `pie/imaging/manifest.py`; Test `tests/test_imaging_regressions.py`

**Interfaces:**
- Produces: `features.fs7_tables(ppmi_dir) -> DataFrame` keyed `PATNO`, `EVENT_ID`, with columns `fs7_cth_<region>`, `fs7_sa_<region>`, `fs7_<aseg column>` (includes `fs7_EstimatedTotalIntraCranialVol`) and `mriqc_<metric>`; `manifest.assemble_features(..., ppmi_dir=None)` joins it where the manifest's T1 `EVENT_ID` equals the table's.

- [ ] **Step 1: Failing test** (synthetic tables; subject 2's T1 is a V04 session, so FS7 (BL) must not attach):

```python
def test_fs7_tables_attach_only_to_the_same_visit(tmp_path):
    im = tmp_path / "PPMI/Imaging"; im.mkdir(parents=True)
    pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": "BL", "lh_cuneus": [2.1, 2.2]}).to_csv(im / "FS7_APARC_CTH_01Jan2020.csv", index=False)
    pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": "BL", "lh_cuneus": [900., 950.]}).to_csv(im / "FS7_APARC_SA_01Jan2020.csv", index=False)
    pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": "BL", "EstimatedTotalIntraCranialVol": [1.5e6, 1.6e6]}).to_csv(im / "FS7_ASEG_VOL_01Jan2020.csv", index=False)
    pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": "BL", "cnr": [3.0, 3.1]}).to_csv(im / "MRIQC_01Jan2020.csv", index=False)
    t = features.fs7_tables(tmp_path / "PPMI").set_index("PATNO")
    assert t.loc[1, "fs7_cth_lh_cuneus"] == 2.1 and t.loc[1, "fs7_sa_lh_cuneus"] == 900 and t.loc[2, "mriqc_cnr"] == 3.1
    pd.DataFrame({"PATNO": [1, 2], "EVENT_ID": ["BL", "V04"], "IMAGEID": ["I1", "I2"], "SCAN_DATE": ["2020-01-01"] * 2,
                  "vol_Left_Putamen": [1., 1.]}).to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    f = manifest.assemble_features(tmp_path, ppmi_dir=tmp_path / "PPMI").set_index("PATNO")
    assert f.loc[1, "fs7_EstimatedTotalIntraCranialVol"] == 1.5e6 and np.isnan(f.loc[2, "fs7_EstimatedTotalIntraCranialVol"])
```

- [ ] **Step 2: Run to verify it fails** (`AttributeError: fs7_tables`).
- [ ] **Step 3: Implement** in `features.py`:

```python
FS7_TABLES = {"FS7_APARC_CTH": "fs7_cth_", "FS7_APARC_SA": "fs7_sa_", "FS7_ASEG_VOL": "fs7_", "MRIQC": "mriqc_"}


def fs7_tables(ppmi_dir):
    """PPMI's own FreeSurfer 7.3.2 (DK atlas thickness and area, aseg volumes with eTIV) and MRIQC tables, one row per
    PATNO + EVENT_ID, prefixed by source. Baseline scans only in the 2025 McGill/Nipoppy release."""
    from .viewer.catalog import latest_table

    out = None
    for stem, prefix in FS7_TABLES.items():
        path = latest_table(Path(ppmi_dir, "Imaging"), stem)
        if path is None:
            continue
        t = pd.read_csv(path, low_memory=False).drop_duplicates(["PATNO", "EVENT_ID"])
        t = t.rename(columns={c: prefix + c for c in t.columns if c not in ("PATNO", "EVENT_ID")})
        out = t if out is None else out.merge(t, on=["PATNO", "EVENT_ID"], how="outer")
    return out if out is not None else pd.DataFrame(columns=["PATNO", "EVENT_ID"])
```

In `manifest.assemble_features`, add parameter `ppmi_dir=None`. After building `df`, add the block:

```python
    if ppmi_dir is not None:
        from .features import fs7_tables
        fs7 = fs7_tables(ppmi_dir)
        df = df.merge(fs7.rename(columns={"EVENT_ID": "_fs7_event"}), left_on=["PATNO", "EVENT_ID"],
                      right_on=["PATNO", "_fs7_event"], how="left").drop(columns="_fs7_event")
```
(`EVENT_ID` reaches `df` through the IDP merge; `_baseline_idps` keeps it.)
- [ ] **Step 4: Run the test** → PASS; also run `tests/test_manifest.py tests/test_imaging_audit.py`.
- [ ] **Step 5: Docs.** Add an `fs7_*` row to the assembled-features paragraph in `imaging.md`, and note that the tables use DK (FastSurfer uses DKT). Stage.

### Task 1.2: FreeSurfer eTIV via FastSurfer's talairach registration

**Files:** Modify `pie/imaging/fastsurfer.py`, `pie/imaging/features.py`, `pie/imaging/run.py`; Test `tests/test_imaging_regressions.py`

**Interfaces:**
- Consumes: `freesurfer.fs_env`, `freesurfer.run` (Task 0.1).
- Produces:
  - `fastsurfer.etiv_from_xfm(xfm_path) -> float`: FreeSurfer's definition, 1948106 / det(the linear part of `talairach.xfm`) (Buckner 2004 atlas scaling, as `mri_segstats --etiv` computes it).
  - `fastsurfer.talairach_etiv(subjects_dir, sid, three_t=False, env=None) -> float`: runs `recon_surf/talairach-reg.sh` once and writes `mri/transforms/talairach.xfm`.
  - `build_idp_table` adds column `eTIV` when that file exists.
  - `run.py --etiv`: backfills every finished subject; resumable, since subjects that already have the xfm are skipped.

- [ ] **Step 1: Failing test**

```python
def test_etiv_follows_freesurfer_atlas_scaling_and_backfill_resumes(tmp_path, monkeypatch):
    from pie.imaging import fastsurfer
    xfm = tmp_path / "talairach.xfm"
    xfm.write_text("MNI Transform File\nTransform_Type = Linear;\nLinear_Transform =\n"
                   "1.1 0 0 1.0\n0 1.1 0 2.0\n0 0 1.1 3.0;\n")
    assert abs(fastsurfer.etiv_from_xfm(xfm) - 1948106 / 1.1 ** 3) < 1e-6
    calls = []
    def fake_run(cmd, log, cwd, env=None, timeout=3600):
        calls.append(cmd)
        out = Path(cwd) / "transforms"; out.mkdir(exist_ok=True); shutil.copy(xfm, out / "talairach.xfm")
    monkeypatch.setattr(fastsurfer.freesurfer, "run", fake_run)
    sub = tmp_path / "fs" / "S1" / "mri"; sub.mkdir(parents=True)
    for f in ("orig.mgz", "orig_nu.mgz", "aparc.DKTatlas+aseg.deep.mgz"): (sub / f).touch()
    cwds = []
    monkeypatch.setattr(fastsurfer.freesurfer, "run", lambda cmd, log, cwd, env=None, timeout=3600: (cwds.append(Path(cwd)), fake_run(cmd, log, cwd)))
    v = fastsurfer.talairach_etiv(tmp_path / "fs", "S1", env={})
    assert abs(v - 1948106 / 1.331) < 1e-3 and "talairach-reg.sh" in str(calls[0][0])
    assert cwds == [(tmp_path / "fs" / "S1" / "mri").resolve()]    # works inside the subject, nothing in /tmp
    fastsurfer.talairach_etiv(tmp_path / "fs", "S1", env={})
    assert len(calls) == 1                                        # resumes: an existing xfm is not recomputed
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** in `fastsurfer.py`:

```python
from . import freesurfer

ETIV_SCALE = 1948106.0   # FreeSurfer's atlas scale factor (mri_segstats --etiv)


def etiv_from_xfm(xfm_path):
    """eTIV (mm^3) = 1948106 / det(linear part of the MNI305 talairach.xfm), FreeSurfer's definition."""
    lines = Path(xfm_path).read_text().split("Linear_Transform =", 1)[1].replace(";", " ").split()
    m = np.array([float(v) for v in lines[:12]]).reshape(3, 4)[:, :3]
    return float(ETIV_SCALE / np.linalg.det(m))


def talairach_etiv(subjects_dir, sid, three_t=False, env=None, fastsurfer_home=FASTSURFER_HOME, python=PYTHON):
    """FastSurfer's talairach registration (FreeSurfer talairach_avi on orig_nu) for one finished subject; returns eTIV.
    Idempotent: an existing mri/transforms/talairach.xfm is reused."""
    mri = Path(subjects_dir).resolve() / sid / "mri"
    xfm = mri / "transforms" / "talairach.xfm"
    if not xfm.exists():
        cmd = [Path(fastsurfer_home) / "recon_surf" / "talairach-reg.sh", mri.parent / "scripts" / "talairach.log",
               "--py", python, "--asegdkt_segfile", mri / "aparc.DKTatlas+aseg.deep.mgz", "--dir", mri,
               "--conformed_name", mri / "orig.mgz", "--norm_name", mri / "orig_nu.mgz"] + (["--3T"] if three_t else [])
        (mri.parent / "scripts").mkdir(exist_ok=True)
        freesurfer.run(cmd, mri.parent / "scripts" / "pie_talairach.log", mri, env=env if env is not None else freesurfer.fs_env())
    return etiv_from_xfm(xfm)
```
(`import numpy as np` at the top of `fastsurfer.py`.) In `features.build_idp_table`, after `row.update(measures)`, add:

```python
        xfm = Path(subjects_dir) / s.image_id / "mri" / "transforms" / "talairach.xfm"
        row["eTIV"] = etiv_from_xfm(xfm) if xfm.exists() else np.nan
```
(import `etiv_from_xfm` beside `parse_stats`). In `run.py`, add `ap.add_argument("--etiv", action="store_true", help="add FreeSurfer eTIV (talairach) to every finished subject; needs PIE_FREESURFER_HOME, PIE_FS_LICENSE and tcsh")`. Handle it after sessions load, before segmentation:

```python
def _etiv_job(args):                       # module level: ProcessPoolExecutor pickles functions by name
    fs_dir, sid = args
    try:
        return sid, talairach_etiv(fs_dir, sid), ""
    except Exception as e:
        return sid, float("nan"), f"{type(e).__name__}: {str(e)[:200]}"

# in main(), after `done` is computed:
    if a.etiv:
        from concurrent.futures import ProcessPoolExecutor
        sids = sorted(done)
        with ProcessPoolExecutor(a.workers) as ex:
            for i, (sid, v, err) in enumerate(ex.map(_etiv_job, [(work / "fastsurfer", s) for s in sids]), 1):
                if err or i % 50 == 0:
                    print(f"{i}/{len(sids)} {sid} {err or round(v)}", flush=True)
        a.features_only = True             # rebuild fastsurfer_idps.csv with the eTIV column
```
(`from pie.imaging.fastsurfer import talairach_etiv` joins the existing import line.)
- [ ] **Step 4: Run tests** → PASS (`tests/test_imaging_regressions.py tests/test_imaging.py`).
- [ ] **Step 5: Stage.**

### Task 1.3: Experiment E1 — eTIV validity (gate stated before running)

**Gate:** Pearson r ≥ 0.95 between PIE eTIV and PPMI `FS7_ASEG_VOL.EstimatedTotalIntraCranialVol` on the same 40 baseline scans as the 25 Sep TIV check (`random_state=0`), and |mean difference| < 5 %. Test both with and without `--3T`; keep the setting closer to FS7.

- [ ] Run (≈ 40 × 2 min, 8 workers, < 10 min):

```bash
export PIE_FREESURFER_HOME=~/Parkinsons/fmri_local_20260911/runtime/rootfs/opt/freesurfer PIE_FS_LICENSE=~/Parkinsons/license.txt
mkdir -p ~/pie_validation/20260926 && cd /home/cameron/PIE
venv_imaging/bin/python - <<'EOF'
import pandas as pd
a = pd.read_csv("PPMI/Imaging/FS7_ASEG_VOL_08Sep2026.csv")[["PATNO", "EstimatedTotalIntraCranialVol"]]
idp = pd.read_csv("Imaging/derived/fastsurfer_idps.csv", low_memory=False)
idp = idp[idp.EVENT_ID.eq("BL")].drop_duplicates("PATNO").merge(a, on="PATNO").sample(40, random_state=0)
idp[["IMAGEID", "EstimatedTotalIntraCranialVol"]].to_csv("/home/cameron/pie_validation/20260926/etiv_sample.csv", index=False)
EOF
PYTHONPATH=. venv_imaging/bin/python - <<'EOF'
import pandas as pd, numpy as np
from concurrent.futures import ProcessPoolExecutor
from pie.imaging.run import _etiv_job
s = pd.read_csv("/home/cameron/pie_validation/20260926/etiv_sample.csv")
with ProcessPoolExecutor(8) as ex:
    res = dict((sid, (v, err)) for sid, v, err in ex.map(_etiv_job, [("Imaging/derived/fastsurfer", i) for i in s.IMAGEID]))
s["pie_etiv"] = s.IMAGEID.map(lambda i: res[i][0])
print("errors:", [e for _, e in res.values() if e][:3])
ok = s.pie_etiv.notna()
print("r", np.corrcoef(s.pie_etiv[ok], s.EstimatedTotalIntraCranialVol[ok])[0, 1], "mean rel diff", (s.pie_etiv / s.EstimatedTotalIntraCranialVol - 1)[ok].mean())
EOF
```
- [ ] Record the result in `imaging.md`'s validity table. **If the gate fails:** open 3 overlays of `talairach.xfm`-registered `orig_nu` against `mni305.cor.mgz` (`freeview`, or a `qc.montage` of the resampled image) and look for a failed registration before changing anything.

### Task 1.4: One TIV column for analyses

**Files:** Modify `pie/imaging/manifest.py`; Test `tests/test_imaging_regressions.py`

**Interfaces:** Produces `tiv_mm3` and `tiv_source` in `assemble_features` output. The priority order is `eTIV` (PIE, same T1 as the volumes) → `fs7_EstimatedTotalIntraCranialVol` (same visit) → NaN. `tiv_source` takes `pie_talairach | ppmi_fs7 | none`.

- [ ] **Step 1: Failing test**

```python
def test_tiv_prefers_pie_etiv_then_same_visit_fs7(tmp_path):
    pd.DataFrame({"PATNO": [1, 2, 3], "EVENT_ID": ["BL", "BL", "V04"], "IMAGEID": ["I1", "I2", "I3"], "SCAN_DATE": ["2020-01-01"] * 3,
                  "vol_Left_Putamen": [1.] * 3, "eTIV": [1.4e6, np.nan, np.nan]}).to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    im = tmp_path / "PPMI/Imaging"; im.mkdir(parents=True)
    pd.DataFrame({"PATNO": [1, 2, 3], "EVENT_ID": "BL", "EstimatedTotalIntraCranialVol": [1.5e6, 1.6e6, 1.7e6]}).to_csv(im / "FS7_ASEG_VOL_01Jan2020.csv", index=False)
    f = manifest.assemble_features(tmp_path, ppmi_dir=tmp_path / "PPMI").set_index("PATNO")
    assert (f.loc[1, "tiv_mm3"], f.loc[1, "tiv_source"]) == (1.4e6, "pie_talairach")
    assert (f.loc[2, "tiv_mm3"], f.loc[2, "tiv_source"]) == (1.6e6, "ppmi_fs7")
    assert np.isnan(f.loc[3, "tiv_mm3"]) and f.loc[3, "tiv_source"] == "none"
```
- [ ] **Step 2: Fail.** **Step 3: Implement** at the end of `assemble_features`:

```python
    pie_tiv = df["eTIV"] if "eTIV" in df else pd.Series(np.nan, index=df.index)
    fs7_tiv = df["fs7_EstimatedTotalIntraCranialVol"] if "fs7_EstimatedTotalIntraCranialVol" in df else pd.Series(np.nan, index=df.index)
    df["tiv_mm3"] = pie_tiv.fillna(fs7_tiv)
    df["tiv_source"] = np.where(pie_tiv.notna(), "pie_talairach", np.where(fs7_tiv.notna(), "ppmi_fs7", "none"))
```
- [ ] **Step 4: Pass.** **Step 5: Docs.** Replace the head-size advice in `imaging.md` ("Tissue volumes and head-size adjustment") with `tiv_mm3`, and keep `tiv_from_registration` documented as superseded. Stage.

### Task 1.5: Surface stream pilot (D3)

**Gate:** On 5 baseline scans that also appear in `FS7_APARC_CTH`, FastSurfer thickness (with `--fsaparc`, which gives the DK `aparc`) must correlate with PPMI's FS7 thickness at a median across the 68 regions of r ≥ 0.80. The per-region comparison pools the 5 subjects × 68 regions, reported per region as the absolute difference, plus an overall r over 340 pairs ≥ 0.85.

- [ ] Implement `fastsurfer.surfaces(subjects_dir, sid, threads=4, env=None)`:

```python
def surfaces(subjects_dir, sid, threads=4, env=None, fastsurfer_home=FASTSURFER_HOME, python=PYTHON):
    """FastSurfer recon-surf (surfaces, DK aparc via --fsaparc, cortical thickness) on a segmented subject. Needs
    FreeSurfer binaries; the fMRIPrep rootfs has 7.3.2 while FastSurfer 2.6 expects 7.4.1, hence --ignore_fs_version
    (validated only by Task 1.5's pilot). Idempotent: stats/lh.aparc.stats marks completion."""
    sd = Path(subjects_dir).resolve()
    if (sd / sid / "stats" / "lh.aparc.stats").exists():
        return sd / sid
    env = env if env is not None else freesurfer.fs_env()
    cmd = [Path(fastsurfer_home) / "recon_surf" / "recon-surf.sh", "--sid", sid, "--sd", sd, "--t1", sd / sid / "mri" / "orig.mgz",
           "--asegdkt_segfile", sd / sid / SEG_FILE, "--fs_license", env["FS_LICENSE"], "--threads", threads, "--parallel",
           "--fsaparc", "--py", python, "--ignore_fs_version"]
    freesurfer.run(cmd, sd / sid / "scripts" / "pie_recon_surf.log", sd, env=env, timeout=4 * 3600)
    return sd / sid
```
Unit test (write it first; it fails with `AttributeError: surfaces`):

```python
def test_surface_stream_command_and_resume(tmp_path, monkeypatch):
    from pie.imaging import fastsurfer
    calls = []
    def fake_run(cmd, log, cwd, env=None, timeout=3600):
        calls.append([str(c) for c in cmd])
        (Path(cwd) / "S1" / "stats").mkdir(parents=True, exist_ok=True); (Path(cwd) / "S1" / "stats" / "lh.aparc.stats").touch()
    monkeypatch.setattr(fastsurfer.freesurfer, "run", fake_run)
    (tmp_path / "S1" / "scripts").mkdir(parents=True)
    fastsurfer.surfaces(tmp_path, "S1", env={"FS_LICENSE": "/lic.txt"})
    fastsurfer.surfaces(tmp_path, "S1", env={"FS_LICENSE": "/lic.txt"})
    assert len(calls) == 1 and "--fsaparc" in calls[0] and "--ignore_fs_version" in calls[0] and "/lic.txt" in calls[0]
```
- [ ] Pilot (you, ~1 h per scan, 5 in parallel): `for i in <5 ids>; do systemd-run --user --scope -p MemoryMax=8G venv_imaging/bin/python -c "from pie.imaging.fastsurfer import surfaces; surfaces('Imaging/derived/fastsurfer', '$i')" & done; wait`. Put the ids in `~/pie_validation/20260926/surf_sample.txt`, never in the repo.
- [ ] Compare with a short script: read `stats/lh.aparc.stats` / `rh.aparc.stats` (`ThickAvg` column) against `fs7_cth_lh_<region>`.
- [ ] **Pass:** add `run.py --surf` (loops `surfaces` over finished subjects with `--workers`) and a `features.parse_aparc_stats` → `cth_<hemi>_<region>` columns in `build_idp_table`. Document it; the cohort run moves to Phase 8. **Fail:** document "cortical thickness = PPMI FS7 tables (baseline); FastSurfer surfaces need FreeSurfer 7.4.1" and stop. D4 falls back to PPMI-control z-scores.

---

## Phase 2 — Neuromelanin (W3)

**Gate (pre-registered):** the primary NM measure is the method whose bilateral SN CNR gives HC > PD AUROC ≥ 0.70 with the 95 % bootstrap CI lower bound > 0.55, on the fixed sample of 45 HC + 45 PD. The methods are `nm_template` (`nmt_sn_mean_cnr`) and the current `nm.py` (`nm_sn_mean_cnr`, processing version v5). If both pass, the higher AUROC wins. If neither passes, NM is marked not validated (Task 2.5).

### Task 2.1: Fixed sample and current `nm.py` on it

- [ ] Build the sample outside the repo: all QC-passing HC, plus an equal random PD sample (`random_state=0`), from `Imaging/derived/nm/nm_features.csv`. Write `~/pie_validation/20260926/nm/patnos.txt`, and copy `Imaging/derived/nm/nm_index.csv` into `~/pie_validation/20260926/nm/` so the 45 GB zips are not re-indexed:

```bash
cd /home/cameron/PIE && V=~/pie_validation/20260926/nm && mkdir -p $V && cp Imaging/derived/nm/nm_index.csv $V/
PYTHONPATH=. venv_imaging/bin/python - <<'EOF'
import pandas as pd
from pie.imaging.labels import covariates
from pie.imaging.manifest import QC
d = pd.read_csv("Imaging/derived/nm/nm_features.csv"); d = d[d.error.fillna("") == ""]; d = d[QC["nm"](d)]
d = d.merge(covariates("PPMI")[["PATNO", "COHORT"]], left_on="patno", right_on="PATNO")
hc = d[d.COHORT == "Healthy Control"].patno
pdp = d[d.COHORT == "Parkinson's Disease"].patno.sample(len(hc), random_state=0)
open("/home/cameron/pie_validation/20260926/nm/patnos.txt", "w").write("\n".join(map(str, list(hc) + list(pdp))))
EOF
```
- [ ] Run current `nm.py` (you, ~20 min):

```bash
Z="/media/cameron/Seagate Portable Drive/PPMI/Imaging"
systemd-run --user --scope -p MemoryMax=24G venv_imaging/bin/python -m pie.imaging.nm \
  --zips "$Z/First_Study_MRI_Full_NM.zip" "$Z/First_Study_MRI_Full_NM_dataset.zip" "$Z/First_Study_NM_Missing2.zip" "$Z/First_Study_NM_Missing2_dataset.zip" \
  --sessions Imaging/derived/sessions.csv --fastsurfer-dir Imaging/derived/fastsurfer --work-dir ~/pie_validation/20260926/nm \
  --patnos ~/pie_validation/20260926/nm/patnos.txt --workers 8 --keep-nifti
```
(Confirm the zip list against the `zip` column of `nm_index.csv` before running.)

### Task 2.2: `nm_template` on the same sample

- [ ] Run the four stages in order (you; `syn` ≈ 45 new registrations × 7.5 CPU-min, 8 workers ≈ 45 min; 46 are cached from 25 Sep):

```bash
for s in syn normalize template features; do systemd-run --user --scope -p MemoryMax=24G venv_imaging/bin/python -m pie.imaging.nm_template $s \
  --sessions Imaging/derived/sessions.csv --fastsurfer-dir Imaging/derived/fastsurfer --work-dir ~/pie_validation/20260926/nm --workers 8; done
```
- [ ] Look at `~/pie_validation/20260926/nm/template/template_qc.png`. The red SN mask must lie on the bright band, and the lime crus mask anterolateral in the dark peduncle. If not, stop and fix the masks before any analysis.

### Task 2.3: Analysis against the gate

- [ ] A script in `~/pie_validation/20260926/nm/analyse.py` (outside the repo) reports, for `nmt_sn_mean_cnr`, `nmt_sn_lat_mean_cnr`, `nmt_sn_post_mean_cnr`, `nm_sn_mean_cnr` and `nm_sn_posterior_mean_cnr`:
  - HC and PD mean (SD);
  - AUROC for HC > PD with a 2,000-resample bootstrap 95 % CI (seed 0);
  - Cohen's d;
  - Spearman ρ with age in HC.
- [ ] Apply the gate and write the result into `imaging.md` (validity table) and `imaging_nm_datscan.md`, pass or fail.

### Task 2.4: NM-template features in the assembled table

**Files:** Modify `pie/imaging/manifest.py`; Test `tests/test_imaging_regressions.py`

**Interfaces:** Adds `QC["nmt"]`: `nmt_sn_cov_l >= 0.9 & nmt_sn_cov_r >= 0.9 & nmt_crus_cv_l < 0.3 & nmt_crus_cv_r < 0.3`. `assemble_features` reads `<nm dir>/nm_template_features.csv` when present, keeps the `nmt_*_cnr` columns, and blanks them when `nmt_qc_pass` is False. `feature_blocks()["nm"]` includes `nmt_*`.

- [ ] **Step 1: Failing test**

```python
def test_assembly_reads_nm_template_features_with_their_own_qc(tmp_path):
    _idps(tmp_path)
    (tmp_path / "nm").mkdir()
    pd.DataFrame({"patno": [1, 2], "error": ["", ""], "nmt_sn_mean_cnr": [.2, .3], "nmt_sn_cov_l": [1., .5], "nmt_sn_cov_r": [1., 1.],
                  "nmt_crus_cv_l": [.1, .1], "nmt_crus_cv_r": [.1, .1], "nmt_crus_mode_l": [100., 100.]}).to_csv(tmp_path / "nm" / "nm_template_features.csv", index=False)
    f = manifest.assemble_features(tmp_path).set_index("PATNO")
    assert f.loc[1, "nmt_sn_mean_cnr"] == .2 and np.isnan(f.loc[2, "nmt_sn_mean_cnr"]) and not f.loc[2, "nmt_qc_pass"]
    assert "nmt_crus_mode_l" not in f and "nmt_sn_mean_cnr" in manifest.feature_blocks(f.columns)["nm"]
```
- [ ] **Step 2: Fail.** **Step 3: Implement.** Add the QC lambda. In `assemble_features`, after the `nm` block:

```python
    nmt = _read(_modality_dir(derived, "nm", modality_dirs) / "nm_template_features.csv")
    if nmt is not None:
        nmt["nmt_qc_pass"] = QC["nmt"](nmt)
        cols = [c for c in nmt.columns if c.startswith("nmt_") and c.endswith("_cnr")]
        nmt.loc[~nmt["nmt_qc_pass"], cols] = np.nan
        df = df.merge(nmt[["patno", "nmt_qc_pass"] + cols].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
```
Change `feature_blocks` to `"nm": [c for c in cols if c.startswith(("nm_", "nmt_")) and c.endswith(("_cnr", "_voxels"))]`.
- [ ] **Step 4: Pass** (with `tests/test_manifest.py`). **Step 5: Stage.**

### Task 2.5: If neither NM method passes

- [ ] Render the worst 16 by `reg_nm_t1_mi` and a random 16: `venv_imaging/bin/python -m pie.imaging.qc --work-dir ~/pie_validation/20260926/nm --modality nm --out ~/pie_validation/20260926/nm/qc --n 32 --worst reg_nm_t1_mi`. Classify each as registration wrong, slab not covering the SN, or anatomy right but no contrast.
- [ ] Add `nm_validated = False` to `manifest.assemble_features` output, which the NM block writes, and state in the docs that NM measures are exploratory. Test: `assert not f["nm_validated"].any()` when the constant is False.

---

## Phase 3 — Nigral diffusion accuracy (W4)

**Reference measurement:** PPMI's hand-drawn nigral ROIs (`DTI_Regions_of_Interest_*.csv`; CIND; 263 subjects). **Gate for any mapping or correction change:** on the fixed validation set (up to 120 scans with a same-month manual ROI, `random_state=0`):
- Spearman ρ with manual SN FA rises by ≥ 0.05 over the current affine mapping.
- ρ with manual SN MD becomes > 0.20 (currently −0.01).
- The median absolute FA bias shrinks.

### Task 3.1: Manual ROI table reader

**Files:** Modify `pie/imaging/labels.py`; Test `tests/test_imaging_regressions.py`

**Interfaces:** Produces `labels.ppmi_dti_roi_table(ppmi_dir) -> DataFrame[PATNO, DTI_DATE (month), sn_fa, sn_md, sn_rostral_fa, sn_caudal_fa, peduncle_fa]`. SN = mean of the six ROIs, caudal = ROI3 and ROI6 (the PPMI methods doc: I rostral, II middle, III caudal per side). MD = mean of E1–E3.

- [ ] **Step 1: Failing test**

```python
def test_ppmi_manual_dti_rois_become_one_row_per_scan(tmp_path):
    im = tmp_path / "PPMI/Imaging"; im.mkdir(parents=True)
    rows = []
    for m, v in (("FA", .3), ("E1", 1.2e-3), ("E2", .7e-3), ("E3", .5e-3)):
        rows.append({"PATNO": 1, "PAG_NAME": "DTIROI", "INFODT": "01/2011", "Measure": m, "Tissue": "SN",
                     **{f"ROI{i}": v + (0.01 if i in (3, 6) and m == "FA" else 0) for i in range(1, 7)}, "REF1": .6, "REF2": .62, "RUNDATE": "2015-01-01"})
    pd.DataFrame(rows).to_csv(im / "DTI_Regions_of_Interest_01Jan2020.csv", index=False)
    t = labels.ppmi_dti_roi_table(tmp_path / "PPMI").iloc[0]
    assert abs(t.sn_fa - (0.3 + 0.02 / 6)) < 1e-9 and abs(t.sn_caudal_fa - 0.31) < 1e-9 and abs(t.sn_md - 0.8e-3) < 1e-12
    assert abs(t.peduncle_fa - 0.61) < 1e-9 and t.DTI_DATE == pd.Timestamp("2011-01-01")
```
- [ ] **Step 2: Fail.** **Step 3: Implement**

```python
def ppmi_dti_roi_table(ppmi_dir):
    """PPMI's hand-drawn nigral DTI ROIs (Schuff et al. 2015; three per side: I rostral, II middle, III caudal; two
    cerebral-peduncle references), one row per PATNO and scan month: a reference for validating automated SN values."""
    t = _latest(ppmi_dir, "Imaging", "DTI_Regions_of_Interest")
    roi = [f"ROI{i}" for i in range(1, 7)]
    t[roi + ["REF1", "REF2"]] = t[roi + ["REF1", "REF2"]].apply(pd.to_numeric, errors="coerce")
    t["DTI_DATE"] = _month(t["INFODT"])
    w = t.pivot_table(index=["PATNO", "DTI_DATE"], columns="Measure", values=roi + ["REF1", "REF2"], aggfunc="first")
    out = pd.DataFrame(index=w.index)
    out["sn_fa"] = w.xs("FA", axis=1, level=1)[roi].mean(axis=1)
    out["sn_rostral_fa"] = w.xs("FA", axis=1, level=1)[["ROI1", "ROI4"]].mean(axis=1)
    out["sn_caudal_fa"] = w.xs("FA", axis=1, level=1)[["ROI3", "ROI6"]].mean(axis=1)
    out["sn_md"] = sum(w.xs(e, axis=1, level=1)[roi].mean(axis=1) for e in ("E1", "E2", "E3")) / 3
    out["peduncle_fa"] = w.xs("FA", axis=1, level=1)[["REF1", "REF2"]].mean(axis=1)
    return out.reset_index()
```
- [ ] **Step 4: Pass. Step 5: Stage.**

### Task 3.2: Experiment E4a — affine vs SyN atlas mapping on saved maps

- [ ] Build the validation set, up to 120 same-month matches of `ppmi_dti_roi_table` and `Imaging/derived/dwi` subjects that have `fa.nii.gz`. Symlink each subject's `fa md fw fat b0 aseg_dwi pauli_dwi .nii.gz` *files* into fresh directories under `~/pie_validation/20260926/dwi/<patno>/`, so no output lands in `Imaging/derived`.
- [ ] Run `dwi_refine` (you, ~120 × 5 CPU-min, 8 workers ≈ 75 min): `systemd-run --user --scope -p MemoryMax=24G venv_imaging/bin/python -m pie.imaging.dwi_refine --work-dir ~/pie_validation/20260926/dwi --sessions Imaging/derived/sessions.csv --fastsurfer-dir Imaging/derived/fastsurfer --workers 8`.
- [ ] Compare `sn_mean_fa`, `sn_t_mean_fa`, `sn_posterior_mean_fa` and `sn_mean_md` from `Imaging/derived/dwi/dwi_features.csv` (affine) and from `dwi_features_syn.csv` (SyN) against the manual values. Record ρ, mean bias and n. Caveat for the docs: these saved maps predate the 9 September gradient-rotation fix. That fix affects FA slightly, but it affects both arms equally.

### Task 3.3: Fieldmap-less distortion correction for single-shell scans

**Files:** Modify `pie/imaging/dwi.py`; Test `tests/test_imaging_regressions.py`

**Interfaces:** Produces `dwi.register_b0_to_t1_sdc(b0_img, t1_img, t1_mask_img, pe_axis, seed=0) -> (labels_fn, info)`. `labels_fn(label_img) -> np.ndarray (z, y, x)` pulls a T1-space label image onto the b0 grid through a rigid + phase-encoding-restricted SyN (ANTs `SyNOnly`, `restrict_transformation` a 3-vector with 1 on the physical axis closest to the PE direction). `info = {"sdc_metric", "sdc_max_displacement_mm"}`. Transforms are deleted after the call by `features._drop_transforms`. The approach is fMRIPrep's fieldmap-less SyN-SDC idea (Wang et al. 2017), restricted to the PE axis.

- [ ] **Step 1: Failing test** (synthetic head: a T1-like block image and a "b0" with inverted contrast, shifted smoothly along y only; SDC must recover the label better than rigid, and must not move along x or z):

```python
def test_pe_restricted_sdc_recovers_a_phase_encoding_shift():
    shape = (40, 48, 40)
    zz, yy, xx = np.indices(shape)
    t1 = np.zeros(shape, np.float32); t1[8:32, 8:40, 8:32] = 1.0; t1[16:24, 18:30, 14:26] = 2.0     # "nucleus"
    lab = np.zeros(shape, np.int16); lab[16:24, 18:30, 14:26] = 7
    shift = 3.0 * np.exp(-((xx - 20) ** 2 + (zz - 20) ** 2) / 200.0)                                 # voxels along y
    from scipy.ndimage import map_coordinates
    b0 = map_coordinates(3.0 - t1, [zz, yy - shift, xx], order=1).astype(np.float32)                   # inverted contrast
    aff = np.diag([1., 1., 1., 1.])
    to_img = lambda a: nib.Nifti1Image(np.transpose(a, (2, 1, 0)), aff)                              # stored (x, y, z)
    fn, info = dwi.register_b0_to_t1_sdc(to_img(b0), to_img(t1), to_img((t1 > 0).astype(np.float32)), pe_axis="j")
    got = fn(to_img(lab)) == 7
    truth = map_coordinates((lab == 7).astype(float), [zz, yy - shift, xx], order=0) > 0.5
    dice = 2 * (got & truth).sum() / (got.sum() + truth.sum())
    assert dice > 0.8 and info["sdc_max_displacement_mm"] < 8 and info["sdc_offaxis_max_mm"] < 0.1
    shutil.rmtree(info["transform_dir"])
```
- [ ] **Step 2: Fail.** **Step 3: Implement**

```python
def register_b0_to_t1_sdc(b0_img, t1_img, t1_mask_img, pe_axis, seed=0):
    """Rigid b0 -> T1, then a SyN restricted to the phase-encoding direction (fieldmap-less susceptibility correction,
    after fMRIPrep's SyN-SDC). ``pe_axis``: 'i', 'j' or 'k' of the b0 grid (sign irrelevant). Returns a function that
    pulls T1-space label images onto the b0 grid, and QC. Temporary ANTs transforms are deleted before returning."""
    import tempfile

    import ants

    from .features import _to_ants

    tdir = tempfile.mkdtemp(prefix="pie_sdc_")             # ANTs writes here, not loose into /tmp; caller removes it
    fixed = _to_ants(b0_img)
    t1 = np.asarray(t1_img.dataobj, np.float32) * (np.asarray(t1_mask_img.dataobj) > 0)
    moving = _to_ants(nib.Nifti1Image(t1, t1_img.affine))
    rigid = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid", aff_metric="mattes",
                              random_seed=int(seed), outprefix=f"{tdir}/rigid_")
    axis_vox = {"i": 0, "j": 1, "k": 2}[pe_axis[0]]
    pe_phys = int(np.argmax(np.abs(np.asarray(b0_img.affine)[:3, axis_vox])))   # physical axis closest to the PE axis
    restrict = tuple(float(i == pe_phys) for i in range(3))
    syn = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyNOnly", initial_transform=rigid["fwdtransforms"][0],
                            syn_metric="mattes", restrict_transformation=restrict, flow_sigma=4, total_sigma=0,
                            reg_iterations=(100, 50, 20), random_seed=int(seed), outprefix=f"{tdir}/syn_")
    transforms = syn["fwdtransforms"]
    field = ants.image_read(transforms[0]).numpy()                                  # (x, y, z, 3) displacement, mm
    off_axis = [i for i in range(3) if i != pe_phys]
    info = {"sdc_max_displacement_mm": float(np.abs(field[..., pe_phys]).max()),
            "sdc_offaxis_max_mm": float(np.abs(field[..., off_axis]).max()), "transform_dir": tdir}

    def labels_fn(label_img):
        out = ants.apply_transforms(fixed=fixed, moving=_to_ants(label_img), transformlist=transforms, interpolator="genericLabel")
        return np.transpose(np.rint(out.numpy()).astype(np.int32), (2, 1, 0))

    return labels_fn, info
```
Callers `shutil.rmtree(info["transform_dir"])` when done with `labels_fn`, and so does the test. Add `assert info["sdc_offaxis_max_mm"] < 0.1` to the test. If that fails because `restrict_transformation` is ignored for a SyN stage, pass `**{"restrict_deformation": "x".join(str(int(r)) for r in restrict)}` instead and re-run. Per ANTsPy's docs, restriction works only without a preceding stage in the same call, which is why the rigid runs in its own call.
- [ ] **Step 4: Pass. Step 5: Stage.**

### Task 3.4: Experiment E4b — SDC on the validation set

- [ ] For the Task 3.2 scans, compute SN/posterior-SN FA and MD from the saved maps with labels pulled by `register_b0_to_t1_sdc`. The CIT168 chain becomes `[MNI→T1 affine or SyN] + SDC`; FastSurfer labels come through SDC. Use `pe_axis` from `pe_direction` in `dwi_features.csv` (default `j`). Write a script under `~/pie_validation/20260926/dwi/`; it runs < 10 min per 30 subjects with 8 workers, so run it in chunks or in your terminal.
- [ ] Apply the Phase 3 gate to affine, SyN, affine+SDC and SyN+SDC. The winner is the variant with the highest ρ_FA among those passing.

### Task 3.5: Make the winning mapping the default

**Files:** Modify `pie/imaging/dwi.py` (`process_subject`, `PROCESSING_VERSION` constant); Test `tests/test_imaging_regressions.py`

- [ ] **Step 1: Failing test.** `process_subject` records `atlas_mapping` (`affine | syn | affine_sdc | syn_sdc`) and `processing_version == "2026-09-27-<winner>-v4"`. The manifest's `dwi_batch` appends `atlas_mapping`, so old and new rows never share a batch:

```python
def test_dwi_batch_separates_mapping_versions(tmp_path):
    pd.DataFrame({"PATNO": [1, 2], "IMAGEID": ["I1", "I2"], "SCAN_DATE": ["2022-01-01"] * 2, "vol_Left_Putamen": [1., 1.]}).to_csv(tmp_path / "fastsurfer_idps.csv", index=False)
    (tmp_path / "dwi").mkdir()
    pd.DataFrame({"patno": [1, 2], "motion_mm_max": [1., 1.], "n_sn_l": [4, 4], "n_sn_r": [4, 4], "fa_wm_median": [.4, .4], "manufacturer": ["Siemens"] * 2,
                  "shells": ["1000"] * 2, "fw_method": ["singleshell_prior"] * 2, "atlas_mapping": ["affine", "affine_sdc"], "sn_l_fa": [.4, .4]}).to_csv(tmp_path / "dwi" / "dwi_features.csv", index=False)
    m = manifest.build_manifest(tmp_path)
    assert m["dwi_batch"].nunique() == 2
```
- [ ] **Step 3: Implement.** In `build_manifest`: `dwi["dwi_batch"] = ... + "_" + dwi.get("atlas_mapping", pd.Series("affine", index=dwi.index)).astype(str)`. In `process_subject`, replace the label pulls with the winning variant behind a module constant `ATLAS_MAPPING = "<winner>"` and record `row["atlas_mapping"] = ATLAS_MAPPING`.
- [ ] **Step 4: Pass**, including `tests/test_dwi.py`. **Step 5: Docs** (`imaging_dwi.md` pipeline step 8, plus the E4a/E4b table). Stage.

### Task 3.6: eddy for scans with a reverse-PE b0 (after the D2 reboot)

**Files:** Modify `pie/imaging/dwi.py`; Test `tests/test_imaging_regressions.py`

**Interfaces:**
- Consumes: `dwi_correction.build_eddy_command`, `dwi_correction.validate_corrected_geometry`, `dwi_correction.topup_config`, `dwi_acquisition.pe_row`.
- Produces: `dwi.eddy_correct(ds, work_dir, cuda=True) -> ds` (corrected data, eddy-rotated bvecs, `eddy=True`, `eddy_outlier_fraction`, `motion_mm_max` from `eddy.eddy_movement_rms`). `process_subject(..., eddy=False)` with CLI `--eddy`. When `eddy` is set and a reverse-PE b0 exists, it replaces both `susceptibility_correct` and `preprocess`'s rigid motion correction. Gradients are rotated exactly once (by eddy).

- [ ] **Step 1: Failing test** (fake FSL on a temporary `FSLDIR`):

```python
def test_eddy_correct_consumes_eddy_outputs_once_and_cleans_up(tmp_path, monkeypatch):
    fsl = tmp_path / "fsl" / "bin"; fsl.mkdir(parents=True)
    (fsl / "topup").write_text("#!/bin/sh\nfor a in \"$@\"; do case $a in --out=*) o=${a#--out=};; esac; done\n"
                               "cp b0_pair.nii.gz ${o}_fieldcoef.nii.gz; printf '0 0 0 0 0 0\\n0 0 0 0 0 0\\n' > ${o}_movpar.txt\n")
    (fsl / "eddy_cuda").write_text("#!/bin/sh\ncp raw.nii.gz eddy.nii.gz\n"
        "python3 -c \"import numpy as n; v=n.loadtxt('bvecs'); v[:, 2:] = n.roll(v[:, 2:], 1, axis=0); n.savetxt('eddy.eddy_rotated_bvecs', v)\"\n"
        "printf 'hdr\\n0 1\\n0 0\\n0 0\\n0 0\\n' > eddy.eddy_outlier_map\nprintf '0 0.5\\n0 0.7\\n0 0.2\\n0 0.1\\n' > eddy.eddy_movement_rms\n")
    for f in fsl.iterdir(): f.chmod(0o755)
    monkeypatch.setattr(dwi, "FSLDIR", str(tmp_path / "fsl"))
    g = np.eye(3)[:, [0, 1]]
    ds = {"data": np.ones((4, 4, 4, 4), np.float32), "affine": np.eye(4), "bvals": np.array([0., 0., 1000., 1000.]),
          "bvecs": np.c_[np.zeros((3, 2)), g], "meta": {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05},
          "rev_b0": np.ones((4, 4, 4, 1), np.float32), "rev_meta": {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05}}
    out = dwi.eddy_correct(ds, tmp_path / "work", cuda=True)
    assert out["eddy"] and out["bvecs_rotated"] and abs(out["eddy_outlier_fraction"] - 1 / 8) < 1e-9
    assert np.allclose(out["bvecs"][:, 2:], np.roll(g, 1, axis=0)) and out["data"].shape == ds["data"].shape
    assert not (tmp_path / "work" / "eddy").exists()
```
- [ ] **Step 3: Implement.** Write `raw.nii.gz` (uncorrected), `bvals`, `bvecs`, `index.txt` (all 1), `acqparams.txt` (two rows from `pe_row(meta)` / `pe_row(rev_meta)`), and `eddy_mask.nii.gz` (median-Otsu on the mean b0). Run topup on the b0 pair with `topup_config(shape)` (no cropping), then `build_eddy_command(FSLDIR + "/bin", meta, n_slices, use_topup=True, raw_is_uncorrected=True, cuda=cuda)`. Run it with a 2-hour timeout, read the outputs, validate the geometry and delete the directory. Record `eddy_cmd_record` (the slice-timing decision) in the row.
- [ ] **Step 4: Pass. Pilot** (you, after the reboot): 5 three-shell subjects, `--eddy --fsl --keep-nifti` into `~/pie_validation/20260926/dwi_eddy/`. **Gate:**
  - median `eddy_outlier_fraction` < 0.05;
  - `fa_wm_median` within ±0.03 of the rigid-only run;
  - posterior-SN multi-shell FW age trend kept (checked in the Phase 8 cohort).

  On pass, `--eddy` becomes the default for scans with a reverse-PE b0 (`processing_version` bump). Stage.

---

## Phase 4 — FLAIR white-matter hyperintensities (W5)

### Task 4.1: Registration failures

- [ ] **Diagnose** (you, < 10 min): re-run 30 of the 177 QC-failing and 10 passing subjects with `--keep-nifti` into `~/pie_validation/20260926/flair/`, sample `random_state=0`, patnos outside the repo. Then `python -m pie.imaging.qc --work-dir ~/pie_validation/20260926/flair --modality flair --out …/qc --n 40 --worst reg_flair_t1_mi`. Classify each: true misregistration vs aligned but MI > −0.2 (a threshold problem), and 2D vs 3D.
- [ ] **Fix per the diagnosis.**
  - If misregistered: in `register_flair_to_t1`, use the *unmasked* conformed T1 as the fixed image (the NM lesson: skull and eyes pin the pose). Try `MOMENTS` and `GEOMETRY` initialisers and keep the better final metric. For 2D FLAIR (slice > 3 mm), register at 3 mm.
  - If aligned: replace the MI threshold in `QC["flair"]` with the brain-mask Dice between the resampled FLAIR (Otsu) and the T1 mask. That needs `flair_mask_dice` in the row; gate ≥ 0.85.
- [ ] **Test** (synthetic): a FLAIR-like volume rotated 10° and shifted 8 mm with a head outside the brain registers back to < 1 mm error with the new code. **Gate:** < 3 % of the 177 fail after the fix, and the gallery shows alignment. Stage.

### Task 4.2: TrUE-Net WMH segmentation (D5)

**Files:** Modify `pie/imaging/flair.py`; Test `tests/test_imaging_regressions.py`

**Interfaces:** Produces `flair.truenet_wmh(flair_t1_img, t1_img, brain_mask_img, model="mwsc", work_dir=None) -> (lesion_mask (z, y, x) bool, {"wmh_truenet_mm3", "wmh_truenet_n_lesions"})`. This is a thin wrapper around the `truenet evaluate` CLI of Sundaresan et al. 2021 (NeuroImage 244:118583, triplanar U-Net trained on the MICCAI WMH challenge and UK Biobank). The input is FLAIR (+T1) prepared by `prepare_truenet_data`; the output probability map is thresholded at 0.5 and components < `MIN_LESION_MM3` are dropped. The working directory is removed afterwards.

- [ ] **Install spike** (separate step, recorded in `imaging.md` Setup):
  1. `venv_imaging/bin/pip install git+https://git.fmrib.ox.ac.uk/fsl/truenet.git` (verify the URL and licence first; if unreachable, use FSL's `update_fsl_package fsl_truenet`).
  2. Download the `mwsc` and `ukbb` pretrained models to `third_party/weights/truenet/`, with SHA-256 recorded in `WEIGHTS.md`.
  3. Run it on 1 scan.
- [ ] **Step 1: Failing test** with a fake `truenet` executable on `PATH` that writes a probability NIfTI with two blobs (one of 3 voxels, one of 40) to the expected output name. Assert the 40-voxel lesion survives, the 3-voxel one is dropped at 1 mm, the volume is right, and `work_dir` is gone afterwards.
- [ ] **Step 3: Implement** the wrapper, plus `process_subject(..., wmh_method="threshold")` with `--wmh-method {threshold,truenet}`. The row records `wmh_method`.
- [ ] **Step 4: Pass. Stage.**

### Task 4.3: Experiment E5 — which WMH measure

**Gate (pre-registered):** the primary vascular covariate is the measure with the highest Spearman ρ with age on 60 QC-passing subjects (age-stratified, 20 per tertile, `random_state=0`), provided ρ ≥ 0.35 and 10 overlays show lesions on hyperintense white matter, not on cortex, septum or choroid. The candidates are the threshold WMH, TrUE-Net WMH, and FastSurfer `vol_WM_hypointensities` (no FLAIR needed; currently ρ = 0.46).

- [ ] Run both FLAIR methods on the 60 (you, < 30 min), compute ρ, and render 10 overlays each.
- [ ] Record the result. If FastSurfer's T1 measure wins, document it as primary and keep the FLAIR measures as secondary.

### Task 4.4: Assembled features follow the winner

- [ ] Add `manifest.PRIMARY_WMH` (set by the Task 4.3 result: `"vol_WM_hypointensities"`, `"flair_wmh_log_mm3"` or `"flair_wmh_truenet_mm3"`). `assemble_features` adds `flair_wmh_truenet_*` when present, and `feature_blocks()` gains `"vascular": [PRIMARY_WMH] + other WMH columns`. Test first:

```python
def test_vascular_block_lists_the_validated_wmh_measure_first(monkeypatch):
    monkeypatch.setattr(manifest, "PRIMARY_WMH", "vol_WM_hypointensities")
    cols = ["flair_wmh_log_mm3", "vol_WM_hypointensities", "flair_wmh_truenet_mm3", "dwi_sn_l_fa"]
    assert manifest.feature_blocks(cols)["vascular"] == ["vol_WM_hypointensities", "flair_wmh_log_mm3", "flair_wmh_truenet_mm3"]
```
Implementation in `feature_blocks`: `wmh = [c for c in cols if c.startswith("flair_wmh") or c == "vol_WM_hypointensities"]; blocks["vascular"] = sorted(wmh, key=lambda c: c != PRIMARY_WMH)`. Stage.

---

## Phase 5 — Local fMRI measures (W6)

### Task 5.1: ALFF and fALFF

**Files:** Modify `pie/imaging/fmri_connectivity.py`; Test `tests/test_fmri_connectivity.py`

**Interfaces:** Produces `alff_falff(series, tr, keep, band=(0.01, 0.08)) -> (alff, falff)`:
- `series` has shape time × columns (the full length);
- `keep` is the boolean retained-frame mask from `nuisance_design`;
- censored frames are filled by linear interpolation across retained neighbours before the FFT (the XCP-D approach);
- ALFF is the mean amplitude of the one-sided spectrum in `band`, and fALFF is ALFF's band sum over the sum across (0, Nyquist].

- [ ] **Step 1: Failing test**

```python
def test_alff_falff_prefer_the_slow_band_and_survive_censoring():
    from pie.imaging.fmri_connectivity import alff_falff
    tr, n = 1.0, 600
    t = np.arange(n) * tr
    rng = np.random.default_rng(0)
    y = np.column_stack([np.sin(2 * np.pi * 0.05 * t), np.sin(2 * np.pi * 0.2 * t)]) + 0.1 * rng.normal(size=(n, 2))
    keep = np.ones(n, bool)
    a, f = alff_falff(y, tr, keep)
    assert a[0] > 5 * a[1] and f[0] > 0.5 > f[1]
    keep[rng.choice(n, 120, replace=False)] = False
    a2, f2 = alff_falff(np.where(keep[:, None], y, 1e6), tr, keep)       # censored values must not leak in
    assert abs(a2[0] / a[0] - 1) < 0.15 and f2[0] > 0.5
```
- [ ] **Step 2: Fail.** **Step 3: Implement**

```python
def _interpolate_censored(series, keep):
    """Linear interpolation of censored frames from retained neighbours (edges held at the nearest retained value)."""
    series = np.asarray(series, float).copy()
    kept = np.flatnonzero(keep)
    if len(kept) < 2:
        raise ValueError("fewer than two retained frames")
    for t in np.flatnonzero(~np.asarray(keep, bool)):
        j = np.searchsorted(kept, t)
        p, q = kept[max(j - 1, 0)], kept[min(j, len(kept) - 1)]
        w = 0.0 if q == p else (t - p) / (q - p)
        series[t] = (1 - w) * series[p] + w * series[q]
    return series


def alff_falff(series, tr, keep, band=(0.01, 0.08)):
    """ALFF (mean amplitude in ``band``) and fALFF (band amplitude / amplitude over (0, Nyquist]) per column, on
    nuisance residuals with censored frames interpolated first (XCP-D's order). Zang et al. 2007; Zou et al. 2008."""
    y = _interpolate_censored(series, keep)
    y = y - y.mean(axis=0)
    amp = np.abs(np.fft.rfft(y, axis=0)) * 2 / len(y)
    freq = np.fft.rfftfreq(len(y), d=tr)
    inband, total = (freq >= band[0]) & (freq <= band[1]), freq > 0
    return amp[inband].mean(axis=0), amp[inband].sum(axis=0) / amp[total].sum(axis=0)
```
- [ ] **Step 4: Pass. Step 5: Stage.**

### Task 5.2: ReHo (Kendall's W, 27-voxel neighbourhood)

- [ ] **Step 1: Failing test**

```python
def test_reho_is_high_for_coherent_neighbourhoods_and_low_for_noise():
    from pie.imaging.fmri_connectivity import reho
    rng = np.random.default_rng(0)
    n, shape = 200, (6, 6, 6)
    common = rng.normal(size=n)
    data = rng.normal(size=shape + (n,))
    data[:3] = common + 0.05 * rng.normal(size=(3, 6, 6, n))      # coherent half
    mask = np.ones(shape, bool)
    w = reho(data, mask, np.ones(n, bool))
    assert w[1, 3, 3] > 0.9 and w[4, 3, 3] < 0.2
```
- [ ] **Step 3: Implement**

```python
def reho(data, mask, keep):
    """Regional homogeneity (Zang et al. 2004): Kendall's W of each in-mask voxel and its in-mask 26 neighbours over
    the retained frames. ``data`` is (x, y, z, t). Returns an (x, y, z) map, NaN outside the mask."""
    from scipy import ndimage
    from scipy.stats import rankdata

    y = np.asarray(data, float)[..., np.asarray(keep, bool)]
    n = y.shape[-1]
    ranks = np.where(mask[..., None], rankdata(y, axis=-1), 0.0)
    kernel = np.ones((3, 3, 3))
    k = ndimage.convolve(mask.astype(float), kernel, mode="constant")
    rsum = np.stack([ndimage.convolve(ranks[..., t], kernel, mode="constant") for t in range(n)], axis=-1)
    s = ((rsum - k[..., None] * (n + 1) / 2) ** 2).sum(axis=-1)
    w = 12 * s / (k ** 2 * (n ** 3 - n))
    return np.where(mask & (k > 1), w, np.nan)
```
- [ ] **Step 4: Pass. Stage.**

### Task 5.3: `local_measures` on fMRIPrep derivatives

**Interfaces:** Produces `local_measures(bold, brain_mask, confounds_tsv, confounds_json, atlas, *, tr, config=ConnectivityConfig()) -> dict`. It reuses `nuisance_design` (same censoring and regressors) and returns parcel-mean `alff_<id>`, `falff_<id>`, `reho_<id>`, plus the `temporal` audit. ReHo is computed on unsmoothed residuals, before any smoothing.

- [ ] **Step 1: Failing test.** Reuse `tests/test_fmri_connectivity.py`'s synthetic BOLD/confounds fixture (read it first) and assert keys, finite values for covered parcels, and that `temporal["retained_frames"]` matches `extract_connectivity`'s.
- [ ] **Step 3: Implement.** Load as in `extract_connectivity`, residualise the in-mask voxel series on `design` for the kept rows (the censored rows stay NaN, then interpolate), call `alff_falff` on the voxels and `reho` on the residual 4D. Average per parcel on the atlas resampled to the BOLD grid (nearest).
- [ ] **Step 4: Pass.**
- [ ] **E6 plausibility (you, < 10 min):** 10 subjects from `~/Parkinsons/fmri_local_20260911/fmriprep` with `atlases.schaefer400_mni2009c()`. Expected: ALFF and ReHo in grey-matter parcels exceed the white-matter-adjacent limbic parcels, and the posterior cingulate / precuneus rank in the top quartile for ReHo. Record the result, then stage.

---

## Phase 6 — Normative deviation scores (W7)

### Task 6.1: `normative.py`

**Files:** Create `pie/imaging/normative.py`; Test `tests/test_normative.py`

**Interfaces:** Produces:
- `fit(frame, features, covariates, reference) -> dict`: per feature, OLS on reference rows with an intercept, the covariates and age² when `"age"` is present; stores the coefficients and the residual SD;
- `zscores(frame, model) -> DataFrame` with columns `z_<feature>`.

The covariates are numeric columns the caller encodes (age, sex 0/1, `tiv_mm3`, one-hot batch). Fitting on reference rows only (controls, or the training fold) keeps it CV-safe.

- [ ] **Step 1: Failing test**

```python
import numpy as np, pandas as pd
from pie.imaging import normative

def test_normative_z_is_standard_in_controls_and_detects_a_shift():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"age": rng.uniform(50, 80, n), "sex": rng.integers(0, 2, n), "tiv_mm3": rng.normal(1.5e6, 1e5, n)})
    df["vol"] = 5000 - 20 * (df.age - 65) + 300 * df.sex + 0.002 * (df.tiv_mm3 - 1.5e6) + rng.normal(0, 100, n)
    ref = np.arange(n) < 400
    df.loc[~ref, "vol"] -= 200                                  # patients: 2 SD smaller
    z = normative.zscores(df, normative.fit(df, ["vol"], ["age", "sex", "tiv_mm3"], ref))["z_vol"]
    assert abs(z[ref].mean()) < 0.05 and abs(z[ref].std() - 1) < 0.1 and abs(z[~ref].mean() + 2) < 0.2
    assert abs(np.corrcoef(z[ref], df.age[ref])[0, 1]) < 0.05
```
- [ ] **Step 3: Implement**

```python
"""normative.py — deviation (z) scores against a reference group, e.g. PPMI healthy controls.

A linear model per feature (intercept, covariates, age^2 when an 'age' covariate is given) fitted on the reference
rows only; z = (observed - predicted) / residual SD of the reference. Fit inside cross-validation folds on the training
controls. Pretrained lifespan models (Rutherford et al. 2022, eLife) need FreeSurfer Destrieux thickness, which PIE
does not produce unless the surface stream runs.
"""
import numpy as np
import pandas as pd


def _design(frame, covariates):
    X = [np.ones(len(frame))] + [frame[c].to_numpy(float) for c in covariates]
    if "age" in covariates:
        X.append((frame["age"].to_numpy(float) - 65.0) ** 2)
    return np.column_stack(X)


def fit(frame, features, covariates, reference):
    ref = np.asarray(reference, bool)
    X = _design(frame, covariates)
    model = {"covariates": list(covariates), "features": {}}
    for f in features:
        y = frame[f].to_numpy(float)
        ok = ref & np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() <= X.shape[1] + 2:
            raise ValueError(f"{f}: too few reference rows")
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        resid = y[ok] - X[ok] @ beta
        model["features"][f] = (beta, float(resid.std(ddof=X.shape[1])))
    return model


def zscores(frame, model):
    X = _design(frame, model["covariates"])
    return pd.DataFrame({f"z_{f}": (frame[f].to_numpy(float) - X @ beta) / sd for f, (beta, sd) in model["features"].items()},
                        index=frame.index)
```
- [ ] **Step 4: Pass.** **Step 5: Docs.** An `imaging.md` example on synthetic data, plus a real-data note once `tiv_mm3` exists (Phase 8). Stage.

---

## Phase 7 — Longitudinal FastSurfer (W8)

**Precondition:** Task 1.5 passed. Only 29 subjects have ≥ 2 T1 sessions in the current download, so this phase delivers the tool and a pilot, not a cohort result.

### Task 7.1: Time-point lists and the `--long` runner

**Interfaces:** Produces:
- `run.long_timepoints(sessions, subjects_dir) -> {patno: [image_id, ...]}`: ≥ 2 finished, non-masked sessions ordered by date;
- `run.py --long`: calls `third_party/FastSurfer/long_fastsurfer.sh --tid <patno> --t1s <niftis> --tpids <image_ids> --sd <work>/fastsurfer_long --fs_license <license> --py <python> --3T?`. The flag set is copied from `long_fastsurfer.sh --help` in the implementation step; the test asserts the exact list.

A finished subject (`<tid>/stats/aseg.stats` present) is skipped.

- [ ] **Step 1: Failing test**

```python
def test_long_timepoints_need_two_real_dated_finished_sessions(tmp_path):
    for sid in ("A", "B", "C", "D"):
        (tmp_path / sid / "stats").mkdir(parents=True); (tmp_path / sid / "stats" / "aseg+DKT.stats").touch()
    sessions = pd.DataFrame({"patno": [1, 1, 1, 2], "image_id": ["B", "A", "C", "D"],
                             "session_date": ["2012-01-01", "2010-01-01", "9999-01-01", "2010-01-01"]})
    assert run.long_timepoints(sessions, tmp_path) == {1: ["A", "B"]}
```
Implementation:

```python
def long_timepoints(sessions, subjects_dir):
    """PATNO -> finished FastSurfer image IDs, oldest first, for subjects with >= 2 real-dated sessions."""
    s = sessions.assign(d=pd.to_datetime(sessions["session_date"], errors="coerce"))
    s = s[s["d"].dt.year.lt(2100) & s["image_id"].map(lambda i: (Path(subjects_dir) / str(i) / STATS_FILE).exists())]
    out = {int(p): g.sort_values("d")["image_id"].tolist() for p, g in s.groupby("patno")}
    return {p: ids for p, ids in out.items() if len(ids) >= 2}
```
The `--long` runner test monkeypatches `freesurfer.run`. It asserts the argument list starts with `long_fastsurfer.sh --tid 1 --t1s` and that a second call is skipped once `fastsurfer_long/1/stats/aseg.stats` exists.
- [ ] **Step 3: Implement. Step 4: Pass.**
- [ ] **Pilot (you, GPU + ~3 h per subject):** 3 of the 29. Compare cross-sectional vs longitudinal hippocampal volume change: the longitudinal version should have a smaller within-subject SD of the annualised change. Document it. Stage.

---

## Phase 8 — Cohort re-runs and documentation (W9)

All runs are yours (D1). Outputs go to new directories so the 7–8 September tables stay for comparison:
```bash
D=/home/cameron/PIE/Imaging/derived        # T1 / FastSurfer stay here
N="/media/cameron/Seagate Portable Drive/PPMI/Imaging/derived_20260927"; Z="/media/cameron/Seagate Portable Drive/PPMI/Imaging"
export PIE_FREESURFER_HOME=~/Parkinsons/fmri_local_20260911/runtime/rootfs/opt/freesurfer PIE_FS_LICENSE=~/Parkinsons/license.txt
```

### Task 8.1: Runs, in order (estimated wall time with 8–10 workers)

- [ ] eTIV: `venv_imaging/bin/python -m pie.imaging.run --zips Imaging/MRI_First_Study.zip Imaging/MRI_First_Study_dataset.zip --work-dir $D --etiv --workers 10` (~1,850 × 2 CPU-min, ≈ 6 h).
- [ ] NM: `nm` then the four `nm_template` stages with `--work-dir "$N/nm" --keep-nifti` (≈ 3 h + 8 h).
- [ ] DWI: `dwi --work-dir "$N/dwi" --keep-nifti --fsl --denoise [--eddy]` with the Phase 3 defaults (≈ 12–24 h; eddy adds ~26 min per three-shell subject on the GPU).
- [ ] FLAIR: `flair --work-dir "$N/flair" --keep-nifti --wmh-method <winner>` (≈ 4 h).
- [ ] DaT: `datscan --requantify` is not needed (no code change); keep `datscan_full`.
- [ ] Surfaces (only if Task 1.5 passed and you choose D3 = full): `run --surf` (≈ 1,850 CPU-h; days).
- [ ] Every command runs under `systemd-run --user --scope -p MemoryMax=24G`, with `df -h /` checked first.

### Task 8.2: Validity table from the new tables

- [ ] Re-run the 25 September checks with `modality_dirs={"dwi": N/dwi, "nm": N/nm, "flair": N/flair}`: age trends, PD vs HC AUROC, agreement with PPMI references. Replace every number in `imaging.md` → "Real-data validity checks", and delete the "written before the 9–21 September fixes" caveat.
- [ ] Update `Limitations` so that only unresolved items remain.

### Task 8.3: Review

- [ ] Run all affected test files (the union of the files named in this plan).
- [ ] Run the whole-branch review (superpowers:requesting-code-review), then commit when you say so.

---

## Self-review notes

- **Coverage:** W1 (Tasks 1.2–1.4), W2 (1.1, 1.5), W3 (2.1–2.5), W4 (3.1–3.6), W5 (4.1–4.4), W6 (5.1–5.3), W7 (6.1), W8 (7.1), W9 (8.1–8.2).
- **Deep-learning WMH segmentation** is Task 4.2 (TrUE-Net). SynthSeg- and SAMSEG-lesion options need a full FreeSurfer ≥ 7.4 with TensorFlow, which is not on this machine; they are not planned.
- **Contingent branches are decision rules, not open items.** 1.5 fail → FS7 tables only. 2.x fail → `nm_validated = False`. 3.x → the gate picks the mapping. 4.3 → the gate picks the covariate.
