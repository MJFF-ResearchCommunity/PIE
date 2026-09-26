"""
Subject manifest and wide feature table across PIE's imaging modalities.

Each modality pipeline chooses its own session and writes its own CSV under the derived directory. This module
joins them per subject into (a) a *manifest* — which session of each modality was used, its date, the interval to
the T1 session and to the DaTscan, the scanner batch of each modality, and a QC flag per modality — and (b) an
*assembled feature table* with modality prefixes (`vol_*`/FastSurfer measures for T1, `dat_*`, `dwi_*`, `nm_*`,
`flair_*`), QC-failed values blanked. Studies then only add labels and cohort logic.

    from pie.imaging.manifest import build_manifest, assemble_features
    man = build_manifest("Imaging/derived")            # one row per subject
    feats = assemble_features("Imaging/derived")        # one row per subject, manifest columns + features
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

def _col(d, c):
    return pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)


# QC rules per modality (the same thresholds the study used); keep in one place
T1_KEY_VOLUMES = ["vol_Left_Putamen", "vol_Right_Putamen", "vol_Left_Caudate", "vol_Right_Caudate", "vol_Left_Thalamus", "vol_Right_Thalamus"]
QC = {
    "t1": lambda d: (d[[c for c in T1_KEY_VOLUMES if c in d]] > 0).all(axis=1),   # a failed FastSurfer run leaves empty labels (0 mm3): ratios/asymmetries undefined
    "dat": lambda d: (d["reg_metric"].abs() >= 0.4) & (d["n_label_voxels"] >= 100),
    # the study's nigral diffusion rule (split-half ICC FA 0.88, MD 0.97 under it): >= 20 SN voxels, >= 95 % of the SN in
    # the brain mask, >= 80 % physically admissible tensor fits, mean motion <= 2 mm, both posterior SN halves measured;
    # tables written before these columns existed fail it (re-run them)
    "dwi": lambda d: ((d["n_sn_l"] + d["n_sn_r"] >= 20) & (_col(d, "sn_brain_mask_fraction") >= 0.95)
                      & (_col(d, "sn_physical_fraction") >= 0.8) & (_col(d, "motion_mm_mean") <= 2)
                      & np.isfinite(_col(d, "sn_posterior_l_fa")) & np.isfinite(_col(d, "sn_posterior_r_fa")) & (d["fa_wm_median"] > 0.25)),
    "nm": lambda d: (d["n_sn_l"] >= 20) & (d["n_sn_r"] >= 20) & (d["sn_slab_coverage"] >= 0.5) & (d["repeat_motion_mm_max"] < 3)
    & (d["nm_ref_l_sd"] < 0.4 * d["nm_ref_l_mean"]) & (d["nm_ref_r_sd"] < 0.4 * d["nm_ref_r_mean"]),   # reference ring partly outside the slab -> CV ~1, CNR garbage
    "flair": lambda d: (d["reg_flair_t1_mi"] < -0.2) & (d["wm_mm3"] > 200000) & (d["flair_wm_mad"] > 0),
    # nm_template: the template SN mask has data on both sides and the crus reference is homogeneous
    "nmt": lambda d: (d["nmt_sn_cov_l"] >= 0.9) & (d["nmt_sn_cov_r"] >= 0.9) & (d["nmt_crus_cv_l"] < 0.3) & (d["nmt_crus_cv_r"] < 0.3),
}
# Neither NM measure passed its pre-registered PD-vs-HC gate (AUROC >= 0.70, CI lower bound > 0.55) on 45 HC + 45 PD
# (26 September 2026: nm_template 0.58 [0.46, 0.70], nm.py 0.54 [0.42, 0.67]); assembled NM columns are exploratory.
NM_VALIDATED = False
DWI_METRIC_SUFFIXES = ("_fa", "_md", "_ad", "_rd", "_fw", "_fat", "_mdt", "_mk")   # dwi.METRICS as column suffixes


def is_dwi_feature(column):
    """ROI metrics use metric-last names; the optional tract pipeline uses side-last names."""
    c = column.removeprefix("dwi_")
    return (not c.startswith("n_") and c.endswith(DWI_METRIC_SUFFIXES)) or bool(
        re.fullmatch(r"nst_(?:afd|seed_success|fa|md)_[lr]", c))


def _masked_dates(s):
    """LONI masks some acquisition dates as 9999-...: a date, but not a day anything can be measured from."""
    s = pd.to_datetime(s, errors="coerce")
    return s.where(s.dt.year < 2100)


DEFAULT_DIRS = {"dat": "datscan_full"}     # modality -> directory under the derived root when it is not the modality name


def _modality_dir(derived, modality, modality_dirs):
    return Path((modality_dirs or {}).get(modality, derived / DEFAULT_DIRS.get(modality, modality)))


def _baseline_idps(derived):
    idps = pd.read_csv(derived / "fastsurfer_idps.csv", parse_dates=["SCAN_DATE"], low_memory=False)
    idps["_usable"] = QC["t1"](idps)
    # Prefer the earliest usable session, as build_dataset does; retain a failed subject only for QC reporting.
    return idps.sort_values(["PATNO", "_usable", "SCAN_DATE", "IMAGEID"], ascending=[True, False, True, True])\
        .drop_duplicates("PATNO", keep="first").drop(columns="_usable")


def _modality_metadata(d, mod, index_file, subjects, flag):
    dates = _dates_from_index(index_file, subjects, flag)
    inferred = pd.to_datetime(d["patno"].map(dates), errors="coerce")
    recorded = pd.to_datetime(d.get("acquisition_date", pd.Series(index=d.index, dtype=object)), errors="coerce")
    d[f"{mod}_date"] = _masked_dates(recorded.fillna(inferred))
    d[f"{mod}_date_source"] = np.where(recorded.notna(), "recorded", "index_inferred")
    cols = [f"{mod}_date_source"]
    for c in ("fs_image_id", "processing_version", "source_image_ids", "fw_method", "topup", "denoised", "bvecs_rotated", "eddy"):
        if c in d:
            name = c if c == "fw_method" else f"{mod}_{c}"
            d[name] = d[c]
            cols.append(name)
    return cols


def _read(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    d = pd.read_csv(p, low_memory=False)
    if "error" in d:
        d = d[d["error"].fillna("") == ""]
    return (d.drop_duplicates("patno") if "patno" in d else d).copy()


def _vendor(s):
    return s.astype(str).str.upper().str.extract(r"(SIEMENS|GE|PHILIPS|PICKER|MARCONI|ADAC)")[0].fillna("OTHER")


def _dates_from_index(index_csv, subjects, flag_col=None):
    """patno -> session date used (the date with most files among selected series)."""
    p = Path(index_csv)
    if not p.exists():
        return {}
    idx = pd.read_csv(p, dtype={"image_id": str})
    if flag_col and flag_col in idx:
        idx = idx[idx[flag_col].astype(bool)]
    out = {}
    for patno, g in idx[idx["patno"].isin(subjects)].groupby("patno"):
        out[int(patno)] = g.groupby("date")["n_files"].sum().idxmax()
    return out


def build_manifest(derived_dir, modality_dirs=None):
    derived = Path(derived_dir)
    idps = _baseline_idps(derived)
    man = pd.DataFrame({"PATNO": idps["PATNO"].astype(int), "t1_image_id": idps["IMAGEID"].astype(str), "t1_date": _masked_dates(idps["SCAN_DATE"])})
    subjects = set(man["PATNO"])
    # DaTscan (PIE SBRs): date from the SPECT index member path
    dat = _read(_modality_dir(derived, "dat", modality_dirs) / "datscan_sbr.csv")
    if dat is not None:
        dat["dat_qc_pass"] = QC["dat"](dat)
        spect_idx = derived / "spect_index.csv"
        dates = {}
        if spect_idx.exists():
            si = pd.read_csv(spect_idx, dtype={"image_id": str})
            si["date"] = si["member"].str.split("/").str[3].str[:10]
            dates = si.drop_duplicates("image_id").set_index("image_id")["date"].to_dict()
        dat["dat_date"] = _masked_dates(dat["image_id"].astype(str).map(dates))
        dat["dat_batch"] = _vendor(dat["hdr_manufacturer"]) + "_" + dat["hdr_model"].astype(str).str[:12]
        extra = ["fs_image_id"] if "fs_image_id" in dat else []     # the T1 the stored SPECT transform refers to
        man = man.merge(dat[["patno", "image_id", "dat_date", "dat_batch", "dat_qc_pass"] + extra]
                        .rename(columns={"patno": "PATNO", "image_id": "dat_image_id", "fs_image_id": "dat_fs_image_id"}), on="PATNO", how="left")
    # DWI
    dwi_dir = _modality_dir(derived, "dwi", modality_dirs)
    dwi = _read(dwi_dir / "dwi_features.csv")
    if dwi is not None:
        dwi["dwi_qc_pass"] = QC["dwi"](dwi)
        dwi["dwi_batch"] = _vendor(dwi["manufacturer"]) + "_" + dwi["shells"].astype(str).str.replace(" ", "-") + "_" + dwi["fw_method"].astype(str)
        dwi["dwi_batch"] += np.where(dwi.get("eddy", pd.Series(False, index=dwi.index)).astype(str).str.lower().eq("true"), "_eddy", "")
        flag = lambda c: dwi.get(c, pd.Series(False, index=dwi.index)).astype(str).str.lower().eq("true")
        dwi["dwi_correction"] = np.where(flag("eddy"), "eddy", np.where(flag("topup"), "topup", "rigid"))
        # the study's batch: vendor | shells | voxel size | number of volumes (direction count)
        dwi["dwi_acquisition_batch"] = _vendor(dwi["manufacturer"]) + "|" + dwi["shells"].astype(str) + "|" + \
            dwi.get("voxel_mm", pd.Series("", index=dwi.index)).astype(str) + "|" + dwi.get("n_volumes", pd.Series("", index=dwi.index)).astype(str)
        extra = _modality_metadata(dwi, "dwi", dwi_dir / "dwi_index.csv", subjects, "selected")
        man = man.merge(dwi[["patno", "dwi_date", "dwi_batch", "dwi_acquisition_batch", "dwi_correction", "dwi_qc_pass"] + extra].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
    # NM
    nm_dir = _modality_dir(derived, "nm", modality_dirs)
    nmf = _read(nm_dir / "nm_features.csv")
    if nmf is not None:
        nmf["nm_qc_pass"] = QC["nm"](nmf)
        nmf["nm_batch"] = _vendor(nmf["manufacturer"]) + "_" + nmf["voxel_mm"].astype(str)
        # the study's batch: vendor | voxel | TR | TE | MT preparation (the NM contrast depends on all of them)
        num = lambda c: pd.to_numeric(nmf.get(c, pd.Series(np.nan, index=nmf.index)), errors="coerce").round(4).astype(str)
        nmf["nm_acquisition_batch"] = (_vendor(nmf["manufacturer"]) + "|" + nmf["voxel_mm"].astype(str) + "|" + num("tr_s") + "|"
                                       + num("te_s") + "|" + num("mt_flag"))
        extra = _modality_metadata(nmf, "nm", nm_dir / "nm_index.csv", subjects, "selected")
        man = man.merge(nmf[["patno", "nm_date", "nm_batch", "nm_acquisition_batch", "nm_qc_pass"] + extra].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
    # FLAIR
    flair_dir = _modality_dir(derived, "flair", modality_dirs)
    fl = _read(flair_dir / "flair_features.csv")
    if fl is not None:
        fl["flair_qc_pass"] = QC["flair"](fl)
        fl["flair_batch"] = _vendor(fl["manufacturer"]) + "_" + np.where(fl["flair_3d"].astype(bool), "3D", "2D")
        extra = _modality_metadata(fl, "flair", flair_dir / "flair_index.csv", subjects, "selected")
        man = man.merge(fl[["patno", "flair_date", "flair_batch", "flair_qc_pass"] + extra].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
    for mod in ("dat", "dwi", "nm", "flair"):
        if f"{mod}_date" in man:
            man[f"{mod}_days_from_t1"] = (man[f"{mod}_date"] - man["t1_date"]).dt.days
        if f"{mod}_fs_image_id" in man:
            recorded = man[f"{mod}_fs_image_id"].notna()
            mismatch = recorded & man[f"{mod}_fs_image_id"].astype(str).ne(man["t1_image_id"])
            man[f"{mod}_t1_mismatch"] = mismatch
            man.loc[mismatch, f"{mod}_qc_pass"] = False
    return man


def assemble_features(derived_dir, modality_dirs=None, single_shell_fw=False, ppmi_dir=None):
    """Manifest + features per subject: FastSurfer IDPs (as in fastsurfer_idps.csv), `dat_*` SBRs (occipital `sbr_*` and
    cerebral-WM `sbrwm_*` references, their anterior/posterior halves, putamen/caudate ratios and asymmetry indices),
    `dwi_*`, `nm_*` (ratios/volumes only), `flair_*`. Values of QC-failed modalities are blanked; the QC flags stay.

    Single-shell free water (``fw_method == "singleshell_prior"``) and its tissue FA are blanked unless
    ``single_shell_fw``: on 762 PPMI-1 scans its brain median did not rise with age (r = +0.02, against +0.43 for
    white-matter MD and +0.49 for multi-shell free water), it tracked nigral MD at r = 0.12 and it did not separate
    PD from controls. The estimator is ill-posed without a second shell (Golub et al. 2021, MRM, doi:10.1002/mrm.28599)."""
    from .labels import sbr_indices

    derived = Path(derived_dir)
    man = build_manifest(derived, modality_dirs=modality_dirs)
    idps = _baseline_idps(derived).drop(columns=["SCAN_DATE"])
    idps["t1_qc_pass"] = QC["t1"](idps)   # failed segmentation: T1 measures blanked, flag kept (as for the other modalities)
    idps.loc[~idps["t1_qc_pass"], [c for c in idps.columns if c not in ("PATNO", "IMAGEID", "t1_qc_pass")]] = np.nan
    df = man.merge(idps, on="PATNO", how="left")
    if ppmi_dir is not None and "EVENT_ID" in df:   # PPMI's FS7 / MRIQC values, only for the visit of the T1 PIE used
        from .features import fs7_tables
        fs7 = fs7_tables(ppmi_dir).rename(columns={"EVENT_ID": "_fs7_event"})
        df = df.merge(fs7, left_on=["PATNO", "EVENT_ID"], right_on=["PATNO", "_fs7_event"], how="left").drop(columns="_fs7_event")
    blocks = {}
    dat = _read(_modality_dir(derived, "dat", modality_dirs) / "datscan_sbr.csv")
    if dat is not None:
        for ref in ("sbr", "sbrwm"):
            if all(f"{ref}_{r}_{s}" in dat for r in ("caudate", "putamen") for s in "lr"):
                idx = sbr_indices(*(dat[f"{ref}_{r}_{s}"] for r in ("caudate", "putamen") for s in "lr"))
                dat[[f"{ref}_{c}" for c in idx]] = idx.to_numpy()
        cols = [c for c in dat.columns if c.startswith(("sbr_", "sbrwm_"))]
        blocks["dat"] = dat[["patno"] + cols].rename(columns={c: f"dat_{c}" for c in cols})
    dwi = _read(_modality_dir(derived, "dwi", modality_dirs) / "dwi_features.csv")
    if dwi is not None:
        cols = [c for c in dwi.columns if is_dwi_feature(c)]
        if not single_shell_fw and "fw_method" in dwi:
            dwi.loc[dwi["fw_method"].eq("singleshell_prior"), [c for c in cols if c.endswith(("_fw", "_fat"))]] = np.nan
        blocks["dwi"] = dwi[["patno"] + cols].rename(columns={c: f"dwi_{c}" for c in cols})
    nmf = _read(_modality_dir(derived, "nm", modality_dirs) / "nm_features.csv")
    if nmf is not None:
        cols = [c for c in nmf.columns if c.startswith("nm_") and c.endswith(("_cnr", "_voxels"))]
        blocks["nm"] = nmf[["patno"] + cols]
        df["nm_validated"] = NM_VALIDATED
    nmt = _read(_modality_dir(derived, "nm", modality_dirs) / "nm_template_features.csv")
    if nmt is not None:      # template-space CNR (Cassidy/Wengler); its own QC, so a failed template row blanks only these
        nmt["nmt_qc_pass"] = QC["nmt"](nmt)
        cols = [c for c in nmt.columns if c.startswith("nmt_") and c.endswith("_cnr")]
        nmt.loc[~nmt["nmt_qc_pass"], cols] = np.nan
        df = df.merge(nmt[["patno", "nmt_qc_pass"] + cols].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
    nmb = _read(_modality_dir(derived, "nm", modality_dirs) / "nm_published_features.csv")
    if nmb is not None:      # published Biondetti territories vs background; coverage already blanks a region
        cols = [c for c in nmb.columns if c.startswith("nmb_") and c.endswith("_cnr")]
        df = df.merge(nmb[["patno"] + cols].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
    fl = _read(_modality_dir(derived, "flair", modality_dirs) / "flair_features.csv")
    if fl is not None:
        cols = ["wmh_log_mm3", "wmh_pv_mm3", "wmh_deep_mm3", "wmh_frac_wm", "wmh_n_lesions", "wmh_mm3"]
        blocks["flair"] = fl[["patno"] + cols].rename(columns={c: f"flair_{c}" for c in cols})
    for mod, b in blocks.items():
        df = df.merge(b.rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
        feat_cols = [c for c in b.columns if c != "patno"]
        if f"{mod}_qc_pass" in df:
            # A failed left-side metric must not let valid-looking right-side metrics escape QC.
            bad = ~df[f"{mod}_qc_pass"].fillna(False).astype(bool)
            df.loc[bad, feat_cols] = np.nan
    if "nm_qc_pass" in df:   # the slab's own QC (motion, coverage) holds for every NM method read from it
        df.loc[~df["nm_qc_pass"].fillna(False).astype(bool), [c for c in df if c.startswith(("nmt_", "nmb_")) and c.endswith("_cnr")]] = np.nan
    # one intracranial volume for head-size adjustment: PIE's FreeSurfer eTIV (same T1 as the volumes), else PPMI's
    # FreeSurfer 7 eTIV of the same visit; never MaskVol or BrainSegVol
    nan = pd.Series(np.nan, index=df.index)
    pie_tiv, fs7_tiv = df.get("eTIV", nan), df.get("fs7_EstimatedTotalIntraCranialVol", nan)
    df["tiv_mm3"] = pie_tiv.fillna(fs7_tiv)
    df["tiv_source"] = np.where(pie_tiv.notna(), "pie_talairach", np.where(fs7_tiv.notna(), "ppmi_fs7", "none"))
    return df


def feature_blocks(columns):
    """Modality block -> feature columns, for block-wise harmonisation / stacking."""
    cols = list(columns)
    return {"dat": [c for c in cols if c.startswith("dat_sbr")],
            "dwi": [c for c in cols if c.startswith("dwi_") and is_dwi_feature(c)],
            "nm": [c for c in cols if c.startswith(("nm_", "nmt_", "nmb_")) and c.endswith(("_cnr", "_voxels"))],
            "flair": [c for c in cols if c.startswith("flair_wmh")]}
