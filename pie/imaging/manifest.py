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


def _nm_slab_coverage_ok(d):
    """Each side's atlas SN >= 90 % inside the slab (``nm.slab_coverage``, the study's bounded geometric rule, 27 Sep
    2026). Rows written before then carry only the unbounded volume ratio ``sn_slab_coverage`` and keep its old bar."""
    side = np.minimum(_col(d, "sn_slab_coverage_l"), _col(d, "sn_slab_coverage_r"))
    return (side >= 0.9).where(side.notna(), _col(d, "sn_slab_coverage") >= 0.5)


# QC rules per modality (the same thresholds the study used); keep in one place
T1_KEY_VOLUMES = ["vol_Left_Putamen", "vol_Right_Putamen", "vol_Left_Caudate", "vol_Right_Caudate", "vol_Left_Thalamus", "vol_Right_Thalamus"]
QC = {
    # a failed FastSurfer run leaves empty labels (0 mm3); FastSurfer's own check fails a segmented brain under 0.7 L; a
    # visual "fail" (t1_visual_qc.csv, see t1_visual_ratings) fails. Robust-z outliers are flagged for review, not failed.
    "t1": lambda d: ((d[[c for c in T1_KEY_VOLUMES if c in d]] > 0).all(axis=1) & ~(_col(d, "BrainSegVol") < 7e5)
                     & ~d.get("t1_visual_qc", pd.Series("", index=d.index)).astype(str).str.lower().eq("fail")),
    # correlation metric: -1 is a perfect fit; a failed registration (inf) or an anti-correlated one must not pass
    "dat": lambda d: (-d["reg_metric"] >= 0.4) & (d["n_label_voxels"] >= 100),
    # the study's nigral diffusion rule (split-half ICC FA 0.88, MD 0.97 under it): >= 20 SN voxels, >= 95 % of the SN in
    # the brain mask, >= 80 % physically admissible tensor fits, mean motion <= 2 mm, both posterior SN halves measured;
    # tables written before these columns existed fail it (re-run them)
    "dwi": lambda d: ((d["n_sn_l"] + d["n_sn_r"] >= 20) & (_col(d, "sn_brain_mask_fraction") >= 0.95)
                      & (_col(d, "sn_physical_fraction") >= 0.8) & (_col(d, "motion_mm_mean") <= 2)
                      & np.isfinite(_col(d, "sn_posterior_l_fa")) & np.isfinite(_col(d, "sn_posterior_r_fa")) & (d["fa_wm_median"] > 0.25)),
    "nm": lambda d: (d["n_sn_l"] >= 20) & (d["n_sn_r"] >= 20) & _nm_slab_coverage_ok(d) & (d["repeat_motion_mm_max"] < 3)
    & (d["nm_ref_l_sd"] < 0.4 * d["nm_ref_l_mean"]) & (d["nm_ref_r_sd"] < 0.4 * d["nm_ref_r_mean"])   # reference ring partly outside the slab -> CV ~1, CNR garbage
    & ~(_col(d, "nm_clipped_fraction") > 0.01)   # scanner-clipped repeats (tables before 26 Sep 2026 lack the column: pass)
    & ~(_col(d, "n_repeats") < 3)   # the study's >= 3 averaged repeats (a single-repeat slab has ~2x the noise; tables without the column: pass)
    # the study's voxel checks on the measured masks: >= 95 % with signal and reached by every repeat (tables before
    # 27 Sep 2026 lack the columns: pass)
    & ~d.reindex(columns=[f"nm_{k}_{s}" for k in ("valid", "support") for s in ("l", "r", "ref")]).lt(0.95).any(axis=1),
    # S-I coverage under 130 mm leaves WM outside the slab (27 of 486 PPMI 2D FLAIRs; tables before Oct 2026 lack the column: pass)
    "flair": lambda d: (d["reg_flair_t1_mi"] < -0.2) & (d["wm_mm3"] > 200000) & (d["flair_wm_mad"] > 0) & ~(_col(d, "flair_si_mm") < 130),
    # nm_template: the template SN mask has data on both sides and the crus reference is homogeneous
    "nmt": lambda d: (d["nmt_sn_cov_l"] >= 0.9) & (d["nmt_sn_cov_r"] >= 0.9) & (d["nmt_crus_cv_l"] < 0.3) & (d["nmt_crus_cv_r"] < 0.3),
}
# No NM measure has passed a pre-registered PD-vs-HC gate (AUROC >= 0.70, CI lower bound > 0.55; 26 September 2026):
# 45 HC + 45 PD: nm_template 0.58 [0.46, 0.70], nm.py 0.54 [0.42, 0.67]. Re-test with the published Biondetti
# territories on 106 held-out PD + the 39 usable HC: sensorimotor contrast 0.62 [0.53, 0.72], rho with putamen SBR
# 0.18; 8-territory model 0.67 [97.5 % CI 0.55, 0.78], rho 0.20 (both fail). Assembled NM columns are exploratory.
# The default NM measure (11 Oct 2026) is snceg (``nm_primary_cr``, ``nm_primary_volume_mm3``): the only NM measure that
# reproduced a published PPMI result (Lillebostad 2025 TSE-FS: AUROC 0.710 [0.611, 0.799] vs 0.743). It is not validated
# on PPMI's 2D GRE-MT; a Siemens-only re-test is pre-registered (documentation/preregistration_nm_snceg.md).
NM_VALIDATED = False
# The study's posterior-nigral MD (``dwis_md_sn_posterior_mean``, the DWI primary) failed its pre-registered
# PPMI 2.0 PD-vs-HC gate (AUROC >= 0.60, lower bound > 0.50; 29 Sep 2026): 0.574 [0.495, 0.652], 218 PD / 82 HC, rho with
# the lowest putamen SBR 0.03. Close to the meta-analytic nigral MD effect (Schwarz 2013, d 0.26), so likely a real small
# effect on an endpoint the literature never supported. Assembled nigral DWI columns are exploratory.
DWI_NIGRAL_VALIDATED = False
# The pre-registered replacement endpoint (user decision, 11 Oct 2026): multi-shell posterior-SN free water
# (``dwisf_fw_sn_posterior_mean``; documentation/preregistration_dwi_fw.md), untested, so the flag above covers it too.
STUDY_DWI = {"dwis": r"dwis_(?:fa|md|ad|rd)_\w+", "dwisf": r"dwisf_(?:fw|fat|mdt)_\w+"}
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


DEFAULT_DIRS = {"dat": "datscan_full", "pet_av133": "pet"}     # modality -> directory under the derived root when it is not the modality name


def _modality_dir(derived, modality, modality_dirs):
    return Path((modality_dirs or {}).get(modality, derived / DEFAULT_DIRS.get(modality, modality)))


T1_QC_VOLUMES = T1_KEY_VOLUMES + ["vol_Left_Hippocampus", "vol_Right_Hippocampus", "vol_Left_Lateral_Ventricle",
                                   "vol_Right_Lateral_Ventricle", "BrainSegVol"]


def t1_outliers(idps, z_max=4.0, min_batch=20):
    """Robust z of log key volumes (``T1_QC_VOLUMES``; median and 1.4826 x MAD) within scanner batch (vendor x field
    strength; a batch under ``min_batch`` scans is scored against all scans): ``t1_robust_z_max`` and ``t1_outlier``
    (any |z| > ``z_max``), the scans to review first. Not a QC failure: atrophy is an outlier too."""
    cols = [c for c in T1_QC_VOLUMES if c in idps]
    v = np.log(idps[cols].apply(pd.to_numeric, errors="coerce").where(lambda x: x > 0))
    batch = _vendor(idps.get("Manufacturer", pd.Series("", index=idps.index))) + "_" + \
        idps.get("MagneticFieldStrength", pd.Series("", index=idps.index)).astype(str)
    batch = batch.where(batch.map(batch.value_counts()) >= min_batch, "all")
    z = pd.DataFrame(index=idps.index, columns=cols, dtype=float)
    for b in batch.unique():
        ref = v if b == "all" else v[batch == b]
        mad = 1.4826 * (ref - ref.median()).abs().median()
        z.loc[batch == b] = ((v[batch == b] - ref.median()) / mad.replace(0, np.nan)).to_numpy()
    zmax = z.abs().max(axis=1)
    return pd.DataFrame({"t1_robust_z_max": zmax, "t1_outlier": zmax > z_max}, index=idps.index)


def t1_visual_ratings(derived):
    """``<derived>/t1_visual_qc.csv`` (IMAGEID, rating: pass | borderline | fail, optional rater, date) as IMAGEID ->
    rating; empty when absent. A "fail" fails ``QC["t1"]``; review the ``t1_outlier`` scans first (``qc --modality t1``)."""
    p = Path(derived) / "t1_visual_qc.csv"
    if not p.exists():
        return {}
    r = pd.read_csv(p, dtype={"IMAGEID": str})
    return r.drop_duplicates("IMAGEID", keep="last").set_index("IMAGEID")["rating"].astype(str).str.lower().to_dict()


def _baseline_idps(derived):
    idps = pd.read_csv(derived / "fastsurfer_idps.csv", parse_dates=["SCAN_DATE"], low_memory=False)
    idps = idps.join(t1_outliers(idps))
    ratings = t1_visual_ratings(derived)
    if ratings:
        idps["t1_visual_qc"] = idps["IMAGEID"].astype(str).map(ratings)
    idps["_usable"] = QC["t1"](idps)
    # Prefer the earliest usable session, as build_dataset does; retain a failed subject only for QC reporting.
    return idps.sort_values(["PATNO", "_usable", "SCAN_DATE", "IMAGEID"], ascending=[True, False, True, True])\
        .drop_duplicates("PATNO", keep="first").drop(columns="_usable")


T1_MEASURE = lambda c: c.startswith(("vol_", "sum_", "asym_", "cth_", "area_")) or c.endswith("Vol") or c == "eTIV"


def t1_visits(derived_dir, stream="cross"):
    """Every T1 visit (one per participant and visit day), not only the baseline: the longitudinal table.

    ``stream``: "cross" reads ``fastsurfer_idps.csv``; "long" reads ``fastsurfer_long_idps.csv`` (FastSurfer's
    longitudinal stream, ``run --long``; within-subject template, so change is less noisy). QC as ``QC["t1"]``
    (measures blanked, identifiers kept), outlier flags, and:
    ``years_from_bl`` (from the participant's earliest real-dated usable visit), ``visit_index``,
    ``scanner_id`` (DeviceSerialNumber, else StationName, else vendor + model) with ``scanner_changed_since_bl`` and
    ``field_changed_since_bl`` (scanner changes are batch effects: harmonise with ``harmonize.long_combat`` or
    restrict to same-scanner follow-up), ``tiv_subject_mm3`` (the median eTIV over the participant's usable visits:
    head size does not change, so one TIV per participant, as FreeSurfer's longitudinal stream does)."""
    derived = Path(derived_dir)
    name = {"cross": "fastsurfer_idps.csv", "long": "fastsurfer_long_idps.csv"}[stream]
    v = pd.read_csv(derived / name, parse_dates=["SCAN_DATE"], low_memory=False, dtype={"DeviceSerialNumber": str, "StationName": str})
    v = v.join(t1_outliers(v))
    ratings = t1_visual_ratings(derived)
    if ratings:
        v["t1_visual_qc"] = v["IMAGEID"].astype(str).map(ratings)
    v["t1_qc_pass"] = QC["t1"](v)
    v.loc[~v["t1_qc_pass"], [c for c in v.columns if T1_MEASURE(c)]] = np.nan
    v["SCAN_DATE"] = _masked_dates(v["SCAN_DATE"])
    v = v.sort_values(["PATNO", "SCAN_DATE"]).reset_index(drop=True)
    pick = lambda c: v.get(c, pd.Series(np.nan, index=v.index)).replace("", np.nan)
    v["scanner_id"] = pick("DeviceSerialNumber").fillna(pick("StationName")).fillna(
        _vendor(pick("Manufacturer").fillna("")) + "_" + pick("ManufacturersModelName").fillna("").astype(str)).astype(str)
    usable = v[v["t1_qc_pass"] & v["SCAN_DATE"].notna()]
    first = usable.groupby("PATNO").first()
    v["years_from_bl"] = (v["SCAN_DATE"] - v["PATNO"].map(first["SCAN_DATE"])).dt.days / 365.25
    v["visit_index"] = v.groupby("PATNO").cumcount()
    v["scanner_changed_since_bl"] = (v["scanner_id"] != v["PATNO"].map(first["scanner_id"])).where(v["PATNO"].isin(first.index))
    if "MagneticFieldStrength" in v:
        v["field_changed_since_bl"] = (v["MagneticFieldStrength"] != v["PATNO"].map(first["MagneticFieldStrength"])).where(v["PATNO"].isin(first.index))
    if "eTIV" in v:
        v["tiv_subject_mm3"] = v["PATNO"].map(v.groupby("PATNO")["eTIV"].median())
    return v


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
        fl["flair_wmh_method"] = fl.get("wmh_method", pd.Series("threshold", index=fl.index)).fillna("threshold")
        extra = _modality_metadata(fl, "flair", flair_dir / "flair_index.csv", subjects, "selected")
        man = man.merge(fl[["patno", "flair_date", "flair_batch", "flair_wmh_method", "flair_qc_pass"] + extra].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
    # AV-133 PET (pet.py): one row per visit, so the visit nearest the T1; the stored registration refers to fs_image_id
    pet = pet_av133_rows(derived, modality_dirs, man)
    if pet is not None:
        man = man.merge(pet[["PATNO", "pet_av133_image_id", "pet_av133_date", "pet_av133_batch", "pet_av133_fs_image_id",
                             "pet_av133_qc_pass"]], on="PATNO", how="left")
    for mod in ("dat", "dwi", "nm", "flair", "pet_av133"):
        if f"{mod}_date" in man:
            man[f"{mod}_days_from_t1"] = (man[f"{mod}_date"] - man["t1_date"]).dt.days
        if f"{mod}_fs_image_id" in man:
            recorded = man[f"{mod}_fs_image_id"].notna()
            mismatch = recorded & man[f"{mod}_fs_image_id"].astype(str).ne(man["t1_image_id"])
            man[f"{mod}_t1_mismatch"] = mismatch
            man.loc[mismatch, f"{mod}_qc_pass"] = False
    return man


# AV-133 PET (pet.py) is unvalidated: agreement with PPMI's PET_SBR_Analysis (`pet --agreement`) has not been run.
PET_AV133_VALIDATED = False


def pet_av133_rows(derived, modality_dirs, man):
    """``pet/pet_av133.csv`` per participant: the visit nearest the manifest's T1 (dated visits first), with
    ``pet_av133_*`` names, the QC of ``pet.qc_pass`` (occipital; cerebellum-referenced values under their own rule) and a
    failed QC when the stored registration refers to another T1. None when the table is absent."""
    from .pet import qc_pass

    p = _modality_dir(Path(derived), "pet_av133", modality_dirs) / "pet_av133.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    d = pd.read_csv(p, low_memory=False, dtype={"image_id": str, "fs_image_id": str})
    d = d[d.get("error", pd.Series("", index=d.index)).fillna("") == ""].copy()
    d["_qc"], d["_qc_cb"] = qc_pass(d), qc_pass(d, "cerebellum")
    d = d.merge(man[["PATNO", "t1_date", "t1_image_id"]], left_on="patno", right_on="PATNO")
    d["pet_av133_date"] = _masked_dates(d["acquisition_date"])
    d["_gap"] = (d["pet_av133_date"] - d["t1_date"]).dt.days.abs().fillna(np.inf)
    d = d.sort_values(["PATNO", "_gap"]).drop_duplicates("PATNO")
    sbr = [c for c in d.columns if re.fullmatch(r"sbr(?:cb)?_\w+", c)] + (["cb_occ"] if "cb_occ" in d else [])
    d.loc[~d["_qc"], sbr] = np.nan
    d.loc[~d["_qc_cb"], [c for c in sbr if c.startswith("sbrcb_")]] = np.nan
    text = lambda c: d.get(c, pd.Series("", index=d.index)).fillna("").astype(str)
    d["pet_av133_qc_pass"] = d["_qc"] & ~(text("fs_image_id").ne("") & text("fs_image_id").ne(d["t1_image_id"].astype(str)))
    d.loc[~d["pet_av133_qc_pass"], sbr] = np.nan
    d["pet_av133_batch"] = _vendor(text("manufacturer")) + "_" + text("model").str[:12] + "_" + text("postfilter")
    out = d.rename(columns={"image_id": "pet_av133_image_id", "fs_image_id": "pet_av133_fs_image_id", **{c: f"pet_av133_{c}" for c in sbr}})
    keep = ["PATNO", "pet_av133_image_id", "pet_av133_date", "pet_av133_batch", "pet_av133_fs_image_id", "pet_av133_qc_pass"]
    return out.reindex(columns=keep + [f"pet_av133_{c}" for c in sbr])


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
    measures = [c for c in idps.columns if T1_MEASURE(c)]
    idps.loc[~idps["t1_qc_pass"], measures] = np.nan     # identifiers, visit and scanner stay
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
        # the study's nigral measures (``dwi --study``) carry their own QC, as nm_template does: the tensor measure
        # ``dwis_*`` and the pre-registered multi-shell free-water endpoint ``dwisf_*``
        for pre in STUDY_DWI:
            if f"{pre}_qc_pass" in dwi:
                scols = [c for c in dwi.columns if re.fullmatch(STUDY_DWI[pre], c)]
                ok = dwi[f"{pre}_qc_pass"].astype(str).str.lower().eq("true")
                dwi.loc[~ok, scols] = np.nan
                df = df.merge(dwi[["patno", f"{pre}_qc_pass"] + scols].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
        df["dwi_nigral_validated"] = DWI_NIGRAL_VALIDATED
    nmf = _read(_modality_dir(derived, "nm", modality_dirs) / "nm_features.csv")
    if nmf is not None:
        cols = [c for c in nmf.columns if c.startswith("nm_") and c.endswith(("_cnr", "_voxels"))]   # nm.py ring contrast: diagnostic
        blocks["nm"] = nmf[["patno"] + cols]
        df["nm_validated"] = NM_VALIDATED
        # vendor and slab thickness (thin-axis extent) so analyses can restrict to Siemens / full coverage: 41 % of PIE's
        # NM is GE, most of it 20 mm slabs against the protocol's 24 mm, and GE separated PD from HC far worse
        dims = lambda c: nmf.get(c, pd.Series("", index=nmf.index)).astype(str).str.split("x", expand=True).apply(pd.to_numeric, errors="coerce")
        df = df.merge(pd.DataFrame({"PATNO": nmf["patno"], "nm_vendor": _vendor(nmf["manufacturer"]),
                                    "nm_slab_mm": (dims("shape") * dims("voxel_mm")).min(axis=1)}), on="PATNO", how="left")
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
    nat = _read(_modality_dir(derived, "nm", modality_dirs) / "nm_native_features.csv")
    if nat is not None:      # native-space published measures (nm_native): Langley SNc volume, snceg volume and contrast
        cols = [c for c in nat.columns if c.startswith(("nml_sn_volume", "nms_sn", "nm_bion_")) and c.endswith(("_mm3", "_cr", "_cnr"))]
        df = df.merge(nat[["patno"] + cols].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
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
        df.loc[~df["nm_qc_pass"].fillna(False).astype(bool),
               [c for c in df if c.startswith(("nmt_", "nmb_", "nml_", "nms_", "nm_bion_")) and c.endswith(("_cnr", "_cr", "_mm3"))]] = np.nan
    if "nms_sn_mean_cr" in df:   # PIE's default NM measure (11 Oct 2026): snceg contrast vs the crus on the SN's slices, and volume
        df["nm_primary_cr"], df["nm_primary_volume_mm3"] = df["nms_sn_mean_cr"], df["nms_sn_volume_mm3"]
    # one intracranial volume for head-size adjustment: PIE's FreeSurfer eTIV (same T1 as the volumes), else PPMI's
    # FreeSurfer 7 eTIV of the same visit; never MaskVol or BrainSegVol
    nan = pd.Series(np.nan, index=df.index)
    pie_tiv, fs7_tiv = df.get("eTIV", nan), df.get("fs7_EstimatedTotalIntraCranialVol", nan)
    df["tiv_mm3"] = pie_tiv.fillna(fs7_tiv)
    df["tiv_source"] = np.where(pie_tiv.notna(), "pie_talairach", np.where(fs7_tiv.notna(), "ppmi_fs7", "none"))
    if ppmi_dir is not None and "dat_date" in df:
        df = df.merge(core_lab_sbr(ppmi_dir, df[["PATNO", "dat_date"]]), on=["PATNO", "dat_date"], how="left")
    pet = pet_av133_rows(derived, modality_dirs, man)
    if pet is not None:
        df = df.merge(pet[["PATNO"] + [c for c in pet if c.startswith(("pet_av133_sbr", "pet_av133_cb_occ"))]], on="PATNO", how="left")
        df["pet_av133_validated"] = PET_AV133_VALIDATED
    return df


def core_lab_sbr(ppmi_dir, scans):
    """PPMI's core-lab SBRs for the DaTscans PIE reconstructed (``scans``: PATNO, dat_date), matched on participant and
    scan month: ``dat_core_{caudate,putamen}_{l,r}``, their indices and ``dat_core_source``. Xing (cerebral-WM
    reference) where it analysed the scan, else archived Invicro (occipital). These are PPMI's endpoint values, so
    they take no PIE DaT QC; PIE's own ``dat_sbr_*`` stay a screening reconstruction for cohorts without a core lab.
    The two sources' scales differ: stratify or harmonise on ``dat_core_source``, never pool raw values."""
    from .labels import dat_sbr_table, sbr_indices

    tables = []
    for source in ("xing", "invicro"):
        try:
            tables.append(dat_sbr_table(ppmi_dir, source)[0])
        except FileNotFoundError:
            pass
    cols = {f"DATSCAN_{r.upper()}_{s.upper()}": f"dat_core_{r}_{s}" for r in ("caudate", "putamen") for s in "lr"}
    out = scans.dropna().drop_duplicates().assign(_month=lambda x: x["dat_date"].dt.to_period("M").dt.to_timestamp())
    if not tables:
        return out.drop(columns="_month")
    core = pd.concat(tables, ignore_index=True).drop_duplicates(["PATNO", "DATSCAN_DATE"], keep="first")   # Xing first
    core = core[["PATNO", "DATSCAN_DATE", "sbr_source", *cols]].rename(columns={**cols, "DATSCAN_DATE": "_month", "sbr_source": "dat_core_source"})
    out = out.merge(core, on=["PATNO", "_month"], how="left").drop(columns="_month")
    idx = sbr_indices(*(out[f"dat_core_{r}_{s}"] for r in ("caudate", "putamen") for s in "lr"))
    out[[f"dat_core_{c}" for c in idx]] = idx.to_numpy()
    return out


def feature_blocks(columns):
    """Modality block -> feature columns, for block-wise harmonisation / stacking."""
    cols = list(columns)
    return {"dat": [c for c in cols if c.startswith("dat_sbr")],
            "dat_core": [c for c in cols if c.startswith("dat_core_") and c != "dat_core_source"],
            "dwi": [c for c in cols if (c.startswith("dwi_") and is_dwi_feature(c)) or any(re.fullmatch(r, c) for r in STUDY_DWI.values())],
            "nm": [c for c in cols if c.startswith(("nm_", "nmt_", "nmb_", "nml_", "nms_")) and c.endswith(("_cnr", "_voxels", "_cr", "_mm3"))
                   and not c.startswith("nm_primary_")],   # aliases of nms_* columns
            "flair": [c for c in cols if c.startswith("flair_wmh") and c != "flair_wmh_method"],
            "pet_av133": [c for c in cols if c.startswith("pet_av133_sbr")]}
