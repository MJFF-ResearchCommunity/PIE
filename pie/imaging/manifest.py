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

import numpy as np
import pandas as pd

# QC rules per modality (the same thresholds the study used); keep in one place
QC = {
    "dat": lambda d: (d["reg_metric"].abs() >= 0.4) & (d["n_label_voxels"] >= 100),
    "dwi": lambda d: (d["motion_mm_max"] < 6) & (d["n_sn_l"] >= 3) & (d["n_sn_r"] >= 3) & (d["fa_wm_median"] > 0.25),
    "nm": lambda d: (d["n_sn_l"] >= 20) & (d["n_sn_r"] >= 20) & (d["sn_slab_coverage"] >= 0.5) & (d["repeat_motion_mm_max"] < 3),
    "flair": lambda d: (d["reg_flair_t1_mi"] < -0.2) & (d["wm_mm3"] > 200000) & (d["flair_wm_mad"] > 0),
}
DAT_COLS = ["sbr_caudate_l", "sbr_caudate_r", "sbr_putamen_l", "sbr_putamen_r", "reg_metric", "hdr_manufacturer", "hdr_model", "hdr_scale_fit", "image_id"]


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


def build_manifest(derived_dir):
    derived = Path(derived_dir)
    idps = pd.read_csv(derived / "fastsurfer_idps.csv", parse_dates=["SCAN_DATE"], low_memory=False)
    idps = idps.sort_values(["PATNO", "SCAN_DATE"]).drop_duplicates("PATNO", keep="first")
    man = pd.DataFrame({"PATNO": idps["PATNO"].astype(int), "t1_image_id": idps["IMAGEID"].astype(str), "t1_date": idps["SCAN_DATE"]})
    subjects = set(man["PATNO"])
    # DaTscan (PIE SBRs): date from the SPECT index member path
    dat = _read(derived / "datscan_full" / "datscan_sbr.csv")
    if dat is not None:
        dat["dat_qc_pass"] = QC["dat"](dat)
        spect_idx = derived / "spect_index.csv"
        dates = {}
        if spect_idx.exists():
            si = pd.read_csv(spect_idx, dtype={"image_id": str})
            si["date"] = si["member"].str.split("/").str[3].str[:10]
            dates = si.drop_duplicates("image_id").set_index("image_id")["date"].to_dict()
        dat["dat_date"] = pd.to_datetime(dat["image_id"].astype(str).map(dates), errors="coerce")
        dat["dat_batch"] = _vendor(dat["hdr_manufacturer"]) + "_" + dat["hdr_model"].astype(str).str[:12]
        man = man.merge(dat[["patno", "image_id", "dat_date", "dat_batch", "dat_qc_pass"]].rename(columns={"patno": "PATNO", "image_id": "dat_image_id"}), on="PATNO", how="left")
    # DWI
    dwi = _read(derived / "dwi" / "dwi_features.csv")
    if dwi is not None:
        dwi["dwi_qc_pass"] = QC["dwi"](dwi)
        dwi["dwi_batch"] = _vendor(dwi["manufacturer"]) + "_" + dwi["shells"].astype(str).str.replace(" ", "-") + "_" + dwi["fw_method"].astype(str)
        dd = _dates_from_index(derived / "dwi" / "dwi_index.csv", subjects, "selected" if "selected" in pd.read_csv(derived / "dwi" / "dwi_index.csv", nrows=0).columns else None)
        dwi["dwi_date"] = pd.to_datetime(dwi["patno"].map(dd), errors="coerce")
        man = man.merge(dwi[["patno", "dwi_date", "dwi_batch", "dwi_qc_pass", "topup" if "topup" in dwi else "fw_method"]].rename(columns={"patno": "PATNO", "topup": "dwi_topup"}), on="PATNO", how="left")
    # NM
    nmf = _read(derived / "nm" / "nm_features.csv")
    if nmf is not None:
        nmf["nm_qc_pass"] = QC["nm"](nmf)
        nmf["nm_batch"] = _vendor(nmf["manufacturer"]) + "_" + nmf["voxel_mm"].astype(str)
        dd = _dates_from_index(derived / "nm" / "nm_index.csv", subjects, "selected" if "selected" in pd.read_csv(derived / "nm" / "nm_index.csv", nrows=0).columns else "nm")
        nmf["nm_date"] = pd.to_datetime(nmf["patno"].map(dd), errors="coerce")
        man = man.merge(nmf[["patno", "nm_date", "nm_batch", "nm_qc_pass"]].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
    # FLAIR
    fl = _read(derived / "flair" / "flair_features.csv")
    if fl is not None:
        fl["flair_qc_pass"] = QC["flair"](fl)
        fl["flair_batch"] = _vendor(fl["manufacturer"]) + "_" + np.where(fl["flair_3d"].astype(bool), "3D", "2D")
        dd = _dates_from_index(derived / "flair" / "flair_index.csv", subjects, "selected" if "selected" in pd.read_csv(derived / "flair" / "flair_index.csv", nrows=0).columns else "flair")
        fl["flair_date"] = pd.to_datetime(fl["patno"].map(dd), errors="coerce")
        man = man.merge(fl[["patno", "flair_date", "flair_batch", "flair_qc_pass"]].rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
    for mod in ("dat", "dwi", "nm", "flair"):
        if f"{mod}_date" in man:
            man[f"{mod}_days_from_t1"] = (man[f"{mod}_date"] - man["t1_date"]).dt.days
    return man


def assemble_features(derived_dir):
    """Manifest + features per subject: FastSurfer IDPs (as in fastsurfer_idps.csv), `dat_*` raw SBRs, `dwi_*`, `nm_*`
    (ratios/volumes only), `flair_*`. Values of QC-failed modalities are blanked; the QC flags stay."""
    derived = Path(derived_dir)
    man = build_manifest(derived)
    idps = pd.read_csv(derived / "fastsurfer_idps.csv", parse_dates=["SCAN_DATE"], low_memory=False)
    idps = idps.sort_values(["PATNO", "SCAN_DATE"]).drop_duplicates("PATNO", keep="first").drop(columns=["SCAN_DATE"])
    df = man.merge(idps, on="PATNO", how="left")
    blocks = {}
    dat = _read(derived / "datscan_full" / "datscan_sbr.csv")
    if dat is not None:
        cols = [c for c in DAT_COLS if c in dat and c not in ("image_id", "hdr_manufacturer", "hdr_model", "hdr_scale_fit", "reg_metric")]
        blocks["dat"] = dat[["patno"] + cols].rename(columns={c: f"dat_{c}" for c in cols})
    dwi = _read(derived / "dwi" / "dwi_features.csv")
    if dwi is not None:
        cols = [c for c in dwi.columns if c.endswith(("_fa", "_md", "_fw", "_fat")) and not c.startswith("n_")]
        blocks["dwi"] = dwi[["patno"] + cols].rename(columns={c: f"dwi_{c}" for c in cols})
    nmf = _read(derived / "nm" / "nm_features.csv")
    if nmf is not None:
        cols = [c for c in nmf.columns if c.startswith("nm_") and c.endswith(("_cnr", "_voxels"))]
        blocks["nm"] = nmf[["patno"] + cols]
    fl = _read(derived / "flair" / "flair_features.csv")
    if fl is not None:
        cols = ["wmh_log_mm3", "wmh_pv_mm3", "wmh_deep_mm3", "wmh_frac_wm", "wmh_n_lesions", "wmh_mm3"]
        blocks["flair"] = fl[["patno"] + cols].rename(columns={c: f"flair_{c}" for c in cols})
    for mod, b in blocks.items():
        df = df.merge(b.rename(columns={"patno": "PATNO"}), on="PATNO", how="left")
        feat_cols = [c for c in b.columns if c != "patno"]
        if f"{mod}_qc_pass" in df:
            bad = df[f"{mod}_qc_pass"].fillna(False).astype(bool).eq(False) & df[feat_cols[0]].notna()
            df.loc[bad, feat_cols] = np.nan
    return df


def feature_blocks(columns):
    """Modality block -> feature columns, for block-wise harmonisation / stacking."""
    cols = list(columns)
    return {"dat": [c for c in cols if c.startswith("dat_sbr")],
            "dwi": [c for c in cols if c.startswith("dwi_") and c.endswith(("_fa", "_md", "_fw", "_fat"))],
            "nm": [c for c in cols if c.startswith("nm_") and c.endswith(("_cnr", "_voxels"))],
            "flair": [c for c in cols if c.startswith("flair_wmh")]}
