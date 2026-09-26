"""
features.py — assemble the wide imaging-derived-phenotype (IDP) table from FastSurfer stats.

One row per processed session: identifiers (PATNO, EVENT_ID, IMAGEID, scan date), scanner
metadata from the dcm2niix sidecar, every FastSurfer regional volume (mm^3, prefix ``vol_``),
global measures (MaskVol, BrainSegVol, ...), and a few derived indices that matter for PD:
left/right asymmetry of subcortical structures and bilateral totals.
"""

import hashlib
import re
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from pie.imaging.fastsurfer import STATS_FILE, parse_stats, valid_etiv

BILATERAL = ["Putamen", "Caudate", "Pallidum", "Thalamus", "Hippocampus", "Amygdala", "Accumbens-area",
             "Lateral-Ventricle", "Cerebellum-Cortex", "Cerebellum-White-Matter", "VentralDC"]
META_COLS = ["Manufacturer", "ManufacturersModelName", "MagneticFieldStrength", "SoftwareVersions",
             "InstitutionName", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "SliceThickness"]


def _clean(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def build_idp_table(sessions, subjects_dir):
    """``sessions`` needs patno, image_id, session_date, EVENT_ID, protocol_phase and sidecar metadata."""
    rows = []
    for s in sessions.itertuples(index=False):
        stats = Path(subjects_dir) / s.image_id / STATS_FILE
        if not stats.exists():
            continue
        d = parse_stats(stats)
        row = {"PATNO": s.patno, "EVENT_ID": s.EVENT_ID, "IMAGEID": s.image_id, "SCAN_DATE": s.session_date,
               "protocol_phase": s.protocol_phase}
        row.update({c: getattr(s, c, None) for c in META_COLS})
        measures = {k: v for k, v in d.items() if k.endswith("Vol")}
        row.update(measures)
        row["eTIV"] = valid_etiv(Path(subjects_dir) / s.image_id / "mri")     # FreeSurfer eTIV, see fastsurfer.talairach_etiv
        for k, v in d.items():
            if k not in measures:
                row[f"vol_{_clean(k)}"] = v
        for st in BILATERAL:
            l, r = d.get(f"Left-{st}"), d.get(f"Right-{st}")
            if l is not None and r is not None:
                row[f"sum_{_clean(st)}"] = l + r
                row[f"asym_{_clean(st)}"] = (l - r) / (l + r) if (l + r) else None
        vent = [d.get(k, 0.0) for k in ["Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "3rd-Ventricle",
                                        "4th-Ventricle", "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent"]]
        row["sum_Ventricles"] = sum(vent)
        rows.append(row)
    return pd.DataFrame(rows)


FS7_TABLES = {"FS7_APARC_CTH": "fs7_cth_", "FS7_APARC_SA": "fs7_sa_", "FS7_ASEG_VOL": "fs7_", "MRIQC": "mriqc_"}


def fs7_tables(ppmi_dir):
    """PPMI's own FreeSurfer 7.3.2 tables (DK-atlas thickness and area, aseg volumes with eTIV) and MRIQC image-quality
    metrics, one row per PATNO + EVENT_ID, columns prefixed by source. The 2025 McGill/Nipoppy release covers baseline
    scans only; its T1 is PPMI's choice for that visit, which need not be the series PIE segmented."""
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


# ====================================================================================================
# Tissue volumes, intracranial volume and head-size adjustment
# Merged from volumes.py; kept together with the rest of the features measures.
#
# Whole-brain tissue volumes, an atrophy-robust intracranial volume estimate, and head-size adjustment.
#
# FastSurfer's ``BrainSegVol`` is a brain volume and shrinks with atrophy, so dividing a regional volume by it partly
# removes the disease effect one is trying to measure; ``MaskVol`` is a dilated brain mask, not an intracranial
# measurement. Head-size adjustment needs total intracranial volume (TIV), which does not change with atrophy.
# Studies using CAT12 or FreeSurfer adjust by TIV (e.g. Droby et al. 2025).
#
# Preferred TIV source: FastSurfer run with ``--tal_reg`` writes FreeSurfer's eTIV (``EstimatedTotalIntraCranialVol``)
# to the aseg stats, an established and validated measure. ``tiv_from_registration`` below is a fallback for runs
# without it. Smoke test on 8 PPMI subjects (18 September 2026): 1,238 to 1,523 mL, a plausible adult range, but
# correlation with MaskVol only 0.65 and 2 of 8 below the head-fit threshold. Against FreeSurfer 7.3.2 eTIV (PPMI's
# FS7_ASEG_VOL table) on 40 baseline scans (25 September 2026): r = 0.81, 9 % low, where MaskVol gave r = 0.85. Prefer
# eTIV; for PPMI baseline scans it is in FS7_ASEG_VOL_*.csv.
#
#     tissue_volumes(seg)                   total grey matter, white matter, brainstem, ventricles, subcortical grey (mm^3)
#     fetch_template(cache_dir)             TemplateFlow MNI152NLin2009cAsym res-02 head T1w and GM/WM/CSF maps (sha256-checked)
#     tiv_from_registration(head, ...)      warp the template's intracranial probability into subject space by an
#                                           affine head-to-head registration and sum it (the atlas-scaling principle of
#                                           Buckner et al. 2004, computed as a warped volume so no transform direction
#                                           or determinant convention can be misread)
#     adjust_for_head_size(v, tiv, ...)     residual (Jack 1989) or proportion method, fitted on a reference subset
#
# The registration must be head to head, skull included: a brain-to-brain affine inherits the atrophy that TIV is
# meant to be free of. Pass FastSurfer's conformed ``orig.mgz`` or the raw T1, not a skull-stripped image.
# ====================================================================================================

# FreeSurfer / FastSurfer aseg labels (aparc+aseg cortex labels 1000-2999 are counted as cortical grey matter)
CORTICAL_GM = (3, 42)
SUBCORTICAL_GM = (10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60)
CEREBELLAR_GM = (8, 47)
WHITE_MATTER = (2, 41, 7, 46, 77, 251, 252, 253, 254, 255)
BRAINSTEM = (16,)
VENTRICLES = (4, 5, 14, 15, 43, 44, 72)

TEMPLATEFLOW = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
TEMPLATE_FILES = {
    "t1w": ("tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz", "7c4e551ae8150ac1f468595bd62074aa15aed747cd334fce6d177182972b6220"),
    "gm": ("tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz", "66aab92950c4256a6560e7c5293484c5e62de1f8bfddee4e20e35ba725b2cf7d"),
    "wm": ("tpl-MNI152NLin2009cAsym_res-02_label-WM_probseg.nii.gz", "88d15b840383ab37dfd2d7355b89a9eadc23d5d93ab681afc3131be266fd1969"),
    "csf": ("tpl-MNI152NLin2009cAsym_res-02_label-CSF_probseg.nii.gz", "c98c2fb2beb6ebca86742a0580e8c7accb1df5b012da38fce0e93b057983c02c"),
}
TEMPLATE_TIV_MM3 = 1_884_000.0   # sum of the three maps x 8 mm^3, verified 18 September 2026 (1,884.0 mL)


def _download(url, path):
    """Fetch with an explicit User-Agent: NeuroVault's CDN answers 403 to urllib's default one."""
    req = urllib.request.Request(url, headers={"User-Agent": "parkinsons-insight-engine (+https://github.com/MJFF-ResearchCommunity/PIE)"})
    with urllib.request.urlopen(req, timeout=120) as response, open(path, "wb") as out:
        out.write(response.read())


def _voxel_mm3(img):
    return float(abs(np.linalg.det(img.affine[:3, :3])))


def _to_ants(img):
    import ants
    # registration inputs only: real FA and T1 maps carry NaN outside the brain, which ANTs rejects
    data = np.nan_to_num(np.asarray(img.dataobj, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return ants.from_nibabel_nifti(nib.Nifti1Image(data, img.affine))


def _drop_transforms(*regs):
    """Delete the transform files ``ants.registration`` writes to the temp directory. ANTsPy never removes them: a SyN
    leaves ~140 MB, and a cohort run filled a 1.9 TB root disk with them (September 2026)."""
    for reg in regs:
        for f in {*reg["fwdtransforms"], *reg["invtransforms"]}:
            Path(f).unlink(missing_ok=True)


def _seed_ants(seed):
    """Seed antsRegistration (``--random-seed``) for every later call in this process. ANTsPy 0.6 swallows a
    ``random_seed=`` keyword (it lands in **kwargs unused); the seed reaches antsRegistration only through
    ``ants.config``. on=False keeps ITK multithreaded: even seeded and single-threaded, Mattes/MI registrations are not
    bit-reproducible in ANTsPy 0.6.3 (translations differ by ~0.01 mm between runs; ANTs' Repro mode, GC metric, would be)."""
    import ants

    ants.config.set_ants_deterministic(False, seed_value=int(seed))


def _from_ants(ants_img, like):
    """Back to nibabel on the grid of ``like``; refuses any output whose geometry differs from ``like``."""
    import ants
    out = ants.to_nibabel_nifti(ants_img)
    if out.shape[:3] != like.shape[:3] or not np.allclose(out.affine, like.affine, atol=1e-3):
        raise ValueError("resampled image is not on the subject grid; orientation handling is wrong")
    return np.asarray(out.dataobj)


def tissue_volumes(seg_img):
    """Volumes (mm^3) from a FastSurfer/FreeSurfer segmentation (aseg or aparc+aseg)."""
    seg = np.asarray(seg_img.dataobj).astype(np.int32)
    v = _voxel_mm3(seg_img)
    count = lambda labels: float(np.isin(seg, labels).sum() * v)
    cortex = count(CORTICAL_GM) + float(((seg >= 1000) & (seg < 3000)).sum() * v)
    out = {"cortical_gm": cortex, "subcortical_gm": count(SUBCORTICAL_GM), "cerebellar_gm": count(CEREBELLAR_GM),
           "white_matter": count(WHITE_MATTER), "brainstem": count(BRAINSTEM), "ventricles": count(VENTRICLES)}
    out["total_gm"] = out["cortical_gm"] + out["subcortical_gm"] + out["cerebellar_gm"]
    for name, labels in (("caudate", (11, 50)), ("putamen", (12, 51)), ("pallidum", (13, 52))):
        out[f"{name}_l"], out[f"{name}_r"] = count(labels[:1]), count(labels[1:])
    return out


def fetch_template(cache_dir, download=True):
    """TemplateFlow head T1w and the summed GM+WM+CSF probability (intracranial map), sha256-verified."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    imgs = {}
    for key, (name, digest) in TEMPLATE_FILES.items():
        path = cache / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            _download(TEMPLATEFLOW + name, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name}: checksum differs from the verified TemplateFlow release")
        imgs[key] = nib.load(path)
    icv = sum(np.asarray(imgs[k].dataobj, float) for k in ("gm", "wm", "csf"))
    return imgs["t1w"], nib.Nifti1Image(np.clip(icv, 0, 1).astype(np.float32), imgs["gm"].affine)


def tiv_from_registration(head_img, template_head, template_icv, seed=0, min_correlation=0.5):
    """TIV (mm^3): affine head-to-head registration, then the template's intracranial map summed in subject space.

    Returns {"tiv_mm3", "template_head_correlation", "qc_pass"}. The correlation is between the warped template
    head and the subject head over the union of their non-zero voxels, so a misplaced or mis-scaled head lowers it;
    below ``min_correlation`` the registration is rejected rather than trusted.
    """
    import ants

    fixed, moving = _to_ants(head_img), _to_ants(template_head)
    _seed_ants(seed)
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine", aff_metric="mattes")
    warped_icv = ants.apply_transforms(fixed=fixed, moving=_to_ants(template_icv), transformlist=reg["fwdtransforms"],
                                       interpolator="linear")
    warped_head = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg["fwdtransforms"], interpolator="linear")
    _drop_transforms(reg)
    subject, template = np.asarray(head_img.dataobj, float), _from_ants(warped_head, head_img).astype(float)
    inside = ((subject > 0) | (template > 0)) & np.isfinite(template)   # union: a misplaced head lowers r
    r = float(np.corrcoef(subject[inside], template[inside])[0, 1]) if inside.sum() > 10 else float("nan")
    tiv = float(np.clip(_from_ants(warped_icv, head_img), 0, 1).sum() * _voxel_mm3(head_img))
    return {"tiv_mm3": tiv, "template_head_correlation": r, "qc_pass": bool(np.isfinite(r) and r >= min_correlation)}


def adjust_for_head_size(values, tiv, method="residual", reference=None):
    """Head-size-adjusted volumes.

    residual    v - b * (tiv - mean_tiv), with b and mean_tiv fitted on ``reference`` rows only (e.g. controls, or
                the training fold of a prediction model, so no evaluation row informs the adjustment)
    proportion  v / tiv * mean_tiv over the reference rows
    """
    v, t = np.asarray(values, float), np.asarray(tiv, float)
    ref = np.ones(len(v), bool) if reference is None else np.asarray(reference, bool)
    ok = ref & np.isfinite(v) & np.isfinite(t)
    if ok.sum() < 3:
        raise ValueError("too few reference rows to fit the adjustment")
    mean_tiv = float(t[ok].mean())
    if method == "proportion":
        return v / t * mean_tiv
    if method == "residual":
        slope = float(np.polyfit(t[ok], v[ok], 1)[0])
        return v - slope * (t - mean_tiv)
    raise ValueError("method must be 'residual' or 'proportion'")
