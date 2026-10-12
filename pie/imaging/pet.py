"""
pet.py — [18F]AV-133 (VMAT2) PET: LONI DICOM -> time-weighted mean -> striatal binding ratios on the subject's own
FastSurfer segmentation.

PPMI's method (IND, "AV-133 PET Image Processing Methods for Calculation of SBR", rev. 25 Apr 2014; acquisition from the
PET Technical Operations Manual, protocol 004, v4.0 2022): a 222 MBq bolus, 2 x 5 min frames from 80 (+-5) min after
injection, attenuation / scatter / randoms / decay corrected, iterative reconstruction in quantitative units (Bq/mL),
submitted without post-filtering. Frames are checked for motion (corrected where necessary) and averaged time-weighted;
the mean is co-registered to the subject's MRI, an IND striatal VOI template (caudate, anterior and posterior putamen,
left / right) is placed in MNI space, and SBR = target / occipital - 1. The newer sub-regional rows of PPMI's
``PET_SBR_Analysis`` add ten striatal sub-regions and the cerebellum, still against the occipital lobe.

PIE, per scan (every AV-133 visit, all measured on the participant's earliest finished FastSurfer T1, as IND carries the
baseline VOIs to follow-ups):

    index_pet / select_series : LONI series + a first-file DICOM probe (modality, tracer, frames, post-filter); one
                                reconstruction per participant-day, unfiltered preferred; separate static series stacked
    load_scan                 : dcm2niix (PET-BIDS sidecar) + frame timing read from every DICOM header
    mean_image                : frames from 60 min after injection on; inter-frame rigid motion (MI), frames aligned when
                                it exceeds 1 mm; time-weighted mean
    register_pet_to_t1        : rigid Mattes MI, PET mean -> conformed T1, metric restricted to the brain + 10 mm
    quantify                  : PET resampled onto the T1; FastSurfer caudate / putamen with anterior / posterior halves
                                (``datscan.sbr_from_arrays``) against occipital cortex (``sbr_*``) and cerebellar cortex
                                (``sbrcb_*``); ``cb_occ`` = cerebellum / occipital - 1 (PPMI's ``PET_CBM``)

Values are not corrected for partial volume, and the ROIs are whole anatomical structures rather than IND's small VOIs, so
absolute SBRs should sit below PPMI's; ``agreement_table`` reports how far, per PPMI pipeline (never pooled).

    venv_imaging/bin/python -m pie.imaging.pet --zips <AV-133 PET zips> --sessions <derived>/sessions.csv \\
        --fastsurfer-dir <derived>/fastsurfer --work-dir <derived>/pet --workers 4 [--keep-nifti]
    ... same arguments --agreement PPMI      # pet_agreement.csv against PET_SBR_Analysis, no processing
"""

import json
import re
import shutil
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom

from pie.stats.agreement import agreement, icc21 as _icc21

from .datscan import CAUDATE_L, CAUDATE_R, ITK_SEED, OCCIPITAL, PUTAMEN_L, PUTAMEN_R, _sitk_from_nib, sbr_from_arrays, synthetic_spect

AV133 = r"AV.?133|VMAT|DTBZ|florbenazine"
OTHER_TRACERS = r"DPA.?714|C05.?05|MK.?6240|UCB|SV2A|PE2I|PI.?2620|DOPA|FD4|PIB|AV.?45|AV.?1451|florbeta|flortaucipir"
UNFILTERED = r"all.?pass|no.?filt|unfilt|filter.?(?:none|0)"
FILTERED = r"gauss|hann|butter|filt|fwhm"
CEREBELLUM = [8, 47]                   # FastSurfer left / right cerebellar cortex
ANALYSIS_FROM_MIN = 60.0               # analysis frames start at or after this (protocol: 2 x 5 min from 80 +- 5 min)
PROTOCOL_START_MIN = (75.0, 85.0)
MOTION_CORRECT_MM = 1.0                # IND: frames "motion corrected where necessary"
# a linear activity or count scale: SBRs are ratios, so Philips CNTS (Bq/mL factor in a private tag) is as good as Bq/mL
RATIO_UNITS = {"bq/ml", "bqml", "g/ml", "gml", "cnts", "counts"}
REGIONS = ["caudate_l", "caudate_r", "putamen_l_ant", "putamen_r_ant", "putamen_l_post", "putamen_r_post"]
PPMI_COLUMNS = dict(zip(REGIONS, ["PET_LCAUD_S", "PET_RCAUD_S", "PET_LPUTANT_S", "PET_RPUTANT_S", "PET_LPUTPOST_S", "PET_RPUTPOST_S"]))
TRACERS = {"AV-133": r"AV.?133", "DPA-714": r"DPA.?714", "C05-05": r"C05.?05"}
PROCESSING_VERSION = "2026-10-11-av133-v1"

# Tracers PIE processes: series pattern, analysis frames (start, min after injection), the protocol's start window
# (``timing_ok``), outcome table. SV2A ([18F]SynVesT-1) and tau ([18F]PI-2620): PPMI's static scans start ~55-60 min
# after injection (SV2A_PET_Acquisition_Metadata, TAU_PET_Metadata); SUVR against the centrum semiovale (SV2A) and the
# ventral cerebellum (tau), as PPMI's SV2A_PET_SUVR_Analysis and TAU_PET_Analysis.
TRACER_CONFIG = {
    "av133": {"pattern": AV133, "from_min": ANALYSIS_FROM_MIN, "window": PROTOCOL_START_MIN, "csv": "pet_av133.csv"},
    "sv2a": {"pattern": r"SV2A|SynVesT|SDM.?8|UCB.?J", "from_min": 40.0, "window": (45.0, 70.0), "csv": "pet_sv2a.csv", "reference": "cs"},
    "tau": {"pattern": r"PI.?2620|\btau\b", "from_min": 40.0, "window": (45.0, 70.0), "csv": "pet_tau.csv", "reference": "cerebellum_ventral"},
}
# FastSurfer aseg / DKT labels for the SUVR tracers, named after the PPMI region they approximate (Hammers-atlas names in
# SV2A_PET_SUVR_Analysis; Braak VI after Schoell et al. 2016: pericalcarine, postcentral, precentral, paracentral)
_LR = lambda l, r: {"_l": l, "_r": r}
SUVR_REGIONS = {
    **{f"caudate{k}": (v, f"CAUDATENUCL{k.upper()}") for k, v in _LR([11], [50]).items()},
    **{f"putamen{k}": (v, f"PUTAMEN{k.upper()}") for k, v in _LR([12], [51]).items()},
    **{f"pallidum{k}": (v, f"PALLIDUM{k.upper()}") for k, v in _LR([13], [52]).items()},
    **{f"thalamus{k}": (v, f"THALAMUS{k.upper()}") for k, v in _LR([10], [49]).items()},
    **{f"amygdala{k}": (v, f"AMYGDALA{k.upper()}") for k, v in _LR([18], [54]).items()},
    **{f"hippocampus{k}": (v, f"HIPPOCAMPUS{k.upper()}") for k, v in _LR([17], [53]).items()},
    **{f"accumbens{k}": (v, f"NUCLACCUMB{k.upper()}") for k, v in _LR([26], [58]).items()},
    **{f"insula{k}": (v, f"INSULA{k.upper()}") for k, v in _LR([1035], [2035]).items()},
    **{f"fusiform{k}": (v, f"G_FUS{k.upper()}") for k, v in _LR([1007], [2007]).items()},
    **{f"cuneus{k}": (v, f"OL_CUNEUS{k.upper()}") for k, v in _LR([1005], [2005]).items()},
    **{f"lingual{k}": (v, f"OL_LING_G{k.upper()}") for k, v in _LR([1013], [2013]).items()},
    **{f"precentral{k}": (v, f"FL_PRECEN_G{k.upper()}") for k, v in _LR([1024], [2024]).items()},
    **{f"postcentral{k}": (v, f"PL_POSTCE_G{k.upper()}") for k, v in _LR([1022], [2022]).items()},
    **{f"superiorfrontal{k}": (v, f"FL_SUP_FR_G{k.upper()}") for k, v in _LR([1028], [2028]).items()},
    **{f"superiorparietal{k}": (v, f"PL_SUP_PA_G{k.upper()}") for k, v in _LR([1029], [2029]).items()},
    **{f"cingulate_ant{k}": (v, f"G_CING_ANT{k.upper()}") for k, v in _LR([1002, 1026], [2002, 2026]).items()},
    **{f"cingulate_post{k}": (v, f"G_CING_POST{k.upper()}") for k, v in _LR([1023, 1010], [2023, 2010]).items()},
    **{f"parahippocampal{k}": (v, f"G_PARAH_AMB{k.upper()}") for k, v in _LR([1016, 1006], [2016, 2006]).items()},
    "brainstem": ([16], "BRAINSTEM"),
    "braak6": ([1021, 2021, 1022, 2022, 1024, 2024, 1017, 2017], "BRAAK_6"),
}


# ------------------------------------------------------------------------------------------ index / select
def _text(v):
    return "\\".join(map(str, v)) if isinstance(v, (list, pydicom.multival.MultiValue)) else str(v if v is not None else "")


def _probe(zip_path, prefixes):
    """Header fields of the first DICOM of every series prefix in one zip (no pixel data)."""
    out, first = {}, {}
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            p = n.rsplit("/", 1)[0] + "/"
            if p in prefixes and n.endswith(".dcm"):
                first.setdefault(p, n)
        for p, n in first.items():
            with z.open(n) as fh:
                h = pydicom.dcmread(fh, stop_before_pixels=True, force=True)
            r = (h.get("RadiopharmaceuticalInformationSequence") or [pydicom.Dataset()])[0]
            out[p] = {"modality": _text(h.get("Modality")), "tracer": _text(r.get("Radiopharmaceutical")),
                      "n_frames": int(h.get("NumberOfTimeSlices") or 1), "units": _text(h.get("Units")),
                      "kernel": _text(h.get("ConvolutionKernel")), "image_type": _text(h.get("ImageType")),
                      "manufacturer": _text(h.get("Manufacturer")), "model": _text(h.get("ManufacturerModelName"))}
    return out


def index_pet(zips, index_csv=None):
    """``batch.index_series`` rows (zip, prefix, patno, desc, date, image_id, n_files) plus the first-file probe; cached
    in ``index_csv``, zips missing from the cache are indexed and appended."""
    from .batch import index_series

    zips = [str(z) for z in zips]
    cached = pd.read_csv(index_csv, dtype={"image_id": str}) if index_csv and Path(index_csv).exists() else None
    todo = [z for z in zips if cached is None or z not in set(cached["zip"].astype(str))]
    parts = [cached] if cached is not None else []
    if todo:
        idx = index_series(todo)
        probes = {}
        for zp, g in idx.groupby("zip") if len(idx) else ():
            probes.update(_probe(zp, set(g["prefix"])))
        parts.append(pd.concat([idx, pd.DataFrame([probes.get(p, {}) for p in idx.get("prefix", [])], index=idx.index)], axis=1))
    idx = pd.concat(parts, ignore_index=True)
    if index_csv:
        idx.to_csv(index_csv, index=False)
    return idx[idx["zip"].astype(str).isin(zips)].reset_index(drop=True)


def _postfilter(text):
    t = str(text).lower()
    return "unfiltered" if re.search(UNFILTERED, t) else "filtered" if re.search(FILTERED, t) else "unknown"


def select_series(idx, prefer="unfiltered", tracer="av133"):
    """Flag one PET reconstruction per participant-day (``selected``). Candidates: modality PT whose description or
    DICOM tracer names no other PPMI tracer (sites often leave the scanner's FDG default, so FDG in the tracer field
    alone does not exclude: the PPMI table join decides the tracer). Order: the ``prefer``red post-filter class (the
    TOM asks for unfiltered images; ``postfilter`` from description + convolution kernel), then most frames, then most
    files. A single-frame winner takes every single-frame series of its class with the same file count and kernel: the
    TOM allows sequential statics, one series per frame."""
    d = idx.copy()
    d["kernel"] = d["kernel"].fillna("").astype(str)
    d["postfilter"] = (d["desc"].astype(str) + " " + d["kernel"]).map(_postfilter)
    named = d["desc"].astype(str) + " " + d["tracer"].fillna("").astype(str)
    if tracer == "av133":
        d["candidate"] = d["modality"].eq("PT") & ~named.str.contains(OTHER_TRACERS, case=False, regex=True)
    else:      # SV2A and tau zips hold only their tracer in PPMI's downloads, but a name must say so
        d["candidate"] = d["modality"].eq("PT") & named.str.contains(TRACER_CONFIG[tracer]["pattern"], case=False, regex=True)
    rank = d["postfilter"].map({prefer: 0, "unknown": 1}).fillna(2)
    d["selected"] = False
    for _, g in d[d["candidate"]].assign(_rank=rank).groupby(["patno", "date"]):
        g = g.sort_values(["_rank", "n_frames", "n_files"], ascending=[True, False, False])
        best = g.iloc[0]
        keep = [best.name]
        if best["n_frames"] <= 1:
            same = (g["n_frames"] <= 1) & g["postfilter"].eq(best["postfilter"]) & g["n_files"].eq(best["n_files"]) & g["kernel"].eq(best["kernel"])
            keep = list(g.index[same])
        d.loc[keep, "selected"] = True
    return d


# ------------------------------------------------------------------------------------------ frames
def _num(v):
    return float(v) if v not in (None, "") else np.nan


def _dt(date, time):
    time = str(time or "").strip()
    if not time:
        return pd.NaT
    return pd.to_datetime(f"{date or '19000101'} {time if '.' in time else time + '.0'}", format="%Y%m%d %H%M%S.%f", errors="coerce")


def injection_time(h):
    """Radiopharmaceutical start (DICOM date-time, else the time on the acquisition date), NaT when absent."""
    r = (h.get("RadiopharmaceuticalInformationSequence") or [None])[0]
    if r is None:
        return pd.NaT
    stamp = str(r.get("RadiopharmaceuticalStartDateTime", "") or "")
    if stamp:
        return _dt(stamp[:8], stamp[8:].split("+")[0].split("-")[0])
    return _dt(h.get("AcquisitionDate") or h.get("SeriesDate"), r.get("RadiopharmaceuticalStartTime"))


def series_frames(headers):
    """Frames of one or more PET series from their DICOM headers (one per file, any order), sorted by time:
    ``series``, ``start_s`` (seconds after injection, NaN without an injection time), ``duration_s``
    (ActualFrameDuration), ``n_slices``. Within a series frames are told apart by FrameReferenceTime (Type 1); their
    starts are FrameReferenceTime - duration / 2 (the frame mid-time, as dcm2niix's FrameTimesStart) relative to the
    series' first frame, anchored at the series' earliest AcquisitionTime (the start of acquisition, which every vendor
    sets). Overlapping frames (e.g. a summed image stacked with its own frames) are refused."""
    d = pd.DataFrame([{"series": str(h.get("SeriesInstanceUID", "")), "frt": _num(h.get("FrameReferenceTime")),
                       "duration_s": _num(h.get("ActualFrameDuration")) / 1000,
                       "acq": _dt(h.get("AcquisitionDate") or h.get("SeriesDate"), h.get("AcquisitionTime")),
                       "inj": injection_time(h)} for h in headers])
    f = d.groupby(["series", "frt", "duration_s"], dropna=False).agg(acq=("acq", "min"), inj=("inj", "first"),
                                                                      n_slices=("acq", "size")).reset_index()
    rel = f["frt"] / 1000 - f["duration_s"] / 2
    rel = (rel - rel.groupby(f["series"]).transform("min")).fillna(0)
    when = f.groupby("series")["acq"].transform("min") + pd.to_timedelta(rel, unit="s")
    f["start_s"] = (when - f["inj"]).dt.total_seconds()
    f.loc[f["start_s"] < -6 * 3600, "start_s"] += 86400           # injected before midnight, dates absent
    f = f.assign(_when=when).sort_values(["_when", "frt"]).reset_index(drop=True)
    gaps = f["_when"].diff().dt.total_seconds().iloc[1:].to_numpy()
    if (gaps < f["duration_s"].iloc[:-1].to_numpy() - 1).any():
        raise ValueError("PET frames overlap in time (a summed image among the frames?)")
    return f[["series", "start_s", "duration_s", "n_slices"]]


def load_scan(series_rows, work, cache):
    """Convert the selected series (``batch.convert_series``) and read their frame timing. Returns (data x,y,z,t
    float32, frames, affine, PET-BIDS sidecar, first DICOM header). Frames and volumes are matched per series in time
    order; dcm2niix's FrameDuration must agree. The sidecar keeps dcm2niix's PET-BIDS fields (TracerName,
    InjectedRadioactivity, Units, DecayCorrection, ReconMethodName, ...) with FrameTimesStart / FrameDuration of the
    stacked frames relative to TimeZero = injection (InjectionStart 0, ScanStart = first frame)."""
    from .batch import convert_series

    vols, frames, sidecar, affine, first = [], [], None, None, None
    for r in series_rows:
        niis = convert_series(r["zip"], r["prefix"], Path(work) / "nii", archive_cache=cache)
        if len(niis) != 1:
            raise ValueError(f"{r['image_id']}: dcm2niix wrote {len(niis)} images")
        with cache.open(r["zip"]) as z:
            heads = []
            for n in z.namelist():
                if n.startswith(r["prefix"]) and n.endswith(".dcm"):
                    with z.open(n) as fh:
                        heads.append(pydicom.dcmread(fh, stop_before_pixels=True, force=True))
        f = series_frames(heads)
        img = nib.load(niis[0])
        data = np.asanyarray(img.dataobj, dtype=np.float32)
        data = data[..., None] if data.ndim == 3 else data
        meta = json.loads(Path(niis[0][:-7] + ".json").read_text()) if Path(niis[0][:-7] + ".json").exists() else {}
        dur, fts = (np.atleast_1d(meta.get(k, [])).astype(float) for k in ("FrameDuration", "FrameTimesStart"))
        if (data.shape[-1] != len(f) or (len(dur) == len(f) and not np.allclose(dur, f["duration_s"], atol=1, equal_nan=True))
                or (len(fts) > 1 and not (np.diff(fts) > 0).all())):     # dcm2niix's volumes must run in time order, as the frames do
            raise ValueError(f"{r['image_id']}: {data.shape[-1]} volumes but frames {f['duration_s'].tolist()} s in the headers")
        if affine is None:
            affine, sidecar, first = img.affine, meta, heads[0]
        elif data.shape[:3] != vols[0].shape[:3] or not np.allclose(img.affine, affine, atol=1e-3):
            raise ValueError("stacked PET series are not on one grid")
        vols += [data[..., k] for k in range(data.shape[-1])]
        frames.append(f)
    frames = pd.concat(frames, ignore_index=True)
    order = np.argsort(frames["start_s"].to_numpy() if frames["start_s"].notna().all() else np.arange(len(frames)), kind="stable")
    frames = frames.iloc[order].reset_index(drop=True)
    data = np.stack([vols[k] for k in order], axis=-1)
    inj = injection_time(first)
    sidecar = dict(sidecar, TimeZero=inj.strftime("%H:%M:%S") if pd.notna(inj) else None, InjectionStart=0,
                   ScanStart=float(frames["start_s"].iloc[0]), FrameTimesStart=frames["start_s"].tolist(),
                   FrameDuration=frames["duration_s"].tolist())
    return np.nan_to_num(data), frames, affine, sidecar, first


# ------------------------------------------------------------------------------------------ mean image
def time_weighted_mean(data, durations):
    """Mean over the last axis weighted by frame duration (equal weights when a duration is unknown)."""
    w = np.asarray(durations, dtype=float)
    w = np.ones_like(w) if not np.isfinite(w).all() or w.sum() <= 0 else w
    return (np.tensordot(data, w, axes=([-1], [0])) / w.sum()).astype(np.float32)


def _displacement_mm(tx, radius=50.0):
    """Largest displacement of the corners of a +-radius mm cube about the transform centre (rotation counts)."""
    c = np.array(tx.GetCenter())
    corners = c + radius * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return float(max(np.linalg.norm(np.array(tx.TransformPoint(tuple(map(float, p)))) - p) for p in corners))


def mean_image(data, frames, affine, sampling_seed=ITK_SEED, from_min=ANALYSIS_FROM_MIN, window=PROTOCOL_START_MIN):
    """Analysis frames (starting >= ANALYSIS_FROM_MIN after injection; all frames when timing is unknown or none
    qualifies), rigid inter-frame motion of each against the first (MI, ``dwi._register_volume``), frames resampled onto
    the first when the largest displacement exceeds MOTION_CORRECT_MM, then the time-weighted mean.
    Returns (mean x,y,z, info dict of QC fields)."""
    import SimpleITK as sitk

    from .dwi import _register_volume, _sitk_native

    start = frames["start_s"].to_numpy(dtype=float)
    use = start >= from_min * 60 if np.isfinite(start).all() else np.ones(len(frames), bool)
    use = use if use.any() else np.ones(len(frames), bool)
    d, dur = data[..., use], frames["duration_s"].to_numpy(dtype=float)[use]
    motion, corrected = np.nan, False
    if d.shape[-1] > 1:
        ref = _sitk_native(d[..., 0], affine)
        moving = [_sitk_native(d[..., k], affine) for k in range(1, d.shape[-1])]
        txs = [_register_volume(ref, m, sampling=0.2, iterations=100, sampling_seed=sampling_seed) for m in moving]
        motion = max(_displacement_mm(t) for t in txs)
        if motion > MOTION_CORRECT_MM:
            d = np.stack([d[..., 0]] + [np.transpose(sitk.GetArrayFromImage(sitk.Resample(m, ref, t, sitk.sitkLinear, 0.0)), (2, 1, 0))
                                        for m, t in zip(moving, txs)], axis=-1)
            corrected = True
    s = start[use]
    info = {"n_frames": int(len(frames)), "n_frames_used": int(use.sum()), "frame_minutes": float(np.nansum(dur) / 60),
            "start_min": float(s[0] / 60), "end_min": float((s[-1] + dur[-1]) / 60),
            "timing_ok": bool(window[0] <= s[0] / 60 <= window[1]) if np.isfinite(s[0]) else np.nan,
            "motion_mm": motion, "motion_corrected": corrected}
    return time_weighted_mean(d, dur), info


# ------------------------------------------------------------------------------------------ registration / SBR
def register_pet_to_t1(pet_img, t1_img, aparc_img, sampling_seed=ITK_SEED):
    """Rigid Mattes MI registration of the PET mean (moving, native voxel order) to the conformed T1 (fixed), the metric
    restricted to the FastSurfer brain dilated 10 mm (skull margin kept, neck and face out), started from the
    centres of mass. Returns (transform T1 point -> PET point, metric; more negative is better)."""
    import SimpleITK as sitk

    from .dwi import _sitk_native

    fixed = _sitk_from_nib(t1_img)
    moving = _sitk_native(np.asanyarray(pet_img.dataobj).astype(np.float32), pet_img.affine)
    lab = _sitk_from_nib(nib.Nifti1Image(np.asanyarray(aparc_img.dataobj).astype(np.float32), aparc_img.affine))
    mask = sitk.BinaryDilate(lab > 0.5, [max(1, int(round(10.0 / s))) for s in fixed.GetSpacing()])
    init = sitk.CenteredTransformInitializer(sitk.Mask(fixed, mask), moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricFixedMask(mask)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=int(sampling_seed))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=300, relaxationFactor=0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4, 2, 1])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init, inPlace=False)
    tx = reg.Execute(fixed, moving)
    return tx, float(reg.GetMetricValue())


def quantify(pet_img, t1_img, aparc_img, sampling_seed=ITK_SEED):
    """Register (``register_pet_to_t1``), resample the PET onto the T1 grid and measure FastSurfer ROIs there (voxels
    outside the PET field of view are left out). Returns (row dict, transform). Columns: ``sbr_*`` against occipital
    cortex (``datscan.sbr_from_arrays``: caudate / putamen per side and their anterior / posterior halves split at the
    median anterior coordinate), ``sbrcb_*`` against cerebellar cortex, ``mean_*``, ``cb_occ``, voxel counts
    ``n_<roi>``, field-of-view coverage ``cov_{striatum,occipital,cerebellum}``, ``reg_pet_t1_mi`` and
    ``reg_template_r`` (correlation, over the head, with ``datscan.synthetic_spect``'s striatum-hot template)."""
    import SimpleITK as sitk

    from .dwi import _sitk_native

    tx, metric = register_pet_to_t1(pet_img, t1_img, aparc_img, sampling_seed)
    fixed = _sitk_from_nib(t1_img)
    moving = _sitk_native(np.asanyarray(pet_img.dataobj).astype(np.float32), pet_img.affine)
    ones = sitk.Image(moving.GetSize(), sitk.sitkFloat32) + 1
    ones.CopyInformation(moving)
    S = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0))
    inside = sitk.GetArrayFromImage(sitk.Resample(ones, fixed, tx, sitk.sitkLinear, 0.0)) > 0.99
    L = np.rint(sitk.GetArrayFromImage(_sitk_from_nib(nib.Nifti1Image(np.asanyarray(aparc_img.dataobj).astype(np.float32), aparc_img.affine)))).astype(np.int32)
    T = sitk.GetArrayFromImage(_sitk_from_nib(synthetic_spect(t1_img, aparc_img)))
    head = (T > 0.05) & inside
    out = {"reg_pet_t1_mi": metric, "reg_template_r": float(np.corrcoef(S[head], T[head])[0, 1]) if head.sum() > 100 else np.nan}
    groups = {"striatum": [CAUDATE_L, CAUDATE_R, PUTAMEN_L, PUTAMEN_R], "occipital": OCCIPITAL, "cerebellum": CEREBELLUM}
    for name, ids in groups.items():
        m = np.isin(L, ids)
        out[f"cov_{name}"] = float(inside[m].mean()) if m.any() else np.nan
    L = np.where(inside, L, 0)
    if not L.any():
        raise ValueError("no FastSurfer label inside the PET field of view")
    box = tuple(slice(max(0, a.min() - 2), a.max() + 3) for a in np.nonzero(L))
    S, L = S[box], L[box]
    ap = np.broadcast_to(np.arange(L.shape[1], dtype=float)[None, :, None], L.shape)   # canonical grid: +y index = anterior
    sbr = sbr_from_arrays(S, L, search_vox=0, ap=ap, spacing_mm=float(fixed.GetSpacing()[1]))
    out.update({k: v for k, v in sbr.items() if not k.startswith("sbrwm_") and k not in ("mean_wm", "n_wm_voxels", "shift_vox")})
    cb = np.isin(L, CEREBELLUM)
    out["mean_cerebellum"] = mean_cb = float(S[cb].mean()) if cb.sum() >= 50 else np.nan
    for k in [k for k in sbr if k.startswith("sbr_")]:
        out["sbrcb_" + k[4:]] = out["mean_" + k[4:]] / mean_cb - 1 if mean_cb > 0 else np.nan
    out["cb_occ"] = mean_cb / out["mean_occipital"] - 1 if out["mean_occipital"] > 0 else np.nan
    for name, ids in {"caudate_l": [CAUDATE_L], "caudate_r": [CAUDATE_R], "putamen_l": [PUTAMEN_L], "putamen_r": [PUTAMEN_R],
                      "occipital": OCCIPITAL, "cerebellum": CEREBELLUM}.items():
        out[f"n_{name}"] = int(np.isin(L, ids).sum())
    return out, tx


def suvr_reference(L, ijk_world_z, name, spacing_mm=1.0):
    """Reference masks on the T1 label grid: ``cs`` = centrum semiovale, cerebral white matter at least 8 mm from any
    other label, above the lateral ventricles' top (SV2A); ``cerebellum_ventral`` = cerebellar cortex below its median
    height, at least 6 mm from the cerebrum (tau)."""
    from scipy import ndimage

    if name == "cs":
        wm = np.isin(L, [2, 41])
        core = ndimage.distance_transform_edt(wm) * spacing_mm >= 8
        vent = np.isin(L, [4, 43])
        top = ijk_world_z[vent].max() if vent.any() else -np.inf
        return core & (ijk_world_z > top)
    cb = np.isin(L, CEREBELLUM)
    cerebrum = (L > 0) & ~np.isin(L, CEREBELLUM + [7, 46, 16, 15])
    far = ndimage.distance_transform_edt(~cerebrum) * spacing_mm >= 6
    return cb & far & (ijk_world_z < np.median(ijk_world_z[cb])) if cb.any() else cb


def quantify_suvr(pet_img, t1_img, aparc_img, reference, sampling_seed=ITK_SEED):
    """SV2A / tau: register (``register_pet_to_t1``), resample the PET onto the T1 and give ``suvr_<region>`` (region
    mean / reference mean, ``SUVR_REGIONS``), ``mean_reference``, ``n_reference``, ``cov_reference`` (share of the
    reference inside the field of view) and ``reg_pet_t1_mi``. Returns (row dict, transform)."""
    import SimpleITK as sitk

    from .dwi import _sitk_native

    tx, metric = register_pet_to_t1(pet_img, t1_img, aparc_img, sampling_seed)
    fixed = _sitk_from_nib(t1_img)
    moving = _sitk_native(np.asanyarray(pet_img.dataobj).astype(np.float32), pet_img.affine)
    ones = sitk.Image(moving.GetSize(), sitk.sitkFloat32) + 1
    ones.CopyInformation(moving)
    S = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0))
    inside = sitk.GetArrayFromImage(sitk.Resample(ones, fixed, tx, sitk.sitkLinear, 0.0)) > 0.99
    L = np.rint(sitk.GetArrayFromImage(_sitk_from_nib(nib.Nifti1Image(np.asanyarray(aparc_img.dataobj).astype(np.float32), aparc_img.affine)))).astype(np.int32)
    zz, yy, xx = np.indices(L.shape)          # sitk arrays are (z, y, x): world (LPS) z of each voxel
    world = np.array(fixed.GetOrigin())[2] + np.einsum("j,j...->...", np.array(fixed.GetDirection()).reshape(3, 3)[2] * np.array(fixed.GetSpacing()),
                                                         np.stack([xx, yy, zz]))
    ref = suvr_reference(L, world, reference, float(np.mean(fixed.GetSpacing())))
    out = {"reg_pet_t1_mi": metric, "cov_reference": float(inside[ref].mean()) if ref.any() else np.nan}
    ref &= inside
    out["n_reference"] = int(ref.sum())
    out["mean_reference"] = mref = float(S[ref].mean()) if ref.sum() >= 50 else np.nan
    for name, (ids, _) in SUVR_REGIONS.items():
        m = np.isin(L, ids) & inside
        out[f"suvr_{name}"] = float(S[m].mean() / mref) if m.sum() >= 20 and mref > 0 else np.nan
    return out, tx


def _suv(mean, sidecar, header, start_s):
    """Mean of a region as SUV (body-weight), from Bq/mL with the injected dose decayed to the image's decay-correction
    reference (START: the first frame; ADMIN: injection); NaN when units, dose or weight are missing. A units check."""
    units = str(sidecar.get("Units", ""))
    if units in ("g/mL", "GML"):
        return float(mean)
    dose, weight = float(sidecar.get("InjectedRadioactivity") or np.nan) * 1e6, float(header.get("PatientWeight") or np.nan)
    corr = str(sidecar.get("DecayCorrection", "")).upper()
    if units not in ("Bq/mL", "BQML") or corr not in ("START", "ADMIN") or not (dose > 0 and weight > 0):
        return np.nan
    half = float(sidecar.get("RadionuclideHalfLife") or 6586.2)               # 18F
    dose = dose * 2 ** (-start_s / half) if corr == "START" and np.isfinite(start_s) else dose
    return float(mean / (dose / (weight * 1000)))


# ------------------------------------------------------------------------------------------ driver
def process_scan(patno, series_rows, fastsurfer_dir, work_dir, keep_nifti=False, sampling_seed=ITK_SEED, tracer="av133"):
    """One participant-day: convert, average, register and quantify. Returns a flat row for ``TRACER_CONFIG[tracer]["csv"]``."""
    import SimpleITK as sitk

    from .archives import ZipArchiveCache

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    date = series_rows[0]["date"]
    work = Path(work_dir) / str(patno) / date
    with ZipArchiveCache() as cache:
        data, frames, affine, sidecar, header = load_scan(series_rows, work, cache)
    cfg = TRACER_CONFIG[tracer]
    mean, info = mean_image(data, frames, affine, sampling_seed, cfg["from_min"], cfg["window"])
    pet = nib.Nifti1Image(mean, affine)
    mri = Path(fastsurfer_dir) / "mri"
    t1, aparc = nib.load(mri / "orig.mgz"), nib.load(mri / "aparc.DKTatlas+aseg.deep.mgz")
    q, tx = quantify(pet, t1, aparc, sampling_seed) if tracer == "av133" else quantify_suvr(pet, t1, aparc, cfg["reference"], sampling_seed)
    tracer_name = str(sidecar.get("TracerName", ""))
    desc = ";".join(sorted({str(r["desc"]) for r in series_rows}))
    row = {"patno": patno, "image_id": ";".join(r["image_id"] for r in series_rows), "acquisition_date": date,
           "series_desc": desc, "n_series": len(series_rows), "postfilter": series_rows[0].get("postfilter", ""),
           "tracer": tracer_name, "tracer_av133": bool(re.search(AV133, tracer_name + " " + desc, re.I)),
           "manufacturer": str(sidecar.get("Manufacturer", "")), "model": str(sidecar.get("ManufacturersModelName", "")),
           "recon": str(sidecar.get("ReconMethodName") or sidecar.get("ReconstructionMethod") or ""),
           "units": str(sidecar.get("Units", "")), "units_ok": str(sidecar.get("Units", "")).lower() in RATIO_UNITS,
           "decay_correction": str(sidecar.get("DecayCorrection", "")), "injected_mbq": sidecar.get("InjectedRadioactivity", np.nan),
           "shape": "x".join(map(str, mean.shape)), "voxel_mm": "x".join(str(round(float(z), 2)) for z in pet.header.get_zooms()[:3]),
           **info, **q, "fs_image_id": Path(fastsurfer_dir).name, "processing_version": PROCESSING_VERSION, "pie_tracer": tracer}
    ref_mean = q["mean_occipital"] if tracer == "av133" else q["mean_reference"]
    row["suv_reference"] = _suv(ref_mean, sidecar, header, float(frames["start_s"].iloc[0]))
    if keep_nifti:
        nib.save(pet, work / "pet_mean.nii.gz")
        (work / "pet.json").write_text(json.dumps(sidecar, indent=1, default=str))
        sitk.WriteTransform(tx, str(work / "t1_to_pet.tfm"))          # T1 point -> PET point
    else:
        shutil.rmtree(work, ignore_errors=True)
    return row


def _job(args):
    patno, rows, fs_dir, work_dir, keep, tracer = args
    try:
        out = process_scan(patno, rows, fs_dir, work_dir, keep_nifti=keep, tracer=tracer)
        out["error"] = ""
    except Exception as e:
        out = {"patno": patno, "image_id": ";".join(r["image_id"] for r in rows), "acquisition_date": rows[0]["date"],
               "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


# ------------------------------------------------------------------------------------------ QC / agreement
def qc_pass(d, reference="occipital"):
    """Suggested QC rule (thresholds to confirm on the first real batch): no error, an activity or count scale, registration
    correlation with the synthetic template >= 0.4 (the DaTscan rule), the striatum and the reference region inside the
    PET field of view (>= 0.99 / 0.95), inter-frame motion <= 5 mm (single-frame scans have none to measure) and every
    PPMI-comparable striatal region measured. ``reference='cerebellum'`` checks cerebellar coverage for ``sbrcb_*``."""
    col = lambda c, fill: d[c] if c in d else pd.Series(fill, index=d.index)
    num = lambda c: pd.to_numeric(col(c, np.nan), errors="coerce")
    sbr = d.reindex(columns=[f"sbr_{r}" for r in REGIONS]).apply(pd.to_numeric, errors="coerce")
    return (col("error", "").fillna("").eq("") & col("units_ok", False).astype(str).str.lower().eq("true")
            & (num("reg_template_r") >= 0.4) & (num("cov_striatum") >= 0.99) & (num(f"cov_{reference}") >= 0.95)
            & ~(num("motion_mm") > 5) & sbr.notna().all(axis=1))


def qc_pass_suvr(d):
    """SV2A / tau rows: no error, an activity or count scale, >= 95 % of the reference region in the field of view, at
    least 50 reference voxels, inter-frame motion <= 5 mm (single-frame scans pass)."""
    col = lambda c, fill: d[c] if c in d else pd.Series(fill, index=d.index)
    num = lambda c: pd.to_numeric(col(c, np.nan), errors="coerce")
    return (col("error", "").fillna("").eq("") & col("units_ok", False).astype(str).str.lower().eq("true")
            & (num("cov_reference") >= 0.95) & (num("n_reference") >= 50) & ~(num("motion_mm") > 5))


def agreement_table_suvr(pie, labels, qc=True):
    """PIE's ``suvr_<region>`` against PPMI's region of the same name in ``SUVR_REGIONS`` (``labels.suvr_labels``), on
    the same scans (PATNO + scan month): one row per region present in both."""
    p = pie[pie.get("error", pd.Series("", index=pie.index)).fillna("") == ""].copy()
    p = p[qc_pass_suvr(p)] if qc else p
    p = p.assign(PATNO=p["patno"].astype(int), month=pd.to_datetime(p["acquisition_date"]).dt.to_period("M"))
    j = p.merge(labels.assign(month=labels["PET_SCAN_DATE"].dt.to_period("M")), on=["PATNO", "month"])
    return pd.DataFrame([{"region": name, "ppmi_region": ppmi, **agreement(j[f"suvr_{name}"], j[ppmi])}
                         for name, (_, ppmi) in SUVR_REGIONS.items() if ppmi in j and f"suvr_{name}" in j])


def agreement_table(pie, labels, qc=True):
    """PIE (``pet_av133.csv``) against PPMI (``labels.pet_labels``) on the same scans (PATNO + scan month), AV-133 only,
    one row per PPMI pipeline x reference x region: the pipelines are never pooled. References: occipital (every
    pipeline), cerebellum (``sbrcb_*`` against PPMI values re-referenced from the sub-regional rows), and the
    cerebellum / occipital ratio itself (``cb_occ`` vs PET_CBM). ``qc`` keeps PIE rows passing ``qc_pass``
    (cerebellar coverage additionally for the cerebellum rows)."""
    p = pie[pie.get("error", pd.Series("", index=pie.index)).fillna("") == ""].copy()
    p["occ_ok"], p["cb_ok"] = (qc_pass(p), qc_pass(p, "cerebellum")) if qc else (True, True)
    p = p.assign(PATNO=p["patno"].astype(int), month=pd.to_datetime(p["acquisition_date"]).dt.to_period("M"))
    lab = labels[labels["tracer"].eq("AV-133") & labels["analysed"]]
    j = p.merge(lab.assign(month=lab["PET_SCAN_DATE"].dt.to_period("M")), on=["PATNO", "month"], suffixes=("_pie", "_ppmi"))
    rows = []
    for pipeline, g in j.groupby("pipeline"):
        pairs = [("occipital", r, f"sbr_{r}", "occ_ok") for r in REGIONS] + [("cerebellum", r, f"sbrcb_{r}", "cb_ok") for r in REGIONS]
        for ref, region, c, ok in pairs + [("occipital", "cerebellum", "cb_occ", "cb_ok")]:
            if g[f"{c}_ppmi"].notna().any():
                h = g[g[ok].astype(bool)]
                rows.append({"pipeline": pipeline, "reference": ref, "region": region, **agreement(h[f"{c}_pie"], h[f"{c}_ppmi"])})
    return pd.DataFrame(rows)


def main(argv=None):
    import argparse

    from .batch import add_common_args, done_subjects, fastsurfer_by_patno, filter_jobs, run_batch
    from .labels import pet_labels

    ap = add_common_args(argparse.ArgumentParser(description="PET: [18F]AV-133 striatal binding ratios; SV2A and PI-2620 tau SUVRs"))
    ap.add_argument("--tracer", choices=list(TRACER_CONFIG), default="av133", help="which PPMI PET tracer the zips hold")
    ap.add_argument("--recon", choices=["unfiltered", "filtered"], default="unfiltered",
                    help="post-filter class preferred when a visit has both reconstructions (the TOM asks for unfiltered)")
    ap.add_argument("--agreement", metavar="PPMI_DIR", help="write pet_agreement.csv (PIE vs PET_SBR_Analysis per pipeline, "
                    "reference and region) from the finished pet_av133.csv, then exit")
    a = ap.parse_args(argv)
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_csv = work / TRACER_CONFIG[a.tracer]["csv"]
    if a.agreement:
        from .labels import suvr_labels
        pie = pd.read_csv(out_csv, dtype={"image_id": str})
        t = agreement_table(pie, pet_labels(a.agreement)) if a.tracer == "av133" else agreement_table_suvr(pie, suvr_labels(a.agreement, a.tracer))
        t.to_csv(work / ("pet_agreement.csv" if a.tracer == "av133" else f"pet_agreement_{a.tracer}.csv"), index=False)
        print(t.round(3).to_string(index=False))
        return
    idx = select_series(index_pet(a.zips, work / "pet_index.csv"), a.recon, a.tracer)
    idx.to_csv(work / f"pet_index_{a.tracer}.csv", index=False)      # the choice of series for this tracer
    fs = fastsurfer_by_patno(a.sessions, a.fastsurfer_dir)
    done = done_subjects(out_csv, a.retry_errors, key="image_id")
    jobs = []
    for (patno, _), g in idx[idx["selected"]].groupby(["patno", "date"]):
        rows = g.sort_values("image_id").to_dict("records")
        if int(patno) in fs and ";".join(r["image_id"] for r in rows) not in done:
            jobs.append((int(patno), rows, fs[int(patno)], str(work), a.keep_nifti, a.tracer))
    print(f"{idx[idx['selected']].groupby(['patno', 'date']).ngroups} PET visits selected, {len(done)} done", flush=True)
    run_batch(filter_jobs(jobs, a.patnos, a.limit), _job, out_csv, workers=a.workers, pid_file=a.pid_file)


if __name__ == "__main__":
    main()
