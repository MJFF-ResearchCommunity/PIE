"""Descriptive native-EPI summaries; never activation or connectivity estimates."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def summarize_bold(data):
    """All frames, population temporal SD. No filtering, detrending or censoring.

    Keep a fixed, explicitly heuristic foreground for aggregate traces. This is
    not a brain segmentation or a motion estimate; all-zero/nonfinite voxels are
    excluded. Welford avoids a second full-size 4D allocation.
    """
    if data.ndim != 4 or data.shape[3] < 2:
        raise ValueError("BOLD inspection requires a 4D acquisition with at least two frames")
    mean = np.zeros(data.shape[:3], np.float64)
    m2 = np.zeros_like(mean)
    valid = np.ones(mean.shape, bool)
    for i in range(data.shape[3]):
        frame = np.asarray(data[..., i], dtype=np.float64)
        finite = np.isfinite(frame)
        valid &= finite
        frame = np.where(finite, frame, 0)
        delta = frame - mean
        mean += delta / (i + 1)
        m2 += delta * (frame - mean)
    sd = np.sqrt(np.maximum(0, m2 / data.shape[3]))
    positive = mean[valid & (mean > 0)]
    threshold = 0.2 * float(np.percentile(positive, 95)) if positive.size else 0.
    foreground = valid & (mean > threshold)
    labels, count = ndimage.label(foreground)
    if count:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        foreground = labels == int(sizes.argmax())
    tsnr = np.divide(mean, sd, out=np.zeros_like(mean), where=foreground & (sd > np.finfo(np.float32).eps))
    traces, dvars = [], [None]
    previous = None
    for i in range(data.shape[3]):
        values = np.asarray(data[..., i][foreground], dtype=np.float64)
        traces.append(float(values.mean()) if values.size else None)
        if i:
            dvars.append(float(np.sqrt(np.mean((values - previous) ** 2))) if values.size else None)
        previous = values
    return {
        "mean": np.where(valid, mean, 0).astype(np.float32),
        "sd": np.where(valid, sd, 0).astype(np.float32),
        "tsnr": tsnr.astype(np.float32),
    }, {"mean_signal": traces, "raw_dvars": dvars, "foreground_voxels": int(foreground.sum()),
        "excluded_nonfinite_voxels": int((~valid).sum()), "foreground_threshold": threshold,
        "foreground_definition": "Largest connected component with temporal mean above 20% of the positive-mean 95th percentile; finite at every frame. Intensity-defined foreground, not an anatomical brain mask.",
        "method": "All acquired frames; arithmetic temporal mean; population temporal SD (ddof=0); tSNR = mean/SD within the fixed foreground (zero-SD voxels set to 0). Raw DVARS = RMS successive-frame intensity differences within that same foreground. No standardization, detrending, censoring, motion correction, or denoising.",
        "warning": "Descriptive unprocessed signal only. These traces and maps do not establish activation, connectivity, head motion, disease, or a QC pass/fail."}


def add_fmri_summary(store, scan, prepared, folder: Path):
    from .images import volume_info
    image = nib.load(scan.path)
    if len(image.shape) != 4 or image.shape[3] < 2:
        return
    # Limit materialization to bounded examples; raw volumes remain available if
    # a much larger imported run needs an offline/chunked preprocessing workflow.
    if int(np.prod(image.shape)) * 4 > 1_500_000_000:
        prepared["fmri_unavailable"] = "This run exceeds the 1.5 GB summary memory budget. Prepare summaries offline."
        return
    summary_path = folder / "bold_summary_v1.json"
    paths = {key: folder / f"bold_{key}_v1.nii.gz" for key in ("mean", "sd", "tsnr")}
    if not summary_path.exists() or not all(p.exists() for p in paths.values()):
        maps, summary = summarize_bold(image.get_fdata(dtype=np.float32, caching="unchanged"))
        for key, data in maps.items():
            out = nib.Nifti1Image(data, image.affine)
            out.set_qform(image.affine, 1); out.set_sform(image.affine, 1)
            out.header.set_xyzt_units(image.header.get_xyzt_units()[0])
            temporary = paths[key].with_name(f"bold_{key}_v1.tmp.nii.gz")
            nib.save(out, temporary); temporary.replace(paths[key])
        temporary = summary_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(summary, allow_nan=False)); temporary.replace(summary_path)
    summary = json.loads(summary_path.read_text())
    summary["frames"] = image.shape[3]
    factor = {"sec": 1., "msec": 0.001, "usec": 0.000001}.get(image.header.get_xyzt_units()[1])
    step = float(image.header.get_zooms()[3])
    summary["tr_seconds"] = step * factor if factor and np.isfinite(step) and step > 0 else None
    summary["source_fingerprint"] = prepared["fingerprint"]
    prepared["fmri"] = summary
    for key, name, units, cmap in (("mean", "Temporal mean BOLD", scan.units, "gray"), ("sd", "Temporal standard deviation", scan.units, "viridis"), ("tsnr", "Temporal signal-to-noise ratio", "mean / temporal SD (dimensionless)", "viridis")):
        info = volume_info(paths[key])
        prepared["extra"].append({"key": key, "name": name, "units": units, "url": store.asset(paths[key]), "colormap": cmap, "cal_min": 0 if key == "tsnr" else info["cal_min"], "cal_max": info["cal_max"]})
