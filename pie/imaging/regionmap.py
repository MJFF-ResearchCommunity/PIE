"""Regional results painted onto a FastSurfer label atlas, on the atlas's own grid: a statistic map registered to its
reference T1 by construction (no warping), plus a sidecar with the statistic, threshold, colour limits and source hash.

    python -m pie.imaging.regionmap --table results.csv --value d [--p p --alpha 0.05] --fastsurfer-dir <subject> \
        --out <dir>/<name> [--statistic "Cohen's d (PD - HC)" --units SD] [--manifest viewer.json --reference-id <scan>]
"""
import hashlib
import re

import nibabel as nib
import numpy as np
import pandas as pd

PREFIX = re.compile(r"^(vol|cth|area)_")


def normalise(name):
    """FreeSurfer label or PIE feature name -> one comparable key ('Left-Putamen', 'vol_Left_Putamen' -> 'left_putamen')."""
    m = PREFIX.match(name)
    stem = name[m.end():] if m else name
    if m and m.group(1) in ("cth", "area"):           # cth_lh_precentral -> ctx_lh_precentral
        stem = "ctx_" + stem
    return re.sub(r"[-_ ]+", "_", stem).lower()


def match_features(features, lut):
    labels = {normalise(n): int(i) for i, n in zip(lut["ID"], lut["LabelName"]) if int(i) != 0}
    got = {f: labels[normalise(f)] for f in features if normalise(f) in labels}
    return got, [f for f in features if f not in got]


def paint(atlas_img, values):
    a = np.asarray(atlas_img.dataobj).astype(np.int32)
    out = np.zeros(a.shape, np.float32)
    for label, v in values.items():
        if np.isfinite(v):
            out[a == label] = v
    return nib.Nifti1Image(out, atlas_img.affine)


def region_map(table, atlas_path, value, feature="feature", p=None, alpha=None, statistic="", units=""):
    from .viewer.catalog import DKT_LUT

    atlas = nib.load(atlas_path)
    got, unmatched = match_features(table[feature].astype(str).tolist(), pd.read_csv(DKT_LUT, sep="\t"))
    rows = table[table[feature].isin(got)]
    keep = rows if p is None or alpha is None else rows[pd.to_numeric(rows[p], errors="coerce") < alpha]
    vals = {got[f]: float(v) for f, v in zip(keep[feature], pd.to_numeric(keep[value], errors="coerce"))}
    shown = [v for v in vals.values() if np.isfinite(v)]
    side = {"statistic": statistic or value, "units": units, "value_column": value,
            "threshold": None if p is None or alpha is None else {"p_column": p, "alpha": alpha},
            "vmax": float(max(map(abs, shown))) if shown else 0.0, "n_matched": len(got), "n_shown": len(shown),
            "unmatched": unmatched, "atlas": str(atlas_path),
            "source_sha256": hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()}
    return paint(atlas, vals), side
