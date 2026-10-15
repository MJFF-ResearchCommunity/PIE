"""Regional results painted onto a FastSurfer label atlas, on the atlas's own grid: a statistic map registered to its
reference T1 by construction (no warping), plus a sidecar with the statistic, threshold, colour limits and source hash.

    python -m pie.imaging.regionmap --table results.csv --value d [--p p --alpha 0.05] --fastsurfer-dir <subject> \
        --out <dir>/<name> [--statistic "Cohen's d (PD - HC)" --units SD] [--manifest viewer.json --reference-id <scan>]
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

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
    by_label = {}
    for f, label in got.items():
        by_label.setdefault(label, []).append(f)
    shared = [" / ".join(fs) for fs in by_label.values() if len(fs) > 1]
    if shared:
        raise ValueError(f"one value per region: {'; '.join(shared)} map to the same atlas label; make one map per measure")
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


def cortex_only(map_img, atlas_img, hemi):
    """The map restricted to one hemisphere's cortical labels (FreeSurfer ctx-lh 1000-1999, ctx-rh 2000-2999), so a
    surface projection never shows ventricle or deep-nucleus values as cortex."""
    lo = 1000 if hemi == "left" else 2000
    a = np.asarray(atlas_img.dataobj)
    return nib.Nifti1Image(np.where((a >= lo) & (a < lo + 1000), np.asarray(map_img.dataobj), 0).astype(np.float32), map_img.affine)


def figures(map_path, sidecar_path, background, out_dir, template=False):
    """Publication figures from the same map and sidecar the viewer shows: ortho slices always; for template maps also a
    glass brain and an fsaverage5 surface view. SVG text stays editable. Returns the files written."""
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting, surface

    side = json.loads(Path(sidecar_path).read_text())
    vmax, thr = float(side["vmax"]), side.get("threshold")
    title = side["statistic"] + ("" if not thr else f", {thr['p_column']} < {thr['alpha']}")
    if vmax == 0:
        title += " (no region passes the threshold)"
    img = nib.load(map_path)
    # symmetric in every view: nilearn otherwise maps a one-signed map onto the whole diverging scale
    kw = {"cmap": "cold_hot", "vmax": vmax or 1.0, "threshold": vmax / 1000 if vmax else 1e-6, "symmetric_cbar": True}
    out_dir, stem = Path(out_dir), Path(map_path).name.split(".")[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(display, name, svg=True):
        for ext in ("png", "svg") if svg else ("png",):
            f = out_dir / f"{stem}_{name}.{ext}"
            display.savefig(f, dpi=300)
            written.append(f)
        if hasattr(display, "close"):        # nilearn display; a plain matplotlib figure is closed by pyplot
            display.close()
        else:
            plt.close(display)

    with plt.rc_context({"svg.fonttype": "none"}):
        save(plotting.plot_stat_map(img, bg_img=str(background), display_mode="ortho", title=title, **kw), "slices")
        if template:
            save(plotting.plot_glass_brain(img, plot_abs=False, title=title, colorbar=True, **kw), "glass")
            atlas = nib.load(side["atlas"])          # the label grid the map was painted on (sidecar)
            fs = datasets.fetch_surf_fsaverage("fsaverage5")
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={"projection": "3d"})
            for ax, (hemi, view) in zip(axes, [(h, v) for h in ("left", "right") for v in ("lateral", "medial")]):
                tex = surface.vol_to_surf(cortex_only(img, atlas, hemi), fs[f"pial_{hemi}"])
                plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], tex, hemi=hemi, view=view, bg_map=fs[f"sulc_{hemi}"],
                                            axes=ax, figure=fig, colorbar=False, **kw)
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-kw["vmax"], kw["vmax"]), cmap=plotting.cm.cold_hot),
                         ax=axes, shrink=0.6, label=side.get("units") or None)
            fig.suptitle(title)
            fig.text(0.5, 0.02, "MNI152 volume projected to fsaverage (approximate); cortical labels only", ha="center")
            save(fig, "surface", svg=False)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True, help="CSV with one row per region: feature name, value, optional p")
    ap.add_argument("--value", required=True)
    ap.add_argument("--feature", default="feature")
    ap.add_argument("--p")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--fastsurfer-dir", required=True, type=Path, help="FastSurfer subject folder whose mri/aparc.DKTatlas+aseg.deep.mgz is the canvas")
    ap.add_argument("--out", required=True, type=Path, help="output prefix: writes PREFIX.nii.gz and PREFIX.json")
    ap.add_argument("--statistic", default="")
    ap.add_argument("--units", default="")
    ap.add_argument("--manifest", type=Path, help="viewer manifest to add the map to, as a verified overlay")
    ap.add_argument("--reference-id", help="scan the atlas belongs to (required with --manifest): in the manifest, or found by the viewer in --repo")
    ap.add_argument("--repo", type=Path, default=Path("."), help="repository whose PIE outputs the viewer indexes (for --reference-id mri-<image id>)")
    a = ap.parse_args(argv)
    img, side = region_map(pd.read_csv(a.table), a.fastsurfer_dir / "mri" / "aparc.DKTatlas+aseg.deep.mgz", a.value, a.feature,
                           a.p, a.alpha, a.statistic, a.units)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    nii, js = a.out.with_name(a.out.name + ".nii.gz"), a.out.with_name(a.out.name + ".json")
    nib.save(img, nii)
    js.write_text(json.dumps(side, indent=1))
    print(f"{nii}: {side['n_shown']} of {side['n_matched']} matched regions shown; {len(side['unmatched'])} features unmatched")
    if a.manifest:
        if not a.reference_id:
            ap.error("--manifest needs --reference-id")
        doc = json.loads(a.manifest.read_text())
        ref = next((s for s in doc["scans"] if s["id"] == a.reference_id), None)
        if ref is not None:
            ref_path = (a.manifest.parent / ref["path"]).resolve()
        else:                                        # a T1 the viewer discovers in PIE's outputs (mri-<image id>)
            from .viewer.catalog import Catalog
            scan = Catalog(a.repo, manifest=a.manifest).scans.get(a.reference_id)
            if scan is None:
                ap.error(f"{a.reference_id} is neither in {a.manifest} nor among the scans the viewer finds in {a.repo}")
            ref, ref_path = {"subject": scan.subject, "space": scan.space, "visit": scan.visit}, scan.path
        from .viewer.images import same_grid
        if not same_grid(img, nib.load(ref_path)):
            ap.error(f"{a.reference_id} is not on the grid of {a.fastsurfer_dir}'s atlas; paint the map on that scan's own segmentation")
        sid = f"stat-{a.out.name}"
        rule = "all matched regions" if side["threshold"] is None else f"{a.p} < {a.alpha}"
        doc["scans"] = [s for s in doc["scans"] if s["id"] != sid] + [{
            "id": sid, "subject": ref["subject"], "modality": "Statistic", "kind": "statistic", "visit": ref.get("visit", "Imported acquisition"),
            "path": os.path.relpath(nii, a.manifest.parent), "space": ref.get("space", f"sub-{ref['subject']}:native:{a.reference_id}"), "registration": "verified",
            "reference_id": a.reference_id, "units": a.units, "description": side["statistic"], "metadata": side,
            "qc": "Regional statistic painted on the reference's own label grid",
            "provenance": f"pie.imaging.regionmap from a table with sha256 {side['source_sha256'][:12]}; shown: {rule}."}]
        a.manifest.write_text(json.dumps(doc, indent=1))
    return side


if __name__ == "__main__":
    main()
