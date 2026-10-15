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


def _diverging():
    """The Brain Viewer's scale for figures: NiiVue's cool (blue -> cyan) for negatives and warm (orange -> yellow) for
    positives, both brightest at the largest magnitude (NiiVue src/cmaps/{cool,warm}.json, BSD-2-Clause)."""
    from matplotlib.colors import LinearSegmentedColormap
    rgb = lambda r, g, b: (r / 255, g / 255, b / 255)
    stops = [(0.0, rgb(0, 254, 255)), (0.25, rgb(0, 196, 255)), (0.5, rgb(0, 127, 255)),
             (0.5, rgb(255, 127, 0)), (0.75, rgb(255, 196, 0)), (1.0, rgb(255, 254, 0))]
    return LinearSegmentedColormap.from_list("pie_warm_cool", stops)


STAT_CMAP = _diverging()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def region_map(table, atlas_path, value, feature="feature", p=None, alpha=None, statistic="", units="", source_sha256=None):
    """The map and its sidecar. ``source_sha256`` is the hash of the source file when there is one (the CLI passes it);
    otherwise the table is hashed as CSV."""
    from .viewer.catalog import DKT_LUT

    atlas = nib.load(atlas_path)
    got, unmatched = match_features(table[feature].astype(str).tolist(), pd.read_csv(DKT_LUT, sep="\t"))
    by_label = {}
    for f, label in got.items():
        by_label.setdefault(label, []).append(f)
    shared = [" / ".join(fs) for fs in by_label.values() if len(fs) > 1]
    if shared:
        raise ValueError(f"one value per region: {'; '.join(shared)} map to the same atlas label; make one map per measure")
    present = set(np.unique(np.asarray(atlas.dataobj)).astype(int).tolist())
    absent = [f for f, label in got.items() if label not in present]       # in the lookup table, not in this segmentation
    got = {f: label for f, label in got.items() if label in present}
    rows = table[table[feature].isin(got)]
    keep = rows if p is None or alpha is None else rows[pd.to_numeric(rows[p], errors="coerce") < alpha]
    vals = {got[f]: float(v) for f, v in zip(keep[feature], pd.to_numeric(keep[value], errors="coerce"))}
    shown = [v for v in vals.values() if np.isfinite(v)]
    side = {"statistic": statistic or value, "units": units, "value_column": value,
            "threshold": None if p is None or alpha is None else {"p_column": p, "alpha": alpha},
            "vmax": float(max(map(abs, shown))) if shown else 0.0, "n_matched": len(got), "n_shown": len(shown),
            "unmatched": unmatched, "absent_from_atlas": absent,
            "atlas": str(Path(atlas_path).resolve()), "atlas_sha256": _sha256(atlas_path),
            "source_sha256": source_sha256 or hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()}
    return paint(atlas, vals), side


def surface_values(map_img, atlas_img, vertex_labels, hemi):
    """Per-vertex values for a surface view: each vertex takes its parcel's value when the parcel is one of this
    hemisphere's cortical labels (ctx-lh 1000-1999, ctx-rh 2000-2999); anything else (ventricles, deep nuclei, the other
    hemisphere, background) stays 0. Whole parcels, no interpolation fringe."""
    lo = 1000 if hemi == "left" else 2000
    a, m = np.asarray(atlas_img.dataobj).astype(int), np.asarray(map_img.dataobj)
    labels = np.rint(np.asarray(vertex_labels)).astype(int)
    out = np.zeros(len(labels), np.float32)
    for label in np.unique(labels):
        inside = a == label
        if lo <= label < lo + 1000 and inside.any():
            out[labels == label] = m[inside][0]
    return out


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
    kw = {"cmap": STAT_CMAP, "vmax": vmax or 1.0, "threshold": vmax / 1000 if vmax else 1e-6, "symmetric_cbar": True}
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
            everywhere = nib.Nifti1Image(np.ones(atlas.shape, np.int8), atlas.affine)   # background counts as label 0, not "no sample"
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={"projection": "3d"})
            for ax, (hemi, view) in zip(axes, [(h, v) for h in ("left", "right") for v in ("lateral", "medial")]):
                # the parcel label most often met between pial and white surface, then that parcel's value
                labels = surface.vol_to_surf(atlas, fs[f"pial_{hemi}"], inner_mesh=fs[f"white_{hemi}"], mask_img=everywhere,
                                             interpolation="nearest_most_frequent")
                plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], surface_values(img, atlas, labels, hemi), hemi=hemi, view=view,
                                            bg_map=fs[f"sulc_{hemi}"], axes=ax, figure=fig, colorbar=False, **kw)
                ax.set_title(f"{hemi.capitalize()} {view}")
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-kw["vmax"], kw["vmax"]), cmap=STAT_CMAP),
                         ax=axes, shrink=0.6, label=side.get("units") or None)
            fig.suptitle(title)
            fig.text(0.5, 0.02, "MNI152 parcels sampled onto fsaverage (approximate); cortical labels only", ha="center")
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
    if (a.p is None) != (a.alpha is None):
        ap.error("--p and --alpha go together: a p column without a level would show every region")
    img, side = region_map(pd.read_csv(a.table), a.fastsurfer_dir / "mri" / "aparc.DKTatlas+aseg.deep.mgz", a.value, a.feature,
                           a.p, a.alpha, a.statistic, a.units, source_sha256=_sha256(a.table))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    nii, js = a.out.with_name(a.out.name + ".nii.gz"), a.out.with_name(a.out.name + ".json")
    nib.save(img, nii)
    js.write_text(json.dumps(side, indent=1))
    print(f"{nii}: {side['n_shown']} of {side['n_matched']} matched regions shown; {len(side['unmatched'])} features unmatched; "
          f"{len(side['absent_from_atlas'])} not in this segmentation")
    if a.manifest:
        if not a.reference_id:
            ap.error("--manifest needs --reference-id")
        doc = json.loads(a.manifest.read_text())
        ref = next((s for s in doc["scans"] if s["id"] == a.reference_id), None)
        if ref is not None:
            ref_path = (a.manifest.parent / ref["path"]).resolve()
            ref_atlas = (a.manifest.parent / ref["atlas"]).resolve() if ref.get("atlas") else None
        else:                                        # a T1 the viewer discovers in PIE's outputs (mri-<image id>)
            from .viewer.catalog import Catalog
            scan = Catalog(a.repo, manifest=a.manifest).scans.get(a.reference_id)
            if scan is None:
                ap.error(f"{a.reference_id} is neither in {a.manifest} nor among the scans the viewer finds in {a.repo}")
            ref, ref_path, ref_atlas = {"subject": scan.subject, "space": scan.space, "visit": scan.visit}, scan.path, scan.atlas
        from .viewer.images import same_grid
        if not same_grid(img, nib.load(ref_path)):
            ap.error(f"{a.reference_id} is not on the grid of {a.fastsurfer_dir}'s atlas; paint the map on that scan's own segmentation")
        if ref_atlas and _sha256(ref_atlas) != side["atlas_sha256"]:
            ap.error(f"{a.reference_id}'s segmentation is not {a.fastsurfer_dir}'s atlas; paint the map on that scan's own segmentation")
        sid = f"stat-{a.out.name}"
        taken = next((s for s in doc["scans"] if s["id"] == sid and s.get("reference_id") != a.reference_id), None)
        if taken:
            ap.error(f"{sid} already overlays {taken.get('reference_id')}; choose another --out name")
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
