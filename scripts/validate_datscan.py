"""Agreement of PIE's DaTscan SBRs with a PPMI core lab, cross-fitted.

    venv_imaging/bin/python scripts/validate_datscan.py --sbr Imaging/derived/datscan_2026_10/datscan_sbr.csv \
        --index Imaging/derived/spect_index.csv --ppmi-dir PPMI [--source xing] [--method deming] [--folds 5]

QC-passing PIE scans (manifest.QC["dat"]) are matched to the core lab's on PATNO and scan month. PIE's SBRs (``sbrwm_*``
for Xing's white-matter reference, ``sbr_*`` for Invicro's occipital one) are mapped onto the core lab's scale with
``datscan.calibrate`` per camera model, fitted on the other folds (participants split by PATNO), so no scan calibrates
itself. Per region: Pearson r of the raw values, overall and split by whether the hottest placement reached or passed the
old +-2-voxel window edge; CCC, ICC(2,1) and Bland-Altman bias with 95 % limits of the calibrated values. Then the deficit
call (lowest putamen <= ``--threshold`` x the age/sex expectation of the matched visually-negative controls) as Cohen's
kappa between PIE and the core lab, in prodromals and in all scans. Prints aggregates only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pie.imaging import datscan, labels, manifest
from pie.stats.agreement import ccc, icc21 as _icc21

REGIONS = [f"{r}_{s}" for r in ("putamen", "caudate") for s in "lr"]


def icc21(a, b):
    return _icc21(np.column_stack([a, b]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sbr", required=True, help="datscan_sbr.csv")
    ap.add_argument("--index", required=True, help="spect_index.csv (scan dates from the member paths)")
    ap.add_argument("--ppmi-dir", required=True, help="PPMI download (core-lab SBR and visual-read tables, subject characteristics)")
    ap.add_argument("--source", choices=["xing", "invicro"], default="xing")
    ap.add_argument("--method", choices=["deming", "equipercentile"], default="deming")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--patnos", help="optional text file of PATNOs: compare two runs on the same participants")
    ap.add_argument("--image-ids", help="optional text file of image IDs: compare two runs on the same scans")
    ap.add_argument("--qc", choices=["manifest", "basic"], default="manifest", help="manifest: QC['dat']; basic: only a "
                    "finished registration (finite metric) with >= 100 label voxels, for like-for-like comparisons of two runs")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="deficit cut-off on the core lab's percent-of-expected (0.65 was derived on Invicro's scale; "
                         "labels.xing_deficit_threshold gives Xing's equivalent)")
    a = ap.parse_args(argv)

    d = pd.read_csv(a.sbr, low_memory=False, dtype={"image_id": str})
    d = d[d["error"].fillna("") == ""] if "error" in d else d
    d = d[d["patno"].isin({int(x) for x in Path(a.patnos).read_text().split()})] if a.patnos else d
    d = d[d["image_id"].isin(set(Path(a.image_ids).read_text().split()))] if a.image_ids else d
    col = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
    if "shift_x_mm" in d:      # per-axis placement (tables since October 2026); older tables only carry the norm
        old_edge = d[["shift_x_mm", "shift_y_mm", "shift_z_mm"]].abs().max(axis=1) / col("hdr_spacing_mm").fillna(3.9) >= 2 - 1e-6
    else:
        old_edge = col("shift_vox") >= 2 - 1e-6
    qc = manifest.QC["dat"](d) if a.qc == "manifest" else (np.isfinite(pd.to_numeric(d["reg_metric"], errors="coerce")) & (d["n_label_voxels"] >= 100))
    sf = d.get("hdr_scale_fit", pd.Series(False, index=d.index)).astype(str).str.lower().eq("true")
    versor = np.array([[float(v) for v in s.split()[:3]] if isinstance(s, str) else [np.nan] * 3
                       for s in d.get("reg_params", pd.Series(np.nan, index=d.index))]).reshape(-1, 3)
    rot = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(versor, axis=1), 0, 1)))     # older tables lack reg_rot_deg
    edge_now = f"{col('shift_edge').eq(1).mean():.1%}" if "shift_edge" in d else "n/a (older table)"
    print(f"PIE scans {len(d)}, QC pass {int(qc.sum())}; placement at or past the old +-2-voxel edge {old_edge.mean():.1%}, "
          f"on the current window edge {edge_now}; rotation > 30 deg {(rot > 30).mean():.1%}; "
          f"scale clamped {int(col('reg_scale_at_bound').eq(1).sum())} of {int(sf.sum())} scale-fit scans")
    for ax in "xyz":
        print(f"  scale-fit reg_scale_{ax} 5/50/95 %:", col(f"reg_scale_{ax}")[sf].quantile([0.05, 0.5, 0.95]).round(3).tolist())

    idx = pd.read_csv(a.index, dtype={"image_id": str}).drop_duplicates("image_id").set_index("image_id")
    month = d["image_id"].map(idx["member"].str.split("/").str[3].str[:7])
    pre = "sbrwm" if a.source == "xing" else "sbr"
    pie = pd.DataFrame({"PATNO": d["patno"].astype(int), "DATSCAN_DATE": pd.to_datetime(month, format="%Y-%m", errors="coerce"),
                        "hdr_model": d.get("hdr_model"), "old_edge": old_edge, "metric": -col("reg_metric"),
                        **{f"pie_{r}": col(f"{pre}_{r}") for r in REGIONS}})[qc]
    ref = labels.dat_sbr_scored(a.ppmi_dir, a.source).drop_duplicates(["PATNO", "DATSCAN_DATE"])
    ref = ref.assign(**{f"ref_{r}_{s}": ref[f"DATSCAN_{r.upper()}_{s.upper()}"] for r in ("putamen", "caudate") for s in "lr"})
    m = pie.merge(ref, on=["PATNO", "DATSCAN_DATE"]).reset_index(drop=True)
    print(f"matched to {a.source}: {len(m)} scans of {m['PATNO'].nunique()} participants; calibration {a.method}, {a.folds}-fold by PATNO")

    fold = m["PATNO"] % a.folds
    for r in REGIONS:
        m[f"cal_{r}"] = np.nan
        for f in range(a.folds):
            held = fold == f
            train = m.assign(**{f"ref_{r}": m[f"ref_{r}"].where(~held)})          # held-out targets are invisible to the fit
            m.loc[held, f"cal_{r}"] = datscan.calibrate(train, {f"pie_{r}": f"ref_{r}"}, method=a.method)[f"pie_{r}_cal"][held]
    for k in ("pie", "ref", "cal"):
        m[f"{k}_putamen_min"] = m[[f"{k}_putamen_l", f"{k}_putamen_r"]].min(axis=1)

    out = []
    for r in REGIONS + ["putamen_min"]:
        g = m.dropna(subset=[f"pie_{r}", f"ref_{r}", f"cal_{r}"])
        x, y, c = (g[f"{k}_{r}"].to_numpy(float) for k in ("pie", "ref", "cal"))
        rr = lambda s: np.corrcoef(x[s], y[s])[0, 1] if s.sum() > 2 else np.nan
        edge = g["old_edge"].to_numpy(bool)
        diff = c - y
        out.append({"region": r, "n": len(g), "r_raw": rr(np.ones_like(edge)), "n_inside": int((~edge).sum()), "r_inside": rr(~edge),
                    "r_edge": rr(edge), "ccc": ccc(c, y), "icc21": icc21(c, y), "bias": diff.mean(),
                    "loa_low": diff.mean() - 1.96 * diff.std(ddof=1), "loa_high": diff.mean() + 1.96 * diff.std(ddof=1)})
    out = pd.DataFrame(out)
    print(out.round(3).to_string(index=False))
    g = m.dropna(subset=["pie_putamen_min", "ref_putamen_min", "metric"])
    tert = pd.qcut(g["metric"].rank(method="first"), 3, labels=["low", "mid", "high"])     # ranks: ties (synthetic data) still split
    print("putamen_min raw r by registration-metric tertile:", {str(t): round(float(np.corrcoef(x["pie_putamen_min"], x["ref_putamen_min"])[0, 1]), 3)
                                                                 for t, x in g.groupby(tert, observed=True)},
          "metric cut points", g["metric"].quantile([1 / 3, 2 / 3]).round(3).tolist())

    m["pie_pct"], _ = labels.percent_expected(m, "cal_putamen_min")
    m["ref_pct"], _ = labels.percent_expected(m, "ref_putamen_min")
    for name, g in (("prodromal", m[m["COHORT"] == "Prodromal"]), ("all", m)):
        g = g.dropna(subset=["pie_pct", "ref_pct"])
        if g.empty:
            print(f"deficit call, {name}: no matched scans")
            continue
        p, q = g["pie_pct"] <= a.threshold, g["ref_pct"] <= a.threshold
        print(f"deficit call (<= {a.threshold:.2f} of expected), {name}: n {len(g)}, PIE {p.mean():.1%}, {a.source} {q.mean():.1%}, "
              f"kappa {labels.kappa(p, q):.3f}")
        out.attrs[f"kappa_{name}"] = labels.kappa(p, q)
    return out


if __name__ == "__main__":
    main()
