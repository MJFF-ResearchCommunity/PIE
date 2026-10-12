"""Stage 1 of the staged NM validation (documentation/preregistration_nm_staged_validation.md): split-half reliability
of each candidate neuromelanin measure on PPMI Siemens 2D GRE-MT, and the outcome-blind choice of the primary measure.
Reads no cohort label, diagnosis, clinical or DaT value.

Each session's repeats 1 + 3 and 2 + 4 are averaged separately (``nm --split-half``) and measured like the full slab
(``nm_native --image nm_half_{a,b}.nii.gz [--n4]``). Reliability is ICC(A,1) between the halves, projected to the
5-repeat average by Spearman-Brown (m = 2.5).

    PYTHONPATH=. python scripts/nm_split_half.py --nm-dir <work dir> --out <dir>
    PYTHONPATH=. python scripts/nm_split_half.py --synthetic --out <dir>      # dry run on fake values
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED, B, M = 20261012, 2000, 2.5
CURRENT = "nms_sn_mean_cr"                  # the default before this stage (preregistration_nm_snceg.md)
MEASURES = ["nms_sn_mean_cr", "nms_sn_mean_cr_mode", "nmf_sn_mean_cr", "nmf_sn_mean_cr_mode"]
VARIANTS = [(m, n4) for n4 in (False, True) for m in MEASURES]      # order = tie-break preference after CURRENT
GATE_ICC, GATE_LO, TIE = 0.75, 0.50, 0.02


def name(v):
    return v[0] + ("_n4" if v[1] else "")


def icc_a1(x, y):
    """ICC(A,1), two-way, absolute agreement, single measurement (McGraw & Wong 1996), for n subjects x 2 halves."""
    d = np.column_stack([x, y]).astype(float)
    n, k = d.shape
    grand = d.mean()
    msr = k * ((d.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((d.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    mse = ((d - d.mean(axis=1, keepdims=True) - d.mean(axis=0, keepdims=True) + grand) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def spearman_brown(r, m=M):
    return m * r / (1 + (m - 1) * r)


def load(nm_dir):
    """Siemens QC-passing full slabs with both halves; per-variant values on half a, half b and the full slab."""
    from pie.imaging.manifest import QC, _vendor

    nm_dir = Path(nm_dir)
    nm = pd.read_csv(nm_dir / "nm_features.csv", low_memory=False)
    nm = nm[nm["error"].fillna("") == ""]
    sh = pd.read_csv(nm_dir / "nm_split_half.csv")
    flow = {"slabs": len(nm)}
    dims = lambda c: nm[c].astype(str).str.split("x", expand=True).apply(pd.to_numeric, errors="coerce")
    steps = [("qc_nm", QC["nm"](nm)), ("siemens", _vendor(nm["manufacturer"]).eq("SIEMENS")),
             ("slab_ge_24mm", (dims("shape") * dims("voxel_mm")).min(axis=1) >= 24 - 1e-6),
             ("halves_made", nm["patno"].isin(sh.loc[sh["error"].fillna("") == "", "patno"]))]
    keep = pd.Series(True, index=nm.index)
    for step, ok in steps:
        flow[f"excluded_{step}"] = int((keep & ~ok.fillna(False)).sum())
        keep &= ok.fillna(False)
    base = nm.loc[keep.to_numpy(), ["patno", "model"]]
    tables = {}
    for img in ("", "_nm_half_a", "_nm_half_b"):
        for n4 in ("", "_n4"):
            t = pd.read_csv(nm_dir / f"nm_native_features{img}{n4}.csv", low_memory=False)
            t = t[t["error"].fillna("") == ""]
            tables[(img or "_full", bool(n4))] = t.set_index("patno")
    d = base.set_index("patno")
    for v in VARIANTS:
        for img in ("_full", "_nm_half_a", "_nm_half_b"):
            d[f"{name(v)}{img}"] = tables[(img, v[1])][v[0]].reindex(d.index)
    return d.reset_index(), flow


def synthetic(n=420):
    """Fake values only: every variant = a subject value + half noise of a variant-specific size."""
    rng = np.random.default_rng(3)
    true = rng.normal(0.25, 0.05, n)
    d = pd.DataFrame({"patno": np.arange(n), "model": rng.choice(["A", "B", "C"], n)})
    for i, v in enumerate(VARIANTS):
        e = 0.02 + 0.005 * i
        for img, s in (("_full", e / np.sqrt(M)), ("_nm_half_a", e), ("_nm_half_b", e)):
            d[f"{name(v)}{img}"] = true + rng.normal(0, s, n)
    return d, {"synthetic": True}


def reliability(d, cols_a, cols_b, n_boot=B):
    x, y = d[cols_a].to_numpy(float), d[cols_b].to_numpy(float)
    r = icc_a1(x, y)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boot.append(spearman_brown(icc_a1(x[i], y[i])))
    return {"n": int(len(x)), "icc_half": r, "icc_projected": spearman_brown(r), "ci95": list(np.percentile(boot, [2.5, 97.5]))}


def choose(rel):
    """Highest projected ICC; the current default is kept unless another variant beats it by more than TIE; otherwise,
    among the variants within TIE of the best, the earliest in VARIANTS wins (closest to the current default first,
    without N4 before with)."""
    best = max(r["icc_projected"] for r in rel.values())
    cur = rel[CURRENT]["icc_projected"]
    if cur >= best - TIE:
        return CURRENT
    return next(name(v) for v in VARIANTS if rel[name(v)]["icc_projected"] >= best - TIE)


def eta2(y, groups):
    ok = np.isfinite(y)
    y, g = y[ok], np.asarray(groups)[ok]
    ss_b = sum(len(y[g == k]) * (y[g == k].mean() - y.mean()) ** 2 for k in np.unique(g))
    return float(ss_b / ((y - y.mean()) ** 2).sum())


def analyse(d, flow, n_boot=B):
    cols = [name(v) for v in VARIANTS]
    finite = np.all([np.isfinite(d[f"{c}{img}"]) for c in cols for img in ("_nm_half_a", "_nm_half_b")], axis=0)
    res = {"flow": {**flow, "excluded_any_variant_missing_on_a_half": int((~finite).sum()), "analysed": int(finite.sum())},
           "missing_rate": {c: float((~(np.isfinite(d[f"{c}_nm_half_a"]) & np.isfinite(d[f"{c}_nm_half_b"]))).mean()) for c in cols}}
    common = d[finite]
    res["reliability"] = {c: reliability(common, f"{c}_nm_half_a", f"{c}_nm_half_b", n_boot) for c in cols}
    res["reliability_own_rows"] = {c: reliability(d.dropna(subset=[f"{c}_nm_half_a", f"{c}_nm_half_b"]), f"{c}_nm_half_a", f"{c}_nm_half_b", 200)["icc_projected"] for c in cols}
    primary = choose(res["reliability"])
    r = res["reliability"][primary]
    res["primary"] = primary
    res["technical_gate"] = {"icc_projected": r["icc_projected"], "ci95_low": r["ci95"][0],
                             "pass": bool(r["icc_projected"] >= GATE_ICC and r["ci95"][0] > GATE_LO)}
    full = d[[f"{c}_full" for c in cols]].rename(columns=lambda c: c[:-5])
    res["agreement_full_spearman"] = full.corr(method="spearman").round(3).to_dict()
    model = d["model"].astype(str).str.upper().str.replace("MAGNETOM ", "").str.replace(" ", "_")
    res["scanner_model_eta2_full"] = {c: eta2(d[f"{c}_full"].to_numpy(float), model) for c in cols}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nm-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args(argv)
    d, flow = synthetic() if a.synthetic else load(a.nm_dir)
    res = analyse(d, flow, a.n_boot)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("synthetic_split_half.json" if a.synthetic else "nm_split_half_result.json")).write_text(json.dumps(res, indent=1, default=float))
    if a.synthetic:
        print(f"synthetic dry run: n {res['flow']['analysed']}; primary {res['primary']}; gate {res['technical_gate']['pass']}")
    else:
        for c, r in res["reliability"].items():
            print(f"{c:28s} ICC half {r['icc_half']:.3f}  projected {r['icc_projected']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  missing {res['missing_rate'][c]:.3f}")
        print("primary:", res["primary"], "| technical gate:", "PASS" if res["technical_gate"]["pass"] else "FAIL")
    return res


if __name__ == "__main__":
    main()
