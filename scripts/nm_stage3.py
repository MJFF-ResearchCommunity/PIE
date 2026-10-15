"""NM Stage 3 and the Stage 2b replication (documentation/preregistration_nm_staged_validation.md, Amendment 1).

Supersedes the fresh-clinical mode of nm_staged.py, which coded only the first half of the 13 Oct 2026 pass rule.
nm_staged.py itself stays byte-identical, because the frozen DWI free-water analysis imports from it.

    fresh-clinical   Stage 3: HC vs de novo PD among the --fresh lists, minus --exclude (scans used by any earlier
                     analysis). Vendor-stratified AUROC (HC > PD). Pass: 95 % CI lower bound > 0.50 AND upper bound
                     >= 0.633 (the like-for-like PPMI GRE-MT benchmark).
    prodromal-fresh  Stage 2b-R: Stage 2b repeated on prodromal participants in --fresh, minus --exclude. Vendor-stratified
                     AUROC (DaT-normal > DaT-deficient). Pass: CI lower bound > 0.50.

Descriptive secondaries (no gate): the lower of the two sides' contrast and the snceg nigral volume.

    PYTHONPATH=. python scripts/nm_stage3.py <analysis> --nm-dir <work> --stage1 <json> --fresh <list> [...] \
        --exclude <list> [...] --ppmi PPMI --out <dir>
    PYTHONPATH=. python scripts/nm_stage3.py <analysis> --synthetic --out <dir>      # dry run: every value and label fake
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nm_staged import (B, auc_with_ci, fresh_clinical, label_fresh, label_prodromal_dat, load,  # noqa: E402
                       primary_of, prodromal_dat, synthetic)

BENCHMARK = 0.633     # Langley, Hwang, Huddleston & Hu, ISMRM 2023 abstract 3195: PPMI Siemens GRE-MT, de novo PD vs controls
SECONDARY = {"lower_side_cr": ["nms_sn_l_cr", "nms_sn_r_cr"], "sn_volume_mm3": ["nms_sn_volume_mm3"]}


def stage3_pass(ci95):
    """Both halves of the 13 Oct rule: above chance, and not shown worse than the published method."""
    return bool(ci95[0] > 0.50 and ci95[1] >= BENCHMARK)


def ids(files):
    return {int(x) for f in files or [] for x in Path(f).read_text().replace(",", " ").split()}


def restrict(d, flow, fresh, exclude):
    """Rows in the fresh lists, minus every participant an earlier analysis used; both counted."""
    flow["not_in_fresh_lists"] = int((~d["patno"].isin(fresh)).sum())
    d = d[d["patno"].isin(fresh)]
    flow["excluded_used_before"] = int(d["patno"].isin(exclude).sum())
    return d[~d["patno"].isin(exclude)].copy(), flow


def add_secondary(d, nm_dir, suffix):
    """Secondary measures from the same native-feature table, on the rows the primary analysis kept."""
    nat = pd.read_csv(Path(nm_dir) / f"nm_native_features{suffix}.csv", low_memory=False)
    nat = nat[nat["error"].fillna("") == ""].drop_duplicates("patno").set_index("patno")
    for name, cols in SECONDARY.items():
        v = nat.reindex(d["patno"])[cols].apply(pd.to_numeric, errors="coerce")
        d[f"sec_{name}"] = v.min(axis=1, skipna=False).to_numpy()
    return d


def secondaries(pos, neg):
    out = {}
    for name in SECONDARY:
        c = f"sec_{name}"
        p, n = pos.dropna(subset=[c]), neg.dropna(subset=[c])
        out[name] = auc_with_ci(p[c], n[c], p["vendor"], n["vendor"], B) if len(p) and len(n) else None
    return out


def run(analysis, d, flow):
    if analysis == "fresh-clinical":
        res = fresh_clinical(d, flow)
        res["pass"] = stage3_pass(res["primary"]["ci95"])
        pos, neg = d[d["group"].eq("HC")], d[d["group"].eq("PD_denovo")]
    else:
        res = prodromal_dat(d, flow)
        x = d.dropna(subset=["dat_d_nsdiss"])
        pos, neg = x[x["dat_d_nsdiss"].eq(0)], x[x["dat_d_nsdiss"].eq(1)]
    res["pass_rule"] = (f"CI lower > 0.50 and CI upper >= {BENCHMARK}" if analysis == "fresh-clinical" else "CI lower > 0.50")
    res["secondary"]["descriptive"] = secondaries(pos, neg)
    return res


def fake(analysis):
    """Fully synthetic rows (nm_staged.synthetic plus fake secondaries): no real value, label or identifier."""
    d, flow = synthetic("fresh-clinical" if analysis == "fresh-clinical" else "prodromal-dat")
    rng = np.random.default_rng(15)
    for name in SECONDARY:
        d[f"sec_{name}"] = d["y"] + rng.normal(0, 0.05, len(d))
    return d, flow


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("analysis", choices=["fresh-clinical", "prodromal-fresh"])
    ap.add_argument("--nm-dir")
    ap.add_argument("--stage1")
    ap.add_argument("--fresh", nargs="+", help="PATNO lists written before the download (comma- or whitespace-separated)")
    ap.add_argument("--exclude", nargs="+", help="PATNO lists of every participant whose NM an earlier analysis used")
    ap.add_argument("--ppmi", default="PPMI")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(argv)
    if a.synthetic:
        d, flow = fake(a.analysis)
    else:
        column, suffix = primary_of(a.stage1)
        d, flow = load(a.nm_dir, column, suffix)
        flow["primary_measure"] = column + suffix
        d, flow = restrict(d, flow, ids(a.fresh), ids(a.exclude))
        d = add_secondary(d, a.nm_dir, suffix)
        d = label_fresh(d, a.ppmi, ids(a.fresh)) if a.analysis == "fresh-clinical" else label_prodromal_dat(d, a.ppmi)
    res = run(a.analysis, d, flow)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = ("synthetic_" if a.synthetic else "") + a.analysis.replace("-", "_")
    (out / f"{tag}_result.json").write_text(json.dumps(res, indent=1, default=float))
    p = res["primary"]
    if a.synthetic:
        print(f"synthetic dry run ({a.analysis}): n {p['n_pos']} / {p['n_neg']}; pass={res['pass']}")
    else:
        print(f"{a.analysis}: AUROC {p['auroc']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] (n {p['n_pos']} / {p['n_neg']}) "
              f"-> {'PASS' if res['pass'] else 'FAIL'} ({res['pass_rule']})")
    return res


if __name__ == "__main__":
    main()
