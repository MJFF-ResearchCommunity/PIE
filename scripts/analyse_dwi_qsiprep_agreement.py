"""Pre-registered analysis (documentation/preregistration_dwi_qsiprep.md): agreement of posterior-SN free water between
PIE's own preprocessing (``dwi --eddy --study``, ``dwisf_*``) and QSIPrep 26.0.0's, with PIE's free-water fit and
region rule applied to QSIPrep's preprocessed DWI (``dwi_qsiprep compare``, ``dwiqf_*``).

    select    <out>/qsiprep_order.csv: every QC-passing PPMI 2.0 dwisf_* session in a seeded random order within its
              correction tier; the first 20 per tier are the sample. Reads identifiers, dates, errors, QC flags and
              tiers only (no measured value, no label)
    analyse   <out>/qsiprep_agreement.json: ICC(2,1) of the primary (gate >= 0.75), Lin's CCC, Bland-Altman, per measure
              and per tier

    PYTHONPATH=. python scripts/analyse_dwi_qsiprep_agreement.py select --tables <dwi_features.csv>[,...] --out <dir>
    PYTHONPATH=. python scripts/analyse_dwi_qsiprep_agreement.py analyse --tables <...> --dwiq <dwiq_features.csv> --out <dir>
    PYTHONPATH=. python scripts/analyse_dwi_qsiprep_agreement.py analyse --synthetic --out <dir>   # dry run: all values fake
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pie.stats.agreement import agreement, icc21

SEED, B, N_PER_TIER, GATE_ICC = 20261015, 2000, 20, 0.75
TIER = {"topup_eddy": "topup_eddy", "eddy_no_reverse_PE": "eddy_only"}      # dwisf_tier -> stratum; rigid tiers are not drawn
PRIMARY = "fw_sn_posterior_mean"
FW_SECONDARY = ["fw_sn_posterior_l", "fw_sn_posterior_r", "fw_sn_anterior_mean", "fw_sn_mean", "fat_sn_posterior_mean",
                "mdt_sn_posterior_mean"]                      # dwisf_<m> against dwiqf_<m>, both free-water QCs passed
TENSOR = ["fa_sn_posterior_mean", "md_sn_posterior_mean"]     # dwis_<m> against dwiq_<m>, both tensor QCs passed
SELECT_COLS = ["patno", "acquisition_date", "error", "dwisf_qc_pass", "dwisf_tier"]


def candidates(tables):
    """QC-passing, error-free PPMI 2.0 (2020 on) dwisf_* sessions, the earliest per participant, in a drawn tier."""
    d = pd.concat([pd.read_csv(t, usecols=SELECT_COLS) for t in tables], ignore_index=True)    # usecols: no value read
    d["date"] = pd.to_datetime(d["acquisition_date"], errors="coerce")
    d = d[(d["date"].dt.year >= 2020) & d["error"].fillna("").astype(str).eq("") & d["dwisf_qc_pass"].astype(str).eq("True")]
    d = d.sort_values("date").drop_duplicates("patno").assign(tier=lambda x: x["dwisf_tier"].map(TIER))
    return d.dropna(subset=["tier"])[["patno", "acquisition_date", "tier"]].reset_index(drop=True)


def draw(c, seed=SEED):
    """Every candidate with its rank in a seeded random order within its tier (rank 0 is drawn first)."""
    rng = np.random.default_rng(seed)
    c = c.sort_values("patno").reset_index(drop=True)          # the draw depends on the candidate set, not the file order
    c["rank"] = -1
    for _, g in c.groupby("tier"):
        c.loc[g.index, "rank"] = rng.permutation(len(g))
    return c.sort_values(["tier", "rank"]).reset_index(drop=True)


def quotas(order, n=2 * N_PER_TIER):
    """20 per tier; a tier with fewer contributes all of its sessions and the other tier makes up the 40."""
    size = order["tier"].value_counts().to_dict()
    q = {t: min(N_PER_TIER, k) for t, k in size.items()}
    short = n - sum(q.values())
    for t in sorted(q):
        add = min(size[t] - q[t], short)
        q[t], short = q[t] + add, short - add
    return q


def analysable(order, dwiq):
    """Per tier, the first quota sessions in draw order with an error-free QSIPrep row; sessions passed over before them
    are technical failures (replaced by the next in the same tier), tallied by exception type."""
    ok = set(dwiq.loc[dwiq["error"].fillna("").astype(str).eq(""), "patno"])
    err = dwiq.set_index("patno")["error"].fillna("").astype(str).str.split(":").str[0].to_dict()
    used, failed = [], Counter()
    for t, k in quotas(order).items():
        n = 0
        for p in order.loc[order["tier"].eq(t), "patno"]:
            if n == k:
                break
            if p in ok:
                used.append(p)
                n += 1
            else:
                failed[err.get(p) or "no QSIPrep output"] += 1
    return used, dict(failed)


def load(tables, dwiq_csv, out):
    """The analysed sessions: PIE's row and QSIPrep's row side by side (identifiers dropped) and the flow."""
    order = pd.read_csv(Path(out) / "qsiprep_order.csv")
    q = pd.read_csv(dwiq_csv, low_memory=False)
    used, failed = analysable(order, q)
    pie = pd.concat([pd.read_csv(t, low_memory=False) for t in tables], ignore_index=True)
    pie = pie.assign(_d=pd.to_datetime(pie["acquisition_date"], errors="coerce")).sort_values("_d").drop_duplicates("patno")
    d = order[order["patno"].isin(used)].merge(pie.drop(columns=["acquisition_date"]), on="patno").merge(q, on="patno", suffixes=("", "_q"))
    flow = {"candidates": order["tier"].value_counts().to_dict(), "quota": quotas(order), "technical_failures": failed}
    if "dwiq_fs_image_id" in d:            # QSIPrep's comparison must use the T1 of PIE's row
        same = d["dwiq_fs_image_id"].astype(str).eq(d["fs_image_id"].astype(str))
        flow["different_t1_excluded"] = int((~same).sum())
        d = d[same]
    return d.drop(columns=[c for c in d if c in ("patno", "fs_image_id", "dwiq_fs_image_id", "source_image_ids")]), flow


def synthetic(n_per_tier=20, icc=0.85, bias=0.01):
    """Fully synthetic sessions (no real value, label or identifier): for dry runs and the test."""
    rng = np.random.default_rng(3)
    n = 2 * n_per_tier
    d = pd.DataFrame({"tier": np.repeat(["topup_eddy", "eddy_only"], n_per_tier), "dwisf_qc_pass": True, "dwis_qc_pass": True,
                      "dwiqf_qc_pass": rng.random(n) > 0.05, "dwiq_qc_pass": True, "dwiqf_qc_failed_checks": ""})
    d.loc[~d["dwiqf_qc_pass"], "dwiqf_qc_failed_checks"] = "fw_physical_fraction"
    base = {"fw": 0.18, "fat": 0.55, "mdt": 8e-4, "fa": 0.45, "md": 9e-4}
    for m in [PRIMARY] + FW_SECONDARY + TENSOR:
        b = base[m.split("_")[0]]
        true = b * (1 + 0.15 * rng.normal(size=n))
        e = 0.15 * b * np.sqrt((1 - icc) / icc)
        pre = ("dwis_", "dwiq_") if m in TENSOR else ("dwisf_", "dwiqf_")
        d[pre[0] + m] = true + rng.normal(0, e, n)
        d[pre[1] + m] = true + bias * b / 0.18 + rng.normal(0, e, n)
    return d, {"synthetic": True}


def _agree(x, y, n_boot):
    """agreement(QSIPrep, PIE): bias is QSIPrep - PIE; ICC(2,1) with a 2,000-resample pair bootstrap CI."""
    x, y = pd.to_numeric(x, errors="coerce").to_numpy(float), pd.to_numeric(y, errors="coerce").to_numpy(float)
    a = agreement(y, x)
    if a["n"] >= 5:
        w = np.c_[y, x][np.isfinite(x) & np.isfinite(y)]
        rng = np.random.default_rng(SEED)
        a["icc21_ci95"] = [float(v) for v in np.percentile([icc21(w[rng.integers(0, len(w), len(w))]) for _ in range(n_boot)], [2.5, 97.5])]
        a["bias_pct_of_pie_mean"] = 100 * a["bias"] / a["ref_mean"]
    return a


def analyse(d, flow, n_boot=B):
    t = lambda c: d[c].astype(str).eq("True")
    fw_ok = t("dwisf_qc_pass") & t("dwiqf_qc_pass")
    ten_ok = t("dwis_qc_pass") & t("dwiq_qc_pass")
    fails = d.loc[~t("dwiqf_qc_pass"), "dwiqf_qc_failed_checks"].fillna("").astype(str).str.split(";").explode()
    res = {"flow": {**flow, "analysed_sessions": int(len(d)), "both_fw_qc_pass": int(fw_ok.sum()),
                    "qsiprep_fw_qc_fail_by_check": fails[fails.ne("")].value_counts().to_dict(),
                    "fw_qc_concordance": float((t("dwisf_qc_pass") == t("dwiqf_qc_pass")).mean()),
                    "by_tier": d.loc[fw_ok, "tier"].value_counts().to_dict()}}
    f = d[fw_ok]
    res["primary"] = _agree(f["dwisf_" + PRIMARY], f["dwiqf_" + PRIMARY], n_boot)
    res["pass"] = bool(res["primary"].get("icc21", -np.inf) >= GATE_ICC)
    res["primary_by_tier"] = {k: _agree(g["dwisf_" + PRIMARY], g["dwiqf_" + PRIMARY], n_boot) for k, g in f.groupby("tier")}
    res["secondary"] = {**{m: _agree(f["dwisf_" + m], f["dwiqf_" + m], n_boot) for m in FW_SECONDARY},
                        **{m: _agree(d.loc[ten_ok, "dwis_" + m], d.loc[ten_ok, "dwiq_" + m], n_boot) for m in TENSOR}}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["select", "analyse"])
    ap.add_argument("--tables", help="comma-separated PIE dwi_features.csv paths (dwi --eddy --study)")
    ap.add_argument("--dwiq", help="dwiq_features.csv from dwi_qsiprep compare")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="dry run on fully synthetic data (reads nothing real)")
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "select":
        order = draw(candidates(a.tables.split(",")))
        order.to_csv(out / "qsiprep_order.csv", index=False)
        print(f"candidates by tier {order['tier'].value_counts().to_dict()}; quota {quotas(order)}")
        return order
    d, flow = synthetic() if a.synthetic else load(a.tables.split(","), a.dwiq, out)
    res = analyse(d, flow, a.n_boot)
    (out / ("synthetic_qsiprep_agreement.json" if a.synthetic else "qsiprep_agreement.json")).write_text(json.dumps(res, indent=1, default=float))
    p = res["primary"]
    print(f"{'synthetic dry run: ' if a.synthetic else ''}posterior-SN FW, QSIPrep vs PIE (n {p['n']}): ICC(2,1) {p.get('icc21', np.nan):.3f} "
          f"{np.round(p.get('icc21_ci95', [np.nan] * 2), 3).tolist()}, CCC {p.get('ccc', np.nan):.3f}, bias {p.get('bias', np.nan):+.4f} "
          f"[{p.get('loa_low', np.nan):+.4f}, {p.get('loa_high', np.nan):+.4f}] -> {'PASS' if res['pass'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    main()
