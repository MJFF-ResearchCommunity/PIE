"""Parkinson's-specific helpers: LEDD, UPDRS aggregation, H&Y summary."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


# Conversion factors from Tomlinson et al. 2010 (Mov Disord), widely used in PD.
# Values are milligrams of levodopa equivalent per milligram of drug, except
# where noted. ``tolcapone`` / ``entacapone`` are multipliers applied to the
# daily levodopa dose (reflecting COMT-inhibition boosting levodopa bioavailability).
LEDD_FACTORS: Dict[str, float] = {
    "levodopa_ir": 1.0,
    "levodopa_cr": 0.75,
    "levodopa_entacapone": 1.33,
    "pramipexole": 100.0,
    "ropinirole": 20.0,
    "rotigotine": 30.0,
    "apomorphine": 10.0,
    "rasagiline": 100.0,
    "selegiline_oral": 10.0,
    "selegiline_sublingual": 80.0,
    "safinamide": 100.0,
    "amantadine": 1.0,
    "tolcapone": 0.5,
    "entacapone": 0.33,
}


def compute_ledd(doses_mg: Dict[str, float]) -> Dict[str, Any]:
    """Total LEDD from a dict of {drug_name_in_LEDD_FACTORS: mg/day}.

    Unknown drugs are reported separately with a note and contribute 0 to the
    total. This makes failures visible to the user instead of silently dropping
    doses.
    """
    per_drug: Dict[str, Dict[str, Any]] = {}
    total = 0.0
    for drug, dose in doses_mg.items():
        factor = LEDD_FACTORS.get(drug)
        if factor is None:
            per_drug[drug] = {
                "dose_mg": dose,
                "factor": None,
                "ledd_mg": None,
                "note": f"Unknown drug {drug!r}; see LEDD_FACTORS for supported names",
            }
            continue
        ledd = dose * factor
        per_drug[drug] = {"dose_mg": dose, "factor": factor, "ledd_mg": ledd}
        total += ledd
    return {"total_ledd_mg": total, "per_drug": per_drug}


def aggregate_updrs(df: pd.DataFrame,
                    part1_cols: Optional[List[str]] = None,
                    part2_cols: Optional[List[str]] = None,
                    part3_cols: Optional[List[str]] = None,
                    part4_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Aggregate MDS-UPDRS parts into per-row totals.

    Returns a DataFrame with columns for each supplied part plus
    ``updrs_total``. Part 3 column total is called ``updrs_motor`` because
    that's the terminology clinicians use.
    """
    result = pd.DataFrame(index=df.index)
    for name, cols in [
        ("updrs_part1", part1_cols),
        ("updrs_part2", part2_cols),
        ("updrs_motor", part3_cols),
        ("updrs_part4", part4_cols),
    ]:
        if cols:
            result[name] = df[cols].sum(axis=1, skipna=False)
    parts = [c for c in result.columns if c.startswith("updrs_")]
    if parts:
        result["updrs_total"] = result[parts].sum(axis=1, skipna=False)
    return result


def hoehn_yahr_summary(series: pd.Series) -> Dict[str, Any]:
    """Counts, proportions, median/mean stage for a Hoehn & Yahr series."""
    clean = series.dropna()
    total = len(clean)
    if total == 0:
        return {"n": 0, "counts": {}, "proportions": {}, "median_stage": None, "mean_stage": None}
    counts = clean.value_counts().sort_index()
    return {
        "n": int(total),
        "counts": {float(k): int(v) for k, v in counts.items()},
        "proportions": {float(k): float(v / total) for k, v in counts.items()},
        "median_stage": float(clean.median()),
        "mean_stage": float(clean.mean()),
    }
