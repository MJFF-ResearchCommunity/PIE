"""Parkinson's-specific helpers: LEDD, UPDRS aggregation, H&Y summary."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


# Tomlinson et al. 2010 (Mov Disord 25:2649-2653): mg levodopa equivalent per mg/day of drug.
# ``levodopa_entacapone`` is the levodopa component of a levodopa/carbidopa/entacapone
# tablet; its 1.33 already includes the entacapone boost (1 + 0.33).
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
    "amantadine": 1.0,
}

# COMT inhibitors have no LED of their own: they add a fraction of the LED of the
# levodopa taken with them, whatever their own dose. Entacapone 0.33 and tolcapone 0.5:
# Tomlinson et al. 2010; opicapone 0.5: Schade et al. 2020 (Mov Disord Clin Pract
# 7:343-345). Jost et al. 2023 (Mov Disord 38:1236-1252) confirms all three.
COMT_FACTORS: Dict[str, float] = {"entacapone": 0.33, "tolcapone": 0.5, "opicapone": 0.5}

# A fixed LED for any therapeutic dose: Jost et al. 2023 set safinamide 50 or 100 mg/d
# equal to 150 mg immediate-release levodopa.
FLAT_LEDD_MG: Dict[str, float] = {"safinamide": 150.0}

# The levodopa a separately taken COMT inhibitor boosts (levodopa_entacapone already is).
_COMT_BOOSTED = ("levodopa_ir", "levodopa_cr")


def compute_ledd(doses_mg: Dict[str, float]) -> Dict[str, Any]:
    """Total LEDD for one participant-visit regimen, given as {drug: mg/day}.

    - ``LEDD_FACTORS`` drugs contribute dose × factor.
    - ``FLAT_LEDD_MG`` drugs (safinamide) contribute a fixed LED for any dose > 0.
    - ``COMT_FACTORS`` drugs (entacapone, tolcapone, opicapone) count as taken when their
      dose is > 0, and contribute factor × the LED of ``levodopa_ir`` + ``levodopa_cr``.
      Their own dose does not enter the LED. At most one COMT inhibitor is allowed,
      counting the entacapone inside ``levodopa_entacapone``.

    Unknown drugs are reported with a note and contribute 0, so a failure is visible
    instead of silently lowering the total.
    """
    inhibitors = [d for d, dose in doses_mg.items() if d in COMT_FACTORS and dose > 0]
    if doses_mg.get("levodopa_entacapone", 0) > 0:
        inhibitors.append("levodopa_entacapone")
    if len(inhibitors) > 1:
        raise ValueError(f"More than one COMT inhibitor ({inhibitors}); the LED conversion "
                         "assumes a single one")
    levodopa_led = sum(doses_mg.get(d, 0) * LEDD_FACTORS[d] for d in _COMT_BOOSTED)

    per_drug: Dict[str, Dict[str, Any]] = {}
    total = 0.0
    for drug, dose in doses_mg.items():
        if drug in LEDD_FACTORS:
            factor = LEDD_FACTORS[drug]
            per_drug[drug] = {"dose_mg": dose, "factor": factor, "ledd_mg": dose * factor}
        elif drug in FLAT_LEDD_MG:
            per_drug[drug] = {"dose_mg": dose, "factor": None,
                              "ledd_mg": FLAT_LEDD_MG[drug] if dose > 0 else 0.0,
                              "note": f"fixed {FLAT_LEDD_MG[drug]:g} mg LED for any dose > 0"}
        elif drug in COMT_FACTORS:
            factor = COMT_FACTORS[drug]
            per_drug[drug] = {"dose_mg": dose, "factor": factor,
                              "ledd_mg": factor * levodopa_led if dose > 0 else 0.0,
                              "note": f"{factor:g} x levodopa LED ({levodopa_led:g} mg from "
                                      "levodopa_ir + levodopa_cr)"}
        else:
            per_drug[drug] = {
                "dose_mg": dose,
                "factor": None,
                "ledd_mg": None,
                "note": f"Unknown drug {drug!r}; see LEDD_FACTORS, COMT_FACTORS and FLAT_LEDD_MG",
            }
            continue
        total += per_drug[drug]["ledd_mg"]
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
