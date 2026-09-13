"""Cohort rules that PPMI studies keep getting wrong, in one testable place.

Nothing here fits a model or touches an image; these are the decisions that decide
*who is in the analysis and what their covariates are*, which is where silent errors
do the most damage. Each function refuses to guess: an unverified encoding raises, an
unknown genotype stays unknown, a conflicting value is dropped rather than resolved by
row order.

    from pie.experiment import cohort

    frame["carrier"] = cohort.carrier_status(frame)
    frame["sex_male"] = cohort.decode_sex(demographics)
    frame = frame[cohort.concurrent_visit(frame, "T1_EVENT_ID", "SAA_EVENT_ID")]
"""

import numpy as np
import pandas as pd

GENES = ["LRRK2_carrier", "GBA_carrier", "SNCA_carrier"]

# PPMI codes SEX per data module, not per study. Decoding one table's convention across
# all of them silently mislabels everyone's sex; see decode_sex.
SEX_ENCODINGS = {"SCREEN": {0: 0.0, 1: 1.0}, "PARTICIPANT_PROFILE": {1: 1.0, 2: 0.0}}

# PPMI screening and baseline are one clinical occasion split across two visit codes.
SCREENING_BASELINE = ("SC", "BL")
_UNKNOWN_VISITS = ("", "UNK", "UNKNOWN")


def carrier_status(frame, genes=GENES):
    """1 if any listed variant is present, 0 only if every gene is an explicit negative, else NaN.

    A participant untested for one gene is not a genetic control, and treating missing
    genotype as non-carrier inflates any carrier-versus-control contrast.
    """
    g = frame[list(genes)]
    status = pd.Series(np.nan, index=frame.index, dtype=float)
    status.loc[g.eq(0).all(axis=1)] = 0.0
    status.loc[g.eq(1).any(axis=1)] = 1.0
    return status


def concurrent_visit(frame, left, right, equivalent=SCREENING_BASELINE):
    """Boolean mask: the two visit columns name the same occasion.

    Visit codes, not dates: an assay RUNDATE is when the laboratory thawed the sample,
    not when it was drawn, so date arithmetic on it fabricates concurrency. Codes in
    `equivalent` (screening and baseline by default) count as the same occasion; an
    unknown code on the left is never concurrent with anything.
    """
    if left not in frame or right not in frame:
        return pd.Series(False, index=frame.index)
    a = frame[left].astype("string").str.strip().str.upper()
    b = frame[right].astype("string").str.strip().str.upper()
    known = a.notna() & ~a.isin(list(_UNKNOWN_VISITS))
    same = a.eq(b) | (a.isin(list(equivalent)) & b.isin(list(equivalent)))
    return (known & same).fillna(False)


def complete_case_mask(frame, features):
    """Rows where every listed feature is finite. Infinities count as missing."""
    if not list(features):
        raise ValueError("complete-case restriction needs at least one feature")
    return np.isfinite(frame[list(features)].to_numpy(dtype=float)).all(axis=1)


def decode_sex(table, module_column="PAG_NAME", sex_column="SEX", encodings=SEX_ENCODINGS):
    """Male indicator from a PPMI demographics export, decoded per source module.

    Raises on a module whose coding has not been verified rather than applying another
    module's mapping to it — the failure mode this exists to prevent is a whole-cohort
    sex flip that every downstream model quietly absorbs.
    """
    result = pd.Series(np.nan, index=table.index, dtype=float)
    unexpected = set(table[module_column].dropna()) - set(encodings)
    if unexpected:
        raise ValueError(f"Unverified demographic module: {sorted(unexpected)}")
    for module, encoding in encodings.items():
        mask = table[module_column].eq(module)
        result.loc[mask] = pd.to_numeric(table.loc[mask, sex_column], errors="coerce").map(encoding)
    return result


def unique_per_participant(table, value, by="PATNO"):
    """One value per participant, keeping only participants whose sources agree.

    Longitudinal PPMI exports repeat a nominally fixed field per visit and the repeats
    sometimes disagree; `.first()` alone would resolve that by row order, which is
    arbitrary. Participants with conflicting values are dropped, not silently decided.
    """
    groups = table.groupby(by)[value]
    agrees = groups.nunique(dropna=True).eq(1)
    return groups.first().loc[agrees]
