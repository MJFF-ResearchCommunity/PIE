"""
cohorts.py — one integer participant key across cohorts, so PIE's PPMI-shaped tables take other studies' data.

PIE keys every table on an integer ``patno``. PPMI keeps its own PATNO. Other cohorts get a key in their own range,
computed from the identifiers alone (no registry, so every pipeline and work dir derives the same key):
- ADNI: from its RID (``002_S_0295`` -> 100000295);
- any other LONI project or BIDS dataset: 10^15 + the first 12 hex digits of sha1("<cohort>/<participant id>")
  (collision odds ~2e-7 at 10,000 participants).
Ranges never overlap, so cohorts can be pooled. The index and sessions tables keep ``cohort`` and ``participant_id``.

    subject_key("ADNI", "002_S_0295")        # 100000295
    subject_key("PPMI", "3001")              # 3001
    subject_key("PPMI", "00000JAN00")        # None (LONI phantom scan)
"""

import hashlib
import re

OFFSET = {"PPMI": 0, "ADNI": 10 ** 8, "HASHED": 10 ** 15}
_ADNI = re.compile(r"\d{3}_S_(\d{4})")


def subject_key(cohort, name):
    """Integer key of a LONI subject folder (``cohort`` = the zip's top folder, e.g. PPMI, ADNI) or a BIDS label
    (``cohort`` = "BIDS:<dataset>"); None for a non-participant (PPMI phantom scans)."""
    cohort, name = str(cohort), str(name)
    if cohort.upper() == "PPMI":
        return int(name) if name.isdigit() else None
    m = _ADNI.fullmatch(name) if cohort.upper() == "ADNI" else None
    if m:
        return OFFSET["ADNI"] + int(m.group(1))
    return OFFSET["HASHED"] + int(hashlib.sha1(f"{cohort}/{name}".encode()).hexdigest()[:12], 16)


def cohort_of(patno):
    """"PPMI", "ADNI" or "OTHER" (a hashed key: look the participant up in the index's participant_id column)."""
    patno = int(patno)
    return "OTHER" if patno >= OFFSET["HASHED"] else "ADNI" if patno >= OFFSET["ADNI"] else "PPMI"
