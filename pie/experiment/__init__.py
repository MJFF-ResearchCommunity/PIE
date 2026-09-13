"""pie.experiment — running a reproducible PPMI experiment.

Three concerns that every study repeats, and that decide whether its numbers survive review:

- `cohort`      who is in the analysis, and the PPMI encodings that mislabel them
- `prediction`  nested model selection whose preprocessing never sees the test partition
- `provenance`  hashes, environment and manifests, so a result can be tied to the code that made it

Nothing here loads PPMI tables or touches images; `pie.imaging` and `pie_clean` do that.
See `documentation/experiment.md`.
"""

from pie.experiment import cohort, prediction, provenance
from pie.experiment.cohort import (
    carrier_status, complete_case_mask, concurrent_visit, decode_sex, unique_per_participant,
)
from pie.experiment.prediction import (
    Candidate, CovariateDesign, ImageDesign, candidate_grid, nested_fold, paired_metrics,
)
from pie.experiment.provenance import (
    code_hashes, environment, jsonable, read_json, save_json, sha256, verify_manifest, write_manifest,
)

__all__ = [
    "cohort", "prediction", "provenance",
    "carrier_status", "complete_case_mask", "concurrent_visit", "decode_sex", "unique_per_participant",
    "Candidate", "CovariateDesign", "ImageDesign", "candidate_grid", "nested_fold", "paired_metrics",
    "code_hashes", "environment", "jsonable", "read_json", "save_json", "sha256",
    "verify_manifest", "write_manifest",
]
