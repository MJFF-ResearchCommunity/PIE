"""pie.stats — classical statistics primitives for Parkinson's research.

Each submodule provides pure functions that take pandas Series / DataFrames
and return plain dicts of results. This makes them trivially consumable from
the workbench's FastAPI layer AND from notebooks / CLI scripts.
"""

from pie.stats.describe import summary_statistics, normality_test, missingness_report
from pie.stats.compare import (
    independent_ttest, paired_ttest, welch_ttest,
    mann_whitney, wilcoxon_signed_rank,
    one_way_anova, kruskal_wallis, tukey_hsd, dunn_posthoc,
    chi_square, fisher_exact, mcnemar,
    cohens_d, hedges_g, eta_squared,
)
from pie.stats.correlate import correlate_pair, partial_correlation, correlation_matrix
from pie.stats.regress import linear_regression, logistic_regression, ancova
from pie.stats.longitudinal import linear_mixed_model, change_from_baseline
from pie.stats.survive import kaplan_meier, logrank_test, cox_regression
from pie.stats.multitest import adjust_pvalues
from pie.stats.pd_helpers import compute_ledd, aggregate_updrs, hoehn_yahr_summary

__all__ = [
    "summary_statistics", "normality_test", "missingness_report",
    "independent_ttest", "paired_ttest", "welch_ttest",
    "mann_whitney", "wilcoxon_signed_rank",
    "one_way_anova", "kruskal_wallis", "tukey_hsd", "dunn_posthoc",
    "chi_square", "fisher_exact", "mcnemar",
    "cohens_d", "hedges_g", "eta_squared",
    "correlate_pair", "partial_correlation", "correlation_matrix",
    "linear_regression", "logistic_regression", "ancova",
    "linear_mixed_model", "change_from_baseline",
    "kaplan_meier", "logrank_test", "cox_regression",
    "adjust_pvalues",
    "compute_ledd", "aggregate_updrs", "hoehn_yahr_summary",
]
