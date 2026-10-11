"""The pre-registered snceg analysis on fully synthetic data (no real value, label or identifier)."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("analyse_nm_snceg", Path(__file__).resolve().parent.parent / "scripts" / "analyse_nm_snceg.py")
nm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nm)


def test_delong_matches_the_mann_whitney_auroc_and_the_rule_reads_the_ci():
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 60), rng.normal(0, 1, 300)
    a, se = nm.delong(pos, neg)
    assert abs(a - mannwhitneyu(pos, neg).statistic / (60 * 300)) < 1e-12 and 0.02 < se < 0.06
    assert nm.rule(0.70, 0.80) and not nm.rule(0.60, 0.72) and not nm.rule(0.45, 0.75)


def test_synthetic_dry_run_recovers_the_built_in_effect():
    d, flow = nm.synthetic(auc=0.74)
    res = nm.analyse(d, flow, n_boot=50)
    p = res["primary"]
    assert p["n_hc"] == 70 and p["n_pd"] == 350 and 0.62 < p["auroc"] < 0.86
    assert set(res["secondary"]) == {"nms_sn_volume_mm3", "nms_sn_mean_cr_allslices", "nms_sn_mean_cr_age_sex_adjusted", "nms_sn_mean_cr_all_pd"}
    assert res["secondary"]["nms_sn_mean_cr_all_pd"]["n_pd"] == 390
