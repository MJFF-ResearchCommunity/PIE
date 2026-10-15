"""The pre-registered cortical-thickness and MRIQC analysis on fully synthetic data (no real value, label or identifier)."""
import importlib.util
from pathlib import Path

import numpy as np

from pie.stats.agreement import ccc

spec = importlib.util.spec_from_file_location("ts", Path(__file__).resolve().parent.parent / "scripts" / "analyse_t1_surfaces.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)


def test_synthetic_thickness_passes_its_gates_and_flat_age_fails_only_the_age_gate():
    res = ts.analyse(*ts.synthetic(), n_boot=30)
    assert res["flow"]["n_regions"] == 68 and res["flow"]["analysed"] == 600 and res["validated"]
    assert res["agreement_fs7"]["all"]["median_ccc"] >= 0.80 and res["same_day"]["n_pairs"] == 29
    assert res["age_hc"]["pie"]["r_sex_batch_adjusted"] < -0.30 and res["age_hc"]["pie"]["n_hc"] == 165
    assert set(res["agreement_fs7"]["by_field_strength"]) == {"3.0", "1.5"} and res["agreement_fs7"]["area_no_gate"]["n"] == 600
    flat = ts.analyse(*ts.synthetic(age_slope=0.0), n_boot=30)
    assert flat["gates"] == {"agreement_fs7": True, "same_day": True, "age": False} and not flat["validated"]


def test_vectorised_ccc_matches_the_shared_helper_and_mriqc_orientation_finds_flagged_scans():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    Y[:, 0] = X[:, 0] + 0.1
    assert np.allclose(ts._ccc_cols(X, Y), [ccc(X[:, k], Y[:, k]) for k in range(3)])
    b = ts.part_b(ts.synthetic()[2], n_boot=30)
    assert all(b[k]["auroc_flag"]["auroc"] > 0.7 and b[k]["spearman_robust_z"] > 0 for k in ts.IQM)
    assert b["worst_5pct_any_iqm"]["flagged_fraction"] > b["worst_5pct_any_iqm"]["flagged_fraction_rest"]
