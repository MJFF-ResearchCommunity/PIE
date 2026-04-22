import pandas as pd
import pytest
from pie.stats.pd_helpers import compute_ledd, aggregate_updrs, hoehn_yahr_summary


def test_compute_ledd_levodopa_only():
    r = compute_ledd({"levodopa_ir": 300})
    assert r["total_ledd_mg"] == pytest.approx(300.0)


def test_compute_ledd_with_dopa_agonist():
    r = compute_ledd({"levodopa_ir": 300, "pramipexole": 4})
    assert r["total_ledd_mg"] == pytest.approx(700.0)


def test_compute_ledd_unknown_drug():
    r = compute_ledd({"levodopa_ir": 300, "made_up_drug": 50})
    assert r["total_ledd_mg"] == pytest.approx(300.0)
    assert r["per_drug"]["made_up_drug"]["ledd_mg"] is None
    assert "Unknown drug" in r["per_drug"]["made_up_drug"]["note"]


def test_aggregate_updrs():
    df = pd.DataFrame({
        "np1_1": [1, 2], "np1_2": [2, 1], "np2_1": [0, 1],
        "np3_1": [3, 2], "np3_2": [2, 1], "np4_1": [0, 0],
    })
    r = aggregate_updrs(df, part1_cols=["np1_1", "np1_2"], part2_cols=["np2_1"],
                        part3_cols=["np3_1", "np3_2"], part4_cols=["np4_1"])
    assert r["updrs_total"].tolist() == [8, 7]
    assert r["updrs_motor"].tolist() == [5, 3]


def test_hoehn_yahr_summary():
    s = pd.Series([1, 2, 2, 3, 3, 3, 4, 5])
    r = hoehn_yahr_summary(s)
    assert r["counts"][3] == 3
    assert r["proportions"][3] == pytest.approx(3 / 8)
    assert r["median_stage"] == pytest.approx(3.0)


def test_hoehn_yahr_summary_empty():
    r = hoehn_yahr_summary(pd.Series([], dtype=float))
    assert r["n"] == 0
    assert r["counts"] == {}
