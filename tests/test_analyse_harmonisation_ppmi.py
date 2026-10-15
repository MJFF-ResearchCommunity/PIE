"""The pre-registered PPMI harmonisation analysis on fully synthetic data (no real value, label or identifier)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("analyse_harmonisation_ppmi",
                                              Path(__file__).resolve().parent.parent / "scripts" / "analyse_harmonisation_ppmi.py")
hp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp)


def test_known_batch_effects_are_removed_while_age_and_group_effects_survive():
    d, flow = hp.synthetic()
    res = hp.analyse(d, flow, repeats=1)
    b = res["batch_auroc"]
    assert b["raw_logistic"]["mean"] > 0.9 and b["raw_forest"]["mean"] > 0.9               # the batches are real
    assert b["harmonised_logistic"]["mean"] <= 0.6 and b["harmonised_forest"]["mean"] <= 0.6
    assert res["flow"]["hc_excluded_small_batch"] > 0 and len(res["flow"]["batches_used"]) == 8   # a batch under 10 HC dropped
    assert res["pass"] and all(res["gates"].values())
    assert res["pd_hc_d"]["median_abs_d"]["protected"] > 0.1 and res["confounding_report"]["cramers_v"] > 0


def test_gate_fails_without_harmonisation():
    d, _ = hp.synthetic(n_hc=150, n_pd=200, seed=2)
    hc = d[d["dx"].eq("HC") & d["batch"].isin(d.loc[d["dx"].eq("HC"), "batch"].value_counts().loc[lambda s: s >= 10].index)]
    assert hp.batch_auroc(hc.reset_index(drop=True), "raw", hp.SEED) > hp.GATE_AUROC
