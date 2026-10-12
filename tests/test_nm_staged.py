import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nm_staged as S  # noqa: E402


def test_stratified_auroc_ignores_an_offset_between_vendors():
    """HC sit mostly on the bright vendor and PD on the dark one: a pooled AUROC credits the vendor, the stratified one
    compares within vendor only and finds no group difference."""
    rng = np.random.default_rng(0)
    hc_v = np.r_[["A"] * 90, ["B"] * 10]
    pd_v = np.r_[["A"] * 10, ["B"] * 90]
    off = {"A": 1.0, "B": 0.0}
    hc = rng.normal(0, 0.3, 100) + [off[v] for v in hc_v]
    pdv = rng.normal(0, 0.3, 100) + [off[v] for v in pd_v]
    pooled = S.stratified_auc(hc, pdv, np.zeros(100), np.zeros(100))
    strat = S.auc_with_ci(hc, pdv, hc_v, pd_v, n_boot=200)
    assert pooled > 0.8 and abs(strat["auroc"] - 0.5) < 0.1 and strat["ci95"][0] < 0.5 < strat["ci95"][1]


def test_dry_runs_use_fake_data_only(tmp_path):
    for analysis in ("prodromal-dat", "fresh-clinical"):
        res = S.main([analysis, "--synthetic", "--n-boot", "100", "--out", str(tmp_path)])
        assert res["flow"] == {"synthetic": True} or res["flow"].get("synthetic")
        assert 0.5 < res["primary"]["auroc"] < 0.8
