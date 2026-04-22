import pytest
from pie.stats.multitest import adjust_pvalues


def test_adjust_pvalues_bonferroni():
    r = adjust_pvalues([0.01, 0.04, 0.03, 0.005], method="bonferroni")
    assert r["adjusted"] == pytest.approx([0.04, 0.16, 0.12, 0.02])
    # Only the p=0.04 → 0.16 survives α=0.05? No, 0.04→0.16 > 0.05. 0.005→0.02 < 0.05, so only index 3 rejected.
    assert r["rejected"][3] is True


def test_adjust_pvalues_fdr_bh():
    r = adjust_pvalues([0.01, 0.02, 0.03, 0.04], method="fdr_bh")
    assert r["method"] == "fdr_bh"
    assert all(isinstance(x, bool) for x in r["rejected"])


def test_adjust_pvalues_unknown_method():
    with pytest.raises(ValueError):
        adjust_pvalues([0.01], method="not_a_method")
