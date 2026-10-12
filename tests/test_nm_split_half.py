import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nm_split_half as S  # noqa: E402


def test_icc_matches_the_variance_ratio_and_penalises_a_shift_between_halves():
    rng = np.random.default_rng(0)
    true = rng.normal(0, 1, 20000)
    x, y = true + rng.normal(0, 0.5, true.size), true + rng.normal(0, 0.5, true.size)
    assert abs(S.icc_a1(x, y) - 1 / 1.25) < 0.01                         # var(true) / (var(true) + var(noise))
    assert S.icc_a1(x, y + 1.0) < 0.6                                    # absolute agreement: a systematic offset counts
    assert abs(S.spearman_brown(0.8, 2.5) - 2.5 * 0.8 / (1 + 1.5 * 0.8)) < 1e-12


def test_the_current_default_is_kept_unless_beaten_by_more_than_the_tie_margin():
    rel = {S.name(v): {"icc_projected": 0.80} for v in S.VARIANTS}
    rel["nmf_sn_mean_cr_mode_n4"]["icc_projected"] = 0.815
    assert S.choose(rel) == S.CURRENT
    rel["nmf_sn_mean_cr_mode_n4"]["icc_projected"] = 0.85
    assert S.choose(rel) == "nmf_sn_mean_cr_mode_n4"
    rel["nmf_sn_mean_cr"]["icc_projected"] = 0.84                        # within the margin of the best, earlier in order
    assert S.choose(rel) == "nmf_sn_mean_cr"
