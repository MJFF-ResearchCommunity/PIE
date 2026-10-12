import numpy as np

from pie.stats.agreement import agreement, ccc, icc21


def test_agreement_statistics_on_known_values():
    sf = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float)
    assert round(icc21(sf), 2) == 0.29                                  # Shrout & Fleiss (1979), 6 targets x 4 judges
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(ccc(x, x + 1), 2.5 / 3.5)
    a = agreement(x + 1, x)
    assert a["bias"] == 1 and np.isclose(a["pearson_r"], 1) and a["test_mean"] == 3.5 and a["ref_mean"] == 2.5
    assert agreement([1, np.nan], [1, 2]) == {"n": 1}
