"""Agreement between two measurements of the same quantity (a pipeline against a reference, a scan against its repeat).

Correlation alone does not show agreement: report Lin's concordance correlation, ICC(2,1) and Bland-Altman bias with 95 %
limits beside Pearson's r (and, where it exists, the reference's own inter-lab agreement).
"""
import numpy as np


def icc21(data):
    """ICC(2,1), two-way random effects, absolute agreement, single rater (Shrout & Fleiss 1979): rows = subjects,
    columns = raters/methods."""
    data = np.asarray(data, float)
    n, k = data.shape
    m = data.mean()
    ssr, ssc = k * ((data.mean(axis=1) - m) ** 2).sum(), n * ((data.mean(axis=0) - m) ** 2).sum()
    msr, msc = ssr / (n - 1), ssc / (k - 1)
    mse = (((data - m) ** 2).sum() - ssr - ssc) / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def ccc(x, y):
    """Lin's (1989) concordance correlation coefficient, population moments."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(2 * np.mean((x - x.mean()) * (y - y.mean())) / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def agreement(test, ref):
    """Agreement of ``test`` with ``ref`` over pairs where both are finite: n, Pearson r, ICC(2,1), Lin's CCC,
    Bland-Altman bias (test - ref) and 95 % limits of agreement, and both means. Fewer than 3 pairs: only n."""
    x, y = np.asarray(ref, dtype=float), np.asarray(test, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return {"n": int(len(x))}
    diff = y - x
    sd = diff.std(ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):                       # a constant column: r undefined (NaN)
        r = float(np.corrcoef(x, y)[0, 1])
    return {"n": int(len(x)), "pearson_r": r, "icc21": icc21(np.column_stack([y, x])), "ccc": ccc(x, y),
            "bias": float(diff.mean()), "loa_low": float(diff.mean() - 1.96 * sd), "loa_high": float(diff.mean() + 1.96 * sd),
            "test_mean": float(y.mean()), "ref_mean": float(x.mean())}
