# Statistics (`pie.stats`)

Classical tests for an analysis frame that is already built. Every function takes pandas
`Series`/`DataFrame`s (or arrays, or a dict of arrays) and returns a plain `dict` of Python
floats, ints, lists and dicts. They return dicts rather than statsmodels/scipy result objects
so that one call works in a notebook, a script and a JSON response without conversion.
The one exception is `aggregate_updrs`, which returns a `DataFrame` because it builds
columns.

```python
from pie import stats            # every function below is re-exported here
from pie.stats import welch_ttest, adjust_pvalues
```

| Module | Functions |
|---|---|
| `describe.py` | `summary_statistics`, `normality_test`, `missingness_report` |
| `compare.py` | `independent_ttest`, `welch_ttest`, `paired_ttest`, `mann_whitney`, `wilcoxon_signed_rank`, `one_way_anova`, `kruskal_wallis`, `tukey_hsd`, `dunn_posthoc`, `chi_square`, `fisher_exact`, `mcnemar`, `cohens_d`, `hedges_g`, `eta_squared` |
| `correlate.py` | `correlate_pair`, `partial_correlation`, `correlation_matrix` |
| `regress.py` | `linear_regression`, `logistic_regression`, `ancova` |
| `longitudinal.py` | `linear_mixed_model`, `change_from_baseline` |
| `survive.py` | `kaplan_meier`, `logrank_test`, `cox_regression` |
| `multitest.py` | `adjust_pvalues` |
| `pd_helpers.py` | `compute_ledd`, `aggregate_updrs`, `hoehn_yahr_summary`, `LEDD_FACTORS`, `COMT_FACTORS`, `FLAT_LEDD_MG` |
| `small_sample.py` | `bootstrap_partial_correlation`, `naive_subset_search`, `nested_subset_search`, `subset_search_null`. Not re-exported by `pie.stats`: import from `pie.stats.small_sample` |

## Conventions

- **Missing values are dropped, never imputed.** Independent-sample tests drop NaN in each
  array separately. Paired tests drop a pair if either side is missing. Frame functions drop
  a row if any column they name is missing. Each result reports `n` (or `n1`/`n2`/`n_pairs`/
  `n_obs`), so check that before you read the p-value.
- **One row per independent unit.** PPMI tables have one row per participant per visit.
  Every test here except `linear_mixed_model` and `change_from_baseline` treats rows as
  independent. Stacking visits into a t-test counts each participant several times and
  makes p-values too small. Reduce to one row per `PATNO` first, or use the mixed model.
- **Optional dependencies are imported inside the function.** `partial_correlation`
  needs `pingouin`, `dunn_posthoc` needs `scikit-posthocs`, and the three survival functions
  need `lifelines`. All three are in `requirements.txt`. Without them, `import pie.stats`
  still works and only those calls raise `ModuleNotFoundError`.
- **Fixed thresholds.** The `reject` flags in `tukey_hsd`, `dunn_posthoc`, the
  `violates_ph` flag in `cox_regression` and the `missingness_report` interpretation all use
  0.05. `normality_test` and `adjust_pvalues` accept an `alpha`.

## Which test do I use?

| Question | Data | Default choice | Alternative |
|---|---|---|---|
| Do two groups differ? | continuous, independent (PD vs HC) | `welch_ttest` | `mann_whitney` for skewed/ordinal data (UPDRS items, H&Y). `independent_ttest` only when equal variances are justified |
| Did it change within a person? | two paired measurements (BL vs V04) | `paired_ttest` | `wilcoxon_signed_rank` |
| Do 3+ groups differ? | continuous | `one_way_anova`, then `tukey_hsd` | `kruskal_wallis`, then `dunn_posthoc` |
| Are two categorical variables associated? | contingency table | `chi_square` | `fisher_exact` for a 2×2 table with small expected counts (< 5) |
| Did a yes/no outcome change within a person? | paired binary | `mcnemar` | — |
| Do two measures move together? | two continuous | `correlate_pair(method="pearson")` | `"spearman"` (monotone, outlier-robust), `"kendall"` (small n, many ties) |
| Same, holding covariates fixed? | continuous + covariates | `partial_correlation` | `method="spearman"` |
| Many correlations at once? | variable list | `correlation_matrix` (FDR-adjusted) | — |
| Group difference adjusted for age/sex? | continuous outcome | `ancova` | `linear_regression` with dummy-coded group |
| What predicts a continuous outcome? | continuous | `linear_regression` | — |
| What predicts a binary outcome? | 0/1 | `logistic_regression` | — |
| How does a score evolve across visits? | repeated visits per participant | `linear_mixed_model` | `change_from_baseline` for descriptive change |
| Time to an event (conversion, H&Y ≥ 3)? | time + event flag | `kaplan_meier` + `logrank_test` | `cox_regression` to adjust for covariates |
| I ran many tests | list of p-values | `adjust_pvalues(method="fdr_bh")` | `"holm"` when any false positive is costly |

Don't choose between a t-test and a rank test by looking at `normality_test` on the same data.
At PPMI sample sizes Shapiro-Wilk rejects departures from normality too small to matter, and
at small n it has too little power to detect real ones. Choose from the measurement scale
(ordinal clinical scores → rank tests) and use the plots.

## describe

| Function | Returns |
|---|---|
| `summary_statistics(df, variables)` | `{var: {n, n_missing, pct_missing, mean, median, std, min, max, q1, q3, iqr, skew, kurtosis}}` |
| `normality_test(series, test="shapiro", alpha=0.05)` | `{test, statistic, p_value, n, is_normal, alpha}` |
| `missingness_report(df, variables=None)` | `{n_rows, per_column: {col: {n_missing, pct_missing}}, little_mcar}`, where `little_mcar` is `{statistic, p_value, dof, n_patterns, n_rows, interpretation}` or `None` |

```python
import numpy as np, pandas as pd
from pie import stats

rng = np.random.default_rng(0)
df = pd.DataFrame({"PATNO": np.arange(1, 101),
                   "age": rng.normal(65, 8, 100),
                   "NP3TOT": np.r_[rng.normal(25, 10, 90), [np.nan] * 10],
                   "COHORT": rng.choice(["PD", "HC"], 100)})

s = stats.summary_statistics(df, ["age", "NP3TOT"])
s["NP3TOT"]["n"], s["NP3TOT"]["pct_missing"]          # (90, 10.0)
stats.normality_test(df["age"])["is_normal"]
m = stats.missingness_report(df, ["age", "NP3TOT"])
m["per_column"]["NP3TOT"], m["little_mcar"]["p_value"]
```

- `summary_statistics` raises `KeyError` for an absent column and `ValueError` for a
  non-numeric one. It fails instead of returning something meaningless for a categorical
  column. The statistic keys are only present when `n > 0`. `std` needs n > 1, `skew` n > 2 and
  `kurtosis` n > 3, otherwise NaN. Skew and (excess) kurtosis are bias-corrected.
- `normality_test`: `test="shapiro"` or `"ks"` (Kolmogorov-Smirnov against a normal with the
  sample's own mean and SD). Nothing switches automatically at n > 5000; pick `"ks"` yourself.
  The KS p-value has no Lilliefors correction for the estimated parameters, so it is too
  large and favours "normal". `is_normal` is just `p_value > alpha`: failing to reject is not
  evidence of normality.
- `missingness_report`: `little_mcar` is Little's (1988) MCAR test over the named numeric
  columns. EM estimates the mean and covariance, and d² compares each missingness pattern's
  observed means with them, on Σ pⱼ − p degrees of freedom. It reproduces
  `naniar::mcar_test` on R's `airquality` data (d² = 35.1, df = 14, p = 0.0014). In
  simulation it rejects about 5 % of MCAR datasets at n = 200 and all of the MAR ones. The
  p-value is asymptotic and assumes the columns are jointly normal, so it is looser at small
  n and on skewed or ordinal scores. A small p means missingness depends on observed values,
  so the data are not MCAR. A large p does not show MCAR, and no test can rule out
  missingness that depends on the unobserved values themselves. All-missing columns and
  rows are ignored, and `n_rows` is the number of rows used. The result is `None` when
  nothing is missing, fewer than two numeric columns have data, or the covariance is
  singular (for example a constant column).

## compare

### Two groups

| Function | Returns |
|---|---|
| `independent_ttest(a, b)` | `{test, statistic, p_value, df, n1, n2, mean1, mean2, cohens_d, hedges_g}` |
| `welch_ttest(a, b)` | `{test, statistic, p_value, df, n1, n2, mean1, mean2, cohens_d}` (`df` is fractional) |
| `paired_ttest(a, b)` | `{test, statistic, p_value, df, n_pairs, mean_diff}` (`mean_diff` = mean of `a − b`) |
| `mann_whitney(a, b, alternative="two-sided")` | `{test, u_statistic, p_value, n1, n2, median1, median2}` |
| `wilcoxon_signed_rank(a, b)` | `{test, statistic, p_value, n_pairs}` |

```python
import numpy as np
from pie import stats

rng = np.random.default_rng(1)
pd_group = rng.normal(28, 10, 60)          # synthetic motor scores
hc_group = rng.normal(4, 3, 40)

r = stats.welch_ttest(pd_group, hc_group)
r["statistic"], r["p_value"], r["df"], r["cohens_d"]
stats.mann_whitney(pd_group, hc_group, alternative="greater")["p_value"]

bl = rng.normal(20, 6, 30)
v04 = bl + rng.normal(2, 3, 30)
stats.paired_ttest(v04, bl)["mean_diff"]   # positive = V04 higher
stats.wilcoxon_signed_rank(v04, bl)["n_pairs"]
```

- The sign is always `a − b`: `statistic`, `cohens_d` and `mean_diff` are negative when `a` is
  lower.
- Paired functions raise `ValueError` if `a` and `b` differ in length. Before pairing,
  align on `PATNO` (a merge or pivot), or the pairs are just row order.
- `mann_whitney` reports U for `a`. `alternative` is passed to scipy: `"two-sided"`,
  `"less"` or `"greater"`. `wilcoxon_signed_rank` uses scipy's default handling of zero
  differences, which discards them. With many tied scores, `n_pairs` overstates how much data the test used.

### Three or more groups

Groups are passed as a dict `{label: values}`. The labels come back in the output.

| Function | Returns |
|---|---|
| `one_way_anova(groups)` | `{test, statistic, p_value, df_between, df_within, eta_squared, n_per_group, mean_per_group}` |
| `kruskal_wallis(groups)` | `{test, statistic, p_value, df, n_per_group, median_per_group}` |
| `tukey_hsd(groups)` | `{method: "tukey_hsd", pairwise: [{group1, group2, mean_diff, p_adj, lower, upper, reject}]}` |
| `dunn_posthoc(groups, p_adjust="bonferroni")` | `{method: "dunn_<p_adjust>", pairwise: [{group1, group2, p_adj, reject}]}` |
| `eta_squared(groups)` | `float` (the ANOVA's η²) |

```python
import numpy as np, pandas as pd
from pie import stats

rng = np.random.default_rng(2)
df = pd.DataFrame({"COHORT": np.repeat(["HC", "PD", "PRODROMAL"], 30),
                   "score": np.r_[rng.normal(0, 1, 30), rng.normal(1.5, 1, 30), rng.normal(0.7, 1, 30)]})
groups = {k: g["score"].to_numpy() for k, g in df.groupby("COHORT")}

stats.one_way_anova(groups)["eta_squared"]
for pair in stats.tukey_hsd(groups)["pairwise"]:
    print(pair["group1"], pair["group2"], round(pair["mean_diff"], 2), pair["reject"])

stats.kruskal_wallis(groups)["p_value"]
stats.dunn_posthoc(groups, p_adjust="holm")["pairwise"]   # needs scikit-posthocs
```

- Run the post-hoc only after the omnibus test rejects. Tukey pairs with ANOVA and Dunn
  pairs with Kruskal-Wallis.
- `tukey_hsd` `mean_diff` is `group2 − group1` (statsmodels' convention), the reverse of the
  two-group functions.
- `dunn_posthoc`'s `p_adjust` goes straight to `scikit_posthocs.posthoc_dunn`, which accepts
  the `statsmodels` `multipletests` method names (`"bonferroni"`, `"holm"`, `"fdr_bh"`, …).
- `one_way_anova` raises with fewer than two groups. Neither omnibus test checks
  homogeneity of variance.

### Categorical

| Function | Returns |
|---|---|
| `chi_square(table)` | `{test, statistic, p_value, dof, expected}` |
| `fisher_exact(table)` | `{test, odds_ratio, p_value}`, 2×2 only (`ValueError` otherwise) |
| `mcnemar(b, c, exact=True)` | `{test, statistic, p_value, b, c}` |

```python
import pandas as pd
from pie import stats

df = pd.DataFrame({"COHORT": ["PD"] * 40 + ["HC"] * 30,
                   "hyposmia": [1] * 28 + [0] * 12 + [1] * 5 + [0] * 25})
table = pd.crosstab(df["COHORT"], df["hyposmia"]).to_numpy().tolist()

r = stats.chi_square(table)
r["statistic"], r["p_value"], r["expected"]          # check expected counts >= 5
stats.fisher_exact(table)["odds_ratio"]

# paired binary: b = positive only at BL, c = positive only at follow-up
stats.mcnemar(b=3, c=15)["p_value"]
```

- `chi_square` applies scipy's Yates continuity correction to a 2×2 table (dof = 1), so
  its statistic is smaller than the textbook uncorrected one. Larger tables are uncorrected.
- `fisher_exact`'s odds ratio is `(t[0][0]·t[1][1]) / (t[0][1]·t[1][0])`, so it depends on
  row and column order. `pd.crosstab` sorts labels alphabetically.
- `mcnemar` takes only the two **discordant** counts, not the full table. With
  `exact=True` it runs the binomial test, and `statistic` is `min(b, c)`. With `exact=False` it is the
  continuity-corrected chi-square.

### Effect sizes

| Function | Definition |
|---|---|
| `cohens_d(a, b)` | `(mean(a) − mean(b)) / pooled SD`. NaN if either group has < 2 values or the pooled SD is 0 |
| `hedges_g(a, b)` | `d · (1 − 3 / (4(n₁+n₂) − 9))`, the small-sample correction, with n₁, n₂ the non-missing counts |
| `eta_squared(groups)` | `SS_between / SS_total` |

Report an effect size next to every p-value: with a large enough sample, any
difference is significant. The t-test dicts already carry `cohens_d` (and `hedges_g` for
`independent_ttest`).

## correlate

| Function | Returns |
|---|---|
| `correlate_pair(a, b, method="pearson")` | `{method, r, p_value, n}` |
| `partial_correlation(df, x, y, covariates, method="pearson")` | `{method: "partial_<method>", r, p_value, n, covariates, ci_lower, ci_upper}` |
| `correlation_matrix(df, variables, method="pearson", fdr_method="fdr_bh")` | `{method, n, matrix, p_values, p_values_adjusted, fdr_method}` |

```python
import numpy as np, pandas as pd
from pie import stats

rng = np.random.default_rng(3)
age = rng.normal(65, 8, 150)
duration = 0.3 * (age - 50) + rng.normal(0, 2, 150)
df = pd.DataFrame({"age": age, "duration": duration,
                   "NP3TOT": 2 * duration + rng.normal(0, 6, 150),
                   "MCATOT": rng.normal(27, 2, 150)})

stats.correlate_pair(df["duration"], df["NP3TOT"], method="spearman")
stats.partial_correlation(df, "age", "NP3TOT", covariates=["duration"])   # needs pingouin

m = stats.correlation_matrix(df, ["age", "duration", "NP3TOT", "MCATOT"])
m["matrix"]["duration"]["NP3TOT"], m["p_values_adjusted"]["duration"]["NP3TOT"]
```

- `correlate_pair` **aligns the two Series on their index**, not their position, then
  drops incomplete pairs. Two Series from different tables with different indexes pair
  unrelated rows or none at all. Take both columns from one merged frame. It needs ≥ 3
  complete pairs. For `"kendall"`, `r` holds τ.
- `partial_correlation` supports `"pearson"` and `"spearman"` (pingouin's methods). The CI
  keys are `None` if the installed pingouin reports no interval.
- `correlation_matrix` drops rows **listwise** across all `variables`, so `n` is the rows
  complete on every variable, and one sparse column shrinks every cell. `matrix`,
  `p_values` and `p_values_adjusted` are nested dicts (`result["matrix"][v][w]`) and the
  diagonals are 1.0. Only the off-diagonal upper triangle is adjusted, as one family.
  `fdr_method` accepts any `statsmodels` `multipletests` method.

## regress

| Function | Returns |
|---|---|
| `linear_regression(df, outcome, predictors, standardize=False)` | `{model: "ols", n, coefficients, intercept, r_squared, adj_r_squared, f_statistic, f_p_value, diagnostics}` |
| `logistic_regression(df, outcome, predictors)` | `{model: "logit", n, coefficients, intercept, pseudo_r2, log_likelihood, auc, roc_curve}` |
| `ancova(df, outcome, group, covariates)` | `{model: "ancova", n, effects, r_squared}` |

`coefficients` is a list with one dict per predictor. The intercept is reported separately.

- OLS: `{predictor, estimate, std_error, t_statistic, p_value, ci_lower, ci_upper}`
- Logit: `{predictor, estimate, std_error, z_statistic, p_value, odds_ratio, or_ci_lower, or_ci_upper}`
- OLS `diagnostics`: `{vif: {predictor: float}, durbin_watson, residuals, fitted, standardized_residuals}`
- ANCOVA `effects`: `{source, sum_sq, df, f_statistic, p_value}`. The group row is named
  `"group"` and the last row is `"Residual"` (with `f_statistic`/`p_value` `None`).

```python
import numpy as np, pandas as pd
from pie import stats

rng = np.random.default_rng(4)
n = 200
df = pd.DataFrame({"age": rng.normal(65, 8, n), "sex_male": rng.integers(0, 2, n),
                   "COHORT": rng.choice(["HC", "PD"], n)})
df["is_pd"] = (df["COHORT"] == "PD").astype(int)
df["DATSCAN_PUTAMEN"] = 2.5 - 0.02 * (df["age"] - 65) - 1.0 * df["is_pd"] + rng.normal(0, 0.3, n)

ols = stats.linear_regression(df, "DATSCAN_PUTAMEN", ["age", "sex_male", "is_pd"])
ols["coefficients"][2]["estimate"], ols["diagnostics"]["vif"]

logit = stats.logistic_regression(df, "is_pd", ["DATSCAN_PUTAMEN", "age"])
logit["coefficients"][0]["odds_ratio"], logit["auc"]

a = stats.ancova(df, "DATSCAN_PUTAMEN", "COHORT", ["age", "sex_male"])
next(e for e in a["effects"] if e["source"] == "group")["p_value"]
```

- `linear_regression` and `logistic_regression` take numeric predictor columns as they are,
  with no formula. Dummy-code categorical predictors yourself (`is_pd` above). `ancova`
  wraps `group` in `C()`, so there it can be a string.
- `standardize=True` z-scores the **predictors only**, not the outcome. The coefficients are then
  change in outcome per predictor SD.
- VIF is computed only with ≥ 2 predictors (a single predictor gets NaN). Values above about
  5–10 mean the coefficients are unstable.
- `logistic_regression`'s `auc` and `roc_curve` (`{fpr, tpr}`, downsampled to ≤ 100 points)
  are **training-set** values and optimistic. Use `pie.experiment.prediction` for
  discrimination you intend to report. The outcome is cast to `int`, so it must be 0/1. `pseudo_r2` is
  McFadden's.
- `ancova` uses type-II sums of squares and fits no group × covariate interaction, so it
  assumes the covariate slopes are equal across groups. If you doubt that, test it with an interaction term in
  statsmodels.

## longitudinal

| Function | Returns |
|---|---|
| `linear_mixed_model(df, outcome, fixed_effects, group, random_slopes=None)` | `{model: "lmm", n_obs, n_groups, fixed_effects, random_effect_variance, residual_variance, log_likelihood, ml_log_likelihood, aic, bic}` |
| `change_from_baseline(df, subject, time, outcome, baseline_time=0)` | `{n_subjects, n_without_baseline, baseline_time, summary_by_time, per_subject}` |

`fixed_effects` rows are `{predictor, estimate, std_error, z_statistic, p_value, ci_lower, ci_upper}`.
A categorical (string) predictor gets one row per non-reference level, named like
`cohort[T.PD]`. `summary_by_time` is `{time: {n, mean_change, sd_change, mean_pct_change}}`
for every non-baseline time, over subjects that have a baseline. `per_subject` is a list of records with the subject, time, outcome,
`change` and `pct_change` columns.

```python
import numpy as np, pandas as pd
from pie import stats

rng = np.random.default_rng(5)
rows = []
for patno in range(1, 41):
    start, slope = rng.normal(20, 5), rng.normal(2, 0.5)
    for year in range(4):
        rows.append({"PATNO": patno, "years": year, "is_pd": patno % 2,
                     "NP3TOT": start + slope * year + rng.normal(0, 1.5)})
long = pd.DataFrame(rows)

lmm = stats.linear_mixed_model(long, "NP3TOT", ["years", "is_pd"], group="PATNO",
                               random_slopes=["years"])
lmm["fixed_effects"][0]            # progression per year
lmm["n_obs"], lmm["n_groups"]      # (160, 40)
lmm["aic"], lmm["bic"]             # from the maximum-likelihood refit

cfb = stats.change_from_baseline(long, "PATNO", "years", "NP3TOT", baseline_time=0)
cfb["summary_by_time"][3]["mean_change"], cfb["n_without_baseline"]
```

- The mixed model has a random intercept per `group` (use `PATNO`). `random_slopes` adds
  per-participant slopes, and every name in it must also be in `fixed_effects`, or it raises
  `ValueError`. A random slope is a participant's deviation around the fixed slope, so a
  random slope with no fixed slope forces the average slope to zero.
- Coefficients, `log_likelihood` and the variances come from the REML fit (statsmodels'
  default, less biased variance components). REML likelihoods can't compare models whose
  fixed effects differ, so `aic`, `bic` and `ml_log_likelihood` come from a
  maximum-likelihood refit of the same model. Compare models with those, fitted on the
  same rows. `random_effect_variance` is the random-intercept variance only.
- Each coefficient is matched to its exact formula term, so names that contain one another
  (`time`, `time2`) are safe.
- `time` is used as a number. For PPMI visit codes, convert `EVENT_ID` to years
  since baseline first. `BL`, `V04`… are labels, not times.
- `change_from_baseline` compares `time` to `baseline_time` by equality, so `baseline_time="BL"`
  works with `EVENT_ID` strings. It needs one row per subject and time. Duplicates raise
  `ValueError` instead of picking a baseline by row order. Subjects with no baseline row are
  counted in `n_without_baseline` and left out of `summary_by_time`. They stay in
  `per_subject` with NaN change. `pct_change` is NaN where the baseline is 0.

## survive

| Function | Returns |
|---|---|
| `kaplan_meier(df, time, event, group=None)` | `{timeline, survival, ci_lower, ci_upper}` |
| `logrank_test(df, time, event, group)` | `{test: "logrank", statistic, p_value, n_groups}` |
| `cox_regression(df, time, event, covariates)` | `{model: "cox_ph", n, n_events, coefficients, concordance, log_likelihood, ph_test, ph_test_error}` |

`event` is 1 when the event was observed and 0 when the participant was censored. `kaplan_meier`
evaluates every curve on one shared 100-point `timeline` from 0 to the largest time.
`survival`, `ci_lower` and `ci_upper` are dicts keyed by `str(group level)`, or `"_overall"`
with no group. Cox `coefficients` rows are
`{predictor, coef, hazard_ratio, se, z_statistic, p_value, hr_ci_lower, hr_ci_upper}`, and
`ph_test` rows are `{predictor, test_statistic, p_value, violates_ph}`.

```python
import numpy as np, pandas as pd
from pie import stats

rng = np.random.default_rng(6)
n = 200
risk = rng.integers(0, 2, n)                          # e.g. synthetic hyposmia flag
t = rng.exponential(np.where(risk == 1, 4.0, 10.0))   # years to conversion
df = pd.DataFrame({"years": np.minimum(t, 8.0), "converted": (t < 8.0).astype(int),
                   "risk": risk, "age": rng.normal(65, 8, n)})

km = stats.kaplan_meier(df, "years", "converted", group="risk")
km["survival"]["1"][-1]                               # survival at end of follow-up
stats.logrank_test(df, "years", "converted", "risk")["p_value"]

cox = stats.cox_regression(df, "years", "converted", ["risk", "age"])
cox["coefficients"][0]["hazard_ratio"], cox["concordance"], cox["ph_test"]
```

- `logrank_test` runs the two-sample test for exactly two levels and the multivariate
  log-rank for more. `statistic` is the χ² either way.
- `cox_regression` passes the covariates to lifelines as they are, so they must be numeric.
  `ph_test` is a Schoenfeld-residual test on rank-transformed time. A `violates_ph` covariate
  means its hazard ratio changes over follow-up, and the single HR is an average. If the
  test itself fails (for example with a perfectly separating covariate), `ph_test` is `[]`,
  `ph_test_error` holds the exception text and a `RuntimeWarning` is raised. When the test
  runs, `ph_test_error` is `None`.
- Time zero must mean the same thing for everyone (for example baseline visit date). Mixing
  enrolment and diagnosis origins creates immortal-time bias that no test here detects.

## multitest

| Function | Returns |
|---|---|
| `adjust_pvalues(p_values, method="fdr_bh", alpha=0.05)` | `{method, alpha, n_tests, original, adjusted, rejected}` |

| `method` | Controls | Use when |
|---|---|---|
| `"bonferroni"` | FWER | few tests, and any false positive is costly |
| `"holm"` | FWER | same guarantee as Bonferroni and never less powerful. Prefer it |
| `"sidak"` | FWER | tests are independent |
| `"fdr_bh"` | FDR | default: many independent or positively correlated tests (ROI-wise, feature screens) |
| `"fdr_by"` | FDR | arbitrary dependence between tests. Conservative |
| `"fdr_tsbh"` | FDR | two-stage BH, which estimates the share of true nulls. More power when many effects are real |

Any other string raises `ValueError`.

```python
from pie import stats

p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
r = stats.adjust_pvalues(p, method="fdr_bh")
list(zip(r["adjusted"], r["rejected"]))
stats.adjust_pvalues(p, method="holm", alpha=0.01)["rejected"]
stats.adjust_pvalues([0.01, float("nan"), 0.04], method="holm")["adjusted"]   # [0.02, nan, 0.04]
```

- Adjust over the whole family you actually tested, all ROIs and all outcomes, not just
  the ones that looked interesting.
- A NaN p-value (a test that couldn't be computed) stays NaN in `adjusted`, is never
  `rejected`, and doesn't count as a test. The rest are adjusted as a family of `n_tests`.
  If a failed test should still count towards the correction, give it p = 1 instead.
- `rejected` depends on `alpha`. `adjusted` does too for `fdr_tsbh`.

## pd_helpers

| Function | Returns |
|---|---|
| `compute_ledd(doses_mg)` | `{total_ledd_mg, per_drug: {drug: {dose_mg, factor, ledd_mg[, note]}}}` |
| `aggregate_updrs(df, part1_cols=None, part2_cols=None, part3_cols=None, part4_cols=None)` | `DataFrame` with `updrs_part1`, `updrs_part2`, `updrs_motor`, `updrs_part4`, `updrs_total` |
| `hoehn_yahr_summary(series)` | `{n, counts, proportions, median_stage, mean_stage}` |

### LEDD

`compute_ledd` takes one participant-visit's regimen as `{drug: mg/day}`. Drugs fall into
three groups because the literature converts them in three different ways:

| Group | Keys | LED contributed |
|---|---|---|
| `LEDD_FACTORS` (Tomlinson et al. 2010) | `levodopa_ir` 1.0, `levodopa_cr` 0.75, `levodopa_entacapone` 1.33, `pramipexole` 100, `ropinirole` 20, `rotigotine` 30, `apomorphine` 10, `rasagiline` 100, `selegiline_oral` 10, `selegiline_sublingual` 80, `amantadine` 1.0 | dose × factor |
| `COMT_FACTORS` | `entacapone` 0.33, `tolcapone` 0.5 (Tomlinson 2010), `opicapone` 0.5 (Schade et al. 2020) | factor × LED of `levodopa_ir` + `levodopa_cr`, when the inhibitor's dose is > 0 |
| `FLAT_LEDD_MG` | `safinamide` 150 (Jost et al. 2023) | 150 mg for any dose > 0 |

```python
from pie import stats

r = stats.compute_ledd({"levodopa_ir": 300, "pramipexole": 1.5, "rasagiline": 1,
                        "entacapone": 600,            # own dose ignored: 0.33 x levodopa LED
                        "istradefylline": 20})
r["total_ledd_mg"]                                    # 300 + 150 + 100 + 99 = 649.0
r["per_drug"]["entacapone"]["note"]
r["per_drug"]["istradefylline"]["note"]               # unknown drug: reported, adds 0

stats.compute_ledd({"levodopa_ir": 300, "levodopa_cr": 200, "opicapone": 50,
                    "safinamide": 100})["total_ledd_mg"]   # 450 + 0.5 x 450 + 150 = 825.0
```

- **COMT inhibitors have no LED of their own. They add to levodopa.** Their contribution
  is the factor times the LED of the levodopa they boost (IR mg + 0.75 × CR mg). This
  follows Jost et al. 2023: first the LED of the levodopa medications, then that × 0.33 or
  × 0.5. The inhibitor's own mg only marks it as taken. The whole daily IR + CR levodopa is
  boosted, which is Tomlinson's convention. Earlier versions multiplied the inhibitor's own
  dose, so `{"entacapone": 600}` added 198 mg whatever the levodopa.
- `levodopa_entacapone` is the levodopa component of a levodopa/carbidopa/entacapone
  tablet. Its 1.33 already includes the entacapone, so it is not boosted again. It counts as
  a COMT inhibitor, and more than one inhibitor in a regimen raises `ValueError`. Don't also
  list those tablets under `levodopa_ir`.
- Safinamide is a flat equivalence, not a per-mg factor. Jost et al. 2023 (*Mov Disord*
  38:1236–1252) set 50 or 100 mg/day equal to 150 mg levodopa. Schade et al. 2020 (*Mov
  Disord Clin Pract* 7:343–345) had proposed 100 mg. Earlier versions used 100 per mg, which
  turned 100 mg into 10,000 mg LEDD.
- Keys must match exactly (lower-case, underscore). Unknown drugs contribute 0 and carry a
  `note`, so check `per_drug` for notes instead of trusting the total. Brand names,
  `"Sinemet"` or `"levodopa"` alone are all unknown.
- Doses are not validated. Convert free-text PPMI medication logs to mg/day yourself. The
  rotigotine factor is per mg/24 h patch. Pramipexole's 100 is per mg of **salt**. Jost
  et al. give 142.86 per mg of base.

### MDS-UPDRS totals

```python
import numpy as np, pandas as pd
from pie import stats

visits = pd.DataFrame({"PATNO": [1, 2, 3],
                       "NP3SPCH": [1, 0, 2], "NP3FACXP": [2, 1, np.nan], "NP3RIGN": [1, 0, 1],
                       "NP2SPCH": [1, 0, 1], "NP2SALV": [0, 0, 2]})
part3 = ["NP3SPCH", "NP3FACXP", "NP3RIGN"]          # item columns only, never NP3TOT
part2 = ["NP2SPCH", "NP2SALV"]

totals = stats.aggregate_updrs(visits, part2_cols=part2, part3_cols=part3)
visits.join(totals)          # updrs_part2, updrs_motor, updrs_total; row 3 motor/total NaN
```

- `aggregate_updrs` sums the listed item columns row-wise with `skipna=False`. **One missing
  item makes that part and `updrs_total` NaN**, rather than a silently lower score.
  Prorating or imputing items is a decision for the analysis, not the helper.
- Only parts you pass get a column. Part III is named `updrs_motor`. `updrs_total` is the
  sum of the parts you supplied, so with only Part III it equals `updrs_motor`. Say which
  parts a "total" includes.
- List **item** columns. Including a precomputed total such as `NP3TOT` double-counts.
  Recode non-score codes to NaN first. The items must already be numeric.
- Row alignment is by index. The result has the input's index and no `PATNO`/`EVENT_ID`, so
  `join` it back.
- In PPMI, Part I is split across a rater form and a patient questionnaire, and Part III is
  recorded per medication state. Merge the two Part I forms and filter Part III to one state
  (OFF or ON) before summing. Otherwise one visit contributes two rows.

### Hoehn & Yahr

```python
import pandas as pd
from pie import stats

hy = pd.Series([1, 2, 2, 2, 3, 2, 1, None, 4])
r = stats.hoehn_yahr_summary(hy)
r["n"], r["counts"], r["median_stage"]      # (8, {1.0: 2, 2.0: 4, 3.0: 1, 4.0: 1}, 2.0)
```

Stages are keys cast to `float`. Every non-NaN value counts as a stage, so recode "unable to
rate" codes to NaN first. H&Y is ordinal: report the median and counts. `mean_stage` is
included only because some papers report it. On an empty or all-NaN series it returns
`n = 0` and `None` stages.

## small_sample

Small imaging samples invite two mistakes: reading a correlation that a confound produced, and
reporting a score from features that were chosen on the same participants that scored them. These four
functions address both. They are the only `pie.stats` functions not re-exported at package level:

```python
from pie.stats import small_sample as ss         # not: from pie import stats
```

| Function | Returns |
|---|---|
| `bootstrap_partial_correlation(df, x, y, covars=(), method="pearson", n_boot=1000, seed=0, ci=0.95)` | `{r, p, n, covariates, method, ci_low, ci_high, n_boot}` |
| `naive_subset_search(X, y, metric="precision", max_size=None)` | `{subset, <metric>, roc_auc, n_subsets_searched}` |
| `nested_subset_search(X, y, metric="precision", max_size=None, inner_folds=5, seed=0)` | `{roc_auc, precision, selection_frequency, held_out_scores}` |
| `subset_search_null(X, y, n_permutations=100, metric="precision", max_size=None, seed=0)` | `{observed, null_median, null_95th, p, n_permutations}` |

### A correlation that is really age

`bootstrap_partial_correlation` reports the partial correlation with an analytic p-value and a percentile
bootstrap interval. Unlike `correlate.partial_correlation` it needs no `pingouin`.

```python
import numpy as np, pandas as pd
from pie.stats import small_sample as ss

rng = np.random.default_rng(0)
age = rng.normal(65, 8, 120)
df = pd.DataFrame({"age": age,
                   "nm_contrast": -0.006 * age + rng.normal(0, 0.05, 120),   # both just track age
                   "updrs_motor": 0.5 * age + rng.normal(0, 4, 120)})

ss.bootstrap_partial_correlation(df, "nm_contrast", "updrs_motor")["r"]            # -0.53
r = ss.bootstrap_partial_correlation(df, "nm_contrast", "updrs_motor", ["age"])
r["r"], r["p"], (r["ci_low"], r["ci_high"])       # -0.10, 0.28, (-0.28, 0.09): the interval spans 0
```

Rows with a missing value in any named column are dropped, and `n` reports how many were used. Fewer than
`len(covars) + 4` complete rows raises `ValueError`. Use `method="spearman"` for ranks.

### Choosing features and scoring them on the same people

Searching feature subsets and then reporting the winner's cross-validated score is optimistic: the search
already saw every participant. `nested_subset_search` repeats the whole search inside each outer training
fold, so the reported score comes from held-out participants only. The difference is not subtle — here five
features are pure noise and cannot predict anything:

```python
import numpy as np
from pie.stats import small_sample as ss

rng = np.random.default_rng(3)
X = rng.normal(size=(30, 5))                      # 5 features, no signal at all
y = np.array([0] * 10 + [1] * 20)

ss.naive_subset_search(X, y, metric="roc_auc")["roc_auc"]                     # 0.73  (searched 31 subsets)
ss.nested_subset_search(X, y, metric="roc_auc", inner_folds=3)["roc_auc"]     # 0.44  (chance, correctly)
```

`0.73` from noise is what an honest-looking write-up can contain. To size that bias for your own sample,
`subset_search_null` reruns the naive search on shuffled labels:

```python
null = ss.subset_search_null(X, y, n_permutations=20, metric="roc_auc")
null["observed"], null["null_median"], null["p"]          # 0.73, 0.66, 0.24
```

The naive design scores 0.66 on labels that carry no information, so the observed 0.73 is unremarkable
(`p = 0.24`). Report the nested number; use the other two to show what the naive one is worth.
`nested_subset_search` also returns `selection_frequency`, the fraction of outer folds that kept each
feature — a feature chosen in a third of folds is not a finding.

The classifier is a linear SVM (`C=1.0`) on standardised features, with a leave-one-out outer loop. Both
searches are exhaustive over subsets up to `max_size` (all sizes when `None`), so cost grows as 2^p: cap
`max_size` beyond about 15 features.

## Tests

```bash
venv_py311/bin/python -m pytest tests/test_stats_*.py -q
```

The survival, `partial_correlation` and `dunn_posthoc` tests need `lifelines`, `pingouin` and
`scikit-posthocs` (`pip install -r requirements.txt`). Without them those tests fail on import
and the rest pass.
