"""Survival analysis: KM, log-rank, Cox PH; interval censoring, competing risks, illness-death."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _sps

_Z95 = float(_sps.norm.ppf(0.975))
_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)  # nodes/weights on [-1, 1]


def kaplan_meier(df: pd.DataFrame, time: str, event: str,
                 group: Optional[str] = None) -> Dict[str, Any]:
    """Kaplan-Meier survival estimates, optionally stratified by ``group``."""
    from lifelines import KaplanMeierFitter

    cols = [time, event] + ([group] if group else [])
    clean = df[cols].dropna()
    timeline = np.linspace(0, float(clean[time].max()), 100)

    curves: Dict[str, List[float]] = {}
    ci_lo: Dict[str, List[float]] = {}
    ci_hi: Dict[str, List[float]] = {}

    if group:
        for lvl, g in clean.groupby(group):
            km = KaplanMeierFitter().fit(g[time], g[event], timeline=timeline)
            curves[str(lvl)] = km.survival_function_.iloc[:, 0].tolist()
            ci = km.confidence_interval_
            ci_lo[str(lvl)] = ci.iloc[:, 0].tolist()
            ci_hi[str(lvl)] = ci.iloc[:, 1].tolist()
    else:
        km = KaplanMeierFitter().fit(clean[time], clean[event], timeline=timeline)
        curves["_overall"] = km.survival_function_.iloc[:, 0].tolist()
        ci = km.confidence_interval_
        ci_lo["_overall"] = ci.iloc[:, 0].tolist()
        ci_hi["_overall"] = ci.iloc[:, 1].tolist()

    return {
        "timeline": timeline.tolist(),
        "survival": curves,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
    }


def logrank_test(df: pd.DataFrame, time: str, event: str, group: str) -> Dict[str, Any]:
    """Log-rank test comparing survival across groups."""
    from lifelines.statistics import logrank_test as _lr, multivariate_logrank_test as _mlr
    clean = df[[time, event, group]].dropna()
    levels = clean[group].unique()
    if len(levels) == 2:
        a = clean[clean[group] == levels[0]]
        b = clean[clean[group] == levels[1]]
        res = _lr(a[time], b[time], a[event], b[event])
    else:
        res = _mlr(clean[time], clean[group], event_observed=clean[event])
    return {
        "test": "logrank",
        "statistic": float(res.test_statistic),
        "p_value": float(res.p_value),
        "n_groups": int(len(levels)),
    }


def cox_regression(df: pd.DataFrame, time: str, event: str,
                   covariates: List[str]) -> Dict[str, Any]:
    """Cox proportional-hazards regression with Schoenfeld PH diagnostics."""
    from lifelines import CoxPHFitter

    clean = df[[time, event, *covariates]].dropna().copy()
    cph = CoxPHFitter().fit(clean, duration_col=time, event_col=event)

    rows: List[Dict[str, Any]] = []
    for cov in covariates:
        beta = float(cph.params_[cov])
        se = float(cph.standard_errors_[cov])
        rows.append({
            "predictor": cov,
            "coef": beta,
            "hazard_ratio": float(np.exp(beta)),
            "se": se,
            "z_statistic": beta / se if se > 0 else float("nan"),
            "p_value": float(cph.summary.loc[cov, "p"]),
            "hr_ci_lower": float(np.exp(cph.confidence_intervals_.loc[cov, "95% lower-bound"])),
            "hr_ci_upper": float(np.exp(cph.confidence_intervals_.loc[cov, "95% upper-bound"])),
        })

    # Schoenfeld PH test per covariate. lifelines can fail here (e.g. perfectly separating
    # predictors); the fit is still reported, but the failure is surfaced, never swallowed:
    # an empty ph_test must not read as "assumption holds".
    ph_rows: List[Dict[str, Any]] = []
    ph_error: Optional[str] = None
    try:
        from lifelines.statistics import proportional_hazard_test
        ph_res = proportional_hazard_test(cph, clean, time_transform="rank")
        for cov in covariates:
            if cov in ph_res.summary.index:
                ph_rows.append({
                    "predictor": cov,
                    "test_statistic": float(ph_res.summary.loc[cov, "test_statistic"]),
                    "p_value": float(ph_res.summary.loc[cov, "p"]),
                    "violates_ph": bool(ph_res.summary.loc[cov, "p"] < 0.05),
                })
    except Exception as exc:
        import warnings
        ph_rows = []
        ph_error = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"Proportional-hazards test failed: {ph_error}", RuntimeWarning, stacklevel=2)

    return {
        "model": "cox_ph",
        "n": int(len(clean)),
        "n_events": int(clean[event].sum()),
        "coefficients": rows,
        "concordance": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_),
        "ph_test": ph_rows,
        "ph_test_error": ph_error,
    }


# --- Interval censoring -----------------------------------------------------------------
# A visit-based outcome is interval-censored: it happened after the last visit it was absent
# (left) and at or before the first visit it was present (right). right = inf/NaN means it was
# never seen (right-censored at left); left == right is an exactly known time.

def _bounds(df: pd.DataFrame, left: str, right: str):
    L = df[left].to_numpy(float)
    R = df[right].fillna(np.inf).to_numpy(float)
    if (R < L).any():
        raise ValueError(f"{right} < {left} in {int((R < L).sum())} rows")
    return L, R


def turnbull(df: pd.DataFrame, left: str, right: str, tol: float = 1e-6,
             max_iter: int = 100_000) -> Dict[str, Any]:
    """Turnbull NPMLE of S(t) for interval-censored times (left, right].

    Only the mass of each Turnbull interval (lower, upper] is identified, not where inside it
    falls, so ``survival[j]`` is S(t) from ``upper[j]`` up to ``lower[j+1]``. EM stops when no
    mass moves by ``tol`` in an iteration (EM is slow: 1e-8 takes ~8x the iterations of
    1e-6 and moved S by < 1e-4 in testing). Not lifelines'
    ``fit_interval_censoring``: that treats intervals as closed [L, R], so on a visit grid it
    puts mass on visit dates shared by adjacent intervals and the curve comes out wrong.
    """
    L, R = _bounds(df.dropna(subset=[left]), left, right)
    exact = L == R
    # Innermost intervals: a left end directly followed by a right end. At a tie a right end
    # sorts first ((L, R] excludes L); an exact time t enters as the left end t- (sorts first).
    ends = sorted({(v, 1) for v in L[~exact]} | {(v, -1) for v in L[exact]} | {(v, 0) for v in R})
    iv = [(a, b, ka < 0) for (a, ka), (b, kb) in zip(ends, ends[1:]) if ka != 0 and kb == 0]
    lo, up, point = (np.array(x) for x in zip(*iv))
    A = ((((L[:, None] < lo) | ((L[:, None] == lo) & ~point)) & (up <= R[:, None]))
         | (exact[:, None] & point & (lo == L[:, None]))).astype(float)
    s = np.full(len(lo), 1 / len(lo))
    for it in range(1, max_iter + 1):  # EM / self-consistency
        new = s * (A / (A @ s)[:, None]).mean(0)
        done = np.abs(new - s).max() < tol
        s = new
        if done:
            break
    return {
        "lower": lo.tolist(), "upper": up.tolist(), "mass": s.tolist(),
        "survival": (1 - np.cumsum(s)).clip(0, 1).tolist(),
        "n": int(len(L)), "iterations": it, "converged": bool(done),
    }


def interval_censored_regression(df: pd.DataFrame, left: str, right: str,
                                 covariates: List[str], dist: str = "weibull") -> Dict[str, Any]:
    """Parametric AFT regression for interval-censored times (left, right] (lifelines).

    ``dist`` is "weibull", "lognormal" or "loglogistic". ``time_ratio`` = exp(coef) multiplies
    the event time. Weibull is also a proportional-hazards model, so it adds
    ``hazard_ratio`` = exp(-coef * shape) with a delta-method CI.
    """
    from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter

    fitter = {"weibull": WeibullAFTFitter, "lognormal": LogNormalAFTFitter,
              "loglogistic": LogLogisticAFTFitter}[dist]()
    d = df[[left, right, *covariates]].dropna(subset=[left, *covariates]).copy()
    L, R = _bounds(d, left, right)
    # log-time families are undefined at 0, and S(1e-9) is 1 for any plausible fit
    d["_lo"], d["_hi"] = np.maximum(L, 1e-9), np.maximum(R, 1e-9)
    fitter.fit_interval_censoring(d[["_lo", "_hi", *covariates]], "_lo", "_hi")
    p, a = fitter._primary_parameter_name, fitter._ancillary_parameter_name
    rows: List[Dict[str, Any]] = []
    for cov in covariates:
        b, se = float(fitter.params_[(p, cov)]), float(fitter.standard_errors_[(p, cov)])
        row = {"predictor": cov, "coef": b, "se": se,
               "p_value": float(fitter.summary.loc[(p, cov), "p"]),
               "time_ratio": float(np.exp(b)), "tr_ci_lower": float(np.exp(b - _Z95 * se)),
               "tr_ci_upper": float(np.exp(b + _Z95 * se))}
        if dist == "weibull":
            k = float(np.exp(fitter.params_[(a, "Intercept")]))  # rho_ is log shape
            idx = [(p, cov), (a, "Intercept")]
            g = np.array([-k, -b * k])  # gradient of log HR = -coef * exp(log shape)
            s = float(np.sqrt(g @ fitter.variance_matrix_.loc[idx, idx].to_numpy() @ g))
            row.update(hazard_ratio=float(np.exp(-b * k)),
                       hr_ci_lower=float(np.exp(-b * k - _Z95 * s)),
                       hr_ci_upper=float(np.exp(-b * k + _Z95 * s)))
        rows.append(row)
    return {
        "model": f"{dist}_aft_interval",
        "n": int(len(d)),
        "n_right_censored": int(np.isinf(R).sum()),
        "n_exact": int((L == R).sum()),
        "coefficients": rows,
        "log_likelihood": float(fitter.log_likelihood_),
        "aic": float(fitter.AIC_),
    }


def visit_intervals(visits: pd.DataFrame, subject: str, time: str, diagnosed: str,
                    time_upper: Optional[str] = None) -> pd.DataFrame:
    """One row per subject: the interval (left, right] holding diagnosis, from a long visit table.

    ``left`` is the last undiagnosed visit before the first diagnosed one, ``right`` the
    earliest a diagnosed visit can have been, inf if never diagnosed. Diagnosis is absorbing:
    undiagnosed visits after it are flagged ``reverted``, not used. ``prevalent`` (diagnosed
    at the first visit) gets left = NaN. Masked dates: put the window a visit can lie in in
    [time, time_upper]; undiagnosed visits use its start and diagnosed ones its end, so the
    interval always holds the truth. Rows without a time or status are dropped; a subject
    whose only diagnosed visits are undated is flagged ``undated_diagnosis`` (it looks
    undiagnosed). Datetimes become years since the subject's first visit.
    """
    v = pd.DataFrame({"_id": visits[subject], "_lo": visits[time], "_dx": visits[diagnosed],
                      "_hi": visits[time_upper] if time_upper else visits[time]})
    undated = set(v.loc[v["_lo"].isna() & (v["_dx"] == 1), "_id"])
    v = v.dropna(subset=["_lo", "_dx"])
    v["_hi"] = v["_hi"].fillna(v["_lo"])
    if pd.api.types.is_datetime64_any_dtype(v["_lo"]):
        t0 = v.groupby("_id")["_lo"].transform("min")
        v["_lo"], v["_hi"] = ((v[c] - t0).dt.days / 365.25 for c in ("_lo", "_hi"))
    rows = []
    for sid, g in v.sort_values("_lo").groupby("_id", sort=False):
        dx = g["_dx"].to_numpy().astype(bool)
        first = int(np.argmax(dx)) if dx.any() else len(g)
        before = g["_lo"].to_numpy()[:first]
        rows.append({
            subject: sid,
            "left": float(before.max()) if len(before) else np.nan,
            "right": float(g["_hi"].to_numpy()[dx].min()) if dx.any() else np.inf,
            "last_visit": float(g["_lo"].max()),
            "n_visits": int(len(g)),
            "prevalent": bool(dx.any() and first == 0),
            "reverted": bool((~dx[first:]).any()),
            "undated_diagnosis": bool(sid in undated and not dx.any()),
        })
    return pd.DataFrame(rows)


# --- Competing risks --------------------------------------------------------------------
# event_type: 0 = censored, otherwise an integer cause code (e.g. 1 = PD, 2 = death, 3 = DLB).

def cumulative_incidence(df: pd.DataFrame, time: str, event_type: str,
                         group: Optional[str] = None) -> Dict[str, Any]:
    """Aalen-Johansen cumulative incidence of each cause, optionally by ``group``.

    1 - KM with competing events censored overstates risk; these curves and overall survival
    sum to 1. Ties are handled exactly (no jitter, unlike lifelines' AalenJohansenFitter).
    CIs: delta-method variance (Collett, Modelling Survival Data 3rd ed. p. 411; the formula
    lifelines uses) on the log(-log) scale.
    """
    cols = [time, event_type] + ([group] if group else [])
    clean = df[cols].dropna()
    clean = clean.assign(**{event_type: clean[event_type].astype(int)})
    causes = sorted(int(c) for c in clean[event_type].unique() if c != 0)
    timeline = np.linspace(0, float(clean[time].max()), 100)
    cs = np.cumsum
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {"cif": {}, "ci_lower": {}, "ci_upper": {}}
    for lvl, g in (clean.groupby(group) if group else [("_overall", clean)]):
        t, e = g[time].to_numpy(float), g[event_type].to_numpy()
        times, d = np.unique(t[e != 0], return_counts=True)
        n = len(t) - np.searchsorted(np.sort(t), times)          # at risk just before each time
        S = np.r_[1.0, np.cumprod(1 - d / n)][:-1]              # overall survival S(t-)
        a = np.divide(d, n * (n - d), out=np.zeros(len(d)), where=n > d)
        at = np.searchsorted(times, timeline, "right") - 1
        for key in out:
            out[key][str(lvl)] = {}
        for k in causes:
            tk = np.sort(t[e == k])
            dk = np.searchsorted(tk, times, "right") - np.searchsorted(tk, times, "left")
            F = cs(S * dk / n)
            c = S * dk / n ** 2
            var = (F ** 2 * cs(a) - 2 * F * cs(F * a) + cs(F ** 2 * a)
                   + cs(S ** 2 * dk * (n - dk) / n ** 3) - 2 * (F * cs(c) - cs(F * c)))
            with np.errstate(divide="ignore", invalid="ignore"):
                w = _Z95 * np.sqrt(np.maximum(var, 0)) / (F * np.abs(np.log(F)))
                lo, hi = np.where(F > 0, F ** np.exp(w), 0.0), np.where(F > 0, F ** np.exp(-w), 0.0)
            for key, val in zip(out, (F, lo, hi)):
                out[key][str(lvl)][str(k)] = (np.where(at >= 0, val[np.maximum(at, 0)], 0.0).tolist()
                                              if len(val) else [0.0] * len(timeline))
    return {"timeline": timeline.tolist(), "causes": causes, **out}


def cause_specific_cox(df: pd.DataFrame, time: str, event_type: str, cause: int,
                       covariates: List[str]) -> Dict[str, Any]:
    """Cox model for the hazard of one cause, other causes censored at their event time.

    Answers "does x act on the rate of this event among those still event-free"; for
    "does x change who ends up with this event" use ``fine_gray``.
    """
    ev = (df[event_type] == cause).astype(float).where(df[event_type].notna())
    res = cox_regression(df.assign(_event=ev), time, "_event", covariates)
    return {**res, "model": "cause_specific_cox", "cause": cause}


def fine_gray(df: pd.DataFrame, time: str, event_type: str, cause: int,
              covariates: List[str]) -> Dict[str, Any]:
    """Fine-Gray subdistribution-hazard regression with robust (sandwich) SEs.

    IPCW-weighted Cox (Geskus 2011, Biometrics 67:39): after a competing event a subject
    stays in the risk set with weight G(t-)/G(T_i-), G the censoring KM (left limits and
    Breslow ties, as cmprsk::crr), so point estimates match crr. The weights sit in an
    n x (cause event times) matrix whose competing-subject rows are exactly Geskus's expanded
    rows; lifelines on the long data gives the same fit but its robust SE is a Python loop
    over rows (64 s at n = 2000). The sandwich ignores the estimation of G, which Geskus
    found negligible; crr's SE adds that term.
    """
    d = df[[time, event_type, *covariates]].dropna()
    t, e, X = d[time].to_numpy(float), d[event_type].to_numpy(), d[covariates].to_numpy(float)
    cu, dc = np.unique(t[e == 0], return_counts=True)
    G = np.r_[1.0, np.cumprod(1 - dc / (len(t) - np.searchsorted(np.sort(t), cu)))]

    def G_minus(x):
        return G[np.searchsorted(cu, x)]

    te, dk = np.unique(t[e == cause], return_counts=True)
    comp = (e != 0) & (e != cause)
    # ponytail: dense n x m weights; chunk over event times if n * m outgrows memory
    W = (t[:, None] >= te).astype(float)
    W[comp] = np.where(t[comp, None] < te, G_minus(te) / G_minus(t[comp])[:, None], W[comp])
    ev = e == cause

    def parts(beta):  # weighted risk sets, Breslow ties
        r = W * np.exp(X @ beta)[:, None]
        S0 = r.sum(0)
        xbar = (r.T @ X) / S0[:, None]
        c = r @ (dk / S0)  # each subject's weighted cumulative baseline hazard
        return r, S0, xbar, c, X.T @ (X * c[:, None]) - (xbar.T * dk) @ xbar

    beta = np.zeros(X.shape[1])
    for _ in range(50):  # Newton-Raphson
        r, S0, xbar, c, info = parts(beta)
        step = np.linalg.solve(info, X[ev].sum(0) - dk @ xbar)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    r, S0, xbar, c, info = parts(beta)
    resid = (r * (dk / S0)) @ xbar - X * c[:, None]  # score residuals (Lin & Wei 1989)
    resid[ev] += X[ev] - xbar[np.searchsorted(te, t[ev])]
    inv = np.linalg.inv(info)
    se = np.sqrt(np.diag(inv @ (resid.T @ resid) @ inv))
    rows: List[Dict[str, Any]] = []
    for cov, b, s in zip(covariates, beta, se):
        rows.append({
            "predictor": cov, "coef": float(b), "hazard_ratio": float(np.exp(b)), "se": float(s),
            "z_statistic": float(b / s), "p_value": float(2 * _sps.norm.sf(abs(b / s))),
            "hr_ci_lower": float(np.exp(b - _Z95 * s)), "hr_ci_upper": float(np.exp(b + _Z95 * s)),
        })
    return {
        "model": "fine_gray", "cause": cause, "n": int(len(d)),
        "n_events": int(ev.sum()), "n_competing": int(comp.sum()), "coefficients": rows,
    }


# --- Illness-death model ----------------------------------------------------------------
# States 0 healthy, 1 PD, 2 dead; Markov intensities on time since baseline. Following
# Touraine, Gerds & Joly (2017, J Stat Softw 79(7); SmoothHazard) and Leffondre et al.
# (2013, Int J Epidemiol 42:1177): a subject who dies undiagnosed may have developed PD
# between the last visit and death, so the likelihood integrates over that possibility.

def _hazard(kind: str, theta: np.ndarray, breaks: np.ndarray):
    """(intensity, cumulative intensity) for one transition."""
    if kind == "weibull":
        k, s = np.exp(theta)
        return (lambda t: k / s * (t / s) ** (k - 1)), (lambda t: (t / s) ** k)
    rate = np.exp(theta)
    width = np.diff(np.r_[breaks, np.inf])
    return (lambda t: rate[np.searchsorted(breaks, t, "right") - 1],
            lambda t: (np.clip(np.asarray(t)[..., None] - breaks, 0, width) * rate).sum(-1))


def illness_death(df: pd.DataFrame, left: str, right: str, time: str, death: str,
                  covariates: Sequence[str] = (), death_covariates: Sequence[str] = (),
                  hazard: str = "weibull", breaks: Optional[Sequence[float]] = None,
                  profile: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Interval-censored illness-death model (healthy -> PD, healthy -> dead, PD -> dead) by ML.

    ``left``/``right``: last visit without PD and first visit with PD (right = inf/NaN if
    never diagnosed, left == right for an exact date). ``time``: death or last contact,
    ``death``: 1 if dead at ``time``. Everyone is healthy at time 0 (prevalent cases have
    no ``left`` and are dropped). ``covariates`` act on healthy -> PD; ``death_covariates``
    get separate coefficients on healthy -> dead and PD -> dead. Intensities are Weibull or
    piecewise-constant on ``breaks`` (start at 0, last piece open-ended). SEs from the
    inverse numerical Hessian. ``absolute_risk`` is P(PD by t), deaths accounted for, for
    covariates at ``profile`` (default 0); its CI simulates 500 parameter draws from their
    asymptotic normal distribution, as SmoothHazard does.
    """
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_hess3

    if hazard not in ("weibull", "piecewise"):
        raise ValueError(f"hazard must be 'weibull' or 'piecewise', not {hazard!r}")
    if hazard == "piecewise" and (breaks is None or breaks[0] != 0 or np.any(np.diff(breaks) <= 0)):
        raise ValueError("piecewise needs increasing breaks starting at 0")
    covariates, death_covariates = list(covariates), list(death_covariates)
    allc = list(dict.fromkeys(covariates + death_covariates))
    d = df[[left, right, time, death, *allc]].dropna(subset=[left, time, death, *allc])
    L, R = _bounds(d, left, right)
    T, dead = d[time].to_numpy(float), d[death].to_numpy().astype(bool)
    diag = np.isfinite(R)
    if (T < L).any() or (diag & (T < R)).any():
        raise ValueError(f"{time} must be >= {left}, and >= {right} when diagnosed")
    X1, XD = d[covariates].to_numpy(float), d[death_covariates].to_numpy(float)
    exact, U = diag & (R == L), np.where(diag, R, T)
    brk = np.asarray(breaks if hazard == "piecewise" else [0.0], float)
    nb = 2 if hazard == "weibull" else len(brk)
    p1, pd_ = len(covariates), len(death_covariates)

    def unpack(par):
        th, b01, b02, b12 = np.split(par, np.cumsum([3 * nb, p1, pd_]))
        return [_hazard(hazard, x, brk) for x in th.reshape(3, nb)], b01, b02, b12

    def nll(par):
        ((a01, A01), (a02, A02), (a12, A12)), b01, b02, b12 = unpack(par)
        e01, e02, e12 = (np.exp(X @ b)[:, None] for X, b in ((X1, b01), (XD, b02), (XD, b12)))
        A12T = A12(T)[:, None]

        def onset(u):  # density of PD onset at u, then alive with PD until T
            return np.exp(-A01(u) * e01 - A02(u) * e02 - (A12T - A12(u)) * e12) * a01(u) * e01

        with np.errstate(all="ignore"):  # Weibull shape < 1 is infinite at u = 0 on empty intervals
            # ponytail: the quadrature runs across piecewise-constant jumps; split at breaks if
            # piecewise fits ever need more than the ~1e-4 log-lik accuracy seen with Weibull
            u = L[:, None] + (U - L)[:, None] * (_GL_X + 1) / 2
            quad = np.where(U > L, (onset(u) * _GL_W).sum(1) * (U - L) / 2, 0.0)
            onset_lik = np.where(exact, onset(U[:, None])[:, 0], quad)
            h02 = np.where(dead, a02(T) * e02[:, 0], 1.0)
            h12 = np.where(dead, a12(T) * e12[:, 0], 1.0)
            healthy = np.exp(-A01(T) * e01[:, 0] - A02(T) * e02[:, 0]) * h02
            lik = np.where(diag, 0.0, healthy) + onset_lik * h12
        return -np.log(np.maximum(lik, 1e-300)).sum()

    n = len(d)
    crude = np.log([(diag.sum() + 1) / T.sum(), ((dead & ~diag).sum() + 1) / T.sum(),
                    ((dead & diag).sum() + 1) / max((T - R)[diag].sum(), T.sum() / n)])
    th0 = [[0.0, -r] for r in crude] if hazard == "weibull" else [[r] * nb for r in crude]
    x0 = np.r_[np.ravel(th0), np.zeros(p1 + 2 * pd_)]
    res = minimize(lambda p: nll(p) / n, x0, method="BFGS", options={"gtol": 1e-6})
    H = approx_hess3(res.x, nll)
    V = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(V), 0, None))

    rows: List[Dict[str, Any]] = []
    names = ([("01", c) for c in covariates] + [("02", c) for c in death_covariates]
             + [("12", c) for c in death_covariates])
    for i, (tr, c) in enumerate(names, start=3 * nb):
        b, s = float(res.x[i]), float(se[i])
        rows.append({"transition": tr, "predictor": c, "coef": b, "se": s,
                     "hazard_ratio": float(np.exp(b)),
                     "hr_ci_lower": float(np.exp(b - _Z95 * s)),
                     "hr_ci_upper": float(np.exp(b + _Z95 * s)),
                     "p_value": float(2 * _sps.norm.sf(abs(b / s))) if s > 0 else float("nan")})
    th = res.x[:3 * nb].reshape(3, nb)
    baseline = {tr: (dict(zip(("shape", "scale"), np.exp(th[j]).tolist())) if hazard == "weibull"
                     else {"breaks": brk.tolist(), "rates": np.exp(th[j]).tolist()})
                for j, tr in enumerate(("01", "02", "12"))}

    prof = profile or {}
    z1 = np.array([prof.get(c, 0.0) for c in covariates], float)
    zd = np.array([prof.get(c, 0.0) for c in death_covariates], float)
    timeline = np.linspace(0, float(T.max()), 100)

    def risk(par):
        ((a01, A01), (a02, A02), _), b01, b02, _ = unpack(par)
        e01, e02 = np.exp(z1 @ b01), np.exp(zd @ b02)
        u = timeline[:, None] * (_GL_X + 1) / 2
        with np.errstate(all="ignore"):
            f = np.exp(-A01(u) * e01 - A02(u) * e02) * a01(u) * e01
            return np.where(timeline > 0, (f * _GL_W).sum(1) * timeline / 2, 0.0)

    draws = np.random.default_rng(0).multivariate_normal(res.x, V, 500, method="eigh")
    ci = np.nanpercentile([risk(p) for p in draws], [2.5, 97.5], axis=0)
    return {
        "model": "illness_death", "hazard": hazard, "n": int(n),
        "n_diagnosed": int(diag.sum()), "n_deaths": int(dead.sum()),
        "n_deaths_after_diagnosis": int((dead & diag).sum()),
        "coefficients": rows, "baseline": baseline,
        "timeline": timeline.tolist(), "absolute_risk": risk(res.x).tolist(),
        "risk_ci_lower": ci[0].tolist(), "risk_ci_upper": ci[1].tolist(),
        "log_likelihood": float(-nll(res.x)), "aic": float(2 * nll(res.x) + 2 * len(res.x)),
        "converged": bool(res.success) and bool(np.all(np.linalg.eigvalsh(H) > 0)),
    }
