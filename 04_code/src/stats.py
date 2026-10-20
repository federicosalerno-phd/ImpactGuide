"""Statistics of the impact trials, for every analysis that needs them (analysis.py, interfaces.py, the evaluation).

    median_iqr(v)                       median, first and third quartile
    bootstrap(v, stat)                  percentile interval of a statistic over resamples (trials drawn with replacement)
    holm(p)                             Holm's step-down adjustment of a family of p values
    wilcoxon_paired(x, y)               Wilcoxon signed-rank test of paired values
    auc(score, positive)                area under the ROC curve (Mann-Whitney U, ties one half)
    mixed_model(df, formula, groups)    linear mixed model with a random intercept per group (statsmodels), ordinary
                                        least squares when there is one group only; one row per term
    conformal_u(pred, tre, gate, alpha) the threshold u of the predicted TRE, by conformal risk control on the
                                        validation samples only (conformal.calibrate)
    guarantee(pred, tre, gate, u)       what u gives on the test samples: accepted share, silent errors observed
    calibration_bins(pred, tre, k)      predicted against observed TRE, by quantile bins of the predicted one
    risk_coverage(pred, tre, shares)    the error of the corrections accepted, the most certain first, against the
                                        share accepted

The values are plain floats; NaN where a statistic cannot be computed (too few values, no difference at all).

    python stats.py test        checks of every function; exit code 1 on failure
"""
from __future__ import annotations

import math
import sys
import warnings

import numpy as np

import conformal

BOOTSTRAP = 2000
SEED = 15
LIMIT_MM = conformal.LIMIT_MM       # a correction accepted with its true TRE over this is a silent error
MIN_PAIRS = 6                       # a paired test on fewer pairs is not made


def finite(v) -> np.ndarray:
    v = np.asarray(v, np.float64).ravel()
    return v[np.isfinite(v)]


def median_iqr(v) -> tuple[float, float, float]:
    v = finite(v)
    if not len(v):
        return math.nan, math.nan, math.nan
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return float(med), float(q1), float(q3)


def bootstrap(v, stat=np.median, n=BOOTSTRAP, seed=SEED, level=0.95) -> tuple[float, float]:
    """The percentile interval of stat(axis=1) over n resamples of v."""
    v = finite(v)
    if len(v) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = stat(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def holm(p) -> list[float]:
    """Holm's step-down adjustment; a NaN stays NaN and is left out of the family."""
    p = [float(x) for x in p]
    idx = [i for i, x in enumerate(p) if math.isfinite(x)]
    out = [math.nan] * len(p)
    m, run = len(idx), 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: p[i])):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out


def wilcoxon_paired(x, y) -> tuple[float, float]:
    """(W, p) of the two-sided Wilcoxon signed-rank test of x against y; NaN with fewer than MIN_PAIRS pairs or no
    difference at all."""
    from scipy import stats as sps
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < MIN_PAIRS or not np.any(x != y):
        return math.nan, math.nan
    try:
        r = sps.wilcoxon(x, y)
    except ValueError:
        return math.nan, math.nan
    return float(r.statistic), float(r.pvalue)


def auc(score, positive) -> float:
    """Area under the ROC curve of score for the positive cases: Mann-Whitney U over n1 n0, ties counted one half."""
    s = np.asarray(score, np.float64)
    y = np.asarray(positive, bool)
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return math.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mixed_model(df, formula: str, groups: str) -> tuple[list[dict], dict]:
    """A linear mixed model with a random intercept per value of the column groups (REML), or ordinary least
    squares when there is one group only. Returns ([{"term", "coef", "se", "ci_low", "ci_high", "p"}],
    {"method": "mixedlm" or "ols", "n", "groups", "converged"}); the variance of the groups is the term "Group Var"."""
    import statsmodels.formula.api as smf
    k = int(df[groups].nunique())
    info = {"n": int(len(df)), "groups": k, "converged": True}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if k >= 2:
            fit = smf.mixedlm(formula, df, groups=df[groups]).fit(reml=True)
            info["method"] = "mixedlm"
            info["converged"] = bool(getattr(fit, "converged", True))
        else:
            fit = smf.ols(formula, df).fit()
            info["method"] = "ols"
    if any("converge" in str(w.message).lower() for w in caught):
        info["converged"] = False
    ci = fit.conf_int()
    rows = [{"term": name, "coef": float(fit.params[name]), "se": float(fit.bse[name]),
             "ci_low": float(ci.loc[name].iloc[0]), "ci_high": float(ci.loc[name].iloc[1]),
             "p": float(fit.pvalues[name]) if name != "Group Var" else math.nan} for name in fit.params.index]
    return rows, info


def conformal_u(pred, tre, gate=None, alpha=0.05, limit=LIMIT_MM, groups=None) -> float:
    """The largest u of the predicted TRE (95%) whose expected rate of silent errors (accepted, true TRE over
    limit) stays under alpha, by conformal risk control on these samples, which must be the validation ones only:
    -inf accepts nothing, +inf accepts every correction through the gate."""
    pred = np.asarray(pred, np.float64)
    gate = np.ones(len(pred), bool) if gate is None else np.asarray(gate, bool)
    return conformal.calibrate(pred, np.asarray(tre, np.float64), gate, alpha, limit, groups)


def guarantee(pred, tre, gate, u, alpha=0.05, limit=LIMIT_MM) -> dict:
    """What a threshold u gives on test samples: n, accepted share, silent errors observed (share of all samples,
    as the guarantee bounds them, and share of the accepted), and whether the observed rate is within alpha."""
    pred, tre = np.asarray(pred, np.float64), np.asarray(tre, np.float64)
    gate = np.ones(len(pred), bool) if gate is None else np.asarray(gate, bool)
    n = len(pred)
    acc = gate & (pred < u)
    bad = conformal.silent(pred, tre, gate, u, limit)
    return {"n": n, "accepted": float(acc.mean()) if n else math.nan,
            "silent": float(bad.mean()) if n else math.nan,
            "silent_of_accepted": float(bad.sum() / acc.sum()) if acc.any() else math.nan,
            "within": bool(n and bad.mean() <= alpha)}


def calibration_bins(pred, tre, k: int = 5) -> list[dict]:
    """The calibration diagram: the samples in k bins of equal count by predicted TRE; for each bin the median
    predicted, the median and 95th percentile observed, and the share observed under the predicted value."""
    pred, tre = np.asarray(pred, np.float64), np.asarray(tre, np.float64)
    keep = np.isfinite(pred) & np.isfinite(tre)
    pred, tre = pred[keep], tre[keep]
    if len(pred) < k:
        return []
    order = np.argsort(pred, kind="stable")
    out = []
    for i, part in enumerate(np.array_split(order, k), 1):
        p, t = pred[part], tre[part]
        out.append({"bin": i, "n": int(len(part)), "pred_median_mm": float(np.median(p)),
                    "obs_median_mm": float(np.median(t)), "obs_p95_mm": float(np.quantile(t, 0.95)),
                    "covered": float(np.mean(t <= p))})
    return out


def risk_coverage(pred, tre, shares=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), limit=LIMIT_MM) -> list[dict]:
    """The error against the share accepted: the corrections ranked by predicted TRE, the most certain first; for
    each share, the median and 95th percentile TRE of those accepted and their share over limit."""
    pred, tre = np.asarray(pred, np.float64), np.asarray(tre, np.float64)
    keep = np.isfinite(pred) & np.isfinite(tre)
    pred, tre = pred[keep], tre[keep]
    if not len(pred):
        return []
    t = tre[np.argsort(pred, kind="stable")]
    out = []
    for s in shares:
        m = max(1, int(round(s * len(t))))
        out.append({"share": float(s), "n": m, "tre_median_mm": float(np.median(t[:m])),
                    "tre_p95_mm": float(np.quantile(t[:m], 0.95)), "over_limit": float(np.mean(t[:m] > limit))})
    return out


def selftest() -> int:
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)
    h = holm([0.01, 0.04, 0.03, math.nan])
    expect(np.allclose(h[:3], [0.03, 0.06, 0.06]) and math.isnan(h[3]), f"Holm's adjustment ({h})")
    expect(abs(auc([1, 2, 3, 4], [0, 0, 1, 1]) - 1) < 1e-12 and abs(auc([1, 1, 1, 1], [0, 1, 0, 1]) - 0.5) < 1e-12
           and math.isnan(auc([1, 2], [1, 1])), "AUC, with ties and with one class only")
    rng = np.random.default_rng(0)
    x = rng.normal(1, 0.3, 40)
    w, p = wilcoxon_paired(x, x - 0.5)
    expect(p < 1e-6 and math.isnan(wilcoxon_paired(x, x)[1]) and math.isnan(wilcoxon_paired(x[:5], x[:5] + 1)[1]),
           "Wilcoxon: a clear shift found; no difference or too few pairs give no test")
    lo, hi = bootstrap(rng.normal(5, 1, 200))
    expect(lo < 5 < hi and hi - lo < 0.5, f"bootstrap interval of the median ({lo:.2f}, {hi:.2f})")
    import pandas as pd
    df = pd.DataFrame({"g": np.repeat(["a", "b", "c", "d"], 30), "e": np.tile(np.arange(30.0), 4)})
    df["y"] = 0.1 * df["e"] + df["g"].map({"a": 0, "b": 0.5, "c": -0.3, "d": 0.2}) + rng.normal(0, 0.1, len(df))
    rows, info = mixed_model(df, "y ~ e", "g")
    slope = next(r for r in rows if r["term"] == "e")
    expect(info["method"] == "mixedlm" and abs(slope["coef"] - 0.1) < 0.01 and slope["p"] < 1e-6,
           f"mixed model: slope {slope['coef']:.4f} over {info['groups']} groups")
    rows1, info1 = mixed_model(df[df.g == "a"], "y ~ e", "g")
    expect(info1["method"] == "ols" and len(rows1) == 2, "one group only: ordinary least squares")
    # conformal: on validation samples whose predicted TRE is honest, the guarantee holds on new ones
    def samples(n, seed):
        r = np.random.default_rng(seed)
        tre = np.abs(r.normal(0.8, 0.9, n))
        return tre * r.uniform(1.0, 2.5, n), tre
    seen = [guarantee(*samples(1000, 900 + s), None, conformal_u(*samples(300, 100 + s), alpha=0.05), alpha=0.05)
            for s in range(80)]
    silent_mean = float(np.mean([g["silent"] for g in seen]))
    accepted = float(np.mean([g["accepted"] for g in seen]))
    expect(silent_mean <= 0.05 + 0.005 and 0.3 < accepted < 1.0,
           f"conformal u from 80 validation sets of 300: {silent_mean:.3f} silent errors on average on new test "
           f"sets (alpha 0.05), {accepted:.2f} accepted")
    pt, tt = samples(4000, 2)
    bins = calibration_bins(pt, tt, 5)
    rc = risk_coverage(pt, tt)
    expect(len(bins) == 5 and bins[0]["pred_median_mm"] < bins[-1]["pred_median_mm"]
           and all(b["covered"] > 0.5 for b in bins) and rc[0]["tre_median_mm"] < rc[-1]["tre_median_mm"],
           "calibration bins rise with the prediction; the most certain corrections have the smaller error")
    print(f"{'FAILED: ' + str(len(fails)) if fails else 'all passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(selftest())
    print(__doc__)
