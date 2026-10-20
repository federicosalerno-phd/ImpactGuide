"""Conformal calibration of the threshold u on the predicted TRE.

A correction is accepted when its final misalignment score is at most tau and its predicted 95% TRE is below u. Silent
error: an accepted correction whose true TRE is above LIMIT_MM. Loss of a calibration sample at u:
L_i(u) = 1[score_i <= tau and pred_i < u and tre_i > LIMIT_MM], bounded by 1 and non-decreasing in u. Conformal risk
control (Angelopoulos, Bates, Fisch, Lei and Schuster, "Conformal risk control", ICLR 2024) takes the largest u with

    (n R_n(u) + 1) / (n + 1) <= alpha,   R_n(u) = mean of L_i(u) over the n calibration samples,

which bounds the expected silent error rate of a new sample by alpha when calibration and test samples are
exchangeable (same protocol). With n < 1 / alpha - 1 no u qualifies and nothing is accepted.

Groups: rows that are not exchangeable on their own (e.g. the view sets of one sample, which share its perturbation and
masks) form one sample whose loss is the mean of its rows' losses; n counts groups, and a split never puts the rows
of one group on both sides. Without groups every row is a sample.
"""
import numpy as np

LIMIT_MM = 2.0


def silent(pred, tre, gate, u, limit=LIMIT_MM):
    """Loss of each sample at u: accepted (gate and pred < u) with true TRE above limit."""
    return np.asarray(gate, bool) & (np.asarray(pred, float) < u) & (np.asarray(tre, float) > limit)


def _groups(n, groups):
    """Group index of each row (0..g-1, in order of first appearance) and the number of groups; every row its own
    group without groups (the same indices as groups of one row each)."""
    if groups is None:
        return np.arange(n), n
    _, first, inv = np.unique(np.asarray(groups), return_index=True, return_inverse=True)
    rank = np.empty(len(first), int)
    rank[np.argsort(first)] = np.arange(len(first))
    return rank[inv.ravel()], len(first)


def calibrate(pred, tre, gate, alpha=0.05, limit=LIMIT_MM, groups=None):
    """Largest u whose finite-sample corrected risk is at most alpha: -inf (accept nothing), a value, or +inf."""
    pred, tre, gate = np.asarray(pred, float), np.asarray(tre, float), np.asarray(gate, bool)
    g, n = _groups(len(pred), groups)
    budget = alpha * (n + 1) - 1          # sum of the sample losses allowed in the calibration set
    if budget < -1e-9:
        return -np.inf
    bad = np.flatnonzero(gate & (tre > limit))
    if not len(bad):
        return np.inf
    bad = bad[np.argsort(pred[bad], kind="stable")]
    size = np.bincount(g, minlength=n)
    loss = np.cumsum(1.0 / size[g[bad]])  # risk sum once u passes each bad prediction, in order
    over = np.flatnonzero(loss > budget + 1e-9)
    return float(pred[bad[over[0]]]) if len(over) else np.inf


def splits(pred, tre, gate, alphas, n_splits=1000, cal_frac=0.5, seed=0, limit=LIMIT_MM, groups=None):
    """Repeated random calibration and test splits, by group: for each alpha the observed silent error rate and
    acceptance rate on the test part (means over its samples), mean and 5th-95th percentiles over the splits, and the
    median u."""
    pred, tre, gate = np.asarray(pred, float), np.asarray(tre, float), np.asarray(gate, bool)
    g, n = _groups(len(pred), groups)
    n_cal = int(round(cal_frac * n))
    size = np.bincount(g, minlength=n)
    rng = np.random.default_rng(seed)
    out = {a: {"silent": [], "accepted": [], "u": []} for a in alphas}

    def mean_of_groups(x, rows):         # mean over the test samples of the mean over their rows
        return float(np.sum(x / size[g[rows]]) / len(np.unique(g[rows])))

    for _ in range(n_splits):
        in_cal = np.zeros(n, bool)
        in_cal[rng.permutation(n)[:n_cal]] = True
        cal, test = np.flatnonzero(in_cal[g]), np.flatnonzero(~in_cal[g])
        gc = g[cal]
        for a in alphas:
            u = calibrate(pred[cal], tre[cal], gate[cal], a, limit, None if groups is None else gc)
            out[a]["silent"].append(mean_of_groups(silent(pred[test], tre[test], gate[test], u, limit), test))
            out[a]["accepted"].append(mean_of_groups(gate[test] & (pred[test] < u), test))
            out[a]["u"].append(u)
    rows = []
    for a in alphas:
        s, acc, u = (np.asarray(out[a][k]) for k in ("silent", "accepted", "u"))
        rows.append({"alpha": a, "n_cal": n_cal, "n_test": n - n_cal, "silent_mean": float(s.mean()),
                     "silent_p05": float(np.percentile(s, 5)), "silent_p95": float(np.percentile(s, 95)),
                     "accepted_mean": float(acc.mean()), "u_median_mm": float(np.median(u)),
                     "u_none": float(np.mean(~np.isfinite(u) & (u < 0)))})
    return rows
