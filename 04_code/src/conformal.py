"""Conformal calibration of the threshold u on the predicted TRE.

A correction is accepted when its final misalignment score is at most tau and its predicted 95% TRE is below u. Silent
error: an accepted correction whose true TRE is above LIMIT_MM. Loss of a calibration sample at u:
L_i(u) = 1[score_i <= tau and pred_i < u and tre_i > LIMIT_MM], bounded by 1 and non-decreasing in u. Conformal risk
control (Angelopoulos, Bates, Fisch, Lei and Schuster, "Conformal risk control", ICLR 2024) takes the largest u with

    (n R_n(u) + 1) / (n + 1) <= alpha,   R_n(u) = mean of L_i(u) over the n calibration samples,

which bounds the expected silent error rate of a new sample by alpha when calibration and test samples are
exchangeable (same protocol). With n < 1 / alpha - 1 no u qualifies and nothing is accepted.
"""
import numpy as np

LIMIT_MM = 2.0


def silent(pred, tre, gate, u, limit=LIMIT_MM):
    """Loss of each sample at u: accepted (gate and pred < u) with true TRE above limit."""
    return np.asarray(gate, bool) & (np.asarray(pred, float) < u) & (np.asarray(tre, float) > limit)


def calibrate(pred, tre, gate, alpha=0.05, limit=LIMIT_MM):
    """Largest u whose finite-sample corrected risk is at most alpha: -inf (accept nothing), a value, or +inf."""
    pred, tre, gate = np.asarray(pred, float), np.asarray(tre, float), np.asarray(gate, bool)
    n = len(pred)
    m = int(np.floor(alpha * (n + 1) - 1 + 1e-9))     # silent errors allowed in the calibration set
    if m < 0:
        return -np.inf
    bad = np.sort(pred[gate & (tre > limit)])
    return float(bad[m]) if len(bad) > m else np.inf


def splits(pred, tre, gate, alphas, n_splits=1000, cal_frac=0.5, seed=0, limit=LIMIT_MM):
    """Repeated random calibration and test splits: for each alpha the observed silent error rate and acceptance rate
    on the test part, mean and 5th-95th percentiles over the splits, and the median u."""
    pred, tre, gate = np.asarray(pred, float), np.asarray(tre, float), np.asarray(gate, bool)
    n = len(pred)
    n_cal = int(round(cal_frac * n))
    rng = np.random.default_rng(seed)
    out = {a: {"silent": [], "accepted": [], "u": []} for a in alphas}
    for _ in range(n_splits):
        idx = rng.permutation(n)
        cal, test = idx[:n_cal], idx[n_cal:]
        for a in alphas:
            u = calibrate(pred[cal], tre[cal], gate[cal], a, limit)
            out[a]["silent"].append(silent(pred[test], tre[test], gate[test], u, limit).mean())
            out[a]["accepted"].append((gate[test] & (pred[test] < u)).mean())
            out[a]["u"].append(u)
    rows = []
    for a in alphas:
        s, acc, u = (np.asarray(out[a][k]) for k in ("silent", "accepted", "u"))
        rows.append({"alpha": a, "n_cal": n_cal, "n_test": n - n_cal, "silent_mean": float(s.mean()),
                     "silent_p05": float(np.percentile(s, 5)), "silent_p95": float(np.percentile(s, 95)),
                     "accepted_mean": float(acc.mean()), "u_median_mm": float(np.median(u)),
                     "u_none": float(np.mean(~np.isfinite(u) & (u < 0)))})
    return rows
