"""Misalignment score, correction and conformal threshold on synthetic perturbations of the bench phone views.

Input: the photo-resolution run of render_phone_views.py (DATA, 4080 x 3072: four views, ground truth, mandible mesh in
the board frame), the views of the old bench scene. The mandible mesh is given in the board frame, so its model frame
is M itself and the true registration is the identity (T_M_B = I; this B is not the head frame of the bench scene).
Observed mask of each view: the mesh rendered by mesh_overlay.py at the true registration, its
outline moved by a smooth random field (NOISE_MM, correlation CORR_MM) as a stand-in for SAM. In a share DEFECT of the
samples one view also gets a segmentation defect (a disc cut out of the mask or added to it at its outline).
Drape samples (stress set): a drape covers the part of the object beyond a random plane (a share DRAPE_FRAC of the
vertices), the same in every view, and the correction does not know it (no ignore mask); they give the failures that
the conformal test needs.
Perturbations of the registration (stale registration T_M_B0 = P, true update P^-1):
- a1: rotation about the A1 ball centre, random axis, angle uniform up to MAX_DEG;
- rigid: rotation about the model centroid, angle uniform up to MAX_DEG, plus a translation uniform up to MAX_MM.
For every sample and view set (1 view: a random one; 2 views: 1+3 or 2+4; 4 views): score at the true registration
(pre-impact) and at the stale one, correction (correct.py) from the stale one.
Targets of the TRE: TARGETS vertices by farthest point sampling (stand-in for the clinical targets).
Samples with an even index are the validation half (tau), the odd ones the test half (detection).
The first TIMING samples of each perturbation run one at a time in this process: their times go in the tables; the
others run in parallel worker processes. Samples already in samples.csv are not run again.

Writes
- WORK/samples.csv (WORK: t14 in the work folder, environment variable IMPACTGUIDE_WORK): one row per sample and
  view set;
- 05_results/detection_synthetic.csv: samples without defects; tau (95th percentile of the pre-impact scores of the
  validation half), AUC, sensitivity and specificity on the test half, false alarms on its pre-impact images; for
  comparison the threshold on the 2 mm label (Youden, validation half) and its sensitivity and specificity;
- 05_results/correction_synthetic.csv: per perturbation and view set, samples without defects; kappa_95 is the kappa
  that would make the predicted 95% TRE cover 95% of them (indication only: kappa is calibrated on validation data);
- 05_results/conformal_synthetic.csv: all samples, drapes included; observed silent error rate against alpha over random
  calibration and test splits, with the score gate (tau) and with u alone, for the 2 mm limit and, since synthetic
  corrections never get there, for STRESS_MM (the guarantee holds for any limit).

    python 04_code/src/eval_correction_synthetic.py [--samples N] [--drape N] [--workers N] [--timing N] [--summary]
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conformal as cf        # noqa: E402
import correct as co          # noqa: E402
import mesh_overlay as mo     # noqa: E402
import score as sc            # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
DATA = WORK_ROOT / "synthetic_pose"
WORK = WORK_ROOT / "t14"
RESULTS = ROOT / "05_results"
A1_BALL_Z = 7.0                 # ball centre above the pad frame origin, P["a1"]["base"][1] in build_test_bench.py
NOISE_MM, CORR_MM = 0.3, 2.0    # outline noise of the observed masks
DEFECT = 0.3
DRAPE_FRAC = (0.2, 0.5)
MAX_DEG, MAX_MM = 5.0, 5.0
TARGETS = 64
LIMIT_MM = 2.0                  # registration lost above this mean TRE
STRESS_MM = 0.2                 # lower limit for the conformal test, where synthetic corrections do fail
ALPHAS = (0.02, 0.05, 0.1, 0.2)
SETS = {"1_view": 1, "2_views": 2, "4_views": 4}
SEED = 0

_ctx = {}


def load(data=DATA):
    gt = json.loads((data / "gt.json").read_text(encoding="utf-8"))
    m = np.load(data / "mandible_board.npz")
    wb = np.array(gt["world_from_board"])
    c = np.linalg.solve(wb, np.r_[np.array(gt["bench"]["pad_origin"]) + [0.0, 0.0, A1_BALL_Z], 1.0])[:3]
    return gt, m["vertices"], m["faces"], c


def farthest_points(X, n):
    i = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
    d = np.linalg.norm(X - X[i[0]], axis=1)
    for _ in range(n - 1):
        i.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(X - X[i[-1]], axis=1))
    return X[i]


def init(data=str(DATA)):
    cv2.setNumThreads(1)
    gt, v, f, c = load(Path(data))
    ren = mo.MeshRenderer(v, f)
    views = []
    for vw in gt["views"]:
        K, T = np.array(vw["K"]), np.array(vw["camera_from_board"])
        r = ren.render(K, None, T, gt["image_size"])
        views.append({"K": K, "T": T, "mask": r.mask, "faces": r.faces, "offset": r.offset,
                      "scale": sc.mm_per_px(K, T, v), "axis": T[2, :3]})
    _ctx.update(ren=ren, views=views, c=c, targets=farthest_points(v, TARGETS), size=tuple(gt["image_size"]))


def smooth_noise(shape, corr_px, rng):
    gh, gw = int(shape[0] / corr_px) + 4, int(shape[1] / corr_px) + 4
    z = rng.standard_normal((gh, gw)).astype(np.float32)
    f = cv2.resize(z, (int(gw * corr_px), int(gh * corr_px)), interpolation=cv2.INTER_CUBIC)[:shape[0], :shape[1]]
    return f / f.std()


def observed(mask, scale, rng, defect=None):
    """Mask with its outline moved by a smooth random field, and an optional defect ("missing" or "spurious")."""
    fd = co.Field(mask)
    d = fd.img[..., 0]
    n = smooth_noise(d.shape, CORR_MM / scale, rng) * (NOISE_MM / scale)
    out = np.zeros_like(mask)
    out[fd.y0:fd.y0 + d.shape[0], fd.x0:fd.x0 + d.shape[1]] = np.where(d + n < 0.5, 255, 0).astype(np.uint8)
    if defect:
        ys, xs = np.nonzero(sc.outline(out))
        if not len(xs):          # the noise left no outline: nothing to cut or to add
            return out
        k = rng.integers(len(xs))
        ys_m, xs_m = np.nonzero(out)
        size = max(np.ptp(xs_m), np.ptp(ys_m))
        frac = rng.uniform(0.10, 0.25) if defect == "missing" else rng.uniform(0.05, 0.15)
        disc = cv2.circle(np.zeros_like(out), (int(xs[k]), int(ys[k])), int(frac * size), 255, -1) > 0
        out[disc] = 0 if defect == "missing" else 255
    return out


def drape(views, ren, rng):
    """A drape over the part of the object beyond a random plane (a share DRAPE_FRAC of the vertices): per view the
    pixels whose visible triangle is beyond the plane, and the share."""
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    frac = rng.uniform(*DRAPE_FRAC)
    beyond = ren.v[ren.f].mean(1) @ n > np.quantile(ren.v @ n, 1 - frac)
    covered = []
    for v in views:
        m = np.zeros(v["mask"].shape, bool)
        (x0, y0), fb = v["offset"], v["faces"]
        m[y0:y0 + fb.shape[0], x0:x0 + fb.shape[1]] = (fb >= 0) & beyond[np.maximum(fb, 0)]
        covered.append(m)
    return covered, float(frac)


def perturbation(kind, rng, c, o):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    ang = rng.uniform(0, MAX_DEG)
    R = co.rodrigues(ax * np.radians(ang))
    P = np.eye(4)
    P[:3, :3] = R
    if kind == "a1":
        P[:3, 3] = (np.eye(3) - R) @ c
        return P, ang, 0.0
    t = rng.normal(size=3)
    t *= rng.uniform(0, MAX_MM) / np.linalg.norm(t)
    P[:3, 3] = (np.eye(3) - R) @ o + t
    return P, ang, float(np.linalg.norm(t))


def tre(T, X):
    return float(np.linalg.norm(X @ T[:3, :3].T + T[:3, 3] - X, axis=1).mean())


def run_sample(job):
    """All view sets of one sample. Truth T_M_B = I, so the error of an estimate T is T itself."""
    kind, i, draped = job
    ren, views, c, X = _ctx["ren"], _ctx["views"], _ctx["c"], _ctx["targets"]
    rng = np.random.default_rng([SEED, i, 0 if kind == "a1" else 1] + ([1] if draped else []))
    o = ren.v.mean(0)
    P, ang, tr = perturbation(kind, rng, c, o)
    if draped:
        covered, frac = drape(views, ren, rng)
        bad_view, defect = 4, "drape"
        obs = [np.where(cov, 0, observed(v["mask"], v["scale"], rng)).astype(np.uint8) for v, cov in zip(views, covered)]
    else:
        frac = 0.0
        bad_view = int(rng.integers(4)) if rng.uniform() < DEFECT else -1
        defect = ("missing" if rng.uniform() < 0.5 else "spurious") if bad_view >= 0 else ""
        obs = [observed(v["mask"], v["scale"], rng, defect if k == bad_view else None) for k, v in enumerate(views)]
    pre, stale = [], []
    for v, ob in zip(views, obs):
        pre.append(sc.view_score(ob, v["mask"], v["scale"])["score_mm"])
        T = v["T"] @ P
        stale.append(sc.view_score(ob, ren.render(v["K"], None, T, _ctx["size"]).mask, sc.mm_per_px(v["K"], T, ren.v))
                     ["score_mm"])
    one, pair = int(rng.integers(4)), [(0, 2), (1, 3)][int(rng.integers(2))]
    rows = []
    pbar = X.mean(0)
    for name, n in SETS.items():
        idx = [one] if n == 1 else list(pair) if n == 2 else [0, 1, 2, 3]
        cv = [co.View(views[k]["K"], views[k]["T"], obs[k]) for k in idx]
        res = co.correct(ren, cv, P, X, kind, center=c if kind == "a1" else None)
        T = res.T_M_B
        d = T[:3, :3] @ pbar + T[:3, 3] - pbar
        z = views[idx[0]]["axis"]
        rows.append({
            "kind": kind, "sample": i, "half": "validation" if i % 2 == 0 else "test", "set": name,
            "views": "+".join(str(k + 1) for k in idx), "defect": defect if draped or bad_view in idx else "",
            "occluded": frac, "angle_deg": ang, "trans_mm": tr, "tre_before_mm": tre(P, X),
            "score_pre_mm": float(np.mean([pre[k] for k in idx])), "score_before_mm": float(np.mean([stale[k] for k in idx])),
            "score_after_mm": res.score_mm, "tre_after_mm": tre(T, X),
            "e_r_deg": float(np.degrees(np.linalg.norm(co.rotvec(T[:3, :3])))), "e_t_mm": float(np.linalg.norm(d)),
            "e_t_axis_mm": float(abs(d @ z)), "e_t_plane_mm": float(np.linalg.norm(d - (d @ z) * z)),
            "tre_pred_mean_mm": res.tre_pred_mean, "tre_pred_95_mm": res.tre_pred_95, "sigma_mm": res.sigma_mm,
            "n_points": res.n_points, "renders": res.renders, "start": res.start, "time_s": res.time_s,
            "render_s": res.render_ms / 1e3, "sequential": 0})
    return rows


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    for r in rows:
        for k in ("sample", "n_points", "renders", "start", "sequential"):
            r[k] = int(r[k])
        for k, v in r.items():
            if k.endswith(("_mm", "_deg", "_s")):
                r[k] = float(v)
    clean = [r for r in rows if not r["defect"]]

    det = []
    for name in SETS:
        val = [r for r in clean if r["set"] == name and r["half"] == "validation"]
        test = [r for r in clean if r["set"] == name and r["half"] == "test"]
        tau = sc.threshold([r["score_pre_mm"] for r in val])
        s_test, y_test = [r["score_before_mm"] for r in test], [r["tre_before_mm"] > LIMIT_MM for r in test]
        d = sc.detection(s_test, y_test, tau)
        s_val, y_val = np.array([r["score_before_mm"] for r in val]), np.array([r["tre_before_mm"] > LIMIT_MM for r in val])
        if y_val.any() and (~y_val).any():
            youden = max(s_val, key=lambda t: np.mean(s_val[y_val] > t) + np.mean(s_val[~y_val] <= t))
            dy = sc.detection(s_test, y_test, youden)
        else:                    # the Youden index needs lost and valid registrations in the validation half
            youden, dy = float("nan"), {"sensitivity": float("nan"), "specificity": float("nan")}
        det.append({"key": name, "tau_mm": f"{tau:.3f}", "auc": f"{d['auc']:.3f}", "sensitivity": f"{d['sensitivity']:.3f}",
                    "specificity": f"{d['specificity']:.3f}", "n_lost": d["n_lost"], "n_valid": d["n_valid"],
                    "false_alarm_pre": f"{np.mean([r['score_pre_mm'] > tau for r in test]):.3f}",
                    "n_validation": len(val), "tau_youden_mm": f"{youden:.3f}",
                    "sensitivity_youden": f"{dy['sensitivity']:.3f}", "specificity_youden": f"{dy['specificity']:.3f}"})
    taus = {r["key"]: float(r["tau_mm"]) for r in det}
    table(RESULTS / "detection_synthetic.csv", det)

    cor = []
    for kind in ("a1", "rigid"):
        for name in SETS:
            g = [r for r in clean if r["kind"] == kind and r["set"] == name]
            seq = [r for r in g if r["sequential"]]
            a = lambda k: np.array([r[k] for r in g])
            ta, pred = a("tre_after_mm"), a("tre_pred_95_mm")
            cor.append({"key": f"{kind}_{name}", "n": len(g), "tre_before_median_mm": f"{np.median(a('tre_before_mm')):.2f}",
                        "tre_after_median_mm": f"{np.median(ta):.3f}", "tre_after_p95_mm": f"{np.percentile(ta, 95):.3f}",
                        "tre_after_max_mm": f"{ta.max():.3f}", "below_1mm": f"{np.mean(ta < 1):.3f}",
                        "below_2mm": f"{np.mean(ta < 2):.3f}", "e_r_median_deg": f"{np.median(a('e_r_deg')):.3f}",
                        "e_t_median_mm": f"{np.median(a('e_t_mm')):.3f}",
                        "e_t_axis_median_mm": f"{np.median(a('e_t_axis_mm')):.3f}",
                        "e_t_plane_median_mm": f"{np.median(a('e_t_plane_mm')):.3f}",
                        "tre_pred_mean_median_mm": f"{np.median(a('tre_pred_mean_mm')):.3f}",
                        "tre_pred_95_median_mm": f"{np.median(pred):.3f}", "coverage_95": f"{np.mean(ta <= pred):.3f}",
                        "kappa_95": f"{np.percentile(ta / pred, 95) ** 2:.1f}",
                        "accepted_tau": f"{np.mean(a('score_after_mm') <= taus[name]):.3f}",
                        "renders_median": f"{np.median(a('renders')):.0f}",
                        "time_median_s": f"{np.median([r['time_s'] for r in seq]):.1f}" if seq else "",
                        "render_share": f"{np.sum([r['render_s'] for r in seq]) / np.sum([r['time_s'] for r in seq]):.2f}"
                        if seq else "", "n_timed": len(seq)})
    table(RESULTS / "correction_synthetic.csv", cor)

    con = []
    combos = [("all", g, lim) for lim in (LIMIT_MM, STRESS_MM) for g in ("tau", "none")] + \
             [(name, "none", STRESS_MM) for name in SETS]
    for name, gname, lim in combos:
        g = [r for r in rows if name in ("all", r["set"])]
        a = lambda k: np.array([r[k] for r in g])
        fail = a("tre_after_mm") > lim
        gate = a("score_after_mm") <= np.array([taus[r["set"]] for r in g]) if gname == "tau" else np.ones(len(g), bool)
        for s in cf.splits(a("tre_pred_95_mm"), a("tre_after_mm"), gate, ALPHAS, seed=SEED, limit=lim):
            con.append({"key": f"{name}_{gname}_{lim:g}mm_alpha{round(100 * s['alpha']):02d}", "gate": gname,
                        "limit_mm": lim, "alpha": s["alpha"], "n": len(g),
                        "drape": int(sum(r["defect"] == "drape" for r in g)),
                        "failed": f"{fail.mean():.3f}", "silent_gate_only": f"{(gate & fail).mean():.3f}",
                        "silent_mean": f"{s['silent_mean']:.4f}", "silent_p05": f"{s['silent_p05']:.4f}",
                        "silent_p95": f"{s['silent_p95']:.4f}", "accepted_mean": f"{s['accepted_mean']:.3f}",
                        "accepted_gate_only": f"{gate.mean():.3f}", "u_median_mm": f"{s['u_median_mm']:.3f}",
                        "u_none": f"{s['u_none']:.3f}"})
    table(RESULTS / "conformal_synthetic.csv", con)
    return det, cor, con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100, help="samples per perturbation")
    ap.add_argument("--drape", type=int, default=60, help="drape samples per perturbation")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--timing", type=int, default=5, help="samples per perturbation run one at a time, for the times")
    ap.add_argument("--summary", action="store_true", help="only summarize the existing samples.csv")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "samples.csv"
    rows = []
    if out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            rows = [{**r, "occluded": r.get("occluded") or 0.0} for r in csv.DictReader(fh)]
    if not a.summary:
        t0 = time.perf_counter()
        done = {(r["kind"], int(r["sample"]), r["defect"] == "drape") for r in rows}
        jobs = [(k, i, False) for k in ("a1", "rigid") for i in range(a.samples)] + \
               [(k, i, True) for k in ("a1", "rigid") for i in range(a.drape)]
        jobs = [j for j in jobs if j not in done]
        timed = [j for j in jobs if j[1] < a.timing and not j[2]]
        init()
        cv2.setNumThreads(-1)
        for j in timed:
            for r in run_sample(j):
                r["sequential"] = 1
                rows.append(r)
        print(f"{len(timed)} samples one at a time: {time.perf_counter() - t0:.0f} s", flush=True)
        rest = [j for j in jobs if j not in timed]
        with ProcessPoolExecutor(a.workers, initializer=init) as ex:
            for k, rr in enumerate(ex.map(run_sample, rest, chunksize=1)):
                rows.extend(rr)
                if (k + 1) % 20 == 0:
                    print(f"{k + 1}/{len(rest)} {time.perf_counter() - t0:.0f} s", flush=True)
        table(out, rows)
        print(f"{len(rows)} rows in {out}, {time.perf_counter() - t0:.0f} s")
    det, cor, con = summarize(rows)
    for t in (det, cor, con):
        for r in t:
            print(r)


if __name__ == "__main__":
    main()
