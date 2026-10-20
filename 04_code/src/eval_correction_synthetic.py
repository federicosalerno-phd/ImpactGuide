"""Misalignment score, correction and conformal threshold on synthetic perturbations of the bench phone views.

Input: the photo-resolution run of render_phone_views.py (DATA, 4080 x 3072: four views, ground truth, mandible mesh in
the board frame), the views of the old bench scene. The mandible mesh is given in the board frame, so its model frame
is M itself and the true registration is the identity (T_M_B = I; this B is not the head frame of the bench scene).
Observed mask of each view: the mesh rendered by mesh_overlay.py at the true registration, its
outline moved by a smooth random field (NOISE_MM, correlation CORR_MM) as a stand-in for SAM. In a share DEFECT of the
samples one view also gets a segmentation defect (a disc cut out of the mask or added to it at its outline).
Drape samples: a drape covers the part of the object beyond a random plane (a share DRAPE_FRAC of the vertices), the
same in every view; it is segmented by its colour in the trials, so here it is a known occluder (DRAPE_KNOWN: its
pixels, dilated, are the ignore mask of the score and of the correction; without it the pre-impact score rises above
tau and the detection raises false alarms).
Perturbations of the registration (stale registration T_M_B0 = P, true update P^-1):
- a1: rotation about the A1 ball centre, random axis, angle uniform up to MAX_DEG;
- rigid: rotation about the model centroid, angle uniform up to MAX_DEG, plus a translation uniform up to MAX_MM.
For every sample and view set (1 view: a random one; 2 views: 1+3 or 2+4; 4 views): score at the true registration
(pre-impact) and at the stale one, correction (correct.py) from the stale one (not for rigid with one view: never 6
degrees of freedom from one view, correct.MIN_VIEWS).
Targets of the TRE: TARGETS vertices by farthest point sampling (stand-in for the clinical targets).
Samples with an even index are the validation half (tau), the odd ones the test half (detection).
The first TIMING samples of each perturbation run one at a time in this process: their times go in the tables; the
others run in parallel worker processes. Samples already in samples.csv are not run again.

Writes
- WORK/samples.csv (WORK: t14 in the work folder, environment variable IMPACTGUIDE_WORK): one row per sample and
  view set;
- 05_results/detection_synthetic.csv: samples without defects; scores as score.py defines the alarm (trimmed mean,
  score.TRIM); tau the alarm's one threshold (score.THRESHOLD_FILE), and beside it (tau_val_mm) the same rule on the
  validation half: a lost registration (TRE before above 2 mm) flagged in at least SENSITIVITY of the cases; AUC,
  sensitivity and
  specificity on the test half, false alarms on its pre-impact images; the same for the threshold of a simple change
  (95th percentile of the pre-impact scores of the validation half) and for the Youden threshold (comparison); the
  drape samples apart (false alarms, sensitivity and specificity at tau);
- 05_results/correction_synthetic.csv: per perturbation and view set, samples without defects; kappa_95 is the kappa
  that would make the predicted 95% TRE cover 95% of them (indication only: kappa is calibrated on validation data);
- 05_results/conformal_synthetic.csv: all samples, drapes included, per view set and for the three together;
  observed silent error rate against alpha over random calibration and test splits by sample (the view sets of one
  sample never on both sides), with the score gate (tau; the samples that set tau left out) and with u alone, for the
  2 mm limit and, since synthetic corrections never get there, for STRESS_MM (the guarantee holds for any limit).

    python 04_code/src/eval_correction_synthetic.py [--samples N] [--drape N] [--workers N] [--timing N]
                                                    [--renderer gl|cpu] [--summary]
Renderer: gl_overlay.py on the GPU (default) or mesh_overlay.py on the CPU (the renderer of the runs before 20/10).
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
LIMIT_MM = sc.LOST_MM           # registration lost above this mean TRE (score.py: the one definition of the alarm)
SENSITIVITY = sc.SENSITIVITY    # tau: a lost registration is detected in at least this share of the validation half
DRAPE_KNOWN = True              # the drape is segmented by its colour and passed as a known occluder (ignore mask)
DILATE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))    # the occluders dilated by 3 px, as refiner.DILATE_PX
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


def renderer(v, f, kind="gl"):
    """Renderer with the interface of mesh_overlay.MeshRenderer: "gl" gl_overlay.py (GPU), "cpu" mesh_overlay.py."""
    if kind == "gl":
        import gl_overlay
        return gl_overlay.GLMeshRenderer(v, f)
    return mo.MeshRenderer(v, f)


def init(data=str(DATA), kind="gl"):
    cv2.setNumThreads(1)
    gt, v, f, c = load(Path(data))
    ren = renderer(v, f, kind)
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
    ignore = [None] * len(views)
    if draped:
        covered, frac = drape(views, ren, rng)
        bad_view, defect = 4, "drape"
        obs = [np.where(cov, 0, observed(v["mask"], v["scale"], rng)).astype(np.uint8) for v, cov in zip(views, covered)]
        if DRAPE_KNOWN:          # segmented by its colour: a known occluder, dilated as in refiner.ignore_mask
            ignore = [cv2.dilate(cov.astype(np.uint8) * 255, DILATE) for cov in covered]
    else:
        frac = 0.0
        bad_view = int(rng.integers(4)) if rng.uniform() < DEFECT else -1
        defect = ("missing" if rng.uniform() < 0.5 else "spurious") if bad_view >= 0 else ""
        obs = [observed(v["mask"], v["scale"], rng, defect if k == bad_view else None) for k, v in enumerate(views)]
    pre, stale = [], []
    for v, ob, ig in zip(views, obs, ignore):
        pre.append(sc.view_score(ob, v["mask"], v["scale"], ig, trim=sc.TRIM)["score_mm"])
        T = v["T"] @ P
        stale.append(sc.view_score(ob, ren.render(v["K"], None, T, _ctx["size"]).mask, sc.mm_per_px(v["K"], T, ren.v),
                                   ig, trim=sc.TRIM)["score_mm"])
    one, pair = int(rng.integers(4)), [(0, 2), (1, 3)][int(rng.integers(2))]
    rows = []
    pbar = X.mean(0)
    for name, n in SETS.items():
        idx = [one] if n == 1 else list(pair) if n == 2 else [0, 1, 2, 3]
        row = {"kind": kind, "sample": i, "half": "validation" if i % 2 == 0 else "test", "set": name,
               "views": "+".join(str(k + 1) for k in idx), "defect": defect if draped or bad_view in idx else "",
               "occluded": frac, "angle_deg": ang, "trans_mm": tr, "tre_before_mm": tre(P, X),
               "score_pre_mm": float(np.mean([pre[k] for k in idx])),
               "score_before_mm": float(np.mean([stale[k] for k in idx]))}
        if not co.allowed(kind, len(idx)):       # never 6 degrees of freedom from one view: detection only
            rows.append({**row, "sequential": 0})
            continue
        cv = [co.View(views[k]["K"], views[k]["T"], obs[k], None, ignore[k]) for k in idx]
        res = co.correct(ren, cv, P, X, kind, center=c if kind == "a1" else None)
        T = res.T_M_B
        d = T[:3, :3] @ pbar + T[:3, 3] - pbar
        z = views[idx[0]]["axis"]
        rows.append({
            **row, "score_after_mm": res.score_mm, "tre_after_mm": tre(T, X),
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


def holm(p):
    """Holm-Bonferroni adjusted p values of a family of tests (NaN: no test, left out of the family)."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok], kind="stable")]
    out[order] = np.minimum(1.0, np.maximum.accumulate((len(ok) - np.arange(len(ok))) * p[order]))
    return out


tau_lost = sc.tau_lost     # score.py holds the rule


def summarize(rows):
    for r in rows:
        for k in ("sample", "n_points", "renders", "start", "sequential"):
            r[k] = int(r[k]) if r.get(k) not in (None, "") else -1
        for k, v in list(r.items()):
            if k.endswith(("_mm", "_deg", "_s")):
                r[k] = float(v) if v not in (None, "") else float("nan")
    clean = [r for r in rows if not r["defect"]]
    done = lambda rr: [r for r in rr if np.isfinite(r.get("tre_after_mm", float("nan")))]   # corrected rows

    # detection per view set: tau from the validation half so that a lost registration (TRE before above LIMIT_MM)
    # is flagged in at least SENSITIVITY of the cases; the 95th percentile of the pre-impact scores stays as the
    # threshold of a simple change, the Youden threshold for comparison; drapes (known occluders) apart, all of them
    det = []
    for name in SETS:
        val = [r for r in clean if r["set"] == name and r["half"] == "validation"]
        test = [r for r in clean if r["set"] == name and r["half"] == "test"]
        dra = [r for r in rows if r["set"] == name and r["defect"] == "drape"]
        s_val, y_val = np.array([r["score_before_mm"] for r in val]), np.array([r["tre_before_mm"] > LIMIT_MM for r in val])
        tau_change = sc.threshold([r["score_pre_mm"] for r in val])
        # the alarm's one tau (score.THRESHOLD_FILE, chosen by its analysis); the same rule on this validation half
        # beside it, for comparison
        tau_val = sc.tau_lost(s_val, y_val)
        tau = sc.load_threshold()[0]
        s_test, y_test = [r["score_before_mm"] for r in test], [r["tre_before_mm"] > LIMIT_MM for r in test]
        d, dc = sc.detection(s_test, y_test, tau), sc.detection(s_test, y_test, tau_change)
        if y_val.any() and (~y_val).any():
            youden = max(s_val, key=lambda t: np.mean(s_val[y_val] > t) + np.mean(s_val[~y_val] <= t))
            dy = sc.detection(s_test, y_test, youden)
        else:                    # the Youden index needs lost and valid registrations in the validation half
            youden, dy = float("nan"), {"sensitivity": float("nan"), "specificity": float("nan")}
        dd = sc.detection([r["score_before_mm"] for r in dra], [r["tre_before_mm"] > LIMIT_MM for r in dra], tau)
        f3 = lambda x: f"{x:.3f}"
        dv = sc.detection(s_test, y_test, tau_val) if np.isfinite(tau_val) else {"sensitivity": float("nan"),
                                                                                    "specificity": float("nan")}
        det.append({"key": name, "tau_mm": f3(tau), "auc": f3(d["auc"]), "sensitivity": f3(d["sensitivity"]),
                    "tau_val_mm": f3(tau_val), "sensitivity_val_tau": f3(dv["sensitivity"]),
                    "specificity_val_tau": f3(dv["specificity"]),
                    "specificity": f3(d["specificity"]), "n_lost": d["n_lost"], "n_valid": d["n_valid"],
                    "false_alarm_pre": f3(np.mean([r["score_pre_mm"] > tau for r in test])),
                    "n_validation": len(val), "tau_change_mm": f3(tau_change),
                    "sensitivity_change": f3(dc["sensitivity"]), "specificity_change": f3(dc["specificity"]),
                    "false_alarm_pre_change": f3(np.mean([r["score_pre_mm"] > tau_change for r in test])),
                    "tau_youden_mm": f3(youden), "sensitivity_youden": f3(dy["sensitivity"]),
                    "specificity_youden": f3(dy["specificity"]), "drape_n": len(dra),
                    "drape_false_alarm_pre": f3(np.mean([r["score_pre_mm"] > tau for r in dra])) if dra else "",
                    "drape_false_alarm_pre_change": f3(np.mean([r["score_pre_mm"] > tau_change for r in dra]))
                    if dra else "",
                    "drape_sensitivity": f3(dd["sensitivity"]) if dra else "",
                    "drape_specificity": f3(dd["specificity"]) if dra else ""})
    taus = {r["key"]: float(r["tau_mm"]) for r in det}
    table(RESULTS / "detection_synthetic.csv", det)

    cor = []
    for kind in ("a1", "rigid"):
        for name in SETS:
            g = done([r for r in clean if r["kind"] == kind and r["set"] == name])
            if not g:            # not corrected (6 degrees of freedom from one view)
                continue
            seq = [r for r in g if r["sequential"] == 1]
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

    # conformal test per view set (and "all": the three view sets of a sample as one sample). A sample (kind, index,
    # drape) is one group: its rows share the perturbation and the masks, so a split keeps them on one side. With the
    # score gate the samples whose pre-impact scores set tau (validation half, no drape) are left out.
    con = []
    combos = [(name, g, lim) for name in ("all", *SETS) for lim in (LIMIT_MM, STRESS_MM) for g in ("tau", "none")]
    for name, gname, lim in combos:
        g = done([r for r in rows if name in ("all", r["set"])
                  and (gname == "none" or r["defect"] == "drape" or r["half"] == "test")])
        a = lambda k: np.array([r[k] for r in g])
        groups = [f"{r['kind']}/{r['sample']}/{int(r['defect'] == 'drape')}" for r in g]
        fail = a("tre_after_mm") > lim
        gate = a("score_after_mm") <= np.array([taus[r["set"]] for r in g]) if gname == "tau" else np.ones(len(g), bool)
        for s in cf.splits(a("tre_pred_95_mm"), a("tre_after_mm"), gate, ALPHAS, seed=SEED, limit=lim, groups=groups):
            con.append({"key": f"{name}_{gname}_{lim:g}mm_alpha{round(100 * s['alpha']):02d}", "gate": gname,
                        "limit_mm": lim, "alpha": s["alpha"], "n": len(g), "samples": len(set(groups)),
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
    ap.add_argument("--renderer", choices=("gl", "cpu"), default="gl", help="gl_overlay.py (GPU) or mesh_overlay.py")
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
        init(str(DATA), a.renderer)
        cv2.setNumThreads(-1)
        for j in timed:
            for r in run_sample(j):
                r["sequential"] = 1
                rows.append(r)
        print(f"{len(timed)} samples one at a time: {time.perf_counter() - t0:.0f} s", flush=True)
        rest = [j for j in jobs if j not in timed]
        with ProcessPoolExecutor(a.workers, initializer=init, initargs=(str(DATA), a.renderer)) as ex:
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
