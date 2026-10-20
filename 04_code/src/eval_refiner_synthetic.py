"""Optimisation only, network only and network then optimisation on the test split of the digital twin.

Test split of the twin dataset (6 sessions, 72 samples, never seen in training or early stopping). Per sample the 4
photos at half resolution (2040 x 1536) with their marker pose T_C_M and the true registration T_M_B (head frame B:
coordinates of the bench scene); observed masks: the SAM 2 masks of refiner.py prepare without the known covers and
occluders, which (dilated) are the ignore mask of correct.py.
Perturbations as in eval_correction_synthetic.py (its perturbation()): "a1" rotation about the ball centre up to 5 deg,
"rigid" rotation about the face centroid up to 5 deg plus translation up to 5 mm; stale registration T_M_B0 = T_M_B . P;
PERTURBATIONS per sample and kind. View sets as in eval_correction_synthetic.py: 1 view (a random photo), 2 views
(p1+p3 or p2+p4), 4 views.
Methods on the same views, masks and perturbations (rigid with one view: the network only, since the optimisation
never runs 6 degrees of freedom from one view, correct.MIN_VIEWS):
- opt: correct.correct from T_M_B0, multi-start (zero plus +-2.5 deg about the axes), as eval_correction_synthetic.py;
- net: refiner.Refiner.refine, ITERS iterations with a new render each;
- net_opt: the network, then refiner.optimize from its output (one start);
- net_prior: the same with the network's prediction as a Gaussian prior on the update, its predicted standard
  deviations scaled per mode on the validation split (sigma_scale) (extra).
Renderer: gl_overlay.py on the decimated face for both (masks and visibility buffers only for the optimisation).
TRE: mean error at the 7 clinical targets (role "proposed"). Time per correction: one at a time in this process, GPU
for the renders and the network, CPU for the rest.

Writes
- refiner.WORK/eval_samples.csv: one row per sample, kind, perturbation, view set and method;
- 05_results/correction_network_synthetic.csv: per kind, view set and method: TRE before and after (median, 95th
  percentile, maximum), share below 1 and 2 mm, time (median, 95th percentile); for net the coverage of its predicted
  variance (share of parameters within 1.96 sigma, raw and scaled); for the others the paired comparison with opt
  on the same samples (median difference, share of samples where it is lower, Wilcoxon p over the test samples on
  the mean difference of their perturbations and its Holm adjustment over the table, change of the median, change of
  the share below 2 mm in points); a row on the observed masks of the test photos and one on the training.
Perturbations are seeded from the key of the test sample (state_seed), so adding sessions does not change them.

    python 04_code/src/eval_refiner_synthetic.py [--perturbations N] [--states N] [--summary]
"""
import argparse
import csv
import json
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correct as co                      # noqa: E402
import eval_correction_synthetic as ecs   # noqa: E402
import gl_overlay                         # noqa: E402
import refiner as rf                      # noqa: E402
import synth_data as sd                   # noqa: E402

RESULTS = rf.RESULTS
OUT = rf.WORK / "eval_samples.csv"
SETS = {"1_view": 1, "2_views": 2, "4_views": 4}
KINDS = ("a1", "rigid")
METHODS = ("opt", "net", "net_opt", "net_prior")
PERTURBATIONS = 2
SEED = 0


class MaskRenderer(gl_overlay.GLMeshRenderer):
    """gl_overlay.py without the edges: correct.py only needs the mask and the visibility buffer."""

    def render(self, K, dist, T, size, **kw):
        kw.setdefault("edges", False)
        return super().render(K, dist, T, size, **kw)


def test_states(split="test"):
    """Per test sample (state): its 4 photos, in order p1..p4."""
    out = {}
    for s in sd.sessions(rf.SYNTH):
        if int(s.name[1:]) >= sd.CHECK_FROM:
            continue
        for fr in sd.frames(s):
            if fr["split"] == split and fr["mode"] == "photo":
                out.setdefault(f"{s.name}/{fr['state_key']}", []).append((s, fr))
    return {k: sorted(v, key=lambda x: x[1]["view"]) for k, v in sorted(out.items())}


def load_views(photos):
    vs = []
    for s, fr in photos:
        d = sd.load(s, fr)
        sam = cv2.imread(str(rf.SAM_DIR / s.name / (fr["image"] + "_sam.png")), cv2.IMREAD_GRAYSCALE)
        obs = rf.observed_mask(sam, d["labels"])
        ign = rf.ignore_mask(d["labels"])
        vs.append({"rgb": d["rgb"], "obs": obs, "K": d["K"], "dist": d["dist"], "T_C_M": d["T_C_M"], "T_M_B": d["T_M_B"],
                   "view": co.View(d["K"], d["T_C_M"], obs, d["dist"], ign), "image": f"{s.name}/{fr['image']}"})
    return vs


def coverage(T_true, T_est, mu_lv, mode, face):
    """Share of the parameters of the remaining update within 1.96 predicted sigma (last iteration of the network)."""
    rest = rf.params(np.linalg.inv(T_est) @ T_true, mode, face.c, face.o)
    return float(np.mean(np.abs(rest) < 1.96 * np.exp(0.5 * mu_lv)))


def sigma_scale(net, face):
    """Scale of the predicted standard deviations per mode, from the validation split (rf.ValSet cases, 4 views and
    1 view): the 95th percentile of |remaining parameter| / sigma over 1.96, so that 1.96 scaled sigma covers 95%."""
    vs = rf.ValSet(face, net.ren)
    z = {"a1": [], "rigid": []}
    for mode, views, T_M_B, T0 in vs.cases:
        T, _, mu, lv, _ = net.refine([vs.cache.source(i) for i in views], T0, mode)
        z[mode].extend(np.abs(rf.params(np.linalg.inv(T) @ T_M_B, mode, face.c, face.o)) / np.exp(0.5 * lv))
    return {m: float(np.percentile(v, 95) / 1.96) for m, v in z.items()}


def state_seed(key):
    """Seed of the perturbations of a test sample, from its key (sNNNN/rR_sS): the same whatever other samples are
    run or added (the position in the list of samples changed when sessions were added)."""
    return zlib.crc32(key.encode("utf-8"))


def run(a):
    face = rf.load_face()
    ren = MaskRenderer(face.v, face.f)
    net = rf.Refiner(face=face)
    scale = sigma_scale(net, face)
    print(f"sigma scale from validation: {scale}", flush=True)
    rf.WORK.joinpath("sigma_scale.json").write_text(json.dumps(scale), encoding="utf-8")
    X = face.targets
    states = test_states()
    keys = list(states)[:a.states] if a.states else list(states)
    rows = []
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    stale = sorted({r["state"] for r in rows if r.get("seed") != str(state_seed(r["state"]))})
    if stale:
        raise SystemExit(f"{OUT} has samples drawn with other seeds ({len(stale)} states, e.g. {stale[0]}): delete it")
    done = {(r["state"], r["kind"], int(r["perturbation"])) for r in rows}
    t0 = time.perf_counter()
    for si, key in enumerate(keys):
        todo = [(k, j) for k in KINDS for j in range(a.perturbations) if (key, k, j) not in done]
        if not todo:
            continue
        vs = load_views(states[key])
        T_M_B = vs[0]["T_M_B"]
        for kind, j in todo:
            rng = np.random.default_rng([SEED, state_seed(key), KINDS.index(kind), j])
            P, ang, tr = ecs.perturbation(kind, rng, face.c, face.o)
            T0 = T_M_B @ P
            one, pair = int(rng.integers(4)), [(0, 2), (1, 3)][int(rng.integers(2))]
            center = face.c if kind == "a1" else None
            for name, n in SETS.items():
                idx = [one] if n == 1 else list(pair) if n == 2 else [0, 1, 2, 3]
                views = [vs[i]["view"] for i in idx]
                base = {"state": key, "seed": state_seed(key), "kind": kind, "perturbation": j, "set": name,
                        "views": "+".join(f"p{i + 1}" for i in idx), "angle_deg": ang, "trans_mm": tr,
                        "tre_before_mm": rf.tre(np.linalg.inv(T_M_B) @ T0, X)}
                optimise = co.allowed(kind, len(views))    # never 6 degrees of freedom from one view: the net only
                if optimise:
                    res = co.correct(ren, views, T0, X, kind, center=center)
                    rows.append({**base, "method": "opt", "tre_after_mm": rf.tre(np.linalg.inv(T_M_B) @ res.T_M_B, X),
                                 "time_s": res.time_s, "score_after_mm": res.score_mm, "renders": res.renders,
                                 "tre_pred_95_mm": res.tre_pred_95})
                t = time.perf_counter()
                srcs = [rf.make_source(vs[i]["rgb"], vs[i]["obs"], vs[i]["K"], vs[i]["dist"], vs[i]["T_C_M"], T0, face.o)
                        for i in idx]
                T_net, _, mu, lv, _ = net.refine(srcs, T0, kind)
                t_net = time.perf_counter() - t
                rows.append({**base, "method": "net", "tre_after_mm": rf.tre(np.linalg.inv(T_M_B) @ T_net, X),
                             "time_s": t_net, "coverage": coverage(T_M_B, T_net, lv, kind, face),
                             "coverage_scaled": coverage(T_M_B, T_net, lv + 2 * np.log(scale[kind]), kind, face),
                             "sigma_mean": float(np.mean(np.exp(0.5 * lv)))})
                if not optimise:
                    continue
                r2 = rf.optimize(ren, views, T_net, X, kind, center=center)
                rows.append({**base, "method": "net_opt", "tre_after_mm": rf.tre(np.linalg.inv(T_M_B) @ r2.T_M_B, X),
                             "time_s": t_net + r2.time_s, "score_after_mm": r2.score_mm, "renders": r2.renders,
                             "tre_pred_95_mm": r2.tre_pred_95})
                r3 = rf.optimize(ren, views, T_net, X, kind, center=center, sigma=scale[kind] * np.exp(0.5 * lv))
                rows.append({**base, "method": "net_prior", "tre_after_mm": rf.tre(np.linalg.inv(T_M_B) @ r3.T_M_B, X),
                             "time_s": t_net + r3.time_s, "score_after_mm": r3.score_mm, "renders": r3.renders,
                             "tre_pred_95_mm": r3.tre_pred_95})
        ecs.table(OUT, rows)
        print(f"{si + 1}/{len(keys)} {key}: {len(rows)} rows, {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def paired(a, b, groups):
    """a against b on the same samples: median difference (mm) and share where a is lower, over the pairs; Wilcoxon
    signed-rank p (two-sided) over the groups, on the mean difference of each: the perturbations of one test sample
    share its views and masks, so they are not independent pairs. NaN when there are fewer than 6 groups."""
    d = np.asarray(a, float) - np.asarray(b, float)
    inv = np.unique(np.asarray(groups), return_inverse=True)[1].ravel()
    dg = np.bincount(inv, weights=d) / np.bincount(inv)
    try:
        p = float(wilcoxon(dg[dg != 0]).pvalue) if np.sum(dg != 0) >= 6 else float("nan")
    except ValueError:
        p = float("nan")
    return f"{np.median(d):.3f}", f"{np.mean(d < 0):.3f}", p


def p_text(p):
    return "" if not np.isfinite(p) else f"{p:.3g}"


def summarize(rows):
    sp = rf.WORK / "sigma_scale.json"
    scales = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    for r in rows:
        r["perturbation"] = int(r["perturbation"])
        for k in ("angle_deg", "trans_mm", "tre_before_mm", "tre_after_mm", "time_s", "coverage", "coverage_scaled",
                  "sigma_mean"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    out = []
    for kind in KINDS:
        for name in SETS:
            ref = {(r["state"], r["perturbation"]): r["tre_after_mm"] for r in rows
                   if r["kind"] == kind and r["set"] == name and r["method"] == "opt"}
            for method in METHODS:
                g = [r for r in rows if r["kind"] == kind and r["set"] == name and r["method"] == method]
                if not g:
                    continue
                ta, ts = np.array([r["tre_after_mm"] for r in g]), np.array([r["time_s"] for r in g])
                row = {"key": f"{kind}_{name}_{method}", "n": len(g),
                       "tre_before_median_mm": f"{np.median([r['tre_before_mm'] for r in g]):.2f}",
                       "tre_after_median_mm": f"{np.median(ta):.3f}", "tre_after_p95_mm": f"{np.percentile(ta, 95):.3f}",
                       "tre_after_max_mm": f"{ta.max():.3f}", "below_1mm": f"{np.mean(ta < 1):.3f}",
                       "below_2mm": f"{np.mean(ta < 2):.3f}", "time_median_s": f"{np.median(ts):.2f}",
                       "time_p95_s": f"{np.percentile(ts, 95):.2f}"}
                if method == "net":
                    row["coverage_1p96_sigma"] = f"{np.mean([r['coverage'] for r in g]):.3f}"
                    row["coverage_1p96_sigma_scaled"] = f"{np.mean([r['coverage_scaled'] for r in g]):.3f}"
                    row["sigma_scale"] = f"{scales.get(kind, float('nan')):.2f}"
                both = [r for r in g if (r["state"], r["perturbation"]) in ref]
                if method != "opt" and both:     # on the samples both methods ran, by test sample (state)
                    tn = np.array([r["tre_after_mm"] for r in both])
                    to = np.array([ref[r["state"], r["perturbation"]] for r in both])
                    row.update(zip(("vs_opt_diff_median_mm", "vs_opt_lower_share", "vs_opt_p"),
                                   paired(tn, to, [r["state"] for r in both])))
                    row["vs_opt_states"] = len({r["state"] for r in both})
                    row["vs_opt_median_change"] = f"{(np.median(tn) - np.median(to)) / np.median(to):.3f}"
                    row["vs_opt_below_2mm_points"] = f"{100 * (np.mean(tn < 2) - np.mean(to < 2)):.1f}"
                out.append(row)
    # many comparisons: Holm-adjusted p over all of them (the family of this table)
    tested = [r for r in out if "vs_opt_p" in r]
    for r, q in zip(tested, ecs.holm([r["vs_opt_p"] for r in tested])):
        r["vs_opt_p"], r["vs_opt_p_holm"] = p_text(r["vs_opt_p"]), p_text(q)
    mp = rf.SAM_DIR / "masks.csv"
    if mp.exists():
        with mp.open(encoding="utf-8") as fh:
            m = [r for r in csv.DictReader(fh) if r["split"] == "test" and r["mode"] == "photo"]
        iou = np.array([float(r["iou"]) for r in m])
        od = np.array([float(r["outline_mean_px"]) * float(r["mm_per_px"]) for r in m])
        out.append({"key": "masks_test_photos", "n": len(m), "iou_median": f"{np.median(iou):.3f}",
                    "iou_p5": f"{np.percentile(iou, 5):.3f}", "outline_mean_median_mm": f"{np.median(od):.3f}",
                    "outline_mean_p95_mm": f"{np.percentile(od, 95):.3f}"})
    lp = RESULTS / "training_log.csv"
    if lp.exists():
        with lp.open(encoding="utf-8") as fh:
            log = list(csv.DictReader(fh))
        best = min(log[1:] or log, key=lambda r: float(r["val_tre_mean_mm"]))
        out.append({"key": "training", "n": int(log[-1]["samples"] or 0), "epochs": int(log[-1]["epoch"]),
                    "best_epoch": int(best["epoch"]), "hours": log[-1]["hours"],
                    "val_tre_mean_mm": best["val_tre_mean_mm"], "val_tre_median_mm": best["val_tre_median_mm"],
                    "val_tre_before_mean_mm": best["val_tre_before_mean_mm"]})
    ecs.table(RESULTS / "correction_network_synthetic.csv", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturbations", type=int, default=PERTURBATIONS, help="per sample and kind")
    ap.add_argument("--states", type=int, default=0, help="only the first N test samples (0: all)")
    ap.add_argument("--summary", action="store_true", help="only summarize the existing eval_samples.csv")
    a = ap.parse_args()
    if a.summary:
        with OUT.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    else:
        rows = run(a)
    for r in summarize(rows):
        print(json.dumps(r))


if __name__ == "__main__":
    main()
