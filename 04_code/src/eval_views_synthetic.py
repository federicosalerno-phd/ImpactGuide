"""Guided, random, fixed and all views for the correction, on synthetic perturbations.

Source "face" (default when synth_twin.py has written it): the phantom face of the digital twin, decimated
(SYNTH/meshes, head frame B: coordinates of the bench scene), the 7 clinical targets (03_data/processed/targets, role
"proposed", columns x_B, y_B, z_B), the A1 ball centre and the photo intrinsics of the twin, the arc of the arm fitted
on the twin frames with the phone level (tilt below 0.1 deg), the known occluders of the A1 variant (marker, A1 parts,
splint) at the registration of the scene.
Source "mandible" (fallback): the renders of render_phone_views.py in synthetic_pose (the mandible mesh in the board
frame M, so the true registration is the identity), 64 placeholder targets.
Truth: the registration of the scene T_M_B. Candidates: the phone along the arc, camera azimuth AZIMUTHS (35 to 95 deg
every 5); T_C_M = T_C_B . T_M_B^-1. Observed mask of each candidate: the anatomy rendered (mesh_overlay.py, or
gl_overlay.py on the GPU, the default, about 3 times faster than --renderer cpu) at the truth, its outline moved by the smooth random field of eval_correction_synthetic.py, minus the occluders in front of it
(depth test on the visibility buffers); the occluders, dilated by DILATE_PX, are the ignore mask of correct.py.
Perturbations as in eval_correction_synthetic.py (a1: up to 5 deg about the ball centre; rigid: up to 5 deg and 5 mm),
stale registration T_M_B0 = T_M_B . P.
Strategies per sample, the same noisy masks for all:
- guided: the first K views of the greedy plan of view_guidance.py at the stale registration, K = 1..K_MAX;
- random: the first K of a random order of the eligible candidates;
- fixed: the photo positions of the protocol (azimuths PHOTO_AZ): 1 (a random one), 2 (1+3 or 2+4), 4;
- all: every eligible candidate.
Each set is corrected with correct.py (same settings as eval_correction_synthetic.py; rigid needs two views at least,
correct.MIN_VIEWS, so its one-view sets are left out). The first TIMING samples of
each perturbation run one at
a time in this process: their times go in the table; the others run in parallel worker processes.

Writes
- WORK/views_samples.csv (WORK: t14 in the work folder, environment variable IMPACTGUIDE_WORK): one row per sample,
  strategy and K;
- 05_results/views_synthetic.csv: per perturbation, strategy and K: TRE after the correction (median, 95th percentile,
  maximum), share below THRESHOLDS, predicted RMS TRE of the plan, guided views most often chosen, guided against
  random and fixed with the same K on the same samples (paired; Wilcoxon p and its Holm adjustment over the table);
  per perturbation and strategy: views needed for the 95th percentile of the TRE to go below each threshold;
  agreement of the predicted RMS TRE with the TRE after the correction (Spearman, ratio; each corrected view set of a
  sample once).

    python 04_code/src/eval_views_synthetic.py [--source face|mandible] [--renderer cpu|gl] [--samples N]
                                               [--workers N] [--timing N] [--summary]
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correct as co                      # noqa: E402
import eval_correction_synthetic as ecs   # noqa: E402
import mesh_overlay as mo                 # noqa: E402
import score as sc                        # noqa: E402
import view_guidance as vg                # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ecs.WORK_ROOT / "synth"
WORK = ecs.WORK
RESULTS = ROOT / "05_results"
TARGETS_CSV = ROOT / "03_data" / "processed" / "targets" / "mandible_targets.csv"
AZIMUTHS = tuple(range(35, 96, 5))
PHOTO_AZ = (40.0, 55.0, 70.0, 85.0)     # camera azimuths of the photo positions, build_test_bench.py P["phone"]
SWEEP_AZ = (35.0, 100.0)                # ends of the sweep, build_test_bench.py P["arm"]["sweep_deg"]
LEVEL_DEG = 0.1                         # twin frames used for the arc: tilt below this
DILATE_PX = 3
K_MAX = 8
THRESHOLDS = (2.0, 1.0, 0.5, 0.25, 0.1)
KINDS = ("a1", "rigid")
SEED = 0

_ctx = {}


def renderer(v, f, kind="cpu"):
    """Renderer with the interface of mesh_overlay.MeshRenderer: "cpu" mesh_overlay.py, "gl" gl_overlay.py (GPU)."""
    if kind == "gl":
        import gl_overlay
        return gl_overlay.GLMeshRenderer(v, f)
    return mo.MeshRenderer(v, f)


def face_available():
    return (SYNTH / "meshes" / "face_phantom_r20k.npz").exists() and any(SYNTH.glob("sessions/*/frames.jsonl"))


def load_face():
    probe = json.loads((SYNTH / "probe.json").read_text(encoding="utf-8"))
    frames, session = [], None
    for p in sorted(SYNTH.glob("sessions/*/frames.jsonl")):
        session = session or json.loads((p.parent / "session.json").read_text(encoding="utf-8"))
        for line in p.read_text(encoding="utf-8").splitlines():
            fr = json.loads(line)
            if abs(fr["tilt_deg"]) < LEVEL_DEG:
                frames.append(fr)
    pairs = list(zip(PHOTO_AZ, probe["arm_deg"])) + list(zip(SWEEP_AZ, probe["sweep_deg"]))
    arc = vg.Arc.fit([np.array(fr["T_C_B"]) for fr in frames], [fr["arm_deg"] for fr in frames], pairs)
    m = np.load(SYNTH / "meshes" / "face_phantom_r20k.npz")
    occ = np.load(SYNTH / "meshes" / "occluders.npz")
    names = sorted({k.split("__")[0] for k in occ.files if "__" in k and not k.startswith("b1_")})
    ov, of, n = [], [], 0
    for name in names:
        ov.append(occ[name + "__v"].astype(np.float64))
        of.append(occ[name + "__f"].astype(np.int64) + n)
        n += len(ov[-1])
    with TARGETS_CSV.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["role"] == "proposed"]
    ph = session["intrinsics"]["photo"]
    return {"source": "face", "v": m["vertices"].astype(np.float64), "f": m["faces"].astype(np.int64),
            "K": np.array(ph["K"]), "size": tuple(ph["size"]), "T": np.array(occ["T_M_B"]),
            "c": np.array(probe["ball_centre"]), "targets": np.array([[float(r[k]) for k in ("x_B", "y_B", "z_B")]
                                                                       for r in rows]),
            "target_ids": [r["id"] for r in rows], "arc": arc, "n_frames": len(frames),
            "occluders": (np.vstack(ov), np.vstack(of), names)}


def load_mandible():
    gt, v, f, c = ecs.load()
    vw = gt["views"]
    arc = vg.Arc.fit([np.array(x["camera_from_board"]) for x in vw], [x["arm_deg"] for x in vw],
                     [(x["camera_deg"], x["arm_deg"]) for x in vw])
    return {"source": "mandible", "v": v, "f": f, "K": np.array(vw[0]["K"]), "size": tuple(gt["image_size"]),
            "T": np.eye(4), "c": c, "targets": ecs.farthest_points(v, ecs.TARGETS),
            "target_ids": [f"v{k}" for k in range(ecs.TARGETS)], "arc": arc, "n_frames": len(vw), "occluders": None}


def depth(ren, r, K, T):
    """Depth (camera z, mm) of the visible triangle at each pixel centre (pinhole), inf where there is none."""
    out = np.full(r.mask.shape, np.inf)
    (x0, y0), buf = r.offset, r.faces
    ys, xs = np.nonzero(buf >= 0)
    if not len(ys):
        return out
    tri = ren.v[ren.f[buf[ys, xs]]] @ T[:3, :3].T + T[:3, 3]           # n x 3 x 3, camera frame
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    d = np.c_[xs + x0, ys + y0, np.ones(len(xs))] @ np.linalg.inv(K).T  # rays with z = 1
    den = np.sum(n * d, 1)
    z = np.where(np.abs(den) > 1e-12, np.sum(n * tri[:, 0], 1) / np.where(np.abs(den) > 1e-12, den, 1.0),
                 tri[:, :, 2].mean(1))
    out[ys + y0, xs + x0] = z
    return out


def init(source="face", kind="cpu"):
    cv2.setNumThreads(1)
    s = load_face() if source == "face" else load_mandible()
    ren = renderer(s["v"], s["f"], kind)
    cands = s["arc"].candidates(AZIMUTHS, s["K"], s["size"], s["T"])
    occ_ren = renderer(*s["occluders"][:2], kind) if s["occluders"] else None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1, 2 * DILATE_PX + 1))
    views = []
    for cd in cands:
        T = cd.T_C_M @ s["T"]
        r = ren.render(cd.K, None, T, cd.size)
        front = np.zeros(r.mask.shape, bool)
        if occ_ren is not None:
            ro = occ_ren.render(cd.K, None, T, cd.size)
            front = (ro.mask > 0) & (depth(occ_ren, ro, cd.K, T) < depth(ren, r, cd.K, T))
            cd.ignore = cv2.dilate(front.astype(np.uint8), kernel) * 255
        views.append({"mask": r.mask, "front": front, "scale": sc.mm_per_px(cd.K, T, ren.v),
                      "occluded": float((front & (r.mask > 0)).sum() / max((r.mask > 0).sum(), 1))})
    _ctx.update(s=s, ren=ren, cands=cands, views=views, photo=[AZIMUTHS.index(int(a)) for a in PHOTO_AZ])


def run_sample(job):
    kind, i = job
    s, ren, cands, views = _ctx["s"], _ctx["ren"], _ctx["cands"], _ctx["views"]
    X, c, Tt = s["targets"], s["c"], s["T"]
    rng = np.random.default_rng([SEED, i, KINDS.index(kind), 3])
    P, ang, tr = ecs.perturbation(kind, rng, c, ren.v.mean(0))
    T0 = Tt @ P
    obs = [np.where(v["front"], 0, ecs.observed(v["mask"], v["scale"], rng)).astype(np.uint8) for v in views]
    center = c if kind == "a1" else None
    pl = vg.plan(ren, cands, T0, X, kind, c, K_MAX)
    rnd = [int(k) for k in rng.permutation(pl.eligible)]
    ph = _ctx["photo"]
    one, pair = ph[int(rng.integers(4))], [(ph[0], ph[2]), (ph[1], ph[3])][int(rng.integers(2))]
    sets = [("guided", k, pl.order[:k]) for k in range(1, min(K_MAX, len(pl.order)) + 1)] + \
           [("random", k, rnd[:k]) for k in range(1, min(K_MAX, len(rnd)) + 1)] + \
           [("fixed", 1, [one]), ("fixed", 2, list(pair)), ("fixed", 4, ph)] + \
           ([("all", len(pl.eligible), pl.eligible)] if pl.eligible else [])     # no eligible view: no "all" set
    sets = [x for x in sets if co.allowed(kind, len(x[2]))]     # never 6 degrees of freedom from one view
    done, rows = {}, []
    for strategy, k, sel in sets:
        key = tuple(sorted(sel))
        if key not in done:
            vs = [co.View(cands[j].K, cands[j].T_C_M, obs[j], None, cands[j].ignore) for j in sel]
            done[key] = co.correct(ren, vs, T0, X, kind, center=center)
        res = done[key]
        E = np.linalg.inv(Tt) @ res.T_M_B
        rows.append({"kind": kind, "sample": i, "strategy": strategy, "k": k,
                     "views": "+".join(f"{cands[j].angle_deg:g}" for j in sel), "angle_deg": ang, "trans_mm": tr,
                     "tre_before_mm": ecs.tre(P, X), "tre_after_mm": ecs.tre(E, X),
                     "pred_rms_mm": pl.predicted(sel), "tre_pred_95_mm": res.tre_pred_95, "score_after_mm": res.score_mm,
                     "e_r_deg": float(np.degrees(np.linalg.norm(co.rotvec(E[:3, :3])))), "time_s": res.time_s,
                     "renders": res.renders, "plan_s": pl.time_s, "n_eligible": len(pl.eligible),
                     "plan_order": "+".join(f"{cands[j].angle_deg:g}" for j in pl.order), "sequential": 0})
    return rows


def first_below(p95, thr):
    ks = sorted(p95)
    return next((k for k in ks if p95[k] < thr), "")


def paired(a, b):
    """Guided against another strategy on the same samples: median difference (mm), share where guided is lower,
    Wilcoxon signed-rank p (two-sided; equal pairs, the same views chosen, left out)."""
    d = np.asarray(a) - np.asarray(b)
    try:
        p = float(wilcoxon(d[d != 0]).pvalue) if np.sum(d != 0) >= 6 else float("nan")
    except ValueError:
        p = float("nan")
    return f"{np.median(d):.3f}", f"{np.mean(d < 0):.2f}", p


def summarize(rows):
    for r in rows:
        for k in ("sample", "k", "renders", "n_eligible", "sequential"):
            r[k] = int(r[k])
        for k, v in r.items():
            if k.endswith(("_mm", "_deg", "_s")):
                r[k] = float(v)
    out = []
    for kind in KINDS:
        g = [r for r in rows if r["kind"] == kind]
        if not g:
            continue
        for strategy in ("guided", "random", "fixed", "all"):
            gs = [r for r in g if r["strategy"] == strategy]
            p95 = {}
            for k in sorted({r["k"] for r in gs}):
                gk = [r for r in gs if r["k"] == k]
                ta = np.array([r["tre_after_mm"] for r in gk])
                seq = [r["plan_s"] for r in gk if r["sequential"]]
                p95[k] = float(np.percentile(ta, 95))
                mode, cnt = Counter("+".join(sorted(r["views"].split("+"), key=float)) for r in gk).most_common(1)[0]
                row = {"key": f"{kind}_{strategy}_k{k}", "n": len(gk),
                       "tre_before_median_mm": f"{np.median([r['tre_before_mm'] for r in gk]):.2f}",
                       "tre_after_median_mm": f"{np.median(ta):.3f}", "tre_after_p95_mm": f"{p95[k]:.3f}",
                       "tre_after_max_mm": f"{ta.max():.3f}"}
                row.update({f"below_{t:g}mm".replace(".", "p"): f"{np.mean(ta < t):.3f}" for t in THRESHOLDS})
                row.update({"pred_rms_median_mm": f"{np.median([r['pred_rms_mm'] for r in gk]):.3f}",
                            "tre_pred_95_median_mm": f"{np.median([r['tre_pred_95_mm'] for r in gk]):.3f}",
                            "e_r_median_deg": f"{np.median([r['e_r_deg'] for r in gk]):.3f}",
                            "time_median_s": f"{np.median([r['time_s'] for r in gk]):.1f}",
                            "views_mode": mode, "views_mode_share": f"{cnt / len(gk):.2f}",
                            "plan_median_s": f"{np.median(seq):.2f}" if seq else ""})
                if strategy == "guided":
                    for other in ("random", "fixed"):
                        ref = {r["sample"]: r["tre_after_mm"] for r in g if r["strategy"] == other and r["k"] == k}
                        both = [r for r in gk if r["sample"] in ref]
                        if both:
                            row.update(zip([f"vs_{other}_diff_median_mm", f"vs_{other}_lower_share", f"vs_{other}_p"],
                                           paired([r["tre_after_mm"] for r in both], [ref[r["sample"]] for r in both])))
                out.append(row)
            if strategy != "all":
                need = {"key": f"{kind}_{strategy}_needed"}
                need.update({f"needed_{t:g}mm".replace(".", "p"): first_below(p95, t) for t in THRESHOLDS})
                out.append(need)
        # one row per corrected view set: the strategies often pick the same set, which is corrected once
        u = list({(r["sample"], tuple(sorted(r["views"].split("+"), key=float))): r for r in g}.values())
        pred, ta = np.array([r["pred_rms_mm"] for r in u]), np.array([r["tre_after_mm"] for r in u])
        out.append({"key": f"{kind}_criterion", "n": len(u), "spearman_pred_vs_tre": f"{spearmanr(pred, ta)[0]:.2f}",
                    "tre_over_pred_median": f"{np.median(ta / pred):.2f}",
                    "tre_over_pred_p95": f"{np.percentile(ta / pred, 95):.2f}"})
    # many comparisons: Holm-adjusted p over all of them (the family of this table)
    cols = [(r, f"vs_{o}_p") for r in out for o in ("random", "fixed") if f"vs_{o}_p" in r]
    for (r, c), q in zip(cols, ecs.holm([r[c] for r, c in cols])):
        r[c], r[c + "_holm"] = ("" if not np.isfinite(x) else f"{x:.3f}" for x in (r[c], q))
    ecs.table(RESULTS / "views_synthetic.csv", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("face", "mandible"), default="face" if face_available() else "mandible")
    ap.add_argument("--renderer", choices=("cpu", "gl"), default="gl", help="gl_overlay.py (GPU) or mesh_overlay.py")
    ap.add_argument("--samples", type=int, default=60, help="samples per perturbation")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--timing", type=int, default=2, help="samples per perturbation run one at a time, for the times")
    ap.add_argument("--summary", action="store_true", help="only summarize the existing views_samples.csv")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "views_samples.csv"
    rows = []
    if out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    if not a.summary:
        t0 = time.perf_counter()
        done = {(r["kind"], int(r["sample"])) for r in rows}
        jobs = [(k, i) for i in range(a.samples) for k in KINDS if (k, i) not in done]
        timed = [j for j in jobs if j[1] < a.timing]
        init(a.source, a.renderer)
        s = _ctx["s"]
        print(f"source {s['source']}: arc from {s['n_frames']} poses, fit {s['arc'].rms_mm:.2f} mm "
              f"{s['arc'].rms_deg:.3f} deg; occluded share per candidate "
              f"{[round(v['occluded'], 3) for v in _ctx['views']]}; {time.perf_counter() - t0:.0f} s", flush=True)
        cv2.setNumThreads(-1)
        for j in timed:
            for r in run_sample(j):
                r["sequential"] = 1
                rows.append(r)
        print(f"{len(timed)} samples one at a time: {time.perf_counter() - t0:.0f} s", flush=True)
        rest = [j for j in jobs if j not in timed]
        with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.source, a.renderer)) as ex:
            for k, rr in enumerate(ex.map(run_sample, rest, chunksize=1)):
                rows.extend(rr)
                if (k + 1) % 10 == 0:
                    ecs.table(out, rows)
                    print(f"{k + 1}/{len(rest)} {time.perf_counter() - t0:.0f} s", flush=True)
        ecs.table(out, rows)
        print(f"{len(rows)} rows in {out}, {time.perf_counter() - t0:.0f} s")
    for r in summarize(rows):
        print(r)


if __name__ == "__main__":
    main()
