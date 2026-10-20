"""Quick checks of the correction, the conformal test and the statistics of the evaluations: the distance field beyond
its crop, one scale of millimetres per pixel, the conformal splits by sample, the seeds of the test samples, the
paired tests by sample with the Holm adjustment, and each corrected view set once in the criterion of the view
guidance. CPU, a few seconds, no data needed.

    python 04_code/src/test_research.py
"""
import csv
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conformal as cf                    # noqa: E402
import correct as co                      # noqa: E402
import eval_correction_synthetic as ecs   # noqa: E402
import eval_refiner_synthetic as ers      # noqa: E402
import eval_views_synthetic as evs        # noqa: E402
import refiner as rf                      # noqa: E402
import score as sc                        # noqa: E402

FAILED = []


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILED.append(what)


def test_field_beyond_crop():
    """Outside its crop the distance goes on with the gradient: residual and Jacobian agree, and the distance keeps
    growing like the true one (a clamped distance stays flat while its gradient says it changes)."""
    m = np.zeros((600, 800), np.uint8)
    cv2.circle(m, (400, 300), 60, 255, -1)
    f = co.Field(m)
    h, w = f.img.shape[:2]
    inside = np.array([[400.0 + 75.0, 300.0]])                  # in the crop, outside the disc
    beyond = np.array([[f.x0 + w + 30.0, 300.0], [400.0, f.y0 - 25.5], [f.x0 - 40.0, f.y0 - 40.0]])
    for name, uv in (("inside the crop", inside), ("beyond the crop", beyond)):
        e = 1e-3
        s = f.sample(uv)
        fd = np.stack([(f.sample(uv + [e, 0])[:, 0] - f.sample(uv - [e, 0])[:, 0]) / (2 * e),
                       (f.sample(uv + [0, e])[:, 0] - f.sample(uv - [0, e])[:, 0]) / (2 * e)], 1)
        check(np.allclose(fd, s[:, 1:3], atol=0.02), f"distance field {name}: finite differences equal the gradient "
              f"(largest gap {np.abs(fd - s[:, 1:3]).max():.4f})")
    true = np.linalg.norm(beyond - [400.0, 300.0], axis=1) - 60.0
    got = f.sample(beyond)[:, 0]
    check(np.all(np.abs(got - true) < 0.15 * true), f"distance beyond the crop close to the true one "
          f"({np.round(got, 1).tolist()} against {np.round(true, 1).tolist()} px)")
    far = f.sample(np.array([[f.x0 + w + 10.0, 300.0], [f.x0 + w + 60.0, 300.0]]))[:, 0]
    check(far[1] - far[0] > 45.0, f"distance grows beyond the crop (+{far[1] - far[0]:.1f} px over 50 px)")
    check(np.array_equal(f.sample(inside), co.Field(m).sample(inside)), "inside the crop: unchanged bilinear values")


def test_one_scale():
    """The crop of the network and the scale of the score use the same mm per pixel (mean focal), also with fx != fy."""
    K = np.array([[1500.0, 0, 1000.0], [0, 1530.0, 760.0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 3] = [4.0, -6.0, 320.0]
    v = np.array([[1.0, 2.0, 3.0]])
    _, side = rf.window(K, None, T, v[0], 100.0)
    check(abs(side * sc.mm_per_px(K, T, v) - 100.0) < 1e-9, "window of refiner.py and score.mm_per_px agree")
    for name in ("refiner.py", "synth_data.py"):
        text = (Path(__file__).resolve().parent / name).read_text(encoding="utf-8")
        check("/ K[0, 0]" not in text and '["K"][0, 0]' not in text, f"{name}: no mm per pixel from fx alone")


def test_conformal_groups():
    """The rows of one sample stay together: three copies of each sample give the same u and the same split results
    as the samples alone, while splitting rows lets copies of a test sample into the calibration."""
    rng = np.random.default_rng(4)
    n = 60
    pred = rng.uniform(0.0, 1.0, n)
    tre = pred * rng.uniform(0.5, 2.0, n)
    gate = rng.random(n) < 0.8
    rep = lambda x: np.repeat(x, 3)
    groups = np.repeat(np.arange(n), 3)
    for a in (0.1, 0.2):
        u1 = cf.calibrate(pred, tre, gate, a, limit=0.6)
        u3 = cf.calibrate(rep(pred), rep(tre), rep(gate), a, limit=0.6, groups=groups)
        check(u1 == u3, f"alpha {a}: copies of a sample in one group leave u unchanged ({u1:.3f})")
    check(cf.calibrate(pred, tre, gate, 0.1, limit=0.6, groups=np.arange(n)) == cf.calibrate(pred, tre, gate, 0.1, limit=0.6),
          "one row per group: same u as without groups")
    s1 = cf.splits(pred, tre, gate, (0.1, 0.2), n_splits=200, limit=0.6)
    s3 = cf.splits(rep(pred), rep(tre), rep(gate), (0.1, 0.2), n_splits=200, limit=0.6, groups=groups)
    sr = cf.splits(rep(pred), rep(tre), rep(gate), (0.1, 0.2), n_splits=200, limit=0.6)
    keys = ("silent_mean", "accepted_mean", "u_median_mm")
    check(all(abs(x[k] - y[k]) < 1e-12 for x, y in zip(s1, s3) for k in keys),
          "splits by group on three copies equal the splits of the samples alone")
    check(any(abs(x[k] - y[k]) > 1e-6 for x, y in zip(s1, sr) for k in keys),
          "splits by row on three copies differ (copies on both sides)")
    check(s3[0]["n_cal"] == 30 and s3[0]["n_test"] == 30, "calibration and test counted in samples")


def test_seeds():
    """The perturbations of a test sample come from its key, not from its place in the list."""
    keys = ["s0012/r0_s1", "s0012/r1_s2", "s0013/r0_s0"]
    more = ["s0011/r3_s2"] + keys            # a session added before them
    c, o = np.array([0.0, -50.0, -20.0]), np.array([2.0, 28.0, 12.0])

    def draw(seed, kind="rigid", j=1):
        rng = np.random.default_rng([ers.SEED, seed, ers.KINDS.index(kind), j])
        return ecs.perturbation(kind, rng, c, o)[0]

    old = [np.array_equal(draw(keys.index(k)), draw(more.index(k))) for k in keys]          # seed = position
    new = [np.array_equal(draw(ers.state_seed(k)), draw(ers.state_seed(more[more.index(k)]))) for k in keys]
    check(all(new) and not any(old), f"same perturbations when a session is added (by position: {sum(old)} of 3 kept)")
    check(ers.state_seed("s0012/r0_s1") == 2347867907 and ers.state_seed(keys[0]) != ers.state_seed(keys[1]),
          "seed: crc32 of the key, stable across runs")
    text = (Path(__file__).resolve().parent / "eval_refiner_synthetic.py").read_text(encoding="utf-8")
    check("default_rng([SEED, si," not in text, "eval_refiner_synthetic.py: no seed from the enumeration index")


def test_statistics():
    """Paired tests over test samples, not over perturbations; Holm adjustment; the criterion of the view guidance
    counts each corrected view set of a sample once."""
    rng = np.random.default_rng(5)
    d = rng.normal(0.1, 0.2, 20)
    a, b = d + 1.0, np.full(20, 1.0)
    g1 = [f"s{k}" for k in range(20)]
    p1 = ers.paired(a, b, g1)[2]
    p2 = ers.paired(np.r_[a, a], np.r_[b, b], g1 + g1)[2]           # the same pairs twice per sample
    p_naive = ers.wilcoxon(np.r_[d, d]).pvalue
    check(abs(p1 - p2) < 1e-12 and p_naive < p1, f"two perturbations per sample count once (p {p1:.4f}, "
          f"as independent pairs {p_naive:.4f})")
    q = ecs.holm([0.01, 0.04, 0.03, 0.005, float("nan")])
    check(np.allclose(q[:4], [0.03, 0.06, 0.06, 0.02]) and np.isnan(q[4]), f"Holm adjustment {np.round(q, 3).tolist()}")
    rows = []
    for i in range(8):
        tre = rng.uniform(0.1, 1.0, 3)
        sets = [("guided", 1, "40", 0), ("guided", 2, "40+70", 1), ("random", 1, "40", 0), ("random", 2, "70+40", 1),
                ("fixed", 1, "55", 2), ("all", 3, "40+55+70", 1)]
        for strategy, k, views, j in sets:
            rows.append({"kind": "a1", "sample": i, "strategy": strategy, "k": k, "views": views, "angle_deg": 1.0,
                         "trans_mm": 0.0, "tre_before_mm": 2.0, "tre_after_mm": tre[j],
                         "pred_rms_mm": 0.2 + 0.1 * j, "tre_pred_95_mm": 0.5, "score_after_mm": 0.1, "e_r_deg": 0.1,
                         "time_s": 1.0, "renders": 3, "plan_s": 0.1, "n_eligible": 3, "plan_order": "40+70+55",
                         "sequential": 0})
    with tempfile.TemporaryDirectory() as tmp:
        evs.RESULTS = Path(tmp)
        out = evs.summarize([{k: str(v) for k, v in r.items()} for r in rows])
        with (Path(tmp) / "views_synthetic.csv").open(encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
    crit = next(r for r in out if r["key"] == "a1_criterion")
    check(crit["n"] == 8 * 4, f"criterion over the distinct view sets ({crit['n']} of {len(rows)} rows)")
    check(any(r.get("vs_random_p_holm") is not None for r in written), "Holm column in views_synthetic.csv")


def test_trimmed_and_views():
    """Trimmed residuals: an outline piece the mask lacks (15% of the points 4 mm off) does not pull the estimate,
    while with the Huber loss alone it does; never 6 degrees of freedom from one view; the cut edge of an open mesh."""
    class Linear(co.Problem):
        def residuals(self, R, s, data, jac=True):
            r = np.r_[s[0] - target, 10.0 * co.rotvec(R), s[1:]]
            J = np.zeros((len(r), 6))
            J[:len(target), 3] = 1.0
            J[len(target):len(target) + 3, :3] = 10.0 * np.eye(3)
            J[len(target) + 3:, 4:] = np.eye(2)
            return (r, J) if jac else r

    rng = np.random.default_rng(6)
    target = rng.normal(0.0, 0.1, 200)
    target[:30] += 4.0                                   # the missing piece: 15% of the points
    data = [{"X": np.zeros((len(target) + 5, 3))}]
    ren = type("R", (), {"v": np.zeros((4, 3)), "f": np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])})()
    est = {}
    for trim in (0.0, 0.2):
        pb = Linear(ren, [], np.eye(4), "rigid", trim=trim)
        R, s = np.eye(3), np.zeros(3)
        for _ in range(5):                               # passes: the trimmed set is chosen again at each
            R, s, _ = pb.lm(R, s, data, iters=30)
        est[trim] = s[0]
    clean = np.mean(target[30:])
    check(abs(est[0.2] - clean) < 0.02 and abs(est[0.0] - clean) > 0.05,
          f"trimmed residuals ignore the missing piece (estimate {est[0.2]:.3f} mm, Huber alone {est[0.0]:.3f}, "
          f"clean {clean:.3f})")
    check(co.allowed("a1", 1) and not co.allowed("rigid", 1) and co.allowed("rigid", 2), "MIN_VIEWS: rigid needs two views")
    try:
        co.correct(ren, [object()], np.eye(4), np.zeros((1, 3)), "rigid")
        raised = False
    except ValueError:
        raised = True
    check(raised, "correct(): no 6 degrees of freedom from one view")
    square = type("R", (), {"f": np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [4, 7, 5], [5, 7, 6]])})()
    cut = co.cut_triangles(square)
    check(cut.tolist() == [True, True, False, False, False, False], "cut edge: triangles on an open border only")


def test_tau():
    """tau of the detection: the largest threshold that still flags 95% of the lost registrations of the validation."""
    rng = np.random.default_rng(7)
    s = np.r_[rng.uniform(1.0, 3.0, 40), rng.uniform(0.0, 1.5, 60)]
    lost = np.r_[np.ones(40, bool), np.zeros(60, bool)]
    tau = ecs.tau_lost(s, lost)
    sens = np.mean(s[lost] > tau)
    above = np.sort(s[lost])[np.sum(s[lost] <= tau)]
    check(sens >= 0.95 and np.mean(s[lost] > above) < 0.95, f"tau {tau:.3f}: sensitivity {sens:.3f}, the next lost "
          f"score would drop it below 0.95")
    check(np.isnan(ecs.tau_lost(s, np.zeros(100, bool))), "no lost registration: no tau")


if __name__ == "__main__":
    for t in (test_field_beyond_crop, test_one_scale, test_conformal_groups, test_seeds, test_statistics,
              test_trimmed_and_views, test_tau):
        t()
    print(f"{len(FAILED)} failed" if FAILED else "all passed")
    sys.exit(1 if FAILED else 0)
