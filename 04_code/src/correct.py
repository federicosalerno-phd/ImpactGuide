"""Correction of the registration from the outlines of the observed anatomy, with its uncertainty.

Object agnostic. Input: a renderer of the anatomy in the head frame B (the frame its vertices are given in: for the
bench, the coordinates of the bench scene) with the interface of mesh_overlay.MeshRenderer (render(K, dist, T, size)
giving the silhouette mask, the visibility buffer of triangle indices and its offset; vertices v and faces f), the
registration before the impact T_M_B0 and, per view, the observed binary mask (SAM), the intrinsics and the marker pose
T_C_M. Poses are 4 x 4 in mm as in marker_pose.py; the AR draws the model with T_C_M . T_M_B.

Model of the impact: T_M_B = T_M_B0 . dT, dT in the frame B:
- "a1": rotation about the ball centre c (B frame), X' = R (X - c) + c, 3 parameters;
- "rigid": X' = R (X - o) + o + s about the model centroid o, 6 parameters.
The state (R, s) is updated on the manifold: R <- exp(d_r) R, s <- s + d_t.

Residuals: points every STEP_PX px along the outline of the silhouette rendered at the current estimate, lifted to the
mesh through the visibility buffer (barycentric coordinates in their triangle, clamped to it); residual = signed distance
of the projected point from the outline of the observed mask, positive outside, in mm on the object plane (score.py
scale), bilinear on a signed distance field computed once per view on a crop around the mask (extended linearly
beyond it). Analytic Jacobian: gradient of the field x
derivative of the projection (cv2.projectPoints, lens distortion included) x derivative of the point.

Only the silhouette counts: outline points on the cut edge of the mesh (triangles on an open border, the neck and back
of a face model) are left out, since in clinic the skin goes on there.

Optimization: Levenberg-Marquardt with Huber loss (IRLS weights; scipy's "lm" takes no robust loss) on the points of one
render, optionally trimmed (trim: in each view that share of the points farthest from the observed outline weighs zero
for that render, so that an outline piece the mask lacks, from an unmodelled occluder, cannot pull the estimate; off
by default, TRIM), then a new render at the new estimate and new points, until the update is below TOL_DEG and TOL_MM. Six
degrees of freedom need at least two views (MIN_VIEWS): from one view the depth is barely observed. Multi-start:
zero plus +-START_DEG about the three axes of B; every start runs for PROBE renders, the KEEP with the lowest
misalignment score (score.py, the one used for the decision) go on to convergence, the lowest final score wins.

Uncertainty (Laplace): cov = kappa sigma^2 (J^T W J)^-1 at the solution, sigma^2 from the weighted residuals; propagated
to the targets (B frame) as predicted TRE, mean and 95th percentile of the mean target error over draws from N(0, cov).
kappa is calibrated on validation data; with kappa = 1 the residuals, sampled every STEP_PX px, count as independent.
Decision: the correction is applied when the final score is at most tau and the predicted 95% TRE is below u
(conformal.py); otherwise warning, and the AR keeps the old registration flagged as unreliable.
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import score as sc

STEP_PX = 5            # spacing of the outline points
HUBER_MM = 0.5         # Huber threshold of the residuals
TRIM = 0.0             # share of the outline points of each view left out in each pass (those farthest from the mask):
                       # off; 0.2 worsened the median TRE on both synthetic sets (20/10), kept for unmodelled occluders
MIN_VIEWS = {"a1": 1, "rigid": 2}   # never 6 degrees of freedom from one view: the depth is barely observed there
START_DEG = 2.5        # multi-start offsets about the axes of B
TOL_DEG, TOL_MM = 0.02, 0.02   # convergence: update of one render below both
PROBE, KEEP = 2, 2     # every start runs for PROBE renders, the KEEP best go on to convergence
MAX_RENDERS = 15       # renders per start (outer iterations)
INNER = 5              # Levenberg-Marquardt iterations on the points of one render
DRAWS, SEED = 2000, 0  # Monte Carlo of the predicted TRE


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rodrigues(w):
    return cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))[0]


def rotvec(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


class Field:
    """Signed distance (px) from the outline of a binary mask, positive outside, and its gradient, on a crop around it.

    The outline sits on the centres of the outermost mask pixels (distance 0 there, 1 on the first pixel outside), as
    for the outline points of the rendered mask, so that the two agree when the masks are equal.
    """

    def __init__(self, mask):
        m = np.asarray(mask) > 0
        h, w = m.shape
        ys, xs = np.nonzero(m)
        if not len(ys):
            raise ValueError("empty observed mask")
        margin = int(0.25 * max(np.ptp(xs), np.ptp(ys))) + 50
        self.x0, self.y0 = max(xs.min() - margin, 0), max(ys.min() - margin, 0)
        c = m[self.y0:min(ys.max() + margin + 1, h), self.x0:min(xs.max() + margin + 1, w)]
        d_out = cv2.distanceTransform(np.where(c, 0, 255).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d_in = cv2.distanceTransform(np.where(c, 255, 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d = np.where(c, 1.0 - d_in, d_out).astype(np.float32)
        gy, gx = np.gradient(d)
        self.img = np.dstack([d, gx, gy])

    def sample(self, uv):
        """Signed distance and gradient (N x 3) at image points uv (N x 2, px), bilinear. Outside the crop the
        distance goes on linearly from the nearest border point with the gradient there (first order), so that the
        gradient stays the derivative of the distance: a clamped distance with the border gradient gave a Jacobian
        that did not match the residual."""
        h, w = self.img.shape[:2]
        u, v = uv[:, 0] - self.x0, uv[:, 1] - self.y0
        x = np.clip(u, 0, w - 1.001)
        y = np.clip(v, 0, h - 1.001)
        ix, iy = x.astype(int), y.astype(int)
        fx, fy = (x - ix)[:, None], (y - iy)[:, None]
        i = self.img
        out = (i[iy, ix] * (1 - fx) * (1 - fy) + i[iy, ix + 1] * fx * (1 - fy) + i[iy + 1, ix] * (1 - fx) * fy
               + i[iy + 1, ix + 1] * fx * fy)
        off = (u != x) | (v != y)
        if off.any():
            out[off, 0] += out[off, 1] * (u - x)[off] + out[off, 2] * (v - y)[off]
        return out


@dataclass
class View:
    K: np.ndarray
    T_C_M: np.ndarray
    observed: np.ndarray            # H x W binary mask of the anatomy (SAM)
    dist: np.ndarray = None
    ignore: np.ndarray = None       # known occluders: outline points there are left out
    sdf: Field = field(default=None, repr=False)

    def __post_init__(self):
        self.K = np.asarray(self.K, np.float64)
        self.dist = np.zeros(5) if self.dist is None else np.asarray(self.dist, np.float64)
        self.size = (self.observed.shape[1], self.observed.shape[0])
        self.keep = sc.kept(self.observed.shape, sc.BORDER_PX, self.ignore)
        if self.sdf is None:
            self.sdf = Field(self.observed)


@dataclass
class Result:
    delta: np.ndarray        # dT (4 x 4, B frame)
    T_M_B: np.ndarray        # corrected registration T_M_B0 . dT
    params: np.ndarray       # rotation vector of R (deg), then s (mm) for "rigid"
    cov: np.ndarray          # covariance of the update (rad, mm), kappa included
    sigma_mm: float          # residual scale
    tre_pred_mean: float     # predicted TRE at the targets (mm)
    tre_pred_95: float
    score_mm: float          # misalignment score after the correction (mean of the views)
    view_scores: list
    n_points: int
    renders: int             # renders per view, all starts
    start: int               # winning start
    starts: list = field(default_factory=list)   # (start, score_mm, renders, converged) of every start
    render_ms: float = 0.0
    time_s: float = 0.0


def huber_rho(r, k=HUBER_MM):
    a = np.abs(r)
    return np.where(a <= k, 0.5 * a * a, k * (a - 0.5 * k))


def huber_w(r, k=HUBER_MM):
    return np.minimum(1.0, k / np.maximum(np.abs(r), 1e-12))


class Problem:
    """Residuals and Jacobian of one parametrization over a set of views.

    A subclass can add a Gaussian prior on the update (prior(), prior_info): its residuals enter the cost with weight
    1 (quadratic, no Huber) and stay out of the residual scale, the degrees of freedom and the point count; the
    Laplace covariance adds its information to that of the outline residuals."""

    prior_info = None       # information matrix (inverse covariance, rad and mm) of the prior; None: no prior

    def __init__(self, renderer, views, T_M_B0, mode="a1", center=None, trim=TRIM, cut=False):
        if mode not in ("a1", "rigid"):
            raise ValueError(mode)
        if mode == "a1" and center is None:
            raise ValueError("a1 needs the ball centre (B frame)")
        self.renderer, self.views, self.mode = renderer, views, mode
        self.trim = trim
        self.cut_faces = None if cut else cut_triangles(renderer)
        self.T0 = np.asarray(T_M_B0, np.float64)
        self.o = np.asarray(center, np.float64) if mode == "a1" else np.asarray(renderer.v, np.float64).mean(0)
        self.p = 3 if mode == "a1" else 6
        self.ms = 0.0
        self.renders = 0

    def delta(self, R, s):
        D = np.eye(4)
        D[:3, :3], D[:3, 3] = R, self.o + s - R @ self.o
        return D

    def update(self, R, s, d):
        return rodrigues(d[:3]) @ R, (s + d[3:] if self.p == 6 else s)

    def setup(self, R, s, step=STEP_PX):
        """Render every view at (R, s): outline points lifted to the mesh, rendered masks, scales."""
        out = []
        for v in self.views:
            T = v.T_C_M @ self.T0 @ self.delta(R, s)
            r = self.renderer.render(v.K, v.dist, T, v.size)
            self.ms += r.ms
            self.renders += 1
            X = self.lift(v, T, r, step)
            out.append({"X": X, "mask": r.mask, "scale": sc.mm_per_px(v.K, T, self.renderer.v),
                        "A": (v.T_C_M @ self.T0)[:3, :3], "b": (v.T_C_M @ self.T0)[:3, 3]})
        return out

    def lift(self, v, T, r, step):
        """Outline points of the rendered silhouette, every step px, as points of the mesh (B frame)."""
        x0, y0 = r.offset
        h, w = r.faces.shape
        cs, _ = cv2.findContours(r.mask[y0:y0 + h, x0:x0 + w], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE, offset=(x0, y0))
        cs = [c.reshape(-1, 2)[::step] for c in cs if len(c) >= 3 * step]
        if not cs:
            return np.empty((0, 3))
        p = np.vstack(cs)
        p = p[v.keep[p[:, 1], p[:, 0]]]
        fid = r.faces[p[:, 1] - y0, p[:, 0] - x0]
        p, fid = p[fid >= 0], fid[fid >= 0]
        if self.cut_faces is not None:      # the silhouette only: not the cut edge of the mesh
            p, fid = p[~self.cut_faces[fid]], fid[~self.cut_faces[fid]]
        tri = self.renderer.v[self.renderer.f[fid]]                       # n x 3 x 3, B frame
        uv = cv2.projectPoints(tri.reshape(-1, 3), rotvec(T[:3, :3]), T[:3, 3], v.K, v.dist)[0].reshape(-1, 3, 2)
        e1, e2, q = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0], p - uv[:, 0]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        ok = np.abs(det) > 1e-9
        sdet = np.where(ok, det, 1.0)
        b1 = np.where(ok, (q[:, 0] * e2[:, 1] - q[:, 1] * e2[:, 0]) / sdet, 1 / 3)
        b2 = np.where(ok, (e1[:, 0] * q[:, 1] - e1[:, 1] * q[:, 0]) / sdet, 1 / 3)
        b = np.clip(np.stack([1 - b1 - b2, b1, b2], 1), 0, None)
        b /= b.sum(1, keepdims=True)
        return np.einsum("nk,nkj->nj", b, tri)

    def residuals(self, R, s, data, jac=True):
        rs, Js = [], []
        for v, d in zip(self.views, data):
            X = d["X"]
            if not len(X):
                continue
            Y = (X - self.o) @ R.T
            Xc = (Y + self.o + s) @ d["A"].T + d["b"]
            uv, J = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), v.K, v.dist)
            f = v.sdf.sample(uv.reshape(-1, 2))
            rs.append(d["scale"] * f[:, 0])
            if jac:
                dudx = J[:, 3:6].reshape(-1, 2, 3)
                a = d["scale"] * np.einsum("ni,nij->nj", f[:, 1:3], dudx) @ d["A"]   # d r / d X' (B frame)
                Js.append(np.cross(Y, a) if self.p == 3 else np.hstack([np.cross(Y, a), a]))
        r = np.concatenate(rs) if rs else np.empty(0)
        return (r, np.vstack(Js) if Js else np.empty((0, self.p))) if jac else r

    def prior(self, R, s):
        """Residuals of the prior on the update at (R, s) and their Jacobian, weight 1: none here."""
        return np.empty(0), np.empty((0, self.p))

    def lm(self, R, s, data, iters=INNER):
        """Levenberg-Marquardt with Huber loss on the points of one render (plus the quadratic prior, if any).
        Returns R, s and the total update."""
        r, J = self.residuals(R, s, data)
        rp, Jp = self.prior(R, s)
        if len(r) + len(rp) < self.p:
            return R, s, np.zeros(self.p)
        keep = self.kept(r, data)           # fixed for the pass, so that the costs compare
        cost, lam, total = (keep * huber_rho(r)).sum() + 0.5 * rp @ rp, 1e-3, np.zeros(self.p)
        for _ in range(iters):
            w = huber_w(r) * keep
            H = J.T @ (J * w[:, None]) + Jp.T @ Jp
            g = J.T @ (w * r) + Jp.T @ rp
            while lam < 1e8:
                d = np.linalg.solve(H + lam * np.diag(np.diag(H)) + 1e-12 * np.eye(self.p), -g)
                R2, s2 = self.update(R, s, d)
                r2, J2 = self.residuals(R2, s2, data)
                rp2, Jp2 = self.prior(R2, s2)
                c2 = (keep * huber_rho(r2)).sum() + 0.5 * rp2 @ rp2
                if c2 <= cost:
                    R, s, r, J, rp, Jp, cost, lam = R2, s2, r2, J2, rp2, Jp2, c2, max(lam / 3, 1e-7)
                    total += d
                    break
                lam *= 4
            else:
                break
            if np.degrees(np.linalg.norm(d[:3])) < TOL_DEG / 10 and np.linalg.norm(d[3:]) < TOL_MM / 10:
                break
        return R, s, total

    def run(self, st, renders):
        """Advance one start (dict with R, s, data, renders, done) by up to `renders` renders: Levenberg-Marquardt on
        the points of the last render, then a new render at the new estimate, until the update is small."""
        if st["data"] is None:
            st["data"] = self.setup(st["R"], st["s"])
            st["renders"] += 1
            renders -= 1
        while renders > 0 and not st["done"]:
            st["R"], st["s"], total = self.lm(st["R"], st["s"], st["data"])
            st["data"] = self.setup(st["R"], st["s"])
            st["renders"] += 1
            renders -= 1
            st["done"] = np.degrees(np.linalg.norm(total[:3])) < TOL_DEG and np.linalg.norm(total[3:]) < TOL_MM
        st["view_scores"] = self.view_scores(st["data"])
        st["score"] = sc.score(st["view_scores"])
        return st

    def kept(self, r, data):
        """Residuals kept by the trimming: in each view the share 1 - trim closest to the observed outline (all of them
        without trimming). An outline piece the mask lacks (a drape, a finger, a SAM error, skin that moved) then weighs
        nothing instead of pulling the estimate with the Huber weight."""
        keep = np.ones(len(r), bool)
        if not self.trim:
            return keep
        i = 0
        for d in data:
            n = len(d["X"])
            if n:
                a = np.abs(r[i:i + n])
                keep[i:i + n] = a <= np.quantile(a, 1.0 - self.trim)
                i += n
        return keep

    def view_scores(self, data):
        return [sc.view_score(v.observed, d["mask"], d["scale"], v.ignore) for v, d in zip(self.views, data)]

    def covariance(self, R, s, data, kappa=1.0):
        """Laplace covariance of the update: kappa sigma^2 (J^T W J)^-1 from the outline residuals alone, or with a
        prior (J^T W J / (kappa sigma^2) + prior_info)^-1. Returns it, sigma (mm) and the number of outline points."""
        r, J = self.residuals(R, s, data)
        keep = self.kept(r, data)
        r, J = r[keep], J[keep]
        w = huber_w(r)
        dof = max(len(r) - self.p, 1)
        sigma2 = float(np.sum(w * r * r) / dof)
        H = J.T @ (J * w[:, None])
        try:
            cov = (kappa * sigma2 * np.linalg.inv(H) if self.prior_info is None
                   else np.linalg.inv(H / max(kappa * sigma2, 1e-12) + self.prior_info))
        except np.linalg.LinAlgError:
            cov = np.full((self.p, self.p), np.inf)
        return cov, float(np.sqrt(sigma2)), len(r)


def starts(start_deg=START_DEG):
    """Zero plus +-start_deg about the three axes."""
    w = [np.zeros(3)]
    for k in range(3):
        for sgn in (1, -1):
            e = np.zeros(3)
            e[k] = sgn * np.radians(start_deg)
            w.append(e)
    return w


def predicted_tre(cov, R, o, targets, draws=DRAWS, seed=SEED, q=95.0):
    """Mean and q-th percentile of the mean error at the targets (B frame, mm) for updates drawn from N(0, cov)."""
    p = cov.shape[0]
    if not np.all(np.isfinite(cov)):
        return float("inf"), float("inf")
    lam, U = np.linalg.eigh((cov + cov.T) / 2)
    d = np.random.default_rng(seed).standard_normal((draws, p)) * np.sqrt(np.clip(lam, 0, None)) @ U.T
    Y = (np.asarray(targets) - o) @ R.T
    e = np.cross(d[:, None, :3], Y[None])
    if p == 6:
        e = e + d[:, None, 3:]
    tre = np.linalg.norm(e, axis=2).mean(1)
    return float(tre.mean()), float(np.percentile(tre, q))


def allowed(mode, n_views):
    """True when the correction runs with this many views (MIN_VIEWS)."""
    return n_views >= MIN_VIEWS[mode]


def cut_triangles(renderer):
    """Triangles of the mesh on its cut edge (a vertex on an open border: neck and back of a face model), as a bool
    array over the triangles; cached on the renderer. In clinic the skin goes on there, so the outline points on them
    are left out (the silhouette only)."""
    cached = getattr(renderer, "_cut_triangles", None)
    if cached is not None or getattr(renderer, "f", None) is None:     # no mesh given: no cut edge
        return cached
    f = np.asarray(renderer.f)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    u, n = np.unique(e, axis=0, return_counts=True)
    border = np.zeros(int(f.max()) + 1 if len(f) else 0, bool)
    border[u[n == 1].ravel()] = True
    cut = border[f].any(1)
    try:
        renderer._cut_triangles = cut
    except AttributeError:
        pass
    return cut


def correct(renderer, views, T_M_B0, targets, mode="a1", center=None, start_deg=START_DEG, kappa=1.0,
            probe=PROBE, keep=KEEP, max_renders=MAX_RENDERS, trim=TRIM, cut=False):
    """Correct the registration T_M_B0 from the views. targets: points (B frame) for the predicted TRE.

    Every start runs for `probe` renders; the `keep` starts with the lowest score then run to convergence (at most
    max_renders renders each); the lowest final score wins. trim: share of the outline points of each view left out
    in each pass; cut: also the outline points on the cut edge of the mesh. Raises ValueError with fewer views than
    MIN_VIEWS (6 degrees of freedom need two).
    """
    if not allowed(mode, len(views)):
        raise ValueError(f"{mode}: at least {MIN_VIEWS[mode]} views")
    t0 = time.perf_counter()
    pb = Problem(renderer, views, T_M_B0, mode, center, trim, cut)
    runs = [pb.run({"i": i, "R": rodrigues(w), "s": np.zeros(3), "data": None, "renders": 0, "done": False}, probe)
            for i, w in enumerate(starts(start_deg))]
    for st in sorted(runs, key=lambda x: x["score"])[:keep]:
        pb.run(st, max_renders - st["renders"])
    best = min(runs, key=lambda x: x["score"])
    R, s, data = best["R"], best["s"], best["data"]
    cov, sigma, n_pts = pb.covariance(R, s, data, kappa)
    tre_mean, tre_95 = predicted_tre(cov, R, pb.o, targets)
    D = pb.delta(R, s)
    params = np.r_[np.degrees(rotvec(R)), s] if pb.p == 6 else np.degrees(rotvec(R))
    return Result(D, pb.T0 @ D, params, cov, sigma, tre_mean, tre_95, best["score"],
                  [v["score_mm"] for v in best["view_scores"]], n_pts, pb.renders // len(views), best["i"],
                  [(st["i"], st["score"], st["renders"], st["done"]) for st in runs], pb.ms, time.perf_counter() - t0)


def decide(result, tau, u):
    """True: apply the correction; False: warning, keep the old registration flagged as unreliable."""
    return bool(result.score_mm <= tau and result.tre_pred_95 < u)
