"""View guidance: which views of the anatomy to take for the correction.

Object agnostic, like correct.py: any mesh with a renderer with the interface of mesh_overlay.MeshRenderer, any set of
candidate views (offline: the video frames with their T_C_M; live: hypothetical positions of the arm, only rendered).

Expected information of a candidate view at the current estimate of the registration T_M_B: the outline residuals of
correct.py with the view's own render taken as the observation (expected residual zero, gradient of the distance field
along the outline normal), so the Jacobian J_v (points x parameters; rad, then mm for "rigid") comes from the render
alone. Fisher information
    I_v = J_v^T J_v / (KAPPA SIGMA_MM^2),
SIGMA_MM the outline noise and KAPPA the inflation for residuals that are correlated along the outline (correct.py takes
them every STEP_PX px). Expected covariance after the views S: Sigma_S = (Sigma0^-1 + sum_S I_v)^-1, Sigma0 the prior
from the largest displacements (standard deviation of each parameter = its largest amplitude, MAX_DEG and MAX_MM).
Criterion: predicted mean squared target error tr(W Sigma_S), W = mean over the targets of A^T A, A the derivative of
the target displacement with respect to the parameters (A1: -[Y]x, rigid: [-[Y]x, I], Y = target - centre); its square
root is the predicted RMS TRE in mm. Greedy choice: K times the eligible view that lowers the criterion most. A view is
eligible when the board is seen (at least MIN_CORNERS ChArUco corners inside the image, board facing the camera) and its
outline gives at least MIN_POINTS points off the image border and the known occluders.

Arc of the arm (live candidates): the phone turns with the arm about a fixed axis. Arc.fit takes camera poses at known
arm angles (frames of the digital twin), finds the axis and the camera at arm angle 0 and gives the camera at any arm
angle; camera azimuths about the aim point of the bench map to arm angles by linear interpolation of known pairs
(build_test_bench.py: photo positions and the ends of the sweep).
"""
import time
from dataclasses import dataclass

import numpy as np

import correct as co

SIGMA_MM = 0.3       # outline noise of the observed masks (mm on the object plane)
KAPPA = 5.0          # inflation for correlated residuals (5-7 for 95% coverage on the synthetic views)
MAX_DEG, MAX_MM = 5.0, 5.0    # largest displacements of the impact: prior Sigma0
MIN_CORNERS = 8      # ChArUco corners in the image for the board pose
MIN_POINTS = 30      # outline points of the anatomy
BOARD = {"squares": 5, "square_mm": 9.0}


@dataclass
class Candidate:
    name: str
    K: np.ndarray
    T_C_M: np.ndarray
    size: tuple                     # (w, h)
    dist: np.ndarray = None
    ignore: np.ndarray = None       # known occluders (H x W), outline points there are left out
    angle_deg: float = float("nan")     # camera azimuth (arm candidates)


def prior(mode, max_deg=MAX_DEG, max_mm=MAX_MM):
    s = [np.radians(max_deg)] * 3 + ([max_mm] * 3 if mode == "rigid" else [])
    return np.diag(np.square(s))


def target_weight(targets, centre, mode):
    """W (p x p): tr(W Sigma) is the expected mean squared displacement of the targets for an update ~ N(0, Sigma)."""
    Y = np.asarray(targets, np.float64) - np.asarray(centre, np.float64)
    A = np.stack([-co.hat(y) for y in Y])                                 # n x 3 x 3
    if mode == "rigid":
        A = np.concatenate([A, np.broadcast_to(np.eye(3), A.shape)], 2)   # n x 3 x 6
    return np.einsum("nij,nik->jk", A, A) / len(Y)


def board_corners(cand, board=BOARD, margin_px=10):
    """ChArUco corners inside the image (margin_px from the border) when the board faces the camera, else 0."""
    n, sq = board["squares"], board["square_mm"]
    g = np.arange(1, n) * sq
    P = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    Xc = np.c_[P, np.zeros(len(P))] @ cand.T_C_M[:3, :3].T + cand.T_C_M[:3, 3]
    if np.mean(Xc, 0) @ cand.T_C_M[:3, 2] <= 0 or np.any(Xc[:, 2] <= 0):   # z of the board into it, away from the camera
        return 0
    uv = Xc @ np.asarray(cand.K).T
    uv = uv[:, :2] / uv[:, 2:]
    w, h = cand.size
    return int(np.sum((uv[:, 0] >= margin_px) & (uv[:, 1] >= margin_px) & (uv[:, 0] < w - margin_px)
                      & (uv[:, 1] < h - margin_px)))


class _Rendered:
    """A renderer whose render is already done: the Problem of fisher() reuses it instead of rendering again."""

    def __init__(self, renderer, r):
        self.v, self.f, self.r = renderer.v, renderer.f, r

    def render(self, *args, **kwargs):
        return self.r


def fisher(renderer, cand, T_M_B, mode="a1", center=None, sigma_mm=SIGMA_MM, kappa=KAPPA):
    """Fisher information (p x p) of one candidate at the registration T_M_B and its number of outline points."""
    T = cand.T_C_M @ T_M_B
    r = renderer.render(cand.K, cand.dist, T, cand.size)
    p = 3 if mode == "a1" else 6
    if not r.mask.any():
        return np.zeros((p, p)), 0
    pb = co.Problem(_Rendered(renderer, r), [co.View(cand.K, cand.T_C_M, r.mask, cand.dist, cand.ignore)], T_M_B, mode,
                    center)
    data = pb.setup(np.eye(3), np.zeros(3))
    _, J = pb.residuals(np.eye(3), np.zeros(3), data)
    return J.T @ J / (kappa * sigma_mm ** 2), len(J)


def criterion(infos, sel, Sigma0_inv, W):
    """Predicted RMS TRE (mm) after the views sel."""
    H = Sigma0_inv + sum((infos[k] for k in sel), np.zeros_like(Sigma0_inv))
    return float(np.sqrt(max(np.trace(W @ np.linalg.inv(H)), 0.0)))


def greedy(infos, Sigma0, W, k_max, eligible=None):
    """Greedy choice of up to k_max views: order and predicted RMS TRE after each (the prior alone first)."""
    Si = np.linalg.inv(Sigma0)
    left = list(range(len(infos)) if eligible is None else eligible)
    order, pred = [], [criterion(infos, [], Si, W)]
    while left and len(order) < k_max:
        c = [criterion(infos, order + [k], Si, W) for k in left]
        order.append(left.pop(int(np.argmin(c))))
        pred.append(min(c))
    return order, pred


@dataclass
class Plan:
    order: list          # indices of the chosen candidates, in the order of choice
    pred_rms: list       # predicted RMS TRE (mm): prior, then after each chosen view
    infos: list          # Fisher information of every candidate
    n_points: list
    eligible: list
    W: np.ndarray
    Sigma0: np.ndarray
    time_s: float

    def predicted(self, sel):
        return criterion(self.infos, list(sel), np.linalg.inv(self.Sigma0), self.W)


def plan(renderer, candidates, T_M_B, targets, mode="a1", center=None, k_max=8, max_deg=MAX_DEG, max_mm=MAX_MM,
         sigma_mm=SIGMA_MM, kappa=KAPPA, min_corners=MIN_CORNERS, min_points=MIN_POINTS):
    """Views to take, best first, at the current estimate T_M_B (before any view: the registration before the impact)."""
    t0 = time.perf_counter()
    centre = np.asarray(center, np.float64) if mode == "a1" else np.asarray(renderer.v, np.float64).mean(0)
    infos, npts = zip(*[fisher(renderer, c, T_M_B, mode, center, sigma_mm, kappa) for c in candidates])
    eligible = [k for k, c in enumerate(candidates) if npts[k] >= min_points and board_corners(c) >= min_corners]
    S0, W = prior(mode, max_deg, max_mm), target_weight(targets, centre, mode)
    order, pred = greedy(list(infos), S0, W, k_max, eligible)
    return Plan(order, pred, list(infos), list(npts), eligible, W, S0, time.perf_counter() - t0)


def turn(axis, pivot, deg):
    """4 x 4 rotation by deg about the line through pivot along axis."""
    R = co.rodrigues(np.asarray(axis) * np.radians(deg))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, pivot - R @ pivot
    return T


class Arc:
    """Camera of the phone on the arm: T_B_C(phi) = turn(axis, pivot, phi) . T_B_C0."""

    def __init__(self, axis, pivot, T_B_C0, pairs=None):
        self.axis, self.pivot, self.T_B_C0 = np.asarray(axis), np.asarray(pivot), np.asarray(T_B_C0)
        self.pairs = None if pairs is None else np.asarray(sorted(pairs), np.float64)   # (azimuth, arm angle)
        self.rms_mm = self.rms_deg = 0.0

    @classmethod
    def fit(cls, T_C_B, arm_deg, pairs=None, min_step_deg=5.0):
        """Axis, pivot and camera at arm angle 0 from camera poses (camera from B) at known arm angles."""
        B = [np.linalg.inv(T) for T in T_C_B]
        phi = np.asarray(arm_deg, np.float64)
        ax, rows, rhs = [], [], []
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                d = phi[j] - phi[i]
                if abs(d) < min_step_deg:
                    continue
                R = B[j][:3, :3] @ B[i][:3, :3].T
                ax.append(co.rotvec(R) / np.radians(d))
                rows.append(np.eye(3) - R)
                rhs.append(B[j][:3, 3] - R @ B[i][:3, 3])
        if not ax:
            raise ValueError("arc fit needs poses at least min_step_deg apart")
        a = np.mean(ax, 0)
        a /= np.linalg.norm(a)
        p = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        p -= a * (a @ p)
        T0 = [turn(a, p, -f) @ Bi for f, Bi in zip(phi, B)]
        U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in T0))
        R0 = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        T_B_C0 = np.eye(4)
        T_B_C0[:3, :3], T_B_C0[:3, 3] = R0, np.mean([T[:3, 3] for T in T0], 0)
        arc = cls(a, p, T_B_C0, pairs)
        res = [arc.T_B_C(f) for f in phi]
        arc.rms_mm = float(np.sqrt(np.mean([np.sum((r[:3, 3] - b[:3, 3]) ** 2) for r, b in zip(res, B)])))
        arc.rms_deg = float(np.degrees(np.sqrt(np.mean([np.sum(co.rotvec(r[:3, :3] @ b[:3, :3].T) ** 2)
                                                        for r, b in zip(res, B)]))))
        return arc

    def T_B_C(self, arm_deg):
        return turn(self.axis, self.pivot, arm_deg) @ self.T_B_C0

    def T_C_B(self, arm_deg):
        return np.linalg.inv(self.T_B_C(arm_deg))

    def arm_of(self, azimuth_deg):
        """Arm angle for a camera azimuth: linear interpolation of the pairs, linear extrapolation beyond them.

        Raises:
            ValueError: the arc has fewer than two (azimuth, arm angle) pairs.
        """
        if self.pairs is None or len(self.pairs) < 2:
            raise ValueError("arm_of needs at least two (azimuth, arm angle) pairs")
        az, arm = self.pairs[:, 0], self.pairs[:, 1]
        x = float(azimuth_deg)
        if x < az[0]:
            return float(arm[0] + (x - az[0]) * (arm[1] - arm[0]) / (az[1] - az[0]))
        if x > az[-1]:
            return float(arm[-1] + (x - az[-1]) * (arm[-1] - arm[-2]) / (az[-1] - az[-2]))
        return float(np.interp(x, az, arm))

    def candidates(self, azimuths_deg, K, size, T_M_B, dist=None):
        """Candidates at the camera azimuths: T_C_M = T_C_B . T_M_B^-1 (T_M_B: registration of the board)."""
        Ti = np.linalg.inv(T_M_B)
        return [Candidate(f"cam{a:g}", np.asarray(K, np.float64), self.T_C_B(self.arm_of(a)) @ Ti, tuple(size), dist,
                          None, float(a)) for a in azimuths_deg]
