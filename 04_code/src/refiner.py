"""Refinement network: a multi-view render-and-compare network that initialises the correction of correct.py.

Inputs per view: a 256 x 256 crop of fixed scale (CROP_MM, about 0.8 mm per pixel) centred on the projection of the
face centroid at the current estimate of the registration, with 6 channels: RGB, observed face mask (SAM 2), face
mask and edges rendered at the estimate (undeformed face, gl_overlay.py); view geometry: pose of the camera with
respect to the estimated head frame B (coordinates of the bench scene; rotation in the 6D representation, translation
/ 300 mm), not T_C_M.
Network: ImageNet ResNet-18 shared by the views, first layer widened to 6 channels (RGB weights copied, the other
channels initialised to their mean), 512 image features plus 64 from the geometry branch, a per-view layer, fusion by
mean and maximum over the views (any number), a trunk and two heads: A1 (3 parameters, rotation about the ball
centre c) and general (6 parameters, rotation about the face centroid o and translation), each parameter with its
log-variance. Parameters in degrees (rotation vector) and mm, frame B, update T_M_B <- T_M_B . dT as in correct.py.
Kinematic layer: dT(w) = [R(w), (I - R(w)) c] for A1, [R(w), o + s - R(w) o] for the general case.
Loss: point matching on POINTS face vertices (mean distance in mm between the points moved by the predicted and by
the true update) plus beta-NLL (Seitzer et al., ICLR 2022, beta = BETA) of the parameters on their predicted variance.
Inference: ITERS iterations with a new render at each (Refiner.refine), then the optimisation of correct.py starts from
the output (optimize: one start, no multi-start), or uses the output with its predicted variance as a Gaussian prior
(optimize(..., sigma)).

Data: the digital twin of synth_twin.py (SYNTH, split by session 180/48/72 samples). Observed masks:
SAM 2.1 tiny in fp32 (sam2_segment.py), prompted once per image with the box of the visible face (the face mesh
without the part under the surgical cap) rendered at the registration the system has before the correction (T_M_B0),
plus a 10% margin; the pixels of the known covers and occluders (marker, A1 or B1 parts, splint, cap, drape, pendulum,
the skull seen through the eye holes; in practice rendered from T_C_M or segmented by colour) are then taken out of
the SAM mask, and the same pixels, dilated by DILATE_PX, are the ignore mask of correct.py. Perturbations of the
registration are drawn on the fly around the true one (70% up to MAX_DEG and MAX_MM, 30% up to a fifth of that).

    python 04_code/src/refiner.py prepare [--sessions 0:25]
        SAM masks of every image (SAM_DIR/sNNNN/<image>_sam.png, raw, full resolution) and their quality against the
        exact labels (SAM_DIR/masks.csv, masks.json); the cache of the training sources (WORK/cache_<split>_*.npy):
        a REGION_MM square around the true face centroid at the crop scale, RGB and observed mask.
    python 04_code/src/refiner.py train [--hours H] [--epochs N] [--workers N] [--resume]
        training with early stopping on the validation split: weights in MODELS (refiner.pt the best, refiner_last.pt
        to resume), log in 05_results/training_log.csv.

Library: RefinerNet, Refiner(path).refine(sources, T_M_B0, mode, centre), make_source(...), optimize(...),
observed_mask(sam, labels), ignore_mask(labels).
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import correct as co          # noqa: E402
import synth_data as sd       # noqa: E402

ROOT = SRC.parents[1]
WORK_ROOT = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
SYNTH = WORK_ROOT / "synth"
SAM_DIR = WORK_ROOT / "synth_sam"
WORK = WORK_ROOT / "refiner"
MODELS = WORK_ROOT / "models" / "refiner"
os.environ.setdefault("TORCH_HOME", str(WORK_ROOT / "models" / "torch"))     # ImageNet weights of ResNet-18
RESULTS = ROOT / "05_results"
TARGETS_CSV = ROOT / "03_data" / "processed" / "targets" / "mandible_targets.csv"

CROP_MM, CROP_PX = 200.0, 256
REGION_MM, REGION_PX = 288.0, 368          # cached source around the face, same scale as the crop (1.28 px per mm)
OCCLUDERS = (2, 3, 4, 5, 6, 7, 9)          # labels: marker, drape, interface, head spheres, head, pendulum, cap
DILATE_PX = 3
BOX_MARGIN = 0.10
MAX_DEG, MAX_MM, FINE = 6.0, 6.0, 0.3      # training perturbations: 70% up to the maximum, 30% up to a fifth
POSE_NOISE_DEG, POSE_NOISE_MM = 0.05, 0.2  # per-view jitter of T_C_M during training
ITERS = 4
POINTS = 500
BETA, NLL_W = 0.5, 0.5
PRIOR_MM = math.sqrt(5.0) * 0.3              # outline residual scale of view_guidance.py (kappa 5, sigma 0.3 mm)
CROPS = 16                                 # crops per forward pass (V views x 16 / V samples)
ACCUM = 32                                 # samples per optimiser step
VIEWS = (1, 2, 3, 4)
EPOCH_CROPS = 7200                         # one epoch: twice as many crops as training images
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SEED = 0


# Geometry and data

@dataclass
class Face:
    v: np.ndarray            # decimated face (frame B, mm)
    f: np.ndarray
    o: np.ndarray            # centroid of the decimated vertices: centre of the general update (as correct.py)
    c: np.ndarray            # A1 ball centre (frame B)
    targets: np.ndarray      # the clinical targets of role "proposed" (frame B, columns x_B, y_B, z_B)
    visible: np.ndarray      # faces not under the surgical cap (prompt box of SAM)


def load_face(root=SYNTH):
    m = np.load(root / "meshes" / "face_phantom_r20k.npz")
    v, f = m["vertices"].astype(np.float64), m["faces"].astype(np.int64)
    probe = json.loads((root / "probe.json").read_text(encoding="utf-8"))
    with TARGETS_CSV.open(encoding="utf-8") as fh:
        X = np.array([[float(r[k]) for k in ("x_B", "y_B", "z_B")] for r in csv.DictReader(fh) if r["role"] == "proposed"])
    visible = f
    wp, fp = root / "meshes" / "face_weights.npz", root / "meshes" / "face_phantom_full.npz"
    if wp.exists() and fp.exists():
        from scipy.spatial import cKDTree
        covered = np.load(wp)["covered"][cKDTree(np.load(fp)["vertices"]).query(v)[1]]
        visible = f[~covered[f].any(1)]
    return Face(v, f, v.mean(0), np.array(probe["ball_centre"], np.float64), X, visible)


def occluders(labels):
    return np.isin(labels, OCCLUDERS)


def observed_mask(sam, labels):
    """Observed face mask: the SAM mask without the known covers and occluders (uint8 0/255)."""
    return np.where((np.asarray(sam) > 0) & ~occluders(labels), 255, 0).astype(np.uint8)


def ignore_mask(labels, dilate_px=DILATE_PX):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    return cv2.dilate(occluders(labels).astype(np.uint8) * 255, k)


def project(K, dist, T, X):
    X = np.atleast_2d(X)
    return cv2.projectPoints(X @ T[:3, :3].T + T[:3, 3], np.zeros(3), np.zeros(3), K,
                             np.zeros(5) if dist is None else dist)[0].reshape(-1, 2)


def window(K, dist, T, point, side_mm):
    """Centre (px) and side (px) of a square of side_mm around a point of B seen by the camera T (camera from B)."""
    z = float(T[2, :3] @ point + T[2, 3])
    return project(K, dist, T, point)[0], side_mm * (K[0, 0] + K[1, 1]) / 2 / z


@dataclass
class Source:
    """A view resampled around the face at the crop scale: image (BGR uint8), observed mask (uint8), intrinsics of
    the resampled image, lens distortion and marker pose."""
    img: np.ndarray
    obs: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    T_C_M: np.ndarray


def make_source(img, obs, K, dist, T_C_M, T_M_B, point, region_mm=REGION_MM, out_px=REGION_PX):
    """Resample the square of side region_mm around `point` (frame B) seen at T_C_M . T_M_B to out_px (area
    interpolation, zero outside the image). The intrinsics follow the resampling, so the distortion model still holds."""
    K = np.asarray(K, np.float64)
    dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
    (cx, cy), side = window(K, dist, T_C_M @ T_M_B, point, region_mm)
    S = max(int(round(side)), 8)
    x0, y0 = int(round(cx - S / 2)), int(round(cy - S / 2))
    h, w = img.shape[:2]

    def cut(a):
        out = np.zeros((S, S) + a.shape[2:], a.dtype)
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + S, w), min(y0 + S, h)
        if xb > xa and yb > ya:
            out[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]
        return cv2.resize(out, (out_px, out_px), interpolation=cv2.INTER_AREA)

    return Source(cut(img), cut(obs), sd.crop_K(K, (x0 + S / 2, y0 + S / 2), S, out_px), dist, np.asarray(T_C_M, np.float64))


def crop_affine(centre, side, out_px=CROP_PX):
    """Affine map of the square (centre, side) of an image to out_px pixels, pixel centres as sd.crop_K."""
    s = out_px / side
    x0, y0 = centre[0] - side / 2, centre[1] - side / 2
    return np.array([[s, 0.0, (0.5 - x0) * s - 0.5], [0.0, s, (0.5 - y0) * s - 0.5]])


def view_inputs(ren, src, T_C_B, point):
    """Crop of a source at the estimate T_C_B (camera from B): RGB, observed mask, rendered mask and edges (each
    256 x 256 uint8) and the view geometry (9 floats)."""
    centre, side = window(src.K, src.dist, T_C_B, point, CROP_MM)
    M = crop_affine(centre, side)
    rgb = cv2.warpAffine(src.img, M, (CROP_PX, CROP_PX), flags=cv2.INTER_LINEAR, borderValue=0)
    obs = cv2.warpAffine(src.obs, M, (CROP_PX, CROP_PX), flags=cv2.INTER_LINEAR, borderValue=0)
    Kc = sd.crop_K(src.K, centre, side, CROP_PX)
    r = ren.render(Kc, src.dist, T_C_B, (CROP_PX, CROP_PX))
    edge = np.zeros((CROP_PX, CROP_PX), np.uint8)
    if len(r.segments):
        cv2.polylines(edge, list(np.round(r.segments * 16).astype(np.int32)), False, 255, 1, cv2.LINE_8, 4)
    return {"rgb": rgb, "obs": obs, "ren": r.mask, "edge": edge, "g": geometry(T_C_B)}


def geometry(T_C_B):
    R = T_C_B[:3, :3]
    return np.r_[R[:, 0], R[:, 1], T_C_B[:3, 3] / 300.0].astype(np.float32)


def stack(d):
    """6 x H x W uint8: RGB (from BGR), observed mask, rendered mask, rendered edges."""
    return np.concatenate([d["rgb"][..., ::-1].transpose(2, 0, 1), d["obs"][None], d["ren"][None], d["edge"][None]], 0)


def normalise(x):
    """B x V x 6 x H x W uint8 tensor to float: ImageNet statistics for RGB, masks to [-1, 1]."""
    x = x.float() / 255.0
    m = torch.as_tensor(MEAN, device=x.device).view(1, 1, 3, 1, 1)
    s = torch.as_tensor(STD, device=x.device).view(1, 1, 3, 1, 1)
    return torch.cat([(x[:, :, :3] - m) / s, x[:, :, 3:] * 2 - 1], 2)


# Update parametrisation

def delta(theta, mode, c, o):
    """4 x 4 update dT in frame B from parameters (deg, then mm for "rigid")."""
    R = co.rodrigues(np.radians(np.asarray(theta[:3], np.float64)))
    D = np.eye(4)
    D[:3, :3] = R
    D[:3, 3] = (np.eye(3) - R) @ c if mode == "a1" else o + np.asarray(theta[3:6], np.float64) - R @ o
    return D


def params(D, mode, c, o):
    """Inverse of delta(): parameters of a 4 x 4 update (for "a1" the translation is implied by the rotation)."""
    w = np.degrees(co.rotvec(D[:3, :3]))
    return w if mode == "a1" else np.r_[w, D[:3, 3] - o + D[:3, :3] @ o]


def perturbation(mode, rng, c, o, max_deg=MAX_DEG, max_mm=MAX_MM, fine=FINE):
    """Displacement P of the registration (T_M_B0 = T_M_B . P): random axis, angle (and translation) uniform up to the
    maximum, or up to a fifth of it in a share `fine` of the draws."""
    k = 0.2 if rng.random() < fine else 1.0
    ax = rng.normal(size=3)
    th = np.radians(rng.uniform(0, max_deg * k)) * ax / np.linalg.norm(ax)
    if mode == "a1":
        return delta(np.degrees(th), "a1", c, o)
    t = rng.normal(size=3)
    t *= rng.uniform(0, max_mm * k) / np.linalg.norm(t)
    R = co.rodrigues(th)
    D = np.eye(4)
    D[:3, :3], D[:3, 3] = R, (np.eye(3) - R) @ o + t
    return D


def so3_exp(w):
    """Rotation matrices (B x 3 x 3) from rotation vectors (B x 3, rad), differentiable."""
    th = w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    s, c1 = torch.sin(th)[..., None], (1 - torch.cos(th))[..., None]
    return torch.eye(3, device=w.device, dtype=w.dtype) + s * Kx + c1 * Kx @ Kx


def move(theta, X, mode, c, o):
    """Kinematic layer: points X (N x 3, frame B) moved by the updates theta (B x p): B x N x 3."""
    R = so3_exp(torch.deg2rad(theta[:, :3]))
    p = c if mode == "a1" else o
    Y = torch.einsum("bij,nj->bni", R, X - p) + p
    return Y if mode == "a1" else Y + theta[:, None, 3:6]


def losses(mu, logvar, theta, X, mode, c, o):
    """Point matching (mean distance of the moved points, mm) and beta-NLL of the parameters."""
    pm = (move(mu, X, mode, c, o) - move(theta, X, mode, c, o)).norm(dim=2).mean()
    var = logvar.exp()
    nll = (0.5 * (logvar + (mu - theta) ** 2 / var) * var.detach() ** BETA).mean()
    return pm, nll


# Network

class RefinerNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        w = r.conv1.weight.data
        conv = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        conv.weight.data[:, :3] = w
        conv.weight.data[:, 3:] = w.mean(1, keepdim=True)
        r.conv1, r.fc = conv, nn.Identity()
        self.enc = r
        self.geo = nn.Sequential(nn.Linear(9, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.view = nn.Sequential(nn.Linear(512 + 64, 512), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, 256), nn.ReLU(inplace=True))
        self.heads = nn.ModuleDict({"a1": nn.Linear(256, 6), "rigid": nn.Linear(256, 12)})
        for h in self.heads.values():      # start at the identity update with unit variance
            nn.init.zeros_(h.bias)
            h.weight.data *= 0.1

    def forward(self, x, g, mode):
        """x: B x V x 6 x H x W (normalised), g: B x V x 9. Returns parameters and log-variances (B x p each)."""
        B, V = x.shape[:2]
        f = self.enc(x.flatten(0, 1))
        h = self.view(torch.cat([f, self.geo(g.flatten(0, 1))], 1)).view(B, V, -1)
        out = self.heads[mode](self.trunk(torch.cat([h.mean(1), h.amax(1)], 1)))
        p = out.shape[1] // 2
        return out[:, :p], out[:, p:].clamp(-12.0, 8.0)


# Inference

class Refiner:
    """The trained network with its renderer: refine(sources, T_M_B0, mode, centre) runs ITERS render-and-compare
    iterations and returns the refined registration."""

    def __init__(self, path=MODELS / "refiner.pt", device=None, face=None, renderer=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.net = RefinerNet(pretrained=False).to(self.device)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.face = face or load_face()
        if renderer is None:
            import gl_overlay
            renderer = gl_overlay.GLMeshRenderer(self.face.v, self.face.f)
        self.ren = renderer
        self.info = {k: ck[k] for k in ("epoch", "val") if k in ck}

    def sources(self, images, masks, Ks, dists, T_C_Ms, T_M_B0):
        """Sources around the face at the registration T_M_B0, one per view (full-resolution images and masks)."""
        return [make_source(im, m, K, d, T, T_M_B0, self.face.o) for im, m, K, d, T in zip(images, masks, Ks, dists, T_C_Ms)]

    @torch.no_grad()
    def refine(self, sources, T_M_B0, mode, centre=None, iters=ITERS):
        """Returns T_M_B, the total update (4 x 4, frame B), the last parameters and log-variances, the time (s)."""
        t0 = time.perf_counter()
        c = self.face.c if centre is None else np.asarray(centre, np.float64)
        T, total = np.asarray(T_M_B0, np.float64).copy(), np.eye(4)
        mu = lv = None
        for _ in range(iters):
            ds = [view_inputs(self.ren, s, s.T_C_M @ T, self.face.o) for s in sources]
            x = torch.from_numpy(np.stack([stack(d) for d in ds]))[None].to(self.device)
            g = torch.from_numpy(np.stack([d["g"] for d in ds]))[None].to(self.device)
            m, l = self.net(normalise(x), g, mode)
            mu, lv = m[0].cpu().numpy().astype(np.float64), l[0].cpu().numpy().astype(np.float64)
            D = delta(mu, mode, c, self.face.o)
            T, total = T @ D, total @ D
        return T, total, mu, lv, time.perf_counter() - t0


class PriorProblem(co.Problem):
    """correct.Problem with a Gaussian prior N(0, diag(sigma^2)) on the update from the registration it starts at (the
    network output; sigma: its predicted standard deviations, deg then mm): extra residuals PRIOR_MM . p / sigma, so
    that the prior weighs as the outline residuals do (kappa sigma^2 of view_guidance.py)."""

    def __init__(self, renderer, views, T_M_B0, mode="a1", center=None, sigma=None):
        super().__init__(renderer, views, T_M_B0, mode, center)
        sig = np.asarray(sigma, np.float64)
        self.w = PRIOR_MM / np.maximum(np.r_[np.radians(sig[:3]), sig[3:]], 1e-6)

    def residuals(self, R, s, data, jac=True):
        out = super().residuals(R, s, data, jac)
        rp = self.w * np.r_[co.rotvec(R), s][:self.p]
        if not jac:
            return np.r_[out, rp]
        return np.r_[out[0], rp], np.vstack([out[1], np.diag(self.w)])


def optimize(renderer, views, T_M_B_init, targets, mode="a1", center=None, max_renders=co.MAX_RENDERS, kappa=1.0,
             sigma=None):
    """The optimisation of correct.py from one start, the given registration (the network output): same residuals,
    Levenberg-Marquardt, convergence test and Laplace uncertainty, no multi-start; with sigma (the network's predicted
    standard deviations) a Gaussian prior on the update (PriorProblem). Returns a correct.Result."""
    t0 = time.perf_counter()
    pb = (co.Problem(renderer, views, T_M_B_init, mode, center) if sigma is None
          else PriorProblem(renderer, views, T_M_B_init, mode, center, sigma))
    st = pb.run({"i": 0, "R": np.eye(3), "s": np.zeros(3), "data": None, "renders": 0, "done": False}, max_renders)
    R, s, data = st["R"], st["s"], st["data"]
    cov, sigma_r, n_pts = pb.covariance(R, s, data, kappa)
    tre_mean, tre_95 = co.predicted_tre(cov, R, pb.o, targets)
    D = pb.delta(R, s)
    p = np.r_[np.degrees(co.rotvec(R)), s] if pb.p == 6 else np.degrees(co.rotvec(R))
    return co.Result(D, pb.T0 @ D, p, cov, sigma_r, tre_mean, tre_95, st["score"], [v["score_mm"] for v in st["view_scores"]],
                     n_pts, pb.renders // len(views), 0, [(0, st["score"], st["renders"], st["done"])], pb.ms,
                     time.perf_counter() - t0)


def tre(E, X):
    """Mean target error (mm) of the error transform E = T_true^-1 . T_estimate (frame B)."""
    return float(np.linalg.norm(X @ E[:3, :3].T + E[:3, 3] - X, axis=1).mean())


# SAM masks and cache

def cmd_prepare(a):
    import gl_overlay
    import sam2_segment as ss
    lo, hi = (int(x) for x in a.sessions.split(":"))
    face = load_face()
    vis_ren = gl_overlay.GLMeshRenderer(face.v, face.visible)
    pred = ss.load_predictor("tiny")
    rows = []
    SAM_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    jobs = [(s, fr) for s in sd.sessions(SYNTH) if lo <= int(s.name[1:]) < hi for fr in sd.frames(s)]
    if not jobs:
        raise SystemExit(f"prepare: no rendered images in sessions {a.sessions} of {SYNTH}")
    splits = {}
    for s, fr in jobs:
        splits.setdefault(fr["split"], []).append(f"{s.name}/{fr['image']}")
    store = {}
    for split, names in splits.items():
        n = len(names)
        store[split] = {"img": np.lib.format.open_memmap(WORK / f"cache_{split}_img.npy", "w+", np.uint8,
                                                         (n, REGION_PX, REGION_PX, 3)),
                        "obs": np.lib.format.open_memmap(WORK / f"cache_{split}_obs.npy", "w+", np.uint8,
                                                         (n, REGION_PX, REGION_PX)),
                        "meta": [], "n": 0}
    last = None
    for s, fr in jobs:
        if s != last:
            if last is not None:
                print(f"{last.name}: {len(rows)} images, {time.perf_counter() - t_start:.0f} s", flush=True)
            (SAM_DIR / s.name).mkdir(exist_ok=True)
            last = s
        d = sd.load(s, fr)
        out = SAM_DIR / s.name / (fr["image"] + "_sam.png")
        T0 = d["T_C_M"] @ d["T_M_B0"]
        r = vis_ren.render(d["K"], d["dist"], T0, tuple(fr["size"]), edges=False)
        ys, xs = np.nonzero(r.mask)
        t = time.perf_counter()
        score = float("nan")
        if out.exists():
            sam = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        elif not len(xs):
            sam = np.zeros(d["labels"].shape, np.uint8)
            cv2.imwrite(str(out), sam)
        else:
            b = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
            ctr, half = (b[:2] + b[2:]) / 2, (b[2:] - b[:2]) / 2 * (1 + BOX_MARGIN)
            h, w = d["labels"].shape
            box = np.r_[np.maximum(ctr - half, 0), np.minimum(ctr + half, [w - 1, h - 1])]
            m, score = ss.segment(pred, cv2.cvtColor(d["rgb"], cv2.COLOR_BGR2RGB), box)
            sam = np.where(m, 255, 0).astype(np.uint8)
            cv2.imwrite(str(out), sam)
        sam_ms = (time.perf_counter() - t) * 1e3
        gt = d["masks"]["face"]
        obs = observed_mask(sam, d["labels"])
        o = obs > 0
        T = d["T_C_B"]
        mmpp = float(np.median(face.v @ T[2, :3] + T[2, 3]) / d["K"][0, 0])
        od = sd.outline_distance(o, gt) if o.any() and gt.any() else np.array([np.inf])
        iou = lambda m_: float((m_ & gt).sum() / max((m_ | gt).sum(), 1))
        rows.append({"image": f"{s.name}/{fr['image']}", "split": fr["split"], "mode": fr["mode"],
                     "drape": int(bool(fr["drape"])), "sam_score": round(score, 4), "iou_raw": round(iou(sam > 0), 4),
                     "iou": round(iou(o), 4), "outline_mean_px": round(float(od.mean()), 2),
                     "outline_p95_px": round(float(np.percentile(od, 95)), 2), "mm_per_px": round(mmpp, 4),
                     "face_px": int(gt.sum()), "sam_ms": round(sam_ms, 1)})
        src = make_source(d["rgb"], obs, d["K"], d["dist"], d["T_C_M"], d["T_M_B"], face.o)
        st = store[fr["split"]]
        st["img"][st["n"]], st["obs"][st["n"]] = src.img, src.obs
        st["n"] += 1
        st["meta"].append({"K": src.K, "dist": src.dist, "T_C_M": d["T_C_M"], "T_M_B": d["T_M_B"], "T_M_B0": d["T_M_B0"],
                           "state": f"{s.name}/{fr['state_key']}", "view": fr["view"], "mode": fr["mode"],
                           "image": f"{s.name}/{fr['image']}"})
    print(f"{last.name}: {len(rows)} images, {time.perf_counter() - t_start:.0f} s", flush=True)
    sd_rows = sorted(rows, key=lambda r: r["image"])
    with (SAM_DIR / "masks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(sd_rows[0]))
        w.writeheader()
        w.writerows(sd_rows)
    summ = {}
    for split in ("train", "val", "test", "all"):
        g = [r for r in rows if split in ("all", r["split"])]
        if not g:
            continue
        a_ = lambda k: np.array([r[k] for r in g], float)
        od_mm = a_("outline_mean_px") * a_("mm_per_px")
        summ[split] = {"images": len(g), "iou_raw_median": round(float(np.median(a_("iou_raw"))), 4),
                       "iou_median": round(float(np.median(a_("iou"))), 4), "iou_p5": round(float(np.percentile(a_("iou"), 5)), 4),
                       "iou_below_0p9": round(float(np.mean(a_("iou") < 0.9)), 4),
                       "outline_mean_mm_median": round(float(np.median(od_mm)), 3),
                       "outline_mean_mm_p95": round(float(np.percentile(od_mm, 95)), 3),
                       "drape_iou_median": round(float(np.median([r["iou"] for r in g if r["drape"]] or [np.nan])), 4),
                       "sam_ms_median": round(float(np.median(a_("sam_ms"))), 1)}
    (SAM_DIR / "masks.json").write_text(json.dumps({"prompt": "box of the visible face at T_M_B0, 10% margin",
                                                    "model": "sam2.1 tiny fp32", "summary": summ}, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    for split, st in store.items():
        st["img"].flush()
        st["obs"].flush()
        es = st["meta"]
        np.savez(WORK / f"cache_{split}_meta.npz", K=np.stack([e["K"] for e in es]), dist=np.stack([e["dist"] for e in es]),
                 T_C_M=np.stack([e["T_C_M"] for e in es]), T_M_B=np.stack([e["T_M_B"] for e in es]),
                 T_M_B0=np.stack([e["T_M_B0"] for e in es]), state=np.array([e["state"] for e in es]),
                 view=np.array([e["view"] for e in es]), mode=np.array([e["mode"] for e in es]),
                 image=np.array([e["image"] for e in es]))
        print(f"cache {split}: {st['n']} images", flush=True)
    del store
    print(f"prepare: {time.perf_counter() - t_start:.0f} s")


class Cache:
    """Cached sources of one split, grouped by state (memory-mapped)."""

    def __init__(self, split, root=WORK):
        self.img = np.load(root / f"cache_{split}_img.npy", mmap_mode="r")
        self.obs = np.load(root / f"cache_{split}_obs.npy", mmap_mode="r")
        m = np.load(root / f"cache_{split}_meta.npz")
        self.meta = {k: m[k] for k in m.files}
        self.states = {}
        for i, s in enumerate(self.meta["state"]):
            self.states.setdefault(str(s), []).append(i)
        self.keys = sorted(self.states)

    def source(self, i):
        return Source(np.asarray(self.img[i]), np.asarray(self.obs[i]), self.meta["K"][i], self.meta["dist"][i],
                      self.meta["T_C_M"][i])

    def photos(self, key):
        return sorted((i for i in self.states[key] if self.meta["mode"][i] == "photo"), key=lambda i: self.meta["view"][i])


# Training

def occlude(d, rng):
    """Paste a random occluder (glove, instrument) on the crop and take it out of the observed mask."""
    h, w = d["obs"].shape
    ys, xs = np.nonzero(d["obs"] > 127)
    if not len(ys):
        return
    k = rng.integers(len(ys))
    ax = (int(rng.uniform(0.04, 0.15) * w), int(rng.uniform(0.02, 0.08) * h))
    m = cv2.ellipse(np.zeros((h, w), np.uint8), (int(xs[k]), int(ys[k])), ax, float(rng.uniform(0, 180)), 0, 360, 255, -1) > 0
    d["rgb"][m] = rng.integers(0, 256, 3)
    d["obs"][m] = 0


class TrainStream(torch.utils.data.IterableDataset):
    """Endless micro-batches from the cached training split: a mode, V views per sample, CROPS // V samples, a
    perturbation per sample around the true registration, per-view jitter of T_C_M, photometric augmentation and
    pasted occluders."""

    def __init__(self, split="train", seed=SEED):
        super().__init__()
        self.split, self.seed = split, seed

    def __iter__(self):
        import gl_overlay
        cv2.setNumThreads(1)
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        torch.set_num_threads(1)
        rng = np.random.default_rng([self.seed, wid])
        face = load_face()
        ren = gl_overlay.GLMeshRenderer(face.v, face.f)
        cache = Cache(self.split)
        # states with at least V images, for each number of views V (all of them when every state has its photos)
        keys = {V: [k for k in cache.keys if len(cache.states[k]) >= V] for V in VIEWS}
        if not any(keys.values()):
            raise ValueError(f"training split {self.split}: no state with {min(VIEWS)} or more images")
        while True:
            mode = "a1" if rng.random() < 0.5 else "rigid"
            V = int(rng.choice(VIEWS))
            if not keys[V]:
                continue
            n = CROPS // V
            xs, gs, th = [], [], []
            for key in rng.choice(keys[V], n, replace=True):
                idx = rng.choice(cache.states[key], V, replace=False)
                T_M_B = cache.meta["T_M_B"][idx[0]]
                P = perturbation(mode, rng, face.c, face.o)
                Test = T_M_B @ P
                th.append(params(np.linalg.inv(P), mode, face.c, face.o))
                xv, gv = [], []
                for i in idx:
                    src = cache.source(i)
                    J = np.eye(4)
                    J[:3, :3] = co.rodrigues(np.radians(rng.normal(0, POSE_NOISE_DEG, 3)))
                    J[:3, 3] = rng.normal(0, POSE_NOISE_MM, 3)
                    d = view_inputs(ren, src, J @ src.T_C_M @ Test, face.o)
                    d["rgb"] = sd.augment(d["rgb"], rng, max_blur_px=1.0)
                    if rng.random() < 0.15:
                        occlude(d, rng)
                    xv.append(stack(d))
                    gv.append(d["g"])
                xs.append(np.stack(xv))
                gs.append(np.stack(gv))
            yield {"x": torch.from_numpy(np.stack(xs)), "g": torch.from_numpy(np.stack(gs)),
                   "theta": torch.from_numpy(np.stack(th).astype(np.float32)), "mode": mode}


class ValSet:
    """Fixed validation cases: every validation state, both modes, one perturbation each (uniform up to 5 deg and
    5 mm, as the evaluation), with its 4 photos and with one of them."""

    def __init__(self, face, ren, split="val", seed=SEED):
        import eval_correction_synthetic as ecs
        self.cache, self.face, self.ren = Cache(split), face, ren
        self.cases = []
        for k, key in enumerate(self.cache.keys):
            ph = self.cache.photos(key)
            if not ph:           # a state without photos has no validation case (the seeds follow k, not the cases)
                continue
            for mode in ("a1", "rigid"):
                rng = np.random.default_rng([seed, k, 0 if mode == "a1" else 1, 7])
                P = ecs.perturbation(mode, rng, face.c, face.o)[0]
                T_M_B = self.cache.meta["T_M_B"][ph[0]]
                for views in (ph, [ph[k % len(ph)]]):
                    self.cases.append((mode, views, T_M_B, T_M_B @ P))

    @torch.no_grad()
    def run(self, net, device, iters=ITERS):
        net.eval()
        rows = []
        for mode, views, T_M_B, T0 in self.cases:
            srcs = [self.cache.source(i) for i in views]
            T, out = T0.copy(), []
            for _ in range(iters):
                ds = [view_inputs(self.ren, s, s.T_C_M @ T, self.face.o) for s in srcs]
                x = torch.from_numpy(np.stack([stack(d) for d in ds]))[None].to(device)
                g = torch.from_numpy(np.stack([d["g"] for d in ds]))[None].to(device)
                mu, _ = net(normalise(x), g, mode)
                T = T @ delta(mu[0].cpu().numpy().astype(np.float64), mode, self.face.c, self.face.o)
                out.append(tre(np.linalg.inv(T_M_B) @ T, self.face.targets))
            rows.append({"mode": mode, "views": len(views), "before": tre(np.linalg.inv(T_M_B) @ T0, self.face.targets),
                         "first": out[0], "last": out[-1]})
        net.train()
        last = np.array([r["last"] for r in rows])
        return {"val_tre_mean_mm": float(last.mean()), "val_tre_median_mm": float(np.median(last)),
                "val_tre_p95_mm": float(np.percentile(last, 95)),
                "val_tre1_mean_mm": float(np.mean([r["first"] for r in rows])),
                "val_tre_before_mean_mm": float(np.mean([r["before"] for r in rows])),
                **{f"val_tre_{m}_{v}v_mm": float(np.mean([r["last"] for r in rows if r["mode"] == m and r["views"] == v]))
                   for m in ("a1", "rigid") for v in (1, 4)}}


def lr_factor(step, progress, warmup=100):
    """Warm-up, then cosine over the progress (0 to 1: the larger of steps and time used over their budget)."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def cmd_train(a):
    import gl_overlay
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False      # same speed, no workspace trials on the 4 GB card
    dev = torch.device("cuda")
    MODELS.mkdir(parents=True, exist_ok=True)
    face = load_face()
    ren = gl_overlay.GLMeshRenderer(face.v, face.f)
    rng = np.random.default_rng(SEED)
    X = torch.from_numpy(face.v[rng.choice(len(face.v), POINTS, replace=False)]).float().to(dev)
    c, o = torch.from_numpy(face.c).float().to(dev), torch.from_numpy(face.o).float().to(dev)
    net = RefinerNet(pretrained=not a.resume).to(dev)
    enc = [p for n, p in net.named_parameters() if n.startswith("enc.")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": rest, "lr": a.lr * 3}], weight_decay=1e-4)
    mb_per_epoch = EPOCH_CROPS // CROPS
    total_steps = a.epochs * mb_per_epoch * CROPS // (ACCUM * 2)     # about 2 crops per sample on average
    log_path = RESULTS / "training_log.csv"
    epoch0, step, best, best_epoch, since, log = 0, 0, float("inf"), -1, 0, []
    if a.resume:
        ck = torch.load(MODELS / "refiner_last.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        epoch0, step, best, best_epoch, since = ck["epoch"] + 1, ck["step"], ck["best"], ck["best_epoch"], ck["since"]
        with log_path.open(encoding="utf-8") as fh:
            log = [r for r in csv.DictReader(fh) if int(r["epoch"]) < epoch0]
    vs = ValSet(face, ren)
    if not a.resume:
        v0 = vs.run(net, dev)
        log.append({"epoch": 0, "step": 0, "samples": 0, "lr": 0.0, "loss": "", "train_pm_mm": "", "train_nll": "",
                    **{k: round(x, 4) for k, x in v0.items()}, "epoch_s": 0, "val_s": "", "hours": 0.0, "gpu_peak_mb": "",
                    "best": 0})
        write_log(log_path, log)
        print(json.dumps(log[-1]), flush=True)
        epoch0 = 1
    dl = torch.utils.data.DataLoader(TrainStream(seed=SEED + epoch0), batch_size=None, num_workers=a.workers,
                                     persistent_workers=True, prefetch_factor=4)
    it = iter(dl)
    t_start = time.perf_counter() - (float(log[-1]["hours"]) * 3600 if log else 0.0)
    samples = int(log[-1]["samples"]) if log and log[-1]["samples"] != "" else 0
    acc = 0
    net.train()
    opt.zero_grad(set_to_none=True)
    for epoch in range(epoch0, a.epochs + 1):
        t_ep = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        sums, n_mb = np.zeros(3), 0
        for _ in range(mb_per_epoch):
            b = next(it)
            mode = b["mode"]
            theta = b["theta"].to(dev)
            mu, lv = net(normalise(b["x"].to(dev, non_blocking=True)), b["g"].to(dev), mode)
            pm, nll = losses(mu, lv, theta, X, mode, c, o)
            loss = pm + NLL_W * nll
            n = theta.shape[0]
            (loss * n / ACCUM).backward()
            acc += n
            samples += n
            sums += [loss.item(), pm.item(), nll.item()]
            n_mb += 1
            if acc >= ACCUM:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                f = lr_factor(step, max(step / total_steps, (time.perf_counter() - t_start) / 3600 / a.hours))
                opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = a.lr * f, a.lr * 3 * f
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                acc = 0
        ep_s = time.perf_counter() - t_ep
        t_v = time.perf_counter()
        v = vs.run(net, dev)
        val_s = time.perf_counter() - t_v
        better = v["val_tre_mean_mm"] < best
        if better:
            best, best_epoch, since = v["val_tre_mean_mm"], epoch, 0
            torch.save({"model": net.state_dict(), "epoch": epoch, "val": v,
                        "config": {"crop_mm": CROP_MM, "crop_px": CROP_PX, "iters": ITERS}}, MODELS / "refiner.pt")
        else:
            since += 1
        hours = (time.perf_counter() - t_start) / 3600
        log.append({"epoch": epoch, "step": step, "samples": samples, "lr": round(opt.param_groups[0]["lr"], 8),
                    "loss": round(sums[0] / n_mb, 4), "train_pm_mm": round(sums[1] / n_mb, 4),
                    "train_nll": round(sums[2] / n_mb, 4), **{k: round(x, 4) for k, x in v.items()},
                    "epoch_s": round(ep_s, 1), "val_s": round(val_s, 1), "hours": round(hours, 3),
                    "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20), "best": int(better)})
        write_log(log_path, log)
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step, "best": best,
                    "best_epoch": best_epoch, "since": since}, MODELS / "refiner_last.pt")
        print(json.dumps(log[-1]), flush=True)
        if since >= a.patience:
            print(f"early stop: best epoch {best_epoch}, validation TRE {best:.3f} mm", flush=True)
            break
        if hours >= a.hours:
            print(f"time limit: {hours:.2f} h, best epoch {best_epoch}", flush=True)
            break


def write_log(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, restval="")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--sessions", default="0:900")
    pt = sub.add_parser("train")
    pt.add_argument("--epochs", type=int, default=80)
    pt.add_argument("--hours", type=float, default=3.0)
    pt.add_argument("--patience", type=int, default=10)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--resume", action="store_true")
    a = p.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
