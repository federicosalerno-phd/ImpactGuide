"""Quick checks of refiner.py: update parametrisation and kinematic layer, crop geometry, network shapes and
view invariance, losses, masks, and the crop of a twin view when the twin dataset is there. CPU, about 10 s.

    python 04_code/src/test_refiner.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correct as co      # noqa: E402
import refiner as rf      # noqa: E402

FAILED = []


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILED.append(what)


def test_rotations():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(8, 3)) * 0.1
    R = rf.so3_exp(torch.from_numpy(w)).numpy()
    ref = np.stack([co.rodrigues(x) for x in w])
    check(np.abs(R - ref).max() < 1e-9, "so3_exp equals cv2.Rodrigues")


def test_parametrisation():
    rng = np.random.default_rng(2)
    c, o = np.array([0.0, -51.2, -19.3]), np.array([1.8, 28.7, 11.7])
    X = rng.normal(size=(50, 3)) * 60
    for mode in ("a1", "rigid"):
        th = np.r_[rng.normal(size=3) * 3, rng.normal(size=3) * 3][: 3 if mode == "a1" else 6]
        D = rf.delta(th, mode, c, o)
        check(np.allclose(rf.params(D, mode, c, o), th, atol=1e-9), f"{mode}: params(delta(theta)) == theta")
        Y = rf.move(torch.from_numpy(th[None]), torch.from_numpy(X), mode, torch.from_numpy(c), torch.from_numpy(o))[0]
        check(np.allclose(Y.numpy(), X @ D[:3, :3].T + D[:3, 3], atol=1e-9), f"{mode}: kinematic layer equals delta()")
        P = rf.perturbation(mode, rng, c, o)
        th_inv = rf.params(np.linalg.inv(P), mode, c, o)
        check(np.allclose(rf.delta(th_inv, mode, c, o) @ P, np.eye(4), atol=1e-9), f"{mode}: target undoes the perturbation")
    D = rf.delta(np.array([2.0, -1.0, 0.5]), "a1", c, o)
    check(np.allclose(D[:3, :3] @ c + D[:3, 3], c), "a1: the ball centre is a fixed point")
    pb = co.Problem(type("R", (), {"v": X})(), [], np.eye(4), "rigid")
    R = co.rodrigues(np.radians([1.0, 2.0, -1.5]))
    s = np.array([0.4, -0.3, 1.1])
    check(np.allclose(rf.delta(np.r_[np.degrees(co.rotvec(R)), s], "rigid", c, X.mean(0)), pb.delta(R, s)),
          "rigid: same update as correct.Problem.delta")
    pp = rf.PriorProblem(type("R", (), {"v": X})(), [], np.eye(4), "rigid", sigma=np.array([1.0, 2.0, 0.5, 1.0, 3.0, 2.0]))
    d = np.array([0.01, -0.02, 0.005, 0.3, -0.1, 0.2])
    r, J = pp.residuals(co.rodrigues(d[:3]), d[3:], [])
    sig = np.r_[np.radians([1.0, 2.0, 0.5]), [1.0, 3.0, 2.0]]
    check(np.allclose(r, rf.PRIOR_MM * d / sig) and np.allclose(J, np.diag(rf.PRIOR_MM / sig)),
          "prior residuals and Jacobian of PriorProblem")


def test_crop_geometry():
    K = np.array([[1500.0, 0, 1019.5], [0, 1500.0, 767.5], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 3] = [5.0, -8.0, 330.0]
    p = np.array([[3.0, 4.0, 2.0]])
    # make_source: a dot drawn at the projection of p lands at the projection with the resampled intrinsics
    img = np.zeros((1536, 2040, 3), np.uint8)
    u = rf.project(K, None, T, p)[0]
    cv2.circle(img, (int(round(u[0])), int(round(u[1]))), 6, (255, 255, 255), -1)
    src = rf.make_source(img, img[..., 0], K, None, np.eye(4), T, p[0])
    ys, xs = np.nonzero(src.obs > 0)
    w = src.obs[ys, xs].astype(float)
    us = rf.project(src.K, None, T, p)[0]
    got = np.array([np.sum(xs * w) / w.sum(), np.sum(ys * w) / w.sum()])
    check(np.linalg.norm(got - us) < 0.3, f"make_source keeps the projection (error {np.linalg.norm(got - us):.3f} px)")
    check(abs(us[0] - (rf.REGION_PX - 1) / 2) < 0.6 and abs(us[1] - (rf.REGION_PX - 1) / 2) < 0.6, "source centred on the point")
    # crop_affine agrees with synth_data.crop_K
    centre, side = rf.window(src.K, src.dist, T, p[0] + [10.0, -5.0, 0.0], rf.CROP_MM)
    M = rf.crop_affine(centre, side)
    Kc = rf.sd.crop_K(src.K, centre, side, rf.CROP_PX)
    q = rf.project(src.K, None, T, p)[0]
    check(np.allclose(M @ np.r_[q, 1.0], rf.project(Kc, None, T, p)[0], atol=1e-9), "crop_affine agrees with crop_K")
    check(abs(side - rf.CROP_MM * src.K[0, 0] / 332.0) < 1e-6, "crop side from the depth of the point")
    check(abs(src.K[0, 0] / 332.0 * rf.REGION_MM - rf.REGION_PX) < 1.0, "source at the crop scale")


def test_network():
    torch.manual_seed(0)
    net = rf.RefinerNet(pretrained=False).eval()
    x = torch.randint(0, 255, (2, 3, 6, 128, 128), dtype=torch.uint8)
    g = torch.randn(2, 3, 9)
    with torch.no_grad():
        mu, lv = net(rf.normalise(x), g, "rigid")
        mu2, _ = net(rf.normalise(x[:, [2, 0, 1]]), g[:, [2, 0, 1]], "rigid")
        mu1, lv1 = net(rf.normalise(x[:, :1]), g[:, :1], "a1")
    check(mu.shape == (2, 6) and lv.shape == (2, 6) and mu1.shape == (2, 3) and lv1.shape == (2, 3), "head shapes (6 and 3)")
    check(torch.allclose(mu, mu2, atol=1e-5), "fusion does not depend on the order of the views")
    check(float(mu.abs().max()) < 1.0, "small initial update")
    w = net.enc.conv1.weight
    check(w.shape[1] == 6 and torch.allclose(w[:, 3], w[:, :3].mean(1)), "first layer: 6 channels, extra ones at the RGB mean")
    th = torch.randn(2, 6, dtype=torch.float64)
    X = torch.randn(40, 3, dtype=torch.float64) * 50
    c, o = torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    pm, nll = rf.losses(th, torch.zeros(2, 6, dtype=torch.float64), th, X, "rigid", c, o)
    check(float(pm) < 1e-9 and abs(float(nll)) < 1e-12, "losses vanish at the truth with unit variance")
    mu = th.clone().requires_grad_(True)
    lv = torch.zeros(2, 6, dtype=torch.float64, requires_grad=True)
    pm, nll = rf.losses(mu + 0.5, lv, th, X, "rigid", c, o)
    (pm + nll).backward()
    check(bool(torch.isfinite(mu.grad).all() and torch.isfinite(lv.grad).all() and lv.grad.abs().sum() > 0),
          "finite gradients for parameters and variances")


def test_masks():
    lab = np.zeros((40, 40), np.uint8)
    lab[10:30, 10:30] = 1
    lab[10:30, 25:30] = 9          # cap over part of the face
    sam = np.zeros((40, 40), np.uint8)
    sam[8:32, 8:32] = 255
    obs = rf.observed_mask(sam, lab)
    check(obs[20, 27] == 0 and obs[20, 15] == 255, "observed mask: covers taken out of the SAM mask")
    ign = rf.ignore_mask(lab)
    check(ign[20, 23] == 255 and ign[20, 15] == 0, "ignore mask: covers dilated")


def test_twin_crop():
    if not (rf.SYNTH / "meshes" / "face_phantom_r20k.npz").exists():
        print("skip  twin crop (no twin dataset)")
        return
    import gl_overlay
    face = rf.load_face()
    s = rf.sd.sessions(rf.SYNTH)[0]
    fr = rf.sd.frames(s)[0]
    d = rf.sd.load(s, fr)
    ren = gl_overlay.GLMeshRenderer(face.v, face.f)
    src = rf.make_source(d["rgb"], d["masks"]["face"].astype(np.uint8) * 255, d["K"], d["dist"], d["T_C_M"], d["T_M_B"], face.o)
    v = rf.view_inputs(ren, src, src.T_C_M @ d["T_M_B"], face.o)
    occ = rf.make_source(d["rgb"], rf.occluders(d["labels"]).astype(np.uint8) * 255, d["K"], d["dist"], d["T_C_M"],
                         d["T_M_B"], face.o)
    free = rf.view_inputs(ren, occ, occ.T_C_M @ d["T_M_B"], face.o)["obs"] < 128
    a, b = (v["ren"] > 0) & free, (v["obs"] > 127) & free
    iou = (a & b).sum() / max((a | b).sum(), 1)
    check(iou > 0.95, f"twin crop at the truth: rendered and exact face masks agree off the covers (IoU {iou:.3f})")
    if "--save" in sys.argv:
        cv2.imwrite(sys.argv[sys.argv.index("--save") + 1], np.hstack([v["rgb"], np.dstack([v["obs"], v["ren"], v["edge"]])]))
    check(v["edge"].any() and stack_ok(v), "edges drawn, 6-channel stack")
    check(len(face.visible) < len(face.f), f"visible face for the SAM box: {len(face.visible)} of {len(face.f)} triangles")


def stack_ok(v):
    x = rf.stack(v)
    return x.shape == (6, rf.CROP_PX, rf.CROP_PX) and x.dtype == np.uint8


if __name__ == "__main__":
    for t in (test_rotations, test_parametrisation, test_crop_geometry, test_network, test_masks, test_twin_crop):
        t()
    print(f"{len(FAILED)} failed" if FAILED else "all passed")
    sys.exit(1 if FAILED else 0)
