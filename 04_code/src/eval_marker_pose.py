"""Marker pose and mandible overlay on the synthetic phone views of the bench, against the ground truth.

Input: two runs of render_phone_views.py, one at the photo resolution (DATA, 4080 x 3072) and one at the resolution of
a live stream (LIVE, 1920 x 1440, same field of view), both in the work folder (environment variable
IMPACTGUIDE_WORK):
    blender --background --factory-startup --python 04_code/src/render_phone_views.py
    blender --background --factory-startup --python 04_code/src/render_phone_views.py -- --out WORK/synthetic_pose_live --size 1920 1440
For each photo position and both resolutions:
- ChArUco detection and PnP on the clean render; pose error, and the error it puts on the mandible (3D, mm, and in the
  image, px) over all mandible vertices;
- jitter on a static view: TRIALS copies with sensor noise (Gaussian, NOISE DN) saved as JPEG (quality JPEG_Q), spread
  of the estimates around their mean;
- the overlay renderer (mesh_overlay.py) with the true pose against Blender's own mask of the mandible (IoU, mean
  distance between the outlines), and with the estimated pose (photo resolution);
- at the live resolution, the error with a Gaussian blur of BLUR px (defocus, image processing of the phone).
Writes 05_results/pose_synthetic.csv (rows view_1..view_4 and mean) and the check sheet
02_cad/renders/check_pose_overlay.png (photo resolution, estimated pose: corners, board axes, mandible silhouette and
edges).

    python 04_code/src/eval_marker_pose.py [--data DIR] [--live DIR] [--trials N]
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marker_pose as mp   # noqa: E402
import mesh_overlay as mo  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
DATA = WORK_ROOT / "synthetic_pose"
LIVE = WORK_ROOT / "synthetic_pose_live"
CSV = ROOT / "05_results" / "pose_synthetic.csv"
SHEET = ROOT / "02_cad" / "renders" / "check_pose_overlay.png"
TRIALS, NOISE, JPEG_Q, SEED = 30, 3.0, 90, 0
BLUR = 1.5
GREEN = (90, 220, 60)


def load(data, board_spec):
    gt = json.loads((data / "gt.json").read_text(encoding="utf-8"))
    spec = {k: gt["board"][k] for k in board_spec}
    if spec != board_spec:
        raise SystemExit(f"board of the renders in {data} {spec} differs from marker_pose.BOARD {board_spec}")
    return gt


def outline(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return np.vstack([c.reshape(-1, 2) for c in cs])


def outline_distance(a, b):
    """Mean symmetric distance (px) between the outlines of two binary masks."""
    d = []
    for p, other in ((outline(a), b), (outline(b), a)):
        img = np.full(a.shape, 255, np.uint8)
        img[tuple(outline(other)[:, ::-1].T)] = 0
        d.append(cv2.distanceTransform(img, cv2.DIST_L2, 5)[p[:, 1], p[:, 0]])
    return float(np.mean(np.concatenate(d)))


def iou(a, b):
    a, b = a > 0, b > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def mean_pose(Ts):
    """Chordal mean of the rotations, mean of the translations.

    Raises:
        ValueError: no pose.
    """
    if not len(Ts):
        raise ValueError("mean_pose of no pose")
    u, _, vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    T = np.eye(4)
    T[:3, :3] = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    T[:3, 3] = np.mean([T_[:3, 3] for T_ in Ts], axis=0)
    return T


def jitter(board, detector, image, K, X, rng, trials):
    """Spread of the pose over noisy JPEG copies of one image: RMS deviation from the mean pose."""
    Ts = []
    for _ in range(trials):
        noisy = np.clip(image + rng.normal(0.0, NOISE, image.shape), 0, 255).astype(np.uint8)
        jpg = cv2.imdecode(cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1], cv2.IMREAD_COLOR)
        est = mp.estimate_pose(board, *mp.detect(detector, jpg)[:2], K)
        if est is not None:
            Ts.append(est["T"])
    if not Ts:                      # the board was found in none of the noisy copies
        nan = float("nan")
        return {"found": 0, "rot_deg": nan, "trans_mm": nan, "tre_mm": nan, "overlay_px": nan}
    Tm = mean_pose(Ts)
    dev = [mp.pose_error(T, Tm) for T in Ts]
    tgt = [mp.target_error(T, Tm, X, K) for T in Ts]
    rms = lambda key, rows: float(np.sqrt(np.mean([r[key] ** 2 for r in rows])))
    return {"found": len(Ts), "rot_deg": rms("rot_deg", dev), "trans_mm": rms("trans_mm", dev),
            "tre_mm": rms("tre_rms_mm", tgt), "overlay_px": rms("overlay_rms_px", tgt)}


def evaluate(board, image, K, T_gt, X, rng, trials):
    """Detection, pose and its errors on one image, and the jitter over noisy copies.

    Raises:
        ValueError: the board is not found in the image.
    """
    detector = mp.make_detector(board, K)
    t0 = time.perf_counter()
    det = mp.detect(detector, image)
    detect_ms = (time.perf_counter() - t0) * 1e3
    est = mp.estimate_pose(board, det[0], det[1], K)
    if est is None:
        raise ValueError(f"board not found ({len(det[1])} corners)")
    return {"det": det, "est": est, "detect_ms": detect_ms, "err": mp.pose_error(est["T"], T_gt),
            "tgt": mp.target_error(est["T"], T_gt, X, K), "gt_rms": mp.reprojection_rms(board, det[0], det[1], T_gt, K),
            "jit": jitter(board, detector, image, K, X, rng, trials)}


def panel(image, det, est, ren, K, title, width=1000):
    out = mo.overlay(image, ren, GREEN, 0.25, 3)
    T = est["T"]
    cv2.drawFrameAxes(out, K, np.zeros(5), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], 20.0, 4)
    for c in det[0]:
        cv2.circle(out, tuple(np.round(c * 16).astype(int)), 6 * 16, (0, 0, 255), 3, cv2.LINE_AA, 4)
    pts = np.vstack([det[0], outline(ren.mask)])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, (y1 - y0) * 4 / 3) / 2 + 0.12 * max(x1 - x0, y1 - y0)
    xa, ya = int(max(cx - half, 0)), int(max(cy - half * 3 / 4, 0))
    crop = out[ya:int(cy + half * 3 / 4), xa:int(cx + half)]
    crop = cv2.resize(crop, (width, round(width * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.putText(crop, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return crop


def sheet(panels):
    h = max(p.shape[0] for p in panels)
    pad = lambda p: cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return np.vstack([np.hstack([pad(p) for p in panels[i:i + 2]]) for i in range(0, len(panels), 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--live", type=Path, default=LIVE)
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()
    gt, gt_live = load(args.data, mp.BOARD), load(args.live, mp.BOARD)
    mesh = np.load(args.data / "mandible_board.npz")
    X = mesh["vertices"]
    renderer = mo.MeshRenderer(X, mesh["faces"])
    board = mp.make_board()
    w, h = gt["image_size"]
    rng = np.random.default_rng(SEED)
    r3 = lambda x: round(x, 3) + 0.0   # no -0.0
    rows, panels = [], []
    for v, vl in zip(gt["views"], gt_live["views"]):
        k, K, T_gt, K_live = v["k"], np.array(v["K"]), np.array(v["camera_from_board"]), np.array(vl["K"])
        if not np.allclose(T_gt, vl["camera_from_board"], atol=1e-6):
            raise SystemExit(f"view {k}: the photo and live renders have different poses")
        image = cv2.imread(str(args.data / f"view_{k}.png"))
        blender_mask = np.where(cv2.imread(str(args.data / f"mandible_mask_{k}.png"), cv2.IMREAD_UNCHANGED)[..., 3] > 127,
                                255, 0).astype(np.uint8)
        try:
            a = evaluate(board, image, K, T_gt, X, rng, args.trials)
        except ValueError as e:
            raise SystemExit(f"view {k}, photo: {e}") from e
        ren_gt = renderer.render(K, None, T_gt, (w, h))
        ren = renderer.render(K, None, a["est"]["T"], (w, h))
        live = cv2.imread(str(args.live / f"view_{k}.png"))
        try:
            b = evaluate(board, live, K_live, T_gt, X, rng, args.trials)
        except ValueError as e:
            raise SystemExit(f"view {k}, live: {e}") from e
        blurred = cv2.GaussianBlur(live, (0, 0), BLUR)
        est_blur = mp.estimate_pose(board, *mp.detect(mp.make_detector(board, K_live), blurred)[:2], K_live)
        if est_blur is None:
            raise SystemExit(f"view {k}, live with a blur of {BLUR} px: board not found")
        tgt_blur = mp.target_error(est_blur["T"], T_gt, X, K_live)
        ren_live = renderer.render(K_live, None, b["est"]["T"], tuple(gt_live["image_size"]))
        for name, r in (("photo", a), ("live", b)):
            if r["jit"]["found"] < args.trials:
                print(f"view {k}, {name}: pose found in {r['jit']['found']} of {args.trials} noisy copies")

        rows.append({
            "key": f"view_{k}", "camera_deg": round(v["camera_deg"]), "distance_mm": round(v["distance_mm"]),
            "corners": len(a["det"][1]), "markers": a["det"][2], "reproj_px": r3(a["est"]["rms_px"]), "gt_reproj_px": r3(a["gt_rms"]),
            "rot_deg": r3(a["err"]["rot_deg"]), "trans_mm": r3(a["err"]["trans_mm"]), "depth_mm": r3(a["err"]["depth_mm"]),
            "tre_rms_mm": r3(a["tgt"]["tre_rms_mm"]), "tre_max_mm": r3(a["tgt"]["tre_max_mm"]),
            "overlay_px": round(a["tgt"]["overlay_rms_px"], 2),
            "jit_rot_deg": round(a["jit"]["rot_deg"], 4), "jit_trans_mm": round(a["jit"]["trans_mm"], 4),
            "jit_tre_mm": round(a["jit"]["tre_mm"], 4), "jit_overlay_px": r3(a["jit"]["overlay_px"]),
            "iou_gt": round(iou(ren_gt.mask, blender_mask), 4), "outline_px": round(outline_distance(ren_gt.mask, blender_mask), 2),
            "iou_est": round(iou(ren.mask, blender_mask), 4),
            "live_corners": len(b["det"][1]), "live_rot_deg": r3(b["err"]["rot_deg"]), "live_trans_mm": r3(b["err"]["trans_mm"]),
            "live_tre_mm": r3(b["tgt"]["tre_rms_mm"]), "live_overlay_px": round(b["tgt"]["overlay_rms_px"], 2),
            "live_jit_tre_mm": round(b["jit"]["tre_mm"], 4), "live_jit_overlay_px": r3(b["jit"]["overlay_px"]),
            "live_blur_tre_mm": r3(tgt_blur["tre_rms_mm"]), "live_blur_overlay_px": round(tgt_blur["overlay_rms_px"], 2),
            "detect_ms": round(a["detect_ms"]), "render_ms": round(ren.ms),
            "live_detect_ms": round(b["detect_ms"]), "live_render_ms": round(ren_live.ms),
        })
        title = (f"view {k}, camera {v['camera_deg']:.0f} deg: rotation {a['err']['rot_deg']:.3f} deg, "
                 f"mandible {a['tgt']['tre_rms_mm']:.3f} mm, {a['tgt']['overlay_rms_px']:.2f} px")
        panels.append(panel(image, a["det"], a["est"], ren, K, title))
        print({key: rows[-1][key] for key in ("key", "rot_deg", "trans_mm", "tre_rms_mm", "jit_tre_mm", "iou_gt", "outline_px",
                                              "live_tre_mm", "live_jit_tre_mm", "live_blur_tre_mm", "render_ms", "live_render_ms")})

    mean = {"key": "mean"}
    for col in rows[0]:
        if col != "key":
            vals = [r[col] for r in rows]
            digits = max(len(str(x).split(".")[1]) if "." in str(x) else 0 for x in vals)
            mean[col] = round(float(np.mean(vals)), digits) if digits else round(float(np.mean(vals)))
    rows.append(mean)
    with CSV.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    cv2.imwrite(str(SHEET), sheet(panels))
    print(f"written {CSV.relative_to(ROOT)} and {SHEET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
