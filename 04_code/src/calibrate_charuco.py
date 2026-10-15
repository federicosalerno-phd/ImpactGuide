"""ChArUco calibration of the phone camera intrinsics (T5.1, T5.3).

The calibration board is 11 x 8 squares of 24 mm with 18 mm DICT_5X5_100
markers, centred on an A4 page glued to a flat support. The bench marker uses
DICT_4X4_50, so the two boards never share a marker. If the printed squares
do not measure 24 mm, pass the caliper value with --square (the marker size
scales with it).

Usage:
    python calibrate_charuco.py board [--out DIR]
        printable page: PDF at 1:1 scale and PNG
    python calibrate_charuco.py synth [--out DIR] [--n 40] [--seed 0]
        synthetic photos of the board taken with known intrinsics (truth.json)
    python calibrate_charuco.py calibrate IMAGES [--out FILE] [--truth FILE]
        intrinsics from a folder of photos, compared with the truth if given
    python calibrate_charuco.py check IMAGES --intrinsics FILE
        start-of-session check: are the stored intrinsics still valid?
    python calibrate_charuco.py test [--work DIR]
        synth + calibrate + check on synthetic photos; exit code 1 on failure

Real photos go in 03_data/raw/calibration/<yyyymmdd>_<mode>/, intrinsics in
03_data/processed/calibration/intrinsics_<mode>.json. Synthetic photos and the
printable board go in C:\\work\\marker-impact\\calibration.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

WORK = Path(r"C:\work\marker-impact\calibration")
BOARD = {"squares": (11, 8), "square": 24.0, "marker": 18.0, "dictionary": "DICT_5X5_100"}
PAGE = (297.0, 210.0)  # A4 landscape, mm
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MIN_CORNERS = 12  # per view
MIN_VIEWS = 8

# Synthetic camera: Poco X7 Pro main camera in 12.5 MP photo mode (4080 x 3072,
# 26 mm equivalent focal length, i.e. about 3070 px), mild barrel distortion.
SYNTH_SIZE = (4080, 3072)
SYNTH_K = np.array([[3071.4, 0.0, 2052.6], [0.0, 3069.8, 1528.9], [0.0, 0.0, 1.0]])
SYNTH_DIST = np.array([0.105, -0.262, 0.0008, -0.0005, 0.214])
SYNTH_RANGE = (260.0, 400.0)  # board centre distance, mm
SYNTH_TILT = 50.0  # maximum board tilt, degrees
SYNTH_ROLL = 45.0  # maximum rotation about the optical axis, degrees
SYNTH_BLUR = (0.4, 2.0)  # Gaussian defocus sigma, px
SYNTH_NOISE = (1.5, 5.0)  # Gaussian noise sigma, grey levels
TEXTURE_PPM = 12  # board texture resolution for rendering, px per mm

# Acceptance on synthetic photos (test). RMS from PLAN.md T5.1; the others
# bound the error on the recovered camera model.
MAX_RMS = 0.5  # px
MAX_FOCAL_ERROR = 0.002  # relative
MAX_CENTRE_ERROR = 4.0  # px
MAX_MODEL_DIFFERENCE = 1.0  # px, anywhere in the image
# Start-of-session check: changes below these floors are accepted even when they
# exceed 3 standard deviations. Refocusing from 28 to 35 cm changes f by about 0.4 %.
CHECK_FOCAL = 0.001  # relative
CHECK_CENTRE = 2.0  # px


def make_board(square=BOARD["square"]):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD["dictionary"]))
    marker = square * BOARD["marker"] / BOARD["square"]
    return cv2.aruco.CharucoBoard(BOARD["squares"], square, marker, d)


def page_image(board, ppm):
    """A4 page with the board centred, `ppm` px per mm, and the board origin on the page in mm."""
    nx, ny = board.getChessboardSize()
    sq = board.getSquareLength()
    bw, bh = round(nx * sq * ppm), round(ny * sq * ppm)
    pw, ph = round(PAGE[0] * ppm), round(PAGE[1] * ppm)
    x0, y0 = (pw - bw) // 2, (ph - bh) // 2
    page = np.full((ph, pw), 255, np.uint8)
    page[y0:y0 + bh, x0:x0 + bw] = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    return page, (x0 / ppm, y0 / ppm)


def write_board(out, square):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    board = make_board(square)
    nx, ny = board.getChessboardSize()
    ppm = 24  # about 610 dpi; the PDF places the image at its exact size in mm
    page, (ox, oy) = page_image(board, ppm)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"charuco_{nx}x{ny}_{square:g}mm_{BOARD['dictionary'].lower()}_a4"
    cv2.imwrite(str(out / f"{stem}.png"), page)
    mm = 1 / 25.4
    fig = plt.figure(figsize=(PAGE[0] * mm, PAGE[1] * mm))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(page, cmap="gray", vmin=0, vmax=255, interpolation="none",
              extent=(0, PAGE[0], PAGE[1], 0))
    ax.set_xlim(0, PAGE[0])
    ax.set_ylim(PAGE[1], 0)
    ax.axis("off")
    note = (f"ChArUco {nx} x {ny}, square {square:g} mm, marker {board.getMarkerLength():g} mm, "
            f"{BOARD['dictionary']}. Print at 100% (actual size). Check: 10 squares = {10 * square:g} mm.")
    ax.text(ox, oy + ny * square + 2.5, note, fontsize=6, va="top", color="0.4")
    fig.savefig(out / f"{stem}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"board: {out / stem}.pdf, .png")
    return out / f"{stem}.pdf"


def rotation(axis_angle):
    return cv2.Rodrigues(np.asarray(axis_angle, float).reshape(3, 1))[0]


def normalised_bounds(K, dist, size):
    """Largest |x| and |y| of the normalised image plane seen inside the image."""
    w, h = size
    edge = [(u, v) for u in np.linspace(0, w - 1, 33) for v in (0, h - 1)]
    edge += [(u, v) for v in np.linspace(0, h - 1, 25) for u in (0, w - 1)]
    xn = cv2.undistortPoints(np.array(edge, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    return np.abs(xn).max(axis=0)


def sample_pose(rng, board, K, dist, size):
    """Random board pose with at least 24 chessboard corners inside the image."""
    nx, ny = board.getChessboardSize()
    sq = board.getSquareLength()
    centre = np.array([nx * sq / 2, ny * sq / 2, 0.0])
    corners = board.getChessboardCorners()
    xmax, ymax = normalised_bounds(K, dist, size)
    w, h = size
    while True:
        a = rng.uniform(0, 2 * np.pi)
        tilt = np.radians(rng.uniform(0, SYNTH_TILT)) * np.array([np.cos(a), np.sin(a), 0])
        roll = np.radians(rng.uniform(-SYNTH_ROLL, SYNTH_ROLL)) * np.array([0, 0, 1.0])
        R = rotation(roll) @ rotation(tilt)
        aim = np.linalg.solve(K, [rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h, 1])
        t = rng.uniform(*SYNTH_RANGE) * aim / aim[2] - R @ centre
        cam = corners @ R.T + t
        xn = cam[:, :2] / cam[:, 2:]
        seen = (np.abs(xn[:, 0]) < xmax) & (np.abs(xn[:, 1]) < ymax)
        uv = cv2.projectPoints(corners, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2)
        seen &= (uv[:, 0] > 20) & (uv[:, 0] < w - 21) & (uv[:, 1] > 20) & (uv[:, 1] < h - 21)
        if seen.sum() >= 24:
            return R, t


def pixel_rays(K, dist, size, rows=256):
    """Normalised image coordinates (x, y) of the ray through every pixel centre."""
    w, h = size
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-12)
    rays = np.empty((h, w, 2), np.float32)
    u = np.arange(w, dtype=np.float64)
    for v0 in range(0, h, rows):
        v1 = min(v0 + rows, h)
        uu, vv = np.meshgrid(u, np.arange(v0, v1, dtype=np.float64))
        pix = np.stack([uu, vv], -1).reshape(-1, 1, 2)
        rays[v0:v1] = cv2.undistortPoints(pix, K, dist, None, None, None, crit).reshape(v1 - v0, w, 2)
    return rays


def board_maps(rays, R, t, origin, ppm):
    """remap() maps from the image pixels to the page texture."""
    # the ray [x, y, 1] meets the board plane z = 0 at [X, Y, 1] ~ inv([r1 r2 t]) [x, y, 1]
    A = np.linalg.inv(np.column_stack([R[:, 0], R[:, 1], t])).astype(np.float32)
    x, y = rays[..., 0], rays[..., 1]
    s = A[2, 0] * x + A[2, 1] * y + A[2, 2]
    front = s > 1e-9
    s = np.where(front, s, 1)
    # texture pixel i covers [i, i + 1) / ppm mm, remap() samples pixel centres at integers
    mx = np.where(front, ((A[0, 0] * x + A[0, 1] * y + A[0, 2]) / s + origin[0]) * ppm - 0.5, -1e4)
    my = np.where(front, ((A[1, 0] * x + A[1, 1] * y + A[1, 2]) / s + origin[1]) * ppm - 0.5, -1e4)
    return mx.astype(np.float32), my.astype(np.float32)


def render(page, origin, rays, R, t, rng):
    h, w = rays.shape[:2]
    size = (w, h)
    mx, my = board_maps(rays, R, t, origin, TEXTURE_PPM)
    paper = cv2.remap(page, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = cv2.remap(np.full_like(page, 255), mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    black, white = rng.uniform(25, 60), rng.uniform(170, 235)
    img = black + (white - black) * paper.astype(np.float32) / 255
    # background: grey with low-frequency texture
    bg = cv2.resize(rng.uniform(60, 140, (12, 16)).astype(np.float32), size, interpolation=cv2.INTER_CUBIC)
    alpha = inside.astype(np.float32) / 255
    img = alpha * img + (1 - alpha) * bg
    # uneven lighting
    gx, gy = rng.uniform(-0.25, 0.25, 2)
    uu, vv = np.meshgrid(np.linspace(-0.5, 0.5, w, dtype=np.float32), np.linspace(-0.5, 0.5, h, dtype=np.float32))
    img *= 1 + gx * uu + gy * vv
    blur = rng.uniform(*SYNTH_BLUR)
    img = cv2.GaussianBlur(img, (0, 0), blur)
    noise = rng.uniform(*SYNTH_NOISE)
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), blur, noise


def synth(out, n, seed, K=SYNTH_K, dist=SYNTH_DIST, size=SYNTH_SIZE, prefix="synth"):
    rng = np.random.default_rng(seed)
    board = make_board()
    page, origin = page_image(board, TEXTURE_PPM)
    page = cv2.GaussianBlur(page, (0, 0), 0.6)  # anti-aliasing of the texture
    rays = pixel_rays(K, dist, size)
    out.mkdir(parents=True, exist_ok=True)
    views = []
    for i in range(n):
        R, t = sample_pose(rng, board, K, dist, size)
        img, blur, noise = render(page, origin, rays, R, t, rng)
        name = f"{prefix}_{i:03d}.jpg"
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        views.append({"image": name, "rvec": cv2.Rodrigues(R)[0].ravel().tolist(), "tvec": t.tolist(),
                      "blur_px": round(blur, 3), "noise_grey": round(noise, 3)})
    truth = {"image_size": list(size), "K": K.tolist(), "dist": dist.tolist(), "seed": seed, "views": views}
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"synth: {n} photos in {out}")


def find_images(folder):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        sys.exit(f"no photos in {folder}")
    return paths


def detect(paths, board):
    """Chessboard corners of every usable photo: {name: (object points, image points)} and image size."""
    detector = cv2.aruco.CharucoDetector(board)
    views, skipped, size = {}, [], None
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)  # applies the EXIF orientation
        if img is None:
            skipped.append((p.name, "unreadable"))
            continue
        if size is None:
            size = img.shape[::-1]
        elif img.shape[::-1] != size:
            sys.exit(f"{p.name}: size {img.shape[::-1]} differs from {size}; one camera mode per folder")
        corners, ids, _, _ = detector.detectBoard(img)
        if ids is None or len(ids) < MIN_CORNERS:
            skipped.append((p.name, f"{0 if ids is None else len(ids)} corners"))
            continue
        if board.checkCharucoCornersCollinear(ids):
            skipped.append((p.name, "collinear corners"))
            continue
        obj, imgp = board.matchImagePoints(corners, ids)
        views[p.name] = (obj.astype(np.float32), imgp.astype(np.float32))
    return views, skipped, size


def solve(views, size):
    obj = [v[0] for v in views.values()]
    img = [v[1] for v in views.values()]
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    rms, K, dist, _, _, std, _, per_view = cv2.calibrateCameraExtended(obj, img, size, None, None, criteria=crit)
    return rms, K, dist.ravel(), std.ravel(), dict(zip(views, per_view.ravel()))


def calibrate(views, size):
    """Calibrate, drop the views with an error above 3 times the median, calibrate again."""
    if len(views) < MIN_VIEWS:
        sys.exit(f"only {len(views)} usable photos, at least {MIN_VIEWS} needed")
    rms, K, dist, std, per_view = solve(views, size)
    limit = max(3 * float(np.median(list(per_view.values()))), 0.3)
    dropped = sorted(k for k, e in per_view.items() if e > limit)
    if dropped and len(views) - len(dropped) >= MIN_VIEWS:
        views = {k: v for k, v in views.items() if k not in dropped}
        rms, K, dist, std, per_view = solve(views, size)
    else:
        dropped = []
    return {
        "image_size": list(size), "K": K.tolist(), "dist": dist.tolist(),
        "rms_px": round(float(rms), 4), "n_views": len(views),
        "n_corners": int(sum(len(v[0]) for v in views.values())),
        "std": {k: round(float(s), 4) for k, s in zip(["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"], std)},
        "per_view_rms_px": {k: round(float(e), 4) for k, e in per_view.items()},
        "dropped": dropped,
    }


def model_difference(K1, d1, K2, d2, size, step=32):
    """Pixel displacement between two camera models over the image: max and RMS.

    Each pixel is back-projected with model 1 and projected again with model 2.
    """
    w, h = size
    uu, vv = np.meshgrid(np.arange(0, w, step, dtype=np.float64), np.arange(0, h, step, dtype=np.float64))
    pix = np.stack([uu, vv], -1).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-12)
    xn = cv2.undistortPoints(pix, np.asarray(K1), np.asarray(d1), None, None, None, crit).reshape(-1, 2)
    rays = np.column_stack([xn, np.ones(len(xn))])
    back = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), np.asarray(K2), np.asarray(d2))[0]
    d = np.linalg.norm(back.reshape(-1, 2) - pix.reshape(-1, 2), axis=1)
    return float(d.max()), float(np.sqrt(np.mean(d ** 2)))


def compare(cal, ref):
    """Differences of `cal` from `ref` (truth or stored intrinsics)."""
    K, Kr = np.array(cal["K"]), np.array(ref["K"])
    dmax, drms = model_difference(Kr, ref["dist"], K, cal["dist"], cal["image_size"])
    return {
        "fx_rel": float(K[0, 0] / Kr[0, 0] - 1), "fy_rel": float(K[1, 1] / Kr[1, 1] - 1),
        "cx_px": float(K[0, 2] - Kr[0, 2]), "cy_px": float(K[1, 2] - Kr[1, 2]),
        "model_max_px": dmax, "model_rms_px": drms,
    }


def summary(cal, skipped):
    K = np.array(cal["K"])
    s = cal["std"]
    print(f"views {cal['n_views']} (corners {cal['n_corners']}), RMS {cal['rms_px']:.3f} px")
    print(f"fx {K[0, 0]:.2f} +/- {s['fx']:.2f}  fy {K[1, 1]:.2f} +/- {s['fy']:.2f}  "
          f"cx {K[0, 2]:.2f} +/- {s['cx']:.2f}  cy {K[1, 2]:.2f} +/- {s['cy']:.2f}")
    print("dist " + " ".join(f"{d:.5f}" for d in cal["dist"]))
    for name, why in skipped:
        print(f"skipped {name}: {why}")
    for name in cal["dropped"]:
        print(f"dropped {name}: per-view error above 3 times the median")


def print_compare(c):
    print(f"error fx {100 * c['fx_rel']:+.3f} %  fy {100 * c['fy_rel']:+.3f} %  "
          f"cx {c['cx_px']:+.2f} px  cy {c['cy_px']:+.2f} px  "
          f"model max {c['model_max_px']:.3f} px, RMS {c['model_rms_px']:.3f} px")


def run_calibrate(images, out, truth, square):
    board = make_board(square)
    views, skipped, size = detect(find_images(images), board)
    cal = calibrate(views, size)
    nx, ny = board.getChessboardSize()
    cal = {"images": str(Path(images)), "date": date.today().isoformat(),
           "board": {"squares": [nx, ny], "square_mm": square, "marker_mm": board.getMarkerLength(),
                     "dictionary": BOARD["dictionary"]},
           "skipped": dict(skipped), **cal}
    summary(cal, skipped)
    if truth:
        cal["error"] = compare(cal, json.loads(Path(truth).read_text()))
        print_compare(cal["error"])
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(cal, indent=1))
        print(f"intrinsics: {out}")
    return cal


def run_check(images, intrinsics, square):
    """Start-of-session check of the stored intrinsics on about 10 new photos.

    Refocusing changes mostly the focal length and the principal point, so these
    are estimated again with the stored distortion; 10 photos are too few for the
    whole model. The stored intrinsics stay valid when each change is within 3
    combined standard deviations or below the floor, and when they reproject the
    new photos with an RMS below MAX_RMS.

    Returns (ok, report): report holds the RMS of the stored model on these
    photos, the views used and skipped, and per parameter the stored and new
    value, the change, the limit and whether it passed.
    """
    ref = json.loads(Path(intrinsics).read_text())
    views, skipped, size = detect(find_images(images), make_board(square))
    if list(size) != ref["image_size"]:
        sys.exit(f"photo size {size} differs from the calibration {tuple(ref['image_size'])}: wrong camera mode")
    if len(views) < MIN_VIEWS:
        sys.exit(f"only {len(views)} usable photos, at least {MIN_VIEWS} needed")
    for name, why in skipped:
        print(f"skipped {name}: {why}")
    obj = [v[0] for v in views.values()]
    img = [v[1] for v in views.values()]
    K0, d0 = np.array(ref["K"]), np.array(ref["dist"])
    # stored model on these photos: pose of each view by PnP
    sq = sum(float(np.sum((cv2.projectPoints(o, *cv2.solvePnP(o, i, K0, d0)[1:], K0, d0)[0].reshape(-1, 2)
                           - i.reshape(-1, 2)) ** 2)) for o, i in zip(obj, img))
    rms_stored = np.sqrt(sq / sum(len(o) for o in obj))
    print(f"views {len(views)}, stored intrinsics reproject them with RMS {rms_stored:.3f} px")
    ok = bool(rms_stored < MAX_RMS)
    report = {"images": str(Path(images)), "intrinsics": str(Path(intrinsics)), "n_views": len(views),
              "skipped": dict(skipped), "rms_stored_px": round(float(rms_stored), 4), "rms_ok": ok, "params": {}}
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
             | cv2.CALIB_FIX_TANGENT_DIST)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    K, std = cv2.calibrateCameraExtended(obj, img, size, K0.copy(), d0.copy(), flags=flags, criteria=crit)[1:6:4]
    for i, (name, r, c) in enumerate([("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)]):
        delta = K[r, c] - K0[r, c]
        limit = max(3 * np.hypot(std[i, 0], ref["std"][name]), CHECK_FOCAL * K0[r, c] if r == c else CHECK_CENTRE)
        ok &= bool(abs(delta) <= limit)
        report["params"][name] = {"stored": round(float(K0[r, c]), 3), "now": round(float(K[r, c]), 3),
                                  "change": round(float(delta), 3), "limit": round(float(limit), 3),
                                  "ok": bool(abs(delta) <= limit)}
        print(f"{name} {K0[r, c]:.2f} -> {K[r, c]:.2f}  change {delta:+.2f} px, limit {limit:.2f} px"
              + ("" if abs(delta) <= limit else "  CHANGED"))
    print("check: " + ("ok, stored intrinsics still valid" if ok else "FAILED, recalibrate"))
    report["ok"] = ok
    return ok, report


def run_test(work):
    """Synthetic end-to-end test: 40 calibration photos, then two session checks."""
    synth(work / "synth", 40, seed=0)
    cal = run_calibrate(work / "synth", work / "synth_intrinsics.json", work / "synth" / "truth.json",
                        BOARD["square"])
    e = cal["error"]
    ok = (cal["rms_px"] < MAX_RMS and abs(e["fx_rel"]) < MAX_FOCAL_ERROR and abs(e["fy_rel"]) < MAX_FOCAL_ERROR
          and abs(e["cx_px"]) < MAX_CENTRE_ERROR and abs(e["cy_px"]) < MAX_CENTRE_ERROR
          and e["model_max_px"] < MAX_MODEL_DIFFERENCE)
    print("calibration: " + ("PASS" if ok else "FAIL"))
    # session check with the same camera: must pass
    synth(work / "check_same", 10, seed=1, prefix="check")
    same = run_check(work / "check_same", work / "synth_intrinsics.json", BOARD["square"])[0]
    # session check after refocusing (focal length +0.3 %, distortion unchanged): must fail
    K = SYNTH_K.copy()
    K[:2, :2] *= 1.003
    synth(work / "check_refocused", 10, seed=2, K=K, prefix="check")
    moved = run_check(work / "check_refocused", work / "synth_intrinsics.json", BOARD["square"])[0]
    ok = ok and same and not moved
    print("test: " + ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("board")
    a.add_argument("--out", type=Path, default=WORK)
    a.add_argument("--square", type=float, default=BOARD["square"])
    a = sub.add_parser("synth")
    a.add_argument("--out", type=Path, default=WORK / "synth")
    a.add_argument("--n", type=int, default=40)
    a.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("calibrate")
    a.add_argument("images", type=Path)
    a.add_argument("--out", type=Path)
    a.add_argument("--truth", type=Path)
    a.add_argument("--square", type=float, default=BOARD["square"], help="measured square size, mm")
    a = sub.add_parser("check")
    a.add_argument("images", type=Path)
    a.add_argument("--intrinsics", type=Path, required=True)
    a.add_argument("--square", type=float, default=BOARD["square"], help="measured square size, mm")
    a = sub.add_parser("test")
    a.add_argument("--work", type=Path, default=WORK)
    args = ap.parse_args()

    if args.cmd == "board":
        write_board(args.out, args.square)
    elif args.cmd == "synth":
        synth(args.out, args.n, args.seed)
    elif args.cmd == "calibrate":
        run_calibrate(args.images, args.out, args.truth, args.square)
    elif args.cmd == "check":
        sys.exit(0 if run_check(args.images, args.intrinsics, args.square)[0] else 1)
    elif args.cmd == "test":
        sys.exit(0 if run_test(args.work) else 1)


if __name__ == "__main__":
    main()
