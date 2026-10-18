"""ChArUco calibration of the phone camera intrinsics (T5.1, T5.3).

The printed calibration board is 11 x 8 squares of 24 mm with 18 mm
DICT_5X5_100 markers, centred on an A4 page glued to a flat support. The bench
marker uses DICT_4X4_50, so the two boards never share a marker. If the printed
squares do not measure 24 mm, pass the caliper value with --square (the marker
size scales with it).

The same board can be shown on a monitor instead (screen_board.py): same
dictionary and markers about 3/4 of the square, but as many squares as the
monitor holds, each a whole number of its pixels. Its layout and true square
size go with the photos in board.json in their folder (written by ImpactGuide),
which calibrate and check read when it is there; --squares and --marker set
them by hand.

Usage:
    python calibrate_charuco.py board [--out DIR]
        printable page: PDF at 1:1 scale and PNG
    python calibrate_charuco.py synth [--out DIR] [--n 40] [--seed 0]
        synthetic photos of the board taken with known intrinsics (truth.json)
    python calibrate_charuco.py calibrate IMAGES [--out FILE] [--truth FILE] [--square MM --squares NXxNY --marker MM]
        intrinsics from a folder of photos, compared with the truth if given
    python calibrate_charuco.py check IMAGES --intrinsics FILE [--square MM --squares NXxNY --marker MM]
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
PAGE_MARGIN = 10.0  # mm around a board larger than the page (synthetic photos of a board on a monitor)
BOARD_FILE = "board.json"  # the board of the photos of a folder: squares, square_mm, marker_mm, dictionary
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


def make_board(square=BOARD["square"], squares=None, marker=None):
    """The calibration board: the A4 one (11 x 8, marker 3/4 of the square) unless squares or marker are given."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD["dictionary"]))
    if marker is None:
        marker = square * BOARD["marker"] / BOARD["square"]
    nx, ny = squares or BOARD["squares"]
    return cv2.aruco.CharucoBoard((int(nx), int(ny)), float(square), float(marker), d)


def board_spec(board):
    """What board.json and the calibration results say of a board."""
    nx, ny = board.getChessboardSize()
    return {"squares": [int(nx), int(ny)], "square_mm": round(float(board.getSquareLength()), 6),
            "marker_mm": round(float(board.getMarkerLength()), 6), "dictionary": BOARD["dictionary"]}


def folder_board(folder):
    """The board of the photos of a folder, from its board.json: (square, squares, marker), or None."""
    p = Path(folder) / BOARD_FILE
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("dictionary", BOARD["dictionary"]) != BOARD["dictionary"]:
        sys.exit(f"{p}: dictionary {d['dictionary']}, the calibration board uses {BOARD['dictionary']}")
    return float(d["square_mm"]), tuple(int(n) for n in d["squares"]), float(d["marker_mm"])


def texture_ppm(board, ppm=TEXTURE_PPM):
    """A texture resolution near ppm (px per mm) at which a square is a whole number of pixels: its corners then fall
    exactly between texture pixels, as they do for the A4 board (24 mm at 12 px per mm)."""
    sq = board.getSquareLength()
    return round(sq * ppm) / sq


def page_image(board, ppm):
    """A4 page with the board centred (a larger page with a margin when the board does not fit on A4), `ppm` px per
    mm, and the board origin on the page in mm."""
    nx, ny = board.getChessboardSize()
    sq = board.getSquareLength()
    bw, bh = round(nx * sq * ppm), round(ny * sq * ppm)
    page = PAGE
    if nx * sq > PAGE[0] or ny * sq > PAGE[1]:
        page = (max(PAGE[0], nx * sq + 2 * PAGE_MARGIN), max(PAGE[1], ny * sq + 2 * PAGE_MARGIN))
    pw, ph = round(page[0] * ppm), round(page[1] * ppm)
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
            f"{BOARD['dictionary']}. Print at 100% (actual size). On a screen, measure the length between the arrows "
            f"and type it in ImpactGuide.")
    ax.text(ox, oy + ny * square + 2.5, note, fontsize=6, va="top", color="0.4")
    # what to measure: a dimension line over the long side, from the outer edge of the first square
    # to the outer edge of the last one, with the length it has when printed at 100%
    width = nx * square
    y = max(oy - 4.0, 2.0)
    for x in (ox, ox + width):
        ax.plot([x, x], [oy - 0.8, max(oy - 7.0, 0.5)], color="k", lw=0.4)
    ax.annotate("", xy=(ox, y), xytext=(ox + width, y),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="k", shrinkA=0, shrinkB=0))
    ax.text(ox + width / 2, y, f"measure this length: {width:g} mm at 100% ({nx} squares)",
            fontsize=6.5, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1.2))
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


def render(page, origin, rays, R, t, rng, ppm=TEXTURE_PPM):
    h, w = rays.shape[:2]
    size = (w, h)
    mx, my = board_maps(rays, R, t, origin, ppm)
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


def synth(out, n, seed, K=SYNTH_K, dist=SYNTH_DIST, size=SYNTH_SIZE, prefix="synth", board=None):
    """n photos of the board (the A4 one unless given) taken with the camera K, dist; truth.json beside them."""
    rng = np.random.default_rng(seed)
    board = board or make_board()
    ppm = texture_ppm(board)
    page, origin = page_image(board, ppm)
    page = cv2.GaussianBlur(page, (0, 0), 0.6)  # anti-aliasing of the texture
    rays = pixel_rays(K, dist, size)
    out.mkdir(parents=True, exist_ok=True)
    views = []
    for i in range(n):
        R, t = sample_pose(rng, board, K, dist, size)
        img, blur, noise = render(page, origin, rays, R, t, rng, ppm)
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
    rms, K, dist, rvecs, tvecs, std, _, per_view = cv2.calibrateCameraExtended(obj, img, size, None, None,
                                                                              criteria=crit)
    return (rms, K, dist.ravel(), std.ravel(), dict(zip(views, per_view.ravel())),
            dict(zip(views, zip(rvecs, tvecs))))


def view_rms(view, K, dist, pose=None):
    """(RMS in px, residuals N x 2) of one view under a camera model, at its pose (rvec, tvec) or at the one PnP finds."""
    obj, img = view
    if pose is None:
        _, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
        pose = (rvec, tvec)
    proj = cv2.projectPoints(obj, pose[0], pose[1], K, dist)[0].reshape(-1, 2)
    r = proj - img.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(r ** 2, 1)))), r


def corner_residuals(views, poses, K, dist, square):
    """Where on the board the model misses: for each inner corner (i, j) of the chessboard, the RMS of its
    reprojection error over the views that see it, how many do, and its mean error vector (px)."""
    acc = {}
    for name, view in views.items():
        _, r = view_rms(view, K, dist, poses.get(name))
        ij = np.rint(view[0].reshape(-1, 3)[:, :2] / square).astype(int) - 1
        for (i, j), e in zip(ij, r):
            acc.setdefault((int(i), int(j)), []).append(e)
    rows, mags = [], []
    for (i, j), es in sorted(acc.items()):
        es = np.asarray(es)
        m = float(np.sqrt(np.mean(np.sum(es ** 2, 1))))
        mags.append(m)
        mx, my = es.mean(0)
        rows.append([i, j, round(m, 4), len(es), round(float(mx), 4), round(float(my), 4)])
    return {"corners": rows, "max_px": round(float(max(mags)), 4) if mags else 0.0,
            "p95_px": round(float(np.percentile(mags, 95)), 4) if mags else 0.0}


def calibrate(views, size, exclude=(), square=None):
    """Calibrate on the views but the excluded ones (chosen by hand), drop the views with an error above 3 times the
    median, calibrate again. With square (mm), where on the board the error is (corner_residuals). Every view, used
    or not, gets its error under the final model (per_view_all_px: PnP for those left out)."""
    exclude = sorted(set(exclude) & set(views))
    use = {k: v for k, v in views.items() if k not in exclude}
    if len(use) < MIN_VIEWS:
        sys.exit(f"only {len(use)} usable photos, at least {MIN_VIEWS} needed")
    rms, K, dist, std, per_view, poses = solve(use, size)
    limit = max(3 * float(np.median(list(per_view.values()))), 0.3)
    dropped = sorted(k for k, e in per_view.items() if e > limit)
    if dropped and len(use) - len(dropped) >= MIN_VIEWS:
        use = {k: v for k, v in use.items() if k not in dropped}
        rms, K, dist, std, per_view, poses = solve(use, size)
    else:
        dropped = []
    every = {k: (float(per_view[k]) if k in per_view else view_rms(v, K, dist)[0]) for k, v in views.items()}
    out = {
        "image_size": list(size), "K": K.tolist(), "dist": dist.tolist(),
        "rms_px": round(float(rms), 4), "n_views": len(use),
        "n_corners": int(sum(len(v[0]) for v in use.values())),
        "std": {k: round(float(s), 4) for k, s in zip(["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"], std)},
        "per_view_rms_px": {k: round(float(e), 4) for k, e in per_view.items()},
        "per_view_all_px": {k: round(e, 4) for k, e in sorted(every.items())},
        "dropped": dropped, "excluded": exclude, "limit_px": round(limit, 4),
    }
    if square:
        out["residuals"] = corner_residuals(use, poses, K, dist, float(square))
    return out


def plain(cal):
    """The intrinsics in plain units: focal length (px, and the field of view in degrees across and down), where the
    optical centre is from the middle of the image (px, + right and down), how far the lens bends at the corners of
    the image (px between where a point is and where a lens without distortion would put it), with their standard
    deviations."""
    K, d = np.array(cal["K"]), np.array(cal["dist"])
    w, h = cal["image_size"]
    s = cal.get("std", {})
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-12)
    ideal = cv2.undistortPoints(corners, K, d, None, None, K, crit).reshape(-1, 2)
    bend = float(np.max(np.linalg.norm(ideal - corners.reshape(-1, 2), axis=1)))
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "f_std": float(np.hypot(s.get("fx", 0), s.get("fy", 0)) / 2),
            "fov_x": float(np.degrees(2 * np.arctan(w / 2 / K[0, 0]))),
            "fov_y": float(np.degrees(2 * np.arctan(h / 2 / K[1, 1]))),
            "cx_off": float(K[0, 2] - (w - 1) / 2), "cy_off": float(K[1, 2] - (h - 1) / 2),
            "c_std": float(max(s.get("cx", 0), s.get("cy", 0))), "bend_px": bend, "size": [int(w), int(h)]}


# ── the check of each photo as it comes ──────────────────────────────────────
BLUR_PX = 4.0           # a photo is blurred past this sigma (px at 4080 px across, scaled with the width, at least
BLUR_MIN_PX = 2.5       # this): the corners then move and the calibration suffers (T5.1: 3.5 px still calibrates)
BLUR_FLOOR = 0.8        # what the estimate gives for an edge with no blur at all (the derivative's own width)


def blur_px(gray, obj, imgp, square):
    """The blur of a photo of the board, as the sigma (px) of a Gaussian: at each corner found, across the board's two
    sets of edges, the contrast over the steepest slope (an edge blurred by sigma rises by contrast / (sigma sqrt(2
    pi)) per px at most), the larger of the two (a shake blurs one way), the median over the corners."""
    obj2 = obj.reshape(-1, 3)[:, :2].astype(np.float64)
    pts = imgp.reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return float("nan")
    H, _ = cv2.findHomography(obj2, pts)
    if H is None:
        return float("nan")
    out = []
    g = gray.astype(np.float32)
    hh, ww = g.shape
    for (X, Y), (u, v) in zip(obj2, pts):
        p = H @ [X, Y, 1.0]
        ex = H @ [X + square, Y, 1.0]
        ey = H @ [X, Y + square, 1.0]
        jx = ex[:2] / ex[2] - p[:2] / p[2]          # one square along the board's x, in the image
        jy = ey[:2] / ey[2] - p[:2] / p[2]
        side = min(np.linalg.norm(jx), np.linalg.norm(jy))
        r = int(np.clip(0.3 * side, 3, 25))
        x0, y0 = int(round(u)) - r, int(round(v)) - r
        if x0 < 1 or y0 < 1 or x0 + 2 * r + 2 > ww or y0 + 2 * r + 2 > hh:
            continue
        patch = g[y0 - 1:y0 + 2 * r + 2, x0 - 1:x0 + 2 * r + 2]
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1] / 8
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1] / 8
        inner = patch[1:-1, 1:-1]
        c = float(np.percentile(inner, 97) - np.percentile(inner, 3))
        if c < 20:
            continue
        sig = []
        for j in (jx, jy):
            n = np.array([-j[1], j[0]]) / max(np.linalg.norm(j), 1e-9)     # across the edges that run along j
            slope = float(np.percentile(np.abs(gx * n[0] + gy * n[1]), 99))
            sig.append(c / (np.sqrt(2 * np.pi) * max(slope, 1e-6)))
        out.append(max(sig))
    if not out:
        return float("nan")
    est = float(np.median(out))
    return float(np.sqrt(max(0.0, est ** 2 - BLUR_FLOOR ** 2)))


def blur_limit(width):
    return max(BLUR_MIN_PX, BLUR_PX * width / SYNTH_SIZE[0])


def check_photo(path, board, size=None):
    """The check of one photo as it arrives: can it go into the calibration? {"ok", "why" (none: no board found,
    few: fewer than MIN_CORNERS corners, size: another size than the photos before it (the phone held another way),
    blur: blurred past blur_limit, unreadable), "corners", "of", "blur_px", "size", "points" (where the corners are,
    fractions of the width and height, at most 120)}."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    nx, ny = board.getChessboardSize()
    out = {"ok": False, "why": "", "corners": 0, "of": (nx - 1) * (ny - 1), "blur_px": None, "size": None,
           "points": []}
    if img is None:
        out["why"] = "unreadable"
        return out
    w, h = img.shape[::-1]
    out["size"] = [w, h]
    if size and list(size) != [w, h]:
        out["why"] = "size"
        return out
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(img)
    n = 0 if ids is None else len(ids)
    out["corners"] = n
    if n:
        pts = corners.reshape(-1, 2)
        k = np.linspace(0, n - 1, min(n, 120)).astype(int)
        out["points"] = [[round(float(x / w), 4), round(float(y / h), 4)] for x, y in pts[k]]
    if n == 0:
        out["why"] = "none"
        return out
    collinear = board.checkCharucoCornersCollinear(ids)
    b = float("nan")
    if n >= 4 and not collinear:        # blurred is said before few: a blur also loses corners
        obj, imgp = board.matchImagePoints(corners, ids)
        b = blur_px(img, obj, imgp, board.getSquareLength())
        out["blur_px"] = None if b != b else round(b, 2)
    if b == b and b > blur_limit(w):
        out["why"] = "blur"
        return out
    if n < MIN_CORNERS or collinear:
        out["why"] = "few"
        return out
    out["ok"] = True
    return out


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
    for name in cal.get("excluded", []):
        print(f"left out {name}: chosen by hand")


def print_compare(c):
    print(f"error fx {100 * c['fx_rel']:+.3f} %  fy {100 * c['fy_rel']:+.3f} %  "
          f"cx {c['cx_px']:+.2f} px  cy {c['cy_px']:+.2f} px  "
          f"model max {c['model_max_px']:.3f} px, RMS {c['model_rms_px']:.3f} px")


def run_calibrate(images, out, truth, square, squares=None, marker=None, exclude=()):
    board = make_board(square, squares, marker)
    views, skipped, size = detect(find_images(images), board)
    cal = calibrate(views, size, exclude, board.getSquareLength())
    cal = {"images": str(Path(images)), "date": date.today().isoformat(), "board": board_spec(board),
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


def run_check(images, intrinsics, square, squares=None, marker=None):
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
    views, skipped, size = detect(find_images(images), make_board(square, squares, marker))
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
    a = sub.add_parser("check")
    a.add_argument("images", type=Path)
    a.add_argument("--intrinsics", type=Path, required=True)
    for a in (sub.choices["calibrate"], sub.choices["check"]):
        a.add_argument("--square", type=float, help="measured square size, mm (default: board.json of the folder, "
                                                    "else 24)")
        a.add_argument("--squares", type=lambda v: tuple(int(n) for n in v.lower().split("x")),
                       help="squares across and down, e.g. 13x7 (default: board.json of the folder, else 11x8)")
        a.add_argument("--marker", type=float, help="marker size, mm (default: board.json, else 3/4 of the square)")
    a = sub.add_parser("test")
    a.add_argument("--work", type=Path, default=WORK)
    args = ap.parse_args()

    if args.cmd == "board":
        write_board(args.out, args.square)
    elif args.cmd == "synth":
        synth(args.out, args.n, args.seed)
    elif args.cmd in ("calibrate", "check"):
        square, squares, marker = folder_board(args.images) or (BOARD["square"], None, None)
        if args.square is not None or args.squares is not None:     # set by hand: board.json is not mixed in
            square, squares, marker = args.square or BOARD["square"], args.squares, None
        if args.marker is not None:
            marker = args.marker
        if args.cmd == "calibrate":
            run_calibrate(args.images, args.out, args.truth, square, squares, marker)
        else:
            sys.exit(0 if run_check(args.images, args.intrinsics, square, squares, marker)[0] else 1)
    elif args.cmd == "test":
        sys.exit(0 if run_test(args.work) else 1)


if __name__ == "__main__":
    main()
