"""ChArUco calibration of the phone camera intrinsics.

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

Library functions raise CalibrationError (a code and its values) for what the
operator can act on, never SystemExit; they report through a `log` callable
(print when none is given); the calibration results are written in one step
(a temporary file put in place) and never hold NaN.

Besides the intrinsics, a calibration says what spoils them, as codes in
"issues" (ISSUES): a board that is not flat (fit_shape fits it as a curved
surface and keeps the intrinsics of that model), a phone that moved inside
itself between the photos (the optical centre free for each photo, a diagnosis
only), corners of the image the board never reached (coverage), photos taken
with the camera's automatic settings (photo_settings, from the EXIF). The
corners are refined on the photo blurred a little (refine_corners).

Usage:
    python calibrate_charuco.py board [--out DIR]
        printable page: PDF at 1:1 scale and PNG
    python calibrate_charuco.py synth [--out DIR] [--n 40] [--seed 0]
        synthetic photos of the board taken with known intrinsics (truth.json)
    python calibrate_charuco.py calibrate IMAGES [--out FILE] [--truth FILE] [--square MM --squares NXxNY --marker MM]
        intrinsics from a folder of photos, compared with the truth if given
    python calibrate_charuco.py check IMAGES --intrinsics FILE [--square MM --squares NXxNY --marker MM]
        start-of-session check: are the stored intrinsics still valid?
    python calibrate_charuco.py test [--work DIR] [--real PHOTOS]
        synth + calibrate + check on synthetic photos, then a curved board, a phone held in the hand, the corners
        of the image, the camera's settings; exit code 1 on failure. DIR must be empty, missing, or a folder of an
        earlier test (it is emptied). PHOTOS: real photos of a curved board taken by hand in the automatic mode

Real photos go in 03_data/raw/calibration/<yyyymmdd>_<mode>/, intrinsics in
03_data/processed/calibration/intrinsics_<mode>.json. Synthetic photos and the
printable board go in the work folder (IMPACTGUIDE_WORK/calibration).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import cv2
import numpy as np

import arrowhead

WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact")) / "calibration"
BOARD = {"squares": (11, 8), "square": 24.0, "marker": 18.0, "dictionary": "DICT_5X5_100"}
PAGE = (297.0, 210.0)  # A4 landscape, mm
PAGE_MARGIN = 10.0  # mm around a board larger than the page (synthetic photos of a board on a monitor)
BOARD_FILE = "board.json"  # the board of the photos of a folder: squares, square_mm, marker_mm, dictionary
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MIN_CORNERS = 12  # per view
MIN_VIEWS = 8
TEST_MARKER = ".impactguide-test"   # a folder made by a test, which a later test may empty

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
SYNTH_MIN_SEEN = 24  # corners a synthetic pose keeps inside the image (fewer when the board has fewer)
SYNTH_EDGE_PX = 20  # ... at least this far from the edge
SYNTH_TRIES = 10_000  # poses drawn at most before giving up
PRINT_PPM = 24  # px per mm of the printable page (about 610 dpi; the PDF places the image at its size in mm)

# Acceptance on synthetic photos (test). The RMS of the protocol; the others
# bound the error on the recovered camera model.
MAX_RMS = 0.5  # px
MAX_FOCAL_ERROR = 0.002  # relative
MAX_CENTRE_ERROR = 4.0  # px
MAX_MODEL_DIFFERENCE = 1.0  # px, anywhere in the image
# Start-of-session check: changes below these floors are accepted even when they
# exceed 3 standard deviations. Refocusing from 28 to 35 cm changes f by about 0.4 %.
CHECK_FOCAL = 0.001  # relative
CHECK_CENTRE = 2.0  # px
OUTLIER_FACTOR = 3.0  # a view whose error is above this many times the median is dropped...
OUTLIER_FLOOR_PX = 0.3  # ... and above this
OUTLIER_MAX_SHARE = 0.25  # at most this share of the views is dropped (more is a bad series, said by the verdict)
# The board's shape (fit_shape): a quadratic surface, z = a u^2 + b u v + c v^2 with u, v from the board's middle in
# SHAPE_UNIT mm, fitted with the camera and the poses after the flat solve.
SHAPE_UNIT = 100.0      # mm
FLAT_SAG_MM = 0.2       # a board bending more than this (peak to valley of the surface off its plane) is not flat...
FLAT_RMS_DROP = 0.2     # ... nor one whose error falls by more than this share when its shape is fitted
# The shape is used (its intrinsics kept) as soon as it explains the photos better than the noise could: the drop of
# the squared error with its 3 numbers, against the noise left (an F test), past SHAPE_F. A warp of 0.1 mm, flat
# enough to pass, moves the focal length of the flat model by about 0.15 %.
SHAPE_F = 10.0
# The phone kept still (fit_shape, diagnosis only): the optical centre free for each photo, mean zero. Held in the
# hand, the lens stabiliser (or the rolling shutter) moves it by tens of px between photos.
STILL_CENTRE_PX = 5.0   # the centre's spread past this, and past what the noise explains: the phone moved
STILL_CHI2_Z = 3.09     # ... past the 99.9 % quantile of chi-square
# The corners refined on the photo blurred a little (refine_corners), at REFINE_REF px on the long side (scaled with it)
REFINE_REF = 4096
REFINE_SIGMA = 1.5      # px, the blur
REFINE_WIN = 13         # px, the half side of the search window at most...
REFINE_SHARE = 0.12     # ... and at most this share of the square's side there (a marker's edge is 1/8 of it away)
REFINE_EPS = 0.001      # px, when to stop
REFINE_MIN_ANGLE = 60.0  # degrees: a corner whose squares are seen with sides closer than this is not refined
# Where the board reached in the image (coverage): a grid of COVER_GRID x COVER_GRID cells, and the four corners of
# the image, beyond COVER_CORNER of the half diagonal from the middle. A ChArUco corner is found only with the markers
# about it whole in the photo, so the nearest a board's corners come to a corner of the image is about a square and a
# half inside it: at 30-40 cm about 0.84 of the way for the boards of the app (calib_guide's poses for the corners).
COVER_GRID = 3
COVER_CORNER = 0.80
CELL_NAMES = (("top_left", "top", "top_right"), ("left", "centre", "right"), ("bottom_left", "bottom", "bottom_right"))
CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")

# The codes of CalibrationError, for the interface (with the values in params).
CODES = ("board_unsupported", "board_unreadable", "no_photos", "few_photos", "photo_size_mixed",
         "intrinsics_bad", "intrinsics_size", "pose_not_found")
# What spoils a calibration, in its "issues" (the interface names them):
#   board_not_flat   the board is curved (a curved monitor, a sheet not glued flat): shape.sag_mm, shape.focal_shift
#   phone_not_still  the phone moved between the photos (held in the hand): centre.spread_px
#   corners_missing  corners of the image the board never reached: coverage.missing_corners (the model is least sure
#                    there; advice, a calibration can still be good)
#   camera_settings  photos taken with automatic settings: settings {photo: [SETTINGS codes]}
ISSUES = ("board_not_flat", "phone_not_still", "corners_missing", "camera_settings")
ISSUES_NOT_GOOD = ("board_not_flat", "phone_not_still", "camera_settings")    # never a good calibration with these
# What photo_settings and settings_problems find wrong in the camera's settings, from the EXIF (the photo is refused
# with the reason "settings" by check_photo): exposure or white balance automatic, the AI scene or HDR on, a zoom, and
# against the first photo of the series another ISO, exposure time, focal length (another lens) or resolution.
SETTINGS = ("exposure_auto", "white_balance_auto", "ai_scene", "hdr", "zoom", "iso_changed", "exposure_changed",
            "focal_changed", "size_changed")

Log = Callable[[str], None]


class CalibrationError(RuntimeError):
    """A failure the interface can name.

    Attributes:
        code: one of CODES.
        cause: the same as code.
        params: the values a sentence about it needs.
    """

    def __init__(self, code: str, message: str = "", **params: object) -> None:
        super().__init__(message or code)
        self.code = self.cause = code
        self.params = params


class BoardError(CalibrationError):
    """The board of a photo folder cannot be used: another kind of board, or a board.json that cannot be read."""


def _say(log: Log | None) -> Log:
    return log if log is not None else print


def write_text_atomic(path: Path | str, text: str) -> None:
    """Writes `text` (UTF-8) in one step: a temporary file in the same folder, flushed to the disk, then put in place
    of the old one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def finite(value):
    """The value with every NaN or infinity (inside lists and dicts too) made None, for strict JSON."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def write_json_atomic(path: Path | str, doc: object) -> None:
    """JSON written in one step, NaN and infinity as null (strict JSON a browser reads)."""
    write_text_atomic(path, json.dumps(finite(doc), indent=1, allow_nan=False))


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def test_folder(work: Path | str) -> Path:
    """The folder a test writes into, emptied: it must be missing, empty, or made by an earlier test (it holds
    TEST_MARKER). Raises ValueError for any other folder, which is never touched."""
    work = Path(work)
    if work.exists():
        if not work.is_dir():
            raise ValueError(f"{work} is not a folder")
        if any(work.iterdir()):
            if not (work / TEST_MARKER).is_file():
                raise ValueError(f"{work} is not empty and was not made by a test: give an empty or new folder")
            shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / TEST_MARKER).write_text("made by a self-test; a later test may empty this folder\n", encoding="utf-8")
    return work


def make_board(square: float = BOARD["square"], squares: Sequence[int] | None = None,
               marker: float | None = None) -> cv2.aruco.CharucoBoard:
    """The calibration board: the A4 one (11 x 8, marker 3/4 of the square) unless squares or marker are given."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD["dictionary"]))
    if marker is None:
        marker = square * BOARD["marker"] / BOARD["square"]
    nx, ny = squares or BOARD["squares"]
    return cv2.aruco.CharucoBoard((int(nx), int(ny)), float(square), float(marker), d)


def board_spec(board: cv2.aruco.CharucoBoard) -> dict:
    """What board.json and the calibration results say of a board."""
    nx, ny = board.getChessboardSize()
    return {"squares": [int(nx), int(ny)], "square_mm": round(float(board.getSquareLength()), 6),
            "marker_mm": round(float(board.getMarkerLength()), 6), "dictionary": BOARD["dictionary"]}


def folder_board(folder: Path | str) -> tuple[float, tuple[int, int], float] | None:
    """The board of the photos of a folder, from its board.json: (square, squares, marker), or None.

    Raises:
        BoardError: board_unsupported (another dictionary), board_unreadable (not a board.json this reads).
    """
    p = Path(folder) / BOARD_FILE
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        dictionary = d.get("dictionary", BOARD["dictionary"])
        board = float(d["square_mm"]), tuple(int(n) for n in d["squares"]), float(d["marker_mm"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise BoardError("board_unreadable", f"{p}: {e}", file=BOARD_FILE) from e
    if dictionary != BOARD["dictionary"]:
        raise BoardError("board_unsupported", f"{p}: dictionary {dictionary}, the calibration board uses "
                                              f"{BOARD['dictionary']}", dictionary=str(dictionary))
    square, squares, marker = board
    if len(squares) != 2 or min(squares) < 2 or not (0 < marker < square) or not math.isfinite(square):
        raise BoardError("board_unreadable", f"{p}: not a board", file=BOARD_FILE)
    return board


def texture_ppm(board: cv2.aruco.CharucoBoard, ppm: float = TEXTURE_PPM) -> float:
    """A texture resolution near ppm (px per mm) at which a square is a whole number of pixels: its corners then fall
    exactly between texture pixels, as they do for the A4 board (24 mm at 12 px per mm)."""
    sq = board.getSquareLength()
    return round(sq * ppm) / sq


def page_image(board: cv2.aruco.CharucoBoard, ppm: float) -> tuple[np.ndarray, tuple[float, float]]:
    """A4 page with the board centred (a larger page with a margin when the board does not fit on A4), `ppm` px per
    mm, and the board origin on the page in mm. The page's size in mm is the image's size / ppm."""
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


def write_board(out: Path | str, square: float, log: Log | None = None) -> Path:
    """The printable page of the board at `square` mm: PDF at 1:1 scale and PNG, in `out`. The PDF's page is A4 when
    the board fits on it, else larger (its name then says its size). Returns the PDF's path."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out = Path(out)
    board = make_board(square)
    nx, ny = board.getChessboardSize()
    page, (ox, oy) = page_image(board, PRINT_PPM)
    page_mm = (page.shape[1] / PRINT_PPM, page.shape[0] / PRINT_PPM)
    fmt = "a4" if np.allclose(page_mm, PAGE, atol=0.05) else f"{page_mm[0]:.0f}x{page_mm[1]:.0f}mm"
    stem = f"charuco_{nx}x{ny}_{square:g}mm_{BOARD['dictionary'].lower()}_{fmt}"
    ok, png = cv2.imencode(".png", page)
    if not ok:
        raise OSError("the board image could not be encoded")
    _write_bytes_atomic(out / f"{stem}.png", png.tobytes())
    inch = 1 / 25.4
    fig = Figure(figsize=(page_mm[0] * inch, page_mm[1] * inch))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(page, cmap="gray", vmin=0, vmax=255, interpolation="none", extent=(0, page_mm[0], page_mm[1], 0))
    ax.set_xlim(0, page_mm[0])
    ax.set_ylim(page_mm[1], 0)
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
    # the line between the two filled arrowheads of the screen board (arrowhead.py, at their size in mm), running
    # into each one and stopping inside it
    stop = arrowhead.inset(arrowhead.LENGTH_MM, 0.6 * 25.4 / 72)
    ax.plot([ox + stop, ox + width - stop], [y, y], color="k", lw=0.6, solid_capstyle="butt")
    for tip, d in ((ox, -1), (ox + width, 1)):
        ax.add_patch(arrowhead.patch((tip, y), (d, 0), arrowhead.LENGTH_MM, facecolor="k"))
    ax.text(ox + width / 2, y, f"measure this length: {width:g} mm at 100% ({nx} squares)",
            fontsize=6.5, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1.2))
    pdf = out / f"{stem}.pdf"
    fd, tmp = tempfile.mkstemp(prefix=f".{pdf.name}.", suffix=".tmp", dir=out)
    os.close(fd)
    try:
        fig.savefig(tmp, format="pdf", metadata={"CreationDate": None})
        os.replace(tmp, pdf)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _say(log)(f"board: {out / stem}.pdf, .png")
    return pdf


def rotation(axis_angle) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(axis_angle, float).reshape(3, 1))[0]


def normalised_bounds(K, dist, size):
    """Largest |x| and |y| of the normalised image plane seen inside the image."""
    w, h = size
    edge = [(u, v) for u in np.linspace(0, w - 1, 33) for v in (0, h - 1)]
    edge += [(u, v) for v in np.linspace(0, h - 1, 25) for u in (0, w - 1)]
    xn = cv2.undistortPoints(np.array(edge, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    return np.abs(xn).max(axis=0)


def sample_pose(rng, board, K, dist, size, tries: int = SYNTH_TRIES):
    """Random board pose with at least SYNTH_MIN_SEEN chessboard corners inside the image (all of them for a board
    with fewer). Raises CalibrationError pose_not_found after `tries` draws."""
    nx, ny = board.getChessboardSize()
    sq = board.getSquareLength()
    centre = np.array([nx * sq / 2, ny * sq / 2, 0.0])
    corners = board.getChessboardCorners()
    need = min(SYNTH_MIN_SEEN, len(corners))
    xmax, ymax = normalised_bounds(K, dist, size)
    w, h = size
    e = SYNTH_EDGE_PX
    for _ in range(tries):
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
        seen &= (uv[:, 0] > e) & (uv[:, 0] < w - e - 1) & (uv[:, 1] > e) & (uv[:, 1] < h - e - 1)
        if seen.sum() >= need:
            return R, t
    raise CalibrationError("pose_not_found", f"no pose keeps {need} corners in view after {tries} tries")


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


def bent_z(x, bend):
    """The board's depth (mm, + into the board) at x (mm) when it is bent as a cylinder about an axis along its y:
    bend (radius mm, x of the middle mm), its edges toward the camera as on a curved monitor; 0 when bend is None."""
    if not bend:
        return np.zeros_like(x)
    radius, middle = bend
    return -(x - middle) ** 2 / (2 * radius)


def board_maps(rays, R, t, origin, ppm, bend=None):
    """remap() maps from the image pixels to the page texture; bend: the board bent as bent_z says (the texture
    keeps its x and y, a height field), else flat."""
    if bend:
        # the ray s [x, y, 1] in board coordinates: o + s d; met where its z is the surface's at its x
        Rt = R.T.astype(np.float64)
        ox, oy, oz = (float(v) for v in -Rt @ np.asarray(t, np.float64).ravel())
        bend = (float(bend[0]), float(bend[1]))
        x, y = rays[..., 0], rays[..., 1]
        dx, dy, dz = ((float(Rt[k, 0]) * x + float(Rt[k, 1]) * y + float(Rt[k, 2])).astype(np.float32)
                      for k in range(3))
        front = dz > 1e-9
        dz = np.where(front, dz, np.float32(1))
        s = -oz / dz
        for _ in range(5):
            s = (bent_z(ox + s * dx, bend) - oz) / dz
        front &= s > 0
        mx = np.where(front, (ox + s * dx + float(origin[0])) * float(ppm) - 0.5, -1e4)
        my = np.where(front, (oy + s * dy + float(origin[1])) * float(ppm) - 0.5, -1e4)
        return mx.astype(np.float32), my.astype(np.float32)
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


def render(page, origin, rays, R, t, rng, ppm=TEXTURE_PPM, blur: tuple[float, float] | None = None, bend=None,
           shift=None):
    """A photo of the page at pose (R, t): (image, blur sigma px, noise sigma). blur: the (lo, hi) range the
    defocus is drawn from (SYNTH_BLUR when None); bend: the board bent (bent_z); shift: (dx, dy) px the optical
    centre moved for this photo (the image moves with it)."""
    h, w = rays.shape[:2]
    size = (w, h)
    mx, my = board_maps(rays, R, t, origin, ppm, bend)
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
    if shift is not None and np.any(shift):
        img = cv2.warpAffine(img, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), size, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    sigma = rng.uniform(*(blur if blur is not None else SYNTH_BLUR))
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    noise = rng.uniform(*SYNTH_NOISE)
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), sigma, noise


def synth(out, n, seed, K=SYNTH_K, dist=SYNTH_DIST, size=SYNTH_SIZE, prefix="synth", board=None,
          log: Log | None = None, bend_radius=None, centre_sd=0.0):
    """n photos of the board (the A4 one unless given) taken with the camera K, dist; truth.json beside them.
    bend_radius (mm): the board bent as a curved monitor about its vertical middle line (bent_z); centre_sd (px): the
    optical centre moved for each photo by this much (normal, each way), as a phone held in the hand."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    board = board or make_board()
    ppm = texture_ppm(board)
    page, origin = page_image(board, ppm)
    page = cv2.GaussianBlur(page, (0, 0), 0.6)  # anti-aliasing of the texture
    rays = pixel_rays(K, dist, size)
    nx = board.getChessboardSize()[0]
    bend = (float(bend_radius), nx * board.getSquareLength() / 2) if bend_radius else None
    out.mkdir(parents=True, exist_ok=True)
    views = []
    for i in range(n):
        R, t = sample_pose(rng, board, K, dist, size)
        shift = rng.normal(0, centre_sd, 2) if centre_sd else np.zeros(2)
        img, blur, noise = render(page, origin, rays, R, t, rng, ppm, bend=bend, shift=shift)
        name = f"{prefix}_{i:03d}.jpg"
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        views.append({"image": name, "rvec": cv2.Rodrigues(R)[0].ravel().tolist(), "tvec": t.tolist(),
                      "blur_px": round(blur, 3), "noise_grey": round(noise, 3),
                      "centre_shift_px": [round(float(v), 3) for v in shift]})
    truth = {"image_size": list(size), "K": np.asarray(K).tolist(), "dist": np.asarray(dist).tolist(), "seed": seed,
             "bend": list(bend) if bend else None, "views": views}
    write_json_atomic(out / "truth.json", truth)
    _say(log)(f"synth: {n} photos in {out}")


def find_images(folder):
    """The photos of a folder, sorted. Raises CalibrationError no_photos when there are none."""
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if folder.is_dir() else []
    if not paths:
        raise CalibrationError("no_photos", f"no photos in {folder}", folder=folder.name)
    return paths


def square_axes(obj, imgp, square):
    """The board's two sides of a square at each corner found, as vectors in the image (px): (along x, along y), N x 2
    each, through the homography of the corners; None when it cannot be found."""
    obj2 = np.asarray(obj, np.float64).reshape(-1, 3)[:, :2]
    pts = np.asarray(imgp, np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return None
    H, _ = cv2.findHomography(obj2, pts)
    if H is None:
        return None
    one = np.ones((len(obj2), 1))
    p, ex, ey = (np.column_stack([obj2 + off, one]) @ H.T for off in ((0, 0), (square, 0), (0, square)))
    p, ex, ey = (q[:, :2] / q[:, 2:] for q in (p, ex, ey))
    return ex - p, ey - p


def refine_corners(gray, obj, imgp, square):
    """The corners the detector found, refined with cornerSubPix on the photo blurred by REFINE_SIGMA px (at
    REFINE_REF px on its long side, scaled with it), which evens out a screen's pixels and the sensor's noise. The
    window's half side is REFINE_WIN px at most (scaled) and REFINE_SHARE of the square's side at that corner (of the
    nearest corner of the four squares around it, along the image's rows or columns, when the square is seen
    slanted), so it never reaches a marker's edge; the search stops at REFINE_EPS px. A corner whose squares are seen
    with sides closer than REFINE_MIN_ANGLE (a grazing view: the blurred corner is then biased), whose refinement
    fails or wanders farther than half its window, keeps the detector's place. Same shape and type as imgp."""
    pts = np.asarray(imgp, np.float32).reshape(-1, 2)
    axes = square_axes(obj, pts, square)
    if axes is None:
        return imgp
    jx, jy = axes
    reach = np.minimum(np.abs(jx + jy).max(1), np.abs(jx - jy).max(1))      # a square's side, as far as rows go
    cos = np.abs(np.sum(jx * jy, 1)) / np.maximum(np.linalg.norm(jx, axis=1) * np.linalg.norm(jy, axis=1), 1e-9)
    scale = max(gray.shape[:2]) / REFINE_REF
    g = gray.astype(np.float32)
    if REFINE_SIGMA * scale >= 0.3:
        g = cv2.GaussianBlur(g, (0, 0), REFINE_SIGMA * scale)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, REFINE_EPS)
    out = pts.copy()
    for j, (p, side) in enumerate(zip(pts, reach)):
        if not math.isfinite(side) or cos[j] > math.cos(math.radians(REFINE_MIN_ANGLE)):
            continue
        win = int(min(REFINE_WIN * scale, REFINE_SHARE * side))
        if win < 2:
            continue
        q = p.reshape(1, 1, 2).copy()
        try:
            cv2.cornerSubPix(g, q, (win, win), (-1, -1), crit)
        except cv2.error:
            continue
        q = q.reshape(2)
        if np.all(np.isfinite(q)) and np.linalg.norm(q - p) < 0.5 * win:
            out[j] = q
    return out.reshape(np.shape(imgp)).astype(np.asarray(imgp).dtype)


def detect(paths, board, prefer=None, refine=True):
    """Chessboard corners of every usable photo: ({name: (object points, image points)}, [(name, why skipped)],
    image size). One camera mode per folder: the photos of the size most photos have are used (of the size `prefer`,
    (w, h), when some usable photo has it: the stored calibration's, for the session check), any other size is
    skipped with the reason "size"; size is None when no photo can be read. refine: the corners refined
    (refine_corners), photos and the frames of a film alike (calib_video refines its frames the same way, so that a
    folder of frames calibrates as its film did)."""
    detector = cv2.aruco.CharucoDetector(board)
    found, skipped, sizes = {}, [], {}
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)  # applies the EXIF orientation
        if img is None:
            skipped.append((p.name, "unreadable"))
            continue
        size = img.shape[::-1]
        sizes[p.name] = size
        corners, ids, _, _ = detector.detectBoard(img)
        if ids is None or len(ids) < MIN_CORNERS:
            skipped.append((p.name, f"{0 if ids is None else len(ids)} corners"))
            continue
        if board.checkCharucoCornersCollinear(ids):
            skipped.append((p.name, "collinear corners"))
            continue
        obj, imgp = board.matchImagePoints(corners, ids)
        imgp = imgp.astype(np.float32)
        if refine:
            imgp = refine_corners(img, obj, imgp, board.getSquareLength())
        found[p.name] = (obj.astype(np.float32), imgp)
    if not sizes:
        return {}, skipped, None
    count = Counter(sizes[k] for k in found) or Counter(sizes.values())
    size = max(count, key=lambda s: (count[s], s))          # the most usable photos; a tie: the larger size
    if prefer is not None and tuple(prefer) in {sizes[k] for k in found}:
        size = tuple(prefer)
    views = {}
    for name, view in found.items():
        if sizes[name] == size:
            views[name] = view
        else:
            skipped.append((name, "size"))
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


def _round(v, n=4):
    """A rounded float, or None when it is not finite (strict JSON)."""
    v = float(v)
    return round(v, n) if math.isfinite(v) else None


# ── the board's shape and the phone's steadiness ─────────────────────────────
STD_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")


def _shape_terms(obj, middle):
    """u^2, u v, v^2 of each board point, u and v from the board's middle in SHAPE_UNIT mm."""
    P = np.asarray(obj, np.float64).reshape(-1, 3)
    u = (P[:, 0] - middle[0]) / SHAPE_UNIT
    v = (P[:, 1] - middle[1]) / SHAPE_UNIT
    return np.column_stack([u * u, u * v, v * v])


def shape_used(shape):
    """Whether a fit_shape result keeps the model with the board's shape: a board not flat, or a shape that tells
    (SHAPE_F); a result of before SHAPE_F counts by its "flat" alone."""
    return bool(shape and shape.get("coef_mm") and (shape.get("used") or not shape.get("flat", True)))


def board_surface(obj, shape):
    """The board's points on its fitted surface (fit_shape's result; unchanged when None or its shape not used),
    same type."""
    obj = np.asarray(obj)
    if not shape_used(shape):
        return obj
    P = obj.astype(np.float64).reshape(-1, 3).copy()
    P[:, 2] += _shape_terms(P, shape["middle_mm"]) @ np.asarray(shape["coef_mm"], np.float64)
    return P.reshape(obj.shape).astype(obj.dtype)


def _sag(coef, extent, middle, n=21):
    """Peak to valley (mm) of the quadratic surface off its best plane, over the board's extent."""
    X, Y = np.meshgrid(np.linspace(*extent[0], n), np.linspace(*extent[1], n))
    P = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    z = _shape_terms(P, middle) @ np.asarray(coef, np.float64)
    A = np.column_stack([np.ones(X.size), X.ravel(), Y.ravel()])
    dev = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    return float(dev.max() - dev.min())


def _lm(fun, jac, x, iters=100):
    """Levenberg-Marquardt on the normal equations with Marquardt's scaling: (x, jacobian at x)."""
    r = fun(x)
    cost, J, lam = float(r @ r), jac(x), 1e-4
    for _ in range(iters):
        A, g = J.T @ J, J.T @ r
        D = np.diag(A).copy()
        D[D <= 0] = 1.0
        while True:
            try:
                step = np.linalg.solve(A + lam * np.diag(D), -g)
            except np.linalg.LinAlgError:
                step = None
            if step is not None and np.all(np.isfinite(step)):
                x1 = x + step
                r1 = fun(x1)
                c1 = float(r1 @ r1)
                if math.isfinite(c1) and c1 <= cost:
                    break
            lam *= 10
            if lam > 1e10:
                return x, J
        done = cost - c1 <= 1e-12 * cost
        x, r, cost, J = x1, r1, c1, jac(x1)
        lam = max(lam / 10, 1e-9)
        if done:
            break
    return x, J


def _bundle(obj, img, K, dist, poses, middle, shape=True, free_centre=False):
    """Bundle adjustment of the camera (fx, fy, cx, cy, 5 distortion coefficients), the pose of each view, with the
    board's quadratic shape (3 numbers) and the optical centre of each view (2 numbers each, mean zero) when asked.
    obj, img: lists of N x 3, N x 2; poses: [(rvec, tvec)]. Jacobian from projectPoints (the board point's depth
    through the translation's: d(uv)/dP = d(uv)/dt R)."""
    nv = len(obj)
    i_pose, i_shape = 9, 9 + 6 * nv
    i_cen = i_shape + (3 if shape else 0)
    npar = i_cen + (2 * nv if free_centre else 0)
    x0 = np.zeros(npar)
    x0[:4] = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d0 = np.zeros(5)
    d = np.asarray(dist, np.float64).ravel()[:5]
    d0[:len(d)] = d
    x0[4:9] = d0
    for k, (r, t) in enumerate(poses):
        x0[i_pose + 6 * k:i_pose + 6 * k + 6] = np.r_[np.ravel(r), np.ravel(t)]
    obj = [np.asarray(o, np.float64).reshape(-1, 3) for o in obj]
    img = [np.asarray(i, np.float64).reshape(-1, 2) for i in img]
    terms = [_shape_terms(o, middle) for o in obj]
    rows = np.cumsum([0] + [2 * len(o) for o in obj])
    m = int(rows[-1]) + (2 if free_centre else 0)
    w_mean = 100.0          # weight of the centres' mean (px) held at zero

    def project(k, x):
        Kc = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1.0]])
        if free_centre:
            Kc[0, 2] += x[i_cen + 2 * k]
            Kc[1, 2] += x[i_cen + 2 * k + 1]
        P = obj[k]
        if shape:
            P = P.copy()
            P[:, 2] += terms[k] @ x[i_shape:i_shape + 3]
        p = x[i_pose + 6 * k:i_pose + 6 * k + 6]
        uv, J = cv2.projectPoints(P, p[:3], p[3:], Kc, x[4:9])
        return uv.reshape(-1, 2), J

    def fun(x):
        r = np.empty(m)
        for k in range(nv):
            r[rows[k]:rows[k + 1]] = (project(k, x)[0] - img[k]).ravel()
        if free_centre:
            r[rows[-1]:] = w_mean * x[i_cen:].reshape(-1, 2).mean(0)
        return r

    def jac(x):
        J = np.zeros((m, npar))
        for k in range(nv):
            _, Jp = project(k, x)
            sl = slice(rows[k], rows[k + 1])
            J[sl, 0:4] = Jp[:, 6:10]
            J[sl, 4:9] = Jp[:, 10:15]
            J[sl, i_pose + 6 * k:i_pose + 6 * k + 6] = Jp[:, 0:6]
            if shape:
                R = cv2.Rodrigues(x[i_pose + 6 * k:i_pose + 6 * k + 3])[0]
                J[sl, i_shape:i_shape + 3] = (Jp[:, 3:6] @ R[:, 2])[:, None] * np.repeat(terms[k], 2, axis=0)
            if free_centre:
                J[sl, i_cen + 2 * k:i_cen + 2 * k + 2] = Jp[:, 8:10]
        if free_centre:
            J[rows[-1], i_cen::2] = w_mean / nv
            J[rows[-1] + 1, i_cen + 1::2] = w_mean / nv
        return J

    x, J = _lm(fun, jac, x0)
    per = [project(k, x)[0] - img[k] for k in range(nv)]
    allr = np.concatenate(per)
    n_res = allr.size
    s2 = float(np.sum(allr ** 2)) / max(n_res - npar + (2 if free_centre else 0), 1)
    sd = np.sqrt(np.abs(np.diag(np.linalg.pinv(J.T @ J)))) * math.sqrt(s2)
    return {"rms": float(np.sqrt(np.mean(np.sum(allr ** 2, 1)))),
            "K": np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1.0]]), "dist": x[4:9].copy(), "sd": sd[:9],
            "shape": x[i_shape:i_shape + 3].copy() if shape else np.zeros(3),
            "centres": x[i_cen:].reshape(-1, 2).copy() if free_centre else None,
            "sd_centres": sd[i_cen:].reshape(-1, 2) if free_centre else None,
            "poses": [(x[i_pose + 6 * k:i_pose + 6 * k + 3].reshape(3, 1).copy(),
                       x[i_pose + 6 * k + 3:i_pose + 6 * k + 6].reshape(3, 1).copy()) for k in range(nv)],
            "per_view": [float(np.sqrt(np.mean(np.sum(r ** 2, 1)))) for r in per]}


def _chi2_quantile(dof, z=STILL_CHI2_Z):
    """The chi-square quantile at the normal quantile z (Wilson and Hilferty)."""
    k = max(int(dof), 1)
    return k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3


def fit_shape(views, size, K, dist, poses, square=None, centre=True):
    """After the flat solve, the board as a quadratic surface (z = a u^2 + b u v + c v^2, u and v from its middle in
    SHAPE_UNIT mm) fitted with the camera and the poses. The board is not flat ("flat" false) when the surface bends
    more than FLAT_SAG_MM over the board (peak to valley off its plane: on a curved monitor the sagitta, width^2 /
    (8 radius)) or the error falls by more than FLAT_RMS_DROP with it; then the intrinsics of this model are the ones
    to keep. square (mm): the board reaches one square beyond its outer corners (else the corners' extent is used).

    centre: then, as a diagnosis only, the optical centre free for each photo (mean zero) on top of the shape: a phone
    held in the hand moves it (lens stabiliser, rolling shutter). Its spread (px) is the RMS of the offsets with the
    part the noise explains taken away (each offset weighed by its precision); "still" is false when it is above
    STILL_CENTRE_PX and the offsets are beyond chance (chi-square past its 99.9 % quantile).

    Returns {"flat", "sag_mm", "rms_flat_px", "rms_shape_px", "drop" (share), "focal_shift" (relative change of the
    focal length from the flat model), "coef_mm", "middle_mm", "extent_mm", "K", "dist", "std" {fx ... k3},
    "per_view_rms_px", "centre" (None, or {"spread_px", "max_px", "rms_px", "chi2", "dof", "still", "per_view_px"}),
    "_poses" {name: (rvec, tvec)} of the shape model}; with "fitted" false (and flat) when the fit could not be
    made."""
    names = list(views)
    obj = [np.asarray(views[n][0], np.float64).reshape(-1, 3) for n in names]
    img = [np.asarray(views[n][1], np.float64).reshape(-1, 2) for n in names]
    K, dist = np.asarray(K, np.float64), np.asarray(dist, np.float64).ravel()
    sq_flat = sum(float(np.sum(view_rms((o, i), K, dist, poses[n])[1] ** 2)) for n, o, i in zip(names, obj, img))
    npts = sum(len(o) for o in obj)
    rms_flat = math.sqrt(sq_flat / max(npts, 1))
    allp = np.vstack(obj)
    pad = float(square) if square else 0.0
    extent = [[float(allp[:, 0].min() - pad), float(allp[:, 0].max() + pad)],
              [float(allp[:, 1].min() - pad), float(allp[:, 1].max() + pad)]]
    middle = [float(np.mean(extent[0])), float(np.mean(extent[1]))]
    out = {"fitted": False, "flat": True, "used": False, "f_test": None, "sag_mm": None, "rms_flat_px": round(rms_flat, 4), "rms_shape_px": None,
           "drop": None, "focal_shift": None, "coef_mm": None, "middle_mm": [round(v, 3) for v in middle],
           "extent_mm": [[round(v, 3) for v in e] for e in extent], "centre": None}
    try:
        q = _bundle(obj, img, K, dist, [poses[n] for n in names], middle, shape=True)
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return out
    if not (math.isfinite(q["rms"]) and q["rms"] <= rms_flat + 1e-9):
        return out
    sag = _sag(q["shape"], extent, middle)
    drop = 1 - q["rms"] / rms_flat if rms_flat > 0 else 0.0
    f_flat, f_shape = (K[0, 0] + K[1, 1]) / 2, (q["K"][0, 0] + q["K"][1, 1]) / 2
    flat = not (sag > FLAT_SAG_MM or drop > FLAT_RMS_DROP)
    left = max(2 * npts - (9 + 6 * len(names) + 3), 1)       # residuals less the numbers of the model with the shape
    ssr_flat, ssr_shape = rms_flat ** 2 * npts, q["rms"] ** 2 * npts
    f_test = (ssr_flat - ssr_shape) / 3 / (ssr_shape / left) if ssr_shape > 0 else math.inf
    out.update({"fitted": True, "flat": flat, "used": bool(not flat or f_test > SHAPE_F),
                "f_test": round(float(min(f_test, 1e9)), 2), "sag_mm": round(sag, 4),
                "rms_shape_px": round(q["rms"], 4), "drop": round(drop, 4),
                "focal_shift": round(float(f_shape / f_flat - 1), 6),
                "coef_mm": [round(float(c), 6) for c in q["shape"]], "K": q["K"].tolist(), "dist": q["dist"].tolist(),
                "std": {k: _round(s) for k, s in zip(STD_NAMES, q["sd"])},
                "per_view_rms_px": {n: _round(e) for n, e in zip(names, q["per_view"])},
                "_poses": dict(zip(names, q["poses"]))})
    if not centre or len(names) < 3:
        return out
    try:
        c = _bundle(obj, img, q["K"], q["dist"], q["poses"], middle, shape=True, free_centre=True)
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return out
    off, sd = c["centres"], c["sd_centres"]
    var = np.sum(sd ** 2, 1)
    ok = np.isfinite(var) & (var > 0) & np.all(np.isfinite(off), 1)
    if ok.sum() < 3:
        return out
    n2, var = np.sum(off[ok] ** 2, 1), var[ok]
    w = 1 / var
    spread = math.sqrt(max(0.0, float(np.sum(w * (n2 - var)) / np.sum(w))))
    chi2 = float(np.sum((off[ok] / sd[ok]) ** 2))
    dof = 2 * (int(ok.sum()) - 1)
    out["centre"] = {"spread_px": round(spread, 3), "max_px": round(float(np.sqrt(n2.max())), 3),
                     "rms_px": round(c["rms"], 4), "chi2": round(chi2, 2), "dof": dof,
                     "still": not (spread > STILL_CENTRE_PX and chi2 > _chi2_quantile(dof)),
                     "per_view_px": {n: [round(float(a), 2), round(float(b), 2)] for n, (a, b) in zip(names, off)}}
    return out


def coverage(points, size):
    """Where in the image the board's corners fell: points, a list of N x 2 arrays (px; one per photo) or one array.
    {"cells": COVER_GRID x COVER_GRID counts (rows from the top), "corners": {name: count} of the four corners of the
    image (beyond COVER_CORNER of the half diagonal, CORNER_NAMES), "missing_cells", "missing_corners": the names
    (CELL_NAMES, CORNER_NAMES) with none, "reach": the farthest point as a share of the half diagonal}."""
    w, h = size
    if isinstance(points, np.ndarray):
        points = [points]
    arrs = [np.asarray(p, np.float64).reshape(-1, 2) for p in points]
    pts = np.vstack(arrs) if arrs else np.zeros((0, 2))
    n = COVER_GRID
    cells = np.zeros((n, n), int)
    np.add.at(cells, (np.clip((pts[:, 1] / h * n).astype(int), 0, n - 1),
                      np.clip((pts[:, 0] / w * n).astype(int), 0, n - 1)), 1)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) / math.hypot(cx, cy)
    far = r > COVER_CORNER
    left, top = pts[:, 0] < cx, pts[:, 1] < cy
    corners = {"top_left": far & left & top, "top_right": far & ~left & top, "bottom_right": far & ~left & ~top,
               "bottom_left": far & left & ~top}
    corners = {k: int(np.sum(corners[k])) for k in CORNER_NAMES}
    names = CELL_NAMES if n == 3 else [[f"r{i}c{j}" for j in range(n)] for i in range(n)]
    return {"cells": cells.tolist(), "corners": corners,
            "missing_cells": [names[i][j] for i in range(n) for j in range(n) if not cells[i, j]],
            "missing_corners": [k for k in CORNER_NAMES if not corners[k]],
            "reach": round(float(r.max()), 3) if len(r) else 0.0}


def calibrate(views, size, exclude=(), square=None):
    """Calibrate on the views but the excluded ones (chosen by hand), then drop the outliers one at a time: the view
    with the largest error goes when it is above OUTLIER_FACTOR times the median (and above OUTLIER_FLOOR_PX), the
    camera is solved again without it and the errors judged again, until none is above, the views left would be
    fewer than MIN_VIEWS, or OUTLIER_MAX_SHARE of them are gone. One at a time because one bad photo pulls the model
    and raises the others' errors (all at once would drop good ones with it, or none when that left too few), and a
    second bad one may show only once the first is gone. With square (mm), where on the board the error is
    (corner_residuals). Every view, used or not, gets its error under the final model (per_view_all_px: PnP for those
    left out). Raises CalibrationError few_photos with fewer than MIN_VIEWS usable photos.

    After each flat solve fit_shape fits the board's shape: a board that is not flat, or whose shape tells beyond the
    noise (SHAPE_F, "used"), keeps the intrinsics, errors and poses of the model with its shape (rms_px is then that
    model's, rms_flat_px the flat one's) and its errors choose the views dropped. The result also holds "shape" (fit_shape: sag_mm, rms_flat_px, rms_shape_px, drop,
    focal_shift ...), "centre" (the optical centre free for each photo: spread_px, still ...), "coverage" (where the
    board reached in the image, coverage()), "issues" (codes of ISSUES) and "limit_px" (the last outlier limit)."""
    exclude = sorted(set(exclude) & set(views))
    use = {k: v for k, v in views.items() if k not in exclude}
    if len(use) < MIN_VIEWS:
        raise CalibrationError("few_photos", f"only {len(use)} usable photos, at least {MIN_VIEWS} needed",
                               n=len(use), need=MIN_VIEWS)
    most = int(OUTLIER_MAX_SHARE * len(use))
    dropped = []
    while True:
        rms, K, dist, std, per_view, poses = solve(use, size)
        shape = fit_shape(use, size, K, dist, poses, square, centre=False)
        errs = shape["per_view_rms_px"] if shape_used(shape) else per_view
        limit = max(OUTLIER_FACTOR * float(np.median(list(errs.values()))), OUTLIER_FLOOR_PX)
        worst = max(errs, key=lambda k: errs[k])
        if errs[worst] <= limit or len(use) - 1 < MIN_VIEWS or len(dropped) >= most:
            break
        dropped.append(worst)
        use = {k: v for k, v in use.items() if k != worst}
    dropped.sort()
    shape = fit_shape(use, size, K, dist, poses, square)
    shape_poses = shape.pop("_poses", None)
    std = {k: _round(s) for k, s in zip(STD_NAMES, std)}
    if shape_used(shape):
        K, dist, std = np.array(shape["K"]), np.array(shape["dist"]), shape["std"]
        per_view, rms, poses = shape["per_view_rms_px"], shape["rms_shape_px"], shape_poses
    for k in ("K", "dist", "std", "per_view_rms_px"):
        shape.pop(k, None)
    on = {k: (board_surface(v[0], shape), v[1]) for k, v in views.items()}      # the board's points as fitted
    every = {k: (float(per_view[k]) if per_view.get(k) is not None else view_rms(v, K, dist)[0])
             for k, v in on.items()}
    centre = shape.pop("centre")
    cover = coverage([v[1] for v in use.values()], size)
    issues = [code for code, bad in (("board_not_flat", not shape["flat"]),
                                     ("phone_not_still", centre is not None and not centre["still"]),
                                     ("corners_missing", bool(cover["missing_corners"]))) if bad]
    out = {
        "image_size": list(size), "K": np.asarray(K).tolist(), "dist": np.asarray(dist).ravel().tolist(),
        "rms_px": round(float(rms), 4), "rms_flat_px": shape["rms_flat_px"], "n_views": len(use),
        "n_corners": int(sum(len(v[0]) for v in use.values())),
        "std": std,
        "per_view_rms_px": {k: _round(e) for k, e in per_view.items()},
        "per_view_all_px": {k: _round(e) for k, e in sorted(every.items())},
        "dropped": dropped, "excluded": exclude, "limit_px": round(limit, 4),
        "shape": shape, "centre": centre, "coverage": cover, "issues": issues,
    }
    if square:
        out["residuals"] = corner_residuals({k: on[k] for k in use}, poses, K, dist, float(square))
    return out


def _std(s, key):
    """A standard deviation of a stored calibration: 0 when it was not known (None, NaN or missing)."""
    v = (s or {}).get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def plain(cal):
    """The intrinsics in plain units: focal length (px, and the field of view in degrees across and down), where the
    optical centre is from the middle of the image (px, + right and down), how far the lens bends at the corners of
    the image (px between where a point is and where a lens without distortion would put it), with their standard
    deviations (None when not known)."""
    K, d = np.array(cal["K"]), np.array(cal["dist"])
    w, h = cal["image_size"]
    s = cal.get("std") or {}
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-12)
    ideal = cv2.undistortPoints(corners, K, d, None, None, K, crit).reshape(-1, 2)
    bend = float(np.max(np.linalg.norm(ideal - corners.reshape(-1, 2), axis=1)))
    known = all(isinstance(s.get(k), (int, float)) and math.isfinite(s[k]) for k in ("fx", "fy"))
    known_c = all(isinstance(s.get(k), (int, float)) and math.isfinite(s[k]) for k in ("cx", "cy"))
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "f_std": float(np.hypot(s["fx"], s["fy"]) / 2) if known else None,
            "fov_x": float(np.degrees(2 * np.arctan(w / 2 / K[0, 0]))),
            "fov_y": float(np.degrees(2 * np.arctan(h / 2 / K[1, 1]))),
            "cx_off": float(K[0, 2] - (w - 1) / 2), "cy_off": float(K[1, 2] - (h - 1) / 2),
            "c_std": float(max(s["cx"], s["cy"])) if known_c else None, "bend_px": bend, "size": [int(w), int(h)]}


# ── the check of each photo as it comes ──────────────────────────────────────
BLUR_PX = 4.0           # a photo is blurred past this sigma (px at 4080 px on its long side, scaled with it, at least
BLUR_MIN_PX = 2.5       # this): the corners then move and the calibration suffers (3.5 px still calibrates)
BLUR_FLOOR = 0.8        # what the estimate gives for an edge with no blur at all (the derivative's own width)
BLUR_MIN_CONTRAST = 20  # grey levels: a corner patch with less is not measured
BLUR_WINDOW = (3, 25, 0.3)  # px: the patch's half side, at least, at most, and as a share of the square's side


def blur_px(gray, obj, imgp, square):
    """The blur of a photo of the board, as the sigma (px) of a Gaussian: at each corner found, across the board's two
    sets of edges, the contrast over the steepest slope (an edge blurred by sigma rises by contrast / (sigma sqrt(2
    pi)) per px at most), the larger of the two (a shake blurs one way), the median over the corners. NaN when no
    corner can be measured."""
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
    lo, hi, share = BLUR_WINDOW
    for (X, Y), (u, v) in zip(obj2, pts):
        p = H @ [X, Y, 1.0]
        ex = H @ [X + square, Y, 1.0]
        ey = H @ [X, Y + square, 1.0]
        jx = ex[:2] / ex[2] - p[:2] / p[2]          # one square along the board's x, in the image
        jy = ey[:2] / ey[2] - p[:2] / p[2]
        side = min(np.linalg.norm(jx), np.linalg.norm(jy))
        r = int(np.clip(share * side, lo, hi))
        x0, y0 = int(round(u)) - r, int(round(v)) - r
        if x0 < 1 or y0 < 1 or x0 + 2 * r + 2 > ww or y0 + 2 * r + 2 > hh:
            continue
        patch = g[y0 - 1:y0 + 2 * r + 2, x0 - 1:x0 + 2 * r + 2]
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1] / 8
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1] / 8
        inner = patch[1:-1, 1:-1]
        c = float(np.percentile(inner, 97) - np.percentile(inner, 3))
        if c < BLUR_MIN_CONTRAST:
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


def blur_limit(long_side):
    """The blur a photo may have, for the long side of its image (px): the same for a photo held either way."""
    return max(BLUR_MIN_PX, BLUR_PX * long_side / max(SYNTH_SIZE))


# ── the camera's settings, from the EXIF ─────────────────────────────────────
EXIF_IFD = 0x8769
MAKER_JSON = 0x9999     # Xiaomi's camera notes, JSON: Hdr, AIScene, zoomMultiple, OpMode


def _num(v):
    """A number of the EXIF (int, float, rational, or the first of a tuple), or None."""
    if isinstance(v, (tuple, list)):
        v = v[0] if v else None
    try:
        v = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return v if math.isfinite(v) else None


def photo_settings(path):
    """The camera's settings of a photo, from its EXIF (Pillow): {"known": false} when it has none (a synthetic photo,
    a frame of a film), else {"known": true, "program" (ExposureProgram: 1 manual, 2 automatic ...), "white_balance"
    (0 automatic, 1 manual), "iso", "exposure_s", "focal_mm", "zoom" (the camera notes' zoomMultiple, else
    DigitalZoomRatio), "ai_scene" (0 off), "hdr" ("off" ...), "mode" (the camera notes' OpMode), "gain_map" (an Ultra
    HDR JPEG: kept as information, the photo itself is the plain one), "pixels" (width x height), "model"}; a value
    the photo does not carry is None."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            e = ex.get_ifd(EXIF_IFD)
            pixels = int(im.size[0] * im.size[1])
            xmp = im.info.get("xmp") or b""
    except Exception:           # any file Pillow cannot read: nothing known (OpenCV judges the photo itself)
        return {"known": False}
    if not e:
        return {"known": False}
    notes = {}
    raw = e.get(MAKER_JSON) or ex.get(MAKER_JSON)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            got = json.loads(raw.strip("\x00 "))
            notes = got if isinstance(got, dict) else {}
    zoom = _num(notes.get("zoomMultiple"))
    if zoom is None:
        zoom = _num(e.get(41988))           # DigitalZoomRatio, 0 when not used
        zoom = 1.0 if zoom == 0 else zoom
    program, wb, ai = _num(e.get(34850)), _num(e.get(41987)), _num(notes.get("AIScene"))
    hdr = notes.get("Hdr")
    model = ex.get(272)
    return {"known": True, "program": None if program is None else int(program),
            "white_balance": None if wb is None else int(wb), "iso": _num(e.get(34855)),
            "exposure_s": _num(e.get(33434)), "focal_mm": _num(e.get(37386)), "zoom": zoom,
            "ai_scene": None if ai is None else int(ai), "hdr": None if hdr is None else str(hdr),
            "mode": notes.get("OpMode"), "gain_map": (b"GainMap" if isinstance(xmp, bytes) else "GainMap") in xmp,
            "pixels": pixels, "model": str(model).strip("\x00 ") if model else None}


def settings_problems(s, first=None):
    """What is wrong in a photo's settings (photo_settings) for a calibration, as codes of SETTINGS: exposure or white
    balance automatic, the AI scene, HDR or a zoom on; against the first photo of the series (its settings, when
    given) another ISO, exposure time, focal length or resolution. Nothing when the EXIF is not known."""
    if not s or not s.get("known"):
        return []
    out = []
    if s.get("program") is not None and s["program"] != 1:
        out.append("exposure_auto")
    if s.get("white_balance") is not None and s["white_balance"] != 1:
        out.append("white_balance_auto")
    if s.get("ai_scene"):
        out.append("ai_scene")
    if s.get("hdr") is not None and s["hdr"].strip().lower() not in ("off", "0", "false", ""):
        out.append("hdr")
    if s.get("zoom") is not None and abs(s["zoom"] - 1) > 0.01:
        out.append("zoom")
    f = first if first and first.get("known") else None
    if f:
        for key, code, rel in (("iso", "iso_changed", 0.0), ("exposure_s", "exposure_changed", 0.01),
                               ("focal_mm", "focal_changed", 0.002), ("pixels", "size_changed", 0.0)):
            a, b = s.get(key), f.get(key)
            if a is not None and b is not None and abs(a - b) > rel * abs(b) + 1e-9:
                out.append(code)
    return out


def check_photo(path, board, size=None, first=None):
    """The check of one photo as it arrives: can it go into the calibration? {"ok", "why" (none: no board found,
    few: fewer than MIN_CORNERS corners or all on one line ("collinear" then true), size: another size than the
    photos before it (the phone held another way), settings: the camera not set by hand (settings_problems; first:
    the first photo of the series, its path or its photo_settings, for what must not change), blur: blurred past
    blur_limit, or its sharpness could not be measured on any corner, unreadable), "corners", "of", "blur_px", "size",
    "points" (where the corners are, fractions of the width and height, at most 120), "coverage" (where the corners
    fell, coverage(); None without corners), "settings" (photo_settings with "problems": the codes)}."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    nx, ny = board.getChessboardSize()
    out = {"ok": False, "why": "", "corners": 0, "of": (nx - 1) * (ny - 1), "blur_px": None, "size": None,
           "points": [], "coverage": None, "settings": None}
    if img is None:
        out["why"] = "unreadable"
        return out
    w, h = img.shape[::-1]
    out["size"] = [w, h]
    if size and list(size) != [w, h]:
        out["why"] = "size"
        return out
    settings = photo_settings(path)
    if isinstance(first, (str, os.PathLike)):
        first = photo_settings(first)
    problems = settings_problems(settings, first)
    out["settings"] = {**settings, "problems": problems}
    out["why"] = _board_why(img, board, out)
    if problems:
        out["why"] = "settings"
    out["ok"] = not out["why"]
    return out


def _board_why(img, board, out):
    """check_photo's judgement of the board in a photo (none, blur, few or ""), filling corners, points, coverage,
    collinear and blur_px of out."""
    h, w = img.shape[:2]
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(img)
    n = 0 if ids is None else len(ids)
    out["corners"] = n
    if n:
        pts = corners.reshape(-1, 2)
        k = np.linspace(0, n - 1, min(n, 120)).astype(int)
        out["points"] = [[round(float(x / w), 4), round(float(y / h), 4)] for x, y in pts[k]]
        out["coverage"] = coverage([pts], (w, h))
    if n == 0:
        return "none"
    collinear = bool(board.checkCharucoCornersCollinear(ids))
    out["collinear"] = collinear
    if n >= 4 and not collinear:        # blurred is said before few: a blur also loses corners
        obj, imgp = board.matchImagePoints(corners, ids)
        b = blur_px(img, obj, imgp, board.getSquareLength())
        out["blur_px"] = round(b, 2) if math.isfinite(b) else None
        # a blur that could not be measured (every patch at the border or without contrast) is never said as a
        # blur: the photo is judged by its corners below, which a real blur loses
        if math.isfinite(b) and b > blur_limit(max(w, h)):
            return "blur"
    if n < MIN_CORNERS or collinear:
        return "few"
    return ""


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


def summary(cal, skipped, log: Log | None = None):
    say = _say(log)
    K = np.array(cal["K"])
    s = {k: (v if v is not None else float("nan")) for k, v in cal["std"].items()}
    say(f"views {cal['n_views']} (corners {cal['n_corners']}), RMS {cal['rms_px']:.3f} px")
    say(f"fx {K[0, 0]:.2f} +/- {s['fx']:.2f}  fy {K[1, 1]:.2f} +/- {s['fy']:.2f}  "
        f"cx {K[0, 2]:.2f} +/- {s['cx']:.2f}  cy {K[1, 2]:.2f} +/- {s['cy']:.2f}")
    say("dist " + " ".join(f"{d:.5f}" for d in cal["dist"]))
    for name, why in skipped:
        say(f"skipped {name}: {why}")
    for name in cal["dropped"]:
        say(f"dropped {name}: per-view error above 3 times the median")
    for name in cal.get("excluded", []):
        say(f"left out {name}: chosen by hand")
    sh, ce, co = cal.get("shape") or {}, cal.get("centre"), cal.get("coverage")
    if sh.get("fitted"):
        say(f"board shape: bends {sh['sag_mm']:.2f} mm; RMS flat {sh['rms_flat_px']:.3f} px, with the shape "
            f"{sh['rms_shape_px']:.3f} px; focal length {100 * sh['focal_shift']:+.2f} % with the shape"
            + ("" if sh["flat"] else "  NOT FLAT: the intrinsics of the curved board are kept"))
    if ce:
        say(f"optical centre free for each photo: RMS {ce['rms_px']:.3f} px, the centre spreads {ce['spread_px']:.1f} "
            f"px (at most {ce['max_px']:.1f} px)" + ("" if ce["still"] else "  THE PHONE MOVED"))
    if co:
        say(f"the board reached {100 * co['reach']:.0f} % of the way to the image's corners"
            + (f"; never in the corners {', '.join(co['missing_corners'])}" if co["missing_corners"] else ""))
    for name, codes in (cal.get("settings") or {}).items():
        say(f"settings of {name}: {', '.join(codes)}")
    if cal.get("issues"):
        say("issues: " + ", ".join(cal["issues"]))


def print_compare(c, log: Log | None = None):
    _say(log)(f"error fx {100 * c['fx_rel']:+.3f} %  fy {100 * c['fy_rel']:+.3f} %  "
              f"cx {c['cx_px']:+.2f} px  cy {c['cy_px']:+.2f} px  "
              f"model max {c['model_max_px']:.3f} px, RMS {c['model_rms_px']:.3f} px")


def _read_json(path, code):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        raise CalibrationError(code, f"{Path(path).name} cannot be read: {e}", file=Path(path).name) from e


def run_calibrate(images, out, truth, square, squares=None, marker=None, exclude=(), log: Log | None = None,
                  skip=None):
    """Calibrates on the photos of a folder (CalibrationError no_photos, few_photos); with `out` the result is written
    there in one step; with `truth` it is compared with it. Reports through `log` (print when None). "settings":
    {photo: codes of SETTINGS} of the photos used whose camera settings are wrong (against the first one used for
    what must not change), and then the issue camera_settings. skip: {photo: why} left out before their corners are
    looked for (the app's photos that did not pass their check), said in "skipped" with their reason."""
    board = make_board(square, squares, marker)
    skip = {str(k): str(v) for k, v in (skip or {}).items()}
    paths = [p for p in find_images(images) if p.name not in skip]
    if not paths:
        raise CalibrationError("few_photos", f"no photo left of {len(skip)}", n=0, need=MIN_VIEWS)
    views, skipped, size = detect(paths, board)
    skipped = sorted(skipped + [(k, v) for k, v in skip.items()])
    if size is None:
        raise CalibrationError("few_photos", "no photo can be read", n=0, need=MIN_VIEWS)
    cal = calibrate(views, size, exclude, board.getSquareLength())
    wrong, first = {}, None
    for p in paths:
        if p.name in cal["per_view_rms_px"]:
            s = photo_settings(p)
            first = first or (s if s.get("known") else None)
            if problems := settings_problems(s, first):
                wrong[p.name] = problems
    if wrong:
        cal["issues"].append("camera_settings")
    cal = {"images": str(Path(images)), "date": date.today().isoformat(), "board": board_spec(board),
           "skipped": dict(skipped), **cal, "settings": wrong}
    summary(cal, skipped, log)
    if truth:
        cal["error"] = compare(cal, _read_json(truth, "intrinsics_bad"))
        print_compare(cal["error"], log)
    if out:
        write_json_atomic(out, cal)
        _say(log)(f"intrinsics: {out}")
    return cal


def run_check(images, intrinsics, square, squares=None, marker=None, log: Log | None = None):
    """Start-of-session check of the stored intrinsics on about 10 new photos.

    Refocusing changes mostly the focal length and the principal point, so these
    are estimated again with the stored distortion; 10 photos are too few for the
    whole model. The stored intrinsics stay valid when each change is within 3
    combined standard deviations or below the floor, and when they reproject the
    new photos with an RMS below MAX_RMS.

    The board is taken with the shape the calibration found for it (its
    "shape": a bent sheet, a curved monitor) when the photos are of the same
    board (squares and square); else, or when it was flat, as a plane.

    Returns (ok, report): report holds the RMS of the stored model on these
    photos, the views used and skipped, the board's shape used ("shape": flat,
    stored, or other_board) and per parameter the stored and new value, the
    change, the limit and whether it passed. Raises CalibrationError:
    intrinsics_bad (the stored file cannot be read), no_photos, few_photos,
    intrinsics_size (photos of another camera mode).
    """
    say = _say(log)
    ref = _read_json(intrinsics, "intrinsics_bad")
    try:
        K0, d0 = np.array(ref["K"], float), np.array(ref["dist"], float)
        ref_size = [int(v) for v in ref["image_size"]]
    except (KeyError, TypeError, ValueError) as e:
        raise CalibrationError("intrinsics_bad", f"{Path(intrinsics).name}: {e}", file=Path(intrinsics).name) from e
    board = make_board(square, squares, marker)
    stored, mine = ref.get("board") or {}, board_spec(board)
    shape = ref.get("shape") if isinstance(ref.get("shape"), dict) else None
    curved = bool(shape and shape.get("coef_mm") and shape.get("middle_mm")
                  and (shape.get("used") or not shape.get("flat", True)))
    same_board = (list(stored.get("squares") or []) == mine["squares"]
                  and abs(float(stored.get("square_mm") or 0) - mine["square_mm"]) < 1e-6)
    shape_used = "flat" if not curved else "stored" if same_board else "other_board"
    views, skipped, size = detect(find_images(images), board, prefer=ref_size)
    if shape_used == "stored":      # the board's points where the calibration found them
        views = {k: (board_surface(o, shape), i) for k, (o, i) in views.items()}
    if size is None:
        raise CalibrationError("few_photos", "no photo can be read", n=0, need=MIN_VIEWS)
    if list(size) != ref_size:
        raise CalibrationError("intrinsics_size", f"photo size {size} differs from the calibration "
                                                  f"{tuple(ref_size)}: wrong camera mode", got=list(size),
                               want=ref_size)
    if len(views) < MIN_VIEWS:
        raise CalibrationError("few_photos", f"only {len(views)} usable photos, at least {MIN_VIEWS} needed",
                               n=len(views), need=MIN_VIEWS)
    for name, why in skipped:
        say(f"skipped {name}: {why}")
    obj = [v[0] for v in views.values()]
    img = [v[1] for v in views.values()]
    # stored model on these photos: pose of each view by PnP
    sq = sum(float(np.sum((cv2.projectPoints(o, *cv2.solvePnP(o, i, K0, d0)[1:], K0, d0)[0].reshape(-1, 2)
                           - i.reshape(-1, 2)) ** 2)) for o, i in zip(obj, img))
    rms_stored = np.sqrt(sq / sum(len(o) for o in obj))
    say(f"views {len(views)}, stored intrinsics reproject them with RMS {rms_stored:.3f} px"
        + {"flat": "", "stored": ", the board with its stored shape",
           "other_board": ", the board flat (the stored shape is of another board)"}[shape_used])
    ok = bool(rms_stored < MAX_RMS)
    report = {"images": str(Path(images)), "intrinsics": str(Path(intrinsics)), "n_views": len(views),
              "skipped": dict(skipped), "rms_stored_px": round(float(rms_stored), 4), "rms_ok": ok, "params": {},
              "shape": shape_used}
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
             | cv2.CALIB_FIX_TANGENT_DIST)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    K, std = cv2.calibrateCameraExtended(obj, img, size, K0.copy(), d0.copy(), flags=flags, criteria=crit)[1:6:4]
    for i, (name, r, c) in enumerate([("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)]):
        delta = K[r, c] - K0[r, c]
        new_std = float(std[i, 0]) if math.isfinite(float(std[i, 0])) else 0.0
        limit = max(3 * np.hypot(new_std, _std(ref.get("std"), name)),
                    CHECK_FOCAL * K0[r, c] if r == c else CHECK_CENTRE)
        ok &= bool(abs(delta) <= limit)
        report["params"][name] = {"stored": round(float(K0[r, c]), 3), "now": round(float(K[r, c]), 3),
                                  "change": round(float(delta), 3), "limit": round(float(limit), 3),
                                  "ok": bool(abs(delta) <= limit)}
        say(f"{name} {K0[r, c]:.2f} -> {K[r, c]:.2f}  change {delta:+.2f} px, limit {limit:.2f} px"
            + ("" if abs(delta) <= limit else "  CHANGED"))
    say("check: " + ("ok, stored intrinsics still valid" if ok else "FAILED, recalibrate"))
    report["ok"] = ok
    return ok, finite(report)


def _raises(fn, code):
    try:
        fn()
    except CalibrationError as e:
        return e.code == code
    return False


def run_test(work, real=None):
    """Synthetic end-to-end test: 40 calibration photos, then two session checks; then the failures a library must
    report as codes, never by leaving the process; then what spoils a calibration (test_quality; real: a folder of
    photos known to be of a curved board, taken by hand in the automatic mode)."""
    try:
        work = test_folder(work)
    except ValueError as e:
        print(f"FAIL  {e}")
        return False
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
    ok = test_failures(work) and ok
    ok = test_quality(work, cal, real) and ok
    print("test: " + ("PASS" if ok else "FAIL"))
    return ok


def test_failures(work):
    """Coded errors in place of exits, mixed photo sizes, small boards, the printable page, strict JSON, the log."""
    fails = []

    def expect(good, what):
        print(("ok    " if good else "FAIL  ") + what, flush=True)
        if not good:
            fails.append(what)

    bad = work / "board_other"
    bad.mkdir()
    (bad / BOARD_FILE).write_text(json.dumps({"dictionary": "DICT_4X4_50", "squares": [5, 4], "square_mm": 30,
                                              "marker_mm": 22}), encoding="utf-8")
    (work / "board_cut").mkdir()
    (work / "board_cut" / BOARD_FILE).write_text('{"squares": [5,', encoding="utf-8")
    expect(_raises(lambda: folder_board(bad), "board_unsupported")
           and _raises(lambda: folder_board(work / "board_cut"), "board_unreadable")
           and isinstance(CalibrationError("x"), Exception) and not isinstance(CalibrationError("x"), SystemExit),
           "a board.json of another board or cut short: a coded error, never SystemExit")
    expect(_raises(lambda: find_images(work / "board_other"), "no_photos"), "a folder without photos: no_photos")
    mixed = work / "mixed"
    shutil.copytree(work / "synth", mixed)
    turned = cv2.rotate(cv2.imread(str(mixed / "synth_000.jpg")), cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(mixed / "synth_000.jpg"), turned)
    (mixed / "broken.jpg").write_bytes(b"not a photo")
    lines = []
    cal = run_calibrate(mixed, mixed / "out.json", None, BOARD["square"], log=lines.append)
    expect(cal["skipped"].get("synth_000.jpg") == "size" and cal["skipped"].get("broken.jpg") == "unreadable"
           and cal["n_views"] >= MIN_VIEWS, "one photo held the other way: skipped as size, the others calibrate")
    text = (mixed / "out.json").read_text(encoding="utf-8")
    expect("NaN" not in text and json.loads(text)["rms_px"] == cal["rms_px"]
           and not list(mixed.glob(".*.tmp")) and any(x.startswith("views") for x in lines),
           "the result written in one step, strict JSON, reported through log")
    left = run_calibrate(mixed, None, None, BOARD["square"], log=lines.append,
                         skip={"synth_002.jpg": "check", "synth_003.jpg": "check"})
    expect(left["skipped"].get("synth_002.jpg") == left["skipped"].get("synth_003.jpg") == "check"
           and not {"synth_002.jpg", "synth_003.jpg"} & set(left["per_view_all_px"])
           and left["skipped"].get("synth_000.jpg") == "size",
           "photos the app's check put aside: left out before detection, said as skipped with their reason")
    # outliers dropped one at a time: the worst first, the errors judged again; never below MIN_VIEWS
    base, _, bsize = detect(find_images(work / "synth")[:12], make_board(BOARD["square"]))
    rng = np.random.default_rng(7)

    def noisy(names, sigmas):
        out = {k: (o, i.copy()) for k, (o, i) in base.items()}
        for k, s in zip(names, sigmas):
            out[k] = (out[k][0], (out[k][1] + rng.normal(0, s, out[k][1].shape)).astype(np.float32))
        return out
    names = sorted(base)
    two = calibrate(noisy(names[:2], (1.0, 2.0)), bsize)
    few = calibrate({k: v for k, v in noisy(names[:2], (1.0, 2.0)).items() if k in names[:9]}, bsize)
    many = calibrate(noisy(names[:5], (1.0, 1.2, 1.4, 1.6, 1.8)), bsize)
    expect(two["dropped"] == sorted(names[:2]) and few["dropped"] == [names[1]] and few["n_views"] == MIN_VIEWS
           and len(many["dropped"]) == int(OUTLIER_MAX_SHARE * 12) and set(many["dropped"]) <= set(names[:5]),
           f"outliers dropped one at a time: two bad photos of 12 both ({two['dropped']}), of 9 the worse one only, "
           f"8 left ({few['dropped']}), of five bad at most a quarter ({many['dropped']})")
    held = (turned.shape[1], turned.shape[0])
    views, _, size = detect(find_images(mixed), make_board(BOARD["square"]), prefer=held)
    expect(size == held and list(views) == ["synth_000.jpg"],
           "the session check takes the photos of the stored calibration's size, even when fewer than the others")
    real_blur = globals()["blur_px"]
    globals()["blur_px"] = lambda *a, **k: float("nan")      # every corner patch at the border or without contrast
    try:
        unmeasured = check_photo(mixed / "synth_001.jpg", make_board(BOARD["square"]))
    finally:
        globals()["blur_px"] = real_blur
    expect(unmeasured["ok"] and unmeasured["blur_px"] is None and unmeasured["corners"] >= MIN_CORNERS,
           "a photo whose blur cannot be measured: judged by its corners, never said blurred")
    expect(finite({"a": [float("nan"), 1.0], "b": float("inf")}) == {"a": [None, 1.0], "b": None}
           and plain({**cal, "std": {**cal["std"], "fx": None}})["f_std"] is None,
           "NaN and infinity become null; an unknown standard deviation stays unknown")
    junk = work / "junk"
    junk.mkdir()
    for k in range(3):
        (junk / f"x{k}.jpg").write_bytes(b"x")
    expect(_raises(lambda: run_check(junk, work / "synth_intrinsics.json", BOARD["square"]), "few_photos"),
           "a session check on photos that cannot be read: few_photos, not a TypeError")
    (work / "cut.json").write_text("{", encoding="utf-8")
    expect(_raises(lambda: run_check(work / "check_same", work / "cut.json", BOARD["square"]), "intrinsics_bad"),
           "stored intrinsics that cannot be read: intrinsics_bad")
    small = make_board(30.0, (5, 4))
    rng = np.random.default_rng(3)
    R, t = sample_pose(rng, small, SYNTH_K, SYNTH_DIST, SYNTH_SIZE)
    expect(R.shape == (3, 3), "a board with fewer than 24 corners gets a pose (no endless loop)")
    expect(_raises(lambda: sample_pose(rng, small, SYNTH_K, SYNTH_DIST, (40, 30), tries=50), "pose_not_found"),
           "a pose that cannot be found ends with pose_not_found")
    import matplotlib
    backend = matplotlib.get_backend()
    pdf = write_board(work / "print", 30.0, log=lines.append)
    data = pdf.read_bytes()
    import re
    box = [float(v) for v in re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", data).groups()]
    png = cv2.imread(str(pdf.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
    want = (png.shape[1] / PRINT_PPM / 25.4 * 72, png.shape[0] / PRINT_PPM / 25.4 * 72)
    expect(abs(box[0] - want[0]) < 1 and abs(box[1] - want[1]) < 1 and "a4" not in pdf.name
           and matplotlib.get_backend() == backend,
           f"a board larger than A4: a page of its own size ({box[0] / 72 * 25.4:.0f} x {box[1] / 72 * 25.4:.0f} mm), "
           f"not squashed; the plotting backend left as it was")
    a4 = write_board(work / "print", BOARD["square"], log=lines.append)
    expect(a4.name.endswith("_a4.pdf"), "the A4 board keeps its name")
    return not fails


def _with_exif(src, dst, program, white_balance, iso, exposure_s, notes):
    """A copy of a photo carrying the EXIF of a phone's camera (Pillow), for the settings tests."""
    from PIL import Image
    with Image.open(src) as im:
        exif = Image.Exif()
        exif[272] = "test phone"
        ifd = exif.get_ifd(EXIF_IFD)
        ifd.update({34850: program, 41987: white_balance, 34855: iso, 33434: exposure_s, 37386: 4.941, 41988: 1.0,
                    MAKER_JSON: json.dumps(notes)})
        im.save(dst, exif=exif, quality=92)


def _spread(shifts):
    """RMS (px) of the optical centre's offsets from their mean."""
    s = np.asarray(shifts, np.float64)
    return float(np.sqrt(np.mean(np.sum((s - s.mean(0)) ** 2, 1))))


def test_quality(work, cal, real=None):
    """What spoils a calibration: a board bent as a 3000R monitor, the optical centre moving between the photos, the
    corners of the image never reached, the camera's automatic settings; the refined corners against the truth; the
    good synthetic calibration (cal, of work/synth) said free of them. real: a folder of photos known to be of a
    curved board, taken by hand in the camera's automatic mode, whose issues must all be said."""
    fails = []

    def expect(good, what):
        print(("ok    " if good else "FAIL  ") + what, flush=True)
        if not good:
            fails.append(what)

    quiet = [].append
    sh, ce = cal["shape"], cal["centre"]
    expect(sh["fitted"] and sh["flat"] and sh["sag_mm"] < 0.05 and ce is not None and ce["still"]
           and not set(cal["issues"]) & set(ISSUES_NOT_GOOD),
           f"a flat board and a still phone: flat (bends {sh['sag_mm']:.3f} mm, error {sh['rms_flat_px']:.3f} -> "
           f"{sh['rms_shape_px']:.3f} px with the shape), still (the centre spreads {ce and ce['spread_px']} px); "
           f"issues {cal['issues']}, the board reached {100 * cal['coverage']['reach']:.0f} % of the way to the "
           f"image's corners")
    # the refined corners against the truth
    truth = json.loads((work / "synth" / "truth.json").read_text(encoding="utf-8"))
    K, d = np.array(truth["K"]), np.array(truth["dist"])
    paths = find_images(work / "synth")[:12]
    board = make_board()
    raw, fine = detect(paths, board, refine=False)[0], detect(paths, board)[0]
    err = {"raw": [], "fine": []}
    for v in truth["views"]:
        name = v["image"]
        if name in raw and name in fine:
            true = cv2.projectPoints(raw[name][0].astype(np.float64), np.array(v["rvec"]), np.array(v["tvec"]), K,
                                     d)[0].reshape(-1, 2)
            for key, got in (("raw", raw), ("fine", fine)):
                err[key].append(np.linalg.norm(got[name][1].reshape(-1, 2) - true, axis=1))
    e0, e1 = np.concatenate(err["raw"]), np.concatenate(err["fine"])
    r0, r1 = (float(np.sqrt(np.mean(e ** 2))) for e in (e0, e1))
    expect(r1 <= r0 and e1.max() <= e0.max() + 0.05,
           f"refined corners of a printed board against the truth: RMS {r0:.4f} -> {r1:.4f} px, at most "
           f"{e0.max():.3f} -> {e1.max():.3f} px ({len(e1)} corners)")
    # a board bent as a 3000R monitor
    synth(work / "curved", 16, seed=4, bend_radius=3000.0, log=quiet)
    c = run_calibrate(work / "curved", work / "curved_intrinsics.json", work / "curved" / "truth.json",
                      BOARD["square"], log=quiet)
    sag = (BOARD["squares"][0] * BOARD["square"]) ** 2 / (8 * 3000.0)
    s, fx = c["shape"], c["error"]["fx_rel"]
    flat_fx = (1 + fx) / (1 + s["focal_shift"]) - 1
    expect("board_not_flat" in c["issues"] and abs(s["sag_mm"] - sag) < 0.3 and abs(fx) < MAX_FOCAL_ERROR
           and s["rms_flat_px"] > 3 * c["rms_px"] and abs(flat_fx) > 5 * MAX_FOCAL_ERROR,
           f"a board bent as a 3000R monitor: not flat, bends {s['sag_mm']:.2f} mm (truly {sag:.2f}), error "
           f"{s['rms_flat_px']:.3f} -> {c['rms_px']:.3f} px; focal length off the truth {100 * fx:+.3f} % with the "
           f"shape, {100 * flat_fx:+.2f} % without; issues {c['issues']}")
    # a board warped by 0.1 mm only, under FLAT_SAG_MM: its shape is still used, the focal length kept right (a flat
    # board's is not)
    W = BOARD["squares"][0] * BOARD["square"]
    synth(work / "warped", 16, seed=5, bend_radius=W ** 2 / (8 * 0.1), log=quiet)
    c = run_calibrate(work / "warped", work / "warped_intrinsics.json", work / "warped" / "truth.json",
                      BOARD["square"], log=quiet)
    s, fx = c["shape"], c["error"]["fx_rel"]
    expect(s["sag_mm"] < FLAT_SAG_MM and s["used"] and not sh["used"] and abs(fx) < MAX_FOCAL_ERROR
           and abs(s["focal_shift"]) > abs(fx),
           f"a board warped by 0.1 mm: its shape used (F {s['f_test']}, a flat board's {sh['f_test']}), bends "
           f"{s['sag_mm']:.3f} mm, focal length off the truth {100 * fx:+.3f} % (shape shift "
           f"{100 * s['focal_shift']:+.3f} %)")
    # the session check of that calibration, on new photos of the same bent board: the board with its stored shape
    synth(work / "curved_check", 10, seed=6, bend_radius=3000.0, prefix="check", log=quiet)
    ok_c, rep = run_check(work / "curved_check", work / "curved_intrinsics.json", BOARD["square"], log=quiet)
    flat_ref = json.loads((work / "curved_intrinsics.json").read_text(encoding="utf-8"))
    flat_ref["shape"] = {**flat_ref["shape"], "flat": True, "used": False}
    write_json_atomic(work / "curved_as_flat.json", flat_ref)
    ok_f, rep_f = run_check(work / "curved_check", work / "curved_as_flat.json", BOARD["square"], log=quiet)
    flat_ref["shape"]["flat"] = False
    flat_ref["board"] = {**flat_ref["board"], "squares": [12, 9]}
    write_json_atomic(work / "curved_other.json", flat_ref)
    other = run_check(work / "curved_check", work / "curved_other.json", BOARD["square"], log=quiet)
    expect(ok_c and rep["shape"] == "stored" and rep["rms_stored_px"] < MAX_RMS and rep_f["shape"] == "flat"
           and rep_f["rms_stored_px"] > 2 * rep["rms_stored_px"] and other[1]["shape"] == "other_board",
           f"the session check of a bent board's calibration takes the board with its stored shape: passes, "
           f"{rep['rms_stored_px']:.3f} px (as a plane {rep_f['rms_stored_px']:.3f} px, "
           f"{'passes' if ok_f else 'fails'}); a board of other squares stays a plane")
    # the optical centre moving between the photos, as a phone held in the hand
    synth(work / "held", 16, seed=5, centre_sd=15.0, log=quiet)
    h = run_calibrate(work / "held", None, None, BOARD["square"], log=quiet)
    held = json.loads((work / "held" / "truth.json").read_text(encoding="utf-8"))
    true_spread = _spread([v["centre_shift_px"] for v in held["views"]])
    got = h["centre"]["spread_px"] if h["centre"] else math.nan
    expect("phone_not_still" in h["issues"] and "board_not_flat" not in h["issues"]
           and abs(got - true_spread) < 0.35 * true_spread,
           f"the optical centre moving between the photos: the phone moved, the centre spreads {got:.1f} px (truly "
           f"{true_spread:.1f}); the board still flat; issues {h['issues']}")
    # the corners of the image
    w, hh = SYNTH_SIZE
    middle = np.array([[0.5 * w, 0.5 * hh], [0.3 * w, 0.4 * hh], [0.7 * w, 0.6 * hh]])
    c0 = coverage([middle], SYNTH_SIZE)
    c1 = coverage([middle, np.array([[w - 30.0, 25.0]])], SYNTH_SIZE)
    expect(c0["missing_corners"] == list(CORNER_NAMES) and "top_right" in c0["missing_cells"]
           and c1["missing_corners"] == ["top_left", "bottom_right", "bottom_left"] and c1["corners"]["top_right"] == 1
           and "top_right" not in c1["missing_cells"],
           f"corners of the image never reached are named: {c0['missing_corners']}; with one point in the top right "
           f"corner {c1['missing_corners']}")
    one = check_photo(paths[0], board)
    expect(one["ok"] and one["coverage"] is not None and sum(map(sum, one["coverage"]["cells"])) == one["corners"]
           and one["settings"] == {"known": False, "problems": []},
           f"a photo's check says where its corners fell (reach {one['coverage'] and one['coverage']['reach']}); "
           f"without EXIF its settings are not judged")
    # the camera's settings, from the EXIF
    ex = work / "exif"
    ex.mkdir()
    _with_exif(paths[0], ex / "auto.jpg", 2, 0, 125, 0.02, {"Hdr": "on", "AIScene": 4, "zoomMultiple": 2})
    _with_exif(paths[0], ex / "pro.jpg", 1, 1, 100, 0.02, {"Hdr": "off", "AIScene": 0, "zoomMultiple": 1})
    _with_exif(paths[0], ex / "pro_iso.jpg", 1, 1, 200, 0.02, {"Hdr": "off", "AIScene": 0, "zoomMultiple": 1})
    a = check_photo(ex / "auto.jpg", board)
    p = check_photo(ex / "pro.jpg", board, first=ex / "pro.jpg")
    q = check_photo(ex / "pro_iso.jpg", board, first=photo_settings(ex / "pro.jpg"))
    expect(a["why"] == "settings" and not a["ok"] and a["corners"] == one["corners"]
           and set(a["settings"]["problems"]) == {"exposure_auto", "white_balance_auto", "ai_scene", "hdr", "zoom"}
           and p["ok"] and p["settings"]["problems"] == [] and q["why"] == "settings"
           and q["settings"]["problems"] == ["iso_changed"],
           f"camera settings from the EXIF: automatic mode refused ({a['settings']['problems']}), the manual one "
           f"taken, another ISO than the first photo refused ({q['settings']['problems']})")
    if real:
        real = Path(real)
        square, squares, marker = folder_board(real) or (BOARD["square"], None, None)
        r = run_calibrate(real, None, None, square, squares, marker, log=quiet)
        rs, rc = r["shape"], r["centre"] or {}
        first = check_photo(find_images(real)[0], make_board(square, squares, marker))
        expect({"board_not_flat", "phone_not_still", "camera_settings"} <= set(r["issues"])
               and first["why"] == "settings",
               f"{real.name}: {r['n_views']} photos, error {rs['rms_flat_px']:.3f} px flat, {r['rms_px']:.3f} px with "
               f"the shape (bends {rs['sag_mm']:.2f} mm, focal length {100 * rs['focal_shift']:+.2f} %), "
               f"{rc.get('rms_px', math.nan):.3f} px with the centre free (it spreads {rc.get('spread_px')} px); "
               f"fx {r['K'][0][0]:.1f} px; issues {r['issues']}; its first photo refused for "
               f"{first['settings']['problems']}")
    return not fails


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
    a.add_argument("--work", type=Path, default=WORK / "selftest")
    a.add_argument("--real", type=Path, help="also a folder of photos of a curved board taken by hand in the camera's "
                                             "automatic mode: each of these must be said")
    args = ap.parse_args()

    try:
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
            sys.exit(0 if run_test(args.work, args.real) else 1)
    except CalibrationError as e:
        sys.exit(f"calibrate_charuco: {e.code}: {e}")


if __name__ == "__main__":
    main()
