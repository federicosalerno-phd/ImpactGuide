"""Live AR on the phone stream: board pose, face overlay and status light.

Three threads work on the frames of the stream (live.py), always on the newest one, never on a queue:
  track     every frame: the board's ChArUco corners followed from the previous frame by optical flow and put back on
            the chessboard's saddle points, the pose from them (board_tracker.py), then the temporal filter
            (pose_filter.py: One Euro filter on SE(3) in two channels that follow the noise the tracker expects, so a
            precise pose is drawn as it comes and stays glued to the board; outlier gate, glide, hold and fade). A
            frame where the board is lost shortly after it was followed is searched at once where the pose carried on
            puts it. A few milliseconds, so it keeps the rate of the stream.
  detect    now and then (TRACK["detect_every_s"], or at once when the tracking weakens or the board is lost): the
            full ChArUco detection, in a box around the board or in the whole frame; its corners re-anchor the
            tracking a few frames later.
  compose   the newest frame tracked: the face (and the mandible) drawn at its filtered pose with gl_overlay.py, a veil
            at the opacity asked for with nothing on its edge (anti-aliased silhouette, smooth shading), board axes
            and corners on request, the picture turned by quarter turns to stand the way the phone is held (the pose
            stays in the sensor's frame), JPEG. It owns the OpenGL context.
With hide_board (the board on the same monitor as the picture of the live view, as ImpactGuide shows it), the
board itself is covered in the picture shown, so the phone never films a second board inside the first: on a frame
with its own pose the board's area (with half a square around it) is filled with a dark flat colour, on a frame
without one the whole picture is shown blurred and dimmed (nothing in it can be read as a marker).
So each frame is drawn at the pose found on that same frame, at the rate of the stream; when the computer falls
behind, frames are skipped, never queued. The overlay comes late by the stream's latency (about 150 ms on the cable)
plus the processing, but on its own frame. The page's calls (status, configure, next_jpeg) only read or set values;
only a change of the board starts the tracking again (not the layers, the opacity or the rotation).

Frames: C camera (OpenCV), M board (cv2.aruco.CharucoBoard: origin at the outer corner of the first square, x along
the printed columns, y along the rows, z into the board), B head (bench scene, mm).

Boards
    patient   the 5 x 5 board of the marker holder; squares and markers from marker_pose.BOARD
    sheet     the calibration board of calibrate_charuco.py: shown on a monitor (screen_board.py: as many squares as
              the monitor holds, square from its pixel pitch) or printed on A4 (11 x 8, square as measured with the
              caliper); either is enough to try the page without the bench. squares and marker_mm give the layout of
              the one on a monitor; without them it is the A4 one
A pose is used with at least board_tracker.TRACK["min_corners"] corners (8 to start on the calibration board; with
fewer than 10 they must spread over the board and fit tighter) and a reprojection RMS under the limit: 1 px with
intrinsics calibrated for the stream, 3 px with the nominal ones, which leave out the lens distortion. The small
patient board works from about 20 to 64 cm (15/10/2026, board shown at its true size on the laptop screen; at
1920 x 1440, 20 px per square at the far end).

Intrinsics of the streaming mode: 03_data/processed/calibration/intrinsics_stream.json, in the format
calibrate_charuco.py writes for intrinsics_photo.json, calibrated on frames of the stream at the working distance;
scaled if the stream has another size of the same shape (a calibration of another aspect ratio, or a file that cannot
be read, is not used: the nominal camera, with "error": "intrinsics_bad" in its source). Until it exists: nominal Poco
X7 Pro, 3070 px at the 3072 px height of the 4:3 sensor field (as synth_twin.py), so 1439 px at 1920 x 1440, principal
point at the centre, no distortion. The synthetic stand-in renders without distortion: its frames are tracked
without it whatever the calibration says.

Registration T_M_B: a JSON file holding a 4 x 4 "T_M_B" in mm (what ground_truth.py measure writes). Until there is
one, a demo registration at a fixed offset from the board, which the page always labels as demo:
    patient   the registration of the bench design (the scene, meshes/occluders.npz of synth_twin.py)
    sheet     the face lying on the sheet, centred, looking up

Models in B (mm): the decimated face (synth_twin.py meshes) and the mandible of the bench scene
(build_targets.py), both under C:\\work\\marker-impact and written again by those scripts; for the live view each is
decimated again to LIVE_TRIANGLES (quadric edge collapse) and kept in LIVE_MODELS.

States (status()["stream_state"], STATES): idle (no stream: never started, or stopped), starting, running, failed
(the stream gave up, or the last start failed: status()["error"] says why with a code of ERRORS and "cause" the
link's; start() opens a new stream, no stop() needed), closed (after shutdown()). running is True only for a live
stream. A frame that fails (tracking, drawing, one model's renderer) is logged and counted (status()["frame_errors"],
["last_frame_error"]): the picture keeps its last good frame and the threads go on with the next one; "error" says
frames_failing only when no frame has come out for FAILING_S, and clears with the next good one. A model whose
renderer cannot be made is left out of the picture (status()["models"][name] False); the rest goes on.

For the app:
    ar = LiveAR(board="sheet", square_mm=24.0)       # one per process is enough; each compose thread makes and
                                                     # releases its own OpenGL context
                                                     # (squares=(14, 7), marker_mm=... for the board on a monitor;
                                                     # prepare=False: nothing loaded, quick on a GUI thread)
    ar.prepare()                                     # the models loaded (seconds the first time): on a worker thread;
                                                     # start() does it when it was not done
    ar.start()                                       # closes the camera app, opens the stream (a few seconds); after
                                                     # the stream died, a new one
    jpeg, n = ar.next_jpeg(after=n)                  # newest composed frame
    jpeg, n, t = ar.next_frame(after=n)              # the same with the time it was composed (phone_view.py)
    ar.status()                                      # light, corners, RMS, fps, latency, timings, sources; the
                                                     # tremble of the face drawn and as found (tremble_px,
                                                     # tremble_raw_px: px, the last TREMBLE_S)
    ar.configure(board=..., square_mm=..., squares=..., marker_mm=..., face=..., mandible=..., axes=...,
                 opacity=..., view_rotation=..., hide_board=...)
                                                     # opacity 0-1 of the overlay (default OPACITY);
                                                     # view_rotation None (the phone's way up) or 0/90/180/270;
                                                     # hide_board: the board covered in the picture shown
    ar.still()                                       # the newest frame at full size (BGR), for a calibration
    ar.set_registration(path or None)
    ar.set_tracking(hold_s=..., ...)                 # TRACKING and board_tracker.TRACK overrides
    ar.stop(); ar.shutdown()                         # stop: the stream closed (a start under way cancelled)
    MjpegServer(ar).url                              # the frames for an <img>, on 127.0.0.1
    phone_view.ViewServer(ar)                        # the frames for the phone's screen (cable, Wi-Fi, network)

    python live_ar.py view  [--board patient|sheet] [--square MM [--squares NXxNY --marker MM]] [--registration FILE]
                            [--mandible]
                            [--synthetic | --replay VIDEO_OR_FOLDER] [--intrinsics FILE]
    python live_ar.py bench [--synthetic | --replay CLIP.h264] [--seconds S] [--still S] [--segments FILE]
                            [--reference]
                            rate, time per stage, CPU, tremble of the overlay (noise from second differences),
                            jumps per minute, frames without a pose, blinks (and, on the synthetic stream or a clip with
                            a truth file, the error of the overlay; with --reference against a pose found offline on
                            every frame of the clip and smoothed without lag), for the whole run and for each segment
    python live_ar.py clip OUT [--seconds S] [--noise SIGMA]
                            the synthetic session as a clip of the phone (raw H.264 with times and truth), for
                            measuring versions of the live AR on the same frames (bench --replay OUT.h264)
    python live_ar.py test  pose filter; board tracker (board_tracker.py test); synthetic stream: pose against the
                            truth, overlay, light, video rate, a hand over the board, MJPEG; frames of the digital
                            twin with the true registration; the board beside its own live view on one screen
                            (echo); start, stop and a stream that dies; frames that fail; bad settings and
                            calibration files; exit 1 on failure (all on the nominal camera, whatever calibration is
                            on this computer)
    python live_ar.py echo [DIR]  only the echo check: the pose with hide_board, and without it (pictures into DIR)
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import secrets
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import acquisition
import arrowhead
import board_tracker
import calibrate_charuco
import live
import marker_pose
import pose_filter

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STREAM_INTRINSICS = ROOT / "03_data" / "processed" / "calibration" / "intrinsics_stream.json"
WORK = Path(r"C:\work\marker-impact")
FACE_MESH = WORK / "synth" / "meshes" / "face_phantom_r20k.npz"
DESIGN = WORK / "synth" / "meshes" / "occluders.npz"          # T_M_B of the bench scene
MANDIBLE_MESH = WORK / "targets" / "mandible.npz"
TWIN = WORK / "synth" / "sessions"                              # frames of the digital twin, for the test
LIVE_MODELS = WORK / "live_models"                              # the models decimated for the live view
LIVE_REFERENCES = WORK / "live_references"      # reference poses of clips kept inside the project (read-only there)
LIVE_TRIANGLES = {"face": 4000, "mandible": 6000}
CAMERA_APP = acquisition.CAMERA_APP     # the phone's camera app holds the camera while it is open
NOMINAL_F = 3070.0 / 3072.0             # focal length over the height of the 4:3 sensor field (synth_twin.NOMINAL)
# camera to decoded frame on each link, median of the screen flashes (live.py latency): cable 149 ms on 15/10/2026
# with live.CODEC_OPTIONS (169 ms without, 195 ms in the first test); Wi-Fi not measured yet (None: the page says so
# and counts only the processing)
STREAM_LATENCY_MS = {"usb": 149.0, "wifi": None}
MAX_RMS = {"calibrated": 1.0, "nominal": 3.0}
ASPECT_TOLERANCE = 0.01                 # a calibration of a frame shape this far off (share) is not used
DISPLAY_WIDTH = 1280
JPEG_QUALITY = 80
# the look of the overlay: each model a veil in its colour of the palette with a light shading, and nothing on its
# edge (no outline, no rim): the silhouette is the veil's own anti-aliased border, so everything follows the opacity
# together. The veil's opacity is the setting `opacity` (0-1) times fill (capped at 1): barely there at the lowest
# setting, a soft even veil at the default OPACITY, a nearly opaque face at 1. The shading is a light at the camera
# (both sides lit), smooth across the triangles: the colour times 1 - shade * (1 - cos) ** SHADE_GAMMA, cos of the
# angle between the surface and the line of sight (1 face on, 0 edge on); a gamma under 1 spreads it over the
# surface instead of darkening a band along the silhouette.
OPACITY = 0.3
STYLE = {"face": {"fill": 0.95, "shade": 0.45}, "mandible": {"fill": 0.95, "shade": 0.45}}
SHADE_GAMMA = 0.6
SOFT = {"feather_px": 0.8, "smooth_px": 2.0}   # NumPy renderer only: its hard mask feathered and its flat shading
                                               # blurred (display px)
COLORS = {"face": "#F5C542", "mandible": "#E6E6EC", "corners": "#58B36B", "reprojected": "#E6E6EC",
          "hidden": "#16161A"}
# hide_board: the area around the board at the frame's own pose (margin_squares); a frame without its own pose shortly
# after one (at most held_s, a missed frame or two) at the pose drawn, with the wider held_margin_squares; any other
# frame blurred (shrink) and dimmed whole, so the board in the picture is never there to be found
HIDE = {"margin_squares": 0.5, "held_s": 0.2, "held_margin_squares": 1.0, "shrink": 12, "dim": 0.5}
# the tips of the board's axes (display px): the one arrowhead of every drawing (arrowhead.py), length_px long, at most
# max_share of the axis, none on an axis shorter than min_px over max_share
ARROW = {"length_px": 13.0, "max_share": 0.35, "min_px": 5.0, "line_px": 2}
BOARDS = ("patient", "sheet")
BOARD_KEYS = ("board", "square_mm", "squares", "marker_mm")    # the settings that make the board
TRACKING = {
    # the temporal filter of the pose: pose_filter.DEFAULTS (its one source: smoothing, gate, glide, hold and fade);
    # the tracking itself is board_tracker.TRACK (set_tracking takes both)
    **pose_filter.DEFAULTS,
    "strong_corners": 0.5,    # a jump past the filter's gate is believed with at least this share of the
    "strong_rms": 0.6,        #   board's corners and an RMS under this share of the limit, or when the next pose agrees
    "wait_sure": True,        # a new acquisition is drawn once its mirrored pose is ruled out (board_tracker.py)
    "roi_s": 0.4,             # after the board was lost the detection looks where it was for this long
    "recover_s": 0.8,         # and for this long each frame without the board is searched at once where the board
                              #   should be (the pose carried on): the overlay picks it up again on the next frame,
    "recover_px": 250000,     #   when that box, at the scale of the search, is at most this many px (a few ms)
    "views": False,           # separate views, not a video (tests): every frame detected on its own, nothing carried
}
# the stream's state (status()["stream_state"]) and why it failed (status()["error"])
IDLE, STARTING, RUNNING, FAILED, CLOSED = STATES = ("idle", "starting", "running", "failed", "closed")
ERRORS = live.CODES + ("start_failed", "stream_stopped", "replay_unreadable", "frames_failing", "bad_board",
                       "bad_value", "bad_registration", "closed")
FAILING_S = 1.5             # frames failing and none composed for this long: status()["error"] says frames_failing
FAILING_COUNT = 5           #   (and at least this many failures since the last frame composed)
LOG_EVERY_S = 5.0           # a failing frame is logged at most this often (with the count)
FAIL_PAUSE_S = 0.2          # a stream object that raises: the track thread waits this long before asking again
RENDER_FAILS = 3            # a model's renderer failing this many frames in a row: OpenGL gives way to NumPy, then out
POLL_S = 0.2                # the threads wake at least this often (shutdown)
WAIT_FRAME_S = 0.1          # the track thread's wait for the next frame
START_WAIT_S = 30.0         # a start() while another is under way waits for it this long
JOIN_S = 2.0                # shutdown(), close(): each thread is waited for this long
TREMBLE_S = 3.0             # status()["tremble_px"]: the shake of the drawn face over this last stretch of frames,
TREMBLE_POINTS = 40         #   on this many points of it
MJPEG_CLIENTS = 8           # MJPEG clients served at once
MJPEG_SOCKET_S = 10.0       # a client that stops reading is dropped after this


class LiveARError(RuntimeError):
    """A failure of the live AR meant for the user.

    Args:
        msg: what happened, for the log.
        code: why, from ERRORS.
    """

    def __init__(self, msg, code="start_failed"):
        super().__init__(msg)
        self.code = code


# ── boards, intrinsics, registration, models ─────────────────────────────────
def make_board(name, square_mm=None, squares=None, marker_mm=None):
    """(cv2 CharucoBoard, description) of the patient board or of the calibration board (sheet): the A4 one, or with
    squares and marker_mm the one on a monitor (screen_board.py)."""
    if name == "patient":
        b = marker_pose.BOARD
        board = marker_pose.make_board(b)
        n = b["squares"]
        spec = {"name": name, "squares": [n, n], "square_mm": float(b["square_mm"]), "marker_mm": float(b["marker_mm"]),
                "dictionary": b["dictionary"]}
    elif name == "sheet":
        sq = float(square_mm or calibrate_charuco.BOARD["square"])
        board = calibrate_charuco.make_board(sq, squares, marker_mm)
        nx, ny = board.getChessboardSize()
        spec = {"name": name, "squares": [nx, ny], "square_mm": sq, "marker_mm": float(board.getMarkerLength()),
                "dictionary": calibrate_charuco.BOARD["dictionary"]}
    else:
        raise LiveARError(f"unknown board {name}", "bad_value")
    spec["corners"] = int(len(board.getChessboardCorners()))
    return board, spec


def check_board(name, square_mm=None, squares=None, marker_mm=None):
    """The board of these settings, if it can be made (markers smaller than the squares, at least 2 x 2 squares, a
    dictionary large enough); raises LiveARError ("bad_board") otherwise."""
    for v in (square_mm, marker_mm):
        if v is not None and not (np.isfinite(v) and v > 0):
            raise LiveARError(f"board {name}: a size of {v} mm", "bad_board")
    try:
        board, spec = make_board(name, square_mm, squares, marker_mm)
    except (cv2.error, SystemError, ValueError, TypeError) as e:     # SystemError: OpenCV's assertion, wrapped
        raise LiveARError(f"board {name} {squares} {square_mm} {marker_mm}: {e}", "bad_board") from e
    if not 0 < spec["marker_mm"] < spec["square_mm"] or min(spec["squares"]) < 2:
        raise LiveARError(f"board {name}: marker {spec['marker_mm']} mm, square {spec['square_mm']} mm, squares "
                          f"{spec['squares']}", "bad_board")
    return board, spec


def scale_K(K, sx, sy):
    """K for an image resized by sx, sy (pixel centres at integer coordinates)."""
    K = np.array(K, np.float64)
    return np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5], [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5],
                     [0, 0, 1.0]])


def stream_intrinsics(size, path=STREAM_INTRINSICS):
    """K, dist and their source ({"source": calibrated|nominal, "path", "f"}) for a stream of size (w, h). A stream
    turned upright (taller than wide) against a calibration made across gets the calibration turned a quarter: focal
    lengths and principal point swapped, tangential terms swapped (the sense of the turn is not known: fine for a
    principal point near the centre, and labelled "turned"). A file that cannot be read, or a calibration of another
    frame shape (more than ASPECT_TOLERANCE off), is not used: the nominal camera, its source saying
    "error": "intrinsics_bad" and why ("detail"). Never raises."""
    w, h = size
    error = None
    if path and Path(path).is_file():
        try:
            return _calibrated(w, h, Path(path))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
            error = f"{type(e).__name__}: {e}"
            _warn_once(Path(path), error)
    f = NOMINAL_F * min(w, h)       # the sensor's 4:3 field, across or upright
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    src = {"source": "nominal", "path": "", "date": "", "rms_px": None}
    if error:
        src.update({"error": "intrinsics_bad", "detail": error[:200], "path": str(path)})
    src.update({"f": round(float((K[0, 0] + K[1, 1]) / 2), 1), "size": [int(w), int(h)]})
    return K, np.zeros(5), src


def _calibrated(w, h, path):
    """K, dist, source of a calibration file for frames of w x h (stream_intrinsics); raises when it cannot be used."""
    d = json.loads(path.read_text(encoding="utf-8"))
    cw, ch = (int(v) for v in d["image_size"])
    K0 = np.array(d["K"], np.float64)
    dist = np.asarray(d.get("dist", np.zeros(5)), np.float64).ravel()
    if cw <= 0 or ch <= 0 or K0.shape != (3, 3) or not np.all(np.isfinite(K0)) or K0[0, 0] <= 0 or K0[1, 1] <= 0 \
            or not np.all(np.isfinite(dist)):
        raise ValueError(f"image_size {cw} x {ch}, K {K0.tolist()}, dist {dist.tolist()}")
    turned = (w > h) != (cw > ch)
    if turned:
        K0 = np.array([[K0[1, 1], 0, K0[1, 2]], [0, K0[0, 0], K0[0, 2]], [0, 0, 1.0]])
        cw, ch = ch, cw
        if len(dist) >= 4:
            dist = dist.copy()
            dist[[2, 3]] = dist[[3, 2]]
    if abs((w / h) / (cw / ch) - 1.0) > ASPECT_TOLERANCE:
        raise ValueError(f"calibrated at {cw} x {ch}, the stream is {w} x {h}: another shape")
    K = scale_K(K0, w / cw, h / ch)
    src = {"source": "calibrated", "path": str(path), "date": d.get("date", ""), "rms_px": d.get("rms_px"),
           "turned": bool(turned), "f": round(float((K[0, 0] + K[1, 1]) / 2), 1), "size": [int(w), int(h)]}
    return K, dist, src


_WARNED = set()     # calibration files already reported as not usable (path, modification time)


def _warn_once(path, error):
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        key = (str(path), None)
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning("%s cannot be used (%s): the nominal camera instead", path.name, error)


def load_registration(path):
    """T_M_B (4 x 4, mm, in the bench frame B) from a JSON file: {"T_M_B": [[...]]} as ground_truth.py writes it, or
    the bare matrix. Raises LiveARError (bad_registration) for anything else: a file that cannot be read, a result of
    ground_truth.py written before its two frames had their names (its "T_M_B" is in the mandible frame, not in B), a
    result whose head frame was unknown (T_M_B null), a matrix that is not a rigid transform."""
    name = Path(path).name
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LiveARError(f"{name}: not readable as JSON ({e})", "bad_registration") from e
    if isinstance(d, dict) and "G" in d and "T_M_mandible" not in d:
        raise LiveARError(f"{name}: an older ground_truth result, its T_M_B in the mandible frame: measure again",
                          "bad_registration")
    if isinstance(d, dict) and d.get("T_M_B") is None:
        raise LiveARError(f"{name}: no T_M_B (the head frame B was unknown when it was measured)", "bad_registration")
    try:
        T = np.asarray(d["T_M_B"] if isinstance(d, dict) else d, np.float64)
    except (TypeError, ValueError) as e:
        raise LiveARError(f"{name}: T_M_B is not a matrix of numbers", "bad_registration") from e
    if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], [0, 0, 0, 1]) \
            or not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-4) or np.linalg.det(T[:3, :3]) <= 0:
        raise LiveARError(f"{name}: T_M_B is not a 4 x 4 rigid transform", "bad_registration")
    return T


def demo_registration(spec, face_vertices):
    """(T_M_B, kind) of the demo: the bench design for the patient board, the face lying on the sheet otherwise.

    The face of the bench scene looks along -y of B with the top of the head along +z (the phone films it from -y);
    on the sheet it looks along -z of M (away from the sheet, towards the camera), the top of the head towards the top of
    the page (-y of M), centred on the board, its back on the sheet."""
    if spec["name"] == "patient" and DESIGN.is_file():
        return np.load(DESIGN)["T_M_B"].astype(np.float64), "design"
    R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    v = np.asarray(face_vertices, np.float64) @ R.T
    lo, hi = v.min(0), v.max(0)
    nx, ny = spec["squares"]
    sq = spec["square_mm"]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [nx * sq / 2 - (lo[0] + hi[0]) / 2, ny * sq / 2 - (lo[1] + hi[1]) / 2, -hi[2]]
    return T, "lying"


def consistent(faces, n_vertices):
    """The faces without those on an edge that two triangles run the same way (a flipped sliver): the renderers need a
    consistently wound mesh. Returns (faces, number left out)."""
    f = np.asarray(faces, np.int64)
    dropped = 0
    for _ in range(10):       # leaving faces out can turn an edge of three triangles into a bad edge of two
        half = f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        key = half.min(1) * n_vertices + half.max(1)
        order = np.argsort(key, kind="stable")
        k, fwd = key[order], (half[:, 0] < half[:, 1])[order]
        face = np.repeat(np.arange(len(f)), 3)[order]
        _, start, count = np.unique(k, return_index=True, return_counts=True)
        two = start[count == 2]
        bad = two[fwd[two] == fwd[two + 1]]
        if not len(bad):
            break
        drop = np.unique(np.concatenate([face[bad], face[bad + 1]]))
        f = np.delete(f, drop, axis=0)
        dropped += len(drop)
    return f, dropped


def decimated(name, v, f, source, log=None):
    """The model with about LIVE_TRIANGLES[name] triangles (quadric edge collapse, pymeshlab), kept in LIVE_MODELS under
    a name that carries the size and time of the source file, so a new source makes a new one. The full model when
    pymeshlab is missing."""
    target = LIVE_TRIANGLES.get(name, 0)
    if not target or len(f) <= target:
        return v, f
    st = Path(source).stat()
    path = LIVE_MODELS / f"{name}_{target}_{st.st_size}_{int(st.st_mtime)}.npz"
    if path.is_file():
        z = np.load(path)
        return z["v"].astype(np.float64), z["f"].astype(np.int64)
    try:
        import pymeshlab
    except ImportError:
        return v, f
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(v, np.float64), face_matrix=np.asarray(f, np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), preserveboundary=True, preservenormal=True,
                                                preservetopology=True, optimalplacement=True, planarquadric=True,
                                                autoclean=True)
    m = ms.current_mesh()
    v2 = m.vertex_matrix().astype(np.float64)
    f2 = consistent(m.face_matrix().astype(np.int64), len(v2))[0]
    LIVE_MODELS.mkdir(parents=True, exist_ok=True)
    for old in LIVE_MODELS.glob(f"{name}_*.npz"):       # one per model: the one of an older source goes
        old.unlink(missing_ok=True)
    np.savez_compressed(path, v=v2, f=f2)
    if log:
        log(f"{name}: {len(f)} triangles decimated to {len(f2)} for the live view ({path.name})")
    return v2, f2


def load_models(live_view=True, log=None):
    """{name: (vertices in B, faces)} of the models that are on this computer; for the live view decimated."""
    out, src = {}, {}
    if FACE_MESH.is_file():
        z = np.load(FACE_MESH)
        out["face"] = (z["vertices"].astype(np.float64), z["faces"].astype(np.int64))
        src["face"] = FACE_MESH
    if MANDIBLE_MESH.is_file():
        z = np.load(MANDIBLE_MESH)
        T = z["T_B_mandible"]
        out["mandible"] = (z["v"].astype(np.float64) @ T[:3, :3].T + T[:3, 3], z["f"].astype(np.int64))
        src["mandible"] = MANDIBLE_MESH
    for name, (v, f) in out.items():
        f = consistent(f, len(v))[0]
        out[name] = decimated(name, v, f, src[name], log) if live_view else (v, f)
    return out


def bgr(colour):
    """'#RRGGBB' (or with alpha) as an OpenCV BGR tuple."""
    c = colour.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


def close_camera_app(serial=None):
    """Stops the phone's camera app, which otherwise holds the camera and starves the stream. True if it was asked."""
    try:
        live.adb("shell", "am", "force-stop", CAMERA_APP, serial=live.pick_serial(serial), timeout=15)
        return True
    except Exception:   # noqa: BLE001  no phone: the stream says why when it opens
        return False


# ── stand-ins for the phone ──────────────────────────────────────────────────
PAUSE_STEP_S = 0.05     # a stand-in's pause looks at its stop event at least this often


def _pause(stop, seconds):
    """Waits `seconds` unless `stop` (an Event) is set first: with time.sleep, whose timer is fine (about 1 ms) where an
    Event's wait on Windows goes by the 15.6 ms system tick and would bunch the frames of a stand-in."""
    end = time.perf_counter() + seconds
    while not stop.is_set():
        left = end - time.perf_counter()
        if left <= 0:
            return
        time.sleep(min(left, PAUSE_STEP_S))


class _Frames:
    """What the stand-ins share with live.PhoneStream: the newest frame with its time and counter, the rate, alive,
    error and code, close(). Their thread starts with start() (or wait_first()), not in the constructor."""

    def _init_frames(self):
        self.error, self.code, self.cause, self.restarts = None, "", "", 0
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._frame, self._t, self._n = None, 0.0, 0
        self._item, self._session = None, 1
        self._times = deque(maxlen=120)
        self._decode_ms = deque(maxlen=120)
        self._stop = threading.Event()
        self.thread = None

    def start(self):
        """Starts the thread that makes the frames (once)."""
        with self._lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True, name=self.THREAD)
                self.thread.start()
        return self

    def _put(self, img, decode_ms=None, t_capture=None):
        t = time.perf_counter()
        with self._new:
            self._frame, self._t, self._n = img, t, self._n + 1
            self._item = live.Frame(self._n, t, t_capture, bgr=img, session=self._session)
            self._times.append(t)
            if decode_ms is not None:
                self._decode_ms.append(decode_ms)
            self._new.notify_all()
            return self._n

    def latest(self):
        with self._lock:
            return self._frame, self._t, self._n

    def wait_frame(self, after=0, timeout=0.1):
        with self._new:
            self._new.wait_for(lambda: self._n > after or self._stop.is_set(), timeout=timeout)
            if self._n <= after:
                return None, 0.0, after
            return self._frame, self._t, self._n

    def latest_item(self):
        with self._lock:
            return self._item

    def wait_item(self, after=0, timeout=0.1):
        """The newest frame as a live.Frame once its counter is past `after` (None on timeout)."""
        with self._new:
            self._new.wait_for(lambda: self._n > after or self._stop.is_set(), timeout=timeout)
            if self._n <= after:
                return None
            return self._item

    def stats(self):
        with self._lock:
            t = np.array(self._times)
            n = self._n
            dec = float(np.mean(self._decode_ms)) if self._decode_ms else float("nan")
        if len(t) < 2:
            return {"fps": 0.0, "interval_ms": float("nan"), "interval_p95_ms": float("nan"), "decode_ms": dec,
                    "frames": n}
        dt = np.diff(t) * 1e3
        return {"fps": float((len(t) - 1) / (t[-1] - t[0])), "interval_ms": float(dt.mean()),
                "interval_p95_ms": float(np.percentile(dt, 95)), "decode_ms": dec, "frames": n}

    @property
    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def close(self):
        """Stops the thread and waits for it (JOIN_S)."""
        self._stop.set()
        with self._new:
            self._new.notify_all()
        th = self.thread
        if th is not None and th is not threading.current_thread():
            th.join(JOIN_S)

    def _ended(self, error, code):
        """The thread ends on its own (cannot read): why, for alive, error and code."""
        with self._new:
            self.error, self.code = error, code
            self._new.notify_all()

    def wait_first(self, timeout=5.0):
        """Starts the thread and returns once the first frame is in (or it ended, or the timeout passed)."""
        self.start()
        deadline = time.perf_counter() + timeout
        with self._new:
            while self._frame is None and not self._stop.is_set() and self.error is None:
                left = deadline - time.perf_counter()
                if left <= 0 or not self.thread.is_alive():
                    break
                self._new.wait(min(left, 0.05))
        return self


def bench_script(spec, period=24.0):
    """A hand-held session for the synthetic stream, repeated every `period` s: still 0-6 s (phone resting), moving
    slowly 6-12 s, fast 12-18 s, still again 18-24 s; a hand (a disc in the board plane, mm) over half the board for
    0.1 s at 2 s, over all of it for 0.4 s at 4 s (shorter than the hold), over 60 % at 8 s, over all of it for 1.5 s at
    14 s while moving (longer than hold and fade) and for 0.5 s at 20 s. Returns script(t) -> (phase, hands): the phase
    drives SyntheticStream.pose, the hands are (x, y, radius) in board mm."""
    nx, ny = spec["squares"]
    sq = spec["square_mm"]
    W, H = nx * sq, ny * sq
    centre = np.array([W / 2, H / 2])
    big = 0.75 * np.hypot(W, H)
    covers = [(2.0, 2.1, (W * 0.25, H / 2, 0.42 * max(W, H) / 2)), (4.0, 4.4, (*centre, big)),
              (8.0, 8.25, (W * 0.3, H / 2, 0.55 * max(W, H) / 2)), (14.0, 15.5, (*centre, big)),
              (20.0, 20.5, (*centre, big))]

    def script(t):
        u = t % period
        phase = (0.0 if u < 6 else 0.5 * (u - 6) if u < 12 else 3.0 + 2.5 * (u - 12) if u < 18 else 18.0)
        return phase, [h for a, b, h in covers if a <= u < b]
    return script


class SyntheticStream(_Frames):
    """Stand-in for live.PhoneStream: the board on a grey table, filmed by a camera that sways above it, rendered with
    the same pinhole camera the page uses, without lens distortion (distortion_free: the live AR tracks its frames
    without any). truth(n) is the true T_C_M of frame n. With board None there is no board. script(t) -> (phase,
    hands) drives the sway and lays hands over the board (bench_script); exposure_s blurs the frame along the motion
    (the mean of three renders over the exposure); noise is the sensor noise (sigma, grey levels). The frames come
    once start() or wait_first() is called; render() works without."""

    THREAD = "synthetic_stream"
    distortion_free = True

    def __init__(self, board, spec, K, size=(1920, 1440), fps=30, seed=0, script=None, exposure_s=0.0, noise=2.0):
        self._init_frames()
        self.size, self.fps, self.K = tuple(size), fps, np.asarray(K, np.float64)
        self.spec, self.script, self.exposure = spec, script, float(exposure_s)
        self._truth = {}
        rng = np.random.default_rng(seed)
        w, h = self.size
        bg = cv2.resize(rng.uniform(70, 125, (9, 12)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        self._bg = np.clip(bg, 0, 255).astype(np.uint8)
        self._noise = []
        for _ in range(4):       # sensor noise as a part to add and a part to take away
            z = rng.normal(0, noise, (h, w))
            self._noise.append((np.clip(z, 0, 255).astype(np.uint8), np.clip(-z, 0, 255).astype(np.uint8)))
        self._tex = None
        if board is not None:
            ppm = calibrate_charuco.texture_ppm(board, 12.0) if spec["name"] == "sheet" else 24.0
            if spec["name"] == "sheet":
                page, origin = calibrate_charuco.page_image(board, ppm)
            else:
                n, sq = spec["squares"][0], spec["square_mm"]
                margin = int(round(3.0 * ppm))
                side = int(round(n * sq * ppm)) + 2 * margin
                page = board.generateImage((side, side), marginSize=margin, borderBits=1)
                origin = (margin / ppm, margin / ppm)
            self._tex = cv2.GaussianBlur(page, (0, 0), 1.0)
            ox, oy = origin
            self._S = np.array([[1 / ppm, 0, 0.5 / ppm - ox], [0, 1 / ppm, 0.5 / ppm - oy], [0, 0, 1.0]])
        nx, ny = spec["squares"]
        self._centre = np.array([nx * spec["square_mm"] / 2, ny * spec["square_mm"] / 2, 0.0])
        self._dist = 520.0 if spec["name"] == "sheet" else 380.0
        self._pan = 0.25 * min(nx, ny) * spec["square_mm"]

    def pose(self, t):
        """T_C_M at phase t (s with the default sway): the camera sways in front of the board, below its bottom edge,
        tilted 20 to 40 degrees from the normal, and looks at its centre (moved about with a script) with the top of
        the board up in the image."""
        tilt = np.radians(30 + 10 * np.sin(2 * np.pi * t / 6.5))
        phi = np.radians(90 + 40 * np.sin(2 * np.pi * t / 9.0))
        d = self._dist * (1 + 0.08 * np.sin(2 * np.pi * t / 5.0))
        look = self._centre.copy()
        if self.script is not None:     # the hand also pans
            look[:2] += self._pan * np.array([np.sin(2 * np.pi * t / 4.0), 0.7 * np.sin(2 * np.pi * t / 3.0)])
        eye = look + d * np.array([np.sin(tilt) * np.cos(phi), np.sin(tilt) * np.sin(phi), -np.cos(tilt)])
        z = (look - eye) / d
        y = np.array([0, 1.0, 0]) - z[1] * z
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        R_M_C = np.stack([x, y, z], 1)
        T = np.eye(4)
        T[:3, :3] = R_M_C.T
        T[:3, 3] = -R_M_C.T @ eye
        return T

    def state(self, t):
        """(T_C_M, hands) at time t."""
        if self.script is None:
            return self.pose(t), []
        phase, hands = self.script(t)
        return self.pose(phase), hands

    def _board(self, T):
        img = self._bg.copy()
        if self._tex is not None:
            H = self.K @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ self._S
            cv2.warpPerspective(self._tex, H, self.size, dst=img, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_TRANSPARENT)
        return img

    def render(self, t, k=0):
        """The frame at time t (grey) and its true T_C_M."""
        T, hands = self.state(t)
        if self.exposure > 0:
            a, b = self.state(t - self.exposure / 2)[0], self.state(t + self.exposure / 2)[0]
            img = cv2.addWeighted(cv2.addWeighted(self._board(a), 0.5, self._board(b), 0.5, 0), 2 / 3,
                                  self._board(T), 1 / 3, 0)
        else:
            img = self._board(T)
        for x, y, r in hands:          # a hand: skin grey with a soft edge
            ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            disc = np.column_stack([x + r * np.cos(ang), y + r * 0.8 * np.sin(ang), np.zeros_like(ang)])
            if np.all(marker_pose.transform(T, disc)[:, 2] > 1):
                poly = marker_pose.project(disc, T, self.K)
                cv2.fillPoly(img, [np.round(poly * 16).astype(np.int32)], 150, cv2.LINE_AA, 4)
        plus, minus = self._noise[k % len(self._noise)]
        return cv2.subtract(cv2.add(img, plus), minus), T

    def _run(self):
        t0 = time.perf_counter()
        period = 1.0 / self.fps
        nxt = t0
        while not self._stop.is_set():
            img, T = self.render(time.perf_counter() - t0, self._n)
            n = self._put(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            with self._lock:
                self._truth[n] = T
                self._truth.pop(n - 3000, None)
            nxt += period
            _pause(self._stop, nxt - time.perf_counter())

    def truth(self, n):
        with self._lock:
            return self._truth.get(n)


class ReplayStream(_Frames):
    """Stand-in for live.PhoneStream that plays recorded frames in a loop: a list of images, a folder of images or a
    video. A raw H.264 file (live.py record) is decoded as the phone's stream is and played with the recorded timing
    (CLIP_times.npy beside it); CLIP_truth.npz beside it, when there is one, holds the true T_C_M of every frame
    (`T_C_M`, N x 4 x 4). truth(n) gives back the item of `truth` (or of that file) that goes with frame n."""

    VIDEO = (".mp4", ".mov", ".mkv", ".avi", ".h264", ".264")
    THREAD = "replay_stream"

    def __init__(self, source, fps=10, truth=None):
        self._init_frames()
        src = source if isinstance(source, (list, tuple)) else [source]
        src = [p if isinstance(p, np.ndarray) else Path(p) for p in src]      # paths, or the images themselves
        if len(src) == 1 and isinstance(src[0], Path) and src[0].is_dir():
            # the digital twin keeps labels, masks and depth beside its images: they are not frames
            src = sorted(p for p in src[0].iterdir() if p.suffix.lower() in calibrate_charuco.IMAGE_SUFFIXES
                         and not p.stem.endswith(("_label", "_model", "_depth")))
        self._video = src[0] if len(src) == 1 and isinstance(src[0], Path) and src[0].suffix.lower() in self.VIDEO \
            else None
        self._images = [] if self._video else src
        self.fps = fps if self._video else min(fps, 10)       # images: at most 10 a second
        if not self._video and not self._images:
            raise LiveARError(f"nothing to play in {source}", "replay_unreadable")
        self._times_rec, self._pts_rec = None, None
        if self._video is not None:
            stem = self._video.with_suffix("")
            tp = Path(str(stem) + "_times.npy")
            if tp.is_file():
                self._times_rec = np.load(tp)
            pp = Path(str(stem) + "_pts.npy")
            if pp.is_file():
                self._pts_rec = np.load(pp)
            gp = Path(str(stem) + "_truth.npz")
            if truth is None and gp.is_file():
                truth = list(np.load(gp)["T_C_M"])
        self._truth = list(truth) if truth is not None else None
        self._k = 0                       # index of the frame in the recording
        self._index = {}                  # stream counter -> index in the recording

    def _frames(self):
        """(image, decode ms, index in the recording), in a loop."""
        while not self._stop.is_set():
            if self._video is not None and self._video.suffix.lower() in (".h264", ".264"):
                import av
                codec = av.CodecContext.create("h264", "r")
                codec.flags |= av.codec.context.Flags.low_delay
                k = 0
                with open(self._video, "rb") as fh:
                    while not self._stop.is_set():
                        data = fh.read(1 << 18)
                        try:
                            packets = codec.parse(data)
                        except av.error.FFmpegError:        # bytes that are not H.264: skipped
                            packets = []
                        for packet in packets:
                            t0 = time.perf_counter()
                            try:
                                frames = codec.decode(packet)
                            except av.error.FFmpegError:    # a bad packet: skipped (none good: replay_unreadable)
                                continue
                            for frame in frames:
                                img = frame.to_ndarray(format="bgr24")
                                yield img, (time.perf_counter() - t0) * 1e3, k
                                k += 1
                                t0 = time.perf_counter()
                        if not data:
                            break
                if k == 0:
                    self._ended(f"cannot read {self._video.name}", "replay_unreadable")
                    return
            elif self._video is not None:
                cap = cv2.VideoCapture(str(self._video))
                ok, img = cap.read()
                if not ok:
                    self._ended(f"cannot read {self._video.name}", "replay_unreadable")
                    return
                k = 0
                while ok and not self._stop.is_set():
                    yield img, None, k
                    k += 1
                    ok, img = cap.read()
                cap.release()
            else:
                for k, p in enumerate(self._images):
                    img = p if isinstance(p, np.ndarray) else cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        self._ended(f"cannot read {p.name}", "replay_unreadable")
                        return
                    yield img, None, k

    def _run(self):
        rec, pts = self._times_rec, self._pts_rec
        period = 1.0 / self.fps
        t_loop, nxt = time.perf_counter(), time.perf_counter()
        size = None
        for img, dec, k in self._frames():
            if rec is not None and k < len(rec):
                if k == 0:
                    t_loop = time.perf_counter()
                _pause(self._stop, t_loop + rec[k] - time.perf_counter())
            if k == 0 or img.shape[:2] != size:     # a new loop or a new frame size: the tracking starts again
                self._session += 1
                size = img.shape[:2]
            # the capture time: the phone's, on the clock of the replay (the recorded arrival less the quickest lag)
            t_cap = None
            if pts is not None and rec is not None and k < min(len(pts), len(rec)):
                if k == 0:
                    self._lag0 = float(np.min(rec - pts[:len(rec)])) if len(pts) >= len(rec) else 0.0
                t_cap = t_loop + pts[k] + self._lag0
            n = self._put(img, dec, t_cap)
            with self._lock:
                self._index[n] = k
                self._index.pop(n - 20000, None)
            if rec is None or k >= len(rec):
                nxt += period
                _pause(self._stop, nxt - time.perf_counter())
            if self._stop.is_set():
                return

    def truth(self, n):
        if not self._truth or n <= 0:
            return None
        with self._lock:
            k = self._index.get(n)
        if k is None:
            k = (n - 1) % (len(self._images) or len(self._truth))
        return self._truth[k] if k < len(self._truth) else None


# ── the live AR ──────────────────────────────────────────────────────────────
class LiveAR:
    """Phone stream -> board tracked on every frame -> filtered pose -> face overlay (see the module docstring).

    Three threads (track, detect, compose) start with the first start() or attach() and live until shutdown(); the
    compose thread makes, uses and releases its own OpenGL context. stop() and start() only close and open the stream.
    The calls of the page (status, configure, set_tracking, next_jpeg) only read or set values under a short lock;
    start() waits for the stream (seconds) and prepare() loads the models (seconds the first time): both belong on a
    worker thread, never on a GUI thread.

    Args:
        board: "patient" or "sheet" (BOARDS); square_mm, squares, marker_mm: the sheet's layout (None: the A4 one).
        registration: a JSON file with T_M_B (None: the demo registration).
        face, mandible, axes, opacity, hide_board: what is drawn (see configure()).
        intrinsics: the stream's calibration file (None: the nominal camera).
        synthetic, replay, replay_truth, script, exposure_s: a stand-in for the phone (tests, bench).
        serial, size, fps: the phone's stream.
        display_width, colors: the picture's width and the palette.
        log: a callable for progress lines (None: quiet).
        tracking: overrides of TRACKING and board_tracker.TRACK.
        prepare: load the models now (True, as before); False leaves it to prepare() or the first start().
    """

    def __init__(self, board="sheet", square_mm=None, registration=None, face=True, mandible=False, axes=True,
                 intrinsics=STREAM_INTRINSICS, synthetic=False, serial=None, size="1920x1440", fps=30,
                 display_width=DISPLAY_WIDTH, colors=None, log=None, replay=None, replay_truth=None, tracking=None,
                 script=None, exposure_s=0.0, squares=None, marker_mm=None, opacity=OPACITY, hide_board=False,
                 prepare=True):
        if board not in BOARDS:
            raise LiveARError(f"unknown board {board}", "bad_value")
        self.intrinsics_path = Path(intrinsics) if intrinsics else None
        self.synthetic, self.serial, self.size, self.fps = synthetic, serial, size, fps
        self.replay, self.replay_truth = replay, replay_truth     # recorded frames in place of the phone
        self.script, self.exposure = script, exposure_s           # the synthetic stream's session and blur
        self.display_width = display_width
        self.colors = {k: bgr(v) for k, v in {**COLORS, **(colors or {})}.items()}
        self.log = log or (lambda msg: None)
        self.tracking = {**board_tracker.TRACK, **TRACKING, **(tracking or {})}
        self.trace = None             # a list: one row per frame tracked, per detection and per frame shown (bench)
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._det_cv = threading.Condition(self._lock)
        self._job_cv = threading.Condition(self._lock)
        self._state_cv = threading.Condition(self._lock)     # the stream or the state changed
        self._apply_lock = threading.Lock()
        self._prepare_lock = threading.Lock()
        self._want = {"board": board, "square_mm": float(square_mm or calibrate_charuco.BOARD["square"]),
                      "squares": [int(n) for n in squares] if squares else None,
                      "marker_mm": float(marker_mm) if marker_mm else None,
                      "registration": str(registration) if registration else "", "face": bool(face),
                      "mandible": bool(mandible), "axes": bool(axes),
                      "opacity": float(np.clip(opacity, 0.0, 1.0)), "view_rotation": None,
                      "hide_board": bool(hide_board), "version": 1}
        self._tracking_version = 1
        self._stream = None
        self._opening = None            # the phone stream start() is opening: stop() closes it
        self._state, self._gen = IDLE, 0    # the stream's state; each start, stop and attach is a new generation
        self._error, self._error_detail, self._cause = "", "", ""
        self._jpeg, self._jpeg_n, self._bgr, self._jpeg_t = None, 0, None, None
        self._link = None               # the link of the phone's stream (usb, wifi); None for the stand-ins
        self._t_logged = -np.inf        # when a failure was last logged
        self._reset_run()
        self._status = {}
        self._quit = threading.Event()
        self._models, self._prepared = {}, False
        self._unrenderable = {}         # model -> why no renderer could be made for it (left out of the picture)
        self._renderers, self._centroids, self._render_fails, self._numpy_only = {}, {}, {}, set()
        self._gl, self._gl_failed = None, False     # the compose thread's OpenGL context
        self.renderer = ""
        self._applied = 0
        self._setup = None
        self._cam = (None, None, {})       # K, dist, source for the size of the last frame
        self._cam_size = None
        self._threads = []
        self.check = None               # the registration check (live_check.RegistrationCheck), when attached
        self._corrected = None          # (T_M_B, info) of a correction in use in place of the file's or the demo's
        self.synthetic_stream = None    # a callable(ar) -> stand-in stream in place of SyntheticStream (simulation)
        if prepare:
            self.prepare()
        self._apply(dict(self._want))

    def _reset_run(self):
        """What belongs to one run of the stream (under the lock, or before the threads start)."""
        self._pose = (0, None)            # (frame counter, T_C_M) of the last frame with a pose, as measured
        self._masks = (0, {})             # (frame counter, {model: silhouette at the display size}) of that frame
        self._tracker, self._tracker_key = None, None
        self._filter, self._filter_key = None, None
        self._session = None              # the stream session the tracker works on (a new one: start again)
        self._det_req = None              # (Frame, box, marker_px, key) for the detection worker
        self._det_busy = False
        self._det_done = None             # (Detection, key) for the tracker
        self._det_t = None                # capture time of the frame of the last detection asked for
        self._lost_box = None             # (box, marker_px, t) where the board was last, for the search when lost
        self._job = None                  # the newest frame tracked, for the compose thread
        self._latest = None               # summary of the newest frame tracked, for the status
        self._tracked_times = deque(maxlen=60)
        self._shown_times = deque(maxlen=60)
        self._det_times = deque(maxlen=30)
        self._tm_track = deque(maxlen=30)
        self._tm_det = deque(maxlen=30)
        self._tm_show = deque(maxlen=30)
        self._served = deque(maxlen=60)
        self._verdicts = deque(maxlen=90)
        self._recent = deque(maxlen=120)  # (capture time, pose drawn, pose found) of the frames tracked: the tremble
        self._skipped = 0                 # frames tracked but not drawn (the compose thread was busy)
        self._view_k = None               # quarter turns of the composed frame (_view_quarter)
        self._cpu = deque(maxlen=40)      # (wall, process CPU time) for the load
        self._frame_errors = 0            # frames (or stages of one) that failed in this run
        self._fails_since_good = 0        # failures since the last frame composed
        self._last_frame_error = None     # {"where", "what", "t"} of the last one
        self._t_good = time.perf_counter()   # the last frame composed (or the start of the run)

    # ── control, from any thread ─────────────────────────────────────────
    @property
    def running(self):
        """True while a stream is attached and alive; False for one that died (start() opens a new one)."""
        stream = self._stream
        return stream is not None and bool(stream.alive)

    @property
    def closed(self):
        """True after shutdown(): the frames' readers end (next_jpeg and next_frame only time out)."""
        return self._quit.is_set()

    @property
    def state(self):
        """The stream's state (STATES)."""
        with self._lock:
            return self._state_now()

    def prepare(self):
        """Loads the models (decimated for the live view the first time: seconds), once, thread safe. start() and
        attach() call it; an app calls it first on a worker thread so that its GUI never waits for it.

        Returns:
            self.
        """
        with self._prepare_lock:
            if self._prepared:
                return self
            models = load_models(log=self.log)
            with self._lock:
                self._models, self._prepared = models, True
                if self._setup is not None:
                    self._want["version"] += 1      # the demo registration and the model points use the face
                want = dict(self._want)
            if self._setup is not None and not self.running:
                self._apply(want)
        return self

    def start(self, timeout=None):
        """Opens the stream (closing the phone's camera app first) and returns once the first frame is in. A stream
        that died is closed and a new one opened, without stop(). A call while another start is under way waits for
        it; stop() during the start cancels it (the stream opened is closed, start() returns, running stays False).
        timeout is not used (kept for the callers).

        Returns:
            self.

        Raises:
            LiveARError: after shutdown() ("closed"), or a stand-in that cannot play ("replay_unreadable").
            live.LiveError: the phone's stream did not start (its code says why); status() says it too.
        """
        with self._lock:
            if self._quit.is_set():
                raise LiveARError("the live AR was shut down", "closed")
            if self._state == STARTING:
                self._state_cv.wait_for(lambda: self._state != STARTING or self._quit.is_set(), START_WAIT_S)
                return self
            if self._state_now() == RUNNING:
                return self
            dead, self._stream = self._stream, None
            self._gen += 1
            gen = self._gen
            self._state = STARTING
            self._error, self._error_detail, self._cause = "", "", ""
            self._jpeg, self._bgr = None, None
            self._state_cv.notify_all()
        if dead is not None:
            self._close(dead)
        try:
            self.prepare()
            self._ensure_threads()
            stream = self._open(gen)
        except BaseException as e:  # recorded for status() (a SystemExit too: never STARTING for ever), then raised
            with self._lock:
                mine = self._gen == gen
                if mine:
                    self._state = FAILED
                    code = getattr(e, "code", "") if isinstance(e, Exception) else ""    # SystemExit's is its text
                    self._error = code if isinstance(code, str) and code else "start_failed"
                    self._cause = getattr(e, "cause", "") or ""
                    self._state_cv.notify_all()
            logger.info("live AR: start failed (%s): %s", getattr(e, "code", ""), e)
            if not mine and isinstance(e, Exception):   # stop() came meanwhile: a cancelled start, not a failure
                return self
            raise
        if stream is None:          # cancelled while opening
            return self
        with self._lock:
            mine = self._gen == gen and not self._quit.is_set()
            if mine:
                self._install(stream)
        if not mine:
            self._close(stream)
        return self

    def _open(self, gen):
        """The stream of this run, started (the first frame in); None when stop() cancelled the start."""
        if self.replay is not None:
            stream = ReplayStream(self.replay, self.fps, self.replay_truth)
        elif self.synthetic:
            stream = self.synthetic_stream(self) if self.synthetic_stream is not None else self._synthetic_stream()
        else:
            close_camera_app(self.serial)
            stream = live.PhoneStream(self.serial, self.size, self.fps, log=self.log)
        with self._lock:
            cancelled = self._gen != gen
            if not cancelled:
                self._opening = stream
        if cancelled:
            self._close(stream)
            return None
        try:
            if isinstance(stream, live.PhoneStream):
                stream.start()
            else:
                stream.wait_first()
        except live.LiveError as e:
            if e.code == "stopped":     # closed by stop()
                return None
            raise
        finally:
            with self._lock:
                if self._opening is stream:
                    self._opening = None
        if stream.latest_item() is None and not stream.alive:      # a stand-in that ended at once
            stream.close()
            raise LiveARError(stream.error or "the stream ended", getattr(stream, "code", "") or "stream_stopped")
        return stream

    def attach(self, stream):
        """Works on a stream opened elsewhere (anything with wait_item or wait_frame, stats, alive, error, close); the
        one before is closed."""
        self.prepare()
        self._ensure_threads()
        with self._lock:
            if self._quit.is_set():
                raise LiveARError("the live AR was shut down", "closed")
            self._gen += 1
            old = self._stream
            self._install(stream)
        if old is not None and old is not stream:
            self._close(old)
        return self

    def _install(self, stream):
        """The stream of a new run (under the lock)."""
        self._stream = stream
        self._link = getattr(stream, "link", None)
        self._status = {}
        self._reset_run()               # nothing of the stream before
        self._cam_size = None           # intrinsics read again: a calibration may have been saved meanwhile
        self._state = RUNNING
        self._error, self._error_detail, self._cause = "", "", ""
        self._state_cv.notify_all()

    def _synthetic_stream(self):
        w, h = self._size_wh()
        K, _, _ = stream_intrinsics((w, h), self.intrinsics_path)
        setup = self._setup
        return SyntheticStream(setup["board"], setup["spec"], K, (w, h), self.fps, script=self.script,
                               exposure_s=self.exposure)

    def stop(self):
        """Closes the stream, or cancels a start under way; the picture goes. Returns once the stream is closed."""
        with self._lock:
            self._gen += 1
            stream, self._stream = self._stream, None
            opening, self._opening = self._opening, None
            if self._state != CLOSED:
                self._state = IDLE
            self._error, self._error_detail, self._cause = "", "", ""
            self._jpeg, self._bgr = None, None
            for cv in (self._frame_ready, self._det_cv, self._state_cv):
                cv.notify_all()
        for s in (opening, stream):
            if s is not None:
                self._close(s)

    def shutdown(self):
        """Stops the stream and the threads (each joined for JOIN_S); the compose thread releases its renderers and
        its OpenGL context. The object cannot start again."""
        self.stop()
        with self._lock:
            self._quit.set()
            self._state = CLOSED
            for cv in (self._det_cv, self._job_cv, self._frame_ready, self._state_cv):
                cv.notify_all()
            threads = list(self._threads)
        for th in threads:
            th.join(timeout=JOIN_S)
        if self.check is not None:
            self.check.close()

    @staticmethod
    def _close(stream):
        try:
            stream.close()
        except (OSError, RuntimeError) as e:    # a stream that fails to close must not stop the caller
            logger.warning("live AR: closing the stream failed: %s", e)

    def _ensure_threads(self):
        with self._lock:
            if self._threads or self._quit.is_set():
                return
            for name, target in (("live_ar_compose", self._compose_loop), ("live_ar_track", self._track_loop),
                                 ("live_ar_detect", self._detect_loop)):
                th = threading.Thread(target=target, daemon=True, name=name)
                th.start()
                self._threads.append(th)

    def _state_now(self):
        """The state, a stream that died seen at once (under the lock)."""
        if self._state == RUNNING and (self._stream is None or not self._stream.alive):
            self._stream_died(self._stream)
        return self._state

    def _stream_died(self, stream):
        """The stream gave up (under the lock): failed, with its code and the link's cause."""
        self._state = FAILED
        self._error = getattr(stream, "code", "") or "stream_stopped"
        self._cause = getattr(stream, "cause", "") or ""
        self._state_cv.notify_all()
        logger.info("live AR: the stream stopped (%s): %s", self._error, getattr(stream, "error", ""))

    def _error_now(self):
        """status()["error"] (under the lock): why the stream failed, or frames_failing when no frame has come out
        for FAILING_S with failures meanwhile; "" otherwise."""
        if self._state == FAILED:
            return self._error or "stream_stopped"
        if (self._state == RUNNING and self._fails_since_good >= FAILING_COUNT
                and time.perf_counter() - self._t_good > FAILING_S):
            return "frames_failing"
        return ""

    def configure(self, **kw):
        """board, square_mm, squares and marker_mm (the calibration board's layout: None for the A4 one), face,
        mandible, axes, opacity (0-1, of the overlay), view_rotation (None: the phone's way up,
        else 0, 90, 180 or 270 degrees clockwise), hide_board (the board covered in the picture shown): applied from
        the next frame. Only a change of the board starts the tracking again; the layers, the opacity, the rotation and
        hide_board do not touch it. Everything is checked before anything is taken.

        Raises:
            LiveARError: an unknown board or rotation, a value that is not a number ("bad_value"), a board that cannot
                be made ("bad_board": a marker not smaller than its square, too many squares); nothing is changed.
        """
        if kw.get("board") is not None and kw["board"] not in BOARDS:
            raise LiveARError(f"unknown board {kw['board']}", "bad_value")
        if "view_rotation" in kw and kw["view_rotation"] not in (None, 0, 90, 180, 270):
            raise LiveARError(f"view_rotation {kw['view_rotation']}: None, 0, 90, 180 or 270", "bad_value")
        with self._lock:
            want = dict(self._want)
            try:
                if kw.get("board") is not None:
                    want["board"] = kw["board"]
                if kw.get("square_mm") is not None:
                    want["square_mm"] = float(kw["square_mm"])
                if "squares" in kw:
                    want["squares"] = [int(n) for n in kw["squares"]] if kw["squares"] else None
                if "marker_mm" in kw:
                    want["marker_mm"] = float(kw["marker_mm"]) if kw["marker_mm"] else None
                for k in ("face", "mandible", "axes", "hide_board"):
                    if kw.get(k) is not None:
                        want[k] = bool(kw[k])
                if kw.get("opacity") is not None:
                    opacity = float(kw["opacity"])
                    if not np.isfinite(opacity):
                        raise ValueError(f"opacity {opacity}")
                    want["opacity"] = float(np.clip(opacity, 0.0, 1.0))
                if "view_rotation" in kw:
                    want["view_rotation"] = kw["view_rotation"]
            except (TypeError, ValueError) as e:
                raise LiveARError(f"a setting that is not a number: {e}", "bad_value") from e
            board_changed = any(want[k] != self._want[k] for k in BOARD_KEYS)
            if board_changed:
                check_board(want["board"], want["square_mm"], want["squares"], want["marker_mm"])
                want["version"] += 1
            self._want = want
            want = dict(want)
            setup = self._setup
        if not self.running:
            self._apply(want)
        elif self.synthetic and board_changed and (want["board"] != setup["spec"]["name"] or want["board"] == "sheet"):
            self._apply(want)       # the synthetic stream films the board it was made with: a new one for a new board
            self.stop()
            self.start()

    def set_registration(self, path):
        """A JSON file with T_M_B, checked here (raises LiveARError); None for the demo."""
        if path:
            load_registration(path)
        with self._lock:
            self._want["registration"] = str(path) if path else ""
            self._want["version"] += 1
            self._corrected = None          # a correction belongs to the registration it corrected
            want = dict(self._want)
        if self.check is not None:
            self.check.forget()
        if not self.running:
            self._apply(want)

    def set_corrected(self, T_M_B, info=None):
        """A registration corrected from the picture (4 x 4, mm; info: what the status says of it) in place of the
        file's or the demo's, until the next set_registration(); None goes back to that one."""
        with self._lock:
            self._corrected = None if T_M_B is None else (np.array(T_M_B, np.float64), dict(info or {}))
            self._want["version"] += 1
            want = dict(self._want)
        if not self.running:
            self._apply(want)

    def attach_check(self, check):
        """The registration check (live_check.RegistrationCheck): it gets every frame with its own board pose (the
        track thread), draws over the picture (the compose thread) and is in status()["check"]."""
        check.attach(self)
        self.check = check

    def set_tracking(self, **kw):
        """Overrides of TRACKING and of board_tracker.TRACK (unknown keys refused); tracker and filter start again
        from the next frame, the detectors take theirs at once."""
        bad = [k for k in kw if k not in TRACKING and k not in board_tracker.TRACK]
        if bad:
            raise LiveARError(f"unknown tracking settings {bad}", "bad_value")
        with self._lock:
            self.tracking = {**self.tracking, **kw}
            self._tracking_version += 1
            setup, tracking = self._setup, self.tracking
        if setup is not None:
            for d in (setup["detector"], setup["detector_track"]):
                d.configure(tracking)

    # ── what the page reads ──────────────────────────────────────────────
    def next_jpeg(self, after=0, timeout=1.0):
        """(JPEG bytes, frame counter) of a composed frame newer than `after`; (None, after) on timeout, when stopped,
        or once shut down (then after the whole timeout: a client loop never spins; it ends on `closed`)."""
        jpeg, n, _ = self.next_frame(after, timeout)
        return jpeg, n

    def next_frame(self, after=0, timeout=1.0):
        """As next_jpeg, with the time.perf_counter() when the frame was composed: (JPEG, counter, time), or
        (None, after, None)."""
        deadline = time.perf_counter() + timeout
        with self._frame_ready:
            self._frame_ready.wait_for(
                lambda: (self._jpeg is not None and self._jpeg_n != after) or self._quit.is_set(), timeout=timeout)
            if not self._quit.is_set() and self._jpeg is not None and self._jpeg_n != after:
                return self._jpeg, self._jpeg_n, self._jpeg_t
        if self._quit.is_set():
            time.sleep(max(0.0, deadline - time.perf_counter()))
        return None, after, None

    def note_served(self, ms):
        """The MJPEG server wrote a frame to the page in `ms`."""
        with self._lock:
            self._served.append((time.perf_counter(), ms))

    def composed(self):
        """The newest composed frame as a BGR array (None when stopped)."""
        with self._lock:
            return self._bgr

    def last_masks(self):
        """(stream frame counter, {model: silhouette}) of the last frame drawn at its own pose, at the display size."""
        with self._lock:
            return self._masks

    def last_pose(self):
        """(stream frame counter, T_C_M) of the last frame where the board was found, as found (not filtered)."""
        with self._lock:
            return self._pose

    def stream(self):
        return self._stream

    def still(self):
        """(the newest frame of the stream at its full size as BGR, its counter), or (None, 0) when stopped: what a
        calibration of the streaming mode keeps."""
        stream = self._stream
        item = stream.latest_item() if stream is not None and hasattr(stream, "latest_item") else None
        if item is None:
            return None, 0
        return item.bgr().copy(), item.n

    def settings(self):
        """The settings asked for: board, square_mm, squares, marker_mm, registration, face, mandible, axes, opacity,
        view_rotation, hide_board."""
        with self._lock:
            return {k: v for k, v in self._want.items() if k != "version"}

    def status(self):
        """What the page shows: the keys listed in the module docstring (state, error, light, rates, timings,
        sources, layers). Never raises for a bad calibration file (its source says intrinsics_bad)."""
        with self._lock:
            state = self._state_now()
            error, cause, detail = self._error_now(), self._cause, self._error_detail
            st = dict(self._status)
            stream = self._stream
            want = dict(self._want)
            served = list(self._served)
            cpu = list(self._cpu)
            setup = self._setup
            models, prepared, unrenderable = self._models, self._prepared, set(self._unrenderable)
            frame_errors, last = self._frame_errors, self._last_frame_error
            cam = self._cam[2]
            K_now = self._cam[0]
            recent = list(self._recent)
        now = time.perf_counter()
        if prepared:
            have = {k: k in models and k not in unrenderable for k in ("face", "mandible")}
        else:                           # not loaded yet: what is on this computer
            have = {"face": FACE_MESH.is_file(), "mandible": MANDIBLE_MESH.is_file()}
        st.update({"running": state == RUNNING, "starting": state == STARTING, "stream_state": state,
                   "error": error, "cause": cause, "error_detail": detail, "frame_errors": frame_errors,
                   "last_frame_error": None if last is None else {"where": last["where"], "what": last["what"],
                                                                  "age_s": round(now - last["t"], 1)},
                   "synthetic": self.synthetic or self.replay is not None, "replay": self.replay is not None,
                   "renderer": self.renderer,
                   "layers": {k: want[k] for k in ("face", "mandible", "axes")}, "opacity": want["opacity"],
                   "hide_board": want["hide_board"], "models": have,
                   "triangles": {k: int(len(f)) for k, (_, f) in models.items()},
                   "board": setup["spec"], "registration": setup["registration"], "score": None})
        if not st.get("intrinsics"):
            st["intrinsics"] = cam or stream_intrinsics(self._size_wh(), self.intrinsics_path)[2]
        check = self.check
        st["check"] = check.status() if check is not None else None
        if stream is None:
            st.update({"tracked": False, "fps_stream": 0.0, "fps_ar": 0.0, "fps_pose": 0.0})
            return st
        st["tremble_px"], st["tremble_raw_px"] = (None, None)
        if K_now is not None and recent and setup is not None and state == RUNNING:
            X = self._model_points(setup)
            X = X[np.linspace(0, len(X) - 1, min(TREMBLE_POINTS, len(X))).astype(int)]
            st["tremble_px"], st["tremble_raw_px"] = tremble(recent, X, K_now, now_t=recent[-1][0])
        s = stream.stats()
        st["fps_stream"] = round(s["fps"], 1)
        st["frames"] = s["frames"]
        gap = s.get("interval_p95_ms")
        st["gap_p95_ms"] = round(gap, 1) if gap == gap and gap is not None else None   # not NaN
        tm = dict(st.get("timings", {}))
        if s.get("decode_ms") == s.get("decode_ms"):     # not NaN
            tm["decode_ms"] = round(s["decode_ms"], 1)
        recent = [ms for t, ms in served if now - t < 2.0]
        if recent:
            tm["serve_ms"] = round(float(np.mean(recent)), 1)
            st["fps_served"] = round(len(recent) / 2.0, 1)
        st["timings"] = tm
        if len(cpu) > 1 and cpu[-1][0] > cpu[0][0]:
            st["cpu_cores"] = round((cpu[-1][1] - cpu[0][1]) / (cpu[-1][0] - cpu[0][0]), 2)
        st["stream_dropped"] = s.get("dropped", 0)
        late = s.get("late_p95_ms")
        st["late_p95_ms"] = round(late, 1) if late is not None and late == late else None
        if stream.error:
            st["stream_error"] = stream.error
        st["link"] = s.get("link")
        st["link_waiting"] = bool(s.get("waiting"))     # the link dropped: the stream waits for it to come back
        st["link_drops"] = s.get("link_drops", 0)
        st["restarts"] = s.get("restarts", 0)
        if state != RUNNING or error == "frames_failing":
            # stopped, or no frame composed for FAILING_S: the rates and the light of the last good frame are stale
            st.update({"tracked": False, "fps_ar": 0.0, "fps_pose": 0.0})
        return st

    def _size_wh(self):
        w, h = (int(x) for x in self.size.split("x"))
        return w, h

    # ── setup ────────────────────────────────────────────────────────────
    def _apply(self, want):
        """Board, detectors and registration for the wanted settings (track thread, or any thread while stopped)."""
        with self._apply_lock:
            if want["version"] == self._applied and self._setup is not None:
                return
            old = self._setup
            board, spec = make_board(want["board"], want["square_mm"], want["squares"], want["marker_mm"])
            if old is not None and spec == old["spec"]:
                board, detector, detector_track = old["board"], old["detector"], old["detector_track"]
                for d in (detector, detector_track):
                    d.configure(self.tracking)
            else:   # one detector per thread: a detection in the worker never makes the tracking wait
                detector = board_tracker.Detector(board, spec, self.tracking)
                detector_track = board_tracker.Detector(board, spec, self.tracking)
            models = self._models
            reg = {"source": "demo", "path": "", "demo": "", "error": ""}
            T = None
            if want["registration"]:
                try:
                    T = load_registration(want["registration"])
                    reg.update(source="file", path=want["registration"])
                except (OSError, ValueError, KeyError, TypeError, LiveARError) as e:
                    reg["error"] = str(e)
            if T is None and "face" in models:
                T, reg["demo"] = demo_registration(spec, models["face"][0])
            corrected = self._corrected
            if corrected is not None and T is not None:     # a correction of this registration, in its place
                reg.update(corrected=True, base=reg["source"], **corrected[1])
                reg["source"] = "corrected"
                T = corrected[0]
            # one assignment, so a worker never sees the board of one setting with the detector of another
            # the way up of the model in M: +z of the bench frame (the top of the head); the top of the board without
            # a registration
            up = T[:3, :3] @ np.array([0, 0, 1.0]) if T is not None else np.array([0, -1.0, 0])
            self._setup = {"board": board, "spec": spec, "detector": detector, "detector_track": detector_track,
                           "T_M_B": T, "registration": reg, "version": want["version"],
                           "obj": board.getChessboardCorners().astype(np.float64), "up_M": up / np.linalg.norm(up)}
            self._applied = want["version"]

    @property
    def _spec(self):
        return self._setup["spec"]

    def _renderer(self, name):
        """The renderer of a model, made in the compose thread: OpenGL on the thread's own context, or NumPy when
        OpenGL is not there (or failed on this model); None when neither can be made (the model is left out)."""
        r = self._renderers.get(name)
        if r is not None or name in self._unrenderable:
            return r
        v, f = self._models[name]
        kind = "opengl"
        if not self._gl_failed and name not in self._numpy_only:
            try:
                import gl_overlay
                if self._gl is None:
                    self._gl = gl_overlay.standalone_context()
            except Exception as e:  # noqa: BLE001  no OpenGL here (no moderngl, no driver): NumPy for every model
                self._gl_failed = True
                self.log(f"OpenGL not available ({type(e).__name__}: {e}); NumPy renderer")
            else:
                try:
                    r = gl_overlay.GLMeshRenderer(v, f, ctx=self._gl)
                except Exception as e:  # noqa: BLE001  this model only: NumPy
                    self.log(f"{name}: OpenGL renderer not available ({type(e).__name__}: {e}); NumPy renderer")
        if r is None:
            kind = "numpy"
            try:
                import mesh_overlay
                r = mesh_overlay.MeshRenderer(v, f)
            except Exception as e:  # noqa: BLE001  the model is left out, the rest goes on
                with self._lock:
                    self._unrenderable[name] = type(e).__name__
                self.log(f"{name}: no renderer ({type(e).__name__}: {e}); left out of the picture")
                return None
        self._renderers[name] = r
        if self.renderer != "numpy":
            self.renderer = kind
        return r

    def _render_failed(self, name, rend, error):
        """A model that failed to draw on a frame: left out of that frame; after RENDER_FAILS in a row its OpenGL
        renderer gives way to the NumPy one, and a NumPy one that fails as often leaves the model out of the run."""
        n = self._render_fails.get(name, 0) + 1
        self._render_fails[name] = n
        if n < RENDER_FAILS:
            return
        self._render_fails[name] = 0
        self._renderers.pop(name, None)
        if hasattr(rend, "release"):
            self._release(rend)
        if name in self._numpy_only:
            with self._lock:
                self._unrenderable[name] = type(error).__name__
            self.log(f"{name}: left out of the picture after {RENDER_FAILS} failures in a row")
        else:
            self._numpy_only.add(name)
            self.log(f"{name}: OpenGL failed {RENDER_FAILS} times in a row; NumPy renderer")

    @staticmethod
    def _release(rend):
        try:
            rend.release()
        except Exception as e:  # noqa: BLE001  freeing GPU objects at the end: nothing else to do
            logger.debug("renderer release failed: %s", e)

    def _release_gl(self):
        """The renderers and the OpenGL context freed, on the compose thread that made them."""
        for rend in list(self._renderers.values()):
            if hasattr(rend, "release"):
                self._release(rend)
        self._renderers.clear()
        if self._gl is not None:
            try:
                self._gl.release()
            except Exception as e:  # noqa: BLE001  as above
                logger.debug("OpenGL context release failed: %s", e)
            self._gl = None

    def _intrinsics(self, w, h, stream=None):
        """K, dist and their source for frames of w x h: the stream's calibration or the nominal camera; without the
        lens distortion for a stand-in that renders without it (SyntheticStream)."""
        plain = bool(getattr(stream, "distortion_free", False))
        if self._cam_size != (w, h, plain):
            K, dist, src = stream_intrinsics((w, h), self.intrinsics_path)
            if plain and np.any(dist):
                dist = np.zeros_like(dist)
                src = {**src, "distortion": "none (synthetic frames)"}
            self._cam = (K, dist, src)
            self._cam_size = (w, h, plain)
        return self._cam

    def _model_points(self, setup):
        """Points of the face (or of the board) in M: the anchor of the filter and what the tracker weighs the two
        mirrored poses on."""
        T_M_B = setup["T_M_B"]
        models = self._models
        if T_M_B is not None and "face" in models:
            v = models["face"][0]
            v = v[np.linspace(0, len(v) - 1, min(200, len(v))).astype(int)]
            return v @ T_M_B[:3, :3].T + T_M_B[:3, 3]
        nx, ny = setup["spec"]["squares"]
        sq = setup["spec"]["square_mm"]
        return np.array([[0, 0, 0], [nx * sq, 0, 0], [nx * sq, ny * sq, 0], [0, ny * sq, 0]], np.float64)

    def _tracker_for(self, setup, K, item):
        """Tracker and filter of this run, made again when the board, the registration, the settings, the camera or
        the stream session (a restart, a new frame size) change."""
        key = (setup["version"], self._tracking_version, round(float(K[0, 0]), 3), item.size, item.session)
        if self._tracker is None or self._tracker_key != key:
            pts = self._model_points(setup)
            self._tracker = board_tracker.Tracker(setup["board"], setup["spec"], self.tracking, pts)
            # the filter's anchor is the board's centre (its position is what the corners give best); the model
            # reaches `radius` from it, the lever of the rotation
            nx, ny = setup["spec"]["squares"]
            anchor = np.array([nx, ny, 0.0]) * setup["spec"]["square_mm"] / 2
            radius = float(np.sqrt(np.mean(np.sum((pts - anchor) ** 2, axis=1))))
            self._filter = pose_filter.PoseTracker(self.tracking, anchor, radius, float(K[0, 0]))
            self._tracker_key = key
            with self._lock:
                self._det_done, self._lost_box, self._det_t = None, None, None
        return self._tracker, self._filter

    def _next_item(self, stream, after):
        """The newest frame after `after` as a live.Frame (streams without wait_item: wrapped)."""
        if hasattr(stream, "wait_item"):
            return stream.wait_item(after, timeout=WAIT_FRAME_S)
        frame, t, n = stream.wait_frame(after, timeout=WAIT_FRAME_S)
        return None if frame is None else live.Frame(n, t, bgr=frame)

    # ── thread 1: every frame, the board tracked ─────────────────────────
    def _track_loop(self):
        cur, last, dead = None, 0, None
        while not self._quit.is_set():
            with self._lock:
                stream = self._stream
                if stream is None or stream is dead:
                    self._state_cv.wait(POLL_S)
                    continue
            if stream is not cur:
                cur, last = stream, 0
            t0 = time.perf_counter()
            try:
                item = self._next_item(stream, last)
            except Exception as e:  # noqa: BLE001  a stream object that fails: counted, a pause, then again
                self._failed(stream, "stream", e)
                self._quit.wait(FAIL_PAUSE_S)
                continue
            if item is None:
                if not stream.alive:
                    with self._lock:
                        if self._stream is stream and self._state == RUNNING:
                            self._stream_died(stream)
                    dead = stream
                elif time.perf_counter() - t0 < 1e-3:     # a closed stream answers at once: no spinning on it
                    self._quit.wait(WAIT_FRAME_S)
                continue
            last = item.n
            try:
                self._track(stream, item)
            except Exception as e:  # noqa: BLE001  one bad frame must not end the thread nor stop the picture
                self._failed(stream, "track", e)
                self._post_plain(stream, item)

    def _track(self, stream, item):
        t0 = time.perf_counter()
        with self._lock:
            want = dict(self._want)
        if want["version"] != self._applied:
            self._apply(want)
        setup = self._setup
        w, h = item.size
        K, dist, cam = self._intrinsics(w, h, stream)
        limit = MAX_RMS[cam["source"]]
        tracker, filt = self._tracker_for(setup, K, item)
        s = self.tracking
        key = self._tracker_key
        with self._lock:
            done, self._det_done = self._det_done, None
            lost_box, det_t = self._lost_box, self._det_t
        det = done[0] if done is not None and done[1] == key else None
        gray = item.gray()
        if s.get("views"):
            tracker.reset()
            det = setup["detector_track"].detect(gray, n=item.n, t=item.t_capture)
        t1 = time.perf_counter()
        res = tracker.step(gray, item.t_capture, item.n, K, dist, limit, det)
        t2 = time.perf_counter()
        if not res.ok and not s.get("views"):
            res = self._recover(tracker, filt, setup, gray, item, K, dist, limit, res, lost_box)
        if res.ok and not res.sure and filt.t_good is not None and item.t_capture - filt.t_good <= s["hold_s"]:
            # found again after a short loss where the overlay still is: the same of the two mirrored poses
            prior = filt.prior(item.t_capture)
            if prior is not None and tracker._overlay_shift(res.T, prior, K) < 3 * tracker.s["ambiguous_px"]:
                tracker.sure = res.sure = True
        spec = setup["spec"]
        strong = res.ok and res.n >= s["strong_corners"] * spec["corners"] and res.rms <= s["strong_rms"] * limit
        use = res.T if (res.ok and (res.sure or not s.get("wait_sure", True))) else None
        if not s.get("smooth", True) and use is not None:
            strong = True
        verdict = filt.measure(item.t_capture, use, strong, res.noise_px, res.noise_pos_px, res.noise_range_mm)
        T, vis, state = filt.pose_at(item.t_capture)
        tracked, rejected = filt.tracked, filt.counts["rejected"]   # copied here: the filter is this thread's
        age = None if filt.t_good is None else item.t_capture - filt.t_good    # since the last pose taken
        t3 = time.perf_counter()
        # the next detection: in the box around the board, or where it was last, or in the whole frame
        if res.ok and res.box is not None:
            lost_box = (res.box, res.marker_px, item.t_capture)
        ask = None
        if not s.get("views") and tracker.want_detection(res, item.t_capture, det_t):
            box, mpx = (res.box, res.marker_px) if res.ok else (None, None)
            if box is None and lost_box is not None and item.t_capture - lost_box[2] < s["roi_s"]:
                box, mpx = lost_box[0], lost_box[1]
            ask = (item, box, mpx, key)
        tm = {"gray_ms": (t1 - t0) * 1e3, "track_ms": (t2 - t1) * 1e3, "filter_ms": (t3 - t2) * 1e3,
              **{k: v for k, v in res.tm.items() if k != "track_ms"}}
        with self._lock:
            if self._stream is not stream:
                return
            if res.ok and res.box is not None:
                self._lost_box = lost_box
            if ask is not None and not self._det_busy:
                self._det_req, self._det_busy, self._det_t = ask, True, item.t_capture
                self._det_cv.notify_all()
            self._tracked_times.append(time.perf_counter())
            self._tm_track.append(tm)
            self._verdicts.append(verdict)
            self._recent.append((item.t_capture, T if vis > 0 else None, use))
            self._cpu.append((time.perf_counter(), time.process_time()))
            if res.ok:
                self._pose = (item.n, res.T)
            self._latest = {"n": item.n, "ok": res.ok, "sure": res.sure, "corners": res.n if res.ok else
                            (len(det.ids) if det is not None else 0), "rms_px": res.rms, "source": res.source,
                            "limit": limit, "markers": det.markers if det is not None else
                            (self._latest or {}).get("markers", 0), "counts": dict(tracker.counts),
                            "ambiguity_px": res.ambiguity_px}
            if self._job is not None:
                self._skipped += 1
            self._job = {"stream": stream, "item": item, "T": T, "vis": vis, "state": state, "res": res,
                         "verdict": verdict, "K": K, "dist": dist, "cam": cam, "setup": setup, "want": want,
                         "tracked": tracked, "rejected": rejected, "t_tracked": t3, "age": age}
            self._job_cv.notify_all()
            if self.trace is not None:
                self.trace.append({"what": "track", "n": item.n, "t": item.t, "tc": item.t_capture, "ok": res.ok,
                                   "size": list(item.size),
                                   "sure": res.sure, "T_raw": res.T, "T": T, "vis": vis, "state": state,
                                   "verdict": verdict, "corners": res.n, "rms_px": res.rms, "source": res.source,
                                   "reanchored": res.reanchored, "shift_px": res.shift_px, "flipped": res.flipped,
                                   "noise_px": res.noise_px, "noise_pos_px": res.noise_pos_px,
                                   "noise_range_mm": res.noise_range_mm,
                                   "recovered": res.source == "recovered",
                                   "t_done": t3, "tm": tm, "detection": None if det is None else det.n})
        check = self.check
        if check is not None and use is not None:    # a frame with its own pose: to the registration check
            try:
                check.frame(item, res, K, dist, setup, limit)
            except Exception as e:  # noqa: BLE001  the check misses this frame; the picture goes on
                self._failed(stream, "check", e)

    def _post_plain(self, stream, item):
        """A frame whose tracking failed still goes to the picture, without a pose: the video goes on."""
        try:
            setup = self._setup
            K, dist, cam = self._intrinsics(*item.size, stream)
            with self._lock:
                if self._stream is not stream:
                    return
                want = dict(self._want)
                if self._job is not None:
                    self._skipped += 1
                self._job = {"stream": stream, "item": item, "T": None, "vis": 0.0, "state": "lost",
                             "res": board_tracker.TrackResult(), "verdict": "missed", "K": K, "dist": dist, "cam": cam,
                             "setup": setup, "want": want, "tracked": False, "rejected": 0,
                             "t_tracked": time.perf_counter()}
                self._job_cv.notify_all()
        except Exception as e:  # noqa: BLE001  the frame is skipped; the failure was counted already
            logger.debug("live AR: frame %s not shown: %s", item.n, e)

    def _recover(self, tracker, filt, setup, gray, item, K, dist, limit, res, lost_box):
        """The board lost on this frame shortly after it was followed: searched at once in the box where the pose
        carried on puts it (grown by its speed), on this same frame, so the overlay does not wait for the worker."""
        lb = lost_box
        if lb is None or item.t_capture - lb[2] > self.tracking["recover_s"]:
            return res
        prior = filt.prior(item.t_capture)
        box, mpx = (tracker.box_for(prior, K, dist, item.size) if prior is not None else (None, None))
        if box is None:
            box, mpx = lb[0], lb[1]
        # only where it is quick (a small board, or a large one far away): a large box is left to the worker, which
        # takes this same frame at once
        scale = min(1.0, self.tracking["marker_px"] / max(mpx or 1.0, 1.0))
        if (box[2] - box[0]) * (box[3] - box[1]) * scale * scale > self.tracking["recover_px"]:
            return res
        d = setup["detector_track"].detect(gray, box, mpx, n=item.n, t=item.t_capture)
        if len(d.ids) < tracker.min_detect:
            return res
        res2 = tracker.step(gray, item.t_capture, item.n, K, dist, limit, d)
        if res2.ok:
            res2.source = "recovered"
            res2.tm = {**res.tm, **res2.tm, "recover_ms": d.ms}
            return res2
        return res

    # ── thread 2: the detection, now and then ────────────────────────────
    def _detect_loop(self):
        while not self._quit.is_set():
            with self._det_cv:
                self._det_cv.wait_for(lambda: self._det_req is not None or self._quit.is_set(), timeout=POLL_S)
                req, self._det_req = self._det_req, None
                stream = self._stream
                setup = self._setup
            if req is None:
                continue
            item, box, mpx, key = req
            try:
                d = setup["detector"].detect(item.gray(), box, mpx, n=item.n, t=item.t_capture)
            except Exception as e:  # noqa: BLE001  a failed detection: the tracking asks again
                self._failed(stream, "detect", e)
                d = None
            with self._lock:
                self._det_busy = False
                if d is None or self._stream is not stream:
                    continue
                self._det_done = (d, key)
                self._det_times.append(time.perf_counter())
                self._tm_det.append(d.ms)
                if self.trace is not None:
                    self.trace.append({"what": "detect", "n": d.n, "t": item.t, "tc": item.t_capture,
                                       "corners": int(len(d.ids)), "markers": d.markers, "ms": d.ms,
                                       "box": d.box, "t_done": time.perf_counter()})

    # ── thread 3: the newest frame tracked, drawn ────────────────────────
    def _compose_loop(self):
        try:
            # the OpenGL context and the renderers made now, while the stream opens: the first frame with the overlay
            # does not wait for them (a few hundred ms)
            for name in ("face", "mandible"):
                if name in self._models and not self._quit.is_set():
                    self._renderer(name)
            while not self._quit.is_set():
                with self._job_cv:
                    self._job_cv.wait_for(lambda: self._job is not None or self._quit.is_set(), timeout=POLL_S)
                    job, self._job = self._job, None
                if job is None:
                    continue
                try:
                    self._compose(job)
                except Exception as e:  # noqa: BLE001  one bad frame: the picture keeps its last good frame
                    self._failed(job["stream"], "compose", e)
        finally:
            self._release_gl()

    def _compose(self, job):
        tc = time.perf_counter()
        item, T, vis, state, res = job["item"], job["T"], job["vis"], job["state"], job["res"]
        K, dist, cam, setup, want = job["K"], job["dist"], job["cam"], job["setup"], job["want"]
        spec, T_M_B = setup["spec"], setup["T_M_B"]
        w, h = item.size
        dw = min(self.display_width, w)
        s = dw / w
        img = item.bgr(dw, keep=False)
        dh = img.shape[0]
        Kd = scale_K(K, s, dh / h)
        if T is None:
            kind = "none"
        elif job["verdict"] in ("accepted", "restarted"):
            kind = "measured"
        else:
            kind = "fading" if state == "fading" else "holding"
        hidden = None
        if want.get("hide_board"):
            # a missed frame or two: the board covered at the pose drawn, with more margin, not the whole picture
            # blurred (that blinks); after a longer loss the pose drawn may be off the board: blurred
            age = job.get("age")
            held = kind != "measured" and T is not None and age is not None and age <= HIDE["held_s"]
            hidden = self._hide(img, T if kind == "measured" or held else None, Kd, dist, spec,
                                HIDE["held_margin_squares"] if held else HIDE["margin_squares"])
        t2 = time.perf_counter()
        drawn, masks = {}, {}
        with self._lock:        # the newest opacity, not the one when this frame was tracked: a slider is seen at once
            opacity = self._want["opacity"]
        alpha = opacity * vis
        if T is not None and T_M_B is not None and alpha > 0:
            drawn, masks = self._draw_models(img, T @ T_M_B, Kd, dist, want, alpha, job["stream"])
        t3 = time.perf_counter()
        if want["axes"] and T is not None and vis > 0.5:
            self._draw_board(img, T, Kd, dist, spec, setup, res, s)
        k = self._view_quarter(T if T is not None else res.T, setup, want, job["stream"])
        if k:
            img = cv2.rotate(img, (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)[k - 1])
        check = self.check
        if check is not None:       # the way to move the phone and the state, over the picture as it stands
            try:
                check.draw(img, k, s, T @ T_M_B if T is not None and T_M_B is not None and vis > 0 else None, Kd, dist,
                           (dw, dh))
            except Exception as e:  # noqa: BLE001  drawn without them
                self._failed(job["stream"], "check_draw", e)
        t4 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        t5 = time.perf_counter()
        tm_show = {"wait_ms": (tc - job["t_tracked"]) * 1e3, "resize_ms": (t2 - tc) * 1e3,
                   "render_ms": (t3 - t2) * 1e3, "axes_ms": (t4 - t3) * 1e3, "jpeg_ms": (t5 - t4) * 1e3,
                   "compose_ms": (t5 - tc) * 1e3}
        stream = job["stream"]
        with self._lock:
            if self._stream is not stream:
                return
            self._tm_show.append(tm_show)
            self._shown_times.append(t5)
            st = self._status_of(job["tracked"], job["rejected"], state, kind, vis, T, cam, spec, drawn, w, h, dw, dh,
                                 t5 - item.t)
            st["view_rotation"] = 90 * k
            st["display"] = [img.shape[1], img.shape[0]]
            self._status = st
            if kind == "measured" and res.ok:
                self._masks = (item.n, masks)
            if ok:
                self._jpeg, self._jpeg_n, self._bgr, self._jpeg_t = buf.tobytes(), self._jpeg_n + 1, img, t5
                self._t_good, self._fails_since_good = t5, 0
            if self.trace is not None:
                self.trace.append({"what": "shown", "n": item.n, "t": item.t, "tc": item.t_capture, "t_done": t5,
                                   "kind": kind, "state": state, "vis": vis, "T": T, "drawn": drawn, "tm": tm_show,
                                   "size": [w, h], "view": 90 * k, "hidden": hidden, "opacity": opacity})
            self._frame_ready.notify_all()

    def _hide(self, img, T, Kd, dist, spec, margin_squares=HIDE["margin_squares"]):
        """hide_board: the board's area, grown by margin_squares, filled with a dark flat colour at the pose T, or,
        without one, the whole picture blurred and dimmed, in place. Returns "board" or "blurred"."""
        h, w = img.shape[:2]
        if T is not None:
            nx, ny = spec["squares"]
            sq = spec["square_mm"]
            m = margin_squares * sq
            quad = np.array([[-m, -m, 0], [nx * sq + m, -m, 0], [nx * sq + m, ny * sq + m, 0], [-m, ny * sq + m, 0]],
                            np.float64)
            if np.all(marker_pose.transform(T, quad)[:, 2] > 1.0):
                p = marker_pose.project(quad, T, Kd, dist)
                if np.all(np.isfinite(p)) and np.all(np.abs(p) < 8 * max(w, h)):
                    cv2.fillConvexPoly(img, np.round(p * 16).astype(np.int32), self.colors["hidden"], cv2.LINE_AA, 4)
                    return "board"
        k = HIDE["shrink"]
        small = cv2.resize(img, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
        img[:] = cv2.convertScaleAbs(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), alpha=HIDE["dim"])
        return "blurred"

    def _view_quarter(self, T, setup, want, stream):
        """Quarter turns (clockwise) of the composed frame so that it stands the way the phone is held, as in the
        phone's own camera app: the frames come as the sensor gives them, the pose stays in the sensor's frame (the
        intrinsics never change) and only the picture is turned. The way up is the model's: the top of the head
        (+z of the bench frame through the registration), which on the bench and on a board on a monitor is up in
        the room; it changes only once the picture would stand more than 70 degrees off (no flapping near 45) and
        stays when the camera looks along it. Before the first pose: the phone held upright (the sensor's mounting, 90 degrees on the Poco X7
        Pro). view_rotation set to 0, 90, 180 or 270 fixes it."""
        fixed = want.get("view_rotation")
        if fixed in (0, 90, 180, 270):
            return fixed // 90
        if T is not None:
            u = T[:3, :3] @ setup["up_M"]
            if np.hypot(u[0], u[1]) > 0.35:
                theta = np.degrees(np.arctan2(u[1], u[0]))       # the way up in the image (x right, y down)

                def off(k):                                     # how far a turn of k quarters leaves it from up
                    return abs((theta + 90 * k + 90 + 180) % 360 - 180)
                if self._view_k is None or off(self._view_k) > 70:
                    self._view_k = int(min(range(4), key=off))
        if self._view_k is None:
            return int(getattr(stream, "sensor_orientation", 0) or 0) // 90 % 4
        return self._view_k

    def _draw_board(self, img, T, Kd, dist, spec, setup, res, s):
        """Board axes at the drawn pose and the corners where the tracker has them on this frame (dots): only what is
        in front of the camera and inside the image, so a pose seen edge on or a board half out of the frame cannot
        throw the drawing off."""
        h, w = img.shape[:2]
        L = 3 * spec["square_mm"]
        ends = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], np.float64)
        if np.all(marker_pose.transform(T, ends)[:, 2] > 1.0):
            p = marker_pose.project(ends, T, Kd, dist)
            if np.all(np.isfinite(p)) and np.all(np.abs(p) < 4 * max(w, h)):
                for k, c in ((1, (60, 60, 230)), (2, (60, 200, 60)), (3, (230, 120, 40))):
                    self._arrow(img, p[0], p[k], c)
        if res is not None and res.ok and len(res.corners):
            c = (res.corners + 0.5) * s - 0.5
            inside = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < w) & (c[:, 1] < h)
            for x, y in np.round(c[inside] * 16).astype(np.int32):
                cv2.circle(img, (int(x), int(y)), 2 * 16, self.colors["corners"], -1, cv2.LINE_AA, 4)

    @staticmethod
    def _arrow(img, a, b, colour):
        """An axis from a to b (px of img, pixel centres at whole numbers) ending in the arrowhead of every drawing
        (arrowhead.fill: each pixel by the share the triangle covers); the line stops inside the triangle
        (arrowhead.inset), so the two are one shape. An axis seen nearly end on is a line with no tip."""
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        d = b - a
        n = float(np.hypot(*d))
        if n < 1e-6:
            return
        u = d / n
        length = min(ARROW["length_px"], ARROW["max_share"] * n)
        lw = ARROW["line_px"]
        if length < ARROW["min_px"]:
            cv2.line(img, tuple(np.round(a * 16).astype(int)), tuple(np.round(b * 16).astype(int)), colour, lw,
                     cv2.LINE_AA, 4)
            return
        stop = b - u * arrowhead.inset(length, lw)
        cv2.line(img, tuple(np.round(a * 16).astype(int)), tuple(np.round(stop * 16).astype(int)), colour, lw,
                 cv2.LINE_AA, 4)
        arrowhead.fill(img, b, u, length, colour)

    def _status_of(self, tracked, rejected, state, kind, vis, T, cam, spec, drawn, w, h, dw, dh, processing):
        """The status after a frame was drawn (under the lock); tracked and rejected as the track thread copied them
        from its filter."""
        def mean(rows, key):
            vals = [r[key] for r in rows if key in r]
            return round(float(np.mean(vals)), 1) if vals else 0.0
        tm = {k: mean(self._tm_track, k) for k in ("gray_ms", "track_ms", "flow_ms", "filter_ms")}
        tm["detect_ms"] = round(float(np.mean(self._tm_det)), 1) if self._tm_det else 0.0
        tm["pose_ms"] = tm["track_ms"]
        tm.update({k: mean(self._tm_show, k) for k in ("wait_ms", "resize_ms", "render_ms", "axes_ms", "jpeg_ms",
                                                         "compose_ms")})
        tm["draw_ms"] = round(tm["resize_ms"] + tm["axes_ms"] + tm["jpeg_ms"], 1)
        tm["total_ms"] = round(tm["gray_ms"] + tm["track_ms"] + tm["filter_ms"] + tm["compose_ms"], 1)

        def rate(times):
            return round((len(times) - 1) / (times[-1] - times[0]), 1) if len(times) > 1 and times[-1] > times[0] else 0.0
        v = list(self._verdicts)
        latest = self._latest or {}
        st = {"tracked": tracked, "state": state, "kind": kind, "opacity_now": round(float(vis), 2),
              "corners": latest.get("corners", 0), "markers": latest.get("markers", 0),
              "rms_px": None if latest.get("rms_px") is None else round(latest["rms_px"], 3),
              "rms_limit_px": MAX_RMS[cam["source"]], "min_corners": board_tracker.TRACK["min_corners"],
              "intrinsics": cam, "drawn_px": drawn, "timings": tm, "image": [w, h], "display": [dw, dh],
              "fps_ar": rate(self._shown_times), "fps_pose": rate(self._tracked_times),
              "fps_detect": rate(self._det_times), "source": latest.get("source"), "sure": latest.get("sure"),
              "tracking": latest.get("counts", {}), "skipped": self._skipped,
              "missed_share": round(sum(x == "missed" for x in v) / len(v), 3) if v else 0.0,
              "rejected": rejected}
        if T is not None:
            nx, ny = spec["squares"]
            centre = np.array([nx, ny, 0.0]) * [spec["square_mm"], spec["square_mm"], 0] / 2
            st["distance_mm"] = round(float(np.linalg.norm(marker_pose.transform(T, centre[None])[0])), 1)
            st["tilt_deg"] = round(float(np.degrees(np.arccos(min(1.0, abs(T[2, 2]))))), 1)
        stream_ms = 0.0 if self.synthetic or self.replay is not None else STREAM_LATENCY_MS.get(self._link)
        st["latency"] = {"stream_ms": stream_ms, "processing_ms": round(processing * 1e3, 1),
                         "total_ms": round((stream_ms or 0.0) + processing * 1e3, 1), "link": self._link}
        return st

    def _failed(self, stream, where, error):
        """A frame (or a stage of one) that failed at `where`: counted and logged (at most every LOG_EVERY_S); the
        threads go on with the next frame and the picture keeps its last good one."""
        now = time.perf_counter()
        with self._lock:
            if stream is not None and self._stream is not stream:
                return
            self._frame_errors += 1
            self._fails_since_good += 1
            self._last_frame_error = {"where": where, "what": type(error).__name__, "t": now}
            count = self._frame_errors
            log_now = now - self._t_logged >= LOG_EVERY_S
            if log_now:
                self._t_logged = now
        if log_now:
            logger.warning("live AR: %s failed (%d in this run): %s: %s", where, count, type(error).__name__, error)
            logger.debug("live AR: %s failed", where, exc_info=error)
            self.log(f"{where} failed ({count} in this run): {type(error).__name__}: {error}")

    def _draw_models(self, img, T_C_B, Kd, dist, want, alpha, stream=None):
        """The mandible and the face (those asked for) over the picture, camera from model T_C_B, at opacity alpha
        (the setting times the fade): ({model: pixels covered}, {model: silhouette at the display size}). A model
        that fails to draw is left out of this frame (see _render_failed); the others are drawn."""
        drawn, masks = {}, {}
        size = (img.shape[1], img.shape[0])
        for name in ("mandible", "face"):
            rend = self._renderer(name) if want[name] and name in self._models else None
            if rend is None:
                continue
            st = STYLE[name]
            a = min(1.0, st["fill"] * float(alpha))      # a Python float: a NumPy one would make the weights float64
            try:
                if hasattr(rend, "overlay"):       # OpenGL: coverage of each pixel and a light per vertex
                    light = 1.0 - st["shade"] * (1.0 - rend.facing(T_C_B)) ** SHADE_GAMMA if st["shade"] > 0 else None
                    r = rend.overlay(Kd, dist, T_C_B, size, light)
                    self._paint_veil(img, r, self.colors[name], a)
                else:                               # NumPy: hard mask and a light per triangle
                    r = rend.render(Kd, dist, T_C_B, size)
                    shade = self._shade(name, rend, T_C_B, st["shade"]) if st["shade"] > 0 else None
                    self._paint(img, r, self.colors[name], a, shade, SOFT["feather_px"], SOFT["smooth_px"])
            except Exception as e:  # noqa: BLE001  this model left out of this frame; the others drawn
                self._failed(stream, f"render_{name}", e)
                self._render_failed(name, rend, e)
                continue
            self._render_fails[name] = 0
            drawn[name] = int(cv2.countNonZero(r.mask))
            masks[name] = r.mask
        return drawn, masks

    @staticmethod
    def _paint_veil(img, o, colour, a):
        """A model (gl_overlay.Overlay) over the picture as a veil of opacity a: the colour times the light, each
        pixel weighted by the share of it the model covers. Nothing is drawn on the edge: its anti-aliasing is that
        coverage, so the silhouette fades with the veil and stays steady under a sub-pixel motion."""
        bh, bw = o.cover.shape
        if not (bh and bw) or a <= 0:
            return
        x0, y0 = o.offset
        roi = img[y0:y0 + bh, x0:x0 + bw]
        w = o.cover.astype(np.float32) * np.float32(a / 255)
        light = cv2.divide(o.light, o.cover, scale=255)       # the light itself (0 where nothing is covered)
        tint = cv2.multiply(cv2.cvtColor(light, cv2.COLOR_GRAY2BGR), tuple(c / 255 for c in colour) + (0,))
        roi[:] = cv2.blendLinear(roi, tint, 1.0 - w, w)

    def _shade(self, name, rend, T, depth):
        """Brightness of each triangle under a light at the camera (both sides lit), as in STYLE and SHADE_GAMMA, for
        the NumPy renderer (one value per triangle)."""
        if name not in self._centroids:
            self._centroids[name] = rend.v[rend.f].mean(1)
        c = self._centroids[name] @ T[:3, :3].T + T[:3, 3]
        cos = np.abs(np.sum((rend.n @ T[:3, :3].T) * c, 1)) / np.maximum(np.linalg.norm(c, axis=1), 1e-9)
        # 0 first: the visibility buffer holds -1 for the background, so buffer + 1 indexes this table directly
        return np.concatenate([[0], np.rint(255 * (1 - depth * (1 - cos) ** SHADE_GAMMA))]).astype(np.uint8)

    @staticmethod
    def _paint(img, r, colour, a, shade=None, feather=0.8, smooth=2.0):
        """The veil of _paint_veil from a hard mask and a flat light per triangle (the NumPy renderer, without
        OpenGL): against shimmer the silhouette is feathered by about a pixel (feather, px) and the shading blurred
        inside it (smooth, px; normalised, so the edge does not darken)."""
        bh, bw = r.faces.shape
        if not (bh and bw) or a <= 0:
            return
        x0, y0 = r.offset
        H, W = img.shape[:2]
        pad = int(np.ceil(3 * max(feather, 0.5)))
        xa, ya, xb, yb = max(0, x0 - pad), max(0, y0 - pad), min(W, x0 + bw + pad), min(H, y0 + bh + pad)
        roi = img[ya:yb, xa:xb]
        m = r.mask[ya:yb, xa:xb]
        if not cv2.countNonZero(m):
            return
        soft = cv2.GaussianBlur(m, (0, 0), feather) if feather > 0 else m
        w = soft.astype(np.float32) * np.float32(a / 255)
        if shade is None:
            light = np.full(m.shape, 255, np.uint8)
        else:
            sh = np.zeros(m.shape, np.uint8)
            sh[y0 - ya:y0 - ya + bh, x0 - xa:x0 - xa + bw] = shade[r.faces + 1]
            if smooth > 0:      # blurred inside the silhouette only: sum of shade over sum of mask
                num = cv2.GaussianBlur(cv2.bitwise_and(sh, m), (0, 0), smooth)
                den = cv2.GaussianBlur(m, (0, 0), smooth)
                light = cv2.divide(num, cv2.max(den, 1), scale=255)
            else:
                light = sh
        tint = cv2.multiply(cv2.cvtColor(light, cv2.COLOR_GRAY2BGR), tuple(c / 255 for c in colour) + (0,))
        roi[:] = cv2.blendLinear(roi, tint, 1.0 - w, w)


# ── the frames for the page ──────────────────────────────────────────────────
class MjpegServer:
    """The composed frames as MJPEG (multipart/x-mixed-replace) on 127.0.0.1, for an <img> of the page or a browser.
    The path carries a random token (compared in constant time), so only who was given the URL reads the camera; at
    most MJPEG_CLIENTS at once, each socket with a timeout. A client's stream ends when the live AR is shut down."""

    def __init__(self, ar, port=0):
        self.ar = ar
        self.token = secrets.token_hex(8)
        self._slots = threading.BoundedSemaphore(MJPEG_CLIENTS)
        server = self

        class Handler(BaseHTTPRequestHandler):
            timeout = MJPEG_SOCKET_S

            def log_message(self, *a):
                pass

            def do_GET(self):
                if not server._slots.acquire(blocking=False):
                    self.send_error(503)
                    return
                try:
                    self._serve(self.path.split("?", 1)[0])
                finally:
                    server._slots.release()

            def _serve(self, path):
                if _same_path(path, f"/{server.token}.jpg"):
                    jpeg, _ = server.ar.next_jpeg(0, timeout=1.0)
                    if jpeg is None:
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                if not _same_path(path, f"/{server.token}.mjpg"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                n = 0
                try:
                    while not server.closed and not getattr(server.ar, "closed", False):
                        jpeg, n = server.ar.next_jpeg(n, timeout=1.0)
                        if jpeg is None:
                            continue
                        t0 = time.perf_counter()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                        if hasattr(server.ar, "note_served"):
                            server.ar.note_served((time.perf_counter() - t0) * 1e3)
                except (ConnectionError, OSError):
                    pass

        self.closed = False
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="mjpeg")
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/{self.token}.mjpg"

    @property
    def still_url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/{self.token}.jpg"

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=JOIN_S)


def _same_path(path, want):
    return hmac.compare_digest(path.encode("utf-8", "replace"), want.encode("utf-8"))


# ── measures ─────────────────────────────────────────────────────────────────
def model_points(ar, k=400):
    """About k points of the face (or of the board) in M, for the errors in the image."""
    setup = ar._setup
    T = setup["T_M_B"]
    if T is not None and "face" in ar._models:
        v = ar._models["face"][0]
        v = v[np.linspace(0, len(v) - 1, min(k, len(v))).astype(int)]
        return v @ T[:3, :3].T + T[:3, 3]
    return setup["board"].getChessboardCorners().astype(np.float64)


def _noise(P):
    """White noise of a track of points P (frames x points x dims) from its second differences: a smooth motion
    (a hand that holds the phone, a pan) adds almost nothing, a tremble of the overlay all of it. RMS, units of P."""
    if len(P) < 3:
        return None
    d2 = P[1:-1] - (P[:-2] + P[2:]) / 2
    return float(np.sqrt(np.mean(np.sum(d2 ** 2, axis=-1)) / 1.5))


def tremble(rows, X, K, seconds=TREMBLE_S, now_t=None):
    """How much the overlay shakes from frame to frame over the last `seconds` of rows (capture time, pose drawn, pose
    found; None where there is none): the white noise of the points X projected (second differences, as _noise), px
    at the size of K, over runs of frames that follow each other at the stream's pace. Returns (drawn, found), each
    rounded to 0.01 px or None with fewer than 5 frames in such runs."""
    if not rows:
        return None, None
    t_end = rows[-1][0] if now_t is None else now_t
    rows = [r for r in rows if t_end - r[0] <= seconds]
    times = np.array([r[0] for r in rows])
    if len(times) < 5:
        return None, None
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    out = []
    for col in (1, 2):
        d2, frames, run = [], 0, []
        for k, r in enumerate(rows + [(None, None, None)]):
            T = r[col]
            if T is not None and run and r[0] - rows[run[-1]][0] <= 1.6 * step:
                run.append(k)
                continue
            if len(run) >= 3:
                P = np.array([marker_pose.project(X, rows[j][col], K) for j in run])
                d2.append(P[1:-1] - (P[:-2] + P[2:]) / 2)
                frames += len(run)
            run = [k] if T is not None else []
        if frames < 5:
            out.append(None)
            continue
        D = np.concatenate(d2)
        out.append(round(float(np.sqrt(np.mean(np.sum(D ** 2, axis=-1)) / 1.5)), 2))
    return out[0], out[1]


def _jumps(P, t, floor_px=15.0, factor=5.0):
    """Jumps of the overlay: frames where it moves (RMS over its points) more than floor_px and factor times the
    median move of the frames around it (a fast pan moves it a lot, but evenly). Returns (count, per minute)."""
    if len(P) < 5:
        return 0, 0.0
    d = np.sqrt(np.mean(np.sum((P[1:] - P[:-1]) ** 2, axis=-1), axis=1))
    k = 15
    med = np.array([np.median(d[max(0, i - k):i + k + 1]) for i in range(len(d))])
    jump = d > np.maximum(floor_px, factor * med)
    events = int(np.sum(jump[1:] & ~jump[:-1]) + jump[0])
    minutes = max((t[-1] - t[0]) / 60.0, 1e-9)
    return events, events / minutes


def measure_run(X, rows, K, segments=None, truth=None, clip_time=None):
    """Numbers of a run from its trace: rates, time per stage, latency (decoded frame to JPEG), frames without a
    pose, blinks, jumps of the overlay per minute, its tremble (noise from second differences, px at the stream size
    and mm, as measured and as drawn) and, with the truth (truth(n) -> T_C_M, or a reference pose for the frames of a
    recorded clip), its error in the image. segments: {name: (from, to)} in s of clip_time(row) (default: time since
    the first row), each with the same numbers. Works on traces of this version (rows track, detect, shown) and of
    the one before (rows found, shown)."""
    shown = sorted((r for r in rows if r["what"] == "shown"), key=lambda r: r["n"])
    raw = sorted((r for r in rows if r["what"] in ("track", "found")), key=lambda r: r["n"])
    dets = [r for r in rows if r["what"] == "detect"]
    t_first = min((r["t"] for r in rows), default=0.0)
    ct = clip_time or (lambda r: r["t"] - t_first)

    def mean_tm(rr, key):
        v = [r["tm"][key] for r in rr if r.get("tm") and key in r["tm"]]
        return float(np.mean(v)) if v else None

    def numbers(sh, rw, dt):
        out = {}
        if len(sh) > 1:
            ts = np.array([r["t_done"] for r in sh])
            out["fps_video"] = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-9)
            lat = [(r["t_done"] - r["t"]) * 1e3 for r in sh]
            out["latency_ms"] = float(np.median(lat))
            out["latency_p95_ms"] = float(np.percentile(lat, 95))
            for k in ("wait_ms", "resize_ms", "render_ms", "axes_ms", "jpeg_ms", "compose_ms"):
                out[k] = mean_tm(sh, k)
            vis = np.array([r["vis"] > 0 for r in sh])
            out["blinks"] = int(np.sum(vis[:-1] & ~vis[1:]))
            out["drawn_share"] = float(np.mean(vis))
            out["kinds"] = {k: int(sum(r["kind"] == k for r in sh)) for k in ("measured", "predicted", "holding",
                                                                            "fading", "none")}
            dr = [r for r in sh if r["T"] is not None]
            if len(dr) > 4:
                P = np.array([marker_pose.project(X, r["T"], K) for r in dr])
                Q = np.array([marker_pose.transform(r["T"], X) for r in dr])
                out["jitter_drawn_px"], out["jitter_drawn_mm"] = _noise(P), _noise(Q)
                out["jumps"], out["jumps_per_min"] = _jumps(P, np.array([r["t"] for r in dr]))
        if len(rw) > 1:
            tf = np.array([r["t_done"] for r in rw])
            out["fps_pose"] = (len(tf) - 1) / max(tf[-1] - tf[0], 1e-9)
            ok = [r for r in rw if r.get("ok", r.get("found"))]
            out["no_pose_share"] = 1.0 - len(ok) / len(rw)
            for k in ("detect_ms", "pose_ms", "gray_ms", "track_ms", "flow_ms", "filter_ms"):
                out[k] = mean_tm(rw, k)
            Tk = "T_raw" if "T_raw" in rw[0] else "T"
            okT = [r for r in ok if r.get(Tk) is not None]
            if len(okT) > 4:
                P = np.array([marker_pose.project(X, r[Tk], K) for r in okT])
                Q = np.array([marker_pose.transform(r[Tk], X) for r in okT])
                out["jitter_found_px"], out["jitter_found_mm"] = _noise(P), _noise(Q)
                out["jumps_found"], out["jumps_found_per_min"] = _jumps(P, np.array([r["t"] for r in okT]))
        if dt:
            out["detections_per_s"] = len(dt) / max((max(r["t"] for r in dt) - min(r["t"] for r in dt)), 1e-9)
            out["detect_ms"] = float(np.mean([r["ms"] for r in dt]))
        if truth is not None:
            def err(rr, key):
                e = []
                for r in rr:
                    Tt = truth(r["n"])
                    if Tt is not None and r.get(key) is not None:
                        e.append(np.sqrt(np.mean(np.sum((marker_pose.project(X, r[key], K)
                                                         - marker_pose.project(X, Tt, K)) ** 2, axis=1))))
                return np.array(e)
            Tk = "T_raw" if rw and "T_raw" in rw[0] else "T"
            for name, rr, key in (("found", [r for r in rw if r.get("ok", r.get("found"))], Tk),
                                  ("drawn", [r for r in sh if r["kind"] in ("measured", "predicted")], "T"),
                                  ("drawn_predicted", [r for r in sh if r["kind"] == "predicted"], "T"),
                                  ("drawn_held", [r for r in sh if r["kind"] in ("holding", "fading")], "T")):
                e = err(rr, key)
                if len(e):
                    out[f"error_{name}_px"] = (float(np.median(e)), float(np.percentile(e, 95)))
        return out

    out = numbers(shown, raw, dets)
    if segments:
        out["segments"] = {}
        for name, (a, b) in segments.items():
            def inside(r):
                c = ct(r)
                return c is not None and a <= c < b
            out["segments"][name] = numbers([r for r in shown if inside(r)], [r for r in raw if inside(r)],
                                            [r for r in dets if inside(r)])
    return out


def run_bench(ar, seconds, segments=None, warmup=1.0, reference=None):
    """Runs the live AR for `seconds` with the trace on and an MJPEG client reading the frames; returns the numbers of
    measure_run, of the client (frames received per second) and the CPU used (cores, all threads of this process).
    On a recorded clip, segments and reference (T_C_M per recorded frame, NaN where none) go by the frame's index in
    the recording."""
    import urllib.request
    ar.trace = []
    ar.start()
    stream = ar.stream()
    t0 = time.perf_counter()
    c0 = time.process_time()
    server = MjpegServer(ar)
    got = []

    def client():
        try:
            with urllib.request.urlopen(server.url, timeout=5) as r:
                while time.perf_counter() - t0 < seconds + warmup:
                    line = r.readline()
                    if line.startswith(b"Content-Length:"):
                        r.readline()
                        r.read(int(line.split(b":")[1]))
                        got.append(time.perf_counter())
        except OSError:
            pass
    th = threading.Thread(target=client, daemon=True)
    th.start()
    time.sleep(seconds + warmup)
    cpu = (time.process_time() - c0) / (time.perf_counter() - t0)
    rows = [r for r in ar.trace if r["t"] - t0 >= warmup]
    st = ar.status()
    ar.trace = None
    th.join(timeout=2.0)
    server.close()
    index = dict(getattr(stream, "_index", {}))
    rec = getattr(stream, "_times_rec", None)
    ar.stop()
    w, h = st["image"] if "image" in st else ar._size_wh()
    K = stream_intrinsics((w, h), ar.intrinsics_path)[0]
    truth = getattr(stream, "truth", None)
    if reference is not None:
        def truth(n):
            k = index.get(n)
            if k is None or k >= len(reference) or not np.isfinite(reference[k][0, 0]):
                return None
            return reference[k]
    clip_time = None
    if rec is not None:
        def clip_time(r):
            k = index.get(r["n"])
            return None if k is None or k >= len(rec) else float(rec[k])
    out = measure_run(model_points(ar), rows, K, segments, truth, clip_time)
    g = np.array([x for x in got if x - t0 >= warmup])
    out["fps_page"] = (len(g) - 1) / (g[-1] - g[0]) if len(g) > 1 else 0.0
    out["decode_ms"] = st.get("timings", {}).get("decode_ms")
    out["serve_ms"] = st.get("timings", {}).get("serve_ms")
    out["fps_stream"] = st.get("fps_stream")
    out["cpu_cores"] = cpu
    return out


def print_bench(out, title):
    def f(k, fmt="{:.1f}", o=None):
        v = (o or out).get(k)
        if v is None:
            return "-"
        if isinstance(v, tuple):
            return " / ".join(fmt.format(x) for x in v)
        return fmt.format(v)
    print(f"\n{title}")
    print(f"  rate: stream {f('fps_stream')} fps, video {f('fps_video')} fps, received by the page {f('fps_page')} fps, "
          f"poses {f('fps_pose')} fps, detections {f('detections_per_s')} per s; CPU {f('cpu_cores', '{:.2f}')} cores")
    print(f"  per frame: decode {f('decode_ms')} ms, gray {f('gray_ms', '{:.2f}')} ms, tracking {f('track_ms')} ms "
          f"(flow {f('flow_ms')}), detection {f('detect_ms')} ms, filter {f('filter_ms', '{:.2f}')} ms, "
          f"pose {f('pose_ms', '{:.2f}')} ms, image {f('resize_ms')} ms, models drawn {f('render_ms')} ms, "
          f"axes {f('axes_ms')} ms, JPEG {f('jpeg_ms')} ms, to the page {f('serve_ms', '{:.2f}')} ms")
    print(f"  decoded frame to JPEG: median {f('latency_ms')} ms, 95th percentile {f('latency_p95_ms')} ms")
    print(f"  frames without a pose: {f('no_pose_share', '{:.1%}')}; overlay drawn on {f('drawn_share', '{:.1%}')} of "
          f"the frames, blinks {f('blinks', '{:d}')}; frames drawn: {out.get('kinds')}")

    def quality(o, pre="  "):
        print(f"{pre}tremble (noise): as found {f('jitter_found_px', '{:.3f}', o)} px {f('jitter_found_mm', '{:.3f}', o)} "
              f"mm, as drawn {f('jitter_drawn_px', '{:.3f}', o)} px {f('jitter_drawn_mm', '{:.3f}', o)} mm; jumps "
              f"drawn {f('jumps_per_min', '{:.1f}', o)} per min ({f('jumps', '{:d}', o)}), as found "
              f"{f('jumps_found_per_min', '{:.1f}', o)} per min")
        if "error_found_px" in o:
            print(f"{pre}overlay against the reference (median / 95th percentile, px): as found "
                  f"{f('error_found_px', '{:.2f}', o)}; drawn {f('error_drawn_px', '{:.2f}', o)}; held "
                  f"{f('error_drawn_held_px', '{:.2f}', o)}")
    quality(out)
    for name, o in (out.get("segments") or {}).items():
        print(f"  [{name}] video {f('fps_video', o=o)} fps, latency {f('latency_ms', o=o)} ms, no pose "
              f"{f('no_pose_share', '{:.0%}', o)}, drawn {f('drawn_share', '{:.0%}', o)}, blinks {f('blinks', '{:d}', o)}")
        quality(o, "      ")


def clip_reference(clip, board, spec, K, dist, cache=True):
    """A reference pose for every frame of a recorded clip: the full ChArUco detection at full resolution, PnP, then
    smoothed without lag (forward and backward) over runs of frames with a pose: the board's centre at 6 Hz, the
    rotation at 3 Hz (a hand turns the phone slowly; a small board's rotation is its noisiest part). T_C_M per frame
    (NaN where none), kept beside the clip (CLIP_reference_<board>.npz), or in LIVE_REFERENCES for a clip inside the
    project (its data folders are read-only)."""
    from scipy.signal import butter, filtfilt
    clip = Path(clip)
    name = clip.stem + f"_reference_{spec['name']}_{spec['squares'][0]}x{spec['squares'][1]}_v2.npz"
    beside = clip.with_name(name)
    out = LIVE_REFERENCES / name if ROOT in clip.resolve().parents else beside
    for kept in (beside, out):
        if cache and kept.is_file():
            return np.load(kept)["T_C_M"]
    import av
    det = marker_pose.make_detector(board)
    codec = av.CodecContext.create("h264", "r")
    poses = []
    with open(clip, "rb") as fh:
        while True:
            data = fh.read(1 << 20)
            for packet in codec.parse(data):
                for frame in codec.decode(packet):
                    c, ids, _ = marker_pose.detect(det, frame.to_ndarray(format="gray"))
                    e = marker_pose.estimate_pose(board, c, ids, K, dist,
                                                  min_corners=8 if spec["name"] == "sheet" else 6)
                    poses.append(e["T"] if e is not None and e["rms_px"] < 2.0 else np.full((4, 4), np.nan))
            if not data:
                break
    P = np.array(poses)
    # smoothing: anchor at the board centre (translation) and rotation vector, on runs of 10 or more frames
    ok = np.isfinite(P[:, 0, 0])
    b, a = butter(2, 6.0 / 15.0)
    br, ar_ = butter(2, 3.0 / 15.0)
    nx, ny = spec["squares"]
    centre = np.array([nx, ny, 0.0]) * spec["square_mm"] / 2
    ref = np.full_like(P, np.nan)
    k = 0
    while k < len(P):
        if not ok[k]:
            k += 1
            continue
        j = k
        while j < len(P) and ok[j]:
            j += 1
        if j - k >= 10:
            R0 = P[k, :3, :3]
            w = np.array([cv2.Rodrigues(R0.T @ P[i, :3, :3])[0].ravel() for i in range(k, j)])
            c = np.array([P[i, :3, :3] @ centre + P[i, :3, 3] for i in range(k, j)])
            pad = min(9, j - k - 1)
            w, c = filtfilt(br, ar_, w, axis=0, padlen=pad), filtfilt(b, a, c, axis=0, padlen=pad)
            for i in range(k, j):
                ref[i] = np.eye(4)
                ref[i, :3, :3] = R0 @ cv2.Rodrigues(w[i - k])[0]
                ref[i, :3, 3] = c[i - k] - ref[i, :3, :3] @ centre
        k = j
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, T_C_M=ref)
    return ref


# ── command line ─────────────────────────────────────────────────────────────
def cmd_view(a):
    ar = LiveAR(a.board, a.square, a.registration, mandible=a.mandible, synthetic=a.synthetic, log=print,
                replay=a.replay, intrinsics=a.intrinsics or STREAM_INTRINSICS, squares=a.squares, marker_mm=a.marker)
    try:
        ar.start()
        n = 0
        while True:
            jpeg, n = ar.next_jpeg(n, timeout=0.5)
            img = ar.composed()
            if img is not None:
                st = ar.status()
                txt = (f"{'TRACKED' if st.get('tracked') else 'no board'}  corners {st.get('corners', 0)}  "
                       f"RMS {st.get('rms_px')}  stream {st.get('fps_stream')} fps  video {st.get('fps_ar')} fps  "
                       f"poses {st.get('fps_pose')} fps  {st['intrinsics']['source']} intrinsics  "
                       f"registration {st['registration'].get('source')}")
                cv2.imshow("live AR", img)
                cv2.setWindowTitle("live AR", txt)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if ar.status().get("error"):
                print(ar.status()["error"])
                break
    finally:
        ar.shutdown()
        cv2.destroyAllWindows()


def make_clip(out, board="sheet", square_mm=24.0, seconds=24.0, fps=30, exposure_s=0.008, bit_rate=live.BIT_RATE,
              noise=2.0):
    """The synthetic session (bench_script) as the phone would send it: raw H.264 at the stream's bit rate (libx264,
    no B frames, a key frame every 10 s), OUT.h264, with OUT_times.npy (the time of each frame) and OUT_truth.npz (the
    true T_C_M of each frame and the camera K it was rendered with, nominal and without distortion), so that any
    version of the live AR can be measured on the same frames."""
    from fractions import Fraction

    import av
    out = Path(out).with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    b, spec = make_board(board, square_mm)
    K, _, _ = stream_intrinsics((1920, 1440), None)
    s = SyntheticStream(b, spec, K, script=bench_script(spec), exposure_s=exposure_s, noise=noise, seed=1)
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = 1920, 1440, "yuv420p"
    enc.time_base = Fraction(1, fps)
    enc.framerate = Fraction(fps, 1)
    enc.bit_rate = bit_rate
    enc.options = {"bf": "0", "g": str(10 * fps), "preset": "veryfast", "tune": "zerolatency"}
    truth = []
    with open(str(out) + ".h264", "wb") as fh:
        for k in range(int(round(seconds * fps))):
            img, T = s.render(k / fps, k)
            frame = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), format="bgr24").reformat(
                format="yuv420p")
            frame.pts = k
            for packet in enc.encode(frame):
                fh.write(bytes(packet))
            truth.append(T)
        for packet in enc.encode(None):
            fh.write(bytes(packet))
    np.save(str(out) + "_times.npy", np.arange(len(truth)) / fps)
    np.savez_compressed(str(out) + "_truth.npz", T_C_M=np.array(truth), K=K)
    return Path(str(out) + ".h264")


def cmd_bench(a):
    """Numbers of the live AR on the synthetic session (bench_script) or on a recorded clip."""
    board, spec = make_board(a.board, a.square, a.squares, a.marker)
    replay = a.replay
    ar = LiveAR(a.board, a.square, a.registration, mandible=a.mandible, synthetic=replay is None, replay=replay,
                intrinsics=a.intrinsics or STREAM_INTRINSICS, script=bench_script(spec), exposure_s=0.008, log=print,
                squares=a.squares, marker_mm=a.marker)
    if replay is not None:
        ar.fps = 30
    if a.segments:
        segments = {k: tuple(v) for k, v in json.loads(Path(a.segments).read_text(encoding="utf-8")).items()}
    elif replay is None:
        segments = {"still": (1.0, 6.0), "slow": (6.0, 12.0), "fast": (12.0, 18.0)}
    else:
        segments = {"still": (1.0, a.still)} if a.still else None
    reference = None
    if replay is not None and a.reference:
        K, dist, _ = stream_intrinsics((1920, 1440), a.intrinsics or STREAM_INTRINSICS)
        reference = clip_reference(replay, board, spec, K, dist)
    try:
        out = run_bench(ar, a.seconds, segments, warmup=0.0, reference=reference)
    finally:
        ar.shutdown()
    print_bench(out, f"live AR on {'the synthetic session' if replay is None else replay}, {a.seconds:.0f} s, "
                     f"{a.board} board")


def twin_frames(session=None):
    """(frames, image paths, label paths) of the digital twin (synth_twin.py): the pre-impact frames of the first
    registration of a session, which share camera and registration. None when the twin is not on this computer."""
    found = sorted(TWIN.glob("s*/frames.jsonl")) if session is None else [TWIN / session / "frames.jsonl"]
    if not found or not found[0].is_file():
        return None
    rows = [json.loads(line) for line in found[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    first = rows[0]
    same = [r for r in rows if r["state"] == 0 and r["registration"] == first["registration"] and r["K"] == first["K"]
            and r["size"] == first["size"] and r["T_M_B"] == first["T_M_B"]]
    d = found[0].parent
    return same, [d / (r["image"] + ".jpg") for r in same], [d / (r["image"] + "_label.png") for r in same]


def twin_check(ar, expect, out=None):
    """The frames of the digital twin played through the live AR: the patient board as the phone sees it on the
    bench, rendered by Blender with a known camera, board pose and registration. With the true registration loaded
    from a file, the face drawn must cover the face Blender labelled (recall) and stay on the face or on what is in
    front of it (precision); the twin's skin is deformed by up to 4 mm, the overlay is the undeformed phantom. The
    frames are separate views, so each is drawn at its own pose (no smoothing)."""
    import tempfile
    got = twin_frames()
    if got is None:
        print(f"skip  frames of the digital twin not found in {TWIN}", flush=True)
        return
    rows, images, labels = got
    saved = ar.replay, ar.replay_truth, ar.intrinsics_path, ar.settings(), dict(ar.tracking)
    checks, st = {}, {"registration": {}, "intrinsics": {}}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            intr, reg = Path(tmp) / "intrinsics_twin.json", Path(tmp) / "registration_twin.json"
            r0 = rows[0]
            intr.write_text(json.dumps({"image_size": r0["size"], "K": r0["K"], "dist": r0["dist"], "date": "twin",
                                        "rms_px": 0.0}), encoding="utf-8")
            reg.write_text(json.dumps({"T_M_B": r0["T_M_B"]}), encoding="utf-8")
            ar.replay, ar.replay_truth, ar.intrinsics_path = images, list(zip(rows, labels)), intr
            ar.set_registration(reg)
            ar.configure(board="patient", mandible=False, face=True)
            ar.set_tracking(smooth=False, acquire_frames=1, fade_in_s=0.0, views=True, wait_sure=False)
            ar.start()
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 2.0 + 2.5 * len(images) / 10 and len(checks) < len(images):
                time.sleep(0.02)
                n, masks = ar.last_masks()
                pn, T = ar.last_pose()
                if n in checks or pn != n or "face" not in masks:
                    continue
                meta, label = ar.stream().truth(n)
                m = masks["face"] > 0
                lab = cv2.resize(cv2.imread(str(label), cv2.IMREAD_GRAYSCALE), (m.shape[1], m.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
                face = lab == 1
                front = np.isin(lab, (1, 2, 3, 4, 7, 9))      # face, marker, drape, interface, pendulum, cap
                e = marker_pose.pose_error(T, np.asarray(meta["T_C_M"]))
                checks[n] = (meta["image"], (m & face).sum() / max(face.sum(), 1),
                             (m & front).sum() / max(m.sum(), 1), e["rot_deg"], e["trans_mm"])
                if out is not None and len(checks) == 1:
                    cv2.imwrite(str(out), ar.composed())
            st = ar.status()
            ar.stop()
    finally:        # every setting as it was, whatever happened
        ar.stop()
        ar.replay, ar.replay_truth, ar.intrinsics_path = saved[0], saved[1], saved[2]
        ar.set_registration(saved[3]["registration"] or None)
        ar.configure(**{k: saved[3][k] for k in ("board", "square_mm", "squares", "marker_mm", "face", "mandible",
                                                 "axes", "opacity", "view_rotation", "hide_board")})
        ar.set_tracking(**saved[4])
    c = np.array([v[1:] for v in checks.values()]).reshape(-1, 4)
    ok = len(c) >= min(4, len(images)) and c[:, 0].min() >= 0.95 and c[:, 1].min() >= 0.95
    expect(ok and st["registration"].get("source") == "file" and st["intrinsics"].get("source") == "calibrated",
           f"digital twin, {len(c)} of {len(images)} frames ({TWIN.name}, patient board on the bench): face drawn "
           f"over the labelled face, recall min {c[:, 0].min() if len(c) else float('nan'):.3f}, precision min "
           f"{c[:, 1].min() if len(c) else float('nan'):.3f}; board pose against the truth max "
           f"{c[:, 2].max() if len(c) else float('nan'):.3f} deg, {c[:, 3].max() if len(c) else float('nan'):.2f} mm")


def run_test():
    """Pose filter; board tracker; synthetic stream through the whole pipeline: pose against the truth, overlay
    drawn, light, video at the stream rate, file registration, no board, a hand over the board, MJPEG; the frames of
    the digital twin with the true registration; the opacity; frames that turn upright and back; a stream that stalls
    and then comes in a burst; start, stop and a stream that dies; frames that fail; bad settings and calibration
    files; two live ARs with their own OpenGL contexts and their shutdown. Everything on the nominal camera (or a
    calibration file the check writes), whatever calibration is on this computer."""
    import tempfile
    import urllib.request
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    if pose_filter.run_test(TRACKING):
        fails.append("pose filter")
    if board_tracker.run_test():
        fails.append("board tracker")
    expect(all(TRACKING[k] == v for k, v in pose_filter.DEFAULTS.items()),
           f"the filter's settings in TRACKING are pose_filter.DEFAULTS ({len(pose_filter.DEFAULTS)} values, one "
           f"source)")
    # the tremble of the readout: a pose shaken sideways by 0.1 mm at 500 mm is 0.29 px at f 1439 px; a missing pose
    # and a gap in time split the runs, a moving pose (no shake) gives none
    rng = np.random.default_rng(4)
    K_t = stream_intrinsics((1920, 1440), None)[0]
    X_t = rng.uniform(-60, 60, (40, 3))

    def shaken(k, sd):
        T = np.eye(4)
        T[:3, 3] = [rng.normal(0, sd) + 0.5 * k, 0.0, 500.0]
        return T
    rows = [(k / 30 + (0.5 if k >= 100 else 0.0), None if k == 50 else shaken(k, 0.1), shaken(k, 0.0))
            for k in range(300)]
    got = tremble(rows, X_t, K_t, seconds=20.0)
    expect(got[0] is not None and abs(got[0] - 0.29) < 0.05 and got[1] is not None and got[1] < 0.01,
           f"tremble readout: {got[0]} px for a shake of 0.29 px, {got[1]} px for a steady motion")
    # registration files: the bench frame's T_M_B read; an older ground_truth result (mandible frame), a result
    # without a head frame, a reflection and a file that is not JSON refused with bad_registration
    with tempfile.TemporaryDirectory(prefix="live_ar_reg_") as tmp:
        T_ok = np.eye(4)
        T_ok[:3, 3] = [1.0, 2.0, 3.0]
        refl = np.diag([1.0, 1.0, -1.0, 1.0])
        cases = {"new.json": ({"T_M_B": T_ok.tolist(), "T_M_mandible": T_ok.tolist(), "G": T_ok.tolist()}, True),
                 "bare.json": (T_ok.tolist(), True),
                 "older.json": ({"T_M_B": T_ok.tolist(), "G": T_ok.tolist(), "spheres": {}}, False),
                 "no_frame.json": ({"T_M_B": None, "T_M_mandible": T_ok.tolist(), "G": T_ok.tolist()}, False),
                 "reflection.json": ({"T_M_B": refl.tolist()}, False), "text.json": ("{not json", False)}
        got = {}
        for name, (content, _) in cases.items():
            p = Path(tmp) / name
            p.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
            try:
                got[name] = np.allclose(load_registration(p), T_ok)
            except LiveARError as e:
                got[name] = "refused" if e.code == "bad_registration" else e.code
        expect(all((got[n] is True) if ok else got[n] == "refused" for n, (_, ok) in cases.items()),
               f"registration files: read in the bench frame or refused with bad_registration ({got})")
    # a start that ends in a SystemExit of library code: failed, never starting for ever (every later start waits)
    quitter = LiveAR("sheet", 24.0, synthetic=True, intrinsics=None, prepare=False)

    def exits():
        raise SystemExit("a library gave up")
    quitter.prepare = exits
    try:
        quitter.start()
    except SystemExit:
        pass
    st_q = quitter.status()
    quitter.shutdown()
    expect(st_q["stream_state"] == FAILED and st_q["error"] == "start_failed",
           f"a start ended by SystemExit: failed ({st_q['stream_state']}, {st_q['error']}), not left starting")

    def run(ar, seconds):
        ar.start()
        t0 = time.perf_counter()
        errs = []
        while time.perf_counter() - t0 < seconds:
            time.sleep(0.05)
            n, T = ar.last_pose()
            if T is not None:
                Tt = ar.stream().truth(n)
                if Tt is not None:
                    e = marker_pose.pose_error(T, Tt)
                    errs.append((e["rot_deg"], e["trans_mm"]))
        st = ar.status()
        return st, np.array(errs).reshape(-1, 2)

    models = load_models()
    expect("face" in models and len(models["face"][1]) <= 1.2 * LIVE_TRIANGLES["face"],
           f"face model found ({FACE_MESH}), {len(models.get('face', ((), ()))[1])} triangles for the live view; "
           f"mandible {len(models.get('mandible', ((), ()))[1])}")
    ar = LiveAR("sheet", 24.0, synthetic=True, mandible=True, log=print, intrinsics=None)
    try:
        for board in ("sheet", "patient"):
            ar.configure(board=board)
            st, errs = run(ar, 4.0)
            ar.stop()
            tr = st.get("timings", {})
            # bounds of a working pipeline, not of its accuracy (eval_marker_pose.py measures that); the 45 mm board
            # is the looser
            rot, trans = (0.3, 2.0) if board == "sheet" else (1.0, 3.0)
            expect(st.get("tracked") and len(errs) > 20 and errs[:, 0].max() < rot and errs[:, 1].max() < trans,
                   f"{board}: tracked, pose against the truth over {len(errs)} frames: rotation max "
                   f"{errs[:, 0].max() if len(errs) else float('nan'):.3f} deg, translation max "
                   f"{errs[:, 1].max() if len(errs) else float('nan'):.2f} mm (camera at about "
                   f"{st.get('distance_mm')} mm)")
            expect(st.get("rms_px") is not None and st["rms_px"] < 0.5 and st["corners"] >= 8,
                   f"{board}: {st.get('corners')} of {st['board']['corners']} corners, RMS {st.get('rms_px')} px")
            trem = (st.get("tremble_px"), st.get("tremble_raw_px"))
            expect(all(isinstance(x, float) and 0 <= x < 5 for x in trem),
                   f"{board}: the tremble of the face reported, {trem[0]} px drawn, {trem[1]} px as found")
            expect(st.get("drawn_px", {}).get("face", 0) > 1000 and st["registration"]["source"] == "demo",
                   f"{board}: face drawn at the demo registration ({st['registration']['demo']}), "
                   f"{st.get('drawn_px', {}).get('face', 0)} px; mandible {st.get('drawn_px', {}).get('mandible', 0)} px")
            expect(st.get("fps_ar", 0) >= 0.75 * st["fps_stream"] and st.get("fps_pose", 0) >= 10,
                   f"{board}: video {st.get('fps_ar')} fps on a {st['fps_stream']} fps stream, poses "
                   f"{st.get('fps_pose')} fps ({st['renderer']}): corners {tr.get('detect_ms')} ms, pose "
                   f"{tr.get('pose_ms')} ms, render {tr.get('render_ms')} ms, draw and JPEG {tr.get('draw_ms')} ms")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registration.json"
            T = np.eye(4)
            T[:3, 3] = [20.0, 30.0, -40.0]
            p.write_text(json.dumps({"T_M_B": T.tolist()}), encoding="utf-8")
            ar.configure(board="sheet")
            ar.set_registration(p)
            st, _ = run(ar, 1.5)
            ar.stop()
            expect(st["registration"]["source"] == "file", f"registration from a file: {st['registration']['source']}")
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"T_M_B": np.ones((4, 4)).tolist()}), encoding="utf-8")
            try:
                ar.set_registration(bad)
                expect(False, "a matrix that is not rigid is refused")
            except LiveARError:
                expect(True, "a matrix that is not rigid is refused")
            ar.set_registration(None)
        ar.start()
        server = MjpegServer(ar)
        with urllib.request.urlopen(server.url, timeout=5) as r:
            head = r.read(4096)
        with urllib.request.urlopen(server.still_url, timeout=5) as r:
            still = r.read()
        server.close()
        ar.stop()
        expect(head.startswith(b"--frame") and b"\xff\xd8" in head and still[:2] == b"\xff\xd8",
               f"MJPEG on 127.0.0.1: multipart frames and a still ({len(still)} bytes)")
        twin_check(ar, expect)
        hand_check(ar, expect)
        opacity_check(ar, expect)
        settings_check(ar, expect)
        turn_check(ar, expect)
        burst_check(ar, expect)
    finally:
        ar.shutdown()
    echo_check(expect)
    # no board in view: grey light
    board, spec = make_board("sheet", 24.0)
    K, _, _ = stream_intrinsics((1920, 1440), None)
    empty = SyntheticStream(None, spec, K).wait_first()
    ar2 = LiveAR("sheet", 24.0, intrinsics=None)
    ar2.attach(empty)
    time.sleep(1.0)
    st = ar2.status()
    ar2.shutdown()
    # rates relative to the stream's own: a loaded machine slows the stand-in as much as the live view
    expect(not st.get("tracked") and st.get("corners", -1) == 0
           and st.get("fps_ar", 0) >= 0.6 * max(st.get("fps_stream", 0), 1.0),
           f"no board in view: not tracked, grey light, video still at {st.get('fps_ar')} fps")
    lifecycle_check(expect)
    failure_check(expect)
    input_check(expect)
    shutdown_check(expect)
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


def opacity_check(ar, expect):
    """The opacity setting: the face's tint grows with it, 0 draws nothing."""
    ar.configure(board="sheet", mandible=False, axes=False)
    ar.start()
    time.sleep(1.5)
    tints = {}
    for op in (0.0, 0.3, 1.0):
        ar.configure(opacity=op)
        time.sleep(0.6)
        img = ar.composed()
        st = ar.status()
        if img is None:
            continue
        tints[op] = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]))
    ar.stop()
    ar.configure(opacity=OPACITY, axes=True)
    ok = len(tints) == 3 and tints[0.0] < tints[0.3] < tints[1.0] and st.get("opacity") == 1.0
    expect(ok, "opacity of the overlay: mean saturation of the frame " + ", ".join(f"{k:.1f}: {v:.1f}"
                                                                                for k, v in tints.items()))


def settings_check(ar, expect):
    """The opacity slider moved and the layers switched while the board is tracked (a change every 150 ms, as the
    page sends them): the tracking goes on, no frame loses its pose or its overlay (16/10/2026: in the version before,
    each change started the tracking again, and 3 frames in 4 lost the overlay)."""
    ar.configure(board="sheet", mandible=False, axes=False, opacity=OPACITY)
    ar.trace = []
    ar.start()
    time.sleep(1.5)
    t0 = time.perf_counter()
    for k in range(20):
        ar.configure(opacity=0.3 + 0.7 * (k % 2), axes=bool(k % 4 < 2), face=True)
        time.sleep(0.15)
    rows = [r for r in ar.trace if r["t"] >= t0 + 0.1]
    ar.trace = None
    ar.stop()
    ar.configure(opacity=OPACITY, axes=True)
    tr = [r for r in rows if r["what"] == "track"]
    sh = [r for r in rows if r["what"] == "shown"]
    lost = sum(not r["ok"] for r in tr)
    hidden = sum(r["vis"] <= 0 for r in sh)
    expect(tr and lost == 0 and hidden == 0 and not any(r.get("verdict") == "restarted" for r in tr),
           f"opacity and layers changed 20 times while tracking: {lost} of {len(tr)} frames without a pose, {hidden} "
           f"of {len(sh)} drawn without the overlay")


def turn_check(ar, expect):
    """Frames that turn upright (1440 x 1920) and back, as a phone turned in the hand could send: no error, the
    video goes on at every size, the board is found again across."""
    board, spec = make_board("sheet", 24.0)
    K, _, _ = stream_intrinsics((1920, 1440), ar.intrinsics_path)
    s = SyntheticStream(board, spec, K, exposure_s=0.0)
    imgs = []
    for k in range(36):
        img, _ = s.render(0.2 * k, k)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        imgs.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if 12 <= k < 24 else img)
    saved = ar.replay, ar.replay_truth
    ar.replay, ar.replay_truth = imgs, None
    ar.configure(board="sheet", mandible=False)
    ar.trace = []
    ar.start()
    time.sleep(4.5)
    rows, ar.trace = ar.trace, None
    st = ar.status()
    ar.stop()
    ar.replay, ar.replay_truth = saved
    shown = [r for r in rows if r["what"] == "shown"]
    sizes = {tuple(r.get("size", ())) for r in rows if r["what"] == "track"}
    found, views = {}, {}
    for r in rows:
        if r["what"] == "track":
            found.setdefault(tuple(r["size"]), []).append(r["ok"])
        elif r["what"] == "shown" and r["T"] is not None:
            views.setdefault(tuple(r["size"]), []).append(r["view"])
    # the board stands upright in the frames across; turned a quarter clockwise, the picture is turned back
    upright = {k: (np.mean(np.array(v) == (0 if k[0] > k[1] else 270)) if v else 0.0) for k, v in views.items()}
    expect(not st.get("error") and len(shown) >= 25 and len(sizes) == 2
           and all(np.mean(v) > 0.5 for v in found.values()) and all(u > 0.8 for u in upright.values()),
           f"frames turned upright and back: no error ({st.get('error') or 'none'}), {len(shown)} frames shown, "
           f"board found on " + ", ".join(f"{k[0]}x{k[1]} {np.mean(v):.0%}" for k, v in found.items())
           + "; picture turned to stand up on " + ", ".join(f"{k[0]}x{k[1]} {u:.0%}" for k, u in upright.items()))


def burst_check(ar, expect):
    """A clip of the synthetic session whose frames arrive with a 0.6 s stall every 3 s and then all at once (Wi-Fi):
    the newest frame is drawn, the delay does not grow, and the overlay does not jump when the burst comes (the
    filter goes by the capture times)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        clip = make_clip(Path(tmp) / "burst", "sheet", 24.0, seconds=9.0)
        n = len(np.load(Path(tmp) / "burst_times.npy"))
        cap = np.arange(n) / 30.0
        arrive = cap.copy()
        for k in range(n):
            u = cap[k] % 3.0
            if 2.0 <= u < 2.6:          # held back, all delivered at 2.6 s of each period
                arrive[k] = cap[k] - u + 2.6
        np.save(Path(tmp) / "burst_times.npy", arrive)
        np.save(Path(tmp) / "burst_pts.npy", cap)
        saved = ar.replay, ar.replay_truth
        ar.replay, ar.replay_truth = str(clip), None
        ar.configure(board="sheet", mandible=False)
        ar.trace = []
        ar.start()
        stream = ar.stream()
        time.sleep(8.5)
        rows, ar.trace = ar.trace, None
        index = dict(stream._index)
        ar.stop()
        stream.thread.join(timeout=3.0)     # it reads the clip: done before the folder goes
        ar.replay, ar.replay_truth = saved
    K = stream_intrinsics((1920, 1440), ar.intrinsics_path)[0]
    out = measure_run(model_points(ar), [r for r in rows if index.get(r["n"], 10 ** 9) < n], K)
    shown = [r for r in rows if r["what"] == "shown" and index.get(r["n"], 10 ** 9) < n]
    after = [(r["t_done"] - r["t"]) * 1e3 for r in shown if 2.7 <= cap[index[r["n"]]] % 3.0 < 3.0]
    expect(out.get("jumps", 9) == 0 and after and np.median(after) < 120,
           f"stalls of 0.6 s then bursts: {out.get('jumps')} jumps of the overlay, frame to JPEG after a burst "
           f"median {np.median(after) if after else float('nan'):.0f} ms, video {out.get('fps_video', 0):.1f} fps")


def hand_check(ar, expect):
    """The synthetic session with the hand (bench_script) through the pipeline, 18 s: the face stays over the 0.4 s
    cover and fades out only during the 1.5 s one; it is drawn on every frame shown while the board is in view;
    the overlay against the truth while moving."""
    board, spec = make_board("sheet", 24.0)
    saved = ar.script, ar.exposure
    ar.script, ar.exposure = bench_script(spec), 0.008
    ar.configure(board="sheet", mandible=False)
    ar.trace = []
    ar.start()
    t0 = time.perf_counter()
    time.sleep(18.0)
    rows, ar.trace = ar.trace, None
    stream = ar.stream()
    st = ar.status()
    ar.stop()
    ar.script, ar.exposure = saved
    K = stream_intrinsics((1920, 1440), ar.intrinsics_path)[0]
    out = measure_run(model_points(ar), rows, K, {"still": (1.0, 6.0)}, stream.truth, lambda r: r["t"] - t0)
    still = out["segments"]["still"]
    shown = sorted((r for r in rows if r["what"] == "shown"), key=lambda r: r["n"])
    rel = np.array([r["t"] - t0 for r in shown])
    vis = np.array([r["vis"] for r in shown])
    short = vis[(rel >= 4.0) & (rel < 4.4)]
    long_ = vis[(rel >= 14.0 + TRACKING["hold_s"] + TRACKING["fade_s"] + 0.1) & (rel < 15.5)]
    clear = vis[((rel >= 1.0) & (rel < 1.9)) | ((rel >= 4.8) & (rel < 7.9)) | ((rel >= 9.0) & (rel < 13.9))]
    expect(len(short) and short.min() == 1.0 and len(long_) and long_.max() == 0.0 and clear.min() == 1.0
           and out["blinks"] == 1,
           f"hand over the board: face held at full opacity over the 0.4 s cover (min {short.min() if len(short) else -1:.2f}), "
           f"gone in the 1.5 s one (max {long_.max() if len(long_) else -1:.2f}), {out['blinks']} blink in 18 s (the long "
           f"cover), always drawn with the board in view (min {clear.min() if len(clear) else -1:.2f})")
    e = out.get("error_drawn_px", (np.inf, np.inf))
    expect(st.get("fps_ar", 0) >= 0.75 * st["fps_stream"] and e[1] < 6.0
           and still.get("jitter_drawn_px", 9) < still.get("jitter_found_px", 0) + 1e-9,
           f"session: video {out.get('fps_video', 0):.1f} fps on a {st['fps_stream']} fps stream, poses "
           f"{out.get('fps_pose', 0):.1f} fps; overlay against the truth median / 95th percentile "
           f"{e[0]:.2f} / {e[1]:.2f} px drawn, {out.get('error_found_px', (0, 0))[0]:.2f} / "
           f"{out.get('error_found_px', (0, 0))[1]:.2f} px as found; jitter while still "
           f"{still.get('jitter_found_px', 0):.3f} px as found, {still.get('jitter_drawn_px', 0):.3f} px drawn")


class EchoStream(SyntheticStream):
    """The board on a monitor beside the live view of itself, as ImpactGuide shows them on one screen: the camera
    films the monitor, whose video box shows the newest picture the live AR composed (ar.composed()). lay is the
    board's layout (screen_board) in the canvas at canvas_px of the monitor; box the video box (x, y, w, h), monitor
    px; the monitor is monitor_px at the layout's pitch, the app's dark around the canvas."""

    def __init__(self, ar, lay, canvas_px, box, monitor_px=(1920, 1080), K=None, size=(1920, 1440), fps=30, seed=3):
        import screen_board
        board, spec = make_board("sheet", lay["square_mm"], lay["squares"], lay["marker_mm"])
        if K is None:
            K = stream_intrinsics(size, None)[0]
        super().__init__(None, spec, K, size, fps, seed=seed)      # its thread starts later (wait_first)
        self.ar, self.box = ar, box
        mw, mh = monitor_px
        base = np.full((mh, mw), 22, np.uint8)
        cx, cy = canvas_px
        canvas = screen_board.render(lay)
        base[cy:cy + canvas.shape[0], cx:cx + canvas.shape[1]] = canvas
        self._base = base
        p = lay["pitch_mm"]
        ox, oy = cx + lay["origin_px"][0], cy + lay["origin_px"][1]
        # texture pixel (i, j) has its centre at ((i + 0.5 - ox) p, (j + 0.5 - oy) p) mm on the board
        self._S = np.array([[p, 0, (0.5 - ox) * p], [0, p, (0.5 - oy) * p], [0, 0, 1.0]])
        self._dist = 340.0

    def _board(self, T):
        tex = self._base.copy()
        shown = self.ar.composed() if self.ar is not None else None
        if shown is not None:
            x, y, w, h = self.box
            g = cv2.cvtColor(shown, cv2.COLOR_BGR2GRAY)
            s = min(w / g.shape[1], h / g.shape[0])
            gw, gh = max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))
            x0, y0 = x + (w - gw) // 2, y + (h - gh) // 2
            tex[y0:y0 + gh, x0:x0 + gw] = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
        img = self._bg.copy()
        H = self.K @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ self._S
        cv2.warpPerspective(tex, H, self.size, dst=img, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        return img


def echo_check(expect, seconds=10.0, out=None):
    """The board on a monitor and the live view of it on the same monitor (one screen, as ImpactGuide shows them):
    the camera sees the board and, in the video box beside it, the picture of the live view, which shows the board
    again. With hide_board that second board is covered: the pose stays on the real one, with no jump; without it
    the second board is there to be found (counted, not required). out: a folder for the pictures (None: none)."""
    import screen_board
    lay = screen_board.layout((1000, 760), 0.17921)
    res = {}
    for hide in (True, False):
        ar = LiveAR("sheet", lay["square_mm"], squares=lay["squares"], marker_mm=lay["marker_mm"], face=True,
                    axes=True, hide_board=hide, intrinsics=None)
        try:
            stream = EchoStream(ar, lay, (60, 160), (1090, 210, 800, 600))
            ar.attach(stream.wait_first())
            t0 = time.perf_counter()
            errs, seen = [], set()
            while time.perf_counter() - t0 < seconds:
                time.sleep(0.03)
                n, T = ar.last_pose()
                if T is None or n in seen:
                    continue
                seen.add(n)
                Tt = stream.truth(n)
                if Tt is not None and time.perf_counter() - t0 > 1.5:
                    e = marker_pose.pose_error(T, Tt)
                    errs.append((e["rot_deg"], e["trans_mm"]))
            st = ar.status()
            errs = np.array(errs).reshape(-1, 2)
            res[hide] = {"poses": len(errs), "jumps": int(np.sum(errs[:, 1] > 5.0)) if len(errs) else -1,
                         "max_mm": float(errs[:, 1].max()) if len(errs) else float("nan"),
                         "max_deg": float(errs[:, 0].max()) if len(errs) else float("nan"),
                         "fps": st.get("fps_ar", 0), "fps_stream": st.get("fps_stream", 0), "tracked": st.get("tracked")}
            if hide and out is not None:
                shot = ar.composed()
                if shot is not None:
                    Path(out).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(Path(out) / "echo_hidden.jpg"), shot)
                    cv2.imwrite(str(Path(out) / "echo_camera.jpg"), stream.latest()[0])
        finally:
            ar.shutdown()
            stream.close()
    h, o = res[True], res[False]
    without = (f"{o['poses']} poses, {o['jumps']} off by more than 5 mm, at most {o['max_mm']:.1f} mm" if o["poses"]
               else "no pose")
    expect(h["poses"] > 100 and h["jumps"] == 0 and h["max_mm"] < 3.0 and h["max_deg"] < 0.5
           and h["fps"] >= 0.5 * max(h["fps_stream"], 1.0),
           f"board and its live view on one screen, hide_board: {h['poses']} poses, {h['jumps']} off by more than "
           f"5 mm, at most {h['max_mm']:.2f} mm and {h['max_deg']:.2f} deg from the truth, {h['fps']} fps; without it "
           f"{without}")
    return res


class _DyingStream(SyntheticStream):
    """A stand-in that gives up after `frames` frames, as a phone stream whose restarts brought nothing back."""

    THREAD = "dying_stream"

    def __init__(self, board, spec, K, frames=20, **kw):
        super().__init__(board, spec, K, **kw)
        self.frames = frames

    def _run(self):
        for k in range(self.frames):
            if self._stop.is_set():
                return
            img, T = self.render(k / self.fps, k)
            n = self._put(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            with self._lock:
                self._truth[n] = T
            _pause(self._stop, 1.0 / self.fps)
        self._ended("the phone closed the stream", "stream_closed")


def _wait(cond, timeout, step=0.02):
    """cond() polled until true or the timeout passed (a check's helper); its last value."""
    deadline = time.perf_counter() + timeout
    while True:
        value = cond()
        if value or time.perf_counter() > deadline:
            return value
        time.sleep(step)


def _drawn(ar, name="face"):
    return ar.status().get("drawn_px", {}).get(name, 0)


def lifecycle_check(expect):
    """Start, stop and start again; stop() while start() opens the stream; two start() at once; attach() closing the
    stream before; a stream that dies: not running, failed with its code, and start() opens a new one without stop()."""
    ar = LiveAR("sheet", 24.0, synthetic=True, axes=False, intrinsics=None)
    try:
        setup = ar._setup
        K = stream_intrinsics((1920, 1440), None)[0]
        runs = []
        for _ in range(2):
            ar.start()
            n0 = ar.next_jpeg(0, 3.0)[1]
            flowing = _wait(lambda n0=n0: ar.next_jpeg(n0, 0.5)[1] > n0 + 5, 5.0)
            runs.append((ar.running, ar.state, bool(flowing)))
            ar.stop()
            runs.append((ar.running, ar.state))
        expect(runs == [(True, RUNNING, True), (False, IDLE)] * 2,
               f"start, stop, start, stop: running, state and frames {runs}")

        made, opened, gate = [], [], threading.Event()
        real_open, real_make = ar._open, ar._synthetic_stream

        def make():
            made.append(real_make())
            return made[-1]

        def slow_open(gen):
            opened.append(gen)
            gate.wait(5.0)
            return real_open(gen)
        ar._open, ar._synthetic_stream = slow_open, make
        th = threading.Thread(target=ar.start, name="start_check")
        th.start()
        _wait(lambda: opened, 3.0)
        ar.stop()               # while start() opens the stream
        gate.set()
        th.join(10.0)
        cancelled = (not th.is_alive() and not ar.running and ar.state == IDLE and ar.stream() is None
                     and len(made) == 1 and not made[0].alive)
        opened.clear()
        gate.clear()
        starters = [threading.Thread(target=ar.start, name=f"start_check_{k}") for k in range(2)]
        for th in starters:
            th.start()
        _wait(lambda: opened, 3.0)
        time.sleep(0.2)
        gate.set()
        for th in starters:
            th.join(10.0)
        once = len(opened) == 1 and ar.running and len(made) == 2
        ar._open, ar._synthetic_stream = real_open, real_make
        ar.stop()
        expect(cancelled and once, f"stop() during start(): cancelled, the stream it opened closed ({cancelled}); two "
                                   f"start() at once: {len(opened)} stream opened, running")

        s1 = SyntheticStream(setup["board"], setup["spec"], K).wait_first()
        s2 = SyntheticStream(setup["board"], setup["spec"], K).wait_first()
        ar.attach(s1)
        ar.attach(s2)
        replaced = s1._stop.is_set() and not s1.alive and s2.alive and ar.stream() is s2
        ar.stop()
        dying = _DyingStream(setup["board"], setup["spec"], K, frames=15).wait_first()
        ar.attach(dying)
        died = _wait(lambda: not ar.running, 5.0)
        st = ar.status()
        ar.start()              # no stop() first: the dead stream closed, a new one opened
        back = _wait(lambda: ar.status().get("fps_ar", 0) > 5, 5.0)
        st2 = ar.status()
        expect(replaced and died and st["stream_state"] == FAILED and st["error"] == "stream_closed"
               and not st["running"] and back and st2["running"] and st2["error"] == "" and dying._stop.is_set(),
               f"attach() closes the stream before ({replaced}); a stream that dies: running {st['running']}, "
               f"{st['stream_state']}, error {st['error']!r}; start() without stop(): running {st2['running']}, "
               f"{st2.get('fps_ar')} fps")
    finally:
        ar.shutdown()


def failure_check(expect):
    """One frame whose drawing fails: the picture goes on and no error is shown; drawing failing for 2.5 s: the last
    good frame stays, then frames_failing, cleared by the next good frame; a frame whose tracking fails is still
    shown (or a newer one); a model without a renderer is left out and the rest drawn; the NumPy renderer with the
    model outside the picture."""
    ar = LiveAR("sheet", 24.0, synthetic=True, mandible=False, axes=False, intrinsics=None)
    try:
        ar.start()
        _wait(lambda: ar.status().get("tracked") and _drawn(ar) > 1000, 5.0)
        real_draw = ar._draw_models
        fault = {"left": 0, "until": 0.0}

        def flaky(*a, **kw):
            if fault["left"] > 0 or time.perf_counter() < fault["until"]:
                fault["left"] -= 1
                raise RuntimeError("drawing failed (injected)")
            return real_draw(*a, **kw)
        ar._draw_models = flaky
        n0 = ar.next_jpeg(0, 2.0)[1]
        fault["left"] = 1
        errors, t0 = [], time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            errors.append(ar.status()["error"])
            time.sleep(0.05)
        n1 = ar.next_jpeg(n0, 1.0)[1]
        st = ar.status()
        last = st["last_frame_error"] or {}
        expect(st["frame_errors"] == 1 and not any(errors) and n1 - n0 > 15 and last.get("where") == "compose",
               f"one frame whose drawing fails: {st['frame_errors']} counted ({last.get('where')}), no error shown, "
               f"{n1 - n0} frames composed in the second after it")

        fault["until"] = time.perf_counter() + 2.5
        time.sleep(0.4)
        b1 = ar.composed()
        time.sleep(0.6)
        b2 = ar.composed()
        shown = _wait(lambda: ar.status()["error"] == "frames_failing", 2.5)
        during = ar.status()
        stale = during["error"] != "frames_failing" or (during["fps_ar"] == 0 and not during["tracked"])
        _wait(lambda: time.perf_counter() > fault["until"], 3.0)
        cleared = _wait(lambda: ar.status()["error"] == "", 1.5)
        n2 = ar.next_jpeg(0, 1.0)[1]
        time.sleep(0.5)
        n3 = ar.next_jpeg(n2, 1.0)[1]
        expect(b1 is not None and b1 is b2 and shown and stale and cleared and n3 > n2,
               f"drawing failing for 2.5 s: the last good frame kept ({b1 is not None and b1 is b2}), frames_failing "
               f"said ({bool(shown)}) with no rate and no light of the frame of before ({stale}), cleared by the next "
               f"good frame ({bool(cleared)}), frames again ({n3 - n2})")
        ar._draw_models = real_draw

        real_track, bad = ar._track, {}

        def track_once(stream, item):
            if not bad:
                bad["n"] = item.n
                raise ValueError("tracking failed (injected)")
            return real_track(stream, item)
        ar.trace = []
        ar._track = track_once
        _wait(lambda: bad, 2.0)
        time.sleep(0.5)
        rows, ar.trace = ar.trace, None
        ar._track = real_track
        shown = [r for r in rows if r["what"] == "shown"]
        plain = [r for r in shown if r["n"] == bad.get("n")]
        st = ar.status()
        expect(bad and (plain and plain[0]["kind"] == "none" or any(r["n"] > bad["n"] for r in shown))
               and st["error"] == "" and len(shown) > 10,
               f"a frame whose tracking fails: shown without a pose ({bool(plain)}), {len(shown)} frames in 0.5 s, "
               f"no error shown")
    finally:
        ar.shutdown()

    ar = LiveAR("sheet", 24.0, synthetic=True, mandible=True, axes=False, intrinsics=None)
    try:
        v, f = ar._models["mandible"]
        bad_f = f.copy()
        bad_f[::2] = bad_f[::2, ::-1]           # every other triangle turned: the winding is broken
        ar._models = {**ar._models, "mandible": (v, bad_f)}
        ar.start()
        ok = _wait(lambda: _drawn(ar) > 1000, 5.0)
        st = ar.status()
        expect(ok and st["models"]["face"] and not st["models"]["mandible"] and st["error"] == ""
               and "mandible" not in st.get("drawn_px", {}),
               f"a model without a renderer (mandible, broken winding): left out ({st['models']}), the face drawn "
               f"({st.get('drawn_px')}), no error shown")
    finally:
        ar.shutdown()

    ar = LiveAR("sheet", 24.0, mandible=False, intrinsics=None)       # no thread: drawn here with NumPy
    ar._gl_failed = True
    img = np.zeros((960, 1280, 3), np.uint8)
    Kd = scale_K(stream_intrinsics((1920, 1440), None)[0], 1280 / 1920, 960 / 1440)
    drawn, centre = {}, ar._models["face"][0].mean(0)
    for where, x in (("in view", 0.0), ("outside the picture", -5000.0)):
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])         # the face towards the camera
        T[:3, 3] = np.array([x, 0.0, 600.0]) - T[:3, :3] @ centre
        drawn[where] = ar._draw_models(img, T, Kd, None, ar.settings(), 0.3)[0].get("face", -1)
    expect(ar.renderer == "numpy" and drawn["in view"] > 0 and drawn["outside the picture"] == 0
           and ar.status()["frame_errors"] == 0,
           f"NumPy renderer: face in view {drawn['in view']} px, outside the picture {drawn['outside the picture']} "
           f"px, no error")
    ar.shutdown()


def input_check(expect):
    """Calibration files that cannot be used (not JSON, no K, another frame shape): the nominal camera, coded
    intrinsics_bad, status() unharmed; a calibration turned upright still used; the synthetic stream tracked without
    the distortion of the calibration (its frames have none); settings that cannot make a board refused with nothing
    changed; set_tracking reaching the detectors."""
    import tempfile
    K0 = stream_intrinsics((1920, 1440), None)[0]
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        files = {"garbled": "{", "no_K": json.dumps({"image_size": [1920, 1440]}),
                 "wide": json.dumps({"image_size": [1920, 1080], "K": scale_K(K0, 1.0, 0.75).tolist(),
                                     "dist": [0.0] * 5}),
                 "nan": json.dumps({"image_size": [1920, 1440], "K": [[float("nan"), 0, 960], [0, 1, 720], [0, 0, 1]]})}
        got = {}
        for name, text in files.items():
            path = tmp / f"{name}.json"
            path.write_text(text, encoding="utf-8")
            got[name] = stream_intrinsics((1920, 1440), path)[2]
        ar = LiveAR("sheet", 24.0, intrinsics=tmp / "garbled.json", prepare=False)
        st = ar.status()
        upright = tmp / "upright.json"
        Ku = np.array([[K0[1, 1], 0, K0[1, 2]], [0, K0[0, 0], K0[0, 2]], [0, 0, 1.0]])
        upright.write_text(json.dumps({"image_size": [1440, 1920], "K": Ku.tolist(), "dist": [0.0] * 5}),
                           encoding="utf-8")
        turned = stream_intrinsics((1920, 1440), upright)[2]
        expect(all(g["source"] == "nominal" and g.get("error") == "intrinsics_bad" for g in got.values())
               and st["intrinsics"].get("error") == "intrinsics_bad" and turned["source"] == "calibrated"
               and turned["turned"],
               "calibration files that cannot be used (" + ", ".join(got) + "): the nominal camera, intrinsics_bad, "
               "status() unharmed; one made upright still used, turned")

        distorted = tmp / "distorted.json"
        distorted.write_text(json.dumps({"image_size": [1920, 1440], "K": K0.tolist(),
                                         "dist": [-0.25, 0.12, 0.001, -0.001, 0.0]}), encoding="utf-8")
        ar = LiveAR("sheet", 24.0, synthetic=True, axes=False, intrinsics=distorted)
        try:
            ar.start()
            errs, t0 = [], time.perf_counter()
            while time.perf_counter() - t0 < 2.5:
                time.sleep(0.05)
                n, T = ar.last_pose()
                Tt = ar.stream().truth(n) if T is not None else None
                if Tt is not None:
                    e = marker_pose.pose_error(T, Tt)
                    errs.append((e["rot_deg"], e["trans_mm"]))
            st = ar.status()
        finally:
            ar.shutdown()
        errs = np.array(errs).reshape(-1, 2)
        expect(len(errs) > 20 and errs[:, 0].max() < 0.3 and errs[:, 1].max() < 2.0
               and st["intrinsics"]["source"] == "calibrated" and st["intrinsics"].get("distortion"),
               f"synthetic frames with a distorted calibration on file: tracked without the distortion, "
               f"{len(errs)} poses within {errs[:, 0].max() if len(errs) else float('nan'):.3f} deg and "
               f"{errs[:, 1].max() if len(errs) else float('nan'):.2f} mm of the truth")

    ar = LiveAR("sheet", 24.0, prepare=False, intrinsics=None)
    before = ar.settings()
    codes = []
    for kw in ({"squares": [5, 5], "square_mm": 20.0, "marker_mm": 30.0}, {"squares": [1, 6]},
               {"square_mm": float("nan")}, {"opacity": "dim"}, {"board": "round"}, {"view_rotation": 45}):
        try:
            ar.configure(**kw)
            codes.append("taken")
        except LiveARError as e:
            codes.append(e.code)
    ar.set_tracking(marker_px=41.0, min_scale=0.5)
    detectors = (ar._setup["detector"], ar._setup["detector_track"])
    reached = all(d.s["marker_px"] == 41.0 and d.s["min_scale"] == 0.5 for d in detectors)
    expect(codes == ["bad_board"] * 3 + ["bad_value"] * 3 and ar.settings() == before and reached,
           f"settings that cannot make a board refused, nothing changed ({', '.join(codes)}); set_tracking reaches "
           f"both detectors ({reached})")
    ar.shutdown()


def shutdown_check(expect):
    """Two live ARs at once, each with its own OpenGL context, gl_overlay's shared one untouched; a wrong MJPEG
    token refused; after shutdown the threads are gone, the contexts released, an MJPEG client's stream ends, and
    next_jpeg waits its timeout instead of answering at once (no client spins on it)."""
    import urllib.error
    import urllib.request

    import gl_overlay
    shared = gl_overlay._CTX
    ars = [LiveAR("sheet", 24.0, synthetic=True, axes=False, intrinsics=None) for _ in range(2)]
    server = None
    try:
        for ar in ars:
            ar.start()
        drawn = [bool(_wait(lambda ar=ar: _drawn(ar) > 1000, 5.0)) for ar in ars]
        kinds = [ar.renderer for ar in ars]
        server = MjpegServer(ars[0])
        ended = {}

        def client():
            try:
                with urllib.request.urlopen(server.url, timeout=10) as r:
                    while r.read(1 << 16):
                        pass
                ended["eof"] = time.perf_counter()
            except OSError as e:
                ended["error"] = repr(e)
        reader = threading.Thread(target=client, name="mjpeg_check")
        reader.start()
        time.sleep(0.5)
        try:
            urllib.request.urlopen(server.url.replace(server.token, "0" * len(server.token)), timeout=5)
            wrong = 200
        except urllib.error.HTTPError as e:
            wrong = e.code
    finally:
        t0 = time.perf_counter()
        for ar in ars:
            ar.shutdown()
    reader.join(5.0)
    t_end = ended.get("eof", np.inf) - t0
    t1 = time.perf_counter()
    jpeg, _ = ars[0].next_jpeg(0, 0.3)
    waited = time.perf_counter() - t1
    if server is not None:
        server.close()
    gone = not any(th.is_alive() for ar in ars for th in ar._threads)
    released = all(ar._gl is None and not ar._renderers for ar in ars)
    gl = not any(ar._gl_failed for ar in ars)
    expect(all(drawn) and (kinds == ["opengl"] * 2 or not gl) and gone and released and gl_overlay._CTX is shared,
           f"two live ARs at once ({', '.join(kinds)}), each with its own OpenGL context; after shutdown threads gone "
           f"({gone}), contexts released ({released}), the shared context untouched")
    expect(wrong == 404 and t_end < 3.0 and jpeg is None and waited >= 0.25 and ars[0].closed,
           f"MJPEG: wrong token {wrong}; a client's stream ends {t_end:.2f} s after shutdown; next_jpeg after it waits "
           f"{waited:.2f} s of 0.3 s and gives nothing")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["view", "test", "bench", "clip", "echo"])
    ap.add_argument("out", nargs="?", help="clip: the file to write (OUT.h264, OUT_times.npy, OUT_truth.npz); "
                                           "echo: a folder for its pictures")
    ap.add_argument("--board", choices=BOARDS, default="sheet")
    ap.add_argument("--square", type=float, default=None, help="square of the calibration board, mm (the A4 "
                                                               "sheet's as measured, or the one on a monitor)")
    ap.add_argument("--squares", type=lambda v: [int(n) for n in v.lower().split("x")], default=None,
                    help="squares of the board on a monitor, e.g. 14x7 (default: the A4 sheet's 11x8)")
    ap.add_argument("--marker", type=float, default=None, help="marker of the board on a monitor, mm")
    ap.add_argument("--registration", help="JSON file with T_M_B (default: the demo registration)")
    ap.add_argument("--mandible", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="a synthetic stream instead of the phone")
    ap.add_argument("--replay", help="recorded frames instead of the phone: a video (CLIP.h264 of live.py record) or "
                                     "a folder of images")
    ap.add_argument("--intrinsics", help="intrinsics JSON (default: the streaming mode's, or the nominal ones)")
    ap.add_argument("--seconds", type=float, default=24.0, help="bench: how long")
    ap.add_argument("--noise", type=float, default=2.0, help="clip: sensor noise, grey levels")
    ap.add_argument("--still", type=float, default=None,
                    help="bench: the phone is still from 1 s to this many s (default 6 on the synthetic session)")
    ap.add_argument("--segments", help="bench: JSON {name: [from, to]} of parts of the clip (s), each measured")
    ap.add_argument("--reference", action="store_true",
                    help="bench on a clip: the overlay against a reference pose (full detection, smoothed without "
                         "lag; computed once and kept beside the clip)")
    a = ap.parse_args()
    if a.command == "test":
        sys.exit(run_test())
    if a.command == "echo":
        fails = []
        echo_check(lambda ok, what: (print(("ok    " if ok else "FAIL  ") + what, flush=True),
                                     None if ok else fails.append(what)), out=a.out)
        sys.exit(1 if fails else 0)
    if a.command == "clip":
        if not a.out:
            ap.error("clip needs OUT")
        print(make_clip(a.out, a.board, a.square or 24.0, a.seconds, noise=a.noise))
        return
    try:
        cmd_bench(a) if a.command == "bench" else cmd_view(a)
    except (live.LiveError, LiveARError) as e:
        sys.exit(f"live_ar: {e}")


if __name__ == "__main__":
    main()
