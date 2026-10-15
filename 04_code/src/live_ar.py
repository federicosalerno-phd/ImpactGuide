"""Live AR on the phone stream (T20.3; module 8 of ai_design.md): board pose, face overlay and status light.

Three threads work on the frames of the stream (live.py), always on the newest one, never on a queue:
  track     every frame: the board's ChArUco corners followed from the previous frame by optical flow and put back on
            the chessboard's saddle points, the pose from them (board_tracker.py), then the temporal filter
            (pose_filter.py: One Euro filter on SE(3) with a dead band, outlier gate, glide, hold and fade). A few
            milliseconds, so it keeps the rate of the stream.
  detect    now and then (TRACK["detect_every_s"], or at once when the tracking weakens or the board is lost): the
            full ChArUco detection, in a box around the board or in the whole frame; its corners re-anchor the
            tracking a few frames later.
  compose   the newest frame tracked: the face (and the mandible) drawn at its filtered pose with gl_overlay.py, a fill
            and an outline at the opacity asked for, board axes and corners on request, JPEG. It owns the OpenGL
            context.
So each frame is drawn at the pose found on that same frame, at the rate of the stream; when the computer falls
behind, frames are skipped, never queued. The overlay comes late by the stream's latency (about 150 ms on the cable)
plus the processing, but on its own frame. The page's calls (status, configure, next_jpeg) only read or set values.

Frames as ai_design.md (Notazione): C camera (OpenCV), M board (cv2.aruco.CharucoBoard: origin at the outer corner of
the first square, x along the printed columns, y along the rows, z into the board), B head (bench scene, mm).

Boards
    patient   the 5 x 5 board of the marker holder; squares and markers from marker_pose.BOARD
    sheet     the calibration board of calibrate_charuco.py: shown on a monitor (screen_board.py: as many squares as
              the monitor holds, square from its pixel pitch) or printed on A4 (11 x 8, square as measured with the
              caliper); either is enough to try the page without the bench. squares and marker_mm give the layout of
              the one on a monitor; without them it is the A4 one
A pose is used with at least board_tracker.TRACK["min_corners"] corners (8 to start on the calibration board; with
fewer than 10 they must spread over the board and fit tighter) and a reprojection RMS under the limit: 1 px with
intrinsics calibrated for the stream (ai_design.md, module 1), 3 px with the nominal ones, which leave out the lens
distortion. The small patient board works from about 20 to 64 cm (15/10, board shown at its true size on the laptop
screen; at 1920 x 1440, 20 px per square at the far end).

Intrinsics of the streaming mode: 03_data/processed/calibration/intrinsics_stream.json, in the format
calibrate_charuco.py writes for intrinsics_photo.json (T5.3), calibrated on frames of the stream at the working
distance; scaled if the stream has another size. Until it exists: nominal Poco X7 Pro, 3070 px at the 3072 px height of
the 4:3 sensor field (as synth_twin.py), so 1439 px at 1920 x 1440, principal point at the centre, no distortion.

Registration T_M_B: a JSON file holding a 4 x 4 "T_M_B" in mm (what ground_truth.py measure writes). Until there is
one, a demo registration at a fixed offset from the board, which the page always labels as demo:
    patient   the registration of the bench design (the scene, meshes/occluders.npz of synth_twin.py)
    sheet     the face lying on the sheet, centred, looking up

Models in B (mm): the decimated face of T13.1 (synth_twin.py meshes) and the mandible of the bench scene
(build_targets.py), both under C:\\work\\marker-impact and written again by those scripts; for the live view each is
decimated again to LIVE_TRIANGLES (quadric edge collapse) and kept in LIVE_MODELS.

For the app:
    ar = LiveAR(board="sheet", square_mm=24.0)       # one per process: its compose thread owns the OpenGL context
                                                     # (squares=(14, 7), marker_mm=... for the board on a monitor)
    ar.start()                                       # closes the camera app, opens the stream (a few seconds)
    jpeg, n = ar.next_jpeg(after=n)                  # newest composed frame
    jpeg, n, t = ar.next_frame(after=n)              # the same with the time it was composed (phone_view.py)
    ar.status()                                      # light, corners, RMS, fps, latency, timings, sources
    ar.configure(board=..., square_mm=..., squares=..., marker_mm=..., face=..., mandible=..., axes=...,
                 opacity=...)                        # opacity 0-1 of the overlay's fill and outline (default OPACITY)
    ar.set_registration(path or None)
    ar.set_tracking(hold_s=..., ...)                 # TRACKING and board_tracker.TRACK overrides
    ar.stop(); ar.shutdown()
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
                            twin with the true registration; exit 1 on failure
"""
import argparse
import json
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
import board_tracker
import calibrate_charuco
import live
import marker_pose
import pose_filter

ROOT = Path(__file__).resolve().parents[2]
STREAM_INTRINSICS = ROOT / "03_data" / "processed" / "calibration" / "intrinsics_stream.json"
WORK = Path(r"C:\work\marker-impact")
FACE_MESH = WORK / "synth" / "meshes" / "face_phantom_r20k.npz"
DESIGN = WORK / "synth" / "meshes" / "occluders.npz"          # T_M_B of the bench scene
MANDIBLE_MESH = WORK / "targets" / "mandible.npz"
TWIN = WORK / "synth" / "sessions"                              # frames of the digital twin, for the test
LIVE_MODELS = WORK / "live_models"                              # the models decimated for the live view
LIVE_TRIANGLES = {"face": 4000, "mandible": 6000}
CAMERA_APP = acquisition.CAMERA_APP     # the phone's camera app holds the camera while it is open
NOMINAL_F = 3070.0 / 3072.0             # focal length over the height of the 4:3 sensor field (synth_twin.NOMINAL)
# camera to decoded frame on each link, median of the screen flashes (live.py latency): cable 149 ms on 15/10 night
# with live.CODEC_OPTIONS (169 ms without, 195 ms in the first test); Wi-Fi not measured yet (None: the page says so
# and counts only the processing)
STREAM_LATENCY_MS = {"usb": 149.0, "wifi": None}
MAX_RMS = {"calibrated": 1.0, "nominal": 3.0}
DISPLAY_WIDTH = 1280
JPEG_QUALITY = 80
# the look of the overlay: a fill and a thin outline in the colours of the palette, with a light shading (its depth:
# 0 flat, 1 black where the surface is seen edge on). Their opacity is the setting `opacity` (0-1) times fill and line
# (capped at 1): at the default OPACITY a soft tint and a clear outline, at 1 a nearly opaque face.
OPACITY = 0.3
STYLE = {"face": {"fill": 0.85, "line": 2.5, "shade": 0.6}, "mandible": {"fill": 0.9, "line": 2.2, "shade": 0.6}}
COLORS = {"face": "#F5C542", "mandible": "#E6E6EC", "corners": "#58B36B", "reprojected": "#E6E6EC"}
BOARDS = ("patient", "sheet")
TRACKING = {
    # temporal filter of the pose (pose_filter.py); the tracking itself is board_tracker.TRACK (set_tracking takes both)
    "smooth": True,           # False: every frame at its own pose as it comes (the tests of the geometry)
    "min_cutoff_hz": 2.0,     # smoothing while still: lower is calmer and slower to settle
    "beta": 0.25,             # cutoff added per px/s of the model's speed past the dead band: higher lags less
    "d_cutoff_hz": 2.0,       # smoothing of that speed
    "lead": True,             # moving, the filter carries the pose on by its speed first: no lag at a steady speed
    "still_px_s": 15.0,       # dead band: slower than this (px/s at the stream size) is noise, not motion
    "predict_s": 0.1,         # a frame without its own pose is drawn at the pose carried on at most this far
    "gate_px": 40.0,          # a pose moving the model more than this (px, stream size) from the prediction is a jump:
    "strong_corners": 0.5,    #   believed with at least this share of the board's corners
    "strong_rms": 0.6,        #   and an RMS under this share of the limit (gliding there), otherwise only if the next
    "glide_s": 0.15,          #   pose agrees
    "noise_k": 2.0,           # the rotation's dead band: this many times the speed its expected noise gives
    "noise_ref_px": 0.5,      # rotation noise (px per frame at the model) up to which it is smoothed like the position;
    "min_cutoff_floor": 0.4,  #   noisier, its lowest cutoff goes down in proportion, to this share at most
    "wait_sure": True,        # a new acquisition is drawn once its mirrored pose is ruled out (board_tracker.py)
    # loss of the board
    "hold_s": 0.7,            # the last pose stays drawn this long
    "fade_s": 0.35,           # then fades out over this time
    "fade_in_s": 0.1,         # and fades in when the board is found again,
    "acquire_frames": 2,      #   after this many poses in a row
    "roi_s": 0.4,             # after the board was lost the detection looks where it was for this long
    "views": False,           # separate views, not a video (tests): every frame detected on its own, nothing carried
}
_GL_THREAD = None                       # the thread that owns gl_overlay's OpenGL context
_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)


class LiveARError(RuntimeError):
    pass


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
        raise LiveARError(f"unknown board {name}")
    spec["corners"] = int(len(board.getChessboardCorners()))
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
    principal point near the centre, and labelled "turned")."""
    w, h = size
    if path and Path(path).is_file():
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        cw, ch = d["image_size"]
        K0 = np.array(d["K"], np.float64)
        dist = np.asarray(d.get("dist", np.zeros(5)), np.float64).ravel()
        turned = (w > h) != (cw > ch)
        if turned:
            K0 = np.array([[K0[1, 1], 0, K0[1, 2]], [0, K0[0, 0], K0[0, 2]], [0, 0, 1.0]])
            cw, ch = ch, cw
            if len(dist) >= 4:
                dist = dist.copy()
                dist[[2, 3]] = dist[[3, 2]]
        K = scale_K(K0, w / cw, h / ch)
        src = {"source": "calibrated", "path": str(path), "date": d.get("date", ""), "rms_px": d.get("rms_px"),
               "turned": bool(turned)}
    else:
        f = NOMINAL_F * min(w, h)       # the sensor's 4:3 field, across or upright
        K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
        dist = np.zeros(5)
        src = {"source": "nominal", "path": "", "date": "", "rms_px": None}
    src.update({"f": round(float((K[0, 0] + K[1, 1]) / 2), 1), "size": [int(w), int(h)]})
    return K, dist, src


def load_registration(path):
    """T_M_B (4 x 4, mm) from a JSON file: {"T_M_B": [[...]]} as ground_truth.py writes it, or the bare matrix."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    T = np.asarray(d["T_M_B"] if isinstance(d, dict) else d, np.float64)
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3),
                                                                                     atol=1e-4):
        raise LiveARError(f"{Path(path).name}: T_M_B is not a 4 x 4 rigid transform")
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
class _Frames:
    """What the stand-ins share with live.PhoneStream: the newest frame with its time and counter, the rate."""

    def _init_frames(self):
        self.error, self.restarts = None, 0
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._frame, self._t, self._n = None, 0.0, 0
        self._item, self._session = None, 1
        self._times = deque(maxlen=120)
        self._decode_ms = deque(maxlen=120)
        self._stop = threading.Event()

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
        return self.thread.is_alive()

    def close(self):
        self._stop.set()
        with self._new:
            self._new.notify_all()

    def wait_first(self, timeout=5.0):
        t0 = time.perf_counter()
        while self.latest()[0] is None and time.perf_counter() - t0 < timeout:
            time.sleep(0.01)
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
    the same pinhole camera the page uses. truth(n) is the true T_C_M of frame n. With board None there is no board.
    script(t) -> (phase, hands) drives the sway and lays hands over the board (bench_script); exposure_s blurs the
    frame along the motion (the mean of three renders over the exposure); noise is the sensor noise (sigma, grey
    levels)."""

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
        self.thread = threading.Thread(target=self._run, daemon=True, name="synthetic_stream")
        self.thread.start()

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
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def truth(self, n):
        with self._lock:
            return self._truth.get(n)


class ReplayStream(_Frames):
    """Stand-in for live.PhoneStream that plays recorded frames in a loop: a list of images, a folder of images or a
    video. A raw H.264 file (live.py record) is decoded as the phone's stream is and played with the recorded timing
    (CLIP_times.npy beside it); CLIP_truth.npz beside it, when there is one, holds the true T_C_M of every frame
    (`T_C_M`, N x 4 x 4). truth(n) gives back the item of `truth` (or of that file) that goes with frame n."""

    VIDEO = (".mp4", ".mov", ".mkv", ".avi", ".h264", ".264")

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
            raise LiveARError(f"nothing to play in {source}")
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
        self.thread = threading.Thread(target=self._run, daemon=True, name="replay_stream")
        self.thread.start()

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
                        for packet in codec.parse(data):
                            t0 = time.perf_counter()
                            for frame in codec.decode(packet):
                                img = frame.to_ndarray(format="bgr24")
                                yield img, (time.perf_counter() - t0) * 1e3, k
                                k += 1
                                t0 = time.perf_counter()
                        if not data:
                            break
                if k == 0:
                    self.error = f"cannot read {self._video.name}"
                    return
            elif self._video is not None:
                cap = cv2.VideoCapture(str(self._video))
                ok, img = cap.read()
                if not ok:
                    self.error = f"cannot read {self._video.name}"
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
                        self.error = f"cannot read {p.name}"
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
                time.sleep(max(0.0, t_loop + rec[k] - time.perf_counter()))
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
                time.sleep(max(0.0, nxt - time.perf_counter()))
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

    The threads live as long as the object: track, detect and compose (which owns the OpenGL context and so is never
    replaced): stop() and start() only close and open the stream. Meant as one per process (the app keeps one); a
    later one makes its own OpenGL context. Nothing here blocks the caller's thread for long: the calls of the page
    (status, configure, set_tracking, next_jpeg) only read or set values under a lock."""

    def __init__(self, board="sheet", square_mm=None, registration=None, face=True, mandible=False, axes=True,
                 intrinsics=STREAM_INTRINSICS, synthetic=False, serial=None, size="1920x1440", fps=30,
                 display_width=DISPLAY_WIDTH, colors=None, log=None, replay=None, replay_truth=None, tracking=None,
                 script=None, exposure_s=0.0, squares=None, marker_mm=None, opacity=OPACITY):
        if board not in BOARDS:
            raise LiveARError(f"unknown board {board}")
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
        self._apply_lock = threading.Lock()
        self._want = {"board": board, "square_mm": float(square_mm or calibrate_charuco.BOARD["square"]),
                      "squares": [int(n) for n in squares] if squares else None,
                      "marker_mm": float(marker_mm) if marker_mm else None,
                      "registration": str(registration) if registration else "", "face": bool(face),
                      "mandible": bool(mandible), "axes": bool(axes),
                      "opacity": float(np.clip(opacity, 0.0, 1.0)), "version": 1}
        self._tracking_version = 1
        self._stream = None
        self._starting = False
        self._jpeg, self._jpeg_n, self._bgr, self._jpeg_t = None, 0, None, None
        self._link = None               # the link of the phone's stream (usb, wifi); None for the stand-ins
        self._reset_run()
        self._status = {}
        self._error = ""
        self._quit = threading.Event()
        self._models = load_models(log=self.log)
        self._renderers, self._centroids = {}, {}
        self.renderer = ""
        self._applied = 0
        self._setup = None
        self._cam = (None, None, {})       # K, dist, source for the size of the last frame
        self._cam_size = None
        self._apply(dict(self._want))
        self._threads = []
        for name, target in (("live_ar_compose", self._compose_loop), ("live_ar_track", self._track_loop),
                             ("live_ar_detect", self._detect_loop)):
            th = threading.Thread(target=target, daemon=True, name=name)
            th.start()
            self._threads.append(th)

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
        self._skipped = 0                 # frames tracked but not drawn (the compose thread was busy)
        self._cpu = deque(maxlen=40)      # (wall, process CPU time) for the load

    # ── control, from any thread ─────────────────────────────────────────
    @property
    def running(self):
        return self._stream is not None

    def start(self, timeout=20.0):
        """Closes the camera app and opens the stream; returns once the first frame is in (raises on failure)."""
        if self._stream is not None:
            return self
        self._starting, self._error = True, ""
        try:
            if self.replay is not None:
                stream = ReplayStream(self.replay, self.fps, self.replay_truth).wait_first()
            elif self.synthetic:
                stream = self._synthetic_stream()
            else:
                close_camera_app(self.serial)
                stream = live.open_stream(self.serial, self.size, self.fps, log=self.log)
        except Exception as e:
            self._error = str(e)
            raise
        finally:
            self._starting = False
        self.attach(stream)
        return self

    def attach(self, stream):
        """Works on a stream opened elsewhere (anything with wait_item or wait_frame, stats, alive, error, close)."""
        with self._lock:
            self._stream = stream
            self._link = getattr(stream, "link", None)
            self._status = {}
            self._reset_run()               # nothing of the stream before
            self._cam_size = None           # intrinsics read again: a calibration may have been saved meanwhile
        return self

    def _synthetic_stream(self):
        w, h = (int(x) for x in self.size.split("x"))
        K, _, _ = stream_intrinsics((w, h), self.intrinsics_path)
        return SyntheticStream(self._setup["board"], self._spec, K, (w, h), self.fps, script=self.script,
                               exposure_s=self.exposure).wait_first()

    def stop(self):
        with self._lock:
            stream, self._stream = self._stream, None
            self._jpeg, self._bgr = None, None
            self._frame_ready.notify_all()
            self._det_cv.notify_all()
        if stream is not None:
            stream.close()

    def shutdown(self):
        self.stop()
        self._quit.set()
        with self._lock:
            self._det_cv.notify_all()
            self._job_cv.notify_all()
        for th in self._threads:
            th.join(timeout=2.0)

    def configure(self, **kw):
        """board, square_mm, squares and marker_mm (the calibration board's layout: None for the A4 one), face,
        mandible, axes, opacity (0-1, of the overlay's fill and outline): applied from the next frame."""
        if kw.get("board") is not None and kw["board"] not in BOARDS:
            raise LiveARError(f"unknown board {kw['board']}")
        with self._lock:
            before = {k: self._want[k] for k in ("board", "square_mm", "squares", "marker_mm")}
            if kw.get("board") is not None:
                self._want["board"] = kw["board"]
            if kw.get("square_mm") is not None:
                self._want["square_mm"] = float(kw["square_mm"])
            if "squares" in kw:
                self._want["squares"] = [int(n) for n in kw["squares"]] if kw["squares"] else None
            if "marker_mm" in kw:
                self._want["marker_mm"] = float(kw["marker_mm"]) if kw["marker_mm"] else None
            for k in ("face", "mandible", "axes"):
                if kw.get(k) is not None:
                    self._want[k] = bool(kw[k])
            if kw.get("opacity") is not None:
                self._want["opacity"] = float(np.clip(float(kw["opacity"]), 0.0, 1.0))
            self._want["version"] += 1
            want = dict(self._want)
        board_changed = before != {k: want[k] for k in before}
        if not self.running:
            self._apply(want)
        elif self.synthetic and board_changed and (want["board"] != self._spec["name"] or want["board"] == "sheet"):
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
            want = dict(self._want)
        if not self.running:
            self._apply(want)

    def set_tracking(self, **kw):
        """Overrides of TRACKING and of board_tracker.TRACK (unknown keys refused); tracker and filter start again
        from the next frame."""
        bad = [k for k in kw if k not in TRACKING and k not in board_tracker.TRACK]
        if bad:
            raise LiveARError(f"unknown tracking settings {bad}")
        with self._lock:
            self.tracking = {**self.tracking, **kw}
            self._tracking_version += 1

    # ── what the page reads ──────────────────────────────────────────────
    def next_jpeg(self, after=0, timeout=1.0):
        """(JPEG bytes, frame counter) of a composed frame newer than `after`; (None, after) on timeout or when
        stopped."""
        with self._frame_ready:
            self._frame_ready.wait_for(lambda: (self._jpeg is not None and self._jpeg_n != after) or self._quit.is_set(),
                                       timeout=timeout)
            if self._jpeg is None or self._jpeg_n == after:
                return None, after
            return self._jpeg, self._jpeg_n

    def next_frame(self, after=0, timeout=1.0):
        """As next_jpeg, with the time.perf_counter() when the frame was composed: (JPEG, counter, time), or
        (None, after, None)."""
        with self._frame_ready:
            self._frame_ready.wait_for(lambda: (self._jpeg is not None and self._jpeg_n != after) or self._quit.is_set(),
                                       timeout=timeout)
            if self._jpeg is None or self._jpeg_n == after:
                return None, after, None
            return self._jpeg, self._jpeg_n, self._jpeg_t

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

    def settings(self):
        """The settings asked for: board, square_mm, squares, marker_mm, registration, face, mandible, axes, opacity."""
        with self._lock:
            return {k: v for k, v in self._want.items() if k != "version"}

    def status(self):
        with self._lock:
            st = dict(self._status)
            stream = self._stream
            want = dict(self._want)
            served = list(self._served)
            cpu = list(self._cpu)
        st.update({"running": stream is not None, "starting": self._starting, "error": self._error,
                   "synthetic": self.synthetic or self.replay is not None, "replay": self.replay is not None,
                   "renderer": self.renderer,
                   "layers": {k: want[k] for k in ("face", "mandible", "axes")}, "opacity": want["opacity"],
                   "models": {k: k in self._models for k in ("face", "mandible")},
                   "triangles": {k: int(len(f)) for k, (_, f) in self._models.items()},
                   "board": self._spec, "registration": self._setup["registration"], "score": None})
        if not st.get("intrinsics"):
            st["intrinsics"] = self._cam[2] or stream_intrinsics(self._size_wh(), self.intrinsics_path)[2]
        if stream is None:
            st.update({"tracked": False, "fps_stream": 0.0, "fps_ar": 0.0, "fps_pose": 0.0})
            return st
        s = stream.stats()
        st["fps_stream"] = round(s["fps"], 1)
        st["frames"] = s["frames"]
        gap = s.get("interval_p95_ms")
        st["gap_p95_ms"] = round(gap, 1) if gap == gap and gap is not None else None   # not NaN
        tm = dict(st.get("timings", {}))
        if s.get("decode_ms") == s.get("decode_ms"):     # not NaN
            tm["decode_ms"] = round(s["decode_ms"], 1)
        now = time.perf_counter()
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
        if not stream.alive:
            st["error"] = stream.error or "stream stopped"
            st["cause"] = getattr(stream, "cause", "")
        return st

    def _size_wh(self):
        w, h = (int(x) for x in self.size.split("x"))
        return w, h

    # ── setup ────────────────────────────────────────────────────────────
    def _apply(self, want):
        """Board, detector and registration for the wanted settings (worker thread, or any thread while stopped)."""
        with self._apply_lock:
            if want["version"] == self._applied and self._setup is not None:
                return
            old = self._setup
            board, spec = make_board(want["board"], want["square_mm"], want["squares"], want["marker_mm"])
            if old is not None and spec == old["spec"]:
                board, detector = old["board"], old["detector"]
            else:
                detector = board_tracker.Detector(board, spec, self.tracking)
            reg = {"source": "demo", "path": "", "demo": "", "error": ""}
            T = None
            if want["registration"]:
                try:
                    T = load_registration(want["registration"])
                    reg.update(source="file", path=want["registration"])
                except (OSError, ValueError, KeyError, LiveARError) as e:
                    reg["error"] = str(e)
            if T is None and "face" in self._models:
                T, reg["demo"] = demo_registration(spec, self._models["face"][0])
            # one assignment, so a worker never sees the board of one setting with the detector of another
            self._setup = {"board": board, "spec": spec, "detector": detector, "T_M_B": T, "registration": reg,
                           "version": want["version"], "obj": board.getChessboardCorners().astype(np.float64)}
            self._applied = want["version"]

    @property
    def _spec(self):
        return self._setup["spec"]

    def _renderer(self, name):
        """The renderer of a model, made in the compose thread (OpenGL, or NumPy when OpenGL is not there)."""
        if name not in self._renderers:
            v, f = self._models[name]
            try:
                import gl_overlay
                global _GL_THREAD
                if _GL_THREAD not in (None, threading.get_ident()):
                    # gl_overlay keeps one context, bound to the thread that made it: the compose thread of an
                    # earlier LiveAR. This thread makes its own.
                    gl_overlay._CTX = None
                _GL_THREAD = threading.get_ident()
                r, kind = gl_overlay.GLMeshRenderer(v, f), "opengl"
            except Exception as e:  # noqa: BLE001
                self.log(f"{name}: OpenGL renderer not available ({e}); NumPy renderer")
                try:
                    import mesh_overlay
                    r, kind = mesh_overlay.MeshRenderer(v, f), "numpy"
                except Exception as e2:  # noqa: BLE001  the model is left out, the rest goes on
                    self._models.pop(name, None)
                    self._error = f"{name}: {e2}"
                    return None
            self._renderers[name] = r
            if self.renderer != "numpy":
                self.renderer = kind
        return self._renderers[name]

    def _intrinsics(self, w, h):
        if self._cam_size != (w, h):
            self._cam = stream_intrinsics((w, h), self.intrinsics_path)
            self._cam_size = (w, h)
        return self._cam

    def _model_points(self, setup):
        """Points of the face (or of the board) in M: the anchor of the filter and what the tracker weighs the two
        mirrored poses on."""
        T_M_B = setup["T_M_B"]
        if T_M_B is not None and "face" in self._models:
            v = self._models["face"][0]
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
            return stream.wait_item(after, timeout=0.1)
        frame, t, n = stream.wait_frame(after, timeout=0.1)
        return None if frame is None else live.Frame(n, t, bgr=frame)

    # ── thread 1: every frame, the board tracked ─────────────────────────
    def _track_loop(self):
        cur, last = None, 0
        while not self._quit.is_set():
            stream = self._stream
            if stream is None:
                cur = None
                time.sleep(0.02)
                continue
            if stream is not cur:
                cur, last = stream, 0
            item = self._next_item(stream, last)
            if item is None:
                continue
            last = item.n
            try:
                self._track(stream, item)
            except Exception as e:  # noqa: BLE001  a bad frame must not end the thread
                self._failed(stream, e)

    def _track(self, stream, item):
        t0 = time.perf_counter()
        with self._lock:
            want = dict(self._want)
        if want["version"] != self._applied:
            self._apply(want)
        setup = self._setup
        w, h = item.size
        K, dist, cam = self._intrinsics(w, h)
        limit = MAX_RMS[cam["source"]]
        tracker, filt = self._tracker_for(setup, K, item)
        s = self.tracking
        tracker.s = {**tracker.s, **{k: v for k, v in s.items() if k in board_tracker.TRACK}}
        key = self._tracker_key
        with self._lock:
            done, self._det_done = self._det_done, None
        det = done[0] if done is not None and done[1] == key else None
        gray = item.gray()
        if s.get("views"):
            tracker.reset()
            det = setup["detector"].detect(gray, n=item.n, t=item.t_capture)
        t1 = time.perf_counter()
        res = tracker.step(gray, item.t_capture, item.n, K, dist, limit, det)
        t2 = time.perf_counter()
        spec = setup["spec"]
        strong = res.ok and res.n >= s["strong_corners"] * spec["corners"] and res.rms <= s["strong_rms"] * limit
        use = res.T if (res.ok and (res.sure or not s.get("wait_sure", True))) else None
        if not s.get("smooth", True) and use is not None:
            strong = True
        verdict = filt.measure(item.t_capture, use, strong, res.noise_px)
        T, vis, state = filt.pose_at(item.t_capture)
        t3 = time.perf_counter()
        # the next detection: in the box around the board, or where it was last, or in the whole frame
        if res.ok and res.box is not None:
            self._lost_box = (res.box, res.marker_px, item.t_capture)
        ask = None
        if not s.get("views") and tracker.want_detection(res, item.t_capture, self._det_t):
            box, mpx = (res.box, res.marker_px) if res.ok else (None, None)
            lb = self._lost_box
            if box is None and lb is not None and item.t_capture - lb[2] < s["roi_s"]:
                box, mpx = lb[0], lb[1]
            ask = (item, box, mpx, key)
        tm = {"gray_ms": (t1 - t0) * 1e3, "track_ms": (t2 - t1) * 1e3, "filter_ms": (t3 - t2) * 1e3,
              **{k: v for k, v in res.tm.items() if k != "track_ms"}}
        with self._lock:
            if self._stream is not stream:
                return
            if ask is not None and not self._det_busy:
                self._det_req, self._det_busy, self._det_t = ask, True, item.t_capture
                self._det_cv.notify_all()
            self._tracked_times.append(time.perf_counter())
            self._tm_track.append(tm)
            self._verdicts.append(verdict)
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
                         "filter": filt, "t_tracked": t3}
            self._job_cv.notify_all()
            if self.trace is not None:
                self.trace.append({"what": "track", "n": item.n, "t": item.t, "tc": item.t_capture, "ok": res.ok,
                                   "size": list(item.size),
                                   "sure": res.sure, "T_raw": res.T, "T": T, "vis": vis, "state": state,
                                   "verdict": verdict, "corners": res.n, "rms_px": res.rms, "source": res.source,
                                   "reanchored": res.reanchored, "shift_px": res.shift_px, "flipped": res.flipped,
                                   "noise_px": res.noise_px,
                                   "t_done": t3, "tm": tm, "detection": None if det is None else det.n})

    # ── thread 2: the detection, now and then ────────────────────────────
    def _detect_loop(self):
        while not self._quit.is_set():
            with self._det_cv:
                self._det_cv.wait_for(lambda: self._det_req is not None or self._quit.is_set(), timeout=0.2)
                req, self._det_req = self._det_req, None
                stream = self._stream
            if req is None:
                continue
            item, box, mpx, key = req
            try:
                d = self._setup["detector"].detect(item.gray(), box, mpx, n=item.n, t=item.t_capture)
            except Exception as e:  # noqa: BLE001
                self._failed(stream, e)
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
        while not self._quit.is_set():
            with self._job_cv:
                self._job_cv.wait_for(lambda: self._job is not None or self._quit.is_set(), timeout=0.2)
                job, self._job = self._job, None
            if job is None:
                continue
            try:
                self._compose(job)
            except Exception as e:  # noqa: BLE001
                self._failed(job["stream"], e)

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
        t2 = time.perf_counter()
        drawn, masks = {}, {}
        alpha = want["opacity"] * vis
        if T is not None and T_M_B is not None and alpha > 0:
            T_C_B = T @ T_M_B
            for name in ("mandible", "face"):
                rend = self._renderer(name) if want[name] and name in self._models else None
                if rend is not None:
                    gl = type(rend).__name__ == "GLMeshRenderer"     # it can leave out the edges: the outline comes from the mask
                    r = rend.render(Kd, dist, T_C_B, (dw, dh), edges=False) if gl else rend.render(Kd, dist, T_C_B, (dw, dh))
                    drawn[name] = int(np.count_nonzero(r.mask))
                    masks[name] = r.mask
                    st = STYLE[name]
                    shade = self._shade(name, rend, T_C_B, st["shade"]) if st["shade"] > 0 else None
                    self._paint(img, r, self.colors[name], st, alpha, shade)
        t3 = time.perf_counter()
        if want["axes"] and T is not None and vis > 0.5:
            self._draw_board(img, T, Kd, dist, spec, setup, res, s)
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
            st = self._status_of(job["filter"], state, kind, vis, T, cam, spec, drawn, w, h, dw, dh, t5 - item.t)
            self._status = st
            if kind == "measured" and res.ok:
                self._masks = (item.n, masks)
            if ok:
                self._jpeg, self._jpeg_n, self._bgr, self._jpeg_t = buf.tobytes(), self._jpeg_n + 1, img, t5
            if self.trace is not None:
                self.trace.append({"what": "shown", "n": item.n, "t": item.t, "tc": item.t_capture, "t_done": t5,
                                   "kind": kind, "state": state, "vis": vis, "T": T, "drawn": drawn, "tm": tm_show})
            self._frame_ready.notify_all()

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
                q = np.round(p * 16).astype(np.int32)
                for k, c in ((1, (60, 60, 230)), (2, (60, 200, 60)), (3, (230, 120, 40))):
                    cv2.line(img, tuple(q[0]), tuple(q[k]), c, 2, cv2.LINE_AA, 4)
        if res is not None and res.ok and len(res.corners):
            c = (res.corners + 0.5) * s - 0.5
            inside = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < w) & (c[:, 1] < h)
            for x, y in np.round(c[inside] * 16).astype(np.int32):
                cv2.circle(img, (int(x), int(y)), 2 * 16, self.colors["corners"], -1, cv2.LINE_AA, 4)

    def _status_of(self, filt, state, kind, vis, T, cam, spec, drawn, w, h, dw, dh, processing):
        """The status after a frame was drawn (under the lock)."""
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
        st = {"tracked": filt.tracked, "state": state, "kind": kind, "opacity_now": round(float(vis), 2),
              "corners": latest.get("corners", 0), "markers": latest.get("markers", 0),
              "rms_px": None if latest.get("rms_px") is None else round(latest["rms_px"], 3),
              "rms_limit_px": MAX_RMS[cam["source"]], "min_corners": board_tracker.TRACK["min_corners"],
              "intrinsics": cam, "drawn_px": drawn, "timings": tm, "image": [w, h], "display": [dw, dh],
              "fps_ar": rate(self._shown_times), "fps_pose": rate(self._tracked_times),
              "fps_detect": rate(self._det_times), "source": latest.get("source"), "sure": latest.get("sure"),
              "tracking": latest.get("counts", {}), "skipped": self._skipped,
              "missed_share": round(sum(x == "missed" for x in v) / len(v), 3) if v else 0.0,
              "rejected": filt.counts["rejected"]}
        if T is not None:
            nx, ny = spec["squares"]
            centre = np.array([nx, ny, 0.0]) * [spec["square_mm"], spec["square_mm"], 0] / 2
            st["distance_mm"] = round(float(np.linalg.norm(marker_pose.transform(T, centre[None])[0])), 1)
            st["tilt_deg"] = round(float(np.degrees(np.arccos(min(1.0, abs(T[2, 2]))))), 1)
        stream_ms = 0.0 if self.synthetic or self.replay is not None else STREAM_LATENCY_MS.get(self._link)
        st["latency"] = {"stream_ms": stream_ms, "processing_ms": round(processing * 1e3, 1),
                         "total_ms": round((stream_ms or 0.0) + processing * 1e3, 1), "link": self._link}
        return st

    def _failed(self, stream, e):
        if stream is not None and self._stream is stream:
            self._error = f"{type(e).__name__}: {e}"
            self.log(self._error)
        time.sleep(0.01)

    def _shade(self, name, rend, T, depth):
        """Brightness of each triangle under a light at the camera (both sides lit), from 1 face on down to 1 - depth
        edge on: a light hint of the shape."""
        if name not in self._centroids:
            self._centroids[name] = rend.v[rend.f].mean(1)
        c = self._centroids[name] @ T[:3, :3].T + T[:3, 3]
        cos = np.abs(np.sum((rend.n @ T[:3, :3].T) * c, 1)) / np.maximum(np.linalg.norm(c, axis=1), 1e-9)
        # 0 first: the visibility buffer holds -1 for the background, so buffer + 1 indexes this table directly
        return np.concatenate([[0], np.rint(255 * (1 - depth * (1 - cos)))]).astype(np.uint8)

    @staticmethod
    def _paint(img, r, colour, style, alpha=1.0, shade=None):
        """The visible model as a tint over the image and a thin smooth outline; alpha (the opacity setting times
        the fade) scales both."""
        x0, y0 = r.offset
        bh, bw = r.faces.shape
        if not (bh and bw) or alpha <= 0:
            return
        roi = img[y0:y0 + bh, x0:x0 + bw]
        m = r.mask[y0:y0 + bh, x0:x0 + bw]
        if not cv2.countNonZero(m):
            return
        if shade is None:
            tint = np.empty_like(roi)
            tint[:] = colour
        else:
            light = cv2.cvtColor(shade[r.faces + 1], cv2.COLOR_GRAY2BGR)
            tint = cv2.multiply(light, tuple(c / 255 for c in colour) + (0,))
        a = min(1.0, style["fill"] * alpha)
        cv2.copyTo(cv2.addWeighted(roi, 1 - a, tint, a, 0), m, roi)
        outline, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        lines = []
        for c in outline:
            c = c.reshape(-1, 2).astype(np.float32)
            if len(c) >= 7:      # the pixel staircase of the mask, smoothed along the contour
                c = sum(np.roll(c, k, 0) for k in (-2, -1, 0, 1, 2)) / 5
            lines.append(np.round(c * 8).astype(np.int32))
        la = min(1.0, style["line"] * alpha)
        if la >= 0.999:
            cv2.polylines(roi, lines, True, colour, 1, cv2.LINE_AA, 3)
        else:
            layer = roi.copy()
            cv2.polylines(layer, lines, True, colour, 1, cv2.LINE_AA, 3)
            cv2.addWeighted(layer, la, roi, 1 - la, 0, dst=roi)


# ── the frames for the page ──────────────────────────────────────────────────
class MjpegServer:
    """The composed frames as MJPEG (multipart/x-mixed-replace) on 127.0.0.1, for an <img> of the page or a browser.
    The path carries a random token, so only who was given the URL reads the camera."""

    def __init__(self, ar, port=0):
        self.ar = ar
        self.token = secrets.token_hex(8)
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == f"/{server.token}.jpg":
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
                if path != f"/{server.token}.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                n = 0
                try:
                    while not server.closed:
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
        self.closed = True
        self.httpd.shutdown()
        self.httpd.server_close()


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
    (NaN where none), kept beside the clip (CLIP_reference_<board>.npz)."""
    from scipy.signal import butter, filtfilt
    clip = Path(clip)
    out = clip.with_name(clip.stem + f"_reference_{spec['name']}_{spec['squares'][0]}x{spec['squares'][1]}_v2.npz")
    if cache and out.is_file():
        return np.load(out)["T_C_M"]
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
    true T_C_M of each frame), so that any version of the live AR can be measured on the same frames."""
    from fractions import Fraction

    import av
    out = Path(out).with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    b, spec = make_board(board, square_mm)
    K, _, _ = stream_intrinsics((1920, 1440))
    s = SyntheticStream(b, spec, K, script=bench_script(spec), exposure_s=exposure_s, noise=noise, seed=1)
    s.close()
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
    np.savez_compressed(str(out) + "_truth.npz", T_C_M=np.array(truth))
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
    checks = {}
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
            checks[n] = (meta["image"], (m & face).sum() / max(face.sum(), 1), (m & front).sum() / max(m.sum(), 1),
                         e["rot_deg"], e["trans_mm"])
            if out is not None and len(checks) == 1:
                cv2.imwrite(str(out), ar.composed())
        st = ar.status()
        ar.stop()
    ar.replay, ar.replay_truth, ar.intrinsics_path = saved[0], saved[1], saved[2]
    ar.set_registration(saved[3]["registration"] or None)
    ar.configure(board=saved[3]["board"], mandible=saved[3]["mandible"], face=saved[3]["face"])
    ar.set_tracking(**saved[4])
    c = np.array([v[1:] for v in checks.values()]).reshape(-1, 4)
    ok = len(c) >= min(4, len(images)) and c[:, 0].min() >= 0.95 and c[:, 1].min() >= 0.95
    expect(ok and st["registration"]["source"] == "file" and st["intrinsics"]["source"] == "calibrated",
           f"digital twin, {len(c)} of {len(images)} frames ({TWIN.name}, patient board on the bench): face drawn "
           f"over the labelled face, recall min {c[:, 0].min() if len(c) else float('nan'):.3f}, precision min "
           f"{c[:, 1].min() if len(c) else float('nan'):.3f}; board pose against the truth max "
           f"{c[:, 2].max() if len(c) else float('nan'):.3f} deg, {c[:, 3].max() if len(c) else float('nan'):.2f} mm")


def run_test():
    """Pose filter; board tracker; synthetic stream through the whole pipeline: pose against the truth, overlay
    drawn, light, video at the stream rate, file registration, no board, a hand over the board, MJPEG; the frames of
    the digital twin with the true registration; the opacity; frames that turn upright and back; a stream that stalls
    and then comes in a burst."""
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
    ar = LiveAR("sheet", 24.0, synthetic=True, mandible=True, log=print)
    try:
        for board in ("sheet", "patient"):
            ar.configure(board=board)
            st, errs = run(ar, 4.0)
            ar.stop()
            tr = st.get("timings", {})
            # bounds of a working pipeline, not of its accuracy (T5.4 measures that); the 45 mm board is the looser
            rot, trans = (0.3, 2.0) if board == "sheet" else (1.0, 3.0)
            expect(st.get("tracked") and len(errs) > 20 and errs[:, 0].max() < rot and errs[:, 1].max() < trans,
                   f"{board}: tracked, pose against the truth over {len(errs)} frames: rotation max "
                   f"{errs[:, 0].max() if len(errs) else float('nan'):.3f} deg, translation max "
                   f"{errs[:, 1].max() if len(errs) else float('nan'):.2f} mm (camera at about "
                   f"{st.get('distance_mm')} mm)")
            expect(st.get("rms_px") is not None and st["rms_px"] < 0.5 and st["corners"] >= 8,
                   f"{board}: {st.get('corners')} of {st['board']['corners']} corners, RMS {st.get('rms_px')} px")
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
        turn_check(ar, expect)
        burst_check(ar, expect)
    finally:
        ar.shutdown()
    # no board in view: grey light
    board, spec = make_board("sheet", 24.0)
    K, _, _ = stream_intrinsics((1920, 1440))
    empty = SyntheticStream(None, spec, K).wait_first()
    ar2 = LiveAR("sheet", 24.0)
    ar2.attach(empty)
    time.sleep(1.0)
    st = ar2.status()
    ar2.shutdown()
    expect(not st.get("tracked") and st.get("corners", -1) == 0 and st.get("fps_ar", 0) > 20,
           f"no board in view: not tracked, grey light, video still at {st.get('fps_ar')} fps")
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


def turn_check(ar, expect):
    """Frames that turn upright (1440 x 1920) and back, as a phone turned in the hand could send: no error, the
    video goes on at every size, the board is found again across."""
    board, spec = make_board("sheet", 24.0)
    K, _, _ = stream_intrinsics((1920, 1440))
    s = SyntheticStream(board, spec, K, exposure_s=0.0)
    s.close()
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
    found = {}
    for r in rows:
        if r["what"] == "track":
            found.setdefault(tuple(r["size"]), []).append(r["ok"])
    expect(not st.get("error") and len(shown) >= 25 and len(sizes) == 2
           and all(np.mean(v) > 0.5 for v in found.values()),
           f"frames turned upright and back: no error ({st.get('error') or 'none'}), {len(shown)} frames shown, "
           f"board found on " + ", ".join(f"{k[0]}x{k[1]} {np.mean(v):.0%}" for k, v in found.items()))


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["view", "test", "bench", "clip"])
    ap.add_argument("out", nargs="?", help="clip: the file to write (OUT.h264, OUT_times.npy, OUT_truth.npz)")
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
