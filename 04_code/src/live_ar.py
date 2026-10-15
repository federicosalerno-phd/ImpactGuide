"""Live AR on the phone stream (T20.3; module 8 of ai_design.md): board pose, face overlay and status light.

One worker thread takes the newest frame of the stream (live.py), finds the ChArUco corners and solves the board pose
(marker_pose.py: IPPE, then Levenberg-Marquardt), renders the face, and the mandible if asked, at
T_C_B = T_C_M . T_M_B with the OpenGL renderer (gl_overlay.py) at the display size, draws the board axes, the detected
corners and the corners reprojected by the pose, and keeps the composed frame as JPEG. The pose is computed on the
frame it is drawn on, so the overlay comes late (the stream's latency, about 195 ms, plus the processing) but aligned.

Frames as ai_design.md (Notazione): C camera (OpenCV), M board (cv2.aruco.CharucoBoard: origin at the outer corner of
the first square, x along the printed columns, y along the rows, z into the board), B head (bench scene, mm).

Boards
    patient   the 5 x 5 board of the marker holder; squares and markers from marker_pose.BOARD
    sheet     the A4 calibration board of calibrate_charuco.py (11 x 8), square as measured with the caliper: a sheet
              of paper is enough to try the page without the bench
A frame is tracked with at least MIN_CORNERS corners and a reprojection RMS under the limit: 1 px with intrinsics
calibrated for the stream (ai_design.md, module 1), 3 px with the nominal ones, which leave out the lens distortion.

Intrinsics of the streaming mode: 03_data/processed/calibration/intrinsics_stream.json, in the format
calibrate_charuco.py writes for intrinsics_photo.json (T5.3), calibrated on frames of the stream at the working
distance; scaled if the stream has another size. Until it exists: nominal Poco X7 Pro, 3070 px at the 3072 px height of
the 4:3 sensor field (as synth_twin.py), so 1439 px at 1920 x 1440, principal point at the centre, no distortion.

Registration T_M_B: a JSON file holding a 4 x 4 "T_M_B" in mm (what ground_truth.py measure writes). Until there is
one, a demo registration at a fixed offset from the board, which the page always labels as demo:
    patient   the registration of the bench design (the scene, meshes/occluders.npz of synth_twin.py)
    sheet     the face lying on the sheet, centred, looking up

Models in B (mm): the decimated face of T13.1 (synth_twin.py meshes) and the mandible of the bench scene
(build_targets.py), both under C:\\work\\marker-impact and written again by those scripts.

For the app:
    ar = LiveAR(board="sheet", square_mm=24.0)       # one per process: its overlay thread owns the OpenGL context
    ar.start()                                       # closes the camera app, opens the stream (a few seconds)
    jpeg, n = ar.next_jpeg(after=n)                  # newest composed frame
    ar.status()                                      # light, corners, RMS, fps, latency, timings, sources
    ar.configure(board=..., square_mm=..., face=..., mandible=..., axes=...); ar.set_registration(path or None)
    ar.stop(); ar.shutdown()
    MjpegServer(ar).url                              # the frames for an <img>, on 127.0.0.1

    python live_ar.py view  [--board patient|sheet] [--square MM] [--registration FILE] [--mandible]
                            [--synthetic | --replay VIDEO_OR_FOLDER] [--intrinsics FILE]
    python live_ar.py test          synthetic stream: pose against the truth, overlay, light, MJPEG; frames of the
                                    digital twin with the true registration; exit 1 on failure
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
import calibrate_charuco
import live
import marker_pose

ROOT = Path(__file__).resolve().parents[2]
STREAM_INTRINSICS = ROOT / "03_data" / "processed" / "calibration" / "intrinsics_stream.json"
WORK = Path(r"C:\work\marker-impact")
FACE_MESH = WORK / "synth" / "meshes" / "face_phantom_r20k.npz"
DESIGN = WORK / "synth" / "meshes" / "occluders.npz"          # T_M_B of the bench scene
MANDIBLE_MESH = WORK / "targets" / "mandible.npz"
TWIN = WORK / "synth" / "sessions"                              # frames of the digital twin, for the test
CAMERA_APP = acquisition.CAMERA_APP     # the phone's camera app holds the camera while it is open
NOMINAL_F = 3070.0 / 3072.0             # focal length over the height of the 4:3 sensor field (synth_twin.NOMINAL)
STREAM_LATENCY_MS = 195.0               # camera to decoded frame, median of live.py latency (15/10)
MIN_CORNERS = 8
MAX_RMS = {"calibrated": 1.0, "nominal": 3.0}
DISPLAY_WIDTH = 1280
JPEG_QUALITY = 80
FILL_ALPHA = {"face": 0.45, "mandible": 0.60}
SHADE_FLOOR = 0.35                      # brightness of a triangle seen edge on (1 face on)
COLORS = {"face": "#F5C542", "mandible": "#E6E6EC", "corners": "#58B36B", "reprojected": "#E6E6EC"}
BOARDS = ("patient", "sheet")
_GL_THREAD = None                       # the thread that owns gl_overlay's OpenGL context


class LiveARError(RuntimeError):
    pass


# ── boards, intrinsics, registration, models ─────────────────────────────────
def make_board(name, square_mm=None):
    """(cv2 CharucoBoard, description) of the patient board or of the A4 calibration sheet."""
    if name == "patient":
        b = marker_pose.BOARD
        board = marker_pose.make_board(b)
        n = b["squares"]
        spec = {"name": name, "squares": [n, n], "square_mm": float(b["square_mm"]), "marker_mm": float(b["marker_mm"]),
                "dictionary": b["dictionary"]}
    elif name == "sheet":
        sq = float(square_mm or calibrate_charuco.BOARD["square"])
        board = calibrate_charuco.make_board(sq)
        nx, ny = calibrate_charuco.BOARD["squares"]
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
    """K, dist and their source ({"source": calibrated|nominal, "path", "f"}) for a stream of size (w, h)."""
    w, h = size
    if path and Path(path).is_file():
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        cw, ch = d["image_size"]
        K = scale_K(d["K"], w / cw, h / ch)
        dist = np.asarray(d.get("dist", np.zeros(5)), np.float64).ravel()
        src = {"source": "calibrated", "path": str(path), "date": d.get("date", ""), "rms_px": d.get("rms_px")}
    else:
        f = NOMINAL_F * h
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


def load_models():
    """{name: (vertices in B, faces)} of the models that are on this computer."""
    out = {}
    if FACE_MESH.is_file():
        z = np.load(FACE_MESH)
        out["face"] = (z["vertices"].astype(np.float64), z["faces"].astype(np.int64))
    if MANDIBLE_MESH.is_file():
        z = np.load(MANDIBLE_MESH)
        T = z["T_B_mandible"]
        out["mandible"] = (z["v"].astype(np.float64) @ T[:3, :3].T + T[:3, 3], z["f"].astype(np.int64))
    for name, (v, f) in out.items():
        out[name] = (v, consistent(f, len(v))[0])
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


# ── a stand-in for the phone ─────────────────────────────────────────────────
class SyntheticStream:
    """Stand-in for live.PhoneStream: the board on a grey table, filmed by a camera that sways above it, rendered with
    the same pinhole camera the page uses. truth(n) is the true T_C_M of frame n. With board None there is no board."""

    def __init__(self, board, spec, K, size=(1920, 1440), fps=30, seed=0):
        self.size, self.fps, self.K = tuple(size), fps, np.asarray(K, np.float64)
        self.spec = spec
        self.error, self.restarts = None, 0
        self._lock = threading.Lock()
        self._frame, self._t, self._n = None, 0.0, 0
        self._times = deque(maxlen=120)
        self._truth = {}
        self._stop = threading.Event()
        rng = np.random.default_rng(seed)
        w, h = self.size
        bg = cv2.resize(rng.uniform(70, 125, (9, 12)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        self._bg = np.clip(bg, 0, 255).astype(np.uint8)
        self._noise = []
        for _ in range(4):       # sensor noise, sigma 2 grey levels, as a part to add and a part to take away
            z = rng.normal(0, 2.0, (h, w))
            self._noise.append((np.clip(z, 0, 255).astype(np.uint8), np.clip(-z, 0, 255).astype(np.uint8)))
        self._tex = None
        if board is not None:
            ppm = 12.0 if spec["name"] == "sheet" else 24.0
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
        self.thread = threading.Thread(target=self._run, daemon=True, name="synthetic_stream")
        self.thread.start()

    def pose(self, t):
        """T_C_M at time t (s): the camera sways in front of the board, below its bottom edge, tilted 20 to 40 degrees
        from the normal, and looks at its centre with the top of the board up in the image."""
        tilt = np.radians(30 + 10 * np.sin(2 * np.pi * t / 6.5))
        phi = np.radians(90 + 40 * np.sin(2 * np.pi * t / 9.0))
        d = self._dist * (1 + 0.08 * np.sin(2 * np.pi * t / 5.0))
        eye = self._centre + d * np.array([np.sin(tilt) * np.cos(phi), np.sin(tilt) * np.sin(phi), -np.cos(tilt)])
        z = (self._centre - eye) / d
        y = np.array([0, 1.0, 0]) - z[1] * z
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        R_M_C = np.stack([x, y, z], 1)
        T = np.eye(4)
        T[:3, :3] = R_M_C.T
        T[:3, 3] = -R_M_C.T @ eye
        return T

    def _render(self, T):
        img = self._bg.copy()
        if self._tex is not None:
            H = self.K @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ self._S
            cv2.warpPerspective(self._tex, H, self.size, dst=img, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_TRANSPARENT)
        plus, minus = self._noise[self._n % len(self._noise)]
        return cv2.cvtColor(cv2.subtract(cv2.add(img, plus), minus), cv2.COLOR_GRAY2BGR)

    def _run(self):
        t0 = time.perf_counter()
        period = 1.0 / self.fps
        nxt = t0
        while not self._stop.is_set():
            T = self.pose(time.perf_counter() - t0)
            img = self._render(T)
            t = time.perf_counter()
            with self._lock:
                self._frame, self._t, self._n = img, t, self._n + 1
                self._truth[self._n] = T
                self._truth.pop(self._n - 300, None)
                self._times.append(t)
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def latest(self):
        with self._lock:
            return self._frame, self._t, self._n

    def truth(self, n):
        with self._lock:
            return self._truth.get(n)

    def stats(self):
        with self._lock:
            t = np.array(self._times)
            n = self._n
        if len(t) < 2:
            return {"fps": 0.0, "interval_ms": float("nan"), "interval_p95_ms": float("nan"), "frames": n}
        dt = np.diff(t) * 1e3
        return {"fps": float((len(t) - 1) / (t[-1] - t[0])), "interval_ms": float(dt.mean()),
                "interval_p95_ms": float(np.percentile(dt, 95)), "frames": n}

    @property
    def alive(self):
        return self.thread.is_alive()

    def close(self):
        self._stop.set()

    def wait_first(self, timeout=5.0):
        t0 = time.perf_counter()
        while self.latest()[0] is None and time.perf_counter() - t0 < timeout:
            time.sleep(0.01)
        return self


class ReplayStream:
    """Stand-in for live.PhoneStream that plays recorded frames in a loop: a list of images, a folder of images or a
    video. For trying the page on material recorded with the phone, and for the test on the frames of the digital twin;
    truth(n) gives back the item of `truth` that goes with frame n, when one is given."""

    def __init__(self, source, fps=10, truth=None):
        self.fps, self.error, self.restarts = fps, None, 0
        src = source if isinstance(source, (list, tuple)) else [source]
        src = [Path(p) for p in src]
        if len(src) == 1 and src[0].is_dir():
            # the digital twin keeps labels, masks and depth beside its images: they are not frames
            src = sorted(p for p in src[0].iterdir() if p.suffix.lower() in calibrate_charuco.IMAGE_SUFFIXES
                         and not p.stem.endswith(("_label", "_model", "_depth")))
        self._video = src[0] if len(src) == 1 and src[0].suffix.lower() in (".mp4", ".mov", ".mkv", ".avi") else None
        self._images = [] if self._video else src
        if not self._video and not self._images:
            raise LiveARError(f"nothing to play in {source}")
        self._truth = list(truth) if truth is not None else None
        self._lock = threading.Lock()
        self._frame, self._t, self._n = None, 0.0, 0
        self._times = deque(maxlen=120)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="replay_stream")
        self.thread.start()

    def _frames(self):
        while not self._stop.is_set():
            if self._video:
                cap = cv2.VideoCapture(str(self._video))
                ok, img = cap.read()
                if not ok:
                    self.error = f"cannot read {self._video.name}"
                    return
                while ok and not self._stop.is_set():
                    yield img
                    ok, img = cap.read()
                cap.release()
            else:
                for p in self._images:
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        self.error = f"cannot read {p.name}"
                        return
                    yield img

    def _run(self):
        period, nxt = 1.0 / self.fps, time.perf_counter()
        for img in self._frames():
            t = time.perf_counter()
            with self._lock:
                self._frame, self._t, self._n = img, t, self._n + 1
                self._times.append(t)
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))
            if self._stop.is_set():
                return

    latest = SyntheticStream.latest
    stats = SyntheticStream.stats
    close = SyntheticStream.close
    wait_first = SyntheticStream.wait_first

    @property
    def alive(self):
        return self.thread.is_alive()

    def truth(self, n):
        if not self._truth:
            return None
        k = len(self._images) if self._images else len(self._truth)
        return self._truth[(n - 1) % k] if n > 0 else None


# ── the live AR ──────────────────────────────────────────────────────────────
class LiveAR:
    """Phone stream -> board pose -> face overlay.

    Two threads for the life of the object: the pose (stage 1) and the overlay (stage 2), working side by side on
    consecutive frames. The OpenGL context belongs to the overlay thread, so that thread is never replaced: stop() and
    start() only close and open the stream. Meant as one per process (the app keeps one); a later one makes its own
    OpenGL context."""

    def __init__(self, board="sheet", square_mm=None, registration=None, face=True, mandible=False, axes=True,
                 intrinsics=STREAM_INTRINSICS, synthetic=False, serial=None, size="1920x1440", fps=30,
                 display_width=DISPLAY_WIDTH, colors=None, log=None, replay=None, replay_truth=None):
        if board not in BOARDS:
            raise LiveARError(f"unknown board {board}")
        self.intrinsics_path = Path(intrinsics) if intrinsics else None
        self.synthetic, self.serial, self.size, self.fps = synthetic, serial, size, fps
        self.replay, self.replay_truth = replay, replay_truth     # recorded frames in place of the phone
        self.display_width = display_width
        self.colors = {k: bgr(v) for k, v in {**COLORS, **(colors or {})}.items()}
        self.log = log or (lambda msg: None)
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._want = {"board": board, "square_mm": float(square_mm or calibrate_charuco.BOARD["square"]),
                      "registration": str(registration) if registration else "", "face": bool(face),
                      "mandible": bool(mandible), "axes": bool(axes), "version": 1}
        self._stream = None
        self._starting = False
        self._jpeg, self._jpeg_n, self._bgr = None, 0, None
        self._pose = (0, None)            # (frame counter, T_C_M) of the last tracked frame
        self._masks = (0, {})             # (frame counter, {model: silhouette at the display size}) of that frame
        self._times = deque(maxlen=60)
        self._status = {}
        self._error = ""
        self._quit = threading.Event()
        self._models = load_models()
        self._renderers, self._centroids = {}, {}
        self.renderer = ""
        self._applied = 0
        self._setup = None
        self._cam = (None, None, {})       # K, dist, source for the size of the last frame
        self._cam_size = None
        self._found, self._found_cv = None, threading.Condition()
        self._apply(dict(self._want))
        self._thread = threading.Thread(target=self._work, daemon=True, name="live_ar_overlay")
        self._det_thread = threading.Thread(target=self._detect_loop, daemon=True, name="live_ar_pose")
        self._thread.start()
        self._det_thread.start()

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
                stream = ReplayStream(self.replay, min(self.fps, 10), self.replay_truth).wait_first()
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
        with self._lock:
            self._stream = stream
            self._times.clear()
            self._status = {}
            self._pose, self._masks = (0, None), (0, {})     # nothing of the stream before
            self._cam_size = None          # intrinsics read again: a calibration may have been saved meanwhile
        return self

    def _synthetic_stream(self):
        w, h = (int(x) for x in self.size.split("x"))
        K, _, _ = stream_intrinsics((w, h), self.intrinsics_path)
        return SyntheticStream(self._setup["board"], self._spec, K, (w, h), self.fps).wait_first()

    def stop(self):
        with self._lock:
            stream, self._stream = self._stream, None
            self._jpeg, self._bgr, self._found = None, None, None
            self._frame_ready.notify_all()
        if stream is not None:
            stream.close()

    def shutdown(self):
        self.stop()
        self._quit.set()
        with self._found_cv:
            self._found_cv.notify_all()
        self._thread.join(timeout=2.0)
        self._det_thread.join(timeout=2.0)

    def configure(self, **kw):
        """board, square_mm, face, mandible, axes: applied by the worker from the next frame."""
        if kw.get("board") is not None and kw["board"] not in BOARDS:
            raise LiveARError(f"unknown board {kw['board']}")
        with self._lock:
            if kw.get("board") is not None:
                self._want["board"] = kw["board"]
            if kw.get("square_mm") is not None:
                self._want["square_mm"] = float(kw["square_mm"])
            for k in ("face", "mandible", "axes"):
                if kw.get(k) is not None:
                    self._want[k] = bool(kw[k])
            self._want["version"] += 1
            want = dict(self._want)
        if not self.running:
            self._apply(want)
        elif self.synthetic and want["board"] != self._spec["name"]:
            self._apply(want)
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

    def composed(self):
        """The newest composed frame as a BGR array (None when stopped)."""
        with self._lock:
            return self._bgr

    def last_masks(self):
        """(stream frame counter, {model: silhouette}) of the last tracked frame, at the display size."""
        with self._lock:
            return self._masks

    def last_pose(self):
        """(stream frame counter, T_C_M) of the last tracked frame."""
        with self._lock:
            return self._pose

    def stream(self):
        return self._stream

    def settings(self):
        """The settings asked for: board, square_mm, registration, face, mandible, axes."""
        with self._lock:
            return {k: v for k, v in self._want.items() if k != "version"}

    def status(self):
        with self._lock:
            st = dict(self._status)
            stream = self._stream
            want = dict(self._want)
        st.update({"running": stream is not None, "starting": self._starting, "error": self._error,
                   "synthetic": self.synthetic or self.replay is not None, "replay": self.replay is not None,
                   "renderer": self.renderer,
                   "layers": {k: want[k] for k in ("face", "mandible", "axes")},
                   "models": {k: k in self._models for k in ("face", "mandible")},
                   "board": self._spec, "registration": self._setup["registration"], "score": None})
        if not st.get("intrinsics"):
            st["intrinsics"] = self._cam[2] or stream_intrinsics(self._size_wh(), self.intrinsics_path)[2]
        if stream is None:
            st.update({"tracked": False, "fps_stream": 0.0, "fps_ar": 0.0})
            return st
        s = stream.stats()
        st["fps_stream"] = round(s["fps"], 1)
        st["frames"] = s["frames"]
        if stream.error:
            st["stream_error"] = stream.error
        if not stream.alive:
            st["error"] = stream.error or "stream stopped"
        return st

    def _size_wh(self):
        w, h = (int(x) for x in self.size.split("x"))
        return w, h

    # ── the worker ───────────────────────────────────────────────────────
    def _apply(self, want):
        """Board, detector and registration for the wanted settings (worker thread, or any thread while stopped)."""
        old = self._setup
        board, spec = make_board(want["board"], want["square_mm"])
        if old is not None and spec == old["spec"]:
            board, detector = old["board"], old["detector"]
        else:
            detector = marker_pose.make_detector(board)
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
        # one assignment, so the worker never sees the board of one setting with the detector of another
        self._setup = {"board": board, "spec": spec, "detector": detector, "T_M_B": T, "registration": reg}
        self._applied = want["version"]

    @property
    def _spec(self):
        return self._setup["spec"]

    def _renderer(self, name):
        """The renderer of a model, made in the worker thread (OpenGL, or NumPy when OpenGL is not there)."""
        if name not in self._renderers:
            v, f = self._models[name]
            try:
                import gl_overlay
                global _GL_THREAD
                if _GL_THREAD not in (None, threading.get_ident()):
                    # gl_overlay keeps one context, bound to the thread that made it: the overlay thread of an
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

    def _detect_loop(self):
        """Stage 1, a thread of its own: the newest frame, its corners and the board pose. OpenCV leaves the GIL while
        it works, so this runs alongside stage 2 and the overlay keeps the pace of the slower stage, not of the sum."""
        last = (None, -1)
        while not self._quit.is_set():
            stream = self._stream
            if stream is None:
                time.sleep(0.02)
                continue
            frame, t_dec, n = stream.latest()
            if frame is None or (stream, n) == last:
                time.sleep(0.003)
                continue
            last = (stream, n)
            with self._lock:
                want = dict(self._want)
            try:
                if want["version"] != self._applied:
                    self._apply(want)
                found = self._detect(stream, frame, t_dec, n, want, self._setup)
            except Exception as e:  # noqa: BLE001  a bad frame must not end the thread
                self._failed(stream, e)
                continue
            with self._found_cv:
                self._found = found
                self._found_cv.notify_all()

    def _work(self):
        """Stage 2, the thread that owns the OpenGL context: overlay, drawing, JPEG, status."""
        done = None
        while not self._quit.is_set():
            with self._found_cv:
                self._found_cv.wait_for(lambda: self._found is not done or self._quit.is_set(), timeout=0.2)
                found = self._found
            if found is None or found is done:
                continue
            done = found
            try:
                self._compose(found)
            except Exception as e:  # noqa: BLE001
                self._failed(found["stream"], e)

    def _failed(self, stream, e):
        if self._stream is stream:
            self._error = f"{type(e).__name__}: {e}"
            self.log(self._error)
        time.sleep(0.05)

    def _detect(self, stream, frame, t_dec, n, want, setup):
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        if self._cam_size != (w, h):
            self._cam = stream_intrinsics((w, h), self.intrinsics_path)
            self._cam_size = (w, h)
        K, dist, cam = self._cam
        limit = MAX_RMS[cam["source"]]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, n_markers = marker_pose.detect(setup["detector"], gray)
        t1 = time.perf_counter()
        pose = (marker_pose.estimate_pose(setup["board"], corners, ids, K, dist, MIN_CORNERS)
                if len(ids) >= MIN_CORNERS else None)
        t2 = time.perf_counter()
        return {"stream": stream, "frame": frame, "t_dec": t_dec, "n": n, "want": want, "setup": setup, "K": K,
                "dist": dist, "cam": cam, "limit": limit, "corners": corners, "ids": ids, "n_markers": n_markers,
                "pose": pose, "tracked": pose is not None and pose["rms_px"] <= limit,
                "tm": {"detect_ms": (t1 - t0) * 1e3, "pose_ms": (t2 - t1) * 1e3}}

    def _compose(self, d):
        stream, frame, want, pose, tracked = d["stream"], d["frame"], d["want"], d["pose"], d["tracked"]
        board, spec, T_M_B = d["setup"]["board"], d["setup"]["spec"], d["setup"]["T_M_B"]
        K, dist, cam, limit, corners, ids, n_markers = (d[k] for k in ("K", "dist", "cam", "limit", "corners", "ids",
                                                                         "n_markers"))
        n, t_dec, tm = d["n"], d["t_dec"], dict(d["tm"])
        t2 = time.perf_counter()
        h, w = frame.shape[:2]
        dw = min(self.display_width, w)
        s = dw / w
        dh = int(round(h * s))
        img = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR) if dw != w else frame.copy()
        Kd = scale_K(K, s, s)
        drawn, masks = {}, {}
        if tracked and T_M_B is not None:
            T_C_B = pose["T"] @ T_M_B
            for name in ("mandible", "face"):
                rend = self._renderer(name) if want[name] and name in self._models else None
                if rend is not None:
                    gl = type(rend).__name__ == "GLMeshRenderer"     # it can leave out the edges: the outline comes from the mask
                    r = rend.render(Kd, dist, T_C_B, (dw, dh), edges=False) if gl else rend.render(Kd, dist, T_C_B, (dw, dh))
                    drawn[name] = int(np.count_nonzero(r.mask))
                    masks[name] = r.mask
                    self._paint(img, r, self.colors[name], FILL_ALPHA[name], self._shade(name, rend, T_C_B))
        t3 = time.perf_counter()
        tm["render_ms"] = (t3 - t2) * 1e3
        if tracked and want["axes"]:
            rvec = cv2.Rodrigues(pose["T"][:3, :3])[0]
            length = 3 * spec["square_mm"]
            cv2.drawFrameAxes(img, Kd, dist, rvec, pose["T"][:3, 3], length, 2)
        if len(ids):
            pts = corners * s
            if pose is not None:
                proj = marker_pose.project(board.getChessboardCorners()[ids], pose["T"], Kd, dist)
                for p in proj:
                    cv2.circle(img, (int(round(p[0] * 16)), int(round(p[1] * 16))), 2 * 16,
                               self.colors["reprojected"], -1, cv2.LINE_AA, 4)
            for p in pts:
                cv2.circle(img, (int(round(p[0] * 16)), int(round(p[1] * 16))), 5 * 16, self.colors["corners"], 1,
                           cv2.LINE_AA, 4)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        t4 = time.perf_counter()
        tm["draw_ms"] = (t4 - t3) * 1e3
        tm["total_ms"] = sum(tm.values())
        processing = (t4 - t_dec) * 1e3
        st = {"tracked": bool(tracked), "corners": int(len(ids)), "markers": int(n_markers),
              "rms_px": None if pose is None else round(pose["rms_px"], 3), "rms_limit_px": limit,
              "min_corners": MIN_CORNERS, "intrinsics": cam, "drawn_px": drawn,
              "timings": {k: round(v, 1) for k, v in tm.items()}, "image": [w, h], "display": [dw, dh]}
        if pose is not None:
            nx, ny = spec["squares"]
            centre = np.array([nx, ny, 0.0]) * [spec["square_mm"], spec["square_mm"], 0] / 2
            st["distance_mm"] = round(float(np.linalg.norm(marker_pose.transform(pose["T"], centre[None])[0])), 1)
            st["tilt_deg"] = round(float(np.degrees(np.arccos(abs(pose["T"][2, 2])))), 1)
        with self._lock:
            if self._stream is not stream:
                return
            self._times.append(t4)
            if len(self._times) > 1:
                st["fps_ar"] = round((len(self._times) - 1) / (self._times[-1] - self._times[0]), 1)
            else:
                st["fps_ar"] = 0.0
            stream_ms = 0.0 if self.synthetic or self.replay is not None else STREAM_LATENCY_MS
            st["latency"] = {"stream_ms": stream_ms, "processing_ms": round(processing, 1),
                             "total_ms": round(stream_ms + processing, 1)}
            self._status = st
            if tracked:
                self._pose = (n, pose["T"])
                self._masks = (n, masks)
            if ok:
                self._jpeg, self._jpeg_n, self._bgr = buf.tobytes(), self._jpeg_n + 1, img
            self._frame_ready.notify_all()

    def _shade(self, name, rend, T):
        """Brightness of each triangle under a light at the camera (both sides lit), so the model reads as a solid."""
        if name not in self._centroids:
            self._centroids[name] = rend.v[rend.f].mean(1)
        c = self._centroids[name] @ T[:3, :3].T + T[:3, 3]
        cos = np.abs(np.sum((rend.n @ T[:3, :3].T) * c, 1)) / np.maximum(np.linalg.norm(c, axis=1), 1e-9)
        # 0 first: the visibility buffer holds -1 for the background, so buffer + 1 indexes this table directly
        return np.concatenate([[0], np.rint(255 * (SHADE_FLOOR + (1 - SHADE_FLOOR) * cos))]).astype(np.uint8)

    @staticmethod
    def _paint(img, r, colour, alpha, shade=None):
        """The visible model as a shaded tint over the image, and its silhouette outlined."""
        x0, y0 = r.offset
        bh, bw = r.faces.shape
        if not (bh and bw):
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
        cv2.copyTo(cv2.addWeighted(roi, 1 - alpha, tint, alpha, 0), m, roi)
        outline, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(roi, outline, -1, colour, 1, cv2.LINE_AA)


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
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
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


# ── command line ─────────────────────────────────────────────────────────────
def cmd_view(a):
    ar = LiveAR(a.board, a.square, a.registration, mandible=a.mandible, synthetic=a.synthetic, log=print,
                replay=a.replay, intrinsics=a.intrinsics or STREAM_INTRINSICS)
    try:
        ar.start()
        n = 0
        while True:
            jpeg, n = ar.next_jpeg(n, timeout=0.5)
            img = ar.composed()
            if img is not None:
                st = ar.status()
                txt = (f"{'TRACKED' if st.get('tracked') else 'no board'}  corners {st.get('corners', 0)}  "
                       f"RMS {st.get('rms_px')}  stream {st.get('fps_stream')} fps  overlay {st.get('fps_ar')} fps  "
                       f"{st['intrinsics']['source']} intrinsics  registration {st['registration'].get('source')}")
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
    front of it (precision); the twin's skin is deformed by up to 4 mm, the overlay is the undeformed phantom."""
    import tempfile
    got = twin_frames()
    if got is None:
        print(f"skip  frames of the digital twin not found in {TWIN}", flush=True)
        return
    rows, images, labels = got
    saved = ar.replay, ar.replay_truth, ar.intrinsics_path, ar.settings()
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
    c = np.array([v[1:] for v in checks.values()]).reshape(-1, 4)
    ok = len(c) >= min(4, len(images)) and c[:, 0].min() >= 0.95 and c[:, 1].min() >= 0.95
    expect(ok and st["registration"]["source"] == "file" and st["intrinsics"]["source"] == "calibrated",
           f"digital twin, {len(c)} of {len(images)} frames ({TWIN.name}, patient board on the bench): face drawn "
           f"over the labelled face, recall min {c[:, 0].min() if len(c) else float('nan'):.3f}, precision min "
           f"{c[:, 1].min() if len(c) else float('nan'):.3f}; board pose against the truth max "
           f"{c[:, 2].max() if len(c) else float('nan'):.3f} deg, {c[:, 3].max() if len(c) else float('nan'):.2f} mm")


def run_test():
    """Synthetic stream through the whole pipeline: pose against the truth, overlay drawn, light, file registration,
    no board, MJPEG; then the frames of the digital twin with the true registration."""
    import tempfile
    import urllib.request
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

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
    expect("face" in models, f"face model found ({FACE_MESH})")
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
            expect(st.get("rms_px") is not None and st["rms_px"] < 0.5 and st["corners"] >= MIN_CORNERS,
                   f"{board}: {st.get('corners')} of {st['board']['corners']} corners, RMS {st.get('rms_px')} px")
            expect(st.get("drawn_px", {}).get("face", 0) > 1000 and st["registration"]["source"] == "demo",
                   f"{board}: face drawn at the demo registration ({st['registration']['demo']}), "
                   f"{st.get('drawn_px', {}).get('face', 0)} px; mandible {st.get('drawn_px', {}).get('mandible', 0)} px")
            expect(st.get("fps_ar", 0) >= 10,
                   f"{board}: overlay {st.get('fps_ar')} fps on a {st['fps_stream']} fps stream ({st['renderer']}), "
                   f"detect {tr.get('detect_ms')} ms, pose {tr.get('pose_ms')} ms, render {tr.get('render_ms')} ms, "
                   f"draw and JPEG {tr.get('draw_ms')} ms")
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
    finally:
        ar.shutdown()
    # no board in view: grey light
    board, spec = make_board("sheet", 24.0)
    K, _, _ = stream_intrinsics((1920, 1440))
    empty = SyntheticStream(None, spec, K).wait_first()
    ar2 = LiveAR("sheet", 24.0)
    ar2._stream = empty
    time.sleep(1.0)
    st = ar2.status()
    ar2.shutdown()
    expect(not st.get("tracked") and st.get("corners", -1) == 0, "no board in view: not tracked, grey light")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["view", "test"])
    ap.add_argument("--board", choices=BOARDS, default="sheet")
    ap.add_argument("--square", type=float, default=None, help="measured square of the A4 sheet, mm")
    ap.add_argument("--registration", help="JSON file with T_M_B (default: the demo registration)")
    ap.add_argument("--mandible", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="a synthetic stream instead of the phone")
    ap.add_argument("--replay", help="recorded frames instead of the phone: a video or a folder of images")
    ap.add_argument("--intrinsics", help="intrinsics JSON (default: the streaming mode's, or the nominal ones)")
    a = ap.parse_args()
    if a.command == "test":
        sys.exit(run_test())
    try:
        cmd_view(a)
    except (live.LiveError, LiveARError) as e:
        sys.exit(f"live_ar: {e}")


if __name__ == "__main__":
    main()
