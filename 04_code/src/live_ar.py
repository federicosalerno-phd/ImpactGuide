"""Live AR on the phone stream (T20.3; module 8 of ai_design.md): board pose, face overlay and status light.

Two kinds of thread work on the frames of the stream (live.py):
  find      TRACKING["workers"] threads, each on the newest frame not yet taken: the ChArUco corners and the board
            pose. While the board is tracked it is searched only in a box around its last position, on an image scaled
            so that the markers are about TRACKING["marker_px"] wide, with one threshold window, and the corners are
            refined at full resolution; otherwise on the whole frame at full resolution (marker_pose.py). PnP as
            marker_pose.py (IPPE, then Levenberg-Marquardt), taking of the two IPPE solutions the one nearer the last
            pose when they fit almost equally well.
  compose   one thread, which owns the OpenGL context: every frame of the stream in order. It waits for the pose of
            that frame for at most TRACKING["wait_ms"], gives the poses found to the tracker (pose_filter.py: One Euro
            filter on SE(3), outlier gate, hold and fade), draws the face (and the mandible) at the filtered pose with
            gl_overlay.py, light (a soft translucent fill and a thin outline), and keeps the frame as JPEG.
So the video in the page keeps the rate of the stream while the pose is found at the rate the computer allows. Each
frame is drawn at the pose filtered up to that frame: when its own pose is not ready in time, at the pose predicted
from the earlier ones (kind "predicted" in the trace). The overlay comes late (the stream's latency, about 195 ms, plus
the processing) but on its own frame.

Frames as ai_design.md (Notazione): C camera (OpenCV), M board (cv2.aruco.CharucoBoard: origin at the outer corner of
the first square, x along the printed columns, y along the rows, z into the board), B head (bench scene, mm).

Boards
    patient   the 5 x 5 board of the marker holder; squares and markers from marker_pose.BOARD
    sheet     the A4 calibration board of calibrate_charuco.py (11 x 8), square as measured with the caliper: a sheet
              of paper is enough to try the page without the bench
A pose is used with at least MIN_CORNERS corners and a reprojection RMS under the limit: 1 px with intrinsics
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
(build_targets.py), both under C:\\work\\marker-impact and written again by those scripts; for the live view each is
decimated again to LIVE_TRIANGLES (quadric edge collapse) and kept in LIVE_MODELS.

For the app:
    ar = LiveAR(board="sheet", square_mm=24.0)       # one per process: its compose thread owns the OpenGL context
    ar.start()                                       # closes the camera app, opens the stream (a few seconds)
    jpeg, n = ar.next_jpeg(after=n)                  # newest composed frame
    jpeg, n, t = ar.next_frame(after=n)              # the same with the time it was composed (phone_view.py)
    ar.status()                                      # light, corners, RMS, fps, latency, timings, sources
    ar.configure(board=..., square_mm=..., face=..., mandible=..., axes=...); ar.set_registration(path or None)
    ar.set_tracking(hold_s=..., ...)                 # TRACKING overrides
    ar.stop(); ar.shutdown()
    MjpegServer(ar).url                              # the frames for an <img>, on 127.0.0.1
    phone_view.ViewServer(ar)                        # the frames for the phone's screen (cable, Wi-Fi, network)

    python live_ar.py view  [--board patient|sheet] [--square MM] [--registration FILE] [--mandible]
                            [--synthetic | --replay VIDEO_OR_FOLDER] [--intrinsics FILE]
    python live_ar.py bench [--synthetic | --replay CLIP.h264] [--seconds S] [--still S]
                            rate, time per stage, jitter while still, frames without a pose, blinks of the overlay
                            (and, on the synthetic stream or a clip with a truth file, the error of the overlay)
    python live_ar.py clip OUT [--seconds S] [--noise SIGMA]
                            the synthetic session as a clip of the phone (raw H.264 with times and truth), for
                            measuring versions of the live AR on the same frames (bench --replay OUT.h264)
    python live_ar.py test  pose filter; board finder against the full search; synthetic stream: pose against the
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
# camera to decoded frame on each link, median of live.py latency: cable 15/10; Wi-Fi not measured yet (None: the
# page says so and counts only the processing)
STREAM_LATENCY_MS = {"usb": 195.0, "wifi": None}
MIN_CORNERS = 8
MAX_RMS = {"calibrated": 1.0, "nominal": 3.0}
DISPLAY_WIDTH = 1280
JPEG_QUALITY = 80
# the look of the overlay: a soft fill (its opacity), a thin outline (its opacity) and a light shading (its depth:
# 0 flat, 1 black where the surface is seen edge on), in the colours of the palette
STYLE = {"face": {"fill": 0.25, "line": 0.9, "shade": 0.6}, "mandible": {"fill": 0.30, "line": 0.8, "shade": 0.6}}
COLORS = {"face": "#F5C542", "mandible": "#E6E6EC", "corners": "#58B36B", "reprojected": "#E6E6EC"}
BOARDS = ("patient", "sheet")
TRACKING = {
    # temporal filter of the pose (pose_filter.py)
    "smooth": True,           # False: every frame at its own pose as it comes (the tests of the geometry)
    "min_cutoff_hz": 0.5,     # smoothing while still: lower is calmer and slower to settle
    "beta": 0.04,             # cutoff added per px/s of the model's speed in the image: higher lags less in motion
    "d_cutoff_hz": 3.0,       # smoothing of that speed
    "lead": True,             # the filter moves the pose on by its speed first: no lag at a steady speed
    "still_px_s": 60.0,       # slower than this the pose is not moved on: it is noise, not motion
    "predict_s": 0.1,         # the pose is moved on at most this far past the last one found
    "gate_px": 40.0,          # a pose moving the model more than this (px, stream size) from the prediction is a jump:
    "strong_corners": 0.5,    #   believed with at least this share of the board's corners
    "strong_rms": 0.5,        #   and an RMS under this share of the limit, otherwise only if the next pose agrees
    # loss of the board
    "hold_s": 0.7,            # the last pose stays drawn this long
    "fade_s": 0.35,           # then fades out over this time
    "fade_in_s": 0.12,        # and fades in when the board is found again,
    "acquire_frames": 2,      #   after this many poses in a row
    # speed
    "workers": 2,             # threads finding the board, on consecutive frames
    "wait_ms": 100.0,         # a frame waits this long at most for its own pose, then is drawn at the predicted one
    "roi_margin": 0.35,       # while tracked, the board is searched in its last box grown by this share of its size
    "roi_px": 48,             #   and by this many px on each side,
    "roi_s": 0.3,             #   for this long after it was last found
    "marker_px": 32.0,        # the box is scaled so that the markers are about this wide (never enlarged)
    "min_scale": 0.4,
}
_GL_THREAD = None                       # the thread that owns gl_overlay's OpenGL context
_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)


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


# ── the board in one frame ───────────────────────────────────────────────────
def solve_pose(obj, img, K, dist, prior=None, ambiguous=2.0):
    """Board pose by PnP as marker_pose.estimate_pose: IPPE (planar), then Levenberg-Marquardt. When the second IPPE
    solution fits less than `ambiguous` times worse than the first (a small board, or one seen face on, can look
    mirrored), the one with the rotation nearer the prior is taken."""
    _, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    errs = np.ravel(errs)
    order = np.argsort(errs)
    best = int(order[0])
    if prior is not None and len(errs) > 1 and errs[order[1]] < ambiguous * max(errs[best], 1e-9):
        ang = [np.linalg.norm(pose_filter.rotvec(cv2.Rodrigues(rvecs[k])[0] @ prior[:3, :3].T)) for k in order[:2]]
        best = int(order[int(np.argmin(ang))])
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvecs[best].copy(), tvecs[best].copy())
    proj = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
    other = errs[order[1]] if len(errs) > 1 else np.inf
    return {"T": marker_pose.matrix(rvec, tvec), "rms_px": rms, "n": len(obj),
            "ambiguity": float(other / max(errs[order[0]], 1e-12))}


class BoardFinder:
    """ChArUco corners of one frame. Without a prior: marker_pose.detect on the whole frame (full resolution, every
    threshold window). With one (the board's box and marker size on a recent frame): the markers in the box only, on
    the box scaled so that they are about TRACKING["marker_px"] wide and with one threshold window, their corners
    refined at full resolution, then the chessboard corners interpolated and refined at full resolution by OpenCV's
    ChArUco detector, as in the full search."""

    def __init__(self, board, spec, settings):
        self.board, self.spec, self.s = board, spec, settings
        self.full = marker_pose.make_detector(board)
        cp = cv2.aruco.CharucoParameters()
        cp.minMarkers = 1
        cp.tryRefineMarkers = True
        dp = cv2.aruco.DetectorParameters()
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.charuco = cv2.aruco.CharucoDetector(board, cp, dp)
        self.obj = board.getChessboardCorners().astype(np.float64)
        nx, ny = spec["squares"]
        sq = spec["square_mm"]
        self.outline = np.array([[0, 0, 0], [nx * sq, 0, 0], [nx * sq, ny * sq, 0], [0, ny * sq, 0]], np.float64)
        self._markers = {}

    def _detector(self, win):
        if win not in self._markers:
            dp = cv2.aruco.DetectorParameters()
            dp.adaptiveThreshWinSizeMin = dp.adaptiveThreshWinSizeMax = win
            dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
            self._markers[win] = cv2.aruco.ArucoDetector(self.board.getDictionary(), dp)
        return self._markers[win]

    def find(self, gray, prior=None):
        """(corners N x 2 in px of the whole frame, ids, markers found, {"roi": box or None, "scale"})."""
        none = np.empty((0, 2)), np.empty(0, np.int32)
        if prior is None:
            corners, ids, n_markers = marker_pose.detect(self.full, gray)
            return corners, ids, n_markers, {"roi": None, "scale": 1.0}
        x0, y0, x1, y1 = prior["box"]
        crop = gray[y0:y1, x0:x1]
        scale = float(np.clip(self.s["marker_px"] / max(prior["marker_px"], 1.0), self.s["min_scale"], 1.0))
        how = {"roi": (x0, y0, x1, y1), "scale": round(scale, 3)}
        small = crop if scale > 0.95 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale > 0.95:
            scale = 1.0
        win = int(np.clip(round(0.4 * prior["marker_px"] * scale), 5, 23)) | 1
        mc, mids, _ = self._detector(win).detectMarkers(small)
        if mids is None or not len(mids):
            return *none, 0, how
        pts = np.concatenate([c.reshape(4, 2) for c in mc]).astype(np.float32)
        if scale < 1.0:
            pts = (pts + 0.5) / scale - 0.5
        hw = int(np.clip(round(0.08 * prior["marker_px"]), 2, 6))
        pts = cv2.cornerSubPix(crop, pts.reshape(-1, 1, 2), (hw, hw), (-1, -1), _SUBPIX).reshape(-1, 4, 2)
        mc = tuple(p.reshape(1, 4, 2) for p in pts)
        cc, cids, _, mids2 = self.charuco.detectBoard(crop, markerCorners=mc, markerIds=mids)
        n_markers = len(mids2) if mids2 is not None else len(mids)
        if cids is None or not len(cids):
            return *none, n_markers, how
        return (cc.reshape(-1, 2).astype(np.float64) + (x0, y0), cids.ravel().astype(np.int32), int(n_markers), how)

    def prior_from(self, corners, ids, T, K, dist, size):
        """The box and marker size for the search of the next frames, from the corners (and the pose, which also puts
        the parts of the board not found in the box); None when they say too little."""
        if len(ids) < 4:
            return None
        hull_px = cv2.contourArea(cv2.convexHull(corners.astype(np.float32)))
        hull_mm = cv2.contourArea(cv2.convexHull(self.obj[ids, :2].astype(np.float32)))
        if hull_px <= 1 or hull_mm <= 1:
            return None
        marker_px = float(np.sqrt(hull_px / hull_mm) * self.spec["marker_mm"])
        pts = corners
        if T is not None and np.all(marker_pose.transform(T, self.outline)[:, 2] > 1.0):
            pts = np.vstack([corners, marker_pose.project(self.outline, T, K, dist)])
        lo, hi = pts.min(0), pts.max(0)
        grow = (hi - lo) * self.s["roi_margin"] + self.s["roi_px"]
        w, h = size
        x0, y0 = np.clip(np.floor(lo - grow), 0, (w, h)).astype(int)
        x1, y1 = np.clip(np.ceil(hi + grow), 0, (w, h)).astype(int)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        return {"box": (int(x0), int(y0), int(x1), int(y1)), "marker_px": marker_px}


# ── stand-ins for the phone ──────────────────────────────────────────────────
class _Frames:
    """What the stand-ins share with live.PhoneStream: the newest frame with its time and counter, the rate."""

    def _init_frames(self):
        self.error, self.restarts = None, 0
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._frame, self._t, self._n = None, 0.0, 0
        self._times = deque(maxlen=120)
        self._decode_ms = deque(maxlen=120)
        self._stop = threading.Event()

    def _put(self, img, decode_ms=None):
        t = time.perf_counter()
        with self._new:
            self._frame, self._t, self._n = img, t, self._n + 1
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
        src = [Path(p) for p in src]
        if len(src) == 1 and src[0].is_dir():
            # the digital twin keeps labels, masks and depth beside its images: they are not frames
            src = sorted(p for p in src[0].iterdir() if p.suffix.lower() in calibrate_charuco.IMAGE_SUFFIXES
                         and not p.stem.endswith(("_label", "_model", "_depth")))
        self._video = src[0] if len(src) == 1 and src[0].suffix.lower() in self.VIDEO else None
        self._images = [] if self._video else src
        self.fps = fps if self._video else min(fps, 10)       # images: at most 10 a second
        if not self._video and not self._images:
            raise LiveARError(f"nothing to play in {source}")
        self._times_rec = None
        if self._video is not None:
            stem = self._video.with_suffix("")
            tp = Path(str(stem) + "_times.npy")
            if tp.is_file():
                self._times_rec = np.load(tp)
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
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        self.error = f"cannot read {p.name}"
                        return
                    yield img, None, k

    def _run(self):
        rec = self._times_rec
        period = 1.0 / self.fps
        t_loop, nxt = time.perf_counter(), time.perf_counter()
        for img, dec, k in self._frames():
            if rec is not None and k < len(rec):
                if k == 0:
                    t_loop = time.perf_counter()
                time.sleep(max(0.0, t_loop + rec[k] - time.perf_counter()))
            n = self._put(img, dec)
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
    """Phone stream -> board pose -> filtered pose -> face overlay, on every frame of the stream.

    The threads live as long as the object: the finders (stage 1) and the compose thread (stage 2), which owns the
    OpenGL context and so is never replaced: stop() and start() only close and open the stream. Meant as one per
    process (the app keeps one); a later one makes its own OpenGL context."""

    def __init__(self, board="sheet", square_mm=None, registration=None, face=True, mandible=False, axes=True,
                 intrinsics=STREAM_INTRINSICS, synthetic=False, serial=None, size="1920x1440", fps=30,
                 display_width=DISPLAY_WIDTH, colors=None, log=None, replay=None, replay_truth=None, tracking=None,
                 script=None, exposure_s=0.0):
        if board not in BOARDS:
            raise LiveARError(f"unknown board {board}")
        self.intrinsics_path = Path(intrinsics) if intrinsics else None
        self.synthetic, self.serial, self.size, self.fps = synthetic, serial, size, fps
        self.replay, self.replay_truth = replay, replay_truth     # recorded frames in place of the phone
        self.script, self.exposure = script, exposure_s           # the synthetic stream's session and blur
        self.display_width = display_width
        self.colors = {k: bgr(v) for k, v in {**COLORS, **(colors or {})}.items()}
        self.log = log or (lambda msg: None)
        self.tracking = {**TRACKING, **(tracking or {})}
        self.trace = None             # a list: one row per frame searched and per frame shown (bench, tests)
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._found_cv = threading.Condition(self._lock)
        self._apply_lock = threading.Lock()
        self._want = {"board": board, "square_mm": float(square_mm or calibrate_charuco.BOARD["square"]),
                      "registration": str(registration) if registration else "", "face": bool(face),
                      "mandible": bool(mandible), "axes": bool(axes), "version": 1}
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
        self._thread = threading.Thread(target=self._work, daemon=True, name="live_ar_compose")
        self._thread.start()
        self._finders = []
        for k in range(max(1, int(self.tracking["workers"]))):
            th = threading.Thread(target=self._find_loop, daemon=True, name=f"live_ar_find_{k}")
            th.start()
            self._finders.append(th)

    def _reset_run(self):
        """What belongs to one run of the stream (under the lock, or before the threads start)."""
        self._pose = (0, None)            # (frame counter, T_C_M) of the last frame with a pose found
        self._masks = (0, {})             # (frame counter, {model: silhouette at the display size}) of that frame
        self._taken = 0                   # newest frame taken by a finder
        self._busy = set()                # frames being searched
        self._results = {}                # frame counter -> what the search found, until the compose thread uses it
        self._prior = None                # box, marker size and pose of the last frame where the board was found
        self._last_failed = False         # the last search found nothing: the next one looks at the whole frame
        self._latest = None               # the newest search result, for the status
        self._found_times = deque(maxlen=60)
        self._shown_times = deque(maxlen=60)
        self._tm_find = deque(maxlen=30)
        self._tm_show = deque(maxlen=30)
        self._served = deque(maxlen=60)
        self._tracker, self._tracker_key = None, None
        self._verdicts = deque(maxlen=90)

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
        """Works on a stream opened elsewhere (anything with latest, wait_frame, stats, alive, error, close)."""
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
            self._found_cv.notify_all()
        if stream is not None:
            stream.close()

    def shutdown(self):
        self.stop()
        self._quit.set()
        with self._lock:
            self._found_cv.notify_all()
        self._thread.join(timeout=2.0)
        for th in self._finders:
            th.join(timeout=2.0)

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

    def set_tracking(self, **kw):
        """Overrides of TRACKING (unknown keys refused); the tracker starts again from the next frame. The number of
        workers is fixed when the object is made."""
        bad = [k for k in kw if k not in TRACKING]
        if bad:
            raise LiveARError(f"unknown tracking settings {bad}")
        with self._lock:
            self.tracking = {**self.tracking, **kw}
            self._tracking_version += 1
            self._prior = None

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
        """The settings asked for: board, square_mm, registration, face, mandible, axes."""
        with self._lock:
            return {k: v for k, v in self._want.items() if k != "version"}

    def status(self):
        with self._lock:
            st = dict(self._status)
            stream = self._stream
            want = dict(self._want)
            served = list(self._served)
        st.update({"running": stream is not None, "starting": self._starting, "error": self._error,
                   "synthetic": self.synthetic or self.replay is not None, "replay": self.replay is not None,
                   "renderer": self.renderer,
                   "layers": {k: want[k] for k in ("face", "mandible", "axes")},
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
        if stream.error:
            st["stream_error"] = stream.error
        st["link"] = s.get("link")
        st["link_waiting"] = bool(s.get("waiting"))     # the link dropped: the stream waits for it to come back
        st["link_drops"] = s.get("link_drops", 0)
        if not stream.alive:
            st["error"] = stream.error or "stream stopped"
            st["cause"] = getattr(stream, "cause", "")
        return st

    def _size_wh(self):
        w, h = (int(x) for x in self.size.split("x"))
        return w, h

    # ── setup ────────────────────────────────────────────────────────────
    def _apply(self, want):
        """Board, finder and registration for the wanted settings (worker thread, or any thread while stopped)."""
        with self._apply_lock:
            if want["version"] == self._applied and self._setup is not None:
                return
            old = self._setup
            board, spec = make_board(want["board"], want["square_mm"])
            if old is not None and spec == old["spec"]:
                board, finder = old["board"], old["finder"]
            else:
                finder = BoardFinder(board, spec, self.tracking)
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
            # one assignment, so a worker never sees the board of one setting with the finder of another
            self._setup = {"board": board, "spec": spec, "finder": finder, "T_M_B": T, "registration": reg,
                           "version": want["version"]}
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

    # ── stage 1: the board in the newest frame ───────────────────────────
    def _find_loop(self):
        """A finder thread: the newest frame no other finder took, its corners and the board pose. OpenCV leaves the
        GIL while it works, so the finders run alongside each other and alongside the compose thread."""
        while not self._quit.is_set():
            stream = self._stream
            if stream is None:
                time.sleep(0.02)
                continue
            frame, t, n = stream.wait_frame(self._taken, timeout=0.1)
            if frame is None:
                continue
            with self._lock:
                if self._stream is not stream or n <= self._taken:
                    continue
                self._taken = n
                self._busy.add(n)
                want = dict(self._want)
                prior = self._prior
                if (prior is not None and (self._last_failed or t - prior["t"] > self.tracking["roi_s"])) \
                        or not self.tracking.get("smooth", True):
                    prior = None
            try:
                if want["version"] != self._applied:
                    self._apply(want)
                res = self._find(frame, t, n, self._setup, prior)
            except Exception as e:  # noqa: BLE001  a bad frame must not end the thread
                self._failed(stream, e)
                res = {"n": n, "t": t, "found": False, "corners": 0, "markers": 0, "rms_px": None, "T": None,
                       "tm": {}, "error": str(e)}
            with self._lock:
                self._busy.discard(n)
                if self._stream is not stream:
                    continue
                self._results[n] = res
                for k in [k for k in self._results if k < n - 60]:
                    self._results.pop(k)
                self._found_times.append(time.perf_counter())
                if res["tm"]:
                    self._tm_find.append(res["tm"])
                if self._latest is None or n > self._latest["n"]:
                    self._latest = res
                    self._last_failed = res.get("next") is None
                if res.get("next") is not None and (self._prior is None or n > self._prior["n"]):
                    self._prior = {**res["next"], "t": t, "n": n, "T": res["T"]}
                if res["found"] and n > self._pose[0]:
                    self._pose = (n, res["T"])
                if self.trace is not None:
                    self.trace.append({"what": "found", **{k: v for k, v in res.items() if k != "next"},
                                       "t_done": time.perf_counter()})
                self._found_cv.notify_all()

    def _find(self, frame, t, n, setup, prior):
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        K, dist, cam = self._intrinsics(w, h)
        limit = MAX_RMS[cam["source"]]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        finder = setup["finder"]
        finder.s = self.tracking
        corners, ids, n_markers, how = finder.find(gray, prior)
        t1 = time.perf_counter()
        pose = None
        if len(ids) >= MIN_CORNERS:
            pose = solve_pose(finder.obj[ids], corners, K, dist, None if prior is None else prior.get("T"))
        t2 = time.perf_counter()
        found = pose is not None and pose["rms_px"] <= limit
        s = self.tracking
        strong = found and len(ids) >= s["strong_corners"] * setup["spec"]["corners"] and pose["rms_px"] <= \
            s["strong_rms"] * limit
        nxt = finder.prior_from(corners, ids, pose["T"] if found else None, K, dist, (w, h)) if len(ids) >= 4 else None
        return {"n": n, "t": t, "found": bool(found), "strong": bool(strong), "T": pose["T"] if found else None,
                "rms_px": None if pose is None else pose["rms_px"], "corners": int(len(ids)), "markers": int(n_markers),
                "limit": limit, "roi": how["roi"], "scale": how["scale"], "next": nxt,
                "tm": {"detect_ms": (t1 - t0) * 1e3, "pose_ms": (t2 - t1) * 1e3}}

    # ── stage 2: every frame, drawn ──────────────────────────────────────
    def _work(self):
        """The compose thread, which owns the OpenGL context: every frame of the stream in order."""
        pending, cur, seen = deque(), None, 0
        while not self._quit.is_set():
            stream = self._stream
            if stream is None:
                pending.clear()
                cur = None
                time.sleep(0.02)
                continue
            if stream is not cur:
                cur, seen = stream, 0
                pending.clear()
            # without frames waiting, sleep until the next one; with some, only look whether one came
            frame, t, n = stream.wait_frame(seen, timeout=0.0 if pending else 0.05)
            if frame is not None:
                seen = n
                pending.append((n, t, frame))
            if not pending:
                continue
            k = self._ready(pending)
            if k is None:       # the oldest waits for its pose: a finder wakes this thread when one is found
                with self._found_cv:
                    self._found_cv.wait(timeout=0.003)
                continue
            for _ in range(k):
                pending.popleft()
            n, t, frame = pending.popleft()
            try:
                self._compose(stream, frame, t, n)
            except Exception as e:  # noqa: BLE001
                self._failed(stream, e)

    def _ready(self, pending):
        """Index in `pending` of the frame to draw now, None to wait: the oldest once its pose is in (or will not come,
        or it waited wait_ms); the newest ready one when the thread fell behind."""
        now = time.perf_counter()
        wait = self.tracking["wait_ms"] / 1e3
        with self._lock:
            ready = [n in self._results or (n <= self._taken and n not in self._busy) or now - t > wait
                     for n, t, _ in pending]
        if len(pending) >= 3 and any(ready):
            return max(i for i, r in enumerate(ready) if r)
        return 0 if ready[0] else None

    def _tracker_for(self, setup, K):
        """The tracker of this run, made again when the board, the registration or the settings change."""
        key = (setup["version"], self._tracking_version, round(float(K[0, 0]), 3))
        if self._tracker is None or self._tracker_key != key:
            T_M_B = setup["T_M_B"]
            if T_M_B is not None and "face" in self._models:
                v = self._models["face"][0] @ T_M_B[:3, :3].T + T_M_B[:3, 3]
            else:
                nx, ny = setup["spec"]["squares"]
                sq = setup["spec"]["square_mm"]
                v = np.array([[0, 0, 0], [nx * sq, ny * sq, 0]], np.float64)
            anchor = v.mean(0)
            radius = float(np.sqrt(np.mean(np.sum((v - anchor) ** 2, axis=1))))
            self._tracker = pose_filter.PoseTracker(self.tracking, anchor, radius, float(K[0, 0]))
            self._tracker_key = key
        return self._tracker

    def _feed(self, n, tracker):
        """Gives the tracker the poses found on the frames up to n, in order; returns the result of frame n, if any."""
        with self._lock:
            ks = sorted(k for k in self._results if k <= n)
            rs = [self._results.pop(k) for k in ks]
        own = None
        for r in rs:
            r["verdict"] = tracker.measure(r["t"], r["T"] if r["found"] else None, r.get("strong", False))
            self._verdicts.append(r["verdict"])
            if r["n"] == n:
                own = r
        return own

    def _compose(self, stream, frame, t_dec, n):
        tc = time.perf_counter()
        setup = self._setup
        with self._lock:
            want = dict(self._want)
        board, spec, T_M_B = setup["board"], setup["spec"], setup["T_M_B"]
        h, w = frame.shape[:2]
        K, dist, cam = self._intrinsics(w, h)
        tracker = self._tracker_for(setup, K)
        own = self._feed(n, tracker)
        T, vis, state = tracker.pose_at(t_dec)
        if T is None:
            kind = "none"
        elif own is not None and own["verdict"] in ("accepted", "restarted"):
            kind = "measured"
        elif own is None and state == "tracking":
            kind = "predicted"          # its own pose was not ready in time
        else:
            kind = "fading" if state == "fading" else "holding"     # the board was not found on it
        t1 = time.perf_counter()
        dw = min(self.display_width, w)
        s = dw / w
        dh = int(round(h * s))
        img = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR) if dw != w else frame.copy()
        Kd = scale_K(K, s, s)
        t2 = time.perf_counter()
        drawn, masks = {}, {}
        if T is not None and T_M_B is not None:
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
                    self._paint(img, r, self.colors[name], st, vis, shade)
        t3 = time.perf_counter()
        if T is not None and want["axes"] and vis > 0.5:
            rvec = cv2.Rodrigues(T[:3, :3])[0]
            cv2.drawFrameAxes(img, Kd, dist, rvec, T[:3, 3], 3 * spec["square_mm"], 2)
            proj = marker_pose.project(setup["finder"].obj, T, Kd, dist)
            for p in proj:
                cv2.circle(img, (int(round(p[0] * 16)), int(round(p[1] * 16))), 2 * 16, self.colors["reprojected"], -1,
                           cv2.LINE_AA, 4)
        t4 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        t5 = time.perf_counter()
        tm_show = {"wait_ms": (tc - t_dec) * 1e3, "resize_ms": (t2 - t1) * 1e3, "render_ms": (t3 - t2) * 1e3,
                   "axes_ms": (t4 - t3) * 1e3, "jpeg_ms": (t5 - t4) * 1e3, "compose_ms": (t5 - tc) * 1e3}
        with self._lock:
            if self._stream is not stream:
                return
            self._tm_show.append(tm_show)
            self._shown_times.append(t5)
            latest = self._latest
            st = self._status_of(tracker, state, kind, vis, T, latest, cam, spec, drawn, w, h, dw, dh, t5 - t_dec)
            self._status = st
            if kind == "measured":
                self._masks = (n, masks)
            if ok:
                self._jpeg, self._jpeg_n, self._bgr, self._jpeg_t = buf.tobytes(), self._jpeg_n + 1, img, t5
            if self.trace is not None:
                self.trace.append({"what": "shown", "n": n, "t": t_dec, "t_done": t5, "kind": kind, "state": state,
                                   "vis": vis, "T": T, "verdict": None if own is None else own["verdict"],
                                   "drawn": drawn, "tm": tm_show})
            self._frame_ready.notify_all()

    def _status_of(self, tracker, state, kind, vis, T, latest, cam, spec, drawn, w, h, dw, dh, processing):
        """The status after a frame was drawn (under the lock)."""
        def mean(rows, key):
            vals = [r[key] for r in rows if key in r]
            return round(float(np.mean(vals)), 1) if vals else 0.0
        tm = {k: mean(self._tm_find, k) for k in ("detect_ms", "pose_ms")}
        tm.update({k: mean(self._tm_show, k) for k in ("wait_ms", "resize_ms", "render_ms", "axes_ms", "jpeg_ms",
                                                         "compose_ms")})
        tm["draw_ms"] = round(tm["resize_ms"] + tm["axes_ms"] + tm["jpeg_ms"], 1)
        tm["total_ms"] = round(tm["detect_ms"] + tm["pose_ms"] + tm["compose_ms"], 1)

        def rate(times):
            return round((len(times) - 1) / (times[-1] - times[0]), 1) if len(times) > 1 and times[-1] > times[0] else 0.0
        v = list(self._verdicts)
        st = {"tracked": tracker.tracked, "state": state, "kind": kind, "opacity": round(float(vis), 2),
              "corners": 0 if latest is None else latest["corners"], "markers": 0 if latest is None else latest["markers"],
              "rms_px": None if latest is None or latest["rms_px"] is None else round(latest["rms_px"], 3),
              "rms_limit_px": MAX_RMS[cam["source"]], "min_corners": MIN_CORNERS, "intrinsics": cam,
              "drawn_px": drawn, "timings": tm, "image": [w, h], "display": [dw, dh],
              "fps_ar": rate(self._shown_times), "fps_pose": rate(self._found_times),
              "missed_share": round(sum(x == "missed" for x in v) / len(v), 3) if v else 0.0,
              "rejected": tracker.counts["rejected"]}
        if T is not None:
            nx, ny = spec["squares"]
            centre = np.array([nx, ny, 0.0]) * [spec["square_mm"], spec["square_mm"], 0] / 2
            st["distance_mm"] = round(float(np.linalg.norm(marker_pose.transform(T, centre[None])[0])), 1)
            st["tilt_deg"] = round(float(np.degrees(np.arccos(abs(T[2, 2])))), 1)
        stream_ms = 0.0 if self.synthetic or self.replay is not None else STREAM_LATENCY_MS.get(self._link)
        st["latency"] = {"stream_ms": stream_ms, "processing_ms": round(processing * 1e3, 1),
                         "total_ms": round((stream_ms or 0.0) + processing * 1e3, 1), "link": self._link}
        return st

    def _failed(self, stream, e):
        if self._stream is stream:
            self._error = f"{type(e).__name__}: {e}"
            self.log(self._error)
        time.sleep(0.05)

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
    def _paint(img, r, colour, style, vis=1.0, shade=None):
        """The visible model as a soft tint over the image and a thin smooth outline, both at opacity `vis`."""
        x0, y0 = r.offset
        bh, bw = r.faces.shape
        if not (bh and bw) or vis <= 0:
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
        a = style["fill"] * vis
        cv2.copyTo(cv2.addWeighted(roi, 1 - a, tint, a, 0), m, roi)
        outline, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        lines = []
        for c in outline:
            c = c.reshape(-1, 2).astype(np.float32)
            if len(c) >= 7:      # the pixel staircase of the mask, smoothed along the contour
                c = sum(np.roll(c, k, 0) for k in (-2, -1, 0, 1, 2)) / 5
            lines.append(np.round(c * 8).astype(np.int32))
        la = style["line"] * vis
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
    return setup["finder"].obj


def measure_run(X, rows, K, still=None, truth=None, t0=0.0):
    """Numbers of a run from its trace (rows "found" and "shown"): rates, time per stage, frames without a pose,
    blinks of the overlay, jitter of the overlay while still (of the points X in M: px at the stream size and mm, as
    found and as drawn) and, with the truth (truth(n) -> T_C_M), the error of the overlay in the image. still: (from,
    to) in s from t0."""
    found = sorted((r for r in rows if r["what"] == "found"), key=lambda r: r["n"])
    shown = sorted((r for r in rows if r["what"] == "shown"), key=lambda r: r["n"])
    out = {}

    def mean_tm(rr, key):
        v = [r["tm"][key] for r in rr if r.get("tm") and key in r["tm"]]
        return float(np.mean(v)) if v else None
    if len(shown) > 1:
        ts = np.array([r["t_done"] for r in shown])
        out["fps_video"] = (len(ts) - 1) / (ts[-1] - ts[0])
        out["latency_ms"] = float(np.median([(r["t_done"] - r["t"]) * 1e3 for r in shown]))
        for k in ("wait_ms", "resize_ms", "render_ms", "axes_ms", "jpeg_ms", "compose_ms"):
            out[k] = mean_tm(shown, k)
        vis = np.array([r["vis"] > 0 for r in shown])
        out["blinks"] = int(np.sum(vis[:-1] & ~vis[1:]))
        out["kinds"] = {k: int(sum(r["kind"] == k for r in shown)) for k in ("measured", "predicted", "holding",
                                                                           "fading", "none")}
    if len(found) > 1:
        tf = np.array([r["t_done"] for r in found])
        out["fps_pose"] = (len(tf) - 1) / (tf[-1] - tf[0])
        out["detect_ms"] = mean_tm(found, "detect_ms")
        out["pose_ms"] = mean_tm(found, "pose_ms")
        out["no_pose_share"] = float(np.mean([not r["found"] for r in found]))
        out["searched_share"] = len(found) / max(len(shown), 1)

    def jitter(rows_):
        P = np.array([marker_pose.project(X, r["T"], K) for r in rows_])
        Q = np.array([marker_pose.transform(r["T"], X) for r in rows_])
        px = float(np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, axis=2))))
        mm = float(np.sqrt(np.mean(np.sum((Q - Q.mean(0)) ** 2, axis=2))))
        return px, mm
    if still is not None:
        a, b = still
        f_still = [r for r in found if r["found"] and a <= r["t"] - t0 < b]
        s_still = [r for r in shown if r["T"] is not None and a <= r["t"] - t0 < b]
        if len(f_still) > 5:
            out["jitter_found_px"], out["jitter_found_mm"] = jitter(f_still)
        if len(s_still) > 5:
            out["jitter_drawn_px"], out["jitter_drawn_mm"] = jitter(s_still)
    if truth is not None:
        def err(rows_):
            e = []
            for r in rows_:
                Tt = truth(r["n"])
                if Tt is not None and r["T"] is not None:
                    e.append(np.sqrt(np.mean(np.sum((marker_pose.project(X, r["T"], K)
                                                     - marker_pose.project(X, Tt, K)) ** 2, axis=1))))
            return np.array(e)
        for name, rr in (("found", [r for r in found if r["found"]]),
                         ("drawn", [r for r in shown if r["kind"] in ("measured", "predicted")]),
                         ("drawn_predicted", [r for r in shown if r["kind"] == "predicted"]),
                         ("drawn_held", [r for r in shown if r["kind"] in ("holding", "fading")])):
            e = err(rr)
            if len(e):
                out[f"error_{name}_px"] = (float(np.median(e)), float(np.percentile(e, 95)))
    return out


def run_bench(ar, seconds, still=None, warmup=1.0):
    """Runs the live AR for `seconds` with the trace on and an MJPEG client reading the frames; returns the numbers of
    measure_run and of the client (frames received per second)."""
    import urllib.request
    ar.trace = []
    ar.start()
    stream = ar.stream()
    t0 = time.perf_counter()
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
    rows = [r for r in ar.trace if r["t"] - t0 >= warmup]
    st = ar.status()
    ar.trace = None
    th.join(timeout=2.0)
    server.close()
    ar.stop()
    w, h = st["image"] if "image" in st else ar._size_wh()
    K = stream_intrinsics((w, h), ar.intrinsics_path)[0]
    truth = getattr(stream, "truth", None)
    out = measure_run(model_points(ar), rows, K, still, truth, t0)
    g = np.array([x for x in got if x - t0 >= warmup])
    out["fps_page"] = (len(g) - 1) / (g[-1] - g[0]) if len(g) > 1 else 0.0
    out["decode_ms"] = st.get("timings", {}).get("decode_ms")
    out["serve_ms"] = st.get("timings", {}).get("serve_ms")
    out["fps_stream"] = st.get("fps_stream")
    return out


def print_bench(out, title):
    def f(k, fmt="{:.1f}"):
        v = out.get(k)
        if v is None:
            return "-"
        if isinstance(v, tuple):
            return " / ".join(fmt.format(x) for x in v)
        return fmt.format(v)
    print(f"\n{title}")
    print(f"  rate: stream {f('fps_stream')} fps, video {f('fps_video')} fps, received by the page {f('fps_page')} fps, "
          f"poses {f('fps_pose')} fps ({f('searched_share', '{:.0%}')} of the frames searched)")
    print(f"  per frame: decode {f('decode_ms')} ms, corners {f('detect_ms')} ms, PnP {f('pose_ms', '{:.2f}')} ms, "
          f"wait for the pose {f('wait_ms')} ms, resize {f('resize_ms')} ms, models drawn {f('render_ms')} ms, "
          f"axes {f('axes_ms')} ms, JPEG {f('jpeg_ms')} ms, to the page {f('serve_ms', '{:.2f}')} ms")
    print(f"  decoded frame to JPEG: median {f('latency_ms')} ms")
    print(f"  frames searched without a pose: {f('no_pose_share', '{:.1%}')}; overlay blinks: {f('blinks', '{:d}')}; "
          f"frames drawn: {out.get('kinds')}")
    print(f"  jitter while still: as found {f('jitter_found_px', '{:.3f}')} px {f('jitter_found_mm', '{:.3f}')} mm, "
          f"as drawn {f('jitter_drawn_px', '{:.3f}')} px {f('jitter_drawn_mm', '{:.3f}')} mm")
    if "error_found_px" in out:
        print(f"  overlay against the truth (median / 95th percentile, px): as found {f('error_found_px', '{:.2f}')}; "
              f"as drawn while tracked {f('error_drawn_px', '{:.2f}')}, of which at a predicted pose "
              f"{f('error_drawn_predicted_px', '{:.2f}')}; held while the board is lost "
              f"{f('error_drawn_held_px', '{:.2f}')}")


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
    board, spec = make_board(a.board, a.square)
    replay = a.replay
    ar = LiveAR(a.board, a.square, a.registration, mandible=a.mandible, synthetic=replay is None, replay=replay,
                intrinsics=a.intrinsics or STREAM_INTRINSICS, script=bench_script(spec), exposure_s=0.008, log=print)
    if replay is not None:
        ar.fps = 30
    try:
        still = (1.0, a.still) if a.still else ((1.0, 6.0) if replay is None else None)
        out = run_bench(ar, a.seconds, still, warmup=0.0)
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
        ar.set_tracking(smooth=False, acquire_frames=1, fade_in_s=0.0)
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


def finder_check(expect):
    """The fast search (box, scaled image, one threshold window) against the full search on frames of the synthetic
    stream, for both boards: as many corners, as close to their true positions (the two refine the corners from
    different starting points, so single corners differ by up to a few tenths of a pixel either way)."""
    for name in BOARDS:
        board, spec = make_board(name, 24.0)
        K, dist, _ = stream_intrinsics((1920, 1440))
        s = SyntheticStream(board, spec, K, exposure_s=0.0)
        s.close()
        finder = BoardFinder(board, spec, TRACKING)
        err_full, err_fast, counts, t_full, t_fast, scales = [], [], [], [], [], []
        for k in range(12):
            img, T = s.render(0.7 * k, k)
            true = marker_pose.project(finder.obj, T, K)
            t = time.perf_counter()
            c0, i0, _ = marker_pose.detect(finder.full, img)
            t_full.append(time.perf_counter() - t)
            prior = finder.prior_from(c0, i0, T, K, dist, (1920, 1440))
            t = time.perf_counter()
            c1, i1, _, how = finder.find(img, prior)
            t_fast.append(time.perf_counter() - t)
            scales.append(how["scale"])
            counts.append((len(i0), len(i1)))
            err_full.append(np.linalg.norm(c0 - true[i0], axis=1))
            err_fast.append(np.linalg.norm(c1 - true[i1], axis=1))
        counts = np.array(counts)
        rms = [float(np.sqrt(np.mean(np.concatenate(e) ** 2))) for e in (err_full, err_fast)]
        ok = np.all(counts[:, 1] >= counts[:, 0] - 1) and rms[1] <= 1.05 * rms[0] + 0.01
        expect(ok, f"{name}: fast search against the full one on {len(counts)} frames: corners {counts[:, 1].sum()} of "
                   f"{counts[:, 0].sum()}, from the truth {rms[1]:.3f} px RMS against {rms[0]:.3f}; "
                   f"{1e3 * np.median(t_fast):.1f} ms against {1e3 * np.median(t_full):.1f} ms (scale "
                   f"{min(scales):.2f}-{max(scales):.2f})")


def run_test():
    """Pose filter; fast search against the full one; synthetic stream through the whole pipeline: pose against the
    truth, overlay drawn, light, video at the stream rate, file registration, no board, a hand over the board, MJPEG;
    then the frames of the digital twin with the true registration."""
    import tempfile
    import urllib.request
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    if pose_filter.run_test(TRACKING):
        fails.append("pose filter")
    finder_check(expect)

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
            expect(st.get("rms_px") is not None and st["rms_px"] < 0.5 and st["corners"] >= MIN_CORNERS,
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
    out = measure_run(model_points(ar), rows, K, (1.0, 6.0), stream.truth, t0)
    shown = sorted((r for r in rows if r["what"] == "shown"), key=lambda r: r["n"])
    rel = np.array([r["t"] - t0 for r in shown])
    vis = np.array([r["vis"] for r in shown])
    short = vis[(rel >= 4.0) & (rel < 4.4)]
    long_ = vis[(rel >= 14.0 + TRACKING["hold_s"] + TRACKING["fade_s"] + 0.1) & (rel < 15.5)]
    clear = vis[((rel >= 1.0) & (rel < 1.9)) | ((rel >= 4.8) & (rel < 7.9)) | ((rel >= 9.0) & (rel < 13.9))]
    expect(len(short) and short.min() == 1.0 and len(long_) and long_.max() == 0.0 and clear.min() == 1.0
           and out["blinks"] == 1,
           f"hand over the board: face held at full opacity over the 0.4 s cover, gone in the 1.5 s one, "
           f"{out['blinks']} blink in 18 s (the long cover), always drawn with the board in view")
    e = out.get("error_drawn_px", (np.inf, np.inf))
    expect(st.get("fps_ar", 0) >= 0.75 * st["fps_stream"] and e[1] < 6.0
           and out.get("jitter_drawn_px", 9) < out.get("jitter_found_px", 0) + 1e-9,
           f"session: video {out.get('fps_video', 0):.1f} fps on a {st['fps_stream']} fps stream, poses "
           f"{out.get('fps_pose', 0):.1f} fps; overlay against the truth median / 95th percentile "
           f"{e[0]:.2f} / {e[1]:.2f} px drawn, {out.get('error_found_px', (0, 0))[0]:.2f} / "
           f"{out.get('error_found_px', (0, 0))[1]:.2f} px as found; jitter while still "
           f"{out.get('jitter_found_px', 0):.3f} px as found, {out.get('jitter_drawn_px', 0):.3f} px drawn")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["view", "test", "bench", "clip"])
    ap.add_argument("out", nargs="?", help="clip: the file to write (OUT.h264, OUT_times.npy, OUT_truth.npz)")
    ap.add_argument("--board", choices=BOARDS, default="sheet")
    ap.add_argument("--square", type=float, default=None, help="measured square of the A4 sheet, mm")
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
