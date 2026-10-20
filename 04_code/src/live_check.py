"""The registration watched over in the live view: is the face still where the registration draws it, is the board
whole; on request the registration corrected from views of the face, with an arrow that says where to move the phone
for a more useful view and when the views are enough.

Two processes, so that the video never waits. The live view's (live_ar.LiveAR, through the hooks frame(), draw() and
status() of RegistrationCheck) only hands frames over and draws what comes back:
  board     every frame with its own board pose: the reprojection error of the corners with the nominal board (RMS)
            and per quadrant, over the last BOARD["window_s"] (BoardCheck). The board is damaged (bent, scratched)
            when the RMS stays above its limit or one quadrant stays more than BOARD["ratio"] times off the others:
            a fault of its own, said apart from a lost registration, and no correction while it lasts.
  frames    at most DETECT["hz"] a second, one in flight, the newest frame tracked: the square around the face (the
            REGION_MM the network looks at, or the face with a margin) cut from the frame at full resolution, with
            its intrinsics, board pose and the registration in use, to the worker.
The worker (python live_check.py worker, started by RegistrationCheck, below normal priority) loads the models once
and, for each frame:
  1. the face at the registration in use (gl_overlay.py, the decimated phantom of the bench scene, frame B): the box
     of its visible part (without the part under the surgical cap) with a DETECT["box_margin"] margin;
  2. the observed face: SAM 2.1 tiny prompted with that box (sam2_segment.py), less the known occluders (marker,
     holder and spheres moving with the board; splint and the parts of the interface on the head; the cap), which,
     dilated, are the pixels where outline points do not count;
  3. the misalignment score of the view (score.view_score, mm on the face) and the state: the median of the last
     DETECT["median_of"] scores above TAU_MM is a lost registration (back to aligned under DETECT["back_share"] of
     it). TAU_MM is chosen on the validation sessions of the digital twin so that a lost registration (mean target
     error above 2 mm) is found in at least 95 % of the cases (python live_check.py tau);
  4. the frame kept as a key frame when the camera looks at the face from at least KEY["deg"] away from every key
     frame kept (at most KEY["max"]); on the change to lost, the key frames from before it go;
  5. while lost: the expected information of each key frame (view_guidance.fisher, from the render alone), the best
     KEY["views"] of them (greedy) and the predicted 95 % target error after them; "enough" once it is under
     ENOUGH_MM (the rule that says how many views are enough); until then the arrow: the camera moved by GUIDE["deg"]
     around the face in 8 directions, each candidate rendered, the one that lowers the predicted error most (by at
     least GUIDE["gain"] of it), drawn as the way to move the phone.
On request (correct(); with settings auto, by itself once the views are enough), the correction from the best key
frames, with the method of settings method:
  net       the network of refiner.py (render and compare, ITERS iterations) is the estimate; the optimisation of
            correct.py from its output is the check: applied when the score at the estimate is at most TAU_MM, its
            predicted 95 % error (the network's variance, scaled on validation) under U_MM and the optimisation
            lands within CORRECT["agree_mm"] of it (mean distance at the targets)
  opt       the optimisation alone, multi-start (correct.correct), applied when the score at its result is at most
            TAU_MM and its predicted 95 % error (Laplace) under U_MM; with 6 parameters at least two views
  net_opt   the network, then the optimisation from its output, applied as opt
mode "a1" corrects the rotation about the ball of interface A (3 parameters), "rigid" any rigid motion (6). A
correction not applied leaves the registration as it was (flagged by the state); one applied replaces it in the live
view (LiveAR.set_corrected) until undo() or a new registration. Times of every stage are kept and reported.

Frames: C camera (OpenCV), M board, B head (coordinates of the bench scene, mm), as live_ar.py and correct.py.

    check = RegistrationCheck(settings={"method": "net", "mode": "a1", "auto": False})
    ar.attach_check(check)          # the live view hands its frames over and draws the state
    check.status()                  # state, score, views, arrow, last correction, board, times
    check.correct()                 # the correction now; check.undo() goes back
    check.close()

    python live_check.py worker ADDRESS             the worker (started by RegistrationCheck)
    python live_check.py tau [--split val]          the threshold from the twin's video sweeps (validation)
    python live_check.py twin [--split test] [--out CSV]   detection and correction on the twin's sweeps: every
                                                    method on the same views, against the true registration
    python live_check.py bench [--seconds S]        the synthetic stream with a face through the live view: video
                                                    rate with and without the check, detection rate, correction
                                                    times and error; the recorded clips (live_tune): video rate
                                                    and the board check
    python live_check.py test                       checks on the synthetic stream with a face (exit 1 on failure)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import arrowhead            # noqa: E402
import correct as co        # noqa: E402
import score as sc          # noqa: E402
import view_guidance as vg  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = SRC.parents[1]
WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
MESHES = WORK / "synth" / "meshes"
FACE_MESH = MESHES / "face_phantom_r20k.npz"           # the decimated phantom face (frame B), as live_ar.FACE_MESH
FACE_FULL, FACE_WEIGHTS = MESHES / "face_phantom_full.npz", MESHES / "face_weights.npz"   # the part under the cap
OCCLUDERS = MESHES / "occluders.npz"                   # marker, interfaces, splint (frame B of the bench design)
PROBE = WORK / "synth" / "probe.json"                  # the ball centre of interface A (frame B)
TARGETS_CSV = ROOT / "03_data" / "processed" / "targets" / "mandible_targets.csv"
NET = WORK / "models" / "refiner" / "refiner.pt"
SIGMA_SCALE = WORK / "refiner" / "sigma_scale.json"     # the network's standard deviations scaled on validation
TWIN = WORK / "synth" / "sessions"
SAM_MASKS = WORK / "synth_sam"
RESULTS = ROOT / "05_results"
CLIPS = WORK / "live_tune"

METHODS = ("net", "opt", "net_opt")
MODES = ("a1", "rigid")
SEGMENTERS = ("sam", "colour")
SETTINGS = {
    "method": "net",       # METHODS: the network as the estimate with the optimisation as its check (see above)
    "mode": "rigid",       # MODES: "a1" for interface A, "rigid" for B or any
    "auto": False,         # correct by itself once the views are enough
    "segmenter": "sam",    # "colour": the face by its colour on a grey scene (the synthetic stream of the tests)
    "device": None,        # torch device of SAM and the network (None: cuda when there is one)
    "boards": ("patient",),  # the face is watched over on these boards (the calibration board has no face)
}
TAU_MM = 0.67          # misalignment score above which the registration is lost: chosen on the twin's validation
                       # sweeps (python live_check.py tau) to find 95 % of the lost ones (at trim 0.2: AUC 0.94,
                       # 73 % of the others left alone, 8 % false alarms before an impact without a drape)
U_MM = 2.0             # a correction is applied only with a predicted 95 % target error under this
ENOUGH_MM = 1.0        # views are enough once the predicted 95 % target error after them is under this
DETECT = {
    "hz": 3.0,             # frames to the worker at most this often (one in flight)
    "median_of": 3,        # the state from the median of the last scores
    "back_share": 0.9,     # lost: aligned again under this share of TAU_MM
    "min_face_px": 1500,   # the visible face in view must cover this many pixels
    "box_margin": 0.10,    # prompt box of SAM: the visible face grown by this share (half on each side), as the
                           #   network was trained (refiner.BOX_MARGIN); a larger box takes in the background
    "dilate_px": 3,        # the known occluders grown by this before they are left out
    "trim": 0.2,           # score.view_score's trimmed mean: a drape or a hand moves a stretch of the outline
    "in_flight": 2,        # frames sent and not answered yet (the worker's SAM overlaps the rest of the previous)
    "region_mm": 288.0,    # the square sent around the face centre (the network's region, refiner.REGION_MM)
    "crop_margin": 0.15,   #   or the face with this margin, whichever is larger
    "stale_s": 3.0,        # a result older than this is not shown
}
KEY = {"deg": 5.0, "max": 16, "views": 4, "age_s": 300.0}
GUIDE = {"deg": 15.0, "gain": 0.15, "draws": 400, "stale_s": 1.5}
CORRECT = {"agree_mm": 1.5, "draws": 2000}
# the board check: over the last window_s, at least min_frames frames; damaged when the median RMS is above rms_share
# of the tracking limit, or the worst quadrant's mean residual is above ratio times the mean of the others and above
# floor_px (python live_check.py bench prints both on the clips)
BOARD = {"window_s": 2.0, "min_frames": 10, "rms_share": 0.6, "ratio": 2.5, "floor_px": 0.5}
STATES = ("off", "loading", "noface", "checking", "ok", "lost", "board", "correcting", "failed")
DRAW = {"badge_px": 13.0, "arrow_px": 7.0, "head_px": 30.0, "halo_px": 3.0}
COLOURS = {"ok": "#2F7A46", "lost": "#8F6410", "board": "#B0524A", "wait": "#9A9AA6", "mark": "#FFFFFF",
           "arrow": "#F2F2F5", "halo": "#16161A"}
START_S = 120.0        # the worker has this long to load
WORKER_THREADS = 2     # CPU threads of the worker's libraries (OpenCV, NumPy, torch): the rest is the live view's
RESTARTS = 3           # a worker that dies is started again this many times in a run


def bgr(colour):
    c = colour.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (4, 2, 0))


# ── the anatomy and its occluders ───────────────────────────────────────────
@dataclass
class Anatomy:
    v: np.ndarray            # decimated face (frame B, mm)
    f: np.ndarray
    visible: np.ndarray      # its triangles not under the surgical cap
    covered: np.ndarray      # and those under it
    o: np.ndarray            # centroid of the vertices: centre of the rigid update (as correct.py, refiner.py)
    c: np.ndarray            # ball centre of interface A (frame B)
    targets: np.ndarray      # the clinical targets of role "proposed" (frame B)
    occ: dict = field(default_factory=dict)    # (mode, "moving" | "fixed") -> (vertices, faces) in B of the design
    T_M_B_design: np.ndarray = None            # the registration of the design: where the moving parts sit on M


def load_targets(path=TARGETS_CSV):
    with open(path, encoding="utf-8") as fh:
        return np.array([[float(r[k]) for k in ("x_B", "y_B", "z_B")] for r in csv.DictReader(fh)
                         if r["role"] == "proposed"])


def load_anatomy(occluders=True):
    m = np.load(FACE_MESH)
    v, f = m["vertices"].astype(np.float64), m["faces"].astype(np.int64)
    covered = np.zeros(len(f), bool)
    if FACE_WEIGHTS.is_file() and FACE_FULL.is_file():
        from scipy.spatial import cKDTree
        cov_v = np.load(FACE_WEIGHTS)["covered"][cKDTree(np.load(FACE_FULL)["vertices"]).query(v)[1]]
        covered = cov_v[f].any(1)
    probe = json.loads(PROBE.read_text(encoding="utf-8")) if PROBE.is_file() else {}
    c = np.array(probe.get("ball_centre", v.mean(0)), np.float64)
    a = Anatomy(v, f, f[~covered], f[covered], v.mean(0), c, load_targets())
    if occluders and OCCLUDERS.is_file():
        z = np.load(OCCLUDERS)
        parts, moving = occluder_parts(z)
        common = [n for n in parts if n.startswith(("marker_", "gt_sphere_marker")) or n == "splint"]
        for mode, prefix in (("a1", "a1_"), ("rigid", "b1_")):
            sel_all = common + [n for n in parts if n.startswith(prefix)]
            for kind in ("moving", "fixed"):
                sel = [n for n in sel_all if (n in moving) == (kind == "moving")]
                vs, fs, k = [], [], 0
                for n in sel:
                    vs.append(parts[n][0])
                    fs.append(parts[n][1] + k)
                    k += len(vs[-1])
                if vs:
                    a.occ[(mode, kind)] = (np.vstack(vs), np.vstack(fs))
        a.T_M_B_design = np.asarray(z["T_M_B"], np.float64)
    return a


OCC_TRIANGLES = {"splint": 8000}    # the occluders are decimated for their masks: to this many, 3000 for the others
OCC_CACHE = WORK / "live_check"


def occluder_parts(z):
    """{name: (vertices, faces)} of the occluders (decimated, kept in OCC_CACHE for the file they come from) and the
    names of those that move with the board."""
    names = sorted({k[:-3] for k in z.files if k.endswith("__v")})
    moving = set(str(x) for x in z["moving"])
    st = OCCLUDERS.stat()
    path = OCC_CACHE / f"occluders_{st.st_size}_{int(st.st_mtime)}.npz"
    if path.is_file():
        c = np.load(path)
        return {n: (c[n + "__v"].astype(np.float64), c[n + "__f"].astype(np.int64)) for n in names}, moving
    out = {}
    for n in names:
        v, f = z[n + "__v"].astype(np.float64), z[n + "__f"].astype(np.int64)
        target = OCC_TRIANGLES.get(n, 3000)
        if len(f) > target:
            try:
                import pymeshlab
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f.astype(np.int32)))
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target, preservetopology=True,
                                                            autoclean=True)
                m = ms.current_mesh()
                v, f = m.vertex_matrix().astype(np.float64), m.face_matrix().astype(np.int64)
            except ImportError:
                pass
        out[n] = (v, f)
    try:
        OCC_CACHE.mkdir(parents=True, exist_ok=True)
        for old in OCC_CACHE.glob("occluders_*.npz"):
            old.unlink(missing_ok=True)
        np.savez_compressed(path, **{n + "__v": v for n, (v, f) in out.items()},
                            **{n + "__f": f for n, (v, f) in out.items()})
    except OSError:
        pass
    return out, moving


def mean_target_mm(T_a, T_b, X):
    """Mean distance (mm) at the targets X (frame B) between the registrations T_a and T_b (both M from B)."""
    E = np.linalg.inv(T_a) @ T_b
    return float(np.linalg.norm(X @ E[:3, :3].T + E[:3, 3] - X, axis=1).mean())


def project(K, dist, T, X):
    X = np.atleast_2d(np.asarray(X, np.float64))
    Xc = X @ T[:3, :3].T + T[:3, 3]
    return cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), K, np.zeros(5) if dist is None else dist)[0].reshape(-1, 2), Xc[:, 2]


def cut_box(size, T_C_M, K, dist, T_M_B, o, pts):
    """(x0, y0, x1, y1) of the square of a frame of size (w, h) around the face (DETECT["region_mm"] on its centre,
    or the face with DETECT["crop_margin"]), clipped to the frame; None when the face is not in front of the camera
    or not in the frame."""
    w, h = size
    uv, z = project(K, dist, T_C_M @ T_M_B, np.vstack([o, pts]))
    if z.min() <= 1 or not np.all(np.isfinite(uv)):
        return None
    c = uv[0]
    span = np.ptp(uv[1:], axis=0).max() * (1 + 2 * DETECT["crop_margin"])
    side = max(DETECT["region_mm"] * (K[0, 0] + K[1, 1]) / 2 / z[0], span)
    x0, y0 = int(max(0, np.floor(c[0] - side / 2))), int(max(0, np.floor(c[1] - side / 2)))
    x1, y1 = int(min(w, np.ceil(c[0] + side / 2))), int(min(h, np.ceil(c[1] + side / 2)))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def frame_message(img, K, dist, T_C_M, T_M_B, o, pts, n=0, t=None, board=None, mode=None):
    """The message of a frame for the worker (the bench and the tests build it as RegistrationCheck does), or None
    when the face is not in it."""
    h, w = img.shape[:2]
    box = cut_box((w, h), T_C_M, K, dist, T_M_B, o, pts)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    Kc = np.array(K, np.float64)
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0
    return {"type": "frame", "n": n, "t": time.perf_counter() if t is None else t,
            "img": np.ascontiguousarray(img[y0:y1, x0:x1]), "K": Kc, "dist": np.asarray(dist, np.float64),
            "T_C_M": T_C_M, "T_M_B": T_M_B, "K_full": np.asarray(K, np.float64), "T_C_M_full": T_C_M,
            "full": [w, h], "board": board or {"squares": [5, 5], "square_mm": 9.0}, "mode": mode}


def view_dir(T_C_M, T_M_B, o):
    """Unit direction from the face centre to the camera, in the board frame (it does not move with the registration
    much, and a key frame keeps it)."""
    T_M_C = np.linalg.inv(T_C_M)
    p = T_M_B[:3, :3] @ o + T_M_B[:3, 3]
    d = T_M_C[:3, 3] - p
    return d / max(np.linalg.norm(d), 1e-9)


# ── segmentation ────────────────────────────────────────────────────────────
class SamSegmenter:
    def __init__(self, device=None):
        import torch
        import sam2_segment as ss
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ss = ss
        self.pred = ss.load_predictor("tiny", self.device)

    def __call__(self, bgr_img, box):
        m, s = self.ss.segment(self.pred, cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), box, self.device)
        return np.where(m, 255, 0).astype(np.uint8), s


class ColourSegmenter:
    """The face of the synthetic stream: the coloured pixels (saturation above `sat`) in the box, the largest piece,
    holes filled. A stand-in for SAM in the tests (no GPU, no model)."""

    def __init__(self, sat=60):
        self.sat = sat
        self.device = "cpu"

    def __call__(self, bgr_img, box):
        h, w = bgr_img.shape[:2]
        x0, y0, x1, y1 = (int(round(b)) for b in box)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1 + 1, w), min(y1 + 1, h)
        out = np.zeros((h, w), np.uint8)
        if x1 <= x0 or y1 <= y0:
            return out, 0.0
        s = cv2.cvtColor(bgr_img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 1]
        m = (s > self.sat).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            return out, 0.0
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = np.where(lab == k, 255, 0).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(m, cs, -1, 255, -1)
        out[y0:y1, x0:x1] = m
        return out, 1.0


def make_segmenter(kind, device=None):
    if kind == "colour":
        return ColourSegmenter()
    return SamSegmenter(device)


# ── the worker process ──────────────────────────────────────────────────────
@dataclass
class Key:
    """A key frame: the cut of the frame with what the correction needs."""
    id: int
    t: float
    img: np.ndarray
    obs: np.ndarray          # observed face mask (SAM less the occluders)
    ignore: np.ndarray       # known occluders, dilated
    K: np.ndarray            # intrinsics of the cut
    dist: np.ndarray
    T_C_M: np.ndarray
    dir: np.ndarray          # view_dir
    score_mm: float

    def candidate(self):
        return vg.Candidate(f"k{self.id}", self.K, self.T_C_M, (self.img.shape[1], self.img.shape[0]), self.dist,
                            self.ignore)

    def view(self):
        return co.View(self.K, self.T_C_M, self.obs, self.dist, self.ignore)


class Worker:
    """The models and the state of the check (one per process; the methods are called in order by serve())."""

    def __init__(self, cfg=None, send=None):
        t0 = time.perf_counter()
        self.cfg = {**SETTINGS, **(cfg or {})}
        self.send = send or (lambda msg: None)
        self.a = load_anatomy()
        import gl_overlay
        a = self.a
        self.ren = gl_overlay.GLMeshRenderer(a.v, a.f)
        self.ren_vis = gl_overlay.GLMeshRenderer(a.v, a.visible)
        # per mode one scene for the occluders, depth tested against the face: the face (visible part, then the part
        # under the cap), the parts moving with the board (placed in B at each frame), the parts on the head
        self.scene = {}
        nv, nf = len(a.v), len(a.visible) + len(a.covered)
        for mode in MODES:
            mv, mf = a.occ.get((mode, "moving"), (np.empty((0, 3)), np.empty((0, 3), np.int64)))
            fv, ff = a.occ.get((mode, "fixed"), (np.empty((0, 3)), np.empty((0, 3), np.int64)))
            V = np.vstack([a.v, mv, fv])
            F = np.vstack([a.visible, a.covered, mf + nv, ff + nv + len(mv)]).astype(np.int64)
            self.scene[mode] = (gl_overlay.GLMeshRenderer(V, F), mv.copy(), slice(nv, nv + len(mv)), len(a.visible),
                                (nf, nf + len(mf)))
        self.seg = make_segmenter(self.cfg["segmenter"], self.cfg["device"])
        self._net, self._net_tried, self.sigma_scale = None, False, {"a1": 1.0, "rigid": 1.0}
        if SIGMA_SCALE.is_file():
            self.sigma_scale.update(json.loads(SIGMA_SCALE.read_text(encoding="utf-8")))
        self.keys, self.next_id = [], 1
        self.hist = deque(maxlen=DETECT["median_of"])       # (t, score) with the registration in use
        self.state, self.reg = "checking", None
        self.infos = {}                                     # (key id, registration, mode) -> Fisher information
        self.load_s = time.perf_counter() - t0
        if self.cfg["method"] != "opt":
            self.net()

    def net(self):
        """The network (loaded at the first use; None without its weights or torch)."""
        if self._net is None and not self._net_tried:
            self._net_tried = True
            if NET.is_file():
                try:
                    import refiner as rf
                    self._net = rf.Refiner(NET, device=self.cfg["device"], face=self.a, renderer=self.ren)
                except Exception as e:  # noqa: BLE001  without the network the optimisation still corrects
                    logger.warning("network not loaded: %s", e)
        return self._net

    # the masks of one view
    def occluders(self, K, dist, T_C_M, T_M_B, size, mode):
        """Known occluders of the face in a view, where they are in front of it: (moving, all) as uint8 0/255 masks.
        moving: the parts moving with the board, at the design's place on it (exact whatever the registration);
        all: those and the parts on the head and the cap (the face under it), at the registration T_M_B. Only the
        moving ones are taken out of the observed face: the others are where the registration in use puts them,
        which after an impact is not where they are, and a cut there would draw the old outline into the observed
        face. All of them, dilated, are where outline points do not count."""
        w, h = size
        ren, mv, sl, n_vis, (m0, m1) = self.scene[mode]
        a = self.a
        if len(mv) and a.T_M_B_design is not None:       # the moving parts in B as the registration puts the board
            D = np.linalg.inv(T_M_B) @ a.T_M_B_design
            ren.v[sl] = mv @ D[:3, :3].T + D[:3, 3]
        r = ren.render(K, dist, T_C_M @ T_M_B, size, edges=False)
        moving, occ = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
        bh, bw = r.faces.shape
        if bh and bw:
            x0, y0 = r.offset
            occ[y0:y0 + bh, x0:x0 + bw] = np.where(r.faces >= n_vis, 255, 0).astype(np.uint8)
            moving[y0:y0 + bh, x0:x0 + bw] = np.where((r.faces >= m0) & (r.faces < m1), 255, 0).astype(np.uint8)
        return moving, occ

    def box(self, msg):
        """The prompt box of SAM in a frame's cut (the visible face at the registration in use, grown by
        DETECT["box_margin"]), or None when the face is not in view."""
        img, K, dist = msg["img"], np.asarray(msg["K"], np.float64), np.asarray(msg["dist"], np.float64)
        h, w = img.shape[:2]
        T = np.asarray(msg["T_C_M"], np.float64) @ np.asarray(msg["T_M_B"], np.float64)
        m = self.ren_vis.render(K, dist, T, (w, h), edges=False).mask
        if cv2.countNonZero(m) < DETECT["min_face_px"]:
            return None
        x, y, bw, bh = cv2.boundingRect(m)
        b = np.array([x, y, x + bw - 1, y + bh - 1], float)
        ctr, half = (b[:2] + b[2:]) / 2, (b[2:] - b[:2]) / 2 * (1 + DETECT["box_margin"])
        return np.r_[np.maximum(ctr - half, 0), np.minimum(ctr + half, [w - 1, h - 1])]

    def masks(self, msg, sam, mode):
        """(observed, rendered, ignore) of a view from its SAM mask."""
        img, K, dist = msg["img"], np.asarray(msg["K"], np.float64), np.asarray(msg["dist"], np.float64)
        T_C_M, T_M_B = np.asarray(msg["T_C_M"], np.float64), np.asarray(msg["T_M_B"], np.float64)
        h, w = img.shape[:2]
        moving, occ = self.occluders(K, dist, T_C_M, T_M_B, (w, h), mode)
        obs = np.where((sam > 0) & (moving == 0), 255, 0).astype(np.uint8)
        k = 2 * DETECT["dilate_px"] + 1
        ignore = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        ren = self.ren.render(K, dist, T_C_M @ T_M_B, (w, h), edges=False).mask
        return obs, ren, ignore

    # detection: start (the box, on the thread of the OpenGL context), SAM, finish (the rest, same thread); the
    # worker's loop runs SAM on a thread of its own, so the next frame's SAM overlaps this one's finish
    def detect(self, msg):
        """The whole detection of a frame, in order (the bench and the tests)."""
        t0 = time.perf_counter()
        box = self.box(msg)
        t1 = time.perf_counter()
        sam = None if box is None else self.seg(msg["img"], box)[0]
        return self.finish(msg, box, sam, {"box_ms": (t1 - t0) * 1e3, "sam_ms": (time.perf_counter() - t1) * 1e3},
                           t0)

    def finish(self, msg, box, sam, tm, t0=None):
        t0 = time.perf_counter() if t0 is None else t0
        t1 = time.perf_counter()
        K, dist = np.asarray(msg["K"], np.float64), np.asarray(msg["dist"], np.float64)
        T_C_M, T_M_B = np.asarray(msg["T_C_M"], np.float64), np.asarray(msg["T_M_B"], np.float64)
        mode = msg.get("mode") or self.cfg["mode"]
        reg = tuple(np.round(T_M_B[:3].ravel(), 4))
        if reg != self.reg:                      # another registration: its scores start again
            self.reg = reg
            self.hist.clear()
            self.state = "checking"
        out = {"type": "detect", "n": msg.get("n"), "t": msg.get("t"), "reg": reg}
        if box is None or sam is None:
            out.update(state="noface", ms={**tm, "total_ms": (time.perf_counter() - t0) * 1e3})
            return out
        img = msg["img"]
        obs, ren, ignore = self.masks(msg, sam, mode)
        t2 = time.perf_counter()
        T = T_C_M @ T_M_B
        scale = sc.mm_per_px(K, T, self.a.v)
        vs = sc.view_score(obs, ren, scale, ignore, trim=DETECT["trim"])
        if msg.get("trims"):
            out["trim_scores"] = {str(q): sc.view_score(obs, ren, scale, ignore, trim=q)["score_mm"]
                                  for q in msg["trims"]}
        t3 = time.perf_counter()
        tm.update(masks_ms=(t2 - t1) * 1e3, score_ms=(t3 - t2) * 1e3)
        now = float(msg.get("t") or time.perf_counter())
        if vs["n_obs"] >= vg.MIN_POINTS and vs["n_ren"] >= vg.MIN_POINTS:
            self.hist.append((now, vs["score_mm"]))
        med = float(np.median([s for _, s in self.hist])) if self.hist else None
        was = self.state
        if med is None or len(self.hist) < min(2, DETECT["median_of"]):
            state = "checking" if was not in ("ok", "lost") else was
        elif was == "lost":
            state = "lost" if med > DETECT["back_share"] * TAU_MM else "ok"
        else:
            state = "lost" if med > TAU_MM else "ok"
        if state == "lost" and was != "lost":     # the key frames from before the change no longer show the face
            since = min((t for t, s in self.hist if s > TAU_MM), default=now)
            self.keys = [k for k in self.keys if k.t >= since]
        self.state = state
        self.keys = [k for k in self.keys if now - k.t <= KEY["age_s"]]
        d = view_dir(T_C_M, T_M_B, self.a.o)
        if vs["n_obs"] >= vg.MIN_POINTS and all(np.degrees(np.arccos(np.clip(d @ k.dir, -1, 1))) >= KEY["deg"]
                                                for k in self.keys):
            self.keys.append(Key(self.next_id, now, img, obs, ignore, K, dist, T_C_M, d, vs["score_mm"]))
            self.next_id += 1
            if len(self.keys) > KEY["max"]:
                self.keys.pop(0)
        out.update(state=state, score_mm=vs["score_mm"], median_mm=med, points=int(vs["n_obs"]), keys=len(self.keys),
                   box=[float(x) for x in box])
        if state == "lost" or msg.get("guide"):
            t4 = time.perf_counter()
            out["guide"] = self.guide(msg, T_M_B, mode)
            tm["guide_ms"] = (time.perf_counter() - t4) * 1e3
        tm["total_ms"] = (time.perf_counter() - t0) * 1e3
        out["ms"] = tm
        return out

    # view guidance
    def info(self, key, T_M_B, mode):
        k = (key.id, self.reg, mode)
        if k not in self.infos:
            self.infos[k] = vg.fisher(self.ren, key.candidate(), T_M_B, mode, self.a.c, fast=True)[0]
            if len(self.infos) > 8 * KEY["max"]:
                self.infos.pop(next(iter(self.infos)))
        return self.infos[k]

    def centre(self, mode):
        return self.a.c if mode == "a1" else self.a.o

    def predicted(self, infos, mode, draws=GUIDE["draws"]):
        """Predicted mean and 95th percentile of the target error (mm) after views of these informations."""
        S0 = vg.prior(mode)
        H = np.linalg.inv(S0) + sum(infos, np.zeros_like(S0))
        return co.predicted_tre(np.linalg.inv(H), np.eye(3), self.centre(mode), self.a.targets, draws=draws)

    def choose(self, T_M_B, mode, k_max=KEY["views"]):
        """The key frames to correct from, best first (greedy on the predicted error), and the predicted 95 % error
        after them."""
        if not self.keys:
            return [], float("inf")
        infos = [self.info(k, T_M_B, mode) for k in self.keys]
        W = vg.target_weight(self.a.targets, self.centre(mode), mode)
        order, _ = vg.greedy(infos, vg.prior(mode), W, k_max)
        return [self.keys[i] for i in order], self.predicted([infos[i] for i in order], mode)[1]

    def guide(self, msg, T_M_B, mode):
        """Views held, predicted error after them, enough or not, and the way to move the camera."""
        chosen, p95 = self.choose(T_M_B, mode)
        held = [self.info(k, T_M_B, mode) for k in chosen]
        out = {"views": len(chosen), "pred95_mm": p95, "enough": bool(chosen) and p95 <= ENOUGH_MM, "arrow": None}
        if out["enough"]:
            return out
        K, size = np.asarray(msg["K_full"], np.float64), tuple(msg["full"])
        dist = np.asarray(msg["dist"], np.float64)
        T_C_M = np.asarray(msg["T_C_M_full"], np.float64)
        T_C_B = T_C_M @ T_M_B
        T_B_C = np.linalg.inv(T_C_B)
        o = self.a.o
        Ti = np.linalg.inv(T_M_B)
        spec = msg.get("board") or {"squares": 5, "square_mm": 9.0}
        best, gain = None, 0.0
        base = self.predicted(held, mode)[1] if held else self.predicted([], mode)[1]
        for phi in np.radians(np.arange(0, 360, 45)):
            u = np.array([np.cos(phi), np.sin(phi), 0.0])           # where the camera goes, camera frame
            axis_B = T_B_C[:3, :3] @ np.array([-np.sin(phi), np.cos(phi), 0.0])
            for sign in (1, -1):
                Tn = vg.turn(axis_B, o, sign * GUIDE["deg"]) @ T_B_C
                d = T_C_B[:3, :3] @ (Tn[:3, 3] - T_B_C[:3, 3])
                if d @ u > 0:
                    break
            cand = vg.Candidate("g", K, np.linalg.inv(Tn) @ Ti, size, dist)
            if vg.board_corners(cand, spec) < vg.MIN_CORNERS:
                continue
            I, n = vg.fisher(self.ren, cand, T_M_B, mode, self.a.c, fast=True)
            if n < vg.MIN_POINTS:
                continue
            p = self.predicted(held + [I], mode)[1]
            if base - p > gain:
                best, gain = (float(d[0]), float(d[1])), base - p
        if best is not None and gain >= GUIDE["gain"] * base:
            n = np.hypot(*best)
            out["arrow"] = [best[0] / n, best[1] / n]
            out["gain_mm"] = gain
        return out

    # correction
    def correct(self, msg):
        t0 = time.perf_counter()
        T0 = np.asarray(msg["T_M_B"], np.float64)
        mode = msg.get("mode") or self.cfg["mode"]
        method = msg.get("method") or self.cfg["method"]
        tau, u = float(msg.get("tau_mm", TAU_MM)), float(msg.get("u_mm", U_MM))
        out = {"type": "corrected", "id": msg.get("id"), "method": method, "mode": mode, "applied": False,
               "T_M_B0": T0, "apply": msg.get("apply", True), "auto": bool(msg.get("auto"))}
        self.send({"type": "progress", "id": msg.get("id"), "phase": "views"})
        chosen, p95 = self.choose(T0, mode, int(msg.get("views") or KEY["views"]))
        t1 = time.perf_counter()
        out.update(views=len(chosen), pred95_views_mm=p95, key_ids=[k.id for k in chosen])
        tm = {"views_s": t1 - t0}
        need = co.MIN_VIEWS.get(mode, 1) if hasattr(co, "MIN_VIEWS") else 1
        if not chosen or (method != "net" and len(chosen) < need):
            out.update(why="no_views", time_s=tm)     # 6 degrees of freedom are never taken from one view
            return out
        views = [k.view() for k in chosen]
        X = self.a.targets
        centre = self.a.c if mode == "a1" else None
        net = self.net() if method in ("net", "net_opt") else None
        if method in ("net", "net_opt") and net is None:
            method = out["method"] = "opt"
        res = None
        if method == "opt":
            self.send({"type": "progress", "id": msg.get("id"), "phase": "optimisation"})
            res = co.correct(self.ren, views, T0, X, mode, center=centre)
            tm["opt_s"] = res.time_s
            T_new, tre95, score_mm = res.T_M_B, res.tre_pred_95, self.score_at(views, res.T_M_B)
            applied = bool(score_mm <= tau and tre95 < u)
        else:
            import refiner as rf
            self.send({"type": "progress", "id": msg.get("id"), "phase": "network"})
            t2 = time.perf_counter()
            srcs = [rf.make_source(k.img, k.obs, k.K, k.dist, k.T_C_M, T0, self.a.o) for k in chosen]
            T_net, total, mu, lv, _ = net.refine(srcs, T0, mode, self.a.c)
            tm["net_s"] = time.perf_counter() - t2
            self.send({"type": "progress", "id": msg.get("id"), "phase": "optimisation"})
            t3 = time.perf_counter()
            res = rf.optimize(self.ren, views, T_net, X, mode, center=centre) if len(views) >= need else None
            tm["opt_s"] = time.perf_counter() - t3
            if method == "net_opt":
                T_new, tre95, score_mm = res.T_M_B, res.tre_pred_95, self.score_at(views, res.T_M_B)
                applied = bool(score_mm <= tau and tre95 < u)
            else:
                t4 = time.perf_counter()
                sig = self.sigma_scale.get(mode, 1.0) * np.exp(0.5 * np.asarray(lv, np.float64))
                cov = np.diag(np.square(np.r_[np.radians(sig[:3]), sig[3:]]))
                o = self.a.c if mode == "a1" else self.a.o
                _, tre95 = co.predicted_tre(cov, total[:3, :3], o, X, draws=CORRECT["draws"])
                score_mm = self.score_at(views, T_net)
                tm["check_s"] = time.perf_counter() - t4
                T_new = T_net
                if res is None:            # one view of 6 degrees of freedom: no check, never applied
                    applied, gap = False, float("inf")
                else:
                    gap = mean_target_mm(T_net, res.T_M_B, X)
                    out.update(agree_mm=gap, opt_score_mm=res.score_mm, T_M_B_opt=res.T_M_B)
                    applied = bool(score_mm <= tau and tre95 < u and gap <= CORRECT["agree_mm"])
        tm["total_s"] = time.perf_counter() - t0
        why = "" if applied else ("score" if score_mm > tau else "uncertain" if tre95 >= u else "disagree")
        out.update(applied=bool(applied), T_M_B=T_new, tre95_mm=float(tre95), score_mm=float(score_mm),
                   moved_mm=mean_target_mm(T0, T_new, X), why=why, time_s=tm)
        return out

    def score_at(self, views, T_M_B):
        s = []
        for v in views:
            T = v.T_C_M @ T_M_B
            r = self.ren.render(v.K, v.dist, T, v.size, edges=False)
            s.append(sc.view_score(v.observed, r.mask, sc.mm_per_px(v.K, T, self.a.v), v.ignore,
                                   trim=DETECT["trim"])["score_mm"])
        return float(np.mean(s))

    def handle(self, msg):
        kind = msg.get("type")
        if kind == "frame":
            return self.detect(msg)
        if kind == "correct":
            return self.correct(msg)
        if kind == "reset":
            self.keys, self.hist, self.reg = [], deque(maxlen=DETECT["median_of"]), None
            self.infos.clear()
            return {"type": "reset"}
        if kind == "config":
            self.cfg.update({k: v for k, v in msg.items() if k in SETTINGS})
            if self.cfg["method"] != "opt":
                self.net()
            return {"type": "config", "net": self._net is not None}
        return {"type": "error", "where": "handle", "what": f"unknown message {kind}"}


def serve(address, key):
    """The worker's loop: two connections from the live view (commands in, results out), one message at a time."""
    from multiprocessing.connection import Listener
    # a library that prints must not get in the way: the protocol is on the pipes, stdout goes to stderr
    try:
        os.dup2(2, 1)
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, format="check worker: %(message)s", stream=sys.stderr)
    cv2.setNumThreads(WORKER_THREADS)
    try:
        import torch
        torch.set_num_threads(WORKER_THREADS)
    except ImportError:
        pass
    t0 = time.perf_counter()
    with Listener(address, family="AF_PIPE", authkey=key) as lst:
        logger.info("listening (%.1f s)", time.perf_counter() - t0)
        cmd = lst.accept()
        res = lst.accept()
        try:
            cfg = cmd.recv()
            logger.info("loading %s (%.1f s)", cfg, time.perf_counter() - t0)
            w = Worker(cfg, send=res.send)
            logger.info("loaded (%.1f s)", time.perf_counter() - t0)
            if os.name == "nt":     # loaded at the normal priority; from now on the video comes first
                import ctypes
                k32 = ctypes.windll.kernel32
                k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
            res.send({"type": "ready", "load_s": w.load_s, "net": w._net is not None, "device": w.seg.device,
                      "segmenter": w.cfg["segmenter"], "pid": os.getpid()})
        except Exception as e:  # noqa: BLE001  said to the live view, which shows the check as failed
            logger.exception("worker failed to load")
            res.send({"type": "failed", "what": f"{type(e).__name__}: {e}"})
            return 1
        return _loop(w, cmd, res)


def _loop(w, cmd, res):
    """Messages in order; a frame's SAM runs on a thread of its own while the main thread (the one with the OpenGL
    context) starts the next frame and finishes the one before, so SAM's time is not added to the rest."""
    import queue
    sam_in, sam_out = queue.Queue(), queue.Queue()

    def sam_loop():
        while True:
            job = sam_in.get()
            if job is None:
                return
            msg, box, tm = job
            t = time.perf_counter()
            try:
                m = w.seg(msg["img"], box)[0]
            except Exception as e:  # noqa: BLE001  said with the frame's answer
                m = e
            tm["sam_ms"] = (time.perf_counter() - t) * 1e3
            sam_out.put((msg, box, m, tm))
    th = threading.Thread(target=sam_loop, daemon=True, name="sam")
    th.start()
    waiting = 0

    def send(out):
        try:
            res.send(out)
            return True
        except (EOFError, OSError):
            return False

    def finish(timeout):
        nonlocal waiting
        try:
            msg, box, m, tm = sam_out.get(timeout=timeout) if timeout else sam_out.get_nowait()
        except queue.Empty:
            return None
        waiting -= 1
        if isinstance(m, Exception):
            out = {"type": "error", "where": "frame", "what": f"{type(m).__name__}: {m}", "n": msg.get("n")}
        else:
            try:
                out = w.finish(msg, box, m, tm, msg.get("_t0"))
            except Exception as e:  # noqa: BLE001  one frame failed: said, and the next one comes
                logger.exception("frame failed")
                out = {"type": "error", "where": "frame", "what": f"{type(e).__name__}: {e}", "n": msg.get("n")}
        return send(out)

    try:
        while True:
            while waiting and finish(0) is not None:
                pass
            try:
                if not cmd.poll(0.005 if waiting else 0.2):
                    if waiting:
                        if finish(0.02) is False:
                            return 0
                    continue
                msg = cmd.recv()
            except (EOFError, OSError):
                return 0
            kind = msg.get("type")
            if kind == "quit":
                return 0
            if kind == "frame":
                t0 = time.perf_counter()
                try:
                    box = w.box(msg)
                except Exception as e:  # noqa: BLE001
                    logger.exception("frame failed")
                    if not send({"type": "error", "where": "frame", "what": f"{type(e).__name__}: {e}"}):
                        return 0
                    continue
                if box is None:
                    if not send(w.finish(msg, None, None, {"box_ms": (time.perf_counter() - t0) * 1e3}, t0)):
                        return 0
                    continue
                msg["_t0"] = t0
                sam_in.put((msg, box, {"box_ms": (time.perf_counter() - t0) * 1e3}))
                waiting += 1
                continue
            while waiting:                       # a correction, a reset: the frames under way first
                if finish(1.0) is False:
                    return 0
            try:
                out = w.handle(msg)
            except Exception as e:  # noqa: BLE001  one message failed: said, and the next one comes
                logger.exception("message %s failed", kind)
                out = {"type": "error", "where": kind, "what": f"{type(e).__name__}: {e}", "id": msg.get("id")}
            if not send(out):
                return 0
    finally:
        sam_in.put(None)


# ── the board check ─────────────────────────────────────────────────────────
class BoardCheck:
    """Reprojection error of the board's corners over a short window: RMS and per quadrant (see BOARD)."""

    def __init__(self):
        self.rows = deque()
        self.lock = threading.Lock()

    def add(self, t, corners, ids, T, obj, K, dist, spec, limit_px):
        if T is None or corners is None or len(corners) < 4:
            return
        P = np.asarray(obj, np.float64)[np.asarray(ids, int)]
        uv = cv2.projectPoints(P @ T[:3, :3].T + T[:3, 3], np.zeros(3), np.zeros(3), K,
                               np.zeros(5) if dist is None else dist)[0].reshape(-1, 2)
        r = np.linalg.norm(uv - np.asarray(corners, np.float64).reshape(-1, 2), axis=1)
        nx, ny = spec["squares"]
        half = np.array([nx, ny]) * spec["square_mm"] / 2
        q = (P[:, 0] > half[0]).astype(int) + 2 * (P[:, 1] > half[1]).astype(int)
        means = np.array([r[q == k].mean() if np.any(q == k) else np.nan for k in range(4)])
        with self.lock:
            self.rows.append((t, float(np.sqrt(np.mean(r * r))), means, float(limit_px)))
            while self.rows and t - self.rows[0][0] > BOARD["window_s"]:
                self.rows.popleft()

    def state(self, now=None):
        with self.lock:
            rows = list(self.rows)
        if now is not None:
            rows = [r for r in rows if now - r[0] <= BOARD["window_s"]]
        out = {"frames": len(rows), "damaged": False, "rms_px": None, "ratio": None, "worst_px": None}
        if not rows:
            return out
        rms = float(np.median([r[1] for r in rows]))
        q = np.nanmean(np.stack([r[2] for r in rows]), axis=0)
        ratio, worst = None, None
        if np.sum(np.isfinite(q)) >= 3:
            k = int(np.nanargmax(q))
            rest = np.delete(q, k)
            worst = float(q[k])
            ratio = float(worst / max(np.nanmean(rest), 1e-6))
        limit = rows[-1][3]
        out.update(rms_px=rms, ratio=ratio, worst_px=worst)
        if len(rows) >= BOARD["min_frames"]:
            out["damaged"] = bool(rms > BOARD["rms_share"] * limit
                                  or (ratio is not None and ratio > BOARD["ratio"] and worst > BOARD["floor_px"]))
        return out

    def clear(self):
        with self.lock:
            self.rows.clear()


# ── the live view's side ────────────────────────────────────────────────────
class RegistrationCheck:
    """The check of the live view (see the module docstring): the worker process, the frames to it, the answers, the
    board check, what is drawn and the status. Every method is quick and never raises into the live view's threads.

    Args:
        settings: overrides of SETTINGS (method, mode, auto, segmenter, device, boards).
        python: the interpreter of the worker (default this one).
        log: a callable for progress lines.
    """

    def __init__(self, settings=None, python=None, log=None):
        self.s = {**SETTINGS, **(settings or {})}
        self.python = python or sys.executable
        self.log = log or (lambda msg: None)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._quit = threading.Event()
        self.board = BoardCheck()
        m = np.load(FACE_MESH) if FACE_MESH.is_file() else None
        self._v = None if m is None else m["vertices"].astype(np.float64)
        self._o = None if m is None else self._v.mean(0)
        self._pts = None if m is None else self._v[np.linspace(0, len(self._v) - 1, 300).astype(int)]
        self._proc, self._cmd, self._res = None, None, None
        self._worker, self._worker_info, self._why = "off", {}, ""
        self._restarts = 0
        self._pending = None          # the newest frame offered: (item, T_C_M, K, dist, T_M_B, spec, t)
        self._frames_out = 0          # frames sent and not answered yet (at most DETECT["in_flight"])
        self._busy = None             # a correction or a reset sent and not answered: ("correct", id), ("reset", t)
        self._t_sent = 0.0
        self._want_correct = None     # a correction asked for and not sent yet
        self._want_reset = False      # a reset asked for and not sent yet
        self._last = None             # the last detection answer (with the time it came)
        self._progress = None         # (id, phase, t) of the correction under way
        self._correction = None       # the last correction answer
        self._applied = None          # the correction in use (its answer), None when the registration is not one
        self._next_id = 1
        self._det_times = deque(maxlen=30)
        self._tm = deque(maxlen=30)
        self._send_ms = deque(maxlen=30)
        self._badge = False
        self._auto_after = 0          # key frames needed before the next automatic correction
        self._active = False
        self.ar = None
        self.history = None           # a list (bench, tests): every answer is kept in it
        self._err_tail = []
        self._threads = []

    # hooks of the live view
    def attach(self, ar):
        self.ar = ar

    def active_for(self, spec):
        return self._v is not None and spec is not None and spec.get("name") in self.s["boards"]

    def frame(self, item, res, K, dist, setup, limit_px):
        """A frame tracked with its own board pose (the live view's track thread): the board check, and the frame
        kept for the worker when it is free."""
        spec, T_M_B = setup["spec"], setup["T_M_B"]
        self.board.add(item.t_capture, res.corners, res.ids, res.T, setup["obj"], K, dist, spec, limit_px)
        active = self.active_for(spec) and T_M_B is not None
        with self._cv:
            self._active = active
            if not active or self._quit.is_set():
                return
            self._pending = (item, np.array(res.T, np.float64), K, dist, np.array(T_M_B, np.float64), spec,
                             item.t_capture)
            self._cv.notify_all()
        self._ensure()

    def draw(self, img, k, s, T_C_B, Kd, dist, size):
        """The arrow and the badge over the composed picture (the live view's compose thread): img already turned by
        k quarter turns, s the scale of the display over the frame, T_C_B the camera from the head at the pose drawn,
        Kd and dist the display's camera, size (w, h) of the display before the turn."""
        st = self._shown_state()
        if st is None:
            return
        state, arrow = st
        if arrow is not None and T_C_B is not None and self._o is not None:
            uv, z = project(Kd, dist, T_C_B, np.vstack([self._o, self._pts]))
            if z[0] > 1:
                c = uv[0]
                r = float(np.median(np.linalg.norm(uv[1:] - c, axis=1)))
                self._draw_arrow(img, k, size, c, np.asarray(arrow, float), r)
        if self._badge:
            self._draw_badge(img, state)

    def _shown_state(self):
        with self._lock:
            last, active = self._last, self._active
            correcting = self._progress is not None or self._want_correct is not None
            worker = self._worker
        if not active:
            return None
        now = time.perf_counter()
        if self.board.state(now)["damaged"]:
            return "board", None
        if correcting:
            return "correcting", None
        if worker != "ready" or last is None or now - last["at"] > DETECT["stale_s"]:
            return ("loading" if worker in ("starting", "off") else "failed" if worker == "failed" else "checking"), None
        g = last.get("guide") or {}
        arrow = g.get("arrow") if last.get("state") == "lost" and now - last["at"] <= GUIDE["stale_s"] else None
        return last.get("state", "checking"), arrow

    @staticmethod
    def _turn(p, d, k, size):
        """A point and a direction of the display before k clockwise quarter turns, after them."""
        w, h = size
        x, y = p
        dx, dy = d
        for _ in range(k % 4):
            x, y, dx, dy = h - 1 - y, x, -dy, dx
            w, h = h, w
        return np.array([x, y]), np.array([dx, dy])

    def _draw_arrow(self, img, k, size, c, u, r):
        H, W = img.shape[:2]
        c, u = self._turn(c, u, k, size)
        u = u / max(np.hypot(*u), 1e-9)
        a = c + u * 0.62 * r
        b = c + u * (0.62 * r + max(2.2 * DRAW["head_px"], 0.55 * r))
        m = 1.2 * DRAW["head_px"]
        for p in (a, b):                       # inside the picture
            if not (m <= p[0] < W - m and m <= p[1] < H - m):
                b = np.clip(b, m, [W - m, H - m])
                a = b - u * 2.2 * DRAW["head_px"]
                break
        lw, L, halo = DRAW["arrow_px"], DRAW["head_px"], DRAW["halo_px"]
        for colour, extra in ((bgr(COLOURS["halo"]), halo), (bgr(COLOURS["arrow"]), 0.0)):
            stop = b - u * arrowhead.inset(L, lw)
            cv2.line(img, tuple(np.round(a * 16).astype(int)), tuple(np.round(stop * 16).astype(int)), colour,
                     int(round(lw + 2 * extra)), cv2.LINE_AA, 4)
            arrowhead.fill(img, b + u * extra * 1.6, u, L + extra * 2.6, colour)

    def _draw_badge(self, img, state):
        H, W = img.shape[:2]
        R = DRAW["badge_px"]
        c = np.array([R + 12.0, H - R - 12.0])
        S = 16
        p = tuple(np.round(c * S).astype(int))
        if state == "ok":       # a full disc with a tick
            cv2.circle(img, p, int(R * S), bgr(COLOURS["ok"]), -1, cv2.LINE_AA, 4)
            tick = np.array([[-0.45, 0.02], [-0.12, 0.35], [0.48, -0.32]]) * R + c
            cv2.polylines(img, [np.round(tick * S).astype(np.int32)], False, bgr(COLOURS["mark"]), 3, cv2.LINE_AA, 4)
        elif state in ("lost", "board"):       # a diamond (a fault), with a bar and a dot
            d = np.array([[0, -1.18], [1.18, 0], [0, 1.18], [-1.18, 0]]) * R + c
            cv2.fillConvexPoly(img, np.round(d * S).astype(np.int32), bgr(COLOURS[state]), cv2.LINE_AA, 4)
            cv2.line(img, tuple(np.round((c + [0, -0.5 * R]) * S).astype(int)),
                     tuple(np.round((c + [0, 0.12 * R]) * S).astype(int)), bgr(COLOURS["mark"]), 3, cv2.LINE_AA, 4)
            cv2.circle(img, tuple(np.round((c + [0, 0.45 * R]) * S).astype(int)), int(0.14 * R * S),
                       bgr(COLOURS["mark"]), -1, cv2.LINE_AA, 4)
        else:                   # a ring: something under way
            cv2.circle(img, p, int((R - 2) * S), bgr(COLOURS["wait"]), 3, cv2.LINE_AA, 4)

    # the worker
    def _ensure(self):
        with self._lock:
            if self._worker != "off" or self._quit.is_set():
                return
            self._worker = "starting"
        th = threading.Thread(target=self._run, daemon=True, name="check_worker")
        th.start()
        self._threads.append(th)

    def _run(self):
        """Starts the worker and serves it until close(): the frames and corrections out, the answers in."""
        while not self._quit.is_set():
            ok = self._start()
            if ok:
                self._serve()
            self._stop_proc()
            with self._lock:
                if self._quit.is_set():
                    return
                self._restarts += 1
                if self._restarts > RESTARTS:
                    self._worker = "failed"
                    return
                self._worker = "starting"
            self._quit.wait(2.0)

    def _start(self):
        from multiprocessing.connection import Client
        key = secrets.token_bytes(16)
        address = rf"\\.\pipe\impactguide-check-{os.getpid()}-{secrets.token_hex(6)}"
        env = dict(os.environ)
        env["IMPACTGUIDE_CHECK_KEY"] = key.hex()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(k, str(WORKER_THREADS))     # the worker leaves cores to the live view
        flags = 0
        if os.name == "nt":
            flags = 0x08000000                   # no console window
        try:
            self._proc = subprocess.Popen([self.python, str(Path(__file__).resolve()), "worker", address], cwd=str(SRC),
                                          env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE, creationflags=flags)
        except OSError as e:
            self._fail(f"cannot start: {e}")
            return False
        threading.Thread(target=self._read_err, args=(self._proc,), daemon=True, name="check_worker_log").start()
        t0 = time.perf_counter()

        def connect():
            while not self._quit.is_set():
                try:
                    return Client(address, family="AF_PIPE", authkey=key)
                except (FileNotFoundError, OSError):
                    if self._proc.poll() is not None or time.perf_counter() - t0 > START_S:
                        return None
                    time.sleep(0.05)
            return None
        self._cmd = connect()
        self._res = connect() if self._cmd is not None else None
        if self._res is None:
            if not self._quit.is_set():
                self._fail("the worker did not open its pipe")
            return False
        try:
            self._cmd.send({k: v for k, v in self.s.items() if k in SETTINGS and k != "boards"})
            while not self._res.poll(0.2):
                if self._quit.is_set() or self._proc.poll() is not None or time.perf_counter() - t0 > START_S:
                    self._fail("the worker did not load")
                    return False
            msg = self._res.recv()
        except (EOFError, OSError) as e:
            self._fail(f"the worker went: {e}")
            return False
        if msg.get("type") != "ready":
            self._fail(msg.get("what", "the worker did not load"))
            with self._lock:
                self._restarts = RESTARTS      # a model that cannot load will not load next time either
            return False
        with self._cv:
            self._worker, self._worker_info = "ready", msg
            self._frames_out, self._busy = 0, None
            self._cv.notify_all()
        self.log(f"check worker ready in {time.perf_counter() - t0:.1f} s ({msg.get('segmenter')}, "
                 f"{msg.get('device')}, network {'on' if msg.get('net') else 'off'})")
        return True

    def _fail(self, why):
        time.sleep(0.2)
        tail = list(self._err_tail)[-8:]
        logger.warning("registration check: %s %s", why, " | ".join(tail))
        with self._lock:
            self._why = why

    def _read_err(self, proc):
        tail = self._err_tail = deque(maxlen=20)       # the worker's last lines, for a failure
        try:
            for line in iter(proc.stderr.readline, b""):
                s = line.decode("utf-8", "replace").rstrip()
                if s:
                    tail.append(s)
                    logger.debug("%s", s)
        except (OSError, ValueError):
            pass

    def _serve(self):
        reader = threading.Thread(target=self._read, daemon=True, name="check_read")
        reader.start()
        while not self._quit.is_set() and self._proc.poll() is None and reader.is_alive():
            with self._cv:
                self._cv.wait_for(lambda: self._quit.is_set() or self._sendable(), timeout=0.2)
                if self._quit.is_set():
                    return
                job = self._next_job()
            if job is None:
                continue
            t0 = time.perf_counter()
            try:
                msg = self._message(job)
                if msg is None:
                    with self._cv:
                        self._frames_out -= 1
                        self._cv.notify_all()
                    continue
                self._cmd.send(msg)
            except (EOFError, OSError, ValueError) as e:
                logger.info("registration check: send failed: %s", e)
                return
            self._send_ms.append((time.perf_counter() - t0) * 1e3)
        reader.join(0.5)

    def _sendable(self):
        if self._worker != "ready" or self._busy is not None:
            return False
        if self._want_correct is not None or self._want_reset:
            return self._frames_out == 0        # the frames under way first
        return (self._pending is not None and self._frames_out < DETECT["in_flight"]
                and time.perf_counter() - self._t_sent >= 1.0 / DETECT["hz"])

    def _next_job(self):
        if not self._sendable():
            return None
        if self._want_reset:
            self._want_reset, self._busy = False, ("reset", time.perf_counter())
            return ("reset", None)
        if self._want_correct is not None:
            job, self._want_correct = ("correct", self._want_correct), None
            self._busy = ("correct", job[1]["id"])
            self._progress = (job[1]["id"], "views", time.perf_counter())
            return job
        job, self._pending = ("frame", self._pending), None
        self._frames_out += 1
        self._t_sent = time.perf_counter()
        return job

    def _message(self, job):
        kind, data = job
        if kind == "reset":
            return {"type": "reset"}
        if kind == "correct":
            return {"type": "correct", **data}
        item, T_C_M, K, dist, T_M_B, spec, t = data
        cut = self.cut(item, T_C_M, K, dist, T_M_B)
        if cut is None:
            with self._lock:
                self._last = {"type": "detect", "state": "noface", "at": time.perf_counter()}
            return None
        img, Kc = cut
        return {"type": "frame", "n": item.n, "t": t, "img": img, "K": Kc, "dist": dist, "T_C_M": T_C_M,
                "T_M_B": T_M_B, "K_full": K, "T_C_M_full": T_C_M, "full": list(item.size),
                "board": {"squares": list(spec["squares"]), "square_mm": spec["square_mm"]},
                "mode": self.s["mode"]}

    def cut(self, item, T_C_M, K, dist, T_M_B):
        box = cut_box(item.size, T_C_M, K, dist, T_M_B, self._o, self._pts)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        img = np.ascontiguousarray(item.bgr()[y0:y1, x0:x1])
        Kc = np.array(K, np.float64)
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        return img, Kc

    def _read(self):
        while not self._quit.is_set():
            try:
                if not self._res.poll(0.2):
                    if self._proc.poll() is not None:
                        return
                    continue
                msg = self._res.recv()
            except (EOFError, OSError):
                return
            self._answer(msg)

    def _answer(self, msg):
        now = time.perf_counter()
        kind = msg.get("type")
        apply = None
        with self._cv:
            if self.history is not None:
                self.history.append({**{k: v for k, v in msg.items() if k != "img"}, "at": now})
            if kind == "detect":
                self._frames_out = max(0, self._frames_out - 1)
                self._last = {**msg, "at": now}
                self._det_times.append(now)
                if msg.get("ms"):
                    self._tm.append(msg["ms"])
                auto = (self.s["auto"] and msg.get("state") == "lost" and (msg.get("guide") or {}).get("enough")
                        and self._want_correct is None and self._progress is None
                        and (msg.get("keys") or 0) >= self._auto_after
                        and not self.board.state(now)["damaged"])
                if auto:
                    self._ask(None, None, True, auto=True)
            elif kind == "progress":
                if self._progress is not None and self._progress[0] == msg.get("id"):
                    self._progress = (msg["id"], msg["phase"], self._progress[2])
            elif kind == "corrected" or (kind == "error" and msg.get("where") == "correct"):
                self._busy = None
                started = self._progress[2] if self._progress else now
                self._progress = None
                self._correction = {**msg, "at": now, "wall_s": now - started}
                if kind == "corrected" and msg.get("applied") and msg.get("apply", True) and self.ar is not None:
                    apply = msg
                elif kind == "corrected" and msg.get("auto"):
                    # not applied: no new automatic try before two more key frames
                    self._auto_after = (self._last or {}).get("keys", 0) + 2
            elif kind in ("reset", "config"):
                self._busy = None
            elif kind == "error":
                if msg.get("where") == "frame":
                    self._frames_out = max(0, self._frames_out - 1)
                else:
                    self._busy = None
                logger.info("registration check: %s failed: %s", msg.get("where"), msg.get("what"))
            self._cv.notify_all()
        if apply is not None:
            self._applied = apply
            try:
                self.ar.set_corrected(apply["T_M_B"], {"method": apply["method"], "tre95_mm": apply["tre95_mm"]})
            except Exception as e:  # noqa: BLE001  the registration stays as it was
                logger.warning("registration check: correction not set: %s", e)
                self._applied = None

    def _ask(self, method, mode, apply, auto=False):
        cid = self._next_id
        self._next_id += 1
        self._want_correct = {"id": cid, "method": method or self.s["method"], "mode": mode or self.s["mode"],
                              "T_M_B": self._registration(), "tau_mm": TAU_MM, "u_mm": U_MM, "apply": apply,
                              "auto": auto}
        self._cv.notify_all()
        return cid

    def _registration(self):
        p = self._pending
        if p is not None:
            return p[4]
        if self.ar is not None and self.ar._setup is not None:
            return self.ar._setup["T_M_B"]
        return None

    def _stop_proc(self):
        for c in (self._cmd, self._res):
            try:
                if c is not None:
                    c.close()
            except OSError:
                pass
        self._cmd = self._res = None
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.wait(3.0)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                    p.wait(5.0)
                except (OSError, subprocess.TimeoutExpired):    # gone on its own later; nothing waits for it
                    logger.info("registration check: the worker did not end in time")
        with self._lock:
            self._frames_out, self._busy = 0, None
            self._progress = None
            if self._worker == "ready":
                self._worker = "starting"

    # what the page asks
    def correct(self, method=None, mode=None, apply=True):
        """Asks for a correction now (False when the worker is not ready, one is under way or the board is
        damaged). Returns the id of the correction or False."""
        if method is not None and method not in METHODS or mode is not None and mode not in MODES:
            raise ValueError(f"method {method}, mode {mode}")
        with self._cv:
            if (self._worker != "ready" or self._want_correct is not None or self._progress is not None
                    or self._registration() is None or self.board.state(time.perf_counter())["damaged"]):
                return False
            return self._ask(method, mode, apply)

    def undo(self):
        """The registration before the last correction applied, again."""
        if self._applied is None or self.ar is None:
            return False
        self._applied = None
        self.ar.set_corrected(None)
        return True

    def forget(self):
        """A correction no longer in use (a new registration was loaded)."""
        self._applied = None

    def configure(self, **kw):
        """method, mode, auto (and badge: the state drawn in the picture, for the phone's own screen)."""
        if "badge" in kw:
            self._badge = bool(kw.pop("badge"))
        bad = [k for k in kw if k not in ("method", "mode", "auto")]
        if bad or kw.get("method", "net") not in METHODS or kw.get("mode", "a1") not in MODES:
            raise ValueError(f"settings {kw}")
        if "auto" in kw:
            kw["auto"] = bool(kw["auto"])
        with self._cv:
            self.s.update(kw)       # method and mode go with each frame and correction
            self._auto_after = 0
        return dict(self.s)

    def reset(self):
        """Key frames and scores dropped (a new impact, a new patient)."""
        with self._cv:
            self._want_reset = self._worker == "ready"
            self._last = None
            self._cv.notify_all()
        self.board.clear()

    def status(self):
        now = time.perf_counter()
        with self._lock:
            last, corr, prog = self._last, self._correction, self._progress
            worker, info, why = self._worker, dict(self._worker_info), self._why
            times = list(self._det_times)
            tms = list(self._tm)
            active, applied = self._active, self._applied
            asked = self._want_correct is not None
            s = dict(self.s)
        board = self.board.state(now)
        shown = self._shown_state()
        st = {"active": active, "state": shown[0] if shown else "off", "worker": worker,
              "network": bool(info.get("net")), "device": info.get("device"), "why": why,
              "method": s["method"], "mode": s["mode"], "auto": bool(s["auto"]), "tau_mm": TAU_MM, "u_mm": U_MM,
              "enough_mm": ENOUGH_MM, "board": board, "corrected": applied is not None,
              "correcting": None if prog is None and not asked else (prog[1] if prog else "views"),
              "correcting_s": None if prog is None else round(now - prog[2], 1)}
        recent = [t for t in times if now - t <= 5.0]
        st["hz"] = round((len(recent) - 1) / (recent[-1] - recent[0]), 2) if len(recent) > 2 else 0.0
        if tms:
            keys = sorted({k for t in tms for k in t})
            st["times_ms"] = {k: round(float(np.median([t[k] for t in tms if k in t])), 1) for k in keys}
        if last is not None and now - last["at"] <= DETECT["stale_s"]:
            g = last.get("guide") or {}
            st.update(score_mm=_r(last.get("score_mm")), median_mm=_r(last.get("median_mm")),
                      keys=last.get("keys", 0), views=g.get("views"), pred95_mm=_r(g.get("pred95_mm")),
                      enough=g.get("enough"), arrow=shown is not None and shown[1] is not None)
        if corr is not None:
            st["correction"] = {"id": corr.get("id"), "applied": bool(corr.get("applied")),
                                "method": corr.get("method"), "mode": corr.get("mode"), "why": corr.get("why", ""),
                                "views": corr.get("views"), "tre95_mm": _r(corr.get("tre95_mm")),
                                "score_mm": _r(corr.get("score_mm")), "moved_mm": _r(corr.get("moved_mm")),
                                "agree_mm": _r(corr.get("agree_mm")), "wall_s": _r(corr.get("wall_s")),
                                "time_s": {k: _r(v, 3) for k, v in (corr.get("time_s") or {}).items()},
                                "age_s": round(now - corr["at"], 1), "error": corr.get("what") if
                                corr.get("type") == "error" else ""}
        return st

    def close(self):
        self._quit.set()
        with self._cv:
            self._cv.notify_all()
        try:
            if self._cmd is not None:
                self._cmd.send({"type": "quit"})
        except (EOFError, OSError):
            pass
        for th in self._threads:
            th.join(5.0)
        self._stop_proc()

    # synchronous use (bench, tests)
    def wait_ready(self, timeout=START_S):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            with self._lock:
                if self._worker in ("ready", "failed"):
                    return self._worker == "ready"
            time.sleep(0.05)
        return False

    def wait_idle(self, timeout=30.0):
        t0 = time.perf_counter()
        with self._cv:
            return self._cv.wait_for(lambda: self._frames_out == 0 and self._busy is None and self._want_correct is None
                                     and self._progress is None, timeout)


def _r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ── a synthetic stream with a face ──────────────────────────────────────────
def twin_path(session="s0000", registration=0):
    """(T_C_B of the video sweep of the twin, in order, the registration T_M_B of that sweep) or None."""
    p = TWIN / session / "frames.jsonl"
    if not p.is_file():
        return None
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["mode"] == "video" and r["registration"] == registration and r["state"] == 0]
    if not rows:
        return None
    rows.sort(key=lambda r: r["arm_deg"])
    return [np.array(r["T_C_B"], np.float64) for r in rows], np.array(rows[0]["T_M_B"], np.float64)


def a1_shift(c, deg, axis=(1.0, 0.4, -0.3)):
    """The displacement of interface A: a rotation of deg about an axis through the ball centre c (frame B)."""
    a = np.asarray(axis, np.float64)
    return vg.turn(a / np.linalg.norm(a), np.asarray(c, np.float64), deg)


def _interp(T0, T1, s):
    """Pose between T0 and T1 at s in [0, 1] (rotation along the geodesic, translation linear)."""
    R = T0[:3, :3] @ co.rodrigues(s * co.rotvec(T0[:3, :3].T @ T1[:3, :3]))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, (1 - s) * T0[:3, 3] + s * T1[:3, 3]
    return T


_FACE_STREAM = None


def face_stream_class():
    """FaceStream, made at the first use (it subclasses live_ar.SyntheticStream)."""
    global _FACE_STREAM
    if _FACE_STREAM is not None:
        return _FACE_STREAM
    import live_ar

    class FaceStream(live_ar.SyntheticStream):
        """The synthetic stream with the patient: the board on a grey table and the phantom face in a skin colour
        with a light at the camera, filmed along the arm's sweep of the digital twin, back and forth every
        `period` s (the camera from the head T_C_B of the sweep; the board where the true registration T_M_B puts
        it). bend_deg folds the right half of the board towards the camera (a damaged board). truth(n) is the true
        T_C_M of frame n."""

        THREAD = "face_stream"

        def __init__(self, board, spec, K, size, fps, T_M_B, path, period=12.0, bend_deg=0.0, seed=0, noise=2.0,
                     colour=(118, 152, 214)):
            super().__init__(board, spec, K, size, fps, seed=seed, noise=noise)
            self.T_M_B, self.path = np.asarray(T_M_B, np.float64), [np.asarray(T, np.float64) for T in path]
            self.period, self.bend, self.colour = float(period), float(bend_deg), np.array(colour, np.float32)
            m = np.load(FACE_MESH)
            self._fv, self._ff = m["vertices"].astype(np.float64), m["faces"].astype(np.int64)
            self._ren = None
            self.t_scale = 1.0
            nx, ny = spec["squares"]
            self._W = nx * spec["square_mm"]

        def T_C_B(self, t):
            u = (t * self.t_scale / self.period) % 2.0
            u = u if u <= 1.0 else 2.0 - u                   # back and forth
            x = u * (len(self.path) - 1)
            i = min(int(x), len(self.path) - 2)
            return _interp(self.path[i], self.path[i + 1], x - i)

        def pose(self, t):
            return self.T_C_B(t) @ np.linalg.inv(self.T_M_B)

        def _board(self, T):
            img = super()._board(T)
            if not self.bend or self._tex is None:
                return img
            b = np.radians(self.bend)
            F = np.eye(4)
            F[:3, :3] = [[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]
            p = np.array([self._W / 2, 0, 0])
            F[:3, 3] = p - F[:3, :3] @ p
            Tf = T @ F
            H = self.K @ np.column_stack([Tf[:3, 0], Tf[:3, 1], Tf[:3, 3]]) @ self._S
            col = (self._W / 2 - self._S[0, 2]) / self._S[0, 0]        # texture column of the fold
            right = np.zeros(self._tex.shape, np.uint8)
            right[:, int(np.ceil(col)):] = 255
            m = cv2.warpPerspective(right, H, self.size, flags=cv2.INTER_NEAREST)
            folded = cv2.warpPerspective(self._tex, H, self.size, flags=cv2.INTER_LINEAR)
            np.copyto(img, folded, where=m > 0)
            return img

        def render(self, t, k=0):
            import gl_overlay
            if self._ren is None:
                self._ctx = gl_overlay.standalone_context()
                self._ren = gl_overlay.GLMeshRenderer(self._fv, self._ff, ctx=self._ctx)
            T_C_B = self.T_C_B(t)
            T = T_C_B @ np.linalg.inv(self.T_M_B)
            img = cv2.cvtColor(self._board(T), cv2.COLOR_GRAY2BGR)
            light = 1.0 - 0.55 * np.clip(1.0 - self._ren.facing(T_C_B), 0, 1) ** 0.6
            o = self._ren.overlay(self.K, np.zeros(5), T_C_B, self.size, light)
            bh, bw = o.cover.shape
            if bh and bw:
                x0, y0 = o.offset
                roi = img[y0:y0 + bh, x0:x0 + bw]
                wgt = o.cover.astype(np.float32) / 255
                lit = cv2.divide(o.light, o.cover, scale=255)
                tint = cv2.multiply(cv2.cvtColor(lit, cv2.COLOR_GRAY2BGR), tuple(float(c) / 255 for c in self.colour) + (0,))
                roi[:] = cv2.blendLinear(roi, tint, 1.0 - wgt, wgt)
            if not hasattr(self, "_noise3"):
                self._noise3 = [(cv2.merge([a] * 3), cv2.merge([b] * 3)) for a, b in self._noise]
            plus, minus = self._noise3[k % len(self._noise3)]
            return cv2.subtract(cv2.add(img, plus), minus), T

        def _run(self):
            t0 = time.perf_counter()
            period = 1.0 / self.fps
            nxt = t0
            try:
                while not self._stop.is_set():
                    img, T = self.render(time.perf_counter() - t0, self._n)
                    n = self._put(img)
                    with self._lock:
                        self._truth[n] = T
                        self._truth.pop(n - 3000, None)
                    nxt += period
                    live_ar._pause(self._stop, nxt - time.perf_counter())
            finally:
                if self._ren is not None:
                    self._ren.release()
                    self._ctx.release()
                    self._ren = None

    _FACE_STREAM = FaceStream
    return FaceStream


def face_stream_factory(T_M_B=None, shift_deg=0.0, period=12.0, bend_deg=0.0, session="s0000"):
    """A factory for LiveAR.synthetic_stream: the face stream on the patient board, the face at T_M_B (default: the
    registration of the live view's setup, which the demo makes the bench design's) moved by an A1 shift of
    shift_deg; the plain synthetic stream when the twin's sweep is not on this computer or the board is not the
    patient's."""
    got = twin_path(session)

    def make(ar):
        setup = ar._setup
        w, h = ar._size_wh()
        import live_ar
        K, _, _ = live_ar.stream_intrinsics((w, h), ar.intrinsics_path)
        if got is None or setup["spec"]["name"] != "patient" or not FACE_MESH.is_file():
            return ar._synthetic_stream()
        T = np.asarray(T_M_B if T_M_B is not None else setup["T_M_B"], np.float64)
        if shift_deg:
            c = load_anatomy(occluders=False).c
            T = T @ a1_shift(c, shift_deg)
        return face_stream_class()(setup["board"], setup["spec"], K, (w, h), ar.fps, T, got[0], period, bend_deg)
    return make


# ── the digital twin: threshold and evaluation ──────────────────────────────
def twin_states(split):
    """{(session, registration, state): [frame rows of the video sweep, by arm angle]} of the sessions of a split."""
    import synth_data as sd
    out = {}
    for s in sd.sessions(WORK / "synth"):
        if int(s.name[1:]) >= sd.CHECK_FROM:
            continue
        for fr in sd.frames(s):
            if fr["split"] == split and fr["mode"] == "video":
                out.setdefault((s.name, fr["registration"], fr["state"]), []).append(fr)
    return {k: sorted(v, key=lambda r: r["arm_deg"]) for k, v in sorted(out.items())}


_BOARD_POSE = {}


def board_pose(img, K, dist):
    """T_C_M of the patient board found in an image (ChArUco and PnP), or None."""
    import marker_pose as mp
    key = (round(float(K[0, 0]), 3), img.shape)
    if key not in _BOARD_POSE:
        b = mp.make_board(mp.BOARD)
        _BOARD_POSE[key] = (b, mp.make_detector(b, K, dist))
    b, det = _BOARD_POSE[key]
    corners, ids, _ = mp.detect(det, img)
    got = mp.estimate_pose(b, corners, ids, K, dist, min_corners=8)
    return None if got is None or got["rms_px"] > 1.0 else got["T"]


def run_sweep(w, rows, true_pose=False, guide=True, trims=None):
    """The frames of one sweep of the twin through the worker, with the registration before the impact (T_M_B0):
    one answer per frame, as in the live view."""
    w.handle({"type": "reset"})
    a = w.a
    pts = a.v[np.linspace(0, len(a.v) - 1, 300).astype(int)]
    out = []
    for i, fr in enumerate(rows):
        img = cv2.imread(str(TWIN / f"s{fr['session']:04d}" / (fr["image"] + ".jpg")), cv2.IMREAD_COLOR)
        K, dist = np.array(fr["K"], np.float64), np.array(fr["dist"], np.float64)
        T_C_M = np.array(fr["T_C_M"], np.float64) if true_pose else board_pose(img, K, dist)
        if T_C_M is None:
            continue
        mode = "a1" if fr["interface"] == "a1" else "rigid"
        msg = frame_message(img, K, dist, T_C_M, np.array(fr["T_M_B0"], np.float64), a.o, pts, i, t=float(i),
                            mode=mode)
        if msg is None:
            continue
        msg["guide"] = guide
        if trims:
            msg["trims"] = trims
        r = w.handle(msg)
        r["frame"] = fr["image"]
        out.append(r)
    return out


TRIMS = (0.0, 0.1, 0.2, 0.3)


def cmd_tau(a):
    """The trim and TAU_MM from the validation sweeps: per trim of the score, the decisions (median of
    DETECT["median_of"] views) against the label (mean target error above 2 mm); the trim with the largest AUC, and
    for it the largest threshold that still finds a lost registration in at least 95 % of the decisions, as decided
    for the detection; the 95th percentile before the impact and the Youden threshold beside it."""
    w = Worker({"method": "opt", "segmenter": a.segmenter})
    rows = []
    t0 = time.perf_counter()
    for key, frs in twin_states(a.split).items():
        res = [r for r in run_sweep(w, frs, guide=False, trims=TRIMS) if r.get("trim_scores")]
        lost = frs[0]["target_shift_mm"]["mean"] > 2.0
        for i, r in enumerate(res):
            row = {"state": "/".join(map(str, key)), "lost": lost, "tre_mm": frs[0]["target_shift_mm"]["mean"],
                   "drape": bool(frs[0].get("drape")), "i": i, "pre": frs[0]["state"] == 0,
                   "ms": round(r["ms"]["total_ms"], 1), "sam_ms": round(r["ms"].get("sam_ms", 0), 1)}
            for q in TRIMS:
                vals = [x["trim_scores"][str(q)] for x in res[max(0, i - DETECT["median_of"] + 1):i + 1]]
                row[f"view_{q}"] = r["trim_scores"][str(q)]
                row[f"median_{q}"] = float(np.median(vals))
            rows.append(row)
        print(f"{key}: {len(res)} views, lost {lost}, {time.perf_counter() - t0:.0f} s", flush=True)
    out = WORK / "live_check"
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(out / f"tau_{a.split}.csv", rows)
    return tau_summary(rows)


def tau_summary(rows):
    full = [r for r in rows if r["i"] >= DETECT["median_of"] - 1]
    y = np.array([bool(r["lost"]) for r in full])
    best = None
    for q in TRIMS:
        s = np.array([r[f"median_{q}"] for r in full])
        tau = float(np.percentile(s[y], 5)) if y.any() else float("nan")
        pre = np.array([r[f"view_{q}"] for r in rows if r["pre"]])
        grid = np.unique(s)
        youden = max(grid, key=lambda t: (s[y] > t).mean() + (s[~y] <= t).mean())
        d = sc.detection(s, y, tau)
        line = {"trim": q, "tau_sens95_mm": round(tau, 3), "pre_p95_mm": round(float(np.percentile(pre, 95)), 3),
                "youden_mm": round(float(youden), 3), **{k: round(v, 3) if isinstance(v, float) else v
                                                         for k, v in d.items()}}
        print(json.dumps(line))
        if best is None or d["auc"] > best["auc"]:
            best = line
    print("chosen:", json.dumps(best))
    return 0


def _write_rows(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, keys)
        wr.writeheader()
        wr.writerows(rows)


def cmd_twin(a):
    """Detection and correction on the twin's sweeps of a split, through the worker as in the live view: per frame
    the score and the state with the registration before the impact; after the sweep the correction from the same
    key frames with every method (and the network with 1, 2 and 4 views), against the true registration."""
    out = WORK / "live_check"
    if a.summary:                       # the tables again from the rows of the last run
        det, cor = (_read_rows(out / f"twin_{k}_{a.split}.csv") for k in ("detect", "correct"))
        summary, timing = summarize_twin(det, cor), timing_rows(det, cor)
        print(json.dumps(summary + timing, indent=1))
        if a.results:
            _write_rows(RESULTS / "live_twin.csv", summary)
            _write_rows(RESULTS / "live_timing.csv", timing)
        return summary
    w = Worker({"method": "net", "segmenter": a.segmenter})
    X = w.a.targets
    det, cor = [], []
    t0 = time.perf_counter()
    states = twin_states(a.split)
    keys = list(states)[:a.states] if a.states else list(states)
    for si, key in enumerate(keys):
        frs = states[key]
        fr = frs[0]
        T_true, T0 = np.array(fr["T_M_B"], np.float64), np.array(fr["T_M_B0"], np.float64)
        mode = "a1" if fr["interface"] == "a1" else "rigid"
        lost = fr["target_shift_mm"]["mean"] > 2.0
        res = run_sweep(w, frs, true_pose=a.true_pose)
        base = {"state": "/".join(map(str, key)), "mode": mode, "lost": int(lost),
                "tre_before_mm": round(mean_target_mm(T_true, T0, X), 4), "drape": int(bool(fr.get("drape")))}
        first_enough = next((i for i, r in enumerate(res) if (r.get("guide") or {}).get("enough")), None)
        for i, r in enumerate(res):
            g = r.get("guide") or {}
            det.append({**base, "i": i, "frame": r.get("frame"), "score_mm": r.get("score_mm"),
                        "median_mm": r.get("median_mm"), "decision": r.get("state"), "keys": r.get("keys"),
                        "views": g.get("views"), "pred95_mm": g.get("pred95_mm"), "enough": g.get("enough"),
                        "arrow": int(g.get("arrow") is not None), **{k: round(v, 1) for k, v in r["ms"].items()}})
        if fr["state"] == 0 and not a.pre:
            print(f"{si + 1}/{len(keys)} {key}: before the impact, state {res[-1]['state'] if res else '-'}", flush=True)
            continue
        runs = [(m, KEY["views"]) for m in METHODS] + [("net", 1), ("net", 2)]
        for m, kv in runs:
            try:
                out = w.handle({"type": "correct", "T_M_B": T0, "mode": mode, "method": m, "views": kv})
            except Exception as e:  # noqa: BLE001  this run said, the others go on
                out = {"type": "error", "what": f"{type(e).__name__}: {e}"}
            if out.get("type") != "corrected" or out.get("T_M_B") is None:
                cor.append({**base, "method": m, "views_asked": kv, "error": out.get("what") or out.get("why")})
                continue
            row = {**base, "method": out["method"], "views_asked": kv, "views": out["views"],
                   "tre_after_mm": round(mean_target_mm(T_true, out["T_M_B"], X), 4), "applied": int(out["applied"]),
                   "why": out["why"], "tre95_mm": round(out["tre95_mm"], 4), "score_mm": round(out["score_mm"], 4),
                   "agree_mm": round(out.get("agree_mm", float("nan")), 4), "enough_at": first_enough,
                   "pred95_views_mm": round(out["pred95_views_mm"], 4),
                   **{k: round(v, 3) for k, v in out["time_s"].items()}}
            if out.get("T_M_B_opt") is not None:
                row["tre_check_mm"] = round(mean_target_mm(T_true, out["T_M_B_opt"], X), 4)
            cor.append(row)
        done = [r for r in cor if r["state"] == base["state"] and "tre_after_mm" in r]
        print(f"{si + 1}/{len(keys)} {key} {mode}: TRE {base['tre_before_mm']:.2f} mm -> "
              + ", ".join(f"{r['method']}/{r['views']} {r['tre_after_mm']:.2f}{'' if r['applied'] else '!'}"
                          for r in done) + f", {time.perf_counter() - t0:.0f} s", flush=True)
    out = WORK / "live_check"
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(out / f"twin_detect_{a.split}.csv", det)
    _write_rows(out / f"twin_correct_{a.split}.csv", cor)
    summary, timing = summarize_twin(det, cor), timing_rows(det, cor)
    print(json.dumps(summary + timing, indent=1))
    if a.results:
        _write_rows(RESULTS / "live_twin.csv", summary)
        _write_rows(RESULTS / "live_timing.csv", timing)
    return summary


def _read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k, v in r.items():
                if v in ("", None):
                    r[k] = None
                    continue
                try:
                    r[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                except ValueError:
                    r[k] = None if v == "None" else {"True": True, "False": False}.get(v, v)
            rows.append(r)
    return rows


def timing_rows(det, cor):
    """Times per stage of the detection (one frame, ms) and of the correction (s), median and 95th percentile."""
    rows = []
    for k in ("box_ms", "sam_ms", "masks_ms", "score_ms", "guide_ms", "total_ms"):
        v = [r[k] for r in det if r.get(k) is not None]
        if v:
            rows.append({"key": f"detect_{k[:-3]}", "median": round(float(np.median(v)), 1),
                         "p95": round(float(np.percentile(v, 95)), 1), "max": round(float(np.max(v)), 1),
                         "unit": "ms", "n": len(v)})
    for m in METHODS:
        g = [r for r in cor if r.get("method") == m and r.get("views_asked") == KEY["views"] and r.get("total_s")]
        for k in ("views_s", "net_s", "opt_s", "check_s", "total_s"):
            v = [r[k] for r in g if r.get(k) is not None]
            if v:
                rows.append({"key": f"correct_{m}_{k[:-2]}", "median": round(float(np.median(v)), 3),
                             "p95": round(float(np.percentile(v, 95)), 3), "max": round(float(np.max(v)), 3),
                             "unit": "s", "n": len(v)})
    return rows


def summarize_twin(det, cor):
    rows = []
    d = [r for r in det if r["score_mm"] is not None]
    s, y = np.array([r["score_mm"] for r in d], float), np.array([r["lost"] for r in d], bool)
    if len(d):
        m = sc.detection(s, y, TAU_MM)
        rows.append({"key": "detect_view", "n": len(d), "tau_mm": TAU_MM, "auc": round(m["auc"], 3),
                     "sensitivity": round(m["sensitivity"], 3), "specificity": round(m["specificity"], 3)})
    dm = [r for r in d if r["median_mm"] is not None and r["i"] >= DETECT["median_of"] - 1]
    if dm:
        s, y = np.array([r["median_mm"] for r in dm], float), np.array([r["lost"] for r in dm], bool)
        m = sc.detection(s, y, TAU_MM)
        rows.append({"key": "detect_median", "n": len(dm), "tau_mm": TAU_MM, "auc": round(m["auc"], 3),
                     "sensitivity": round(m["sensitivity"], 3), "specificity": round(m["specificity"], 3)})
    for m_, kv in [(m, KEY["views"]) for m in METHODS] + [("net", 1), ("net", 2)]:
        g = [r for r in cor if r.get("method") == m_ and r.get("views_asked") == kv and "tre_after_mm" in r]
        if not g:
            continue
        ta = np.array([r["tre_after_mm"] for r in g])
        ap = np.array([r["applied"] for r in g], bool)
        ts = np.array([r["total_s"] for r in g])
        rows.append({"key": f"correct_{m_}_{kv}", "n": len(g),
                     "tre_before_median_mm": round(float(np.median([r["tre_before_mm"] for r in g])), 3),
                     "tre_after_median_mm": round(float(np.median(ta)), 3),
                     "tre_after_p95_mm": round(float(np.percentile(ta, 95)), 3),
                     "below_2mm": round(float(np.mean(ta < 2)), 3), "applied": round(float(ap.mean()), 3),
                     "silent": round(float(np.mean(ap & (ta >= 2))), 3),
                     "time_median_s": round(float(np.median(ts)), 2), "time_max_s": round(float(ts.max()), 2)})
    lost = {r["state"]: r for r in cor if r.get("lost") and r.get("method") == "net" and r.get("views_asked") == KEY["views"]}
    if lost:
        e = [r["enough_at"] for r in lost.values()]
        rows.append({"key": "views_enough", "n": len(e), "reached": round(float(np.mean([x is not None for x in e])), 3),
                     "frames_median": float(np.median([x + 1 for x in e if x is not None])) if any(
                         x is not None for x in e) else None})
    return rows


# ── the live view with the check: bench and tests ───────────────────────────
def live_with_check(settings, shift_deg=3.0, bend_deg=0.0, period=12.0, check=True, intrinsics=None):
    """A live view on the synthetic stream with the face (the patient board, demo registration of the bench design;
    the face drawn moved by an A1 shift of shift_deg, so the registration in use is stale by it), with the check
    attached (unless check is False). Returns (ar, check or None, the true registration)."""
    import live_ar
    ar = live_ar.LiveAR(board="patient", synthetic=True, intrinsics=intrinsics, prepare=True)
    T_design = ar._setup["T_M_B"]
    T_true = T_design @ a1_shift(load_anatomy(occluders=False).c, shift_deg) if shift_deg else T_design
    ar.synthetic_stream = face_stream_factory(T_M_B=T_true, period=period, bend_deg=bend_deg)
    chk = None
    if check:
        chk = RegistrationCheck(settings)
        chk.history = []
        ar.attach_check(chk)
    return ar, chk, T_true


def _wait(cond, timeout, step=0.05):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return cond()


def _rate(ar, seconds):
    """Frames composed per second over `seconds`."""
    t0, k0 = time.perf_counter(), ar._jpeg_n
    time.sleep(seconds)
    return (ar._jpeg_n - k0) / (time.perf_counter() - t0)


def cmd_bench(a):
    """Times on the synthetic stream with the face (SAM, the network) and the clips: see the module docstring."""
    rows = []
    if not a.clips:
        settings = {"segmenter": a.segmenter, "method": "net", "mode": "a1"}
        ar, _, T_true = live_with_check(settings, check=False)
        ar.start()
        time.sleep(2.0)
        fps_off = _rate(ar, min(a.seconds, 8.0))
        ar.shutdown()
        ar, chk, T_true = live_with_check(settings)
        ar.start()
        ok = chk.wait_ready()
        _wait(lambda: chk.status().get("state") == "lost", 30)
        time.sleep(1.0)
        fps_on = _rate(ar, a.seconds)
        st = chk.status()
        X = load_targets()
        T0 = ar._setup["T_M_B"]
        corr = []
        for m in METHODS:
            chk.wait_idle(10)
            cid = chk.correct(method=m, apply=False)
            _wait(lambda: (chk.status().get("correction") or {}).get("id") == cid, 30)
            c = dict(chk._correction or {})
            if c.get("T_M_B") is not None:
                corr.append({"method": m, "wall_s": c["wall_s"], **c.get("time_s", {}),
                             "tre_before_mm": mean_target_mm(T_true, T0, X),
                             "tre_after_mm": mean_target_mm(T_true, c["T_M_B"], X), "applied": c.get("applied")})
            print("correction", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                            for k, v in corr[-1].items()} if corr else c.get("what")), flush=True)
        ar.shutdown()
        det = [h for h in chk.history if h.get("type") == "detect" and h.get("ms")]
        ms = {k: [d["ms"][k] for d in det if k in d["ms"]] for k in ("box_ms", "sam_ms", "masks_ms", "score_ms",
                                                                       "guide_ms", "total_ms")}
        at = np.array([d["at"] for d in det])
        hz = (len(at) - 1) / (at[-1] - at[0]) if len(at) > 2 else 0.0
        print(json.dumps({"ready": ok, "fps_off": round(fps_off, 1), "fps_on": round(fps_on, 1), "hz": round(hz, 2),
                          "status": {k: st.get(k) for k in ("state", "hz", "times_ms", "keys", "views", "pred95_mm")}},
                         indent=1), flush=True)
        rows.append({"key": "video_fps_without", "value": round(fps_off, 1), "unit": "fps"})
        rows.append({"key": "video_fps_with", "value": round(fps_on, 1), "unit": "fps"})
        rows.append({"key": "detect_hz", "value": round(hz, 2), "unit": "Hz"})
        for k, v in ms.items():
            if v:
                rows.append({"key": f"detect_{k[:-3]}_median", "value": round(float(np.median(v)), 1), "unit": "ms"})
                rows.append({"key": f"detect_{k[:-3]}_p95", "value": round(float(np.percentile(v, 95)), 1),
                             "unit": "ms"})
        for c in corr:
            for k in ("views_s", "net_s", "opt_s", "check_s", "total_s", "wall_s"):
                if k in c:
                    rows.append({"key": f"correct_{c['method']}_{k[:-2]}", "value": round(c[k], 3), "unit": "s"})
            rows.append({"key": f"correct_{c['method']}_tre_synthetic", "value": round(c["tre_after_mm"], 3),
                         "unit": "mm"})
    for clip, board in (("regression_small.h264", "patient"), ("regression_big.h264", "sheet")):
        path = CLIPS / clip
        if not path.is_file():
            continue
        try:
            out = bench_clip(path, board, a.segmenter, min(a.seconds, 20.0))
        except Exception as e:  # noqa: BLE001  a clip that cannot be played: said, the rest written
            print(clip, f"not played: {type(e).__name__}: {e}", flush=True)
            continue
        print(clip, json.dumps(out), flush=True)
        for k, v in out.items():
            rows.append({"key": f"clip_{board}_{k}", "value": v, "unit": ""})
    if a.results:
        _write_rows(RESULTS / "live_rates.csv", rows)
    return 0


def bench_clip(path, board, segmenter, seconds):
    """A recorded clip through the live view, without and with the check: video rate and the board check (RMS and
    worst quadrant over the run; a clean board never damaged)."""
    import live_ar
    kw = {"squares": (14, 7)} if board == "sheet" else {}
    out = {}
    for on in (False, True):
        ar = live_ar.LiveAR(board=board, replay=path, prepare=True, **kw)
        chk = None
        if on:
            chk = RegistrationCheck({"segmenter": segmenter, "method": "opt", "boards": ("patient",)})
            ar.attach_check(chk)
        ar.start()
        time.sleep(1.5)
        boards, t0 = [], time.perf_counter()
        k0 = ar._jpeg_n
        while time.perf_counter() - t0 < seconds:
            time.sleep(0.25)
            if chk is not None:
                b = chk.board.state(time.perf_counter())
                if b["frames"] >= BOARD["min_frames"]:
                    boards.append(b)
        fps = (ar._jpeg_n - k0) / (time.perf_counter() - t0)
        st = chk.status() if chk is not None else {}
        ar.shutdown()
        out["fps_with" if on else "fps_without"] = round(fps, 1)
        if on:
            out["hz"] = st.get("hz")
            rms = [b["rms_px"] for b in boards if b["rms_px"] is not None]
            rat = [b["ratio"] for b in boards if b["ratio"] is not None]
            out.update(board_rms_p99=round(float(np.percentile(rms, 99)), 3) if rms else None,
                       board_ratio_p99=round(float(np.percentile(rat, 99)), 3) if rat else None,
                       board_damaged=round(float(np.mean([b["damaged"] for b in boards])), 3) if boards else None)
    return out


def run_test():
    """The board check, the turn of the picture, the worker on the synthetic face (aligned, stale, corrected, the
    guidance), then the live view with the check in its own process: lost, corrected, aligned again, the video rate
    kept, a bent board found. The colour segmenter and the optimisation: no GPU model needed."""
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    # the board check: exact corners, one quadrant off (a fold), all off
    import live_ar
    board, spec = live_ar.make_board("patient")
    obj = board.getChessboardCorners().astype(np.float64)
    K = np.array([[1439.0, 0, 959.5], [0, 1439.0, 719.5], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 3] = [-22.5, -22.5, 380.0]
    uv = cv2.projectPoints(obj @ T[:3, :3].T + T[:3, 3], np.zeros(3), np.zeros(3), K, np.zeros(5))[0].reshape(-1, 2)
    ids = np.arange(len(obj))
    got = []
    fold = np.where((obj[:, :1] > 22.5) & (obj[:, 1:2] > 22.5), np.array([[1.6, -0.9]]), 0.0)
    for off in (np.zeros_like(uv), fold, np.full_like(uv, 0.75)):
        bc = BoardCheck()
        rng = np.random.default_rng(1)
        for i in range(20):
            bc.add(i * 0.03, uv + off + rng.normal(0, 0.08, uv.shape), ids, T, obj, K, None, spec, 1.0)
        got.append(bc.state(0.6)["damaged"])
    expect(got == [False, True, True], f"board check: clean, folded quadrant, everything off: damaged {got}")

    # the turn of the picture: a point and a direction as cv2.rotate turns the image
    good = True
    for k in (1, 2, 3):
        rot = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)[k - 1]
        found = []
        for x in (10, 11):
            img = np.zeros((40, 60), np.uint8)
            img[5, x] = 255
            yy, xx = np.argwhere(cv2.rotate(img, rot) == 255)[0]
            found.append(np.array([xx, yy], float))
        q, dd = RegistrationCheck._turn(np.array([10.0, 5.0]), np.array([1.0, 0.0]), k, (60, 40))
        good &= bool(np.allclose(q, found[0]) and np.allclose(dd, found[1] - found[0]))
    expect(bool(good), "the arrow follows the picture through its quarter turns")

    got = twin_path()
    if got is None or not FACE_MESH.is_file():
        print(f"skip  the twin's sweep or the face is not on this computer ({TWIN})", flush=True)
        return 1 if fails else 0
    path, T_twin = got
    # the worker, in this process, on frames of the synthetic face
    w = Worker({"segmenter": "colour", "method": "opt", "mode": "a1"})
    Kn = np.array([[1439.0, 0, 959.5], [0, 1439.0, 719.5], [0, 0, 1]])
    FS = face_stream_class()
    T_true = T_twin
    T_stale = T_true @ np.linalg.inv(a1_shift(w.a.c, 3.0))
    stream = FS(board, spec, Kn, (1920, 1440), 30, T_true, path, period=8.0)
    pts = w.a.v[np.linspace(0, len(w.a.v) - 1, 300).astype(int)]
    res = []
    for reg, want in ((T_true, "ok"), (T_stale, "lost")):
        w.handle({"type": "reset"})
        res = []
        for i, t in enumerate(np.linspace(0, 7.5, 10)):
            img, T_C_M = stream.render(t, i)
            msg = frame_message(img, Kn, np.zeros(5), T_C_M, reg, w.a.o, pts, i, float(t), mode="a1")
            res.append(w.handle(msg))
        states = [r["state"] for r in res]
        scores = [round(r.get("score_mm") or float("nan"), 2) for r in res]
        expect(states[-1] == want and states.count(want) >= 7,
               f"worker: the {'true' if want == 'ok' else 'stale (3 deg)'} registration is {want} "
               f"({states.count(want)} of 10, scores {scores[:5]}... mm, tau {TAU_MM}), {res[-1].get('keys')} key "
               "frames")
    g = res[-1].get("guide") or {}
    expect(g.get("views", 0) >= 1 and np.isfinite(g.get("pred95_mm", np.inf)),
           f"guidance while lost: {g.get('views')} views, predicted 95 % error {g.get('pred95_mm')} mm, arrow "
           f"{g.get('arrow')}")
    out = w.handle({"type": "correct", "T_M_B": T_stale, "mode": "a1", "method": "opt"})
    X = w.a.targets
    tre0, tre1 = mean_target_mm(T_true, T_stale, X), mean_target_mm(T_true, out.get("T_M_B", T_stale), X)
    expect(out.get("type") == "corrected" and tre1 < 0.3 * tre0 and tre1 < 0.6 and out["time_s"]["total_s"] < 10,
           f"correction from {out.get('views')} key frames: target error {tre0:.2f} -> {tre1:.2f} mm, applied "
           f"{out.get('applied')} ({out.get('why')}), {out.get('time_s', {}).get('total_s', 0):.1f} s")
    if stream._ren is not None:
        stream._ren.release()
        stream._ctx.release()

    # the live view with the check in its own process
    settings = {"segmenter": "colour", "method": "opt", "mode": "a1"}
    ar, _, _ = live_with_check(settings, check=False, period=8.0)
    ar.start()
    time.sleep(1.5)
    fps_off = _rate(ar, 4.0)
    ar.shutdown()
    ar, chk, T_true = live_with_check(settings, period=8.0)
    try:
        ar.start()
        ready = chk.wait_ready(START_S)
        lost = _wait(lambda: chk.status().get("state") == "lost", 60)
        fps_on = _rate(ar, 4.0)
        s1 = chk.status()
        # functional here (a loaded machine slows both); the rates themselves are measured by bench
        expect(ready and lost and s1["hz"] >= 1.0 and fps_on >= 0.6 * fps_off,
               f"live view: the stale registration found lost ({s1.get('median_mm')} mm), detection "
               f"{s1.get('hz')} Hz, video {fps_on:.1f} fps with the check, {fps_off:.1f} without")
        chk.configure(badge=True)
        time.sleep(0.5)
        frame = ar.composed()
        R = DRAW["badge_px"]
        h = frame.shape[0]
        patch = frame[int(h - 2 * R - 12):int(h - 12), 12:int(12 + 2 * R)].reshape(-1, 3).astype(int)
        lostc = np.array(bgr(COLOURS["lost"]))
        expect(np.mean(np.abs(patch - lostc).sum(1) < 30) > 0.3, "the state drawn in the picture for the phone")
        chk.wait_idle(10)
        cid = chk.correct()
        done = _wait(lambda: (chk.status().get("correction") or {}).get("id") == cid, 60)
        c = chk.status().get("correction") or {}
        applied = _wait(lambda: ar.status()["registration"]["source"] == "corrected", 5)
        X = load_targets()
        tre = mean_target_mm(T_true, ar._setup["T_M_B"], X)
        expect(bool(done) and c.get("applied") and applied and tre < 0.6 and c.get("wall_s", 99) < 20,
               f"live view: corrected from {c.get('views')} views in {c.get('wall_s')} s, applied, target error "
               f"{tre:.2f} mm")
        ok_again = _wait(lambda: chk.status().get("state") == "ok", 40)
        expect(bool(ok_again), f"live view: aligned again after the correction ({chk.status().get('median_mm')} mm)")
        expect(not chk.status()["board"]["damaged"], "live view: the board whole")
        chk.undo()
        back = _wait(lambda: ar.status()["registration"]["source"] == "demo", 5)
        expect(bool(back), "live view: undo goes back to the registration before the correction")
    finally:
        ar.shutdown()
    ar, chk, _ = live_with_check(settings, shift_deg=0.0, bend_deg=12.0, period=8.0)
    try:
        ar.start()
        bent = _wait(lambda: chk.status()["board"]["damaged"], 30)
        b = chk.status()["board"]
        expect(bool(bent) and chk.correct() is False,
               f"a bent board (12 deg) found damaged (RMS {b.get('rms_px')} px, worst quadrant {b.get('ratio')} x), "
               "no correction while it is")
    finally:
        ar.shutdown()
    print(f"{'FAILED ' + str(len(fails)) if fails else 'all passed'}", flush=True)
    return 1 if fails else 0


# ── command line ────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("address")
    t = sub.add_parser("tau")
    t.add_argument("--split", default="val")
    t.add_argument("--segmenter", default="sam", choices=SEGMENTERS)
    tw = sub.add_parser("twin")
    tw.add_argument("--split", default="test")
    tw.add_argument("--segmenter", default="sam", choices=SEGMENTERS)
    tw.add_argument("--states", type=int, default=0, help="only the first N states")
    tw.add_argument("--pre", action="store_true", help="also correct the states before an impact")
    tw.add_argument("--true-pose", action="store_true", help="the true board pose instead of the one found")
    tw.add_argument("--results", action="store_true", help="the summaries into 05_results (live_twin, live_timing)")
    tw.add_argument("--summary", action="store_true", help="only the summaries, from the rows of the last run")
    b = sub.add_parser("bench")
    b.add_argument("--seconds", type=float, default=20.0)
    b.add_argument("--segmenter", default="sam", choices=SEGMENTERS)
    b.add_argument("--results", action="store_true", help="the rates into 05_results/live_rates.csv")
    b.add_argument("--clips", action="store_true", help="only the recorded clips")
    sub.add_parser("test")
    a = p.parse_args(argv)
    if a.cmd == "worker":
        key = bytes.fromhex(os.environ.get("IMPACTGUIDE_CHECK_KEY", ""))
        return serve(a.address, key)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.cmd == "tau":
        return cmd_tau(a)
    if a.cmd == "twin":
        cmd_twin(a)
        return 0
    if a.cmd == "bench":
        return cmd_bench(a)
    return run_test()


if __name__ == "__main__":
    sys.exit(main())
