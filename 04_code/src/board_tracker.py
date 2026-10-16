"""Board tracking for the live AR (T20.3): the ChArUco corners followed from frame to frame, the full detection only
now and then.

Finding the ChArUco markers in a frame costs 5 to 60 ms and fails on a blurred or partly covered frame; following
corners already found costs about a millisecond. So the live AR follows the corners on every frame of the stream and
runs the detection in a worker thread, every TRACK["detect_every_s"] or as soon as the tracking weakens:

  Detector   the markers in a box around the board where it was (at full resolution for a small board, scaled so that
             the markers are about TRACK["marker_px"] wide for a large one) or, when it is lost, in the whole frame;
             adaptive threshold windows sized to the markers, a lower minimum perimeter, sub-pixel marker corners; then
             the chessboard corners interpolated and refined by OpenCV's ChArUco detector.
  Tracker    each frame: the corners of the previous frame moved by pyramidal Lucas-Kanade optical flow (on the box
             around the board, on the image halved until a square is about TRACK["klt_square_px"] wide; started where
             their speed takes them, with a wider window and a deeper pyramid when they move fast, and again from where
             they were when the motion turns; forward and backward, a corner that does not come back to itself is
             dropped), then put back on the true saddle point of the chessboard at full resolution (cornerSubPix), so
             they never drift; the pose by Levenberg-Marquardt from the pose of the previous frame, corners far from
             the pose dropped (and RANSAC when many are), accepted under the reprojection limit. A detection comes
             back a few frames late: its corners are carried to the current frame by the same flow (from the frame it
             was made on) and re-anchor the ones that drifted (by more than TRACK["reanchor_px"]) or were lost; the
             others stay as tracked, so a detection makes no twitch.
  ambiguity  a small board seen nearly face on has two poses (IPPE) that fit almost equally well, mirrored about the
             line of sight; on the face 15 cm away they can differ by centimetres. The tracker keeps the one it has
             (the pose of each frame starts from the previous one), weighs the two on every frame (log ratio of their
             errors, with forgetting) and changes only on strong evidence; at a new acquisition the pose counts as sure
             only once the two hypotheses agree on the overlay (within TRACK["ambiguous_px"]) or the evidence decides.

    det = Detector(board, spec, settings)
    found = det.detect(gray, box=None, marker_px=None)          # Detection: corners, ids, markers, time
    tr = Tracker(board, spec, settings, anchor_points)          # anchor_points: model points in the board frame (mm)
    res = tr.step(gray, t, n, K, dist, detection=found)         # TrackResult: T, ok, sure, corners, rms, ...

    python board_tracker.py test     synthetic sequences of both boards: tracking against the truth, still and
                                     moving, a detection carried to a later frame, the mirrored pose, the small board
                                     at 25-60 cm; exit 1 on failure
"""
import sys
import time

import cv2
import numpy as np

TRACK = {
    "detect_every_s": 0.25,     # a detection at least this often while tracking (to re-anchor and to add corners)
    "detect_weak_share": 0.7,   # and at once when fewer than this share of the corners last detected are tracked
    "min_corners": 6,           # a pose needs this many corners (tracked)
    "min_corners_sheet": 8,     # and from a detection, on the calibration board
    "few_corners": 10,          # with fewer, the corners must spread over the board and fit tighter
    "few_rms_share": 0.5,       #   (RMS under this share of the limit)
    "klt_square_px": 20.0,      # the optical flow runs on the image halved until a square is about this wide
    "klt_win": 15,              # Lucas-Kanade window (px, scaled image), 21 when the corners move fast
    "klt_levels": 3,            #   pyramid levels, up to 5 when they move fast
    "fb_px": 0.8,               # forward-backward error allowed (px, scaled image)
    "subpix_px": 1.6,           # cornerSubPix may move a corner at most this far from the flow (and half its window)
    "outlier_px": 2.0,          # a corner farther than this (and than 3 sigma) from the pose is dropped
    "ransac_share": 0.3,        # with more outliers than this, RANSAC
    "marker_px": 34.0,          # detection in the box: scaled so that the markers are about this wide (never enlarged)
    "min_scale": 0.35,
    "roi_margin": 0.3,          # the box: the board's outline grown by this share of its size
    "roi_px": 40,               #   and this many px
    "ambiguous_px": 6.0,        # the two IPPE poses closer than this on the overlay: either will do
    "evidence_switch": 12.0,    # evidence (sum of N log error ratios, forgetting) for the other pose to switch to it
    "evidence_sure": 3.0,       # evidence for the pose kept to call it sure
    "evidence_keep": 0.92,      # forgetting per frame
    "corner_noise_px": 0.15,    # noise of a tracked corner from frame to frame (px), for the expected noise of the pose
    "reanchor_px": 0.75,        # a detection re-anchors only the corners it moves more than this: no twitch
}

_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 20, 0.01)
_LK = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 20, 0.02)


def matrix(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    T[:3, 3] = np.ravel(tvec)
    return T


def vectors(T):
    return cv2.Rodrigues(np.asarray(T[:3, :3], np.float64))[0], np.asarray(T[:3, 3], np.float64).reshape(3, 1)


def project(X, T, K, dist):
    rvec, tvec = vectors(T)
    return cv2.projectPoints(np.asarray(X, np.float64), rvec, tvec, K, dist)[0].reshape(-1, 2)


def rot_angle(Ra, Rb):
    return float(np.linalg.norm(cv2.Rodrigues(np.asarray(Ra, np.float64) @ np.asarray(Rb, np.float64).T)[0]))


class Detection:
    """What one ChArUco detection found on frame n (taken at t): corners (N x 2, px of the whole frame), ids, markers;
    the gray frame it was made on (to carry the corners to a later frame); ms it took; the box searched."""

    __slots__ = ("n", "t", "gray", "corners", "ids", "markers", "ms", "box", "scale")

    def __init__(self, n, t, gray, corners, ids, markers, ms, box=None, scale=1.0):
        self.n, self.t, self.gray, self.corners, self.ids, self.markers = n, t, gray, corners, ids, markers
        self.ms, self.box, self.scale = ms, box, scale


class Detector:
    """ChArUco detection tuned for small markers (see the module docstring)."""

    def __init__(self, board, spec, settings=None):
        self.board, self.spec = board, spec
        self.s = {**TRACK, **(settings or {})}
        self.dictionary = board.getDictionary()
        cp = cv2.aruco.CharucoParameters()
        cp.minMarkers = 1
        cp.tryRefineMarkers = True
        dp = cv2.aruco.DetectorParameters()
        dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.charuco = cv2.aruco.CharucoDetector(board, cp, dp)
        self._aruco = {}

    def _markers(self, marker_px, longest):
        """ArUco detector for markers about marker_px wide in an image whose longest side is `longest` (None: any
        size in a whole frame)."""
        key = None if marker_px is None else (int(round(marker_px)), int(longest) // 64)
        if key not in self._aruco:
            dp = cv2.aruco.DetectorParameters()
            if marker_px is None:
                dp.adaptiveThreshWinSizeMin, dp.adaptiveThreshWinSizeMax, dp.adaptiveThreshWinSizeStep = 5, 35, 10
                dp.minMarkerPerimeterRate = 0.018
            else:
                win = int(np.clip(round(0.4 * marker_px), 3, 31)) | 1
                dp.adaptiveThreshWinSizeMin = win
                dp.adaptiveThreshWinSizeMax = min(2 * win + 1, 41) | 1
                dp.adaptiveThreshWinSizeStep = max(win + 1, 2)
                dp.minMarkerPerimeterRate = float(np.clip(4 * 0.45 * marker_px / longest, 0.005, 1.0))
                dp.maxMarkerPerimeterRate = float(np.clip(4 * 2.5 * marker_px / longest, 0.05, 4.0))
            dp.polygonalApproxAccuracyRate = 0.05
            dp.minCornerDistanceRate = 0.05
            dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE      # refined at full resolution below
            self._aruco[key] = cv2.aruco.ArucoDetector(self.dictionary, dp)
        return self._aruco[key]

    def detect(self, gray, box=None, marker_px=None, n=0, t=0.0):
        """Corners of the board in the gray frame: in box (x0, y0, x1, y1) with markers about marker_px wide, or in
        the whole frame. Returns a Detection (no corners: empty arrays)."""
        t0 = time.perf_counter()
        h, w = gray.shape[:2]
        x0, y0, x1, y1 = (0, 0, w, h) if box is None else box
        crop = gray[y0:y1, x0:x1]
        scale = 1.0
        if marker_px is not None:
            scale = float(np.clip(self.s["marker_px"] / max(marker_px, 1.0), self.s["min_scale"], 1.0))
            if scale > 0.9:
                scale = 1.0
        small = crop if scale == 1.0 else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        det = self._markers(None if marker_px is None else marker_px * scale, max(small.shape[:2]))
        mc, mids, _ = det.detectMarkers(small)
        none = Detection(n, t, gray, np.empty((0, 2)), np.empty(0, np.int32), 0, 0.0, box, scale)
        if mids is None or not len(mids):
            none.ms = (time.perf_counter() - t0) * 1e3
            return none
        pts = np.concatenate([c.reshape(4, 2) for c in mc]).astype(np.float32)
        if scale < 1.0:
            pts = (pts + 0.5) / scale - 0.5
        side = float(np.median(np.linalg.norm(pts.reshape(-1, 4, 2) - np.roll(pts.reshape(-1, 4, 2), 1, 1), axis=2)))
        hw = int(np.clip(round(0.08 * side), 1, 5))
        pts = cv2.cornerSubPix(crop, pts.reshape(-1, 1, 2), (hw, hw), (-1, -1), _SUBPIX).reshape(-1, 4, 2)
        mc = tuple(p.reshape(1, 4, 2) for p in pts)
        cc, cids, _, mids2 = self.charuco.detectBoard(crop, markerCorners=mc, markerIds=mids)
        nm = len(mids2) if mids2 is not None else len(mids)
        ms = (time.perf_counter() - t0) * 1e3
        if cids is None or not len(cids):
            none.markers, none.ms = int(nm), ms
            return none
        return Detection(n, t, gray, cc.reshape(-1, 2).astype(np.float64) + (x0, y0), cids.ravel().astype(np.int32),
                         int(nm), ms, box, scale)


class TrackResult:
    """The board on one frame: T (camera from board, mm) when ok; sure False while the mirrored pose is still
    possible; corners and ids where they are on this frame; rms (px), source (tracked, detected, carried), whether
    the pose was re-anchored by a detection (shift: how far the overlay moved, px), the box and marker size for the
    next detection, and the time of the stages."""

    def __init__(self):
        self.ok, self.sure, self.T, self.rms, self.n = False, False, None, None, 0
        self.corners, self.ids = np.empty((0, 2)), np.empty(0, np.int32)
        self.source, self.reanchored, self.shift_px, self.flipped = "none", False, 0.0, False
        self.box, self.marker_px, self.square_px = None, None, None
        self.ambiguity_px, self.evidence = 0.0, 0.0
        self.noise_px = None        # expected noise of the overlay from the rotation alone, px per frame (see _noise)
        self.noise_pos_px = None    # and of the anchor (the board's centre), px per frame
        self.tm = {}
        self.why = ""
        self.counts = None          # corners before, after the flow, after the saddle point refinement


class Tracker:
    """Follows the board's corners from frame to frame (see the module docstring). One per stream, used by one
    thread."""

    def __init__(self, board, spec, settings=None, anchor_points=None):
        self.board, self.spec = board, spec
        self.s = {**TRACK, **(settings or {})}
        self.obj = board.getChessboardCorners().astype(np.float64)
        nx, ny = spec["squares"]
        sq = spec["square_mm"]
        self.sq = float(sq)
        self.outline = np.array([[0, 0, 0], [nx * sq, 0, 0], [nx * sq, ny * sq, 0], [0, ny * sq, 0]], np.float64)
        self.margin_share = (spec["square_mm"] - spec["marker_mm"]) / 2 / spec["square_mm"]
        self.anchor = np.asarray(anchor_points if anchor_points is not None else self.outline, np.float64)
        self.noise_points = np.ascontiguousarray(self.anchor[:: max(1, len(self.anchor) // 24)])
        self.centre = np.array([nx * sq / 2, ny * sq / 2, 0.0])
        self.min_detect = self.s["min_corners_sheet"] if spec.get("name") == "sheet" else self.s["min_corners"]
        self.counts = {"frames": 0, "tracked": 0, "lost": 0, "detections": 0, "reanchors": 0, "disagreements": 0,
                       "flips": 0, "rejected": 0}
        self.reset()

    def reset(self):
        self.prev = None            # (gray, n, t) of the last frame processed
        self.pts = np.empty((0, 2), np.float32)
        self.ids = np.empty(0, np.int32)
        self.motion = None          # per corner displacement of the last step (px per frame)
        self.T = None
        self.size = None
        self.history = {}           # n -> (ids, pts) of the frames tracked, for checking detections
        self.best = 0               # corners of the last detection
        self.evidence = 0.0         # for the other IPPE pose, against the one kept
        self.sure = False

    # ── helpers ──────────────────────────────────────────────────────────
    def _square_px(self, T, K):
        """Side of a square in the image near the board centre, px."""
        nx, ny = self.spec["squares"]
        c = np.array([[nx * self.sq / 2, ny * self.sq / 2, 0], [nx * self.sq / 2 + self.sq, ny * self.sq / 2, 0],
                      [nx * self.sq / 2, ny * self.sq / 2 + self.sq, 0]])
        p = project(c, T, K, np.zeros(5))
        return float(max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0]), 1.0))

    def _subpix(self, gray, pts, square_px):
        """Corners put on the chessboard saddle points at full resolution: window inside the white margin around
        the markers. Returns (pts, ok)."""
        if not len(pts):
            return pts, np.zeros(0, bool)
        hw = int(np.clip(np.floor(0.85 * self.margin_share * square_px), 2, 8))
        hw = min(hw, max(2, int(0.25 * square_px)))
        h, w = gray.shape[:2]
        inside = (pts[:, 0] > hw + 2) & (pts[:, 1] > hw + 2) & (pts[:, 0] < w - hw - 3) & (pts[:, 1] < h - hw - 3)
        out = pts.copy()
        if inside.any():
            ref = cv2.cornerSubPix(gray, pts[inside].reshape(-1, 1, 2).astype(np.float32).copy(), (hw, hw), (-1, -1),
                                   _SUBPIX).reshape(-1, 2)
            out[inside] = ref
        moved = np.linalg.norm(out - pts, axis=1)
        ok = inside & (moved <= min(self.s["subpix_px"], 0.6 * hw + 0.5))
        return out, ok

    @staticmethod
    def _pyramid(gray, levels, cache=None):
        """gray halved `levels` times (pyrDown), reusing the halves already made (cache: the list for this frame)."""
        pyr = cache if cache is not None and cache and cache[0] is gray else [gray]
        while len(pyr) <= levels:
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    def _flow(self, g0, g1, p0, guess, square_px, pyr0=None, pyr1=None):
        """Corners p0 of frame g0 in frame g1 by pyramidal Lucas-Kanade on the box around them, on the image halved
        until a square is about klt_square_px wide; guess: where they are expected in g1. pyr0, pyr1: the halves of
        g0 and g1 already made (lists, filled here). Returns (p1 at full resolution, ok)."""
        if not len(p0):
            return p0, np.zeros(0, bool)
        h, w = g1.shape[:2]
        lv = int(np.clip(np.floor(np.log2(max(square_px, 1.0) / self.s["klt_square_px"]) + 0.5), 0, 3))
        pyr0 = self._pyramid(g0, lv, pyr0)
        pyr1 = self._pyramid(g1, lv, pyr1)
        s = 0.5 ** lv
        both = np.vstack([p0, guess])
        lo, hi = both.min(0), both.max(0)
        grow = 2.5 * square_px + 24
        x0, y0 = (np.clip(np.floor((lo - grow) * s), 0, None)).astype(int)
        x1, y1 = (np.ceil((hi + grow) * s)).astype(int)
        hs, ws = pyr1[lv].shape[:2]
        x1, y1 = min(x1, ws), min(y1, hs)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return p0, np.zeros(len(p0), bool)
        c0, c1 = pyr0[lv][y0:y1, x0:x1], pyr1[lv][y0:y1, x0:x1]
        off = np.array([x0, y0], np.float32)
        a = ((p0 + 0.5) * s - 0.5 - off).astype(np.float32).reshape(-1, 1, 2)
        b0 = ((guess + 0.5) * s - 0.5 - off).astype(np.float32).reshape(-1, 1, 2)
        # fast motion (the guess far from where the corners were): wider window, deeper pyramid
        move = float(np.median(np.linalg.norm((b0 - a).reshape(-1, 2), axis=1)))
        w_ = self.s["klt_win"] + (6 if move > 8 else 0)
        n_lv = self.s["klt_levels"] + (move > 20) + (move > 40)
        win = (w_, w_)

        def lk(start):
            b, st, _ = cv2.calcOpticalFlowPyrLK(c0, c1, a, start.copy(), winSize=win, maxLevel=n_lv, criteria=_LK,
                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(c1, c0, b, a.copy(), winSize=win, maxLevel=n_lv, criteria=_LK,
                                                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            fb = np.linalg.norm((back - a).reshape(-1, 2), axis=1)
            return b, (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.s["fb_px"]), fb
        b, ok, fb = lk(b0)
        if move > 2 and (~ok).sum() > 0.2 * len(ok):
            # where the motion turned (the end of a swing) the guess from the speed overshoots: from where they were
            b2, ok2, fb2 = lk(a)
            better = ok2 & (~ok | (fb2 < fb))
            b[better], ok = b2[better], ok | ok2
        p1 = (b.reshape(-1, 2) + off + 0.5) / s - 0.5
        return p1.astype(np.float32), ok

    def _ippe(self, obj, img, K, dist, refine=True):
        """Both IPPE poses (refined by Levenberg-Marquardt): [(T, rms)] best first."""
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        out = []
        for r, t, e in zip(rvecs, tvecs, np.ravel(errs)):
            if refine:
                r, t = cv2.solvePnPRefineLM(obj, img, K, dist, r.copy(), t.copy())
                proj = cv2.projectPoints(obj, r, t, K, dist)[0].reshape(-1, 2)
                e = np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1)))
            out.append((matrix(r, t), float(e)))
        return sorted(out, key=lambda x: x[1])

    def _fit(self, ids, pts, K, dist, prior):
        """Pose from corners: LM from the prior (or IPPE without one), corners far from the pose dropped, RANSAC when
        many are. Returns (T, rms, keep mask) or (None, rms, keep)."""
        obj = self.obj[ids]
        img = pts.astype(np.float64)
        keep = np.ones(len(ids), bool)
        if prior is not None:
            rvec, tvec = vectors(prior)
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, rvec.copy(), tvec.copy(), True, cv2.SOLVEPNP_ITERATIVE)
            T = matrix(rvec, tvec) if ok else None
        else:
            T = self._ippe(obj, img, K, dist)[0][0]
        if T is None:
            return None, None, keep
        for _ in range(2):
            r = np.linalg.norm(project(obj, T, K, dist) - img, axis=1)
            med = np.median(r[keep]) if keep.any() else 0.0
            thr = max(self.s["outlier_px"], 3 * 1.4826 * med)
            bad = r > thr
            if not bad.any():
                break
            if bad.mean() > self.s["ransac_share"] and len(ids) >= 6:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, dist, reprojectionError=self.s["outlier_px"],
                                                          iterationsCount=60, flags=cv2.SOLVEPNP_SQPNP)
                if not ok or inl is None or len(inl) < 4:
                    return None, None, keep
                keep = np.zeros(len(ids), bool)
                keep[inl.ravel()] = True
            else:
                keep = ~bad
            if keep.sum() < 4:
                return None, None, keep
            rvec, tvec = vectors(T)
            ok, rvec, tvec = cv2.solvePnP(obj[keep], img[keep], K, dist, rvec.copy(), tvec.copy(), True,
                                          cv2.SOLVEPNP_ITERATIVE)
            T = matrix(rvec, tvec)
        r = np.linalg.norm(project(obj[keep], T, K, dist) - img[keep], axis=1)
        return T, float(np.sqrt(np.mean(r ** 2))), keep

    def _spread_ok(self, ids):
        """The corners span the board in two directions (not a line, not a corner of it)."""
        xy = self.obj[ids, :2]
        if len(xy) < 4:
            return False
        sv = np.linalg.svd(xy - xy.mean(0), compute_uv=False) / np.sqrt(len(xy))
        return bool(sv[1] >= 0.5 * self.sq)

    def _noise(self, ids, T, K, dist):
        """How much the rotation of a pose from these corners shakes the overlay from frame to frame, px: corner noise
        TRACK["corner_noise_px"] carried through the PnP (covariance from the Jacobian) to the model points, rotation
        part only. A small board far away gives a large value: the filter then smooths its rotation more."""
        rvec, tvec = vectors(T)
        _, J = cv2.projectPoints(self.obj[ids], rvec, tvec, K, dist)
        J = J[:, :6]
        try:
            cov = np.linalg.inv(J.T @ J) * self.s["corner_noise_px"] ** 2
        except np.linalg.LinAlgError:
            return None
        X = self.noise_points
        _, Jf = cv2.projectPoints(X, rvec, tvec, K, dist)
        Jr = Jf[:, :3]
        rot = float(np.sqrt(max(np.trace(Jr @ cov[:3, :3] @ Jr.T), 0.0) / len(X)))
        _, Jc = cv2.projectPoints(self.centre[None], rvec, tvec, K, dist)
        Jc = Jc[:, :6]
        pos = float(np.sqrt(max(np.trace(Jc @ cov @ Jc.T), 0.0)))
        return rot, pos

    def _overlay_shift(self, Ta, Tb, K):
        return float(np.sqrt(np.mean(np.sum((project(self.anchor, Ta, K, np.zeros(5))
                                             - project(self.anchor, Tb, K, np.zeros(5))) ** 2, axis=1))))

    def _weigh(self, ids, pts, K, dist, res):
        """Both IPPE poses on this frame's corners: evidence for the other one, with forgetting; switch on strong
        evidence; sure once the two agree on the overlay or the evidence decides."""
        sols = self._ippe(self.obj[ids], pts.astype(np.float64), K, dist, refine=False)
        if len(sols) < 2:
            self.sure = True
            return
        ang = [rot_angle(T[:3, :3], self.T[:3, :3]) for T, _ in sols]
        mine = int(np.argmin(ang))
        other = sols[1 - mine]
        e_mine, e_other = max(sols[mine][1], 1e-3), max(other[1], 1e-3)
        res.ambiguity_px = self._overlay_shift(sols[0][0], sols[1][0], K)
        step = float(np.clip(len(ids) * np.log(e_mine / e_other), -4.0, 4.0))
        self.evidence = self.s["evidence_keep"] * self.evidence + step
        res.evidence = self.evidence
        if self.evidence > self.s["evidence_switch"]:
            T, rms, keep = self._fit(ids, pts, K, dist, other[0])
            if T is not None:
                res.flipped = True
                res.shift_px = self._overlay_shift(T, self.T, K)
                self.T = T
                self.counts["flips"] += 1
                self.evidence = -self.s["evidence_sure"]
        if res.ambiguity_px < self.s["ambiguous_px"] or self.evidence < -self.s["evidence_sure"]:
            self.sure = True

    def _accept(self, res, T, rms, ids, pts, limit, K, dist, source):
        """Checks of a pose; on success the tracker takes it."""
        n = len(ids)
        if T is None or rms is None:
            res.why = "no pose"
            return False
        few = n < self.s["few_corners"]
        if rms > limit * (self.s["few_rms_share"] if few else 1.0):
            res.why = f"rms {rms:.2f}"
            return False
        if few and not self._spread_ok(ids):
            res.why = "corners on a line"
            return False
        if np.any((T[:3, :3] @ self.outline.T + T[:3, 3:4])[2] <= 1.0):
            res.why = "board behind the camera"
            return False
        self.T, self.pts, self.ids = T, pts.astype(np.float32), ids.astype(np.int32)
        res.ok, res.T, res.rms, res.n, res.source = True, T, rms, n, source
        res.corners, res.ids = self.pts.astype(np.float64), self.ids
        return True

    # ── one frame ────────────────────────────────────────────────────────
    def step(self, gray, t, n, K, dist, limit, detection=None):
        """The board on frame n (gray, full resolution, taken at t). detection: a Detection made on this frame or on
        an earlier one (carried here by the flow). limit: reprojection RMS allowed, px."""
        res = TrackResult()
        tm0 = time.perf_counter()
        K = np.asarray(K, np.float64)
        dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
        h, w = gray.shape[:2]
        if self.size != (w, h):
            self.reset()
            self.size = (w, h)
        self.counts["frames"] += 1
        prev = self.prev
        pyr = [gray]                # its halves, made by the flow when needed and kept for the next frame
        tracking = prev is not None and self.T is not None and len(self.ids) >= self.s["min_corners"]
        square_px = self._square_px(self.T, K) if self.T is not None else None
        # 1. the tracked corners moved to this frame
        if tracking:
            gap = max(1, n - prev[1])
            guess = self.pts + (self.motion * gap if self.motion is not None and len(self.motion) == len(self.pts)
                                else 0)
            p1, ok = self._flow(prev[0], gray, self.pts, guess.astype(np.float32), square_px, prev[3], pyr)
            ps, ok2 = self._subpix(gray, p1, square_px)
            res.counts = (len(self.pts), int(ok.sum()), int((ok & ok2).sum()))
            # on a blurred frame the saddle point is smeared and its refinement wanders off: the flow's position is
            # kept there (the next sharp frame refines it again, a detection re-anchors it)
            p1 = np.where(ok2[:, None], ps, p1).astype(np.float32)
            new_motion = (p1 - self.pts) / gap
            ids, pts, motion = self.ids[ok], p1[ok], new_motion[ok]
            res.tm["flow_ms"] = (time.perf_counter() - tm0) * 1e3
            T, rms, keep = (self._fit(ids, pts, K, dist, self.T) if len(ids) >= self.s["min_corners"]
                            else (None, None, None))
            if T is not None:
                ids, pts, motion = ids[keep], pts[keep], motion[keep]
            if self._accept(res, T, rms, ids, pts, limit, K, dist, "tracked"):
                self.motion = motion.astype(np.float32)
                self.counts["tracked"] += 1
            else:
                self.counts["lost"] += 1
                self.T, self.motion = None, None
                self.pts, self.ids = np.empty((0, 2), np.float32), np.empty(0, np.int32)
                self.sure = False
        tm1 = time.perf_counter()
        # 2. a detection: its corners carried to this frame take the place of the tracked ones
        if detection is not None and len(detection.ids) >= min(self.min_detect, self.s["min_corners"]):
            self._merge(detection, gray, n, K, dist, limit, res, square_px, pyr)
        res.tm["detect_merge_ms"] = (time.perf_counter() - tm1) * 1e3
        # 3. the mirrored pose
        if res.ok and len(self.ids) >= 4:
            self._weigh(self.ids, self.pts, K, dist, res)
        res.sure = bool(res.ok and self.sure)
        if res.ok:
            res.T = self.T
            res.noise_px, res.noise_pos_px = self._noise(self.ids, self.T, K, dist) or (None, None)
            self.history[n] = (self.ids.copy(), self.pts.copy())
            res.square_px = self._square_px(self.T, K)
            res.box, res.marker_px = self.box_for(self.T, K, dist, (w, h))
        for k in [k for k in self.history if k < n - 90]:
            del self.history[k]
        self.prev = (gray, n, t, pyr)
        res.tm["track_ms"] = (time.perf_counter() - tm0) * 1e3
        return res

    def _merge(self, d, gray, n, K, dist, limit, res, square_px, pyr):
        """Corners of a detection made on frame d.n carried to frame n and taken as the tracked set (re-anchoring),
        with the tracked corners it missed kept when they fit the new pose."""
        self.counts["detections"] += 1
        before = self.T
        # how the detection compares with what was tracked on its own frame
        past = self.history.get(d.n)
        if past is not None:
            common, ia, ib = np.intersect1d(d.ids, past[0], return_indices=True)
            if len(common):
                dev = np.linalg.norm(d.corners[ia] - past[1][ib], axis=1)
                if np.median(dev) > 1.5 or dev.max() > 4.0:
                    self.counts["disagreements"] += 1
        if d.n == n:
            pts, ok = d.corners.astype(np.float32), np.ones(len(d.ids), bool)
        else:
            # where each corner should be now: tracked, or from the current pose, or where it was
            guess = d.corners.astype(np.float32).copy()
            if self.T is not None:
                guess = project(self.obj[d.ids], self.T, K, dist).astype(np.float32)
                if len(self.ids):
                    common, ia, ib = np.intersect1d(d.ids, self.ids, return_indices=True)
                    guess[ia] = self.pts[ib]
            sp = square_px or self._square_from(d)
            pts, ok = self._flow(d.gray, gray, d.corners.astype(np.float32), guess, sp, None, pyr)
            pts, ok2 = self._subpix(gray, pts, sp)
            ok &= ok2
        ids, pts = d.ids[ok], pts[ok]
        if len(ids) < self.min_detect:
            return
        motion = None
        if self.T is not None and len(self.ids):
            # where the detection agrees with the tracked corners they stay as they are (and keep their speed): the
            # pose does not twitch at every detection; it moves only the corners that drifted or were lost
            common, ia, ib = np.intersect1d(ids, self.ids, return_indices=True)
            agree = np.linalg.norm(pts[ia] - self.pts[ib], axis=1) < self.s["reanchor_px"]
            pts[ia[agree]] = self.pts[ib[agree]]
            if self.motion is not None and len(self.motion) == len(self.ids):
                mean = self.motion.mean(0)
                motion = np.tile(mean, (len(ids), 1)).astype(np.float32)
                motion[ia] = self.motion[ib]
            # the tracked corners the detection missed
            extra = ~np.isin(self.ids, ids)
            if extra.any():
                ids = np.concatenate([ids, self.ids[extra]])
                pts = np.vstack([pts, self.pts[extra]])
                if motion is not None:
                    motion = np.vstack([motion, self.motion[extra]])
        if self.T is not None:
            T, rms, keep = self._fit(ids, pts, K, dist, self.T)
        else:
            T, rms, keep = self._fit(ids, pts, K, dist, None)
        if T is None:
            return
        ids, pts = ids[keep], pts[keep]
        if motion is not None:
            motion = motion[keep]
        was_ok = res.ok
        if not self._accept(res, T, rms, ids, pts, limit, K, dist, "detected" if d.n == n else "carried"):
            self.counts["rejected"] += 1
            if was_ok:          # the tracked pose stands
                res.ok = True
            return
        self.best = len(ids)
        self.motion = motion
        res.reanchored = True
        self.counts["reanchors"] += 1
        if before is None:
            self.evidence, self.sure = 0.0, False
        else:
            res.shift_px = self._overlay_shift(T, before, K)

    def _square_from(self, d):
        """Side of a square in px from a detection's corners (no pose yet)."""
        if len(d.ids) < 2:
            return 30.0
        xy = self.obj[d.ids, :2]
        dd = np.linalg.norm(xy[:, None] - xy[None], axis=2)
        dp = np.linalg.norm(d.corners[:, None] - d.corners[None], axis=2)
        m = (dd > 0) & (dd < 1.5 * self.sq)
        return float(np.median(dp[m] / dd[m] * self.sq)) if m.any() else 30.0

    def box_for(self, T, K, dist, size):
        """The box to search the board in on a later frame, and the marker size in px."""
        w, h = size
        if np.any((T[:3, :3] @ self.outline.T + T[:3, 3:4])[2] <= 1.0):
            return None, None
        p = project(self.outline, T, K, dist)
        lo, hi = p.min(0), p.max(0)
        grow = (hi - lo) * self.s["roi_margin"] + self.s["roi_px"]
        x0, y0 = np.clip(np.floor(lo - grow), 0, (w, h)).astype(int)
        x1, y1 = np.clip(np.ceil(hi + grow), 0, (w, h)).astype(int)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None, None
        marker_px = self._square_px(T, K) * self.spec["marker_mm"] / self.sq
        return (int(x0), int(y0), int(x1), int(y1)), float(marker_px)

    def want_detection(self, res, t, last_t):
        """A detection is due: lost, weak, or the last one older than detect_every_s."""
        if not res.ok:
            return True
        if self.best and len(self.ids) < self.s["detect_weak_share"] * self.best:
            return True
        return last_t is None or t - last_t >= self.s["detect_every_s"]


# ── checks ───────────────────────────────────────────────────────────────────
def run_test():
    """Synthetic frames of both boards (live_ar.SyntheticStream renders them with the stream's camera)."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import live_ar
    import marker_pose
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    K, dist, _ = live_ar.stream_intrinsics((1920, 1440))
    for name in ("sheet", "patient"):
        board, spec = live_ar.make_board(name, 24.0)
        s = live_ar.SyntheticStream(board, spec, K, exposure_s=0.0, noise=2.0, script=live_ar.bench_script(spec))
        s.close()
        det = Detector(board, spec)
        tr = Tracker(board, spec)
        errs, kinds, tms, last_det = [], [], [], None
        for k in range(150):         # 5 s at 30 fps: still, then moving (bench_script phases)
            t = 4.5 + k / 30.0       # from 4.5 s: still, the hand leaves, moving slowly from 6 s
            img, T_true = s.render(t, k)
            d = None
            if tr.want_detection(TrackResult() if k == 0 else res, t, last_det) and k % 3 == 0:
                box, mpx = (None, None) if k == 0 or not res.ok else (res.box, res.marker_px)
                d = det.detect(img, box, mpx, n=k, t=t)
                last_det = t
            res = tr.step(img, t, k, K, dist, 1.0, d)
            kinds.append(res.source if res.ok else "lost")
            if res.ok:
                e = marker_pose.target_error(res.T, T_true, tr.outline, K)
                errs.append(e["overlay_rms_px"])
                tms.append(res.tm.get("track_ms", 0))
        errs = np.array(errs)
        expect(len(errs) >= 140 and np.percentile(errs, 95) < 0.6,
               f"{name}: {len(errs)} of 150 frames with a pose, board outline against the truth median "
               f"{np.median(errs):.3f} px, 95th percentile {np.percentile(errs, 95):.3f} px; tracked "
               f"{kinds.count('tracked')}, from detections {kinds.count('detected') + kinds.count('carried')}; "
               f"{np.median(tms):.1f} ms per frame; counts {tr.counts}")
    # a detection carried over 3 frames while moving fast
    board, spec = live_ar.make_board("sheet", 24.0)
    s = live_ar.SyntheticStream(board, spec, K, exposure_s=0.0, noise=2.0, script=live_ar.bench_script(spec))
    s.close()
    det, tr = Detector(board, spec), Tracker(board, spec)
    img0, _ = s.render(13.0, 0)
    d0 = det.detect(img0, n=0, t=13.0)
    tr.step(img0, 13.0, 0, K, dist, 1.0, d0)
    imgs = [s.render(13.0 + k / 30, k) for k in range(1, 5)]
    d1 = det.detect(imgs[0][0], n=1, t=13.0 + 1 / 30)
    for k, (img, T_true) in enumerate(imgs, start=1):
        res = tr.step(img, 13.0 + k / 30, k, K, dist, 1.0, d1 if k == 4 else None)
    e = marker_pose.target_error(res.T, imgs[-1][1], tr.outline, K)["overlay_rms_px"] if res.ok else np.inf
    expect(res.ok and res.source == "carried" and e < 0.6,
           f"detection of frame 1 carried to frame 4 while moving fast: {res.source}, {e:.3f} px from the truth")
    # small board at several distances, nearly face on
    board, spec = live_ar.make_board("patient")
    det = Detector(board, spec)
    rows = []
    for dist_mm in (200, 250, 350, 500, 600, 700):
        s = live_ar.SyntheticStream(board, spec, K, exposure_s=0.0, noise=2.0)
        s.close()
        s._dist = float(dist_mm)
        tr = Tracker(board, spec)
        found, n_ok, err = 0, 0, []
        for k in range(20):
            img, T_true = s.render(1.0 + k / 30.0, k)
            d = det.detect(img, n=k) if k % 5 == 0 else None
            if d is not None and len(d.ids):
                found += 1
            res = tr.step(img, k / 30, k, K, dist, 1.0, d)
            if res.ok:
                n_ok += 1
                err.append(marker_pose.pose_error(res.T, T_true)["rot_deg"])
        rows.append((dist_mm, found, n_ok, float(np.median(err)) if err else np.nan))
    print("      small board, distance mm / detections with corners of 4 / frames with a pose of 20 / rotation "
          "error deg: " + "; ".join(f"{a} / {b} / {c} / {d:.2f}" for a, b, c, d in rows))
    expect(all(r[2] >= 18 for r in rows if r[0] <= 500), "small board tracked from 20 to 50 cm")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(run_test())
    sys.exit(__doc__)
