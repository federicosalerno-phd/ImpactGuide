"""The photo taken by itself: when the phone's preview shows the board where the guide of the next photo wants it,
and the phone has been still a moment, the shutter goes, at most once every GAP_S.

What is compared. The guide of a pose (calib_guide.guide) says where the board's four corners fall in the photo, in
fractions of its width and height (top left, top right, bottom right, bottom left of the board). The preview gives the
same four corners: the ChArUco corners found in it, the plane through them (a homography from the board's own
coordinates), the board's corners on that plane. Both sets of corners are turned into a pose of the board with the
nominal camera of the mode (calib_guide.camera), and the poses are compared where the calibration needs them to
differ: the board's tilt (the angle between the two normals, within TILT_DEG), where its centre falls (within AT of
the photo's sides), how far it is (within DIST_RATIO either way) and its turn on the screen (within ROLL_DEG). Corner
distances alone cannot tell a board tilted by 40 degrees from one seen straight on. The board is still once its
corners have moved less than STILL between the looks of HOLD_S.

Where the preview comes from:
    photo   the frame is the photo itself, upright (the live stream of the streaming mode, the simulation)
    display the phone's own screen, mirrored (PreviewStream: the camera app is on it, and only one app at a time can
            use the camera). The camera app stays upright on the phone held up; its preview of a 4:3 photo is as
            wide as the screen and 4/3 as tall (preview_rect finds it between the app's dark bands). Held across,
            the photo is the preview turned a quarter: both turns are tried, the one that matches counts.

Watch is the judgement alone (fed frames, it says the state and when a photo is due) and runs on a scripted clock in
the test; Watcher runs it on a thread over a frame source and calls back when a photo is due.

States (Watch.feed): none (no board in the preview), far (a board, not where the guide wants it), hold (where the guide
wants it, not still yet), wait (ready, but the last photo is younger than GAP_S), ready (a photo is due).

Usage:
    python auto_shot.py test      synthetic previews of the plans' poses (calib_guide.photo), photo and display
                                  frames, both turns, the dark bands found, the clock scripted; exit 1 on failure
    python auto_shot.py watch     the phone's screen mirrored, the board looked for in it, the state printed (Ctrl+C)
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

import calib_guide
import calibrate_charuco

TILT_DEG = 15.0         # the board's normal within this of the guide's
AT = 0.12               # its centre within this of the guide's (fractions of the photo's sides)
DIST_RATIO = 1.25       # its distance within this factor of the guide's
ROLL_DEG = 15.0         # its turn on the screen within this of the guide's
STILL = 0.012           # the corners moved less than this between two looks (fractions of the photo's sides): still
HOLD_S = 0.6            # still this long before the photo
GAP_S = 5.0             # at least this long between two photos
RETRY_S = 1.0           # a photo that could not be taken is asked for again this much later at the soonest
LOOK_HZ = 6.0           # looks at the preview per second
LOOK_PX = 1280          # the preview is looked at with its long side at most this
MIN_SEEN = 6            # corners to place the board's plane
PHOTO_ASPECT = 4 / 3    # the photo's width over its height (the preview of the camera app, upright, is 3:4)
DARK = 24               # a grey level below this belongs to the camera app's dark bands
PREVIEW_SIZE = 1600     # the mirrored screen's long side
PREVIEW_FPS = 15
PREVIEW_BIT_RATE = 6_000_000


def board_outline(gray: np.ndarray, board: cv2.aruco.CharucoBoard, detector=None) -> tuple[np.ndarray | None, int]:
    """(the board's four corners in the image, top left, top right, bottom right, bottom left; corners found), or
    (None, corners found) when they cannot place its plane."""
    detector = detector or cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(gray)
    n = 0 if ids is None else len(ids)
    if n < MIN_SEEN or board.checkCharucoCornersCollinear(ids):
        return None, n
    obj, img = board.matchImagePoints(corners, ids)
    H, _ = cv2.findHomography(obj.reshape(-1, 3)[:, :2].astype(np.float64), img.reshape(-1, 2).astype(np.float64), 0)
    if H is None:
        return None, n
    nx, ny = board.getChessboardSize()
    s = board.getSquareLength()
    c = np.array([[0, 0], [nx * s, 0], [nx * s, ny * s], [0, ny * s]], np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(c, H).reshape(4, 2), n


def preview_rect(gray: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, w, h) of the camera app's preview on the mirrored screen held up: as wide as the screen, PHOTO_ASPECT
    times as tall, where the rows that are not the app's dark bands are (centred on the screen when they cannot
    tell)."""
    H, W = gray.shape[:2]
    h = min(float(H), W * PHOTO_ASPECT)
    rows = (gray > DARK).mean(axis=1) > 0.35
    best, start, run = (0, 0), None, 0
    for i, r in enumerate(np.append(rows, False)):
        if r and start is None:
            start = i
        if not r and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    run = best[1] - best[0]
    y = (H - h) / 2
    if 0.8 * h <= run <= 1.08 * h:
        y = min(max(0.0, (best[0] + best[1]) / 2 - h / 2), H - h)
    return 0.0, float(y), float(W), float(h)


def in_photo(pts: np.ndarray, size: tuple[int, int], kind: str, rect=None) -> list[np.ndarray]:
    """The board's corners in fractions of the photo (x right, y down): one reading for a photo frame, the two turns
    of a phone held across for the mirrored screen."""
    pts = np.asarray(pts, np.float64)
    if kind == "photo":
        return [pts / np.array(size, np.float64)]
    x, y, w, h = rect
    a, b = (pts[:, 0] - x) / w, (pts[:, 1] - y) / h
    return [np.stack([b, 1 - a], 1), np.stack([1 - b, a], 1)]


def distance(seen: np.ndarray, want) -> float:
    """Mean distance of the four corners (fractions of the photo's sides)."""
    return float(np.mean(np.hypot(*(np.asarray(seen) - np.asarray(want, np.float64)).T)))


def outline_pose(outline, board_wh: tuple[float, float], mode: str = "photo") -> dict | None:
    """The pose of a board of board_wh mm whose four corners fall at `outline` (fractions of the photo), with the
    nominal camera of the mode: {normal (unit, camera frame), at (its centre, fractions), dist (mm), roll (degrees)}."""
    K, (w, h) = calib_guide.camera(mode)
    W, H = board_wh
    obj = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], np.float64)
    img = np.asarray(outline, np.float64) * [w, h]
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    R = cv2.Rodrigues(rvec)[0]
    centre = R @ [W / 2, H / 2, 0] + tvec.ravel()
    if centre[2] <= 0:
        return None
    uv = (K @ centre)[:2] / centre[2] / [w, h]
    x_axis = R[:, 0]
    return {"normal": R[:, 2], "at": uv, "dist": float(np.linalg.norm(centre)),
            "roll": math.degrees(math.atan2(x_axis[1], x_axis[0]))}


def pose_error(seen: dict, want: dict) -> float:
    """How far a pose is from the guide's, against what a match allows: 1 at the edge of a match, 0 the same pose."""
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(seen["normal"], want["normal"]))))))
    at = float(np.hypot(*(seen["at"] - want["at"])))
    dist = abs(math.log(seen["dist"] / want["dist"])) / math.log(DIST_RATIO)
    roll = abs((seen["roll"] - want["roll"] + 180) % 360 - 180)
    return max(tilt / TILT_DEG, at / AT, dist, roll / ROLL_DEG)


class Watch:
    """The judgement over the preview: fed one frame at a time, it says the state, and once `ready` the photo is due
    (the caller takes it and calls shot()). The guide (target) and the board are set as they change."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, gap_s: float = GAP_S, hold_s: float = HOLD_S):
        self.clock, self.gap_s, self.hold_s = clock, gap_s, hold_s
        self.want = None            # the guide's pose of the next photo (None: no photo wanted)
        self.key = None             # what the guide is of (a pose of a series): a new one starts the stillness again
        self.mode = "photo"
        self._board_key = None
        self.board = self.detector = self.board_wh = None
        self.last_shot = -math.inf  # when the last photo was taken
        self._prev = None           # the corners at the last look
        self._still_since = None
        self.state, self.error, self.corners = "none", None, 0
        self._rect, self._rect_t = None, -math.inf

    def target(self, want, board_spec: dict | None, key=None, mode: str = "photo") -> None:
        """The guide's outline of the next photo and the board it is of ({squares, square_mm, marker_mm}), for the
        camera of the mode; None when no photo is wanted (the plan full, the board not on show)."""
        bk = None if not board_spec else (tuple(board_spec["squares"]), float(board_spec["square_mm"]),
                                           float(board_spec.get("marker_mm") or 0))
        if bk != self._board_key:
            self._board_key = bk
            self.board = None if bk is None else calibrate_charuco.make_board(bk[1], bk[0], bk[2] or None)
            self.detector = None if bk is None else cv2.aruco.CharucoDetector(self.board)
            self.board_wh = None if bk is None else (bk[0][0] * bk[1], bk[0][1] * bk[1])
        if key != self.key:
            self._still_since = None
        self.mode = mode
        self.want = None if want is None or bk is None else outline_pose(want, self.board_wh, mode)
        self.key = key

    def shot(self) -> None:
        """A photo was taken (by itself or by hand): the next one no sooner than gap_s, after a new stillness."""
        self.last_shot = self.clock()
        self._still_since = None

    def _none(self) -> str:
        self.state, self.error, self._prev, self._still_since = "none", None, None, None
        return self.state

    def feed(self, img: np.ndarray, kind: str = "photo") -> str:
        """One look at the preview (BGR or grey); returns the state."""
        now = self.clock()
        if self.want is None or self.board is None:
            return self._none()
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, LOOK_PX / max(gray.shape[:2]))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
        pts, self.corners = board_outline(small, self.board, self.detector)
        if pts is None:
            return self._none()
        rect = None
        if kind == "display":
            if self._rect is None or now - self._rect_t > 2.0:     # the bands do not move: looked for now and then
                self._rect, self._rect_t = preview_rect(small), now
            rect = self._rect
        size = small.shape[1], small.shape[0]
        best = None
        for seen in in_photo(pts, size, kind, rect):
            pose = outline_pose(seen, self.board_wh, self.mode)
            if pose is not None:
                e = pose_error(pose, self.want)
                if best is None or e < best[0]:
                    best = (e, seen)
        if best is None:
            return self._none()
        self.error, seen = best
        moved = math.inf if self._prev is None else distance(seen, self._prev)
        self._prev = seen
        if self.error > 1.0:
            self.state, self._still_since = "far", None
            return self.state
        if moved > STILL:
            self.state, self._still_since = "hold", None
            return self.state
        if self._still_since is None:
            self._still_since = now
        if now - self._still_since < self.hold_s:
            self.state = "hold"
        elif now - self.last_shot < self.gap_s:
            self.state = "wait"
        else:
            self.state = "ready"
        return self.state

    def score(self) -> float:
        """How near the board is to the guide's pose: 1 within a match, 0 at three times its limits or more (or
        unseen)."""
        if self.error is None:
            return 0.0
        return round(max(0.0, min(1.0, 1 - (self.error - 1) / 2)), 3)


class Watcher:
    """Watch on a thread: every 1/LOOK_HZ s the newest frame of `source` (() -> (image, frame number, kind) or None)
    is looked at; `on_state(state, score)` hears each change of state, `fire()` is called when a photo is due (the
    caller then calls watch.shot(), or not when it could not take one)."""

    def __init__(self, source: Callable[[], tuple | None], fire: Callable[[], None],
                 on_state: Callable[[str, float], None] | None = None, watch: Watch | None = None):
        self.source, self.fire, self.on_state = source, fire, on_state or (lambda s, x: None)
        self.watch = watch or Watch()
        self._stop = threading.Event()
        self._lock = threading.Lock()       # target() from the caller's thread, feed() on this one
        self._last = None
        self._said = None
        self._fired = False                 # a photo asked for and not yet taken or refused
        self.thread = threading.Thread(target=self._run, daemon=True, name="auto_shot")

    def start(self) -> Watcher:
        self.thread.start()
        return self

    def stop(self, wait: bool = True) -> None:
        """Stops the looks; waits for the thread unless `wait` is False (it ends at its next look)."""
        self._stop.set()
        if wait and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(2.0)

    def target(self, want, board_spec, key=None, mode: str = "photo") -> None:
        with self._lock:
            self.watch.target(want, board_spec, key, mode)
            self._last = None           # the newest frame looked at again against the new guide

    def shot(self) -> None:
        with self._lock:
            self.watch.shot()
            self._fired = False

    def refused(self) -> None:
        """The photo asked for could not be taken (the phone busy): asked again a second later at the soonest."""
        with self._lock:
            self._fired = False
            w = self.watch
            w.last_shot = max(w.last_shot, w.clock() - w.gap_s + RETRY_S)

    def _run(self) -> None:
        period = 1.0 / LOOK_HZ
        while not self._stop.wait(period):
            try:
                got = self.source()
            except Exception:           # a source that failed this time: nothing seen
                got = None
            if not got or got[0] is None:
                state = "off"
                with self._lock:
                    self.watch.state, self.watch.error = "off", None
            else:
                img, n, kind = got
                if n == self._last and self.watch.state not in ("hold", "wait", "ready"):
                    continue
                self._last = n
                with self._lock:
                    state = self.watch.feed(img, kind)
                    due = state == "ready" and not self._fired
                    if due:
                        self._fired = True
                if due:
                    self.fire()
            score = self.watch.score()
            if (state, round(score, 1)) != self._said:
                self._said = (state, round(score, 1))
                self.on_state(state, score)


def preview_stream(serial=None, **kw):
    """The phone's screen mirrored (scrcpy's display source): PreviewStream, not started."""
    import live

    class PreviewStream(live.PhoneStream):
        """live.PhoneStream of the phone's screen in place of its camera: the camera app keeps the camera, and its
        preview on the screen is what the photo will be. Nothing on the phone is changed (no rotation lock)."""

        def __init__(self, serial=None, **kw):
            # its tunnel on a port of its own, as every stream's (live.PhoneStream, phone_link.forward_port)
            kw.setdefault("bit_rate", PREVIEW_BIT_RATE)
            super().__init__(serial, **kw)
            self.lock_rotation = False

        def _phone_setup(self):
            """Nothing to put in shape: the screen is mirrored as it is."""

        def _server_command(self):
            args = [f"scid={self.scid:08x}", "log_level=info", "video_source=display", "audio=false", "control=false",
                    "tunnel_forward=true", "send_device_meta=false", "send_dummy_byte=false", "send_stream_meta=true",
                    "send_frame_meta=true", "video_codec=h264", f"video_bit_rate={self.bit_rate}",
                    f"max_size={PREVIEW_SIZE}", f"max_fps={PREVIEW_FPS}", "cleanup=true"]
            if self.codec_options:
                args.append(f"video_codec_options={self.codec_options}")
            return [live.adb_exe(), "-s", self.serial, "shell", f"CLASSPATH={live.SERVER_REMOTE}", "app_process", "/",
                    "com.genymobile.scrcpy.Server", live.SCRCPY_VERSION] + args

    return PreviewStream(serial, **kw)


# ── test ─────────────────────────────────────────────────────────────────────
FAILS = []


def expect(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILS.append(what)


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def as_display(photo: np.ndarray, turn: int, band: tuple[int, int] = (180, 260)) -> np.ndarray:
    """A photo as the camera app shows it on the screen held up, the phone held across (turn 1: its top to the left,
    -1: to the right): the preview turned upright, as wide as the screen, between dark bands of the app."""
    up = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE if turn == 1 else cv2.ROTATE_90_COUNTERCLOCKWISE)
    top, bottom = band
    out = np.full((up.shape[0] + top + bottom, up.shape[1]), 8, np.uint8)
    out[top:top + up.shape[0]] = up
    cv2.putText(out, "12:00", (20, top // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 200, 2)    # the app's own marks
    cv2.circle(out, (out.shape[1] // 2, out.shape[0] - bottom // 2), 50, 230, -1)
    return out


def run_test() -> int:
    boards = calib_guide.boards()
    size = (1024, 768)
    for name, kind, spec, lay in boards:
        rng = np.random.default_rng(3)
        plan = "quick"
        own, mixed, unseen = [], 0, 0
        poses = calib_guide.PLANS[plan]
        board = calibrate_charuco.make_board(spec["square_mm"], spec["squares"], spec.get("marker_mm"))
        wh = calib_guide.board_mm(spec)
        wants = [outline_pose(calib_guide.guide(p, spec)["outline"], wh) for p in poses]
        for i in range(len(poses)):
            img = calib_guide.photo(plan, i, spec, kind, "photo", lay, rng=rng, size=size)
            pts, n = board_outline(img, board)
            seen = outline_pose(in_photo(pts, size, "photo")[0], wh) if pts is not None else None
            if seen is None:
                unseen += 1
                continue
            own.append(pose_error(seen, wants[i]))
            mixed += sum(pose_error(seen, wants[j]) <= 1 for j in range(len(poses)) if j != i)
        expect(unseen == 0 and sum(e <= 1 for e in own) >= len(poses) - 1 and mixed <= 1,
               f"{name}: the board of each pose of the plan (with the operator's slips) taken for its own guide "
               f"({sum(e <= 1 for e in own)} of {len(poses)}, error at most {max(own):.2f}), for another pose's "
               f"{mixed} times")
    # the state over time, on a scripted clock: still frames of the right pose are due once still for HOLD_S, then
    # not again before GAP_S; another guide: far; no board: none
    name, kind, spec, lay = boards[0]
    clock = _Clock()
    w = Watch(clock)
    g0 = calib_guide.guide(calib_guide.PLANS["quick"][0], spec)["outline"]
    img = calib_guide.photo("quick", 0, spec, kind, "photo", lay, rng=np.random.default_rng(5), size=size)
    states = []
    w.target(g0, spec, ("s", 0))
    for _ in range(8):
        states.append(w.feed(img))
        clock.t += 0.17
    first_ready = states.index("ready") if "ready" in states else -1
    w.shot()
    after = []
    for _ in range(int(GAP_S / 0.17) + 6):
        after.append(w.feed(img))
        clock.t += 0.17
    expect(states[0] == "hold" and 0.17 * first_ready >= HOLD_S - 1e-9 and "wait" in after and after[-1] == "ready"
           and after.index("ready") * 0.17 >= GAP_S - 0.2,
           f"still on the guide's pose: hold, ready after {0.17 * first_ready:.2f} s; after a photo wait, ready again "
           f"after {after.index('ready') * 0.17 if 'ready' in after else -1:.2f} s")
    w.target(calib_guide.guide(calib_guide.PLANS["quick"][3], spec)["outline"], spec, ("s", 3))
    far_state = w.feed(img)
    w.target(g0, spec, ("s", 0))
    none_state = w.feed(np.full((768, 1024), 120, np.uint8))
    shaky = []
    for k in range(10):
        moved = calib_guide.photo("quick", 0, spec, kind, "photo", lay, rng=np.random.default_rng(100 + k), size=size)
        shaky.append(w.feed(moved))
        clock.t += 0.17
    expect(far_state == "far" and none_state == "none" and "ready" not in shaky,
           f"another pose's guide: far; no board: none; the phone moving between looks: never ready ({set(shaky)})")
    # the mirrored screen: the preview found between the app's bands, either turn of the phone held across
    for turn in (1, -1):
        disp = as_display(img, turn)
        x, y, rw, rh = preview_rect(disp)
        w2 = Watch(clock)
        w2.target(g0, spec, ("d", turn))
        st = [w2.feed(disp, "display") for _ in range(1)]
        for _ in range(6):
            clock.t += 0.17
            st.append(w2.feed(disp, "display"))
        expect(abs(y - 180) <= 2 and abs(rh - img.shape[1]) <= 2 and st[-1] == "ready" and w2.error <= 1,
               f"the phone's screen, held across with its top to the {'left' if turn == 1 else 'right'}: the preview "
               f"at y {y:.0f} ({rh:.0f} px tall), the board matched (error {w2.error:.2f}), ready")
    bare = np.full((1600, 720), 90, np.uint8)
    expect(preview_rect(bare) == (0.0, (1600 - 960) / 2, 720.0, 960.0),
           "a screen without the app's bands: the preview taken as centred")
    # the thread: a source of still frames fires once, then waits for the gap
    fired = []
    watcher = Watcher(lambda: (img, 1, "photo"), lambda: fired.append(time.monotonic()),
                      watch=Watch(gap_s=0.8, hold_s=0.2))
    watcher.target(g0, spec, ("t", 0))
    watcher.start()
    t0 = time.monotonic()
    while len(fired) < 1 and time.monotonic() - t0 < 5:
        time.sleep(0.05)
    watcher.shot()
    while len(fired) < 2 and time.monotonic() - t0 < 8:
        time.sleep(0.05)
    watcher.stop()
    expect(len(fired) == 2 and fired[1] - fired[0] >= 0.8, f"on a thread: due, taken, due again after the gap "
                                                           f"({len(fired)} photos)")
    print(f"\n{len(FAILS)} failures" if FAILS else "\nall checks passed", flush=True)
    return 1 if FAILS else 0


def cmd_watch(a) -> int:
    """The phone's screen mirrored and looked at against the first pose of the quick plan of the A4 sheet."""
    _, _, spec, _ = calib_guide.boards()[2] if a.board == "sheet" else calib_guide.boards()[0]
    s = preview_stream(a.serial)
    s.start()
    w = Watch()
    w.target(calib_guide.guide(calib_guide.PLANS["quick"][a.pose], spec)["outline"], spec)
    try:
        while True:
            img, _, n = s.latest()
            if img is not None:
                st = w.feed(img, "display")
                print(f"{n:6d} {st:5s} corners {w.corners:3d} error {w.error if w.error is not None else '-'}",
                      flush=True)
            time.sleep(1 / LOOK_HZ)
    except KeyboardInterrupt:
        return 0
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    a = sub.add_parser("watch")
    a.add_argument("--serial")
    a.add_argument("--board", choices=("screen", "sheet"), default="screen")
    a.add_argument("--pose", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "test":
        return run_test()
    return cmd_watch(a)


if __name__ == "__main__":
    sys.exit(main())
