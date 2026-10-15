"""Temporal filter of the board pose for the live AR (T20.3): what the overlay draws on each frame of the stream.

The board pose from one frame (marker_pose.py, PnP on the ChArUco corners) shakes by a fraction of a pixel from frame
to frame, now and then comes out wrong on a frame with few corners, and is missing on frames where the board is
covered or blurred. Drawn as it comes, the overlay trembles and blinks. PoseTracker turns the poses of the processed
frames into a calm pose for every frame shown:

  filter    One Euro filter (Casiez, Roussel and Vogel, CHI 2012) on SE(3): the position of an anchor (the centre of
            the model, so that a rotation of the board does not move the anchor) and the rotation, moved towards each
            new pose by the same fraction, the rotation along the geodesic (slerp). The cutoff grows with the speed of
            the model in the image (px/s: anchor speed plus model radius times angular speed, times f / z): strong
            smoothing while the phone is still, little smoothing and so little lag while it moves.
  gate      a pose that would move the model in the image by more than the gate from the predicted one is a jump.
            A jump measured on many corners at a low reprojection error is believed at once (the filter restarts from
            it); a jump on few corners is rejected, unless the next weak pose confirms it.
  hold      when the board is lost the last pose stays drawn for hold_s (predicted for at most predict_s, then still),
            then fades out over fade_s; a single missed frame changes nothing. After a loss the overlay comes back
            only after acquire_frames poses in a row, fading in over fade_in_s.

    tr = PoseTracker(settings, anchor, radius_mm, f_px)     # settings as live_ar.TRACKING
    tr.measure(t, T, strong)        # each processed frame: T the PnP pose (None if not found), t in seconds
    T, vis, state = tr.pose_at(t)   # the pose to draw on the frame taken at t, its opacity 0-1 and the state

    python pose_filter.py test      deterministic checks of filter, gate, hold and fade; exit 1 on failure
"""
import sys

import cv2
import numpy as np

STATES = ("lost", "acquiring", "tracking", "holding", "fading")


def rotvec(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def rotmat(w):
    return cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))[0]


def smoothing(cutoff_hz, dt):
    """Weight of the new sample in an exponential smoother of cutoff frequency cutoff_hz sampled every dt."""
    tau = 1.0 / (2 * np.pi * max(cutoff_hz, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroPose:
    """One Euro filter on a rigid transform T (4 x 4, camera from board, mm)."""

    def __init__(self, min_cutoff_hz, beta, d_cutoff_hz, anchor, radius_mm, f_px, lead=True, still_px_s=0.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff_hz), float(beta), float(d_cutoff_hz)
        self.anchor = np.asarray(anchor, np.float64).reshape(3)
        self.radius, self.f = float(radius_mm), float(f_px)
        self.lead, self.still = bool(lead), float(still_px_s)
        self.t = None

    def reset(self, T, t):
        self.R = np.array(T[:3, :3], np.float64)
        self.p = self.R @ self.anchor + T[:3, 3]
        self.v, self.w = np.zeros(3), np.zeros(3)      # anchor velocity (mm/s, camera), angular velocity (rad/s, body)
        self.t = float(t)

    def speed(self):
        """Speed of the model in the image, px/s."""
        return self.f / max(self.p[2], 1.0) * (np.linalg.norm(self.v) + self.radius * np.linalg.norm(self.w))

    def update(self, T, t):
        if self.t is None:
            self.reset(T, t)
            return
        dt = max(float(t) - self.t, 1e-3)
        R_m = np.asarray(T[:3, :3], np.float64)
        p_m = R_m @ self.anchor + T[:3, 3]
        dw = rotvec(self.R.T @ R_m)
        a_d = smoothing(self.d_cutoff, dt)
        self.v += a_d * ((p_m - self.p) / dt - self.v)
        self.w += a_d * (dw / dt - self.w)
        a = smoothing(self.min_cutoff + self.beta * self.speed(), dt)
        if self.lead and self.speed() >= self.still:
            # blend towards the pose moved on by the filtered velocity: no lag at a steady speed
            R0, p0 = self.R @ rotmat(self.w * dt), self.p + self.v * dt
        else:
            R0, p0 = self.R, self.p
        self.p = p0 + a * (p_m - p0)
        self.R = R0 @ rotmat(a * rotvec(R0.T @ R_m))
        self.t = float(t)

    def predict(self, t, horizon=None, still_px_s=0.0):
        """T at time t, extrapolated with the filtered velocities for at most `horizon` seconds past the last pose;
        not extrapolated while the model moves slower than still_px_s in the image (the speed of noise)."""
        dt = float(t) - self.t
        if horizon is not None:
            dt = min(dt, horizon)
        dt = max(dt, 0.0)
        if still_px_s > 0 and self.speed() < still_px_s:
            dt = 0.0
        R = self.R @ rotmat(self.w * dt)
        p = self.p + self.v * dt
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p - R @ self.anchor
        return T


def image_shift(T_a, T_b, anchor, radius, f):
    """How far (px) the model moves in the image between poses T_a and T_b: anchor plus radius times the rotation."""
    pa = T_a[:3, :3] @ anchor + T_a[:3, 3]
    pb = T_b[:3, :3] @ anchor + T_b[:3, 3]
    ang = np.linalg.norm(rotvec(T_a[:3, :3].T @ T_b[:3, :3]))
    return f / max(pa[2], 1.0) * (np.linalg.norm(pa - pb) + radius * ang)


class PoseTracker:
    """Gate, One Euro filter, hold and fade (see the module docstring). Not thread safe: the caller locks."""

    def __init__(self, settings, anchor=(0, 0, 0), radius_mm=100.0, f_px=1439.0):
        self.s = dict(settings)
        self.anchor = np.asarray(anchor, np.float64).reshape(3)
        self.radius, self.f = float(radius_mm), float(f_px)
        self.filter = OneEuroPose(self.s["min_cutoff_hz"], self.s["beta"], self.s["d_cutoff_hz"], self.anchor,
                                  self.radius, self.f, self.s.get("lead", True), self.s["still_px_s"])
        self.counts = {"accepted": 0, "rejected": 0, "missed": 0, "restarted": 0}
        self.clear()

    def clear(self):
        self.filter.t = None
        self.t_good = None          # time of the last accepted pose
        self.streak = 0             # accepted poses in a row
        self.shown = False          # the overlay is on (with hysteresis)
        self.candidate = None       # (T, t) of a rejected weak jump, waiting for the next weak pose
        self.vis, self.t_vis = 0.0, None
        self.last_raw = None        # (T, t) of the last accepted pose, unfiltered

    # ── the processed frames ─────────────────────────────────────────────
    def measure(self, t, T=None, strong=False):
        """A processed frame taken at t: T the PnP pose that passed the RMS limit, None when the board was not found.
        strong: many corners and a low reprojection error. Returns accepted, restarted, rejected or missed."""
        if T is None:
            self.streak = 0
            self.counts["missed"] += 1
            return "missed"
        s = self.s
        if not s.get("smooth", True):
            self.filter.reset(T, t)
            return self._accept(T, t, "accepted")
        f = self.filter
        if f.t is None or self.t_good is None or t - self.t_good > s["hold_s"] + s["fade_s"]:
            f.reset(T, t)
            return self._accept(T, t, "restarted")
        if t <= f.t:            # an older frame than the filter's: it comes too late
            return "rejected"
        pred = f.predict(t, s["predict_s"], s["still_px_s"])
        gate = s["gate_px"] + 2.0 * f.speed() * min(t - f.t, s["predict_s"])
        if image_shift(T, pred, self.anchor, self.radius, self.f) > gate:
            cand, self.candidate = self.candidate, (T, t)
            confirmed = (cand is not None and t - cand[1] <= 2 * s["predict_s"]
                         and image_shift(T, cand[0], self.anchor, self.radius, self.f) < gate)
            if strong or confirmed:
                f.reset(T, t)
                return self._accept(T, t, "restarted")
            self.counts["rejected"] += 1
            return "rejected"
        f.update(T, t)
        return self._accept(T, t, "accepted")

    def _accept(self, T, t, kind):
        self.candidate = None
        self.t_good = t
        self.last_raw = (T, t)
        self.streak += 1
        if self.streak >= self.s["acquire_frames"] or self.shown:
            self.shown = True
        self.counts[kind] += 1
        return kind

    # ── the frames shown ─────────────────────────────────────────────────
    def prior(self, t):
        """The predicted pose at t for the search of the next frame (None when lost)."""
        if self.filter.t is None or self.t_good is None or t - self.t_good > self.s["hold_s"]:
            return None
        return self.filter.predict(t, self.s["predict_s"], self.s["still_px_s"])

    def pose_at(self, t):
        """(T, visibility 0-1, state) for the frame taken at t: T None when nothing is to be drawn. Frames are asked for
        in time order: the visibility moves with the time between them."""
        s = self.s
        age = None if self.t_good is None else t - self.t_good
        on = self.shown and age is not None and age <= s["hold_s"]
        dt = 0.0 if self.t_vis is None else max(t - self.t_vis, 0.0)
        self.t_vis = t
        if on:
            self.vis = min(1.0, self.vis + dt / max(s["fade_in_s"], 1e-6))
        else:
            self.vis = max(0.0, self.vis - dt / max(s["fade_s"], 1e-6))
            if self.vis == 0.0 and self.shown and age is not None and age > s["hold_s"]:
                self.shown = False
        if self.vis <= 0.0 or self.filter.t is None:
            return None, 0.0, "acquiring" if self.streak else "lost"
        T = self.filter.predict(t, s["predict_s"], s["still_px_s"])
        if on:
            state = "tracking" if age <= max(s["predict_s"], 1e-9) else "holding"
        else:
            state = "fading"
        return T, self.vis, state

    @property
    def tracked(self):
        """The light: on while tracking or holding, off once the hold is over."""
        return bool(self.shown and self.t_good is not None and self.t_vis is not None
                    and self.t_vis - self.t_good <= self.s["hold_s"])


# ── checks ───────────────────────────────────────────────────────────────────
def _pose(rx, ry, rz, x, y, z):
    T = np.eye(4)
    T[:3, :3] = rotmat(np.radians([rx, ry, rz]))
    T[:3, 3] = [x, y, z]
    return T


def run_test(settings=None):
    """The filter on scripted pose sequences at 30 fps with known noise: jitter while still, lag while moving, an
    outlier, single missed frames, a hand over the board, the fade and the comeback."""
    if settings is None:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        from live_ar import TRACKING as settings
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    rng = np.random.default_rng(1)
    anchor, radius, f = np.array([100.0, 70.0, -60.0]), 80.0, 1439.0
    dt = 1 / 30

    def noisy(T, rot_deg=0.03, trans_mm=0.1):
        N = _pose(*rng.normal(0, rot_deg, 3), *rng.normal(0, trans_mm, 3))
        return T @ N

    def shift(Ta, Tb):
        return image_shift(Ta, Tb, anchor, radius, f)

    # still: jitter of the drawn model, raw against filtered
    base = _pose(160, 5, 3, -90, -60, 480)
    tr = PoseTracker(settings, anchor, radius, f)
    raw, out = [], []
    for k in range(150):
        t = k * dt
        m = noisy(base)
        tr.measure(t, m, strong=True)
        T, vis, state = tr.pose_at(t)
        if k >= 30:
            raw.append(shift(m, base))
            out.append(shift(T, base))
    raw, out = np.sqrt(np.mean(np.square(raw))), np.sqrt(np.mean(np.square(out)))
    expect(out < 0.4 * raw, f"still: model jitter {raw:.2f} px raw, {out:.2f} px drawn (RMS, f {f:.0f} px)")

    # moving at 300 px/s then 900 px/s: lag of the drawn model behind the truth
    for speed_px in (300.0, 900.0):
        tr = PoseTracker(settings, anchor, radius, f)
        v = speed_px * 480 / f       # mm/s sideways at 480 mm
        err = []
        for k in range(120):
            t = k * dt
            truth = _pose(160, 5, 3, -90 + v * t, -60, 480)
            tr.measure(t, noisy(truth), strong=True)
            T, vis, _ = tr.pose_at(t)
            if k >= 30:
                err.append(shift(T, truth))
        e = np.sqrt(np.mean(np.square(err)))
        expect(e < 0.015 * speed_px + 0.5, f"moving at {speed_px:.0f} px/s: drawn model behind the truth by "
                                           f"{e:.2f} px RMS")

    # a weak outlier is rejected, a strong jump is followed at once, two weak poses that agree are believed
    tr = PoseTracker(settings, anchor, radius, f)
    for k in range(30):
        tr.measure(k * dt, noisy(base), strong=True)
        tr.pose_at(k * dt)
    far = _pose(160, 5, 3, -40, -60, 480)          # 50 mm sideways: about 150 px
    r1 = tr.measure(30 * dt, far, strong=False)
    T, _, _ = tr.pose_at(30 * dt)
    r2 = tr.measure(31 * dt, noisy(base), strong=True)
    expect(r1 == "rejected" and r2 == "accepted" and shift(T, base) < 1.0,
           f"weak single-frame jump rejected ({r1}), the overlay stays ({shift(T, base):.2f} px)")
    r3 = tr.measure(32 * dt, far, strong=True)
    T, _, _ = tr.pose_at(32 * dt)
    expect(r3 == "restarted" and shift(T, far) < 1.0, f"strong jump followed at once ({r3})")
    tr.measure(33 * dt, base, strong=False)
    r4 = tr.measure(34 * dt, noisy(base), strong=False)
    expect(r4 == "restarted", f"two weak poses in a row that agree are believed ({r4})")

    # missed frames: every third frame lost, the overlay never blinks
    tr = PoseTracker(settings, anchor, radius, f)
    vis, lights = [], []
    for k in range(90):
        t = k * dt
        tr.measure(t, None if k % 3 == 2 else noisy(base), strong=True)
        _, v, _ = tr.pose_at(t)
        if k >= 10:
            vis.append(v)
            lights.append(tr.tracked)
    expect(min(vis) == 1.0 and all(lights), f"one frame in three missed: overlay always at full opacity "
                                            f"(min {min(vis):.2f}), light always on")

    # a hand over the board for 0.4 s: held; for 2 s: held, then faded out, then back after acquire_frames
    for cover in (0.4, 2.0):
        tr = PoseTracker(settings, anchor, radius, f)
        rows = []
        for k in range(int((1.0 + cover + 1.0) / dt)):
            t = k * dt
            hidden = 1.0 <= t < 1.0 + cover
            tr.measure(t, None if hidden else noisy(base), strong=True)
            T, v, st = tr.pose_at(t)
            rows.append((t, v, st, tr.tracked, T is not None and shift(T, base) < 2.0))
        seen = 1.0 - dt                 # the last frame with the board
        hold, fade = settings["hold_s"], settings["fade_s"]
        during = [r for r in rows if 1.0 <= r[0] < 1.0 + cover]
        held = [r for r in during if r[0] - seen < hold - 1e-6]
        after = [r for r in rows if r[0] >= 1.0 + cover + 0.4]
        if cover < hold:
            ok = all(r[1] == 1.0 and r[3] and r[4] for r in during) and all(r[1] == 1.0 for r in after)
            expect(ok, f"board covered {cover:.1f} s: the face stays where it was, light on, no fade")
        else:
            gone = [r for r in during if r[0] - seen >= hold + fade + dt - 1e-6]
            back = next((r[0] - 1.0 - cover for r in rows if r[0] >= 1.0 + cover and r[1] > 0), None)
            fading = [r[1] for r in during if hold + 1e-6 < r[0] - seen < hold + fade - dt]
            ok = (all(r[1] == 1.0 and r[4] for r in held) and gone and all(r[1] == 0 and not r[3] for r in gone)
                  and fading and all(0 < v < 1 for v in fading) and np.all(np.diff(fading) < 0)
                  and back is not None and back >= (settings["acquire_frames"] - 1) * dt - 1e-6
                  and all(r[1] == 1.0 for r in after))
            expect(ok, f"board covered {cover:.1f} s: held {hold:.2f} s, faded over {fade:.2f} s in {len(fading)} "
                       f"frames, back {back * 1e3 if back is not None else -1:.0f} ms after the hand left")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(run_test())
    sys.exit(__doc__)
