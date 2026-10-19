"""Temporal filter of the board pose for the live AR: what the overlay draws on each frame of the stream.

The board pose from one frame (marker_pose.py, PnP on the ChArUco corners) shakes by a fraction of a pixel from frame
to frame, now and then comes out wrong on a frame with few corners, and is missing on frames where the board is
covered or blurred. Drawn as it comes, the overlay trembles and blinks. PoseTracker turns the poses of the processed
frames into a calm pose for every frame shown:

  filter    One Euro filter (Casiez, Roussel and Vogel, CHI 2012) on SE(3), in two channels: the position of an
            anchor (the board's centre) and the rotation (along the geodesic, slerp). Each cutoff grows with that
            channel's speed in the image (px/s: the anchor's; the model's lever times the angular speed, times f / z)
            beyond a dead band (still_px_s, the speed that noise alone gives): strong smoothing while the phone is
            still, almost none, and so almost no lag, as soon as it moves; moving, the pose is first carried on by
            the filtered velocity (the lead), so a steady motion is not left behind either. The rotation of a small
            board is noisier (the tracker says how much, noise_px): its dead band is wider and its smoothing at rest
            stronger, while the position keeps following the hand.
  gate      a pose that would move the model in the image by more than the gate from the predicted one is a jump.
            A jump measured on many corners at a low reprojection error is believed at once (the filter restarts from
            it, gliding there over glide_s instead of jumping); a jump on few corners is rejected, unless the next
            weak pose confirms it.
  hold      when the board is lost the last pose stays drawn for hold_s (predicted for at most predict_s, then still),
            then fades out over fade_s; a single missed frame changes nothing. After a loss the overlay comes back
            only after acquire_frames poses in a row, fading in over fade_in_s.

    tr = PoseTracker(settings, anchor, radius_mm, f_px)     # settings: overrides of DEFAULTS (live_ar.TRACKING);
                                                            # anchor: the board's centre, radius: how far the model
                                                            # reaches from it
    tr.measure(t, T, strong, noise_px)  # each processed frame: T the PnP pose (None if not found), t in seconds
    T, vis, state = tr.pose_at(t)   # the pose to draw on the frame taken at t, its opacity 0-1 and the state

The settings are DEFAULTS (the one place their values live; live_ar.TRACKING starts from them) with the caller's
overrides. The expected noise of a pose is per frame; the speed it gives is that noise over the interval between
frames, measured on the frames themselves (the median of the last FRAME_DT_WINDOW intervals), so a 10 fps replay or a
link that drops frames is smoothed as much as a 30 fps stream, not three times more.

    python pose_filter.py test      deterministic checks of filter, gate, hold and fade; exit 1 on failure
"""
from __future__ import annotations

import sys
from collections import deque

import cv2
import numpy as np

STATES = ("lost", "acquiring", "tracking", "holding", "fading")
DEFAULTS = {
    "smooth": True,           # False: every frame at its own pose as it comes (the tests of the geometry)
    "min_cutoff_hz": 2.0,     # smoothing while still: lower is calmer and slower to settle
    "beta": 0.25,             # cutoff added per px/s of the model's speed past the dead band: higher lags less
    "d_cutoff_hz": 2.0,       # smoothing of that speed
    "lead": True,             # moving, the filter carries the pose on by its speed first: no lag at a steady speed
    "still_px_s": 8.0,        # dead band at least this (px/s at the stream size): slower is noise, not motion
    "predict_s": 0.3,         # a frame without its own pose is drawn at the pose carried on, at most this far,
    "predict_tau_s": 0.12,    #   the speed dying away with this time constant (no freeze, no running off)
    "glide_gap_s": 0.1,       # a jump after this long without the board is taken at once (no glide from a stale pose)
    "gate_px": 40.0,          # a pose moving the model more than this (px, stream size) from the prediction is a jump,
    "glide_s": 0.15,          #   believed (when strong or confirmed) by gliding there over this time
    "noise_k": 2.0,           # each channel's dead band: this many times the speed its expected noise gives
    "noise_ref_px": 0.5,      # the lowest cutoff scales with noise_ref_px over the expected noise (px per frame),
    "min_cutoff_floor": 0.4,  #   down to this share for a noisy measure (a small board far away)
    "min_cutoff_cap": 3.0,    #   and up to this many times for a precise one: the overlay stays glued to the board
    "hold_s": 0.7,            # the board lost: the last pose stays drawn this long
    "fade_s": 0.35,           # then fades out over this time
    "fade_in_s": 0.1,         # and fades in when the board is found again,
    "acquire_frames": 2,      #   after this many poses in a row
}
FRAME_DT_S = 1 / 30           # interval between frames until it has been measured (s)
FRAME_DT_RANGE_S = (1 / 240, 0.5)   # a measured interval is held within this (a frame rate from 240 to 2 fps)
FRAME_DT_WINDOW = 15          # frames over which the interval is measured (median)


def rotvec(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def rotmat(w):
    return cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))[0]


def smoothing(cutoff_hz, dt):
    """Weight of the new sample in an exponential smoother of cutoff frequency cutoff_hz sampled every dt."""
    tau = 1.0 / (2 * np.pi * max(cutoff_hz, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroPose:
    """One Euro filter on a rigid transform T (4 x 4, camera from board, mm), in two channels: the position of the
    anchor (the board's centre) and the rotation (which a small board gives far less precisely, and which the lever of
    the model, radius mm from the anchor, magnifies in the image). Each channel has its own cutoff from its own speed
    in the image (px/s: the anchor's; radius times the angular speed) past its own dead band, and both follow the noise
    the tracker expects on this frame (noise_px for the rotation at the model, noise_pos_px for the anchor, px per
    frame): a precise measure (many corners, a large board) is followed almost as it comes, so the overlay stays glued
    to the board even when the hand trembles; a noisy one (a small board far away) is smoothed while still."""

    def __init__(self, settings, anchor, radius_mm, f_px):
        self.s = {**DEFAULTS, **settings}
        self.anchor = np.asarray(anchor, np.float64).reshape(3)
        self.radius, self.f = float(radius_mm), float(f_px)
        self.t = None
        self.noise = None           # expected noise of the rotation at the model, px per frame (smoothed)
        self.noise_pos = None       # and of the anchor
        self.frame_dt = FRAME_DT_S  # interval between frames (s): what turns a noise per frame into a speed

    def reset(self, T, t):
        self.R = np.array(T[:3, :3], np.float64)
        self.p = self.R @ self.anchor + T[:3, 3]
        self.v, self.w = np.zeros(3), np.zeros(3)      # anchor velocity (mm/s, camera), angular velocity (rad/s, body)
        self.t = float(t)

    def speeds(self):
        """Speed in the image (px/s) of the anchor and, from the rotation alone, of the model at the radius."""
        k = self.f / max(self.p[2], 1.0)
        return k * np.linalg.norm(self.v), k * self.radius * np.linalg.norm(self.w)

    def speed(self):
        return sum(self.speeds())

    def _channel(self, noise, dt):
        """(dead band px/s, lowest cutoff Hz) of a channel whose measure has this noise (px per frame; None: unknown)."""
        s = self.s
        if noise is None:
            return s["still_px_s"], s["min_cutoff_hz"]
        ratio = s["noise_ref_px"] / max(noise, 1e-6)
        cut = s["min_cutoff_hz"] * float(np.clip(ratio, s["min_cutoff_floor"], s["min_cutoff_cap"]))
        return max(s["still_px_s"], s["noise_k"] * noise / max(dt, 1e-3)), cut

    def dead_bands(self, dt=None):
        """(anchor, rotation) dead bands, px/s, for frames dt apart (None: the measured frame interval)."""
        dt = self.frame_dt if dt is None else dt
        return self._channel(self.noise_pos, dt)[0], self._channel(self.noise, dt)[0]

    def update(self, T, t, noise_px=None, noise_pos_px=None):
        def mix(old, new):
            return None if new is None else float(new) if old is None else 0.8 * old + 0.2 * float(new)
        if self.t is None:
            self.reset(T, t)
            self.noise, self.noise_pos = mix(None, noise_px), mix(None, noise_pos_px)
            return
        s = self.s
        dt = max(float(t) - self.t, 1e-3)
        self.noise = mix(self.noise, noise_px) if noise_px is not None else self.noise
        self.noise_pos = mix(self.noise_pos, noise_pos_px) if noise_pos_px is not None else self.noise_pos
        R_m = np.asarray(T[:3, :3], np.float64)
        p_m = R_m @ self.anchor + T[:3, 3]
        dw = rotvec(self.R.T @ R_m)
        a_d = smoothing(s["d_cutoff_hz"], dt)
        self.v += a_d * ((p_m - self.p) / dt - self.v)
        self.w += a_d * (dw / dt - self.w)
        v_px, w_px = self.speeds()
        dead_t, min_t = self._channel(self.noise_pos, self.frame_dt)
        dead_r, min_r = self._channel(self.noise, self.frame_dt)
        a_t = smoothing(min_t + s["beta"] * max(0.0, v_px - dead_t), dt)
        a_r = smoothing(min_r + s["beta"] * max(0.0, w_px - dead_r), dt)
        lead = s["lead"]
        # moving, blend towards the pose carried on by the filtered velocity: no lag at a steady speed
        p0 = self.p + self.v * dt if lead and v_px >= dead_t else self.p
        R0 = self.R @ rotmat(self.w * dt) if lead and w_px >= dead_r else self.R
        self.p = p0 + a_t * (p_m - p0)
        self.R = R0 @ rotmat(a_r * rotvec(R0.T @ R_m))
        self.t = float(t)

    def predict(self, t, horizon=None, still_px_s=0.0, tau=None):
        """T at time t, carried on with the filtered velocities for at most `horizon` seconds past the last pose, the
        velocity dying away with time constant tau (s; None: kept); a channel moving slower than its dead band (the
        speed of noise) is not carried on."""
        dt = float(t) - self.t
        if horizon is not None:
            dt = min(dt, horizon)
        dt = max(dt, 0.0)
        if tau:
            dt = tau * (1.0 - np.exp(-dt / tau))
        v_px, w_px = self.speeds()
        dead_t, dead_r = self.dead_bands()
        R = self.R @ rotmat(self.w * dt) if w_px >= dead_r else self.R
        p = self.p + self.v * dt if v_px >= dead_t else self.p
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
    """Gate, One Euro filter, hold and fade (see the module docstring). Not thread safe: one thread uses it (the live
    AR's track thread), and what other threads need of it is copied out by that thread.

    Args:
        settings: overrides of DEFAULTS (unknown keys are kept and ignored).
        anchor: the point of the board the position channel follows (board frame, mm).
        radius_mm: how far the model reaches from the anchor (the lever of the rotation).
        f_px: focal length (px), for the speeds in the image.
    """

    def __init__(self, settings=None, anchor=(0, 0, 0), radius_mm=100.0, f_px=1439.0):
        self.s = {**DEFAULTS, **(settings or {})}
        self.anchor = np.asarray(anchor, np.float64).reshape(3)
        self.radius, self.f = float(radius_mm), float(f_px)
        self.filter = OneEuroPose(self.s, self.anchor, self.radius, self.f)
        self.counts = {"accepted": 0, "rejected": 0, "missed": 0, "restarted": 0, "late": 0}
        self._t_frame = None                            # capture time of the last frame measured
        self._intervals = deque(maxlen=FRAME_DT_WINDOW)
        self.clear()

    def clear(self):
        self.filter.t = None
        self.t_good = None          # time of the last accepted pose
        self.streak = 0             # accepted poses in a row
        self.shown = False          # the overlay is on (with hysteresis)
        self.candidate = None       # (T, t) of a rejected weak jump, waiting for the next weak pose
        self.vis, self.t_vis = 0.0, None
        self.last_raw = None        # (T, t) of the last accepted pose, unfiltered
        self.glide = None           # (t, anchor offset, rotation offset): the overlay glides from where it was

    # ── the processed frames ─────────────────────────────────────────────
    def measure(self, t, T=None, strong=False, noise_px=None, noise_pos_px=None):
        """A processed frame taken at t: T the PnP pose that passed the RMS limit, None when the board was not found.
        strong: many corners and a low reprojection error. noise_px, noise_pos_px: the noise the tracker expects on
        the model from the rotation and on the anchor (px per frame). Returns accepted, restarted, rejected or
        missed (a frame older than the filter's is rejected and counted as late)."""
        self._frame_interval(t)
        if T is None:
            self.streak = 0
            self.counts["missed"] += 1
            return "missed"
        s = self.s
        if not s["smooth"]:
            self.filter.reset(T, t)
            return self._accept(T, t, "accepted")
        f = self.filter
        if f.t is None or self.t_good is None or t - self.t_good > s["hold_s"] + s["fade_s"]:
            f.reset(T, t)
            f.noise, f.noise_pos = noise_px, noise_pos_px
            self.glide = None
            return self._accept(T, t, "restarted")
        if t <= f.t:            # an older frame than the filter's: it comes too late
            self.counts["late"] += 1
            return "rejected"
        pred = self._predicted(t)
        gate = s["gate_px"] + 2.0 * f.speed() * min(t - f.t, s["predict_s"])
        if image_shift(T, pred, self.anchor, self.radius, self.f) > gate:
            cand, self.candidate = self.candidate, (T, t)
            confirmed = (cand is not None and t - cand[1] <= 2 * s["predict_s"]
                         and image_shift(T, cand[0], self.anchor, self.radius, self.f) < gate)
            if strong or confirmed:
                # a jump while the board was followed (a flip, a correction): glide there; after frames without the
                # board the drawn pose is stale: go there at once
                if t - self.t_good > s["glide_gap_s"]:
                    f.reset(T, t)
                    f.noise, f.noise_pos = noise_px, noise_pos_px
                    self.glide = None
                else:
                    self._restart_gliding(T, t)
                return self._accept(T, t, "restarted")
            self.counts["rejected"] += 1
            return "rejected"
        f.update(T, t, noise_px, noise_pos_px)
        return self._accept(T, t, "accepted")

    def _frame_interval(self, t):
        """The interval between frames from the capture times of the frames measured (median, held in range), for the
        filter's dead bands."""
        if self._t_frame is not None and t > self._t_frame:
            self._intervals.append(t - self._t_frame)
            self.filter.frame_dt = float(np.clip(np.median(self._intervals), *FRAME_DT_RANGE_S))
        if self._t_frame is None or t > self._t_frame:
            self._t_frame = t

    def _restart_gliding(self, T, t):
        """The filter starts again from T; while the overlay is shown it glides there from where it was drawn."""
        f = self.filter
        old = self._predicted(t) if f.t is not None and self.vis > 0 else None
        f.reset(T, t)
        gs = self.s["glide_s"]
        if old is None or gs <= 0:
            self.glide = None
            return
        p_old = old[:3, :3] @ self.anchor + old[:3, 3]
        self.glide = (t, p_old - f.p, rotvec(old[:3, :3] @ f.R.T))

    def _glided(self, T, t):
        if self.glide is None:
            return T
        t0, dp, dw = self.glide
        u = (t - t0) / max(self.s["glide_s"], 1e-6)
        if u >= 1.0:
            self.glide = None
            return T
        k = 1.0 - max(u, 0.0)
        k = k * k * (3 - 2 * k)         # smooth start and end
        R = rotmat(k * dw) @ T[:3, :3]
        p = T[:3, :3] @ self.anchor + T[:3, 3] + k * dp
        out = np.eye(4)
        out[:3, :3] = R
        out[:3, 3] = p - R @ self.anchor
        return out

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
        return self._predicted(t)

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
        T = self._glided(self._predicted(t), t)
        if on:
            state = "tracking" if age <= max(s["predict_s"], 1e-9) else "holding"
        else:
            state = "fading"
        return T, self.vis, state

    def _predicted(self, t):
        """The filtered pose carried on to t: at full speed for the frames between poses, then (the board lost while
        moving) slowing down over predict_tau_s, so the overlay neither freezes nor runs off."""
        s = self.s
        return self.filter.predict(t, s["predict_s"], s["still_px_s"], s["predict_tau_s"])

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
    outlier, single missed frames, a hand over the board, the fade and the comeback; the defaults filling missing
    settings; the frame interval measured (a 10 fps stream). settings: overrides of DEFAULTS (None: DEFAULTS)."""
    settings = {**DEFAULTS, **(settings or {})}
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    rng = np.random.default_rng(1)
    anchor, radius, f = np.array([100.0, 70.0, -60.0]), 80.0, 1439.0
    dt = 1 / 30

    def noisy(T, rot_deg=0.03, trans_mm=0.1, about=None):
        """T with noise: a small rotation (about the board origin, or about the given point, where the corners
        are) and a small shift."""
        N = _pose(*rng.normal(0, rot_deg, 3), *rng.normal(0, trans_mm, 3))
        if about is not None:
            N[:3, 3] += about - N[:3, :3] @ about
        return T @ N

    def shift(Ta, Tb):
        return image_shift(Ta, Tb, anchor, radius, f)

    # still: jitter of the drawn model, raw against filtered; a large board (noise mostly in the position) and a
    # small one (noise mostly in the rotation, which the tracker reports: noise_px)
    base = _pose(160, 5, 3, -90, -60, 480)
    for name, rot, trans, limit in (("large board", 0.03, 0.1, 0.5), ("small board", 0.4, 0.1, 0.3)):
        tr = PoseTracker(settings, anchor, radius, f)
        noise_px = f / 480 * radius * np.radians(rot) * np.sqrt(2)     # rotation noise at the model, px per frame
        raw, out = [], []
        for k in range(150):
            t = k * dt
            m = noisy(base, rot, trans, about=anchor)
            tr.measure(t, m, strong=True, noise_px=noise_px if rot > 0.1 else None)
            T, vis, state = tr.pose_at(t)
            if k >= 30:
                raw.append(shift(m, base))
                out.append(shift(T, base))
        raw, out = np.sqrt(np.mean(np.square(raw))), np.sqrt(np.mean(np.square(out)))
        expect(out < limit * raw, f"still, {name}: model jitter {raw:.2f} px raw, {out:.2f} px drawn (RMS, f {f:.0f} px)")

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
    glide = settings["glide_s"]
    path = []
    for k in range(int(np.ceil(glide / dt)) + 2):
        if k:
            tr.measure((32 + k) * dt, far, strong=True)
        T, _, _ = tr.pose_at((32 + k) * dt)
        path.append(shift(T, far))
    smooth = all(b <= a + 1e-6 for a, b in zip(path, path[1:])) and max(np.diff(path), default=0) <= 0
    expect(r3 == "restarted" and path[-1] < 1.0 and smooth,
           f"strong jump followed ({r3}), gliding there over {glide:.2f} s: "
           + ", ".join(f"{v:.0f}" for v in path) + " px from the new pose")
    tr.measure((34 + len(path)) * dt, base, strong=False)
    r4 = tr.measure((35 + len(path)) * dt, noisy(base), strong=False)
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
    # the defaults fill what the settings leave out: no hidden second set of values
    tr = PoseTracker({}, anchor, radius, f)
    expect(tr.s == DEFAULTS and tr.filter.s == DEFAULTS and PoseTracker(None).s == DEFAULTS,
           "missing settings taken from DEFAULTS (one source of the filter's values)")

    # the frame interval is measured: at 10 fps a noise per frame is a speed three times lower than at 30 fps
    noise_px = 1.2
    bands = {}
    for fps in (30.0, 10.0):
        tr = PoseTracker(settings, anchor, radius, f)
        for k in range(40):
            tr.measure(k / fps, noisy(base), strong=True, noise_px=noise_px)
            tr.pose_at(k / fps)
        bands[fps] = (tr.filter.frame_dt, tr.filter.dead_bands()[1], tr.filter.noise)
    ok = all(abs(fdt - 1 / fps) < 1e-9
             and abs(band - max(settings["still_px_s"], settings["noise_k"] * nz * fps)) < 1e-6
             for fps, (fdt, band, nz) in bands.items())
    expect(ok and bands[10.0][1] < bands[30.0][1],
           "frame interval measured: "
           + ", ".join(f"{fps:.0f} fps {1e3 * fdt:.1f} ms, rotation dead band {band:.0f} px/s"
                       for fps, (fdt, band, _) in bands.items()))
    # a late frame is counted apart and leaves the interval as it was
    tr = PoseTracker(settings, anchor, radius, f)
    for k in range(5):
        tr.measure(k * dt, noisy(base), strong=True)
    r = tr.measure(2 * dt, noisy(base), strong=True)
    expect(r == "rejected" and tr.counts["late"] == 1 and abs(tr.filter.frame_dt - dt) < 1e-9,
           f"a frame older than the filter's rejected ({r}) and counted as late, the interval unchanged")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(run_test())
    sys.exit(__doc__)
