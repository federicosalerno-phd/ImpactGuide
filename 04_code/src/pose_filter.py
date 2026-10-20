"""Temporal filter of the board pose for the live AR: what the overlay draws on each frame of the stream.

The board pose from one frame (marker_pose.py, PnP on the ChArUco corners) shakes by a fraction of a pixel from frame
to frame, now and then comes out wrong on a frame with few corners, and is missing on frames where the board is
covered or blurred. Drawn as it comes, the overlay trembles and blinks. PoseTracker turns the poses of the processed
frames into a calm pose for every frame shown:

  filter    One Euro filter (Casiez, Roussel and Vogel, CHI 2012) on SE(3), in three channels: the position of an
            anchor (the board's centre), the board's turn about the line of sight to it, and its tilt against that
            line (along the geodesic, slerp). Each cutoff grows with that channel's speed in the image (px/s: the
            anchor's; the model's lever times the angular speed, times f / z) beyond a dead band (still_px_s, or the
            speed that the channel's noise alone gives): strong smoothing while the phone is still, almost none, and
            so almost no lag, as soon as it moves; moving, the pose is first carried on by the filtered velocity (the
            lead), so a steady motion is not left behind either. The tilt of a small board is what its corners give
            worst (the tracker says how noisy, noise_px): its dead band is wider and its smoothing at rest stronger.
            A hand that trembles turns the phone, which moves the board's centre across the picture and leaves its
            orientation against the line of sight as it was: the tremble goes through the position, followed as it
            comes, and the face stays on the board instead of trailing it.
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
# The values: 19/10/2026, against the face drawn on a hand-held phone with the truth known (a tremor of 6-12 Hz, a
# drift, slow and fast moves; corners with the tracker's noise; both boards) and against the reference of the two
# regression clips of the phone (live_tune): the precise channels open up at once with the speed (beta, and a cap
# four times higher) so the face follows the hand, the tilt keeps its smoothing (tilt_beta, a higher floor).
DEFAULTS = {
    "smooth": True,           # False: every frame at its own pose as it comes (the tests of the geometry)
    "min_cutoff_hz": 2.0,     # smoothing while still: lower is calmer and slower to settle
    "beta": 1.0,              # cutoff added per px/s of the model's speed past the dead band: higher lags less
    "turn_beta": 1.0,         #   the same for the turn about the line of sight
    "tilt_beta": 0.1,         #   and for the tilt, the noisy channel (20/10: from 0.25, the patient board's clip
                              #   shakes 5 % less moving, error 6 % lower at its fast end, no lag; the large one alike)
    "d_cutoff_hz": 2.0,       # smoothing of that speed
    "lead": True,             # moving, the filter carries the pose on by its speed first: no lag at a steady speed
    "still_px_s": 8.0,        # dead band at least this (px/s at the stream size): slower is noise, not motion
    "predict_s": 0.3,         # a frame without its own pose is drawn at the pose carried on, at most this far,
    "predict_tau_s": 0.12,    #   the speed dying away with this time constant (no freeze, no running off)
    "glide_gap_s": 0.1,       # a jump after this long without the board is taken at once (no glide from a stale pose)
    "gate_px": 40.0,          # a pose moving the model more than this (px, stream size) from the prediction is a jump,
    "glide_s": 0.15,          #   believed (when strong or confirmed) by gliding there over this time
    "noise_k": 2.0,           # each channel's dead band: this many times the speed its expected noise gives
    "turn_share": 0.12,       # noise of the turn about the line of sight, as a share of the rotation's (synthetic
                              #   PnP of both boards: 0.10 and 0.13)
    "noise_ref_px": 0.5,      # the lowest cutoff scales with noise_ref_px over the expected noise (px per frame),
    "min_cutoff_floor": 0.5,  #   down to this share for a noisy measure (a small board far away)
    "min_cutoff_cap": 12.0,   #   and up to this many times for a precise one: the overlay stays glued to the board
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


def line_of_sight(p):
    """The rotation taking the optical axis onto the line of sight to the point p (the shortest one)."""
    u = np.asarray(p, np.float64) / max(float(np.linalg.norm(p)), 1e-9)
    axis = np.array([-u[1], u[0], 0.0])
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        return np.eye(3)
    return rotmat(axis / s * np.arctan2(s, u[2]))


def turn_z(phi):
    """The rotation by phi about the optical axis."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def split_turn(D):
    """D = turn_z(phi) @ S: its turn about the optical axis and the rest, the tilt (swing and twist)."""
    w = rotvec(D)
    a = float(np.linalg.norm(w))
    if a < 1e-12:
        return 0.0, np.eye(3)
    phi = 2.0 * np.arctan2(np.sin(a / 2) * w[2] / a, np.cos(a / 2))
    return phi, turn_z(-phi) @ D


def wrap(a):
    """An angle in (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def towards(u, v):
    """The rotation vector of the shortest turn taking the unit vector u onto the unit vector v."""
    axis = np.cross(u, v)
    s = float(np.linalg.norm(axis))
    if s < 1e-15:
        return np.zeros(3)
    return axis / s * np.arctan2(s, float(np.dot(u, v)))


class OneEuroPose:
    """One Euro filter on a rigid transform T (4 x 4, camera from board, mm), in four channels, each with its own
    cutoff from its own speed in the image past its own dead band, after the noise the tracker expects on it:
      direction  of the line of sight to the anchor, the board's centre: its place in the picture, what the
                 corners give best (noise_pos_px);
      range      the anchor's distance from the camera: what the size of the board in the picture gives, poorly for a
                 small board (noise_range_mm), and which the model's lever (radius mm from the anchor) turns into a
                 shift of the face;
      turn       the board's turn about the line of sight: the corners turning in the picture, about eight times
                 better given than the tilt (turn_share of noise_px);
      tilt       the board's orientation against the line of sight: what a small board gives worst, its PnP pose
                 shaking there by pixels on the face (noise_px).
    The pose is split along the line of sight because a hand that trembles turns the phone: the board moves in the
    picture, but its distance and its orientation against the line of sight to its centre stay as they were. So the
    tremble of the hand is carried by the direction, followed as it comes, and the smoothing of the range and the
    tilt (where the noise is) leaves the face on the board instead of trailing it. A precise measure (many corners, a
    large board) is followed almost as it comes; a noisy one (a small board far away) is smoothed while still. The
    tilt is taken from a reference orientation (the one at the last restart) so that its split from the turn never
    meets the half turn where it is undefined."""

    def __init__(self, settings, anchor, radius_mm, f_px):
        self.s = {**DEFAULTS, **settings}
        self.anchor = np.asarray(anchor, np.float64).reshape(3)
        self.radius, self.f = float(radius_mm), float(f_px)
        self.t = None
        self.noise = None           # expected noise of the rotation at the model, px per frame (smoothed)
        self.noise_pos = None       # of the anchor in the picture, px per frame
        self.noise_range = None     # of its distance, mm per frame
        self.frame_dt = FRAME_DT_S  # interval between frames (s): what turns a noise per frame into a speed

    def reset(self, T, t):
        R = np.array(T[:3, :3], np.float64)
        p = R @ self.anchor + T[:3, 3]
        self.rho = max(float(np.linalg.norm(p)), 1.0)     # the anchor's distance (mm) and direction
        self.u = p / self.rho
        self.ref = line_of_sight(self.u).T @ R      # the orientation against the line of sight at the restart
        self.phi, self.S = 0.0, np.eye(3)           # turn and tilt from it
        # angular velocity of the direction (rad/s, camera), speed of the distance (mm/s), turn rate (rad/s),
        # angular velocity of the tilt (rad/s)
        self.w_dir, self.v_rho, self.w_turn, self.w = np.zeros(3), 0.0, 0.0, np.zeros(3)
        self.t = float(t)

    @property
    def p(self):
        """The anchor in the camera (mm)."""
        return self.rho * self.u

    def _rotation(self, u, phi, S):
        return line_of_sight(u) @ turn_z(phi) @ S @ self.ref

    @property
    def R(self):
        """The rotation of the filtered pose."""
        return self._rotation(self.u, self.phi, self.S)

    def _parts(self, T):
        """(direction, distance, turn, tilt) of a pose, against the reference orientation."""
        R = np.asarray(T[:3, :3], np.float64)
        p = R @ self.anchor + T[:3, 3]
        rho = max(float(np.linalg.norm(p)), 1.0)
        u = p / rho
        phi, S = split_turn(line_of_sight(u).T @ R @ self.ref.T)
        return u, rho, phi, S

    def _range_px(self):
        """px on the model per mm of the anchor's distance: the lever seen from the camera."""
        return self.f * self.radius / self.rho ** 2

    def _speeds(self):
        """Speed in the image (px/s) of the anchor (its direction), and of the model at the radius from the
        distance, the turn and the tilt."""
        k = self.f / self.rho
        return (self.f * float(np.linalg.norm(self.w_dir)), self._range_px() * abs(self.v_rho),
                k * self.radius * abs(self.w_turn), k * self.radius * float(np.linalg.norm(self.w)))

    def speeds(self):
        """Speed in the image (px/s) of the anchor and, from the rotation (turn and tilt), of the model at the radius."""
        d, g, q, r = self._speeds()
        return d + g, q + r

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

    def _noises(self):
        """The noise of each channel as px per frame: direction, distance, turn, tilt (None: unknown)."""
        return (self.noise_pos, None if self.noise_range is None else self._range_px() * self.noise_range,
                None if self.noise is None else self.s["turn_share"] * self.noise, self.noise)

    def dead_bands(self, dt=None):
        """(anchor, tilt) dead bands, px/s, for frames dt apart (None: the measured frame interval)."""
        dt = self.frame_dt if dt is None else dt
        return self._channel(self.noise_pos, dt)[0], self._channel(self.noise, dt)[0]

    def update(self, T, t, noise_px=None, noise_pos_px=None, noise_range_mm=None):
        def mix(old, new):
            return old if new is None else float(new) if old is None else 0.8 * old + 0.2 * float(new)
        if self.t is None:
            self.reset(T, t)
            self.noise, self.noise_pos = mix(None, noise_px), mix(None, noise_pos_px)
            self.noise_range = mix(None, noise_range_mm)
            return
        s = self.s
        dt = max(float(t) - self.t, 1e-3)
        self.noise = mix(self.noise, noise_px)
        self.noise_pos = mix(self.noise_pos, noise_pos_px)
        self.noise_range = mix(self.noise_range, noise_range_mm)
        u_m, rho_m, phi_m, S_m = self._parts(T)
        a_d = smoothing(s["d_cutoff_hz"], dt)
        self.w_dir += a_d * (towards(self.u, u_m) / dt - self.w_dir)
        self.v_rho += a_d * ((rho_m - self.rho) / dt - self.v_rho)
        self.w_turn += a_d * (wrap(phi_m - self.phi) / dt - self.w_turn)
        self.w += a_d * (rotvec(self.S.T @ S_m) / dt - self.w)
        speed = self._speeds()
        chans = [self._channel(n, self.frame_dt) for n in self._noises()]
        betas = (s["beta"], s["tilt_beta"], s["turn_beta"], s["tilt_beta"])
        a = [smoothing(c[1] + b * max(0.0, v - c[0]), dt) for v, c, b in zip(speed, chans, betas)]
        lead = [s["lead"] and v >= c[0] for v, c in zip(speed, chans)]
        # moving, blend towards the pose carried on by the filtered velocity: no lag at a steady speed
        u0 = rotmat(self.w_dir * dt) @ self.u if lead[0] else self.u
        rho0 = self.rho + self.v_rho * dt if lead[1] else self.rho
        phi0 = self.phi + self.w_turn * dt if lead[2] else self.phi
        S0 = self.S @ rotmat(self.w * dt) if lead[3] else self.S
        u = rotmat(a[0] * towards(u0, u_m)) @ u0
        self.u = u / np.linalg.norm(u)
        self.rho = max(rho0 + a[1] * (rho_m - rho0), 1.0)
        self.phi = phi0 + a[2] * wrap(phi_m - phi0)
        self.S = S0 @ rotmat(a[3] * rotvec(S0.T @ S_m))
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
        on = [v >= self._channel(n, self.frame_dt)[0] for v, n in zip(self._speeds(), self._noises())]
        u = rotmat(self.w_dir * dt) @ self.u if on[0] else self.u
        rho = max(self.rho + self.v_rho * dt, 1.0) if on[1] else self.rho
        phi = self.phi + self.w_turn * dt if on[2] else self.phi
        S = self.S @ rotmat(self.w * dt) if on[3] else self.S
        R = self._rotation(u, phi, S)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = rho * u - R @ self.anchor
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
    def measure(self, t, T=None, strong=False, noise_px=None, noise_pos_px=None, noise_range_mm=None):
        """A processed frame taken at t: T the PnP pose that passed the RMS limit, None when the board was not found.
        strong: many corners and a low reprojection error. noise_px, noise_pos_px, noise_range_mm: the noise the
        tracker expects on the model from the rotation (px per frame), on the anchor in the picture (px) and on its
        distance (mm). Returns accepted, restarted, rejected or missed (a frame older than the filter's is rejected
        and counted as late)."""
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
            f.noise, f.noise_pos, f.noise_range = noise_px, noise_pos_px, noise_range_mm
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
                    f.noise, f.noise_pos, f.noise_range = noise_px, noise_pos_px, noise_range_mm
                    self.glide = None
                else:
                    self._restart_gliding(T, t)
                return self._accept(T, t, "restarted")
            self.counts["rejected"] += 1
            return "rejected"
        f.update(T, t, noise_px, noise_pos_px, noise_range_mm)
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

    def noisy(T, rot_deg=0.03, trans_mm=0.1, about=None, turn=None):
        """T with noise: a small rotation (about the board origin, or about the given point, where the corners
        are) and a small shift. turn: the rotation as PnP gives it, in the camera, about the given point: about
        the axes across the line of sight (nearly the optical axis here) rot_deg, about it `turn` times as much."""
        if turn is None:
            N = _pose(*rng.normal(0, rot_deg, 3), *rng.normal(0, trans_mm, 3))
            if about is not None:
                N[:3, 3] += about - N[:3, :3] @ about
            return T @ N
        c = T[:3, :3] @ about + T[:3, 3]
        N = _pose(*(rng.normal(0, rot_deg, 3) * [1.0, 1.0, turn]), *rng.normal(0, trans_mm, 3))
        N[:3, 3] += c - N[:3, :3] @ c
        return N @ T

    def shift(Ta, Tb):
        return image_shift(Ta, Tb, anchor, radius, f)

    # still: jitter of the drawn model, raw against filtered; a large board (noise mostly in the position) and a
    # small one (noise mostly in the rotation, which the tracker reports: noise_px; as PnP gives it, the tilt
    # against the line of sight, the turn about it turn_share as large)
    base = _pose(160, 5, 3, -90, -60, 480)
    # (the small board's limit: a third of the raw jitter; its tilt is smoothed a little less at rest since 19/10, so
    # that the face keeps up with a hand-held phone)
    for name, rot, trans, limit in (("large board", 0.03, 0.1, 0.5), ("small board", 0.4, 0.1, 0.35)):
        tr = PoseTracker(settings, anchor, radius, f)
        noise_px = f / 480 * radius * np.radians(rot) * np.sqrt(2)     # rotation noise at the model, px per frame
        raw, out = [], []
        for k in range(150):
            t = k * dt
            m = noisy(base, rot, trans, about=anchor, turn=settings["turn_share"] if rot > 0.1 else None)
            tr.measure(t, m, strong=True, noise_px=noise_px if rot > 0.1 else None,
                       noise_pos_px=f / 480 * trans if rot > 0.1 else None, noise_range_mm=trans if rot > 0.1 else None)
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
