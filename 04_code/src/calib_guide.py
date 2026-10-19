"""The guided photos of a calibration: where the phone goes for each photo, the same plan on screen and in the
simulation.

A plan is a list of poses of the phone relative to the board. A pose says
    view    where the phone looks from: front, left, right, above, below, or a corner (above_left ...): the phone moved
            that way and turned back to the board, by
    tilt    degrees between the phone's line of sight and the board's normal,
    at      where the centre of the board falls on the phone's screen (fractions of its width and height),
    dist    near, mid or far: the distance of the board's centre (distances(): the board spans SPAN of the frame at
            mid, within RANGE_MM of the camera mode; the photo mode keeps to 25-40 cm, where the focus is locked),
    roll    degrees the phone is turned about its line of sight (clockwise as the operator sees the screen); never more
            than MAX_ROLL, so the phone's own orientation (and so the photo's size) never flips;
    corner  (a pose for a corner of the photo only) top_left, top_right, bottom_right or bottom_left: `at` is then worked
            out for each board (placed) so that the board's own corner falls just inside that corner of the photo
            (REACH): its corners then reach as near the corner of the image as they can be found (the corner nearest
            needs the markers of the board's edge whole in the photo), where a lens model is least sure
            (calibrate_charuco.coverage). The phone looks from the opposite side, so that the board's corner in the
            corner of the photo is the far one, its squares the smallest.
The phone is held across (landscape) with its top to the left for every photo: a photo turned the other way has another
size and would not calibrate with the rest (calibrate_charuco.detect), so the check of each photo says so. A pose is
where the phone is relative to the board: with the phone still on a stand the board is moved and turned instead (the
page draws it so for a physical board).

PLANS: quick (15 poses: the five views at 35 degrees, the four corners of the photo, two turns, the edges at two
distances), accurate (40: eight directions at 25 and 35 degrees, the nine places of the screen, turns, the four corners
of the photo, near views, as the calibration protocol asks), check (10: the session check). Each plan covers tilts in
every direction, the whole image and its four corners, and the three distances. No pose is tilted more than MAX_TILT:
past it the board is deep enough to leave the depth of field of the phone's f/1.5 lens at 30-40 cm.

camera_pose(pose, board) gives T_C_M for a board (squares, square_mm) with the nominal camera of the mode (CAMERAS);
guide(pose, board) what the page draws: the board as the phone's screen shows it (outline and black squares, in
fractions of the screen) and the numbers of the side view; render_plan() photos of a plan rendered as the phone would
see them (calibrate_charuco.render: blur, noise, light), with the operator's slips (JITTER), for the simulation.

Board frame M as cv2.aruco.CharucoBoard: origin at the top left corner, x right, y down, z into the board. Camera C as
OpenCV: x right, y down, z forward, in the photo as it is shown (upright on the phone held across).

Usage:
    python calib_guide.py show [--plan quick|accurate|check]    the poses of a plan and the corners each keeps in view
    python calib_guide.py test [--work DIR]                     the plans in simulation: every pose sees enough of the
                                                                screen board, the 3D-printed plate and the A4 sheet;
                                                                photos of each plan (with slips) calibrate within the
                                                                limits of calibrate_charuco.py; exit 1 on failure.
                                                                DIR must be empty, missing, or a folder of an earlier
                                                                test (it is emptied)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

import calibrate_charuco

WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact")) / "calib_guide"
RAYS_KEPT = 2           # pixel ray tables kept (about 100 MB each at the photo size)
# the nominal camera of each mode: the Poco X7 Pro main camera, 26 mm equivalent, 4:3, no distortion
CAMERAS = {"photo": {"size": (4080, 3072), "f": 3070.0}, "stream": {"size": (1920, 1440), "f": 1439.0}}
RANGE_MM = {"photo": (250.0, 400.0), "stream": (250.0, 600.0)}     # the board's centre from the phone
SPAN = 0.6              # at mid the board spans this share of the frame (the larger of its two sides)
NEAR, FAR = 0.85, 1.2   # near and far, against mid
MAX_ROLL = 35.0
MAX_TILT = 35.0
REACH = 0.025           # a pose for a corner of the photo: the board's own corner this far inside it (of the frame;
                        # the lens moves the corners of the image outwards by about 1.5 %)
# the corner of the board that goes to each corner of the photo (the board's top up on the screen): x and y of M, 0 or 1
CORNER_SIDES = {"top_left": (0, 0), "top_right": (1, 0), "bottom_right": (1, 1), "bottom_left": (0, 1)}
MIN_SEEN = 24            # corners a pose must keep inside the frame (as calibrate_charuco.sample_pose)
EDGE_PX = 20             # ... at least this far from the frame's edge
DIAG = 1 / math.sqrt(2)
VIEWS = {"front": (0, 0), "left": (-1, 0), "right": (1, 0), "above": (0, 1), "below": (0, -1),
         "above_left": (-DIAG, DIAG), "above_right": (DIAG, DIAG), "below_left": (-DIAG, -DIAG),
         "below_right": (DIAG, -DIAG)}
# the operator does not match a pose exactly: the simulation adds this much (uniform, +/-)
JITTER = {"tilt_deg": 6.0, "dir_deg": 12.0, "at": 0.06, "dist": 0.08, "roll_deg": 6.0}


def P(view, tilt, at=(0.5, 0.5), dist="mid", roll=0.0, corner=None):
    p = {"view": view, "tilt": float(tilt if view != "front" else 0.0), "at": [float(at[0]), float(at[1])],
         "dist": dist, "roll": float(roll)}
    if corner:
        p["corner"] = corner
    return p


_CORNERS = (("above_left", (0.3, 0.3)), ("above_right", (0.7, 0.3)), ("below_left", (0.3, 0.7)),
            ("below_right", (0.7, 0.7)))
# the four corners of the photo, each seen from the opposite side (`at` only a first guess: placed works it out)
_REACH = (("below_right", (0.3, 0.3), "top_left"), ("below_left", (0.7, 0.3), "top_right"),
          ("above_right", (0.3, 0.7), "bottom_left"), ("above_left", (0.7, 0.7), "bottom_right"))
_DIRS = ("left", "right", "above", "below", "above_left", "above_right", "below_left", "below_right")


def _toward(u, v):
    """The view named after where the board sits on the screen (left half: from the left, top half: from above)."""
    h = "left" if u < 0.5 else "right" if u > 0.5 else ""
    t = "above" if v < 0.5 else "below" if v > 0.5 else ""
    return "_".join(x for x in (t, h) if x) or "front"


# the nine places of the screen, each with the phone turned a little that way
_PLACES = [((u, v), _toward(u, v)) for v in (0.28, 0.5, 0.72) for u in (0.28, 0.5, 0.72)]

PLANS = {
    "quick": [P("front", 0), P("left", 35), P("right", 35), P("above", 35), P("below", 35)]
    + [P(v, 20, at, "far", corner=c) for v, at, c in _REACH]
    + [P("front", 0, dist="near", roll=25), P("left", 15, dist="near", roll=-25),
       P("left", 25, (0.68, 0.5), "far", 15), P("right", 25, (0.32, 0.5), "far", -15),
       P("above", 25, (0.5, 0.68), "near"), P("below", 25, (0.5, 0.32), "near")],
    "accurate": [P(v, 25) for v in _DIRS] + [P(v, 35) for v in _DIRS]
    + [P(view, 0 if view == "front" else 15, at, "far") for at, view in _PLACES]
    + [P("front", 0, dist="near", roll=20), P("front", 0, dist="mid", roll=-20), P("front", 0, dist="near", roll=-35),
       P("front", 0, dist="mid", roll=35)]
    + [P(v, 25, at, "far", corner=c) for v, at, c in _REACH]
    + [P("front", 0, dist="near"), P("left", 30, dist="near"), P("right", 30, dist="near"),
       P("above", 30, dist="near"), P("below", 30, dist="near"), P("front", 0, dist="far"), P("above_left", 20)],
    "check": [P("front", 0), P("left", 35), P("right", 35), P("above", 35), P("below", 35)]
    + [P(v, 30, at, "far") for v, at in _CORNERS] + [P("front", 0, dist="near", roll=25)],
}
# what the counter allows of each plan: at least `min` photos kept to calibrate, at most the plan's length
LIMITS = {"quick": {"min": 12}, "accurate": {"min": 30}, "check": {"min": 8}}


def limits(plan):
    """{"plan", "n" (poses), "min", "max"} of a plan."""
    n = len(PLANS[plan])
    return {"plan": plan, "n": n, "min": min(LIMITS[plan]["min"], n), "max": n}


def distances(board_mm, mode="photo"):
    """{near, mid, far} in mm for a board of board_mm (width, height): at mid the board spans SPAN of the frame
    (facing, the larger of its two sides against the frame's), within the mode's RANGE_MM: a small board is taken
    from as near as the focus allows, a large one from farther, so that it covers the image."""
    c = CAMERAS[mode]
    w, h = c["size"]
    W, H = board_mm
    mid = c["f"] * max(W / (SPAN * w), H / (SPAN * h))
    lo, hi = RANGE_MM[mode]
    mid = min(hi, max(lo, mid))
    return {"near": max(lo, NEAR * mid), "mid": mid, "far": min(hi, FAR * mid)}


def camera(mode):
    """(K, (w, h)) of the nominal camera of a mode."""
    c = CAMERAS[mode]
    w, h = c["size"]
    K = np.array([[c["f"], 0, (w - 1) / 2], [0, c["f"], (h - 1) / 2], [0, 0, 1.0]])
    return K, (w, h)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def _rot_between(a, b):
    """The rotation turning the unit vector a onto b (Rodrigues); half a turn about an axis across a when b is -a."""
    a, b = _unit(a), _unit(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = _unit(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def direction(pose):
    """(yaw, pitch) in degrees of a pose: yaw > 0 the phone to the right of the board, pitch > 0 above it (a pose may
    give them itself, as the slips of the simulation do)."""
    if "yaw" in pose:
        return float(pose["yaw"]), float(pose["pitch"])
    dx, dy = VIEWS[pose["view"]]
    t = pose["tilt"]
    return t * dx, t * dy


def placed(pose, board_mm, mode="photo", K=None, size=None):
    """The pose with its `at` worked out when it is a pose for a corner of the photo ("corner"): the place of the
    board's centre on the screen at which the board's own corner on that side falls REACH inside that corner of the
    photo, for this board and camera (the pose then says which corner in "reach"); any other pose as it is."""
    if not pose.get("corner"):
        return pose
    nominal_K, nominal_size = camera(mode)
    K = nominal_K if K is None else np.asarray(K, np.float64)
    w, h = nominal_size if size is None else size
    sx, sy = CORNER_SIDES[pose["corner"]]
    corner = np.array([[sx * board_mm[0], sy * board_mm[1], 0.0]])
    want = np.array([(sx + (1 - 2 * sx) * REACH) * w, (sy + (1 - 2 * sy) * REACH) * h])
    at = np.asarray(pose["at"], np.float64)
    for _ in range(12):         # turning the phone a little moves the whole picture about as much: a few steps
        uv, _ = project(_camera_pose({**pose, "at": at.tolist()}, board_mm, mode, K, (w, h)), corner, K)
        step = (want - uv[0]) / [w, h]
        at = np.clip(at + step, 0.05, 0.95)
        if np.abs(step).max() < 1e-4:
            break
    out = {k: v for k, v in pose.items() if k != "corner"}
    return {**out, "at": [round(float(at[0]), 4), round(float(at[1]), 4)], "reach": pose["corner"]}


def camera_pose(pose, board_mm, mode="photo", K=None, size=None):
    """T_C_M (4 x 4) of a pose for a board of board_mm = (width, height): the phone at the pose's distance from the
    board's centre, on the side its view says, looking at the centre with the board's top up, turned by roll, then
    turned in place so that the centre falls at `at` on the screen (for a pose for a corner of the photo, where
    placed() puts it). K and size default to the mode's nominal camera (each on its own)."""
    return _camera_pose(placed(pose, board_mm, mode, K, size), board_mm, mode, K, size)


def _camera_pose(pose, board_mm, mode="photo", K=None, size=None):
    nominal_K, nominal_size = camera(mode)
    K = nominal_K if K is None else np.asarray(K, np.float64)
    size = nominal_size if size is None else size
    W, H = board_mm
    centre = np.array([W / 2, H / 2, 0.0])
    yaw, pitch = (math.radians(a) for a in direction(pose))
    d = distances((W, H), mode)[pose["dist"]] if isinstance(pose["dist"], str) else float(pose["dist"])
    # from the centre towards the phone: z of M points into the board, so the phone is on -z; y of M points down
    out = _unit([math.sin(yaw) * math.cos(pitch), -math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])
    eye = centre + d * out
    z = -out
    y = np.array([0, 1.0, 0]) - z[1] * z
    if np.linalg.norm(y) < 1e-9:                # looking straight along the board's y: its x gives the up instead
        y = np.array([1.0, 0, 0]) - z[0] * z
    y = _unit(y)
    x = np.cross(y, z)
    R = np.stack([x, y, z])                      # rows: the camera's axes in M, so R maps M to C
    r = math.radians(pose["roll"])              # the phone turned clockwise: the board turns the other way on screen
    Rz = np.array([[math.cos(r), math.sin(r), 0], [-math.sin(r), math.cos(r), 0], [0, 0, 1.0]])
    R = Rz @ R
    t = -R @ eye
    w, h = size
    ray = np.linalg.solve(K, [pose["at"][0] * w, pose["at"][1] * h, 1.0])
    D = _rot_between([0, 0, 1.0], ray)
    T = np.eye(4)
    T[:3, :3] = D @ R
    T[:3, 3] = D @ t
    return T


def board_mm(spec):
    nx, ny = spec["squares"]
    return nx * spec["square_mm"], ny * spec["square_mm"]


def project(T, pts, K):
    cam = pts @ T[:3, :3].T + T[:3, 3]
    uv = cam[:, :2] / cam[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, cam[:, 2]


def seen(pose, spec, mode="photo"):
    """Chessboard corners of the board inside the frame (EDGE_PX from its edge) at this pose."""
    K, (w, h) = camera(mode)
    board = calibrate_charuco.make_board(spec["square_mm"], spec["squares"], spec.get("marker_mm"))
    uv, z = project(camera_pose(pose, board_mm(spec), mode), board.getChessboardCorners().astype(np.float64), K)
    ok = (z > 1) & (uv[:, 0] >= EDGE_PX) & (uv[:, 0] < w - EDGE_PX) & (uv[:, 1] >= EDGE_PX) & (uv[:, 1] < h - EDGE_PX)
    return int(ok.sum()), int(len(ok))


def findable(pose, spec, mode="photo"):
    """The chessboard corners of the board (px, N x 2) the detector can find at this pose with the nominal camera:
    each with the four squares about it inside the frame (EDGE_PX from its edge), so that the markers it is found
    from are whole in the photo."""
    K, (w, h) = camera(mode)
    nx, ny = spec["squares"]
    q = spec["square_mm"]
    xs, ys = np.meshgrid(np.arange(nx + 1) * q, np.arange(ny + 1) * q)
    grid = np.stack([xs, ys, np.zeros_like(xs)], -1).reshape(-1, 3)
    uv, z = project(camera_pose(pose, board_mm(spec), mode), grid, K)
    uv = uv.reshape(ny + 1, nx + 1, 2)
    inside = ((z.reshape(ny + 1, nx + 1) > 1) & (uv[..., 0] >= EDGE_PX) & (uv[..., 0] < w - EDGE_PX)
              & (uv[..., 1] >= EDGE_PX) & (uv[..., 1] < h - EDGE_PX))
    return np.array([uv[j, i] for j in range(1, ny) for i in range(1, nx) if inside[j - 1:j + 2, i - 1:i + 2].all()]
                    ).reshape(-1, 2)


def guide(pose, spec, mode="photo", n=0, plan="quick"):
    """What the page draws for a pose: the screen's aspect; the board's outline and its black squares as the phone's
    screen shows them, in fractions of the screen's width and height (x right, y down; parts may lie outside it); and
    the side view: yaw and pitch (degrees), the distance (cm), the roll."""
    K, (w, h) = camera(mode)
    W, H = board_mm(spec)
    pose = placed(pose, (W, H), mode)
    T = camera_pose(pose, (W, H), mode)
    nx, ny = spec["squares"]
    sq = spec["square_mm"]

    def norm(pts):
        uv, _ = project(T, np.asarray(pts, np.float64), K)
        return [[round(float(u / w), 4), round(float(v / h), 4)] for u, v in uv]
    outline = norm([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]])
    cells = [norm([[i * sq, j * sq, 0], [(i + 1) * sq, j * sq, 0], [(i + 1) * sq, (j + 1) * sq, 0],
                   [i * sq, (j + 1) * sq, 0]]) for j in range(ny) for i in range(nx) if (i + j) % 2 == 0]
    yaw, pitch = direction(pose)
    d = distances((W, H), mode)[pose["dist"]]
    return {"n": n, "plan": plan, "of": len(PLANS[plan]), "view": pose["view"], "tilt": round(pose["tilt"]),
            "yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(pose["roll"]), "dist": pose["dist"],
            "dist_cm": int(5 * round(d / 50)), "aspect": w / h, "outline": outline, "cells": cells,
            "at": pose["at"], "board": [round(W, 1), round(H, 1)]}


def jittered(pose, rng):
    """The pose as an operator gets it: tilt, its direction, the place on screen, the distance and the roll a little
    off (JITTER)."""
    j = JITTER
    yaw, pitch = direction(pose)
    tilt = max(0.0, math.hypot(yaw, pitch) + rng.uniform(-j["tilt_deg"], j["tilt_deg"]))
    ang = math.atan2(pitch, yaw) if (yaw or pitch) else rng.uniform(-math.pi, math.pi)
    ang += math.radians(rng.uniform(-j["dir_deg"], j["dir_deg"]))
    return {"yaw": tilt * math.cos(ang), "pitch": tilt * math.sin(ang),
            "at": [pose["at"][0] + rng.uniform(-j["at"], j["at"]), pose["at"][1] + rng.uniform(-j["at"], j["at"])],
            "dist_mul": 1 + rng.uniform(-j["dist"], j["dist"]), "roll": pose["roll"] + rng.uniform(-j["roll_deg"],
                                                                                                   j["roll_deg"])}


def free_pose(yaw, pitch, at, dist, roll):
    """A pose of any yaw and pitch (degrees) and distance (mm), for the slips of the simulation."""
    return {"view": "front", "tilt": math.hypot(yaw, pitch), "yaw": float(yaw), "pitch": float(pitch),
            "at": list(at), "dist": float(dist), "roll": float(roll)}


# ── synthetic photos of a plan ───────────────────────────────────────────────
def texture(kind, spec, lay=None, ppm=None):
    """(page, origin_mm, ppm) of what the phone photographs: "screen" the monitor's board (screen_board.render of its
    layout, white margins, the app's dark around it), "plate" the 3D-printed plate (white plate with its margins,
    a grey table), "sheet" the A4 sheet."""
    board = calibrate_charuco.make_board(spec["square_mm"], spec["squares"], spec.get("marker_mm"))
    if kind == "screen":
        import screen_board
        img = screen_board.render(lay)
        p = lay["pitch_mm"]
        pad = 60
        page = np.full((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad), 30, np.uint8)     # the app around the board
        page[pad:-pad, pad:-pad] = img
        x0, y0 = lay["origin_px"]
        # one texture pixel per screen pixel: ppm = 1 / pitch, the board's corner on a pixel boundary
        return page, ((x0 + pad) * p, (y0 + pad) * p), 1.0 / p
    ppm = ppm or calibrate_charuco.texture_ppm(board)
    if kind == "sheet":
        page, origin = calibrate_charuco.page_image(board, ppm)
        return page, origin, ppm
    import charuco_plate
    plate = charuco_plate.load()
    PW, PH, _ = plate["plate"]["size_mm"]
    OX, OY = plate["plate"]["board_origin_mm"][:2]          # plate coordinates, Y up: the board's top left corner
    W, H = board_mm(spec)
    s = spec["square_mm"] / plate["board"]["square_mm"]       # the plate as measured
    table = 20.0                                              # mm of table around the plate
    tw, th = int(round((PW * s + 2 * table) * ppm)), int(round((PH * s + 2 * table) * ppm))
    page = np.full((th, tw), 110, np.uint8)
    x0 = y0 = int(round(table * ppm))
    page[y0:y0 + int(round(PH * s * ppm)), x0:x0 + int(round(PW * s * ppm))] = 250
    bx, by = int(round((table + OX * s) * ppm)), int(round((table + (PH - OY) * s) * ppm))
    bw, bh = int(round(W * ppm)), int(round(H * ppm))
    page[by:by + bh, bx:bx + bw] = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    return page, (bx / ppm, by / ppm), ppm


def render_plan(out, plan, spec, kind, mode="photo", lay=None, seed=0, K=None, dist=None, prefix="photo", only=None,
                blur=None):
    """Photos of the poses of a plan, as the phone takes them with the operator's slips (jittered), with the camera
    K, dist (calibrate_charuco.SYNTH_K, SYNTH_DIST for the photo mode; scaled for the stream), in out with truth.json.
    only: indexes of the plan to render; blur: (lo, hi) sigma in px in place of calibrate_charuco.SYNTH_BLUR."""
    rng = np.random.default_rng(seed)
    w, h = CAMERAS[mode]["size"]
    if K is None:
        s = w / calibrate_charuco.SYNTH_SIZE[0]
        K = calibrate_charuco.SYNTH_K.copy()
        K[:2] *= s
        K[:2, 2] = (calibrate_charuco.SYNTH_K[:2, 2] + 0.5) * s - 0.5
    if dist is None:
        dist = calibrate_charuco.SYNTH_DIST
    page, origin, ppm = texture(kind, spec, lay)
    page = cv2.GaussianBlur(page, (0, 0), 0.6)
    rays = calibrate_charuco.pixel_rays(K, dist, (w, h))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    views = []
    for i, pose in enumerate(PLANS[plan]):
        if only is not None and i not in only:
            continue
        j = jittered(placed(pose, board_mm(spec), mode, K, (w, h)), rng)
        d = distances(board_mm(spec), mode)[pose["dist"]] * j["dist_mul"]
        T = camera_pose(free_pose(j["yaw"], j["pitch"], j["at"], d, j["roll"]), board_mm(spec), mode, K, (w, h))
        img, b, nz = calibrate_charuco.render(page, origin, rays, T[:3, :3], T[:3, 3], rng, ppm, blur=blur)
        name = f"{prefix}_{i:03d}.{'jpg' if mode == 'photo' else 'png'}"
        if mode == "photo":
            cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            cv2.imwrite(str(out / name), img)
        views.append({"image": name, "pose": i, "rvec": cv2.Rodrigues(T[:3, :3])[0].ravel().tolist(),
                      "tvec": T[:3, 3].tolist(), "blur_px": round(b, 3), "noise_grey": round(nz, 3)})
    calibrate_charuco.write_json_atomic(out / "truth.json", {"image_size": [w, h], "K": np.asarray(K).tolist(),
                                                             "dist": np.asarray(dist).tolist(), "seed": seed,
                                                             "views": views})
    return out


_RAYS: dict[tuple[int, int], np.ndarray] = {}     # the pixel rays of the newest sizes, RAYS_KEPT at most
_RAYS_LOCK = threading.Lock()


def _rays(K, size):
    """The pixel rays of the synthetic camera at this size, kept for the next photo (the oldest size let go)."""
    key = (int(size[0]), int(size[1]))
    with _RAYS_LOCK:
        hit = _RAYS.pop(key, None)
        if hit is not None:
            _RAYS[key] = hit                    # now the newest
            return hit
    rays = calibrate_charuco.pixel_rays(K, calibrate_charuco.SYNTH_DIST, key)
    with _RAYS_LOCK:
        _RAYS[key] = rays
        while len(_RAYS) > RAYS_KEPT:
            del _RAYS[next(iter(_RAYS))]
    return rays


def photo(plan, i, spec, kind, mode="photo", lay=None, rng=None, size=None):
    """One photo of pose i of a plan as the phone takes it (with the operator's slips), grey: what the simulated phone
    of ImpactGuide gives for a calibration. The camera is calibrate_charuco's synthetic one, scaled to the size of the
    mode (or `size`)."""
    rng = rng or np.random.default_rng()
    w, h = size or CAMERAS[mode]["size"]
    s = w / calibrate_charuco.SYNTH_SIZE[0]
    K = calibrate_charuco.SYNTH_K.copy()
    K[:2] *= s
    K[:2, 2] = (calibrate_charuco.SYNTH_K[:2, 2] + 0.5) * s - 0.5
    rays = _rays(K, (w, h))
    page, origin, ppm = texture(kind, spec, lay)
    page = cv2.GaussianBlur(page, (0, 0), 0.6)
    pose = placed(PLANS[plan][i % len(PLANS[plan])], board_mm(spec), mode, K, (w, h))
    j = jittered(pose, rng)
    d = distances(board_mm(spec), mode)[pose["dist"]] * j["dist_mul"]
    T = camera_pose(free_pose(j["yaw"], j["pitch"], j["at"], d, j["roll"]), board_mm(spec), mode, K, (w, h))
    img, _, _ = calibrate_charuco.render(page, origin, rays, T[:3, :3], T[:3, 3], rng, ppm)
    return img


# ── test ─────────────────────────────────────────────────────────────────────
FAILS = []


def expect(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def boards():
    """The three boards of the app: on this PC's monitor (the 15.6 in panel, 1560 x 880 px of the stage), the 3D-printed
    plate (nominal) and the A4 sheet (24 mm squares). (name, kind, spec, layout)."""
    import charuco_plate
    import screen_board
    lay = screen_board.layout((1560, 880), 0.17921)
    out = [("screen", "screen", screen_board.spec(lay), lay)]
    out.append(("plate", "plate", charuco_plate.spec(), None))
    sheet = calibrate_charuco.board_spec(calibrate_charuco.make_board())
    out.append(("sheet", "sheet", sheet, None))
    return out


def test_geometry(work):
    """The edges of the geometry and the simulation: opposite vectors, a pose seen from straight above, a camera
    given without its size, the blur of a plan passed along (nothing global changed), the rays kept bounded."""
    a = np.array([0, 0, 1.0])
    R = _rot_between(a, -a)
    expect(np.allclose(R @ a, -a) and np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-9,
           "the rotation between opposite vectors is half a turn, not the identity")
    _, _, spec, _ = boards()[2]
    K, size = camera("photo")
    up = camera_pose(free_pose(0, 90, (0.5, 0.5), 300.0, 0), board_mm(spec))
    given = camera_pose(PLANS["quick"][1], board_mm(spec), K=K)
    expect(np.isfinite(up).all() and np.allclose(up[:3, :3] @ up[:3, :3].T, np.eye(3))
           and np.allclose(given, camera_pose(PLANS["quick"][1], board_mm(spec))),
           "a pose from straight above (pitch 90 degrees) and a camera given without its size: a proper pose")
    before = calibrate_charuco.SYNTH_BLUR
    out = render_plan(work / "blur", "check", spec, "sheet", seed=3, only=[0], blur=(3.0, 3.0))
    b = json.loads((out / "truth.json").read_text(encoding="utf-8"))["views"][0]["blur_px"]
    expect(calibrate_charuco.SYNTH_BLUR == before and abs(b - 3.0) < 1e-9,
           f"the blur of a plan reaches the render ({b} px) without changing calibrate_charuco.SYNTH_BLUR")
    _RAYS.clear()
    for w in (400, 320, 240, 400):
        photo("check", 0, spec, "sheet", size=(w, w * 3 // 4))
    expect(len(_RAYS) <= RAYS_KEPT and (400, 300) in _RAYS, f"pixel rays kept for {len(_RAYS)} sizes at most")
    _RAYS.clear()


def cmd_show(a):
    for name, kind, spec, lay in boards():
        print(f"{name}: {spec['squares'][0]} x {spec['squares'][1]} of {spec['square_mm']:.2f} mm")
        for i, p in enumerate(PLANS[a.plan]):
            s, n = seen(p, spec, a.mode)
            print(f"  {i:2d} {p['view']:12s} tilt {p['tilt']:4.0f}  at {p['at']}  {p['dist']:4s} roll {p['roll']:4.0f}"
                  f"  {s}/{n} corners")


def cmd_test(a):
    try:
        work = calibrate_charuco.test_folder(a.work)
    except ValueError as e:
        expect(False, str(e))
        return 1
    test_geometry(work)
    # every pose of every plan keeps enough of each board in view, in both modes
    for name, kind, spec, lay in boards():
        for mode in CAMERAS:
            for plan in PLANS:
                low = [(i, seen(p, spec, mode)[0]) for i, p in enumerate(PLANS[plan])
                       if seen(p, spec, mode)[0] < MIN_SEEN]
                expect(not low, f"{name}, {mode}, {plan} ({len(PLANS[plan])} poses): every pose keeps at least "
                                f"{MIN_SEEN} corners in the frame {low or ''}")
    # the plans span what a calibration needs: tilts in every direction up to MAX_TILT, the whole image, turns
    for plan in ("quick", "accurate"):
        ps = PLANS[plan]
        dirs = {p["view"] for p in ps}
        tilts = max(p["tilt"] for p in ps + PLANS["check"])
        rolls = (min(p["roll"] for p in ps), max(p["roll"] for p in ps))
        ats = np.array([p["at"] for p in ps])
        expect(dirs >= set(VIEWS) and 30 <= tilts <= MAX_TILT and rolls[0] <= -15 and rolls[1] >= 15
               and ats[:, 0].min() <= 0.32 and ats[:, 0].max() >= 0.68 and ats[:, 1].min() <= 0.32
               and ats[:, 1].max() >= 0.68 and max(abs(p["roll"]) for p in ps) <= MAX_ROLL
               and {p["dist"] for p in ps} == {"near", "mid", "far"},
               f"{plan}: {len(ps)} poses, all nine views, tilt up to {tilts:.0f} degrees, turns {rolls}, the board "
               f"from {ats.min(0).round(2)} to {ats.max(0).round(2)} of the screen, three distances")
    # ... and the four corners of the photo, with every board in both modes: the corners that can be found (their
    # squares whole in the frame, as the markers about them must be) beyond calibrate_charuco.COVER_CORNER
    for name, kind, spec, lay in boards():
        for mode in CAMERAS:
            K, (w, h) = camera(mode)
            for plan in ("quick", "accurate"):
                cov = calibrate_charuco.coverage([findable(p, spec, mode) for p in PLANS[plan]], (w, h))
                expect(not cov["missing_corners"],
                       f"{name}, {mode}, {plan}: the board reaches the four corners of the photo ({cov['corners']}, "
                       f"{100 * cov['reach']:.0f} % of the way)")
    # the guide's picture is the projection of the pose: the outline holds the board's centre where the pose says
    _, _, spec, _ = boards()[1]
    g = guide(PLANS["quick"][1], spec)
    c = np.mean(g["outline"], 0)
    expect(abs(g["yaw"] + 35) < 1e-6 and g["cells"] and abs(c[0] - 0.5) < 0.12 and abs(c[1] - 0.5) < 0.05,
           f"guide: pose 2 seen from the left, 35 degrees; the outline's middle at {c.round(3)} of the screen")
    # a pose for a corner of the photo: the board's own corner REACH inside that corner, whatever the board
    far = []
    for name, kind, spec, lay in boards():
        K, (w, h) = camera("photo")
        for p in PLANS["quick"]:
            if not p.get("corner"):
                continue
            q = placed(p, board_mm(spec))
            sx, sy = CORNER_SIDES[p["corner"]]
            W, H = board_mm(spec)
            uv = project(camera_pose(q, (W, H)), np.array([[sx * W, sy * H, 0.0]]), K)[0][0] / [w, h]
            far.append(float(np.abs(uv - [sx + (1 - 2 * sx) * REACH, sy + (1 - 2 * sy) * REACH]).max()))
    expect(len(far) == 12 and max(far) < 1e-3 and all("corner" not in placed(p, (200.0, 150.0)) for p in PLANS["quick"]),
           f"the poses for the corners of the photo put the board's own corner {100 * REACH:.1f} % inside them "
           f"(at most {max(far):.5f} off)")
    # simulation: each plan's photos (with the operator's slips) calibrate within the limits
    limits_ = {"quick": (0.5, 0.004, 6.0, 2.5), "accurate": (0.5, 0.002, 4.0, 1.5)}
    for name, kind, spec, lay in boards():
        for plan in ("quick", "accurate"):
            if plan == "accurate" and name != "screen" and not a.all:
                continue
            out = work / f"{name}_{plan}"
            render_plan(out, plan, spec, kind, lay=lay, seed=7)
            cal = calibrate_charuco.run_calibrate(out, None, out / "truth.json", spec["square_mm"],
                                                  tuple(spec["squares"]), spec["marker_mm"], log=lambda line: None)
            e = cal["error"]
            rms, focal, centre, model = limits_[plan]
            expect(cal["rms_px"] < rms and abs(e["fx_rel"]) < focal and abs(e["fy_rel"]) < focal
                   and abs(e["cx_px"]) < centre and abs(e["cy_px"]) < centre and e["model_max_px"] < model,
                   f"{name}, {plan}: {cal['n_views']} of {len(PLANS[plan])} photos, RMS {cal['rms_px']:.3f} px, "
                   f"fx {100 * e['fx_rel']:+.3f} %, fy {100 * e['fy_rel']:+.3f} %, centre {e['cx_px']:+.2f}, "
                   f"{e['cy_px']:+.2f} px, model {e['model_max_px']:.2f} px at most")
    print(f"\n{len(FAILS)} failures" if FAILS else "\nall checks passed", flush=True)
    return 1 if FAILS else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("show")
    a.add_argument("--plan", choices=list(PLANS), default="quick")
    a.add_argument("--mode", choices=list(CAMERAS), default="photo")
    a = sub.add_parser("test")
    a.add_argument("--work", type=Path, default=WORK / "selftest")     # a folder of the test's own
    a.add_argument("--all", action="store_true", help="also the accurate plan on the plate and the sheet")
    args = ap.parse_args()
    if args.cmd == "show":
        cmd_show(args)
        return 0
    return cmd_test(args)


if __name__ == "__main__":
    sys.exit(main())
