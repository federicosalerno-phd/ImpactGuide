"""The calibration on the bench: the phone on the arm, the 3D-printed board on a ball joint on the post.

The head is taken off its post, the pendulum latched up and the ball joint put on the post in its place (its adapter
on the quick release; the joint of the scanner, its ball with a 1/4" stud); the board screws on the ball. The joint is
set to each of the seven poses of POSES in turn (pan: the board turned to face the arm at an angle; lean: the board
tilted back; turn: the board turned in its own plane), and for each pose the arm goes over the notches of POSE_STOPS (a
notch within MERGE_DEG of a trial position is that position): STEPS, one photo each. The phone's pose at any arm angle
is the turn of the scene's camera about the arm's pivot; the board's pose for each pose of the joint is
drawing_frame(), from the joint's geometry as the scene has it (the ball's centre on the normal through the middle of
the board, H_BOARD behind its back). So the board's outline on the screen is known for every step (guide), the photo is
taken by itself when the board is there, still, and at a new stop (auto_shot new_stop), and a joint set to another pose
is told (identify).

On the joint the board reaches the corners of the image only to about 0.4-0.66 of the half diagonal (the calibration
asks for calibrate_charuco.COVER_CORNER): the plan ends with a few photos of the board held by hand near the corners of
the image (calib_guide.BENCH_HAND), and the calibration uses all the photos together.

The film of the bench: the arm stops a second at each notch. choose_stops() keeps one still frame per stop, never a
frame taken on the move; a stop where the arm never came still is said ("slower").

Scene frame W in mm; board frame M as calibrate_charuco's board (origin at its top left corner, x right, y down, z
into the board); camera C as OpenCV, the photo upright on the phone held across.

Usage:
    python calib_bench.py show      the steps: pose, arm angle, corners in view, tilt, distance
    python calib_bench.py test      every step sees the board well, the outlines tell the poses apart, the stops of
                                    a film are found and the moving ones said; exit 1 on failure
"""
from __future__ import annotations

import argparse
import functools
import math
import sys

import numpy as np

PIVOT = np.array([-15.78907585, -77.98216248, 0.0])        # the arm's pivot (scene, mm), about z
REF_ARM = 35.866612529942785                                # the arm angle of REF_T_W_C (the first trial position)
REF_T_W_C = np.array([[0.81042, 0.19652, -0.5519, 201.54981], [0.58585, -0.27206, 0.76339, -268.67637],
                      [-0.00013, -0.942, -0.33561, 104.26263], [0.0, 0.0, 0.0, 1.0]])     # the phone, cradle at 0
# The ball joint on the post (scene, mm): the bottom of its base on the adapter, the axis of the joint (azimuth,
# elevation, degrees) and the ball's centre this far along it from the base's bottom; the back of the board this far
# from the ball's centre along its neck (the stud's plate and a washer); the plate (its STL: x right, y up, z out of
# the front, the back at z 0) with the drawing's top left corner at BOARD_ORIGIN, centred on the neck, its top along
# the neck frame's -x
JOINT = {"base": (0.0, 19.0, -33.5), "axis_deg": (-60.0, 35.0), "ball_mm": 26.0}
_AZ, _EL = (np.radians(a) for a in JOINT["axis_deg"])
BALL_CENTRE = np.array(JOINT["base"]) + JOINT["ball_mm"] * np.array([np.cos(_EL) * np.cos(_AZ),
                                                                     np.cos(_EL) * np.sin(_AZ), np.sin(_EL)])
H_BOARD = 25.65
PLATE_MM = (240.0, 196.0, 6.0)
BOARD_ORIGIN = (6.0, 182.0, 6.0)
# The poses of the joint: pan psi (the board's normal, degrees from +x towards +y), lean (the normal's elevation: 0
# the board upright) and turn (the board about its normal, anticlockwise seen from the front); each photographed with
# the arm at the notches between the two of POSE_STOPS
POSES = {"P1": (-30.0, 20.0, 0.0), "P2": (-30.0, 10.0, -45.0), "P3": (-50.0, 0.0, -45.0), "P4": (-20.0, 50.0, 90.0),
         "P5": (-30.0, 60.0, 45.0), "P6": (-60.0, 60.0, 45.0), "P7": (-50.0, 70.0, 0.0)}
POSE_STOPS = {p: (30, 90) for p in POSES}
TRIALS = (35.867, 51.295, 66.019, 80.06)    # the arm at the four positions of the trials
NOTCHES = tuple(range(30, 91, 10))
MERGE_DEG = 2.0         # closer than this the two photos would be the same: 50 and 80 are positions 2 and 4
BOARD = {"squares": (11, 8), "square_mm": 20.0}       # the plate on the joint
WRONG = 1.0             # identify: a step of another pose this near (mean corner distance, of the photo) and nearer
SLIP = {"arm_deg": 0.3, "head_deg": 1.5}    # the simulation: the arm at its notch, the joint set by hand


def stops(pose):
    """The arm angles of a pose, in order, with their trial position (0: a notch only): the notches of POSE_STOPS, a
    notch within MERGE_DEG of a trial position being the position."""
    lo, hi = POSE_STOPS[pose]
    out = []
    for a in NOTCHES:
        if lo <= a <= hi:
            k = next((i + 1 for i, t in enumerate(TRIALS) if abs(a - t) <= MERGE_DEG), 0)
            out.append((float(TRIALS[k - 1]) if k else float(a), k))
    return out


STEPS = [{"view": "bench", "pose": p, "ball": list(POSES[p]), "arm": round(a, 2), "trial": k, "first": j == 0,
          "tilt": 0.0, "at": [0.5, 0.5], "dist": "mid", "roll": 0.0}
         for p in POSES for j, (a, k) in enumerate(stops(p))]


def _rz(deg):
    a = math.radians(deg)
    M = np.eye(4)
    M[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return M


def _ry(deg):
    a = math.radians(deg)
    M = np.eye(4)
    M[0, 0], M[0, 2], M[2, 0], M[2, 2] = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
    return M


def _at(v):
    M = np.eye(4)
    M[:3, 3] = v
    return M


def stem_frame(psi, lean, roll):
    """World matrix of the neck of the ball (origin at the ball's centre, z along the neck towards the board): what
    turns with the ball (its stud, the board)."""
    return _at(BALL_CENTRE) @ _rz(psi) @ _ry(90.0 - lean) @ _rz(roll)


def drawing_frame(psi, lean, roll):
    """T_W_M of the board's drawing for a pose of the joint (origin at its top left corner, x right, y down, z
    into the plate)."""
    pw, ph, _ = PLATE_MM
    stl = np.array([[0, -1, 0, ph / 2], [1, 0, 0, -pw / 2], [0, 0, 1, H_BOARD], [0, 0, 0, 1.0]])
    return stem_frame(psi, lean, roll) @ stl @ _at(BOARD_ORIGIN) @ np.diag([1.0, -1.0, -1.0, 1.0])


@functools.lru_cache(maxsize=None)
def square_arm(psi):
    """The arm angle (degrees) at which the board of pan psi, upright, is seen square from the side: the phone in the
    vertical plane of its normal (the yaw of guide() 0)."""
    def yaw(arm):
        T = np.linalg.inv(phone(arm)) @ drawing_frame(psi, 0.0, 0.0)
        R, t = T[:3, :3], T[:3, 3]
        eye = -R.T @ t - [110.0, 80.0, 0.0]
        return math.atan2(eye[0], -eye[2])
    lo, hi = 0.0, 120.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if (yaw(lo) > 0) == (yaw(mid) > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def face(pose):
    """The arm angle the board of a pose faces (degrees, to 5): with the arm there, the pan is turned until the board
    faces the phone squarely."""
    return int(5 * round(square_arm(POSES[pose][0]) / 5))


def phone(arm):
    """T_W_C of the phone with the arm at `arm` degrees (the cradle at 0)."""
    a = math.radians(arm - REF_ARM)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    M = np.eye(4)
    M[:3, :3] = Rz
    M[:3, 3] = PIVOT - Rz @ PIVOT
    return M @ REF_T_W_C


def camera_pose(step, rng=None):
    """T_C_M of a step; with rng, as the bench gives it: the arm within SLIP arm_deg of its notch, each angle of the
    joint within SLIP head_deg."""
    arm, ball = step["arm"], list(step["ball"])
    if rng is not None:
        arm += rng.uniform(-SLIP["arm_deg"], SLIP["arm_deg"])
        ball = [b + rng.uniform(-SLIP["head_deg"], SLIP["head_deg"]) for b in ball]
    return np.linalg.inv(phone(arm)) @ drawing_frame(*ball)


def _project(T, pts, K):
    cam = np.asarray(pts, np.float64) @ T[:3, :3].T + T[:3, 3]
    return cam[:, :2] / cam[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]], cam[:, 2]


def outline(step, spec, K, size):
    """The board's four corners (top left, top right, bottom right, bottom left) at a step, fractions of the photo."""
    nx, ny = spec["squares"]
    W, H = nx * spec["square_mm"], ny * spec["square_mm"]
    uv, _ = _project(camera_pose(step), [[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], K)
    return uv / np.array(size, np.float64)


def guide(step, spec, K, size, n=0, plan="bench", of=None):
    """What the page draws for a step, as calib_guide.guide: the board's outline and black squares on the screen
    (fractions), the side view (yaw, pitch, distance, roll), the arm angle, and the pose of the joint: its name,
    the arm angle the board faces (face), how far it leans back and how far it is turned in its plane (turn, > 0
    anticlockwise seen from the front); first: the pose's first step (the joint to set)."""
    w, h = size
    T = camera_pose(step)
    nx, ny = spec["squares"]
    sq = spec["square_mm"]
    W, H = nx * sq, ny * sq

    def norm(pts):
        uv, _ = _project(T, pts, K)
        return [[round(float(u / w), 4), round(float(v / h), 4)] for u, v in uv]
    cells = [norm([[i * sq, j * sq, 0], [(i + 1) * sq, j * sq, 0], [(i + 1) * sq, (j + 1) * sq, 0],
                   [i * sq, (j + 1) * sq, 0]]) for j in range(ny) for i in range(nx) if (i + j) % 2 == 0]
    R, t = T[:3, :3], T[:3, 3]
    eye = -R.T @ t - [W / 2, H / 2, 0]                  # the phone from the board's centre, board frame
    d = float(np.linalg.norm(eye))
    yaw = math.degrees(math.atan2(eye[0], -eye[2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -eye[1] / d))))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    centre, _ = _project(T, [[W / 2, H / 2, 0]], K)
    _, lean, turn = step["ball"]
    return {"n": n, "plan": plan, "of": len(STEPS) if of is None else of, "view": "bench", "pose": step["pose"],
            "face": face(step["pose"]), "lean": round(lean), "turn": round(turn), "first": bool(step.get("first")),
            "arm": round(step["arm"]), "trial": step["trial"], "tilt": round(math.hypot(yaw, pitch)),
            "yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll), "dist": "mid",
            "dist_cm": int(5 * round(d / 50)), "aspect": w / h, "outline": norm([[0, 0, 0], [W, 0, 0], [W, H, 0],
                                                                                  [0, H, 0]]),
            "cells": cells, "at": [round(float(centre[0, 0] / w), 4), round(float(centre[0, 1] / h), 4)],
            "board": [round(W, 1), round(H, 1)]}


def identify(seen, want, spec, K, size):
    """Which step the board seen (four corners, fractions of the photo) is at: (index of the nearest step, its mean
    corner distance, the pose of the joint when it is set to another pose than step `want`'s, else None). A `want`
    past the steps of the joint (the photos by hand at the end of the plan): (None, inf, None)."""
    if not 0 <= int(want) < len(STEPS):
        return None, math.inf, None
    seen = np.asarray(seen, np.float64)
    d = [float(np.mean(np.hypot(*(outline(s, spec, K, size) - seen).T))) for s in STEPS]
    best = int(np.argmin(d))
    pose = STEPS[want]["pose"]
    near_want = min(d[i] for i, s in enumerate(STEPS) if s["pose"] == pose)
    other = STEPS[best]["pose"] != pose
    wrong = STEPS[best]["pose"] if other and d[best] < WRONG * near_want and d[best] < 0.05 else None
    return best, d[best], wrong


# ── the film: one still frame per stop ───────────────────────────────────────
STOP_FACTOR = 4.0       # a stop: the corners moving less than this many times the film's still limit ...
STOP_MIN_S = 0.4        # ... for at least this long


def choose_stops(recs, still_px_s, describe=None):
    """The frames of a bench film to calibrate on, one per stop (indexes into recs, by time), and the times (s) of the
    stops where the arm never came still ("slower"). recs as calib_video.judge leaves them (n, t, motion, why,
    quality); a stop is a run of frames with the board whose corners move less than STOP_FACTOR times still_px_s for
    STOP_MIN_S at least; in it the usable frame (why "") of the best quality. describe(rec), when given, sets
    rec["pose"] for each frame chosen."""
    order = sorted(range(len(recs)), key=lambda k: recs[k]["t"])
    runs, cur = [], []
    for k in order:
        r = recs[k]
        if r["n"] and r.get("why") in ("", "moving", "blur") and r["motion"] < STOP_FACTOR * still_px_s:
            cur.append(k)
            continue
        if cur:
            runs.append(cur)
        cur = []
    if cur:
        runs.append(cur)
    picked, slower = [], []
    for run in runs:
        if recs[run[-1]]["t"] - recs[run[0]]["t"] < STOP_MIN_S:
            continue
        good = [k for k in run if recs[k].get("why") == ""]
        if not good:
            slower.append(round(float(recs[run[len(run) // 2]]["t"]), 2))
            continue
        k = max(good, key=lambda j: recs[j]["quality"])
        if describe is not None:
            recs[k]["pose"] = describe(recs[k])
        picked.append(k)
    return picked, slower


# ── test ─────────────────────────────────────────────────────────────────────
FAILS = []


def expect(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def _camera():
    K = np.array([[3070.0, 0, 2039.5], [0, 3070.0, 1535.5], [0, 0, 1]])
    return K, (4080, 3072)


def corners_in(step, spec=BOARD, K=None, size=None, edge=20):
    """The inner corners of the board inside the photo (edge px from its sides) at a step (px, N x 2)."""
    if K is None:
        K, size = _camera()
    nx, ny = spec["squares"]
    q = spec["square_mm"]
    pts = [[i * q, j * q, 0] for j in range(1, ny) for i in range(1, nx)]
    uv, z = _project(camera_pose(step), pts, K)
    w, h = size
    return uv[(z > 50) & (uv[:, 0] >= edge) & (uv[:, 0] < w - edge) & (uv[:, 1] >= edge) & (uv[:, 1] < h - edge)]


def seen_corners(step, spec=BOARD, K=None, size=None, edge=20):
    """How many inner corners of the board fall inside the photo at a step."""
    return len(corners_in(step, spec, K, size, edge))


def run_test():
    K, size = _camera()
    per = {p: len(stops(p)) for p in POSES}
    arms = sorted({s["arm"] for s in STEPS})
    expect(len(STEPS) == sum(per.values()) == 49 and set(per.values()) == {7}
           and any(abs(a - 51.295) < 0.01 for a in arms) and 80.06 in arms and 50.0 not in arms
           and sum(s["first"] for s in STEPS) == 7,
           f"{len(STEPS)} steps: the seven poses at their notches {per} (50 and 80 are positions 2 and 4)")
    # the board's drawing on the joint where the plate's own file has it
    try:
        import charuco_plate
        meta = charuco_plate.load()["plate"]
        same = (tuple(meta["size_mm"]) == PLATE_MM and tuple(meta["board_origin_mm"]) == BOARD_ORIGIN
                and meta["board_axes"] == {"x": [1, 0, 0], "y": [0, -1, 0], "z": [0, 0, -1]})
    except (ImportError, OSError, KeyError, ValueError) as e:
        same = False
        meta = {"error": str(e)}
    D = drawing_frame(0.0, 0.0, 0.0)
    middle = D @ [110.0, 80.0, 0.0, 1.0]
    expect(same and np.allclose(D[:3, :3] @ D[:3, :3].T, np.eye(3)) and np.allclose(D[:3, 2], [-1, 0, 0])
           and np.allclose(D[:3, 1], [0, 0, -1]) and abs(middle[0] - BALL_CENTRE[0] - H_BOARD - PLATE_MM[2]) < 1e-9,
           f"the drawing on the joint: plate {PLATE_MM}, upright facing +x at pan 0, its middle at "
           f"{np.round(middle[:3], 1)} (the plate's file: {meta.get('size_mm', meta)})")
    low = [(s["pose"], s["arm"], seen_corners(s)) for s in STEPS if seen_corners(s) < 24]
    expect(not low, f"every step keeps at least 24 corners in the photo {low or ''}")
    gs = [guide(s, BOARD, K, size) for s in STEPS]
    tilts, dists = [g["tilt"] for g in gs], [g["dist_cm"] for g in gs]
    expect(min(tilts) >= 3 and max(tilts) <= 62 and 20 <= min(dists) and max(dists) <= 35,
           f"tilts {min(tilts)}-{max(tilts)} degrees, distances {min(dists)}-{max(dists)} cm")
    # the steps of the joint cover the image but not its corners: the photos by hand do (calib_guide)
    import calibrate_charuco
    cov = calibrate_charuco.coverage([corners_in(s) for s in STEPS], size)
    expect(not cov["missing_cells"] and cov["reach"] < calibrate_charuco.COVER_CORNER,
           f"the joint's steps reach all {sum(map(len, cov['cells']))} cells of the image, its corners only "
           f"{100 * cov['reach']:.0f} % of the way (the photos by hand for the rest)")
    # the pan said as the arm angle the board faces: there the board is seen square from the side
    FACES = [65, 65, 40, 75, 65, 30, 40]
    yaws = {p: guide({**STEPS[0], "pose": p, "ball": [POSES[p][0], 0.0, 0.0], "arm": face(p)}, BOARD, K, size)["yaw"]
            for p in POSES}
    expect(all(abs(y) < 4 for y in yaws.values()) and [face(p) for p in POSES] == FACES,
           f"the board faces the arm at the angle said: yaw {yaws} degrees there")
    g = gs[0]
    ok = (all(-0.2 < u < 1.2 and -0.2 < v < 1.2 for u, v in g["outline"]) and g["cells"] and g["view"] == "bench"
          and g["pose"] == "P1" and g["face"] == FACES[0] and g["first"] and not gs[1]["first"])
    expect(ok, f"the guide of step 1: {g['pose']}, facing the arm at {g['face']}, arm {g['arm']}, outline "
               f"{g['outline'][0]} ...")
    # the outline tells the pose: the board of every step, seen where it is, is that step; asked for another pose at
    # the same arm angle, the pose the joint is set to is said; past the steps (by hand) nothing
    rng = np.random.default_rng(1)
    hits, wrong, asked = 0, 0, 0
    nx, ny = BOARD["squares"]
    W, H = nx * BOARD["square_mm"], ny * BOARD["square_mm"]
    for i, s in enumerate(STEPS):
        uv, _ = _project(camera_pose(s, rng), [[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], K)
        seen = uv / np.array(size, np.float64)
        best, d, _ = identify(seen, i, BOARD, K, size)
        hits += STEPS[best]["pose"] == s["pose"] and abs(STEPS[best]["arm"] - s["arm"]) < 2
        for j, t in enumerate(STEPS):
            if t["pose"] != s["pose"] and abs(t["arm"] - s["arm"]) < 0.1:
                asked += 1
                wrong += identify(seen, j, BOARD, K, size)[2] == s["pose"]
    by_hand = identify([[0, 0], [1, 0], [1, 1], [0, 1]], len(STEPS), BOARD, K, size)
    expect(hits == len(STEPS) and wrong == asked and by_hand[0] is None and by_hand[2] is None,
           f"the outline tells the step ({hits} of {len(STEPS)}) and a joint in another pose ({wrong} of {asked})")
    # the film: three stops, one still, one where the arm crept, one too short; the moves between them
    recs, t = [], 0.0
    plan = [("move", 1.0), ("still", 1.0), ("move", 1.0), ("creep", 1.0), ("move", 1.0), ("still", 0.2),
            ("move", 1.0), ("still", 1.2)]
    for kind, dur in plan:
        for _ in range(int(round(dur * 10))):
            m = {"move": 300.0, "still": 5.0, "creep": 60.0}[kind]
            recs.append({"i": len(recs), "t": round(t, 2), "n": 50, "motion": m, "why": "" if m <= 30 else "moving",
                         "quality": 0.5 + 0.01 * len(recs)})
            t += 0.1
    picked, slower = choose_stops(recs, 30.0)
    expect(len(picked) == 2 and len(slower) == 1 and all(recs[k]["motion"] <= 30 for k in picked),
           f"a film of the bench: {len(picked)} frames, one per still stop, the stop on the move said at {slower} s")
    print(f"\n{len(FAILS)} failures" if FAILS else "\nall checks passed", flush=True)
    return 1 if FAILS else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("show", "test"))
    a = ap.parse_args()
    if a.cmd == "test":
        return run_test()
    K, size = _camera()
    for i, s in enumerate(STEPS):
        g = guide(s, BOARD, K, size)
        print(f"{i + 1:2d} {s['pose']} (face {g['face']:3d}, lean {g['lean']:2d}, turn {g['turn']:3d})"
              f" arm {s['arm']:6.2f} {'position ' + str(s['trial']) if s['trial'] else '':10s}"
              f" corners {seen_corners(s):2d}  tilt {g['tilt']:3d}  {g['dist_cm']} cm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
