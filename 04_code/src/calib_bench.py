"""The calibration on the bench: the phone on the arm, the 3D-printed plate standing in the seats of the board holder.

The head is taken off its post, the pendulum latched up and the board holder put on the post in its place. For each of
its three seats the arm goes from one end to the other and stops at every 10-degree notch from 30 to 90 degrees and at
the four positions of the trials (a notch within MERGE_DEG of a position is the position): STEPS, one photo each. The
phone's pose at any arm angle is the turn of the scene's camera about the arm's pivot; each seat's board pose comes
from the scene (the board holder study). So the board's outline on the screen is known for every step (guide), the
photo is taken by itself when the board is there, still, and at a new stop (auto_shot new_stop), and a board in
another seat is told (identify).

The film of the bench: the arm stops a second at each notch. choose_stops() keeps one still frame per stop, never a
frame taken on the move; a stop where the arm never came still is said ("slower").

Scene frame W in mm; board frame M as calibrate_charuco's board (origin at its top left corner, x right, y down, z
into the board); camera C as OpenCV, the photo upright on the phone held across.

Usage:
    python calib_bench.py show      the steps: seat, arm angle, corners in view, tilt, distance
    python calib_bench.py test      every step sees the board well, the outlines tell the seats apart, the stops of
                                    a film are found and the moving ones said; exit 1 on failure
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

PIVOT = np.array([-15.78907585, -77.98216248, 0.0])        # the arm's pivot (scene, mm), about z
REF_ARM = 35.866612529942785                                # the arm angle of REF_T_W_C (the first trial position)
REF_T_W_C = np.array([[0.81042, 0.19652, -0.5519, 201.54981], [0.58585, -0.27206, 0.76339, -268.67637],
                      [-0.00013, -0.942, -0.33561, 104.26263], [0.0, 0.0, 0.0, 1.0]])     # the phone, cradle at 0
SEATS = [np.array(T) for T in (      # T_W_M of the board in seat 1 (low), 2 (high, behind), 3 (right)
    [[0.309, 0.0829, -0.9474, -45.3841], [0.9511, -0.0269, 0.3078, -194.7502], [0.0, -0.9962, -0.0872, 56.8171],
     [0, 0, 0, 1.0]],
    [[0.1392, 0.0863, -0.9865, -73.6986], [0.9903, -0.0121, 0.1386, -205.2093], [0.0, -0.9962, -0.0872, 116.8171],
     [0, 0, 0, 1.0]],
    [[0.309, 0.2462, -0.9187, -4.6769], [0.9511, -0.08, 0.2985, -52.6536], [0.0, -0.9659, -0.2588, 91.3082],
     [0, 0, 0, 1.0]])]
TRIALS = (35.867, 51.295, 66.019, 80.06)    # the arm at the four positions of the trials
NOTCHES = tuple(range(30, 91, 10))
MERGE_DEG = 2.0         # closer than this the two photos would be the same: 50 and 80 are positions 2 and 4
BOARD = {"squares": (11, 8), "square_mm": 20.0}       # the plate the seats are made for
WRONG = 1.0             # identify: a step of another seat this near (mean corner distance, of the photo) and nearer


def stops():
    """The arm angles of one seat, in order: the notches and the trial positions (a notch within MERGE_DEG of a
    position is the position)."""
    out = [(float(a), i + 1) for i, a in enumerate(TRIALS)]
    out += [(float(a), 0) for a in NOTCHES if all(abs(a - t) > MERGE_DEG for t in TRIALS)]
    return sorted(out)


STEPS = [{"view": "bench", "seat": s + 1, "arm": round(a, 2), "trial": k, "tilt": 0.0, "at": [0.5, 0.5],
          "dist": "mid", "roll": 0.0} for s in range(len(SEATS)) for a, k in stops()]


def phone(arm):
    """T_W_C of the phone with the arm at `arm` degrees (the cradle at 0)."""
    a = math.radians(arm - REF_ARM)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    M = np.eye(4)
    M[:3, :3] = Rz
    M[:3, 3] = PIVOT - Rz @ PIVOT
    return M @ REF_T_W_C


def camera_pose(step, rng=None):
    """T_C_M of a step; with rng, as the bench gives it: the arm within 0.3 degrees, the board within 1 mm along its
    seat."""
    arm, T = step["arm"], SEATS[step["seat"] - 1].copy()
    if rng is not None:
        arm += rng.uniform(-0.3, 0.3)
        T[:3, 3] += T[:3, 0] * rng.uniform(-1.0, 1.0)
    return np.linalg.inv(phone(arm)) @ T


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
    (fractions), the side view (yaw, pitch, distance, roll), and the seat and arm angle."""
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
    return {"n": n, "plan": plan, "of": len(STEPS) if of is None else of, "view": "bench", "seat": step["seat"],
            "arm": round(step["arm"]), "trial": step["trial"], "tilt": round(math.hypot(yaw, pitch)),
            "yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll), "dist": "mid",
            "dist_cm": int(5 * round(d / 50)), "aspect": w / h, "outline": norm([[0, 0, 0], [W, 0, 0], [W, H, 0],
                                                                                  [0, H, 0]]),
            "cells": cells, "at": [round(float(centre[0, 0] / w), 4), round(float(centre[0, 1] / h), 4)],
            "board": [round(W, 1), round(H, 1)]}


def identify(seen, want, spec, K, size):
    """Which step the board seen (four corners, fractions of the photo) is at: (index of the nearest step, its mean
    corner distance, the seat of the board when it stands in another seat than step `want`'s, else None)."""
    seen = np.asarray(seen, np.float64)
    d = [float(np.mean(np.hypot(*(outline(s, spec, K, size) - seen).T))) for s in STEPS]
    best = int(np.argmin(d))
    other = STEPS[best]["seat"] != STEPS[want]["seat"]
    near_want = min(d[i] for i, s in enumerate(STEPS) if s["seat"] == STEPS[want]["seat"])
    wrong = STEPS[best]["seat"] if other and d[best] < WRONG * near_want and d[best] < 0.05 else None
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


def seen_corners(step, spec=BOARD, K=None, size=None, edge=20):
    """Inner corners of the board inside the photo (edge px from its sides) at a step."""
    if K is None:
        K, size = _camera()
    nx, ny = spec["squares"]
    q = spec["square_mm"]
    pts = [[i * q, j * q, 0] for j in range(1, ny) for i in range(1, nx)]
    uv, z = _project(camera_pose(step), pts, K)
    w, h = size
    return int(np.sum((z > 50) & (uv[:, 0] >= edge) & (uv[:, 0] < w - edge) & (uv[:, 1] >= edge) & (uv[:, 1] < h - edge)))


def run_test():
    K, size = _camera()
    per_seat = len(stops())
    expect(len(STEPS) == 3 * per_seat and per_seat == 9 and [s for s, _ in stops()][0] == 30.0,
           f"{len(STEPS)} steps, {per_seat} a seat: the notches 30-90 and the four positions (50 and 80 are positions 2 and 4)")
    low = [(i, seen_corners(s)) for i, s in enumerate(STEPS) if seen_corners(s) < 24]
    expect(not low, f"every step keeps at least 24 corners in the photo {low or ''}")
    tilts = [guide(s, BOARD, K, size)["tilt"] for s in STEPS]
    dists = [guide(s, BOARD, K, size)["dist_cm"] for s in STEPS]
    expect(min(tilts) >= 3 and max(tilts) <= 55 and 25 <= min(dists) and max(dists) <= 40,
           f"tilts {min(tilts)}-{max(tilts)} degrees, distances {min(dists)}-{max(dists)} cm")
    g = guide(STEPS[0], BOARD, K, size)
    ok = all(-0.2 < u < 1.2 and -0.2 < v < 1.2 for u, v in g["outline"]) and g["cells"] and g["view"] == "bench"
    expect(ok, f"the guide of step 1: seat {g['seat']}, arm {g['arm']}, outline {g['outline'][0]} ...")
    # the outline tells the seat: the board of every step, seen where it is, is that step; asked for the same arm angle
    # in another seat, the seat it stands in is said
    rng = np.random.default_rng(1)
    hits, wrong = 0, 0
    for i, s in enumerate(STEPS):
        T = camera_pose(s, rng)
        nx, ny = BOARD["squares"]
        W, H = nx * BOARD["square_mm"], ny * BOARD["square_mm"]
        uv, _ = _project(T, [[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], K)
        seen = uv / np.array(size, np.float64)
        best, d, _ = identify(seen, i, BOARD, K, size)
        hits += STEPS[best]["seat"] == s["seat"] and abs(STEPS[best]["arm"] - s["arm"]) < 2
        other = next(j for j, t in enumerate(STEPS) if t["seat"] != s["seat"] and abs(t["arm"] - s["arm"]) < 0.1)
        wrong += identify(seen, other, BOARD, K, size)[2] == s["seat"]
    expect(hits == len(STEPS) and wrong == len(STEPS),
           f"the outline tells the step ({hits} of {len(STEPS)}) and a board in the wrong seat ({wrong} of {len(STEPS)})")
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
        print(f"{i + 1:2d} seat {s['seat']} arm {s['arm']:6.2f} {'position ' + str(s['trial']) if s['trial'] else '':10s}"
              f" corners {seen_corners(s):2d}  tilt {g['tilt']:3d}  {g['dist_cm']} cm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
