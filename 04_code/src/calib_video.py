"""Calibration from a film of the board: the frames worth calibrating on, chosen from the film itself.

The operator films the calibration board moving the phone slowly through the poses of a plan (the same poses as the
photos: tilted every way, near and far, the board in every part of the frame), holding still for about a second at
each. The film is read once, frame after frame. On SCAN_FPS frames a second the board's corners are found with
calibrate_charuco's detector, refined as the photos' are (calibrate_charuco.refine_corners), and followed with optical
flow to the frames just before and after: how far they move is the frame's motion (px/s). A frame can be used when
    corners   it shows at least FRAME_MIN_SHARE of the board's corners (and calibrate_charuco.MIN_CORNERS), not on one
              line;
    still     its corners move less than MOTION_PX_S: the phone's rolling shutter reads the rows one after the other,
              so a moving board comes out skewed, and a calibration would take the skew for the lens (at
              MOTION_PX_S, about 1 px a frame at 30 fps, a phone held still by hand, the skew stays near half a
              pixel over the whole height of the frame);
    sharp     its blur (calibrate_charuco.blur_px, the sigma of a Gaussian) is under FILM_BLUR_PX.
FILM_BLUR_PX and MOTION_PX_S are for a frame 1920 px on its long side and scale with it. Among the frames that can be
used the frames are then chosen one at a time: the next one is the frame that adds the most, its quality (corners,
sharpness, stillness) times how far its pose is from every pose already chosen (tilt and its direction, distance,
roll, where the board sits in the frame), plus the cells of the frame its corners cover first, so the corners of the
image are reached too. A frame closer than DUPLICATE to one already chosen is a near duplicate and never chosen. It
stops at the number of frames asked for (the plan's), or when no frame adds a pose not yet had.

The frames chosen are written as lossless grey PNG beside each other (VID_..._f00123.png), so the folder calibrates
as a folder of photos (calibrate_charuco.run_calibrate); each is checked against the frame first read (CRC), so a
decoder that gives another frame on the second pass is said, never calibrated on. Beside them, in film/, the film
itself, frames.json (every frame looked at: corners, motion, blur, why it was not used, which were chosen) and
chosen.jpg (the frames chosen, small, with their time in the film).

Camera modes. Intrinsics hold only for the camera mode their images were taken in: the same size, the same part of
the sensor, the same processing, the same focus. The phone's camera app records its video in another mode than its
photos: 16:9 (1920 x 1080 or 3840 x 2160) out of the 4:3 sensor, scaled differently, with electronic stabilisation
when it is on (each frame then moved and cropped by its own amount: no single camera model fits) and continuous
autofocus unless the focus is locked. A calibration from a film of the camera app is therefore the calibration of
that video mode ("video": the trial videos from the arm), never of the photos ("photo": 12.5 MP JPEG) nor of the live
stream ("stream": scrcpy, 1920 x 1440). Each mode keeps its own file (intrinsics_<mode>.json) with its mode and image
size written in it, and a calibration is used only for frames of its own mode and size. narrower_than() says when a
film sees a narrower field than the photos would at its size: stabilisation (or a zoom) was on.

Usage:
    python calib_video.py choose VIDEO OUT [--target 40] [--square MM --squares NXxNY --marker MM]
        the frames of a film worth calibrating on, written into OUT (frames.json and chosen.jpg in OUT/film)
    python calib_video.py calibrate VIDEO [--target 40] [--out FILE] [--truth FILE] [--work DIR] [board options]
        choose the frames, then calibrate on them; compared with the truth of a synthetic film if given
    python calib_video.py synth OUT.mp4 [--plan quick|accurate] [--size 1280x720] [--fps 30] [--seed 0]
        a synthetic film of the A4 board (rolling shutter, hand tremor, motion blur, H.264), truth beside it
    python calib_video.py compare CAL REF
        two calibrations of different modes side by side, scaled to the same field (focal, centre, distortion)
    python calib_video.py test [--work DIR]
        synthetic films: the frames chosen are still, sharp and of different poses, and calibrate within the limits;
        a film always moving, one without the board and one that cannot be read are said as such; exit 1 on failure

The board: --square, --squares and --marker, else the board.json of OUT or of the film's folder, else the A4 board.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import tempfile
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np

import calibrate_charuco as cc

WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact")) / "calib_video"
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".mkv", ".h264", ".avi")
FILM_DIR = "film"                   # beside the frames chosen: the film, frames.json, chosen.jpg
FRAMES_FILE = "frames.json"
SHEET_FILE = "chosen.jpg"
REF_LONG = 1920                     # px: the long side the limits below are given for

SCAN_FPS = 10.0                     # frames looked at per second of film
FRAME_MIN_SHARE = 0.25              # of the board's corners, at least, in a frame
MOTION_PX_S = 30.0                  # px/s at most (90th percentile of the corners), at REF_LONG: 1 px a frame at 30 fps
FILM_BLUR_PX = 2.0                  # blur sigma at most, px at REF_LONG
NOMINAL_F_SHARE = 0.752             # the focal length of the Poco X7 Pro main camera over the frame's long side
# how far two poses are, in steps (a sum of squares): a step of tilt, of distance (log), of place in the frame, of roll
TILT_STEP, DIST_STEP, PLACE_STEP, ROLL_STEP = 10.0, 0.15, 0.15, 15.0
DUPLICATE = 1.0                     # closer than this to a frame chosen: a near duplicate
NOVELTY_CAP = 3.0                   # farther than this counts as this: then quality and coverage decide
COVER_GRID = (8, 6)                 # cells of the frame for its coverage
COVER_WEIGHT = 0.3                  # per cell covered first
HOLDOUT = 40                        # frames that could be used but were not, reprojected to judge the model
SHEET_THUMB = 320                   # px: the width of a frame in chosen.jpg
WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))

# the codes of the CalibrationError raised here (with calibrate_charuco.CODES for the others)
CODES = ("film_unreadable", "film_no_board", "film_few", "film_frames_differ")

Log = Callable[[str], None]
Progress = Callable[[float], None]


def _say(log):
    return log if log is not None else print


# ── reading a film ───────────────────────────────────────────────────────────
def _open(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise cc.CalibrationError("film_unreadable", f"{Path(path).name} cannot be opened", file=Path(path).name)
    return cap


def probe(path):
    """{size, fps, frames (None when the container does not say), codec, name} of a film; raises film_unreadable."""
    path = Path(path)
    if not path.is_file():
        raise cc.CalibrationError("film_unreadable", f"{path} is not a file", file=path.name)
    cap = _open(path)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise cc.CalibrationError("film_unreadable", f"{path.name}: no frame can be decoded", file=path.name)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((code >> 8 * k) & 0xFF) for k in range(4)).strip("\0 ") if code > 0 else ""
    finally:
        cap.release()
    times = _pts(path)
    if times is not None and len(times) > 1:            # a raw stream recorded by live.py: its capture times
        fps = (len(times) - 1) / max(float(times[-1] - times[0]), 1e-6)
    return {"name": path.name, "size": [int(frame.shape[1]), int(frame.shape[0])], "fps": round(fps, 3) if fps > 0 else 30.0,
            "frames": n if n > 0 else None, "codec": codec}


def _pts(path):
    """The capture times (s) of a raw stream recorded by live.py (<stem>_pts.npy beside it), or None."""
    p = Path(path).with_name(Path(path).stem + "_pts.npy")
    if not p.is_file():
        return None
    with contextlib.suppress(OSError, ValueError):
        t = np.load(p).astype(np.float64).ravel()
        return t - t[0] if len(t) else None
    return None


def frames(path):
    """(index, time in s, grey frame) of every frame of a film, in order. The time is the container's, the capture
    time of a raw stream recorded by live.py, else index / fps."""
    info = probe(path)
    fps = info["fps"]
    pts = _pts(path)
    cap = _open(path)
    last = -1.0
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if pts is not None and i < len(pts):
                t = float(pts[i])
            else:
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = ms / 1000.0 if ms and ms > 0 and ms / 1000.0 > last else i / fps
            last = t
            yield i, t, (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame)
            i += 1
    finally:
        cap.release()


# ── the frames looked at ─────────────────────────────────────────────────────
def _motion(cur, other, pts, dt):
    """px/s the corners move from frame cur to frame other, dt apart (90th percentile; inf when they are lost)."""
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(cur, other, pts, None, winSize=(21, 21), maxLevel=3, criteria=crit)
    if p1 is None:
        return math.inf
    ok = st.ravel() == 1
    if ok.sum() < max(4, 0.5 * len(pts)):
        return math.inf
    d = np.linalg.norm((p1 - pts).reshape(-1, 2)[ok], axis=1)
    return float(np.percentile(d, 90)) / max(dt, 1e-6)


def _look(board, local, i, t, prev, cur, nxt, tp, tn):
    """What frame i shows: its corners (ids, image and board points), collinear, motion (px/s), blur, CRC."""
    det = getattr(local, "det", None)
    if det is None:
        det = local.det = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = det.detectBoard(cur)
    rec = {"i": i, "t": t, "n": 0, "crc": zlib.crc32(cur.tobytes())}
    if ids is None or len(ids) == 0:
        return rec
    obj, imgp = board.matchImagePoints(corners, ids)
    collinear = bool(board.checkCharucoCornersCollinear(ids))
    imgp = imgp.reshape(-1, 2).astype(np.float32)
    if len(ids) >= 4 and not collinear:     # refined as the photos are (calibrate_charuco.refine_corners)
        imgp = cc.refine_corners(cur, obj, imgp, board.getSquareLength())
    rec.update(n=int(len(ids)), ids=ids.ravel().astype(int), obj=obj.reshape(-1, 3).astype(np.float32),
               img=imgp, collinear=collinear)
    pts = rec["img"].reshape(-1, 1, 2)
    speeds = [_motion(cur, o, pts, abs(to - t)) for o, to in ((prev, tp), (nxt, tn)) if o is not None]
    rec["motion"] = max(speeds) if speeds else math.inf
    if rec["n"] >= 4 and not rec["collinear"]:
        rec["blur"] = cc.blur_px(cur, rec["obj"], rec["img"], board.getSquareLength())
    return rec


def scan(path, board, log: Log | None = None, progress: Progress | None = None, every: int | None = None):
    """Every frame looked at (SCAN_FPS a second, or one in `every`), as dicts: i, t, n (corners), ids, obj, img,
    collinear, motion (px/s), blur (px), crc; and the film's probe() with the number of frames read. Raises
    film_unreadable."""
    info = probe(path)
    stride = every or max(1, int(round(info["fps"] / SCAN_FPS)))
    n_all = info["frames"]
    local = threading.local()
    out, pending = [], deque()
    window = deque(maxlen=3)            # (i, t, grey): the frame before, the frame, the frame after
    last_said = time.monotonic()
    count = 0

    def submit(pool, before, here, after):
        i, t, cur = here
        prev, tp = (before[2], before[1]) if before else (None, 0.0)
        nxt, tn = (after[2], after[1]) if after else (None, 0.0)
        pending.append(pool.submit(_look, board, local, i, t, prev, cur, nxt, tp, tn))
        while len(pending) > 3 * WORKERS:
            out.append(pending.popleft().result())

    with ThreadPoolExecutor(WORKERS, thread_name_prefix="film") as pool:
        for item in frames(path):
            count += 1
            window.append(item)
            if len(window) >= 2:
                here = window[-2]
                if here[0] % stride == 0:
                    submit(pool, window[-3] if len(window) == 3 else None, here, window[-1])
            if progress is not None and time.monotonic() - last_said > 0.5:
                last_said = time.monotonic()
                progress(min(0.99, count / n_all) if n_all else 0.0)
        if window and window[-1][0] % stride == 0:
            submit(pool, window[-2] if len(window) >= 2 else None, window[-1], None)
        while pending:
            out.append(pending.popleft().result())
    if count == 0:
        raise cc.CalibrationError("film_unreadable", f"{info['name']}: no frame", file=info["name"])
    out.sort(key=lambda r: r["i"])
    info.update(frames=count, every=stride, duration_s=round(out[-1]["t"] if out else 0.0, 3))
    _say(log)(f"film {info['name']}: {count} frames {info['size'][0]} x {info['size'][1]} at {info['fps']:.2f} fps, "
              f"{len(out)} looked at, {sum(1 for r in out if r['n'])} with the board")
    return out, info


# ── choosing the frames ──────────────────────────────────────────────────────
def limits(size):
    """{motion_px_s, blur_px} for frames of this size (the long side against REF_LONG)."""
    s = max(size) / REF_LONG
    return {"motion_px_s": MOTION_PX_S * s, "blur_px": max(1.2, FILM_BLUR_PX * s)}


def nominal_K(size):
    w, h = size
    f = NOMINAL_F_SHARE * max(w, h)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def describe(rec, board, K, size):
    """The pose of a frame in plain terms: tilt (degrees, and its direction as tx, ty), distance of the board's centre
    (mm), roll (degrees: the board's x in the frame), where its corners sit (their centre, fractions of the frame)."""
    ok, rvec, tvec = cv2.solvePnP(rec["obj"], rec["img"], K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(rec["obj"], rec["img"], K, None)
    R = cv2.Rodrigues(rvec)[0]
    nrm = R[:, 2]
    tilt = math.degrees(math.acos(min(1.0, abs(float(nrm[2])))))
    side = math.hypot(nrm[0], nrm[1])
    tx, ty = (tilt * nrm[0] / side, tilt * nrm[1] / side) if side > 1e-9 else (0.0, 0.0)
    nx, ny = board.getChessboardSize()
    sq = board.getSquareLength()
    centre = R @ np.array([nx * sq / 2, ny * sq / 2, 0.0]) + tvec.ravel()
    ex = (cv2.projectPoints(np.array([[nx * sq / 2, ny * sq / 2, 0], [nx * sq / 2 + sq, ny * sq / 2, 0]], np.float64),
                            rvec, tvec, K, None)[0].reshape(-1, 2))
    roll = math.degrees(math.atan2(ex[1, 1] - ex[0, 1], ex[1, 0] - ex[0, 0]))
    u, v = rec["img"].mean(0) / np.array(size, np.float64)
    return {"tilt": tilt, "tx": tx, "ty": ty, "dist": float(np.linalg.norm(centre)), "roll": roll,
            "at": [float(u), float(v)]}


def _features(descs):
    a = np.array([[d["tx"] / TILT_STEP, d["ty"] / TILT_STEP, math.log(max(d["dist"], 1.0)) / DIST_STEP,
                   d["at"][0] / PLACE_STEP, d["at"][1] / PLACE_STEP] for d in descs], np.float64)
    return a, np.array([d["roll"] for d in descs], np.float64)


def _distance(X, roll, j):
    """How far each pose is from pose j, in steps."""
    d2 = np.sum((X - X[j]) ** 2, axis=1)
    dr = (roll - roll[j] + 180.0) % 360.0 - 180.0
    return np.sqrt(d2 + (dr / ROLL_STEP) ** 2)


def _cells(img, size):
    gx, gy = COVER_GRID
    w, h = size
    cx = np.clip((img[:, 0] / w * gx).astype(int), 0, gx - 1)
    cy = np.clip((img[:, 1] / h * gy).astype(int), 0, gy - 1)
    m = np.zeros(gx * gy, bool)
    m[cy * gx + cx] = True
    return m


def judge(recs, board, size):
    """Each frame looked at gets why it cannot be used ("" when it can): none (no board), few (too few corners, or
    all on one line), moving, blur; and its quality (0-1) when it can."""
    lim = limits(size)
    nx, ny = board.getChessboardSize()
    total = (nx - 1) * (ny - 1)
    need = max(cc.MIN_CORNERS, int(math.ceil(FRAME_MIN_SHARE * total)))
    for r in recs:
        b = r.get("blur", math.nan)
        if not r["n"]:
            r["why"] = "none"
        elif r["n"] < need or r.get("collinear"):
            r["why"] = "few"
        elif not r["motion"] <= lim["motion_px_s"]:
            r["why"] = "moving"
        elif math.isfinite(b) and b > lim["blur_px"]:
            r["why"] = "blur"
        else:
            r["why"] = ""
            sharp = 1.0 / (1.0 + (b / lim["blur_px"]) ** 2) if math.isfinite(b) else 0.8
            r["quality"] = math.sqrt(r["n"] / total) * sharp * (1.0 - 0.5 * r["motion"] / lim["motion_px_s"])
    return need


def choose(recs, board, size, target):
    """The frames to calibrate on, in the order chosen (indexes into recs), and the smallest distance between two of
    them (steps). judge() first."""
    cand = [k for k, r in enumerate(recs) if r.get("why") == ""]
    if not cand:
        return [], math.inf
    K = nominal_K(size)
    for k in cand:
        recs[k]["pose"] = describe(recs[k], board, K, size)
    X, roll = _features([recs[k]["pose"] for k in cand])
    q = np.array([recs[k]["quality"] for k in cand])
    cells = np.array([_cells(recs[k]["img"], size) for k in cand])
    dmin = np.full(len(cand), np.inf)
    covered = np.zeros(cells.shape[1], bool)
    picked = []
    while len(picked) < target:
        gain = q * (np.minimum(dmin, NOVELTY_CAP) + COVER_WEIGHT * (cells & ~covered).sum(1))
        gain[dmin < DUPLICATE] = -np.inf
        if picked:
            gain[picked] = -np.inf
        j = int(np.argmax(gain))
        if not np.isfinite(gain[j]):
            break
        picked.append(j)
        dmin = np.minimum(dmin, _distance(X, roll, j))
        covered |= cells[j]
    closest = math.inf
    for a in range(len(picked)):
        d = _distance(X, roll, picked[a])[picked]
        d[a] = np.inf
        closest = min(closest, float(d.min()) if len(d) > 1 else math.inf)
    return [cand[j] for j in picked], closest


# ── writing what was chosen ──────────────────────────────────────────────────
def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def frame_name(video, i):
    return f"{Path(video).stem}_f{i:05d}.png"


def extract(video, picks, out_dir):
    """The frames picked ({index: crc}) written as grey PNG into out_dir; {index: (path, grey)}. Raises
    film_frames_differ when a frame read again is not the one first read."""
    out_dir = Path(out_dir)
    want = dict(picks)
    got = {}
    last = max(want) if want else -1
    for i, _, gray in frames(video):
        if i > last:
            break
        if i not in want:
            continue
        if zlib.crc32(gray.tobytes()) != want[i]:
            raise cc.CalibrationError("film_frames_differ", f"frame {i} decoded differently the second time",
                                      frame=i)
        ok, png = cv2.imencode(".png", gray)
        if not ok:
            raise OSError(f"frame {i} could not be encoded")
        p = out_dir / frame_name(video, i)
        _write_bytes_atomic(p, png.tobytes())
        got[i] = (p, gray)
    missing = sorted(set(want) - set(got))
    if missing:
        raise cc.CalibrationError("film_frames_differ", f"frames {missing[:5]} not found the second time",
                                  frame=missing[0])
    return got


def contact_sheet(items, path):
    """chosen.jpg: the frames chosen, small, in the order of the film, each with its time."""
    if not items:
        return
    thumbs = []
    for t, gray in items:
        h, w = gray.shape
        s = SHEET_THUMB / w
        im = cv2.cvtColor(cv2.resize(gray, (SHEET_THUMB, max(1, round(h * s))), interpolation=cv2.INTER_AREA),
                          cv2.COLOR_GRAY2BGR)
        label = f"{t:.1f} s"
        cv2.rectangle(im, (0, 0), (8 + 11 * len(label), 22), (32, 32, 32), -1)
        cv2.putText(im, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        thumbs.append(im)
    cols = min(6, len(thumbs))
    th = max(t.shape[0] for t in thumbs)
    rows = [thumbs[k:k + cols] for k in range(0, len(thumbs), cols)]
    gap = 4
    sheet = np.full((len(rows) * (th + gap) + gap, cols * (SHEET_THUMB + gap) + gap, 3), 24, np.uint8)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            y, x = gap + r * (th + gap), gap + c * (SHEET_THUMB + gap)
            sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
    ok, jpg = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        _write_bytes_atomic(Path(path), jpg.tobytes())


def _r(v, n=3):
    return round(float(v), n) if v is not None and math.isfinite(float(v)) else None


def run_choose(video, out_dir, board, target, mode="video", film_dir=None, log: Log | None = None,
               progress: Progress | None = None, need=None, stops=False):
    """The frames of a film worth calibrating on, written into out_dir (PNG), with frames.json and chosen.jpg in
    film_dir (out_dir/film by default). Returns the summary written to frames.json, with "checks": what
    calibrate_charuco.check_photo would say of each frame written ({name: check}), for the app. Raises
    film_unreadable, film_no_board (no frame shows the board), film_few (fewer than `need` frames can be used:
    film again, holding still at each pose), film_frames_differ."""
    say = _say(log)
    video, out_dir = Path(video), Path(out_dir)
    film_dir = Path(film_dir) if film_dir else out_dir / FILM_DIR
    recs, info = scan(video, board, log, progress)
    size = tuple(info["size"])
    min_corners = judge(recs, board, size)
    slower = []
    if stops:       # the bench: one still frame per stop of the arm (calib_bench), the stops on the move said
        import calib_bench
        K = nominal_K(size)
        picked, slower = calib_bench.choose_stops(recs, limits(size)["motion_px_s"],
                                                  lambda r: describe(r, board, K, size))
        closest = math.inf
    else:
        picked, closest = choose(recs, board, size, target)
    counts = {"looked": len(recs), "board": sum(1 for r in recs if r["n"]),
              **{w: sum(1 for r in recs if r.get("why") == w) for w in ("few", "moving", "blur")},
              "usable": sum(1 for r in recs if r.get("why") == ""), "chosen": len(picked)}
    say(f"frames looked at {counts['looked']}, with the board {counts['board']}, too few corners {counts['few']}, "
        f"moving {counts['moving']}, blurred {counts['blur']}, usable {counts['usable']}, chosen {len(picked)}")
    refused = None
    if not counts["board"]:
        refused = cc.CalibrationError("film_no_board", f"{video.name}: the board is in no frame", file=video.name)
    elif need is not None and len(picked) < need:
        refused = cc.CalibrationError("film_few", f"{len(picked)} frames can be used, at least {need} needed",
                                      n=len(picked), need=need)
    got = {} if refused else extract(video, {recs[k]["i"]: recs[k]["crc"] for k in picked}, out_dir)
    nx, ny = board.getChessboardSize()
    of = (nx - 1) * (ny - 1)
    w, h = size
    checks, chosen = {}, []
    for order, k in enumerate(picked):
        r = recs[k]
        name = got[r["i"]][0].name if r["i"] in got else frame_name(video, r["i"])
        pts = r["img"]
        sel = np.linspace(0, len(pts) - 1, min(len(pts), 120)).astype(int)
        b = r.get("blur", math.nan)
        checks[name] = {"ok": True, "why": "", "corners": r["n"], "of": of, "blur_px": _r(b, 2), "size": [w, h],
                        "points": [[round(float(x / w), 4), round(float(y / h), 4)] for x, y in pts[sel]],
                        "collinear": False, "film_s": _r(r["t"]), "motion_px_s": _r(r["motion"], 2)}
        pose = r["pose"]
        chosen.append({"name": name, "frame": r["i"], "t_s": _r(r["t"]), "order": order, "corners": r["n"],
                       "motion_px_s": _r(r["motion"], 2), "blur_px": _r(b, 2), "tilt_deg": _r(pose["tilt"], 1),
                       "dist_mm": _r(pose["dist"], 1), "roll_deg": _r(pose["roll"], 1),
                       "at": [_r(pose["at"][0]), _r(pose["at"][1])]})
    chosen.sort(key=lambda c: c["frame"])
    lim = limits(size)
    summary = {"video": video.name, "mode": mode, "date": date.today().isoformat(), "size": [w, h],
               "fps": info["fps"], "frames": info["frames"], "duration_s": info["duration_s"], "codec": info["codec"],
               "every": info["every"], "target": target, "board": cc.board_spec(board),
               "limits": {"min_corners": min_corners, "motion_px_s": round(lim["motion_px_s"], 2),
                          "blur_px": round(lim["blur_px"], 2), "duplicate_steps": DUPLICATE},
               "counts": counts, "closest_steps": _r(closest, 2), "refused": refused.code if refused else "",
               "slower": slower,
               "chosen": chosen,
               "looked": [[r["i"], _r(r["t"]), r["n"], _r(r.get("motion", math.nan), 1), _r(r.get("blur", math.nan), 2),
                           r.get("why", "")] for r in recs]}
    # what was looked at is kept even when the film is refused: frames.json says why
    cc.write_json_atomic(film_dir / FRAMES_FILE, summary)
    if refused:
        raise refused
    contact_sheet([(c["t_s"] or 0.0, got[c["frame"]][1]) for c in chosen], film_dir / SHEET_FILE)
    say(f"chosen frames: {out_dir} ({len(chosen)}), {film_dir / FRAMES_FILE}, {film_dir / SHEET_FILE}")
    summary["checks"] = checks
    summary["_views"] = {frame_name(video, recs[k]["i"]): (recs[k]["obj"], recs[k]["img"]) for k in picked}
    summary["_holdout"] = [(recs[k]["obj"], recs[k]["img"]) for k in _holdout(recs, picked)]
    return summary


def _holdout(recs, picked):
    """Frames that could be used but were not chosen, spread over the film: HOLDOUT at most."""
    taken = set(picked)
    rest = [k for k, r in enumerate(recs) if r.get("why") == "" and k not in taken]
    if len(rest) <= HOLDOUT:
        return rest
    return [rest[int(j)] for j in np.linspace(0, len(rest) - 1, HOLDOUT)]


def holdout_error(views, K, dist):
    """Median and largest RMS (px) of frames not calibrated on, each at the pose PnP finds under the model."""
    e = [cc.view_rms(v, np.asarray(K), np.asarray(dist))[0] for v in views]
    return (float(np.median(e)), float(np.max(e))) if e else (None, None)


def film_of(folder):
    """frames.json of a folder of frames chosen from a film, or None."""
    p = Path(folder) / FILM_DIR / FRAMES_FILE
    if not p.is_file():
        return None
    with contextlib.suppress(OSError, ValueError):
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def narrower_than(cal, ref):
    """How much narrower (relative, > 0 narrower) the field of `cal` is than `ref`'s would be at cal's size, both
    across the long side: the focal length of cal against ref's scaled to cal's long side. The camera app's video and
    the live stream are read from the whole width of the sensor, so a film whose focal length is several percent
    longer than the photos' scaled one was cropped: electronic stabilisation (or a zoom) was on."""
    K, Kr = np.array(cal["K"], float), np.array(ref["K"], float)
    s = max(cal["image_size"]) / max(ref["image_size"])
    f, fr = (K[0, 0] + K[1, 1]) / 2, (Kr[0, 0] + Kr[1, 1]) / 2 * s
    return float(f / fr - 1)


def side_by_side(cal, ref):
    """Two calibrations of different modes in the same units: focal length scaled to cal's long side, field of view,
    the optical centre's offset from the middle (in cal's px), distortion coefficients, and how far the lens bends
    at the corners of cal's frame under each model (ref scaled and cropped to cal's frame, centred)."""
    out = {}
    for tag, c in (("cal", cal), ("ref", ref)):
        p = cc.plain(c)
        out[tag] = {"size": c["image_size"], "rms_px": c.get("rms_px"), "n": c.get("n_views"),
                    "f_px": (p["fx"] + p["fy"]) / 2, "fov_x": p["fov_x"], "fov_y": p["fov_y"],
                    "cx_off": p["cx_off"], "cy_off": p["cy_off"], "bend_px": p["bend_px"],
                    "dist": [float(v) for v in c["dist"]]}
    s = max(cal["image_size"]) / max(ref["image_size"])
    out["scale"] = s
    out["f_ref_scaled"] = out["ref"]["f_px"] * s
    out["narrower"] = narrower_than(cal, ref)
    out["centre_ref_scaled"] = [out["ref"]["cx_off"] * s, out["ref"]["cy_off"] * s]
    # the lens of ref seen through cal's frame: same normalised distortion, so the bend at cal's corners
    w, h = cal["image_size"]
    Kr = np.array(ref["K"], float)
    Kc = np.array([[Kr[0, 0] * s, 0, (w - 1) / 2 + out["centre_ref_scaled"][0]],
                   [0, Kr[1, 1] * s, (h - 1) / 2 + out["centre_ref_scaled"][1]], [0, 0, 1]])
    out["bend_ref_px"] = cc.plain({"K": Kc.tolist(), "dist": ref["dist"], "image_size": [w, h], "std": {}})["bend_px"]
    out["model_max_px"], out["model_rms_px"] = cc.model_difference(Kc, ref["dist"], cal["K"], cal["dist"], (w, h))
    return out


def print_side_by_side(d, log: Log | None = None):
    say = _say(log)
    c, r = d["cal"], d["ref"]
    say(f"{'':22s}{'film':>14s}{'reference':>14s}")
    say(f"{'size':22s}{c['size'][0]:>8d} x {c['size'][1]:<5d}{r['size'][0]:>8d} x {r['size'][1]:<5d}")
    say(f"{'RMS px (frames)':22s}{c['rms_px']:>10.3f} ({c['n']}){r['rms_px']:>10.3f} ({r['n']})")
    say(f"{'focal px at film size':22s}{c['f_px']:>14.1f}{d['f_ref_scaled']:>14.1f}   film narrower by "
        f"{100 * d['narrower']:+.2f} %")
    say(f"{'field of view deg':22s}{c['fov_x']:>8.2f} x {c['fov_y']:<5.2f}{r['fov_x']:>8.2f} x {r['fov_y']:<5.2f}")
    say(f"{'centre off middle px':22s}{c['cx_off']:>+7.1f},{c['cy_off']:>+6.1f}"
        f"{d['centre_ref_scaled'][0]:>+7.1f},{d['centre_ref_scaled'][1]:>+6.1f}  (reference scaled)")
    say(f"{'bend at corners px':22s}{c['bend_px']:>14.1f}{d['bend_ref_px']:>14.1f}  (reference in the film's frame)")
    for k, name in enumerate(("k1", "k2", "p1", "p2", "k3")):
        say(f"{name:22s}{c['dist'][k]:>14.5f}{r['dist'][k]:>14.5f}")
    say(f"model difference over the film's frame: max {d['model_max_px']:.2f} px, RMS {d['model_rms_px']:.2f} px")


# ── calibrate a film ─────────────────────────────────────────────────────────
def run_calibrate(video, work, board, target, out=None, truth=None, mode="video", log: Log | None = None,
                  need=cc.MIN_VIEWS):
    """Choose the frames of a film into `work`, calibrate on them (calibrate_charuco.calibrate), write the result to
    `out` if given, compare with the truth of a synthetic film if given. Returns the calibration with "film"."""
    say = _say(log)
    summary = run_choose(video, work, board, target, mode, log=log, need=need)
    views, size = summary.pop("_views"), tuple(summary["size"])
    hold = summary.pop("_holdout")
    summary.pop("checks")
    cal = cc.calibrate(views, size, square=board.getSquareLength())
    med, worst = holdout_error(hold, cal["K"], cal["dist"])
    cal = {"images": str(Path(work)), "date": date.today().isoformat(), "board": cc.board_spec(board), "mode": mode,
           "film": {k: summary[k] for k in ("video", "size", "fps", "frames", "duration_s", "codec", "counts",
                                            "limits", "closest_steps")},
           "holdout_px": {"median": _r(med), "max": _r(worst), "n": len(hold)}, **cal}
    cc.summary(cal, [], log)
    if med is not None:
        say(f"frames not used ({len(hold)}), reprojected by the model: median {med:.3f} px, at most {worst:.3f} px")
    if truth:
        cal["error"] = cc.compare(cal, json.loads(Path(truth).read_text(encoding="utf-8")))
        cc.print_compare(cal["error"], log)
    if out:
        cc.write_json_atomic(out, cal)
        say(f"intrinsics: {out}")
    return cal


# ── a synthetic film ─────────────────────────────────────────────────────────
def _slerp(Ra, Rb, s):
    return Ra @ cv2.Rodrigues(s * cv2.Rodrigues(Ra.T @ Rb)[0])[0]


def _ease(s):
    return s * s * (3 - 2 * s)


def synth_film(path, spec=None, kind="sheet", size=(1280, 720), fps=30.0, plan="quick", pause_s=0.8, move_s=1.2,
               readout_s=0.02, exposure_s=1 / 60, tremor_px_s=4.0, seed=0, lay=None, poses=None):
    """A film of the board as the phone takes it moving through the poses of a plan (calib_guide.PLANS), holding
    still pause_s at each (with a hand's tremor of about tremor_px_s) and moving move_s between them; the rows read
    one after the other over readout_s (rolling shutter), blurred by the motion during exposure_s, sensor noise, H.264
    in MP4. The camera is calibrate_charuco's synthetic one scaled to `size` (16:9: the middle rows of its 4:3
    field). Truth beside it (<stem>_truth.json: K, dist, size, fps, and per frame the speed of the board in the
    frame and whether it is held still). pause_s 0 with poses: one long move. Returns the truth."""
    import av
    import calib_guide
    rng = np.random.default_rng(seed)
    spec = spec or {**cc.board_spec(cc.make_board()), "kind": "sheet"}
    w, h = size
    s = w / cc.SYNTH_SIZE[0]
    K = cc.SYNTH_K.copy()
    K[:2] *= s
    K[0, 2] = (cc.SYNTH_K[0, 2] + 0.5) * s - 0.5
    K[1, 2] = (cc.SYNTH_K[1, 2] + 0.5) * s - 0.5 - (cc.SYNTH_SIZE[1] * s - h) / 2
    dist = cc.SYNTH_DIST
    bmm = calib_guide.board_mm(spec)
    plan_poses = poses if poses is not None else calib_guide.PLANS[plan]
    keys = []
    for pose in plan_poses:
        j = calib_guide.jittered(pose, rng)
        d = calib_guide.distances(bmm, "stream")[pose["dist"]] * j["dist_mul"]
        T = calib_guide.camera_pose(calib_guide.free_pose(j["yaw"], j["pitch"], j["at"], d, j["roll"]), bmm,
                                    "stream", K, size)
        R, t = T[:3, :3], T[:3, 3]
        keys.append((R, -R.T @ t))              # orientation, and the camera's centre in the board's frame
    seg = pause_s + move_s
    total = len(keys) * pause_s + (len(keys) - 1) * move_s
    f_tr = rng.uniform(6.0, 10.0, 3)
    ph_tr = rng.uniform(0, 2 * np.pi, (3, 3))
    amp = tremor_px_s / (2 * np.pi * f_tr.mean() * K[0, 0])        # rad

    def pose_at(t):
        k = int(min(len(keys) - 1, t // seg)) if seg > 0 else 0
        local = t - k * seg
        if local <= pause_s or k == len(keys) - 1:
            R, C = keys[k]
            still = True
        else:
            u = _ease(min(1.0, (local - pause_s) / move_s))
            Ra, Ca = keys[k]
            Rb, Cb = keys[k + 1]
            R, C, still = _slerp(Ra, Rb, u), Ca + u * (Cb - Ca), False
        wob = amp * np.array([np.sum(np.sin(2 * np.pi * f_tr * t + ph_tr[a])) / 3 for a in range(3)])
        R = cv2.Rodrigues(wob)[0] @ R
        return R, -R @ C, still

    ppm = cc.texture_ppm(cc.make_board(spec["square_mm"], spec["squares"], spec.get("marker_mm")))
    page, origin, ppm = calib_guide.texture(kind, spec, lay, ppm)
    page = cv2.GaussianBlur(page, (0, 0), 0.6)
    ones = np.full_like(page, 255)
    rays = cc.pixel_rays(K, dist, size)
    black, white = rng.uniform(30, 50), rng.uniform(190, 225)
    bg = cv2.resize(rng.uniform(60, 140, (12, 16)).astype(np.float32), size, interpolation=cv2.INTER_CUBIC)
    gx, gy = rng.uniform(-0.2, 0.2, 2)
    uu, vv = np.meshgrid(np.linspace(-0.5, 0.5, w, dtype=np.float32), np.linspace(-0.5, 0.5, h, dtype=np.float32))
    light = 1 + gx * uu + gy * vv
    bands = np.linspace(0, h, 13).astype(int)
    corners = cc.make_board(spec["square_mm"], spec["squares"], spec.get("marker_mm")).getChessboardCorners()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.mp4")
    out = av.open(str(tmp), "w")
    st = out.add_stream("libx264", rate=int(round(fps)))
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": "16", "preset": "veryfast"}
    n = int(round(total * fps)) + 1
    per_frame = []
    try:
        for i in range(n):
            t0 = i / fps
            mx = np.empty((h, w), np.float32)
            my = np.empty((h, w), np.float32)
            for b0, b1 in zip(bands[:-1], bands[1:]):
                R, tt, _ = pose_at(t0 + (b0 + b1) / 2 / h * readout_s)
                mx[b0:b1], my[b0:b1] = cc.board_maps(rays[b0:b1], R, tt, origin, ppm)
            paper = cv2.remap(page, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            inside = cv2.remap(ones, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            img = black + (white - black) * paper.astype(np.float32) / 255
            a = inside.astype(np.float32) / 255
            img = (a * img + (1 - a) * bg) * light
            # the board's speed in the frame (px/s, 90th percentile of the corners) over the exposure
            Ra, ta, still = pose_at(t0)
            Rb, tb, _ = pose_at(t0 + 1 / fps)
            pa = cv2.projectPoints(corners, cv2.Rodrigues(Ra)[0], ta, K, dist)[0].reshape(-1, 2)
            pb = cv2.projectPoints(corners, cv2.Rodrigues(Rb)[0], tb, K, dist)[0].reshape(-1, 2)
            speed = float(np.percentile(np.linalg.norm(pb - pa, axis=1), 90)) * fps
            sigma = min(3.0, math.hypot(0.5, speed * exposure_s / 2.5))     # past 3 px it is blurred anyway
            img = cv2.GaussianBlur(img, (0, 0), sigma)
            img += rng.standard_normal(img.shape, dtype=np.float32) * 2.5
            gray = np.clip(np.rint(img), 0, 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(gray, format="gray").reformat(format="yuv420p")
            for pkt in st.encode(frame):
                out.mux(pkt)
            per_frame.append([round(speed, 2), bool(still)])
        for pkt in st.encode():
            out.mux(pkt)
    finally:
        out.close()
    os.replace(tmp, path)
    truth = {"image_size": [w, h], "K": K.tolist(), "dist": list(map(float, dist)), "fps": fps, "seed": seed,
             "plan": plan, "pause_s": pause_s, "move_s": move_s, "readout_s": readout_s, "frames": per_frame}
    cc.write_json_atomic(path.with_name(path.stem + "_truth.json"), truth)
    return truth


# ── test ─────────────────────────────────────────────────────────────────────
def run_test(work):
    """Synthetic films: the frames chosen are still and sharp, of different poses, and calibrate within the limits of
    calibrate_charuco; the same film calibrated on one frame a second whatever it shows does worse; a film always
    moving, one without the board and one that cannot be read are said as such."""
    fails = []

    def expect(good, what):
        print(("ok    " if good else "FAIL  ") + what, flush=True)
        if not good:
            fails.append(what)

    try:
        work = cc.test_folder(work)
    except ValueError as e:
        print(f"FAIL  {e}")
        return False
    board = cc.make_board()
    t0 = time.monotonic()
    film = work / "film.mp4"
    truth = synth_film(film, plan="quick", pause_s=0.7, move_s=0.9, seed=4)
    lines = []
    out = work / "frames"
    t1 = time.monotonic()
    cal = run_calibrate(film, out, board, 15, out=work / "intrinsics_film.json", truth=film.with_name("film_truth.json"),
                        log=lines.append)
    t2 = time.monotonic()
    for x in lines:
        print("      " + x)
    print(f"      film rendered in {t1 - t0:.0f} s, frames chosen and calibrated in {t2 - t1:.0f} s")
    summary = json.loads((out / FILM_DIR / FRAMES_FILE).read_text(encoding="utf-8"))
    chosen = summary["chosen"]
    speeds = [truth["frames"][c["frame"]][0] for c in chosen]
    still = [truth["frames"][c["frame"]][1] for c in chosen]
    lim = limits(truth["image_size"])
    expect(len(chosen) >= 12 and max(speeds) < 1.5 * lim["motion_px_s"] and sum(still) >= len(chosen) - 1,
           f"{len(chosen)} frames chosen from {summary['frames']}, all still: board speed at most {max(speeds):.1f} "
           f"px/s (limit {lim['motion_px_s']:.1f}), {sum(still)} in a pause")
    seg = truth["pause_s"] + truth["move_s"]
    pauses = {int(c["frame"] / truth["fps"] // seg) for c in chosen}
    expect(len(pauses) >= 12 and summary["closest_steps"] >= DUPLICATE,
           f"different poses: {len(pauses)} of the 15 pauses, the two closest {summary['closest_steps']:.2f} steps "
           f"apart (near duplicates under {DUPLICATE})")
    e = cal["error"]
    # every frame written is the frame read, and the folder calibrates as photos do (the app's way)
    pngs = sorted(out.glob("*.png"))
    again = cc.run_calibrate(out, None, None, board.getSquareLength(), log=lambda s: None)
    expect(len(pngs) == len(chosen) and again["n_views"] == cal["n_views"]
           and abs(again["rms_px"] - cal["rms_px"]) < 1e-3 and (out / FILM_DIR / SHEET_FILE).is_file()
           and not list(out.glob(".*.tmp")) and "NaN" not in (out / FILM_DIR / FRAMES_FILE).read_text(encoding="utf-8"),
           f"{len(pngs)} frames as lossless PNG, the folder calibrating the same as photos (RMS {again['rms_px']:.4f}), "
           f"frames.json strict JSON, chosen.jpg beside it")
    # against the best one could do (the middle frame of every pause, known from the truth) and the naive way (one
    # frame a second, whatever it shows)
    fps = truth["fps"]
    best = {int(round((k * seg + truth["pause_s"] / 2) * fps)) for k in range(15)}
    picks = {"best": best, "naive": {i for i in range(len(truth["frames"])) if i % int(fps) == 0}}
    for name in picks:
        (work / name).mkdir()
    for i, _, gray in frames(film):
        for name, want in picks.items():
            if i in want:
                cv2.imwrite(str(work / name / f"{name}{i:05d}.png"), gray)
    ref = {name: cc.run_calibrate(work / name, None, film.with_name("film_truth.json"), board.getSquareLength(),
                                  log=lambda s: None) for name in picks}
    for name, c in ref.items():
        x = c["error"]
        print(f"      {name} frames ({c['n_views']}): RMS {c['rms_px']:.3f} px, fx {100 * x['fx_rel']:+.3f} %, centre "
              f"{x['cx_px']:+.2f}, {x['cy_px']:+.2f} px, model {x['model_max_px']:.2f} px at most, RMS "
              f"{x['model_rms_px']:.2f}")
    be, ne = ref["best"]["error"], ref["naive"]["error"]
    good = (cal["rms_px"] < cc.MAX_RMS and abs(e["fx_rel"]) < 0.002 and abs(e["fy_rel"]) < 0.002
            and abs(e["cx_px"]) < 2 and abs(e["cy_px"]) < 2
            and e["model_rms_px"] <= 1.25 * be["model_rms_px"] + 0.15 and e["model_max_px"] <= 1.25 * be["model_max_px"] + 0.3)
    expect(good, f"calibrated on them: RMS {cal['rms_px']:.3f} px, fx {100 * e['fx_rel']:+.3f} %, fy "
                 f"{100 * e['fy_rel']:+.3f} %, centre {e['cx_px']:+.2f}, {e['cy_px']:+.2f} px, model {e['model_max_px']:.2f} "
                 f"px at most (RMS {e['model_rms_px']:.2f}), as good as the middle of every pause "
                 f"({be['model_max_px']:.2f}, RMS {be['model_rms_px']:.2f}); frames not used "
                 f"{cal['holdout_px']['median']} px")
    expect(ref["naive"]["rms_px"] > cal["rms_px"] and ne["model_rms_px"] > e["model_rms_px"],
           f"better than one frame a second: RMS {cal['rms_px']:.3f} against {ref['naive']['rms_px']:.3f} px, model "
           f"RMS {e['model_rms_px']:.2f} against {ne['model_rms_px']:.2f} px")
    # a film of the whole width of the sensor sees the photos' field at its size; one cropped by stabilisation does not
    photos = {"K": [[3000.0, 0, 2047.5], [0, 3000.0, 1535.5], [0, 0, 1]], "image_size": [4096, 3072]}
    whole = {"K": [[1406.25, 0, 959.5], [0, 1406.25, 539.5], [0, 0, 1]], "image_size": [1920, 1080]}
    cut = {"K": [[1406.25 * 1.15, 0, 959.5], [0, 1406.25 * 1.15, 539.5], [0, 0, 1]], "image_size": [1920, 1080]}
    expect(abs(narrower_than(whole, photos)) < 1e-9 and abs(narrower_than(cut, photos) - 0.15) < 1e-9,
           "a 16:9 film of the whole sensor width sees the photos' field; one cropped by 13 % is said 15 % narrower")
    # always moving: no frame is still
    import calib_guide
    sweep = [calib_guide.P("left", 40, dist="mid"), calib_guide.P("right", 40, dist="mid")]
    synth_film(work / "moving.mp4", plan="quick", poses=sweep, pause_s=0.0, move_s=4.0, seed=5)
    try:
        run_choose(work / "moving.mp4", work / "moving_frames", board, 15, need=cc.MIN_VIEWS, log=lambda s: None)
        moved = False
    except cc.CalibrationError as x:
        moved = x.code == "film_few"
    expect(moved and not list((work / "moving_frames").glob("*.png")) if (work / "moving_frames").exists() else moved,
           "a film always moving: film_few, no frame written")
    blank = work / "blank.mp4"
    wr = cv2.VideoWriter(str(blank), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(20):
        wr.write(np.full((240, 320, 3), 90, np.uint8))
    wr.release()
    codes = []
    for p in (blank, work / "cut.mp4", work / "missing.mp4"):
        if p.name == "cut.mp4":
            p.write_bytes(film.read_bytes()[:2000])
        try:
            run_choose(p, work / f"{p.stem}_frames", board, 15, need=cc.MIN_VIEWS, log=lambda s: None)
            codes.append("")
        except cc.CalibrationError as x:
            codes.append(x.code)
    expect(codes == ["film_no_board", "film_unreadable", "film_unreadable"],
           f"a film without the board, one cut short, one missing: {codes}")
    print("test: " + ("PASS" if not fails else "FAIL"))
    return not fails


def _board_from(args, *folders):
    if args.square is not None or args.squares is not None:
        return cc.make_board(args.square or cc.BOARD["square"], args.squares, args.marker)
    for f in folders:
        got = cc.folder_board(f) if f else None
        if got:
            square, squares, marker = got
            return cc.make_board(square, squares, args.marker or marker)
    return cc.make_board(cc.BOARD["square"], None, args.marker)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("choose")
    a.add_argument("video", type=Path)
    a.add_argument("out", type=Path)
    a = sub.add_parser("calibrate")
    a.add_argument("video", type=Path)
    a.add_argument("--out", type=Path)
    a.add_argument("--truth", type=Path)
    a.add_argument("--work", type=Path)
    a.add_argument("--mode", default="video")
    for a in (sub.choices["choose"], sub.choices["calibrate"]):
        a.add_argument("--target", type=int, default=40)
        a.add_argument("--square", type=float)
        a.add_argument("--squares", type=lambda v: tuple(int(n) for n in v.lower().split("x")))
        a.add_argument("--marker", type=float)
    a = sub.add_parser("synth")
    a.add_argument("out", type=Path)
    a.add_argument("--plan", default="quick")
    a.add_argument("--size", type=lambda v: tuple(int(n) for n in v.lower().split("x")), default=(1280, 720))
    a.add_argument("--fps", type=float, default=30.0)
    a.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("compare")
    a.add_argument("cal", type=Path)
    a.add_argument("ref", type=Path)
    a = sub.add_parser("test")
    a.add_argument("--work", type=Path, default=WORK / "selftest")
    args = ap.parse_args()
    try:
        if args.cmd == "choose":
            board = _board_from(args, args.out, args.video.parent)
            s = run_choose(args.video, args.out, board, args.target)
            print(json.dumps({k: s[k] for k in ("counts", "closest_steps", "limits")}))
        elif args.cmd == "calibrate":
            work = args.work or WORK / args.video.stem
            board = _board_from(args, work, args.video.parent)
            run_calibrate(args.video, work, board, args.target, args.out, args.truth, args.mode)
        elif args.cmd == "synth":
            synth_film(args.out, plan=args.plan, size=args.size, fps=args.fps, seed=args.seed)
            print(f"synth: {args.out}")
        elif args.cmd == "compare":
            print_side_by_side(side_by_side(json.loads(args.cal.read_text(encoding="utf-8")),
                                            json.loads(args.ref.read_text(encoding="utf-8"))))
        elif args.cmd == "test":
            sys.exit(0 if run_test(args.work) else 1)
    except cc.CalibrationError as e:
        sys.exit(f"calib_video: {e.code}: {e}")


if __name__ == "__main__":
    main()
