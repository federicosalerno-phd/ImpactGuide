"""Scans of the bench: every export of the scanner bound to its trial and step and measured at once (the six spheres,
the true registration, the marker's shift and the error at the targets), the two online values of the trial log;
the calibration of the spheres' radii; the repeatability of the scans; the scans of a day processed into
gt_poses.csv; and, in simulation, a scanner that exports synthetic scans of the bench as it stands.

Binding (bind). A trial has up to four scan steps (trials.trial_steps): the reference scans before the impact
(ref_scan_azNN, only when a reference is due) and the scans after it (scan_azNN), one per arm position of
trials.Config.scan_az. A new export goes to a step of its side of the impact (before it while time_impact is empty,
after it once set) whose file is not filed yet: the one at the arm position the scan was taken from when the scan
tells it (arm_of), else the step the trial's page shows when it has nothing yet, else the first one with nothing
yet. A step whose file is filed never takes another; a scan that no step waits for is not used.

Arm position (arm_of). The scanner's place seen from the head's spheres: the origin of the scan's frame carried into
the mandible frame by the head rig's pose, then into the bench's axes, as azimuth, elevation and distance from the
centroid of the six spheres; the nearest of the scanner poses of the bench scene (ground_truth.SCANNER_POSES,
written by the twin's export) within ARM_TOL_DEG and ARM_RANGE_MM, and clearly nearer than the next one. A scan
whose frame is not the scanner's own tells nothing (None).

Measures (Desk). Each view (one export) is measured alone as it comes (ground_truth.measure with the rig
calibration: spheres found, rigs identified, the arm position, a registration of its own); then the views of a set
together (all views aligned on the six centres): the trial's post-impact set and its set of before (its reference
scans, or the post-impact scans of the trial before, as pre_scan says). Results are kept per trial in
STORE/<phase>/<trial>.json by the content of the scans (SHA-256) and of the rig calibration, so nothing is measured
twice and a new calibration measures again. From the registrations T_M_mandible of before and after: dT, the
rotation, the shift of the marker (the board's centre of the bench scene, else the centroid of the marker's
spheres), the error at each target (ground_truth.load_targets) and the cap's shift since the last reference scans
(pre_scan followed back to the trial that had them; the cap's point from the bench scene, else the marker's
spheres), which decides when the holder goes back with the jig. With one post-impact view in, a provisional result.

Log (Desk.log_values). spheres_ok (every view of the post-impact set has its six spheres) and cap_shift_mm (two
decimals) go into the trial's row once the set is measured: into an empty field, or changed with a reason while the
field still holds the value written here before (the scans measured again after a calibration); a value typed by
hand is never changed.

Runs without an impact (Capture). calibration: at least ground_truth.MIN_CALIBRATION_SCANS scans of the bench,
turned between them, from which ground_truth.calibrate measures the radius of each sphere and the rigs' shapes, once;
repeat: REPEAT_N scans of the bench left alone, whose registrations scatter at the targets by the repeatability of
the scans (spread: under REPEAT_LIMIT_MM it holds). Each export is copied into RAW/ground_truth/<kind>/ under the
next free name (cal_001.ply...), never over a file, and measured; the run is kept in STORE/<kind>.json, the
repeatability's rows in STORE/repeatability.csv.

Processing (process). The trials of a day (or all): their scans measured with the calibration of now, what is
missing for each (trials.Run.missing), the log's values, and STORE/<phase>/gt_poses.csv written again from every
trial's result (one row per trial: setting, registrations before and after in both head frames, dT, rotation,
shifts, the error at each target, flags).

Simulation (SimScanner). A scanner that writes, in place of the EinScan's export, a synthetic scan of the two rigs
(ground_truth's model: the projector and its two cameras, noise, spikes, posts, board, face, flying points) seen
from the scanner pose of the bench scene at the step's arm position, with the marker where the trials left it: each
impact moves it once (A turns about its ball, more at a higher hole; B comes back within a tenth of a millimetre), a
reposition puts it back with the jig within a few tenths, a calibration's scans turn the bench between them.

    python scan_desk.py process [--phase campaign] [--day YYYY-MM-DD | --all]
    python scan_desk.py test [--work DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

import ground_truth as gt
import trials

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "03_data" / "processed"
STORE = "ground_truth"              # processed/STORE: <phase>/<trial>.json, <phase>/gt_poses.csv, <kind>.json
CAPTURES = {"calibration": "cal", "repeat": "rep", "reseat": "bre"}  # a run without an impact: its files' prefix
CAPTURE_DIRS = {"calibration": "calibration", "repeat": "repeatability", "reseat": "reseating"}   # in RAW/STORE
REPEAT_N = 20                       # scans of a repeatability run
RESEAT_N = 31                       # scans of a reseating run of holder B: the first seating and 30 lifts
REPEAT_LIMIT_MM = 0.05              # the registration's spread at the targets the scans must stay under
ARM_TOL_DEG = 6.0                   # a scan is from an arm position this close to its scanner pose
ARM_GAP_DEG = 4.0                   # and this much closer to it than to the next one
ARM_RANGE_MM = 60.0                 # at a distance this close to the pose's own
SIM_NOISE_MM = 0.05                 # the simulated scanner's depth noise (EinScan-SP: accuracy 0.05 mm)
SIM_JIG = (0.15, 0.10)              # a reposition with the jig: within this many degrees and mm
SIM_TURN_DEG = (25.0, 8.0)          # a calibration's scans: the bench turned up to this much about z, and tilted
POINTS_S = 10.0                     # the bench scene's points read again this often
SCAN_KEY = re.compile(r"^(ref_)?scan_az(\d+)$")
LOG_VALUES = ("spheres_ok", "cap_shift_mm")
# what a measure of a file that holds no spheres (or is no cloud) raises
UNMEASURABLE = (ValueError, IndexError, KeyError, TypeError, OSError, ZeroDivisionError, np.linalg.LinAlgError)
EDIT_REASON = "measured again from the scans"
POSES_NAME = "gt_poses.csv"
_SHA: dict = {}                     # (path, size, mtime) -> SHA-256 of the file
_LOCK = threading.Lock()            # one measure at a time on the store


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: dict) -> None:
    """Written whole beside its name, then put in place (a reader sees the old file or the new one)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    part.write_text(json.dumps(gt.jsonable(data), indent=1, allow_nan=False), encoding="utf-8")
    for i in range(40):
        try:
            os.replace(part, path)
            return
        except PermissionError:         # a reader holds it for a moment (Windows)
            if i == 39:
                part.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def file_sha(path: Path | str) -> str:
    """SHA-256 of a file, computed once for each size and time of it."""
    st = os.stat(path)
    key = (str(path), st.st_size, st.st_mtime_ns)
    s = _SHA.get(key)
    if s is None:
        s = trials.sha256_file(path)
        _SHA[key] = s
    return s


def stamp(path: Path | str) -> str:
    """The time a file was written (its export, for a scan still in the scanner's folder)."""
    return datetime.fromtimestamp(os.stat(path).st_mtime).astimezone().isoformat(timespec="seconds")


def age_s(iso: str | None, then: float | None = None) -> float | None:
    try:
        return round((then or time.time()) - datetime.fromisoformat(iso).timestamp(), 1)
    except (TypeError, ValueError):
        return None


def mat(x) -> np.ndarray | None:
    return None if x is None else np.asarray(x, np.float64)


# ── the steps of a trial ────────────────────────────────────────────────────
def slot_of(key: str, cfg: trials.Config) -> dict | None:
    """A scan step: {key, when (ref: before the impact, post: after it), az (camera azimuth), arm (its position,
    1-based, among the photos'), file (its name)}; None for any other step."""
    m = SCAN_KEY.match(key or "")
    if not m:
        return None
    az = int(m.group(2))
    photo = [int(a) for a in cfg.photo_az]
    return {"key": key, "when": "ref" if m.group(1) else "post", "az": az,
            "arm": photo.index(az) + 1 if az in photo else None}


def slots(run: trials.Run, trial: str, rows: list[dict] | None = None) -> list[dict]:
    """The scan steps of a logged trial, in their order, each with its file name (rows: the log, read once by the
    caller; read here when None)."""
    if rows is None:
        steps = run.steps(trial)
    else:
        i = next((i for i, r in enumerate(rows) if r["trial"] == trial), None)
        if i is None:
            return []
        steps = trials.trial_steps(run.config, rows[i], rows[i - 1] if i else None)
    out = []
    for st in steps:
        s = slot_of(st.key, run.config)
        if s:
            s["file"] = st.files[0]
            out.append(s)
    return out


def filed(run: trials.Run, trial: str, slot: dict) -> Path | None:
    found = run.find(trial, slot["file"])
    return found[0] if found else None


def bind(run: trials.Run, row: dict, staged=(), arm: int | None = None, prefer: str | None = None) -> str | None:
    """The step a new export of the open trial `row` goes to (its key), or None when no step waits for one.

    Args:
        staged: the keys of the steps with a file staged (not filed yet).
        arm: the arm position the scan was taken from (arm_of), None when the scan does not tell.
        prefer: the step the trial's page shows.
    """
    trial, after = row["trial"], bool(row.get("time_impact"))
    open_ = [s for s in slots(run, trial) if (s["when"] == "post") == after and filed(run, trial, s) is None]
    free = [s for s in open_ if s["key"] not in staged]
    if arm is not None:
        same = [s for s in open_ if s["arm"] == arm]
        if same:
            return next((s for s in same if s["key"] not in staged), same[0])["key"]
    if prefer and any(s["key"] == prefer for s in free):
        return prefer
    return free[0]["key"] if free else None


# ── where a scan was taken from ─────────────────────────────────────────────
def scanner_positions(path: Path = gt.SCANNER_POSES, photo_az=None) -> list[dict]:
    """The scanner at each arm position, as the bench scene has it: [{arm, camera_deg, azimuth_deg, elevation_deg,
    distance_mm}] (empty when the file is missing); arm is the camera azimuth's place among photo_az when given."""
    try:
        pos = json.loads(Path(path).read_text(encoding="utf-8"))["positions"]
        photo = [int(a) for a in photo_az] if photo_az else []
        out = []
        for p in pos:
            cam = float(p["camera_deg"])
            arm = next((i + 1 for i, a in enumerate(photo) if abs(a - cam) < 0.5), int(p.get("position", 0)))
            out.append({"arm": arm, "camera_deg": cam, "azimuth_deg": float(p["azimuth_deg"]),
                        "elevation_deg": float(p["elevation_deg"]), "distance_mm": float(p["distance_mm"])})
        return out
    except (OSError, ValueError, KeyError, TypeError):
        return []


def direction(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])


def arm_of(result: dict, positions: list[dict], R_W_mandible) -> dict:
    """The arm position a measured view was taken from: {arm (None when it does not tell), az_deg, el_deg, dist_mm,
    off_deg}. result: ground_truth.measure() of that view alone."""
    try:
        T = gt.inverse(mat(result["rigs"]["mandible"]["T_scan_S"]))         # mandible from scan
        C = np.array([np.asarray(s["centre"], np.float64) for s in result["spheres"].values()])
    except (KeyError, TypeError, ValueError):
        return {"arm": None}
    u = np.asarray(R_W_mandible, np.float64) @ (T[:3, 3] - gt.transform(T, C).mean(0))
    d = float(np.linalg.norm(u))
    if d <= 0 or not positions:
        return {"arm": None}
    u = u / d
    out = {"arm": None, "az_deg": round(math.degrees(math.atan2(u[0], -u[1])), 2),
           "el_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, u[2])))), 2), "dist_mm": round(d, 1)}
    near = sorted((math.degrees(math.acos(max(-1.0, min(1.0, float(u @ direction(p["azimuth_deg"],
                                                                                    p["elevation_deg"])))))), p)
                  for p in positions)
    off, best = near[0][0], near[0][1]
    out["off_deg"] = round(off, 2)
    gap = near[1][0] - off if len(near) > 1 else 180.0
    if off <= ARM_TOL_DEG and gap >= ARM_GAP_DEG and abs(d - best["distance_mm"]) <= ARM_RANGE_MM:
        out["arm"] = best["arm"]
    return out


# ── points of the bench, from its scene ─────────────────────────────────────
def scene_points(info: dict | None) -> dict:
    """The rotation of the mandible in the bench's axes and, in the mandible frame, the cap's strike point and the
    board's centre, from the twin's model of the bench scene (twin.Watcher.info); what it lacks is None (the rotation
    then the one of ground_truth's scene)."""
    out = {"R_W_mandible": gt.SCENE["R_W_mandible"], "cap": None, "board": None}
    try:
        T = mat(info["T_W_mandible"])
        if T.shape != (4, 4):
            return out
    except (KeyError, TypeError, ValueError):
        return out
    out["R_W_mandible"] = T[:3, :3]
    Ti = gt.inverse(T)
    for key, name in (("cap", "strike_point"), ("board", "board_centre")):
        with contextlib.suppress(KeyError, TypeError, ValueError):
            p = np.asarray(info[name], np.float64)
            if p.shape == (3,):
                out[key] = gt.transform(Ti, [p])[0]
    return out


def targets() -> list[dict]:
    """The targets of the error, in the mandible frame, with their names: [{id, name, mandible}]."""
    t = gt.load_targets()
    names = {}
    with contextlib.suppress(OSError, ValueError, KeyError):
        with open(gt.TARGETS_CSV, newline="", encoding="utf-8-sig") as f:
            names = {r["id"]: r.get("name") or r["id"] for r in csv.DictReader(f)}
    return [{"id": k, "name": names.get(k, k), "mandible": [float(x) for x in v]} for k, v in t.items()]


def marker_centroid(T_M_mandible, rigs=None) -> np.ndarray:
    """The centroid of the marker's spheres in the mandible frame, at the registration T_M_mandible."""
    tpl = np.asarray((rigs or gt.RIGS)["marker"], np.float64)
    return gt.transform(gt.inverse(mat(T_M_mandible)), tpl).mean(0)


def carried(dT, p) -> float:
    """How far the marker's motion of dT (dT^-1 in the mandible frame) carries a point of the marker that was at p."""
    p = np.asarray(p, np.float64)
    return float(np.linalg.norm(gt.transform(gt.inverse(mat(dT)), [p])[0] - p))


# ── the measures of the trials ──────────────────────────────────────────────
def view_summary(result: dict | None, path: Path, sha: str, cal: str, seconds: float, positions, R,
                 error: str = "") -> dict:
    """What is kept of one view's measure."""
    out = {"sha": sha, "file": Path(path).name, "size": os.stat(path).st_size, "exported": stamp(path),
           "measured": now_iso(), "seconds": round(seconds, 2), "cal": cal}
    if result is None:
        m = re.search(r"(\d+) spheres found", error)
        out.update(ok=False, n_spheres=int(m.group(1)) if m else 0, identified=False,
                   why="few_spheres" if m else "not_identified", arm={"arm": None})
        return out
    info = (result.get("views") or [{}])[0]
    out.update(ok=True, n_spheres=int(info.get("n_spheres", 6)), identified=True, why="",
               T_M_mandible=result["T_M_mandible"], T_M_B=result.get("T_M_B"),
               calibrated=bool(result.get("calibrated")), warnings=list(result.get("warnings") or []),
               arm=arm_of(result, positions, R),
               spheres={n: {"radius": s["radius"], "rms_mm": s["rms_mm"], "coverage": s["coverage"],
                            "n_points": s["n_points"], "centre": s["centre"]} for n, s in result["spheres"].items()})
    return out


def set_summary(result: dict | None, shas: list[str], cal: str, seconds: float, error: str = "") -> dict:
    out = {"shas": list(shas), "cal": cal, "measured": now_iso(), "seconds": round(seconds, 2)}
    if result is None:
        out.update(ok=False, why="no_spheres")
        return out
    out.update(ok=True, T_M_mandible=result["T_M_mandible"], T_M_B=result.get("T_M_B"),
               T_B_mandible=result.get("T_B_mandible"), n_views=int(result.get("n_views_used", len(shas))),
               views_used=list(result.get("views_used") or []), warnings=list(result.get("warnings") or []),
               calibrated=bool(result.get("calibrated")),
               spheres={n: {"radius": s["radius"], "rms_mm": s["rms_mm"], "view_spread_mm": s["view_spread_mm"]}
                        for n, s in result["spheres"].items()})
    return out


class Desk:
    """The scans of one phase: measured in a worker (update, process), read on the page's side (state), never two
    measures at once. store: STORE's folder; calibration: the rig calibration (ground_truth.CALIBRATION_JSON by
    default); info: a callable giving the twin's model extras (twin.Watcher.info) or None."""

    def __init__(self, run: trials.Run, store: Path, calibration: Path | None = None, info=None) -> None:
        self.run = run
        self.folder = Path(store) / run.phase
        self.calibration = Path(calibration) if calibration else gt.CALIBRATION_JSON
        self.info = info or (lambda: None)
        self.positions = []
        self._targets = None
        self._frame = None
        self._points = None

    # ── what everything is measured with ────────────────────────────────
    def cal_key(self) -> str:
        """The content of the rig calibration ('' when there is none): a new one measures everything again."""
        try:
            return hashlib.sha256(self.calibration.read_bytes()).hexdigest()[:16]
        except OSError:
            return ""

    def calibration_state(self) -> dict:
        try:
            c = gt.load_calibration(self.calibration)
        except (ValueError, KeyError, TypeError):
            return {"done": False, "bad": True}
        if c is None:
            return {"done": False}
        sd = c.get("radius_sd_mm") or {}
        return {"done": True, "date": c.get("date", ""), "scans": len(c.get("scans") or []),
                "radii": {k: round(v, 4) for k, v in c["radii"].items()},
                "sd_max_mm": round(max(sd.values()), 4) if sd else None}

    def targets(self) -> list[dict]:
        if self._targets is None:
            self._targets = targets()
        return self._targets

    def frame(self):
        if self._frame is None:
            self._frame = gt.load_frame()
        return self._frame

    def points(self) -> dict:
        """scene_points of the twin's model, read again at most every POINTS_S."""
        if self._points is None or time.monotonic() - self._points[0] > POINTS_S:
            self._points = (time.monotonic(), scene_points(self.info()))
        return self._points[1]

    def _measure(self, scans):
        """ground_truth.measure with the calibration; (result, error text)."""
        try:
            return gt.measure([str(p) for p in scans], calibration=self.calibration, T_B_mandible=self.frame()), ""
        except UNMEASURABLE as e:          # no spheres in it, or not a cloud at all (a few points, a broken file)
            return None, str(e)

    # ── the store ───────────────────────────────────────────────────────
    def path(self, trial: str) -> Path:
        return self.folder / f"{trial}.json"

    def load(self, trial: str) -> dict:
        d = read_json(self.path(trial)) or {}
        d.setdefault("trial", trial)
        d.setdefault("views", {})
        d.setdefault("sets", {})
        return d

    def save(self, trial: str, data: dict) -> None:
        write_json(self.path(trial), data)

    def measured_trials(self) -> list[str]:
        try:
            return sorted(p.stem for p in self.folder.glob("*.json"))
        except OSError:
            return []

    # ── a view, a set ───────────────────────────────────────────────────
    def view(self, data: dict, path: Path, measure: bool) -> dict | None:
        """The measure of one view (kept by its content and the calibration); None when it is still to make."""
        sha, cal = file_sha(path), self.cal_key()
        v = data["views"].get(sha)
        if v and v.get("cal") == cal:
            return v
        if not measure:
            return None
        if not self.positions:
            self.positions = scanner_positions(photo_az=self.run.config.photo_az)
        t0 = time.perf_counter()
        r, err = self._measure([path])
        v = view_summary(r, path, sha, cal, time.perf_counter() - t0, self.positions,
                         self.points()["R_W_mandible"], err)
        data["views"][sha] = v
        return v

    def view_alone(self, path: Path) -> dict:
        """One export measured alone, kept nowhere (to bind it before it is staged)."""
        if not self.positions:
            self.positions = scanner_positions(photo_az=self.run.config.photo_az)
        t0 = time.perf_counter()
        r, err = self._measure([path])
        return view_summary(r, path, file_sha(path), self.cal_key(), time.perf_counter() - t0, self.positions,
                            self.points()["R_W_mandible"], err)

    def keep_view(self, trial: str, v: dict) -> None:
        with _LOCK:
            d = self.load(trial)
            d["views"][v["sha"]] = v
            self.save(trial, d)

    def set_of(self, data: dict, which: str, paths: list[Path], views: list[dict | None], measure: bool) -> dict | None:
        """The measure of a set of views (which: post or ref), kept by their contents and the calibration; one view
        is its own set; None when it is still to make."""
        if not paths or any(v is None for v in views):
            return None
        shas, cal = [v["sha"] for v in views], self.cal_key()
        s = data["sets"].get(which)
        if s and s.get("shas") == shas and s.get("cal") == cal:
            return s
        if len(views) == 1:
            v = views[0]
            s = {"shas": shas, "cal": cal, "measured": v["measured"], "seconds": v["seconds"], "ok": v["ok"],
                 "why": "" if v["ok"] else "no_spheres"}
            if v["ok"]:
                s.update(T_M_mandible=v["T_M_mandible"], T_M_B=v.get("T_M_B"), n_views=1, views_used=[0],
                         warnings=v.get("warnings", []), calibrated=v.get("calibrated", False))
        elif not measure:
            return None
        else:
            good = [p for p, v in zip(paths, views) if v["ok"]]
            t0 = time.perf_counter()
            r, err = self._measure(good) if good else (None, "no view identified")
            s = set_summary(r, shas, cal, time.perf_counter() - t0, err)
            if r is not None and len(good) < len(paths):
                s["warnings"] = s["warnings"] + [f"{len(paths) - len(good)} of {len(paths)} views without their spheres"]
        data["sets"][which] = s
        return s

    # ── a trial ─────────────────────────────────────────────────────────
    def files(self, trial: str, staged: dict | None = None, log: list[dict] | None = None) -> dict:
        """{step key: its file} for the trial's scan steps: filed, else staged (the scanner's export)."""
        out = {}
        for s in slots(self.run, trial, log):
            p = filed(self.run, trial, s)
            if p is None and staged and staged.get(s["key"]):
                q = Path(staged[s["key"]])
                p = q if q.is_file() else None
            out[s["key"]] = p
        return out

    @staticmethod
    def before_of(trial: str, rows: dict) -> tuple[str, str]:
        """(trial, which) of the scans of before: its own reference scans (ref), or the post-impact scans of the trial
        pre_scan names (post); ('', '') when the log says none."""
        pre = str(rows.get(trial, {}).get("pre_scan") or "")
        if pre.endswith("_ref"):
            return pre[:-4], "ref"
        return (pre, "post") if pre in rows else ("", "")

    @staticmethod
    def last_reference(trial: str, rows: dict) -> str:
        """The trial whose reference scans the marker's position of `trial` is reckoned from ('' when none)."""
        seen, t = set(), trial
        while t and t not in seen:
            seen.add(t)
            pre = str(rows.get(t, {}).get("pre_scan") or "")
            if pre.endswith("_ref"):
                return pre[:-4]
            t = pre if pre in rows else ""
        return ""

    def registration(self, trial: str, which: str, log: list[dict], measure: bool, staged=None, data=None):
        """(state, set) of a trial's set of scans (post or ref): ready, partial (some of its views), missing,
        measuring, failed. A trial's data loaded here is saved again when something was measured for it."""
        own = data is None
        data = self.load(trial) if own else data
        sl = [s for s in slots(self.run, trial, log) if s["when"] == which]
        if not sl:
            return "missing", None
        files = self.files(trial, staged, log)
        paths = [files[s["key"]] for s in sl if files.get(s["key"])]
        if not paths:
            return "missing", None
        before = json.dumps(gt.jsonable(data["sets"].get(which)), sort_keys=True), len(data["views"])
        views = [self.view(data, p, measure) for p in paths]
        s = self.set_of(data, which, paths, views, measure)
        if own and measure and (json.dumps(gt.jsonable(data["sets"].get(which)), sort_keys=True),
                                len(data["views"])) != before:
            self.save(trial, data)
        if s is None:
            return "measuring", None
        if not s["ok"]:
            return "failed", s
        return ("ready" if len(paths) == len(sl) else "partial"), s

    def evaluate(self, trial: str, staged: dict | None = None, measure: bool = False) -> dict:
        """The scans of a trial and what they say (see the module's doc); measure: make what is missing (a worker),
        else only read what is kept. Returns the state the page shows, with result.log (the values for the trial's
        row, once the post-impact set is measured) and todo (something is still to measure)."""
        with _LOCK if measure else contextlib.nullcontext():
            return self._evaluate(trial, staged, measure)

    def _evaluate(self, trial: str, staged: dict | None, measure: bool) -> dict:
        log = self.run.log.rows()
        rows = {r["trial"]: r for r in log}
        row = rows.get(trial)
        if row is None:
            return {"trial": None, "slots": [], "before": {}, "result": {"state": "none"}, "todo": False}
        data = self.load(trial)
        sl = slots(self.run, trial, log)
        files = self.files(trial, staged, log)
        after = bool(row.get("time_impact"))
        todo = False
        out_slots = []
        for s in sl:
            p = files.get(s["key"])
            v = self.view(data, p, measure) if p else None
            if p is not None:
                state = "measuring" if v is None else ("ok" if v["ok"] and v.get("n_spheres", 0) >= 6 else "bad")
                todo |= v is None
            elif (s["when"] == "post") == after:
                state = "waiting"
            else:
                state = "later" if s["when"] == "post" else "skipped"
            seen = (v or {}).get("arm") or {}
            out_slots.append({"key": s["key"], "when": s["when"], "az": s["az"], "arm": s["arm"], "state": state,
                              "file": p.name if p else "", "filed": bool(p) and filed(self.run, trial, s) is not None,
                              "n_spheres": (v or {}).get("n_spheres"), "arm_seen": seen.get("arm"),
                              "exported": (v or {}).get("exported", ""),
                              "arm_off": bool(seen.get("arm") and s["arm"] and seen["arm"] != s["arm"])})
        before_trial, before_which = self.before_of(trial, rows)
        b_state, b_set = "missing", None
        if before_trial:
            b_state, b_set = self.registration(before_trial, before_which, log, measure,
                                               staged if before_trial == trial else None,
                                               data if before_trial == trial else None)
            todo |= b_state == "measuring"
        p_state, p_set = self.registration(trial, "post", log, measure, staged, data)
        todo |= p_state == "measuring"
        post_files = [files[s["key"]] for s in sl if s["when"] == "post" and files.get(s["key"])]
        n_post = sum(1 for s in sl if s["when"] == "post")
        if not post_files:
            res = {"state": "waiting" if after else "none"}
        elif p_state == "measuring":
            res = {"state": "measuring"}
        elif p_state == "failed":
            res = {"state": "failed", "why": "no_spheres"}
        elif b_state == "measuring":
            res = {"state": "measuring"}
        elif b_set is None or b_state == "failed":
            res = {"state": "failed", "why": "no_reference" if b_state == "missing" else "reference_spheres"}
        else:
            res = self.result(trial, log, rows, data, b_set, p_set, measure)
            res["state"] = "ready" if p_state == "ready" else "partial"
            todo |= res.pop("todo", False)
        res["views"], res["of"] = len(post_files), n_post
        if n_post and len(post_files) == n_post and p_set is not None and p_set["ok"]:
            # a set that could not be measured logs nothing: the operator scans again, or says it by hand
            views = [data["views"].get(file_sha(p)) for p in post_files]
            ok = all(v and v["ok"] and v.get("n_spheres", 0) >= 6 for v in views)
            out_log = {"spheres_ok": "1" if ok else "0"}
            if res.get("cap_shift_mm") is not None:
                out_log["cap_shift_mm"] = f"{res['cap_shift_mm']:.2f}"
            res["log"] = out_log
            # from the last export to the result, the first time this set was measured
            key = "|".join(p_set["shas"])
            lat = data.setdefault("latency", {})
            last = max((v["exported"] for v in views if v), default="")
            if key not in lat and measure and last:
                lat[key] = age_s(last, datetime.fromisoformat(p_set["measured"]).timestamp())
            res["seconds"] = lat.get(key)
        res["logged"] = data.get("logged", {})
        res["hand"] = data.get("hand", [])
        if measure:
            data["result"] = {k: v for k, v in res.items() if k not in ("logged", "hand")}
            data["row"] = {k: row.get(k, "") for k in ("session", "holder", "angle_deg", "direction_deg", "drape",
                                                         "status", "pre_scan", "reference")}
            self.save(trial, data)
        return {"trial": {k: row.get(k, "") for k in ("trial", "holder", "angle_deg", "direction_deg", "drape",
                                                         "status", "pre_scan", "reference", "time_impact",
                                                         "spheres_ok", "cap_shift_mm", "session")},
                "slots": out_slots, "before": {"trial": before_trial, "which": before_which, "state": b_state},
                "result": res, "todo": todo}

    def result(self, trial, log, rows, data, b_set, p_set, measure) -> dict:
        """The displacement of the trial from its sets of before and after: rotation, shift of the marker, the error
        at each target, the cap's shift since the last reference."""
        T0, T1 = mat(b_set["T_M_mandible"]), mat(p_set["T_M_mandible"])
        tg = self.targets()
        d = gt.displacement(T0, T1, {t["id"]: t["mandible"] for t in tg})
        pts = self.points()
        board = pts["board"] if pts["board"] is not None else marker_centroid(T0)
        out = {"rot_deg": round(d["rot_deg"], 4), "trans_mm": [round(float(x), 4) for x in d["trans_mm"]],
               "shift_mm": round(carried(d["dT"], board), 4), "dT": d["dT"],
               "tre": [{"id": t["id"], "name": t["name"], "mm": round(d["tre_mm"][t["id"]], 4)} for t in tg],
               "tre_max_mm": round(d["tre_max_mm"], 4), "tre_mean_mm": round(d["tre_mean_mm"], 4),
               "calibrated": bool(p_set.get("calibrated")) and bool(b_set.get("calibrated")),
               "warnings": sorted(set(p_set.get("warnings", []) + b_set.get("warnings", []))),
               "cap_limit_mm": float(self.run.config.reposition_mm),
               "cap_from": "holder" if trials.cap_point_M() is not None else "scene" if pts["cap"] is not None
               else "spheres", "measured": p_set.get("measured", "")}
        ref = self.last_reference(trial, rows)
        st, rset = (self.registration(ref, "ref", log, measure, None, data if ref == trial else None) if ref
                    else ("missing", None))
        out["reference"] = ref
        if rset is not None and rset.get("ok") and st in ("ready", "partial"):
            Tr = mat(rset["T_M_mandible"])
            cap_M = trials.cap_point_M()            # the holder's strike point in M (pilot.py model)
            if cap_M is not None:
                cap = gt.transform(gt.inverse(Tr), [cap_M])[0]
            else:
                cap = pts["cap"] if pts["cap"] is not None else marker_centroid(Tr)
            out["cap_shift_mm"] = round(carried(gt.inverse(Tr) @ T1, cap), 4)
            out["reposition"] = out["cap_shift_mm"] > out["cap_limit_mm"]
        else:
            out["cap_shift_mm"] = None
            out["todo"] = st == "measuring"
        return out

    # ── the log ─────────────────────────────────────────────────────────
    def log_values(self, trial: str, values: dict) -> dict:
        """The online values written into the trial's row (see the module's doc): {written, changed, hand}."""
        if not values:
            return {}
        log = self.run.log
        row = log.row(trial)
        data = self.load(trial)
        ours = data.get("logged") or {}
        write, change, hand = {}, {}, []
        for k, v in values.items():
            v = trials.check_value(k, v)
            cur = row.get(k, "")
            if cur == v:
                continue
            if cur == "":
                write[k] = v
            elif cur == ours.get(k):
                change[k] = v
            else:
                hand.append(k)
        if write:
            log.record(trial, write)
        if change:
            log.edit_many(trial, change, EDIT_REASON)
        if write or change or hand != data.get("hand", []):
            with _LOCK:
                data = self.load(trial)
                data["logged"] = {**ours, **write, **change}
                data["hand"] = hand
                self.save(trial, data)
        return {"written": write, "changed": change, "hand": hand}

    # ── the twin's cache ────────────────────────────────────────────────
    def twin_entry(self, trial: str, cache: Path | None, dT) -> bool:
        """The marker's motion kept for the twin (twin.scan_result) once the scans are filed, so that it shows the
        trial at once; False when there is nothing to keep yet."""
        if cache is None or dT is None:
            return False
        import twin
        row = self.run.log.row(trial)
        post, ref = twin.scans_of(self.run, trial), twin.reference_scans(self.run, row)
        if not post or not ref:
            return False
        key = twin.scan_key(post, ref, self.calibration)
        path = Path(cache) / "scans" / f"{self.run.phase}_{trial}.json"
        old = read_json(path)
        if old and old.get("key") == key:
            return True
        write_json(path, {"key": key, "dT": mat(dT).tolist(), "warnings": [], "calibrated": bool(self.cal_key()),
                          "measured": now_iso()})
        return True

    # ── gt_poses.csv ────────────────────────────────────────────────────
    def poses_rows(self) -> list[dict]:
        rows = {r["trial"]: r for r in self.run.log.rows()}
        out = []
        frame = self.frame()
        for t in self.measured_trials():
            if t not in rows:
                continue
            d = self.load(t)
            res, r = d.get("result") or {}, rows[t]
            sets = d.get("sets") or {}
            post = sets.get("post") or {}
            bt, bw = self.before_of(t, rows)
            before = (d if bt == t else self.load(bt)).get("sets", {}).get(bw) if bt else None
            row = {"trial": t, **{k: r.get(k, "") for k in ("phase", "session", "holder", "angle_deg", "direction_deg",
                                                            "drape", "status", "pre_scan")},
                   "state": res.get("state", ""), "views": res.get("views", ""),
                   "spheres_ok": (res.get("log") or {}).get("spheres_ok", ""),
                   "calibrated": int(bool(res.get("calibrated"))) if res else "",
                   "rot_deg": res.get("rot_deg", ""), "shift_mm": res.get("shift_mm", ""),
                   "cap_shift_mm": res.get("cap_shift_mm") if res.get("cap_shift_mm") is not None else "",
                   "tre_max_mm": res.get("tre_max_mm", ""), "tre_mean_mm": res.get("tre_mean_mm", ""),
                   "warnings": "; ".join(res.get("warnings", [])), "measured": res.get("measured", "")}
            for tg in res.get("tre", []):
                row[f"tre_{tg['id']}_mm"] = tg["mm"]
            for name, s in (("pre", before), ("post", post if post.get("ok") else None)):
                T = mat((s or {}).get("T_M_mandible")) if s and s.get("ok") else None
                row[f"T_M_mandible_{name}"] = " ".join(f"{x:.6f}" for x in T.ravel()) if T is not None else ""
                TB = T @ gt.inverse(frame) if T is not None and frame is not None else None
                row[f"T_M_B_{name}"] = " ".join(f"{x:.6f}" for x in TB.ravel()) if TB is not None else ""
            dT = mat(res.get("dT")) if res.get("dT") is not None else None
            row["dT_mandible"] = " ".join(f"{x:.6f}" for x in dT.ravel()) if dT is not None else ""
            dTB = frame @ dT @ gt.inverse(frame) if dT is not None and frame is not None else None
            row["dT_B"] = " ".join(f"{x:.6f}" for x in dTB.ravel()) if dTB is not None else ""
            out.append(row)
        return out

    def write_poses(self) -> Path:
        rows = self.poses_rows()
        tre = [f"tre_{t['id']}_mm" for t in self.targets()]
        cols = ["trial", "phase", "session", "holder", "angle_deg", "direction_deg", "drape", "status", "pre_scan",
                "state", "views", "spheres_ok", "calibrated", "rot_deg", "shift_mm", "cap_shift_mm", "tre_max_mm",
                "tre_mean_mm"] + tre + ["T_M_mandible_pre", "T_M_mandible_post", "T_M_B_pre", "T_M_B_post",
                                        "dT_mandible", "dT_B", "warnings", "measured"]
        path = self.folder / POSES_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        trials.write_csv(path, rows, cols)
        return path


def process(desk: Desk, day: str | None = None, every: bool = False, cache: Path | None = None,
            progress=None) -> dict:
    """The scans of a day's trials (session == day, today when None; every: all trials) measured with the calibration
    of now, what is missing for each, their values logged, gt_poses.csv written again. Returns {trials, measured,
    failed, missing: {trial: [[kind, item]]}, logged: {trial: fields}, file}. progress(i, n, trial) is called before
    each trial."""
    run = desk.run
    day = day or date.today().isoformat()
    rows = [r for r in run.log.rows() if r["status"] != trials.VOID and (every or str(r.get("session")) == day)]
    out = {"trials": len(rows), "measured": 0, "failed": [], "missing": {}, "logged": {}, "day": "" if every else day}
    for i, r in enumerate(rows):
        t = r["trial"]
        if progress:
            progress(i, len(rows), t)
        with contextlib.suppress(trials.TrialsError, OSError, ValueError):
            miss = [list(m) for m in run.missing(t) if m[0] not in ("online",)]
            if miss:
                out["missing"][t] = miss
        if not any(filed(run, t, s) for s in slots(run, t)):
            continue
        st = desk.evaluate(t, measure=True)
        res = st["result"]
        if res.get("state") in ("ready", "partial"):
            out["measured"] += 1
        elif res.get("state") == "failed":
            out["failed"].append(t)
        if res.get("log"):
            with contextlib.suppress(trials.TrialsError, OSError, ValueError):
                done = desk.log_values(t, res["log"])
                if done.get("written") or done.get("changed"):
                    out["logged"][t] = sorted({**done["written"], **done["changed"]})
        with contextlib.suppress(OSError, ValueError, KeyError):
            desk.twin_entry(t, cache, res.get("dT"))
    out["file"] = str(desk.write_poses())
    return out


# ── runs without an impact ──────────────────────────────────────────────────
class Capture:
    """A run of scans without an impact: kind calibration (the spheres' radii and the rigs' shapes, once), repeat
    (the repeatability of the scans) or reseat (holder B lifted and let go between the scans: the repeatability of
    its seats). raw: the raw data's folder (RAW); store: STORE's folder."""

    def __init__(self, kind: str, raw: Path, store: Path, calibration: Path | None = None) -> None:
        if kind not in CAPTURES:
            raise ValueError(f"no run of kind {kind}")
        self.kind = kind
        self.folder = Path(raw) / STORE / CAPTURE_DIRS[kind]
        self.store = Path(store)
        self.path = self.store / f"{kind}.json"
        self.calibration = Path(calibration) if calibration else gt.CALIBRATION_JSON
        self.frame = gt.load_frame()

    def load(self) -> dict:
        d = read_json(self.path) or {}
        d.setdefault("kind", self.kind)
        d.setdefault("files", [])
        d.setdefault("want", self.default_want())
        return d

    def default_want(self) -> int:
        return {"repeat": REPEAT_N, "reseat": RESEAT_N}.get(self.kind, gt.MIN_CALIBRATION_SCANS)

    def start(self, want: int | None = None) -> dict:
        """A new run (the scans of an earlier one stay in RAW)."""
        least = gt.MIN_CALIBRATION_SCANS if self.kind == "calibration" else 3
        d = {"kind": self.kind, "started": now_iso(), "files": [], "on": True,
             "want": max(least, int(want or self.default_want()))}
        write_json(self.path, d)
        return d

    def stop(self) -> dict:
        d = self.load()
        d["on"] = False
        write_json(self.path, d)
        return d

    def next_name(self, suffix: str) -> Path:
        prefix = CAPTURES[self.kind]
        self.folder.mkdir(parents=True, exist_ok=True)
        taken = {p.stem.lower() for p in self.folder.iterdir()}
        n = 1
        while f"{prefix}_{n:03d}" in taken:
            n += 1
        return self.folder / f"{prefix}_{n:03d}{suffix.lower()}"

    def add(self, src: Path) -> dict:
        """An export of the run: copied under the next free name (never over a file), checked against its source,
        measured alone (free radii for a calibration, which measures them; the calibration's for the
        repeatability)."""
        src = Path(src)
        dest = self.next_name(src.suffix)
        part = dest.with_name(dest.name + trials.PART_SUFFIX)
        with contextlib.suppress(FileNotFoundError):
            part.unlink()
        size, digest = trials.copy_verified(src, part)
        trials._place_new(part, dest)
        t0 = time.perf_counter()
        entry = {"name": dest.name, "source": src.name, "sha": digest, "bytes": size, "exported": stamp(src),
                 "added": now_iso()}
        try:
            if self.kind == "calibration":
                r = gt.measure(str(dest), rigs=gt.RIGS, radii=None, T_B_mandible=self.frame)
            else:
                r = gt.measure(str(dest), calibration=self.calibration, T_B_mandible=self.frame)
            entry.update(ok=True, n_spheres=int(r["views"][0]["n_spheres"]), T_M_mandible=r["T_M_mandible"],
                         calibrated=bool(r.get("calibrated")), warnings=r["warnings"],
                         radii={n: s["radius"] for n, s in r["spheres"].items()})
        except UNMEASURABLE as e:
            m = re.search(r"(\d+) spheres found", str(e))
            entry.update(ok=False, n_spheres=int(m.group(1)) if m else 0, why="few_spheres" if m else "not_identified")
        entry["seconds"] = round(time.perf_counter() - t0, 2)
        d = self.load()
        d["files"].append(entry)
        if self.kind in ("repeat", "reseat"):       # a reseating's scatter is the holder's repeatability
            d["result"] = spread([f for f in d["files"] if f.get("ok")], targets())
        write_json(self.path, d)
        if self.kind in ("repeat", "reseat"):
            self.write_rows(d)
        return entry

    def calibrate(self) -> dict:
        """The rig calibration from the run's good scans (ground_truth.calibrate), written where every measure reads
        it."""
        d = self.load()
        good = [self.folder / f["name"] for f in d["files"] if f.get("ok")]
        if len(good) < gt.MIN_CALIBRATION_SCANS:
            raise ValueError(f"{len(good)} good scans: a calibration takes at least {gt.MIN_CALIBRATION_SCANS}")
        cal = gt.calibrate(good, out=self.calibration, T_B_mandible=self.frame)
        d["on"] = False
        d["calibrated"] = now_iso()
        write_json(self.path, d)
        return cal

    def write_rows(self, d: dict) -> Path:
        """The repeatability's rows: each scan's distance from the mean registration at each target."""
        tg = targets()
        res = d.get("result") or {}
        dev = res.get("deviation_mm") or {}
        cols = ["name", "exported", "ok", "n_spheres"] + [f"dev_{t['id']}_mm" for t in tg] + ["dev_max_mm"]
        rows = []
        for f in d["files"]:
            r = {k: f.get(k, "") for k in ("name", "exported", "n_spheres")}
            r["ok"] = int(bool(f.get("ok")))
            for t in tg:
                r[f"dev_{t['id']}_mm"] = (dev.get(f["name"]) or {}).get(t["id"], "")
            r["dev_max_mm"] = max((v for v in (dev.get(f["name"]) or {}).values()), default="")
            rows.append(r)
        path = self.store / ("reseating.csv" if self.kind == "reseat" else "repeatability.csv")
        trials.write_csv(path, rows, cols)
        return path


def spread(files: list[dict], tg: list[dict]) -> dict:
    """The repeatability of the registration over scans of a bench left alone: at each target (fixed to the head, in
    the marker's frame M) the scatter of where each scan puts it about their mean, as an SD (root mean square
    distance, n - 1); spread_mm, the largest of them, holds under REPEAT_LIMIT_MM."""
    n = len(files)
    if n < 2:
        return {"n": n}
    P = np.array([t["mandible"] for t in tg], np.float64)
    Q = np.array([gt.transform(mat(f["T_M_mandible"]), P) for f in files])        # scans x targets x 3
    E = np.linalg.norm(Q - Q.mean(0), axis=2)
    sd = np.sqrt((E ** 2).sum(0) / (n - 1))
    radii = {}
    for k in files[0].get("radii") or {}:
        v = [f["radii"][k] for f in files if k in (f.get("radii") or {})]
        radii[k] = round(float(np.std(v, ddof=1)), 5) if len(v) > 1 else None
    return {"n": n, "sd_mm": {t["id"]: round(float(s), 5) for t, s in zip(tg, sd)},
            "spread_mm": round(float(sd.max()), 5), "median_mm": round(float(np.median(sd)), 5),
            "p95_mm": round(float(np.percentile(E, 95)), 5), "limit_mm": REPEAT_LIMIT_MM,
            "ok": bool(sd.max() < REPEAT_LIMIT_MM), "radius_sd_mm": radii,
            "deviation_mm": {f["name"]: {t["id"]: round(float(e), 5) for t, e in zip(tg, E[i])}
                             for i, f in enumerate(files)}}


# ── the simulated scanner ───────────────────────────────────────────────────
def scan_pose(az_deg: float, centroid, jitter=None) -> np.ndarray:
    """T_W_scan of the scanner at the arm position of camera azimuth az_deg: the bench scene's pose of that position
    (azimuth, elevation, distance from the spheres' centroid), else ground_truth's nominal one; optical axis on the
    centroid, x horizontal (as ground_truth.scanner_poses)."""
    pos = scanner_positions()
    p = min(pos, key=lambda q: abs(q["camera_deg"] - az_deg)) if pos else None
    if p and abs(p["camera_deg"] - az_deg) < 0.5:
        a, e, r = math.radians(p["azimuth_deg"]), math.radians(p["elevation_deg"]), p["distance_mm"]
    else:
        a, e, r = math.radians(az_deg), math.radians(gt.SCENE["elevation_deg"]), gt.SCENE["range_mm"]
    if jitter is not None:
        a, e, r = a + jitter[0], e + jitter[1], r + jitter[2]
    d = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
    z = -d
    x = gt.unit(np.cross([0.0, 0.0, 1.0], z))
    return gt.pose(np.column_stack([x, np.cross(z, x), z]), np.asarray(centroid) + r * d)


def synthetic_view(T_W_M, T_W_mandible, az_deg: float, rng, noise: float = SIM_NOISE_MM, spacing: float = 0.17,
                   rigs=gt.RIGS, radius: float = gt.SPHERE_D / 2) -> np.ndarray:
    """One simulated export of the scanner at an arm position, in the scanner's own frame (ground_truth's model of
    the EinScan-SP: ground_truth.synthetic_scan, one view)."""
    T_W = {"marker": T_W_M, "mandible": T_W_mandible}
    centres, posts = [], []
    hub = gt.transform(T_W_mandible, [gt.HUB_MANDIBLE])[0]
    for name, tpl in rigs.items():
        r_post, dirs = gt.SCENE["posts"][name]
        for i, c in enumerate(gt.transform(T_W[name], tpl)):
            a = gt.unit(hub - c) if dirs is None else T_W[name][:3, :3] @ np.asarray(dirs[i])
            centres.append(c)
            posts.append((r_post, a))
    centres = np.array(centres)
    jit = (rng.normal(0, math.radians(0.5)), rng.normal(0, math.radians(0.5)), rng.normal(0, 3))
    T = scan_pose(az_deg, centres.mean(0), jit)
    O, xs = T[:3, 3], T[:3, 0]
    cams = [O + gt.SCENE["baseline_mm"] * xs, O - gt.SCENE["baseline_mm"] * xs]
    pts = [gt.sample_sphere(c, j, centres, O, cams, rng, radius, spacing, noise, posts[j])
           for j, c in enumerate(centres)]
    pts.append(gt.clutter_points(T_W_M, O, centres, rng, noise, spacing * 2.5, radius))
    return gt.transform(gt.inverse(T), np.vstack(pts))


class SimScanner:
    """The bench's scanner in simulation: exports synthetic scans of the bench as the trials leave it (see the
    module's doc). Its state (the marker's pose, the trials whose impact moved it, those put back with the jig)
    is kept in `state`, so a simulation goes on across sessions of the app."""

    def __init__(self, state: Path) -> None:
        self.state_path = Path(state)
        self.lock = threading.Lock()
        T_W_mandible, T_W_M = gt.scene_poses()
        self.T_W_mandible, self.T_W_M0 = T_W_mandible, T_W_M

    def _load(self) -> dict:
        d = read_json(self.state_path) or {}
        d.setdefault("T_W_M", self.T_W_M0.tolist())
        d.setdefault("moved", [])
        d.setdefault("reset", [])
        d.setdefault("n", 0)
        return d

    def info(self) -> dict:
        """What twin.simulated_motion needs, in ground_truth's scene: the head, the ball of A, the board's centre."""
        T = mat(self._load()["T_W_M"])
        return {"T_W_mandible": self.T_W_mandible.tolist(),
                "ball_centre": gt.transform(T, [gt.SCENE["ball_M"]])[0].tolist(),
                "board_centre": gt.transform(T, [(22.5, 22.5, 0.0)])[0].tolist()}

    def rng(self, *parts) -> np.random.Generator:
        return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16))

    def export(self, folder: Path, name: str, az: float, row: dict | None = None, when: str = "post",
               kind: str = "trial") -> Path:
        """A scan written into the export folder as `name` (whole, then named), taken at arm position az: of the
        trial `row` before (ref) or after (post) its impact, or of a run without an impact (kind calibration or
        repeat)."""
        import twin
        with self.lock:
            d = self._load()
            d["n"] += 1
            T_W_M, T_W_mand = mat(d["T_W_M"]), self.T_W_mandible
            trial = (row or {}).get("trial", "")
            rng = self.rng(trial, when, az, d["n"])
            if kind == "trial" and row is not None:
                reasons = set(str(row.get("reference") or "").split(";"))
                if when == "ref" and reasons & {"first", "holder", "reposition"} and trial not in d["reset"]:
                    axis = gt.unit(rng.normal(size=3))
                    R = twin.rot(axis, rng.uniform(0, SIM_JIG[0]))
                    t = gt.unit(rng.normal(size=3)) * rng.uniform(0, SIM_JIG[1])
                    c = gt.transform(self.T_W_M0, [gt.SCENE["ball_M"]])[0]
                    T_W_M = gt.pose(R, c - R @ c + t) @ self.T_W_M0
                    d["reset"].append(trial)
                if when == "post" and trial not in d["moved"]:
                    D = twin.simulated_motion(row, self.info())
                    T_W_M = D @ T_W_M
                    d["moved"].append(trial)
                d["T_W_M"] = T_W_M.tolist()
            elif kind == "reseat":         # holder B lifted and let go: back in its seats within twin's B_MM, B_DEG
                c = gt.transform(self.T_W_M0, [gt.SCENE["ball_M"]])[0]
                R = twin.rot(gt.unit(rng.normal(size=3)), rng.uniform(*twin.B_DEG))
                t = gt.unit(rng.normal(size=3)) * rng.uniform(*twin.B_MM)
                T_W_M = gt.pose(R, c - R @ c + t) @ self.T_W_M0
            elif kind == "calibration":
                c = gt.transform(T_W_mand, [gt.HUB_MANDIBLE])[0]
                R = twin.rot([0, 0, 1], rng.uniform(-SIM_TURN_DEG[0], SIM_TURN_DEG[0])) @ \
                    twin.rot(gt.unit(np.r_[rng.normal(size=2), 0.0]), rng.uniform(-SIM_TURN_DEG[1], SIM_TURN_DEG[1]))
                Q = gt.pose(R, c - R @ c)
                T_W_M, T_W_mand = Q @ T_W_M, Q @ T_W_mand
            write_json(self.state_path, d)
        cloud = synthetic_view(T_W_M, T_W_mand, az, rng)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / name
        part = folder / (name + ".writing")
        gt.write_ply(part, cloud)
        os.replace(part, dest)
        return dest


# ── command line and test ───────────────────────────────────────────────────
def selftest(work: Path) -> int:
    """A simulated phase: reference scans, impacts on A and B, a provisional result, the arm positions seen in the
    scans, the binding, the log's values (empty, again after a calibration, never over a typed one), the cap's shift
    since the reference, a calibration and a repeatability run, gt_poses.csv; each result within 30 s of its
    export."""
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    work = Path(work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    raw, protocol, store = work / "raw", work / "protocol", work / "processed" / STORE
    cal_path = work / "processed" / "targets" / "rig_calibration.json"
    run = trials.Run("campaign", raw, protocol)
    sim = SimScanner(work / "sim_scanner.json")
    export = work / "scanner_export"
    desk = Desk(run, store, cal_path)
    cfg = run.config
    expect(slot_of(f"ref_scan_az{cfg.scan_az[0]}", cfg) == {"key": f"ref_scan_az{cfg.scan_az[0]}", "when": "ref",
                                                           "az": cfg.scan_az[0],
                                                           "arm": list(cfg.photo_az).index(cfg.scan_az[0]) + 1}
           and slot_of("photo_az40", cfg) is None,
           f"the scan steps: before or after the impact, arm position from the configuration ({cfg.scan_az})")

    def scan(row, key, name=None):
        s = slot_of(key, cfg)
        return sim.export(export, name or f"export_{row['trial']}_{key}.ply", s["az"], row, s["when"])

    # trial 1: reference scans, then the impact
    plan = run.next()
    row = run.start(plan, swing_mass_g=150)
    t1 = row["trial"]
    keys = [s["key"] for s in slots(run, t1)]
    expect(any(k.startswith("ref_") for k in keys) and sum(not k.startswith("ref_") for k in keys) == len(cfg.scan_az),
           f"{t1}: reference scans and post-impact scans ({keys})")
    staged = {}
    first = scan(run.log.row(t1), keys[0])
    v = desk.view_alone(first)
    expect(v["ok"] and v["n_spheres"] == 6 and v["arm"].get("arm") == slot_of(keys[0], cfg)["arm"],
           f"a view alone: six spheres, taken at arm position {v['arm'].get('arm')} "
           f"(azimuth {v['arm'].get('az_deg')}°, {v['arm'].get('off_deg')}° from the scene's pose)")
    # a 3MF export (a zip of XML, here in microns) reads as the same points, in mm
    import zipfile
    pts = gt.load_cloud(first)[:2000]
    tmf = work / "export.3mf"
    with zipfile.ZipFile(tmf, "w") as z:
        z.writestr("3D/3dmodel.model", '<?xml version="1.0"?><model unit="micron" xmlns="http://schemas.microsoft.com/'
                   '3dmanufacturing/core/2015/02"><resources><object id="1" type="model"><mesh><vertices>'
                   + "".join(f'<vertex x="{x * 1000:.3f}" y="{y * 1000:.3f}" z="{z * 1000:.3f}" />' for x, y, z in pts)
                   + "</vertices></mesh></object></resources></model>")
    back = gt.load_cloud(tmf)
    expect(back.shape == pts.shape and float(np.abs(back - pts).max()) < 1e-5,
           f"a 3MF export read as its points, in mm ({len(back)} points)")
    # an export of the second position first: bound by its arm position, not by the order
    second = scan(run.log.row(t1), keys[1])
    v2 = desk.view_alone(second)
    k2 = bind(run, run.log.row(t1), staged, v2["arm"].get("arm"))
    expect(k2 == keys[1], f"bound by the arm position it was taken from: {k2}")
    k_none = bind(run, run.log.row(t1), staged)
    expect(k_none == keys[0], "without one, the first step still waiting")
    staged = {keys[0]: str(first), keys[1]: str(second)}
    expect(bind(run, run.log.row(t1), staged) is None, "every reference step has its scan: a new scan is not used")
    expect(bind(run, run.log.row(t1), staged, v["arm"]["arm"]) == keys[0], "the same arm again: it replaces its scan")
    for k in keys[:2]:
        run.store(t1, slots(run, t1)[keys.index(k)]["file"], staged[k])
    run.confirm(t1, "release")
    row = run.log.row(t1)
    expect(bind(run, row, {}) == keys[2], f"after the impact: the post-impact steps ({keys[2]})")
    p1 = scan(row, keys[2])
    t0 = time.time()
    st = desk.evaluate(t1, {keys[2]: str(p1)}, measure=True)
    res = st["result"]
    expect(res["state"] == "partial" and res["views"] == 1 and res.get("rot_deg") is not None and "log" not in res,
           f"one view in: a provisional result ({res.get('rot_deg')}°), nothing logged yet")
    p2 = scan(row, keys[3])
    st = desk.evaluate(t1, {keys[2]: str(p1), keys[3]: str(p2)}, measure=True)
    res = st["result"]
    took = time.time() - t0
    expect(res["state"] == "ready" and res["views"] == 2 and res.get("log", {}).get("spheres_ok") == "1"
           and len(res["tre"]) == len(targets()) and res["seconds"] is not None and res["seconds"] < 30,
           f"both views: the marker's shift {res.get('shift_mm')} mm, {res.get('rot_deg')}°, worst target "
           f"{res.get('tre_max_mm')} mm, measured {res.get('seconds')} s after the export ({took:.1f} s here)")
    # the truth of the simulation: the motion it gave the marker
    import twin
    D = twin.simulated_motion(run.log.row(t1), sim.info())   # drawn again from the trial: the same motion
    Tm = sim.T_W_mandible
    dT_true = gt.inverse(Tm) @ gt.inverse(D) @ Tm
    P = np.array([t["mandible"] for t in targets()])
    err = float(np.abs(np.linalg.norm(gt.transform(dT_true, P) - P, axis=1)
                       - np.array([t["mm"] for t in res["tre"]])).max())
    expect(err < 0.1, f"the error at the targets as the simulated impact made it, within {err:.3f} mm "
                      f"(free radii, no calibration yet)")
    expect(res.get("cap_shift_mm") is not None and res["reference"] == t1 and not res["calibrated"],
           f"the cap's shift since the reference of {t1}: {res.get('cap_shift_mm')} mm; not calibrated said")
    w = desk.log_values(t1, res["log"])
    r1 = run.log.row(t1)
    expect(w["written"] and r1["spheres_ok"] == "1" and r1["cap_shift_mm"] == res["log"]["cap_shift_mm"],
           f"the log: spheres_ok 1, cap_shift_mm {r1['cap_shift_mm']}")
    st_read = desk.evaluate(t1, {keys[2]: str(p1), keys[3]: str(p2)})
    expect(not st_read["todo"] and st_read["result"]["state"] == "ready",
           "read again on the page's side: kept, nothing to measure")
    for k, p in ((keys[2], p1), (keys[3], p2)):
        run.store(t1, slots(run, t1)[keys.index(k)]["file"], p)
    run.confirm(t1, "spheres")
    for s in run.steps(t1):
        for n in s.files:
            if not run.find(t1, n):
                src = work / "stand_in" / n
                src.parent.mkdir(parents=True, exist_ok=True)
                src.write_bytes(b"stand-in " + n.encode())
                run.store(t1, n, src)
    run.confirm(t1, "close")
    # trial 2: no reference of its own, the post-impact scans of t1 are its before
    row = run.start(run.next(), swing_mass_g=150)
    t2 = row["trial"]
    keys2 = [s["key"] for s in slots(run, t2)]
    run.confirm(t2, "release")
    row = run.log.row(t2)
    files2 = {k: str(scan(row, k)) for k in keys2}
    st2 = desk.evaluate(t2, files2, measure=True)
    res2 = st2["result"]
    expect(st2["before"]["trial"] == t1 and st2["before"]["which"] == "post" and res2["state"] == "ready"
           and res2["reference"] == t1 and res2["cap_shift_mm"] is not None,
           f"{t2}: before it the scans after {t1}'s impact; the cap's shift still since {t1}'s reference "
           f"({res2['cap_shift_mm']} mm, the trial alone {res2['shift_mm']} mm)")
    run.log.record(t2, {"cap_shift_mm": "9.99"})            # typed by hand before the measure
    w2 = desk.log_values(t2, res2["log"])
    expect(run.log.row(t2)["cap_shift_mm"] == "9.99" and "cap_shift_mm" in w2["hand"],
           "a value typed by hand is never changed")
    # a calibration run: three scans, the bench turned between them; then everything is measured again
    cap = Capture("calibration", raw, store, cal_path)
    cap.start()
    for i in range(gt.MIN_CALIBRATION_SCANS):
        src = sim.export(export, f"cal_export_{i}.ply", cfg.scan_az[0], kind="calibration")
        e = cap.add(src)
    names = sorted(p.name for p in cap.folder.iterdir())
    expect(e["ok"] and len(names) == gt.MIN_CALIBRATION_SCANS and names[0] == "cal_001.ply",
           f"a calibration's scans copied into the raw data under their names ({', '.join(names)})")
    cal = cap.calibrate()
    off = max(abs(r - gt.SPHERE_D / 2) for r in cal["radii"].values())
    expect(cal_path.is_file() and off < 0.05 and desk.calibration_state()["done"],
           f"the radii calibrated ({off:.3f} mm from the drawn ones), said as done")
    st1 = desk.evaluate(t1, measure=False)
    expect(st1["todo"], "a new calibration: the scans are to measure again")
    st1 = desk.evaluate(t1, measure=True)
    w3 = desk.log_values(t1, st1["result"]["log"])
    expect(st1["result"]["calibrated"] and (w3["changed"] or run.log.row(t1)["cap_shift_mm"]
                                            == st1["result"]["log"]["cap_shift_mm"]),
           f"measured again with the calibrated radii; the log's value changed with its reason "
           f"({run.log.row(t1)['cap_shift_mm']} mm)")
    edits = [e for e in run.log.edits() if e["trial"] == t1 and e["field"] == "cap_shift_mm"]
    expect(not w3["changed"] or (edits and edits[-1]["reason"] == EDIT_REASON), "the change in the audit trail")
    # repeatability: scans of a bench left alone
    rep = Capture("repeat", raw, store, cal_path)
    rep.start(5)
    for i in range(5):
        rep.add(sim.export(export, f"rep_export_{i}.ply", cfg.scan_az[0], kind="repeat"))
    rr = rep.load()["result"]
    expect(rr["n"] == 5 and 0 < rr["spread_mm"] < 0.2 and (store / "repeatability.csv").is_file(),
           f"repeatability over 5 scans: spread {rr['spread_mm']} mm at the worst target (limit {REPEAT_LIMIT_MM}, "
           f"{'held' if rr['ok'] else 'not held'})")
    # the day's processing
    out = process(desk, every=True)
    with open(out["file"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expect(len(rows) == 2 and rows[0]["trial"] == t1 and rows[0]["T_M_B_post"].count(" ") == 15
           and float(rows[0]["tre_max_mm"]) > 0 and t2 in out["missing"],
           f"the scans processed: gt_poses.csv with {len(rows)} trials, registrations and errors; {t2} not complete")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("process", help="the scans of a day's trials into gt_poses.csv")
    pr.add_argument("--phase", default="campaign", choices=tuple(trials.PREFIX))
    pr.add_argument("--day", help="YYYY-MM-DD (today by default)")
    pr.add_argument("--all", action="store_true", help="every trial of the phase")
    te = sub.add_parser("test")
    te.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "scan_desk_test")
    a = p.parse_args(argv)
    if a.cmd == "test":
        return selftest(a.work)
    import twin
    run = trials.Run(a.phase, trials.RAW, trials.PROTOCOL)
    w = twin.Watcher(twin.CACHE, blender=None)
    desk = Desk(run, PROCESSED / STORE, info=w.info)
    out = process(desk, a.day, a.all, twin.CACHE,
                  progress=lambda i, n, t: print(f"{i + 1}/{n} {t}", flush=True))
    print(f"{out['measured']} of {out['trials']} trials measured, written to {out['file']}")
    for t, m in out["missing"].items():
        print(f"{t}: missing " + "; ".join(f"{k} {v}" for k, v in m))
    if out["failed"]:
        print("not measured: " + ", ".join(out["failed"]))
    return 1 if out["missing"] or out["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
