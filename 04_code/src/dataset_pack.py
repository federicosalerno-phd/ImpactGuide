"""The dataset of the bench as it is published: what it holds, whether it is whole and coherent, and the package.

Real trials (trials.py), per phase: the log, its edits and the manifest of the raw files (bytes and SHA-256 of each
file as it was filed), and per trial its photos, videos and scans; the ground truth of each trial as the Scans
page measured it from its scans (scan_desk.py, its store processed/ground_truth): the marker's motion dT in the
mandible frame, the registrations of before and after, the error left at each clinical target; never measured here,
a trial without it is said. With them the protocol (configuration and sequence of each phase, the log's data
dictionary), the phone's calibration (one per camera mode), the targets with the rig calibration of the spheres and
the scans of the runs without an impact (the rig calibration, the repeatability). Only closed trials (status ok) go
into the package; void trials stay in the log, open ones are said.

Synthetic dataset of the digital twin (synth_twin.py, indexed by synth_data.py): sessions of photos and video frames
with their label images and poses (frames.jsonl), split by session into train, val and test; the check sessions
(900 and above) stay out. Its fingerprint (scene_print) says which version of the bench scene it was rendered on,
compared with the model of the twin in use (twin.py).

inventory() says what there is from the logs, the indexes and the folders, without reading the files; check() reads
every file: SHA-256 against the manifest for the trials, every frame against its images for the synthetic sessions;
export() writes the package into a folder or a zip, never over one that exists (it is written beside its place as
<name>.part and given its name last; a stopped or failed export leaves nothing):

    impactguide_dataset_YYYYMMDD/
        README.md                   what it is, its layout, frames and units, versions, licence
        LICENSE.txt                 data and models CC BY 4.0, code MIT
        index.json                  counts, versions, and the problems found while packing
        files.csv                   every other file with its bytes and SHA-256
        verify.py                   checks every file against files.csv (Python 3, standard library only)
        real/protocol/              configuration and sequence of each phase, the log's data dictionary
        real/calibration/           intrinsics of the phone, one file per camera mode
        real/targets/               clinical targets, mandible frame, rig calibration of the spheres
        real/<phase>/               trial_log.csv, trial_log_edits.csv, file_manifest.csv, ground_truth.csv
        real/<phase>/<trial>/       the raw files as filed
        real/<phase>/ground_truth/<trial>.json
        real/ground_truth/<run>/    the scans of the rig calibration and of the repeatability, with their results
        synthetic/                  index.json, index.csv, meshes/, drapes/, sessions/sNNNN/

Local paths are never packed: a JSON value that is an absolute path keeps only its last part.

    python dataset_pack.py inventory [--raw DIR] [--protocol DIR] [--processed DIR] [--synth DIR]
    python dataset_pack.py check     [same options]
    python dataset_pack.py export OUT [--zip] [--no-real] [--no-synthetic] [same options]
    python dataset_pack.py test [--work DIR]     a small dataset made here, checked, spoiled and packed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]
sys.path.insert(0, str(SRC))

import trials  # noqa: E402

WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
SYNTH = WORK / "synth"
PROCESSED = ROOT / "03_data" / "processed"
TARGETS = PROCESSED / "targets"
DEFAULT_OUT = WORK / "dataset"
VERIFIER = SRC / "dataset_verify.py"

FORMAT = 1                      # the layout of the package; one more at each change of it
NAME = "impactguide_dataset"
AUTHOR = "Federico Salerno"
AFFILIATION = "Politecnico di Torino"
CODE_URL = "https://github.com/federicosalerno-phd/ImpactGuide"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"
PHASES = ("campaign", "pilot")
CAMERA_MODES = ("photo", "video", "stream")
CHECK_FROM = 900                # synthetic sessions from here on are the check set of the renderer
TARGET_FILES = ("mandible_targets.csv", "mandible_frame.json", "rig_calibration.json")
GT_STORE = "ground_truth"       # scan_desk.STORE: processed/<it>/<phase>/<trial>.json; raw/<it>/<run>/ the runs' scans
GT_RUNS = ("calibration", "repeat")     # the runs of scans without an impact (scan_desk.CAPTURES)
TRUTH_STATES = ("ready", "partial")     # a trial's result that goes into the package (partial: some views)
STORED = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".npz", ".zip", ".glb", ".gz"}   # already compressed
SKIP = {"desktop.ini", "thumbs.db"}
BLOCK = 1 << 20
LIST_MAX = 3                    # names kept for each kind of problem
PART = ".part"
ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?:Users|home|mnt|tmp)/)")
MAGIC = {".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",), ".png": (b"\x89PNG\r\n\x1a\n",)}
# how bad each problem is: err spoils the dataset, warn is to know
SEVERITY = {
    "missing": "err", "duplicate": "err", "empty": "err", "unreadable": "err", "changed": "err",
    "frames_missing": "err", "bad_image": "err", "no_index": "err", "no_session": "err", "log_corrupt": "err",
    "content": "warn", "unexpected": "warn", "unlisted": "warn", "log": "warn", "online": "warn",
    "bad_field": "warn", "open": "warn", "orphan": "warn", "dup_id": "warn", "no_truth": "warn",
    "truth_stale": "warn", "no_calibration": "warn", "no_rig": "warn", "scene_old": "warn",
    "scene_unknown": "warn", "index_stale": "warn",
}
ORDER = list(SEVERITY)


class Stopped(Exception):
    """The operator asked a check or an export to stop."""


class Job:
    """A check or an export on its way: what it has done of what it has to, and the request to stop it. Read from
    another thread (progress()) while it runs."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.stage = ""
        self.files = self.files_done = 0
        self.bytes = self.bytes_done = 0
        self.started = time.time()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def tick(self, n_bytes: int = 0, files: int = 0) -> None:
        """Counts what was done; raises Stopped once a stop was asked."""
        if self._stop.is_set():
            raise Stopped()
        self.bytes_done += n_bytes
        self.files_done += files

    def progress(self) -> dict:
        done = self.bytes_done / self.bytes if self.bytes else (self.files_done / self.files if self.files else 0.0)
        return {"kind": self.kind, "stage": self.stage, "files": self.files, "files_done": self.files_done,
                "bytes": self.bytes, "bytes_done": self.bytes_done, "fraction": round(min(max(done, 0.0), 1.0), 4),
                "seconds": round(time.time() - self.started, 1), "stopping": self.stopped}


class Problems:
    """What is wrong, grouped by kind: how many, and the first names of each."""

    def __init__(self) -> None:
        self._by: dict[str, dict] = {}

    def add(self, code: str, name: str = "") -> None:
        p = self._by.setdefault(code, {"code": code, "level": SEVERITY.get(code, "warn"), "n": 0, "names": []})
        p["n"] += 1
        if name and len(p["names"]) < LIST_MAX and name not in p["names"]:
            p["names"].append(str(name))

    def extend(self, items: list[dict]) -> None:
        for p in items:
            for _ in range(int(p.get("n", 1))):
                self.add(p["code"])
            got = self._by[p["code"]]["names"]
            for n in p.get("names", []):
                if len(got) < LIST_MAX and n not in got:
                    got.append(n)

    def list(self) -> list[dict]:
        return sorted(self._by.values(), key=lambda p: (p["level"] != "err", ORDER.index(p["code"])
                                                        if p["code"] in ORDER else len(ORDER)))


def level(problems: list[dict]) -> str:
    """ok, warn or err: the worst of the problems."""
    if any(p["level"] == "err" for p in problems):
        return "err"
    return "warn" if problems else "ok"


class Sources:
    """Where the parts of the dataset are: the raw trials, the protocol, the processed data (calibration, ground
    truth), the targets, the rig calibration and the synthetic dataset. The project's own by default; the app gives
    its simulation's."""

    def __init__(self, raw=None, protocol=None, processed=None, synth=None, targets=None, rig=None) -> None:
        self.raw = Path(raw or trials.RAW)
        self.protocol = Path(protocol or trials.PROTOCOL)
        self.processed = Path(processed or PROCESSED)
        self.synth = Path(synth or SYNTH)
        own = self.processed / "targets"
        self.targets = Path(targets) if targets else (own if own.is_dir() else TARGETS)
        self.calibration = self.processed / "calibration"
        self.rig = Path(rig) if rig else self.targets / "rig_calibration.json"


# ── reading ─────────────────────────────────────────────────────────────────
def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def runs(src: Sources) -> list[trials.Run]:
    """The phases with a trial log."""
    return [trials.Run(p, src.raw, src.protocol) for p in PHASES if (src.raw / p / trials.LOG_NAME).is_file()]


def kind_of(name: str) -> str:
    """photo, drape_photo, video, drape_video, highspeed, scan or ref_scan, from a file name."""
    stem = Path(name).stem.lower()
    k = trials.file_kind(name) or "other"
    if k == "scan" and "_ref_scan" in stem:
        return "ref_scan"
    if k in ("photo", "video") and "_drape_" in stem:
        return "drape_" + k
    return k


def trial_files(run: trials.Run, trial: str) -> list[Path]:
    """The files the steps of a trial file, those that are there (the first of each name)."""
    out = []
    for st in run.steps(trial):
        for n in st.files:
            hit = run.find(trial, n)
            if hit:
                out.append(hit[0])
    return out


def manifest_of(run: trials.Run) -> dict[tuple[str, str], dict]:
    return {(m["trial"], m["file"].lower()): m for m in run.log.manifest()}


class Truth:
    """The ground truth of the trials as the Scans page measures it (scan_desk.py): its store, processed/ground_truth/
    <phase>/<trial>.json, one result per trial (dT in the mandible frame, rotation, shifts, the error at each target,
    the registrations of before and after), read here, never measured here: a trial without one is said.
    info: the twin's model extras (twin.Watcher.info) for the board's centre and the cap, or None."""

    def __init__(self, src: Sources, info=None) -> None:
        self.src = src
        self.info = info if callable(info) else (lambda: info)
        self._desks: dict = {}

    def desk(self, run: trials.Run):
        import scan_desk
        d = self._desks.get(run.phase)
        if d is None or d.run is not run:
            d = self._desks[run.phase] = scan_desk.Desk(run, self.src.processed / scan_desk.STORE, self.src.rig,
                                                        self.info)
        return d

    def data(self, run: trials.Run, trial: str) -> dict:
        return self.desk(run).load(trial)

    def state(self, run: trials.Run, trial: str) -> str:
        """ready, partial (some of the scans after the impact), failed, measuring, waiting or none."""
        return str((self.data(run, trial).get("result") or {}).get("state") or "none")

    def views(self, run: trials.Run, trial: str) -> set:
        """The SHA-256 of the scans its result was measured from."""
        return set((self.data(run, trial).get("views") or {}).keys())

    def poses(self, run: trials.Run) -> list[dict]:
        """One row per measured trial, as gt_poses.csv has them (scan_desk.Desk.poses_rows)."""
        return self.desk(run).poses_rows()

    def columns(self, run: trials.Run) -> list[str]:
        tre = [f"tre_{t['id']}_mm" for t in self.desk(run).targets()]
        return (["trial", "phase", "session", "holder", "angle_deg", "direction_deg", "drape", "status", "pre_scan",
                 "state", "views", "spheres_ok", "calibrated", "rot_deg", "shift_mm", "cap_shift_mm", "tre_max_mm",
                 "tre_mean_mm"] + tre + ["T_M_mandible_pre", "T_M_mandible_post", "T_M_B_pre", "T_M_B_post",
                                         "dT_mandible", "dT_B", "warnings", "measured"])


def _matrix(text) -> list | None:
    try:
        v = [float(x) for x in str(text).split()]
    except ValueError:
        return None
    return np.asarray(v).reshape(4, 4).round(6).tolist() if len(v) == 16 else None


def truth_record(run: trials.Run, pose: dict, data: dict) -> dict:
    """The ground truth of a trial as the package holds it, from its row of gt_poses and its result."""
    res = data.get("result") or {}
    views = [{"file": v.get("file", ""), "spheres": v.get("n_spheres"), "calibrated": v.get("calibrated"),
              "arm_deg": (v.get("arm") or {}).get("arm"),
              "sphere_fits": {n: {k: x.get(k) for k in ("radius", "rms_mm", "coverage", "n_points", "centre")}
                              for n, x in (v.get("spheres") or {}).items()}}
             for v in (data.get("views") or {}).values() if v.get("ok") is not None]
    return {"trial": pose["trial"], "phase": run.phase,
            **{k: pose.get(k, "") for k in ("session", "holder", "angle_deg", "direction_deg", "drape", "pre_scan")},
            "state": res.get("state", ""), "frame": "mandible", "units": "mm, degrees",
            "dT": res.get("dT"), "dT_B": _matrix(pose.get("dT_B")),
            "T_M_mandible_before": _matrix(pose.get("T_M_mandible_pre")),
            "T_M_mandible_after": _matrix(pose.get("T_M_mandible_post")),
            **{k: res.get(k) for k in ("rot_deg", "trans_mm", "shift_mm", "cap_shift_mm", "tre", "tre_max_mm",
                                       "tre_mean_mm", "calibrated", "warnings", "reference", "measured")},
            "scans": sorted(views, key=lambda v: v["file"])}


# ── what there is ───────────────────────────────────────────────────────────
def real_inventory(src: Sources, truth=None) -> dict:
    """The real trials, from the logs and the trial folders (no file read): per phase the trials by status, the
    files by kind, their bytes, the sessions (days), the ground truth ready; the problems of the completeness
    check (trials.py), open trials, trials without ground truth, a missing calibration."""
    probs = Problems()
    phases, total = [], Counter()
    for run in runs(src):
        try:
            rows = run.log.rows()
            rep = run.check(deep=False)
        except trials.TrialsError as e:
            probs.add("log_corrupt", run.phase)
            phases.append({"phase": run.phase, "error": getattr(e, "code", "log_corrupt")})
            continue
        status = {r["trial"]: r["status"] for r in rows}
        for t in rep["trials"]:
            if t["status"] == trials.VOID:
                continue
            for kind, item in t["problems"]:
                probs.add(kind if kind in SEVERITY else "content", f"{item}" if kind not in ("log", "online",
                                                                                           "bad_field")
                          else f"{t['trial']} {item}")
        for t in rep["open"]:
            probs.add("open", t)
        for t in rep["orphans"]:
            probs.add("orphan", t)
        for t in rep["duplicates"]:
            probs.add("dup_id", t)
        ok = [r for r in rows if r["status"] == trials.OK]
        files, nbytes = Counter(), 0
        for r in ok:
            for p in trial_files(run, r["trial"]):
                files[kind_of(p.name)] += 1
                nbytes += p.stat().st_size
        scanned = [r for r in ok if any(kind_of(p.name) == "scan" for p in trial_files(run, r["trial"]))]
        truth = truth or Truth(src)
        ready = 0
        for r in scanned:
            if truth.state(run, r["trial"]) == "ready":
                ready += 1
            else:
                probs.add("no_truth", r["trial"])
        days = Counter(r["session"] for r in ok if r.get("session"))
        cfg = read_json(run.config_path) or {}
        ph = {"phase": run.phase, "trials": len(rows), "ok": len(ok),
              "open": sum(s == trials.OPEN for s in status.values()),
              "void": sum(s == trials.VOID for s in status.values()),
              "files": dict(files), "bytes": nbytes, "scanned": len(scanned), "truth": ready,
              "sessions": [{"session": d, "trials": n} for d, n in sorted(days.items())],
              "config_written": cfg.get("written", ""), "pending": len(rep["pending"])}
        phases.append(ph)
        total.update({"trials": len(ok), "bytes": nbytes, "scanned": len(scanned), "truth": ready,
                      "photos": files["photo"] + files["drape_photo"],
                      "videos": files["video"] + files["drape_video"] + files["highspeed"],
                      "scans": files["scan"] + files["ref_scan"]})
    calib = {m: (src.calibration / f"intrinsics_{m}.json").is_file() for m in CAMERA_MODES}
    rig = src.rig.is_file()
    if total["trials"] and not calib["photo"]:
        probs.add("no_calibration", "photo")
    if total["scanned"] and not rig:
        probs.add("no_rig")
    problems = probs.list()
    return {"present": bool(total["trials"]), "phases": phases,
            **{k: int(total[k]) for k in ("trials", "photos", "videos", "scans", "scanned", "truth", "bytes")},
            "calibration": calib, "rig": rig, "problems": problems, "level": level(problems)}


def synth_sessions(root: Path) -> tuple[list[Path], list[Path]]:
    """The sessions of the dataset and those of the check set, each with its frames.jsonl."""
    found = sorted(p for p in (Path(root) / "sessions").glob("s[0-9][0-9][0-9][0-9]") if (p / "frames.jsonl").is_file())
    return [p for p in found if int(p.name[1:]) < CHECK_FROM], [p for p in found if int(p.name[1:]) >= CHECK_FROM]


def read_frames(session: Path) -> list[dict]:
    out = []
    for line in (session / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({})
    return out


def scene_state(scene_print: str, model_print: str) -> str:
    """current (rendered on the scene of the twin's model), old (on another), unknown (no fingerprint recorded),
    no_model (no model of the twin to compare with)."""
    if not scene_print:
        return "unknown"
    if not model_print:
        return "no_model"
    return "current" if scene_print == model_print else "old"


def synth_inventory(root: Path, model_print: str = "") -> dict:
    """The synthetic dataset from its index and its folders (only the frames.jsonl read): sessions, samples,
    images, splits, bytes, the scene it was rendered on (fingerprint, Blender, date) against the twin's model;
    the problems: no index, an index older than the sessions, a session without its session.json."""
    root = Path(root)
    idx = read_json(root / "index.json")
    sessions, check = synth_sessions(root)
    probs = Problems()
    if not sessions:
        return {"present": False, "root": str(root), "problems": [], "level": "ok"}
    if not isinstance(idx, dict):
        probs.add("no_index")
        idx = {}
    frames_n, blender, rendered, prints, nbytes, labels = 0, set(), [], set(), 0, {}
    for s in sessions:
        frames_n += len(read_frames(s))
        info = read_json(s / "session.json")
        if not isinstance(info, dict):
            probs.add("no_session", s.name)
            continue
        if info.get("blender"):
            blender.add(str(info["blender"]))
        if info.get("scene_mtime"):
            rendered.append(str(info["scene_mtime"]))
        prints.add(str(info.get("scene_print") or ""))
        labels = labels or {str(k): v for k, v in (info.get("labels") or {}).items() if isinstance(v, int)}
        nbytes += sum(e.stat().st_size for e in os.scandir(s) if e.is_file() and e.name.lower() not in SKIP)
    for sub in ("meshes", "drapes"):
        d = root / sub
        nbytes += sum(p.stat().st_size for p in d.glob("*") if p.is_file() and p.name.lower() not in SKIP) \
            if d.is_dir() else 0
    if idx and (idx.get("sessions") != len(sessions) or idx.get("images") != frames_n):
        probs.add("index_stale")
    sp = str(idx.get("scene_print") or "")
    state = scene_state(sp, model_print)
    if state == "old":
        probs.add("scene_old")
    elif state == "unknown":
        probs.add("scene_unknown")
    problems = probs.list()
    splits = idx.get("splits") or {}
    return {"present": True, "root": str(root), "sessions": len(sessions), "check_sessions": len(check),
            "samples": idx.get("samples"), "images": frames_n, "photos": idx.get("photos"),
            "video_frames": idx.get("video_frames"), "splits": splits, "bytes": nbytes, "labels": labels,
            "scene": {"print": sp, "model_print": model_print, "state": state, "blender": sorted(blender),
                      "rendered": [min(rendered), max(rendered)] if rendered else []},
            "problems": problems, "level": level(problems)}


def inventory(src: Sources, model_print: str = "", truth=None) -> dict:
    return {"real": real_inventory(src, truth), "synthetic": synth_inventory(src.synth, model_print),
            "time": datetime.now().astimezone().isoformat(timespec="seconds")}


# ── reading every file ─────────────────────────────────────────────────────
def sha256_of(path: Path, job: Job | None = None) -> tuple[int, str]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
            n += len(block)
            if job:
                job.tick(len(block))
    return n, h.hexdigest()


def image_ok(path: Path) -> bool:
    """A photo or label image that is whole at its start (its signature) and not empty."""
    want = MAGIC.get(path.suffix.lower())
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        return bool(head) and (not want or any(head.startswith(m) for m in want))
    except OSError:
        return False


def frame_files(session: Path, frame: dict) -> list[Path]:
    stem = str(frame.get("image") or "")
    return [session / f"{stem}.jpg", session / f"{stem}_label.png"] if stem else []


def check(src: Sources, job: Job, parts=("real", "synthetic"), model_print: str = "", truth=None) -> dict:
    """Every file read: the trials' files against the manifest of their phase (bytes and SHA-256: changed, or
    unlisted when the manifest does not hold them), the scans of a trial with ground truth against the scans it was
    measured from (truth_stale), every synthetic frame against its photo and label image (there and whole). With the
    problems of the inventory."""
    truth = truth or Truth(src)
    inv = inventory(src, model_print, truth)
    real, synth = Problems(), Problems()
    plan = []
    if "real" in parts:
        real.extend(inv["real"]["problems"])
        for run in runs(src):
            try:
                man = manifest_of(run)
                ok = [r for r in run.log.rows() if r["status"] == trials.OK]
            except trials.TrialsError:
                continue
            for r in ok:
                measured = truth.views(run, r["trial"]) if truth.state(run, r["trial"]) == "ready" else None
                for p in trial_files(run, r["trial"]):
                    plan.append((measured if kind_of(p.name) in ("scan", "ref_scan") else None, p,
                                 man.get((r["trial"], p.name.lower()))))
    sessions = synth_sessions(src.synth)[0] if "synthetic" in parts else []
    if "synthetic" in parts:
        synth.extend(inv["synthetic"]["problems"])
    job.stage = "real"
    job.bytes = sum(p.stat().st_size for _, p, _ in plan)
    job.files = len(plan) + len(sessions)
    for measured, p, m in plan:
        try:
            n, digest = sha256_of(p, job)
        except OSError:
            real.add("unreadable", p.name)
            job.tick(files=1)
            continue
        if m is None:
            real.add("unlisted", p.name)
        elif str(m.get("sha256", "")).lower() != digest or str(m.get("bytes", "")) not in ("", str(n)):
            real.add("changed", p.name)
        if measured is not None and digest not in measured:
            real.add("truth_stale", p.name)
        job.tick(files=1)
    job.stage = "synthetic"
    for s in sessions:
        bad = 0
        for f in read_frames(s):
            files = frame_files(s, f)
            if not files:
                bad += 1
                continue
            for p in files:
                if not p.is_file():
                    bad += 1
                elif not image_ok(p):
                    synth.add("bad_image", f"{s.name}/{p.name}")
        if bad:
            synth.add("frames_missing", s.name)
        job.tick(files=1)
    out = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "parts": list(parts),
           "files": job.files_done, "bytes": job.bytes_done,
           "real": {"problems": real.list()}, "synthetic": {"problems": synth.list()}}
    for k in ("real", "synthetic"):
        out[k]["level"] = level(out[k]["problems"])
    out["level"] = "err" if "err" in (out["real"]["level"], out["synthetic"]["level"]) else \
        ("warn" if "warn" in (out["real"]["level"], out["synthetic"]["level"]) else "ok")
    return out


# ── the package ─────────────────────────────────────────────────────────────
def clean_json(value):
    """A JSON value with every absolute path cut to its last part (no local folder leaves this computer)."""
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, str) and ABS_PATH.match(value):
        return re.split(r"[\\/]", value.rstrip("\\/"))[-1]
    return value


class _Writer:
    """Files into the package being written (a folder, or a zip), each with its bytes and SHA-256."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self.rows: list[dict] = []

    def _row(self, rel: str, n: int, digest: str) -> tuple[int, str]:
        self.rows.append({"path": rel, "bytes": n, "sha256": digest})
        return n, digest

    def add_bytes(self, rel: str, data: bytes, listed: bool = True) -> tuple[int, str]:
        self._put(rel, io.BytesIO(data), len(data), time.time())
        digest = hashlib.sha256(data).hexdigest()
        return self._row(rel, len(data), digest) if listed else (len(data), digest)

    def add_file(self, src: Path, rel: str) -> tuple[int, str]:
        st = src.stat()
        h = hashlib.sha256()
        with open(src, "rb") as f:
            n = self._put(rel, f, st.st_size, st.st_mtime, h)
        self.job.tick(files=1)
        return self._row(rel, n, h.hexdigest())


class FolderWriter(_Writer):
    def __init__(self, folder: Path, job: Job) -> None:
        super().__init__(job)
        self.folder = folder
        folder.mkdir(parents=True)

    def _put(self, rel, f, size, mtime, h=None) -> int:
        out = self.folder / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out, "xb") as fo:
            for block in iter(lambda: f.read(BLOCK), b""):
                fo.write(block)
                n += len(block)
                if h is not None:
                    h.update(block)
                    self.job.tick(len(block))
        os.utime(out, (mtime, mtime))
        return n

    def close(self) -> None:
        pass


class ZipWriter(_Writer):
    def __init__(self, path: Path, top: str, job: Job) -> None:
        super().__init__(job)
        self.top = top
        self.zip = zipfile.ZipFile(path, "x", allowZip64=True)

    def _put(self, rel, f, size, mtime, h=None) -> int:
        t = time.localtime(max(mtime, 315532800))      # zip times start in 1980
        zi = zipfile.ZipInfo(f"{self.top}/{rel}", date_time=t[:6])
        zi.compress_type = zipfile.ZIP_STORED if Path(rel).suffix.lower() in STORED else zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        n = 0
        with self.zip.open(zi, "w", force_zip64=size > (1 << 31)) as fo:
            for block in iter(lambda: f.read(BLOCK), b""):
                fo.write(block)
                n += len(block)
                if h is not None:
                    h.update(block)
                    self.job.tick(len(block))
        return n

    def close(self) -> None:
        self.zip.close()


def package_name(day: datetime | None = None) -> str:
    return f"{NAME}_{(day or datetime.now()).strftime('%Y%m%d')}"


def free_target(folder: Path, name: str, suffix: str) -> Path:
    """folder/name+suffix, or name_2, name_3... when it is taken (with its .part)."""
    for i in range(1, 1000):
        p = Path(folder) / f"{name}{'' if i == 1 else f'_{i}'}{suffix}"
        if not p.exists() and not p.with_name(p.name + PART).exists():
            return p
    raise FileExistsError(name)


def _plan_real(src: Sources) -> tuple[list[tuple[Path, str, dict | None, bool]], list[tuple[trials.Run, dict]]]:
    """(source, path in the package, manifest row, a raw file of a trial) of every real file, and the closed
    trials; nothing at all without a closed trial."""
    plan, closed = [], []
    if not any(r["status"] == trials.OK for run in runs(src) for r in run.log.rows()):
        return plan, closed
    for name in ("trial_log_dictionary.csv",):
        if (src.protocol / name).is_file():
            plan.append((src.protocol / name, f"real/protocol/{name}", None, False))
    for m in CAMERA_MODES:
        p = src.calibration / f"intrinsics_{m}.json"
        if p.is_file():
            plan.append((p, f"real/calibration/{p.name}", None, False))
    for name in TARGET_FILES:
        if (src.targets / name).is_file():
            plan.append((src.targets / name, f"real/targets/{name}", None, False))
    for run in runs(src):
        for p in (run.config_path, run.sequence_path):
            if p.is_file():
                plan.append((p, f"real/protocol/{p.name}", None, False))
        for name in (trials.LOG_NAME, trials.EDITS_NAME, trials.MANIFEST_NAME):
            if (run.folder / name).is_file():
                plan.append((run.folder / name, f"real/{run.phase}/{name}", None, False))
        man = manifest_of(run)
        for r in run.log.rows():
            if r["status"] != trials.OK:
                continue
            closed.append((run, r))
            for p in trial_files(run, r["trial"]):
                plan.append((p, f"real/{run.phase}/{r['trial']}/{p.name}", man.get((r["trial"], p.name.lower())), True))
    # the runs without an impact: the scans of the rig calibration and of the repeatability, and what they gave
    store = src.processed / GT_STORE
    for kind in GT_RUNS:
        for p in sorted((src.raw / GT_STORE / kind).glob("*")) if (src.raw / GT_STORE / kind).is_dir() else ():
            if p.is_file() and p.name.lower() not in SKIP and not p.name.endswith(PART):
                plan.append((p, f"real/{GT_STORE}/{kind}/{p.name}", None, False))
        if (store / f"{kind}.json").is_file():
            plan.append((store / f"{kind}.json", f"real/{GT_STORE}/{kind}/{kind}.json", None, False))
    if (store / "repeatability.csv").is_file():
        plan.append((store / "repeatability.csv", f"real/{GT_STORE}/repeat/repeatability.csv", None, False))
    return plan, closed


def _plan_synth(root: Path) -> list[tuple[Path, str, None, bool]]:
    plan = []
    if not synth_sessions(root)[0]:
        return plan
    for name in ("index.json", "index.csv"):
        if (root / name).is_file():
            plan.append((root / name, f"synthetic/{name}", None, False))
    for sub in ("meshes", "drapes"):
        for p in sorted((root / sub).glob("*")) if (root / sub).is_dir() else ():
            if p.is_file() and p.name.lower() not in SKIP:
                plan.append((p, f"synthetic/{sub}/{p.name}", None, False))
    for s in synth_sessions(root)[0]:
        for p in sorted(s.iterdir()):
            if p.is_file() and p.name.lower() not in SKIP and not p.name.endswith((PART, ".tmp")):
                plan.append((p, f"synthetic/sessions/{s.name}/{p.name}", None, False))
    return plan


def export(src: Sources, dest: Path, job: Job, as_zip: bool = False, parts=("real", "synthetic"),
           model_print: str = "", truth=None, made_by: str = "ImpactGuide", simulated: bool = False) -> dict:
    """The package written into dest (a folder for its folder, or the zip file's path), beside its place as .part
    and named last. Returns where it is, its files and bytes, and the problems found while packing. Stopped (and
    nothing left) when the job is stopped; FileExistsError when dest exists; ValueError("empty") when there is
    nothing to pack. simulated: the trials are a simulation's (said in the README and the index)."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    truth = truth or Truth(src)
    inv = inventory(src, model_print, truth)
    probs = Problems()
    plan, closed = _plan_real(src) if "real" in parts else ([], [])
    if "real" in parts:
        probs.extend(inv["real"]["problems"])
    if "synthetic" in parts:
        plan += _plan_synth(src.synth)
        probs.extend(inv["synthetic"]["problems"])
    if not plan:
        raise ValueError("empty")
    job.files = len(plan)
    job.bytes = sum(p.stat().st_size for p, _, _, _ in plan)
    part = dest.with_name(dest.name + PART)
    top = dest.stem if as_zip else dest.name
    w = ZipWriter(part, top, job) if as_zip else FolderWriter(part, job)
    try:
        job.stage = "files"
        for p, rel, m, raw_file in plan:
            if p.suffix.lower() == ".json" and rel.startswith("real/"):
                d = read_json(p)
                data = (json.dumps(clean_json(d), indent=1, ensure_ascii=False) + "\n").encode("utf-8") \
                    if d is not None else p.read_bytes()
                w.add_bytes(rel, data)
                job.tick(p.stat().st_size, files=1)
                continue
            n, digest = w.add_file(p, rel)
            if m is not None and (str(m.get("sha256", "")).lower() != digest or str(m.get("bytes", "")) not in
                                  ("", str(n))):
                probs.add("changed", p.name)
            elif m is None and raw_file:        # a trial's raw file the manifest does not hold
                probs.add("unlisted", p.name)
        job.stage = "truth"
        truths = {}
        by_phase = {}
        for run, row in closed:
            by_phase.setdefault(run.phase, (run, set()))[1].add(row["trial"])
        for phase, (run, ids) in by_phase.items():
            job.tick()
            rows = [r for r in truth.poses(run) if r["trial"] in ids and r.get("state") in TRUTH_STATES]
            for r in rows:
                rec = clean_json(truth_record(run, r, truth.data(run, r["trial"])))
                truths.setdefault(phase, []).append(rec)
                w.add_bytes(f"real/{phase}/ground_truth/{r['trial']}.json",
                            (json.dumps(rec, indent=1) + "\n").encode("utf-8"))
            if rows:
                w.add_bytes(f"real/{phase}/ground_truth.csv", truth_csv(rows, truth.columns(run)))
        job.stage = "index"
        summary = package_summary(inv, parts, truths, made_by, simulated)
        summary["problems"] = probs.list()
        summary["level"] = level(summary["problems"])
        w.add_bytes("README.md", readme(summary).encode("utf-8"))
        w.add_bytes("LICENSE.txt", license_text().encode("utf-8"))
        if VERIFIER.is_file():
            w.add_bytes("verify.py", VERIFIER.read_bytes())
        w.add_bytes("index.json", (json.dumps(summary, indent=1) + "\n").encode("utf-8"))
        rows = list(w.rows)
        buf = io.StringIO()
        cw = csv.DictWriter(buf, ("path", "bytes", "sha256"), lineterminator="\n")
        cw.writeheader()
        cw.writerows(rows)
        w.add_bytes("files.csv", buf.getvalue().encode("utf-8"), listed=False)
        w.close()
        os.replace(part, dest)
    except BaseException:
        try:
            w.close()
        except Exception:      # noqa: BLE001  the partial package goes in any case
            pass
        if part.is_dir():
            shutil.rmtree(part, ignore_errors=True)
        elif part.exists():
            part.unlink()
        raise
    total = sum(r["bytes"] for r in rows)
    return {"path": str(dest), "zip": as_zip, "files": len(rows) + 1, "bytes": total,
            "problems": summary["problems"], "level": summary["level"],
            "time": datetime.now().astimezone().isoformat(timespec="seconds")}


def truth_csv(rows: list[dict], cols: list[str]) -> bytes:
    """gt_poses.csv of the closed trials with ground truth (scan_desk.Desk.write_poses' columns)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, cols, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows({k: "" if v is None else v for k, v in r.items()} for r in rows)
    return buf.getvalue().encode("utf-8")


def package_summary(inv: dict, parts, truths: dict, made_by: str, simulated: bool = False) -> dict:
    real, synth = inv["real"], inv["synthetic"]
    out = {"name": NAME, "format": FORMAT, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
           "made_by": made_by, "author": AUTHOR, "affiliation": AFFILIATION, "code": CODE_URL,
           "license": {"data_and_models": "CC-BY-4.0", "code": "MIT"}, "parts": list(parts),
           "simulated": bool(simulated)}
    out["parts"] = [p for p in parts if (real if p == "real" else synth).get("present")]
    if "real" in out["parts"]:
        out["real"] = {k: real.get(k) for k in ("trials", "photos", "videos", "scans", "scanned", "calibration",
                                                "rig")}
        out["real"]["truth"] = sum(len(v) for v in truths.values())
        out["real"]["phases"] = [{k: p.get(k) for k in ("phase", "ok", "void", "files", "sessions",
                                                        "config_written")} for p in real["phases"]]
    if "synthetic" in parts and synth.get("present"):
        out["synthetic"] = {k: synth.get(k) for k in ("sessions", "samples", "images", "photos", "video_frames",
                                                      "splits", "labels")}
        sc = synth["scene"]
        out["synthetic"]["scene"] = {"fingerprint": sc["print"], "state": sc["state"], "blender": sc["blender"],
                                     "rendered": sc["rendered"]}
    return out


# ── the words of the package ────────────────────────────────────────────────
def _n(v) -> str:
    return "-" if v is None else f"{v:,}".replace(",", " ")


def readme(s: dict) -> str:
    """README.md of the package, from its summary."""
    L = ["# ImpactGuide bench dataset", "",
         "Impact trials on a phantom head: a ChArUco marker fixed to the teeth through a dental splint is struck by "
         "a pendulum, and the trials record how far it moved. Each trial holds phone photos and a video of the face "
         "and the marker, structured light scans of six reference spheres (three on the marker holder, three on a "
         "frame fixed to the mandible), and the ground truth of the marker's motion measured from those scans. "
         "With them comes a synthetic dataset rendered from the digital twin of the bench.", "",
         f"Created {s['created'][:10]} with {s['made_by']}. Author: {s['author']}, {s['affiliation']}. "
         f"Code of the bench and of the app: {s['code']}", ""]
    L += ["## Contents", "", "| Part | What | Count |", "|---|---|---|"]
    r = s.get("real")
    if r:
        L += [f"| Real | closed trials | {_n(r['trials'])} |", f"| Real | photos | {_n(r['photos'])} |",
              f"| Real | videos | {_n(r['videos'])} |", f"| Real | scans | {_n(r['scans'])} |",
              f"| Real | trials with ground truth | {_n(r['truth'])} of {_n(r['scanned'])} scanned |"]
    y = s.get("synthetic")
    if y:
        sp = y.get("splits") or {}
        L += [f"| Synthetic | sessions | {_n(y['sessions'])} |", f"| Synthetic | samples | {_n(y['samples'])} |",
              f"| Synthetic | images | {_n(y['images'])} ({_n(y['photos'])} photos, {_n(y['video_frames'])} "
              f"video frames) |"]
        if sp:
            L.append("| Synthetic | sessions per split | " + ", ".join(
                f"{k} {_n(v.get('sessions'))}" for k, v in sp.items()) + " |")
    L += ["", "## Layout", "", "```",
          "README.md                 this file",
          "LICENSE.txt               data and models CC BY 4.0, code MIT",
          "index.json                counts, versions and the problems found while packing",
          "files.csv                 every file with its bytes and SHA-256",
          "verify.py                 checks every file against files.csv"]
    if r:
        L += ["real/protocol/            configuration and sequence of each phase, data dictionary of the log",
              "real/calibration/         phone intrinsics, one file per camera mode (photo, video, stream)",
              "real/targets/             clinical targets, mandible frame, rig calibration of the spheres",
              "real/<phase>/             trial_log.csv, trial_log_edits.csv, file_manifest.csv, ground_truth.csv",
              "real/<phase>/<trial>/     the raw files of a closed trial",
              "real/<phase>/ground_truth/<trial>.json   the ground truth of a trial",
              "real/ground_truth/calibration/          scans of the rig calibration (no impact), calibration.json",
              "real/ground_truth/repeat/               scans of the repeatability run, repeat.json, repeatability.csv"]
    if y:
        L += ["synthetic/                index.json, index.csv (one row per image)",
              "synthetic/meshes/         face meshes and the occluders of the bench (NumPy .npz)",
              "synthetic/drapes/         drape meshes (NumPy .npz)",
              "synthetic/sessions/sNNNN/ session.json, frames.jsonl, <image>.jpg, <image>_label.png"]
    L += ["```", ""]
    if r:
        L += ["## Real trials", "",
              "The phases are `campaign` (trials t001, t002...) and `pilot` (p001...). Every file name starts with "
              "the trial id and carries no condition: the conditions are in `trial_log.csv` (columns described in "
              "`real/protocol/trial_log_dictionary.csv`), and every later correction of a logged value is in "
              "`trial_log_edits.csv` with the old and new value, the time and the reason. `file_manifest.csv` holds "
              "the bytes and the SHA-256 of each raw file as it was filed. Only closed trials are included; void "
              "trials stay in the log with their reason.", "",
              "| File | What |", "|---|---|",
              "| `<trial>_photo_azNN.jpg` | photo from arm position NN (degrees) |",
              "| `<trial>_video.mp4` | video taken while the arm sweeps between its end stops |",
              "| `<trial>_drape_photo_azNN.jpg`, `<trial>_drape_video.mp4` | the same with the drape on (drape trials) |",
              "| `<trial>_scan_azNN.ply` | scan after the impact, arm at NN degrees |",
              "| `<trial>_ref_scan_azNN.ply` | reference scan before the impact, when one was taken |",
              "| `<trial>_highspeed.mp4` | side video at 200 fps or more (pilot only) |", "",
              "The scans before an impact are the reference scans of the trial (`pre_scan` = `<trial>_ref`) or the "
              "post-impact scans of the trial before (`pre_scan` = that trial).", "",
              "### Ground truth", "",
              "Poses are 4 x 4 matrices in mm. The mandible frame is the local frame of the mandible model: origin "
              "on the hinge axis, x to the patient's left, y down, z backwards. `real/targets/mandible_frame.json` "
              "gives the head frame B of the bench scene from it (`T_B_mandible`). With T_M_mandible the pose of "
              "the mandible frame in the marker frame M (the ChArUco board: origin at the outer corner of its first "
              "square, x along the columns, y along the rows, z into the board), measured from the scans before and "
              "after the impact, `dT` = T_M_mandible(before)^-1 T_M_mandible(after), and the error the registration "
              "made before the impact leaves at a target p (mandible frame) is |p - dT p|. The registrations come "
              "from the six spheres fitted in the scans with the radii of the rig calibration "
              "(`real/targets/rig_calibration.json`; `calibrated` = 0 when the radii were fitted freely). Each "
              "`ground_truth/<trial>.json` holds `state` (ready, or partial when only some of the scans after the "
              "impact were measured), `dT` and the same in frame B (`dT_B`), the registrations of before and after "
              "(`T_M_mandible_before`, `T_M_mandible_after`), the rotation (`rot_deg`), the translation "
              "(`trans_mm`), how far the motion carries the centre of the board (`shift_mm`), the shift of the cap "
              "since the last reference scans (`cap_shift_mm`), the error at each target of `mandible_targets.csv` "
              "(`tre`, mm) with their mean and maximum, and the fit of each sphere in each scan. "
              "`ground_truth.csv` has one row per trial, with the matrices as 16 numbers row by row.", ""]
        if s.get("simulated"):
            L += ["This package was made in simulation: its trials, files and ground truth are simulated.", ""]
        if r.get("calibration"):
            modes = [m for m, ok in r["calibration"].items() if ok]
            L += ["### Camera", "",
                  "`real/calibration/intrinsics_<mode>.json` holds the phone's intrinsics of each calibrated "
                  "camera mode (" + (", ".join(modes) if modes else "none yet")
                  + "): image size, camera matrix `K` and distortion `dist` (OpenCV model k1, k2, p1, p2, k3), "
                  "with their errors.", ""]
    if y:
        sc = y.get("scene") or {}
        L += ["## Synthetic dataset", "",
              "Rendered from the digital twin of the bench in Blender"
              + (f" {', '.join(sc.get('blender') or [])}" if sc.get("blender") else "")
              + ". Each session draws a face deformation, lights, the marker holder and its motions after the "
              "impact; each sample is a state of the marker seen in photos from the arm positions and in video "
              "frames along the sweep. The sessions are split into train, val and test (column `split`), never a "
              "session across two of them.", "",
              "`frames.jsonl` has one line per image: `image` (the stem of `<image>.jpg` and of the label image "
              "`<image>_label.png`), `mode` (photo or video), `size`, `K` and `dist`, the poses `T_C_B` (camera "
              "from head frame B), `T_C_M` (camera from marker), `T_M_B` (marker from head after the motion), "
              "`T_M_B0` (before it) and `dT`, the `motion` (kind, degrees, mm), `target_shift_mm`, the arm and "
              "tilt angles, deformation, drape, lights, the pendulum direction and the pixel count of each label. "
              "`session.json` gives the intrinsics, the label values, the targets and the scene the session was "
              "rendered on." + (" Label values: " + ", ".join(
                  f"{v} {k.replace('_', ' ')}" for k, v in sorted((y.get("labels") or {}).items(),
                                                                key=lambda kv: kv[1])) + "."
                                   if y.get("labels") else ""), ""]
    L += ["## Versions", "", f"- Package format {s['format']}, made by {s['made_by']}."]
    if r:
        for p in r.get("phases") or []:
            if p.get("config_written"):
                L.append(f"- {p['phase']}: configuration written {p['config_written']}.")
    if y:
        sc = y.get("scene") or {}
        state = {"current": "the scene of the twin in use when packed",
                 "old": "an older version of the bench scene than the twin in use when packed",
                 "unknown": "a scene whose fingerprint was not recorded",
                 "no_model": "a scene not compared with the twin (no model when packed)"}.get(sc.get("state"), "")
        L.append(f"- Synthetic: rendered on {state}"
                 + (f" (fingerprint {sc['fingerprint']})" if sc.get("fingerprint") else "")
                 + (f", scene saved {sc['rendered'][0]}" if sc.get("rendered") else "") + ".")
    if s.get("problems"):
        L += ["", "Known problems when packed (see `index.json`): "
              + ", ".join(f"{p['code']} ({p['n']})" for p in s["problems"]) + "."]
    L += ["", "## Checksums", "", "`files.csv` lists every file with its size and SHA-256. To check a copy:", "",
          "```", "python verify.py", "```", "",
          "## Licence", "",
          f"Data and models: Creative Commons Attribution 4.0 International (CC BY 4.0), {CC_BY}. "
          f"Code (`verify.py`): MIT. See `LICENSE.txt`. Please credit {s['author']}, ImpactGuide, {s['code']}.",
          ""]
    return "\n".join(L)


def license_text() -> str:
    year = datetime.now().year
    return f"""Copyright (c) {year} {AUTHOR}

DATA AND MODELS

The data (photos, videos, scans, logs, calibration, ground truth, synthetic
images, labels and poses) and the models (meshes) of this dataset are licensed
under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
The full license text is at {CC_BY}legalcode

Attribution: {AUTHOR}, ImpactGuide bench dataset, {CODE_URL}

CODE

verify.py is licensed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# ── tests ───────────────────────────────────────────────────────────────────
def make_synth(root: Path, sessions=(0, 1, 2, 900), frames: int = 3, scene_print: str = "abcd") -> None:
    """A small synthetic dataset laid out as synth_twin.py writes it, with its index."""
    root = Path(root)
    jpg, png = b"\xff\xd8\xff\xe0" + b"j" * 60, b"\x89PNG\r\n\x1a\n" + b"p" * 60
    n = 0
    for s in sessions:
        d = root / "sessions" / f"s{s:04d}"
        d.mkdir(parents=True, exist_ok=True)
        split = "test" if s == 0 else "train"
        lines = []
        for i in range(frames):
            stem = f"r0_s{i}_p1"
            (d / f"{stem}.jpg").write_bytes(jpg)
            (d / f"{stem}_label.png").write_bytes(png)
            lines.append(json.dumps({"image": stem, "session": s, "split": split, "mode": "photo"}))
        (d / "frames.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (d / "session.json").write_text(json.dumps({"session": s, "blender": "5.2.2 LTS", "scene_print": scene_print,
                                                    "scene_mtime": "2026-10-18 23:25:05"}), encoding="utf-8")
        n += frames if s < CHECK_FROM else 0
    (root / "meshes").mkdir(exist_ok=True)
    (root / "meshes" / "face.npz").write_bytes(b"PK" + b"m" * 30)
    (root / "index.json").write_text(json.dumps({"sessions": sum(s < CHECK_FROM for s in sessions), "samples": n,
                                                 "images": n, "photos": n, "video_frames": 0, "scene_print": scene_print,
                                                 "splits": {"train": {"sessions": 2}, "test": {"sessions": 1}}}),
                                     encoding="utf-8")
    (root / "index.csv").write_text("session,image\n", encoding="utf-8")


def make_trials(raw: Path, protocol: Path, n: int = 2, scans: bool = True) -> trials.Run:
    """n closed trials of the campaign with stand-in files filed as the app files them."""
    run = trials.Run("campaign", raw, protocol)
    src = Path(tempfile.mkdtemp(prefix="ds_src_"))
    try:
        for _ in range(n):
            plan = run.next()
            row = run.start(plan, session="2026-10-26", swing_mass_g="137.2", washers_per_side="3")
            t = row["trial"]
            for st in run.steps(t):
                for name in st.files:
                    if not scans and trials.file_kind(name) == "scan":
                        continue
                    f = src / name
                    f.write_bytes(name.encode() * 8)
                    run.store(t, name, f)
                values = {}
                if st.key == "spheres":
                    values = {"spheres_ok": "1", "cap_shift_mm": "0.5"}
                if st.key == "close":
                    values = {"second_hit": "0", "cap_replaced": "0"}
                if st.key == "release":
                    values = {}
                try:
                    run.confirm(t, st.key, **values)
                except trials.TrialsError:
                    pass
    finally:
        shutil.rmtree(src, ignore_errors=True)
    return run


def fake_truth(run: trials.Run, processed: Path, trial: str, deg: float = 1.5) -> None:
    """A result of the Scans page's store for a trial (scan_desk.py's layout), measured from its scans as they are:
    the registration of before the identity, the one of after turned by deg about x."""
    a = np.radians(deg)
    dT = np.eye(4)
    dT[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    views = {}
    for f in trial_files(run, trial):
        if kind_of(f.name) in ("scan", "ref_scan"):
            sha = sha256_of(f)[1]
            views[sha] = {"sha": sha, "file": f.name, "ok": True, "n_spheres": 6, "calibrated": True,
                          "arm": {"arm": 55},
                          "spheres": {"marker_1": {"radius": 7.5, "rms_mm": 0.01, "coverage": 0.4, "n_points": 900,
                                                   "centre": [0.0, 0.0, 0.0]}}}
    data = {"trial": trial, "views": views,
            "sets": {"ref": {"ok": True, "T_M_mandible": np.eye(4).tolist()},
                     "post": {"ok": True, "T_M_mandible": dT.tolist()}},
            "result": {"state": "ready", "dT": dT.tolist(), "rot_deg": deg, "trans_mm": [0.0, 0.0, 0.0],
                       "shift_mm": 0.8, "cap_shift_mm": 0.5,
                       "tre": [{"id": "Pog", "name": "pogonion", "mm": 1.2}], "tre_max_mm": 1.2, "tre_mean_mm": 1.2,
                       "calibrated": True, "warnings": [], "reference": trial, "measured": "2026-10-26T10:00:00+02:00"}}
    out = processed / GT_STORE / run.phase / f"{trial}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data), encoding="utf-8")


def selftest(work: Path) -> int:
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    raw, protocol, processed, synth = work / "raw", work / "protocol", work / "processed", work / "synth"
    protocol.mkdir()
    run = make_trials(raw, protocol, 2)
    make_synth(synth)
    (processed / "calibration").mkdir(parents=True)
    (processed / "calibration" / "intrinsics_photo.json").write_text(json.dumps(
        {"images": r"C:\Users\someone\calibration\photos", "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}), encoding="utf-8")
    src = Sources(raw, protocol, processed, synth)
    ids = [x["trial"] for x in run.log.rows()]
    truth = Truth(src)
    r = inventory(src, "abcd", truth)["real"]
    expect(r["truth"] == 0 and any(p["code"] == "no_truth" and p["n"] == 2 for p in r["problems"]),
           "inventory: trials scanned but not measured by the Scans page: said, never measured here")
    for t in ids:
        fake_truth(run, processed, t)
    truth = Truth(src)
    inv = inventory(src, "abcd", truth)
    r, y = inv["real"], inv["synthetic"]
    expect(r["trials"] == 2 and r["photos"] == 8 and r["videos"] == 2 and r["scans"] >= 4 and r["truth"] == 2,
           f"inventory: 2 closed trials, {r['photos']} photos, {r['videos']} videos, {r['scans']} scans, ground "
           f"truth for {r['truth']}")
    expect(y["sessions"] == 3 and y["check_sessions"] == 1 and y["images"] == 9 and y["scene"]["state"] == "current"
           and y["level"] == "ok", "inventory: 3 synthetic sessions (the check one apart), 9 images, current scene")
    expect(synth_inventory(synth, "ffff")["scene"]["state"] == "old"
           and any(p["code"] == "scene_old" for p in synth_inventory(synth, "ffff")["problems"]),
           "inventory: a dataset rendered on another scene is said old")
    c = check(src, Job("check"), model_print="abcd", truth=truth)
    expect(c["level"] in ("ok", "warn") and not [p for p in c["real"]["problems"] if p["level"] == "err"]
           and c["synthetic"]["level"] == "ok", f"check: every file read, nothing wrong ({c['files']} files)")
    # spoil one raw file, take one image away
    victim = trial_files(run, ids[0])[0]
    victim.write_bytes(b"spoiled")
    gone = synth / "sessions" / "s0001" / "r0_s1_p1_label.png"
    gone.unlink()
    bad = synth / "sessions" / "s0002" / "r0_s0_p1.jpg"
    bad.write_bytes(b"not a jpeg at all")
    c = check(src, Job("check"), model_print="abcd", truth=truth)
    codes = {p["code"]: p for p in c["real"]["problems"] + c["synthetic"]["problems"]}
    expect("changed" in codes and victim.name in codes["changed"]["names"],
           f"check: a raw file changed since it was filed is found ({victim.name})")
    expect("truth_stale" in codes and victim.name in codes["truth_stale"]["names"],
           "check: a scan that is no longer the one its ground truth was measured from is said")
    expect("frames_missing" in codes and "s0001" in codes["frames_missing"]["names"]
           and "bad_image" in codes and c["level"] == "err", "check: a missing label image and a broken photo found")
    # a job stopped at once
    j = Job("check")
    j.stop()
    try:
        check(src, j, model_print="abcd", truth=truth)
        expect(False, "check: stopped")
    except Stopped:
        expect(True, "check: a stop asked is heeded")
    # the package, as a folder and as a zip
    out = work / "out"
    out.mkdir()
    dest = free_target(out, package_name(), "")
    res = export(src, dest, Job("export"), model_print="abcd", truth=truth, made_by="ImpactGuide test",
                 simulated=True)
    names = {r_["path"] for r_ in csv.DictReader(io.StringIO((dest / "files.csv").read_text(encoding="utf-8")))}
    expect(dest.is_dir() and (dest / "README.md").is_file() and (dest / "LICENSE.txt").is_file()
           and (dest / "index.json").is_file() and f"real/campaign/{ids[0]}/{victim.name}" in names
           and f"real/campaign/ground_truth/{ids[0]}.json" in names and "synthetic/sessions/s0000/frames.jsonl" in
           names and not any("s0900" in n for n in names), f"export: folder with README, licence, index, the trials, "
           f"their ground truth and the synthetic sessions, not the check set ({res['files']} files)")
    cal = json.loads((dest / "real" / "calibration" / "intrinsics_photo.json").read_text(encoding="utf-8"))
    expect(cal["images"] == "photos", "export: no local path in the package (a folder kept by its last part)")
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in dest.rglob("*")
                     if p.suffix in (".md", ".txt", ".json", ".csv"))
    paths = re.findall(r"[A-Za-z]:\\\\(?:Users|work)\\\\\w+", text)
    expect(not paths, f"export: no folder of this computer in the package's texts ({paths[:2] or 'none'})")
    rec = json.loads((dest / "real" / "campaign" / "ground_truth" / f"{ids[0]}.json").read_text(encoding="utf-8"))
    gtc = list(csv.DictReader(io.StringIO((dest / "real" / "campaign" / "ground_truth.csv").read_text("utf-8"))))
    expect(len(rec["dT"]) == 4 and rec["state"] == "ready" and len(rec["T_M_mandible_after"]) == 4
           and rec["tre_max_mm"] > 0 and rec["scans"] and len(gtc) == 2 and gtc[0]["dT_mandible"],
           f"export: ground truth of {ids[0]} with dT, the registrations and the error at the targets; one row per "
           f"trial in ground_truth.csv")
    idx = json.loads((dest / "index.json").read_text(encoding="utf-8"))
    readme_text = (dest / "README.md").read_text(encoding="utf-8")
    expect(any(p["code"] == "changed" for p in idx["problems"]) and idx["license"]["code"] == "MIT"
           and idx["simulated"] and "made in simulation" in readme_text,
           "export: the changed file said in the index; licences named; a simulation said as such")
    import subprocess
    v = subprocess.run([sys.executable, str(dest / "verify.py")], capture_output=True, text=True, cwd=dest)
    expect(v.returncode == 0, f"export: verify.py finds every file as listed ({v.stdout.strip().splitlines()[-1:]})")
    (dest / "real" / "campaign" / ids[0] / victim.name).write_bytes(b"again")
    v = subprocess.run([sys.executable, str(dest / "verify.py")], capture_output=True, text=True, cwd=dest)
    expect(v.returncode == 1, "export: verify.py finds a file changed after packing")
    zdest = free_target(out, package_name(), ".zip")
    res = export(src, zdest, Job("export"), as_zip=True, model_print="abcd", truth=truth)
    with zipfile.ZipFile(zdest) as z:
        zn = z.namelist()
        top = zdest.stem
        same = z.read(f"{top}/README.md").decode("utf-8").splitlines()[0] == "# ImpactGuide bench dataset"
        jpg = next(i for i in z.infolist() if i.filename.endswith(".jpg"))
    expect(zdest.is_file() and same and f"{top}/files.csv" in zn and jpg.compress_type == zipfile.ZIP_STORED
           and res["zip"], f"export: zip with the same layout under {top}/, photos stored as they are")
    # never over an existing one, nothing left by a stop
    try:
        export(src, dest, Job("export"), truth=truth)
        expect(False, "export: refused over an existing package")
    except FileExistsError:
        expect(True, "export: never over an existing package")
    j = Job("export")
    j.stop()
    d2 = free_target(out, package_name(), "")
    try:
        export(src, d2, j, truth=truth)
    except Stopped:
        pass
    expect(not d2.exists() and not d2.with_name(d2.name + PART).exists(), "export: a stopped export leaves nothing")
    try:
        export(Sources(work / "none", work / "none", work / "none", work / "none"), out / "x", Job("export"))
        expect(False, "export: nothing to pack")
    except ValueError:
        expect(True, "export: nothing to pack is said")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


def _sources(a) -> Sources:
    return Sources(a.raw, a.protocol, a.processed, a.synth)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inventory", "check", "export"):
        q = sub.add_parser(name)
        if name == "export":
            q.add_argument("out", type=Path, help="the folder to write the package into")
            q.add_argument("--zip", action="store_true")
            q.add_argument("--no-real", action="store_true")
            q.add_argument("--no-synthetic", action="store_true")
        for opt in ("raw", "protocol", "processed", "synth"):
            q.add_argument(f"--{opt}", type=Path)
    t = sub.add_parser("test")
    t.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "dataset_pack_test")
    a = p.parse_args(argv)
    if a.cmd == "test":
        return selftest(a.work)
    src = _sources(a)
    import twin
    cur = twin.read_json(twin.CACHE / twin.CURRENT) or {}
    info = twin.read_json(twin.CACHE / str(cur.get("info", ""))) if cur else None
    truth = Truth(src, info)
    model = str(cur.get("phone_print", ""))
    if a.cmd == "inventory":
        print(json.dumps(inventory(src, model, truth), indent=1))
        return 0
    if a.cmd == "check":
        res = check(src, Job("check"), model_print=model, truth=truth)
        print(json.dumps(res, indent=1))
        return 1 if res["level"] == "err" else 0
    parts = [k for k, off in (("real", a.no_real), ("synthetic", a.no_synthetic)) if not off]
    a.out.mkdir(parents=True, exist_ok=True)
    dest = free_target(a.out, package_name(), ".zip" if a.zip else "")
    res = export(src, dest, Job("export"), as_zip=a.zip, parts=parts, model_print=model, truth=truth)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
