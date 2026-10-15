"""Impact trials: randomised sequence, trial log, guided steps and file completeness (T9.1).

The campaign is 2 marker holders x 9 goniometer holes (10 to 50
degrees every 5) x 3 directions (0 frontal, 45, 90 lateral from the right) x 2
repetitions = 108 impacts, plus 20 trials with a drape over part of the face.
The pilot (T9.2) runs holder A at increasing angles, with a 240 fps side video.

Sequence. One block per holder (A, then B): the holder is changed once. Each
repetition inside a holder block is a randomised complete block: every angle x
direction cell once plus its share of the drape trials (10 per holder, every
angle at least once, directions 4/3/3), in random order from a fixed seed.
Directions are not grouped (direction_run = 1): `design` shows why.

Trial ids are given when a trial starts (t001, t002, ... campaign; p001, ...
pilot). A sequence slot is done when a trial with status ok or open has the
same holder, angle, direction and drape: the logged values count, not the slot
that was proposed. A void trial keeps its files and its cell is done again by
a later trial; a different setting done by mistake fills the matching slot.

Reference scans precede the impact when the trial is the first of its phase or
of its holder block, the previous trial has no post-impact scans, its cap moved
more than 3 mm from where the last reference left it (A goes back with the
jig, B is reseated by hand), the previous trial had the drape (taken off at its
end), the session day changed, or the operator asks (e.g. after replacing the
cap). Otherwise the post-impact scans of the previous trial give the
pre-impact registration (log column pre_scan). The drape goes on before any
reference scan and the impact, and comes off after the video, so scans, photos
and video of a drape trial all see the same state.

Files, in 03_data/raw/<phase>/. Every name starts with the trial id, so a file
stays identifiable outside its folder; names carry no conditions, which live in
the log and can be corrected there without touching raw files.

    campaign/
        trial_log.csv            one row per trial (LOG_COLUMNS)
        trial_log_edits.csv      every change to a logged value: time, trial,
                                 field, old, new, reason
        t001/
            t001_ref_scan_az40.ply   reference scans, only when a reference
            t001_ref_scan_az55.ply   precedes the impact
            t001_scan_az40.ply       post-impact scans, arm at 40 and 55 degrees
            t001_scan_az55.ply
            t001_photo_az40.jpg      photos from the 4 arm positions
            t001_photo_az55.jpg
            t001_photo_az70.jpg
            t001_photo_az85.jpg
            t001_video.mp4           video from the arm, 35 to 95 degrees
    pilot/
        p001/                    the same, plus p001_highspeed.mp4

Configuration and sequence in 03_data/processed/protocol (config_<phase>.json,
sequence_<phase>.csv), with the log template and its data dictionary.

ImpactGuide (T20.1) drives a phase with Run: next() gives the trial to do (or
the open one, with the step to resume from), start() opens it, steps() lists
what to do, store() files a photo, video or scan under its name, confirm()
closes a step, close fills the log row; log.edit() and void() keep an audit
trail; configure() regenerates the sequence around the trials already done.

Usage:
    python trials.py sequence [--phase campaign|pilot] [--seed N] [--repetitions N]
                              [--drape N] [--direction-run N]
        writes configuration, sequence, log template and dictionary
    python trials.py check [--phase campaign|pilot] [--deep]
        what is missing for each trial; exit code 1 if anything is
    python trials.py design [--n 2000]
        direction changes and false positives of the direction effect for
        runs of 1, 3 and 9 trials per platter setting
    python trials.py test [--work DIR]
        simulated campaign with fake files; exit code 1 on failure
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "03_data" / "raw"
PROTOCOL = ROOT / "03_data" / "processed" / "protocol"
WORK = Path(r"C:\work\marker-impact\trials_test")

PREFIX = {"campaign": "t", "pilot": "p"}
SUFFIXES = {
    "scan": (".ply", ".stl", ".obj", ".asc"),
    "photo": (".jpg", ".jpeg"),
    "video": (".mp4",),
    "highspeed": (".mp4", ".mov"),
}
IGNORED = {"desktop.ini"}
OPEN, OK, VOID = "open", "ok", "void"
VIDEO_SPEED = 8  # deg/s at most: motion blur under 1 px at 1/500 s
HIGHSPEED_FPS = 200  # minimum frame rate of the pilot side video

LOG_COLUMNS = (
    ("trial", "", "start", "Trial id in execution order: t001... campaign, p001... pilot"),
    ("phase", "", "start", "campaign or pilot"),
    ("session", "date", "start", "Acquisition day, YYYY-MM-DD; the dataset is split by session"),
    ("holder", "", "start", "Marker holder: A slips and stays, B returns by itself"),
    ("angle_deg", "deg", "start", "Release angle, goniometer hole"),
    ("direction_deg", "deg", "start", "Impact direction: 0 frontal, 45, 90 lateral from the right"),
    ("washers_per_side", "", "start", "M8 washers on each side of the mass block"),
    ("swing_mass_g", "g", "start", "Weighed mass of the swinging part"),
    ("drape", "0/1", "start", "1 if a drape covers part of the face from before the impact to after the video"),
    ("reference", "", "start", "Why reference scans precede the impact (first, holder, no_scan, reposition, "
     "drape, session, manual; ';' separated); empty if none"),
    ("pre_scan", "", "start", "Scans giving the pre-impact registration: <trial>_ref, or the previous trial "
     "(its post-impact scans)"),
    ("time_start", "ISO 8601", "start", "Trial opened"),
    ("time_impact", "ISO 8601", "release", "Pendulum released"),
    ("cap_shift_mm", "mm", "scan", "Online: cap displacement since the last reference, from the post-impact "
     "scans; above the threshold the next trial starts with a reposition. Final values come from the "
     "ground-truth processing"),
    ("spheres_ok", "0/1", "scan", "Online: all six spheres found in the post-impact scans"),
    ("second_hit", "0/1", "close", "1 if the pendulum hit the cap again on the rebound"),
    ("cap_replaced", "0/1", "close", "1 if the impact cap was replaced before this impact"),
    ("time_end", "ISO 8601", "close", "Trial closed"),
    ("status", "", "", "open while acquiring, ok when closed, void if not valid (its cell is done again)"),
    ("void_reason", "", "", "Why the trial is void"),
    ("notes", "", "", "Free text"),
)
LOG_FIELDS = tuple(c[0] for c in LOG_COLUMNS)
REQUIRED = ("session", "holder", "angle_deg", "direction_deg", "washers_per_side", "swing_mass_g", "drape",
            "pre_scan", "time_start", "time_impact", "second_hit", "cap_replaced", "time_end")
ONLINE = ("cap_shift_mm", "spheres_ok")
EDIT_FIELDS = ("time", "trial", "field", "old", "new", "reason")
SEQUENCE_FIELDS = ("slot", "holder", "replicate", "angle_deg", "direction_deg", "drape", "change")
SETTINGS = ("holder", "direction_deg", "angle_deg", "washers_per_side", "drape")


@dataclass(frozen=True)
class Config:
    phase: str = "campaign"
    holders: tuple = ("A", "B")
    angles_deg: tuple = (10, 15, 20, 25, 30, 35, 40, 45, 50)
    directions_deg: tuple = (0, 45, 90)
    repetitions: int = 2
    drape_trials: int = 20  # in all, split evenly between holders and repetitions
    order: str = "random"  # or "ascending" (pilot)
    direction_run: int = 1  # trials in a row with the same platter setting
    seed: int = 20261020
    washers_per_side: int = 3
    reposition_mm: float = 3.0  # cap shift since the last reference
    scan_az: tuple = (40, 55)  # arm positions, camera azimuth in degrees
    photo_az: tuple = (40, 55, 70, 85)
    video_az: tuple = (35, 95)
    video_min_s: float = 7.0
    photo_px: tuple = (4080, 3072)
    highspeed: bool = False  # 240 fps side video of the impact


PILOT = Config(phase="pilot", holders=("A",), repetitions=1, drape_trials=0, order="ascending", seed=0,
               highspeed=True)


def default_config(phase):
    return PILOT if phase == "pilot" else Config(phase=phase)


def config_from_dict(d):
    names = {f.name for f in fields(Config)}
    return Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names})


# Sequence


def split(n, k):
    """n items in k parts as even as possible, larger parts first."""
    return [n // k + (i < n % k) for i in range(k)]


def fill_levels(levels, used, n, rng):
    """n more values, each one a level among the least used so far, ties at random."""
    count = Counter({v: 0 for v in levels})
    count.update(used)
    out = []
    for _ in range(n):
        low = min(count.values())
        candidates = [v for v in levels if count[v] == low]
        pick = candidates[rng.integers(len(candidates))]
        out.append(pick)
        count[pick] += 1
    return out


def drape_cells(cfg, n, rng, fixed=()):
    """(angle, direction) of the n drape trials of one holder: the fixed ones (already done) first, the
    others drawn so that angles and directions are used as evenly as possible, randomly paired."""
    fixed = list(fixed)[:n]
    k = n - len(fixed)
    angles = fill_levels(cfg.angles_deg, [a for a, _ in fixed], k, rng)
    directions = fill_levels(cfg.directions_deg, [d for _, d in fixed], k, rng)
    rng.shuffle(directions)
    new = list(zip(angles, directions))
    rng.shuffle(new)
    return fixed + new


def order_block(cfg, main, extra, rng):
    """Order of one repetition: main cells in runs of direction_run trials with the same direction,
    drape cells as runs of one, runs shuffled."""
    if cfg.order == "ascending":
        key = lambda c: (cfg.directions_deg.index(c[1]), c[0])
        return sorted(main, key=key) + sorted(extra, key=key)
    runs = []
    for d in cfg.directions_deg:
        cells = [c for c in main if c[1] == d]
        rng.shuffle(cells)
        runs += [cells[i:i + cfg.direction_run] for i in range(0, len(cells), cfg.direction_run)]
    runs += [[c] for c in extra]
    rng.shuffle(runs)
    return [c for run in runs for c in run]


def design(cfg, fixed=None):
    """All trials of the configuration in the seeded order. fixed: {holder: [(angle, direction)]} of
    drape trials already done, kept among the drape cells."""
    rng = np.random.default_rng(cfg.seed)
    main = [(a, d, 0) for d in cfg.directions_deg for a in cfg.angles_deg]
    rows = []
    for holder, n_drape in zip(cfg.holders, split(cfg.drape_trials, len(cfg.holders))):
        drape = drape_cells(cfg, n_drape, rng, (fixed or {}).get(holder, ()))
        start = 0
        for rep, k in enumerate(split(n_drape, cfg.repetitions), 1):
            extra = [(a, d, 1) for a, d in drape[start:start + k]]
            start += k
            rows += [{"holder": holder, "replicate": rep, "angle_deg": a, "direction_deg": d, "drape": dr}
                     for a, d, dr in order_block(cfg, main, extra, rng)]
    return rows


def cell(r):
    return (r["holder"], int(float(r["angle_deg"])), int(float(r["direction_deg"])), int(float(r["drape"] or 0)))


def changes(prev, row):
    """Settings that differ from the previous trial (all of them for the first)."""
    if prev is None:
        return [n for n in SETTINGS if n in row]
    return [n for n in SETTINGS if n in row and n in prev and str(prev[n]) != str(row[n])]


def make_sequence(cfg, done=()):
    """Sequence rows. done: trials already logged (open or ok), in execution order; their cells come
    first, the remaining ones follow in the seeded order, the holder in use first."""
    fixed = {}
    for t in done:
        h, a, d, dr = cell(t)
        if dr:
            fixed.setdefault(h, []).append((a, d))
    left = design(cfg, fixed)
    head = []
    for t in done:
        hit = next((r for r in left if cell(r) == cell(t)), None)
        if hit is not None:
            left.remove(hit)
            head.append(hit)
    if done:
        left.sort(key=lambda r: r["holder"] != done[-1]["holder"])
    rows, prev = head + left, None
    for i, r in enumerate(rows, 1):
        r["slot"] = i
        r["change"] = ";".join(changes(prev, r))
        prev = r
    return rows


def match(sequence, trials):
    """Slots filled by logged trials: each open or ok trial fills the first free slot with its cell.
    Returns ({slot: trial}, [trials outside the matrix])."""
    free = {}
    for r in sequence:
        free.setdefault(cell(r), []).append(int(r["slot"]))
    filled, extra = {}, []
    for t in trials:
        if t["status"] not in (OPEN, OK):
            continue
        slots = free.get(cell(t))
        if slots:
            filled[slots.pop(0)] = t["trial"]
        else:
            extra.append(t["trial"])
    return filled, extra


# Files and log


def now(when=None):
    return (when or datetime.now()).isoformat(timespec="seconds")


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for _ in range(20):  # Drive can hold the file for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    os.replace(tmp, path)


def write_templates(folder):
    """Empty trial log and its data dictionary."""
    write_csv(folder / "trial_log_template.csv", [], LOG_FIELDS)
    columns = ("column", "unit", "filled", "description")
    write_csv(folder / "trial_log_dictionary.csv", [dict(zip(columns, c)) for c in LOG_COLUMNS], columns)


def file_name(trial, kind, az=None, ref=False):
    stem = f"{trial}_{'ref_' if ref else ''}{kind}" + ("" if az is None else f"_az{az}")
    return stem + SUFFIXES[kind][0]


def file_kind(name):
    stem = Path(name).stem
    return next((k for k in ("highspeed", "video", "photo", "scan") if f"_{k}" in stem), None)


def deep_check(cfg, path):
    """Problems inside a photo or video: pixel size, duration, frame rate."""
    kind = file_kind(path.name)
    if kind == "photo":
        from PIL import Image
        try:
            with Image.open(path) as im:
                size = im.size
        except Exception:
            return [("unreadable", path.name)]
        if sorted(size) != sorted(cfg.photo_px):
            return [("content", f"{path.name}: {size[0]} x {size[1]} px, expected {cfg.photo_px[0]} x {cfg.photo_px[1]}")]
    elif kind in ("video", "highspeed"):
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps or frames <= 0:
            return [("unreadable", path.name)]
        if kind == "video" and frames / fps < cfg.video_min_s:
            return [("content", f"{path.name}: {frames / fps:.1f} s, expected at least {cfg.video_min_s:g} s")]
        if kind == "highspeed" and fps < HIGHSPEED_FPS:
            return [("content", f"{path.name}: {fps:.0f} fps, expected 240")]
    return []


class Log:
    """trial_log.csv, one row per trial, and trial_log_edits.csv, the audit trail."""

    def __init__(self, folder):
        self.path = Path(folder) / "trial_log.csv"
        self.edits_path = Path(folder) / "trial_log_edits.csv"

    def rows(self):
        return read_csv(self.path)

    def edits(self):
        return read_csv(self.edits_path)

    def row(self, trial):
        for r in self.rows():
            if r["trial"] == trial:
                return r
        raise KeyError(trial)

    def _update(self, trial, values):
        rows = self.rows()
        r = next((r for r in rows if r["trial"] == trial), None)
        if r is None:
            raise KeyError(trial)
        old = dict(r)
        r.update(values)
        write_csv(self.path, rows, LOG_FIELDS)
        return old

    def add(self, row):
        rows = self.rows()
        if any(r["trial"] == row["trial"] for r in rows):
            raise ValueError(f"{row['trial']} is already in the log")
        rows.append({c: str(row.get(c, "")) for c in LOG_FIELDS})
        write_csv(self.path, rows, LOG_FIELDS)

    def fill(self, trial, **values):
        """Sets empty fields; a value already logged changes only through edit()."""
        values = {k: str(v) for k, v in values.items()}
        r = self.row(trial)
        for k, v in values.items():
            if k not in LOG_FIELDS:
                raise KeyError(k)
            if r[k] not in ("", v):
                raise ValueError(f"{trial} {k} is already {r[k]!r}: change it with edit() and a reason")
        self._update(trial, values)

    def edit(self, trial, field, value, reason, when=None):
        """Changes a logged value and records old value, new value, time and reason."""
        if not str(reason).strip():
            raise ValueError("an edit needs a reason")
        if field not in LOG_FIELDS or field in ("trial", "phase"):
            raise ValueError(f"{field} cannot be edited")
        new = str(value)
        old = self.row(trial)[field]
        if old == new:
            return
        self._update(trial, {field: new})
        edits = self.edits() + [dict(zip(EDIT_FIELDS, (now(when), trial, field, old, new, reason)))]
        write_csv(self.edits_path, edits, EDIT_FIELDS)

    def set_status(self, trial, status, reason="", when=None):
        """Open trials change status freely; a closed one only through an edit with a reason."""
        if self.row(trial)["status"] == OPEN:
            self._update(trial, {"status": status})
        else:
            self.edit(trial, "status", status, reason, when)


# Steps


@dataclass(frozen=True)
class Step:
    key: str  # stable, can index a translation
    text: str
    files: tuple = ()  # names expected in the trial folder after this step (default suffix)
    fields: tuple = ()  # log fields this step fills


def pendulum_text(row, ch):
    parts = [("direction_deg", f"direction {row['direction_deg']}°"), ("angle_deg", f"hole {row['angle_deg']}°"),
             ("washers_per_side", f"{row['washers_per_side']} washers per side")]
    text = "Set the pendulum: " + ", ".join(t + (" (change)" if k in ch else "") for k, t in parts) + "."
    if "direction_deg" in ch:
        text += " To turn the platter pull its pin, turn it and pin it again."
    return text


def trial_steps(cfg, row, prev=None):
    """Steps of one trial in order."""
    t, holder = row["trial"], row["holder"]
    reasons = {r for r in row["reference"].split(";") if r}
    ch = set(changes(prev, row))
    s = []
    if "holder" in ch:
        s.append(Step("holder", f"Mount marker holder {holder} on the splint."))
    if reasons & {"first", "holder", "reposition"}:
        s.append(Step("reposition", "Put marker A in its start position with the jig, then take the jig off."
                      if holder == "A" else "Seat marker B in its seats and check that it is home."))
    drape = str(row["drape"]) == "1"
    if drape:
        s.append(Step("drape_on", "Lay the drape over part of the face, clear of the marker, the spheres and "
                      "the pendulum path. It stays on until after the video."))
    for az in cfg.scan_az if reasons else ():
        s.append(Step(f"ref_scan_az{az}", f"Reference scan: pendulum hooked up, arm at {az}°, panels off. "
                      "Scan and export.", (file_name(t, "scan", az, ref=True),)))
    s.append(Step("pendulum", pendulum_text(row, ch)))
    if cfg.highspeed:
        s.append(Step("highspeed_start", "Start the 240 fps side video, with the ruler in view."))
    s.append(Step("release", f"Raise the pendulum to the {row['angle_deg']}° hole and release it by hand. "
                  "Stop it on the rebound (no second hit) and hook it up high.", fields=("time_impact",)))
    if cfg.highspeed:
        s.append(Step("highspeed", "Stop the side video and copy it.", (file_name(t, "highspeed"),)))
    for az in cfg.scan_az:
        s.append(Step(f"scan_az{az}", f"Scan: arm at {az}°, panels off. Scan and export.", (file_name(t, "scan", az),)))
    s.append(Step("spheres", "Six spheres found? Cap shift since the last reference, in mm.", fields=ONLINE))
    for az in cfg.photo_az:
        s.append(Step(f"photo_az{az}", f"Photo: arm at {az}°, panels on. Shoot from the PC.", (file_name(t, "photo", az),)))
    a, b = cfg.video_az
    s.append(Step("video", f"Video: lift the index plunger and sweep the arm from end stop to end stop "
                  f"({a}° to {b}°) in about {round((b - a) / VIDEO_SPEED)} s, then back to {cfg.photo_az[0]}°.",
                  (file_name(t, "video"),)))
    if drape:
        s.append(Step("drape_off", "Take the drape off; the next trial starts with reference scans."))
    s.append(Step("close", "Check second hit, cap replaced and notes, then close the trial.",
                  fields=("second_hit", "cap_replaced", "time_end")))
    return s


def cap_shift(T_ref, T_now, cap_M):
    """Distance (mm) between the cap centre in the mandible frame B at the reference and now.
    T_ref, T_now: 4 x 4 T_M_B, mapping B to M (T_C_B = T_C_M T_M_B); cap_M: cap centre in M."""
    p = np.append(np.asarray(cap_M, float), 1.0)
    return float(np.linalg.norm(np.linalg.solve(T_ref, p)[:3] - np.linalg.solve(T_now, p)[:3]))


# A phase driven step by step


class Run:
    """One acquisition phase (campaign or pilot): configuration, sequence, log and trial folders."""

    def __init__(self, phase="campaign", raw=RAW, protocol=PROTOCOL):
        self.phase = phase
        self.folder = Path(raw) / phase
        self.protocol = Path(protocol)
        self.log = Log(self.folder)
        self.config_path = self.protocol / f"config_{phase}.json"
        self.sequence_path = self.protocol / f"sequence_{phase}.csv"
        if self.config_path.exists() and self.sequence_path.exists():
            self.config = config_from_dict(json.loads(self.config_path.read_text(encoding="utf-8"))["config"])
            self.sequence = read_csv(self.sequence_path)
        else:
            self.configure(default_config(phase))

    def configure(self, cfg):
        """Sets the configuration (matrix, repetitions, drape trials, seed...) and regenerates the
        sequence: trials already logged keep their cells, the others follow the new seeded order.
        An unchanged configuration keeps the sequence as it is."""
        if cfg.phase != self.phase:
            raise ValueError(f"configuration for {cfg.phase}, run for {self.phase}")
        new = json.loads(json.dumps(asdict(cfg)))
        old = json.loads(self.config_path.read_text(encoding="utf-8")) if self.config_path.exists() else None
        if old is None or old["config"] != new or not self.sequence_path.exists():
            done = [r for r in self.log.rows() if r["status"] in (OPEN, OK)]
            write_csv(self.sequence_path, make_sequence(cfg, done), SEQUENCE_FIELDS)
        if old is None or old["config"] != new:
            history = old["history"] + [{"written": old["written"], "config": old["config"]}] if old else []
            doc = {"written": now(), "config": new, "history": history}
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        self.config = cfg
        self.sequence = read_csv(self.sequence_path)

    def pending(self):
        filled, _ = match(self.sequence, self.log.rows())
        return [s for s in self.sequence if int(s["slot"]) not in filled]

    def new_id(self):
        p = PREFIX[self.phase]
        n = [int(r["trial"][1:]) for r in self.log.rows()]
        if self.folder.exists():
            n += [int(d.name[1:]) for d in self.folder.glob(p + "*") if d.is_dir() and d.name[1:].isdigit()]
        return f"{p}{max(n, default=0) + 1:03d}"

    def find(self, trial, name):
        """Files of the trial folder with the stem of name and a suffix allowed for its kind."""
        folder, stem = self.folder / trial, Path(name).stem
        allowed = SUFFIXES[file_kind(name)]
        if not folder.exists():
            return []
        return sorted(p for p in folder.iterdir() if p.is_file() and p.stem == stem and p.suffix.lower() in allowed)

    def has_scans(self, trial):
        return all(self.find(trial, file_name(trial, "scan", az)) for az in self.config.scan_az)

    def next(self, today=None, reference=None):
        """The trial to do now: the open one (resume, with the index of the step to resume from) or a
        plan for the next pending slot of the holder in use; None when the sequence is complete.
        reference: a reason for reference scans chosen by the operator (e.g. "manual" after a new cap)."""
        rows = self.log.rows()
        for r in rows:
            if r["status"] == OPEN:
                return {**r, "resume": True, "step": self.resume_step(r["trial"]), "changes": [], "warnings": []}
        pending = self.pending()
        if not pending:
            return None
        prev = rows[-1] if rows else None
        s = next((x for x in pending if prev and x["holder"] == prev["holder"]), pending[0])
        today = (today or date.today()).isoformat()
        plan = {"trial": self.new_id(), "phase": self.phase, "session": today, "holder": s["holder"],
                "angle_deg": s["angle_deg"], "direction_deg": s["direction_deg"],
                "washers_per_side": str(self.config.washers_per_side), "drape": s["drape"], "slot": s["slot"]}
        reasons, warnings = [], []
        if prev is None:
            reasons.append("first")
        else:
            if prev["holder"] != s["holder"]:
                reasons.append("holder")
            if not self.has_scans(prev["trial"]):
                reasons.append("no_scan")
            elif not prev["cap_shift_mm"]:
                warnings.append(f"cap shift of {prev['trial']} unknown: check the cap, ask for a reference if it moved")
            elif float(prev["cap_shift_mm"]) > self.config.reposition_mm:
                reasons.append("reposition")
            if prev["drape"] == "1":
                reasons.append("drape")
            if prev["session"] != today:
                reasons.append("session")
        if reference:
            reasons.append(reference)
        plan["reference"] = ";".join(dict.fromkeys(reasons))
        plan["pre_scan"] = f"{plan['trial']}_ref" if reasons else prev["trial"]
        same_mass = prev is not None and prev["washers_per_side"] == plan["washers_per_side"]
        plan["swing_mass_g"] = prev["swing_mass_g"] if same_mass else ""
        plan.update(resume=False, step=0, changes=changes(prev, plan), warnings=warnings)
        return plan

    def start(self, plan, when=None, **values):
        """Opens a planned trial: log row with status open, and its folder."""
        row = {**{k: plan.get(k, "") for k in LOG_FIELDS}, **values, "time_start": now(when), "status": OPEN}
        self.log.add(row)
        (self.folder / plan["trial"]).mkdir(parents=True, exist_ok=True)
        return self.log.row(plan["trial"])

    def steps(self, trial):
        rows = self.log.rows()
        i = next(i for i, r in enumerate(rows) if r["trial"] == trial)
        return trial_steps(self.config, rows[i], rows[i - 1] if i else None)

    def step_done(self, row, step):
        return all(self.find(row["trial"], n) for n in step.files) and all(row[f] for f in step.fields)

    def resume_step(self, trial):
        """Index of the first step to do: the one after the last step with all its files and fields."""
        row, last = self.log.row(trial), -1
        for i, s in enumerate(self.steps(trial)):
            if (s.files or s.fields) and self.step_done(row, s):
                last = i
        return last + 1

    def target(self, trial, name, suffix=None):
        """Path for a file of the trial; suffix of the source file if it differs from the default."""
        kind = file_kind(name)
        suffix = (suffix or Path(name).suffix).lower()
        if kind is None or suffix not in SUFFIXES[kind]:
            raise ValueError(f"{name} with suffix {suffix}: not a {kind} file")
        return self.folder / trial / (Path(name).stem + suffix)

    def store(self, trial, name, source, move=False):
        """Files a photo, video or scan under its name; never overwrites a raw file."""
        source = Path(source)
        dest = self.target(trial, name, source.suffix)
        if self.find(trial, name):
            raise FileExistsError(f"{self.find(trial, name)[0].name} is already there")
        dest.parent.mkdir(parents=True, exist_ok=True)
        (shutil.move if move else shutil.copy2)(source, dest)
        return dest

    def missing(self, trial, deep=False):
        """What is missing or wrong for a trial, as (kind, item): missing, duplicate, empty,
        unexpected, unreadable or content (deep only) for files; log for required log fields;
        online for the values computed from the scans. Void trials only report unexpected files."""
        row = self.log.row(trial)
        names = [n for s in self.steps(trial) for n in s.files]
        out = []
        if row["status"] != VOID:
            for n in names:
                hits = self.find(trial, n)
                if not hits:
                    out.append(("missing", n))
                elif len(hits) > 1:
                    out.append(("duplicate", ", ".join(h.name for h in hits)))
                elif hits[0].stat().st_size == 0:
                    out.append(("empty", hits[0].name))
                elif deep:
                    out += deep_check(self.config, hits[0])
            out += [("log", f) for f in REQUIRED if not row[f]]
            out += [("online", f) for f in ONLINE if not row[f]]
        expected = {Path(n).stem: n for n in names}
        folder = self.folder / trial
        for p in sorted(folder.iterdir()) if folder.exists() else ():
            if p.name in IGNORED:
                continue
            n = expected.get(p.stem)
            if not p.is_file() or n is None or p.suffix.lower() not in SUFFIXES[file_kind(n)]:
                out.append(("unexpected", p.name))
        return out

    def confirm(self, trial, key, when=None, **values):
        """Closes a step: checks its files and fills its log fields; "close" sets status ok only if
        nothing required is missing (online values may still be empty)."""
        step = next(s for s in self.steps(trial) if s.key == key)
        lacking = [n for n in step.files if not self.find(trial, n)]
        if lacking:
            raise FileNotFoundError(", ".join(lacking))
        if key == "release":
            values.setdefault("time_impact", now(when))
        if key == "close":
            values = {"second_hit": 0, "cap_replaced": 0, "time_end": now(when), **values}
            problems = [p for p in self.missing(trial)
                        if p[0] not in ("online", "unexpected") and not (p[0] == "log" and p[1] in values)]
            if problems:
                raise ValueError(f"{trial} cannot be closed: " + "; ".join(f"{k} {v}" for k, v in problems))
        if values:
            self.log.fill(trial, **values)
        if key == "close":
            self.log.set_status(trial, OK)

    def void(self, trial, reason, when=None):
        """Marks a trial not valid; its files stay and its cell goes back into the sequence."""
        row = self.log.row(trial)
        self.log.set_status(trial, VOID, reason, when)
        if row["void_reason"]:
            self.log.edit(trial, "void_reason", reason, reason, when)
        else:
            self.log.fill(trial, void_reason=reason)
        if not row["time_end"]:
            self.log.fill(trial, time_end=now(when))

    def check(self, deep=False):
        """Completeness of the phase: problems per logged trial, pending slots, trials outside the
        matrix, trial folders without a log row."""
        rows = self.log.rows()
        filled, extra = match(self.sequence, rows)
        logged = {r["trial"] for r in rows}
        orphans = sorted(d.name for d in self.folder.iterdir() if d.is_dir() and d.name not in logged) \
            if self.folder.exists() else []
        return {"trials": [{"trial": r["trial"], "status": r["status"], "problems": self.missing(r["trial"], deep)}
                           for r in rows],
                "pending": [s for s in self.sequence if int(s["slot"]) not in filled],
                "extra": extra, "orphans": orphans}


# Design check


def design_check(n=2000, seed=1, runs=(1, 3, 9), drifts=(0.0, 1.0), setup_sd=0.5, change_s=30):
    """False-positive rate of the direction effect when directions have none, per run length.
    One holder block (54 trials + 10 drape trials). Response = linear drift over the block (total
    drift in noise SDs) + platter setup error (setup_sd, new at every direction change) + noise (SD 1).
    Planned analysis: angle + direction + replicate, F test on direction at 0.05, drape trials left out."""
    from scipy import stats

    def dummies(v):
        levels = np.unique(v)
        return (v[:, None] == levels[None, 1:]).astype(float)

    rng = np.random.default_rng(seed)
    out = []
    for run in runs:
        rates, n_changes = [], 0
        for drift in drifts:
            hits = 0
            for _ in range(n):
                rows = design(Config(holders=("A",), drape_trials=10, direction_run=run,
                                     seed=int(rng.integers(2**32))))
                m = len(rows)
                d = np.array([r["direction_deg"] for r in rows])
                new = np.r_[True, d[1:] != d[:-1]]
                y = (drift * np.arange(m) / (m - 1) + rng.normal(0, setup_sd, new.sum())[np.cumsum(new) - 1]
                     + rng.normal(0, 1, m))
                keep = np.array([r["drape"] == 0 for r in rows])
                a = np.array([r["angle_deg"] for r in rows])[keep]
                rep = np.array([r["replicate"] for r in rows])[keep]
                X0 = np.column_stack([np.ones(keep.sum()), dummies(a), dummies(rep)])
                X1 = np.column_stack([X0, dummies(d[keep])])
                rss = [np.sum((y[keep] - X @ np.linalg.lstsq(X, y[keep], rcond=None)[0]) ** 2) for X in (X0, X1)]
                df = keep.sum() - X1.shape[1]
                f = (rss[0] - rss[1]) / 2 / (rss[1] / df)
                hits += stats.f.sf(f, 2, df) < 0.05
                n_changes += new.sum() - 1
            rates.append(hits / n)
        mean_changes = n_changes / (n * len(drifts))
        out.append({"run": run, "changes": mean_changes, "minutes": mean_changes * change_s / 60,
                    **{f"false_pos_drift_{dr:g}": r for dr, r in zip(drifts, rates)}})
    return out


# Test


def run_test(work=WORK):
    """Sequence properties, then a simulated campaign with fake files: references, a void trial,
    a wrong setting corrected by an edit, an app closed mid-trial, a missing video, a trial voided
    after closing, a configuration change, then completeness checks."""
    if work.exists():
        shutil.rmtree(work)
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            fails.append(what)

    cfg = Config()
    seq = make_sequence(cfg)
    count = Counter(cell(r) for r in seq)
    expect(len(seq) == 128, "campaign: 128 trials")
    expect(all(count[(h, a, d, 0)] == 2 for h in cfg.holders for a in cfg.angles_deg for d in cfg.directions_deg),
           "every holder x angle x direction cell twice")
    expect([r["holder"] for r in seq] == ["A"] * 64 + ["B"] * 64, "one block per holder, A then B")
    full = Counter((a, d) for a in cfg.angles_deg for d in cfg.directions_deg)
    expect(all(Counter((r["angle_deg"], r["direction_deg"]) for r in seq
                       if r["holder"] == h and r["replicate"] == k and not r["drape"]) == full
               for h in cfg.holders for k in (1, 2)), "each repetition is a complete block")
    ok_drape = True
    for h in cfg.holders:
        dr = [r for r in seq if r["holder"] == h and r["drape"]]
        ok_drape &= len(dr) == 10 and set(r["angle_deg"] for r in dr) == set(cfg.angles_deg)
        ok_drape &= sorted(Counter(r["direction_deg"] for r in dr).values()) == [3, 3, 4]
        ok_drape &= [sum(r["replicate"] == k for r in dr) for k in (1, 2)] == [5, 5]
    expect(ok_drape, "10 drape trials per holder, every angle, directions 4/3/3, 5 per repetition")
    expect(make_sequence(cfg) == seq and make_sequence(replace(cfg, seed=1)) != seq, "fixed seed")

    def n_changes(rows):
        return sum(a["direction_deg"] != b["direction_deg"] for a, b in zip(rows, rows[1:]))

    grouped = make_sequence(replace(cfg, direction_run=9))
    expect(n_changes(grouped) < n_changes(seq) / 2, f"runs of 9: {n_changes(grouped)} direction changes "
           f"instead of {n_changes(seq)}")
    pilot = make_sequence(PILOT)
    expect(len(pilot) == 27 and all(pilot[i]["angle_deg"] < pilot[i + 1]["angle_deg"]
                                    for i in range(26) if pilot[i]["direction_deg"] == pilot[i + 1]["direction_deg"]),
           "pilot: 27 trials, angles ascending in each direction")

    raw, protocol, src = work / "raw", work / "protocol", work / "src"
    src.mkdir(parents=True)
    fakes = {}
    for suffix in {s[0] for s in SUFFIXES.values()}:
        fakes[suffix] = src / f"fake{suffix}"
        fakes[suffix].write_bytes(b"fake")
    fake = lambda name: fakes[Path(name).suffix]
    run = Run("campaign", raw, protocol)
    rng = np.random.default_rng(7)
    clock = datetime(2026, 10, 20, 9, 0)
    first_b = edited = None
    ok_ref = ok_pre = True
    for k in range(1, 300):
        day = date(2026, 10, 20) + timedelta(days=(k - 1) // 45)
        if k == 61:
            run.configure(replace(run.config, drape_trials=24))
            valid = sum(r["status"] == OK for r in run.log.rows())
            expect(len(run.sequence) == 132 and len(run.pending()) == 132 - valid,
                   f"24 drape trials from trial 61: 132 slots, the {valid} valid trials kept")
            seq_before = run.sequence
            run.configure(run.config)
            expect(run.sequence == seq_before, "an unchanged configuration keeps the sequence")
        plan = run.next(today=day)
        if plan is None:
            break
        rows = run.log.rows()
        prev = rows[-1] if rows else None
        want = set()
        if prev is None:
            want.add("first")
        else:
            if prev["holder"] != plan["holder"]:
                want.add("holder")
                first_b = plan["trial"]
            if not run.has_scans(prev["trial"]):
                want.add("no_scan")
            elif float(prev["cap_shift_mm"]) > 3:
                want.add("reposition")
            if prev["drape"] == "1":
                want.add("drape")
            if prev["session"] != day.isoformat():
                want.add("session")
        ok_ref &= set(filter(None, plan["reference"].split(";"))) == want
        ok_pre &= plan["pre_scan"] == (f"{plan['trial']}_ref" if want else prev["trial"])
        if k == 6:
            expect(cell(plan) == cell(run.log.row("t005")) and "no_scan" in want,
                   "void trial t005: its cell comes next, with reference scans")
        if k == 41:
            expect(cell(plan) == cell(run.log.row("t040")), "t040 voided after closing: its cell comes next")
        t = run.start(plan, when=clock, **({} if plan["swing_mass_g"] else {"swing_mass_g": 179.0}))["trial"]
        steps = run.steps(t)
        for st in steps:
            if k == 20 and st.key == "photo_az70":
                run = Run("campaign", raw, protocol)
                again = run.next(today=day)
                expect(again["resume"] and again["trial"] == t and steps[again["step"]].key == "photo_az70",
                       f"app closed during {t}: resumes at photo_az70")
            if k == 30 and st.key == "video":
                continue
            for name in st.files:
                run.store(t, name, fake(name))
            values = {}
            if st.key == "spheres":
                shift = rng.uniform(0, 6) if plan["holder"] == "A" else rng.uniform(0, 0.5)
                values = {"cap_shift_mm": f"{shift:.2f}", "spheres_ok": 1}
            if k == 30 and st.key == "close":
                try:
                    run.confirm(t, "close", when=clock)
                    expect(False, f"{t} closed without its video")
                except ValueError as e:
                    expect("video" in str(e) and run.log.row(t)["status"] == OPEN,
                           f"{t} without video cannot be closed")
                run.store(t, file_name(t, "video"), fake(file_name(t, "video")))
            run.confirm(t, st.key, when=clock, **values)
            if k == 5 and st.key == "release":
                run.void(t, "second hit: the pendulum was not stopped")
                break
        if edited is None and k >= 9 and plan["drape"] == "0":
            free = [s for s in run.pending() if s["angle_deg"] != plan["angle_deg"]
                    and (s["holder"], s["direction_deg"], s["drape"]) == (plan["holder"], plan["direction_deg"], "0")]
            run.log.edit(t, "angle_deg", free[0]["angle_deg"], f"hole {free[0]['angle_deg']} used, not the planned one")
            edited = t
        if k == 40:
            run.void(t, "scans exported from the wrong project")
        clock += timedelta(minutes=5)
    expect(ok_ref, "reference scans exactly when due (first, holder, no_scan, reposition, drape, session)")
    expect(ok_pre, "pre_scan: own reference or previous trial")
    expect(first_b is not None and run.steps(first_b)[0].key == "holder", f"holder change at {first_b}")
    rows = run.log.rows()
    report = run.check()
    expect(not report["pending"] and not report["extra"] and not report["orphans"], "sequence complete")
    done = Counter(cell(r) for r in rows if r["status"] == OK)
    expect(all(done[(h, a, d, 0)] == 2 for h in cfg.holders for a in cfg.angles_deg for d in cfg.directions_deg)
           and all(sum(n for c, n in done.items() if c[0] == h and c[3]) == 12 for h in cfg.holders),
           f"valid trials: every cell twice and 12 drape trials per holder; {len(rows)} trials in all")
    expect(all(not t["problems"] for t in report["trials"]), "no problems in any trial")
    edits = run.log.edits()
    expect([(e["trial"], e["field"]) for e in edits] == [(edited, "angle_deg"), ("t040", "status")],
           f"audit trail: the angle of {edited} and the status of t040")
    try:
        run.log.fill(edited, angle_deg=50)
        expect(False, "logged value overwritten without edit()")
    except ValueError:
        expect(True, "logged values change only through edit()")
    try:
        run.store("t003", file_name("t003", "photo", 40), fake("t003_photo_az40.jpg"))
        expect(False, "raw file overwritten")
    except FileExistsError:
        expect(True, "store() never overwrites a raw file")

    t3 = raw / "campaign" / "t003"
    run.find("t003", file_name("t003", "photo", 40))[0].unlink()
    (t3 / "IMG_20261020_101112.jpg").write_bytes(b"x")
    run.find("t003", file_name("t003", "scan", 55))[0].write_bytes(b"")
    (t3 / "t003_video.mov").write_bytes(b"x")
    got = set(run.missing("t003"))
    want = {("missing", "t003_photo_az40.jpg"), ("unexpected", "IMG_20261020_101112.jpg"),
            ("empty", "t003_scan_az55.ply"), ("unexpected", "t003_video.mov")}
    expect(got == want, "completeness: missing photo, stray file, empty scan, wrong video suffix")

    import cv2
    cases = {"good.jpg": (3072, 4080), "small.jpg": (480, 640)}
    for name, shape in cases.items():
        cv2.imwrite(str(src / f"t900_photo_{name}"), np.zeros(shape + (3,), np.uint8))
    for name, seconds in (("long", 8), ("short", 3)):
        w = cv2.VideoWriter(str(src / f"t900_video_{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        for _ in range(30 * seconds):
            w.write(np.zeros((48, 64, 3), np.uint8))
        w.release()
    deep = {p.name: deep_check(cfg, p) for p in src.glob("t900_*")}
    expect(not deep["t900_photo_good.jpg"] and deep["t900_photo_small.jpg"][0][0] == "content"
           and not deep["t900_video_long.mp4"] and deep["t900_video_short.mp4"][0][0] == "content",
           "deep check: photo size and video duration")

    print(f"\n{len(fails)} failures" if fails else "\nall checks passed")
    return 1 if fails else 0


def print_check(report):
    bad = [t for t in report["trials"] if t["problems"]]
    for t in bad:
        print(f"{t['trial']} ({t['status']}): " + "; ".join(f"{k} {v}" for k, v in t["problems"]))
    n = Counter(t["status"] for t in report["trials"])
    print(f"{len(report['trials'])} trials logged ({', '.join(f'{v} {k}' for k, v in n.items()) or 'none'}), "
          f"{len(bad)} with problems, {len(report['pending'])} slots pending, "
          f"outside the matrix: {', '.join(report['extra']) or 'none'}, "
          f"folders without a log row: {', '.join(report['orphans']) or 'none'}")
    return 1 if bad or report["orphans"] else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sequence")
    a.add_argument("--phase", default="campaign", choices=PREFIX)
    a.add_argument("--seed", type=int)
    a.add_argument("--repetitions", type=int)
    a.add_argument("--drape", type=int, dest="drape_trials")
    a.add_argument("--direction-run", type=int)
    a = sub.add_parser("check")
    a.add_argument("--phase", default="campaign", choices=PREFIX)
    a.add_argument("--deep", action="store_true")
    a = sub.add_parser("design")
    a.add_argument("--n", type=int, default=2000)
    a = sub.add_parser("test")
    a.add_argument("--work", type=Path, default=WORK)
    args = p.parse_args()

    if args.cmd == "sequence":
        run = Run(args.phase)
        new = {k: getattr(args, k) for k in ("seed", "repetitions", "drape_trials", "direction_run")
               if getattr(args, k) is not None}
        run.configure(replace(run.config, **new))
        write_templates(run.protocol)
        seq = run.sequence
        dir_changes = sum(a["direction_deg"] != b["direction_deg"] for a, b in zip(seq, seq[1:]))
        holders = Counter(r["holder"] for r in seq)
        print(f"{run.sequence_path}: {len(seq)} trials ({', '.join(f'{h} {n}' for h, n in holders.items())}), "
              f"{sum(r['drape'] == '1' for r in seq)} with drape, {dir_changes} direction changes, "
              f"{len(run.pending())} pending")
    elif args.cmd == "check":
        sys.exit(print_check(Run(args.phase).check(args.deep)))
    elif args.cmd == "design":
        print("run  changes per holder  minutes  false positives: no drift  drift 1 SD")
        for r in design_check(args.n):
            print(f"{r['run']:3d}  {r['changes']:18.1f}  {r['minutes']:7.1f}  "
                  f"{r['false_pos_drift_0']:24.3f}  {r['false_pos_drift_1']:10.3f}")
    elif args.cmd == "test":
        sys.exit(run_test(args.work))


if __name__ == "__main__":
    main()
