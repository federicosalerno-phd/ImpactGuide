"""Impact trials: randomised sequence, trial log, guided steps and file completeness.

The campaign is 2 marker holders x 9 goniometer holes (10 to 50
degrees every 5) x 3 directions (0 frontal, 45, 90 lateral from the right) x 2
repetitions = 108 impacts, plus 20 trials with a drape over part of the face.
The pilot runs holder A at increasing angles, with a 240 fps side video.

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
jig, B is reseated by hand), the session day changed, or the operator asks
(e.g. after replacing the cap). Otherwise the post-impact scans of the previous
trial give the pre-impact registration (log column pre_scan).

Drape trials. The drape goes on before any reference scan and the impact.
After the impact the photos and the video are taken with the drape, the drape
comes off slowly without touching the marker, the photos and the video are
taken again, and then come the scans. So the post-impact scans see the same
state as the photos and the video without the drape, as in every other trial,
and the next trial needs no reference scans because of the drape.

Files, in 03_data/raw/<phase>/. Every name starts with the trial id, so a file
stays identifiable outside its folder; names carry no conditions, which live in
the log and can be corrected there without touching raw files.

    campaign/
        trial_log.csv            one row per trial (LOG_COLUMNS)
        trial_log_edits.csv      every change to a logged value: time, trial,
                                 field, old, new, reason
        file_manifest.csv        every raw file filed: time, trial, file,
                                 bytes, SHA-256
        trial_log.lock           only while a program writes the log
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
            t001_drape_photo_az40.jpg  drape trials only: photos and video
            ...                        with the drape, before it comes off
            t001_drape_video.mp4
    pilot/
        p001/                    the same, plus p001_highspeed.mp4

Configuration and sequence in 03_data/processed/protocol (config_<phase>.json,
sequence_<phase>.csv), with the log template and its data dictionary.

Writing. Every CSV and JSON is written whole into a temporary file of the same
folder, flushed to the disk and put in place in one step, so a crash leaves the
old file or the new one, never a part of one. A raw file is copied beside its
final name as <name>.part, flushed, compared with its source (size and SHA-256)
and only then given its name, which is never taken twice (case does not count);
the source of a move is deleted last. One program writes a phase at a time
(trial_log.lock): the app and this command line wait for each other, and a log
held open by another program is said as log_locked. Closing a trial is one
write. Every change to a value already logged goes to the audit trail before
the log. Logged values are checked before anything is written (bad_field).
The log is read as a spreadsheet program may save it: with a byte order mark,
';' between fields and decimal commas, in the Windows code page, days written
dd/mm/yyyy, columns of an older version missing; a file that cannot be read is
log_corrupt and is never written over. Times carry their offset from UTC.

Errors meant for the operator are TrialsError with a code (CODES) and its
values (params); each is also the built-in exception callers expect.

A phase is driven with Run: next() gives the trial to do (or the open one, with
the step to resume from), start() opens it, steps() lists what to do, store()
files a photo, video or scan under its name, confirm() closes a step, close
fills the log row; log.edit() and void() keep an audit trail; configure()
regenerates the sequence around the trials already done.

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
        simulated campaign with fake files; exit code 1 on failure. DIR must be
        empty, missing, or a folder of an earlier test (it is emptied)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import ClassVar
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

_LOGGER = logging.getLogger("trials")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "03_data" / "raw"
PROTOCOL = ROOT / "03_data" / "processed" / "protocol"
WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact")) / "trials_test"

PRE_IMPACT = "pre_impact"      # positions of the marker without an impact (pre_impact.py), ids m001...
PREFIX = {"campaign": "t", "pilot": "p", PRE_IMPACT: "m"}
LIGHTS = ("full", "half", "left_off", "right_off")      # the panels of a pre-impact position
SUFFIXES = {
    "scan": (".ply", ".stl", ".obj", ".asc", ".3mf"),
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
    ("drape", "0/1", "start", "1 if a drape covers part of the face from before the impact; photos and video are "
     "taken with it, then again after taking it off, then the scans"),
    ("set", "", "start", "Pre-impact: the set of positions (1, 2, 3...); the dataset is split by set"),
    ("position", "", "start", "Pre-impact: the marker's position in its set (1 from the jig, then moved by hand)"),
    ("light", "", "start", "Pre-impact: the panels for the photos and the video: full, half, left_off, right_off"),
    ("reference", "", "start", "Why reference scans precede the impact (first, holder, no_scan, reposition, "
     "session, manual; ';' separated); empty if none"),
    ("pre_scan", "", "start", "Scans giving the pre-impact registration: <trial>_ref, or the previous trial "
     "(its post-impact scans)"),
    ("time_start", "ISO 8601", "start", "Trial opened"),
    ("config_version", "", "start", "Version of the phase configuration in force when the trial started "
     "(config_<phase>.json, 1 for the first, one more at each change): the trial's steps and expected files follow "
     "it, whatever the configuration is today; empty for trials logged before versions were kept"),
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
MANIFEST_FIELDS = ("time", "trial", "file", "bytes", "sha256")

LOG_NAME, EDITS_NAME, MANIFEST_NAME, LOCK_NAME = ("trial_log.csv", "trial_log_edits.csv", "file_manifest.csv",
                                                  "trial_log.lock")
PART_SUFFIX = ".part"           # a raw file being copied, beside its final name
TEST_MARKER = ".impactguide-test"   # a folder made by a test, which a later test may empty
NUMERIC_FIELDS = ("angle_deg", "direction_deg", "washers_per_side", "swing_mass_g", "cap_shift_mm", "set", "position")
SEQUENCE_NUMERIC = ("slot", "replicate", "angle_deg", "direction_deg", "drape")
FLAG_FIELDS = ("drape", "spheres_ok", "second_hit", "cap_replaced")
TIME_FIELDS = ("time_start", "time_impact", "time_end")
TEXT_FIELDS = ("void_reason", "notes")
NOT_EDITABLE = ("trial", "phase", "config_version")
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")    # a spreadsheet would run a cell that starts so
MAX_TEXT = 2000                 # characters of a note or a reason
NOTE_JOIN = "; "                # notes added later follow the earlier ones

WRITE_RETRIES = 20              # a file held for a moment (Drive, a virus scanner, a spreadsheet)...
WRITE_RETRY_S = 0.25            # ... is tried again this often, then said as log_locked
LOCK_TIMEOUT_S = 10.0           # another program writing the same phase is waited for this long
LOCK_STALE_S = 60.0             # a lock file older than this was left by a program that stopped
LOCK_POLL_S = 0.05
COPY_BLOCK = 1 << 20            # bytes read at a time when copying and hashing

# The codes of TrialsError, for the interface (with the values in params).
CODES = ("log_locked", "log_corrupt", "config_corrupt", "bad_field", "unknown_trial", "unknown_step", "exists",
         "missing_file", "copy_failed", "cannot_close", "not_open", "reason", "trial_open")

TRIAL_RE = re.compile(r"[a-z]\d{3,}")
PRE_SCAN_RE = re.compile(r"[a-z]\d{3,}(?:_ref)?")
REASONS_RE = re.compile(r"[a-z_]+(?:;[a-z_]+)*")
HOLDER_RE = re.compile(r"[A-Za-z0-9_-]{1,16}")
COMMA_NUMBER_RE = re.compile(r"-?\d+,\d+")
DAY_FIRST_RE = re.compile(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})")


# Errors


class TrialsError(Exception):
    """A failure the interface can name.

    Attributes:
        code: one of CODES.
        cause: the same as code (the name the app reads on other modules' errors).
        params: the values a sentence about it needs (trial, field, file...).
    """

    def __init__(self, code: str, message: str = "", **params: object) -> None:
        super().__init__(message or code)
        self.code = self.cause = code
        self.params = params


class LogError(TrialsError, ValueError):
    """A logged value, a log file or a request on the log that is not valid."""


class LogLocked(TrialsError, PermissionError):
    """The log or one of its files is held by another program."""


class NotFound(TrialsError, KeyError):
    """No such trial or step."""


class FileClash(TrialsError, FileExistsError):
    """A raw file of that name is already filed."""


class FilesMissing(TrialsError, FileNotFoundError):
    """The files a step needs are not there."""


class CopyFailed(TrialsError, OSError):
    """A copy that does not match its source; nothing was filed."""


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
    photo_px: tuple = (4096, 3072)  # the camera app's 12.5 MP photo, as its files are (the calibration's, when kept)
    highspeed: bool = False  # 240 fps side video of the impact


PILOT = Config(phase="pilot", holders=("A",), repetitions=1, drape_trials=0, order="ascending", seed=0,
               highspeed=True)


def default_config(phase: str) -> Config:
    """The configuration a phase starts with."""
    if phase == PRE_IMPACT:
        from pre_impact import PRE          # the sets of positions: their own defaults
        return PRE
    return PILOT if phase == "pilot" else Config(phase=phase)


def config_from_dict(d: dict) -> Config:
    """A Config from its JSON form (lists become tuples, unknown keys are left out)."""
    names = {f.name for f in fields(Config)}
    return Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names})


# The cap and the repositioning rule

HOLDER_JSON = PROTOCOL / "marker_holder.json"   # pilot.py model: the marker holder read from the bench scene


def cap_point_M(path: Path | str = HOLDER_JSON) -> np.ndarray | None:
    """The cap's strike point in the board frame M (mm; the point of the cap's axis at the height the pendulum's nose
    hits), from the marker holder of the bench scene; None when the file is missing or cannot be read."""
    try:
        p = np.asarray(json.loads(Path(path).read_text(encoding="utf-8"))["strike_point_M"], np.float64)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return p if p.shape == (3,) and np.isfinite(p).all() else None


def cap_shift_mm(T_ref: np.ndarray, T_now: np.ndarray, p_M: np.ndarray | None = None) -> float:
    """How far the cap moved on the head between two registrations T_M_head (4 x 4, board M from a head frame, any
    one of the head's frames): |T_now^-1 p - T_ref^-1 p| with p the strike point in M (cap_point_M())."""
    p = cap_point_M() if p_M is None else np.asarray(p_M, np.float64)
    if p is None:
        raise ValueError("no cap point: run pilot.py model")
    h = np.append(p, 1.0)
    return float(np.linalg.norm((np.linalg.solve(T_now, h) - np.linalg.solve(T_ref, h))[:3]))


# Sequence


def split(n: int, k: int) -> list[int]:
    """n items in k parts as even as possible, larger parts first."""
    return [n // k + (i < n % k) for i in range(k)]


def fill_levels(levels: Sequence, used: Iterable, n: int, rng: np.random.Generator) -> list:
    """n more values, each one a level among the least used so far, ties at random. Values used that are no longer
    levels (a configuration changed since) do not count."""
    levels = list(levels)
    if not levels:
        if n:
            raise ValueError("no levels to choose from")
        return []
    count = Counter({v: 0 for v in levels})
    count.update(v for v in used if v in count)
    out = []
    for _ in range(n):
        low = min(count[v] for v in levels)
        candidates = [v for v in levels if count[v] == low]
        pick = candidates[rng.integers(len(candidates))]
        out.append(pick)
        count[pick] += 1
    return out


def drape_cells(cfg: Config, n: int, rng: np.random.Generator, fixed: Iterable = ()) -> list:
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


def order_block(cfg: Config, main: list, extra: list, rng: np.random.Generator) -> list:
    """Order of one repetition: main cells in runs of direction_run trials with the same direction,
    drape cells as runs of one, runs shuffled."""
    if cfg.order == "ascending":
        def key(c):
            return (cfg.directions_deg.index(c[1]), c[0])
        return sorted(main, key=key) + sorted(extra, key=key)
    runs = []
    for d in cfg.directions_deg:
        cells = [c for c in main if c[1] == d]
        rng.shuffle(cells)
        runs += [cells[i:i + cfg.direction_run] for i in range(0, len(cells), cfg.direction_run)]
    runs += [[c] for c in extra]
    rng.shuffle(runs)
    return [c for run in runs for c in run]


def design(cfg: Config, fixed: dict | None = None) -> list[dict]:
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


def cell(r: dict) -> tuple:
    """(holder, angle, direction, drape) of a trial or a slot; (holder, "set", set, position) of a pre-impact one.
    Raises ValueError on a value that is not a number."""
    if str(r.get("position") or "") != "":
        return (r["holder"], "set", int(float(r.get("set") or r.get("replicate"))), int(float(r["position"])))
    return (r["holder"], int(float(r["angle_deg"])), int(float(r["direction_deg"])), int(float(r["drape"] or 0)))


def _cell_or_none(r: dict) -> tuple | None:
    """cell(r), or None for a row whose values cannot be read (a hand-edited log): such a row fills no slot."""
    try:
        return cell(r)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def changes(prev: dict | None, row: dict) -> list[str]:
    """Settings that differ from the previous trial (all of them for the first)."""
    if prev is None:
        return [n for n in SETTINGS if n in row]
    return [n for n in SETTINGS if n in row and n in prev and str(prev[n]) != str(row[n])]


def make_sequence(cfg: Config, done: Sequence[dict] = ()) -> list[dict]:
    """Sequence rows. done: trials already logged (open or ok), in execution order; their cells come
    first, the remaining ones follow in the seeded order, the holder in use first."""
    done = [t for t in done if _cell_or_none(t) is not None]
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


def match(sequence: Sequence[dict], trials: Sequence[dict]) -> tuple[dict, list]:
    """Slots filled by logged trials: each open or ok trial fills the first free slot with its cell.
    Returns ({slot: trial}, [trials outside the matrix]). A trial whose values cannot be read fills nothing
    (Run.check reports it)."""
    free = {}
    for r in sequence:
        c = _cell_or_none(r)
        if c is not None:
            free.setdefault(c, []).append(int(float(r["slot"])))
    filled, extra = {}, []
    for t in trials:
        if t["status"] not in (OPEN, OK):
            continue
        c = _cell_or_none(t)
        if c is None:
            continue
        slots = free.get(c)
        if slots:
            filled[slots.pop(0)] = t["trial"]
        else:
            extra.append(t["trial"])
    return filled, extra


# Files and log


def now(when: datetime | None = None) -> str:
    """ISO 8601 local time with its offset from UTC, to the second (a naive `when` is local time)."""
    return (when or datetime.now()).astimezone().isoformat(timespec="seconds")


def _retry(fn: Callable, name: str):
    """fn(), tried again while another program holds the file; then LogLocked."""
    for attempt in range(WRITE_RETRIES):
        try:
            return fn()
        except PermissionError as e:
            if attempt == WRITE_RETRIES - 1:
                raise LogLocked("log_locked", f"{name} is held by another program", file=name) from e
            time.sleep(WRITE_RETRY_S)
    return None


def _fsync_folder(folder: Path) -> None:
    """The new name made durable too, where the system allows it (not on Windows)."""
    if os.name == "nt":
        return
    with contextlib.suppress(OSError):
        fd = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_text_atomic(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    """Writes `text` to `path` in one step: a temporary file in the same folder, flushed to the disk, then put in
    place of the old one. A reader sees the old file or the new one. Raises LogLocked when the file stays held by
    another program."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _retry(lambda: os.replace(tmp, path), path.name)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_folder(path.parent)


def _decode(raw: bytes, name: str) -> str:
    """Text of a CSV as a spreadsheet program may save it: UTF-8 (with or without a byte order mark), else the
    Windows code page."""
    for encoding in ("utf-8-sig", "cp1252"):
        with contextlib.suppress(UnicodeDecodeError):
            return raw.decode(encoding)
    raise LogError("log_corrupt", f"{name}: not text", file=name)


def read_csv(path: Path | str, required: Sequence[str] = ()) -> list[dict]:
    """The rows of a CSV as dicts ([] when it does not exist). Read as written here or as a spreadsheet program
    saves it: byte order mark, ';' between fields, the Windows code page; short rows filled with ''; blank rows
    left out. Raises LogError log_corrupt when a column of `required` is missing or a row has more fields than the
    header, LogLocked when the file stays held by another program."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        raw = _retry(path.read_bytes, path.name)
    except FileNotFoundError:
        return []
    text = _decode(raw, path.name)
    if not text.strip():
        return []
    first = text.splitlines()[0]
    delimiter = ";" if first.count(";") > first.count(",") else ","
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    header = [(h or "").strip() for h in (reader.fieldnames or [])]
    reader.fieldnames = header
    lacking = [c for c in required if c not in header]
    if lacking:
        raise LogError("log_corrupt", f"{path.name}: no column {', '.join(lacking)}", file=path.name,
                       missing=lacking)
    rows = []
    for r in reader:
        if any(str(v or "").strip() for v in r.get(None) or []):    # blank extra fields (a trailing comma): no harm
            raise LogError("log_corrupt", f"{path.name}, line {reader.line_num}: more fields than columns",
                           file=path.name, line=reader.line_num)
        r.pop(None, None)
        row = {k: "" if v is None else v for k, v in r.items()}
        if any(v.strip() for v in row.values()):
            rows.append(row)
    return rows


def write_csv(path: Path | str, rows: Iterable[dict], columns: Sequence[str]) -> None:
    """Writes rows (dicts; keys outside `columns` are left out, None is written empty) in one step
    (write_text_atomic)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, columns, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows({k: "" if v is None else v for k, v in r.items()} for r in rows)
    write_text_atomic(path, buf.getvalue())


def write_templates(folder: Path) -> None:
    """Empty trial log and its data dictionary."""
    write_csv(folder / "trial_log_template.csv", [], LOG_FIELDS)
    columns = ("column", "unit", "filled", "description")
    write_csv(folder / "trial_log_dictionary.csv", [dict(zip(columns, c)) for c in LOG_COLUMNS], columns)


def sha256_file(path: Path | str) -> str:
    """SHA-256 of a file, hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(COPY_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def copy_verified(source: Path | str, dest: Path | str) -> tuple[int, str]:
    """Copies source to dest (a new name), flushed to the disk, then reads dest back: (bytes, SHA-256). Raises
    CopyFailed, with nothing left at dest, when the copy differs from what was read from the source."""
    source, dest = Path(source), Path(dest)
    h = hashlib.sha256()
    n = 0
    made = False                        # dest was made by this call: only then is it removed on a failure
    try:
        with open(source, "rb") as fi:
            with open(dest, "xb") as fo:
                made = True
                for block in iter(lambda: fi.read(COPY_BLOCK), b""):
                    h.update(block)
                    n += len(block)
                    fo.write(block)
                fo.flush()
                os.fsync(fo.fileno())
        digest = h.hexdigest()
        if dest.stat().st_size != n or sha256_file(dest) != digest:
            raise CopyFailed("copy_failed", f"{dest.name}: the copy differs from {source.name}", file=dest.name)
    except BaseException:
        if made:
            with contextlib.suppress(OSError):
                dest.unlink()
        raise
    return n, digest


def _place_new(part: Path, dest: Path) -> None:
    """Gives part its final name, never over a file already there (FileExistsError)."""
    if os.name == "nt":
        _retry(lambda: os.rename(part, dest), dest.name)       # Windows: rename never replaces
        return
    try:
        os.link(part, dest)                                     # fails when dest exists
        os.unlink(part)
    except FileExistsError:
        raise
    except OSError:                                             # no hard links on this file system
        if dest.exists():
            raise FileExistsError(dest) from None
        os.replace(part, dest)


def file_name(trial: str, kind: str, az: int | None = None, ref: bool = False, drape: bool = False) -> str:
    """ref: reference scan before the impact; drape: photo or video taken with the drape on."""
    stem = f"{trial}_{'ref_' if ref else ''}{'drape_' if drape else ''}{kind}" + ("" if az is None else f"_az{az}")
    return stem + SUFFIXES[kind][0]


def file_kind(name: str) -> str | None:
    """photo, video, highspeed or scan from a file name, whatever its case (None when none)."""
    stem = Path(name).stem.lower()
    return next((k for k in ("highspeed", "video", "photo", "scan") if f"_{k}" in stem), None)


EXIF_IFD = 0x8769               # the EXIF block of a photo...
EXPOSURE_PROGRAM = 34850        # ... and its ExposureProgram: 1 manual (the camera app's Pro mode), 2 automatic...
MANUAL_PROGRAM = 1


def deep_check_codes(cfg: Config, path: Path, sizes: dict | None = None) -> list[dict]:
    """Problems inside a photo or video, as codes: [{"code": "photo_size" | "photo_auto" | "video_size" |
    "video_short" | "video_fps" | "unreadable", "file", "got", "want"}] (empty when the file is fine). sizes: the
    image size of each camera mode's calibration ({"photo", "video"}, Run.camera_sizes): a photo or a trial video of
    another size (either way round) was taken in another mode than the one calibrated; a photo without a
    calibration is held against cfg.photo_px. photo_auto: the photo's exposure was not set by hand (its EXIF says so:
    the camera app left its Pro mode, as after a video), so it is not taken as the calibration's were."""
    kind = file_kind(path.name)
    sizes = sizes or {}
    if kind == "photo":
        from PIL import Image
        try:
            with Image.open(path) as im:
                size = im.size
                program = im.getexif().get_ifd(EXIF_IFD).get(EXPOSURE_PROGRAM)
        except Exception:   # noqa: BLE001  PIL raises many kinds for a file that is not an image
            return [{"code": "unreadable", "file": path.name, "got": None, "want": None}]
        want = sizes.get("photo") or cfg.photo_px
        out = []
        if sorted(size) != sorted(want):
            out.append({"code": "photo_size", "file": path.name, "got": [int(size[0]), int(size[1])],
                        "want": [int(want[0]), int(want[1])]})
        if isinstance(program, int) and program != MANUAL_PROGRAM:
            out.append({"code": "photo_auto", "file": path.name, "got": program, "want": MANUAL_PROGRAM})
        return out
    elif kind in ("video", "highspeed"):
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        cap.release()
        if not fps or frames <= 0:
            return [{"code": "unreadable", "file": path.name, "got": None, "want": None}]
        want = sizes.get("video")
        if kind == "video" and want and all(size) and sorted(size) != sorted(want):
            return [{"code": "video_size", "file": path.name, "got": size, "want": [int(want[0]), int(want[1])]}]
        if kind == "video" and frames / fps < cfg.video_min_s:
            return [{"code": "video_short", "file": path.name, "got": round(frames / fps, 1),
                     "want": cfg.video_min_s}]
        if kind == "highspeed" and fps < HIGHSPEED_FPS:
            return [{"code": "video_fps", "file": path.name, "got": round(fps), "want": HIGHSPEED_FPS}]
    return []


def deep_check(cfg: Config, path: Path, sizes: dict | None = None) -> list[tuple[str, str]]:
    """Problems inside a photo or video: pixel size, exposure mode, duration, frame rate, as (kind, plain
    sentence); sizes as in deep_check_codes."""
    out = []
    for p in deep_check_codes(cfg, path, sizes):
        if p["code"] == "unreadable":
            out.append(("unreadable", p["file"]))
        elif p["code"] in ("photo_size", "video_size"):
            out.append(("content", f"{p['file']}: {p['got'][0]} x {p['got'][1]} px, expected {p['want'][0]} x "
                                   f"{p['want'][1]}"))
        elif p["code"] == "photo_auto":
            out.append(("content", f"{p['file']}: exposure not set by hand (the camera app out of its Pro mode)"))
        elif p["code"] == "video_short":
            out.append(("content", f"{p['file']}: {p['got']:.1f} s, expected at least {p['want']:g} s"))
        else:
            out.append(("content", f"{p['file']}: {p['got']:.0f} fps, expected at least {p['want']}"))
    return out


# Logged values


def _text(value: object) -> str:
    """A value as the log writes it: '' for None, 1 or 0 for a truth value."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value).strip()


def _number(text: str) -> float | None:
    """A finite number (a decimal comma is read too), or None."""
    try:
        v = float(text.replace(",", "."))
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def check_value(field: str, value: object, holders: Sequence[str] | None = None) -> str:
    """The value of a log field as it is logged, checked: numbers, flags 0/1, the status, a holder of the
    configuration, ISO days and times, trial ids, the reasons for reference scans; free text with a leading
    apostrophe when it would start a spreadsheet formula. An empty value is allowed but for trial, phase and status.

    Raises:
        LogError: bad_field, with the field and the value.
    """
    text = _text(value)

    def bad(why: str = "") -> LogError:
        return LogError("bad_field", f"{field}: {text!r} is not valid" + (f" ({why})" if why else ""), field=field,
                        value=text)

    if field not in LOG_FIELDS:
        raise bad("no such column")
    if text == "":
        if field in ("trial", "phase", "status"):
            raise bad("empty")
        return text
    if field == "trial" and not TRIAL_RE.fullmatch(text):
        raise bad()
    if field == "phase" and text not in PREFIX:
        raise bad()
    if field == "status" and text not in (OPEN, OK, VOID):
        raise bad()
    if field == "light" and text not in LIGHTS:
        raise bad("one of " + ", ".join(LIGHTS))
    if field == "holder" and not (text in holders if holders else HOLDER_RE.fullmatch(text)):
        raise bad("not a holder of the configuration")
    if field in FLAG_FIELDS:
        v = _number(text)
        if v not in (0.0, 1.0):
            raise bad("0 or 1")
        return str(int(v))
    if field in NUMERIC_FIELDS:
        v = _number(text)
        if v is None:
            raise bad("a number")
        if field in ("washers_per_side", "set", "position") and (v < 0 or not v.is_integer()):
            raise bad("a whole number")
        if field in ("swing_mass_g",) and v <= 0:
            raise bad("above 0")
        if field == "cap_shift_mm" and v < 0:
            raise bad("0 or more")
        return text.replace(",", ".")
    if field == "session":
        try:
            date.fromisoformat(text)
        except ValueError:
            raise bad("YYYY-MM-DD") from None
    if field in TIME_FIELDS:
        try:
            datetime.fromisoformat(text)
        except ValueError:
            raise bad("ISO 8601") from None
    if field == "reference" and not REASONS_RE.fullmatch(text):
        raise bad()
    if field == "pre_scan" and not PRE_SCAN_RE.fullmatch(text):
        raise bad()
    if field == "config_version" and not (text.isdigit() and int(text) >= 1):
        raise bad("a whole number from 1")
    if field in TEXT_FIELDS:
        if len(text) > MAX_TEXT:
            raise bad(f"at most {MAX_TEXT} characters")
        if text.startswith(FORMULA_START):
            text = "'" + text
    return text


def _append_note(old: str, new: str) -> str:
    """Notes added to those already logged (a note already there is not added twice)."""
    if not old or old == new:
        return new
    if not new or new in old.split(NOTE_JOIN):
        return old
    return old + NOTE_JOIN + new


def _normalise(rows: list[dict], numeric: Sequence[str], days: Sequence[str] = ()) -> list[dict]:
    """Values as a spreadsheet program may have saved them read back as written here: decimal commas in numbers,
    days as dd/mm/yyyy."""
    for r in rows:
        for k in numeric:
            v = r.get(k)
            if v and COMMA_NUMBER_RE.fullmatch(v):
                r[k] = v.replace(",", ".")
        for k in days:
            m = DAY_FIRST_RE.fullmatch(r.get(k) or "")
            if m:
                r[k] = f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return rows


def _process_alive(pid: int) -> bool:
    """Whether the process `pid` runs (asked, never signalled: on Windows os.kill(pid, 0) would end it). A process
    that cannot be asked about (another user's) counts as running."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == 5                 # ERROR_ACCESS_DENIED: it exists
    try:
        code = wintypes.DWORD()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        return not ok or code.value == 259                  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


class PhaseLock:
    """One writer at a time in a phase folder, across programs (the app and the command line): a lock file made
    with an exclusive create, held while the outermost holder is inside. Re-entrant in a thread; threads of one
    process share it through the file's path."""

    _guard = threading.Lock()
    _shared: ClassVar[dict[str, tuple[threading.RLock, list[int]]]] = {}

    def __init__(self, folder: Path | str, name: str = LOCK_NAME) -> None:
        self.path = Path(folder) / name
        key = os.path.normcase(os.path.abspath(self.path))
        with PhaseLock._guard:
            if key not in PhaseLock._shared:
                PhaseLock._shared[key] = (threading.RLock(), [0])
            self._rlock, self._depth = PhaseLock._shared[key]

    def __enter__(self) -> PhaseLock:
        self._rlock.acquire()
        try:
            if self._depth[0] == 0:
                self._take()
        except BaseException:
            self._rlock.release()
            raise
        self._depth[0] += 1
        return self

    def __exit__(self, *exc: object) -> None:
        self._depth[0] -= 1
        try:
            if self._depth[0] == 0:
                try:
                    _retry(self.path.unlink, self.path.name)
                except FileNotFoundError:
                    pass
                except (OSError, TrialsError) as e:
                    # never raised: the write it guarded is done; the file left names this process, so the next
                    # take (here or after a restart) knows it for a stale one
                    _LOGGER.warning("%s: lock not removed (%s), taken for stale next time", self.path, e)
        finally:
            self._rlock.release()

    def _stale(self) -> bool:
        """A lock nobody holds: left by this process (no thread of it is inside, or it would not be asked), by a
        program that has ended, or older than LOCK_STALE_S."""
        try:
            pid = int(self.path.read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            pid = None
        if pid is not None and (pid == os.getpid() or not _process_alive(pid)):
            return True
        try:
            return time.time() - self.path.stat().st_mtime > LOCK_STALE_S
        except OSError:
            return False

    def _take(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except (FileExistsError, PermissionError):
                if self._stale():
                    _LOGGER.warning("%s: a lock left by a program that stopped, removed", self.path)
                    with contextlib.suppress(OSError):
                        self.path.unlink()
                    continue
                if time.monotonic() > deadline:
                    raise LogLocked("log_locked", f"{self.path.name}: another program is writing",
                                    file=self.path.name) from None
                time.sleep(LOCK_POLL_S)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()} {now()}\n")
            return


class Log:
    """trial_log.csv, one row per trial, and trial_log_edits.csv, the audit trail."""

    def __init__(self, folder: Path | str, holders: Sequence[str] | None = None) -> None:
        self.path = Path(folder) / LOG_NAME
        self.edits_path = Path(folder) / EDITS_NAME
        self.manifest_path = Path(folder) / MANIFEST_NAME
        self.holders = tuple(holders) if holders else None      # set by Run from its configuration
        self.lock = PhaseLock(folder)

    def rows(self) -> list[dict]:
        """Every logged trial, with every column of LOG_FIELDS (an older log's missing ones empty)."""
        rows = read_csv(self.path, required=("trial", "status"))
        for r in rows:
            for c in LOG_FIELDS:
                r.setdefault(c, "")
        return _normalise(rows, NUMERIC_FIELDS, ("session",))

    def edits(self) -> list[dict]:
        return read_csv(self.edits_path, required=EDIT_FIELDS)

    def manifest(self) -> list[dict]:
        return read_csv(self.manifest_path, required=MANIFEST_FIELDS)

    def row(self, trial: str) -> dict:
        """The log row of a trial. Raises NotFound (a KeyError) unknown_trial."""
        for r in self.rows():
            if r["trial"] == trial:
                return r
        raise NotFound("unknown_trial", trial, trial=trial)

    @staticmethod
    def _columns(rows: Sequence[dict]) -> list[str]:
        """LOG_FIELDS, then any column a person added to the log (kept)."""
        extra = []
        for r in rows:
            extra += [k for k in r if k not in LOG_FIELDS and k not in extra and k]
        return list(LOG_FIELDS) + extra

    def _update(self, trial: str, values: dict) -> dict:
        """Sets values of one trial in one write; returns the row as it was."""
        with self.lock:
            rows = self.rows()
            r = next((r for r in rows if r["trial"] == trial), None)
            if r is None:
                raise NotFound("unknown_trial", trial, trial=trial)
            old = dict(r)
            r.update(values)
            write_csv(self.path, rows, self._columns(rows))
            return old

    def _audit(self, trial: str, changed: Sequence[tuple[str, str, str]], reason: str, when=None) -> None:
        """Rows of the audit trail (field, old, new), written before the log changes. A change whose last row is
        already there (the log, still holding the old value, could not be written then) is not added again: a
        retry leaves one row per change, never one per attempt."""
        with self.lock:
            stamp = now(when)
            before = self.edits()
            last = {}
            for e in before:
                if e["trial"] == trial:
                    last[e["field"]] = (e["old"], e["new"])
            rows = [dict(zip(EDIT_FIELDS, (stamp, trial, f, old, new, reason)))
                    for f, old, new in changed if last.get(f) != (old, new)]
            if rows:
                write_csv(self.edits_path, before + rows, EDIT_FIELDS)

    def add(self, row: dict) -> None:
        """A new trial; its values are checked first."""
        clean = {c: check_value(c, row.get(c, ""), self.holders) for c in LOG_FIELDS}
        with self.lock:
            rows = self.rows()
            if any(r["trial"] == clean["trial"] for r in rows):
                raise LogError("exists", f"{clean['trial']} is already in the log", trial=clean["trial"])
            rows.append(clean)
            write_csv(self.path, rows, self._columns(rows))

    def record(self, trial: str, values: dict, status: str | None = None, change_reason: str | None = None,
               when=None) -> None:
        """Sets fields of one trial, and its status, in one write. Empty fields are set; notes are added to those
        already logged; a field logged with another value is changed only on an open trial and only when
        change_reason is given, and that change goes to the audit trail first. On a trial that is no longer open
        every change goes to the audit trail first.

        Raises:
            LogError: bad_field (a value that is not valid, or one already logged that only edit() may change).
            NotFound: unknown_trial.
        """
        clean = {k: check_value(k, v, self.holders) for k, v in values.items()}
        if status is not None:
            clean["status"] = check_value("status", status)
        with self.lock:
            r = self.row(trial)
            is_open = r["status"] == OPEN
            new, audit = {}, []
            for k, given in clean.items():
                old = r[k]
                v = _append_note(old, given) if k == "notes" else given
                if k == "notes" and v != old:
                    v = check_value("notes", v, self.holders)   # the notes joined: still within MAX_TEXT
                if v == old:
                    continue
                if old and k not in ("notes", "status"):
                    if not (is_open and change_reason):
                        raise LogError("bad_field", f"{trial} {k} is already {old!r}: change it with edit() and a "
                                                    f"reason", trial=trial, field=k, value=v)
                    audit.append((k, old, v))
                elif not is_open:
                    audit.append((k, old, v))
                new[k] = v
            if not new:
                return
            if audit:
                why = change_reason if is_open else "added after the trial was closed"
                self._audit(trial, audit, why or "changed", when)
            self._update(trial, new)

    def fill(self, trial: str, **values: object) -> None:
        """Sets empty fields; a value already logged changes only through edit(). Notes are added to those already
        logged. On a trial already closed the change goes to the audit trail as well."""
        self.record(trial, values)

    def edit(self, trial: str, field: str, value: object, reason: str, when=None) -> None:
        """Changes a logged value and records old value, new value, time and reason (the audit row first)."""
        self.edit_many(trial, {field: value}, reason, when)

    def edit_many(self, trial: str, values: dict, reason: str, when=None) -> None:
        """edit() of several fields of one trial: audit rows first, then one write of the log."""
        reason = _text(reason)
        if not reason:
            raise LogError("reason", "an edit needs a reason")
        reason = check_value("notes", reason)
        for field in values:
            if field not in LOG_FIELDS or field in NOT_EDITABLE:
                raise LogError("bad_field", f"{field} cannot be edited", field=field)
        clean = {k: check_value(k, v, self.holders) for k, v in values.items()}
        with self.lock:
            rows = self.rows()
            r = next((x for x in rows if x["trial"] == trial), None)
            if r is None:
                raise NotFound("unknown_trial", trial, trial=trial)
            if clean.get("status") == OPEN and r["status"] != OPEN:
                others = [x["trial"] for x in rows if x["status"] == OPEN and x["trial"] != trial]
                if others:
                    raise LogError("trial_open", f"{others[0]} is open: close or void it first", trial=others[0])
            changed = [(k, r[k], v) for k, v in clean.items() if r[k] != v]
            if not changed:
                return
            self._audit(trial, changed, reason, when)
            self._update(trial, {k: v for k, _, v in changed})

    def set_status(self, trial: str, status: str, reason: str = "", when=None) -> None:
        """Open trials change status freely; a closed one only through an edit with a reason."""
        if self.row(trial)["status"] == OPEN:
            self.record(trial, {}, status=status)
        else:
            self.edit(trial, "status", status, reason, when)

    def add_manifest(self, trial: str, name: str, size: int, digest: str) -> None:
        """A raw file filed: its size and SHA-256 in the manifest."""
        with self.lock:
            rows = self.manifest() + [dict(zip(MANIFEST_FIELDS, (now(), trial, name, str(size), digest)))]
            write_csv(self.manifest_path, rows, MANIFEST_FIELDS)


# Steps


@dataclass(frozen=True)
class Step:
    key: str  # stable, can index a translation
    text: str
    files: tuple = ()  # names expected in the trial folder after this step (default suffix)
    fields: tuple = ()  # log fields this step fills


def pendulum_text(row: dict, ch: set) -> str:
    parts = [("direction_deg", f"direction {row['direction_deg']}°"), ("angle_deg", f"hole {row['angle_deg']}°"),
             ("washers_per_side", f"{row['washers_per_side']} washers per side")]
    text = "Set the pendulum: " + ", ".join(t + (" (change)" if k in ch else "") for k, t in parts) + "."
    if "direction_deg" in ch:
        text += " To turn the platter pull its pin, turn it and pin it again."
    return text


def trial_steps(cfg: Config, row: dict, prev: dict | None = None) -> list[Step]:
    """Steps of one trial in order."""
    if cfg.phase == PRE_IMPACT:
        from pre_impact import position_steps      # a position of the marker: no pendulum, no impact
        return position_steps(cfg, row, prev)
    t, holder = row["trial"], row["holder"]
    reasons = {r for r in (row.get("reference") or "").split(";") if r}
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
                      "the pendulum path. It stays on for the first photos and video after the impact."))
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

    def scans():
        for az in cfg.scan_az:
            s.append(Step(f"scan_az{az}", f"Scan: arm at {az}°, panels off. Scan and export.",
                          (file_name(t, "scan", az),)))
        s.append(Step("spheres", "Six spheres found? Cap shift since the last reference, in mm.", fields=ONLINE))

    def views(with_drape=False):
        tag, what = ("drape_", " with the drape") if with_drape else ("", "")
        for az in cfg.photo_az:
            s.append(Step(f"{tag}photo_az{az}", f"Photo{what}: arm at {az}°, panels on. Shoot from the PC.",
                          (file_name(t, "photo", az, drape=with_drape),)))
        a, b = cfg.video_az
        s.append(Step(f"{tag}video", f"Video{what}: lift the index plunger and sweep the arm from end stop to end "
                      f"stop ({a}° to {b}°) in about {round((b - a) / VIDEO_SPEED)} s, then back to "
                      f"{cfg.photo_az[0]}°. Then the camera app back in its Pro mode.",
                      (file_name(t, "video", drape=with_drape),)))

    if drape:
        views(with_drape=True)
        s.append(Step("drape_off", "Take the drape off slowly, without touching the marker or the pendulum."))
        views()
        scans()
    else:
        scans()
        views()
    s.append(Step("close", "Check second hit, cap replaced and notes, then close the trial.",
                  fields=("second_hit", "cap_replaced", "time_end")))
    return s


# A phase driven step by step


class Run:
    """One acquisition phase (campaign or pilot): configuration, sequence, log and trial folders.

    The configuration and the sequence are read again when their files change (another program, the command
    line); a missing sequence is made again from the configuration, never the configuration from the defaults.

    Versions. Each configuration written has a version (1 for the first, one more at each change; the earlier ones
    stay in its history) and each trial logs the version it started under (config_version): the steps and the
    files a trial expects follow its own configuration, so a trial closed under one stays complete after the
    configuration changed (a trial logged before versions were kept follows today's).

    Camera sizes. A photo or video filed is checked against the size of the camera mode's calibration when one is
    kept in `calibration` (intrinsics_photo.json, intrinsics_video.json; by default beside the protocol folder),
    else a photo against the configuration's photo_px.

    Run("pre_impact") is a pre_impact.PreImpactRun: the sets of marker positions, with their own sequence and steps.
    """

    pendulum = True                 # the trials weigh the swinging part and release the pendulum
    required = REQUIRED             # log fields a closed trial must have
    online = ONLINE                 # values the scans give while the phase runs
    sequence_fields = SEQUENCE_FIELDS
    sequence_numeric = SEQUENCE_NUMERIC

    def __new__(cls, phase: str = "campaign", *args, **kwargs):
        if cls is Run and phase == PRE_IMPACT:
            from pre_impact import PreImpactRun
            return super().__new__(PreImpactRun)
        return super().__new__(cls)

    def _make_sequence(self, cfg: Config, done: Sequence[dict]) -> list[dict]:
        return make_sequence(cfg, done)

    def __init__(self, phase: str = "campaign", raw: Path | str = RAW, protocol: Path | str = PROTOCOL,
                 calibration: Path | str | None = None) -> None:
        self.phase = phase
        self.folder = Path(raw) / phase
        self.protocol = Path(protocol)
        self.calibration = Path(calibration) if calibration is not None else self.protocol.parent / "calibration"
        self.log = Log(self.folder)
        self.config_path = self.protocol / f"config_{phase}.json"
        self.sequence_path = self.protocol / f"sequence_{phase}.csv"
        self._protocol_lock = PhaseLock(self.protocol, f"protocol_{phase}.lock")
        self._config = default_config(phase)
        self._version = 1                   # the configuration's version (config_version of the trials it starts)
        self._versions: dict[int, Config] = {}     # every version of the configuration, today's included
        self._sizes: dict[str, tuple] = {}  # mode -> (file stamp, size) of the calibrations read (camera_sizes)
        self._sequence: list[dict] = []
        self._stamp = None
        self._load()

    # -- configuration and sequence
    def _mtimes(self) -> tuple:
        def stamp(p):
            try:
                st = p.stat()
                return (st.st_mtime_ns, st.st_size)
            except OSError:
                return None
        return stamp(self.config_path), stamp(self.sequence_path)

    def _read_config(self) -> Config:
        try:
            doc = json.loads(_retry(lambda: self.config_path.read_text(encoding="utf-8-sig"), self.config_path.name))
            cfg = config_from_dict(doc["config"])
        except (ValueError, KeyError, TypeError) as e:
            raise LogError("config_corrupt", f"{self.config_path.name} cannot be read: {e}",
                           file=self.config_path.name) from e
        if cfg.phase != self.phase:
            raise LogError("config_corrupt", f"{self.config_path.name} is for {cfg.phase}", file=self.config_path.name)
        self._read_versions(doc, cfg)
        return cfg

    @staticmethod
    def _doc_version(doc: dict) -> int:
        """The version of a configuration file's own configuration: its "version", else one more than its history
        (a file written before versions were kept)."""
        v = doc.get("version")
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 1 else len(doc.get("history") or []) + 1

    def _read_versions(self, doc: dict, cfg: Config) -> None:
        """Every version of the configuration from its file: today's and those of its history (an entry that cannot
        be read is left out: its trials follow today's)."""
        version = self._doc_version(doc)
        versions = {}
        for i, h in enumerate(doc.get("history") or []):
            try:
                v = h.get("version")
                v = v if isinstance(v, int) and not isinstance(v, bool) and v >= 1 else i + 1
                old = config_from_dict(h["config"])
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if old.phase == self.phase and v < version:
                versions[v] = old
        versions[version] = cfg
        self._version, self._versions = version, versions

    @property
    def config_version(self) -> int:
        """The version of today's configuration (config_version of the trials started now)."""
        self._refresh()
        return self._version

    def config_for(self, row: dict) -> Config:
        """The configuration a logged trial follows: that of its config_version, else (logged before versions were
        kept, or a version no longer in the history) today's."""
        self._refresh()
        v = str(row.get("config_version") or "")
        return self._versions.get(int(v), self._config) if v.isdigit() else self._config

    def camera_sizes(self) -> dict[str, list[int] | None]:
        """{"photo", "video"}: the image size of each camera mode's calibration kept in `calibration`, or None (no
        calibration, or one that cannot be read). Read again when its file changes."""
        out = {}
        for mode in ("photo", "video"):
            p = self.calibration / f"intrinsics_{mode}.json"
            try:
                st = p.stat()
                stamp = (st.st_mtime_ns, st.st_size)
            except OSError:
                out[mode] = None
                continue
            got = self._sizes.get(mode)
            if got is None or got[0] != stamp:
                size = None
                try:
                    s = json.loads(p.read_text(encoding="utf-8"))["image_size"]
                    size = [int(s[0]), int(s[1])] if len(s) == 2 and int(s[0]) > 0 and int(s[1]) > 0 else None
                except (OSError, ValueError, KeyError, TypeError, IndexError):
                    _LOGGER.warning("%s cannot be read: its size is not checked", p.name)
                got = self._sizes[mode] = (stamp, size)
            out[mode] = got[1]
        return out

    def file_problems(self, trial: str, path: Path | str) -> list[dict]:
        """deep_check_codes of a file for a trial: with the trial's own configuration and the calibrations' sizes."""
        return deep_check_codes(self.config_for(self.log.row(trial)), Path(path), self.camera_sizes())

    def _read_sequence(self) -> list[dict]:
        return _normalise(read_csv(self.sequence_path, required=self.sequence_fields), self.sequence_numeric)

    def _load(self) -> None:
        if self.config_path.exists():
            self._config = self._read_config()
            if self.sequence_path.exists():
                self._sequence = self._read_sequence()
            else:
                self.configure(self._config)        # the sequence alone is made again
        else:
            self.configure(default_config(self.phase))
        self.log.holders = self._config.holders
        self._stamp = self._mtimes()

    def _refresh(self) -> None:
        if self._mtimes() != self._stamp:
            self._load()

    @property
    def config(self) -> Config:
        self._refresh()
        return self._config

    @property
    def sequence(self) -> list[dict]:
        self._refresh()
        return self._sequence

    def configure(self, cfg: Config) -> None:
        """Sets the configuration (matrix, repetitions, drape trials, seed...) and regenerates the
        sequence: trials already logged keep their cells, the others follow the new seeded order.
        An unchanged configuration keeps the sequence as it is. The sequence is written first, the
        configuration last, each in one step."""
        if cfg.phase != self.phase:
            raise ValueError(f"configuration for {cfg.phase}, run for {self.phase}")
        bad = [h for h in cfg.holders if not HOLDER_RE.fullmatch(str(h))]
        if bad or not cfg.holders:      # a holder the log would take but its check refuse: trials never closed
            raise LogError("bad_field", f"holder names {bad or 'none'}", field="holder")
        new = json.loads(json.dumps(asdict(cfg)))
        with self._protocol_lock:
            old = None
            if self.config_path.exists():
                try:
                    old = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
                    old["config"]
                except (ValueError, KeyError, TypeError) as e:
                    raise LogError("config_corrupt", f"{self.config_path.name} cannot be read",
                                   file=self.config_path.name) from e
            if old is None or old["config"] != new or not self.sequence_path.exists():
                done = [r for r in self.log.rows() if r["status"] in (OPEN, OK)]
                write_csv(self.sequence_path, self._make_sequence(cfg, done), self.sequence_fields)
            if old is None or old["config"] != new:
                was = self._doc_version(old) if old else 0
                history = old.get("history", []) + [{"written": old.get("written", ""), "version": was,
                                                     "config": old["config"]}] if old else []
                doc = {"written": now(), "version": was + 1, "config": new, "history": history}
                write_text_atomic(self.config_path, json.dumps(doc, indent=1) + "\n")
                self._read_versions(doc, cfg)
            elif old is not None:
                self._read_versions(old, cfg)
            self._config = cfg
            self._sequence = self._read_sequence()
            self.log.holders = cfg.holders
            self._stamp = self._mtimes()

    def pending(self) -> list[dict]:
        filled, _ = match(self.sequence, self.log.rows())
        return [s for s in self._sequence if int(float(s["slot"])) not in filled]

    def new_id(self) -> str:
        p = PREFIX[self.phase]
        n = [int(r["trial"][1:]) for r in self.log.rows() if r["trial"][:1] == p and r["trial"][1:].isdigit()]
        if self.folder.exists():
            n += [int(d.name[1:]) for d in self.folder.glob(p + "*") if d.is_dir() and d.name[1:].isdigit()]
        return f"{p}{max(n, default=0) + 1:03d}"

    def find(self, trial: str, name: str) -> list[Path]:
        """Files of the trial folder with the stem of name (case does not count) and a suffix allowed for its kind."""
        folder, stem = self.folder / trial, Path(name).stem.lower()
        allowed = SUFFIXES[file_kind(name)]
        if not folder.exists():
            return []
        return sorted(p for p in folder.iterdir()
                      if p.is_file() and p.stem.lower() == stem and p.suffix.lower() in allowed)

    def has_scans(self, trial: str) -> bool:
        """The trial has every post-impact scan of its own configuration."""
        try:
            cfg = self.config_for(self.log.row(trial))
        except NotFound:
            cfg = self.config
        return all(self.find(trial, file_name(trial, "scan", az)) for az in cfg.scan_az)

    def next(self, today: date | None = None, reference: str | None = None, settings: dict | None = None) -> dict | None:
        """The trial to do now: the open one (resume, with the index of the step to resume from) or a
        plan for the next pending slot of the holder in use; None when the sequence is complete.
        reference: a reason for reference scans chosen by the operator (e.g. "manual" after a new cap).
        settings: holder, angle_deg, direction_deg and drape chosen by the operator instead of the next
        slot (an extra trial); it fills a pending slot with the same cell, if any, or stays outside.
        The plan's warnings are plain sentences; warning_codes the same as codes with their values."""
        rows = self.log.rows()
        open_rows = [r for r in rows if r["status"] == OPEN]
        if open_rows:
            r = open_rows[0]
            warnings, codes = [], []
            if len(open_rows) > 1:
                ids = [x["trial"] for x in open_rows]
                warnings.append(f"several trials are open ({', '.join(ids)}): close or void all but one")
                codes.append({"code": "several_open", "trials": ids})
            return {**r, "resume": True, "step": self.resume_step(r["trial"]), "changes": [], "warnings": warnings,
                    "warning_codes": codes}
        pending = self.pending()
        prev = rows[-1] if rows else None
        if settings:
            s = {k: str(settings[k]) for k in ("holder", "angle_deg", "direction_deg")}
            s["drape"] = str(int(settings.get("drape") or 0))
            wanted = _cell_or_none(s)
            s["slot"] = next((x["slot"] for x in pending if wanted is not None and _cell_or_none(x) == wanted), "")
        elif not pending:
            return None
        else:
            s = next((x for x in pending if prev and x["holder"] == prev["holder"]), pending[0])
        today = (today or date.today()).isoformat()
        plan = {"trial": self.new_id(), "phase": self.phase, "session": today, "holder": s["holder"],
                "angle_deg": s["angle_deg"], "direction_deg": s["direction_deg"],
                "washers_per_side": str(self._config.washers_per_side), "drape": s["drape"], "slot": s["slot"]}
        reasons, warnings, codes = [], [], []
        if prev is None:
            reasons.append("first")
        else:
            shift = _number(prev["cap_shift_mm"]) if prev["cap_shift_mm"] else None
            if prev["holder"] != s["holder"]:
                reasons.append("holder")
            if not self.has_scans(prev["trial"]):
                reasons.append("no_scan")
            elif shift is None:
                warnings.append(f"cap shift of {prev['trial']} unknown: check the cap, ask for a reference if it moved")
                codes.append({"code": "cap_unknown", "trial": prev["trial"]})
            elif shift > self._config.reposition_mm:
                reasons.append("reposition")
            if prev["session"] != today:
                reasons.append("session")
        if reference:
            reasons.append(reference)
        plan["reference"] = ";".join(dict.fromkeys(reasons))
        plan["pre_scan"] = f"{plan['trial']}_ref" if reasons else prev["trial"]
        same_mass = prev is not None and prev["washers_per_side"] == plan["washers_per_side"]
        plan["swing_mass_g"] = prev["swing_mass_g"] if same_mass else ""
        plan.update(resume=False, step=0, changes=changes(prev, plan), warnings=warnings, warning_codes=codes)
        return plan

    def start(self, plan: dict, when: datetime | None = None, **values: object) -> dict:
        """Opens a planned trial: log row with status open, and its folder. Raises LogError trial_open when a trial
        is open already, bad_field for a value that is not valid. The trial logs the version of the configuration it
        starts under (config_version): its steps and files follow it from then on."""
        row = {**{k: plan.get(k, "") for k in LOG_FIELDS}, **values, "time_start": now(when), "status": OPEN,
               "config_version": str(self.config_version)}
        with self.log.lock:
            others = [r["trial"] for r in self.log.rows() if r["status"] == OPEN]
            if others:
                raise LogError("trial_open", f"{others[0]} is open: close or void it first", trial=others[0])
            self.log.add(row)
        (self.folder / plan["trial"]).mkdir(parents=True, exist_ok=True)
        return self.log.row(plan["trial"])

    def steps(self, trial: str) -> list[Step]:
        """The steps of a logged trial. Raises NotFound unknown_trial."""
        rows = self.log.rows()
        i = next((i for i, r in enumerate(rows) if r["trial"] == trial), None)
        if i is None:
            raise NotFound("unknown_trial", trial, trial=trial)
        return trial_steps(self.config_for(rows[i]), rows[i], rows[i - 1] if i else None)

    def _step(self, trial: str, key: str) -> Step:
        step = next((s for s in self.steps(trial) if s.key == key), None)
        if step is None:
            raise NotFound("unknown_step", f"{trial} has no step {key}", trial=trial, step=key)
        return step

    def step_done(self, row: dict, step: Step) -> bool:
        return all(self.find(row["trial"], n) for n in step.files) and all(row[f] for f in step.fields)

    def resume_step(self, trial: str) -> int:
        """Index of the first step to do: the one after the last step with all its files and fields."""
        row, last = self.log.row(trial), -1
        for i, s in enumerate(self.steps(trial)):
            if (s.files or s.fields) and self.step_done(row, s):
                last = i
        return last + 1

    def target(self, trial: str, name: str, suffix: str | None = None) -> Path:
        """Path for a file of the trial; suffix of the source file if it differs from the default."""
        kind = file_kind(name)
        suffix = (suffix or Path(name).suffix).lower()
        if kind is None or suffix not in SUFFIXES[kind]:
            raise ValueError(f"{name} with suffix {suffix}: not a {kind} file")
        return self.folder / trial / (Path(name).stem + suffix)

    def store(self, trial: str, name: str, source: Path | str, move: bool = False) -> Path:
        """Files a photo, video or scan under its name; never overwrites a raw file (FileClash exists). The copy is
        made beside its final name, checked against the source (size and SHA-256) and only then named; its size
        and hash go to the manifest; with move the source is deleted last. CopyFailed when the copy differs."""
        source = Path(source)
        dest = self.target(trial, name, source.suffix)
        if self.find(trial, name):
            raise FileClash("exists", f"{self.find(trial, name)[0].name} is already there",
                            file=self.find(trial, name)[0].name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + PART_SUFFIX)
        with contextlib.suppress(FileNotFoundError):
            part.unlink()                       # a copy cut short earlier
        size, digest = copy_verified(source, part)
        try:
            with self.log.lock:
                if self.find(trial, name):
                    raise FileClash("exists", f"{self.find(trial, name)[0].name} is already there",
                                    file=self.find(trial, name)[0].name)
                try:
                    _place_new(part, dest)
                except FileExistsError:
                    raise FileClash("exists", f"{dest.name} is already there", file=dest.name) from None
                _fsync_folder(dest.parent)
                try:
                    self.log.add_manifest(trial, dest.name, size, digest)
                except (OSError, TrialsError) as e:
                    # the file is filed, checked: a failure now would leave it there and have the step done again;
                    # without its manifest row, the deep check has no size to compare it with
                    _LOGGER.warning("%s filed, but its manifest row was not written: %s", dest.name, e)
        except BaseException:
            with contextlib.suppress(OSError):
                part.unlink()
            raise
        if move:
            try:
                source.unlink()
            except OSError as e:
                _LOGGER.warning("%s filed, but its source %s could not be deleted: %s", dest.name, source, e)
        return dest

    def missing(self, trial: str, deep: bool = False) -> list[tuple[str, str]]:
        """What is missing or wrong for a trial, as (kind, item): missing, duplicate, empty,
        unexpected, unreadable, content or changed (deep only: the size differs from the manifest) for files;
        log for required log fields; bad_field for a logged value that is not valid; online for the values
        computed from the scans. Void trials only report unexpected files."""
        row = self.log.row(trial)
        names = [n for s in self.steps(trial) for n in s.files]
        out = []
        if row["status"] != VOID:
            sizes = {m["file"].lower(): m["bytes"] for m in self.log.manifest() if m["trial"] == trial} if deep else {}
            for n in names:
                hits = self.find(trial, n)
                if not hits:
                    out.append(("missing", n))
                elif len(hits) > 1:
                    out.append(("duplicate", ", ".join(h.name for h in hits)))
                elif hits[0].stat().st_size == 0:
                    out.append(("empty", hits[0].name))
                elif deep:
                    want = sizes.get(hits[0].name.lower())
                    if want and want != str(hits[0].stat().st_size):
                        out.append(("changed", hits[0].name))
                    out += deep_check(self.config_for(row), hits[0], self.camera_sizes())
            out += [("log", f) for f in self.required if not row[f]]
            out += [("online", f) for f in self.online if not row[f]]
        for f in LOG_FIELDS:
            try:
                check_value(f, row[f])          # holders of past trials need not be in today's configuration
            except LogError:
                out.append(("bad_field", f))
        expected = {Path(n).stem.lower(): n for n in names}
        folder = self.folder / trial
        for p in sorted(folder.iterdir()) if folder.exists() else ():
            if p.name in IGNORED:
                continue
            n = expected.get(p.stem.lower())
            if not p.is_file() or n is None or p.suffix.lower() not in SUFFIXES[file_kind(n)]:
                out.append(("unexpected", p.name))
        return out

    def confirm(self, trial: str, key: str, when: datetime | None = None, **values: object) -> None:
        """Closes a step: checks its files and fills its log fields, in one write; "close" sets status ok only if
        nothing required is missing (online values may still be empty). Confirming again is safe: the times are
        set only when empty, a value given again unchanged changes nothing, a new value on an open trial goes to
        the audit trail first, and notes given at close are added to those already logged."""
        step = self._step(trial, key)
        lacking = [n for n in step.files if not self.find(trial, n)]
        if lacking:
            raise FilesMissing("missing_file", ", ".join(lacking), files=lacking)
        values = {k: v for k, v in values.items() if v is not None}
        with self.log.lock:
            row = self.log.row(trial)
            defaults = {}
            if key == "release":
                defaults = {"time_impact": now(when)}
            if key == "close":
                if row["status"] == VOID:
                    raise LogError("not_open", f"{trial} is void", trial=trial)
                defaults = {"second_hit": 0, "cap_replaced": 0, "time_end": now(when)}
            for k, v in defaults.items():
                if k not in values and not row[k]:
                    values[k] = v
            if key == "close":
                problems = [p for p in self.missing(trial)
                            if p[0] not in ("online", "unexpected") and not (p[0] == "log" and p[1] in values)]
                if problems:
                    raise LogError("cannot_close", f"{trial} cannot be closed: "
                                   + "; ".join(f"{k} {v}" for k, v in problems), trial=trial,
                                   problems=[list(p) for p in problems])
            status = OK if key == "close" and row["status"] == OPEN else None
            if values or status:
                self.log.record(trial, values, status=status, change_reason=f"confirmed again at step {key}",
                                when=when)

    def void(self, trial: str, reason: str, when: datetime | None = None) -> None:
        """Marks a trial not valid; its files stay and its cell goes back into the sequence. An open trial is voided
        in one write; a closed one through edits with the reason, in the audit trail first. Voiding again with the
        same reason changes nothing."""
        reason = _text(reason)
        if not reason:
            raise LogError("reason", "a trial is voided with a reason", trial=trial)
        with self.log.lock:
            row = self.log.row(trial)
            clean_reason = check_value("void_reason", reason)
            if row["status"] == VOID and row["void_reason"] == clean_reason:
                return
            changes_ = {"status": VOID, "void_reason": clean_reason}
            if not row["time_end"]:
                changes_["time_end"] = now(when)
            if row["status"] == OPEN and not row["void_reason"]:
                self.log.record(trial, {k: v for k, v in changes_.items() if k != "status"}, status=VOID)
            else:
                self.log.edit_many(trial, changes_, reason, when)

    def check(self, deep: bool = False) -> dict:
        """Completeness of the phase: problems per logged trial, pending slots, trials outside the
        matrix, trial folders without a log row, trial ids logged twice, trials open at once."""
        rows = self.log.rows()
        filled, extra = match(self.sequence, rows)
        logged = {r["trial"] for r in rows}
        orphans = sorted(d.name for d in self.folder.iterdir() if d.is_dir() and d.name not in logged) \
            if self.folder.exists() else []
        ids = Counter(r["trial"] for r in rows)
        return {"trials": [{"trial": r["trial"], "status": r["status"], "problems": self.missing(r["trial"], deep)}
                           for r in rows],
                "pending": [s for s in self._sequence if int(float(s["slot"])) not in filled],
                "extra": extra, "orphans": orphans,
                "duplicates": sorted(t for t, n in ids.items() if n > 1),
                "open": [r["trial"] for r in rows if r["status"] == OPEN]}


# Design check


def design_check(n: int = 2000, seed: int = 1, runs: Sequence[int] = (1, 3, 9), drifts: Sequence[float] = (0.0, 1.0),
                 setup_sd: float = 0.5, change_s: float = 30) -> list[dict]:
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
                df_dir = X1.shape[1] - X0.shape[1]     # the direction levels but one
                f = (rss[0] - rss[1]) / df_dir / (rss[1] / df)
                hits += stats.f.sf(f, df_dir, df) < 0.05
                n_changes += new.sum() - 1
            rates.append(hits / n)
        mean_changes = n_changes / (n * len(drifts))
        out.append({"run": run, "changes": mean_changes, "minutes": mean_changes * change_s / 60,
                    **{f"false_pos_drift_{dr:g}": r for dr, r in zip(drifts, rates)}})
    return out


# Test


def test_folder(work: Path | str) -> Path:
    """The folder a test writes into, emptied: it must be missing, empty, or made by an earlier test (it holds
    TEST_MARKER). Raises ValueError for any other folder, which is never touched."""
    work = Path(work)
    if work.exists():
        if not work.is_dir():
            raise ValueError(f"{work} is not a folder")
        if any(work.iterdir()):
            if not (work / TEST_MARKER).is_file():
                raise ValueError(f"{work} is not empty and was not made by a test: give an empty or new folder")
            shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / TEST_MARKER).write_text("made by a self-test; a later test may empty this folder\n", encoding="utf-8")
    return work


def _raises(fn: Callable, kind: type, code: str | None = None) -> bool:
    try:
        fn()
    except kind as e:
        return code is None or getattr(e, "code", None) == code
    return False


def run_test(work: Path | str = WORK) -> int:
    """Sequence properties, then a simulated campaign with fake files: references, a void trial,
    a wrong setting corrected by an edit, an app closed mid-trial, a missing video, a trial voided
    after closing, a configuration change, then completeness checks and the integrity of the log: a close
    that fails half way, steps confirmed twice, a log saved by a spreadsheet, a bad cell, copies cut short,
    a deleted sequence, a second program writing."""
    try:
        work = test_folder(work)
    except ValueError as e:
        print(f"FAIL  {e}")
        return 1
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            fails.append(what)

    stray = Path(tempfile.mkdtemp(prefix="trials_guard_"))
    try:
        (stray / "data.csv").write_text("x\n", encoding="utf-8")
        refused = _raises(lambda: test_folder(stray), ValueError)
        expect(refused and (stray / "data.csv").is_file(),
               "a test given a folder with files it did not make refuses it and deletes nothing")
    finally:
        shutil.rmtree(stray, ignore_errors=True)

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
    rng0 = np.random.default_rng(0)
    expect(len(fill_levels((10, 20), [10, 10, 99, 99, 99], 3, rng0)) == 3,
           "fill_levels: values used that are no longer levels do not stop it")

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

    def fake(name):
        return fakes[Path(name).suffix]

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
            odd = [_raises(lambda h=h: run.configure(replace(run.config, holders=h)), LogError, "bad_field")
                   for h in (("A.2",), ("B 2",), ())]
            expect(all(odd) and run.sequence == seq_before,
                   "holder names a trial could not be closed with (A.2, B 2, none): refused, nothing written")
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
    expect(ok_ref, "reference scans exactly when due (first, holder, no_scan, reposition, session)")
    expect(ok_pre, "pre_scan: own reference or previous trial")
    expect(first_b is not None and run.steps(first_b)[0].key == "holder", f"holder change at {first_b}")
    rows = run.log.rows()
    report = run.check()
    expect(not report["pending"] and not report["extra"] and not report["orphans"] and not report["duplicates"]
           and not report["open"], "sequence complete")
    done = Counter(cell(r) for r in rows if r["status"] == OK)
    expect(all(done[(h, a, d, 0)] == 2 for h in cfg.holders for a in cfg.angles_deg for d in cfg.directions_deg)
           and all(sum(n for c, n in done.items() if c[0] == h and c[3]) == 12 for h in cfg.holders),
           f"valid trials: every cell twice and 12 drape trials per holder; {len(rows)} trials in all")
    expect(all(not t["problems"] for t in report["trials"]), "no problems in any trial")
    dr = next(r["trial"] for r in rows if r["status"] == OK and r["drape"] == "1")
    keys = [st.key for st in run.steps(dr)]
    names = sorted(p.name for p in (raw / "campaign" / dr).iterdir())
    expect(keys.index("drape_video") < keys.index("drape_off") < keys.index("photo_az40") < keys.index("video")
           < keys.index(f"scan_az{cfg.scan_az[0]}") < keys.index("spheres")
           and names == sorted(n for st in run.steps(dr) for n in st.files)
           and len([n for n in names if "_ref_" not in n]) == 12
           and file_name(dr, "photo", 40, drape=True) in names and file_name(dr, "video", drape=True) in names,
           f"drape trial {dr}: photos and video with the drape, drape off, photos and video, then scans "
           f"({len(names)} files)")
    plain = next(r["trial"] for r in rows if r["status"] == OK and r["drape"] == "0")
    keys = [st.key for st in run.steps(plain)]
    expect(keys.index("spheres") < keys.index("photo_az40") and not any(k.startswith("drape") for k in keys),
           f"trial without drape {plain}: scans, then photos and video")
    edits = run.log.edits()
    expect([(e["trial"], e["field"]) for e in edits] == [(edited, "angle_deg"), ("t040", "status"),
                                                         ("t040", "void_reason")],
           f"audit trail: the angle of {edited}, the status and the reason of t040 voided after closing")
    expect(_raises(lambda: run.log.fill(edited, angle_deg=50), ValueError, "bad_field"),
           "logged values change only through edit()")
    expect(_raises(lambda: run.store("t003", file_name("t003", "photo", 40), fake("t003_photo_az40.jpg")),
                   FileExistsError, "exists"), "store() never overwrites a raw file")
    t3 = raw / "campaign" / "t003"
    expect(_raises(lambda: run.store("t003", "T003_PHOTO_AZ40.jpg", fake("t003_photo_az40.jpg")), FileExistsError),
           "store() never overwrites a raw file named in another case")
    manifest = {(m["trial"], m["file"]): m for m in run.log.manifest()}
    m3 = manifest.get(("t003", "t003_photo_az40.jpg"), {})
    expect(len(manifest) == sum(len(list((raw / "campaign" / r["trial"]).iterdir())) for r in rows)
           and m3.get("bytes") == "4" and m3.get("sha256") == hashlib.sha256(b"fake").hexdigest()
           and not list((raw / "campaign").rglob("*" + PART_SUFFIX)),
           f"every raw file in the manifest with its size and SHA-256 ({len(manifest)} files), no copy left half way")

    run.find("t003", file_name("t003", "photo", 40))[0].unlink()
    (t3 / "IMG_20261020_101112.jpg").write_bytes(b"x")
    run.find("t003", file_name("t003", "scan", cfg.scan_az[1]))[0].write_bytes(b"")
    (t3 / "t003_video.mov").write_bytes(b"x")
    got = set(run.missing("t003"))
    want = {("missing", "t003_photo_az40.jpg"), ("unexpected", "IMG_20261020_101112.jpg"),
            ("empty", file_name("t003", "scan", cfg.scan_az[1])), ("unexpected", "t003_video.mov")}
    expect(got == want, "completeness: missing photo, stray file, empty scan, wrong video suffix")
    (t3 / file_name("t003", "scan", cfg.scan_az[0])).write_bytes(b"fake but longer")
    expect(("changed", file_name("t003", "scan", cfg.scan_az[0])) in run.missing("t003", deep=True),
           "deep completeness: a raw file whose size is not the one filed is said")

    # each trial follows the configuration it started under (config_version), whatever today's says
    rows = run.log.rows()
    closed = next(r for r in rows if r["status"] == OK and r["trial"] != "t003")
    early, late = rows[0], rows[-1]
    before, v0 = run.missing(closed["trial"]), run.config_version
    run.configure(replace(run.config, scan_az=(55, 85)))
    doc = json.loads(run.config_path.read_text(encoding="utf-8"))
    expect(closed["config_version"].isdigit() and run.config_version == v0 + 1 == doc["version"]
           and [h.get("version") for h in doc["history"]] == list(range(1, v0 + 1))
           and run.missing(closed["trial"]) == before and run.config_for(closed).scan_az == cfg.scan_az
           and run.config_for(early).drape_trials == 20 and run.config_for(late).drape_trials == 24
           and run.config_for({"config_version": ""}) == run.config and run.has_scans(closed["trial"]),
           f"{closed['trial']} follows its own configuration (version {closed['config_version']}): the arm "
           f"positions changed today ({run.config_version}), nothing of it missing; an older trial its own drape "
           f"count; a trial logged without a version today's")
    expect(_raises(lambda: run.log.edit(closed["trial"], "config_version", "1", "test"), LogError),
           "the configuration version of a trial is never edited")
    run.configure(replace(run.config, scan_az=cfg.scan_az))

    import cv2
    cases = {"good.jpg": (cfg.photo_px[1], cfg.photo_px[0]), "small.jpg": (480, 640)}
    for name, shape in cases.items():
        cv2.imwrite(str(src / f"t900_photo_{name}"), np.zeros(shape + (3,), np.uint8))
    for name, seconds in (("long", 8), ("short", 3)):
        w = cv2.VideoWriter(str(src / f"t900_video_{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        for _ in range(30 * seconds):
            w.write(np.zeros((48, 64, 3), np.uint8))
        w.release()
    deep = {p.name: deep_check(cfg, p) for p in src.glob("t900_*")}
    codes = {p.name: deep_check_codes(cfg, p) for p in src.glob("t900_*")}
    expect(not deep["t900_photo_good.jpg"] and deep["t900_photo_small.jpg"][0][0] == "content"
           and not deep["t900_video_long.mp4"] and deep["t900_video_short.mp4"][0][0] == "content",
           "deep check: photo size and video duration")
    expect(codes["t900_photo_small.jpg"][0]["code"] == "photo_size"
           and codes["t900_photo_small.jpg"][0]["got"] == [640, 480]
           and codes["t900_video_short.mp4"][0]["code"] == "video_short" and not codes["t900_photo_good.jpg"],
           "deep check as codes: photo_size with the size found and wanted, video_short")
    # against the calibrations kept: a video or a photo of another size than its camera mode's calibration
    calib = work / "kept_calibration"       # not the default beside the protocol, which has none
    calib.mkdir()
    for mode, size in (("photo", [640, 480]), ("video", [1920, 1080])):
        (calib / f"intrinsics_{mode}.json").write_text(json.dumps({"image_size": size}), encoding="utf-8")
    sized = Run("campaign", raw, protocol, calibration=calib)
    sizes = sized.camera_sizes()
    long_ = deep_check_codes(cfg, src / "t900_video_long.mp4", sizes)
    expect(sizes == {"photo": [640, 480], "video": [1920, 1080]}
           and long_ == [{"code": "video_size", "file": "t900_video_long.mp4", "got": [64, 48], "want": [1920, 1080]}]
           and not deep_check_codes(cfg, src / "t900_photo_small.jpg", sizes)
           and deep_check_codes(cfg, src / "t900_photo_good.jpg", sizes)[0]["code"] == "photo_size"
           and Run("campaign", raw, protocol).camera_sizes() == {"photo": None, "video": None},
           "against the calibrations kept: a trial's video of another size than intrinsics_video.json, a photo held "
           "against intrinsics_photo.json (else the configuration)")
    # the photo's exposure, from its EXIF: not set by hand (the camera app out of its Pro mode) is said
    from PIL import Image
    for name, program in (("pro", MANUAL_PROGRAM), ("auto", 2)):
        ex = Image.Exif()
        ex.get_ifd(EXIF_IFD)[EXPOSURE_PROGRAM] = program
        Image.new("RGB", (cfg.photo_px[0], cfg.photo_px[1])).save(src / f"t900_photo_{name}.jpg", exif=ex)
    auto = deep_check_codes(cfg, src / "t900_photo_auto.jpg")
    expect(not deep_check_codes(cfg, src / "t900_photo_pro.jpg") and [p["code"] for p in auto] == ["photo_auto"]
           and deep_check(cfg, src / "t900_photo_auto.jpg")[0][0] == "content",
           "a trial photo whose exposure was not set by hand (out of the Pro mode): said; one in Pro: nothing")

    test_integrity(work, expect)
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed")
    return 1 if fails else 0


def _first_trial(run: Run, clock: datetime) -> tuple[str, list[Step]]:
    plan = run.next(today=clock.date())
    t = run.start(plan, when=clock, swing_mass_g=150)["trial"]
    return t, run.steps(t)


def _do_steps(run: Run, t: str, steps: Sequence[Step], until: str, src: Path, clock: datetime,
              spheres: dict | None = None) -> None:
    """The steps of t up to (not including) `until` (every step for ''), with stand-in files."""
    for st in steps:
        if st.key == until:
            return
        for name in st.files:
            p = src / ("fake" + Path(name).suffix)
            if not p.exists():
                p.write_bytes(b"fake")
            run.store(t, name, p)
        values = (spheres or {"cap_shift_mm": "0.40", "spheres_ok": 1}) if st.key == "spheres" else {}
        run.confirm(t, st.key, when=clock, **values)


def test_integrity(work: Path, expect: Callable) -> None:
    """The trial log under failures: one transaction per close, steps confirmed twice, the audit trail first,
    validated values, a log saved by a spreadsheet, a bad cell, copies cut short, a deleted sequence, two
    programs writing, times with their offset."""
    global WRITE_RETRIES, LOCK_TIMEOUT_S  # noqa: PLW0603  the test shortens the waits of the module it tests
    raw, protocol, src = work / "raw2", work / "protocol2", work / "src2"
    src.mkdir(parents=True)
    run = Run("campaign", raw, protocol)
    clock = datetime(2026, 10, 25, 2, 30)              # the hour that comes twice when summer time ends
    t, steps = _first_trial(run, clock)
    expect(re.fullmatch(r"2026-10-25T02:30:00[+-]\d\d:\d\d", run.log.row(t)["time_start"]) is not None,
           f"times carry their offset from UTC ({run.log.row(t)['time_start']})")
    _do_steps(run, t, steps, "pendulum", src, clock)
    run.confirm(t, "pendulum")
    run.confirm(t, "release", when=clock)
    first = run.log.row(t)["time_impact"]
    run.confirm(t, "release", when=clock + timedelta(minutes=3))
    expect(run.log.row(t)["time_impact"] == first, "the release confirmed twice: no error, its time kept")
    _do_steps(run, t, steps[[s.key for s in steps].index("release") + 1:], "close", src, clock)
    run.confirm(t, "spheres", cap_shift_mm="0.40", spheres_ok=1)
    n_edits = len(run.log.edits())
    run.confirm(t, "spheres", cap_shift_mm="0.55", spheres_ok=1)
    e = run.log.edits()[-1] if run.log.edits() else {}
    expect(run.log.row(t)["cap_shift_mm"] == "0.55" and len(run.log.edits()) == n_edits + 1
           and e.get("field") == "cap_shift_mm" and e.get("old") == "0.40" and "spheres" in e.get("reason", ""),
           "a step confirmed again with a new value: changed, the old value in the audit trail")

    # a close that fails half way: nothing is written, the next close works
    real_replace = os.replace
    old_retries = WRITE_RETRIES
    WRITE_RETRIES = 2

    def held(a, b):
        if Path(b).name == LOG_NAME:
            raise PermissionError("held by a spreadsheet")
        return real_replace(a, b)
    os.replace = held
    try:
        locked = _raises(lambda: run.confirm(t, "close", when=clock, notes="first try"), PermissionError,
                         "log_locked")
    finally:
        os.replace = real_replace
        WRITE_RETRIES = old_retries
    row = run.log.row(t)
    leftovers = [p.name for p in (raw / "campaign").iterdir() if p.name.endswith(".tmp")]
    expect(locked and row["status"] == OPEN and not row["time_end"] and not row["notes"] and not leftovers,
           "the log held by another program during the close: log_locked, nothing half written, no temporary file")
    run.log.fill(t, time_end=now(clock))               # what an older version left after a crash between writes
    run.confirm(t, "close", when=clock + timedelta(minutes=9), notes="cap checked")
    row = run.log.row(t)
    expect(row["status"] == OK and row["time_end"] == now(clock) and row["notes"] == "cap checked",
           "a trial left with its end time by a crash can still be closed, the time kept")
    run.confirm(t, "close", when=clock, notes="cap checked")
    run.confirm(t, "close", when=clock, notes="cap replaced after")
    expect(run.log.row(t)["notes"] == "cap checked; cap replaced after" and run.log.row(t)["status"] == OK,
           "closed twice: no error; notes at close are added to those logged, once")
    expect(run.log.edits()[-1]["field"] == "notes" and run.log.edits()[-1]["new"] == "cap checked; cap replaced after",
           "notes added after the close are in the audit trail")
    expect(_raises(lambda: run.confirm(t, "close", second_hit=1), ValueError, "bad_field"),
           "a closed trial's value changes only through edit()")
    long_note = "x" * (MAX_TEXT - 10)
    expect(_raises(lambda: run.confirm(t, "close", when=clock, notes=long_note), ValueError, "bad_field")
           and run.log.row(t)["notes"] == "cap checked; cap replaced after",
           f"notes added at the close that would pass {MAX_TEXT} characters in all: refused, the notes as they were")

    # the audit trail first: when the log cannot be written, the edit is in the trail and the log as it was
    def log_down(a, b):
        if Path(b).name == LOG_NAME:
            raise PermissionError("held")
        return real_replace(a, b)
    os.replace, WRITE_RETRIES = log_down, 2
    try:
        _raises(lambda: run.log.edit(t, "swing_mass_g", "151", "weighed again"), PermissionError)
        n_trail = len(run.log.edits())
        _raises(lambda: run.log.edit(t, "swing_mass_g", "151", "weighed again"), PermissionError)
        again = len(run.log.edits()) == n_trail
    finally:
        os.replace, WRITE_RETRIES = real_replace, old_retries
    e = run.log.edits()[-1]
    expect(e["field"] == "swing_mass_g" and e["new"] == "151" and run.log.row(t)["swing_mass_g"] == "150",
           "an edit writes the audit trail before the log")
    run.log.edit(t, "swing_mass_g", "151", "weighed again")
    rows151 = [x for x in run.log.edits() if x["trial"] == t and x["field"] == "swing_mass_g" and x["new"] == "151"]
    expect(again and len(rows151) == 1 and run.log.row(t)["swing_mass_g"] == "151",
           "the same edit tried again after a failed write: one row in the trail, then the log changed")
    run.log.edit(t, "swing_mass_g", "150", "back to the first weighing")

    # values are checked before anything is written
    bad = [_raises(lambda: run.log.edit(t, "angle_deg", "abc", "typo"), ValueError, "bad_field"),
           _raises(lambda: run.log.edit(t, "holder", "Z", "typo"), ValueError, "bad_field"),
           _raises(lambda: run.log.edit(t, "drape", "2", "typo"), ValueError, "bad_field"),
           _raises(lambda: run.log.edit(t, "status", "done", "typo"), ValueError, "bad_field"),
           _raises(lambda: run.log.edit(t, "time_end", "yesterday", "typo"), ValueError, "bad_field"),
           _raises(lambda: run.log.edit(t, "angle_deg", "40", " "), ValueError, "reason"),
           _raises(lambda: run.void(t, ""), ValueError, "reason")]
    expect(all(bad), f"values that are not valid are refused before any write: {bad}")
    run.log.edit(t, "notes", "=HYPERLINK(\"x\")", "typed by mistake")
    expect(run.log.row(t)["notes"].startswith("'="), "a note that would start a spreadsheet formula is kept as text")
    expect(_raises(lambda: run.steps("t999"), KeyError, "unknown_trial")
           and _raises(lambda: run.confirm(t, "nothing"), KeyError, "unknown_step"),
           "an unknown trial or step: a coded error, not StopIteration")
    t2, steps2 = _first_trial(run, clock + timedelta(hours=1))
    expect(_raises(lambda: run.log.edit(t, "status", OPEN, "reopened"), ValueError, "trial_open")
           and _raises(lambda: run.start(run.next(today=clock.date()) or {}, swing_mass_g=150), ValueError,
                       "trial_open"),
           "a second trial cannot be opened, by an edit or a start")

    # a raw file copied half way is never filed; a move deletes its source last
    src_file = src / "photo_source.jpg"
    src_file.write_bytes(os.urandom(300_000))
    real_fsync = os.fsync
    name = file_name(t2, "photo", 40)

    def torn(fd):
        os.ftruncate(fd, 1000)
        real_fsync(fd)
    os.fsync = torn
    try:
        failed = _raises(lambda: run.store(t2, name, src_file, move=True), OSError, "copy_failed")
    finally:
        os.fsync = real_fsync
    expect(failed and not run.find(t2, name) and src_file.is_file()
           and not list((raw / "campaign" / t2).glob("*" + PART_SUFFIX)),
           "a copy cut short: copy_failed, nothing filed, no part left, the source kept")
    dest = run.store(t2, name, src_file, move=True)
    expect(dest.is_file() and not src_file.exists() and sha256_file(dest) == run.log.manifest()[-1]["sha256"],
           "a verified move: filed with its hash, the source deleted last")
    other = src / "video_source.mp4"
    other.write_bytes(os.urandom(50_000))
    real_manifest = run.log.add_manifest

    def no_manifest(*a, **k):
        raise LogLocked("log_locked", "the manifest is held", file=MANIFEST_NAME)
    run.log.add_manifest = no_manifest
    try:
        vdest = run.store(t2, file_name(t2, "video"), other, move=True)
        stored = vdest.is_file() and not other.exists()
    except (OSError, TrialsError):
        stored = False
    finally:
        run.log.add_manifest = real_manifest
    expect(stored, "a manifest row that cannot be written: the file still filed and its source moved, never a "
                   "failure that has the step done again over a file already there")
    run.void(t2, "test trial")
    run.void(t2, "test trial")
    expect(run.log.row(t2)["status"] == VOID and run.log.row(t2)["time_end"], "voided twice: no error")
    # a trailing comma on a row (an editor's habit) is no harm; text beyond the columns is
    comma = src / "comma.csv"
    comma.write_text("trial,angle_deg\nt001,10,\nt002,20, \n", encoding="utf-8")
    beyond = src / "beyond.csv"
    beyond.write_text("trial,angle_deg\nt001,10,oops\n", encoding="utf-8")
    expect([r["angle_deg"] for r in read_csv(comma, ("trial",))] == ["10", "20"]
           and _raises(lambda: read_csv(beyond, ("trial",)), LogError, "log_corrupt"),
           "a row with a blank field beyond the columns read; one with text there refused (log_corrupt)")
    # a copy whose source is missing never deletes a file already at its destination
    there = src / "already_there.bin"
    there.write_bytes(b"keep me")
    expect(_raises(lambda: copy_verified(src / "no_such_file.bin", there), OSError)
           and there.read_bytes() == b"keep me", "a failed copy leaves a file it did not make where it was")

    # a log saved by a spreadsheet: byte order mark, ';', decimal commas, days as dd/mm/yyyy, a column missing
    rows = run.log.rows()
    cols = [c for c in LOG_FIELDS if c != "notes"]
    buf = io.StringIO()
    sheet = csv.writer(buf, delimiter=";", lineterminator="\r\n")
    sheet.writerow(cols)
    for r in rows:
        sheet.writerow([r[c].replace(".", ",") if c in NUMERIC_FIELDS else
                        "/".join(reversed(r[c].split("-"))) if c == "session" else r[c] for c in cols])
    excel = work / "excel" / "campaign"
    excel.mkdir(parents=True)
    (excel / LOG_NAME).write_bytes(b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"))
    back = Log(excel).rows()
    same = [{k: r[k] for k in cols} for r in back] == [{k: r[k] for k in cols} for r in rows]
    expect(same and all(r["notes"] == "" for r in back),
           "a log saved by a spreadsheet (byte order mark, ';', decimal commas, dd/mm/yyyy, a column missing) "
           "reads as written")
    (excel / LOG_NAME).write_bytes(("trial,status,notes\nt001,ok,caf\xe9\n").encode("cp1252"))
    expect(Log(excel).rows()[0]["notes"] == "caf\u00e9", "a log saved in the Windows code page reads")
    (excel / LOG_NAME).write_text("id,state\nt001,ok\n", encoding="utf-8")
    expect(_raises(lambda: Log(excel).rows(), ValueError, "log_corrupt"),
           "a log without its trial column: log_corrupt, not a KeyError")
    (excel / LOG_NAME).write_text("trial,status\nt001,ok,extra,more\n", encoding="utf-8")
    expect(_raises(lambda: Log(excel).rows(), ValueError, "log_corrupt"), "a row longer than its header: log_corrupt")

    # one bad cell typed by hand: the module goes on and says where
    text = (raw / "campaign" / LOG_NAME).read_text(encoding="utf-8")
    first_row = text.splitlines()[1]
    parts = next(csv.reader([first_row]))
    parts[LOG_FIELDS.index("angle_deg")] = "abc"
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(parts)
    (raw / "campaign" / LOG_NAME).write_text(text.replace(first_row + "\n", buf.getvalue(), 1), encoding="utf-8")
    try:
        rep = run.check()
        plan = run.next(today=clock.date())
        fine = plan is not None and ("bad_field", "angle_deg") in rep["trials"][0]["problems"]
    except Exception as e:  # noqa: BLE001  the check is that nothing raises
        fine = False
        print(f"      {type(e).__name__}: {e}")
    expect(fine, "a cell made unreadable by hand: next() and check() go on, check() says which field")
    run.log.edit(t, "angle_deg", "40", "typo fixed")

    # a deleted sequence file: the configuration stays, the sequence is made again
    run.configure(replace(run.config, drape_trials=22))
    run.sequence_path.unlink()
    again = Run("campaign", raw, protocol)
    expect(again.config.drape_trials == 22 and len(again.sequence) == 130 and run.config.drape_trials == 22,
           "a deleted sequence: the configuration kept (22 drape trials), the sequence made again")
    other = Run("campaign", raw, protocol)
    other.configure(replace(other.config, drape_trials=24))
    expect(len(run.sequence) == 132 and run.config.drape_trials == 24,
           "another program changes the configuration: this run reads it again")
    cfg_text = run.config_path.read_text(encoding="utf-8")
    run.config_path.write_text("{ cut short", encoding="utf-8")
    expect(_raises(lambda: Run("campaign", raw, protocol), ValueError, "config_corrupt")
           and run.config_path.read_text(encoding="utf-8") == "{ cut short",
           "a configuration that cannot be read: config_corrupt, never written over")
    run.config_path.write_text(cfg_text, encoding="utf-8")

    # two programs: a lock held by another waits, then log_locked; one left by a program that stopped is removed
    lock = raw / "campaign" / LOCK_NAME
    lock.write_text(f"{os.getppid()} 2026-10-25T03:00:00+01:00\n", encoding="utf-8")    # a program that runs
    old_timeout, LOCK_TIMEOUT_S = LOCK_TIMEOUT_S, 0.3
    try:
        held_lock = _raises(lambda: run.log.edit(t, "angle_deg", "45", "test"), PermissionError, "log_locked")
    finally:
        LOCK_TIMEOUT_S = old_timeout
    os.utime(lock, (time.time() - 2 * LOCK_STALE_S,) * 2)
    run.log.edit(t, "angle_deg", "45", "test")
    expect(held_lock and run.log.row(t)["angle_deg"] == "45" and not lock.exists(),
           "another program writing: waited for, then log_locked; a lock left by a stopped program is removed")
    # a lock left by this process (its removal failed) or by a program that has ended: taken at once, not after
    # LOCK_STALE_S; a failed removal never fails the write it guarded
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    quick = []
    for pid in (os.getpid(), child.pid):
        lock.write_text(f"{pid} 2026-10-25T03:00:00+01:00\n", encoding="utf-8")
        t0 = time.monotonic()
        run.log.edit(t, "angle_deg", "40" if pid == os.getpid() else "45", "test")
        quick.append(time.monotonic() - t0 < LOCK_TIMEOUT_S and not lock.exists())
    real_unlink = Path.unlink

    def stuck(self, *a, **k):
        if self.name == LOCK_NAME:
            raise PermissionError("held by a scanner")
        return real_unlink(self, *a, **k)
    Path.unlink = stuck
    try:
        run.log.edit(t, "angle_deg", "40", "test")
        survived = lock.exists()
    except (OSError, TrialsError):
        survived = False
    finally:
        Path.unlink = real_unlink
    run.log.edit(t, "angle_deg", "45", "test")
    expect(all(quick) and survived and not lock.exists() and run.log.row(t)["angle_deg"] == "45",
           "a lock of this process or of an ended program taken at once; a lock that cannot be removed does not "
           "fail the write, and is taken back by the next one")
    expect(check_value("time_start", "2026-10-20T09:00:00") == "2026-10-20T09:00:00"
           and check_value("swing_mass_g", "179,5") == "179.5" and check_value("drape", True) == "1"
           and _text(None) == "", "old times without offset still valid; decimal comma, truth values, None as ''")
    expect(deep_check(PILOT, _fake_video(src / "p001_highspeed.mp4", 30))[0][1].endswith(f"{HIGHSPEED_FPS}"),
           "the high speed message gives the threshold it checks")
    t4, steps4 = _first_trial(run, clock + timedelta(hours=2))
    p3 = run.next(today=clock.date())
    _do_steps(run, t4, steps4, "", src, clock, spheres={"spheres_ok": 1})
    p4 = run.next(today=clock.date())
    expect(p3["resume"] and p3["warning_codes"] == [] and run.log.row(t4)["status"] == OK
           and p4["warning_codes"] == [{"code": "cap_unknown", "trial": t4}] and "unknown" in p4["warnings"][0],
           "warnings as codes beside the sentences: a trial closed without its cap shift (cap_unknown)")


def _fake_video(path: Path, fps: int) -> Path:
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    for _ in range(fps):
        w.write(np.zeros((24, 32, 3), np.uint8))
    w.release()
    return path


def print_check(report: dict) -> int:
    bad = [t for t in report["trials"] if t["problems"]]
    for t in bad:
        print(f"{t['trial']} ({t['status']}): " + "; ".join(f"{k} {v}" for k, v in t["problems"]))
    n = Counter(t["status"] for t in report["trials"])
    print(f"{len(report['trials'])} trials logged ({', '.join(f'{v} {k}' for k, v in n.items()) or 'none'}), "
          f"{len(bad)} with problems, {len(report['pending'])} slots pending, "
          f"outside the matrix: {', '.join(report['extra']) or 'none'}, "
          f"folders without a log row: {', '.join(report['orphans']) or 'none'}")
    if report.get("duplicates"):
        print(f"trial ids logged twice: {', '.join(report['duplicates'])}")
    if len(report.get("open", [])) > 1:
        print(f"trials open at once: {', '.join(report['open'])}")
    return 1 if bad or report["orphans"] or report.get("duplicates") or len(report.get("open", [])) > 1 else 0


def main() -> None:
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

    try:
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
    except TrialsError as e:
        sys.exit(f"trials: {e.code}: {e}")


if __name__ == "__main__":
    main()
