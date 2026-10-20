"""ImpactGuide: the desktop app of the marker impact bench.

A frameless SlantUI window hosting a page (ui.html and ui/) that talks to this
backend over QWebChannel. As UniGuide, it opens at home, four large choices, and
each choice is a module with its own steps on the rail: Calibrate the phone
(board, photos, calibration), Live AR (board, view), Trials (next trial, log,
sequence, setup) and Scans and results (the next pages). The app guides the
protocol step by step and keeps what the bench produces.

The calibration board is drawn by the page itself, in the work area of the step
that shows it, at its true size: laid out for the pixels of that area at the
pitch of the monitor the window is on (screen_board.py), never in a window of
its own; or it is physical: the 3D-printed plate (charuco_plate.py) or a printed
A4 sheet, measured with a caliper. The photos of a calibration follow a plan of
poses (calib_guide.py), each one taken at once (Take, the space bar, or by
itself when the phone holds the pose, auto_shot.py), copied and checked in the
background while the next one is taken.

This file is the host: the window, the bridge, the background jobs. The logic
is in 04_code/src (trials.py, acquisition.py, calibrate_charuco.py, live_ar.py,
phone_link.py, phone_view.py, screen_board.py), which the host only calls; everything there also
runs without the app. The live frames reach the page as MJPEG from 127.0.0.1
(live_ar.MjpegServer), and the phone's own screen from a port of this PC
(phone_view.ViewServer), made when the operator opens that view and bound to
127.0.0.1 (the phone reaches it through adb) unless he asks to show the view on
another screen of the same network. The twin of the bench (twin.py) has its 3D model made again from the bench
scene by a Blender in the background once the scene has rested for two minutes; the page shows it from the trial's
page and as the Scans step. The link to the phone is found and watched by itself
(phone_link, "auto": the cable when the phone is on it, else the Wi-Fi it was
paired on); every part asks phone_link for the serial, and the Phone panel only
shows the link and pairs Wi-Fi when that is wanted.

    python impactguide_app.py [--simulate DIR] [--replay PATH] [--lang en|it|fr|es|de|zh] [--page MODULE]

--simulate keeps every file under DIR and stands in for the phone and the
scanner (a simulated phone on the cable that can also be paired on Wi-Fi with
the code 123456 or the QR code), for trying the app without the bench and for
test_app.py. --replay
gives the live page recorded frames (a video of the phone, a folder of images,
the frames of the digital twin) in place of the phone's stream.

Errors: the page only ever gets a code (and its values) to say in the
operator's language; the whole error, with its trace, goes to errors.log
(%LOCALAPPDATA%\\ImpactGuide, or DIR/logs under --simulate). An exception that
escapes a Qt callback is logged and the app goes on.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
SRC = APP.parent / "src"
LOCAL = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ImpactGuide"
# The project may sit on a synced drive: bytecode goes to a local cache, never next to the sources.
if not sys.pycache_prefix:
    sys.pycache_prefix = str(LOCAL / "pycache")
sys.path.insert(0, str(SRC))

import argparse  # noqa: E402
import base64  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import logging.handlers  # noqa: E402
import math  # noqa: E402
import queue  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import date, datetime  # noqa: E402
from typing import Any, ClassVar  # noqa: E402

import slantui.css as slantui_css  # noqa: E402
import slantui.js as slantui_js  # noqa: E402
from slantui.shell import Application, Bridge, Splash, Window, pyqtSignal, pyqtSlot  # noqa: E402
from slantui.shell.qt import (QApplication, QColor, QDesktopServices, QFileDialog, QTimer, QUrl,  # noqa: E402
                              USE_QT6)
from slantui.tokens import PALETTES  # noqa: E402

import acquisition  # noqa: E402
import auto_shot  # noqa: E402
import calib_bench  # noqa: E402
import calib_guide  # noqa: E402
import calib_video  # noqa: E402
import calibrate_charuco  # noqa: E402
import charuco_plate  # noqa: E402
import live_ar  # noqa: E402
import live_check  # noqa: E402
import phone_link  # noqa: E402
import phone_view  # noqa: E402
import screen_board  # noqa: E402
import trials  # noqa: E402
import twin  # noqa: E402
import scans_page  # noqa: E402  (the Scans page's side of this backend, beside this file)
import dataset_page  # noqa: E402  (the Dataset page's side of this backend, beside this file)

APP_NAME = "ImpactGuide"
APP_VERSION = "0.1.0"
APP_ID = "ImpactGuide.Bench"
PALETTE = "gold-dark"
# The languages of the interface, in the order the picker lists them (ui/90-boot.js): the code (the
# file in ui/strings/), the name in the language itself, the flag (a file in ui/flags/, flag-icons:
# English has the flag gb and Chinese, Simplified, the flag cn).
LANGUAGES = (("en", "English", "gb"), ("it", "Italiano", "it"), ("fr", "Français", "fr"),
             ("es", "Español", "es"), ("de", "Deutsch", "de"), ("zh", "中文", "cn"))
LANGS = tuple(code for code, _, _ in LANGUAGES)
PAGES = ("calibration", "live", "trials", "results")     # the modules of home, each with its steps (ui/01-shell.js)
ICON = APP / "ui" / "impactguide-icon.svg"  # the round app icon: desktop, taskbar, window
MARK = APP / "ui" / "impactguide-mark.svg"  # the same drawing without its disc, cut to fill the band's square
# the camera modes, each with its own calibration (intrinsics_<mode>.json), never used for another: the phone's photos
# (camera app), the films of its camera app (the trial videos; calibrated on frames chosen from a film), the frames of
# the live stream
CALIB_MODES = ("photo", "video", "stream")
FILM_MAX_S = 120            # a film for a calibration stops by itself after this long
FILM_SIM_SIZE = (1280, 720)  # the simulated phone's film of the board
NARROWER_MAX = 0.03         # a film (or the stream) whose field is this much narrower than the photos' was cropped
IMAGE_SUFFIXES = tuple(sorted(calibrate_charuco.IMAGE_SUFFIXES))
CHECK_FIELDS = ("time", "mode", "folder", "ok", "n_views", "rms_stored_px", "fx_change", "fy_change",
                "cx_change", "cy_change")
BOARD_KINDS = ("screen", "plate", "sheet")  # on this monitor, the 3D-printed plate, a printed A4 sheet
MIN_WINDOW = (720, 560)     # the app's least size
PHOTO_CHECKS = "photo_checks.json"    # the check of each photo of a series, beside the photos
REJECTED = "rejected"                 # the photos of a series put aside by their check (or by hand)
THUMB_W = 160               # the small picture of each photo of a series (the Photos step), px wide
STRIP_N = 12                # the newest photos of a series the Photos step shows, each with how its check went
PREVIEW_RETRY_S = 5.0       # the phone's screen mirrored for the photo taken by itself: tried again this often
SHEET_SPREAD_MAX = 0.30     # mm between the readings of the A4 sheet's long side (a caliper, or a steel ruler)
SHEET_NOMINAL_TOL = 0.02    # a printer that scales the page by more than 2 %, or a misreading
PLATE_MODEL = APP / "ui" / "models" / "charuco_plate.glb"   # the plate in 3D for the page (tools/build_plate_model.py)
GUESS_PITCH = 25.4 / 96     # mm: a monitor that says nothing of its size is laid out as a 96 dpi one, to be measured
COMMA_LANGS = ("it", "fr", "es", "de")
WINDOW_SIZE = (1440, 900)   # the window at its first start
JOB_PUMP_MS = 150           # results of background jobs handed to the GUI thread this often
SCAN_POLL_MS = 1000         # the scanner's export folder listed this often, in a worker
TWIN_POLL_MS = 10000        # the bench scene looked at this often, for the model of the twin (twin.py)
FAKE_SCAN_MS = 1200         # in simulation the stand-in scanner exports this long after the watch starts
SPLASH_HIDE_MS = 200        # the loading screen goes this long after the page says it is up
SPLASH_MAX_MS = 25000       # and at the latest this long after the start
READING_MM = (20.0, 1000.0)  # a caliper reading of the plate or the sheet outside this range is a typing slip
QR_CACHE = 8                # QR codes kept as SVG (the Phone panel asks every second)
SERIES_KEEP_S = 6 * 3600    # a series goes on past midnight while its last photo is younger than this
F_STD_REL_MAX = 0.005       # a focal length less certain than this (std over value) is not a good calibration
# what spoils a calibration (calibrate_charuco.ISSUES), the most serious first: the first one is said in full
ISSUE_ORDER = ("board_not_flat", "phone_not_still", "camera_settings", "corners_missing")


def _not_good(cal: dict) -> tuple[str, ...]:
    """The issues that keep a calibration from being good (calibrate_charuco.ISSUES_NOT_GOOD): on the bench the phone
    sits on the arm, and a centre the poses pin less well is only advice."""
    bench = cal.get("plan") == "bench"
    return tuple(i for i in calibrate_charuco.ISSUES_NOT_GOOD if not (bench and i == "phone_not_still"))
EXTRA_MAX = 4               # photos past the plan for the corners of the photo the board has not reached yet
LOG_MAX_BYTES = 2_000_000   # errors.log is turned over at this size (one older file kept)
PREPARE_WAIT_S = 120.0      # a Keep waits this long at most for the copy ahead of its file (_prepare_staged)
PAGE_ERRORS_MAX = 200       # errors of the page's scripts written to the log in a run, at most...
PAGE_ERROR_CHARS = 500      # ... each cut to this length
WRITE_RETRIES = 20          # a file held for a moment (Drive, an antivirus) is replaced again this many times
WRITE_RETRY_S = 0.25        # this far apart
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000  # Windows: enough rights to ask whether a process still runs
STILL_ACTIVE = 259          # Windows: the exit code of a process that has not ended
EVENT_MODIFY_STATE = 0x0002  # Windows: the right to raise an event
LAUNCHER_EVENT = r"Local\ImpactGuideUp"     # the event the desktop launcher waits on (scripts/launcher.ps1)
# a folder on the phone, matched whole: no spaces, quotes, "..", "//" or line ends
CAMERA_DIR_RE = re.compile(r"(?!.*\.\.)(?:/[\w.\-]+)+")
# AcquisitionError causes whose detail is a file name the page may show (the others carry adb's own text)
NAMED_DETAIL = ("not_found", "copy_failed", "exists")
# the phone-screen server with no viewer (phone_view.ViewServer.stats, before the server is made)
VIEW_IDLE = {"viewers": 0, "fps": 0.0, "added_ms": None, "added_p95_ms": None, "page_ms": None, "kb": None,
             "mbit_s": 0.0, "frames": 0}

log = logging.getLogger("impactguide")


# ── errors, logs, files ──────────────────────────────────────────────────────
class UserError(Exception):
    """A refusal the operator can act on (no open trial, a folder that is not there): answered with its code,
    logged as one line, not as a fault.

    Args:
        code: the error code the page names (``err.<code>`` in ui/strings).
        detail: a short plain detail (a file or a trial name), or "".
        **values: the values of the sentence, sent as ``args``.
    """

    def __init__(self, code: str, detail: str = "", **values: Any) -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail, self.values = code, detail, values


class _LogState:
    """The file handler the app put on the root logger (one at a time)."""

    handler: logging.Handler | None = None


def setup_log(path: Path) -> None:
    """Sends the warnings and errors of the app and of the modules it calls to `path`, turned over at
    LOG_MAX_BYTES. The same path again changes nothing; another one (a Backend of other Roots) moves the file there.

    Args:
        path: the errors.log to write.
    """
    path = Path(path)
    old = _LogState.handler
    if old is not None and getattr(old, "baseFilename", "") == os.path.abspath(path):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if sys.stderr is not None:          # pythonw has no console
            sys.stderr.write(f"no log folder {path.parent}: {e}\n")
        return
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=LOG_MAX_BYTES, backupCount=1, encoding="utf-8",
                                                   delay=True)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(name)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    if old is not None:
        root.removeHandler(old)
        old.close()
    root.addHandler(handler)
    _LogState.handler = handler
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)


def install_excepthooks() -> None:
    """An exception nothing caught is logged with its trace and the app goes on. PyQt aborts the process on an
    exception out of a Qt callback (a timer, a signal) unless sys.excepthook is the app's own; threads log theirs."""
    def on_exception(kind: type[BaseException], value: BaseException, tb: Any) -> None:
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, tb)
            return
        log.critical("uncaught exception", exc_info=(kind, value, tb))

    def on_thread_exception(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "?"
        log.critical("uncaught exception in thread %s", name, exc_info=(args.exc_type, args.exc_value,
                                                                         args.exc_traceback))
    sys.excepthook = on_exception
    threading.excepthook = on_thread_exception


def guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    """For a function Qt calls (a timer, a signal): an exception out of it, SystemExit from library code included,
    is logged and the call returns None, so neither the event loop nor the app goes down."""
    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (Exception, SystemExit):
            log.exception("in %s", getattr(fn, "__qualname__", fn))
            return None
    return call


def clean_json(value: Any) -> Any:
    """`value` with every NaN or infinite float turned into None: strict JSON (the page's JSON.parse) has neither."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def to_json(value: Any) -> str:
    """What crosses the bridge: strict JSON, NaN and infinities as null."""
    return json.dumps(clean_json(value), allow_nan=False)


INTERNAL_ANSWER = to_json({"error": "internal", "args": {}, "message": ""})   # when even a failure cannot be said


def plain_value(v: Any) -> Any:
    """One value of an error's sentence as JSON can carry it: numbers, text and flags as they are, lists and tuples
    as lists of such values (a size, the names of files), anything else (a path, an object) as its text."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [plain_value(x) for x in v]
    return str(v)


def plain_args(values: dict[str, Any]) -> dict[str, Any]:
    """The values of an error's sentence, each through plain_value."""
    return {str(k): plain_value(v) for k, v in values.items()}


def write_text_atomic(path: Path, text: str) -> None:
    """Writes `text` to `path` through a temporary file in the same folder, flushed and synced, then put in its place:
    a crash or a power cut leaves the old file or the new one, never half of one. The replacement is tried again while
    another program (Drive, an antivirus) holds the file for a moment.

    Args:
        path: the file to write (its folder is made if needed).
        text: what it holds, written as UTF-8.

    Raises:
        OSError: the folder cannot be written, or the file stays held.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(WRITE_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == WRITE_RETRIES - 1:
                    raise
                time.sleep(WRITE_RETRY_S)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_json_atomic(path: Path, data: Any) -> None:
    """`data` as indented JSON in `path`, atomically (write_text_atomic); the same text as json.dumps(indent=1)."""
    write_text_atomic(path, json.dumps(data, indent=1))


def append_csv_row(path: Path, row: dict[str, Any], columns: tuple[str, ...]) -> None:
    """One row added at the end of a CSV (its header first when the file is new or empty), flushed and synced: the
    rows already there are never rewritten, so a crash can at most lose the row being added.

    Args:
        path: the CSV file.
        row: the values by column name (other keys are ignored).
        columns: the header of the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns, extrasaction="ignore", lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_json(path: Path, what: str) -> Any:
    """The JSON in `path`, or None when it is missing or cannot be read (said in the log, with `what` it is)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("%s unreadable (%s): %s", what, path, e)
        return None


def captured(fn: Callable[..., Any], *args: Any, sink: list[str] | None = None, **kwargs: Any) -> tuple[Any, str]:
    """Runs fn(*args, log=..., **kwargs) (calibrate_charuco's run_calibrate, run_check, write_board) and returns (its
    result, the lines it logged). Nothing is taken from stdout: a redirection would catch every thread's prints.
    The lines also go into `sink` when given, so that the caller still has them when fn raises."""
    lines: list[str] = sink if sink is not None else []
    result = fn(*args, log=lines.append, **kwargs)
    return result, log_text(lines)


def log_text(lines: list[str]) -> str:
    """Logged lines as the text of a log file."""
    return "".join(f"{line}\n" for line in lines)


def process_alive(pid: int) -> bool:
    """Whether the process `pid` still runs (a process of another user counts as gone). Never signals it: on Windows
    os.kill(pid, 0) would end it."""
    if pid == os.getpid():
        return True
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    code = ctypes.c_uint32()
    try:
        ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
    finally:
        k32.CloseHandle(handle)
    return bool(ok) and code.value == STILL_ACTIVE


def open_externally(path: Path | str) -> None:
    """A file or a folder opened by Windows with its own program (a folder in Explorer). The one place the app hands a
    path to the shell: tests replace it."""
    if hasattr(os, "startfile"):
        os.startfile(str(path))
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


# ── where things are ─────────────────────────────────────────────────────────
class Roots:
    """Data folders: the project's 03_data, or everything under one folder in simulation. Also where the app keeps its
    own files: preferences, staging, the printable sheet (board), the page it shows (local) and errors.log (log)."""

    def __init__(self, simulate: Path | str | None = None) -> None:
        self.simulate = Path(simulate).resolve() if simulate else None
        if self.simulate:
            base = self.simulate
            self.raw, self.protocol, self.processed = base / "raw", base / "protocol", base / "processed"
            self.staging, self.board = base / "staging", base / "board"
            self.prefs = base / "preferences.json"
            self.export_default = base / "scanner_export"
            self.local = base
            self.log = base / "logs" / "errors.log"
        else:
            # the installed app keeps what is recorded with it in the user's Documents (IMPACTGUIDE_DATA, set by
            # scripts/impactguide_start.pyw), laid out as the project's 03_data; else the project's 03_data
            data = os.environ.get("IMPACTGUIDE_DATA")
            if data:
                self.raw, self.processed = Path(data) / "raw", Path(data) / "processed"
                self.protocol = self.processed / "protocol"
            else:
                self.raw, self.protocol = trials.RAW, trials.PROTOCOL
                self.processed = trials.ROOT / "03_data" / "processed"
            self.staging, self.board = LOCAL / "staging", LOCAL / "boards"
            self.prefs = LOCAL / "preferences.json"
            self.export_default = None
            self.local = LOCAL
            self.log = LOCAL / "errors.log"
        self.calib_raw = self.raw / "calibration"
        self.calib_out = self.processed / "calibration"

    def data_folders(self) -> tuple[Path, ...]:
        """The folders whose subfolders the page may ask to open."""
        return (self.raw, self.protocol, self.processed, self.staging, self.board)


class Prefs:
    """What outlives a session on this computer: language, page, phase, scanner folder, phone.

    Saved atomically on every change. A file that cannot be parsed is kept aside as preferences.bad.json (and said in
    the log) and the defaults are used; a file that cannot be read at all is left alone and nothing is saved this
    session, so a moment's lock never costs the pairing or the caliper readings."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "lang": "", "page": "trials", "phase": "campaign", "scanner_folder": "",
        "calib_mode": "photo", "calib_kind": "calibration", "calib_plan": "quick",
        # the photo taken by itself when the phone holds the guide's pose still (auto_shot.py); never a countdown
        "calib_auto": True,
        # SlantUI's sounds (the speaker on the band, Ctrl+M): on until switched off
        "sound": True,
        "phone": {"camera_dir": acquisition.CAMERA_DIR, "shutter": "volume_down", "switch_modes": True,
                  "link": phone_link.AUTO, "wifi_address": "", "phone_id": "", "wifi_paired": False},
        "live": {"board": "sheet", "face": True, "mandible": False, "axes": True, "registration": "",
                 "opacity": live_ar.OPACITY, "check_method": live_check.SETTINGS["method"], "check_auto": False},
        # the calibration board: on this monitor (in the page), the 3D-printed plate or a printed A4 sheet,
        # with the caliper readings of each (mm); chosen: the first question of Calibrate was answered
        "board": {"kind": "screen", "chosen": False, "plate": {}, "sheet": {}},
        # the ruler check of each monitor (screen_board.with_ruler): pitch_mm, measured_mm, data_mm, date
        "monitors": {}}

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.data = copy.deepcopy(self.DEFAULTS)
        self.saving = True          # False when the file exists but could not be read: it is never overwritten
        for k, v in self._read().items():
            if isinstance(self.data.get(k), dict):
                if isinstance(v, dict):
                    self.data[k].update(v)
                else:
                    log.warning("preferences: %s is not a group of settings, the defaults are used", k)
            else:
                self.data[k] = v
        # what older versions kept: the printed board is the sheet; the video mode is now the stream's
        b = self.data["board"]
        if b.get("kind") == "printed":
            b["kind"] = "sheet"
        for k in ("monitor", "place", "split"):
            b.pop(k, None)
        if self.data.get("calib_mode") not in CALIB_MODES:
            self.data["calib_mode"] = "photo"
        self.data.pop("square_mm", None)
        phone = self.data["phone"]
        if not isinstance(phone.get("camera_dir"), str) or not CAMERA_DIR_RE.fullmatch(phone["camera_dir"]):
            log.warning("preferences: camera folder %r refused, the default is used", phone.get("camera_dir"))
            phone["camera_dir"] = acquisition.CAMERA_DIR
        if phone.get("shutter") not in acquisition.KEYCODES:
            phone["shutter"] = self.DEFAULTS["phone"]["shutter"]
        self._saved = json.dumps(self.data, indent=1)

    def _read(self) -> dict[str, Any]:
        """What the file holds, {} when it is missing or broken (a broken one kept aside)."""
        if not self.path.exists():
            return {}
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("preferences unreadable (%s): the defaults are used and nothing is saved this session", e)
            self.saving = False
            return {}
        try:
            saved = json.loads(text)
            if not isinstance(saved, dict):
                raise ValueError("not a JSON object")
            return saved
        except ValueError as e:
            bad = self.path.with_name(self.path.stem + ".bad.json")
            log.warning("preferences broken (%s): kept as %s, the defaults are used", e, bad.name)
            try:
                os.replace(self.path, bad)
            except OSError:
                log.warning("preferences: %s could not be kept aside", self.path, exc_info=True)
                self.saving = False
            return {}

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def set(self, key: str, value: Any) -> None:
        """Sets one preference and saves the file (only when something changed)."""
        self.data[key] = value
        self.save()

    def save(self) -> None:
        """Writes the preferences atomically; a failure is logged, never raised (the app goes on with them)."""
        text = json.dumps(self.data, indent=1)
        if not self.saving or text == self._saved:
            return
        try:
            write_text_atomic(self.path, text)
            self._saved = text
        except OSError:
            log.warning("preferences not saved to %s", self.path, exc_info=True)


def file_url(path: Path | str) -> str:
    """The file:// URL the page loads a local file by."""
    return QUrl.fromLocalFile(str(path)).toString()


def kind_of(step: trials.Step) -> str:
    """What a step asks for: photo, video, scan, highspeed, spheres, release, close or plain."""
    if step.files:
        return trials.file_kind(step.files[0])
    return step.key if step.key in ("spheres", "release", "close") else "plain"


class ScanWatch:
    """The scanner's export folder watched for the scan of one step of one trial. poll() runs in a worker (the folder
    may be a network share or Drive), one at a time: acquisition.ExportWatch takes a file only once its size and time
    have stayed the same for acquisition.SCAN_SETTLE_S and no program holds it open, so a large scan still being
    written is never staged cut short."""

    def __init__(self, trial: str, key: str, folder: Path) -> None:
        self.trial, self.key = trial, key
        self.watch = acquisition.ExportWatch(folder)

    def poll(self) -> Path | None:
        """The scan, once it is whole; None until then."""
        return self.watch.poll()


# ── the bridge ───────────────────────────────────────────────────────────────
class Backend(Bridge):
    """What the page may ask. Every slot answers JSON and never raises: an exception out of a slot
    would take the window down, so a failure is an answer {"error": code, "args": {...}, "message": ""}.

    Args:
        roots: where the data and the app's own files are.
        prefs: the preferences of this computer.
        lang: the language asked for on the command line (else the one kept, else English).
        replay: recorded frames for the live page in place of the phone.
    """

    happened = pyqtSignal(str)  # results of background jobs: {"type": ..., ...}

    def __init__(self, roots: Roots, prefs: Prefs, lang: str | None = None, replay: Path | None = None) -> None:
        super().__init__()
        self.roots, self.prefs = roots, prefs
        setup_log(roots.log)
        self.replay = replay        # recorded frames for the live page in place of the phone
        self.lang = lang if lang in LANGS else (prefs["lang"] if prefs["lang"] in LANGS else "en")  # English unless chosen
        self.start_page = "home"    # the page opens at home, or at a page asked for on the command line
        self.on_page_ready = None
        self._runs = {}
        self._jobs = queue.Queue()
        self._busy = set()
        self.cur = {}            # trial in progress: id, step index, staged files, recording, watched scan
        self.opts = {"reference": False, "settings": None}
        self._phone = None
        self.phone_status = {"ok": False, "cause": "unknown"}
        self._pending_cal = {}
        self._film = None           # a film for a calibration: {"folder", "since", "recording", "n"} while it records
        self._watch = None
        self._scan_error = ""       # why the scanner folder could not be listed, last said in the log
        self._live = None
        self._live_server = None
        self._check = None          # the registration check of the live view (live_check.py), made with it
        self._view = None           # the live AR for the phone's screen (phone_view.ViewServer), made when opened
        self._frames = LiveFrames(self)     # what that server reads, the same across a rebind
        self._view_lan = False      # the operator asked to show that view on another screen of the network
        self._view_bound = ""       # where the server listens now: 127.0.0.1, or 0.0.0.0 for the network
        self._view_opened = False   # opened on the phone through adb: its tunnel is laid again after a reconnection
        self._phone_lock = threading.Lock()     # one adb conversation with the phone's camera at a time
        self._check_queue = []      # [(folder, photo)] waiting for their check, in the order they came
        self._check_now = None      # (folder, photo) under its check now
        self._importing = set()     # (folder, name) of photos added by hand while they are copied in
        # the photos of a calibration on their way, each with the pose of the plan it is for: shooting (the shutter
        # not pressed yet), shot (pressed, the file not whole yet), copying, checking; gone once checked
        self._shots = []
        self._copy_queue = []       # [(shot, check)] taken and waiting for their copy, one copy at a time
        self._take_wanted = False   # Take came while a photo was being taken: the next one follows it at once
        self._calib_waiting = None  # Calibrate came while photos were on their way: the photos left out, until then
        self._auto = None           # the photo taken by itself (auto_shot.Watcher) while the Photos step is open
        self._auto_view = {"state": "off", "score": 0.0}
        self._auto_mode = "photo"   # the camera mode it watches for
        self._auto_snapshot = None  # what it watches for, for the simulated view (read on its thread)
        self._sim_view = None       # the simulated phone's view of the guide's pose, drawn once per pose
        self._watching = False      # the page has the Photos step open
        self._preview = None        # the phone's screen mirrored for it (auto_shot.preview_stream), photo mode
        self._preview_try = 0.0     # time.monotonic() of its last start
        self._countdown_said = False    # the camera app counted down all the same: said once
        self._calib_log = {}        # mode -> what the last calibration or check said (calib_log_open)
        self._lay_scale = 1.0       # screen pixels per device pixel of the page when the board was laid out
        self._plate_warned = False  # the plate file could not be read, and the log said so
        self._refused_readings = set()  # caliper readings of the preferences that are not lengths, said once
        self._connections = []      # [(signal, slot)] made to Qt objects, undone at shutdown
        self._qr = ""               # the pairing QR code on show, while it waits for the phone
        self._wifi_on = False       # Wireless debugging switched on from the PC, over the cable
        self._svgs = {}
        self._link_was = {}
        self._area = None           # the page's area for the board (board_area), device pixels
        self._board_on = False      # a step shows the board on this monitor now
        self._lay = None            # the board laid out for that area (screen_board.layout)
        self._lay_err, self._lay_mon, self._lay_exact = "", None, True
        self._board_img = None      # its image, as the page draws it
        self._board_want = None     # (key, layout) whose image waits for the one being made (_board_image)
        self._page_errors = 0       # errors of the page's scripts this run (page_error)
        self._screen_watch = False
        self._watched_screen = None
        self._live_for = "live"     # what the live AR runs for: the live view, or a calibration of the stream
        self._win_mons = (None, {})  # what Windows says of the monitors, read again when Qt's screens change
        self.window = None          # the app's window (build): the monitor it is on sizes the board
        self._via = None            # the link the phone was last found on (cable or Wi-Fi)
        self.link = self._make_link()
        app = QApplication.instance()
        if app is not None:         # a monitor plugged in or out: the board is laid out again
            self._connect(app.screenAdded, self._screens_changed)
            self._connect(app.screenRemoved, self._screens_changed)
        self._pump = QTimer(self)
        self._pump.setInterval(JOB_PUMP_MS)
        self._pump.timeout.connect(self._drain)
        self._pump.start()
        self._scan_timer = QTimer(self)
        self._scan_timer.setInterval(SCAN_POLL_MS)
        self._scan_timer.timeout.connect(self._poll_scan)
        self._scan_timer.start()
        # the twin: its model made again from the bench scene once the scene rests (twin.py); a simulation starts
        # from the app's own model; IMPACTGUIDE_TWIN_AUTO=0 makes none (the tests)
        auto = os.environ.get("IMPACTGUIDE_TWIN_AUTO", "1") != "0"
        self.twin = twin.Watcher(roots.local / "twin", blender=True if auto else None,
                                 seed=twin.CACHE if roots.simulate else None)
        self._twin_said = (self.twin.current() or {}).get("version")
        # the Scans page: the scanner's folder watched for every scan, each measured as it comes (scans_page.py)
        self.scans = scans_page.ScansPage(self)
        # the Dataset page: what the dataset holds, its check, the package (dataset_page.py)
        self.dataset = dataset_page.DatasetPage(self)
        self._twin_timer = QTimer(self)
        self._twin_timer.setInterval(TWIN_POLL_MS)
        self._twin_timer.timeout.connect(self._twin_poll)
        self._twin_timer.start()
        self.link.watch(self._on_link)

    def _connect(self, signal: Any, slot: Callable[..., Any]) -> None:
        """signal.connect(slot), undone at shutdown (a second Backend of the same process must not hear the first's)."""
        signal.connect(slot)
        self._connections.append((signal, slot))

    @guarded
    def _screens_changed(self, *_: Any) -> None:
        self._emit({"type": "board", "action": "screen"})

    # ── background jobs ──────────────────────────────────────────────────
    def _spawn(self, name: str, fn: Callable[[], Any], done: Callable[[Any, BaseException | None], None]) -> bool:
        """Runs fn() on a thread of its own; done(result, error) runs on the GUI thread through the pump. fn gets
        values computed on the GUI thread (never Qt objects or the backend's state). False when a job of that name
        is still running (nothing is started)."""
        if name in self._busy:
            return False
        self._busy.add(name)

        def work() -> None:
            try:
                self._jobs.put((name, done, fn(), None))
            except BaseException as e:  # SystemExit from the command-line helpers as well
                self._jobs.put((name, done, None, e))

        threading.Thread(target=work, daemon=True, name=f"job-{name}").start()
        return True

    @guarded
    def _drain(self) -> None:
        """The results of the jobs that ended, each handed to its done() on this (the GUI) thread."""
        while True:
            try:
                name, done, result, error = self._jobs.get_nowait()
            except queue.Empty:
                break
            self._busy.discard(name)
            try:
                done(result, error)
            except (Exception, SystemExit):
                log.exception("after the job %s", name or "link")
                # the job named, so the page ends that work only
                self._emit({"type": "error", "cause": "internal", "error": "internal", "job": name or "link",
                            "args": {}, "message": ""})

    @guarded
    def _poll_scan(self) -> None:
        """The scanner's export folder listed in a worker; the scan, once complete, staged for the step that waits.
        While the trial's page waits for no step of its own, the Scans page's watch takes every export."""
        watch = self._watch
        if watch is None:
            self.scans.poll()
            return

        def done(found: Path | None, error: BaseException | None) -> None:
            if error is not None:
                said = f"{type(error).__name__}: {error}"
                if said != self._scan_error:        # once per cause: a share that stays out never floods the log
                    log.warning("scanner folder not listed", exc_info=error)
                    self._scan_error = said
                return
            self._scan_error = ""
            if found is None or self._watch is not watch:
                return
            self._watch = None
            self.scans.consumed(found)          # never taken again by the Scans page's watch
            if self.cur.get("trial") != watch.trial:
                log.info("scan %s left: trial %s is no longer open", found.name, watch.trial)
                return
            # staged for this step, unless the scan was taken at the arm position of another step that waits
            self.scans.came(found, prefer=watch.key)
        self._spawn("scan_poll", watch.poll, done)

    def _emit(self, payload: dict[str, Any]) -> None:
        self.happened.emit(to_json(payload))

    def _err(self, e: BaseException | None, cause: str | None = None) -> dict[str, Any]:
        """The answer of a failure: {"error": code, "args": {values of its sentence}, "message": a short plain
        detail or ""}, with "cause" when the link is why. Faults go to the log with their trace, refusals as a line;
        exception texts, adb output and traces never reach the page.

        Args:
            e: what was raised.
            cause: the code to answer with when the error has none of its own the page can name.
        """
        if isinstance(e, UserError):
            log.info("refused: %s", e)
            return {"error": e.code, "args": plain_args(e.values), "message": e.detail}
        log.warning("%s", cause or "failed", exc_info=(type(e), e, e.__traceback__) if e is not None else None)
        if isinstance(e, phone_link.LinkError):
            return {"error": e.cause, "cause": e.cause, "args": {}, "message": ""}
        if isinstance(e, acquisition.AcquisitionError):
            return {"error": e.cause, "args": {}, "message": e.detail if e.cause in NAMED_DETAIL else ""}
        if isinstance(e, SystemExit):           # library code that still ends with sys.exit(message)
            return {"error": cause or "failed", "args": {}, "message": ""}
        link_cause = getattr(e, "cause", "")      # live.LiveError: the link is why
        if link_cause in phone_link.CAUSES:
            return {"error": cause or link_cause, "cause": link_cause, "args": {}, "message": ""}
        code = getattr(e, "code", None)           # a coded error of the modules: its code and values
        if isinstance(code, str) and code:
            values = next((v for v in (getattr(e, "values", None), getattr(e, "params", None)) if isinstance(v, dict)),
                          {})
            return {"error": code, "args": plain_args(values), "message": ""}
        if isinstance(e, FileExistsError):
            return {"error": "exists", "args": {}, "message": ""}
        return {"error": cause or "internal", "args": {}, "message": ""}

    def _answer(self, fn: Callable[[], Any]) -> str:
        """fn()'s answer as JSON, or the answer of its failure: a slot must never raise."""
        try:
            return to_json(fn())
        except (Exception, SystemExit) as e:
            try:
                return to_json(self._err(e))
            except Exception:
                log.exception("the answer of a failure could not be made")
                return INTERNAL_ANSWER

    def _live_event(self, action: str, purpose: str | None = None, **fields: Any) -> dict[str, Any]:
        """A `live` event, with what the stream runs for (live or calibration): the page's two modules each take
        their own."""
        return {"type": "live", "action": action, "purpose": purpose or self._live_for, **fields}

    def _live_started(self, purpose: str) -> Callable[[Any, BaseException | None], None]:
        """The done() of a start of the live stream: failed with why; started; or stopped (reason cancelled) when a
        Stop came while it opened, LiveAR.start then returning without a stream."""
        def done(_: Any, error: BaseException | None) -> None:
            if error is not None:
                self._emit(self._live_event("failed", purpose, **self._err(error, "live_failed")))
            elif self._live is not None and self._live.running:
                self._emit(self._live_event("started", purpose))
            else:
                self._emit(self._live_event("stopped", purpose, reason="cancelled"))
        return done

    # ── the app ──────────────────────────────────────────────────────────
    @pyqtSlot(result=str)
    def app_info(self) -> str:
        if self.on_page_ready is not None:
            ready, self.on_page_ready = self.on_page_ready, None
            guarded(ready)()
        return to_json({"name": APP_NAME, "version": APP_VERSION, "lang": self.lang,
                        "langs": [{"code": c, "native": n, "flag": f} for c, n, f in LANGUAGES],
                        "page": self.start_page,
                        "simulate": bool(self.roots.simulate), "qt": "PyQt6" if USE_QT6 else "PyQt5",
                        "link": self.link.via, "sound": bool(self.prefs["sound"])})

    @pyqtSlot(str)
    @guarded
    def set_lang(self, code: str) -> None:
        if code in LANGS:
            self.lang = code
            self.prefs.set("lang", code)
            if self._view is not None:
                self._view.words, self._view.lang = view_words(code), code

    @pyqtSlot(str)
    @guarded
    def set_page(self, page: str) -> None:
        if page in PAGES:
            self.prefs.set("page", page)

    @pyqtSlot(bool)
    @guarded
    def set_sound(self, on: bool) -> None:
        """The sounds switched on or off on the band (SlantUI's switch): kept for the next start."""
        self.prefs.set("sound", bool(on))

    @pyqtSlot(str)
    @guarded
    def page_error(self, text: str) -> None:
        """An error of the page's scripts (ui/00-errors.js), written to the log only, at most PAGE_ERRORS_MAX of
        them a run."""
        self._page_errors += 1
        if self._page_errors <= PAGE_ERRORS_MAX:
            log.warning("page: %s", str(text)[:PAGE_ERROR_CHARS])

    def _data_folder(self, path: Path | str) -> Path:
        """`path` as an existing folder under the data roots (Roots.data_folders).

        Raises:
            UserError: not_found when there is no such folder, bad_value when it lies elsewhere.
        """
        p = Path(path).resolve()
        if not p.is_dir():
            raise UserError("not_found", p.name)
        roots = [r.resolve() for r in self.roots.data_folders()]
        if not any(p == r or r in p.parents for r in roots):
            raise UserError("bad_value")
        return p

    @pyqtSlot(str, result=str)
    def open_path(self, path: str) -> str:
        """A folder of the data opened in Explorer; any other path is refused (never run, never opened)."""
        def go() -> dict[str, Any]:
            open_externally(self._data_folder(path))
            return {"ok": True}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def trial_open_folder(self, trial: str) -> str:
        """The folder of a trial of the phase in use opened in Explorer."""
        def go() -> dict[str, Any]:
            run = self.run()
            if not re.fullmatch(r"[A-Za-z]\d+", trial or ""):
                raise UserError("bad_value")
            open_externally(self._data_folder(run.folder / trial))
            return {"ok": True}
        return self._answer(go)

    # ── the phone ────────────────────────────────────────────────────────
    def phone(self) -> acquisition.Phone:
        """The phone (a stand-in in simulation), made at first use on the GUI thread."""
        if self._phone is None:
            if self.roots.simulate:
                self._phone = acquisition.FakePhone(photo_px=self.run().config.photo_px, link=self.link)
            else:
                p = self.prefs["phone"]
                self._phone = acquisition.Phone(camera_dir=p["camera_dir"], shutter=p["shutter"],
                                                switch_modes=bool(p["switch_modes"]), link=self.link)
        return self._phone

    def _with_phone(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        """fn wrapped for a worker: one conversation with the phone at a time, whichever job asks."""
        def call() -> Any:
            with self._phone_lock:
                return fn()
        return call

    def _phone_event(self) -> dict[str, Any]:
        """The `phone` event: the phone's state without adb's own text (that stays in the log), with the files copied
        from the phone that could not be deleted there (left_on_phone)."""
        return {"type": "phone", **{k: v for k, v in self.phone_status.items() if k != "detail"},
                "left_on_phone": self._left_on_phone()}

    def _left_on_phone(self) -> list[str]:
        return list(getattr(self._phone, "left_on_phone", None) or [])

    def _check_left(self, before: int) -> None:
        """After an acquisition: files copied but left on the phone (their deletion failed) logged and said."""
        left = self._left_on_phone()
        if len(left) > before:
            log.warning("left on the phone after the copy: %s", ", ".join(left[before:]))
            self._emit(self._phone_event())

    @pyqtSlot()
    @guarded
    def phone_refresh(self) -> None:
        phone = self.phone()

        def done(status: dict[str, Any] | None, error: BaseException | None) -> None:
            self.phone_status = dict(status) if status else {"ok": False, **self._err(error)}
            if self.phone_status.get("detail"):
                log.info("phone: %s (%s)", self.phone_status.get("cause"), self.phone_status["detail"])
            self.phone_status.pop("detail", None)
            self.phone_status["link"] = self.link.via
            self._emit(self._phone_event())
        self._spawn("phone_status", self._with_phone(phone.status), done)

    @pyqtSlot(result=str)
    def phone_prefs(self) -> str:
        return to_json({**self.prefs["phone"], "adb": acquisition.find_adb() or "", "status": self.phone_status})

    # ── the link: cable or Wi-Fi ─────────────────────────────────────────
    def _make_link(self):
        """The process's link (phone_link.get()), found by itself: the cable when the phone is on it, else the Wi-Fi
        it was paired on (remembered between sessions). In simulation a simulated phone on the cable that pairs on
        Wi-Fi with the code 123456 or the QR code."""
        p = self.prefs["phone"]
        if self.roots.simulate:
            link = phone_link.set_default(phone_link.Link(runner=phone_link.SimulatedAdb()))
        else:
            link = phone_link.get()
        return link.configure(mode=phone_link.AUTO, address=p.get("wifi_address", ""), phone_id=p.get("phone_id", ""),
                              paired=bool(p.get("wifi_paired")))

    def _on_link(self, status: dict[str, Any]) -> None:
        """From the link's own thread: handed to the GUI thread through the pump."""
        self._jobs.put(("", self._link_changed, status, None))

    def _link_changed(self, st: dict[str, Any], _error: BaseException | None) -> None:
        """A change of the link (on the GUI thread): the preferences, the pill, the page told; on a switch between
        cable and Wi-Fi a running live stream follows the phone; the phone's own view gets its tunnel again."""
        was, self._link_was = self._link_was, st
        kept = self.link.remembered()
        if kept != {k: self.prefs["phone"].get(k) for k in kept}:
            self.prefs.set("phone", {**self.prefs["phone"], **kept})
        self.phone_status = {**self.phone_status, "ok": st["ok"], "cause": st["cause"] or ("" if st["ok"] else "unknown"),
                             "link": st["via"], "state": st["state"], "model": st["model"], "serial": st["serial"]}
        self._emit(self._phone_event())
        back = st["ok"] and was.get("state") in ("reconnecting", "lost") and was.get("via") == st["via"]
        self._emit({"type": "link", "action": "state", "back": back, **st})
        switched = st["ok"] and self._via is not None and st["via"] != self._via
        if switched:
            # the phone is now on the other link (cable plugged in or out): the app goes on with it, a live view too
            self._emit({"type": "link", "action": "via", "via": st["via"]})
            if self._live is not None and self._live.running and not self._live.synthetic:
                self._live.stop()
                self._spawn("live", self._live.start, self._live_started(self._live_for))
        if st["ok"]:
            self._via = st["via"]
        became_ok = st["ok"] and (not was.get("ok") or was.get("via") != st["via"])
        if (back or became_ok) and self._view_opened and self._view is not None:
            # adb drops its tunnels with the link: the phone's page goes on once the tunnel is laid again
            port, link = self._view.port, self.link

            def laid(_: Any, error: BaseException | None) -> None:
                if error is not None:
                    log.warning("phone view: tunnel not laid again", exc_info=error)
            self._spawn("view_tunnel", lambda: link.reverse(port), laid)

    def _link_state(self) -> dict[str, Any]:
        st = self.link.status()
        live = self._live.status() if self._live is not None and self._live.running else {}
        st["stream"] = {"running": bool(live), "fps": live.get("fps_stream", 0.0),
                        "gap_p95_ms": live.get("gap_p95_ms"), "waiting": live.get("link_waiting", False),
                        "link": live.get("link")}
        # the latency of the link in use (the mode is "auto" unless a test asks for one)
        st["latency_ms"] = live_ar.STREAM_LATENCY_MS.get(st["via"] or st["mode"])
        st["qr"] = self._svg(self._qr) if self._qr and st["qr_waiting"] else ""
        st["busy"] = sorted(self._busy & {"link", "pair", "qr", "connect", "view_open", "wifi_on"})
        st["wifi_on"] = self._wifi_on
        st["simulate"] = bool(self.roots.simulate)
        st["view"] = self._view_state()
        return st

    @pyqtSlot(result=str)
    def link_state(self) -> str:
        return self._answer(self._link_state)

    @pyqtSlot(result=str)
    def link_wifi(self) -> str:
        """Wi-Fi wanted: with the phone on the cable, Wireless debugging is switched on from here (one step less on
        the phone), then a pairing QR code is shown; the phone that scans it is paired and connected, and the app
        goes on with it by itself when the cable is unplugged."""
        def go() -> dict[str, Any]:
            def done(on: Any, error: BaseException | None) -> None:
                if error is not None:
                    log.warning("Wireless debugging not switched on from the PC", exc_info=error)
                self._wifi_on = bool(on)
                self.link_qr()
                self._emit({"type": "link", "action": "pairing"})
            if self.link.status()["via"] == phone_link.USB and self.link.status()["ok"]:
                if not self._spawn("wifi_on", self.link.wifi_debugging_on, done):
                    return {"error": "busy"}
            else:
                self._wifi_on = False
                self.link_qr()
            return self._link_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def link_mode(self, mode: str) -> str:
        """auto (the cable when the phone is on it, else Wi-Fi), or only the cable, or only Wi-Fi: for the tests and
        the command line; the panel never asks. A live view running on the other link stops."""
        def go() -> dict[str, Any]:
            if mode not in phone_link.MODES:
                return {"error": "bad_value", "message": "link"}
            if mode != self.link.mode:
                if self._live is not None and self._live.running and not self._live.synthetic:
                    self._live.stop()
                    self._emit(self._live_event("stopped", reason="link"))
                self.link.configure(mode=mode)
                self.prefs.set("phone", {**self.prefs["phone"], **self.link.remembered()})
                if not self._phone_in_use():
                    self._phone = None
                self._spawn("link", self.link.check, self._logged("link check"))
            return self._link_state()
        return self._answer(go)

    def _phone_in_use(self) -> bool:
        """A photo, a video or a status is being asked of the phone, or a video records: the Phone object stays (it
        knows what the recording started from)."""
        return (bool(self.cur.get("recording")) or bool(getattr(self._phone, "recording", False))
                or bool(self._busy & {"phone", "phone_status"}))

    @staticmethod
    def _logged(what: str) -> Callable[[Any, BaseException | None], None]:
        """A done() for a job whose result nobody waits for: its failure goes to the log."""
        def done(_: Any, error: BaseException | None) -> None:
            if error is not None:
                log.warning("%s failed", what, exc_info=error)
        return done

    def _link_job(self, name: str, fn: Callable[[], Any]) -> bool:
        def done(_: Any, error: BaseException | None) -> None:
            if error is not None:
                self._emit({"type": "link", "action": "failed", "job": name, **self._err(error, "pair_failed")})
            else:
                self.prefs.set("phone", {**self.prefs["phone"], **self.link.remembered()})
                self._emit({"type": "link", "action": "paired" if name in ("pair", "qr") else "connected"})
        return self._spawn(name, fn, done)

    @pyqtSlot(str, str, result=str)
    def link_pair(self, address: str, code: str) -> str:
        """Pairing with the code and the address the phone shows, then the connection; 'link' events say how."""
        def go() -> dict[str, Any]:
            if not self._link_job("pair", lambda: self.link.pair_code(address, code)):
                return {"error": "busy"}
            return self._link_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def link_qr(self) -> str:
        """A new pairing QR code; the phone that scans it is paired and connected in the background."""
        def go() -> dict[str, Any]:
            self._qr = self.link.qr_start()     # a job already waiting goes on with the new code

            def done(result: Any, error: BaseException | None) -> None:
                self._qr = ""
                if error is not None:
                    self._emit({"type": "link", "action": "failed", "job": "qr", **self._err(error, "pair_failed")})
                elif result == "ready":
                    self.prefs.set("phone", {**self.prefs["phone"], **self.link.remembered()})
                    self._emit({"type": "link", "action": "paired"})
            if "qr" not in self._busy:
                self._spawn("qr", self.link.qr_wait, done)
            return self._link_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def link_qr_cancel(self) -> str:
        def go() -> dict[str, Any]:
            self._qr = ""
            self.link.qr_cancel()
            return self._link_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def link_connect(self, address: str) -> str:
        """adb connect to the address the phone shows under IP address & Port (empty: the one it announces)."""
        def go() -> dict[str, Any]:
            if not self._link_job("connect", lambda: self.link.connect(address)):
                return {"error": "busy"}
            return self._link_state()
        return self._answer(go)

    # ── the live view on the phone's own screen ──────────────────────────
    def _svg(self, text: str) -> str:
        """A QR code as SVG, kept: the pane asks for its state every second."""
        if text not in self._svgs:
            if len(self._svgs) > QR_CACHE:
                self._svgs.clear()
            self._svgs[text] = phone_view.qr_svg(text)
        return self._svgs[text]

    def _view_host(self) -> str:
        """Where the phone-screen server listens: every interface only when the operator asked to show the view on
        another screen of the network; else this PC alone, which is enough for the phone through adb. In simulation
        always this PC alone: the tests never wake the firewall."""
        return "0.0.0.0" if self._view_lan and not self.roots.simulate else "127.0.0.1"

    def view(self, port: int = phone_view.PORT) -> phone_view.ViewServer:
        """The server of the phone's screen, made now if there is none, on `port` when it is free (Windows may then
        ask once about its firewall, when it listens on the network)."""
        if self._view is None:
            self._view = phone_view.ViewServer(self._frames, host=self._view_host(), port=port,
                                               words=view_words(self.lang), colors=dict(PALETTES[PALETTE].values),
                                               lang=self.lang)
            self._view_bound = self._view_host()
        return self._view

    def _rebind_view(self) -> None:
        """The server listening where it should now (view_lan changed): made again, with the same address for the
        phone's page when the port is free again; the tunnel laid again when the port changed."""
        old = self._view
        if old is None or self._view_bound == self._view_host():
            return
        token, port = old.token, old.port
        old.close()
        self._view = None
        view = self.view(port)      # the same port when it is free again: the phone's page keeps its address
        view.token = token          # (the handler reads the token at each request)
        if view.port != port and self._view_opened:
            # another port: the tunnel laid for it and the phone's page opened again at its new address
            link, url, new_port = self.link, view.url_phone, view.port

            def relay() -> None:
                link.reverse(new_port)
                link.open_url(url, new_port)
            self._spawn("view_tunnel", relay, self._logged("phone view moved to another port"))

    def _view_state(self) -> dict[str, Any]:
        """The phone's own view for the Phone panel; nothing is made or opened by asking."""
        view = self._view
        live = self._live.status() if self._live is not None and self._live.running else {}
        lat = live.get("latency") or {}
        url = ""
        if view is not None and self._view_lan:
            url = view.url_lan(phone_view.lan_ip(self.link.status()["phone_ip"]))
        return {"url_lan": url, "qr": self._svg(url) if url else "", "url_phone": view.url_phone if view else "",
                "opened": self._view_opened, "lan": self._view_lan, "live": bool(live), "pc_ms": lat.get("total_ms"),
                "camera_ms": lat.get("stream_ms"), "synthetic": bool(live.get("synthetic")),
                **(view.stats() if view is not None else VIEW_IDLE)}

    @pyqtSlot(result=str)
    def view_open(self) -> str:
        """The live view in the phone's browser, through adb (the phone's 127.0.0.1 is this PC's)."""
        def go() -> dict[str, Any]:
            view, link = self.view(), self.link
            url, port = view.url_phone, view.port

            def done(_: Any, error: BaseException | None) -> None:
                if error is not None:
                    self._emit({"type": "view", "action": "failed", **self._err(error, "unreachable")})
                else:
                    self._view_opened = True
                    self._emit({"type": "view", "action": "opened"})
            if not self._spawn("view_open", lambda: link.open_url(url, port), done):
                return {"error": "busy"}
            return self._link_state()
        return self._answer(go)

    @pyqtSlot(bool, result=str)
    def view_lan(self, on: bool) -> str:
        """The phone's view shown (on) or not (off) to other screens of the same network: the server then listens on
        every interface of this PC, else on 127.0.0.1 only. Answers the link state, whose view.url_lan and view.qr are
        empty while it is off."""
        def go() -> dict[str, Any]:
            ended = self._view_lan and not on
            self._view_lan = bool(on)
            if self._view_lan:
                self.view()
            self._rebind_view()
            if ended and self._view is not None:
                self._end_sharing()
            return self._link_state()
        return self._answer(go)

    def _end_sharing(self) -> None:
        """The view no longer shown to other screens: a new token, so an address handed out over the network ends;
        the phone's own page, when it was open, is opened again at the new address (through adb)."""
        self._view.rotate_token()
        if self._view_opened:
            link, url, port = self.link, self._view.url_phone, self._view.port
            self._spawn("view_open", lambda: link.open_url(url, port), self._logged("phone view opened again"))

    @pyqtSlot(str, result=str)
    def set_phone_prefs(self, payload: str) -> str:
        def go() -> dict[str, Any]:
            p = {**self.prefs["phone"], **json.loads(payload)}
            p["switch_modes"] = bool(p["switch_modes"])
            if p["shutter"] not in acquisition.KEYCODES:
                return {"error": "bad_value", "message": "shutter"}
            if not isinstance(p.get("camera_dir"), str) or not CAMERA_DIR_RE.fullmatch(p["camera_dir"]):
                return {"error": "bad_value", "message": "camera_dir"}
            if self._phone_in_use():
                return {"error": "busy"}
            self.prefs.set("phone", p)
            self._phone = None
            return {"ok": True, **p}
        return self._answer(go)

    # ── trials: the run of a phase ───────────────────────────────────────
    def run(self, phase: str | None = None) -> trials.Run:
        """The run of a phase (the one in use by default), kept for the session."""
        phase = phase or self.prefs["phase"]
        if phase not in self._runs:
            # the calibrations kept: a trial's photo or video of another size than its camera mode's is said
            self._runs[phase] = trials.Run(phase, self.roots.raw, self.roots.protocol, calibration=self.roots.calib_out)
        return self._runs[phase]

    def scanner_folder(self) -> Path | None:
        """The scanner's export folder (the stand-in's in simulation), or None when none was chosen."""
        if self.roots.simulate:
            return self.roots.export_default
        f = self.prefs["scanner_folder"]
        return Path(f) if f else None

    def _plan(self, run: trials.Run) -> dict[str, Any] | None:
        settings = self.opts["settings"]
        return run.next(reference="manual" if self.opts["reference"] else None, settings=settings)

    def _open_trial(self, run: trials.Run) -> dict[str, str] | None:
        """The open trial row, or None; keeps self.cur on it."""
        row = next((r for r in run.log.rows() if r["status"] == trials.OPEN), None)
        if row is None:
            if self.cur.get("trial"):
                self._drop_staged()
                self.cur = {}
            return None
        if self.cur.get("trial") != row["trial"]:
            self._drop_staged()
            self.cur = {"trial": row["trial"], "step": run.resume_step(row["trial"]), "staged": {},
                        "recording": "", "resumed": True}
        return row

    def _drop_staged(self) -> None:
        """The files staged for the open trial deleted (the copies in staging, never a file of the scanner)."""
        for item in (self.cur.get("staged") or {}).values():
            trials.Run.discard(item.get("prepared"))       # its copy made ahead in the trial's folder
            if item.get("temp"):
                self._unlink(Path(item["path"]))
        self._watch = None

    @staticmethod
    def _unlink(path: Path) -> None:
        """A file of the app's own deleted; a failure is logged (it stays in staging)."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("%s not deleted", path, exc_info=True)

    def _steps_json(self, run, row, steps, current):
        out = []
        for i, st in enumerate(steps):
            if row is not None and (st.files or st.fields):
                done = run.step_done(row, st)
            else:
                done = row is not None and i < current
            out.append({"key": st.key, "text": st.text, "files": list(st.files), "fields": list(st.fields),
                        "kind": kind_of(st), "done": bool(done)})
        return out

    def _totals(self, run):
        rows = run.log.rows()
        filled, extra = trials.match(run.sequence, rows)
        return {"slots": len(run.sequence), "filled": len(filled), "pending": len(run.sequence) - len(filled),
                "extra": len(extra), "void": sum(r["status"] == trials.VOID for r in rows), "logged": len(rows)}

    def _trial_state(self):
        run = self.run()
        rows = run.log.rows()
        row = self._open_trial(run)
        state = {"phase": run.phase, "phases": list(trials.PREFIX), "simulate": bool(self.roots.simulate),
                 "totals": self._totals(run), "config": json.loads(json.dumps(asdict(run.config))),
                 "scanner_folder": str(self.scanner_folder() or ""), "phone": self.phone_status,
                 "options": self.opts, "trial": None, "steps": [], "step": 0, "staged": {},
                 "recording": "", "watching": ""}
        if row is not None:
            trial = row["trial"]
            i = next(i for i, r in enumerate(rows) if r["trial"] == trial)
            prev = rows[i - 1] if i else None
            steps = run.steps(trial)
            cur = min(self.cur["step"], len(steps) - 1)
            filled, _ = trials.match(run.sequence, rows)
            slot = next((s for s, t in filled.items() if t == trial), "")
            state["trial"] = {**row, "open": True, "slot": slot, "resumed": self.cur.get("resumed", False),
                              "changes": trials.changes(prev, row), "warnings": [], "warning_codes": [],
                              "prev": self._prev_json(prev)}
            state["steps"] = self._steps_json(run, row, steps, cur)
            state["step"] = cur
            state["staged"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("temp", "preparing", "prepared")}
                               for k, v in self.cur["staged"].items()}
            state["recording"] = self.cur.get("recording", "")
            state["watching"] = self._watch.key if self._watch else ""
            return state
        plan = self._plan(run)
        if plan is None:
            state["complete"] = True
            return state
        prev = rows[-1] if rows else None
        preview_row = {**{k: plan.get(k, "") for k in trials.LOG_FIELDS}}
        steps = trials.trial_steps(run.config, preview_row, prev)
        # warning_codes: the warnings as codes with their values, for the page to say (trials.Run.next)
        state["trial"] = {**preview_row, "open": False, "slot": plan["slot"], "changes": plan["changes"],
                          "warnings": plan["warnings"], "warning_codes": plan.get("warning_codes", []),
                          "extra": plan["slot"] == "", "prev": self._prev_json(prev)}
        state["steps"] = self._steps_json(run, None, steps, 0)
        return state

    @staticmethod
    def _prev_json(prev):
        if prev is None:
            return None
        return {k: prev[k] for k in ("trial", "holder", "angle_deg", "direction_deg", "drape", "cap_shift_mm",
                                     "status", "session")}

    @pyqtSlot(result=str)
    def trials_state(self):
        return self._answer(self._trial_state)

    @pyqtSlot(str, result=str)
    def set_phase(self, phase):
        def go():
            if phase not in trials.PREFIX:
                return {"error": "bad_value", "message": phase}
            if self.cur.get("trial"):
                return {"error": "trial_open", "message": self.cur["trial"]}
            self.prefs.set("phase", phase)
            self.opts = {"reference": False, "settings": None}
            return self._trial_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def trial_options(self, payload):
        """{reference: bool, settings: {holder, angle_deg, direction_deg, drape} | null} for the next trial."""
        def go():
            o = json.loads(payload)
            s = o.get("settings")
            if s:
                run = self.run()
                cfg = run.config
                s = {"holder": str(s["holder"]), "angle_deg": int(s["angle_deg"]),
                     "direction_deg": int(s["direction_deg"]), "drape": int(bool(int(s.get("drape") or 0)))}
                if s["holder"] not in cfg.holders:
                    return {"error": "bad_value", "message": "holder"}
            self.opts = {"reference": bool(o.get("reference")), "settings": s or None}
            return self._trial_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def trial_start(self, payload):
        def go():
            run = self.run()
            if self._open_trial(run) is not None:
                return {"error": "trial_open", "message": self.cur["trial"]}
            plan = self._plan(run)
            if plan is None:
                return {"error": "complete"}
            values = {k: v for k, v in json.loads(payload or "{}").items() if k in ("swing_mass_g", "notes") and v != ""}
            if getattr(run, "pendulum", True) and not (values.get("swing_mass_g") or plan.get("swing_mass_g")):
                return {"error": "swing_mass"}
            if "swing_mass_g" in values:
                values["swing_mass_g"] = f"{float(values['swing_mass_g']):g}"
            row = run.start(plan, **values)
            self.cur = {"trial": row["trial"], "step": 0, "staged": {}, "recording": "", "resumed": False}
            self.opts = {"reference": False, "settings": None}
            return self._trial_state()
        return self._answer(go)

    def _current(self, key: str) -> tuple[trials.Run, dict[str, str], list[trials.Step], int]:
        """(run, open trial row, its steps, index of the step `key`).

        Raises:
            UserError: no_trial when no trial is open, no_step when the open one has no such step.
        """
        run = self.run()
        row = self._open_trial(run)
        if row is None:
            raise UserError("no_trial")
        steps = run.steps(row["trial"])
        i = next((i for i, s in enumerate(steps) if s.key == key), None)
        if i is None:
            raise UserError("no_step", key)
        return run, row, steps, i

    @pyqtSlot(int, result=str)
    def step_go(self, index):
        def go():
            run = self.run()
            row = self._open_trial(run)
            if row is not None:
                n = len(run.steps(row["trial"]))
                self.cur["step"] = max(0, min(int(index), n - 1))
                self._watch = None
            return self._trial_state()
        return self._answer(go)

    @pyqtSlot(str, str, result=str)
    def step_confirm(self, key: str, payload: str) -> str:
        """Closes a step of the open trial: its file filed (never over one already there), its values logged; the
        last step closes the trial, with its notes, in one go."""
        def go() -> dict[str, Any]:
            run, row, steps, i = self._current(key)
            trial, step = row["trial"], steps[i]
            values = {k: v for k, v in json.loads(payload or "{}").items() if v not in ("", None)}
            if step.files:
                name = step.files[0]
                staged = self.cur["staged"].get(key)
                if run.find(trial, name):
                    if staged:
                        # the step was done before and a new file is staged: the filed one is never replaced
                        return {"error": "exists", "message": name}
                elif not staged:
                    return {"error": "missing_file", "message": name}
                else:
                    # its copy into the trial's folder was made ahead in a worker (_prepare_staged): only named here;
                    # one still on its way is waited for (the rest of its copy); stale or failed, it is made now
                    hold = staged.get("preparing")
                    if hold is not None and not hold["event"].is_set():
                        hold["event"].wait(PREPARE_WAIT_S)
                    prepared = staged.get("prepared") or (hold["prep"] if hold is not None else None)
                    run.store(trial, name, staged["path"], move=staged["temp"], prepared=prepared)
                    del self.cur["staged"][key]
                    self.scans.arrived(trial, key)      # a scan filed: its trial's results kept for the twin
            if key == "spheres":
                values = {k: values[k] for k in ("spheres_ok", "cap_shift_mm") if k in values}
                if "cap_shift_mm" in values:
                    values["cap_shift_mm"] = f"{float(values['cap_shift_mm']):.2f}"
            elif key == "close":
                notes = str(values.pop("notes", "")).strip()
                values = {k: int(bool(int(values.get(k, 0)))) for k in ("second_hit", "cap_replaced")}
                if notes:       # in the closing write itself: trials adds them to notes typed at the start
                    values["notes"] = notes
                run.confirm(trial, key, **values)
                self._drop_staged()
                self.cur = {}
                return {**self._trial_state(), "closed": trial}
            else:
                values = {}
            run.confirm(trial, key, **values)
            self.cur["step"] = min(i + 1, len(steps) - 1)
            self._watch = None
            return self._trial_state()
        return self._answer(go)

    def _stage(self, key: str, path: Path | str, temp: bool, trial: str | None = None) -> None:
        """A file for the step `key` of the open trial (of `trial`, when given), kept in staging (temp: the app's own
        copy) until the step is confirmed; any older staged copy of that step deleted.

        Raises:
            UserError: no_trial, no_step (see _current); trial_changed when the open trial is not `trial` (it was
                voided by an edit of the log, which leaves the app's own idea of the open trial behind).
        """
        run, row, steps, i = self._current(key)
        if trial is not None and row["trial"] != trial:
            raise UserError("trial_changed", trial=trial)
        step = steps[i]
        name = step.files[0]
        path = Path(path)
        problems, codes = [], []
        if temp:
            target = self.roots.staging / row["trial"] / (Path(name).stem + path.suffix.lower())
            target.parent.mkdir(parents=True, exist_ok=True)
            if path != target:
                target.unlink(missing_ok=True)
                shutil.move(str(path), str(target))
            path = target
            # with the trial's own configuration and the sizes of the calibrations kept, as codes for the page to say
            codes = run.file_problems(row["trial"], path)
            problems = [v for _, v in trials.deep_check(run.config_for(row), path, run.camera_sizes())]
        old = self.cur["staged"].get(key)
        if old:
            trials.Run.discard(old.get("prepared"))
        if old and old.get("temp") and Path(old["path"]) != path:
            self._unlink(Path(old["path"]))
        self.cur["staged"][key] = {"path": str(path), "url": file_url(path), "name": path.name,
                                   "size": path.stat().st_size, "problems": problems, "problem_codes": codes,
                                   "temp": temp, "kind": trials.file_kind(name)}

    def _staged(self, trial: str, key: str, path: Path | str, temp: bool) -> None:
        """A file that arrived for the step `key` of `trial`: staged and said when that trial is still the open one;
        when it is not (voided, or another one started meanwhile) the app's copy is deleted and the page told, so a
        file is never filed under the wrong trial."""
        if self.cur.get("trial") != trial:
            log.info("%s arrived for %s, which is no longer open: dropped", Path(path).name, trial)
            if temp:
                self._unlink(Path(path))
            self._emit({"type": "acquire_failed", "key": key, "trial": trial, "error": "trial_changed",
                        "args": {"trial": trial}, "message": ""})
            return
        try:
            self._stage(key, path, temp, trial)
        except (UserError, LookupError, OSError, ValueError) as e:
            if temp and isinstance(e, UserError) and e.code in ("no_trial", "trial_changed"):
                self._unlink(Path(path))      # its trial is no longer open: never staged (nor kept) for another
            self._emit({"type": "acquire_failed", "key": key, "trial": trial, **self._err(e, "copy_failed")})
            return
        self._prepare_staged(trial, key)
        self._emit({"type": "staged", "key": key, "trial": trial})

    def _prepare_staged(self, trial: str, key: str) -> None:
        """The file staged for the step `key` copied ahead into its trial's folder by a worker (trials.Run.prepare: the
        copy to the data drive, flushed and read back), while the operator looks at it: Keep then only names it,
        and the window never waits for a copy to a synced drive. Kept with the staged file as "prepared"; a copy
        that comes when that file is no longer the staged one is deleted."""
        item = (self.cur.get("staged") or {}).get(key)
        if not item or self.cur.get("trial") != trial:
            return
        run = self.run()
        name = next((st.files[0] for st in run.steps(trial) if st.key == key and st.files), None)
        if name is None or run.find(trial, name):
            return
        src = item["path"]
        hold = item["preparing"] = {"event": threading.Event(), "prep": None}

        def work() -> dict:
            try:
                hold["prep"] = run.prepare(trial, name, src)
                return hold["prep"]
            finally:
                hold["event"].set()     # a Keep that came meanwhile waits for this, never copies a second time

        def done(prep: dict | None, error: BaseException | None) -> None:
            if error is not None:
                log.warning("%s not copied ahead: copied when it is kept", name, exc_info=error)
                return
            if self.cur.get("trial") != trial or (self.cur.get("staged") or {}).get(key) is not item:
                trials.Run.discard(prep)        # discarded, taken again or its trial closed meanwhile
                return
            item["prepared"] = prep
        if not self._spawn(f"prepare_{key}", work, done):
            item.pop("preparing", None)     # one still running for an older file of the step: copied at Keep
        self.scans.arrived(trial, key)          # a scan: measured at once (scans_page.py)

    def _acquired(self, trial: str, key: str, video: bool = False) -> Callable[[Any, BaseException | None], None]:
        """The done() of a photo or (video) a recording stopped, taken for the step `key` of `trial`."""
        left = len(self._left_on_phone())

        def done(path: Any, error: BaseException | None) -> None:
            if video and self.cur.get("trial") == trial:
                self.cur["recording"] = ""
            self._check_left(left)
            if error is not None:
                self._emit({"type": "acquire_failed", "key": key, "trial": trial, **self._err(error)})
                return
            # the camera app's own timer counted down all the same (no intent lifts it in its Pro mode): said once
            if not video and getattr(self.phone(), "countdown_seen", False) and not self._countdown_said:
                self._countdown_said = True
                self._emit({"type": "calib_photo", "action": "countdown", "folder": ""})
            self._staged(trial, key, path, temp=True)
        return done

    @pyqtSlot(str, result=str)
    def step_shoot(self, key: str) -> str:
        def go() -> dict[str, Any]:
            run, row, steps, i = self._current(key)
            blocked = self._free_camera()
            if blocked:
                return blocked
            out, phone = self.roots.staging / row["trial"], self.phone()
            # the camera app left in its mode, as for a calibration's photos (Pro, the focus by hand): no intent to it,
            # not even the one that lifts its timer (acquisition.Phone.take, keep_mode)
            started = self._spawn("phone", self._with_phone(lambda: phone.shoot(out, keep_mode=True)),
                                  self._acquired(row["trial"], key))
            return {"ok": started} if started else {"error": "busy"}
        return self._answer(go)

    @pyqtSlot(str, str, result=str)
    def step_video(self, key: str, action: str) -> str:
        def go() -> dict[str, Any]:
            run, row, steps, i = self._current(key)
            trial, out, phone = row["trial"], self.roots.staging / row["trial"], self.phone()
            if action == "start":
                blocked = self._free_camera()
                if blocked:
                    return blocked

                def started(_: Any, error: BaseException | None) -> None:
                    if error is not None:
                        if self.cur.get("trial") == trial:
                            self.cur["recording"] = ""
                        self._emit({"type": "acquire_failed", "key": key, "trial": trial, **self._err(error)})
                    elif self.cur.get("trial") != trial:
                        # the trial was voided (or another started) while the recording started: stopped at once,
                        # its file dropped and said as trial_changed (_staged), the phone never left recording
                        log.info("recording for %s stopped: the trial is no longer open", trial)
                        self._spawn("phone", self._with_phone(lambda: phone.stop_video(out)),
                                    self._acquired(trial, key, video=True))
                    else:
                        self._emit({"type": "recording", "key": key, "trial": trial})
                self.cur["recording"] = key
                ok = self._spawn("phone", self._with_phone(phone.start_video), started)
                if not ok:
                    self.cur["recording"] = ""
                return {"ok": ok} if ok else {"error": "busy"}
            ok = self._spawn("phone", self._with_phone(lambda: phone.stop_video(out)),
                             self._acquired(trial, key, video=True))
            return {"ok": ok} if ok else {"error": "busy"}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def step_import(self, key: str) -> str:
        """A file chosen by hand for the step `key`: copied into staging by a worker (a scan may be large, or on a
        share), then staged like a photo taken ('staged' or 'acquire_failed' events say when)."""
        def go() -> dict[str, Any]:
            run, row, steps, i = self._current(key)
            kind = trials.file_kind(steps[i].files[0])
            pattern = " ".join("*" + s for s in trials.SUFFIXES[kind])
            path, _ = QFileDialog.getOpenFileName(None, "ImpactGuide", str(self.scanner_folder() or Path.home()),
                                                  f"{kind} ({pattern})")
            if not path:
                return {"ok": False}
            src, tmp = Path(path), self.roots.staging / row["trial"] / ("import_" + Path(path).name)

            def work() -> Path:
                tmp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tmp)
                return tmp
            if not self._spawn("import_file", work, self._acquired(row["trial"], key)):
                return {"error": "busy"}
            return self._trial_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def step_discard(self, key: str) -> str:
        def go() -> dict[str, Any]:
            item = (self.cur.get("staged") or {}).pop(key, None)
            if item:
                trials.Run.discard(item.get("prepared"))
            if item and item.get("temp"):
                self._unlink(Path(item["path"]))
            return self._trial_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def scan_watch(self, key: str) -> str:
        """Waits for the next scan exported into the scanner folder; it is staged for this step."""
        def go() -> dict[str, Any]:
            run, row, steps, i = self._current(key)
            folder = self.scanner_folder()
            if folder is None:
                return {"error": "no_scanner_folder"}
            folder.mkdir(parents=True, exist_ok=True)
            if self._watch is None or self._watch.key != key:
                self._watch = ScanWatch(row["trial"], key, folder)
                if self.roots.simulate:
                    stem = Path(steps[i].files[0]).stem
                    export = folder / f"export_{stem}.ply"
                    if self.scans.synthetic():      # a synthetic scan of the bench as the trials left it
                        self.scans.sim_export(row["trial"], key, folder, export.name)
                    else:
                        QTimer.singleShot(FAKE_SCAN_MS,
                                          guarded(lambda: acquisition.fake_scan(export.parent, export.name)))
            return {"ok": True, "folder": str(folder)}
        return self._answer(go)

    @pyqtSlot(result=str)
    def choose_scanner_folder(self):
        def go():
            start = self.prefs["scanner_folder"] or str(Path.home())
            folder = QFileDialog.getExistingDirectory(None, "ImpactGuide", start)
            if folder:
                self.prefs.set("scanner_folder", folder)
                self._watch = None
            return self._trial_state()
        return self._answer(go)

    @pyqtSlot(str, str, result=str)
    def trial_void(self, trial, reason):
        def go():
            if not reason.strip():
                return {"error": "reason"}
            run = self.run()
            run.void(trial, reason.strip())
            if self.cur.get("trial") == trial:
                self._drop_staged()
                self.cur = {}
            return self._trial_state()
        return self._answer(go)

    # ── trials: log, sequence, configuration, completeness ──────────────
    @pyqtSlot(result=str)
    def log_rows(self):
        def go():
            run = self.run()
            cols = [dict(zip(("name", "unit", "filled", "description"), c)) for c in trials.LOG_COLUMNS]
            return {"phase": run.phase, "rows": run.log.rows(), "edits": run.log.edits(), "columns": cols,
                    "folder": str(run.folder)}
        return self._answer(go)

    @pyqtSlot(str, str, str, str, result=str)
    def log_edit(self, trial, field, value, reason):
        def go():
            run = self.run()
            if field == "status" and value == trials.VOID:
                run.void(trial, reason)
            else:
                run.log.edit(trial, field, value, reason)
            return json.loads(self.log_rows())
        return self._answer(go)

    @pyqtSlot(result=str)
    def sequence_rows(self):
        def go():
            run = self.run()
            rows = run.log.rows()
            filled, extra = trials.match(run.sequence, rows)
            plan = None if self.cur.get("trial") else self._plan(run)
            return {"phase": run.phase, "rows": [{**s, "trial": filled.get(int(s["slot"]), "")} for s in run.sequence],
                    "extra": extra, "next": plan["slot"] if plan else "", "totals": self._totals(run)}
        return self._answer(go)

    @pyqtSlot(result=str)
    def config_get(self):
        def go():
            run = self.run()
            doc = json.loads(run.config_path.read_text(encoding="utf-8")) if run.config_path.exists() else {}
            return {"phase": run.phase, "config": json.loads(json.dumps(asdict(run.config))),
                    "written": doc.get("written", ""), "history": doc.get("history", []),
                    "totals": self._totals(run), "default": json.loads(json.dumps(asdict(trials.default_config(run.phase))))}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def config_apply(self, payload):
        def go():
            run = self.run()
            if run.phase == trials.PRE_IMPACT:          # the sets of positions: only how many
                try:
                    n = int(json.loads(payload).get("repetitions"))
                except (TypeError, ValueError):
                    return {"error": "bad_value"}
                if not 1 <= n <= 20:
                    return {"error": "bad_value"}
                before = len(run.sequence)
                run.configure(trials.replace(run.config, repetitions=n))
                return {**json.loads(self.config_get()), "before": before}
            d = {**asdict(run.config), **json.loads(payload)}
            d["phase"] = run.phase
            d["holders"] = [str(h).strip() for h in d["holders"] if str(h).strip()]
            for k in ("angles_deg", "directions_deg"):
                d[k] = sorted({int(round(float(v))) for v in d[k]})
            for k in ("repetitions", "drape_trials", "direction_run", "seed", "washers_per_side"):
                d[k] = int(d[k])
            d["reposition_mm"] = float(d["reposition_mm"])
            d["highspeed"] = bool(d["highspeed"])
            if (not d["holders"] or not d["angles_deg"] or not d["directions_deg"] or d["repetitions"] < 1
                    or d["drape_trials"] < 0 or d["direction_run"] < 1 or d["order"] not in ("random", "ascending")
                    or d["washers_per_side"] < 0 or d["reposition_mm"] <= 0):
                return {"error": "bad_value"}
            before = len(run.sequence)
            run.configure(trials.config_from_dict(d))
            return {**json.loads(self.config_get()), "before": before}
        return self._answer(go)

    @pyqtSlot(bool, result=str)
    def check_phase(self, deep):
        def go():
            report = self.run().check(deep)
            return {"trials": [{**t, "problems": [list(p) for p in t["problems"]]} for t in report["trials"]],
                    "pending": len(report["pending"]), "extra": report["extra"], "orphans": report["orphans"],
                    "duplicates": report["duplicates"], "open": report["open"]}
        return self._answer(go)

    # ── calibration ──────────────────────────────────────────────────────
    # Three steps (ui/10-calibration.js): the board (on this monitor, in the page; or a physical one: the 3D-printed
    # plate or a printed sheet, with its measure), the photos (a plan of poses, quick or accurate, each photo checked
    # as it comes and kept or put aside), the calibration (its errors per photo and over the board, the intrinsics in
    # plain units, a verdict, photos left out by hand, saved). Three camera modes, each calibrated on its own images
    # and never used for another: the phone's photos (the camera app, 12.5 MP, what the measurements of the bench
    # use), its video (the camera app's films, the trial videos: frames chosen from a film) and the live stream (its
    # frames, for the live AR).
    def _intrinsics_path(self, mode):
        return self.roots.calib_out / f"intrinsics_{mode}.json"

    def _checks_path(self):
        return self.roots.calib_out / "session_checks.csv"

    @staticmethod
    def _cal_summary(cal):
        if not cal:
            return None
        K = cal["K"]
        return {"date": cal.get("date", ""), "rms_px": cal["rms_px"], "n_views": cal["n_views"],
                "n_corners": cal.get("n_corners", 0), "image_size": cal["image_size"], "images": cal.get("images", ""),
                "fx": K[0][0], "fy": K[1][1], "cx": K[0][2], "cy": K[1][2], "dropped": cal.get("dropped", []),
                "skipped": cal.get("skipped", {}), "square_mm": (cal.get("board") or {}).get("square_mm")}

    def _mode(self):
        m = self.prefs["calib_mode"]
        return m if m in CALIB_MODES else "photo"

    def _calib_plan(self):
        """The plan of poses of the photos now: the session check has its own, else quick or accurate."""
        if self._series_kind() == "check":
            return "check"
        p = self.prefs["calib_plan"]
        return p if p in ("quick", "accurate", "bench") else "quick"

    def _folders(self):
        """The folders of calibration photos, newest first: name, path, kind, board.json, "files" (the photos in it)
        and "n" (those that count, _counted)."""
        folders = []
        if self.roots.calib_raw.is_dir():
            for d in sorted((d for d in self.roots.calib_raw.iterdir() if d.is_dir()), key=lambda d: d.name,
                            reverse=True):
                photos = [p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
                board = None
                with contextlib.suppress(OSError, ValueError):
                    board = json.loads((d / calibrate_charuco.BOARD_FILE).read_text(encoding="utf-8"))
                folders.append({"name": d.name, "path": str(d), "files": len(photos),
                                "n": len(self._counted(d, photos)), "kind": d.name.split("_", 1)[-1], "board": board})
        return folders

    def _counted(self, d: Path, photos: list[Path]) -> list[Path]:
        """The photos of a folder that count (for the plan and the calibration): those whose check kept them. In a
        folder with no checks at all (photos from before the checks were kept, or put there by hand) every photo
        counts, the calibration's own detection judging them."""
        raw = read_json(Path(d) / PHOTO_CHECKS, "photo checks")
        if not isinstance(raw, dict):
            # the frames of a film count once chosen (their checks written with them), never while they are written
            return [] if (Path(d) / calib_video.FILM_DIR).is_dir() else list(photos)
        return [p for p in photos if (raw.get(p.name) or {}).get("ok")]

    def _unchecked(self, d: Path, checks: dict[str, dict[str, Any]]) -> list[Path]:
        """The photos of a series with checks that have none of their own and none on its way (taken, copied, waiting
        for or under their check): put there by hand, or left when the app stopped before their check. Oldest
        first."""
        busy = {x["name"] for x in self._shots_in(d) if x["name"]} | {p.name for f, p in self._check_queue
                                                                       if str(f) == str(d)}
        busy |= {name for f, name in self._importing if str(f) == str(d)}     # still being copied in
        if self._check_now is not None and str(self._check_now[0]) == str(d):
            busy.add(self._check_now[1].name)
        try:
            photos = sorted((p for p in Path(d).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                             and p.name not in checks and p.name not in busy), key=lambda p: p.stat().st_mtime)
        except OSError:
            return []
        return photos

    @staticmethod
    def _photo_checks(d: Path | str) -> dict[str, dict[str, Any]]:
        """{name: check} of the photos of a series (calibrate_charuco.check_photo), kept beside them."""
        checks = read_json(Path(d) / PHOTO_CHECKS, "photo checks")
        return checks if isinstance(checks, dict) else {}

    @staticmethod
    def _write_checks(d: Path, checks: dict[str, dict[str, Any]]) -> None:
        """The photo checks of a series written back, atomically (a crash never leaves every photo unchecked)."""
        write_json_atomic(d / PHOTO_CHECKS, checks)

    def _series_detail(self, f, active):
        """The series of photos in progress: its photos in the order they came, each with its check; the corners of
        the photos kept (where the board has been on the screen); whether it is of the board in use."""
        d = Path(f["path"])
        checks = self._photo_checks(d)
        photos = sorted((p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.stat().st_mtime)
        same = f["files"] == 0 or self._same_board(f["board"], active)
        rows = [{"name": p.name, **{k: (checks.get(p.name) or {}).get(k) for k in ("ok", "why", "corners", "of",
                                                                                  "blur_px")}} for p in photos]
        points = []
        for p in photos:
            pts = (checks.get(p.name) or {}).get("points") or []
            points.append(pts[::max(1, len(pts) // 30)])
        last = None
        if checks:
            name = max(checks, key=lambda k: (checks[k].get("seq", 0), checks[k].get("time", "")))
            # the picture the page shows it by: its small one, else the photo where it is now (put aside when it
            # failed its check)
            at = next((p for p in (self._thumb_path(d, name), d / name, d / REJECTED / name) if p.is_file()), None)
            last = {"name": name, "url": file_url(at) if at else "",
                    **{k: checks[name].get(k) for k in ("ok", "why", "corners", "of", "blur_px", "points", "time")},
                    "problems": list((checks[name].get("settings") or {}).get("problems") or [])}
        return {**f, "same": same, "photos": rows, "points": points, "last": last,
                "missing_corners": self._missing_corners(photos, checks)}

    @staticmethod
    def _missing_corners(photos: list[Path], checks: dict[str, dict[str, Any]]) -> list[str]:
        """The corners of the photo (calibrate_charuco.CORNER_NAMES) the board has not reached yet in the photos kept
        of a series (of the size of the first one): all four before the first."""
        import numpy as np
        kept = [checks[p.name] for p in photos if (checks.get(p.name) or {}).get("ok")]
        kept = [c for c in kept if c.get("points") and c.get("size")]
        if not kept:
            return list(calibrate_charuco.CORNER_NAMES)
        size = kept[0]["size"]
        kept = [c for c in kept if c["size"] == size]
        reached = set()
        for c in kept:      # each check's own coverage, of all its corners; else its points (a few of them)
            cov = c.get("coverage") or calibrate_charuco.coverage(
                [np.asarray(c["points"], np.float64).reshape(-1, 2) * size], tuple(size))
            reached |= {k for k, v in (cov.get("corners") or {}).items() if v}
        return [k for k in calibrate_charuco.CORNER_NAMES if k not in reached]

    # ── the photos on their way: the pose each is for, the next pose, the strip of the Photos step ──
    def _shots_in(self, d: Path | str) -> list[dict[str, Any]]:
        """The photos of the folder d on their way (taken, copied or checked now)."""
        d = str(Path(d))
        return [x for x in self._shots if str(x["path"]) == d]

    def _poses_taken(self, d: Path, names: list[str], checks: dict[str, dict[str, Any]]) -> set[int]:
        """The poses of the plan the series has a photo for: kept, being checked, or taken and on its way. A photo
        with no pose of its own (imported, or of a version that did not keep it) stands for the first free pose."""
        shots = self._shots_in(d)
        by_name = {x["name"]: x for x in shots if x["name"]}
        used, loose = set(), 0
        for name in names:
            c = checks.get(name)
            if c is not None and not c.get("ok"):
                continue            # failed and not put aside (the move failed): its pose comes again
            pose = c.get("pose") if c is not None else (by_name.get(name) or {}).get("pose")
            if isinstance(pose, int) and pose >= 0 and pose not in used:
                used.add(pose)
            else:
                loose += 1
        used |= {x["pose"] for x in shots if x["state"] in ("shot", "copying") and isinstance(x["pose"], int)}
        i = 0
        while loose > 0:
            if i not in used:
                used.add(i)
                loose -= 1
            i += 1
        return used

    def _thumb_path(self, d: Path, name: str) -> Path:
        """The small picture of a photo of a series (made as it is copied, or at its check), in the app's cache."""
        return self.roots.local / "cache" / "thumbs" / Path(d).name / (Path(name).stem + ".jpg")

    @staticmethod
    def _make_thumb(src: Path, out: Path) -> None:
        """The small picture of a photo, THUMB_W wide (decoded at an eighth of its size: quick); never fails the
        caller (a picture that cannot be made is only missing)."""
        import cv2
        try:
            img = cv2.imread(str(src), cv2.IMREAD_REDUCED_GRAYSCALE_8)
            if img is None:
                return
            h = max(1, round(img.shape[0] * THUMB_W / img.shape[1]))
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), cv2.resize(img, (THUMB_W, h), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        except (OSError, ValueError, cv2.error):
            log.warning("no small picture of %s", src, exc_info=True)

    def _strip(self, d: Path, checks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """The newest STRIP_N photos of the series in the order they were taken, each with its pose, its small
        picture and how its check went: kept, retake (with why), or checking (still on its way)."""
        out = []
        for name, c in sorted(checks.items(), key=lambda kv: (kv[1].get("seq", 0), kv[1].get("time", ""))):
            t = self._thumb_path(d, name)
            out.append({"name": name, "pose": c.get("pose"), "state": "kept" if c.get("ok") else "retake",
                        "why": c.get("why") or "", "thumb": file_url(t) if t.is_file() else ""})
        shots = self._shots_in(d)
        on_way = {x["name"] for x in shots if x["name"]}
        with contextlib.suppress(OSError):
            for p in sorted((p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.name not in checks
                             and p.name not in on_way), key=lambda p: p.stat().st_mtime):
                out.append({"name": p.name, "pose": None, "state": "checking", "why": "", "thumb": ""})
        for x in shots:
            if x["state"] != "shooting":
                t = self._thumb_path(d, x["name"]) if x["name"] else None
                out.append({"name": x["name"], "pose": x["pose"], "state": "checking", "why": "",
                            "thumb": file_url(t) if t is not None and t.is_file() else ""})
        return out[-STRIP_N:]

    def _guide_board(self, series, active):
        """The board the guide draws: the series' own (board.json) when it has photos, else the one in use."""
        b = (series or {}).get("board") if series and series.get("files") else None
        return b or active

    def _calib_state(self):
        r = self.roots
        modes = {}
        for mode in CALIB_MODES:
            p = self._intrinsics_path(mode)
            cal = None
            if p.exists():
                with contextlib.suppress(OSError, ValueError):
                    cal = json.loads(p.read_text(encoding="utf-8"))
            modes[mode] = {"stored": self._cal_summary(cal), "path": str(p),
                           "pending": self._cal_summary(self._pending_cal.get(mode))}
        checks = trials.read_csv(self._checks_path())
        last = {}
        for c in checks:
            last[c["mode"]] = c
        bs = self._board_state()
        active = bs["active"]
        kind, mode, plan = self._series_kind(), self._mode(), self._calib_plan()
        lim = calib_guide.limits(plan)
        f = self._series()
        series = self._series_detail(f, active) if f is not None else None
        n = series["n"] if series and series["same"] else 0
        # the photos on their way count at once, each for its pose: the guide shows the next free pose as soon as the
        # shutter went, their checks come later (a photo put aside frees its pose again)
        on_way = 0
        taken: set[int] = set()
        if series and series["same"]:
            d = Path(series["path"])
            checks_d = self._photo_checks(d)
            if checks_d:        # a photo of a checked series without a check of its own counts once checked
                orphans = self._unchecked(d, checks_d)
                if orphans:
                    self._queue_checks(d, orphans)
            # a photo counts once kept (_counted), and meanwhile from the shutter on, for its pose
            on_way = sum(1 for x in self._shots_in(d) if x["state"] in ("shot", "copying", "checking"))
            taken = self._poses_taken(d, [p["name"] for p in series["photos"]], checks_d)
            series["shots"] = self._strip(d, checks_d)
        elif series:
            series["shots"] = []
        n += on_way
        pose = next((i for i in range(len(calib_guide.PLANS[plan])) if i not in taken), None)
        # every pose of the plan has its photo but the board has not reached a corner of the photo yet (once the photos
        # on their way are checked): that corner's pose once more, up to EXTRA_MAX photos past the plan
        extra = None
        if (pose is None and kind == "calibration" and series and series["same"] and not self._on_way()
                and series.get("missing_corners") and n < lim["max"] + EXTRA_MAX):
            extra = series["missing_corners"][0]
            pose = next((i for i, p in enumerate(calib_guide.PLANS[plan]) if p.get("corner") == extra), None)
        nxt = None
        if active and (n < lim["max"] or extra) and pose is not None and mode != "video":   # a film: every pose
            g = self._guide_board(series if series and series["same"] else None, active)
            with contextlib.suppress(KeyError, TypeError, ValueError):
                nxt = calib_guide.guide(calib_guide.PLANS[plan][pose], g, mode, pose, plan)
                # what is moved for it: a physical board in the hand, the phone still on its stand; the phone, before
                # the board on this monitor
                nxt["move"] = "board" if bs.get("kind") in ("plate", "sheet") else "phone"
                nxt["extra"] = extra
        try:                                # the photo taken by itself never costs the page its state
            self._auto_target(nxt, active, series, mode)
        except Exception:
            log.warning("the photo taken by itself not told the next pose", exc_info=True)
        result = self._result(mode)
        # the last calibration of this mode found that the phone moved between its photos: the Photos step says to
        # keep it still on a stand where it checks for a phone held still
        moved = bool(result and "phone_not_still" in result["not_good"])
        live = None
        if self._live is not None:
            st = self._live.status()
            live = {"running": st["running"], "starting": st.get("starting"), "tracked": st.get("tracked"),
                    "url": self._live_server.url if self._live_server else "", "purpose": self._live_for,
                    "display": st.get("display") or st.get("image"), "error": bool(st.get("error")),
                    "fps_ar": st.get("fps_ar", 0.0), "stream_state": st.get("stream_state", "")}
        return {"mode": mode, "plan": plan, "kind": kind, "limits": lim, "count": n, "series": series, "next": nxt,
                "plans": {k: calib_guide.limits(k) for k in ("quick", "accurate", "bench")},
                "modes": modes, "last_check": last, "result": result, "board_ui": bs, "phone": self.phone_status,
                "busy": sorted(self._busy), "min_views": calibrate_charuco.MIN_VIEWS,
                "max_rms": calibrate_charuco.MAX_RMS, "live": live, "simulate": bool(r.simulate),
                "raw": str(r.calib_raw), "out": str(r.calib_out), "film": self._film_state(series),
                "pending": len(self._shots), "calib_waiting": self._calib_waiting is not None,
                "auto": {"on": bool(self.prefs["calib_auto"]), **self._auto_view}, "moved": moved}

    @pyqtSlot(result=str)
    def calib_state(self):
        return self._answer(self._calib_state)

    def _result(self, mode: str) -> dict[str, Any] | None:
        """The calibration waiting to be saved (or the one stored) of a mode, as the result step draws it: the
        verdict, the error of each photo (used, dropped by the 3-times-the-median rule, left out by hand), where on
        the board the error is, the intrinsics in plain units. The verdict is the one of the plan the photos were
        taken with (stored with the calibration), whatever plan is chosen now; a focal length known to worse than
        F_STD_REL_MAX is never good, whatever its error, and a calibration with one of
        calibrate_charuco.ISSUES_NOT_GOOD (a board not flat, the phone moved, automatic settings) is not good enough:
        its photos must be taken again (what is wrong in `issues`, _issues)."""
        pending = self._pending_cal.get(mode)
        cal = pending
        if cal is None:
            p = self._intrinsics_path(mode)
            with contextlib.suppress(OSError, ValueError):
                cal = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        if not cal:
            return None
        used = set(cal.get("per_view_rms_px", {}))
        dropped, excluded = set(cal.get("dropped", [])), set(cal.get("excluded", []))
        every = cal.get("per_view_all_px") or cal.get("per_view_rms_px", {})
        photos = [{"name": k, "err": v, "state": "used" if k in used else "excluded" if k in excluded
                   else "dropped" if k in dropped else "used"} for k, v in sorted(every.items())]
        rms = cal["rms_px"]
        plan = cal.get("plan") if cal.get("plan") in ("quick", "accurate", "bench") else self._calib_plan()
        few = cal["n_views"] < calib_guide.limits(plan if plan != "check" else "quick")["min"]
        plain = calibrate_charuco.plain(cal)
        f_std = plain.get("f_std")                # None when the calibration does not know it (std NaN or missing)
        focal_rel = f_std / plain["fx"] if f_std is not None and plain["fx"] else math.inf
        focal_ok = math.isfinite(focal_rel) and focal_rel <= F_STD_REL_MAX
        # a film or the stream seeing a narrower field than the photos: cropped (stabilisation or a zoom on), so its
        # frames were not all the same camera; never good
        narrower = None
        ref = self._intrinsics_path("photo")
        if mode != "photo" and ref.exists():
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
                narrower = calib_video.narrower_than(cal, json.loads(ref.read_text(encoding="utf-8")))
        cropped = narrower is not None and narrower > NARROWER_MAX
        issues = self._issues(cal)
        verdict = "poor" if issues["not_good"] else \
            "good" if rms < calibrate_charuco.MAX_RMS and not few and focal_ok and not cropped else \
            "fair" if rms < 2 * calibrate_charuco.MAX_RMS else "poor"
        b = cal.get("board") or {}
        res = cal.get("residuals")
        grid = [int(b["squares"][0]) - 1, int(b["squares"][1]) - 1] if b.get("squares") else None
        return {"saved": pending is None, "date": cal.get("date", ""), "rms_px": rms, "n_views": cal["n_views"],
                "cropped": cropped, "narrower": narrower, "film": cal.get("film"),
                "of": len(every), "few": few, "focal_ok": focal_ok, "verdict": verdict, "photos": photos,
                "limit_px": cal.get("limit_px"), "max_rms": calibrate_charuco.MAX_RMS,
                "residuals": res, "grid": grid, "plain": plain, "images": cal.get("images", ""), **issues}

    @staticmethod
    def _issues(cal: dict[str, Any]) -> dict[str, Any]:
        """What spoils a calibration (calibrate_charuco.ISSUES, in its "issues"), the most serious first (ISSUE_ORDER),
        with what the page says of each: how much the board bends (sag_mm), how far the optical centre moved from photo
        to photo (centre_px: the phone moved), the corners of the photo the board never reached, the photos taken
        with automatic settings and the problems found in them (calibrate_charuco.SETTINGS); not_good: the issues
        that keep it from being good (calibrate_charuco.ISSUES_NOT_GOOD). A calibration made before these were
        known has none."""
        have = set(cal.get("issues") or [])
        issues = [i for i in ISSUE_ORDER if i in have]
        sh, ce, co = cal.get("shape") or {}, cal.get("centre") or {}, cal.get("coverage") or {}
        settings = cal.get("settings") or {}
        codes = sorted({c for v in settings.values() for c in v if c in calibrate_charuco.SETTINGS},
                       key=calibrate_charuco.SETTINGS.index)
        return {"issues": issues, "not_good": [i for i in issues if i in _not_good(cal)],
                "flat": sh.get("flat", True), "sag_mm": sh.get("sag_mm"), "focal_shift": sh.get("focal_shift"),
                "rms_flat_px": cal.get("rms_flat_px", sh.get("rms_flat_px")),
                "still": ce.get("still", True), "centre_px": ce.get("spread_px"), "centre_max_px": ce.get("max_px"),
                "missing_corners": [c for c in calibrate_charuco.CORNER_NAMES if c in (co.get("missing_corners") or [])],
                "reach": co.get("reach"), "settings_n": len(settings), "settings": codes}

    @staticmethod
    def _good(cal: dict[str, Any]) -> bool:
        """A calibration worked out well enough to say so: its error under the limit and none of the issues that
        keep it from being good."""
        return cal["rms_px"] < calibrate_charuco.MAX_RMS and not set(cal.get("issues") or []) & set(_not_good(cal))

    # ── calibration: the series of photos ───────────────────────────────
    def _series_kind(self):
        """What the photos are for: a calibration, or the session check of the one stored (Advanced)."""
        k = self.prefs["calib_kind"]
        return k if k in ("calibration", "check") else "calibration"

    def _series_tag(self):
        return self._mode() if self._series_kind() == "calibration" else "check_" + self._mode()

    def _series(self, folders: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """The folder the photos go to now: the newest of this mode and kind when it is today's, or when its last
        photo is younger than SERIES_KEEP_S (a series under way at midnight goes on); else None (one is made at the
        first photo). An older folder is a series of its own, kept as it is."""
        if folders is None:
            folders = self._folders()
        today = f"{date.today():%Y%m%d}"
        newest = next((f for f in folders if f["kind"] == self._series_tag()), None)
        if newest is None or newest["name"].startswith(today):
            return newest
        try:
            last = max((p.stat().st_mtime for p in Path(newest["path"]).iterdir()
                        if p.suffix.lower() in IMAGE_SUFFIXES), default=None)
        except OSError:
            last = None
        return newest if last is not None and time.time() - last < SERIES_KEEP_S else None

    @staticmethod
    def _same_board(folder_board, active):
        """The photos of a folder are of the board in use now: the same squares, square and marker."""
        if not folder_board or not active:
            return False
        return (list(folder_board.get("squares") or []) == list(active.get("squares") or [])
                and abs(float(folder_board.get("square_mm", 0)) - float(active["square_mm"])) < 1e-6
                and abs(float(folder_board.get("marker_mm", 0)) - float(active["marker_mm"])) < 1e-6)

    def _new_series(self):
        """A new folder for today's photos of this mode and kind: 20261016_photo, then 20261016b_photo..."""
        today, tag = f"{date.today():%Y%m%d}", self._series_tag()
        for suffix in [""] + list("bcdefghijklmnopqrstuvwxyz"):
            d = self.roots.calib_raw / f"{today}{suffix}_{tag}"
            if not d.exists():
                d.mkdir(parents=True)
                return d
        raise RuntimeError("too many series today")

    def _series_folder(self):
        """The folder for the next photo: the series of today if it has no photos yet or its photos are of the board
        in use; else a new one (the board changed: the photos start again)."""
        f = self._series()
        if f is not None and (f["files"] == 0 or self._same_board(f["board"], self._board()["active"])):
            return Path(f["path"])
        return self._new_series()

    def _series_hold(self):
        """The board of the series in progress, kept on the monitor while it fits (screen_board.layout, hold)."""
        f = self._series()
        b = f["board"] if f is not None and f["files"] > 0 else None
        return b if b and b.get("kind") == "screen" and b.get("square_px") else None

    @pyqtSlot(str, result=str)
    def calib_set(self, payload):
        """{mode: photo or stream, plan: quick or accurate, kind: calibration or check, auto: the photo taken by
        itself on or off}."""
        def go():
            c = json.loads(payload)
            if c.get("mode") in CALIB_MODES:
                self.prefs.set("calib_mode", c["mode"])
            if c.get("plan") in ("quick", "accurate", "bench"):
                self.prefs.set("calib_plan", c["plan"])
            if c.get("kind") in ("calibration", "check"):
                self.prefs.set("calib_kind", c["kind"])
            if isinstance(c.get("auto"), bool):
                self.prefs.set("calib_auto", c["auto"])
            self._auto_sync()
            return self._calib_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_series_new(self):
        """Start again: today's photos go to a new folder from the next one on."""
        def go():
            d = self._new_series()
            self._folder_board(d)
            return {**self._calib_state(), "folder": d.name}
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_take(self):
        """The next photo of the plan into the series: a photo from the phone's camera app, or the newest frame of the
        live stream, at once (never a countdown). The board must be on show when it is the one on this monitor, and
        the plan not full. A Take while a photo is being taken is taken right after it. The guide moves to the next
        pose as soon as the shutter went (calib_photo action shot); each photo is copied and checked in the
        background (calib_photo with its check) while the next ones are taken."""
        return self._answer(lambda: self._take(False))

    def _take(self, auto: bool) -> dict[str, Any]:
        """calib_take, or the photo taken by itself (auto)."""
        def go():
            if self._mode() == "video" and self._film is not None:
                return self._film_stop()            # a film: the same button stops it
            st = self._board()
            if st["kind"] == "screen" and (not self._board_on or self._lay is None):
                return {"error": "board_small" if self._board_on else "board_hidden"}
            if self._mode() == "video":
                return self._film_start()
            if "phone" in self._busy:           # a photo is being taken: this one follows it at once
                self._take_wanted = True
                return {"ok": True, "queued": True}
            s = self._calib_state()
            if s["count"] >= s["limits"]["max"] and not (s["next"] and s["next"].get("extra")):
                return {"error": "calib_full"}
            if s["next"] is None:               # no board to take, or every pose has its photo
                return {"error": "calib_full" if s["board_ui"]["active"] else "board_hidden"}
            if self._mode() == "stream" and not (self._live is not None and self._live.running
                                                 and self._live_for == "calibration"):
                return {"error": "stream_off"}
            d = self._series_folder()
            pose = s["next"]["n"] if s["series"] is None or Path(s["series"]["path"]) == d else 0
            if self._mode() == "stream":
                return self._grab_into(d, pose, auto)
            return self._shoot_into(d, pose=pose, auto=auto)
        return go()

    def _take_queued(self) -> None:
        """A Take that came while the last photo was being taken: taken now (said when it cannot be)."""
        if not self._take_wanted:
            return
        self._take_wanted = False
        try:
            r = self._take(False)
        except (Exception, SystemExit) as e:        # said, never out of a done()
            r = self._err(e)
        if r.get("error"):
            self._emit({"type": "acquire_failed", "key": "calibration", "error": r["error"],
                        "args": r.get("args", {}), "message": r.get("message", "")})

    def _pressed(self, shot: dict[str, Any]) -> None:
        """The shutter went for this photo (on the GUI thread): its pose counts from now, the page says so at once
        (a flash and a sound) and shows the guide of the next pose."""
        if shot not in self._shots:
            return
        shot["state"] = "shot"
        shot.setdefault("t_press", time.monotonic())
        if self._auto is not None:
            self._auto.shot()
            self._calib_state()             # the watcher on the next pose at once
        self._emit({"type": "calib_photo", "action": "shot", "folder": shot["folder"], "pose": shot["pose"],
                    "auto": shot["auto"]})

    def _drop_shot(self, shot: dict[str, Any]) -> None:
        if shot in self._shots:
            self._shots.remove(shot)

    def _on_way(self) -> bool:
        """Photos are being taken, copied or checked."""
        return self._checking() or bool(self._shots) or bool(self._copy_queue)

    def _maybe_calibrate(self) -> None:
        """A Calibrate that came while photos were on their way: started once the last one is checked."""
        if self._calib_waiting is None or self._on_way():
            return
        leave, self._calib_waiting = self._calib_waiting, None
        r = json.loads(self.calib_run(json.dumps(leave)))
        if r.get("error"):
            self._emit({"type": "calib", "action": "calibrate", "mode": self._mode(), "ok": False, **r})

    # ── calibration: the photo taken by itself (auto_shot.py) ────────────
    # While the Photos step is open and the option is on, a watcher looks at the preview: the frames of the live
    # stream in the streaming mode, the phone's own screen mirrored in the photo mode (the camera app has the camera,
    # its preview on the screen is the photo to come), the simulated phone's view in simulation. When the board is
    # where the guide wants it and the phone holds still a moment, at least auto_shot.GAP_S after the last photo, the
    # photo is taken as by Take.
    @pyqtSlot(bool, result=str)
    def calib_watch(self, on: bool) -> str:
        """The page has the Photos step open (on) or left it."""
        def go() -> dict[str, Any]:
            self._watching = bool(on)
            self._auto_sync()
            return {"ok": True, "auto": {"on": bool(self.prefs["calib_auto"]), **self._auto_view}}
        return self._answer(go)

    def _auto_wanted(self) -> bool:
        return self._watching and bool(self.prefs["calib_auto"]) and self._mode() in ("photo", "stream")

    def _auto_sync(self) -> None:
        """The watcher running while it is wanted, stopped otherwise; the mirrored screen only with it, in the photo
        mode."""
        want = self._auto_wanted()
        if want and self._auto is None:
            self._auto_mode, self._auto_snapshot = self._mode(), None
            self._auto = auto_shot.Watcher(self._auto_frame, self._auto_due, self._auto_said)
            self._auto.start()
            self._calib_state()             # its first guide
        elif not want and self._auto is not None:
            w, self._auto = self._auto, None
            w.stop(wait=False)              # it ends at its next look; a mirror it opens late is closed by itself
            self._auto_view = {"state": "off", "score": 0.0}
        if not want or self._mode() != "photo":
            self._preview_close()

    def _preview_close(self) -> None:
        p, self._preview = self._preview, None
        if p is not None:
            try:
                p.close()
            except Exception:
                log.warning("the phone's mirrored screen not closed", exc_info=True)

    def _auto_target(self, nxt: dict[str, Any] | None, active: dict[str, Any] | None, series: dict[str, Any] | None,
                     mode: str) -> None:
        """The watcher told the guide of the next photo and its board (nothing wanted when there is none)."""
        w = self._auto
        if w is None:
            return
        self._auto_mode = mode
        spec = self._guide_board(series if series and series["same"] else None, active) if active and nxt else None
        key = ((series or {}).get("name", ""), nxt["n"]) if nxt else None
        w.target(nxt["outline"] if spec else None, spec, key, "stream" if mode == "stream" else "photo",
                 new_stop=bool(nxt and nxt.get("view") == "bench"))
        self._auto_snapshot = {"pose": nxt["n"], "plan": nxt["plan"], "spec": spec, "kind": spec.get("kind") or "screen",
                               "lay": self._lay} if spec else None

    def _auto_frame(self) -> tuple[Any, int, str] | None:
        """The newest frame for the watcher (on its thread): (image, frame number, kind) or None."""
        mode = self._auto_mode
        if mode == "stream":
            ar = self._live
            if ar is None or not ar.running or self._live_for != "calibration":
                return None
            img, n = ar.still()
            return (img, n, "photo") if img is not None else None
        if self.roots.simulate:
            return self._sim_preview()
        p = self._preview
        if p is not None and p.alive:
            img, _, n = p.latest()
            return (img, n, "display") if img is not None else None
        if time.monotonic() - self._preview_try < PREVIEW_RETRY_S or (self._live is not None and self._live.running):
            return None
        self._preview_try = time.monotonic()
        if p is not None:
            self._preview = None
            with contextlib.suppress(Exception):
                p.close()
        try:
            p = auto_shot.preview_stream()
            p.start(timeout=8.0)
        except Exception as e:
            log.info("the phone's screen not mirrored for the photo taken by itself: %s", e)
            return None
        if self._auto is None or self._auto_mode != "photo":    # stopped meanwhile
            with contextlib.suppress(Exception):
                p.close()
            return None
        self._preview = p
        return None

    def _sim_preview(self) -> tuple[Any, int, str] | None:
        """In simulation the phone's view is the board at the guide's pose, as an operator who follows the guide holds
        it (calib_guide.photo, small): drawn once per pose."""
        snap = self._auto_snapshot
        if snap is None:
            return None
        key = (snap["plan"], snap["pose"], json.dumps(snap["spec"], sort_keys=True, default=str))
        cached = getattr(self, "_sim_view", None)
        if cached is None or cached[0] != key:
            tex = {"screen": "screen", "plate": "plate", "sheet": "sheet"}.get(snap["kind"], "screen")
            img = calib_guide.photo(snap["plan"], snap["pose"], snap["spec"], tex, "photo",
                                    snap["lay"] if tex == "screen" else None, size=(1024, 768))
            cached = self._sim_view = (key, img, (cached[2] + 1) if cached else 1)
        return cached[1], cached[2], "photo"

    def _auto_due(self) -> None:
        """A photo is due (on the watcher's thread): taken on the GUI thread."""
        self._jobs.put(("", lambda _r, _e: self._auto_fire(), None, None))

    def _auto_fire(self) -> None:
        w = self._auto
        if w is None:
            return
        if not self._auto_wanted() or "phone" in self._busy:
            w.refused()
            return
        try:
            r = self._take(True)
        except (Exception, SystemExit) as e:        # said in the log, tried again later
            r = self._err(e)
        if r.get("error") or r.get("queued"):
            w.refused()

    def _auto_said(self, state: str, score: float) -> None:
        """The watcher's state changed (on its thread): kept for calib_state and said to the page."""
        def say(_r: Any, _e: Any) -> None:
            if self._auto is None:
                return
            seat = None
            if state == "far":
                try:
                    seat = self._wrong_seat()
                except Exception:
                    log.warning("the seat of the board not told", exc_info=True)
            self._auto_view = {"state": state, "score": score, "seat": seat}
            self._emit({"type": "calib_auto", "state": state, "score": score, "seat": seat})
        self._jobs.put(("", say, None, None))

    def _wrong_seat(self) -> int | None:
        """On the bench, the seat the board stands in when it is not the one of the next step (calib_bench.identify),
        else None."""
        snap, w = self._auto_snapshot, self._auto
        if not snap or snap["plan"] != "bench" or w is None or w.watch._prev is None:
            return None
        K, size = calib_guide.camera("stream" if self._auto_mode == "stream" else "photo")
        return calib_bench.identify(w.watch._prev, snap["pose"], snap["spec"], K, size)[2]

    # ── calibration: a film of the board (the camera app's video mode) ───
    # The camera app films the board while the operator moves the phone through the poses, holding still a moment at
    # each; it records as for a trial video (acquisition.Phone: copied, checked and deleted on the phone). The film
    # goes to film/ in a series folder of its own (frames of two films are never mixed), the frames worth calibrating
    # on are chosen from it (calib_video.py) and written beside it as the series' photos, and the calibration of the
    # video mode runs on them by itself (a session check waits for its button). It is the calibration of the camera
    # app's video, never of its photos nor of the live stream.
    def _film_state(self, series: dict[str, Any] | None) -> dict[str, Any]:
        """The film under way (recording, starting or stopping, since when) and what the series' film gave."""
        f = self._film
        got = calib_video.film_of(series["path"]) if series and self._mode() == "video" else None
        return {"recording": bool(f and f["recording"]), "starting": bool(f and not f["recording"] and not f["stopping"]),
                "stopping": bool(f and f["stopping"]), "since": f["since"] if f else None, "max_s": FILM_MAX_S,
                "choosing": "film" in self._busy,
                "last": {k: got.get(k) for k in ("video", "duration_s", "counts", "refused", "slower")} if got else None}

    def _film_start(self) -> dict[str, Any]:
        """The camera app starts filming (its video mode): calib_film events say when it records."""
        if self._checking() or {"phone", "film"} & self._busy:
            return {"error": "busy"}
        blocked = self._free_camera()
        if blocked:
            return blocked
        phone = self.phone()
        film = {"since": time.time(), "recording": False, "stopping": False}
        self._film = film

        def started(_: Any, error: BaseException | None) -> None:
            if self._film is not film:
                return
            if error is not None:
                self._film = None
                self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error)})
                return
            film["recording"], film["since"] = True, time.time()
            self._emit({"type": "calib_film", "action": "recording"})
            QTimer.singleShot(FILM_MAX_S * 1000, lambda: self._film_timeout(film))
        if not self._spawn("phone", self._with_phone(phone.start_video), started):
            self._film = None
            return {"error": "busy"}
        return {"ok": True, "film": "starting"}

    @guarded
    def _film_timeout(self, film: dict[str, Any]) -> None:
        if self._film is film and film["recording"] and not film["stopping"]:
            log.info("calibration film stopped after %d s", FILM_MAX_S)
            self._film_stop()

    def _film_stop(self) -> dict[str, Any]:
        """The film stops and comes to a series folder of its own; then its frames are chosen (_film_choose)."""
        film = self._film
        if film is None:
            return {"error": "nothing"}
        if not film["recording"] or film["stopping"] or "phone" in self._busy:
            return {"error": "busy"}
        f = self._series()
        empty = f is not None and f["files"] == 0 and not (Path(f["path"]) / calib_video.FILM_DIR).exists()
        d = Path(f["path"]) if empty else self._new_series()
        self._folder_board(d)
        board = self._board_args(d)         # here: it may ask Qt for the monitors
        plan = self._calib_plan()
        lim = calib_guide.limits(plan)
        sim = self._film_scene(plan) if self.roots.simulate else None
        phone, staging = self.phone(), self.roots.staging / "calibration"

        def work() -> Path:
            path = Path(phone.stop_video(staging))
            if sim is not None:
                sim(path)
            dest = d / calib_video.FILM_DIR / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
            return dest
        left = len(self._left_on_phone())

        def done(dest: Path | None, error: BaseException | None) -> None:
            self._film = None
            self._check_left(left)
            if error is not None:
                self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error)})
                return
            self._film_choose(d, dest, board, lim["max"])
        if not self._spawn("phone", self._with_phone(work), done):
            return {"error": "busy"}
        film["stopping"] = True
        return {"ok": True, "film": "stopping", "folder": d.name}

    def _film_scene(self, plan: str) -> Callable[[Path], None]:
        """In simulation the phone's film shows the board in use, moved through the poses of the plan."""
        st = self._board()
        spec, kind = st["active"], st["kind"]
        lay = self._lay if kind == "screen" else None

        def film(path: Path) -> None:
            calib_video.synth_film(path, spec, kind, size=FILM_SIM_SIZE, plan=plan, pause_s=0.5, move_s=0.7,
                                   lay=lay, seed=int(time.time()) % 1000)
            path.with_name(path.stem + "_truth.json").unlink(missing_ok=True)
        return film

    def _film_choose(self, d: Path, video: Path, board: tuple[float, Any, float], target: int) -> None:
        """The frames of the film worth calibrating on, chosen in a worker and written into d as its photos (each with
        its check); then, for a calibration, the calibration on them."""
        mode = self._mode()
        bench = self._calib_plan() == "bench"       # the arm stops at each notch: one still frame a stop
        square, squares, marker = board
        said: list[str] = []

        def work() -> dict[str, Any]:
            b = calibrate_charuco.make_board(square, squares, marker)
            return calib_video.run_choose(video, d, b, target, mode=mode, log=said.append,
                                          need=calibrate_charuco.MIN_VIEWS, stops=bench)

        def done(summary: dict[str, Any] | None, error: BaseException | None) -> None:
            for line in said:
                log.info("film: %s", line)
            self._keep_log(mode, log_text(said))
            if error is not None:
                self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error, "film_unreadable")})
                return
            checks = self._photo_checks(d)
            now = datetime.now().isoformat(timespec="seconds")
            seq = max((c.get("seq", 0) for c in checks.values()), default=0)
            for name, c in sorted(summary["checks"].items(), key=lambda kv: kv[1].get("film_s") or 0):
                seq += 1
                checks[name] = {**c, "time": now, "mode": mode, "seq": seq}
            self._write_checks(d, checks)
            self._emit({"type": "calib_film", "action": "chosen", "folder": d.name, "n": len(summary["checks"]),
                        "seconds": summary["duration_s"], "counts": summary["counts"],
                        "slower": len(summary.get("slower") or [])})
            if self._series_kind() == "calibration":
                self.calib_run("[]")
        if not self._spawn("film", work, done):
            self._emit({"type": "acquire_failed", "key": "calibration", "error": "busy", "args": {}, "message": ""})
            return
        self._emit({"type": "calib_film", "action": "choosing", "folder": d.name})

    def _film_import(self) -> dict[str, Any]:
        """A film chosen by hand (the camera app's own recording, copied by cable) into a series of its own."""
        if self._film is not None or {"phone", "film", "import"} & self._busy:
            return {"error": "busy"}
        pattern = " ".join("*" + s for s in calib_video.VIDEO_SUFFIXES)
        path, _ = QFileDialog.getOpenFileName(None, "ImpactGuide", str(Path.home()), f"video ({pattern})")
        if not path:
            return {**self._calib_state(), "added": 0, "skipped": 0}
        src = Path(path)
        f = self._series()
        empty = f is not None and f["files"] == 0 and not (Path(f["path"]) / calib_video.FILM_DIR).exists()
        d = Path(f["path"]) if empty else self._new_series()
        self._folder_board(d)
        board = self._board_args(d)
        target = calib_guide.limits(self._calib_plan())["max"]
        dest = d / calib_video.FILM_DIR / src.name

        def work() -> Path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            return dest

        def done(copied: Path | None, error: BaseException | None) -> None:
            if error is not None:
                self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error, "copy_failed")})
                return
            self._film_choose(d, copied, board, target)
        if not self._spawn("import", work, done):
            return {"error": "busy"}
        return {**self._calib_state(), "added": 0, "skipped": 0, "film": "copying"}

    def _sim_scene(self):
        """In simulation the phone photographs the board in use at the next pose of the plan (calib_guide.photo)."""
        s = self._calib_state()
        st = self._board()
        kind = st["kind"]
        spec = self._guide_board(s["series"] if s["series"] and s["series"]["same"] else None, st["active"])
        lay = self._lay if kind == "screen" else None
        tex_kind = {"screen": "screen", "plate": "plate", "sheet": "sheet"}[kind]
        n = s["next"]["n"] if s["next"] else s["count"]     # the pose the photo now taken is for
        size = self.phone().photo_px

        def scene():
            import cv2
            img = calib_guide.photo(s["plan"], n, spec, tex_kind, "photo", lay, size=tuple(size))
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return scene

    def _shoot_into(self, d: Path, check: bool = True, pose: int | None = None, auto: bool = False) -> dict[str, Any]:
        """A photo from the phone for the pose `pose` of the plan into the folder d, then checked. The phone's job is
        only the photo (the shutter and the file whole on the phone); its copy (moved into d: d may be on Drive) and
        its check follow in the background, while the next photo can be taken."""
        blocked = self._free_camera()
        if blocked:
            return blocked
        if "phone" in self._busy:
            return {"error": "busy"}
        self._folder_board(d)
        phone = self.phone()
        if self.roots.simulate and check:
            phone.scene = self._sim_scene()
        shot = {"path": d, "folder": d.name, "pose": pose, "name": "", "state": "shooting", "auto": auto,
                "t0": time.monotonic()}

        def pressed() -> None:      # on the worker: said on the GUI thread, through the jobs' queue
            shot["t_press"] = time.monotonic()
            self._jobs.put(("", lambda _r, _e: self._pressed(shot), None, None))

        def work() -> str:
            # the camera app left in the mode the operator set (Pro, the focus by hand): no intent to it, not even the
            # one that lifts its own timer, which may bring it back to its Photo mode (acquisition.Phone.take)
            return phone.take(pressed=pressed, keep_mode=True)

        def done(name: str | None, error: BaseException | None) -> None:
            if self.roots.simulate:
                phone.scene = None
            try:
                if error is not None:
                    self._drop_shot(shot)
                    if self._auto is not None:
                        self._auto.refused()
                    self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error)})
                    return
                if shot["state"] == "shooting":     # the press was said after the photo (a quick fake phone)
                    self._pressed(shot)
                shot["name"], shot["state"], shot["t_taken"] = name, "copying", time.monotonic()
                self._copy_queue.append((shot, check))
                self._next_copy()
                if getattr(phone, "countdown_seen", False) and not self._countdown_said:
                    self._countdown_said = True
                    self._emit({"type": "calib_photo", "action": "countdown", "folder": d.name})
            finally:
                self._take_queued()
                self._maybe_calibrate()
        self._shots.append(shot)
        if not self._spawn("phone", self._with_phone(work), done):
            self._drop_shot(shot)
            return {"error": "busy"}
        return {"ok": True, "folder": d.name}

    def _next_copy(self) -> None:
        """The next photo taken copied from the phone (a copier of its own: the next photo may be taken meanwhile),
        its small picture made, moved into its folder, then queued for its check."""
        if "phone_copy" in self._busy or not self._copy_queue:
            return
        shot, check = self._copy_queue.pop(0)
        d, name = Path(shot["path"]), shot["name"]
        copier = self.phone().copier()
        staging = self.roots.staging / "calibration"
        thumb = self._thumb_path(d, name)
        left = len(self._left_on_phone())

        def work() -> Path:
            path = Path(copier.pull(name, staging))
            self._make_thumb(path, thumb)
            dest = d / path.name
            if dest.exists():       # never over a photo of the series: this one stays in staging
                log.warning("%s already in %s: the new photo stays in %s", dest.name, d, path)
                raise UserError("exists", dest.name)
            shutil.move(str(path), str(dest))
            return dest

        def done(dest: Path | None, error: BaseException | None) -> None:
            try:
                self._check_left(left)
                if error is not None:
                    self._drop_shot(shot)
                    self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error)})
                    return
                shot["state"], shot["t_copied"] = "checking", time.monotonic()
                if not check:
                    self._drop_shot(shot)
                self._arrived(d, dest, check)
            finally:
                self._next_copy()
                self._maybe_calibrate()
        if not self._spawn("phone_copy", work, done):
            self._copy_queue.insert(0, (shot, check))

    def _grab_into(self, d: Path, pose: int | None = None, auto: bool = False) -> dict[str, Any]:
        """The newest frame of the live stream, at its full size, into the series as a PNG (lossless), for the pose
        `pose` of the plan; checked in the background as a photo is."""
        ar = self._live
        if ar is None or not ar.running or self._live_for != "calibration":
            return {"error": "stream_off"}
        img, n = ar.still()
        if img is None:
            return {"error": "stream_off"}
        self._folder_board(d)
        name = f"frame_{datetime.now():%Y%m%d_%H%M%S}_{n:06d}.png"
        thumb = self._thumb_path(d, name)
        shot = {"path": d, "folder": d.name, "pose": pose, "name": name, "state": "shooting", "auto": auto,
                "t0": time.monotonic()}

        def work() -> Path:
            import cv2
            cv2.imwrite(str(d / name), img)
            self._make_thumb(d / name, thumb)
            return d / name

        def done(path: Path | None, error: BaseException | None) -> None:
            try:
                if error is not None:
                    self._drop_shot(shot)
                    self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error)})
                    return
                shot["state"] = "checking"
                self._arrived(d, path, True)
            finally:
                self._take_queued()
                self._maybe_calibrate()
        self._shots.append(shot)
        if not self._spawn("phone", work, done):
            self._drop_shot(shot)
            return {"error": "busy"}
        self._pressed(shot)         # the frame is the one of this moment
        return {"ok": True, "folder": d.name}

    def _arrived(self, d: Path, dest: Path, check: bool = True) -> None:
        """A photo is in the series: checked as soon as the checks before it are done (board found, corners, blur, the
        same size as the others). One that fails goes aside, to the folder's rejected/, and the page says why; its
        pose comes again."""
        if not check:
            self._emit({"type": "calib_photo", "folder": d.name, "name": dest.name})
            return
        self._queue_checks(d, [dest])

    def _checking(self) -> bool:
        """Photos are being checked, or wait for their check."""
        return "photo_check" in self._busy or bool(self._check_queue)

    def _queue_checks(self, d: Path, photos: list[Path]) -> None:
        """Photos of the folder d queued for their check, one worker checking them one after the other in the order
        they came: none is ever left unchecked because another check was running. In a folder never checked before
        its photos already there go first, oldest first: they are the series the new ones join (their size, their
        camera settings), and from now on they count only once checked (_counted)."""
        checked = self._photo_checks(d)
        if photos and not (Path(d) / PHOTO_CHECKS).exists():
            new = {Path(p).name for p in photos}
            photos = [p for p in self._unchecked(Path(d), {}) if p.name not in new] + list(photos)
        now = self._check_now
        for p in photos:
            # a photo is checked once: never again while it waits, while it is checked, or once it has its check
            if (any(str(f) == str(d) and Path(q) == Path(p) for f, q in self._check_queue) or Path(p).name in checked
                    or (now is not None and str(now[0]) == str(d) and Path(now[1]) == Path(p))):
                continue
            self._check_queue.append((d, p))
        self._next_check()

    def _next_check(self) -> None:
        """The next photo of the queue checked in a worker, with the board and the size worked out here first."""
        while self._check_queue and "photo_check" not in self._busy:
            d, dest = self._check_queue.pop(0)
            if not dest.exists():
                log.info("photo %s gone before its check", dest)
                for x in [x for x in self._shots_in(d) if x["name"] == dest.name]:
                    self._drop_shot(x)
                # said, so that the page ends the work it started for this photo
                self._emit({"type": "acquire_failed", "key": "calibration", "error": "not_found",
                            "args": {"name": dest.name}, "message": ""})
                continue
            try:
                square, squares, marker = self._board_args(d)
            except UserError as e:
                self._check_queue = [(f, p) for f, p in self._check_queue if f != d]
                for x in [x for x in self._shots_in(d) if x["state"] == "checking"]:
                    self._drop_shot(x)
                self._emit({"type": "acquire_failed", "key": "calibration", **self._err(e)})
                continue
            try:
                checks = self._photo_checks(d)
                kept = [p for p in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime)
                        if p.suffix.lower() in IMAGE_SUFFIXES and p != dest and (checks.get(p.name) or {}).get("ok")]
                size = (checks.get(kept[0].name) or {}).get("size") if kept else None
                # the camera's settings of the first photo kept (its check has them, else its file): a photo with
                # another ISO, exposure time, lens or resolution is refused (calibrate_charuco.settings_problems)
                first = ((checks.get(kept[0].name) or {}).get("settings") or kept[0]) if kept else None
            except OSError as e:        # the folder could not be listed (Drive, a file gone meanwhile)
                try:
                    self._checked(d, dest, None, e)     # the photo counts as not checked, put aside, said
                except OSError:
                    log.warning("photo %s: its failed check not written", dest, exc_info=True)
                    self._emit({"type": "acquire_failed", "key": "calibration", **self._err(e)})
                continue

            def work(dest: Path = dest, size: Any = size, square: float = square, squares: Any = squares,
                     marker: float = marker, thumb: Path = self._thumb_path(d, dest.name),
                     first: Any = first) -> dict[str, Any]:
                board = calibrate_charuco.make_board(square, squares, marker)
                if not thumb.is_file():         # a photo added by hand: its small picture made here
                    self._make_thumb(dest, thumb)
                return calibrate_charuco.check_photo(dest, board, size, first)

            def done(res: dict[str, Any] | None, error: BaseException | None, d: Path = d, dest: Path = dest) -> None:
                self._check_now = None
                try:
                    self._checked(d, dest, res, error)
                finally:
                    self._next_check()
                    self._maybe_calibrate()
            if self._spawn("photo_check", work, done):
                self._check_now = (d, dest)

    def _checked(self, d: Path, dest: Path, res: dict[str, Any] | None, error: BaseException | None) -> None:
        """The check of a photo kept beside it (atomically); a photo that failed it goes aside."""
        if error is not None:
            log.warning("photo %s not checked", dest, exc_info=error)
            res = {"ok": False, "why": "unreadable", "corners": 0, "of": 0, "blur_px": None, "points": []}
        res["time"] = datetime.now().isoformat(timespec="seconds")
        res["mode"] = self._mode()
        shot = next((x for x in self._shots_in(d) if x["name"] == dest.name), None)
        if shot is not None:                # the pose of the plan it was taken for, kept with its check
            res["pose"] = shot["pose"]
            # seconds from Take (or the photo taken by itself) to the shutter, the photo whole on the phone, copied,
            # checked: how long the next photo waited, if at all
            t0, now = shot.get("t0"), time.monotonic()
            if t0 is not None:
                res["timing_s"] = {k: round(shot[t] - t0, 2) for k, t in (("shutter", "t_press"),
                                   ("taken", "t_taken"), ("copied", "t_copied")) if shot.get(t) is not None}
                res["timing_s"]["checked"] = round(now - t0, 2)
            res["auto"] = bool(shot.get("auto"))
            self._drop_shot(shot)
        if not res["ok"]:
            aside = d / REJECTED
            try:
                aside.mkdir(exist_ok=True)
                shutil.move(str(dest), str(aside / dest.name))
            except OSError:
                log.warning("photo %s not put aside", dest, exc_info=True)
        allc = self._photo_checks(d)
        res["seq"] = 1 + max((c.get("seq", 0) for c in allc.values()), default=0)    # the order they came in
        allc[dest.name] = res
        self._write_checks(d, allc)
        if self._auto is not None and not res["ok"]:
            self._calib_state()             # its pose comes again: the watcher back on it
        self._emit({"type": "calib_photo", "folder": d.name, "name": dest.name,
                    "check": {k: res.get(k) for k in ("ok", "why", "corners", "of", "blur_px")}})

    @pyqtSlot(result=str)
    def calib_retake(self) -> str:
        """The last photo kept goes aside after all (the operator knows better): its pose comes again."""
        def go() -> dict[str, Any]:
            f = self._series()
            if f is None:
                return {"error": "nothing"}
            d = Path(f["path"])
            checks = self._photo_checks(d)
            kept = [k for k, c in checks.items() if c.get("ok") and (d / k).exists()]
            if not kept:
                return {"error": "nothing"}
            name = max(kept, key=lambda k: (checks[k].get("seq", 0), checks[k].get("time", "")))
            (d / REJECTED).mkdir(exist_ok=True)
            shutil.move(str(d / name), str(d / REJECTED / name))
            checks[name] = {**checks[name], "ok": False, "why": "operator"}
            self._write_checks(d, checks)
            return self._calib_state()
        return self._answer(go)

    @pyqtSlot(bool, result=str)
    def calib_stream(self, on):
        """The live stream for a calibration of the streaming mode: on while the Photos step of that mode is open,
        drawn plain (corners only) and, with the board on this monitor, with the board hidden in the picture."""
        def go():
            if not on:
                if self._live is not None and self._live_for == "calibration":
                    self._live.stop()
                self._live_purpose("live")
                return self._calib_state()
            ar = self.live()
            self._live_purpose("calibration")
            if ar.running:
                return self._calib_state()
            if "phone" in self._busy or self.cur.get("recording") or self._film is not None:
                return {"error": "busy"}
            if not self._spawn("live", ar.start, self._live_started("calibration")):
                return {"error": "live_starting"}
            return self._calib_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_board_pdf(self) -> str:
        """The A4 sheet to print (a board one may have instead of the plate), opened."""
        def go() -> dict[str, Any]:
            pdf, said = captured(calibrate_charuco.write_board, self.roots.board, calibrate_charuco.BOARD["square"])
            log.info("printable sheet: %s", said.strip())
            self._open(pdf)
            return {"ok": True}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def plate_open(self, what: str) -> str:
        """The 3D-printed plate: its two-colour print file (3MF, in the slicer) or the folder with its STL."""
        def go() -> dict[str, Any]:
            plate = charuco_plate.load()
            folder = charuco_plate.PLATE_FILE.parent
            path = folder / plate["files"]["3mf"] if what == "3mf" else folder
            if not path.exists():
                return {"error": "not_found", "message": path.name}
            self._open(path)
            return {"ok": True}
        return self._answer(go)

    @staticmethod
    def _open(path: Path | str) -> None:
        """A file or a folder opened by Windows (the file itself: a folder window opens behind this frameless one)."""
        open_externally(path)

    def _folder_board(self, d: Path) -> None:
        """board.json in a photo folder: the active board, written while the folder has no photos (so it is the
        board the first photo is of); a folder with photos keeps the one it has, or has none (photos imported by
        hand into a folder without one: their board is not known, so none is made up)."""
        p = d / calibrate_charuco.BOARD_FILE
        if any(x.suffix.lower() in IMAGE_SUFFIXES for x in d.iterdir()):
            return
        a = self._board()["active"]
        if not a:
            return
        write_json_atomic(p, {**a, "written": datetime.now().isoformat(timespec="seconds")})

    def _board_args(self, d: Path) -> tuple[float, tuple[int, int], float]:
        """(square, squares, marker) for calibrate_charuco: the folder's board.json, else the active board. Run it on
        the GUI thread: the active board asks Qt for the monitors.

        Raises:
            UserError: board_unsupported (another kind of board) or board_unreadable (a board.json that cannot be
                read), with the folder's name; no_board when there is neither board.json nor an active board.
        """
        try:
            got = calibrate_charuco.folder_board(d)
        except calibrate_charuco.BoardError as e:
            log.warning("board.json of %s refused: %s", d, e)
            raise UserError(e.code, d.name, **e.params) from e
        if got:
            return got
        a = self._board()["active"]
        if not a:
            raise UserError("no_board")
        return a["square_mm"], tuple(a["squares"]), a["marker_mm"]

    def _calib_dir(self, name: str) -> Path:
        """A folder of calibration photos by name (only directly under calib_raw)."""
        d = (self.roots.calib_raw / name).resolve()
        if d.parent != self.roots.calib_raw.resolve() or not d.is_dir():
            raise UserError("not_found", name)
        return d

    @pyqtSlot(str, result=str)
    def calib_shoot(self, name: str) -> str:
        """A photo from the phone into a folder chosen by name, under the name the phone gave it (not checked)."""
        return self._answer(lambda: self._shoot_into(self._calib_dir(name), check=False))

    @pyqtSlot(str, result=str)
    def calib_folder_new(self, kind: str) -> str:
        """Today's folder for calibration photos (photo, stream) or for the session check (check_photo...)."""
        def go() -> dict[str, Any]:
            if kind not in CALIB_MODES + tuple("check_" + m for m in CALIB_MODES):
                return {"error": "bad_value"}
            d = self.roots.calib_raw / f"{date.today():%Y%m%d}_{kind}"
            d.mkdir(parents=True, exist_ok=True)
            self._folder_board(d)
            return {**self._calib_state(), "folder": d.name}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def calib_import(self, name: str) -> str:
        """Photo files added to a folder by name, or to the series (name empty), each checked as a photo taken. The
        copies are made in a worker (the folder may be on Drive); each photo is checked once it is there. In the video
        mode: a film of the camera app (copied by cable), into a series of its own, its frames chosen as for a film
        taken here."""
        def go() -> dict[str, Any]:
            if self._mode() == "video" and not name:
                return self._film_import()
            d = self._calib_dir(name) if name else self._series_folder()
            pattern = " ".join("*" + s for s in IMAGE_SUFFIXES)
            paths, _ = QFileDialog.getOpenFileNames(None, "ImpactGuide", str(Path.home()), f"photos ({pattern})")
            if paths:
                self._folder_board(d)
            todo = [(Path(p), d / Path(p).name) for p in paths if not (d / Path(p).name).exists()]
            skipped = len(paths) - len(todo)
            if todo:
                def work() -> list[Path]:
                    for src, dest in todo:
                        shutil.copy2(src, dest)
                    return [dest for _, dest in todo]

                coming = {(str(d), dest.name) for _, dest in todo}

                def done(copied: list[Path] | None, error: BaseException | None) -> None:
                    self._importing -= coming
                    if error is not None:
                        self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error, "copy_failed")})
                        copied = [dest for _, dest in todo if dest.exists()]
                    self._queue_checks(d, copied or [])
                if not self._spawn("import", work, done):
                    return {"error": "busy"}
                self._importing |= coming       # never taken for photos left unchecked while they are copied in
            return {**self._calib_state(), "added": len(todo), "skipped": skipped}
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_open(self) -> str:
        def go() -> dict[str, Any]:
            f = self._series()
            if f is None:
                return {"error": "nothing"}
            open_externally(f["path"])
            return {"ok": True}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def calib_run(self, exclude: str) -> str:
        """Calibrates the camera mode on the photos of the series, leaving out those named in exclude (a JSON list);
        the result waits for calib_save. With photos still on their way (taken, copied or checked) it starts by
        itself once the last one is checked (waiting)."""
        def go() -> dict[str, Any]:
            leave = [str(x) for x in json.loads(exclude or "[]")]
            if self._on_way():
                self._calib_waiting = leave
                return {"ok": True, "waiting": True}
            f = self._series()
            skip: dict[str, str] = {}
            if f is not None and (Path(f["path"]) / PHOTO_CHECKS).exists():
                # only the photos their check kept; one without a check yet is checked first
                d = Path(f["path"])
                checks = self._photo_checks(d)
                orphans = self._unchecked(d, checks)
                if orphans:
                    self._queue_checks(d, orphans)
                    self._calib_waiting = leave
                    return {"ok": True, "waiting": True}
                skip = {p.name: "check" for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                        and not (checks.get(p.name) or {}).get("ok")}
            if f is None or not f["n"]:
                return {"error": "nothing"}
            d, mode, plan = Path(f["path"]), self._mode(), self._calib_plan()
            board = self._board_args(d)         # here: it may ask Qt for the monitors
            said_so_far: list[str] = []         # what it logged, kept for "Open the log" even when it fails

            def work() -> tuple[dict[str, Any], str]:
                return captured(calibrate_charuco.run_calibrate, d, None, None, *board, exclude=leave,
                                sink=said_so_far, skip=skip)

            def done(result: tuple[dict[str, Any], str] | None, error: BaseException | None) -> None:
                if error is not None:
                    self._keep_log(mode, log_text(said_so_far))
                    self._emit({"type": "calib", "action": "calibrate", "mode": mode, "ok": False,
                                **self._err(error, "calibration")})
                    return
                cal, said = result
                cal["plan"] = plan              # the verdict keeps the plan the photos were taken with
                cal["mode"] = mode              # its camera mode, never used for another
                film = calib_video.film_of(d)
                if film:                        # frames chosen from a film: which, and what the film was
                    cal["film"] = {k: film.get(k) for k in ("video", "size", "fps", "frames", "duration_s", "codec",
                                                            "counts", "limits", "closest_steps")}
                self._pending_cal[mode] = cal
                self._keep_log(mode, said)
                self._emit({"type": "calib", "action": "calibrate", "mode": mode, "ok": self._good(cal),
                            "result": self._cal_summary(cal)})
            ok = self._spawn("calibration", work, done)
            return {"ok": ok} if ok else {"error": "busy"}
        return self._answer(go)

    def _log_path(self, mode: str) -> Path:
        return self.roots.calib_out / f"last_{mode}.log"

    def _keep_log(self, mode: str, said: str) -> None:
        """What the last calibration or check of a mode said, kept for calib_log_open."""
        self._calib_log[mode] = said

    @pyqtSlot(result=str)
    def calib_log_open(self) -> str:
        """The log of the last calibration or session check of the mode in use, written beside the intrinsics
        (last_<mode>.log) and opened as a text file: the page shows only plain words, never the log itself."""
        def go() -> dict[str, Any]:
            mode = self._mode()
            path = self._log_path(mode)
            if mode in self._calib_log:
                write_text_atomic(path, self._calib_log[mode])
            if not path.exists():
                return {"error": "nothing"}
            self._open(path)
            return {"ok": True}
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_save(self) -> str:
        def go() -> dict[str, Any]:
            mode = self._mode()
            cal = self._pending_cal.get(mode)
            if cal is None:
                return {"error": "nothing"}
            write_json_atomic(self._intrinsics_path(mode), cal)
            self._pending_cal.pop(mode, None)
            return self._calib_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_check(self) -> str:
        """Start-of-session check of the stored intrinsics on the photos of the series."""
        def go() -> dict[str, Any]:
            f = self._series()
            mode = self._mode()
            if f is None or not f["n"]:
                return {"error": "nothing"}
            if self._on_way():                  # photos still being taken, copied or checked
                return {"error": "busy"}
            d = Path(f["path"])
            intr = self._intrinsics_path(mode)
            if not intr.exists():
                return {"error": "no_intrinsics", "message": mode}
            board = self._board_args(d)         # here: it may ask Qt for the monitors
            said_so_far: list[str] = []

            def work() -> tuple[tuple[bool, dict[str, Any]], str]:
                return captured(calibrate_charuco.run_check, d, intr, *board, sink=said_so_far)

            def done(result: Any, error: BaseException | None) -> None:
                if error is not None:
                    self._keep_log(mode, log_text(said_so_far))
                    self._emit({"type": "calib", "action": "check", "mode": mode, "ok": False,
                                **self._err(error, "check")})
                    return
                (ok, report), said = result
                p = report["params"]
                line = {"time": datetime.now().isoformat(timespec="seconds"), "mode": mode, "folder": d.name,
                        "ok": int(ok), "n_views": report["n_views"], "rms_stored_px": report["rms_stored_px"],
                        **{f"{k}_change": p[k]["change"] for k in ("fx", "fy", "cx", "cy")}}
                self._keep_log(mode, said)
                try:
                    append_csv_row(self._checks_path(), line, CHECK_FIELDS)
                except OSError:
                    log.warning("session check not logged in %s", self._checks_path(), exc_info=True)
                self._emit({"type": "calib", "action": "check", "mode": mode, "ok": ok, "report": report})
            started = self._spawn("calibration", work, done)
            return {"ok": started} if started else {"error": "busy"}
        return self._answer(go)

    # ── the calibration board: on this monitor, in the page; or physical ─
    # On the monitor the board is drawn by the page itself, in the work area of the step that shows it (a canvas of
    # the page, ui/15-board.js): the page tells the area's size in its device pixels (board_area), which on a window
    # whose page is not scaled are the monitor's own pixels; the board is laid out for them at the pitch of the
    # monitor the window is on (screen_board), and drawn one image pixel on one screen pixel. It is laid out again
    # when the area changes (the window resized, the step changed) or the window goes to another monitor; nothing
    # moves or resizes the window. The physical boards: the 3D-printed plate (charuco_plate.py, its 3D model shown
    # by the page) and a printed A4 sheet, each with its length measured three times.
    def _screens(self):
        """([monitor of screen_board.monitors], {key: QScreen}): what Qt and Windows say of each monitor now, with
        this computer's ruler checks. Windows is asked again only when Qt's screens change."""
        app = QApplication.instance()
        screens = list(app.screens()) if app is not None else []
        qt = [screen_board.qt_screen(sc, sc is app.primaryScreen()) for sc in screens]
        sig = json.dumps([[q["device"], q["geometry"], q["dpr"]] for q in qt])
        if self._win_mons[0] != sig:
            self._win_mons = (sig, screen_board.windows_monitors())
        mons = screen_board.monitors(qt, self._win_mons[1], self.prefs["monitors"])
        return mons, {m["key"]: sc for m, sc in zip(mons, screens)}

    def _app_screen(self):
        """The screen the app's window is on (the main one without a window: the tests)."""
        app = QApplication.instance()
        sc = self.window.screen() if self.window is not None else None
        return sc or (app.primaryScreen() if app is not None else None)

    def _app_monitor(self):
        """The monitor (screen_board.monitors) the app's window is on."""
        mons, _ = self._screens()
        sc = self._app_screen()
        dev = sc.name() if sc is not None else ""
        return next((m for m in mons if m["device"] == dev), None) or (mons[0] if mons else None)

    def _watch_screen(self) -> None:
        """The page is told when the window goes to another monitor or its monitor changes (resolution, scaling):
        the board is laid out again for it."""
        if self.window is None or self._screen_watch:
            return
        self._screen_watch = True

        @guarded
        def moved(*_: Any) -> None:
            sc = self._app_screen()
            if sc is not None and sc is not self._watched_screen:
                for sig in (getattr(self._watched_screen, "geometryChanged", None),
                            getattr(self._watched_screen, "physicalDotsPerInchChanged", None)):
                    with contextlib.suppress(TypeError, RuntimeError, AttributeError):
                        sig.disconnect(moved)
                self._watched_screen = sc
                self._connect(sc.geometryChanged, moved)
                self._connect(sc.physicalDotsPerInchChanged, moved)
            self._emit({"type": "board", "action": "screen"})
        self._connect(self.window.screenChanged, moved)
        sc = self._app_screen()
        if sc is not None:
            self._watched_screen = sc
            self._connect(sc.geometryChanged, moved)
            self._connect(sc.physicalDotsPerInchChanged, moved)

    def _kind(self) -> str:
        k = self.prefs["board"].get("kind")
        return k if k in BOARD_KINDS else "screen"

    @staticmethod
    def _sheet_reading(values: list[Any]) -> dict[str, Any]:
        """The A4 sheet's long side as measured: mean, spread and problems (as charuco_plate.reading).

        Raises:
            ValueError: no readings, or one that is not a finite length above 0 (as charuco_plate.reading).
        """
        v = [float(x) for x in values]
        if not v or not all(math.isfinite(x) and x > 0 for x in v):
            raise ValueError(f"sheet readings {values!r}: not lengths")
        nominal = calibrate_charuco.BOARD["squares"][0] * calibrate_charuco.BOARD["square"]
        mean, spread = sum(v) / len(v), max(v) - min(v)
        problems = []
        if len(v) < charuco_plate.MIN_READINGS:
            problems.append("few_readings")
        if spread > SHEET_SPREAD_MAX:
            problems.append("spread")
        if abs(mean / nominal - 1) > SHEET_NOMINAL_TOL:
            problems.append("off_nominal")
        return {"which": "long", "mean_mm": round(mean, 3), "spread_mm": round(spread, 3), "n": len(v),
                "nominal_mm": nominal, "scale": mean / nominal, "problems": problems}

    def _reading(self, read: Callable[..., dict[str, Any]], values: Any, *args: Any) -> dict[str, Any] | None:
        """read(values, *args) for readings kept in the preferences, or None when there are none or they are not
        lengths (a file edited by hand): the board is then the nominal one, never an error on every board state
        (and the log says it once)."""
        if not values:
            return None
        try:
            return read(values, *args)
        except (ValueError, TypeError):
            if repr(values) not in self._refused_readings:
                self._refused_readings.add(repr(values))
                log.warning("caliper readings %r refused: the nominal board is used", values)
            return None

    def _physical(self) -> dict[str, Any]:
        """The plate and the sheet: their nominal sizes, what to measure, the readings typed, and the board each
        gives (as measured when the readings can be used, else nominal)."""
        b = self.prefs["board"]
        out = {}
        try:
            plate = charuco_plate.load()
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
            if not self._plate_warned:          # asked at every board state: said once
                log.warning("plate file unreadable: the plate is not offered", exc_info=True)
                self._plate_warned = True
            plate = None
        if plate is not None:
            pr = b.get("plate") or {}
            long_r = self._reading(charuco_plate.reading, pr.get("long"), "long", plate)
            short_r = self._reading(charuco_plate.reading, pr.get("short"), "short", plate)
            use_long = long_r["mean_mm"] if long_r and not long_r["problems"] else None
            use_short = short_r["mean_mm"] if short_r and not short_r["problems"] and use_long else None
            spec = charuco_plate.spec(use_long, use_short, plate)
            spec["measured"] = use_long is not None
            out["plate"] = {"board": plate["board"], "plate_mm": plate["plate"]["size_mm"],
                            "measure": plate["measure"], "files": plate["files"], "readings": pr,
                            "long": long_r, "short": short_r, "spec": spec, "model": PLATE_MODEL.is_file(),
                            "min_readings": charuco_plate.MIN_READINGS, "spread_max": charuco_plate.SPREAD_MAX}
        sr = b.get("sheet") or {}
        long_s = self._reading(self._sheet_reading, sr.get("long"))
        sq = long_s["mean_mm"] / calibrate_charuco.BOARD["squares"][0] if long_s and not long_s["problems"] \
            else calibrate_charuco.BOARD["square"]
        spec = {"kind": "sheet", **calibrate_charuco.board_spec(calibrate_charuco.make_board(sq)),
                "measured": bool(long_s and not long_s["problems"])}
        out["sheet"] = {"squares": list(calibrate_charuco.BOARD["squares"]), "square_mm": calibrate_charuco.BOARD["square"],
                        "nominal_mm": calibrate_charuco.BOARD["squares"][0] * calibrate_charuco.BOARD["square"],
                        "readings": sr, "long": long_s, "spec": spec, "min_readings": charuco_plate.MIN_READINGS,
                        "spread_max": SHEET_SPREAD_MAX}
        return out

    def _board(self):
        """The board as it stands: its kind, the monitor the window is on with the layout the page shows (while it
        does), the physical boards, and the active board, what the calibration and the live AR use."""
        kind = self._kind()
        mon = self._app_monitor()
        phys = self._physical()
        screen = screen_board.spec(self._lay, self._lay_mon or mon) if self._lay else None
        if kind == "screen":
            active = screen
        else:
            active = phys.get(kind, phys["sheet"])["spec"]
        return {"kind": kind, "mon": self._lay_mon or mon, "lay": self._lay, "problem": self._lay_err,
                "physical": phys, "active": active}

    def _board_state(self):
        st = self._board()
        mon, lay = st["mon"], st["lay"]
        view = None
        if lay:
            view = {k: lay[k] for k in ("squares", "square_px", "marker_px", "square_mm", "marker_mm", "width_mm",
                                        "height_mm", "pitch_mm", "origin_px", "size_px", "line_y")}
            view["width_u_mm"] = lay["width_mm"] * ((mon or {}).get("rel_u") or 0.0)
        keys = ("key", "label", "device", "primary", "px", "native", "is_native", "dpr", "diag_in", "pitch_mm",
                "pitch_data_mm", "rel_u", "source", "sources", "agree", "spread", "ruler")
        b = self.prefs["board"]
        return {"kind": st["kind"], "chosen": bool(b.get("chosen")), "mon": {k: mon[k] for k in keys} if mon else None,
                "layout": view, "problem": st["problem"], "physical": st["physical"], "active": st["active"],
                "shown": self._board_on, "exact": self._lay_exact,
                "held": bool(self._lay and self._series_hold()
                             and self._lay["square_px"] == self._series_hold().get("square_px"))}

    @pyqtSlot(result=str)
    def board_state(self):
        return self._answer(self._board_state)

    def _num(self, v: float, n: int = 1) -> str:
        text = f"{v:.{n}f}"
        return text.replace(".", ",") if self.lang in COMMA_LANGS else text

    @pyqtSlot(str, result=str)
    def board_set(self, payload: str) -> str:
        """{kind (screen, plate, sheet; the first question answered), plate_long, plate_short, sheet_long (the
        readings typed, mm, a list each), ruler_mm (the length between the lines above the board on the monitor, as a
        ruler reads it), forget (the ruler check of this monitor)}."""
        def go() -> dict[str, Any]:
            c = json.loads(payload)
            b = dict(self.prefs["board"])
            if c.get("kind") in BOARD_KINDS:
                b["kind"], b["chosen"] = c["kind"], True
            for key, part, which in (("plate_long", "plate", "long"), ("plate_short", "plate", "short"),
                                     ("sheet_long", "sheet", "long")):
                if key in c:
                    vals = [float(x) for x in (c[key] or []) if x not in (None, "") and float(x) > 0]
                    if any(not READING_MM[0] <= v <= READING_MM[1] for v in vals):   # NaN fails too
                        return {"error": "bad_value", "message": ""}
                    b[part] = {**(b.get(part) or {}), which: vals}
            self.prefs.set("board", b)
            if c.get("ruler_mm") is not None or c.get("forget"):
                mon, lay = self._lay_mon or self._app_monitor(), self._lay
                if mon is None or lay is None:
                    return {"error": "board_hidden"}
                checks = dict(self.prefs["monitors"])
                if c.get("forget"):
                    checks.pop(mon["key"], None)
                else:
                    try:
                        page_pitch = screen_board.ruler_pitch(lay, float(c["ruler_mm"]))
                    except (ValueError, TypeError):
                        return {"error": "ruler_bad", "message": self._num(lay["width_mm"])}
                    # the board was laid out in the page's pixels, `_lay_scale` screen pixels each: the monitor's own
                    # pitch is per screen pixel (the next layout multiplies by the page's scale again)
                    scale = self._lay_scale
                    width_px = lay["squares"][0] * lay["square_px"] * scale
                    checks[mon["key"]] = {"pitch_mm": page_pitch / scale, "measured_mm": float(c["ruler_mm"]),
                                          "data_mm": round(mon["pitch_data_mm"] * width_px, 2)
                                          if mon["pitch_data_mm"] else None,
                                          "expected_mm": round(lay["width_mm"], 2),
                                          "date": datetime.now().isoformat(timespec="seconds")}
                self.prefs.set("monitors", checks)
                if self._area is not None:
                    self._relay()                      # the same area, the board in the new millimetres
            self._sync_live()
            return self._board_state()
        return self._answer(go)

    def _relay(self, area: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """The board for the page's area (device pixels) at the pitch of the monitor the window is on: the layout,
        whether the page's pixels are the screen's (exact) and an error (no_monitor, too_small, across)."""
        c = area or self._area
        mon = self._app_monitor()
        lay, err, exact, scale = None, "", True, 1.0
        if mon is None:
            err = "no_monitor"
        else:
            w, h = int(c["w"]), int(c["h"])
            scale, inside = self._page_scale(c)
            if not (math.isfinite(scale) and scale > 0):     # a minimised window has no client area
                scale, inside = 1.0, True
            exact = abs(scale - 1.0) < 1e-9
            pitch = (mon["pitch_mm"] or GUESS_PITCH) * scale
            try:
                lay = screen_board.layout((w, h), pitch, self._series_hold())
            except (ValueError, ZeroDivisionError):
                err = "too_small"
            if lay is not None and not inside:
                err = "across"
        self._lay, self._lay_err, self._lay_mon, self._lay_exact = lay, err, mon, exact
        self._lay_scale = scale
        return lay

    def _page_scale(self, c):
        """(how many screen pixels one device pixel of the page is, whether the area lies on one monitor): the
        page's device pixels against the window's client area in physical pixels, from Windows. 1 and True off
        Windows (the tests' offscreen window)."""
        if self.window is None or os.name != "nt" or QApplication.instance().platformName() != "windows":
            return 1.0, True
        got = screen_board.window_area(int(self.window.winId()))
        if not got:
            return 1.0, True
        (cx, cy, cw, ch), (mx, my, mw, mh), _ = got
        iw, dpr = float(c.get("iw") or 0), float(c.get("dpr") or 1)
        scale = cw / (iw * dpr) if iw > 0 else 1.0
        if abs(cw - iw * dpr) <= 1.0:          # Chromium rounds the page's width to whole device pixels
            scale = 1.0
        x0, y0 = cx + float(c.get("x", 0)) * scale, cy + float(c.get("y", 0)) * scale
        x1, y1 = x0 + float(c["w"]) * scale, y0 + float(c["h"]) * scale
        inside = mx - 0.5 <= x0 and y0 >= my - 0.5 and x1 <= mx + mw + 0.5 and y1 <= my + mh + 0.5
        return scale, inside

    @pyqtSlot(str, result=str)
    def board_area(self, payload):
        """The page's area for the board: {w, h (device pixels; 0 when no step shows it), dpr, x, y (its corner in
        the page, device pixels), iw (the page's width, CSS pixels), have (the key of the board the page has)}.
        Answers the layout with its key and, when the page's board is not this one, "pending": its image (PNG, one
        pixel per device pixel) follows as a board event, made in a worker (_board_image)."""
        def go():
            c = json.loads(payload)
            self._watch_screen()
            if not c.get("w") or not c.get("h"):
                self._board_on = False
                return {"shown": False}
            self._area = c
            self._board_on = True
            before = self._live_board()
            lay = self._relay(c)
            if self._live_board() != before:
                self._sync_live()
            out = {"shown": True, "problem": self._lay_err, "exact": self._lay_exact, "state": self._board_state()}
            if lay is None:
                out["key"] = ""
                return out
            key = self._lay_key(lay)
            out["key"] = key
            if key != c.get("have"):
                out["pending"] = True
                self._board_image(key, lay)
            return out
        return self._answer(go)

    @staticmethod
    def _lay_key(lay: dict[str, Any]) -> str:
        """What tells one board laid out from another: its size, origin, squares, square and marker in pixels."""
        return (f"{lay['size_px'][0]}x{lay['size_px'][1]}:{lay['origin_px'][0]},{lay['origin_px'][1]}:"
                f"{lay['squares'][0]}x{lay['squares'][1]}:{lay['square_px']}:{lay['marker_px']}")

    def _board_image(self, key: str, lay: dict[str, Any]) -> None:
        """The image of the board laid out (screen_board.render) and its PNG, made in a worker (a whole monitor's
        area takes some hundred ms, which the window never waits for), then sent to the page as a board event
        {action: image, key, png (a data URL)}. A layout wanted while one is made follows it; an image that is no
        longer the board laid out is not sent."""
        if "board_image" in self._busy:
            self._board_want = (key, lay)
            return
        self._board_want = None
        lay = dict(lay)

        def work() -> tuple[Any, str]:
            import cv2
            img = screen_board.render(lay)
            ok, png = cv2.imencode(".png", img)
            if not ok:
                raise ValueError("the board's PNG was not made")
            return img, "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")

        def done(res: tuple[Any, str] | None, error: BaseException | None) -> None:
            nxt, self._board_want = self._board_want, None
            if error is not None:
                log.warning("board image not made", exc_info=error)
            elif self._lay is not None and self._lay_key(self._lay) == key:
                self._board_img = res[0]
                self._emit({"type": "board", "action": "image", "key": key, "png": res[1]})
            if nxt is not None and nxt[0] != key:
                self._board_image(*nxt)
        self._spawn("board_image", work, done)

    def _live_board(self):
        """The calibration board as live_ar wants it, and whether to hide it in the picture: the board on this
        monitor is drawn beside the live view on the same screen."""
        a = self._board()["active"]
        if not a:
            return {"square_mm": calibrate_charuco.BOARD["square"], "squares": None, "marker_mm": None,
                    "hide_board": False}
        return {"square_mm": float(a["square_mm"]), "squares": list(a["squares"]), "marker_mm": float(a["marker_mm"]),
                "hide_board": a.get("kind") == "screen"}

    def _sync_live(self):
        """The live view, if made, on the board as it is now."""
        if self._live is None:
            return
        lb = self._live_board()
        if self._live_for == "live" and self.prefs["live"]["board"] != "sheet":
            lb["hide_board"] = False
        now = self._live.settings()
        if any(now.get(k) != v for k, v in lb.items()):
            self._live.configure(**lb)

    def _live_purpose(self, purpose):
        """The live AR for the live view (the operator's layers) or for a calibration of the streaming mode (the
        calibration board, the corners only)."""
        if self._live is None:
            return
        self._live_for = purpose
        if purpose == "calibration":
            self._live.configure(board="sheet", face=False, mandible=False, axes=True)
        else:
            p = self.prefs["live"]
            self._live.configure(board=p["board"] if p["board"] in live_ar.BOARDS else "sheet", face=p["face"],
                                 mandible=p["mandible"], axes=p["axes"])
        self._sync_live()

    # ── live AR ──────────────────────────────────────────────────────────
    def live(self) -> live_ar.LiveAR:
        """The live AR, made at first use and kept for the session: its overlay thread owns the OpenGL
        context (live_ar.py). In simulation a synthetic board stands in for the phone. Made without its models
        (prepare=False): the start job loads them, never the GUI thread."""
        if self._live is None:
            p = self.prefs["live"]
            lb = self._live_board()
            self._live = live_ar.LiveAR(board=p["board"] if p["board"] in live_ar.BOARDS else "sheet",
                                        square_mm=lb["square_mm"], squares=lb["squares"], marker_mm=lb["marker_mm"],
                                        registration=p["registration"] or None,
                                        face=p["face"], mandible=p["mandible"], axes=p["axes"],
                                        opacity=self._opacity(p.get("opacity")),
                                        intrinsics=self.roots.calib_out / "intrinsics_stream.json",
                                        synthetic=bool(self.roots.simulate), colors=live_colors(),
                                        replay=self.replay, hide_board=lb["hide_board"] and p["board"] == "sheet",
                                        prepare=False)
            self._live_server = live_ar.MjpegServer(self._live)
            # the registration watched over (live_check.py: its worker is a process of its own, started with the
            # first frame of the patient board); in simulation the synthetic stream shows the face, moved by a
            # small impact from the demo registration, the face is found by its colour and corrected by the
            # optimisation (the network knows the twin's pictures, not this flat synthetic face)
            sim = bool(self.roots.simulate)
            self._check = live_check.RegistrationCheck(
                {"method": self._check_method(p.get("check_method")), "auto": bool(p.get("check_auto")),
                 **({"segmenter": "colour", "method": "opt"} if sim else {})})
            self._live.attach_check(self._check)
            if sim:
                self._live.synthetic_stream = live_check.face_stream_factory(shift_deg=3.0)
        return self._live

    @staticmethod
    def _check_method(v):
        return v if v in live_check.METHODS else live_check.SETTINGS["method"]

    def _check_mode(self):
        """How the registration is corrected: the interface of the trial under way (A: about its ball, 3 parameters;
        B: any rigid motion), any rigid motion without one."""
        try:
            row = self._open_trial(self.run())
        except Exception:  # noqa: BLE001  no trials folder yet: the general case
            row = None
        return "a1" if row is not None and row.get("holder") == "A" else "rigid"

    @pyqtSlot(result=str)
    def live_correct(self):
        """The registration corrected now from the views of the face the live view keeps (live_check.py); the outcome
        comes in the state (check.correction). Answers the state, with check_refused when it cannot start now."""
        def go():
            self.live()
            got = self._check.correct(mode=self._check_mode())
            st = self._live_state()
            if got is False:
                st["check_refused"] = True
            return st
        return self._answer(go)

    @pyqtSlot(result=str)
    def live_correct_undo(self):
        """The registration before the last correction, again."""
        def go():
            self.live()
            self._check.undo()
            return self._live_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def live_check_settings(self, payload):
        """{method, auto} of the correction, kept for the next session."""
        def go():
            given = json.loads(payload)
            p = dict(self.prefs["live"])
            if "method" in given:
                if given["method"] not in live_check.METHODS:
                    return {"error": "bad_value", "message": "method"}
                p["check_method"] = given["method"]
            if "auto" in given:
                p["check_auto"] = bool(given["auto"])
            self.prefs.set("live", p)
            self.live()
            self._check.configure(method=self._check_method(p.get("check_method")), auto=bool(p.get("check_auto")))
            return self._live_state()
        return self._answer(go)

    def _free_camera(self) -> dict[str, Any] | None:
        """The phone's camera serves one at a time: a photo or a video stops the live view first (the
        acquisition then brings the camera app back). An error while the live view is still starting, or while a
        film for a calibration records (the same key would stop it)."""
        if "live" in self._busy:
            return {"error": "live_starting"}
        if self._film is not None:
            return {"error": "busy"}
        if self._live is not None and self._live.running:
            self._live.stop()
            self._emit(self._live_event("stopped", reason="photo"))
        elif self._live is not None and self._live.status().get("stream_state") == live_ar.FAILED:
            self._live.stop()           # a stream that gave up still holds its tunnel: let it go before the photo
        return None

    def _live_state(self):
        self.live()
        self._sync_live()       # the board changed here, on the Calibration module, or its area moved
        ar = self._live
        ln = self.link.status()
        if self._check is not None:     # the state drawn in the picture too while the phone's own screen shows it
            self._check.configure(badge=self._view is not None and (self._view.stats() or {}).get("viewers", 0) > 0)
        st = ar.status()
        # the stream's own code goes as live_error: an "error" at the top of an answer says the call itself failed
        st["live_error"] = st.pop("error", "") or ""
        return {**st, "url": self._live_server.url, "prefs": self.prefs["live"], "phone": self.phone_status,
                "busy": sorted(self._busy), "simulate": bool(self.roots.simulate),
                "intrinsics_file": str(self.roots.calib_out / "intrinsics_stream.json"),
                "phone_link": {**{k: ln[k] for k in ("state", "cause", "rtt_ms", "model")}, "mode": ln["via"]},
                "phone_view": self._view.stats() if self._view is not None else None,
                "board_ui": self._board_state()}

    @pyqtSlot(result=str)
    def live_state(self):
        return self._answer(self._live_state)

    @pyqtSlot(result=str)
    def live_start(self):
        """Closes the camera app and opens the stream in the background; 'live' events say how it went."""
        def go():
            ar = self.live()
            self._live_purpose("live")
            if ar.running:
                return self._live_state()
            if "phone" in self._busy or self.cur.get("recording") or self._film is not None:
                return {"error": "busy"}
            if not self._spawn("live", ar.start, self._live_started("live")):
                return {"error": "live_starting"}
            return self._live_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def live_stop(self):
        def go():
            if self._live is not None:
                self._live.stop()
            return self._live_state()
        return self._answer(go)

    @staticmethod
    def _opacity(v):
        """The opacity of the virtual elements, 0 to 1 (live_ar's default when missing or not a number)."""
        try:
            return min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            return live_ar.OPACITY

    @pyqtSlot(str, result=str)
    def live_settings(self, payload):
        """{board, face, mandible, axes, opacity}, kept for the next session too."""
        def go():
            given = {k: v for k, v in json.loads(payload).items()
                     if k in ("board", "face", "mandible", "axes", "opacity")}
            p = {**self.prefs["live"], **given}
            if p["board"] not in live_ar.BOARDS:
                return {"error": "bad_value", "message": "board"}
            for k in ("face", "mandible", "axes"):
                p[k] = bool(p[k])
            p["opacity"] = self._opacity(p.get("opacity"))
            self.prefs.set("live", p)
            self._live_for = "live"
            self.live().configure(board=p["board"], face=p["face"], mandible=p["mandible"], axes=p["axes"],
                                  opacity=p["opacity"])
            return self._live_state()
        return self._answer(go)

    @pyqtSlot(float, result=str)
    def live_opacity(self, value):
        """The opacity while its slider moves: the live view takes it from its next frame, and nothing else is read or
        written (the slider let go keeps it through live_settings). Answers {opacity}."""
        def go():
            v = self._opacity(value)
            if self._live is not None and self._live_for == "live":
                self._live.configure(opacity=v)
            return {"opacity": v}
        return self._answer(go)

    @pyqtSlot(result=str)
    def live_registration_load(self):
        """A registration file (JSON with T_M_B, as ground_truth.py writes it) in place of the demo."""
        def go():
            last = self.prefs["live"]["registration"]
            start = str(Path(last).parent) if last else str(self.roots.processed)
            path, _ = QFileDialog.getOpenFileName(None, "ImpactGuide", start, "registration (*.json)")
            if not path:
                return self._live_state()
            try:
                self.live().set_registration(path)
            except (live_ar.LiveARError, OSError, ValueError, KeyError, TypeError):
                return {"error": "bad_registration", "message": Path(path).name}
            self.prefs.set("live", {**self.prefs["live"], "registration": path})
            return self._live_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def live_registration_demo(self):
        def go():
            self.live().set_registration(None)
            self.prefs.set("live", {**self.prefs["live"], "registration": ""})
            return self._live_state()
        return self._answer(go)

    # ── the scans (scans_page.py, scan_desk.py, ui/22-scans.js) ──────────
    @pyqtSlot(str, result=str)
    def scans_state(self, payload: str) -> str:
        """The Scans page: {"trial": id or ""} -> the scanner's folder and whether it is watched, the rig
        calibration, the trial shown (the open one, else the one asked for, else the newest measured) with its scan
        steps, their state and its result, the trials with scans, the runs without an impact."""
        return self._answer(lambda: self.scans.state(json.loads(payload or "{}")))

    @pyqtSlot(str, str, int, result=str)
    def scans_capture(self, kind: str, action: str, want: int) -> str:
        """A run of scans without an impact (kind calibration or repeat): start (of want scans) or stop."""
        return self._answer(lambda: self.scans.capture_cmd(kind, action, want or None))

    @pyqtSlot(result=str)
    def scans_calibrate(self) -> str:
        """The spheres' radii calibrated from the calibration run's scans, in a worker (a `scans` event)."""
        return self._answer(self.scans.calibrate)

    @pyqtSlot(str, result=str)
    def scans_process(self, payload: str) -> str:
        """The scans of a day's trials ({"day": "YYYY-MM-DD" or "", "all": bool}) measured and written into
        gt_poses.csv, in a worker (a `scans` event)."""
        def go() -> dict[str, Any]:
            a = json.loads(payload or "{}")
            return self.scans.process(str(a.get("day") or ""), bool(a.get("all")))
        return self._answer(go)

    @pyqtSlot(result=str)
    def scans_simulate(self) -> str:
        """In simulation: the stand-in scanner exports the scan the open trial waits for."""
        return self._answer(self.scans.simulate)

    @pyqtSlot(result=str)
    def scans_open(self) -> str:
        """The folder of the measured scans opened in Explorer."""
        def go() -> dict[str, Any]:
            folder = self.scans.store() / self.run().phase
            folder.mkdir(parents=True, exist_ok=True)
            open_externally(folder)
            return {"ok": True}
        return self._answer(go)

    # ── the dataset (dataset_page.py, dataset_pack.py, ui/28-dataset.js) ─
    @pyqtSlot(str, result=str)
    def dataset_state(self, payload: str) -> str:
        """The Dataset page: {"fresh": bool} -> what the dataset holds (None while it is read, in a worker: a
        `dataset` event once read), the check or export on its way, the last check and package, what goes in."""
        return self._answer(lambda: self.dataset.state(json.loads(payload or "{}")))

    @pyqtSlot(str, result=str)
    def dataset_parts(self, payload: str) -> str:
        """What goes into the package and the check: {"real": bool, "synthetic": bool}, kept for the next start."""
        return self._answer(lambda: self.dataset.set_parts(json.loads(payload or "{}")))

    @pyqtSlot(result=str)
    def dataset_check(self) -> str:
        """Every file of the dataset read and compared, in a worker (a `dataset` event, checked, at the end)."""
        return self._answer(self.dataset.check)

    @pyqtSlot(str, result=str)
    def dataset_export(self, payload: str) -> str:
        """The package written where the operator chooses: {"to": "folder"} a folder to put it in, {"to": "zip"}
        the zip file; in a worker (a `dataset` event, exported, at the end). {"cancelled": true} when nothing was
        chosen."""
        def go() -> dict[str, Any]:
            as_zip = json.loads(payload or "{}").get("to") == "zip"
            start = self.dataset.start_folder()
            if as_zip:
                name = dataset_page.dataset_pack.package_name() + ".zip"
                chosen, _ = QFileDialog.getSaveFileName(None, "ImpactGuide", str(start / name), "zip (*.zip)")
            else:
                chosen = QFileDialog.getExistingDirectory(None, "ImpactGuide", str(start))
            if not chosen:
                return {"cancelled": True}
            dest = self.dataset.target(chosen, as_zip)
            return self.dataset.export(dest, as_zip, f"{APP_NAME} {APP_VERSION}")
        return self._answer(go)

    @pyqtSlot(result=str)
    def dataset_stop(self) -> str:
        """The check or export on its way stopped (its partial package removed)."""
        return self._answer(lambda: self.dataset.stop())

    @pyqtSlot(result=str)
    def dataset_open(self) -> str:
        """The last package's folder (or the zip's) opened in Explorer."""
        def go() -> dict[str, Any]:
            folder = self.dataset.opened()
            if folder is None:
                raise UserError("not_found")
            open_externally(folder)
            return {"ok": True}
        return self._answer(go)

    # ── the twin of the bench (twin.py, ui/18-twin.js) ───────────────────
    @guarded
    def _twin_poll(self) -> None:
        """The bench scene looked at: once it rests and differs from the model's scene, the model is made again in a
        worker (Blender in the background, below the app's priority), never while the live AR runs."""
        if self._live is not None and getattr(self._live, "running", False):
            return
        if self.twin.due():
            self._twin_export()

    def _twin_export(self) -> bool:
        def done(cur: dict | None, error: BaseException | None) -> None:
            if isinstance(error, twin.Busy):
                return                  # saved meanwhile: tried again once it rests
            if error is not None:
                log.warning("twin model not made", exc_info=(type(error), error, error.__traceback__))
                self._emit({"type": "twin", "action": "failed"})
                return
            new = cur.get("version") != self._twin_said
            self._twin_said = cur.get("version")
            v = twin.versions(cur.get("phone_print", ""))
            self._emit({"type": "twin", "action": "updated" if new else "same", "version": cur.get("version"),
                        **{k: v[k] for k in ("stale", "dataset", "network")}})
        started = self._spawn("twin_export", self.twin.run, done)
        if started:
            self._emit({"type": "twin", "action": "exporting"})
        return started

    def _twin_model(self) -> dict[str, Any]:
        cur = self.twin.current()
        busy = self.twin.running or "twin_export" in self._busy
        if cur is None:
            return {"state": "exporting" if busy else ("none" if self.twin.can_export() else "no_blender")}
        return {"state": "exporting" if busy else "ready", "url": file_url(self.twin.cache / cur["model"]),
                "scene": Path(str(cur.get("scene", ""))).name,
                **{k: cur.get(k) for k in ("version", "hash", "phone_print", "scene_saved", "exported", "export_s",
                                           "bytes", "triangles")}}

    def _twin_trial(self, run: trials.Run, mode: str, want: str) -> tuple[dict[str, str] | None, list, int]:
        """The trial the twin shows, its steps and the step it is at: in Trials the open one at its step (else the
        next, planned, at its first); in Scans and results the one asked for, else the newest with its scans, at
        its first scan."""
        rows = run.log.rows()
        if mode == "results":
            if not want:
                done = twin.scanned_trials(run)
                want = done[0]["trial"] if done else (rows[-1]["trial"] if rows else "")
            row = next((r for r in rows if r["trial"] == want), None)
            steps = run.steps(row["trial"]) if row is not None else []
            first = next((i for i, st in enumerate(steps) if st.key.startswith("scan_")), 0)
            return row, steps, first
        row = self._open_trial(run)
        if row is not None:
            steps = run.steps(row["trial"])
            return row, steps, min(self.cur.get("step", 0), len(steps) - 1)
        plan = self._plan(run)
        if not plan:
            return None, [], 0
        row = {k: plan.get(k, "") for k in trials.LOG_FIELDS}
        return row, trials.trial_steps(run.config, row, rows[-1] if rows else None), 0

    @pyqtSlot(str, result=str)
    def twin_state(self, payload: str) -> str:
        """The twin for the page: {"mode": "trials" | "results", "trial": id or ""} -> the model (url, version...),
        whether the dataset and the network were made on this scene, the phone's field of view, the trial shown
        (its setting, its steps and the one it is at, the arm's positions) and, after its scans, the marker's motion with the TRE at each target; in Scans and results
        the trials with scans as well. The scans are measured in a worker: "measuring", then a `twin` event."""
        def go() -> dict[str, Any]:
            ask_ = json.loads(payload or "{}")
            mode = "results" if ask_.get("mode") == "results" else "trials"
            run = self.run()
            cur = self.twin.current()
            out = {"model": self._twin_model(), "versions": twin.versions(cur.get("phone_print", "") if cur else ""),
                   "phone": twin.phone_view(self.roots.calib_out), "mode": mode, "trial": None, "scan": None}
            if mode == "results":
                out["trials"] = twin.scanned_trials(run)
            row, steps, at = self._twin_trial(run, mode, str(ask_.get("trial") or ""))
            if row is None:
                return out
            cfg = run.config_for(row)       # a logged trial's own settings; the plan's (no version yet) today's
            out["trial"] = twin.setting(row, steps[at].key if steps else None, cfg.photo_az)
            # its steps, for the twin to put the bench as each has it; the arm's positions and its sweep
            out["steps"], out["step"] = [st.key for st in steps], at
            out["arm"] = {"photo_az": list(cfg.photo_az), "scan_az": list(cfg.scan_az), "video_az": list(cfg.video_az),
                          "video_deg_s": trials.VIDEO_SPEED}
            info = self.twin.info(cur)
            sim = bool(self.roots.simulate)
            res = twin.scan_result(run, row, info, self.twin.cache, simulate=sim, measure=False)
            if res.get("state") == "measuring":
                trial = row["trial"]

                def done(r: dict | None, error: BaseException | None) -> None:
                    if error is not None:
                        log.warning("scans of %s not measured", trial, exc_info=(type(error), error,
                                                                                 error.__traceback__))
                    self._emit({"type": "twin", "action": "scan", "trial": trial,
                                "state": (r or {}).get("state", "failed")})
                self._spawn("twin_scan", lambda: twin.scan_result(run, row, info, self.twin.cache, simulate=sim), done)
            out["scan"] = res
            return out
        return self._answer(go)

    @pyqtSlot(result=str)
    def twin_update(self) -> str:
        """The model made again now, from the scene as it is saved (Advanced): a `twin` event when it is ready."""
        def go() -> dict[str, Any]:
            if not self.twin.can_export():
                return {"error": "twin_no_blender"}
            self._twin_export()
            return {"model": self._twin_model()}
        return self._answer(go)

    # ── analysis and charts (analysis.py, chart_builder.py, ui/24-analysis.js, ui/26-charts.js) ──
    def _results_dirs(self) -> dict[str, Path | None]:
        """Where the analysis writes its tables and the charts their scripts and figures: the project's results,
        figures and pictures folders; all under the simulation's folder in simulation, so that simulated tables
        never reach the project's. charts: where an export goes without asking (simulation only)."""
        if self.roots.simulate:
            b = self.roots.simulate
            return {"results": b / "results", "figures": b / "figures", "pictures": b / "pictures",
                    "charts": b / "charts", "interfaces": self.roots.raw / "interfaces"}
        return {"results": trials.ROOT / "05_results", "figures": APP.parent / "figures",
                "pictures": trials.ROOT / "06_pictures", "charts": None, "interfaces": self.roots.raw / "interfaces"}

    @pyqtSlot(result=str)
    def analysis_state(self) -> str:
        """The last analysis (analysis.summary) or None, whether one runs and how far (trials done, of)."""
        def go() -> dict[str, Any]:
            return {"running": "analysis" in self._busy, "result": getattr(self, "_analysis", None),
                    "progress": getattr(self, "_analysis_at", None), "simulate": bool(self.roots.simulate),
                    "phase": self.prefs["phase"]}
        return self._answer(go)

    @pyqtSlot(result=str)
    def analysis_run(self) -> str:
        """The analysis of the phase in use made again in a worker (the scans of a trial measured once, then kept):
        its tables written into the results folder (only the changed ones), then an `analysis` event, ready or
        failed (no_trials: no closed trial with its scans yet)."""
        def go() -> dict[str, Any]:
            run = self.run()
            rows, pending = run.log.rows(), run.pending()
            info, cache, d, sim = self.twin.info(), self.twin.cache, self._results_dirs(), bool(self.roots.simulate)
            store = self.scans.store()          # the reseating run of holder B (scan_desk, kind reseat)
            self._analysis_at = None

            def progress(i: int, n: int) -> None:
                self._analysis_at = [i, n]      # read by analysis_state; a plain pair, replaced whole

            def work() -> dict:
                import analysis
                return analysis.analyse(run, info, cache, d["results"], simulate=sim, rows=rows, pending=pending,
                                        progress=progress, interfaces_raw=d["interfaces"],
                                        interfaces_store=store)

            def done(res: dict | None, error: BaseException | None) -> None:
                self._analysis_at = None
                if error is not None:
                    self._emit({"type": "analysis", "action": "failed", **self._err(error, "analysis_failed")})
                    return
                self._analysis = res
                self._emit({"type": "analysis", "action": "ready", "n": res["n"], "written": res["written"]})
            return {"started": self._spawn("analysis", work, done), "running": True}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def results_open(self, what: str) -> str:
        """A folder of the analysis opened in Explorer: results, figures, pictures or charts (the exports)."""
        def go() -> dict[str, Any]:
            folder = self._results_dirs().get(what) if what in ("results", "figures", "pictures", "charts") else None
            if folder is None:
                return {"error": "not_found"}
            folder.mkdir(parents=True, exist_ok=True)
            open_externally(folder)
            return {"ok": True}
        return self._answer(go)

    @pyqtSlot(result=str)
    def chart_state(self) -> str:
        """What the Charts page offers: the charts to start from, the pinned ones (their scripts in the figures
        folder), the tables of the results folder with their columns, the kinds, palettes, scales, legends and
        page widths, and whether the data are simulated."""
        def go() -> dict[str, Any]:
            import chart_builder as cb
            d = self._results_dirs()
            return {"presets": cb.presets(), "saved": cb.saved(d["figures"]), "tables": cb.tables(d["results"]),
                    "kinds": list(cb.KINDS), "palettes": list(cb.PALETTES), "scales": list(cb.SCALES),
                    "legends": list(cb.LEGENDS), "widths": list(cb.WIDTHS_MM), "heights": list(cb.HEIGHT_MM),
                    "default_height": {str(k): v for k, v in cb.DEFAULT_HEIGHT.items()},
                    "simulate": bool(self.roots.simulate), "analysis": getattr(self, "_analysis", None) is not None,
                    "analysing": "analysis" in self._busy}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def chart_name(self, slug: str) -> str:
        """A free figure name for a new chart: figS<n>_<slug>."""
        def go() -> dict[str, Any]:
            import chart_builder as cb
            d = self._results_dirs()
            return {"name": cb.next_name(d["figures"], d["pictures"], str(slug or "chart")[:40])}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def chart_preview(self, payload: str) -> str:
        """{"spec", "id"} drawn in a worker, the newest one only when several come while one draws: a `chart`
        event, preview (svg, the checked spec) or failed, with the id it was asked with."""
        def go() -> dict[str, Any]:
            ask_ = json.loads(payload or "{}")
            self._chart_want = (ask_.get("spec") or {}, ask_.get("id"))
            self._chart_next()
            return {"queued": True}
        return self._answer(go)

    def _chart_next(self) -> None:
        want = getattr(self, "_chart_want", None)
        if want is None or "chart" in self._busy:
            return
        self._chart_want = None
        spec, tag = want
        d, sim = self._results_dirs(), bool(self.roots.simulate)

        def work() -> dict:
            import chart_builder as cb
            r = cb.preview(spec, d["results"], simulated=sim)
            r.pop("source", None)
            return r

        def done(r: dict | None, error: BaseException | None) -> None:
            if error is not None:
                self._emit({"type": "chart", "action": "failed", "id": tag, **self._err(error, "chart_failed")})
            else:
                self._emit({"type": "chart", "action": "preview", "id": tag, **r})
            self._chart_next()
        self._spawn("chart", work, done)

    @pyqtSlot(str, result=str)
    def chart_export(self, payload: str) -> str:
        """The chart as PDF, SVG and PNG at 300 dpi, under a name the operator chooses (into the simulation's charts
        folder in simulation): a `chart` event, exported (the files) or failed."""
        def go() -> dict[str, Any]:
            spec = json.loads(payload or "{}").get("spec") or {}
            d, sim = self._results_dirs(), bool(self.roots.simulate)
            name = str(spec.get("name") or "chart")
            if d["charts"] is not None:
                stem = d["charts"] / name
            else:
                start = str(Path.home() / "Documents" / f"{name}.pdf")
                path, _ = QFileDialog.getSaveFileName(None, "ImpactGuide", start, "PDF, SVG, PNG (*.pdf)")
                if not path:
                    return {"cancelled": True}
                stem = Path(path).with_suffix("")

            def work() -> list:
                import chart_builder as cb
                return [p.name for p in cb.export(spec, d["results"], stem, simulated=sim)]

            def done(files: list | None, error: BaseException | None) -> None:
                if error is not None:
                    self._emit({"type": "chart", "action": "failed", **self._err(error, "chart_failed")})
                else:
                    self._emit({"type": "chart", "action": "exported", "files": files})
            return {"started": self._spawn("chart_export", work, done)}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def chart_pin(self, payload: str) -> str:
        """{"spec", "replace"}: the chart's script written into the figures folder as <name>.py and run there on
        its own, which writes the figure (PDF and PNG) into the pictures folder. A script of that name is replaced
        only with replace (else the answer is script_exists, at once). Then a `chart` event, pinned or failed."""
        def go() -> dict[str, Any]:
            import chart_builder as cb
            ask_ = json.loads(payload or "{}")
            spec, replace = ask_.get("spec") or {}, bool(ask_.get("replace"))
            d, sim = self._results_dirs(), bool(self.roots.simulate)
            name = str(spec.get("name") or "")
            if name and not replace and (d["figures"] / f"{name}.py").exists():
                return {"error": "script_exists", "args": {"name": name}}

            def work() -> dict:
                return cb.pin(spec, d["results"], d["figures"], d["pictures"], simulated=sim, replace=replace)

            def done(r: dict | None, error: BaseException | None) -> None:
                if error is not None:
                    self._emit({"type": "chart", "action": "failed", **self._err(error, "chart_failed")})
                    return
                self._emit({"type": "chart", "action": "pinned", "name": r["name"], "replaced": r["replaced"],
                            "script": Path(r["script"]).name, "files": [Path(f).name for f in r["files"]]})
            return {"started": self._spawn("chart_pin", work, done)}
        return self._answer(go)

    # ── the holders measured on their own (interfaces.py, ui/28-holders.js) ──
    def _holders(self) -> dict[str, Any]:
        import interfaces
        raw, store = self.roots.raw / "interfaces", self.scans.store()
        run = self.scans.capture_state().get("reseat") or {}
        return {"a": interfaces.a_state(raw), "b": {**interfaces.b_state(store), "scans": run.get("n", 0),
                                                    "good": run.get("good", 0), "want": run.get("want"),
                                                    "files": run.get("files", [])[-6:]},
                "b_on": self.scans.capture == "reseat", "measuring": any(w.startswith("capture:")
                                                                         for w in self.scans.busy()),
                "scanner": bool(self.scanner_folder()), "simulate": bool(self.roots.simulate),
                "reseats": interfaces.RESEATS}

    @pyqtSlot(result=str)
    def holders_state(self) -> str:
        """Holder A's slip torque as it stands (the repeat in progress, the masses tried, the repeats done) and holder
        B's reseating (the run of scans, each seating's error against the first)."""
        return self._answer(self._holders)

    @pyqtSlot(str, result=str)
    def holders_a(self, payload: str) -> str:
        """{"mass_g", "arm_mm", "slipped"}: one mass tried on A, written at once into a_slip_torque.csv."""
        def go() -> dict[str, Any]:
            import interfaces
            ask_ = json.loads(payload or "{}")
            try:
                mass = float(str(ask_.get("mass_g", "")).replace(",", "."))
                arm = float(str(ask_.get("arm_mm", "")).replace(",", "."))
                interfaces.record_a(self.roots.raw / "interfaces", mass, arm, bool(ask_.get("slipped")))
            except ValueError:
                return {"error": "bad_value", "args": {"field": "mass_g"}}
            return self._holders()
        return self._answer(go)

    @pyqtSlot(result=str)
    def holders_a_undo(self) -> str:
        """The last mass tried on A taken back."""
        def go() -> dict[str, Any]:
            import interfaces
            interfaces.undo_a(self.roots.raw / "interfaces")
            return self._holders()
        return self._answer(go)

    @pyqtSlot(str, int, result=str)
    def holders_b(self, action: str, want: int) -> str:
        """The reseating run of B started or stopped: the Scans page's run of kind reseat (want scans, the first
        seating and the lifts after it; 0: interfaces.RESEATS + 1). Each export is measured as it comes."""
        def go() -> dict[str, Any]:
            import interfaces
            if action not in ("start", "stop"):
                return {"error": "bad_value", "args": {"field": "action"}}
            if action == "start" and not self.scanner_folder():
                return {"error": "no_scanner_folder"}
            self.scans.capture_cmd("reseat", action, int(want) if want else interfaces.RESEATS + 1)
            return self._holders()
        return self._answer(go)

    # ── the window is closing ────────────────────────────────────────────
    @guarded
    def shutdown(self) -> None:
        """Everything this backend started or connected let go: timers, Qt connections, its listener on the link
        (a process singleton), the servers and the live AR. Each part is let go even when one before it fails."""
        self._pump.stop()
        self._scan_timer.stop()
        self._twin_timer.stop()
        self.twin.cancel()              # a model being made: its Blender stopped
        self.dataset.shutdown()         # a check or an export: stopped, its partial package removed
        self._check_queue.clear()
        self._frames.close()            # the phone's page stops waiting for frames
        for signal, slot in self._connections:
            with contextlib.suppress(TypeError, RuntimeError):
                signal.disconnect(slot)
        self._connections.clear()
        # the link (a process singleton in real use) forgets this backend's listener and removes its adb tunnels
        self._watching = False
        steps = [("photo taken by itself", self._auto_sync),
                 ("staged files", self._drop_staged), ("link listener", lambda: self.link.unwatch(self._on_link)),
                 ("link", self.link.close)]
        if self._view is not None:
            steps.append(("phone view", self._view.close))
        if self._live_server is not None:
            steps.append(("live server", self._live_server.close))
        if self._live is not None:
            steps.append(("live AR", self._live.shutdown))
        for what, step in steps:
            try:
                step()
            except Exception:
                log.warning("shutdown: %s not closed", what, exc_info=True)


class LiveFrames:
    """The phone's screen reads the live AR's frames through this, so the server can be up before the live AR is
    made (and while it is off). Once closed (the app closing, or the live AR shut down) a reader is answered at once
    and the long poll of the phone's page ends."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._closed = threading.Event()

    @property
    def closed(self) -> bool:
        return self._closed.is_set() or bool(getattr(self.backend._live, "closed", False))

    def close(self) -> None:
        """No more frames: every reader waiting returns now."""
        self._closed.set()

    def next_frame(self, after: int = 0, timeout: float = 1.0) -> tuple[bytes | None, int, float | None]:
        """(JPEG, counter, time composed) of the next frame after `after`, or (None, after, None) within timeout."""
        if self.closed:
            return None, after, None
        ar = self.backend._live
        if ar is None:
            self._closed.wait(timeout)
            return None, after, None
        return ar.next_frame(after, timeout)


def view_words(lang: str) -> dict[str, str]:
    """The words of the page on the phone's screen, in the app's language (English where one is missing)."""
    words = {}
    for code in ("en", lang):
        try:
            d = json.loads((APP / "ui" / "strings" / f"{code}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        words.update({k[len("view.page."):]: v for k, v in d.items() if k.startswith("view.page.")})
    return words


# ── the window ───────────────────────────────────────────────────────────────
def role(name):
    return PALETTES[PALETTE][name]


def live_colors():
    """The colours the live overlay is drawn in, from the palette's roles."""
    return {"face": role("accent"), "mandible": role("text-1"), "corners": role("ok-text"),
            "reprojected": role("text-1")}


PAGE_PREFIX = "page-"       # the page of each running app in a folder of its own: page-<process id>


def prepare_page(base: Path | str) -> Path:
    """The page as the window loads it: ui.html and ui/ copied to a folder of this process under `base`
    (page-<pid>), with SlantUI's stylesheets and scripts written beside them. Nothing is generated next to the
    sources, which may sit on a synced drive. The folders of apps no longer running are removed first (best effort);
    the one of an app still open is never touched, so a second start cannot trip on its locked files.

    Args:
        base: the folder holding the pages (the app's local folder, or the --simulate folder).

    Returns:
        The ui.html to load.
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    for old in base.iterdir():
        if not old.is_dir():
            continue
        if old.name == "page":          # the single folder of older versions
            shutil.rmtree(old, ignore_errors=True)
        elif old.name.startswith(PAGE_PREFIX) and old.name[len(PAGE_PREFIX):].isdigit():
            pid = int(old.name[len(PAGE_PREFIX):])
            if pid != os.getpid() and not process_alive(pid):
                shutil.rmtree(old, ignore_errors=True)
    dest = base / f"{PAGE_PREFIX}{os.getpid()}"
    shutil.rmtree(dest, ignore_errors=True)      # left by a process of the same id long gone
    ignore = shutil.ignore_patterns("desktop.ini", "__pycache__")
    shutil.copytree(APP / "ui", dest / "ui", ignore=ignore, dirs_exist_ok=True)
    shutil.copy2(APP / "ui.html", dest / "ui.html")
    (dest / "ui" / "slantui.css").write_text(slantui_css.bundle(), encoding="utf-8")
    (dest / "ui" / "slantui.js").write_text(slantui_js.bundle(), encoding="utf-8")
    return dest / "ui.html"


def mark_file() -> Path:
    """The mark: the band (before and after the page) and the loading screen, at SlantUI's sizes."""
    return MARK


def _mark_pixmap(size, source=None):
    if USE_QT6:
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QPainter, QPixmap
        from PyQt6.QtSvg import QSvgRenderer
    else:
        from PyQt5.QtCore import QRectF
        from PyQt5.QtGui import QPainter, QPixmap
        from PyQt5.QtSvg import QSvgRenderer
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    r = QSvgRenderer(str(source or ICON))
    if r.isValid():
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r.render(p, QRectF(0, 0, size, size))
        p.end()
    return pm


def app_icon():
    if USE_QT6:
        from PyQt6.QtGui import QIcon
    else:
        from PyQt5.QtGui import QIcon
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(_mark_pixmap(size, ICON))
    return icon


def tell_launcher_up() -> None:
    """Raises the event the desktop launcher waits on: the page is on screen."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.OpenEventW.restype = ctypes.c_void_p
        k32.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        k32.SetEvent.argtypes = [ctypes.c_void_p]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = k32.OpenEventW(EVENT_MODIFY_STATE, False, LAUNCHER_EVENT)
        if handle:
            k32.SetEvent(handle)
            k32.CloseHandle(handle)
    except (OSError, AttributeError):
        log.info("launcher event not raised", exc_info=True)    # started by hand: no launcher waits


def splash_words(lang: str) -> tuple[str, str]:
    """The two words of the loading screen in the app's language."""
    try:
        words = json.loads((APP / "ui" / "strings" / f"{lang}.json").read_text(encoding="utf-8"))
        return words.get("win.splash.starting", "starting"), words.get("win.splash.ready", "ready")
    except (OSError, ValueError):
        return "starting", "ready"


def build(args: argparse.Namespace) -> tuple[Any, Any, Backend]:
    """Application, backend and window, shown; returns (app, window, backend). Uncaught exceptions are logged from
    here on (install_excepthooks), into the errors.log of these roots."""
    roots = Roots(args.simulate)
    setup_log(roots.log)
    install_excepthooks()
    prefs = Prefs(roots.prefs)
    app = Application(APP_NAME, app_id=APP_ID)
    app.setWindowIcon(app_icon())
    backend = Backend(roots, prefs, lang=args.lang, replay=args.replay)
    if args.page in PAGES:
        backend.start_page = args.page
    page = prepare_page(roots.local)
    title = APP_NAME + (" (simulation)" if roots.simulate else "")
    # the band the window draws before the page has one shows the mark as a button, as the page
    # always does (ui/01-shell.js, showHome)
    win = Window(page, bridge=backend, title=title, palette=PALETTE, size=WINDOW_SIZE, min_size=MIN_WINDOW,
                 logo=mark_file(), brand_action=True)
    backend.window = win
    win.setIcon(app_icon())
    app.aboutToQuit.connect(backend.shutdown)
    starting, ready = splash_words(backend.lang)
    splash = Splash(window=win, logo=mark_file(), text=starting, busy=True)
    splash.show()

    def page_is_up() -> None:
        splash.set_progress(1.0, ready)
        QTimer.singleShot(SPLASH_HIDE_MS, guarded(splash.hide))
        tell_launcher_up()

    backend.on_page_ready = page_is_up
    QTimer.singleShot(SPLASH_MAX_MS, guarded(lambda: splash.hide() if splash.visible else None))
    win.show()
    return app, win, backend


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--simulate", type=Path, help="keep every file under this folder, with a stand-in phone and scanner")
    p.add_argument("--lang", choices=LANGS)
    p.add_argument("--page", choices=PAGES)
    p.add_argument("--replay", type=Path, help="the live page plays these recorded frames (a video or a folder of "
                                               "images) instead of the phone's stream")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """The app: the window until it is closed; the exit code."""
    app, win, backend = build(parse(argv))
    return app.run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        tb = traceback.format_exc()
        try:
            LOCAL.mkdir(parents=True, exist_ok=True)
            (LOCAL / "startup_error.log").write_text(tb, encoding="utf-8")
        except OSError:
            tb += "\n(startup_error.log could not be written)\n"
        if sys.stderr is not None:      # pythonw has no console
            sys.stderr.write(tb)
        raise
