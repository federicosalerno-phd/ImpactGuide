"""Raw files during a trial: the phone over ADB, by cable or Wi-Fi, and the scanner's export folder.

Phone. The camera app stays open on the phone in its cradle and the phone is
never touched: a key event sent over ADB is the shutter (the volume key, set
as the shutter in the camera app), the new file in the camera folder is waited
for until its size and time stay the same for PHONE_SETTLE_S and the media
store no longer marks it pending, and it is pulled into a staging folder with
the name the phone gave it. The copy lands as a hidden .part file beside its
final name and gets that name only once it has the size and the SHA-256 of the
file on the phone; only then is the file deleted on the phone and taken out of
Android's media store, so the gallery does not show it (pull, remove). A copy
that fails or differs is deleted on the PC instead, the file stays on the phone
and the cause is copy_failed. ImpactGuide then files it under the trial name
(trials.Run.store) or discards it, so nothing reaches 03_data/raw before the
operator keeps it. For the video the same key starts and stops the recording;
with switch_modes the camera app is first moved to its video mode and back to
its photo mode afterwards through the standard Android camera intents.
Which phone, on which link (cable or Wi-Fi), is phone_link.py's to say: every
operation asks it for the serial first, so a Wi-Fi link that came back on a
new port is followed, and a link that is down is said as such (wifi_lost).

No countdown. The camera app of HyperOS has a timer of its own (3, 5 or 10 s,
or the volume key set to a timer) kept in its settings, which adb cannot read
or change. Its countdown takes the seconds from the intent that started the
camera first (android.intent.extra.TIMER_DURATION_SECONDS, the extra of the
voice assistant) and only then from its timer, and drops the extra after one
photo (com.android.camera 6.0, getCountDownTimes). So before every photo the
camera gets that extra at 0 (arm: an intent to the running camera, which it
takes as a new intent), whatever its timer says: the shutter is immediate.
The intent is sent right after each photo, ready for the next one; the camera
takes READY_S to be back from it. Each photo's delay after the press (the
phone's clock against the time in the file's name) is kept in last_shot; a
photo that still came COUNTDOWN_S or more after the press says that the camera
counted down all the same (countdown_seen), for the app to say.

The camera's own mode kept (take with keep_mode, the photos of a calibration
and of the trials, taken in its Pro mode with the focus set by hand, so that
the trials' photos are taken as the calibration's): no intent at all goes to the
camera app then, neither the timer's (the camera takes it as a new start, which
may bring it back to its Photo mode) nor the photo mode's; brought back on
screen it is started as its icon starts it. Its own timer must then be off.

A photo is taken (take: the shutter, the new file waited for until whole) and
copied (pull) apart, so that the copy of one may run while the next is taken
(copier: the same phone with a lock of its own); shoot does both.

Every adb call is an argument list with a timeout (scaled with the file's size
for a hash or a copy); what goes to the phone's shell is quoted, and the camera
folder and the file names are checked first. adb's own text goes to the log,
never into an error's detail. One operation at a time on a Phone.

The camera serves one at a time: the live view of ImpactGuide (live_ar.py)
closes the camera app before it opens the stream, and ImpactGuide stops the
live view before a photo or a video; the camera app is then started again
here when another app is on screen.

Scanner. The scanner software exports into a folder; ExportWatch reports the
first scan file that appears there (or is written again) after it started,
once its size and time have stayed the same for SCAN_SETTLE_S and no program
holds it open for writing.

FakePhone and fake_scan write stand-in files, for the tests and for the
simulation mode of ImpactGuide; FakeAdb stands in for adb and the phone's camera
folder, for the test of Phone itself.

Usage:
    python acquisition.py devices                 phones seen over ADB
    python acquisition.py shoot [--out DIR]       one photo from the phone into DIR
    python acquisition.py test                    fake phone, fake adb and export folder; exit 1 on failure
    python acquisition.py clear-check             on the phone itself: a test file put in the camera folder,
                                                  pulled, deleted there and out of the media store; the
                                                  folder and the store as they were (no photo is taken)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import phone_link

_LOGGER = logging.getLogger("acquisition")

PHOTO_SUFFIXES = (".jpg", ".jpeg")
VIDEO_SUFFIXES = (".mp4",)
SCAN_SUFFIXES = (".ply", ".stl", ".obj", ".asc")
KEYCODES = {"volume_down": 25, "volume_up": 24, "camera": 27}
CAMERA_DIR = "/sdcard/DCIM/Camera"
CAMERA_APP = "com.android.camera"  # the camera app of the Poco X7 Pro (HyperOS)
CAMERA_ACTIVITY = CAMERA_APP + "/.Camera"
TIMER_EXTRA = "android.intent.extra.TIMER_DURATION_SECONDS"   # the camera app's countdown for its next photo
OPEN_S = 2.0                       # the camera app ready after it was started
MODE_S = 1.5                       # the camera app in its other mode after the intent
READY_S = 1.0                      # the camera app back after the intent that lifts its countdown (arm)
COUNTDOWN_S = 1.6                  # a photo this long after the press was counted down by the camera app
WAKE_S = 0.6                       # the camera app back on screen after the phone was woken
POLL_S = 0.25                      # between two looks at the camera folder
PHONE_SETTLE_S = 0.6               # a new file on the phone is whole once unchanged this long (and not pending)
SCAN_SETTLE_S = 2.0                # a new scan is whole once unchanged this long (and not held open for writing)
VIDEO_START_S = 3.0                # a recording that leaves no trace in the camera folder this long is said in the log
SHOT_TIMEOUT_S = 20.0              # a photo is waited for this long
VIDEO_TIMEOUT_S = 60.0             # the video after the stop
SHELL_TIMEOUT_S = 20.0             # a command in the phone's shell
HASH_RATE_BPS = 20e6               # bytes per second the phone hashes at least (sha256sum)
PULL_TIMEOUT_S = 30.0              # adb pull: this, plus the size at...
PULL_RATE_BPS = 2e6                # ... the slowest Wi-Fi we accept
PUSH_TIMEOUT_S = 60.0
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: no console flash under pythonw
MEDIA_URI = "content://media/external/file"      # Android's media store: what the gallery lists
SAFE_NAME = re.compile(r"\w[\w.\-]*")             # file names the camera gives; nothing else is ever deleted
SAFE_DIR = re.compile(r"(?:/[\w.\-]+)+")          # a folder on the phone: absolute, no spaces or quotes
NAME_TIME = re.compile(r"(?:IMG|VID)_(\d{8})_(\d{6})")  # the time the camera writes in a file's name (phone's clock)
PHONE_TIME = re.compile(r"(\d{9,}\.\d+)([+-]\d{4})")    # date +%s.%N%z: seconds since the epoch and the zone
LOOK_SEP = "@@@@"                                       # between the answers of one call to the phone (look)
PART = ".part"                                    # a copy in progress beside its final name


def run_adb(cmd: Sequence[str], timeout: float) -> tuple[int, str, str]:
    """adb as a process: (return code, stdout, stderr)."""
    r = subprocess.run(list(cmd), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                       creationflags=NO_WINDOW, check=False)
    return r.returncode, r.stdout, r.stderr


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class AcquisitionError(RuntimeError):
    """A failure with a cause the interface can name: no_adb, no_device, unauthorized, many_devices,
    adb_failed, timeout, not_found, copy_failed (the file stayed on the phone), bad_value (a camera folder or a
    shutter key that cannot be used), busy (a recording already running), and the causes of the link
    (phone_link.CAUSES). detail is a short plain detail (a file name, a command), never adb's text."""

    def __init__(self, cause: str, detail: str = "") -> None:
        super().__init__(f"{cause}: {detail}" if detail else cause)
        self.cause = self.code = cause
        self.detail = detail


def phone_time(out: str) -> tuple[float, int] | None:
    """(seconds since the epoch, the zone's offset in s) of the phone's `date +%s.%N%z`, None when not there."""
    m = None
    for m in PHONE_TIME.finditer(out or ""):
        pass
    if m is None:
        return None
    z = m.group(2)
    return float(m.group(1)), (1 if z[0] == "+" else -1) * (int(z[1:3]) * 3600 + int(z[3:5]) * 60)


def shot_delay(name: str, pressed: tuple[float, int] | None) -> float | None:
    """Seconds from the press to the photo: the time in its name (to the second, taken at its middle) against the
    phone's clock at the press. None when either is missing or the two cannot be of the same photo."""
    m = NAME_TIME.search(name or "")
    if not m or pressed is None:
        return None
    at, offset = pressed
    try:
        named = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone(timedelta(seconds=offset))).timestamp() + 0.5
    except ValueError:
        return None
    d = named - at
    return round(d, 2) if -5.0 <= d <= 30.0 else None


def find_adb() -> str | None:
    """The adb executable of the whole app (phone_link.find_adb: IMPACTGUIDE_ADB, the one shipped with
    scrcpy, the PATH, adbutils, the Android SDK). None if there is none."""
    return phone_link.find_adb()


def new_stable_file(listing: dict, before: dict, sizes: dict, suffixes: Sequence[str], now: float | None = None,
                    settle_s: float = 0.0) -> str | None:
    """One poll of a folder listing {name: (size, mtime)}: the first file that is new or rewritten since `before`,
    has an allowed suffix, is not hidden or pending, is not empty and has had the same size and time for at least
    settle_s (at least two polls; sizes keeps what each poll saw). None if there is none yet."""
    now = time.monotonic() if now is None else now
    for name, (size, mtime) in sorted(listing.items(), key=lambda kv: kv[1][1]):
        if name.startswith(".") or "pending" in name.lower() or not name.lower().endswith(tuple(suffixes)):
            continue
        if name in before and before[name] == (size, mtime):
            continue
        seen = sizes.get(name)
        if not isinstance(seen, tuple) or seen[:2] != (size, mtime):
            sizes[name] = (size, mtime, now)
            continue
        if size > 0 and now - seen[2] >= settle_s:
            return name
    return None


def free_to_read(path: Path | str) -> bool:
    """No program holds the file open for writing: it opens for writing here (on Windows a writer that shares
    only reading refuses that). A read-only file cannot tell and counts as free."""
    try:
        fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    except PermissionError:
        return not os.access(path, os.W_OK)
    except OSError:
        return False
    os.close(fd)
    return True


def _fsync_file(path: Path) -> None:
    with open(path, "rb+") as f:
        os.fsync(f.fileno())


class Phone:
    """The phone in the cradle, over ADB, on the link chosen in the app (phone_link). One operation at a time."""

    def __init__(self, serial: str | None = None, camera_dir: str = CAMERA_DIR, shutter: str = "volume_down",
                 switch_modes: bool = True, adb: str | None = None, link: phone_link.Link | None = None,
                 runner: Callable | None = None, clear: bool = True) -> None:
        camera_dir = str(camera_dir).rstrip("/") or "/"
        if not SAFE_DIR.fullmatch(camera_dir) or ".." in camera_dir.split("/"):
            raise AcquisitionError("bad_value", "camera_dir")
        if str(shutter) not in KEYCODES and not str(shutter).isdigit():
            raise AcquisitionError("bad_value", "shutter")
        self.serial, self.camera_dir, self.shutter, self.switch_modes = serial, camera_dir, str(shutter), switch_modes
        self.fixed = serial         # given by hand: used as it is
        self.link = link or phone_link.get()
        self.adb = adb or find_adb()
        self.runner = runner or run_adb     # (cmd, timeout) -> (code, stdout, stderr); FakeAdb in the test
        self.clear = clear                  # files pulled are deleted from the phone
        self.left_on_phone = []             # files copied but not deleted there (the deletion failed)
        self.settle_s = PHONE_SETTLE_S
        self.recording_seen = None          # the last recording left a trace in the camera folder (None: none yet)
        self.no_countdown = True            # every photo with the camera app's own timer lifted (arm)
        self.countdown_seen = False         # a photo came COUNTDOWN_S after its press all the same
        self.last_shot = None               # {"name", "delay_s"} of the last photo taken
        self._armed = False                 # the intent that lifts the countdown is in place for the next photo
        self._armed_pid = None              # ... for this run (process) of the camera app
        self._ready_at = 0.0                # time.monotonic() when the camera app is back from that intent
        self._lock = threading.RLock()      # one operation at a time: the app's jobs share this object
        self._real_dir = None
        self._before_video = None

    @property
    def recording(self) -> bool:
        """A recording was started here and not stopped yet."""
        return self._before_video is not None

    def _run(self, *args: str, timeout: float = SHELL_TIMEOUT_S, device: bool = True) -> str:
        if not self.adb:
            raise AcquisitionError("no_adb")
        cmd = [self.adb] + (["-s", self.serial] if device and self.serial else []) + list(args)
        what = args[0] if args[0] != "shell" or len(args) < 2 else "shell " + args[1].split(" ", 1)[0]
        try:
            code, out, err = self.runner(cmd, timeout)
        except subprocess.TimeoutExpired:
            raise AcquisitionError("timeout", what) from None
        except OSError as e:
            _LOGGER.warning("adb could not be started: %s", e)
            raise AcquisitionError("no_adb") from e
        if code != 0:
            _LOGGER.warning("adb %s failed (%s): %s", what, code, (err or out).strip()[-300:])
            if device and not self.fixed and not self.link.ready():
                # the phone went away in the middle: said as the link's state, not as adb's text
                raise AcquisitionError("wifi_lost" if self.link.via == phone_link.WIFI else "no_device", what)
            raise AcquisitionError("adb_failed", what)
        return out

    def _shell(self, *words: object, tail: str = "", timeout: float = SHELL_TIMEOUT_S) -> str:
        """A command in the phone's shell, every word quoted; tail is added as it is (globs, redirections)."""
        line = phone_link.shell_line(*words) + (" " + tail if tail else "")
        return self._run("shell", line, timeout=timeout)

    def devices(self) -> list[dict]:
        """[{serial, state, model, link}] of what adb sees; starts the adb server if needed."""
        try:
            return [{"serial": d["serial"], "state": d["state"], "model": d["model"], "link": d["transport"]}
                    for d in self.link.devices()]
        except phone_link.LinkError as e:
            raise AcquisitionError(e.cause, e.detail) from e

    def connect(self) -> str:
        """The phone on the chosen link (phone_link); the serial, or AcquisitionError with the cause."""
        with self._lock:
            if self.fixed:
                if not any(d["serial"] == self.fixed and d["state"] == "device" for d in self.devices()):
                    raise AcquisitionError("no_device", self.fixed)
                self.serial = self.fixed
                return self.serial
            try:
                self.serial = self.link.serial()
            except phone_link.LinkError as e:
                raise AcquisitionError(e.cause, e.detail) from e
            return self.serial

    def status(self) -> dict:
        """{ok, cause, serial, model, foreground, link, left_on_phone}: never raises."""
        with self._lock:
            try:
                self.connect()
                model = next((d["model"] for d in self.devices() if d["serial"] == self.serial), "")
                return {"ok": True, "cause": "", "serial": self.serial, "model": model,
                        "foreground": self.foreground(), "link": self.link.via, "left_on_phone": list(self.left_on_phone)}
            except AcquisitionError as e:
                return {"ok": False, "cause": e.cause, "detail": e.detail, "serial": self.serial or "", "model": "",
                        "foreground": "", "link": self.link.via, "left_on_phone": list(self.left_on_phone)}

    @staticmethod
    def _front_of(out: str) -> str:
        m = re.search(r"(?:mResumedActivity|topResumedActivity|ResumedActivity)[^\n]*?\s([\w.]+)/", out or "")
        return m.group(1) if m else ""

    @staticmethod
    def _files_of(out: str) -> dict:
        files = {}
        for line in (out or "").splitlines():
            parts = line.strip().rsplit("|", 2)
            if len(parts) == 3 and parts[1].isdigit():
                files[parts[0].rsplit("/", 1)[-1]] = (int(parts[1]), int(parts[2] or 0))
        return files

    def foreground(self) -> str:
        """Package of the app on screen ('' if unknown)."""
        try:
            out = self._shell("dumpsys", "activity", "activities")
        except AcquisitionError:
            return ""
        return self._front_of(out)

    def listing(self) -> dict:
        """{name: (size, mtime)} of the camera folder, the media store's pending files (.pending-*) included."""
        d = shlex.quote(self.camera_dir)
        return self._files_of(self._shell("stat", "-c", "%n|%s|%Y", tail=f"{d}/* {d}/.pending-* 2>/dev/null; true"))

    def look(self) -> tuple[str, str, dict]:
        """(the app on screen, the camera app's process, the listing of the camera folder) in one call to the phone:
        what a photo needs to know before its shutter (each call costs a round trip, about 0.2 s on the cable)."""
        d = shlex.quote(self.camera_dir)
        out = self._shell("dumpsys", "activity", "activities",
                          tail=f"| grep -E 'ResumedActivity'; echo {LOOK_SEP}; pidof {CAMERA_APP}; echo {LOOK_SEP}; "
                               f"stat -c '%n|%s|%Y' {d}/* {d}/.pending-* 2>/dev/null; true")
        parts = out.split(LOOK_SEP)
        if len(parts) != 3:         # a shell that did not run it all: one call each
            return self.foreground(), self.camera_pid(), self.listing()
        return self._front_of(parts[0]), parts[1].strip(), self._files_of(parts[2])

    def press(self) -> tuple[float, int] | None:
        """The shutter, the phone woken first in the same go; the phone's clock right after it (phone_time)."""
        out = self._shell("input", "keyevent", "KEYCODE_WAKEUP", str(KEYCODES.get(self.shutter, self.shutter)),
                          tail="; date +%s.%N%z")
        return phone_time(out)

    def arm(self) -> None:
        """The camera app's countdown lifted for its next photo: the running camera gets an intent with the
        assistant's timer at 0 (see the module's text); the camera app comes to the front if it was not there.
        It is back READY_S later."""
        self._shell("am", "start", "-n", CAMERA_ACTIVITY, "-a", "android.intent.action.MAIN", "--ei", TIMER_EXTRA, "0")
        self._armed = True
        self._ready_at = time.monotonic() + READY_S

    def open_camera(self, keep_mode: bool = False) -> bool:
        """Brings the camera app back when another app is on screen: the live view of ImpactGuide
        closes it to free the camera (live_ar.py). Left alone when the app on screen is unknown. The phone is
        woken only when the camera app is not on screen already. keep_mode: started as its icon starts it (the
        launcher's intent, nothing else in it), so that it comes back in the mode it was left in."""
        if self.foreground() == CAMERA_APP:
            return False
        self._shell("input", "keyevent", "KEYCODE_WAKEUP")
        time.sleep(WAKE_S)
        front = self.foreground()
        if front and front != CAMERA_APP:
            if keep_mode:
                self._shell("am", "start", "-n", CAMERA_ACTIVITY, "-a", "android.intent.action.MAIN", "-c",
                            "android.intent.category.LAUNCHER")
            elif self.no_countdown:
                self.arm()
            else:
                self._shell("am", "start", "-a", "android.media.action.STILL_IMAGE_CAMERA")
            time.sleep(OPEN_S)
            return True
        return False

    def _mode(self, action: str) -> None:
        if self.switch_modes:
            self._shell("am", "start", "-a", f"android.media.action.{action}")
            time.sleep(MODE_S)

    def _pending(self, name: str) -> bool:
        """The media store still marks the file pending (being written). False when it cannot tell."""
        try:
            out = self._shell("content", "query", "--uri", MEDIA_URI, "--projection", "is_pending", "--where",
                              f"_data='{self.real_dir()}/{name}'")
        except AcquisitionError:
            return False
        return "is_pending=1" in out

    def _wait(self, before: dict, suffixes: Sequence[str], timeout: float, pending: bool = True) -> str:
        """The new file of the camera folder once its size stays the same (and, with `pending`, once the media
        store no longer marks it pending: a query of about 1.5 s, which take leaves to the copy, whole)."""
        sizes, deadline = {}, time.monotonic() + timeout
        while time.monotonic() < deadline:
            name = new_stable_file(self.listing(), before, sizes, suffixes, time.monotonic(), self.settle_s)
            if name and not (pending and self._pending(name)):
                return name
            if name:
                sizes.pop(name, None)       # pending: whole only after it stays unchanged once more
            time.sleep(POLL_S)
        raise AcquisitionError("timeout", "no new file in the camera folder")

    def whole(self, name: str, timeout: float = SHOT_TIMEOUT_S) -> None:
        """Waits while the media store marks a file pending (the camera still writing it)."""
        deadline = time.monotonic() + timeout
        while self._pending(name):
            if time.monotonic() > deadline:
                raise AcquisitionError("timeout", name)
            time.sleep(POLL_S)

    def real_dir(self) -> str:
        """The camera folder as the media store names it (/sdcard is a link to /storage/emulated/0)."""
        if self._real_dir is None:
            out = self._shell("readlink", "-f", self.camera_dir).strip()
            self._real_dir = out if SAFE_DIR.fullmatch(out) else self.camera_dir
        return self._real_dir

    def remote(self, name: str) -> tuple[int, str]:
        """(size, SHA-256) of a file in the camera folder."""
        path = f"{self.camera_dir}/{name}"
        size = self._shell("stat", "-c", "%s", path).strip()
        if not size.isdigit():
            raise AcquisitionError("not_found", name)
        digest = self._shell("sha256sum", path, timeout=SHELL_TIMEOUT_S + int(size) / HASH_RATE_BPS).split()
        if not digest or not re.fullmatch(r"[0-9a-fA-F]{64}", digest[0]):
            raise AcquisitionError("not_found", name)
        return int(size), digest[0].lower()

    def in_media(self, name: str) -> bool:
        """Whether the media store (the gallery) still lists the file."""
        out = self._shell("content", "query", "--uri", MEDIA_URI, "--projection", "_id", "--where",
                          f"_data='{self.real_dir()}/{name}'")
        return "Row:" in out

    def remove(self, name: str) -> bool:
        """The file deleted from the phone and taken out of the media store, so that the gallery does not show it.
        True when both are gone (checked); False otherwise, and the file is noted in left_on_phone."""
        if not SAFE_NAME.fullmatch(name):
            self.left_on_phone.append(name)
            return False
        with self._lock:
            path = f"{self.camera_dir}/{name}"
            try:
                self._shell("rm", "-f", path)
                try:
                    self._shell("content", "delete", "--uri", MEDIA_URI, "--where", f"_data='{self.real_dir()}/{name}'")
                except AcquisitionError:
                    # without the media store's own delete, a scan of the path that is gone takes it out (old Android)
                    self._shell("am", "broadcast", "-a", "android.intent.action.MEDIA_SCANNER_SCAN_FILE", "-d",
                                f"file://{self.real_dir()}/{name}")
                gone = name not in self.listing() and not self.in_media(name)
            except AcquisitionError:
                gone = False
            if not gone:
                self.left_on_phone.append(name)
                _LOGGER.warning("%s copied but left on the phone", name)
            return gone

    def pull(self, name: str, out_dir: Path | str, whole: bool = False) -> Path:
        """The file copied into out_dir and checked against the phone's (size and SHA-256): it lands as a hidden
        .part file and gets its name only once checked, then it is deleted from the phone (remove) unless clear is
        off. A copy that fails or differs is deleted here, the file stays on the phone and the cause is
        copy_failed. The file is first waited for while the media store marks it pending, unless the caller knows
        it whole."""
        if not SAFE_NAME.fullmatch(name):
            raise AcquisitionError("not_found", name)
        with self._lock:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dest, part = out_dir / name, out_dir / f".{name}{PART}"
            part.unlink(missing_ok=True)
            try:
                if not whole:
                    self.whole(name)
                size, digest = self.remote(name)
                self._run("pull", f"{self.camera_dir}/{name}", str(part), timeout=PULL_TIMEOUT_S + size / PULL_RATE_BPS)
                same = part.is_file() and part.stat().st_size == size and sha256(part) == digest
            except AcquisitionError as e:
                part.unlink(missing_ok=True)
                raise AcquisitionError("copy_failed", name) from e
            if not same:
                part.unlink(missing_ok=True)
                raise AcquisitionError("copy_failed", name)
            try:
                _fsync_file(part)
                os.replace(part, dest)
            except OSError as e:
                part.unlink(missing_ok=True)
                raise AcquisitionError("copy_failed", name) from e
            if self.clear:
                self.remove(name)
            return dest

    def take(self, timeout: float = SHOT_TIMEOUT_S, pressed: Callable[[], None] | None = None,
             keep_mode: bool = False) -> str:
        """One photo on the phone without the camera app's countdown: the shutter, then the new file waited for
        until its size stays the same; its name in the camera folder (pull waits until the media store has it whole
        and copies it). pressed() is called right after the shutter, from this thread. Before the shutter one call
        to the phone (look) when the camera app is on screen and was told already. keep_mode: the camera app left
        in the mode it is in (no intent to it: its own timer then counts down, if on)."""
        with self._lock:
            self.connect()
            front, pid, before = self.look()
            if front != CAMERA_APP:
                self.open_camera(keep_mode)
                front, pid, before = self.look()
            lift = self.no_countdown and not keep_mode
            if keep_mode:
                self._armed = False
            # the intent sent after the last photo holds only for the same run of the camera app: one started again
            # meanwhile (by hand, after the live view) gets it again
            if lift and (not self._armed or pid != self._armed_pid):
                self.arm()
                self._armed_pid = pid
            wait = self._ready_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            at = self.press()
            self._armed = False
            if pressed is not None:
                pressed()
            name = self._wait(before, PHOTO_SUFFIXES, timeout, pending=False)
            delay = shot_delay(name, at)
            self.last_shot = {"name": name, "delay_s": delay}
            if delay is not None and delay >= COUNTDOWN_S:
                self.countdown_seen = True
                _LOGGER.warning("%s came %.1f s after the shutter: the camera app counted down", name, delay)
            if lift:
                try:
                    self.arm()                  # ready for the next photo
                    self._armed_pid = pid
                except AcquisitionError as e:   # the next take arms again
                    _LOGGER.warning("the countdown could not be lifted for the next photo (%s)", e)
            return name

    def camera_pid(self) -> str:
        """The process of the camera app ('' when it does not run or the phone does not say)."""
        try:
            return self._shell("pidof", CAMERA_APP).strip()
        except AcquisitionError:
            return ""

    def shoot(self, out_dir: Path | str, timeout: float = SHOT_TIMEOUT_S, keep_mode: bool = False) -> Path:
        """One photo: shutter, wait for the file, pull it into out_dir. Returns the local path. keep_mode as in
        take: no intent to the camera app (a trial's photos, taken in the Pro mode of the calibration)."""
        with self._lock:
            return self.pull(self.take(timeout, keep_mode=keep_mode), out_dir)

    def copier(self) -> Phone:
        """This phone for copying a photo (pull) while the next one is taken here: the same link, adb and list of
        files left on the phone, a lock of its own."""
        c = copy.copy(self)
        c._lock = threading.RLock()
        return c

    def start_video(self) -> None:
        """Starts a recording. Refused (busy) while one started here runs: the same key would stop it. Whether the
        camera folder shows the recording is kept in recording_seen (and said in the log when it does not)."""
        with self._lock:
            if self._before_video is not None:
                raise AcquisitionError("busy", "recording")
            self.connect()
            self.open_camera()
            self._mode("VIDEO_CAMERA")
            before = self.listing()
            self.press()
            self._before_video = before         # only once the shutter was pressed: a failed press leaves none
            try:
                self.recording_seen = self._trace(before)
            except AcquisitionError as e:       # the press went out: the phone records, Stop must stay possible
                self.recording_seen = None
                _LOGGER.warning("the camera folder could not be listed after the shutter (%s): recording assumed", e)
                return
            if not self.recording_seen:
                _LOGGER.warning("no trace of the recording in the camera folder after %.0f s", VIDEO_START_S)

    def _trace(self, before: dict) -> bool:
        """A file new or growing in the camera folder (the media store's pending one included) soon after the
        shutter: the recording runs."""
        deadline = time.monotonic() + VIDEO_START_S
        while True:
            now = self.listing()
            if any(before.get(k) != v for k, v in now.items()):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(POLL_S)

    def stop_video(self, out_dir: Path | str, timeout: float = VIDEO_TIMEOUT_S) -> Path:
        """Stops the recording started by start_video and pulls the video into out_dir."""
        with self._lock:
            if self._before_video is None:
                raise AcquisitionError("not_found", "no recording started")
            self.connect()          # over Wi-Fi the serial may have changed while recording
            self.press()
            try:
                return self.pull(self._wait(self._before_video, VIDEO_SUFFIXES, timeout), out_dir, whole=True)
            finally:
                self._before_video = None
                try:
                    self._mode("STILL_IMAGE_CAMERA")
                except AcquisitionError:
                    pass


class FakePhone:
    """Writes stand-in files the way the phone names them: a grey photo at the photo size with
    the time written on it, and an 8 s video. With a link (a simulated one, phone_link.SimulatedAdb)
    it answers only while the phone is there on the chosen link, with the link's cause otherwise."""

    def __init__(self, photo_px=(4080, 3072), video_s=8.0, link=None):
        self.photo_px, self.video_s, self.link = photo_px, video_s, link
        self.scene = None           # () -> image: what the camera sees (a calibration board), in place of the grey
        self.left_on_phone = []
        self.countdown_seen = False
        self.last_shot = None
        self._recording = False
        self._n = 0
        self._taken = {}            # photos taken and not yet copied: name -> image
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._recording

    def _check(self):
        if self.link is not None:
            try:
                self.link.serial()
            except phone_link.LinkError as e:
                raise AcquisitionError(e.cause, e.detail) from e

    def _name(self, prefix, suffix):
        self._n += 1
        return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{self._n:03d}{suffix}"

    def status(self):
        mode = self.link.via if self.link is not None else phone_link.USB
        try:
            self._check()
        except AcquisitionError as e:
            return {"ok": False, "cause": e.cause, "serial": "", "model": "", "foreground": "", "link": mode,
                    "left_on_phone": []}
        return {"ok": True, "cause": "", "serial": "simulated", "model": "simulated phone", "foreground": "simulated",
                "link": mode, "left_on_phone": []}

    def take(self, timeout=SHOT_TIMEOUT_S, pressed=None, keep_mode=False):
        import cv2
        import numpy as np
        self._check()
        scene = self.scene
        if pressed is not None:
            pressed()
        w, h = self.photo_px
        if scene is not None:
            img = scene()
        else:
            img = np.full((h, w, 3), 96, np.uint8)
            cv2.putText(img, f"simulated photo {datetime.now():%H:%M:%S}", (w // 12, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, w / 900, (230, 230, 230), max(2, w // 600))
        name = self._name("IMG", ".jpg")
        with self._lock:
            self._taken[name] = (img, scene is not None)
        self.last_shot = {"name": name, "delay_s": 0.0}
        return name

    def pull(self, name, out_dir):
        import cv2
        with self._lock:
            img, real = self._taken.pop(name)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92 if real else 70])
        return path

    def copier(self):
        return self

    def shoot(self, out_dir, timeout=SHOT_TIMEOUT_S, keep_mode=False):
        return self.pull(self.take(timeout, keep_mode=keep_mode), out_dir)

    def start_video(self):
        self._check()
        if self._recording:
            raise AcquisitionError("busy", "recording")
        self._recording = True

    def stop_video(self, out_dir, timeout=VIDEO_TIMEOUT_S):
        import cv2
        import numpy as np
        self._check()
        if not self._recording:
            raise AcquisitionError("not_found", "no recording started")
        self._recording = False
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / self._name("VID", ".mp4")
        fps, size = 10, (320, 240)
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for i in range(int(self.video_s * fps)):
            frame = np.full((size[1], size[0], 3), 64, np.uint8)
            cv2.putText(frame, f"{i / fps:.1f} s", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2)
            w.write(frame)
        w.release()
        return path


class FakeAdb:
    """adb and a phone's camera folder, for the test of Phone: a folder on the PC stands for /sdcard/DCIM/Camera and a
    set of paths for the media store. The shutter key adds a photo (a JPEG of `photo_bytes` random bytes); in video
    mode the first press starts a recording (a pending file, as the media store names one being written) and the
    second turns it into the video. corrupt makes the next pull write a file cut short, fail_pull makes it fail,
    deaf makes the camera ignore the next press. timer_s is the camera app's own timer: a photo comes that long after
    the press (its name says so), unless the intent of arm came before it and `honours` (the camera reads the extra,
    for one photo). The phone's clock (date) is this computer's, in the zone +02:00. Every shell line is read as
    the phone's sh reads it."""

    def __init__(self, folder, photo_bytes=40000):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.media = {f"/storage/emulated/0/DCIM/Camera/{p.name}" for p in self.folder.iterdir()}
        self.photo_bytes, self.corrupt, self.fail_pull, self.deaf = photo_bytes, False, False, False
        self.video_mode, self.recording, self.n, self.calls = False, None, 0, []
        self.pending = set()            # media store paths still being written
        self.lines = []                 # shell lines as adb passes them
        self.timer_s, self.honours, self.armed = 0.0, True, False
        self.pid = 4242                 # the camera app's process; a new one is a camera started again (the extra lost)
        self.front = CAMERA_APP         # the app on screen; any start of the camera app brings it back
        self.pending_s = 0.0            # a new photo stays pending in the media store this long
        self.delays = []                # seconds from each press to its photo, as the camera took it
        self.zone = timezone(timedelta(hours=2))

    def _local(self, path):
        name = path.strip("'").rsplit("/", 1)[-1]
        return self.folder / name

    def _bytes(self):
        import random
        return bytes(random.getrandbits(8) for _ in range(self.photo_bytes))

    def _new(self, suffix, at=None):
        self.n += 1
        if at is None:
            p = self.folder / f"{'IMG' if suffix == '.jpg' else 'VID'}_20261016_1200{self.n:02d}{suffix}"
        else:       # named by the time the camera took it, as the camera does (a second one in that second: _1)
            stem = f"IMG_{datetime.fromtimestamp(at, self.zone):%Y%m%d_%H%M%S}"
            p = self.folder / f"{stem}{suffix}"
            k = 0
            while p.exists():
                k += 1
                p = self.folder / f"{stem}_{k}{suffix}"
        p.write_bytes(self._bytes())
        path = f"/storage/emulated/0/DCIM/Camera/{p.name}"
        self.media.add(path)
        if self.pending_s:          # the media store says it whole only this much later
            self.pending.add(path)
            threading.Timer(self.pending_s, lambda: self.pending.discard(path)).start()
        return p

    def __call__(self, cmd, timeout):
        args = cmd[3:] if len(cmd) > 2 and cmd[1] == "-s" else cmd[1:]
        self.calls.append(args)
        if args[0] == "pull":
            if self.fail_pull:
                self.fail_pull = False
                return 1, "", "adb: error: failed to copy"
            src, dest = self._local(args[1]), Path(args[2])
            if not src.is_file():
                return 1, "", "no such file"
            data = src.read_bytes()
            if self.corrupt:
                self.corrupt, data = False, data[:len(data) // 2]
            dest.write_bytes(data)
            return 0, "1 file pulled", ""
        if args[0] != "shell":
            return 0, "", ""
        c = " ".join(args[1:])
        self.lines.append(c)
        if c.startswith("dumpsys activity activities |"):      # look: several commands in one line
            out = []
            for part in (p.strip() for p in c.split("; ")):
                if part.startswith("dumpsys"):
                    out.append(self(["adb", "shell", "dumpsys activity activities"], timeout)[1])
                elif part.startswith("echo "):
                    out.append(part[5:] + "\n")
                elif part.startswith("pidof"):
                    out.append(f"{self.pid}\n")
                elif part.startswith("stat"):
                    out.append(self(["adb", "shell", part + "; true"], timeout)[1])
            return 0, "".join(out), ""
        if c.startswith("stat -c '%n|%s|%Y'"):
            rows = [f"/sdcard/DCIM/Camera/{p.name}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
                    for p in sorted(self.folder.iterdir())]
            return 0, "\n".join(rows) + "\n", ""
        words = shlex.split(c)
        if words[:3] == ["stat", "-c", "%s"]:
            p = self._local(words[-1])
            return (0, f"{p.stat().st_size}\n", "") if p.is_file() else (1, "", "No such file")
        if words[0] == "sha256sum":
            p = self._local(words[-1])
            return (0, f"{sha256(p)}  {words[-1]}\n", "") if p.is_file() else (1, "", "No such file")
        if words[:2] == ["readlink", "-f"]:
            return 0, "/storage/emulated/0/DCIM/Camera\n", ""
        if words[:2] == ["rm", "-f"]:
            self._local(words[-1]).unlink(missing_ok=True)
            return 0, "", ""
        if words[0] == "content":
            where = words[words.index("--where") + 1] if "--where" in words else ""
            m = re.fullmatch(r"_data='([^']+)'", where)
            path = m.group(1) if m else ""
            if words[1] == "delete":
                self.media.discard(path)
                return 0, "", ""
            if "is_pending" in words:
                if path not in self.media:
                    return 0, "No result found.\n", ""
                return 0, f"Row: 0 is_pending={int(path in self.pending)}\n", ""
            return 0, ("Row: 0 _id=1\n" if path in self.media else "No result found.\n"), ""
        if words[:2] == ["dumpsys", "activity"]:
            return 0, f"  mResumedActivity: ActivityRecord{{1 u0 {self.front}/.Main t1}}\n", ""
        if words[0] == "pidof":
            return 0, f"{self.pid}\n", ""
        if words[:2] == ["am", "start"]:
            self.front = CAMERA_APP
            self.video_mode = "android.media.action.VIDEO_CAMERA" in words
            if TIMER_EXTRA in words and words[words.index(TIMER_EXTRA) + 1] == "0":
                self.armed = True
            return 0, "", ""
        keys = words[2:words.index(";")] if ";" in words else words[2:]
        clock = f"{time.time():.9f}+0200\n" if "date" in words else ""
        if words[:2] == ["input", "keyevent"] and any(k != "KEYCODE_WAKEUP" for k in keys):
            if self.deaf:
                self.deaf = False
                return 0, clock, ""
            if self.video_mode:
                if self.recording is None:
                    self.n += 1
                    self.recording = self.folder / f".pending-1760000000-VID_20261016_1200{self.n:02d}.mp4"
                    self.recording.write_bytes(b"\0" * 100)
                else:
                    final = self.folder / self.recording.name.split("-", 2)[-1]
                    final.write_bytes(self._bytes())
                    self.recording.unlink()
                    self.recording = None
                    self.media.add(f"/storage/emulated/0/DCIM/Camera/{final.name}")
            else:
                countdown = 0.0 if self.armed and self.honours else self.timer_s
                self.armed = False
                self.delays.append(countdown + 0.2)
                self._new(".jpg", at=time.time() + countdown + 0.2 if clock else None)
            return 0, clock, ""
        return 0, "", ""


def folder_listing(folder):
    folder = Path(folder)
    out = {}
    if folder.is_dir():
        for p in folder.iterdir():
            if p.is_file():
                st = p.stat()
                out[p.name] = (st.st_size, st.st_mtime_ns)
    return out


class ExportWatch:
    """The scanner's export folder: poll() returns the first scan file written there after the watch started, once
    its size and time have stayed the same for settle_s and no program holds it open for writing; None until
    then."""

    def __init__(self, folder: Path | str, suffixes: Sequence[str] = SCAN_SUFFIXES, settle_s: float = SCAN_SETTLE_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.folder, self.suffixes = Path(folder), tuple(suffixes)
        self.settle_s, self.clock = settle_s, clock
        self.before = folder_listing(self.folder)
        self.sizes = {}

    def poll(self) -> Path | None:
        name = new_stable_file(folder_listing(self.folder), self.before, self.sizes, self.suffixes, self.clock(),
                               self.settle_s)
        if not name:
            return None
        path = self.folder / name
        if not free_to_read(path):
            self.sizes.pop(name, None)          # still being written: settled only after another wait
            return None
        return path


def fake_scan(folder, name=None):
    """A small ASCII PLY standing in for an exported scan."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name or f"scan_{datetime.now():%Y%m%d_%H%M%S_%f}.ply")
    pts = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)]
    head = f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    path.write_text(head + "".join(f"{x} {y} {z}\n" for x, y, z in pts), encoding="ascii")
    return path


class _Clock:
    """Time the tests move by hand."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _hold_for_writing(path):
    """Opens a file for writing the way a scanner's exporter does on Windows (others may only read it): a handle to
    close, or None off Windows."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    generic_write, share_read, open_existing = 0x40000000, 0x1, 3
    h = k32.CreateFileW(str(path), generic_write, share_read, None, open_existing, 0, None)
    return None if h in (None, wintypes.HANDLE(-1).value) else (k32, h)


def run_test():
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            fails.append(what)

    before = {"IMG_1.jpg": (100, 1)}
    sizes = {}
    grow = {"IMG_1.jpg": (100, 1), "IMG_2.jpg": (50, 2), ".pending-IMG_3.jpg": (10, 3)}
    first = new_stable_file(grow, before, sizes, PHOTO_SUFFIXES)
    second = new_stable_file({**grow, "IMG_2.jpg": (80, 3)}, before, sizes, PHOTO_SUFFIXES)
    third = new_stable_file({**grow, "IMG_2.jpg": (80, 3)}, before, sizes, PHOTO_SUFFIXES)
    expect(first is None and second is None and third == "IMG_2.jpg",
           "a new photo is taken only once its size stops growing; old and pending files are skipped")
    sizes = {}
    seen = [new_stable_file({"IMG_2.jpg": (80, 3)}, {}, sizes, PHOTO_SUFFIXES, now=t, settle_s=2.0)
            for t in (0.0, 0.5, 1.5, 2.1)]
    expect(seen == [None, None, None, "IMG_2.jpg"], "with a settle time: whole only once unchanged that long")

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        phone = FakePhone(photo_px=(640, 480), video_s=2)
        photo = phone.shoot(tmp / "staging")
        phone.start_video()
        busy = False
        try:
            phone.start_video()
        except AcquisitionError as e:
            busy = e.cause == "busy"
        video = phone.stop_video(tmp / "staging")
        import cv2
        img = cv2.imread(str(photo))
        cap = cv2.VideoCapture(str(video))
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        expect(img is not None and img.shape[:2] == (480, 640) and photo.name.startswith("IMG_"),
               f"fake photo {photo.name}")
        expect(fps > 0 and abs(frames / fps - 2) < 0.2 and video.name.startswith("VID_") and busy,
               f"fake video {video.name}; a second start while recording refused")

        export = tmp / "export"
        export.mkdir()
        fake_scan(export, "old.ply")
        clock = _Clock()
        watch = ExportWatch(export, clock=clock)
        expect(watch.poll() is None, "export folder: nothing new yet")
        scan = fake_scan(export, "t001_after.ply")
        got = [watch.poll()]
        clock.t += 1.0
        got.append(watch.poll())
        clock.t += 1.5
        got.append(watch.poll())
        expect(got[:2] == [None, None] and got[2] == scan,
               f"export folder: the new scan once unchanged for {SCAN_SETTLE_S:g} s")
        with open(scan, "a", encoding="ascii") as f:
            f.write("0 0 1\n")
        clock.t += 1.0
        grown = watch.poll()
        clock.t += 2.5
        expect(grown is None and watch.poll() == scan, "a scan written again waits for its new size to settle")
        w3 = ExportWatch(export, clock=clock)
        held = _hold_for_writing(fake_scan(export, "t002_after.ply"))
        if held is not None:
            k32, h = held
            polls = []
            for _ in range(3):
                polls.append(w3.poll())
                clock.t += 2.5
            k32.CloseHandle(h)
            polls.append(w3.poll())
            clock.t += 2.5
            polls.append(w3.poll())
            expect(polls[:3] == [None] * 3 and polls[-1] == export / "t002_after.ply",
                   "a scan still held open by its writer is not taken; once closed it is")
        fake_scan(export, "old.ply").write_text("ply\nrewritten\n", encoding="ascii")
        w2 = ExportWatch(export, clock=clock)
        expect(w2.poll() is None, "export folder: files already there are not new")

    # Phone itself on a fake adb: a photo and a video copied, checked and deleted from the phone and the gallery;
    # a copy that fails or differs leaves the file on the phone
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        camera = tmp / "phone_camera"
        camera.mkdir()
        (camera / "IMG_20261001_090000.jpg").write_bytes(b"an old photo of the bench")
        adb = FakeAdb(camera)
        link = phone_link.Link(runner=phone_link.SimulatedAdb())
        phone = Phone(link=link, adb="adb", runner=adb)
        before, media = folder_listing(camera), set(adb.media)
        photo = phone.shoot(tmp / "pc")
        same = photo.is_file() and photo.stat().st_size == adb.photo_bytes
        expect(same and folder_listing(camera) == before and adb.media == media and not phone.left_on_phone,
               f"photo {photo.name}: on the PC, checked (size and SHA-256), gone from the phone's camera folder and "
               f"from the gallery; the folder as it was ({len(before)} file)")
        phone.start_video()
        video = phone.stop_video(tmp / "pc")
        expect(video.suffix == ".mp4" and video.is_file() and folder_listing(camera) == before and adb.media == media
               and phone.recording_seen and not phone.recording,
               f"video {video.name}: its recording seen in the camera folder; on the PC, gone from the phone and the "
               f"gallery")
        for what in ("corrupt", "fail_pull"):
            setattr(adb, what, True)
            n = len(list(camera.iterdir()))
            try:
                phone.shoot(tmp / "pc")
                cause = ""
            except AcquisitionError as e:
                cause = e.cause
            left = [p.name for p in camera.iterdir() if p.name not in before]
            on_pc = [p.name for p in (tmp / "pc").iterdir() if p.name in left or p.name.endswith(PART)]
            expect(cause == "copy_failed" and len(left) == 1 and len(list(camera.iterdir())) == n + 1 and not on_pc
                   and f"/storage/emulated/0/DCIM/Camera/{left[0]}" in adb.media,
                   f"a copy that {'differs' if what == 'corrupt' else 'fails'}: copy_failed, {left} stays on the "
                   f"phone and in the gallery, nothing half copied on the PC")
            for p in camera.iterdir():          # the next case starts from the folder as it was
                if p.name not in before:
                    p.unlink()
                    adb.media.discard(f"/storage/emulated/0/DCIM/Camera/{p.name}")
        rm = [c for c in adb.lines if c.startswith("rm -f")]
        expect(all(re.fullmatch(r"rm -f '?/sdcard/DCIM/Camera/(IMG|VID)_\w+\.(jpg|mp4)'?", c) for c in rm)
               and len(rm) == 2, f"deleted on the phone: only the two files the app took ({len(rm)} rm)")
        # a file the media store still marks pending is not taken until it is whole
        adb.pending.add("/storage/emulated/0/DCIM/Camera/IMG_20261016_120099.jpg")
        pend = camera / "IMG_20261016_120099.jpg"
        listed = phone.listing()
        pend.write_bytes(b"x" * 10)
        adb.media.add("/storage/emulated/0/DCIM/Camera/IMG_20261016_120099.jpg")
        phone.settle_s = 0.0
        timer = threading.Timer(1.5, lambda: adb.pending.clear())
        timer.start()
        t0 = time.monotonic()
        name = phone._wait(listed, PHOTO_SUFFIXES, 10)
        timer.join()
        expect(name == pend.name and time.monotonic() - t0 >= 1.4,
               "a file the media store marks pending is waited for until it is whole")
        pend.unlink()
        adb.media.discard("/storage/emulated/0/DCIM/Camera/IMG_20261016_120099.jpg")
        phone.settle_s = PHONE_SETTLE_S
        # a press the camera ignores: no recording left behind for the stop to start
        adb.deaf = True
        phone.start_video()
        quiet = phone.recording_seen is False and phone.recording
        adb.recording = None
        phone._before_video = None
        expect(quiet, "a recording that leaves no trace in the camera folder is said (recording_seen False)")
        # the camera folder unreadable just after the press (an adb hiccup on Wi-Fi): the phone records, Stop works
        adb.deaf = False

        def unreadable(before):
            raise AcquisitionError("adb_failed", "listing failed")
        phone._trace = unreadable
        try:
            phone.start_video()
            went = True
        except AcquisitionError:
            went = False
        del phone._trace
        expect(went and phone.recording and phone.recording_seen is None,
               "the folder unreadable right after the shutter: the recording counts as started, Stop stays possible")
        adb.recording = None
        phone._before_video = None
        failing = Phone(link=link, adb="adb", runner=lambda cmd, t: (_ for _ in ()).throw(OSError("bad press")))
        try:
            failing.start_video()
        except AcquisitionError:
            pass
        expect(not failing.recording, "a start whose press fails leaves no recording for the stop to press again")
        # what reaches the phone's shell: quoted words, a timeout scaled with the file
        big = camera / "VID_20261016_130000.mp4"
        big.write_bytes(b"\0" * 2_000_000)
        timeouts = []
        phone.runner = lambda cmd, t: (timeouts.append((cmd[3:5], t)), adb(cmd, t))[1]
        phone.remote(big.name)
        hash_t = next(t for c, t in timeouts if c[1].startswith("sha256sum"))
        expect(hash_t > SHELL_TIMEOUT_S, f"the hash of a large file has a longer timeout ({hash_t:.1f} s)")
        big.unlink()
        # no countdown, whatever the camera app's own timer says: its extra at 0 before every photo
        phone.runner = adb
        adb.timer_s, adb.delays, adb.lines = 3.0, [], []
        fresh = Phone(link=link, adb="adb", runner=adb)
        got = [fresh.shoot(tmp / "pc") for _ in range(3)]
        order = [("arm" if TIMER_EXTRA in c else "press") for c in adb.lines
                 if TIMER_EXTRA in c or (c.startswith("input keyevent") and "; date" in c)]
        expect(adb.delays == [0.2] * 3 and order == ["arm", "press"] * 3 + ["arm"] and not fresh.countdown_seen
               and all(p.is_file() for p in got) and fresh.last_shot["delay_s"] is not None
               and fresh.last_shot["delay_s"] < COUNTDOWN_S,
               f"the camera's timer at 3 s: 3 photos each at once ({adb.delays} s), the countdown lifted before each "
               f"press ({' '.join(order)}); measured {fresh.last_shot and fresh.last_shot['delay_s']} s")
        adb.pid, adb.armed, adb.lines = 5151, False, []      # the camera app started again: what it was told is gone
        fresh.shoot(tmp / "pc")
        again = [("arm" if TIMER_EXTRA in c else "press") for c in adb.lines
                 if TIMER_EXTRA in c or (c.startswith("input keyevent") and "; date" in c)]
        expect(again == ["arm", "press", "arm"] and adb.delays[-1] == 0.2,
               f"the camera app started again between two photos: told again before the press ({' '.join(again)})")
        adb.honours, adb.delays = False, []
        fresh.shoot(tmp / "pc")
        expect(fresh.countdown_seen and fresh.last_shot["delay_s"] >= COUNTDOWN_S,
               f"a camera that counts down all the same: said (countdown_seen, {fresh.last_shot['delay_s']} s)")
        adb.timer_s, adb.honours = 0.0, True
        # the camera's mode kept (a calibration in Pro mode): no intent reaches the camera app, neither on screen nor
        # brought back from another app (then as its icon starts it); the next photo without keep_mode lifts the
        # countdown again
        adb.lines = []
        kept = fresh.take(keep_mode=True)
        on_screen = [c for c in adb.lines if c.startswith("am start")]
        adb.front, adb.lines = "com.other.app", []
        back = fresh.take(keep_mode=True)
        brought = [c for c in adb.lines if c.startswith("am start")]
        adb.lines = []
        third = fresh.take()
        lifted = [c for c in adb.lines if TIMER_EXTRA in c]
        expect(kept and back and not on_screen and len(brought) == 1 and "category.LAUNCHER" in brought[0]
               and TIMER_EXTRA not in brought[0] and "STILL_IMAGE" not in brought[0] and lifted,
               f"the camera's mode kept: no intent on screen, brought back as its icon starts it ({brought}); the "
               f"countdown lifted again without it ({len(lifted)} intents)")
        for n in (kept, back, third):
            (camera / n).unlink(missing_ok=True)
        # a photo the media store still says pending once its size settled: taken at once, its copy waits for it
        adb.pending_s = 2.0
        t0 = time.monotonic()
        name = fresh.take()
        t_take = time.monotonic() - t0
        got = fresh.pull(name, tmp / "pc")
        t_pull = time.monotonic() - t0
        adb.pending_s = 0.0
        expect(t_take < 2.0 <= t_pull and got.stat().st_size == adb.photo_bytes,
               f"a photo still pending in the media store: taken in {t_take:.1f} s, copied once whole ({t_pull:.1f} s)")
        # a photo copied while the next is taken: the copier has its own lock, both photos whole, the phone as before
        names = [fresh.take()]
        copier = fresh.copier()
        copied = []
        th = threading.Thread(target=lambda: copied.append(copier.pull(names[0], tmp / "pc2")))
        th.start()
        names.append(fresh.take())
        th.join()
        copied.append(fresh.pull(names[1], tmp / "pc2"))
        expect(len(copied) == 2 and all(p.is_file() and p.stat().st_size == adb.photo_bytes for p in copied)
               and sorted(k for k in folder_listing(camera) if k.startswith("IMG_")) == sorted(
                   k for k in before if k.startswith("IMG_")) and copier._lock is not fresh._lock
               and copier.left_on_phone is fresh.left_on_phone,
               f"a photo copied while the next is taken: {', '.join(p.name for p in copied)}; the camera folder as it "
               f"was")
    for bad in ("/sdcard/DCIM/Camera; rm -rf /", "/sdcard/My Photos", "sdcard/DCIM", "/sdcard/../data", "/x'y"):
        try:
            Phone(camera_dir=bad, link=link, adb="adb")
            refused = False
        except AcquisitionError as e:
            refused = e.cause == "bad_value"
        expect(refused, f"camera folder {bad!r} refused before anything reaches the phone")
    try:
        Phone(shutter="power; reboot", link=link, adb="adb")
        refused = False
    except AcquisitionError as e:
        refused = e.cause == "bad_value"
    expect(refused, "a shutter key that is not a key code is refused")
    gone = Phone(link=link, adb="adb", runner=lambda cmd, t: (_ for _ in ()).throw(FileNotFoundError("adb")))
    try:
        gone.listing()
        cause = ""
    except AcquisitionError as e:
        cause = e.cause
    expect(cause == "no_adb", f"adb that cannot be started: no_adb, not a raw error ({cause})")
    expect(not SAFE_NAME.fullmatch("..") and not SAFE_NAME.fullmatch(".pending-x.jpg")
           and SAFE_NAME.fullmatch("IMG_20261016_120000.jpg"), "only the camera's own file names are ever deleted")
    sim = phone_link.SimulatedAdb()
    shy = Phone(link=phone_link.Link(phone_link.USB, runner=sim), adb="adb",
                runner=lambda cmd, t: (1, "", "error: device offline\njava.lang.IllegalStateException at ..."))
    try:
        shy.foreground() or shy.listing()
        detail = ""
    except AcquisitionError as e:
        detail = e.detail
    expect("java" not in detail and "offline" not in detail, f"adb's text stays in the log ({detail!r})")

    # the link decides: a simulated phone on Wi-Fi that drops in the middle
    sim = phone_link.SimulatedAdb(auto_connect=False)
    link = phone_link.Link(phone_link.WIFI, runner=sim)
    link.pair_code(f"{sim.ip}:{sim.pair_port}", sim.code)
    with tempfile.TemporaryDirectory() as tmp:
        phone = FakePhone(photo_px=(64, 48), link=link)
        ok_photo = phone.shoot(Path(tmp)).is_file()
        sim.drop()
        link.check()
        try:
            phone.shoot(Path(tmp))
            cause = ""
        except AcquisitionError as e:
            cause = e.cause
    expect(ok_photo and cause == "wifi_reconnecting" and phone.status()["cause"] == "wifi_reconnecting",
           f"photo over Wi-Fi, then the link drops: refused with a plain cause ({cause})")
    sim.back()
    serial = link.wait_ready(10)
    expect(serial and Phone(link=link).connect() == serial == link.serial(),
           f"the phone over ADB asks the link for its serial: {serial}")
    print(f"      adb found: {find_adb() or 'none on this PC'}")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed")
    return 1 if fails else 0


def clear_check():
    """On the phone on the cable or Wi-Fi: a small test file put in its camera folder (adb push, so the media store
    lists it as a photo), pulled as a photo is, deleted there and out of the media store; then the folder and the
    store as they were before. No photo is taken and nothing else is touched."""
    phone = Phone()
    phone.connect()
    name = f"IMG_{datetime.now():%Y%m%d_%H%M%S}_impactguide_check.jpg"
    real = f"{phone.real_dir()}/{name}"
    before = phone.listing()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / name
        import numpy as np
        import cv2
        cv2.imwrite(str(src), np.full((64, 64, 3), 128, np.uint8))
        phone._run("push", str(src), f"{phone.camera_dir}/{name}", timeout=PUSH_TIMEOUT_S)
        listed = False
        for _ in range(20):         # the media store takes in a file written over adb by itself
            listed = phone.in_media(name)
            if listed:
                break
            phone._shell("am", "broadcast", "-a", "android.intent.action.MEDIA_SCANNER_SCAN_FILE", "-d",
                         f"file://{real}")
            time.sleep(0.5)
        got = phone.pull(name, Path(tmp) / "pc")
        same = sha256(got) == sha256(src)
    after = phone.listing()
    gone = name not in after and not phone.in_media(name)
    print(f"test file in the camera folder, listed by the media store: {listed}")
    print(f"pulled and identical: {same}; deleted from the phone and the media store: {gone}")
    print(f"camera folder as before: {after == before} ({len(before)} files)")
    return 0 if same and gone and after == before else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices")
    a = sub.add_parser("shoot")
    a.add_argument("--out", type=Path, default=Path("."))
    sub.add_parser("test")
    sub.add_parser("clear-check")
    args = p.parse_args()
    try:
        if args.cmd == "devices":
            phone = Phone()
            print(f"adb: {phone.adb or 'not found'}")
            for d in phone.devices():
                print(f"{d['serial']}  {d['state']}  {d['model']}")
            print(phone.status())
        elif args.cmd == "shoot":
            print(Phone().shoot(args.out))
        elif args.cmd == "test":
            sys.exit(run_test())
        elif args.cmd == "clear-check":
            sys.exit(clear_check())
    except AcquisitionError as e:
        sys.exit(f"acquisition: {e}")


if __name__ == "__main__":
    main()
