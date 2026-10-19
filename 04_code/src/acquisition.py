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
from datetime import datetime
from pathlib import Path

import phone_link

_LOGGER = logging.getLogger("acquisition")

PHOTO_SUFFIXES = (".jpg", ".jpeg")
VIDEO_SUFFIXES = (".mp4",)
SCAN_SUFFIXES = (".ply", ".stl", ".obj", ".asc")
KEYCODES = {"volume_down": 25, "volume_up": 24, "camera": 27}
CAMERA_DIR = "/sdcard/DCIM/Camera"
CAMERA_APP = "com.android.camera"  # the camera app of the Poco X7 Pro (HyperOS)
OPEN_S = 2.0                       # the camera app ready after it was started
MODE_S = 1.5                       # the camera app in its other mode after the intent
POLL_S = 0.4                       # between two looks at the camera folder
PHONE_SETTLE_S = 1.0               # a new file on the phone is whole once unchanged this long (and not pending)
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

    def foreground(self) -> str:
        """Package of the app on screen ('' if unknown)."""
        try:
            out = self._shell("dumpsys", "activity", "activities")
        except AcquisitionError:
            return ""
        m = re.search(r"(?:mResumedActivity|topResumedActivity|ResumedActivity)[^\n]*?\s([\w.]+)/", out)
        return m.group(1) if m else ""

    def listing(self) -> dict:
        """{name: (size, mtime)} of the camera folder, the media store's pending files (.pending-*) included."""
        d = shlex.quote(self.camera_dir)
        out = self._shell("stat", "-c", "%n|%s|%Y", tail=f"{d}/* {d}/.pending-* 2>/dev/null; true")
        files = {}
        for line in out.splitlines():
            parts = line.strip().rsplit("|", 2)
            if len(parts) == 3 and parts[1].isdigit():
                files[parts[0].rsplit("/", 1)[-1]] = (int(parts[1]), int(parts[2] or 0))
        return files

    def press(self) -> None:
        self._shell("input", "keyevent", "KEYCODE_WAKEUP")
        self._shell("input", "keyevent", str(KEYCODES.get(self.shutter, self.shutter)))

    def open_camera(self) -> bool:
        """Brings the camera app back when another app is on screen: the live view of ImpactGuide
        closes it to free the camera (live_ar.py). Left alone when the app on screen is unknown."""
        self._shell("input", "keyevent", "KEYCODE_WAKEUP")
        front = self.foreground()
        if front and front != CAMERA_APP:
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

    def _wait(self, before: dict, suffixes: Sequence[str], timeout: float) -> str:
        sizes, deadline = {}, time.monotonic() + timeout
        while time.monotonic() < deadline:
            name = new_stable_file(self.listing(), before, sizes, suffixes, time.monotonic(), self.settle_s)
            if name and not self._pending(name):
                return name
            if name:
                sizes.pop(name, None)       # pending: whole only after it stays unchanged once more
            time.sleep(POLL_S)
        raise AcquisitionError("timeout", "no new file in the camera folder")

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

    def pull(self, name: str, out_dir: Path | str) -> Path:
        """The file copied into out_dir and checked against the phone's (size and SHA-256): it lands as a hidden
        .part file and gets its name only once checked, then it is deleted from the phone (remove) unless clear is
        off. A copy that fails or differs is deleted here, the file stays on the phone and the cause is
        copy_failed."""
        if not SAFE_NAME.fullmatch(name):
            raise AcquisitionError("not_found", name)
        with self._lock:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dest, part = out_dir / name, out_dir / f".{name}{PART}"
            part.unlink(missing_ok=True)
            try:
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

    def shoot(self, out_dir: Path | str, timeout: float = SHOT_TIMEOUT_S) -> Path:
        """One photo: shutter, wait for the file, pull it into out_dir. Returns the local path."""
        with self._lock:
            self.connect()
            self.open_camera()
            before = self.listing()
            self.press()
            return self.pull(self._wait(before, PHOTO_SUFFIXES, timeout), out_dir)

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
                return self.pull(self._wait(self._before_video, VIDEO_SUFFIXES, timeout), out_dir)
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
        self._recording = False
        self._n = 0

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

    def shoot(self, out_dir, timeout=SHOT_TIMEOUT_S):
        import cv2
        import numpy as np
        self._check()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        w, h = self.photo_px
        if self.scene is not None:
            img = self.scene()
        else:
            img = np.full((h, w, 3), 96, np.uint8)
            cv2.putText(img, f"simulated photo {datetime.now():%H:%M:%S}", (w // 12, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, w / 900, (230, 230, 230), max(2, w // 600))
        path = out_dir / self._name("IMG", ".jpg")
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92 if self.scene is not None else 70])
        return path

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
    deaf makes the camera ignore the next press. Every shell line is read as the phone's sh reads it."""

    def __init__(self, folder, photo_bytes=40000):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.media = {f"/storage/emulated/0/DCIM/Camera/{p.name}" for p in self.folder.iterdir()}
        self.photo_bytes, self.corrupt, self.fail_pull, self.deaf = photo_bytes, False, False, False
        self.video_mode, self.recording, self.n, self.calls = False, None, 0, []
        self.pending = set()            # media store paths still being written
        self.lines = []                 # shell lines as adb passes them

    def _local(self, path):
        name = path.strip("'").rsplit("/", 1)[-1]
        return self.folder / name

    def _bytes(self):
        import random
        return bytes(random.getrandbits(8) for _ in range(self.photo_bytes))

    def _new(self, suffix):
        self.n += 1
        p = self.folder / f"{'IMG' if suffix == '.jpg' else 'VID'}_20261016_1200{self.n:02d}{suffix}"
        p.write_bytes(self._bytes())
        self.media.add(f"/storage/emulated/0/DCIM/Camera/{p.name}")
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
            return 0, f"  mResumedActivity: ActivityRecord{{1 u0 {CAMERA_APP}/.Camera t1}}\n", ""
        if words[:2] == ["am", "start"]:
            self.video_mode = "android.media.action.VIDEO_CAMERA" in words
            return 0, "", ""
        if words[:2] == ["input", "keyevent"] and words[-1] != "KEYCODE_WAKEUP":
            if self.deaf:
                self.deaf = False
                return 0, "", ""
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
                self._new(".jpg")
            return 0, "", ""
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
