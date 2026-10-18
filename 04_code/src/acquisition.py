"""Raw files during a trial: the phone over ADB, by cable or Wi-Fi, and the scanner's export folder (T20.1).

Phone. The camera app stays open on the phone in its cradle and the phone is
never touched: a key event sent over ADB is the shutter (the volume key, set
as the shutter in the camera app), the new file in the camera folder is waited
for until its size stops growing, and it is pulled into a staging folder with
the name the phone gave it. Nothing stays on the phone: once the copy on the
PC has the size and the SHA-256 of the file on the phone, the file is deleted
there and taken out of Android's media store, so the gallery does not show it
(pull, remove). A copy that fails or differs is deleted on the PC instead, the
file stays on the phone and the cause is copy_failed. ImpactGuide then files it under the trial name
(trials.Run.store) or discards it, so nothing reaches 03_data/raw before the
operator keeps it. For the video the same key starts and stops the recording;
with switch_modes the camera app is first moved to its video mode and back to
its photo mode afterwards through the standard Android camera intents.
Which phone, on which link (cable or Wi-Fi), is phone_link.py's to say: every
operation asks it for the serial first, so a Wi-Fi link that came back on a
new port is followed, and a link that is down is said as such (wifi_lost).

The camera serves one at a time: the live view of ImpactGuide (live_ar.py)
closes the camera app before it opens the stream, and ImpactGuide stops the
live view before a photo or a video; the camera app is then started again
here when another app is on screen.

Scanner. The scanner software exports into a folder; ExportWatch reports the
first scan file that appears there (or is written again) after it started,
once its size has stopped growing.

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

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import phone_link

PHOTO_SUFFIXES = (".jpg", ".jpeg")
VIDEO_SUFFIXES = (".mp4",)
SCAN_SUFFIXES = (".ply", ".stl", ".obj", ".asc")
KEYCODES = {"volume_down": 25, "volume_up": 24, "camera": 27}
CAMERA_DIR = "/sdcard/DCIM/Camera"
CAMERA_APP = "com.android.camera"  # the camera app of the Poco X7 Pro (HyperOS)
OPEN_S = 2.0                       # the camera app ready after it was started
POLL_S = 0.4
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: no console flash under pythonw
MEDIA_URI = "content://media/external/file"      # Android's media store: what the gallery lists
SAFE_NAME = re.compile(r"[\w.\-]+")               # file names the camera gives; nothing else is ever deleted


def run_adb(cmd, timeout):
    """adb as a process: (return code, stdout, stderr)."""
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                       creationflags=NO_WINDOW)
    return r.returncode, r.stdout, r.stderr


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class AcquisitionError(RuntimeError):
    """A failure with a cause the interface can name: no_adb, no_device, unauthorized, many_devices,
    adb_failed, timeout, not_found, copy_failed (the file stayed on the phone), and the causes of the link
    (phone_link.CAUSES)."""

    def __init__(self, cause, detail=""):
        super().__init__(f"{cause}: {detail}" if detail else cause)
        self.cause, self.detail = cause, detail


def find_adb():
    """The adb executable of the whole app (phone_link.find_adb: IMPACTGUIDE_ADB, the one shipped with
    scrcpy, the PATH, adbutils, the Android SDK). None if there is none."""
    return phone_link.find_adb()


def new_stable_file(listing, before, sizes, suffixes):
    """One poll of a folder listing {name: (size, mtime)}: the first file that is new or rewritten
    since `before`, has an allowed suffix, is not hidden or pending, is not empty and has the same
    size as in the previous poll (sizes keeps the sizes across polls). None if there is none yet."""
    for name, (size, mtime) in sorted(listing.items(), key=lambda kv: kv[1][1]):
        if name.startswith(".") or "pending" in name.lower() or not name.lower().endswith(suffixes):
            continue
        if name in before and before[name] == (size, mtime):
            continue
        last, sizes[name] = sizes.get(name), size
        if size > 0 and last == size:
            return name
    return None


class Phone:
    """The phone in the cradle, over ADB, on the link chosen in the app (phone_link)."""

    def __init__(self, serial=None, camera_dir=CAMERA_DIR, shutter="volume_down", switch_modes=True, adb=None,
                 link=None, runner=None, clear=True):
        self.serial, self.camera_dir, self.shutter, self.switch_modes = serial, camera_dir, shutter, switch_modes
        self.fixed = serial         # given by hand: used as it is
        self.link = link or phone_link.get()
        self.adb = adb or find_adb()
        self.runner = runner or run_adb     # (cmd, timeout) -> (code, stdout, stderr); FakeAdb in the test
        self.clear = clear                  # files pulled are deleted from the phone
        self.left_on_phone = []             # files copied but not deleted there (the deletion failed)
        self._real_dir = None
        self._before_video = None

    def _run(self, *args, timeout=20, device=True):
        if not self.adb:
            raise AcquisitionError("no_adb")
        cmd = [self.adb] + (["-s", self.serial] if device and self.serial else []) + list(args)
        try:
            code, out, err = self.runner(cmd, timeout)
        except subprocess.TimeoutExpired:
            raise AcquisitionError("timeout", " ".join(args))
        if code != 0:
            detail = (err or out).strip()[-300:]
            if device and not self.fixed and not self.link.ready():
                # the phone went away in the middle: said as the link's state, not as adb's text
                raise AcquisitionError("wifi_lost" if self.link.via == phone_link.WIFI else "no_device", detail)
            raise AcquisitionError("adb_failed", detail)
        return out

    def devices(self):
        """[{serial, state, model, link}] of what adb sees; starts the adb server if needed."""
        try:
            return [{"serial": d["serial"], "state": d["state"], "model": d["model"], "link": d["transport"]}
                    for d in self.link.devices()]
        except phone_link.LinkError as e:
            raise AcquisitionError(e.cause, e.detail)

    def connect(self):
        """The phone on the chosen link (phone_link); the serial, or AcquisitionError with the cause."""
        if self.fixed:
            if not any(d["serial"] == self.fixed and d["state"] == "device" for d in self.devices()):
                raise AcquisitionError("no_device", self.fixed)
            self.serial = self.fixed
            return self.serial
        try:
            self.serial = self.link.serial()
        except phone_link.LinkError as e:
            raise AcquisitionError(e.cause, e.detail)
        return self.serial

    def status(self):
        """{ok, cause, serial, model, foreground, link}: never raises."""
        try:
            self.connect()
            model = next((d["model"] for d in self.devices() if d["serial"] == self.serial), "")
            return {"ok": True, "cause": "", "serial": self.serial, "model": model, "foreground": self.foreground(),
                    "link": self.link.via}
        except AcquisitionError as e:
            return {"ok": False, "cause": e.cause, "detail": e.detail, "serial": self.serial or "", "model": "",
                    "foreground": "", "link": self.link.via}

    def foreground(self):
        """Package of the app on screen ('' if unknown)."""
        try:
            out = self._run("shell", "dumpsys", "activity", "activities")
        except AcquisitionError:
            return ""
        m = re.search(r"(?:mResumedActivity|topResumedActivity|ResumedActivity)[^\n]*?\s([\w.]+)/", out)
        return m.group(1) if m else ""

    def listing(self):
        """{name: (size, mtime)} of the camera folder."""
        out = self._run("shell", f"stat -c '%n|%s|%Y' {self.camera_dir}/* 2>/dev/null; true")
        files = {}
        for line in out.splitlines():
            parts = line.strip().rsplit("|", 2)
            if len(parts) == 3 and parts[1].isdigit():
                files[parts[0].rsplit("/", 1)[-1]] = (int(parts[1]), int(parts[2] or 0))
        return files

    def press(self):
        self._run("shell", "input", "keyevent", "KEYCODE_WAKEUP")
        self._run("shell", "input", "keyevent", str(KEYCODES.get(self.shutter, self.shutter)))

    def open_camera(self):
        """Brings the camera app back when another app is on screen: the live view of ImpactGuide
        closes it to free the camera (live_ar.py). Left alone when the app on screen is unknown."""
        self._run("shell", "input", "keyevent", "KEYCODE_WAKEUP")
        front = self.foreground()
        if front and front != CAMERA_APP:
            self._run("shell", "am", "start", "-a", "android.media.action.STILL_IMAGE_CAMERA")
            time.sleep(OPEN_S)
            return True
        return False

    def _mode(self, action):
        if self.switch_modes:
            self._run("shell", "am", "start", "-a", f"android.media.action.{action}")
            time.sleep(1.5)

    def _wait(self, before, suffixes, timeout):
        sizes, t0 = {}, time.time()
        while time.time() - t0 < timeout:
            name = new_stable_file(self.listing(), before, sizes, suffixes)
            if name:
                return name
            time.sleep(POLL_S)
        raise AcquisitionError("timeout", "no new file in " + self.camera_dir)

    def real_dir(self):
        """The camera folder as the media store names it (/sdcard is a link to /storage/emulated/0)."""
        if self._real_dir is None:
            out = self._run("shell", f"readlink -f {self.camera_dir}").strip()
            self._real_dir = out if out.startswith("/") else self.camera_dir
        return self._real_dir

    def remote(self, name):
        """(size, SHA-256) of a file in the camera folder."""
        size = self._run("shell", f"stat -c %s '{self.camera_dir}/{name}'").strip()
        digest = self._run("shell", f"sha256sum '{self.camera_dir}/{name}'").split()
        if not size.isdigit() or not digest:
            raise AcquisitionError("not_found", name)
        return int(size), digest[0].lower()

    def in_media(self, name):
        """Whether the media store (the gallery) still lists the file."""
        out = self._run("shell", f"content query --uri {MEDIA_URI} --projection _id "
                                 f"--where \"_data='{self.real_dir()}/{name}'\"")
        return "Row:" in out

    def remove(self, name):
        """The file deleted from the phone and taken out of the media store, so that the gallery does not show it.
        True when both are gone (checked); False otherwise, and the file is noted in left_on_phone."""
        if not SAFE_NAME.fullmatch(name):
            self.left_on_phone.append(name)
            return False
        path = f"{self.camera_dir}/{name}"
        try:
            self._run("shell", f"rm -f '{path}'")
            try:
                self._run("shell", f"content delete --uri {MEDIA_URI} --where \"_data='{self.real_dir()}/{name}'\"")
            except AcquisitionError:
                # without the media store's own delete, a scan of the path that is gone takes it out
                self._run("shell", "am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE "
                                   f"-d file://{self.real_dir()}/{name}")
            gone = name not in self.listing() and not self.in_media(name)
        except AcquisitionError:
            gone = False
        if not gone:
            self.left_on_phone.append(name)
        return gone

    def pull(self, name, out_dir):
        """The file copied into out_dir and checked against the phone's (size and SHA-256), then deleted from the
        phone (remove) unless clear is off. A copy that fails or differs is deleted here, the file stays on the phone
        and the cause is copy_failed."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / name
        if not SAFE_NAME.fullmatch(name):
            raise AcquisitionError("not_found", name)
        try:
            size, digest = self.remote(name)
            self._run("pull", f"{self.camera_dir}/{name}", str(dest), timeout=180)
            same = dest.is_file() and dest.stat().st_size == size and sha256(dest) == digest
        except AcquisitionError as e:
            dest.unlink(missing_ok=True)
            raise AcquisitionError("copy_failed", name) from e
        if not same:
            dest.unlink(missing_ok=True)
            raise AcquisitionError("copy_failed", name)
        if self.clear:
            self.remove(name)
        return dest

    def shoot(self, out_dir, timeout=20):
        """One photo: shutter, wait for the file, pull it into out_dir. Returns the local path."""
        self.connect()
        self.open_camera()
        before = self.listing()
        self.press()
        return self.pull(self._wait(before, PHOTO_SUFFIXES, timeout), out_dir)

    def start_video(self):
        self.connect()
        self.open_camera()
        self._mode("VIDEO_CAMERA")
        self._before_video = self.listing()
        self.press()

    def stop_video(self, out_dir, timeout=60):
        """Stops the recording started by start_video and pulls the video into out_dir."""
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
        self._recording = False
        self._n = 0

    def _check(self):
        if self.link is not None:
            try:
                self.link.serial()
            except phone_link.LinkError as e:
                raise AcquisitionError(e.cause, e.detail)

    def _name(self, prefix, suffix):
        self._n += 1
        return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{self._n:03d}{suffix}"

    def status(self):
        mode = self.link.via if self.link is not None else phone_link.USB
        try:
            self._check()
        except AcquisitionError as e:
            return {"ok": False, "cause": e.cause, "serial": "", "model": "", "foreground": "", "link": mode}
        return {"ok": True, "cause": "", "serial": "simulated", "model": "simulated phone", "foreground": "simulated",
                "link": mode}

    def shoot(self, out_dir, timeout=20):
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
        self._recording = True

    def stop_video(self, out_dir, timeout=60):
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
    set of paths for the media store. The shutter key adds a photo (a JPEG of `photo_bytes` random bytes), the second
    press of a video recording adds a video. corrupt makes the next pull write a file cut short, fail_pull makes it
    fail."""

    def __init__(self, folder, photo_bytes=40000):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.media = {f"/storage/emulated/0/DCIM/Camera/{p.name}" for p in self.folder.iterdir()}
        self.photo_bytes, self.corrupt, self.fail_pull = photo_bytes, False, False
        self.video_mode, self.recording, self.n, self.calls = False, False, 0, []

    def _local(self, path):
        name = path.strip("'").rsplit("/", 1)[-1]
        return self.folder / name

    def _new(self, suffix):
        import random
        self.n += 1
        p = self.folder / f"{'IMG' if suffix == '.jpg' else 'VID'}_20261016_1200{self.n:02d}{suffix}"
        p.write_bytes(bytes(random.getrandbits(8) for _ in range(self.photo_bytes)))
        self.media.add(f"/storage/emulated/0/DCIM/Camera/{p.name}")

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
        if c.startswith("stat -c '%n|%s|%Y'"):
            rows = [f"/sdcard/DCIM/Camera/{p.name}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
                    for p in sorted(self.folder.iterdir())]
            return 0, "\n".join(rows) + "\n", ""
        if c.startswith("stat -c %s"):
            p = self._local(c.split()[-1])
            return (0, f"{p.stat().st_size}\n", "") if p.is_file() else (1, "", "No such file")
        if c.startswith("sha256sum"):
            p = self._local(c.split()[-1])
            return (0, f"{sha256(p)}  {c.split()[-1]}\n", "") if p.is_file() else (1, "", "No such file")
        if c.startswith("readlink -f"):
            return 0, "/storage/emulated/0/DCIM/Camera\n", ""
        if c.startswith("rm -f"):
            self._local(c.split()[-1]).unlink(missing_ok=True)
            return 0, "", ""
        if c.startswith("content "):
            m = re.search(r"_data='([^']+)'", c)
            path = m.group(1) if m else ""
            if c.startswith("content delete"):
                self.media.discard(path)
                return 0, "", ""
            return 0, ("Row: 0 _id=1\n" if path in self.media else "No result found.\n"), ""
        if c.startswith("dumpsys activity"):
            return 0, f"  mResumedActivity: ActivityRecord{{1 u0 {CAMERA_APP}/.Camera t1}}\n", ""
        if c.startswith("am start"):
            self.video_mode = "VIDEO_CAMERA" in c
            return 0, "", ""
        if c.startswith("input keyevent") and not c.endswith("KEYCODE_WAKEUP"):
            if self.video_mode:
                if self.recording:
                    self._new(".mp4")
                self.recording = not self.recording
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
    """The scanner's export folder: poll() returns the first scan file written there after the
    watch started, once its size has stopped growing; None until then."""

    def __init__(self, folder, suffixes=SCAN_SUFFIXES):
        self.folder, self.suffixes = Path(folder), tuple(suffixes)
        self.before = folder_listing(self.folder)
        self.sizes = {}

    def poll(self):
        name = new_stable_file(folder_listing(self.folder), self.before, self.sizes, self.suffixes)
        return self.folder / name if name else None


def fake_scan(folder, name=None):
    """A small ASCII PLY standing in for an exported scan."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name or f"scan_{datetime.now():%Y%m%d_%H%M%S_%f}.ply")
    pts = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)]
    head = f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    path.write_text(head + "".join(f"{x} {y} {z}\n" for x, y, z in pts), encoding="ascii")
    return path


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

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        phone = FakePhone(photo_px=(640, 480), video_s=2)
        photo = phone.shoot(tmp / "staging")
        phone.start_video()
        video = phone.stop_video(tmp / "staging")
        import cv2
        img = cv2.imread(str(photo))
        cap = cv2.VideoCapture(str(video))
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        expect(img is not None and img.shape[:2] == (480, 640) and photo.name.startswith("IMG_"),
               f"fake photo {photo.name}")
        expect(fps > 0 and abs(frames / fps - 2) < 0.2 and video.name.startswith("VID_"), f"fake video {video.name}")

        export = tmp / "export"
        export.mkdir()
        fake_scan(export, "old.ply")
        watch = ExportWatch(export)
        expect(watch.poll() is None, "export folder: nothing new yet")
        scan = fake_scan(export, "t001_after.ply")
        got = [watch.poll(), watch.poll()]
        expect(got[0] is None and got[1] == scan, "export folder: the new scan once its size is stable")
        fake_scan(export, "old.ply").write_text("ply\nrewritten\n", encoding="ascii")
        w2 = ExportWatch(export)
        expect(w2.poll() is None, "export folder: files already there are not new")

    # Phone itself on a fake adb: a photo and a video copied, checked and deleted from the phone and the gallery;
    # a copy that fails or differs leaves the file on the phone
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        camera = tmp / "phone_camera"
        camera.mkdir()
        (camera / "IMG_20261001_090000.jpg").write_bytes(b"an old photo of Federico's")
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
        expect(video.suffix == ".mp4" and video.is_file() and folder_listing(camera) == before and adb.media == media,
               f"video {video.name}: on the PC, gone from the phone and the gallery")
        for what in ("corrupt", "fail_pull"):
            setattr(adb, what, True)
            n = len(list(camera.iterdir()))
            try:
                phone.shoot(tmp / "pc")
                cause = ""
            except AcquisitionError as e:
                cause = e.cause
            left = [p.name for p in camera.iterdir() if p.name not in before]
            on_pc = [p.name for p in (tmp / "pc").iterdir() if p.name in left]
            expect(cause == "copy_failed" and len(left) == 1 and len(list(camera.iterdir())) == n + 1 and not on_pc
                   and f"/storage/emulated/0/DCIM/Camera/{left[0]}" in adb.media,
                   f"a copy that {'differs' if what == 'corrupt' else 'fails'}: copy_failed, {left} stays on the "
                   f"phone and in the gallery, nothing half copied on the PC")
            for p in camera.iterdir():          # the next case starts from the folder as it was
                if p.name not in before:
                    p.unlink()
                    adb.media.discard(f"/storage/emulated/0/DCIM/Camera/{p.name}")
        rm = [c for c in adb.calls if c[0] == "shell" and c[1].startswith("rm -f")]
        expect(all(re.fullmatch(r"rm -f '/sdcard/DCIM/Camera/(IMG|VID)_[\w]+\.(jpg|mp4)'", c[1]) for c in rm)
               and len(rm) == 2, f"deleted on the phone: only the two files the app took ({len(rm)} rm)")

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
    expect(find_adb() is not None, f"adb found: {find_adb()}")
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
        phone._run("push", str(src), f"{phone.camera_dir}/{name}", timeout=60)
        listed = False
        for _ in range(20):         # the media store takes in a file written over adb by itself
            listed = phone.in_media(name)
            if listed:
                break
            phone._run("shell", f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{real}")
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


if __name__ == "__main__":
    main()
