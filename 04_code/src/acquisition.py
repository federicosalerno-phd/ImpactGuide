"""Raw files during a trial: the phone over USB (ADB) and the scanner's export folder (T20.1).

Phone. The camera app stays open on the phone in its cradle and the phone is
never touched: a key event sent over ADB is the shutter (the volume key, set
as the shutter in the camera app), the new file in the camera folder is waited
for until its size stops growing, and it is pulled into a staging folder with
the name the phone gave it. ImpactGuide then files it under the trial name
(trials.Run.store) or discards it, so nothing reaches 03_data/raw before the
operator keeps it. For the video the same key starts and stops the recording;
with switch_modes the camera app is first moved to its video mode and back to
its photo mode afterwards through the standard Android camera intents.

Scanner. The scanner software exports into a folder; ExportWatch reports the
first scan file that appears there (or is written again) after it started,
once its size has stopped growing.

FakePhone and fake_scan write stand-in files, for the tests and for the
simulation mode of ImpactGuide.

Usage:
    python acquisition.py devices                 phones seen over ADB
    python acquisition.py shoot [--out DIR]       one photo from the phone into DIR
    python acquisition.py test                    fake phone and export folder; exit 1 on failure
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PHOTO_SUFFIXES = (".jpg", ".jpeg")
VIDEO_SUFFIXES = (".mp4",)
SCAN_SUFFIXES = (".ply", ".stl", ".obj", ".asc")
KEYCODES = {"volume_down": 25, "volume_up": 24, "camera": 27}
CAMERA_DIR = "/sdcard/DCIM/Camera"
POLL_S = 0.4
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: no console flash under pythonw


class AcquisitionError(RuntimeError):
    """A failure with a cause the interface can name: no_adb, no_device, unauthorized, many_devices,
    adb_failed, timeout, not_found."""

    def __init__(self, cause, detail=""):
        super().__init__(f"{cause}: {detail}" if detail else cause)
        self.cause, self.detail = cause, detail


def find_adb():
    """The adb executable: IMPACTGUIDE_ADB, the PATH, the one bundled with the adbutils package,
    the Android SDK. None if there is none."""
    candidates = [os.environ.get("IMPACTGUIDE_ADB"), shutil.which("adb")]
    try:
        import adbutils
        candidates.append(str(Path(adbutils.__file__).parent / "binaries" / ("adb.exe" if os.name == "nt" else "adb")))
    except ImportError:
        pass
    sdk = os.environ.get("ANDROID_HOME") or os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk")
    candidates.append(os.path.join(sdk, "platform-tools", "adb.exe" if os.name == "nt" else "adb"))
    return next((c for c in candidates if c and Path(c).is_file()), None)


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
    """The phone in the cradle, over ADB."""

    def __init__(self, serial=None, camera_dir=CAMERA_DIR, shutter="volume_down", switch_modes=True, adb=None):
        self.serial, self.camera_dir, self.shutter, self.switch_modes = serial, camera_dir, shutter, switch_modes
        self.adb = adb or find_adb()
        self._before_video = None

    def _run(self, *args, timeout=20, device=True):
        if not self.adb:
            raise AcquisitionError("no_adb")
        cmd = [self.adb] + (["-s", self.serial] if device and self.serial else []) + list(args)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, creationflags=NO_WINDOW)
        except subprocess.TimeoutExpired:
            raise AcquisitionError("timeout", " ".join(args))
        if r.returncode != 0:
            raise AcquisitionError("adb_failed", (r.stderr or r.stdout).strip()[-300:])
        return r.stdout

    def devices(self):
        """[{serial, state, model}] of what adb sees; starts the adb server if needed."""
        out = self._run("devices", "-l", timeout=30, device=False)
        found = []
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                model = next((p.split(":", 1)[1] for p in parts[2:] if p.startswith("model:")), "")
                found.append({"serial": parts[0], "state": parts[1], "model": model.replace("_", " ")})
        return found

    def connect(self):
        """Picks the one phone that is ready; the serial, or AcquisitionError with the cause."""
        found = self.devices()
        ready = [d for d in found if d["state"] == "device"]
        if self.serial and any(d["serial"] == self.serial for d in ready):
            return self.serial
        if not found:
            raise AcquisitionError("no_device")
        if not ready:
            raise AcquisitionError("unauthorized" if any(d["state"] == "unauthorized" for d in found) else "no_device",
                                   found[0]["state"])
        if len(ready) > 1:
            raise AcquisitionError("many_devices", ", ".join(d["serial"] for d in ready))
        self.serial = ready[0]["serial"]
        return self.serial

    def status(self):
        """{ok, cause, serial, model, foreground}: never raises."""
        try:
            self.connect()
            model = next((d["model"] for d in self.devices() if d["serial"] == self.serial), "")
            return {"ok": True, "cause": "", "serial": self.serial, "model": model, "foreground": self.foreground()}
        except AcquisitionError as e:
            return {"ok": False, "cause": e.cause, "detail": e.detail, "serial": self.serial or "", "model": "",
                    "foreground": ""}

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

    def pull(self, name, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / name
        self._run("pull", f"{self.camera_dir}/{name}", str(dest), timeout=180)
        if not dest.is_file():
            raise AcquisitionError("not_found", name)
        return dest

    def shoot(self, out_dir, timeout=20):
        """One photo: shutter, wait for the file, pull it into out_dir. Returns the local path."""
        self.connect()
        before = self.listing()
        self.press()
        return self.pull(self._wait(before, PHOTO_SUFFIXES, timeout), out_dir)

    def start_video(self):
        self.connect()
        self._mode("VIDEO_CAMERA")
        self._before_video = self.listing()
        self.press()

    def stop_video(self, out_dir, timeout=60):
        """Stops the recording started by start_video and pulls the video into out_dir."""
        if self._before_video is None:
            raise AcquisitionError("not_found", "no recording started")
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
    the time written on it, and an 8 s video."""

    def __init__(self, photo_px=(4080, 3072), video_s=8.0):
        self.photo_px, self.video_s = photo_px, video_s
        self._recording = False
        self._n = 0

    def _name(self, prefix, suffix):
        self._n += 1
        return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{self._n:03d}{suffix}"

    def status(self):
        return {"ok": True, "cause": "", "serial": "simulated", "model": "simulated phone", "foreground": "simulated"}

    def shoot(self, out_dir, timeout=20):
        import cv2
        import numpy as np
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        w, h = self.photo_px
        img = np.full((h, w, 3), 96, np.uint8)
        cv2.putText(img, f"simulated photo {datetime.now():%H:%M:%S}", (w // 12, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, w / 900, (230, 230, 230), max(2, w // 600))
        path = out_dir / self._name("IMG", ".jpg")
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return path

    def start_video(self):
        self._recording = True

    def stop_video(self, out_dir, timeout=60):
        import cv2
        import numpy as np
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

    expect(find_adb() is not None, f"adb found: {find_adb()}")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed")
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices")
    a = sub.add_parser("shoot")
    a.add_argument("--out", type=Path, default=Path("."))
    sub.add_parser("test")
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


if __name__ == "__main__":
    main()
