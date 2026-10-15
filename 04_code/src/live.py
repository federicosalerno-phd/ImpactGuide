"""Live video from the phone camera to the PC over USB, for the live AR page of ImpactGuide (T20.3, ex T5.5).

The phone runs the scrcpy server (shipped with scrcpy, which also brings adb) with the back camera as video source;
the stream comes over an adb tunnel as raw H.264 (no scrcpy metadata), is decoded here with PyAV and only the newest
frame is kept, with the time it was decoded. Nothing is installed on the phone and it needs no app: USB debugging only.

For the app page:
    s = open_stream(size="1920x1440", fps=30)    # starts the server, returns once the first frame is in
    frame, t, n = s.latest()                     # BGR array, time.perf_counter() of decoding, frame counter
    frame, t, n = s.wait_frame(after=n)          # the next one, as soon as it is decoded
    s.stats()                                    # fps, mean decode interval and decoding time, frames
    s.close()

Latency (camera to decoded frame) is measured with `latency`: the PC screen flashes, the phone looks at it, and the
time from each flash to the first frame that sees it is taken (includes the screen's own delay, a few ms).

    python live.py devices                       phones seen over adb
    python live.py cameras                       cameras of the phone and their sizes (scrcpy --list-camera-sizes)
    python live.py view [--size WxH] [--fps N]   window with the stream, fps and ChArUco corners
    python live.py latency [--size WxH]          flashes the screen, the phone films it
    python live.py bench [--size WxH] [--seconds S]   fps and decode interval without a window
    python live.py record OUT [--seconds S]      the raw H.264 of the stream (OUT.h264) and the decoding time of each
                                                 frame (OUT_times.npy), to replay the live AR on it
                                                 (live_ar.py view --replay OUT.h264, live_ar.py bench --replay ...)

Environment: IMPACTGUIDE_SCRCPY = folder with scrcpy-server and adb.exe (default: found on the PATH or in the WinGet
packages folder).
"""
import argparse
import glob
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

SERVER_REMOTE = "/data/local/tmp/scrcpy-server.jar"
SCRCPY_VERSION = "4.1"          # the server refuses a client of another version
PORT = 27183
BIT_RATE = 20_000_000           # high, so that compression does not move the ChArUco corners
RESTARTS = 3                    # server restarts after the stream broke while running


class LiveError(RuntimeError):
    pass


def find_scrcpy():
    """Folder holding scrcpy-server and adb.exe."""
    cands = [os.environ.get("IMPACTGUIDE_SCRCPY")]
    exe = shutil.which("scrcpy")
    if exe:
        cands.append(str(Path(exe).resolve().parent))
    local = os.environ.get("LOCALAPPDATA", "")
    cands += sorted(glob.glob(os.path.join(local, "Microsoft", "WinGet", "Packages", "Genymobile.scrcpy*", "scrcpy-win64-*")),
                    reverse=True)
    for c in cands:
        if c and (Path(c) / "scrcpy-server").is_file() and (Path(c) / "adb.exe").is_file():
            return Path(c)
    raise LiveError("scrcpy not found: winget install Genymobile.scrcpy, or set IMPACTGUIDE_SCRCPY")


def adb(*args, serial=None, timeout=30, check=True):
    cmd = [str(find_scrcpy() / "adb.exe")] + (["-s", serial] if serial else []) + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if check and r.returncode != 0:
        raise LiveError(f"adb {' '.join(args)}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout


def devices():
    """[(serial, state, model)] of the phones adb sees (state "device" = ready, "unauthorized" = accept on the phone)."""
    out = []
    for line in adb("devices", "-l").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            model = next((p.split(":", 1)[1] for p in parts[2:] if p.startswith("model:")), "")
            out.append((parts[0], parts[1], model))
    return out


def pick_serial(serial=None):
    ready = [d for d in devices() if d[1] == "device"]
    if serial:
        return serial
    if not ready:
        raise LiveError("no phone ready over adb (cable, USB debugging, authorisation on the phone)")
    if len(ready) > 1:
        raise LiveError(f"more than one phone: {[d[0] for d in ready]}")
    return ready[0][0]


def list_cameras(serial=None):
    """Text of scrcpy --list-camera-sizes (cameras, sizes, fps)."""
    exe = find_scrcpy() / "scrcpy.exe"
    r = subprocess.run([str(exe), "-s", pick_serial(serial), "--list-camera-sizes"], capture_output=True, text=True,
                       timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.stdout + r.stderr


class PhoneStream:
    """Back camera of the phone over USB; a reader thread decodes and keeps only the newest frame."""

    def __init__(self, serial=None, size="1920x1440", fps=30, bit_rate=BIT_RATE, camera_id=None, port=PORT,
                 restarts=RESTARTS, log=None, record=None):
        self.serial = pick_serial(serial)
        self.record = record            # an open binary file that gets the raw H.264 (`record`)
        self.frame_times = []           # decoding time of every frame while recording
        self.size, self.fps, self.bit_rate, self.camera_id, self.port = size, fps, bit_rate, camera_id, port
        self.max_restarts, self.restarts = restarts, 0
        self.scid = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        self.log = log or (lambda msg: None)
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._frame, self._t, self._n = None, 0.0, 0
        self._times = deque(maxlen=120)
        self._decode_ms = deque(maxlen=120)
        self._stop = threading.Event()
        self.server, self.sock, self.thread, self.error = None, None, None, None
        self.server_lines = []

    # -- start and stop
    def start(self, timeout=15.0):
        name = f"scrcpy_{self.scid:08x}"
        adb("forward", f"tcp:{self.port}", f"localabstract:{name}", serial=self.serial)
        self._launch()
        self.thread = threading.Thread(target=self._read, args=(timeout,), daemon=True)
        self.thread.start()
        t0 = time.perf_counter()
        while self._frame is None:
            if self.error or self.server.poll() is not None or time.perf_counter() - t0 > timeout:
                msg = self.error or (f"server exited ({self.server.returncode})" if self.server.poll() is not None
                                     else "no frame within the timeout")
                self.close()
                raise LiveError(f"{msg}; server log: {' | '.join(self.server_lines[-6:])}")
            time.sleep(0.02)
        return self

    def close(self):
        self._stop.set()
        with self._new:
            self._new.notify_all()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        if self.server and self.server.poll() is None:
            self.server.terminate()
        try:
            adb("forward", "--remove", f"tcp:{self.port}", serial=self.serial, check=False)
        except (LiveError, subprocess.TimeoutExpired):
            pass

    def _launch(self):
        """Copy the server to the phone (cleanup=true deletes it when the previous server exits) and start it; it
        lives as long as this adb shell."""
        d = find_scrcpy()
        adb("push", str(d / "scrcpy-server"), SERVER_REMOTE, serial=self.serial)
        args = [f"scid={self.scid:08x}", "log_level=info", "video_source=camera", "audio=false", "control=false",
                "tunnel_forward=true", "raw_stream=true", "video_codec=h264", f"video_bit_rate={self.bit_rate}",
                f"camera_size={self.size}", f"camera_fps={self.fps}", "cleanup=true"]
        args.append(f"camera_id={self.camera_id}" if self.camera_id is not None else "camera_facing=back")
        cmd = [str(d / "adb.exe"), "-s", self.serial, "shell", f"CLASSPATH={SERVER_REMOTE}", "app_process", "/",
               "com.genymobile.scrcpy.Server", SCRCPY_VERSION] + args
        self.server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._server_log, args=(self.server,), daemon=True).start()

    @property
    def alive(self):
        """False once the reader has given up (error set and no restart left)."""
        return self.thread is not None and self.thread.is_alive()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # -- what the page reads
    def latest(self):
        """(BGR frame, time.perf_counter() when it was decoded, frame counter); frame None before the first."""
        with self._lock:
            return self._frame, self._t, self._n

    def wait_frame(self, after=0, timeout=0.1):
        """The newest frame once its counter is past `after`: (frame, t, n), or (None, 0.0, after) on timeout."""
        with self._new:
            self._new.wait_for(lambda: self._n > after or self._stop.is_set(), timeout=timeout)
            if self._n <= after:
                return None, 0.0, after
            return self._frame, self._t, self._n

    def stats(self):
        """Measured fps over the last frames, mean and 95th percentile of the decode interval (ms), mean time to
        decode and convert a frame (ms), frame count."""
        with self._lock:
            t = np.array(self._times)
            n = self._n
            dec = float(np.mean(self._decode_ms)) if self._decode_ms else float("nan")
        if len(t) < 2:
            return {"fps": 0.0, "interval_ms": float("nan"), "interval_p95_ms": float("nan"), "decode_ms": dec,
                    "frames": n}
        dt = np.diff(t) * 1e3
        return {"fps": float((len(t) - 1) / (t[-1] - t[0])), "interval_ms": float(dt.mean()),
                "interval_p95_ms": float(np.percentile(dt, 95)), "decode_ms": dec, "frames": n}

    # -- threads
    def _server_log(self, server):
        for line in server.stdout:
            line = line.strip()
            if line:
                self.server_lines.append(line)
                self.log(line)

    def _connect(self, timeout):
        """The adb tunnel accepts at once even before the server listens; then it closes: retry until data comes."""
        t0 = time.perf_counter()
        while not self._stop.is_set() and time.perf_counter() - t0 < timeout:
            s = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
            s.settimeout(2.0)
            try:
                first = s.recv(1 << 16)
                if first:
                    return s, first
            except socket.timeout:
                pass
            s.close()
            time.sleep(0.1)
        raise LiveError("no stream from the server")

    def _read(self, timeout):
        """Decode until stopped. When the stream breaks after it has run (another app took the camera, the server
        died) the server is started again, at most max_restarts times; .error holds the last failure."""
        import av
        while not self._stop.is_set():
            try:
                self.sock, data = self._connect(timeout)
                self.sock.settimeout(None)
                codec = av.CodecContext.create("h264", "r")
                codec.flags |= av.codec.context.Flags.low_delay
                while not self._stop.is_set():
                    if self.record is not None:
                        self.record.write(data)
                    for packet in codec.parse(data):
                        t0 = time.perf_counter()
                        for frame in codec.decode(packet):
                            img = frame.to_ndarray(format="bgr24")
                            t = time.perf_counter()
                            with self._new:
                                self._frame, self._t, self._n = img, t, self._n + 1
                                self._times.append(t)
                                self._decode_ms.append((t - t0) * 1e3)
                                self._new.notify_all()
                            if self.record is not None:
                                self.frame_times.append(t)
                            t0 = time.perf_counter()
                    data = self.sock.recv(1 << 18)
                    if not data:
                        raise LiveError("stream closed by the phone")
            except Exception as e:      # reported by start() or by the page through .error
                if self._stop.is_set():
                    return
                self.error = f"{type(e).__name__}: {e}; server: {' | '.join(self.server_lines[-3:])}"
                if self._frame is None or self.restarts >= self.max_restarts:
                    return
                self.restarts += 1
                self.log(f"restart {self.restarts}: {self.error}")
                try:
                    self.sock.close()
                except (OSError, AttributeError):
                    pass
                if self.server.poll() is None:
                    self.server.terminate()
                time.sleep(0.5)
                self._launch()


def open_stream(serial=None, size="1920x1440", fps=30, **kw):
    """Start the phone camera stream and return the PhoneStream once the first frame has arrived."""
    return PhoneStream(serial, size, fps, **kw).start()


# -- command line tools
def charuco_corners(img):
    import marker_pose as mp
    if not hasattr(charuco_corners, "det"):
        charuco_corners.det = mp.make_detector(mp.make_board())
    return mp.detect(charuco_corners.det, img)


def cmd_view(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    with PhoneStream(a.serial, a.size, a.fps) as s:
        last, detect = -1, True
        while True:
            img, t, n = s.latest()
            if n != last:
                last = n
                vis = img.copy()
                info = s.stats()
                txt = f"{img.shape[1]}x{img.shape[0]}  {info['fps']:.1f} fps  interval {info['interval_ms']:.0f} ms"
                if detect:
                    t0 = time.perf_counter()
                    c, ids, nm = charuco_corners(img)
                    txt += f"  ChArUco {len(ids)} corners, {nm} markers, {1e3 * (time.perf_counter() - t0):.0f} ms"
                    for p in c:
                        cv2.circle(vis, (int(p[0]), int(p[1])), 6, (90, 220, 60), 2)
                cv2.putText(vis, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 220, 60), 2)
                h = 900
                cv2.imshow("phone", cv2.resize(vis, (int(vis.shape[1] * h / vis.shape[0]), h)))
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord("d"):
                detect = not detect
            if s.error:
                print(s.error)
                break
    cv2.destroyAllWindows()


def cmd_bench(a):
    with PhoneStream(a.serial, a.size, a.fps) as s:
        img, _, _ = s.latest()
        time.sleep(a.seconds)
        st = s.stats()
    print(f"{img.shape[1]}x{img.shape[0]}: {st['fps']:.1f} fps, interval {st['interval_ms']:.1f} ms "
          f"(95th percentile {st['interval_p95_ms']:.1f}), {st['frames']} frames")


def cmd_record(a):
    """Raw H.264 of the stream from its first frame, and the decoding time of each frame (s from the first), so that
    a replay keeps the timing of the phone. The stream is recorded as it comes: no restart if it breaks."""
    out = Path(a.out).with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".h264", "wb") as raw:
        s = PhoneStream(a.serial, a.size, a.fps, record=raw, restarts=0)
        s.start()
        t0 = time.perf_counter()
        try:
            while time.perf_counter() - t0 < a.seconds and s.alive:
                time.sleep(0.5)
                print(f"{time.perf_counter() - t0:5.1f} s  {s.latest()[2]} frames", flush=True)
        finally:
            s.close()
            time.sleep(0.3)
    t = np.array(s.frame_times)
    np.save(str(out) + "_times.npy", t - t[0] if len(t) else t)
    print(f"{out}.h264: {len(t)} frames, {(len(t) - 1) / max(t[-1] - t[0], 1e-9):.1f} fps" if len(t) > 1
          else "no frame recorded")


def cmd_latency(a, flashes=12, period=0.8):
    """Full-screen window that turns white and black; the phone films its centre. Latency of each flash: from the
    screen change (after waitKey has shown it) to the first decoded frame whose brightness crossed half way between
    the settled levels of the two states (the last frame of each period). The first two flashes only set the levels."""
    with PhoneStream(a.serial, a.size, a.fps) as s:
        cv2.namedWindow("flash", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("flash", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        black, white = np.zeros((720, 1280), np.uint8), np.full((720, 1280), 255, np.uint8)
        levels = {0: [], 1: []}
        lat, state = [], 0
        cv2.imshow("flash", black)
        cv2.waitKey(500)
        for k in range(2 * flashes):
            state = 1 - state
            cv2.imshow("flash", white if state else black)
            cv2.waitKey(1)
            t_flip = time.perf_counter()
            seen, v = None, None
            mid = (np.median(levels[0]) + np.median(levels[1])) / 2 if levels[0] and levels[1] else None
            while time.perf_counter() - t_flip < period:
                img, t, n = s.latest()
                h, w = img.shape[:2]
                v = float(img[h // 3:2 * h // 3, w // 3:2 * w // 3].mean())
                if seen is None and mid is not None and t > t_flip and ((state and v > mid) or (not state and v < mid)):
                    seen = t - t_flip
                cv2.waitKey(1)
            levels[state].append(v)
            if seen is not None:
                lat.append(seen * 1e3)
        cv2.destroyAllWindows()
    if not lat:
        print("no flash seen: point the phone at the centre of the screen")
        return
    lat = np.array(lat)
    print(f"latency screen to decoded frame: median {np.median(lat):.0f} ms, range {lat.min():.0f}-{lat.max():.0f} ms "
          f"over {len(lat)} flashes (contrast {np.median(levels[1]) - np.median(levels[0]):.0f} grey levels)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["devices", "cameras", "view", "bench", "latency", "record"])
    ap.add_argument("out", nargs="?", help="record: the file to write (OUT.h264 and OUT_times.npy)")
    ap.add_argument("--serial")
    ap.add_argument("--size", default="1920x1440")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    try:
        if a.command == "devices":
            for d in devices():
                print(*d)
        elif a.command == "cameras":
            print(list_cameras(a.serial))
        else:
            if a.command == "record" and not a.out:
                ap.error("record needs OUT")
            {"view": cmd_view, "bench": cmd_bench, "latency": cmd_latency, "record": cmd_record}[a.command](a)
    except LiveError as e:
        sys.exit(f"live: {e}")


if __name__ == "__main__":
    main()
