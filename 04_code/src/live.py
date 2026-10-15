"""Live video from the phone camera to the PC, by cable or Wi-Fi, for the live AR page of ImpactGuide (T20.3, ex T5.5).

The phone runs the scrcpy server (shipped with scrcpy, which also brings adb) with the back camera as video source;
the stream comes over an adb tunnel as H.264 with scrcpy's frame meta (the capture time of each frame on the phone),
is decoded here with PyAV and only the newest frame is kept, with the time it was decoded and the time it was taken
(on this computer's clock). Frames that come in a burst are decoded but only the newest is passed on. Nothing is installed on the phone and it needs no app: USB debugging, or
wireless debugging for Wi-Fi. Which phone, on which link, is phone_link.py's to say (one place for the serial); over
Wi-Fi the bit rate is lower, and when the link drops the stream waits for phone_link to bring it back (up to
LINK_WAIT_S) and starts again on the serial the phone has then.

For the app page:
    s = open_stream(size="1920x1440", fps=30)    # starts the server, returns once the first frame is in
    frame, t, n = s.latest()                     # BGR array, time.perf_counter() of decoding, frame counter
    frame, t, n = s.wait_frame(after=n)          # the next one, as soon as it is decoded
    item = s.wait_item(after=n)                  # the same as a Frame: n, t, t_capture, size, gray(), bgr(width)
    s.stats()                                    # fps, mean decode interval and decoding time, frames
    s.close()

Latency (camera to decoded frame) is measured with `latency`: the PC screen flashes, the phone looks at it, and the
time from each flash to the first frame that sees it is taken (includes the screen's own delay, a few ms).

    python live.py devices                       phones seen over adb, by link
    python live.py cameras                       cameras of the phone and their sizes (scrcpy --list-camera-sizes)
    python live.py view [--size WxH] [--fps N]   window with the stream, fps and ChArUco corners
    python live.py latency [--size WxH]          flashes the screen, the phone films it
    python live.py bench [--size WxH] [--seconds S]   fps and decode interval without a window
    python live.py record OUT [--seconds S]      the raw H.264 of the stream (OUT.h264), the decoding time of each
                                                 frame (OUT_times.npy) and its capture time on the phone
                                                 (OUT_pts.npy), both in s from the first, to replay the live AR on it
                                                 (live_ar.py view --replay OUT.h264, live_ar.py bench --replay ...)

    python live.py test                          the reader without the phone (framing, bursts, capture times)

Every command takes --link usb|wifi (default usb; wifi: the phone paired in ImpactGuide, connected again at its last
address or the one it announces) or --serial.

Environment: IMPACTGUIDE_SCRCPY = folder with scrcpy-server and adb.exe (default: found on the PATH or in the WinGet
packages folder).
"""
import argparse
import os
import select
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import phone_link

SERVER_REMOTE = "/data/local/tmp/scrcpy-server.jar"
SCRCPY_VERSION = "4.1"          # the server refuses a client of another version
PORT = 27183
BIT_RATE = 20_000_000           # high, so that compression does not move the ChArUco corners
# per link: over Wi-Fi the camera stream shares the air with the AR view sent back to the phone (phone_view.py)
BIT_RATES = {phone_link.USB: BIT_RATE, phone_link.WIFI: 5_000_000}
# per link: 1920x1440 at 20 Mbit/s fills a cable but stalls a shared Wi-Fi link; 1280x960 keeps the board
# readable at 40-60 cm and leaves room for the AR view going back to the phone (15/10, Wi-Fi stalls)
SIZES = {phone_link.USB: "1920x1440", phone_link.WIFI: "1280x960"}
RESTARTS = 8                    # restarts in a row that bring no frame back before the stream gives up (about 20 s)
RESTART_GOOD_FRAMES = 30        # a restart that brought this many frames was a good one
# the phone's screen is kept on while the stream runs: when it goes off and comes back (the phone moved), the lock
# screen's face unlock opens the front camera and the system takes the back camera from the stream (Poco X7 Pro,
# 15/10). A key event that does nothing (KEYCODE_UNKNOWN) counts as user activity and keeps it on.
KEEP_AWAKE_S = 10.0
LINK_WAIT_S = 60.0              # after the link dropped, the stream waits this long for it to come back
STALL_S = 3.0                   # no data from the phone this long: the stream is broken (a Wi-Fi link can die silently)
HOLD_MS = 60.0                  # a decoded frame waits at most this long for the newer ones of a burst
OFFSET_WINDOW_S = 20.0          # capture times: the quickest arrival of this long is the reference
BEHIND_S, BEHIND_HOLD_S = 1.5, 3.0      # frames this late for this long: start again rather than lag behind
# options of the phone's H.264 encoder (MediaFormat keys, scrcpy's key:type=value list); empty: the defaults
# latency 1 (the encoder gives each frame out as soon as it can) and realtime priority: 169 to 149 ms from the screen to
# the decoded frame on the Poco X7 Pro (15/10, live.py latency)
CODEC_OPTIONS = "latency:int=1,priority:int=0"


class LiveError(RuntimeError):
    """A failure of the stream; `cause` is a link cause (phone_link.CAUSES) when the link is why."""

    def __init__(self, msg, cause=""):
        super().__init__(msg)
        self.cause = cause


def find_scrcpy():
    """Folder holding scrcpy-server and adb.exe."""
    d = phone_link.scrcpy_dir()
    if d is None:
        raise LiveError("scrcpy not found: winget install Genymobile.scrcpy, or set IMPACTGUIDE_SCRCPY", "no_adb")
    return d


def adb(*args, serial=None, timeout=30, check=True):
    """adb of the whole app (phone_link.find_adb: the one shipped with scrcpy)."""
    exe = phone_link.find_adb() or str(find_scrcpy() / "adb.exe")
    cmd = [exe] + (["-s", serial] if serial else []) + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if check and r.returncode != 0:
        raise LiveError(f"adb {' '.join(args)}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout


def devices():
    """[(serial, state, model, link)] of the phones adb sees (state "device" = ready, "unauthorized" = accept on the
    phone; link usb or wifi)."""
    try:
        return [(d["serial"], d["state"], d["model"], d["transport"]) for d in phone_link.get().devices()]
    except phone_link.LinkError as e:
        raise LiveError(f"adb: {e.cause}", e.cause)


def pick_serial(serial=None):
    """`serial` if given, else the phone on the link chosen in the app (phone_link: the one place that knows)."""
    if serial:
        return serial
    try:
        return phone_link.get().serial()
    except phone_link.LinkError as e:
        raise LiveError(f"no phone ready on the {phone_link.get().mode} link: {e.cause}", e.cause)


def list_cameras(serial=None):
    """Text of scrcpy --list-camera-sizes (cameras, sizes, fps)."""
    exe = find_scrcpy() / "scrcpy.exe"
    r = subprocess.run([str(exe), "-s", pick_serial(serial), "--list-camera-sizes"], capture_output=True, text=True,
                       timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.stdout + r.stderr


class Frame:
    """One decoded frame: counter n, decoding time t and capture time t_capture (time.perf_counter() seconds), size
    (w, h), session (it changes when the stream starts again or changes size). The pixels are converted only when
    asked, once: gray() is the luma plane at full size (what the board tracking needs, no colour conversion), bgr(width)
    the colour image scaled to that width in the same pass (what the overlay is drawn on)."""

    __slots__ = ("n", "t", "t_capture", "session", "_av", "_bgr", "_gray", "_scaled", "_lock")

    def __init__(self, n, t, t_capture=None, av_frame=None, bgr=None, session=0):
        self.n, self.t, self.session = n, t, session
        self.t_capture = t if t_capture is None else t_capture
        self._av, self._bgr, self._gray, self._scaled = av_frame, bgr, None, {}
        self._lock = threading.Lock()

    @property
    def size(self):
        if self._av is not None:
            return self._av.width, self._av.height
        h, w = self._bgr.shape[:2]
        return w, h

    def gray(self):
        if self._gray is None:
            if self._av is not None:
                plane = self._av.planes[0]
                w, h = self.size
                self._gray = np.frombuffer(plane, np.uint8).reshape(h, plane.line_size)[:, :w].copy()
            else:
                self._gray = self._bgr if self._bgr.ndim == 2 else cv2.cvtColor(self._bgr, cv2.COLOR_BGR2GRAY)
        return self._gray

    def bgr(self, width=None, keep=True):
        """The colour image (width None: full size); a scaled one keeps the aspect ratio. keep False: a scaled image
        the caller may draw on (not kept for other callers)."""
        w, h = self.size
        if width is None or int(width) >= w:
            with self._lock:
                if self._bgr is None:
                    self._bgr = self._av.to_ndarray(format="bgr24")
                elif self._bgr.ndim == 2:
                    return cv2.cvtColor(self._bgr, cv2.COLOR_GRAY2BGR)
                return self._bgr if keep else self._bgr.copy()
        width = int(width)
        height = int(round(h * width / w))
        with self._lock:
            img = self._scaled.get(width)
            if img is None:
                if self._av is not None:
                    img = self._av.reformat(width=width, height=height, format="bgr24",
                                            interpolation="BILINEAR").to_ndarray()
                else:
                    src = self._bgr if self._bgr.ndim == 3 else cv2.cvtColor(self._bgr, cv2.COLOR_GRAY2BGR)
                    img = cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)
                if keep:
                    self._scaled[width] = img
            elif not keep:
                img = img.copy()
            return img


class _Framing:
    """The scrcpy 4.1 video socket with frame meta: the codec id (4 bytes), then 12-byte headers. A session header
    (top bit set) carries the size of a new capture session; a packet header the capture time in microseconds with the
    config and key frame flags, and the size of the packet that follows. Config packets (SPS, PPS) are kept and put in
    front of the next frame's packet."""

    PTS_MASK = (1 << 61) - 1

    def __init__(self):
        self.buf = bytearray()
        self.codec = None
        self.config = b""

    def feed(self, data):
        """[("session", (w, h)) | ("frame", (pts_us, key, payload))] of the complete units in what has come so far."""
        b = self.buf
        b += data
        out, off = [], 0
        if self.codec is None:
            if len(b) < 4:
                return out
            self.codec, off = bytes(b[:4]), 4
        while len(b) - off >= 12:
            head = int.from_bytes(b[off:off + 8], "big")
            if head >> 63:
                out.append(("session", (int.from_bytes(b[off + 4:off + 8], "big"),
                                        int.from_bytes(b[off + 8:off + 12], "big"))))
                off += 12
                continue
            size = int.from_bytes(b[off + 8:off + 12], "big")
            if len(b) - off - 12 < size:
                break
            payload = bytes(b[off + 12:off + 12 + size])
            off += 12 + size
            if (head >> 62) & 1:
                self.config = payload
                continue
            if self.config:
                payload, self.config = self.config + payload, b""
            out.append(("frame", (head & self.PTS_MASK, bool((head >> 61) & 1), payload)))
        del b[:off]
        return out


class PhoneStream:
    """Back camera of the phone, by cable or Wi-Fi. A reader thread decodes every packet (H.264 needs them all) but
    passes on only the newest frame: when packets come in a burst (the computer was busy, Wi-Fi held them back) the
    older frames of the burst are dropped, never queued. Each frame carries its capture time from the phone's clock
    (frame meta of scrcpy), mapped to this computer's clock, so a burst does not bunch the times the tracking sees.
    When the stream breaks (another app took the camera, the phone was turned, the server died) it starts again by
    itself, quietly, as long as restarts bring frames back; when it stays behind (a slow Wi-Fi link) for BEHIND_HOLD_S
    it starts again too, dropping what the link held back."""

    def __init__(self, serial=None, size="1920x1440", fps=30, bit_rate=None, camera_id=None, port=PORT,
                 restarts=RESTARTS, log=None, record=None, codec_options=None, drop_late=True):
        self.fixed = serial             # given by hand (command line): no link to follow
        self.serial = pick_serial(serial)
        self.link = phone_link.transport_of(self.serial)
        self.record = record            # an open binary file that gets the raw H.264 (`record`)
        self.frame_times = []           # decoding time of every frame while recording
        self.frame_pts = []             # and its capture time (s, phone clock)
        bit_rate = bit_rate or BIT_RATES.get(self.link, BIT_RATE)
        if size == SIZES[phone_link.USB]:   # the default asked for: the link decides
            size = SIZES.get(self.link, size)
        self.size, self.fps, self.bit_rate, self.camera_id, self.port = size, fps, bit_rate, camera_id, port
        self.codec_options = CODEC_OPTIONS if codec_options is None else codec_options
        self.drop_late = drop_late      # False: every frame is passed on
        self.max_restarts, self.restarts, self.failed_restarts = restarts, 0, 0
        self.keep_awake = True          # the phone's screen kept on (see KEEP_AWAKE_S)
        self.waiting = False            # the link dropped: waiting for it to come back
        self.link_drops = 0             # times the stream waited for the link and went on
        self.cause = ""                 # the link cause when the stream gave up because of the link
        self.scid = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        self.log = log or (lambda msg: None)
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._item, self._n, self._session = None, 0, 0
        self._times = deque(maxlen=120)
        self._decode_ms = deque(maxlen=120)
        self._late_ms = deque(maxlen=120)
        self._offsets = deque()         # (arrival, arrival - capture), increasing offsets, over OFFSET_WINDOW_S
        self.dropped = 0                # frames decoded but not passed on (a newer one had come with them)
        self.sessions = []              # (w, h) of each capture session the phone announced
        self._stop = threading.Event()
        self.server, self.sock, self.thread, self.error = None, None, None, None
        self.server_lines = []

    # -- start and stop
    def start(self, timeout=15.0):
        """Starts the server and returns once the first frame is in (raises LiveError otherwise)."""
        self._launch()
        self.thread = threading.Thread(target=self._read, args=(timeout,), daemon=True, name="phone_stream")
        self.thread.start()
        if self.keep_awake:
            threading.Thread(target=self._awake, daemon=True, name="phone_awake").start()
        t0 = time.perf_counter()
        while self._item is None:
            if self.error and not self.thread.is_alive() or time.perf_counter() - t0 > timeout:
                msg = self.error or "no frame within the timeout"
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
        """The tunnel (again after a reconnection: adb drops it with the link), the server copied to the phone
        (cleanup=true deletes it when the previous server exits) and started; it lives as long as this adb shell."""
        d = find_scrcpy()
        adb("forward", f"tcp:{self.port}", f"localabstract:scrcpy_{self.scid:08x}", serial=self.serial)
        adb("push", str(d / "scrcpy-server"), SERVER_REMOTE, serial=self.serial)
        args = [f"scid={self.scid:08x}", "log_level=info", "video_source=camera", "audio=false", "control=false",
                "tunnel_forward=true", "send_device_meta=false", "send_dummy_byte=false", "send_stream_meta=true",
                "send_frame_meta=true", "video_codec=h264", f"video_bit_rate={self.bit_rate}",
                f"camera_size={self.size}", f"camera_fps={self.fps}", "cleanup=true"]
        if self.codec_options:
            args.append(f"video_codec_options={self.codec_options}")
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
            item = self._item
        if item is None:
            return None, 0.0, 0
        return item.bgr(), item.t, item.n

    def wait_frame(self, after=0, timeout=0.1):
        """The newest frame once its counter is past `after`: (BGR frame, t, n), or (None, 0.0, after) on timeout."""
        item = self.wait_item(after, timeout)
        if item is None:
            return None, 0.0, after
        return item.bgr(), item.t, item.n

    def latest_item(self):
        """The newest Frame (None before the first)."""
        with self._lock:
            return self._item

    def wait_item(self, after=0, timeout=0.1):
        """The newest Frame once its counter is past `after`, else None on timeout."""
        with self._new:
            self._new.wait_for(lambda: self._n > after or self._stop.is_set(), timeout=timeout)
            if self._n <= after or self._item is None:
                return None
            return self._item

    def stats(self):
        """Measured fps over the last frames, mean and 95th percentile of the decode interval (ms), mean time to
        decode a frame (ms), how late the frames came against the quickest of the last seconds (median and 95th
        percentile, ms), frames dropped in bursts, frame count, size, link, restarts, waits for the link."""
        with self._lock:
            t = np.array(self._times)
            n = self._n
            dec = float(np.mean(self._decode_ms)) if self._decode_ms else float("nan")
            late = np.array(self._late_ms)
            item = self._item
        more = {"link": self.link, "restarts": self.restarts, "link_drops": self.link_drops, "waiting": self.waiting,
                "bit_rate": self.bit_rate, "dropped": self.dropped, "sessions": len(self.sessions),
                "size": list(item.size) if item is not None else None,
                "late_ms": float(np.median(late)) if len(late) else float("nan"),
                "late_p95_ms": float(np.percentile(late, 95)) if len(late) else float("nan")}
        if len(t) < 2:
            return {"fps": 0.0, "interval_ms": float("nan"), "interval_p95_ms": float("nan"), "decode_ms": dec,
                    "frames": n, **more}
        dt = np.diff(t) * 1e3
        return {"fps": float((len(t) - 1) / (t[-1] - t[0])), "interval_ms": float(dt.mean()),
                "interval_p95_ms": float(np.percentile(dt, 95)), "decode_ms": dec, "frames": n, **more}

    # -- threads
    def _server_log(self, server):
        for line in server.stdout:
            line = line.strip()
            if line:
                self.server_lines.append(line)
                self.log(line)

    def _awake(self):
        """Keeps the phone's screen on while the stream runs (KEEP_AWAKE_S)."""
        while not self._stop.wait(KEEP_AWAKE_S if self._item is not None else 1.0):
            try:
                adb("shell", "input", "keyevent", "0", serial=self.serial, timeout=10, check=False)
            except (LiveError, subprocess.TimeoutExpired, OSError):
                pass

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
            if self.server is not None and self.server.poll() is not None:
                time.sleep(0.2)         # its last lines
                raise LiveError(f"server exited ({self.server.returncode})")
            time.sleep(0.1)
        raise LiveError("no stream from the server")

    def _link_up(self):
        """The phone answers on its link (adb lists it and a round trip comes back)."""
        if self.fixed:
            try:
                return any(d[0] == self.fixed and d[1] == "device" for d in devices())
            except LiveError:
                return False
        return phone_link.get().reachable()

    def _capture_time(self, arrival, pts_s):
        """The capture time on this computer's clock: the phone's capture time plus the smallest arrival offset of the
        last OFFSET_WINDOW_S (the frames that came quickest), and how late this frame came against those (ms)."""
        off = arrival - pts_s
        q = self._offsets
        while q and q[-1][1] >= off:
            q.pop()
        q.append((arrival, off))
        while q[0][0] < arrival - OFFSET_WINDOW_S:
            q.popleft()
        base = q[0][1]
        return pts_s + base, (off - base) * 1e3

    def _publish(self, frame, pts_us, t0, decode_ms):
        t = time.perf_counter()
        t_cap, late = self._capture_time(t, pts_us / 1e6) if pts_us else (t, 0.0)
        with self._new:
            self._n += 1
            self._item = Frame(self._n, t, t_cap, av_frame=frame, session=self._session)
            self._times.append(t)
            self._decode_ms.append(decode_ms)
            self._late_ms.append(late)
            self._new.notify_all()
        return late

    def _read(self, timeout):
        """Decode until stopped. When the stream breaks after it has run: if the link dropped (Wi-Fi out of reach),
        wait for it to come back (LINK_WAIT_S, not counted as a restart); otherwise start the server again; it gives
        up after max_restarts restarts in a row that brought no frames back. Then on the serial the phone has now
        (over Wi-Fi its port may have changed). .error holds the last failure, .cause the link's when it gave up."""
        import av
        relaunch = False
        while not self._stop.is_set():
            got = 0
            try:
                if relaunch:
                    relaunch = False
                    self.serial = pick_serial(self.fixed)
                    self._launch()
                self.sock, data = self._connect(timeout)
                self.sock.settimeout(STALL_S)
                self.waiting = False
                with self._lock:
                    self._session += 1
                    self._offsets.clear()
                framing = _Framing()
                codec = av.CodecContext.create("h264", "r")
                codec.flags |= av.codec.context.Flags.low_delay
                behind_since = None
                while not self._stop.is_set():
                    newest, held_since = None, None
                    while True:
                        for kind, unit in framing.feed(data):
                            if kind == "session":
                                if self.sessions and tuple(unit) != self.sessions[-1]:
                                    self.log(f"capture size now {unit[0]}x{unit[1]}")
                                    with self._lock:
                                        self._session += 1
                                self.sessions.append(tuple(unit))
                                continue
                            pts, key, payload = unit
                            if self.record is not None:
                                self.record.write(payload)
                            t0 = time.perf_counter()
                            packet = av.Packet(payload)
                            packet.pts = pts
                            for frame in codec.decode(packet):
                                if newest is not None:
                                    self.dropped += 1
                                newest = (frame, pts, t0, (time.perf_counter() - t0) * 1e3)
                                held_since = held_since or time.perf_counter()
                                if self.record is not None:
                                    self.frame_times.append(time.perf_counter())
                                    self.frame_pts.append(pts / 1e6)
                                if not self.drop_late:
                                    self._publish(*newest)
                                    got += 1
                                    newest = None
                        # more is already waiting (a burst): decode it before a frame is passed on, but never hold
                        # one longer than HOLD_MS
                        if newest is None or (time.perf_counter() - held_since) * 1e3 > HOLD_MS \
                                or not select.select([self.sock], [], [], 0)[0]:
                            break
                        data = self.sock.recv(1 << 18)
                        if not data:
                            raise LiveError("stream closed by the phone")
                    if newest is not None:
                        late = self._publish(*newest)
                        got += 1
                        behind_since = (behind_since or time.perf_counter()) if late > BEHIND_S * 1e3 else None
                        if behind_since is not None and time.perf_counter() - behind_since > BEHIND_HOLD_S:
                            raise LiveError(f"stream {late / 1e3:.1f} s behind")
                    data = self.sock.recv(1 << 18)
                    if not data:
                        raise LiveError("stream closed by the phone")
            except Exception as e:      # reported by start() or by the page through .error
                if self._stop.is_set():
                    return
                self.error = f"{type(e).__name__}: {e}; server: {' | '.join(self.server_lines[-3:])}"
                # the camera taken by another client (the camera app just closed, the face unlock): try again
                busy = any("open cameras" in line or "Camera disconnected" in line for line in self.server_lines[-30:])
                if self._item is None and not busy:
                    return
                self.server_lines.clear()
                try:
                    self.sock.close()
                except (OSError, AttributeError):
                    pass
                if self.server is not None and self.server.poll() is None:
                    self.server.terminate()
                self.failed_restarts = 0 if got >= RESTART_GOOD_FRAMES else self.failed_restarts + 1
                if self._link_up():
                    if self.failed_restarts > self.max_restarts:
                        return
                    self.restarts += 1
                    self.log(f"restart {self.restarts}: {self.error}")
                else:
                    self.waiting = True
                    self.log(f"link down, waiting up to {LINK_WAIT_S:.0f} s: {self.error}")
                    if self.fixed or phone_link.get().wait_ready(LINK_WAIT_S, stop=self._stop) is None:
                        if not self._stop.is_set():
                            self.cause = phone_link.get().status()["cause"] or "wifi_lost"
                        return
                    self.link_drops += 1
                    self.failed_restarts = 0
                time.sleep(min(3.0, 0.3 * 2 ** min(self.failed_restarts, 4)))
                relaunch = True


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
        s = PhoneStream(a.serial, a.size, a.fps, record=raw, restarts=0, drop_late=False)
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
    pts = np.array(s.frame_pts)
    np.save(str(out) + "_pts.npy", pts - pts[0] if len(pts) else pts)
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


# -- checks without the phone
def _test_clip(n=90, size=(640, 480), fps=30):
    """n frames of H.264 (a moving pattern) as (packets with key flags), encoded here with libx264."""
    from fractions import Fraction
    import av
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = size[0], size[1], "yuv420p"
    enc.time_base, enc.framerate = Fraction(1, fps), Fraction(fps, 1)
    enc.options = {"bf": "0", "g": "30", "preset": "ultrafast", "tune": "zerolatency"}
    out = []
    x = np.arange(size[0])[None, :]
    y = np.arange(size[1])[:, None]
    for k in range(n):
        img = ((np.sin((x + 7 * k) / 13.0) + np.cos((y - 3 * k) / 17.0)) * 60 + 128).astype(np.uint8)
        fr = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), format="bgr24").reformat(
            format="yuv420p")
        fr.pts = k
        out += [(bytes(p), bool(p.is_keyframe)) for p in enc.encode(fr)]
    out += [(bytes(p), bool(p.is_keyframe)) for p in enc.encode(None)]
    return out


def _framed(packets, fps=30, size=(640, 480), t0_us=5_000_000):
    """The packets as scrcpy 4.1 sends them with stream and frame meta: codec id, a session header, then for each
    frame its header (capture time, key flag, size) and the packet; the SPS and PPS of the first one as a config
    packet of their own."""
    out = [b"h264", (0x80000000).to_bytes(4, "big") + size[0].to_bytes(4, "big") + size[1].to_bytes(4, "big")]
    for k, (p, key) in enumerate(packets):
        pts = t0_us + int(round(k * 1e6 / fps))
        if k == 0:      # config apart: everything before the first slice
            cut = p.find(b"\x00\x00\x01\x65")
            cut = cut - 1 if cut > 0 and p[cut - 1] == 0 else cut
            if cut > 0:
                out.append(((1 << 62)).to_bytes(8, "big") + cut.to_bytes(4, "big") + p[:cut])
                p = p[cut:]
        head = pts | ((1 << 61) if key else 0)
        out.append(head.to_bytes(8, "big") + len(p).to_bytes(4, "big") + p)
    return out


class _FakePhone(PhoneStream):
    """PhoneStream reading a local socket that plays a framed clip: no phone, no adb."""

    def __init__(self, units, schedule, **kw):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        self.__dict__.update(_fake_state(srv.getsockname()[1], kw))
        self.srv, self._units, self._schedule = srv, units, schedule
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.srv.accept()
        t0 = time.perf_counter()
        for k, u in enumerate(self._units):
            time.sleep(max(0.0, t0 + self._schedule[k] - time.perf_counter()))
            try:
                conn.sendall(u)
            except OSError:
                return
        time.sleep(0.5)
        conn.close()

    def _launch(self):
        self.server = None

    def _link_up(self):
        return True

    def close(self):
        self._stop.set()
        with self._new:
            self._new.notify_all()
        for s in (self.sock, self.srv):
            try:
                s.close()
            except (OSError, AttributeError):
                pass


def _fake_state(port, kw):
    """The attributes PhoneStream.__init__ sets, without asking phone_link for a serial."""
    s = object.__new__(PhoneStream)
    s.__dict__.update(fixed="fake", serial="fake", link="usb", record=None, frame_times=[], frame_pts=[],
                      size="640x480", fps=30, bit_rate=BIT_RATE, camera_id=None, port=port, codec_options="",
                      drop_late=kw.get("drop_late", True), max_restarts=0, restarts=0, failed_restarts=0,
                      keep_awake=False, waiting=False, link_drops=0, cause="", scid=0, log=lambda m: None,
                      _item=None, _n=0, _session=0, dropped=0, sessions=[], server=None, sock=None, thread=None,
                      error=None, server_lines=[])
    s._lock = threading.Lock()
    s._new = threading.Condition(s._lock)
    s._times, s._decode_ms, s._late_ms = deque(maxlen=120), deque(maxlen=120), deque(maxlen=120)
    s._offsets = deque()
    s._stop = threading.Event()
    return s.__dict__


def run_test():
    """The reader without the phone: the scrcpy framing cut anywhere, a burst after a stall (only the newest frame
    passed on, capture times still evenly spaced, the delay recovered at once), a session of a new size."""
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    packets = _test_clip()
    units = _framed(packets)
    data = b"".join(units)
    rng = np.random.default_rng(0)
    cuts = np.sort(rng.integers(1, len(data), 200))
    fr = _Framing()
    got = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(data)]):
        got += fr.feed(data[a:b])
    frames = [u for k, u in got if k == "frame"]
    sessions = [u for k, u in got if k == "session"]
    expect(len(frames) == len(packets) and sessions == [(640, 480)] and frames[0][2].startswith(packets[0][0][:4])
           and all(f[2].endswith(p[0][-16:]) for f, p in zip(frames, packets)),
           f"framing cut in 200 places: {len(frames)} of {len(packets)} frames, session {sessions}, config in front "
           f"of the first frame")
    # a stall of 0.6 s at frame 30, then the held frames in a burst; the rest at 30 fps
    sched = [0.0, 0.0]
    for k in range(len(units) - 2):
        f = k - 1 if k > 0 else 0
        sched.append(f / 30 if f < 30 else max(f / 30, 30 / 30 + 0.6) if f < 48 else f / 30 + 0.0)
    s = _FakePhone(units, sched)
    s.start(timeout=5.0)
    seen = []
    t_end = time.perf_counter() + 4.5
    n = 0
    while time.perf_counter() < t_end:
        it = s.wait_item(n, 0.2)
        if it is not None:
            n = it.n
            seen.append((it.n, it.t, it.t_capture, it.gray().shape))
            time.sleep(0.005)
    st = s.stats()
    s.close()
    caps = np.array([x[2] for x in seen])
    arr = np.array([x[1] for x in seen])
    lag = arr - caps
    expect(st["dropped"] >= 5 and len(seen) >= len(packets) - st["dropped"] - 3,
           f"burst after a 0.6 s stall: {st['dropped']} frames decoded but not passed on, {len(seen)} passed on")
    steps = np.diff(caps)
    expect(np.all(steps > 0) and np.min(steps) > 0.03 and seen[-1][3] == (480, 640),
           f"capture times evenly spaced through the burst (steps {np.min(steps) * 1e3:.1f}-"
           f"{np.max(steps) * 1e3:.1f} ms), gray frames 640 x 480")
    expect(np.median(lag[-15:]) < 0.06 and lag.max() < 0.1,
           f"no frame passed on late: delay over the quickest at most {lag.max() * 1e3:.0f} ms (the burst's older frames "
           f"dropped), after the burst {np.median(lag[-15:]) * 1e3:.0f} ms")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


def saved_wifi():
    """The Wi-Fi address and phone id ImpactGuide keeps (preferences.json in LOCALAPPDATA/ImpactGuide), if any."""
    import json
    try:
        prefs = json.loads((Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ImpactGuide" / "preferences.json")
                           .read_text(encoding="utf-8"))
        return {"address": prefs["phone"].get("wifi_address", ""), "phone_id": prefs["phone"].get("phone_id", "")}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["devices", "cameras", "view", "bench", "latency", "record", "test"])
    ap.add_argument("out", nargs="?", help="record: the file to write (OUT.h264 and OUT_times.npy)")
    ap.add_argument("--serial")
    ap.add_argument("--link", choices=phone_link.MODES, default=phone_link.USB,
                    help="cable or Wi-Fi (the last Wi-Fi address is read from ImpactGuide's preferences)")
    ap.add_argument("--size", default="1920x1440")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    if a.command == "test":
        sys.exit(run_test())
    if a.link == phone_link.WIFI:
        phone_link.get().configure(mode=phone_link.WIFI, **saved_wifi())
        if not a.serial and phone_link.get().wait_ready(20) is None:
            sys.exit(f"live: no phone on Wi-Fi ({phone_link.get().status()['cause']})")
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
