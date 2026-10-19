"""The link to the phone, by cable (USB) or by Wi-Fi (Android 11+ wireless debugging), found by itself.

The one place that knows which device serial to use. Everything that talks to the phone asks here: live.py (the
camera stream), acquisition.py (photos and videos), live_ar.py (closing the camera app), the app (the choice, the
pairing, the state of the link). One adb for all of them: the one shipped with scrcpy.

    link = phone_link.get()                          # the link of this process: "auto" unless told otherwise
    link.configure(mode="wifi")                      # or "usb", or "auto": the cable when the phone is on it, else
                                                     # the Wi-Fi it was paired on; link.via says which is in use
    link.serial()                                    # the phone on the chosen link, or LinkError(cause)
    link.pair_code("192.168.1.23:37011", "123456")   # the code and the address the phone shows
    payload = link.qr_start(); link.qr_poll()        # a QR code the phone scans: "waiting" until it is paired
    link.connect("192.168.1.23:41235")               # or found by itself over mDNS
    link.watch(listener)                             # checks the link every WATCH_S; brings Wi-Fi back when it drops
    link.unwatch(listener)                           # that listener hears no more
    link.status()                                    # mode, state, cause, serial, model, address, round trip, drops
    link.open_url(url, port)                         # the phone's browser on a page of this PC, through adb
    link.close()                                     # the watch stopped and joined, the tunnels of adb reverse removed

Threads. One check of the link at a time: a caller that comes while another check runs gets its result. Listeners
hear every change in the order it happened, from one notifier thread, each with the status as it was at that change.
Every adb call has a timeout and an argument list; what goes to the phone's own shell is quoted; pairing codes and
passwords never reach the log.

Wi-Fi, once per PC: on the phone Developer options > Wireless debugging, then "Pair device with pairing code" (a
6-digit code and an address: adb pair ADDRESS CODE) or "Pair device with QR code" (the app shows
WIFI:T:ADB;S:<name>;P:<password>;; the phone announces _adb-tls-pairing._tcp under that name over mDNS and the app
pairs with the password). Then the phone announces _adb-tls-connect._tcp: adb connects by itself
(ADB_MDNS_AUTO_CONNECT) or the app does (adb connect IP:PORT). The port changes each time Wireless debugging is turned
on, so after a drop the app tries the last address, then the one the phone announces now.

Networks that keep devices apart (eduroam and most institutional Wi-Fi) let neither mDNS nor the connection through:
the phone's own hotspot or a dedicated router.

A failure carries a cause that the interface says in plain words (LinkError.cause, one of CAUSES), never adb's own
text, which only goes to the log.

    python phone_link.py devices                what adb sees, by link
    python phone_link.py pair ADDRESS CODE      pairing with the code the phone shows, then the connection
    python phone_link.py connect [ADDRESS]      the Wi-Fi connection (default: the address the phone announces)
    python phone_link.py watch [--seconds S]    the link checked every 2 s, with its round trip and its drops
    python phone_link.py test                   selection, pairing and reconnection on a simulated phone; exit 1 on
                                                failure
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import queue
import re
import secrets
import shlex
import shutil
import socket
import string
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

_LOGGER = logging.getLogger("phone_link")

USB, WIFI, AUTO = "usb", "wifi", "auto"
MODES = (AUTO, USB, WIFI)           # AUTO: the cable when the phone is on it, else Wi-Fi if it was ever paired
MDNS_CONNECT = "_adb-tls-connect._tcp"
MDNS_PAIRING = "_adb-tls-pairing._tcp"
WATCH_S = 2.0                   # the watch looks at the link this often
BACKOFF_S = (1.0, 2.0, 4.0, 8.0)  # between reconnection attempts; then every 8 s, for as long as Wi-Fi is chosen
LOST_AFTER_S = 20.0             # without the phone this long the link is "lost" (the attempts go on)
QR_TIMEOUT_S = 180.0            # a QR code waits this long for the phone to scan it
PAIR_TIMEOUT_S = 30.0
CONNECT_AFTER_PAIR_S = 12.0     # after pairing, the phone's connection service is waited for this long
ADB_TIMEOUT_S = 20.0            # any adb call without a timeout of its own
DEVICES_TIMEOUT_S = 15.0        # adb devices (it may start the adb server)
MDNS_TIMEOUT_S = 10.0
SHELL_TIMEOUT_S = 8.0           # a short command in the phone's shell
CONNECT_TIMEOUT_S = 10.0        # adb connect, adb disconnect
REVERSE_TIMEOUT_S = 10.0        # adb reverse
UNREVERSE_TIMEOUT_S = 3.0       # adb reverse --remove, at close: never holds the app's exit for long
OPEN_URL_TIMEOUT_S = 15.0
PING_TIMEOUT_S = 3.0
CONNECT_WAIT_S = 5.0            # after adb connect, the phone is waited for this long
POLL_S = 0.5                    # between two looks while waiting for the phone
CHECK_WAIT_S = 30.0             # a check that comes while another one runs waits this long for its result
WAIT_SLICE_S = 1.0              # wait_ready waits at most this long on another check before it looks at its stop
JOIN_S = 2.0                    # the watch and the notifier are waited for this long at close
NOTIFY_IDLE_S = 5.0             # the notifier thread ends after this long without news (it starts again on the next)
ADB_PORT = 5037                 # the adb server's port unless ANDROID_ADB_SERVER_PORT says otherwise
NO_WINDOW = 0x08000000 if os.name == "nt" else 0    # CREATE_NO_WINDOW: no console flash under pythonw
ADDRESS_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3}|\[[0-9A-Fa-f:]+\]):(\d{1,5})")
HOST_PORT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*:\d{1,5}")    # a TCP transport by name: localhost:5555
VERSION_RE = re.compile(r"(\d+(?:\.\d+)*)")
STATES = ("device", "offline", "unauthorized", "authorizing", "connecting", "recovery", "sideload", "bootloader",
          "host", "rescue", "unknown")
# The causes the interface names (strings link.cause.<cause> in ui/strings).
CAUSES = ("no_adb", "no_device", "unauthorized", "many_devices", "offline", "timeout", "wifi_none", "wifi_connecting",
          "wifi_reconnecting", "wifi_lost", "pair_code", "pair_failed", "unreachable", "not_paired",
          "paired_no_connect", "qr_timeout", "bad_address", "mdns_off")


class LinkError(RuntimeError):
    """A failure of the link with a cause the interface can name (CAUSES); detail is a short plain detail (an
    address, a command name) for the log, never a secret.

    Attributes:
        cause: one of CAUSES.
        code: the same as cause.
        detail: for the log.
    """

    def __init__(self, cause: str, detail: str = "") -> None:
        super().__init__(f"{cause}: {detail}" if detail else cause)
        self.cause = self.code = cause
        self.detail = detail


def shell_line(*words: object) -> str:
    """A command for the phone's own shell, every word quoted (adb passes the line to the phone's sh as it is)."""
    return " ".join(shlex.quote(str(w)) for w in words)


def _version(path: str) -> tuple[int, ...]:
    """The version in a scrcpy folder's name as numbers (scrcpy-win64-v3.10 -> (3, 10)), () when none."""
    m = VERSION_RE.search(Path(path).name.rsplit("-", 1)[-1])
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


# ── adb ──────────────────────────────────────────────────────────────────────
def scrcpy_dir() -> Path | None:
    """Folder holding scrcpy-server and adb.exe (IMPACTGUIDE_SCRCPY, scrcpy on the PATH, the WinGet packages, the
    newest version first), or None."""
    cands = [os.environ.get("IMPACTGUIDE_SCRCPY")]
    exe = shutil.which("scrcpy")
    if exe:
        cands.append(str(Path(exe).resolve().parent))
    local = os.environ.get("LOCALAPPDATA", "")
    cands += sorted(glob.glob(os.path.join(local, "Microsoft", "WinGet", "Packages", "Genymobile.scrcpy*",
                                           "scrcpy-win64-*")), key=_version, reverse=True)
    adb = "adb.exe" if os.name == "nt" else "adb"
    for c in cands:
        if c and (Path(c) / "scrcpy-server").is_file() and (Path(c) / adb).is_file():
            return Path(c)
    return None


_ADB_FOUND: dict[tuple, str] = {}
_ADB_ENV = ("IMPACTGUIDE_ADB", "IMPACTGUIDE_SCRCPY", "PATH", "LOCALAPPDATA", "ANDROID_HOME")


def find_adb() -> str | None:
    """The adb of the whole app: IMPACTGUIDE_ADB, the one shipped with scrcpy (the stream needs its server, and two
    adb versions stop each other's server), the PATH, the one bundled with adbutils, the Android SDK. None if none.
    The answer is kept while the environment is the same and the file is there."""
    key = tuple(os.environ.get(k, "") for k in _ADB_ENV)
    hit = _ADB_FOUND.get(key)
    if hit and Path(hit).is_file():
        return hit
    found = _find_adb()
    if found:
        _ADB_FOUND[key] = found
    return found


def _find_adb() -> str | None:
    name = "adb.exe" if os.name == "nt" else "adb"
    cands = [os.environ.get("IMPACTGUIDE_ADB")]
    d = scrcpy_dir()
    if d is not None:
        cands.append(str(d / name))
    cands.append(shutil.which("adb"))
    try:
        import adbutils
        cands.append(str(Path(adbutils.__file__).parent / "binaries" / name))
    except ImportError:
        pass
    sdk = os.environ.get("ANDROID_HOME") or os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk")
    cands.append(os.path.join(sdk, "platform-tools", name))
    return next((c for c in cands if c and Path(c).is_file()), None)


def adb_port() -> int:
    """The adb server's port: ANDROID_ADB_SERVER_PORT, else ADB_PORT."""
    try:
        return int(os.environ.get("ANDROID_ADB_SERVER_PORT", ADB_PORT))
    except ValueError:
        return ADB_PORT


class AdbRunner:
    """Runs adb: (exit code, stdout and stderr together). LinkError("no_adb") without adb, ("timeout") on timeout
    (its detail the command's name only: the arguments may hold a pairing code)."""

    def __init__(self, exe: str | None = None) -> None:
        self.exe = exe or find_adb()

    def __call__(self, args: Sequence[object], serial: str | None = None,
                 timeout: float = ADB_TIMEOUT_S) -> tuple[int, str]:
        if not self.exe:
            raise LinkError("no_adb")
        cmd = [self.exe] + (["-s", serial] if serial else []) + [str(a) for a in args]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, creationflags=NO_WINDOW, check=False)
        except subprocess.TimeoutExpired:
            raise LinkError("timeout", str(args[0]) if args else "") from None
        except OSError as e:
            raise LinkError("no_adb", str(e)) from e
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    def ping(self, serial: str, port: int | None = None, timeout: float = PING_TIMEOUT_S) -> float:
        """Round trip to the phone in ms: a shell `echo` through the adb server's socket (no process started here,
        so the time is the link's and adbd's)."""
        t0 = time.perf_counter()
        with socket.create_connection(("127.0.0.1", port or adb_port()), timeout=timeout) as s:
            s.settimeout(timeout)
            for req in (f"host:transport:{serial}", "shell:echo 1"):
                data = req.encode()
                s.sendall(b"%04x" % len(data) + data)
                status = b""
                while len(status) < 4:
                    chunk = s.recv(4 - len(status))
                    if not chunk:
                        raise OSError("closed")
                    status += chunk
                if status != b"OKAY":
                    raise OSError(status.decode(errors="replace"))
            while s.recv(4096):
                pass
        return (time.perf_counter() - t0) * 1e3


def transport_of(serial: str) -> str:
    """wifi (IP:PORT, HOST:PORT, or the name adb gives a phone it found over mDNS), usb, or other (an emulator)."""
    if serial.startswith("emulator-"):
        return "other"
    if MDNS_CONNECT in serial or ADDRESS_RE.fullmatch(serial) or HOST_PORT_RE.fullmatch(serial):
        return WIFI
    return USB


def parse_devices(text):
    """[{serial, state, model, transport}] from `adb devices -l` (lines of the adb server starting are skipped)."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[1] not in STATES or line.startswith(("*", "List of")):
            continue
        model = next((p.split(":", 1)[1] for p in parts[2:] if p.startswith("model:")), "")
        out.append({"serial": parts[0], "state": parts[1], "model": model.replace("_", " "),
                    "transport": transport_of(parts[0])})
    return out


def parse_mdns(text):
    """[{name, service, address}] from `adb mdns services`."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        svc = next((p.rstrip(".") for p in parts[1:] if p.rstrip(".") in (MDNS_CONNECT, MDNS_PAIRING)), None)
        addr = next((p for p in parts if ADDRESS_RE.fullmatch(p)), None)
        if parts and svc and addr:
            name = parts[0]
            name = name.split("." + svc)[0] if ("." + svc) in name else name
            out.append({"name": name, "service": svc, "address": addr})
    return out


def parse_pair(text):
    """(ok, cause, phone id from the guid adb prints) of `adb pair`."""
    m = re.search(r"Successfully paired to \S+(?: \[guid=([^\]]+)\])?", text)
    if m:
        guid = m.group(1) or ""
        g = re.match(r"adb-(.+)-[0-9A-Za-z]+$", guid)
        return True, "", g.group(1) if g else ""
    low = text.lower()
    if "wrong password" in low or "password" in low:
        return False, "pair_code", ""
    if "unable to start pairing" in low or "failed to connect" in low or "connection refused" in low \
            or "timed out" in low or "unreachable" in low or "no route" in low:
        return False, "unreachable", ""
    return False, "pair_failed", ""


def parse_connect(text):
    """(ok, cause) of `adb connect`."""
    low = text.lower()
    if re.search(r"(already )?connected to ", low) and "failed" not in low and "cannot" not in low:
        return True, ""
    if "authenticate" in low:
        return False, "not_paired"
    if "resolve host" in low or "bad port" in low:
        return False, "bad_address"
    return False, "unreachable"


def host_of(address):
    m = ADDRESS_RE.fullmatch(address or "")
    return m.group(1) if m else ""


def phone_id_of(serial):
    """The phone's own serial (ro.serialno) inside the name adb gives it over mDNS (adb-<id>-<random>), or ''."""
    m = re.match(r"adb-(.+)-[0-9A-Za-z]+\." + re.escape(MDNS_CONNECT), serial)
    return m.group(1) if m else ""


# ── the link ─────────────────────────────────────────────────────────────────
class Link:
    """The chosen link and what is known of it. Thread-safe; the watch runs on its own thread, the listeners hear
    from one notifier thread, in order."""

    def __init__(self, mode: str = AUTO, address: str = "", phone_id: str = "", runner: Callable | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] | None = None) -> None:
        self.run = runner or AdbRunner()
        self.clock, self.sleep = clock, sleep
        self.log = log or _LOGGER.debug
        self._lock = threading.RLock()          # the state below
        self._attempting = threading.Lock()     # one reconnection attempt at a time
        self._check_cond = threading.Condition()    # one check at a time; the others wait for its result
        self._checking, self._check_gen = False, 0
        self._watch_lock = threading.Lock()     # the watch thread and the listeners
        self._listeners = []
        self._watch, self._stop = None, threading.Event()
        self._news = queue.Queue()              # status snapshots for the listeners, in the order of the changes
        self._notifier = None
        self._reversed = {}                     # port -> serial: tunnels of adb reverse laid by this link
        self.mode = mode if mode in MODES else AUTO
        self.via = WIFI if self.mode == WIFI else USB   # the link in use now: the mode, or what AUTO found
        self._told = None                   # (state, cause, via) the listeners last heard
        self.address = address or ""        # the last Wi-Fi address that connected (IP:PORT)
        self.phone_id = phone_id or ""      # the phone's ro.serialno, to know it again under another name
        self.wifi_paired = False            # paired for Wi-Fi with this PC (kept by the app with the address)
        self._ids = {}                      # adb serial -> ro.serialno
        self._info = {}                     # adb serial -> {model, ip}
        self.rtt = deque(maxlen=5)
        self.events = deque(maxlen=40)      # (wall time, what): drops, reconnections, pairings
        self._qr = None
        self._reset_state()

    def _reset_state(self):
        self.state, self.cause, self.serial_now = "unknown", "", ""
        self.since = self.clock()
        self.down_since, self.tries, self.next_try = None, 0, 0.0
        self.drops, self.reconnects, self.silent = 0, 0, 0
        self.rtt.clear()

    # -- the choice
    def configure(self, mode=None, address=None, phone_id=None, paired=None):
        with self._lock:
            changed = mode in MODES and mode != self.mode
            if changed:
                self.mode = mode
                if mode != AUTO:
                    self.via = mode
                self._reset_state()
            if address is not None:
                self.address = address
            if phone_id is not None:
                self.phone_id = phone_id
            if paired is not None:
                self.wifi_paired = bool(paired)
        if changed:
            self._notify()
        return self

    def remembered(self):
        """What the app keeps between sessions."""
        return {"link": self.mode, "wifi_address": self.address, "phone_id": self.phone_id,
                "wifi_paired": self._paired()}

    # -- what adb sees
    def _adb(self, args: Sequence[object], serial: str | None = None, timeout: float = ADB_TIMEOUT_S,
             cause: str | None = None, secret: Sequence[str] = ()) -> tuple[int, str]:
        """adb with these arguments (a list, never a shell string); its answer goes to the log with the secrets
        (pairing codes, passwords) left out. A timeout is said as `cause` when given."""
        try:
            code, out = self.run(args, serial=serial, timeout=timeout)
        except LinkError as e:
            if cause and e.cause == "timeout":
                raise LinkError(cause, e.detail) from e
            raise
        shown = " ".join("***" if str(a) in secret else str(a) for a in args)
        text = out.strip()[-200:]
        for s in secret:
            text = text.replace(str(s), "***")
        self.log(f"adb {shown}: {text}")
        return code, out

    def _shell(self, *words: object, serial: str, timeout: float = SHELL_TIMEOUT_S) -> tuple[int, str]:
        """A command in the phone's shell, every word quoted."""
        return self._adb(["shell", shell_line(*words)], serial=serial, timeout=timeout)

    def devices(self) -> list[dict]:
        """[{serial, state, model, transport}] (LinkError no_adb or timeout)."""
        return parse_devices(self._adb(["devices", "-l"], timeout=DEVICES_TIMEOUT_S)[1])

    def mdns(self) -> list[dict]:
        """The adb services announced on the network: [{name, service, address}] ([] when mDNS is off)."""
        try:
            return parse_mdns(self._adb(["mdns", "services"], timeout=MDNS_TIMEOUT_S)[1])
        except LinkError:
            return []

    def _id_of(self, serial: str) -> str:
        with self._lock:
            if serial in self._ids:
                return self._ids[serial]
        pid = phone_id_of(serial)
        if not pid:
            try:
                code, out = self._shell("getprop", "ro.serialno", serial=serial)
                pid = out.strip() if code == 0 else ""
            except LinkError:
                pid = ""
        if pid:
            with self._lock:
                self._ids[serial] = pid
        return pid

    def _one_phone(self, ready):
        """The same phone seen twice on Wi-Fi (its address and its mDNS name) is one phone: the name adb keeps up by
        itself, or the one used last."""
        if len(ready) < 2:
            return ready
        ids = {d["serial"]: self._id_of(d["serial"]) for d in ready}
        if len(set(ids.values())) == 1 and "" not in ids.values():
            last = next((d for d in ready if d["serial"] == self.serial_now), None)
            named = next((d for d in ready if MDNS_CONNECT in d["serial"]), None)
            return [last or named or ready[0]]
        return ready

    def _paired(self):
        """Wi-Fi was paired with this PC: a pairing here, an address that connected, or the app remembers one."""
        return bool(self.address or self.wifi_paired)

    def _use(self, via):
        """AUTO found the phone on this link (or looks for it there)."""
        with self._lock:
            if via != self.via:
                self.via = via
                self.down_since, self.tries, self.next_try, self.silent = None, 0, 0.0, 0
                self.rtt.clear()
                self._event("via " + via)

    def _auto(self, found):
        """AUTO: the phone on the cable if it is there, else the one on Wi-Fi; nothing there, the Wi-Fi it was paired on
        (to bring back) or no phone at all."""
        usb = [d for d in found if d["transport"] == USB]
        ready = [d for d in usb if d["state"] == "device"]
        if len(ready) > 1 and self.phone_id:
            ready = [d for d in ready if d["serial"] == self.phone_id] or ready
        if len(ready) == 1:
            self._use(USB)
            return ready[0]["serial"]
        if len(ready) > 1:
            raise LinkError("many_devices", ", ".join(d["serial"] for d in ready))
        wifi = [d for d in found if d["transport"] == WIFI]
        on_air = self._one_phone([d for d in wifi if d["state"] == "device"])
        if len(on_air) == 1:
            self._use(WIFI)
            return on_air[0]["serial"]
        if len(on_air) > 1:
            raise LinkError("many_devices", ", ".join(d["serial"] for d in on_air))
        waiting = [d for d in usb + wifi if d["state"] in ("unauthorized", "authorizing")]
        if waiting:
            self._use(waiting[0]["transport"] if waiting[0]["transport"] in (USB, WIFI) else USB)
            raise LinkError("unauthorized")
        if usb:
            self._use(USB)
            raise LinkError("offline")
        if self._paired():
            self._use(WIFI)
            return self._wifi_cause()
        self._use(USB)
        raise LinkError("no_device")

    def _wifi_cause(self):
        with self._lock:
            if self.state in ("ready", "reconnecting"):
                raise LinkError("wifi_reconnecting")
            if self.state == "lost":
                raise LinkError("wifi_lost")
            raise LinkError("wifi_connecting" if self.address or self.serial_now else "wifi_none")

    def serial(self):
        """The adb serial of the phone on the chosen link (AUTO: on the cable, else on Wi-Fi; `via` says which).
        LinkError: no_adb, no_device, unauthorized, many_devices, offline (USB); wifi_none, wifi_connecting,
        wifi_reconnecting, wifi_lost, unauthorized, many_devices (Wi-Fi)."""
        found = self.devices()
        if self.mode == AUTO:
            return self._auto(found)
        mine = [d for d in found if d["transport"] == self.mode]
        ready = [d for d in mine if d["state"] == "device"]
        if self.mode == WIFI:
            ready = self._one_phone(ready)
        if len(ready) == 1:
            return ready[0]["serial"]
        if len(ready) > 1:
            raise LinkError("many_devices", ", ".join(d["serial"] for d in ready))
        if any(d["state"] in ("unauthorized", "authorizing") for d in mine):
            raise LinkError("unauthorized")
        if self.mode == USB:
            raise LinkError("offline" if mine else "no_device")
        return self._wifi_cause()

    def ready(self):
        """True when the phone is there on the chosen link (never raises)."""
        try:
            self.serial()
            return True
        except LinkError:
            return False

    def reachable(self):
        """ready() and a round trip that comes back: over Wi-Fi adb may still list a phone that has gone."""
        try:
            serial = self.serial()
        except LinkError:
            return False
        ping = getattr(self.run, "ping", None)
        if ping is None:
            return True
        try:
            ping(serial)
            return True
        except (OSError, LinkError):
            return False

    def wait_ready(self, timeout: float, stop: threading.Event | None = None, step: float = POLL_S) -> str | None:
        """Waits for the phone on the chosen link, helping the reconnection along (check). The serial, or None after
        the timeout or when `stop` (an Event) is set."""
        t_end = self.clock() + timeout
        while self.clock() < t_end and not (stop is not None and stop.is_set()):
            # another check under way (a reconnection can take 10 s): waited for in slices, so stop is seen
            st = self.check(wait=max(0.0, min(t_end - self.clock(), WAIT_SLICE_S)))
            if st["state"] == "ready":
                return st["serial"]
            self.sleep(step)
        return None

    # -- Wi-Fi: pairing and connection
    def discover(self):
        """Addresses of the phone's connection service announced over mDNS, this phone's first."""
        svcs = [s for s in self.mdns() if s["service"] == MDNS_CONNECT]
        mine = [s for s in svcs if self.phone_id and self.phone_id in s["name"]]
        return [s["address"] for s in mine] + [s["address"] for s in svcs if s not in mine and (len(svcs) == 1 or
                                                                                               not self.phone_id)]

    def _try_connect(self, address: str) -> tuple[bool, str]:
        code, out = self._adb(["connect", address], timeout=CONNECT_TIMEOUT_S, cause="unreachable")
        return parse_connect(out)

    def connect(self, address=None):
        """adb connect to `address` (IP:PORT), else to the last one, else to the one the phone announces; waits for
        the phone to be ready and returns its serial."""
        address = (address or "").strip()
        if address and not ADDRESS_RE.fullmatch(address):
            raise LinkError("bad_address", address)
        cands = [address] if address else ([self.address] if self.address else []) + self.discover()
        if not cands:
            raise LinkError("wifi_none")
        cause = "unreachable"
        for a in dict.fromkeys(cands):
            ok, cause = self._try_connect(a)
            if ok:
                with self._lock:
                    self.address = a
                serial = self._wait_wifi(CONNECT_WAIT_S, host_of(a))
                if serial:
                    self._up(serial, "connected")
                    return serial
                cause = "unauthorized"
        raise LinkError(cause)

    def _wait_wifi(self, seconds, host=""):
        t_end = self.clock() + seconds
        while True:
            try:
                ready = [d for d in self.devices() if d["transport"] == WIFI and d["state"] == "device"]
            except LinkError:
                ready = []
            ready = self._one_phone(ready)
            pick = next((d for d in ready if host and host_of(d["serial"]) == host), None) or (
                ready[0] if len(ready) == 1 else None)
            if pick:
                return pick["serial"]
            if self.clock() >= t_end:
                return ""
            self.sleep(POLL_S)

    def _pair_result(self, out: str) -> None:
        ok, cause, pid = parse_pair(out)
        if not ok:
            raise LinkError(cause, out.strip()[-200:])
        with self._lock:
            self.wifi_paired = True
            if pid:
                self.phone_id = pid
        self._event("paired")

    def _after_pairing(self, host):
        """The phone connects by itself after pairing (adb's mDNS auto-connect), or the app connects to the service
        it announces on the same host. Its serial, or LinkError("paired_no_connect")."""
        t_end = self.clock() + CONNECT_AFTER_PAIR_S
        while True:
            serial = self._wait_wifi(0, host)
            if serial:
                svc = next((s for s in self.mdns() if s["service"] == MDNS_CONNECT and host_of(s["address"]) == host),
                           None)
                if svc and not ADDRESS_RE.fullmatch(serial):
                    with self._lock:
                        self.address = svc["address"]
                elif ADDRESS_RE.fullmatch(serial):
                    with self._lock:
                        self.address = serial
                self._up(serial, "connected")
                return serial
            for s in self.mdns():
                if s["service"] == MDNS_CONNECT and host_of(s["address"]) == host:
                    ok, _ = self._try_connect(s["address"])
                    if ok:
                        with self._lock:
                            self.address = s["address"]
            if self.clock() >= t_end:
                raise LinkError("paired_no_connect")
            self.sleep(POLL_S)

    def pair_code(self, address, code):
        """Pairing with the 6-digit code the phone shows under "Pair device with pairing code", at the address it
        shows there (when empty: the one phone announcing a pairing service). Then the connection; its serial."""
        code = re.sub(r"\s", "", str(code or ""))
        if not re.fullmatch(r"\d{6}", code):
            raise LinkError("pair_code")
        address = (address or "").strip()
        if not address:
            svcs = [s for s in self.mdns() if s["service"] == MDNS_PAIRING]
            if len(svcs) != 1:
                raise LinkError("bad_address")
            address = svcs[0]["address"]
        if not ADDRESS_RE.fullmatch(address):
            raise LinkError("bad_address", address)
        _, out = self._adb(["pair", address, code], timeout=PAIR_TIMEOUT_S, cause="unreachable", secret=(code,))
        self._pair_result(out)
        return self._after_pairing(host_of(address))

    def qr_start(self):
        """A new QR code for "Pair device with QR code": the text to encode."""
        name = "impactguide-" + secrets.token_hex(3)
        password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
        with self._lock:
            self._qr = {"name": name, "password": password, "t0": self.clock()}
        payload = f"WIFI:T:ADB;S:{name};P:{password};;"
        shown = getattr(self.run, "show_qr", None)      # the simulated phone scans it
        if shown:
            shown(payload)
        return payload

    def qr_poll(self):
        """One look for the phone that scanned the QR code: "waiting", or "ready" once it is paired and connected
        (LinkError on failure, qr_timeout after QR_TIMEOUT_S)."""
        with self._lock:
            qr = self._qr
        if qr is None:
            raise LinkError("qr_timeout")
        if self.clock() - qr["t0"] > QR_TIMEOUT_S:
            self._qr_cancel_if(qr)
            raise LinkError("qr_timeout")
        hit = next((s for s in self.mdns() if s["service"] == MDNS_PAIRING and s["name"] == qr["name"]), None)
        if hit is None:
            return "waiting"
        self._qr_cancel_if(qr)              # this code is used up; a newer one shown meanwhile stays
        _, out = self._adb(["pair", hit["address"], qr["password"]], timeout=PAIR_TIMEOUT_S, cause="unreachable",
                           secret=(qr["password"],))
        self._pair_result(out)
        self._after_pairing(host_of(hit["address"]))
        return "ready"

    def _qr_cancel_if(self, qr: dict) -> None:
        """The QR code cancelled only if it is still this one (a new code shown in between is left alone)."""
        with self._lock:
            if self._qr is qr:
                self._qr = None

    def qr_wait(self, stop=None, step=1.0):
        """qr_poll until the phone is paired ("ready"), the code is cancelled ("cancelled") or LinkError."""
        while True:
            with self._lock:
                if self._qr is None:
                    return "cancelled"
            if stop is not None and stop.is_set():
                return "cancelled"
            if self.qr_poll() == "ready":
                return "ready"
            self.sleep(step)

    def qr_cancel(self):
        with self._lock:
            self._qr = None

    @property
    def qr_waiting(self):
        return self._qr is not None

    # -- the state, and the watch
    def check(self, wait: float = CHECK_WAIT_S) -> dict:
        """One look at the link: its state, the round trip, and on Wi-Fi an attempt to bring it back when the phone
        is gone (the last address, then the one the phone announces), spaced by BACKOFF_S. Returns status(). One
        check at a time: a caller that comes while another check runs waits for it (at most `wait` s) and gets its
        result, or the state as it is then."""
        with self._check_cond:
            if self._checking:
                gen = self._check_gen
                self._check_cond.wait_for(lambda: self._check_gen != gen, timeout=wait)
                return self.status()
            self._checking = True
        try:
            return self._check()
        finally:
            with self._check_cond:
                self._checking = False
                self._check_gen += 1
                self._check_cond.notify_all()

    def _check(self) -> dict:
        try:
            serial = self.serial()
        except LinkError as e:
            self._down(e)
            return self.status()
        if not self._up(serial) and self.via == WIFI:
            # adb still lists the phone but no round trip comes back twice in a row: the Wi-Fi link died without
            # adb noticing. Dropping the entry lets the reconnection start.
            with self._lock:
                self.silent += 1
                stale = self.silent >= 2
            if stale:
                try:
                    self._adb(["disconnect", serial], timeout=CONNECT_TIMEOUT_S)
                except LinkError:
                    pass
                with self._lock:
                    self.silent = 0
                self._down(LinkError("wifi_reconnecting"))
        return self.status()

    def _down(self, e):
        now = self.clock()
        with self._lock:
            wifi = self.via == WIFI
            if not wifi or e.cause in ("no_adb", "many_devices", "unauthorized", "timeout"):
                self._set("none", e.cause)
                return
            if not self._paired() and not self.serial_now:
                self._set("unpaired", "wifi_none")
                return
            if self.state == "ready":
                self.drops += 1
                self._event("dropped")
            first = self.down_since is None
            if first:
                self.down_since, self.tries, self.next_try = now, 0, now
            gone = self.state in ("ready", "reconnecting") or (first and self.drops)
            if now - self.down_since > LOST_AFTER_S:
                self._set("lost", "wifi_lost")
            else:
                self._set("reconnecting" if gone else "connecting",
                          "wifi_reconnecting" if gone else "wifi_connecting")
            due = now >= self.next_try
        if due and self._attempting.acquire(blocking=False):
            try:
                ok = self._attempt()
            finally:
                self._attempting.release()
            with self._lock:
                self.tries += 1
                self.next_try = self.clock() + BACKOFF_S[min(self.tries - 1, len(BACKOFF_S) - 1)]
            if ok:
                try:
                    self._up(self.serial())
                except LinkError:
                    pass

    def _attempt(self):
        """One try to bring Wi-Fi back: a stale entry dropped, the last address, then the ones the phone announces.
        True if one connected."""
        try:
            for d in self.devices():
                if d["transport"] == WIFI and d["state"] != "device" and ADDRESS_RE.fullmatch(d["serial"]):
                    self._adb(["disconnect", d["serial"]], timeout=CONNECT_TIMEOUT_S)
        except LinkError:
            pass
        cands = ([self.address] if self.address else []) + self.discover()
        for a in dict.fromkeys(cands):
            try:
                ok, cause = self._try_connect(a)
            except LinkError:
                continue
            if ok:
                with self._lock:
                    self.address = a
                return True
        return False

    def _up(self, serial, why=None):
        with self._lock:
            back = self.via == WIFI and self.state in ("reconnecting", "lost")
            if back:
                self.reconnects += 1
                self._event("reconnected")
            elif why:
                self._event(why)
            self.down_since, self.tries = None, 0
            new = serial != self.serial_now
            self.serial_now = serial
            if self.via == WIFI and ADDRESS_RE.fullmatch(serial):
                self.address = serial
        if new:
            self._identify(serial)
        answered = self._ping(serial)
        with self._lock:
            if answered:
                self.silent = 0
            self._set("ready", "")
        return answered

    def _identify(self, serial: str) -> None:
        info = {"model": "", "ip": host_of(serial)}
        try:
            info["model"] = next((d["model"] for d in self.devices() if d["serial"] == serial), "")
            pid = self._id_of(serial)
            if pid:
                with self._lock:
                    self.phone_id = pid
            if not info["ip"]:
                code, out = self._shell("ip", "-4", "-o", "addr", "show", "wlan0", serial=serial)
                m = re.search(r"inet (\d{1,3}(?:\.\d{1,3}){3})/", out)
                info["ip"] = m.group(1) if m else ""
        except LinkError:
            pass
        with self._lock:
            self._info[serial] = info

    def _ping(self, serial):
        """The round trip, kept; False when none came back."""
        ping = getattr(self.run, "ping", None)
        if ping is None:
            return True
        try:
            ms = float(ping(serial))
        except (OSError, LinkError, ValueError):
            return False
        with self._lock:
            self.rtt.append(ms)
        return True

    def _event(self, what: str) -> None:
        with self._lock:
            self.events.append((time.strftime("%H:%M:%S"), what))
        self.log(f"link {what}")

    def _set(self, state: str, cause: str) -> None:
        """Under the lock. Listeners hear of a change of state, cause or link in use, with the status of that
        moment."""
        if (state, cause) != (self.state, self.cause):
            self.state, self.cause, self.since = state, cause, self.clock()
        if (state, cause, self.via) != self._told:
            self._told = (state, cause, self.via)
            if self._listeners:
                self._post(self.status())

    def _notify(self) -> None:
        """The listeners hear the status as it is now (a change of mode)."""
        if self._listeners:
            self._post(self.status())

    def _post(self, status: dict) -> None:
        """A status for the listeners, delivered after those posted before it, by the notifier thread (started
        here when it is not running)."""
        with self._watch_lock:
            news = self._news
            news.put(status)
            if self._notifier is None or not self._notifier.is_alive():
                self._notifier = threading.Thread(target=self._deliver, args=(news,), daemon=True,
                                                  name="phone_link_notify")
                self._notifier.start()

    def _deliver(self, news: queue.Queue) -> None:
        me = threading.current_thread()
        while True:
            try:
                st = news.get(timeout=NOTIFY_IDLE_S)
            except queue.Empty:
                with self._watch_lock:
                    if news.empty():
                        if self._notifier is me:
                            self._notifier = None
                        return
                continue
            if st is None:                      # close()
                return
            for fn in list(self._listeners):
                try:
                    fn(st)
                except Exception:   # noqa: BLE001  a listener never stops the link
                    _LOGGER.exception("a listener of the link failed")

    def status(self) -> dict:
        with self._lock:
            info = self._info.get(self.serial_now, {}) if self.state == "ready" else {}
            rtt = sorted(self.rtt)
            return {"mode": self.mode, "via": self.via, "state": self.state, "cause": self.cause,
                    "ok": self.state == "ready", "paired": self._paired(),
                    "serial": self.serial_now if self.state == "ready" else "", "model": info.get("model", ""),
                    "phone_ip": info.get("ip", "") or host_of(self.address), "address": self.address,
                    "rtt_ms": round(rtt[len(rtt) // 2], 1) if rtt else None, "drops": self.drops,
                    "reconnects": self.reconnects, "for_s": round(self.clock() - self.since, 1),
                    "qr_waiting": self._qr is not None, "events": list(self.events)[-6:]}

    def watch(self, listener: Callable[[dict], None] | None = None, period: float = WATCH_S) -> Link:
        """Starts the thread that checks the link every `period` (once per Link, again after close); listener(status)
        on every change of state, cause or link in use, from the notifier thread, in order."""
        with self._watch_lock:
            if listener is not None and listener not in self._listeners:
                self._listeners.append(listener)
            if self._watch is None or not self._watch.is_alive():
                self._stop = threading.Event()
                self._watch = threading.Thread(target=self._loop, args=(period, self._stop), daemon=True,
                                               name="phone_link")
                self._watch.start()
        return self

    def unwatch(self, listener: Callable[[dict], None]) -> None:
        """That listener hears no more (the watch goes on)."""
        with self._watch_lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _loop(self, period: float, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.check()
            except Exception as e:   # noqa: BLE001  the watch never dies
                _LOGGER.exception("link check failed")
                self.log(f"link check: {e}")
            stop.wait(period)

    def close(self) -> None:
        """Stops the watch and the notifier (each waited for JOIN_S), forgets the listeners, cancels a QR code and
        removes the tunnels of adb reverse laid here. watch() starts again afterwards."""
        with self._watch_lock:
            watch, notifier, news = self._watch, self._notifier, self._news
            self._watch, self._notifier, self._news = None, None, queue.Queue()
            self._listeners.clear()
            self._stop.set()
        self.qr_cancel()
        if notifier is not None:
            news.put(None)
        me = threading.current_thread()
        for th in (watch, notifier):
            if th is not None and th is not me:
                th.join(JOIN_S)
                if th.is_alive():
                    _LOGGER.warning("%s still running after close", th.name)
        self.unreverse()

    # -- the phone's browser on a page of this PC
    def reverse(self, port: int) -> str:
        """The phone's own 127.0.0.1:port reaches this PC's through adb (adb reverse), on cable or Wi-Fi alike; adb
        drops it with the link, so it is laid again after a reconnection. Removed by unreverse() or close()."""
        serial = self.serial()
        code, out = self._adb(["reverse", f"tcp:{int(port)}", f"tcp:{int(port)}"], serial=serial,
                              timeout=REVERSE_TIMEOUT_S)
        if code != 0:
            raise LinkError("unreachable", out.strip()[-200:])
        with self._lock:
            self._reversed[int(port)] = serial
        return serial

    def unreverse(self, port: int | None = None) -> None:
        """adb reverse --remove: the phone's 127.0.0.1:port no longer reaches this PC (every port laid here when
        None). Never raises: a phone that is gone took its tunnels with it."""
        with self._lock:
            ports = [int(port)] if port is not None else list(self._reversed)
            serials = {p: self._reversed.pop(p, None) for p in ports}
        for p, serial in serials.items():
            try:
                self._adb(["reverse", "--remove", f"tcp:{p}"], serial=serial, timeout=UNREVERSE_TIMEOUT_S)
            except LinkError as e:
                self.log(f"adb reverse --remove tcp:{p}: {e.cause}")

    def open_url(self, url: str, port: int | None = None) -> str:
        """Opens `url` in the phone's browser. With `port`, through adb reverse first, so the page needs neither the
        network nor the PC's firewall. The URL is quoted for the phone's shell."""
        serial = self.reverse(port) if port else self.serial()
        code, out = self._shell("am", "start", "-a", "android.intent.action.VIEW", "-d", url, serial=serial,
                                timeout=OPEN_URL_TIMEOUT_S)
        if code != 0 or "Error" in out:
            raise LinkError("unreachable", out.strip()[-200:])
        return serial

    def wifi_debugging_on(self) -> bool:
        """With the phone on the cable, Wireless debugging is switched on from here (a setting adb's shell may
        write), one step less for pairing. True if the phone took it; False otherwise, never raises."""
        try:
            serial = self.serial()
            if self.via != USB:
                return False
            self._shell("settings", "put", "global", "adb_wifi_enabled", "1", serial=serial)
            code, out = self._shell("settings", "get", "global", "adb_wifi_enabled", serial=serial)
            return code == 0 and out.strip() == "1"
        except LinkError:
            return False


_DEFAULT = None
_DEFAULT_LOCK = threading.Lock()


def get() -> Link:
    """The link of this process (made at first use, in AUTO mode: the cable when the phone is on it, else Wi-Fi)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = Link()
        return _DEFAULT


def set_default(link):
    """Puts `link` in place of the process's link (the app in simulation, the tests)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is not None and _DEFAULT is not link:
            _DEFAULT.close()
        _DEFAULT = link
    return link


# ── a phone that is not there ────────────────────────────────────────────────
class SimulatedAdb:
    """adb as it answers for one phone, for the tests and the simulation mode of ImpactGuide: the phone on the cable
    or not, wireless debugging on a network that lets devices talk or not, pairing with a code or a QR code, adb's
    mDNS auto-connect, the Wi-Fi link dropping and coming back (on the same port or, when wireless debugging was
    turned on again, on a new one)."""

    def __init__(self, usb=True, code="123456", ip="192.168.43.17", phone_id="PX7P0042", model="2412DPC0AG",
                 auto_connect=True, scan_after_s=2.0, clock=time.monotonic, stale=False):
        self.usb, self.code, self.ip, self.phone_id, self.model = usb, code, ip, phone_id, model
        self.auto_connect, self.scan_after_s, self.clock = auto_connect, scan_after_s, clock
        self.stale = stale              # adb keeps listing the Wi-Fi phone as ready after it has gone
        self.network = True             # the network lets the phone and the PC talk (False: out of range, eduroam)
        self.debugging = True           # wireless debugging is on
        self.paired = False
        self.pair_port, self.connect_port = 37011, 41235
        self.connected = {}             # adb serial -> address
        self.qr = None                  # (name, password, time shown)
        self.opened = []                # URLs the phone's browser was asked to open
        self.reversed = set()
        self.calls = []
        self.timeouts = []              # the timeout of every call
        self.shell_lines = []           # what the phone's shell was given, as adb passes it
        self._lock = threading.Lock()

    # what the tests do to the phone
    def drop(self):
        """The Wi-Fi goes away (out of range, router off)."""
        self.network = False

    def back(self, new_port=False):
        """The Wi-Fi is back; with new_port, wireless debugging was turned on again and listens on another port."""
        self.network = True
        if new_port:
            self.connect_port += 1
            self.connected.clear()

    def show_qr(self, payload):
        m = re.fullmatch(r"WIFI:T:ADB;S:([^;]+);P:([^;]+);;", payload)
        self.qr = (m.group(1), m.group(2), self.clock()) if m else None

    def ping(self, serial):
        if not self._reachable(serial):
            raise OSError("closed")
        return 3.0 if transport_of(serial) == USB else 9.0

    def _addr(self, port):
        return f"{self.ip}:{port}"

    def _mdns_name(self):
        return f"adb-{self.phone_id}-sim01.{MDNS_CONNECT}"

    def _reachable(self, serial):
        if transport_of(serial) == USB:
            return self.usb and serial == self.phone_id
        addr = self.connected.get(serial)
        return bool(addr) and self.network and self.debugging and addr == self._addr(self.connect_port)

    def __call__(self, args, serial=None, timeout=ADB_TIMEOUT_S):
        with self._lock:
            self.calls.append((serial, tuple(args)))
            self.timeouts.append(timeout)
            return self._answer([str(a) for a in args], serial)

    def _answer(self, args, serial):
        cmd = args[0]
        if self.auto_connect and self.paired and self.network and self.debugging and not self.connected:
            self.connected[self._mdns_name()] = self._addr(self.connect_port)
        if cmd == "devices":
            lines = ["List of devices attached"]
            if self.usb:
                lines.append(f"{self.phone_id}\tdevice product:sim model:{self.model} device:sim transport_id:1")
            for s in self.connected:
                state = "device" if self._reachable(s) or self.stale else "offline"
                lines.append(f"{s}\t{state} product:sim model:{self.model} device:sim transport_id:2")
            return 0, "\n".join(lines) + "\n"
        if cmd == "mdns":
            lines = ["List of discovered mdns services"]
            if self.network and self.debugging:
                if self.paired:
                    lines.append(f"adb-{self.phone_id}-sim01\t{MDNS_CONNECT}\t{self._addr(self.connect_port)}")
                if self.qr and self.clock() - self.qr[2] >= self.scan_after_s:
                    lines.append(f"{self.qr[0]}\t{MDNS_PAIRING}\t{self._addr(self.pair_port)}")
            return 0, "\n".join(lines) + "\n"
        if cmd == "pair":
            addr, code = args[1], args[2] if len(args) > 2 else ""
            if not self.network or host_of(addr) != self.ip:
                return 1, "Failed: Unable to start pairing client.\n"
            want = self.qr[1] if self.qr and addr == self._addr(self.pair_port) and self.qr[1] == code else self.code
            if code != want or addr != self._addr(self.pair_port):
                return 1, "Failed: Wrong password or connection was dropped.\n"
            self.paired, self.qr = True, None
            return 0, f"Successfully paired to {addr} [guid=adb-{self.phone_id}-sim01]\n"
        if cmd == "connect":
            addr = args[1]
            if not self.network or host_of(addr) != self.ip:
                return 1, f"failed to connect to '{addr}': Connection timed out\n"
            if addr != self._addr(self.connect_port) or not self.debugging:
                return 1, f"failed to connect to '{addr}': Connection refused\n"
            if not self.paired:
                return 1, f"failed to authenticate to {addr}\n"
            if self.connected.get(addr) == addr:
                return 0, f"already connected to {addr}\n"
            self.connected[addr] = addr
            return 0, f"connected to {addr}\n"
        if cmd == "disconnect":
            self.connected.pop(args[1], None)
            return 0, f"disconnected {args[1]}\n"
        if serial is not None and not self._reachable(serial):
            return 1, f"adb: device '{serial}' not found\n"
        if cmd == "reverse":
            if args[1] == "--remove":
                self.reversed.discard(args[2])
            elif args[1] == "--remove-all":
                self.reversed.clear()
            else:
                self.reversed.add(args[1])
            return 0, ""
        if cmd == "shell":
            words = shlex.split(" ".join(args[1:]))     # the phone's sh reads the line as adb passes it
            self.shell_lines.append(" ".join(args[1:]))
            line = " ".join(words)
            if "ro.serialno" in line:
                return 0, self.phone_id + "\n"
            if line.startswith("ip "):
                return 0, f"30: wlan0    inet {self.ip}/24 brd 192.168.43.255 scope global wlan0\n"
            if "android.intent.action.VIEW" in line:
                self.opened.append(words[-1])
                return 0, "Starting: Intent { act=android.intent.action.VIEW }\n"
            if line.startswith("settings put global adb_wifi_enabled"):
                self.debugging = words[-1] == "1"
                return 0, ""
            if line.startswith("settings get global adb_wifi_enabled"):
                return 0, ("1" if self.debugging else "0") + "\n"
            return 0, ""
        return 0, ""


class FakeClock:
    """Time the tests move by hand."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


# ── command line ─────────────────────────────────────────────────────────────
def run_test():
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    def raises(fn, cause):
        try:
            fn()
        except LinkError as e:
            return e.cause == cause
        return False

    sample = ("* daemon not running; starting now at tcp:5037\n* daemon started successfully\n"
              "List of devices attached\n"
              "PX7P0042\tdevice product:garnet model:2412DPC0AG device:garnet transport_id:1\n"
              "192.168.43.17:41235\tdevice product:garnet model:2412DPC0AG device:garnet transport_id:3\n"
              "adb-PX7P0042-AbC12x._adb-tls-connect._tcp\tdevice product:garnet model:2412DPC0AG transport_id:4\n"
              "10.0.0.9:5555\toffline\nemulator-5554\tdevice\nR5CT0001\tunauthorized transport_id:5\n")
    d = parse_devices(sample)
    expect([x["transport"] for x in d] == [USB, WIFI, WIFI, WIFI, "other", USB]
           and d[0]["model"] == "2412DPC0AG" and d[3]["state"] == "offline",
           "adb devices: cable, address, mDNS name, offline, emulator and unauthorized told apart")
    m = parse_mdns("List of discovered mdns services\nadb-PX7P0042-AbC12x\t_adb-tls-connect._tcp\t192.168.43.17:41235\n"
                   "impactguide-a1b2c3\t_adb-tls-pairing._tcp.\t192.168.43.17:37011\n")
    expect(m == [{"name": "adb-PX7P0042-AbC12x", "service": MDNS_CONNECT, "address": "192.168.43.17:41235"},
                 {"name": "impactguide-a1b2c3", "service": MDNS_PAIRING, "address": "192.168.43.17:37011"}],
           "adb mdns services: connection and pairing services with their addresses")
    expect(parse_pair("Successfully paired to 1.2.3.4:5 [guid=adb-PX7P0042-AbC12x]") == (True, "", "PX7P0042")
           and parse_pair("Failed: Wrong password or connection was dropped.")[1] == "pair_code"
           and parse_pair("Failed: Unable to start pairing client.")[1] == "unreachable"
           and parse_connect("connected to 1.2.3.4:5")[0] and parse_connect("already connected to 1.2.3.4:5")[0]
           and parse_connect("failed to authenticate to 1.2.3.4:5")[1] == "not_paired"
           and parse_connect("failed to connect to '1.2.3.4:5': Connection refused")[1] == "unreachable",
           "adb pair and adb connect answers read as causes")

    # one place picks the serial: the cable or Wi-Fi, never both
    clock = FakeClock()
    sim = SimulatedAdb(clock=clock, auto_connect=False)
    link = Link(USB, runner=sim, clock=clock, sleep=clock.sleep)
    expect(link.serial() == "PX7P0042", "cable: the phone on USB")
    link.configure(mode=WIFI)
    expect(raises(link.serial, "wifi_none") and link.check()["state"] == "unpaired",
           "Wi-Fi, never paired: no serial, the steps to pair are due")
    expect(raises(lambda: link.pair_code("192.168.43.17:37011", "000000"), "pair_code"),
           "pairing with a wrong code: said as a wrong code")
    expect(raises(lambda: link.pair_code("192.168.43.17:37011", "12 34"), "pair_code"),
           "a code that is not 6 digits is refused before adb")
    sim.network = False
    expect(raises(lambda: link.pair_code("192.168.43.17:37011", "123456"), "unreachable"),
           "pairing on a network that keeps devices apart: unreachable")
    sim.network = True
    serial = link.pair_code("192.168.43.17:37011", "123456")
    st = link.status()
    expect(serial == "192.168.43.17:41235" and st["state"] == "ready" and st["address"] == serial
           and link.phone_id == "PX7P0042" and st["model"] == "2412DPC0AG" and st["rtt_ms"] == 9.0,
           f"pairing with the code, then connected by the app to the announced service ({serial})")
    expect(link.serial() == serial, "Wi-Fi chosen with the phone also on the cable: the Wi-Fi serial")
    link.configure(mode=USB)
    expect(link.serial() == "PX7P0042", "back to the cable: the USB serial, the Wi-Fi entry left alone")
    link.configure(mode=WIFI)

    # the same phone twice on Wi-Fi (its address and its mDNS name) is one phone
    sim.connected[sim._mdns_name()] = sim._addr(sim.connect_port)
    expect(link.serial() in (serial, sim._mdns_name()), "the same phone under its address and its mDNS name: one phone")
    del sim.connected[sim._mdns_name()]

    # the Wi-Fi drops and comes back on the same port
    link.check()
    sim.drop()
    st = link.check()
    expect(st["state"] == "reconnecting" and st["cause"] == "wifi_reconnecting" and st["drops"] == 1
           and raises(link.serial, "wifi_reconnecting"), "Wi-Fi dropped: reconnecting, said plainly")
    clock.t += LOST_AFTER_S + 1
    st = link.check()
    expect(st["state"] == "lost" and st["cause"] == "wifi_lost", f"still gone after {LOST_AFTER_S:.0f} s: lost")
    sim.back()
    for _ in range(4):
        clock.t += BACKOFF_S[-1]
        st = link.check()
    expect(st["state"] == "ready" and st["reconnects"] == 1 and link.serial() == serial,
           "Wi-Fi back on the same port: reconnected by itself")
    # a check under way elsewhere (a reconnection can take 10 s): wait_ready still sees its stop, in a second or so
    with link._check_cond:
        link._checking = True
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    t0 = time.perf_counter()
    got = link.wait_ready(60, stop=stop)
    waited = time.perf_counter() - t0
    with link._check_cond:
        link._checking = False
    expect(got in (None, serial) and waited < WAIT_SLICE_S + 1.5,
           f"another check under way: wait_ready waits for it in slices, never {CHECK_WAIT_S:.0f} s ({waited:.1f} s)")

    # wireless debugging turned on again: a new port, found again over mDNS
    sim.drop()
    link.check()
    sim.back(new_port=True)
    for _ in range(4):
        clock.t += BACKOFF_S[-1]
        st = link.check()
    expect(st["state"] == "ready" and st["address"] == "192.168.43.17:41236" and st["reconnects"] == 2,
           f"Wi-Fi back on a new port: found over mDNS and reconnected ({st['address']})")

    # attempts are spaced: no reconnection storm while the phone is away
    sim.drop()
    link.check()
    n0 = sum(1 for c in sim.calls if c[1][0] == "connect")
    for _ in range(10):
        clock.t += 0.2
        link.check()
    n1 = sum(1 for c in sim.calls if c[1][0] == "connect")
    expect(n1 - n0 <= 2, f"attempts spaced by the backoff: {n1 - n0} connections tried in 2 s")
    sim.back()
    expect(link.wait_ready(30) == "192.168.43.17:41236",
           "wait_ready: the stream waits for the link and gets the serial back")

    # a Wi-Fi link that dies without adb noticing: adb still lists the phone, no round trip comes back
    sim.stale = True
    link.check()
    sim.drop()
    first = link.check()
    second = link.check()
    expect(first["state"] == "ready" and second["state"] in ("reconnecting", "lost") and not link.reachable(),
           f"silent drop (adb still lists the phone): seen at the second round trip missed ({second['state']})")
    sim.back()
    sim.stale = False
    for _ in range(4):
        clock.t += BACKOFF_S[-1]
        st = link.check()
    expect(st["state"] == "ready" and link.reachable(), "and reconnected when the Wi-Fi is back")

    # pairing with a QR code, with adb's own auto-connect (mDNS name as serial)
    clock2 = FakeClock()
    sim2 = SimulatedAdb(usb=False, clock=clock2, scan_after_s=3.0)
    link2 = Link(WIFI, runner=sim2, clock=clock2, sleep=clock2.sleep)
    payload = link2.qr_start()
    first = link2.qr_poll()
    clock2.t += 3.5
    second = link2.qr_poll()
    st = link2.status()
    expect(re.fullmatch(r"WIFI:T:ADB;S:impactguide-[0-9a-f]{6};P:[A-Za-z0-9]{10};;", payload) and first == "waiting"
           and second == "ready" and st["state"] == "ready" and st["serial"] == sim2._mdns_name()
           and st["address"] == "192.168.43.17:41235",
           "QR code: waiting, then the phone scans it, pairs and connects (adb auto-connect)")
    link2.qr_start()
    sim2.qr = None
    clock2.t += QR_TIMEOUT_S + 1
    expect(raises(link2.qr_poll, "qr_timeout") and not link2.qr_waiting, "a QR code nobody scans times out")
    sim2.usb = True
    link2.configure(mode=USB)
    expect(raises(link2.serial, "no_device") is False and link2.serial() == "PX7P0042",
           "cable chosen while the phone is also on Wi-Fi: the cable")
    sim2.usb = False
    expect(raises(link2.serial, "no_device"), "cable chosen, no cable: no_device")

    # AUTO: the cable when the phone is on it, else the Wi-Fi it was paired on, found and followed by itself
    clock4 = FakeClock()
    sim4 = SimulatedAdb(clock=clock4, auto_connect=False)
    link4 = Link(runner=sim4, clock=clock4, sleep=clock4.sleep)
    heard4 = []
    link4._listeners.append(lambda s: heard4.append(s["via"]))
    st = link4.check()
    expect(link4.mode == AUTO and st["via"] == USB and st["state"] == "ready" and st["serial"] == "PX7P0042"
           and not st["paired"], "auto: the phone on the cable, found at once, nothing paired yet")
    sim4.debugging = False
    expect(link4.wifi_debugging_on() and sim4.debugging, "auto, on the cable: Wireless debugging switched on from the PC")
    link4.pair_code(f"{sim4.ip}:{sim4.pair_port}", sim4.code)
    st = link4.check()
    expect(st["via"] == USB and st["serial"] == "PX7P0042" and st["paired"],
           "auto, paired on Wi-Fi with the cable in: still the cable")
    sim4.usb = False
    st = link4.check()
    expect(st["via"] == WIFI and st["state"] == "ready" and st["serial"] == f"{sim4.ip}:{sim4.connect_port}"
           and not link4.wifi_debugging_on(), f"auto, cable out: the paired Wi-Fi at once ({st['serial']})")
    sim4.usb = True
    st = link4.check()
    expect(st["via"] == USB and st["state"] == "ready", "auto, cable in again: the cable")
    sim4.usb = False
    link4.check()
    sim4.drop()
    st = link4.check()
    dropped = st["via"] == WIFI and st["state"] == "reconnecting"
    sim4.back()
    for _ in range(4):
        clock4.t += BACKOFF_S[-1]
        st = link4.check()
    expect(dropped and st["state"] == "ready" and st["via"] == WIFI, "auto, on Wi-Fi: a drop, then back by itself")
    time.sleep(0.2)
    expect(USB in heard4 and WIFI in heard4, f"auto: the listeners hear the link in use change ({heard4})")
    link5 = Link(runner=SimulatedAdb(usb=False))
    st = link5.check()
    expect(st["state"] == "none" and st["cause"] == "no_device" and st["via"] == USB and not st["paired"],
           "auto, no phone and never paired: no phone, said as such")

    # the browser of the phone on a page of this PC
    link.configure(mode=USB)
    link.open_url("http://127.0.0.1:8790/v/abc/", port=8790)
    expect(sim.opened == ["http://127.0.0.1:8790/v/abc/"] and "tcp:8790" in sim.reversed,
           "open_url: adb reverse, then the phone's browser on the page")

    # listeners hear the changes, from the watch thread
    heard = []
    sim3 = SimulatedAdb()
    link3 = Link(USB, runner=sim3)
    link3.watch(lambda s: heard.append(s["state"]), period=0.05)
    time.sleep(0.3)
    sim3.usb = False
    time.sleep(0.3)
    watcher = link3._watch
    link3.close()
    expect("ready" in heard and heard[-1] == "none", f"the watch tells the app: {heard}")
    expect(set(CAUSES) >= {"no_device", "wifi_lost", "pair_code", "unreachable", "qr_timeout"}, "causes listed")
    test_threads(expect, watcher)
    test_commands(expect)
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed")
    return 1 if fails else 0


class _SlowAdb(SimulatedAdb):
    """A simulated phone whose adb answers `devices` slowly, counting how many calls run at once."""

    def __init__(self, delay_s=0.05, **kw):
        super().__init__(**kw)
        self.delay_s, self.running, self.most = delay_s, 0, 0
        self._count = threading.Lock()

    def __call__(self, args, serial=None, timeout=ADB_TIMEOUT_S):
        if args and args[0] == "devices":
            with self._count:
                self.running += 1
                self.most = max(self.most, self.running)
            time.sleep(self.delay_s)
            with self._count:
                self.running -= 1
        return super().__call__(args, serial, timeout)


def test_threads(expect, watcher):
    """The watch, the listeners and concurrent checks."""
    expect(watcher is not None and not watcher.is_alive(), "close() stops the watch and waits for it")
    # listeners: one notifier thread, every change in order, with the status of that moment
    sim = SimulatedAdb()
    link = Link(USB, runner=sim)
    heard, threads = [], set()

    def listener(st):
        heard.append((st["state"], st["cause"]))
        threads.add(threading.current_thread().name)
        time.sleep(0.02)                    # a slow listener: the changes still arrive in order
    link._listeners.append(listener)
    link.check()
    for usb in (False, True, False, True):
        sim.usb = usb
        link.check()
    t_end = time.time() + 3
    while len(heard) < 5 and time.time() < t_end:
        time.sleep(0.02)
    expect(heard == [("ready", ""), ("none", "no_device"), ("ready", ""), ("none", "no_device"), ("ready", "")]
           and threads == {"phone_link_notify"},
           f"listeners hear every change in order, from one notifier thread ({len(heard)} changes, {threads})")
    # the watch starts again after close; a listener taken off hears no more
    got = []
    link.watch(got.append, period=0.05)
    time.sleep(0.15)
    link.close()
    first = link._watch
    link.watch(got.append, period=0.05)
    again = link._watch is not None and link._watch.is_alive() and first is None
    link.unwatch(got.append)
    n = len(got)
    sim.usb = False
    time.sleep(0.3)
    link.close()
    expect(again and len(got) == n, "watch() starts again after close(); unwatch() takes a listener off")
    # one check at a time: a caller that comes during a check gets its result
    slow = _SlowAdb(delay_s=0.1)
    link = Link(USB, runner=slow)
    out = []
    ths = [threading.Thread(target=lambda: out.append(link.check()["state"])) for _ in range(6)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(10)
    expect(slow.most == 1 and out == ["ready"] * 6, f"checks from 6 threads: one at a time ({slow.most} at once), "
                                                     "each gets the state")


def test_commands(expect):
    """What reaches adb and the phone's shell: quoted, with timeouts, secrets kept out of the log, tunnels
    removed."""
    sim = SimulatedAdb()
    lines = []
    link = Link(USB, runner=sim, log=lines.append)
    url = "http://127.0.0.1:8790/v/a&b;reboot/?x=1 2"
    link.open_url(url, port=8790)
    line = next(x for x in sim.shell_lines if "VIEW" in x)
    expect(sim.opened[-1] == url and shlex.split(line)[-1] == url and "'" in line,
           "open_url: the address quoted for the phone's shell, & and ; kept inside it")
    link.close()
    expect(not sim.reversed, "close() removes the tunnels of adb reverse it laid")
    expect(all(t is not None and 0 < t <= 60 for t in sim.timeouts), "every adb call has a timeout")
    link.configure(mode=WIFI)
    link.pair_code(f"{sim.ip}:{sim.pair_port}", sim.code)
    expect(lines and not any(sim.code in x for x in lines), "the pairing code never reaches the log")
    old = link._qr if link.qr_start() else None
    link.qr_start()
    link._qr_cancel_if(old)
    expect(link.qr_waiting, "a new QR code is not cancelled by the pairing of the old one")
    link.close()
    clock = FakeClock()
    sim_qr, qr_lines = SimulatedAdb(usb=False, clock=clock, scan_after_s=1.0), []
    link_qr = Link(WIFI, runner=sim_qr, clock=clock, sleep=clock.sleep, log=qr_lines.append)
    password = re.search(r"P:([^;]+);", link_qr.qr_start()).group(1)
    clock.t += 2
    paired = link_qr.qr_poll()
    expect(paired == "ready" and any("pair" in x and "***" in x for x in qr_lines)
           and not any(password in x for x in qr_lines), "the QR code's password never reaches the log")
    try:
        AdbRunner(exe=sys.executable)(["-c", "import time; time.sleep(5)", "123456"], timeout=0.5)
        detail = None
    except LinkError as e:
        detail = (e.cause, e.detail)
    expect(detail == ("timeout", "-c"), f"a timeout says the command, not its arguments ({detail})")
    vs = ["scrcpy-win64-v3.9", "scrcpy-win64-v3.10", "scrcpy-win64-v2.7"]
    expect(sorted(vs, key=_version, reverse=True)[0].endswith("v3.10")
           and transport_of("localhost:5555") == WIFI and transport_of("PX7P0042") == USB,
           "scrcpy versions in number order; localhost:5555 is not the cable")
    old_port = os.environ.get("ANDROID_ADB_SERVER_PORT")
    os.environ["ANDROID_ADB_SERVER_PORT"] = "5099"
    try:
        port = adb_port()
    finally:
        if old_port is None:
            os.environ.pop("ANDROID_ADB_SERVER_PORT")
        else:
            os.environ["ANDROID_ADB_SERVER_PORT"] = old_port
    expect(port == 5099 and adb_port() == int(old_port or ADB_PORT), "the adb server's port from the environment")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["devices", "pair", "connect", "watch", "test"])
    ap.add_argument("address", nargs="?")
    ap.add_argument("code", nargs="?")
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args()
    if a.command == "test":
        sys.exit(run_test())
    link = Link(WIFI if a.command in ("pair", "connect") else USB, log=print if a.command != "devices" else None)
    try:
        if a.command == "devices":
            print(f"adb: {link.run.exe}")
            for d in link.devices():
                print(f"{d['serial']:48s} {d['state']:12s} {d['transport']:5s} {d['model']}")
            for s in link.mdns():
                print(f"mdns {s['name']:40s} {s['service']:24s} {s['address']}")
        elif a.command == "pair":
            print(link.pair_code(a.address or "", a.code or ""))
        elif a.command == "connect":
            print(link.connect(a.address))
        else:
            link.configure(mode=WIFI if any(d["transport"] == WIFI for d in link.devices()) else USB)
            t0 = time.time()
            while time.time() - t0 < a.seconds:
                st = link.check()
                print(f"{time.time() - t0:5.1f} s  {st['mode']:4s} {st['state']:12s} {st['cause']:18s} "
                      f"rtt {st['rtt_ms']} ms  drops {st['drops']}  reconnects {st['reconnects']}", flush=True)
                time.sleep(WATCH_S)
    except LinkError as e:
        sys.exit(f"phone_link: {e.cause}" + (f" ({e.detail})" if e.detail else ""))


if __name__ == "__main__":
    main()
