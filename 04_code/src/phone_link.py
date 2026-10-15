"""The link to the phone, by cable (USB) or by Wi-Fi (Android 11+ wireless debugging), chosen in ImpactGuide (T20.3).

The one place that knows which device serial to use. Everything that talks to the phone asks here: live.py (the
camera stream), acquisition.py (photos and videos), live_ar.py (closing the camera app), the app (the choice, the
pairing, the state of the link). One adb for all of them: the one shipped with scrcpy.

    link = phone_link.get()                          # the link of this process
    link.configure(mode="wifi")                      # or "usb"; the app keeps the choice in its preferences
    link.serial()                                    # the phone on the chosen link, or LinkError(cause)
    link.pair_code("192.168.1.23:37011", "123456")   # the code and the address the phone shows
    payload = link.qr_start(); link.qr_poll()        # a QR code the phone scans: "waiting" until it is paired
    link.connect("192.168.1.23:41235")               # or found by itself over mDNS
    link.watch(listener)                             # checks the link every WATCH_S; brings Wi-Fi back when it drops
    link.status()                                    # mode, state, cause, serial, model, address, round trip, drops
    link.open_url(url, port)                         # the phone's browser on a page of this PC, through adb

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
import argparse
import glob
import os
import re
import secrets
import shutil
import socket
import string
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

USB, WIFI = "usb", "wifi"
MODES = (USB, WIFI)
MDNS_CONNECT = "_adb-tls-connect._tcp"
MDNS_PAIRING = "_adb-tls-pairing._tcp"
WATCH_S = 2.0                   # the watch looks at the link this often
BACKOFF_S = (1.0, 2.0, 4.0, 8.0)  # between reconnection attempts; then every 8 s, for as long as Wi-Fi is chosen
LOST_AFTER_S = 20.0             # without the phone this long the link is "lost" (the attempts go on)
QR_TIMEOUT_S = 180.0            # a QR code waits this long for the phone to scan it
PAIR_TIMEOUT_S = 30.0
CONNECT_AFTER_PAIR_S = 12.0     # after pairing, the phone's connection service is waited for this long
NO_WINDOW = 0x08000000 if os.name == "nt" else 0    # CREATE_NO_WINDOW: no console flash under pythonw
ADDRESS_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3}|\[[0-9A-Fa-f:]+\]):(\d{1,5})")
STATES = ("device", "offline", "unauthorized", "authorizing", "connecting", "recovery", "sideload", "bootloader",
          "host", "rescue", "unknown")
# The causes the interface names (strings link.cause.<cause> in ui/strings).
CAUSES = ("no_adb", "no_device", "unauthorized", "many_devices", "offline", "timeout", "wifi_none", "wifi_connecting",
          "wifi_reconnecting", "wifi_lost", "pair_code", "pair_failed", "unreachable", "not_paired",
          "paired_no_connect", "qr_timeout", "bad_address", "mdns_off")


class LinkError(RuntimeError):
    """A failure of the link with a cause the interface can name (CAUSES); detail is adb's text, for the log."""

    def __init__(self, cause, detail=""):
        super().__init__(f"{cause}: {detail}" if detail else cause)
        self.cause, self.detail = cause, detail


# ── adb ──────────────────────────────────────────────────────────────────────
def scrcpy_dir():
    """Folder holding scrcpy-server and adb.exe (IMPACTGUIDE_SCRCPY, scrcpy on the PATH, the WinGet packages), or
    None."""
    cands = [os.environ.get("IMPACTGUIDE_SCRCPY")]
    exe = shutil.which("scrcpy")
    if exe:
        cands.append(str(Path(exe).resolve().parent))
    local = os.environ.get("LOCALAPPDATA", "")
    cands += sorted(glob.glob(os.path.join(local, "Microsoft", "WinGet", "Packages", "Genymobile.scrcpy*",
                                           "scrcpy-win64-*")), reverse=True)
    adb = "adb.exe" if os.name == "nt" else "adb"
    for c in cands:
        if c and (Path(c) / "scrcpy-server").is_file() and (Path(c) / adb).is_file():
            return Path(c)
    return None


def find_adb():
    """The adb of the whole app: IMPACTGUIDE_ADB, the one shipped with scrcpy (the stream needs its server, and two
    adb versions stop each other's server), the PATH, the one bundled with adbutils, the Android SDK. None if none."""
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


class AdbRunner:
    """Runs adb: (exit code, stdout and stderr together). LinkError("no_adb") without adb, ("timeout") on timeout."""

    def __init__(self, exe=None):
        self.exe = exe or find_adb()

    def __call__(self, args, serial=None, timeout=20.0):
        if not self.exe:
            raise LinkError("no_adb")
        cmd = [self.exe] + (["-s", serial] if serial else []) + [str(a) for a in args]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, creationflags=NO_WINDOW)
        except subprocess.TimeoutExpired:
            raise LinkError("timeout", " ".join(str(a) for a in args))
        except OSError as e:
            raise LinkError("no_adb", str(e))
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    def ping(self, serial, port=5037, timeout=3.0):
        """Round trip to the phone in ms: a shell `echo` through the adb server's socket (no process started here,
        so the time is the link's and adbd's)."""
        t0 = time.perf_counter()
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
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


def transport_of(serial):
    """wifi (IP:PORT, or the name adb gives a phone it found over mDNS), usb, or other (an emulator)."""
    if serial.startswith("emulator-"):
        return "other"
    if MDNS_CONNECT in serial or ADDRESS_RE.fullmatch(serial):
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
    """The chosen link and what is known of it. Thread-safe; the watch runs on its own thread."""

    def __init__(self, mode=USB, address="", phone_id="", runner=None, clock=time.monotonic, sleep=time.sleep,
                 log=None):
        self.run = runner or AdbRunner()
        self.clock, self.sleep = clock, sleep
        self.log = log or (lambda msg: None)
        self._lock = threading.RLock()
        self._attempting = threading.Lock()
        self._listeners = []
        self._watch, self._stop = None, threading.Event()
        self.mode = mode if mode in MODES else USB
        self.address = address or ""        # the last Wi-Fi address that connected (IP:PORT)
        self.phone_id = phone_id or ""      # the phone's ro.serialno, to know it again under another name
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
    def configure(self, mode=None, address=None, phone_id=None):
        with self._lock:
            changed = mode in MODES and mode != self.mode
            if changed:
                self.mode = mode
                self._reset_state()
            if address is not None:
                self.address = address
            if phone_id is not None:
                self.phone_id = phone_id
        if changed:
            self._notify()
        return self

    def remembered(self):
        """What the app keeps between sessions."""
        return {"link": self.mode, "wifi_address": self.address, "phone_id": self.phone_id}

    # -- what adb sees
    def _adb(self, args, serial=None, timeout=20.0, cause=None):
        try:
            code, out = self.run(args, serial=serial, timeout=timeout)
        except LinkError as e:
            if cause and e.cause == "timeout":
                raise LinkError(cause, e.detail)
            raise
        self.log(f"adb {' '.join(str(a) for a in args)}: {out.strip()[-200:]}")
        return code, out

    def devices(self):
        """[{serial, state, model, transport}] (LinkError no_adb or timeout)."""
        return parse_devices(self._adb(["devices", "-l"], timeout=15)[1])

    def mdns(self):
        """The adb services announced on the network: [{name, service, address}] ([] when mDNS is off)."""
        try:
            return parse_mdns(self._adb(["mdns", "services"], timeout=10)[1])
        except LinkError:
            return []

    def _id_of(self, serial):
        if serial not in self._ids:
            pid = phone_id_of(serial)
            if not pid:
                try:
                    code, out = self._adb(["shell", "getprop", "ro.serialno"], serial=serial, timeout=8)
                    pid = out.strip() if code == 0 else ""
                except LinkError:
                    pid = ""
            if pid:
                self._ids[serial] = pid
            return pid
        return self._ids[serial]

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

    def serial(self):
        """The adb serial of the phone on the chosen link. LinkError: no_adb, no_device, unauthorized, many_devices,
        offline (USB); wifi_none, wifi_connecting, wifi_reconnecting, wifi_lost, unauthorized, many_devices (Wi-Fi)."""
        found = self.devices()
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
        with self._lock:
            if self.state in ("ready", "reconnecting"):
                raise LinkError("wifi_reconnecting")
            if self.state == "lost":
                raise LinkError("wifi_lost")
            raise LinkError("wifi_connecting" if self.address or self.serial_now else "wifi_none")

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

    def wait_ready(self, timeout, stop=None, step=0.5):
        """Waits for the phone on the chosen link, helping the reconnection along (check). The serial, or None after
        the timeout or when `stop` (an Event) is set."""
        t_end = self.clock() + timeout
        while self.clock() < t_end and not (stop is not None and stop.is_set()):
            st = self.check()
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

    def _try_connect(self, address):
        code, out = self._adb(["connect", address], timeout=10, cause="unreachable")
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
                serial = self._wait_wifi(5.0, host_of(a))
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
            self.sleep(0.5)

    def _pair_result(self, out):
        ok, cause, pid = parse_pair(out)
        if not ok:
            raise LinkError(cause, out.strip()[-200:])
        if pid:
            with self._lock:
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
            self.sleep(0.5)

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
        _, out = self._adb(["pair", address, code], timeout=PAIR_TIMEOUT_S, cause="unreachable")
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
            self.qr_cancel()
            raise LinkError("qr_timeout")
        hit = next((s for s in self.mdns() if s["service"] == MDNS_PAIRING and s["name"] == qr["name"]), None)
        if hit is None:
            return "waiting"
        self.qr_cancel()
        _, out = self._adb(["pair", hit["address"], qr["password"]], timeout=PAIR_TIMEOUT_S, cause="unreachable")
        self._pair_result(out)
        self._after_pairing(host_of(hit["address"]))
        return "ready"

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
    def check(self):
        """One look at the link: its state, the round trip, and on Wi-Fi an attempt to bring it back when the phone
        is gone (the last address, then the one the phone announces), spaced by BACKOFF_S. Returns status()."""
        try:
            serial = self.serial()
        except LinkError as e:
            self._down(e)
            return self.status()
        if not self._up(serial) and self.mode == WIFI:
            # adb still lists the phone but no round trip comes back twice in a row: the Wi-Fi link died without
            # adb noticing. Dropping the entry lets the reconnection start.
            with self._lock:
                self.silent += 1
                stale = self.silent >= 2
            if stale:
                try:
                    self._adb(["disconnect", serial], timeout=10)
                except LinkError:
                    pass
                with self._lock:
                    self.silent = 0
                self._down(LinkError("wifi_reconnecting"))
        return self.status()

    def _down(self, e):
        now = self.clock()
        with self._lock:
            wifi = self.mode == WIFI
            if not wifi or e.cause in ("no_adb", "many_devices", "unauthorized", "timeout"):
                self._set("none", e.cause)
                return
            if not self.address and not self.serial_now:
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
                    self._adb(["disconnect", d["serial"]], timeout=10)
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
            back = self.mode == WIFI and self.state in ("reconnecting", "lost")
            if back:
                self.reconnects += 1
                self._event("reconnected")
            elif why:
                self._event(why)
            self.down_since, self.tries = None, 0
            new = serial != self.serial_now
            self.serial_now = serial
            if self.mode == WIFI and ADDRESS_RE.fullmatch(serial):
                self.address = serial
        if new:
            self._identify(serial)
        answered = self._ping(serial)
        with self._lock:
            if answered:
                self.silent = 0
            self._set("ready", "")
        return answered

    def _identify(self, serial):
        info = {"model": "", "ip": host_of(serial)}
        try:
            info["model"] = next((d["model"] for d in self.devices() if d["serial"] == serial), "")
            pid = self._id_of(serial)
            if pid:
                with self._lock:
                    self.phone_id = pid
            if not info["ip"]:
                code, out = self._adb(["shell", "ip", "-4", "-o", "addr", "show", "wlan0"], serial=serial, timeout=8)
                m = re.search(r"inet (\d{1,3}(?:\.\d{1,3}){3})/", out)
                info["ip"] = m.group(1) if m else ""
        except LinkError:
            pass
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

    def _event(self, what):
        self.events.append((time.strftime("%H:%M:%S"), what))
        self.log(f"link {what}")

    def _set(self, state, cause):
        """Under the lock. Listeners hear of a change of state or cause."""
        if (state, cause) != (self.state, self.cause):
            self.state, self.cause, self.since = state, cause, self.clock()
            if self._listeners:
                threading.Thread(target=self._notify, daemon=True).start()

    def _notify(self):
        st = self.status()
        for fn in list(self._listeners):
            try:
                fn(st)
            except Exception:   # noqa: BLE001  a listener never stops the link
                pass

    def status(self):
        with self._lock:
            info = self._info.get(self.serial_now, {}) if self.state == "ready" else {}
            rtt = sorted(self.rtt)
            return {"mode": self.mode, "state": self.state, "cause": self.cause, "ok": self.state == "ready",
                    "serial": self.serial_now if self.state == "ready" else "", "model": info.get("model", ""),
                    "phone_ip": info.get("ip", "") or host_of(self.address), "address": self.address,
                    "rtt_ms": round(rtt[len(rtt) // 2], 1) if rtt else None, "drops": self.drops,
                    "reconnects": self.reconnects, "for_s": round(self.clock() - self.since, 1),
                    "qr_waiting": self._qr is not None, "events": list(self.events)[-6:]}

    def watch(self, listener=None, period=WATCH_S):
        """Starts the thread that checks the link every `period` (once per Link); listener(status) on every change of
        state or cause, from that thread."""
        if listener is not None and listener not in self._listeners:
            self._listeners.append(listener)
        if self._watch is None:
            self._stop.clear()
            self._watch = threading.Thread(target=self._loop, args=(period,), daemon=True, name="phone_link")
            self._watch.start()
        return self

    def _loop(self, period):
        while not self._stop.is_set():
            try:
                self.check()
            except Exception as e:   # noqa: BLE001  the watch never dies
                self.log(f"link check: {e}")
            self._stop.wait(period)

    def close(self):
        self._stop.set()
        self.qr_cancel()

    # -- the phone's browser on a page of this PC
    def reverse(self, port):
        """The phone's own 127.0.0.1:port reaches this PC's through adb (adb reverse), on cable or Wi-Fi alike; adb
        drops it with the link, so it is laid again after a reconnection."""
        serial = self.serial()
        code, out = self._adb(["reverse", f"tcp:{port}", f"tcp:{port}"], serial=serial, timeout=10)
        if code != 0:
            raise LinkError("unreachable", out.strip()[-200:])
        return serial

    def open_url(self, url, port=None):
        """Opens `url` in the phone's browser. With `port`, through adb reverse first, so the page needs neither the
        network nor the PC's firewall."""
        serial = self.reverse(port) if port else self.serial()
        code, out = self._adb(["shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url],
                              serial=serial, timeout=15)
        if code != 0 or "Error" in out:
            raise LinkError("unreachable", out.strip()[-200:])
        return serial


_DEFAULT = None
_DEFAULT_LOCK = threading.Lock()


def get():
    """The link of this process (made at first use, on cable)."""
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

    def __call__(self, args, serial=None, timeout=20.0):
        with self._lock:
            self.calls.append((serial, tuple(args)))
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
                return 1, f"Failed: Unable to start pairing client.\n"
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
            self.reversed.add(args[1])
            return 0, ""
        if cmd == "shell":
            line = " ".join(args[1:])
            if "ro.serialno" in line:
                return 0, self.phone_id + "\n"
            if line.startswith("ip "):
                return 0, f"30: wlan0    inet {self.ip}/24 brd 192.168.43.255 scope global wlan0\n"
            if "android.intent.action.VIEW" in line:
                self.opened.append(args[-1])
                return 0, "Starting: Intent { act=android.intent.action.VIEW }\n"
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
    link3.close()
    expect("ready" in heard and heard[-1] == "none", f"the watch tells the app: {heard}")
    expect(set(CAUSES) >= {"no_device", "wifi_lost", "pair_code", "unreachable", "qr_timeout"}, "causes listed")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed")
    return 1 if fails else 0


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
