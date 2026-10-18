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
poses (calib_guide.py), each one checked as it comes.

This file is the host: the window, the bridge, the background jobs. The logic
is in 04_code/src (trials.py, acquisition.py, calibrate_charuco.py, live_ar.py,
phone_link.py, phone_view.py, screen_board.py), which the host only calls; everything there also
runs without the app. The live frames reach the page as MJPEG from 127.0.0.1
(live_ar.MjpegServer), and the phone's own screen from a port of this PC
(phone_view.ViewServer). The link to the phone is found and watched by itself
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
"""
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
import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import date, datetime  # noqa: E402

import slantui.css as slantui_css  # noqa: E402
import slantui.js as slantui_js  # noqa: E402
from slantui.shell import Application, Bridge, Splash, Window, pyqtSignal, pyqtSlot  # noqa: E402
from slantui.shell.qt import (QApplication, QColor, QDesktopServices, QFileDialog, QTimer, QUrl,  # noqa: E402
                              USE_QT6)
from slantui.tokens import PALETTES  # noqa: E402

import acquisition  # noqa: E402
import calib_guide  # noqa: E402
import calibrate_charuco  # noqa: E402
import charuco_plate  # noqa: E402
import live_ar  # noqa: E402
import phone_link  # noqa: E402
import phone_view  # noqa: E402
import screen_board  # noqa: E402
import trials  # noqa: E402

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
CALIB_MODES = ("photo", "stream")     # the phone's photos (camera app), the frames of the live stream
IMAGE_SUFFIXES = tuple(sorted(calibrate_charuco.IMAGE_SUFFIXES))
CHECK_FIELDS = ("time", "mode", "folder", "ok", "n_views", "rms_stored_px", "fx_change", "fy_change",
                "cx_change", "cy_change")
BOARD_KINDS = ("screen", "plate", "sheet")  # on this monitor, the 3D-printed plate, a printed A4 sheet
MIN_WINDOW = (720, 560)     # the app's least size
PHOTO_CHECKS = "photo_checks.json"    # the check of each photo of a series, beside the photos
REJECTED = "rejected"                 # the photos of a series put aside by their check (or by hand)
SHEET_SPREAD_MAX = 0.30     # mm between the readings of the A4 sheet's long side (a caliper, or a steel ruler)
SHEET_NOMINAL_TOL = 0.02    # a printer that scales the page by more than 2 %, or a misreading
PLATE_MODEL = APP / "ui" / "models" / "charuco_plate.glb"   # the plate in 3D for the page (tools/build_plate_model.py)
GUESS_PITCH = 25.4 / 96     # mm: a monitor that says nothing of its size is laid out as a 96 dpi one, to be measured
COMMA_LANGS = ("it", "fr", "es", "de")


# ── where things are ─────────────────────────────────────────────────────────
class Roots:
    """Data folders: the project's 03_data, or everything under one folder in simulation."""

    def __init__(self, simulate=None):
        self.simulate = Path(simulate).resolve() if simulate else None
        if self.simulate:
            base = self.simulate
            self.raw, self.protocol, self.processed = base / "raw", base / "protocol", base / "processed"
            self.staging, self.board = base / "staging", base / "board"
            self.prefs = base / "preferences.json"
            self.export_default = base / "scanner_export"
        else:
            self.raw, self.protocol = trials.RAW, trials.PROTOCOL
            self.processed = trials.ROOT / "03_data" / "processed"
            self.staging, self.board = LOCAL / "staging", calibrate_charuco.WORK
            self.prefs = LOCAL / "preferences.json"
            self.export_default = None
        self.calib_raw = self.raw / "calibration"
        self.calib_out = self.processed / "calibration"


class Prefs:
    """What outlives a session on this computer: language, page, phase, scanner folder, phone."""

    DEFAULTS = {"lang": "", "page": "trials", "phase": "campaign", "scanner_folder": "",
                "calib_mode": "photo", "calib_kind": "calibration", "calib_plan": "quick",
                "phone": {"camera_dir": acquisition.CAMERA_DIR, "shutter": "volume_down", "switch_modes": True,
                          "link": phone_link.AUTO, "wifi_address": "", "phone_id": "", "wifi_paired": False},
                "live": {"board": "sheet", "face": True, "mandible": False, "axes": True, "registration": "",
                         "opacity": live_ar.OPACITY},
                # the calibration board: on this monitor (in the page), the 3D-printed plate or a printed A4 sheet,
                # with the caliper readings of each (mm); chosen: the first question of Calibrate was answered
                "board": {"kind": "screen", "chosen": False, "plate": {}, "sheet": {}},
                # the ruler check of each monitor (screen_board.with_ruler): pitch_mm, measured_mm, data_mm, date
                "monitors": {}}

    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(json.dumps(self.DEFAULTS))
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if isinstance(v, dict) and isinstance(self.data.get(k), dict):
                    self.data[k].update(v)
                else:
                    self.data[k] = v
        except (OSError, ValueError):
            pass
        # what older versions kept: the printed board is the sheet; the video mode is now the stream's
        b = self.data["board"]
        if b.get("kind") == "printed":
            b["kind"] = "sheet"
        for k in ("monitor", "place", "split"):
            b.pop(k, None)
        if self.data.get("calib_mode") not in ("photo", "stream"):
            self.data["calib_mode"] = "photo"
        self.data.pop("square_mm", None)

    def __getitem__(self, key):
        return self.data[key]

    def set(self, key, value):
        self.data[key] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        except OSError:
            pass


def file_url(path):
    return QUrl.fromLocalFile(str(path)).toString()


def kind_of(step):
    """What a step asks for: photo, video, scan, highspeed, spheres, release, close or plain."""
    if step.files:
        return trials.file_kind(step.files[0])
    return step.key if step.key in ("spheres", "release", "close") else "plain"


# ── the bridge ───────────────────────────────────────────────────────────────
class Backend(Bridge):
    """What the page may ask. Every slot answers JSON and never raises: an exception out of a slot
    would take the window down, so a failure is an answer with an `error` cause."""

    happened = pyqtSignal(str)  # results of background jobs: {"type": ..., ...}

    def __init__(self, roots, prefs, lang=None, replay=None):
        super().__init__()
        self.roots, self.prefs = roots, prefs
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
        self._watch = None
        self._live = None
        self._live_server = None
        self._view = None           # the live AR for the phone's screen (phone_view.ViewServer), made at first use
        self._view_opened = False   # opened on the phone through adb: its tunnel is laid again after a reconnection
        self._qr = ""               # the pairing QR code on show, while it waits for the phone
        self._wifi_on = False       # Wireless debugging switched on from the PC, over the cable
        self._svgs = {}
        self._link_was = {}
        self._area = None           # the page's area for the board (board_area), device pixels
        self._board_on = False      # a step shows the board on this monitor now
        self._lay = None            # the board laid out for that area (screen_board.layout)
        self._lay_err, self._lay_mon, self._lay_exact = "", None, True
        self._board_img = None      # its image, as the page draws it
        self._screen_watch = False
        self._watched_screen = None
        self._live_for = "live"     # what the live AR runs for: the live view, or a calibration of the stream
        self._win_mons = (None, {})  # what Windows says of the monitors, read again when Qt's screens change
        self.window = None          # the app's window (build): the monitor it is on sizes the board
        self._via = None            # the link the phone was last found on (cable or Wi-Fi)
        self.link = self._make_link()
        app = QApplication.instance()
        if app is not None:         # a monitor plugged in or out: the board is laid out again
            app.screenAdded.connect(lambda *_: self._emit({"type": "board", "action": "screen"}))
            app.screenRemoved.connect(lambda *_: self._emit({"type": "board", "action": "screen"}))
        self._pump = QTimer(self)
        self._pump.setInterval(150)
        self._pump.timeout.connect(self._drain)
        self._pump.start()
        self.link.watch(self._on_link)

    # ── background jobs ──────────────────────────────────────────────────
    def _spawn(self, name, fn, done):
        """Runs fn() on a thread; done(result, error) runs on the GUI thread through the pump."""
        if name in self._busy:
            return False
        self._busy.add(name)

        def work():
            try:
                self._jobs.put((name, done, fn(), None))
            except BaseException as e:  # SystemExit from the command-line helpers as well
                self._jobs.put((name, done, None, e))

        threading.Thread(target=work, daemon=True).start()
        return True

    def _drain(self):
        while True:
            try:
                name, done, result, error = self._jobs.get_nowait()
            except queue.Empty:
                break
            self._busy.discard(name)
            try:
                done(result, error)
            except Exception:
                self._emit({"type": "error", "cause": "internal", "message": traceback.format_exc()[-800:]})
        if self._watch is not None:
            trial, key, watch = self._watch
            try:
                found = watch.poll()
            except OSError:
                found = None
            if found is not None and self.cur.get("trial") == trial:
                self._watch = None
                self._stage(key, found, temp=False)
                self._emit({"type": "staged", "key": key})

    def _emit(self, payload):
        self.happened.emit(json.dumps(payload))

    @staticmethod
    def _err(e, cause=None):
        log_error(e, cause)
        if isinstance(e, phone_link.LinkError):
            return {"error": e.cause, "cause": e.cause, "message": ""}
        if isinstance(e, acquisition.AcquisitionError):
            return {"error": e.cause, "message": e.detail}
        if isinstance(e, SystemExit):
            return {"error": cause or "failed", "message": str(e.code)}
        link_cause = getattr(e, "cause", "")      # live.LiveError: the link is why
        if link_cause in phone_link.CAUSES:
            return {"error": cause or link_cause, "cause": link_cause, "message": ""}
        return {"error": cause or type(e).__name__, "message": str(e)}

    def _answer(self, fn):
        try:
            return json.dumps(fn())
        except Exception as e:  # noqa: BLE001  a slot must never raise
            return json.dumps({**self._err(e), "trace": traceback.format_exc()[-1200:]})

    # ── the app ──────────────────────────────────────────────────────────
    @pyqtSlot(result=str)
    def app_info(self):
        if self.on_page_ready is not None:
            ready, self.on_page_ready = self.on_page_ready, None
            ready()
        return json.dumps({"name": APP_NAME, "version": APP_VERSION, "lang": self.lang,
                           "langs": [{"code": c, "native": n, "flag": f} for c, n, f in LANGUAGES],
                           "page": self.start_page,
                           "simulate": bool(self.roots.simulate), "qt": "PyQt6" if USE_QT6 else "PyQt5",
                           "link": self.link.via})

    @pyqtSlot(str)
    def set_lang(self, code):
        if code in LANGS:
            self.lang = code
            self.prefs.set("lang", code)
            if self._view is not None:
                self._view.words, self._view.lang = view_words(code), code

    @pyqtSlot(str)
    def set_page(self, page):
        if page in PAGES:
            self.prefs.set("page", page)

    @pyqtSlot(str, result=str)
    def open_path(self, path):
        def go():
            p = Path(path)
            if not p.exists():
                return {"error": "not_found", "message": path}
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
            return {"ok": True}
        return self._answer(go)

    # ── the phone ────────────────────────────────────────────────────────
    def phone(self):
        if self._phone is None:
            if self.roots.simulate:
                self._phone = acquisition.FakePhone(photo_px=self.run().config.photo_px, link=self.link)
            else:
                p = self.prefs["phone"]
                self._phone = acquisition.Phone(camera_dir=p["camera_dir"], shutter=p["shutter"],
                                                switch_modes=bool(p["switch_modes"]), link=self.link)
        return self._phone

    @pyqtSlot()
    def phone_refresh(self):
        def done(status, error):
            self.phone_status = status if status else {"ok": False, **self._err(error)}
            self.phone_status["link"] = self.link.via
            self._emit({"type": "phone", **self.phone_status})
        self._spawn("phone_status", lambda: self.phone().status(), done)

    @pyqtSlot(result=str)
    def phone_prefs(self):
        return json.dumps({**self.prefs["phone"], "adb": acquisition.find_adb() or "", "status": self.phone_status})

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

    def _on_link(self, status):
        """From the link's own thread: handed to the GUI thread through the pump."""
        self._jobs.put(("", self._link_changed, status, None))

    def _link_changed(self, st, _error):
        was, self._link_was = self._link_was, st
        kept = self.link.remembered()
        if kept != {k: self.prefs["phone"].get(k) for k in kept}:
            self.prefs.set("phone", {**self.prefs["phone"], **kept})
        self.phone_status = {**self.phone_status, "ok": st["ok"], "cause": st["cause"] or ("" if st["ok"] else "unknown"),
                             "link": st["via"], "state": st["state"], "model": st["model"], "serial": st["serial"]}
        self._emit({"type": "phone", **self.phone_status})
        back = st["ok"] and was.get("state") in ("reconnecting", "lost") and was.get("via") == st["via"]
        self._emit({"type": "link", "action": "state", "back": back, **st})
        if st["ok"] and self._via is not None and st["via"] != self._via:
            # the phone is now on the other link (cable plugged in or out): the app goes on with it, a live view too
            self._emit({"type": "link", "action": "via", "via": st["via"]})
            if self._live is not None and self._live.running and not self._live.synthetic:
                self._live.stop()
                self._spawn("live", self._live.start,
                            lambda _, error: self._emit({"type": "live", "action": "failed", **self._err(error, "live_failed")}
                                                        if error is not None else {"type": "live", "action": "started"}))
        if st["ok"]:
            self._via = st["via"]
        if back and self._view_opened and self._view is not None:
            # adb drops its tunnels with the link: the phone's page goes on once the tunnel is laid again
            port = self._view.port
            self._spawn("view_tunnel", lambda: self.link.reverse(port), lambda r, e: None)

    def _link_state(self):
        st = self.link.status()
        live = self._live.status() if self._live is not None and self._live.running else {}
        st["stream"] = {"running": bool(live), "fps": live.get("fps_stream", 0.0),
                        "gap_p95_ms": live.get("gap_p95_ms"), "waiting": live.get("link_waiting", False),
                        "link": live.get("link")}
        st["latency_ms"] = live_ar.STREAM_LATENCY_MS.get(st["mode"])
        st["qr"] = self._svg(self._qr) if self._qr and st["qr_waiting"] else ""
        st["busy"] = sorted(self._busy & {"link", "pair", "qr", "connect", "view_open", "wifi_on"})
        st["wifi_on"] = self._wifi_on
        st["simulate"] = bool(self.roots.simulate)
        st["view"] = self._view_state()
        return st

    @pyqtSlot(result=str)
    def link_state(self):
        return self._answer(self._link_state)

    @pyqtSlot(result=str)
    def link_wifi(self):
        """Wi-Fi wanted: with the phone on the cable, Wireless debugging is switched on from here (one step less on
        the phone), then a pairing QR code is shown; the phone that scans it is paired and connected, and the app
        goes on with it by itself when the cable is unplugged."""
        def go():
            def done(on, _error):
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
    def link_mode(self, mode):
        """auto (the cable when the phone is on it, else Wi-Fi), or only the cable, or only Wi-Fi: for the tests and
        the command line; the panel never asks. A live view running on the other link stops."""
        def go():
            if mode not in phone_link.MODES:
                return {"error": "bad_value", "message": "link"}
            if mode != self.link.mode:
                if self._live is not None and self._live.running and not self._live.synthetic:
                    self._live.stop()
                    self._emit({"type": "live", "action": "stopped", "reason": "link"})
                self.link.configure(mode=mode)
                self.prefs.set("phone", {**self.prefs["phone"], **self.link.remembered()})
                self._phone = None
                self._spawn("link", self.link.check, lambda r, e: None)
            return self._link_state()
        return self._answer(go)

    def _link_job(self, name, fn):
        def done(_, error):
            if error is not None:
                self._emit({"type": "link", "action": "failed", "job": name, **self._err(error, "pair_failed")})
            else:
                self.prefs.set("phone", {**self.prefs["phone"], **self.link.remembered()})
                self._emit({"type": "link", "action": "paired" if name in ("pair", "qr") else "connected"})
        return self._spawn(name, fn, done)

    @pyqtSlot(str, str, result=str)
    def link_pair(self, address, code):
        """Pairing with the code and the address the phone shows, then the connection; 'link' events say how."""
        def go():
            if not self._link_job("pair", lambda: self.link.pair_code(address, code)):
                return {"error": "busy"}
            return self._link_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def link_qr(self):
        """A new pairing QR code; the phone that scans it is paired and connected in the background."""
        def go():
            self._qr = self.link.qr_start()     # a job already waiting goes on with the new code

            def done(result, error):
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
    def link_qr_cancel(self):
        def go():
            self._qr = ""
            self.link.qr_cancel()
            return self._link_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def link_connect(self, address):
        """adb connect to the address the phone shows under IP address & Port (empty: the one it announces)."""
        def go():
            if not self._link_job("connect", lambda: self.link.connect(address)):
                return {"error": "busy"}
            return self._link_state()
        return self._answer(go)

    # ── the live view on the phone's own screen ──────────────────────────
    def _svg(self, text):
        """A QR code as SVG, kept: the pane asks for its state every second."""
        if text not in self._svgs:
            if len(self._svgs) > 8:
                self._svgs.clear()
            self._svgs[text] = phone_view.qr_svg(text)
        return self._svgs[text]

    def view(self):
        """The server of the phone's screen, made at first use (Windows may then ask once about its firewall)."""
        if self._view is None:
            # in simulation only this PC: the tests never wake the firewall
            self._view = phone_view.ViewServer(LiveFrames(self), host="127.0.0.1" if self.roots.simulate else "0.0.0.0",
                                               words=view_words(self.lang), colors=dict(PALETTES[PALETTE].values),
                                               lang=self.lang)
        return self._view

    def _view_state(self):
        view = self.view()
        url = view.url_lan(phone_view.lan_ip(self.link.status()["phone_ip"]))
        live = self._live.status() if self._live is not None and self._live.running else {}
        lat = live.get("latency") or {}
        return {"url_lan": url, "qr": self._svg(url), "url_phone": view.url_phone, "opened": self._view_opened,
                "live": bool(live), "pc_ms": lat.get("total_ms"), "camera_ms": lat.get("stream_ms"),
                "synthetic": bool(live.get("synthetic")), **view.stats()}

    @pyqtSlot(result=str)
    def view_open(self):
        """The live view in the phone's browser, through adb (the phone's 127.0.0.1 is this PC's)."""
        def go():
            view = self.view()

            def done(_, error):
                if error is not None:
                    self._emit({"type": "view", "action": "failed", **self._err(error, "unreachable")})
                else:
                    self._view_opened = True
                    self._emit({"type": "view", "action": "opened"})
            if not self._spawn("view_open", lambda: self.link.open_url(view.url_phone, view.port), done):
                return {"error": "busy"}
            return self._link_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def set_phone_prefs(self, payload):
        def go():
            p = {**self.prefs["phone"], **json.loads(payload)}
            p["switch_modes"] = bool(p["switch_modes"])
            if p["shutter"] not in acquisition.KEYCODES:
                return {"error": "bad_value", "message": "shutter"}
            self.prefs.set("phone", p)
            self._phone = None
            return {"ok": True, **p}
        return self._answer(go)

    # ── trials: the run of a phase ───────────────────────────────────────
    def run(self, phase=None):
        phase = phase or self.prefs["phase"]
        if phase not in self._runs:
            self._runs[phase] = trials.Run(phase, self.roots.raw, self.roots.protocol)
        return self._runs[phase]

    def scanner_folder(self):
        if self.roots.simulate:
            return self.roots.export_default
        f = self.prefs["scanner_folder"]
        return Path(f) if f else None

    def _plan(self, run):
        settings = self.opts["settings"]
        return run.next(reference="manual" if self.opts["reference"] else None, settings=settings)

    def _open_trial(self, run):
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

    def _drop_staged(self):
        for item in (self.cur.get("staged") or {}).values():
            if item.get("temp"):
                with contextlib.suppress(OSError):
                    Path(item["path"]).unlink()
        self._watch = None

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
                              "changes": trials.changes(prev, row), "warnings": [],
                              "prev": self._prev_json(prev)}
            state["steps"] = self._steps_json(run, row, steps, cur)
            state["step"] = cur
            state["staged"] = {k: {kk: vv for kk, vv in v.items() if kk != "temp"}
                               for k, v in self.cur["staged"].items()}
            state["recording"] = self.cur.get("recording", "")
            state["watching"] = self._watch[1] if self._watch else ""
            return state
        plan = self._plan(run)
        if plan is None:
            state["complete"] = True
            return state
        prev = rows[-1] if rows else None
        preview_row = {**{k: plan.get(k, "") for k in trials.LOG_FIELDS}}
        steps = trials.trial_steps(run.config, preview_row, prev)
        state["trial"] = {**preview_row, "open": False, "slot": plan["slot"], "changes": plan["changes"],
                          "warnings": plan["warnings"], "extra": plan["slot"] == "",
                          "prev": self._prev_json(prev)}
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
            if not (values.get("swing_mass_g") or plan.get("swing_mass_g")):
                return {"error": "swing_mass"}
            if "swing_mass_g" in values:
                values["swing_mass_g"] = f"{float(values['swing_mass_g']):g}"
            row = run.start(plan, **values)
            self.cur = {"trial": row["trial"], "step": 0, "staged": {}, "recording": "", "resumed": False}
            self.opts = {"reference": False, "settings": None}
            return self._trial_state()
        return self._answer(go)

    def _current(self, key):
        run = self.run()
        row = self._open_trial(run)
        if row is None:
            raise LookupError("no open trial")
        steps = run.steps(row["trial"])
        i = next((i for i, s in enumerate(steps) if s.key == key), None)
        if i is None:
            raise LookupError(f"no step {key}")
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
    def step_confirm(self, key, payload):
        def go():
            run, row, steps, i = self._current(key)
            trial, step = row["trial"], steps[i]
            values = {k: v for k, v in json.loads(payload or "{}").items() if v not in ("", None)}
            if step.files:
                name = step.files[0]
                staged = self.cur["staged"].get(key)
                if not run.find(trial, name):
                    if not staged:
                        return {"error": "missing_file", "message": name}
                    run.store(trial, name, staged["path"], move=staged["temp"])
                if staged:
                    del self.cur["staged"][key]
            if key == "spheres":
                values = {k: values[k] for k in ("spheres_ok", "cap_shift_mm") if k in values}
                if "cap_shift_mm" in values:
                    values["cap_shift_mm"] = f"{float(values['cap_shift_mm']):.2f}"
            elif key == "close":
                notes = values.pop("notes", "")
                values = {k: int(bool(int(values.get(k, 0)))) for k in ("second_hit", "cap_replaced")}
                if notes:
                    values["notes"] = notes
            else:
                values = {}
            if key == "close":
                notes = values.pop("notes", None)
                run.confirm(trial, key, **values)
                if notes:
                    run.log.fill(trial, notes=notes)
                self._drop_staged()
                self.cur = {}
                return {**self._trial_state(), "closed": trial}
            run.confirm(trial, key, **values)
            self.cur["step"] = min(i + 1, len(steps) - 1)
            self._watch = None
            return self._trial_state()
        return self._answer(go)

    def _stage(self, key, path, temp):
        run, row, steps, i = self._current(key)
        step = steps[i]
        name = step.files[0]
        path = Path(path)
        problems = []
        if temp:
            target = self.roots.staging / row["trial"] / (Path(name).stem + path.suffix.lower())
            target.parent.mkdir(parents=True, exist_ok=True)
            if path != target:
                if target.exists():
                    target.unlink()
                shutil.move(str(path), str(target))
            path = target
            problems = [v for _, v in trials.deep_check(run.config, path)]
        old = self.cur["staged"].get(key)
        if old and old.get("temp") and Path(old["path"]) != path:
            with contextlib.suppress(OSError):
                Path(old["path"]).unlink()
        self.cur["staged"][key] = {"path": str(path), "url": file_url(path), "name": path.name,
                                   "size": path.stat().st_size, "problems": problems, "temp": temp,
                                   "kind": trials.file_kind(name)}

    @pyqtSlot(str, result=str)
    def step_shoot(self, key):
        def go():
            run, row, steps, i = self._current(key)
            blocked = self._free_camera()
            if blocked:
                return blocked
            out = self.roots.staging / row["trial"]

            def done(path, error):
                if error is not None:
                    self._emit({"type": "acquire_failed", "key": key, **self._err(error)})
                    return
                self._stage(key, path, temp=True)
                self._emit({"type": "staged", "key": key})

            started = self._spawn("phone", lambda: self.phone().shoot(out), done)
            return {"ok": started} if started else {"error": "busy"}
        return self._answer(go)

    @pyqtSlot(str, str, result=str)
    def step_video(self, key, action):
        def go():
            run, row, steps, i = self._current(key)
            out = self.roots.staging / row["trial"]
            if action == "start":
                blocked = self._free_camera()
                if blocked:
                    return blocked

                def started(_, error):
                    if error is not None:
                        self.cur["recording"] = ""
                        self._emit({"type": "acquire_failed", "key": key, **self._err(error)})
                    else:
                        self._emit({"type": "recording", "key": key})
                self.cur["recording"] = key
                ok = self._spawn("phone", lambda: self.phone().start_video(), started)
                return {"ok": ok} if ok else {"error": "busy"}

            def done(path, error):
                self.cur["recording"] = ""
                if error is not None:
                    self._emit({"type": "acquire_failed", "key": key, **self._err(error)})
                    return
                self._stage(key, path, temp=True)
                self._emit({"type": "staged", "key": key})
            ok = self._spawn("phone", lambda: self.phone().stop_video(out), done)
            return {"ok": ok} if ok else {"error": "busy"}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def step_import(self, key):
        def go():
            run, row, steps, i = self._current(key)
            kind = trials.file_kind(steps[i].files[0])
            pattern = " ".join("*" + s for s in trials.SUFFIXES[kind])
            path, _ = QFileDialog.getOpenFileName(None, "ImpactGuide", str(self.scanner_folder() or Path.home()),
                                                  f"{kind} ({pattern})")
            if not path:
                return {"ok": False}
            tmp = self.roots.staging / row["trial"] / ("import_" + Path(path).name)
            tmp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, tmp)
            self._stage(key, tmp, temp=True)
            return self._trial_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def step_discard(self, key):
        def go():
            item = (self.cur.get("staged") or {}).pop(key, None)
            if item and item.get("temp"):
                with contextlib.suppress(OSError):
                    Path(item["path"]).unlink()
            return self._trial_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def scan_watch(self, key):
        """Waits for the next scan exported into the scanner folder; it is staged for this step."""
        def go():
            run, row, steps, i = self._current(key)
            folder = self.scanner_folder()
            if folder is None:
                return {"error": "no_scanner_folder"}
            folder.mkdir(parents=True, exist_ok=True)
            if self._watch is None or self._watch[1] != key:
                self._watch = (row["trial"], key, acquisition.ExportWatch(folder))
                if self.roots.simulate:
                    stem = Path(steps[i].files[0]).stem
                    QTimer.singleShot(1200, lambda: acquisition.fake_scan(folder, f"export_{stem}.ply"))
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
                    "pending": len(report["pending"]), "extra": report["extra"], "orphans": report["orphans"]}
        return self._answer(go)

    # ── calibration ──────────────────────────────────────────────────────
    # Three steps (ui/10-calibration.js): the board (on this monitor, in the page; or a physical one: the 3D-printed
    # plate or a printed sheet, with its measure), the photos (a plan of poses, quick or accurate, each photo checked
    # as it comes and kept or put aside), the calibration (its errors per photo and over the board, the intrinsics in
    # plain units, a verdict, photos left out by hand, saved). Two camera modes: the phone's photos (the camera app,
    # 12.5 MP, what the measurements of the bench use) and the live stream (its frames, for the live AR).
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
        return p if p in ("quick", "accurate") else "quick"

    def _folders(self):
        """The folders of calibration photos, newest first: name, path, number of photos, kind, board.json."""
        folders = []
        if self.roots.calib_raw.is_dir():
            for d in sorted((d for d in self.roots.calib_raw.iterdir() if d.is_dir()), key=lambda d: d.name,
                            reverse=True):
                n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
                board = None
                with contextlib.suppress(OSError, ValueError):
                    board = json.loads((d / calibrate_charuco.BOARD_FILE).read_text(encoding="utf-8"))
                folders.append({"name": d.name, "path": str(d), "n": n, "kind": d.name.split("_", 1)[-1],
                                "board": board})
        return folders

    @staticmethod
    def _photo_checks(d):
        """{name: check} of the photos of a series (calibrate_charuco.check_photo), kept beside them."""
        try:
            return json.loads((Path(d) / PHOTO_CHECKS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _series_detail(self, f, active):
        """The series of photos in progress: its photos in the order they came, each with its check; the corners of
        the photos kept (where the board has been on the screen); whether it is of the board in use."""
        d = Path(f["path"])
        checks = self._photo_checks(d)
        photos = sorted((p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.stat().st_mtime)
        same = f["n"] == 0 or self._same_board(f["board"], active)
        rows = [{"name": p.name, **{k: (checks.get(p.name) or {}).get(k) for k in ("ok", "why", "corners", "of",
                                                                                  "blur_px")}} for p in photos]
        points = []
        for p in photos:
            pts = (checks.get(p.name) or {}).get("points") or []
            points.append(pts[::max(1, len(pts) // 30)])
        last = None
        if checks:
            name = max(checks, key=lambda k: (checks[k].get("seq", 0), checks[k].get("time", "")))
            last = {"name": name, **{k: checks[name].get(k) for k in ("ok", "why", "corners", "of", "blur_px",
                                                                       "points", "time")}}
        return {**f, "same": same, "photos": rows, "points": points, "last": last}

    def _guide_board(self, series, active):
        """The board the guide draws: the series' own (board.json) when it has photos, else the one in use."""
        b = (series or {}).get("board") if series and series.get("n") else None
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
        nxt = None
        if active and n < lim["max"]:
            g = self._guide_board(series if series and series["same"] else None, active)
            with contextlib.suppress(KeyError, TypeError, ValueError):
                nxt = calib_guide.guide(calib_guide.PLANS[plan][n], g, mode, n, plan)
        result = self._result(mode)
        live = None
        if self._live is not None:
            st = self._live.status()
            live = {"running": st["running"], "starting": st.get("starting"), "tracked": st.get("tracked"),
                    "url": self._live_server.url if self._live_server else "", "purpose": self._live_for,
                    "display": st.get("display") or st.get("image"), "error": bool(st.get("error"))}
        return {"mode": mode, "plan": plan, "kind": kind, "limits": lim, "count": n, "series": series, "next": nxt,
                "plans": {k: calib_guide.limits(k) for k in ("quick", "accurate")},
                "modes": modes, "last_check": last, "result": result, "board_ui": bs, "phone": self.phone_status,
                "busy": sorted(self._busy), "min_views": calibrate_charuco.MIN_VIEWS,
                "max_rms": calibrate_charuco.MAX_RMS, "live": live, "simulate": bool(r.simulate),
                "raw": str(r.calib_raw), "out": str(r.calib_out)}

    @pyqtSlot(result=str)
    def calib_state(self):
        return self._answer(self._calib_state)

    def _result(self, mode):
        """The calibration waiting to be saved (or the one stored) of a mode, as the result step draws it: the
        verdict, the error of each photo (used, dropped by the 3-times-the-median rule, left out by hand), where on
        the board the error is, the intrinsics in plain units."""
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
        few = cal["n_views"] < calib_guide.limits(self._calib_plan() if self._calib_plan() != "check" else "quick")["min"]
        verdict = "good" if rms < calibrate_charuco.MAX_RMS and not few else \
            "fair" if rms < 2 * calibrate_charuco.MAX_RMS else "poor"
        b = cal.get("board") or {}
        res = cal.get("residuals")
        grid = [int(b["squares"][0]) - 1, int(b["squares"][1]) - 1] if b.get("squares") else None
        return {"saved": pending is None, "date": cal.get("date", ""), "rms_px": rms, "n_views": cal["n_views"],
                "of": len(every), "few": few, "verdict": verdict, "photos": photos,
                "limit_px": cal.get("limit_px"), "max_rms": calibrate_charuco.MAX_RMS,
                "residuals": res, "grid": grid, "plain": calibrate_charuco.plain(cal), "images": cal.get("images", "")}

    # ── calibration: the series of photos ───────────────────────────────
    def _series_kind(self):
        """What the photos are for: a calibration, or the session check of the one stored (Advanced)."""
        k = self.prefs["calib_kind"]
        return k if k in ("calibration", "check") else "calibration"

    def _series_tag(self):
        return self._mode() if self._series_kind() == "calibration" else "check_" + self._mode()

    def _series(self, folders=None):
        """The folder the photos go to now: the newest of today's for this mode and kind, or None (one is made at the
        first photo). A folder of another day is a series of its own, kept as it is."""
        if folders is None:
            folders = self._folders()
        today = f"{date.today():%Y%m%d}"
        return next((f for f in folders if f["kind"] == self._series_tag() and f["name"].startswith(today)), None)

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
        if f is not None and (f["n"] == 0 or self._same_board(f["board"], self._board()["active"])):
            return Path(f["path"])
        return self._new_series()

    def _series_hold(self):
        """The board of the series in progress, kept on the monitor while it fits (screen_board.layout, hold)."""
        f = self._series()
        b = f["board"] if f is not None and f["n"] > 0 else None
        return b if b and b.get("kind") == "screen" and b.get("square_px") else None

    @pyqtSlot(str, result=str)
    def calib_set(self, payload):
        """{mode: photo or stream, plan: quick or accurate, kind: calibration or check}."""
        def go():
            c = json.loads(payload)
            if c.get("mode") in CALIB_MODES:
                self.prefs.set("calib_mode", c["mode"])
            if c.get("plan") in ("quick", "accurate"):
                self.prefs.set("calib_plan", c["plan"])
            if c.get("kind") in ("calibration", "check"):
                self.prefs.set("calib_kind", c["kind"])
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
        live stream. The board must be on show when it is the one on this monitor, and the plan not full. The photo
        is checked as it comes (calib_photo events)."""
        def go():
            st = self._board()
            if st["kind"] == "screen" and (not self._board_on or self._lay is None):
                return {"error": "board_small" if self._board_on else "board_hidden"}
            s = self._calib_state()
            if s["count"] >= s["limits"]["max"]:
                return {"error": "calib_full"}
            if "photo_check" in self._busy:
                return {"error": "busy"}
            if self._mode() == "stream" and not (self._live is not None and self._live.running
                                                 and self._live_for == "calibration"):
                return {"error": "stream_off"}
            d = self._series_folder()
            if self._mode() == "stream":
                return self._grab_into(d)
            return self._shoot_into(d)
        return self._answer(go)

    def _sim_scene(self):
        """In simulation the phone photographs the board in use at the next pose of the plan (calib_guide.photo)."""
        s = self._calib_state()
        st = self._board()
        kind = st["kind"]
        spec = self._guide_board(s["series"] if s["series"] and s["series"]["same"] else None, st["active"])
        lay = self._lay if kind == "screen" else None
        tex_kind = {"screen": "screen", "plate": "plate", "sheet": "sheet"}[kind]
        n = s["count"]
        size = self.phone().photo_px

        def scene():
            import cv2
            img = calib_guide.photo(s["plan"], n, spec, tex_kind, "photo", lay, size=tuple(size))
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return scene

    def _shoot_into(self, d, check=True):
        blocked = self._free_camera()
        if blocked:
            return blocked
        self._folder_board(d)
        phone = self.phone()
        if self.roots.simulate and check:
            phone.scene = self._sim_scene()

        def done(path, error):
            if self.roots.simulate:
                phone.scene = None
            if error is not None:
                self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error)})
                return
            dest = d / Path(path).name
            if dest.exists():
                self._emit({"type": "acquire_failed", "key": "calibration", "error": "exists", "message": dest.name})
                return
            shutil.move(str(path), str(dest))
            self._arrived(d, dest, check)
        ok = self._spawn("phone", lambda: phone.shoot(self.roots.staging / "calibration"), done)
        return {"ok": ok, "folder": d.name} if ok else {"error": "busy"}

    def _grab_into(self, d):
        """The newest frame of the live stream, at its full size, into the series as a PNG (lossless)."""
        ar = self._live
        if ar is None or not ar.running or self._live_for != "calibration":
            return {"error": "stream_off"}
        img, n = ar.still()
        if img is None:
            return {"error": "stream_off"}
        self._folder_board(d)
        name = f"frame_{datetime.now():%Y%m%d_%H%M%S}_{n:06d}.png"

        def work():
            import cv2
            cv2.imwrite(str(d / name), img)
            return d / name

        def done(path, error):
            if error is not None:
                self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error)})
                return
            self._arrived(d, path, True)
        ok = self._spawn("phone", work, done)
        return {"ok": ok, "folder": d.name} if ok else {"error": "busy"}

    def _arrived(self, d, dest, check=True):
        """A photo is in the series: checked at once (board found, corners, blur, the same size as the others). One
        that fails goes aside, to the folder's rejected/, and the page says why; its pose comes again."""
        if not check:
            self._emit({"type": "calib_photo", "folder": d.name, "name": dest.name})
            return
        args = self._board_args(d)
        checks = self._photo_checks(d)
        kept = [p for p in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime)
                if p.suffix.lower() in IMAGE_SUFFIXES and p != dest and (checks.get(p.name) or {}).get("ok")]
        size = (checks.get(kept[0].name) or {}).get("size") if kept else None

        def work():
            board = calibrate_charuco.make_board(args[0], args[1], args[2])
            return calibrate_charuco.check_photo(dest, board, size)

        def done(res, error):
            if error is not None:
                res = {"ok": False, "why": "unreadable", "corners": 0, "of": 0, "blur_px": None, "points": []}
            res["time"] = datetime.now().isoformat(timespec="seconds")
            res["mode"] = self._mode()
            if not res["ok"]:
                aside = d / REJECTED
                aside.mkdir(exist_ok=True)
                with contextlib.suppress(OSError):
                    shutil.move(str(dest), str(aside / dest.name))
            allc = self._photo_checks(d)
            res["seq"] = 1 + max((c.get("seq", 0) for c in allc.values()), default=0)    # the order they came in
            allc[dest.name] = res
            (d / PHOTO_CHECKS).write_text(json.dumps(allc, indent=1), encoding="utf-8")
            self._emit({"type": "calib_photo", "folder": d.name, "name": dest.name,
                        "check": {k: res.get(k) for k in ("ok", "why", "corners", "of", "blur_px")}})
        if not self._spawn("photo_check", work, done):
            self._emit({"type": "calib_photo", "folder": d.name, "name": dest.name})

    @pyqtSlot(result=str)
    def calib_retake(self):
        """The last photo kept goes aside after all (the operator knows better): its pose comes again."""
        def go():
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
            (d / PHOTO_CHECKS).write_text(json.dumps(checks, indent=1), encoding="utf-8")
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
            if "phone" in self._busy or self.cur.get("recording"):
                return {"error": "busy"}

            def done(_, error):
                if error is not None:
                    self._emit({"type": "live", "action": "failed", **self._err(error, "live_failed")})
                else:
                    self._emit({"type": "live", "action": "started", "purpose": "calibration"})
            if not self._spawn("live", ar.start, done):
                return {"error": "live_starting"}
            return self._calib_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_board_pdf(self):
        """The A4 sheet to print (a board one may have instead of the plate), opened."""
        def go():
            with contextlib.redirect_stdout(io.StringIO()):
                pdf = calibrate_charuco.write_board(self.roots.board, calibrate_charuco.BOARD["square"])
            self._open(pdf)
            return {"ok": True}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def plate_open(self, what):
        """The 3D-printed plate: its two-colour print file (3MF, in the slicer) or the folder with its STL."""
        def go():
            plate = charuco_plate.load()
            folder = charuco_plate.PLATE_FILE.parent
            path = folder / plate["files"]["3mf"] if what == "3mf" else folder
            if not path.exists():
                return {"error": "not_found", "message": path.name}
            self._open(path)
            return {"ok": True}
        return self._answer(go)

    @staticmethod
    def _open(path):
        """A file or a folder opened by Windows (the file itself: a folder window opens behind this frameless one)."""
        if hasattr(os, "startfile"):
            os.startfile(str(path))
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _folder_board(self, d):
        """board.json in a photo folder: the active board, written while the folder has no photos (so it is the
        board the first photo is of); a folder with photos keeps the one it has."""
        p = d / calibrate_charuco.BOARD_FILE
        if p.exists() and any(x.suffix.lower() in IMAGE_SUFFIXES for x in d.iterdir()):
            return
        a = self._board()["active"]
        if not a:
            return
        p.write_text(json.dumps({**a, "written": datetime.now().isoformat(timespec="seconds")}, indent=1),
                     encoding="utf-8")

    def _board_args(self, d):
        """(square, squares, marker) for calibrate_charuco: the folder's board.json, else the active board."""
        got = calibrate_charuco.folder_board(d)
        if got:
            return got
        a = self._board()["active"]
        return a["square_mm"], tuple(a["squares"]), a["marker_mm"]

    def _calib_dir(self, name):
        d = (self.roots.calib_raw / name).resolve()
        if d.parent != self.roots.calib_raw.resolve() or not d.is_dir():
            raise LookupError(f"no folder {name}")
        return d

    @pyqtSlot(str, result=str)
    def calib_shoot(self, name):
        """A photo from the phone into a folder chosen by name, under the name the phone gave it (not checked)."""
        return self._answer(lambda: self._shoot_into(self._calib_dir(name), check=False))

    @pyqtSlot(str, result=str)
    def calib_folder_new(self, kind):
        """Today's folder for calibration photos (photo, stream) or for the session check (check_photo...)."""
        def go():
            if kind not in CALIB_MODES + tuple("check_" + m for m in CALIB_MODES):
                return {"error": "bad_value"}
            d = self.roots.calib_raw / f"{date.today():%Y%m%d}_{kind}"
            d.mkdir(parents=True, exist_ok=True)
            self._folder_board(d)
            return {**self._calib_state(), "folder": d.name}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def calib_import(self, name):
        """Photo files added to a folder by name, or to the series (name empty), each checked as a photo taken."""
        def go():
            d = self._calib_dir(name) if name else self._series_folder()
            pattern = " ".join("*" + s for s in IMAGE_SUFFIXES)
            paths, _ = QFileDialog.getOpenFileNames(None, "ImpactGuide", str(Path.home()), f"photos ({pattern})")
            if paths:
                self._folder_board(d)
            added = skipped = 0
            for p in paths:
                dest = d / Path(p).name
                if dest.exists():
                    skipped += 1
                    continue
                shutil.copy2(p, dest)
                added += 1
            if added and not name:
                self._check_all(d)
            return {**self._calib_state(), "added": added, "skipped": skipped}
        return self._answer(go)

    def _check_all(self, d):
        """The photos of a folder that have no check yet, checked one after the other (files added by hand)."""
        args = self._board_args(d)
        todo = [p for p in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime)
                if p.suffix.lower() in IMAGE_SUFFIXES and p.name not in self._photo_checks(d)]

        def work():
            board = calibrate_charuco.make_board(args[0], args[1], args[2])
            out, size = {}, None
            for p in todo:
                r = calibrate_charuco.check_photo(p, board, size)
                if r["ok"] and size is None:
                    size = r["size"]
                out[p.name] = r
            return out

        def done(res, error):
            if error is not None or not res:
                return
            allc = self._photo_checks(d)
            for name, r in res.items():
                r["time"] = datetime.now().isoformat(timespec="seconds")
                r["seq"] = 1 + max((c.get("seq", 0) for c in allc.values()), default=0)
                allc[name] = r
                if not r["ok"]:
                    (d / REJECTED).mkdir(exist_ok=True)
                    with contextlib.suppress(OSError):
                        shutil.move(str(d / name), str(d / REJECTED / name))
            (d / PHOTO_CHECKS).write_text(json.dumps(allc, indent=1), encoding="utf-8")
            self._emit({"type": "calib_photo", "folder": d.name, "name": ""})
        self._spawn("photo_check", work, done)

    @pyqtSlot(result=str)
    def calib_open(self):
        def go():
            f = self._series()
            if f is None:
                return {"error": "nothing"}
            QDesktopServices.openUrl(QUrl.fromLocalFile(f["path"]))
            return {"ok": True}
        return self._answer(go)

    @staticmethod
    def _captured(fn):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = fn()
        return result, buf.getvalue()

    @pyqtSlot(str, result=str)
    def calib_run(self, exclude):
        """Calibrates the camera mode on the photos of the series, leaving out those named in exclude (a JSON list);
        the result waits for calib_save."""
        def go():
            f = self._series()
            if f is None or not f["n"]:
                return {"error": "nothing"}
            d, mode = Path(f["path"]), self._mode()
            leave = [str(x) for x in json.loads(exclude or "[]")]

            def work():
                board = self._board_args(d)
                return self._captured(lambda: calibrate_charuco.run_calibrate(d, None, None, *board, exclude=leave))

            def done(result, error):
                if error is not None:
                    self._emit({"type": "calib", "action": "calibrate", "mode": mode, "ok": False,
                                **self._err(error, "calibration")})
                    return
                cal, log = result
                self._pending_cal[mode] = cal
                self._emit({"type": "calib", "action": "calibrate", "mode": mode, "log": log,
                            "ok": cal["rms_px"] < calibrate_charuco.MAX_RMS, "result": self._cal_summary(cal)})
            ok = self._spawn("calibration", work, done)
            return {"ok": ok} if ok else {"error": "busy"}
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_save(self):
        def go():
            mode = self._mode()
            cal = self._pending_cal.pop(mode, None)
            if cal is None:
                return {"error": "nothing"}
            p = self._intrinsics_path(mode)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(cal, indent=1), encoding="utf-8")
            return self._calib_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_check(self):
        """Start-of-session check of the stored intrinsics on the photos of the series."""
        def go():
            f = self._series()
            mode = self._mode()
            if f is None or not f["n"]:
                return {"error": "nothing"}
            d = Path(f["path"])
            intr = self._intrinsics_path(mode)
            if not intr.exists():
                return {"error": "no_intrinsics", "message": mode}

            def work():
                return self._captured(lambda: calibrate_charuco.run_check(d, intr, *self._board_args(d)))

            def done(result, error):
                if error is not None:
                    self._emit({"type": "calib", "action": "check", "mode": mode, "ok": False,
                                **self._err(error, "check")})
                    return
                (ok, report), log = result
                p = report["params"]
                line = {"time": datetime.now().isoformat(timespec="seconds"), "mode": mode, "folder": d.name,
                        "ok": int(ok), "n_views": report["n_views"], "rms_stored_px": report["rms_stored_px"],
                        **{f"{k}_change": p[k]["change"] for k in ("fx", "fy", "cx", "cy")}}
                trials.write_csv(self._checks_path(), trials.read_csv(self._checks_path()) + [line], CHECK_FIELDS)
                self._emit({"type": "calib", "action": "check", "mode": mode, "ok": ok, "log": log, "report": report})
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

    def _watch_screen(self):
        """The page is told when the window goes to another monitor or its monitor changes (resolution, scaling):
        the board is laid out again for it."""
        if self.window is None or self._screen_watch:
            return
        self._screen_watch = True

        def moved(*_):
            sc = self._app_screen()
            if sc is not None and sc is not self._watched_screen:
                for sig in (getattr(self._watched_screen, "geometryChanged", None),
                            getattr(self._watched_screen, "physicalDotsPerInchChanged", None)):
                    with contextlib.suppress(TypeError, RuntimeError, AttributeError):
                        sig.disconnect(moved)
                self._watched_screen = sc
                sc.geometryChanged.connect(moved)
                sc.physicalDotsPerInchChanged.connect(moved)
            self._emit({"type": "board", "action": "screen"})
        self.window.screenChanged.connect(moved)
        sc = self._app_screen()
        if sc is not None:
            self._watched_screen = sc
            sc.geometryChanged.connect(moved)
            sc.physicalDotsPerInchChanged.connect(moved)

    def _kind(self):
        k = self.prefs["board"].get("kind")
        return k if k in BOARD_KINDS else "screen"

    @staticmethod
    def _sheet_reading(values):
        """The A4 sheet's long side as measured: mean, spread and problems (as charuco_plate.reading)."""
        v = [float(x) for x in values]
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

    def _physical(self):
        """The plate and the sheet: their nominal sizes, what to measure, the readings typed, and the board each
        gives (as measured when the readings can be used, else nominal)."""
        b = self.prefs["board"]
        out = {}
        try:
            plate = charuco_plate.load()
        except (OSError, ValueError, KeyError):
            plate = None
        if plate is not None:
            pr = b.get("plate") or {}
            long_r = charuco_plate.reading(pr["long"], "long", plate) if pr.get("long") else None
            short_r = charuco_plate.reading(pr["short"], "short", plate) if pr.get("short") else None
            use_long = long_r["mean_mm"] if long_r and not long_r["problems"] else None
            use_short = short_r["mean_mm"] if short_r and not short_r["problems"] and use_long else None
            spec = charuco_plate.spec(use_long, use_short, plate)
            spec["measured"] = use_long is not None
            out["plate"] = {"board": plate["board"], "plate_mm": plate["plate"]["size_mm"],
                            "measure": plate["measure"], "files": plate["files"], "readings": pr,
                            "long": long_r, "short": short_r, "spec": spec, "model": PLATE_MODEL.is_file(),
                            "min_readings": charuco_plate.MIN_READINGS, "spread_max": charuco_plate.SPREAD_MAX}
        sr = b.get("sheet") or {}
        long_s = self._sheet_reading(sr["long"]) if sr.get("long") else None
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

    def _num(self, v, n=1):
        text = f"{v:.{n}f}"
        return text.replace(".", ",") if self.lang in COMMA_LANGS else text

    @pyqtSlot(str, result=str)
    def board_set(self, payload):
        """{kind (screen, plate, sheet; the first question answered), plate_long, plate_short, sheet_long (the
        readings typed, mm, a list each), ruler_mm (the length between the lines above the board on the monitor, as a
        ruler reads it), forget (the ruler check of this monitor)}."""
        def go():
            c = json.loads(payload)
            b = dict(self.prefs["board"])
            if c.get("kind") in BOARD_KINDS:
                b["kind"], b["chosen"] = c["kind"], True
            for key, part, which in (("plate_long", "plate", "long"), ("plate_short", "plate", "short"),
                                     ("sheet_long", "sheet", "long")):
                if key in c:
                    vals = [float(x) for x in (c[key] or []) if x not in (None, "") and float(x) > 0]
                    if any(not 20 <= v <= 1000 for v in vals):
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
                        pitch = screen_board.ruler_pitch(lay, float(c["ruler_mm"]))
                    except (ValueError, TypeError):
                        return {"error": "ruler_bad", "message": self._num(lay["width_mm"])}
                    width_px = lay["squares"][0] * lay["square_px"]
                    checks[mon["key"]] = {"pitch_mm": pitch, "measured_mm": float(c["ruler_mm"]),
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

    def _relay(self, area=None):
        """The board for the page's area (device pixels) at the pitch of the monitor the window is on: the layout,
        whether the page's pixels are the screen's (exact) and an error (no_monitor, too_small, across)."""
        c = area or self._area
        mon = self._app_monitor()
        lay, err, exact = None, "", True
        if mon is None:
            err = "no_monitor"
        else:
            w, h = int(c["w"]), int(c["h"])
            scale, inside = self._page_scale(c)
            exact = abs(scale - 1.0) < 1e-9
            pitch = (mon["pitch_mm"] or GUESS_PITCH) * scale
            try:
                lay = screen_board.layout((w, h), pitch, self._series_hold())
            except ValueError:
                err = "too_small"
            if lay is not None and not inside:
                err = "across"
        self._lay, self._lay_err, self._lay_mon, self._lay_exact = lay, err, mon, exact
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
        Answers the layout and, when the page's board is not this one, the image to draw (PNG, one pixel per device
        pixel), with its key."""
        def go():
            import cv2
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
            key = (f"{lay['size_px'][0]}x{lay['size_px'][1]}:{lay['origin_px'][0]},{lay['origin_px'][1]}:"
                   f"{lay['squares'][0]}x{lay['squares'][1]}:{lay['square_px']}:{lay['marker_px']}")
            out["key"] = key
            if key != c.get("have"):
                img = screen_board.render(lay)
                self._board_img = img
                ok, png = cv2.imencode(".png", img)
                out["png"] = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")
            return out
        return self._answer(go)

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
    def live(self):
        """The live AR, made at first use and kept for the session: its overlay thread owns the OpenGL
        context (live_ar.py). In simulation a synthetic board stands in for the phone."""
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
                                        replay=self.replay, hide_board=lb["hide_board"] and p["board"] == "sheet")
            self._live_server = live_ar.MjpegServer(self._live)
        return self._live

    def _free_camera(self):
        """The phone's camera serves one at a time: a photo or a video stops the live view first (the
        acquisition then brings the camera app back). An error while the live view is still starting."""
        if "live" in self._busy:
            return {"error": "live_starting"}
        if self._live is not None and self._live.running:
            self._live.stop()
            self._emit({"type": "live", "action": "stopped", "reason": "photo"})
        return None

    def _live_state(self):
        self.live()
        self._sync_live()       # the board changed here, on the Calibration module, or its area moved
        ar = self._live
        ln = self.link.status()
        return {**ar.status(), "url": self._live_server.url, "prefs": self.prefs["live"], "phone": self.phone_status,
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
            if "phone" in self._busy or self.cur.get("recording"):
                return {"error": "busy"}

            def done(_, error):
                if error is not None:
                    self._emit({"type": "live", "action": "failed", **self._err(error, "live_failed")})
                else:
                    self._emit({"type": "live", "action": "started"})
            if not self._spawn("live", ar.start, done):
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

    # ── the window is closing ────────────────────────────────────────────
    def shutdown(self):
        self._pump.stop()
        self._drop_staged()
        self.link.close()
        if self._view is not None:
            self._view.close()
        if self._live_server is not None:
            self._live_server.close()
        if self._live is not None:
            self._live.shutdown()


class LiveFrames:
    """The phone's screen reads the live AR's frames through this, so the server can be up before the live AR is
    made (and while it is off)."""

    def __init__(self, backend):
        self.backend = backend

    def next_frame(self, after=0, timeout=1.0):
        ar = self.backend._live
        if ar is None:
            threading.Event().wait(timeout)
            return None, after, None
        return ar.next_frame(after, timeout)


def view_words(lang):
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


def prepare_page(dest):
    """The page as the window loads it: ui.html and ui/ copied to a local folder, with SlantUI's
    stylesheets and scripts and the mark in the accent written beside them. Nothing is generated
    next to the sources, which may sit on a synced drive."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    ignore = shutil.ignore_patterns("desktop.ini", "__pycache__")
    shutil.copytree(APP / "ui", dest / "ui", ignore=ignore)
    shutil.copy2(APP / "ui.html", dest / "ui.html")
    (dest / "ui" / "slantui.css").write_text(slantui_css.bundle(), encoding="utf-8")
    (dest / "ui" / "slantui.js").write_text(slantui_js.bundle(), encoding="utf-8")
    return dest / "ui.html"


ERROR_LOG = LOCAL / "errors.log"


def log_error(e, cause=None):
    """The whole error, with its trace, in a log beside the preferences: the screen only says what to do."""
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        if ERROR_LOG.exists() and ERROR_LOG.stat().st_size > 2_000_000:
            ERROR_LOG.replace(ERROR_LOG.with_suffix(".old.log"))
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__)) if isinstance(e, BaseException) else str(e)
        with ERROR_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {cause or ''}\n{tb}\n")
    except OSError:
        pass


def mark_file():
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


def tell_launcher_up():
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
        handle = k32.OpenEventW(0x0002, False, r"Local\ImpactGuideUp")
        if handle:
            k32.SetEvent(handle)
            k32.CloseHandle(handle)
    except Exception:
        pass


def splash_words(lang):
    try:
        words = json.loads((APP / "ui" / "strings" / f"{lang}.json").read_text(encoding="utf-8"))
        return words.get("win.splash.starting", "starting"), words.get("win.splash.ready", "ready")
    except (OSError, ValueError):
        return "starting", "ready"


def build(args):
    """Application, backend and window, shown; returns (app, window, backend)."""
    roots = Roots(args.simulate)
    prefs = Prefs(roots.prefs)
    app = Application(APP_NAME, app_id=APP_ID)
    app.setWindowIcon(app_icon())
    backend = Backend(roots, prefs, lang=args.lang, replay=args.replay)
    if args.page in PAGES:
        backend.start_page = args.page
    page = prepare_page((roots.simulate or LOCAL) / "page")
    title = APP_NAME + (" (simulation)" if roots.simulate else "")
    # the band the window draws before the page has one shows the mark as a button, as the page
    # always does (ui/01-shell.js, showHome)
    win = Window(page, bridge=backend, title=title, palette=PALETTE, size=(1440, 900), min_size=MIN_WINDOW,
                 logo=mark_file(), brand_action=True)
    backend.window = win
    win.setIcon(app_icon())
    app.aboutToQuit.connect(backend.shutdown)
    starting, ready = splash_words(backend.lang)
    splash = Splash(window=win, logo=mark_file(), text=starting, busy=True)
    splash.show()

    def page_is_up():
        splash.set_progress(1.0, ready)
        QTimer.singleShot(200, splash.hide)
        tell_launcher_up()

    backend.on_page_ready = page_is_up
    QTimer.singleShot(25000, lambda: splash.hide() if splash.visible else None)
    win.show()
    return app, win, backend


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--simulate", type=Path, help="keep every file under this folder, with a stand-in phone and scanner")
    p.add_argument("--lang", choices=LANGS)
    p.add_argument("--page", choices=PAGES)
    p.add_argument("--replay", type=Path, help="the live page plays these recorded frames (a video or a folder of "
                                               "images) instead of the phone's stream")
    return p.parse_args(argv)


def main(argv=None):
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
            pass
        sys.stderr.write(tb)
        raise
