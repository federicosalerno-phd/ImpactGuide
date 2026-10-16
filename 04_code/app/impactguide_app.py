"""ImpactGuide: the desktop app of the marker impact bench.

A frameless SlantUI window hosting a page (ui.html and ui/) that talks to this
backend over QWebChannel. As UniGuide, it opens at home, four large choices, and
each choice is a module with its own steps on the rail: Calibrate the phone
(board, photos, calibration), Live AR (board, view), Trials (next trial, log,
sequence, setup) and Scans and results (the next pages). The app guides the
protocol step by step and keeps what the bench produces.

The calibration board is shown on a monitor (board_window.py, screen_board.py):
full screen on a monitor of its own when there are two, else in a window beside
the app on the same monitor, at any split, sized from the pixels of that window
and the pitch of its monitor; or it is printed.

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
from slantui.shell.qt import (QApplication, QColor, QDesktopServices, QFileDialog, QRect, QTimer, QUrl,  # noqa: E402
                              USE_QT6)
from slantui.tokens import PALETTES  # noqa: E402

import acquisition  # noqa: E402
import calibrate_charuco  # noqa: E402
import live_ar  # noqa: E402
import phone_link  # noqa: E402
import phone_view  # noqa: E402
import screen_board  # noqa: E402
import trials  # noqa: E402

import board_window  # noqa: E402  (beside this file)

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
CALIB_MODES = ("photo", "video")
IMAGE_SUFFIXES = tuple(sorted(calibrate_charuco.IMAGE_SUFFIXES))
CHECK_FIELDS = ("time", "mode", "folder", "ok", "n_views", "rms_stored_px", "fx_change", "fy_change",
                "cx_change", "cy_change")
BOARD_KINDS = ("screen", "printed")
BOARD_PLACES = ("auto", "beside", "full")   # auto: full screen on another monitor if there is one, else beside the app
MIN_WINDOW = (720, 560)     # the app's least size: room for the board beside it on one monitor
# Photos of a calibration series: enough for a calibration (protocol T5.1: 40, at least 30), and for a session check
ENOUGH = {"calibration": 30, "check": 10}
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

    DEFAULTS = {"lang": "", "page": "trials", "phase": "campaign", "scanner_folder": "", "square_mm": 24.0,
                "calib_mode": "photo", "calib_kind": "calibration",
                "phone": {"camera_dir": acquisition.CAMERA_DIR, "shutter": "volume_down", "switch_modes": True,
                          "link": phone_link.AUTO, "wifi_address": "", "phone_id": "", "wifi_paired": False},
                "live": {"board": "sheet", "face": True, "mandible": False, "axes": True, "registration": "",
                         "opacity": live_ar.OPACITY},
                # the calibration board: on a monitor or printed. On a monitor it goes full screen on another monitor
                # (the one chosen, else the smallest without the app) or beside the app at a split (the board's share)
                "board": {"kind": "screen", "monitor": "", "place": "auto", "split": 0.5},
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
        self._board_win = None      # the board on a monitor (board_window.BoardWindow), while it is shown
        self._dock = None           # the board docked beside the app (board_window.Dock), while it is
        self._win_mons = (None, {})  # what Windows says of the monitors, read again when Qt's screens change
        self.window = None          # the app's window (build), for the board beside it
        self._via = None            # the link the phone was last found on (cable or Wi-Fi)
        self.link = self._make_link()
        app = QApplication.instance()
        if app is not None:
            app.screenRemoved.connect(self._screen_removed)
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
            if self._board_win is not None:
                self._board_win.words = self._board_words()
                self._board_win.update()

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
        folders = self._folders()
        b = calibrate_charuco.BOARD
        nx, ny = b["squares"]
        board_file = r.board / f"charuco_{nx}x{ny}_{float(self.prefs['square_mm']):g}mm_{b['dictionary'].lower()}_a4.pdf"
        bs = self._board_state()
        kind = self._series_kind()
        f = self._series(folders)
        series = None
        if f is not None:
            photos = sorted((p for p in Path(f["path"]).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
                            key=lambda p: p.stat().st_mtime)
            same = f["n"] == 0 or self._same_board(f["board"], bs["active"])
            series = {**f, "same": same, "last": file_url(photos[-1]) if photos and same else ""}
        return {"board": {**b, "squares": list(b["squares"])}, "square_mm": self.prefs["square_mm"],
                "active": bs["active"], "board_ui": bs, "mode": self.prefs["calib_mode"], "kind": kind,
                "enough": ENOUGH[kind], "series": series, "modes": modes, "last_check": last, "folders": folders,
                "raw": str(r.calib_raw), "out": str(r.calib_out), "board_file": str(board_file) if board_file.exists() else "",
                "board_dir": str(r.board), "phone": self.phone_status, "busy": sorted(self._busy),
                "min_views": calibrate_charuco.MIN_VIEWS, "max_rms": calibrate_charuco.MAX_RMS}

    @pyqtSlot(result=str)
    def calib_state(self):
        return self._answer(self._calib_state)

    # ── calibration: the series of photos ───────────────────────────────
    def _series_kind(self):
        """What the photos are for: a calibration, or the session check of the one stored (Advanced)."""
        k = self.prefs["calib_kind"]
        return k if k in ENOUGH else "calibration"

    def _series_tag(self):
        mode = self.prefs["calib_mode"] if self.prefs["calib_mode"] in CALIB_MODES else "photo"
        return mode if self._series_kind() == "calibration" else "check_" + mode

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
        """The board of the series in progress, kept on the monitor while it fits (board_window, hold)."""
        f = self._series()
        b = f["board"] if f is not None and f["n"] > 0 else None
        return b if b and b.get("kind") == "screen" and b.get("square_px") else None

    @pyqtSlot(str, result=str)
    def calib_set(self, payload):
        """{mode: photo or video, kind: calibration or check}: what the next photos are for."""
        def go():
            c = json.loads(payload)
            if c.get("mode") in CALIB_MODES:
                self.prefs.set("calib_mode", c["mode"])
            if c.get("kind") in ENOUGH:
                self.prefs.set("calib_kind", c["kind"])
            if self._board_win is not None:
                self._board_win.hold = self._series_hold()
            return self._calib_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_series_new(self):
        """Start again: today's photos go to a new folder from the next one on."""
        def go():
            d = self._new_series()
            self._folder_board(d)
            if self._board_win is not None:
                self._board_win.hold = None
            return {**self._calib_state(), "folder": d.name}
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_take(self):
        """A photo from the phone into the series. The board must be on show when it is the one on a monitor."""
        def go():
            st = self._board()
            if st["kind"] == "screen" and (self._board_win is None or self._board_win.lay is None):
                return {"error": "board_hidden" if self._board_win is None else "board_small"}
            d = self._series_folder()
            return self._shoot_into(d)
        return self._answer(go)

    @pyqtSlot(str, float, result=str)
    def calib_settings(self, mode, square):
        def go():
            if mode in CALIB_MODES:
                self.prefs.set("calib_mode", mode)
            if 5 <= square <= 100:
                self.prefs.set("square_mm", round(float(square), 3))
            return self._calib_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def calib_board(self):
        def go():
            with contextlib.redirect_stdout(io.StringIO()):
                pdf = calibrate_charuco.write_board(self.roots.board, float(self.prefs["square_mm"]))
            # the PDF itself, not its folder: a folder window opens behind this frameless window
            if hasattr(os, "startfile"):
                os.startfile(str(pdf))
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf)))
            return self._calib_state()
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def calib_folder_new(self, kind):
        """Today's folder for calibration photos (photo, video) or for the session check (check_photo...)."""
        def go():
            if kind not in CALIB_MODES + tuple("check_" + m for m in CALIB_MODES):
                return {"error": "bad_value"}
            d = self.roots.calib_raw / f"{date.today():%Y%m%d}_{kind}"
            d.mkdir(parents=True, exist_ok=True)
            self._folder_board(d)
            return {**self._calib_state(), "folder": d.name}
        return self._answer(go)

    def _folder_board(self, d):
        """board.json in a photo folder: the active board, written while the folder has no photos (so it is the
        board the first photo is of); a folder with photos keeps the one it has."""
        p = d / calibrate_charuco.BOARD_FILE
        if p.exists() and any(x.suffix.lower() in IMAGE_SUFFIXES for x in d.iterdir()):
            return
        a = self._board()["active"]
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
        """A photo from the phone into a folder chosen by name, under the name the phone gave it."""
        return self._answer(lambda: self._shoot_into(self._calib_dir(name)))

    def _shoot_into(self, d):
        blocked = self._free_camera()
        if blocked:
            return blocked
        self._folder_board(d)

        def done(path, error):
            if error is not None:
                self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error)})
                return
            dest = d / Path(path).name
            if dest.exists():
                self._emit({"type": "acquire_failed", "key": "calibration", "error": "exists", "message": dest.name})
                return
            shutil.move(str(path), str(dest))
            if self._board_win is not None:
                self._board_win.hold = self._series_hold()     # the board of these photos stays while it fits
            self._emit({"type": "calib_photo", "folder": d.name, "name": dest.name})
        ok = self._spawn("phone", lambda: self.phone().shoot(self.roots.staging / "calibration"), done)
        return {"ok": ok, "folder": d.name} if ok else {"error": "busy"}

    @pyqtSlot(str, result=str)
    def calib_import(self, name):
        """Photo files added to a folder by name, or to the series (name empty)."""
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
            return {**self._calib_state(), "added": added, "skipped": skipped}
        return self._answer(go)

    @staticmethod
    def _captured(fn):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = fn()
        return result, buf.getvalue()

    @pyqtSlot(str, str, result=str)
    def calib_run(self, name, mode):
        """Calibrates the camera mode on the photos of the folder; the result waits for calib_save."""
        def go():
            d = self._calib_dir(name)
            active = self._board()["active"]

            def work():
                board = calibrate_charuco.folder_board(d) or (active["square_mm"], active["squares"],
                                                             active["marker_mm"])
                return self._captured(lambda: calibrate_charuco.run_calibrate(d, None, None, *board))

            def done(result, error):
                if error is not None:
                    self._emit({"type": "calib", "action": "calibrate", "mode": mode, "ok": False, **self._err(error, "calibration")})
                    return
                cal, log = result
                self._pending_cal[mode] = cal
                self._emit({"type": "calib", "action": "calibrate", "mode": mode, "log": log,
                            "ok": cal["rms_px"] < calibrate_charuco.MAX_RMS, "result": self._cal_summary(cal)})
            ok = self._spawn("calibration", work, done)
            return {"ok": ok} if ok else {"error": "busy"}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def calib_save(self, mode):
        def go():
            cal = self._pending_cal.pop(mode, None)
            if cal is None:
                return {"error": "nothing"}
            p = self._intrinsics_path(mode)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(cal, indent=1), encoding="utf-8")
            return self._calib_state()
        return self._answer(go)

    @pyqtSlot(str, str, result=str)
    def calib_check(self, name, mode):
        """Start-of-session check of the stored intrinsics on the photos of the folder."""
        def go():
            d = self._calib_dir(name)
            intr = self._intrinsics_path(mode)
            if not intr.exists():
                return {"error": "no_intrinsics", "message": mode}
            active = self._board()["active"]

            def work():
                board = calibrate_charuco.folder_board(d) or (active["square_mm"], active["squares"],
                                                             active["marker_mm"])
                return self._captured(lambda: calibrate_charuco.run_check(d, intr, *board))

            def done(result, error):
                if error is not None:
                    self._emit({"type": "calib", "action": "check", "mode": mode, "ok": False, **self._err(error, "check")})
                    return
                (ok, report), log = result
                p = report["params"]
                line = {"time": datetime.now().isoformat(timespec="seconds"), "mode": mode, "folder": name,
                        "ok": int(ok), "n_views": report["n_views"], "rms_stored_px": report["rms_stored_px"],
                        **{f"{k}_change": p[k]["change"] for k in ("fx", "fy", "cx", "cy")}}
                trials.write_csv(self._checks_path(), trials.read_csv(self._checks_path()) + [line], CHECK_FIELDS)
                self._emit({"type": "calib", "action": "check", "mode": mode, "ok": ok, "log": log, "report": report})
            started = self._spawn("calibration", work, done)
            return {"ok": started} if started else {"error": "busy"}
        return self._answer(go)

    # ── the calibration board: on a monitor (the default) or printed ─────
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

    def _monitor_of(self, device):
        """The monitor (screen_board.monitors, with its ruler check) of a device name: the board window asks it each
        time it is laid out, so a ruler check or a monitor changed is taken at once."""
        mons, _ = self._screens()
        return next((m for m in mons if m["device"] == device), None)

    def _app_screen(self):
        """The screen the app's window is on (the main one without a window: the tests)."""
        app = QApplication.instance()
        sc = self.window.screen() if self.window is not None else None
        return sc or (app.primaryScreen() if app is not None else None)

    def _split(self):
        """The board's share of the monitor's width beside the app, as the operator last left it."""
        try:
            return min(0.8, max(0.2, float(self.prefs["board"].get("split", 0.5))))
        except (TypeError, ValueError):
            return 0.5

    def _beside_rect(self, screen):
        """Where the board goes beside the app before the dock places it: the right share of the work area."""
        g = screen.availableGeometry()
        bw = round(g.width() * self._split())
        return QRect(g.x() + g.width() - bw, g.y(), bw, g.height())

    def _board(self):
        """The board as it stands: kind; where it goes on a monitor (full screen on another monitor, or beside the app
        on the app's own) with its layout there, the real one while it is shown; the printed board; and the active
        board, what the Calibration and Live AR modules use."""
        mons, screens = self._screens()
        b = self.prefs["board"]
        kind = b.get("kind") if b.get("kind") in BOARD_KINDS else "screen"
        place = b.get("place") if b.get("place") in BOARD_PLACES else "auto"
        app_sc = self._app_screen()
        app_dev = app_sc.name() if app_sc is not None else ""
        others = [m for m in mons if m["device"] != app_dev]
        if place == "auto":
            place = "full" if others else "beside"
        if place == "full":
            mon = next((m for m in mons if m["key"] == b.get("monitor")), None) or next(
                (m for m in mons if m["key"] == screen_board.default_monitor(others or mons)), None)
        else:
            mon = next((m for m in mons if m["device"] == app_dev), None) or (mons[0] if mons else None)
        win = self._board_win
        lay, error = None, ""
        if win is not None:
            lay, error, mon = win.lay, win.error, win.mon or mon
        elif mon is not None:
            sc = screens.get(mon["key"])
            if place == "full" or sc is None:
                px = mon["px"]
            else:
                r, dpr = self._beside_rect(sc), float(sc.devicePixelRatio())
                px = (int(round(r.width() * dpr)), int(round(r.height() * dpr)))
            try:
                lay = screen_board.layout(px, mon["pitch_mm"] or GUESS_PITCH)
            except ValueError:
                error = "too_small"
        screen = screen_board.spec(lay, mon) if lay else None
        printed = {"kind": "printed",
                   **calibrate_charuco.board_spec(calibrate_charuco.make_board(float(self.prefs["square_mm"])))}
        active = screen if kind == "screen" and screen else printed
        return {"kind": kind, "place": place, "mons": mons, "screens": screens, "mon": mon, "lay": lay,
                "error": error, "printed": printed, "active": active, "others": bool(others)}

    def _board_state(self):
        st = self._board()
        mon, lay = st["mon"], st["lay"]
        view = None
        if lay:
            view = {k: lay[k] for k in ("squares", "square_px", "marker_px", "square_mm", "marker_mm", "width_mm",
                                        "height_mm", "pitch_mm")}
            view["width_u_mm"] = lay["width_mm"] * ((mon or {}).get("rel_u") or 0.0)
        keys = ("key", "label", "device", "primary", "px", "native", "is_native", "dpr", "diag_in", "pitch_mm",
                "pitch_data_mm", "rel_u", "source", "sources", "agree", "spread", "ruler")
        win = self._board_win
        return {"kind": st["kind"], "place": self.prefs["board"].get("place", "auto"), "placed": st["place"],
                "others": st["others"], "monitors": [{k: m[k] for k in keys} for m in st["mons"]],
                "monitor": mon["key"] if mon else "", "mon": {k: mon[k] for k in keys} if mon else None,
                "layout": view, "error": st["error"], "printed": st["printed"], "active": st["active"],
                "shown": win is not None, "shown_on": win.key if win is not None else "",
                "docked": bool(self._dock is not None and self._dock.docked),
                "held": bool(win is not None and win.hold and win.lay
                              and win.lay["square_px"] == win.hold.get("square_px")
                              and list(win.lay["squares"]) == list(win.hold.get("squares") or []))}

    @pyqtSlot(result=str)
    def board_state(self):
        return self._answer(self._board_state)

    def _num(self, v, n=1):
        text = f"{v:.{n}f}"
        return text.replace(".", ",") if self.lang in COMMA_LANGS else text

    @pyqtSlot(str, result=str)
    def board_set(self, payload):
        """{kind, place (auto, beside, full), monitor (full screen on that one), dock (the board shown in a window
        docked beside the app again), printed_mm (the printed board's long side as measured), ruler_mm (the length
        between the arrows of the board on a monitor, as a ruler reads it), forget (the ruler check of that
        monitor)}."""
        def go():
            c = json.loads(payload)
            b = dict(self.prefs["board"])
            again = False
            if c.get("kind") in BOARD_KINDS:
                b["kind"] = c["kind"]
            if c.get("place") in BOARD_PLACES and c["place"] != b.get("place"):
                b["place"], again = c["place"], True
            if c.get("monitor") and c["monitor"] != b.get("monitor"):
                b["monitor"], b["place"], again = str(c["monitor"]), "full", True
            self.prefs.set("board", b)
            if c.get("printed_mm") is not None:
                sq = float(c["printed_mm"]) / calibrate_charuco.BOARD["squares"][0]
                if not 5 <= sq <= 100:
                    return {"error": "bad_value", "message": str(c["printed_mm"])}
                self.prefs.set("square_mm", round(sq, 3))
            if c.get("ruler_mm") is not None or c.get("forget"):
                st = self._board()
                mon, lay = st["mon"], st["lay"]
                if mon is None or lay is None:
                    return {"error": "no_monitor"}
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
                if self._board_win is not None:
                    self._board_win.relayout()       # the same window, the board in the new millimetres
            if self._board_win is not None:
                if b["kind"] != "screen":
                    self._hide_board()
                elif again or (c.get("dock") and not self._board_win.full):
                    self._hide_board()                  # another place or monitor, or docked again
                    self._show_board(self._board())
            self._sync_live()
            return self._board_state()
        return self._answer(go)

    def _board_words(self):
        words = {}
        for code in ("en", self.lang):
            with contextlib.suppress(OSError, ValueError):
                words.update(json.loads((APP / "ui" / "strings" / f"{code}.json").read_text(encoding="utf-8")))
        return {"title": words.get("board.window.title", "ImpactGuide"),
                "measure": words.get("board.window.measure", "{0} mm"), "info": words.get("board.window.info", ""),
                "small": words.get("board.window.small", ""), "comma": self.lang in COMMA_LANGS}

    def _min_board(self, mon, screen):
        """The least size of the board's window, logical pixels: the smallest board at that monitor's pitch."""
        w, h = screen_board.min_px((mon or {}).get("pitch_mm") or GUESS_PITCH)
        dpr = float(screen.devicePixelRatio()) if screen is not None else 1.0
        return int(w / dpr) + 2, int(h / dpr) + 2

    def _show_board(self, st):
        """The board where st (a _board()) puts it, in place of the one on show if any: full screen on its monitor, or
        in a window docked beside the app (the app makes room for it)."""
        mon = st["mon"]
        screen = st["screens"].get(mon["key"]) if mon else None
        if screen is None:
            return False
        self._hide_board()
        hold = self._series_hold()
        if st["place"] == "full":
            win = board_window.BoardWindow(screen, self._monitor_of, self._board_words(), full=True,
                                           owner=self.window, hold=hold)
        else:
            win = board_window.BoardWindow(screen, self._monitor_of, self._board_words(), full=False,
                                           rect=self._beside_rect(screen), owner=self.window, hold=hold)
            if self.window is not None:
                dock = board_window.Dock(self.window, win, self._split(), min_board=self._min_board(mon, screen))
                if dock.fits():
                    self._dock = dock
                    dock.split_set.connect(self._split_set)
                    dock.undocked.connect(lambda d=dock: self._emit({"type": "board", "action": "undocked"})
                                          if self._dock is d else None)
        win.closed.connect(lambda why, w=win: self._board_closed(w, why))
        win.relaid.connect(lambda w=win: self._board_relaid(w))
        self._board_win = win
        win.show_board()
        if self._dock is not None:
            self._dock.dock()
        return True

    def _split_set(self, split):
        self.prefs.set("board", {**self.prefs["board"], "split": round(float(split), 3)})

    def _hide_board(self):
        win, self._board_win = self._board_win, None
        dock, self._dock = self._dock, None
        if dock is not None:
            dock.give_back()
        if win is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                win.closed.disconnect()
            with contextlib.suppress(TypeError, RuntimeError):
                win.relaid.disconnect()
            win.close()

    def _screen_removed(self, screen):
        if self._board_win is not None and self._board_win.screen_ is screen:
            self._board_win.changed()

    def _board_closed(self, win, why):
        if self._board_win is win:
            self._board_win = None
            dock, self._dock = self._dock, None
            if dock is not None:
                dock.give_back()
            self._emit({"type": "board", "action": "closed", "reason": why})

    def _board_relaid(self, win):
        """The board on show changed (the window moved, resized, went to another monitor): the live view takes it at
        once, and the page is told."""
        if win is not self._board_win:
            return
        self._sync_live()
        self._emit({"type": "board", "action": "relaid", "error": win.error})

    @pyqtSlot(result=str)
    def board_show(self):
        def go():
            st = self._board()
            if st["kind"] != "screen":
                self.prefs.set("board", {**self.prefs["board"], "kind": "screen"})
                st = self._board()
            if not self._show_board(st):
                return {"error": "no_monitor"}
            self._sync_live()
            return self._board_state()
        return self._answer(go)

    @pyqtSlot(result=str)
    def board_hide(self):
        def go():
            self._hide_board()
            return self._board_state()
        return self._answer(go)

    def _live_board(self):
        """The calibration board as live_ar wants it: the layout of the one on a monitor, None for the A4 one."""
        a = self._board()["active"]
        screen = a["kind"] == "screen"
        return {"square_mm": float(a["square_mm"]), "squares": list(a["squares"]) if screen else None,
                "marker_mm": float(a["marker_mm"]) if screen else None}

    def _sync_live(self):
        """The live view, if made, on the board as it is now."""
        if self._live is None:
            return
        lb = self._live_board()
        now = self._live.settings()
        if any(now.get(k) != v for k, v in lb.items()):
            self._live.configure(**lb)

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
                                        replay=self.replay)
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
        self._sync_live()       # the board changed here, on the Calibration module, or its window moved
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
        self._hide_board()
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
