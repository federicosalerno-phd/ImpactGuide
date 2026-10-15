"""ImpactGuide: the desktop app of the marker impact bench.

A frameless SlantUI window hosting a page (ui.html and ui/) that talks to this
backend over QWebChannel. The app guides the protocol step by step and keeps
what the bench produces: the phone calibration, the live AR view and the
impact trials today; scans, analysis, charts and the dataset are the next pages.

This file is the host: the window, the bridge, the background jobs. The logic
is in 04_code/src (trials.py, acquisition.py, calibrate_charuco.py, live_ar.py),
which the host only calls; everything there also runs without the app. The
live frames reach the page as MJPEG from 127.0.0.1 (live_ar.MjpegServer).

    python impactguide_app.py [--simulate DIR] [--replay PATH] [--lang en|it|fr|es|de|zh] [--page NAME]

--simulate keeps every file under DIR and stands in for the phone and the
scanner, for trying the app without the bench and for test_app.py. --replay
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
from slantui.shell.qt import QColor, QDesktopServices, QFileDialog, QTimer, QUrl, USE_QT6  # noqa: E402
from slantui.tokens import PALETTES  # noqa: E402

import acquisition  # noqa: E402
import calibrate_charuco  # noqa: E402
import live_ar  # noqa: E402
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
PAGES = ("calibration", "live", "trials", "scans", "analysis", "charts", "dataset")
ICON = APP / "ui" / "impactguide-icon.svg"  # the round app icon: desktop, taskbar, window
MARK = APP / "ui" / "impactguide-mark.svg"  # the same drawing without its disc, cut to fill the band's square
CALIB_MODES = ("photo", "video")
IMAGE_SUFFIXES = tuple(sorted(calibrate_charuco.IMAGE_SUFFIXES))
CHECK_FIELDS = ("time", "mode", "folder", "ok", "n_views", "rms_stored_px", "fx_change", "fy_change",
                "cx_change", "cy_change")


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
                "calib_mode": "photo", "phone": {"camera_dir": acquisition.CAMERA_DIR, "shutter": "volume_down",
                                                  "switch_modes": True},
                "live": {"board": "sheet", "face": True, "mandible": False, "axes": True, "registration": ""}}

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
        self._pump = QTimer(self)
        self._pump.setInterval(150)
        self._pump.timeout.connect(self._drain)
        self._pump.start()

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
        if isinstance(e, acquisition.AcquisitionError):
            return {"error": e.cause, "message": e.detail}
        if isinstance(e, SystemExit):
            return {"error": cause or "failed", "message": str(e.code)}
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
                           "page": self.prefs["page"] if self.prefs["page"] in PAGES else "trials",
                           "simulate": bool(self.roots.simulate), "qt": "PyQt6" if USE_QT6 else "PyQt5"})

    @pyqtSlot(str)
    def set_lang(self, code):
        if code in LANGS:
            self.lang = code
            self.prefs.set("lang", code)

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
                self._phone = acquisition.FakePhone(photo_px=self.run().config.photo_px)
            else:
                p = self.prefs["phone"]
                self._phone = acquisition.Phone(camera_dir=p["camera_dir"], shutter=p["shutter"],
                                                switch_modes=bool(p["switch_modes"]))
        return self._phone

    @pyqtSlot()
    def phone_refresh(self):
        def done(status, error):
            self.phone_status = status if status else {"ok": False, **self._err(error)}
            self._emit({"type": "phone", **self.phone_status})
        self._spawn("phone_status", lambda: self.phone().status(), done)

    @pyqtSlot(result=str)
    def phone_prefs(self):
        return json.dumps({**self.prefs["phone"], "adb": acquisition.find_adb() or "", "status": self.phone_status})

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
        folders = []
        if r.calib_raw.is_dir():
            for d in sorted((d for d in r.calib_raw.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True):
                n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
                folders.append({"name": d.name, "path": str(d), "n": n, "kind": d.name.split("_", 1)[-1]})
        b = calibrate_charuco.BOARD
        nx, ny = b["squares"]
        board_file = r.board / f"charuco_{nx}x{ny}_{float(self.prefs['square_mm']):g}mm_{b['dictionary'].lower()}_a4.pdf"
        return {"board": {**b, "squares": list(b["squares"])}, "square_mm": self.prefs["square_mm"],
                "mode": self.prefs["calib_mode"], "modes": modes, "last_check": last, "folders": folders,
                "raw": str(r.calib_raw), "out": str(r.calib_out), "board_file": str(board_file) if board_file.exists() else "",
                "board_dir": str(r.board), "phone": self.phone_status, "busy": sorted(self._busy),
                "min_views": calibrate_charuco.MIN_VIEWS, "max_rms": calibrate_charuco.MAX_RMS}

    @pyqtSlot(result=str)
    def calib_state(self):
        return self._answer(self._calib_state)

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
            return {**self._calib_state(), "folder": d.name}
        return self._answer(go)

    def _calib_dir(self, name):
        d = (self.roots.calib_raw / name).resolve()
        if d.parent != self.roots.calib_raw.resolve() or not d.is_dir():
            raise LookupError(f"no folder {name}")
        return d

    @pyqtSlot(str, result=str)
    def calib_shoot(self, name):
        """A photo from the phone into the folder, under the name the phone gave it."""
        def go():
            d = self._calib_dir(name)
            blocked = self._free_camera()
            if blocked:
                return blocked

            def done(path, error):
                if error is not None:
                    self._emit({"type": "acquire_failed", "key": "calibration", **self._err(error)})
                    return
                dest = d / Path(path).name
                if dest.exists():
                    self._emit({"type": "acquire_failed", "key": "calibration", "error": "exists", "message": dest.name})
                    return
                shutil.move(str(path), str(dest))
                self._emit({"type": "calib_photo", "folder": name, "name": dest.name})
            ok = self._spawn("phone", lambda: self.phone().shoot(self.roots.staging / "calibration"), done)
            return {"ok": ok} if ok else {"error": "busy"}
        return self._answer(go)

    @pyqtSlot(str, result=str)
    def calib_import(self, name):
        def go():
            d = self._calib_dir(name)
            pattern = " ".join("*" + s for s in IMAGE_SUFFIXES)
            paths, _ = QFileDialog.getOpenFileNames(None, "ImpactGuide", str(Path.home()), f"photos ({pattern})")
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
            square = float(self.prefs["square_mm"])

            def work():
                return self._captured(lambda: calibrate_charuco.run_calibrate(d, None, None, square))

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
            square = float(self.prefs["square_mm"])

            def work():
                return self._captured(lambda: calibrate_charuco.run_check(d, intr, square))

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

    # ── live AR ──────────────────────────────────────────────────────────
    def live(self):
        """The live AR, made at first use and kept for the session: its overlay thread owns the OpenGL
        context (live_ar.py). In simulation a synthetic board stands in for the phone."""
        if self._live is None:
            p = self.prefs["live"]
            self._live = live_ar.LiveAR(board=p["board"] if p["board"] in live_ar.BOARDS else "sheet",
                                        square_mm=float(self.prefs["square_mm"]), registration=p["registration"] or None,
                                        face=p["face"], mandible=p["mandible"], axes=p["axes"],
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
        ar = self.live()
        square = float(self.prefs["square_mm"])
        if ar.settings()["square_mm"] != square:      # changed on the Calibration page
            ar.configure(square_mm=square)
        return {**ar.status(), "url": self._live_server.url, "prefs": self.prefs["live"], "phone": self.phone_status,
                "busy": sorted(self._busy), "simulate": bool(self.roots.simulate),
                "intrinsics_file": str(self.roots.calib_out / "intrinsics_stream.json")}

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

    @pyqtSlot(str, result=str)
    def live_settings(self, payload):
        """{board, face, mandible, axes}, kept for the next session too."""
        def go():
            given = {k: v for k, v in json.loads(payload).items() if k in ("board", "face", "mandible", "axes")}
            p = {**self.prefs["live"], **given}
            if p["board"] not in live_ar.BOARDS:
                return {"error": "bad_value", "message": "board"}
            for k in ("face", "mandible", "axes"):
                p[k] = bool(p[k])
            self.prefs.set("live", p)
            self.live().configure(board=p["board"], face=p["face"], mandible=p["mandible"], axes=p["axes"])
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
        if self._live_server is not None:
            self._live_server.close()
        if self._live is not None:
            self._live.shutdown()


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
    if args.page in PAGES:
        prefs.set("page", args.page)
    app = Application(APP_NAME, app_id=APP_ID)
    app.setWindowIcon(app_icon())
    backend = Backend(roots, prefs, lang=args.lang, replay=args.replay)
    page = prepare_page((roots.simulate or LOCAL) / "page")
    title = APP_NAME + (" (simulation)" if roots.simulate else "")
    # the band the window draws before the page has one shows the mark as a button when the page will:
    # away from home, the trial guide (ui/01-shell.js, showHome)
    win = Window(page, bridge=backend, title=title, palette=PALETTE, size=(1440, 900), min_size=(1180, 700),
                 logo=mark_file(), brand_action=prefs["page"] != "trials")
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
