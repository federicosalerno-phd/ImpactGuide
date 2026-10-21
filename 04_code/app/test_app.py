"""ImpactGuide test: complete trials simulated with stand-in files, and the calibration.

backend   the slots the page calls, without a window: a trial from the first
          step to the log row, a cap shift over the threshold and the
          reposition that follows, the app closed in the middle of a trial and
          resumed, a drape trial with photos and video with and without the
          drape, a correction of a logged value, a void trial, a new
          configuration, the completeness check of trials.py, the live AR view
          on a synthetic stream (both boards, a photo that stops it), the link
          to a simulated phone found by itself (no phone, the cable, Wi-Fi
          paired with the QR code and Wireless debugging switched on from the
          PC, Wi-Fi taken by itself when the cable is out and the cable when it
          is back, a drop and the reconnection, pairing with a code, plain
          words for each failure, the live view on the phone's screen and its
          added latency); the calibration board drawn by the page (the first
          question, the layout from the pixels of the page's area and the pitch
          of the monitor, the image sent pixel for pixel and only when it
          changes, laid out again for another area and taken at once by the
          live view and the calibration, too small said, the ruler check; the
          3D-printed plate and the A4 sheet with their caliper readings); the
          guided photos (the simulated phone photographing the board in use at
          the pose of the plan: each photo checked as it comes, kept, or put
          aside as blurred, without the board or of the phone held the other
          way; the plan's least and most; a board changed after the photos);
          the calibration with its errors per photo and over the board, a photo
          left out by hand and the calibration again, saved; the session check;
          the streaming mode on frames of the live stream; the twin of the
          bench (its model made again once the scene rests for two minutes,
          a new version only for a changed scene, dataset and network out of
          date by their fingerprints, the trial's setting, the marker's
          motion and the error at the targets after the scans). Then what must
          never come back: an exception out of a Qt callback that closes the
          app, a library's SystemExit, a traceback or a raw error text sent to
          the page, preferences or checks lost to a half-written file, the
          page folder of an app still open, the phone's view on the network
          before it is asked for, a path the page sends opened whatever it is,
          a photo filed under a trial voided meanwhile, close notes refused
          after notes typed at the start, a staged file dropped by a confirm,
          a stream event without its purpose, a calibration judged by today's
          plan, a series cut at midnight, a photo left unchecked, a scan staged
          before it is whole, the ruler check at another page scale.
--ui DIR  the same through the real window: the app opens at home (no rail),
          the page's own buttons are pressed, one trial is run to its log
          row, the mark on the band leads home and a choice of home to its
          module and its steps; Calibrate: the first question answered, the
          board drawn by the page in its work area and grabbed back from the
          screen, pixel for pixel, in the board step, after the window is
          resized by hand, in the photos, beside the live picture of the
          streaming mode and beside the live view (every corner found in the
          grab); the ruler check; the plate in 3D and its readings; twelve
          photos taken with the page's Take button, calibrated, the worst left
          out with a press on its bar, calibrated again, saved; the live view
          started from its step, its opacity set with the slider, and left; the
          twin opened from the trial's page within a second, its ready views,
          the trial after its scans with the marker's motion enlarged; the
          Phone panel in the panel down the right (home and a module) and Wi-Fi
          paired with the QR code; then every step of every module and the
          Phone panel, with their Advanced parts open, are checked for words
          that are cut or wrap and for anything laid over the page, at the
          smallest window and at 1920 x 1032, in the six languages (each view
          measured twice, only what stays wrong counts, with the window at its
          size); last the board pixel for pixel at 125 and 150 % display
          scaling, set over the screen's own. The window lies right of every
          screen, out of the way of whoever uses the computer (the focus given
          back to them), and comes onto the screen only while the screen itself
          is grabbed. It is grabbed into DIR at each stage, and on every view
          in English and Italian (large) and in English and German (smallest).

    python test_app.py [--work DIR] [--ui SHOTS] [--no-calibration]

Exit code 1 on failure. Everything is written under --work (default: a folder
in the system temp), never in the project's data; the app's errors.log of the
run is DIR/backend/logs (DIR/ui/logs with --ui). --work is emptied at the
start only when it is new, empty, or a work folder of this test (it carries
the file .impactguide-test): any other folder is refused, never deleted.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import impactguide_app as ig  # noqa: E402  (sets the bytecode cache and the path to 04_code/src)

import calibrate_charuco  # noqa: E402
import scans_page  # noqa: E402
import screen_board  # noqa: E402
import trials  # noqa: E402

FAILS = []
TEST_MARKER = ".impactguide-test"       # a work folder of this test carries this file
# what a work folder of earlier versions of this test holds (they wrote no marker yet)
TEST_ENTRIES = {"backend", "ui", "scale_1.25", "scale_1.5", TEST_MARKER}


def expect(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_work(folder):
    """The work folder emptied and marked as this test's own; False, and nothing touched, when it is a folder of
    something else (not empty and neither marked nor laid out as this test's work folder)."""
    folder = Path(folder)
    if folder.exists():
        entries = {p.name for p in folder.iterdir()}
        if entries and TEST_MARKER not in entries and not entries <= TEST_ENTRIES:
            return False
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    (folder / TEST_MARKER).write_text("work folder of test_app.py: emptied at each run\n", encoding="utf-8")
    return True


def fresh(folder):
    """A folder of this test's own inside its work folder, made empty (a run on a reused work folder, or a check
    called again, finds the one of before)."""
    folder = Path(folder)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    return folder


# ── the backend, without a window ────────────────────────────────────────────
class Driver:
    """Calls the slots as the page does and waits for the background jobs."""

    def __init__(self, app, work):
        self.app, self.work = app, work
        self.events = []
        self.backend = None
        self.open_backend()

    def open_backend(self):
        roots = ig.Roots(self.work)
        self.backend = ig.Backend(roots, ig.Prefs(roots.prefs), lang="en")
        self.backend.happened.connect(lambda raw: self.events.append(json.loads(raw)))

    def call(self, slot, *args):
        return json.loads(getattr(self.backend, slot)(*args))

    def wait(self, test, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.app.processEvents()
            if test():
                return True
            time.sleep(0.05)
        return False

    def wait_event(self, kind, timeout=30):
        n = len(self.events)
        ok = self.wait(lambda: any(e["type"] == kind for e in self.events[n:]), timeout)
        return next((e for e in self.events[n:] if e["type"] == kind), None) if ok else None

    def staged(self, key):
        return key in (self.backend.cur.get("staged") or {})

    def do_step(self, step, spheres=None, close=None):
        """One step as the operator does it; returns the new state."""
        key, kind = step["key"], step["kind"]
        if kind == "photo":
            self.call("step_shoot", key)
            assert self.wait(lambda: self.staged(key)), f"photo {key} never arrived"
        elif kind == "video":
            self.call("step_video", key, "start")
            assert self.wait_event("recording"), "recording never started"
            self.call("step_video", key, "stop")
            assert self.wait(lambda: self.staged(key)), f"video {key} never arrived"
        elif kind == "scan":
            a = self.call("scan_watch", key)
            assert a.get("ok"), a
            assert self.wait(lambda: self.staged(key)), f"scan {key} never exported"
        values = {}
        if kind == "spheres":
            values = spheres or {"spheres_ok": "1", "cap_shift_mm": "0.8"}
        if kind == "close":
            values = close or {"second_hit": "0", "cap_replaced": "0", "notes": ""}
        return self.call("step_confirm", key, json.dumps(values))

    def run_trial(self, mass="", stop_after=None, spheres=None, close=None):
        """Starts the planned trial (or continues the open one) and does its steps; stops after the
        step stop_after if given. Returns (trial id, keys done)."""
        s = self.call("trials_state")
        if not s["trial"]["open"]:
            s = self.call("trial_start", json.dumps({"swing_mass_g": mass}))
        trial, done = s["trial"]["trial"], []
        while True:
            step = s["steps"][s["step"]]
            s = self.do_step(step, spheres, close)
            assert "error" not in s, s
            done.append(step["key"])
            if s.get("closed") or step["key"] == stop_after:
                return trial, done


class Opened:
    """Stands in for ig.open_externally: records what the app would have opened, opens nothing."""

    def __init__(self):
        self.paths = []

    def __call__(self, path):
        self.paths.append(Path(path))

    def __enter__(self):
        self.real, ig.open_externally = ig.open_externally, self
        return self

    def __exit__(self, *_):
        ig.open_externally = self.real


def test_robust(d, work):
    """What the app must survive or refuse, without a trial: an exception out of a Qt callback or out of a job's
    result, a library's SystemExit, a path from the page that is not a data folder, broken or half-written files, a
    second start while the page folder of the first is held, a calibration log kept off stdout, a scan file
    still being written; errors said with a code only, their details in the run's own errors.log."""
    from slantui.shell.qt import QTimer
    b = d.backend
    # answers: strict JSON, a code, no traceback, no exception text
    a = d.call("step_confirm", "photo_az40", "{}")
    expect(ig.to_json({"a": float("nan"), "b": [float("inf"), 1.5]}) == '{"a": null, "b": [null, 1.5]}'
           and a.get("error") == "no_trial" and "trace" not in a and a.get("message") == "" and a.get("args") == {},
           f"answers: NaN sent as null, a refusal said by its code with no trace or exception text ({a})")
    calls = {}

    def exits(*_):
        calls["n"] = calls.get("n", 0) + 1
        sys.exit("library code that ends the process")
    a = b._answer(exits)
    expect(json.loads(a).get("error") == "failed" and calls["n"] == 1,
           f"a SystemExit inside a slot is an answer, the app goes on ({a})")

    class Coded(Exception):
        """A coded error of the modules: its code and the values of its sentence."""

        def __init__(self):
            super().__init__("trial_log.csv is open in another program")
            self.code, self.params = "log_locked", {"file": Path("trial_log.csv"), "tries": 20}

    def locked():
        raise Coded()
    c = json.loads(b._answer(locked))
    expect(c == {"error": "log_locked", "args": {"file": "trial_log.csv", "tries": 20}, "message": ""},
           f"a coded error of the modules: its code and its values to the page, never its text ({c})")
    # open_path: only folders of the data; a program or a folder elsewhere is refused and never opened
    with Opened() as opened:
        exe = d.call("open_path", sys.executable)
        home = d.call("open_path", str(Path.home()))
        b.run().folder.mkdir(parents=True, exist_ok=True)
        ok = d.call("open_path", str(b.run().folder))
        bad_trial = d.call("trial_open_folder", "..\\..")
    expect(exe.get("error") == "not_found" and home.get("error") == "bad_value" and ok.get("ok")
           and bad_trial.get("error") == "bad_value" and opened.paths == [b.run().folder.resolve()],
           "open_path: a program and a folder outside the data refused and not opened, a data folder opened")
    # an exception out of a Qt callback, and out of a job's done(): logged in this run's errors.log, the app goes on
    ig.install_excepthooks()
    QTimer.singleShot(0, lambda: (_ for _ in ()).throw(RuntimeError("callback failure B01")))
    d.wait(lambda: False, 0.3)
    n = len(d.events)
    b._jobs.put(("probe", lambda r, e: 1 / 0, None, None))
    got = d.wait(lambda: any(x["type"] == "error" for x in d.events[n:]), 5)
    err = next((x for x in d.events[n:] if x["type"] == "error"), {})
    logged = b.roots.log.read_text(encoding="utf-8") if b.roots.log.exists() else ""
    expect(got and err.get("error") == "internal" and err.get("job") == "probe" and err.get("message") == ""
           and "callback failure B01" in logged
           and "ZeroDivisionError" in logged and b.roots.log.parent == work / "logs",
           f"an exception in a Qt callback and in a job's result: logged with its trace in {b.roots.log.parent}, "
           f"the page told in a code, the app still running")
    expect(ig.Roots().board == ig.LOCAL / "boards" and ig.Roots().log == ig.LOCAL / "errors.log",
           "the printable sheet and the log in the app's own folder, not in a developer's path")
    # an error of the page's scripts (ui/00-errors.js): in the log only, cut short, at most PAGE_ERRORS_MAX a run
    was = b._page_errors
    b.page_error("TypeError: probe of the page line 12 15-board.js" + "x" * 2000)
    b._page_errors = ig.PAGE_ERRORS_MAX
    b.page_error("one too many of the page")
    b._page_errors = was + 1
    logged = b.roots.log.read_text(encoding="utf-8") if b.roots.log.exists() else ""
    expect("page: TypeError: probe of the page line 12" in logged and "x" * (ig.PAGE_ERROR_CHARS + 1) not in logged
           and "one too many of the page" not in logged,
           "an error of the page's scripts: written to the log, cut short, never past the most a run")
    # preferences: a broken file kept aside, the defaults used; saving atomic; a camera folder that is no path refused
    prefs_dir = work / "prefs_check"
    fresh(prefs_dir)
    pp = prefs_dir / "preferences.json"
    pp.write_text('{"lang": "it", "phone": {', encoding="utf-8")
    p = ig.Prefs(pp)
    kept_aside = (prefs_dir / "preferences.bad.json").read_text(encoding="utf-8")
    p.set("lang", "de")
    p2 = ig.Prefs(pp)
    pp.write_text(json.dumps({"phone": {"camera_dir": "/sdcard/DCIM'; reboot; '"}}), encoding="utf-8")
    p3 = ig.Prefs(pp)
    leftovers = [x.name for x in prefs_dir.iterdir() if x.suffix == ".tmp"]
    expect(p["lang"] == "de" and kept_aside.startswith('{"lang"') and p2["lang"] == "de" and not leftovers
           and p3["phone"]["camera_dir"] == ig.acquisition.CAMERA_DIR,
           "preferences: a broken file kept as preferences.bad.json and the defaults used, saved whole, a camera "
           "folder with shell characters refused")
    # the other files the app writes: atomic JSON, CSV rows added at the end
    target = prefs_dir / "checks.json"
    ig.write_json_atomic(target, {"a": 1})
    ig.write_json_atomic(target, {"a": 2})
    csv_path = prefs_dir / "rows.csv"
    ig.append_csv_row(csv_path, {"time": "1", "mode": "photo"}, ("time", "mode"))
    ig.append_csv_row(csv_path, {"time": "2", "mode": "stream"}, ("time", "mode"))
    rows = trials.read_csv(csv_path)
    expect(json.loads(target.read_text(encoding="utf-8")) == {"a": 2} and [r["time"] for r in rows] == ["1", "2"]
           and not [x for x in prefs_dir.iterdir() if x.suffix == ".tmp"],
           "files: JSON replaced whole, no temporary file left; CSV rows added after the header, once")
    # a second start while the first app's page folder is held: a folder per process, stale ones removed
    pages = fresh(work / "pages")
    (pages / "page" / "ui").mkdir(parents=True)
    (pages / "page-99999999").mkdir()
    first = ig.prepare_page(pages)
    with open(first.parent / "ui" / "00-strings.js", "rb"):     # held, as the window of a running app holds it
        second = ig.prepare_page(pages)
    expect(first == second and second.is_file() and second.parent.name == f"page-{os.getpid()}"
           and not (pages / "page").exists() and not (pages / "page-99999999").exists(),
           "page folder: one per running app, the folders of apps gone removed, a held file no obstacle")
    # what a calibration says: through its log argument, never by taking over stdout (which every thread shares)
    stdout = sys.stdout

    def with_log(log=None):
        log("one line")
        log("another")
        return 3
    expect(ig.captured(with_log) == (3, "one line\nanother\n") and sys.stdout is stdout,
           "calibration log: the lines it logs, kept for the log file; stdout left alone")

    def failing(log=None):
        log("before the failure")
        raise ValueError("stopped")
    sink = []
    try:
        ig.captured(failing, sink=sink)
    except ValueError:
        pass
    expect(ig.log_text(sink) == "before the failure\n", "calibration log: what it said before failing is kept")
    expect(ig.plain_args({"got": (4080, 3072), "files": ["a.jpg", Path("b.jpg")], "path": Path("x"), "n": 3})
           == {"got": [4080, 3072], "files": ["a.jpg", "b.jpg"], "path": "x", "n": 3},
           "error values: a size and a list of names go as lists, a path as its text")
    expect(ig.CAMERA_DIR_RE.fullmatch("/storage/emulated/0/DCIM/Camera")
           and not any(ig.CAMERA_DIR_RE.fullmatch(x) for x in ("/sdcard/DCIM\n", "//sdcard", "/a/../b", "/a b", "a/b",
                                                                "/a'b", "/a/")),
           "camera folder of the phone: a plain absolute folder only")
    # a scan still being written: staged only once unchanged for acquisition.SCAN_SETTLE_S
    scans = work / "scan_check"
    fresh(scans)
    watch = ig.ScanWatch("t000", "scan_az40", scans)
    (scans / "export.ply").write_bytes(b"ply\n" + b"0" * 1000)
    settle = ig.acquisition.SCAN_SETTLE_S
    t0, found = time.monotonic(), None
    while found is None and time.monotonic() - t0 < settle + 6:
        found = watch.poll()
        took = time.monotonic() - t0
        time.sleep(0.25)
    expect(found == scans / "export.ply" and took >= settle - 0.05,
           f"scanner folder: a new scan taken only after {settle:g} s without change ({took:.1f} s)")
    # the test never deletes a folder that is not its own
    foreign = work / "foreign"
    fresh(foreign)
    (foreign / "data.csv").write_text("x\n", encoding="utf-8")
    expect(not prepare_work(foreign) and (foreign / "data.csv").is_file(),
           "--work: a folder that is not this test's is refused and left as it is")


def test_trial_fixes(d, work):
    """A photo that arrives after its trial was voided is not filed under the next one; notes typed at the start and
    at the close are both kept, the trial closed; a staged file is never dropped by confirming a step done before;
    a staged file carries its problems as codes; a trial's photos leave the camera app in its mode (no intent)."""
    b = d.backend
    run = b.run()
    # a photo on its way while the trial is voided: dropped, said, nothing staged
    s = d.call("trial_start", json.dumps({"swing_mass_g": "137.2"}))
    trial = s["trial"]["trial"]
    key = next(st["key"] for st in s["steps"] if st["kind"] == "photo")
    phone = b.phone()
    gate = threading.Event()
    real_shoot = phone.shoot
    phone.shoot = lambda out, **kw: (gate.wait(10), real_shoot(out, **kw))[1]
    try:
        a = d.call("step_shoot", key)
        d.call("trial_void", trial, "voided while a photo was on its way")
        n = len(d.events)
        gate.set()
        e = d.wait_event("acquire_failed", timeout=20)
    finally:
        del phone.shoot
    left = list((work / "staging" / trial).glob("*.*")) if (work / "staging" / trial).is_dir() else []
    expect(a.get("ok") and e is not None and e.get("error") == "trial_changed" and e.get("trial") == trial
           and e.get("args") == {"trial": trial} and not d.staged(key) and not left
           and not any(x["type"] == "staged" for x in d.events[n:]),
           f"{trial}: a photo arriving after its trial was voided is dropped and said ({e and e.get('error')}), "
           f"never staged for another trial")
    # notes typed at the start, more at the close: both kept, the trial closed
    s = d.call("trial_start", json.dumps({"swing_mass_g": "137.2", "notes": "start note"}))
    trial = s["trial"]["trial"]
    photo = next(st for st in s["steps"] if st["kind"] == "photo")
    d.run_trial(stop_after=photo["key"])
    # the photo step done, then a new photo staged for it: confirming again never drops it, nor replaces the filed one
    i = next(i for i, st in enumerate(d.call("trials_state")["steps"]) if st["key"] == photo["key"])
    d.call("step_go", i)
    # the new photo chosen by hand (Import): copied by a worker, then staged
    chosen = work / "chosen_photo.jpg"
    shutil.copy2(run.find(trial, photo["files"][0])[0], chosen)

    class Dialog:
        @staticmethod
        def getOpenFileName(*_):
            return str(chosen), ""
    real_dialog, ig.QFileDialog = ig.QFileDialog, Dialog
    try:
        imported = d.call("step_import", photo["key"])
    finally:
        ig.QFileDialog = real_dialog
    d.wait(lambda: d.staged(photo["key"]))
    staged = dict(b.cur["staged"][photo["key"]]) if d.staged(photo["key"]) else {"path": str(work / "none")}
    again = d.call("step_confirm", photo["key"], "{}")
    still = d.staged(photo["key"]) and Path(staged["path"]).is_file()
    d.call("step_discard", photo["key"])
    expect(imported.get("trial") and again.get("error") == "exists" and still and not Path(staged["path"]).exists()
           and isinstance(staged.get("problem_codes"), list) and len(run.find(trial, photo["files"][0])) == 1,
           f"{trial}: a photo imported by hand (copied in the background, then staged); the step confirmed again "
           f"with it staged is refused (exists), the staged photo kept until discarded, the filed one untouched")
    d.call("step_go", i + 1)
    d.run_trial(close={"second_hit": "0", "cap_replaced": "0", "notes": "close note"})
    row = run.log.row(trial)
    expect(row["status"] == "ok" and row["notes"] == "start note; close note",
           f"{trial}: notes of the start and of the close both kept ({row['notes']!r}), the trial closed")
    # a photo for a trial the app still takes for open, though it is not (voided by an edit of the log): dropped,
    # said, never staged for the trial that is open now (if any)
    stray = fresh(work / "staging" / "Z999") / "photo.jpg"
    stray.write_bytes(b"not a photo")
    kept_cur = b.cur
    b.cur = {"trial": "Z999", "step": 0, "staged": {}, "recording": "", "resumed": False}
    n = len(d.events)
    try:
        b._staged("Z999", photo["key"], stray, temp=True)
    finally:
        b.cur = kept_cur
    e = next((x for x in d.events[n:] if x["type"] == "acquire_failed"), None)
    expect(e is not None and e.get("error") in ("no_trial", "trial_changed") and not stray.exists()
           and not any(x["type"] == "staged" for x in d.events[n:]),
           f"a photo for a trial voided behind the app's back: dropped and said ({e and e.get('error')})")
    # a trial's photo leaves the camera app in its mode (the Pro mode of the calibration): no intent to it, as for a
    # calibration's photo; a camera that counted down all the same (its own timer on) is said once
    s = d.call("trial_start", json.dumps({"swing_mass_g": "137.2"}))
    trial = s["trial"]["trial"]
    photos = [st["key"] for st in s["steps"] if st["kind"] == "photo"]
    seen, real_take = [], phone.take

    def take(timeout=None, pressed=None, keep_mode=False):
        seen.append(keep_mode)
        phone.countdown_seen = True         # as the phone says after a photo that came seconds after its press
        return real_take(pressed=pressed, keep_mode=keep_mode)
    phone.take = take
    said, b._countdown_said = b._countdown_said, False
    n = len(d.events)
    try:
        d.run_trial(stop_after=photos[1])          # its first two photos
    finally:
        del phone.take
        phone.countdown_seen = False
    told = [x for x in d.events[n:] if x["type"] == "calib_photo" and x.get("action") == "countdown"]
    d.call("trial_void", trial, "test of the camera's mode")
    b._countdown_said = said
    expect(len(seen) >= 2 and all(k is True for k in seen) and len(told) == 1,
           f"{trial}: a trial's photos leave the camera app in its mode ({seen}); its countdown said once "
           f"({len(told)})")


def test_backend(app, work, calibration=True):
    d = Driver(app, work)
    run = lambda: d.backend.run()  # noqa: E731
    s = d.call("trials_state")
    expect(s["trial"]["trial"] == "t001" and s["totals"]["slots"] == 128 and not s["trial"]["open"],
           "first plan: t001, 128 slots, not started")
    expect("warning_codes" in s["trial"], "the plan carries its warnings as codes too")
    test_robust(d, work)
    a = d.call("trial_start", json.dumps({}))
    expect(a.get("error") == "swing_mass", "no trial without the swinging mass")

    # 1. a whole trial, with the cap shift over the threshold
    t1, keys = d.run_trial(mass="137.2", spheres={"spheres_ok": "1", "cap_shift_mm": "4.2"})
    row = run().log.row(t1)
    names = sorted(p.name for p in (work / "raw" / "campaign" / t1).iterdir())
    want = sorted(n for st in run().steps(t1) for n in st.files)
    expect(row["status"] == "ok" and row["swing_mass_g"] == "137.2" and row["cap_shift_mm"] == "4.20"
           and row["time_impact"] and row["time_end"] and names == want and len(names) == 9,
           f"{t1}: {len(keys)} steps, log row ok, {len(names)} files with their names")
    expect(not trials.Run("campaign", work / "raw", work / "protocol").missing(t1),
           f"{t1}: nothing missing for the completeness check")
    expect(not list((work / "staging").rglob("*.*")), "staging folder empty after the trial")

    # 2. the next one starts with the reposition, and the app is closed in the middle
    s = d.call("trials_state")
    reasons = s["trial"]["reference"].split(";")
    expect("reposition" in reasons and s["steps"][0]["key"] == "reposition"
           and s["trial"]["swing_mass_g"] == "137.2",
           f"{s['trial']['trial']}: reposition with the jig first ({s['trial']['reference']}), mass carried over")
    t2, _ = d.run_trial(stop_after="photo_az55")
    old = d.backend
    had = len(old._connections)
    old.shutdown()
    expect(had >= 2 and not old._connections and not old._pump.isActive() and not old._scan_timer.isActive(),
           f"app closed: its {had} connections to Qt undone and its timers stopped (a backend made after it is alone)")
    d.open_backend()
    s = d.call("trials_state")
    step = s["steps"][s["step"]]
    expect(s["trial"]["trial"] == t2 and s["trial"]["open"] and s["trial"]["resumed"] and step["key"] == "photo_az70",
           f"{t2}: app closed after photo_az55, resumes at {step['key']}")
    d.run_trial()
    expect(run().log.row(t2)["status"] == "ok", f"{t2}: closed after the resume")

    # 3. a drape trial: photos and video with the drape, drape off, photos and video, scans
    seq = run().pending()
    slot = next(r for r in seq if r["drape"] == "1" and r["holder"] == "A")
    s = d.call("trial_options", json.dumps({"reference": False, "settings": {
        "holder": slot["holder"], "angle_deg": slot["angle_deg"], "direction_deg": slot["direction_deg"], "drape": 1}}))
    expect(s["trial"]["slot"] == slot["slot"] and not s["trial"]["extra"],
           f"settings of drape slot {slot['slot']} chosen by hand: it fills that slot")
    t3, keys = d.run_trial()
    names = sorted(p.name for p in (work / "raw" / "campaign" / t3).iterdir())
    scan0 = f"scan_az{trials.Config().scan_az[0]}"
    order = [k for k in keys if k in ("drape_video", "drape_off", "photo_az40", "video", scan0)]
    expect(order == ["drape_video", "drape_off", "photo_az40", "video", scan0]
           and trials.file_name(t3, "photo", 85, drape=True) in names and trials.file_name(t3, "video", drape=True) in names
           and len([n for n in names if "_ref_" not in n]) == 12,
           f"{t3}: drape trial, {len(names)} files, with and without the drape, scans last")

    # 4. an extra trial outside the matrix (a hole the configuration does not have)
    s = d.call("trial_options", json.dumps({"reference": True, "settings": {
        "holder": "A", "angle_deg": 55, "direction_deg": 0, "drape": 0}}))
    t4 = s["trial"]["trial"]
    expect("manual" in s["trial"]["reference"] and s["trial"]["extra"],
           f"{t4}: extra trial outside the matrix, reference scans asked by hand")
    t4, _ = d.run_trial(close={"second_hit": "1", "cap_replaced": "1", "notes": "cap changed"})
    row = run().log.row(t4)
    expect(row["second_hit"] == "1" and row["cap_replaced"] == "1" and row["notes"] == "cap changed",
           f"{t4}: second hit, new cap and notes logged")

    # 5. a correction with its audit trail; raw files untouched
    photo = run().find(t1, trials.file_name(t1, "photo", 40))[0]
    before = sha(photo)
    L = d.call("log_edit", t1, "angle_deg", "45", "hole 45 used by mistake")
    e = L["edits"][-1] if L.get("edits") else {}
    expect(run().log.row(t1)["angle_deg"] == "45" and e.get("field") == "angle_deg" and e.get("reason")
           and e.get("old") and e.get("new") == "45" and e.get("time") and sha(photo) == before,
           f"{t1}: hole corrected, old and new value, time and reason in the audit trail, photo unchanged")
    bad = d.call("log_edit", t1, "angle_deg", "40", "")
    expect("error" in bad, "a correction without a reason is refused")

    # 6. a trial voided halfway: its files stay, its slot comes back
    s = d.call("trials_state")
    t5, _ = d.run_trial(stop_after="release")
    cell = trials.cell(run().log.row(t5))
    a = d.call("trial_void", t5, "pendulum hooked badly")
    row = run().log.row(t5)
    expect(row["status"] == "void" and row["void_reason"] == "pendulum hooked badly"
           and any(trials.cell(p) == cell for p in run().pending()) and (work / "raw" / "campaign" / t5).is_dir(),
           f"{t5}: void, reason logged, files kept, its cell pending again")

    # 7. a new configuration: the sequence is generated again around the trials done
    C = d.call("config_apply", json.dumps({"drape_trials": 22}))
    expect(C.get("totals", {}).get("slots") == 130 and C["totals"]["filled"] == 3 and C["totals"]["extra"] == 1
           and len(C["history"]) == 1,
           f"22 drape trials: {C.get('totals')}, the valid trials kept, the old configuration in the history")
    bad = d.call("config_apply", json.dumps({"repetitions": 0}))
    expect(bad.get("error") == "bad_value", "a configuration with no repetitions is refused")

    # 8. completeness of the phase
    R = d.call("check_phase", True)
    problems = {t["trial"]: t["problems"] for t in R["trials"] if t["problems"]}
    expect(not problems and not R["orphans"] and R["extra"] == [t4] and R["duplicates"] == [] and R["open"] == [],
           f"completeness: no problems, no stray folders, outside the matrix {R['extra']}")
    test_twin(d, work, t1)
    test_trial_fixes(d, work)

    # 9. live AR on the synthetic stream: on, board tracked, the other board, a photo stops it first
    s = d.call("live_state")
    expect(not s["running"] and s["registration"]["source"] == "demo" and s["intrinsics"]["source"] == "nominal",
           "live view: off at the start, demo registration, nominal intrinsics")
    d.call("live_start")
    e = d.wait_event("live", timeout=30)
    tracked = d.wait(lambda: (lambda x: x.get("tracked") and x.get("fps_ar", 0) > 0)(d.call("live_state")), 15)
    s = d.call("live_state")
    expect(e is not None and e["action"] == "started" and e.get("purpose") == "live" and tracked and s["fps_ar"] > 0
           and s["corners"] == s["board"]["corners"] and s["drawn_px"].get("face", 0) > 0,
           f"live view: started (an event of the live view's own), sheet tracked ({s.get('corners')} corners, RMS "
           f"{s.get('rms_px')} px), face drawn, {s.get('fps_ar')} fps")
    d.call("live_settings", json.dumps({"board": "patient", "mandible": True}))
    tracked = d.wait(lambda: (lambda x: x.get("tracked") and x["board"]["name"] == "patient")(d.call("live_state")), 15)
    s = d.call("live_state")
    expect(tracked and s["registration"]["demo"] == "design" and s["layers"]["mandible"],
           "live view: patient board tracked after the switch, bench design registration, mandible on")
    s = d.call("live_settings", json.dumps({"opacity": 0.55}))
    bad = d.call("live_settings", json.dumps({"opacity": 7}))
    expect(abs(s.get("opacity", -1) - 0.55) < 1e-9 and bad.get("opacity") == 1.0
           and ig.Prefs(d.backend.roots.prefs)["live"]["opacity"] == 1.0,
           "live view: the opacity of the virtual elements set through live_settings, kept, held between 0 and 1")
    d.call("live_settings", json.dumps({"opacity": 0.3}))
    # the slider moving: every value to the live view at once, quick, nothing saved until it is let go
    t0 = time.perf_counter()
    moved = [d.call("live_opacity", 0.3 + 0.004 * k) for k in range(100)]
    per_ms = (time.perf_counter() - t0) * 10
    expect(all(abs(a.get("opacity", -1) - (0.3 + 0.004 * k)) < 1e-9 for k, a in enumerate(moved))
           and abs(d.backend._live.settings()["opacity"] - 0.696) < 1e-9 and d.call("live_opacity", -2)["opacity"] == 0.0
           and ig.Prefs(d.backend.roots.prefs)["live"]["opacity"] == 0.3 and per_ms < 20,
           f"live view: the opacity slider's values taken one by one (live_opacity, {per_ms:.2f} ms each), held "
           "between 0 and 1, the preference left for the slider's release")
    d.call("live_settings", json.dumps({"opacity": 0.3}))
    folder = d.call("calib_folder_new", "check_photo")["folder"]
    n = len(d.events)
    d.call("calib_shoot", folder)
    stopped = d.wait(lambda: any(x["type"] == "live" and x.get("action") == "stopped" for x in d.events[n:]), 10)
    got = d.wait_event("calib_photo", timeout=20)
    said = next((x for x in d.events[n:] if x["type"] == "live" and x.get("action") == "stopped"), {})
    expect(stopped and got is not None and not d.call("live_state")["running"] and said.get("purpose") == "live",
           "live view: a photo from the phone stops it first, then the photo is taken")
    d.call("live_settings", json.dumps({"board": "sheet", "mandible": False}))
    # a stream that failed: its code reaches the page as live_error, never as the answer's own error (which the
    # page takes for a failed call: no picture, no reason, Start never offered again)
    ar = d.backend._live
    real_status = ar.status
    ar.status = lambda: {**real_status(), "error": "stream_closed", "stream_state": "failed"}
    try:
        answers = [d.call("live_state"), d.call("live_settings", json.dumps({"opacity": 0.3}))]
    finally:
        del ar.status
    expect(all("error" not in x and x.get("live_error") == "stream_closed" and "running" in x for x in answers),
           "live view: a failed stream said by its state (live_error), the answer itself not a failure")

    # 10. the link to the phone: cable or Wi-Fi, pairing, a drop and its return, the phone's own screen
    test_link(d)

    # 11. the calibration board (in the page, or physical) and the guided photos, checked one by one, calibrated
    test_board(d)
    if calibration:
        test_series(d)
        test_flow(d)
        test_judged(d)
    test_midnight(d)
    d.backend.shutdown()
    t0 = time.monotonic()
    frame = d.backend._frames.next_frame(0, 5.0)
    expect(frame[0] is None and time.monotonic() - t0 < 0.5,
           "app closed: the phone's page waiting for a frame is answered at once")


def test_twin(d, work, scanned):
    """The twin of the bench: its model made again by itself once the scene has rested for two minutes (a stand-in
    exporter and a clock moved by hand), a new version each time the scene changes and none when it does not, the
    page told to load it again; dataset and network said out of date by their fingerprints; the setting of the trial
    in progress; the marker's motion and the error at each target after the scans of a trial (simulated)."""
    import twin
    b = d.backend
    scene = fresh(work / "twin_scene") / "bench.blend"
    scene.write_bytes(b"bench 1")
    now = [time.time()]
    os.utime(scene, (now[0], now[0]))
    targets = [{"id": f"T{i}", "name": f"T{i}", "mandible": [10.0 * i, 80.0, -60.0 + 5 * i],
                "world": [10.0 * i, 0.0, -60.0]} for i in range(7)]

    def export(copy, out):
        out.mkdir(parents=True, exist_ok=True)
        body = copy.read_bytes()
        (out / "bench.glb").write_bytes(b"glTF" + body)
        (out / "bench.json").write_text(json.dumps({
            "phone_print": hashlib.sha256(body).hexdigest()[:16], "targets": targets, "board_centre": [0, -91, 9],
            "ball_centre": [0, -51, -19], "T_W_mandible": [[1, 0, 0, 0], [0, 0, 1, 96], [0, -1, 0, 17], [0, 0, 0, 1]]}),
            encoding="utf-8")
    real = b.twin
    b.twin = twin.Watcher(fresh(work / "twin_cache"), scene, blender=None, clock=lambda: now[0], export=export)
    b._twin_said = None
    n = len(d.events)
    b._twin_poll()
    now[0] += twin.QUIET_S - 5
    b._twin_poll()
    d.wait(lambda: False, 0.4)
    expect(not any(e["type"] == "twin" for e in d.events[n:]) and d.call("twin_state", "{}")["model"]["state"] == "none",
           "twin: the scene saved under two minutes ago is not exported")
    now[0] += 10
    b._twin_poll()
    up = d.wait(lambda: any(x["type"] == "twin" and x.get("action") == "updated" for x in d.events[n:]), 20)
    said = [x["action"] for x in d.events[n:] if x["type"] == "twin"]
    m1 = d.call("twin_state", "{}")["model"]
    expect(said[:1] == ["exporting"] and up and m1["state"] == "ready" and m1["version"] == 1
           and m1["url"].startswith("file:") and m1["url"].endswith(".glb"),
           f"twin: two minutes without a save, the model is made and the page told (version {m1.get('version')})")
    scene.write_bytes(b"bench 2")
    os.utime(scene, (now[0], now[0]))
    b._twin_poll()
    now[0] += 61
    b._twin_poll()
    busy = "twin_export" in b._busy
    now[0] += 61
    n = len(d.events)
    b._twin_poll()
    up = d.wait(lambda: any(x["type"] == "twin" and x.get("action") == "updated" for x in d.events[n:]), 20)
    m2 = d.call("twin_state", "{}")["model"]
    expect(not busy and up and m2["version"] == 2 and m2["url"] != m1["url"] and m2["phone_print"] != m1["phone_print"],
           "twin: the scene changed, waited for its rest again, a new version to load (2)")
    os.utime(scene, (now[0], now[0]))       # saved again with nothing changed
    b._twin_poll()
    now[0] += twin.QUIET_S + 1
    n = len(d.events)
    b._twin_poll()
    same = d.wait(lambda: any(x["type"] == "twin" and x.get("action") == "same" for x in d.events[n:]), 20)
    m3 = d.call("twin_state", "{}")["model"]
    expect(same and m3["version"] == 2 and m3["url"] == m2["url"], "twin: saved without a change, the same model")
    # dataset and network: made on this scene, on another one, before fingerprints
    ds, net = work / "twin_scene" / "index.json", work / "twin_scene" / "refiner.json"
    old = twin.DATASET_INDEX, twin.NETWORK_LOG
    twin.DATASET_INDEX, twin.NETWORK_LOG = ds, net
    try:
        ds.write_text(json.dumps({"scene_print": m2["phone_print"]}), encoding="utf-8")
        net.write_text(json.dumps({"scene_print": m1["phone_print"]}), encoding="utf-8")
        v1 = d.call("twin_state", "{}")["versions"]
        net.write_text(json.dumps({"scene_print": m2["phone_print"]}), encoding="utf-8")
        v2 = d.call("twin_state", "{}")["versions"]
        ds.write_text(json.dumps({"samples": 300}), encoding="utf-8")
        v3 = d.call("twin_state", "{}")["versions"]
    finally:
        twin.DATASET_INDEX, twin.NETWORK_LOG = old
    expect(v1["dataset"] == "current" and v1["network"] == "old" and v1["stale"] and not v2["stale"]
           and v3["dataset"] == "old" and v3["stale"],
           "twin: the network made on the scene before said out of date, then current; a dataset without a "
           "fingerprint out of date")
    # the trial in progress, and a trial after its scans
    s = d.call("twin_state", json.dumps({"mode": "trials"}))
    plan = d.call("trials_state")["trial"]
    t = s["trial"]
    first = d.call("trials_state")["steps"][0]["key"]          # the arm where the first step is taken
    arm = int(first.rsplit("_az", 1)[1]) if "_az" in first else 40
    expect(t and t["trial"] == plan["trial"] and t["holder"] == plan["holder"]
           and t["angle_deg"] == float(plan["angle_deg"]) and t["direction_deg"] == float(plan["direction_deg"])
           and t["arm_az"] == arm and abs(s["phone"]["aspect"] - 4080 / 3072) < 1e-3,
           f"twin: the setting of the trial to do ({t and t['trial']}: holder, hole, direction, arm at {arm}°)")
    keys = [x["key"] for x in d.call("trials_state")["steps"]]
    expect(s.get("steps") == keys and s.get("step") == 0 and "release" in keys
           and s.get("arm", {}).get("video_az") and s["arm"].get("photo_az") and s["arm"].get("video_deg_s"),
           f"twin: the steps of the trial to do, as the trial's page has them ({len(keys)}), at the first; the "
           f"arm's positions and its sweep")
    r = d.call("twin_state", json.dumps({"mode": "results", "trial": scanned}))
    sc = r["scan"] or {}
    D = sc.get("D") or []
    expect(r["trial"]["trial"] == scanned and sc.get("state") == "ready" and sc.get("simulated")
           and len(sc.get("tre", [])) == 7 and len(D) == 4 and sc.get("tre_max_mm", 0) > 0
           and any(x["trial"] == scanned for x in r["trials"]) and r["steps"][r["step"]].startswith("scan_"),
           f"twin: {scanned} after its scans, at its first scan, the marker's motion and the error at the 7 targets "
           f"(simulated, worst {sc.get('tre_max_mm')} mm)")
    r2 = d.call("twin_state", json.dumps({"mode": "results", "trial": scanned}))
    expect(r2["scan"]["D"] == D, "twin: the same trial, the same motion")
    b.twin = real
    b._twin_said = (real.current() or {}).get("version")


def test_flow(d):
    """The photos without waiting: never a countdown (none in the preferences, the page or the host; the shutter at
    once); a Take while a photo is taken follows it; the guide on the next pose at the shutter, the check later on the
    photo's small picture, its pose kept with it; Calibrate with photos on their way waits for them, and is refused
    only under the calibration's own least; the photo taken by itself when the simulated phone holds the guide's pose,
    once the gap is over, and not once the Photos step is left."""
    import re
    import numpy as np
    b = d.backend
    raw = b.roots.calib_raw
    had = {x.name for x in raw.iterdir()} if raw.is_dir() else set()
    phone = b.phone()
    phone.photo_px = (2040, 1536)
    d.call("calib_set", json.dumps({"mode": "photo", "plan": "quick", "kind": "calibration", "auto": False}))
    d.call("board_area", area(1500, 820))
    d.call("calib_series_new")
    # no countdown anywhere: nothing in the preferences that delays a photo, nothing in the page's Take or keys
    def keys(x, at=""):
        out = []
        for k, v in x.items():
            out.append(at + k)
            if isinstance(v, dict):
                out += keys(v, at + k + ".")
        return out
    page = (ig.APP / "ui" / "10-calibration.js").read_text(encoding="utf-8")
    take_src = page[page.index("  take: function () {"):page.index("  run: function () {")]
    keys_src = page[page.index("if (!CAL.keys) {"):page.index("$('calRetake').onclick")]
    expect(not [k for k in keys(ig.Prefs.DEFAULTS) if re.search(r"timer|countdown|delay", k, re.I)]
           and not re.search(r"setTimeout|setInterval|countdown", take_src + keys_src),
           "no countdown: nothing in the preferences, in the page's Take or its keys, that waits before a photo")
    # every method the backend calls on itself is there (a call to one not written yet made calib_state fail on
    # every answer: "Something went wrong" on the Photos step, 19/10); the photo taken by itself, failing, never
    # costs the page its state
    import ast
    tree = ast.parse(Path(ig.__file__).read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Backend")
    called = {x.func.attr for x in ast.walk(cls) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
              and isinstance(x.func.value, ast.Name) and x.func.value.id == "self"}
    missing = sorted(a for a in called if not hasattr(ig.Backend, a) and not hasattr(b, a))
    b._auto_target = lambda *a: 1 / 0
    try:
        st = d.call("calib_state")
    finally:
        del b._auto_target
    expect(not missing and "error" not in st and st.get("next"),
           f"the backend calls only methods it has ({missing or 'all there'}); a failing watcher leaves calib_state "
           f"answering")
    # Take, and Take again at once: the second follows the first; the shutter of each said before its check
    n0 = len(d.events)
    t0 = time.monotonic()
    a1 = d.call("calib_take")
    a2 = d.call("calib_take")
    first_shot = d.wait(lambda: any(x["type"] == "calib_photo" and x.get("action") == "shot" for x in d.events[n0:]), 20)
    t_shot = time.monotonic() - t0
    s_mid = d.call("calib_state")
    both = d.wait(lambda: sum(1 for x in d.events[n0:] if x["type"] == "calib_photo" and x.get("check")) >= 2, 90)
    ev = d.events[n0:]
    kinds = [("shot" if x.get("action") == "shot" else "check") for x in ev
             if x["type"] == "calib_photo" and (x.get("action") == "shot" or x.get("check"))]
    at = [i for i, k in enumerate(kinds) if k == "shot"], [i for i, k in enumerate(kinds) if k == "check"]
    s = d.call("calib_state")
    folder = raw / s["series"]["name"]
    checks = b._photo_checks(folder)
    poses = sorted(c.get("pose") for c in checks.values())
    expect(a1.get("ok") and a2.get("queued") and first_shot and t_shot < 1.5 and len(at[0]) == 2 and len(at[1]) == 2
           and at[0][0] < at[1][0] and at[0][1] < at[1][1]
           and both and poses == [0, 1] and s_mid["next"]["n"] >= 1 and s["next"]["n"] == 2 and s["count"] == 2,
           f"Take twice: the second taken right after the first ({a2}); the shutter {t_shot:.2f} s after Take, both "
           f"shutters before their checks ({' '.join(kinds)}); the guide on pose {s_mid['next']['n'] + 1} at once; "
           f"each photo kept with its pose ({poses})")
    shots = s["series"]["shots"]
    tim = [c.get("timing_s") or {} for c in checks.values()]
    expect(len(shots) == 2 and all(x["state"] == "kept" and x["thumb"].startswith("file:") for x in shots)
           and all(b._thumb_path(folder, x["name"]).is_file() for x in shots)
           and all(t.get("shutter") is not None and t["checked"] >= t["shutter"] for t in tim),
           f"the strip: each photo a small picture with its check (kept); timings kept with the checks {tim}")
    # a photo put aside frees its pose: the guide comes back to it, the one after keeps its own
    b._sim_scene = lambda: (lambda: np.full((1536, 2040, 3), 90, np.uint8))
    try:
        n1 = len(d.events)
        d.call("calib_take")
        d.wait(lambda: any(x["type"] == "calib_photo" and x.get("check") for x in d.events[n1:]), 60)
    finally:
        del b._sim_scene
    s = d.call("calib_state")
    expect(s["next"]["n"] == 2 and s["series"]["shots"][-1]["state"] == "retake" and s["count"] == 2,
           f"a photo without the board: retake on its small picture, its pose asked again ({s['next']['n'] + 1})")
    # Calibrate always pressable from the calibration's own least: with photos on their way it waits for them
    while d.call("calib_state")["count"] < calibrate_charuco.MIN_VIEWS - 1:
        n1 = len(d.events)
        d.call("calib_take")
        d.wait(lambda: any(x["type"] == "calib_photo" and x.get("check") for x in d.events[n1:]), 60)
    n1 = len(d.events)
    d.call("calib_take")
    d.wait(lambda: any(x["type"] == "calib_photo" and x.get("action") == "shot" for x in d.events[n1:]), 20)
    r = d.call("calib_run", "[]")
    s = d.call("calib_state")
    e = d.wait_event("calib", timeout=600)
    expect(r.get("waiting") and s["calib_waiting"] and e and e.get("action") == "calibrate"
           and not d.call("calib_state")["calib_waiting"],
           f"Calibrate with a photo still on its way: waits for its check, then calibrates by itself "
           f"({s['count']} photos, fewer than the {s['limits']['min']} advised; {e and ('ok' if e.get('ok') else e.get('error'))})")
    # the photo taken by itself: the simulated phone shows the board at the guide's pose, held still
    d.call("calib_set", json.dumps({"auto": True}))
    w = d.call("calib_watch", True)
    b._auto.watch.gap_s, b._auto.watch.hold_s = 1.5, 0.2
    n2 = len(d.events)
    got = d.wait(lambda: sum(1 for x in d.events[n2:] if x["type"] == "calib_photo" and x.get("action") == "shot"
                             and x.get("auto")) >= 2, 60)
    autos = [x for x in d.events[n2:] if x["type"] == "calib_photo" and x.get("action") == "shot"]
    states = {x["state"] for x in d.events[n2:] if x["type"] == "calib_auto"}
    d.call("calib_watch", False)
    d.wait(lambda: not b._shots and not b._checking(), 60)
    n3 = len(d.events)
    d.wait(lambda: False, 3)
    later = [x for x in d.events[n3:] if x["type"] == "calib_photo" and x.get("action") == "shot"]
    expect(w.get("ok") and got and len(autos) >= 2 and all(x.get("auto") for x in autos) and "hold" in states
           and b._auto is None and not later,
           f"the photo taken by itself: {len(autos)} photos once the phone held the guide's pose ({sorted(states)}); "
           f"none once the Photos step is left")
    # the phone's video: never a photo taken by itself, nor a film started by one, while filming or not
    d.call("calib_set", json.dumps({"mode": "video", "auto": True}))
    d.call("calib_watch", True)
    in_video = b._auto is None
    b._auto_fire()
    film = d.call("calib_state").get("film") or {}
    d.call("calib_watch", False)
    d.call("calib_set", json.dumps({"mode": "photo"}))
    expect(in_video and not film.get("recording") and not film.get("starting") and "phone" not in b._busy,
           "the phone's video: the photo taken by itself does not watch, and its trigger starts no film")
    d.call("calib_set", json.dumps({"auto": True}))
    phone.photo_px = b.run().config.photo_px
    for x in raw.iterdir():
        if x.name not in had:
            shutil.rmtree(x)
    b._pending_cal.clear()


def test_judged(d):
    """The calibration judges its photos: never good with a board not flat, a phone that moved or automatic settings
    (the worst first; a corner never reached only advice); each photo checked against the camera's settings of the
    first one kept; every pose taken but a corner of the photo without the board: that corner's pose once more, past
    the plan; the camera app left in its mode for a calibration photo (no intent to it)."""
    import calib_guide
    b = d.backend
    raw = b.roots.calib_raw
    had = {x.name for x in raw.iterdir()} if raw.is_dir() else set()
    d.call("calib_set", json.dumps({"mode": "photo", "plan": "quick", "kind": "calibration", "auto": False}))
    base = {"K": [[3000.0, 0, 2039.5], [0, 3000.0, 1535.5], [0, 0, 1]], "dist": [0.0] * 5, "image_size": [4080, 3072],
            "rms_px": 0.3, "n_views": 15, "std": {k: 0.4 for k in ("fx", "fy", "cx", "cy")},
            "per_view_rms_px": {"a.jpg": 0.3}, "plan": "quick", "mode": "photo"}
    said = {}
    for tag, issues in (("clean", []), ("corner", ["corners_missing"]),
                        ("bad", ["corners_missing", "camera_settings", "board_not_flat", "phone_not_still"])):
        b._pending_cal["photo"] = {**base, "issues": issues, "shape": {"flat": "board_not_flat" not in issues, "sag_mm": 2.6},
                                   "centre": {"still": "phone_not_still" not in issues, "spread_px": 25.0},
                                   "coverage": {"missing_corners": ["top_right"] if issues else [], "reach": 0.7},
                                   "settings": {"a.jpg": ["exposure_auto"]} if "camera_settings" in issues else {}}
        said[tag] = d.call("calib_state")["result"]
    bad = said["bad"]
    expect(said["clean"]["verdict"] == "good" and said["corner"]["verdict"] == "good" and not said["corner"]["not_good"]
           and said["corner"]["missing_corners"] == ["top_right"] and bad["verdict"] == "poor"
           and bad["issues"] == list(ig.ISSUE_ORDER) and bad["sag_mm"] == 2.6 and bad["settings"] == ["exposure_auto"]
           and not ig.Backend._good({**base, "issues": ["phone_not_still"]}) and ig.Backend._good(base),
           f"the verdict: good with a corner never reached (advice), not good with {bad['not_good']}, the worst first "
           f"({bad['issues']})")
    b._pending_cal.clear()
    # a photo checked against the camera's settings of the first photo kept in its series
    folder = raw / "20990101_judged_photo"
    folder.mkdir(parents=True)
    for k, name in enumerate(("a.jpg", "b.jpg")):
        (folder / name).write_bytes(b"x")
        os.utime(folder / name, (1e9 + k, 1e9 + k))
    first = {"known": True, "program": 1, "white_balance": 1, "iso": 100.0, "exposure_s": 0.02, "problems": []}
    b._write_checks(folder, {"a.jpg": {"ok": True, "why": "", "size": [4080, 3072], "settings": first, "seq": 1}})
    got, real = {}, calibrate_charuco.check_photo

    def fake(path, board, size=None, first=None):
        got.update(name=Path(path).name, size=size, first=first)
        return {"ok": True, "why": "", "corners": 50, "of": 70, "blur_px": 1.0, "size": [4080, 3072], "points": [],
                "coverage": None, "settings": {"known": False, "problems": []}}
    calibrate_charuco.check_photo = fake
    try:
        b._queue_checks(folder, [folder / "b.jpg"])
        d.wait(lambda: "b.jpg" in b._photo_checks(folder), 20)
    finally:
        calibrate_charuco.check_photo = real
    expect(got.get("name") == "b.jpg" and got.get("first") == first and got.get("size") == [4080, 3072],
           f"a photo checked against the settings of the first one kept ({got.get('first')})")
    # every pose of the plan taken, the board never in the top right corner: that corner's pose again, past the plan
    d.call("calib_series_new")
    plan = calib_guide.PLANS["quick"]
    b._poses_taken = lambda *a: set(range(len(plan)))
    try:
        b._missing_corners = lambda *a: ["top_right"]
        s = d.call("calib_state")
        b._missing_corners = lambda *a: []
        done = d.call("calib_state")
    finally:
        del b._poses_taken
        b.__dict__.pop("_missing_corners", None)
    i = next(j for j, p in enumerate(plan) if p.get("corner") == "top_right")
    expect(s["next"] and s["next"]["extra"] == "top_right" and s["next"]["n"] == i and done["next"] is None,
           f"every pose taken, no board yet in the top right corner: its pose ({i + 1}) once more; none once reached")
    # a calibration photo leaves the camera app in its mode (Pro): no intent to it
    phone, seen = b.phone(), {}
    real_take = phone.take

    def take(timeout=None, pressed=None, keep_mode=False):
        seen["keep_mode"] = keep_mode
        return real_take(pressed=pressed, keep_mode=keep_mode)
    phone.take = take
    try:
        n = len(d.events)
        b._shoot_into(folder, check=False)
        d.wait(lambda: any(x["type"] == "calib_photo" and x.get("name") for x in d.events[n:]), 30)
    finally:
        del phone.take
    expect(seen.get("keep_mode") is True, f"a calibration photo: the camera app kept in its mode ({seen})")
    for x in raw.iterdir():
        if x.name not in had:
            shutil.rmtree(x)


def test_midnight(d):
    """A series of photos under way at midnight goes on the next day; one left hours ago is a series of its own."""
    from datetime import date, timedelta
    b = d.backend
    d.call("calib_set", json.dumps({"mode": "photo", "kind": "calibration"}))
    tag = b._series_tag()
    today = f"{date.today():%Y%m%d}"
    for x in list(b.roots.calib_raw.iterdir()) if b.roots.calib_raw.is_dir() else []:
        if x.name.startswith(today) and x.name.split("_", 1)[-1] == tag:
            expect(False, f"no series of today before this check ({x.name})")
            return
    yesterday = b.roots.calib_raw / f"{date.today() - timedelta(days=1):%Y%m%d}_{tag}"
    fresh(yesterday)
    photo = yesterday / "IMG_0001.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    goes_on = (b._series() or {}).get("name")
    old = time.time() - ig.SERIES_KEEP_S - 60
    os.utime(photo, (old, old))
    left = b._series()
    shutil.rmtree(yesterday)
    expect(goes_on == yesterday.name and left is None,
           f"a series under way at midnight goes on ({goes_on}); left {ig.SERIES_KEEP_S // 3600} h ago, a new one "
           f"starts")


def png_image(data_url):
    """The grey image of a PNG data URL, as the page draws it."""
    import base64
    import cv2
    import numpy as np
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)


def board_png(d, answer, timeout=20):
    """The image of the board a board_area answer said would follow (a board event with its key, made in a worker),
    as the page draws it; None when none came."""
    if answer.get("png"):
        return png_image(answer["png"])
    if not answer.get("pending"):
        return None

    def hit():
        return next((e for e in reversed(d.events) if e.get("type") == "board" and e.get("action") == "image"
                     and e.get("key") == answer["key"]), None)
    d.wait(lambda: hit() is not None, timeout)
    e = hit()
    return png_image(e["png"]) if e else None


def area(w, h, have=""):
    """An area of the page for the board, as ui/15-board.js measures it (offscreen: one device pixel per pixel)."""
    return json.dumps({"w": w, "h": h, "dpr": 1, "x": 0, "y": 0, "iw": w, "have": have})


def test_board(d):
    """The calibration board, drawn by the page in the work area of the step that shows it: the first question, the
    layout from the area's pixels and the pitch of the monitor the window is on, the image sent pixel for pixel and
    only when it changes, laid out again when the area changes and taken at once by the live view and the
    calibration, held while photos are of it, too small said as such, the ruler check; the physical boards: the
    3D-printed plate and the A4 sheet with their caliper readings, mean, spread and what is wrong with them."""
    import numpy as np
    s = d.call("board_state")
    expect(s["kind"] == "screen" and not s["chosen"] and not s["shown"] and s["layout"] is None,
           "board: the first question not answered yet, nothing on show")
    # the page shows the board in an area of 1500 x 820 device pixels
    a = d.call("board_area", area(1500, 820))
    st, lay = a["state"], a["state"]["layout"]
    mon = st["mon"]
    expect(a.get("pending") and "png" not in a,
           "the board's image is made in a worker: the answer has the layout, the image follows as an event")
    img = board_png(d, a)
    ref = screen_board.render(d.backend._lay)
    expect(a["shown"] and lay is not None and img is not None and img.shape == (820, 1500)
           and np.array_equal(img, ref) and abs(lay["square_mm"] - lay["square_px"] * lay["pitch_mm"]) < 1e-9
           and lay["marker_px"] % 7 == 0 and st["active"]["kind"] == "screen" and st["exact"],
           f"board on this monitor, in the page: {lay and lay['squares']} squares of {lay and lay['square_px']} px = "
           f"{lay and round(lay['square_mm'], 3)} mm at {mon and mon['pitch_mm']:.5f} mm ({mon and mon['source']}), "
           f"the image as large as the area and the same pixel for pixel")
    again = d.call("board_area", area(1500, 820, a["key"]))
    expect(again["key"] == a["key"] and "png" not in again and not again.get("pending"),
           "the same area again: the same board, no image sent")
    # the area changes: laid out again, and the live view and the calibration take it at once
    b = d.call("board_area", area(1100, 820, a["key"]))
    live, cal = d.call("live_state"), d.call("calib_state")
    lay2 = b["state"]["layout"]
    img2 = board_png(d, b)
    expect(img2 is not None and img2.shape == (820, 1100) and lay2["squares"] != lay["squares"] and live["board"]["squares"] == lay2["squares"]
           and abs(live["board"]["square_mm"] - lay2["square_mm"]) < 1e-9 and cal["board_ui"]["active"]["squares"]
           == lay2["squares"] and live["hide_board"],
           f"a narrower area: laid out again {lay['squares']} -> {lay2['squares']}, taken at once by the live view "
           f"(the board hidden in its picture) and the calibration")
    tiny = d.call("board_area", area(300, 260, b["key"]))
    expect(tiny["problem"] == "too_small" and tiny["state"]["layout"] is None and tiny["key"] == "",
           "an area too small for any board: said, no board")
    off = d.call("board_area", area(0, 0))
    refused = d.call("calib_take")
    expect(not off["shown"] and refused.get("error") == "board_hidden",
           "no step shows the board: a photo is refused until it is on show")
    # the ruler check: the pitch of this monitor from the length read, kept; a misreading refused; forgotten
    d.call("board_area", area(1500, 820))
    width = d.backend._lay["width_mm"]
    r = d.call("board_set", json.dumps({"ruler_mm": round(width * 1.004, 1)}))
    bad = d.call("board_set", json.dumps({"ruler_mm": round(width * 1.09, 1)}))
    kept = ig.Prefs(d.backend.roots.prefs)["monitors"].get(mon["key"], {})
    expect(r["mon"]["source"] == "ruler" and abs(r["mon"]["pitch_mm"] / mon["pitch_mm"] - round(width * 1.004, 1) / width)
           < 1e-9 and bad.get("error") == "ruler_bad" and kept.get("measured_mm") == round(width * 1.004, 1)
           and abs(r["layout"]["pitch_mm"] - r["mon"]["pitch_mm"]) < 1e-12,
           f"ruler check: {round(width * 1.004, 1)} mm read for {width:.1f} mm, pitch {r['mon']['pitch_mm']:.5f} mm "
           f"kept for this monitor and the board laid out again with it; 9 % off refused")
    f = d.call("board_set", json.dumps({"forget": True}))
    expect(f["mon"]["source"] == mon["source"], "ruler check forgotten")
    # the ruler check with the page at another scale than the screen (125 %): the monitor's own pitch is kept, so
    # the next layout is right; a window with no client area (minimised) lays the board out without failing
    b = d.backend
    b._page_scale = lambda c: (1.25, True)
    try:
        scaled = d.call("board_area", area(1500, 820))
        lay_s = b._lay
        read = round(lay_s["width_mm"] * 1.004, 1)
        r_s = d.call("board_set", json.dumps({"ruler_mm": read}))
        kept_s = ig.Prefs(b.roots.prefs)["monitors"].get(mon["key"], {})
        page_pitch = read / (lay_s["squares"][0] * lay_s["square_px"])
        scale_ok = (not scaled["exact"] and abs(kept_s.get("pitch_mm", 0) - page_pitch / 1.25) < 1e-12
                    and abs(r_s["layout"]["pitch_mm"] - kept_s["pitch_mm"] * 1.25) < 1e-12)
        d.call("board_set", json.dumps({"forget": True}))
        b._page_scale = lambda c: (0.0, True)
        zero = d.call("board_area", area(1500, 820))
    finally:
        del b._page_scale
    d.call("board_area", area(1500, 820))
    expect(scale_ok and "error" not in zero and zero["exact"] and zero["state"]["layout"] is not None,
           f"ruler check with the page at 125 %: the monitor's pitch kept per screen pixel "
           f"({kept_s.get('pitch_mm', 0):.5f} mm), the board laid out again right; no client area: laid out at 1")
    # caliper readings that are not lengths (a preferences file edited by hand): refused, the nominal board used
    kept_board = dict(b.prefs["board"])
    b.prefs.set("board", {**kept_board, "sheet": {"long": [float("nan"), 264.2, 264.4]}})
    odd = d.call("board_state")
    b.prefs.set("board", kept_board)
    try:
        b._sheet_reading([float("inf")])
        refused = False
    except ValueError:
        refused = True
    expect("error" not in odd and odd["physical"]["sheet"]["long"] is None and not odd["physical"]["sheet"]["spec"]
           ["measured"] and refused,
           "A4 sheet: readings that are not finite refused, the nominal board used, the board state still answers")
    loose = d.work / "loose_photos"
    fresh(loose)
    (loose / "imported.jpg").write_bytes(b"\xff\xd8\xff")
    b._folder_board(loose)
    expect(not (loose / calibrate_charuco.BOARD_FILE).exists(),
           "photos imported into a folder without board.json: no board made up for them")
    # the 3D-printed plate: three readings, the mean and the spread, the square from them
    p = d.call("board_set", json.dumps({"kind": "plate", "plate_long": [220.05, 220.02, 220.08]}))
    info, a_ = p["physical"]["plate"], p["active"]
    live = d.call("live_state")
    expect(p["kind"] == "plate" and p["chosen"] and abs(info["long"]["mean_mm"] - 220.05) < 1e-9
           and abs(info["long"]["spread_mm"] - 0.06) < 1e-9 and not info["long"]["problems"]
           and a_["kind"] == "plate" and a_["squares"] == [11, 8] and a_["measured"]
           and abs(a_["square_mm"] - 220.05 / 11) < 1e-6 and abs(live["board"]["square_mm"] - a_["square_mm"]) < 1e-9
           and not live["hide_board"] and info["model"],
           f"3D-printed plate: 3 readings, mean {info['long']['mean_mm']} mm, spread {info['long']['spread_mm']} mm, "
           f"square {a_['square_mm']:.4f} mm, used by the live view (nothing hidden); its 3D model there")
    w = d.call("board_set", json.dumps({"plate_long": [220.05, 220.40, 220.08]}))
    s2 = d.call("board_set", json.dumps({"plate_long": [220.05, 220.02, 220.08], "plate_short": [160.03, 160.05, 160.01]}))
    expect("spread" in w["physical"]["plate"]["long"]["problems"] and not w["active"]["measured"]
           and abs(w["active"]["square_mm"] - 20.0) < 1e-9 and s2["active"].get("anisotropy_ok") is True,
           "plate: readings 0.38 mm apart said as such and the nominal square kept; the short side gives X against Y")
    sh = d.call("board_set", json.dumps({"kind": "sheet", "sheet_long": [264.3, 264.2, 264.4]}))
    expect(sh["active"]["kind"] == "sheet" and abs(sh["active"]["square_mm"] - 264.3 / 11) < 1e-3
           and sh["active"]["squares"] == [11, 8],
           f"A4 sheet: 3 readings of the top row, square {sh['active']['square_mm']:.4f} mm")
    d.call("board_set", json.dumps({"kind": "screen", "plate_short": []}))


def test_series(d):
    """The photos of a calibration as the Photos step takes them, the simulated phone photographing the board in use
    at the pose of the plan (calib_guide.photo): a series per day and mode made at the first photo; each photo
    checked as it comes, kept, or put aside with the reason (blurred, no board, the phone turned) and its pose asked
    again; a board changed after photos starts a new series; the plan's least and most; the calibration with its
    errors per photo and over the board, a photo left out by hand and the calibration again, saved; the session
    check; the streaming mode on frames of the live stream."""
    import cv2
    import numpy as np
    import calib_guide
    b = d.backend
    raw = b.roots.calib_raw
    had = {x.name for x in raw.iterdir()} if raw.is_dir() else set()
    phone = b.phone()
    phone.photo_px = (2040, 1536)        # half the phone's photo: the same checks, faster
    d.call("calib_set", json.dumps({"mode": "photo", "plan": "quick", "kind": "calibration"}))
    d.call("board_area", area(1500, 820))
    s = d.call("calib_state")
    g = s["next"]
    expect(s["count"] == 0 and s["limits"] == {"plan": "quick", "n": 15, "min": 12, "max": 15} and g and g["n"] == 0
           and g["outline"] and len(g["cells"]) > 10,
           f"plan quick: 15 poses, at least 12 photos; the guide of the first: {g and g['view']}, {g and g['dist_cm']} cm")

    def take():
        n = len(d.events)
        a = d.call("calib_take")
        e = d.wait(lambda: any(x["type"] == "calib_photo" and x.get("check") for x in d.events[n:]), 60)
        return a, next((x for x in d.events[n:] if x["type"] == "calib_photo" and x.get("check")), None) if e else None
    a, e = take()
    s = d.call("calib_state")
    first = s["series"]
    expect(e and e["check"]["ok"] and s["count"] == 1 and first["name"].endswith("_photo") and first["same"]
           and s["next"]["n"] == 1 and (raw / first["name"] / ig.PHOTO_CHECKS).is_file(),
           f"first photo: checked as it came and kept ({e and e['check']['corners']} corners, blur "
           f"{e and e['check']['blur_px']} px), in {first['name']}; the guide moves to the second pose")
    # two photos arriving while a check runs: both checked, one after the other (none left unchecked)
    q = d.work / "queue_check"
    fresh(q)
    shutil.copy2(raw / first["name"] / calibrate_charuco.BOARD_FILE, q / calibrate_charuco.BOARD_FILE)
    src = next(p for p in (raw / first["name"]).iterdir() if p.suffix.lower() in ig.IMAGE_SUFFIXES)
    for name in ("one", "two"):
        shutil.copy2(src, q / f"{name}{src.suffix}")
    b._arrived(q, q / f"one{src.suffix}")
    b._arrived(q, q / f"two{src.suffix}")
    queued = b._checking() and len(b._check_queue) == 1
    both = d.wait(lambda: {f"one{src.suffix}", f"two{src.suffix}"} <= set(b._photo_checks(q)), 60)
    checks = b._photo_checks(q)
    expect(queued and both and all(c["ok"] for c in checks.values()) and not b._checking(),
           "two photos at once: the second waits for the first's check, then is checked too")
    n = len(d.events)
    b._queue_checks(q, [q / f"gone{src.suffix}"])
    gone = next((x for x in d.events[n:] if x["type"] == "acquire_failed"), None)
    expect(gone is not None and gone.get("key") == "calibration" and gone.get("error") == "not_found"
           and not b._checking() and not b._check_queue,
           "a photo gone before its check: said, so the page ends the work it started; the queue goes on")
    # photos that fail: blurred, without the board, the phone held the other way
    for why, scene in (("blur", lambda: cv2.cvtColor(cv2.GaussianBlur(calib_guide.photo(
            "quick", 1, b._board()["active"], "screen", "photo", b._lay, size=(2040, 1536)), (0, 0), 4), cv2.COLOR_GRAY2BGR)),
            ("none", lambda: np.full((1536, 2040, 3), 90, np.uint8)),
            ("size", lambda: cv2.cvtColor(calib_guide.photo("quick", 1, b._board()["active"], "screen", "photo", b._lay,
                                                           size=(1536, 2040)), cv2.COLOR_GRAY2BGR))):
        b._sim_scene = (lambda f: (lambda: f))(scene)
        a, e = take()
        s = d.call("calib_state")
        aside = sorted((raw / first["name"] / ig.REJECTED).iterdir()) if (raw / first["name"] / ig.REJECTED).is_dir() else []
        expect(e and not e["check"]["ok"] and e["check"]["why"] == why and s["count"] == 1 and s["next"]["n"] == 1
               and any(x.name == e["name"] for x in aside) and s["series"]["last"]["why"] == why,
               f"a photo {('blurred', 'without the board', 'of the phone held the other way')[('blur', 'none', 'size').index(why)]}: "
               f"retake ({e and e['check']['why']}, blur {e and e['check']['blur_px']} px), put aside in rejected/, "
               f"the same pose asked again")
    del b._sim_scene
    # the rest of the plan: enough at 12, full at 15, past it a photo for each corner of the photo the board has not
    # reached yet (at most ig.EXTRA_MAX), then full
    for _ in range(14):
        take()
    extras = 0
    while extras < ig.EXTRA_MAX and (d.call("calib_state")["next"] or {}).get("extra"):
        take()
        extras += 1
    s = d.call("calib_state")
    n_full = s["count"]
    full = d.call("calib_take")
    expect(n_full == 15 + extras and s["next"] is None and full.get("error") == "calib_full",
           f"the plan full at 15 photos and {extras} for the corners of the photo ({s['count']}): no more taken")
    # a board changed after photos: the series is not of it, the next photo starts a new one
    d.call("board_area", area(1100, 820))
    s = d.call("calib_state")
    expect(s["series"]["n"] == n_full and not s["series"]["same"] and s["count"] == 0,
           "board laid out anew after the photos: the series is not of it, the count starts again")
    d.call("board_area", area(1500, 820))
    s = d.call("calib_state")
    expect(s["series"]["same"] and s["count"] == n_full,
           "the board of the photos back (held while it fits): the series again")
    # the photos counted by their checks, not by the files of the folder: one put there by hand is checked before it
    # counts (no board in it: put aside), one whose check failed but that stayed (its move failed) neither counts
    # nor goes into the calibration
    sd = raw / s["series"]["name"]
    kept_one = next(k for k, c in b._photo_checks(sd).items() if c.get("ok") and (sd / k).is_file())
    cv2.imwrite(str(sd / "by_hand.jpg"), np.full((1536, 2040), 90, np.uint8))
    shutil.copy2(sd / kept_one, sd / "stuck.jpg")
    ch = b._photo_checks(sd)
    ch["stuck.jpg"] = {**ch[kept_one], "ok": False, "why": "blur", "seq": 0}
    b._write_checks(sd, ch)
    s1 = d.call("calib_state")
    checked = d.wait(lambda: "by_hand.jpg" in b._photo_checks(sd) and not b._checking(), 60)
    s2 = d.call("calib_state")
    expect(checked and s1["count"] == n_full and s2["count"] == n_full and s2["series"]["n"] == n_full
           and not b._photo_checks(sd)["by_hand.jpg"]["ok"] and (sd / ig.REJECTED / "by_hand.jpg").is_file()
           and (sd / "stuck.jpg").is_file(),
           f"photos counted by their checks: one put there by hand checked before it counts (put aside: "
           f"{b._photo_checks(sd).get('by_hand.jpg', {}).get('why')}), one failed but left there not counted "
           f"({s1['count']}, {s2['count']} of {n_full})")
    # the calibration: errors per photo and over the board, the intrinsics in plain units
    d.call("calib_run", "[]")
    e = d.wait_event("calib", timeout=600)
    s = d.call("calib_state")
    r = s["result"]
    expect(r and "stuck.jpg" not in {p["name"] for p in r["photos"]}
           and (b._pending_cal["photo"].get("skipped") or {}).get("stuck.jpg") == "check",
           "a photo its check refused is left out of the calibration even when it stayed in the folder")
    expect(e and e["ok"] and r and not r["saved"] and r["verdict"] == "good" and r["n_views"] >= 14
           and len(r["photos"]) == n_full
           and r["residuals"]["corners"] and r["grid"] and abs(r["plain"]["fx"] / (3071.4 * 2040 / 4080) - 1) < 0.003
           and r["focal_ok"],
           f"calibration: {r and r['verdict']}, RMS {r and r['rms_px']} px on {r and r['n_views']} of {n_full} photos, "
           f"focal {r and round(r['plain']['fx'], 1)} px (true {3071.4 * 2040 / 4080:.1f}, +/- "
           f"{r and r['plain']['f_std'] is not None and round(r['plain']['f_std'], 2)} px), field of view "
           f"{r and round(r['plain']['fov_x'], 1)} deg, lens bend {r and round(r['plain']['bend_px'], 1)} px, "
           f"{r and len(r['residuals']['corners'])} corners over the board")
    # a calibration whose focal uncertainty is not known (std NaN or missing): judged, never good, the state answers
    pending = b._pending_cal["photo"]
    unknown = []
    for std in ({"fx": float("nan"), "fy": 1.0}, None):
        b._pending_cal["photo"] = {**pending, "std": std}
        u = d.call("calib_state").get("result")
        unknown.append(bool(u) and u["focal_ok"] is False and u["verdict"] != "good" and u["plain"]["f_std"] is None)
    b._pending_cal["photo"] = pending
    expect(all(unknown), f"a calibration without its focal uncertainty: answered, focal not ok, not good ({unknown})")
    acc = d.call("calib_set", json.dumps({"plan": "accurate"}))["result"]
    d.call("calib_set", json.dumps({"plan": "quick"}))
    expect(acc and acc["verdict"] == "good" and not acc["few"],
           f"the accurate plan chosen after calibrating: the verdict stays that of the quick plan the photos were "
           f"taken with ({acc and acc['verdict']})")
    worst = max(r["photos"], key=lambda p: p["err"] or 0)["name"]
    d.call("calib_run", json.dumps([worst]))
    d.wait_event("calib", timeout=600)
    r2 = d.call("calib_state")["result"]
    st = {p["name"]: p["state"] for p in r2["photos"]}
    expect(st[worst] == "excluded" and r2["n_views"] == r["n_views"] - (0 if worst in r.get("dropped", []) else 1)
           and len(r2["photos"]) == n_full,
           f"the photo with the largest error left out by hand, calibrated again: {r2['n_views']} photos, RMS "
           f"{r2['rms_px']} px; the photo still shown, as left out")
    sv = d.call("calib_save")
    saved = json.loads((b.roots.calib_out / "intrinsics_photo.json").read_text(encoding="utf-8"))
    expect(sv["result"]["saved"] and saved["excluded"] == [worst] and "residuals" in saved
           and sv["modes"]["photo"]["stored"] is not None,
           "saved as intrinsics_photo.json, with the photo left out and the errors over the board")
    # the session check: its own plan of 10, then Check against the stored calibration
    c = d.call("calib_set", json.dumps({"kind": "check"}))
    expect(c["plan"] == "check" and c["limits"]["max"] == 10 and c["count"] == 0,
           "session check: a plan of its own, 10 photos")
    for _ in range(8):
        take()
    d.call("calib_check")
    e = d.wait_event("calib", timeout=600)
    checks = trials.read_csv(b.roots.calib_out / "session_checks.csv")
    expect(e and e["action"] == "check" and e["ok"] and checks and checks[-1]["ok"] == "1",
           "session check on 8 photos: the stored calibration still valid, logged in session_checks.csv")
    d.call("calib_set", json.dumps({"kind": "calibration"}))
    # the streaming mode: frames of the live stream (synthetic here), checked like photos
    d.call("calib_set", json.dumps({"mode": "stream"}))
    refused = d.call("calib_take")
    n0 = len(d.events)
    d.call("calib_stream", True)
    up = d.wait(lambda: (lambda x: x["live"] and x["live"]["running"])(d.call("calib_state")), 30)
    d.wait(lambda: any(x["type"] == "live" and x.get("action") == "started" for x in d.events[n0:]), 10)
    began = next((x for x in d.events[n0:] if x["type"] == "live" and x.get("action") == "started"), {})
    d.wait(lambda: d.call("live_state").get("tracked"), 15)
    a, e = take()
    s = d.call("calib_state")
    frame = next(iter(sorted((raw / s["series"]["name"]).glob("frame_*.png"))), None) if s["series"] else None
    shape = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE).shape if frame else None
    live = d.call("live_state")
    expect(refused.get("error") == "stream_off" and up and e and e["check"]["ok"] and s["series"]["name"].endswith("_stream")
           and shape == (1440, 1920) and not live["layers"]["face"] and live["hide_board"]
           and began.get("purpose") == "calibration",
           f"streaming mode: refused until the live video runs (its events those of the calibration), then a frame "
           f"of the stream kept "
           f"({e and e['check']['corners']} corners), saved whole as PNG {shape}, the picture plain (no face) with the "
           f"board on this monitor hidden in it")
    d.call("calib_stream", False)
    s = d.call("calib_state")
    expect(not s["live"]["running"], "the Photos step left: the live video stops")
    test_film(d, raw)
    d.call("calib_set", json.dumps({"mode": "photo"}))
    phone.photo_px = b.run().config.photo_px
    for x in raw.iterdir():          # the series of this test go
        if x.name not in had:
            shutil.rmtree(x)
    b._pending_cal.clear()
    (b.roots.calib_out / "intrinsics_photo.json").unlink(missing_ok=True)
    (b.roots.calib_out / "intrinsics_video.json").unlink(missing_ok=True)


def test_film(d, raw):
    """The video mode: a film of the camera app, the simulated phone filming the board through the poses of the plan.
    Film and Stop on the same button, the camera busy while it films; the frames chosen by themselves (still, sharp,
    of different poses) and calibrated; saved as the video mode's own calibration, the photos' left as it was; a film
    added by hand goes the same way, in a series of its own."""
    import calib_video
    b = d.backend
    photo_sha = sha(b.roots.calib_out / "intrinsics_photo.json")
    d.call("calib_set", json.dumps({"mode": "video", "plan": "quick"}))
    n0 = len(d.events)
    started = d.call("calib_take")
    rec = d.wait(lambda: any(x["type"] == "calib_film" and x.get("action") == "recording" for x in d.events[n0:]), 20)
    s = d.call("calib_state")
    other = d.call("calib_stream", True)        # the camera films: the live video waits
    d.call("calib_stream", False)
    stopped = d.call("calib_take")

    def ended(n):
        return any((x["type"] == "calib" and x.get("mode") == "video") or x["type"] == "acquire_failed"
                   for x in d.events[n:])
    done = d.wait(lambda: ended(n0), 300)
    v = d.call("calib_state")
    folder = raw / v["series"]["name"] if v["series"] else None
    films = sorted((folder / calib_video.FILM_DIR).glob("*.mp4")) if folder else []
    r = v["result"] or {}
    chosen = next((x for x in d.events[n0:] if x["type"] == "calib_film" and x.get("action") == "chosen"), {})
    failed = next((x for x in d.events[n0:] if x["type"] == "acquire_failed"), None)
    expect(started.get("film") == "starting" and rec and s["film"]["recording"] and other.get("error") == "busy"
           and stopped.get("film") == "stopping" and done and not failed and folder and folder.name.endswith("_video")
           and len(films) == 1 and v["count"] == chosen.get("n") and v["count"] >= 12 and v["next"] is None
           and r.get("verdict") == "good" and r.get("cropped") is False,
           f"video mode: Film, the camera busy while it films, Stop; {chosen.get('n')} frames chosen by themselves "
           f"from {chosen.get('seconds')} s ({chosen.get('counts')}), calibrated: {r.get('verdict')}, RMS "
           f"{r.get('rms_px')} px, field {r.get('narrower')} narrower than the photos{'; ' + str(failed) if failed else ''}")
    d.call("calib_save")
    vid = json.loads((b.roots.calib_out / "intrinsics_video.json").read_text(encoding="utf-8"))
    expect(vid.get("mode") == "video" and vid["image_size"] == list(ig.FILM_SIM_SIZE) and films
           and (vid.get("film") or {}).get("video") == films[0].name
           and sha(b.roots.calib_out / "intrinsics_photo.json") == photo_sha,
           "saved as intrinsics_video.json, with its mode, its size and its film; the photos' calibration untouched")
    hand = Path(b.roots.local) / "film_by_hand.mp4"
    shutil.copy2(films[0], hand)

    class Dialog:
        @staticmethod
        def getOpenFileName(*_):
            return str(hand), ""
    real_dialog, ig.QFileDialog = ig.QFileDialog, Dialog
    n1 = len(d.events)
    try:
        added = d.call("calib_import", "")
    finally:
        ig.QFileDialog = real_dialog
    again = d.wait(lambda: ended(n1), 300)        # as long as the first film's: the choice is slow on a busy PC
    w = d.call("calib_state")
    expect(added.get("film") == "copying" and again and w["series"]["name"] != v["series"]["name"]
           and w["count"] == v["count"],
           f"a film added by hand: a series of its own ({w['series']['name']}), the same {w['count']} frames chosen"
           f"{'' if again else ' (its calibration not ended in 300 s)'}")
    hand.unlink(missing_ok=True)


def test_link(d):
    """The Phone panel against the simulated phone (phone_link.SimulatedAdb): the link found by itself and followed
    (no phone, the cable, Wi-Fi when the cable is out, the cable again), one place for the serial, Wi-Fi paired only
    when asked for (with the cable in, Wireless debugging switched on from the PC and a QR code), pairing with a code,
    a Wi-Fi drop and the reconnection, plain words for every failure, the live view sent to the phone's browser and
    its added latency."""
    import re
    import live
    import live_ar
    import phone_link
    import phone_view
    b = d.backend
    sim = b.link.run

    def act(fn, kind, action=None, timeout=20):
        """Calls fn and waits for an event of that kind (and action); the event, or None."""
        n = len(d.events)
        fn()

        def hit():
            return next((e for e in d.events[n:] if e["type"] == kind and (action is None or e.get("action") == action)),
                        None)
        return hit() if d.wait(lambda: hit() is not None, timeout) else None

    ok = d.wait(lambda: d.call("link_state")["state"] == "ready", 10)
    s = d.call("link_state")
    expect(ok and s["mode"] == "auto" and s["via"] == "usb" and not s["paired"]
           and b.link.serial() == sim.phone_id == live.pick_serial(),
           "link found by itself: the simulated phone on the cable, nothing asked; the stream asks the same link")
    v0 = s["view"]
    expect(b._view is None and v0["url_lan"] == "" and v0["qr"] == "" and v0["url_phone"] == "" and not v0["lan"]
           and v0["viewers"] == 0 and s["latency_ms"] == live_ar.STREAM_LATENCY_MS["usb"],
           "Phone panel open: the phone's view neither made nor on the network until asked for; the delay of the "
           "cable's stream said")

    # no phone at all, never paired: said plainly, a photo refused; the cable again, found again
    folder = d.call("calib_folder_new", "check_photo")["folder"]
    e = act(lambda: setattr(sim, "usb", False), "phone", timeout=10)
    f = act(lambda: d.call("calib_shoot", folder), "acquire_failed")
    expect(e is not None and e.get("cause") == "no_device" and f is not None and f["error"] == "no_device",
           f"cable out and never paired: no phone, a photo refused in plain words ({f and f['error']})")
    e = act(lambda: setattr(sim, "usb", True), "phone", timeout=10)
    expect(e is not None and e.get("ok") and e.get("link") == "usb", "cable in again: the phone, at once")

    # Use Wi-Fi with the cable in: Wireless debugging switched on from here, the QR code, scanned after 2 s
    sim.debugging = False
    n = len(d.events)
    d.call("link_wifi")
    shown = d.wait(lambda: d.call("link_state")["qr"].startswith("<svg"), 10)
    on = d.call("link_state")["wifi_on"]
    ok = d.wait(lambda: any(x["type"] == "link" and x.get("action") == "paired" for x in d.events[n:]), 25)
    s = d.call("link_state")
    expect(shown and on and sim.debugging and ok and s["via"] == "usb" and s["paired"]
           and b.prefs["phone"]["wifi_paired"],
           "Use Wi-Fi on the cable: Wireless debugging on from the PC, QR code shown and scanned, paired and kept; "
           "the cable still in use")

    # the cable out: the app goes on over Wi-Fi by itself, and says so; photos over Wi-Fi
    n = len(d.events)
    sim.usb = False
    ok = d.wait(lambda: any(x["type"] == "link" and x.get("action") == "via" and x.get("via") == "wifi"
                            for x in d.events[n:]), 15)
    s = d.call("link_state")
    expect(ok and s["state"] == "ready" and s["via"] == "wifi" and phone_link.transport_of(s["serial"]) == "wifi"
           and live.pick_serial() == s["serial"] and b.phone_status.get("link") == "wifi",
           f"cable out: on Wi-Fi by itself ({s['serial']}), the same serial for stream and photos, the page told")
    e = act(lambda: d.call("calib_shoot", folder), "calib_photo")
    expect(e is not None, "a photo over Wi-Fi")

    # the Wi-Fi drops and comes back by itself
    e = act(sim.drop, "phone", timeout=10)
    f = act(lambda: d.call("calib_shoot", folder), "acquire_failed")
    expect(e is not None and e.get("state") == "reconnecting" and f is not None and f["error"] == "wifi_reconnecting",
           f"Wi-Fi dropped: the pill says reconnecting, a photo is refused in plain words ({f and f['error']})")
    e = act(sim.back, "link", "state", timeout=25)
    ok = d.wait(lambda: d.call("link_state")["state"] == "ready", 25)
    s = d.call("link_state")
    expect(ok and s["reconnects"] >= 1 and s["drops"] >= 1, f"Wi-Fi back: reconnected by itself "
                                                           f"({s['drops']} drop, {s['reconnects']} reconnection)")

    # pairing with a code (Advanced): a wrong code said as such, the right one pairs
    e = act(lambda: d.call("link_pair", f"{sim.ip}:{sim.pair_port}", "000000"), "link", "failed")
    expect(e is not None and e["cause"] == "pair_code", f"pairing with a wrong code: said as such ({e and e['cause']})")
    e = act(lambda: d.call("link_pair", f"{sim.ip}:{sim.pair_port}", sim.code), "link", "paired")
    expect(e is not None and d.call("link_state")["state"] == "ready", "pairing with the code: paired and connected")

    # the cable in again: back on the cable by itself
    n = len(d.events)
    sim.usb = True
    ok = d.wait(lambda: any(x["type"] == "link" and x.get("action") == "via" and x.get("via") == "usb"
                            for x in d.events[n:]), 15)
    expect(ok and d.call("link_state")["via"] == "usb" and live.pick_serial() == sim.phone_id,
           "cable in again: back on the cable by itself, the USB serial")

    # the live view on the phone's own screen
    d.call("live_start")
    d.wait(lambda: d.call("live_state").get("fps_ar", 0) > 0, 30)
    e = act(lambda: d.call("view_open"), "view", timeout=15)
    v = d.call("link_state")["view"]
    port = b.view().port
    expect(e is not None and e["action"] == "opened" and sim.opened[-1:] == [v["url_phone"]]
           and f"tcp:{port}" in sim.reversed and v["url_phone"].startswith(f"http://127.0.0.1:{port}/v/"),
           "Open on the phone: the tunnel through adb, then the phone's browser on the page")
    th = threading.Thread(target=phone_view.client, args=(v["url_phone"], 5.0), daemon=True)
    th.start()
    d.wait(lambda: not th.is_alive(), 20)
    v = d.call("link_state")["view"]
    expect(v["viewers"] == 1 and v["fps"] > 10 and v["added_ms"] is not None,
           f"the phone's page on the live AR (a client in its place): {v['fps']} fps, added latency median "
           f"{v['added_ms']} ms (95th {v['added_p95_ms']} ms), {v['kb']} kB per frame, {v['mbit_s']} Mbit/s")
    d.call("live_stop")

    # another screen of the network: its address and code only while asked for (in simulation still this PC alone)
    token = b.view().token
    on = d.call("view_lan", True)["view"]
    off = d.call("view_lan", False)["view"]
    reopened = d.wait(lambda: sim.opened[-1:] == [b.view().url_phone], 10)
    expect(on["lan"] and on["url_lan"].startswith("http://") and on["qr"].startswith("<svg") and not off["lan"]
           and off["url_lan"] == "" and off["qr"] == "" and b._view_bound == "127.0.0.1" and b.view().token != token
           and reopened,
           "another screen: the address and its code shown only while asked for, a new address once it ends (the "
           "phone's own page opened again at it); the tests stay on this PC")
    # the cable out while the phone's view is open: its tunnel laid again on Wi-Fi
    sim.reversed.clear()
    n = len(d.events)
    sim.usb = False
    relaid = d.wait(lambda: f"tcp:{port}" in sim.reversed, 20)
    sim.usb = True
    back = (d.wait(lambda: any(x["type"] == "link" and x.get("action") == "via" and x.get("via") == "usb"
                               for x in d.events[n:]), 15)
            and d.wait(lambda: d.call("link_state")["state"] == "ready", 15))
    phones = [x for x in d.events if x["type"] == "phone"]
    expect(relaid and back and phones and all("detail" not in x and isinstance(x.get("left_on_phone"), list)
                                              for x in phones),
           "the phone's view: its tunnel laid again when the phone goes from the cable to Wi-Fi; the cable again; no "
           "adb text in any phone event, the files left on the phone in each")

    # the words: every cause in the six languages, every language with every key and the same placeholders
    strings = {c: json.loads((ig.APP / "ui" / "strings" / f"{c}.json").read_text(encoding="utf-8")) for c in ig.LANGS}
    keys = set(strings["en"])
    missing = [f"{c}:link.cause.{k}" for c in ig.LANGS for k in phone_link.CAUSES if f"link.cause.{k}" not in strings[c]]
    differ = [c for c in ig.LANGS if set(strings[c]) != keys]
    marks = [f"{c}:{k}" for c in ig.LANGS for k in keys & set(strings[c])
             if sorted(re.findall(r"\{\d\}", strings[c][k])) != sorted(re.findall(r"\{\d\}", strings["en"][k]))]
    used = set()
    for f in sorted((ig.APP / "ui").glob("*.js")) + [ig.APP / "ui.html"]:
        used |= set(re.findall(r"""(?:T\(|data-t(?:-title|-ph)?=)['"]((?:link|view|pill\.phone|live|board)\.[\w.]+)['"]""",
                               f.read_text(encoding="utf-8")))
    unknown = sorted(k for k in used if k not in keys and not k.endswith("."))   # a prefix with its end added
    expect(not missing and not differ and not marks and not unknown,
           f"words: {len(keys)} keys in each of the six languages, every link cause said, same placeholders "
           f"({missing or differ or marks or unknown or 'all there'})")


# ── the real window ──────────────────────────────────────────────────────────
DRIVE = r"""
(function () {
  const s = TRIALS.state;
  if (!s || !s.trial) return 'wait';
  const t = s.trial;
  if (!t.open) {
    if (window.__closed) return 'closed ' + window.__closed;
    if (!$('planMass').value) $('planMass').value = '137.2';
    $('planStart').click(); return 'start';
  }
  if (!window.__open) window.__open = JSON.parse(JSON.stringify(s));   // an open trial, for the sweep of the step panel
  const st = s.steps[s.step];
  const staged = s.staged[st.key];
  if (st.kind === 'photo' && !staged && !st.done) { if (!$('stShoot').disabled) $('stShoot').click(); return 'shoot ' + st.key; }
  if (st.kind === 'video' && !staged && !st.done) {
    if (s.recording !== st.key) { $('stRec').click(); return 'record ' + st.key; }
    if (!window.__recAt) window.__recAt = Date.now();
    if (Date.now() - window.__recAt > 1500) { window.__recAt = 0; $('stRec').click(); return 'stop ' + st.key; }
    return 'recording';
  }
  if (st.kind === 'scan' && !staged && !st.done) return 'waiting ' + st.key;
  if (st.kind === 'spheres') { $('stSpheres').checked = true; $('stShift').value = '1.4'; }
  if (st.kind === 'close') { $('stNotes').value = 'simulated through the page'; window.__closed = t.trial; }
  if ($('stOk').disabled) return 'disabled ' + st.key;
  $('stOk').click(); return 'confirm ' + st.key;
})()
"""

# What the sweep of every page needs in the page: open or close an Advanced part (SlantUI's fold), show a step of
# an open trial, and find the words that do not fit. A word does not fit when it is cut by a box (its own or one
# around it that clips) or when a label that must stay on one line wraps.
FIT = r"""
window.__adv = function (id, open) { if ($(id)) setFold($(id), !!open); };
window.__closeAdv = function () { closeFolds(); };
window.__step = function (base, adv) {
  const o = window.__open;
  if (!o) return;
  const s = JSON.parse(JSON.stringify(o));
  let k = s.steps.findIndex(function (x) { return x.key.replace(/_az\d+$/, '') === base; });
  if (k < 0) k = 0;
  s.step = k; s.staged = {}; s.recording = ''; s.watching = s.steps[k].key;
  trialsView = 'guide'; showView('trials-guide');
  TRIALS.render(s);
  __adv('stAdv', !!adv);
};
window.__fit = function () {
  const out = [];
  const ONE = '.btn-label, .tab span, .sec-lbl, .pi-l, .pi-row > span, .mitem-name, .mbadge, .chip, .stpill span, .combo-v,'
    + ' .toggle, .sl-l, .sl-v, .rp-title, .rp-sub, .home-hd, .home-sub, .choice-name, .choice-line, .stcell-l, .stcell-v,'
    + ' .fold-name, .tile-l, .tile-v, .tile-c, .live-light span, .trial-sub, .now-n, .ct-fname, .covl div, .lk-word,'
    + ' .czoom, .seq-n, .seq-v, .seq-t, .tool-group .btn, .trial-id, .count-of, .count-say, .guide-n, .guide-w,'
    + ' .guide-side-l, .guide-key span, .guide-leg-l, .ph-last-t, .bd-ask-hd, .bd-steps-hd, .bd-step-n, .fig-note, .bd-tool, .guide-d,'
    + ' .mg, .ml';
  const SKIP = '.log, .note-path, textarea, .combopop, .pickerpop, .toast';
  const shown = function (el) { return el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden'; };
  const name = function (el) {
    return (el.id ? '#' + el.id : (typeof el.className === 'string' && el.className ? '.' + el.className.split(' ')[0] : el.tagName))
      + ' "' + el.textContent.trim().replace(/\s+/g, ' ').slice(0, 48) + '"';
  };
  const clips = function (el) { return getComputedStyle(el).overflowX !== 'visible'; };
  document.querySelectorAll('.app *').forEach(function (el) {
    if (el.closest(SKIP) || !shown(el)) return;
    let own = false;
    el.childNodes.forEach(function (n) { if (n.nodeType === 3 && n.textContent.trim()) own = true; });
    if (!own) return;
    const cs = getComputedStyle(el);
    if (cs.display !== 'inline' && clips(el) && el.scrollWidth > el.clientWidth + 1) { out.push('cut ' + name(el)); return; }
    const r = el.getBoundingClientRect();
    for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
      if (!clips(a)) continue;
      const ar = a.getBoundingClientRect();
      if (r.right > ar.right + 1 || r.left < ar.left - 1) { out.push('out ' + name(el) + ' of ' + name(a)); break; }
    }
  });
  document.querySelectorAll('.app :is(' + ONE + ')').forEach(function (el) {
    if (el.closest(SKIP) || !shown(el) || !el.textContent.trim()) return;
    const range = document.createRange();
    range.selectNodeContents(el);
    let lines = 0, last = -1e9;
    Array.from(range.getClientRects()).filter(function (x) { return x.width > 0.5 && x.height > 0.5; })
      .map(function (x) { return x.bottom; }).sort(function (a, b) { return a - b; })
      .forEach(function (b) { if (b - last > 6) { lines++; last = b; } });
    if (lines > 1) out.push('wraps ' + name(el));
  });
  // nothing laid over the page: the Phone panel is the panel down the right, never over the stage
  if (LINK.open) {
    const rp = document.querySelector('.rp').getBoundingClientRect(), st = document.querySelector('.stage').getBoundingClientRect();
    if (rp.left < st.right - 1) out.push('over the stage: the Phone panel');
  }
  return JSON.stringify(Array.from(new Set(out)));
};
"""

# The viewer navigated as Blender's, in the twin: the whole bench framed at the opening; the middle button turning
# as Blender's turntable (about Z, on past the pole), Shift moving, Ctrl zooming, the wheel zooming to the centre,
# Alt + left for a touchpad; the keypad's views (1 3 7, with Ctrl), 9, 8 and 4 by 15 degrees, 5, +, 0, Home; a click
# selecting a part with its outline and its name, Alt+A and a click on nothing selecting nothing; the gizmo's letters,
# a click on its Z looking from the top, its hand and lens.
NAV = r"""
(function (v) {
  const out = {}, c = v.canvas, b = c.getBoundingClientRect();
  const settle = function () { v.shown.q.copy(v.goal.q); v.shown.dist = v.goal.dist; v.shown.at.copy(v.goal.at); v.place(v.shown); };
  const back = function () { return new THREE.Vector3(0, 0, 1).applyQuaternion(v.goal.q); };
  const elev = function () { return Math.asin(Math.max(-1, Math.min(1, back().dot(v.U)))) * 180 / Math.PI; };
  const azim = function () { const d = back(); return Math.atan2(d.dot(v.F), d.dot(v.R)) * 180 / Math.PI; };
  const ev = function (type, x, y, o) {
    const e = new PointerEvent(type, Object.assign({ pointerId: 7, bubbles: true, cancelable: true, clientX: b.left + x,
                                                     clientY: b.top + y, pointerType: 'mouse' }, o || {}));
    c.dispatchEvent(e);
  };
  const drag = function (x, y, dx, dy, o) {
    ev('pointerdown', x, y, o); ev('pointermove', x + dx / 2, y + dy / 2, o); ev('pointermove', x + dx, y + dy, o);
    ev('pointerup', x + dx, y + dy, o);
    settle();
  };
  const key = function (code, o) {
    window.dispatchEvent(new KeyboardEvent('keydown', Object.assign({ code: code, bubbles: true, cancelable: true }, o || {})));
    settle();
  };
  const near = function (a, b2, tol) { return Math.abs(a - b2) < (tol || 0.5); };
  const W = v.w, H = v.h;
  settle();
  // where the model falls on the screen: its farthest point from the centre (1: the edge) and the centre of its extent
  const extent = function () {
    v.eye.updateMatrixWorld();
    let x0 = 9, x1 = -9, y0 = 9, y1 = -9;
    v.points().forEach(function (p) {
      const n = new THREE.Vector3(p[0], p[1], p[2]).project(v.eye);
      x0 = Math.min(x0, n.x); x1 = Math.max(x1, n.x); y0 = Math.min(y0, n.y); y1 = Math.max(y1, n.y);
    });
    return { reach: Math.max(-x0, x1, -y0, y1), cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
  };
  const framedNow = function () {
    const e = extent();
    return e.reach <= 1.0001 && e.reach > 0.85 && (Math.abs(e.cx) < 0.02 || Math.abs(e.cy) < 0.02);
  };
  // 1. the whole model framed (what is drawn centred and filling the view), turning about the model's middle
  const box = v.bounds;
  out.framed = framedNow() && v.goal.at.distanceTo(box.getCenter(new THREE.Vector3())) < v.radius * 0.3;
  out.reach = Math.round(extent().reach * 1000) / 1000;
  // 2. middle button: turntable about Z, on past the pole
  let e0 = elev(), a0 = azim();
  drag(W / 2, H / 2, 120, 0, { button: 1, buttons: 4 });
  out.turn_across = near(elev(), e0) && !near(azim(), a0, 5);
  e0 = elev();
  drag(W / 2, H / 2, 0, -Math.min(W, H) * 0.6, { button: 1, buttons: 4 });
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(v.goal.q);
  out.turn_up = elev() < e0 - 30;
  drag(W / 2, H / 2, 0, Math.min(W, H) * 1.4, { button: 1, buttons: 4 });
  const up2 = new THREE.Vector3(0, 1, 0).applyQuaternion(v.goal.q);
  out.past_pole = up2.dot(v.U) < 0 || up.dot(v.U) < 0 || elev() > 80;     // the view goes over the top, no stop
  key('Home');
  // 3. Shift + middle moves, Ctrl + middle zooms, the wheel zooms to the centre
  let q0 = v.goal.q.clone(), at0 = v.goal.at.clone(), d0 = v.goal.dist;
  drag(W / 2, H / 2, 80, 40, { button: 1, buttons: 4, shiftKey: true });
  out.shift_moves = v.goal.q.angleTo(q0) < 1e-5 && v.goal.at.distanceTo(at0) > 0 && near(v.goal.dist, d0, 1e-9);
  at0 = v.goal.at.clone(); d0 = v.goal.dist;
  drag(W / 2, H / 2, 0, -80, { button: 1, buttons: 4, ctrlKey: true });
  out.ctrl_zooms_in = v.goal.dist < d0 * 0.8 && v.goal.at.distanceTo(at0) < 1e-9;
  d0 = v.goal.dist;
  c.dispatchEvent(new WheelEvent('wheel', { deltaY: 100, clientX: b.left + 20, clientY: b.top + 20, bubbles: true, cancelable: true }));
  settle();
  out.wheel_zooms_centre = v.goal.dist > d0 * 1.05 && v.goal.at.distanceTo(at0) < 1e-9;
  // 4. Alt + left as the middle button (a touchpad); Shift + Alt + left moves
  e0 = elev(); a0 = azim();
  drag(W / 2, H / 2, 120, 0, { button: 0, buttons: 1, altKey: true });
  out.alt_left_turns = near(elev(), e0) && !near(azim(), a0, 5);
  at0 = v.goal.at.clone(); q0 = v.goal.q.clone();
  drag(W / 2, H / 2, 60, 0, { button: 0, buttons: 1, altKey: true, shiftKey: true });
  out.shift_alt_left_moves = v.goal.q.angleTo(q0) < 1e-5 && v.goal.at.distanceTo(at0) > 0;
  // 5. the keypad and Home
  v.hot = true;
  const views = {};
  [['Numpad1', {}, 'front'], ['Numpad1', { ctrlKey: true }, 'back'], ['Numpad3', {}, 'right'], ['Numpad3', { ctrlKey: true }, 'left'],
   ['Numpad7', {}, 'top'], ['Numpad7', { ctrlKey: true }, 'bottom']].forEach(function (k) {
    key(k[0], k[1]);
    const d = back();
    views[k[2]] = [Math.round(d.dot(v.R) * 1000) / 1000, Math.round(d.dot(v.F) * 1000) / 1000, Math.round(d.dot(v.U) * 1000) / 1000];
  });
  out.views = views;
  out.keypad_views = views.front[1] === -1 && views.back[1] === 1 && views.right[0] === 1 && views.left[0] === -1
    && views.top[2] === 1 && views.bottom[2] === -1;
  key('Numpad1');
  key('Numpad9');
  out.opposite = Math.round(back().dot(v.F) * 1000) / 1000 === 1;
  key('Numpad1');
  e0 = elev();
  key('Numpad8');
  out.step_up = near(elev(), e0 + 15, 0.01);
  a0 = azim();
  key('Numpad4');
  out.step_side = near(Math.abs(azim() - a0), 15, 0.01);
  key('Numpad5');
  out.ortho = v.ortho && v.eye.isOrthographicCamera;
  key('Numpad5');
  out.persp = !v.ortho && v.eye.isPerspectiveCamera;
  d0 = v.goal.dist;
  key('NumpadAdd');
  out.plus = near(v.goal.dist, d0 / 1.2, 1e-6);
  key('Home');
  out.home = framedNow();
  // 6. a click selects a part, its outline and its name; Alt+A and a click on nothing select nothing
  const t = new THREE.Vector3().fromArray(TWIN.extras.target || [0, 0, 0]).project(v.eye);
  const tx = (t.x + 1) / 2 * W, ty = (1 - t.y) / 2 * H;
  ev('pointerdown', tx, ty, { button: 0, buttons: 1 }); ev('pointerup', tx, ty, { button: 0 });
  out.selected = v.sel ? v.sel.name : '';
  out.label = v.label.style.display !== 'none' && v.label.textContent === PARTS.name(out.selected)
    && PARTS.known(out.selected);
  out.outline_meshes = v.sel ? v.sel.parts.length : 0;
  key('KeyA', { altKey: true });
  out.alt_a = !v.sel;
  ev('pointerdown', tx, ty, { button: 0, buttons: 1 }); ev('pointerup', tx, ty, { button: 0 });
  const had = !!v.sel;
  ev('pointerdown', 3, H - 3, { button: 0, buttons: 1 }); ev('pointerup', 3, H - 3, { button: 0 });
  out.click_nothing = had && !v.sel;
  // 7. the gizmo: a click on Z looks from the top; its letters; the hand and the lens there
  const z = v.gz.axes.find(function (a) { return a.axis === 2 && a.sign > 0; });
  out.letters = v.gz.axes.filter(function (a) { return a.sign > 0; }).map(function (a) { return a.letter; }).join('');
  const zb = z.fill.getBoundingClientRect(), zx = zb.left + zb.width / 2, zy = zb.top + zb.height / 2;
  const pe = function (type, el, x, y) { el.dispatchEvent(new PointerEvent(type, { pointerId: 9, bubbles: true, clientX: x, clientY: y, button: 0 })); };
  pe('pointermove', v.gz.disc, zx, zy);
  pe('pointerdown', v.gz.disc, zx, zy);
  pe('pointerup', v.gz.disc, zx, zy);
  settle();
  out.gizmo_top = Math.round(back().dot(v.U) * 1000) / 1000 === 1;
  out.tools = v.gz.tools.map(function (t) { return t.name; }).join() === 'zoom,pan,camera,persp';
  // 8. the keypad's 0: the phone's view
  key('Numpad0');
  out.camera = TWIN.view;
  key('Home');
  return JSON.stringify(out);
})(TWIN.viewer)
"""
# The phone's link said by ear: nothing for the first state the page hears (the start), the phone gone and come back
# after it. The sounds are caught, never played; the page's phone put back as it was.
SOUND_PHONE = r"""
(function () {
  const real = PHONE, heard = phoneHeard, play = window.playSound, got = [];
  window.playSound = function (n) { got.push(n); return true; };
  try {
    phoneHeard = false;
    PHONE = { ok: false, cause: 'unknown', link: 'usb' };
    const ev = function (ok) { onEvent(JSON.stringify(Object.assign({}, real, { type: 'phone', ok: ok }))); };
    ev(true);
    const start = got.slice();
    ev(false);
    ev(true);
    return JSON.stringify([start, got]);
  } finally {
    window.playSound = play;
    PHONE = real;
    phoneHeard = heard;
    showPhone();
  }
})()
"""
# The plate in 3D on the board step: the board's kind, its model loaded, its box shown, no GL context to be had.
PLATE_SHOWN = ("JSON.stringify([BOARD.state.kind, !!PLATE3D.model, !$('bd3d').classList.contains('hidden'),"
               " !!PLATE3D.failed])")
# The plate's viewer (glTF's Y up): framed at the opening, the keypad's 7 from its top, 1 from its front, Home.
PLATE_NAV = r"""
(function (v) {
  const back = function () { return new THREE.Vector3(0, 0, 1).applyQuaternion(v.goal.q); };
  const settle = function () { v.shown.q.copy(v.goal.q); v.shown.dist = v.goal.dist; v.shown.at.copy(v.goal.at); v.place(v.shown); };
  const reach = function () {
    v.eye.updateMatrixWorld();
    return Math.max.apply(null, v.points().map(function (p) {
      const n = new THREE.Vector3(p[0], p[1], p[2]).project(v.eye); return Math.max(Math.abs(n.x), Math.abs(n.y)); }));
  };
  const key = function (code) { window.dispatchEvent(new KeyboardEvent('keydown', { code: code, bubbles: true, cancelable: true })); settle(); };
  settle();
  const r0 = reach();
  v.hot = true;
  key('Numpad7'); const top = Math.round(back().dot(v.U) * 1000) / 1000;
  key('Numpad1'); const front = Math.round(back().dot(v.F) * 1000) / 1000;
  key('Home'); const r1 = reach();
  v.hot = false;
  return JSON.stringify({ framed: r0 <= 1.0001 && r0 > 0.85, top: top, front: front, home: r1 <= 1.0001 && r1 > 0.85 });
})(PLATE3D.viewer)
"""
# The twin through the trial's steps: each with its pose (the arm, the pendulum, the drape, the marker after the
# impact), the scroller's buttons and the arrows; then back at the step the trial is at. Every part's name in words.
TWIN_STEPS = r"""
(function () {
  const out = { poses: [] }, back = TWIN.entry;
  for (let i = 0; i < TWIN.entries.length; i++) {
    TWIN.goEntry(i);
    const p = TWIN.probe();
    out.poses.push([p.pose.key, p.pose.sub, p.armAz, p.swing, p.drape, p.pose.after, p.sweeping, p.stepTitle]);
  }
  TWIN.goEntry(0);
  $('twNext').click(); out.next = TWIN.entry === 1;
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
  out.arrow = TWIN.entry === 2;
  $('twPrev').click(); out.prev = TWIN.entry === 1;
  TWIN.goEntry(back);
  out.steps = TWIN.steps.length; out.entries = TWIN.entries.length; out.angle = Number(TWIN.state.trial.angle_deg);
  out.drapeTrial = !!TWIN.state.trial.drape; out.cameraDeg = TWIN.extras.arm.camera_deg;
  const names = {};
  TWIN.model.traverse(function (m) { (m.userData.objects || []).forEach(function (o) { names[o[0]] = 1; }); });
  out.unnamed = Object.keys(names).filter(function (n) { return !PARTS.known(n); });
  return JSON.stringify(out);
})()
"""
# The phone's lens in the twin's view from the phone (ui/18-lens.js): a calibration with a plain phone's distortion
# laid on, the view from the phone; points of the world on rays through a grid of the photo out to its corners, where
# the bent frame shows each (read from the shader) and where the page puts each (labels, clicks). LENS_JS sets it,
# LENS_READ reads it; lens_errors() holds them against OpenCV's projectPoints.
LENS_CAL = {"size": [4080, 3072], "K": [[3050.0, 0.0, 2051.0], [0.0, 3046.0, 1529.0], [0.0, 0.0, 1.0]],
            "dist": [0.11, -0.32, 0.001, -0.0007, 0.29]}
LENS_JS = "TWIN.lensTest = " + json.dumps(LENS_CAL) + "; TWIN.look('phone1')"
LENS_READ = r"""
(function () {
  const v = TWIN.viewer, L = LENS.of(TWIN.lensTest), pts = [];
  TWIN.look('phone1', true);          // (there at once, whatever came in between)
  v.place(v.shown);
  v.eye.updateMatrixWorld();
  const on = TWIN.probe().lens;
  [0.02, 0.25, 0.5, 0.75, 0.98].forEach(function (a) { [0.02, 0.3, 0.5, 0.7, 0.98].forEach(function (b) {
    const p = LENS.ideal(L, a * L.w - 0.5, b * L.h - 0.5), d = 300 + 400 * a * b;
    pts.push(new THREE.Vector3(p[0] * d, -p[1] * d, -d).applyMatrix4(v.eye.matrixWorld).toArray());
  }); });
  const m = LENS.measure(v, pts);
  v.o.changed();                      // turned by hand: the free view, straight
  const off = !TWIN.probe().lens && TWIN.view === '';
  TWIN.lensTest = null; v.draw();
  return JSON.stringify({ on: on, off: off, pts: pts, m: m });
})()
"""


# The board's holder on the post in the head's place (ui/18-twin.js setMount), the board taken to seat 2, then the head
# again; nothing when the model has no holder (made before it was in the scene)
BOARD_READ = r"""
(function () {
  if (!TWIN.hasBoard()) return JSON.stringify({ has: false });
  const at = function () { const p = TWIN.probe(); return [p.mount, p.seat, p.headShown, p.boardShown, p.holderA || p.holderB, p.boardAt, p.swing]; };
  TWIN.setMount('board', 0); const s1 = at();
  $('twSeats').children[1].click(); const s2 = at();
  $('twMount').children[0].click(); const h = at();
  return JSON.stringify({ has: true, s1: s1, s2: s2, head: h, seats: $('twSeats').children.length });
})()
"""


def lens_errors(got: dict) -> dict:
    """The twin's lens against OpenCV: each point projected with LENS_CAL from the eye the page drew with, put in the
    box as the page fits the photo (centred, all of it), and the worst distance (photo px) to where the bent frame
    shows it and to where the page puts it."""
    import cv2
    import numpy as np
    m = got.get("m") or {}
    if not m:
        return {"n": 0, "shown": float("inf"), "put": float("inf")}
    M = np.array(m["eye"], float).reshape(4, 4).T                   # three's matrices are by columns
    X = np.array(got["pts"], float)
    C = (X - M[:3, 3]) @ M[:3, :3]                                   # in the eye (three's: y up, looking down -z)
    C = C * [1, -1, -1]                                              # OpenCV's: y down, looking down +z
    K, dist = np.array(LENS_CAL["K"]), np.array(LENS_CAL["dist"])
    uv = cv2.projectPoints(C.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    f = m["fit"]
    box = np.column_stack([f["ox"] + (uv[:, 0] + 0.5) * f["s"], f["oy"] + (uv[:, 1] + 0.5) * f["s"]])
    worst = {}
    for key in ("shown", "put"):
        at = m["shown"] if key == "shown" else m["fwd"]
        d = [np.hypot(*(np.array(a) - b)) / f["s"] if a is not None else float("inf") for a, b in zip(at, box)]
        worst[key] = float(max(d))
    inside = (uv[:, 0] > -0.5) & (uv[:, 0] < LENS_CAL["size"][0]) & (uv[:, 1] > -0.5) & (uv[:, 1] < LENS_CAL["size"][1])
    bend = float(np.max(np.hypot(*(uv - (C[:, :2] / C[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2])).T)))
    return {"n": len(uv), "inside": int(inside.sum()), "bend_px": bend, **worst}


# The sweep: home, every step of every module, its Advanced part open, every kind of step of a trial, the Phone
# panel (on the cable, Advanced, pairing on Wi-Fi), at the smallest window and at the largest of this screen, in
# the six languages. (name, script, wait ms, picture).
SWEEP = [
    ("home", "LINK.hide(); __closeAdv(); go('home')", 1300, True),
    ("phone_home", "LINK.pairing = false; LINK.show()", 1600, True),
    ("phone_adv", "__adv('lkAdv', true)", 900, True),
    ("phone_pair", "clearInterval(LINK.timer); __adv('lkAdv', false); LINK.pairing = true;"
                   "LINK.render(Object.assign({}, LINK.state, {state: 'none', cause: 'no_device', paired: false,"
                   " wifi_on: false, qr: ''}))", 600, True),
    ("cal_ask", "LINK.hide(); __closeAdv(); go('calibration', 'board');"
                "setTimeout(function () { BOARD.render(Object.assign({}, BOARD.state, {chosen: false})); }, 600)",
     1500, True),
    ("cal_board", "BOARD.set({kind: 'screen'})", 1600, True),
    # a monitor with a long name, whatever this computer has (the Philips 40" of the bench, as Windows names it)
    ("cal_board_long", "BOARD.render(Object.assign({}, BOARD.state, {mon: Object.assign({}, BOARD.state.mon,"
                       " {label: 'PHI PHL BDM4037U', diag_in: 40.0})}))", 600, False),
    ("cal_board_ruler", "__adv('calBdRuler', true); BOARDVIEW.label()", 900, True),
    ("cal_plate", "__adv('calBdRuler', false); BOARD.set({kind: 'plate'})", 2200, True),
    ("cal_plate_short", "__adv('calBdShort', true)", 900, True),
    ("cal_sheet", "__adv('calBdShort', false); BOARD.set({kind: 'sheet'})", 1300, True),
    ("cal_photos_phys", "go('calibration', 'photos')", 1500, True),
    ("cal_bench", "BOARD.set({kind: 'plate'}); CAL.set({plan: 'bench'})", 2200, True),
    ("cal_bench_quick", "CAL.set({plan: 'quick'})", 900, False),
    ("cal_photos", "BOARD.set({kind: 'screen'}); go('calibration', 'photos')", 1800, True),
    ("cal_photos_adv", "__adv('calPhAdv', true)", 900, True),
    ("cal_stream", "__adv('calPhAdv', false); CAL.set({mode: 'stream'})", 5000, True),
    ("cal_video", "CAL.set({mode: 'video'})", 1800, True),
    ("cal_photo_mode", "CAL.set({mode: 'photo'})", 1500, False),
    ("cal_result", "go('calibration', 'result')", 1300, True),
    ("cal_result_adv", "__adv('calResAdv', true)", 700, False),
    ("live_board", "__closeAdv(); go('live', 'board')", 1300, True),
    ("live_patient", "LIVE.settings({board: 'patient'})", 900, True),
    ("live_on", "LIVE.settings({board: 'sheet'}); go('live', 'view');"
                "setTimeout(function () { if (!(LIVE.state && (LIVE.state.running || LIVE.state.starting))) $('lvStart').click(); }, 400)",
     6500, True),
    ("live_on_adv", "__adv('lvAdv', true); $('lvAdv').scrollIntoView()", 900, True),
    ("live_off", "__adv('lvAdv', false); LIVE.stop()", 1500, True),
    ("trials_plan", "go('trials', 'guide')", 1300, True),
    ("trials_plan_adv", "__adv('planAdv', true); TRIALS.extraOpen = true; TRIALS.renderPlan(TRIALS.state)", 900, True),
    ("step_holder", "__step('holder')", 700, True),
    ("step_pendulum", "__step('pendulum')", 700, False),
    ("step_ref_scan", "__step('ref_scan')", 700, True),
    ("step_release", "__step('release')", 700, False),
    ("step_spheres", "__step('spheres')", 700, True),
    ("step_photo", "__step('drape_photo', true)", 700, True),
    ("step_video", "__step('drape_video')", 700, False),
    ("step_close", "__step('close')", 700, True),
    ("trials_back", "TRIALS.extraOpen = false; __closeAdv(); TRIALS.refresh()", 900, False),
    ("log", "go('trials', 'log'); __adv('lgAdv', true)", 1300, True),
    ("sequence", "__closeAdv(); go('trials', 'sequence')", 1300, True),
    ("config", "go('trials', 'config'); __adv('cfgAdv', true)", 1300, True),
    ("twin", "__closeAdv(); go('trials', 'guide'); setTimeout(function () { TWIN.open('trials'); }, 300)", 1500, True),
    ("twin_adv", "__adv('twAdv', true)", 900, False),
    ("results", "__closeAdv(); go('results', 'scans')", 900, True),
]
# the least window, the window at its first start (ig.WINDOW_SIZE) and a large one
SIZES = (("min",) + ig.MIN_WINDOW, ("first",) + ig.WINDOW_SIZE, ("large", 1920, 1032))
SHOTS = {("large", "en"), ("large", "it"), ("min", "en"), ("min", "de")}
SIM_PHOTO = (2040, 1536)        # the simulated phone's photos in the window test: half the phone's, the same checks
# the window at each display scaling, in physical pixels (Qt's are these over the scaling): as large as a 1920 x 1080
# screen holds, so the board fits beside the panel; less on a screen that has less room
SCALE_WINDOW = {"1.25": (1750, 1000), "1.5": (1860, 1020)}
SCALES = ("1.25", "1.5")
FIT_AGAIN_MS = 450      # the words of a view measured twice this far apart: only what stays wrong in both counts
AWAY_GAP = 64           # the window under test lies this far right of every screen, out of the way of the computer's user
# Chromium in the window test: WebGL given back after a loss of the GPU's process (grabbing the window can cause one),
# never blocked for the page for a while after it, nor the GPU given up after a few losses
TEST_CHROMIUM = "--disable-domain-blocking-for-3d-apis --disable-gpu-process-crash-limit"


def Qt_key(name):
    from slantui.shell.qt import Qt
    return getattr(Qt.Key, "Key_" + name)


def safe_output():
    """Printing never stops the test: a console that cannot write a sign (cp1252 and the inch sign, a Chinese word)
    writes it as an escape instead of raising UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and (stream.encoding or "").lower().replace("-", "") != "utf8":
            try:
                stream.reconfigure(errors="backslashreplace")
            except (AttributeError, ValueError):
                pass


def build_window(argv):
    """The app's window for the window tests: Chromium as TEST_CHROMIUM has it, and the focus given back to the window
    that had it (the computer may be in use: what is typed there never reaches the app under test)."""
    import ctypes
    real = ig.Application
    user32 = ctypes.windll.user32 if os.name == "nt" else None
    before = user32.GetForegroundWindow() if user32 else None

    def application(*a, **kw):
        kw["chromium"] = (kw.get("chromium", "") + " " + TEST_CHROMIUM).strip()
        return real(*a, **kw)
    ig.Application = application
    try:
        app, win, backend = ig.build(ig.parse(argv))
    finally:
        ig.Application = real
    if user32 and before:
        user32.SetForegroundWindow(before)
    return app, win, backend


def one_window_test(wait_s=2700):
    """One window test at a time on this computer (two would grab each other's windows and screens): a named mutex,
    waited for up to wait_s, held until this process ends. Returns whether it is held."""
    if os.name != "nt":
        return True
    import ctypes
    k = ctypes.windll.kernel32
    k.CreateMutexW.restype = ctypes.c_void_p
    k.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    h = k.CreateMutexW(None, 0, "Local\\ImpactGuideWindowTest")
    if not h:
        return False
    t0, said = time.monotonic(), False
    while time.monotonic() - t0 < wait_s:
        if k.WaitForSingleObject(h, 5000) in (0, 0x80):     # WAIT_OBJECT_0, WAIT_ABANDONED (its holder ended)
            globals()["_WINDOW_TEST_MUTEX"] = h
            return True
        if not said:
            said = True
            print("waiting for another window test of ImpactGuide to end", flush=True)
    return False


def away_origin(app):
    """Where the window under test lies: right of every screen, at the top of the first one."""
    vg = app.primaryScreen().virtualGeometry()
    return vg.x() + vg.width() + AWAY_GAP, vg.y()


def native_scale(app):
    """The display scaling of the screen at the top left of the desktop, as Windows has it (Qt's own factor, if one
    is set in the environment, taken out)."""
    if os.name == "nt":     # asked of Windows now: Qt's screens stay as they were if the displays changed meanwhile
        import ctypes
        from ctypes import wintypes
        u, s = ctypes.windll.user32, ctypes.windll.shcore
        u.MonitorFromPoint.restype = ctypes.c_void_p
        mon = u.MonitorFromPoint(wintypes.POINT(0, 0), 1)       # MONITOR_DEFAULTTOPRIMARY
        x, y = ctypes.c_uint(), ctypes.c_uint()
        if mon and s.GetDpiForMonitor(ctypes.c_void_p(mon), 0, ctypes.byref(x), ctypes.byref(y)) == 0 and x.value:
            return x.value / 96.0
    screen = next((s for s in app.screens() if s.geometry().contains(0, 0)), None) or app.primaryScreen()
    return screen.devicePixelRatio() / float(os.environ.get("QT_SCALE_FACTOR") or 1.0)


def grab_screen(screen):
    """The screen as it is, in physical pixels, grey (what a camera filming it would see, minus the camera)."""
    import numpy as np
    from slantui.shell.qt import QImage
    q = screen.grabWindow(0).toImage().convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = q.width(), q.height()
    ptr = q.constBits()
    ptr.setsize(q.bytesPerLine() * h)
    return np.frombuffer(ptr, np.uint8).reshape(h, q.bytesPerLine())[:, :w].copy()


def grab_composed(win):
    """The window as Windows composes it, even under other windows (PrintWindow, PW_RENDERFULLCONTENT), grey, in
    physical pixels, with where its top left corner is on the screen: for when Windows does not let a window of a
    process in the background go on top of the one in use (it says SetWindowPos went well, and the window stays
    under). (image, (x, y)) or (None, None)."""
    import ctypes
    from ctypes import wintypes
    import cv2
    import numpy as np
    u, g = ctypes.windll.user32, ctypes.windll.gdi32
    for f in (u.GetWindowDC, u.PrintWindow, u.ReleaseDC, g.CreateCompatibleDC, g.CreateCompatibleBitmap,
              g.SelectObject, g.GetDIBits, g.DeleteObject, g.DeleteDC):
        f.restype = ctypes.c_void_p if f in (u.GetWindowDC, g.CreateCompatibleDC, g.CreateCompatibleBitmap,
                                             g.SelectObject) else ctypes.c_int
    u.GetWindowDC.argtypes = [wintypes.HWND]
    u.ReleaseDC.argtypes = [wintypes.HWND, ctypes.c_void_p]
    u.PrintWindow.argtypes = [wintypes.HWND, ctypes.c_void_p, ctypes.c_uint]
    g.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    g.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    g.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    g.GetDIBits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.c_uint]
    g.DeleteObject.argtypes = [ctypes.c_void_p]
    g.DeleteDC.argtypes = [ctypes.c_void_p]

    class BIH(ctypes.Structure):
        # ctypes reads the class attribute: a mutable list is what it expects
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),  # noqa: RUF012
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]
    h = wintypes.HWND(int(win.winId()))
    r = wintypes.RECT()
    u.GetWindowRect(h, ctypes.byref(r))
    w, hh = r.right - r.left, r.bottom - r.top
    if w <= 0 or hh <= 0:
        return None, None
    wdc = u.GetWindowDC(h)
    mdc = g.CreateCompatibleDC(wdc)
    bmp = g.CreateCompatibleBitmap(wdc, w, hh)
    g.SelectObject(mdc, bmp)
    ok = u.PrintWindow(h, mdc, 2)
    bih = BIH(ctypes.sizeof(BIH), w, -hh, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = (ctypes.c_ubyte * (w * hh * 4))()
    rows = g.GetDIBits(mdc, bmp, 0, hh, buf, ctypes.byref(bih), 0)
    g.DeleteObject(bmp)
    g.DeleteDC(mdc)
    u.ReleaseDC(h, wdc)
    if not ok or rows != hh:
        return None, None
    img = np.frombuffer(buf, np.uint8).reshape(hh, w, 4)[:, :, :3]
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2GRAY), (r.left, r.top)


def on_top(win, top):
    """The window above every other one (without taking the focus), or back among them: the computer may be in use
    while the test runs, and a grab of the screen must see the app (HWND_TOPMOST, HWND_NOTOPMOST; SWP_NOMOVE, NOSIZE,
    NOACTIVATE, and SHOWWINDOW, restored first if Windows started it minimised). Returns what Windows says of it:
    {"ok", "visible", "iconic", "topmost", "rect"}."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_uint]
    u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    h = wintypes.HWND(int(win.winId()))
    if top and u.IsIconic(h):
        u.ShowWindow(h, 9)          # SW_RESTORE
    ok = u.SetWindowPos(h, wintypes.HWND(-1 if top else -2), 0, 0, 0, 0, 0x13 | (0x40 if top else 0))
    r = wintypes.RECT()
    u.GetWindowRect(h, ctypes.byref(r))
    return {"ok": bool(ok), "visible": bool(u.IsWindowVisible(h)), "iconic": bool(u.IsIconic(h)),
            "topmost": bool(u.GetWindowLongW(h, -20) & 0x8), "rect": [r.left, r.top, r.right, r.bottom]}


def board_on_screen(app, win, backend, out=None, seen=None):
    """The board the page draws, as the screen shows it: the screen grabbed, the board's image (screen_board.render)
    looked for in it and compared pixel for pixel over the board and its two thin lines, and the board found in the
    grab by the calibration's own detection. Returns (identical, corners found, corners, where on the screen); the
    last grab of the screen in seen["screen"] and what Windows said of the window in seen["window"] when seen is
    given."""
    import cv2
    import numpy as np
    shown = on_top(win, True)
    if seen is not None:
        seen["window"] = shown

    def look(img, at=(0, 0)):
        top = lay["ext_top"] - 2
        box = ref[top:y0 + ny * s + 2, x0 - 2:x0 + nx * s + 2]
        patch = box[: min(200, box.shape[0]), : min(200, box.shape[1])]
        m = cv2.matchTemplate(img, patch, cv2.TM_SQDIFF)
        _, _, loc, _ = cv2.minMaxLoc(m)      # where (its score is not exactly 0 on a large grab: float rounding)
        own = img[loc[1]:loc[1] + box.shape[0], loc[0]:loc[0] + box.shape[1]]
        return own.shape == box.shape and bool(np.array_equal(own, box)), at[0] + loc[0], at[1] + loc[1], own
    for attempt in range(4):        # a board being laid out again after a resize: grabbed again a moment later
        for _ in range(8 if not attempt else 12):
            app.processEvents()
            time.sleep(0.05)
        lay, ref = backend._lay, backend._board_img
        if lay is None or ref is None:
            continue
        x0, y0 = lay["origin_px"]
        nx, ny = lay["squares"]
        s = lay["square_px"]
        img = grab_screen(win.screen())
        if seen is not None:
            seen["screen"] = img
        same, gx, gy, own = look(img)
        if not same and not shown["topmost"]:
            # Windows kept the window under the one in use: the window as it composes it, where it lies on the screen
            comp, at = grab_composed(win)
            if comp is not None:
                same, gx, gy, own = look(comp, at)
                if seen is not None:
                    seen["composed"] = same
        if same:
            break
    on_top(win, False)
    if lay is None or ref is None:
        return False, 0, 0, None
    found = 0
    if same and out is not None:
        cv2.imwrite(str(out), own)
        views, _, _ = calibrate_charuco.detect([Path(out)], screen_board.make_board(lay))
        found = len(views[Path(out).name][0]) if views else 0
    return same, found, (nx - 1) * (ny - 1), [gx, gy]


def test_ui(work, shots):
    from slantui.shell.qt import QTimer
    import twin
    shots.mkdir(parents=True, exist_ok=True)
    # the twin's model for the page: the app's own (a simulation takes it), else made now from the bench scene
    tw = twin.Watcher(work / "twin", seed=twin.CACHE)
    if tw.current() is None and tw.can_export() and tw.scene.exists():
        tw.run()
    twin_model = tw.current() is not None
    app, win, backend = build_window(["--simulate", str(work), "--lang", "en"])
    away = away_origin(app)

    def place(w, h, on=False):
        """The window w x h (Qt's pixels) right of every screen, out of the way; or (on) whole on the first screen,
        for a grab of the screen itself. A window minimised meanwhile is put back first."""
        if win.isMinimized():
            win.showNormal()
        win.setGeometry(*((0, 0) if on else away), w, h)
    place(1440, 900)
    log = []
    taken = set()
    misfits = []       # (size, language, state, what) for every word that does not fit
    board = {"grabs": []}

    def grab_board(name):
        # the board as the screen shows it: the window on the screen for this moment only, then back out of the way
        place(win.width(), win.height(), on=True)
        try:
            same, found, corners, where = board_on_screen(app, win, backend, shots / f"{name}.png")
        finally:
            place(win.width(), win.height())
        lay = backend._lay
        board["grabs"].append((name, same, found, corners, lay and lay["squares"], lay and round(lay["square_mm"], 3),
                               where))
        if same:
            taken.add(name)

    def grab(name):
        if name.startswith("board_"):
            grab_board(name)
            return
        img = win.grabWindow()
        for _ in range(10):  # now and then the first grab comes back at a fraction of the size
            if img.width() >= win.width():
                break
            app.processEvents()
            img = win.grabWindow()
        img.save(str(shots / f"{name}.png"))
        taken.add(name)

    def js(script, cb=None):
        win.run_js(script, cb or (lambda v: None))

    def drive():
        def got(v):
            v = str(v)
            log.append(v)
            if v.startswith("closed"):
                finish()
                return
            # pictures at the moments that show the most
            if v.startswith("shoot drape_photo_az40") or v.startswith("shoot photo_az40"):
                QTimer.singleShot(1400, lambda: grab("ui_02_photo_received" if "drape" not in v else "ui_03_drape_photo"))
            if v.startswith("waiting scan_az") and "ui_04_scan_wait" not in taken:
                grab("ui_04_scan_wait")
            if v == "start":
                QTimer.singleShot(900, lambda: grab("ui_01_trial_started"))
            QTimer.singleShot(2200 if v.startswith(("shoot", "stop")) else 700, drive)
        js(DRIVE, got)

    def lang(code):
        return f"$('lang').value='{code}'; $('lang').dispatchEvent(new Event('change'));"

    # the guided photos through the page: Take until the plan's least, each photo checked as it comes
    def take_photos(n_want, then):
        state = {"tries": 0}

        def step():
            state["tries"] += 1
            if state["tries"] > 400:
                log.append("photos timeout")
                then()
                return

            def got(v):
                d = json.loads(v)
                if d["n"] == 1 and "cal_14_first" not in taken:
                    grab("cal_14_first")
                if d["n"] >= n_want:
                    log.append("photos " + v)
                    then()
                    return
                if not d["busy"] and not d["disabled"]:
                    js("$('calTake').click()")
                QTimer.singleShot(900, step)
            js("JSON.stringify({n: CAL.state ? CAL.state.count : 0, busy: CAL.state ? CAL.busy(CAL.state) : true,"
               " disabled: $('calTake').disabled, guide: CAL.state && CAL.state.next && CAL.state.next.n,"
               " last: CAL.state && CAL.state.series && CAL.state.series.last && CAL.state.series.last.ok})", got)
        step()

    def wait_calib(then, tag):
        state = {"tries": 0}

        def step():
            state["tries"] += 1

            def got(v):
                d = json.loads(v)
                if (d["r"] and not d["busy"]) or state["tries"] > 200:
                    log.append(tag + " " + v)
                    then()
                    return
                QTimer.singleShot(1200, step)
            js("JSON.stringify({r: CAL.state && CAL.state.result && {rms: CAL.state.result.rms_px, n: CAL.state.result.n_views,"
               " verdict: CAL.state.result.verdict, saved: CAL.state.result.saved, left: CAL.state.result.photos.filter("
               "function (p) { return p.state === 'excluded'; }).length}, busy: CAL.state.busy.indexOf('calibration') >= 0,"
               " what: $('calRun').dataset.what})", got)
        QTimer.singleShot(1500, step)

    def finish():
        backend.phone().photo_px = SIM_PHOTO
        # (name, script, wait ms, what to read after the wait or None); a name starting with "-" takes no picture
        steps = [
            ("ui_05_closed", "", 900, None),
            ("ui_06_log", "go('trials', 'log')", 1200, None),
            ("ui_07_sequence", "go('trials', 'sequence')", 1200, None),
            ("ui_08_config", "go('trials', 'config')", 1200, None),
            # the twin of the bench: from the trial's page, its ready views, the trial after its scans
            ("ui_08t_twin", "go('trials', 'guide'); setTimeout(function () { $('gTwin').click(); }, 700)", 3200,
             ("twin_open", "JSON.stringify(TWIN.probe())")),
            ("-twin_nav", "", 300, ("twin_nav", NAV)),
            ("-twin_steps", "", 300, ("twin_steps", TWIN_STEPS)),
            ("ui_08t2_twin_selected", "(function () { const v = TWIN.viewer, t = new THREE.Vector3().fromArray("
                                      "TWIN.extras.target).project(v.eye); v.select(v.pick((t.x + 1) / 2 * v.w,"
                                      " (1 - t.y) / 2 * v.h)); })()", 900, None),
            ("ui_08u_twin_top", "TWIN.look('top')", 1200, ("twin_top", "JSON.stringify(TWIN.probe())")),
            ("-twin_phone", "TWIN.look('phone2')", 1000, ("twin_phone", "JSON.stringify(TWIN.probe())")),
            ("-twin_lens", LENS_JS, 1500, ("twin_lens", LENS_READ)),
            ("-twin_board", "", 300, ("twin_board", BOARD_READ)),
            ("ui_08v_twin_scans", "go('results', 'scans'); setTimeout(function () { TWIN.pick = '';"
                                  " TWIN.open('results'); }, 400)", 2900, ("twin_scans", "JSON.stringify(TWIN.probe())")),
            ("ui_08w_twin_gain", "$('twGain').value = '12'; $('twGain').dispatchEvent(new Event('input'))", 1000,
             ("twin_gain", "JSON.stringify(TWIN.probe())")),
            ("ui_08b_home", "document.querySelector('.tbar-logo-btn').click()", 1200,
             ("home", "JSON.stringify([page, !!document.querySelector('.tbar-logo-btn'),"
                      " document.querySelector('.app').classList.contains('at-home'),"
                      " document.querySelector('.tbar-logo-btn').title === T('win.home.hint'),"
                      " getComputedStyle(document.querySelector('.rp')).display,"
                      " getComputedStyle(document.querySelector('.topbar')).display, $('rail').children.length])")),
            ("ui_08c_tile", "document.querySelector('.choice[data-go=calibration]').click()", 1500,
             ("tile", "JSON.stringify([page, STEP.calibration, Array.from($('rail').children).map(function (t) {"
                      " return t.dataset.step; }), getComputedStyle(document.querySelector('.topbar')).display,"
                      " !$('bdAsk').classList.contains('hidden')])")),
            # the first question: this monitor; the board drawn by the page in its work area, pixel for pixel
            ("cal_09_ask", "", 300, None),
            ("cal_10_screen", "document.querySelector('#bdAsk .choice[data-kind=screen]').click()", 2500,
             ("screen", "JSON.stringify([BOARD.state.kind, BOARD.state.chosen, !!BOARD.state.layout, BOARDVIEW.key,"
                        " BOARDVIEW.canvas.width, BOARDVIEW.canvas.height, BOARDVIEW.dev])")),
            ("board_10_screen", "", 300, None),
            # the window resized: laid out again for the new area, still pixel for pixel; nothing moved the window
            ("-resize", lambda: place(1700, 960), 2500, None),
            ("board_11_resized", "", 300,
             ("resized", "JSON.stringify([BOARD.state.layout && BOARD.state.layout.squares, BOARDVIEW.key])")),
            ("-geo", lambda: board.update(geo=[win.x(), win.y(), win.width(), win.height()]), 200, None),
            ("cal_12_ruler", "setFold($('calBdRuler'), true); BOARDVIEW.label()", 900, None),
            ("cal_13_ruler_typed", "$('calBdRulerV').value = (BOARD.state.layout.width_mm * 1.003).toFixed(1);"
                                   "$('calBdRulerV').dispatchEvent(new Event('change'))", 1500,
             ("ruler", "JSON.stringify(BOARD.state.mon.source)")),
            ("-ruler_forget", "BOARD.set({forget: true}); setFold($('calBdRuler'), false)", 1200, None),
            # physical: the plate in 3D beside what to do; its three readings
            ("cal_13b_plate", "document.querySelector('#calBdKinds .mitem[data-kind=physical]').click()", 3000,
             ("plate", PLATE_SHOWN)),
            # no GL context (Chromium lost its GPU's process, which grabbing the window can cause): asked once more
            ("-plate_again", "if (PLATE3D.failed) { PLATE3D.failed = false; PLATE3D.viewer = null; PLATE3D.show(); }",
             2500, ("plate_again", PLATE_SHOWN)),
            ("-plate_nav", "", 300, ("plate_nav", PLATE_NAV)),
            ("cal_13c_readings", "var i = $('calBdLong').querySelectorAll('input'); i[0].value = '220.05';"
                                 " i[1].value = '220.02'; i[2].value = '220.11'; i[2].dispatchEvent(new Event('change'))", 1500,
             ("readings", "JSON.stringify(BOARD.state.physical.plate.long)")),
            ("cal_13d_turned", lambda: turn_plate(), 1200, None),
            ("cal_13e_sheet", "document.querySelector('#calBdWhich .mitem[data-kind=sheet]').click()", 1500, None),
            ("-back_screen", "document.querySelector('#calBdKinds .mitem[data-kind=screen]').click()", 2000, None),
            # the photos, one after the other with the guide, each checked; then the calibration (the photo taken by
            # itself off here, so that the count is the page's own; under the calibration's least, Calibrate says so)
            ("-run_few", "CAL.set({auto: false}); go('calibration', 'result')", 1500,
             ("run_few", "JSON.stringify([CAL.state.count, $('calRunL').textContent, $('calRun').disabled,"
                         " CAL.state.min_views])")),
            ("cal_14_photos", "go('calibration', 'photos')", 2500,
             ("photos_board", "JSON.stringify([BOARDVIEW.host && BOARDVIEW.host.id, BOARDVIEW.key])")),
            ("board_14_photos", "", 300, None),
            ("-take", lambda: None, 0, ("take", 12)),
            # the space bar takes the next photo at once: the guide flashes and moves on
            ("-space", "window.__n0 = CAL.state.count; window.__g0 = CAL.state.next && CAL.state.next.n;"
                       " document.body.dispatchEvent(new KeyboardEvent('keydown', {key: ' ', bubbles: true}));"
                       " document.body.dispatchEvent(new KeyboardEvent('keyup', {key: ' ', bubbles: true}))", 600,
             ("space_flash", "JSON.stringify([$('phGuide').classList.contains('shot'), $('cpCard').classList.contains('shot')])")),
            ("cal_14b_space", "", 2500,
             ("space", "JSON.stringify([window.__n0, CAL.state.count, window.__g0, CAL.state.next && CAL.state.next.n,"
                       " $('calAutoSay').textContent, (CAL.state.series.shots || []).length])")),
            # the phone's video: the space bar never starts or stops a film, and nothing is taken by itself
            ("-space_video", "CAL.set({mode: 'video'}); setTimeout(function () { document.body.dispatchEvent("
                             "new KeyboardEvent('keydown', {key: ' ', bubbles: true})); }, 900)", 2600,
             ("space_video", "JSON.stringify([CAL.state.mode, !!(CAL.state.film && (CAL.state.film.recording"
                             " || CAL.state.film.starting)), $('calAutoRow').classList.contains('hidden'),"
                             " CAL.state.busy.indexOf('phone') >= 0])")),
            ("-space_photo", "CAL.set({mode: 'photo'})", 1500, None),
            ("cal_15_enough", "", 600, None),
            ("cal_16_result", "go('calibration', 'result'); setTimeout(function () { $('calRun').click(); }, 800)", 0,
             ("calib", "first")),
            ("cal_17_calibrated", "", 600, None),
            ("cal_18_left_out", "var r = CAL.state.result; var m = r.photos.slice().sort(function (a, b) {"
                                " return b.err - a.err; })[0]; window.__out = m.name;"
                                " document.querySelectorAll('#crBars .bar')[r.photos.indexOf(m)]"
                                ".dispatchEvent(new MouseEvent('click', {bubbles: true}))", 900,
             ("leave", "JSON.stringify([$('calRun').dataset.what, CAL.exclude])")),
            ("-again", "$('calRun').click()", 0, ("calib", "again")),
            ("cal_19_again", "", 600, None),
            # a calibration that is not good enough is saved only with a second press, the button saying so
            ("-save_poor", "window.__verdict = CAL.state.result.verdict; CAL.state.result.verdict = 'poor';"
                           " $('calRun').click(); CAL.state.result.verdict = window.__verdict", 500,
             ("save_poor", "JSON.stringify([$('calRunL').textContent, CAL.state.result.saved, !!$('calRun')._armed])")),
            ("cal_20_saved", "$('calRun').click()", 1800,
             ("saved", "JSON.stringify([CAL.state.result && CAL.state.result.saved,"
                       " CAL.state.result && CAL.state.result.photos.filter(function (p) { return p.state === 'excluded'; })"
                       ".map(function (p) { return p.name; }), window.__out])")),
            # the streaming mode: the board beside the live picture, the board hidden in it; the stream still on and
            # its frames in the picture (the page's live view never stops Calibrate's stream)
            ("cal_21_stream", "go('calibration', 'photos'); CAL.set({mode: 'stream'})", 6000,
             ("stream", "JSON.stringify([CAL.state.live && CAL.state.live.running, !$('phLive').classList.contains('hidden'),"
                        " BOARDVIEW.host && BOARDVIEW.host.id, $('phLiveImg').naturalWidth > 0])")),
            ("-stream_on", lambda: log.append("stream_on " + json.dumps(bool(backend._live and backend._live.running))),
             0, None),
            ("board_21_stream", "", 300, None),
            ("-stream_off", "CAL.set({mode: 'photo'}); go('calibration', 'board')", 2000, None),
            ("ui_18_it_home", lang("it") + " go('home')", 1800, None),
            ("ui_19_it_live", "go('live', 'view')", 1500, None),
            ("ui_20_it_live_on", "$('lvStart').click()", 7000,
             ("live_board", "JSON.stringify([BOARDVIEW.host && BOARDVIEW.host.id, LIVE.state && LIVE.state.hide_board])")),
            ("board_20_live", "", 300, None),
            # the opacity of the virtual elements, from its slider: the live setting `opacity` through live_settings
            ("ui_21_opacity", "$('lvOpacity').value = '0.65'; $('lvOpacity').dispatchEvent(new Event('input'));"
                              "$('lvOpacity').dispatchEvent(new Event('change'))", 1200,
             ("opacity", "JSON.stringify([LIVE.sentAt > 0, LIVE.state && LIVE.state.opacity, $('lvOpacityV').textContent])")),
            ("ui_22_live_board", lang("en") + "window.__live = LIVE.state && LIVE.state.tracked;"
                                              " setTimeout(function () { go('live', 'board'); }, 900)", 2500, None),
            ("ui_23_live_patient", "LIVE.settings({board: 'patient'})", 1500, None),
            ("-live_sheet", "LIVE.settings({board: 'sheet'})", 800, None),
            # the Phone panel: in the panel down the right, at home too; Wi-Fi paired with the QR code
            ("ui_24_phone_home", "go('home'); LINK.show()", 2000,
             ("phone", "JSON.stringify([LINK.open, getComputedStyle(document.querySelector('.rp')).display,"
                       " document.querySelector('.rp-page.show').dataset.panel])")),
            ("ui_25_phone_wifi", "$('lkWifi').click()", 1200, None),
            ("ui_26_phone_paired", "", 4500, ("paired", "JSON.stringify([LINK.state && LINK.state.paired,"
                                                        " LINK.state && LINK.state.via])")),
            ("ui_27_phone_in_module", "go('trials', 'guide'); LINK.show()", 1500, None),
            ("ui_28_phone_closed", "LINK.hide()", 900,
             ("back", "JSON.stringify([LINK.open, document.querySelector('.rp-page.show').dataset.panel])")),
            ("-sound_phone", "", 200, ("sound_phone", SOUND_PHONE)),
            ("ui_29_results", "go('results', 'analysis')", 1200, None),
        ]
        # the sweep of every module and panel, at two sizes and in six languages
        for size, w, h in SIZES:
            steps.append((f"-size_{size}", (lambda w=w, h=h: place(w, h)), 1200, None))
            for code in ig.LANGS:
                steps.append((f"-lang_{size}_{code}", lang(code), 1500, None))
                for n, (state, script, wait, pic) in enumerate(SWEEP):
                    shot = pic and (size, code) in SHOTS
                    steps.append(((f"{size}_{code}_{n:02d}_{state}" if shot else f"-{size}_{code}_{state}"), script, wait,
                                  ("fit", "__fit()", size, code, state)))
        steps.append(("-restore", lambda: place(1440, 900), 800, None))

        def turn_plate():
            js("if (PLATE3D.viewer) PLATE3D.viewer.turn(0.9, -0.3);")

        size_of = {name: (w, h) for name, w, h in SIZES}

        def measured(what, v, again=None):
            if what[0] == "fit":
                try:
                    found = json.loads(v)
                    later = json.loads(again) if again is not None else None
                except (TypeError, ValueError):
                    misfits.append((what[2], what[3], what[4], f"no answer ({v}, {again})"))
                    return
                for m in found:
                    if later is None or m in later:     # still wrong a moment later: not a view that was settling
                        misfits.append((what[2], what[3], what[4], m))
            else:
                log.append(what[0] + " " + str(v))

        def fit(what, then, tries=0):
            """The words of the view measured twice, FIT_AGAIN_MS apart, with the window at the size of this pass: a
            window that is not (minimised, moved or resized from outside the test) is put back and measured again."""
            want = size_of[what[2]]
            if (win.width(), win.height()) != want or win.isMinimized():
                if tries < 2:
                    place(*want)
                    QTimer.singleShot(1500, lambda: fit(what, then, tries + 1))
                    return
                misfits.append((what[2], what[3], what[4], f"window at {win.width()} x {win.height()}, not "
                                                           f"{want[0]} x {want[1]}"))
                then()
                return
            js(what[1], lambda v: QTimer.singleShot(
                FIT_AGAIN_MS, lambda: js(what[1], lambda v2: (measured(what, v, v2), then()))))

        def nxt():
            if not steps:
                js("JSON.stringify(window.__ruler || '')", lambda v: log.append("ruler0 " + str(v)))
                js("JSON.stringify([!!window.__live, !!(LIVE.state && LIVE.state.running)])",
                   lambda v: log.append("live " + str(v)))
                js("JSON.stringify(window.__errs||[])", lambda v: (log.append("errors " + str(v)), app.quit()))
                return
            name, script, wait, what = steps.pop(0)
            if callable(script):
                script()
            elif script:
                js(script)

            def after():
                if not name.startswith("-"):
                    grab(name)
                if what is None:
                    nxt()
                elif what[0] == "take":
                    take_photos(what[1], nxt)
                elif what[0] == "calib":
                    wait_calib(lambda: (grab(name) if not name.startswith("-") else None, nxt()), "calib_" + what[1])
                elif what[0] == "fit":
                    fit(what, nxt)
                else:
                    js(what[1], lambda v: (measured(what, v), nxt()))
            if what is not None and what[0] == "calib":
                QTimer.singleShot(wait, lambda: wait_calib(nxt, "calib_" + what[1]))
            else:
                QTimer.singleShot(wait, after)
        nxt()

    def first():
        # the page's errors are kept from its first script on (ui/00-errors.js); a page without it gets a late keeper
        js("if(!window.__errs){window.__errs=[];window.addEventListener('error',function(e){"
           "window.__errs.push(e.message+' line '+e.lineno)});}")
        js(FIT)
        js("JSON.stringify([page, document.querySelector('.app').classList.contains('at-home'),"
           " getComputedStyle(document.querySelector('.topbar')).display])",
           lambda v: log.append("start " + str(v)))
        grab("ui_00_home")
        js("go('trials', 'guide')")
        # the drape slot comes first, so the run shows both kinds of views
        QTimer.singleShot(1500, drive_drape)

    def drive_drape():
        grab("ui_00_plan")
        run = backend.run()
        slot = next(r for r in run.pending() if r["drape"] == "1")
        js("TRIALS.extraOpen=true; TRIALS.renderPlan(TRIALS.state);"
           f"$('exHolder').value='{slot['holder']}'; $('exAngle').value='{slot['angle_deg']}';"
           f"$('exDir').value='{slot['direction_deg']}'; $('exDrape').checked=true; $('exApply').click();")
        QTimer.singleShot(1200, lambda: (grab("ui_00_plan_drape"), drive()))

    QTimer.singleShot(4000, first)
    QTimer.singleShot(3000000, app.quit)
    app.run()
    win.hide()      # out of the way: the scaling checks that follow open their windows where it stood
    app.processEvents()
    errors = next((x for x in log if x.startswith("errors")), "errors none")
    run = trials.Run("campaign", work / "raw", work / "protocol")
    rows = run.log.rows()
    expect(rows and rows[-1]["status"] == "ok" and rows[-1]["drape"] == "1" and not run.missing(rows[-1]["trial"]),
           f"window: drape trial {rows[-1]['trial'] if rows else '?'} run with the page's buttons, nothing missing")
    expect(errors in ("errors []", "errors none"), f"window: no script errors ({errors})")

    def read(tag):
        x = next((x for x in log if x.startswith(tag + " ")), tag + " ?")
        try:
            return json.loads(x.split(" ", 1)[1])
        except (ValueError, IndexError):
            return None
    start = read("start")
    expect(start == ["home", True, "none"], f"window: the app opens at home, with no rail, toolbar or panel ({start})")
    home = read("home")
    expect(home == ["home", True, True, True, "none", "none", 0],
           f"window: the mark on the band leads home from a module, a raised button with its hint; no rail and no "
           f"panel at home ({home})")
    tile = read("tile")
    expect(tile == ["calibration", "board", ["board", "photos", "result"], "flex", True],
           f"window: Calibrate opens at its first step, the first question on the stage ({tile})")
    screen = read("screen")
    expect(screen and screen[:3] == ["screen", True, True] and screen[3] and screen[6] == screen[4:6],
           f"window: this monitor chosen, the board drawn in the page, the canvas as large as its area in device "
           f"pixels ({screen})")
    grabs = board["grabs"]
    expect(len(grabs) == 5 and all(g[1] and g[2] == g[3] for g in grabs),
           f"window: the board as the screen shows it, in the board step, after a resize, in the photos, beside the "
           f"stream and beside the live view: the page's image pixel for pixel, every corner found in the grab "
           f"({grabs})")
    resized = read("resized")
    expect(resized and grabs[0][4] != grabs[1][4] and board.get("geo") == [*away, 1700, 960],
           f"window: the window resized by hand, the board laid out again for its new area ({grabs[0][4]} -> "
           f"{grabs[1][4]}); nothing moved or resized the window ({board.get('geo')})")
    ruler = read("ruler")
    expect(ruler == "ruler", f"window: the ruler check typed in the page corrected the size ({ruler})")
    plate, plate2, readings = read("plate"), read("plate_again"), read("readings")
    shown = any(p and p[:3] == ["plate", True, True] for p in (plate, plate2))
    no_gl = bool(plate and plate[3] and plate2 and plate2[3])     # no GL context, asked twice
    if no_gl and not shown:
        print("skip  window: the plate in 3D, no GL context twice (WebGL taken away after a loss of the GPU's "
              "process); its readings checked", flush=True)
    expect((shown or no_gl) and plate and plate[0] == "plate" and readings and abs(readings["mean_mm"] - 220.06) < 1e-6
           and abs(readings["spread_mm"] - 0.09) < 1e-6,
           f"window: the physical board, the 3D-printed plate in 3D, three caliper readings with their mean and spread "
           f"({plate}, {plate2}, {readings})")
    photos = read("photos")
    expect(photos and photos["n"] >= 12 and photos["last"],
           f"window: photos taken with the page's Take button, each checked as it came ({photos})")
    few, flash, space = read("run_few"), read("space_flash"), read("space")
    space_video = read("space_video")
    expect(space_video == ["video", False, True, False],
           f"window: in the phone's video the space bar starts no film and the photo taken by itself is not offered "
           f"({space_video})")
    expect(few and few[0] < few[3] and few[2] and few[1] == "At least 8 photos",
           f"window: under the calibration's least, Calibrate says how many it takes ({few})")
    expect(flash == [True, True] and space and space[1] == space[0] + 1 and space[3] != space[2]
           and space[4] == "Space bar or Enter: take" and space[5] >= 12,
           f"window: the space bar takes the photo at once, the guide and the counter flash ({flash}), the guide moves "
           f"on, the strip shows the photos ({space})")
    c1, c2 = read("calib_first"), read("calib_again")
    leave, saved = read("leave"), read("saved")
    sound_phone = read("sound_phone")
    expect(sound_phone == [[], ["off", "on"]],
           f"window: the phone's link by ear: no sound for the state at the start, then gone and back ({sound_phone})")
    poor = read("save_poor")
    expect(poor == ["Not good enough: save anyway?", False, True],
           f"window: a calibration not good enough is not saved at the first press, the button asks ({poor})")
    expect(c1 and c1["r"] and c1["r"]["verdict"] == "good" and leave and leave[0] == "again" and c2 and c2["r"]
           and c2["r"]["left"] == 1 and saved and saved[0] and saved[1] == [saved[2]],
           f"window: calibrated ({c1 and c1['r']}), the worst photo left out with a press on its bar, calibrated again "
           f"({c2 and c2['r']}), saved with it left out ({saved})")
    stream, stream_on = read("stream"), read("stream_on")
    expect(stream == [True, True, "bdHostPhotos", True] and stream_on is True,
           f"window: the streaming mode, the live picture beside the board on this monitor, its frames shown and the "
           f"stream still on ({stream}, {stream_on})")
    lb = read("live_board")
    expect(lb == ["bdHostLive", True], f"window: the live view beside the board, the board hidden in its picture ({lb})")
    opacity = read("opacity")
    expect(opacity is not None and opacity[0] and "65" in opacity[2] and opacity[1] == 0.65,
           f"window: the opacity slider sends `opacity` 0.65 through live_settings ({opacity})")
    live = read("live")
    expect(live == [True, False], f"window: live view started from its step, board tracked, stopped when the step "
                                  f"is left ({live})")
    phone, paired, back = read("phone"), read("paired"), read("back")
    expect(phone == [True, "flex", "phone"] and paired == [True, "usb"] and back == [False, "trials-plan"],
           f"window: the Phone panel in the panel down the right, at home too ({phone}); Wi-Fi paired with the QR "
           f"code, the cable still in use ({paired}); closed, the step's panel back ({back})")
    checked = len(SIZES) * len(ig.LANGS) * len(SWEEP)
    for m in misfits[:80]:
        print("      " + " · ".join(m), flush=True)
    expect(not misfits, f"window: every word fits, nothing over the page, on {len(SWEEP)} views, at "
                        f"{', '.join(f'{w} x {h}' for _, w, h in SIZES)}, in the six languages "
                        f"({checked} views, {len(misfits)} words that do not fit)")
    t_open, t_top, t_phone, t_scans, t_gain = (read(k) for k in ("twin_open", "twin_top", "twin_phone", "twin_scans",
                                                                  "twin_gain"))
    if not twin_model:
        print("skip  window: the twin, no model of the bench (no Blender, no scene)", flush=True)
    else:
        moving = {"marker", "marker_a", "holder_a", "holder_b", "pendulum", "pendulum_swing", "arm"}
        if t_open and t_open.get("noGl"):
            # Chromium takes WebGL away for a while after it lost its GPU process, which grabbing the window can cause
            print("skip  window: the twin had no GL context (WebGL taken away after a loss of the GPU's process); "
                  "the rest of its opening checked", flush=True)
            t_open = {**t_open, "drawn": True}
        expect(t_open and t_open["loaded"] and t_open["drawn"] and t_open["view"] == ""
               and moving <= set(t_open["nodes"]) and 0 < t_open["openMs"] < 1000 and t_open["loadMs"] < 1000
               and t_open["holderA"] != t_open["holderB"],
               f"window: the twin from the trial's page, the whole bench framed, one holder shown, the model "
               f"loaded in {t_open and t_open['loadMs']} ms, open in {t_open and t_open['openMs']} ms")
        expect(t_top and t_top["view"] == "top" and t_top["fov"] == 30 and t_phone and t_phone["view"] == "phone2"
               and t_phone["armAz"] == 70 and t_phone["fov"] > 40,
               f"window: the twin from above, and from the phone at 70° with the phone's field of view "
               f"({t_phone and t_phone['fov']}°)")
        t_lens = read("twin_lens") or {}
        le = lens_errors(t_lens)
        expect(t_lens.get("on") and t_lens.get("off") and le["n"] == le["inside"] == 25 and le["bend_px"] > 20
               and le["shown"] < 0.5 and le["put"] < 0.5,
               f"window: the twin from the phone through its lens, as OpenCV projects {le['n']} points out to the "
               f"photo's corners (bent up to {le.get('bend_px', 0):.0f} px): worst {le['shown']:.3f} px in the "
               f"frame, {le['put']:.3f} px where the page puts them; straight once turned by hand")
        tb = read("twin_board") or {}
        if not tb.get("has"):
            print("skip  window: the twin's board holder, not in this model (made before it)", flush=True)
        else:
            s1, s2, hd = tb["s1"], tb["s2"], tb["head"]
            expect(s1[:5] == ["board", 0, False, True, False] and s2[:4] == ["board", 1, False, True]
                   and s2[5] != s1[5] and hd[:5] == ["head", 1, True, False, True] and tb["seats"] == 3
                   and s1[6] == 30,
                   f"window: the twin with the board's holder on the post in the head's place, the board moved to "
                   f"seat 2 ({s1[5]} -> {s2[5]} mm), the pendulum hooked up; the head again")
        size = lambda p: sum(x * x for x in (p or {}).get("marker") or [0]) ** 0.5  # noqa: E731
        expect(t_scans and t_scans["mode"] == "results" and t_scans["points"] == 7 and t_gain and t_gain["ghost"]
               and t_gain["gain"] == 12 and size(t_gain) > 5 * size(t_scans) > 0,
               f"window: the twin in Scans and results, the marker moved as the scans say, its place before as a "
               f"ghost, the 7 targets in colour; enlarged 12 times ({size(t_scans):.3f} -> {size(t_gain):.3f} mm)")
        nav = read("twin_nav") or {}
        bad = sorted(k for k, x in nav.items() if x is False)
        expect(nav and not bad and nav.get("selected") and nav.get("outline_meshes", 0) > 0
               and nav.get("letters") == "XYZ" and nav.get("camera", "").startswith("phone"),
               f"window: the twin navigated as Blender's (turntable, Shift, Ctrl, wheel, Alt + left, keypad, Home, "
               f"a part selected with its outline and name: {nav.get('selected')}, gizmo); not as it should: "
               f"{bad or 'none'}")
        st = read("twin_steps") or {}
        poses = st.get("poses") or []
        rel = [p for p in poses if p[0] == "release"]
        at = {p[0]: p for p in poses}
        photos = [p for p in poses if p[0].startswith("photo_az")]
        first_after = next((i for i, p in enumerate(poses) if p[5]), len(poses))
        expect(st and st["entries"] == st["steps"] + 1 and [p[1] for p in rel] == ["raised", "impact"]
               and abs(rel[0][3] - st["angle"]) < 0.2 and abs(rel[1][3]) < 0.2
               and all(p[2] == int(p[0].rsplit("az", 1)[1]) for p in photos) and photos
               and at.get("video", [0] * 7)[6] and not any(p[5] for p in poses[:first_after])
               and poses[first_after - 1][0] == "release" and all(p[5] for p in poses[first_after:])
               and (st["drapeTrial"] or not any(p[4] for p in poses)) and all(p[7] for p in poses)
               and st["next"] and st["arrow"] and st["prev"] and not st["unnamed"],
               f"window: the twin through the {st.get('steps')} steps of the trial (the pendulum raised at "
               f"{st.get('angle')}° then hanging at the impact, the arm at each photo, the video's sweep, the marker "
               f"moved only after the impact, the drape only with it), its buttons and arrows; every part's name in "
               f"words (unknown: {st.get('unnamed')})")
    pnav = read("plate_nav") or {}
    if no_gl and not shown:
        print("skip  window: the plate's viewer navigated, no GL context (see above)", flush=True)
    else:
        expect(pnav.get("framed") and pnav.get("top") == 1 and pnav.get("front") == -1 and pnav.get("home"),
               f"window: the plate's viewer framed at the opening, the keypad's 7 from its top and 1 from its front, "
               f"Home ({pnav})")
    expect(len(taken) >= 10, f"window: {len(taken)} pictures in {shots}")
    print("page actions: " + ", ".join(x for x in log if not x.startswith(("confirm", "shoot", "record", "stop",
                                                                            "recording", "waiting", "disabled"))),
          flush=True)


def scale_check(work, scale, out):
    """The board in the page at a display scaling (Qt's own, as Windows does at 125 or 150 %: QT_SCALE_FACTOR, set
    before Qt starts): the window put at whole physical pixels, the board as the screen shows it against the page's
    image, pixel for pixel. Prints one JSON line."""
    from slantui.shell.qt import QTimer
    app, win, backend = build_window(["--simulate", str(work), "--lang", "en"])
    f = float(scale)
    # SCALE_WINDOW's physical pixels in Qt's, at most the room the screen has
    dpr, room = win.screen().devicePixelRatio(), win.screen().availableGeometry()
    pw, ph = SCALE_WINDOW[scale]
    win.setGeometry(0, 0, min(round(pw / dpr), room.width()), min(round(ph / dpr), room.height()))
    res = {}

    def go():
        win.run_js("go('calibration', 'board'); BOARD.set({kind: 'screen'})")
        QTimer.singleShot(3500, grab)

    def grab():
        seen = {}
        same, found, corners, where = board_on_screen(app, win, backend, Path(out) / f"board_scale_{scale}.png", seen)
        if not same and "screen" in seen:      # what the screen showed instead, to see why
            import cv2
            cv2.imwrite(str(Path(out) / f"board_scale_{scale}_screen.png"), seen["screen"])
        lay = backend._lay
        if not same:
            res.update(window=seen.get("window"))
        if "composed" in seen:
            res.update(composed=seen["composed"])
        res.update(scale=f, dpr=win.screen().devicePixelRatio(), identical=same, corners_found=found, corners=corners,
                   squares=lay and lay["squares"], square_px=lay and lay["square_px"],
                   square_mm=lay and round(lay["square_mm"], 4), exact=backend._lay_exact, area=backend._area)
        print(json.dumps(res), flush=True)
        app.quit()
    QTimer.singleShot(4500, go)
    QTimer.singleShot(60000, app.quit)
    app.run()
    return 0 if res.get("identical") and res.get("corners_found") == res.get("corners") else 1


def test_scales(work, shots):
    """The board pixel for pixel at 125 and 150 % (scale_check in a process of its own, Qt's scaling being set before
    it starts). Qt's factor is the scaling wanted over the screen's own: on a screen at 150 % the check at 150 % runs
    at the screen's own scaling and the one at 125 % with a factor of 0.83; on one at 100 %, 1.25 and 1.5."""
    import subprocess
    from slantui.shell.qt import QApplication
    app = QApplication.instance()
    native = native_scale(app) if app is not None else 1.0
    for scale in SCALES:
        factor = float(scale) / native
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env.pop("QT_SCALE_FACTOR", None)
        if abs(factor - 1.0) > 1e-6:
            env["QT_SCALE_FACTOR"] = f"{factor:.6f}"
        runs = 0
        while runs < 2:         # once more if the screen was not the app's (the computer in use meanwhile)
            runs += 1
            r = subprocess.run([sys.executable, str(Path(__file__)), "--scale-check", scale, "--work",
                                str(work / f"scale_{scale}"), "--ui", str(shots)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env, timeout=180, check=False)
            line = next((x for x in r.stdout.splitlines() if x.startswith("{")), "")
            if not line:        # no answer: what the check said instead, to see why
                said = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or [""])[-1]
                line = json.dumps({"exit": r.returncode, "said": said[-300:]})
            d = json.loads(line)
            if d.get("identical") and d.get("corners_found") == d.get("corners"):
                break
        expect(d.get("identical") and d.get("corners_found") == d.get("corners")
               and abs(float(d.get("dpr") or 0) - float(scale)) < 1e-3,
               f"window at {int(float(scale) * 100)} % display scaling (the screen at {native * 100:.0f} %, Qt's "
               f"factor {factor:.4g}): the board on the screen as the page drew it, pixel for pixel, every corner "
               f"found ({line}{', second run' if runs > 1 else ''})")


def test_scans(app, work):
    """The Scans page's backend on synthetic scans of the bench (scan_desk.SimScanner, IMPACTGUIDE_SIM_SCANS):
    exports taken by the folder's own watch without the trial's page asking, each bound to its step by the arm
    position it was taken from (the second position exported first), an export no step waits for said unused, the
    marker's shift and the error at the targets shown within 30 s of the export, the log's two values written for
    the trial's spheres step, the twin on the measured motion at once; the spheres' radii calibrated from a run of
    three scans without an impact and the trial measured again with them; a repeatability run and its spread; the
    day's processing into gt_poses.csv."""
    old = os.environ.get(scans_page.SIM_SCANS_ENV)
    os.environ[scans_page.SIM_SCANS_ENV] = "synthetic"
    try:
        _test_scans(app, fresh(work))
    finally:
        if old is None:
            os.environ.pop(scans_page.SIM_SCANS_ENV, None)
        else:
            os.environ[scans_page.SIM_SCANS_ENV] = old


def _test_scans(app, work):
    import csv
    import scan_desk
    d = Driver(app, work)
    b = d.backend
    s = d.call("trial_start", json.dumps({"swing_mass_g": "150"}))
    trial = s["trial"]["trial"]
    keys = [x["key"] for x in s["steps"]]
    refs = [k for k in keys if k.startswith("ref_scan_")]
    posts = [k for k in keys if k.startswith("scan_")]
    folder = b.scanner_folder()
    cfg = b.run().config

    def export(key, name):
        sl = scan_desk.slot_of(key, cfg)
        return b.scans.sim().export(folder, name, sl["az"], b.run().log.row(trial), sl["when"])

    def scans_events(n):
        return [e for e in d.events[n:] if e["type"] == "scans"]

    d.wait(lambda: b.scans.watch is not None, 10)
    export(refs[1], "export_ref_second.ply")
    got = d.wait(lambda: d.staged(refs[1]), 40)
    expect(got and not d.staged(refs[0]) and len(refs) == 2,
           f"scans: an export taken by the folder's watch without the trial's page asking, bound by the arm position "
           f"it was taken from to {refs[1]}, not to the first step waiting ({refs[0]})")
    export(refs[0], "export_ref_first.ply")
    d.wait(lambda: d.staged(refs[0]), 40)
    # the steps before the impact as the operator confirms them
    while True:
        s = d.call("trials_state")
        st = s["steps"][s["step"]]
        if st["key"] == "release":
            break
        s = d.call("step_confirm", st["key"], "{}") if st["kind"] == "scan" else d.do_step(st)
        assert "error" not in s, s
    n = len(d.events)
    export(refs[0], "export_stray.ply")         # every reference step has its scan, the impact is still to come
    stray = d.wait(lambda: any(e.get("action") == "unused" for e in scans_events(n)), 20)
    expect(stray and not any(e["type"] == "staged" for e in d.events[n:]),
           "scans: an export no step waits for is said unused and given to no step")
    d.do_step(d.call("trials_state")["steps"][d.call("trials_state")["step"]])          # the release
    n = len(d.events)
    t0 = time.time()
    # the trial's page waits at the first scan after the impact and its own watch takes an export of the second
    # position: it goes to the second scan's step
    sl = scan_desk.slot_of(posts[1], cfg)
    second = b.scans.sim().export(work / "elsewhere", f"export_{posts[1]}.ply", sl["az"], b.run().log.row(trial),
                                  sl["when"])
    b.scans.came(second, prefer=posts[0])
    got = d.wait(lambda: d.staged(posts[1]), 40)
    expect(got and not d.staged(posts[0]),
           f"scans: an export of the second arm position taken while the trial's page waits at the first goes to "
           f"{posts[1]}")
    export(posts[0], f"export_{posts[0]}.ply")
    d.wait(lambda: d.staged(posts[0]), 40)
    ready = d.wait(lambda: any(e.get("action") == "measured" and e.get("state") == "ready" for e in scans_events(n)), 60)
    ev = next((e for e in scans_events(n) if e.get("action") == "measured" and e.get("state") == "ready"), {})
    st = d.call("scans_state", "{}")
    res = st["result"]
    row = b.run().log.row(trial)
    expect(ready and ev.get("seconds") is not None and ev["seconds"] < 30 and res["state"] == "ready"
           and len(res["tre"]) == 7 and res["tre_max_mm"] > 0 and res["views"] == len(posts)
           and all(x["state"] == "ok" and x["n_spheres"] == 6 for x in st["slots"]),
           f"scans: the marker's shift {res.get('shift_mm')} mm, {res.get('rot_deg')}°, the worst of 7 targets "
           f"{res.get('tre_max_mm')} mm, {ev.get('seconds')} s after the export ({time.time() - t0:.0f} s for both)")
    expect(row["spheres_ok"] == "1" and row["cap_shift_mm"] == res["log"]["cap_shift_mm"]
           and set(ev.get("logged", [])) == {"spheres_ok", "cap_shift_mm"} and st["calibration"]["done"] is False,
           f"scans: spheres_ok and cap_shift_mm ({row['cap_shift_mm']} mm) written in the log for the spheres step; "
           f"not calibrated said")
    seen = [x["arm_seen"] for x in st["slots"]]
    expect(seen == [x["arm"] for x in st["slots"]] and all(seen),
           f"scans: each scan taken from the arm position of its step ({seen})")
    # the trial to its end, its scans kept as staged: the twin shows the measured motion at once
    for _ in range(40):
        s = d.call("trials_state")
        st = s["steps"][s["step"]]
        if st["kind"] == "scan":
            s = d.call("step_confirm", st["key"], "{}")
        else:
            s = d.do_step(st, spheres={"spheres_ok": row["spheres_ok"], "cap_shift_mm": row["cap_shift_mm"]})
        assert "error" not in s, s
        if s.get("closed"):
            break
    d.wait(lambda: not b.scans.busy(), 60)
    tw = d.call("twin_state", json.dumps({"mode": "results", "trial": trial}))["scan"] or {}
    expect(b.twin.current() is None or (tw.get("state") == "ready" and not tw.get("simulated")
                                         and abs(tw.get("tre_max_mm", 0) - res["tre_max_mm"]) < 1e-3),
           f"scans: the twin on the measured motion of {trial} at once, not a simulated one")
    # the spheres' radii, once: three scans of the bench turned between them
    n = len(d.events)
    d.call("scans_capture", "calibration", "start", 0)
    got = d.wait(lambda: d.call("scans_state", "{}")["capture"]["calibration"]["good"] >= 3, 60)
    expect(got and d.call("scans_state", "{}")["capture"]["on"] == "calibration",
           "scans: a calibration run, three scans taken while it is on")
    a = d.call("scans_calibrate")
    done = d.wait(lambda: any(e.get("action") == "calibrated" for e in scans_events(n)), 60)
    d.wait(lambda: not b.scans.busy(), 5)
    st = d.call("scans_state", json.dumps({"trial": trial}))
    d.wait(lambda: not b.scans.busy(), 60)
    st = d.call("scans_state", json.dumps({"trial": trial}))
    cal = b.scans.calibration()
    expect(a.get("ok") and done and st["calibration"]["done"] and cal.is_file() and cal.parent.parent == b.roots.processed
           and st["result"].get("calibrated") and st["capture"]["on"] == "",
           f"scans: the radii calibrated (into the simulation's own file), the trial measured again with them")
    # repeatability: four scans of the bench left alone
    n = len(d.events)
    d.call("scans_capture", "repeat", "start", 4)
    got = d.wait(lambda: any(e.get("action") == "captured" and e.get("done") for e in scans_events(n)), 90)
    r = d.call("scans_state", "{}")["capture"]["repeat"]["result"]
    expect(got and r.get("n") == 4 and 0 < r.get("spread_mm", 1) < 0.05 and r.get("ok"),
           f"scans: repeatability over {r.get('n')} scans, spread {r.get('spread_mm')} mm at the worst target")
    # the day's processing
    n = len(d.events)
    d.call("scans_process", json.dumps({"day": "", "all": False}))
    done = d.wait(lambda: any(e.get("action") == "processed" for e in scans_events(n)), 90)
    poses = b.scans.store() / b.run().phase / scan_desk.POSES_NAME
    rows = []
    if poses.is_file():
        with open(poses, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    expect(done and len(rows) == 1 and rows[0]["trial"] == trial and rows[0]["calibrated"] == "1"
           and rows[0]["T_M_B_post"].count(" ") == 15,
           f"scans: today's trials processed into {poses.name} ({len(rows)} trial, calibrated, registrations)")
    b.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "impactguide_test")
    p.add_argument("--ui", type=Path, help="also run a trial through the window and save pictures here")
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--scale-check", help=argparse.SUPPRESS)
    args = p.parse_args()
    safe_output()
    os.environ.setdefault("IMPACTGUIDE_TWIN_AUTO", "0")     # the twin's model is never made by the tests' backends
    # the stand-in scanner's small files for the trials' tests; test_scans turns on the synthetic scans
    os.environ.setdefault(scans_page.SIM_SCANS_ENV, "stand_in")
    if not prepare_work(args.work):
        print(f"{args.work} is not empty and is not a work folder of this test: nothing deleted. Choose another "
              f"--work, or empty it by hand.", flush=True)
        return 2
    if args.scale_check:
        return scale_check(args.work, args.scale_check, args.ui or args.work)
    if args.ui:
        if not one_window_test():
            print("another window test of ImpactGuide still runs: this one runs all the same", flush=True)
        test_ui(args.work / "ui", args.ui)
        test_scales(args.work, args.ui)
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from slantui.shell.qt import QApplication
        app = QApplication(sys.argv[:1])
        test_backend(app, args.work / "backend", calibration=not args.no_calibration)
        test_scans(app, args.work / "scans")
    print(f"\n{len(FAILS)} failures" if FAILS else "\nall checks passed", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
