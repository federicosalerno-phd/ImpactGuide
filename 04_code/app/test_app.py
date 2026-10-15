"""ImpactGuide test: complete trials simulated with stand-in files (T20.1).

backend   the slots the page calls, without a window: a trial from the first
          step to the log row, a cap shift over the threshold and the
          reposition that follows, the app closed in the middle of a trial and
          resumed, a drape trial with photos and video with and without the
          drape, a correction of a logged value, a void trial, a new
          configuration, the completeness check of trials.py, the live AR view
          on a synthetic stream (both boards, a photo that stops it), the
          calibration and the session check on synthetic board photos.
--ui DIR  the same through the real window: the page's own buttons are
          pressed, one trial is run to its log row, the other views are opened,
          the live view is started from its page and left, and the window is
          grabbed into DIR at each stage, in English and in Italian.

    python test_app.py [--work DIR] [--ui SHOTS] [--no-calibration]

Exit code 1 on failure. Everything is written under --work (default: a folder
in the system temp), never in the project's data.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import impactguide_app as ig  # noqa: E402  (sets the bytecode cache and the path to 04_code/src)

import calibrate_charuco  # noqa: E402
import trials  # noqa: E402

FAILS = []


def expect(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def test_backend(app, work, calibration=True):
    d = Driver(app, work)
    run = lambda: d.backend.run()  # noqa: E731
    s = d.call("trials_state")
    expect(s["trial"]["trial"] == "t001" and s["totals"]["slots"] == 128 and not s["trial"]["open"],
           "first plan: t001, 128 slots, not started")
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
    d.backend.shutdown()
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
    order = [k for k in keys if k in ("drape_video", "drape_off", "photo_az40", "video", "scan_az40")]
    expect(order == ["drape_video", "drape_off", "photo_az40", "video", "scan_az40"]
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
    expect(not problems and not R["orphans"] and R["extra"] == [t4],
           f"completeness: no problems, no stray folders, outside the matrix {R['extra']}")

    # 9. live AR on the synthetic stream: on, board tracked, the other board, a photo stops it first
    s = d.call("live_state")
    expect(not s["running"] and s["registration"]["source"] == "demo" and s["intrinsics"]["source"] == "nominal",
           "live view: off at the start, demo registration, nominal intrinsics")
    d.call("live_start")
    e = d.wait_event("live", timeout=30)
    tracked = d.wait(lambda: (lambda x: x.get("tracked") and x.get("fps_ar", 0) > 0)(d.call("live_state")), 15)
    s = d.call("live_state")
    expect(e is not None and e["action"] == "started" and tracked and s["fps_ar"] > 0
           and s["corners"] == s["board"]["corners"] and s["drawn_px"].get("face", 0) > 0,
           f"live view: started, sheet tracked ({s.get('corners')} corners, RMS {s.get('rms_px')} px), "
           f"face drawn, {s.get('fps_ar')} fps")
    d.call("live_settings", json.dumps({"board": "patient", "mandible": True}))
    tracked = d.wait(lambda: (lambda x: x.get("tracked") and x["board"]["name"] == "patient")(d.call("live_state")), 15)
    s = d.call("live_state")
    expect(tracked and s["registration"]["demo"] == "design" and s["layers"]["mandible"],
           f"live view: patient board tracked after the switch, bench design registration, mandible on")
    folder = d.call("calib_folder_new", "check_video")["folder"]
    n = len(d.events)
    d.call("calib_shoot", folder)
    stopped = d.wait(lambda: any(x["type"] == "live" and x.get("action") == "stopped" for x in d.events[n:]), 10)
    got = d.wait_event("calib_photo", timeout=20)
    expect(stopped and got is not None and not d.call("live_state")["running"],
           "live view: a photo from the phone stops it first, then the photo is taken")
    d.call("live_settings", json.dumps({"board": "sheet", "mandible": False}))

    # 10. the calibration and the session check, on synthetic photos of the board
    if calibration:
        s = d.call("calib_folder_new", "photo")
        folder = work / "raw" / "calibration" / s["folder"]
        print("      writing 12 synthetic calibration photos...", flush=True)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            calibrate_charuco.synth(folder, 12, seed=3)
        (folder / "truth.json").unlink()
        d.call("calib_run", s["folder"], "photo")
        e = d.wait_event("calib", timeout=600)
        expect(e is not None and e["ok"] and e["result"]["n_views"] >= 8,
               f"calibration: RMS {e and e.get('result', {}).get('rms_px')} px on {e and e.get('result', {}).get('n_views')} photos")
        s = d.call("calib_save", "photo")
        expect(s["modes"]["photo"]["stored"] is not None and (work / "processed" / "calibration" / "intrinsics_photo.json").is_file(),
               "calibration saved as intrinsics_photo.json")
        s = d.call("calib_folder_new", "check_photo")
        folder = work / "raw" / "calibration" / s["folder"]
        with contextlib.redirect_stdout(io.StringIO()):
            calibrate_charuco.synth(folder, 8, seed=4)
        (folder / "truth.json").unlink()
        d.call("calib_check", s["folder"], "photo")
        e = d.wait_event("calib", timeout=600)
        checks = trials.read_csv(work / "processed" / "calibration" / "session_checks.csv")
        expect(e is not None and e["ok"] and len(checks) == 1 and checks[0]["ok"] == "1",
               "session check: the stored calibration is still valid, logged in session_checks.csv")
    d.backend.shutdown()


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


def test_ui(work, shots):
    from slantui.shell.qt import QTimer
    shots.mkdir(parents=True, exist_ok=True)
    app, win, backend = ig.build(ig.parse(["--simulate", str(work), "--lang", "en", "--page", "trials"]))
    log = []
    plan = []          # (name, js, wait ms), run in order
    taken = set()

    def grab(name):
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
            if v.startswith("confirm photo_az40") and "photo" not in taken:
                pass
            if v.startswith("closed"):
                finish()
                return
            # pictures at the moments that show the most
            if v.startswith("shoot drape_photo_az40") or v.startswith("shoot photo_az40"):
                QTimer.singleShot(1400, lambda: grab("ui_02_photo_received" if "drape" not in v else "ui_03_drape_photo"))
            if v.startswith("waiting scan_az40") and "ui_04_scan_wait" not in taken:
                grab("ui_04_scan_wait")
            if v == "start":
                QTimer.singleShot(900, lambda: grab("ui_01_trial_started"))
            QTimer.singleShot(2200 if v.startswith(("shoot", "stop")) else 700, drive)
        js(DRIVE, got)

    def finish():
        steps = [
            ("ui_05_closed", "", 900),
            ("ui_06_log", "TRIALS.setView('log')", 1200),
            ("ui_07_sequence", "TRIALS.setView('sequence')", 1200),
            ("ui_08_config", "TRIALS.setView('config')", 1200),
            ("ui_08b_home", "document.querySelector('.tbar-logo-btn').click()", 1200),
            ("ui_09_calibration", "go('calibration')", 1500),
            ("ui_10_it_guide", "$('lang').value='it'; $('lang').dispatchEvent(new Event('change')); go('trials'); TRIALS.setView('guide')", 1800),
            ("ui_11_it_calibration", "go('calibration')", 1500),
            ("ui_12_it_live", "go('live')", 1500),
            ("ui_13_it_live_on", "$('lvStart').click()", 7000),
            ("ui_14_live_patient", "$('lang').value='en'; $('lang').dispatchEvent(new Event('change'));"
                                   "LIVE.settings({board: 'patient', mandible: true})", 4000),
            ("ui_15_live_left", "window.__live = LIVE.state && LIVE.state.tracked; go('trials')", 1500),
        ]

        def nxt():
            if name_done and name_done[-1] == "ui_08b_home":
                js("JSON.stringify([page, trialsView, !!document.querySelector('.tbar-logo-btn')])",
                   lambda v: log.append("home " + str(v)))
            if not steps:
                js("JSON.stringify([!!window.__live, !!(LIVE.state && LIVE.state.running)])",
                   lambda v: log.append("live " + str(v)))
                js("JSON.stringify(window.__errs||[])", lambda v: (log.append("errors " + str(v)), app.quit()))
                return
            name, script, wait = steps.pop(0)
            if script:
                js(script)
            QTimer.singleShot(wait, lambda: (grab(name), name_done.append(name), nxt()))
        name_done = []
        nxt()

    def first():
        js("window.__errs=[];window.addEventListener('error',function(e){window.__errs.push(e.message+' line '+e.lineno)});")
        grab("ui_00_plan")
        # the drape slot comes first, so the run shows both kinds of views
        drive_drape()

    def drive_drape():
        run = backend.run()
        slot = next(r for r in run.pending() if r["drape"] == "1")
        js("TRIALS.extraOpen=true; TRIALS.renderPlan(TRIALS.state);"
           f"$('exHolder').value='{slot['holder']}'; $('exAngle').value='{slot['angle_deg']}';"
           f"$('exDir').value='{slot['direction_deg']}'; $('exDrape').checked=true; $('exApply').click();")
        QTimer.singleShot(1200, lambda: (grab("ui_00_plan_drape"), drive()))

    QTimer.singleShot(4000, first)
    QTimer.singleShot(400000, app.quit)
    app.run()
    errors = next((x for x in log if x.startswith("errors")), "errors none")
    run = trials.Run("campaign", work / "raw", work / "protocol")
    rows = run.log.rows()
    expect(rows and rows[-1]["status"] == "ok" and rows[-1]["drape"] == "1" and not run.missing(rows[-1]["trial"]),
           f"window: drape trial {rows[-1]['trial'] if rows else '?'} run with the page's buttons, nothing missing")
    expect(errors in ("errors []", "errors none"), f"window: no script errors ({errors})")
    home = next((x for x in log if x.startswith("home")), "home ?")
    expect(home == 'home ["trials","guide",true]',
           f"window: the mark on the band leads back to the trial guide and stays a raised button there ({home})")
    live = next((x for x in log if x.startswith("live")), "live ?")
    expect(live == "live [true,false]",
           f"window: live view started from the page, board tracked, stopped when the page is left ({live})")
    expect(len(taken) >= 10, f"window: {len(taken)} pictures in {shots}")
    print("page actions: " + ", ".join(log[:-1]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "impactguide_test")
    p.add_argument("--ui", type=Path, help="also run a trial through the window and save pictures here")
    p.add_argument("--no-calibration", action="store_true")
    args = p.parse_args()
    if args.work.exists():
        shutil.rmtree(args.work)
    if args.ui:
        test_ui(args.work / "ui", args.ui)
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from slantui.shell.qt import QApplication
        app = QApplication(sys.argv[:1])
        test_backend(app, args.work / "backend", calibration=not args.no_calibration)
    print(f"\n{len(FAILS)} failures" if FAILS else "\nall checks passed", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
