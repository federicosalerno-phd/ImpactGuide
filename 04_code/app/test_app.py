"""ImpactGuide test: complete trials simulated with stand-in files (T20.1), and the calibration (T5.1, T20.3).

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
          the streaming mode on frames of the live stream.
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
          Phone panel in the panel down the right (home and a module) and Wi-Fi
          paired with the QR code; then every step of every module and the
          Phone panel, with their Advanced parts open, are checked for words
          that are cut or wrap and for anything laid over the page, at the
          smallest window and at 1920 x 1032, in the six languages; last the
          board pixel for pixel at 125 and 150 % display scaling. The window is
          grabbed into DIR at each stage, and on every view in English and
          Italian (large) and in English and German (smallest).

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
import screen_board  # noqa: E402
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
    s = d.call("live_settings", json.dumps({"opacity": 0.55}))
    bad = d.call("live_settings", json.dumps({"opacity": 7}))
    expect(abs(s.get("opacity", -1) - 0.55) < 1e-9 and bad.get("opacity") == 1.0
           and ig.Prefs(d.backend.roots.prefs)["live"]["opacity"] == 1.0,
           "live view: the opacity of the virtual elements set through live_settings, kept, held between 0 and 1")
    d.call("live_settings", json.dumps({"opacity": 0.3}))
    folder = d.call("calib_folder_new", "check_photo")["folder"]
    n = len(d.events)
    d.call("calib_shoot", folder)
    stopped = d.wait(lambda: any(x["type"] == "live" and x.get("action") == "stopped" for x in d.events[n:]), 10)
    got = d.wait_event("calib_photo", timeout=20)
    expect(stopped and got is not None and not d.call("live_state")["running"],
           "live view: a photo from the phone stops it first, then the photo is taken")
    d.call("live_settings", json.dumps({"board": "sheet", "mandible": False}))

    # 10. the link to the phone: cable or Wi-Fi, pairing, a drop and its return, the phone's own screen
    test_link(d)

    # 11. the calibration board (in the page, or physical) and the guided photos, checked one by one, calibrated
    test_board(d)
    if calibration:
        test_series(d)
    d.backend.shutdown()


def png_image(data_url):
    """The grey image of a PNG data URL, as the page draws it."""
    import base64
    import cv2
    import numpy as np
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)


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
    img = png_image(a["png"]) if a.get("png") else None
    ref = screen_board.render(d.backend._lay)
    expect(a["shown"] and lay is not None and img is not None and img.shape == (820, 1500)
           and np.array_equal(img, ref) and abs(lay["square_mm"] - lay["square_px"] * lay["pitch_mm"]) < 1e-9
           and lay["marker_px"] % 7 == 0 and st["active"]["kind"] == "screen" and st["exact"],
           f"board on this monitor, in the page: {lay and lay['squares']} squares of {lay and lay['square_px']} px = "
           f"{lay and round(lay['square_mm'], 3)} mm at {mon and mon['pitch_mm']:.5f} mm ({mon and mon['source']}), "
           f"the image as large as the area and the same pixel for pixel")
    again = d.call("board_area", area(1500, 820, a["key"]))
    expect(again["key"] == a["key"] and "png" not in again, "the same area again: the same board, no image sent")
    # the area changes: laid out again, and the live view and the calibration take it at once
    b = d.call("board_area", area(1100, 820, a["key"]))
    live, cal = d.call("live_state"), d.call("calib_state")
    lay2 = b["state"]["layout"]
    expect(b.get("png") and lay2["squares"] != lay["squares"] and live["board"]["squares"] == lay2["squares"]
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
        return a, next((x for x in d.events[n:] if x["type"] == "calib_photo"), None) if e else None
    a, e = take()
    s = d.call("calib_state")
    first = s["series"]
    expect(e and e["check"]["ok"] and s["count"] == 1 and first["name"].endswith("_photo") and first["same"]
           and s["next"]["n"] == 1 and (raw / first["name"] / ig.PHOTO_CHECKS).is_file(),
           f"first photo: checked as it came and kept ({e and e['check']['corners']} corners, blur "
           f"{e and e['check']['blur_px']} px), in {first['name']}; the guide moves to the second pose")
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
    # the rest of the plan: enough at 12, full at 15
    for _ in range(14):
        take()
    s = d.call("calib_state")
    full = d.call("calib_take")
    expect(s["count"] == 15 and s["next"] is None and full.get("error") == "calib_full",
           f"the plan full at 15 photos ({s['count']}): no more taken")
    # a board changed after photos: the series is not of it, the next photo starts a new one
    d.call("board_area", area(1100, 820))
    s = d.call("calib_state")
    expect(s["series"]["n"] == 15 and not s["series"]["same"] and s["count"] == 0,
           "board laid out anew after the photos: the series is not of it, the count starts again")
    d.call("board_area", area(1500, 820))
    s = d.call("calib_state")
    expect(s["series"]["same"] and s["count"] == 15, "the board of the photos back (held while it fits): the series again")
    # the calibration: errors per photo and over the board, the intrinsics in plain units
    d.call("calib_run", "[]")
    e = d.wait_event("calib", timeout=600)
    s = d.call("calib_state")
    r = s["result"]
    expect(e and e["ok"] and r and not r["saved"] and r["verdict"] == "good" and r["n_views"] >= 14 and len(r["photos"]) == 15
           and r["residuals"]["corners"] and r["grid"] and abs(r["plain"]["fx"] / (3071.4 * 2040 / 4080) - 1) < 0.003,
           f"calibration: {r and r['verdict']}, RMS {r and r['rms_px']} px on {r and r['n_views']} of 15 photos, "
           f"focal {r and round(r['plain']['fx'], 1)} px (true {3071.4 * 2040 / 4080:.1f}), field of view "
           f"{r and round(r['plain']['fov_x'], 1)} deg, lens bend {r and round(r['plain']['bend_px'], 1)} px, "
           f"{r and len(r['residuals']['corners'])} corners over the board")
    worst = max(r["photos"], key=lambda p: p["err"])["name"]
    d.call("calib_run", json.dumps([worst]))
    d.wait_event("calib", timeout=600)
    r2 = d.call("calib_state")["result"]
    st = {p["name"]: p["state"] for p in r2["photos"]}
    expect(st[worst] == "excluded" and r2["n_views"] == r["n_views"] - (0 if worst in r.get("dropped", []) else 1)
           and len(r2["photos"]) == 15,
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
    d.call("calib_stream", True)
    up = d.wait(lambda: (lambda x: x["live"] and x["live"]["running"])(d.call("calib_state")), 30)
    d.wait(lambda: d.call("live_state").get("tracked"), 15)
    a, e = take()
    s = d.call("calib_state")
    frame = next(iter(sorted((raw / s["series"]["name"]).glob("frame_*.png"))), None) if s["series"] else None
    shape = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE).shape if frame else None
    live = d.call("live_state")
    expect(refused.get("error") == "stream_off" and up and e and e["check"]["ok"] and s["series"]["name"].endswith("_stream")
           and shape == (1440, 1920) and not live["layers"]["face"] and live["hide_board"],
           f"streaming mode: refused until the live video runs, then a frame of the stream kept "
           f"({e and e['check']['corners']} corners), saved whole as PNG {shape}, the picture plain (no face) with the "
           f"board on this monitor hidden in it")
    d.call("calib_stream", False)
    s = d.call("calib_state")
    expect(not s["live"]["running"], "the Photos step left: the live video stops")
    d.call("calib_set", json.dumps({"mode": "photo"}))
    phone.photo_px = b.run().config.photo_px
    for x in raw.iterdir():          # the series of this test go
        if x.name not in had:
            shutil.rmtree(x)
    b._pending_cal.clear()
    (b.roots.calib_out / "intrinsics_photo.json").unlink(missing_ok=True)


def test_link(d):
    """The Phone panel against the simulated phone (phone_link.SimulatedAdb): the link found by itself and followed
    (no phone, the cable, Wi-Fi when the cable is out, the cable again), one place for the serial, Wi-Fi paired only
    when asked for (with the cable in, Wireless debugging switched on from the PC and a QR code), pairing with a code,
    a Wi-Fi drop and the reconnection, plain words for every failure, the live view sent to the phone's browser and
    its added latency."""
    import re
    import threading
    import live
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
    + ' .guide-side-l, .guide-key span, .ph-last-t, .bd-ask-hd, .bd-steps-hd, .bd-step-n, .fig-note, .bd-tool, .guide-d';
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
    ("cal_board_ruler", "__adv('calBdRuler', true); BOARDVIEW.label()", 900, True),
    ("cal_plate", "__adv('calBdRuler', false); BOARD.set({kind: 'plate'})", 2200, True),
    ("cal_plate_short", "__adv('calBdShort', true)", 900, True),
    ("cal_sheet", "__adv('calBdShort', false); BOARD.set({kind: 'sheet'})", 1300, True),
    ("cal_photos_phys", "go('calibration', 'photos')", 1500, True),
    ("cal_photos", "BOARD.set({kind: 'screen'}); go('calibration', 'photos')", 1800, True),
    ("cal_photos_adv", "__adv('calPhAdv', true)", 900, True),
    ("cal_stream", "__adv('calPhAdv', false); CAL.set({mode: 'stream'})", 5000, True),
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
    ("results", "__closeAdv(); go('results', 'scans')", 900, True),
]
SIZES = (("min",) + ig.MIN_WINDOW, ("large", 1920, 1032))
SHOTS = {("large", "en"), ("large", "it"), ("min", "en"), ("min", "de")}
SIM_PHOTO = (2040, 1536)        # the simulated phone's photos in the window test: half the phone's, the same checks
# the window at each display scaling, in Qt's pixels: as large as the 1920 x 1080 screen holds (1750 x 1000 and
# 1860 x 1020 physical), so the board fits beside the panel
SCALE_WINDOW = {"1.25": (1400, 800), "1.5": (1240, 680)}


def Qt_key(name):
    from slantui.shell.qt import Qt
    return getattr(Qt.Key, "Key_" + name)


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
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
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
    shots.mkdir(parents=True, exist_ok=True)
    app, win, backend = ig.build(ig.parse(["--simulate", str(work), "--lang", "en"]))
    win.setGeometry(0, 0, 1440, 900)        # on the monitor, whole: the board must lie on one monitor
    log = []
    taken = set()
    misfits = []       # (size, language, state, what) for every word that does not fit
    board = {"grabs": []}

    def grab_board(name):
        same, found, corners, where = board_on_screen(app, win, backend, shots / f"{name}.png")
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
            if v.startswith("waiting scan_az40") and "ui_04_scan_wait" not in taken:
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
            ("-resize", lambda: win.setGeometry(0, 0, 1700, 960), 2500, None),
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
             ("plate", "JSON.stringify([BOARD.state.kind, !!PLATE3D.model, !$('bd3d').classList.contains('hidden')])")),
            ("cal_13c_readings", "var i = $('calBdLong').querySelectorAll('input'); i[0].value = '220.05';"
                                 " i[1].value = '220.02'; i[2].value = '220.11'; i[2].dispatchEvent(new Event('change'))", 1500,
             ("readings", "JSON.stringify(BOARD.state.physical.plate.long)")),
            ("cal_13d_turned", lambda: turn_plate(), 1200, None),
            ("cal_13e_sheet", "document.querySelector('#calBdWhich .mitem[data-kind=sheet]').click()", 1500, None),
            ("-back_screen", "document.querySelector('#calBdKinds .mitem[data-kind=screen]').click()", 2000, None),
            # the photos, one after the other with the guide, each checked; then the calibration
            ("cal_14_photos", "go('calibration', 'photos')", 2500,
             ("photos_board", "JSON.stringify([BOARDVIEW.host && BOARDVIEW.host.id, BOARDVIEW.key])")),
            ("board_14_photos", "", 300, None),
            ("-take", lambda: None, 0, ("take", 12)),
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
            ("ui_29_results", "go('results', 'analysis')", 1200, None),
        ]
        # the sweep of every module and panel, at two sizes and in six languages
        for size, w, h in SIZES:
            steps.append((f"-size_{size}", (lambda w=w, h=h: win.setGeometry(0, 0, w, h)), 1200, None))
            for code in ig.LANGS:
                steps.append((f"-lang_{size}_{code}", lang(code), 1500, None))
                for n, (state, script, wait, pic) in enumerate(SWEEP):
                    shot = pic and (size, code) in SHOTS
                    steps.append(((f"{size}_{code}_{n:02d}_{state}" if shot else f"-{size}_{code}_{state}"), script, wait,
                                  ("fit", "__fit()", size, code, state)))
        steps.append(("-restore", lambda: win.setGeometry(0, 0, 1440, 900), 800, None))

        def turn_plate():
            js("PLATE3D.turned = true; PLATE3D.view.az += 0.9; PLATE3D.view.el = 0.6; PLATE3D.draw();")

        def measured(what, v):
            if what[0] == "fit":
                try:
                    for m in json.loads(v):
                        misfits.append((what[2], what[3], what[4], m))
                except (TypeError, ValueError):
                    misfits.append((what[2], what[3], what[4], f"no answer ({v})"))
            else:
                log.append(what[0] + " " + str(v))

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
                else:
                    js(what[1], lambda v: (measured(what, v), nxt()))
            if what is not None and what[0] == "calib":
                QTimer.singleShot(wait, lambda: wait_calib(nxt, "calib_" + what[1]))
            else:
                QTimer.singleShot(wait, after)
        nxt()

    def first():
        js("window.__errs=[];window.addEventListener('error',function(e){window.__errs.push(e.message+' line '+e.lineno)});")
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
    expect(resized and grabs[0][4] != grabs[1][4] and board.get("geo") == [0, 0, 1700, 960],
           f"window: the window resized by hand, the board laid out again for its new area ({grabs[0][4]} -> "
           f"{grabs[1][4]}); nothing moved or resized the window ({board.get('geo')})")
    ruler = read("ruler")
    expect(ruler == "ruler", f"window: the ruler check typed in the page corrected the size ({ruler})")
    plate, readings = read("plate"), read("readings")
    expect(plate == ["plate", True, True] and readings and abs(readings["mean_mm"] - 220.06) < 1e-6
           and abs(readings["spread_mm"] - 0.09) < 1e-6,
           f"window: the physical board, the 3D-printed plate in 3D, three caliper readings with their mean and spread "
           f"({plate}, {readings})")
    photos = read("photos")
    expect(photos and photos["n"] >= 12 and photos["last"],
           f"window: photos taken with the page's Take button, each checked as it came ({photos})")
    c1, c2 = read("calib_first"), read("calib_again")
    leave, saved = read("leave"), read("saved")
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
                        f"{SIZES[0][1]} x {SIZES[0][2]} and {SIZES[1][1]} x {SIZES[1][2]}, in the six languages "
                        f"({checked} views, {len(misfits)} words that do not fit)")
    expect(len(taken) >= 10, f"window: {len(taken)} pictures in {shots}")
    print("page actions: " + ", ".join(x for x in log if not x.startswith(("confirm", "shoot", "record", "stop",
                                                                            "recording", "waiting", "disabled"))),
          flush=True)


def scale_check(work, scale, out):
    """The board in the page at a display scaling (Qt's own, as Windows does at 125 or 150 %: QT_SCALE_FACTOR, set
    before Qt starts): the window put at whole physical pixels, the board as the screen shows it against the page's
    image, pixel for pixel. Prints one JSON line."""
    from slantui.shell.qt import QTimer
    app, win, backend = ig.build(ig.parse(["--simulate", str(work), "--lang", "en"]))
    f = float(scale)
    win.setGeometry(0, 0, *SCALE_WINDOW[scale])
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
    it starts)."""
    import subprocess
    for scale in ("1.25", "1.5"):
        env = dict(os.environ, QT_SCALE_FACTOR=scale)
        for attempt in range(2):        # once more if the screen was not the app's (the computer in use meanwhile)
            r = subprocess.run([sys.executable, str(Path(__file__)), "--scale-check", scale, "--work",
                                str(work / f"scale_{scale}"), "--ui", str(shots)], capture_output=True, text=True,
                               env=env, timeout=180)
            line = next((x for x in r.stdout.splitlines() if x.startswith("{")), "{}")
            d = json.loads(line)
            if d.get("identical") and d.get("corners_found") == d.get("corners"):
                break
        expect(d.get("identical") and d.get("corners_found") == d.get("corners") and d.get("dpr") == float(scale),
               f"window at {int(float(scale) * 100)} % display scaling: the board on the screen as the page drew it, "
               f"pixel for pixel, every corner found ({line}{', second run' if attempt else ''})")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "impactguide_test")
    p.add_argument("--ui", type=Path, help="also run a trial through the window and save pictures here")
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--scale-check", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.scale_check:
        if args.work.exists():
            shutil.rmtree(args.work)
        return scale_check(args.work, args.scale_check, args.ui or args.work)
    if args.work.exists():
        shutil.rmtree(args.work)
    if args.ui:
        test_ui(args.work / "ui", args.ui)
        test_scales(args.work, args.ui)
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from slantui.shell.qt import QApplication
        app = QApplication(sys.argv[:1])
        test_backend(app, args.work / "backend", calibration=not args.no_calibration)
    print(f"\n{len(FAILS)} failures" if FAILS else "\nall checks passed", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
