"""ImpactGuide test of Analysis and Charts, the second and third steps of Scans and results.

backend   in simulation, without a window: one trial run through the slots as the page runs it, then the analysis of
          the phase in a worker (the trial done and every slot still to do, simulated and said so), its tables
          written into the simulation's results folder and never into the project's, the same analysis again
          writing nothing; the tables a chart reads, a free figure name, every chart to start from drawn as an SVG
          for the page, the newest preview only when several are asked at once, a chart that cannot be drawn said
          by its code, the export (PDF, SVG, PNG at 300 dpi), the pin (the figure's script written and run on its
          own, the figure made; a script of the same name replaced only when asked), the pinned chart offered
          again, a folder opened; no trace or exception text in any answer or event.
--ui DIR  the same pages in the real window (out of the way, right of every screen): the analysis started by the
          step itself, its breakdowns, the chart preview on its paper; every view in the six languages at the
          smallest window and at 1920 x 1032, measured for words cut or wrapped as test_app.py measures them, and a
          picture of each into DIR.

    python test_results.py [--work DIR] [--ui SHOTS]

Exit code 1 on failure. Everything is written under --work (default: a folder in the system temp).
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_app as ta  # noqa: E402  (the app, its paths, the window helpers and the measure of the words)

ig = ta.ig
expect = ta.expect

PRESET_KEYS = ("tre_energy", "tre_direction", "displacement", "tre_targets", "share_2mm", "cumulative", "score",
               "torque_a", "reseat_b")


def clean_text(x):
    """No trace or exception text anywhere in an answer or an event."""
    s = json.dumps(x)
    return "Traceback" not in s and "Error:" not in s and 'File "' not in s


def test_backend(app, work):
    import trials
    project_results = trials.ROOT / "05_results"
    before = {p.name: p.stat().st_mtime_ns for p in project_results.glob("*.csv")}
    d = ta.Driver(app, work)
    b = d.backend
    s = d.call("analysis_state")
    expect(s.get("result") is None and not s.get("running") and s.get("simulate"),
           "analysis: none yet in a new simulation")
    trial, _ = d.run_trial(mass="137.2")
    a = d.call("analysis_run")
    expect(a.get("started"), "analysis started in a worker")
    again = d.call("analysis_run")
    expect(again.get("started") is False, "a second Update while it runs starts nothing")
    ev = d.wait_event("analysis", timeout=240)
    expect(ev is not None and ev.get("action") == "ready", f"analysis ready ({ev})")
    s = d.call("analysis_state")
    r = s.get("result") or {}
    expect(r.get("n") == 128 and r.get("simulated") and r.get("n_drape") == 20,
           f"the campaign of 128 trials (the one done and the simulated rest), said as simulated ({r.get('n')})")
    res = work / "results"
    names = sorted(p.name for p in res.glob("*.csv"))
    expect({"trials_metrics.csv", "trials_tre.csv", "impact.csv", "stats.csv", "mixed_model.csv", "detection.csv",
            "correction.csv", "uncertainty.csv", "interfaces.csv"} <= set(names),
           f"tables in the simulation's results folder ({len(names)})")
    import csv
    with open(res / "trials_metrics.csv", newline="", encoding="utf-8") as f:
        rows = {row["trial"]: row for row in csv.DictReader(f)}
    expect(trial in rows and rows[trial]["simulated"] == "1", f"the trial run through the page is in the analysis "
                                                              f"({trial}), its stand-in scans said simulated")
    after = {p.name: p.stat().st_mtime_ns for p in project_results.glob("*.csv")}
    expect(after == before, "nothing written into the project's results folder")
    d.call("analysis_run")
    ev = d.wait_event("analysis", timeout=240)
    expect(ev and ev.get("action") == "ready" and ev.get("written") == [], "the same analysis again writes nothing")

    st = d.call("chart_state")
    expect(st.get("simulate") and {"trials_tre.csv", "trials_metrics.csv", "trials_targets.csv", "interfaces_a.csv",
                                   "interfaces_b.csv"} <= set(st.get("tables", {}))
           and [p["key"] for p in st.get("presets", [])] == list(PRESET_KEYS) and st.get("widths") == [90, 140, 190],
           "charts: the tables, the charts to start from, the page widths")
    name = d.call("chart_name", "tre energy").get("name")
    expect(name == "figS1_tre_energy", f"a free figure name ({name})")
    shown = []
    for i, p in enumerate(st["presets"]):
        n = len(d.events)
        d.call("chart_preview", json.dumps({"spec": dict(p["spec"], name=name), "id": i + 1}))
        d.wait(lambda: any(e["type"] == "chart" and e.get("id") == i + 1 for e in d.events[n:]), 60)
        e = next((e for e in d.events[n:] if e["type"] == "chart" and e.get("id") == i + 1), {})
        if e.get("action") == "preview" and e.get("svg", "").startswith("data:image/svg+xml;base64,"):
            shown.append(p["key"])
        elif e:
            print("     ", p["key"], e.get("error"))
    expect(shown == list(PRESET_KEYS), f"every chart drawn for the page ({len(shown)} of {len(PRESET_KEYS)})")
    n = len(d.events)
    for i in range(3):
        d.call("chart_preview", json.dumps({"spec": dict(st["presets"][0]["spec"], name=name), "id": 100 + i}))
    d.wait(lambda: any(e["type"] == "chart" and e.get("id") == 102 for e in d.events[n:]), 60)
    ids = [e.get("id") for e in d.events[n:] if e["type"] == "chart"]
    expect(102 in ids and 101 not in ids, f"three previews asked at once: the newest drawn, the one between left ({ids})")
    n = len(d.events)
    d.call("chart_preview", json.dumps({"spec": dict(st["presets"][0]["spec"], kind="pie"), "id": 200}))
    d.wait(lambda: any(e["type"] == "chart" and e.get("id") == 200 for e in d.events[n:]), 30)
    e = next((e for e in d.events[n:] if e.get("id") == 200), {})
    expect(e.get("action") == "failed" and e.get("error") == "bad_spec" and clean_text(e),
           f"a chart that cannot be drawn is said by its code ({e.get('error')})")
    spec = dict(st["presets"][0]["spec"], name=name)
    d.call("chart_export", json.dumps({"spec": spec}))
    e = d.wait_event("chart", timeout=60)
    files = sorted((work / "charts").glob(name + ".*"))
    expect(e and e.get("action") == "exported" and [f.suffix for f in files] == [".pdf", ".png", ".svg"],
           f"exported: {', '.join(f.name for f in files)}")
    d.call("chart_pin", json.dumps({"spec": spec}))
    e = d.wait_event("chart", timeout=120)
    script = work / "figures" / f"{name}.py"
    expect(e and e.get("action") == "pinned" and script.is_file() and (work / "pictures" / f"{name}.pdf").is_file()
           and (work / "pictures" / f"{name}.png").is_file() and not list((work / "figures").glob("__pycache__")),
           "pinned: the figure's script written and run on its own, PDF and PNG made")
    a = d.call("chart_pin", json.dumps({"spec": dict(spec, kind="box")}))
    expect(a.get("error") == "script_exists", "a script of the same name: asked first, never replaced at once")
    d.call("chart_pin", json.dumps({"spec": dict(spec, kind="box"), "replace": True}))
    e = d.wait_event("chart", timeout=120)
    saved = {x["name"]: x["spec"] for x in d.call("chart_state").get("saved", [])}
    expect(e and e.get("replaced") and saved.get(name, {}).get("kind") == "box",
           "replaced when asked; the pinned chart offered again with its spec")
    with ta.Opened() as opened:
        d.call("results_open", "results")
        d.call("results_open", "nowhere")
    expect([p.name for p in opened.paths] == ["results"], "the results folder opened; an unknown one refused")
    expect(all(clean_text(e) for e in d.events if e["type"] in ("analysis", "chart")),
           "no trace or exception text in the events of the two pages")
    b.shutdown()


def test_holders(app, work):
    """The Holders step of Trials: A's masses one press at a time into a_slip_torque.csv (a slip ends the repeat,
    the last press taken back), B's reseating run of the Scans page on synthetic scans (each measured as it comes,
    each seating against the first), and the analysis reading both."""
    import interfaces
    d = ta.Driver(app, work)
    s = d.call("holders_state")
    expect(s["a"]["repeat"] == 1 and s["a"]["n"] == 0 and s["b"]["n"] == 0 and not s["b_on"],
           "holders: nothing measured in a new simulation")
    for mass, slipped in ((300, False), (340, False), (355, True), (330, False), (362, True)):
        s = d.call("holders_a", json.dumps({"mass_g": str(mass), "arm_mm": "60", "slipped": slipped}))
    a_file = work / "raw" / "interfaces" / interfaces.A_FILE
    expect(s["a"]["n"] == 2 and s["a"]["repeat"] == 3 and a_file.is_file()
           and abs(s["a"]["mean_nm"] - (interfaces.torque_nm(355, 60) + interfaces.torque_nm(362, 60)) / 2) < 1e-4,
           f"A: five presses, two slips, the torque live ({s['a']['mean_nm']} N m), written into {a_file.name}")
    s = d.call("holders_a_undo")
    expect(s["a"]["n"] == 1 and s["a"]["repeat"] == 2, "A: the last press taken back")
    bad = d.call("holders_a", json.dumps({"mass_g": "abc", "arm_mm": "60", "slipped": False}))
    expect(bad.get("error") == "bad_value" and clean_text(bad), "A: a mass that is not a number refused by its code")
    old = os.environ.get(ta.scans_page.SIM_SCANS_ENV)
    os.environ[ta.scans_page.SIM_SCANS_ENV] = "synthetic"
    try:
        s = d.call("holders_b", "start", 4)
        expect(s.get("b_on"), "B: the reseating run started on the Scans page's run")
        d.wait(lambda: d.call("holders_state")["b"]["scans"] >= 4, 240)
        d.wait(lambda: not d.call("holders_state")["measuring"], 120)
        s = d.call("holders_state")
        expect(s["b"]["scans"] == 4 and s["b"]["n"] == s["b"]["good"] - 1 >= 2 and s["b"]["median_mm"] is not None
               and s["b"]["median_mm"] < 0.5, f"B: four scans measured as they came, {s['b']['n']} seatings against "
                                              f"the first, median {s['b']['median_mm']} mm")
        s = d.call("holders_b", "stop", 0)
        expect(not s["b_on"] and (work / "processed" / "ground_truth" / "reseat.json").is_file(),
               "B: the run stopped, kept in reseat.json")
    finally:
        if old is None:
            os.environ.pop(ta.scans_page.SIM_SCANS_ENV, None)
        else:
            os.environ[ta.scans_page.SIM_SCANS_ENV] = old
    d.call("analysis_run")
    ev = d.wait_event("analysis", timeout=240)
    r = d.call("analysis_state").get("result") or {}
    h = r.get("interfaces") or {}
    expect(ev and ev.get("action") == "ready" and (h.get("a") or {}).get("n") == 1
           and (h.get("b") or {}).get("n") == s["b"]["n"] and h.get("simulated_parts") == [],
           "the analysis reads the holders measured through the page (none of them simulated)")
    d.backend.shutdown()


# ── the window ──────────────────────────────────────────────────────────────
VIEWS = (
    ("analysis", "LINK.hide(); __closeAdv(); go('results', 'analysis')", 1500),
    ("analysis_target", "ANALYSIS.by = 'target'; ANALYSIS.render()", 700),
    ("analysis_adv", "ANALYSIS.by = 'energy'; ANALYSIS.render(); __adv('anAdv', true)", 800),
    ("charts", "__closeAdv(); go('results', 'charts')", 2600),
    ("charts_adv", "__adv('chAdv', true)", 900),
    ("charts_box", "__adv('chAdv', false); $('chFrom').value = 'p:tre_targets'; $('chFrom').dispatchEvent("
                   "new Event('change'))", 2600),
    ("holders_a", "__closeAdv(); go('trials', 'holders'); HOLDERS.which = 'a'", 1500),
    ("holders_a_adv", "__adv('hoAdv', true)", 800),
    ("holders_b", "__adv('hoAdv', false); HOLDERS.which = 'b'; HOLDERS.render()", 900),
)


def test_ui(work, shots, only=None):
    from slantui.shell.qt import QTimer
    shots.mkdir(parents=True, exist_ok=True)
    app, win, backend = ta.build_window(["--simulate", str(work), "--lang", "en"])
    away = ta.away_origin(app)
    log, misfits, taken = [], [], []
    views = [v for v in VIEWS if not only or v[0].startswith(only)]
    # a few masses on A, so its step shows a repeat in progress and repeats done
    import interfaces
    for mass, slipped in ((300, False), (340, False), (355, True), (330, False)):
        interfaces.record_a(work / "raw" / "interfaces", mass, 60, slipped)

    def place(w, h):
        if win.isMinimized():
            win.showNormal()
        win.setGeometry(*away, w, h)
    place(1440, 900)

    def js(script, cb=None):
        win.run_js(script, cb or (lambda v: None))

    def grab(name):
        img = win.grabWindow()
        for _ in range(10):
            if img.width() >= win.width():
                break
            app.processEvents()
            img = win.grabWindow()
        img.save(str(shots / f"{name}.png"))
        taken.append(name)

    steps = [("-ready", "", 0, ("until", "!!(typeof Bridge !== 'undefined' && Bridge.ready && typeof page !== 'undefined' && "
                                     "$('lang') && $('lang').value)", 90)),
             ("-fit", ta.FIT, 1500, None),
             ("res_01_analysis_running", "go('results', 'analysis')", 1500, None),
             ("-wait_analysis", "", 0, ("until", "!!(ANALYSIS.state && ANALYSIS.state.result)", 240)),
             ("res_02_analysis", "ANALYSIS.render()", 900,
              ("state", "(function () { const r = ANALYSIS.state && ANALYSIS.state.result; return JSON.stringify(r ?"
                        " [r.n, r.simulated, !$('anSim').classList.contains('hidden'), $('anGrid').children.length]"
                        " : null); })()")),
             ("res_03_charts", "go('results', 'charts')", 600, ("until", "!!(CHARTS.shown && $('chImg').naturalWidth > 0)", 60)),
             ("-charts_state", "", 300, ("state", "JSON.stringify([CHARTS.spec && CHARTS.spec.name, CHARTS.shown ?"
                                                  " CHARTS.shown.width_mm : null, $('chImg').naturalWidth,"
                                                  " !$('chSim').classList.contains('hidden')])"))]
    for size, w, h in ta.SIZES:
        steps.append((f"-size_{size}", (lambda w=w, h=h: place(w, h)), 1200, None))
        for code in ig.LANGS:
            steps.append((f"-lang_{size}_{code}", f"$('lang').value='{code}'; $('lang').dispatchEvent(new Event('change'));",
                          1500, None))
            for n, (view, script, wait) in enumerate(views):
                steps.append((f"{size}_{code}_{n:02d}_{view}", script, wait, ("fit", size, code, view)))
    steps.append(("-restore", lambda: place(1440, 900), 600, None))

    size_of = {name: (w, h) for name, w, h in ta.SIZES}

    def measured(what, v, again):
        try:
            found, later = json.loads(v), json.loads(again)
        except (TypeError, ValueError):
            misfits.append((what[1], what[2], what[3], f"no answer ({v})"))
            return
        for m in found:
            if m in later:
                misfits.append((what[1], what[2], what[3], m))

    def nxt():
        if not steps:
            js("JSON.stringify(window.__errs||[])", lambda v: (log.append("errors " + str(v)), app.quit()))
            return
        name, script, wait, what = steps.pop(0)
        if callable(script):
            script()
        elif script:
            js(script)

        def after():
            if what is not None and what[0] == "until":
                t0 = time.monotonic()

                def check(v=None):
                    if v is True or v == "true":
                        log.append(f"{name} ok")
                        nxt()
                    elif time.monotonic() - t0 > what[2]:
                        log.append(f"{name} timeout")
                        nxt()
                    else:
                        QTimer.singleShot(500, lambda: js(what[1], check))
                check()
                return
            if not name.startswith("-"):
                grab(name)
            if what is None:
                nxt()
            elif what[0] == "fit":
                want = size_of[what[1]]
                if (win.width(), win.height()) != want:
                    place(*want)
                js("__fit()", lambda v: QTimer.singleShot(ta.FIT_AGAIN_MS, lambda: js(
                    "__fit()", lambda v2: (measured(what, v, v2), nxt()))))
            else:
                js(what[1], lambda v: (log.append(f"{what[0]} {v}"), nxt()))
        QTimer.singleShot(wait, after)

    QTimer.singleShot(1000, nxt)
    app.run()
    for line in log:
        print("      " + line)
    expect("-ready ok" in log, "window: the page up before the test drives it")
    states = [line for line in log if line.startswith("state ")]
    a = json.loads(states[0][6:] or "null") if states else None
    expect(a is not None and a[0] == 128 and a[1] and a[2] and a[3] >= 8,
           f"window: the analysis started by its step, 128 trials, simulated data said, the cards ({a})")
    c = json.loads(states[1][6:] or "null") if len(states) > 1 else None
    expect(c is not None and c[0] and c[1] == 90 and c[2] > 0 and c[3],
           f"window: the chart drawn on its paper, its name, simulated data said ({c})")
    for m in misfits[:60]:
        print("      ", m)
    expect(not misfits, f"window: every word fits on {len(views)} views, at {len(ta.SIZES)} sizes, in "
                        f"{len(ig.LANGS)} languages ({len(misfits)} that do not)")
    expect(len([t for t in taken if not t.startswith("res_")]) == len(views) * len(ta.SIZES) * len(ig.LANGS),
           f"window: {len(taken)} pictures in {shots}")
    expect(any(line == "errors []" for line in log), "window: no script error on the page")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "impactguide_test_results")
    p.add_argument("--ui", type=Path, help="also drive the window and save pictures here")
    p.add_argument("--only", help="only the views (and checks) of this step: holders")
    args = p.parse_args()
    ta.safe_output()
    os.environ.setdefault("IMPACTGUIDE_TWIN_AUTO", "0")
    os.environ.setdefault(ta.scans_page.SIM_SCANS_ENV, "stand_in")
    if not ta.prepare_work(args.work):
        print(f"{args.work} is not a work folder of this test: nothing deleted", flush=True)
        return 2
    if args.ui:
        if not ta.one_window_test():
            print("another window test of ImpactGuide still runs: this one runs all the same", flush=True)
        test_ui(args.work / "ui", args.ui, args.only)
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from slantui.shell.qt import QApplication
        app = QApplication(sys.argv[:1])
        if not args.only:
            test_backend(app, args.work / "backend")
        test_holders(app, args.work / "holders")
    print(f"\n{len(ta.FAILS)} failures" if ta.FAILS else "\nall checks passed", flush=True)
    return 1 if ta.FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
