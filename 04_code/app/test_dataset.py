"""ImpactGuide test of the Dataset step (dataset_page.py, ui/28-dataset.js): the dataset read in a worker, the check
of every file, the package written into a folder and into a zip chosen in the dialog, never over another one, one
job at a time, nothing to pack said in the app's words, the last package opened; then, with --ui, the step through
the real window, its words checked for cuts and wraps in English, Italian and German at the smallest window and at
1920 x 1032, with pictures.

    python test_dataset.py [--work DIR] [--ui SHOTS]

Two simulated trials are run through the backend's slots (test_app.Driver), with stand-in scans: their ground truth
is written as the Scans page's store has it (dataset_pack.fake_truth); a small synthetic dataset stands in for the
twin's (dataset_pack.make_synth). Exit code 1 on failure. Everything is written under --work.
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
os.environ.setdefault("IMPACTGUIDE_TWIN_AUTO", "0")
os.environ.setdefault("IMPACTGUIDE_SIM_SCANS", "stand_in")
import test_app as ta  # noqa: E402  (the app's own test: its driver, its checks of the words, its window)

import dataset_pack  # noqa: E402

ig = ta.ig
expect = ta.expect


class Dialog:
    """Stands in for QFileDialog: answers what it is told to."""
    folder = ""
    zip_path = ""

    @classmethod
    def getExistingDirectory(cls, *a, **k):
        return cls.folder

    @classmethod
    def getSaveFileName(cls, *a, **k):
        return cls.zip_path, "zip (*.zip)"


def prepared(d, work):
    """Two closed trials, their ground truth in the Scans page's store, a small synthetic dataset."""
    b = d.backend
    t1, _ = d.run_trial(mass="137.2")
    t2, _ = d.run_trial()
    run = b.run()
    for t in (t1, t2):
        dataset_pack.fake_truth(run, b.roots.processed, t)
    synth = work / "synth"
    dataset_pack.make_synth(synth)
    b.dataset.synth = synth
    return t1, t2


def wait_dataset(d, action, timeout=60):
    n = len(d.events)
    ok = d.wait(lambda: any(e["type"] == "dataset" and e.get("action") == action for e in d.events[n:]), timeout)
    return next((e for e in d.events[n:] if e["type"] == "dataset" and e.get("action") == action), None) if ok else None


def test_backend(app, work):
    d = ta.Driver(app, work / "backend")
    b = d.backend
    t1, t2 = prepared(d, work)
    n = len(d.events)
    s = d.call("dataset_state", json.dumps({"fresh": True}))
    read = d.wait(lambda: any(e["type"] == "dataset" and e.get("action") == "read" for e in d.events[n:]), 60)
    s = d.call("dataset_state", "{}")
    inv = s.get("inventory") or {}
    real, syn = inv.get("real") or {}, inv.get("synthetic") or {}
    expect(read and real.get("trials") == 2 and real.get("truth") == 2 and real.get("photos") == 8
           and syn.get("sessions") == 3 and syn.get("images") == 9 and s["parts"] == {"real": True, "synthetic": True},
           f"dataset: read in a worker, then 2 trials with their ground truth, {real.get('photos')} photos, "
           f"{syn.get('sessions')} synthetic sessions")
    # the check of every file
    a = d.call("dataset_check")
    busy = d.call("dataset_check")
    expect(a.get("job") and a["job"]["kind"] == "check" and busy.get("error") == "ds_busy"
           and set(busy) <= {"error", "args", "message"},
           "dataset: the check runs in a worker, a second one meanwhile refused in the app's words")
    e = wait_dataset(d, "checked")
    s = d.call("dataset_state", "{}")
    expect(e is not None and s["check"] and s["check"]["level"] in ("ok", "warn") and s["job"] is None,
           f"dataset: every file checked ({(s['check'] or {}).get('files')} files, {e and e.get('problems')} to review)")
    # the package into a folder
    out = work / "packages"
    out.mkdir(exist_ok=True)
    real_dialog, ig.QFileDialog = ig.QFileDialog, Dialog
    try:
        Dialog.folder = ""
        a = d.call("dataset_export", json.dumps({"to": "folder"}))
        expect(a == {"cancelled": True}, "dataset: the dialog closed with nothing chosen: nothing done")
        Dialog.folder = str(out)
        a = d.call("dataset_export", json.dumps({"to": "folder"}))
        e = wait_dataset(d, "exported")
        s = d.call("dataset_state", "{}")
        x = s.get("export") or {}
        pkg = Path(x.get("path", ""))
        idx = json.loads((pkg / "index.json").read_text(encoding="utf-8")) if pkg.is_dir() else {}
        expect(e is not None and pkg.parent == out and (pkg / "README.md").is_file() and (pkg / "files.csv").is_file()
               and (pkg / "real" / "campaign" / t1 / f"{t1}_photo_az40.jpg").is_file()
               and (pkg / "real" / "campaign" / "ground_truth" / f"{t2}.json").is_file()
               and idx.get("simulated") and x.get("there") and x.get("name") == pkg.name,
               f"dataset: package written into the folder chosen ({x.get('name')}, {x.get('files')} files)")
        a = d.call("dataset_export", json.dumps({"to": "folder"}))
        wait_dataset(d, "exported")
        second = Path((d.call("dataset_state", "{}").get("export") or {}).get("path", ""))
        expect(second != pkg and second.is_dir() and pkg.is_dir(),
               f"dataset: a second package beside the first, never over it ({second.name})")
        Dialog.zip_path = str(out / "bench.zip")
        a = d.call("dataset_export", json.dumps({"to": "zip"}))
        wait_dataset(d, "exported")
        z = Path((d.call("dataset_state", "{}").get("export") or {}).get("path", ""))
        expect(z == out / "bench.zip" and z.is_file() and json.loads(
               (HERE.parent.parent / "04_code" / "app" / "ui" / "strings" / "en.json").read_text(encoding="utf-8"))
               .get("ds.done.export"), "dataset: package written as the zip chosen")
        with ta.Opened() as opened:
            a = d.call("dataset_open")
        expect(a.get("ok") and opened.paths == [out], "dataset: Open shows the folder of the last package")
    finally:
        ig.QFileDialog = real_dialog
    # nothing to pack
    s = d.call("dataset_parts", json.dumps({"real": False, "synthetic": False}))
    a = d.call("dataset_check")
    expect(s["parts"] == {"real": False, "synthetic": False} and a.get("error") == "ds_empty",
           "dataset: nothing chosen for the package: said in the app's words")
    d.call("dataset_parts", json.dumps({"real": True, "synthetic": True}))
    # the last check and package after a restart
    b.shutdown()
    d.open_backend()
    s = d.call("dataset_state", "{}")
    expect(s["check"] and s["export"] and s["export"]["path"] == str(z),
           "dataset: the last check and package said again after the app is opened again")
    d.backend.shutdown()


def test_ui(work, shots):
    """The Dataset step in the real window: its words in English, Italian and German at the smallest window and at
    1920 x 1032, nothing cut, nothing wrapping."""
    shots.mkdir(parents=True, exist_ok=True)
    app, win, backend = ta.build_window(["--simulate", str(work), "--lang", "en"])
    away = ta.away_origin(app)
    d = ta.Driver.__new__(ta.Driver)
    d.app, d.work, d.events, d.backend = app, work, [], backend
    backend.happened.connect(lambda raw: d.events.append(json.loads(raw)))
    synth = work / "synth"
    dataset_pack.make_synth(synth)
    backend.dataset.synth = synth
    got = {}

    def js(script, timeout=20):
        key = object()
        got.pop(key, None)
        box = []
        win.run_js(script, lambda v: box.append(v))
        d.wait(lambda: bool(box), timeout)
        return box[0] if box else None

    up = d.wait(lambda: js("typeof DATASET !== 'undefined' && !!document.querySelector('.rp')", 2) is True, 60)
    expect(up, "window: the page is up")
    js(ta.FIT)
    misfits = []
    for w, h in (ig.MIN_WINDOW, (1920, 1032)):
        win.setGeometry(*away, w, h)
        for code in ("en", "it", "de"):
            js(f"$('lang').value='{code}'; $('lang').dispatchEvent(new Event('change')); go('results', 'dataset');")
            d.wait(lambda: js("!!(DATASET.state && DATASET.state.inventory)", 2) is True, 30)
            d.wait(lambda: False, 0.6)
            js("__adv('dsAdv', true)")
            d.wait(lambda: False, 0.4)
            bad = json.loads(js("__fit()") or "[]")
            if bad:     # measured again: only what stays wrong counts
                d.wait(lambda: False, 0.6)
                bad = json.loads(js("__fit()") or "[]")
            misfits += [f"{w}x{h} {code}: {m}" for m in bad]
            img = win.grabWindow()
            img.save(str(shots / f"dataset_{code}_{w}x{h}.png"))
            js("__adv('dsAdv', false)")
    shown = js("JSON.stringify([document.querySelector('.view.show').dataset.view, "
               "document.querySelector('.rp-page.show').dataset.panel, $('dsSynthTiles').children.length])")
    expect(json.loads(shown or "[]") == ["dataset", "dataset", 6], f"window: the Dataset view and panel ({shown})")
    expect(not misfits, "window: every word of the Dataset step fits (en, it, de; smallest and 1920 x 1032)"
           + ("".join("\n      " + m for m in misfits[:12])))
    win.close()
    backend.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "impactguide_test_dataset")
    p.add_argument("--ui", type=Path, help="also check the step through the window and save pictures here")
    args = p.parse_args()
    ta.safe_output()
    if not ta.prepare_work(args.work):
        print(f"{args.work} is not empty and is not a work folder of this test: nothing deleted.", flush=True)
        return 2
    if args.ui:
        if not ta.one_window_test():
            print("another window test of ImpactGuide still runs: this one runs all the same", flush=True)
        test_ui(args.work / "ui", args.ui)
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from slantui.shell.qt import QApplication
        app = QApplication(sys.argv[:1])
        test_backend(app, args.work)
        del app
    print(f"\n{len(ta.FAILS)} failures" if ta.FAILS else "\nall checks passed", flush=True)
    return 1 if ta.FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
