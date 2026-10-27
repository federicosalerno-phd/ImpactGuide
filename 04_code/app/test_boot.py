"""Self-test of the start behind the desktop icon (scripts/impactguide_boot.py, first_window.py, quick_start.py).

    python test_boot.py

No window is opened (the first window is drawn off screen) and nothing is written outside a temporary folder. It
checks: the names the start and the app share (events, mutex, environment, taskbar identity) are one; where the app
opens (the first window's rectangle and its turn) is read back exactly, and nonsense is refused; what the app leaves
for the next start is written only when it changed, with the mark's fingerprint the start compares, and the start's
own copy refreshed only where the desktop launcher put one; the start runs quick_start.py when the app has it; a
second start finds the first one's mutex; the app's two folders come last on sys.path and none of their modules has
the name of one elsewhere; ground_truth.py and the app's start import no scipy; the first window, drawn from what
the app kept, shows the band at the top and the empty window's colour under it; its first frame is kept as a picture
of the right size, and the window of Windows alone that shows it opens where Qt opens the first window.
Exit code 1 on failure.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent
SRC = APP.parent / "src"
SCRIPTS = APP / "scripts"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("IMPACTGUIDE_TWIN_AUTO", "0")
sys.path.insert(0, str(SCRIPTS))
FAILS: list[str] = []
KEEP: list = []                     # Qt objects kept to the end of the process


def fw_frame_ms():
    from slantui import firstwindow
    return firstwindow.MOTION_FRAME_MS


def expect(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def fresh_python(code: str) -> str:
    env = dict(os.environ)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(APP),
                         timeout=600)
    return (out.stdout + out.stderr).strip()


def main() -> int:
    import impactguide_boot as boot
    sys.path.insert(0, str(APP))
    import first_window as fw
    import quick_start as qs
    import impactguide_app as ig

    # one name for each thing the start and the app share
    start_pyw = (SCRIPTS / "impactguide_start.pyw").read_text(encoding="utf-8")
    expect(boot.SHOWN_EVENT == fw.SHOWN_EVENT and boot.FIRST_ENV == fw.FIRST_ENV and boot.MUTEX == fw.MUTEX
           and boot.UP_EVENT == ig.LAUNCHER_EVENT and boot.APP_ID == ig.APP_ID == qs.APP_ID
           and qs.APP_NAME == ig.APP_NAME and f'MUTEX = "{boot.MUTEX}"' in start_pyw
           and (APP / boot.ENTRY).is_file() and APP.joinpath(*boot.MARK) == ig.MARK and APP.joinpath(*boot.ICON) == ig.ICON,
           "names: events, mutex, environment, identity and the app's pictures the same on both sides")

    # where the app opens
    with tempfile.TemporaryDirectory() as tmp:
        env = {fw.FIRST_ENV: "-1203,88,2160,1350,1790000000.250"}
        alone = fw.opening(env, Path(tmp))
        (Path(tmp) / fw.FIRST_SINCE).write_text("1790000002.500", encoding="utf-8")
        later = fw.opening(env, Path(tmp))
        (Path(tmp) / fw.FIRST_SINCE).write_text("1789000000.000", encoding="utf-8")
        stale = fw.opening(env, Path(tmp))
        expect(alone == ((-1203, 88, 2160, 1350), 1790000000.25) and later[1] == 1790000002.5
               and stale[1] == 1790000000.25 and fw.opening({fw.FIRST_ENV: "1,2,3"}, Path(tmp)) is None
               and fw.opening({fw.FIRST_ENV: "1,2,0,5,1.0"}, Path(tmp)) is None and fw.opening({}, Path(tmp)) is None
               and boot.FIRST_SINCE == fw.FIRST_SINCE,
               "opening: the first window's rectangle and turn read back exactly, the turn of this start's first "
               "window taken over, an older one and nonsense refused")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        launch = tmp / "launch"
        expect(fw.keep(launch, palette=ig.PALETTE, text="x", size=(1440, 900)) is None,
               "keep: nothing without the launcher's folder")
        launch.mkdir()
        from PyQt6.QtGui import QColor, QImage
        from PyQt6.QtWidgets import QApplication
        KEEP.append(QApplication.instance() or QApplication(sys.argv[:1]))
        band = QImage(1440, 44, QImage.Format.Format_RGB32)     # as wide as the window at the scale of no screen
        band.fill(QColor("#336699"))
        fw.keep(launch, palette=ig.PALETTE, text="avvio", size=(1440, 900), band=band, close=[1398, 0, 42, 28],
                mark=ig.MARK, wait=True)
        info = json.loads((launch / "first.json").read_text(encoding="utf-8"))
        stamp = (launch / "first.json").stat().st_mtime_ns, (launch / "band-100.png").stat().st_mtime_ns
        fw.keep(launch, palette=ig.PALETTE, text="avvio", size=(1440, 900), band=band, close=[1398, 0, 42, 28],
                mark=ig.MARK, wait=True)
        again = (launch / "first.json").stat().st_mtime_ns, (launch / "band-100.png").stat().st_mtime_ns
        expect(info["palette"] == ig.PALETTE and info["text"] == "avvio" and info["size"] == [1440, 900]
               and info["close"] == [1398, 0, 42, 28] and info["bands"] == {"band-100.png": boot.crc(ig.MARK.read_bytes())}
               and APP.joinpath(*Path(info["logo"]).parts) == ig.MARK
               and set(info["look"]) >= {"surface", "accent", "track", "text", "band", "line", "font"}
               and stamp == again and not (launch / fw.BOOT.name).exists()
               and not list(launch.glob("*.tmp")) and not (launch / "mark.svg").exists(),
               "keep: first.json and the band of this scale written once, unchanged files left alone, the mark's fingerprint as "
               "the start computes it, no copy of the mark, the start not copied where the launcher put none")
        (launch / fw.BOOT.name).write_text("old", encoding="utf-8")
        fw.keep(launch, palette=ig.PALETTE, text="avvio", size=(1440, 900), band=band, scale=1.5, mark=ig.MARK,
                wait=True)
        two = json.loads((launch / "first.json").read_text(encoding="utf-8"))
        expect((launch / fw.BOOT.name).read_bytes() == fw.BOOT.read_bytes()
               and set(two["bands"]) == {"band-100.png", "band-150.png"} and two["close"] == [1398, 0, 42, 28],
               "keep: the launcher's copy of the start brought up to date; a band per screen scale, the X kept")
        expect(boot.cached(tmp / "none", launch)["text"] == "avvio" and boot.cached(tmp / "none") == {},
               "start: what the app kept read from the first folder that has it")

        # the start runs quick_start.py beside the app, or the app itself
        s = boot.Start.__new__(boot.Start)
        s.python, s.args, s.app = ["py"], ["--lang", "it"], APP / "impactguide_app.py"
        lone = tmp / "lone" / "impactguide_app.py"
        lone.parent.mkdir()
        lone.write_text("", encoding="utf-8")
        cmd = s.command()
        s.app = lone
        expect(cmd == ["py", str(APP / "quick_start.py"), "--lang", "it"]
               and s.command() == ["py", str(lone), "--lang", "it"],
               "start: quick_start.py when the app has it, else the app's own file")

        # the first window, off screen, from what the app kept
        from slantui import firstwindow
        win = firstwindow.FirstWindow(look=info["look"], logo=ig.MARK.read_bytes(), text="avvio",
                                      band=[str(launch / "band-100.png")], size=(1440, 900), title="test",
                                      icon=ig.ICON.read_bytes(), close_rect=info["close"])
        win.resize(1440, 900)
        img = win.grab().toImage()
        dpr = img.devicePixelRatio()
        top, body = img.pixelColor(int(700 * dpr), int(10 * dpr)), img.pixelColor(int(30 * dpr), int(120 * dpr))
        expect(dpr == 1.0 and top.name() == "#336699" and body.name().lower() == info["look"]["surface"].lower(),
               f"first window: the band kept at the top, the empty window's colour under it ({top.name()}, "
               f"{body.name()})")
        expect(not win.windowIcon().isNull() and win.rect_text().endswith(",1440,900"),
               "first window: the app's icon from its .svg, the rectangle given to the app")
        KEEP.append(win)

        # the first frame as a picture, kept per scale with its size, for the start's window of Windows alone
        bmp, motion = fw.first_picture(palette=ig.PALETTE, text="avvio", mark=ig.MARK, band=band, size=(1440, 900),
                                       scale=1.0)
        fw.keep(launch, palette=ig.PALETTE, text="avvio", size=(1440, 900), band=band, scale=1.0, mark=ig.MARK,
                picture=bmp, picture_size=(1440, 900), motion=motion, wait=True)
        pics = json.loads((launch / "first.json").read_text(encoding="utf-8"))["pictures"]
        from PyQt6.QtGui import QImage as _QImage
        shot = _QImage(str(launch / "first-100.bmp"))
        moving = pics.get("first-100.bmp", {}).pop("motion", None)
        expect(bmp[:2] == b"BM" and pics == {"first-100.bmp": {"size": [1440, 900], "mark": info["bands"]["band-100.png"]}}
               and shot.width() == 1440 and shot.pixelColor(700, 10).name() == "#336699"
               and shot.pixelColor(30, 120).name().lower() == info["look"]["surface"].lower(),
               "first frame: a BMP of the first window's first frame, band and empty window, kept with its size")
        # its motion: the ring's and the bar's frames, the first of each the first frame's own pixels
        same = bool(moving)
        for part in ("ring", "bar"):
            m = (moving or {}).get(part) or {}
            atlas = _QImage(str(launch / m.get("file", "none")))
            x, y, w, h = m.get("rect") or (0, 0, 0, 0)
            same = same and not atlas.isNull() and atlas.width() == w and atlas.height() == h * m.get("n", 0) > 0
            same = same and all(atlas.pixelColor(i, j) == shot.pixelColor(x + i, y + j)
                                for i in range(0, w, 7) for j in range(0, h, 3))
        frames = [atlas.copy(0, k * h, w, h) for k in (0, 1)] if same else []
        expect(same and moving["frame_ms"] == fw_frame_ms() and frames[0] != frames[1],
               f"first frame: its motion kept beside it, the ring and the bar frame by frame from the first frame "
               f"({moving and {k: v for k, v in moving.items() if k != 'frame_ms'}})")

    # the window of Windows alone opens where Qt opens the first window (a native window made, never shown)
    probe = ("import sys, ctypes; from ctypes import wintypes; sys.path.insert(0, r'%s'); import impactguide_boot as b; "
             "sw = b.StaticWindow({}, (1440, 900)); ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)); "
             "native, f = sw.place(); from slantui import firstwindow as fw; app = fw.application(); "
             "w = fw.FirstWindow(look=fw.look(), size=(1440, 900)); r = wintypes.RECT(); "
             "ctypes.windll.user32.GetWindowRect(wintypes.HWND(int(w.winId())), ctypes.byref(r)); "
             "print(max(abs(a - c) for a, c in zip(native, (r.left, r.top, r.right - r.left, r.bottom - r.top))))"
             % SCRIPTS)
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, cwd=str(APP),
                         timeout=120)
    last = (out.stdout.strip().splitlines() or ["?"])[-1]
    expect(last.isdigit() and int(last) <= 2,
           f"first frame: placed by Windows alone where Qt places the first window (off by {last} px at most)")

    # a second start finds the first one's mutex
    name = f"ImpactGuideTest-{os.getpid()}"
    probe = ("import sys; sys.path.insert(0, r'%s'); import impactguide_boot as b; b.MUTEX = %r; "
             "b.bring_to_front = lambda: False; print(b.already_open())" % (SCRIPTS, name))
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.CreateMutexW.restype = ctypes.c_void_p
    first = fresh_python(probe)
    h = k32.CreateMutexW(None, 0, name)
    second = fresh_python(probe)
    k32.CloseHandle(ctypes.c_void_p(h))
    expect(first.endswith("False") and second.endswith("True"),
           "one window: a start while another holds the mutex brings that one forward instead")

    # the app's folders last, and no module of theirs shadowed by (or shadowing) one elsewhere
    norm = [os.path.normcase(p) for p in sys.path]
    ours = [os.path.normcase(str(APP)), os.path.normcase(str(SRC))]
    elsewhere = [p for p in sys.path if os.path.normcase(p) not in ours and p != str(SCRIPTS) and p]
    names = sorted({f.stem for d in (APP, SRC) for f in d.glob("*.py")})
    clash = [n for n in names if n in sys.builtin_module_names
             or importlib.machinery.PathFinder.find_spec(n, elsewhere) is not None]
    expect(norm[-2:] == ours and not clash,
           f"sys.path: the app's folders last, none of their {len(names)} modules named as one elsewhere {clash}")

    # the app's folders answered from one listing each: found, checked, and a module added later found once asked
    import module_finder
    finders = [sys.path_importer_cache.get(str(d)) for d in (APP, SRC)]
    loaded = [type(sys.modules[n].__spec__.loader).__name__ for n in ("trials", "acquisition", "scans_page")]
    with tempfile.TemporaryDirectory() as tmp:
        f = module_finder.FolderFinder(tmp)
        before = f.find_spec("added_later")
        (Path(tmp) / "added_later.py").write_text("X = 1\n", encoding="utf-8")
        f.invalidate_caches()
        spec = f.find_spec("added_later")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    expect(all(isinstance(x, module_finder.FolderFinder) for x in finders) and set(loaded) == {"_Loader"}
           and before is None and mod.X == 1,
           "modules: the app's two folders answered from a listing, a module added later found after invalidate_caches")

    out = fresh_python("import sys; sys.path.append(r'%s'); import ground_truth; a = 'scipy' in sys.modules; "
                       "sys.path.append(r'%s'); import impactguide_app; print(a, 'scipy' in sys.modules)" % (SRC, APP))
    expect(out.endswith("False False"), f"imports: no scipy at the app's start, nor for ground_truth ({out[-40:]})")

    print(f"\n{len(FAILS)} failures" if FAILS else "\nall checks passed", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
