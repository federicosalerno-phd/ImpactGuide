"""The app as its start runs it (scripts/impactguide_boot.py): its browser made ready while its modules load.

    pythonw quick_start.py [the app's arguments, as impactguide_app.py takes them]

Two slow things stand between the start and the app's window, and impactguide_app.py does them one after the other:
the app's modules (hundreds of imports, its own from a folder that may be on a synced drive) and the browser's first
start in the process (seconds, and on a busy computer many more, all of it blocking the thread that asks). The second
is paid once: a second view in the same process opens at once. So here the application object is made first, the
modules are imported by a thread, and meanwhile this thread, where Qt wants it, starts the browser with a page of
nothing in a view never shown, from inside Qt's loop (a call into Qt from Python would hold Python's lock, and the
modules' thread with it, until the browser is up). Then the app goes on as impactguide_app.main(), on the same application object, and
its window's view opens at once. The view of nothing is let go once the app's window is up.

Until its page is up the process runs above the normal priority: someone is waiting for it, and a render or a sync
going on meanwhile must not take the computer from under it; it goes back to normal then. How long each part took
is one line in the launcher's launch.log.

impactguide_app.py, run itself, still starts as it always did.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

T0 = time.perf_counter()

APP = Path(__file__).resolve().parent
LOCAL = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ImpactGuide"
APP_NAME = "ImpactGuide"            # impactguide_app.APP_NAME and APP_ID (test_boot.py holds them together)
APP_ID = "ImpactGuide.Bench"
WARM_KEEP_MS = 20000                # the view of nothing is let go this long after the app is under way
UP_EVENT = r"Local\ImpactGuideUp"   # the page is up (impactguide_app.tell_launcher_up): normal priority again
PRIORITY_MAX_S = 180.0              # and at the latest after this long
ABOVE_NORMAL, NORMAL = 0x8000, 0x0020
# the view of nothing is made by QML from inside Qt's loop: a call into Qt from Python holds Python's lock until it
# returns, and the modules' thread would stand still for the whole of the browser's start. A word in the band's
# weight is laid out first: Qt's first semibold text lists every installed family once more (1 to 2 s with two
# thousand fonts, 21/10), and here it is paid beside the modules instead of in the window's band
WARM_QML = b'''import QtQuick 2.15
import QtWebEngine 1.10
Item {
    property bool made: maker.status === Loader.Ready
    Loader { id: maker; active: false; sourceComponent: Component { Item {
        Text { text: "ImpactGuide"; font.family: "Segoe UI"; font.weight: Font.DemiBold; font.pixelSize: 18 }
        WebEngineView { url: "about:blank" }
    } } }
    Timer { interval: 0; running: true; onTriggered: {
        if (warmStorage) {          // the browser's data on disk (SlantUI keep_browser_data, the same settings)
            var p = WebEngine.defaultProfile;
            p.storageName = "SlantUI";
            p.offTheRecord = false;
            p.persistentStoragePath = warmStorage + "/storage";
            p.cachePath = warmStorage + "/cache";
            p.downloadPath = warmStorage + "/downloads";
            p.httpCacheType = WebEngineProfile.MemoryHttpCache;
            p.persistentCookiesPolicy = WebEngineProfile.NoPersistentCookies;
        }
        maker.active = true;
    } }
}
'''
LOOK_MS = 10                        # how often the loop looks whether the browser and the modules are there
WARM_MAX_S = 90.0                   # and the browser waited for at most this long
WEB_DIR = "web"                     # the browser's data under the app's local folder (impactguide_app.WEB_DIR)

if not sys.pycache_prefix:          # the app's own cache (impactguide_app.py), never beside the sources
    sys.pycache_prefix = str(LOCAL / "pycache")
# this folder last, as impactguide_app.py keeps it: the imports of the libraries never look at the synced drive
_here = os.path.normcase(str(APP))
sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p) if p else os.getcwd()) != _here]
sys.path.append(str(APP))


def warm_browser(storage: Path | None = None):
    """A view with a page of nothing, never shown: the browser's first start, paid here; with `storage`, the browser's
    data kept on disk there (set from QML too: made from Python, the profile would make its settings with Python's
    lock held)."""
    from slantui.shell.qt import QByteArray, QQuickView, QUrl
    view = QQuickView()
    try:
        from PyQt6.QtQml import QQmlComponent
    except ImportError:             # pragma: no cover - the PyQt5 fallback of SlantUI
        from PyQt5.QtQml import QQmlComponent
    if storage is not None:
        Path(storage).mkdir(parents=True, exist_ok=True)
    view.rootContext().setContextProperty("warmStorage", Path(storage).as_posix() if storage is not None else "")
    comp = QQmlComponent(view.engine())
    comp.setData(QByteArray(WARM_QML), QUrl())
    item = comp.create(view.rootContext())
    return view, comp, item


def side_by_side(loader: threading.Thread, item) -> float:
    """Qt's loop runs until the browser is made (WARM_QML) and the modules are loaded; when the browser was made,
    from the start. Python's lock is free while Qt makes it, so the modules load meanwhile."""
    from slantui.shell.qt import QEventLoop, QTimer
    loop = QEventLoop()
    made = []
    since = time.perf_counter()

    def look():
        # never waited for past WARM_MAX_S: a view that is not made (a QML error) leaves the cost to the app's view
        if not made and (item is None or item.property("made") or time.perf_counter() - since > WARM_MAX_S):
            made.append(time.perf_counter() - T0)
        if made and not loader.is_alive():
            loop.quit()
    timer = QTimer()
    timer.timeout.connect(look)
    timer.start(LOOK_MS)
    loop.exec()
    timer.stop()
    return made[0]


def browser_storage(argv: list[str]) -> Path:
    """Where the browser keeps its data (its compiled shaders above all), as impactguide_app.build has it: the app's
    local folder, the simulation's folder under --simulate."""
    base = LOCAL
    if "--simulate" in argv and argv.index("--simulate") + 1 < len(argv):
        base = Path(argv[argv.index("--simulate") + 1])
    return base / WEB_DIR


def priority(cls: int) -> None:
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass(k32.GetCurrentProcess(), cls)
        except (OSError, AttributeError):
            pass


def normal_when_up() -> None:
    """Back to the normal priority once the page is up (or after PRIORITY_MAX_S), from a thread of its own."""
    def wait():
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.OpenEventW.restype = ctypes.c_void_p
            k32.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
            k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            handle = k32.OpenEventW(0x00100000, False, UP_EVENT)      # SYNCHRONIZE
            if handle:
                k32.WaitForSingleObject(handle, int(PRIORITY_MAX_S * 1000))
            else:
                time.sleep(PRIORITY_MAX_S)
        except (OSError, AttributeError):
            pass
        priority(NORMAL)
    threading.Thread(target=wait, name="priority", daemon=True).start()


def note(line: str) -> None:
    """One line in the launcher's launch.log, when there is one."""
    folder = LOCAL / "launch"
    if folder.is_dir():
        try:
            with open(folder / "launch.log", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
        except OSError:
            pass


def main(argv: list[str]) -> int:
    priority(ABOVE_NORMAL)
    normal_when_up()
    box: dict = {}

    def load():
        try:
            import impactguide_app
            box["app"] = impactguide_app
        except BaseException:       # written where a failed start is looked for, as impactguide_app.py does
            box["error"] = traceback.format_exc()
        box["modules"] = time.perf_counter() - T0

    loader = threading.Thread(target=load, name="modules", daemon=True)
    loader.start()                  # first: the imports of the toolkit below are shared with it
    from slantui.shell import Application
    app = Application(APP_NAME, app_id=APP_ID)
    try:
        warm = warm_browser(browser_storage(argv))
    except Exception:               # no warm start: the app's own view pays for the browser, as before
        warm = None
    browser = side_by_side(loader, warm[2] if warm is not None else None)
    loader.join()
    if "error" in box:
        raise RuntimeError(box["error"])
    note(f"app: modules {box['modules']:.1f} s, browser {browser:.1f} s, side by side")
    ig = box["app"]
    if warm is not None:
        from slantui.shell.qt import QTimer
        QTimer.singleShot(WARM_KEEP_MS, lambda: [o.deleteLater() for o in warm if o is not None])
    return ig.main(argv, app=app)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except BaseException:
        tb = traceback.format_exc()
        try:
            LOCAL.mkdir(parents=True, exist_ok=True)
            (LOCAL / "startup_error.log").write_text(tb, encoding="utf-8")
        except OSError:
            pass
        if sys.stderr is not None:
            sys.stderr.write(tb)
        sys.exit(1)
