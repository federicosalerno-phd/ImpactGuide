"""The start of ImpactGuide as the Windows installer lays it out (build_installer.py): the shortcut runs

    <install folder>\\python\\pythonw.exe -E -s <install folder>\\impactguide_start.pyw [the app's arguments]

-E and -s keep out whatever Python the computer has of its own (its variables and its user packages). Before the app
starts, this says where the installed app keeps what is recorded with it (IMPACTGUIDE_DATA: Documents\\ImpactGuide,
laid out as the project's 03_data, never touched by the uninstaller; the protocol and the targets that ship in
03_data/ are copied there when missing, never over the user's own), its work (IMPACTGUIDE_WORK: the models of the
live view, beside the app) and scrcpy (IMPACTGUIDE_SCRCPY) when it ships with it, and gives the twin its model at
the first start (the one that ships in twin/, copied into the user's cache only when that has none).
One window at a time: a second start brings the open one to the front. If the app cannot start, a message says
where the details are.
"""
import ctypes
import os
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / "04_code" / "app" / "impactguide_app.py"
LOCAL = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ImpactGuide"
MUTEX = "ImpactGuideRunning"        # the installer waits for it to go (AppMutex of impactguide.iss)
TITLE = "ImpactGuide"
ERROR_ALREADY_EXISTS = 183
SW_RESTORE = 9


def already_open():
    """True when another ImpactGuide holds the mutex: its window is brought to the front."""
    k32, u32 = ctypes.windll.kernel32, ctypes.windll.user32
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    handle = k32.CreateMutexW(None, 0, MUTEX)
    if not handle or k32.GetLastError() != ERROR_ALREADY_EXISTS:
        globals()["_MUTEX"] = handle        # held until this process ends
        return False
    found = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def look(hwnd, _):
        n = u32.GetWindowTextLengthW(hwnd)
        if n and u32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(n + 1)
            u32.GetWindowTextW(hwnd, buf, n + 1)
            if buf.value.startswith(TITLE):
                found.append(hwnd)
                return False
        return True
    u32.EnumWindows(WNDENUMPROC(look), None)
    for hwnd in found[:1]:
        u32.ShowWindow(ctypes.c_void_p(hwnd), SW_RESTORE)
        u32.SetForegroundWindow(ctypes.c_void_p(hwnd))
    return True


def seed_twin():
    """The twin's model that ships with the app, into the user's cache when it has none yet."""
    src, dst = HERE / "twin", LOCAL / "twin"
    if not (src / "current.json").is_file() or (dst / "current.json").is_file():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "current.json":
            shutil.copy2(f, dst / f.name)
    shutil.copy2(src / "current.json", dst / "current.json")      # last: the model is whole once it is named


def documents():
    """The user's Documents folder (where Windows has it, OneDrive included)."""
    try:
        guid =(ctypes.c_ubyte * 16).from_buffer_copy(
            bytes.fromhex("D0 9A D3 FD 8F 23 AF 46 AD B4 6C 85 48 03 69 C7".replace(" ", "")))  # FOLDERID_Documents
        out = ctypes.c_wchar_p()
        ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out))
        path = out.value
        ctypes.windll.ole32.CoTaskMemFree(out)
        if path:
            return Path(path)
    except (OSError, AttributeError, ValueError):
        pass
    return Path.home() / "Documents"


def seed_data(data):
    """The protocol and the targets that ship with the app, into the data folder where they are missing."""
    src = HERE / "03_data"
    for f in src.rglob("*") if src.is_dir() else ():
        q = data / f.relative_to(src)
        if f.is_file() and not q.exists():
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, q)


def say(text):
    ctypes.windll.user32.MessageBoxW(None, text, TITLE, 0x10)


def main():
    if os.name == "nt" and already_open():
        return 0
    data = Path(os.environ.setdefault("IMPACTGUIDE_DATA", str(documents() / "ImpactGuide")))
    os.environ.setdefault("IMPACTGUIDE_WORK", str(HERE / "work"))
    if (HERE / "scrcpy" / "scrcpy-server").is_file():
        os.environ.setdefault("IMPACTGUIDE_SCRCPY", str(HERE / "scrcpy"))
    sys.pycache_prefix = str(LOCAL / "pycache")
    try:
        seed_data(data)
    except OSError:
        pass                                # the trials' page makes its protocol again from its defaults
    try:
        seed_twin()
    except OSError:
        pass                                # the twin then says it has no model; the app starts all the same
    sys.argv = [str(APP)] + sys.argv[1:]
    sys.path.insert(0, str(APP.parent))
    try:
        runpy.run_path(str(APP), run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            say(f"ImpactGuide stopped.\n\nDetails: {LOCAL / 'errors.log'}")
        return e.code if isinstance(e.code, int) else 0
    except BaseException:
        say(f"ImpactGuide did not start.\n\nDetails: {LOCAL / 'startup_error.log'}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
