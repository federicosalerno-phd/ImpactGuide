"""The start behind the ImpactGuide icon: a window at once, the app behind it.

    pythonw impactguide_boot.py [--app PATH] [the app's arguments]

The desktop shortcut runs it from %LOCALAPPDATA%\\ImpactGuide\\launch (install_shortcut.ps1 copies it there, and the
app keeps that copy current), so it is on the local disk even when the project is on Google Drive; the installed app
runs it through impactguide_start.pyw (run()).

The app needs seconds before its window can open (its imports, the browser inside it), many more on a busy computer
with the project on a synced drive. So this does first what needs almost nothing: one window at a time (a second
click brings the open one to the front, before anything else is loaded), then a window on screen in a few tenths of
a second (slantui.firstwindow: the app's window, empty and loading, with the band the app drew the last time, the
app's own mark and icon read from its ui folder). At its first frame it starts the app as a process of its own
(quick_start.py, which loads the app's modules while its browser gets ready), and the app opens its window on the
same rectangle with its loading screen over it; when that window is on screen this one is closed under it. If the
app stops before its page is up, the launcher says so in plain words (launcher.ps1 -Failed, or a message box in the
installed app). If the project's folder is not there yet (Google Drive not mounted), launcher.ps1 does the waiting,
in its small window, and comes back here. Closing this window (Alt+F4, the taskbar, its X) calls the start off.

What the app leaves here for the next start (first_window.py): first.json (palette, look, words, size, the X, which
mark each band was drawn with) and band-<scale>.png.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import zlib
from ctypes import wintypes
from pathlib import Path

LOCAL = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ImpactGuide"
LAUNCH = LOCAL / "launch"
TITLE = "ImpactGuide"
APP_ID = "ImpactGuide.Bench"                # the app's taskbar identity (impactguide_app.APP_ID)
MUTEX = "ImpactGuideRunning"                # held by the app while it runs (and by the installer's AppMutex)
SHOWN_EVENT = r"Local\ImpactGuideShown"     # raised by the app when its window is on screen
UP_EVENT = r"Local\ImpactGuideUp"           # raised by the app when its page is up (launcher.ps1 waits on it too)
FIRST_ENV = "IMPACTGUIDE_FIRST_WINDOW"      # x,y,w,h (screen pixels),since of this window, for the app's own
FIRST_SINCE = "first-since"                 # in LAUNCH: when the first window's motion began, for the app's Splash
ENTRY = "quick_start.py"                    # beside impactguide_app.py: the app, its modules loaded while its browser starts
MARK = ("ui", "impactguide-mark.svg")       # the app's own pictures, beside impactguide_app.py (never a copy)
ICON = ("ui", "impactguide-icon.svg")
UP_WAIT_S = 120.0                           # after its window is on screen the app is watched this long for its page
STATIC_WAIT_S = 3.0                         # the first frame's window is waited for this long to be painted
CLOSE_WAIT_S = 1.0                          # and for its thread to end once it is closed
TIMER_MS = 10                               # how often it looks whether its motion has a new frame to show
WM_GONE = 0x8001                            # WM_APP + 1: close the first frame's window (from another thread)
SPAWN_LATEST_S = 2.0                        # without a first frame, the app is started at the first window's, or
                                            # after this
ASSETS_WAIT_S = 0.3                         # the mark and the icon read from the app's folder: waited for this long,
                                            # then the window opens without them and they come when read
POLL_MS = 25
NOT_READABLE_EXIT = 2                       # what Python answers when it cannot open the script
ERROR_ALREADY_EXISTS = 183
SW_RESTORE = 9
ASFW_ANY = 0xFFFFFFFF
ABOVE_NORMAL = 0x8000                       # someone is waiting for this window: a render meanwhile does not delay it

_held = []                                  # the mutex handle, held until this process ends


def log(line: str) -> None:
    """A line in launch.log, as launcher.ps1 writes them."""
    try:
        LAUNCH.mkdir(parents=True, exist_ok=True)
        with open(LAUNCH / "launch.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
    except OSError:
        pass


def since_created() -> float:
    """Seconds since this process was created, which is the click but for the tens of ms of wscript."""
    try:
        k32 = _k32()
        times = [wintypes.FILETIME() for _ in range(4)]
        k32.GetProcessTimes(k32.GetCurrentProcess(), *[ctypes.byref(t) for t in times])
        now = wintypes.FILETIME()
        k32.GetSystemTimeAsFileTime(ctypes.byref(now))
        secs = lambda t: (t.dwHighDateTime << 32 | t.dwLowDateTime) / 1e7      # noqa: E731
        return max(0.0, secs(now) - secs(times[0]))
    except (OSError, AttributeError):
        return 0.0


def _k32():
    k32 = ctypes.windll.kernel32
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    k32.CreateEventW.restype = wintypes.HANDLE
    k32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.ResetEvent.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.c_void_p] * 4
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    return k32


def already_open() -> bool:
    """True when ImpactGuide already runs: its window is brought to the front. Otherwise the mutex is held from
    here on, so a second click made while this one is still starting finds it too."""
    if os.name != "nt":
        return False
    k32 = _k32()
    handle = k32.CreateMutexW(None, False, MUTEX)
    if handle and k32.GetLastError() != ERROR_ALREADY_EXISTS:
        _held.append(handle)
        return False
    if handle:
        k32.CloseHandle(handle)
    bring_to_front()
    return True


def release() -> None:
    """The mutex let go, for a start handed over to launcher.ps1, which comes back here."""
    while _held:
        _k32().CloseHandle(_held.pop())


def bring_to_front() -> bool:
    """The largest visible window titled ImpactGuide (the app, or a first window still loading), restored and
    brought to the front."""
    u32 = ctypes.windll.user32
    found = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def look(hwnd, _):
        n = u32.GetWindowTextLengthW(hwnd)
        if n and u32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(n + 1)
            u32.GetWindowTextW(hwnd, buf, n + 1)
            if buf.value.startswith(TITLE):
                r = wintypes.RECT()
                u32.GetWindowRect(hwnd, ctypes.byref(r))
                found.append(((r.right - r.left) * (r.bottom - r.top), hwnd))
        return True
    u32.EnumWindows(proc(look), 0)
    if not found:
        return False
    hwnd = max(found)[1]
    if u32.IsIconic(hwnd):
        u32.ShowWindow(hwnd, SW_RESTORE)
    u32.SetForegroundWindow(hwnd)
    return True


def readable(path: Path) -> bool:
    """On Drive a file can be listed before it can be read: only a real read counts."""
    try:
        with open(path, "rb") as f:
            return bool(f.read(64))
    except OSError:
        return False


def drive_present(path: Path) -> bool:
    """Whether the path's drive letter exists at all, asked without touching the drive."""
    letter = path.drive[:1].upper()
    if os.name != "nt" or not letter.isalpha():
        return True
    return bool(ctypes.windll.kernel32.GetLogicalDrives() & (1 << (ord(letter) - ord("A"))))


def cached(*folders: Path | None) -> dict:
    """What the app left for this window (first_window.py), from the first of these folders that has it."""
    for folder in folders:
        if folder is None:
            continue
        try:
            data = json.loads((Path(folder) / "first.json").read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            continue
    return {}


def crc(data: bytes | None) -> str | None:
    """The fingerprint first_window.py writes of the mark a band was drawn with."""
    return None if data is None else f"{zlib.crc32(data):08x}"


class Assets(threading.Thread):
    """The app's mark and icon, read from its own folder (a synced drive, perhaps slow) while the toolkit loads."""

    def __init__(self, app_dir: Path, mark: str | None = None):
        super().__init__(name="assets", daemon=True)
        # the mark the app's loading screen shows (first.json says which, as the app has it), and its icon
        logo = Path(app_dir).joinpath(*(Path(mark).parts if mark else MARK))
        self.paths = [logo, Path(app_dir).joinpath(*ICON)]
        self.data: list[bytes | None] = [None, None]
        self.start()

    def run(self) -> None:
        for i, p in enumerate(self.paths):
            try:
                self.data[i] = p.read_bytes()
            except OSError:
                pass

    def result(self, wait: float = 0.0) -> tuple[bytes | None, bytes | None] | None:
        """(mark, icon) once read, else None."""
        self.join(wait)
        return None if self.is_alive() else (self.data[0], self.data[1])


class StaticWindow:
    """The first frame, put on screen with Windows alone (user32, gdi32), before any toolkit is loaded.

    On a busy computer Qt needs a second or more to show its first window, Windows a tenth. This window shows the
    picture the app left of its first window's first frame (first-<scale>.bmp: the band, the empty window, the ring
    at the start of its turn), or without one the empty window's colour, at the place and size the first window opens
    at, worked out as Qt works it out (slantui.firstwindow.place). The first window then opens over it, on its
    rectangle, and this one is let go. With the frames of the motion the app kept beside the picture
    (first-<scale>-ring.bmp, first-<scale>-bar.bmp) the ring turns and the bar runs from the very first frame, in a
    thread of its own while the toolkit loads, and the first window carries the motion on from where this one has it
    (FirstWindow since): never a ring standing still. Turning by itself it is the start's only window ("alone"):
    the toolkit is not loaded at all (its imports would hold up the motion), it has the app's taskbar button and
    icon, and closing it (its band's X, Alt+F4, the taskbar) calls the start off; it lasts until the app's window is
    on screen. Still, it answers nothing but its painting and lasts the second the toolkit needs.
    """

    def __init__(self, info: dict, size=(1440, 900)):
        self.info, self.size = info, (int(size[0]), int(size[1]))
        self.hwnd = None
        self.bitmap = None
        self.brush = None
        self.rect = None
        self.since = None
        self._proc = None
        self.thread = None              # the window lives in a thread of its own (show)
        self.parts = []                 # the moving parts: [x, y, w, h, frames, bitmap, frame shown]
        self.frame_ms = 25.0
        self._closed = False
        self.alone = False              # turning by itself: the start's only window, until the app's
        self.cancelled = False          # closed by the person (alone only): the start is called off

    @staticmethod
    def _colour(hex_: str) -> int:
        v = hex_.lstrip("#")
        return int(v[0:2], 16) | int(v[2:4], 16) << 8 | int(v[4:6], 16) << 16          # COLORREF: 0x00bbggrr

    def place(self):
        """(x, y, w, h) in the screen's pixels and the screen's scale: the middle of the screen the pointer is on,
        as slantui.firstwindow.place has it in Qt's pixels."""
        u32 = ctypes.windll.user32
        pt = wintypes.POINT()
        u32.GetCursorPos(ctypes.byref(pt))
        u32.MonitorFromPoint.restype = wintypes.HANDLE
        u32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        mon = u32.MonitorFromPoint(pt, 2)                        # MONITOR_DEFAULTTONEAREST

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(mi)
        u32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        u32.GetMonitorInfoW(mon, ctypes.byref(mi))
        dx, dy = wintypes.UINT(96), wintypes.UINT(96)
        try:
            ctypes.windll.shcore.GetDpiForMonitor.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                              ctypes.c_void_p]
            ctypes.windll.shcore.GetDpiForMonitor(mon, 0, ctypes.byref(dx), ctypes.byref(dy))
        except (OSError, AttributeError):
            pass
        f = dx.value / 96.0
        ox, oy, wk = mi.rcMonitor.left, mi.rcMonitor.top, mi.rcWork
        # Qt keeps each screen's corner where Windows has it and divides the rest by the scale
        ax, ay = ox + (wk.left - ox) / f, oy + (wk.top - oy) / f
        aw, ah = int((wk.right - wk.left) / f), int((wk.bottom - wk.top) / f)
        w, h = min(self.size[0], aw), min(self.size[1], ah)
        x, y = int(ax) + (aw - w) // 2, int(ay) + (ah - h) // 2
        return (round(ox + (x - ox) * f), round(oy + (y - oy) * f), round(w * f), round(h * f)), f

    def show(self) -> bool:
        """The window on screen, turning (its own thread: the start's thread loads the toolkit meanwhile, and a
        window whose messages wait for it would stand still). True once its first frame is painted."""
        if os.name != "nt":
            return False
        ready, box = threading.Event(), {}

        def run() -> None:
            try:
                box["ok"] = self._show()
            except (OSError, AttributeError, ValueError, TypeError, ctypes.ArgumentError):
                box["ok"] = False
            ready.set()
            try:
                if box["ok"] and not self._closed:
                    self._loop()
                elif self.hwnd:                 # called off while it was being made
                    ctypes.windll.user32.DestroyWindow(self.hwnd)
            finally:
                self._free()
        self.thread = threading.Thread(target=run, name="first_frame", daemon=True)
        self.thread.start()
        if not ready.wait(STATIC_WAIT_S) or not box.get("ok"):
            self.close()
            return False
        return True

    def _motion(self, pic: dict, u32) -> None:
        """The frames of the ring and of the bar the app kept beside the first frame (first_window.keep), loaded;
        none when a file is missing (the first frame then stands still, as it always did)."""
        moving = pic.get("motion") if isinstance(pic, dict) else None
        if not isinstance(moving, dict):
            return
        try:
            self.frame_ms = float(moving["frame_ms"])
            parts = []
            for part in ("ring", "bar"):
                m = moving[part]
                x, y, w, h = (int(v) for v in m["rect"])
                n = int(m["n"])
                if not (LAUNCH / m["file"]).is_file() or n < 1 or w < 1 or h < 1:
                    return
                bmp = u32.LoadImageW(None, str(LAUNCH / m["file"]), 0, 0, 0, 0x0010 | 0x2000)
                if not bmp:
                    return
                parts.append([x, y, w, h, n, bmp, -1])
        except (KeyError, TypeError, ValueError):
            return
        self.parts = parts

    def _show(self) -> bool:
        u32, g32 = ctypes.windll.user32, ctypes.windll.gdi32
        u32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        u32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))   # per monitor, as Qt itself asks for
        self.rect, scale = self.place()
        x, y, w, h = self.rect
        name = f"first-{round(scale * 100)}.bmp"
        pics = self.info.get("pictures") if isinstance(self.info.get("pictures"), dict) else {}
        size = pics[name].get("size") if isinstance(pics.get(name), dict) else None
        u32.LoadImageW.restype = wintypes.HANDLE
        u32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int,
                                   wintypes.UINT]
        if size and abs(size[0] - w) <= 2 and abs(size[1] - h) <= 2 and (LAUNCH / name).is_file():
            self.bitmap = u32.LoadImageW(None, str(LAUNCH / name), 0, 0, 0, 0x0010 | 0x2000)   # FROMFILE, DIBSECTION
            if self.bitmap:
                self._motion(pics[name], u32)
        look = self.info.get("look") if isinstance(self.info.get("look"), dict) else {}
        g32.CreateSolidBrush.restype = wintypes.HANDLE
        g32.CreateSolidBrush.argtypes = [wintypes.DWORD]
        self.brush = g32.CreateSolidBrush(self._colour(look.get("surface", "#0D0D0F")))

        self.alone = bool(self.parts)
        close = self.info.get("close") if isinstance(self.info.get("close"), list) else None
        close = [v * scale for v in close] if close and len(close) == 4 else None
        if self.alone:              # the app's taskbar button from the first frame: its identity, then its icon
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(APP_ID))
            except (OSError, AttributeError):
                pass

        WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                     wintypes.LPARAM)
        u32.DefWindowProcW.restype = ctypes.c_ssize_t
        u32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]

        class PAINTSTRUCT(ctypes.Structure):
            _fields_ = [("hdc", wintypes.HDC), ("fErase", wintypes.BOOL), ("rcPaint", wintypes.RECT),
                        ("fRestore", wintypes.BOOL), ("fIncUpdate", wintypes.BOOL), ("rgbReserved", ctypes.c_byte * 32)]
        u32.BeginPaint.restype = wintypes.HDC
        u32.BeginPaint.argtypes = [wintypes.HWND, ctypes.c_void_p]
        u32.EndPaint.argtypes = [wintypes.HWND, ctypes.c_void_p]
        u32.InvalidateRect.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.BOOL]
        u32.DestroyWindow.argtypes = [wintypes.HWND]
        u32.PostQuitMessage.argtypes = [ctypes.c_int]
        g32.CreateCompatibleDC.restype = wintypes.HDC
        g32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        g32.SelectObject.restype = wintypes.HANDLE
        g32.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
        g32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HDC,
                               ctypes.c_int, ctypes.c_int, wintypes.DWORD]
        g32.DeleteDC.argtypes = [wintypes.HDC]
        g32.ExcludeClipRect.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        g32.SelectClipRgn.argtypes = [wintypes.HDC, wintypes.HANDLE]
        t0 = time.perf_counter()

        def frame(n: int) -> int:           # the frame of a part now: frame 0 is the first frame's own
            return int((time.perf_counter() - t0) * 1000.0 / self.frame_ms) % n

        def proc(hwnd, msg, wp, lp):
            if msg == 0x000F:                                    # WM_PAINT
                ps = PAINTSTRUCT()
                hdc = u32.BeginPaint(hwnd, ctypes.byref(ps))
                if self.bitmap:
                    mem = g32.CreateCompatibleDC(hdc)
                    old = g32.SelectObject(mem, self.bitmap)
                    for px, py, pw, ph, _n, _b, _k in self.parts:    # the moving parts come next, never twice
                        g32.ExcludeClipRect(hdc, px, py, px + pw, py + ph)
                    g32.BitBlt(hdc, 0, 0, w + 2, h + 2, mem, 0, 0, 0x00CC0020)  # SRCCOPY
                    g32.SelectClipRgn(hdc, None)
                    for part in self.parts:
                        px, py, pw, ph, n, bmp, _k = part
                        part[6] = k = frame(n)
                        g32.SelectObject(mem, bmp)
                        g32.BitBlt(hdc, px, py, pw, ph, mem, 0, k * ph, 0x00CC0020)
                    g32.SelectObject(mem, old)
                    g32.DeleteDC(mem)
                u32.EndPaint(hwnd, ctypes.byref(ps))
                return 0
            if msg == 0x0113:                                    # WM_TIMER: the parts whose frame changed
                for px, py, pw, ph, n, _b, k in self.parts:
                    if frame(n) != k:
                        r = wintypes.RECT(px, py, px + pw, py + ph)
                        u32.InvalidateRect(hwnd, ctypes.byref(r), False)
                return 0
            if msg == WM_GONE:                                   # close(), from the start's thread
                u32.DestroyWindow(hwnd)
                return 0
            if msg == 0x0002:                                    # WM_DESTROY: the loop ends
                u32.PostQuitMessage(0)
                return 0
            if msg == 0x0010:                                    # WM_CLOSE: Alt+F4, the taskbar
                if self.alone:
                    self.cancelled = True                        # the start's thread calls it off and closes it
                return 0                                         # else the first window, a moment away, closes
            if msg == 0x0201 and self.alone and close:           # WM_LBUTTONDOWN on the band's X
                px, py = ctypes.c_short(lp & 0xFFFF).value, ctypes.c_short((lp >> 16) & 0xFFFF).value
                if close[0] <= px < close[0] + close[2] and close[1] <= py < close[1] + close[3]:
                    self.cancelled = True
                return 0
            return u32.DefWindowProcW(hwnd, msg, wp, lp)
        self._proc = WNDPROC(proc)                               # kept: Windows calls it for as long as it lives

        class WNDCLASSW(ctypes.Structure):
            _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                        ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                        ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HANDLE),
                        ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]
        k32 = ctypes.windll.kernel32
        k32.GetModuleHandleW.restype = wintypes.HMODULE
        hinst = k32.GetModuleHandleW(None)
        u32.LoadCursorW.restype = wintypes.HANDLE
        u32.LoadCursorW.argtypes = [wintypes.HINSTANCE, ctypes.c_void_p]
        cursor = 32650 if self.alone else 32514                  # IDC_APPSTARTING for seconds, IDC_WAIT for one
        wc = WNDCLASSW(0, self._proc, 0, 0, hinst, None, u32.LoadCursorW(None, ctypes.c_void_p(cursor)),
                       self.brush, None, "ImpactGuideFirstFrame")
        u32.RegisterClassW.argtypes = [ctypes.c_void_p]
        u32.RegisterClassW(ctypes.byref(wc))
        u32.CreateWindowExW.restype = wintypes.HWND
        u32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                        wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
        # alone: the app's taskbar button (WS_EX_APPWINDOW); else a tool window, the first window brings the button
        self.hwnd = u32.CreateWindowExW(0x00040000 if self.alone else 0x00000080, "ImpactGuideFirstFrame", TITLE,
                                        0x80000000, x, y, w, h, None, None, hinst, None)
        if not self.hwnd:
            return False
        if self.alone and (LAUNCH / "icon.ico").is_file():
            u32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            u32.GetSystemMetrics.argtypes = [ctypes.c_int]
            for which, side in ((1, 0), (0, u32.GetSystemMetrics(49))):     # ICON_BIG (default size), ICON_SMALL
                flags = 0x0010 | (0x0040 if not side else 0)                # LOADFROMFILE, DEFAULTSIZE
                icon = u32.LoadImageW(None, str(LAUNCH / "icon.ico"), 1, side, side, flags)
                if icon:
                    u32.SendMessageW(self.hwnd, 0x0080, which, icon)            # WM_SETICON
        try:
            dwm = ctypes.windll.dwmapi
            dwm.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
            # round corners, no border: as the Window's; and no fade in, so its first frame is on screen at once
            for attr, val in ((33, 2), (34, 0xFFFFFFFE), (3, 1)):
                v = ctypes.c_uint(val)
                dwm.DwmSetWindowAttribute(self.hwnd, attr, ctypes.byref(v), 4)
        except (OSError, AttributeError):
            pass
        u32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        u32.ShowWindow(self.hwnd, 5)                             # SW_SHOW
        u32.UpdateWindow.argtypes = [wintypes.HWND]
        u32.UpdateWindow(self.hwnd)                              # painted now, before anything else is loaded
        u32.SetForegroundWindow.argtypes = [wintypes.HWND]
        u32.SetForegroundWindow(self.hwnd)
        if self.parts:
            u32.SetTimer.restype = ctypes.c_size_t
            u32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, ctypes.c_void_p]
            u32.SetTimer(self.hwnd, 1, TIMER_MS, None)
        self.since = time.time() - (time.perf_counter() - t0)    # the motion's start: frame 0
        return True

    def _loop(self) -> None:
        """The window's messages, until close() ends it."""
        u32 = ctypes.windll.user32
        msg = wintypes.MSG()
        u32.GetMessageW.restype = ctypes.c_int
        u32.GetMessageW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.UINT]
        u32.TranslateMessage.argtypes = [ctypes.c_void_p]
        u32.DispatchMessageW.restype = ctypes.c_ssize_t
        u32.DispatchMessageW.argtypes = [ctypes.c_void_p]
        while u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            u32.TranslateMessage(ctypes.byref(msg))
            u32.DispatchMessageW(ctypes.byref(msg))

    def _free(self) -> None:
        g32 = ctypes.windll.gdi32
        g32.DeleteObject.argtypes = [wintypes.HANDLE]
        for h in [self.bitmap, self.brush] + [p[5] for p in self.parts]:
            if h:
                g32.DeleteObject(h)
        self.bitmap = self.brush = None
        self.parts = []

    def close(self) -> None:
        """The window goes (its thread destroys it and ends), waited for a moment."""
        if os.name != "nt":
            return
        self._closed = True
        hwnd, self.hwnd = self.hwnd, None
        if hwnd:
            u32 = ctypes.windll.user32
            u32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            u32.PostMessageW(hwnd, WM_GONE, 0, 0)
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(CLOSE_WAIT_S)


class Start:
    """One start: the first window, the app's process, and what is watched until its page is up."""

    def __init__(self, app: Path, args: list[str], python: list[str], *, env: dict | None = None,
                 cwd: Path = LAUNCH, assets: Assets | None = None, fallback: Path | None = None,
                 on_fail=None, on_unreadable=None):
        self.app, self.args, self.python, self.cwd = Path(app), list(args), list(python), Path(cwd)
        self.fallback = fallback                    # first.json as installed, until the app keeps its own
        self.info = cached(LAUNCH, fallback)
        logo = self.info.get("logo") if isinstance(self.info.get("logo"), str) else None
        self.assets = assets if assets is not None else Assets(self.app.parent, logo)
        self.env = dict(os.environ if env is None else env)
        self.on_fail, self.on_unreadable = on_fail, on_unreadable
        self.t0 = time.perf_counter() - since_created()        # the times in launch.log are from the click
        self.proc = None
        self.first = None
        self.static = None
        self.native = None                  # the windows' rectangle in the screen's pixels
        self.since = None
        self.qapp = None
        self.shown_at = None
        self.code = None
        k32 = _k32()
        self.shown = k32.CreateEventW(None, True, False, SHOWN_EVENT)
        self.up = k32.CreateEventW(None, True, False, UP_EVENT)
        for h in (self.shown, self.up):
            if h:
                k32.ResetEvent(h)

    # ── the window ──────────────────────────────────────────────────────
    def open_window(self) -> None:
        """The first frame (StaticWindow), then the first window over it, when SlantUI has one to offer (an older
        SlantUI has not: the app then opens as it always did, with nothing on screen until its own window)."""
        info = self.info
        size = info.get("size") or (1440, 900)
        self.static = StaticWindow(info, size)
        if self.static.show():
            log(f"first frame up after {time.perf_counter() - self.t0:.2f} s")
            # the app at once: the windows that follow open on this one's rectangle, in the screen's pixels
            self.native, self.since = self.static.rect, self.static.since
            self.spawn()
            if self.static.alone:           # turning by itself: no toolkit (its imports would hold the motion up)
                return
        try:
            from slantui import firstwindow
        except ImportError:
            self.static.close()
            return
        self.qapp = firstwindow.application(APP_ID)
        self.qapp.setQuitOnLastWindowClosed(False)
        got = self.assets.result(ASSETS_WAIT_S)
        self.pictures_due = got is None
        mark, icon = got if got is not None else (None, None)
        fresh = self.bands(mark)
        lk = info.get("look") if isinstance(info.get("look"), dict) else None
        # the first frame turning already: the first window takes its motion on from there (a SlantUI without
        # `since` starts its own)
        moving = {"since": self.static.since} if self.native is not None and self.static.parts else {}
        try:
            try:
                self.first = firstwindow.FirstWindow(
                    palette=info.get("palette"), look=lk, logo=mark, text=info.get("text", ""),
                    band=fresh, size=(int(size[0]), int(size[1])), title=TITLE, icon=icon,
                    close_rect=info.get("close"), **moving)
            except TypeError:
                if not moving:
                    raise
                self.first = firstwindow.FirstWindow(
                    palette=info.get("palette"), look=lk, logo=mark, text=info.get("text", ""),
                    band=fresh, size=(int(size[0]), int(size[1])), title=TITLE, icon=icon,
                    close_rect=info.get("close"))
        except (KeyError, TypeError, ValueError):     # a first.json of another version: drawn from the tokens
            self.first = firstwindow.FirstWindow(palette=info.get("palette"), logo=mark, title=TITLE, icon=icon)
        if self.native is not None:         # onto the first frame's rectangle, before it is shown
            place_native(int(self.first.winId()), self.native)
        self.first.closed.connect(self.cancel)
        self.first.shown.connect(self.first_shown)

    def first_shown(self) -> None:
        """The first window's first frame is on screen: the first frame goes, the app is told when this motion
        began (its Splash carries it on), and is started now if there was no first frame to start it at."""
        self.static.close()
        try:
            (LAUNCH / FIRST_SINCE).write_text(f"{self.first.since:.3f}", encoding="utf-8")
        except OSError:
            pass
        if self.native is None:
            self.native, self.since = native_rect(int(self.first.winId())), self.first.since
        self.spawn()

    def bands(self, mark: bytes | None) -> list[str]:
        """The bands, one per screen scale, drawn with the mark the app has now (a band of a mark changed since
        waits for the app's next)."""
        bands = self.info.get("bands") if isinstance(self.info.get("bands"), dict) else {}
        return [str(LAUNCH / n) for n, c in bands.items() if mark is not None and c == crc(mark)
                and (LAUNCH / n).is_file()]

    # ── the app ─────────────────────────────────────────────────────────
    def command(self) -> list[str]:
        """The app's process: quick_start.py when the app has it, else the app's own file."""
        entry = self.app.parent / ENTRY
        return self.python + [str(entry if entry.is_file() else self.app)] + self.args

    def spawn(self) -> None:
        """The app, once: as soon as there is a frame on screen (the first frame, or the first window's)."""
        if self.proc is not None or self.code == "cancelled":
            return
        if self.native is not None:         # where the windows are, in the screen's pixels, and since when
            self.env[FIRST_ENV] = ",".join(str(int(v)) for v in self.native) + f",{self.since or time.time():.3f}"
        err = LAUNCH / "last-error.txt"
        try:
            LAUNCH.mkdir(parents=True, exist_ok=True)
            out = open(err, "wb")
        except OSError:
            out = subprocess.DEVNULL
        try:
            ctypes.windll.user32.AllowSetForegroundWindow(ASFW_ANY)   # its window may come to the front
        except (OSError, AttributeError):
            pass
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(self.command(), env=self.env, stderr=out, stdout=subprocess.DEVNULL,
                                     stdin=subprocess.DEVNULL, cwd=str(self.cwd), creationflags=flags)
        log(f"started pid {self.proc.pid} after {time.perf_counter() - self.t0:.2f} s: {self.python[0]}")

    def cancel(self) -> None:
        """The first window was closed before the app's was up: the start is called off."""
        log("closed while starting")
        if self.proc is not None and self.proc.poll() is None:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.code = "cancelled"
        self._quit()

    def _quit(self) -> None:
        if self.qapp is not None:
            self.qapp.quit()

    def _set(self, handle) -> bool:
        return bool(handle) and _k32().WaitForSingleObject(handle, 0) == 0

    def tick(self) -> bool:
        """One look at the app; False once there is nothing more to wait for."""
        now = time.perf_counter() - self.t0
        if self.static is not None and self.static.cancelled and self.code != "cancelled":
            self.cancel()                   # the first frame, alone, closed by the person
            self.static.close()
            return False
        if getattr(self, "pictures_due", False) and self.first is not None:
            got = self.assets.result()
            if got is not None:                     # the app's mark and icon, read at last from its folder
                self.pictures_due = False
                self.first.set_pictures(logo=got[0], icon=got[1], band=self.bands(got[0]) or None)
        if self.proc is None:
            if self.code == "cancelled":
                return False
            if now > SPAWN_LATEST_S:        # a first window never painted (a locked screen): the app all the same
                self.spawn()
            return True
        if self.shown_at is None and self._set(self.shown):
            self.shown_at = now
            log(f"window after {now:.1f} s")
            if self.static is not None:
                self.static.close()
            if self.first is not None:
                self.first.close()
        if self._set(self.up):
            if self.first is not None and self.first.isVisible():
                self.first.close()
            log(f"up after {now:.1f} s")
            self.code = 0
            return False
        code = self.proc.poll()
        if code is not None:
            if self.static is not None:
                self.static.close()
            if self.first is not None:
                self.first.close()
            if self.code != "cancelled":
                self.code = code
            if code != 0 and self.code != "cancelled":
                log(f"exited with code {code}")
                if (code == NOT_READABLE_EXIT and self.shown_at is None and self.on_unreadable is not None
                        and not readable(self.app)):
                    self.on_unreadable()
                elif self.on_fail is not None:
                    self.on_fail(LAUNCH / "last-error.txt")
            return False
        # a window on screen and no page for this long: the app is left to itself
        return self.shown_at is None or now - self.shown_at < UP_WAIT_S

    def run(self) -> int:
        self.open_window()
        if self.first is None:
            self.spawn()
            try:
                while self.tick():
                    time.sleep(POLL_MS / 1000)
            finally:
                if self.static is not None:
                    self.static.close()
            return 0 if self.code in (0, None, "cancelled") else 1
        from PyQt6.QtCore import QTimer
        timer = QTimer()
        timer.setInterval(POLL_MS)
        timer.timeout.connect(lambda: self.tick() or self._quit())
        timer.start()
        self.first.shown.connect(lambda: log(f"first window up after {time.perf_counter() - self.t0:.2f} s"))
        self.first.show()
        self.qapp.exec()
        timer.stop()
        return 0 if self.code in (0, None, "cancelled") else 1


def native_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    r = wintypes.RECT()
    u32 = ctypes.windll.user32
    u32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.c_void_p]
    if not u32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(r)):
        return None
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def place_native(hwnd: int, rect) -> None:
    """A window onto a rectangle in the screen's pixels (first_window.place_native does the same for the app's)."""
    u32 = ctypes.windll.user32
    u32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, wintypes.UINT]
    x, y, w, h = (int(v) for v in rect)
    u32.SetWindowPos(wintypes.HWND(hwnd), None, x, y, w, h, 0x0004 | 0x0010)   # NOZORDER, NOACTIVATE


def run(app: Path, args: list[str], python: list[str], **kw) -> int:
    """The installed app's start (impactguide_start.pyw), and this file's own."""
    return Start(app, args, python, **kw).run()


# ── the desktop shortcut's start ─────────────────────────────────────────────
def launcher(*extra: str) -> None:
    """launcher.ps1 beside this file, with no console: the wait for the drive, or the message of a failed start."""
    ps1 = Path(__file__).resolve().parent / "launcher.ps1"
    if not ps1.is_file():
        return
    subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-STA", "-File", str(ps1), *extra],
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), close_fds=True)


def target() -> Path | None:
    try:
        data = json.loads((LAUNCH / "target.json").read_text(encoding="utf-8-sig"))
        return Path(data["app"]) if data.get("app") else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main(argv: list[str]) -> int:
    if os.name == "nt":
        k32 = _k32()
        k32.SetPriorityClass(k32.GetCurrentProcess(), ABOVE_NORMAL)
    given = None
    if len(argv) >= 2 and argv[0] == "--app":       # launcher.ps1, once it has found the folder
        given, argv = Path(argv[1]), argv[2:]
    if already_open():
        log("already open: brought to the front")
        return 0
    app = given or target()
    log(f"click: target=[{app}]")
    if app is None or (given is None and not drive_present(app)):
        release()                                   # the launcher takes over, and starts this again when it can
        launcher()
        return 0
    start = Start(app, argv, [sys.executable], on_fail=lambda err: launcher("-Failed", str(err)),
                  on_unreadable=None if given else (lambda: (release(), launcher())))
    return start.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
