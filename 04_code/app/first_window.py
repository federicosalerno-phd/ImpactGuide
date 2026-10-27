"""The app's side of its first window (scripts/impactguide_boot.py, slantui.firstwindow).

The start puts a window on screen the moment the icon is clicked and starts the app behind it, telling it where that
window is. The app opens its own window on the same rectangle, its loading screen turning on from where that one was
(opening), says when its window is on screen (raise_event SHOWN_EVENT), so the first one can go, and holds the
instance mutex while it runs (hold_instance), so that a click while it is open only brings it to the front.

Once its page is up, it leaves the next start what it needs to look like this window (keep): first.json (the
palette, the line under the ring, the size, where the band's close button is, which mark each band was drawn with),
band-<scale>.png (the band as the window drew it, per screen scale), and the start itself, when the launch folder holds an older copy of it.
The mark and the icon are not copied: the start reads the app's own (ui/), so a new drawing is seen at once. A file
is written only when it changed, by a worker thread, and a failure to write costs nothing but a plainer first window
next time.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import zlib
from pathlib import Path

FIRST_ENV = "IMPACTGUIDE_FIRST_WINDOW"      # x,y,w,h (screen pixels),since of the first window (impactguide_boot)
FIRST_SINCE = "first-since"                 # in the launch folder: when the first window's motion began
SHOWN_EVENT = r"Local\ImpactGuideShown"     # the app's window is on screen (impactguide_boot.Start waits on it)
MUTEX = "ImpactGuideRunning"                # one app at a time (impactguide_boot.already_open, the installer)
EVENT_MODIFY_STATE = 0x0002
APP = Path(__file__).resolve().parent
BOOT = APP / "scripts" / "impactguide_boot.py"
LAUNCH = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ImpactGuide" / "launch"

log = logging.getLogger("impactguide")
_held: list = []                            # the mutex handle, for the life of the process


def opening(env=None, launch: Path | None = None) -> tuple[tuple[int, int, int, int], float] | None:
    """The first window's rectangle (in the screen's pixels) and when its motion began, or None when the app was not
    started behind one. The start is told the rectangle at once, and when the motion began a moment later, once its
    first window is up (FIRST_SINCE): that one wins when it is of this start."""
    raw = (os.environ if env is None else env).get(FIRST_ENV, "")
    try:
        x, y, w, h, since = raw.split(",")
        rect, since = (int(x), int(y), int(w), int(h)), float(since)
    except ValueError:
        return None
    if rect[2] <= 0 or rect[3] <= 0:
        return None
    try:
        later = float((Path(launch or LAUNCH) / FIRST_SINCE).read_text(encoding="utf-8"))
        if since <= later <= since + 120:
            since = later
    except (OSError, ValueError):
        pass
    return rect, since


def place_native(win, rect) -> None:
    """The window onto a rectangle in the screen's pixels, before it is shown (Qt's own pixels may round a screen's
    differently from one process to the next; the screen's cannot)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        u32 = ctypes.windll.user32
        u32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, wintypes.UINT]
        x, y, w, h = (int(v) for v in rect)
        u32.SetWindowPos(wintypes.HWND(int(win.winId())), None, x, y, w, h, 0x0004 | 0x0010)   # NOZORDER, NOACTIVATE
    except (OSError, AttributeError, ValueError):
        log.info("window not placed on the first window", exc_info=True)


def hold_instance() -> None:
    """The mutex a start looks for, held until the process ends (Windows only)."""
    if os.name != "nt" or _held:
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.CreateMutexW.restype = ctypes.c_void_p
        k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        handle = k32.CreateMutexW(None, 0, MUTEX)
        if handle:
            _held.append(handle)
    except (OSError, AttributeError):
        log.info("instance mutex not held", exc_info=True)


def raise_event(name: str) -> None:
    """Raises a named event a start waits on; nothing when no start waits (the app started by hand)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.OpenEventW.restype = ctypes.c_void_p
        k32.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        k32.SetEvent.argtypes = [ctypes.c_void_p]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = k32.OpenEventW(EVENT_MODIFY_STATE, False, name)
        if handle:
            k32.SetEvent(handle)
            k32.CloseHandle(handle)
    except (OSError, AttributeError):
        log.info("event %s not raised", name, exc_info=True)


def band_picture(win, width: float):
    """The window's band as it drew it at the start, ``width`` wide, where its close button is ([x, y, w, h] in the
    band's pixels) and the scale it was drawn at; (None, None, 1.0) on a window with no band of its own."""
    band = getattr(win, "band", None)
    if band is None:
        return None, None, 1.0
    from slantui.shell.band import band_layout, render_band
    scale = band.scale()
    lay = band_layout(width, scale, band.brand)
    img = render_band(width, scale, band.palette, band.brand, strip=band.strip, layout=lay)
    r = lay.buttons[-1]
    return img, [round(v / lay.scale, 2) for v in (r.x(), r.y(), r.width(), r.height())], scale


def first_picture(*, palette: str, text: str, mark: Path | None, band, size: tuple[int, int], scale: float):
    """The first frame of the next start's first window, as a picture at ``scale`` (BMP bytes, what Windows draws
    with nothing else loaded: impactguide_boot.StaticWindow), drawn by the first window itself, and its motion
    (the ring's and the bar's frames, each a BMP, firstwindow.FirstWindow.motion: the start turns them from its
    first frame on, never a still ring); (None, None) with a SlantUI that has no first window, the motion None with
    one that has no motion."""
    try:
        from slantui import firstwindow
    except ImportError:
        return None, None
    win = firstwindow.FirstWindow(look=firstwindow.look(palette), logo=Path(mark).read_bytes() if mark else None,
                                  text=text, band=band, size=size)
    try:
        win.resize(int(size[0]), int(size[1]))
        motion = None
        if hasattr(win, "motion"):
            m = win.motion(scale)
            motion = {"frame_ms": m["frame_ms"]}
            for part in ("ring", "bar"):
                motion[part] = {"rect": m[part]["rect"], "n": m[part]["n"], "bmp": _encode(m[part]["image"], "BMP")}
            if not all(motion[part]["bmp"] for part in ("ring", "bar")):
                motion = None
        return _encode(win.picture(scale), "BMP"), motion
    finally:
        win.deleteLater()


def _write_if_changed(path: Path, data: bytes) -> bool:
    try:
        if path.is_file() and path.read_bytes() == data:
            return False
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return True
    except OSError:
        log.info("could not keep %s for the next start", path.name, exc_info=True)
        return False


def look(palette: str) -> dict | None:
    """What the first window is drawn with (slantui.firstwindow.look), so the start need not read the tokens; None
    with a SlantUI that has no first window."""
    try:
        from slantui import firstwindow
    except ImportError:
        return None
    return firstwindow.look(palette)


def keep(launch: Path, *, palette: str, text: str, size: tuple[int, int], band=None, scale: float = 1.0,
         close=None, mark: Path | None = None, picture: bytes | None = None, picture_size=None, boot: Path = BOOT,
         motion: dict | None = None, wait: bool = False) -> threading.Thread | None:
    """What the next first window needs, into ``launch`` (the folder of the desktop launcher), by a worker thread.

    The band is kept once per screen scale (band-150.png: the first window opens on the screen of the pointer, which
    may not be the one the app was on), each with the fingerprint of the mark it was drawn with, and so is the
    picture of the first frame (first-150.bmp, first_picture), with its size in the screen's pixels, and its motion
    (first-150-ring.bmp and first-150-bar.bmp, each with its place, its frames and their interval).

    The start is refreshed only where the desktop launcher put a copy of it (install_shortcut.ps1): an installed
    app starts from its own folder. Returns the thread (joined first when ``wait``), or None with no launch folder.
    """
    launch = Path(launch)
    if not launch.is_dir():
        return None
    png = _png(band) if band is not None else None     # small: a band, encoded here, written by the thread
    lk = look(palette)

    def work():
        try:
            old = json.loads((launch / "first.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        old = old if isinstance(old, dict) else {}
        bands = dict(old["bands"]) if isinstance(old.get("bands"), dict) else {}
        info = {"palette": palette, "text": text, "size": [int(size[0]), int(size[1])]}
        if lk is not None:
            info["look"] = lk
        crc = None
        try:        # the mark of the loading screen and of the band: the start reads this one, and leaves out a band
            if mark is not None:        # drawn with another
                info["logo"] = Path(mark).resolve().relative_to(APP).as_posix()
                crc = f"{zlib.crc32(Path(mark).read_bytes()):08x}"
        except (OSError, ValueError):
            pass
        if png is not None and crc is not None:
            name = f"band-{round(scale * 100)}.png"
            _write_if_changed(launch / name, png)
            bands[name] = crc
        info["bands"] = {n: c for n, c in bands.items() if (launch / n).is_file()}
        pictures = dict(old["pictures"]) if isinstance(old.get("pictures"), dict) else {}
        if picture is not None and picture_size is not None:
            name = f"first-{round(scale * 100)}.bmp"
            _write_if_changed(launch / name, picture)
            pictures[name] = {"size": [int(picture_size[0]), int(picture_size[1])], "mark": crc}
            if motion is not None:
                moving = {"frame_ms": motion["frame_ms"]}
                for part in ("ring", "bar"):
                    file = f"first-{round(scale * 100)}-{part}.bmp"
                    _write_if_changed(launch / file, motion[part]["bmp"])
                    moving[part] = {"file": file, "rect": motion[part]["rect"], "n": motion[part]["n"]}
                pictures[name]["motion"] = moving
        info["pictures"] = {n: v for n, v in pictures.items() if (launch / n).is_file()}
        if close is not None or old.get("close"):
            info["close"] = close if close is not None else old["close"]
        _write_if_changed(launch / "first.json", json.dumps(info, ensure_ascii=False, indent=1).encode("utf-8"))
        try:
            if boot is not None and Path(boot).is_file() and (launch / boot.name).is_file():
                _write_if_changed(launch / boot.name, Path(boot).read_bytes())
        except OSError:
            log.info("could not keep %s for the next start", boot.name, exc_info=True)

    t = threading.Thread(target=work, name="first_window", daemon=True)
    t.start()
    if wait:
        t.join()
    return t


def _png(img) -> bytes | None:
    """A QImage as PNG bytes, here on the caller's thread (Qt's own encoder)."""
    return _encode(img, "PNG")


def _encode(img, kind: str) -> bytes | None:
    """A QImage as the bytes of a file of this kind (PNG, BMP), with Qt's own encoder."""
    try:
        from PyQt6.QtCore import QBuffer, QIODevice
    except ImportError:                     # pragma: no cover - the PyQt5 fallback of SlantUI
        from PyQt5.QtCore import QBuffer, QIODevice
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly if hasattr(QIODevice, "OpenModeFlag") else QIODevice.WriteOnly)
    ok = img.save(buf, kind)
    return bytes(buf.data()) if ok else None
