"""The calibration board shown on a monitor at a known size.

Instead of a printed sheet, ImpactGuide shows the ChArUco calibration board on a monitor, inside its own window: in
the work area of the page, laid out for the pixels of that area. Every square is a whole number of the monitor's own
pixels, drawn without resampling, so the true size follows from the monitor's pixel pitch: square (mm) = pixels per
square x pitch (mm per pixel). Nothing has to be printed or measured; a ruler check, once per monitor, can make the
pitch more exact. window_area tells where a window is on its monitor, in physical pixels.

Same family as the A4 board of calibrate_charuco.py: a cv2.aruco.CharucoBoard with DICT_5X5_100 markers about 3/4 of
the square, so calibrate_charuco.py, marker_pose.py and live_ar.py find both with the same code. Only the number of
squares and their size follow the monitor.

Pixel pitch (mm per physical pixel), from what the monitor and Windows say:
    edid      the image size in whole mm of the monitor's first detailed timing, for its native resolution: the
              registry's copy of the EDID, or WMI when the registry has none
    qt        QScreen.physicalSize() in mm, which Qt also reads from the monitor
    edid_cm   the basic image size of the EDID in whole cm: a last resort
    ruler     the length between the arrows measured with a ruler and typed in the app, once per monitor
A size in whole mm is the true one rounded or cut by the maker, so the true size lies between n - 0.5 and n + 1 mm and
is taken as n + 0.25: the pitch is the least squares fit of square pixels to width and height, within about 0.75 mm
over the width. A size whose aspect does not match the pixels' (some monitors give an aspect code, projectors 0) is not
used. Physical pixels: the mode Windows drives the monitor at (EnumDisplaySettings), which is Qt's logical size times
its device pixel ratio (Windows display scaling: 125 %, 150 %...). In a mode other than the native one the GPU or the
monitor stretches the picture: the board is then neither crisp nor of a known size, and the app says so.

Layout: the largest board that fits inside a quiet margin (MARGIN_MM on three sides, TOP_MM on top for the dimension
line), with squares of whole pixels within SQUARE_MM, markers of a whole multiple of 7 pixels (5 x 5 bits and their
black border) centred in their square by whole pixels, at most 100 markers (the dictionary's size); of the boards
within AREA_TOL of the largest area, the one whose square is nearest the A4 board's 24 mm. Above the board two thin
lines continue its outer left and right edges, with an arrow between them: what a ruler should read.

Usage:
    python screen_board.py info [--png DIR]   the monitors of this computer, their pitch and the board each shows
    python screen_board.py test [--work DIR]  pitch on fake screens (100, 125, 150 %, portrait, scaled mode, bad
                                              EDID), layouts, the board image found by the calibration and live code,
                                              a calibration on synthetic photos of it; exit code 1 on failure. DIR
                                              must be empty, missing, or a folder of an earlier test (it is emptied)

Each monitor has a key (what its ruler check is kept under): its EDID id and serial number and its pixels; when the
EDID gives no serial (WMI, many panels) the serial text, else where Windows says it is plugged in, so that two
identical monitors keep their own checks; a check kept under the older key of a monitor still applies while that
key names one monitor only.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import subprocess
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

import arrowhead
import calibrate_charuco

_LOGGER = logging.getLogger("screen_board")

WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact")) / "screen_board"
WMI_TIMEOUT_S = 10.0                                    # PowerShell's WMI query, at most
DICTIONARY = calibrate_charuco.BOARD["dictionary"]
MARKER_RATIO = calibrate_charuco.BOARD["marker"] / calibrate_charuco.BOARD["square"]    # 0.75, as the A4 board
TARGET_MM = calibrate_charuco.BOARD["square"]           # 24 mm: of two boards about as large, the nearer
SQUARE_MM = (18.0, 36.0)                                # square sizes tried
MIN_SQUARES = (5, 4)
MAX_MARKERS = 100                                       # DICT_5X5_100
AREA_TOL = 0.03                                         # boards within 3 % of the largest area count as large
MARKER_BITS = 7                                         # 5 x 5 bits and a black border of one bit
MARGIN_MM = 8.0                                         # quiet margin left, right and below the board
TOP_MM = 14.0                                           # above the board: the dimension line and its length
MIN_MARGIN_PX, MIN_TOP_PX = 24, 56
LINE_GAP_MM = 6.0                                       # from the dimension line to the board
EXT_MM = 2.5                                            # the extension lines go this far past the dimension line
ARROW_MM = (arrowhead.LENGTH_MM, arrowhead.HALF_MM)     # arrowhead length and half width: the one arrowhead
LINE_PX = 2                                             # lines two pixels thick, centred on the board's edges
EDID_OFFSET_MM, EDID_HALF_MM = 0.25, 0.75               # whole mm, rounded or cut: n + 0.25 +/- 0.75
CM_OFFSET_MM, CM_HALF_MM = 2.5, 7.5                     # whole cm, the same in cm
ASPECT_TOL = {"edid": 0.03, "qt": 0.03, "edid_cm": 0.06}
PITCH_RANGE = (0.05, 1.5)                               # mm: anything else is not a monitor's pixel
AGREE = 0.01                                            # edid and qt within 1 % agree
RULER_MAX = 0.05                                        # a reading more than 5 % off the expected one is a misreading
RULER_HALF_MM = 0.3                                     # a ruler read at both lines


# ── what the monitors say ────────────────────────────────────────────────────
def parse_edid(data):
    """The fields of an EDID base block that size the monitor: maker and product, serial, name, the basic size in cm,
    the image size in mm and the resolution of the first detailed timing (the native one). None if not an EDID."""
    b = bytes(data or b"")
    if len(b) < 128 or b[:8] != bytes.fromhex("00ffffffffffff00"):
        return None
    m = (b[8] << 8) | b[9]
    make = "".join(chr(64 + ((m >> s) & 31)) for s in (10, 5, 0))
    product = f"{b[10] | (b[11] << 8):04X}"
    out = {"make": make, "product": product, "id": make + product, "serial": int.from_bytes(b[12:16], "little"),
           "serial_text": "", "name": "", "texts": [], "cm": (b[21], b[22]), "mm": None, "native": None}
    for o in (54, 72, 90, 108):
        d = b[o:o + 18]
        if d[0] or d[1]:
            if out["native"] is None:
                out["native"] = (d[2] | ((d[4] & 0xF0) << 4), d[5] | ((d[7] & 0xF0) << 4))
                out["mm"] = (d[12] | ((d[14] & 0xF0) << 4), d[13] | ((d[14] & 0x0F) << 8))
        elif d[3] in (0xFC, 0xFE, 0xFF):
            text = d[5:18].split(b"\n")[0].decode("cp437", "replace").strip()
            if d[3] == 0xFC:
                out["name"] = out["name"] or text
            elif d[3] == 0xFF:
                out["serial_text"] = out["serial_text"] or text
            elif text:
                out["texts"].append(text)
    return out


def windows_monitors():
    """{GDI device name (as QScreen.name()): {"mode": (w, h) px, "hz", "orientation", "string", "instance", "edid"}}
    for the monitors on the desktop. The EDID is the registry's copy, else WMI's (slower); {} off Windows."""
    if sys.platform != "win32":
        return {}
    try:
        out = _windows_monitors()
    except Exception:      # noqa: BLE001  a monitor that cannot be read is sized by Qt alone
        _LOGGER.exception("the monitors could not be read from Windows; Qt alone sizes them")
        return {}
    if any(not m["edid"] for m in out.values()):
        wmi = wmi_edid()
        for m in out.values():
            if not m["edid"] and m["instance"] in wmi:
                m["edid"] = wmi[m["instance"]]
    return out


def _windows_monitors():
    import ctypes
    import winreg
    from ctypes import wintypes

    class DisplayDevice(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32), ("DeviceString", wintypes.WCHAR * 128),
                    ("StateFlags", wintypes.DWORD), ("DeviceID", wintypes.WCHAR * 128),
                    ("DeviceKey", wintypes.WCHAR * 128)]

    class DevMode(ctypes.Structure):
        _fields_ = [("dmDeviceName", wintypes.WCHAR * 32), ("dmSpecVersion", wintypes.WORD),
                    ("dmDriverVersion", wintypes.WORD), ("dmSize", wintypes.WORD), ("dmDriverExtra", wintypes.WORD),
                    ("dmFields", wintypes.DWORD), ("dmPositionX", wintypes.LONG), ("dmPositionY", wintypes.LONG),
                    ("dmDisplayOrientation", wintypes.DWORD), ("dmDisplayFixedOutput", wintypes.DWORD),
                    ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short), ("dmYResolution", ctypes.c_short),
                    ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short), ("dmFormName", wintypes.WCHAR * 32),
                    ("dmLogPixels", wintypes.WORD), ("dmBitsPerPel", wintypes.DWORD), ("dmPelsWidth", wintypes.DWORD),
                    ("dmPelsHeight", wintypes.DWORD), ("dmDisplayFlags", wintypes.DWORD),
                    ("dmDisplayFrequency", wintypes.DWORD), ("dmICMMethod", wintypes.DWORD),
                    ("dmICMIntent", wintypes.DWORD), ("dmMediaType", wintypes.DWORD), ("dmDitherType", wintypes.DWORD),
                    ("dmReserved1", wintypes.DWORD), ("dmReserved2", wintypes.DWORD),
                    ("dmPanningWidth", wintypes.DWORD), ("dmPanningHeight", wintypes.DWORD)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)     # a private copy: its function types stay ours
    out = {}
    i = 0
    while True:
        a = DisplayDevice()
        a.cb = ctypes.sizeof(a)
        if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(a), 0):
            break
        i += 1
        if not a.StateFlags & 0x1:                         # DISPLAY_DEVICE_ATTACHED_TO_DESKTOP
            continue
        dm = DevMode()
        dm.dmSize = ctypes.sizeof(dm)
        if not user32.EnumDisplaySettingsW(a.DeviceName, -1, ctypes.byref(dm)):    # ENUM_CURRENT_SETTINGS
            continue
        m = DisplayDevice()
        m.cb = ctypes.sizeof(m)
        edid, instance, string = None, "", ""
        # the monitor's interface name: \\?\DISPLAY#<maker and product>#<instance>#{class}
        if user32.EnumDisplayDevicesW(a.DeviceName, 0, ctypes.byref(m), 1):    # EDD_GET_DEVICE_INTERFACE_NAME
            string = m.DeviceString
            parts = m.DeviceID.split("#")
            if len(parts) >= 3:
                instance = f"DISPLAY\\{parts[1]}\\{parts[2]}_0"
                key = rf"SYSTEM\CurrentControlSet\Enum\DISPLAY\{parts[1]}\{parts[2]}\Device Parameters"
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                        edid = parse_edid(winreg.QueryValueEx(k, "EDID")[0])
                except OSError:
                    edid = None
        out[a.DeviceName] = {"mode": (int(dm.dmPelsWidth), int(dm.dmPelsHeight)), "hz": int(dm.dmDisplayFrequency),
                             "orientation": int(dm.dmDisplayOrientation), "string": string, "instance": instance,
                             "edid": edid}
    return out


WMI_SCRIPT = (
    "$b=@{}; Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorBasicDisplayParams | ForEach-Object "
    "{ $b[$_.InstanceName]=@($_.MaxHorizontalImageSize,$_.MaxVerticalImageSize) }; "
    "$n=@{}; Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorID | ForEach-Object "
    "{ $n[$_.InstanceName]=@((($_.ManufacturerName | Where-Object {$_}) | ForEach-Object {[char]$_}) -join '', "
    "(($_.ProductCodeID | Where-Object {$_}) | ForEach-Object {[char]$_}) -join '', "
    "(($_.UserFriendlyName | Where-Object {$_}) | ForEach-Object {[char]$_}) -join '') }; "
    "@(Get-CimInstance -Namespace root\\wmi -ClassName WmiMonitorListedSupportedSourceModes | ForEach-Object "
    "{ $m=$_.MonitorSourceModes[$_.PreferredMonitorSourceModeIndex]; [pscustomobject]@{i=$_.InstanceName; "
    "px=@($m.HorizontalActivePixels,$m.VerticalActivePixels); mm=@($m.HorizontalImageSize,$m.VerticalImageSize); "
    "cm=$b[$_.InstanceName]; id=$n[$_.InstanceName]} }) | ConvertTo-Json -Compress -Depth 4")


def wmi_edid(timeout=WMI_TIMEOUT_S):
    """{WMI instance name: the fields of parse_edid} from WMI's monitor classes, when the registry has no EDID."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", WMI_SCRIPT],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
        rows = json.loads(r.stdout or "[]")
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        _LOGGER.warning("WMI gave no monitor sizes: %s", e)
        return {}
    return wmi_rows(rows)


def _pair(v):
    """(a, b) of two positive whole numbers, or None."""
    try:
        a, b = (int(x) for x in v)
    except (TypeError, ValueError):
        return None
    return (a, b) if a > 0 and b > 0 else None


def wmi_rows(rows):
    """The monitors of WMI's answer (a list of rows, or one row) as parse_edid fields; a row without its instance or
    pixels is left out (and said in the log). A size that is not given or not positive (a TV or a projector) is None,
    as parse_edid has it: the monitor keeps its identity, and its pitch comes from elsewhere."""
    out = {}
    for row in rows if isinstance(rows, list) else [rows]:
        try:
            ident = (list(row.get("id") or []) + ["", "", ""])[:3]
            make, product, name = (str(x or "") for x in ident)
            px, mm = _pair(row.get("px")), _pair(row.get("mm"))
            if not row.get("i") or px is None:
                raise ValueError(f"instance {row.get('i')}, pixels {row.get('px')}")
            out[str(row["i"])] = {"make": make, "product": product, "id": make + product, "serial": 0,
                                  "serial_text": "", "name": name, "texts": [], "cm": _pair(row.get("cm")) or (0, 0),
                                  "mm": mm, "native": px}
        except (AttributeError, TypeError, ValueError) as e:
            _LOGGER.warning("a monitor row of WMI left out: %s", e)
    return out


def qt_screen(screen, primary=False):
    """What a QScreen says, as plain values (the input of estimate)."""
    g, ps = screen.geometry(), screen.physicalSize()
    return {"device": screen.name(), "geometry": (g.x(), g.y(), g.width(), g.height()),
            "dpr": float(screen.devicePixelRatio()), "mm": (float(ps.width()), float(ps.height())),
            "manufacturer": screen.manufacturer(), "model": screen.model(), "serial": screen.serialNumber(),
            "primary": bool(primary), "refresh_hz": float(screen.refreshRate())}


def window_area(hwnd):
    """((x, y, w, h) of a window's client area on the desktop, (x, y, w, h) of its monitor, the monitor's device name),
    all in physical pixels, from Windows; None off Windows or when Windows does not answer. The process is per-monitor
    aware (Qt 6), so these are the monitor's own pixels."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class MonitorInfo(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD), ("szDevice", wintypes.WCHAR * 32)]

    u = ctypes.WinDLL("user32", use_last_error=True)          # a private copy: the argument types set here stay here
    u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    u.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    u.MonitorFromWindow.restype = ctypes.c_void_p
    u.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MonitorInfo)]
    h = wintypes.HWND(hwnd)
    rc, pt = wintypes.RECT(), wintypes.POINT(0, 0)
    if not u.GetClientRect(h, ctypes.byref(rc)) or not u.ClientToScreen(h, ctypes.byref(pt)):
        return None
    mi = MonitorInfo()
    mi.cbSize = ctypes.sizeof(MonitorInfo)
    if not u.GetMonitorInfoW(u.MonitorFromWindow(h, 2), ctypes.byref(mi)):    # MONITOR_DEFAULTTONEAREST
        return None
    m = mi.rcMonitor
    return ((pt.x, pt.y, rc.right - rc.left, rc.bottom - rc.top), (m.left, m.top, m.right - m.left, m.bottom - m.top),
            mi.szDevice)


# ── the pitch ────────────────────────────────────────────────────────────────
def oriented(size, px):
    """A size (w, h) turned like the pixels (a monitor turned to portrait)."""
    if size and size[0] != size[1] and (size[0] > size[1]) != (px[0] > px[1]):
        return (size[1], size[0])
    return tuple(size) if size else None


def fit_pitch(px, mm, offset=0.0):
    """Least squares pitch (mm) of square pixels px = (w, h) over an image of mm + offset."""
    w, h = px
    return ((mm[0] + offset) * w + (mm[1] + offset) * h) / (w * w + h * h)


def plausible(px, mm, tol):
    try:
        if not mm or len(mm) != 2 or min(mm) <= 0 or min(px) <= 0:
            return False
    except TypeError:                   # a size with an empty part
        return False
    if abs((mm[0] / mm[1]) / (px[0] / px[1]) - 1) > tol:
        return False
    return PITCH_RANGE[0] <= fit_pitch(px, mm) <= PITCH_RANGE[1]


def physical_px(qt, win=None):
    """((w, h) physical pixels, where from): the mode Windows drives the monitor at, else Qt's logical size times its
    device pixel ratio, snapped to the native size when within the rounding of a fractional ratio."""
    if win and win.get("mode"):
        return tuple(int(v) for v in win["mode"]), "windows"
    _, _, w, h = qt["geometry"]
    dpr = float(qt.get("dpr") or 1.0)
    px = (int(round(w * dpr)), int(round(h * dpr)))
    native = ((win or {}).get("edid") or {}).get("native")
    if native:
        native = oriented(native, px)
        if all(abs(a - b) <= math.ceil(dpr) for a, b in zip(px, native)):
            return tuple(native), "qt"
    return px, "qt"


def estimate(qt, win=None):
    """The monitor of one QScreen (qt_screen) with what Windows says of it (windows_monitors): physical pixels, pitch,
    its source and how sure it is, and whether the sources agree. pitch_mm is None when nothing says the size."""
    win = win or {}
    edid = win.get("edid") or {}
    px, px_from = physical_px(qt, win)
    native = oriented(edid.get("native"), px) if edid.get("native") else None
    base = native or px                 # the pixels the sizes in mm are given for
    sizes = {"edid": oriented(edid.get("mm"), base), "qt": oriented(qt.get("mm"), base),
             "edid_cm": oriented(tuple(10 * c for c in edid["cm"]), base) if edid.get("cm") else None}
    offsets = {"edid": (EDID_OFFSET_MM, EDID_HALF_MM), "qt": (EDID_OFFSET_MM, EDID_HALF_MM),
               "edid_cm": (CM_OFFSET_MM, CM_HALF_MM)}
    sources = {k: fit_pitch(base, mm, offsets[k][0]) for k, mm in sizes.items() if plausible(base, mm, ASPECT_TOL[k])}
    source = next((k for k in ("edid", "qt", "edid_cm") if k in sources), None)
    is_native = native is None or tuple(px) == tuple(native)
    pitch = rel_u = None
    if source:
        mm = sizes[source]
        pitch = sources[source] * (base[0] / px[0] if not is_native else 1.0)   # a mode stretched over the panel
        rel_u = offsets[source][1] * (base[0] + base[1]) / ((mm[0] + offsets[source][0]) * base[0]
                                                            + (mm[1] + offsets[source][0]) * base[1])
    fine = [sources[k] for k in ("edid", "qt") if k in sources]
    spread = max(fine) / min(fine) - 1 if len(fine) > 1 else 0.0
    legacy = None
    if edid.get("id"):
        key = legacy = f"{edid['id']}-{int(edid.get('serial') or 0):08X}"
        if not edid.get("serial"):      # no serial number: the serial text, else where it is plugged in
            text = str(edid.get("serial_text") or qt.get("serial") or "").strip()
            where = str(win.get("instance") or "")
            if text:
                key = f"{edid['id']}-{text}"
            elif where:
                key = f"{edid['id']}-{zlib.crc32(where.encode('utf-8')):08X}"
    else:
        key = "-".join(str(v) for v in (qt.get("manufacturer"), qt.get("model"), qt.get("serial")) if v) or qt["device"]
    label = (edid.get("name") or (edid.get("texts") or [""])[-1] or f"{edid.get('make', '')} {edid.get('product', '')}"
             .strip() or " ".join(v for v in (qt.get("manufacturer"), qt.get("model")) if v) or qt["device"])
    # a word with no letter nor digit is a placeholder (Qt's manufacturer "-" for a monitor without EDID): left out
    label = " ".join(w for w in str(label).split() if any(c.isalnum() for c in w)) or qt["device"]
    size_mm = [px[0] * pitch, px[1] * pitch] if pitch else None
    return {"device": qt["device"], "key": f"{key}-{px[0]}x{px[1]}",
            "legacy_key": f"{legacy}-{px[0]}x{px[1]}" if legacy and legacy != key else None,
            "label": label, "primary": bool(qt.get("primary")),
            "geometry": list(qt["geometry"]), "dpr": float(qt.get("dpr") or 1.0), "px": list(px), "px_from": px_from,
            "native": list(native) if native else None, "is_native": bool(is_native), "pitch_mm": pitch,
            "pitch_data_mm": pitch, "rel_u": rel_u, "source": source or "none",
            "sources": {k: round(v, 6) for k, v in sources.items()}, "agree": None if len(fine) < 2 else spread <= AGREE,
            "spread": spread, "size_mm": size_mm,
            "diag_in": math.hypot(*size_mm) / 25.4 if size_mm else None, "refresh_hz": qt.get("refresh_hz") or
            win.get("hz"), "ruler": None}


def with_ruler(mon, correction):
    """The monitor with its ruler check (what ImpactGuide keeps per monitor: pitch_mm, measured_mm, expected_mm,
    date) in place of the pitch from its data."""
    if not correction or not correction.get("pitch_mm"):
        return mon
    mon = dict(mon)
    mon.update(pitch_mm=float(correction["pitch_mm"]), source="ruler", ruler=dict(correction),
               rel_u=RULER_HALF_MM / float(correction["measured_mm"]))
    mon["size_mm"] = [mon["px"][0] * mon["pitch_mm"], mon["px"][1] * mon["pitch_mm"]]
    mon["diag_in"] = math.hypot(*mon["size_mm"]) / 25.4
    return mon


def monitors(qt_screens, win=None, corrections=None):
    """estimate() of every screen, with the ruler checks applied. A monitor without a serial number whose check is
    not under its key, while no other monitor of its model and size is connected, keeps the one check of its model and
    size: the one under its older key (one that two identical monitors would share), or the one kept when it was
    plugged in elsewhere (its key follows where it is plugged in)."""
    win = win or {}
    corrections = corrections or {}
    out = [estimate(q, win.get(q["device"])) for q in qt_screens]
    olds = [m.get("legacy_key") for m in out]
    for m in out:
        old = m.get("legacy_key")
        if not old or m["key"] in corrections or olds.count(old) != 1:
            continue
        model, size = old.split("-")[0], old.rsplit("-", 1)[-1]
        same = [k for k in corrections if k.split("-")[0] == model and k.rsplit("-", 1)[-1] == size]
        if old in corrections:
            m["key"] = old
        elif len(same) == 1:
            m["key"] = same[0]
    return [with_ruler(m, corrections.get(m["key"])) for m in out]


def default_monitor(mons):
    """The monitor the board goes on unless one is chosen: the smallest (of two of the same size, not the main one),
    among those whose size is known if any is."""
    known = [m for m in mons if m["pitch_mm"]] or list(mons)
    if not known:
        return None
    return min(known, key=lambda m: (round(m["diag_in"] or 1e9, 1), m["primary"]))["key"]


# ── the board ────────────────────────────────────────────────────────────────
def marker_px(square_px):
    """Marker side in px for a square: a multiple of 7 (each bit a whole number of pixels), centred by whole pixels
    (square - marker even), nearest 3/4 of the square, at least 2 px of white around it. None if there is none."""
    best = None
    k0 = MARKER_RATIO * square_px / MARKER_BITS
    for k in range(max(2, int(k0) - 2), int(k0) + 3):
        m = MARKER_BITS * k
        if (square_px - m) % 2 or square_px - m < 4:
            continue
        if best is None or abs(m - MARKER_RATIO * square_px) < abs(best - MARKER_RATIO * square_px):
            best = m
    return best


def fit_markers(nx, ny):
    """The most squares within nx x ny with at most MAX_MARKERS markers (one in every other square)."""
    best = (0, 0)
    for y in range(min(ny, 2 * MAX_MARKERS + 1), 0, -1):
        x = min(nx, (2 * MAX_MARKERS + 1) // y)
        if x * y // 2 <= MAX_MARKERS and x * y > best[0] * best[1]:
            best = (x, y)
    return best


def _check_area(px, pitch_mm):
    """(w, h, pitch) checked: whole positive pixels and a finite pitch above 0, else ValueError."""
    try:
        w, h = (int(v) for v in px)
        p = float(pitch_mm)
    except (TypeError, ValueError):
        raise ValueError(f"not an area and a pitch: {px}, {pitch_mm}") from None
    if w <= 0 or h <= 0 or not math.isfinite(p) or p <= 0:
        raise ValueError(f"no board fits {w} x {h} px at a pitch of {p} mm")
    return w, h, p


def _margins(w, h, p):
    margin = max(MIN_MARGIN_PX, math.ceil(MARGIN_MM / p))
    top = max(MIN_TOP_PX, math.ceil(TOP_MM / p))
    return margin, top, w - 2 * margin, h - top - margin


def _placed(w, h, p, margin, top, ah, s, m, nx, ny):
    x0 = (w - nx * s) // 2
    y0 = top + (ah - ny * s) // 2
    line_y = y0 - max(12, round(LINE_GAP_MM / p))
    return {"size_px": [w, h], "pitch_mm": p, "squares": [nx, ny], "square_px": s, "marker_px": m,
            "origin_px": [x0, y0], "margin_px": margin, "top_px": top,
            "square_mm": s * p, "marker_mm": m * p, "width_mm": nx * s * p, "height_mm": ny * s * p,
            "line_y": line_y, "ext_top": line_y - max(6, round(EXT_MM / p)),
            # whole pixels long: the base, across the line, falls between two columns of pixels and is sharp
            "arrow_px": [max(8, round(ARROW_MM[0] / p)), max(8, round(ARROW_MM[0] / p)) * arrowhead.HALF]}


def place(px, pitch_mm, held):
    """A board already chosen (held: squares, square_px, marker_px, pitch_mm, as a layout or spec gives them) laid out
    again on an area of px = (w, h) physical pixels at pitch_mm, centred: the same squares of the same pixels, so the
    same board in mm. None when the pitch is another (another monitor) or the board no longer fits the area."""
    if not held or abs(float(held.get("pitch_mm") or 0) - float(pitch_mm)) > 1e-12:
        return None
    w, h = (int(v) for v in px)
    p = float(pitch_mm)
    margin, top, aw, ah = _margins(w, h, p)
    nx, ny = (int(v) for v in held["squares"])
    s, m = int(held["square_px"]), int(held["marker_px"])
    if nx * s > aw or ny * s > ah:
        return None
    return _placed(w, h, p, margin, top, ah, s, m, nx, ny)


def layout(px, pitch_mm, hold=None):
    """The board for a screen, or an area of a page on it, of px = (w, h) physical pixels at pitch_mm: squares,
    pixels per square and per marker, where it goes, its true size, and the dimension line above it. With hold (a
    board already in use, see place) that board is kept, centred, as long as it fits and the pitch is the same: an
    area moved or resized a little does not change a board that photos have been taken of. Raises ValueError if no
    board fits."""
    w, h, p = _check_area(px, pitch_mm)
    kept = place(px, pitch_mm, hold) if hold else None
    if kept:
        return kept
    margin, top, aw, ah = _margins(w, h, p)
    found = []
    for s in range(max(16, math.ceil(SQUARE_MM[0] / p)), math.floor(SQUARE_MM[1] / p) + 1):
        m = marker_px(s)
        if m is None:
            continue
        nx, ny = fit_markers(aw // s, ah // s)
        if nx < MIN_SQUARES[0] or ny < MIN_SQUARES[1]:
            continue
        found.append((nx * ny * s * s, s, m, nx, ny))
    if not found:
        raise ValueError(f"no board fits a screen of {w} x {h} px at {p:.4f} mm per px")
    biggest = max(f[0] for f in found)
    _, s, m, nx, ny = min((f for f in found if f[0] >= (1 - AREA_TOL) * biggest),
                          key=lambda f: abs(f[1] * p - TARGET_MM))
    return _placed(w, h, p, margin, top, ah, s, m, nx, ny)


def min_px(pitch_mm):
    """The smallest area (w, h) in physical pixels that holds a board at this pitch: MIN_SQUARES of the smallest
    square, with the margins. An area smaller than this shows no board."""
    _, _, p = _check_area((1, 1), pitch_mm)
    s = max(16, math.ceil(SQUARE_MM[0] / p))
    margin = max(MIN_MARGIN_PX, math.ceil(MARGIN_MM / p))
    top = max(MIN_TOP_PX, math.ceil(TOP_MM / p))
    return MIN_SQUARES[0] * s + 2 * margin, MIN_SQUARES[1] * s + top + margin


def board_px(lay):
    """The board in pixel units (square = square_px): the one the image is drawn from."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    return cv2.aruco.CharucoBoard(tuple(lay["squares"]), float(lay["square_px"]), float(lay["marker_px"]), d)


def make_board(lay):
    """The board in mm, for detection and pose (calibrate_charuco.make_board with this layout)."""
    return calibrate_charuco.make_board(lay["square_mm"], lay["squares"], lay["marker_mm"])


def board_tile(lay):
    """The board alone, one pixel per screen pixel: black squares, white squares each with its marker."""
    b = board_px(lay)
    s, mp = lay["square_px"], lay["marker_px"]
    nx, ny = lay["squares"]
    d = b.getDictionary()
    tile = np.zeros((ny * s, nx * s), np.uint8)
    for mid, c in zip(b.getIds().ravel(), b.getObjPoints()):
        c = np.asarray(c).reshape(-1, 3)
        x, y = int(round(c[:, 0].min())), int(round(c[:, 1].min()))
        tile[y // s * s:(y // s + 1) * s, x // s * s:(x // s + 1) * s] = 255
        tile[y:y + mp, x:x + mp] = cv2.aruco.generateImageMarker(d, int(mid), mp, borderBits=1)
    return tile


def render(lay):
    """The whole screen, one array pixel per screen pixel: white, the board, and above it the two extension lines
    (2 px, centred on the board's outer left and right edges) with the dimension line and its arrowheads between
    them. The words go on top in the app, in the band above (none over the board or the lines' ends). Black and
    white only, but for the arrowheads' edges (arrowhead.fill: grey where a pixel is partly covered); the line runs
    into each arrowhead and stops inside it, so the point is the triangle's own."""
    w, h = lay["size_px"]
    s = lay["square_px"]
    nx, ny = lay["squares"]
    x0, y0 = lay["origin_px"]
    x1 = x0 + nx * s
    img = np.full((h, w), 255, np.uint8)
    img[y0:y0 + ny * s, x0:x1] = board_tile(lay)
    ya, ye = lay["line_y"], lay["ext_top"]
    half = LINE_PX // 2
    for x in (x0, x1):
        img[ye:y0, x - half:x + LINE_PX - half] = 0
    length = float(lay["arrow_px"][0])
    stop = math.ceil(arrowhead.inset(length, LINE_PX))
    img[ya - half:ya + LINE_PX - half, x0 + stop:x1 - stop] = 0
    yc = ya - 0.5                                    # centre of the two rows of the line
    for tip, d in ((x0 - 0.5, -1), (x1 - 0.5, 1)):    # tips on the lines' centres, pointing out
        arrowhead.fill(img, (tip, yc), (d, 0), length, 0)
    return img


def corners_px(lay):
    """Where each chessboard corner of the board is on the screen, px (pixel centres at integers)."""
    obj = board_px(lay).getChessboardCorners()[:, :2]
    return obj + np.array(lay["origin_px"], np.float64) - 0.5


def spec(lay, mon=None):
    """The board for the Calibration and Live AR pages: what calibrate_charuco and live_ar need, and where it is."""
    out = {"kind": "screen", "squares": list(lay["squares"]), "square_mm": lay["square_mm"],
           "marker_mm": lay["marker_mm"], "dictionary": DICTIONARY, "square_px": lay["square_px"],
           "marker_px": lay["marker_px"], "pitch_mm": lay["pitch_mm"], "width_mm": lay["width_mm"],
           "height_mm": lay["height_mm"]}
    if mon:
        out.update(monitor=mon["key"], monitor_label=mon["label"], pitch_source=mon["source"], rel_u=mon["rel_u"],
                   is_native=mon["is_native"])
    return out


def ruler_pitch(lay, measured_mm):
    """The pitch from the length read with a ruler between the arrows of this layout. Raises ValueError when the
    reading is more than RULER_MAX off the expected length (the wrong lines, or a slip)."""
    measured_mm = float(measured_mm)
    if not abs(measured_mm / lay["width_mm"] - 1) <= RULER_MAX:
        raise ValueError(f"{measured_mm} mm is too far from {lay['width_mm']:.1f} mm")
    return measured_mm / (lay["squares"][0] * lay["square_px"])


# ── command line ─────────────────────────────────────────────────────────────
def _qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv[:1])


def describe(m):
    if not m["pitch_mm"]:
        return f"{m['device']} {m['label']}: {m['px'][0]} x {m['px'][1]} px, size unknown"
    lay = layout(m["px"], m["pitch_mm"])
    return (f"{m['device']} {m['label']}{' (main)' if m['primary'] else ''}: {m['px'][0]} x {m['px'][1]} px "
            f"({m['px_from']}), scale {m['dpr']:g}, {m['diag_in']:.1f} in, pitch {m['pitch_mm']:.5f} mm from "
            f"{m['source']} (+/-{100 * m['rel_u']:.2f} %), sources {m['sources']}, agree {m['agree']}"
            f"{'' if m['is_native'] else ', NOT NATIVE ' + str(m['native'])}\n"
            f"    board {lay['squares'][0]} x {lay['squares'][1]}, square {lay['square_px']} px = "
            f"{lay['square_mm']:.3f} mm, marker {lay['marker_px']} px = {lay['marker_mm']:.3f} mm, "
            f"{lay['width_mm']:.2f} x {lay['height_mm']:.2f} mm (+/-{lay['width_mm'] * m['rel_u']:.2f} mm across)")


def cmd_info(args):
    app = _qt_app()
    qt = [qt_screen(s, s is app.primaryScreen()) for s in app.screens()]
    mons = monitors(qt, windows_monitors())
    default = default_monitor(mons)
    for m in mons:
        print(("* " if m["key"] == default else "  ") + describe(m))
        if args.png and m["pitch_mm"]:
            args.png.mkdir(parents=True, exist_ok=True)
            out = args.png / f"screen_board_{m['key']}.png"
            cv2.imwrite(str(out), render(layout(m["px"], m["pitch_mm"])))
            print(f"    {out}")


# ── test ─────────────────────────────────────────────────────────────────────
FAILS = []


def expect(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def fake(geometry, dpr, mm, mode=None, edid_mm=None, native=None, cm=None, primary=False, name="fake"):
    """A QScreen and what Windows says of it, as estimate() takes them."""
    qt = {"device": name, "geometry": geometry, "dpr": dpr, "mm": mm, "manufacturer": "", "model": name,
          "serial": "", "primary": primary}
    edid = None
    if edid_mm is not None or native is not None or cm is not None:
        edid = {"make": "TST", "product": "0001", "id": "TST0001", "serial": 7, "name": name, "texts": [],
                "cm": cm or (0, 0), "mm": edid_mm, "native": native}
    return qt, {"mode": mode, "edid": edid} if (mode or edid) else None


# (name, qt and windows, true pitch mm or None, expected physical px, native)
FAKES = [
    ("15.6 in FHD at 100 %", fake((0, 0, 1920, 1080), 1.0, (344, 193), (1920, 1080), (344, 193), (1920, 1080), (34, 19)),
     0.17925, (1920, 1080), True),
    ("15.6 in FHD at 125 %", fake((0, 0, 1536, 864), 1.25, (344, 193), (1920, 1080), (344, 193), (1920, 1080),
                                  (34, 19)), 0.17925, (1920, 1080), True),
    ("15.6 in FHD at 150 %, Qt only", fake((0, 0, 1280, 720), 1.5, (344, 193)), 0.17925, (1920, 1080), True),
    ("27 in 4K at 150 %", fake((1920, 0, 2560, 1440), 1.5, (597, 336), (3840, 2160), (597, 336), (3840, 2160),
                               (60, 34)), 0.1554, (3840, 2160), True),
    ("13.3 in 2560 x 1600 at 150 %, no mode", fake((0, 0, 1707, 1067), 1.5, (286, 179), None, (286, 179),
                                                   (2560, 1600), (29, 18)), 0.1120, (2560, 1600), True),
    ("24 in FHD turned to portrait at 125 %", fake((0, 0, 864, 1536), 1.25, (527, 296), (1080, 1920), (527, 296),
                                                   (1920, 1080), (53, 30)), 0.2745, (1080, 1920), True),
    ("15.6 in FHD driven at 1280 x 720", fake((0, 0, 1280, 720), 1.0, (344, 193), (1280, 720), (344, 193),
                                               (1920, 1080), (34, 19)), 0.17925 * 1.5, (1280, 720), False),
    ("EDID with an aspect code, Qt with 0 mm", fake((0, 0, 1920, 1080), 1.0, (0, 0), (1920, 1080), (16, 9),
                                                     (1920, 1080), (79, 0)), None, (1920, 1080), True),
    ("EDID size wrong, Qt right", fake((0, 0, 1920, 1200), 1.0, (518, 324), (1920, 1200), (160, 90), (1920, 1200),
                                       (52, 32)), 0.2700, (1920, 1200), True),
]


def test_pitch():
    for name, (qt, win), truth, px, native in FAKES:
        m = estimate(qt, win)
        if truth is None:
            expect(m["pitch_mm"] is None and m["source"] == "none", f"{name}: no usable size, pitch unknown")
            continue
        err = m["pitch_mm"] / truth - 1
        expect(tuple(m["px"]) == px and m["is_native"] == native and abs(err) < 0.004,
               f"{name}: {m['px'][0]} x {m['px'][1]} px ({m['px_from']}), pitch {m['pitch_mm']:.5f} mm from "
               f"{m['source']}, {100 * err:+.2f} % from the true {truth} mm (+/-{100 * m['rel_u']:.2f} %), "
               f"native {m['is_native']}")
    # the same panel at 100, 125 and 150 %: the same pixels and the same pitch
    p = [estimate(*FAKES[k][1])["pitch_mm"] for k in (0, 1, 2)]
    expect(max(p) - min(p) < 1e-12, f"display scaling changes nothing: {[round(v, 6) for v in p]}")
    # the sources disagree: the EDID is kept and the disagreement said
    qt, win = fake((0, 0, 1920, 1080), 1.0, (355, 200), (1920, 1080), (344, 193), (1920, 1080), (34, 19))
    m = estimate(qt, win)
    expect(m["source"] == "edid" and m["agree"] is False and m["spread"] > 0.02,
           f"EDID 344 x 193 mm against Qt 355 x 200 mm: EDID kept, disagreement {100 * m['spread']:.1f} % said")
    m = estimate(*FAKES[0][1])
    expect(m["agree"] is True and m["source"] == "edid", "EDID and Qt agree on the laptop panel")
    # the default monitor: the smaller, and of two alike not the main one
    laptop = fake((0, 0, 1920, 1080), 1.0, (344, 193), primary=True, name="laptop")
    a = estimate(*laptop)
    b = estimate(*fake((1920, 0, 2560, 1440), 1.0, (597, 336), name="big"))
    c = estimate(*fake((0, 0, 1920, 1080), 1.0, (344, 193), name="laptop2"))
    expect(default_monitor([a, b]) == a["key"] and default_monitor([b, a, c]) == c["key"],
           "default monitor: the smaller one; of two alike, the one that is not the main")
    qt4k, _ = fake((0, 0, 2560, 1440), 1.5, (878, 485), name="PHI PHL BDM4037U")
    qt4k.update(manufacturer="-", device="PHL BDM4037U")
    got = estimate(qt4k, None)["label"]
    expect(got == "PHI PHL BDM4037U", f"a monitor without EDID, Qt's manufacturer '-': named {got!r}, no dash")
    # a ruler check replaces the pitch, for that monitor only
    lay = layout(a["px"], a["pitch_mm"])
    new = ruler_pitch(lay, lay["width_mm"] * 1.004)
    r = monitors([laptop[0]], {}, {a["key"]: {"pitch_mm": new, "measured_mm": lay["width_mm"] * 1.004,
                                              "expected_mm": lay["width_mm"], "date": "2026-10-15"}})[0]
    expect(abs(new / a["pitch_mm"] - 1.004) < 1e-9 and r["source"] == "ruler" and r["pitch_mm"] == new
           and r["rel_u"] < a["rel_u"],
           f"ruler: {lay['width_mm'] * 1.004:.2f} mm read for {lay['width_mm']:.2f} mm gives pitch "
           f"{new:.5f} mm (+0.40 %), +/-{100 * r['rel_u']:.2f} %")
    try:
        ruler_pitch(lay, lay["width_mm"] * 1.08)
        expect(False, "ruler: a reading 8 % off is refused")
    except ValueError:
        expect(True, "ruler: a reading 8 % off is refused")
    # EDID parsing on a real block (a 15.6 in BOE panel: 1920 x 1080, 344 x 193 mm)
    blob = base64.b64decode(
        "AP///////wAwrrpAAAAAAAAcAQSlIhN44lj1plhVnSYOUFQAAAABAQEBAQEBAQEBAQEBAQEBnDuANnE4PEAwIDYAWMEQ"
        "AAAaqy+ANnE4PEAwIDYAWMEQAAAaAAAADwDRCTDRCTAeCgIJ5ZIHAAAA/gBOVjE1NkZITS1ONjEKAOI=")
    e = parse_edid(blob)
    expect(e is not None and e["id"] == "LEN40BA" and e["native"] == (1920, 1080) and e["mm"] == (344, 193)
           and e["cm"] == (34, 19) and e["texts"] == ["NV156FHM-N61"],
           f"EDID: {e and e['id']}, {e and e['native']} px, {e and e['mm']} mm, {e and e['cm']} cm, "
           f"{e and e['texts']}")


def test_layouts():
    cases = [(f[0], estimate(*f[1])) for f in FAKES if f[2]]
    cases += [(f"pitch {p} mm on {w} x {h}", {"px": [w, h], "pitch_mm": p}) for w, h, p in
              ((1366, 768, 0.252), (2880, 1800, 0.1046), (3840, 2160, 0.1245), (1280, 1024, 0.294),
               (5120, 1440, 0.2331))]
    try:
        layout((1024, 600), 0.1497)
        expect(False, "a 7 in screen of 1024 x 600 px is refused: too small for 5 x 4 squares of 18 mm")
    except ValueError:
        expect(True, "a 7 in screen of 1024 x 600 px is refused: too small for 5 x 4 squares of 18 mm")
    bad = []
    for name, m in cases:
        lay = layout(m["px"], m["pitch_mm"])
        s, mp = lay["square_px"], lay["marker_px"]
        nx, ny = lay["squares"]
        x0, y0 = lay["origin_px"]
        w, h = lay["size_px"]
        ok = (isinstance(s, int) and mp % MARKER_BITS == 0 and (s - mp) % 2 == 0 and abs(mp / s - MARKER_RATIO) <= 0.1
              and nx * ny // 2 <= MAX_MARKERS and nx >= MIN_SQUARES[0] and ny >= MIN_SQUARES[1]
              and SQUARE_MM[0] - 1e-9 <= lay["square_mm"] <= SQUARE_MM[1] + 1e-9
              and x0 >= lay["margin_px"] and x0 + nx * s <= w - lay["margin_px"]
              and y0 >= lay["top_px"] and y0 + ny * s <= h - lay["margin_px"]
              and abs(lay["square_mm"] - s * m["pitch_mm"]) < 1e-12 and lay["ext_top"] > 0)
        if not ok:
            bad.append(name)
        print(f"      {name}: {nx} x {ny} squares of {s} px = {lay['square_mm']:.3f} mm, markers {mp} px, "
              f"board {lay['width_mm']:.1f} x {lay['height_mm']:.1f} mm "
              f"({100 * nx * ny * s * s / (w * h):.0f} % of the screen)")
    expect(not bad, f"layouts: whole pixels, markers of 7k px centred, within margins and the dictionary, "
                    f"{len(cases)} screens ({bad or 'all good'})")
    # a board in an area of the app's window, of any size: the area is the page's, the pitch the monitor's
    p = 0.17921
    sizes = [(round(1920 * f), 1032) for f in (0.35, 0.45, 0.5, 0.62)] + [(700, 1300), (1920, 1032)]
    lays = [layout(px, p) for px in sizes]
    ok = all(L["origin_px"][0] + L["squares"][0] * L["square_px"] <= px[0] - L["margin_px"]
             and abs(L["width_mm"] - L["squares"][0] * L["square_px"] * p) < 1e-9 for L, px in zip(lays, sizes))
    expect(ok and len({L["squares"][0] for L in lays}) > 2,
           "areas of the page of any size, upright too: " + ", ".join(f"{px[0]} x {px[1]} px {L['squares'][0]} x "
                                                             f"{L['squares'][1]}" for L, px in zip(lays, sizes)))
    small = min_px(p)
    try:
        layout((small[0] - 40, small[1]), p)
        refused = False
    except ValueError:
        refused = True
    expect(refused and layout(small, p)["squares"] == list(MIN_SQUARES),
           f"smallest area that holds a board at {p} mm: {small[0]} x {small[1]} px, 5 x 4 squares; narrower refused")
    # the board held while it fits: same squares and pixels, centred again; another pitch or too small: laid out anew
    held = layout((960, 1032), p)
    moved = layout((1010, 1040), p, hold=held)
    shrunk = layout((700, 1032), p, hold=held)
    other = layout((960, 1032), p * 1.1, hold=held)
    expect(moved["squares"] == held["squares"] and moved["square_px"] == held["square_px"]
           and moved["origin_px"] != held["origin_px"] and shrunk["squares"] != held["squares"]
           and other["pitch_mm"] != held["pitch_mm"],
           f"held board: kept when the area grows ({moved['squares']}, centred again), laid out anew when it no "
           f"longer fits ({shrunk['squares']}) or the monitor changes")


def test_image(work):
    import marker_pose
    work.mkdir(parents=True, exist_ok=True)
    for k in (0, 3, 5):
        name, (qt, win), *_ = FAKES[k]
        m = estimate(qt, win)
        lay = layout(m["px"], m["pitch_mm"])
        img = render(lay)
        nx, ny = lay["squares"]
        s = lay["square_px"]
        x0, y0 = lay["origin_px"]
        tile = img[y0:y0 + ny * s, x0:x0 + nx * s]
        same = np.array_equal(tile, board_px(lay).generateImage((nx * s, ny * s), marginSize=0, borderBits=1))
        x1 = x0 + nx * s
        # grey only on the arrowheads' edges, the two arrows mirror images of each other
        L, ya = lay["arrow_px"][0], lay["line_y"]
        hy, hx = math.ceil(L * arrowhead.HALF) + 1, math.ceil(L) + 1
        heads = np.zeros(img.shape, bool)
        heads[ya - hy:ya + hy, x0 - 1:x0 + hx] = heads[ya - hy:ya + hy, x1 - hx:x1 + 1] = True
        crisp = set(np.unique(img[~heads]).tolist()) <= {0, 255} and len(np.unique(img[heads])) > 4
        left, right = img[ya - hy:ya + hy, x0 - 1:x0 + hx], img[ya - hy:ya + hy, x1 - hx:x1 + 1]
        crisp = crisp and np.array_equal(left, right[:, ::-1]) and np.array_equal(left, left[::-1])
        lines = (img[y0 - 1, x0 - 1:x0 + 1].tolist() == [0, 0] and img[y0 - 1, x0 + 1] == 255
                 and img[y0 - 1, x1 - 1:x1 + 1].tolist() == [0, 0] and img[y0 - 1, x1 - 2] == 255)
        expect(same and crisp and lines, f"{name}: image black and white but for the arrowheads' edges (the two "
                                         f"mirror images), the board as OpenCV draws it, the lines on its outer edges")
        path = work / f"screen_{k}.png"
        cv2.imwrite(str(path), img)
        # the calibration's detection, on the image itself
        board = make_board(lay)
        views, skipped, _ = calibrate_charuco.detect([path], board)
        truth = corners_px(lay)
        n = (nx - 1) * (ny - 1)
        err = float("nan")
        if views:
            obj, imgp = views[path.name]
            ids = np.rint(obj[:, :2].reshape(-1, 2) / lay["square_mm"]).astype(int)
            want = (ids * s + np.array(lay["origin_px"]) - 0.5)
            err = float(np.abs(imgp.reshape(-1, 2) - want).max())
        expect(views and len(views[path.name][0]) == n and err < 0.05,
               f"{name}: calibrate_charuco.detect finds {len(views[path.name][0]) if views else 0} corners of "
               f"{n}, at most {err:.3f} px from where they are drawn")
        # the live detection (marker_pose), on the image and on a photo-like view of it
        c, cids, nm = marker_pose.detect(marker_pose.make_detector(board), img)
        e1 = float(np.abs(c - truth[cids]).max()) if len(cids) else float("nan")
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]],
                                                    [0, img.shape[0]]]),
                                        np.float32([[60, 40], [img.shape[1] - 150, 110],
                                                    [img.shape[1] - 90, img.shape[0] - 30], [20, img.shape[0] - 70]]))
        photo = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA, borderValue=128)
        photo = cv2.GaussianBlur(photo, (0, 0), 1.2)
        photo = np.clip(photo.astype(np.float32) * 0.7 + 30 + np.random.default_rng(k).normal(0, 2, photo.shape),
                        0, 255).astype(np.uint8)
        c2, cids2, _ = marker_pose.detect(marker_pose.make_detector(board), photo)
        t2 = cv2.perspectiveTransform(truth[cids2].reshape(-1, 1, 2), H).reshape(-1, 2) if len(cids2) else None
        e2 = float(np.abs(c2 - t2).max()) if len(cids2) else float("nan")
        expect(len(cids) == n and nm == nx * ny // 2 and e1 < 0.05 and len(cids2) == n and e2 < 0.3,
               f"{name}: marker_pose.detect finds {len(cids)} corners and {nm} markers, at most {e1:.3f} px off; "
               f"on a tilted, blurred, noisy view {len(cids2)} corners, at most {e2:.3f} px off")


def test_calibration(work):
    """A calibration on synthetic photos of the board of the laptop panel, with the board's own layout."""
    m = estimate(*FAKES[0][1])
    lay = layout(m["px"], m["pitch_mm"])
    out = work / "synth"
    quiet = lambda line: None   # noqa: E731
    calibrate_charuco.synth(out, 14, seed=5, board=make_board(lay), log=quiet)
    cal = calibrate_charuco.run_calibrate(out, None, out / "truth.json", lay["square_mm"], lay["squares"],
                                          lay["marker_mm"], log=quiet)
    e = cal["error"]
    expect(cal["rms_px"] < calibrate_charuco.MAX_RMS and abs(e["fx_rel"]) < calibrate_charuco.MAX_FOCAL_ERROR
           and abs(e["fy_rel"]) < calibrate_charuco.MAX_FOCAL_ERROR and e["model_max_px"] < 1.0
           and cal["board"]["squares"] == lay["squares"],
           f"calibration on {cal['n_views']} synthetic photos of the {lay['squares'][0]} x {lay['squares'][1]} "
           f"board: RMS {cal['rms_px']:.3f} px, fx {100 * e['fx_rel']:+.3f} %, fy {100 * e['fy_rel']:+.3f} %, "
           f"model {e['model_max_px']:.2f} px at most")
    # board.json in the folder: the command line reads it
    (out / calibrate_charuco.BOARD_FILE).write_text(json.dumps(spec(lay, m)), encoding="utf-8")
    sq, squares, marker = calibrate_charuco.folder_board(out)
    expect(squares == tuple(lay["squares"]) and abs(sq - lay["square_mm"]) < 1e-12
           and abs(marker - lay["marker_mm"]) < 1e-12, "board.json of a photo folder read back")


def test_monitors():
    """What Windows says, read with care: WMI rows with parts missing, two identical monitors without a serial
    number, a ruler check kept under the older key, an area or pitch that cannot hold a board."""
    rows = [{"i": "DISPLAY\\AAA0001\\1_0", "px": [1920, 1080], "mm": [344, 193], "cm": [34, 19], "id": ["AAA", "0001",
                                                                                                          "Panel"]},
            {"i": "DISPLAY\\AAA0001\\2_0", "px": [1920, 1080], "mm": None, "id": ["AAA", "0001", ""]},
            {"i": "DISPLAY\\AAA0001\\3_0", "px": [None, 1080], "mm": [344, 193]},
            {"px": [1920, 1080], "mm": [344, 193]}, "not a row",
            {"i": "DISPLAY\\TV00001\\4_0", "px": [3840, 2160], "mm": [0, 0], "id": ["TV0", "0001", "TV"]}]
    got = wmi_rows(rows)
    expect(list(got) == ["DISPLAY\\AAA0001\\1_0", "DISPLAY\\AAA0001\\2_0", "DISPLAY\\TV00001\\4_0"]
           and got["DISPLAY\\AAA0001\\1_0"]["mm"] == (344, 193) and got["DISPLAY\\AAA0001\\2_0"]["mm"] is None
           and got["DISPLAY\\TV00001\\4_0"]["mm"] is None and got["DISPLAY\\TV00001\\4_0"]["id"] == "TV00001"
           and not plausible((1920, 1080), (None, None), 0.03) and not plausible((3840, 2160), None, 0.03),
           "WMI rows: without pixels or instance left out; without a size (a TV) kept with no size, as an EDID "
           "without one; never a crash of every board slot")
    edid = {"make": "AAA", "product": "0001", "id": "AAA0001", "serial": 0, "serial_text": "", "name": "Panel",
            "texts": [], "cm": (34, 19), "mm": (344, 193), "native": (1920, 1080)}
    twins = []
    for k in (1, 2):
        qt, _ = fake((1920 * (k - 1), 0, 1920, 1080), 1.0, (344, 193), name=f"\\\\.\\DISPLAY{k}")
        twins.append((qt, {"mode": (1920, 1080), "instance": f"DISPLAY\\AAA0001\\{k}_0", "edid": edid}))
    qts = [q for q, _ in twins]
    win = {q["device"]: w for q, w in twins}
    plain = monitors(qts, win)
    old = "AAA0001-00000000-1920x1080"
    kept = monitors(qts[:1], {qts[0]["device"]: twins[0][1]}, {old: {"pitch_mm": 0.18, "measured_mm": 300}})
    shared = monitors(qts, win, {old: {"pitch_mm": 0.18, "measured_mm": 300}})
    expect(plain[0]["key"] != plain[1]["key"] and plain[0]["legacy_key"] == old
           and kept[0]["key"] == old and kept[0]["source"] == "ruler"
           and all(m["key"] != old and m["source"] != "ruler" for m in shared),
           "two identical monitors without a serial keep their own ruler checks; a check under the older key still "
           "applies to a monitor that is alone with it")
    # the monitor plugged in elsewhere: its key changes with the port; alone of its model, it keeps its check
    moved = (dict(twins[0][1], instance="DISPLAY\\AAA0001\\7_0"))
    check1 = {plain[0]["key"]: {"pitch_mm": 0.18, "measured_mm": 300}}
    alone = monitors(qts[:1], {qts[0]["device"]: moved}, check1)
    both = monitors(qts, win, {**check1, plain[1]["key"]: {"pitch_mm": 0.181, "measured_mm": 300}})
    moved_both = monitors(qts[:1], {qts[0]["device"]: moved}, {**check1, plain[1]["key"]: {"pitch_mm": 0.181,
                                                                                            "measured_mm": 300}})
    expect(alone[0]["key"] == plain[0]["key"] and alone[0]["source"] == "ruler"
           and [m["ruler"]["pitch_mm"] for m in both] == [0.18, 0.181] and moved_both[0]["source"] != "ruler",
           "a monitor without a serial plugged in elsewhere keeps its ruler check while it is the only one of its "
           "model; with two checks of that model, neither is guessed")
    bad = []
    for px, p in (((1920, 1080), 0.0), ((1920, 1080), float("nan")), ((0, 1080), 0.18), ((1920, 1080), -1)):
        try:
            layout(px, p)
            bad.append((px, p))
        except ValueError:
            pass
    expect(not bad, f"an area or a pitch that cannot hold a board: ValueError, never a division by zero {bad or ''}")


def cmd_test(args):
    try:
        work = calibrate_charuco.test_folder(args.work)
    except ValueError as e:
        expect(False, str(e))
        return 1
    test_pitch()
    test_layouts()
    test_monitors()
    test_image(work)
    test_calibration(work)
    print(f"\n{len(FAILS)} failures" if FAILS else "\nall checks passed", flush=True)
    return 1 if FAILS else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("info")
    a.add_argument("--png", type=Path, help="also write the board of each monitor here")
    a = sub.add_parser("test")
    a.add_argument("--work", type=Path, default=WORK / "selftest")     # a folder of the test's own
    args = ap.parse_args()
    if args.cmd == "info":
        cmd_info(args)
    else:
        sys.exit(cmd_test(args))


if __name__ == "__main__":
    main()
