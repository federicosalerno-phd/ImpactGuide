"""ImpactGuide: the calibration board on a monitor, one image pixel on one screen pixel, full screen or in a window.

Two ways to show it. Full screen on a monitor of its own, when the computer has two (the app on one, the board on the
other). Or in a window of its own, when it has one: docked beside the app (Dock), the two sharing the monitor at any
split, which the operator moves by dragging the edge between them, as two snapped windows do. The window can be
undocked by moving it, and then moved, resized, snapped (Windows Snap, Win and the arrows, a double click fills the
screen) or put on another monitor. It belongs to the app's window: it stays above it and goes down with it.

Wherever it is, the board is laid out for the pixels it is shown on, the window's own area in physical pixels (the part
on its monitor), at the pitch of that monitor (screen_board: the EDID, Qt, a ruler check, with Windows scaling), and
laid out again when the window is moved, resized or goes to another monitor; `relaid` tells the app, which hands the new
board to the calibration and the live view. A board the photos of a calibration are of is held (`hold`,
screen_board.layout) while it still fits, so nudging the window changes nothing. The image carries the screen's device
pixel ratio and is drawn at a whole physical pixel, so Qt lays its pixels on the screen's one to one at any scaling.
Nothing zooms it: the wheel, pinch gestures and the zoom keys are taken and dropped. Esc or the small round button in
the corner closes it; full screen, the pointer hides when it stops, so it never covers a marker. The few words on it
(the length between the arrows, what the board is) are drawn in the band above the board, never over the board or the
ends of the lines.

    python board_window.py check [--scale 1.25] [--out DIR] [--window]

shows the board on the main monitor for a moment (full screen, or with --window in a window of 60 % of it), grabs the
screen and compares it with the image pixel for pixel (and finds the board in the grab with the calibration's
detection); with --scale, Qt scales the window itself as Windows does at that display scaling (QT_SCALE_FACTOR), which
tries the 125 % and 150 % path on a 100 % monitor. Prints a JSON line; exit code 1 when the screen does not hold the
image exactly.
"""
import json
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
# The project may sit on a synced drive: bytecode goes to a local cache, never next to the sources.
if not sys.pycache_prefix:
    sys.pycache_prefix = str(Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ImpactGuide" / "pycache")
sys.path.insert(0, str(APP.parent / "src"))

import numpy as np  # noqa: E402

from slantui.shell import win32  # noqa: E402
from slantui.shell.qt import (QColor, QCursor, QEvent, QFont, QImage, QObject, QPainter, QPointF, QRect,  # noqa: E402
                              QRectF, Qt, QTimer, QWidget, USE_QT6, pyqtSignal)

import screen_board  # noqa: E402

HIDE_POINTER_MS = 1500
RELAY_MS = 60               # a move or a resize is laid out again this long after it stops changing
INK = QColor(0, 0, 0)
GREY = QColor(110, 110, 110)
BUTTON = QColor(232, 232, 232)
BUTTON_HOT = QColor(205, 205, 205)
WM_NCHITTEST = 0x0084
HT = {"client": 1, "caption": 2, "left": 10, "right": 11, "top": 12, "topleft": 13, "topright": 14, "bottom": 15,
      "bottomleft": 16, "bottomright": 17}
EDGE_PX = 7                 # the band along the window's edges that resizes it, in logical pixels

if USE_QT6:
    _BASE = Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
    _ON_TOP = Qt.WindowType.WindowStaysOnTopHint
    _ESC = Qt.Key.Key_Escape
    _BLANK, _ARROW = Qt.CursorShape.BlankCursor, Qt.CursorShape.ArrowCursor
    _AA, _TEXT_AA = QPainter.RenderHint.Antialiasing, QPainter.RenderHint.TextAntialiasing
    _GRAY8 = QImage.Format.Format_Grayscale8
    _CENTER = Qt.AlignmentFlag.AlignCenter
    _LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    _OPAQUE = Qt.WidgetAttribute.WA_OpaquePaintEvent
    _NO_BG = Qt.WidgetAttribute.WA_NoSystemBackground
    _TOUCH = Qt.WidgetAttribute.WA_AcceptTouchEvents
    _SWALLOW = {QEvent.Type.NativeGesture, QEvent.Type.Gesture, QEvent.Type.TouchBegin, QEvent.Type.TouchUpdate,
                QEvent.Type.TouchEnd, QEvent.Type.Wheel}
    _NO_PEN = Qt.PenStyle.NoPen
    _SOLID = Qt.PenStyle.SolidLine
    _MINIMIZED = Qt.WindowState.WindowMinimized
else:  # pragma: no cover - PyQt5
    _BASE = Qt.Window | Qt.FramelessWindowHint
    _ON_TOP = Qt.WindowStaysOnTopHint
    _ESC = Qt.Key_Escape
    _BLANK, _ARROW = Qt.BlankCursor, Qt.ArrowCursor
    _AA, _TEXT_AA = QPainter.Antialiasing, QPainter.TextAntialiasing
    _GRAY8 = QImage.Format_Grayscale8
    _CENTER = Qt.AlignCenter
    _LEFT = Qt.AlignLeft | Qt.AlignVCenter
    _OPAQUE, _NO_BG, _TOUCH = Qt.WA_OpaquePaintEvent, Qt.WA_NoSystemBackground, Qt.WA_AcceptTouchEvents
    _SWALLOW = {QEvent.NativeGesture, QEvent.Gesture, QEvent.TouchBegin, QEvent.TouchUpdate, QEvent.TouchEnd,
                QEvent.Wheel}
    _NO_PEN = Qt.NoPen
    _SOLID = Qt.SolidLine
    _MINIMIZED = Qt.WindowMinimized


def to_qimage(img, dpr):
    """A grey image as a QImage that Qt draws one pixel per device pixel on a screen of this ratio."""
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape
    q = QImage(img.data, w, h, w, _GRAY8).copy()
    q.setDevicePixelRatio(dpr)
    return q


# ── where a window is, in physical pixels ────────────────────────────────────
def _native():
    from slantui.shell.qt import QApplication
    app = QApplication.instance()
    return os.name == "nt" and app is not None and app.platformName() == "windows"


def native_area(hwnd):
    """((x, y, w, h) of the client area on the desktop, (x, y, w, h) of its monitor, the monitor's device name), all in
    physical pixels, from Windows; None off Windows. The process is per-monitor aware (Qt 6), so these are pixels."""
    import ctypes
    from ctypes import wintypes

    class MonitorInfo(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD), ("szDevice", wintypes.WCHAR * 32)]

    u = ctypes.windll.user32
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


def _window_rect(hwnd):
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    r = wintypes.RECT()
    u.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


class BoardWindow(QWidget):
    """The board, full screen on `screen` (full=True) or in a window of its own at `rect` (logical pixels).
    monitor_of(device name) gives the monitor (screen_board.monitors) a device is, with its pitch. `closed` says why
    it went: "user" (Esc, the button) or "changed" (its monitor went away). `relaid` says the board changed: another
    size, another monitor, or none fits (error "too_small"). `moved` is every change of its place or size, for Dock."""

    closed = pyqtSignal(str)
    relaid = pyqtSignal()
    moved = pyqtSignal()

    def __init__(self, screen, monitor_of, words, full=True, rect=None, owner=None, hold=None):
        super().__init__(None, (_BASE | _ON_TOP) if full else _BASE)
        self.monitor_of, self.words, self.full = monitor_of, words, full
        self.hold = hold                # a board to keep while it fits (screen_board.place)
        self.screen_ = screen
        self.key = ""                   # the monitor's key (screen_board) the board is laid out for
        self.mon = None
        self.lay = None
        self.error = ""
        self.image = None
        self._img = None
        self.dpr = float(screen.devicePixelRatio())
        self.offset = (0, 0)            # where the laid out area starts in the window, physical pixels
        self.area = (0, 0)              # its size, physical pixels
        self._why = "user"
        self._hot = False
        self._watched = None
        self.setWindowTitle(words.get("title", "ImpactGuide"))
        self.setAttribute(_OPAQUE, True)
        self.setAttribute(_NO_BG, True)
        self.setAttribute(_TOUCH, True)
        self.setMouseTracking(True)
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(HIDE_POINTER_MS)
        self._idle.timeout.connect(lambda: self.setCursor(QCursor(_BLANK)) if self.full else None)
        self._relay = QTimer(self)
        self._relay.setSingleShot(True)
        self._relay.setInterval(RELAY_MS)
        self._relay.timeout.connect(self.relayout)
        self.winId()                    # the native window, so it can belong to the app's and be dressed
        handle = self.windowHandle()
        if owner is not None and handle is not None:
            handle.setTransientParent(owner)
        if handle is not None:
            handle.screenChanged.connect(self._screen_changed)
        if USE_QT6:
            self.setScreen(screen)
        if full:
            self.setGeometry(screen.geometry())
        else:
            if _native():
                win32.dress(int(self.winId()))    # a real frame under the hood: Windows snaps and moves it
            self.setGeometry(rect if rect is not None else screen.availableGeometry())
        self._watch(screen)

    # ── the screen it is on ──────────────────────────────────────────────
    def _watch(self, screen):
        if self._watched is not None:
            for sig in (self._watched.geometryChanged, self._watched.logicalDotsPerInchChanged,
                        self._watched.physicalDotsPerInchChanged):
                try:
                    sig.disconnect(self._screen_moved)
                except (TypeError, RuntimeError):
                    pass
        self._watched = screen
        if screen is not None:
            for sig in (screen.geometryChanged, screen.logicalDotsPerInchChanged, screen.physicalDotsPerInchChanged):
                sig.connect(self._screen_moved)

    def _screen_changed(self, screen):
        if screen is None:
            return
        self.screen_ = screen
        self._watch(screen)
        self._relay.start()

    def _screen_moved(self, *_):
        """The monitor changed under the board (resolution, scaling, place): full screen it follows the monitor, and
        either way the board is laid out again."""
        if self.full and self._watched is not None:
            self.setGeometry(self._watched.geometry())
        self._relay.start()

    def changed(self, *_):
        """Its monitor is gone: the window goes and says why."""
        self._why = "changed"
        self.close()

    # ── the board for where the window is now ────────────────────────────
    def measure(self):
        """(device pixel ratio, offset and size of the area the board goes in, physical pixels, monitor device): the
        window's own area on its monitor, from Windows; elsewhere (tests off screen) the window's size in pixels."""
        sc = self.screen() or self.screen_
        dpr = float(sc.devicePixelRatio()) if sc is not None else 1.0
        if _native():
            got = native_area(int(self.winId()))
            if got:
                (cx, cy, cw, ch), (mx, my, mw, mh), device = got
                x0, y0 = max(cx, mx), max(cy, my)
                x1, y1 = min(cx + cw, mx + mw), min(cy + ch, my + mh)
                return dpr, (x0 - cx, y0 - cy), (max(0, x1 - x0), max(0, y1 - y0)), device
        return dpr, (0, 0), (int(round(self.width() * dpr)), int(round(self.height() * dpr))), \
            (sc.name() if sc is not None else "")

    def relayout(self):
        """The board for the area the window shows now, at the pitch of its monitor; `relaid` when the board itself
        changed (squares, pixels, pitch) or none fits any more."""
        dpr, offset, area, device = self.measure()
        mon = self.monitor_of(device)
        before = self.spec()
        lay, error = None, ""
        if mon is None:
            error = "no_monitor"
        else:
            try:
                lay = screen_board.layout(area, mon["pitch_mm"] or screen_board_guess(), self.hold)
            except ValueError:
                error = "too_small"
        same = (lay is not None and self.lay is not None and lay["size_px"] == self.lay["size_px"]
                and lay["origin_px"] == self.lay["origin_px"] and lay["square_px"] == self.lay["square_px"]
                and lay["squares"] == self.lay["squares"] and lay["pitch_mm"] == self.lay["pitch_mm"])
        self.dpr, self.offset, self.area, self.mon = dpr, offset, area, mon
        self.key = mon["key"] if mon else ""
        self.error = error
        if not same:
            self.lay = lay
            self.image = screen_board.render(lay) if lay else None
            self._img = to_qimage(self.image, dpr) if lay else None
        elif self._img is not None and self._img.devicePixelRatio() != dpr:
            self._img = to_qimage(self.image, dpr)
        self.update()
        if self.spec() != before:
            self.relaid.emit()

    def spec(self):
        """What the app keeps of the board on show: screen_board.spec with its monitor, or None."""
        return screen_board.spec(self.lay, self.mon) if self.lay else None

    # ── where things go, in logical pixels of the window ─────────────────
    def _mm(self, mm):
        return mm / ((self.lay or {}).get("pitch_mm") or screen_board_guess()) / self.dpr

    def _num(self, v, n):
        text = f"{v:.{n}f}"
        return text.replace(".", ",") if self.words.get("comma") else text

    def _px(self, x, y=None):
        """A point of the laid out area (physical pixels from its corner) in the window's logical pixels."""
        if y is None:
            return x / self.dpr
        return QPointF((self.offset[0] + x) / self.dpr, (self.offset[1] + y) / self.dpr)

    def _button(self):
        r = max(14.0, self._mm(3.5))
        m = max(10.0, self._mm(2.5))
        right = (self.offset[0] + self.area[0]) / self.dpr
        return QRectF(right - m - 2 * r, self.offset[1] / self.dpr + m, 2 * r, 2 * r)

    def show_board(self):
        if self.full:
            self.showFullScreen()
            self.raise_()
            self.activateWindow()
            self._idle.start()
        else:
            self.show()
            self.raise_()
        self.relayout()

    # ── drawing ──────────────────────────────────────────────────────────
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        p.setRenderHint(_AA, True)
        p.setRenderHint(_TEXT_AA, True)
        f = QFont("Segoe UI")
        lay = self.lay
        if lay is not None and self._img is not None:
            p.drawImage(self._px(0, 0), self._img)        # one image pixel on one screen pixel: never scaled
            x0, _ = lay["origin_px"]
            x1 = x0 + lay["squares"][0] * lay["square_px"]
            # the length between the arrows, on a white patch in the middle of the dimension line
            f.setPixelSize(max(12, round(self._mm(3.4))))
            p.setFont(f)
            text = self.words.get("measure", "{0} mm").replace("{0}", self._num(lay["width_mm"], 1))
            fm = p.fontMetrics()
            tw, th = fm.horizontalAdvance(text) + 2 * fm.averageCharWidth(), fm.height() * 1.15
            c = self._px((x0 + x1) / 2, lay["line_y"] - 0.5)
            box = QRectF(c.x() - tw / 2, c.y() - th / 2, tw, th)
            p.fillRect(box, QColor(255, 255, 255))
            p.setPen(INK)
            p.drawText(box, _CENTER, text)
            # what the board is, in the corner band
            f.setPixelSize(max(11, round(self._mm(2.6))))
            p.setFont(f)
            p.setPen(GREY)
            b = self._button()
            left = self._px(lay["margin_px"], 0).x()
            nx, ny = lay["squares"]
            info = self.words.get("info", "").replace("{0}", str(nx)).replace("{1}", str(ny)).replace(
                "{2}", self._num(lay["square_mm"], 2))
            p.drawText(QRectF(left, b.top(), max(0.0, b.left() - left - b.width() / 2), b.height()), _LEFT, info)
        else:
            # no board fits: say so in the middle, and how to get one
            f.setPixelSize(16)
            p.setFont(f)
            p.setPen(GREY)
            p.drawText(QRectF(self.rect()), _CENTER, self.words.get("small", ""))
        # the close button: a grey disc with a cross
        b = self._button()
        p.setPen(_NO_PEN)
        p.setBrush(BUTTON_HOT if self._hot else BUTTON)
        p.drawEllipse(b)
        pen = p.pen()
        pen.setColor(INK)
        pen.setWidthF(max(1.5, b.width() / 14))
        pen.setStyle(_SOLID)
        p.setPen(pen)
        k = b.width() * 0.22
        c = b.center()
        p.drawLine(QPointF(c.x() - k, c.y() - k), QPointF(c.x() + k, c.y() + k))
        p.drawLine(QPointF(c.x() - k, c.y() + k), QPointF(c.x() + k, c.y() - k))
        p.end()

    # ── place and size: laid out again once they settle ──────────────────
    def moveEvent(self, e):
        super().moveEvent(e)
        self.moved.emit()
        self._relay.start()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.moved.emit()
        self._relay.start()

    def nativeEvent(self, eventType, message):
        """In a window: the frame Windows keeps under it is not drawn (win32.answer), the edges resize it and the rest
        moves it (WM_NCHITTEST), so Windows snaps it like any window; the close button is the page's own. Full
        screen: nothing. It is not called: in PyQt6 calling super().nativeEvent() from an override crashes."""
        if self.full or not _native():
            return False, 0
        answer = win32.answer(int(message))
        if answer is not None:
            return answer
        try:
            import ctypes
            from ctypes import wintypes
            msg = wintypes.MSG.from_address(int(message))
            if msg.message != WM_NCHITTEST:
                return False, 0
            x = ctypes.c_short(msg.lParam & 0xFFFF).value
            y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
            left, top, right, bottom = _window_rect(int(self.winId()))
            edge = max(4, round(EDGE_PX * self.dpr))
            if not self.isMaximized():
                h = "left" if x < left + edge else "right" if x >= right - edge else ""
                v = "top" if y < top + edge else "bottom" if y >= bottom - edge else ""
                if h or v:
                    return True, HT[v + h]
            if self._button().contains(QPointF((x - left) / self.dpr, (y - top) / self.dpr)):
                return True, HT["client"]
            return True, HT["caption"]
        except Exception:      # noqa: BLE001  a hit test never takes the window down
            return False, 0

    # ── input: Esc and the button close; nothing zooms the board ─────────
    def keyPressEvent(self, e):
        if e.key() == _ESC:
            self.close()
        e.accept()            # the zoom keys (Ctrl and +, -, 0) and every other key: nothing

    def wheelEvent(self, e):
        e.accept()

    def event(self, e):
        if e.type() in _SWALLOW:
            e.accept()
            return True
        return super().event(e)

    def _pos(self, e):
        return e.position() if USE_QT6 else QPointF(e.pos())

    def mouseMoveEvent(self, e):
        self.setCursor(QCursor(_ARROW))
        self._idle.start()
        hot = self._button().contains(self._pos(e))
        if hot != self._hot:
            self._hot = hot
            self.update(self._button().toAlignedRect().adjusted(-2, -2, 2, 2))

    def leaveEvent(self, e):
        if self._hot:
            self._hot = False
            self.update(self._button().toAlignedRect().adjusted(-2, -2, 2, 2))
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if self._button().contains(self._pos(e)):
            self.close()
        e.accept()

    def mouseDoubleClickEvent(self, e):
        e.accept()

    def closeEvent(self, e):
        self._relay.stop()
        self._watch(None)
        self.closed.emit(self._why)
        super().closeEvent(e)


def screen_board_guess():
    """A monitor that says nothing of its size is laid out as a 96 dpi one, to be measured with a ruler."""
    return 25.4 / 96


# ── docked beside the app ────────────────────────────────────────────────────
class Dock(QObject):
    """The board window beside the app window on the app's monitor, the two sharing its work area: the app on the
    left, the board on the right, at a split (the board's share of the width). Dragging the edge between them, from
    either window, moves the split and the other window follows, as with two windows Windows has snapped together;
    the app resized from below or above takes the board along. Moving either window, maximising the app or making
    the board too narrow undocks it: the board then stays where it is, a window of its own. When the board closes,
    the app goes back to where it was. `split` reports a split the operator set; `undocked` that it is free."""

    split_set = pyqtSignal(float)
    undocked = pyqtSignal()

    def __init__(self, app_win, board, split=0.5, min_board=(320, 300)):
        super().__init__(board)
        self.app, self.board = app_win, board
        self.split = min(0.8, max(0.2, float(split)))
        self.min_board = min_board
        self.docked = False
        self.saved = None               # (geometry, maximised) of the app before it made room
        self._want_app = self._want_board = None
        self._follow = QTimer(self)
        self._follow.setSingleShot(True)
        self._follow.setInterval(0)
        self._follow.timeout.connect(self._settle)
        self._from = set()

    def work(self):
        sc = self.app.screen()
        return sc.availableGeometry() if sc is not None else QRect(0, 0, 1920, 1040)

    def _app_min(self):
        return max(320, int(self.app.minimumWidth()))

    def rects(self, split=None):
        """(app, board) rectangles for a split of the work area, within the two windows' least widths."""
        w = self.work()
        bw = round(w.width() * (self.split if split is None else split))
        bw = max(self.min_board[0], min(bw, w.width() - self._app_min()))
        aw = w.width() - bw
        return QRect(w.x(), w.y(), aw, w.height()), QRect(w.x() + aw, w.y(), bw, w.height())

    def fits(self):
        w = self.work()
        return w.width() - self._app_min() >= self.min_board[0] and w.height() >= self.min_board[1]

    def dock(self, then=None):
        """Makes room: the app on the left share, the board on the right one. A maximised app is restored first."""
        if self.saved is None:
            self.saved = (QRect(self.app.geometry()), bool(self.app.isMaximized()))

        def place():
            a, b = self.rects()
            self._set(a, b)
            if not self.docked:
                self.docked = True
                for sig in (self.app.xChanged, self.app.yChanged, self.app.widthChanged, self.app.heightChanged):
                    sig.connect(self._app_moved)
                self.board.moved.connect(self._board_moved)
            if then:
                then()
        if self.app.isMaximized():
            self.app.showNormal()
            QTimer.singleShot(320, place)
        else:
            place()

    def _set(self, a=None, b=None):
        if a is not None:
            self._want_app = QRect(a)
            self.app.setGeometry(a)
        if b is not None:
            self._want_board = QRect(b)
            self.board.setGeometry(b)

    def _app_moved(self, *_):
        self._from.add("app")
        self._follow.start()

    def _board_moved(self):
        self._from.add("board")
        self._follow.start()

    def _settle(self):
        """What changed, once Qt has delivered every part of it."""
        came, self._from = self._from, set()
        if not self.docked:
            return
        a, b = QRect(self.app.geometry()), QRect(self.board.geometry())
        w = self.work()
        if "board" in came and b != self._want_board:
            want = self._want_board
            if want is not None and b.size() == want.size():
                self.undock()                                     # moved by the operator: free
                return
            if want is not None and b.right() == want.right() and b.top() == want.top() \
                    and b.height() == want.height():
                aw = max(self._app_min(), b.left() - a.left())    # the edge between them, from the board's side
                bw = w.right() + 1 - (a.left() + aw)
                if bw < self.min_board[0]:
                    self.undock()
                    return
                self._set(QRect(a.left(), a.top(), aw, a.height()), QRect(a.left() + aw, a.top(), bw, a.height()))
                self._report()
                return
            self.undock()
            return
        if "app" in came and a != self._want_app:
            want = self._want_app
            if self.app.isMaximized() or (self.app.windowState() & _MINIMIZED) or (want is not None
                                                                                 and a.left() != want.left()):
                self.undock()
                return
            bw = w.right() - a.right()
            if bw < self.min_board[0]:
                self.undock()
                return
            self._want_app = QRect(a)
            self._set(None, QRect(a.right() + 1, a.top(), bw, a.height()))
            self._report()

    def _report(self):
        w = self.work()
        if w.width() > 0 and self._want_board is not None:
            self.split = self._want_board.width() / w.width()
            self.split_set.emit(self.split)

    def undock(self):
        if not self.docked:
            return
        self.docked = False
        for sig in (self.app.xChanged, self.app.yChanged, self.app.widthChanged, self.app.heightChanged):
            try:
                sig.disconnect(self._app_moved)
            except (TypeError, RuntimeError):
                pass
        try:
            self.board.moved.disconnect(self._board_moved)
        except (TypeError, RuntimeError):
            pass
        self.undocked.emit()

    def give_back(self):
        """The board is closing: undocked, and the app where it was before it made room, unless the operator has
        moved it since."""
        was = self.docked
        self.undock()
        if self.saved is None:
            return
        geo, maxed = self.saved
        self.saved = None
        if not was and (self._want_app is None or QRect(self.app.geometry()) != self._want_app):
            return              # undocked and the app moved since: it stays where the operator put it
        if maxed:
            self.app.showMaximized()
        else:
            self.app.setGeometry(geo)


# ── check on this computer ───────────────────────────────────────────────────
def grab_screen(screen):
    """The screen as it is, in physical pixels, grey (what a camera filming it would see, minus the camera)."""
    pm = screen.grabWindow(0)
    q = pm.toImage().convertToFormat(_GRAY8)
    w, h = q.width(), q.height()
    ptr = q.constBits()
    ptr.setsize(q.bytesPerLine() * h)
    return np.frombuffer(ptr, np.uint8).reshape(h, q.bytesPerLine())[:, :w].copy()


def board_in_grab(img, win):
    """The board as the screen grab of its monitor shows it, compared with the image the window drew: (identical,
    the window's own area cut out of the grab, nothing of the desktop around it). The window's place on its monitor
    comes from Windows (physical pixels)."""
    lay = win.lay
    if lay is None:
        return False, None
    ox, oy = win.offset
    if _native():
        got = native_area(int(win.winId()))
        if got:
            (cx, cy, _, _), (mx, my, _, _), _ = got
            ox, oy = ox + cx - mx, oy + cy - my
    w, h = win.area
    own = img[oy:oy + h, ox:ox + w]
    x0, y0 = lay["origin_px"]
    nx, ny = lay["squares"]
    s = lay["square_px"]
    box = (slice(y0 - 2, y0 + ny * s + 2), slice(x0 - 2, x0 + nx * s + 2))
    ok = own.shape == win.image.shape and bool(np.array_equal(own[box], win.image[box]))
    return ok, own.copy()


def check(scale=None, out=None, hold_ms=1200, window=False):
    """The board on the main screen, grabbed back and compared with the image (see the module's text)."""
    if scale:
        os.environ["QT_SCALE_FACTOR"] = str(scale)
    from slantui.shell.qt import QApplication
    import calibrate_charuco
    app = QApplication.instance() or QApplication(sys.argv[:1])
    screen = app.primaryScreen()
    mon = screen_board.monitors([screen_board.qt_screen(screen, True)], screen_board.windows_monitors())[0]
    words = {"measure": "{0} mm", "info": "check", "title": "ImpactGuide"}
    rect = None
    if window:
        g = screen.availableGeometry()
        rect = QRect(g.x() + g.width() * 2 // 5, g.y(), g.width() - g.width() * 2 // 5, g.height())
    win = BoardWindow(screen, lambda device: mon, words, full=not window, rect=rect)
    result = {}

    def grab():
        img = grab_screen(screen)
        lay = win.lay
        result.update(scale=scale or 1.0, dpr=win.dpr, window=window, screen_px=[img.shape[1], img.shape[0]],
                      area_px=list(win.area), squares=lay and lay["squares"])
        same, own = board_in_grab(img, win)
        result["board_identical"] = same
        if out and same:      # only the board's own area is kept, never the desktop around it
            Path(out).mkdir(parents=True, exist_ok=True)
            name = f"board_window_grab_{'w_' if window else ''}{str(scale or 1).replace('.', '_')}.png"
            import cv2
            cv2.imwrite(str(Path(out) / name), own)
            result["grab"] = str(Path(out) / name)
            views, _, _ = calibrate_charuco.detect([Path(out) / name], screen_board.make_board(lay))
            result["corners_found"] = len(views[name][0]) if views else 0
            result["corners"] = (lay["squares"][0] - 1) * (lay["squares"][1] - 1)
        win.close()
        app.quit()

    win.show_board()
    QTimer.singleShot(hold_ms, grab)
    QTimer.singleShot(hold_ms + 8000, app.quit)
    app.exec() if USE_QT6 else app.exec_()
    print(json.dumps(result))
    return 0 if result.get("board_identical") else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["check"])
    ap.add_argument("--scale", type=float, help="Qt's own scaling of the window (QT_SCALE_FACTOR), e.g. 1.25")
    ap.add_argument("--out", type=Path, help="keep the grab here and find the board in it")
    ap.add_argument("--window", action="store_true", help="in a window of 60 % of the screen instead of full screen")
    a = ap.parse_args()
    sys.exit(check(a.scale, a.out, window=a.window))
