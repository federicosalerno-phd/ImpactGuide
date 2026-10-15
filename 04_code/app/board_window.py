"""ImpactGuide: the calibration board full screen on a monitor, one image pixel on one screen pixel.

screen_board.py draws the board for the monitor's physical pixels; this window covers that monitor and puts the image
on it without resampling, at any Windows display scaling: the image carries the screen's device pixel ratio, so Qt
lays its pixels on the screen's one to one. It stays on top, so no other window covers part of the board. Nothing
zooms it: the wheel, pinch gestures and the zoom keys are taken and dropped. Esc or the small round button in the
corner closes it; the pointer hides when it stops, so it never covers a marker. The few words on it (the length between
the arrows, what the board is, how to close it) are drawn in the band above the board, never over the board or the
ends of the lines.

    python board_window.py check [--scale 1.25] [--out DIR]

shows the board on the main monitor for a moment, grabs the screen and compares it with the image pixel for pixel (and
finds the board in the grab with the calibration's detection); with --scale, Qt scales the window itself as Windows
does at that display scaling (QT_SCALE_FACTOR), which tries the 125 % and 150 % path on a 100 % monitor. Prints a JSON
line; exit code 1 when the screen does not hold the image exactly.
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

from slantui.shell.qt import (QColor, QCursor, QEvent, QFont, QImage, QPainter, QPointF, QRectF,  # noqa: E402
                              Qt, QTimer, QWidget, USE_QT6, pyqtSignal)

import screen_board  # noqa: E402

HIDE_POINTER_MS = 1500
INK = QColor(0, 0, 0)
GREY = QColor(110, 110, 110)
BUTTON = QColor(232, 232, 232)
BUTTON_HOT = QColor(205, 205, 205)

if USE_QT6:
    _FRAMELESS = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint
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
else:  # pragma: no cover - PyQt5
    _FRAMELESS = Qt.FramelessWindowHint | Qt.Window | Qt.WindowStaysOnTopHint
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


def to_qimage(img, dpr):
    """A grey image as a QImage that Qt draws one pixel per device pixel on a screen of this ratio."""
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape
    q = QImage(img.data, w, h, w, _GRAY8).copy()
    q.setDevicePixelRatio(dpr)
    return q


class BoardWindow(QWidget):
    """The board on one screen. `closed` says why it went: "user" (Esc, the button) or "changed" (the screen's size,
    scaling or place changed under it, so the image no longer fits its pixels)."""

    closed = pyqtSignal(str)

    def __init__(self, screen, lay, words):
        super().__init__(None, _FRAMELESS)
        self.lay, self.words = lay, words
        self.screen_ = screen
        self.key = ""                   # the monitor's key (screen_board), set by the app
        self._why = "user"
        self._hot = False
        self.dpr = float(screen.devicePixelRatio())
        self.image = screen_board.render(lay)
        self._img = to_qimage(self.image, self.dpr)
        self.setWindowTitle(words.get("title", "ImpactGuide"))
        self.setAttribute(_OPAQUE, True)
        self.setAttribute(_NO_BG, True)
        self.setAttribute(_TOUCH, True)
        self.setMouseTracking(True)
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(HIDE_POINTER_MS)
        self._idle.timeout.connect(lambda: self.setCursor(QCursor(_BLANK)))
        if USE_QT6:
            self.setScreen(screen)
        self.setGeometry(screen.geometry())
        for sig in (screen.geometryChanged, screen.logicalDotsPerInchChanged, screen.physicalDotsPerInchChanged):
            sig.connect(self.changed)

    # ── where things go, in logical pixels of the window ─────────────────
    def _mm(self, mm):
        return mm / self.lay["pitch_mm"] / self.dpr

    def _px(self, px):
        return px / self.dpr

    def _button(self):
        r = max(14.0, self._mm(3.5))
        m = max(10.0, self._mm(2.5))
        return QRectF(self.width() - m - 2 * r, m, 2 * r, 2 * r)

    def show_board(self):
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self._idle.start()

    def changed(self, *_):
        """The screen changed under the board (size, scaling, place, or it is gone): the image no longer fits its
        pixels, so the window goes and says why."""
        self._why = "changed"
        self.close()

    # ── drawing ──────────────────────────────────────────────────────────
    def paintEvent(self, e):
        p = QPainter(self)
        p.drawImage(QPointF(0, 0), self._img)        # one image pixel on one screen pixel: never scaled
        p.setRenderHint(_AA, True)
        p.setRenderHint(_TEXT_AA, True)
        lay = self.lay
        x0, _ = lay["origin_px"]
        x1 = x0 + lay["squares"][0] * lay["square_px"]
        # the length between the arrows, on a white patch in the middle of the dimension line
        f = QFont("Segoe UI")
        f.setPixelSize(max(12, round(self._mm(3.4))))
        p.setFont(f)
        text = self.words.get("measure", "")
        fm = p.fontMetrics()
        tw, th = fm.horizontalAdvance(text) + 2 * fm.averageCharWidth(), fm.height() * 1.15
        cx, cy = self._px((x0 + x1) / 2), self._px(lay["line_y"] - 0.5)
        box = QRectF(cx - tw / 2, cy - th / 2, tw, th)
        p.fillRect(box, QColor(255, 255, 255))
        p.setPen(INK)
        p.drawText(box, _CENTER, text)
        # what the board is, and how to close it, in the corner band
        f.setPixelSize(max(11, round(self._mm(2.6))))
        p.setFont(f)
        p.setPen(GREY)
        b = self._button()
        info = QRectF(self._px(lay["margin_px"]), b.top(), b.left() - self._px(lay["margin_px"]) - b.width(),
                      b.height())
        p.drawText(info, _LEFT, self.words.get("info", ""))
        # the close button: a grey disc with a cross
        p.setPen(_NO_PEN)
        p.setBrush(BUTTON_HOT if self._hot else BUTTON)
        p.drawEllipse(b)
        pen = p.pen()
        pen.setColor(INK)
        pen.setWidthF(max(1.5, b.width() / 14))
        pen.setStyle(Qt.PenStyle.SolidLine if USE_QT6 else Qt.SolidLine)
        p.setPen(pen)
        k = b.width() * 0.22
        c = b.center()
        p.drawLine(QPointF(c.x() - k, c.y() - k), QPointF(c.x() + k, c.y() + k))
        p.drawLine(QPointF(c.x() - k, c.y() + k), QPointF(c.x() + k, c.y() - k))
        p.end()

    # ── input: Esc and the button close; nothing zooms or moves the board ─
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

    def mousePressEvent(self, e):
        if self._button().contains(self._pos(e)):
            self.close()
        e.accept()

    def mouseDoubleClickEvent(self, e):
        e.accept()

    def closeEvent(self, e):
        for sig in (self.screen_.geometryChanged, self.screen_.logicalDotsPerInchChanged,
                    self.screen_.physicalDotsPerInchChanged):
            try:
                sig.disconnect(self.changed)
            except (TypeError, RuntimeError):
                pass
        self.closed.emit(self._why)
        super().closeEvent(e)


def grab_screen(screen):
    """The screen as it is, in physical pixels, grey (what a camera filming it would see, minus the camera)."""
    pm = screen.grabWindow(0)
    q = pm.toImage().convertToFormat(_GRAY8)
    w, h = q.width(), q.height()
    ptr = q.constBits()
    ptr.setsize(q.bytesPerLine() * h)
    return np.frombuffer(ptr, np.uint8).reshape(h, q.bytesPerLine())[:, :w].copy()


def check(scale=None, out=None, hold_ms=1200):
    """The board on the main screen, grabbed back and compared with the image (see the module's text)."""
    if scale:
        os.environ["QT_SCALE_FACTOR"] = str(scale)
    from slantui.shell.qt import QApplication
    import calibrate_charuco
    app = QApplication.instance() or QApplication(sys.argv[:1])
    screen = app.primaryScreen()
    mon = screen_board.monitors([screen_board.qt_screen(screen, True)], screen_board.windows_monitors())[0]
    lay = screen_board.layout(mon["px"], mon["pitch_mm"] or 0.25)
    words = {"measure": f"{lay['width_mm']:.1f} mm", "info": "check", "title": "ImpactGuide"}
    win = BoardWindow(screen, lay, words)
    result = {}

    def grab():
        img = grab_screen(screen)
        ref = win.image
        x0, y0 = lay["origin_px"]
        nx, ny = lay["squares"]
        s = lay["square_px"]
        h, w = ref.shape
        result.update(scale=scale or 1.0, dpr=win.dpr, screen_px=[img.shape[1], img.shape[0]], image_px=[w, h])
        if img.shape[:2] == ref.shape:
            # the board and the lines' ends only: the words are drawn over the band above the board
            region = (slice(y0 - 2, y0 + ny * s + 2), slice(x0 - 2, x0 + nx * s + 2))
            diff = np.abs(img[region].astype(int) - ref[region].astype(int))
            result.update(board_identical=bool(diff.max() == 0), differing_px=int((diff > 0).sum()))
        if out and result.get("board_identical"):     # a grab of anything else is not kept: it is someone's desktop
            Path(out).mkdir(parents=True, exist_ok=True)
            name = f"board_window_grab_{str(scale or 1).replace('.', '_')}.png"
            import cv2
            cv2.imwrite(str(Path(out) / name), img)
            result["grab"] = str(Path(out) / name)
            path = Path(out) / name
            views, _, _ = calibrate_charuco.detect([path], screen_board.make_board(lay))
            result["corners_found"] = len(views[name][0]) if views else 0
            result["corners"] = (nx - 1) * (ny - 1)
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
    a = ap.parse_args()
    sys.exit(check(a.scale, a.out))
