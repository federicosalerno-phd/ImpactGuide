"""The files the desktop launcher needs and cannot make for itself.

icon.ico is what Windows shows on the shortcut and on the taskbar, drawn from
ui/impactguide-icon.svg, the same icon the window
carries in its title band. theme.json holds the few SlantUI roles the
launcher's waiting window paints itself with. first.json is what the start's
first window is drawn with (impactguide_boot.py), with the app's own mark and
icon, until the app has run once and kept its own (first_window.py, with the
band). All are
written at install time, by install_shortcut.ps1, because the launcher has to
work before the drive holding the project is mounted.

    python scripts/launcher_assets.py <out-dir>
"""
import io
import json
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import impactguide_app as app  # noqa: E402

from PyQt6.QtCore import QBuffer, QSize  # noqa: E402
from PyQt6.QtGui import QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402  (WebEngine wants the widgets application)

DIB_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128)
PNG_SIZE = 256
THEME_ROLES = {"surface": "surface-1", "raised": "surface-2", "line": "surface-3", "control": "control",
               "controlHover": "control-hover", "text": "text-1", "muted": "text-3", "accent": "accent",
               "error": "err-text"}


def _argb(icon, size):
    pm = icon.pixmap(QSize(size, size))
    if pm.width() != size or pm.height() != size:
        pm = pm.scaled(size, size)
    return pm.toImage().convertToFormat(QImage.Format.Format_ARGB32)


def _dib(img):
    """A BITMAPINFOHEADER image: header, pixels bottom up, an all-zero AND mask."""
    w, h = img.width(), img.height()
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, 0, 0, 0, 0, 0)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    raw, stride = bytes(ptr), img.bytesPerLine()
    rows = [raw[y * stride: y * stride + w * 4] for y in range(h - 1, -1, -1)]
    return header + b"".join(rows) + b"\x00" * (((w + 31) // 32) * 4 * h)


def _png(img):
    buf = QBuffer()
    buf.open(QBuffer.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def write_ico(icon, path):
    images = [(s, _dib(_argb(icon, s))) for s in DIB_SIZES] + [(PNG_SIZE, _png(_argb(icon, PNG_SIZE)))]
    out = io.BytesIO()
    out.write(struct.pack("<HHH", 0, 1, len(images)))
    offset = 6 + 16 * len(images)
    for size, blob in images:
        byte = 0 if size >= 256 else size
        out.write(struct.pack("<BBBBHHII", byte, byte, 0, 0, 1, 32, len(blob), offset))
        offset += len(blob)
    for _, blob in images:
        out.write(blob)
    Path(path).write_bytes(out.getvalue())
    return len(images)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    out.mkdir(parents=True, exist_ok=True)
    qt = QApplication(sys.argv[:1])  # kept in a name: collected at once, it takes the process down
    count = write_ico(app.app_icon(), out / "icon.ico")
    theme = {k: app.role(v) for k, v in THEME_ROLES.items()}
    (out / "theme.json").write_text(json.dumps(theme, indent=2), encoding="utf-8")
    try:
        lang = json.loads((app.LOCAL / "preferences.json").read_text(encoding="utf-8")).get("lang")
    except (OSError, ValueError, AttributeError):
        lang = None
    lang = lang if lang in app.LANGS else "en"      # as the app chooses it (Backend: English unless chosen)
    app.first_window.keep(out, palette=app.PALETTE, text=app.splash_words(lang)[0], size=app.WINDOW_SIZE,
                          mark=app.MARK, wait=True)
    print(f"icon.ico: {count} sizes")
    print(f"theme.json: palette {app.PALETTE}")
    print(f"first.json: the first window in {lang}")
    del qt
    return 0


if __name__ == "__main__":
    sys.exit(main())
