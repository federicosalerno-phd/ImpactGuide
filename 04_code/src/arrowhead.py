"""The one arrowhead of every drawing: a filled, slim isosceles triangle with its point on what it shows.

Its shape is the arrow of the dimension line over the board on a monitor (screen_board.py): LENGTH_MM long and
HALF_MM half wide, so half its width is HALF of its length (38.2 degrees at the point). Every drawing scales that one
triangle and never changes its proportions; the page draws the same one (SlantUI's arrowHead, icons.js).

    points(tip, direction, length)              the three corners: the point, then the two ends of the base
    fill(img, tip, direction, length, colour)   the triangle painted into an image, coordinates in pixels with the
                                                pixel centres at whole numbers (as OpenCV): each pixel takes the share
                                                of it the triangle covers (16 x 16 samples), so the edges are straight
                                                and the point sharp at any size, never a staircase
    inset(length, line_width)                   how far from the point a line ending in the arrow stops: inside the
                                                triangle, where it is wider than the line, so line and arrow make one
                                                shape, with no gap and with the triangle's own point
    patch(tip, direction, length, **style)      the same triangle as a matplotlib polygon (the printable PDF)

    python arrowhead.py test                    the checks; exit code 1 on failure
"""
from __future__ import annotations

import math
import sys

import numpy as np

LENGTH_MM, HALF_MM = 2.6, 0.9           # the dimension arrow over the screen board
HALF = HALF_MM / LENGTH_MM              # half the width over the length
SAMPLES = 16                            # per pixel side in fill(): 256 levels of coverage


def points(tip, direction, length):
    """The corners (3 x 2): the point at tip, the base `length` behind it across `direction` (any length but 0)."""
    t = np.asarray(tip, np.float64)
    u = np.asarray(direction, np.float64)
    n = float(np.hypot(*u))
    if not n > 0:
        raise ValueError("an arrowhead needs a direction")
    u = u / n
    base = t - u * float(length)
    side = np.array([-u[1], u[0]]) * float(length) * HALF
    return np.array([t, base + side, base - side])


def fill(img, tip, direction, length, colour):
    """Paint the arrowhead into img (H x W or H x W x C, uint8) in colour, over what is there by the share of each
    pixel it covers. Returns img."""
    tri = points(tip, direction, length)
    h, w = img.shape[:2]
    x0, y0 = (max(0, int(math.floor(v + 0.5))) for v in tri.min(axis=0))
    x1, y1 = (int(math.floor(v + 0.5)) + 1 for v in tri.max(axis=0))
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return img
    k = (np.arange(SAMPLES) + 0.5) / SAMPLES - 0.5
    xs = (np.arange(x0, x1)[:, None] + k[None, :]).ravel()          # sample columns, pixel by pixel
    ys = (np.arange(y0, y1)[:, None] + k[None, :]).ravel()
    X, Y = np.meshgrid(xs, ys)
    a, b, c = tri
    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    s = 1.0 if area > 0 else -1.0
    inside = np.ones(X.shape, bool)
    for p, q in ((a, b), (b, c), (c, a)):
        inside &= s * ((q[0] - p[0]) * (Y - p[1]) - (q[1] - p[1]) * (X - p[0])) >= 0
    cov = inside.reshape(y1 - y0, SAMPLES, x1 - x0, SAMPLES).mean(axis=(1, 3))
    region = img[y0:y1, x0:x1].astype(np.float64)
    col = np.asarray(colour, np.float64)
    if region.ndim == 3:
        cov = cov[..., None]
    img[y0:y1, x0:x1] = np.clip(np.rint(region * (1 - cov) + col * cov), 0, 255).astype(img.dtype)
    return img


def inset(length, line_width):
    """Where a line of line_width ending in an arrowhead of length stops, as a distance back from the point: past
    half the arrow and where the triangle is twice as wide as the line, never past the base."""
    return min(float(length), max(0.5 * float(length), float(line_width) / HALF))


def patch(tip, direction, length, **style):
    """The arrowhead as a matplotlib Polygon in data units, filled, without an outline unless the style asks."""
    from matplotlib.patches import Polygon
    style.setdefault("linewidth", 0)
    style.setdefault("closed", True)
    return Polygon(points(tip, direction, length), **style)


def _test():
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    p = points((10.0, 5.0), (-3.0, 0.0), 13.0)
    expect(np.allclose(p, [[10, 5], [23, 5 - 13 * HALF], [23, 5 + 13 * HALF]]),
           "points: the point at the tip, the base one length behind it, half a width of HALF of the length")
    ang = 2 * math.degrees(math.atan(HALF))
    expect(abs(ang - 38.2) < 0.1, f"the angle at the point is {ang:.1f} degrees")
    img = np.full((24, 40), 255, np.uint8)
    fill(img, (5.5, 11.5), (-1, 0), 12.0, 0)
    mirror = np.full((24, 40), 255, np.uint8)
    fill(mirror, (33.5, 11.5), (1, 0), 12.0, 0)
    vals = np.unique(img)
    expect(len(vals) > 10 and vals.min() == 0 and vals.max() == 255,
           f"fill: grey where a pixel is partly covered ({len(vals)} levels), black inside, white outside")
    expect(np.array_equal(img, img[::-1]) and np.array_equal(img, mirror[:, ::-1]),
           "fill: the same arrow up and down, and pointing the other way the mirror image")
    cov = (255 - img.astype(float)).sum() / 255
    expect(abs(cov - 12.0 * 12.0 * HALF) < 0.05, f"fill: the ink is the triangle's area ({cov:.3f} px2)")
    rgb = np.full((24, 40, 3), (40, 50, 60), np.uint8)
    fill(rgb, (20.0, 12.0), (0.6, -0.8), 9.0, (200, 100, 0))
    expect(tuple(rgb[12 + 3, 20 - 2].tolist()) == (200, 100, 0) and tuple(rgb[0, 0].tolist()) == (40, 50, 60),
           "fill: in colour, at any angle, over what is there")
    edge = np.full((8, 8), 255, np.uint8)
    fill(edge, (0.0, 3.5), (-1, 0), 30.0, 0)
    expect(edge.shape == (8, 8), "fill: an arrow out of the image is cut at its edge")
    d = inset(11.3, 2)
    expect(d * HALF >= 1.0 + 0.9 and d <= 11.3, f"inset: a 2 px line stops {d:.2f} px back, inside the triangle")
    expect(inset(2.6, 0.21) == 1.3, "inset: a thin line stops half the arrow back")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(_test())
    print(__doc__)
