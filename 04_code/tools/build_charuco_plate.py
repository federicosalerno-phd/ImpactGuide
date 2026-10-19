"""The 3D-printed ChArUco calibration plate of the phone camera (T5.1): the physical board of ImpactGuide's calibration.

The board is the A4 one of calibrate_charuco.py scaled to the printer's bed: 11 x 8 squares of 20 mm, markers of
15 mm (3/4 of the square), DICT_5X5_100, the same marker ids in the same squares. Square = (length read between the
ticks) / 11, as for the A4 sheet, so the code that finds the A4 board finds this one.

Plate (mm, plate coordinates: X right, Y up, Z up from the bed face, origin at the plate's lower left corner):
    white body   PLATE_W x PLATE_H x BASE, corners rounded in plan (vertical edges only: the face and the bed face keep
                 sharp edges), solid, flat on the bed. Printed with PRINT it is a sandwich: a bottom skin of 3 solid
                 layers, a top skin of 5 (under the black: 3 white and the 2 black), a 10 % grid between them and 2
                 walls. The skins hold most of the plastic (about 10 g a layer; a millimetre of grid about 7 g): the
                 plastic is saved in PRINT, not in BASE. BASE buys flatness: a bow after the plate leaves the bed goes
                 as 1 / (distance between the skins), the sag under its own weight as its square, and the face must
                 stay flat within 0.1 mm along its diagonals (a board curved by 2 mm biases the focal length by about
                 1 %). 8.0 keeps the skins 7.4 apart: held flat by one edge the plate sags by at most 0.25 mm, nearly
                 all a tilt (the 70 corners leave their best plane by 0.01 mm RMS); leaned or two-handed, by about
                 0.01 mm. A thinner plate saves little (5.0: 20 g less) and bows more, as 1 / d (5.0: 1.7 times).
    black body   INK above the white face (two layers of 0.2 mm, from BASE to BASE + INK): the black squares, the
                 markers, the two dimension marks and the label. Two black cells that touch only at a corner (the
                 chessboard corners, and diagonal bits inside a marker) are parted there by CORNER_GAP (a diamond
                 centred on the corner cut away), so every black piece is a closed solid of its own and no edge is
                 shared by four faces. The cut is symmetric about the corner and below what the printer resolves:
                 the corner the detector finds does not move. (A diamond joining the cells instead, 1 mm wide so as
                 to print, moved the corners by up to 6 px in close tilted photos.)
    marks        above the board two ticks whose outer faces are flush with the board's left and right edges, joined by
                 a bar with the nominal length (220.0 mm); right of the board two ticks flush with its top and bottom
                 edges (160.0 mm). A caliper's outside jaws read between the outer faces of each pair.
    label        below the board: the board's parameters.
Board coordinates (as OpenCV's CharucoBoard): x right, y down from the board's top left corner, z into the plate; in
plate coordinates the board's corner is at (MARGIN left, PLATE_H - MARGIN top, BASE + INK).

Outputs in 02_cad/stl (NAME_*): white and black STL (one body per colour), a 3MF with one object of two parts (white
on filament 1, black on filament 2) that carries PRINT as the object's own settings (Bambu Studio applies them over
any process preset), placed on the X2D's bed where both nozzles reach, and NAME.json with the board's nominal sizes,
what to measure and how to print, read by 04_code/src/charuco_plate.py.

Checks (build runs them; check runs them on the files):
    layout       the black cells equal OpenCV's own image of the board, cell for cell
    mesh         each body closed and two-manifold, outward and positive volume, no self intersections, the bodies
                 touch but do not overlap, the plate fits the bed
    walls        every black feature at least MIN_WIDTH wide (eroded by just under half of it, none vanishes or splits)
    detection    synthetic photos of the face rasterised from the black mesh, at 30 to 60 cm, frontal and tilted up
                 to 45 degrees, stream and photo mode: calibrate_charuco.detect finds all 70 corners, and a flat
                 board's pose fits them within MAX_FIT_PX
render: Blender (background) renders of the plate as printed, light and relief included: a preview and four views at
known poses; the corners found, their error against the true projection (on the black top: the relief shifts the
whole board by up to 0.2 mm, which the pose absorbs) and the fit of a flat board's pose.

Usage:
    python build_charuco_plate.py build     write STL, 3MF and JSON, then check
    python build_charuco_plate.py check     check the files in 02_cad/stl
    python build_charuco_plate.py render    Blender renders and detection on them, in C:\\work\\marker-impact\\charuco_plate
"""

import argparse
import json
import math
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from manifold3d import CrossSection, FillRule, JoinType, Manifold, OpType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "04_code" / "src"))
import calibrate_charuco  # noqa: E402

OUT = ROOT / "02_cad" / "stl"
WORK = Path(r"C:\work\marker-impact\charuco_plate")
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")

# the board: the A4 board of calibrate_charuco.py with 20 mm squares
SQUARES = tuple(calibrate_charuco.BOARD["squares"])                 # (11, 8)
SQUARE = 20.0
MARKER = SQUARE * calibrate_charuco.BOARD["marker"] / calibrate_charuco.BOARD["square"]    # 15.0
DICTIONARY = calibrate_charuco.BOARD["dictionary"]
NAME = f"charuco_plate_{SQUARES[0]}x{SQUARES[1]}_{SQUARE:g}mm"
CELLS = 56              # grid cells per square: the marker's 7 bits are 6 cells, its white margin 7 (2.5 mm)

# the plate, mm
BASE = 8.0              # white body; the black starts at this height (top of layer 40 at 0.2 mm)
INK = 0.4               # black above the white face: two layers of 0.2 mm
MARGIN = {"left": 6.0, "right": 14.0, "top": 14.0, "bottom": 22.0}
CORNER_R = 5.0          # plate corners in plan
CORNER_GAP = 0.1        # two black cells touching only at a corner are parted by this there
TICK_W, TICK_L = 1.6, 9.0   # a tick: width along the measured length, length across it
TICK_GAP = 2.0          # from the board's edge to the ticks
BAR = 1.2               # dimension bar between the ticks
CAP = 5.0               # cap height of the words (before BOLD)
BOLD = 0.25             # the font's outlines grown by this: its thinnest joins (m, q, r) become 1 mm
TRACK = 0.4             # extra space between letters, for BOLD
WORD_GAP = 1.6          # between the words and the bar
LABEL = (f"ChArUco {SQUARES[0]} x {SQUARES[1]}   {DICTIONARY}", f"square {SQUARE:g} mm   marker {MARKER:g} mm")
LABEL_GAP, LINE_PITCH = 4.0, 8.5    # board to the first line's cap, cap to cap
FONT = ("DejaVu Sans", "bold")
MIN_WIDTH = 1.0         # narrowest black feature
FLAT_TOL = 0.02         # Bezier flattening, mm

PLATE_W = MARGIN["left"] + SQUARES[0] * SQUARE + MARGIN["right"]     # 240
PLATE_H = MARGIN["top"] + SQUARES[1] * SQUARE + MARGIN["bottom"]     # 196

# Bambu Lab X2D: bed 256 x 256; both nozzles reach x 20.5 to 256. The 3MF places the plate turned by 90 degrees
# (long side front to back) inside that area, so it prints with one nozzle and a filament change or with both nozzles.
BED = (256.0, 256.0)
BOTH_NOZZLES_X = (20.5, 256.0)
BED_EDGE = 5.0
LAYER_H = 0.2
# Bambu Studio settings of the plate (0.20 mm layers, PLA), written into the 3MF as the object's own: the X2D's 0.20
# Standard but for the grid, 10 % instead of 15 (5 solid layers above it carry the face), and no brim (sharp bed edge)
PRINT = {"wall_loops": "2", "top_shell_layers": "5", "top_shell_thickness": "1", "bottom_shell_layers": "3",
         "bottom_shell_thickness": "0", "sparse_infill_density": "10%", "sparse_infill_pattern": "grid",
         "enable_support": "0", "brim_type": "no_brim"}

# detection check: the camera of the live stream (live_ar.py, nominal Poco X7 Pro) and the photo mode
STREAM = {"size": (1920, 1440), "f": 1439.0}
PHOTO = {"size": calibrate_charuco.SYNTH_SIZE, "K": calibrate_charuco.SYNTH_K, "dist": calibrate_charuco.SYNTH_DIST}
VIEWS = [  # (camera, distance mm, tilt about board x, tilt about board y, roll; degrees)
    ("stream", 300, 0, 0, 0), ("stream", 450, 0, 0, 0), ("stream", 600, 0, 0, 0),
    ("stream", 300, 45, 0, 0), ("stream", 450, 0, 45, 0), ("stream", 600, 40, 0, 0),
    ("stream", 600, 30, 30, 30), ("photo", 600, 45, 0, 20), ("photo", 300, 0, 40, 0),
]
TEX_PPM = 12            # px per mm of the face texture (a square is 240 px)
PHOTO_BLUR, PHOTO_NOISE = 1.0, 3.0      # px, grey levels
MAX_FIT_PX = 0.3        # residual of a flat board's pose fitted to the corners found (a calibration asks 0.5 in all)


# ── geometry ─────────────────────────────────────────────────────────────────
def to_plate(p):
    """Board coordinates (x right, y down from the board's top left corner) to plate coordinates."""
    p = np.asarray(p, float).reshape(-1, 2)
    return np.column_stack([MARGIN["left"] + p[:, 0], PLATE_H - MARGIN["top"] - p[:, 1]])


def ccw(p):
    p = np.asarray(p, float)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return p if area > 0 else p[::-1]


def rect(x0, y0, x1, y1):
    """A rectangle in board coordinates, as a counter-clockwise contour in plate coordinates."""
    return ccw(to_plate([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))


def make_board():
    return calibrate_charuco.make_board(SQUARE, SQUARES, MARKER)


def board_cells():
    """The board's black cells on a grid of CELLS per square: True = black."""
    board = make_board()
    nx, ny = SQUARES
    off, bit = (CELLS - 6 * 7) // 2, 6
    black = np.ones((ny * CELLS, nx * CELLS), bool)
    d = board.getDictionary()
    for mid, c in zip(board.getIds().ravel(), board.getObjPoints()):
        c = np.asarray(c).reshape(-1, 3)
        col, row = int(c[:, 0].min() // SQUARE), int(c[:, 1].min() // SQUARE)
        y0, x0 = row * CELLS, col * CELLS
        black[y0:y0 + CELLS, x0:x0 + CELLS] = False
        bits = cv2.aruco.generateImageMarker(d, int(mid), 7, borderBits=1) == 0
        black[y0 + off:y0 + off + 7 * bit, x0 + off:x0 + off + 7 * bit] = np.kron(bits, np.ones((bit, bit), bool))
    return black


def cell_mm(i):
    return i * SQUARE / CELLS


def board_contours(black):
    """Contours of the board's black (rectangles of merged cell runs) and of the diamonds cut away at every corner
    where two black cells touch only diagonally."""
    out, cuts, open_runs = [], [], {}
    rows = black.shape[0]
    for i in range(rows + 1):
        runs = set()
        if i < rows:
            r = np.flatnonzero(np.diff(np.r_[0, black[i].astype(np.int8), 0]))
            runs = set(zip(r[::2].tolist(), r[1::2].tolist()))
        for run in list(open_runs):
            if run not in runs:
                j0, j1 = run
                out.append(rect(cell_mm(j0), cell_mm(open_runs.pop(run)), cell_mm(j1), cell_mm(i)))
        for run in runs:
            open_runs.setdefault(run, i)
    a, b = black[:-1, :-1], black[:-1, 1:]
    c, d = black[1:, :-1], black[1:, 1:]
    pinch = (a & d & ~b & ~c) | (b & c & ~a & ~d)
    g = CORNER_GAP / math.sqrt(2)                      # half diagonal of the diamond: the corners end CORNER_GAP apart
    for i, j in zip(*np.nonzero(pinch)):
        x, y = cell_mm(j + 1), cell_mm(i + 1)
        cuts.append(ccw(to_plate([(x + g, y), (x, y + g), (x - g, y), (x, y - g)])))
    return out, cuts


def flatten(path):
    """Contours of a matplotlib path with its Bezier curves cut into chords within FLAT_TOL."""
    from matplotlib.path import Path as MPath
    polys, cur = [], []

    def curve(ctrl):
        p = np.vstack([cur[-1], ctrl])
        # a Bezier curve departs from its chords by at most 1/8 of its second differences times (1/n)^2 x k(k-1)
        k = len(p) - 1
        dd = np.linalg.norm(np.diff(p, 2, axis=0), axis=1).max()
        n = max(2, min(48, math.ceil(math.sqrt(k * (k - 1) * dd / (8 * FLAT_TOL)))))
        for t in np.linspace(0, 1, n + 1)[1:]:
            k = len(p) - 1
            w = np.array([math.comb(k, m) * t ** m * (1 - t) ** (k - m) for m in range(k + 1)])
            cur.append(w @ p)

    for verts, code in path.iter_segments(curves=True, simplify=False):
        v = np.asarray(verts, float).reshape(-1, 2)
        if code == MPath.MOVETO:
            if len(cur) > 2:
                polys.append(np.array(cur))
            cur = [v[-1]]
        elif code == MPath.LINETO:
            cur.append(v[-1])
        elif code in (MPath.CURVE3, MPath.CURVE4):
            curve(v)
        elif code == MPath.CLOSEPOLY:
            if len(cur) > 2:
                polys.append(np.array(cur))
            cur = []
    if len(cur) > 2:
        polys.append(np.array(cur))
    out = []
    for p in polys:
        if np.allclose(p[0], p[-1]):
            p = p[:-1]
        keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]
        out.append(p[keep])
    return out


def words(text, cap=CAP):
    """A line of text in text coordinates (mm; x along the line, y up, baseline at 0): the glyphs placed as the font
    lays them out, TRACK more apart each, grown by BOLD. Returns its contours and its ink extent along x."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath, TextToPath
    fp = FontProperties(family=FONT[0], weight=FONT[1])
    em = cap / TextPath((0, 0), "H", size=1.0, prop=fp).get_extents().height
    t2p = TextToPath()
    glyphs = t2p.get_glyphs_with_font(t2p._get_font(fp), text, return_new_glyphs_only=False)[0]
    parts = []
    for k, (ch, g) in enumerate(zip(text, glyphs)):
        if not ch.isspace():
            x = g[1] / t2p.FONT_SCALE * em + k * TRACK
            parts.append(CrossSection(flatten(TextPath((x, 0), ch, size=em, prop=fp)), FillRule.EvenOdd))
    cs = CrossSection.batch_boolean(parts, OpType.Add).offset(BOLD, JoinType.Round, circular_segments=32)
    x0, _, x1, _ = cs.bounds()
    return [np.asarray(p, float) for p in cs.to_polygons()], float(x0), float(x1)


def placed_words(text, centre, vertical=False):
    """Text centred (its cap height too) on a point in board coordinates; vertical reads from bottom to top, the
    letters' tops towards the board."""
    polys, x0, x1 = words(text)
    cx, cy = centre
    out = []
    for p in polys:
        u, v = p[:, 0] - (x0 + x1) / 2, p[:, 1] - CAP / 2
        q = np.column_stack([cx - v, cy - u]) if vertical else np.column_stack([cx + u, cy - v])
        out.append(to_plate(q))
    return out, x1 - x0


def marks():
    """Dimension marks and label: list of (name, contours, fill rule), and the label's widest line."""
    nx, ny = SQUARES
    L, H = nx * SQUARE, ny * SQUARE
    out = []
    # above the board: the long side
    y1, y0 = -TICK_GAP, -TICK_GAP - TICK_L
    yc = (y0 + y1) / 2
    text, w = placed_words(f"{L:.1f} mm", (L / 2, yc))
    gap = w / 2 + WORD_GAP
    out.append(("long", [rect(0, y0, TICK_W, y1), rect(L - TICK_W, y0, L, y1),
                         rect(TICK_W, yc - BAR / 2, L / 2 - gap, yc + BAR / 2),
                         rect(L / 2 + gap, yc - BAR / 2, L - TICK_W, yc + BAR / 2)], FillRule.Positive))
    out.append(("long_words", text, FillRule.EvenOdd))
    # right of the board: the short side
    x0, x1 = L + TICK_GAP, L + TICK_GAP + TICK_L
    xc = (x0 + x1) / 2
    text, w = placed_words(f"{H:.1f} mm", (xc, H / 2), vertical=True)
    gap = w / 2 + WORD_GAP
    out.append(("short", [rect(x0, 0, x1, TICK_W), rect(x0, H - TICK_W, x1, H),
                          rect(xc - BAR / 2, TICK_W, xc + BAR / 2, H / 2 - gap),
                          rect(xc - BAR / 2, H / 2 + gap, xc + BAR / 2, H - TICK_W)], FillRule.Positive))
    out.append(("short_words", text, FillRule.EvenOdd))
    # below the board: the label, centred under it
    widest = 0.0
    for k, line in enumerate(LABEL):
        text, w = placed_words(line, (L / 2, H + LABEL_GAP + BOLD + CAP / 2 + k * LINE_PITCH))
        out.append((f"label_{k}", text, FillRule.EvenOdd))
        widest = max(widest, w)
    return out, widest


def build():
    """The two bodies (manifold3d.Manifold), the black cross-section and facts for the JSON and the checks."""
    black_cells = board_cells()
    contours, cuts = board_contours(black_cells)
    parts = [CrossSection(contours, FillRule.Positive) - CrossSection(cuts, FillRule.Positive)]
    mk, label_w = marks()
    for _, c, rule in mk:
        parts.append(CrossSection(c, rule))
    ink = CrossSection.batch_boolean(parts, OpType.Add)
    r = CORNER_R
    outline = CrossSection.square((PLATE_W - 2 * r, PLATE_H - 2 * r)).translate((r, r)).offset(
        r, JoinType.Round, circular_segments=72)
    white = Manifold.extrude(outline, BASE)
    black = Manifold.extrude(ink, INK).translate((0.0, 0.0, BASE))
    facts = {"corner_cuts": len(cuts), "label_width_mm": round(label_w, 1), "cells": black_cells}
    return white, black, ink, facts


# ── files ────────────────────────────────────────────────────────────────────
def mesh_arrays(m):
    mesh = m.to_mesh()
    V = np.asarray(mesh.vert_properties, np.float64)[:, :3]
    return V, flip_slivers(V, np.asarray(mesh.tri_verts, np.int64))


def flip_slivers(V, F, eps=1e-9):
    """Remove the zero-area triangles the cap triangulation leaves where vertices of two contours lie on one line:
    such a triangle (a, b, c) has c on its edge ab; with the neighbour (b, a, d) across ab it is turned into (a, d, c)
    and (d, b, c), which cover the same flat area. The mesh stays closed and two-manifold."""
    F = F.copy()

    def area(t):
        p = V[F[t]]
        return 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))

    edges = {}
    for t, (i, j, k) in enumerate(F):
        edges[(i, j)], edges[(j, k)], edges[(k, i)] = t, t, t
    for _ in range(5):
        todo = [t for t in range(len(F)) if area(t) < eps]
        if not todo:
            break
        for t in todo:
            tri = F[t]
            if area(t) >= eps:
                continue
            lengths = [np.linalg.norm(V[tri[(e + 1) % 3]] - V[tri[e]]) for e in range(3)]
            e = int(np.argmax(lengths))
            a, b, c = tri[e], tri[(e + 1) % 3], tri[(e + 2) % 3]
            u = edges.get((b, a))
            if u is None:
                continue
            d = next(v for v in F[u] if v not in (a, b))
            if (c, d) in edges or (d, c) in edges or abs(V[d, 2] - V[c, 2]) > 1e-9 or abs(V[a, 2] - V[c, 2]) > 1e-9:
                continue
            for x, y in ((a, b), (b, c), (c, a), (b, a), (a, d), (d, b)):
                edges.pop((x, y), None)
            F[t], F[u] = (a, d, c), (d, b, c)
            for tt in (t, u):
                i, j, k = F[tt]
                edges[(i, j)], edges[(j, k)], edges[(k, i)] = tt, tt, tt
    return F


def write_stl(path, V, F, title):
    T = V[F].astype(np.float32)
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")])
    rec["n"], rec["v"] = N, T.reshape(-1, 9)
    with open(path, "wb") as fh:
        fh.write(title.encode("ascii")[:80].ljust(80, b" ") + struct.pack("<I", len(F)))
        fh.write(rec.tobytes())


def read_stl(path):
    """Binary STL: vertices merged where their coordinates are equal, triangles."""
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=[("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")])
    P = rec["v"].reshape(-1, 3).astype(np.float64)
    V, inv = np.unique(P, axis=0, return_inverse=True)
    return V, inv.reshape(-1, 3)


def placement():
    """Where the 3MF puts the plate: turned by 90 degrees (long side along the bed's y), centred in the area both
    nozzles reach. Returns the 3 x 4 transform and the footprint on the bed."""
    w, h = PLATE_H, PLATE_W                         # footprint after the turn
    x0 = (BOTH_NOZZLES_X[0] + BOTH_NOZZLES_X[1] - w) / 2
    y0 = (BED[1] - h) / 2
    # plate (X, Y) -> bed (x0 + PLATE_H - Y, y0 + X): a turn of +90 degrees about z
    M = np.array([[0.0, -1.0, 0.0, x0 + PLATE_H], [1.0, 0.0, 0.0, y0], [0.0, 0.0, 1.0, 0.0]])
    return M, (x0, y0, x0 + w, y0 + h)


def write_3mf(path, bodies):
    """3MF (core spec, mm): one object of two parts, each on its filament, with PRINT as the object's settings (Bambu
    Studio's model_settings.config), placed on the bed by its build item. bodies: [(name, V, F, filament)]."""
    M, _ = placement()
    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
           f'<metadata name="Title">{NAME}</metadata>', '<resources>']
    comps = []
    for i, (name, V, F, _) in enumerate(bodies, 1):
        xml.append(f'<object id="{i}" name="{name}" type="model"><mesh><vertices>')
        xml.append("".join(f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in V))
        xml.append("</vertices><triangles>")
        xml.append("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in F))
        xml.append("</triangles></mesh></object>")
        comps.append(f'<component objectid="{i}"/>')
    top = len(bodies) + 1
    xml.append(f'<object id="{top}" name="{NAME}" type="model"><components>{"".join(comps)}</components></object>')
    t = " ".join(f"{v:.6f}" for v in M.T.ravel())
    xml += ["</resources>", "<build>", f'<item objectid="{top}" transform="{t}"/>', "</build>", "</model>"]
    cfg = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>", f'  <object id="{top}">',
           f'    <metadata key="name" value="{NAME}"/>', '    <metadata key="extruder" value="1"/>']
    cfg += [f'    <metadata key="{k}" value="{v}"/>' for k, v in PRINT.items()]
    for i, (name, _, _, filament) in enumerate(bodies, 1):
        cfg += [f'    <part id="{i}" subtype="normal_part">', f'      <metadata key="name" value="{name}"/>',
                f'      <metadata key="extruder" value="{filament}"/>', "    </part>"]
    cfg += ["  </object>", "</config>"]
    ct = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
          'content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
          'relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-'
          '3dmodel+xml"/><Default Extension="config" ContentType="text/xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/'
            '2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.'
            'com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in (("[Content_Types].xml", ct), ("_rels/.rels", rels), ("3D/3dmodel.model", "\n".join(xml)),
                           ("Metadata/model_settings.config", "\n".join(cfg))):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, text)


def spec_json():
    """What NAME.json says: the board, what to measure, the plate."""
    nx, ny = SQUARES
    return {
        "name": NAME,
        "board": {"dictionary": DICTIONARY, "squares": [nx, ny], "square_mm": SQUARE, "marker_mm": MARKER,
                  "legacy_pattern": False, "size_mm": [nx * SQUARE, ny * SQUARE],
                  "corners": (nx - 1) * (ny - 1), "same_as": "calibrate_charuco.BOARD at this square"},
        "measure": {
            "long": {"nominal_mm": nx * SQUARE, "squares": nx, "axis": "board x",
                     "between": "outer faces of the two black ticks above the board, flush with its left and right "
                                "edges"},
            "short": {"nominal_mm": ny * SQUARE, "squares": ny, "axis": "board y",
                      "between": "outer faces of the two black ticks right of the board, flush with its top and "
                                 "bottom edges"},
            "tool": "caliper, outside jaws square to the face, tips on the white face",
        },
        "plate": {"size_mm": [PLATE_W, PLATE_H, BASE + INK], "base_mm": BASE, "ink_mm": INK,
                  "corner_radius_mm": CORNER_R, "board_origin_mm": [MARGIN["left"], PLATE_H - MARGIN["top"],
                                                                     BASE + INK],
                  "board_axes": {"x": [1, 0, 0], "y": [0, -1, 0], "z": [0, 0, -1]},
                  "corner_gap_mm": CORNER_GAP, "min_feature_mm": MIN_WIDTH},
        "files": {"white": f"{NAME}_white.stl", "black": f"{NAME}_black.stl", "3mf": f"{NAME}.3mf"},
        "print": {"layer_mm": LAYER_H, "colour_change_mm": BASE, "filament_1": "white", "filament_2": "black",
                  "settings": PRINT},
    }


def cmd_build(_args):
    white, black, *_ = build()
    OUT.mkdir(parents=True, exist_ok=True)
    bodies = []
    for name, m, fil in (("white", white, 1), ("black", black, 2)):
        V, F = mesh_arrays(m)
        write_stl(OUT / f"{NAME}_{name}.stl", V, F, f"{NAME} {name}")
        bodies.append((name, V, F, fil))
    write_3mf(OUT / f"{NAME}.3mf", bodies)
    (OUT / f"{NAME}.json").write_text(json.dumps(spec_json(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"written: {OUT / NAME}_white.stl, _black.stl, .3mf, .json")
    return cmd_check(_args)


# ── checks ───────────────────────────────────────────────────────────────────
FAILS = []


def expect(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILS.append(what)


def check_layout(cells):
    nx, ny = SQUARES
    img = make_board().generateImage((nx * CELLS, ny * CELLS), marginSize=0, borderBits=1)
    same = np.array_equal(img < 128, cells)
    expect(same, f"layout: black cells equal OpenCV's board image ({nx * CELLS} x {ny * CELLS} cells, "
                 f"{int(cells.sum())} black)")


def check_mesh(name, V, F):
    """Closed two-manifold, consistently outward, positive volume, no degenerate or self-intersecting triangles."""
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    directed = np.unique(e, axis=0)
    two = bool(np.all(counts == 2)) and len(directed) == len(e)
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    vol = float(np.sum(np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2]))) / 6)
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F.astype(np.int32)))
    topo = ms.get_topological_measures()
    ms.compute_selection_by_self_intersections_per_face()
    self_x = ms.current_mesh().selected_face_number()
    expect(two and topo["boundary_edges"] == 0 and topo["non_two_manifold_edges"] == 0
           and topo["non_two_manifold_vertices"] == 0,
           f"{name}: closed and two-manifold ({len(F)} triangles, {topo['connected_components_number']} shells, "
           f"every edge in two triangles, opposite ways)")
    expect(vol > 0 and area.min() > 1e-8, f"{name}: outward, volume {vol / 1000:.1f} cm3, smallest triangle "
                                          f"{area.min():.2e} mm2")
    expect(self_x == 0, f"{name}: no self intersections ({self_x} triangles)")
    return vol


def check_widths(ink):
    """Every black feature at least MIN_WIDTH wide: eroded by just under half of it, no part vanishes or splits."""
    r = MIN_WIDTH / 2 - 0.02
    bad = []
    comps = ink.decompose()
    for k, comp in enumerate(comps):
        n = sum(1 for c in comp.offset(-r, JoinType.Miter, 2.0).decompose() if c.area() > 1e-9)
        if n != 1:
            b = comp.bounds()
            bad.append((k, n, [round(v, 1) for v in b]))
    expect(not bad, f"walls: {len(comps)} black features, all at least {MIN_WIDTH} mm wide"
                    + (f"; too thin: {bad[:6]}" if bad else ""))


def check_fit():
    M, (x0, y0, x1, y1) = placement()
    ok = (BOTH_NOZZLES_X[0] + BED_EDGE / 2 <= x0 and x1 <= BOTH_NOZZLES_X[1] - BED_EDGE / 2
          and BED_EDGE <= y0 and y1 <= BED[1] - BED_EDGE and BASE + INK < 256)
    expect(ok, f"bed: plate {PLATE_W:g} x {PLATE_H:g} x {BASE + INK:g}, placed at x {x0:.1f}-{x1:.1f}, "
               f"y {y0:.1f}-{y1:.1f} (both nozzles: x {BOTH_NOZZLES_X[0]}-{BOTH_NOZZLES_X[1]})")


def face_texture(Vb, Fb, ppm=TEX_PPM, ss=4):
    """The face seen from above, rasterised from the black mesh's top triangles: white 255, black 0, the plate's
    rounded corners grey; page coordinates in mm from the plate's top left corner (as calibrate_charuco's page)."""
    s = ppm * ss
    w, h = round(PLATE_W * s), round(PLATE_H * s)
    img = np.full((h, w), 255, np.uint8)
    T = Vb[Fb]
    top = np.all(np.abs(T[:, :, 2] - (BASE + INK)) < 1e-4, axis=1)
    for tri in T[top]:
        pts = np.column_stack([tri[:, 0] * s, (PLATE_H - tri[:, 1]) * s])
        cv2.fillConvexPoly(img, np.round(pts * 16).astype(np.int32), 0, lineType=cv2.LINE_8, shift=4)
    r = CORNER_R * s
    mask = np.zeros_like(img)
    cv2.rectangle(mask, (round(r), 0), (w - 1 - round(r), h - 1), 255, -1)
    cv2.rectangle(mask, (0, round(r)), (w - 1, h - 1 - round(r)), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(mask, (round(cx), round(cy)), round(r), 255, -1)
    img[mask == 0] = 128
    return cv2.resize(img, (round(PLATE_W * ppm), round(PLATE_H * ppm)), interpolation=cv2.INTER_AREA)


def view_pose(dist, ax, ay, roll):
    """R, t of the board (board coordinates, mm) in the camera: the board's centre on the optical axis."""
    nx, ny = SQUARES
    centre = np.array([nx * SQUARE / 2, ny * SQUARE / 2, 0.0])
    R = calibrate_charuco.rotation([0, 0, math.radians(roll)]) @ calibrate_charuco.rotation(
        [math.radians(ax), 0, 0]) @ calibrate_charuco.rotation([0, math.radians(ay), 0])
    t = np.array([0.0, 0.0, float(dist)]) - R @ centre
    return R, t


def camera(kind):
    if kind == "stream":
        w, h = STREAM["size"]
        K = np.array([[STREAM["f"], 0, (w - 1) / 2], [0, STREAM["f"], (h - 1) / 2], [0, 0, 1.0]])
        return K, np.zeros(5), (w, h)
    return PHOTO["K"], PHOTO["dist"], PHOTO["size"]


def detect_views(paths, truths):
    """Run calibrate_charuco.detect on each image. Per image: corners found, of all, their RMS error against the true
    projection, and the RMS residual of the best pose of a flat board (solvePnP with the true camera): what a
    calibration sees of the relief, beyond a shift of the whole board."""
    board = make_board()
    n_all = len(board.getChessboardCorners())
    rows = []
    for p, (K, dist, R, t) in zip(paths, truths):
        views = calibrate_charuco.detect([p], board)[0]
        if p.name not in views:
            rows.append((p.name, 0, n_all, float("nan"), float("nan")))
            continue
        obj, img = views[p.name]
        obj, img = obj.astype(np.float64).reshape(-1, 3), img.astype(np.float64).reshape(-1, 2)
        proj = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2)
        err = np.linalg.norm(proj - img, axis=1)
        _, rv, tv = cv2.solvePnP(obj, img, K, dist, cv2.Rodrigues(R)[0], t.reshape(3, 1).copy(), True)
        fit = np.linalg.norm(cv2.projectPoints(obj, rv, tv, K, dist)[0].reshape(-1, 2) - img, axis=1)
        rows.append((p.name, len(obj), n_all, float(np.sqrt(np.mean(err ** 2))), float(np.sqrt(np.mean(fit ** 2)))))
    return rows


def photo(page, origin, rays, R, t, rng, blur=PHOTO_BLUR, noise=PHOTO_NOISE):
    """A photo of the face (calibrate_charuco's camera model and texture mapping) with a fixed, ordinary focus blur
    and noise: black 40, white 215 grey levels, the table 100."""
    mx, my = calibrate_charuco.board_maps(rays, R, t, origin, TEX_PPM)
    face = cv2.remap(page, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=85)
    img = 40 + 175 * face.astype(np.float32) / 255
    img = cv2.GaussianBlur(img, (0, 0), blur) + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def check_detection(Vb, Fb):
    out = WORK / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    page = face_texture(Vb, Fb)
    cv2.imwrite(str(WORK / "face_texture.png"), page)
    page = cv2.GaussianBlur(page, (0, 0), 0.6)
    origin = (MARGIN["left"], MARGIN["top"])
    rng = np.random.default_rng(0)
    paths, truths = [], []
    rays_cache = {}
    for i, (kind, dist, ax, ay, roll) in enumerate(VIEWS):
        K, d, size = camera(kind)
        if kind not in rays_cache:
            rays_cache[kind] = calibrate_charuco.pixel_rays(K, d, size)
        R, t = view_pose(dist, ax, ay, roll)
        img = photo(page, origin, rays_cache[kind], R, t, rng)
        p = out / f"view_{i:02d}_{kind}_{dist}mm_{ax}_{ay}_{roll}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
        truths.append((K, d, R, t))
    rows = detect_views(paths, truths)
    for name, n, n_all, rms, fit in rows:
        expect(n == n_all and fit < MAX_FIT_PX, f"detection: {name}: {n} of {n_all} corners, error {rms:.3f} px RMS, "
                                                f"flat board fit {fit:.3f} px")


def cmd_check(_args):
    white, black, ink, facts = build()
    check_layout(facts["cells"])
    files = {k: OUT / f"{NAME}_{k}.stl" for k in ("white", "black")}
    for k, p in files.items():
        expect(p.is_file(), f"file {p.name}")
    if FAILS:
        return 1
    Vw, Fw = read_stl(files["white"])
    Vb, Fb = read_stl(files["black"])
    vw = check_mesh("white", Vw, Fw)
    vb = check_mesh("black", Vb, Fb)
    both = (white ^ black).volume()
    expect(abs(both) < 1e-6 and abs((white + black).volume() / (vw + vb) - 1) < 1e-6,
           f"bodies touch at z = {BASE:g} without overlapping (common volume {both:.2e} mm3)")
    zb = Vb[:, 2]
    expect(abs(zb.min() - BASE) < 1e-5 and abs(zb.max() - BASE - INK) < 1e-5 and abs(Vw[:, 2].max() - BASE) < 1e-5,
           f"heights: white 0-{Vw[:, 2].max():.3f}, black {zb.min():.3f}-{zb.max():.3f}")
    check_widths(ink)
    expect(facts["label_width_mm"] <= SQUARES[0] * SQUARE - 10,
           f"label {facts['label_width_mm']} mm long, clear of the caliper's place at the lower right")
    print(f"      {facts['corner_cuts']} corners parted by {CORNER_GAP} mm")
    check_fit()
    with zipfile.ZipFile(OUT / f"{NAME}.3mf") as z:
        names = set(z.namelist())
        cfg = z.read("Metadata/model_settings.config").decode() if "Metadata/model_settings.config" in names else ""
    expect({"3D/3dmodel.model", "Metadata/model_settings.config"} <= names
           and all(f'<metadata key="{k}" value="{v}"/>' in cfg for k, v in PRINT.items()),
           f"3MF {NAME}.3mf, with the print settings " + ", ".join(f"{k} {v}" for k, v in PRINT.items()))
    n = BASE / LAYER_H
    expect(abs(n - round(n)) < 1e-9, f"colour change at the top of a layer: BASE {BASE:g} = {round(n)} x {LAYER_H:g}")
    spec = json.loads((OUT / f"{NAME}.json").read_text(encoding="utf-8"))
    expect(spec == json.loads(json.dumps(spec_json())), f"{NAME}.json up to date")
    check_detection(Vb, Fb)
    print("check: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
    return 1 if FAILS else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    sub.add_parser("render")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    sys.exit({"build": cmd_build, "check": cmd_check, "render": cmd_render}[args.cmd](args))


# ── Blender renders ──────────────────────────────────────────────────────────
RENDER_VIEWS = [  # (name, camera, distance mm, tilt about board x, about board y, roll; degrees)
    ("stream_300", "stream", 300, 15, 0, 0), ("stream_450_tilt45", "stream", 450, 45, 0, 0),
    ("stream_600_oblique", "stream", 600, 30, 30, 30), ("photo_300_tilt40", "photo", 300, 0, 40, 0),
]
SAMPLES = 64
RENDER_TILTED_SHARE = 0.9   # beyond 30 degrees of tilt a marker of the far side may be missed (its 4 corners)

BLENDER_SCRIPT = r'''
import json, sys, bpy
from mathutils import Matrix
job = json.loads(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8").read())
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = job["samples"]
scene.cycles.use_denoising = False
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"
def material(name, rgb, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = 0.25    # matte PLA
    return m
for path, rgb, rough in ((job["white"], (0.80, 0.80, 0.78), 0.6), (job["black"], (0.025, 0.025, 0.025), 0.55)):
    bpy.ops.wm.stl_import(filepath=path)
    ob = bpy.context.selected_objects[0]
    ob.data.materials.append(material(ob.name, rgb, rough))
bpy.ops.mesh.primitive_plane_add(size=20000, location=(job["plate"][0] / 2, job["plate"][1] / 2, -0.01))
bpy.context.object.data.materials.append(material("table", (0.18, 0.17, 0.16), 0.8))
world = bpy.data.worlds.new("world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
scene.world = world
light = bpy.data.lights.new("lamp", "SUN")
light.energy, light.angle = 2.4, 0.2             # W/m2; 11 degrees wide, soft shadows like a ceiling lamp
lamp = bpy.data.objects.new("lamp", light)
lamp.rotation_euler = (0.45, -0.3, 0.4)
scene.collection.objects.link(lamp)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.sensor_fit = "HORIZONTAL"
cam.data.sensor_width = 36.0
cam.data.clip_start, cam.data.clip_end = 10.0, 10000.0
for v in job["views"]:
    w, h = v["size"]
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = w, h, 100
    cam.data.lens = v["f"] * 36.0 / w
    cam.matrix_world = Matrix(v["matrix"])
    scene.render.filepath = v["path"]
    scene.render.image_settings.color_mode = "RGB" if v["colour"] else "BW"
    bpy.ops.render.render(write_still=True)
'''


def blender_camera(R, t):
    """Blender's camera matrix for the board pose R, t (board coordinates in the OpenCV camera)."""
    A = np.diag([1.0, -1.0, -1.0])                       # board axes in plate coordinates
    O = np.array([MARGIN["left"], PLATE_H - MARGIN["top"], BASE + INK])
    Rwc = A @ R.T                                        # OpenCV camera axes in plate coordinates
    C = O - Rwc @ t
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = Rwc @ np.diag([1.0, -1.0, -1.0]), C
    return M


def cmd_render(_args):
    """Blender renders of the printed plate (relief, matte white and black, soft light): a preview and RENDER_VIEWS,
    each found by calibrate_charuco.detect and compared with the true projection of the corners (on the black top)."""
    out = WORK / "blender"
    out.mkdir(parents=True, exist_ok=True)
    views, truths = [], []
    for name, kind, dist, ax, ay, roll in RENDER_VIEWS:
        K, _, (w, h) = camera(kind)
        K = np.array([[K[0, 0], 0, (w - 1) / 2], [0, K[0, 0], (h - 1) / 2], [0, 0, 1.0]])   # Blender's: centred
        R, t = view_pose(dist, ax, ay, roll)
        views.append({"size": [w, h], "f": float(K[0, 0]), "matrix": blender_camera(R, t).tolist(),
                      "path": str(out / f"{name}.png"), "colour": False})
        truths.append((K, np.zeros(5), R, t))
    # preview: the whole plate from the front, low, its edge in view
    R, t = view_pose(390, -48, 12, 0)
    views.append({"size": [1800, 1350], "f": 1500.0, "matrix": blender_camera(R, t).tolist(),
                  "path": str(out / "preview.png"), "colour": True})
    job = {"white": str(OUT / f"{NAME}_white.stl"), "black": str(OUT / f"{NAME}_black.stl"), "samples": SAMPLES,
           "plate": [PLATE_W, PLATE_H], "views": views}
    (out / "job.json").write_text(json.dumps(job), encoding="utf-8")
    (out / "render.py").write_text(BLENDER_SCRIPT, encoding="utf-8")
    r = subprocess.run([str(BLENDER), "--background", "--factory-startup", "--python-exit-code", "1",
                        "--python", str(out / "render.py"), "--", str(out / "job.json")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        return 1
    rows = detect_views([Path(v["path"]) for v in views[:-1]], truths)
    for (name, n, n_all, rms, fit), (_, _, _, ax, ay, _) in zip(rows, RENDER_VIEWS):
        need = n_all if max(abs(ax), abs(ay)) <= 30 else math.ceil(RENDER_TILTED_SHARE * n_all)
        expect(n >= need and fit < MAX_FIT_PX, f"render: {name}: {n} of {n_all} corners (at least {need}), error "
                                               f"{rms:.3f} px RMS against the black top, flat board fit {fit:.3f} px")
    print(f"preview: {out / 'preview.png'}")
    print("render: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    main()
