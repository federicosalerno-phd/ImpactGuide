"""Charts of the results in the shared figure style, each one also a figure script that draws it on its own.

A chart is a spec (a dict, DEFAULT's keys): its kind (KINDS), the table it reads (a CSV of the results folder), the
column along x, the value (y) and the column that makes the series, the page width (style.WIDTHS_MM: one column,
one and a half, the full width) and the height in mm, the scale of the values, the palette, the limits drawn as
dotted lines, fixed ticks of the values, where the legend goes. script() turns a spec into the source of a figure
script: a plain matplotlib program that reads its table, draws with style.py (Abadi, the shared colours and
encodings) and writes the figure with style.save (vector PDF with embedded fonts and a 300 dpi PNG). The same source
draws the preview (preview(): an SVG for the page), the exports (export(): PDF and SVG, vector, and PNG at 300 dpi)
and, pinned (pin()), it is written into the figures folder as <name>.py and run there in a process of its own: the
figure depends on the table and on style.py only. The spec rides in the script's first comment line, so a pinned
chart opens again (saved()).

Kinds: summary (median and interquartile range per level, a line per series), box (box plots per level and series),
points (every row a point: jittered over levels, or against a numeric x), share (bars: the share of the rows at or
under the first limit), cumulative (the share of rows at or under each value). Palettes: style (the encodings of
style.py for the conditions and the interfaces, the categorical hues otherwise), categorical, mono (ink only, the
series told apart by marker and line). Values drawn from simulated tables are marked as placeholder data (style.save
and a watermark).

    python chart_builder.py test     checks; exit code 1 on failure
"""
from __future__ import annotations

import base64
import copy
import csv
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "04_code" / "figures"          # style.py and the figure scripts
SPEC_LINE = "# chart spec: "
KINDS = ("summary", "box", "points", "share", "cumulative")
PALETTES = ("style", "categorical", "mono")
SCALES = ("linear", "log")
LEGENDS = ("upper left", "upper right", "lower right", "best", "above", "none")
WIDTHS_MM = (90, 140, 190)      # style.WIDTHS_MM, for the page's lists without loading matplotlib (the test compares)
HEIGHT_MM = (30, 230)
DEFAULT_HEIGHT = {90: 62, 140: 75, 190: 90}
MAX_LIMITS, MAX_TICKS = 4, 12
TABLE_RE = re.compile(r"[a-z0-9_]{1,60}\.csv")
COLUMN_RE = re.compile(r"[a-z0-9_]{1,40}")
RUN_TIMEOUT_S = 180
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
DASHES = ("-", "--", ":", "-.")
# What the builder knows of a column: its axis label (English, as the figures are), whether it holds levels (to
# group by) or values (to draw), and for levels how a tick is written ({} the value) or the name of each value.
COLUMNS = {
    "energy_level": ("Impact energy level", "level", "E{}"),
    "angle_deg": ("Release angle (°)", "level", "{:g}°"),
    "direction_deg": ("Impact direction (°)", "level", "{:g}°"),
    "drape": ("Drape", "level", {0: "Without drape", 1: "With drape"}),
    "condition": ("Condition", "level", None),
    "holder": ("Interface", "level", "{}"),
    "target": ("Target", "level", "{}"),
    "session": ("Session", "level", "{}"),
    "input": ("View input", "level", "{}"),
    "curve": ("Curve", "level", "{}"),
    "seating": ("Seating", "value", None),
    "repeat": ("Repeat", "value", None),
    "tre_mm": ("TRE (mm)", "value", None),
    "tre_mean_mm": ("Mean TRE (mm)", "value", None),
    "tre_max_mm": ("Largest TRE (mm)", "value", None),
    "tre_ai_mm": ("TRE after the AI step (mm)", "value", None),
    "tre_corrected_mm": ("TRE after the correction (mm)", "value", None),
    "tre_pred95_mm": ("Predicted TRE, 95% (mm)", "value", None),
    "rot_deg": ("Marker rotation (°)", "value", None),
    "shift_mm": ("Marker shift (mm)", "value", None),
    "energy_mj": ("Impact energy (mJ)", "value", None),
    "score": ("Misalignment score (mm)", "value", None),
    "time_s": ("Time (s)", "value", None),
    "torque_nm": ("Slip torque (N·m)", "value", None),
    "mass_g": ("Mass (g)", "value", None),
    "x": ("x", "value", None),
    "y": ("y", "value", None),
}
SHORT = {"tre_mm": "TRE", "tre_mean_mm": "TRE", "tre_max_mm": "TRE", "tre_ai_mm": "TRE", "tre_corrected_mm": "TRE",
         "rot_deg": "rotation", "shift_mm": "shift", "score": "score", "time_s": "time"}
SKIP_COLUMNS = {"trial", "phase", "simulated", "key", "notes", "lost", "detected", "applied", "warning", "n"}

DEFAULT = {"name": "", "kind": "summary", "data": "trials_tre.csv", "x": "energy_level", "y": "tre_mm",
           "series": "condition", "width": 90, "height": 62, "scale": "log", "palette": "style",
           "limits": [1, 2], "y_ticks": None, "legend": "upper left"}
# Charts to start from: (key, spec). The figure name is chosen when pinned.
PRESETS = (
    ("tre_energy", {"kind": "summary", "data": "trials_tre.csv", "x": "energy_level", "y": "tre_mm",
                    "series": "condition", "width": 90, "height": 62, "scale": "log", "limits": [1, 2],
                    "y_ticks": [0.2, 0.5, 1, 2, 5, 10], "legend": "upper left"}),
    ("tre_direction", {"kind": "summary", "data": "trials_tre.csv", "x": "direction_deg", "y": "tre_mm",
                       "series": "condition", "width": 90, "height": 62, "scale": "log", "limits": [1, 2],
                       "legend": "upper left"}),
    ("displacement", {"kind": "summary", "data": "trials_metrics.csv", "x": "energy_level", "y": "rot_deg",
                      "series": "holder", "width": 90, "height": 62, "scale": "linear", "limits": [],
                      "legend": "upper left"}),
    ("tre_targets", {"kind": "box", "data": "trials_targets.csv", "x": "target", "y": "tre_mm", "series": "holder",
                     "width": 140, "height": 70, "scale": "log", "limits": [1, 2], "legend": "upper right"}),
    ("share_2mm", {"kind": "share", "data": "trials_tre.csv", "x": "energy_level", "y": "tre_mm",
                   "series": "condition", "width": 140, "height": 62, "scale": "linear", "limits": [2],
                   "legend": "above"}),
    ("cumulative", {"kind": "cumulative", "data": "trials_tre.csv", "x": "", "y": "tre_mm", "series": "condition",
                    "width": 90, "height": 62, "scale": "log", "limits": [1, 2], "legend": "lower right"}),
    ("score", {"kind": "points", "data": "trials_metrics.csv", "x": "tre_mean_mm", "y": "score", "series": "holder",
               "width": 90, "height": 70, "scale": "linear", "limits": [], "legend": "upper left"}),
    ("torque_a", {"kind": "points", "data": "interfaces_a.csv", "x": "repeat", "y": "torque_nm", "series": "",
                  "width": 90, "height": 60, "scale": "linear", "limits": [], "legend": "none"}),
    ("reseat_b", {"kind": "box", "data": "interfaces_b.csv", "x": "", "y": "tre_mean_mm", "series": "",
                  "width": 90, "height": 60, "scale": "linear", "limits": [], "legend": "none"}),
)
_LOCK = threading.Lock()        # pyplot draws one figure at a time, whatever thread asks


class ChartError(Exception):
    """A chart that cannot be made, with a code the page says: bad_spec, no_table, no_column, no_rows, no_font,
    script_exists, script_failed."""

    def __init__(self, code: str, message: str = "", **values) -> None:
        super().__init__(message or code)
        self.code, self.values = code, values


# ── the style ───────────────────────────────────────────────────────────────
def style_module():
    """style.py of the figures folder (imported once); ChartError no_font when its font is not installed."""
    if "style" in sys.modules and getattr(sys.modules["style"], "PICTURES_DIR", None) is not None:
        return sys.modules["style"]
    if str(FIGURES) not in sys.path:
        sys.path.insert(0, str(FIGURES))
    try:
        import style
    except ValueError as e:              # matplotlib: the font is not found
        raise ChartError("no_font", str(e)) from e
    return style


# ── tables ──────────────────────────────────────────────────────────────────
def header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f), [])


def tables(folder: Path) -> dict:
    """The tables of the results folder a chart can read: {file: {"levels": [...], "values": [...], "rows": n}}
    (columns in the table's order; a column with fewer than 13 distinct values is a level too)."""
    out = {}
    folder = Path(folder)
    if not folder.is_dir():
        return out
    for p in sorted(folder.glob("*.csv")):      # read with csv: no pandas for the page's lists
        if not TABLE_RE.fullmatch(p.name):
            continue
        try:
            with open(p, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        if len(rows) < 2 or not rows[0] or rows[0][0] == "key":
            continue
        levels, values = [], []
        for i, c in enumerate(rows[0]):
            cells = [r[i] for r in rows[1:] if i < len(r) and r[i] != ""]
            if c in SKIP_COLUMNS or not COLUMN_RE.fullmatch(c) or not cells:
                continue
            try:
                [float(x) for x in cells]
                numeric = True
            except ValueError:
                numeric = False
            known = COLUMNS.get(c, (c, None, None))[1]
            if known == "level" or (known is None and (not numeric or len(set(cells)) <= 12)):
                levels.append(c)
            if numeric and known != "level":
                values.append(c)
        out[p.name] = {"levels": levels, "values": values, "rows": len(rows) - 1}
    return out


def check(spec: dict, folder: Path) -> dict:
    """The spec filled from DEFAULT and checked against the table it reads: ChartError bad_spec, no_table or
    no_column when it does not hold."""
    s = copy.deepcopy(DEFAULT)
    s.update({k: v for k, v in (spec or {}).items() if k in DEFAULT})
    bad = lambda what: ChartError("bad_spec", what, field=what)     # noqa: E731
    if s["kind"] not in KINDS:
        raise bad("kind")
    if s["palette"] not in PALETTES:
        raise bad("palette")
    if s["scale"] not in SCALES:
        raise bad("scale")
    if s["legend"] not in LEGENDS:
        raise bad("legend")
    try:
        s["width"] = int(s["width"])
        s["height"] = float(s["height"])
    except (TypeError, ValueError):
        raise bad("width") from None
    style = style_module()
    if s["width"] not in style.WIDTHS_MM:
        raise bad("width")
    if not HEIGHT_MM[0] <= s["height"] <= HEIGHT_MM[1]:
        raise bad("height")
    s["height"] = int(s["height"]) if float(s["height"]).is_integer() else round(s["height"], 1)
    for k, most in (("limits", MAX_LIMITS), ("y_ticks", MAX_TICKS)):
        v = s[k]
        if v in (None, "") and k == "y_ticks":
            s[k] = None
            continue
        if not isinstance(v, (list, tuple)) or len(v) > most:
            raise bad(k)
        try:
            v = [float(x) for x in v]
        except (TypeError, ValueError):
            raise bad(k) from None
        if not all(math.isfinite(x) for x in v) or (k == "y_ticks" and s["scale"] == "log" and min(v, default=1) <= 0):
            raise bad(k)
        s[k] = [int(x) if x.is_integer() else x for x in v]
    if s["name"] and not style.NAME_PATTERN.fullmatch(s["name"]):
        raise bad("name")
    if not TABLE_RE.fullmatch(str(s["data"])):
        raise bad("data")
    path = Path(folder) / s["data"]
    if not path.is_file():
        raise ChartError("no_table", s["data"], name=s["data"])
    cols = header(path)
    for k in ("x", "y", "series"):
        v = s[k] or ""
        if v and (not COLUMN_RE.fullmatch(v) or v not in cols):
            raise ChartError("no_column", v, field=k)
        s[k] = v
    if not s["y"]:
        raise ChartError("no_column", "", field="y")
    if s["kind"] in ("summary", "share") and not s["x"]:
        raise ChartError("no_column", "", field="x")
    return s


# ── the script ──────────────────────────────────────────────────────────────
def label_of(col: str) -> str:
    return COLUMNS.get(col, (col.replace("_", " ").capitalize(), None, None))[0]


def is_level(col: str, data) -> bool:
    known = COLUMNS.get(col, (None, None, None))[1]
    if known:
        return known == "level"
    import pandas as pd
    return not pd.api.types.is_numeric_dtype(data[col]) or data[col].nunique() <= 12


def series_keys(spec: dict, data) -> list:
    if not spec["series"]:
        return ["all"]
    vals = [v for v in data[spec["series"]].dropna().unique().tolist()]
    if spec["series"] == "condition":
        order = ["A", "A_AI", "B", "B_AI"]
        return [v for v in order if v in vals] + sorted(v for v in vals if v not in order)
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=str)


def looks_code(spec: dict, keys: list) -> list[str]:
    """The line(s) that define LOOKS: key -> label, colour, line, marker, filled."""
    if spec["palette"] == "style" and spec["series"] == "condition" and set(keys) <= {"A", "A_AI", "B", "B_AI"}:
        return ["LOOKS = style.CONDITIONS"]
    lines = ["LOOKS = {"]
    for i, k in enumerate(keys):
        label = "None" if not spec["series"] else repr(legend_label(spec["series"], k))
        if spec["palette"] == "mono":
            color, dash, marker, filled = "style.INK", DASHES[i % len(DASHES)], MARKERS[i % len(MARKERS)], i % 2 == 0
        elif spec["palette"] == "style" and spec["series"] == "holder" and k in ("A", "B"):
            color, dash, marker, filled = f'style.INTERFACE_COLORS["{k}"]', "-", "o" if k == "A" else "s", True
        else:
            color, dash, marker, filled = f"style.CATEGORICAL[{i % 8}]", "-", MARKERS[i % len(MARKERS)], True
        lines.append(f"    {k!r}: dict(label={label}, color={color}, linestyle={dash!r}, marker={marker!r}, "
                     f"filled={filled}),")
    lines.append("}")
    return lines


def legend_label(col: str, key) -> str:
    if col == "holder":
        return f"Interface {key}"
    if col == "drape":
        return {0: "Without drape", 1: "With drape"}.get(int(key), str(key))
    ticks = COLUMNS.get(col, (None, None, None))[2]
    if isinstance(ticks, str):
        try:
            return ticks.format(key)
        except (ValueError, IndexError):
            return str(key)
    return str(key)


def ticks_code(col: str) -> str:
    t = COLUMNS.get(col, (None, None, "{}"))[2]
    if isinstance(t, dict):
        return f"TICK_NAMES = {t!r}"
    if col == "condition":
        return 'TICK_NAMES = {k: v["label"] for k, v in style.CONDITIONS.items()}'
    return f"TICK = {(t or '{}')!r}"


def dodge(keys: list, step: float) -> dict:
    n = len(keys)
    return {k: round((i - (n - 1) / 2) * step, 6) for i, k in enumerate(keys)}


STEP = {"summary": 0.08, "box": 0.7, "points": 0.6, "share": 0.8}


def script(spec: dict, data, data_expr: str, placeholder: bool = False, style_dir: Path | None = None,
           pictures: Path | None = None) -> str:
    """The source of the figure script of a checked spec. data: the table (for the series present and for whether
    x holds levels); data_expr: the Python expression of the table's path in the script (style.RESULTS_DIR / ... in
    the project); style_dir and pictures: where style.py is and where the figure goes, for a script kept outside the
    figures folder (a simulation)."""
    s = spec
    keys = series_keys(s, data)
    x_level = bool(s["x"]) and is_level(s["x"], data)
    kind = s["kind"]
    step = STEP.get(kind, 0)
    if kind == "box":
        step = STEP["box"] / max(1, len(keys)) if len(keys) > 1 else 0
    elif kind in ("points", "share"):
        step = STEP[kind] / max(1, len(keys)) if len(keys) > 1 else 0
    name = s["name"] or "figS0_chart"
    what = {"summary": "median and interquartile range", "box": "box plots", "points": "every row",
            "share": f"share at or under {s['limits'][0] if s['limits'] else 2:g}",
            "cumulative": "cumulative share"}[kind]
    doc = (f'"""{name}: {label_of(s["y"])}, {what}' + (f', by {label_of(s["x"]).lower()}' if s["x"] else "")
           + (f', one series per {label_of(s["series"]).lower()}' if s["series"] else "") + ".\n\n"
           f"Reads {s['data']} of the results folder. Written by ImpactGuide's chart builder from the spec in the first\n"
           f"comment line; it depends only on its table and on style.py. Run it on its own or with build_figures.py.\n"
           '"""')
    L = [SPEC_LINE + json.dumps({k: s[k] for k in DEFAULT}, ensure_ascii=False), doc]
    imports = ["import matplotlib.pyplot as plt", "import numpy as np", "import pandas as pd"]
    if style_dir is not None or not data_expr.startswith("style."):
        imports.append("from pathlib import Path")
    tick_imports = (["FixedLocator", "FuncFormatter", "NullLocator"] if s["y_ticks"] else
                    ["FuncFormatter", "LogLocator", "NullLocator"] if s["scale"] == "log" else [])
    if kind == "box":
        imports.append("from matplotlib.lines import Line2D")
    if tick_imports:
        imports.append(f"from matplotlib.ticker import {', '.join(tick_imports)}")
    L += [""] + imports + [""]
    if style_dir is not None:
        L += ["import sys", "", f"sys.path.insert(0, {str(style_dir)!r})  # where style.py is", ""]
    L += ["import style  # noqa: E402" if style_dir is not None else "import style", ""]
    if pictures is not None:
        L += [f"style.PICTURES_DIR = Path({str(pictures)!r})", ""]
    L += [f"NAME = {name!r}", f"DATA = {data_expr}", f"PLACEHOLDER = {bool(placeholder)}",
          f"X, Y, SERIES = {s['x'] or None!r}, {s['y']!r}, {s['series'] or None!r}",
          f"WIDTH_MM, HEIGHT_MM = {s['width']}, {s['height']}",
          f"LIMITS = {tuple(s['limits'])!r}"]
    if s["y_ticks"]:
        L.append(f"Y_TICKS = {list(s['y_ticks'])!r}")
    L += looks_code(s, keys)
    if step:
        L.append(f"DODGE = {dodge([k for k in keys], step)!r}")
    if kind == "box":
        L.append(f"BOX_WIDTH = {round(step * 0.8, 4) if step else 0.5}")
    elif kind == "share":
        L.append(f"BAR_WIDTH = {round(step * 0.9, 4) if step else 0.6}")
    elif kind == "points":
        L.append(f"SPREAD = {round(step * 0.3, 4) if step else 0.12}  # half the jitter across a level")
    if x_level:
        L.append(ticks_code(s["x"]))
    L += [f"X_LABEL, Y_LABEL = {label_of(s['x']) if s['x'] else ''!r}, {label_of(s['y'])!r}", "", ""]
    L += ["def draw(data):", '    """The figure from the table."""']
    L += ["    data = data[data[Y].notna()]"]
    if not s["series"]:
        L += ['    data = data.assign(_series="all")', '    series = "_series"']
    else:
        L += ["    series = SERIES"]
    if x_level:
        L += ["    levels = sorted(data[X].unique())", "    pos = {v: i + 1 for i, v in enumerate(levels)}"]
    elif not s["x"] and kind != "cumulative":
        L += ['    data = data.assign(_x=0)', '    levels, pos = [0], {0: 1}']
    body = {"summary": _summary, "box": _box, "points": _points, "share": _share, "cumulative": _cumulative}[kind]
    L += body(s, x_level, bool(step))
    L += _finish(s, x_level, kind)
    L += ["", "", "def main():", "    style.save(draw(pd.read_csv(DATA)), NAME, placeholder=PLACEHOLDER)", "", "",
          'if __name__ == "__main__":', "    main()", ""]
    return "\n".join(L)


def _hlines():
    return ["    for value in LIMITS:",
            '        ax.axhline(value, color=style.MUTED, linewidth=0.6, linestyle=":", zorder=0)']


def _xcol(x_level):
    return '"_x"' if not x_level else "X"


def _summary(s, x_level, dodged):
    off = " + DODGE[key]" if dodged else ""
    xc = "X" if s["x"] else '"_x"'
    return ([f"    summary = data.groupby([series, {xc}])[Y].quantile([0.25, 0.5, 0.75]).unstack()",
             "    fig, ax = plt.subplots(figsize=style.figsize(WIDTH_MM, HEIGHT_MM))"] + _hlines() +
            ["    for key, look in LOOKS.items():",
             "        if key not in summary.index:",
             "            continue",
             "        stats = summary.loc[key]",
             f"        x = np.array([pos[v] for v in stats.index]){off}",
             "        median = stats[0.5].to_numpy()",
             "        ax.errorbar(",
             '            x, median, yerr=[median - stats[0.25].to_numpy(), stats[0.75].to_numpy() - median],',
             '            label=look["label"], color=look["color"], linestyle=look["linestyle"],',
             '            marker=look["marker"], markerfacecolor=look["color"] if look["filled"] else "white",',
             "            markeredgewidth=0.9, elinewidth=0.8, capsize=0,",
             "        )"])


def _box(s, x_level, dodged):
    off = " + DODGE[key]" if dodged else ""
    xc = "X" if s["x"] else '"_x"'
    return (["    fig, ax = plt.subplots(figsize=style.figsize(WIDTH_MM, HEIGHT_MM))"] + _hlines() +
            ["    handles = []",
             "    for key, look in LOOKS.items():",
             "        part = data[data[series] == key]",
             "        if part.empty:",
             "            continue",
             f"        groups = [(pos[v]{off}, part.loc[part[{xc}] == v, Y].to_numpy()) for v in levels]",
             "        groups = [(x, g) for x, g in groups if len(g)]",
             '        face = "white"',
             '        edge = dict(color=look["color"], linewidth=0.8)',
             "        ax.boxplot([g for _, g in groups], positions=[x for x, _ in groups], widths=BOX_WIDTH,",
             "                   patch_artist=True, manage_ticks=False,",
             '                   boxprops=dict(facecolor=face, edgecolor=look["color"], linewidth=0.8),',
             '                   whiskerprops=dict(edge, linestyle=look["linestyle"]), capprops=edge,',
             '                   medianprops=dict(color=look["color"], linewidth=1.4),',
             '                   flierprops=dict(marker=look["marker"], markersize=2.5, markeredgewidth=0.6,',
             '                                   markeredgecolor=look["color"],',
             '                                   markerfacecolor=look["color"] if look["filled"] else "white"))',
             '        handles.append(Line2D([], [], color=look["color"], linestyle=look["linestyle"],',
             '                              marker=look["marker"], markersize=4, label=look["label"],',
             '                              markerfacecolor=look["color"] if look["filled"] else "white"))'])


def _points(s, x_level, dodged):
    off = " + DODGE[key]" if dodged else ""
    xc = "X" if s["x"] else '"_x"'
    common = ['                   s=9, marker=look["marker"], linewidths=0.6, edgecolors=look["color"],',
              '                   facecolors=look["color"] if look["filled"] else "white", label=look["label"],',
              "                   zorder=2)"]
    if x_level or not s["x"]:
        return (["    fig, ax = plt.subplots(figsize=style.figsize(WIDTH_MM, HEIGHT_MM))"] + _hlines() +
                ["    rng = np.random.default_rng(0)  # the same jitter at every run",
                 "    for key, look in LOOKS.items():",
                 "        part = data[data[series] == key]",
                 "        if part.empty:",
                 "            continue",
                 f"        x = np.array([pos[v] for v in part[{xc}]]){off}",
                 "        ax.scatter(x + rng.uniform(-SPREAD, SPREAD, len(x)), part[Y],"] + common +
                ["        for v in levels:",
                 f"            here = part.loc[part[{xc}] == v, Y]",
                 "            if len(here):",
                 f"                c = pos[v]{off}",
                 '                ax.hlines(here.median(), c - 2 * SPREAD, c + 2 * SPREAD, color=look["color"],',
                 "                          linewidth=1.2, zorder=3)"])
    return (["    fig, ax = plt.subplots(figsize=style.figsize(WIDTH_MM, HEIGHT_MM))"] + _hlines() +
            ["    for key, look in LOOKS.items():",
             "        part = data[data[series] == key]",
             "        if part.empty:",
             "            continue",
             "        ax.scatter(part[X], part[Y],"] + common)


def _share(s, x_level, dodged):
    off = " + DODGE[key]" if dodged else ""
    return (["    limit = LIMITS[0] if LIMITS else 2",
             "    fig, ax = plt.subplots(figsize=style.figsize(WIDTH_MM, HEIGHT_MM))",
             "    for key, look in LOOKS.items():",
             "        part = data[data[series] == key]",
             "        if part.empty:",
             "            continue",
             "        share = part.groupby(X)[Y].apply(lambda v: 100 * (v <= limit).mean())",
             f"        x = np.array([pos[v] for v in share.index]){off}",
             '        ax.bar(x, share.to_numpy(), width=BAR_WIDTH, label=look["label"], linewidth=0.8,',
             '               edgecolor=look["color"], color=look["color"] if look["filled"] else "white",',
             '               hatch=None if look["filled"] else "/////", linestyle=look["linestyle"])',
             "    ax.set_ylim(0, 100)"])


def _cumulative(s, x_level, dodged):
    return (["    fig, ax = plt.subplots(figsize=style.figsize(WIDTH_MM, HEIGHT_MM))",
             "    for value in LIMITS:",
             '        ax.axvline(value, color=style.MUTED, linewidth=0.6, linestyle=":", zorder=0)',
             "    for key, look in LOOKS.items():",
             "        v = np.sort(data.loc[data[series] == key, Y].to_numpy())",
             "        if not len(v):",
             "            continue",
             "        share = 100 * np.arange(1, len(v) + 1) / len(v)",
             '        ax.step(v, share, where="post", label=look["label"], color=look["color"],',
             '                linestyle=look["linestyle"])',
             "    ax.set_ylim(0, 100)"])


def _finish(s, x_level, kind):
    L = []
    value_axis = "x" if kind == "cumulative" else "y"
    if s["scale"] == "log":
        L.append(f'    ax.set_{value_axis}scale("log")')
    if s["y_ticks"] or s["scale"] == "log":
        locator = "FixedLocator(Y_TICKS)" if s["y_ticks"] else "LogLocator(subs=(1.0, 2.0, 5.0))"
        L += [f"    ax.{value_axis}axis.set_major_locator({locator})",
              f'    ax.{value_axis}axis.set_major_formatter(FuncFormatter(lambda v, _: f"{{v:g}}"))',
              f"    ax.{value_axis}axis.set_minor_locator(NullLocator())"]
    if kind == "cumulative":
        L += ['    ax.set_ylabel("Trials (%)")', "    ax.set_xlabel(Y_LABEL)"]
    else:
        if kind == "share":
            short = SHORT.get(s["y"], label_of(s["y"]).split(" (")[0])
            unit = label_of(s["y"]).split("(")[-1].rstrip(")") if "(" in label_of(s["y"]) else ""
            L.append(f'    ax.set_ylabel(f"Trials with {short} \\u2264 {{limit:g}} {unit} (%)".replace("  ", " "))')
        else:
            L.append("    ax.set_ylabel(Y_LABEL)")
        if x_level:
            if isinstance(COLUMNS.get(s["x"], (0, 0, "{}"))[2], dict) or s["x"] == "condition":
                L.append("    ax.set_xticks(list(pos.values()), [TICK_NAMES.get(v, str(v)) for v in levels])")
            else:
                L.append("    ax.set_xticks(list(pos.values()), [TICK.format(v) for v in levels])")
            L += ["    ax.set_xlim(min(pos.values()) - 0.4, max(pos.values()) + 0.4)", "    ax.set_xlabel(X_LABEL)"]
        elif not s["x"]:
            L += ["    ax.set_xticks([])", "    ax.set_xlim(0.4, 1.6)"]
        else:
            L.append("    ax.set_xlabel(X_LABEL)")
    if s["series"] and s["legend"] != "none":
        hl = ", handlelength=2.6" if kind in ("summary", "cumulative") else ""
        handles = "handles=handles, " if kind == "box" else ""
        if s["legend"] == "above":
            L.append(f'    ax.legend({handles}loc="lower left", bbox_to_anchor=(0, 1.01), ncols=len(LOOKS), '
                     f'borderaxespad=0{hl})')
        else:
            L.append(f"    ax.legend({handles}loc={s['legend']!r}{hl})")
    L += ["    if PLACEHOLDER:",
          '        ax.text(0.5, 0.5, "placeholder data", transform=ax.transAxes, ha="center", va="center",',
          "                rotation=20, fontsize=20, color=style.GRID, zorder=0)",
          "    return fig"]
    return L


# ── drawing ─────────────────────────────────────────────────────────────────
def module_of(source: str, name: str = "chart"):
    """The script's source as a module, without running its main()."""
    style_module()
    mod = types.ModuleType(name)
    mod.__file__ = str(FIGURES / f"{name}.py")
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    return mod


def data_expr(folder: Path, table: str) -> str:
    style = style_module()
    if Path(folder).resolve() == Path(style.RESULTS_DIR).resolve():
        return f"style.RESULTS_DIR / {table!r}"
    return f"Path({str(Path(folder) / table)!r})"


def make(spec: dict, folder: Path, simulated: bool = False, style_dir: Path | None = None,
         pictures: Path | None = None) -> tuple[dict, str]:
    """(the checked spec, the script's source) for the table in folder."""
    import pandas as pd
    s = check(spec, folder)
    data = pd.read_csv(Path(folder) / s["data"])
    if not len(data[data[s["y"]].notna()]):
        raise ChartError("no_rows", s["data"])
    return s, script(s, data, data_expr(folder, s["data"]), simulated, style_dir, pictures)


def figure(source: str, path: Path):
    """The figure the source draws from the table at path (pyplot held for the call)."""
    import pandas as pd
    mod = module_of(source)
    return mod.draw(pd.read_csv(path)), mod


def preview(spec: dict, folder: Path, simulated: bool = False) -> dict:
    """The chart as the page shows it: {"spec", "svg" (data URL), "width_mm", "height_mm", "rows"}."""
    import matplotlib.pyplot as plt
    s, source = make(spec, folder, simulated)
    with _LOCK:
        fig, _ = figure(source, Path(folder) / s["data"])
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", metadata={"Date": None})
        plt.close(fig)
    return {"spec": s, "svg": "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "width_mm": s["width"], "height_mm": s["height"], "source": source}


def export(spec: dict, folder: Path, stem: Path, simulated: bool = False) -> list[Path]:
    """The chart written as <stem>.pdf and .svg (vector, fonts embedded in the PDF, text kept as text in the SVG)
    and <stem>.png at 300 dpi; the files written."""
    import matplotlib.pyplot as plt
    style = style_module()
    s, source = make(spec, folder, simulated)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    keywords = {"Keywords": style.PLACEHOLDER_TAG} if simulated else {}
    out = []
    with _LOCK:
        fig, _ = figure(source, Path(folder) / s["data"])
        for suffix, kw in ((".pdf", {"metadata": {"CreationDate": None, **keywords}}),
                           (".svg", {"metadata": {"Date": None}}),
                           (".png", {"dpi": style.DPI, "metadata": keywords})):
            path = stem.with_name(stem.name + suffix)
            fig.savefig(path, format=suffix[1:], **kw)
            out.append(path)
        plt.close(fig)
    return out


def saved(figures: Path) -> list[dict]:
    """The pinned charts of a figures folder: [{"name", "spec"}], by name."""
    out = []
    for p in sorted(Path(figures).glob("*.py")) if Path(figures).is_dir() else []:
        try:
            first = p.read_text(encoding="utf-8").split("\n", 1)[0]
        except OSError:
            continue
        if first.startswith(SPEC_LINE):
            try:
                out.append({"name": p.stem, "spec": json.loads(first[len(SPEC_LINE):])})
            except ValueError:
                continue
    return out


def next_name(figures: Path, pictures: Path, slug: str) -> str:
    """A free supplementary name for a new chart: figS<n>_<slug>, n one more than any figure there."""
    n = 0
    for folder in (figures, pictures):
        for p in Path(folder).glob("figS*") if Path(folder).is_dir() else []:
            m = re.match(r"figS(\d+)_", p.name)
            if m:
                n = max(n, int(m.group(1)))
    slug = re.sub(r"[^a-z0-9_]", "_", slug.lower()).strip("_") or "chart"
    return f"figS{n + 1}_{slug}"


def pin(spec: dict, folder: Path, figures: Path, pictures: Path, simulated: bool = False,
        replace: bool = False) -> dict:
    """The chart's script written into figures as <name>.py and run there in a process of its own, which writes
    the figure (PDF and PNG) into pictures. A script already there is replaced only with replace (ChartError
    script_exists). {"name", "script", "files", "replaced"}."""
    style = style_module()
    s = check(spec, folder)
    if not s["name"]:
        raise ChartError("bad_spec", "name", field="name")
    figures, pictures = Path(figures), Path(pictures)
    outside = figures.resolve() != FIGURES.resolve()
    s, source = make(s, folder, simulated, FIGURES if outside else None,
                     pictures if pictures.resolve() != Path(style.PICTURES_DIR).resolve() else None)
    target = figures / f"{s['name']}.py"
    existed = target.exists()
    if existed and not replace:
        raise ChartError("script_exists", s["name"], name=s["name"])
    figures.mkdir(parents=True, exist_ok=True)
    pictures.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(source, encoding="utf-8", newline="\n")
    os.replace(tmp, target)
    files = run_script(target)
    return {"name": s["name"], "script": str(target), "files": [str(p) for p in files], "replaced": existed,
            "spec": s}


def run_script(path: Path, pictures: Path | None = None) -> list[Path]:
    """Runs a figure script in a process of its own (no bytecode written beside it); the figure files it says it
    wrote or found unchanged. ChartError script_failed when it fails."""
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if sys.pycache_prefix:
        env["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        exe = exe[:-len("pythonw.exe")] + "python.exe"
    try:
        r = subprocess.run([exe, "-B", str(path)], cwd=str(Path(path).parent), env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=RUN_TIMEOUT_S,
                           creationflags=0x08000000 if os.name == "nt" else 0)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ChartError("script_failed", str(e)) from e
    if r.returncode != 0:
        raise ChartError("script_failed", (r.stderr or r.stdout)[-2000:])
    names = re.findall(r"^(\S+\.(?:pdf|png)): (?:written|unchanged)$", r.stdout, re.MULTILINE)
    return [Path(n) for n in names]


def presets() -> list[dict]:
    out = []
    for key, p in PRESETS:
        s = copy.deepcopy(DEFAULT)
        s.update(p)
        s["height"] = p.get("height", DEFAULT_HEIGHT[s["width"]])
        out.append({"key": key, "spec": s})
    return out


# ── tests ───────────────────────────────────────────────────────────────────
def selftest() -> int:
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)
    import analysis
    work = Path(tempfile.mkdtemp(prefix="charts_test_"))
    try:
        res = work / "results"
        analysis.analyse(analysis._run_in(work), None, work / "cache", res, simulate=True,
                         interfaces_raw=work / "raw" / "interfaces")
        t = tables(res)
        expect({"trials_tre.csv", "trials_metrics.csv", "trials_targets.csv", "interfaces_a.csv"} <= set(t)
               and "condition" in t["trials_tre.csv"]["levels"] and "tre_mm" in t["trials_tre.csv"]["values"]
               and "impact.csv" not in t, "the tables a chart reads, with their levels and values (key tables left out)")
        import matplotlib
        drawn = []
        for p in presets():
            spec = dict(p["spec"], name=f"figS9{len(drawn)}_{p['key']}")
            try:
                v = preview(spec, res, simulated=True)
                ok = v["svg"].startswith("data:image/svg+xml;base64,") and len(v["svg"]) > 2000
            except ChartError as e:
                ok = False
                print("     ", p["key"], e.code, e)
            drawn.append((p["key"], ok))
        expect(all(ok for _, ok in drawn), "every preset drawn: " + ", ".join(k for k, ok in drawn if ok)
               + ("; failed " + ", ".join(k for k, ok in drawn if not ok) if not all(ok for _, ok in drawn) else ""))
        for kind in KINDS:
            for palette in PALETTES:
                spec = dict(DEFAULT, kind=kind, palette=palette, series="holder", x="energy_level" if kind != "cumulative"
                            else "", scale="linear", y_ticks=None)
                try:
                    preview(spec, res)
                except ChartError as e:
                    expect(False, f"{kind} with palette {palette}: {e.code} {e}")
        files = export(dict(presets()[0]["spec"], name="figS1_tre"), res, work / "out" / "figS1_tre", simulated=True)
        pdf = files[0].read_bytes()
        expect([f.suffix for f in files] == [".pdf", ".svg", ".png"] and b"Abadi" in pdf and b"FontFile2" in pdf
               and b"<svg" in files[1].read_bytes()[:400] and files[2].stat().st_size > 10000,
               "export: PDF with Abadi embedded, SVG, PNG at 300 dpi")
        from PIL import Image
        with Image.open(files[2]) as im:
            dpi = im.info.get("dpi", (0, 0))[0]
            w_mm = im.size[0] / dpi * 25.4 if dpi else 0
        expect(abs(dpi - 300) < 1 and abs(w_mm - 90) < 0.5, f"PNG at {dpi:.0f} dpi, {w_mm:.1f} mm wide")
        figs, pics = work / "figures", work / "pictures"
        out = pin(dict(presets()[0]["spec"], name="figS2_tre"), res, figs, pics, simulated=True)
        expect(Path(out["script"]).is_file() and (pics / "figS2_tre.pdf").is_file() and (pics / "figS2_tre.png").is_file()
               and not out["replaced"] and not list(figs.glob("__pycache__")),
               "pinned: the script written and run on its own, PDF and PNG made, no bytecode beside it")
        try:
            pin(dict(presets()[0]["spec"], name="figS2_tre"), res, figs, pics, simulated=True)
            expect(False, "an existing script is never replaced without asking")
        except ChartError as e:
            expect(e.code == "script_exists", "an existing script is never replaced without asking")
        again = pin(dict(presets()[1]["spec"], name="figS2_tre"), res, figs, pics, simulated=True, replace=True)
        back = {c["name"]: c["spec"] for c in saved(figs)}
        expect(again["replaced"] and back.get("figS2_tre", {}).get("x") == "direction_deg",
               "replaced when asked; the pinned chart opens again from its script")
        meta = (pics / "figS2_tre.pdf").read_bytes()
        expect(b"placeholder data" in meta, "a chart of simulated tables is marked as placeholder data")
        expect(next_name(figs, pics, "TRE by energy") == "figS3_tre_by_energy", "the next free name")
        expect(tuple(style_module().WIDTHS_MM) == WIDTHS_MM, "the page widths are style.py's")
        for bad, code in (({"kind": "pie"}, "bad_spec"), ({"width": 120}, "bad_spec"), ({"data": "nope.csv"}, "no_table"),
                          ({"y": "nope"}, "no_column"), ({"name": "Fig 1"}, "bad_spec"), ({"limits": "1,2"}, "bad_spec"),
                          ({"data": "../x.csv"}, "bad_spec")):
            try:
                check(dict(DEFAULT, **bad), res)
                expect(False, f"refused: {bad}")
            except ChartError as e:
                expect(e.code == code, f"refused as {code}: {bad}")
        matplotlib.pyplot.close("all")
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    print(f"{'FAILED: ' + str(len(fails)) if fails else 'all passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(selftest())
    print(__doc__)
