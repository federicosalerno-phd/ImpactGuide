"""The 3D-printed ChArUco plate: the physical calibration board of ImpactGuide (T5.1).

The plate (04_code/tools/build_charuco_plate.py; STL, 3MF and JSON in 02_cad/stl) carries the A4 board of
calibrate_charuco.py at 20 mm instead of 24: 11 x 8 squares, markers 3/4 of the square, DICT_5X5_100, the same marker
ids in the same squares. Its nominal sizes and what to measure are in charuco_plate_11x8_20mm.json, read here.

How the calibration treats it:
1. Measure (once per printed plate, at room temperature): with a caliper, the outer faces of the two black ticks above
   the board (nominal 220.0 mm, 11 squares), three readings. reading() gives their mean and spread and says whether
   they can be used (MIN_READINGS, SPREAD_MAX, within NOMINAL_TOL of the nominal length).
2. The board is spec(long_mm): squares, square_mm = long_mm / 11, marker_mm = 3/4 of it, dictionary; the same keys as
   calibrate_charuco.board_spec and board.json, plus kind "plate", so calibrate_charuco.folder_board, calibrate,
   check and live_ar use it as they use the A4 sheet. make_board(long_mm) is the cv2 board.
3. Optional check: the two ticks right of the board (nominal 160.0 mm, 8 squares). Their scale against the long one
   tells whether the printer's X and Y agree (ANISOTROPY_MAX); with it, spec() takes the mean of the two scales.
The plate has the A4 sheet's layout, so the app's printed-board path (square = measured long side / 11, marker 3/4)
already gives this board when the plate's 220 mm reading is typed in; this module adds its nominal values, the
reading's checks and the kind.

Usage:
    python charuco_plate.py info                         the plate and what to measure
    python charuco_plate.py spec 220.05 220.02 220.08 [--short 160.01 160.04 159.99]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import calibrate_charuco

ROOT = Path(__file__).resolve().parents[2]
PLATE_FILE = ROOT / "02_cad" / "stl" / "charuco_plate_11x8_20mm.json"
KIND = "plate"
MIN_READINGS = 3
SPREAD_MAX = 0.10       # mm between the largest and the smallest reading: more means a jaw not square or not home
NOMINAL_TOL = 0.01      # a mean more than 1 % off the nominal length is a misreading (a printer is within 0.5 %)
ANISOTROPY_MAX = 0.0015     # X and Y scales of the print may differ by 0.15 % at most


def load(path=PLATE_FILE):
    """The plate's JSON; its board must be the A4 board of calibrate_charuco at the plate's square."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    b = d["board"]
    if b["dictionary"] != calibrate_charuco.BOARD["dictionary"] or tuple(b["squares"]) != tuple(
            calibrate_charuco.BOARD["squares"]) or abs(b["marker_mm"] / b["square_mm"] - calibrate_charuco.BOARD[
            "marker"] / calibrate_charuco.BOARD["square"]) > 1e-9:
        raise ValueError(f"{path}: the plate is not the calibration board of calibrate_charuco.py")
    return d


def reading(values_mm, which="long", plate=None):
    """Caliper readings of one length of the plate ("long": the ticks above the board, "short": right of it).
    Returns mean_mm, spread_mm, n, nominal_mm, scale (mean / nominal) and problems: a list of
    "few_readings", "spread", "off_nominal" (empty when the mean can be used)."""
    plate = plate or load()
    nominal = float(plate["measure"][which]["nominal_mm"])
    v = np.asarray([float(x) for x in values_mm], float)
    if not len(v):
        raise ValueError("no readings")
    mean, spread = float(v.mean()), float(v.max() - v.min())
    problems = []
    if len(v) < MIN_READINGS:
        problems.append("few_readings")
    if spread > SPREAD_MAX:
        problems.append("spread")
    if abs(mean / nominal - 1) > NOMINAL_TOL:
        problems.append("off_nominal")
    return {"which": which, "mean_mm": round(mean, 3), "spread_mm": round(spread, 3), "n": int(len(v)),
            "nominal_mm": nominal, "scale": mean / nominal, "problems": problems}


def spec(long_mm=None, short_mm=None, plate=None):
    """The plate's board for calibrate_charuco and live_ar (board.json keys, kind "plate"). long_mm, short_mm: the mean
    readings (None: nominal). With short_mm the square follows the mean of the two scales, and anisotropy (short scale
    / long scale - 1) and anisotropy_ok are given."""
    plate = plate or load()
    b, m = plate["board"], plate["measure"]
    s_long = float(long_mm) / m["long"]["nominal_mm"] if long_mm is not None else 1.0
    scale = s_long
    out = {"kind": KIND, "name": plate["name"], "squares": list(b["squares"]), "dictionary": b["dictionary"],
           "nominal_square_mm": b["square_mm"], "measured_long_mm": long_mm}
    if short_mm is not None:
        s_short = float(short_mm) / m["short"]["nominal_mm"]
        out["measured_short_mm"] = short_mm
        out["anisotropy"] = round(s_short / s_long - 1, 6)
        out["anisotropy_ok"] = abs(out["anisotropy"]) <= ANISOTROPY_MAX
        scale = (s_long + s_short) / 2
    out["square_mm"] = round(b["square_mm"] * scale, 6)
    out["marker_mm"] = round(b["marker_mm"] * scale, 6)
    out["scale"] = round(scale, 6)
    return out


def make_board(long_mm=None, short_mm=None, plate=None):
    """The cv2 CharucoBoard of the plate at its measured size (nominal without readings)."""
    s = spec(long_mm, short_mm, plate)
    return calibrate_charuco.make_board(s["square_mm"], s["squares"], s["marker_mm"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    a = sub.add_parser("spec")
    a.add_argument("long", type=float, nargs="+", help="caliper readings of the ticks above the board, mm")
    a.add_argument("--short", type=float, nargs="+", help="caliper readings of the ticks right of the board, mm")
    args = ap.parse_args()
    plate = load()
    if args.cmd == "info":
        b, m = plate["board"], plate["measure"]
        print(f"{plate['name']}: {b['squares'][0]} x {b['squares'][1]} squares of {b['square_mm']:g} mm, markers "
              f"{b['marker_mm']:g} mm, {b['dictionary']}")
        for k in ("long", "short"):
            print(f"  {k}: {m[k]['nominal_mm']:.1f} mm between the {m[k]['between']}")
        print(f"  {m['tool']}; at least {MIN_READINGS} readings, spread within {SPREAD_MAX} mm")
        return 0
    r = reading(args.long, "long", plate)
    print(f"long: mean {r['mean_mm']:.3f} mm, spread {r['spread_mm']:.3f} mm over {r['n']} "
          f"({100 * (r['scale'] - 1):+.3f} % of {r['nominal_mm']:.1f})" + (f"  {r['problems']}" if r["problems"] else ""))
    short = None
    if args.short:
        q = reading(args.short, "short", plate)
        short = q["mean_mm"]
        print(f"short: mean {q['mean_mm']:.3f} mm, spread {q['spread_mm']:.3f} mm over {q['n']} "
              f"({100 * (q['scale'] - 1):+.3f} % of {q['nominal_mm']:.1f})" + (f"  {q['problems']}" if q["problems"]
                                                                             else ""))
    s = spec(r["mean_mm"], short, plate)
    print(json.dumps(s, indent=1))
    return 0 if not r["problems"] and s.get("anisotropy_ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
