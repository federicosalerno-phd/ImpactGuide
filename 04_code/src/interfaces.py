"""The two marker holders measured on their own, before the impacts: the slip torque of A and the reseating of B.

A, slip torque. Calibrated masses hung from the arm of holder A at a known distance from the centre of its ball,
one more at a time until the joint slips: the torque of the slip is m g r. Every mass tried is a row of
<raw>/interfaces/a_slip_torque.csv (A_COLUMNS: repeat, mass_g, arm_mm the centre of the ball to the hook, slipped 0
or 1, time, notes), written by the app one press at a time (record_a) or by hand; a repeat ends at its first slip,
and its slip torque is the one of that mass. A file without the column slipped holds one row per repeat, the mass
that slipped.

B, reseating. The marker of holder B lifted from its seats and let go, then scanned, RESEATS times after a first
seating: a run without an impact of the scan desk (scan_desk.Capture, kind reseat: the scans in
<raw>/ground_truth/reseating, each measured as it comes, in <processed>/ground_truth/reseat.json). The first good
scan is the reference; every later one gives the rotation, the translation and the TRE at the seven targets that a
registration made at the reference would carry (mm).

Tables, into the results folder, only when their content changes:
    interfaces_a.csv     one row per repeat: repeat, mass_g, arm_mm, torque_nm
    interfaces_b.csv     one row per seating after the reference: seating, rot_deg, shift_mm, tre_mean_mm, tre_max_mm
    interfaces.csv       key table: a_torque (n, mean, SD, bootstrap interval of the mean, min, max) and b_reseat
                         (n, TRE median, mean, 95th percentile and max, rotation and shift)

With simulate=True the parts not measured yet are simulated (simulated: True in the summary): ten repeats of a
torque near SIM_TORQUE_NM, thirty seatings of B within twin's B_MM and B_DEG.

    python interfaces.py simulate --out DIR      the tables from simulated measurements
    python interfaces.py run [--out DIR]         the tables from the measurements in 03_data
    python interfaces.py test                    checks; exit code 1 on failure
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

import stats
import twin

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "03_data" / "raw" / "interfaces"
STORE = ROOT / "03_data" / "processed" / "ground_truth"      # scan_desk's store: reseat.json
G = 9.81
A_FILE = "a_slip_torque.csv"
A_COLUMNS = ("repeat", "mass_g", "arm_mm", "slipped", "time", "notes")
B_RUN = "reseat.json"
A_REPEATS = 10              # repeats of the slip torque of A
RESEATS = 30                # lifts of B after its first seating
SIM_TORQUE_NM, SIM_TORQUE_SD = 0.209, 0.012     # simulated A (EXPERIMENT: about 0.2 N m)
SIM_ARM_MM = 60.0
SIM_REPEATS, SIM_SEATINGS = A_REPEATS, RESEATS
SIM_BOARD_MANDIBLE = (0.0, 15.74, -187.10)       # the board's centre in the mandible frame (the bench as exported)


def torque_nm(mass_g: float, arm_mm: float) -> float:
    return mass_g / 1000 * G * arm_mm / 1000


def _num(v) -> float:
    return float(str(v).replace(",", "."))


def read_steps(folder: Path) -> list[dict]:
    """Every row of a_slip_torque.csv as written ([] when the file is missing); a row whose numbers cannot be read
    is left out. A file without the column slipped: every row a slip."""
    path = Path(folder) / A_FILE
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    delim = ";" if lines and ";" in lines[0] else ","
    out = []
    for r in csv.DictReader(io.StringIO(text), delimiter=delim):
        try:
            m, a = _num(r["mass_g"]), _num(r["arm_mm"])
            rep = int(float(r.get("repeat") or len(out) + 1))
        except (KeyError, TypeError, ValueError):
            continue
        sl = r.get("slipped")
        slipped = 1 if sl is None else int(str(sl).strip().lower() in ("1", "1.0", "yes", "true"))
        out.append({"repeat": rep, "mass_g": m, "arm_mm": a, "slipped": slipped, "time": r.get("time") or "",
                    "notes": r.get("notes") or ""})
    return out


def first_slips(rows: list[dict]) -> list[dict]:
    """The repeats done: the first slip of each, with its torque."""
    out, seen = [], set()
    for r in rows:
        if r["slipped"] and r["repeat"] not in seen:
            seen.add(r["repeat"])
            out.append({"repeat": r["repeat"], "mass_g": r["mass_g"], "arm_mm": r["arm_mm"],
                        "torque_nm": torque_nm(r["mass_g"], r["arm_mm"])})
    return out


def read_a(folder: Path) -> list[dict]:
    """The repeats of the slip torque of A: the first slip of each repeat ([] when there is none)."""
    return first_slips(read_steps(folder))


def write_steps(folder: Path, rows: list[dict]) -> None:
    from analysis import csv_text, write_if_changed
    write_if_changed(Path(folder) / A_FILE, csv_text(A_COLUMNS, rows))


def a_state(folder: Path, rows: list[dict] | None = None) -> dict:
    """The slip torque of A as it stands: the repeat in progress, its masses so far (the largest held), the repeats
    done with their torque, mean and SD."""
    rows = read_steps(folder) if rows is None else rows
    done = first_slips(rows)
    rep = max([r["repeat"] for r in rows], default=0)
    if not rows or any(r["repeat"] == rep and r["slipped"] for r in rows):
        rep += 1
    cur = [r for r in rows if r["repeat"] == rep]
    t = np.array([d["torque_nm"] for d in done])
    last = rows[-1] if rows else None
    return {"repeat": rep, "of": A_REPEATS, "tried": [{"mass_g": r["mass_g"], "slipped": r["slipped"]} for r in cur],
            "held_g": max([r["mass_g"] for r in cur if not r["slipped"]], default=None),
            "arm_mm": last["arm_mm"] if last else None, "last_mass_g": last["mass_g"] if last else None,
            "done": [{"repeat": d["repeat"], "mass_g": d["mass_g"], "torque_nm": round(d["torque_nm"], 5)}
                     for d in done],
            "n": int(len(t)), "mean_nm": round(float(t.mean()), 5) if len(t) else None,
            "sd_nm": round(float(t.std(ddof=1)), 5) if len(t) > 1 else None, "complete": len(t) >= A_REPEATS}


def record_a(folder: Path, mass_g: float, arm_mm: float, slipped: bool, when: datetime | None = None) -> dict:
    """One mass tried on A: a row of the repeat in progress (the next one after a slip), written at once; the new
    a_state. ValueError on a mass or an arm that is not a plausible positive number."""
    mass_g, arm_mm = float(mass_g), float(arm_mm)
    if not (math.isfinite(mass_g) and 0 < mass_g <= 20000 and math.isfinite(arm_mm) and 5 <= arm_mm <= 500):
        raise ValueError("mass or arm")
    rows = read_steps(folder)
    rep = a_state(folder, rows)["repeat"]
    rows.append({"repeat": rep, "mass_g": round(mass_g, 2), "arm_mm": round(arm_mm, 2), "slipped": int(bool(slipped)),
                 "time": (when or datetime.now()).astimezone().isoformat(timespec="seconds"), "notes": ""})
    write_steps(folder, rows)
    return a_state(folder)


def undo_a(folder: Path) -> dict:
    """The last mass tried taken back (a slip as well: its repeat goes on)."""
    rows = read_steps(folder)
    if rows:
        write_steps(folder, rows[:-1])
    return a_state(folder)


def simulated_a() -> list[dict]:
    rng = np.random.default_rng(81)
    out = []
    for i in range(SIM_REPEATS):
        t = SIM_TORQUE_NM + rng.normal(0, SIM_TORQUE_SD)
        m = round(t / (G * SIM_ARM_MM / 1000) * 1000, 1)        # the masses come in tenths of a gram
        out.append({"repeat": i + 1, "mass_g": m, "arm_mm": SIM_ARM_MM, "torque_nm": torque_nm(m, SIM_ARM_MM)})
    return out


def read_b(store: Path) -> list[dict]:
    """Every good seating of the reseating run after the first one, against it ([] with fewer than two)."""
    import ground_truth as gt
    try:
        d = json.loads((Path(store) / B_RUN).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    good = [f for f in d.get("files") or [] if f.get("ok") and f.get("T_M_mandible")]
    if len(good) < 2:
        return []
    targets = gt.load_targets()
    ref = np.asarray(good[0]["T_M_mandible"], np.float64)
    out = []
    for k, f in enumerate(good[1:], 1):
        r = gt.displacement(ref, np.asarray(f["T_M_mandible"], np.float64), targets)
        out.append({"seating": k, "rot_deg": r["rot_deg"], "shift_mm": float(np.linalg.norm(r["trans_mm"])),
                    "tre_mean_mm": r["tre_mean_mm"], "tre_max_mm": r["tre_max_mm"]})
    return out


def b_state(store: Path) -> dict:
    """The reseating of B as it stands: seatings measured after the first, the median, 95th percentile and largest
    TRE against the first."""
    rows = read_b(store)
    e = np.array([r["tre_mean_mm"] for r in rows])
    return {"n": len(rows), "of": RESEATS, "median_mm": round(float(np.median(e)), 4) if len(e) else None,
            "p95_mm": round(float(np.quantile(e, 0.95)), 4) if len(e) else None,
            "max_mm": round(float(max(r["tre_max_mm"] for r in rows)), 4) if rows else None,
            "last_mm": round(float(e[-1]), 4) if len(e) else None, "complete": len(rows) >= RESEATS}


def simulated_b() -> list[dict]:
    """Thirty seatings of B: a turn of twin.B_DEG and a shift of twin.B_MM about the board's centre."""
    import ground_truth as gt
    targets = gt.load_targets()
    P = np.array(list(targets.values()), np.float64)
    c = np.asarray(SIM_BOARD_MANDIBLE, np.float64)
    rng = np.random.default_rng(82)
    out = []
    for k in range(1, SIM_SEATINGS + 1):
        axis = rng.normal(size=3)
        R = twin.rot(axis, rng.uniform(*twin.B_DEG))
        s = rng.normal(size=3)
        s *= rng.uniform(*twin.B_MM) / np.linalg.norm(s)
        t = c - R @ c + s
        tre = np.linalg.norm(P @ R.T + t - P, axis=1)
        out.append({"seating": k, "rot_deg": float(np.degrees(math.acos(min(1.0, (np.trace(R) - 1) / 2)))),
                    "shift_mm": float(np.linalg.norm(t)), "tre_mean_mm": float(tre.mean()),
                    "tre_max_mm": float(tre.max())})
    return out


def rnd(x, n=4):
    return "" if x is None or not math.isfinite(float(x)) else round(float(x), n)


def key_rows(a: list[dict], b: list[dict]) -> list[dict]:
    rows = []
    if a:
        t = np.array([r["torque_nm"] for r in a])
        lo, hi = stats.bootstrap(t, stat=np.mean, seed=83)
        rows.append({"key": "a_torque", "n": len(t), "mean": rnd(t.mean()), "sd": rnd(t.std(ddof=1) if len(t) > 1
                                                                                       else math.nan),
                     "ci_low": rnd(lo), "ci_high": rnd(hi), "min": rnd(t.min()), "max": rnd(t.max()), "unit": "N m"})
    if b:
        e = np.array([r["tre_mean_mm"] for r in b])
        rows.append({"key": "b_reseat", "n": len(e), "mean": rnd(e.mean()), "sd": rnd(e.std(ddof=1)),
                     "median": rnd(np.median(e)), "p95": rnd(np.quantile(e, 0.95)),
                     "max": rnd(max(r["tre_max_mm"] for r in b)),
                     "rot_median_deg": rnd(np.median([r["rot_deg"] for r in b])),
                     "shift_median_mm": rnd(np.median([r["shift_mm"] for r in b])), "unit": "mm"})
    return rows


def analyse(raw: Path, results: Path, simulate: bool = False, store: Path | None = None) -> dict | None:
    """The tables of the two holders written into results (only the changed ones), and a summary for the page:
    {"a": {...} or None, "b": {...} or None, "simulated", "written"}; None when there is nothing at all. raw: the
    folder of a_slip_torque.csv; store: scan_desk's store, with the reseating run (STORE by default)."""
    from analysis import csv_text, write_if_changed
    a, b = read_a(raw), read_b(STORE if store is None else store)
    sim = {"a": False, "b": False}
    if simulate and not a:
        a, sim["a"] = simulated_a(), True
    if simulate and not b:
        b, sim["b"] = simulated_b(), True
    if not a and not b:
        return None
    keyed = key_rows(a, b)
    made = {}
    if a:
        made["interfaces_a.csv"] = (("repeat", "mass_g", "arm_mm", "torque_nm"),
                                    [{**r, "torque_nm": rnd(r["torque_nm"], 5)} for r in a])
    if b:
        made["interfaces_b.csv"] = (("seating", "rot_deg", "shift_mm", "tre_mean_mm", "tre_max_mm"),
                                    [{k: (v if k == "seating" else rnd(v, 5)) for k, v in r.items()} for r in b])
    cols = ["key"]
    for r in keyed:
        cols += [c for c in r if c not in cols]
    made["interfaces.csv"] = (tuple(cols), keyed)
    written = [n for n, (c, rows) in made.items() if write_if_changed(Path(results) / n, csv_text(c, rows))]
    by = {r["key"]: r for r in keyed}
    return {"a": by.get("a_torque"), "b": by.get("b_reseat"), "simulated": sim["a"] or sim["b"],
            "simulated_parts": [k for k, v in sim.items() if v], "written": written,
            "a_values": [rnd(r["torque_nm"], 5) for r in a], "b_values": [rnd(r["tre_mean_mm"], 5) for r in b]}


def selftest() -> int:
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)
    work = Path(tempfile.mkdtemp(prefix="interfaces_test_"))
    try:
        s = analyse(work / "raw", work / "out", simulate=True, store=work / "store")
        expect(s["simulated"] and s["a"]["n"] == SIM_REPEATS and s["b"]["n"] == SIM_SEATINGS
               and abs(s["a"]["mean"] - SIM_TORQUE_NM) < 0.02 and s["b"]["p95"] < 0.3,
               f"simulated: A {s['a']['mean']} N m over {s['a']['n']} repeats, B TRE 95th percentile "
               f"{s['b']['p95']} mm over {s['b']['n']} seatings")
        with open(work / "out" / "interfaces.csv", encoding="utf-8") as f:
            head = f.readline().strip().split(",")
        expect(head[0] == "key" and analyse(work / "raw", work / "out", simulate=True,
                                            store=work / "store")["written"] == [],
               "interfaces.csv is a key table; the same data write nothing the second time")
        (work / "raw").mkdir()
        (work / "raw" / A_FILE).write_text("repeat;mass_g;arm_mm;notes\n1;350,5;60;\n2;361;60;\n3;bad;60;\n",
                                           encoding="utf-8")
        s2 = analyse(work / "raw", work / "out2", simulate=True, store=work / "store")
        expect(s2["a"]["n"] == 2 and abs(s2["a"]["mean"] - torque_nm(355.75, 60)) < 1e-4
               and s2["simulated_parts"] == ["b"],
               "the measured repeats of A are read (semicolons, decimal commas, a bad row left out); B still simulated")
        expect(analyse(work / "none", work / "out3", store=work / "store") is None
               and not (work / "out3").exists(), "nothing measured and no simulation: nothing written")
        # the guided steps of A: masses one at a time, a slip ends the repeat, the last press taken back
        g = work / "guided"
        st = record_a(g, 300, 60, False)
        st = record_a(g, 340, 60, False)
        expect(st["repeat"] == 1 and st["held_g"] == 340 and st["n"] == 0, "A: masses held, the repeat goes on")
        st = record_a(g, 360, 60, True)
        expect(st["repeat"] == 2 and st["n"] == 1 and abs(st["mean_nm"] - torque_nm(360, 60)) < 1e-4,
               f"A: a slip ends the repeat, its torque {st['mean_nm']} N m")
        st = undo_a(g)
        expect(st["repeat"] == 1 and st["n"] == 0 and len(read_steps(g)) == 2, "A: the last press taken back")
        record_a(g, 360, 60, True)
        record_a(g, 350, 60, True)
        expect(len(read_a(g)) == 2 and read_steps(g)[-1]["repeat"] == 2, "A: the next repeat after a slip")
        try:
            record_a(g, -1, 60, False)
            expect(False, "A: a mass that is not positive refused")
        except ValueError:
            expect(len(read_steps(g)) == 4, "A: a mass that is not positive refused, nothing written")
        # B from the reseating run of the scan desk: the first good scan the reference
        import ground_truth as gt
        T0 = np.eye(4)
        files = [{"name": "bre_001.ply", "ok": True, "T_M_mandible": T0.tolist()},
                 {"name": "bre_002.ply", "ok": False}]
        for k in range(3):
            T = np.eye(4)
            T[:3, 3] = [0.02 * (k + 1), 0, 0]
            files.append({"name": f"bre_{k + 3:03d}.ply", "ok": True, "T_M_mandible": T.tolist()})
        (work / "store").mkdir(exist_ok=True)
        (work / "store" / B_RUN).write_text(json.dumps({"kind": "reseat", "files": files}), encoding="utf-8")
        b = read_b(work / "store")
        bs = b_state(work / "store")
        expect(len(b) == 3 and abs(b[0]["tre_mean_mm"] - 0.02) < 1e-9 and bs["n"] == 3
               and abs(bs["max_mm"] - 0.06) < 1e-9 and gt.load_targets(),
               f"B: the seatings after the first against it, a bad scan left out (median {bs['median_mm']} mm)")
        expect(abs(torque_nm(355.0, 60) - 0.209) < 0.001, "torque of 355 g at 60 mm: 0.209 N m")
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    print(f"{'FAILED: ' + str(len(fails)) if fails else 'all passed'}")
    return 1 if fails else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "simulate"):
        q = sub.add_parser(name)
        q.add_argument("--out", type=Path, default=None)
    sub.add_parser("test")
    a = p.parse_args(argv)
    if a.cmd == "test":
        return selftest()
    if a.cmd == "simulate" and a.out is None:
        p.error("simulate needs --out (simulated tables never go into the results folder)")
    s = analyse(RAW, a.out or ROOT / "05_results", simulate=a.cmd == "simulate")
    print("nothing measured yet" if s is None else f"written: {', '.join(s['written']) or 'nothing changed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
