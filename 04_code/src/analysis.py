"""Analysis of the impact trials: the metrics of each trial, the four conditions, the statistics and the tables.

Trials. A trial counts once it is closed (status ok) and its post-impact scans give the marker's motion
(twin.scan_result: measured with ground_truth.measure and kept in the twin's cache): the true displacement of the
marker (rotation in degrees, shift of the board's centre in mm) and the TRE that the registration of before the
impact leaves at the seven targets of the mandible (mm; the trial's TRE is their mean). The impact energy comes from
the hole: E = M g d (1 - cos theta), with M the logged swinging mass and d the distance from the axis to its centre of
mass (PENDULUM_COM_MM); the energy level is the rank of the hole in the configuration (E1 the lowest).

Conditions. Each trial of holder A gives A (no correction) and A_AI (after the AI step), each trial of holder B gives
B and B_AI. The AI step is read from trials_ai.csv in the results folder, one row per trial and view input, as the
evaluation of the AI writes it (AI_COLUMNS): score, tau, detected, applied, warning, tre_ai_mm (the TRE after the
step: the corrected one when the correction was applied, else the uncorrected one), tre_pred95_mm, time_s. Without
it there are only A and B, and detection and correction wait. The registration is lost when the mean TRE without
correction is over LOST_MM. Drape trials are reported apart: the tables of the matrix leave them out.

Statistics. Wilcoxon signed-rank tests, adjusted with Holm's method as one family: A against A_AI and B against B_AI
on the same trials; A against B, A_AI against B and A_AI against B_AI on the same cells (hole, direction and
repetition). A linear mixed model TRE ~ energy * interface + direction + (1 | session) on the TRE without correction
(statsmodels; ordinary least squares while there is one session only). Bootstrap intervals over trials (BOOTSTRAP
resamples) for the medians, the AUC, sensitivity and specificity. Detection: the AUC of the score against the
registration lost, sensitivity and specificity at tau. Correction, over the lost registrations: the share under 1 and
2 mm after the AI step, the warnings, the corrections that made the TRE worse, and the safe outcome (TRE under 2 mm
after the step, or a warning).

Tables, written into the results folder only when their content changes (TABLES):
    trials_metrics.csv   one row per trial: setting, energy, displacement, TRE, the AI step
    trials_tre.csv       one row per trial and condition of the matrix: trial, energy_level, condition, tre_mm, ...
    trials_targets.csv   one row per trial and target: the TRE without correction at that target
    impact.csv           TRE per condition: overall, per energy level, per direction, with the drape
    impact_targets.csv   TRE at each target, per interface
    stats.csv            the paired tests
    mixed_model.csv      the terms of the mixed model
    detection.csv        with the AI step: AUC, sensitivity and specificity per view input
    correction.csv       with the AI step: the outcome of the correction per condition and view input
The last six are key tables: "key" is their first column, one row per group, values rounded as they are to be read.

Simulation. With simulate=True every slot of the sequence still to do becomes a simulated trial (twin.simulated_motion:
A turns about its ball, more for a higher hole; B comes back within a tenth of a millimetre), spread over SIM_DAYS
sessions, and every trial without an AI step gets a simulated one. Every simulated value is said so (column
simulated, summary["simulated"]).

    python analysis.py simulate --out DIR     the tables of a simulated campaign
    python analysis.py run [--phase campaign] [--out DIR]
    python analysis.py test                   checks on a simulated campaign; exit code 1 on failure
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import stats
import trials
import twin
from stats import bootstrap, holm, median_iqr

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "05_results"

G = 9.81
PENDULUM_COM_MM = 236.0     # axis to the centre of mass of the swinging part (model estimate until weighed and balanced)
NOMINAL_MASS_G = 137.0      # the swinging part when the log has no mass (model estimate)
LOST_MM = 2.0               # mean TRE without correction over this: registration lost
LIMITS_MM = (1.0, 2.0)      # the shares under these TRE
BOOTSTRAP = stats.BOOTSTRAP
SEED = stats.SEED
ALPHA = 0.05              # silent errors allowed by the threshold u (conformal risk control)
CONDITIONS = ("A", "A_AI", "B", "B_AI")
AI_FILE = "trials_ai.csv"
AI_COLUMNS = ("trial", "input", "score", "tau", "detected", "applied", "warning", "tre_ai_mm", "tre_corrected_mm",
              "tre_pred95_mm", "score_final", "time_s")
VALIDATION_FILE = "validation_ai.csv"
VALIDATION_COLUMNS = ("sample", "group", "score_final", "tau", "tre_pred95_mm", "tre_mm")
MAIN_INPUT = "4_views"      # the view input of the four conditions when the AI step has several
SIM_DAYS = 4                # sessions of a simulated campaign
SIM_FIRST_DAY = date(2026, 10, 29)
SIM_TAU, SIM_U = 0.5, 2.0   # simulated AI step: score threshold and limit of the predicted TRE (mm)
SIM_FAIL = 0.08             # simulated corrections that fail
SIM_VALIDATION = 300        # simulated validation samples
# The bench as last exported, for when the twin has no model: the head frame in the world and the centres the
# simulated motion turns about (twin's model extras); the targets come from ground_truth.load_targets()
FALLBACK_T_W_MANDIBLE = [[1.0, 0.0, 0.0, 0.0], [0.0, -0.04143, 0.999141, 96.364182],
                         [0.0, -0.999141, -0.04143, 16.660217], [0.0, 0.0, 0.0, 1.0]]
FALLBACK_BALL = [0.0008, -51.2314, -19.3163]
FALLBACK_BOARD = [0.0008, -91.2314, 8.6845]

METRIC_COLUMNS = ("trial", "phase", "session", "holder", "angle_deg", "energy_level", "energy_mj", "direction_deg",
                  "drape", "rot_deg", "shift_mm", "tre_mean_mm", "tre_max_mm", "lost", "input", "score", "detected",
                  "applied", "warning", "tre_ai_mm", "tre_corrected_mm", "tre_pred95_mm", "time_s", "simulated")
TRE_COLUMNS = ("trial", "energy_level", "condition", "tre_mm", "holder", "angle_deg", "energy_mj", "direction_deg",
               "session")
TARGET_COLUMNS = ("trial", "holder", "target", "tre_mm", "energy_level", "angle_deg", "direction_deg", "drape",
                  "session")
TABLES = ("trials_metrics.csv", "trials_tre.csv", "trials_targets.csv", "impact.csv", "impact_targets.csv",
          "stats.csv", "mixed_model.csv", "detection.csv", "correction.csv", "uncertainty.csv", "uncertainty_curve.csv")
PER_ROW = ("trials_metrics.csv", "trials_tre.csv", "trials_targets.csv", "uncertainty_curve.csv")   # not key tables


class AnalysisError(Exception):
    """A failure the interface can name: code no_trials (no closed trial with its scans)."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


# ── numbers ─────────────────────────────────────────────────────────────────
def num(v, default=math.nan) -> float:
    try:
        x = float(str(v).replace(",", "."))
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def flag(v) -> int:
    return 1 if str(v).strip().lower() in ("1", "1.0", "true", "yes") else 0


def rnd(x, n=3):
    """x rounded to n decimals for a table ('' when it is not a number)."""
    if x is None or isinstance(x, str):
        return x if isinstance(x, str) else ""
    if not math.isfinite(float(x)):
        return ""
    v = round(float(x), n)
    return int(v) if n == 0 else (0.0 if v == 0 else v)


def sig(x, n=3):
    """x with n significant digits, as text ('' when it is not a number)."""
    if x is None or not math.isfinite(float(x)):
        return ""
    return f"{float(x):.{n}g}"


def clean(x):
    """A value for the page: NaN and infinities as None, numpy scalars as Python ones."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def energy_mj(angle_deg: float, mass_g: float = NOMINAL_MASS_G, com_mm: float = PENDULUM_COM_MM) -> float:
    """The energy of the pendulum released from a hole: M g d (1 - cos theta), in mJ."""
    return mass_g / 1000 * G * com_mm / 1000 * (1 - math.cos(math.radians(angle_deg))) * 1000


def seed_of(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


# ── the trials ──────────────────────────────────────────────────────────────
def model_info(info: dict | None) -> dict:
    """The twin model's extras the analysis needs (targets, head frame, centres), or the fallback of the bench as
    last exported with the targets of ground_truth."""
    if info and info.get("targets"):
        return info
    import ground_truth as gt
    targets = [{"id": k, "name": k, "mandible": list(map(float, v))} for k, v in gt.load_targets().items()]
    return {"targets": targets, "T_W_mandible": FALLBACK_T_W_MANDIBLE, "ball_centre": FALLBACK_BALL,
            "board_centre": FALLBACK_BOARD}


def energy_levels(cfg: trials.Config, angles=()) -> dict[float, int]:
    """Rank of each hole (E1 the lowest): the configuration's holes, then any other logged one."""
    holes = sorted({float(a) for a in cfg.angles_deg} | {float(a) for a in angles if math.isfinite(float(a))})
    return {a: i + 1 for i, a in enumerate(holes)}


def simulated_rows(cfg: trials.Config, pending: list[dict], logged: list[dict], phase: str) -> list[dict]:
    """The slots still to do as simulated log rows: ids after the last logged one, SIM_DAYS sessions, the nominal mass."""
    prefix = trials.PREFIX.get(phase, "t")
    n0 = max([int(r["trial"][1:]) for r in logged if r["trial"][1:].isdigit()], default=0)
    total = len(logged) + len(pending)
    per_day = max(1, math.ceil(total / SIM_DAYS))
    out = []
    for i, s in enumerate(pending):
        k = len(logged) + i
        out.append({"trial": f"{prefix}{n0 + i + 1:03d}", "phase": phase, "status": "ok",
                    "session": (SIM_FIRST_DAY + timedelta(days=k // per_day)).isoformat(),
                    "holder": str(s["holder"]), "angle_deg": str(s["angle_deg"]), "direction_deg": str(s["direction_deg"]),
                    "drape": str(int(float(s.get("drape") or 0))), "swing_mass_g": str(NOMINAL_MASS_G),
                    "_planned": True})
    return out


def motion(run, row: dict, info: dict, cache: Path, simulate: bool) -> dict:
    """The marker's motion of a trial (twin.scan_result's summary), or of a simulated planned one."""
    if row.get("_planned"):
        D = twin.simulated_motion(row, info)
        T = np.asarray(info.get("T_W_mandible") or np.eye(4), np.float64)
        dT = twin.inverse(T) @ twin.inverse(D) @ T
        return twin.summary(D, dT, info, True, {})
    return twin.scan_result(run, row, info, cache, simulate=simulate, measure=True)


def read_ai(path: Path) -> dict[str, dict]:
    """The AI step of each trial from trials_ai.csv: {trial: {input: row}} ({} when there is none)."""
    if not Path(path).is_file():
        return {}
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("trial"):
                out.setdefault(r["trial"], {})[r.get("input") or MAIN_INPUT] = r
    return out


def simulated_correction(rng, tre: float, drape: bool) -> tuple[float, float]:
    """(TRE after a simulated correction, its predicted 95% TRE): mostly a few tenths of a millimetre, with an honest
    prediction; one in SIM_FAIL fails (1.5 to 6 mm) with a prediction that may be too confident."""
    if rng.uniform() < SIM_FAIL:
        resid = rng.uniform(1.5, 6.0)
        return resid, resid * rng.uniform(0.6, 1.6)
    resid = abs(rng.normal(0.35, 0.18)) * (1 + (rng.uniform(0.5, 2.0) if drape else 0)) + 0.02 * tre
    return resid, resid * rng.uniform(1.4, 2.4)


def simulated_ai(m: dict) -> dict:
    """A simulated AI step for a trial: a score that grows with the TRE (more spread with the drape), a correction
    when the score is over SIM_TAU, applied when its predicted TRE (95%) is under SIM_U, else a warning."""
    rng = np.random.default_rng(seed_of("ai " + m["trial"]))
    tre, drape = m["tre_mean_mm"], bool(m["drape"])
    score = tre * rng.uniform(0.55, 0.95) + abs(rng.normal(0, 0.12)) + (rng.uniform(0, 0.35) if drape else 0)
    detected = score > SIM_TAU
    applied = warning = 0
    after, corrected, pred = tre, math.nan, math.nan
    if detected:
        corrected, pred = simulated_correction(rng, tre, drape)
        if pred < SIM_U:
            applied, after = 1, corrected
        else:
            warning = 1
    return {"input": MAIN_INPUT, "score": score, "tau": SIM_TAU, "detected": int(detected), "applied": applied,
            "warning": warning, "tre_ai_mm": after, "tre_corrected_mm": corrected, "tre_pred95_mm": pred,
            "score_final": math.nan, "time_s": rng.uniform(3.5, 8.5) if detected else rng.uniform(0.4, 0.9)}


def read_validation(path: Path) -> list[dict]:
    """The corrections of the validation samples (VALIDATION_COLUMNS) from validation_ai.csv ([] when missing)."""
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{"sample": r.get("sample", ""), "group": r.get("group") or r.get("sample", ""),
                 "score_final": num(r.get("score_final")), "tau": num(r.get("tau")),
                 "tre_pred95_mm": num(r.get("tre_pred95_mm")), "tre_mm": num(r.get("tre_mm"))}
                for r in csv.DictReader(f)]


def simulated_validation(n: int = SIM_VALIDATION) -> list[dict]:
    """Simulated corrections of validation samples (pre-impact images with synthetic perturbations): the same
    model as the simulated AI step."""
    rng = np.random.default_rng(SEED + 1)
    out = []
    for i in range(n):
        tre = abs(rng.normal(2.5, 1.5))
        resid, pred = simulated_correction(rng, tre, rng.uniform() < 0.15)
        out.append({"sample": f"v{i + 1:03d}", "group": f"v{i + 1:03d}", "score_final": math.nan, "tau": math.nan,
                    "tre_pred95_mm": pred, "tre_mm": resid})
    return out


def collect(run, info: dict | None, cache: Path, results: Path, simulate: bool = False, rows=None, pending=None,
            progress=None) -> tuple[list[dict], int]:
    """The metrics of every closed trial of the run with its scans (and, with simulate, of the simulated slots still
    to do), each a dict of METRIC_COLUMNS plus "tre" {target: mm}, and how many closed trials with scans gave no
    motion (no reference, spheres not found). rows and pending: the log and the slots still to do as read on the
    caller's thread (read here when None). progress(done, of) after each trial."""
    info = model_info(info)
    cfg = run.config
    rows = run.log.rows() if rows is None else rows
    ok = [r for r in rows if r.get("status") == trials.OK]
    todo = [r for r in ok if twin.scans_of(run, r["trial"])]
    if simulate:
        todo += simulated_rows(cfg, run.pending() if pending is None else pending, rows, run.phase)
    levels = energy_levels(cfg, [num(r.get("angle_deg")) for r in todo])
    ai = read_ai(Path(results) / AI_FILE)
    out, failed = [], 0
    for i, r in enumerate(todo):
        res = motion(run, r, info, cache, simulate)
        if progress:
            progress(i + 1, len(todo))
        if res.get("state") != "ready":
            failed += 1
            continue
        angle = num(r.get("angle_deg"))
        mass = num(r.get("swing_mass_g"), NOMINAL_MASS_G)
        m = {"trial": r["trial"], "phase": run.phase, "session": str(r.get("session") or ""),
             "holder": str(r.get("holder") or "").upper(), "angle_deg": angle, "energy_level": levels.get(angle, 0),
             "energy_mj": energy_mj(angle, mass), "direction_deg": num(r.get("direction_deg")),
             "drape": flag(r.get("drape")), "rot_deg": float(res["rot_deg"]), "shift_mm": float(res["shift_mm"]),
             "tre_mean_mm": float(res["tre_mean_mm"]), "tre_max_mm": float(res["tre_max_mm"]),
             "tre": {t["id"]: float(t["mm"]) for t in res["tre"]},
             "simulated": int(bool(res.get("simulated")) or bool(r.get("_planned")))}
        m["lost"] = int(m["tre_mean_mm"] > LOST_MM)
        step = (ai.get(m["trial"]) or {})
        step = step.get(MAIN_INPUT) or (next(iter(step.values())) if step else None)
        if step is not None:
            m.update({"input": step.get("input") or MAIN_INPUT, "score": num(step.get("score")),
                      "tau": num(step.get("tau")), "detected": flag(step.get("detected")),
                      "applied": flag(step.get("applied")), "warning": flag(step.get("warning")),
                      "tre_ai_mm": num(step.get("tre_ai_mm")), "tre_corrected_mm": num(step.get("tre_corrected_mm")),
                      "tre_pred95_mm": num(step.get("tre_pred95_mm")), "score_final": num(step.get("score_final")),
                      "time_s": num(step.get("time_s"))})
        elif simulate:
            m.update(simulated_ai(m))
            m["simulated"] = 1
        out.append(m)
    return out, failed


# ── the conditions ──────────────────────────────────────────────────────────
def has_ai(metrics: list[dict]) -> bool:
    return any(math.isfinite(m.get("tre_ai_mm", math.nan)) for m in metrics)


def condition_rows(metrics: list[dict], drape: bool | None = False) -> list[dict]:
    """One row per trial and condition (drape: False the matrix only, True the drape trials only, None all)."""
    out = []
    for m in metrics:
        if drape is not None and bool(m["drape"]) != drape:
            continue
        h = m["holder"]
        base = {k: m[k] for k in ("trial", "energy_level", "holder", "angle_deg", "energy_mj", "direction_deg",
                                  "session", "drape")}
        out.append({**base, "condition": h, "tre_mm": m["tre_mean_mm"]})
        if math.isfinite(m.get("tre_ai_mm", math.nan)):
            out.append({**base, "condition": f"{h}_AI", "tre_mm": m["tre_ai_mm"]})
    return out


def describe(v, seed=SEED) -> dict:
    v = np.asarray([x for x in v if math.isfinite(x)], np.float64)
    med, q1, q3 = median_iqr(v)
    lo, hi = bootstrap(v, seed=seed)
    return {"n": int(len(v)), "median": med, "q1": q1, "q3": q3, "ci_low": lo, "ci_high": hi,
            "max": float(v.max()) if len(v) else math.nan,
            "below_1mm": float((v <= LIMITS_MM[0]).mean()) if len(v) else math.nan,
            "below_2mm": float((v <= LIMITS_MM[1]).mean()) if len(v) else math.nan}


def groups(metrics: list[dict]) -> list[tuple[str, str, object, list[dict]]]:
    """The groups of the TRE tables: (key, kind, level, rows): overall, per energy level, per direction (the matrix),
    and the drape trials."""
    main = condition_rows(metrics, drape=False)
    out = [("all", "all", None, main)]
    for lv in sorted({r["energy_level"] for r in main}):
        out.append((f"e{lv}", "energy", lv, [r for r in main if r["energy_level"] == lv]))
    for d in sorted({r["direction_deg"] for r in main}):
        out.append((f"d{int(d)}", "direction", d, [r for r in main if r["direction_deg"] == d]))
    out.append(("drape", "drape", 1, condition_rows(metrics, drape=True)))
    return out


def impact_table(metrics: list[dict]) -> list[dict]:
    rows = []
    for key, kind, level, part in groups(metrics):
        for c in CONDITIONS:
            sub = [r for r in part if r["condition"] == c]
            if not sub:
                continue
            d = describe([r["tre_mm"] for r in sub], seed=seed_of(c + key))
            row = {"key": f"{c}_{key}", "n": d["n"], "tre_median_mm": rnd(d["median"]), "tre_q1_mm": rnd(d["q1"]),
                   "tre_q3_mm": rnd(d["q3"]), "tre_ci_low_mm": rnd(d["ci_low"]), "tre_ci_high_mm": rnd(d["ci_high"]),
                   "tre_max_mm": rnd(d["max"]), "below_1mm": rnd(d["below_1mm"]), "below_2mm": rnd(d["below_2mm"])}
            if kind == "energy":
                row["angle_deg"] = rnd(sub[0]["angle_deg"], 0)
                row["energy_mj"] = rnd(float(np.mean([r["energy_mj"] for r in sub])), 1)
            if "_" not in c:            # the true displacement, for the trials without correction
                ids = {r["trial"] for r in sub}
                ms = [m for m in metrics if m["trial"] in ids]
                rot = [m["rot_deg"] for m in ms]
                shift = [m["shift_mm"] for m in ms]
                row.update({"rot_median_deg": rnd(np.median(rot)), "rot_max_deg": rnd(max(rot)),
                            "shift_median_mm": rnd(np.median(shift)), "shift_max_mm": rnd(max(shift)),
                            "lost": rnd(float(np.mean([m["lost"] for m in ms])))})
            rows.append(row)
    return rows


def target_table(metrics: list[dict]) -> list[dict]:
    rows = []
    for h in sorted({m["holder"] for m in metrics}):
        ms = [m for m in metrics if m["holder"] == h and not m["drape"]]
        for t in (ms[0]["tre"] if ms else {}):
            v = [m["tre"][t] for m in ms if t in m["tre"]]
            med, q1, q3 = median_iqr(v)
            rows.append({"key": f"{h}_{t}", "n": len(v), "tre_median_mm": rnd(med), "tre_q1_mm": rnd(q1),
                         "tre_q3_mm": rnd(q3), "tre_max_mm": rnd(max(v))})
    return rows


def paired(metrics: list[dict], a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    """The TRE of conditions a and b paired: on the same trials when they share the holder, else on the same cells
    (hole, direction and repetition, in the order the trials were done)."""
    main = condition_rows(metrics, drape=False)
    by = {}
    for r in main:
        by.setdefault(r["condition"], []).append(r)
    ra, rb = by.get(a, []), by.get(b, [])
    if a[0] == b[0]:
        tb = {r["trial"]: r["tre_mm"] for r in rb}
        pairs = [(r["tre_mm"], tb[r["trial"]]) for r in ra if r["trial"] in tb]
    else:
        def cells(rows):
            seen, out = {}, {}
            for r in sorted(rows, key=lambda r: r["trial"]):
                c = (r["angle_deg"], r["direction_deg"])
                seen[c] = seen.get(c, 0) + 1
                out[c + (seen[c],)] = r["tre_mm"]
            return out
        ca, cb = cells(ra), cells(rb)
        pairs = [(ca[k], cb[k]) for k in ca if k in cb]
    x = np.array([p[0] for p in pairs], np.float64)
    y = np.array([p[1] for p in pairs], np.float64)
    return x, y


def tests_table(metrics: list[dict]) -> list[dict]:
    family = [("A", "A_AI"), ("B", "B_AI"), ("A", "B"), ("A_AI", "B"), ("A_AI", "B_AI")]
    present = {r["condition"] for r in condition_rows(metrics, drape=False)}
    rows, ps = [], []
    for a, b in family:
        if a not in present or b not in present:
            continue
        x, y = paired(metrics, a, b)
        w, p = stats.wilcoxon_paired(x, y)
        ps.append(p)
        rows.append({"key": f"{a}_vs_{b}", "n": len(x), "median_1_mm": rnd(np.median(x) if len(x) else math.nan),
                     "median_2_mm": rnd(np.median(y) if len(y) else math.nan),
                     "median_diff_mm": rnd(np.median(x - y) if len(x) else math.nan), "w": rnd(w, 1), "p": p})
    for row, p_adj in zip(rows, holm(ps)):
        row["p_holm"] = sig(p_adj)
        row["p"] = sig(row["p"])
    return rows


TERM_KEYS = {"Intercept": "intercept", "energy_mj": "energy_mj", "C(holder)[T.B]": "holder_b",
             "energy_mj:C(holder)[T.B]": "energy_mj_x_holder_b", "Group Var": "session_var"}


def term_key(name: str) -> str:
    if name in TERM_KEYS:
        return TERM_KEYS[name]
    if name.startswith("C(direction_deg)[T."):
        return "direction_" + name[len("C(direction_deg)[T."):].rstrip("]").replace(".0", "").replace(".", "p")
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def mixed_table(metrics: list[dict]) -> tuple[list[dict], dict]:
    """The mixed model TRE ~ energy * interface + direction + (1 | session) on the TRE without correction of the
    matrix: (rows of terms, {"method", "n", "groups", "converged"}); no rows when it cannot be fitted."""
    import pandas as pd
    main = [r for r in condition_rows(metrics, drape=False) if "_" not in r["condition"]]
    if len(main) < 8 or len({r["holder"] for r in main}) < 2:
        return [], {"method": "", "n": len(main), "groups": 0, "converged": False}
    df = pd.DataFrame(main)
    df["direction_deg"] = df["direction_deg"].round().astype(int)
    terms, info = stats.mixed_model(df, "tre_mm ~ energy_mj * C(holder) + C(direction_deg)", "session")
    rows = [{"key": term_key(t["term"]), "coef": sig(t["coef"], 4), "se": sig(t["se"], 3), "ci_low": sig(t["ci_low"], 4),
             "ci_high": sig(t["ci_high"], 4), "p": sig(t["p"])} for t in terms]
    rows.append({"key": "model", "method": info["method"], "n": info["n"], "groups": info["groups"],
                 "converged": int(info["converged"])})
    return rows, info


def detection_table(metrics: list[dict]) -> list[dict]:
    rows = []
    for drape in (False, True):
        ms = [m for m in metrics if bool(m["drape"]) == drape and math.isfinite(m.get("score", math.nan))]
        for inp in sorted({m["input"] for m in ms}):
            sub = [m for m in ms if m["input"] == inp]
            lost = np.array([m["lost"] for m in sub], bool)
            score = np.array([m["score"] for m in sub])
            det = np.array([m["detected"] for m in sub], bool)
            sens = float(det[lost].mean()) if lost.any() else math.nan
            spec = float((~det[~lost]).mean()) if (~lost).any() else math.nan
            rng = np.random.default_rng(seed_of("det" + inp + str(drape)))
            boots = {"auc": [], "sens": [], "spec": []}
            for _ in range(BOOTSTRAP if len(sub) >= 2 else 0):
                k = rng.integers(0, len(sub), len(sub))
                lk, dk = lost[k], det[k]
                boots["auc"].append(stats.auc(score[k], lk))
                boots["sens"].append(dk[lk].mean() if lk.any() else math.nan)
                boots["spec"].append((~dk[~lk]).mean() if (~lk).any() else math.nan)

            def ci(name):
                v = np.asarray(boots[name], np.float64)
                v = v[np.isfinite(v)]
                return (rnd(np.quantile(v, 0.025)), rnd(np.quantile(v, 0.975))) if len(v) else ("", "")
            taus = [m["tau"] for m in sub if math.isfinite(m.get("tau", math.nan))]
            a_lo, a_hi = ci("auc")
            s_lo, s_hi = ci("sens")
            p_lo, p_hi = ci("spec")
            rows.append({"key": inp + ("_drape" if drape else ""), "n": len(sub), "n_lost": int(lost.sum()),
                         "tau_mm": rnd(np.median(taus)) if taus else "", "auc": rnd(stats.auc(score, lost)),
                         "auc_ci_low": a_lo, "auc_ci_high": a_hi, "sensitivity": rnd(sens),
                         "sensitivity_ci_low": s_lo, "sensitivity_ci_high": s_hi, "specificity": rnd(spec),
                         "specificity_ci_low": p_lo, "specificity_ci_high": p_hi})
    return rows


def correction_table(metrics: list[dict]) -> list[dict]:
    rows = []
    for drape in (False, True):
        for h in sorted({m["holder"] for m in metrics}):
            ms = [m for m in metrics if m["holder"] == h and bool(m["drape"]) == drape
                  and math.isfinite(m.get("tre_ai_mm", math.nan))]
            for inp in sorted({m["input"] for m in ms}):
                sub = [m for m in ms if m["input"] == inp]
                lost = [m for m in sub if m["lost"]]
                after = np.array([m["tre_ai_mm"] for m in lost], np.float64)
                before = np.array([m["tre_mean_mm"] for m in lost], np.float64)
                warn = np.array([m["warning"] for m in lost], bool)
                n = len(lost)
                share = (lambda v: rnd(float(np.mean(v))) if n else "")
                applied = [m for m in sub if m["applied"]]
                rows.append({"key": f"{h}_AI_{inp}" + ("_drape" if drape else ""), "n": len(sub), "n_lost": n,
                             "tre_before_median_mm": rnd(np.median(before)) if n else "",
                             "tre_after_median_mm": rnd(np.median(after)) if n else "",
                             "below_1mm": share(after <= LIMITS_MM[0]), "below_2mm": share(after <= LIMITS_MM[1]),
                             "warnings": share(warn), "safe": share((after <= LOST_MM) | warn),
                             "applied": rnd(len(applied) / len(sub)) if sub else "",
                             "worse": rnd(float(np.mean([m["tre_ai_mm"] > m["tre_mean_mm"] for m in applied])))
                             if applied else "",
                             "time_median_s": rnd(np.median([m["time_s"] for m in sub
                                                             if math.isfinite(m.get("time_s", math.nan))]), 2)
                             if sub else ""})
    return rows


def corrections(metrics: list[dict], drape: bool) -> list[dict]:
    """The corrections made on the trials (the AI step found the registration lost), with their TRE whether applied
    or not and their predicted 95% TRE."""
    return [m for m in metrics if bool(m["drape"]) == drape and m.get("detected")
            and math.isfinite(m.get("tre_corrected_mm", math.nan)) and math.isfinite(m.get("tre_pred95_mm", math.nan))]


def gate_of(rows: list[dict]) -> np.ndarray:
    """Through the gate: the final score at most tau (every one when the final score is not given)."""
    return np.array([not (math.isfinite(r.get("score_final", math.nan)) and math.isfinite(r.get("tau", math.nan)))
                     or r["score_final"] <= r["tau"] for r in rows], bool)


def uncertainty_table(metrics: list[dict], validation: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """How far the predicted TRE can be trusted: the threshold u by conformal risk control on the validation samples
    only (ALPHA), what it gives on the trials (the test: accepted, silent errors observed, the guarantee met), the
    calibration diagram and the error against the share accepted. (key rows, curve rows, summary); nothing without
    validation samples or corrections."""
    test = corrections(metrics, False)
    if not validation or not test:
        return [], [], {}
    pv = np.array([v["tre_pred95_mm"] for v in validation])
    tv = np.array([v["tre_mm"] for v in validation])
    gv = gate_of(validation)
    u = stats.conformal_u(pv, tv, gv, ALPHA, LOST_MM, [v["group"] for v in validation])
    on_val = stats.guarantee(pv, tv, gv, u, ALPHA, LOST_MM)
    u_text = rnd(u) if math.isfinite(u) else ("inf" if u > 0 else "-inf")
    rows = [{"key": "validation", "n": len(validation), "alpha": ALPHA, "u_mm": u_text,
             "accepted": rnd(on_val["accepted"]), "silent": rnd(on_val["silent"])}]
    out = {"u": u if math.isfinite(u) else None, "alpha": ALPHA, "n_validation": len(validation)}
    for drape in (False, True):
        part = corrections(metrics, drape)
        if not part:
            continue
        pt = np.array([m["tre_pred95_mm"] for m in part])
        tt = np.array([m["tre_corrected_mm"] for m in part])
        g = stats.guarantee(pt, tt, gate_of(part), u, ALPHA, LOST_MM)
        key = "test_drape" if drape else "test"
        rows.append({"key": key, "n": g["n"], "alpha": ALPHA, "accepted": rnd(g["accepted"]),
                     "silent": rnd(g["silent"]), "silent_of_accepted": rnd(g["silent_of_accepted"]),
                     "within": int(g["within"])})
        out[key] = clean(g)
    pt = np.array([m["tre_pred95_mm"] for m in test])
    tt = np.array([m["tre_corrected_mm"] for m in test])
    curve = []
    for b in stats.calibration_bins(pt, tt, 5):
        rows.append({"key": f"bin_{b['bin']}", "n": b["n"], "pred_median_mm": rnd(b["pred_median_mm"]),
                     "obs_median_mm": rnd(b["obs_median_mm"]), "obs_p95_mm": rnd(b["obs_p95_mm"]),
                     "covered": rnd(b["covered"])})
        curve.append({"curve": "calibration", "x": rnd(b["pred_median_mm"], 4), "y": rnd(b["obs_median_mm"], 4),
                      "y_p95": rnd(b["obs_p95_mm"], 4), "n": b["n"]})
    for c in stats.risk_coverage(pt, tt):
        rows.append({"key": f"cov_{int(round(c['share'] * 100))}", "n": c["n"],
                     "tre_median_mm": rnd(c["tre_median_mm"]), "tre_p95_mm": rnd(c["tre_p95_mm"]),
                     "over_limit": rnd(c["over_limit"])})
        curve.append({"curve": "accepted", "x": rnd(c["share"], 2), "y": rnd(c["tre_median_mm"], 4),
                      "y_p95": rnd(c["tre_p95_mm"], 4), "n": c["n"]})
    return rows, curve, out


# ── the tables ──────────────────────────────────────────────────────────────
def per_trial_tables(metrics: list[dict]) -> dict[str, tuple[tuple, list[dict]]]:
    met = []
    for m in metrics:
        r = {c: m.get(c, "") for c in METRIC_COLUMNS}
        for c in ("rot_deg", "shift_mm", "tre_mean_mm", "tre_max_mm", "score", "tre_ai_mm", "tre_corrected_mm",
                  "tre_pred95_mm"):
            r[c] = rnd(m.get(c, math.nan), 4)
        r["energy_mj"] = rnd(m["energy_mj"], 2)
        r["time_s"] = rnd(m.get("time_s", math.nan), 2)
        for c in ("angle_deg", "direction_deg"):
            r[c] = rnd(m[c], 0)
        if not math.isfinite(m.get("tre_ai_mm", math.nan)):
            for c in ("input", "score", "detected", "applied", "warning", "tre_ai_mm", "tre_corrected_mm",
                      "tre_pred95_mm", "time_s"):
                r[c] = ""
        met.append(r)
    tre = [{"trial": r["trial"], "energy_level": r["energy_level"], "condition": r["condition"],
            "tre_mm": rnd(r["tre_mm"], 4), "holder": r["holder"], "angle_deg": rnd(r["angle_deg"], 0),
            "energy_mj": rnd(r["energy_mj"], 2), "direction_deg": rnd(r["direction_deg"], 0), "session": r["session"]}
           for r in condition_rows(metrics, drape=False)]
    tg = [{"trial": m["trial"], "holder": m["holder"], "target": t, "tre_mm": rnd(v, 4),
           "energy_level": m["energy_level"], "angle_deg": rnd(m["angle_deg"], 0),
           "direction_deg": rnd(m["direction_deg"], 0), "drape": m["drape"], "session": m["session"]}
          for m in metrics for t, v in m["tre"].items()]
    return {"trials_metrics.csv": (METRIC_COLUMNS, met), "trials_tre.csv": (TRE_COLUMNS, tre),
            "trials_targets.csv": (TARGET_COLUMNS, tg)}


def key_columns(rows: list[dict]) -> tuple:
    cols = ["key"]
    for r in rows:
        cols += [c for c in r if c not in cols]
    return tuple(cols)


def csv_text(columns, rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: ("" if r.get(c) is None else r.get(c, "")) for c in columns})
    return buf.getvalue()


def write_if_changed(path: Path, text: str) -> bool:
    """Writes text into path in one step, only when it differs from what is there; whether it was written."""
    path = Path(path)
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with_suppress = Path(tmp)
        if with_suppress.exists():
            with_suppress.unlink()
        raise
    return True


def tables(metrics: list[dict], validation: list[dict] | None = None) -> tuple[dict, dict, dict]:
    """Every table of the metrics: {file: (columns, rows)}, the mixed model's fit and the uncertainty's summary."""
    out = per_trial_tables(metrics)
    out["impact.csv"] = impact_table(metrics)
    out["impact_targets.csv"] = target_table(metrics)
    out["stats.csv"] = tests_table(metrics)
    model, fit = mixed_table(metrics)
    out["mixed_model.csv"] = model
    if has_ai(metrics):
        out["detection.csv"] = detection_table(metrics)
        out["correction.csv"] = correction_table(metrics)
    unc_rows, curve, unc = uncertainty_table(metrics, validation or [])
    if unc_rows:
        out["uncertainty.csv"] = unc_rows
        out["uncertainty_curve.csv"] = (("curve", "x", "y", "y_p95", "n"), curve)
    for name, v in list(out.items()):
        if isinstance(v, list):
            out[name] = (key_columns(v), v)
    return out, fit, unc


def write_tables(made: dict, folder: Path) -> list[str]:
    """The tables written into folder (only the changed ones); the names of the files written."""
    written = []
    for name, (columns, rows) in made.items():
        if rows and write_if_changed(Path(folder) / name, csv_text(columns, rows)):
            written.append(name)
    return written


# ── the summary for the page ────────────────────────────────────────────────
def summary(metrics: list[dict], made: dict, fit: dict, simulate: bool, written: list[str], folder: Path,
            missing: int, unc: dict | None = None, holders: dict | None = None) -> dict:
    """What the Analysis page shows: counts, the four conditions, the displacement, the breakdowns (energy,
    direction, drape, target), the tests, the model, detection and correction."""
    def keyed(name):
        return {r["key"]: r for r in made.get(name, ((), []))[1]}
    imp = keyed("impact.csv")
    conds = [c for c in CONDITIONS if f"{c}_all" in imp]

    def cell(c, k):
        r = imp.get(f"{c}_{k}")
        return None if r is None else {"n": r["n"], "median": r["tre_median_mm"], "q1": r["tre_q1_mm"],
                                       "q3": r["tre_q3_mm"], "below_1mm": r["below_1mm"],
                                       "below_2mm": r["below_2mm"]}
    by = {"energy": [], "direction": [], "drape": [], "target": []}
    for key, kind, level, _ in groups(metrics):
        if kind in ("energy", "direction"):
            row = {"key": key, "level": level, "cells": {c: cell(c, key) for c in conds}}
            if kind == "energy":
                any_r = next((imp[f"{c}_{key}"] for c in conds if f"{c}_{key}" in imp), {})
                row.update({"angle_deg": any_r.get("angle_deg"), "energy_mj": any_r.get("energy_mj")})
            by[kind].append(row)
    by["drape"] = [{"key": "all", "level": 0, "cells": {c: cell(c, "all") for c in conds}},
                   {"key": "drape", "level": 1, "cells": {c: cell(c, "drape") for c in conds}}]
    tg = keyed("impact_targets.csv")
    for t in (metrics[0]["tre"] if metrics else {}):
        by["target"].append({"key": t, "cells": {h: ({"n": tg[f"{h}_{t}"]["n"], "median": tg[f"{h}_{t}"]["tre_median_mm"],
                                                       "q1": tg[f"{h}_{t}"]["tre_q1_mm"], "q3": tg[f"{h}_{t}"]["tre_q3_mm"]}
                                                      if f"{h}_{t}" in tg else None) for h in ("A", "B")}})
    disp = []
    for h in ("A", "B"):
        r = imp.get(f"{h}_all")
        if r:
            disp.append({"holder": h, "rot_median": r["rot_median_deg"], "rot_max": r["rot_max_deg"],
                         "shift_median": r["shift_median_mm"], "shift_max": r["shift_max_mm"], "lost": r["lost"]})
    out = {"simulated": bool(simulate) or any(m["simulated"] for m in metrics), "n": len(metrics),
           "n_holder": {h: sum(1 for m in metrics if m["holder"] == h) for h in sorted({m["holder"] for m in metrics})},
           "n_drape": sum(1 for m in metrics if m["drape"]), "missing": missing, "ai": has_ai(metrics),
           "conditions": [{"key": c, **cell(c, "all")} for c in conds], "displacement": disp, "by": by,
           "tests": [{"key": r["key"], "n": r["n"], "diff": r["median_diff_mm"], "p": num(r["p"]),
                      "p_holm": num(r["p_holm"])} for r in made.get("stats.csv", ((), []))[1]],
           "model": {**fit, "terms": [{"key": r["key"], "coef": num(r.get("coef")), "ci_low": num(r.get("ci_low")),
                                       "ci_high": num(r.get("ci_high")), "p": num(r.get("p"))}
                                      for r in made.get("mixed_model.csv", ((), []))[1] if r["key"] != "model"]},
           "detection": [{"key": r["key"], "n": r["n"], "n_lost": r["n_lost"], "auc": r["auc"],
                          "auc_ci": [r["auc_ci_low"], r["auc_ci_high"]], "sensitivity": r["sensitivity"],
                          "specificity": r["specificity"], "tau": r["tau_mm"]}
                         for r in made.get("detection.csv", ((), []))[1]],
           "correction": [{"key": r["key"], "n_lost": r["n_lost"], "after": r["tre_after_median_mm"],
                           "below_1mm": r["below_1mm"], "below_2mm": r["below_2mm"], "warnings": r["warnings"],
                           "worse": r["worse"], "safe": r["safe"]}
                          for r in made.get("correction.csv", ((), []))[1]],
           "uncertainty": unc or None, "interfaces": holders,
           "written": written, "folder": str(folder), "tables": sorted(n for n, (_, rows) in made.items() if rows)}
    return clean(out)


def analyse(run, info: dict | None, cache: Path, results: Path, simulate: bool = False, rows=None, pending=None,
            progress=None, interfaces_raw: Path | None = None, interfaces_store: Path | None = None) -> dict:
    """The whole analysis of a run: metrics, tables written into results, the page's summary; with interfaces_raw
    also the two holders measured on their own (interfaces.py). Raises AnalysisError no_trials when no trial has its
    motion (nothing is written then)."""
    metrics, failed = collect(run, info, cache, results, simulate, rows, pending, progress)
    if not metrics:
        raise AnalysisError("no_trials")
    validation = read_validation(Path(results) / VALIDATION_FILE)
    if not validation and simulate and has_ai(metrics):
        validation = simulated_validation()
    made, fit, unc = tables(metrics, validation)
    written = write_tables(made, results)
    holders = None
    if interfaces_raw is not None:
        import interfaces
        holders = interfaces.analyse(interfaces_raw, results, simulate=simulate,
                                     store=interfaces_store or Path(cache) / "no_reseating")
        if holders:
            written += holders["written"]
    return summary(metrics, made, fit, simulate, written, results, failed, unc, holders)


# ── command line ────────────────────────────────────────────────────────────
def _run_in(work: Path) -> trials.Run:
    return trials.Run("campaign", raw=work / "raw", protocol=work / "protocol")


def selftest() -> int:
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)
    work = Path(tempfile.mkdtemp(prefix="analysis_test_"))
    try:
        run = _run_in(work)
        out = work / "results"
        s = analyse(run, None, work / "cache", out, simulate=True)
        cfg = run.config
        n_main = len(cfg.holders) * len(cfg.angles_deg) * len(cfg.directions_deg) * cfg.repetitions
        expect(s["n"] == n_main + cfg.drape_trials and s["simulated"] and s["n_drape"] == cfg.drape_trials,
               f"simulated campaign: {s['n']} trials ({n_main} in the matrix, {s['n_drape']} with the drape), said so")
        import pandas as pd
        m = pd.read_csv(out / "trials_metrics.csv")
        a = m[(m.holder == "A") & (m.drape == 0)]
        rho = a[["energy_level", "tre_mean_mm"]].corr("spearman").iloc[0, 1]
        b = m[m.holder == "B"]
        expect(rho > 0.6 and b.tre_mean_mm.max() < 0.5,
               f"A slips more with energy (Spearman {rho:.2f}), B comes back (TRE at most {b.tre_mean_mm.max():.3f} mm)")
        tre = pd.read_csv(out / "trials_tre.csv")
        expect(list(tre.columns[:4]) == ["trial", "energy_level", "condition", "tre_mm"]
               and set(tre.condition) == set(CONDITIONS) and len(tre) == 2 * n_main,
               f"trials_tre.csv: the four conditions of the {n_main} trials of the matrix ({len(tre)} rows)")
        hi = tre[(tre.energy_level == tre.energy_level.max())]
        expect(hi[hi.condition == "A_AI"].tre_mm.median() < hi[hi.condition == "A"].tre_mm.median(),
               "at the highest energy the AI step lowers the TRE of A")
        import re
        ids = re.compile(r"[A-Za-z0-9_.-]+")
        keyed = [n for n in TABLES if n not in PER_ROW]
        good = True
        for n in keyed:
            with open(out / n, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            good &= rows[0][0] == "key" and all(ids.fullmatch(c) for c in rows[0]) and \
                len({r[0] for r in rows[1:]}) == len(rows) - 1 and all(ids.fullmatch(r[0]) for r in rows[1:])
        expect(good, f"the key tables have a first column key and plain ids ({', '.join(keyed)})")
        st = {r["key"]: r for r in s["tests"]}
        expect(all(st[k]["p_holm"] >= st[k]["p"] for k in st if st[k]["p"] is not None)
               and st["A_vs_A_AI"]["p_holm"] < 0.05,
               f"paired tests with Holm: adjusted p never below the raw one, A against A+AI significant "
               f"(p {st['A_vs_A_AI']['p_holm']:.2g})")
        terms = {t["key"]: t for t in s["model"]["terms"]}
        expect(s["model"]["method"] == "mixedlm" and s["model"]["groups"] == SIM_DAYS
               and terms["energy_mj"]["coef"] > 0 and terms["energy_mj_x_holder_b"]["coef"] < 0,
               f"mixed model over {s['model']['groups']} sessions: TRE of A grows with energy "
               f"({terms['energy_mj']['coef']:.3g} mm/mJ), B's much less")
        det = {r["key"]: r for r in s["detection"]}
        expect(det[MAIN_INPUT]["auc"] > 0.8 and 0.5 < det[MAIN_INPUT]["sensitivity"] <= 1,
               f"detection on the simulated AI step: AUC {det[MAIN_INPUT]['auc']}, sensitivity "
               f"{det[MAIN_INPUT]['sensitivity']}")
        again = analyse(run, None, work / "cache", out, simulate=True)
        expect(again["written"] == [] and again["by"] == s["by"], "a second run with the same data writes nothing")
        expect(s["by"]["energy"][0]["angle_deg"] == min(cfg.angles_deg) and len(s["by"]["target"]) == 7
               and len(s["by"]["direction"]) == len(cfg.directions_deg),
               "breakdowns: per energy level with its hole, per direction, the seven targets")
        expect(abs(energy_mj(10, 137, 236) - 4.8) < 0.05 and abs(energy_mj(50, 137, 236) - 113.0) < 0.5,
               "energy of the holes as the experiment's table (10 degrees 4.8 mJ, 50 degrees 113 mJ)")
        unc = s["uncertainty"]
        expect(unc and unc["n_validation"] == SIM_VALIDATION and unc["u"] is not None and unc["test"]["n"] > 10
               and (out / "uncertainty.csv").is_file() and (out / "uncertainty_curve.csv").is_file(),
               f"uncertainty: u {unc['u']:.2f} mm from the {unc['n_validation']} validation samples only, "
               f"{unc['test']['silent']:.3f} silent errors on the {unc['test']['n']} corrections of the trials")
        s2 = analyse(run, None, work / "cache", work / "r2", simulate=True, interfaces_raw=work / "raw" / "interfaces")
        expect(s2["interfaces"] and s2["interfaces"]["a"]["n"] == 10 and "interfaces.csv" in s2["written"],
               "with the holders: the slip torque of A and the reseating of B in the same run")
        try:
            analyse(_run_in(work / "empty"), None, work / "cache", work / "none", simulate=False)
            expect(False, "no trials: said as no_trials")
        except AnalysisError as e:
            expect(e.code == "no_trials" and not (work / "none").exists(), "no trials: no_trials, nothing written")
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
        q.add_argument("--phase", default="campaign", choices=tuple(trials.PREFIX))
        q.add_argument("--out", type=Path, default=None)
    sub.add_parser("test")
    a = p.parse_args(argv)
    if a.cmd == "test":
        return selftest()
    simulate = a.cmd == "simulate"
    if simulate and a.out is None:
        p.error("simulate needs --out (simulated tables never go into the results folder)")
    watcher = twin.Watcher(twin.CACHE, blender=None)
    run = trials.Run(a.phase)
    try:
        s = analyse(run, watcher.info(), watcher.cache, a.out or RESULTS, simulate=simulate,
                    progress=lambda i, n: print(f"\r{i}/{n} trials", end="", flush=True))
    except AnalysisError as e:
        print(f"\n{e.code}")
        return 1
    print(f"\n{s['n']} trials; written: {', '.join(s['written']) or 'nothing changed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
