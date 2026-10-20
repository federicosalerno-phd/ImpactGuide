"""Pilot impacts: the pendulum from the model of its parts, the energy of each goniometer hole, the tip speed from the
240 fps side video, the slip threshold of holder A, the power of the campaign and the scanner's export format. All of
it runs on simulated data before the pilot (test, synthetic) and on the pilot's own files after it (analyse).

Model (model). The swinging part read from the bench scene (Blender in the background, on a copy of the scene, never
saved): every mesh under pendulum_swing that the renders show (design alternatives left out), solid, with the density
of its material (DENSITY): mass, centre of mass, moment of inertia about the axle, period of small swings, radius of
the impact point and the effective mass there. The impact point is the scene's strike point, checked against the nose
of the tip (the nose itself when the two are more than STRIKE_TOL_MM apart). The same run reads the square marker holder P1:
the ChArUco frame M of its label (as marker_pose.py: origin at the outer corner of the first square, x along the
printed columns, y along the rows, z into the board) and, in M, the cap's strike point, its centroid and axis and the
three sphere centres. Written to 03_data/processed/protocol/pendulum_model.json and marker_holder.json; the strike point
in M is what the 3 mm rule of the repositioning uses (trials.cap_point_M, trials.cap_shift_mm).

Energies (energy). Contact is the pendulum's rest pose (rotation 0 in the scene: the tip touches the cap). Raised by
theta away from the marker, the centre of mass c (h towards the marker, z up, from the axle) rises by
dh = c_z cos(theta) - c_h sin(theta) - c_z, so E = M g dh; at contact omega = sqrt(2 E / I), the tip moves at
v = omega r. TRANSFER is the share of E that enters the marker (rigid impact, restitution 0.5, the marker's effective
mass at the cap about its joint: frontal, 45 degrees, lateral); a 50 g instrument dropped from h = E / (m g) carries the
same energy. Tables: 05_results/pendulum_model.csv and pendulum_energy.csv.

Side video (track_video). One or more small ArUco stickers (DICT_4X4_50) on the side face of the swinging part, in the
swing plane: one near the axle (slow, never blurred) and one on the mass block (far from the axle, precise). For each
sticker the frames before the release fix a homography from its plane to the image; every frame's corners are taken
back into that plane and a 2D rigid fit gives the turn of the pendulum, so the angle needs neither a ruler nor a
camera square to the plane. Contact is where the quadratic fits of the descent and of the rebound meet: the angular
speed just before it, the release angle (from the frames before the release to contact), the rebound angle and the
apparent restitution. Frame rate: the one given, else the capture rate of an Android slow motion
(com.android.capture.fps), else the container's.

Slip threshold (slip_threshold). The rotation of A measured by the scans against the pendulum's energy, per direction:
rot = k max(0, E - E0) by least squares over a grid of E0, with a bootstrap interval; E5 = E0 + 5 / k is the energy
that turns A by 5 degrees. The same fit on the energy that enters the marker (TRANSFER) pools the directions.

Power (power). The planned comparisons (paired by trial, or matched by cell) with a Wilcoxon signed-rank test at the
Holm level of the first of COMPARISONS comparisons (alpha / m): Monte Carlo power to see a shift of DELTA_MM, with the
differences drawn from the pilot (centred, then shifted), and the number of impacts (and repetitions of the 27 cells)
that reach POWER.

Scanner export (scan_format). What a file exported by the scanner is: format (PLY ascii or binary, STL, OBJ, text),
points, extent (a cloud of a few millimetres is in metres), spheres found and rigs identified.

Files of the pilot (analyse): the log and the side videos in 03_data/raw/pilot (trials.py), the scans' results in
03_data/processed/ground_truth/pilot/gt_poses.csv (scan_desk.py). Tables in 05_results: pilot_trials.csv (one row per
trial), pilot_energy.csv (per hole), pilot_threshold.csv (per direction), pilot_power.csv (per comparison), pilot.csv
(summary). synthetic writes the same analysis of a simulated pilot into the work folder and its checks against the
truth into 05_results/pilot_synthetic.csv.

Usage:
    python pilot.py model [--scene BLEND]      Blender on a copy of the bench scene; writes the two protocol files
    python pilot.py energy                     the energy tables from pendulum_model.json
    python pilot.py video FILE [--fps N]       one side video: release angle, speed at contact, rebound
    python pilot.py analyse [--corrected CSV]  every pilot trial; CSV: trial, tre_after_mm (corrections, if any)
    python pilot.py scan-format FILE ...       can these scanner exports be read and measured?
    python pilot.py synthetic [--work DIR]     a simulated pilot through the whole analysis
    python pilot.py test [--work DIR]          checks on simulated data; exit code 1 on a failure
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Blender's Python (this file runs inside it too) ignores PYTHONPYCACHEPREFIX: apply it here, and write no bytecode
# next to the sources without it
if os.environ.get("PYTHONPYCACHEPREFIX"):
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
else:
    sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trials  # noqa: E402  (the protocol: holes, directions, arm positions, paths)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "03_data" / "processed" / "protocol"
RESULTS = ROOT / "05_results"
RAW = ROOT / "03_data" / "raw"
GT_POSES = ROOT / "03_data" / "processed" / "ground_truth" / "pilot" / "gt_poses.csv"
MODEL_JSON = PROTOCOL / "pendulum_model.json"
CAP_JSON = PROTOCOL / "marker_holder.json"
WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
SCENE = WORK / "test_bench_draft.blend"
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
STABLE_S = 120                  # the scene is copied only when it has not been saved for this long

G_MM = 9806.65                  # mm / s^2
HOLES_DEG = trials.Config().angles_deg
DIRECTIONS_DEG = trials.Config().directions_deg
# g / cm^3; printed parts solid (the mass block and the tip are printed at 100 % infill, the rest is light)
DENSITY = {"aluminium": 2.70, "steel": 7.85, "spring_steel": 7.85, "brass": 8.5, "petg": 1.27, "pla": 1.24,
           "sphere": 1.41, "magnet": 7.5, "rubber": 1.2}
SWING = "pendulum_swing"
TIP = "pendulum_tip"
WASHER = "pendulum_mass_washer_l1"
P1SQ = {"label": "marker_charuco__p1sq", "cap": "marker_strike_cap__p1sq", "holder": "marker_holder__p1sq",
        "spheres": [f"gt_sphere_marker_{i}__p1sq" for i in (1, 2, 3)]}
BOARD_MM = 45.0                 # the printed squares: 5 x 9 mm (marker_pose.BOARD), inside the 50 mm label image
STRIKE_TOL_MM = 1.0             # the scene's strike point within this of the pendulum's nose: it is the contact
# share of the pendulum's energy that enters the marker, per direction (rigid impact, restitution 0.5, 100 g effective
# at the tip against the marker's 31 / 21 / 15 g at the cap about its joint)
TRANSFER = {0: 0.40, 45: 0.33, 90: 0.26}
TURN_MJ = 17.0                  # energy that turns A by 5 degrees against its 0.2 N m (0.2 x 0.087 J)
PLIERS_G = 50.0                 # an instrument dropped on the marker, for the energy as a height
DELTA_MM = 0.5                  # difference of TRE the campaign must see...
POWER = 0.8                     # ... with this power
ALPHA = 0.05
COMPARISONS = 4                 # A vs A+AI, B vs B+AI, B vs A+AI, A vs B (Holm: the first at alpha / 4)
CELLS = len(HOLES_DEG) * len(DIRECTIONS_DEG)
PLANNED_N = CELLS * trials.Config().repetitions
ARUCO_DICT = "DICT_4X4_50"
FIT_FRAMES = 16                 # frames (67 ms at 240 fps) of the cubic fits before and after contact
STATIC_PX = 0.6                 # corners within this of the first frame: the pendulum still held


# ── mass properties (numpy only: also used inside Blender) ─────────────────
def mass_properties(co, tris, rho, origin=(0.0, 0.0, 0.0)):
    """Mass (g), centre of mass and second moments S = integral of x x^T dm (g mm^2, about origin) of closed
    triangle meshes: co (N x 3, mm), tris (T x 3 vertex indices, outward winding), rho (g / cm^3, one value or one
    per triangle). A mesh wound inside out comes out with a negative mass, which the caller flips."""
    co = np.asarray(co, np.float64) - np.asarray(origin, np.float64)
    tris = np.asarray(tris, np.int64)
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    w = vol * np.broadcast_to(np.asarray(rho, np.float64), vol.shape) / 1000.0
    m = float(w.sum())
    s = a + b + c
    com = (w[:, None] * s / 4.0).sum(0) / m if m else np.zeros(3)
    S = np.einsum("t,tij->ij", w / 20.0, np.einsum("ti,tj->tij", a, a) + np.einsum("ti,tj->tij", b, b)
                  + np.einsum("ti,tj->tij", c, c) + np.einsum("ti,tj->tij", s, s))
    return m, com, S


def inertia_about(S, axis):
    """Moment of inertia about an axis through the origin of S (unit vector axis)."""
    u = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    return float(np.trace(S) - u @ S @ u)


def density_of(material):
    """g / cm^3 of a material of the scene by its name (petg_white -> petg); None when unknown."""
    name = (material or "").lower()
    if name in DENSITY:
        return DENSITY[name]
    return next((v for k, v in DENSITY.items() if name.startswith(k + "_") or name.startswith(k)), None)


# ── inside Blender ──────────────────────────────────────────────────────────
def blender_model():
    """The pendulum and the square holder P1 read from the open scene (runs inside Blender); a dict of plain lists."""
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    hidden = set()

    def walk(coll, off):
        off = off or coll.hide_render
        if off:
            hidden.add(coll.name)
        for ch in coll.children:
            walk(ch, off)
    walk(bpy.context.scene.collection, False)

    def shown(ob):
        return not ob.hide_render and not any(c.name in hidden for c in ob.users_collection)

    def mesh_of(ob):
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        Mw = np.array(ob.matrix_world, np.float64)
        co = co.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
        tv = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tv)
        tm = np.empty(len(me.loop_triangles), np.int64)
        me.loop_triangles.foreach_get("material_index", tm)
        mats = [m.name if m else "" for m in me.materials]
        ev.to_mesh_clear()
        return co, tv.reshape(-1, 3), tm, mats

    def below(ob):
        out = []
        for ch in ob.children:
            out += [ch] + below(ch)
        return out

    sw = bpy.data.objects[SWING]
    Ms = np.array(sw.matrix_world, np.float64)
    pivot, axis = Ms[:3, 3].copy(), Ms[:3, 0] / np.linalg.norm(Ms[:3, 0])
    cap_ob = bpy.data.objects.get(P1SQ["cap"]) or bpy.data.objects["marker_strike_cap"]
    cco = mesh_of(cap_ob)[0]
    cap_c = cco.mean(0)                             # on the cap's axis (a solid of revolution)
    parts, unknown = [], []
    m_tot, first, S_tot = 0.0, np.zeros(3), np.zeros((3, 3))
    for ob in below(sw):
        if ob.type != "MESH" or "__" in ob.name or not shown(ob):
            continue
        co, tv, tm, mats = mesh_of(ob)
        if not len(tv):
            continue
        rho = np.array([density_of(mats[i] if i < len(mats) else "") or np.nan for i in tm])
        if np.isnan(rho).any():
            unknown.append(ob.name)
            rho = np.nan_to_num(rho, nan=DENSITY["steel"])
        m, com, S = mass_properties(co, tv, rho, pivot)
        if m < 0:                                   # wound inside out
            m, S = -m, -S
        parts.append({"name": ob.name, "materials": sorted({mats[i] for i in tm if i < len(mats)}),
                      "mass_g": round(m, 3), "com_mm": [round(float(x), 3) for x in com]})
        m_tot += m
        first += m * com
        S_tot += S
    com = first / m_tot
    # h: horizontal, across the axle, towards the cap (the marker); z: up
    z = np.array([0.0, 0.0, 1.0])
    to_cap = cap_c - pivot
    h = to_cap - (to_cap @ axis) * axis - (to_cap @ z) * z
    h /= np.linalg.norm(h)
    co, tv, _, _ = mesh_of(bpy.data.objects[TIP])
    rel = co - pivot
    apex = co[np.argmax(rel @ h)]                   # the nose's point that touches the cap
    # the contact: the scene's strike point (the bench writes it), checked against the nose; a point of the scene
    # more than STRIKE_TOL_MM off the nose is not the contact of the parts as they stand, and the nose is used
    sp = bpy.context.scene.get("strike_point")
    sp = np.array(list(sp), np.float64) if sp is not None else None
    gap = float(np.linalg.norm(sp - apex)) if sp is not None else None
    contact, source = (sp, "scene") if gap is not None and gap <= STRIKE_TOL_MM else (apex, "tip")
    ra = contact - pivot
    r_impact = float(np.linalg.norm(ra - (ra @ axis) * axis))
    washer = None
    if WASHER in bpy.data.objects:
        wco, wtv, wtm, wmats = mesh_of(bpy.data.objects[WASHER])
        washer = abs(mass_properties(wco, wtv, [density_of(wmats[i]) or DENSITY["steel"] for i in wtm])[0])
    model = {"pivot_W": pivot.tolist(), "axis_W": axis.tolist(), "h_W": h.tolist(), "tip_apex_W": apex.tolist(),
             "contact_W": contact.tolist(), "contact_from": source, "mass_g": m_tot, "com_W": (pivot + com).tolist(),
             "c_h_mm": float(com @ h), "c_z_mm": float(com @ z), "c_axis_mm": float(com @ axis),
             "inertia_g_mm2": inertia_about(S_tot, axis), "impact_radius_mm": r_impact,
             "washer_g": washer, "parts": parts, "unknown_density": unknown,
             "scene_strike_point_W": sp.tolist() if sp is not None else None, "scene_strike_to_nose_mm": gap}
    model["holder"] = holder_p1sq(bpy, mesh_of, contact)
    if model["holder"] is not None:
        model["holder"].update(contact_from=source, scene_strike_to_nose_mm=gap)
    return model


def holder_p1sq(bpy, mesh_of, apex):
    """The board frame M of the square P1's label, and in M its cap (the point of its axis at the height the nose
    hits, the contact point, centroid, axis, ends) and spheres."""
    obs = bpy.data.objects
    if P1SQ["label"] not in obs or P1SQ["cap"] not in obs:
        return None
    lab = obs[P1SQ["label"]]
    me = lab.data
    uvl = me.uv_layers.active
    Mw = np.array(lab.matrix_world, np.float64)
    co = np.array([v.co for v in me.vertices], np.float64) @ Mw[:3, :3].T + Mw[:3, 3]
    P, UV = [], []
    for loop in me.loops:
        P.append(co[loop.vertex_index])
        UV.append(uvl.data[loop.index].uv[:])
    P, UV = np.array(P), np.array(UV, np.float64)
    A = np.column_stack([np.ones(len(UV)), UV])               # p = O + U u + V v (u right, v up the image)
    sol, *_ = np.linalg.lstsq(A, P, rcond=None)
    O, U, V = sol
    fit_mm = float(np.abs(A @ sol - P).max())
    size = float(np.linalg.norm(U))
    m = (size - BOARD_MM) / 2.0
    origin = O + U * (m / size) + V * (1.0 - m / size)        # top left of the printed squares
    ex, ey = U / size, -V / np.linalg.norm(V)
    ez = np.cross(ex, ey)
    R = np.column_stack([ex, ey, ez])
    T_W_M = np.eye(4)
    T_W_M[:3, :3], T_W_M[:3, 3] = R, origin

    def in_M(p):
        return (R.T @ (np.asarray(p, np.float64) - origin)).tolist()

    cap = obs[P1SQ["cap"]]
    cco, ctv, ctm, cmats = mesh_of(cap)
    cm, ccom, _ = mass_properties(cco, ctv, 1.0)
    caxis = np.array(cap.matrix_world, np.float64)[:3, 2]
    caxis /= np.linalg.norm(caxis)
    centre = cco.mean(0)
    along = (cco - centre) @ caxis
    radial = np.linalg.norm((cco - centre) - np.outer(along, caxis), axis=1)
    strike = centre + caxis * ((np.asarray(apex) - centre) @ caxis)     # on the axis, at the nose's height
    contact_off = float(np.linalg.norm(np.cross(np.asarray(apex) - centre, caxis)))   # the cap's radius if touching
    holder = obs.get(P1SQ["holder"])
    behind = None
    if holder is not None:
        hco = mesh_of(holder)[0]
        behind = float((hco.mean(0) - origin) @ ez)
    spheres = []
    for name in P1SQ["spheres"]:
        if name in obs:
            sco = mesh_of(obs[name])[0]
            spheres.append(in_M((sco.min(0) + sco.max(0)) / 2))
    return {"T_W_M": T_W_M.tolist(), "label_size_mm": size, "uv_fit_mm": fit_mm, "orthogonality": float(abs(ex @ ey)),
            "holder_behind_mm": behind, "strike_point_M": in_M(strike), "contact_point_M": in_M(apex),
            "contact_from_axis_mm": contact_off,
            "cap_centroid_M": in_M(ccom) if cm else in_M(centre),
            "cap_axis_M": (R.T @ caxis).tolist(), "cap_diameter_mm": float(2 * radial.max()),
            "cap_top_M": in_M(centre + caxis * along.max()), "cap_bottom_M": in_M(centre + caxis * along.min()),
            "spheres_M": spheres}


# ── the model: Blender on a copy of the scene ───────────────────────────────
def refresh_copy(src: Path, dst: Path) -> str:
    """The scene copied when it has not been saved for STABLE_S seconds and is newer than the copy."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return "up to date"
    age = time.time() - src.stat().st_mtime
    if age < STABLE_S:
        if dst.exists():
            return f"scene saved {age:.0f} s ago: the older copy kept"
        raise SystemExit(f"{src} saved {age:.0f} s ago: try again in two minutes")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def run_model(scene: Path | None = None, out: Path = PROTOCOL) -> dict:
    """Blender in the background on a copy of the scene; pendulum_model.json and marker_holder.json written."""
    work = WORK / "pilot"
    work.mkdir(parents=True, exist_ok=True)
    src = Path(scene) if scene else SCENE
    copy = work / "pilot_scene.blend"
    if scene:
        copy = src
        note = "given"
    else:
        note = refresh_copy(src, copy)
    raw = work / "model_raw.json"
    raw.unlink(missing_ok=True)
    cmd = [str(BLENDER), "--background", "--factory-startup", "--python-exit-code", "1", str(copy), "--python",
           str(Path(__file__).resolve()), "--", "blender-model", str(raw)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode or not raw.exists():
        raise SystemExit("Blender failed:\n" + res.stdout[-3000:] + res.stderr[-3000:])
    m = json.loads(raw.read_text(encoding="utf-8"))
    saved = datetime.fromtimestamp(src.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    stamp = {"scene": src.name, "scene_saved": saved, "written": trials.now(), "copy": note}
    model = {**stamp, **derive(m)}
    holder = m.pop("holder", None)
    out.mkdir(parents=True, exist_ok=True)
    trials.write_text_atomic(out / MODEL_JSON.name, json.dumps(model, indent=1) + "\n")
    if holder:
        cap = {**stamp, "holder": "square P1 (marker_alt_p1sq_plate52)",
               "frame": "M: ChArUco board, origin at the outer corner of the first square, x along the printed "
                        "columns, y along the rows, z into the board (mm)",
               "point": "strike_point_M: on the cap's axis at the height the pendulum's nose hits; the 3 mm rule "
                        "measures how far it moved", **holder}
        trials.write_text_atomic(out / CAP_JSON.name, json.dumps(cap, indent=1) + "\n")
    return model


def derive(m: dict) -> dict:
    """The model's numbers from the raw read of the scene: distance and angle of the centre of mass, period, effective
    mass at the impact point."""
    M, I, r = m["mass_g"], m["inertia_g_mm2"], m["impact_radius_mm"]
    d = math.hypot(m["c_h_mm"], m["c_z_mm"])
    out = {k: v for k, v in m.items() if k != "holder"}
    out.update(com_mm=d, com_angle_deg=math.degrees(math.atan2(m["c_h_mm"], -m["c_z_mm"])),
               period_s=2 * math.pi * math.sqrt(I / (M * G_MM * d)), effective_mass_g=I / r ** 2,
               washers_per_side=trials.Config().washers_per_side)
    return out


def load_model(path: Path = MODEL_JSON) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── energies ────────────────────────────────────────────────────────────────
def rise_mm(model: dict, theta_deg) -> np.ndarray:
    """Rise of the centre of mass when the arm is raised by theta from contact, away from the marker."""
    t = np.radians(theta_deg)
    ch, cz = model["c_h_mm"], model["c_z_mm"]
    return cz * np.cos(t) - ch * np.sin(t) - cz


def energy_mj(model: dict, theta_deg, mass_g: float | None = None) -> np.ndarray:
    """Energy (mJ) of a release from theta; with a weighed mass the model is scaled to it."""
    M = mass_g or model["mass_g"]
    return M * G_MM * rise_mm(model, theta_deg) * 1e-6


def inertia(model: dict, mass_g: float | None = None) -> float:
    """g mm^2 about the axle; scaled with a weighed mass (the same parts, another density)."""
    return model["inertia_g_mm2"] * ((mass_g or model["mass_g"]) / model["mass_g"])


def contact_omega(model: dict, theta_deg, mass_g: float | None = None) -> np.ndarray:
    """rad / s at contact after a release from rest at theta (no friction in the axle)."""
    return np.sqrt(2 * energy_mj(model, theta_deg, mass_g) * 1e6 / inertia(model, mass_g))


def energy_rows(model: dict) -> list[dict]:
    rows = []
    for a in HOLES_DEG:
        E = float(energy_mj(model, a))
        w = float(contact_omega(model, a))
        rows.append({"key": f"hole_{a}", "angle_deg": a, "energy_mj": f"{E:.1f}", "omega_rad_s": f"{w:.2f}",
                     "speed_m_s": f"{w * model['impact_radius_mm'] / 1000:.2f}",
                     "pliers_drop_mm": f"{E / (PLIERS_G * G_MM * 1e-6):.0f}",
                     **{f"marker_{d}_mj": f"{E * f:.1f}" for d, f in TRANSFER.items()}})
    return rows


def threshold_hole(model: dict, energy: float) -> int | None:
    """The first hole whose energy reaches `energy` (mJ), None if none does."""
    return next((a for a in HOLES_DEG if energy_mj(model, a) >= energy), None)


def model_rows(model: dict) -> list[dict]:
    need = {d: TURN_MJ / f for d, f in TRANSFER.items()}
    rows = [("mass_g", f"{model['mass_g']:.1f}", "g"), ("washer_g", f"{model['washer_g'] or 0:.1f}", "g"),
            ("washers_per_side", model["washers_per_side"], ""), ("com_mm", f"{model['com_mm']:.1f}", "mm"),
            ("com_angle_deg", f"{model['com_angle_deg']:.1f}", "deg"),
            ("inertia_g_cm2", f"{model['inertia_g_mm2'] / 100:.0f}", "g cm2"),
            ("period_s", f"{model['period_s']:.3f}", "s"), ("impact_radius_mm", f"{model['impact_radius_mm']:.1f}", "mm"),
            ("effective_mass_g", f"{model['effective_mass_g']:.1f}", "g"), ("turn_marker_mj", f"{TURN_MJ:.0f}", "mJ")]
    for d, e in need.items():
        rows.append((f"turn_pendulum_{d}_mj", f"{e:.0f}", "mJ"))
        rows.append((f"turn_hole_{d}_deg", threshold_hole(model, e) or "", "deg"))
    return [{"key": k, "value": v, "unit": u} for k, v, u in rows]


def write_energy(model: dict, out: Path = RESULTS) -> list[Path]:
    paths = [out / "pendulum_model.csv", out / "pendulum_energy.csv"]
    trials.write_csv(paths[0], model_rows(model), ("key", "value", "unit"))
    rows = energy_rows(model)
    trials.write_csv(paths[1], rows, list(rows[0]))
    return paths


# ── the side video ──────────────────────────────────────────────────────────
def video_fps(path, fps: float | None = None) -> tuple[float, str]:
    """The capture rate of a video: given, an Android slow motion's capture rate, else the container's."""
    if fps:
        return float(fps), "given"
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        meta = {**dict(c.metadata or {}), **dict(s.metadata or {})}
        for k, v in meta.items():
            if "capture.fps" in k.lower() or k.lower() == "capture_fps":
                try:
                    return float(v), "capture"
                except ValueError:
                    pass
        rate = s.average_rate or s.guessed_rate or s.base_rate
    return float(rate or 0.0), "container"


def aruco_detector():
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 3
    return cv2.aruco.ArucoDetector(d, p)


def refine_corners(gray: np.ndarray, c: np.ndarray, samples: int = 14) -> np.ndarray:
    """A sticker's corners again, as the meeting points of its four outer edges: along each edge the crossing of the
    middle grey between the black border and the white margin, found to a fraction of a pixel on the profiles across
    it, and a straight line through them. The detector's own corners are pulled inwards by a few tenths of a pixel,
    differently at each turn; a crossing of the middle grey stays where the edge is, blurred or not."""
    from scipy.ndimage import map_coordinates
    g = gray.astype(np.float32)
    side = float(np.mean(np.linalg.norm(c - np.roll(c, -1, 0), axis=1)))
    reach = max(1.5, min(3.0, 0.4 * side / 6))                    # within the one-cell black border
    steps = np.linspace(-reach, reach, int(reach * 8) + 1)
    centre = c.mean(0)
    lines = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-d[1], d[0]])
        if nrm @ ((a + b) / 2 - centre) < 0:
            nrm = -nrm                                            # outwards
        pts = []
        for t in np.linspace(0.18, 0.82, samples):
            p = a + t * (b - a)
            q = p[None, :] + steps[:, None] * nrm[None, :]
            prof = map_coordinates(g, [q[:, 1], q[:, 0]], order=1, mode="nearest")
            lo, hi = prof[: len(prof) // 3].mean(), prof[-(len(prof) // 3):].mean()
            if hi - lo < 20:
                continue
            mid = (lo + hi) / 2
            j = np.nonzero((prof[:-1] < mid) & (prof[1:] >= mid))[0]
            if not len(j):
                continue
            j = j[np.argmin(np.abs(steps[j]))]
            s = steps[j] + (mid - prof[j]) / (prof[j + 1] - prof[j]) * (steps[1] - steps[0])
            pts.append(p + s * nrm)
        if len(pts) < 4:
            return c
        pts = np.array(pts)
        m = pts.mean(0)
        u = np.linalg.svd(pts - m)[2][0]
        lines.append((m, u))
    out = np.empty_like(c)
    for i in range(4):
        (p1, u1), (p2, u2) = lines[i - 1], lines[i]
        A = np.column_stack([u1, -u2])
        if abs(np.linalg.det(A)) < 1e-6:
            return c
        s = np.linalg.solve(A, p2 - p1)
        out[i] = p1 + s[0] * u1
    return out if np.abs(out - c).max() < 3.0 else c


def detect_stickers(path) -> tuple[dict, int, tuple]:
    """{id: {frame: corners 4 x 2}} of the stickers in every frame (refine_corners), the number of frames and the
    frame size."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    det = aruco_detector()
    out, k, size = {}, 0, (0, 0)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners, ids, _ = det.detectMarkers(gray)
        for c, i in zip(corners or (), (ids.ravel() if ids is not None else ())):
            out.setdefault(int(i), {})[k] = refine_corners(gray, c.reshape(4, 2).astype(np.float64))
        k += 1
    cap.release()
    return out, k, size


def rigid2d(A, B):
    """Rotation angle (rad) and translation of the 2D rigid motion taking points A onto B."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    ang = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    return ang, cb - R @ ca


SQUARE = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])


def still_run(frames: dict) -> list:
    """The frames from the start in which the sticker has not moved (the pendulum held before the release)."""
    keys = sorted(frames)
    first = frames[keys[0]]
    run = []
    for k in keys:
        if np.abs(frames[k] - first).max() <= STATIC_PX and (not run or k - run[-1] <= 2):
            run.append(k)
        else:
            break
    return run


def sticker_angles(frames: dict) -> tuple[dict, dict]:
    """{frame: polar angle (rad) of the sticker's centre about the axle, in the swing plane} and the fit's numbers.

    The swing plane, the axle on it and the sticker are fitted together over every frame (bundle adjustment): the
    plane's homography H (unit: the distance of the sticker's centre from the axle), the sticker's size s and mounting
    angle b and one angle a_k per frame, so that H ((cos a_k, sin a_k) + s R(a_k + b) square) meets the detected
    corners. A homography from one small square alone, carried across the whole arc, is off by degrees; a free size
    takes up the shrink of the detected corners (a few tenths of a pixel), which would otherwise scale the angles."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    keys = sorted(frames)
    X = np.stack([frames[k] for k in keys])                         # N x 4 x 2
    run = still_run(frames)
    ref = np.median(np.stack([frames[k] for k in run]), 0) if run else X[0]
    H0 = cv2.getPerspectiveTransform(SQUARE.astype(np.float32), ref.astype(np.float32)).astype(np.float64)
    Hi = np.linalg.inv(H0)
    C = cv2.perspectiveTransform(X.mean(1).reshape(-1, 1, 2), Hi).reshape(-1, 2)
    A = np.column_stack([C, np.ones(len(C))])                     # circle through the centres (Kasa)
    sol = np.linalg.lstsq(A, (C ** 2).sum(1), rcond=None)[0]
    c0 = sol[:2] / 2
    r0 = math.sqrt(max(sol[2] + c0 @ c0, 1e-9))
    a0 = np.unwrap(np.arctan2(C[:, 1] - c0[1], C[:, 0] - c0[0]))
    if np.ptp(a0) < math.radians(2.0):
        raise ValueError("the sticker barely moves")
    # the sticker's own turn in the first frame against its polar angle
    p_ref = cv2.perspectiveTransform(X[0].reshape(-1, 1, 2), Hi).reshape(-1, 2)
    b0 = rigid2d(SQUARE, p_ref - C[0])[0] - a0[0]
    # plane frame with the axle at the origin and the sticker's centre at distance 1
    Hq = H0 @ np.array([[r0, 0, c0[0]], [0, r0, c0[1]], [0, 0, 1.0]])
    Hq /= Hq[2, 2]
    n = len(keys)

    def model_px(x):
        H = np.append(x[:8], 1.0).reshape(3, 3)
        size, b, a = x[8], x[9], np.concatenate([[a0[0]], x[10:]])
        cen = np.column_stack([np.cos(a), np.sin(a)])
        ca, sa = np.cos(a + b) * size, np.sin(a + b) * size
        P = cen[:, None, :] + np.stack([ca[:, None] * SQUARE[:, 0] - sa[:, None] * SQUARE[:, 1],
                                        sa[:, None] * SQUARE[:, 0] + ca[:, None] * SQUARE[:, 1]], -1)
        q = P @ H[:, :2].T + H[:, 2]
        return q[..., :2] / q[..., 2:]

    def resid(x):
        return (model_px(x) - X).ravel()
    x0 = np.concatenate([Hq.ravel()[:8], [1.0 / r0, b0], a0[1:]])
    S = lil_matrix((n * 8, len(x0)), dtype=int)
    S[:, :10] = 1
    for k in range(1, n):
        S[k * 8:(k + 1) * 8, 9 + k] = 1
    fit = least_squares(resid, x0, jac_sparsity=S, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=200)
    a = np.concatenate([[a0[0]], fit.x[10:]])
    res = np.abs(fit.fun.reshape(n, 8)).max(1)
    return dict(zip(keys, a)), {"still": len(run), "r_sides": float(1.0 / fit.x[8]), "rms_px": float(np.sqrt(np.mean(
        fit.fun ** 2))), "worst_px": float(res.max())}


def _fit(t, y, t0, deg=2):
    """Polynomial through (t, y) about t0: coefficients of y = sum c_i (t - t0)^i, i = 0..deg."""
    return np.polynomial.polynomial.polyfit(t - t0, y, deg)


def swing_events(t: np.ndarray, psi: np.ndarray, n_fit: int = FIT_FRAMES) -> dict:
    """Release, contact and rebound of one swing from the turn psi(t) (rad), positive towards the marker, zero at the
    frames before the release. Contact: where the cubic fits of the last frames before it and of the first after
    it meet; the descent's slope there is the speed of the impact."""
    P = np.polynomial.polynomial
    k_top = int(np.argmax(psi))
    top = psi[k_top]
    # the first time the swing reaches its top: the contact (a second hit comes later and no higher)
    k_c = int(np.argmax(psi >= top - math.radians(0.3)))
    w = slice(max(k_c - 3, 0), k_c + 4)
    k_c = w.start + int(np.argmax(psi[w]))
    i_d = np.arange(max(k_c - n_fit, 0), k_c)
    i_r = np.arange(k_c + 1, min(k_c + 1 + n_fit, len(t)))
    if len(i_d) < 5 or len(i_r) < 5:
        raise ValueError("too few frames around the contact")
    tc0 = t[k_c]
    cd, cr = _fit(t[i_d], psi[i_d], tc0, 3), _fit(t[i_r], psi[i_r], tc0, 3)
    dt = (t[1] - t[0]) if len(t) > 1 else 0.0
    roots = P.polyroots(P.polysub(cd, cr))
    real = [float(r.real) for r in np.atleast_1d(roots) if abs(r.imag) < 1e-9
            and t[i_d[-1]] - tc0 - dt <= r.real <= t[i_r[0]] - tc0 + dt]
    tau = min(real, key=abs) if real else 0.0
    psi_c = float(P.polyval(tau, cd))
    w_c = float(P.polyval(tau, P.polyder(cd)))
    w_a = float(P.polyval(tau, P.polyder(cr)))
    still = psi[: max(k_c // 4, 1)]
    noise = float(np.std(still)) if len(still) > 3 else 0.0
    moving = np.nonzero(psi[:k_c] > max(5 * noise, math.radians(0.2)))[0]
    k_r = int(moving[0]) - 1 if len(moving) else 0
    # rebound: the first turn back after the contact
    after = psi[k_c + 1:]
    k_b = None
    if len(after) > 5:
        d = np.diff(after)
        up = (d[:-2] > 0) & (d[1:-1] > 0) & (d[2:] > 0)          # rising for three frames: past the top
        k_b = k_c + 1 + int(np.argmax(up)) if up.any() else k_c + 1 + int(np.argmin(after))
    psi_b, caught = None, None
    if k_b is not None:
        j = np.arange(max(k_b - 6, k_c + 1), min(k_b + 7, len(t)))
        if len(j) >= 5:
            cb = _fit(t[j], psi[j], t[k_b])
            psi_b = cb[0] - cb[1] ** 2 / (4 * cb[2]) if cb[2] > 0 else float(psi[k_b])
            caught = bool(cb[2] <= 0)
        else:
            psi_b = float(psi[k_b])
    return {"t_release_s": float(t[max(k_r, 0)]), "t_contact_s": float(tc0 + tau), "psi_contact": float(psi_c),
            "omega_contact": float(w_c), "omega_after": float(w_a), "psi_rebound": psi_b, "rebound_caught": caught,
            "noise_deg": math.degrees(noise), "frames_fit": [int(len(i_d)), int(len(i_r))]}


def track_video(path, fps: float | None = None, model: dict | None = None, mass_g: float | None = None) -> dict:
    """One side video: release angle, angular speed and tip speed at contact (and what the model predicts),
    rebound angle, apparent restitution, energies."""
    rate, how = video_fps(path, fps)
    if rate <= 0:
        raise ValueError(f"{Path(path).name}: no frame rate")
    dets, n_frames, size = detect_stickers(path)
    stickers = {i: f for i, f in dets.items() if len(f) >= 0.3 * n_frames}
    if not stickers:
        raise ValueError(f"{Path(path).name}: no sticker found in a third of the frames")
    per = {}
    for i, f in stickers.items():
        try:
            per[i] = sticker_angles(f)
        except ValueError:
            continue
    if not per:
        raise ValueError(f"{Path(path).name}: no sticker moves along an arc")
    # the sticker farthest from the axle (the most precise angle) among those seen in nine frames out of ten,
    # else the one seen most
    seen = {i: len(a) / n_frames for i, (a, _) in per.items()}
    good = [i for i in per if seen[i] >= 0.9 * max(seen.values())]

    def radius_px(i):
        c = stickers[i][min(stickers[i])]
        return per[i][1]["r_sides"] * float(np.linalg.norm(c[1] - c[0]))
    use = max(good, key=radius_px)
    ang, info = per[use]
    frames = sorted(ang)
    a = np.array([ang[k] for k in frames])
    turn = a - np.median(a[:max(info["still"], 1)])
    t = np.array(frames, np.float64) / rate
    sign = 1.0 if turn[np.argmax(np.abs(turn))] > 0 else -1.0
    ev = swing_events(t, sign * turn)
    out = {"file": Path(path).name, "fps": rate, "fps_from": how, "frames": n_frames, "size": list(size),
           "stickers": sorted(stickers), "sticker": use, "still_frames": info["still"],
           "fit_rms_px": round(info["rms_px"], 3), "seen": round(len(frames) / max(n_frames, 1), 3), **ev}
    out["release_deg"] = math.degrees(ev["psi_contact"])
    out["rebound_deg"] = None if ev["psi_rebound"] is None else math.degrees(ev["psi_contact"] - ev["psi_rebound"])
    out["omega_rad_s"] = abs(ev["omega_contact"])
    out["restitution"] = abs(ev["omega_after"]) / abs(ev["omega_contact"]) if ev["omega_contact"] else None
    if model:
        r = model["impact_radius_mm"]
        I = inertia(model, mass_g)
        out["speed_m_s"] = out["omega_rad_s"] * r / 1000
        out["speed_model_m_s"] = float(contact_omega(model, out["release_deg"], mass_g)) * r / 1000
        out["speed_ratio"] = out["speed_m_s"] / out["speed_model_m_s"]
        out["energy_release_mj"] = float(energy_mj(model, out["release_deg"], mass_g))
        out["energy_contact_mj"] = 0.5 * I * out["omega_rad_s"] ** 2 * 1e-6
        if out["rebound_deg"] is not None:
            out["energy_rebound_mj"] = float(energy_mj(model, out["rebound_deg"], mass_g))
            out["energy_left_mj"] = out["energy_contact_mj"] - out["energy_rebound_mj"]
    return out


# ── slip threshold and power ─────────────────────────────────────────────────
def hinge_fit(E, rot, grid: int = 200) -> dict:
    """rot = k max(0, E - E0): E0 and k by least squares (k >= 0), over a grid of E0 and then a finer one about the
    best."""
    E, rot = np.asarray(E, np.float64), np.asarray(rot, np.float64)
    if len(E) < 3 or np.ptp(E) <= 0:
        return {"e0": None, "k": None, "rms": None}

    def best_of(cands, best):
        for e0 in cands:
            x = np.maximum(E - e0, 0.0)
            xx = x @ x
            if xx <= 0:
                continue
            k = max(float(x @ rot) / xx, 0.0)
            sse = float(np.sum((rot - k * x) ** 2))
            if sse < best[0] - 1e-15:
                best = (sse, e0, k)
        return best
    step = E.max() / grid
    best = best_of(np.linspace(0.0, E.max(), grid + 1), (np.inf, None, None))
    if best[1] is not None:
        best = best_of(np.linspace(max(best[1] - step, 0.0), best[1] + step, 41), best)
    sse, e0, k = best
    return {"e0": e0, "k": k, "rms": math.sqrt(sse / len(E)) if e0 is not None else None}


def slip_threshold(E, rot, n_boot: int = 1000, seed: int = 0, turn_deg: float = 5.0) -> dict:
    """Onset of slip (E0), the energy that turns by turn_deg (E5) and 95 % bootstrap intervals."""
    E, rot = np.asarray(E, np.float64), np.asarray(rot, np.float64)
    fit = hinge_fit(E, rot)
    out = {"n": len(E), **fit, "e5": None, "e0_lo": None, "e0_hi": None, "e5_lo": None, "e5_hi": None}
    if fit["e0"] is None or not fit["k"]:
        return out
    out["e5"] = fit["e0"] + turn_deg / fit["k"]
    rng = np.random.default_rng(seed)
    e0s, e5s = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(E), len(E))
        f = hinge_fit(E[i], rot[i], grid=100)
        if f["e0"] is not None and f["k"]:
            e0s.append(f["e0"])
            e5s.append(f["e0"] + turn_deg / f["k"])
    if e0s:
        out["e0_lo"], out["e0_hi"] = np.percentile(e0s, [2.5, 97.5]).tolist()
        out["e5_lo"], out["e5_hi"] = np.percentile(e5s, [2.5, 97.5]).tolist()
    return out


def signed_rank_p(D: np.ndarray) -> np.ndarray:
    """Two-sided p of the Wilcoxon signed-rank test for each row of D (normal approximation, ties averaged)."""
    from scipy.stats import norm, rankdata
    n = D.shape[1]
    ranks = rankdata(np.abs(D), axis=1)
    w = np.where(D > 0, ranks, 0.0).sum(1) + 0.5 * np.where(D == 0, ranks, 0.0).sum(1)
    mu, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return 2 * norm.sf(np.abs(w - mu) / sd)


def power_at(diffs, n: int, delta: float = DELTA_MM, sims: int = 2000, seed: int = 0,
             alpha: float = ALPHA / COMPARISONS) -> float:
    """Monte Carlo power of the signed-rank test to see a shift of delta with n differences drawn from diffs
    (centred on their mean)."""
    d = np.asarray(diffs, np.float64)
    d = d - d.mean()
    rng = np.random.default_rng(seed)
    D = rng.choice(d, (sims, n)) + delta
    return float(np.mean(signed_rank_p(D) < alpha))


def n_needed(diffs, delta: float = DELTA_MM, power: float = POWER, n_max: int = 400, **kw) -> int | None:
    """Fewest differences that reach the power (None beyond n_max)."""
    lo, hi = 6, n_max
    if power_at(diffs, hi, delta, **kw) < power:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if power_at(diffs, mid, delta, **kw) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def comparison_diffs(groups: dict, rng: np.random.Generator, n_pairs: int = 2000) -> dict:
    """The differences of each planned comparison: paired by trial where both values belong to the same impacts,
    else matched by cell (cell present in both groups), else drawn at random from the two groups."""
    plan = (("a_vs_a_ai", "A", "A+AI", "trial"), ("b_vs_b_ai", "B", "B+AI", "trial"),
            ("b_vs_a_ai", "B", "A+AI", "cell"), ("a_vs_b", "A", "B", "cell"))
    out = {}
    for key, x, y, how in plan:
        X, Y = groups.get(x) or {}, groups.get(y) or {}
        if not X or not Y:
            out[key] = (None, "no data")
            continue
        if how == "trial":
            common = sorted(set(X["trial"]) & set(Y["trial"]))
            if len(common) >= 5:
                xi = dict(zip(X["trial"], X["tre"]))
                yi = dict(zip(Y["trial"], Y["tre"]))
                out[key] = (np.array([xi[t] - yi[t] for t in common]), "paired")
                continue
        cx, cy = {}, {}
        for c, v in zip(X["cell"], X["tre"]):
            cx.setdefault(c, []).append(v)
        for c, v in zip(Y["cell"], Y["tre"]):
            cy.setdefault(c, []).append(v)
        cells = sorted(set(cx) & set(cy))
        if len(cells) >= 5:
            out[key] = (np.array([np.mean(cx[c]) - np.mean(cy[c]) for c in cells]), "cells")
        else:
            out[key] = (rng.choice(np.asarray(X["tre"]), n_pairs) - rng.choice(np.asarray(Y["tre"]), n_pairs), "random")
    return out


def power_rows(groups: dict, sims: int = 2000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for key, (d, how) in comparison_diffs(groups, rng).items():
        if d is None:
            rows.append({"key": key, "pairing": how})
            continue
        n = n_needed(d, sims=sims, seed=seed)
        rows.append({"key": key, "pairing": how, "n_pilot": len(d) if how != "random" else "",
                     "sd_diff_mm": f"{np.std(d, ddof=1):.3f}", "delta_mm": DELTA_MM,
                     "alpha": f"{ALPHA / COMPARISONS:.4f}", "planned_n": PLANNED_N,
                     "power_planned": f"{power_at(d, PLANNED_N, sims=sims, seed=seed):.3f}",
                     "n_needed": n if n is not None else "", "repetitions_needed": math.ceil(n / CELLS) if n else ""})
    return rows


# ── scanner exports ─────────────────────────────────────────────────────────
def ply_header(path) -> dict:
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            return {}
        out = {"format": "", "elements": {}, "properties": []}
        for line in f:
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format":
                out["format"] = tok[1]
            elif tok[0] == "element":
                out["elements"][tok[1]] = int(tok[2])
                cur = tok[1]
            elif tok[0] == "property" and cur == "vertex":
                out["properties"].append(tok[-1])
            elif tok[0] == "end_header":
                break
        return out


def scan_format(path) -> dict:
    """What a scanner export holds and whether ground_truth.py can measure it (codes: unreadable, empty, metres,
    spheres, rigs, ok)."""
    import ground_truth as gt
    path = Path(path)
    out = {"file": path.name, "suffix": path.suffix.lower(), "bytes": path.stat().st_size}
    if out["suffix"] == ".ply":
        h = ply_header(path)
        out["ply"] = h.get("format", "?")
        out["faces"] = h.get("elements", {}).get("face", 0)
        out["properties"] = " ".join(h.get("properties", []))
    try:
        pts = gt.load_cloud(path)
    except Exception as e:  # noqa: BLE001  any reader error is a file the scanner should export otherwise
        return {**out, "code": "unreadable", "detail": f"{type(e).__name__}: {e}"}
    out["points"] = int(len(pts))
    if not len(pts):
        return {**out, "code": "empty"}
    ext = float(np.ptp(pts, 0).max())
    out["extent_mm"] = round(ext, 2)
    if ext < 2.0:
        return {**out, "code": "metres"}
    try:
        res = gt.measure(pts, rigs=None, radii=None, T_B_mandible=np.eye(4))
    except ValueError as e:
        msg = str(e)
        return {**out, "code": "spheres" if "spheres found" in msg else "rigs", "detail": msg}
    out["spheres"] = len(res["spheres"])
    out["T_M_mandible"] = np.round(res["T_M_mandible"], 6).tolist()
    return {**out, "code": "ok"}


# ── the pilot's files ───────────────────────────────────────────────────────
def _num(v):
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


def analyse(raw: Path = RAW / "pilot", poses: Path = GT_POSES, model: dict | None = None,
            corrected: Path | None = None, out: Path = RESULTS, sims: int = 2000, log=print) -> dict:
    """The pilot: one row per trial (video and scans), per hole, per direction, per comparison; the tables in out."""
    model = model or load_model()
    rows = [r for r in trials.Log(raw).rows() if r["status"] == trials.OK]
    gt_rows = {r["trial"]: r for r in trials.read_csv(poses)} if Path(poses).exists() else {}
    fixed = {r["trial"]: _num(r.get("tre_after_mm")) for r in trials.read_csv(corrected)} if corrected else {}
    per = []
    for r in rows:
        t = r["trial"]
        row = {"trial": t, "holder": r["holder"], "angle_deg": r["angle_deg"], "direction_deg": r["direction_deg"],
               "swing_mass_g": r["swing_mass_g"]}
        mass = _num(r["swing_mass_g"])
        vids = sorted((raw / t).glob(f"{t}_highspeed.*")) if (raw / t).exists() else []
        if vids:
            try:
                v = track_video(vids[0], model=model, mass_g=mass)
                row.update({k: v.get(k) for k in ("fps", "release_deg", "omega_rad_s", "speed_m_s", "speed_model_m_s",
                                                  "speed_ratio", "rebound_deg", "restitution", "energy_release_mj",
                                                  "energy_contact_mj", "energy_rebound_mj", "energy_left_mj",
                                                  "noise_deg", "seen")})
                row["video"] = "ok"
            except (ValueError, OSError) as e:
                row["video"] = f"failed: {e}"
                log(f"{t}: video not measured ({e})")
        else:
            row["video"] = "none"
        g = gt_rows.get(t, {})
        row.update({k: g.get(k, "") for k in ("rot_deg", "shift_mm", "cap_shift_mm", "tre_mean_mm", "tre_max_mm")})
        if t in fixed:
            row["tre_after_mm"] = fixed[t]
        row["energy_nominal_mj"] = float(energy_mj(model, _num(r["angle_deg"]) or 0.0, mass))
        per.append(row)
    return summarise(per, model, out, sims)


def _fmt(v, nd=3):
    if v is None or v == "":
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return v


def summarise(per: list[dict], model: dict, out: Path, sims: int = 2000) -> dict:
    """Tables of the pilot from its per-trial rows."""
    out.mkdir(parents=True, exist_ok=True)
    cols = ["trial", "holder", "angle_deg", "direction_deg", "swing_mass_g", "video", "fps", "release_deg",
            "omega_rad_s", "speed_m_s", "speed_model_m_s", "speed_ratio", "rebound_deg", "restitution",
            "energy_nominal_mj", "energy_release_mj", "energy_contact_mj", "energy_rebound_mj", "energy_left_mj",
            "noise_deg", "seen", "rot_deg", "shift_mm", "cap_shift_mm", "tre_mean_mm", "tre_max_mm", "tre_after_mm"]
    trials.write_csv(out / "pilot_trials.csv", [{k: _fmt(r.get(k)) for k in cols} for r in per], cols)
    # per hole
    erows = []
    for a in HOLES_DEG:
        hs = [r for r in per if _num(r["angle_deg"]) == a and r.get("speed_m_s") is not None]
        v = np.array([r["speed_m_s"] for r in hs])
        e = {"key": f"hole_{a}", "angle_deg": a, "n": len(hs),
             "energy_model_mj": f"{float(energy_mj(model, a)):.1f}",
             "speed_model_m_s": f"{float(contact_omega(model, a)) * model['impact_radius_mm'] / 1000:.3f}"}
        if len(hs):
            e.update(speed_mean_m_s=f"{v.mean():.3f}", speed_sd_m_s=f"{v.std(ddof=1):.3f}" if len(v) > 1 else "",
                     speed_cv=f"{v.std(ddof=1) / v.mean():.3f}" if len(v) > 1 else "",
                     speed_ratio_mean=f"{np.mean([r['speed_ratio'] for r in hs]):.3f}",
                     release_mean_deg=f"{np.mean([r['release_deg'] for r in hs]):.2f}",
                     release_sd_deg=f"{np.std([r['release_deg'] for r in hs], ddof=1):.2f}" if len(hs) > 1 else "",
                     energy_contact_mean_mj=f"{np.mean([r['energy_contact_mj'] for r in hs]):.1f}",
                     restitution_mean=_fmt(np.mean([r["restitution"] for r in hs if r.get("restitution")] or [np.nan])))
        erows.append(e)
    ecols = ["key", "angle_deg", "n", "energy_model_mj", "speed_model_m_s", "speed_mean_m_s", "speed_sd_m_s",
             "speed_cv", "speed_ratio_mean", "release_mean_deg", "release_sd_deg", "energy_contact_mean_mj",
             "restitution_mean"]
    trials.write_csv(out / "pilot_energy.csv", erows, ecols)
    # slip threshold of A
    trows = []
    A = [r for r in per if r["holder"] == "A" and _num(r.get("rot_deg")) is not None]

    def energy_of(r):
        return r["energy_contact_mj"] if r.get("energy_contact_mj") else r["energy_nominal_mj"]
    for key, sel, scale in [(f"dir_{d}", [r for r in A if _num(r["direction_deg"]) == d], None) for d in DIRECTIONS_DEG] \
            + [("marker_all", A, "marker")]:
        if scale == "marker":
            E = [energy_of(r) * TRANSFER.get(int(_num(r["direction_deg"])), np.nan) for r in sel]
        else:
            E = [energy_of(r) for r in sel]
        s = slip_threshold(E, [_num(r["rot_deg"]) for r in sel])
        hole = threshold_hole(model, s["e5"]) if s["e5"] is not None and scale is None else None
        trows.append({"key": key, "n": s["n"], "energy": "marker" if scale else "pendulum",
                      **{k: _fmt(s[k], 2) for k in ("e0", "e0_lo", "e0_hi", "e5", "e5_lo", "e5_hi")},
                      "k_deg_per_mj": _fmt(s["k"], 4), "rms_deg": _fmt(s["rms"], 3),
                      "e5_expected_mj": f"{TURN_MJ / TRANSFER[int(key[4:])]:.1f}" if key.startswith("dir_") else
                      f"{TURN_MJ:.1f}", "hole_e5_deg": hole or ""})
    tcols = ["key", "n", "energy", "e0", "e0_lo", "e0_hi", "e5", "e5_lo", "e5_hi", "k_deg_per_mj", "rms_deg",
             "e5_expected_mj", "hole_e5_deg"]
    trials.write_csv(out / "pilot_threshold.csv", trows, tcols)
    # power
    groups = {}
    for r in per:
        cell = (r["angle_deg"], r["direction_deg"])
        tre = _num(r.get("tre_mean_mm"))
        if tre is not None:
            g = groups.setdefault(r["holder"], {"trial": [], "tre": [], "cell": []})
            g["trial"].append(r["trial"])
            g["tre"].append(tre)
            g["cell"].append(cell)
        aft = r.get("tre_after_mm")
        if aft is not None and aft != "":
            g = groups.setdefault(r["holder"] + "+AI", {"trial": [], "tre": [], "cell": []})
            g["trial"].append(r["trial"])
            g["tre"].append(float(aft))
            g["cell"].append(cell)
    prows = power_rows(groups, sims=sims)
    pcols = ["key", "pairing", "n_pilot", "sd_diff_mm", "delta_mm", "alpha", "planned_n", "power_planned", "n_needed",
             "repetitions_needed"]
    trials.write_csv(out / "pilot_power.csv", prows, pcols)
    # summary
    vids = [r for r in per if r.get("speed_ratio") is not None]
    summary = [("trials", len(per)), ("videos", len(vids)),
               ("speed_ratio_median", _fmt(np.median([r["speed_ratio"] for r in vids])) if vids else ""),
               ("speed_cv_median", _fmt(np.nanmedian([_num(e.get("speed_cv")) or np.nan for e in erows]))
                if any(e.get("speed_cv") for e in erows) else ""),
               ("restitution_median", _fmt(np.median([r["restitution"] for r in vids if r.get("restitution")]))
                if vids else ""),
               ("repetitions_needed", max([int(p["repetitions_needed"]) for p in prows if p.get("repetitions_needed")],
                                          default=""))]
    trials.write_csv(out / "pilot.csv", [{"key": k, "value": v} for k, v in summary], ("key", "value"))
    return {"trials": per, "energy": erows, "threshold": trows, "power": prows, "summary": dict(summary)}


# ── simulation ──────────────────────────────────────────────────────────────
class Sim:
    """A pendulum of the model filmed from the side at 240 fps, with stickers in its swing plane."""

    def __init__(self, model: dict, size=(1280, 720), fps=240.0, exposure_s=1 / 1000, sub=4, distance_mm=520.0,
                 yaw_deg=11.0, pitch_deg=-7.0, focal_px=1150.0, stickers=((3, 14.0, 13.0), (7, 20.0, 262.0)),
                 seed=0):
        import cv2
        self.m, self.size, self.fps, self.exp, self.sub = model, size, fps, exposure_s, sub
        self.rng = np.random.default_rng(seed)
        w, h = size
        K = np.array([[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1.0]])
        a, b = math.radians(yaw_deg), math.radians(pitch_deg)
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
        R = Rx @ Ry
        look = np.array([-60.0, 150.0])           # plane point at the image centre (mm: x towards the marker, y down)
        t = np.array([0, 0, distance_mm]) - R[:, :2] @ look
        self.H = K @ np.column_stack([R[:, 0], R[:, 1], t])
        bg = self.rng.normal(0, 1, (h // 8 + 1, w // 8 + 1)).astype(np.float32)
        bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 1.5), (w, h), interpolation=cv2.INTER_CUBIC)
        self.bg = np.clip(125 + 28 * bg / bg.std(), 40, 210).astype(np.uint8)
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
        self.stickers = []
        for sid, side, r in stickers:
            px = 120
            img = cv2.aruco.generateImageMarker(d, sid, px)
            pad = px // 6                          # white quiet zone of one cell
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            full = side * (px + 2 * pad) / px
            self.stickers.append((sid, side, r, img, full))
        I, M = model["inertia_g_mm2"], model["mass_g"]
        self.k = M * G_MM * math.hypot(model["c_h_mm"], model["c_z_mm"]) / I     # 1 / s^2
        self.alpha = math.radians(model.get("com_angle_deg", 0.0))

    def plane_to_image(self, theta):
        """Homography of the swing plane turned by theta (raised away from the marker) into the image."""
        c, s = math.cos(theta), math.sin(theta)
        Rp = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])      # x towards the marker, y down: raise = +theta
        return self.H @ Rp

    def draw(self, theta):
        import cv2
        img = self.bg.copy()
        H = self.plane_to_image(theta)

        def pt(x, y):
            p = H @ np.array([x, y, 1.0])
            return p[:2] / p[2]
        arm = np.array([pt(-4, 0), pt(4, 0), pt(4, 250), pt(-4, 250)], np.float32)
        cv2.fillConvexPoly(img, np.round(arm * 16).astype(np.int32), 175, cv2.LINE_AA, 4)
        block = np.array([pt(-15, 248), pt(15, 248), pt(15, 285), pt(-15, 285)], np.float32)
        cv2.fillConvexPoly(img, np.round(block * 16).astype(np.int32), 60, cv2.LINE_AA, 4)
        hub = np.array([pt(-9, -9), pt(9, -9), pt(9, 24), pt(-9, 24)], np.float32)
        cv2.fillConvexPoly(img, np.round(hub * 16).astype(np.int32), 70, cv2.LINE_AA, 4)
        for _, side, r, mk, full in self.stickers:
            n = mk.shape[0]
            S = np.array([[full / n, 0, -full / 2], [0, full / n, r - full / 2], [0, 0, 1.0]])
            Hm = H @ S
            q = np.array([pt(-full / 2, r - full / 2), pt(full / 2, r + full / 2), pt(full / 2, r - full / 2),
                          pt(-full / 2, r + full / 2)])
            x0, y0 = np.floor(q.min(0)).astype(int) - 2
            x1, y1 = np.ceil(q.max(0)).astype(int) + 2
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, self.size[0]), min(y1, self.size[1])
            if x1 <= x0 or y1 <= y0:
                continue
            T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
            roi = img[y0:y1, x0:x1].copy()
            img[y0:y1, x0:x1] = cv2.warpPerspective(mk, T @ Hm, (int(x1 - x0), int(y1 - y0)), dst=roi,
                                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        return img

    def swing(self, theta0_deg, restitution=0.5, damping=0.04, hold_s=0.25, catch_s=0.08, total_s=1.2):
        """theta(t) of a release from rest at theta0 (raised), contact at 0, rebound with the restitution, caught
        catch_s after the rebound's top; returns (t, theta) at 20 kHz and the truth."""
        dt = 5e-5
        th, w, t = math.radians(theta0_deg), 0.0, 0.0
        T, TH = [], []
        truth = {"release_deg": theta0_deg, "t_release_s": hold_s}
        hit, top, caught = False, None, None
        while t < total_s:
            T.append(t)
            TH.append(th)
            if t >= hold_s and caught is None:
                def acc(x, v):
                    return -self.k * (math.sin(x + self.alpha) - math.sin(self.alpha)) - damping * v
                k1 = (w, acc(th, w))
                k2 = (w + dt / 2 * k1[1], acc(th + dt / 2 * k1[0], w + dt / 2 * k1[1]))
                k3 = (w + dt / 2 * k2[1], acc(th + dt / 2 * k2[0], w + dt / 2 * k2[1]))
                k4 = (w + dt * k3[1], acc(th + dt * k3[0], w + dt * k3[1]))
                th2 = th + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                w2 = w + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                if not hit and th2 <= 0.0:
                    f = th / (th - th2)            # the contact inside the step
                    wc = w + f * (w2 - w)
                    truth.update(t_contact_s=t + f * dt, omega_rad_s=abs(wc), omega_after=abs(wc) * restitution)
                    th2, w2, hit = 0.0, -restitution * wc, True
                elif hit and top is None and w2 <= 0.0 < w:
                    top = th2
                    truth["rebound_deg"] = math.degrees(top)
                    truth["t_rebound_s"] = t
                if top is not None and caught is None and t >= truth["t_rebound_s"] + catch_s:
                    caught = th2
                th, w = th2, w2
            t += dt
        return np.array(T), np.array(TH), truth

    def video(self, path, theta0_deg, restitution=0.5, noise=2.0, total_s=1.1):
        """The side video of one release (mp4 at self.fps); returns the truth."""
        import cv2
        T, TH, truth = self.swing(theta0_deg, restitution, total_s=total_s + 0.05)
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size, isColor=False)
        phase = self.rng.uniform(0, 1 / self.fps)
        n = int(total_s * self.fps)
        for k in range(n):
            t0 = phase + k / self.fps
            acc = np.zeros(self.size[::-1], np.float32)
            for j in range(self.sub):
                tj = t0 + (j / max(self.sub - 1, 1) - 0.5) * self.exp
                acc += self.draw(float(np.interp(tj, T, TH)))
            img = acc / self.sub + self.rng.normal(0, noise, acc.shape)
            wr.write(np.clip(img, 0, 255).astype(np.uint8))
        wr.release()
        truth["speed_m_s"] = truth["omega_rad_s"] * self.m["impact_radius_mm"] / 1000
        truth["first_frame_s"] = phase
        return truth


def toy_model(mass_g=180.0, L=280.0, tip_r=284.0) -> dict:
    """A pendulum of point masses for the checks when no model of the scene is at hand."""
    I = mass_g * L ** 2 * 1.02
    return {"mass_g": mass_g, "c_h_mm": 3.0, "c_z_mm": -L * 0.9, "inertia_g_mm2": I, "impact_radius_mm": tip_r,
            "com_mm": math.hypot(3.0, L * 0.9), "com_angle_deg": math.degrees(math.atan2(3.0, L * 0.9)),
            "period_s": 2 * math.pi * math.sqrt(I / (mass_g * G_MM * math.hypot(3.0, L * 0.9))),
            "effective_mass_g": I / tip_r ** 2, "washer_g": 6.2, "washers_per_side": 3}


def synth_pilot(work: Path, model: dict, videos: int = 9, seed: int = 1, e0_marker: float = 3.0,
                torque_nmm: float = 190.0, lever_mm: float = 90.0, log=print) -> dict:
    """A simulated pilot in work: the log (A at the 9 holes in the 3 directions, 6 impacts on B), side videos for the
    first `videos` trials (A, frontal), gt_poses.csv as the scans would give it, and the corrections; the truth."""
    rng = np.random.default_rng(seed)
    raw = work / "raw" / "pilot"
    raw.mkdir(parents=True, exist_ok=True)
    sim = Sim(model, seed=seed)
    log_rows, poses, fixed, truth = [], [], [], {}
    plan = [("A", a, d) for d in DIRECTIONS_DEG for a in HOLES_DEG] + [("B", a, 0) for a in (20, 30, 40, 50, 30, 50)]
    for i, (holder, a, d) in enumerate(plan, 1):
        t = f"p{i:03d}"
        theta0 = a + rng.normal(0, 0.25)                          # by hand: the arm a little off its hole
        E = float(energy_mj(model, theta0))
        Em = TRANSFER[d] * E
        if holder == "A":
            rot = math.degrees(max(Em - e0_marker, 0.0) / torque_nmm) + abs(rng.normal(0, 0.02))
            tre = math.radians(rot) * lever_mm + abs(rng.normal(0, 0.04))
            after = abs(rng.normal(0.15, 0.08)) + 0.03 * tre
        else:
            rot = abs(rng.normal(0, 0.03))
            tre = abs(rng.normal(0.08, 0.04))
            after = abs(rng.normal(0.07, 0.03))
        log_rows.append({"trial": t, "phase": "pilot", "session": "2026-10-28", "holder": holder, "angle_deg": a,
                         "direction_deg": d, "washers_per_side": 3, "swing_mass_g": f"{model['mass_g']:.1f}",
                         "drape": 0, "status": "ok"})
        poses.append({"trial": t, "holder": holder, "angle_deg": a, "direction_deg": d, "rot_deg": f"{rot:.4f}",
                      "tre_mean_mm": f"{tre:.4f}", "tre_max_mm": f"{tre * 1.3:.4f}"})
        fixed.append({"trial": t, "tre_after_mm": f"{after:.4f}"})
        truth[t] = {"holder": holder, "angle_deg": a, "direction_deg": d, "release_deg": theta0, "energy_mj": E,
                    "rot_deg": rot}
        if i <= videos:
            (raw / t).mkdir(exist_ok=True)
            vt = sim.video(raw / t / f"{t}_highspeed.mp4", theta0, restitution=0.55 if Em < e0_marker else 0.35)
            truth[t].update({f"video_{k}": v for k, v in vt.items()})
            log(f"  video {t}: hole {a}, {vt['speed_m_s']:.3f} m/s")
    trials.write_csv(raw / trials.LOG_NAME, log_rows, trials.LOG_FIELDS)
    pose_path = work / "gt_poses.csv"
    trials.write_csv(pose_path, poses, list(poses[0]))
    fix_path = work / "corrected.csv"
    trials.write_csv(fix_path, fixed, ("trial", "tre_after_mm"))
    thr = {d: e0_marker / f for d, f in TRANSFER.items()}
    return {"raw": raw, "poses": pose_path, "corrected": fix_path, "truth": truth, "e0_pendulum": thr,
            "e0_marker": e0_marker, "k_marker": math.degrees(1 / torque_nmm)}


def synthetic(work: Path, out: Path = RESULTS, log=print) -> list[dict]:
    """A simulated pilot through the whole analysis (tables in work); its checks in out/pilot_synthetic.csv."""
    model = load_model() if MODEL_JSON.exists() else toy_model()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    t0 = time.perf_counter()
    sp = synth_pilot(work, model, log=log)
    res = analyse(sp["raw"], sp["poses"], model, sp["corrected"], work / "tables", sims=1500, log=log)
    rows = checks(res, sp)
    rows.append({"key": "runtime_s", "estimate": f"{time.perf_counter() - t0:.0f}"})
    trials.write_csv(out / "pilot_synthetic.csv", rows, ("key", "truth", "estimate", "error"))
    return rows


def checks(res: dict, sp: dict) -> list[dict]:
    """The analysis of a simulated pilot against its truth."""
    rows = []
    vid = [r for r in res["trials"] if r.get("speed_m_s") is not None]
    if vid:
        err_v = [r["speed_m_s"] / sp["truth"][r["trial"]]["video_speed_m_s"] - 1 for r in vid]
        err_a = [r["release_deg"] - sp["truth"][r["trial"]]["release_deg"] for r in vid]
        err_b = [r["rebound_deg"] - sp["truth"][r["trial"]]["video_rebound_deg"] for r in vid
                 if r.get("rebound_deg") is not None]
        rows += [{"key": "videos", "truth": len(vid), "estimate": len(vid), "error": 0},
                 {"key": "speed_err_max_pct", "truth": 0, "estimate": f"{100 * max(np.abs(err_v)):.2f}",
                  "error": f"{100 * max(np.abs(err_v)):.2f}"},
                 {"key": "speed_err_mean_pct", "truth": 0, "estimate": f"{100 * np.mean(err_v):.2f}",
                  "error": f"{100 * np.mean(err_v):.2f}"},
                 {"key": "release_err_max_deg", "truth": 0, "estimate": f"{max(np.abs(err_a)):.3f}",
                  "error": f"{max(np.abs(err_a)):.3f}"},
                 {"key": "rebound_err_max_deg", "truth": 0, "estimate": f"{max(np.abs(err_b)) if err_b else 0:.3f}",
                  "error": f"{max(np.abs(err_b)) if err_b else 0:.3f}"}]
    for tr in res["threshold"]:
        if tr["key"].startswith("dir_"):
            d = int(tr["key"][4:])
            want = sp["e0_pendulum"][d]
        else:
            want = sp["e0_marker"]
        got = _num(tr["e0"])
        rows.append({"key": f"threshold_{tr['key']}_mj", "truth": f"{want:.2f}", "estimate": tr["e0"],
                     "error": f"{got - want:.2f}" if got is not None else ""})
    for p in res["power"]:
        rows.append({"key": f"power_{p['key']}", "truth": "", "estimate": p.get("power_planned", ""),
                     "error": p.get("n_needed", "")})
    return rows


# ── checks ──────────────────────────────────────────────────────────────────
def run_test(work: Path) -> int:
    """Checks on simulated data (nothing written outside work); exit code 1 on a failure."""
    import ground_truth as gt
    if work.exists():
        if any(work.iterdir()) and not (work / trials.TEST_MARKER).exists():
            raise SystemExit(f"{work} is not empty and not a folder of an earlier test")
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / trials.TEST_MARKER).write_text("pilot\n", encoding="utf-8")
    fails = []

    def expect(ok, what):
        print(("  ok   " if ok else "  FAIL ") + what, flush=True)
        if not ok:
            fails.append(what)

    # mass properties of a box (outward winding) against the closed forms
    a, b, c = 30.0, 24.0, 40.0
    V = np.array([[x, y, z] for x in (0, a) for y in (0, b) for z in (0, c)], np.float64)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = np.array([t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])
    shift = np.array([5.0, -200.0, 7.0])
    m, com, S = mass_properties(V + shift, F, 1.27, origin=(0, 0, 0))
    rho = 1.27e-3
    m_true = rho * a * b * c
    cm = shift + np.array([a, b, c]) / 2
    Ix_true = m_true * (b * b + c * c) / 12 + m_true * (cm[1] ** 2 + cm[2] ** 2)
    expect(abs(m - m_true) < 1e-9 and np.allclose(com, cm) and abs(inertia_about(S, (1, 0, 0)) / Ix_true - 1) < 1e-9,
           f"box: {m:.4f} g, centre {np.round(com, 3).tolist()}, inertia about x within 1e-9")
    expect(mass_properties(V, F[:, ::-1], 1.0)[0] < 0, "a mesh wound inside out gives a negative mass")

    # energies of the toy pendulum against the simple formula
    model = load_model() if MODEL_JSON.exists() else toy_model()
    tm = toy_model()
    tm0 = {**tm, "c_h_mm": 0.0}
    E30 = float(energy_mj(tm0, 30.0))
    simple = tm0["mass_g"] * G_MM * abs(tm0["c_z_mm"]) * (1 - math.cos(math.radians(30))) * 1e-6
    w = float(contact_omega(tm0, 30.0))
    T = 2 * math.pi * math.sqrt(tm0["inertia_g_mm2"] / (tm0["mass_g"] * G_MM * abs(tm0["c_z_mm"])))
    w_formula = (2 * math.pi / T) * math.sqrt(2 * (1 - math.cos(math.radians(30))))
    expect(abs(E30 / simple - 1) < 1e-12 and abs(w / w_formula - 1) < 1e-12,
           f"energy and speed at 30 deg as M g d (1 - cos) and (2 pi / T) sqrt(2 (1 - cos)): {E30:.1f} mJ")
    expect(float(energy_mj(tm, 30.0)) < E30, "a centre of mass forward of the arm gives less energy")
    rows = energy_rows(model)
    expect(len(rows) == 9 and all(float(r2["energy_mj"]) > float(r1["energy_mj"]) for r1, r2 in zip(rows, rows[1:])),
           f"energy table: 9 holes, rising ({rows[0]['energy_mj']} to {rows[-1]['energy_mj']} mJ)")
    mr = {r["key"]: r["value"] for r in model_rows(model)}
    expect(mr["turn_pendulum_0_mj"] == "42" and mr["turn_pendulum_90_mj"] == "65",
           f"17 mJ in the marker: {mr['turn_pendulum_0_mj']} mJ at the pendulum frontal, "
           f"{mr['turn_pendulum_90_mj']} lateral (holes {mr['turn_hole_0_deg']} and {mr['turn_hole_90_deg']})")

    # side videos: speed, release and rebound against the simulation
    sim = Sim(model, size=(1280, 720), seed=3)
    worst = {"speed": 0.0, "release": 0.0, "rebound": 0.0}
    for k, (hole, e) in enumerate(((10, 0.6), (30, 0.45), (50, 0.3))):
        p = work / f"side_{hole}.mp4"
        tr = sim.video(p, hole + 0.3 * (k - 1), restitution=e, total_s=1.0)
        r = track_video(p, model=model)
        worst["speed"] = max(worst["speed"], abs(r["speed_m_s"] / tr["speed_m_s"] - 1))
        worst["release"] = max(worst["release"], abs(r["release_deg"] - tr["release_deg"]))
        worst["rebound"] = max(worst["rebound"], abs(r["rebound_deg"] - tr["rebound_deg"]))
        print(f"       hole {hole}: {r['speed_m_s']:.4f} m/s (true {tr['speed_m_s']:.4f}), release "
              f"{r['release_deg']:.3f} ({tr['release_deg']:.3f}), rebound {r['rebound_deg']:.3f} "
              f"({tr['rebound_deg']:.3f}), restitution {r['restitution']:.3f} ({e}), fps {r['fps']:.0f} "
              f"{r['fps_from']}, stickers {r['stickers']}, seen {r['seen']:.2f}", flush=True)
    expect(worst["speed"] < 0.015, f"tip speed from the video within 1.5 % ({100 * worst['speed']:.2f} %)")
    expect(worst["release"] < 0.2, f"release angle within 0.2 deg ({worst['release']:.3f})")
    expect(worst["rebound"] < 0.4, f"rebound angle within 0.4 deg ({worst['rebound']:.3f})")
    slow = work / "side_10.mp4"
    expect(video_fps(slow)[0] == 240.0 and video_fps(slow, 239.5) == (239.5, "given"),
           "frame rate: 240 from the container, a given one wins")

    # slip threshold and power
    rng = np.random.default_rng(5)
    E = np.repeat(np.array([float(energy_mj(model, h2)) for h2 in HOLES_DEG]), 3)
    rot = np.maximum(E - 30.0, 0) * 0.12 + rng.normal(0, 0.03, len(E))
    s = slip_threshold(E, rot, n_boot=300)
    expect(abs(s["e0"] - 30.0) < 3.0 and s["e0_lo"] <= 30.0 <= s["e0_hi"] and abs(s["e5"] - (30 + 5 / 0.12)) < 3,
           f"slip onset {s['e0']:.1f} mJ (true 30, 95 % {s['e0_lo']:.1f}-{s['e0_hi']:.1f}), 5 deg at {s['e5']:.1f} mJ")
    from scipy.stats import norm
    sd = 1.0
    d = rng.normal(0, sd, 5000)
    n = n_needed(d, delta=0.5, sims=1500, seed=2)
    z = norm.ppf(1 - ALPHA / COMPARISONS / 2) + norm.ppf(POWER)
    n_t = (z * sd / 0.5) ** 2 * math.pi / 3                     # t test, signed-rank efficiency 3 / pi
    expect(n is not None and abs(n / n_t - 1) < 0.15, f"power: {n} differences for 0.5 mm at sd 1 (normal theory "
                                                       f"{n_t:.0f})")

    # scanner exports: one synthetic scan in every format
    T_W_mandible, T_W_M = gt.scene_poses()
    clouds, _ = gt.synthetic_scan(T_W_M, T_W_mandible, np.random.default_rng(4), merged=True)
    P = np.asarray(clouds[0] if isinstance(clouds, list) else clouds, np.float64)
    written = write_formats(work / "scans", P)
    got = {name: scan_format(path) for name, path in written.items()}
    ref = np.array(got["ply_binary"]["T_M_mandible"]) if got["ply_binary"]["code"] == "ok" else None
    for name, g in got.items():
        if name == "ply_metres":
            expect(g["code"] == "metres", f"scan export {name}: said to be in metres")
            continue
        same = ref is not None and g["code"] == "ok" and np.abs(np.array(g["T_M_mandible"]) - ref).max() < 2e-3
        expect(same, f"scan export {name}: {g['code']}, {g.get('points')} points, same registration")
    (work / "scans" / "broken.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 9\n")
    expect(scan_format(work / "scans" / "broken.ply")["code"] in ("unreadable", "empty"), "a broken export: refused")

    # cap point and the 3 mm rule
    cap = trials.cap_point_M()
    expect(cap is not None and np.isfinite(cap).all(), f"cap point in M: {None if cap is None else np.round(cap, 2)}")
    if cap is not None:
        c_M = np.array([22.5, 22.5, 20.0])                      # a joint centre behind the board
        lever = float(np.linalg.norm(cap - c_M))
        ang = math.radians(3.0)
        Rz = np.array([[1, 0, 0], [0, math.cos(ang), -math.sin(ang)], [0, math.sin(ang), math.cos(ang)]])
        turn = np.eye(4)
        turn[:3, :3] = Rz
        turn[:3, 3] = c_M - Rz @ c_M
        T_ref = np.eye(4)
        T_now = turn @ T_ref                                    # the marker turned about c by 3 deg
        got_mm = trials.cap_shift_mm(T_ref, T_now, cap)
        axis_x = np.array([1.0, 0, 0])
        rr = (cap - c_M) - ((cap - c_M) @ axis_x) * axis_x
        want_mm = 2 * np.linalg.norm(rr) * math.sin(ang / 2)
        expect(abs(got_mm - want_mm) < 1e-9, f"cap shift of a 3 deg turn about the joint: {got_mm:.3f} mm "
                                             f"(lever {lever:.1f} mm)")
    print(f"{'FAIL' if fails else 'ok'}: pilot, {len(fails)} failed", flush=True)
    return 1 if fails else 0


def write_formats(folder: Path, P: np.ndarray) -> dict:
    """One cloud as the scanner might export it: PLY binary with normals, colours and faces, PLY ascii, STL binary
    and ascii (a triangle per point), OBJ with normals, ASC with normals, and a PLY in metres."""
    folder.mkdir(parents=True, exist_ok=True)
    P = np.asarray(P, np.float64)
    n = len(P)
    N = np.tile([0.0, 0.0, 1.0], (n, 1))
    out = {}
    head = (f"ply\nformat binary_little_endian 1.0\ncomment exported\nelement vertex {n}\nproperty float x\n"
            "property float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nelement face 1\n"
            "property list uchar int vertex_indices\nend_header\n")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.zeros(n, dt)
    v["x"], v["y"], v["z"] = P.T
    v["nz"] = 1.0
    v["r"] = v["g"] = v["b"] = 200
    face = np.array([3], "u1").tobytes() + np.array([0, 1, 2], "<i4").tobytes()
    out["ply_binary"] = folder / "scan_binary.ply"
    out["ply_binary"].write_bytes(head.encode() + v.tobytes() + face)
    out["ply_ascii"] = folder / "scan_ascii.ply"
    lines = [f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in P]
    out["ply_ascii"].write_text(f"ply\nformat ascii 1.0\nelement vertex {n}\nproperty float x\nproperty float y\n"
                                "property float z\nend_header\n" + "\n".join(lines) + "\n", encoding="ascii")
    tri = np.zeros(n, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    tri["v"] = np.repeat(P[:, None, :], 3, 1)
    out["stl_binary"] = folder / "scan_binary.stl"
    out["stl_binary"].write_bytes(b"\0" * 80 + np.uint32(n).tobytes() + tri.tobytes())
    out["stl_ascii"] = folder / "scan_ascii.stl"
    body = "".join(f"facet normal 0 0 1\nouter loop\nvertex {x:.4f} {y:.4f} {z:.4f}\nvertex {x:.4f} {y:.4f} {z:.4f}\n"
                   f"vertex {x:.4f} {y:.4f} {z:.4f}\nendloop\nendfacet\n" for x, y, z in P)
    out["stl_ascii"].write_text("solid scan\n" + body + "endsolid scan\n", encoding="ascii")
    out["obj"] = folder / "scan.obj"
    out["obj"].write_text("# exported\n" + "".join(f"v {x:.4f} {y:.4f} {z:.4f} 0.8 0.8 0.8\nvn 0 0 1\n"
                                                   for x, y, z in P), encoding="ascii")
    out["asc"] = folder / "scan.asc"
    out["asc"].write_text("".join(f"{x:.4f} {y:.4f} {z:.4f} 0 0 1\n" for x, y, z in P), encoding="ascii")
    out["ply_metres"] = folder / "scan_metres.ply"
    import ground_truth as gt
    gt.write_ply(out["ply_metres"], P / 1000.0)
    return out


# ── command line ────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" in argv:                                 # inside Blender: the arguments after --
        argv = argv[argv.index("--") + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("--scene", type=Path)
    bm = sub.add_parser("blender-model")
    bm.add_argument("out", type=Path)
    sub.add_parser("energy")
    v = sub.add_parser("video")
    v.add_argument("file", type=Path)
    v.add_argument("--fps", type=float)
    v.add_argument("--mass", type=float)
    a = sub.add_parser("analyse")
    a.add_argument("--raw", type=Path, default=RAW / "pilot")
    a.add_argument("--poses", type=Path, default=GT_POSES)
    a.add_argument("--corrected", type=Path)
    s = sub.add_parser("scan-format")
    s.add_argument("files", type=Path, nargs="+")
    sy = sub.add_parser("synthetic")
    sy.add_argument("--work", type=Path, default=WORK / "pilot" / "synthetic")
    t = sub.add_parser("test")
    t.add_argument("--work", type=Path, default=WORK / "pilot" / "test")
    args = ap.parse_args(argv)
    if args.cmd == "blender-model":
        args.out.write_text(json.dumps(blender_model(), indent=1), encoding="utf-8")
        return 0
    if args.cmd == "model":
        mod = run_model(args.scene)
        for r in model_rows(mod):
            print(f"{r['key']:24s} {r['value']} {r['unit']}")
        return 0
    if args.cmd == "energy":
        for p in write_energy(load_model()):
            print(p)
        return 0
    if args.cmd == "video":
        r = track_video(args.file, args.fps, load_model() if MODEL_JSON.exists() else None, args.mass)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, indent=1))
        return 0
    if args.cmd == "analyse":
        res = analyse(args.raw, args.poses, corrected=args.corrected)
        print(json.dumps(res["summary"], indent=1))
        return 0
    if args.cmd == "scan-format":
        bad = 0
        for f in args.files:
            r = scan_format(f)
            bad += r["code"] != "ok"
            print(f"{r['file']}: {r['code']}, {r.get('points', 0)} points, extent {r.get('extent_mm', '-')} mm"
                  + (f", PLY {r['ply']}" if "ply" in r else "") + (f" ({r['detail']})" if r.get("detail") else ""))
        return 1 if bad else 0
    if args.cmd == "synthetic":
        for r in synthetic(args.work):
            print(f"{r['key']:32s} truth {r.get('truth', '')!s:>8} estimate {r.get('estimate', '')!s:>8} "
                  f"error {r.get('error', '')}")
        return 0
    return run_test(args.work)


if __name__ == "__main__":
    sys.exit(main())
