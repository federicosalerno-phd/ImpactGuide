"""Build the whole test bench in Blender (T2.1 and T2.2).

Run inside Blender 5.2. From the Python console of a running Blender:
    ns = {"__file__": PATH, "__name__": "build_test_bench"}
    exec(open(PATH).read(), ns); ns["build"](); ns["render_checks"]()
From the command line:
    blender --background --python 04_code/tools/build_test_bench.py -- [--render] [--save]
The trial animation: animate_trial(), then render_trial_step() from a timer until it returns False, then
    blender --background --factory-startup --python 04_code/tools/build_test_bench.py -- --encode
Checks: interference(variant) for the parts of one marker interface, arm_clearance() for the sensor arm
against pendulum, phantom and mount over its whole sweep. show(variant=...) picks the visible interface
(a1, a2, b1, b2, b3, b4), set_direction(deg) turns the pendulum, set_arm(deg) turns the sensor arm.

The phantom, splint, wafer, plates, mandible spheres and silicone (signed distance fields, about 10 minutes)
are cached in CACHE as phantom_stage_<hash>.blend, keyed by the values of P they depend on; with the cache a
build takes a few seconds. Raise STAGE_VERSION when their code changes.

The scene is in millimetres at real scale (1 Blender unit = 1 mm, unit scale 0.001).
Every dimension lives in P. Catalogue parts use catalogue sizes; the other values are
first guesses, to be reviewed in T2.4 and fixed in T3.1 (02_cad/specs.md).

Frames. World: origin at the lower incisal point with the mouth closed, x towards the
patient's left, -y anterior, z up. Pad frame: origin on the top face of the splint pad,
on the axis of the interface, axes parallel to the world. The pendulum strikes from the
patient's right (0 deg frontal, 45 deg, 90 deg lateral), so that the left side stays free
for the surgical window, the smartphone and the scanner.
"""
import json
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[2]
BLEND = ROOT / "02_cad" / "blender" / "test_bench.blend"
RENDERS = ROOT / "02_cad" / "renders"
CACHE = Path(r"C:\work\marker-impact\models")
ANIM = Path(r"C:\work\marker-impact\anim\trial")
FONTS = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
ENV_PYTHON = Path(r"C:\envs\marker-impact\Scripts\python.exe")

P = {
    # Phantom: BodyParts3D 4.3 skull and mandible (DBCLS, CC BY-SA 2.1 JP), placeholder until T3.1
    "skull_url": "https://upload.wikimedia.org/wikipedia/commons/7/7b/BodyParts3D_Skull.stl",
    "mandible_url": "https://upload.wikimedia.org/wikipedia/commons/6/6b/BodyParts3D_FJ6399_Mandible.stl",
    "condyle_drop": 4.0,           # hinge axis below the top of the condyles
    "opening_step_deg": 0.25,
    "opening_margin_deg": 0.25,
    "wafer_min": 2.0,              # intermaxillary wafer thickness where the arches are closest
    "tmj_space": 1.5,              # condyle to fossa clearance (articular disc), the model has them in contact
    # Splint (PETG): cap on the lower crowns, bar through the lips, pad for the interfaces
    "voxel": 0.2,
    "splint_wall": 2.5,
    "splint_clearance": 0.15,
    "splint_depth": 5.0,           # cap edge below the occlusal plane
    "bar_width": 16.0,
    "bar_height": 8.0,             # bar top at the lower incisal edge
    "bar_length": 50.0,            # lower incisal point to interface axis
    "pad_x": (-15.0, 15.0),
    "pad_y": (-24.0, 18.0),
    "pad_thickness": 10.0,
    "m3_clearance_d": 3.4,
    "m3_insert_d": 4.0,
    "m3_insert_depth": 6.0,
    "m3_head": (5.5, 3.0),         # ISO 4762 M3 head diameter, height
    # Intermaxillary fixation: occlusal wafer plus two lateral plates
    "imf_plate": (10.0, 3.0),      # width, thickness
    "imf_maxilla": (32.0, 18.0),   # y, z of the upper screw
    "imf_mandible": (46.0, -10.0), # y, z of the lower screw
    # Marker (pad frame): ChArUco print on an aluminium support in a PETG holder
    "marker_center": (0.0, -40.0, 35.0),
    "marker_azimuth_deg": 50.0,    # board normal turned from anterior towards the patient's left
    "marker_elevation_deg": 45.0,  # board normal above the horizontal
    "marker_size": 50.0,
    "marker_al": 1.5,
    "marker_print": 0.15,
    "holder_size": 58.0,
    "holder_thickness": 4.0,
    "holder_boss": (16.0, 6.0),    # diameter, length
    "strike_post": (8.0, 24.0),    # vertical post on the holder edge where the pendulum strikes: core diameter, length
    "strike_cap": (14.0, 3.0, 21.0),  # replaceable solid PETG cap on the post: outer diameter, top, length
    "charuco": {"squares": 5, "square": 9.0, "marker": 6.0, "dictionary": "DICT_4X4_50", "px_per_mm": 20},   # 15/10: markers 6 (were 7)
    "arm_section": (12.0, 8.0),
    # Interfaces between pad and marker arm, all on the same pad pattern (three M3 holes at r 9 mm, 90, 210 and
    # 330 deg, and a 6.5 mm centre hole), so one splint takes every variant. A slips and stays (the AI corrects
    # it), B goes back by itself. Catalogue sizes from the standards and the sellers, to be confirmed in T3.
    "fit": 0.05,                   # radial clearance of balls in printed seats (no overlap in the model)
    "pad_holes": {"r": 9.0, "deg": (90.0, 210.0, 330.0), "centre_d": 6.5},
    "m4_head": (7.0, 4.0),         # ISO 4762 M4 head diameter, height
    # A1: ball stud DIN 71803 C 10 M6 screwed into the arm, its ball clamped between a cup and a cap by three
    # M3 screws through the pad (ISO 4762 M3 x 30, two DIN 2093 A 6 discs each, nuts under the pad)
    "a1": {"ball_d": 10.0, "base": (12.0, 7.0), "cap": 5.5, "gap": 0.5, "seat_deg": 50.0, "opening": 11.0,
           "neck_d": 6.0, "l2": 12.5, "sw": 8.0, "hex_h": 4.0, "thread": (6.0, 11.0),
           "disc": (6.0, 3.2, 0.3, 0.25), "screw_l": 30.0, "nut": (5.5, 2.4), "root": (13.0, 13.0)},
    # A2: one-axis friction hinge, fork on the pad, tongue of the arm between PTFE washers, shoulder screw
    # ISO 7379 5 x 16 (M4) with two DIN 2093 A 10 discs on the shoulder and a lock nut ISO 10511
    "a2": {"foot": (12.0, 7.0), "cheek": (3.0, 4.5, 7.0), "axle_z": 16.0, "washer": (10.0, 5.2, 0.5), "tongue_top": 9.0,
           "screw": {"shoulder_d": 5.0, "shoulder_l": 16.0, "thread_d": 4.0, "thread_l": 6.0, "head_d": 8.0, "head_k": 3.5},
           "disc": (10.0, 5.2, 0.5, 0.25), "nut": (7.0, 5.0), "root": (12.0, 12.0)},
    # B1: kinematic coupling, three balls ISO 3290 5 mm on three pairs of dowels ISO 8734 3 x 10, one central
    # ISO 4762 M4 x 20 into a heat-set insert, compression spring under an ISO 7093 washer: preload and stop
    "b1": {"base": (15.0, 6.0), "ball_d": 5.0, "pitch_r": 10.0, "groove_deg": (30.0, 150.0, 270.0), "pin": (3.0, 10.0),
           "pin_sink": 0.5, "relief": 1.5, "plate_h": 5.0, "hole_d": 9.0, "spring": (0.8, 8.0, 7.0),
           "washer": (12.0, 4.3, 1.0), "screw_l": 20.0, "insert": (5.6, 6.0), "root": (10.0, -12.5, 13.0)},
    # B2: Thorlabs KB25/M magnetic kinematic base (25 x 25 mm, two magnet pairs, about 16 N); plate
    # thickness to be confirmed on the drawing; bottom plate on the pad with one M4 and a nut, cord as a tether
    "b2": {"plate": (25.0, 6.0), "gap": 0.5, "ball_d": 4.0, "ball_r": 8.0, "root": (14.0, 12.0), "screw_l": 16.0},
    # B3: spring steel rod (EN 10270-1) glued in two sockets, bends and springs back
    "b3": {"foot": (12.0, 7.0), "boss": (6.0, 8.0), "rod": (4.0, 15.0), "socket": 10.0, "root": (12.0, 12.0)},
    # B4: as B1 with two N42 disc magnets in place of screw and spring, cord as a tether
    "b4": {"magnet": (6.0, 3.0), "gap": 1.0, "boss_r": 3.75, "root": (12.0, 12.0)},
    "tether_d": 1.2,
    # Ground truth: matte printed spheres, three on the marker holder and three on the mandible
    "gt_sphere_d": 10.0,
    "gt_marker_post": (4.0, 9.0),  # diameter, length beyond the holder corner
    "gt_mandible_post": (5.0, 24.0),
    "gt_mandible": (((0.0, 8.0, -30.0), (0.0, -0.45, -1.0)),
                    ((22.0, 18.0, -30.0), (0.55, -0.25, -1.0)),
                    ((34.0, 40.0, -25.0), (0.7, 0.0, -1.0))),
    # Silicone soft tissue with the surgical window on the left mandibular body (placeholder: Federico already
    # has a printed skull and a silicone face, to be modelled in T3.1)
    "silicone": {"thickness": 10.0, "voxel": 0.6, "smooth": (3, 6), "x": (-80.0, 80.0), "y": (-80.0, 55.0), "z": (-58.0, 42.0)},
    "window": {"guess": (24.0, 14.0, -18.0), "normal": (0.75, -0.55, -0.1), "size": (50.0, 32.0)},
    # Phantom mount: contoured block bonded under the cranial base, located on an aluminium plate by two
    # dowel pins (hole and slot) and clamped by one M6 screw; the plate stands on two 2020 posts
    "mount": {"x": 32.0, "y": (100.0, 168.0), "bottom": -70.0, "cap": 45.0, "grid": 2.0,
              "plate": (52.0, 10.0),       # half width, thickness
              "post_x": 42.0,
              "dowel": (6.0, 20.0),        # ISO 8734 6 x 20, pressed through the plate
              "dowel_dy": 22.0, "slot": 3.0,
              "insert": (8.0, 12.0),       # heat-set M6 insert in the block: hole diameter, depth
              "knob": (15.0, 12.0)},       # star knob with M6 stud: radius, height
    "profile": 20.0,               # 2020 B-type extrusion, slot 6 (in the lab) for every frame
    # Base board: birch plywood 18 mm on rubber feet, cut to the footprint plus a margin
    "base_z": -250.0,
    "base_t": 18.0,
    "base_margin": 20.0,
    "feet": (30.0, 20.0),
    # Pendulum: axle on two KFL08 flange units (608 bearing inside) screwed to two aluminium drop plates under
    # the beams; the left plate is also the protractor (holes every 5 deg, arc slot for the servo shaft); latch:
    # MG90S servo in a PETG bracket on the outer face, hook on the horn, pin on the arm
    "pendulum": {"L": 300.0, "R_c": 290.0, "R_index": 250.0, "direction_deg": 0.0, "release_deg": 30.0,
                 "max_release_deg": 50.0, "frame_width": 80.0, "turntable": (150.0, 12.0), "turntable_back": 40.0,
                 "turntable_front": 30.0, "arm_d": 10.0, "block": (30.0, 24.0, 30.0),
                 "nose_r": 8.0, "tip_shank": 16.0, "washers_per_side": 3, "washer": (24.0, 8.4, 2.0), "latch_r": 150.0,
                 "drop_plate": 6.0, "axle_drop": 25.0, "sector": (165.0, -5.0, 72.0), "holes_r": 128.0, "slot_w": 7.0,
                 "strut_foot": 150.0, "spacer_d": 12.0, "pin": (4.0, 13.0),
                 "kfl08": {"length": 48.0, "width": 27.0, "holes": 36.0, "hole_d": 5.0, "flange": 4.0, "housing": 8.5,
                           "total": 12.0, "boss_d": 26.0, "collar_d": 15.0},
                 "servo": {"body": (22.8, 12.2, 22.5), "top": 26.5, "spline": (4.8, 28.5), "shaft_x": 6.1,
                           "ears": (32.5, 2.5, 15.9, 27.5), "horn": (14.0, 3.0), "open_deg": 70.0}},
    # Sensor arm: one 2020 arm turning about the fixed pivot shaft of the pendulum turntable on two 6001 bearings
    # (the collar holds the hub down), carried at track_r by a bogie with two 608 wheels rolling on a printed
    # track with one index hole per position; the post at the end of the arm overhangs the track and carries
    # the scanner (low, it has to see the spheres under the mandible) and the phone (high)
    "arm": {"shaft": (12.0, 60.0, 8.0), "hub": (44.0, 22.0), "bearing": (12.0, 28.0, 8.0), "thrust": (40.0, 1.0),
            "collar": (22.0, 8.0), "track_r": 300.0, "track": (50.0, 10.0), "track_segments": 3, "wheel": (22.0, 7.0, 8.0),
            "wheel_dt": 30.0, "bogie": (94.0, 60.0, 6.0), "post_plate": (64.0, 64.0, 6.0), "post_gap": 8.0,
            "sweep_deg": (35.0, 100.0), "tick_deg": 5.0,
            "plunger": {"dt": 20.0, "dr": -18.0, "pin_d": 5.0, "body": (10.0, 12.0), "knob": (8.0, 9.0), "lift": 6.0}},
    # Smartphone: Poco X7 Pro 160.95 x 75.24 x 8.43 (Xiaomi specs) in a cover, about 164.6 x 78.9 x 11.1,
    # landscape, 1x camera; main lens about 16 and 20 mm from the corner of the camera end. Cradle on a tilt
    # hinge (knob and lever) for the video; camera azimuths of the four photo positions about the target
    "phone": {"size": (164.6, 78.9, 11.1), "lens_offset": (16.0, 20.0), "distance": 280.0,
              "elevation_deg": 20.0, "azimuths_deg": (40.0, 55.0, 70.0, 85.0), "focal_35mm": 25.0, "shown": 1,
              "tilt_deg": (-12.0, 12.0)},
    # Structured light scanner EinScan-SP V2: stand-off 290 to 480 mm, single capture 200 x 150 mm, 2.5 kg;
    # head proportions from the Shining 3D product photos; at least min_distance from the sphere centroid
    "scanner": {"head": (200.0, 55.0, 75.0), "baseline": 140.0, "min_distance": 300.0, "elevation_deg": -12.0,
                "field": (200.0, 150.0), "stand_off": (290.0, 480.0)},
}

COLORS = {  # sRGB; interface A orange and B blue as in the figures
    "pla_cranium": ("#d6d3cc", 0.0, 0.6),
    "pla_mandible": ("#f0e2bc", 0.0, 0.6),
    "petg_splint": ("#76808b", 0.0, 0.45),
    "petg_a": ("#eb6834", 0.0, 0.45),
    "petg_b": ("#2a78d6", 0.0, 0.45),
    "petg_marker": ("#303030", 0.0, 0.6),
    "petg_fixture": ("#5d6b78", 0.0, 0.5),
    "petg_white": ("#f1eee6", 0.0, 0.4),
    "brass": ("#c9a13b", 1.0, 0.35),
    "steel": ("#b4b8bc", 1.0, 0.25),
    "aluminium": ("#d4d7da", 1.0, 0.4),
    "profile": ("#c3c7cc", 1.0, 0.45),
    "magnet": ("#c4c4bc", 1.0, 0.3),
    "silicone": ("#e7b49c", 0.0, 0.55),
    "sphere": ("#f2f2f2", 0.0, 0.95),
    "phone": ("#1d1d20", 0.3, 0.3),
    "scanner": ("#c9ccd1", 0.9, 0.3),
    "glass": ("#111317", 0.2, 0.08),
    "lens": ("#0d0d12", 0.8, 0.1),
    "electronics": ("#1e5aa8", 0.0, 0.5),
    "rubber": ("#222222", 0.0, 0.9),
    "table": ("#ececec", 0.0, 0.8),
    "wood": ("#d6bd95", 0.0, 0.6),
    "anodised": ("#2a2b2f", 0.6, 0.45),
    "ptfe": ("#f4f3ee", 0.0, 0.5),
    "cord": ("#e2c62e", 0.0, 0.8),
    "print_white": ("#f4f4f2", 0.0, 0.9),
    "print_black": ("#141414", 0.0, 0.9),
}


# ---------------------------------------------------------------- geometry helpers

def frame(origin, z, x_hint=(1.0, 0.0, 0.0)):
    z = Vector(z).normalized()
    x = Vector(x_hint)
    if abs(x.normalized().dot(z)) > 0.99:
        x = Vector((0.0, 1.0, 0.0)) if abs(z.y) < 0.9 else Vector((1.0, 0.0, 0.0))
    x = (x - x.dot(z) * z).normalized()
    m = Matrix((x, z.cross(x), z)).transposed().to_4x4()
    m.translation = Vector(origin)
    return m


def revolve(profile, segments=48):
    """Closed (r, z) polygon revolved about z; points with r == 0 lie on the axis."""
    bm = bmesh.new()
    rings = []
    for r, z in profile:
        if r == 0:
            rings.append([bm.verts.new((0.0, 0.0, z))])
        else:
            rings.append([bm.verts.new((r * math.cos(2 * math.pi * i / segments),
                                        r * math.sin(2 * math.pi * i / segments), z))
                          for i in range(segments)])
    for k in range(len(rings)):
        a, b = rings[k], rings[(k + 1) % len(rings)]
        if len(a) == 1 and len(b) == 1:
            continue
        for i in range(segments):
            j = (i + 1) % segments
            if len(a) == 1:
                bm.faces.new((a[0], b[i], b[j]))
            elif len(b) == 1:
                bm.faces.new((a[i], a[j], b[0]))
            else:
                bm.faces.new((a[i], a[j], b[j], b[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def cylinder(r, z0, z1, segments=48):
    return revolve([(0, z0), (r, z0), (r, z1), (0, z1)], segments)


def tube(ri, ro, z0, z1, segments=48):
    return revolve([(ri, z0), (ro, z0), (ro, z1), (ri, z1)], segments)


def box(lo, hi):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    lo, hi = Vector(lo), Vector(hi)
    bmesh.ops.scale(bm, vec=hi - lo, verts=bm.verts)
    bmesh.ops.translate(bm, vec=(lo + hi) / 2, verts=bm.verts)
    return bm


def sphere(r, c=(0.0, 0.0, 0.0)):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=r)
    bmesh.ops.translate(bm, vec=Vector(c), verts=bm.verts)
    return bm


def extrude_polygon(pts, z0, z1):
    bm = bmesh.new()
    bot = [bm.verts.new((x, y, z0)) for x, y in pts]
    top = [bm.verts.new((x, y, z1)) for x, y in pts]
    bm.faces.new(bot[::-1])
    bm.faces.new(top)
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        bm.faces.new((bot[i], bot[j], top[j], top[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def slot_profile(s=20.0):
    """Outline of a 2020 B-type extrusion, slot 6: opening 6.2, lip 1.8, chamber 11 wide and 6.1 deep with a
    sloping floor, corners chamfered 0.5 (the corner cavities do not show from outside)."""
    h, o, lip, w, d, f, c = s / 2, 3.1, 1.8, 5.5, 6.1, 3.0, 0.5
    side = [(-h + c, -h), (-o, -h), (-o, -h + lip), (-w, -h + lip), (-f, -h + d), (f, -h + d), (w, -h + lip),
            (o, -h + lip), (o, -h), (h - c, -h)]
    pts = []
    for k in range(4):
        a = math.pi / 2 * k
        ca, sa = math.cos(a), math.sin(a)
        pts += [(x * ca - y * sa, x * sa + y * ca) for x, y in side]
    return pts


def extrude_section(outer, z0, z1, hole_r=0.0, segments=20):
    """Prism of a polygon with an optional central round hole (the caps are triangulated)."""
    loops = [list(outer)]
    if hole_r:
        loops.append([(hole_r * math.cos(2 * math.pi * i / segments), hole_r * math.sin(2 * math.pi * i / segments))
                      for i in range(segments)])
    flat = bmesh.new()
    edges = []
    for loop in loops:
        lv = [flat.verts.new((x, y, 0.0)) for x, y in loop]
        edges += [flat.edges.new((lv[i], lv[(i + 1) % len(lv)])) for i in range(len(lv))]
    bmesh.ops.triangle_fill(flat, use_beauty=True, use_dissolve=False, edges=edges)
    flat.verts.index_update()
    tris = [[v.index for v in f.verts] for f in flat.faces]
    pts = [v.co.copy() for v in flat.verts]
    flat.free()
    bm = bmesh.new()
    bot = [bm.verts.new((p.x, p.y, z0)) for p in pts]
    top = [bm.verts.new((p.x, p.y, z1)) for p in pts]
    for t in tris:
        bm.faces.new([bot[i] for i in t])
        bm.faces.new([top[i] for i in t])
    k = 0
    for loop in loops:
        n = len(loop)
        for i in range(n):
            a, b = k + i, k + (i + 1) % n
            bm.faces.new((bot[a], bot[b], top[b], top[a]))
        k += n
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def hexagon(sw, z0, z1):
    """Hexagonal prism of width across flats sw about z."""
    r = sw / math.sqrt(3)
    return revolve([(0, z0), (r, z0), (r, z1), (0, z1)], 6)


def socket_screw(d, dk, k, length, z_head=0.0):
    """ISO 4762 screw about z: head below z_head, shank of the given length towards +z."""
    return revolve([(0, z_head - k), (dk / 2, z_head - k), (dk / 2, z_head), (d / 2, z_head),
                    (d / 2, z_head + length), (0, z_head + length)], 32)


def nut(sw, m, z0, d):
    """Hexagon nut about z from z0 to z0 + m, bore of the nominal diameter d (the thread is not modelled)."""
    r = sw / math.sqrt(3)
    return extrude_section([(r * math.cos(math.pi / 3 * i), r * math.sin(math.pi / 3 * i)) for i in range(6)], z0, z0 + m, d / 2 + 0.02, 24)


def disc_spring(de, di, t, h0, z0, inner_up=True):
    """DIN 2093 disc spring about z standing on z0, cone height h0."""
    if inner_up:
        prof = [(di / 2, z0 + h0), (de / 2, z0), (de / 2, z0 + t), (di / 2, z0 + h0 + t)]
    else:
        prof = [(di / 2, z0), (de / 2, z0 + h0), (de / 2, z0 + h0 + t), (di / 2, z0 + t)]
    return revolve(prof, 48)


def tube_curve(points, radius, closed=False):
    """Swept round wire along a polyline (springs, cords), as a bmesh."""
    cu = bpy.data.curves.new("tmp_wire", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    cu.use_fill_caps = not closed
    sp = cu.splines.new("POLY")
    sp.points.add(len(points) - 1)
    for p, co in zip(sp.points, points):
        p.co = (*co, 1.0)
    sp.use_cyclic_u = closed
    ob = bpy.data.objects.new("tmp_wire", cu)
    bpy.context.scene.collection.objects.link(ob)
    me = evaluated_mesh(ob)
    delete(ob)
    bm = bmesh.new()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    return bm


def compression_spring(z0, z1, wire, od, closed_turns=1.0):
    """Helical compression spring about z between z0 and z1, closed and flat at both ends."""
    rm = (od - wire) / 2
    n_active = max(2, round((z1 - z0 - 2 * wire) / (wire * 2.2)))
    turns = n_active + 2 * closed_turns
    pts = []
    steps = int(turns * 24)
    for k in range(steps + 1):
        u = k / 24
        if u < closed_turns:
            z = z0 + wire / 2
        elif u > turns - closed_turns:
            z = z1 - wire / 2
        else:
            z = z0 + wire / 2 + (z1 - z0 - wire) * (u - closed_turns) / n_active
        a = 2 * math.pi * u
        pts.append((rm * math.cos(a), rm * math.sin(a), z))
    return tube_curve(pts, wire / 2)


def moved(bm, matrix):
    bm.transform(matrix)
    return bm


def bar(p0, p1, w, t, extend=0.0, x_hint=(1.0, 0.0, 0.0)):
    """Box of section w (along x_hint) by t from p0 to p1, extended at both ends."""
    p0, p1 = Vector(p0), Vector(p1)
    return moved(box((-w / 2, -t / 2, -extend), (w / 2, t / 2, (p1 - p0).length + extend)),
                 frame(p0, p1 - p0, x_hint))


def profile_bar(p0, p1, s, x_hint=(1.0, 0.0, 0.0)):
    """2020 extrusion from p0 to p1 with its slots and the 4.2 mm central bore."""
    p0, p1 = Vector(p0), Vector(p1)
    return moved(extrude_section(slot_profile(s), 0.0, (p1 - p0).length, 2.1), frame(p0, p1 - p0, x_hint))


def along_x(bm, y, z):
    return moved(bm, frame((0.0, y, z), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))


def polar(r, deg, z=0.0):
    a = math.radians(deg)
    return Vector((r * math.cos(a), r * math.sin(a), z))


def azimuth_dir(az_deg, el_deg):
    """Unit vector turned az from anterior (-y) towards the patient's left (+x), el above the horizontal."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))


def arc_polygon(r_in, r_out, az0, az1, n=64):
    angles = np.radians(np.linspace(az0, az1, n))
    outer = [(r_out * math.sin(a), -r_out * math.cos(a)) for a in angles]
    inner = [(r_in * math.sin(a), -r_in * math.cos(a)) for a in angles[::-1]]
    return outer + inner


# ---------------------------------------------------------------- blender helpers

def new_obj(name, bm, coll, mat=None, matrix=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    if mat is not None:
        me.materials.append(mat)
    if matrix is not None:
        ob.matrix_world = matrix
    return ob


def new_obj_parts(name, items, coll, matrix=None):
    """One object from several bmeshes with their own materials: items = [(bm, material)]."""
    mats, bm, me = [], bmesh.new(), bpy.data.meshes.new("tmp_join")
    for b, m in items:
        if m not in mats:
            mats.append(m)
        for f in b.faces:
            f.material_index = mats.index(m)
        b.to_mesh(me)
        b.free()
        bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    ob = new_obj(name, bm, coll, mats[0], matrix)
    for m in mats[1:]:
        ob.data.materials.append(m)
    return ob


def mesh_obj(name, me, coll, mat):
    ob = bpy.data.objects.new(name, me)
    me.name = name
    coll.objects.link(ob)
    me.materials.clear()
    me.materials.append(mat)
    return ob


def empty(name, coll, size=25.0):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "ARROWS"
    ob.empty_display_size = size
    coll.objects.link(ob)
    return ob


def delete(ob):
    data = ob.data
    bpy.data.objects.remove(ob)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
        elif isinstance(data, bpy.types.Curve):
            bpy.data.curves.remove(data)


def evaluated_mesh(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    return bpy.data.meshes.new_from_object(ob.evaluated_get(dg))


def replace_mesh(ob, me):
    old = ob.data
    mats = list(old.materials)
    ob.data = me
    me.materials.clear()
    for m in mats:
        me.materials.append(m)
    bpy.data.meshes.remove(old)
    me.name = ob.name


def boolean(ob, bms, op):
    """Apply a boolean with cutters given as bmeshes in the local frame of ob."""
    tmp = bpy.data.collections.new("tmp_cutters")
    bpy.context.scene.collection.children.link(tmp)
    cutters = [new_obj(f"tmp_cut_{i}", bm, tmp, matrix=ob.matrix_world.copy()) for i, bm in enumerate(bms)]
    mod = ob.modifiers.new("boolean", "BOOLEAN")
    mod.operation = op
    mod.operand_type = "COLLECTION"
    mod.collection = tmp
    mod.solver = "MANIFOLD"
    me = evaluated_mesh(ob)
    ob.modifiers.remove(mod)
    replace_mesh(ob, me)
    for c in cutters:
        delete(c)
    bpy.data.collections.remove(tmp)
    return ob


def part(name, bms, coll, mat, matrix=None, cut=()):
    """One printed or bought part: union of bms minus cut, all in the frame given by matrix."""
    bms = list(bms)
    ob = new_obj(name, bms[0], coll, mat, matrix)
    if len(bms) > 1:
        boolean(ob, bms[1:], "UNION")
    if cut:
        boolean(ob, list(cut), "DIFFERENCE")
    return ob


def mesh_coords(me):
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def world_coords(ob):
    return transform_coords(mesh_coords(ob.data), ob.matrix_world)


def transform_coords(co, m):
    m = np.array(m)
    return co @ m[:3, :3].T + m[:3, 3]


def bvh(co, me):
    return BVHTree.FromPolygons([tuple(c) for c in co], [tuple(p.vertices) for p in me.polygons])


def bvh_world(ob):
    return bvh(world_coords(ob), ob.data)


def center_origin(ob):
    co = mesh_coords(ob.data)
    c = Vector(((co.min(0) + co.max(0)) / 2).tolist())
    ob.data.transform(Matrix.Translation(-c))
    ob.matrix_world = ob.matrix_world @ Matrix.Translation(c)


def set_origin(ob, world_point):
    local = ob.matrix_world.inverted() @ Vector(world_point)
    ob.data.transform(Matrix.Translation(-local))
    ob.matrix_world = ob.matrix_world @ Matrix.Translation(local)


def attach(child, parent):
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


def child_of(ob, parent):
    """Parent ob so that its mesh coordinates are read in the parent frame."""
    ob.parent = parent
    ob.matrix_parent_inverse = Matrix()
    ob.matrix_basis = Matrix()


def collection(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def layer_collection(name, lc=None):
    lc = lc or bpy.context.view_layer.layer_collection
    if lc.name == name:
        return lc
    for child in lc.children:
        found = layer_collection(name, child)
        if found:
            return found
    return None


def srgb_to_linear(hex_color):
    h = hex_color.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


def material(name):
    hex_color, metallic, roughness = COLORS[name]
    m = bpy.data.materials.new(name)
    c = (*srgb_to_linear(hex_color), 1.0)
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = c
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    m.diffuse_color = c
    m.metallic = metallic
    m.roughness = roughness
    if name == "wood":   # birch plywood face veneer: fine streaks along x
        nt = m.node_tree
        coord = nt.nodes.new("ShaderNodeTexCoord")
        stretch = nt.nodes.new("ShaderNodeVectorMath")
        stretch.operation = "MULTIPLY"
        stretch.inputs[1].default_value = (0.003, 0.09, 0.09)
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 1.0
        noise.inputs["Detail"].default_value = 6.0
        noise.inputs["Roughness"].default_value = 0.55
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.35, 0.7
        ramp.color_ramp.elements[0].color = (*srgb_to_linear("#d2b98f"), 1.0)
        ramp.color_ramp.elements[1].color = (*srgb_to_linear("#e2cfac"), 1.0)
        nt.links.new(coord.outputs["Object"], stretch.inputs[0])
        nt.links.new(stretch.outputs["Vector"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def sdf_eval(expr, voxel):
    """Mesh from a signed distance expression.

    expr: ("obj", object, reach_mm) | ("offset", expr, d) | ("union", [expr]) |
          ("intersect", [expr]) | ("diff", expr, [expr]) | ("mean", expr, width, iterations);
          reach sets the narrow band.
    """
    ng = bpy.data.node_groups.new("tmp_sdf", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = ng.nodes, ng.links
    cache = {}

    def build(e):
        kind = e[0]
        if kind == "obj":
            key = (e[1].name, e[2])
            if key not in cache:
                info = nodes.new("GeometryNodeObjectInfo")
                info.transform_space = "RELATIVE"
                info.inputs["Object"].default_value = e[1]
                grid = nodes.new("GeometryNodeMeshToSDFGrid")
                grid.inputs["Voxel Size"].default_value = voxel
                grid.inputs["Band Width"].default_value = max(3, int(math.ceil(e[2] / voxel)) + 3)
                links.new(info.outputs["Geometry"], grid.inputs["Mesh"])
                cache[key] = grid.outputs["SDF Grid"]
            return cache[key]
        if kind == "mean":
            node = nodes.new("GeometryNodeSDFGridMean")
            links.new(build(e[1]), node.inputs["Grid"])
            node.inputs["Width"].default_value = e[2]
            node.inputs["Iterations"].default_value = e[3]
            return node.outputs["Grid"]
        if kind == "offset":
            node = nodes.new("GeometryNodeSDFGridOffset")
            links.new(build(e[1]), node.inputs["Grid"])
            node.inputs["Distance"].default_value = e[2]
            return node.outputs["Grid"]
        node = nodes.new("GeometryNodeSDFGridBoolean")
        if kind == "diff":
            node.operation = "DIFFERENCE"
            links.new(build(e[1]), node.inputs["Grid 1"])
            subs = e[2]
        else:
            node.operation = "UNION" if kind == "union" else "INTERSECT"
            subs = e[1]
        for s in subs:
            links.new(build(s), node.inputs["Grid 2"])
        return node.outputs["Grid"]

    to_mesh = nodes.new("GeometryNodeGridToMesh")
    to_mesh.inputs["Threshold"].default_value = 0.0
    links.new(build(expr), to_mesh.inputs["Grid"])
    out = nodes.new("NodeGroupOutput")
    links.new(to_mesh.outputs["Mesh"], out.inputs["Geometry"])
    host = bpy.data.objects.new("tmp_sdf_host", bpy.data.meshes.new("tmp_sdf_host"))
    bpy.context.scene.collection.objects.link(host)
    host.modifiers.new("sdf", "NODES").node_group = ng
    me = evaluated_mesh(host)
    delete(host)
    bpy.data.node_groups.remove(ng)
    return me


# ---------------------------------------------------------------- scene

def reset_scene():
    for data in (bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                 bpy.data.images, bpy.data.node_groups, bpy.data.cameras, bpy.data.lights,
                 bpy.data.collections, bpy.data.worlds, bpy.data.texts):
        for item in list(data):
            data.remove(item)


def setup_scene():
    sc = bpy.context.scene
    us = sc.unit_settings
    us.system = "METRIC"
    us.scale_length = 0.001
    us.length_unit = "MILLIMETERS"
    # pure white background: viewport colour, and in renders a white world seen only by camera rays
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.background_type = "VIEWPORT"
                    space.shading.background_color = (1.0, 1.0, 1.0)
                    space.shading.color_type = "MATERIAL"
                    space.clip_start = 0.5
                    space.clip_end = 20000.0
    sc.display.shading.background_type = "VIEWPORT"
    sc.display.shading.background_color = (1.0, 1.0, 1.0)
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = 64
    sc.render.film_transparent = False
    sc.render.resolution_x, sc.render.resolution_y = 1600, 1200
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    world = bpy.data.worlds.new("white")
    sc.world = world
    nt = world.node_tree
    nt.nodes.clear()
    path = nt.nodes.new("ShaderNodeLightPath")
    light = nt.nodes.new("ShaderNodeBackground")
    light.inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)
    white = nt.nodes.new("ShaderNodeBackground")
    white.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(light.outputs["Background"], mix.inputs[1])
    nt.links.new(white.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


# ---------------------------------------------------------------- phantom and splint

def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "marker-impact-research/0.1"})
        path.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    return path


def import_stl(path):
    before = set(bpy.data.objects)
    bpy.ops.wm.stl_import(filepath=str(path))
    ob = (set(bpy.data.objects) - before).pop()
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    return ob


def islands(me):
    """Vertex index arrays of the loose parts of a mesh."""
    n = len(me.vertices)
    ev = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in ev.reshape(-1, 2):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([root(i) for i in range(n)])
    return [np.flatnonzero(roots == r) for r in np.unique(roots)]


def submesh(me, keep):
    bm = bmesh.new()
    bm.from_mesh(me)
    keep = set(int(i) for i in keep)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index not in keep], context="VERTS")
    return bm


def splint_expr(teeth, bone, clip, wall, add=(), sub=()):
    reach = wall + 1.0
    cap = ("intersect", [("offset", ("obj", teeth, reach), wall), ("obj", clip, 0.0)])
    body = ("union", [cap] + [("obj", ob, 0.0) for ob in add]) if add else cap
    cut = [("offset", ("obj", teeth, reach), P["splint_clearance"])]
    if bone is not None:
        cut.append(("offset", ("obj", bone, 1.0), P["splint_clearance"]))
    cut += [("obj", ob, 0.0) for ob in sub]
    return ("diff", body, cut)


def build_phantom_and_splint(colls, mats, tmp):
    skull = import_stl(fetch(P["skull_url"], CACHE / "bp3d_skull.stl"))
    mand = import_stl(fetch(P["mandible_url"], CACHE / "bp3d_mandible.stl"))

    # the skull file also holds the mandible and lower teeth: drop the islands found in the mandible file
    mand_isl = islands(mand.data)
    mco = mesh_coords(mand.data)
    boxes = [np.concatenate([mco[i].min(0), mco[i].max(0)]) for i in mand_isl]
    sco = mesh_coords(skull.data)
    keep = [i for i in islands(skull.data)
            if not any(np.allclose(np.concatenate([sco[i].min(0), sco[i].max(0)]), b, atol=0.05) for b in boxes)]
    bone_idx = mand_isl[int(np.argmax([len(i) for i in mand_isl]))]
    teeth_isl = [i for i in mand_isl if len(i) != len(bone_idx)]

    # world origin at the lower incisal point: top of the two teeth closest to the midline
    tips = np.array([mco[i][np.argmax(mco[i][:, 2])] for i in teeth_isl])
    centres = np.array([mco[i].mean(0) for i in teeth_isl])
    origin = tips[np.argsort(np.abs(centres[:, 0]))[:2]].mean(0)
    rel = [sco[i] - origin for i in keep]
    maxilla_isl = [i for i, c in zip(keep, rel)
                   if c[:, 0].min() < 50 and c[:, 0].max() > -50 and c[:, 1].min() < 65 and c[:, 1].max() > -20
                   and c[:, 2].min() < 30 and c[:, 2].max() > -6]

    cranium = new_obj("phantom_cranium", submesh(skull.data, np.concatenate(keep)), colls["phantom"], mats["pla_cranium"])
    mandible = new_obj("phantom_mandible", submesh(mand.data, np.arange(len(mco))), colls["phantom"], mats["pla_mandible"])
    teeth = new_obj("tmp_teeth", submesh(mand.data, np.concatenate(teeth_isl)), tmp)
    bone = new_obj("tmp_bone", submesh(mand.data, bone_idx), tmp)
    upper = [new_obj(f"tmp_upper_{k}", submesh(skull.data, isl), tmp) for k, isl in enumerate(maxilla_isl)]
    for ob in (skull, mand):
        delete(ob)
    shift = Matrix.Translation(Vector((-origin).tolist()))
    for ob in [cranium, mandible, teeth, bone] + upper:
        ob.data.transform(shift)
    tips = tips - origin

    # move the mandible (forwards and down, smallest step) until the condyles clear the fossae and the
    # tympanic plates, plus the articular space; teeth excluded, in the closed position they are in occlusion
    cran_tree = bvh_world(cranium)
    mc = mesh_coords(mandible.data)
    rami = [tuple(f.vertices) for f in mandible.data.polygons if mc[list(f.vertices), 1].min() > 55.0]
    steps = sorted(((dy, dz) for dy in np.arange(0.0, 6.01, 0.5) for dz in np.arange(0.0, 6.01, 0.5)), key=lambda q: q[0] ** 2 + q[1] ** 2)
    move = np.zeros(3)
    for dy, dz in steps:
        if not BVHTree.FromPolygons([tuple(c) for c in mc - np.array([0.0, dy, dz])], rami).overlap(cran_tree):
            if dy or dz:
                n = math.hypot(dy, dz)
                move = np.array([0.0, -dy, -dz]) * (1.0 + P["tmj_space"] / n)
            break
    drop = -move[2]
    lower = Matrix.Translation(Vector(move.tolist()))
    for ob in (mandible, teeth, bone):
        ob.data.transform(lower)
    tips = tips + move

    # occlusal plane through the cusp tips, and the box that clips the cap below it
    a, b, c = np.linalg.lstsq(np.c_[tips[:, 0], tips[:, 1], np.ones(len(tips))], tips[:, 2], rcond=None)[0]
    normal = Vector((-a, -b, 1.0)).normalized()
    centre = Vector((tips[:, 0].mean(), tips[:, 1].mean(), a * tips[:, 0].mean() + b * tips[:, 1].mean() + c))
    clip_m = frame(centre - normal * P["splint_depth"], normal, (1.0, 0.0, 0.0))
    clip = new_obj("tmp_clip", box((-90, -90, 0), (90, 90, 40)), tmp, matrix=clip_m)

    # hinge axis through the condyles, below their top
    bco = mesh_coords(bone.data)
    left = bco[bco[:, 0] < 0][np.argmax(bco[bco[:, 0] < 0][:, 2])]
    right = bco[bco[:, 0] > 0][np.argmax(bco[bco[:, 0] > 0][:, 2])]
    hinge = Vector(((left + right) / 2 - np.array([0.0, 0.0, P["condyle_drop"]])).tolist())
    axis = Vector((right - left).tolist()).normalized()

    def opening(deg):
        return Matrix.Translation(hinge) @ Matrix.Rotation(math.radians(deg), 4, axis) @ Matrix.Translation(-hinge)

    sign = 1.0 if (opening(1.0) @ Vector((0, 0, 0))).z < 0 else -1.0

    # open the mandible until a cap thicker by the wafer clears the upper teeth
    cap = sdf_eval(splint_expr(teeth, None, clip, P["splint_wall"] + P["wafer_min"]), P["voxel"])
    cap_co = mesh_coords(cap)
    theta = None
    for k in range(int(20.0 / P["opening_step_deg"]) + 1):
        deg = k * P["opening_step_deg"]
        if not bvh(transform_coords(cap_co, opening(sign * deg)), cap).overlap(cran_tree):
            theta = deg + P["opening_margin_deg"]
            break
    bpy.data.meshes.remove(cap)
    if theta is None:
        raise RuntimeError("the splint cap does not clear the upper teeth within 20 degrees")
    rot = opening(sign * theta)
    for ob in (mandible, teeth, bone):
        ob.data.transform(rot)
    clip.matrix_world = rot @ clip_m
    incisal = rot @ Vector(move.tolist())

    # bar and pad in the opened position, pad top level with the lower incisal edge
    o = Vector((incisal.x, incisal.y - P["bar_length"], incisal.z))
    t = P["pad_thickness"]
    w = P["bar_width"] / 2
    add = [new_obj("tmp_bar", box((o.x - w, o.y + P["pad_y"][1] - 2.0, o.z - P["bar_height"] + 0.3),
                                  (o.x + w, incisal.y + 3.0, o.z - 0.3)), tmp)]
    splint = mesh_obj("splint", sdf_eval(splint_expr(teeth, bone, clip, P["splint_wall"], add), P["voxel"]),
                      colls["splint"], mats["petg_splint"])
    holes = [(h, P["m3_clearance_d"] / 2) for h in pad_points()] + [(Vector(), P["pad_holes"]["centre_d"] / 2)]
    boolean(splint, [box((o.x + P["pad_x"][0], o.y + P["pad_y"][0], o.z - t), (o.x + P["pad_x"][1], o.y + P["pad_y"][1], o.z))], "UNION")
    boolean(splint, [moved(cylinder(r, o.z - t - 1, o.z + 1, 32), Matrix.Translation((o.x + h.x, o.y + h.y, 0.0)))
                     for h, r in holes], "DIFFERENCE")
    set_origin(splint, o)
    set_origin(mandible, hinge)
    clash = len(bvh_world(splint).overlap(cran_tree))
    return {"cranium": cranium, "mandible": mandible, "splint": splint, "teeth": teeth, "upper": upper,
            "clip": clip, "pad_origin": o, "opening_deg": theta, "splint_cranium_overlaps": clash, "tmj_drop": drop,
            "condyle_overlaps": len(bvh_world(mandible).overlap(cran_tree))}


def bent_plate(trees, side, p0, p1, width, thick, gap=0.5, n=28, ext=6.0):
    """Plate bent on the outermost lateral surface (along x) between two points; returns the mesh and
    the outer points above the two ends, where the screw heads sit."""
    t = p1 - p0
    length = t.length
    t.normalize()
    b = Vector((0.0, -t.z, t.y)).normalized()
    centres, outer = [], []
    for k in range(n + 1):
        c = p0 + t * (-ext + (length + 2 * ext) * k / n)
        best = 0.0
        for off in (-width / 2, 0.0, width / 2):
            q = c + b * off
            for tr in trees:
                hit = tr.ray_cast(Vector((side * 150.0, q.y, q.z)), Vector((-side, 0.0, 0.0)))[0]
                if hit is not None:
                    best = max(best, side * hit.x)
        centres.append(c)
        outer.append(best)
    xs = np.pad(np.array(outer), 1, mode="edge")
    xs = np.maximum(np.maximum(xs[:-2], xs[1:-1]), xs[2:]) + gap
    bm = bmesh.new()
    rows = []
    for c, x in zip(centres, xs):
        row = []
        for depth in (0.0, thick):
            for off in (-width / 2, width / 2):
                q = c + b * off
                row.append(bm.verts.new((side * (x + depth), q.y, q.z)))
        rows.append(row)  # inner-, inner+, outer-, outer+
    for r0, r1 in zip(rows[:-1], rows[1:]):
        for i, j in ((0, 1), (1, 3), (3, 2), (2, 0)):
            bm.faces.new((r0[i], r0[j], r1[j], r1[i]))
    bm.faces.new((rows[0][0], rows[0][2], rows[0][3], rows[0][1]))
    bm.faces.new((rows[-1][0], rows[-1][1], rows[-1][3], rows[-1][2]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    k0 = int(round(ext / (length + 2 * ext) * n))
    k1 = n - k0
    heads = [Vector((side * (xs[k] + thick), centres[k].y, centres[k].z)) for k in (k0, k1)]
    return bm, heads


def build_imf(ph, coll, mats, tmp):
    """Occlusal wafer between the upper teeth and the splint, and two plates that lock the mandible."""
    wall = P["splint_wall"]
    # one signed distance grid per upper bone: the islands overlap, a single grid would get the sign wrong
    region = ("intersect", [("offset", ("obj", ph["teeth"], wall + 6.0), wall + 5.0),
                            ("union", [("offset", ("obj", u, 4.0), 3.0) for u in ph["upper"]]),
                            ("obj", ph["clip"], 0.0)])
    wafer = mesh_obj("imf_wafer", sdf_eval(("diff", region, [("offset", ("obj", ph["splint"], 1.0), 0.15)]
                                            + [("offset", ("obj", u, 4.0), 0.15) for u in ph["upper"]]), 0.25),
                     coll, mats["petg_fixture"])
    center_origin(wafer)

    cran, mand = bvh_world(ph["cranium"]), bvh_world(ph["mandible"])
    trees = [cran, mand, bvh_world(ph["splint"]), bvh_world(wafer)]
    pw, pt = P["imf_plate"]
    parts = [wafer]
    for side, name in ((1.0, "left"), (-1.0, "right")):
        ends = []
        for tree, (y, z) in ((cran, P["imf_maxilla"]), (mand, P["imf_mandible"])):
            hit = tree.ray_cast(Vector((side * 90.0, y, z)), Vector((-side, 0.0, 0.0)))[0]
            assert hit is not None, f"no bone for the {name} plate at y={y}, z={z}"
            ends.append(hit)
        plate, heads = bent_plate(trees, side, ends[0], ends[1], pw, pt)
        ob = new_obj(f"imf_plate_{name}", plate, coll, mats["aluminium"])
        boolean(ob, [moved(cylinder(1.65, -2.0, pt + 4.0, 24), frame(h - Vector((side * (pt + 1.0), 0, 0)), (side, 0.0, 0.0)))
                     for h in heads], "DIFFERENCE")
        center_origin(ob)
        parts.append(ob)
        for k, h in enumerate(heads):
            screw = new_obj(f"imf_screw_{name}_{k + 1}", moved(screw_m3(12.0, 0.0), frame(h + Vector((side * 0.3, 0.0, 0.0)), (-side, 0.0, 0.0))),
                            coll, mats["steel"])
            center_origin(screw)
            parts.append(screw)
    return parts


# ---------------------------------------------------------------- marker

def charuco_png():
    c = P["charuco"]
    n, sq, mk, ppm, size = c["squares"], c["square"], c["marker"], c["px_per_mm"], P["marker_size"]
    path = CACHE / f"charuco_{n}x{n}_{sq:g}mm_{mk:g}mm_{c['dictionary'].lower()}.png"
    if not path.exists():
        code = (
            "import cv2\n"
            f"d = cv2.aruco.getPredefinedDictionary(cv2.aruco.{c['dictionary']})\n"
            f"b = cv2.aruco.CharucoBoard(({n}, {n}), {sq}, {mk}, d)\n"
            f"img = b.generateImage(({round(n * sq * ppm)}, {round(n * sq * ppm)}), marginSize=0, borderBits=1)\n"
            f"m = {round((size - n * sq) / 2 * ppm)}\n"
            "img = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=255)\n"
            f"cv2.imwrite(r'{path}', img)\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(ENV_PYTHON), "-c", code], check=True)
    return path


def charuco_cells():
    """Cell edges of the printed board as fractions of its side (the same along u and v) and the black cells [u, v],
    read from the board image at the cell centres. Edges exact in mm: margin, squares, marker bits (4 x 4 plus a
    one-bit border)."""
    c = P["charuco"]
    n, sq, mk, size = c["squares"], c["square"], c["marker"], P["marker_size"]
    m, bit = (size - n * sq) / 2, mk / 6
    edges = {0.0, size}
    for k in range(n):
        x0 = m + k * sq
        edges |= {round(e, 6) for e in (x0, x0 + sq, *(x0 + (sq - mk) / 2 + j * bit for j in range(7)))}
    edges = np.array(sorted(edges)) / size
    img = bpy.data.images.load(str(charuco_png()))
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    dark = px.reshape(h, w, 4)[..., 0] < 0.5                      # rows from the bottom, as v
    lo, hi = edges[:-1] * 0.8 + edges[1:] * 0.2, edges[:-1] * 0.2 + edges[1:] * 0.8
    black = np.zeros((len(lo), len(lo)), bool)
    for i in range(len(lo)):
        for j in range(len(lo)):
            f = dark[round(lo[j] * h):round(hi[j] * h), round(lo[i] * w):round(hi[i] * w)].mean()
            assert f < 0.01 or f > 0.99, (i, j, f)                 # one colour per cell
            black[i, j] = f > 0.5
    return edges, black


def charuco_bmesh(corners):
    """Printed board as geometry (13/10), so the pattern shows in solid view too: one face per cell, material 0 white
    and 1 black, UV map as the old textured square. corners: points at UV (0, 0), (1, 0), (1, 1), (0, 1)."""
    edges, black = charuco_cells()
    p00, p10, p11, p01 = corners
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    grid = [[bm.verts.new(p00.lerp(p10, u).lerp(p01.lerp(p11, u), v)) for v in edges] for u in edges]
    k = len(edges) - 1
    for i in range(k):
        for j in range(k):
            ij = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            face = bm.faces.new([grid[a][b] for a, b in ij])
            face.material_index = int(black[i, j])
            for loop, (a, b) in zip(face.loops, ij):
                loop[uv].uv = (edges[a], edges[b])
    return bm


def marker_frame():
    normal = azimuth_dir(P["marker_azimuth_deg"], P["marker_elevation_deg"])
    a = math.radians(P["marker_azimuth_deg"])
    return frame(P["marker_center"], normal, (math.cos(a), math.sin(a), 0.0)), normal


def build_marker(coll, mats, pad):
    mf, normal = marker_frame()
    s, h = P["marker_size"] / 2, P["holder_size"] / 2
    front = -P["marker_print"]
    back = front - P["holder_thickness"]
    mid = (front + back) / 2
    boss_d, boss_l = P["holder_boss"]
    post_d, post_l = P["strike_post"]
    edge = mf @ Vector((0.0, h - 2.0, mid))
    gd, (gpd, gpl) = P["gt_sphere_d"], P["gt_marker_post"]
    corners = [Vector((-h, -h)), Vector((h, -h)), Vector((h, h))]
    posts, centres = [], []
    for cxy in corners:
        diag = cxy.normalized()
        start = mf @ Vector((cxy.x - diag.x * 3, cxy.y - diag.y * 3, mid))
        end = mf @ Vector((cxy.x + diag.x * gpl, cxy.y + diag.y * gpl, mid))
        posts.append(moved(cylinder(gpd / 2, 0.0, (end - start).length, 24), frame(start, end - start)))
        centres.append(mf @ Vector((cxy.x + diag.x * (gpl + gd / 2), cxy.y + diag.y * (gpl + gd / 2), mid)))
    holder = part("marker_holder",
                  [moved(box((-h, -h, back), (h, h, front)), mf),
                   moved(cylinder(boss_d / 2, back - boss_l, back + 0.5), mf),
                   moved(cylinder(post_d / 2, -2.0, post_l, 32), Matrix.Translation(edge))] + posts,
                  coll, mats["petg_marker"], pad,
                  cut=[moved(box((-s - 0.2, -s - 0.2, front - P["marker_al"]), (s + 0.2, s + 0.2, 1.0)), mf)])
    plate = new_obj("marker_plate", moved(box((-s, -s, front - P["marker_al"]), (s, s, front)), mf),
                    coll, mats["aluminium"], pad)
    # replaceable cap pushed on the post core: the pendulum wears the cap, never the holder
    cap_d, cap_top, cap_l = P["strike_cap"]
    ri, z_top = post_d / 2 + P["fit"], post_l + cap_top
    cap = new_obj("marker_strike_cap", moved(revolve([(ri, z_top - cap_l), (cap_d / 2, z_top - cap_l), (cap_d / 2, z_top),
                                                      (0, z_top), (0, post_l + P["fit"]), (ri, post_l + P["fit"])], 48),
                                             Matrix.Translation(edge)), coll, mats["petg_white"], pad)

    bm = charuco_bmesh([mf @ Vector((x, y, 0.0)) for x, y in ((-s, -s), (s, -s), (s, s), (-s, s))])
    charuco = new_obj("marker_charuco", bm, coll, mats["print_white"], pad)
    charuco.data.materials.append(mats["print_black"])
    strike = pad @ (edge + Vector((0.0, 0.0, post_l / 2 + 2.0)))
    return [holder, plate, charuco, cap], strike, [pad @ c for c in centres]


def arm_path():
    """End flange behind the marker holder (pad frame) and the board normal."""
    mf, normal = marker_frame()
    end = mf @ Vector((0.0, 0.0, -P["marker_print"] - P["holder_thickness"] - P["holder_boss"][1]))
    return end, normal


def arm_tail(knee, end, normal):
    w, t = P["arm_section"]
    flange = moved(cylinder(P["holder_boss"][0] / 2, -14.0, 0.0), frame(end, normal))
    tip = end - normal * 8.0
    return [bar(knee - (tip - knee).normalized() * 4.0, tip, w, t), flange]


# ---------------------------------------------------------------- interfaces

VARIANTS = {  # key: (collection, label); A slips and stays, B goes back by itself
    "a1": ("a1_ball_stud", "A1 sfera su perno DIN 71803, morsetto a tre viti"),
    "a2": ("a2_friction_hinge", "A2 cerniera ad attrito su un asse"),
    "b1": ("b1_kinematic_spring", "B1 cinematico, vite centrale e molla"),
    "b2": ("b2_kinematic_kb25", "B2 base magnetica Thorlabs KB25/M"),
    "b3": ("b3_flexure_rod", "B3 barra elastica in acciaio armonico"),
    "b4": ("b4_kinematic_magnets", "B4 cinematico con due magneti"),
}

YZX = Matrix(((0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))  # (u, v, w) -> (x=w, y=u, z=v)


def pad_points(deg=None):
    ph = P["pad_holes"]
    return [polar(ph["r"], d) for d in (deg or ph["deg"])]


def screw_m3(length, z_head):
    """ISO 4762 M3, head underside at z_head, shank pointing towards +z."""
    d, k = P["m3_head"]
    return socket_screw(3.0, d, k, length, z_head)


def screw_down(d, dk, k, length, z_head):
    """Socket screw with the head standing on z_head and the shank pointing down."""
    return moved(socket_screw(d, dk, k, length, -z_head), Matrix.Rotation(math.pi, 4, "X"))


def pad_inserts(depth):
    """Cutters for M3 heat-set inserts from below at the pad holes."""
    return [moved(cylinder(P["m3_insert_d"] / 2, -1.0, depth, 32), Matrix.Translation(p)) for p in pad_points()]


def pad_screws(prefix, coll, mats, pad, length=16.0):
    """ISO 4762 M3 from under the pad into the inserts of the base."""
    return [new_obj(f"{prefix}_pad_screw_{i + 1}", moved(screw_m3(length, -P["pad_thickness"] - 0.02), Matrix.Translation(p)),
                    coll, mats["steel"], pad) for i, p in enumerate(pad_points())]


def variant_arm(name, coll, mat, pad, d, xy, z0, length, extra=(), cut=()):
    """Moving arm of a variant: socket of diameter d from z0 at xy, then the bar from the knee in the socket to
    the flange behind the marker holder (the marker pose is the same for every variant)."""
    end, normal = arm_path()
    knee = Vector((xy[0], xy[1], z0 + length - d / 2))
    return part(name, [moved(cylinder(d / 2, z0, z0 + length, 48), Matrix.Translation((xy[0], xy[1], 0.0)))]
                + arm_tail(knee, end, normal) + list(extra), coll, mat, pad, cut=cut)


def ring(r, wire, centre, axis):
    """Small closed ring (screw eye) of mean radius r about axis."""
    prof = [(r + wire * math.cos(t), wire * math.sin(t)) for t in np.linspace(0.0, 2 * math.pi, 13)[:-1]]
    return moved(revolve(prof, 24), frame(centre, axis))


def tether(name, coll, mats, pad, p_top, p_bottom, out):
    """Cord between two screw eyes with some slack outwards (direction out), for the parts that can come off."""
    out = Vector(out).normalized()
    eyes = [ring(1.4, 0.35, p + out * 1.4, Vector((0.0, 0.0, 1.0)).cross(out)) for p in (p_top, p_bottom)]
    a, b = p_top + out * 2.8, p_bottom + out * 2.8
    c = (a + b) / 2 + out * 8.0
    pts = [tuple(a * (1 - s) ** 2 + c * 2 * s * (1 - s) + b * s * s) for s in np.linspace(0.0, 1.0, 13)]
    cord = tube_curve(pts, P["tether_d"] / 2)
    return new_obj_parts(name, [(eyes[0], mats["steel"]), (eyes[1], mats["steel"]), (cord, mats["cord"])], coll, pad)


def build_a1(coll, mats, pad):
    """A1: the ball of a DIN 71803 stud (screwed into the arm) clamped between a cup and a cap; the clamping
    force comes from two disc springs under each of the three screws, which also hold the cup on the pad."""
    a = P["a1"]
    r, (R, H), g = a["ball_d"] / 2, a["base"], a["gap"]
    c = Vector((0.0, 0.0, H))                       # ball centre on the top face of the cup
    z0, z1 = H + g, H + g + a["cap"]
    seat = math.radians(a["seat_deg"])
    zs = H + r * math.cos(seat)                     # top of the seat band, where the opening for the neck starts
    clear = lambda: [moved(cylinder(1.7, -1.0, 40.0, 24), Matrix.Translation(p)) for p in pad_points()]
    cup = part("a1_cup", [cylinder(R, 0.0, H, 96)], coll, mats["petg_a"], pad, cut=[sphere(r + P["fit"], c)] + clear())
    opening = revolve([(0, zs), (r * math.sin(seat), zs), (a["opening"] / 2, z1 + 0.5), (0, z1 + 0.5)], 64)
    cap = part("a1_cap", [cylinder(R, z0, z1, 96)], coll, mats["petg_a"], pad, cut=[sphere(r + P["fit"], c), opening] + clear())
    hz = H + a["l2"]                                # top of the hexagon collar, where the arm sits
    stud = part("a1_ball_stud", [sphere(r, c), cylinder(a["neck_d"] / 2, H, hz - a["hex_h"] + 0.1, 32),
                                 hexagon(a["sw"], hz - a["hex_h"], hz), cylinder(a["thread"][0] / 2, hz - 0.1, hz + a["thread"][1], 32)],
                coll, mats["steel"], pad)
    d, length = a["root"]
    arm = variant_arm("a1_arm", coll, mats["petg_a"], pad, d, (0.0, 0.0), hz, length,
                      cut=[cylinder(a["thread"][0] / 2 + 0.02, hz - 1.0, hz + a["thread"][1] + 0.5, 32)])
    de, di, t, h0 = a["disc"]
    zh = z1 + 2 * (t + h0)
    parts = [cup, cap, stud, arm]
    for i, p in enumerate(pad_points()):
        at = Matrix.Translation(p)
        parts += [new_obj(f"a1_screw_{i + 1}", moved(screw_down(3.0, *P["m3_head"], a["screw_l"], zh), at), coll, mats["steel"], pad),
                  new_obj_parts(f"a1_discs_{i + 1}", [(moved(disc_spring(de, di, t, h0, z1, True), at), mats["steel"]),
                                                      (moved(disc_spring(de, di, t, h0, z1 + t + h0, False), at), mats["steel"])], coll, pad),
                  new_obj(f"a1_nut_{i + 1}", moved(nut(a["nut"][0], a["nut"][1], -P["pad_thickness"] - a["nut"][1] - 0.02, 3.0), at),
                          coll, mats["steel"], pad)]
    return parts


def build_a2(coll, mats, pad):
    """A2: fork on the pad, tongue of the arm between two PTFE washers, shoulder screw along x with two disc
    springs under the head and a lock nut: it turns only about x (sagittal impacts)."""
    a = P["a2"]
    R, H = a["foot"]
    gi, ct, cy = a["cheek"]
    za = a["axle_z"]
    sc = a["screw"]

    def yz(prof, x0, x1):
        return moved(extrude_polygon(prof, x0, x1), YZX)

    cheek = [(-cy, H - 0.5), (cy, H - 0.5)] + [(cy * math.cos(t), za + cy * math.sin(t)) for t in np.linspace(0.0, math.pi, 17)]
    hole = lambda: along_x(cylinder(sc["shoulder_d"] / 2 + 0.1, -20.0, 20.0, 32), 0.0, za)
    fork = part("a2_fork", [cylinder(R, 0.0, H, 96), yz(cheek, gi, gi + ct), yz(cheek, -gi - ct, -gi)], coll, mats["petg_a"], pad,
                cut=pad_inserts(H - 0.5) + [hole()])
    wo, wi, wt = a["washer"]
    tw = gi - wt
    lower = [(cy * math.cos(t), za + cy * math.sin(t)) for t in np.linspace(math.pi, 2 * math.pi, 17)]
    tongue = yz(lower + [(cy, za + a["tongue_top"]), (-cy, za + a["tongue_top"])], -tw, tw)
    d, length = a["root"]
    arm = variant_arm("a2_arm", coll, mats["petg_a"], pad, d, (0.0, 0.0), za + cy + 1.5, length, extra=[tongue], cut=[hole()])
    washers = [new_obj(f"a2_washer_{i + 1}", along_x(tube(wi / 2, wo / 2, x0, x0 + wt, 48), 0.0, za), coll, mats["ptfe"], pad)
               for i, x0 in enumerate((tw, -gi))]
    de, di, t, h0 = a["disc"]
    xo = gi + ct
    xh = xo + 2 * (t + h0)
    x_sh = xh - sc["shoulder_l"]
    x_th = x_sh - sc["thread_l"]
    screw = new_obj("a2_shoulder_screw", along_x(revolve([
        (0, x_th), (sc["thread_d"] / 2, x_th), (sc["thread_d"] / 2, x_sh), (sc["shoulder_d"] / 2, x_sh), (sc["shoulder_d"] / 2, xh),
        (sc["head_d"] / 2, xh), (sc["head_d"] / 2, xh + sc["head_k"]), (0, xh + sc["head_k"])], 48), 0.0, za), coll, mats["steel"], pad)
    discs = new_obj_parts("a2_discs", [(along_x(disc_spring(de, di, t, h0, xo, True), 0.0, za), mats["steel"]),
                                       (along_x(disc_spring(de, di, t, h0, xo + t + h0, False), 0.0, za), mats["steel"])], coll, pad)
    lock = new_obj("a2_nut", along_x(nut(a["nut"][0], a["nut"][1], -xo - a["nut"][1], a["screw"]["thread_d"]), 0.0, za), coll, mats["steel"], pad)
    return [fork, arm, screw, discs, lock] + washers + pad_screws("a2", coll, mats, pad)


def kinematic_levels():
    """Half spacing of the dowel pairs (45 degree contact), dowel axis height, ball centre and plate top."""
    b = P["b1"]
    rb, rp = b["ball_d"] / 2, b["pin"][0] / 2
    half = (rb + rp) * math.sqrt(0.5)
    z_pin = b["base"][1] - b["pin_sink"]
    z_ball = z_pin + (rb + rp) * math.sqrt(0.5)
    return half, z_pin, z_ball, z_ball + b["plate_h"]


def build_kinematic(prefix, coll, mats, pad, magnets):
    """B1 (central screw and compression spring) or B4 (two magnets and a cord): base with three V-grooves made
    of dowel pairs, plate with three balls and the arm. Under the impact the plate tips about two balls and the
    preload brings it back to the only position where all six contacts close."""
    b = P["b1"]
    R, H = b["base"]
    rb = b["ball_d"] / 2
    pin_d, pin_l = b["pin"]
    half, z_pin, z_ball, z_top = kinematic_levels()
    r_in = b["pitch_r"] - pin_l / 2
    base_cut, pins = pad_inserts(H - 1.0), []
    for k, deg in enumerate(b["groove_deg"]):
        rot = Matrix.Rotation(math.radians(deg), 4, "Z")
        radial = rot @ Vector((1.0, 0.0, 0.0))
        for off in (half, -half):
            m = frame(rot @ Vector((r_in, off, z_pin)), radial, (0.0, 0.0, 1.0))
            base_cut.append(moved(cylinder(pin_d / 2 + 0.02, -0.5, pin_l + 0.5, 32), m))
            pins.append((moved(cylinder(pin_d / 2, 0.0, pin_l, 32), m), mats["steel"]))
        base_cut.append(moved(box((r_in - 1.0, -(half - pin_d / 2), H - b["relief"]), (R + 1.0, half - pin_d / 2, H + 1.0)), rot))
    centres = [polar(b["pitch_r"], d, z_ball) for d in b["groove_deg"]]
    plate = [cylinder(R, z_ball, z_top, 96)]
    plate_cut = [sphere(rb + P["fit"], c) for c in centres]
    parts = []
    if magnets:
        md, mh = P["b4"]["magnet"]
        zm = z_ball - P["b4"]["gap"]
        base = part(f"{prefix}_base", [cylinder(R, 0.0, H, 96), cylinder(P["b4"]["boss_r"], H - 0.5, zm, 48)], coll, mats["petg_b"], pad,
                    cut=base_cut + [cylinder(md / 2 + 0.05, zm - mh, zm + 1.0, 32)])
        d, length = P["b4"]["root"]
        arm = variant_arm(f"{prefix}_plate", coll, mats["petg_b"], pad, d, (0.0, 0.0), z_top - 0.5, length, extra=plate,
                          cut=plate_cut + [cylinder(md / 2 + 0.05, z_ball - 1.0, z_ball + mh, 32)])
        parts += [new_obj_parts(f"{prefix}_magnets", [(cylinder(md / 2, zm - mh, zm, 32), mats["magnet"]),
                                                      (cylinder(md / 2, z_ball, z_ball + mh, 32), mats["magnet"])], coll, pad),
                  tether(f"{prefix}_tether", coll, mats, pad, Vector((0.0, R, z_top - 2.0)), Vector((0.0, R, 2.5)), (0.0, 1.0, 0.0))]
    else:
        ins_d, ins_l = b["insert"]
        base = part(f"{prefix}_base", [cylinder(R, 0.0, H, 96)], coll, mats["petg_b"], pad, cut=base_cut + [cylinder(ins_d / 2, -1.0, H + 1.0, 32)])
        d, y, length = b["root"]
        plate.append(moved(cylinder(d / 2, z_ball, z_top, 48), Matrix.Translation((0.0, y, 0.0))))
        arm = variant_arm(f"{prefix}_plate", coll, mats["petg_b"], pad, d, (0.0, y), z_top - 0.5, length, extra=plate,
                          cut=plate_cut + [cylinder(b["hole_d"] / 2, z_ball - 1.0, z_top + 1.0, 48)])
        wire, od, fitted = b["spring"]
        wo, wi, wt = b["washer"]
        zw = z_top + fitted
        parts += [new_obj(f"{prefix}_insert", tube(2.05, ins_d / 2 - 0.05, 0.3, ins_l - 0.3, 32), coll, mats["brass"], pad),
                  new_obj(f"{prefix}_spring", compression_spring(z_top, zw, wire, od), coll, mats["steel"], pad),
                  new_obj(f"{prefix}_washer", tube(wi / 2, wo / 2, zw, zw + wt, 48), coll, mats["steel"], pad),
                  new_obj(f"{prefix}_screw", screw_down(4.0, *P["m4_head"], b["screw_l"], zw + wt), coll, mats["steel"], pad)]
    balls = new_obj_parts(f"{prefix}_balls", [(sphere(rb, c), mats["steel"]) for c in centres], coll, pad)
    pin_ob = new_obj_parts(f"{prefix}_pins", pins, coll, pad)
    return [base, arm, balls, pin_ob] + parts + pad_screws(prefix, coll, mats, pad, 14.0)


def build_b2(coll, mats, pad):
    """B2: Thorlabs KB25/M, bottom plate on the pad with one M4 through the centre hole and a nut, top plate
    screwed under the arm; a cord keeps the marker if the magnets let go."""
    k = P["b2"]
    s, t = k["plate"]
    g = k["gap"]
    hs = s / 2
    zt = t + g
    seats = lambda: [sphere(k["ball_d"] / 2 + P["fit"], polar(k["ball_r"], d, t + g / 2)) for d in (90.0, 210.0, 330.0)]
    bot = part("b2_kb25_bottom", [box((-hs, -hs, 0.0), (hs, hs, t))], coll, mats["anodised"], pad,
               cut=[cylinder(4.1, t - 4.5, t + 1.0, 32), cylinder(2.25, -1.0, t, 32)] + seats())
    top = part("b2_kb25_top", [box((-hs, -hs, zt), (hs, hs, zt + t))], coll, mats["anodised"], pad,
               cut=[cylinder(4.1, zt - 1.0, zt + 4.5, 32), cylinder(2.25, zt, zt + t + 1.0, 32)] + seats())
    balls = new_obj_parts("b2_kb25_balls", [(sphere(k["ball_d"] / 2, polar(k["ball_r"], d, t + g / 2)), mats["steel"])
                                            for d in (90.0, 210.0, 330.0)], coll, pad)
    d, length = k["root"]
    z_arm = zt + t
    arm = variant_arm("b2_arm", coll, mats["petg_b"], pad, d, (0.0, 0.0), z_arm, length,
                      cut=[cylinder(2.05, z_arm - 1.0, z_arm + 11.5, 32)])
    hd, hk = P["m4_head"]
    return [bot, top, balls, arm,
            new_obj("b2_screw_bottom", screw_down(4.0, hd, hk, k["screw_l"], t - 4.5), coll, mats["steel"], pad),
            new_obj("b2_nut", nut(7.0, 3.2, -P["pad_thickness"] - 3.22, 4.0), coll, mats["steel"], pad),
            new_obj("b2_screw_top", socket_screw(4.0, hd, hk, 12.0, zt + 4.5), coll, mats["steel"], pad),
            tether("b2_tether", coll, mats, pad, Vector((0.0, hs, zt + t - 2.0)), Vector((0.0, hs, 2.5)), (0.0, 1.0, 0.0))]


def build_b3(coll, mats, pad):
    """B3: spring steel rod glued in the foot and in the arm, with a set screw in each socket: it bends under
    the impact and springs back, no contacts and no friction."""
    k = P["b3"]
    R, H = k["foot"]
    br, bh = k["boss"]
    rd, free = k["rod"]
    z_r0 = H - 2.0
    z_b1 = H + bh
    z_a0 = z_b1 + free
    z_r1 = z_a0 + k["socket"]
    d, length = k["root"]
    hole = lambda: cylinder(rd / 2 + 0.05, z_r0, z_r1 + 0.5, 32)
    grub = [along_x(cylinder(1.5, rd / 2, r_out + 0.5, 24), 0.0, z) for r_out, z in ((br, H + bh / 2), (d / 2, z_a0 + k["socket"] / 2))]
    foot = part("b3_foot", [cylinder(R, 0.0, H, 96), cylinder(br, H - 0.5, z_b1, 48)], coll, mats["petg_b"], pad,
                cut=pad_inserts(H - 0.5) + [hole(), grub[0]])
    arm = variant_arm("b3_arm", coll, mats["petg_b"], pad, d, (0.0, 0.0), z_a0, length, cut=[hole(), grub[1]])
    rod = new_obj("b3_rod", cylinder(rd / 2, z_r0, z_r1, 32), coll, mats["steel"], pad)
    grubs = new_obj_parts("b3_set_screws", [(along_x(cylinder(1.5, rd / 2 + 0.02, r_out - 0.3, 24), 0.0, z), mats["anodised"])
                                            for r_out, z in ((br, H + bh / 2), (d / 2, z_a0 + k["socket"] / 2))], coll, pad)
    return [foot, arm, rod, grubs] + pad_screws("b3", coll, mats, pad)


def build_interfaces(colls, mats, pad):
    """Every variant in its own collection under 'interfaces'; show(variant=...) picks the visible one."""
    out = {}
    for key, builder in (("a1", build_a1), ("a2", build_a2), ("b1", lambda c, m, p: build_kinematic("b1", c, m, p, False)),
                         ("b2", build_b2), ("b3", build_b3), ("b4", lambda c, m, p: build_kinematic("b4", c, m, p, True))):
        out[key] = builder(colls[VARIANTS[key][0]], mats, pad)
    return out


# ---------------------------------------------------------------- ground truth, silicone, mount

def build_ground_truth(coll, mats, mandible, marker_centres):
    gd = P["gt_sphere_d"]
    spheres = []
    for i, c in enumerate(marker_centres):
        ob = new_obj(f"gt_sphere_marker_{i + 1}", sphere(gd / 2), coll, mats["sphere"], Matrix.Translation(c))
        spheres.append(ob)
    tree = bvh_world(mandible)
    pd, pl = P["gt_mandible_post"]
    posts = []
    for i, (guess, direction) in enumerate(P["gt_mandible"]):
        d = Vector(direction).normalized()
        hit = tree.ray_cast(Vector(guess) + d * 60.0, -d)[0]
        assert hit is not None, f"no mandible surface for sphere {i + 1}"
        post = new_obj(f"gt_post_mandible_{i + 1}", moved(cylinder(pd / 2, 2.0, pl, 24), frame(hit, d)),
                       coll, mats["petg_fixture"])
        center_origin(post)
        posts.append(post)
        spheres.append(new_obj(f"gt_sphere_mandible_{i + 1}", sphere(gd / 2), coll, mats["sphere"],
                               Matrix.Translation(hit + d * (pl + gd / 2))))
    return spheres, posts


def window_frame(mandible):
    w = P["window"]
    d = Vector(w["normal"]).normalized()
    hit = bvh_world(mandible).ray_cast(Vector(w["guess"]) + d * 60.0, -d)[0]
    assert hit is not None, "no mandible surface for the window"
    major = Vector((-d.y, d.x, 0.0)).normalized()
    return frame(hit, d, major), hit


def build_silicone(colls, mats, ph, tmp, extra_cut, window_m):
    """Soft tissue cast on the phantom, with the surgical window and the slot for the splint bar."""
    s = P["silicone"]
    t = s["thickness"]
    cr, md = ("obj", ph["cranium"], t + 1.0), ("obj", ph["mandible"], t + 1.0)
    region = new_obj("tmp_region", box((s["x"][0], s["y"][0], s["z"][0]), (s["x"][1], s["y"][1], s["z"][1])), tmp)
    a, b = P["window"]["size"]
    win = new_obj("tmp_window", moved(cylinder(1.0, -30.0, 30.0, 64), Matrix.Diagonal((a / 2, b / 2, 1.0, 1.0))), tmp,
                  matrix=window_m)
    o = ph["pad_origin"]
    mouth = new_obj("tmp_mouth", box((-26.0, o.y, o.z - P["bar_height"] - 2.0), (26.0, o.y + 70.0, o.z + 3.0)), tmp)
    outer = ("mean", ("intersect", [("union", [("offset", cr, t), ("offset", md, t)]), ("obj", region, 0.0)]), *s["smooth"])
    bone = [("offset", cr, 0.6), ("offset", md, 0.6)]
    cut = bone + [("offset", ("obj", ob, 2.0), 1.0) for ob in extra_cut] + [("obj", win, 0.0), ("obj", mouth, 0.0)]
    cover = mesh_obj("silicone_cover", sdf_eval(("diff", outer, cut), s["voxel"]), colls["silicone"], mats["silicone"])
    center_origin(cover)
    return cover


# ---------------------------------------------------------------- phantom stage, cached

STAGE_KEYS = ("skull_url", "mandible_url", "condyle_drop", "opening_step_deg", "opening_margin_deg", "wafer_min", "tmj_space",
              "voxel", "splint_wall", "splint_clearance", "splint_depth", "bar_width", "bar_height", "bar_length", "pad_x",
              "pad_y", "pad_thickness", "m3_clearance_d", "pad_holes", "imf_plate", "imf_maxilla", "imf_mandible",
              "gt_sphere_d", "gt_mandible_post", "gt_mandible", "silicone", "window")
STAGE_VERSION = 1   # raise it when the code of the phantom, splint, wafer, plates, mandible spheres or silicone changes


def stage_path():
    import hashlib
    blob = json.dumps({k: P[k] for k in STAGE_KEYS}, sort_keys=True) + f"/{STAGE_VERSION}"
    return CACHE / f"phantom_stage_{hashlib.sha1(blob.encode()).hexdigest()[:12]}.blend"


def build_stage(colls, mats, tmp):
    """Phantom, splint, intermaxillary fixation, mandible spheres and silicone: the slow part (signed distance
    fields, about 10 minutes), cached in CACHE under the values of P it depends on."""
    ph = build_phantom_and_splint(colls, mats, tmp)
    imf = build_imf(ph, colls["imf"], mats, tmp)
    spheres, posts = build_ground_truth(colls["ground_truth"], mats, ph["mandible"], [])
    window_m, window_c = window_frame(ph["mandible"])
    cover = build_silicone(colls, mats, ph, tmp, [ph["splint"]] + imf + posts + spheres, window_m)
    info = {k: ph[k] for k in ("opening_deg", "splint_cranium_overlaps", "tmj_drop", "condyle_overlaps")}
    info["pad_origin"] = list(ph["pad_origin"])
    info["window_c"] = list(window_c)
    return {"cranium": ph["cranium"], "mandible": ph["mandible"], "splint": ph["splint"], "imf": imf,
            "spheres": spheres, "posts": posts, "silicone": cover}, info


def stage_objects(st):
    return [st["cranium"], st["mandible"], st["splint"], st["silicone"]] + st["imf"] + st["spheres"] + st["posts"]


def save_stage(path, st, info):
    for ob in stage_objects(st):
        ob["stage_collection"] = ob.users_collection[0].name
    st["splint"]["stage_info"] = json.dumps(info)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.data.libraries.write(str(path), set(stage_objects(st)), fake_user=False, compress=True)


def load_stage(path, colls, mats):
    with bpy.data.libraries.load(str(path), link=False) as (src, dst):
        dst.objects = list(src.objects)
    for ob in dst.objects:
        colls[ob["stage_collection"]].objects.link(ob)
        for slot in ob.material_slots:
            base = slot.material.name.split(".")[0]
            if base in mats and slot.material is not mats[base]:
                slot.material = mats[base]
    for m in list(bpy.data.materials):
        if m.users == 0 and m.name not in mats:
            bpy.data.materials.remove(m)
    by = {ob.name: ob for ob in dst.objects}
    st = {"cranium": by["phantom_cranium"], "mandible": by["phantom_mandible"], "splint": by["splint"],
          "silicone": by["silicone_cover"], "imf": sorted((o for o in dst.objects if o.name.startswith("imf_")), key=lambda o: o.name),
          "spheres": sorted((o for o in dst.objects if o.name.startswith("gt_sphere_mandible")), key=lambda o: o.name),
          "posts": sorted((o for o in dst.objects if o.name.startswith("gt_post_mandible")), key=lambda o: o.name)}
    return st, json.loads(st["splint"]["stage_info"])


def marker_spheres(coll, mats, centres):
    return [new_obj(f"gt_sphere_marker_{i + 1}", sphere(P["gt_sphere_d"] / 2), coll, mats["sphere"], Matrix.Translation(c))
            for i, c in enumerate(centres)]


def build_mount(colls, mats, cranium, z_b):
    """Block bonded under the cranial base with three V-grooves, kinematic cradle with three balls on
    four 2020 posts, clamp rod with a knob under the base plate."""
    mp = P["mount"]
    coll = colls["mount"]
    tree = bvh_world(cranium)
    xs = np.arange(-mp["x"], mp["x"] + 1e-6, mp["grid"])
    ys = np.arange(mp["y"][0], mp["y"][1] + 1e-6, mp["grid"])
    zb = mp["bottom"]
    top = np.full((len(xs), len(ys)), mp["cap"])
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            hit = tree.ray_cast(Vector((x, y, zb - 10.0)), Vector((0.0, 0.0, 1.0)))[0]
            if hit is not None:
                top[i, j] = min(hit.z, mp["cap"])
    pad_top = np.pad(top, 2, mode="edge")
    top = np.min([pad_top[2 + di:2 + di + len(xs), 2 + dj:2 + dj + len(ys)] for di in (-2, -1, 0, 1, 2) for dj in (-2, -1, 0, 1, 2)], axis=0) - 0.8
    bm = bmesh.new()
    T = [[bm.verts.new((x, y, top[i, j])) for j, y in enumerate(ys)] for i, x in enumerate(xs)]
    B = [[bm.verts.new((x, y, zb)) for j, y in enumerate(ys)] for i, x in enumerate(xs)]
    nx, ny = len(xs), len(ys)
    for i in range(nx - 1):
        for j in range(ny - 1):
            bm.faces.new((T[i][j], T[i + 1][j], T[i + 1][j + 1], T[i][j + 1]))
            bm.faces.new((B[i][j], B[i][j + 1], B[i + 1][j + 1], B[i + 1][j]))
    for i in range(nx - 1):
        bm.faces.new((T[i][0], B[i][0], B[i + 1][0], T[i + 1][0]))
        bm.faces.new((T[i][ny - 1], T[i + 1][ny - 1], B[i + 1][ny - 1], B[i][ny - 1]))
    for j in range(ny - 1):
        bm.faces.new((T[0][j], T[0][j + 1], B[0][j + 1], B[0][j]))
        bm.faces.new((T[nx - 1][j], B[nx - 1][j], B[nx - 1][j + 1], T[nx - 1][j + 1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    # block underside: hole and slot for the two dowels (the slot runs along the line of the pins, so the
    # block is located by the hole and oriented by the slot), heat-set M6 insert in the middle
    cy = (mp["y"][0] + mp["y"][1]) / 2
    hx, (y0, y1) = mp["x"], mp["y"]
    pw, pt = mp["plate"]
    dd, dl = mp["dowel"]
    dy, slot = mp["dowel_dy"], mp["slot"]
    ins_d, ins_l = mp["insert"]
    kr, kh = mp["knob"]
    r = dd / 2 + P["fit"]
    z_plate = zb - P["fit"]                       # plate top; the block rests on it
    depth = z_plate - pt + dl - zb + 2.0          # dowel tip plus 2 mm
    at = lambda x, y: Matrix.Translation((x, y, 0.0))
    mount = new_obj("phantom_mount", bm, coll, mats["petg_fixture"])
    boolean(mount, [moved(cylinder(r, zb - 1.0, zb + depth, 32), at(0.0, cy - dy)),
                    moved(cylinder(r, zb - 1.0, zb + depth, 32), at(0.0, cy + dy - slot / 2)),
                    moved(cylinder(r, zb - 1.0, zb + depth, 32), at(0.0, cy + dy + slot / 2)),
                    box((-r, cy + dy - slot / 2, zb - 1.0), (r, cy + dy + slot / 2, zb + depth)),
                    moved(cylinder(ins_d / 2, zb - 1.0, zb + ins_l, 32), at(0.0, cy))], "DIFFERENCE")
    insert = new_obj("phantom_insert", tube(3.05, ins_d / 2 - 0.05, zb + 0.5, zb + ins_l - 0.5, 32), coll, mats["brass"], at(0.0, cy))

    # plate screwed on the end faces of the two posts (M5 into the central bore), dowels pressed through it
    px = mp["post_x"]
    plate = part("phantom_plate", [box((-pw, y0, z_plate - pt), (pw, y1, z_plate))], coll, mats["aluminium"],
                 cut=[moved(cylinder(dd / 2 + 0.02, z_plate - pt - 1.0, z_plate + 1.0, 32), at(0.0, y)) for y in (cy - dy, cy + dy)]
                 + [moved(cylinder(3.3, z_plate - pt - 1.0, z_plate + 1.0, 32), at(0.0, cy))])
    dowels = [new_obj(f"phantom_dowel_{i + 1}", cylinder(dd / 2, z_plate - pt, z_plate - pt + dl, 32), coll, mats["steel"], at(0.0, y))
              for i, y in enumerate((cy - dy, cy + dy))]
    knob = part("phantom_clamp_knob", [cylinder(3.0, z_plate - pt - kh + 4.0, zb + ins_l - 1.0, 24),
                                       cylinder(kr, z_plate - pt - kh, z_plate - pt - P["fit"], 32)],
                coll, mats["rubber"], at(0.0, cy))
    s = P["profile"]
    posts, brackets = [], []
    for i, sx in enumerate((-1.0, 1.0)):
        x = sx * px
        name = f"phantom_post_{i + 1}"
        posts.append(new_obj(name, profile_bar((x, cy, z_b), (x, cy, z_plate - pt), s), coll, mats["profile"]))
        brackets += post_brackets(name, coll, mats["aluminium"], x, cy, z_b, ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (sx, 0.0, 0.0)))
        for j, n in enumerate(((0.0, 1.0, 0.0), (0.0, -1.0, 0.0))):
            brackets.append(corner_bracket(f"{name}_top_bracket_{j + 1}", coll, mats["aluminium"],
                                           Vector((x, cy, z_plate - pt)) + Vector(n) * s / 2, n, (0.0, 0.0, -1.0)))
    parts = [mount, plate, insert, knob] + dowels + posts + brackets
    for ob in parts:
        center_origin(ob)
    return parts, Vector((0.0, cy, 0.0))


# ---------------------------------------------------------------- bench: base, pendulum, phone, scanner

def raised_dir(direction_deg):
    """Horizontal direction towards which the pendulum is raised: 0 anterior, 90 towards the patient's right."""
    a = math.radians(direction_deg)
    return Vector((-math.sin(a), -math.cos(a), 0.0))


def pendulum_geometry(strike, z_b):
    pp = P["pendulum"]
    bd = pp["block"][1]
    y_p = -(P["strike_cap"][0] / 2 + bd / 2 + pp["tip_shank"] + pp["nose_r"])
    z_p = strike.z - z_b + pp["L"]
    return y_p, z_p, z_p + pp["axle_drop"]


def hull2d(pts):
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def kfl08(name, coll, mats, centre, outward, parent):
    """KFL08 two-bolt flange unit (zinc alloy housing with a 608 bearing), flange face on `centre`, housing
    towards `outward`, bolt line along y, two M4 button heads."""
    k = P["pendulum"]["kfl08"]
    re = (k["length"] - k["holes"]) / 2
    pts = [(cy + r * math.cos(t), r * math.sin(t)) for cy, r in ((-k["holes"] / 2, re), (k["holes"] / 2, re), (0.0, k["width"] / 2))
           for t in np.linspace(0.0, 2 * math.pi, 32, endpoint=False)]
    m = frame(centre, outward, (0.0, 1.0, 0.0))
    items = [(moved(extrude_section(hull2d(pts), 0.02, k["flange"], 4.02, 48), m), mats["aluminium"]),
             (moved(tube(4.02, k["boss_d"] / 2, k["flange"] - 0.1, k["housing"], 48), m), mats["aluminium"]),
             (moved(tube(4.02, k["collar_d"] / 2, k["housing"] - 0.1, k["total"], 48), m), mats["steel"])]
    items += [(moved(revolve([(0, k["flange"]), (3.8, k["flange"]), (3.8, k["flange"] + 1.2), (2.0, k["flange"] + 2.2), (0, k["flange"] + 2.2)], 32),
                     m @ Matrix.Translation((s * k["holes"] / 2, 0.0, 0.0))), mats["steel"]) for s in (-1.0, 1.0)]
    ob = new_obj_parts(name, items, coll)
    child_of(ob, parent)
    return ob


def build_pendulum(coll, mats, strike, z_b):
    """Twin 2020 portal on a turntable that rotates about the fixed pivot shaft under the strike point.

    Local frame of pendulum_base: origin on the base board under the strike point, the pendulum is raised
    towards -y and swings in the plane x = 0; pendulum_swing rotates about the axle (x)."""
    pp = P["pendulum"]
    L, Rc, s = pp["L"], pp["R_c"], P["profile"]
    hs = s / 2
    tw, tt = pp["turntable"]
    xb = pp["frame_width"] / 2
    bw, bd, bh = pp["block"]
    y_p, z_p, z_top = pendulum_geometry(strike, z_b)
    base = empty("pendulum_base", coll, 40.0)
    base.matrix_world = Matrix.Translation((strike.x, strike.y, z_b)) @ Matrix.Rotation(math.radians(-pp["direction_deg"]), 4, "Z")

    def local(name, bms, mat, parent=base, cut=()):
        """Part built in the parent frame (booleans before parenting, so cutters share its frame)."""
        ob = part(name, bms if isinstance(bms, list) else [bms], coll, mat, cut=cut)
        child_of(ob, parent)
        return ob

    def yz(prof, x0, x1):
        return moved(extrude_section(prof, x0, x1), YZX)

    shaft_r = P["arm"]["shaft"][0] / 2
    parts = [local("pendulum_turntable", box((-tw / 2, -Rc - pp["turntable_front"], 0.0), (tw / 2, pp["turntable_back"], tt)),
                   mats["aluminium"], cut=[cylinder(shaft_r + 0.1, -1.0, tt + 1.0, 32),
                                           moved(cylinder(4.1, -1.0, tt + 1.0, 32), Matrix.Translation((0.0, -pp["R_index"], 0.0)))]),
             local("pendulum_index_pin", moved(revolve([(0, -P["base_t"]), (4.0, -P["base_t"]), (4.0, tt + 4.0), (10.0, tt + 4.0),
                                                        (10.0, tt + 16.0), (0, tt + 16.0)], 32),
                                               Matrix.Translation((0.0, -pp["R_index"], 0.0))), mats["rubber"]),
             local("pendulum_controller", box((-35.0, -Rc - hs - 26.0, tt + 60.0), (35.0, -Rc - hs, tt + 110.0)), mats["electronics"])]
    y_end = y_p + 50.0
    H = pp["strut_foot"] - hs
    for side, sx in (("left", 1.0), ("right", -1.0)):
        x = sx * xb
        a = Vector((x, -Rc + hs, tt + H))
        b = Vector((x, -Rc + hs + H, tt))
        e = (b - a).normalized() * 15.0
        parts += [
            local(f"pendulum_column_{side}", profile_bar((x, -Rc, tt), (x, -Rc, z_top), s), mats["profile"]),
            local(f"pendulum_beam_{side}", profile_bar((x, -Rc - hs, z_top + hs), (x, y_end, z_top + hs), s), mats["profile"]),
            local(f"pendulum_strut_{side}", profile_bar(a - e, b + e, s), mats["profile"],
                  cut=[box((x - 15, -Rc - 200, -100), (x + 15, -Rc + hs, z_top)), box((x - 15, -Rc - 200, -200), (x + 15, 200, tt))]),
        ]
        gusset = [(-Rc - hs, z_top + s), (-Rc + 90.0, z_top + s), (-Rc + 90.0, z_top), (-Rc + hs, z_top - 80.0), (-Rc - hs, z_top - 80.0)]
        x0 = xb + hs if sx > 0 else -(xb + hs) - 4.0
        parts.append(local(f"pendulum_gusset_{side}", moved(extrude_polygon(gusset, 0.0, 4.0),
                                                             frame((x0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))), mats["aluminium"]))
    for i, (n, x) in enumerate((((1.0, 0.0, 0.0), xb), ((-1.0, 0.0, 0.0), -xb), ((0.0, 1.0, 0.0), xb), ((0.0, 1.0, 0.0), -xb))):
        br = corner_bracket(f"pendulum_bracket_{i + 1}", coll, mats["aluminium"], Vector((x, -Rc, tt)) + Vector(n) * hs, n, (0.0, 0.0, 1.0))
        child_of(br, base)
        parts.append(br)
    for i, (x, y) in enumerate(((xb, -Rc), (-xb, -Rc))):
        x0, x1 = (xb - hs - 20.0, xb + hs) if x > 0 else (-xb - hs, -xb + hs + 20.0)
        parts.append(local(f"pendulum_joint_plate_{i + 1}", box((x0, y - 25.0, z_top + s), (x1, y + 25.0, z_top + s + 3.0)), mats["aluminium"]))
    for name, y, z in (("pendulum_crossbar_top", -Rc, z_top + hs), ("pendulum_crossbar_back", y_p + 40.0, z_top + hs),
                       ("pendulum_crossbar_bottom", -Rc, tt + hs)):
        parts.append(local(name, profile_bar((-xb + hs, y, z), (xb - hs, y, z), s, (0.0, 1.0, 0.0)), mats["profile"]))

    # drop plates under the beams: the left one is also the protractor (holes every 5 deg at holes_r, arc slot
    # for the servo shaft at latch_r); KFL08 units on their outer faces, axle and spacers between them
    dp = pp["drop_plate"]
    x_in = xb - hs
    sr, a0, a1 = pp["sector"]
    lr = pp["latch_r"]
    sv = pp["servo"]
    horn_l, horn_t = sv["horn"]
    pin_d, pin_l = pp["pin"]
    hook_back = pin_d / 2 + 1.5
    delta = math.degrees((horn_l - hook_back) / lr)          # servo shaft ahead of the latch pin
    th = pp["release_deg"]

    def on_arc(r, deg, x=0.0):
        a = math.radians(deg)
        return Vector((x, y_p - r * math.sin(a), z_p - r * math.cos(a)))

    top = z_top + s
    left = [(y_p + 30.0, top), (y_p - 30.0, top), (y_p - 30.0, z_p)] + [(on_arc(sr, a).y, on_arc(sr, a).z) for a in np.linspace(a1, a0, 40)] \
        + [(y_p + 30.0, z_p - 60.0)]
    right = [(y_p + 30.0, top), (y_p - 30.0, top), (y_p - 30.0, z_p - 22.0), (y_p + 30.0, z_p - 22.0)]
    kh = pp["kfl08"]["holes"] / 2
    common = lambda: [along_x(cylinder(4.5, -60.0, 60.0, 32), y_p, z_p)] + [along_x(cylinder(2.6, -60.0, 60.0, 16), y_p + sy * kh, z_p) for sy in (-1, 1)] \
        + [along_x(cylinder(2.75, -60.0, 60.0, 16), y_p + sy * 18.0, z_top + hs) for sy in (-1, 1)]
    holes = [along_x(cylinder(1.6, -60.0, 60.0, 16), on_arc(pp["holes_r"], a).y, on_arc(pp["holes_r"], a).z) for a in range(0, int(a1) - 1, 5)]
    lo, hi = 5.0 + delta - 2.0, pp["max_release_deg"] + delta + 2.0
    band = [(on_arc(lr + pp["slot_w"] / 2, a).y, on_arc(lr + pp["slot_w"] / 2, a).z) for a in np.linspace(lo, hi, 30)] \
        + [(on_arc(lr - pp["slot_w"] / 2, a).y, on_arc(lr - pp["slot_w"] / 2, a).z) for a in np.linspace(hi, lo, 30)]
    parts += [local("pendulum_protractor_plate", yz(left, x_in - dp, x_in), mats["aluminium"], cut=common() + holes + [yz(band, x_in - dp - 1.0, x_in + 1.0)]),
              local("pendulum_drop_plate", yz(right, -x_in, -x_in + dp), mats["aluminium"], cut=common()),
              kfl08("pendulum_kfl08_left", coll, mats, Vector((x_in, y_p, z_p)), (1.0, 0.0, 0.0), base),
              kfl08("pendulum_kfl08_right", coll, mats, Vector((-x_in, y_p, z_p)), (-1.0, 0.0, 0.0), base)]
    k = pp["kfl08"]
    parts.append(local("pendulum_axle", along_x(cylinder(4.0, -x_in - k["total"] - 2.0, x_in + k["total"] + 2.0, 32), y_p, z_p), mats["steel"]))
    parts += [local(f"pendulum_spacer_{i + 1}", along_x(tube(4.05, pp["spacer_d"] / 2, xa, xz, 48), y_p, z_p), mats["steel"])
              for i, (xa, xz) in enumerate(((8.0, x_in - dp), (-x_in + dp, -8.0)))]

    # MG90S on the outer face of the protractor at the release angle: PETG bracket with a pin in the hole at th and
    # a thumb screw in the hole at th + 10, shaft through the arc slot, horn with the hook that holds the latch pin
    sa = th + delta
    a = math.radians(sa)
    X, Y, Z = Vector((0.0, -math.cos(a), math.sin(a))), Vector((0.0, math.sin(a), math.cos(a))), Vector((-1.0, 0.0, 0.0))
    shaft = on_arc(lr, sa, x_in)
    sL, sW, sH = sv["body"]
    xs = -sL / 2 + sv["shaft_x"]
    S = Matrix((X, Y, Z)).transposed().to_4x4()
    S.translation = shaft - X * xs - Z * sv["top"]
    el, et, ez, ehs = sv["ears"]
    zb_top = sv["top"]
    z_ear = ez + et
    rin = (lr - pp["holes_r"]) + 4.0
    pin_hole = on_arc(pp["holes_r"], th, x_in - dp - 0.5)
    screw_hole = on_arc(pp["holes_r"], th + 10.0, x_in - dp)
    parts.append(local("pendulum_servo_bracket", moved(box((-el / 2 - 2.0, -sW / 2 - 4.0, z_ear), (el / 2 + 2.0, rin, zb_top)), S), mats["petg_fixture"],
                       cut=[moved(box((-sL / 2 - 0.2, -sW / 2 - 0.2, z_ear - 1.0), (sL / 2 + 0.2, sW / 2 + 0.2, zb_top + 1.0)), S)]
                       + [moved(cylinder(1.0, z_ear - 1.0, z_ear + 4.0, 16), S @ Matrix.Translation((sx * ehs / 2, 0.0, 0.0))) for sx in (-1, 1)]
                       + [along_x(cylinder(1.52, x_in - 1.0, x_in + 4.0, 16), pin_hole.y, pin_hole.z),
                          along_x(cylinder(1.56, x_in - 1.0, x_in + 6.0, 24), screw_hole.y, screw_hole.z)]))
    servo_bms = [box((-sL / 2, -sW / 2, 0.0), (sL / 2, sW / 2, sH)), box((-el / 2, -sW / 2, ez), (el / 2, sW / 2, z_ear)),
                 moved(cylinder(5.9, sH - 0.1, zb_top, 32), Matrix.Translation((xs, 0.0, 0.0))),
                 moved(cylinder(sv["spline"][0] / 2, zb_top - 0.1, sv["spline"][1], 24), Matrix.Translation((xs, 0.0, 0.0)))]
    parts.append(local("pendulum_servo", [moved(b, S) for b in servo_bms], mats["electronics"],
                       cut=[moved(cylinder(1.0, ez - 1.0, z_ear + 1.0, 16), S @ Matrix.Translation((sx * ehs / 2, 0.0, 0.0))) for sx in (-1, 1)]))
    z_h0 = zb_top + dp + 1.0
    z_hook = zb_top + x_in - (pin_l / 2 + pp["arm_d"] / 2)
    horn = local("pendulum_servo_horn", [moved(cylinder(3.0, sv["spline"][1] + 0.02, z_h0 + 0.1, 24), S @ Matrix.Translation((xs, 0.0, 0.0))),
                                        moved(cylinder(4.0, z_h0, z_h0 + horn_t, 24), S @ Matrix.Translation((xs, 0.0, 0.0))),
                                        moved(box((xs - horn_l - 1.5, -3.0, z_h0), (xs, 3.0, z_h0 + horn_t)), S),
                                        moved(box((xs - horn_l - 1.5, -3.0, z_h0), (xs - horn_l + 1.5, 3.0, z_hook)), S)],
                 mats["petg_white"])
    bpy.context.view_layer.update()
    set_origin(horn, base.matrix_world @ shaft)
    parts.append(horn)
    parts += [local("pendulum_servo_pin", along_x(cylinder(1.5, x_in - dp + 0.5, x_in + 3.0, 16), pin_hole.y, pin_hole.z), mats["steel"]),
              local("pendulum_servo_thumb_screw", moved(revolve([(0, -4.0), (5.0, -4.0), (5.0, 0.0), (1.5, 0.0), (1.5, dp + 4.0), (0, dp + 4.0)], 24),
                                                        frame(screw_hole, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))), mats["steel"])]

    swing = empty("pendulum_swing", coll, 30.0)
    swing.parent = base
    swing.matrix_parent_inverse = Matrix()
    swing.matrix_basis = Matrix.Translation((0.0, y_p, z_p)) @ Matrix.Rotation(math.radians(-th), 4, "X")
    hw = pp["washer"]
    nr, sh = pp["nose_r"], pp["tip_shank"]
    z_block = -L + bh / 2
    arm_r = pp["arm_d"] / 2
    swing_parts = [
        local("pendulum_hub", [along_x(cylinder(10.0, -8.0, 8.0, 48), 0.0, 0.0), box((-8.0, -8.0, -24.0), (8.0, 8.0, 0.0))],
              mats["aluminium"], swing,
              cut=[along_x(cylinder(4.05, -9.0, 9.0, 32), 0.0, 0.0), cylinder(arm_r + 0.05, -25.0, -10.0, 32)]),
        local("pendulum_arm", cylinder(arm_r, z_block - 8.0, -10.0, 32), mats["aluminium"], swing,
              cut=[along_x(cylinder(pin_d / 2 + 0.05, -6.0, 6.0, 16), 0.0, -lr)]),
        local("pendulum_mass_block", box((-bw / 2, -bd / 2, -L - bh / 2), (bw / 2, bd / 2, z_block)), mats["petg_fixture"], swing,
              cut=[cylinder(arm_r + 0.05, z_block - 8.0, z_block + 1.0, 32), along_x(cylinder(4.05, -bw, bw, 6), 0.0, -L)]),
        # solid PETG tip (100 % infill): cylindrical shank and hemispherical nose, replaced when worn
        local("pendulum_tip", moved(revolve([(0, 0.0), (nr, 0.0)] + [(nr * math.cos(t), sh + nr * math.sin(t))
                                                          for t in np.radians(np.linspace(0, 90, 12))[:-1]]
                                            + [(0, sh + nr)], 32),
                                    frame((0.0, bd / 2, -L), (0.0, 1.0, 0.0))), mats["petg_white"], swing),
        local("pendulum_mass_bolt", along_x(revolve([(0, -27.0), (6.5, -27.0), (6.5, -21.0), (4.0, -21.0), (4.0, 21.0),
                                                     (6.5, 21.0), (6.5, 27.0), (0, 27.0)], 6), 0.0, -L), mats["steel"], swing),
        local("pendulum_latch_pin", along_x(cylinder(pin_d / 2, -arm_r + 1.0, arm_r + pin_l, 16), 0.0, -lr), mats["steel"], swing),
    ]
    n = pp["washers_per_side"]
    for side in (1.0, -1.0):
        for k in range(n):
            a0_ = side * (bw / 2 + k * hw[2])
            a1_ = a0_ + side * hw[2]
            swing_parts.append(local(f"pendulum_mass_washer_{'l' if side > 0 else 'r'}{k + 1}",
                                     along_x(tube(hw[1] / 2, hw[0] / 2, min(a0_, a1_), max(a0_, a1_), 48), 0.0, -L), mats["steel"], swing))
    return base, swing, parts + swing_parts


def build_index_bushings(coll, mats, strike, z_b):
    out = []
    for d in (0.0, 45.0, 90.0):
        c = Vector((strike.x, strike.y, 0.0)) + raised_dir(d) * P["pendulum"]["R_index"]
        out.append(new_obj(f"pendulum_index_bushing_{int(d):03d}", tube(4.05, 6.95, z_b - P["base_t"], z_b, 32), coll,
                           mats["steel"], Matrix.Translation((c.x, c.y, 0.0))))
    return out


def corner_bracket(name, coll, mat, corner, u, v, leg=20.0, width=17.0, t=3.0):
    """Cast corner bracket for 2020 extrusions with its two M5 button head screws (ISO 7380): inner corner at
    `corner`, legs along the unit vectors u and v."""
    u, v = Vector(u).normalized(), Vector(v).normalized()
    m = Matrix((u, v, u.cross(v))).transposed().to_4x4()
    m.translation = Vector(corner)
    body = extrude_polygon([(0, 0), (leg, 0), (leg, t), (t, t), (t, leg), (0, leg)], -width / 2, width / 2)
    rib = extrude_polygon([(t, t), (leg - 3.0, t), (t, leg - 3.0)], -1.0, 1.0)
    head = [(0, 0.0), (4.75, 0.0), (4.75, 1.0), (3.0, 2.75), (0, 2.75)]
    screws = [moved(revolve(head, 24), frame((leg * 0.6, t, 0.0), (0.0, 1.0, 0.0))),
              moved(revolve(head, 24), frame((t, leg * 0.6, 0.0), (1.0, 0.0, 0.0)))]
    steel = bpy.data.materials["steel"]
    return new_obj_parts(name, [(moved(body, m), mat), (moved(rib, m), mat)] + [(moved(sc, m), steel) for sc in screws], coll)


def post_brackets(prefix, coll, mat, x, y, z0, normals):
    """Corner brackets at the foot of a vertical 2020 post standing on a horizontal face at z0."""
    hs = P["profile"] / 2
    return [corner_bracket(f"{prefix}_bracket_{i + 1}", coll, mat, Vector((x, y, z0)) + Vector(n) * hs, n, (0.0, 0.0, 1.0))
            for i, n in enumerate(normals)]


# ---------------------------------------------------------------- sensor arm: scanner and phone

def arm_dirs(phi):
    """Outward and tangential unit vectors of the sensor arm at angle phi about the pivot (0 anterior)."""
    a = math.radians(phi)
    return Vector((math.sin(a), -math.cos(a), 0.0)), Vector((math.cos(a), math.sin(a), 0.0))


def turn_about(pivot, deg):
    return Matrix.Translation((pivot.x, pivot.y, 0.0)) @ Matrix.Rotation(math.radians(deg), 4, "Z") @ Matrix.Translation((-pivot.x, -pivot.y, 0.0))


def azimuth_of(p, centre):
    """Azimuth of p seen from centre: 0 anterior, positive towards the patient's left."""
    return math.degrees(math.atan2(p.x - centre.x, -(p.y - centre.y)))


def arm_angle_for(point, pivot, centre, az, lo=-60.0, hi=170.0):
    """Arm angle that brings point (given at arm angle 0) to azimuth az seen from centre."""
    f = lambda phi: azimuth_of(turn_about(pivot, phi) @ point, centre) - az
    a, b = lo, hi
    for _ in range(60):
        m = (a + b) / 2
        if f(a) * f(m) <= 0.0:
            b = m
        else:
            a = m
    return (a + b) / 2


def phone_pose(lens, target):
    """Camera frame (looks along -z, y up) and phone body frame (back face towards the target, y up)."""
    d = (lens - target).normalized()
    a = math.radians(azimuth_of(lens, target))
    return frame(lens, d, (math.cos(a), math.sin(a), 0.0)), frame(lens, -d, (-math.cos(a), -math.sin(a), 0.0))


def solve_arm(pivot, target, centroid):
    """Arm angles of the four photo positions (camera azimuths about the target), lens and scanner glass at arm
    angle 0. The phone keeps its mean distance and has its centre on the radial line of the arm at the shown
    position; the scanner keeps at least min_distance from the sphere centroid at every position."""
    ph, sp = P["phone"], P["scanner"]
    W, Hh, D = ph["size"]
    lx, ly = ph["lens_offset"]
    hp = ph["distance"] * math.sin(math.radians(ph["elevation_deg"]))
    hs = sp["min_distance"] * math.sin(math.radians(sp["elevation_deg"]))
    u0, t0 = arm_dirs(0.0)
    r_l, t_l = ph["distance"] * math.cos(math.radians(ph["elevation_deg"])), 0.0
    for _ in range(8):
        lens0 = Vector((pivot.x, pivot.y, target.z + hp)) + u0 * r_l + t0 * t_l
        phis = [arm_angle_for(lens0, pivot, target, az) for az in ph["azimuths_deg"]]
        dist = [((turn_about(pivot, f) @ lens0) - target).length for f in phis]
        r_l += ph["distance"] - sum(dist) / len(dist)
        f0 = phis[ph["shown"]]
        _, body = phone_pose(turn_about(pivot, f0) @ lens0, target)
        c = body @ Vector((W / 2 - lx, ly - Hh / 2, -D / 2))
        t_l -= (c - pivot).dot(arm_dirs(f0)[1])
    lens0 = Vector((pivot.x, pivot.y, target.z + hp)) + u0 * r_l + t0 * t_l
    phis = [arm_angle_for(lens0, pivot, target, az) for az in ph["azimuths_deg"]]
    dist = [((turn_about(pivot, f) @ lens0) - target).length for f in phis]
    r_s = math.sqrt(sp["min_distance"] ** 2 - hs ** 2)
    for _ in range(8):
        glass0 = Vector((pivot.x, pivot.y, centroid.z + hs)) + u0 * r_s
        ds = [((turn_about(pivot, f) @ glass0) - centroid).length for f in phis]
        r_s += sp["min_distance"] - min(ds)
    glass0 = Vector((pivot.x, pivot.y, centroid.z + hs)) + u0 * r_s
    ds = [((turn_about(pivot, f) @ glass0) - centroid).length for f in phis]
    sweep = [arm_angle_for(lens0, pivot, target, az) for az in P["arm"]["sweep_deg"]]
    return {"lens0": lens0, "glass0": glass0, "phis": phis, "phone_mm": dist, "scanner_mm": ds, "sweep": sweep,
            "scanner_deg": [azimuth_of(turn_about(pivot, f) @ glass0, centroid) for f in phis]}


def scanner_head(coll, mats, F):
    """EinScan-SP V2 head in fixed-scan mode: black glass front with two cameras and the projector, silver
    body, black top strip, holder underneath. F: origin on the glass centre, z backwards, x along the bar."""
    W, H, D = P["scanner"]["head"]
    bl = P["scanner"]["baseline"] / 2
    head = [new_obj("scanner_body", box((-W / 2, -H / 2, 1.5), (W / 2, H / 2, D)), coll, mats["scanner"], F),
            new_obj("scanner_glass", box((-W / 2 + 1, -H / 2 + 1, 0.0), (W / 2 - 1, H / 2 - 1, 1.5)), coll, mats["glass"], F),
            new_obj("scanner_top_strip", box((-0.3 * W, H / 2, 8.0), (0.3 * W, H / 2 + 5.0, D - 8.0)), coll, mats["glass"], F),
            new_obj("scanner_holder", box((-0.25 * W, -H / 2 - 12.0, 10.0), (0.25 * W, -H / 2, D - 10.0)), coll, mats["glass"], F)]
    head += [new_obj(name, cylinder(r, -2.5, 0.0, 48), coll, mats["lens"], F @ Matrix.Translation((x, 0.0, 0.0)))
             for name, x, r in (("scanner_camera_left", -bl, 7.0), ("scanner_projector", 0.0, 10.0), ("scanner_camera_right", bl, 7.0))]
    return head


def hinge_frame(origin, axis):
    """x along the (horizontal) hinge axis, z up."""
    x = Vector(axis).normalized()
    z = Vector((0.0, 0.0, 1.0))
    m = Matrix((x, z.cross(x), z)).transposed().to_4x4()
    m.translation = Vector(origin)
    return m


def hinge_bracket(name, coll, mats, Hf, gap, cheek_t, half_w, drop, base_t=6.0):
    """Printed fork for a tilt hinge: two cheeks normal to the axis at +-gap, half width half_w (rounded about
    the axle), standing on a base plate whose underside is `drop` below the axle."""
    prof = [(-half_w, -drop + base_t - 0.5), (half_w, -drop + base_t - 0.5)] + \
           [(half_w * math.cos(t), half_w * math.sin(t)) for t in np.linspace(0.0, math.pi, 17)]
    bms = [moved(extrude_section(prof, gap, gap + cheek_t), Hf @ YZX), moved(extrude_section(prof, -gap - cheek_t, -gap), Hf @ YZX),
           moved(box((-gap - cheek_t, -half_w - 4.0, -drop), (gap + cheek_t, half_w + 4.0, -drop + base_t)), Hf)]
    return part(name, bms, coll, mats["petg_fixture"], cut=[moved(along_x(cylinder(3.1, -gap - cheek_t - 1.0, gap + cheek_t + 1.0, 24), 0.0, 0.0), Hf)])


def knob(r, h, Hf, x0, sign):
    """Star knob on the hinge axis, against the face at x0, outwards along sign * x."""
    lobes = [((r if i % 2 == 0 else r * 0.62) * math.cos(math.pi * i / 6), (r if i % 2 == 0 else r * 0.62) * math.sin(math.pi * i / 6))
             for i in range(12)]
    m = Hf @ Matrix.Translation((x0, 0.0, 0.0)) @ Matrix.Rotation(math.pi / 2 * sign, 4, "Y")
    return moved(extrude_polygon(lobes, 0.0, h), m)


def phone_cradle(name, coll, body, mat):
    """U cradle for the phone in its cover (it slides in from the top and seats on the bottom and on the end
    opposite the camera), stem and tongue of the tilt hinge, lever to tilt it by hand during the video.
    Returns the part and the hinge point (body frame)."""
    ph = P["phone"]
    W, Hh, D = ph["size"]
    lx, ly = ph["lens_offset"]
    wall, lip = 3.0, 4.0
    x0, x1, y0, y1 = -lx, W - lx, ly - Hh, ly
    c = (x0 + x1) / 2
    yh = y0 - wall - 28.0
    zc = -D / 2
    cradle = [box((x0 - wall, y0 - wall, -D - wall), (x1 + wall, y0 + 10.0, wall)),          # bottom channel
              box((x1, y0 - wall, -D - wall), (x1 + wall, y1 - 15.0, wall)),                    # end stop (screen side of the camera)
              box((x0 - wall, y0 - wall, -D - wall), (x0, y0 + 25.0, wall)),                    # short side guide at the camera end
              box((c - 16.0, y0 - wall - 14.0, -D - wall), (c + 16.0, y0 - wall + 0.5, wall)),  # stem
              # tongue of the hinge: half disc about the axle and a web up to the stem, in the (y, z) plane
              moved(extrude_polygon([(yh + 10.0 * math.cos(t), zc + 10.0 * math.sin(t)) for t in np.linspace(math.pi / 2, 1.5 * math.pi, 13)]
                                    + [(y0 - wall - 13.5, zc - 10.0), (y0 - wall - 13.5, zc + 10.0)], c - 4.0, c + 4.0), YZX),
              moved(cylinder(3.0, 0.0, 55.0, 24), frame((c + 10.0, y0 - wall - 7.0, -D - wall + 0.5), (0.0, 0.0, -1.0))),   # lever
              sphere(7.0, (c + 10.0, y0 - wall - 7.0, -D - wall - 55.0))]
    pocket = box((x0 - 0.05, y0 - 0.05, -D - 0.05), (x1 + 0.05, y1 + 30.0, 0.05))
    screen = box((x0 + lip, y0 + lip, -D - wall - 1.0), (x1 - lip, y1 + 30.0, -D + 0.01))
    back = box((x0 + lip, y0 + lip, -0.01), (x1 - lip, y1 + 30.0, wall + 1.0))
    axle = along_x(cylinder(3.1, c - 6.0, c + 6.0, 24), yh, zc)
    holder = part(name, [moved(b, body) for b in cradle], coll, mat, cut=[moved(b, body) for b in (pocket, screen, back, axle)])
    return holder, Vector((c, yh, zc))


def build_sensor_arm(colls, mats, pivot, target, centroid, z_b):
    """Sensor arm at the shown photo position; every part is a child of sensor_arm_pivot, which turns about the
    vertical through the pivot shaft. The phone and its cradle are children of phone_tilt (hinge axis x)."""
    coll = colls["sensor_arm"]
    A, ph, sp = P["arm"], P["phone"], P["scanner"]
    s, hs = P["profile"], P["profile"] / 2
    tt = P["pendulum"]["turntable"][1]
    sol = solve_arm(pivot, target, centroid)
    f0 = sol["phis"][ph["shown"]]
    u, t = arm_dirs(f0)
    rot0 = turn_about(pivot, f0)
    lens, glass = rot0 @ sol["lens0"], rot0 @ sol["glass0"]
    pv = Vector((pivot.x, pivot.y, 0.0))
    radial = lambda p: (Vector((p.x, p.y, 0.0)) - pv).dot(u)
    at = lambda r, dt=0.0, z=0.0: pv + u * r + t * dt + Vector((0.0, 0.0, z))
    arm_m = lambda r, dt, z: Matrix.Translation(at(r, dt, z)) @ Matrix((u, t, Vector((0.0, 0.0, 1.0)))).transposed().to_4x4()   # x radial, y tangential

    # scanner head aimed at the sphere centroid
    d_s = (glass - centroid).normalized()
    a_s = math.radians(azimuth_of(glass, centroid))
    F = frame(glass, d_s, (math.cos(a_s), math.sin(a_s), 0.0))
    W, H, D = sp["head"]
    head = scanner_head(coll, mats, F)
    corners = [F @ Vector((x, y, z)) for x in (-W / 2, W / 2) for y in (-H / 2 - 12.0, H / 2 + 5.0) for z in (0.0, D)]
    r_post = max(radial(p) for p in corners) + A["post_gap"] + hs

    # heights: hub and arm on the turntable, bogie and post plates on the arm
    sd, sl, sfl = A["shaft"]
    hub_w, hub_h = A["hub"]
    z_hub0 = z_b + tt + A["thrust"][1]
    z_hub1 = z_hub0 + hub_h
    z_arm0, z_arm1 = z_hub1 - s, z_hub1
    r_track = A["track_r"]
    pw, pdp, pt = A["bogie"]
    z_pl1 = z_arm1 + pt
    tw_, tk = A["track"]
    wd, ww, wb = A["wheel"]
    z_axle = z_b + tk + wd / 2

    parts = []
    # pivot shaft with its flange under the board, PTFE thrust washer, hub with two 6001 bearings, collar
    parts.append(new_obj_parts("arm_pivot_shaft", [
        (cylinder(sd / 2, z_b - P["base_t"] - sfl, z_hub1 + A["collar"][1] + 2.0, 32), mats["steel"]),
        (cylinder(30.0, z_b - P["base_t"] - sfl, z_b - P["base_t"], 48), mats["steel"])], coll, Matrix.Translation(pv)))
    parts.append(new_obj("arm_thrust_washer", tube(sd / 2 + 0.1, A["thrust"][0] / 2, z_b + tt, z_hub0, 48), coll, mats["ptfe"], Matrix.Translation(pv)))
    bo, bw_ = A["bearing"][1] / 2, A["bearing"][2]
    hub = part("arm_hub", [moved(box((-hub_w / 2, -hub_w / 2, z_hub0), (hub_w / 2, hub_w / 2, z_hub1)), arm_m(0.0, 0.0, 0.0))], coll, mats["aluminium"],
               cut=[moved(cylinder(bo + 0.05, z_hub0 - 1.0, z_hub1 + 1.0, 48), Matrix.Translation(pv))])
    parts.append(hub)
    parts.append(new_obj_parts("arm_bearings", [(tube(sd / 2 + 0.02, bo, z, z + bw_, 48), mats["steel"]) for z in (z_hub0, z_hub1 - bw_)],
                               coll, Matrix.Translation(pv)))
    parts.append(new_obj("arm_collar", tube(sd / 2 + 0.02, A["collar"][0] / 2, z_hub1, z_hub1 + A["collar"][1], 48), coll, mats["steel"], Matrix.Translation(pv)))
    # arm, bogie plate with two wheel forks, two 608 wheels on M8 axles and the index plunger, post plate at the end
    pp_w, pp_d, pp_t = A["post_plate"]
    parts.append(new_obj("arm_profile", profile_bar(at(hub_w / 2, 0.0, z_arm0 + hs), at(r_post + pp_d / 2, 0.0, z_arm0 + hs), s, t), coll, mats["profile"]))
    pl = A["plunger"]
    p_pin = at(r_track + pl["dr"], pl["dt"])
    bd_, bh_ = pl["body"]
    kr, kh = pl["knob"]
    parts.append(part("arm_bogie_plate", [moved(box((-pdp / 2, -pw / 2, z_arm1), (pdp / 2, pw / 2, z_pl1)), arm_m(r_track, 0.0, 0.0))], coll,
                      mats["aluminium"], cut=[moved(cylinder(bd_ / 2 + 0.02, z_arm1 - 1.0, z_pl1 + 1.0, 32), Matrix.Translation(p_pin))]))
    parts.append(new_obj("arm_post_plate", moved(box((-pp_d / 2, -pp_w / 2, z_arm1), (pp_d / 2, pp_w / 2, z_arm1 + pp_t)), arm_m(r_post, 0.0, 0.0)),
                         coll, mats["aluminium"]))
    for i, sgn in enumerate((-1.0, 1.0)):
        c = arm_m(r_track, sgn * A["wheel_dt"], 0.0)
        axle_m = c @ Matrix.Translation((0.0, 0.0, z_axle)) @ Matrix.Rotation(math.pi / 2, 4, "Y")       # axle along the radius
        fork_cut = [moved(box((-ww / 2 - 0.6, -wd / 2 - 8.0, z_axle - wd / 2 - 3.0), (ww / 2 + 0.6, wd / 2 + 8.0, z_axle + wd / 2 + 1.5)), c),
                    moved(cylinder(wb / 2 + 0.1, -ww / 2 - 8.0, ww / 2 + 8.0, 24), axle_m)]
        parts.append(part(f"arm_wheel_fork_{i + 1}", [moved(box((-ww / 2 - 5.0, -wd / 2 - 4.0, z_axle - 7.0), (ww / 2 + 5.0, wd / 2 + 4.0, z_arm1)), c)],
                          coll, mats["petg_fixture"], cut=fork_cut))
        parts.append(new_obj_parts(f"arm_wheel_{i + 1}", [
            (moved(tube(wb / 2, wd / 2, -ww / 2, ww / 2, 48), axle_m), mats["steel"]),
            (moved(cylinder(wb / 2, -ww / 2 - 5.5, ww / 2 + 12.0, 24), axle_m), mats["steel"]),
            (moved(hexagon(13.0, -ww / 2 - 10.8, -ww / 2 - 5.5), axle_m), mats["steel"]),
            (moved(hexagon(13.0, ww / 2 + 5.5, ww / 2 + 12.0), axle_m), mats["steel"])], coll))
    parts.append(new_obj("arm_plunger_body", moved(tube(pl["pin_d"] / 2 + 0.1, bd_ / 2, z_arm1, z_pl1 + bh_, 32), Matrix.Translation(p_pin)),
                         coll, mats["steel"]))
    pin = part("arm_plunger_pin", [moved(cylinder(pl["pin_d"] / 2, z_b + 2.0, z_pl1 + bh_ + 1.0, 24), Matrix.Translation(p_pin)),
                                   moved(cylinder(kr, z_pl1 + bh_ + 0.5, z_pl1 + bh_ + 0.5 + kh, 32), Matrix.Translation(p_pin))], coll, mats["rubber"])
    parts.append(pin)

    # scanner tilt hinge: aluminium plate screwed under the holder (1/4" screw) with a tongue down to the axle,
    # printed fork with one M6 axle and a star knob, on a boom from the post
    y_pl = -H / 2 - 12.0
    H_s = F @ Vector((0.0, y_pl - 6.0 - 14.0, D / 2))
    axis_s = (F.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
    Hs = hinge_frame(H_s, axis_s)
    tongue = [(10.0 * math.cos(a), 10.0 * math.sin(a)) for a in np.linspace(math.pi, 2 * math.pi, 13)] + [(10.0, 16.0), (-10.0, 16.0)]
    parts.append(part("scanner_tilt_plate", [moved(box((-0.25 * W, y_pl - 6.0, 10.0), (0.25 * W, y_pl, D - 10.0)), F),
                                             moved(extrude_polygon(tongue, -8.0, 8.0), Hs @ YZX)], coll, mats["aluminium"],
                      cut=[moved(along_x(cylinder(3.1, -9.0, 9.0, 24), 0.0, 0.0), Hs)]))
    drop_s = 24.0
    parts.append(hinge_bracket("scanner_tilt_fork", coll, mats, Hs, 8.5, 5.0, 11.0, drop_s, 8.0))
    parts.append(new_obj_parts("scanner_tilt_knob", [(knob(14.0, 10.0, Hs, 13.5, 1.0), mats["rubber"]),
                                                     (moved(along_x(cylinder(3.0, -15.5, 14.0, 16), 0.0, 0.0), Hs), mats["steel"]),
                                                     (moved(along_x(nut(10.0, 5.0, -18.5, 6.0), 0.0, 0.0), Hs), mats["steel"])], coll))
    z_sb = H_s.z - drop_s - hs
    # phone on its cradle and tilt hinge
    cam_m, body = phone_pose(lens, target)
    W_, Hh, Dp = ph["size"]
    lx, ly = ph["lens_offset"]
    cradle, hinge_local = phone_cradle("phone_cradle", coll, body, mats["petg_fixture"])
    H_p = body @ hinge_local
    axis_p = (body.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
    Hp = hinge_frame(H_p, axis_p)
    drop_p = 26.0
    parts.append(hinge_bracket("phone_tilt_fork", coll, mats, Hp, 4.5, 4.0, 11.0, drop_p, 6.0))
    parts.append(new_obj_parts("phone_tilt_knob", [(knob(13.0, 10.0, Hp, 8.5, 1.0), mats["rubber"]),
                                                   (moved(along_x(cylinder(3.0, -10.5, 9.0, 16), 0.0, 0.0), Hp), mats["steel"]),
                                                   (moved(along_x(nut(10.0, 5.0, -13.5, 6.0), 0.0, 0.0), Hp), mats["steel"])], coll))
    z_pb = H_p.z - drop_p - hs
    phone = new_obj("smartphone", box((-lx, ly - Hh, -Dp), (W_ - lx, ly, 0.0)), coll, mats["phone"], body)
    module = new_obj("smartphone_camera_module", box((-12.0, -26.0, 0.0), (12.0, 10.0, 0.01)), coll, mats["lens"], body)
    lens_ob = new_obj("smartphone_lens", cylinder(5.5, 0.01, 0.6, 32), coll, mats["lens"], body)

    # post on the post plate, booms to the two hinges, corner brackets
    post_top = z_pb + hs + 12.0
    p_post = at(r_post)
    z_pp = z_arm1 + pp_t
    parts.append(new_obj("arm_post", profile_bar((p_post.x, p_post.y, z_pp), (p_post.x, p_post.y, post_top), s, t), coll, mats["profile"]))
    parts += post_brackets("arm_post", coll, mats["aluminium"], p_post.x, p_post.y, z_pp, (u, t, -t))
    for name, z, h in (("arm_boom_scanner", z_sb, H_s), ("arm_boom_phone", z_pb, H_p)):
        r_end = radial(h) - 12.0
        parts.append(new_obj(name, profile_bar(at(r_post - hs, 0.0, z), at(r_end, 0.0, z), s, t), coll, mats["profile"]))
        parts.append(corner_bracket(f"{name}_bracket", coll, mats["aluminium"], at(r_post - hs, 0.0, z + hs), -u, (0.0, 0.0, 1.0)))

    # tilt empty for the phone: cradle, phone, lens, knob turn about the hinge axis
    tilt = empty("phone_tilt", coll, 20.0)
    tilt.matrix_world = Hp
    bpy.context.view_layer.update()
    for ob in (cradle, phone, module, lens_ob):
        center_origin(ob)
        attach(ob, tilt)
    cams = []
    for k, f in enumerate(sol["phis"]):
        turn = turn_about(pivot, f - f0)
        for name, m, lensmm in ((f"cam_phone_{k + 1}", cam_m, ph["focal_35mm"]),
                                (f"cam_scanner_{k + 1}", F, 18.0 * sp["min_distance"] / (sp["field"][0] / 2))):
            cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
            cam.data.sensor_width, cam.data.sensor_fit, cam.data.lens = 36.0, "HORIZONTAL", lensmm
            cam.data.clip_start, cam.data.clip_end = 5.0, 5000.0
            cam.matrix_world = turn @ m
            colls["render_setup"].objects.link(cam)
            cams.append(cam)
    for ob in parts + head:
        if ob.type == "MESH":
            center_origin(ob)
    pivot_ob = empty("sensor_arm_pivot", coll, 30.0)
    pivot_ob.location = (pivot.x, pivot.y, z_b)
    bpy.context.view_layer.update()
    for ob in parts + head + [tilt]:
        attach(ob, pivot_ob)
    track = build_track(coll, mats, pv, r_track, sol, f0, p_pin, z_b)
    bpy.context.scene["sensor_arm"] = {
        "pivot": list(pv), "post_r": r_post, "track_r": r_track, "shown_deg": f0, "arm_deg": sol["phis"], "camera_deg": list(ph["azimuths_deg"]),
        "phone_mm": sol["phone_mm"], "scanner_mm": sol["scanner_mm"], "scanner_deg": sol["scanner_deg"], "sweep_deg": sol["sweep"]}
    return parts + head + [cradle, phone, module, lens_ob] + track


def build_track(coll, mats, pv, r_track, sol, f0, p_pin, z_b):
    """Printed PETG track under the wheels, in segments screwed to the board: one index hole per photo position,
    a tick every 5 deg of arm angle on the outer edge and the camera azimuth next to each hole."""
    A = P["arm"]
    tw_, tk = A["track"]
    wheel_deg = math.degrees((A["wheel_dt"] + A["wheel"][0] / 2 + 6.0) / r_track)
    lo, hi = sol["sweep"][0] - wheel_deg, sol["sweep"][1] + wheel_deg
    r0, r1 = r_track - tw_ / 2, r_track + tw_ / 2
    cut = []
    for phi in np.arange(math.ceil(lo / A["tick_deg"]) * A["tick_deg"], hi, A["tick_deg"]):
        u, t = arm_dirs(phi)
        m = Matrix.Translation(pv + u * r1) @ Matrix((u, t, Vector((0.0, 0.0, 1.0)))).transposed().to_4x4()
        cut.append((phi, moved(box((-7.0, -0.4, z_b + tk - 0.6), (1.0, 0.4, z_b + tk + 1.0)), m)))
    holes = []
    for f in sol["phis"]:
        q = turn_about(pv, f - f0) @ p_pin
        holes.append(q)
        cut.append((azimuth_of(q, pv), moved(cylinder(A["plunger"]["pin_d"] / 2 + 0.1, z_b - 1.0, z_b + tk + 1.0, 24), Matrix.Translation((q.x, q.y, 0.0)))))
    n = A["track_segments"]
    edges = np.linspace(lo, hi, n + 1)
    parts = []
    for i in range(n):
        a0, a1 = edges[i] + (0.01 if i else 0.0), edges[i + 1] - (0.01 if i < n - 1 else 0.0)
        seg = moved(extrude_polygon(arc_polygon(r0, r1, a0, a1, 48), z_b, z_b + tk), Matrix.Translation(pv))
        mine = [b for phi, b in cut if a0 + 0.3 < phi < a1 - 0.3]
        screws = [moved(cylinder(2.8, z_b + tk - 2.5, z_b + tk + 1.0, 24), Matrix.Translation(pv + arm_dirs(a)[0] * r_track))
                  for a in (a0 + 3.0, a1 - 3.0)]
        parts.append(part(f"arm_track_{i + 1}", [seg], coll, mats["petg_fixture"], cut=mine + screws))
    m = bpy.data.materials.new("engraving")
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    m.diffuse_color = (0.02, 0.02, 0.02, 1.0)
    for az, q in zip(P["phone"]["azimuths_deg"], holes):
        phi = azimuth_of(q, pv)
        u, t = arm_dirs(phi)
        cu = bpy.data.curves.new(f"arm_track_label_{az:.0f}", "FONT")
        cu.body = f"{az:.0f}"
        cu.size = 8.0
        cu.align_x = "CENTER"
        cu.materials.append(m)
        ob = bpy.data.objects.new(cu.name, cu)
        ob.location = pv + u * (r_track + 11.0) + Vector((0.0, 0.0, z_b + tk + 0.02))
        ob.rotation_euler = (0.0, 0.0, math.radians(phi))
        coll.objects.link(ob)
    for ob in parts:
        center_origin(ob)
    return parts


def build_base(coll, mats, points, z_b, holes):
    """Birch plywood board on rubber feet, cut to the footprint plus the margin; holes: list of (x, y, radius)."""
    pts = np.array(points)
    mg = P["base_margin"]
    x0, y0 = np.floor((pts.min(0) - mg) / 10.0) * 10.0
    x1, y1 = np.ceil((pts.max(0) + mg) / 10.0) * 10.0
    t = P["base_t"]
    fd, fh = P["feet"]
    plate = part("base_board", [box((x0, y0, z_b - t), (x1, y1, z_b))], coll, mats["wood"],
                 cut=[moved(cylinder(r, z_b - t - 1.0, z_b + 1.0, 32), Matrix.Translation((x, y, 0.0))) for x, y, r in holes])
    feet = [new_obj(f"base_foot_{i + 1}", cylinder(fd / 2, z_b - t - fh, z_b - t, 32), coll, mats["rubber"],
                    Matrix.Translation((x, y, 0.0)))
            for i, (x, y) in enumerate(((x0 + 40, y0 + 40), (x1 - 40, y0 + 40), (x1 - 40, y1 - 40), (x0 + 40, y1 - 40)))]
    z_t = z_b - t - fh
    table = new_obj("lab_table", box((x0 - 300, y0 - 250, z_t - 30.0), (x1 + 300, y1 + 250, z_t)), coll, mats["table"])
    for ob in [plate, table] + feet:
        center_origin(ob)
    return [plate, table] + feet, (x1 - x0, y1 - y0)


# ---------------------------------------------------------------- build, render, save

def footprint(coll=None, reach=14.0):
    """xy corners of the world bounding boxes of the meshes that stand on the board (lowest point within reach
    of its top), all or of one collection: arm, post, scanner and phone may overhang the edge."""
    bpy.context.view_layer.update()
    obs = coll.all_objects if coll else bpy.data.objects
    out = []
    for ob in obs:
        if ob.type == "MESH" and not ob.name.startswith(("lab_table", "base_")):
            co = world_coords(ob)
            if co[:, 2].min() < P["base_z"] + reach:
                out += [tuple(co[:, :2].min(0)), tuple(co[:, :2].max(0))]
    return out


def set_arm(phi=None):
    """Turn the sensor arm to the arm angle phi (None: the shown photo position)."""
    sa = bpy.context.scene["sensor_arm"]
    ob = bpy.data.objects["sensor_arm_pivot"]
    ob.rotation_euler.z = 0.0 if phi is None else math.radians(phi - sa["shown_deg"])
    bpy.context.view_layer.update()


def build():
    reset_scene()
    setup_scene()
    colls = {name: collection(name) for name in (
        "phantom", "silicone", "imf", "splint", "marker", "interfaces", "ground_truth", "mount", "pendulum", "sensor_arm",
        "base", "labels", "render_setup")}
    for key, (cname, _) in VARIANTS.items():
        colls[cname] = collection(cname, colls["interfaces"])
    tmp = collection("tmp")
    mats = {name: material(name) for name in COLORS}
    z_b = P["base_z"]

    path = stage_path()
    cached = path.exists()
    if cached:
        st, info = load_stage(path, colls, mats)
    else:
        st, info = build_stage(colls, mats, tmp)
        save_stage(path, st, info)
    pad_origin = Vector(info["pad_origin"])
    pad = Matrix.Translation(pad_origin)
    marker_parts, strike, centres = build_marker(colls["marker"], mats, pad)
    variants = build_interfaces(colls, mats, pad)
    iface = [o for v in variants.values() for o in v]
    for ob in marker_parts + iface:
        center_origin(ob)
    m_spheres = marker_spheres(colls["ground_truth"], mats, centres)
    spheres = m_spheres + st["spheres"]
    mount, _ = build_mount(colls, mats, st["cranium"], z_b)
    for ob in marker_parts + iface + m_spheres:
        attach(ob, st["splint"])
    attach(st["splint"], st["mandible"])
    for ob in st["posts"] + st["spheres"]:
        attach(ob, st["mandible"])
    attach(st["mandible"], st["cranium"])
    for ob in st["imf"] + [st["silicone"]]:
        attach(ob, st["cranium"])
    attach(st["cranium"], mount[0])

    pend_base, swing, pend = build_pendulum(colls["pendulum"], mats, strike, z_b)
    bushings = build_index_bushings(colls["pendulum"], mats, strike, z_b)
    target = (pad_origin + Vector(P["marker_center"]) + Vector(info["window_c"])) / 2
    centroid = sum((ob.matrix_world.translation for ob in spheres), Vector()) / len(spheres)
    sc = bpy.context.scene
    sc["strike_point"], sc["target"], sc["centroid"] = list(strike), list(target), list(centroid)
    build_sensor_arm(colls, mats, strike, target, centroid, z_b)

    # board: everything that stands on it, the pendulum in its three directions, the arm at both ends of the sweep
    foot = footprint()
    for d in (45.0, 90.0):
        set_direction(d)
        foot += footprint(colls["pendulum"])
    set_direction(P["pendulum"]["direction_deg"])
    for f in sc["sensor_arm"]["sweep_deg"]:
        set_arm(f)
        foot += footprint(colls["sensor_arm"])
    set_arm()
    holes = [(strike.x, strike.y, P["arm"]["shaft"][0] / 2 + 0.2)]
    holes += [(b.matrix_world.translation.x, b.matrix_world.translation.y, 7.0) for b in bushings]
    base, base_size = build_base(colls["base"], mats, foot, z_b, holes)

    for ob in list(tmp.objects):
        delete(ob)
    bpy.data.collections.remove(tmp)
    setup_cameras(colls["render_setup"], pad_origin, strike, target, z_b)

    def xy(name, dx=0.0, dy=0.0):
        t = bpy.data.objects[name].matrix_world.translation
        return (t.x + dx, t.y + dy)
    add_labels(colls["labels"], [
        ("label_phantom", "cranio: 2 profili, 2 spine, 1 vite", xy("phantom_plate", -250.0, 40.0)),
        ("label_pendulum", "pendolo, qui a 0 gradi", xy("pendulum_controller", -170.0, 0.0)),
        ("label_turntable", "piatto: spina a 0, 45, 90 gradi", xy("pendulum_index_bushing_045", -150.0, -40.0)),
        ("label_marker", "marker e portamarker (6 varianti)", (strike.x - 190.0, strike.y + 40.0)),
        ("label_arm", "braccio: telefono sopra, scanner sotto", xy("arm_track_3", -30.0, 150.0)),
        ("label_track", "pista e 2 rotelle: 40, 55, 70, 85 gradi", xy("arm_track_1", -120.0, -150.0)),
        ("label_pivot", "perno comune a piatto e braccio", (strike.x - 170.0, strike.y - 70.0)),
    ], 900.0)
    layer_collection("labels").hide_viewport = True
    bpy.data.collections["labels"].hide_render = True
    show()

    sa = sc["sensor_arm"]
    bench = bpy.data.texts.new("bench_info")
    bench.write(
        "Generated by 04_code/tools/build_test_bench.py; edit the script, not the scene.\n"
        "Phantom: BodyParts3D 4.3 skull and mandible FJ6399, The Database Center for Life Science (DBCLS),\n"
        "CC BY-SA 2.1 JP, via Wikimedia Commons. Placeholder until T3.1.\n"
        f"Mandible opened by {info['opening_deg']:.2f} deg about the condylar axis (splint and wafer).\n"
        f"Pad frame origin (world, mm): {tuple(round(x, 2) for x in pad_origin)}.\n"
        f"Strike point (world, mm): {tuple(round(x, 2) for x in strike)}; pendulum_base rotates about z: 0, -45, -90 deg.\n"
        f"Interfaces: {', '.join(v[1] for v in VARIANTS.values())}; show(variant=...) picks one.\n"
        f"Sensor arm angles (deg) of the photo positions: {[round(x, 1) for x in sa['arm_deg']]}; set_arm(angle) turns it.\n"
    )
    return {
        "stage_cached": cached,
        "opening_deg": info["opening_deg"],
        "pad_origin": [round(x, 1) for x in pad_origin],
        "strike": [round(x, 1) for x in strike],
        "target": [round(x, 1) for x in target],
        "centroid": [round(x, 1) for x in centroid],
        "base_size": [round(x) for x in base_size],
        "arm_deg": [round(x, 1) for x in sa["arm_deg"]],
        "sweep_deg": [round(x, 1) for x in sa["sweep_deg"]],
        "phone_mm": [round(x) for x in sa["phone_mm"]],
        "scanner_mm": [round(x) for x in sa["scanner_mm"]],
        "scanner_deg": [round(x, 1) for x in sa["scanner_deg"]],
        "post_r": round(sa["post_r"], 1),
        "splint_cranium_overlaps": info["splint_cranium_overlaps"],
        "tmj_drop": info["tmj_drop"],
        "condyle_overlaps": info["condyle_overlaps"],
        "objects": len([o for o in bpy.data.objects if o.type == "MESH"]),
    }


def add_labels(coll, items, z):
    """Black text labels lying flat at height z, for the annotated top view."""
    m = bpy.data.materials.new("label")
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    m.diffuse_color = (0.0, 0.0, 0.0, 1.0)
    for name, text, (x, y) in items:
        cu = bpy.data.curves.new(name, "FONT")
        cu.body = text
        cu.size = 17.0
        cu.align_x = "CENTER"
        cu.materials.append(m)
        ob = bpy.data.objects.new(name, cu)
        ob.location = (x, y, z)
        coll.objects.link(ob)


def explode(objs, offsets):
    """Move parts by world offsets for an exploded render; returns a function that puts them back."""
    saved = {n: objs[n].matrix_world.copy() for n in offsets}
    for n, off in offsets.items():
        objs[n].matrix_world = Matrix.Translation(Vector(off)) @ objs[n].matrix_world

    def undo():
        for n, m in saved.items():
            objs[n].matrix_world = m
    return undo


def show(variant="a1", silicone=True, labels=False):
    """One marker interface on the pad (they share it), silicone on or off, labels for the top view."""
    for key, (cname, _) in VARIANTS.items():
        layer_collection(cname).hide_viewport = key != variant
        bpy.data.collections[cname].hide_render = key != variant
    for name, visible in (("silicone", silicone), ("labels", labels)):
        if name not in bpy.data.collections:                  # labels: taken out of the scene on 13/10
            continue
        layer_collection(name).hide_viewport = not visible
        bpy.data.collections[name].hide_render = not visible


def look(cam, target):
    cam.rotation_euler = (Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()


def setup_cameras(coll, pad_origin, strike, target, z_b):
    objs = bpy.data.objects
    o = pad_origin
    pb = objs["pendulum_base"].matrix_world
    y_p, z_p, _ = pendulum_geometry(strike, z_b)
    board = objs["base_board"]
    bc = board.matrix_world.translation
    bw, bdp = board.dimensions.x, board.dimensions.y
    sa = bpy.context.scene["sensor_arm"]
    u, t = arm_dirs(sa["shown_deg"])
    w = lambda name: objs[name].matrix_world.translation.copy()
    post, plate = w("arm_post"), w("arm_bogie_plate")
    hp, hs = w("phone_tilt_fork"), w("scanner_tilt_fork")
    foot = w("phantom_post_1")
    foot.z = z_b
    th = math.radians(P["pendulum"]["release_deg"])
    hook = pb @ Vector((15.0, y_p - P["pendulum"]["latch_r"] * math.sin(th), z_p - P["pendulum"]["latch_r"] * math.cos(th)))
    up = Vector((0.0, 0.0, 1.0))
    centre = Vector((bc.x, bc.y, -60.0))
    specs = {
        "cam_overview": ("PERSP", 38.0, centre + Vector((-1050.0, -1250.0, 900.0)), centre),
        "cam_overview_left": ("PERSP", 38.0, centre + Vector((1250.0, 650.0, 850.0)), centre + Vector((40.0, 0.0, 0.0))),
        "cam_top": ("ORTHO", max(bw, bdp * 4 / 3) * 1.3, Vector((bc.x + 40.0, bc.y - 40.0, 1500.0)), Vector((bc.x + 40.0, bc.y - 40.0, 0.0))),
        "cam_right": ("ORTHO", max(bdp, 1000.0) * 1.05, Vector((-1600.0, bc.y, 80.0)), Vector((0.0, bc.y, 80.0))),
        "cam_front": ("ORTHO", max(bw, 1000.0) * 1.05, Vector((bc.x, -1600.0, 80.0)), Vector((bc.x, 0.0, 80.0))),
        "cam_face": ("PERSP", 45.0, target + azimuth_dir(40.0, 12.0) * 190.0, target),
        "cam_pivot": ("PERSP", 40.0, pb @ Vector((330.0, y_p - 190.0, z_p + 30.0)), pb @ Vector((20.0, y_p - 70.0, z_p - 50.0))),
        "cam_servo": ("PERSP", 40.0, hook + pb.to_3x3() @ Vector((-150.0, -120.0, -30.0)), hook + pb.to_3x3() @ Vector((5.0, 0.0, 5.0))),
        "cam_tip": ("PERSP", 50.0, strike + Vector((190.0, -80.0, 25.0)), strike + Vector((0.0, -25.0, -5.0))),
        "cam_arm": ("PERSP", 38.0, post + u * 520.0 - t * 420.0 + up * 380.0, post - u * 170.0 + up * 60.0),
        "cam_carriage": ("PERSP", 40.0, plate + u * 200.0 - t * 120.0 + up * 25.0, plate - up * 18.0),
        "cam_phone_holder": ("PERSP", 40.0, hp + u * 230.0 - t * 200.0 + up * 120.0, hp + up * 25.0),
        "cam_scanner_mount": ("PERSP", 40.0, hs - t * 260.0 + u * 90.0 - up * 25.0, hs + up * 8.0),
        "cam_profile": ("PERSP", 45.0, foot + Vector((-110.0, -120.0, 95.0)), foot + Vector((0.0, 0.0, 18.0))),
        "cam_mount": ("PERSP", 40.0, Vector((-360.0, 470.0, -20.0)), Vector((0.0, 134.0, -140.0))),
        "cam_interface": ("PERSP", 45.0, o + Vector((-125.0, -110.0, 70.0)), o + Vector((0.0, -6.0, 14.0))),
        "cam_exploded": ("PERSP", 50.0, o + Vector((260.0, -230.0, 170.0)), o + Vector((0.0, -18.0, 38.0))),
        "cam_section": ("ORTHO", 75.0, o + Vector((300.0, -14.0, 18.0)), o + Vector((0.0, -14.0, 18.0))),
    }
    for name, (kind, lens, loc, aim) in specs.items():
        cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        coll.objects.link(cam)
        cam.data.type = kind
        if kind == "ORTHO":
            cam.data.ortho_scale = lens
        else:
            cam.data.lens = lens
        cam.data.clip_start, cam.data.clip_end = 1.0, 10000.0
        cam.location = Vector(loc)
        if name == "cam_top":
            cam.rotation_euler = (0.0, 0.0, 0.0)
        else:
            look(cam, aim)
    for name, rot, energy in (("sun_key", (math.radians(50), 0.0, math.radians(-35)), 2.2),
                              ("sun_fill", (math.radians(65), 0.0, math.radians(140)), 0.5)):
        light = bpy.data.lights.new(name, "SUN")
        light.energy = energy
        light.angle = math.radians(8)
        ob = bpy.data.objects.new(name, light)
        coll.objects.link(ob)
        ob.rotation_euler = rot
    bpy.context.scene.camera = bpy.data.objects["cam_overview"]


def set_direction(deg):
    base = bpy.data.objects["pendulum_base"]
    t = base.matrix_world.translation.copy()
    base.matrix_world = Matrix.Translation(t) @ Matrix.Rotation(math.radians(-deg), 4, "Z")
    bpy.context.view_layer.update()


def section(objs, cutter_world):
    """Temporary boolean that removes the part of objs inside a world-space box (for section renders)."""
    tmp = collection("tmp_section")
    cut = new_obj("tmp_section_box", cutter_world, tmp)
    tmp.hide_render = True
    layer_collection("tmp_section").hide_viewport = True
    mods = []
    for ob in objs:
        if ob.type == "MESH":
            m = ob.modifiers.new("section", "BOOLEAN")
            m.operation = "DIFFERENCE"
            m.object = cut
            m.solver = "EXACT"
            mods.append((ob, m))

    def undo():
        for ob, m in mods:
            ob.modifiers.remove(m)
        delete(cut)
        bpy.data.collections.remove(tmp)
    return undo


MARKER_PARTS = ["marker_holder", "marker_plate", "marker_charuco", "marker_strike_cap",
                "gt_sphere_marker_1", "gt_sphere_marker_2", "gt_sphere_marker_3"]


def exploded_offsets(key):
    """World offsets of the exploded view of one interface, as it is assembled."""
    up = lambda dz: (0.0, 0.0, dz)
    off = {}
    if key == "a1":
        off = {"a1_cap": up(22), "a1_ball_stud": up(40), "a1_arm": up(62)}
        for i in (1, 2, 3):
            off.update({f"a1_screw_{i}": up(95), f"a1_discs_{i}": up(32), f"a1_nut_{i}": up(-25)})
        lift = 62
    elif key == "a2":
        off = {"a2_arm": up(40), "a2_shoulder_screw": (45, 0, 0), "a2_discs": (30, 0, 0), "a2_nut": (-30, 0, 0),
               "a2_washer_1": (14, 0, 0), "a2_washer_2": (-14, 0, 0)}
        lift = 40
    elif key in ("b1", "b4"):
        off = {f"{key}_plate": up(40), f"{key}_balls": up(40)}
        if key == "b1":
            off.update({"b1_spring": up(62), "b1_washer": up(78), "b1_screw": up(98), "b1_insert": up(-18)})
        lift = 40
    elif key == "b2":
        off = {"b2_kb25_balls": up(16), "b2_kb25_top": up(32), "b2_screw_top": up(44), "b2_arm": up(62), "b2_nut": up(-25)}
        lift = 62
    else:
        off = {"b3_arm": up(45), "b3_rod": up(22), "b3_set_screws": (25, 0, 0)}
        lift = 45
    off.update({n: up(lift) for n in MARKER_PARTS})
    off.update({f"{key}_pad_screw_{i}": up(-25) for i in (1, 2, 3) if f"{key}_pad_screw_{i}" in bpy.data.objects})
    return off


def render_checks(outdir=None, only=None):
    """Control renders in 02_cad/renders (prefix check_), overwritten at each run; only: names to render."""
    outdir = Path(outdir) if outdir else RENDERS
    sc = bpy.context.scene
    objs = bpy.data.objects
    sa = sc["sensor_arm"]
    phone_body = {o.name for o in objs if o.name.startswith(("smartphone", "phone_cradle"))}
    scanner_body = {o.name for o in objs if o.name.startswith("scanner_")}
    #        name, camera, show flags, hidden objects, pendulum direction, arm angle (None: shown position)
    shots = [
        ("check_00_layout", "cam_top", {"labels": True}, set(), 0.0, None),
        ("check_01_bench", "cam_overview", {}, set(), 0.0, None),
        ("check_01b_bench_left", "cam_overview_left", {}, set(), 0.0, None),
        ("check_02_top_0deg", "cam_top", {}, set(), 0.0, None),
        ("check_03_top_45deg", "cam_top", {}, set(), 45.0, None),
        ("check_04_top_90deg", "cam_top", {}, set(), 90.0, None),
        ("check_05_right", "cam_right", {}, set(), 0.0, None),
        ("check_06_front", "cam_front", {}, set(), 0.0, None),
        ("check_07_face", "cam_face", {}, set(), 0.0, None),
        ("check_08_face_no_silicone", "cam_face", {"silicone": False}, set(), 0.0, None),
        ("check_09_pendulum_head", "cam_pivot", {}, set(), 0.0, None),
        ("check_10_servo_latch", "cam_servo", {}, set(), 0.0, None),
        ("check_11_tip_contact", "cam_tip", {}, set(), 0.0, None),
        ("check_12_sensor_arm", "cam_arm", {}, set(), 0.0, None),
        ("check_13_arm_carriage", "cam_carriage", {}, set(), 0.0, None),
        ("check_14_phone_hinge", "cam_phone_holder", {}, set(), 0.0, None),
        ("check_15_scanner_hinge", "cam_scanner_mount", {}, set(), 0.0, None),
        ("check_16_profile_joint", "cam_profile", {}, set(), 0.0, None),
        ("check_17_phantom_mount", "cam_mount", {"silicone": False}, set(), 0.0, None),
    ]
    shots += [(f"check_{20 + k}_phone_view_{k + 1}", f"cam_phone_{k + 1}", {}, phone_body, 0.0, f) for k, f in enumerate(sa["arm_deg"])]
    shots += [(f"check_{24 + k}_scanner_view_{k + 1}", f"cam_scanner_{k + 1}", {}, scanner_body, 0.0, f) for k, f in enumerate(sa["arm_deg"])]
    shots += [(f"check_{30 + i}_{key}", "cam_interface", {"variant": key}, set(), 0.0, None) for i, key in enumerate(VARIANTS)]
    written = []
    swing = objs["pendulum_swing"]
    release = swing.rotation_euler.x
    for name, cam, flags, hidden, direction, phi in shots:
        if only and name not in only:
            continue
        show(**flags)
        set_direction(direction)
        set_arm(phi)
        swing.rotation_euler.x = 0.0 if name.endswith("tip_contact") else release   # hanging: tip on the cap
        for ob in objs:
            ob.hide_render = ob.name in hidden
        sc.camera = objs[cam]
        sc.render.filepath = str(outdir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        written.append(sc.render.filepath)
    swing.rotation_euler.x = release
    set_arm()

    # exploded views of the interfaces, as they are assembled
    for i, key in enumerate(VARIANTS):
        name = f"check_{36 + i}_exploded_{key}"
        if only and name not in only:
            continue
        show(variant=key)
        set_direction(0.0)
        for ob in objs:
            ob.hide_render = ob.type == "MESH" and (ob.name.startswith(("pendulum", "arm_", "phone", "smartphone", "scanner", "silicone"))
                                                    or ob.name.endswith("_tether"))
        undo = explode(objs, exploded_offsets(key))
        sc.camera = objs["cam_exploded"]
        sc.render.filepath = str(outdir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        written.append(sc.render.filepath)
        undo()

    # sections on the sagittal plane of the pad, seen from the patient's left: A1 and B1
    pad = objs["splint"].matrix_world.translation.copy()
    for name, key in (("check_42_section_a1", "a1"), ("check_43_section_b1", "b1")):
        if only and name not in only:
            continue
        show(variant=key)
        names = {o.name for o in bpy.data.collections[VARIANTS[key][0]].objects}
        for ob in objs:
            ob.hide_render = ob.type == "MESH" and ob.name not in names
        undo = section([objs[n] for n in names], box(pad + Vector((0.0, -90.0, -60.0)), pad + Vector((90.0, 90.0, 200.0))))
        sc.camera = objs["cam_section"]
        sc.render.filepath = str(outdir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        written.append(sc.render.filepath)
        undo()
    for ob in objs:
        ob.hide_render = False
    show()
    set_direction(P["pendulum"]["direction_deg"])
    sc.camera = objs["cam_overview"]
    return written


def contact_sheet(files, out, cols=3, scale=2):
    """Tile renders into one PNG (every scale-th pixel), in reading order."""
    tiles = []
    for f in files:
        img = bpy.data.images.load(f)
        w, h = img.size
        px = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        px = px.reshape(h, w, 4)[::-1][::scale, ::scale]
        tiles.append(px)
        bpy.data.images.remove(img)
    th, tw = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.ones((rows * th, cols * tw, 4))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t[:th, :tw]
    img = bpy.data.images.new("tmp_sheet", cols * tw, rows * th)
    img.pixels.foreach_set(sheet[::-1].astype(np.float32).ravel())
    img.filepath_raw = str(out)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return str(out)


def intersection_volume(a, b):
    """Volume (mm^3) shared by two mesh objects, from an exact boolean on a temporary copy."""
    tmp = bpy.data.objects.new("tmp_iv", a.data.copy())
    bpy.context.scene.collection.objects.link(tmp)
    tmp.matrix_world = a.matrix_world.copy()
    mod = tmp.modifiers.new("iv", "BOOLEAN")
    mod.operation = "INTERSECT"
    mod.object = b
    mod.solver = "EXACT"
    me = evaluated_mesh(tmp)
    bm = bmesh.new()
    bm.from_mesh(me)
    vol = abs(bm.calc_volume(signed=True))
    bm.free()
    bpy.data.meshes.remove(me)
    delete(tmp)
    return vol


def interference(variant="a1", only=None, max_faces=60000):
    """Pairs of visible parts whose surfaces cross, with the shared volume for parts that are not too heavy.
    The interfaces share the pad, so each variant is checked on its own; only: names, at least one per pair."""
    show(variant=variant)
    bpy.context.view_layer.update()
    obs = [o for o in bpy.data.objects if o.type == "MESH" and o.visible_get() and o.name != "lab_table"]
    boxes = {o.name: (world_coords(o).min(0), world_coords(o).max(0)) for o in obs}
    trees = {}

    def tree(o):
        if o.name not in trees:
            trees[o.name] = bvh_world(o)
        return trees[o.name]

    out = []
    for i, a in enumerate(obs):
        for c in obs[i + 1:]:
            if only and a.name not in only and c.name not in only:
                continue
            (a0, a1), (c0, c1) = boxes[a.name], boxes[c.name]
            if np.any(a1 < c0 - 0.01) or np.any(c1 < a0 - 0.01):
                continue
            n = len(tree(a).overlap(tree(c)))
            if not n:
                continue
            vol = None
            if len(a.data.polygons) < max_faces and len(c.data.polygons) < max_faces:
                vol = round(intersection_volume(a, c), 3)
            out.append((a.name, c.name, n, vol))
    show()
    return out


def arm_clearance(step=5.0):
    """Sensor arm against the pendulum (three directions, hanging and at the largest release angle), the
    phantom and the mount, over the whole sweep: list of (direction, release, arm angle, part, part, faces)."""
    sa = bpy.context.scene["sensor_arm"]
    lo, hi = sa["sweep_deg"]
    angles = sorted(set([round(x, 2) for x in sa["arm_deg"]] + list(np.arange(math.floor(lo), hi + 1e-6, step)) + [lo, hi]))
    arm = [o for o in bpy.data.collections["sensor_arm"].all_objects if o.type == "MESH" and o.name != "arm_thrust_washer"]  # it rests on the turntable
    others = [o for c in ("pendulum", "phantom", "silicone", "mount", "marker", "ground_truth") for o in bpy.data.collections[c].all_objects
              if o.type == "MESH"] + [o for o in bpy.data.collections[VARIANTS["a1"][0]].objects if o.type == "MESH"]
    swing = bpy.data.objects["pendulum_swing"]
    release = swing.rotation_euler.x
    out = []
    for d in (0.0, 45.0, 90.0):
        set_direction(d)
        for th in (0.0, P["pendulum"]["max_release_deg"]):
            swing.rotation_euler.x = math.radians(-th)
            bpy.context.view_layer.update()
            fixed = {o.name: (bvh_world(o), world_coords(o).min(0), world_coords(o).max(0)) for o in others}
            for phi in angles:
                set_arm(phi)
                for a in arm:
                    co = world_coords(a)
                    a0, a1 = co.min(0), co.max(0)
                    ta = None
                    for name, (tb, b0, b1) in fixed.items():
                        if np.any(a1 < b0) or np.any(b1 < a0):
                            continue
                        ta = ta or bvh(co, a.data)
                        n = len(ta.overlap(tb))
                        if n:
                            out.append((d, th, round(phi, 1), a.name, name, n))
    swing.rotation_euler.x = release
    set_direction(P["pendulum"]["direction_deg"])
    set_arm()
    return out


TRIAL_STEPS = [  # caption title and text for each step of the trial animation
    ("Armamento", "massa e angolo impostati, pendolo sollevato e agganciato al gancio del servo"),
    ("Sgancio", "il servo apre il gancio e libera il pendolo (movimento rallentato 4 volte)"),
    ("Urto", "il giunto A1 slitta sulla sfera, il marker resta spostato; il pendolo si riaggancia in alto"),
    ("Scansioni", "il braccio porta lo scanner in due posizioni, le scansioni si uniscono sulle 6 sfere"),
    ("Foto dal braccio", "4 posizioni sulla pista, bloccate dal pistoncino: 40, 55, 70 e 85 gradi"),
    ("Video dal braccio", "pistoncino sollevato: il braccio scorre su due rotelle e il telefono si inclina con la leva"),
    ("Ricollocazione del marker", "ogni 5 prove o oltre 10 mm, poi nuova scansione di riferimento"),
    ("Interfaccia B1, stesso urto", "il marker si inclina sulle sfere e la molla lo riporta da solo in sede"),
]
SCAN_POSITIONS = (0, 2)   # photo positions also used for the two scans of the animation


def quat_key(ob, f, loc, q, prev):
    """Key location and quaternion rotation, keeping the quaternion on the side of the previous key."""
    if prev is not None and prev.dot(q) < 0.0:
        q = -q
    ob.location = loc
    ob.rotation_quaternion = q
    ob.keyframe_insert("location", frame=f)
    ob.keyframe_insert("rotation_quaternion", frame=f)
    return q


def key_hidden(ob, keys):
    """keys: (frame, hidden) pairs; boolean keys hold their value until the next one."""
    for f, hidden in keys:
        ob.hide_render = hidden
        ob.keyframe_insert("hide_render", frame=f)


def animate_trial(fps=24, slow=4.0):
    """Key one whole trial on interface A1 (arming, release, impact, two scans and four photos from the sensor
    arm, video while the arm sweeps and the phone tilts, reseating), then the same impact on B1. Returns the
    timeline used by render_trial_step() and encode_trial(); it is also written to ANIM / timeline.json."""
    sc = bpy.context.scene
    objs = bpy.data.objects
    pp, ph, sa = P["pendulum"], P["phone"], sc["sensor_arm"]
    splint = objs["splint"]
    pad = splint.matrix_world.translation.copy()
    strike = Vector(sc["strike_point"])
    target = Vector(sc["target"])
    push = -raised_dir(pp["direction_deg"])
    axis = Vector((0.0, 0.0, 1.0)).cross(push).normalized()
    sc.frame_set(1)
    show()

    # moving bodies of the marker on A1 and on B1, all children of one empty
    motion = empty("marker_motion", bpy.data.collections["marker"], 15.0)
    e0 = Matrix.Translation(pad + Vector((0.0, 0.0, P["a1"]["base"][1])))
    motion.matrix_world = e0
    attach(motion, splint)
    bpy.context.view_layer.update()
    for name in MARKER_PARTS + ["a1_ball_stud", "a1_arm", "b1_plate", "b1_balls"]:
        ob = objs[name]
        mw = ob.matrix_world.copy()
        ob.parent = motion
        ob.matrix_parent_inverse = motion.matrix_world.inverted()
        ob.matrix_world = mw

    def translucent(name, rgb, alpha, emission=0.0):
        m = bpy.data.materials.new(name)
        bsdf = m.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Alpha"].default_value = alpha
        if emission:
            bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
            bsdf.inputs["Emission Strength"].default_value = emission
        m.surface_render_method = "BLENDED"
        m.diffuse_color = (*rgb, alpha)
        return m

    ghost = bpy.data.objects.new("marker_ghost", objs["marker_holder"].data.copy())
    ghost.data.materials.clear()
    ghost.data.materials.append(translucent("ghost_marker", (0.1, 0.8, 0.2), 0.35))
    bpy.data.collections["marker"].objects.link(ghost)
    ghost.matrix_world = objs["marker_holder"].matrix_world.copy()

    # light cone of the scanner, on the arm: apex on the projector, the 200 x 150 field at the working distance
    sp = P["scanner"]
    fw, fh = sp["field"][0] / 2, sp["field"][1] / 2
    bm = bmesh.new()
    apex = bm.verts.new((0.0, 0.0, 0.0))
    rim = [bm.verts.new((x, y, -sp["min_distance"])) for x, y in ((-fw, -fh), (fw, -fh), (fw, fh), (-fw, fh))]
    for i in range(4):
        bm.faces.new((apex, rim[i], rim[(i + 1) % 4]))
    bm.faces.new(rim[::-1])
    k0 = ph["shown"]
    pivot = objs["sensor_arm_pivot"]
    cone = new_obj("scanner_light", bm, bpy.data.collections["sensor_arm"], translucent("scanner_light", (0.25, 0.5, 1.0), 0.15, 0.6),
                   objs[f"cam_scanner_{k0 + 1}"].matrix_world.copy())
    attach(cone, pivot)
    # camera of the video, riding on the phone
    tilt, pin = objs["phone_tilt"], objs["arm_plunger_pin"]
    cam_v = bpy.data.objects.new("cam_video", bpy.data.cameras.new("cam_video"))
    bpy.data.collections["render_setup"].objects.link(cam_v)
    cam_v.data.sensor_width, cam_v.data.sensor_fit, cam_v.data.lens = 36.0, "HORIZONTAL", ph["focal_35mm"]
    cam_v.data.clip_start, cam_v.data.clip_end = 5.0, 5000.0
    cam_v.matrix_world = objs[f"cam_phone_{k0 + 1}"].matrix_world.copy()
    attach(cam_v, tilt)

    # animation camera and its shots
    cam = bpy.data.objects.new("cam_trial", bpy.data.cameras.new("cam_trial"))
    bpy.data.collections["render_setup"].objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    cam.data.clip_start, cam.data.clip_end = 1.0, 10000.0
    side = Vector((-push.y, push.x, 0.0))
    shots = {
        "side": (strike + side * 700.0 - push * 420.0 + Vector((0.0, 0.0, 200.0)), strike - push * 120.0 + Vector((0.0, 0.0, 140.0)), 35.0),
        "mid": (strike + side * 340.0 - push * 230.0 + Vector((0.0, 0.0, 100.0)), strike - push * 70.0 + Vector((0.0, 0.0, 45.0)), 35.0),
        "wide": (target + azimuth_dir(110.0, 42.0) * 1300.0, target + Vector((110.0, -60.0, -60.0)), 35.0),
        "video": (target + azimuth_dir(60.0, 62.0) * 1300.0, target + Vector((90.0, -90.0, -90.0)), 35.0),
        "close": (pad + Vector((190.0, -140.0, 110.0)), pad + Vector((0.0, -18.0, 20.0)), 50.0),
    }
    state = {"q": None}

    def key_cam(f, shot):
        loc, aim, lens = shots[shot]
        state["q"] = quat_key(cam, f, loc, (aim - loc).to_track_quat("-Z", "Y"), state["q"])
        cam.data.lens = lens
        cam.data.keyframe_insert("lens", frame=f)

    swing, horn = objs["pendulum_swing"], objs["pendulum_servo_horn"]
    th0 = pp["release_deg"]
    n_down = round(math.pi / 2 * math.sqrt(pp["L"] / 9810.0) * slow * fps)   # quarter period, small-angle law

    def key_swing(f, deg):
        swing.rotation_euler.x = math.radians(-deg)
        swing.keyframe_insert("rotation_euler", index=0, frame=f)

    def key_hook(f, opened):
        horn.rotation_euler.x = math.radians(pp["servo"]["open_deg"]) if opened else 0.0
        horn.keyframe_insert("rotation_euler", index=0, frame=f)

    def release(f):
        """Hook opens at f, the pendulum leaves at f + 4 and reaches the post at the returned frame."""
        key_hook(f, False)
        key_hook(f + 4, True)
        key_swing(f + 4, th0)
        for k in range(1, n_down + 1):
            key_swing(f + 4 + k, th0 * math.cos(math.pi / 2 * k / n_down))
        return f + 4 + n_down

    def key_motion(f, m):
        motion.matrix_world = m @ e0
        bpy.context.view_layer.update()
        motion.keyframe_insert("location", frame=f)
        motion.keyframe_insert("rotation_euler", frame=f)

    def turn(deg, centre, ax):
        return Matrix.Translation(centre) @ Matrix.Rotation(math.radians(deg), 4, ax) @ Matrix.Translation(-centre)

    phis = list(sa["arm_deg"])
    f_shown = sa["shown_deg"]
    pin_z = pin.location.z
    tilt_x = tilt.rotation_euler.x

    def key_arm(f, phi, lifted, tilt_deg=0.0):
        pivot.rotation_euler.z = math.radians(phi - f_shown)
        pivot.keyframe_insert("rotation_euler", index=2, frame=f)
        pin.location.z = pin_z + (P["arm"]["plunger"]["lift"] if lifted else 0.0)
        pin.keyframe_insert("location", index=2, frame=f)
        tilt.rotation_euler.x = tilt_x + math.radians(tilt_deg)
        tilt.keyframe_insert("rotation_euler", index=0, frame=f)

    def move_arm(f, phi_from, phi_to, frames=18):
        """Plunger up, arm to the next position, plunger down; returns the frame after."""
        key_arm(f, phi_from, False)
        key_arm(f + 6, phi_from, True)
        key_arm(f + 6 + frames, phi_to, True)
        key_arm(f + 12 + frames, phi_to, False)
        return f + 13 + frames

    ball_a = pad + Vector((0.0, 0.0, P["a1"]["base"][1]))
    _, _, z_ball, _ = kinematic_levels()
    balls = [pad + polar(P["b1"]["pitch_r"], d, z_ball) for d in P["b1"]["groove_deg"]]
    pairs = [(balls[i], balls[j]) for i, j in ((0, 1), (1, 2), (2, 0))]
    far = max(pairs, key=lambda q: ((q[0] + q[1]) / 2 - pad).dot(push))
    hinge_b, axis_b = (far[0] + far[1]) / 2, (far[1] - far[0]).normalized()
    if axis_b.dot(axis) < 0.0:
        axis_b = -axis_b
    tilt_a, tilt_b = 7.0, 5.0
    steps, insets, flashes = [], [], []

    # 1 arming: the pendulum hangs on the cap, is raised to the release angle and hooked
    f = 1
    key_cam(f, "side")
    key_cam(f + 72, "side")
    key_cam(f + 89, "mid")
    key_swing(f, 0.0)
    key_hook(f, True)
    key_swing(f + 12, 0.0)
    key_swing(f + 60, th0)
    key_hook(f + 60, True)
    key_hook(f + 72, False)
    key_motion(f, Matrix())
    key_arm(f, f_shown, False)
    steps.append((1, f + 89))
    # 2 release
    f = f + 90
    f_imp = release(f)
    steps.append((f, f_imp - 1))
    # 3 impact: A1 slips and stays; rebound, the hand stops the pendulum and hooks it again
    f = f_imp
    key_motion(f, Matrix())
    key_motion(f + 3, turn(tilt_a, ball_a, axis))
    key_swing(f + 10, 10.0)
    key_swing(f + 20, 6.0)
    key_swing(f + 44, th0)
    key_hook(f + 44, True)
    key_hook(f + 52, False)
    key_cam(f + 60, "mid")
    steps.append((f, f + 60))
    # 4 two scans from the arm
    f = f + 61
    key_cam(f + 20, "wide")
    phi = f_shown
    cone_keys = [(1, True)]
    for k in SCAN_POSITIONS:
        f = move_arm(f, phi, phis[k], 22)
        phi = phis[k]
        cone_keys += [(f + 4, False), (f + 12, True), (f + 15, False), (f + 22, True)]
        insets.append({"kind": "photo", "camera": f"cam_scanner_{k + 1}", "hide": "scanner_", "file": f"inset_scan_{k + 1}.png",
                       "start": f + 4, "end": f + 30, "label": f"scansione da {sa['scanner_deg'][k]:.0f} gradi"})
        f += 31
    key_hidden(cone, cone_keys)
    steps.append((steps[-1][1] + 1, f - 1))
    # 5 photos at the four positions
    for k in range(len(phis)):
        f = move_arm(f, phi, phis[k], 18)
        phi = phis[k]
        flashes.append(f + 1)
        insets.append({"kind": "photo", "camera": f"cam_phone_{k + 1}", "hide": "phone", "file": f"inset_photo_{k + 1}.png", "start": f + 1,
                       "end": f + 24, "label": f"foto {k + 1} di {len(phis)}, {ph['azimuths_deg'][k]:.0f} gradi"})
        f += 25
    steps.append((steps[-1][1] + 1, f - 1))
    # 6 video: plunger up, sweep to the far end and back to the near one while the phone tilts, back to the shown position
    lo, hi = sa["sweep_deg"]
    key_cam(f, "wide")
    key_cam(f + 20, "video")
    key_arm(f, phi, False)
    key_arm(f + 6, phi, True)
    n = 150
    t_amp = max(abs(x) for x in ph["tilt_deg"])
    for s in np.linspace(0.0, 1.0, 31):
        a = phi + (hi - phi) * s / 0.2 if s <= 0.2 else hi + (lo - hi) * (s - 0.2) / 0.8   # out to the far end, then all the way back
        key_arm(f + 10 + round(s * n), a, True, t_amp * math.sin(4.0 * math.pi * s))           # two tilts up and down
    f_v0, f_v1 = f + 10, f + 10 + n
    insets.append({"kind": "video", "camera": "cam_video", "hide": "phone", "file": "inset_video_{:04d}.png", "start": f_v0, "end": f_v1,
                   "label": "video dal braccio"})
    f = f_v1 + 4
    f = move_arm(f, lo, f_shown, 24)
    key_cam(f, "video")
    key_cam(f + 16, "wide")
    steps.append((steps[-1][1] + 1, f + 16))
    # 7 reseating of the marker on A1
    f = f + 17
    key_cam(f + 30, "close")
    key_motion(f + 36, turn(tilt_a, ball_a, axis))
    key_motion(f + 72, Matrix())
    key_cam(f + 96, "close")
    key_hidden(ghost, [(1, True), (f_imp + 1, False), (f + 72, True)])
    steps.append((f, f + 96))
    # 8 the same impact on B1: the plate tips about the two far balls and the spring brings it back
    f = f + 97
    key_cam(f + 30, "mid")
    f_imp_b = release(f + 36)
    key_motion(f_imp_b, Matrix())
    key_motion(f_imp_b + 3, turn(tilt_b, hinge_b, axis_b))
    key_motion(f_imp_b + 9, turn(-0.6, hinge_b, axis_b))
    key_motion(f_imp_b + 13, turn(0.2, hinge_b, axis_b))
    key_motion(f_imp_b + 16, Matrix())
    key_swing(f_imp_b + 10, 10.0)
    key_swing(f_imp_b + 20, 6.0)
    key_swing(f_imp_b + 56, 6.0)
    steps.append((f, f_imp_b + 56))

    sc.render.fps = fps
    sc.frame_start, sc.frame_end = 1, steps[-1][1]
    sc.camera = cam
    sc.frame_set(1)
    timeline = {"fps": fps, "frames": steps[-1][1], "b_from": steps[-1][0],
                "steps": [{"n": i + 1, "title": t, "text": x, "start": s, "end": e} for i, ((t, x), (s, e)) in enumerate(zip(TRIAL_STEPS, steps))],
                "insets": insets, "flashes": flashes}
    ANIM.mkdir(parents=True, exist_ok=True)
    (ANIM / "timeline.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
    return timeline


def render_trial_step(timeline, state, size=(960, 540), inset=(480, 270), samples=16):
    """Render the next frame of the trial animation (and its inset, if any) to ANIM; call repeatedly from a
    timer until it returns False. state: dict, empty at the start."""
    sc = bpy.context.scene
    objs = bpy.data.objects
    f = state.setdefault("frame", 1)
    if f > timeline["frames"]:
        show()
        sc.frame_set(1)
        sc.render.resolution_x, sc.render.resolution_y = 1600, 1200
        sc.eevee.taa_render_samples = 64
        return False
    sc.frame_set(f)
    show(variant="b1" if f >= timeline["b_from"] else "a1")
    sc.eevee.taa_render_samples = samples
    for ins in timeline["insets"]:
        photo = ins["kind"] == "photo" and f == ins["start"]
        if photo or (ins["kind"] == "video" and ins["start"] <= f <= ins["end"]):
            prefixes = ("smartphone", "phone_cradle") if ins["hide"] == "phone" else (ins["hide"],)
            hidden = [o for o in objs if o.name.startswith(prefixes) and not o.hide_render]
            for o in hidden:
                o.hide_render = True
            sc.camera = objs[ins["camera"]]
            sc.render.resolution_x, sc.render.resolution_y = inset
            sc.render.filepath = str(ANIM / ins["file"].format(f))
            bpy.ops.render.render(write_still=True)
            for o in hidden:
                o.hide_render = False
    sc.camera = objs["cam_trial"]
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.filepath = str(ANIM / f"frame_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    state["frame"] = f + 1
    return True


def encode_trial(out=None, still=None):
    """Compose the frames with captions, insets and flashes in the sequencer and write the MP4 (or, with
    still = frame number, that frame as a PNG to check the layout).
    Run in a separate Blender: blender --background --factory-startup --python build_test_bench.py -- --encode [frame out.png]"""
    out = Path(out) if out else RENDERS / "trial_animation.mp4"
    tl = json.loads((ANIM / "timeline.json").read_text(encoding="utf-8"))
    sc = bpy.context.scene      # factory-startup scene: the sequencer output replaces the 3D render
    frames = sorted(ANIM.glob("frame_*.png"))
    first = bpy.data.images.load(str(frames[0]))
    W, H = first.size
    bpy.data.images.remove(first)
    r = sc.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    r.fps = tl["fps"]
    sc.frame_start, sc.frame_end = 1, tl["frames"]
    sc.view_settings.view_transform, sc.view_settings.look = "Standard", "None"
    r.image_settings.media_type = "VIDEO"
    r.image_settings.file_format = "FFMPEG"
    r.ffmpeg.format, r.ffmpeg.codec = "MPEG4", "H264"
    r.ffmpeg.constant_rate_factor, r.ffmpeg.ffmpeg_preset = "HIGH", "GOOD"
    r.filepath = str(out)
    se = sc.sequence_editor_create()
    strips = se.strips
    base = strips.new_image("frames", str(frames[0]), 1, 1, fit_method="ORIGINAL")
    for p in frames[1:]:
        base.elements.append(p.name)
    fonts = {k: bpy.data.fonts.load(str(FONTS / f)) for k, f in (("bold", "Abadi-Bold.ttf"), ("regular", "Abadi.ttf"))}

    def text(name, body, channel, start, end, x, y, size, font="regular", box=None, color=(0.0, 0.0, 0.0, 1.0), anchor="LEFT"):
        t = strips.new_effect(name, "TEXT", channel, frame_start=start, length=end - start + 1)
        t.text, t.font, t.font_size, t.color = body, fonts[font], size, color
        t.location = (x, y)
        t.anchor_x, t.anchor_y, t.alignment_x = anchor, "BOTTOM", anchor
        t.wrap_width = 0.92
        if box:
            t.use_box, t.box_color, t.box_margin = True, box, 0.012
        return t

    n_a = len(tl["steps"]) - 1      # captions on the top channels: below the insets they disappear
    for s in tl["steps"]:
        tag = f"{s['n']} di {n_a}  " if s["n"] <= n_a else ""
        text(f"title_{s['n']}", f"{tag}{s['title']}", 9, s["start"], s["end"], 0.04, 0.115, 30, "bold", box=(1.0, 1.0, 1.0, 0.85))
        text(f"text_{s['n']}", s["text"], 10, s["start"], s["end"], 0.04, 0.045, 20, box=(1.0, 1.0, 1.0, 0.85))
    text("header", "Una prova d'urto su A1 (circa 5 min reali)", 11, 1, tl["b_from"] - 1, 0.04, 0.93, 20,
         box=(1.0, 1.0, 1.0, 0.7))
    iw, ih = 480, 270
    ox, oy = W / 2 - iw / 2 - 18, H / 2 - ih / 2 - 18
    for i, ins in enumerate(tl["insets"]):
        start, end = ins["start"], ins["end"]
        frame = strips.new_effect(f"inset_frame_{i}", "COLOR", 5, frame_start=start, length=end - start + 1)
        frame.color = (0.1, 0.1, 0.1)
        frame.transform.scale_x, frame.transform.scale_y = (iw + 6) / W, (ih + 6) / H
        frame.transform.offset_x, frame.transform.offset_y = ox, oy
        if ins["kind"] == "photo":
            img = strips.new_image(f"inset_{i}", str(ANIM / ins["file"]), 6, start, fit_method="ORIGINAL")
            img.frame_final_duration = end - start + 1
        else:
            files = [ANIM / ins["file"].format(f) for f in range(start, end + 1)]
            img = strips.new_image(f"inset_{i}", str(files[0]), 6, start, fit_method="ORIGINAL")
            for p in files[1:]:
                img.elements.append(p.name)
        img.transform.offset_x, img.transform.offset_y = ox, oy
        text(f"inset_label_{i}", ins["label"], 7, start, end, (W - 18) / W, (H - 18 - ih - 34) / H, 18, box=(1.0, 1.0, 1.0, 0.85),
             anchor="RIGHT")
    for i, f in enumerate(tl["flashes"]):
        flash = strips.new_effect(f"flash_{i}", "COLOR", 8, frame_start=f, length=2)
        flash.color = (1.0, 1.0, 1.0)
        flash.blend_alpha = 0.6
    if still:
        r.image_settings.media_type, r.image_settings.file_format = "IMAGE", "PNG"
        sc.frame_current = still
        r.filepath = str(out)
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)
    return str(out)


def save():
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND), compress=True)


def main(argv):
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    if "--encode" in args:
        rest = args[args.index("--encode") + 1:]
        print(encode_trial(rest[1], int(rest[0])) if len(rest) == 2 else encode_trial())
        return
    print(build())
    if "--render" in args:
        print(render_checks())
    if "--save" in args:
        save()


if __name__ == "__main__":
    main(sys.argv)
