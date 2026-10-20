"""The bench as the twin of ImpactGuide shows it: a light 3D model of the scene, and the fingerprint of what the phone
sees. Runs inside Blender 5, in the background, on a copy of the bench scene (never on the original, which is edited
elsewhere); twin.py starts it and keeps what it writes.

    blender --background --factory-startup COPY.blend --python-exit-code 1 --python twin_export.py -- export OUT
    blender --background --factory-startup --python-exit-code 1 --python twin_export.py -- print
    blender --background --factory-startup --python-exit-code 1 --python twin_export.py -- test

export writes OUT/bench.glb and OUT/bench.json (what the GLB's scene carries as extras, for the host); print says the
fingerprint of the open scene as one JSON line; test builds a small scene in memory and checks both.

The model. What moves is a node of its own, named so the page can move it:
    bench           everything that stays where it is
    marker          marker_motion and its children (board, holder, cap, marker spheres): moved by the impact
    marker_a/_b     the A1 ball stud, the B1 balls: children of marker_motion shown only with their holder
    holder_a/_b     the rest of A1, of B1, shown only with their holder
    holder_a_arm    a1_arm, holder_b_plate b1_plate: they move with the marker
    pendulum        pendulum_base and its frame, turned about the vertical for the direction of the impact
    pendulum_swing  its swinging part (a child of pendulum), turned about the axle for the hole
    drape           a drape of the synthetic dataset (synth/drapes/drape_00.npz), for the trials with one
    arm             sensor_arm_pivot with the phone and the scanner, turned about the vertical for the arm position
    arm_pose_k      what follows the arm without being carried by it (the cables hooked to the phone and the scanner),
                    as it is at each of the four arm positions: the page shows the one of the arm (and pendulum_pose_k
                    for the pendulum's directions, when anything follows it)
The nodes that move with the marker are in world coordinates (their matrix is the marker's motion); pendulum,
pendulum_swing and arm carry the frame of their empty. Each node has one mesh, with one primitive per material and
per 65535 vertices (the meshes of a node merged by material: few draw calls), positions and normals in the node's
frame, mm, z up as in the scene.

Shape. Every part keeps the mesh Blender shows, with its normals as Blender computes them (sharp edges where it has
them): nothing is decimated but the organic meshes of REDUCE (face, cap, splint, skull), collapsed gently to their
count of triangles, and the cables, made again as tubes of eight sides. shape_change() compares each reduced object
before and after (volume, area, box): one whose volume moves by more than SHAPE_VOLUME or whose box by more than
SHAPE_BOX_MM is kept whole instead.

File. Positions are quantized to 16 bits over the box of each primitive (KHR_mesh_quantization), normals to 8 bits,
indices are of 16 bits, and the binary chunk is compressed with zlib (asset.extras.compression; the page inflates it
with DecompressionStream). The largest error of a position is in the extras (quant_mm).

Look. Materials as the Principled BSDF has them (base colour, metallic, roughness, alpha, emission); the scene's lamps
(direction, strength, colour, angle) and the world (the light it gives, the colour the camera sees) go in the extras,
for the page to light the model as Blender does (the scene's view transform is Standard: no tone mapping).

Fingerprint. Only what makes the phone's pictures: face, cap, marker, splint, A1 and B1, the spheres and their frames,
skull and mandible, the lamps and the light panels, the phone's lens and cameras, the arm positions (the scene's
sensor_arm), the materials of all of these and the world. Not the cables, the hub, the wood, the profiles, the
pendulum. Each object counts with its geometry after modifiers in world coordinates (to 1 um), its transform, its
materials (every value of their nodes), each lamp with its values: a dataset or a network made on a scene with
another fingerprint is out of date.
"""
import hashlib
import json
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path

# Blender's Python ignores PYTHONPYCACHEPREFIX: apply it here, and write no bytecode next to the sources without it
if os.environ.get("PYTHONPYCACHEPREFIX"):
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
else:
    sys.dont_write_bytecode = True

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix  # noqa: E402

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]
TARGETS = ROOT / "03_data" / "processed" / "targets" / "mandible_targets.csv"   # TRE targets (role "proposed")

REDUCE = {"silicone_face": 180000, "surgical_cap": 110000, "splint": 110000, "skull": 70000}   # organic only
SHAPE_VOLUME = 0.004    # a reduced mesh keeps its volume within this share...
SHAPE_BOX_MM = 0.25     # ... and its box within this, else it is kept whole
CABLE_RES = (1, 4)      # cables: bevel resolution (eight sides) and resolution along the curve, at most
PENDULUM_DEG = (0.0, 45.0, 90.0)    # the directions of the impact (synth_twin.py, trials.py)
DRAPE_LINEAR = tuple(((c / 255 + 0.055) / 1.055) ** 2.4 for c in (38, 104, 112))   # synth_twin.DRAPE_SRGB
MAX_VERTS = 65535       # vertices of one primitive: indices of 16 bits
ZLIB_LEVEL = 6
SKIP = ("face_screws", "tmp_cam", "marker_ghost", "lab_table", "synth_")     # never in the model (prefixes)
HOLDERS = {"A": "a1_ball_stud", "B": "b1_kinematic_spring"}                  # collections of each holder
WITH_MARKER = {"A": "a1_arm", "B": "b1_plate"}                              # moved with the marker
MARKER, PEND_BASE, PEND_SWING, ARM = "marker_motion", "pendulum_base", "pendulum_swing", "sensor_arm_pivot"
BOARD, BALL, FACE, MANDIBLE = "marker_charuco", "a1_ball_stud", "silicone_face", "mandible"
LENS, PHONE, PHONE_CAM, SCANNER_CAM = "smartphone_lens", "smartphone", "cam_phone_2", "cam_scanner_2"
# what the phone sees (prefixes of object names), lamps and world aside
PHONE_SEES = ("silicone_face", "surgical_cap", "marker_", "gt_sphere_", "gt_frame_", "splint", "a1_", "b1_", "skull",
              "mandible", "smartphone_lens", "light_panel_", "cam_phone")
NODE_PROPS = ("blend_type", "operation", "data_type", "interpolation", "projection", "extension", "distribution",
              "subsurface_method", "use_clamp", "clamp_type", "falloff_type")


# ---------------------------------------------------------------- small helpers

def arr(coll, attr, width, dtype=np.float64):
    a = np.empty(len(coll) * width, dtype)
    coll.foreach_get(attr, a)
    return a.reshape(-1, width) if width > 1 else a


def mat(m):
    return np.array([list(r) for r in m], np.float64)


def rows(m, nd=5):
    return [[round(float(x), nd) for x in r] for r in np.asarray(m)]


def canon(x, nd=6):
    """x as plain JSON values, floats rounded (so that a value saved again unchanged gives the same text)."""
    if isinstance(x, dict):
        return {str(k): canon(v, nd) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [canon(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return canon(x.tolist(), nd)
    if hasattr(x, "to_dict"):
        return canon(x.to_dict(), nd)
    if hasattr(x, "to_list"):
        return canon(x.to_list(), nd)
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return round(x, nd) + 0.0
    try:
        return [canon(v, nd) for v in x]
    except TypeError:
        return str(x)


def digest(x):
    return hashlib.sha256(json.dumps(canon(x), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hidden_collections():
    """The collections the renders leave out: their own render switch off, or that of a collection holding them
    (the design alternatives kept in the scene)."""
    out = set()

    def walk(c, hidden):
        hidden = hidden or c.hide_render
        if hidden:
            out.add(c.name)
        for ch in c.children:
            walk(ch, hidden)
    walk(bpy.context.scene.collection, False)
    return out


def present():
    """The objects of the design as it stands, as the renders show it: in the view layer, not hidden in the viewport
    nor left out of the renders (by themselves or by a collection), plus both holders (one of them is always left out
    of the renders: the page shows the one of the trial)."""
    vl = bpy.context.view_layer
    holders = {ob.name for c in HOLDERS.values() if c in bpy.data.collections
               for ob in bpy.data.collections[c].all_objects}
    hidden = hidden_collections()
    out = []
    for ob in bpy.data.objects:
        if ob.name.startswith(SKIP) or ob.name not in vl.objects:
            continue
        shown = not ob.hide_get() and not ob.hide_render and not any(c.name in hidden for c in ob.users_collection)
        if ob.name in holders or shown:
            out.append(ob)
    return out


def world_mesh(ob, dg):
    """World vertices (mm) and triangles of an object after its modifiers; None for one with no surface."""
    ev = ob.evaluated_get(dg)
    try:
        me = ev.to_mesh()
    except RuntimeError:
        return None
    if me is None:
        return None
    me.calc_loop_triangles()
    co = arr(me.vertices, "co", 3)
    tri = arr(me.loop_triangles, "vertices", 3, np.int64)
    ev.to_mesh_clear()
    mw = mat(ob.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3], tri


# ---------------------------------------------------------------- fingerprint

def material_record(m):
    if m is None:
        return None
    r = {"name": m.name, "diffuse": list(m.diffuse_color), "method": getattr(m, "surface_render_method", "")}
    if m.node_tree is not None:
        r["nodes"] = tree_record(m.node_tree)
    return r


def tree_record(tree):
    nodes = []
    for n in sorted(tree.nodes, key=lambda n: n.name):
        rec = {"name": n.name, "type": n.bl_idname,
               "in": [[s.identifier, None if s.is_linked or not hasattr(s, "default_value") else canon(s.default_value)]
                      for s in n.inputs],
               "props": {p: canon(getattr(n, p)) for p in NODE_PROPS if hasattr(n, p)}}
        if getattr(n, "color_ramp", None) is not None:
            rec["ramp"] = [[e.position, list(e.color)] for e in n.color_ramp.elements] + [n.color_ramp.interpolation]
        if getattr(n, "image", None) is not None:
            im = n.image
            px = np.empty(len(im.pixels), np.float32)
            im.pixels.foreach_get(px)
            rec["image"] = [im.name, list(im.size), hashlib.sha256(np.round(px, 3).tobytes()).hexdigest()]
        if getattr(n, "node_tree", None) is not None:       # a group
            rec["group"] = tree_record(n.node_tree)
        nodes.append(rec)
    links = sorted([l.from_node.name, l.from_socket.identifier, l.to_node.name, l.to_socket.identifier]
                   for l in tree.links)
    return {"nodes": nodes, "links": links}


def object_record(ob, dg):
    r = {"name": ob.name, "type": ob.type, "matrix": rows(ob.matrix_world, 4),
         "materials": [material_record(s.material) for s in ob.material_slots]}
    if ob.type in ("MESH", "CURVE", "SURFACE", "FONT"):
        wm = world_mesh(ob, dg)
        if wm is not None:
            v, f = wm
            r["mesh"] = [len(v), len(f), hashlib.sha256((np.round(v, 3) + 0.0).tobytes()).hexdigest(),
                         hashlib.sha256(f.astype(np.int64).tobytes()).hexdigest()]
    elif ob.type == "LIGHT":
        d = ob.data
        r["light"] = {k: canon(getattr(d, k)) for k in ("type", "color", "energy", "shadow_soft_size", "angle",
                                                         "spot_size", "spot_blend", "size", "size_y", "shape",
                                                         "use_shadow") if hasattr(d, k)}
        if d.node_tree is not None:
            r["nodes"] = tree_record(d.node_tree)
    elif ob.type == "CAMERA":
        d = ob.data
        r["camera"] = {k: canon(getattr(d, k)) for k in ("type", "lens", "sensor_width", "sensor_height", "sensor_fit",
                                                          "shift_x", "shift_y", "clip_start")}
    return r


def fingerprint():
    """The fingerprint of what the phone sees (16 hex digits) and the digest of each part of it, by name."""
    dg = bpy.context.evaluated_depsgraph_get()
    sc = bpy.context.scene
    items = {}
    for ob in present():
        if ob.type == "LIGHT" or ob.name.startswith(PHONE_SEES):
            items[ob.name] = digest(object_record(ob, dg))[:10]
    if sc.world is not None:
        items["(world)"] = digest({"color": list(sc.world.color),
                                   "nodes": tree_record(sc.world.node_tree) if sc.world.node_tree else None})[:10]
    if sc.get("sensor_arm") is not None:
        items["(arm positions)"] = digest(sc["sensor_arm"].to_dict())[:10]
    return digest(sorted(items.items()))[:16], items


# ---------------------------------------------------------------- the model

def in_collection(names):
    return {ob.name for c in names if c in bpy.data.collections for ob in bpy.data.collections[c].all_objects}


def under(ob, name):
    while ob is not None:
        if ob.name == name:
            return True
        ob = ob.parent
    return False


def node_of(ob, coll_a, coll_b):
    if under(ob, MARKER):
        return "marker_a" if ob.name in coll_a else "marker_b" if ob.name in coll_b else "marker"
    if under(ob, WITH_MARKER["A"]):
        return "holder_a_arm"
    if under(ob, WITH_MARKER["B"]):
        return "holder_b_plate"
    if ob.name in coll_a:
        return "holder_a"
    if ob.name in coll_b:
        return "holder_b"
    if under(ob, PEND_SWING):
        return "pendulum_swing"
    if under(ob, PEND_BASE):
        return "pendulum"
    if under(ob, ARM):
        return "arm"
    return "bench"


PARTS = {   # what each node does in the page
    "bench": {}, "marker": {"moves": "marker"}, "marker_a": {"moves": "marker", "holder": "A"},
    "marker_b": {"moves": "marker", "holder": "B"}, "holder_a": {"holder": "A"}, "holder_b": {"holder": "B"},
    "holder_a_arm": {"moves": "marker", "holder": "A"}, "holder_b_plate": {"moves": "marker", "holder": "B"},
    # direction: about the vertical of the world through the base (synth_twin.set_pendulum); hole: about the axle,
    # the swing's own x (rotation_euler.x = -angle); arm: about the pivot's own z (synth_twin.set_arm)
    "pendulum": {"turns": "direction", "axis": [0, 0, 1], "sign": -1, "frame": "world"},
    "pendulum_swing": {"turns": "angle", "axis": [1, 0, 0], "sign": -1, "frame": "local", "parent": "pendulum"},
    "arm": {"turns": "arm", "axis": [0, 0, 1], "sign": 1, "frame": "local"},
}


def node_frames():
    """World matrix of the origin of each node (identity for those in world coordinates)."""
    o = bpy.data.objects
    out = {k: np.eye(4) for k in PARTS}
    for node, name in (("pendulum", PEND_BASE), ("pendulum_swing", PEND_SWING), ("arm", ARM)):
        if name in o:
            out[node] = mat(o[name].matrix_world)
    return out


def shape_of(ob, dg):
    """What a mesh is, in world coordinates: triangles, box, volume (signed, by the divergence theorem) and area."""
    wm = world_mesh(ob, dg)
    if wm is None or not len(wm[1]):
        return None
    v, f = wm
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    return {"tris": int(len(f)), "min": v.min(0), "max": v.max(0),
            "volume": float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0),
            "area": float(np.linalg.norm(cr, axis=1).sum() / 2.0)}


def shape_change(s0, s1):
    """How far a mesh moved from s0 to s1: share of its volume, of its area, and the box (mm)."""
    vol = abs(s1["volume"] - s0["volume"]) / max(abs(s0["volume"]), 1e-9)
    area = abs(s1["area"] - s0["area"]) / max(s0["area"], 1e-9)
    box = float(max(np.abs(s1["min"] - s0["min"]).max(), np.abs(s1["max"] - s0["max"]).max()))
    return {"tris": [s0["tris"], s1["tris"]], "volume": round(vol, 6), "area": round(area, 6), "box_mm": round(box, 4)}


def set_pose(kind, value):
    """The arm turned to a position (value: its angle, degrees, as sensor_arm's arm_deg) or the pendulum to a
    direction (degrees), as synth_twin.py turns them; None puts back the scene's own pose."""
    o = bpy.data.objects
    if kind == "arm" and ARM in o:
        shown = bpy.context.scene["sensor_arm"].to_dict().get("shown_deg", 0.0)
        o[ARM].rotation_euler.z = 0.0 if value is None else math.radians(value - shown)
    elif kind == "pendulum" and PEND_BASE in o:
        b = o[PEND_BASE]
        if "twin_m0" not in b:
            b["twin_m0"] = [list(r) for r in b.matrix_world]
        m0 = Matrix(b["twin_m0"])
        t = m0.translation
        b.matrix_world = m0 if value is None else (Matrix.Translation(t) @ Matrix.Rotation(math.radians(-value), 4, "Z")
                                                     @ Matrix.Translation(-t) @ m0)
    bpy.context.view_layer.update()


def poses(kind):
    """The values a pose of the page takes: the arm's four positions, the pendulum's three directions."""
    if kind == "arm":
        sa = bpy.context.scene.get("sensor_arm")
        return list(sa.to_dict().get("arm_deg", [])) if sa is not None else []
    return list(PENDULUM_DEG)


def followers(objects, kind):
    """The objects that change shape or place when the arm (or the pendulum) moves but are not carried by it: the
    cables, hooked to the phone and the scanner. Found by moving it: only objects with modifiers or constraints,
    and curves, can follow."""
    root = ARM if kind == "arm" else PEND_BASE
    cand = [ob for ob in objects
            if not under(ob, root) and (ob.type != "MESH" or len(ob.modifiers) or len(ob.constraints))]
    values = poses(kind)
    if not cand or len(values) < 2:
        return []

    def shape():
        dg = bpy.context.evaluated_depsgraph_get()
        out = {}
        for ob in cand:
            wm = world_mesh(ob, dg)
            out[ob.name] = None if wm is None else wm[0]
        return out
    base = shape()
    set_pose(kind, values[0] if kind == "pendulum" and values[0] != 0 else values[-1])
    moved = shape()
    set_pose(kind, None)
    return [ob for ob in cand if base[ob.name] is not None and (moved[ob.name] is None or len(moved[ob.name]) !=
            len(base[ob.name]) or np.abs(moved[ob.name] - base[ob.name]).max() > 0.01)]


def coarse_curves(objects):
    """The cables made as tubes of eight sides (CABLE_RES), before anything is taken of them."""
    for ob in objects:
        d = ob.data
        if ob.type == "CURVE" and hasattr(d, "bevel_resolution"):
            d.bevel_resolution = min(d.bevel_resolution, CABLE_RES[0])
            d.resolution_u = min(d.resolution_u, CABLE_RES[1])
    bpy.context.view_layer.update()


def meshed(objects):
    """The objects with each curve (the cables) replaced by a mesh of it, made as a tube of eight sides (CABLE_RES):
    the same path, fewer triangles."""
    out = []
    for ob in objects:
        if ob.type == "MESH":
            out.append(ob)
            continue
        d = ob.data
        if hasattr(d, "bevel_resolution"):
            d.bevel_resolution = min(d.bevel_resolution, CABLE_RES[0])
            d.resolution_u = min(d.resolution_u, CABLE_RES[1])
        dg = bpy.context.evaluated_depsgraph_get()
        try:
            me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        except RuntimeError:
            continue
        name = ob.name
        ob.name = name + ".curve"
        new = bpy.data.objects.new(name, me)
        for c in ob.users_collection:
            c.objects.link(new)
        new.parent, new.matrix_parent_inverse = ob.parent, ob.matrix_parent_inverse.copy()
        new.matrix_world = ob.matrix_world.copy()
        ob.hide_set(True)
        out.append(new)
    bpy.context.view_layer.update()
    return out


def reduce_organic(objects, reduce=None):
    """The organic meshes of REDUCE collapsed gently to their count of triangles, each kept only when its shape holds
    (shape_change within SHAPE_VOLUME and SHAPE_BOX_MM); every other mesh stays as it is. Returns {name: change} of
    the meshes looked at, with kept: whether the reduced mesh is the one used."""
    reduce = REDUCE if reduce is None else reduce
    out = {}
    for ob in objects:
        want = reduce.get(ob.name)
        if ob.type != "MESH" or want is None:
            continue
        dg = bpy.context.evaluated_depsgraph_get()
        s0 = shape_of(ob, dg)
        if s0 is None or s0["tris"] <= want:
            continue
        m = ob.modifiers.new("twin_reduce", "DECIMATE")
        m.decimate_type, m.ratio = "COLLAPSE", want / s0["tris"]
        dg = bpy.context.evaluated_depsgraph_get()
        ch = shape_change(s0, shape_of(ob, dg))
        ch["kept"] = ch["volume"] <= SHAPE_VOLUME and ch["box_mm"] <= SHAPE_BOX_MM
        if not ch["kept"]:
            ob.modifiers.remove(m)
        out[ob.name] = ch
    bpy.context.view_layer.update()
    return out


def corner_normals(me):
    if hasattr(me, "corner_normals"):
        return arr(me.corner_normals, "vector", 3)
    return arr(me.loops, "normal", 3)       # older Blender


def mesh_parts(ob, dg, A):
    """{material name: (positions, normals, triangles)} of an object, in the frame A (4 x 4, node from world)."""
    ev = ob.evaluated_get(dg)
    try:
        me = ev.to_mesh()
    except RuntimeError:
        return {}
    if me is None:
        return {}
    me.calc_loop_triangles()
    if not len(me.loop_triangles):
        ev.to_mesh_clear()
        return {}
    co = arr(me.vertices, "co", 3)
    corners = arr(me.loop_triangles, "loops", 3, np.int64)
    mi = arr(me.loop_triangles, "material_index", 1, np.int64)
    vi = arr(me.loops, "vertex_index", 1, np.int64)
    nrm = corner_normals(me)
    names = [s.material.name if s.material else "" for s in ob.material_slots] or [""]
    ev.to_mesh_clear()
    M = A @ mat(ob.matrix_world)
    L = M[:3, :3]
    pos = co @ L.T + M[:3, 3]
    nm = np.linalg.inv(L)          # normals by the inverse transpose: n' = n L^-1
    flip = np.linalg.det(L) < 0
    out = {}
    for k in np.unique(mi):
        c = corners[mi == k]
        if flip:
            c = c[:, ::-1]
        flat = c.ravel()
        n = nrm[flat] @ nm
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        key = np.concatenate([vi[flat][:, None], np.round(n * 512).astype(np.int64)], 1)
        uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        name = names[min(int(k), len(names) - 1)]
        out[name] = (pos[vi[flat][first]].astype(np.float32), n[first].astype(np.float32),
                     inv.reshape(-1, 3).astype(np.int64))
    return out


def material_values(m):
    """A material as the page draws it: base colour (linear, with alpha), metallic, roughness, and its emission
    (colour times strength, linear) as the Principled BSDF has them; a base colour made by nodes (the wood) is the
    material's viewport colour."""
    if m is None:
        return {"color": [0.8, 0.8, 0.8, 1.0], "metallic": 0.0, "roughness": 0.5, "emission": [0.0, 0.0, 0.0]}
    col, metal, rough, alpha, emit = list(m.diffuse_color), float(m.metallic), float(m.roughness), 1.0, [0.0] * 3
    b = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if m.node_tree else None
    if b is not None:
        def value(name, default):
            s = b.inputs.get(name)
            return default if s is None or s.is_linked else s.default_value
        bc = b.inputs.get("Base Color")
        if bc is not None and not bc.is_linked:
            col = list(bc.default_value)
        metal, rough, alpha = float(value("Metallic", 0.0)), float(value("Roughness", 0.5)), float(value("Alpha", 1.0))
        strength = float(value("Emission Strength", 0.0))
        ec = value("Emission Color", (0.0, 0.0, 0.0, 1.0))
        emit = [float(c) * strength for c in list(ec)[:3]]
        sw = float(value("Sheen Weight", 0.0))
        tint = list(value("Sheen Tint", (1.0, 1.0, 1.0, 1.0)))[:3]
        sheen = [round(sw * float(c), 4) for c in tint] if sw > 0 else None
    else:
        sheen = None
    return {"color": [col[0], col[1], col[2], alpha], "metallic": metal, "roughness": rough, "emission": emit,
            "sheen": sheen}


# ---------------------------------------------------------------- what the page needs besides the model

def ball_centre(ob):
    """Centre of the A1 ball: a sphere fitted to the vertices at the radius of the widest section."""
    v = world_mesh(ob, bpy.context.evaluated_depsgraph_get())[0]
    r = (v[:, 0].max() - v[:, 0].min()) / 2
    c = np.array([(v[:, 0].max() + v[:, 0].min()) / 2, (v[:, 1].max() + v[:, 1].min()) / 2, v[:, 2].min() + r])
    for _ in range(3):
        s = v[np.abs(np.linalg.norm(v - c, axis=1) - r) < 0.3]
        if len(s) < 4:
            break
        x = np.linalg.lstsq(np.c_[2 * s, np.ones(len(s))], (s ** 2).sum(1), rcond=None)[0]
        c, r = x[:3], math.sqrt(max(0.0, x[3] + x[:3] @ x[:3]))
    return c


def lens_camera(o):
    """World from the phone's camera as the dataset renders it (synth_twin.Twin.lens_camera), in the viewer's
    convention (x right, y up, looking along -z): centre of the lens disc, optical axis along its normal away from
    the phone, image x along the long side of the phone."""
    dg = bpy.context.evaluated_depsgraph_get()
    L, _ = world_mesh(o[LENS], dg)
    P, _ = world_mesh(o[PHONE], dg)
    c = L.mean(0)
    z = np.linalg.svd(L - c)[2][2]
    z = z if z @ (c - P.mean(0)) > 0 else -z
    x = np.linalg.svd(P - P.mean(0))[2][0]
    x -= z * (x @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    if y[2] > 0:
        x, y = -x, -y
    T = np.eye(4)
    T[:3, :3] = np.stack([x, -y, -z], 1)      # OpenCV (x, y down, z forward) to the viewer's
    T[:3, 3] = c
    return T


def camera_of(ob):
    sc = bpy.context.scene
    d = ob.data
    aspect = sc.render.resolution_x * sc.render.pixel_aspect_x / (sc.render.resolution_y * sc.render.pixel_aspect_y)
    fit = d.sensor_fit if d.sensor_fit != "AUTO" else ("HORIZONTAL" if aspect >= 1 else "VERTICAL")
    if fit == "HORIZONTAL":
        fov_y = 2 * math.atan(math.tan(d.angle_x / 2) / aspect)
    else:
        fov_y = d.angle_y
    m = mat(ob.matrix_world)
    m[:3, :3] /= np.linalg.norm(m[:3, :3], axis=0)
    return m, math.degrees(fov_y), aspect


def targets(o):
    """The TRE targets (role proposed) in the mandible frame and in the world, from the targets table."""
    if not TARGETS.exists() or MANDIBLE not in o:
        return [], None
    import csv
    with open(TARGETS, newline="", encoding="utf-8-sig") as f:
        rs = [r for r in csv.DictReader(f) if r.get("role", "proposed") == "proposed"]
    T = mat(o[MANDIBLE].matrix_world)
    out = []
    for r in rs:
        p = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
        w = T[:3, :3] @ p + T[:3, 3]
        out.append({"id": r["id"], "name": r.get("name", r["id"]), "side": r.get("side", ""),
                    "mandible": [round(float(x), 4) for x in p], "world": [round(float(x), 4) for x in w]})
    return out, T


def extras(frames):
    o = bpy.data.objects
    sc = bpy.context.scene
    ex = {"units": "mm", "up": [0, 0, 1], "parts": PARTS, "frames": {k: rows(v, 6) for k, v in frames.items()},
          "blender": bpy.app.version_string}
    arm = sc.get("sensor_arm")
    if arm is not None:
        a = arm.to_dict()
        ex["arm"] = {k: canon(a[k], 4) for k in ("arm_deg", "camera_deg", "shown_deg", "scanner_deg") if k in a}
    cams = {}
    A = np.linalg.inv(frames["arm"])
    if LENS in o and PHONE in o:
        cams["phone"] = {"node": "arm", "matrix": rows(A @ lens_camera(o), 6), "source": "lens"}
    if PHONE_CAM in o:
        m, fov, aspect = camera_of(o[PHONE_CAM])
        cams.setdefault("phone", {"node": "arm", "matrix": rows(A @ m, 6), "source": PHONE_CAM})
        cams["phone"].update(fov_y=round(fov, 3), aspect=round(aspect, 4))
    if SCANNER_CAM in o:
        m, fov, aspect = camera_of(o[SCANNER_CAM])
        cams["scanner"] = {"node": "arm", "matrix": rows(A @ m, 6), "fov_y": round(fov, 3), "aspect": round(aspect, 4),
                           "source": SCANNER_CAM}
    ex["cameras"] = cams
    for key in ("target", "centroid", "strike_point"):
        if sc.get(key) is not None:
            ex[key] = [round(float(x), 4) for x in sc[key]]
    if BOARD in o:
        v = world_mesh(o[BOARD], bpy.context.evaluated_depsgraph_get())[0]
        ex["board_centre"] = [round(float(x), 4) for x in (v.min(0) + v.max(0)) / 2]
    if BALL in o:
        ex["ball_centre"] = [round(float(x), 4) for x in ball_centre(o[BALL])]
    tg, T = targets(o)
    ex["targets"] = tg
    if T is not None:
        ex["T_W_mandible"] = rows(T, 6)
    ex["lamps"], ex["world"] = lamps(), world()
    rd = rig_design()                   # the six spheres and the board's frame, for ground_truth.py
    if rd is not None:
        ex["rig_design"] = rd
    cen = spheres_centroid()
    sp = scanner_poses(o, cen, ex.get("arm", {}))
    if sp:
        ex["scanner_poses"] = sp
        ex["spheres_centroid"] = [round(float(x), 4) for x in cen]
    return ex


def board_frame(ob):
    """T_W_M of the marker's board object: origin at the outer corner of the print's grid, x along its columns, y
    along its rows, z = x cross y (its texture's corners, as synth_twin.py has it; the grid of marker_pose.BOARD);
    None when its texture has no corners."""
    import ast
    import re
    text = (Path(__file__).with_name("marker_pose.py")).read_text(encoding="utf-8")
    spec = ast.literal_eval(re.search(r"^BOARD = (\{.*\})$", text, re.M).group(1))
    me, mw = ob.data, ob.matrix_world
    if me.uv_layers.active is None:
        return None
    uv = me.uv_layers.active.data
    # the faces that carry the print (an area in the texture), and the texture laid on them: world = (u, v, 1) A
    U, P = [], []
    for poly in me.polygons:
        q = [uv[i].uv for i in poly.loop_indices]
        area = 0.5 * abs(sum(q[i][0] * q[i - 1][1] - q[i - 1][0] * q[i][1] for i in range(len(q))))
        if area > 1e-8:
            for i in poly.loop_indices:
                U.append((uv[i].uv[0], uv[i].uv[1], 1.0))
                P.append(tuple(mw @ me.vertices[me.loops[i].vertex_index].co))
    if len(U) < 3:
        return None
    U, P = np.array(U), np.array(P)
    A = np.linalg.lstsq(U, P, rcond=None)[0]
    if np.sqrt(((U @ A - P) ** 2).sum(1).mean()) > 0.05:        # not one flat print
        return None
    du, dv, o = A
    tl = o + dv                                                 # the texture's corner (0, 1)
    x, y = du / np.linalg.norm(du), -dv / np.linalg.norm(dv)
    origin = tl + (x + y) * (np.linalg.norm(du) - spec["squares"] * spec["square_mm"]) / 2
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, np.cross(x, y)], 1)
    T[:3, 3] = origin
    return T


def rig_design():
    """The reference spheres as the scene has them, for ground_truth.py: the centre of each gt_sphere_* shown, by its
    name without the design's suffix (marker_1..3, mandible_1..3), and the frame M of the board beside the marker's
    spheres (board_frame), in the bench's world; None unless exactly those six are shown."""
    sp, coll = {}, {}
    for ob in present():
        if ob.type == "MESH" and ob.name.startswith("gt_sphere_"):
            base = ob.name[len("gt_sphere_"):].split("__")[0]
            if base in sp:
                return None             # two designs shown at once
            sp[base] = [round(float(x), 4) for x in ob.matrix_world.translation]
            coll[base] = {c.name for c in ob.users_collection}
    if sorted(sp) != [f"{r}_{k}" for r in ("mandible", "marker") for k in (1, 2, 3)]:
        return None
    out = {"spheres": sp}
    boards = [ob for ob in present() if ob.type == "MESH" and ob.name.startswith(BOARD)]
    boards.sort(key=lambda ob: not ({c.name for c in ob.users_collection} & coll["marker_1"]))
    T = board_frame(boards[0]) if boards else None
    if T is not None:
        out["board"], out["T_W_M"] = boards[0].name, rows(T, 6)
    return out


def spheres_centroid():
    """The centroid of the reference spheres the scanner sees (the gt_sphere_* objects shown in a render: the
    marker's and the mandible's), as the bench's script takes it; None without them."""
    c = [np.array(ob.matrix_world.translation) for ob in present()
         if ob.type == "MESH" and ob.name.startswith("gt_sphere")]
    return np.mean(c, 0) if c else None


def scanner_poses(o, cen, arm):
    """The scanner at each arm position as the scene has it (cam_scanner_<k>, on its wedge): where it is, where it
    looks (its optical axis, the camera's -z), its up (+y) and field (degrees), the arm position's camera azimuth;
    seen from the spheres' centroid (spheres_centroid), its azimuth (0 anterior, -y; positive towards the patient's
    left, +x), elevation and distance. World coordinates, mm."""
    cam_deg = list(arm.get("camera_deg") or [])
    out = []
    for k in range(1, 9):
        ob = o.get(f"cam_scanner_{k}")
        if ob is None:
            continue
        m, fov_y, aspect = camera_of(ob)
        org, axis, up = m[:3, 3], -m[:3, 2], m[:3, 1]
        fov_x = math.degrees(2 * math.atan(math.tan(math.radians(fov_y) / 2) * aspect))
        row = {"camera": ob.name, "position": k, "origin": [round(float(x), 3) for x in org],
               "direction": [round(float(x), 6) for x in axis], "up": [round(float(x), 6) for x in up],
               "fov_deg": [round(fov_x, 3), round(fov_y, 3)]}
        if k <= len(cam_deg):
            row["camera_deg"] = round(float(cam_deg[k - 1]), 3)
        if cen is not None:
            rel = org - cen
            row["azimuth_deg"] = round(math.degrees(math.atan2(rel[0], -rel[1])), 3)
            row["elevation_deg"] = round(math.degrees(math.atan2(rel[2], math.hypot(rel[0], rel[1]))), 3)
            row["distance_mm"] = round(float(np.linalg.norm(rel)), 3)
        out.append(row)
    return out


def lamps():
    """The lamps of the scene: kind, colour, strength (W/m2 for a sun), angle, the direction the light goes (the
    lamp's -z) and where it is."""
    out = []
    for ob in present():
        if ob.type != "LIGHT":
            continue
        d, m = ob.data, mat(ob.matrix_world)
        z = -m[:3, 2] / np.linalg.norm(m[:3, 2])
        out.append({"name": ob.name, "type": d.type, "color": [round(float(c), 4) for c in d.color],
                    "energy": round(float(d.energy), 4), "angle": round(float(getattr(d, "angle", 0.0)), 4),
                    "dir": [round(float(x), 5) for x in z], "pos": [round(float(x), 3) for x in m[:3, 3]]})
    return out


def world():
    """The world: the light it gives (colour times strength, linear) and the colour the camera sees, when a Light Path
    node lets the two differ (Is Camera Ray into the factor of a Mix Shader of two backgrounds)."""
    w = bpy.context.scene.world
    out = {"light": [0.05, 0.05, 0.05], "camera": [0.05, 0.05, 0.05]}
    if w is None:
        return out
    if w.node_tree is None:
        out["light"] = out["camera"] = [round(float(c), 4) for c in w.color]
        return out

    def bg(n):
        c, st = n.inputs["Color"].default_value, n.inputs["Strength"].default_value
        return [round(float(x) * float(st), 4) for x in list(c)[:3]]
    nodes = w.node_tree.nodes
    mix = next((n for n in nodes if n.type == "MIX_SHADER" and n.inputs[0].is_linked
                and n.inputs[0].links[0].from_socket.name == "Is Camera Ray"), None)
    if mix is not None:
        a, b = (mix.inputs[k].links[0].from_node if mix.inputs[k].is_linked else None for k in (1, 2))
        if a is not None and a.type == "BACKGROUND":
            out["light"] = bg(a)
        if b is not None and b.type == "BACKGROUND":
            out["camera"] = bg(b)
        return out
    first = next((n for n in nodes if n.type == "BACKGROUND"), None)
    if first is not None:
        out["light"] = out["camera"] = bg(first)
    return out


# ---------------------------------------------------------------- GLB

def chunks(p, n, t, owner, most=MAX_VERTS):
    """A primitive cut into pieces of at most `most` vertices, its triangles in their order: [(p, n, t, owner)] each
    with its own vertices (owner: the object of each triangle)."""
    if len(p) <= most:
        return [(p, n, t, owner)]
    out, start = [], 0
    while start < len(t):
        lo, hi = 1, len(t) - start          # the most triangles from start whose vertices fit
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(np.unique(t[start:start + mid])) <= most:
                lo = mid
            else:
                hi = mid - 1
        part = t[start:start + lo]
        used, inv = np.unique(part, return_inverse=True)
        out.append((p[used], n[used], inv.reshape(-1, 3), owner[start:start + lo]))
        start += lo
    return out


def runs(owner, names):
    """The objects of a primitive as [name, first triangle, triangles], for the page to pick and outline a part."""
    if not len(owner):
        return []
    cut = np.flatnonzero(np.diff(owner)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(owner)]])
    return [[names[int(owner[a])], int(a), int(b - a)] for a, b in zip(starts, ends)]


def quantized(p):
    """Positions as 16-bit integers over their box, the same step on the three axes: (q, centre, half size);
    p = centre + q / 32767 * half. One scale for the three axes lets the page draw them as they are (the mesh scaled
    by half / 32767 and moved to the centre), the normals unchanged by it."""
    lo, hi = p.min(0).astype(np.float64), p.max(0).astype(np.float64)
    c, h = (lo + hi) / 2, np.full(3, max(float(((hi - lo) / 2).max()), 1e-6))
    q = np.clip(np.round((p - c) / h * 32767), -32767, 32767).astype(np.int16)
    return q, c, h


def drape_mesh():
    """A drape as the dataset has them (WORK/synth/drapes/drape_00.npz: a cloth simulated over the face, mm, world):
    (positions, normals, triangles), or None when there is none."""
    path = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact")) / "synth" / "drapes" / "drape_00.npz"
    if not path.exists():
        return None
    d = np.load(path)
    v, f = d["vertices"].astype(np.float64), d["faces"].astype(np.int64)
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return v.astype(np.float32), n.astype(np.float32), f


def write_glb(path, nodes, materials, ex):
    """nodes: [(name, matrix 4 x 4 or None, {material: (positions, normals, triangles)}, children names)]. Returns
    (triangles, vertices, the largest error of a quantized position in mm, bytes before compression)."""
    doc = {"asset": {"version": "2.0", "generator": "ImpactGuide twin", "extras": {"compression": "zlib"}},
           "extensionsUsed": ["KHR_mesh_quantization", "KHR_materials_emissive_strength"],
           "extensionsRequired": ["KHR_mesh_quantization"], "scene": 0, "nodes": [], "meshes": [],
           "materials": [], "accessors": [], "bufferViews": [], "buffers": []}
    blob = bytearray()
    mat_index = {}

    def view(data, target, stride=None):
        while len(blob) % 4:
            blob.append(0)
        bv = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target}
        if stride:
            bv["byteStride"] = stride
        doc["bufferViews"].append(bv)
        blob.extend(data)
        return len(doc["bufferViews"]) - 1

    def accessor(a, ctype, kind, target, normalized=False, bounds=False):
        """a: (count, 4) for a VEC3 padded to 4 components (8 or 4 bytes a vertex, as KHR_mesh_quantization
        aligns them), else the indices."""
        stride = a.shape[1] * a.itemsize if kind == "VEC3" else None
        acc = {"bufferView": view(a.tobytes(), target, stride), "componentType": ctype,
               "count": int(len(a) if kind != "SCALAR" else a.size), "type": kind}
        if normalized:
            acc["normalized"] = True
        if bounds:
            acc["min"] = [int(x) for x in a[:, :3].min(0)]
            acc["max"] = [int(x) for x in a[:, :3].max(0)]
        doc["accessors"].append(acc)
        return len(doc["accessors"]) - 1

    for name in sorted(materials):
        v = materials[name]
        col, emit = v["color"], v["emission"]
        mat_index[name] = len(doc["materials"])
        m = {"name": name or "plain", "doubleSided": True,
             "pbrMetallicRoughness": {"baseColorFactor": [round(float(c), 5) for c in col],
                                      "metallicFactor": round(v["metallic"], 4),
                                      "roughnessFactor": round(v["roughness"], 4)}}
        top = max(emit)
        if top > 0:
            m["emissiveFactor"] = [round(c / max(top, 1.0), 5) for c in emit]
            if top > 1:
                m["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": round(top, 4)}}
        if col[3] < 1:
            m["alphaMode"] = "BLEND"
        if v.get("sheen"):
            m["extras"] = {"sheen": v["sheen"]}         # the cloth's sheen (the cap): weight times tint
        doc["materials"].append(m)
    index = {n[0]: i for i, n in enumerate(nodes)}
    tris = verts = 0
    err = 0.0
    for name, M, parts, children in nodes:
        nd = {"name": name}
        if M is not None and not np.allclose(M, np.eye(4)):
            nd["matrix"] = [float(x) for x in np.asarray(M).T.ravel()]     # column-major
        prims = []
        for mname, whole in sorted(parts.items()):
            if not len(whole[2]):
                continue
            for p, n, t, owner in chunks(*whole[:4]):
                q, c, h = quantized(p.astype(np.float64))
                err = max(err, float(np.abs(c + q / 32767.0 * h - p).max()))
                qn = np.clip(np.round(n * 127), -127, 127).astype(np.int8)
                qn = np.concatenate([qn, np.zeros((len(qn), 1), np.int8)], 1)   # 4 bytes a normal: aligned
                qp = np.concatenate([q, np.zeros((len(q), 1), np.int16)], 1)       # 8 bytes a position
                prims.append({"attributes": {"POSITION": accessor(qp, 5122, "VEC3", 34962, True, True),
                                             "NORMAL": accessor(qn, 5120, "VEC3", 34962, True)},
                              "indices": accessor(t.astype(np.uint16).ravel(), 5123, "SCALAR", 34963),
                              "material": mat_index[mname], "mode": 4,
                              "extras": {"centre": [float(x) for x in c], "half": [float(x) for x in h],
                                         "objects": runs(owner, whole[4])}})
                tris += len(t)
                verts += len(p)
        if prims:
            nd["mesh"] = len(doc["meshes"])
            doc["meshes"].append({"name": name, "primitives": prims})
        if children:
            nd["children"] = [index[c] for c in children]
        doc["nodes"].append(nd)
    roots = [i for i, n in enumerate(nodes) if not any(n[0] in m[3] for m in nodes)]
    doc["scenes"] = [{"name": "bench", "nodes": roots, "extras": ex}]
    while len(blob) % 4:
        blob.append(0)
    raw = len(blob)
    packed = zlib.compress(bytes(blob), ZLIB_LEVEL)
    doc["buffers"].append({"byteLength": raw})
    doc["asset"]["extras"]["packedLength"] = len(packed)     # the zlib stream alone, without the chunk's padding
    packed += b"\0" * (-len(packed) % 4)
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(packed)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(packed), 0x004E4942))
        f.write(packed)
    return tris, verts, err, raw


def export(out):
    """The model of the open scene into out/bench.glb, its extras into out/bench.json; returns the extras."""
    t0 = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print_, items = fingerprint()           # before anything is added to the scene
    t_print = time.perf_counter() - t0
    every = [ob for ob in present() if ob.type in ("MESH", "CURVE", "SURFACE", "FONT")]
    coarse_curves(every)
    # what follows the arm or the pendulum without being carried by it: a mesh of it at each of their poses
    posed = {}
    for kind in ("arm", "pendulum"):
        done = {ob.name for v in posed.values() for ob in v}
        posed[kind] = [ob for ob in followers(every, kind) if ob.name not in done]
    following = {ob.name for v in posed.values() for ob in v}
    pose_parts, materials = {}, {}
    for kind, obs in posed.items():
        for k, value in enumerate(poses(kind) if obs else []):
            set_pose(kind, value)
            dg = bpy.context.evaluated_depsgraph_get()
            node = f"{kind}_pose_{k}"
            PARTS[node] = {"pose": kind, "index": k, "value": round(float(value), 4)}
            pose_parts[node] = {}
            for ob in obs:
                for mname, piece in mesh_parts(ob, dg, np.eye(4)).items():
                    if mname not in materials:
                        materials[mname] = material_values(bpy.data.materials.get(mname) if mname else None)
                    pose_parts[node].setdefault(mname, []).append((*piece, ob.name))
        set_pose(kind, None)
    objs = meshed([ob for ob in every if ob.name not in following])
    reduced = reduce_organic(objs)
    frames = node_frames()
    inv = {k: np.linalg.inv(v) for k, v in frames.items()}
    coll_a, coll_b = in_collection([HOLDERS["A"]]), in_collection([HOLDERS["B"]])
    dg = bpy.context.evaluated_depsgraph_get()
    acc = {k: {} for k in PARTS}
    acc.update(pose_parts)
    for ob in objs:
        node = node_of(ob, coll_a, coll_b)
        for mname, (p, n, t) in mesh_parts(ob, dg, inv[node]).items():
            if mname not in materials:
                materials[mname] = material_values(bpy.data.materials.get(mname) if mname else None)
            acc[node].setdefault(mname, []).append((p, n, t, ob.name))
    merged = {}
    for node, by_mat in acc.items():
        merged[node] = {}
        for mname, pieces in by_mat.items():
            off, P, N, Tr, O, names = 0, [], [], [], [], []
            for p, n, t, name in pieces:
                P.append(p)
                N.append(n)
                Tr.append(t + off)
                O.append(np.full(len(t), len(names), np.int32))
                names.append(name)
                off += len(p)
            merged[node][mname] = (np.concatenate(P), np.concatenate(N), np.concatenate(Tr), np.concatenate(O), names)
    drape = drape_mesh()
    if drape is not None:
        PARTS["drape"] = {"drape": True}
        materials["drape"] = {"color": [*DRAPE_LINEAR, 1.0], "metallic": 0.0, "roughness": 0.8, "emission": [0.0] * 3,
                              "sheen": [0.3, 0.3, 0.3]}
        merged["drape"] = {"drape": (*drape, np.zeros(len(drape[2]), np.int32), ["drape"])}
    order = ["bench", "marker", "marker_a", "marker_b", "holder_a", "holder_a_arm", "holder_b", "holder_b_plate",
             "pendulum", "pendulum_swing", "arm"] + sorted(pose_parts) + (["drape"] if drape is not None else [])
    nodes = []
    for name in order:
        M = frames.get(name, np.eye(4))
        if PARTS[name].get("parent"):
            M = inv[PARTS[name]["parent"]] @ M
        nodes.append((name, M, merged[name], ["pendulum_swing"] if name == "pendulum" else []))
    ex = extras(frames)
    pts = [(frames.get(n, np.eye(4))[:3, :3] @ m[0].T).T + frames.get(n, np.eye(4))[:3, 3]
           for n in merged for m in merged[n].values()]
    allp = np.concatenate(pts) if pts else np.zeros((1, 3))
    ex["bounds"] = [[round(float(x), 2) for x in allp.min(0)], [round(float(x), 2) for x in allp.max(0)]]
    ex["phone_print"] = print_
    ex["print_items"] = items
    ex["reduced"] = dict(sorted(reduced.items()))
    ex["posed"] = {k: sorted(ob.name for ob in v) for k, v in posed.items()}
    ex["kept_whole"] = len(objs) - sum(1 for v in reduced.values() if v["kept"])
    tris, verts, err, raw = write_glb(out / "bench.glb", nodes, materials, ex)
    ex.update(triangles=tris, vertices=verts, bytes=(out / "bench.glb").stat().st_size, raw_bytes=raw,
              quant_mm=round(err, 5), export_s=round(time.perf_counter() - t0, 2), print_s=round(t_print, 2))
    (out / "bench.json").write_text(json.dumps(ex, indent=1), encoding="utf-8")
    return ex


# ---------------------------------------------------------------- test

def read_glb(path):
    """The JSON of a model and its binary chunk, inflated."""
    b = Path(path).read_bytes()
    n = struct.unpack_from("<I", b, 12)[0]
    doc = json.loads(b[20:20 + n])
    m = struct.unpack_from("<I", b, 20 + n)[0]
    packed = b[28 + n:28 + n + doc["asset"]["extras"]["packedLength"]]
    return doc, zlib.decompress(packed)


def positions(doc, binary, prim):
    """The positions of a primitive back in mm (its node's frame)."""
    acc = doc["accessors"][prim["attributes"]["POSITION"]]
    v = doc["bufferViews"][acc["bufferView"]]
    q = np.frombuffer(binary, np.int16, acc["count"] * 4, v["byteOffset"]).reshape(-1, 4)[:, :3]
    return np.array(prim["extras"]["centre"]) + q / 32767.0 * np.array(prim["extras"]["half"])


def selftest():
    """A small bench built in memory: the fingerprint follows what the phone sees and only that; the model has the
    nodes the page moves, one primitive per material, an organic mesh reduced with its shape kept and the others
    whole, positions read back within the quantization step, lamps and world in the extras."""
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    root = sc.collection

    def material(name, rgb):
        m = bpy.data.materials.new(name)
        if m.node_tree is None:
            m.use_nodes = True
        b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        b.inputs["Base Color"].default_value = (*rgb, 1.0)
        m.diffuse_color = (*rgb, 1.0)
        return m

    def mesh(name, parent=None, coll=root, mat_=None, loc=(0, 0, 0), subdiv=1):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=10, location=loc)
        ob = bpy.context.active_object
        ob.name = name
        for c in ob.users_collection:
            c.objects.unlink(ob)
        coll.objects.link(ob)
        if parent is not None:
            ob.parent = parent
        if mat_ is not None:
            ob.data.materials.append(mat_)
        return ob

    def empty(name, parent=None, loc=(0, 0, 0)):
        ob = bpy.data.objects.new(name, None)
        ob.location = loc
        root.objects.link(ob)
        ob.parent = parent
        return ob

    skin, green, steel, black = (material("silicone_face", (0.9, 0.3, 0.16)), material("petg_green", (0.1, 0.36, 0.08)),
                                 material("steel", (0.46, 0.48, 0.5)), material("cable", (0.01, 0.01, 0.01)))
    ca, cb = bpy.data.collections.new(HOLDERS["A"]), bpy.data.collections.new(HOLDERS["B"])
    root.children.link(ca)
    root.children.link(cb)
    face = mesh(FACE, mat_=skin, subdiv=7)      # 81920 triangles: over its share of the test
    splint = mesh("splint", mat_=green, loc=(0, -40, 0), subdiv=6)      # heavy, but not of REDUCE in the test
    motion = empty(MARKER, splint, (0, -10, 0))
    mesh(BOARD, motion, mat_=steel, loc=(0, -60, 0))
    mesh("a1_ball_stud", motion, ca, steel, (0, -55, 5))
    mesh("a1_arm", splint, ca, steel, (0, -45, 5))
    mesh("a1_cup", splint, ca, steel, (0, -45, 8))
    mesh("b1_balls", motion, cb, steel, (0, -55, -5))
    mesh("b1_plate", splint, cb, green, (0, -45, -5))
    base = empty(PEND_BASE, None, (0, -80, -250))
    swing = empty(PEND_SWING, base, (0, -40, 590))
    mesh("pendulum_arm", swing, mat_=steel, loc=(0, -120, 100))
    pivot = empty(ARM, None, (0, -80, -250))
    mesh(PHONE, pivot, mat_=black, loc=(200, -260, 60))
    cable = mesh("cable_phone", mat_=black, loc=(150, 0, -200))
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 500))
    bpy.context.active_object.name = "sun_key"
    sc["sensor_arm"] = {"arm_deg": [36.0, 51.0, 66.0, 80.0], "camera_deg": [40.0, 55.0, 70.0, 85.0], "shown_deg": 51.0}
    bpy.context.view_layer.update()

    p0, items0 = fingerprint()
    expect(len(p0) == 16 and FACE in items0 and "sun_key" in items0 and "cable_phone" not in items0
           and "pendulum_arm" not in items0, f"fingerprint {p0}: face, lamp and arm in it, cable and pendulum not")
    cable.location.x += 30
    black.diffuse_color = (0.2, 0.2, 0.2, 1)
    bpy.context.view_layer.update()
    expect(fingerprint()[0] == p0, "a cable moved and its colour changed: the same fingerprint")
    face.data.vertices[0].co.x += 0.5
    bpy.context.view_layer.update()
    p1 = fingerprint()[0]
    expect(p1 != p0, "a vertex of the face moved by 0.5 mm: a new fingerprint")
    bpy.data.objects["sun_key"].data.energy *= 1.5
    p2 = fingerprint()[0]
    expect(p2 != p1, "a lamp brighter: a new fingerprint")
    bsdf = next(n for n in skin.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.8, 0.3, 0.2, 1)
    p3 = fingerprint()[0]
    expect(p3 != p2, "the colour of the face changed: a new fingerprint")
    sc["sensor_arm"] = {"arm_deg": [36.0, 51.0, 66.0, 81.0], "camera_deg": [40.0, 55.0, 70.0, 85.0], "shown_deg": 51.0}
    expect(fingerprint()[0] != p3, "an arm position changed: a new fingerprint")
    p4 = fingerprint()[0]
    expect(fingerprint()[0] == p4, "the same scene twice: the same fingerprint")

    out = Path(os.environ.get("TEMP", ".")) / "twin_export_test"
    REDUCE[FACE] = 30000                        # the test's face is lighter than the bench's

    def blender_positions():
        """World vertices of a few small meshes, in the frame of their node (marker: world; arm: the pivot)."""
        dg = bpy.context.evaluated_depsgraph_get()
        res = {}
        for obn, node, frame in ((BOARD, "marker", np.eye(4)), (PHONE, "arm", mat(bpy.data.objects[ARM].matrix_world))):
            v, _ = world_mesh(bpy.data.objects[obn], dg)
            res[node] = (v - frame[:3, 3]) @ frame[:3, :3]
        return res
    ex = export(out)
    doc, binary = read_glb(out / "bench.glb")
    names = [n["name"] for n in doc["nodes"]]
    want = {"bench", "marker", "marker_a", "marker_b", "holder_a", "holder_a_arm", "holder_b", "holder_b_plate",
            "pendulum", "pendulum_swing", "arm"}
    expect(want <= set(names), f"model: the nodes the page moves ({sorted(names)})")
    pend = doc["nodes"][names.index("pendulum")]
    expect(pend.get("children") == [names.index("pendulum_swing")], "model: the swing is a child of the pendulum")
    by = {n["name"]: n for n in doc["nodes"]}
    prims = doc["meshes"][by["marker"]["mesh"]]["primitives"]
    expect(len(prims) == 1 and doc["materials"][prims[0]["material"]]["name"] == "steel",
           "model: the marker merged into one primitive of its one material")
    expect("mesh" in by["holder_a"] and "mesh" in by["marker_a"] and "mesh" in by["holder_b_plate"],
           "model: each holder's parts in their nodes")
    ch = ex["reduced"].get(FACE, {})
    expect(ch.get("kept") and ch["tris"][1] <= 30000 < ch["tris"][0] and ch["volume"] <= SHAPE_VOLUME
           and ch["box_mm"] <= SHAPE_BOX_MM,
           f"model: the organic face reduced gently, its shape kept ({ch})")
    expect("splint" not in ex["reduced"] and "pendulum_arm" not in ex["reduced"],
           "model: the machined parts left whole")
    bv, ok = blender_positions(), True
    by = {n["name"]: n for n in doc["nodes"]}
    for name, want in bv.items():
        got = np.concatenate([positions(doc, binary, pr) for pr in doc["meshes"][by[name]["mesh"]]["primitives"]])
        # every vertex of Blender found in the model within the quantization step
        d = np.min(np.linalg.norm(want[:, None, :] - got[None, :, :], axis=2), axis=1).max()
        ok = ok and d <= max(ex["quant_mm"], 1e-6) * 2
    expect(ok and ex["quant_mm"] < 0.01, f"model: positions read back within {ex['quant_mm']} mm of Blender's")
    expect(ex["raw_bytes"] > ex["bytes"] and any(lmp["name"] == "sun_key" for lmp in ex["lamps"])
           and "light" in ex["world"], f"model: compressed ({ex['bytes']} of {ex['raw_bytes']} bytes), "
                                       f"lamps and world in the extras")
    expect(ex["phone_print"] == p4, "model: it carries the fingerprint of the scene it was made from")
    expect(ex["arm"]["camera_deg"] == [40.0, 55.0, 70.0, 85.0], "model: the arm positions in its extras")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


def main(argv):
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    cmd = args[0] if args else "print"
    if cmd == "export":
        ex = export(args[1])
        print(json.dumps({k: ex[k] for k in ("phone_print", "triangles", "vertices", "bytes", "export_s", "print_s")}))
    elif cmd == "print":
        p, items = fingerprint()
        print(json.dumps({"phone_print": p, "items": items}))
    elif cmd == "test":
        sys.exit(selftest())
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv)
