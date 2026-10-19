"""Digital twin of the test bench: synthetic phone photos and video frames of the face with exact masks and true poses.
Replaces render_phone_views.py, which rebuilt the old scene.

Runs in Blender 5.2 in the background on a copy of the bench scene, never on the original (it is edited elsewhere):
    blender --background --factory-startup --python-exit-code 1 --python 04_code/src/synth_twin.py -- COMMAND [options]
The copy OUT/twin_scene.blend is refreshed from SCENE when SCENE has not been saved for 2 minutes; nothing is ever
saved. Objects are found by name (02_cad/solidworks/part_names.md and the constants below), so the generator reruns
unchanged when the scene changes.

Commands
    copy     only refresh the copy of the scene
    probe    checks of the scene: phone camera from the lens against cam_phone_2, tilt hinge, A1 ball centre, board
    test     checks of the marker motion on both interfaces: the board moves by exactly the sampled transform, in the
             scene, in the label images and in the labels; exit code 1 on a failure (nothing is written to the dataset)
    meshes   face meshes in B for the renderer (full and decimated, with the decimation error), occluders, weights of
             the skin deformation
    drapes   drape shapes by cloth simulation, kept when they cover 10-40% of the face from the photo positions and
             stay clear of marker, spheres and pendulum
    s1       ray casting of the head spheres from every phone position and along the video path (they must stay out
             of sight of the phone), and candidate positions on the bone hidden from the phone and seen by the scanner
    render   the dataset: sessions x registrations x states x (4 photos + video frames); resumable, one state at a time

Head spheres. The head spheres and their frame (gt_sphere_mandible_*, gt_frame_mandible*) are left out of the images
and of the drape simulation unless --head-spheres is given: on the bench they are in sight of the phone, and a
network trained on the twin must never learn to use them. session.json and every frame record the choice.

Frames. B = the head frame: coordinates of the bench scene, mm (not the local frame of the mandible mesh, which
ground_truth.py calls "mandible"; T_B_mandible is in 03_data/processed/targets/mandible_frame.json). Camera C as in
OpenCV (x right, y down, z forward, pixel centres at integer coordinates); board M as cv2.aruco.CharucoBoard (origin at
the outer corner of the first square, x along the printed columns, y along the rows, z into the board). T_C_B camera
from B, T_M_B board from B (registration), T_C_M = T_C_B . T_M_B^-1. Impact: T_M_B = T_M_B0 . dT, dT in B. A1:
dT = [R, (I - R) c] about the ball centre c; general (B1): X' = R (X - o) + o + s about the face centroid o (as
correct.py). Each registration starts from the scene pose moved by Q (A1: a rotation about c; B1: Q = I), and the
impact moves the marker in the scene by dT^-1: world from board = dT^-1 . Q . (world from board in the scene).
Images are pinhole renders (dist = 0) with the nominal intrinsics of the Poco X7 Pro, or the calibrated ones of
03_data/processed/calibration/intrinsics_{photo,video}.json when they exist (their distortion is not rendered),
scaled by --scale.

Output in OUT (default WORK/synth; WORK from the environment variable IMPACTGUIDE_WORK), regenerable from this
script and the seed:
    twin_scene.blend                             copy of the bench scene
    meshes/face_<id>_full.npz, face_<id>_r20k.npz   face in B (vertices mm, triangles), decimation error in the json
    meshes/occluders.npz, meshes/face_weights.npz
    drapes/drape_NN.npz                          drape shapes in B
    s1.json                                      visibility of the head spheres, candidate positions
    sessions/sNNNN/session.json                  face, split, pendulum direction, constants, scene_print (the
                                                 fingerprint of what the phone sees, twin_export.py)
    sessions/sNNNN/frames.jsonl                  one line per image: poses, intrinsics, parameters, label pixel counts,
                                                 shift of the TRE targets under dT (error of the stale registration)
    sessions/sNNNN/<image>.jpg                   RGB, JPEG 95
    sessions/sNNNN/<image>_label.png             labels (LABELS), 8 bit grey, exact: one sample at each pixel centre
    sessions/sNNNN/<image>_model.png             with --model-mask: labels with the undeformed face alone
    sessions/sNNNN/<image>_depth.png             with --depth: z depth, 16 bit, 1/16 mm, 65535 where nothing is hit
Image names: r<registration>_s<state>_<view>, view p1-p4 (photo positions) or v00-v15 (video frames). State s0 of
every registration is before the impact (dT = I).

Typical runs (small batches: the GPU is shared; each run reopens the copy, about 10 s):
    ... -- meshes ; ... -- drapes --count 24 ; ... -- s1
    ... -- render --sessions 0:10 [--registrations 4 --states 3 --video 16]     (dataset: sessions below 900)
    ... -- render --sessions 900:901 --registrations 2 --states 2 --video 2 --model-mask   (check set, synth_data.py)
"""
import argparse
import ast
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]
# Blender's Python ignores PYTHONPYCACHEPREFIX: apply it here, and write no bytecode next to the sources without it
if os.environ.get("PYTHONPYCACHEPREFIX"):
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
else:
    sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC))
import face_model as fm  # noqa: E402

WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
SCENE = WORK / "test_bench_draft.blend"
OUT = WORK / "synth"
CALIB = ROOT / "03_data" / "processed" / "calibration"
STABLE_S = 120          # the original is copied only when not saved for this long
SCALE = 0.5             # rendered resolution / camera resolution
SAMPLES = 16            # EEVEE samples of the RGB images
DECIMATED = 20000       # triangles of the face for the renderer

# Poco X7 Pro main camera, nominal until it is calibrated: 12.5 MP photo (4:3) and 4K video (16:9 from the full sensor
# width), 26 mm equivalent focal length, about 3070 px at 4080 px as in calibrate_charuco.py, principal point at the
# centre
NOMINAL = {"photo": ((4080, 3072), 3070.0), "video": ((3840, 2160), 3070.0 * 3840 / 4080)}

# names in the scene
FACE, SPLINT, BOARD, BALL, MOTION = "silicone_face", "splint", "marker_charuco", "a1_ball_stud", "marker_motion"
LENS, PHONE, PIVOT, TILT = "smartphone_lens", "smartphone", "sensor_arm_pivot", "phone_tilt"
PHONE_CAM, SCANNER_CAM = "cam_phone_2", "cam_scanner_2"      # cameras of the shown arm position (probe, s1)
SKULL, MANDIBLE, PENDULUM_BASE = "skull", "mandible", "pendulum_base"
VARIANTS = {"a1": ("a1_ball_stud", ["a1_arm"]), "b1": ("b1_kinematic_spring", ["b1_plate"])}  # collection, + moving
PANELS = ("light_panel_1_diffuser", "light_panel_2_diffuser")
ROOM = ("sun_key", "sun_fill")
HIDDEN = ("smartphone", "phone_cradle", "phone_usb_plug", "face_screws", "tmp_cam", "marker_ghost")  # prefixes
HEAD_SPHERES = "gt_sphere_mandible_"
HEAD_REF = (HEAD_SPHERES, "gt_frame_mandible")   # head spheres and their frame: hidden unless --head-spheres
CAP = "surgical_cap"    # green surgical cap on the head (prefix), when the scene has it
CAP_RAY_MM = 15.0       # skin is under the cap when the cap is within this distance along the vertex normal
DRAPE = "synth_drape"
DRAPE_CLEAR = ("marker_", "gt_sphere", "gt_frame", "a1_", "b1_")   # the drape keeps 5 mm from these
LABELS = {"background": 0, "face": 1, "marker": 2, "drape": 3, "interface": 4, "head_spheres": 5, "head": 6,
          "pendulum": 7, "bench": 8, "cap": 9}
TARGETS = ROOT / "03_data" / "processed" / "targets" / "mandible_targets.csv"   # TRE targets (role "proposed")
TEST_SCALE = 0.25       # resolution of the label images of the test command
TEST_DEG, TEST_MM = 4.0, 3.0      # displacement of the test command
TEST_TOL_MM = 1e-3      # the board must land within this distance of the sampled pose

# sampling
REG_DEG = 12.0          # A1 start rotated about c up to this angle: many marker-head registrations
B1_SHARE = 0.25         # registrations on B1 (general displacements), the others on A1
MAX_DEG, MAX_MM = 5.0, 5.0
FINE = 0.3              # share of displacements up to a fifth of the maximum (fine steps)
DRAPE_SHARE = 0.25
DRAPE_AMP_MM = 2.0      # skin deformation capped under a drape (the shapes are simulated on the phantom)
ARM_NOISE_DEG, TILT_NOISE_DEG, TILT_MAX = 0.1, 0.3, 12.0
PENDULUM_DEG = (0.0, 45.0, 90.0)
PANEL_W = 0.6e6         # one BP66 at 100% (the scene is in mm: power scales with distance squared)
SPLITS = (("train", 0.7), ("val", 0.15), ("test", 0.15))
PHANTOM_SHARE = 0.3     # sessions with the phantom face when the bank has other faces
SILICONE_SRGB = (242, 152, 112)
SKIN_SRGB = ("f6ede4", "f3e7db", "f7ead0", "eadaba", "d7bd96", "a07e56", "825c43", "604134", "3a312a", "292420")
DRAPE_SRGB = (38, 104, 112)


# ---------------------------------------------------------------- helpers

def arr(coll, attr, width, dtype=np.float64):
    a = np.empty(len(coll) * width, dtype)
    coll.foreach_get(attr, a)
    return a.reshape(-1, width) if width > 1 else a


def mat(m):
    return np.array(m, np.float64)


def lst(m, nd=6):
    return np.round(np.asarray(m, np.float64), nd).tolist()


def srgb_to_linear(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def kelvin_rgb(k):
    """Linear RGB of a black body at k kelvin, green normalised to 1 (Tanner Helland's fit of the CIE data)."""
    t = k / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    rgb = srgb_to_linear(np.clip([r, g, b], 0, 255))
    return (rgb / rgb[1]).tolist()


def rotation(axis, deg):
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = math.radians(deg)
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def about(R, centre, shift=(0.0, 0.0, 0.0)):
    """X' = R (X - c) + c + s as a 4 x 4 matrix."""
    c = np.asarray(centre, np.float64)
    return rigid(R, c - R @ c + np.asarray(shift, np.float64))


def unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def amplitude(rng, top):
    return rng.uniform(0.0, top / 5.0 if rng.random() < FINE else top)


def fibonacci(n):
    i = np.arange(n) + 0.5
    phi, z = np.pi * (1 + 5 ** 0.5) * i, 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def board_spec():
    """Board parameters from marker_pose.BOARD (single source)."""
    text = (SRC / "marker_pose.py").read_text(encoding="utf-8")
    return ast.literal_eval(re.search(r"^BOARD = (\{.*\})$", text, re.M).group(1))


def world_mesh(ob, evaluated=True):
    """World vertices and triangles of a mesh object (modifiers applied)."""
    src = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()) if evaluated else ob
    me = src.to_mesh()
    me.calc_loop_triangles()
    co = arr(me.vertices, "co", 3)
    tri = arr(me.loop_triangles, "vertices", 3, np.int32)
    src.to_mesh_clear()
    mw = mat(ob.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3], tri


def parent_depth(ob):
    """Number of parents above an object (used to set parents before their children)."""
    n = 0
    while ob.parent is not None:
        ob, n = ob.parent, n + 1
    return n


def sphere_of(ob):
    """Centre and radius of a sphere object from its vertices."""
    v, _ = world_mesh(ob, evaluated=False)
    c = (v.min(0) + v.max(0)) / 2
    return c, float(np.linalg.norm(v - c, axis=1).mean())


def ball_centre(ob):
    """Centre of the A1 ball: least squares sphere on the vertices at the radius of the widest section."""
    v, _ = world_mesh(ob, evaluated=False)
    r = (v[:, 0].max() - v[:, 0].min()) / 2
    c = np.array([(v[:, 0].max() + v[:, 0].min()) / 2, (v[:, 1].max() + v[:, 1].min()) / 2, v[:, 2].min() + r])
    for _ in range(3):
        s = v[np.abs(np.linalg.norm(v - c, axis=1) - r) < 0.3]
        x = np.linalg.lstsq(np.c_[2 * s, np.ones(len(s))], (s ** 2).sum(1), rcond=None)[0]
        c, r = x[:3], math.sqrt(x[3] + x[:3] @ x[:3])
    return c, r


def refresh_copy(src, dst, force=False):
    """Copy the scene when the original has not been saved for STABLE_S seconds and differs from the copy."""
    age = time.time() - src.stat().st_mtime
    if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
        return "up to date"
    if age < STABLE_S:
        if dst.exists():
            return f"original saved {age:.0f} s ago, kept the old copy"
        raise SystemExit(f"{src} saved {age:.0f} s ago: wait until it is stable")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def intrinsics(mode, scale):
    """K and size at the rendered resolution; calibrated values when the calibration has written them."""
    path = CALIB / f"intrinsics_{mode}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        (w, h), K, src = d["image_size"], np.array(d["K"], np.float64), path.name
    else:
        (w, h), f = NOMINAL[mode]
        K, src = np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1]]), "nominal"
    f = (K[0, 0] + K[1, 1]) / 2     # square pixels in Blender
    W, H = round(w * scale), round(h * scale)
    sx, sy = W / w, H / h
    Ks = np.array([[f * sx, 0, (K[0, 2] + 0.5) * sx - 0.5], [0, f * sy, (K[1, 2] + 0.5) * sy - 0.5], [0, 0, 1]])
    return Ks, (W, H), src


# ---------------------------------------------------------------- the twin

class Twin:
    def __init__(self):
        self.sc = sc = bpy.context.scene
        o = self.objs = bpy.data.objects
        missing = [n for n in (FACE, SPLINT, BOARD, BALL, MOTION, LENS, PHONE, PIVOT, TILT, SKULL) if n not in o]
        if missing:
            raise SystemExit(f"not in the scene: {missing}")
        self.vl = bpy.context.view_layer
        self.sa = sc["sensor_arm"].to_dict()
        self.pivot, self.tilt = o[PIVOT], o[TILT]
        self.tilt_basis0 = self.tilt.matrix_basis.copy()
        self.board = board_spec()
        self.c, self.ball_r = ball_centre(o[BALL])
        # face, base shape
        f = self.face = o[FACE]
        me = f.data
        self.face_co0 = arr(me.vertices, "co", 3)
        self.face_mw = mat(f.matrix_world)
        self.face_w0 = self.face_co0 @ self.face_mw[:3, :3].T + self.face_mw[:3, 3]
        n = arr(me.vertex_normals, "vector", 3) @ np.linalg.inv(self.face_mw[:3, :3])
        self.face_n0 = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self.o = self.face_w0.mean(0)
        self.face_shape = None          # identity displacement of the current session
        self.face_state = None
        # moving parts of the marker: the empty marker_motion (parent of every marker part, A1 and B1 parts included)
        # is a root of both interfaces, plus the arm or plate of each
        self.roots = {v: [o[MOTION]] + [o[n] for n in extra if n in o] for v, (_, extra) in VARIANTS.items()}
        self.root_mw0 = {ob.name: ob.matrix_world.copy() for v in self.roots.values() for ob in v}
        unique = {ob.name: ob for v in self.roots.values() for ob in v}
        self.root_order = sorted(unique.values(), key=parent_depth)      # each once, parents first
        self.pend_mw0 = o[PENDULUM_BASE].matrix_world.copy() if PENDULUM_BASE in o else None
        # phone camera fixed to the tilt hinge, from the lens geometry
        self.set_arm(self.sa["shown_deg"], 0.0)
        self.cam_in_tilt = np.linalg.inv(mat(self.tilt.matrix_world)) @ self.lens_camera()
        # render objects
        self.hidden = [ob for ob in o if ob.name.startswith(HIDDEN)]
        pend = set(ob.name for ob in bpy.data.collections["pendulum"].all_objects) if "pendulum" in bpy.data.collections else set()
        for ob in o:
            ob.color = (*[self.label_of(ob, pend) / 255.0] * 3, 1.0)
        self.pend_names = pend
        cam = bpy.data.objects.new("synth_cam", bpy.data.cameras.new("synth_cam"))
        sc.collection.objects.link(cam)
        cam.data.clip_start, cam.data.clip_end = 20.0, 5000.0
        self.cam = sc.camera = cam
        self.drape = None
        self.lights = None
        self.albedo_nodes = None

    @staticmethod
    def label_of(ob, pendulum):
        n = ob.name
        if n == FACE:
            return LABELS["face"]
        if n == DRAPE:
            return LABELS["drape"]
        if n.startswith(CAP):
            return LABELS["cap"]
        if n.startswith(("marker_", "gt_sphere_marker")):
            return LABELS["marker"]
        if n == SPLINT or n.startswith(("a1_", "b1_")):
            return LABELS["interface"]
        if n.startswith(HEAD_REF):
            return LABELS["head_spheres"]
        if n in (SKULL, MANDIBLE, "connector_skull") or n.startswith(("screw_", "washer_", "face_screws")):
            return LABELS["head"]
        if n in pendulum:
            return LABELS["pendulum"]
        return LABELS["bench"]

    # --- poses
    def lens_camera(self):
        """OpenCV camera in the world from the lens: centre of the lens disc, optical axis along the disc normal away
        from the phone, image x along the long side of the phone, y pointing down."""
        L, _ = world_mesh(self.objs[LENS], evaluated=False)
        P, _ = world_mesh(self.objs[PHONE], evaluated=False)
        c = L.mean(0)
        z = np.linalg.svd(L - c)[2][2]
        z = z if z @ (c - P.mean(0)) > 0 else -z
        x = np.linalg.svd(P - P.mean(0))[2][0]
        x -= z * (x @ z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        if y[2] > 0:
            x, y = -x, -y
        return rigid(np.stack([x, y, z], 1), c)

    def set_arm(self, phi, tilt):
        self.pivot.rotation_euler.z = math.radians(phi - self.sa["shown_deg"])
        self.tilt.matrix_basis = self.tilt_basis0 @ Matrix.Rotation(math.radians(tilt), 4, "X")
        self.vl.update()

    def camera(self):
        """World from camera (OpenCV) at the current arm angle and tilt."""
        m = mat(self.tilt.matrix_world) @ self.cam_in_tilt
        u, _, vt = np.linalg.svd(m[:3, :3])
        m[:3, :3] = u @ vt
        return m

    def world_from_board(self):
        ob = self.objs[BOARD]
        me, mw = ob.data, ob.matrix_world
        uv = me.uv_layers.active.data
        corner = {}
        for lp in me.loops:
            u, v = uv[lp.index].uv
            if max(abs(u - round(u)), abs(v - round(v))) < 1e-6:
                corner[(round(u), round(v))] = mw @ me.vertices[lp.vertex_index].co
        tl, ex, ey = corner[(0, 1)], corner[(1, 1)] - corner[(0, 1)], corner[(0, 0)] - corner[(0, 1)]
        size = ex.length
        assert abs(ey.length - size) < 1e-3 and abs(ex.dot(ey)) < 1e-3
        x, y = ex.normalized(), ey.normalized()
        origin = tl + (x + y) * (size - self.board["squares"] * self.board["square_mm"]) / 2
        return rigid(np.stack([np.array(x), np.array(y), np.array(x.cross(y))], 1), np.array(origin)), size

    def set_variant(self, variant):
        for v, (cname, _) in VARIANTS.items():
            coll = bpy.data.collections.get(cname)
            if coll is None:
                continue
            coll.hide_render = v != variant
            lc = self.find_layer_collection(cname)
            if lc is not None:
                lc.hide_viewport = v != variant
        self.vl.update()

    def find_layer_collection(self, name, lc=None):
        lc = lc or self.vl.layer_collection
        if lc.name == name:
            return lc
        for ch in lc.children:
            r = self.find_layer_collection(name, ch)
            if r is not None:
                return r
        return None

    def set_marker(self, variant, D):
        """Move the moving parts of the interface `variant` by the world transform D (from the scene pose); the other
        roots go back to their scene pose. Each object is set once, parents before children: marker_motion is a root
        of both interfaces, and setting it once per interface put it back at its scene pose for every interface but
        the last one (the A1 marker never moved)."""
        move = {ob.name for ob in self.roots[variant]}
        Dm = Matrix(np.asarray(D, np.float64).tolist())
        for ob in self.root_order:
            m0 = self.root_mw0[ob.name]
            ob.matrix_world = Dm @ m0 if ob.name in move else m0
        self.vl.update()

    def place_marker(self, reg, st):
        """Put the marker of registration reg at state st: world from board = dT^-1 . Q . (scene pose). Returns the
        world from board as placed and the labels T_M_B (registration at the state) and T_M_B0 (before the impact,
        T_M_B = T_M_B0 . dT)."""
        D = np.linalg.inv(st["dT"]) @ reg["Q"]
        self.set_marker(reg["interface"], D)
        wb, _ = self.world_from_board()
        T_M_B = np.linalg.inv(wb)
        return wb, T_M_B, T_M_B @ np.linalg.inv(st["dT"])

    def set_pendulum(self, deg):
        if self.pend_mw0 is None:
            return
        t = self.pend_mw0.translation
        self.objs[PENDULUM_BASE].matrix_world = (Matrix.Translation(t) @ Matrix.Rotation(math.radians(-deg), 4, "Z")
                                                  @ Matrix.Translation(-t) @ self.pend_mw0)
        self.vl.update()

    def set_face(self, disp=None):
        """Face of the session (identity displacement) plus disp, both in B (mm)."""
        key = (id(self.face_shape), None if disp is None else id(disp))
        if key == self.face_state:
            return
        w = self.face_w0.copy()
        if self.face_shape is not None:
            w += self.face_shape
        if disp is not None:
            w += disp
        local = (w - self.face_mw[:3, 3]) @ np.linalg.inv(self.face_mw[:3, :3]).T
        me = self.face.data
        me.vertices.foreach_set("co", local.astype(np.float32).ravel())
        me.update()
        self.face_state = key

    def set_drape(self, shape):
        if self.drape is None:
            ob = bpy.data.objects.new(DRAPE, bpy.data.meshes.new(DRAPE))
            self.sc.collection.objects.link(ob)
            m = bpy.data.materials.new(DRAPE)
            m.diffuse_color = (*srgb_to_linear(DRAPE_SRGB), 1.0)
            m.use_backface_culling = False
            ob.data.materials.append(m)
            ob.color = (*[LABELS["drape"] / 255.0] * 3, 1.0)
            self.drape = ob
            self.drape_mat = m
            self.drape_bsdf = None
            nt = m.node_tree if m.node_tree else None
            if nt is not None:
                self.drape_bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        ob = self.drape
        if shape is None:
            ob.hide_render = ob.hide_viewport = True
            return
        old = ob.data
        me = bpy.data.meshes.new(DRAPE)
        me.from_pydata(shape["vertices"].tolist(), [], shape["faces"].tolist())
        me.materials.append(self.drape_mat)
        if hasattr(me, "shade_smooth"):
            me.shade_smooth()
        ob.data = me
        bpy.data.meshes.remove(old)
        ob.hide_render = ob.hide_viewport = False
        if self.drape_bsdf is not None:
            self.drape_bsdf.inputs["Base Color"].default_value = (*shape["color"], 1.0)
            self.drape_bsdf.inputs["Roughness"].default_value = 0.9
        self.vl.update()

    # --- light and appearance
    def setup_lights(self):
        o, sc = self.objs, self.sc
        target = np.array(sc["target"]) if "target" in sc else self.o
        self.lights = []
        for i, name in enumerate(PANELS):
            if name not in o:
                continue
            v, _ = world_mesh(o[name], evaluated=False)
            c = v.mean(0)
            _, s, vt = np.linalg.svd(v - c)
            n = vt[2] if vt[2] @ (target - c) > 0 else -vt[2]
            ext = np.ptp((v - c) @ vt[:2].T, axis=0)
            light = bpy.data.lights.new(f"synth_panel_{i + 1}", "AREA")
            light.shape, light.size, light.size_y = "RECTANGLE", float(ext[0]) * 0.9, float(ext[1]) * 0.9
            ob = bpy.data.objects.new(f"synth_panel_{i + 1}", light)
            sc.collection.objects.link(ob)
            front = c + n * (float(((v - c) @ n).max()) + 0.5)        # just outside the diffuser face
            ob.matrix_world = Matrix(rigid(np.stack([vt[0], np.cross(-n, vt[0]), -n], 1), front).tolist())
            self.lights.append(ob)
        for ob in o:              # the panels themselves do not shadow their light
            if ob.name.startswith("light_panel_"):
                ob.visible_shadow = False
        self.room = [(o[n], o[n].data.energy) for n in ROOM if n in o]
        world = bpy.data.worlds.new("synth_world")
        world.use_nodes = True
        self.world_bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
        sc.world = world

    def set_light(self, p):
        for ob, (on, level, kelvin) in zip(self.lights, p["panels"]):
            ob.data.energy = PANEL_W * level if on else 0.0
            ob.data.color = kelvin_rgb(kelvin)
        for (ob, e0), level in zip(self.room, p["room"]):
            ob.data.energy = e0 * level
        self.world_bg.inputs["Color"].default_value = (*p["world_rgb"], 1.0)
        self.world_bg.inputs["Strength"].default_value = p["world"]
        self.exposure = p["exposure"]

    def setup_albedo(self):
        m = self.face.material_slots[0].material
        nt = m.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        coord = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], mix.inputs[0])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        self.albedo_nodes = (noise, mix, bsdf)

    def set_albedo(self, p):
        noise, mix, bsdf = self.albedo_nodes
        mix.inputs[6].default_value = (*p["a"], 1.0)
        mix.inputs[7].default_value = (*p["b"], 1.0)
        noise.inputs["Scale"].default_value = p["noise_scale"]
        noise.inputs["Detail"].default_value = 4.0
        bsdf.inputs["Roughness"].default_value = p["roughness"]

    # --- rendering
    def set_camera(self, world_from_cv, K, size):
        W, H = size
        d, sc = self.cam.data, self.sc
        d.type, d.sensor_fit, d.sensor_width = "PERSP", "HORIZONTAL" if W >= H else "VERTICAL", 36.0
        if W >= H:
            d.sensor_width = 36.0
        else:
            d.sensor_height = 36.0
        side = max(W, H)
        d.lens = K[0, 0] / side * 36.0
        d.shift_x = (W / 2 - 0.5 - K[0, 2]) / side
        d.shift_y = (K[1, 2] - H / 2 + 0.5) / side
        sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
        sc.render.pixel_aspect_x = sc.render.pixel_aspect_y = 1.0
        self.cam.matrix_world = Matrix((world_from_cv @ np.diag([1.0, -1.0, -1.0, 1.0])).tolist())

    def hide_phone(self, hide=True):
        for ob in self.hidden:
            ob.hide_render = hide

    def show_head_spheres(self, show):
        """Head spheres and their frame in the images or not. The dataset hides them: on the bench they are still in
        sight of the phone, and a network must never learn to use them."""
        for ob in self.objs:
            if ob.name.startswith(HEAD_REF):
                ob.hide_render = not show

    def rgb_mode(self, samples):
        sc = self.sc
        sc.render.engine = "BLENDER_EEVEE"
        sc.eevee.taa_render_samples = samples
        sc.view_settings.view_transform, sc.view_settings.look = "Standard", "None"
        sc.view_settings.exposure = getattr(self, "exposure", 0.0)
        sc.render.dither_intensity = 1.0
        sc.render.film_transparent = False
        sc.render.use_compositing = False
        s = sc.render.image_settings
        s.file_format, s.color_mode, s.quality = "JPEG", "RGB", 95

    def label_mode(self, depth=False):
        sc = self.sc
        sc.render.engine = "BLENDER_WORKBENCH"
        sh = sc.display.shading
        sh.light, sh.color_type = "FLAT", "OBJECT"
        sh.show_shadows = sh.show_cavity = sh.show_object_outline = sh.show_specular_highlight = False
        sh.show_xray = sh.show_backface_culling = False
        sc.display.render_aa = "OFF"
        sc.view_settings.view_transform, sc.view_settings.exposure = "Raw", 0.0
        sc.render.dither_intensity = 0.0
        sc.render.film_transparent = True
        s = sc.render.image_settings
        s.file_format, s.color_mode, s.color_depth, s.compression = "PNG", "BW", "8", 90
        sc.render.use_compositing = depth
        if depth:
            self.depth_nodes()

    def depth_nodes(self):
        if getattr(self, "depth_out", None) is not None:
            return
        self.vl.use_pass_z = True
        ng = bpy.data.node_groups.new("synth_depth", "CompositorNodeTree")
        self.sc.compositing_node_group = ng
        rl = ng.nodes.new("CompositorNodeRLayers")
        mul = ng.nodes.new("ShaderNodeMath")
        mul.operation, mul.inputs[1].default_value = "MULTIPLY", 16.0 / 65535.0
        fo = ng.nodes.new("CompositorNodeOutputFile")
        fo.file_output_items.new("FLOAT", "depth")
        fo.format.media_type = "IMAGE"
        fo.format.file_format, fo.format.color_mode, fo.format.color_depth = "PNG", "BW", "16"
        fo.save_as_render = False
        ng.links.new(rl.outputs["Depth"], mul.inputs[0])
        ng.links.new(mul.outputs[0], fo.inputs[0])
        go = ng.nodes.new("NodeGroupOutput")
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        ng.links.new(rl.outputs["Image"], go.inputs[0])
        self.depth_out = fo

    def render(self, path, depth_stem=None):
        if depth_stem is not None and getattr(self, "depth_out", None) is not None:
            self.depth_out.directory, self.depth_out.file_name = str(depth_stem.parent), depth_stem.name + "_"
        self.sc.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)

    @staticmethod
    def label_counts(path):
        img = bpy.data.images.load(str(path), check_existing=False)
        img.colorspace_settings.name = "Non-Color"
        px = np.empty(img.size[0] * img.size[1] * 4, np.float32)
        img.pixels.foreach_get(px)
        bpy.data.images.remove(img)
        lab = np.round(px[0::4] * 255).astype(np.int64)
        counts = np.bincount(lab, minlength=len(LABELS))
        return {k: int(counts[v]) for k, v in LABELS.items()}

    @staticmethod
    def label_image(path):
        """Labels of a label image as an H x W array, first row at the top (Blender stores the bottom row first)."""
        img = bpy.data.images.load(str(path), check_existing=False)
        img.colorspace_settings.name = "Non-Color"
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        bpy.data.images.remove(img)
        return np.round(px[0::4] * 255).astype(np.int64).reshape(h, w)[::-1]


# ---------------------------------------------------------------- commands

def open_twin(out, force_copy=False):
    copy = out / "twin_scene.blend"
    status = refresh_copy(SCENE, copy, force_copy)
    bpy.ops.wm.open_mainfile(filepath=str(copy), load_ui=False)
    import twin_export
    scene_print = twin_export.fingerprint()[0]      # of what the phone sees, before the twin changes the scene
    print(json.dumps({"scene": str(copy), "copy": status, "scene_print": scene_print}))
    tw = Twin()
    tw.scene_print = scene_print
    return tw


def cmd_probe(tw, args):
    o = tw.objs
    rep = {"ball_centre": lst(tw.c, 3), "ball_radius": round(tw.ball_r, 3), "face_centroid": lst(tw.o, 3),
           "face_vertices": len(tw.face_w0), "arm_deg": tw.sa["arm_deg"], "sweep_deg": tw.sa["sweep_deg"],
           "cap": [ob.name for ob in o if ob.name.startswith(CAP)],
           "head_spheres": {ob.name: lst(sphere_of(ob)[0], 2) for ob in o if ob.name.startswith(HEAD_SPHERES)}}
    tw.set_arm(tw.sa["shown_deg"], 0.0)
    m = tw.camera()
    if PHONE_CAM in o:
        c = mat(o[PHONE_CAM].matrix_world.normalized())
        rep["camera_vs_cam_phone_2"] = {
            "centre_mm": round(float(np.linalg.norm(c[:3, 3] - m[:3, 3])), 3),
            "axis_deg": round(math.degrees(math.acos(np.clip(-c[:3, 2] @ m[:3, 2], -1, 1))), 4),
            "x_deg": round(math.degrees(math.acos(np.clip(c[:3, 0] @ m[:3, 0], -1, 1))), 4)}
    tw.set_arm(tw.sa["shown_deg"], 10.0)
    m2 = tw.camera()
    R = m2[:3, :3] @ m[:3, :3].T
    ang = math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    rep["tilt_10deg"] = {"angle_deg": round(ang, 3),
                         "axis_vs_camera_x_deg": round(math.degrees(math.acos(abs(ax @ m[:3, 0]) / np.linalg.norm(ax))), 2),
                         "pitch_deg": round(math.degrees(math.asin(-m2[2, 2])) - math.degrees(math.asin(-m[2, 2])), 3)}
    tw.set_arm(tw.sa["shown_deg"], 0.0)
    wb, size = tw.world_from_board()
    rep["board"] = {"print_mm": round(size, 3), "origin": lst(wb[:3, 3], 3), "normal": lst(wb[:3, 2], 4),
                    "distance_from_camera_mm": round(float(np.linalg.norm(np.linalg.inv(m) @ np.r_[wb[:3, 3], 1])), 1)}
    for k, phi in enumerate(tw.sa["arm_deg"]):
        tw.set_arm(phi, 0.0)
        m = tw.camera()
        d = np.linalg.inv(m) @ np.r_[tw.o, 1]
        rep[f"p{k + 1}"] = {"arm_deg": round(phi, 2), "face_depth_mm": round(float(d[2]), 1),
                            "elevation_deg": round(math.degrees(math.asin(-m[2, 2])), 2)}
    print(json.dumps(rep, indent=1))
    (args.out / "probe.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


def project_points(m, K, X):
    """Pixels (OpenCV convention) of world points X (N x 3) seen by the camera m (world from camera)."""
    x = (np.atleast_2d(X) - m[:3, 3]) @ m[:3, :3]
    return np.c_[K[0, 0] * x[:, 0] / x[:, 2] + K[0, 2], K[1, 1] * x[:, 1] / x[:, 2] + K[1, 2]]


def cmd_test(tw, args):
    """Checks of the marker motion, through the code path of render (Twin.place_marker); exit code 1 on a failure.

    For each interface a start registration Q (A1: rotated by TEST_DEG about the ball centre; B1: the scene pose) and
    an impact dT (A1: TEST_DEG about the ball centre; B1: TEST_DEG about the face centroid plus TEST_MM), large enough
    that a marker left at its scene pose cannot pass:
    - scene: the board lands at dT^-1 . Q . (scene pose) within TEST_TOL_MM; on A1 the ball centre stays in place;
    - labels: T_M_B0 is the registration before the impact (Q only, whatever dT) and T_M_B = T_M_B0 . dT;
    - images: the marker pixels of the label image at photo position p1 move, and points of the board projected with
      the label pose T_C_M = T_C_B . T_M_B^-1 fall on marker pixels, before and after the move;
    - set_marker with the identity puts the board back at the scene pose.
    Nothing is written but a temporary label image in OUT."""
    fails = []

    def check(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    side = tw.board["squares"] * tw.board["square_mm"]
    board_pts = np.array([[side / 2, side / 2, 0.0], [side / 4, side / 4, 0.0], [3 * side / 4, side / 4, 0.0],
                          [side / 4, 3 * side / 4, 0.0], [3 * side / 4, 3 * side / 4, 0.0]])   # in M
    corners = np.array([[0.0, 0.0, 0.0], [side, 0.0, 0.0], [0.0, side, 0.0], [side, side, 0.0]])
    probe_B = np.vstack([tw.c, tw.o, tw.c + [50.0, 0.0, 0.0], tw.o + [0.0, 0.0, 50.0]])       # points of B

    def apply(T, X):
        return X @ T[:3, :3].T + T[:3, 3]

    def gap(A, B, X):
        return float(np.linalg.norm(apply(A, X) - apply(B, X), axis=1).max())

    tw.hide_phone(True)
    tw.show_head_spheres(False)
    tw.label_mode(depth=False)
    K, size, _ = intrinsics("photo", TEST_SCALE)
    tw.set_arm(tw.sa["arm_deg"][0], 0.0)
    cam = tw.camera()
    tw.set_camera(cam, K, size)
    tmp = args.out / "_test_label.png"

    def label():
        tw.render(tmp)
        lab = Twin.label_image(tmp)
        tmp.unlink()
        return lab

    def on_marker(lab, wb):
        uv = np.round(project_points(cam, K, apply(wb, board_pts))).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < lab.shape[1]) & (uv[:, 1] < lab.shape[0])
        return int(sum(lab[v, u] == LABELS["marker"] for u, v in uv[inside]))

    ball = tw.objs[BALL]
    for k, variant in enumerate(VARIANTS):
        if VARIANTS[variant][0] not in bpy.data.collections:
            check(False, f"{variant}: collection {VARIANTS[variant][0]} in the scene")
            continue
        rng = np.random.default_rng([args.seed, 7, k])
        tw.set_variant(variant)
        tw.set_marker(variant, np.eye(4))
        wb0, _ = tw.world_from_board()
        ball0 = mat(ball.matrix_world)
        lab0 = label()
        if variant == "a1":
            reg = {"interface": "a1", "Q": about(rotation(unit(rng), TEST_DEG), tw.c)}
            dT = about(rotation(unit(rng), TEST_DEG), tw.c)
        else:
            reg = {"interface": "b1", "Q": np.eye(4)}
            dT = about(rotation(unit(rng), TEST_DEG), tw.o, unit(rng) * TEST_MM)
        want = np.linalg.inv(dT) @ reg["Q"] @ wb0
        wb, T_M_B, T_M_B0 = tw.place_marker(reg, {"dT": dT})
        moved = gap(want, wb0, corners)
        check(moved > 1.0, f"{variant}: the test displacement moves the board by {moved:.2f} mm")
        err = gap(wb, want, corners)
        check(err < TEST_TOL_MM, f"{variant}: board at dT^-1 . Q . scene pose (off by {err:.5f} mm)")
        if variant == "a1":
            c1 = apply(mat(ball.matrix_world) @ np.linalg.inv(ball0), tw.c[None])[0]
            check(np.linalg.norm(c1 - tw.c) < TEST_TOL_MM,
                  f"a1: the ball turns in its seat (centre moved {np.linalg.norm(c1 - tw.c):.5f} mm)")
        T_reg = np.linalg.inv(reg["Q"] @ wb0)
        check(gap(T_M_B0, T_reg, probe_B) < TEST_TOL_MM,
              f"{variant}: T_M_B0 is the registration before the impact (off by {gap(T_M_B0, T_reg, probe_B):.5f} mm)")
        check(gap(T_M_B, T_M_B0 @ dT, probe_B) < 1e-9, f"{variant}: T_M_B = T_M_B0 . dT")
        lab1 = label()
        m0, m1 = lab0 == LABELS["marker"], lab1 == LABELS["marker"]
        changed = int((m0 ^ m1).sum())
        check(m0.any() and changed > max(20, 0.02 * m0.sum()),
              f"{variant}: marker pixels move in the label image ({changed} of {int(m0.sum())} changed)")
        hits0, hits1 = on_marker(lab0, wb0), on_marker(lab1, np.linalg.inv(T_M_B))
        check(hits0 == len(board_pts) and hits1 == len(board_pts),
              f"{variant}: board points projected with the label pose land on marker pixels "
              f"({hits0}/{len(board_pts)} before, {hits1}/{len(board_pts)} after)")
        tw.set_marker(variant, np.eye(4))
        back = gap(tw.world_from_board()[0], wb0, corners)
        check(back < TEST_TOL_MM, f"{variant}: the identity puts the board back ({back:.5f} mm)")
    tw.set_marker("a1", np.eye(4))
    print(f"{len(fails)} failed" if fails else "all passed", flush=True)
    if fails:
        raise SystemExit(1)


def decimate(vertices, faces, target):
    me = bpy.data.meshes.new("synth_tmp")
    me.from_pydata(vertices.tolist(), [], faces.tolist())
    ob = bpy.data.objects.new("synth_tmp", me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.decimate_type, mod.ratio, mod.use_collapse_triangulate = "COLLAPSE", target / len(faces), True
    bpy.context.view_layer.update()
    v, f = world_mesh(ob)
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    return v, f


def surface_distance(points, vertices, faces):
    bvh = BVHTree.FromPolygons(vertices.tolist(), faces.tolist())
    return np.array([(bvh.find_nearest(Vector(p))[3]) for p in points])


def cap_mesh(tw):
    """World vertices and triangles of the surgical cap (all its mesh objects), or None."""
    obs = [ob for ob in tw.objs if ob.name.startswith(CAP) and ob.type == "MESH" and ob.name in tw.vl.objects]
    if not obs:
        return None
    parts = [world_mesh(ob) for ob in obs]
    offs = np.cumsum([0] + [len(v) for v, _ in parts[:-1]])
    return np.concatenate([v for v, _ in parts]), np.concatenate([f + o for (_, f), o in zip(parts, offs)])


def face_weights(tw, out):
    path = out / "meshes" / "face_weights.npz"
    n = len(tw.face_w0)
    cap = cap_mesh(tw)
    cap_sig = np.r_[len(cap[0]), cap[0].sum(0)] if cap is not None else np.zeros(4)
    if path.exists():
        z = np.load(path)
        if (int(z["vertex_count"]) == n and np.allclose(z["probe"], tw.face_w0[:: max(n // 50, 1)], atol=1e-4)
                and "cap_sig" in z and np.allclose(z["cap_sig"], cap_sig, atol=1e-2)):
            return z["w"]
    sv, sf = world_mesh(tw.objs[SPLINT])
    bvh = BVHTree.FromPolygons(sv.tolist(), sf.tolist())
    near = np.array([bvh.find_nearest(Vector(p))[3] for p in tw.face_w0]) < 3.0
    sk, _ = world_mesh(tw.objs[SKULL], evaluated=False)
    inside = (sk.min(0) + sk.max(0)) / 2
    outer = np.sum(tw.face_n0 * (tw.face_w0 - inside), axis=1) > 0
    e = arr(tw.face.data.edges, "vertices", 2, np.int64)
    cut = e[outer[e[:, 0]] != outer[e[:, 1]]].ravel()     # the mask is a closed shell: its cut is where the outer
    border = np.zeros(n, bool)                            # skin turns into the rim and the inner skin
    border[cut] = True
    bm = bmesh.new()
    bm.from_mesh(tw.face.data)
    border |= np.array([v.is_boundary for v in bm.verts])
    bm.free()

    def dist_to(mask):
        kd = KDTree(int(mask.sum()))
        for i, p in zip(np.flatnonzero(mask), tw.face_w0[mask]):
            kd.insert(p, int(i))
        kd.balance()
        return np.array([kd.find(p)[2] for p in tw.face_w0])

    d_mouth, d_border = dist_to(near), dist_to(border)
    # skin under the surgical cap: the cap sits a few mm above the silicone (in places the silicone crosses its inner
    # surface), so the skin there must not move, or it shows through the cap
    covered = np.zeros(n, bool)
    if cap is not None:
        cbvh = BVHTree.FromPolygons(cap[0].tolist(), cap[1].tolist())
        for i in np.flatnonzero(outer):
            p, nrm = tw.face_w0[i], tw.face_n0[i]
            covered[i] = cbvh.ray_cast(Vector(p - nrm * 0.05), Vector(nrm), CAP_RAY_MM)[0] is not None
    d_cap = dist_to(covered) if covered.any() else None
    w = fm.weights(tw.face_w0, tw.face_n0, inside, d_mouth, d_border, d_cap)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, w=w, d_mouth=d_mouth.astype(np.float32), d_border=d_border.astype(np.float32),
                        covered=covered, vertex_count=n, probe=tw.face_w0[:: max(n // 50, 1)], cap_sig=cap_sig)
    print(json.dumps({"weights": {"mouth_vertices": int(near.sum()), "border_vertices": int(border.sum()),
                                  "outer_vertices": int(outer.sum()), "under_cap_vertices": int(covered.sum()),
                                  "weight_mean": round(float(w.mean()), 3), "weight_zero": round(float((w == 0).mean()), 3)}}))
    return w


def cmd_meshes(tw, args):
    out = args.out / "meshes"
    out.mkdir(parents=True, exist_ok=True)
    faces = {"phantom": None}
    faces.update({k: v["displacement"] for k, v in fm.load_bank(args.out / "faces", len(tw.face_w0)).items()})
    rep = {}
    for fid, disp in faces.items():
        tw.face_shape = disp
        tw.face_state = None
        tw.set_face(None)
        v, f = world_mesh(tw.face)
        np.savez_compressed(out / f"face_{fid}_full.npz", vertices=v.astype(np.float32), faces=f)
        dv, df = decimate(v, f, DECIMATED)
        rng = np.random.default_rng(0)
        sample = v[rng.choice(len(v), min(len(v), 30000), replace=False)]
        e1 = surface_distance(sample, dv, df)          # full surface to decimated
        e2 = surface_distance(dv, v, f)                # decimated vertices to full
        err = {"full_to_decimated_p95": float(np.percentile(e1, 95)), "full_to_decimated_max": float(e1.max()),
               "decimated_to_full_max": float(e2.max())}
        np.savez_compressed(out / f"face_{fid}_r20k.npz", vertices=dv.astype(np.float32), faces=df,
                            error_mm=json.dumps(err))
        rep[fid] = {"full": [len(v), len(f)], "decimated": [len(dv), len(df)], **{k: round(x, 4) for k, x in err.items()}}
    tw.face_shape = None
    tw.face_state = None
    tw.set_face(None)
    # occluders: fixed to the head (B) and moving with the board (M, at the scene registration)
    wb, _ = tw.world_from_board()
    parts = {}
    for ob in tw.objs:
        if ob.type != "MESH" or ob.hide_render:
            continue
        lab = Twin.label_of(ob, tw.pend_names)
        if lab in (LABELS["marker"], LABELS["interface"]):
            v, f = world_mesh(ob)
            parts[f"{ob.name}__v"], parts[f"{ob.name}__f"] = v.astype(np.float32), f
    moving = sorted({ob.name for roots in tw.roots.values() for r in roots for ob in [r, *r.children_recursive]})
    np.savez_compressed(out / "occluders.npz", T_M_B=np.linalg.inv(wb), moving=np.array(moving), **parts)
    rep["occluders"] = sorted(k[:-3] for k in parts if k.endswith("__v"))
    face_weights(tw, args.out)
    (out / "meshes.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


# --- views and samples

def photo_views(tw, rng):
    return [{"view": f"p{k + 1}", "mode": "photo", "arm_deg": phi + rng.normal(0, ARM_NOISE_DEG),
             "tilt_deg": rng.normal(0, TILT_NOISE_DEG)} for k, phi in enumerate(tw.sa["arm_deg"])]


def video_views(tw, rng, n):
    if n <= 0:
        return []
    lo, hi = tw.sa["sweep_deg"]
    step = (hi - lo) / n
    phis = lo + step * (np.arange(n) + rng.uniform(0.0, 1.0, n))
    if rng.random() < 0.5:
        tilt = np.zeros(n)
    else:     # slow tilt with the lever: a ramp between two random angles
        a, b = rng.uniform(-TILT_MAX, TILT_MAX, 2)
        tilt = a + (b - a) * (phis - lo) / (hi - lo)
    return [{"view": f"v{k:02d}", "mode": "video", "arm_deg": float(p), "tilt_deg": float(t)}
            for k, (p, t) in enumerate(zip(phis, tilt))]


def make_session(i, seed, bank):
    rng = np.random.default_rng([seed, i, 0])
    u, acc, split = rng.random(), 0.0, SPLITS[-1][0]
    for name, share in SPLITS:
        acc += share
        if u < acc:
            split = name
            break
    candidates = [k for k, v in bank.items() if v["split"] == split]
    face = "phantom" if not candidates or rng.random() < PHANTOM_SHARE else str(rng.choice(candidates))
    kelvin = rng.uniform(3200, 6500)
    return {"session": i, "split": split, "face": face, "pendulum_deg": float(rng.choice(PENDULUM_DEG)),
            "panel_kelvin": round(float(kelvin), 0), "seed": seed}


def sample_registration(tw, sess, r):
    rng = np.random.default_rng([sess["seed"], sess["session"], 1, r])
    variant = "b1" if rng.random() < B1_SHARE else "a1"
    if variant == "a1":
        Q = about(rotation(unit(rng), rng.uniform(0, REG_DEG)), tw.c)
    else:
        Q = np.eye(4)
    return {"registration": r, "interface": variant, "Q": Q}


def sample_state(tw, sess, reg, s, drapes):
    rng = np.random.default_rng([sess["seed"], sess["session"], 2, reg["registration"], s])
    if s == 0:
        dT, motion = np.eye(4), {"kind": "none", "deg": 0.0, "mm": 0.0}
    elif reg["interface"] == "a1":
        ax, deg = unit(rng), amplitude(rng, MAX_DEG)
        dT, motion = about(rotation(ax, deg), tw.c), {"kind": "a1", "deg": deg, "mm": 0.0, "axis": ax.tolist()}
    else:
        ax, deg = unit(rng), amplitude(rng, MAX_DEG)
        sdir, mm = unit(rng), amplitude(rng, MAX_MM)
        dT = about(rotation(ax, deg), tw.o, sdir * mm)
        motion = {"kind": "rigid", "deg": deg, "mm": mm, "axis": ax.tolist(), "shift_dir": sdir.tolist()}
    drape = int(rng.integers(len(drapes))) if drapes and rng.random() < DRAPE_SHARE else None
    amp = rng.uniform(*fm.AMP_MM)
    if drape is not None:
        amp = min(amp, DRAPE_AMP_MM)
    panels = [[bool(rng.random() < 0.85), float(rng.choice([0.5, 1.0]) * rng.uniform(0.9, 1.1)),
               float(sess["panel_kelvin"] + rng.normal(0, 150))] for _ in PANELS]
    if not any(p[0] for p in panels):
        panels[int(rng.integers(len(panels)))][0] = True
    room = [0.0 if rng.random() < 0.5 else float(rng.uniform(0.1, 0.4)) for _ in ROOM]
    light = {"panels": panels, "room": room, "world": float(rng.uniform(0.02, 0.15)),
             "world_rgb": (np.full(3, rng.uniform(0.5, 1.0)) * rng.uniform(0.9, 1.1, 3)).tolist(),
             "exposure": float(rng.uniform(-0.5, 0.5))}
    if rng.random() < 0.5:     # silicone with small variations
        base = srgb_to_linear(SILICONE_SRGB) * rng.uniform(0.9, 1.1, 3)
        albedo = {"kind": "silicone", "a": base.tolist(), "b": (base * rng.uniform(0.92, 1.0)).tolist()}
    else:                      # procedural skin tone
        k = int(rng.integers(len(SKIN_SRGB) - 1))
        a = srgb_to_linear([int(SKIN_SRGB[k][j:j + 2], 16) for j in (0, 2, 4)])
        b = srgb_to_linear([int(SKIN_SRGB[k + 1][j:j + 2], 16) for j in (0, 2, 4)])
        albedo = {"kind": f"skin{k}", "a": a.tolist(), "b": (a + (b - a) * rng.uniform(0.2, 0.8)).tolist()}
    albedo.update({"noise_scale": float(rng.uniform(0.02, 0.15)), "roughness": float(rng.uniform(0.35, 0.6))})
    def_seed = int(rng.integers(2 ** 31))
    views = photo_views(tw, rng) + video_views(tw, rng, sess["video"])
    return {"state": s, "dT": dT, "motion": motion, "drape": drape, "amp_mm": amp, "def_seed": def_seed,
            "light": light, "albedo": albedo, "views": views}


def load_targets():
    """Clinical targets of the TRE in B (03_data/processed/targets, role "proposed")."""
    import csv
    if not TARGETS.exists():
        return np.empty((0, 3)), []
    with TARGETS.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["role"] == "proposed"]
    return np.array([[float(r[k]) for k in ("x_B", "y_B", "z_B")] for r in rows]), [r["id"] for r in rows]


def load_drapes(out):
    drapes = []
    for p in sorted((out / "drapes").glob("drape_*.npz")):
        z = np.load(p)
        drapes.append({"name": p.stem, "vertices": z["vertices"], "faces": z["faces"]})
    return drapes


def cmd_render(tw, args):
    out = args.out
    w = face_weights(tw, out)
    bank = fm.load_bank(out / "faces", len(tw.face_w0))
    drapes = load_drapes(out)
    targets, target_ids = load_targets()
    tw.setup_lights()
    tw.setup_albedo()
    tw.show_head_spheres(args.head_spheres)
    modes = {m: intrinsics(m, args.scale) for m in ("photo", "video")}
    for K, _, _ in modes.values():          # check of the principal point convention (synth_data.py check)
        K[0, 2] += args.pp_offset[0]
        K[1, 2] += args.pp_offset[1]
    a, b = (int(x) for x in args.sessions.split(":"))
    times = []
    for i in range(a, b):
        sess = make_session(i, args.seed, bank)
        sess["video"] = args.video
        sdir = out / "sessions" / f"s{i:04d}"
        sdir.mkdir(parents=True, exist_ok=True)
        done = set()
        fl = sdir / "frames.jsonl"
        if fl.exists():
            done = {json.loads(line)["state_key"] for line in fl.read_text(encoding="utf-8").splitlines() if line}
        if done and (sdir / "session.json").exists():     # resume only what was started with the same settings
            prev = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
            changed = [k for k in ("face", "split", "video") if prev.get(k) != sess[k]]
            if str(prev.get("head_spheres", "shown")).startswith("shown") != bool(args.head_spheres):
                changed.append("head_spheres")
            if changed:
                raise SystemExit(f"s{i:04d} was rendered with other {changed} (new face bank?): delete it or render "
                                 f"new session indices")
        tw.face_shape = bank[sess["face"]]["displacement"] if sess["face"] != "phantom" else None
        tw.face_state = None
        tw.set_pendulum(sess["pendulum_deg"])
        info = dict(sess, intrinsics={m: {"K": lst(K), "size": s, "source": src} for m, (K, s, src) in modes.items()},
                    labels=LABELS, targets=target_ids, ball_centre=lst(tw.c, 4), face_centroid=lst(tw.o, 4), board=tw.board,
                    scale=args.scale, samples=args.samples, blender=bpy.app.version_string,
                    head_spheres="shown" if args.head_spheres else "hidden (spheres and frame left out of the images)",
                    scene_mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(SCENE.stat().st_mtime)),
                    scene_print=tw.scene_print)
        (sdir / "session.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        for r in range(args.registrations):
            reg = sample_registration(tw, sess, r)
            tw.set_variant(reg["interface"])
            for s in range(args.states):
                key = f"r{r}_s{s}"
                if key in done:
                    continue
                t0 = time.perf_counter()
                st = sample_state(tw, sess, reg, s, drapes)
                wb, T_M_B, T_M_B0 = tw.place_marker(reg, st)      # marker in the world: start Q, then dT^-1
                shift = np.linalg.norm(targets @ st["dT"][:3, :3].T + st["dT"][:3, 3] - targets, axis=1)
                target_mm = {"mean": round(float(shift.mean()), 4), "max": round(float(shift.max()), 4)} if len(shift) else None
                disp, dpar = fm.deformation(tw.face_w0, tw.face_n0, w, np.random.default_rng(st["def_seed"]),
                                            amp_mm=st["amp_mm"])
                tw.set_face(disp)
                if st["drape"] is not None:
                    dr = drapes[st["drape"]]
                    rng = np.random.default_rng(st["def_seed"])
                    color = (srgb_to_linear(DRAPE_SRGB) * rng.uniform(0.85, 1.15, 3)).tolist()
                    tw.set_drape({"vertices": dr["vertices"], "faces": dr["faces"], "color": color})
                else:
                    tw.set_drape(None)
                tw.set_light(st["light"])
                tw.set_albedo(st["albedo"])
                tw.hide_phone(True)
                lines, cams = [], []
                tw.rgb_mode(args.samples)
                t1 = time.perf_counter()
                for v in st["views"]:
                    tw.set_arm(v["arm_deg"], v["tilt_deg"])
                    K, size, _ = modes[v["mode"]]
                    m = tw.camera()
                    cams.append(m)
                    tw.set_camera(m, K, size)
                    tw.render(sdir / f"{key}_{v['view']}.jpg")
                t2 = time.perf_counter()
                tw.label_mode(depth=args.depth)
                counts = []
                for v, m in zip(st["views"], cams):
                    tw.set_arm(v["arm_deg"], v["tilt_deg"])
                    K, size, _ = modes[v["mode"]]
                    tw.set_camera(m, K, size)
                    lab = sdir / f"{key}_{v['view']}_label.png"
                    tw.render(lab, sdir / f"{key}_{v['view']}" if args.depth else None)
                    counts.append(Twin.label_counts(lab))
                t3 = time.perf_counter()
                if args.model_mask:            # undeformed face alone, for the check of mesh_overlay.py
                    tw.set_face(None)
                    keep = {ob.name: ob.hide_render for ob in tw.objs}
                    for ob in tw.objs:
                        ob.hide_render = ob is not tw.face
                    tw.sc.render.use_compositing = False
                    for v, m in zip(st["views"], cams):
                        K, size, _ = modes[v["mode"]]
                        tw.set_camera(m, K, size)
                        tw.render(sdir / f"{key}_{v['view']}_model.png")
                    for ob in tw.objs:
                        ob.hide_render = keep[ob.name]
                t4 = time.perf_counter()
                n = len(st["views"])
                for v, m, cnt in zip(st["views"], cams, counts):
                    K, size, _ = modes[v["mode"]]
                    T_C_B = np.linalg.inv(m)
                    lines.append({
                        "state_key": key, "image": f"{key}_{v['view']}", "session": i, "split": sess["split"],
                        "registration": r, "state": s, "view": v["view"], "mode": v["mode"], "size": list(size),
                        "K": lst(K), "dist": [0.0] * 5, "T_C_B": lst(T_C_B), "T_C_M": lst(T_C_B @ wb),
                        "T_M_B": lst(T_M_B), "T_M_B0": lst(T_M_B0), "dT": lst(st["dT"]), "interface": reg["interface"],
                        "motion": st["motion"], "target_shift_mm": target_mm, "arm_deg": round(v["arm_deg"], 4), "tilt_deg": round(v["tilt_deg"], 4),
                        "face": sess["face"], "deformation": dpar, "drape": None if st["drape"] is None else drapes[st["drape"]]["name"],
                        "light": st["light"], "albedo": st["albedo"], "pendulum_deg": sess["pendulum_deg"],
                        "head_spheres": bool(args.head_spheres), "pixels": cnt, "ms": {"rgb": round((t2 - t1) / n * 1e3), "label": round((t3 - t2) / n * 1e3),
                                              "model": round((t4 - t3) / n * 1e3), "setup": round((t1 - t0) * 1e3)}})
                with fl.open("a", encoding="utf-8") as fh:
                    fh.write("".join(json.dumps(line) + "\n" for line in lines))
                times.append({"state": f"s{i:04d}/{key}", "images": n, "s": round(t4 - t0, 2),
                              "rgb_s": round(t2 - t1, 2), "label_s": round(t3 - t2, 2), "model_s": round(t4 - t3, 2)})
                print(json.dumps(times[-1]), flush=True)
    tw.set_marker("a1", np.eye(4))
    tw.set_face(None)
    if times:
        imgs = sum(t["images"] for t in times)
        print(json.dumps({"states": len(times), "images": imgs, "s_per_image": round(sum(t["s"] for t in times) / imgs, 3),
                          "rgb_s_per_image": round(sum(t["rgb_s"] for t in times) / imgs, 3),
                          "label_s_per_image": round(sum(t["label_s"] for t in times) / imgs, 3)}))


# --- s1: the head spheres seen from the phone

def phone_poses(tw, video_step=2.0, tilts=(-TILT_MAX, 0.0, TILT_MAX)):
    poses = [("photo", f"p{k + 1}", phi, 0.0) for k, phi in enumerate(tw.sa["arm_deg"])]
    lo, hi = tw.sa["sweep_deg"]
    for phi in np.arange(lo, hi + 1e-6, video_step):
        poses += [("video", f"video_{phi:.1f}_{t:+.0f}", float(phi), t) for t in tilts]
    return poses


def visible(tw, dg, cam_pos, pts, target_obj=None, tol=0.3):
    """For each point: True when the segment from the camera reaches it without hitting anything else."""
    out = np.zeros(len(pts), bool)
    for j, p in enumerate(pts):
        d = p - cam_pos
        L = np.linalg.norm(d)
        d /= L
        start = cam_pos + d * 10.0          # in front of the phone's own camera module
        hit, loc, _, _, ob, _ = tw.sc.ray_cast(dg, Vector(start), Vector(d), distance=L - 10.0 - tol)
        out[j] = (not hit) or (target_obj is not None and ob is not None and ob.name == target_obj)
    return out


def in_image(m, K, size, pts):
    x = (pts - m[:3, 3]) @ m[:3, :3]
    ok = x[:, 2] > 1.0
    u = K[0, 0] * x[:, 0] / np.maximum(x[:, 2], 1e-9) + K[0, 2]
    v = K[1, 1] * x[:, 1] / np.maximum(x[:, 2], 1e-9) + K[1, 2]
    return ok & (u >= -0.5) & (v >= -0.5) & (u <= size[0] - 0.5) & (v <= size[1] - 0.5)


def scan_positions(sa):
    """The arm positions the trials scan from (trials.Config.scan_az, camera azimuths: the protocol's own), as
    indices of the scene's arm positions (sensor_arm camera_deg)."""
    import trials
    cam = [float(a) for a in sa["camera_deg"]]
    return [min(range(len(cam)), key=lambda k: abs(cam[k] - float(az))) for az in trials.Config().scan_az]


def scanner_poses(tw):
    """Scanner camera at the trials' scan positions (scan_positions), each the scene's own camera there
    (cam_scanner_<k>, placed by the bench's script with the scanner and its wedge as built); a scene without them:
    cam_scanner_2 turned about the pivot."""
    if SCANNER_CAM not in tw.objs:
        return [], 0.0, 0.0
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    c0 = mat(tw.objs[SCANNER_CAM].matrix_world.normalized()) @ flip
    piv = np.array(tw.sa["pivot"], np.float64)
    out = []
    for k in scan_positions(tw.sa):
        own = tw.objs.get(f"cam_scanner_{k + 1}")
        if own is not None:
            out.append(mat(own.matrix_world.normalized()) @ flip)
        else:
            R = rotation((0, 0, 1), tw.sa["arm_deg"][k] - tw.sa["shown_deg"])
            out.append(about(R, piv) @ c0)
    tx = math.tan(tw.objs[SCANNER_CAM].data.angle_x / 2)
    return out, tx, tx * 0.75          # single capture 200 x 150 mm


def cmd_s1(tw, args):
    dg = bpy.context.evaluated_depsgraph_get()
    tw.hide_phone(True)
    spheres = sorted(ob.name for ob in tw.objs if ob.name.startswith(HEAD_SPHERES) and ob.type == "MESH")
    dirs = fibonacci(400)
    rep = {"spheres": {}, "poses": {}}
    poses = phone_poses(tw)
    modes = {m: intrinsics(m, 1.0) for m in ("photo", "video")}
    frame = tw.objs.get("gt_frame_mandible")
    fv = world_mesh(frame)[0][:: 40] if frame is not None else np.empty((0, 3))
    worst = {}
    for kind, name, phi, tilt in poses:
        tw.set_arm(phi, tilt)
        dg = bpy.context.evaluated_depsgraph_get()
        m = tw.camera()
        K, size, _ = modes[kind]
        row = {}
        for sname in spheres:
            c, r = sphere_of(tw.objs[sname])
            pts = c + dirs * r
            facing = (dirs * (m[:3, 3] - pts)).sum(1) > 0
            cand = facing & in_image(m, K, size, pts)
            vis = np.zeros(len(pts), bool)
            if cand.any():
                vis[cand] = visible(tw, dg, m[:3, 3], pts[cand], sname)
            frac = float(vis.sum() / max(facing.sum(), 1))
            row[sname] = round(frac, 3)
            worst[sname] = max(worst.get(sname, 0.0), frac)
        if len(fv):
            fi = in_image(m, K, size, fv)
            fvis = np.zeros(len(fv), bool)
            if fi.any():
                fvis[fi] = visible(tw, dg, m[:3, 3], fv[fi], "gt_frame_mandible", tol=0.5)
            row["gt_frame_mandible"] = round(float(fvis.mean()), 3)
        rep["poses"][name] = row
    rep["max_visible_fraction"] = {k: round(v, 3) for k, v in worst.items()}
    for sname in spheres:
        c, r = sphere_of(tw.objs[sname])
        rep["spheres"][sname] = {"centre": lst(c, 2), "radius": round(r, 2)}
    rep["candidates"] = s1_candidates(tw, poses, modes)
    tw.set_arm(tw.sa["shown_deg"], 0.0)
    (args.out / "s1.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("max_visible_fraction", "spheres")}, indent=1))
    print(json.dumps({k: v for k, v in rep["candidates"].items() if k != "feasible"}, indent=1))


def merged_bvh(objects):
    vs, fs, n = [], [], 0
    for ob in objects:
        v, f = world_mesh(ob)
        vs.append(v)
        fs.append(f + n)
        n += len(v)
    return BVHTree.FromPolygons(np.concatenate(vs).tolist(), np.concatenate(fs).tolist())


def s1_candidates(tw, poses, modes, r=7.5, reach=30.0, step=6.0):
    """Sphere centres on a grid around the head: clear of every surface by 1 mm, at most `reach` from the bone (skull
    or mandible, where a stalk can hold them; never on the silicone), in the scanner field at its stand-off from both
    scan positions with a free line of sight, and hidden from every phone pose (photo positions and video path).
    Returns counts, the feasible centres and a well spread scalene triple among them."""
    sc, o = tw.sc, tw.objs
    bone = [o[SKULL], o[MANDIBLE]]
    bb = np.concatenate([world_mesh(ob, evaluated=False)[0] for ob in bone])
    lo, hi = bb.min(0) - r - reach, bb.max(0) + r + reach
    others = []
    in_layer = set(ob.name for ob in tw.vl.objects)       # alternatives in excluded collections are not there
    for ob in o:
        if (ob.type != "MESH" or ob.hide_render or ob.name not in in_layer or ob in bone or ob.name.startswith(HIDDEN)
                or ob.name.startswith(HEAD_REF)):
            continue
        v = world_mesh(ob, evaluated=False)[0]
        if len(v) and (v.max(0) >= lo).all() and (v.min(0) <= hi).all():
            others.append(ob)
    bvh_bone, bvh_all = merged_bvh(bone), merged_bvh(bone + others)
    g = np.stack(np.meshgrid(*[np.arange(a, b, step) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    d_bone = np.array([bvh_bone.find_nearest(Vector(p))[3] for p in g])
    g = g[d_bone <= r + reach]
    d_all = np.array([(bvh_all.find_nearest(Vector(p))[3] or 1e9) for p in g])
    free = g[d_all >= r + 1.0]
    tw.set_arm(tw.sa["shown_deg"], 0.0)
    dg = bpy.context.evaluated_depsgraph_get()
    scan, tx, ty = scanner_poses(tw)
    seen = np.ones(len(free), bool)
    for m in scan:
        x = (free - m[:3, 3]) @ m[:3, :3]
        inside = (x[:, 2] > 290) & (x[:, 2] < 480) & (np.abs(x[:, 0] / x[:, 2]) < tx) & (np.abs(x[:, 1] / x[:, 2]) < ty)
        ok = np.zeros(len(free), bool)
        for j in np.flatnonzero(inside & seen):
            d = free[j] - m[:3, 3]
            L = np.linalg.norm(d)
            hit, *_ = sc.ray_cast(dg, Vector(m[:3, 3] + d / L * 10.0), Vector(d / L), distance=L - 10.0 - r)
            ok[j] = not hit
        seen &= ok
    cen = free[seen]
    dirs = np.r_[fibonacci(40) * r, np.zeros((1, 3))]
    hidden = np.ones(len(cen), bool)
    for kind, name, phi, tilt in poses:
        idx = np.flatnonzero(hidden)
        if not len(idx):
            break
        tw.set_arm(phi, tilt)
        dg = bpy.context.evaluated_depsgraph_get()
        m = tw.camera()
        K, size, _ = modes[kind]
        for j in idx:
            p = cen[j] + dirs
            facing = (dirs * (m[:3, 3] - p)).sum(1) >= 0
            cand = facing & in_image(m, K, size, p)
            if cand.any() and visible(tw, dg, m[:3, 3], p[cand]).any():
                hidden[j] = False
    feas = cen[hidden]
    best, best_area = None, 0.0
    rng = np.random.default_rng(0)
    for _ in range(50000 if len(feas) >= 3 else 0):
        t = feas[rng.choice(len(feas), 3, replace=False)]
        d = sorted(np.linalg.norm(t[[0, 1, 2]] - t[[1, 2, 0]], axis=1))
        if d[0] < 50 or d[2] > 110 or d[1] - d[0] < 5 or d[2] - d[1] < 5:
            continue
        area = np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) / 2
        if area > best_area:
            best, best_area = t, area
    return {"grid_near_bone": int(len(g)), "free": int(len(free)), "seen_by_scanner": int(seen.sum()),
            "hidden_from_phone": int(hidden.sum()),
            "seen_box": [lst(cen.min(0), 1), lst(cen.max(0), 1)] if len(cen) else None,
            "feasible_box": [lst(feas.min(0), 1), lst(feas.max(0), 1)] if len(feas) else None,
            "feasible": lst(feas, 1), "seen": lst(cen, 1),
            "triple": None if best is None else lst(best, 1),
            "triple_distances": None if best is None else lst(np.linalg.norm(best[[0, 1, 2]] - best[[1, 2, 0]], axis=1), 1)}


# --- drapes by cloth simulation (in a separate scene in metres)

def cmd_drapes(tw, args):
    out = args.out / "drapes"
    out.mkdir(parents=True, exist_ok=True)
    o = tw.objs
    proxies = {}
    names = [FACE, SKULL, MANDIBLE, SPLINT, "connector_skull", "phantom_post", "gt_frame_mandible", "marker_holder",
             "marker_plate", "base_board"] + [ob.name for ob in o if ob.name.startswith(CAP) and ob.type == "MESH"]
    if not args.head_spheres:        # drapes made for images without the head spheres do not rest on them
        names = [n for n in names if not n.startswith(HEAD_REF)]
    tw.show_head_spheres(args.head_spheres)
    for name in names:
        if name in o:
            v, f = world_mesh(o[name])
            limit = 30000 if name == FACE else 12000     # collision proxies (the cloth rests on the surgical cap)
            if len(f) > limit:
                v, f = decimate(v, f, limit)
            proxies[name] = (v, f)
    clear = []
    for ob in o:
        if not args.head_spheres and ob.name.startswith(HEAD_REF):
            continue
        if ob.type == "MESH" and not ob.hide_render and ob.name.startswith(DRAPE_CLEAR):
            v, f = world_mesh(ob)
            clear.append(BVHTree.FromPolygons(v.tolist(), f.tolist()))
    sim = bpy.data.scenes.new("synth_cloth")
    sim.unit_settings.system, sim.unit_settings.scale_length = "METRIC", 1.0
    sim.frame_start, sim.frame_end = 1, args.frames
    for name, (v, f) in proxies.items():
        me = bpy.data.meshes.new(f"col_{name}")
        me.from_pydata((v * 1e-3).tolist(), [], f.tolist())
        ob = bpy.data.objects.new(f"col_{name}", me)
        sim.collection.objects.link(ob)
        mod = ob.modifiers.new("collision", "COLLISION")
        ob.collision.thickness_outer, ob.collision.cloth_friction = 0.002, 40.0
        # double-sided: cap, mask and skull are thin shells whose inner faces look inwards; single-sided collision
        # pushed the cloth through them towards those faces
        ob.collision.use_culling = False
    face_v = proxies[FACE][0]
    # the cloth starts 4 mm above the highest point of the head (cap included: starting inside the collision
    # distance of the cap lets the cloth slip through the head)
    top = max(p[0][:, 2].max() for k, p in proxies.items() if k in (FACE, SKULL) or k.startswith(CAP))
    rng = np.random.default_rng(args.seed)
    kept, tries = 0, 0
    views = [(phi, 0.0) for phi in tw.sa["arm_deg"]]
    K, size, _ = intrinsics("photo", 0.125)
    base = coverage_counts(tw, views, K, size, args.out)
    dg = sim.view_layers[0].depsgraph
    hv, hf = [], []
    for k in [FACE, SKULL] + [n for n in proxies if n.startswith(CAP)]:
        hv.append(proxies[k][0] * 1e-3)
        hf.append(proxies[k][1] + sum(len(x) for x in hv[:-1]))
    head_bvh = BVHTree.FromPolygons(np.concatenate(hv).tolist(), np.concatenate(hf).tolist())
    full = [world_mesh(o[k]) for k in [FACE, SKULL] + [n for n in proxies if n.startswith(CAP)]]   # mm, not decimated
    offs = np.cumsum([0] + [len(x) for x, _ in full[:-1]])
    head_full = BVHTree.FromPolygons(np.concatenate([x for x, _ in full]).tolist(),
                                     np.concatenate([f + k for (_, f), k in zip(full, offs)]).tolist())
    sk = world_mesh(o[SKULL], evaluated=False)[0]
    head_centre = (sk.min(0) + sk.max(0)) / 2
    while kept < args.count and tries < args.count * 6:
        tries += 1
        lx, ly = rng.uniform(0.10, 0.22), rng.uniform(0.07, 0.15)
        nx, ny = max(int(lx / 0.005), 8), max(int(ly / 0.005), 8)
        gx, gy = np.meshgrid(np.linspace(-lx / 2, lx / 2, nx + 1), np.linspace(-ly / 2, ly / 2, ny + 1), indexing="ij")
        verts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
        idx = np.arange(gx.size).reshape(nx + 1, ny + 1)
        quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        # one edge rests on the top of the head (pinned on the front of the surgical cap or on the forehead: the drape
        # stays where it is laid), the rest hangs towards the face, to the front or to the phone side
        pin = np.array([rng.uniform(-20, 50), rng.uniform(20, 70), top + 4.0])
        th = math.radians(rng.uniform(-15, 70))
        d = np.array([math.sin(th), -math.cos(th), 0.0])          # from the pinned edge to the free edge
        centre = pin + d * (ly * 1e3 / 2 - 5.0)
        yl = -d
        xl = np.cross(yl, [0.0, 0.0, 1.0])
        world = (centre + (verts[:, :1] * xl + verts[:, 1:2] * yl) * 1e3) * 1e-3     # metres
        band = np.flatnonzero(verts[:, 1] > ly / 2 - 0.008)
        pinned = []
        for j in band:
            hit = head_bvh.ray_cast(Vector(world[j]), Vector((0.0, 0.0, -1.0)), 0.05)
            if hit[0] is not None:
                world[j, 2] = hit[0].z + 0.002
                pinned.append(int(j))
        if len(pinned) < 4:
            print(json.dumps({"try": tries, "rejected": "pinned edge off the head"}), flush=True)
            continue
        me = bpy.data.meshes.new("synth_cloth")
        me.from_pydata(world.tolist(), [], quads.tolist())
        cloth = bpy.data.objects.new("synth_cloth", me)
        sim.collection.objects.link(cloth)
        vg = cloth.vertex_groups.new(name="pin")
        vg.add(pinned, 1.0, "REPLACE")
        verts = world
        mod = cloth.modifiers.new("cloth", "CLOTH")
        st = mod.settings
        st.quality, st.mass, st.air_damping = 8, 0.15, 1.0
        st.vertex_group_mass, st.pin_stiffness = "pin", 1.0
        st.tension_stiffness = st.compression_stiffness = 20.0
        st.shear_stiffness, st.bending_stiffness = 10.0, float(rng.uniform(0.5, 3.0))
        cs = mod.collision_settings
        cs.distance_min, cs.use_self_collision, cs.self_distance_min = 0.0015, True, 0.001
        cs.collision_quality, cs.friction = 4, 40.0
        mod.point_cache.frame_start, mod.point_cache.frame_end = 1, args.frames
        t0 = time.perf_counter()
        for fr in range(1, args.frames + 1):
            sim.frame_set(fr)
        ev = cloth.evaluated_get(dg)
        em = ev.to_mesh()
        em.calc_loop_triangles()
        v = arr(em.vertices, "co", 3)
        f = arr(em.loop_triangles, "vertices", 3, np.int32)
        ev.to_mesh_clear()
        moved = len(v) == len(verts) and np.abs(v - verts).max() > 1e-6
        mw = mat(cloth.matrix_world)
        v = (v @ mw[:3, :3].T + mw[:3, 3]) * 1e3
        sim_s = time.perf_counter() - t0
        bpy.data.objects.remove(cloth)
        bpy.data.meshes.remove(me)
        if not moved:
            print(json.dumps({"try": tries, "rejected": "cloth did not move"}), flush=True)
            continue
        v, lifted, deep = lift_outside(v, head_full, head_centre)
        if deep > 0.01 * len(v):
            print(json.dumps({"try": tries, "rejected": "sank into the head", "lifted": lifted, "deep": deep}), flush=True)
            continue
        gap = min(min((t.find_nearest(Vector(p))[3] or 1e9) for p in v[::7]) for t in clear) if clear else 1e9
        if not np.isfinite(v).all() or v[:, 2].min() < face_v[:, 2].min() - 40 or gap < 5.0:
            print(json.dumps({"try": tries, "rejected": "clearance" if gap < 5.0 else "fell", "gap_mm": round(gap, 1)}), flush=True)
            continue
        tw.set_drape({"vertices": v, "faces": f, "color": srgb_to_linear(DRAPE_SRGB).tolist()})
        cov = coverage_counts(tw, views, K, size, args.out)
        tw.set_drape(None)
        frac = [1 - c / max(b, 1) for c, b in zip(cov, base)]
        ok = 0.10 <= float(np.mean(frac)) <= 0.40
        print(json.dumps({"try": tries, "sim_s": round(sim_s, 1), "cover": [round(x, 3) for x in frac], "kept": ok,
                          "gap_mm": round(gap, 1), "lifted": int(lifted), "box": [lst(v.min(0), 0), lst(v.max(0), 0)]}),
              flush=True)
        if ok:
            np.savez_compressed(out / f"drape_{kept:02d}.npz", vertices=v.astype(np.float32), faces=f,
                                cover=np.array(frac), size_m=np.array([lx, ly]), pin=pin, direction_deg=math.degrees(th),
                                lifted=lifted)
            kept += 1
    bpy.data.scenes.remove(sim)
    print(json.dumps({"drapes": kept, "tries": tries}))


def lift_outside(v, bvh, centre, gap=1.0, reach=12.0, rounds=4, deep=2.5):
    """Check after the cloth simulation: drape vertices that sank into the head (the cloth collides with decimated
    proxies; cap, mask and skull are thin shells) are moved out along the ray from the head centre, to `gap` mm beyond
    the surface they are under. A vertex is under a surface when the outward ray leaves it (normal along the ray) within `reach`.
    Returns the vertices, how many were moved and how many by more than `deep` mm."""
    v = np.array(v, np.float64)
    v0 = v.copy()
    lifted = np.zeros(len(v), bool)
    todo = np.arange(len(v))
    for _ in range(rounds):
        nxt = []
        for j in todo:
            d = v[j] - centre
            d /= np.linalg.norm(d)
            loc, nrm, _, _ = bvh.ray_cast(Vector(v[j]), Vector(d), reach)
            if loc is not None and np.dot(np.array(nrm), d) > 0:
                v[j] = np.array(loc) + d * gap
                lifted[j] = True
                nxt.append(j)
        todo = nxt
        if not todo:
            break
    return v, int(lifted.sum()), int((np.linalg.norm(v - v0, axis=1) > deep).sum())


def coverage_counts(tw, views, K, size, out):
    tw.label_mode(depth=False)
    tw.hide_phone(True)
    face_px = []
    tmp = out / "drapes" / "_coverage.png"
    for phi, tilt in views:
        tw.set_arm(phi, tilt)
        tw.set_camera(tw.camera(), K, size)
        tw.render(tmp)
        face_px.append(Twin.label_counts(tmp)["face"])
    tmp.unlink()
    return face_px


def main(argv):
    p = argparse.ArgumentParser(prog="synth_twin.py")
    p.add_argument("command", choices=("probe", "test", "meshes", "drapes", "s1", "render", "copy"))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--force-copy", action="store_true")
    p.add_argument("--sessions", default="0:1", help="range a:b of session indices")
    p.add_argument("--registrations", type=int, default=4)
    p.add_argument("--states", type=int, default=3, help="per registration, the first before the impact")
    p.add_argument("--video", type=int, default=16, help="video frames per state")
    p.add_argument("--scale", type=float, default=SCALE)
    p.add_argument("--samples", type=int, default=SAMPLES)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--model-mask", action="store_true")
    p.add_argument("--depth", action="store_true")
    p.add_argument("--head-spheres", action="store_true",
                   help="show the head spheres and their frame (render, drapes); hidden by default")
    p.add_argument("--pp-offset", type=float, nargs=2, default=(0.0, 0.0), help="px added to cx, cy (checks)")
    p.add_argument("--count", type=int, default=24, help="drapes to keep")
    p.add_argument("--frames", type=int, default=60, help="cloth simulation frames")
    args = p.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "copy":
        print(refresh_copy(SCENE, args.out / "twin_scene.blend", args.force_copy))
        return
    tw = open_twin(args.out, args.force_copy)
    {"probe": cmd_probe, "test": cmd_test, "meshes": cmd_meshes, "drapes": cmd_drapes, "s1": cmd_s1,
     "render": cmd_render}[args.command](tw, args)


if __name__ == "__main__":
    main(sys.argv)
