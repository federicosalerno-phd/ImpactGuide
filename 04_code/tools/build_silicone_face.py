"""Build the silicone face layer from the face mould (02_cad/solidworks/Assembly/face_mould).

The silicone is the cavity of the closed mould:
    inner surface (skull side)  = yellow face of Mould_lower
    outer surface (skin side)   = green face of Mould_upper
    eye and mouth openings      = outer walls of the three black cores (prisms along the mould axis y)
    back face                   = parting plane y = 0 (top of the lower mould plate)
Two details of the mould are not cast into the model:
    - at the neck the two halves leave a 0.2 mm gap over about 20 mm (flash): the lower face is pushed
      0.6 mm through the upper one there;
    - the openings of both faces leave 0.1 to 0.25 mm of clearance around the cores (flash sleeve):
      the rims are moved onto the cores and the surface follows within SNAP_RADIUS.

Steps:
1. faces: the SolidWorks tessellation leaves T-junction cracks between B-rep faces, so the green face is
   stitched; the yellow one (two mirrored halves with a 0.1 mm slot on the midline) is rebuilt from its
   right half, mirrored and bridged, then stitched;
2. solids: both faces closed into solids, silicone = green - yellow - cores (manifold3d);
3. remesh: the tessellation also carries folded micro triangles, so the solid is rebuilt from a voxel
   level set (OpenVDB, Blender), decimated with quadric edge collapse (pymeshlab) and smoothed (Taubin,
   sharp edges held) to remove the 1 mm facets of the SolidWorks tessellation;
4. checks (closed, manifold, one shell, three openings, no self intersections, deviation) and output.

Outputs: 02_cad/blender/silicone_face.glb (mould frame, metres, glTF y up, no normals: shade by angle)
and 02_cad/blender/face_mould.blend (moulds and silicone, face turned to the front view).
Intermediate PLY files go to C:/work/marker-impact. Run with the project environment:
    C:\\envs\\marker-impact\\Scripts\\python.exe 04_code/tools/build_silicone_face.py
Then 02_cad/blender/skull_face.blend (face and skull assembly, origin of the scene on the origin of
Skull.gltf, face placed by hand on the symmetry plane, mandible turning on the M5 axis):
    C:\\envs\\marker-impact\\Scripts\\python.exe 04_code/tools/build_silicone_face.py skull [--force]
Colours (resin face, bone PLA skull) and the cuts of the face (NECK_CUT, BACK_CUT: boolean modifier with the
cutters parented to the face) are refreshed in skull_face.blend, keeping the face placement, with:
    C:\\envs\\marker-impact\\Scripts\\python.exe 04_code/tools/build_silicone_face.py cuts
The Blender steps run this same file inside Blender in background.
"""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import bpy
except ImportError:
    bpy = None
if bpy is None:
    import manifold3d as mf
    import pymeshlab as ml
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "02_cad" / "solidworks" / "Assembly" / "face_mould" / "face_mould.gltf"
OUT_GLB = ROOT / "02_cad" / "blender" / "silicone_face.glb"
OUT_BLEND = ROOT / "02_cad" / "blender" / "face_mould.blend"
SKULL_SRC = ROOT / "02_cad" / "solidworks" / "Assembly" / "Skull" / "Skull.gltf"
SKULL_BLEND = ROOT / "02_cad" / "blender" / "skull_face.blend"
WORK = Path(r"C:\work\marker-impact")
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")

CORE_CUT_Y = -90.0        # the cores are cut here, where they are straight prisms along y, mm
CORE_Y = (-115.0, -50.0)  # y span of the core prisms (covers the silicone around the openings), mm
CORE_GROW = 0.01          # cores grown by this much so the cut never runs exactly on the snapped rims, mm
SNAP_RADIUS = 1.5         # the surface around a rim follows the rim snap within this distance, mm
NECK_Z = -108.0           # yellow faces below this z and facing -z form the neck contact face, mm
NECK_PUSH = 0.6           # how far the neck contact face is pushed through the green one, mm
BACK_Y = 2.0              # the yellow solid is extruded past the parting plane up to this y, mm
VOXEL = 0.1               # voxel size of the level set, mm
TRIANGLES = 600_000       # triangles of the final mesh
TAUBIN_STEPS = 50         # smoothing steps that remove the facets of the SolidWorks tessellation
COLOR = (0.86, 0.68, 0.58, 1.0)
FACE_START = (0.0, -30.0, 48.0)  # rough start of the face on the skull (Blender frame, mm); placed by hand
SCREW_GREY = (0.03, 0.03, 0.03, 1.0)  # dark grey (linear) of the M5 and M3 screws
RESIN_SRGB = (242, 152, 112)  # cured Reschimica RPRO 10, sampled from the product photo
BONE_SRGB = (227, 218, 201)   # bone PLA
# Cuts of the face, in the skull frame (mm), from marks drawn by hand on screenshots of the face:
NECK_CUT = (-64.165, 0.60485)  # neck corners: removed below Z = z0 + m |X| (the two marks averaged, symmetric)
BACK_CUT = (-39.206, 0.05106)  # back of the face: removed behind Y = y0 + k Z


# glTF input

def load_mould():
    """Return {(node name, material colour): (V in mm, F)} with node transforms applied."""
    g = json.loads(SRC.read_text())
    bufs = [(SRC.parent / b["uri"]).read_bytes() for b in g["buffers"]]
    ctype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}
    ncomp = {"SCALAR": 1, "VEC3": 3}

    def accessor(i):
        a = g["accessors"][i]
        bv = g["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        arr = np.frombuffer(bufs[bv["buffer"]], dtype=ctype[a["componentType"]],
                            count=a["count"] * ncomp[a["type"]], offset=off)
        return arr.reshape(-1, ncomp[a["type"]])

    def quat(q):
        x, y, z, w = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    parts = {}
    for n in g["nodes"]:
        if "mesh" not in n:
            continue
        R = quat(n["rotation"]) if "rotation" in n else np.eye(3)
        t = np.array(n.get("translation", [0.0, 0.0, 0.0]))
        for p in g["meshes"][n["mesh"]]["primitives"]:
            V = (accessor(p["attributes"]["POSITION"]).astype(float) @ R.T + t) * 1000.0
            F = accessor(p["indices"]).reshape(-1, 3).astype(np.int64)
            rgb = g["materials"][p["material"]]["pbrMetallicRoughness"]["baseColorFactor"][:3]
            parts[(n["name"], tuple(round(c, 2) for c in rgb))] = (V, F)
    return parts


# mesh helpers

def clean(F):
    return F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]


def weld(V, F, decimals=4):
    U, inv = np.unique(np.round(V, decimals), axis=0, return_inverse=True)
    return U, clean(inv.reshape(-1)[F])


def compact(V, F):
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def merge_groups(V, F, pairs):
    """Merge the vertex pairs (and their chains) into their centroid."""
    n = len(V)
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)),
                                  directed=False)
    V = np.stack([np.bincount(lab, V[:, k]) for k in range(3)], axis=1) / np.bincount(lab)[:, None]
    return V, clean(lab[F])


def edges_of(F):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return E, np.tile(np.arange(len(F)), 3), np.repeat(np.arange(3), len(F))


def boundary_mask(E, n):
    k = np.minimum(E[:, 0], E[:, 1]) * n + np.maximum(E[:, 0], E[:, 1])
    _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    return cnt[inv] == 1


def boundary_loops(F):
    """Directed boundary loops, longest first."""
    E, _, _ = edges_of(F)
    nxt = {}
    for a, c in E[boundary_mask(E, F.max() + 1)]:
        if a in nxt:
            raise ValueError("boundary vertex with two outgoing edges")
        nxt[a] = c
    loops, seen = [], set()
    for s in nxt:
        if s in seen:
            continue
        loop, v = [], s
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        if v != s:
            raise ValueError("open boundary chain")
        loops.append(np.array(loop))
    return sorted(loops, key=len, reverse=True)


def topology(F):
    """Non-manifold edges and same-direction duplicate edges."""
    E, _, _ = edges_of(F)
    _, cnt = np.unique(np.sort(E, axis=1), axis=0, return_counts=True)
    _, cnt_dir = np.unique(E, axis=0, return_counts=True)
    return int((cnt > 2).sum()), int((cnt_dir > 1).sum())


# crack stitching

def boundary_dirs(V, bE):
    nxt = dict(bE.tolist())
    prv = {b: a for a, b in bE.tolist()}
    bv = np.array(sorted(set(nxt) & set(prv)))
    d = V[[nxt[v] for v in bv]] - V[[prv[v] for v in bv]]
    return bv, d / (np.linalg.norm(d, axis=1)[:, None] + 1e-30)


def stitch(V, F, tol=0.05, merge_tol=0.03, cos_min=0.7, max_iter=50):
    """Close T-junction cracks: boundary vertices facing each other across a crack are merged, then
    each boundary vertex lying on an opposite boundary edge splits it."""
    E, _, _ = edges_of(F)
    bv, dv = boundary_dirs(V, E[boundary_mask(E, len(V))])
    pairs = cKDTree(V[bv]).query_pairs(merge_tol, output_type="ndarray")
    if len(pairs):
        pairs = pairs[np.einsum("ij,ij->i", dv[pairs[:, 0]], dv[pairs[:, 1]]) < -cos_min]
        adj = set(map(tuple, np.sort(E, axis=1).tolist()))
        pairs = np.array([(bv[i], bv[j]) for i, j in pairs if (min(bv[i], bv[j]), max(bv[i], bv[j])) not in adj])
        if len(pairs):
            n = len(V)
            _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                                                     shape=(n, n)), directed=False)
            first = np.full(lab.max() + 1, -1)
            for i in range(n - 1, -1, -1):
                first[lab[i]] = i
            F = clean(first[lab][F])
    for _ in range(max_iter):
        E, fid, slot = edges_of(F)
        bm = boundary_mask(E, len(V))
        bE, bf, bs = E[bm], fid[bm], slot[bm]
        bv, dv = boundary_dirs(V, bE)
        adj = set(map(tuple, np.sort(E, axis=1).tolist()))
        A, B = V[bE[:, 0]], V[bE[:, 1]]
        L = np.linalg.norm(B - A, axis=1)
        de = (B - A) / (L[:, None] + 1e-30)
        cands = cKDTree(V[bv]).query_ball_point((A + B) / 2, L / 2 + tol)
        inserts = {}
        for e, cl in enumerate(cands):
            if not cl:
                continue
            cl = np.array(cl)
            a, b, c = bE[e, 0], bE[e, 1], F[bf[e], (bs[e] + 2) % 3]
            ok = np.array([v not in (a, b, c) and (min(v, c), max(v, c)) not in adj for v in bv[cl]], bool)
            cl = cl[ok]
            if len(cl) == 0:
                continue
            p = V[bv[cl]] - A[e]
            t = p @ de[e]
            d = np.linalg.norm(p - np.outer(t, de[e]), axis=1)
            sel = (t > merge_tol) & (t < L[e] - merge_tol) & (d < tol) & (dv[cl] @ de[e] < -cos_min)
            if sel.any():
                inserts[e] = sorted(zip(t[sel], bv[cl][sel]))
        if not inserts:
            break
        done, new = set(), []
        for e, lst in inserts.items():
            f = bf[e]
            if f in done:
                continue
            done.add(f)
            s = bs[e]
            a, b, c = F[f, s], F[f, (s + 1) % 3], F[f, (s + 2) % 3]
            chain = [a] + [v for _, v in lst] + [b]
            new += [(chain[i], chain[i + 1], c) for i in range(len(chain) - 1)]
        keep = np.ones(len(F), bool)
        keep[list(done)] = False
        F = np.concatenate([F[keep], np.array(new, dtype=F.dtype)])
    return clean(F)


def collapse_short_boundary_edges(V, F, tol=0.002):
    """Collapse boundary edges shorter than tol (micro zig-zags of the tessellation on the rims)."""
    E, _, _ = edges_of(F)
    b = E[boundary_mask(E, len(V))]
    b = b[np.linalg.norm(V[b[:, 0]] - V[b[:, 1]], axis=1) < tol]
    if len(b):
        V, F = merge_groups(V, F, b)
        _, first = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
        F = F[np.sort(first)]
    return compact(V, F)


def clip_x(V, F, x0):
    """Keep the part with x >= x0, triangles cut exactly on the plane."""
    V = V.copy()
    V[np.abs(V[:, 0] - x0) < 1e-3, 0] = x0
    side = V[:, 0] >= x0
    keep = F[side[F].all(1)]
    cut = F[side[F].any(1) & ~side[F].all(1)]
    new_v, cache = [], {}

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k not in cache:
            p = V[a] + (x0 - V[a, 0]) / (V[b, 0] - V[a, 0]) * (V[b] - V[a])
            p[0] = x0
            cache[k] = len(V) + len(new_v)
            new_v.append(p)
        return cache[k]

    tris = []
    for f in cut:
        poly = []
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            if side[a]:
                poly.append(a)
            if side[a] != side[b]:
                poly.append(mid(a, b))
        tris += [(poly[0], poly[i], poly[i + 1]) for i in range(1, len(poly) - 1)]
    V = np.vstack([V] + ([np.array(new_v)] if new_v else []))
    F = clean(np.vstack([keep, np.array(tris, np.int64).reshape(-1, 3)]))
    # a crack crossing the plane leaves two cut points a few microns apart
    on = np.where(V[:, 0] == x0)[0]
    pairs = cKDTree(V[on]).query_pairs(0.002, output_type="ndarray")
    if len(pairs):
        V, F = merge_groups(V, F, on[pairs])
    return compact(V, F)


def mirror_bridge(V, F, x0):
    """Mirror the half x >= x0 about x = 0 and bridge the 2 x0 slot along the cut."""
    n = len(V)
    E, _, _ = edges_of(F)
    b = E[boundary_mask(E, n)]
    b = b[(V[b[:, 0], 0] == x0) & (V[b[:, 1], 0] == x0)]
    bridge = np.vstack([np.c_[b[:, 1], b[:, 0], b[:, 0] + n], np.c_[b[:, 1], b[:, 0] + n, b[:, 1] + n]])
    return np.vstack([V, V * [-1, 1, 1]]), np.vstack([F, F[:, ::-1] + n, bridge])


def green_surface(parts):
    V, F = collapse_short_boundary_edges(*weld(*parts[("Mould_upper-1", (0.0, 1.0, 0.0))]))
    return V, stitch(V, F)


def yellow_surface(parts, x0=0.15):
    """Right half mirrored: the two halves are mirror images, the midline carries a 0.1 mm slot."""
    V, F = collapse_short_boundary_edges(*weld(*parts[("Mould_lower-1", (1.0, 1.0, 0.0))]))
    V, F = clip_x(V, F, x0)
    return mirror_bridge(V, stitch(V, F), x0)


def split_loops(V, F):
    """Outer loop and the three hole loops."""
    loops = boundary_loops(F)
    if len(loops) != 4:
        raise RuntimeError(f"expected 4 boundary loops, found {len(loops)}")
    outer = max(loops, key=lambda l: np.ptp(V[l, 0]))
    return outer, [l for l in loops if l is not outer]


# cores

def to_manifold(V, F):
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
    m = mf.Manifold(mf.Mesh(vert_properties=np.ascontiguousarray(V, np.float32),
                            tri_verts=np.ascontiguousarray(F if vol > 0 else F[:, ::-1], np.uint32)))
    if m.status() != mf.Error.NoError:
        raise RuntimeError(f"not a manifold solid: {m.status()}")
    return m


def core_contours(parts):
    """Outer contour (x, z) of each black core, cut at mid depth where the cores are prisms along y."""
    V, F = weld(*parts[("Mould_lower-1", (0.25, 0.25, 0.25))])
    n = len(V)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]])
    _, lab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)), directed=False)
    to_xz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], float)      # (x, y, z) -> (x, z, -y)
    out = []
    for i in np.unique(lab):
        Vi, Fi = compact(V, F[lab[F[:, 0]] == i])
        if np.ptp(Vi[:, 0]) > 150:          # the seal cord around the plate, not a core
            continue
        polys = to_manifold(Vi, Fi).transform(to_xz).slice(-CORE_CUT_Y).to_polygons()
        out.append(np.asarray(max(polys, key=lambda p: abs(mf.CrossSection([p]).area())), float))
    return out


def closest_on_polygon(P, C):
    """Closest points of 2D points P on the closed polygon C."""
    A, AB = C, np.roll(C, -1, axis=0) - C
    out = np.empty_like(P)
    for i, p in enumerate(P):
        t = np.clip(((p - A) * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-30), 0, 1)
        Q = A + t[:, None] * AB
        out[i] = Q[np.argmin(((Q - p) ** 2).sum(1))]
    return out


def snap_holes(V, F, contours):
    """Move the hole rims onto the core contours; the surface follows within SNAP_RADIUS."""
    _, holes = split_loops(V, F)
    V = V.copy()
    rim = np.concatenate(holes)
    shift = np.zeros((len(rim), 3))
    k = 0
    for l in holes:
        c = min(contours, key=lambda C: np.linalg.norm(C.mean(0) - V[l][:, [0, 2]].mean(0)))
        shift[k:k + len(l), [0, 2]] = closest_on_polygon(V[l][:, [0, 2]], c) - V[l][:, [0, 2]]
        k += len(l)
    d, j = cKDTree(V[rim]).query(V, distance_upper_bound=SNAP_RADIUS)
    near = np.isfinite(d)
    V[near] += ((1 - d[near] / SNAP_RADIUS) ** 2)[:, None] * shift[j[near]]
    return V, float(np.linalg.norm(shift, axis=1).max())


def core_prisms(contours):
    """Cores as prisms along y from their outer contours, grown by CORE_GROW."""
    out = []
    for c in contours:
        cs = mf.CrossSection([c], mf.FillRule.Positive)
        if cs.area() <= 0:
            cs = mf.CrossSection([c[::-1]], mf.FillRule.Positive)
        m = mf.Manifold.extrude(cs.offset(CORE_GROW, mf.JoinType.Miter), CORE_Y[1] - CORE_Y[0])
        # local (X, Y, Z) -> world (x, y, z) = (X, -Z + y1, Y): a proper rotation
        out.append(m.transform(np.array([[1, 0, 0, 0], [0, 0, -1, CORE_Y[1]], [0, 1, 0, 0]], float)))
    return out


# solids

def triangulate_loop(P2):
    """Triangles over a closed 2D polygon, indices into P2 (either orientation)."""
    x, y = P2[:, 0], P2[:, 1]
    if (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() > 0:
        return np.asarray(mf.triangulate([np.ascontiguousarray(P2, dtype=np.float64)]), dtype=np.int64)
    T = np.asarray(mf.triangulate([np.ascontiguousarray(P2[::-1], dtype=np.float64)]), dtype=np.int64)
    return len(P2) - 1 - T


def oriented_patch(T, loop, F_surface):
    """Map patch triangles to vertex ids and orient them against the surface boundary."""
    T = loop[T]
    surf = set(map(tuple, edges_of(F_surface)[0].tolist()))
    a, b = (loop[0], loop[1]) if (loop[0], loop[1]) in surf else (loop[1], loop[0])
    return T if (b, a) in set(map(tuple, edges_of(T)[0].tolist())) else T[:, ::-1]


def green_solid(V, F):
    """Region enclosed by the green face and the parting plane, openings filled inside the cores."""
    outer, holes = split_loops(V, F)
    if np.abs(V[outer, 1]).max() > 1e-6:
        raise RuntimeError("green outer loop off the parting plane")
    patches = [oriented_patch(triangulate_loop(V[l][:, [0, 2]]), l, F) for l in [outer] + holes]
    return to_manifold(V, np.vstack([F] + patches))


def yellow_solid(V, F):
    """Lower mould face extruded past the parting plane, neck contact face pushed through the green one."""
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    neck = (V[F].mean(1)[:, 2] < NECK_Z) & (N[:, 2] / np.linalg.norm(N, axis=1) < -0.95)
    V = V.copy()
    V[np.unique(F[neck]), 2] -= NECK_PUSH
    outer, holes = split_loops(V, F)
    patches = [oriented_patch(triangulate_loop(V[l][:, [0, 2]]), l, F) for l in holes]
    top = V[outer].copy()
    top[:, 1] = BACK_Y
    ring = len(V) + np.arange(len(outer))
    V = np.vstack([V, top])
    walls = np.vstack([np.c_[outer, np.roll(outer, -1), np.roll(ring, -1)],
                       np.c_[outer, np.roll(ring, -1), ring]])
    if (outer[0], outer[1]) in set(map(tuple, edges_of(F)[0].tolist())):
        walls = walls[:, ::-1]
    cap = oriented_patch(triangulate_loop(top[:, [0, 2]]), ring, walls)
    return to_manifold(V, np.vstack([F, walls, cap] + patches))


def silicone_solid():
    parts = load_mould()
    contours = core_contours(parts)
    VG, FG = green_surface(parts)
    VY, FY = yellow_surface(parts)
    VG, sg = snap_holes(VG, FG, contours)
    VY, sy = snap_holes(VY, FY, contours)
    print(f"rims moved onto the cores by up to {max(sg, sy):.2f} mm")
    s = mf.Manifold.batch_boolean([green_solid(VG, FG), yellow_solid(VY, FY)] + core_prisms(contours),
                                  mf.OpType.Subtract)
    m = s.to_mesh()
    return np.asarray(m.vert_properties, float)[:, :3], np.asarray(m.tri_verts, np.int64)


# remesh and checks

def save_ply(path, V, F):
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(V, F))
    ms.save_current_mesh(str(path), binary=True, save_vertex_normal=False)


def run_blender(*args, blend=None):
    cmd = [str(BLENDER), "-b", "--factory-startup"] + ([str(blend)] if blend else []) + \
          ["--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--"] + [str(a) for a in args]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def self_intersecting(V, F):
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(V, F))
    ms.compute_selection_by_self_intersections_per_face()
    return ms.current_mesh().face_selection_array().astype(bool)


def inverted_faces(V, F):
    """Faces whose normal points against the normals of their edge neighbours."""
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N /= np.linalg.norm(N, axis=1)[:, None] + 1e-30
    E, fid, _ = edges_of(F)
    key = np.minimum(E[:, 0], E[:, 1]) * len(V) + np.maximum(E[:, 0], E[:, 1])
    o = np.argsort(key)
    k, f = key[o], fid[o]
    same = k[:-1] == k[1:]
    a, c = f[:-1][same], f[1:][same]
    acc = np.zeros_like(N)
    np.add.at(acc, a, N[c])
    np.add.at(acc, c, N[a])
    return np.where(np.einsum("ij,ij->i", N, acc) < 0)[0]


def unfold(V, F, max_iter=10):
    """Relax folded vertices and their ring on the tangent plane until no fold is left."""
    E, _, _ = edges_of(F)
    n = len(V)
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel()
    V = V.copy()
    for _ in range(max_iter):
        bad = np.concatenate([F[inverted_faces(V, F)].ravel(), F[self_intersecting(V, F)].ravel()])
        if len(bad) == 0:
            return V
        ring = np.unique(F[np.isin(F, np.unique(bad)).any(1)])
        for _ in range(5):
            N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
            VN = np.zeros_like(V)
            for i in range(3):
                np.add.at(VN, F[:, i], N)
            VN = VN[ring] / (np.linalg.norm(VN[ring], axis=1)[:, None] + 1e-30)
            d = (A[ring] @ V) / deg[ring][:, None] - V[ring]
            V[ring] += d - np.einsum("ij,ij->i", d, VN)[:, None] * VN
    raise RuntimeError("folds left after relaxation")


def crease_vertices(V, F, deg=35):
    """Vertices on edges whose faces turn by more than deg, with their one-ring."""
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N /= np.linalg.norm(N, axis=1)[:, None] + 1e-30
    E, fid, _ = edges_of(F)
    key = np.minimum(E[:, 0], E[:, 1]) * len(V) + np.maximum(E[:, 0], E[:, 1])
    o = np.argsort(key)
    k, f, e = key[o], fid[o], E[o]
    same = k[:-1] == k[1:]
    turn = np.einsum("ij,ij->i", N[f[:-1][same]], N[f[1:][same]]) < np.cos(np.radians(deg))
    return np.unique(F[np.isin(F, np.unique(e[:-1][same][turn])).any(1)])


def taubin(V, F, fixed, steps=TAUBIN_STEPS):
    """Low-pass smoothing without shrinkage (lambda 0.5, mu -0.53) of the free vertices."""
    E, _, _ = edges_of(F)
    n = len(V)
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    W = A.multiply(1 / np.asarray(A.sum(1))).tocsr()
    free = np.ones(n, bool)
    free[fixed] = False
    V = V.copy()
    for _ in range(steps):
        for f in (0.5, -0.53):
            V[free] += f * ((W @ V)[free] - V[free])
    return V


def decimate(path):
    """Quadric edge collapse of the level set, folds relaxed, facets of the source tessellation smoothed
    away with the sharp edges held."""
    ms = ml.MeshSet()
    ms.load_new_mesh(str(path))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=TRIANGLES, qualitythr=0.5, preservenormal=True,
                                                preservetopology=True, optimalplacement=True, planarquadric=True,
                                                planarweight=0.002, autoclean=True)
    m = ms.current_mesh()
    V, F = m.vertex_matrix().copy(), m.face_matrix().astype(np.int64)
    V = unfold(V, F)
    return unfold(taubin(V, F, crease_vertices(V, F)), F), F


def report(V, F, ref):
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(V, F))
    t = ms.get_topological_measures()
    T = V[F]
    ang = []
    for i in range(3):
        a, b = T[:, (i + 1) % 3] - T[:, i], T[:, (i + 2) % 3] - T[:, i]
        ang.append(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b) /
                                                (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)), -1, 1))))
    amin = np.min(ang, axis=0)
    ms.add_mesh(ml.Mesh(*ref))
    dev = [ms.get_hausdorff_distance(sampledmesh=a, targetmesh=b, samplevert=True, sampleface=True,
                                     samplenum=2 * len(F), maxdist=ml.PureValue(5.0)) for a, b in ((0, 1), (1, 0))]
    return {"triangles": len(F), "vertices": len(V),
            "volume_cm3": round(float(np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum()) / 6000, 2),
            "boundary_edges": t["boundary_edges"], "non_manifold_edges": t["non_two_manifold_edges"],
            "non_manifold_vertices": t["non_two_manifold_vertices"], "shells": t["connected_components_number"],
            "genus": t["genus"], "self_intersecting_faces": int(self_intersecting(V, F).sum()),
            "folded_faces": len(inverted_faces(V, F)),
            "edge_mm_p1_p50_p99": np.percentile(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), [1, 50, 99]).round(2).tolist(),
            "min_angle_deg_p1": round(float(np.percentile(amin, 1)), 1), "triangles_below_15deg": int((amin < 15).sum()),
            "deviation_mm_mean_max": [round(max(d["mean"] for d in dev), 4), round(max(d["max"] for d in dev), 3)]}


# glb output

def write_glb(path, V, F):
    pos = np.ascontiguousarray(V / 1000.0, dtype="<f4")
    idx = np.ascontiguousarray(F, dtype="<u4")
    ib, pb = idx.tobytes(), pos.tobytes()
    pad = (-len(ib)) % 4
    blob = ib + b"\0" * pad + pb
    doc = {
        "asset": {"version": "2.0", "generator": "build_silicone_face.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "silicone_face"}],
        "meshes": [{"name": "silicone_face",
                    "primitives": [{"attributes": {"POSITION": 1}, "indices": 0, "material": 0}]}],
        "materials": [{"name": "silicone", "pbrMetallicRoughness": {
            "baseColorFactor": list(COLOR), "metallicFactor": 0.0, "roughnessFactor": 0.5}}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(ib), "target": 34963},
                        {"buffer": 0, "byteOffset": len(ib) + pad, "byteLength": len(pb), "target": 34962}],
        "accessors": [{"bufferView": 0, "componentType": 5125, "count": idx.size, "type": "SCALAR",
                       "min": [int(idx.min())], "max": [int(idx.max())]},
                      {"bufferView": 1, "componentType": 5126, "count": len(pos), "type": "VEC3",
                       "min": pos.min(0).tolist(), "max": pos.max(0).tolist()}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    blob += b"\0" * ((-len(blob)) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


def build():
    V0, F0 = silicone_solid()
    solid, voxel = WORK / "silicone_solid.ply", WORK / "silicone_voxel.ply"
    save_ply(solid, V0, F0)
    run_blender("voxel", solid, voxel, VOXEL)
    V, F = decimate(voxel)
    r = report(V, F, (V0, F0))
    print(json.dumps(r))
    if r["boundary_edges"] or r["non_manifold_edges"] or r["non_manifold_vertices"] or r["shells"] != 1 \
            or r["genus"] != 3 or r["self_intersecting_faces"] or r["folded_faces"] or r["deviation_mm_mean_max"][1] > 0.25:
        raise RuntimeError("the silicone mesh failed the checks")
    write_glb(OUT_GLB, V, F)
    for p in (solid, voxel):
        p.unlink()
    run_blender("scene", OUT_GLB, OUT_BLEND)
    print("written", OUT_GLB, OUT_BLEND)


def build_skull(force=False):
    """skull_face.blend from face_mould.blend; the face is placed there by hand, so it is not overwritten."""
    if SKULL_BLEND.exists() and not force:
        raise SystemExit(f"{SKULL_BLEND} exists and holds the face placed by hand: add --force to rebuild it")
    run_blender("skull", SKULL_BLEND, blend=OUT_BLEND)
    print("written", SKULL_BLEND)


# Blender side

def blender_voxel(src, dst, size):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.ply_import(filepath=src)
    ob = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("voxel", "REMESH")
    mod.mode, mod.voxel_size, mod.adaptivity = "VOXEL", float(size), 0.0
    bpy.ops.object.modifier_apply(modifier="voxel")
    bpy.ops.wm.ply_export(filepath=dst, export_selected_objects=True, export_normals=False, export_uv=False,
                          export_colors="NONE", ascii_format=False, export_triangulated_mesh=True)


def blender_scene(glb, blend):
    """Moulds and silicone in three collections, the face turned to the front view (-Y), upper mould hidden."""
    import math
    from mathutils import Euler, Vector
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    scene.unit_settings.length_unit = "MILLIMETERS"
    root = bpy.data.objects.new("mould_frame", None)
    root.rotation_euler = (-math.pi / 2, 0, 0)        # glTF frame imported: face towards -Z, top towards -Y
    scene.collection.objects.link(root)
    names = {(1.0, 0.0, 0.0): "mould_lower", (1.0, 1.0, 0.0): "mould_lower_face", (0.25, 0.25, 0.25): "cores",
             (1.0, 1.0, 1.0): "mould_lower_midline", (0.0, 1.0, 0.0): "mould_upper_face",
             (1.0, 0.75, 1.0): "mould_upper"}

    def into(collection_name, path):
        coll = bpy.data.collections.new(collection_name)
        scene.collection.children.link(coll)
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(path))
        new = [o for o in bpy.data.objects if o not in before]
        for o in new:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
        return new

    for o in into("mould", SRC):
        if o.type != "MESH":
            bpy.data.objects.remove(o)
            continue
        mw = o.matrix_world.copy()
        o.parent = root
        o.matrix_parent_inverse = root.matrix_world.inverted()
        o.matrix_world = root.matrix_world @ mw
        name = "mould_lower" if o.name.startswith("Mould_lower") else "mould_upper"
        coll = bpy.data.collections.get(name) or bpy.data.collections.new(name)
        if coll.name not in scene.collection.children:
            scene.collection.children.link(coll)
        o.users_collection[0].objects.unlink(o)
        coll.objects.link(o)
        o.name = name
        for mat in o.data.materials:
            rgba = mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value
            mat.diffuse_color = rgba
            mat.name = names.get(tuple(round(c, 2) for c in rgba[:3]), mat.name)
    bpy.data.collections.remove(bpy.data.collections["mould"])
    for o in into("silicone_face", glb):
        if o.type != "MESH":
            continue
        o.parent = root
        o.data.shade_smooth()
        o.data.set_sharp_from_angle(angle=math.radians(35))
        for mat in o.data.materials:
            mat.diffuse_color = COLOR
    layer = bpy.context.view_layer.layer_collection.children
    layer["mould_upper"].hide_viewport = True
    bpy.context.view_layer.update()
    sil = bpy.data.objects["silicone_face"]
    centre = sum((sil.matrix_world @ Vector(c) for c in sil.bound_box), Vector()) / 8
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            space = area.spaces.active
            space.shading.type = "SOLID"
            space.shading.color_type = "MATERIAL"
            space.clip_start, space.clip_end = 0.001, 10.0
            r3d = space.region_3d
            r3d.view_location = centre
            r3d.view_distance = 0.55
            r3d.view_rotation = Euler((math.radians(75), 0, math.radians(-30))).to_quaternion()
            r3d.view_perspective = "PERSP"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True)


def frame_views(location, distance):
    import math
    from mathutils import Euler
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                r3d = area.spaces.active.region_3d
                r3d.view_location, r3d.view_distance = location, distance
                r3d.view_rotation = Euler((math.radians(75), 0, math.radians(-30))).to_quaternion()


def blender_skull_scene(out):
    """Run on face_mould.blend: moulds removed, skull assembly imported with its origin on the scene origin.

    The face keeps the symmetry plane of the skull (x locked, rotation only about x) and starts at FACE_START;
    the mandible turns about the axis of the two M5 screws (origin on the axis, location locked, rotation
    only about x, assembled angle kept in the custom property closed_angle_deg). M5 and M3 screws dark grey."""
    import math
    from mathutils import Matrix, Vector
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    for name in ("mould_lower", "mould_upper"):
        coll = bpy.data.collections.get(name)
        if coll is not None:
            for o in list(coll.objects):
                bpy.data.objects.remove(o)
            bpy.data.collections.remove(coll)
    face = bpy.data.objects["silicone_face"]
    mw = face.matrix_world.copy()
    face.parent = None
    face.data.transform(mw)
    face.matrix_world = Matrix.Translation(Vector(FACE_START) / 1000)
    if bpy.data.objects.get("mould_frame"):
        bpy.data.objects.remove(bpy.data.objects["mould_frame"])
    face.rotation_mode = "XYZ"
    face.lock_location = (True, False, False)
    face.lock_rotation = (False, True, True)

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(SKULL_SRC))
    imported = [o for o in bpy.data.objects if o not in before]
    new = [o for o in imported if o.type == "MESH"]
    for o in new:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    for o in imported:
        if o.type != "MESH":
            bpy.data.objects.remove(o)

    def put(o, collection_name, name):
        coll = bpy.data.collections.get(collection_name)
        if coll is None:
            coll = bpy.data.collections.new(collection_name)
            scene.collection.children.link(coll)
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)
        o.name = o.data.name = name

    for o in new:
        for mat in o.data.materials:
            mat.diffuse_color = mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value
    grey = bpy.data.materials.new("screw_dark_grey")
    bsdf = grey.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = SCREW_GREY
    bsdf.inputs["Metallic"].default_value = 0.8
    bsdf.inputs["Roughness"].default_value = 0.35
    grey.diffuse_color = SCREW_GREY
    screws = sorted((o for o in new if o.name.startswith("socket head cap screw")),
                    key=lambda o: (o.matrix_world.translation.x, o.matrix_world.translation.z))
    washers = sorted((o for o in new if o.name.startswith("Washer")), key=lambda o: o.matrix_world.translation.x)
    count = {"M5": 0, "M3": 0}
    m5 = []
    for o in screws:
        size = "M5" if sorted(o.dimensions)[0] > 0.007 else "M3"     # head 8.5 mm against 5.5 mm
        count[size] += 1
        put(o, "screws", f"screw_{size}_{count[size]}")
        o.data.materials.clear()
        o.data.materials.append(grey)
        o.data.polygons.foreach_set("material_index", [0] * len(o.data.polygons))
        if size == "M5":
            m5.append(o)
    for i, o in enumerate(washers, 1):
        put(o, "washers", f"washer_{i}")
    for o in new:
        if o.name.startswith("Skull_model"):
            put(o, "skull", "skull")
        elif o.name.startswith("Mandible_model"):
            put(o, "mandible", "mandible")
        elif o.name.startswith("connector_skull"):
            put(o, "connector", "connector_skull")
    for obj, name in (("skull", "bone"), ("connector_skull", "connector"), ("washer_1", "washer")):
        bpy.data.objects[obj].data.materials[0].name = name

    if len(m5) != 2:
        raise RuntimeError(f"expected two M5 screws on the jaw axis, found {len(m5)}")
    ends = [o.matrix_world @ (sum((Vector(c) for c in o.bound_box), Vector()) / 8) for o in m5]
    axis = (ends[1] - ends[0]).normalized()
    if abs(axis.x) < 0.9999:
        raise RuntimeError(f"the M5 axis is not along x: {tuple(axis)}")
    pivot = (ends[0] + ends[1]) / 2
    jaw = bpy.data.objects["mandible"]
    jaw.rotation_mode = "XYZ"
    jaw.data.transform(Matrix.Translation(-(jaw.matrix_world.inverted() @ pivot)))
    jaw.location = pivot
    jaw.lock_location = (True, True, True)
    jaw.lock_rotation = (False, True, True)
    jaw.lock_scale = (True, True, True)
    jaw["closed_angle_deg"] = round(math.degrees(jaw.rotation_euler.x), 4)

    bpy.data.orphans_purge(do_recursive=True)
    skull = bpy.data.objects["skull"]
    frame_views(sum((skull.matrix_world @ Vector(c) for c in skull.bound_box), Vector()) / 8, 0.6)
    face_finish()
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True)


def srgb(c):
    c = np.asarray(c, float) / 255
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)) + (1.0,)


def face_finish():
    """Colours (resin face, bone PLA skull and mandible) and the cuts of the face: two prisms in the skull frame,
    parented to the face, subtracted by a boolean modifier (cutters hidden in the collection face_cuts)."""
    from mathutils import Matrix
    face = bpy.data.objects["silicone_face"]
    resin, bone = srgb(RESIN_SRGB), srgb(BONE_SRGB)
    for mat, rgba in ((face.data.materials[0], resin), (bpy.data.objects["skull"].data.materials[0], bone)):
        mat.diffuse_color = rgba
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = rgba
    face.data.materials[0].node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.45
    bpy.context.scene.view_settings.view_transform = "Standard"     # colours on screen as sampled (sRGB)

    coll = bpy.data.collections.get("face_cuts")
    if coll is None:
        coll = bpy.data.collections.new("face_cuts")
        bpy.context.scene.collection.children.link(coll)
    for o in list(coll.objects):
        bpy.data.objects.remove(o)

    def prism(name, outline, axis, span):
        """Closed prism of a 2D outline (mm) extruded along world axis 0 (x) or 1 (y) over span (mm)."""
        import bmesh
        n = len(outline)
        verts = []
        for a in span:
            for u, v in outline:
                p = [0.0, 0.0, 0.0]
                p[axis], p[1 - axis], p[2] = a / 1000, u / 1000, v / 1000
                verts.append(p)
        faces = [tuple(range(n)), tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(name, me)
        coll.objects.link(o)
        o.parent = face
        o.matrix_parent_inverse = face.matrix_world.inverted()
        o.display_type, o.hide_render = "WIRE", True
        o.hide_set(True)
        return o

    z0, m = NECK_CUT
    far = 250.0
    prism("cut_neck", [(-far, z0 + m * far), (-far, -400.0), (far, -400.0), (far, z0 + m * far), (0.0, z0)], 1, (-400.0, 300.0))
    y0, k = BACK_CUT
    prism("cut_back", [(y0 - 400 * k, -400.0), (400.0, -400.0), (400.0, 400.0), (y0 + 400 * k, 400.0)], 0, (-300.0, 300.0))
    for mod in [m_ for m_ in face.modifiers if m_.type == "BOOLEAN"]:
        face.modifiers.remove(mod)
    mod = face.modifiers.new("cuts", "BOOLEAN")
    mod.operation, mod.operand_type, mod.collection, mod.solver = "DIFFERENCE", "COLLECTION", coll, "MANIFOLD"


def blender_face_cuts(out):
    """Run on skull_face.blend: colours and cuts refreshed in place, the face placement kept."""
    bpy.context.preferences.filepaths.save_version = 0
    face_finish()
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True)


if __name__ == "__main__":
    if bpy is None:
        if sys.argv[1:2] == ["skull"]:
            build_skull(force="--force" in sys.argv)
        elif sys.argv[1:2] == ["cuts"]:
            run_blender("cuts", SKULL_BLEND, blend=SKULL_BLEND)
            print("written", SKULL_BLEND)
        else:
            build()
    else:
        args = sys.argv[sys.argv.index("--") + 1:]
        {"voxel": blender_voxel, "scene": blender_scene, "skull": blender_skull_scene,
         "cuts": blender_face_cuts}[args[0]](*args[1:])
