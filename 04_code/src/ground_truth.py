"""Ground truth from the scanner: sphere centres, rig poses, true registration and impact displacement.

Six white matte spheres D 15 (gt_sphere.SLDPRT) on two scalene rigs, so that the distances inside a rig label its
spheres: three on the marker holder (S_M, 79.3 / 47.1 / 61.6 mm) and three on the mandible frame (S_B, 52.0 / 44.2 / 57.3 mm).
The rig templates RIGS are the design centres of the sphere seats; calibrate_rigs() gives measured ones.

Frames. Poses are 4 x 4 matrices T_a_b (frame a from frame b) in mm. M is the ChArUco board as in marker_pose.py
(origin at the outer corner of the first square, x along the printed columns, y along the rows, z into the board).
Two frames of the head, both rigid with it:
    mandible  the local frame of the mandible model (origin on the hinge axis of the two M5 screws, x to the patient's
              left, y down, z backwards): the frame of the head rig template and of the target columns x, y, z of
              03_data/processed/targets/mandible_targets.csv;
    B         the head frame of the rest of the pipeline and of the AR (coordinates of the bench scene): the frame of
              the face and mandible meshes drawn by the AR and of the target columns x_B, y_B, z_B.
T_B_mandible (03_data/processed/targets/mandible_frame.json, tools/build_targets.py) links them. The rig templates are
given in M and in the mandible frame, so the nominal X = T_M_SM and Y = T_SB_mandible are the identity; the calibrated
ones replace them. For scan k:

    G_k = T_SM_SB = T_scan_SM^-1 T_scan_SB,   T_M_mandible,k = X G_k Y,   T_M_B,k = T_M_mandible,k T_B_mandible^-1
    dT_k = T_M_mandible,k-1^-1 T_M_mandible,k  (in the mandible frame; T_B_mandible dT_k T_B_mandible^-1 in B)

and the TRE of the registration left at k-1 on a target p (mandible frame) is |p - dT_k p|, the same in either frame.
A result file holds both registrations, "T_M_mandible" and "T_M_B" (the one the AR reads); files written before the
two names existed hold the mandible frame under "T_M_B", and load_result() reads them as such.

One call per scan for the app: measure(cloud, reference) returns centres, residuals and the displacement from the
reference scan. It fits each sphere with its calibrated radius: the scanner sees the spheres only in part, where a
free radius trades off against the depth of the centre and the error at the targets about doubles. The radii, and
the rig templates, are measured once (calibrate: three scans or more without an impact, the bench turned between them)
and saved in CALIBRATION_JSON, which every measure() reads; without it measure() says so in its warnings.
synthetic_scan() simulates an EinScan-SP scan of the two rigs.

Usage:
    python ground_truth.py calibrate scan1.ply scan2.ply scan3.ply [--out rig_calibration.json]
    python ground_truth.py measure scan.ply [scan2.ply] [--reference ref.json] [--out result.json]
    python ground_truth.py selftest [--trials 200] [--noise 0.05] [--fixed-radius] [--separate] [--dense]
    python ground_truth.py synth OUTDIR [--impacts 3]
    python ground_truth.py test
"""
import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import trials                   # the protocol: the arm positions the scans are taken from
from scipy.spatial.transform import Rotation

SPHERE_D = 15.0
RIGS = {
    # the final spheres of the bench scene (gt_sphere_marker_k of the square P1 holder, gt_sphere_mandible_k): the
    # marker's, 79.3 / 47.1 / 61.6 mm apart, in the board frame of the nominal scene (SCENE) placed where the bench
    # has them relative to the head; the board's own frame (X = T_M_SM) comes with the rig calibration
    "marker": np.array([[-0.5327, -46.9175, 6.9247], [36.1247, 22.6595, -3.0482], [40.9305, -17.219, -27.5801]]),
    # head rig, 52.0 / 44.2 / 57.3 mm apart, in the mandible frame (not B)
    "mandible": np.array([[-72.0, 101.666, -22.1712], [-24.0, 104.4929, -42.0711], [-20.0, 102.67, 1.8911]]),
}
# Clinical targets in the mandible frame (tools/build_targets.py: columns id, role, x, y, z); until that table exists,
# placeholders: extreme points of the bench mandible model
TARGETS_DIR = Path(__file__).resolve().parents[2] / "03_data" / "processed" / "targets"
TARGETS_CSV = TARGETS_DIR / "mandible_targets.csv"
FRAME_JSON = TARGETS_DIR / "mandible_frame.json"       # T_B_mandible
# the rigs as measured (calibrate: templates and the radius of each sphere), what measure() uses by default
CALIBRATION_JSON = TARGETS_DIR / "rig_calibration.json"
CALIBRATED = "calibrated"       # measure(rigs=, radii=): the saved calibration, else the design and free radii
MIN_CALIBRATION_SCANS = 3       # scans without impact that a calibration takes at least
TARGET_ROLES = ("proposed",)                           # the targets of the TRE; the table also has optional ones
TARGETS_MANDIBLE = {
    "menton": (0.0, 94.0, -80.7),
    "infradentale": (0.6, 53.2, -90.5),
    "gonion_l": (45.9, 69.9, -3.9),
    "gonion_r": (-45.9, 69.9, -3.9),
    "condylion_l": (54.9, -5.0, 0.2),
    "condylion_r": (-54.9, -5.0, 0.2),
}


def load_targets(path=TARGETS_CSV, roles=TARGET_ROLES):
    """Clinical targets in the mandible frame, {id: (x, y, z)}, from the columns x, y, z of the targets table.

    Args:
        path: the table of tools/build_targets.py.
        roles: the roles to keep (column "role": proposed, optional, alternative); None keeps every row. A table of
            the older format, with a 0/1 column "proposed" instead of "role", keeps its rows marked 1 when "proposed"
            is among the roles.
    Returns:
        The targets of the table, or the placeholders TARGETS_MANDIBLE when the table does not exist.
    Raises:
        ValueError: no row of the table has one of the roles.
    """
    if not Path(path).exists():
        return dict(TARGETS_MANDIBLE)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if roles is not None and rows:
        roles = {roles} if isinstance(roles, str) else set(roles)
        if "role" in rows[0]:
            rows = [r for r in rows if r["role"] in roles]
        elif "proposed" in rows[0]:
            rows = [r for r in rows if r["proposed"] == "1"] if "proposed" in roles else []
        if not rows:
            raise ValueError(f"{Path(path).name}: no target with role {sorted(roles)}")
    return {r["id"]: (float(r["x"]), float(r["y"]), float(r["z"])) for r in rows}


def load_frame(path=FRAME_JSON):
    """T_B_mandible (4 x 4, mm): the head frame B from the mandible frame, or None when the file does not exist. The
    file holds the single-precision matrix of the scene; its rotation is made orthonormal (a change below 1e-6)."""
    if not Path(path).exists():
        return None
    T = np.asarray(json.loads(Path(path).read_text(encoding="utf-8"))["T_B_mandible"], np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{Path(path).name}: T_B_mandible is not 4 x 4")
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return T


def load_calibration(path=None):
    """The rigs as calibrate() measured them, {"rigs": {name: 3 x 3}, "radii": {label: mm}, ...}, from path
    (CALIBRATION_JSON if None); None when there is none. A file that does not hold the rigs of RIGS and a radius for
    each of their spheres raises ValueError."""
    path = Path(CALIBRATION_JSON if path is None else path)
    if not path.is_file():
        return None
    c = json.loads(path.read_text(encoding="utf-8"))
    rigs = {n: np.asarray(c["rigs"][n], np.float64) for n in RIGS if n in c.get("rigs", {})}
    labels = [f"{n}_{s + 1}" for n in RIGS for s in range(3)]
    radii = {k: float(v) for k, v in (c.get("radii") or {}).items()}
    if set(rigs) != set(RIGS) or any(t.shape != (3, 3) for t in rigs.values()) or set(labels) - set(radii):
        raise ValueError(f"{path.name}: not a rig calibration (the rigs {sorted(RIGS)} and a radius per sphere)")
    return {**c, "rigs": rigs, "radii": radii}


# ---------------------------------------------------------------- transforms

def transform(T, X):
    return np.asarray(X, np.float64) @ T[:3, :3].T + T[:3, 3]


def inverse(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rotation_angle_deg(R):
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))   # accurate for small angles, unlike arccos


def kabsch(A, B, w=None):
    """T (4 x 4) mapping the points A onto B (B ~ R A + t) in weighted least squares (Arun et al. 1987, with the
    reflection check of Umeyama 1991)."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    w = np.full(len(A), 1.0 / len(A)) if w is None else np.asarray(w, np.float64) / np.sum(w)
    ca, cb = w @ A, w @ B
    H = (A - ca).T @ ((B - cb) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    T = np.eye(4)
    T[:3, :3] = Vt.T @ D @ U.T
    T[:3, 3] = cb - T[:3, :3] @ ca
    return T


def displacement(T_prev, T_next, targets=None):
    """Impact displacement dT = T_prev^-1 T_next from two registrations T_M_mandible (so dT is in the mandible
    frame): rotation (deg, axis), translation (mm) and the TRE (mm) that the registration left at T_prev puts on each
    target (mandible frame; load_targets() if None)."""
    dT = inverse(T_prev) @ T_next
    targets = load_targets() if targets is None else targets
    rv = Rotation.from_matrix(dT[:3, :3]).as_rotvec()
    ang = float(np.linalg.norm(rv))
    names = list(targets)
    P = np.array([targets[n] for n in names], np.float64)
    tre = np.linalg.norm(transform(dT, P) - P, axis=1)
    return {"dT": dT, "rot_deg": float(np.degrees(ang)), "axis": rv / ang if ang > 0 else np.zeros(3),
            "trans_mm": dT[:3, 3].copy(), "tre_mm": dict(zip(names, tre.tolist())),
            "tre_max_mm": float(tre.max()), "tre_mean_mm": float(tre.mean())}


# ---------------------------------------------------------------- point clouds

PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2", "ushort": "u2",
             "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "float": "f4",
             "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply(path):
    """Vertices (N x 3) of a PLY file, ascii or binary; the vertex element must come first."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, elements = None, []
        for line in f:
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                elements.append((tok[1], int(tok[2]), []))
            elif tok[0] == "property":
                elements[-1][2].append(tok[1:])
            elif tok[0] == "end_header":
                break
        name, n, props = elements[0]
        if name != "vertex" or any(p[0] == "list" for p in props):
            raise ValueError(f"{path}: the first PLY element must be vertex, without lists")
        cols = [p[-1] for p in props]
        idx = [cols.index(c) for c in ("x", "y", "z")]
        if fmt == "ascii":
            lines = f.read().splitlines()[:n]
            return np.array(b" ".join(lines).split(), np.float64).reshape(n, -1)[:, idx]
        order = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(c, order + PLY_TYPES[p[0]]) for c, p in zip(cols, props)])
        v = np.frombuffer(f.read(n * dt.itemsize), dt, count=n)
        return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)


def read_stl(path):
    """Unique vertices of an STL mesh (binary or ascii)."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[80:84], "little") if len(raw) >= 84 else -1
    if len(raw) == 84 + 50 * n:
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        V = np.frombuffer(raw, dt, count=n, offset=84)["v"].reshape(-1, 3)
    else:
        V = np.array([ln.split()[1:4] for ln in raw.decode("ascii", "replace").splitlines()
                      if ln.strip().startswith("vertex")], np.float64)
    return np.unique(V.astype(np.float64), axis=0)


def read_text(path):
    """x y z in the first three columns of a text cloud (.asc, .xyz, .txt, .csv, .pts); other lines skipped."""
    rows = []
    for ln in Path(path).read_text(errors="replace").splitlines():
        tok = ln.replace(",", " ").replace(";", " ").split()
        try:
            rows.append([float(t) for t in tok[:3]])
        except ValueError:
            continue
    return np.array([r for r in rows if len(r) == 3], np.float64)


THREEMF_UNITS = {"micron": 0.001, "millimeter": 1.0, "centimeter": 10.0, "inch": 25.4, "foot": 304.8, "meter": 1000.0}


def read_3mf(path):
    """Vertices of a 3MF mesh (a zip of XML models), in mm from the model's unit."""
    import re
    import zipfile
    out = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".model"):
                continue
            text = z.read(name).decode("utf-8", "replace")
            unit = re.search(r"<(?:\w+:)?model\b[^>]*\bunit=\"(\w+)\"", text)
            scale = THREEMF_UNITS.get(unit.group(1), 1.0) if unit else 1.0
            for tag in re.findall(r"<(?:\w+:)?vertex\b([^>]*)>", text):
                a = dict(re.findall(r"\b([xyz])=\"([^\"]+)\"", tag))
                if len(a) == 3:
                    out.append((float(a["x"]) * scale, float(a["y"]) * scale, float(a["z"]) * scale))
    return np.array(out, np.float64).reshape(-1, 3)


def load_cloud(path):
    """Points (N x 3, mm) of a scan export: PLY, STL, OBJ, 3MF, ASC/XYZ/TXT/CSV/PTS or NPY."""
    ext = Path(path).suffix.lower()
    if ext == ".ply":
        return read_ply(path)
    if ext == ".stl":
        return read_stl(path)
    if ext == ".3mf":
        return read_3mf(path)
    if ext == ".npy":
        return np.load(path).astype(np.float64)[:, :3]
    if ext == ".obj":
        return np.array([ln.split()[1:4] for ln in Path(path).read_text(errors="replace").splitlines()
                         if ln.startswith("v ")], np.float64)
    return read_text(path)


def write_ply(path, points):
    P = np.asarray(points, "<f4")
    head = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n" \
           "property float x\nproperty float y\nproperty float z\nend_header\n"
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(P.tobytes())


# ---------------------------------------------------------------- spheres

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    a = np.pi * (1.0 + 5 ** 0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(a), r * np.sin(a), z])


DIRECTIONS = cKDTree(fibonacci_sphere(500))


def fit_sphere(points, centre=None, radius=None, fixed_radius=False, trim=3.0, iters=5):
    """Robust least-squares sphere. Start from centre and radius (algebraic fit if not given), soft-L1 geometric fit,
    then plain geometric fits on the points within trim robust sigmas (MAD) of the surface, until the inliers settle.

    Returns centre, radius, rms (mm, inliers), n, n_in, sigma_centre (3D standard error of the centre, mm), coverage
    (fraction of the sphere with points) and the inlier mask.
    """
    P = np.asarray(points, np.float64)
    if centre is None:
        A = np.c_[2.0 * P, np.ones(len(P))]
        sol = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
        centre = sol[:3]
        radius = np.sqrt(sol[3] + centre @ centre) if radius is None else radius
    elif radius is None:
        radius = np.median(np.linalg.norm(P - centre, axis=1))
    r_fixed = float(radius)

    def res(x, Q):
        return np.linalg.norm(Q - x[:3], axis=1) - (r_fixed if fixed_radius else x[3])

    def jac(x, Q):
        d = Q - x[:3]
        J = -d / np.linalg.norm(d, axis=1)[:, None]
        return J if fixed_radius else np.c_[J, -np.ones(len(Q))]

    x = np.asarray(centre, np.float64) if fixed_radius else np.r_[centre, radius]
    x = least_squares(res, x, jac=jac, args=(P,), loss="soft_l1", f_scale=0.1).x
    mask = np.ones(len(P), bool)
    for _ in range(iters):
        r = res(x, P)
        s = max(1.4826 * np.median(np.abs(r - np.median(r))), 0.002)
        new = np.abs(r) < trim * s
        if new.sum() < 10 or np.array_equal(new, mask):
            break
        mask = new
        x = least_squares(res, x, jac=jac, args=(P[mask],), method="lm").x
    Q = P[mask]
    r = res(x, Q)
    J = jac(x, Q)
    dof = max(len(Q) - J.shape[1], 1)
    cov = (r @ r / dof) * np.linalg.pinv(J.T @ J)
    u = Q - x[:3]
    cells = DIRECTIONS.query(u / np.linalg.norm(u, axis=1)[:, None])[1]
    return {"centre": x[:3].copy(), "radius": r_fixed if fixed_radius else float(x[3]),
            "rms": float(np.sqrt(np.mean(r ** 2))), "n": len(P), "n_in": int(mask.sum()),
            "sigma_centre": float(np.sqrt(np.trace(cov[:3, :3]))),
            "coverage": len(np.unique(cells)) / DIRECTIONS.n, "inliers": mask}


def grid_keys(X, cell):
    k = np.floor(X / cell).astype(np.int64)
    kmin = k.min(0)
    dims = k.max(0) - kmin + 1
    return np.ravel_multi_index((k - kmin).T, dims), kmin, dims


def normals(Q, k=12, chunk=200_000):
    """Unoriented normals by PCA of the k nearest points."""
    nb = cKDTree(Q).query(Q, k=k, workers=-1)[1]
    out = np.empty_like(Q)
    for s in range(0, len(Q), chunk):
        N = Q[nb[s:s + chunk]]
        N = N - N.mean(1, keepdims=True)
        out[s:s + chunk] = np.linalg.eigh(np.einsum("mki,mkj->mij", N, N))[1][:, :, 0]
    return out


def find_spheres(points, radius=SPHERE_D / 2, radius_tol=0.5, voxel=0.5, min_votes=40, max_candidates=40,
                 min_points=150, min_coverage=0.08, max_rms=0.25):
    """Spheres of a nominal radius in a scan: candidate centres by normal voting (each point votes at one radius along
    its normal, both ways), then a robust fit of the points near each candidate. Returns the fits that pass the checks
    (radius within radius_tol, rms, points, coverage), best first."""
    P = np.asarray(points, np.float64)
    flat, _, _ = grid_keys(P, voxel)
    Q = P[np.unique(flat, return_index=True)[1]]                  # one point per voxel for normals and votes
    n = normals(Q)
    votes = np.vstack([Q + radius * n, Q - radius * n])
    flat, kmin, dims = grid_keys(votes, 1.0)
    cells, counts = np.unique(flat, return_counts=True)
    top = np.argsort(counts)[::-1][:2000]
    top = top[counts[top] >= min_votes / 8]
    seeds = (np.array(np.unravel_index(cells[top], dims)).T + kmin + 0.5) * 1.0
    vt = cKDTree(votes)
    score = vt.query_ball_point(seeds, 1.5, return_length=True, workers=-1)
    cand = []
    for i in np.argsort(score)[::-1]:
        if score[i] < min_votes or len(cand) == max_candidates:
            break
        if all(np.linalg.norm(seeds[i] - c) > 2.0 * radius for c in cand):
            cand.append(votes[vt.query_ball_point(seeds[i], 1.5)].mean(0))
    tree = cKDTree(P)
    fits = []
    for c in cand:
        f = None
        for band in (2.0, 1.0):                                  # gather, fit, gather again around the fit
            pts = P[tree.query_ball_point(c, radius + band)]
            if len(pts) < min_points:
                break
            f = fit_sphere(pts, c, radius)
            f["points"] = pts
            c = f["centre"]
        if f is None or abs(f["radius"] - radius) > radius_tol or f["rms"] > max_rms or f["n_in"] < min_points \
                or f["coverage"] < min_coverage:
            continue
        if all(np.linalg.norm(c - g["centre"]) > radius for g in fits):
            fits.append(f)
    return sorted(fits, key=lambda f: -f["n_in"])


# ---------------------------------------------------------------- rigs

def rig_distances(template):
    t = np.asarray(template)
    return np.array([np.linalg.norm(t[0] - t[1]), np.linalg.norm(t[1] - t[2]), np.linalg.norm(t[0] - t[2])])


def identify(centres, rigs=RIGS, tol=0.6, keep=20):
    """Labels of sphere centres from the distances inside each rig (scalene: one labelling per rig); the rigs take
    disjoint spheres. Returns {rig: (i, j, k) indices of spheres 1-3}, the worst distance error (mm) and the margin
    (mm) to the next labelling; None for the labels if the worst error exceeds tol."""
    C = np.asarray(centres, np.float64)
    if len(C) < 3 * len(rigs):
        return None, float("inf"), 0.0
    D = np.linalg.norm(C[:, None] - C[None], axis=2)
    p = np.array(list(itertools.permutations(range(len(C)), 3)))
    options = []
    for name, tpl in rigs.items():
        d = rig_distances(tpl)
        dev = np.abs(np.c_[D[p[:, 0], p[:, 1]], D[p[:, 1], p[:, 2]], D[p[:, 0], p[:, 2]]] - d).max(1)
        o = np.argsort(dev)[:keep]
        options.append([(dev[i], tuple(p[i])) for i in o])
    joint = []
    for combo in itertools.product(*options):
        ids = [i for _, t in combo for i in t]
        if len(set(ids)) == len(ids):
            joint.append((max(e for e, _ in combo), combo))
    if not joint:
        return None, float("inf"), 0.0
    joint.sort(key=lambda j: j[0])
    err = float(joint[0][0])
    margin = float(joint[1][0] - err) if len(joint) > 1 else float("inf")
    labels = {name: t for name, (_, t) in zip(rigs, joint[0][1])}
    return (labels if err <= tol else None), err, margin


def mean_rig(triples, template):
    """Mean shape of a rig from repeated measures (generalised Procrustes), in the frame of the template."""
    mean = np.asarray(template, np.float64)
    for _ in range(5):
        aligned = [transform(kabsch(t, mean), t) for t in triples]
        new = np.mean(aligned, axis=0)
        new = transform(kabsch(new, template), new)
        if np.abs(new - mean).max() < 1e-9:
            break
        mean = new
    return mean


# ---------------------------------------------------------------- one scan

def labelled_spheres(cloud, rigs=RIGS, radius=SPHERE_D / 2, tol=0.6, radii=None, **find):
    """Spheres of one cloud labelled as rig_1..rig_3: {label: fit}, plus identification error and margin. With radii
    (one value, or {label: radius} from calibrate_rigs) the centres are fitted again with the radius fixed: on a
    partial view the free radius trades off against the depth of the centre."""
    fits = find_spheres(cloud, radius, **find)
    labels, err, margin = identify([f["centre"] for f in fits], rigs, tol)
    info = {"n_spheres": len(fits), "id_error_mm": err, "id_margin_mm": margin}
    if labels is None:
        return None, info
    out = {f"{name}_{s + 1}": fits[i] for name, t in labels.items() for s, i in enumerate(t)}
    if radii is not None:
        for n, f in out.items():
            r = radii[n] if isinstance(radii, dict) else radii
            g = fit_sphere(f["points"], f["centre"], r, fixed_radius=True)
            g["radius_free"] = f["radius"]
            out[n] = g
    return out, info


def measure(scans, reference=None, X=None, Y=None, rigs=CALIBRATED, radii=CALIBRATED, targets=None,
            radius=SPHERE_D / 2, tol=0.6, T_B_mandible=None, calibration=None, **find):
    """True registration from one scan and, with a reference, the displacement since then.

    scans: a cloud (N x 3), a path, or a list of them (views of the same scene, merged or in their own frames: each
    view is aligned to the first used view on the six centres). reference: a previous result, its JSON path, or its
    T_M_mandible (4 x 4). X, Y: calibrated T_M_SM and T_SB_mandible (identity if None). rigs, radii: templates and
    sphere radii; CALIBRATED (the default) takes those of the rig calibration (calibration, CALIBRATION_JSON if None:
    calibrate()), and without one the design templates and free radii, said in the warnings: on the partial views
    the scanner has of the spheres a free radius trades off against the depth of the centre, about twice the error
    of a calibrated one. None: the design templates (rigs) or free radii (radii). T_B_mandible: the head frame B from
    the mandible frame (load_frame() if None).
    Returns centres in the frame of the first used view (views_used: the indices of the views that were identified),
    residuals (sphere fits, rig fits, spread between views), G, T_M_mandible, T_M_B (None, with a warning, when
    T_B_mandible is unknown) and, with a reference, the displacement (see displacement; dT_B: dT in B), plus warnings
    and calibrated (whether the rig calibration was used).
    """
    t0 = time.perf_counter()
    if isinstance(scans, (str, Path)) or (isinstance(scans, np.ndarray) and scans.ndim == 2):
        scans = [scans]
    clouds = [load_cloud(s) if isinstance(s, (str, Path)) else np.asarray(s, np.float64) for s in scans]
    views, used, info, warnings = [], [], [], []
    wants = [isinstance(x, str) and x == CALIBRATED for x in (rigs, radii)]
    cal = load_calibration(calibration) if any(wants) else None
    if any(wants) and cal is None:
        warnings.append("no rig calibration (python ground_truth.py calibrate): design rigs and free sphere radii")
    if wants[0]:
        rigs = cal["rigs"] if cal else RIGS
    elif rigs is None:
        rigs = RIGS
    if wants[1]:
        radii = cal["radii"] if cal else None
    for v, cloud in enumerate(clouds):
        lab, inf = labelled_spheres(cloud, rigs, radius, tol, radii, **find)
        info.append(inf)
        if lab is None and inf["n_spheres"] < 3 * len(rigs):
            warnings.append(f"view {v}: {inf['n_spheres']} spheres found, {3 * len(rigs)} needed")
        elif lab is None:
            warnings.append(f"view {v}: rigs not identified (worst distance error {inf['id_error_mm']:.2f} mm)")
        else:
            views.append(lab)
            used.append(v)
    if not views:
        raise ValueError("; ".join(warnings))
    names = list(views[0])
    C0 = np.array([views[0][n]["centre"] for n in names])
    stack = [C0]
    for lab in views[1:]:
        Ci = np.array([lab[n]["centre"] for n in names])
        stack.append(transform(kabsch(Ci, C0), Ci))
    stack = np.array(stack)
    C = stack.mean(0)
    spread = np.linalg.norm(stack - C, axis=2).max(0)
    spheres = {}
    for i, n in enumerate(names):
        f = [lab[n] for lab in views]
        spheres[n] = {"centre": C[i], "radius": float(np.mean([g.get("radius_free", g["radius"]) for g in f])),
                      "radius_fixed": radii is not None,
                      "rms_mm": float(np.max([g["rms"] for g in f])), "n_points": int(sum(g["n_in"] for g in f)),
                      "sigma_centre_mm": float(np.sqrt(np.mean([g["sigma_centre"] ** 2 for g in f]) / len(f))),
                      "coverage": float(np.max([g["coverage"] for g in f])), "view_spread_mm": float(spread[i])}
    poses = {}
    for name, tpl in rigs.items():
        meas = np.array([spheres[f"{name}_{s + 1}"]["centre"] for s in range(3)])
        T = kabsch(tpl, meas)
        poses[name] = {"T_scan_S": T, "residuals_mm": np.linalg.norm(transform(T, tpl) - meas, axis=1)}
    rig_m, rig_b = list(rigs)[:2]
    G = inverse(poses[rig_m]["T_scan_S"]) @ poses[rig_b]["T_scan_S"]
    T_M_mandible = (np.eye(4) if X is None else X) @ G @ (np.eye(4) if Y is None else Y)
    T_B_mandible = load_frame() if T_B_mandible is None else np.asarray(T_B_mandible, np.float64)
    if T_B_mandible is None:
        warnings.append(f"{FRAME_JSON.name} not found: T_M_B (head frame B, read by the AR) not written")
    T_M_B = None if T_B_mandible is None else T_M_mandible @ inverse(T_B_mandible)
    for n, s in spheres.items():
        r_ref = radius if radii is None else (radii[n] if isinstance(radii, dict) else radii)
        if abs(s["radius"] - r_ref) > 0.05:
            warnings.append(f"{n}: radius {s['radius']:.3f} mm, expected {r_ref:.3f}")
        if s["view_spread_mm"] > 0.05:
            warnings.append(f"{n}: views disagree by {s['view_spread_mm']:.3f} mm")
    for name, p in poses.items():
        if p["residuals_mm"].max() > 0.1:
            warnings.append(f"rig {name}: residual {p['residuals_mm'].max():.3f} mm (sphere moved in its seat?)")
    margin = min(info[v]["id_margin_mm"] for v in used)          # of the views used, not of the dropped ones
    if margin < 0.3:
        warnings.append(f"labelling margin {margin:.2f} mm: check the sphere labels")
    out = {"spheres": spheres, "rigs": poses, "G": G, "T_M_mandible": T_M_mandible, "T_M_B": T_M_B,
           "T_B_mandible": T_B_mandible, "views": info, "n_views_used": len(views), "views_used": used,
           "warnings": warnings, "calibrated": cal is not None}
    if reference is not None:
        if isinstance(reference, (str, Path)):
            reference = load_result(reference)
        T_prev = np.asarray(registration_of(reference) if isinstance(reference, dict) else reference, np.float64)
        out["displacement"] = d = displacement(T_prev, T_M_mandible, targets)
        if T_B_mandible is not None:
            d["dT_B"] = T_B_mandible @ d["dT"] @ inverse(T_B_mandible)
    out["time_s"] = time.perf_counter() - t0
    return out


def registration_of(result):
    """T_M_mandible of a result dict of measure(); a result written before the two names existed holds it under
    "T_M_B", and is told apart from a registration in the bench frame B ({"T_M_B": ...}, as live_ar reads it) by its
    "G". Raises ValueError for a dict that is neither."""
    if "T_M_mandible" in result:
        return result["T_M_mandible"]
    if "G" in result:
        return result["T_M_B"]
    raise ValueError("not a result of measure: a registration in the bench frame B, not in the mandible frame")


def calibrate_rigs(results, rigs=RIGS):
    """Rig templates (mean shape, in the frame of the design template) and sphere radii (mean free fit) from repeated
    scans without impact, to pass to measure as rigs and radii."""
    templates = {name: mean_rig([np.array([r["spheres"][f"{name}_{s + 1}"]["centre"] for s in range(3)])
                                 for r in results], tpl) for name, tpl in rigs.items()}
    radii = {n: float(np.mean([r["spheres"][n]["radius"] for r in results])) for n in results[0]["spheres"]}
    return {"rigs": templates, "radii": radii}


def calibrate(scans, out=None, rigs=RIGS, **kw):
    """The rig calibration, once, from MIN_CALIBRATION_SCANS or more scans of the bench without an impact between
    them (the spheres in their seats, the bench moved or turned between scans so that each sphere is seen from other
    sides): the radius of each sphere, the mean of its free fits, and the templates of the two rigs from the centres
    fitted again with those radii. Written to out (CALIBRATION_JSON if None) with the scans it came from and the
    spread of each radius; every measure() takes it from then on. Returns the calibration."""
    from datetime import datetime
    if len(scans) < MIN_CALIBRATION_SCANS:
        raise ValueError(f"{len(scans)} scans: a calibration takes at least {MIN_CALIBRATION_SCANS}")
    free = [measure(s, rigs=rigs, radii=None, **kw) for s in scans]
    radii = calibrate_rigs(free, rigs)["radii"]
    fixed = [measure(s, rigs=rigs, radii=radii, **kw) for s in scans]
    cal = {"rigs": calibrate_rigs(fixed, rigs)["rigs"], "radii": radii,
           "radius_sd_mm": {n: float(np.std([r["spheres"][n]["radius"] for r in free])) for n in radii},
           "scans": [Path(s).name if isinstance(s, (str, Path)) else f"cloud {k}" for k, s in enumerate(scans)],
           "date": datetime.now().astimezone().isoformat(timespec="seconds"),
           "warnings": sorted({w for r in free + fixed for w in r["warnings"]})}
    path = Path(CALIBRATION_JSON if out is None else out)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(jsonable(cal), indent=1, allow_nan=False), encoding="utf-8")
    part.replace(path)
    return cal


def jsonable(x):
    """x with numpy types turned into Python ones and non-finite floats into None (strict JSON has no Infinity)."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save_result(result, path):
    """Write a result of measure() as strict JSON (non-finite values as null)."""
    Path(path).write_text(json.dumps(jsonable(result), indent=1, allow_nan=False), encoding="utf-8")


def load_result(path):
    """A result written by save_result(), with T_M_mandible, T_M_B (None when unknown) and G as arrays. A file
    written before the two names existed holds T_M_mandible under "T_M_B": it is read as such, and T_M_B is then
    derived from its T_B_mandible or from load_frame(). Raises ValueError for a file that is not a result of
    measure(), such as a registration in the bench frame B (no T_M_mandible and no G)."""
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(r, dict) or ("T_M_mandible" not in r and "G" not in r):
        raise ValueError(f"{Path(path).name}: not a result of measure (a registration in the bench frame B?)")
    if "T_M_mandible" not in r:
        r["T_M_mandible"] = r["T_M_B"]
        T = r.get("T_B_mandible") or load_frame()
        r["T_B_mandible"] = T
        r["T_M_B"] = None if T is None else (np.array(r["T_M_mandible"]) @ inverse(np.asarray(T, np.float64))).tolist()
    for k in ("T_M_mandible", "T_M_B", "T_B_mandible", "G"):
        if r.get(k) is not None:
            r[k] = np.array(r[k], np.float64)
    return r


# ---------------------------------------------------------------- synthetic scans

def rot_x(deg):
    a = np.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])


def pose(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# Bench-like scene in W, a world-oriented frame on the hinge (x left, -y forward, z up): the mandible open 2 deg as on
# the bench; the board 57 mm in front of the incisors facing the phone (left, front, up), the A1 ball behind it.
SCENE = {"R_W_mandible": rot_x(-92.3745), "board_W": (15.0, -150.0, -35.0), "board_out": (0.5, -0.8, 0.35),
         "ball_M": (22.5, 50.0, 35.0), "face": ((0.0, -55.0, -45.0), 60.0),
         "views_deg": tuple(float(a) for a in trials.Config().scan_az),
         "elevation_deg": 5.0, "range_mm": 320.0, "baseline_mm": 100.0,
         "posts": {"marker": (1.8, [(0.7071, -0.7071, 0.0), (-0.7071, -0.7071, 0.0), (-1.0, 0.0, 0.0)]),
                   "mandible": (3.0, None)}}            # post radius and direction from the sphere (None: to the hub)
HUB_MANDIBLE = (46.0, 107.1597, -54.9717)               # hub of the head rig posts, mandible frame


def scene_poses():
    """T_W_mandible and T_W_M of the nominal scene."""
    n_out = unit(SCENE["board_out"])
    z = -n_out
    x = unit(np.cross([0.0, 0.0, 1.0], n_out))
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    o = np.asarray(SCENE["board_W"]) - R @ np.array([22.5, 22.5, 0.0])
    return pose(SCENE["R_W_mandible"], np.zeros(3)), pose(R, o)


# The scanner at each arm position as the bench scene has it (its cameras cam_scanner_<k> on their wedge), written by
# the twin's export (twin.py, twin_export.py): azimuth, elevation and distance seen from the spheres' centroid, in a
# frame turned as SCENE's (x left, -y anterior, z up).
SCANNER_POSES = Path(__file__).resolve().parents[2] / "03_data" / "processed" / "protocol" / "scanner_poses.json"
_said = []


def scan_views(path=SCANNER_POSES):
    """(azimuth, elevation, distance) of the scanner at the protocol's scan positions (SCENE views_deg, camera
    azimuths of the arm), from the bench scene's scanner poses; without them the nominal SCENE values (elevation_deg,
    range_mm), said once."""
    try:
        pos = json.loads(Path(path).read_text(encoding="utf-8"))["positions"]
        out = []
        for az in SCENE["views_deg"]:
            p = min(pos, key=lambda q: abs(float(q.get("camera_deg", 1e9)) - az))
            if abs(float(p.get("camera_deg", 1e9)) - az) > 0.5:
                raise KeyError(f"no scanner pose at camera {az} deg")
            out.append((float(p["azimuth_deg"]), float(p["elevation_deg"]), float(p["distance_mm"])))
        return out
    except (OSError, ValueError, KeyError, TypeError) as e:
        if not _said:
            _said.append(1)
            print(f"warning: {Path(path).name} not usable ({e}): simulated scans from the nominal "
                  f"{SCENE['range_mm']:.0f} mm", file=sys.stderr, flush=True)
        return [(az, SCENE["elevation_deg"], SCENE["range_mm"]) for az in SCENE["views_deg"]]


def scanner_poses(centroid, jitter=None):
    """T_W_scan of the scanner views (scan_views): optical axis z on the sphere centroid, x horizontal."""
    out = []
    for i, (az, el, rng_) in enumerate(scan_views()):
        a, e = np.radians(az), np.radians(el)
        if jitter is not None:
            a, e, rng_ = a + jitter[i][0], e + jitter[i][1], rng_ + jitter[i][2]
        d = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
        z = -d
        x = unit(np.cross([0.0, 0.0, 1.0], z))
        out.append(pose(np.column_stack([x, np.cross(z, x), z]), centroid + rng_ * d))
    return out


def blocked(P, target, centres, radius, skip):
    """Segments from P to target crossing one of the spheres other than skip."""
    out = np.zeros(len(P), bool)
    v = target - P
    L = np.linalg.norm(v, axis=1)
    v = v / L[:, None]
    for j, c in enumerate(centres):
        if j == skip:
            continue
        w = c - P
        t = (w * v).sum(1)
        out |= (t > 0) & (t < L) & ((w * w).sum(1) - t * t < radius * radius)
    return out


def sample_sphere(c, j, centres, O, cams, rng, radius, spacing, noise, post, theta_max=75.0):
    """Points of sphere j seen by the projector at O and at least one camera, depth noise along the projector ray,
    spikes on 0.5 % of the points and 3 % at grazing incidence, no points in the post seat; sparse noisy points of
    the black post."""
    w = unit(c - O)
    e1 = unit(np.cross(w, [0.0, 0.0, 1.0]))
    e2 = np.cross(w, e1)
    g = np.arange(-radius * 1.05, radius * 1.05, spacing)
    gi, gj = np.meshgrid(g + rng.uniform(0, spacing), g + rng.uniform(0, spacing))
    q = c + gi.reshape(-1, 1) * e1 + gj.reshape(-1, 1) * e2
    u = unit(q - O)
    oc = c - O
    b = u @ oc
    disc = b * b - (oc @ oc - radius * radius)
    u, b, disc = u[disc > 0], b[disc > 0], disc[disc > 0]
    t = b - np.sqrt(disc)
    p = O + t[:, None] * u
    nrm = (p - c) / radius
    cos_p = -(nrm * u).sum(1)
    cmin = np.cos(np.radians(theta_max))
    ok = (cos_p > cmin) & ~blocked(p, O, centres, radius, j)
    cam_ok = np.zeros(len(p), bool)
    for cam in cams:
        cam_ok |= ((nrm * unit(cam - p)).sum(1) > cmin) & ~blocked(p, cam, centres, radius, j)
    r_post, a = post
    ok &= cam_ok & ((nrm @ a) < np.cos(np.arcsin(r_post / radius)))
    p, u, cos_p = p[ok], u[ok], cos_p[ok]
    dz = rng.normal(0.0, noise, len(p))
    spike = (rng.random(len(p)) < 0.005) | ((cos_p < 0.5) & (rng.random(len(p)) < 0.03))
    dz[spike] += rng.choice([-1.0, 1.0], spike.sum()) * rng.uniform(0.3, 2.0, spike.sum())
    pts = [p + dz[:, None] * u]
    # post: cylinder from the sphere surface outwards, 12 long, 20 % of the points facing the scanner, noise 0.1
    e1 = unit(np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0]))
    e2 = np.cross(a, e1)
    s = np.arange(radius * np.cos(np.arcsin(r_post / radius)), radius + 12.0, 0.3)
    phi = np.arange(0.0, 2 * np.pi, 0.3 / r_post)
    S, F = np.meshgrid(s, phi)
    rad = np.cos(F.reshape(-1, 1)) * e1 + np.sin(F.reshape(-1, 1)) * e2
    q = c + S.reshape(-1, 1) * a + r_post * rad
    keep = ((rad * unit(O - q)).sum(1) > 0.3) & (rng.random(len(q)) < 0.2)
    q = q[keep]
    pts.append(q + rng.normal(0.0, 0.1, q.shape))
    return np.vstack(pts)


def synthetic_scan(T_W_M, T_W_mandible, rng, noise=0.05, spacing=0.17, merged=True, clutter=True, dense=False,
                   jitter=True, rigs=RIGS, radius=SPHERE_D / 2):
    """Simulated EinScan-SP scan (one shot per scan position of the protocol, trials.Config.scan_az) of the two
    rigs placed by T_W_M and T_W_mandible.

    Model: projector on the optical axis, cameras 100 mm to each side, 320 mm from the sphere centroid; a point is
    measured if the projector and one camera see it under 75 deg at most and no other sphere hides it; depth noise
    (sd noise, mm) along the projector ray every spacing mm (EinScan-SP: point distance 0.17 mm, accuracy 0.05 mm);
    spikes; posts. Clutter: the printed board, a face-like surface, flying points. Returns the clouds (one merged in
    the frame of the first view, or one per view in its own frame) and the truth: T_scan_W per view, centres.
    """
    T_W = {"marker": T_W_M, "mandible": T_W_mandible}
    centres, posts = [], []
    hub = transform(T_W_mandible, [HUB_MANDIBLE])[0]
    for name, tpl in rigs.items():
        cw = transform(T_W[name], tpl)
        r_post, dirs = SCENE["posts"][name]
        for i, c in enumerate(cw):
            a = unit(hub - c) if dirs is None else T_W[name][:3, :3] @ np.asarray(dirs[i])
            centres.append(c)
            posts.append((r_post, a))
    centres = np.array(centres)
    centroid = centres.mean(0)
    jit = None if not jitter else [(rng.normal(0, np.radians(0.5)), rng.normal(0, np.radians(0.5)), rng.normal(0, 3))
                                   for _ in SCENE["views_deg"]]
    T_W_scan = scanner_poses(centroid, jit)
    clouds = []
    for T in T_W_scan:
        O, xs = T[:3, 3], T[:3, 0]
        cams = [O + SCENE["baseline_mm"] * xs, O - SCENE["baseline_mm"] * xs]
        pts = [sample_sphere(c, j, centres, O, cams, rng, radius, spacing, noise, posts[j])
               for j, c in enumerate(centres)]
        if clutter:
            pts.append(clutter_points(T_W_M, O, centres, rng, noise, spacing * (1.0 if dense else 2.5), radius))
        P = np.vstack(pts)
        clouds.append(transform(inverse(T_W_scan[0] if merged else T), P))
    truth = {"T_scan_W": [inverse(T) for T in T_W_scan], "centres_W": centres, "T_W_M": T_W_M,
             "T_W_mandible": T_W_mandible}
    return ([np.vstack(clouds)] if merged else clouds), truth


def clutter_points(T_W_M, O, centres, rng, noise, spacing, radius):
    """Board print (50 x 50 at z = 0 of M), face-like sphere R 60 (its side towards the scanner), flying points; none
    within radius + 0.5 of a sphere centre."""
    g = np.arange(-2.5, 47.5, spacing)
    X, Yg = np.meshgrid(g, g)
    board = transform(T_W_M, np.column_stack([X.ravel(), Yg.ravel(), np.zeros(X.size)]))
    fc, fr = SCENE["face"]
    n_face = int(4 * np.pi * fr * fr / spacing ** 2 / 2)
    d = unit(rng.normal(size=(n_face, 3)))
    d = d[(d @ unit(O - np.asarray(fc))) > 0.2]
    face = np.asarray(fc) + fr * d
    P = np.vstack([board, face])
    P = P + unit(O - P) * rng.normal(0.0, noise, (len(P), 1))
    lo, hi = centres.min(0) - 40.0, centres.max(0) + 40.0
    fly = lo + rng.random((max(len(P) // 500, 200), 3)) * (hi - lo)
    P = np.vstack([P, fly])
    near = cKDTree(centres).query(P)[0] < radius + 0.5
    return P[~near]


def random_impact(T_W_M, T_W_mandible, rng, max_deg=5.0, max_mm=2.0):
    """Marker moved on the bone: rotation up to max_deg about the A1 ball centre plus a translation up to max_mm.
    Returns the new T_W_M (T_W_mandible, the head, does not move)."""
    axis = unit(rng.normal(size=3))
    R = Rotation.from_rotvec(axis * np.radians(rng.uniform(0.5, max_deg))).as_matrix()
    c = transform(T_W_M, [SCENE["ball_M"]])[0]
    t = unit(rng.normal(size=3)) * rng.uniform(0.0, max_mm)
    D = pose(R, c - R @ c + t)                                  # motion of the marker, in W
    return D @ T_W_M


# ---------------------------------------------------------------- self test

def selftest(trials=200, noise=0.05, seed=0, merged=True, dense=False, radii=None, verbose=True, targets=None):
    """Monte Carlo on synthetic scans: for each trial a reference scan, a random impact, a second scan; errors of the
    centres, of T_M_mandible on the targets and of the displacement (dT on the targets, TRE). radii: fixed sphere
    radius, as after calibrate_rigs. targets: {id: point of the mandible frame} (load_targets() if None). Pass: 95th
    percentile of the sphere centre error under 0.01 mm, every rig identified. The errors on the targets are
    reported: they grow with the distance of the targets from the rigs."""
    rng = np.random.default_rng(seed)
    T_W_mandible, T_W_M0 = scene_poses()
    targets = load_targets() if targets is None else targets
    tg = np.array(list(targets.values()))
    frame = load_frame()
    rows, fails, times = [], 0, []
    for _ in range(trials):
        T_W_M = random_impact(T_W_M0, T_W_mandible, rng, 3.0, 3.0)       # a different start every trial
        T_W_M1 = random_impact(T_W_M, T_W_mandible, rng)
        res, tru = [], []
        for TM in (T_W_M, T_W_M1):
            clouds, truth = synthetic_scan(TM, T_W_mandible, rng, noise=noise, merged=merged, dense=dense)
            try:
                r = measure(clouds, rigs=RIGS, radii=radii, T_B_mandible=frame)
            except ValueError:
                fails += 1
                break
            times.append(r["time_s"])
            res.append(r)
            tru.append(truth)
        if len(res) < 2:
            continue
        row = {}
        e_c = []
        for r, t in zip(res, tru):
            # centres are in the frame of the first view used (separate views: view 0 may have been dropped)
            C_true = transform(t["T_scan_W"][r["views_used"][0]], t["centres_W"])
            C_est = np.array([r["spheres"][n]["centre"] for n in r["spheres"]])
            e_c.append(np.linalg.norm(C_est - C_true, axis=1).max())
        row["centre"] = max(e_c)
        row["radius"] = max(abs(s["radius"] - SPHERE_D / 2) for r in res for s in r["spheres"].values())
        T_true = [inverse(t["T_W_M"]) @ t["T_W_mandible"] for t in tru]
        row["T_M_mandible"] = max(np.linalg.norm(transform(r["T_M_mandible"], tg) - transform(T, tg), axis=1).max()
                                  for r, T in zip(res, T_true))
        d_est = displacement(res[0]["T_M_mandible"], res[1]["T_M_mandible"], targets)
        d_true = displacement(T_true[0], T_true[1], targets)
        row["dT"] = np.linalg.norm(transform(d_est["dT"], tg) - transform(d_true["dT"], tg), axis=1).max()
        row["tre"] = max(abs(d_est["tre_mm"][k] - d_true["tre_mm"][k]) for k in targets)
        row["rot_deg"] = abs(d_est["rot_deg"] - d_true["rot_deg"])
        row["tre_true"] = d_true["tre_max_mm"]
        rows.append(row)
    keys = ["centre", "radius", "T_M_mandible", "dT", "tre", "rot_deg"]
    nan = float("nan")
    stats = {k: {"median": float(np.median([r[k] for r in rows])) if rows else nan,
                 "p95": float(np.percentile([r[k] for r in rows], 95)) if rows else nan,
                 "max": float(np.max([r[k] for r in rows])) if rows else nan} for k in keys}
    ok = fails == 0 and bool(rows) and stats["centre"]["p95"] < 0.01
    if verbose:
        tre_true = f"{np.median([r['tre_true'] for r in rows]):.2f} mm median" if rows else "none"
        timing = f"{np.median(times):.2f} s median, {np.max(times):.2f} s max" if times else "none"
        print(f"{len(rows)} trials, noise {noise} mm, {'merged' if merged else 'separate'} views"
              f"{', dense clutter' if dense else ''}, {'free' if radii is None else 'fixed'} radius; "
              f"identification failures {fails}; true TRE max per trial {tre_true}; time per scan {timing}")
        print(f"{'error (mm, deg)':<28}{'median':>10}{'p95':>10}{'max':>10}")
        label = {"centre": "sphere centre (worst of 6)", "radius": "sphere radius",
                 "T_M_mandible": "T_M_mandible on targets", "dT": "dT on targets", "tre": "TRE of the impact",
                 "rot_deg": "impact rotation (deg)"}
        for k in keys:
            print(f"{label[k]:<28}{stats[k]['median']:>10.4f}{stats[k]['p95']:>10.4f}{stats[k]['max']:>10.4f}")
        print("PASS" if ok else "FAIL", "(p95 of the sphere centre error under 0.01 mm, no identification failure)")
    return ok, stats, fails


def quick_test():
    """Checks of the targets table, of the two head frames and of the result files, a few seconds. Returns the list
    of the failed checks."""
    import tempfile
    fails = []

    def check(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    rows = []
    if TARGETS_CSV.exists():
        with open(TARGETS_CSV, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        proposed = [r["id"] for r in rows if r["role"] == "proposed"]
        targets = load_targets()
        check(list(targets) == proposed and len(proposed) < len(rows),
              f"load_targets: the {len(proposed)} proposed targets of {len(rows)} ({', '.join(targets)})")
        check(len(load_targets(roles=None)) == len(rows), "load_targets(roles=None): every row")
    else:
        check(load_targets() == TARGETS_MANDIBLE, f"no {TARGETS_CSV.name}: the placeholders")
    T_B_mandible = load_frame()
    if T_B_mandible is not None and rows:
        P = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows])
        Q = np.array([[float(r[k]) for k in ("x_B", "y_B", "z_B")] for r in rows])
        err = float(np.abs(transform(T_B_mandible, P) - Q).max())
        check(err < 2e-3, f"T_B_mandible maps the columns x, y, z onto x_B, y_B, z_B ({err:.4f} mm)")
    frame = pose(rot_x(-92.3745), [0.0, 96.364, 16.66]) if T_B_mandible is None else T_B_mandible
    check(jsonable({"a": [float("inf"), np.float64("nan"), 1.0]}) == {"a": [None, None, 1.0]},
          "jsonable: non-finite values become null")
    T_W_mandible, T_W_M = scene_poses()
    clouds, _ = synthetic_scan(T_W_M, T_W_mandible, np.random.default_rng(0))
    r = measure(clouds, rigs=RIGS, radii=None, T_B_mandible=frame)
    true = inverse(T_W_M) @ T_W_mandible
    tg = np.array(list(load_targets().values()))
    err = float(np.linalg.norm(transform(r["T_M_mandible"], tg) - transform(true, tg), axis=1).max())
    check(err < 0.05, f"measure: T_M_mandible on the targets within 0.05 mm of the truth ({err:.4f} mm)")
    check(np.allclose(r["T_M_B"] @ frame, r["T_M_mandible"], atol=1e-9), "measure: T_M_B = T_M_mandible T_B_mandible^-1")
    with tempfile.TemporaryDirectory(prefix="ground_truth_test_") as tmp:
        path = Path(tmp) / "result.json"
        save_result(r, path)
        text = path.read_text(encoding="utf-8")

        def strict(token):
            raise ValueError(token)
        try:
            json.loads(text, parse_constant=strict)
            check(True, "result file: strict JSON")
        except ValueError as e:
            check(False, f"result file: strict JSON ({e})")
        back = load_result(path)
        check(np.allclose(back["T_M_mandible"], r["T_M_mandible"]) and np.allclose(back["T_M_B"], r["T_M_B"]),
              "result file: T_M_mandible and T_M_B read back")
        old = json.loads(text)
        old["T_M_B"] = old.pop("T_M_mandible")             # a file of the older format
        old.pop("T_B_mandible")
        legacy = Path(tmp) / "legacy.json"
        legacy.write_text(json.dumps(old), encoding="utf-8")
        back = load_result(legacy)
        derived = T_B_mandible is None or np.allclose(back["T_M_B"] @ T_B_mandible, r["T_M_mandible"])
        check(np.allclose(back["T_M_mandible"], r["T_M_mandible"]) and derived,
              "older result file: its T_M_B read as T_M_mandible, T_M_B derived")
        d = measure(clouds, reference=legacy, rigs=RIGS, radii=None, T_B_mandible=frame)["displacement"]
        check(d["rot_deg"] < 1e-6 and np.abs(d["trans_mm"]).max() < 1e-6 and np.allclose(d["dT_B"], np.eye(4)),
              "measure against an older reference file: no displacement on the same scan")
        bench = Path(tmp) / "bench.json"                    # a registration in B, as live_ar reads it
        bench.write_text(json.dumps({"T_M_B": np.eye(4).tolist()}), encoding="utf-8")
        refused = []
        for attempt in (lambda: load_result(bench), lambda: registration_of({"T_M_B": np.eye(4)})):
            try:
                attempt()
            except ValueError:
                refused.append(True)
        check(len(refused) == 2, "a registration in the bench frame is never taken as a result in the mandible frame")
        # the rig calibration, once, from three scans without an impact (spheres 0.1 mm smaller than drawn): the
        # radius of each sphere written and read back; from then on measure takes them, and says when there is none
        crng = np.random.default_rng(1)
        small = SPHERE_D / 2 - 0.1
        cal_scans = [synthetic_scan(random_impact(T_W_M, T_W_mandible, crng, 3.0, 3.0), T_W_mandible, crng,
                                    radius=small)[0] for _ in range(MIN_CALIBRATION_SCANS)]
        cal_path = Path(tmp) / "rig_calibration.json"
        calibrate(cal_scans, out=cal_path, T_B_mandible=frame)
        got = load_calibration(cal_path)
        off = max(abs(v - small) for v in got["radii"].values())
        check(len(got["radii"]) == 6 and off < 0.03 and set(got["rigs"]) == set(RIGS),
              f"calibrate: the radius of each of the six spheres ({off:.3f} mm off) and the two rigs, written and "
              f"read back")
        on = measure(cal_scans[0], T_B_mandible=frame, calibration=cal_path)
        off_cal = measure(cal_scans[0], T_B_mandible=frame, calibration=Path(tmp) / "none.json")
        said = [any("no rig calibration" in w for w in r["warnings"]) for r in (on, off_cal)]
        check(on["calibrated"] and all(s["radius_fixed"] for s in on["spheres"].values()) and not off_cal["calibrated"]
              and not any(s["radius_fixed"] for s in off_cal["spheres"].values()) and said == [False, True],
              "measure: the calibrated radii by default; without a calibration free radii, and it says so")
        try:
            calibrate(cal_scans[:MIN_CALIBRATION_SCANS - 1], out=Path(tmp) / "few.json")
            few = False
        except ValueError:
            few = not (Path(tmp) / "few.json").exists()
        check(few, f"calibrate: refused with fewer than {MIN_CALIBRATION_SCANS} scans, nothing written")
    ok, stats, n_fail = selftest(trials=1, noise=5.0, verbose=False)
    check(not ok and n_fail == 1 and np.isnan(stats["centre"]["p95"]), "selftest: every trial failed gives FAIL, no crash")
    print(f"{len(fails)} failed" if fails else "all passed")
    return fails


# ---------------------------------------------------------------- command line

def print_result(r):
    for n, s in r["spheres"].items():
        c = " ".join(f"{v:9.3f}" for v in s["centre"])
        print(f"{n:<12} centre {c}  r {s['radius']:.3f}  rms {s['rms_mm']:.3f}  "
              f"sd {s['sigma_centre_mm']:.4f}  pts {s['n_points']}  cover {s['coverage']:.2f}")
    for name, p in r["rigs"].items():
        print(f"rig {name}: residuals {np.round(p['residuals_mm'], 4)} mm")
    if "displacement" in r:
        d = r["displacement"]
        t = " ".join(f"{v:.3f}" for v in d["trans_mm"])
        print(f"displacement: {d['rot_deg']:.3f} deg, translation {t} mm, "
              f"TRE max {d['tre_max_mm']:.3f} mean {d['tre_mean_mm']:.3f} mm")
    for w in r["warnings"]:
        print("warning:", w)
    print(f"time {r['time_s']:.2f} s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("scans", nargs="+")
    m.add_argument("--reference")
    m.add_argument("--out")
    s = sub.add_parser("selftest")
    s.add_argument("--trials", type=int, default=200)
    s.add_argument("--noise", type=float, default=0.05)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--separate", action="store_true", help="views in their own frames, not merged")
    s.add_argument("--dense", action="store_true", help="clutter as dense as the spheres")
    s.add_argument("--fixed-radius", action="store_true", help="sphere radius known (calibrate_rigs)")
    y = sub.add_parser("synth", help="write synthetic scans: a reference and impacts, with the truth")
    y.add_argument("outdir")
    y.add_argument("--impacts", type=int, default=3)
    y.add_argument("--seed", type=int, default=0)
    sub.add_parser("test", help="quick checks of the targets, the two head frames and the result files")
    c = sub.add_parser("calibrate", help="the rig calibration, once: the radius of each sphere and the rig templates "
                                         "from scans without an impact, the bench turned between them")
    c.add_argument("scans", nargs="+")
    c.add_argument("--out", help=f"default {CALIBRATION_JSON}")
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        cal = calibrate(a.scans, out=a.out)
        for n, r in cal["radii"].items():
            print(f"{n:<12} radius {r:.3f} mm  sd {cal['radius_sd_mm'][n]:.3f} over {len(cal['scans'])} scans")
        for w in cal["warnings"]:
            print("warning:", w)
        print(f"written: {a.out or CALIBRATION_JSON}")
    elif a.cmd == "measure":
        r = measure(a.scans, reference=a.reference)
        print_result(r)
        if a.out:
            save_result(r, a.out)
    elif a.cmd == "selftest":
        ok = selftest(a.trials, a.noise, a.seed, merged=not a.separate, dense=a.dense,
                      radii=SPHERE_D / 2 if a.fixed_radius else None)[0]
        sys.exit(0 if ok else 1)
    elif a.cmd == "test":
        sys.exit(1 if quick_test() else 0)
    else:
        rng = np.random.default_rng(a.seed)
        out = Path(a.outdir)
        out.mkdir(parents=True, exist_ok=True)
        T_W_mandible, T_W_M = scene_poses()
        truth = {}
        for k in range(a.impacts + 1):
            if k:
                T_W_M = random_impact(T_W_M, T_W_mandible, rng)
            clouds, t = synthetic_scan(T_W_M, T_W_mandible, rng)
            write_ply(out / f"scan_{k:02d}.ply", clouds[0])
            truth[f"scan_{k:02d}"] = {"T_M_mandible": inverse(T_W_M) @ T_W_mandible,
                                      "centres_scan": transform(t["T_scan_W"][0], t["centres_W"])}
        (out / "truth.json").write_text(json.dumps(jsonable(truth), indent=1, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
