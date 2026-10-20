"""Known occluders rendered with the face: the face seen by a camera with the parts in front of it taken out.

Two groups of occluders, as on the bench: parts fixed to the head (frame B: splint, cups and screws of the interface)
and parts fixed to the board (frame M: marker, plate, holder, arm, marker spheres). The face and each group are drawn
by their own renderer (gl_overlay.GLMeshRenderer by default, or mesh_overlay.MeshRenderer), the head parts at the
camera from B, the board parts at the camera from M (the marker pose of the image, whatever the registration), and a
face pixel is occluded where an occluder surface is nearer to the camera than the face surface there (depth from the
visibility buffers, the rays through the pixel centres undistorted). OccludedRenderer has the interface of
MeshRenderer for correct.py: render() gives the visible face (mask and buffer without the occluded pixels), plus the
occluder silhouettes and the ignore mask (occluders dilated by DILATE_PX), so that outline points along an occluder
are left out like those on the ignore mask of the labels.

    occ = Occluders.from_twin(np.load(".../meshes/occluders.npz"), "a1")
    ren = OccludedRenderer(face_vertices, face_faces, occ)
    r = ren.render(K, dist, T_C_B, size, T_C_M=T_C_M)     # r.mask, r.faces, r.offset; r.occluder, r.ignore
"""
from dataclasses import dataclass

import cv2
import numpy as np

import mesh_overlay as mo

DILATE_PX = 3


@dataclass
class Occluders:
    """Occluder meshes: fixed to the head (vertices in B) and fixed to the board (vertices in M); None when empty."""
    head: tuple = None       # (vertices, faces)
    board: tuple = None

    @staticmethod
    def from_twin(npz, variant="a1"):
        """From synth_twin.py meshes/occluders.npz (parts in B at the registration T_M_B of the scene, the names that
        move with the marker in "moving"); the parts of the other interface left out."""
        T = np.asarray(npz["T_M_B"], np.float64)
        moving = {str(n) for n in npz["moving"]}
        other = "b1_" if variant == "a1" else "a1_"
        groups = {"head": ([], []), "board": ([], [])}
        for name in sorted({k.split("__")[0] for k in npz.files if "__" in k}):
            if name.startswith(other):
                continue
            v, f = npz[name + "__v"].astype(np.float64), npz[name + "__f"].astype(np.int64)
            key = "board" if name in moving else "head"
            if key == "board":
                v = v @ T[:3, :3].T + T[:3, 3]
            vs, fs = groups[key]
            fs.append(f + sum(len(x) for x in vs))
            vs.append(v)
        out = {k: (np.vstack(vs), np.vstack(fs)) if vs else None for k, (vs, fs) in groups.items()}
        return Occluders(out["head"], out["board"])


def renderer(v, f, kind="gl"):
    if kind == "gl":
        import gl_overlay
        return gl_overlay.GLMeshRenderer(v, f)
    return mo.MeshRenderer(v, f)


def depth(ren, r, K, dist, T, shape):
    """Camera z (mm) of the visible triangle at each pixel centre, inf where there is none."""
    out = np.full(shape, np.inf)
    (x0, y0), buf = r.offset, r.faces
    ys, xs = np.nonzero(buf >= 0)
    if not len(ys):
        return out
    tri = ren.v[ren.f[buf[ys, xs]]] @ T[:3, :3].T + T[:3, 3]            # n x 3 x 3, camera frame
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    px = np.c_[xs + x0, ys + y0].astype(np.float64).reshape(-1, 1, 2)
    d = np.c_[cv2.undistortPoints(px, K, dist).reshape(-1, 2), np.ones(len(xs))]   # rays with z = 1
    den = np.sum(n * d, 1)
    ok = np.abs(den) > 1e-12
    out[ys + y0, xs + x0] = np.where(ok, np.sum(n * tri[:, 0], 1) / np.where(ok, den, 1.0), tri[:, :, 2].mean(1))
    return out


class OccludedRenderer:
    """The face renderer of correct.py with the known occluders (Occluders) in front of it."""

    def __init__(self, vertices, faces, occluders, kind="gl", dilate_px=DILATE_PX):
        self.face = renderer(vertices, faces, kind)
        self.v, self.f = self.face.v, self.face.f
        self.head = renderer(*occluders.head, kind) if occluders.head is not None else None
        self.board = renderer(*occluders.board, kind) if occluders.board is not None else None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        self.wants_marker_pose = True

    def render(self, K, dist, T, size, T_C_M=None, **kw):
        """Visible face for the camera from B T (4 x 4, mm); the board occluders at T_C_M (skipped without it)."""
        kw.setdefault("edges", False)
        K = np.asarray(K, np.float64)
        dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
        w, h = size
        r = self.face.render(K, dist, T, size, **kw)
        zf = depth(self.face, r, K, dist, T, (h, w))
        sil = np.zeros((h, w), bool)
        front = np.zeros((h, w), bool)
        for ren, P in ((self.head, T), (self.board, T_C_M)):
            if ren is None or P is None:
                continue
            ro = ren.render(K, dist, P, size, edges=False)
            m = ro.mask > 0
            sil |= m
            front |= m & (depth(ren, ro, K, dist, P, (h, w)) < zf)
        hidden = front & (r.mask > 0)
        if hidden.any():
            x0, y0 = r.offset
            bh, bw = r.faces.shape
            r.faces[hidden[y0:y0 + bh, x0:x0 + bw]] = -1
            r.mask[hidden] = 0
        r.occluder = sil            # occluder silhouettes
        r.front = front             # occluders not behind the face (in front of it or of the background)
        r.hidden = hidden           # face pixels taken out
        r.ignore = cv2.dilate(sil.astype(np.uint8), self.kernel) > 0
        return r
