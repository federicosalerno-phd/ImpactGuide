"""Silhouette and edges of a mesh seen by a calibrated camera, for the AR overlay of the mandible (T5.4).

NumPy and OpenCV only. Vertices are projected with cv2.projectPoints (lens distortion included) and the triangles are
painted far to near with their index (painter's algorithm, exact for small triangles that do not intersect): this gives
the silhouette and a visibility buffer. Edges: mesh edges between a triangle facing the camera and one facing away
(occluding contours, the outline included), creases (dihedral angle above crease_deg) and open borders, kept where the
visibility buffer shows them. The mesh must be consistently wound (checked at construction).
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

SHIFT = 4   # sub-pixel bits for the OpenCV drawing functions
KINDS = ("contour", "crease", "border")


@dataclass
class Render:
    mask: np.ndarray       # H x W uint8, 255 on the silhouette
    faces: np.ndarray      # visibility buffer on the region of interest: triangle index, -1 for background
    offset: tuple          # (x0, y0) of the region of interest in the image
    segments: np.ndarray   # M x 2 x 2 visible edges in pixels (OpenCV convention, lens distortion included)
    kind: np.ndarray       # M, index into KINDS
    ms: float              # time spent


class MeshRenderer:
    def __init__(self, vertices, faces, crease_deg=60.0):
        self.v = np.asarray(vertices, np.float64)
        self.f = np.asarray(faces, np.int64)
        nv, nf = len(self.v), len(self.f)
        half = self.f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        lo, hi = half.min(1), half.max(1)
        order = np.argsort(lo * nv + hi, kind="stable")
        _, start, count = np.unique((lo * nv + hi)[order], return_index=True, return_counts=True)
        face_of = np.repeat(np.arange(nf), 3)[order]
        forward = (half[:, 0] < half[:, 1])[order]
        two = count == 2
        self.edges = np.stack([lo[order][start], hi[order][start]], 1)
        self.edge_faces = np.stack([face_of[start], np.where(two, face_of[np.minimum(start + 1, len(order) - 1)], -1)], 1)
        bad = int(np.sum(two & (forward[start] == forward[np.minimum(start + 1, len(order) - 1)])))
        if bad:
            raise ValueError(f"{bad} edges with inconsistent winding")
        self.non_manifold = int(np.sum(count > 2))
        e = np.cross(self.v[self.f[:, 1]] - self.v[self.f[:, 0]], self.v[self.f[:, 2]] - self.v[self.f[:, 0]])
        self.n = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        f0, f1 = self.edge_faces.T
        self.border = f1 < 0
        cos = np.sum(self.n[f0] * self.n[np.maximum(f1, 0)], axis=1)
        self.crease = ~self.border & (cos < np.cos(np.radians(crease_deg))) if crease_deg else np.zeros_like(self.border)

    def render(self, K, dist, T, size, depth_tol=1.0, near=1.0):
        """Silhouette, visibility buffer and visible edges of the mesh for camera from model T (4 x 4, mm)."""
        t0 = time.perf_counter()
        w, h = size
        K = np.asarray(K, np.float64)
        dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
        R, t = T[:3, :3], T[:3, 3]
        xc = self.v @ R.T + t
        uv = cv2.projectPoints(self.v, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2)
        ok = (xc[self.f, 2] > near).all(1)
        idx = np.flatnonzero(ok)
        mask = np.zeros((h, w), np.uint8)
        if not len(idx):
            return Render(mask, np.full((0, 0), -1, np.int32), (0, 0), np.empty((0, 2, 2)), np.empty(0, np.int64), 0.0)
        p = uv[self.f[idx]]
        x0, y0 = np.clip(np.floor(p.reshape(-1, 2).min(0)).astype(int) - 2, 0, (w, h))
        x1, y1 = np.clip(np.ceil(p.reshape(-1, 2).max(0)).astype(int) + 3, 0, (w, h))
        buf = np.full((max(y1 - y0, 0), max(x1 - x0, 0)), -1, np.int32)
        pts = np.round((p - (x0, y0)) * (1 << SHIFT)).astype(np.int32)
        zf = xc[self.f[idx], 2].mean(1)
        for i in np.argsort(-zf, kind="stable"):   # far to near
            cv2.fillConvexPoly(buf, pts[i], int(idx[i]), cv2.LINE_8, SHIFT)
        mask[y0:y1, x0:x1] = np.where(buf >= 0, 255, 0).astype(np.uint8)

        # candidate edges: occluding contours, creases, open borders
        nc = self.n @ R.T
        facing = np.sum(nc * xc[self.f].mean(1), axis=1) < 0
        f0, f1 = self.edge_faces.T
        contour = ~self.border & (facing[f0] != facing[np.maximum(f1, 0)])
        cand = np.flatnonzero((contour | self.crease | self.border) & ok[f0] & (self.border | ok[np.maximum(f1, 0)]))
        a, b = self.edges[cand, 0], self.edges[cand, 1]
        s = np.stack([xc[a] * (1 - u) + xc[b] * u for u in (0.25, 0.5, 0.75)])    # 3 x M x 3, camera frame
        ps = cv2.projectPoints(s.reshape(-1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(s.shape[:2] + (2,))
        ix = np.round(ps[..., 0]).astype(int) - x0
        iy = np.round(ps[..., 1]).astype(int) - y0
        inside = (ix >= 0) & (iy >= 0) & (ix < buf.shape[1]) & (iy < buf.shape[0])
        fid = np.where(inside, buf[np.clip(iy, 0, buf.shape[0] - 1), np.clip(ix, 0, buf.shape[1] - 1)], -1)
        # an edge sample is hidden when the visible triangle there does not touch the edge, crosses the sample's ray
        # in front of it and its plane is farther than depth_tol from the sample (near the outline the visible
        # triangle is a neighbour on the same surface, seen almost edge-on: close to its plane, not in front of it)
        g = np.maximum(fid, 0)
        touches = (self.f[g] == a[None, :, None]).any(2) | (self.f[g] == b[None, :, None]).any(2)
        p0 = xc[self.f[g, 0]]
        off = np.sum(nc[g] * (s - p0), axis=2)                  # signed distance of the sample from the plane
        denom = np.sum(nc[g] * s, axis=2)                       # ray through the sample: x = s * z_hit / s_z
        in_front = np.abs(denom) > 1e-9
        z_hit = np.where(in_front, np.sum(nc[g] * p0, axis=2) * s[..., 2] / np.where(in_front, denom, 1.0), np.inf)
        hidden = (fid >= 0) & ~touches & (z_hit < s[..., 2]) & (np.abs(off) > depth_tol)
        vis = cand[(~hidden).sum(0) >= 2]
        kind = np.where(self.border[vis], 2, np.where(contour[vis], 0, 1))
        segments = uv[self.edges[vis]]
        return Render(mask, buf, (int(x0), int(y0)), segments, kind, (time.perf_counter() - t0) * 1e3)


def draw_segments(img, segments, color, thickness=1, antialias=True):
    lines = [s.reshape(-1, 1, 2) for s in np.round(segments * (1 << SHIFT)).astype(np.int32)]
    cv2.polylines(img, lines, False, color, thickness, cv2.LINE_AA if antialias else cv2.LINE_8, SHIFT)
    return img


def edge_image(render, thickness=1):
    """Binary image (uint8, 255) of the visible edges."""
    return draw_segments(np.zeros_like(render.mask), render.segments, 255, thickness, antialias=False)


def overlay(image, render, color=(90, 220, 60), fill_alpha=0.3, thickness=None):
    """Image with the silhouette tinted and the visible edges drawn (BGR colour)."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    m = render.mask > 0
    out[m] = (out[m] * (1 - fill_alpha) + np.array(color) * fill_alpha).astype(np.uint8)
    thickness = thickness or max(1, round(out.shape[1] / 1000))
    return draw_segments(out, render.segments, color, thickness)
