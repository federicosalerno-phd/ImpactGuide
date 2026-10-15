"""OpenGL renderer of module 1 with the interface of mesh_overlay.MeshRenderer (T13.1): silhouette, visibility buffer
and visible edges of a mesh seen by a calibrated camera.

Same projection as mesh_overlay.py: vertices projected with cv2.projectPoints (lens distortion included), triangles
straight between the projected vertices. The GPU (moderngl, OpenGL 3.3 standalone context) rasterises the index of the
triangle at each pixel centre with a depth test on the camera z, so the visibility is exact where the painter's
algorithm of mesh_overlay.py sorts whole triangles. Only the region of interest (bounding box of the projection) is
drawn and read back. Edges are computed from the buffer with the same test as mesh_overlay.py; render(..., edges=False)
skips them when only mask and buffer are needed (correction, scoring).

    r = GLMeshRenderer(vertices, faces).render(K, dist, T, (w, h))      # same Render as mesh_overlay.py
"""
import time

import cv2
import moderngl
import numpy as np

from mesh_overlay import MeshRenderer, Render

VS = """#version 330
in vec3 in_pos;
void main() { gl_Position = vec4(in_pos, 1.0); }
"""
FS = """#version 330
out int f_id;
void main() { f_id = gl_PrimitiveID + 1; }
"""
_CTX = None


def project(xc, K, dist):
    """Pixel coordinates of camera-frame points with OpenCV's distortion model (k1 k2 p1 p2 [k3 [k4 k5 k6]]), the same
    numbers as cv2.projectPoints, in NumPy (about ten times faster for 10 000 points); longer models go to OpenCV."""
    d = np.zeros(8)
    if len(dist) > 8:
        return cv2.projectPoints(xc, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    d[:len(dist)] = dist
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    z = xc[:, 2]
    x, y = xc[:, 0] / z, xc[:, 1] / z
    if not d.any():
        return np.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], 1)
    r2 = x * x + y * y
    radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)


def context():
    """One standalone context per process (creating one takes about 0.4 s)."""
    global _CTX
    if _CTX is None:
        _CTX = moderngl.create_standalone_context()
    return _CTX


class GLMeshRenderer(MeshRenderer):
    def __init__(self, vertices, faces, crease_deg=60.0):
        super().__init__(vertices, faces, crease_deg)
        ctx = self.ctx = context()
        self.prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.vbo = ctx.buffer(reserve=len(self.v) * 12)
        self.ibo_all = ctx.buffer(self.f.astype("i4").tobytes())
        self.vao_all = ctx.vertex_array(self.prog, [(self.vbo, "3f", "in_pos")], index_buffer=self.ibo_all,
                                        index_element_size=4)
        self.fbo, self.cap = None, (0, 0)
        self.used = np.unique(self.f)
        self.n_dot_c = np.sum(self.n * self.v[self.f].mean(1), axis=1)     # facing test without the 3 x 3 gather

    def _target(self, w, h):
        if w > self.cap[0] or h > self.cap[1]:
            cw, ch = max(w, self.cap[0]), max(h, self.cap[1])
            cw, ch = -(-cw // 256) * 256, -(-ch // 256) * 256
            if self.fbo is not None:
                self.fbo.release()
            self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((cw, ch), 1, dtype="i4")],
                                            depth_attachment=self.ctx.depth_renderbuffer((cw, ch)))
            self.cap = (cw, ch)
        return self.fbo

    def render(self, K, dist, T, size, depth_tol=1.0, near=1.0, edges=True):
        """Silhouette, visibility buffer and visible edges of the mesh for camera from model T (4 x 4, mm)."""
        t0 = time.perf_counter()
        w, h = size
        K = np.asarray(K, np.float64)
        dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
        R, t = T[:3, :3], T[:3, 3]
        xc = self.v @ R.T + t
        front = xc[:, 2] > near
        if front[self.used].all():
            ok, used = None, self.used
        else:
            ok = front[self.f].all(1)
            used = self.f[ok].ravel()
        mask = np.zeros((h, w), np.uint8)
        if not len(used):
            return Render(mask, np.full((0, 0), -1, np.int32), (0, 0), np.empty((0, 2, 2)), np.empty(0, np.int64), 0.0)
        uv = np.zeros((len(self.v), 2))
        uv[front] = project(xc[front], K, dist)
        x0, y0 = np.clip(np.floor(uv[used].min(0)).astype(int) - 2, 0, (w, h))
        x1, y1 = np.clip(np.ceil(uv[used].max(0)).astype(int) + 3, 0, (w, h))
        bw, bh = max(x1 - x0, 0), max(y1 - y0, 0)
        if bw == 0 or bh == 0:
            return Render(mask, np.full((0, 0), -1, np.int32), (int(x0), int(y0)), np.empty((0, 2, 2)),
                          np.empty(0, np.int64), (time.perf_counter() - t0) * 1e3)
        z = xc[:, 2]
        zf = z[used].max() + 1.0
        pos = np.empty((len(self.v), 3), np.float32)
        pos[:, 0] = 2.0 * (uv[:, 0] - x0 + 0.5) / bw - 1.0      # pixel centres of the region at integer + 0.5
        pos[:, 1] = 2.0 * (uv[:, 1] - y0 + 0.5) / bh - 1.0      # rows not flipped: row 0 of the read = row y0
        pos[:, 2] = 2.0 * (z - near) / (zf - near) - 1.0
        self.vbo.write(pos.tobytes())
        fbo = self._target(bw, bh)
        fbo.use()
        fbo.viewport = (0, 0, bw, bh)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        if ok is None:
            self.vao_all.render(moderngl.TRIANGLES)
            ids = None
        else:     # triangles behind the near plane are left out; map the primitive ids back
            ids = np.flatnonzero(ok)
            ibo = self.ctx.buffer(self.f[ids].astype("i4").tobytes())
            vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f", "in_pos")], index_buffer=ibo, index_element_size=4)
            vao.render(moderngl.TRIANGLES)
            vao.release()
            ibo.release()
        raw = np.frombuffer(fbo.read(viewport=(0, 0, bw, bh), components=1, dtype="i4"), np.int32).reshape(bh, bw)
        buf = raw - 1
        if ids is not None:
            buf = np.where(buf >= 0, ids[np.maximum(buf, 0)], -1).astype(np.int32)
        mask[y0:y1, x0:x1] = np.where(buf >= 0, 255, 0).astype(np.uint8)
        if ok is None:
            ok = np.ones(len(self.f), bool)
        if not edges:
            return Render(mask, buf, (int(x0), int(y0)), np.empty((0, 2, 2)), np.empty(0, np.int64),
                          (time.perf_counter() - t0) * 1e3)
        segments, kind = self._edges(xc, uv, ok, buf, x0, y0, R, t, K, dist, depth_tol)
        return Render(mask, buf, (int(x0), int(y0)), segments, kind, (time.perf_counter() - t0) * 1e3)

    def _edges(self, xc, uv, ok, buf, x0, y0, R, t, K, dist, depth_tol):
        """Visible contours, creases and open borders: the edge test of mesh_overlay.MeshRenderer.render."""
        nc = self.n @ R.T
        facing = self.n_dot_c + self.n @ (R.T @ t) < 0
        f0, f1 = self.edge_faces.T
        contour = ~self.border & (facing[f0] != facing[np.maximum(f1, 0)])
        cand = np.flatnonzero((contour | self.crease | self.border) & ok[f0] & (self.border | ok[np.maximum(f1, 0)]))
        a, b = self.edges[cand, 0], self.edges[cand, 1]
        s = np.stack([xc[a] * (1 - u) + xc[b] * u for u in (0.25, 0.5, 0.75)])
        ps = project(s.reshape(-1, 3), K, dist).reshape(s.shape[:2] + (2,))
        ix = np.round(ps[..., 0]).astype(int) - x0
        iy = np.round(ps[..., 1]).astype(int) - y0
        inside = (ix >= 0) & (iy >= 0) & (ix < buf.shape[1]) & (iy < buf.shape[0])
        fid = np.where(inside, buf[np.clip(iy, 0, buf.shape[0] - 1), np.clip(ix, 0, buf.shape[1] - 1)], -1)
        g = np.maximum(fid, 0)
        touches = (self.f[g] == a[None, :, None]).any(2) | (self.f[g] == b[None, :, None]).any(2)
        p0 = xc[self.f[g, 0]]
        off = np.sum(nc[g] * (s - p0), axis=2)
        denom = np.sum(nc[g] * s, axis=2)
        in_front = np.abs(denom) > 1e-9
        z_hit = np.where(in_front, np.sum(nc[g] * p0, axis=2) * s[..., 2] / np.where(in_front, denom, 1.0), np.inf)
        hidden = (fid >= 0) & ~touches & (z_hit < s[..., 2]) & (np.abs(off) > depth_tol)
        vis = cand[(~hidden).sum(0) >= 2]
        kind = np.where(self.border[vis], 2, np.where(contour[vis], 0, 1))
        return uv[self.edges[vis]], kind
