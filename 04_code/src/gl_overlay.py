"""OpenGL renderer with the interface of mesh_overlay.MeshRenderer: silhouette, visibility buffer and visible edges of
a mesh seen by a calibrated camera.

Same projection as mesh_overlay.py: vertices projected with cv2.projectPoints (lens distortion included), triangles
straight between the projected vertices. The GPU (moderngl, OpenGL 3.3 standalone context) rasterises the index of the
triangle at each pixel centre with a depth test on the camera z, so the visibility is exact where the painter's
algorithm of mesh_overlay.py sorts whole triangles. Only the region of interest (bounding box of the projection) is
drawn and read back. Edges are computed from the buffer with the same test as mesh_overlay.py; render(..., edges=False)
skips them when only mask and buffer are needed (correction, scoring).

For the live view, overlay() draws the same triangles for the picture instead: multisampled (SAMPLES per pixel), so
the silhouette comes with its true coverage (anti-aliased, steady under a sub-pixel motion), with a value per vertex
(the light the caller gives) interpolated across the triangles, so a shading is smooth instead of faceted.

    r = GLMeshRenderer(vertices, faces).render(K, dist, T, (w, h))      # same Render as mesh_overlay.py
    o = GLMeshRenderer(vertices, faces).overlay(K, dist, T, (w, h), light)   # Overlay: coverage and light

OpenGL contexts. A context is current on the thread that made it. context() is the one shared context of a process,
for callers that render from one thread (the research scripts). A thread that renders on its own (the compose thread
of the live view) makes its own with standalone_context(), passes it as GLMeshRenderer(..., ctx=ctx), and releases
the renderers (release()) and then the context on that same thread when it ends.

    python gl_overlay.py test     the renderer on and off the image, overlay(), a context of its own released
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

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
VS_LIT = """#version 330
in vec3 in_pos;
in float in_light;
out float v_light;
void main() { gl_Position = vec4(in_pos, 1.0); v_light = in_light; }
"""
FS_LIT = """#version 330
in float v_light;
out vec2 f_out;
void main() { f_out = vec2(v_light, 1.0); }
"""
SAMPLES = 8           # samples per pixel of overlay(): 9 levels of coverage on the silhouette
_CTX = None


@dataclass
class Overlay:
    mask: np.ndarray       # H x W uint8, 255 where more than half of the pixel is covered (the silhouette)
    offset: tuple          # (x0, y0) of the region of interest in the image
    cover: np.ndarray      # region of interest, uint8: share of the pixel covered by the model (0-255)
    light: np.ndarray      # region of interest, uint8: the light times the coverage (premultiplied, 0-255)
    ms: float              # time spent


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
    """The shared standalone context of the process, made at the first call (about 0.4 s) on the calling thread and
    current there only: for callers that render from one thread."""
    global _CTX
    if _CTX is None:
        _CTX = moderngl.create_standalone_context()
    return _CTX


def standalone_context():
    """A new standalone context, current on the calling thread, owned by the caller: renderers made with it render on
    that thread only, and the caller releases them and then the context (ctx.release()) on that thread."""
    return moderngl.create_standalone_context()


def _release_framebuffer(fb):
    """A framebuffer and its attachments (moderngl releases neither with the other)."""
    if fb is None:
        return
    for a in tuple(fb.color_attachments) + ((fb.depth_attachment,) if fb.depth_attachment else ()):
        a.release()
    fb.release()


class GLMeshRenderer(MeshRenderer):
    """MeshRenderer on the GPU (see the module docstring).

    Args:
        vertices: N x 3 vertices (mm).
        faces: M x 3 triangles, consistently wound.
        crease_deg: dihedral angle above which an edge is a crease.
        ctx: the OpenGL context to render with (standalone_context() of the calling thread); None for the shared
            context of the process (context()).
    """

    def __init__(self, vertices, faces, crease_deg=60.0, ctx=None):
        super().__init__(vertices, faces, crease_deg)
        ctx = self.ctx = ctx if ctx is not None else context()
        self.thread = threading.get_ident()     # the thread the context is current on: the only one to render
        self.prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.vbo = ctx.buffer(reserve=len(self.v) * 12)
        self.ibo_all = ctx.buffer(self.f.astype("i4").tobytes())
        self.vao_all = ctx.vertex_array(self.prog, [(self.vbo, "3f", "in_pos")], index_buffer=self.ibo_all,
                                        index_element_size=4)
        self.fbo, self.cap = None, (0, 0)
        self.used = np.unique(self.f)
        self.n_dot_c = np.sum(self.n * self.v[self.f].mean(1), axis=1)     # facing test without the 3 x 3 gather
        # overlay(): positions and a light per vertex, multisampled target and its resolved copy
        self.prog_lit = ctx.program(vertex_shader=VS_LIT, fragment_shader=FS_LIT)
        self.vbo_light = ctx.buffer(reserve=len(self.v) * 4)
        self.vao_lit = ctx.vertex_array(self.prog_lit, [(self.vbo, "3f", "in_pos"), (self.vbo_light, "1f", "in_light")],
                                        index_buffer=self.ibo_all, index_element_size=4)
        self.fbo_ms, self.fbo_res, self.cap_lit = None, None, (0, 0)
        vn = np.zeros_like(self.v)       # vertex normals: the triangles' normals weighted by their area
        e = np.cross(self.v[self.f[:, 1]] - self.v[self.f[:, 0]], self.v[self.f[:, 2]] - self.v[self.f[:, 0]])
        for k in range(3):
            np.add.at(vn, self.f[:, k], e)
        self.vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)

    def _target(self, w, h):
        if w > self.cap[0] or h > self.cap[1]:
            cw, ch = max(w, self.cap[0]), max(h, self.cap[1])
            cw, ch = -(-cw // 256) * 256, -(-ch // 256) * 256
            _release_framebuffer(self.fbo)
            self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((cw, ch), 1, dtype="i4")],
                                            depth_attachment=self.ctx.depth_renderbuffer((cw, ch)))
            self.cap = (cw, ch)
        return self.fbo

    def _place(self, K, dist, T, size, near):
        """The vertices in the camera (xc), projected (uv), the triangles in front of the near plane (ok: None for all)
        and the region of interest (x0, y0, x1, y1), with the vertices written to the buffer in its clip space; the
        region is empty when nothing is in front of the camera or inside the image."""
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
        uv = np.zeros((len(self.v), 2))
        if not len(used):
            return xc, uv, ok, (0, 0, 0, 0), K, dist
        uv[front] = project(xc[front], K, dist)
        x0, y0 = np.clip(np.floor(uv[used].min(0)).astype(int) - 2, 0, (w, h))
        x1, y1 = np.clip(np.ceil(uv[used].max(0)).astype(int) + 3, 0, (w, h))
        bw, bh = max(x1 - x0, 0), max(y1 - y0, 0)
        if bw and bh:
            z = xc[:, 2]
            zf = z[used].max() + 1.0
            pos = np.empty((len(self.v), 3), np.float32)
            pos[:, 0] = 2.0 * (uv[:, 0] - x0 + 0.5) / bw - 1.0      # pixel centres of the region at integer + 0.5
            pos[:, 1] = 2.0 * (uv[:, 1] - y0 + 0.5) / bh - 1.0      # rows not flipped: row 0 of the read = row y0
            pos[:, 2] = 2.0 * (z - near) / (zf - near) - 1.0
            self.vbo.write(pos.tobytes())
        return xc, uv, ok, (int(x0), int(y0), int(x0 + bw), int(y0 + bh)), K, dist

    def facing(self, T):
        """|cos| of the angle between each vertex normal and the line of sight from the camera of model T: 1 face on,
        0 edge on (both sides alike), for the light of overlay()."""
        xc = self.v @ T[:3, :3].T + T[:3, 3]
        c = np.abs(np.sum((self.vn @ T[:3, :3].T) * xc, axis=1))
        return c / np.maximum(np.linalg.norm(xc, axis=1), 1e-9)

    def release(self):
        """Frees the GPU objects of this renderer (on the thread its context is current on); it cannot render after.
        The context itself stays: its owner releases it."""
        for fb in (self.fbo, self.fbo_ms, self.fbo_res):
            _release_framebuffer(fb)
        self.fbo, self.fbo_ms, self.fbo_res = None, None, None
        self.cap, self.cap_lit = (0, 0), (0, 0)
        for name in ("vao_all", "vao_lit", "ibo_all", "vbo", "vbo_light", "prog", "prog_lit"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.release()
                setattr(self, name, None)

    def render(self, K, dist, T, size, depth_tol=1.0, near=1.0, edges=True):
        """Silhouette, visibility buffer and visible edges of the mesh for camera from model T (4 x 4, mm)."""
        t0 = time.perf_counter()
        w, h = size
        R, t = T[:3, :3], T[:3, 3]
        xc, uv, ok, (x0, y0, x1, y1), K, dist = self._place(K, dist, T, size, near)
        bw, bh = x1 - x0, y1 - y0
        mask = np.zeros((h, w), np.uint8)
        if bw == 0 or bh == 0:
            return Render(mask, np.full((0, 0), -1, np.int32), (x0, y0), np.empty((0, 2, 2)), np.empty(0, np.int64),
                          (time.perf_counter() - t0) * 1e3)
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

    def _target_lit(self, w, h):
        if w > self.cap_lit[0] or h > self.cap_lit[1]:
            cw, ch = max(w, self.cap_lit[0]), max(h, self.cap_lit[1])
            cw, ch = -(-cw // 128) * 128, -(-ch // 128) * 128
            for fb in (self.fbo_ms, self.fbo_res):
                _release_framebuffer(fb)
            self.fbo_ms = self.ctx.framebuffer(
                color_attachments=[self.ctx.renderbuffer((cw, ch), 2, samples=SAMPLES)],
                depth_attachment=self.ctx.depth_renderbuffer((cw, ch), samples=SAMPLES))
            self.fbo_res = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer((cw, ch), 2)])
            self.cap_lit = (cw, ch)
        return self.fbo_ms, self.fbo_res

    def overlay(self, K, dist, T, size, light=None, near=1.0):
        """The model for the picture, camera from model T (4 x 4, mm): the share of each pixel of the region of
        interest it covers (multisampled, depth tested per sample) and a light per vertex (len(vertices) values in
        0-1, e.g. from facing(T); None for 1) interpolated across the triangles, premultiplied by that coverage. The
        mask is the silhouette at half coverage."""
        t0 = time.perf_counter()
        w, h = size
        _, _, ok, (x0, y0, x1, y1), _, _ = self._place(K, dist, T, size, near)
        bw, bh = x1 - x0, y1 - y0
        mask = np.zeros((h, w), np.uint8)
        if bw == 0 or bh == 0:
            e = np.zeros((0, 0), np.uint8)
            return Overlay(mask, (x0, y0), e, e, (time.perf_counter() - t0) * 1e3)
        lv = np.ones(len(self.v), np.float32) if light is None else np.clip(np.asarray(light, np.float32), 0, 1)
        self.vbo_light.write(lv.tobytes())
        ms, res = self._target_lit(bw, bh)
        ms.use()
        ms.viewport = (0, 0, bw, bh)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        ms.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        if ok is None:
            self.vao_lit.render(moderngl.TRIANGLES)
        else:     # triangles behind the near plane left out
            ibo = self.ctx.buffer(self.f[np.flatnonzero(ok)].astype("i4").tobytes())
            vao = self.ctx.vertex_array(self.prog_lit, [(self.vbo, "3f", "in_pos"), (self.vbo_light, "1f", "in_light")],
                                        index_buffer=ibo, index_element_size=4)
            vao.render(moderngl.TRIANGLES)
            vao.release()
            ibo.release()
        self.ctx.copy_framebuffer(res, ms)          # the samples averaged: coverage, and light times coverage
        raw = np.frombuffer(res.read(viewport=(0, 0, bw, bh), components=2, dtype="f1"), np.uint8).reshape(bh, bw, 2)
        lit, cover = cv2.split(raw)
        mask[y0:y1, x0:x1] = cv2.compare(cover, 127, cv2.CMP_GT)
        return Overlay(mask, (x0, y0), cover, lit, (time.perf_counter() - t0) * 1e3)

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


# ── checks ───────────────────────────────────────────────────────────────────
def run_test():
    """mesh_overlay's checks on this renderer, overlay() on and off the image, and a context of a thread's own: two
    renderers on two threads, each with its context, released on their thread, the shared context untouched."""
    import mesh_overlay
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    v, f = mesh_overlay.check_cube()
    K = np.array([[600.0, 0, 319.5], [0, 600.0, 239.5], [0, 0, 1]])
    size = (640, 480)
    out = {}

    def own(key):
        ctx = standalone_context()
        ren = GLMeshRenderer(v, f, ctx=ctx)
        rows = {}
        for where, T in mesh_overlay.check_poses().items():
            o = ren.overlay(K, None, T, size, ren.facing(T))
            rows[where] = (int(np.count_nonzero(o.mask)), o.cover.shape)
        ren.release()
        ctx.release()
        out[key] = rows

    shared_before = _CTX
    threads = [threading.Thread(target=own, args=(k,), name=f"gl_check_{k}") for k in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30.0)
    seen = ("inside", "across the left edge")
    ok = len(out) == 2 and all((rows[w][0] > 0) == (w in seen) for rows in out.values() for w in rows)
    expect(ok and _CTX is shared_before, "overlay() on two threads with a context each, released there: covered px "
           + ", ".join(f"{w} {n}" for w, (n, _) in out.get(0, {}).items()))
    if mesh_overlay.run_test(GLMeshRenderer, "gl_overlay"):
        fails.append("render() on and off the image")
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        sys.exit(run_test())
    sys.exit(__doc__)
