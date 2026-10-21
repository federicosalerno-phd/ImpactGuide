// ImpactGuide · the phone's lens in the twin's view from the phone.
//
// The view from the phone shows the bench as the phone's own pictures do: through its calibrated lens (OpenCV's
// model: radial k1 k2 k3, with k4 k5 k6 of the rational one when calibrated, tangential p1 p2), with its own focal
// lengths and optical centre, the picture fitted in the box and centred. The bench is drawn as a pinhole would see
// it, a little wider than the box and at least as finely as the box shows it anywhere, then bent: each pixel of the
// box takes the colour of the ray the lens sends to it (the inverse of the distortion, a few Newton steps in the
// shader), read with a sharp cubic filter. Past the picture's edges the bending goes on as it is at the edge (it
// never folds back). Any other view is drawn straight.
'use strict';

const LENS = {
  NEWTON: 6,            // Newton steps of the inverse of the distortion (in the shader and here)
  GRID: 24,             // the box looked at on a grid this fine for the field and the fineness of the drawing
  MARGIN_PX: 3,         // the drawing this many of its pixels wider than what the bending reads (the cubic's reach)
  DENSITY: 1.0,         // the drawing's pixels per pixel of the box where the box shows it most finely, at least
  MAX_SCALE: 1.6,       // ... and at most this many times the canvas each way
  SAMPLES: 4,           // the drawing's antialiasing, as the canvas has it

  // ── the model ──────────────────────────────────────────────────────────────
  // a calibration as twin.py's lens_of gives it (size, K, dist) made ready; null when it bends nothing (no
  // distortion, the optical centre in the middle, square pixels: the straight view is the same)
  make: function (c) {
    if (!c || !c.size || !c.K) return null;
    const d = (c.dist || []).concat([0, 0, 0, 0, 0, 0, 0, 0]).slice(0, 8).map(Number);
    const L = { w: c.size[0], h: c.size[1], fx: c.K[0][0], fy: c.K[1][1], cx: c.K[0][2], cy: c.K[1][2], k: d };
    if (!(L.w > 0 && L.h > 0 && L.fx > 0 && L.fy > 0) || !d.concat([L.cx, L.cy]).every(isFinite)) return null;
    const straight = d.every(function (x) { return x === 0; }) && Math.abs(L.fx - L.fy) < 1e-6
      && Math.abs(L.cx - (L.w - 1) / 2) < 1e-3 && Math.abs(L.cy - (L.h - 1) / 2) < 1e-3;
    if (straight) return null;
    // how far from the axis the picture reaches (its edges, as ideal points): beyond it the polynomial is not used
    let r2 = 0;
    for (let i = 0; i <= 64; i++) {
      const a = i / 64;
      [[a * L.w - 0.5, -0.5], [a * L.w - 0.5, L.h - 0.5], [-0.5, a * L.h - 0.5], [L.w - 0.5, a * L.h - 0.5]]
        .forEach(function (q) {
          const p = LENS.unbend(L, (q[0] - L.cx) / L.fx, (q[1] - L.cy) / L.fy);
          r2 = Math.max(r2, p[0] * p[0] + p[1] * p[1]);
        });
    }
    L.r2 = r2 * 1.0001;
    L.key = [L.w, L.h, L.fx, L.fy, L.cx, L.cy].concat(d).join();
    return L;
  },
  // ... kept for the calibration it was made from
  of: function (c) {
    if (c !== LENS.lastC) { LENS.lastC = c; LENS.lastL = LENS.make(c); }
    return LENS.lastL;
  },
  lastC: null, lastL: null,

  // an ideal point (x, y on the plane 1 ahead of the lens, y down, as OpenCV has it) where the lens puts it, and the
  // Jacobian of that: [xd, yd, dxd/dx, dxd/dy, dyd/dx, dyd/dy]
  bend: function (L, x, y) {
    const k = L.k, r2 = x * x + y * y, p1 = k[2], p2 = k[3];
    const a = 1 + r2 * (k[0] + r2 * (k[1] + r2 * k[4])), b = 1 + r2 * (k[5] + r2 * (k[6] + r2 * k[7]));
    const da = k[0] + r2 * (2 * k[1] + 3 * r2 * k[4]), db = k[5] + r2 * (2 * k[6] + 3 * r2 * k[7]);
    const R = a / b, dR = (da * b - a * db) / (b * b), cross = 2 * x * y * dR + 2 * p1 * x + 2 * p2 * y;
    return [x * R + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * R + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
            R + 2 * x * x * dR + 2 * p1 * y + 6 * p2 * x, cross, cross, R + 2 * y * y * dR + 6 * p1 * y + 2 * p2 * x];
  },
  // the ideal point the lens puts at (xd, yd): Newton's steps from there
  unbend: function (L, xd, yd) {
    let x = xd, y = yd;
    for (let i = 0; i < LENS.NEWTON; i++) {
      const f = LENS.bend(L, x, y), ex = f[0] - xd, ey = f[1] - yd, det = f[2] * f[5] - f[3] * f[4];
      if (!(Math.abs(det) > 1e-12)) break;
      x -= (f[5] * ex - f[3] * ey) / det;
      y -= (f[2] * ey - f[4] * ex) / det;
    }
    return [x, y];
  },
  // the ideal point of a point of the picture (u, v px, OpenCV's: the first pixel's centre at 0, 0); past its edges
  // from the nearest point of the edge straight on, as the lens bends there
  ideal: function (L, u, v) {
    const uc = Math.max(-0.5, Math.min(L.w - 0.5, u)), vc = Math.max(-0.5, Math.min(L.h - 0.5, v));
    const p = LENS.unbend(L, (uc - L.cx) / L.fx, (vc - L.cy) / L.fy);
    if (uc === u && vc === v) return p;
    const f = LENS.bend(L, p[0], p[1]), det = f[2] * f[5] - f[3] * f[4];
    const dx = (u - uc) / L.fx, dy = (v - vc) / L.fy;
    return [p[0] + (f[5] * dx - f[3] * dy) / det, p[1] + (f[2] * dy - f[4] * dx) / det];
  },
  // where the picture shows an ideal point (px): the inverse of ideal()
  photo: function (L, x, y) {
    const f = LENS.bend(L, x, y), u0 = L.fx * f[0] + L.cx, v0 = L.fy * f[1] + L.cy;
    if (x * x + y * y <= L.r2 && u0 >= -0.5 && u0 <= L.w - 0.5 && v0 >= -0.5 && v0 <= L.h - 0.5) return [u0, v0];
    let u = L.fx * x + L.cx, v = L.fy * y + L.cy;
    for (let i = 0; i < 10; i++) {
      const e = 0.5, a = LENS.ideal(L, u, v), b = LENS.ideal(L, u + e, v), c = LENS.ideal(L, u, v + e);
      const j0 = (b[0] - a[0]) / e, j1 = (c[0] - a[0]) / e, j2 = (b[1] - a[1]) / e, j3 = (c[1] - a[1]) / e;
      const det = j0 * j3 - j1 * j2, ex = x - a[0], ey = y - a[1];
      if (!(Math.abs(det) > 1e-20)) break;
      const du = (j3 * ex - j1 * ey) / det, dv = (j0 * ey - j2 * ex) / det;
      u += du; v += dv;
      if (Math.abs(du) + Math.abs(dv) < 1e-4) break;
    }
    return [u, v];
  },
  // the picture in a box of W x H: its scale (box px per px of the picture) and its top left corner
  fit: function (L, W, H) {
    const s = Math.min(W / L.w, H / L.h);
    return { s: s, ox: (W - L.w * s) / 2, oy: (H - L.h * s) / 2 };
  },
  boxToIdeal: function (L, F, X, Y) { return LENS.ideal(L, (X - F.ox) / F.s - 0.5, (Y - F.oy) / F.s - 0.5); },

  // the drawing the bending reads, for a box of W x H css px on a canvas of DW x DH: the half widths of its field
  // (on the plane 1 ahead, about the lens's axis) and its size in pixels, fine enough that no pixel of the box
  // reads less than DENSITY of its pixels
  field: function (L, W, H, DW, DH) {
    const F = LENS.fit(L, W, H), N = LENS.GRID;
    let tx = 0, ty = 0, finest = Infinity;
    for (let i = 0; i <= N; i++) {
      for (let j = 0; j <= N; j++) {
        const X = W * i / N, Y = H * j / N, p = LENS.boxToIdeal(L, F, X, Y);
        tx = Math.max(tx, Math.abs(p[0])); ty = Math.max(ty, Math.abs(p[1]));
        // a pixel of the box there, on the ideal plane: the smaller way of the map's Jacobian (singular value)
        const e = 0.5, a = LENS.boxToIdeal(L, F, X + e, Y), b = LENS.boxToIdeal(L, F, X, Y + e);
        const m0 = (a[0] - p[0]) / e, m1 = (b[0] - p[0]) / e, m2 = (a[1] - p[1]) / e, m3 = (b[1] - p[1]) / e;
        const T = m0 * m0 + m1 * m1 + m2 * m2 + m3 * m3, D = m0 * m3 - m1 * m2;
        finest = Math.min(finest, Math.sqrt(Math.max(0, (T - Math.sqrt(Math.max(0, T * T - 4 * D * D))) / 2)));
      }
    }
    const kx = DW / W, ky = DH / H;
    const size = function (t, k, D) {
      return Math.max(2, Math.min(Math.floor(D * LENS.MAX_SCALE), Math.ceil(2 * t / finest * k * LENS.DENSITY)));
    };
    let RW = size(tx, kx, DW), RH = size(ty, ky, DH);
    tx *= 1 + 2 * LENS.MARGIN_PX / RW; ty *= 1 + 2 * LENS.MARGIN_PX / RH;
    RW = size(tx, kx, DW); RH = size(ty, ky, DH);
    return { tx: tx, ty: ty, RW: RW, RH: RH, fit: F };
  },

  // ── the pass ───────────────────────────────────────────────────────────────
  // the bending: for a pixel of the box, the ray of the picture it shows, read in the drawing (probe: the place
  // read, for the tests, instead of its colour)
  material: function () {
    const N = String(LENS.NEWTON);
    return new THREE.ShaderMaterial({
      uniforms: { src: { value: null }, box: { value: new THREE.Vector2() }, fit: { value: new THREE.Vector3() },
                  f: { value: new THREE.Vector2() }, c: { value: new THREE.Vector2() }, wh: { value: new THREE.Vector2() },
                  ka: { value: new THREE.Vector4() }, kb: { value: new THREE.Vector4() }, t: { value: new THREE.Vector2() },
                  texel: { value: new THREE.Vector2() }, probe: { value: 0 } },
      vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: [
        'uniform sampler2D src; uniform vec2 box, f, c, wh, t, texel; uniform vec3 fit; uniform vec4 ka, kb;',
        'uniform float probe; varying vec2 vUv;',
        // the lens: where it puts an ideal point, and its Jacobian (by columns)
        'vec2 bend(vec2 p, out mat2 J) {',
        '  float r2 = dot(p, p), x = p.x, y = p.y, p1 = ka.z, p2 = ka.w;',
        '  float a = 1.0 + r2 * (ka.x + r2 * (ka.y + r2 * kb.x)), b = 1.0 + r2 * (kb.y + r2 * (kb.z + r2 * kb.w));',
        '  float da = ka.x + r2 * (2.0 * ka.y + 3.0 * r2 * kb.x), db = kb.y + r2 * (2.0 * kb.z + 3.0 * r2 * kb.w);',
        '  float R = a / b, dR = (da * b - a * db) / (b * b), cr = 2.0 * x * y * dR + 2.0 * p1 * x + 2.0 * p2 * y;',
        '  J = mat2(R + 2.0 * x * x * dR + 2.0 * p1 * y + 6.0 * p2 * x, cr, cr,',
        '           R + 2.0 * y * y * dR + 6.0 * p1 * y + 2.0 * p2 * x);',
        '  return vec2(x * R + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x), y * R + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y);',
        '}',
        'mat2 inv(mat2 J) { return mat2(J[1][1], -J[0][1], -J[1][0], J[0][0]) / (J[0][0] * J[1][1] - J[1][0] * J[0][1]); }',
        'vec2 unbend(vec2 d) {',
        '  vec2 p = d; mat2 J;',
        '  for (int i = 0; i < ' + N + '; i++) { vec2 e = bend(p, J) - d; p -= inv(J) * e; }',
        '  return p;',
        '}',
        // Catmull-Rom in nine reads (the bilinear filter does the rest)
        'vec4 cubic(vec2 uv) {',
        '  vec2 n = 1.0 / texel, q = uv * n, p1 = floor(q - 0.5) + 0.5, g = q - p1;',
        '  vec2 w0 = g * (-0.5 + g * (1.0 - 0.5 * g)), w1 = 1.0 + g * g * (-2.5 + 1.5 * g);',
        '  vec2 w2 = g * (0.5 + g * (2.0 - 1.5 * g)), w3 = g * g * (-0.5 + 0.5 * g), w12 = w1 + w2;',
        '  vec2 a = (p1 - 1.0) * texel, m = (p1 + w2 / w12) * texel, z = (p1 + 2.0) * texel;',
        '  vec4 r = (texture2D(src, vec2(a.x, a.y)) * w0.x + texture2D(src, vec2(m.x, a.y)) * w12.x',
        '          + texture2D(src, vec2(z.x, a.y)) * w3.x) * w0.y;',
        '  r += (texture2D(src, vec2(a.x, m.y)) * w0.x + texture2D(src, vec2(m.x, m.y)) * w12.x',
        '          + texture2D(src, vec2(z.x, m.y)) * w3.x) * w12.y;',
        '  r += (texture2D(src, vec2(a.x, z.y)) * w0.x + texture2D(src, vec2(m.x, z.y)) * w12.x',
        '          + texture2D(src, vec2(z.x, z.y)) * w3.x) * w3.y;',
        '  return clamp(r, 0.0, 1.0);',
        '}',
        'void main() {',
        '  vec2 X = vec2(vUv.x * box.x, (1.0 - vUv.y) * box.y);',
        '  vec2 u = (X - fit.yz) / fit.x - 0.5, uc = clamp(u, vec2(-0.5), wh - 0.5);',
        '  vec2 p = unbend((uc - c) / f);',
        '  if (u != uc) { mat2 J; bend(p, J); p += inv(J) * ((u - uc) / f); }',
        '  vec2 s = vec2(p.x / t.x, -p.y / t.y) * 0.5 + 0.5;',
        '  if (probe > 0.5) { gl_FragColor = vec4(s, 0.0, 1.0); return; }',
        '  gl_FragColor = vec4(cubic(s).rgb, 1.0);',
        '}'].join('\n'),
      depthTest: false, depthWrite: false });
  },

  // the pass of a viewer: get() gives the lens to bend its view with now (LENS.of a calibration), or null.
  // begin(v): the frame then drawn into the lens's picture (the eye's field widened to it); end(v): bent onto the
  // canvas, the eye as it was
  pass: function (get) {
    const P = { get: get, st: null, quad: new THREE.Scene(), cam: new THREE.Camera(), mat: LENS.material() };
    P.quad.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), P.mat));
    P.on = function (v) { return !v.ortho && !!v.r && !!P.get(); };
    // the state for the box as it is (made again when the box, the canvas, the context or the lens change)
    P.prep = function (v) {
      const L = P.get(), r = v.r;
      if (!L || !r || !v.w || !v.h) return null;
      const D = r.getDrawingBufferSize(new THREE.Vector2()), key = L.key + '|' + v.w + 'x' + v.h + '|' + D.x + 'x' + D.y;
      let st = P.st;
      if (!st || st.key !== key || st.r !== r) {
        const fd = LENS.field(L, v.w, v.h, D.x, D.y);
        if (st && st.rt && (st.r !== r || st.rt.width !== fd.RW || st.rt.height !== fd.RH)) { st.rt.dispose(); st.rt = null; }
        let rt = st && st.rt;
        if (!rt) {
          const o = { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, encoding: THREE.sRGBEncoding };
          rt = r.capabilities.isWebGL2 && THREE.WebGLMultisampleRenderTarget
            ? new THREE.WebGLMultisampleRenderTarget(fd.RW, fd.RH, o) : new THREE.WebGLRenderTarget(fd.RW, fd.RH, o);
          if (rt.isWebGLMultisampleRenderTarget) rt.samples = LENS.SAMPLES;
          rt.texture.encoding = THREE.sRGBEncoding;
        }
        st = P.st = { key: key, r: r, L: L, rt: rt, tx: fd.tx, ty: fd.ty, fit: fd.fit, DW: D.x, DH: D.y,
                      fov: 2 * Math.atan(fd.ty) * 180 / Math.PI, aspect: fd.tx / fd.ty };
        const u = P.mat.uniforms;
        u.box.value.set(v.w, v.h); u.fit.value.set(fd.fit.s, fd.fit.ox, fd.fit.oy);
        u.f.value.set(L.fx, L.fy); u.c.value.set(L.cx, L.cy); u.wh.value.set(L.w, L.h);
        u.ka.value.set(L.k[0], L.k[1], L.k[2], L.k[3]); u.kb.value.set(L.k[4], L.k[5], L.k[6], L.k[7]);
        u.t.value.set(fd.tx, fd.ty); u.texel.value.set(1 / fd.RW, 1 / fd.RH);
      }
      return st;
    };
    P.widen = function (v, st) {
      st.fov0 = v.eye.fov; st.aspect0 = v.eye.aspect;
      v.eye.fov = st.fov; v.eye.aspect = st.aspect; v.eye.updateProjectionMatrix();
    };
    P.narrow = function (v, st) { v.eye.fov = st.fov0; v.eye.aspect = st.aspect0; v.eye.updateProjectionMatrix(); };
    P.begin = function (v) {
      if (!P.on(v)) return false;
      const st = P.prep(v);
      if (!st) return false;
      P.widen(v, st);
      v.lensOn = st;
      v.r.setRenderTarget(st.rt);
      return true;
    };
    P.end = function (v) {
      const st = v.lensOn, r = v.r;
      v.lensOn = null;
      P.narrow(v, st);
      r.setRenderTarget(null);
      P.mat.uniforms.src.value = st.rt.texture;
      P.mat.uniforms.probe.value = 0;
      r.render(P.quad, P.cam);
    };
    // a point of the world on the box (css px), through the lens
    P.onBox = function (v, p) {
      const L = P.get(), c = p.clone().applyMatrix4(v.eye.matrixWorldInverse), z = p.clone().project(v.eye).z;
      if (!(c.z < 0)) return { x: NaN, y: NaN, z: 2 };
      const q = LENS.photo(L, c.x / -c.z, c.y / c.z), F = LENS.fit(L, v.w, v.h);
      return { x: F.ox + (q[0] + 0.5) * F.s, y: F.oy + (q[1] + 0.5) * F.s, z: z };
    };
    // the point of the view (normalised, for the eye as it is) a point of the box shows
    P.ndc = function (v, x, y) {
      const L = P.get(), p = LENS.boxToIdeal(L, LENS.fit(L, v.w, v.h), x, y), t = Math.tan(v.eye.fov * Math.PI / 360);
      return new THREE.Vector2(p[0] / (t * v.eye.aspect), -p[1] / t);
    };
    // the bending made ready on a context (its shader compiled with the rest)
    P.warm = function (r) { r.compile(P.quad, P.cam); };
    return P;
  },

  // what the tests read: for points of the world (arrays), where the bent frame shows each, read from the shader
  // itself (the place of the drawing each pixel around the point reads, solved for the point's own place there), and
  // where onBox puts it; css px of the box. With the eye's world matrix, the box and the picture's fit in it.
  // null when the view is not bent (or floats cannot be drawn into)
  measure: function (v, pts) {
    const P = v && v.lens, r = v && v.r;
    if (!P || !P.on(v)) return null;
    const st = P.prep(v), gl = r.getContext();
    if (!st || !r.capabilities.isWebGL2 || !gl.getExtension('EXT_color_buffer_float')) return null;
    const cam = v.eye;
    cam.updateMatrixWorld();
    P.widen(v, st);
    const uvs = pts.map(function (p) {
      const q = new THREE.Vector3().fromArray(p).project(cam);
      return [(q.x + 1) / 2, (q.y + 1) / 2];
    });
    P.narrow(v, st);
    const fwd = pts.map(function (p) { const b = P.onBox(v, new THREE.Vector3().fromArray(p)); return [b.x, b.y]; });
    const rt = new THREE.WebGLRenderTarget(st.DW, st.DH, { type: THREE.FloatType, minFilter: THREE.NearestFilter,
                                                          magFilter: THREE.NearestFilter, depthBuffer: false });
    const u = P.mat.uniforms, px = new Float32Array(16);
    u.probe.value = 1;
    r.setRenderTarget(rt);
    r.render(P.quad, P.cam);
    u.probe.value = 0;
    const shown = fwd.map(function (g, i) {
      let x = g[0], y = g[1];
      for (let it = 0; it < 4; it++) {
        // the pixels around (x, y) (device px, their centres, from the bottom), the places they read
        const X = x * st.DW / v.w - 0.5, Y = (v.h - y) * st.DH / v.h - 0.5, i0 = Math.floor(X), j0 = Math.floor(Y);
        if (!(i0 >= 0 && j0 >= 0 && i0 + 1 < st.DW && j0 + 1 < st.DH)) return null;
        r.readRenderTargetPixels(rt, i0, j0, 2, 2, px);
        const J0 = px[4] - px[0], J1 = px[8] - px[0], J2 = px[5] - px[1], J3 = px[9] - px[1];
        const det = J0 * J3 - J1 * J2, ex = uvs[i][0] - px[0], ey = uvs[i][1] - px[1];
        if (!(Math.abs(det) > 0)) return null;
        const s = (J3 * ex - J1 * ey) / det, t = (J0 * ey - J2 * ex) / det;
        x = (i0 + s + 0.5) * v.w / st.DW; y = v.h - (j0 + t + 0.5) * v.h / st.DH;
        if (s > -0.02 && s < 1.02 && t > -0.02 && t < 1.02) return [x, y];
      }
      return null;
    });
    r.setRenderTarget(null);
    rt.dispose();
    v.draw();
    return { shown: shown, fwd: fwd, eye: cam.matrixWorld.elements.slice(), W: v.w, H: v.h, fit: st.fit,
             drawing: [st.rt.width, st.rt.height], canvas: [st.DW, st.DH] };
  },
};
