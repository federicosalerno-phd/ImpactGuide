// ImpactGuide · a model in 3D, and the 3D-printed plate in it.
//
// VIEW3D is the viewer, for any model of the bench: a canvas in its box, a model
// read from a GLB (GLB.read: its nodes, meshes and plain materials, no loader
// library). It is navigated as Blender's viewport is, with its default keys, so
// that whoever comes from Blender is at home: the middle button turns it as
// Blender's turntable (about the model's vertical, Z, and the view's horizontal,
// with no stop at the poles), Shift + middle moves it, Ctrl + middle zooms, the
// wheel zooms towards the centre of the view (Ctrl and Shift + wheel move it); Alt +
// left stands for the middle button on a touchpad. The keypad: 1 front, 3 right,
// 7 top (Ctrl: the opposite sides), 9 the other side, 5 perspective or
// orthographic, 2 4 6 8 turn by 15 degrees (Ctrl: move), + and - zoom, 0 the
// camera's view (where the page has one), the period frames what is selected;
// Home frames everything. A click of the left button selects a part: it stays as it
// is while the rest of the model fades a little behind a veil, its name in words on
// a small tag tied to the point clicked; the part under the pointer only lightens a
// touch; a click on nothing, Esc or Alt+A let go of it. Top right, Blender's navigation gizmo as Blender
// 5 draws it: the axes with their letters (a click looks along one, a drag turns),
// the lens that zooms and the hand that moves, the camera, perspective or not. At
// the start, and for every new model, the whole model is framed, turning about its
// centre. The view follows the hand with a light damping; it draws only while
// something changes. Its GL context lives only while the box is on screen (made
// again in a few hundredths of a second when it comes back), and one the GPU takes
// away is made again on a new canvas. three.js r128 (ui/libs).
//
// PLATE3D is the plate as it comes off a two-colour printer (white plate, black
// print in relief), ui/models/charuco_plate.glb (tools/build_plate_model.py), in
// the board step when the board is physical. It is made ready while the app is
// idle (the file read, its shaders compiled once), so the step shows it at once.
'use strict';

/** A binary glTF read into three.js: the scene's nodes with their transforms, a mesh per primitive. */
const GLB = {
  MAGIC: 0x46546C67,     // 'glTF'
  TYPES: { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array,
           5126: Float32Array },
  COMPS: { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 },

  // (object, extras of its scene); flat: faces shaded flat (a printed part is all planes), normals then not needed
  read: function (buf, flat) {
    const dv = new DataView(buf);
    if (buf.byteLength < 28 || dv.getUint32(0, true) !== GLB.MAGIC) throw new Error('not a GLB');
    const jl = dv.getUint32(12, true);
    const doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jl)));
    const bin = 28 + jl, bytes = new Uint8Array(buf);
    // an accessor's numbers, copied out of the buffer (a copy is aligned whatever its offset), one after the other
    const data = function (i) {
      const a = doc.accessors[i], v = doc.bufferViews[a.bufferView], Type = GLB.TYPES[a.componentType];
      const n = GLB.COMPS[a.type];
      if (!Type || !n || v.buffer) throw new Error('accessor ' + i);
      const el = n * Type.BYTES_PER_ELEMENT, stride = v.byteStride || el;
      const start = bin + (v.byteOffset || 0) + (a.byteOffset || 0);
      if (stride === el) return new Type(buf.slice(start, start + a.count * el));
      const out = new Type(a.count * n), dst = new Uint8Array(out.buffer);
      for (let k = 0; k < a.count; k++) dst.set(bytes.subarray(start + k * stride, start + k * stride + el), k * el);
      return out;
    };
    const mats = (doc.materials || []).map(function (m) {
      const p = m.pbrMetallicRoughness || {}, c = p.baseColorFactor || [1, 1, 1, 1];
      const spec = ((m.extensions || {}).KHR_materials_specular || {}).specularFactor;
      // a surface that reflects nothing (glTF's specular at 0: the plate's black print): purely diffuse, never turned
      // grey by the lights that go with the eye at a low view (any PBR surface reflects more towards grazing)
      if (spec === 0) {
        return new THREE.MeshPhongMaterial({ name: m.name || '', color: new THREE.Color(c[0], c[1], c[2]),
          opacity: c[3], transparent: c[3] < 1, specular: 0x000000, shininess: 0,
          side: m.doubleSided ? THREE.DoubleSide : THREE.FrontSide, flatShading: !!flat });
      }
      return new THREE.MeshStandardMaterial({ name: m.name || '',
        color: new THREE.Color(c[0], c[1], c[2]), opacity: c[3], transparent: c[3] < 1,
        roughness: p.roughnessFactor === undefined ? 1 : p.roughnessFactor,
        metalness: p.metallicFactor === undefined ? 1 : p.metallicFactor,
        side: m.doubleSided ? THREE.DoubleSide : THREE.FrontSide, flatShading: !!flat });
    });
    const plain = new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.7, metalness: 0,
                                                   flatShading: !!flat });
    // a mesh: its triangles' primitives, each a geometry with its material (shared by every node that uses it)
    const parts = (doc.meshes || []).map(function (m) {
      return m.primitives.filter(function (pr) { return pr.mode === undefined || pr.mode === 4; }).map(function (pr) {
        const g = new THREE.BufferGeometry(), at = pr.attributes;
        g.setAttribute('position', new THREE.BufferAttribute(data(at.POSITION), 3));
        if (pr.indices !== undefined) g.setIndex(new THREE.BufferAttribute(data(pr.indices), 1));
        if (!flat && at.NORMAL !== undefined) g.setAttribute('normal', new THREE.BufferAttribute(data(at.NORMAL), 3));
        else if (!flat) g.computeVertexNormals();
        return [g, pr.material === undefined ? plain : mats[pr.material]];
      });
    });
    const node = function (i) {
      const d = doc.nodes[i], o = new THREE.Group();
      o.name = d.name || '';
      if (d.matrix) new THREE.Matrix4().fromArray(d.matrix).decompose(o.position, o.quaternion, o.scale);
      if (d.translation) o.position.fromArray(d.translation);
      if (d.rotation) o.quaternion.fromArray(d.rotation);
      if (d.scale) o.scale.fromArray(d.scale);
      if (d.mesh !== undefined) parts[d.mesh].forEach(function (p) { o.add(new THREE.Mesh(p[0], p[1])); });
      (d.children || []).forEach(function (c) { o.add(node(c)); });
      return o;
    };
    const sc = (doc.scenes || [])[doc.scene || 0] || { nodes: (doc.nodes || []).map(function (n, i) { return i; }) };
    const group = new THREE.Group();
    sc.nodes.forEach(function (i) { group.add(node(i)); });
    return { object: group, extras: sc.extras || {} };
  },

  // read from a file of the page; done(result) or failed(error)
  load: function (url, flat, done, failed) {
    const x = new XMLHttpRequest();
    x.open('GET', url, true);
    x.responseType = 'arraybuffer';
    x.onload = function () {
      let res = null;
      try { res = GLB.read(x.response, flat); } catch (e) { if (failed) failed(e); return; }
      done(res);
    };
    x.onerror = function () { if (failed) failed(new Error(url)); };
    x.send();
  },
};

/** The viewer: VIEW3D.create(box, canvas, options) gives one (null without three.js). */
const VIEW3D = {
  TURN_TAU: 0.035,      // s, how closely the view follows the hand
  ZOOM_TAU: 0.07,       // s, and the zoom
  FOV: 30,              // degrees, vertical, of the perspective view
  ORBIT: 1.3 * Math.PI, // radians turned across the narrower side of the box
  DRAG_ZOOM: 0.006,     // zoom per pixel dragged (Ctrl + middle, the lens): 100 px up comes 1.8 times closer
  WHEEL: 0.0011,        // zoom per pixel of wheel (a notch of 100: 12 %)
  WHEEL_PAN: 0.5,       // pixels moved per pixel of wheel, with Ctrl (across) or Shift (up and down)
  STEP_DEG: 15,         // the keypad's 2 4 6 8 turn by this much...
  PAN_STEP: 0.1,        // ... and with Ctrl move by this share of the view
  KEY_ZOOM: 1.2,        // the keypad's + and -
  CLICK_PX: 4,          // a press that moves less than this is a click: it selects
  FIT_POINTS: 40000,    // points of the model a view that frames it looks at
  HL_IN_S: 0.18,        // s, a highlight coming in, and going
  HL_OUT_S: 0.18,
  HL_VEIL: 0.32,        // what is not selected behind a veil of the view's colour this thick
  HL_HOVER_LIGHT: 0.08, // the part under the pointer this much lighter
  HL_HOVER: 1,          // (the pointer's highlight comes in as far as a selection's)
  TAG_GAP: 26,          // px between a part and its name's tag
  HOVER_MS: 50,         // the part under the pointer looked for at most this often
  CHUNK: 256,           // triangles a click looks into by one box  SLEEP_MS: 3000,       // the box off screen this long: its GL context is given back
  // the navigation gizmo top right, as Blender 5 draws it (view3d_gizmo_navigate.cc, view3d_gizmo_navigate_type.cc,
  // button2d_gizmo.cc and the default theme): sizes in px of an interface at 100 % (the page's CSS px are Blender's
  // pixels times its scale), the widths of its lines in Blender's own whole pixels (VIEW3D.pixel)
  AXES: [['X', '#ff3352'], ['Y', '#8bdc00'], ['Z', '#2890ff']],   // the theme's axis colours
  GIZMO: 80,            // across (U.gizmo_size_navigate_v3d)
  GIZMO_OFFSET: 10,     // from the box's top and right edges (GIZMO_OFFSET)
  HANDLE: 0.2,          // an axis's circle: its radius, of the gizmo's (AXIS_HANDLE_SIZE); its centre at 1 - this
  DEPTH_BIAS: 0.01,     // an axis this little towards the eye counts as behind (AXIS_DEPTH_BIAS)
  MINI: 28,             // the buttons below: MINI across, MINI + MINI_GAP apart (GIZMO_MINI_SIZE,
  MINI_GAP: 2,          // GIZMO_MINI_OFFSET), on a backdrop as wide; their icons ICON (ICON_DEFAULT_WIDTH)
  ICON: 16,
  HOVER: 'rgba(128,128,128,0.5)',   // the disc behind the gizmo while the pointer is on it (its color_hi)
  BACKDROP: 'rgba(48,48,48,0.7)',   // the buttons' backdrop: the 3D view's header colour (#303030b3)
  ICON_COLOR: '#e6e6e6',            // the 3D view's text: an icon at 60 %; under the pointer white, whole
  // the letters in Blender's own font (Inter, bold, at the gizmo's 10 px), each outline centred on (0, 0) by its
  // own box; box: left, bottom, right, top (px from its origin on the baseline, y up); adv: how far it moves the pen
  GLYPHS: {
    X: { box: [0.249, 0, 7.134, 7.275], adv: 7.383,
         d: 'M-3.442 3.638-0.44-0.635V0.496L-3.21-3.638H-1.479L-0.682-2.416Q-0.459-2.072-0.315-1.794Q-0.171-1.516'
          + '-0.052-1.261Q0.067-1.006 0.213-0.731H-0.183Q-0.032-1.004 0.089-1.259Q0.209-1.514 0.356-1.793Q0.504'
          + '-2.072 0.727-2.416L1.538-3.638H3.226L0.506 0.415V-0.662L3.442 3.638H1.677L0.663 2.114Q0.464 1.806 0.339'
          + ' 1.591Q0.215 1.376 0.115 1.184Q0.015 0.992-0.118 0.752H0.117Q-0.011 0.985-0.113 1.179Q-0.215 1.372'
          + '-0.341 1.589Q-0.467 1.806-0.676 2.114L-1.718 3.638Z' },
    Y: { box: [0.238, 0, 7.071, 7.275], adv: 7.31,
         d: 'M-0.722 3.638V0.902L-3.416-3.638H-1.67L-0.44-1.371Q-0.252-1.028-0.105-0.693Q0.043-0.358 0.198 0.11H'
          + '-0.128Q0.022-0.362 0.165-0.7Q0.308-1.037 0.487-1.371L1.677-3.638H3.416L0.764 0.902V3.638Z' },
    Z: { box: [0.548, 0, 6.094, 7.275], adv: 6.641,
         d: 'M-2.76 3.638V2.746L0.191-1.423Q0.464-1.803 0.79-2.185Q1.117-2.568 1.451-2.945L1.564-2.463Q1.075-2.418'
          + ' 0.584-2.409Q0.093-2.401-0.396-2.401H-2.773V-3.638H2.76V-2.742L-0.137 1.352Q-0.424 1.751-0.764 2.151'
          + 'Q-1.105 2.55-1.451 2.945L-1.564 2.463Q-1.051 2.418-0.541 2.409Q-0.031 2.401 0.479 2.401H2.773V3.638Z' },
    '-': { box: [1.57, 2.408, 4.88, 3.577], adv: 6.465, d: 'M1.655-0.585V0.585H-1.655V-0.585Z' },
  },
  // the buttons' icons, 16 px: the lens that zooms, the hand that moves, the camera, the perspective (a grid going
  // away) and the orthographic view (a flat grid)
  ICONS: {
    zoom: '<path d="M1.4 6.5a5.1 5.1 0 1 0 10.2 0a5.1 5.1 0 1 0-10.2 0ZM5.8 3.9v1.9H3.9v1.4h1.9v1.9h1.4V7.2'
      + 'h1.9V5.8H7.2V3.9Z" fill-rule="evenodd" stroke="none"/><path d="M10.3 10.3l3.9 3.9" fill="none" '
      + 'stroke-width="2.4" stroke-linecap="round"/>',
    pan: '<g stroke="none"><rect x="4.3" y="3.7" width="2" height="6.3" rx="1"/><rect x="6.5" y="2.2" width="2" '
      + 'height="7.6" rx="1"/><rect x="8.7" y="2.8" width="2" height="7" rx="1"/><rect x="10.9" y="4.6" '
      + 'width="1.9" height="5.6" rx="0.95"/><path d="M4.3 8.2h8.5V10c0 2.6-1.9 4.6-4.4 4.6h-.5c-1.5 0-2.5-.6'
      + '-3.3-1.7L2.3 9.9a1 1 0 0 1 1.5-1.3l.5.6z"/></g>',
    camera: '<circle cx="4.6" cy="4.4" r="1.9" fill="none" stroke-width="1.2"/><circle cx="9" cy="4.4" r="1.9" '
      + 'fill="none" stroke-width="1.2"/><rect x="1.8" y="7.2" width="9.4" height="6" rx="1" fill="none" '
      + 'stroke-width="1.3"/><path d="M11.2 9.6l3.4-1.9v5.1l-3.4-1.9z" fill="none" stroke-width="1.2" '
      + 'stroke-linejoin="round"/>',
    persp: '<path d="M4.6 3.2h6.8l3.4 9.6H1.2zM3.5 6.4h9M2.3 9.6h11.4M6.9 3.2l-1.8 9.6M9.1 3.2l1.8 9.6" '
      + 'fill="none" stroke-width="1.2" stroke-linejoin="round"/>',
    ortho: '<path d="M2.5 2.5h11v11h-11zM2.5 6.2h11M2.5 9.8h11M6.2 2.5v11M9.8 2.5v11" fill="none" '
      + 'stroke-width="1.2"/>',
  },
  gizmos: 0,            // gizmos made (the ids of their gradients)
  all: [],              // the viewers made: the keys go to the one under the pointer, or the last one used

  // options: margin (room around the whole model), dir (where the eye sits in the first view, seen from the model's
  // centre), up (the model's vertical, Blender's Z) and front (the way the front view looks, Blender's +Y; by
  // default +Y when up is Z, else -Z as glTF has it), background (a colour role of the page), failed (no GL
  // context can be had), camera() (the keypad's 0: the view of the camera), changed() (the view turned by hand)
  create: function (box, canvas, o) {
    if (!window.THREE) return null;
    const v = Object.create(VIEW3D.proto);
    v.box = box; v.canvas = canvas; v.r = null; v.lost = false; v.w = v.h = 0;
    v.o = Object.assign({ margin: 1.05, dir: [0, -0.8, 0.6], up: [0, 0, 1], background: '--surface-2' }, o || {});
    v.U = new THREE.Vector3().fromArray(v.o.up).normalize();
    const f = new THREE.Vector3().fromArray(v.o.front || (Math.abs(v.U.z) > 0.9 ? [0, 1, 0] : [0, 0, -1]));
    v.F = f.addScaledVector(v.U, -f.dot(v.U)).normalize();
    v.R = new THREE.Vector3().crossVectors(v.F, v.U);
    v.scene = new THREE.Scene();
    v.cam = new THREE.PerspectiveCamera(VIEW3D.FOV, 1, 0.01, 10);
    v.oc = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10);
    v.ortho = false;
    v.eye = v.cam;                  // the camera drawn with: the perspective one, or the orthographic one
    v.scene.add(v.cam, v.oc);
    // the lights go with the eye: a soft sky above and ground below the view, a key light from above right,
    // a fill from the left and a rim from behind
    const white = new THREE.Color(1, 1, 1);
    v.hemi = new THREE.HemisphereLight(white, new THREE.Color(0.3, 0.29, 0.27), 0.7);
    v.scene.add(v.hemi);
    [[0.95, 0.45, 0.75, 0.6], [0.35, -0.8, 0.25, 0.5], [0.3, -0.2, 0.6, -0.8]].forEach(function (l) {
      const d = new THREE.DirectionalLight(white, l[0]);
      d.position.set(l[1], l[2], l[3]);
      d.target = v.cam;
      v.cam.add(d);
    });
    v.root = new THREE.Group();
    v.scene.add(v.root);
    v.radius = 0.1; v.centre = new THREE.Vector3(); v.bounds = null;
    v.goal = { q: new THREE.Quaternion(), dist: 0.5, at: new THREE.Vector3() };
    v.shown = { q: new THREE.Quaternion(), dist: 0.5, at: new THREE.Vector3() };
    v.moved = false;      // moved by hand: a new size of its box keeps the view
    v.sel = null;         // what is selected: {name, parts (the outline's meshes)}
    v.raf = 0; v.last = 0; v.sleeper = 0; v.hot = false; v.used = 0;
    v.overlay();
    v.background();
    v.frameAll(true);
    v.wire();
    v.observer = new ResizeObserver(function () {
      // off screen (its box hidden): the context given back a moment later; on screen again: drawn
      clearTimeout(v.sleeper);
      if (!box.clientWidth || !box.clientHeight) v.sleeper = setTimeout(function () { v.sleep(); }, VIEW3D.SLEEP_MS);
      else v.draw();
    });
    v.observer.observe(box);
    VIEW3D.all.push(v);
    VIEW3D.keys();
    return v;
  },

  // a GL context on a canvas, with what this page always sets; null without WebGL
  context: function (canvas) {
    let r = null;
    try { r = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false }); } catch (e) { return null; }
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.outputEncoding = THREE.sRGBEncoding;
    return r;
  },

  // the keys, once for every viewer, Blender's: the keypad's views, Home, Alt+A. They go to the viewer under the
  // pointer, else to the one used last, while it is on screen and no field of the page has the keyboard
  keys: function () {
    if (VIEW3D.keysOn) return;
    VIEW3D.keysOn = true;
    window.addEventListener('keydown', function (e) {
      const a = document.activeElement;
      if (a && (/^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName) || a.isContentEditable)) return;
      const shown = VIEW3D.all.filter(function (v) { return !v.gone && v.box.clientWidth && v.box.clientHeight; });
      const v = shown.find(function (x) { return x.hot; }) || shown.sort(function (p, q) { return q.used - p.used; })[0];
      if (v && v.key(e)) { e.preventDefault(); e.stopPropagation(); }
    }, true);
  },

  // a colour mixed with white (k: the share of white), as #rrggbb
  tint: function (hex, k) {
    const c = new THREE.Color(hex), w = 1 - k;
    return '#' + [c.r, c.g, c.b].map(function (x) {
      return ('0' + Math.round((x * w + k) * 255).toString(16)).slice(-2);
    }).join('');
  },

  // what the highlight draws with, made once: the black of everything (for the depth) and a colour for each of
  // three highlighted parts (red, green, blue: a channel each) in the mask, a little nearer than their own surface;
  // and the pass that lays it over the picture: no outline, the part selected left as it is while everything else
  // fades a little behind a veil of the view's own colour (less colour, a little see-through), the part under the
  // pointer only a touch lighter. The mask is softened over a pixel at its edges
  kitMade: null,
  kit: function () {
    if (VIEW3D.kitMade) return VIEW3D.kitMade;
    const paint = function (c) {
      return new THREE.MeshBasicMaterial({ color: c, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false,
                                           polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2,
                                           side: THREE.DoubleSide });
    };
    const paints = [paint(0xff0000), paint(0x00ff00), paint(0x0000ff)];
    const m = new THREE.ShaderMaterial({
      uniforms: { mask: { value: null }, px: { value: new THREE.Vector2() }, amt: { value: new THREE.Vector3() },
                  sel: { value: new THREE.Vector3() }, veil: { value: new THREE.Color() } },
      vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      // amt: how far each channel's highlight has come in; sel: 1 for a selection's channel, 0 for the pointer's.
      // Everything but the selected part veiled (a part let go and one taken cross over); the pointer's part
      // lighter
      fragmentShader: 'uniform sampler2D mask; uniform vec2 px; uniform vec3 amt; uniform vec3 sel; uniform vec3 veil;'
        + ' varying vec2 vUv;'
        + 'void main() { vec3 b = texture2D(mask, vUv).rgb * 0.25'
        + '  + (texture2D(mask, vUv + vec2(px.x, 0.0)).rgb + texture2D(mask, vUv - vec2(px.x, 0.0)).rgb'
        + '   + texture2D(mask, vUv + vec2(0.0, px.y)).rgb + texture2D(mask, vUv - vec2(0.0, px.y)).rgb) * 0.125'
        + '  + (texture2D(mask, vUv + px).rgb + texture2D(mask, vUv - px).rgb'
        + '   + texture2D(mask, vUv + vec2(px.x, -px.y)).rgb + texture2D(mask, vUv - vec2(px.x, -px.y)).rgb) * 0.0625;'
        + ' vec3 out_ = sel * amt * (1.0 - b), lit = (1.0 - sel) * amt * b;'
        + ' float v = max(max(out_.x, out_.y), out_.z) * ' + VIEW3D.HL_VEIL.toFixed(3) + ';'
        + ' float h = max(max(lit.x, lit.y), lit.z) * ' + VIEW3D.HL_HOVER_LIGHT.toFixed(3) + ';'
        + ' float a = v + h; if (a < 0.002) discard;'
        + ' gl_FragColor = vec4((veil * v + vec3(h)) / a, min(a, 1.0)); }',
      transparent: true, depthTest: false, depthWrite: false });
    const s = new THREE.Scene();
    s.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), m));
    VIEW3D.kitMade = { black: new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.DoubleSide }), paints: paints,
                       paint: paints[0], quad: { scene: s, cam: new THREE.Camera(), material: m } };
    return VIEW3D.kitMade;
  },

  // the page's accent colour (looked up again every second: the theme can change)
  accent: function () {
    const now = performance.now();
    if (!VIEW3D.accentNow || now - VIEW3D.accentAt > 1000) {
      VIEW3D.accentAt = now;
      VIEW3D.accentNow = new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()
                                         || '#d6a531');
    }
    return VIEW3D.accentNow;
  },
  ease: function (t) { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); },

  // the boxes a click looks into, made once for a mesh of several parts: each part's, and its triangles' CHUNK at
  // a time (in the mesh's own coordinates)
  chunks: function (m) {
    if (m.userData.chunks) return m.userData.chunks;
    const g = m.geometry, pos = g.attributes.position, idx = g.index, p = new THREE.Vector3();
    m.userData.chunks = { parts: m.userData.objects.map(function (o) {
      const part = { box: new THREE.Box3(), chunks: [] };
      for (let f = o[1]; f < o[1] + o[2]; f += VIEW3D.CHUNK) {
        const n = Math.min(VIEW3D.CHUNK, o[1] + o[2] - f), box = new THREE.Box3();
        for (let k = f * 3; k < (f + n) * 3; k++) box.expandByPoint(p.fromBufferAttribute(pos, idx.getX(k)));
        part.box.union(box);
        part.chunks.push({ first: f, count: n, box: box });
      }
      return part;
    }) };
    return m.userData.chunks;
  },

  // ... made for every mesh of a model while the page is idle, a few at a time, so that the first click or the
  // first pointer over it does not wait
  chunkAll: function (root) {
    const list = [];
    root.traverse(function (m) { if (m.isMesh && m.userData.objects && !m.userData.chunks) list.push(m); });
    const go = function (dl) {
      while (list.length) {
        VIEW3D.chunks(list.shift());
        if (!dl || dl.timeRemaining() <= 4) break;
      }
      if (list.length) (window.requestIdleCallback || function (f) { return setTimeout(f, 30); })(go, { timeout: 2000 });
    };
    (window.requestIdleCallback || function (f) { return setTimeout(f, 30); })(go, { timeout: 2000 });
  },
};

VIEW3D.proto = {
  // a context to draw with: the one there is, or one made (again, on a new canvas, when the GPU took it away and did
  // not give it back); false when none can be had
  ready: function () {
    const v = this;
    if (v.r && !v.lost && !v.r.getContext().isContextLost()) return true;
    if (v.r) {
      try { v.r.dispose(); } catch (e) { /* its context is gone already */ }
      v.r = null;
      v.maskRT = null;
      const fresh = v.canvas.cloneNode(false);
      v.canvas.replaceWith(fresh);
      v.canvas = fresh;
      v.wire();
    }
    const r = VIEW3D.context(v.canvas);
    if (!r) return false;
    const c = v.canvas;
    c.addEventListener('webglcontextlost', function () { if (c === v.canvas) v.lost = true; });
    c.addEventListener('webglcontextrestored', function () { if (c === v.canvas) { v.lost = false; v.draw(); } });
    v.r = r; v.lost = false; v.w = v.h = 0;
    v.background();
    return true;
  },

  // the context given back while the box is off screen (the scene stays, ready to be drawn again)
  sleep: function () {
    const v = this;
    if (!v.r || (v.box.clientWidth && v.box.clientHeight)) return;
    if (v.raf) { cancelAnimationFrame(v.raf); v.raf = 0; }
    try { v.r.dispose(); v.r.forceContextLoss(); } catch (e) { /* gone already */ }
    v.r = null;
    v.maskRT = null;
    // a canvas whose context was given back cannot have another: a new one in its place
    const fresh = v.canvas.cloneNode(false);
    v.canvas.replaceWith(fresh);
    v.canvas = fresh;
    v.wire();
  },

  // the shaders compiled once, on a context of their own given back at once: the GPU keeps them, so the first
  // view does not wait for them
  warm: function () {
    const r = VIEW3D.context(document.createElement('canvas'));
    if (!r) return;
    try { r.compile(this.scene, this.eye); } finally { r.dispose(); r.forceContextLoss(); }
  },

  // the colour behind the model (a colour role of the page); the gizmo's colours are mixed with it
  background: function () {
    const c = getComputedStyle(document.documentElement).getPropertyValue(this.o.background).trim() || '#808080';
    this.viewColor = new THREE.Color(c);
    if (this.r) this.r.setClearColor(this.viewColor, 1);
  },

  // a model in place of the one shown: framed whole, turning about its centre
  set: function (obj) {
    const v = this;
    v.hl = null; v.sel = null;
    v.tag.style.visibility = 'hidden'; v.tag.style.opacity = 0; v.lead.style.opacity = 0;
    v.root.children.slice().forEach(function (c) { v.root.remove(c); VIEW3D.free(c); });
    v.root.add(obj);
    VIEW3D.chunkAll(obj);
    const box = new THREE.Box3().setFromObject(obj);
    if (!box.isEmpty()) {
      v.bounds = box;
      box.getCenter(v.centre);
      v.radius = Math.max(1e-6, box.getSize(new THREE.Vector3()).length() / 2);
    }
    v.size();
    v.moved = false;
    v.frameAll(true);
    v.draw();
  },

  // ── where the eye is ───────────────────────────────────────────────────────────────────────────────────────
  // the half height of the view at the point turned about (the orthographic view's own size)
  halfH: function (dist) { return dist * Math.tan(this.cam.fov / 2 * Math.PI / 180); },

  // an orientation looking along f with u up on the screen
  look: function (f, u) {
    const back = f.clone().normalize().negate(), right = new THREE.Vector3().crossVectors(u, back).normalize();
    const up = new THREE.Vector3().crossVectors(back, right);
    return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, back));
  },

  // the eye along q at the distance that holds the box whole (each of its corners, not only its sphere), the
  // point turned about at its centre; the viewer's own field of view again
  // a sample of the points of what is drawn (at most FIT_POINTS), in the world: all the visible meshes under the
  // root, or the meshes given
  points: function (meshes) {
    const f = this.pointsFlat(meshes), out = [];
    for (let i = 0; i < f.length; i += 3) out.push([f[i], f[i + 1], f[i + 2]]);
    return out;
  },

  // ... packed (x y z one after the other); the whole model's kept while what is shown, and where, stays the same
  pointsFlat: function (meshes) {
    const v = this, list = [];
    const visible = function (o) { for (let p = o; p; p = p.parent) { if (!p.visible || p.userData.noPick) return false; } return true; };
    (meshes || []).forEach(function (m) { list.push(m); });
    if (!meshes) v_root_meshes(this.root, list, visible);
    list.forEach(function (m) { m.updateWorldMatrix(true, false); });
    let key = '';
    if (!meshes) {
      key = list.map(function (m) {
        const e = m.matrixWorld.elements;
        return m.id + ':' + m.geometry.attributes.position.version + ':' + e[0].toFixed(5) + ',' + e[1].toFixed(5) + ','
          + e[4].toFixed(5) + ',' + e[12].toFixed(3) + ',' + e[13].toFixed(3) + ',' + e[14].toFixed(3);
      }).join('|');
      if (v.ptsKey === key && v.pts) return v.pts;
    }
    // a part of a mesh (the outline of a selection) counts by the vertices of its own triangles
    const ranged = function (g) { return g.index && isFinite(g.drawRange.count); };
    let n = 0;
    list.forEach(function (m) { const g = m.geometry; n += ranged(g) ? g.drawRange.count : g.attributes.position.count; });
    const step = Math.max(1, Math.ceil(n / VIEW3D.FIT_POINTS));
    let total = 0;
    list.forEach(function (m) {
      const g = m.geometry, c = ranged(g) ? g.drawRange.count : g.attributes.position.count;
      total += Math.ceil(c / step);
    });
    const out = new Float32Array(3 * total);
    let o = 0;
    list.forEach(function (m) {
      const g = m.geometry, pos = g.attributes.position, e = m.matrixWorld.elements;
      // the numbers themselves (interleaved or not), not one call per number
      const arr = pos.isInterleavedBufferAttribute ? pos.data.array : pos.array;
      const stride = pos.isInterleavedBufferAttribute ? pos.data.stride : pos.itemSize, off = pos.offset || 0;
      const add = function (i) {
        const j = i * stride + off, x = arr[j], y = arr[j + 1], z = arr[j + 2];
        out[o++] = e[0] * x + e[4] * y + e[8] * z + e[12];
        out[o++] = e[1] * x + e[5] * y + e[9] * z + e[13];
        out[o++] = e[2] * x + e[6] * y + e[10] * z + e[14];
      };
      if (ranged(g)) {
        const idx = g.index.array;
        for (let k = g.drawRange.start; k < g.drawRange.start + g.drawRange.count; k += step) add(idx[k]);
      } else {
        for (let i = 0; i < pos.count; i += step) add(i);
      }
    });
    const res = o === out.length ? out : out.subarray(0, o);
    if (!meshes) { v.ptsKey = key; v.pts = res; }
    return res;
  },

  // the eye along q holding pts (a sample of the points of the model) whole, centred and filling the view within
  // the margin; the point turned about where the view's centre meets the depth of the box's centre (a few rounds
  // of moving and zooming, each from where the points fall on the screen)
  fitPoints: function (q, pts, c, d) {
    const v = this, right = new THREE.Vector3(1, 0, 0).applyQuaternion(q), up = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
    const back = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    const tH = Math.tan(v.cam.fov / 2 * Math.PI / 180), tW = tH * (v.cam.aspect || 1);
    const at = c.clone();
    // packed (pointsFlat) or as [x, y, z]: each point along the view's axes once, then only moved
    let f = pts;
    if (!ArrayBuffer.isView(f)) { f = new Float32Array(3 * pts.length); pts.forEach(function (p, i) { f.set(p, 3 * i); }); }
    const n = f.length / 3, X = new Float64Array(n), Y = new Float64Array(n), Z = new Float64Array(n);
    const rx = right.x, ry = right.y, rz = right.z, ux = up.x, uy = up.y, uz = up.z, bx = back.x, by = back.y, bz = back.z;
    for (let i = 0, j = 0; i < n; i++, j += 3) {
      const x = f[j], y = f[j + 1], z = f[j + 2];
      X[i] = x * rx + y * ry + z * rz; Y[i] = x * ux + y * uy + z * uz; Z[i] = x * bx + y * by + z * bz;
    }
    const ortho = v.ortho;
    for (let round = 0; round < 6; round++) {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      const aR = at.dot(right), aU = at.dot(up), aB = at.dot(back), lo = d * 0.05;
      for (let i = 0; i < n; i++) {
        let depth = d;
        if (!ortho) { depth = d - (Z[i] - aB); if (depth < lo) depth = lo; }
        const nx = (X[i] - aR) / (depth * tW), ny = (Y[i] - aU) / (depth * tH);
        if (nx < x0) x0 = nx; if (nx > x1) x1 = nx; if (ny < y0) y0 = ny; if (ny > y1) y1 = ny;
      }
      if (!isFinite(x0)) break;
      at.addScaledVector(right, (x0 + x1) / 2 * d * tW).addScaledVector(up, (y0 + y1) / 2 * d * tH);
      d *= Math.max((x1 - x0) / 2, (y1 - y0) / 2) * v.o.margin;
    }
    return { at: at, d: d };
  },

  fit: function (q, box, now, meshes) {
    const v = this, g = v.goal;
    v.cam.fov = v.o.fov || VIEW3D.FOV;
    v.cam.updateProjectionMatrix();
    const c = box && !box.isEmpty() ? box.getCenter(new THREE.Vector3()) : v.centre.clone();
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(q), up = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
    const back = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    const tH = Math.tan(v.cam.fov / 2 * Math.PI / 180), tW = tH * (v.cam.aspect || 1);
    let d = 0, half = 0;
    const corners = [];
    if (box && !box.isEmpty()) {
      [box.min.x, box.max.x].forEach(function (x) { [box.min.y, box.max.y].forEach(function (y) {
        [box.min.z, box.max.z].forEach(function (z) { corners.push(new THREE.Vector3(x, y, z)); }); }); });
    } else {
      corners.push(c.clone().addScalar(v.radius), c.clone().addScalar(-v.radius));
    }
    corners.forEach(function (p) {
      const rel = p.clone().sub(c), x = Math.abs(rel.dot(right)), y = Math.abs(rel.dot(up)), z = rel.dot(back);
      d = Math.max(d, z + x / tW, z + y / tH);
      half = Math.max(half, y, x / (v.cam.aspect || 1));
    });
    d = Math.max(d * v.o.margin, v.radius * 0.05);
    if (v.ortho) d = Math.max(d, half * v.o.margin / tH);
    let at = c;
    const pts = v.pointsFlat(meshes);
    if (pts.length) {
      const f = v.fitPoints(q, pts, c, d);
      at = f.at; d = Math.max(f.d, v.radius * 0.02);
    }
    g.q.copy(q); g.dist = d; g.at.copy(at);
    if (now) { v.shown.q.copy(g.q); v.shown.dist = g.dist; v.shown.at.copy(g.at); }
    v.draw();
  },

  // the whole model, from the way it was first seen (Home, a new model, a new size before any turn)
  frameAll: function (now) {
    const v = this, dir = new THREE.Vector3().fromArray(v.o.dir).normalize();
    v.fit(v.moved ? v.goal.q.clone() : v.look(dir.negate(), v.U), v.bounds, now);
  },

  // what is selected framed (the keypad's period), else the whole model
  frameSelected: function () {
    const v = this;
    if (!v.sel) { v.fit(v.goal.q.clone(), v.bounds); return; }
    v.fit(v.goal.q.clone(), v.sel.box, false, v.sel.parts);
  },

  // a view along one of the model's axes: front, back, right, left, top, bottom (the keypad's 1 3 7, with Ctrl)
  axisView: function (name) {
    const v = this, F = v.F, U = v.U, R = v.R;
    const views = { front: [F, U], back: [F.clone().negate(), U], right: [R.clone().negate(), U], left: [R, U],
                    top: [U.clone().negate(), F], bottom: [U, F.clone().negate()] };
    const w = views[name];
    v.goal.q.copy(v.look(w[0], w[1]));
    v.user();
  },

  // the view seen from the other side (the keypad's 9)
  opposite: function () {
    this.goal.q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI));
    this.user();
  },

  // perspective or orthographic (the keypad's 5); with nothing in front of the eye moved
  setOrtho: function (on) {
    const v = this;
    if (!!on === v.ortho) return;
    v.ortho = !!on;
    const from = v.ortho ? v.cam : v.oc, to = v.ortho ? v.oc : v.cam;
    from.children.slice().forEach(function (l) { to.add(l); if (l.isDirectionalLight) l.target = to; });
    v.eye = to;
    v.draw();
  },

  // turned as Blender's turntable turns it: across about the model's vertical, up and down about the view's own
  // horizontal, with no angle where it stops; right and down in radians (the model follows the hand)
  turn: function (right, down) {
    if (Math.abs(right) + Math.abs(down) < 1e-9) return;
    const g = this.goal;
    g.q.premultiply(new THREE.Quaternion().setFromAxisAngle(this.U, -right));
    g.q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -down));
    this.user();
  },

  // moved by pixels of the box, at the depth of the point turned about
  pan: function (dx, dy) {
    const g = this.goal, per = 2 * this.halfH(g.dist) / (this.h || 1);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(g.q), up = new THREE.Vector3(0, 1, 0).applyQuaternion(g.q);
    g.at.addScaledVector(right, -dx * per).addScaledVector(up, dy * per);
    this.user();
  },

  // closer (f < 1) or further, towards the centre of the view
  zoom: function (f) {
    const g = this.goal, r = this.radius;
    g.dist = Math.max(r * 0.01, Math.min(r * 20, g.dist * f));
    this.user();
  },

  user: function () {
    this.moved = true;
    this.used = performance.now();
    if (this.o.changed) this.o.changed();
    this.draw();
  },

  // the eye where a state puts it, the sky above it; near and far just around the model, as tight as it allows:
  // from the eye to the farthest corner of its box, and from the nearest one (or, with the eye inside the box, from
  // 1.5 % of its size but never beyond a third of the way to the point turned about): depth enough that surfaces
  // a tenth of a millimetre apart never flicker as one zooms
  place: function (s) {
    const v = this, c = v.cam, o = v.oc, r = v.radius;
    const pos = new THREE.Vector3(0, 0, s.dist).applyQuaternion(s.q).add(s.at);
    [c, o].forEach(function (k) { k.quaternion.copy(s.q); k.position.copy(pos); });
    v.hemi.position.set(0, 1, 0.3).applyQuaternion(s.q);
    const d = pos.distanceTo(v.centre);
    let near = Math.max(r * 0.002, d - r * 1.5), far = d + r * 1.5;
    const b = v.bounds;
    if (b && !b.isEmpty()) {
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(s.q);
      let z0 = Infinity, z1 = -Infinity;
      [b.min.x, b.max.x].forEach(function (x) { [b.min.y, b.max.y].forEach(function (y) { [b.min.z, b.max.z].forEach(function (z) {
        const k = (x - pos.x) * fwd.x + (y - pos.y) * fwd.y + (z - pos.z) * fwd.z;
        z0 = Math.min(z0, k); z1 = Math.max(z1, k); }); }); });
      far = Math.max(z1 * 1.02, r * 0.05);
      near = b.containsPoint(pos) || z0 <= 0 ? Math.min(r * 0.015, s.dist * 0.3) : Math.max(r * 0.001, z0 * 0.98);
      near = Math.max(near, far * 1e-5);
    }
    c.near = near;
    c.far = far;
    c.updateProjectionMatrix();
    const hh = v.halfH(s.dist), hw = hh * (c.aspect || 1);
    o.left = -hw; o.right = hw; o.top = hh; o.bottom = -hh;
    o.near = d - r * 2; o.far = d + r * 2;
    o.updateProjectionMatrix();
  },

  // the box's size (false while it is off screen); the drawing and the field of view follow it
  size: function () {
    const w = this.box.clientWidth, h = this.box.clientHeight;
    if (!w || !h) return false;
    if (w !== this.w || h !== this.h) {
      this.w = w; this.h = h;
      if (this.r) this.r.setSize(w, h, false);
      this.cam.aspect = w / h;
      this.cam.updateProjectionMatrix();
      if (!this.moved) this.frameAll(true);
    }
    return true;
  },

  draw: function () {
    const v = this;
    if (v.raf || v.gone) return;
    v.raf = requestAnimationFrame(function (t) { v.raf = 0; v.frame(t); });
  },

  // one frame: the shown view a step closer to the goal, drawn; another frame while they differ
  frame: function (t) {
    const v = this, s = v.shown, g = v.goal;
    if (!v.box.clientWidth || !v.box.clientHeight) { v.last = 0; return; }
    if (!v.ready()) {
      v.last = 0;
      if (v.o.failed) v.o.failed();
      return;
    }
    v.size();
    const dt = v.last ? Math.min(0.05, Math.max(0, (t - v.last) / 1000)) : 1 / 60;
    const kt = 1 - Math.exp(-dt / VIEW3D.TURN_TAU), kz = 1 - Math.exp(-dt / VIEW3D.ZOOM_TAU);
    s.q.slerp(g.q, kt);
    s.at.lerp(g.at, kt);
    s.dist = Math.exp(Math.log(s.dist) + (Math.log(g.dist) - Math.log(s.dist)) * kz);
    const still = s.q.angleTo(g.q) < 2e-4 && Math.abs(Math.log(s.dist / g.dist)) < 2e-4
      && s.at.distanceTo(g.at) < v.radius * 2e-4;
    if (still) { s.q.copy(g.q); s.dist = g.dist; s.at.copy(g.at); }
    v.place(s);
    // a lens on the view (18-lens.js): the frame drawn into its picture, then bent onto the canvas
    const lens = v.lens && v.lens.begin(v);
    v.r.render(v.scene, v.eye);
    const fading = v.highlightDraw(dt);
    if (lens) v.lens.end(v);
    v.gizmoDraw();
    if (still && !fading) v.last = 0;
    else { v.last = t; v.draw(); }
  },

  // ── selection ──────────────────────────────────────────────────────────────────────────────────────────────
  // the part under a point of the box (px), as a name, its meshes and the point hit (world), or null: the nearest
  // visible mesh hit. A mesh of several parts (userData.objects: [name, first triangle, triangles]) is looked into
  // by each part's box, then by the boxes of its triangles CHUNK at a time, nearest first, so that a pointer moving
  // over a face of 180 000 triangles costs a fraction of a millisecond; the other meshes by three's own raycast
  pick: function (x, y) {
    const v = this;
    if (!v.w || !v.h) return null;
    const ray = new THREE.Raycaster();
    ray.setFromCamera(v.ndcOf(x, y), v.eye);
    // what is nearer than the view's near plane is not drawn, and is not picked (the phone's own glass in its view)
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(v.eye.quaternion), near = v.ortho ? -Infinity : v.eye.near;
    const shown = function (o) {
      for (let p = o; p; p = p.parent) { if (!p.visible || p.userData.noPick) return false; }
      return true;
    };
    let best = null;
    const plain = [], hits = [];
    const local = new THREE.Ray(), inv = new THREE.Matrix4(), hit = new THREE.Vector3(), at = new THREE.Vector3();
    const A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3();
    v.root.traverse(function (m) {
      if (!m.isMesh || !shown(m)) return;
      if (!m.userData.objects) { plain.push(m); return; }
      m.updateWorldMatrix(true, false);
      const k = VIEW3D.chunks(m);
      inv.copy(m.matrixWorld).invert();
      const ray0 = new THREE.Ray().copy(ray.ray).applyMatrix4(inv);
      k.parts.forEach(function (p, r) {
        if (!ray0.intersectBox(p.box, at)) return;
        p.chunks.forEach(function (c) {
          if (ray0.intersectBox(c.box, at)) hits.push({ m: m, r: r, c: c, ray: ray0, d: at.applyMatrix4(m.matrixWorld).distanceTo(ray.ray.origin) });
        });
      });
    });
    hits.sort(function (p, q) { return p.d - q.d; });
    for (let i = 0; i < hits.length; i++) {
      const h = hits[i];
      if (best && h.d > best.d) break;
      const g = h.m.geometry, pos = g.attributes.position, idx = g.index;
      local.copy(h.ray);
      for (let k = h.c.first * 3; k < (h.c.first + h.c.count) * 3; k += 3) {
        A.fromBufferAttribute(pos, idx.getX(k)); B.fromBufferAttribute(pos, idx.getX(k + 1));
        C.fromBufferAttribute(pos, idx.getX(k + 2));
        if (!local.intersectTriangle(A, B, C, false, hit)) continue;
        const w = hit.clone().applyMatrix4(h.m.matrixWorld), d = w.distanceTo(ray.ray.origin);
        if (w.clone().sub(ray.ray.origin).dot(fwd) < near) continue;      // nearer than the view: not drawn
        if (!best || d < best.d) best = { d: d, name: h.m.userData.objects[h.r][0], hit: h.m, point: w };
      }
    }
    ray.near = v.ortho ? 0 : near / Math.max(1e-6, ray.ray.direction.dot(fwd));
    const other = ray.intersectObjects(plain, false).find(function (h) { return shown(h.object); });
    if (other && (!best || other.distance < best.d)) {
      let p = other.object;
      while (p && !p.name && p.parent) p = p.parent;
      return { name: (p && p.name) || '', mesh: other.object, hit: other.object, point: other.point.clone() };
    }
    return best ? { name: best.name, hit: best.hit, point: best.point } : null;
  },

  // the highlight: the part selected, the one under the pointer and the one let go, each fading in or out (HL_IN_S,
  // HL_OUT_S); the selected one with its name beside it. pick: as pick gives it, or null
  select: function (pick) { this.highlight('sel', pick); },
  hover: function (pick) { this.highlight('hov', pick); },

  highlight: function (slot, pick) {
    const v = this, hl = v.hl || (v.hl = { sel: null, hov: null, out: [] });
    const cur = hl[slot];
    if (cur && pick && cur.name === pick.name) {
      if (slot === 'sel' && pick.point) cur.anchor = pick.point;
      return;
    }
    if (!cur && !pick) return;
    if (cur) { cur.goal = 0; hl.out.push(cur); }
    hl[slot] = pick ? v.slot(pick, slot) : null;
    if (slot === 'sel') {
      v.sel = hl.sel;           // what the keys and the tests see: {name, parts}
      if (hl.sel && hl.hov && hl.hov.name === hl.sel.name) { hl.hov.goal = 0; hl.out.push(hl.hov); hl.hov = null; }
      v.labelFor(hl.sel);
    }
    v.draw();
  },

  // a part made ready to be highlighted: its triangles (the meshes it is in, their ranges) drawn again into the
  // mask on a layer of their own, its box, the point its name stands at
  slot: function (pick, kind) {
    const v = this, parts = [];
    const add = function (mesh, first, count) {
      const g = new THREE.BufferGeometry();
      Object.keys(mesh.geometry.attributes).forEach(function (k) { g.setAttribute(k, mesh.geometry.attributes[k]); });
      if (mesh.geometry.index) g.setIndex(mesh.geometry.index);
      if (count !== undefined) g.setDrawRange(first * 3, count * 3);
      const s = new THREE.Mesh(g, VIEW3D.kit().paint);
      s.layers.mask = 0;        // drawn only into the mask
      s.userData.noPick = true;
      s.frustumCulled = false;
      mesh.add(s);
      parts.push(s);
    };
    if (pick.mesh) add(pick.mesh);
    else {
      v.root.traverse(function (m) {
        if (!m.isMesh || !m.userData.objects) return;
        for (let p = m; p; p = p.parent) if (!p.visible) return;
        m.userData.objects.forEach(function (o) { if (o[0] === pick.name) add(m, o[1], o[2]); });
      });
    }
    const box = new THREE.Box3();
    const f = v.pointsFlat(parts), q = new THREE.Vector3();
    for (let i = 0; i < f.length; i += 3) box.expandByPoint(q.set(f[i], f[i + 1], f[i + 2]));
    const hit = pick.hit || pick.mesh, mat = (hit && hit.material && hit.material.name) || '';
    return { name: pick.name, kind: kind, parts: parts, box: box, anchor: pick.point || box.getCenter(new THREE.Vector3()),
             a: 0, goal: 1, layer: 0, material: mat };
  },

  // the name of what is selected: a small tag with its name in words (the page's map of names) and, lighter, what
  // it is made of (when the page knows) and its largest size (not with the option size: false); tied by a thin line
  // to the point clicked
  labelFor: function (s) {
    const v = this, t = v.tag;
    if (!s) { t.style.opacity = 0; v.lead.style.opacity = 0; return; }
    v.label.textContent = v.o.name ? v.o.name(s.name) : s.name;
    const size = s.box.isEmpty() || v.o.size === false ? 0
      : Math.max.apply(null, s.box.getSize(new THREE.Vector3()).toArray()) * (v.o.mm || 1);
    const big = size >= 20, mm = big ? Math.round(size) : Math.round(size * 10) / 10;
    const made = v.o.material ? v.o.material(s.material) : '';
    const said = size ? (typeof fmtNum === 'function' ? fmtNum(mm, big ? 0 : 1) : String(mm)) + ' mm' : '';
    v.info.textContent = [made, said].filter(Boolean).join(' \u00b7 ');
    v.info.style.display = v.info.textContent ? '' : 'none';
    t.style.visibility = 'visible';
    t.style.opacity = 0;
    t.__w = t.offsetWidth; t.__h = t.offsetHeight;       // measured once, not at every frame
    s.side = '';                                         // where it stands, chosen at its first frame
  },

  // a frame's highlight: the fades a step on; the mask (the highlighted parts where they are seen, each in a
  // channel) drawn again only when the view or the parts changed; then laid over the picture
  highlightDraw: function (dt) {
    const v = this, hl = v.hl;
    if (!hl) return false;
    let moving = false;
    const step = function (s) {
      if (!s) return;
      const t = s.goal > s.a ? VIEW3D.HL_IN_S : VIEW3D.HL_OUT_S;
      const a = s.a + Math.sign(s.goal - s.a) * dt / t;
      s.a = s.goal > s.a ? Math.min(s.goal, a) : Math.max(s.goal, a);
      if (s.a !== s.goal) moving = true;
    };
    step(hl.sel); step(hl.hov); hl.out.forEach(step);
    hl.out = hl.out.filter(function (s) {
      if (s.a > 0) return true;
      s.parts.forEach(function (m) { if (m.parent) m.parent.remove(m); m.geometry.dispose(); });
      return false;
    });
    const list = [hl.sel, hl.hov].concat(hl.out).filter(function (s) { return s && s.a > 0; }).slice(0, 3);
    v.labelPlace(hl.sel);
    if (!list.length) return moving;
    // (drawn where the frame is: the canvas, or the lens's picture while one bends the view)
    const r = v.r, kit = VIEW3D.kit(), lensRT = v.lensOn ? v.lensOn.rt : null;
    const size = lensRT ? new THREE.Vector2(lensRT.width, lensRT.height) : r.getDrawingBufferSize(new THREE.Vector2());
    if (!v.maskRT || v.maskRT.width !== size.x || v.maskRT.height !== size.y) {
      if (v.maskRT) v.maskRT.dispose();
      v.maskRT = new THREE.WebGLRenderTarget(size.x, size.y);
      v.maskKey = '';
    }
    // where on the screen they are, with room for the glow: what is drawn, and how much (scissored)
    const rect = v.screenRect(list, size, 10 * (window.devicePixelRatio || 1));
    if (!rect) return moving;
    const key = v.eye.matrixWorld.elements.join() + v.eye.projectionMatrix.elements.join() + '|'
      + list.map(function (s) { return s.name + s.kind; }).join() + '|' + size.x + 'x' + size.y;
    const auto = r.autoClear, shadows = r.shadowMap.autoUpdate, pr = r.getPixelRatio();
    r.shadowMap.autoUpdate = false;
    r.autoClear = false;
    if (key !== v.maskKey) {
      v.maskKey = key;
      const clear = r.getClearColor(new THREE.Color()), alpha = r.getClearAlpha(), rt = v.maskRT;
      rt.scissorTest = false;
      r.setRenderTarget(rt);
      r.setClearColor(0x000000, 1);
      r.clear();
      // only where they are, by a camera of that part of the view alone (what is outside it is not even sent):
      // everything in black for the depth, then each part in its own channel, drawn a little nearer than its own
      // surface (it wins over itself and over a surface flush with it, always the same way: no speckles)
      const mc = v.ortho ? (v.maskO || (v.maskO = new THREE.OrthographicCamera()))
        : (v.maskP || (v.maskP = new THREE.PerspectiveCamera()));
      mc.copy(v.eye, false);
      mc.setViewOffset(size.x, size.y, rect.x, size.y - rect.y - rect.h, rect.w, rect.h);
      mc.layers.set(0);
      rt.viewport.set(rect.x, rect.y, rect.w, rect.h);
      rt.scissor.set(rect.x, rect.y, rect.w, rect.h);
      rt.scissorTest = true;
      r.setRenderTarget(rt);
      const sky = v.scene.background;          // (the mask has no sky behind it)
      v.scene.background = null;
      v.scene.overrideMaterial = kit.black;
      r.render(v.scene, mc);
      v.scene.overrideMaterial = null;
      mc.layers.set(1);
      list.forEach(function (s, i) {
        s.parts.forEach(function (m) { m.layers.mask = 2; m.material = kit.paints[i]; });
        r.render(v.scene, mc);
        s.parts.forEach(function (m) { m.layers.mask = 0; });
      });
      v.scene.background = sky;
      rt.viewport.set(0, 0, size.x, size.y);
      rt.scissorTest = false;
      r.setRenderTarget(lensRT);
      r.setClearColor(clear, alpha);
    }
    const u = kit.quad.material.uniforms, amt = [0, 0, 0], sel = [0, 0, 0];
    list.forEach(function (s, i) { amt[i] = VIEW3D.ease(s.a) * (s.kind === 'hov' ? VIEW3D.HL_HOVER : 1); sel[i] = s.kind === 'hov' ? 0 : 1; });
    u.mask.value = v.maskRT.texture;
    u.px.value.set(1 / size.x, 1 / size.y);
    u.sel.value.set(sel[0], sel[1], sel[2]);
    u.amt.value.set(amt[0], amt[1], amt[2]);
    u.veil.value.copy(v.viewColor || new THREE.Color(1, 1, 1));
    // a selection veils the whole view; the pointer's part only lightens its own place
    const whole = sel.some(function (x) { return x; });
    if (!whole) { r.setScissor(rect.x / pr, rect.y / pr, rect.w / pr, rect.h / pr); r.setScissorTest(true); }
    r.render(kit.quad.scene, kit.quad.cam);
    r.setScissorTest(false);
    r.autoClear = auto;
    r.shadowMap.autoUpdate = shadows;
    return moving;
  },

  // the box (device px, from the bottom left, as the scissor wants it) the highlighted parts take on the screen,
  // with a margin; null when none of it is in front of the eye
  screenRect: function (list, size, margin) {
    const v = this, p = new THREE.Vector3();
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, behind = false;
    list.forEach(function (s) {
      const b = s.box;
      if (b.isEmpty()) return;
      [b.min.x, b.max.x].forEach(function (x) { [b.min.y, b.max.y].forEach(function (y) { [b.min.z, b.max.z].forEach(function (z) {
        p.set(x, y, z).applyMatrix4(v.eye.matrixWorldInverse);
        if (!v.ortho && p.z > -v.eye.near) { behind = true; return; }
        p.applyMatrix4(v.eye.projectionMatrix);
        x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
      }); }); });
    });
    if (behind) return { x: 0, y: 0, w: size.x, h: size.y };
    if (!isFinite(x0)) return null;
    const X0 = Math.max(0, Math.floor((x0 + 1) / 2 * size.x - margin)), X1 = Math.min(size.x, Math.ceil((x1 + 1) / 2 * size.x + margin));
    const Y0 = Math.max(0, Math.floor((y0 + 1) / 2 * size.y - margin)), Y1 = Math.min(size.y, Math.ceil((y1 + 1) / 2 * size.y + margin));
    if (X1 <= X0 || Y1 <= Y0) return null;
    return { x: X0, y: Y0, w: X1 - X0, h: Y1 - Y0 };
  },

  // a point of the world on the box (css px from its top left) and its depth as project() gives it; and the point of
  // the view (as a raycaster takes it) a point of the box shows. Through the lens when one bends the view (18-lens.js)
  onBox: function (p) {
    if (this.lens && this.lens.on(this)) return this.lens.onBox(this, p);
    const q = p.clone().project(this.eye);
    return { x: (q.x + 1) / 2 * this.w, y: (1 - q.y) / 2 * this.h, z: q.z };
  },
  ndcOf: function (x, y) {
    if (this.lens && this.lens.on(this)) return this.lens.ndc(this, x, y);
    return new THREE.Vector2(x / this.w * 2 - 1, 1 - y / this.h * 2);
  },

  // the tag beside the part, on the side chosen when it was selected where there is room (right, left, above, below
  // the part as the screen shows it; a big part: near the point clicked), inside the box; its line from the point
  // clicked to the tag's nearest edge; it follows the part as the view turns (in fractions of a pixel: no
  // shaking), faded with the highlight
  labelPlace: function (s) {
    const v = this, t = v.tag, L = v.lead;
    if (!s || s.a <= 0) {
      t.style.opacity = 0; L.style.opacity = 0;
      if (!s) t.style.visibility = 'hidden';
      return;
    }
    const p = v.onBox(s.anchor);
    if (p.z > 1 || p.z < -1) { t.style.opacity = 0; L.style.opacity = 0; return; }
    const ax = p.x, ay = p.y, W = t.__w || t.offsetWidth, H = t.__h || t.offsetHeight;
    const gap = VIEW3D.TAG_GAP, m = 8;
    // the part on the screen
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const b = s.box, q = new THREE.Vector3();
    [b.min.x, b.max.x].forEach(function (x) { [b.min.y, b.max.y].forEach(function (y) { [b.min.z, b.max.z].forEach(function (z) {
      const o = v.onBox(q.set(x, y, z)), sx = o.x, sy = o.y;
      x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy);
    }); }); });
    if (!s.side) {
      s.side = x1 + gap + W + m <= v.w ? 'right' : x0 - gap - W >= m ? 'left' : y0 - gap - H >= m ? 'top'
        : y1 + gap + H + m <= v.h ? 'bottom' : 'near';
    }
    let x = ax + gap, y = ay - H - gap;
    if (s.side === 'right') { x = x1 + gap; y = ay - H / 2; }
    else if (s.side === 'left') { x = x0 - gap - W; y = ay - H / 2; }
    else if (s.side === 'top') { x = ax - W / 2; y = y0 - gap - H; }
    else if (s.side === 'bottom') { x = ax - W / 2; y = y1 + gap; }
    x = Math.max(m, Math.min(v.w - W - m, x)); y = Math.max(m, Math.min(v.h - H - m, y));
    const ex = Math.max(x, Math.min(x + W, ax)), ey = Math.max(y, Math.min(y + H, ay)), o = VIEW3D.ease(s.a);
    t.style.transform = 'translate3d(' + x.toFixed(2) + 'px,' + y.toFixed(2) + 'px,0)';
    t.style.opacity = o;
    L.style.opacity = o;
    v.leadLine.setAttribute('x1', ax.toFixed(2)); v.leadLine.setAttribute('y1', ay.toFixed(2));
    v.leadLine.setAttribute('x2', ex.toFixed(2)); v.leadLine.setAttribute('y2', ey.toFixed(2));
    v.leadDot.setAttribute('cx', ax.toFixed(2)); v.leadDot.setAttribute('cy', ay.toFixed(2));
  },

  // ── the gizmo, the name, the keys ──────────────────────────────────────────────────────────────────────────
  // in the box: Blender's navigation gizmo top right (the axes: a click on one looks along it, a drag turns; below
  // them the lens that zooms and the hand that moves, dragged, the camera's view where there is one and the
  // perspective or orthographic view, clicked) and the name of what is selected
  overlay: function () {
    const v = this, ns = 'http://www.w3.org/2000/svg', S = VIEW3D, R = S.GIZMO / 2;
    const id = 'v3g' + (S.gizmos++);
    const svg = document.createElementNS(ns, 'svg');
    const W = 2 * (S.GIZMO_OFFSET + R), H = 240;
    svg.setAttribute('width', W); svg.setAttribute('height', H);
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.style.cssText = 'position:absolute;top:0;right:0;overflow:visible;touch-action:none;user-select:none;'
      + 'pointer-events:none';
    svg.classList.add('v3-gizmo');
    const el = function (tag, attrs, parent) {
      const e = document.createElementNS(ns, tag);
      Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
      (parent || svg).append(e);
      return e;
    };
    const defs = el('defs', {});
    // the disc behind it under the pointer, and where it takes the pointer
    v.gz = { svg: svg, W: W, hot: -1, active: false, axes: [], tools: [] };
    v.gz.disc = el('circle', { r: R, fill: S.HOVER, 'pointer-events': 'all', 'fill-opacity': 0 });
    v.gz.layer = el('g', { 'pointer-events': 'none' });
    // each axis: its line (positive ones), its circle (the fill and the ring) and its letter, made once; placed,
    // coloured and put in depth order by gizmoDraw
    const k = R * S.HANDLE * 1.25 / 10, G = S.GLYPHS;
    VIEW3D.AXES.forEach(function (a, i) {
      [1, -1].forEach(function (sign) {
        const pos = sign > 0, g = el('g', {}, v.gz.layer);
        const grad = el('linearGradient', { id: id + a[0] + sign, gradientUnits: 'userSpaceOnUse' }, defs);
        const stops = [el('stop', { offset: 0 }, grad), el('stop', { offset: 1 }, grad)];
        const line = el('line', { stroke: 'url(#' + id + a[0] + sign + ')', 'stroke-linecap': 'butt' }, g);
        const fill = el('circle', {}, g), ring = el('circle', { fill: 'none' }, g);
        // the letter: centred on the circle by its outline; '-X' as Blender sets it, from 0.55 of its width to the
        // left, its height centred
        const text = el('g', {}, g), inner = el('g', { transform: 'scale(' + k + ')' }, text);
        if (pos) el('path', { d: G[a[0]].d }, inner);
        else {
          const m = G['-'], L = G[a[0]], w = m.adv + L.adv, y0 = Math.min(m.box[1], L.box[1]);
          const y1 = Math.max(m.box[3], L.box[3]), left = -0.55 * w, base = (y1 - y0) / 2 + y0;
          [[m, 0], [L, m.adv]].forEach(function (p) {
            const b = p[0].box;
            el('path', { d: p[0].d, transform: 'translate(' + (left + p[1] + (b[0] + b[2]) / 2).toFixed(3) + ' '
                                               + (base - (b[1] + b[3]) / 2).toFixed(3) + ')' }, inner);
          });
        }
        v.gz.axes.push({ axis: i, sign: sign, index: 2 * i + (pos ? 1 : 0), opposite: 2 * i + (pos ? 0 : 1),
                         g: g, grad: grad, line: line, stops: stops, fill: fill, ring: ring, text: text,
                         color: a[1], letter: (pos ? '' : '-') + a[0] });
      });
    });
    // the buttons below, on their backdrop
    v.gz.back = el('rect', { fill: S.BACKDROP });
    const tool = function (name, mode, click) {
      const g = el('g', { cursor: click ? 'pointer' : mode === 'pan' ? 'move' : 'ns-resize' });
      const hit = el('circle', { r: S.MINI / 2, fill: 'transparent', 'pointer-events': 'all' }, g);
      const icon = el('g', { 'pointer-events': 'none' }, g);
      icon.innerHTML = S.ICONS[name];
      const t = { name: name, mode: mode, click: click, g: g, hit: hit, icon: icon, lit: false };
      g.addEventListener('pointerenter', function () { t.lit = true; v.gizmoTools(); });
      g.addEventListener('pointerleave', function () { t.lit = false; v.gizmoTools(); });
      v.gz.tools.push(t);
      return t;
    };
    tool('zoom', 'zoom');
    tool('pan', 'pan');
    if (v.o.camera) tool('camera', '', function () { v.o.camera(); });
    tool('persp', '', function () { v.setOrtho(!v.ortho); if (v.o.changed) v.o.changed(); v.gizmoTools(); });
    v.box.append(svg);
    // the name of what is selected: a small tag (its name, and lighter what it is made of and its size), tied by a
    // thin line to the point clicked
    const lead = document.createElementNS(ns, 'svg');
    lead.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;overflow:visible;'
      + 'opacity:0;color:var(--text-1, #222)';
    v.leadLine = el('line', { stroke: 'currentColor', 'stroke-width': 1, 'stroke-opacity': 0.55 }, lead);
    v.leadDot = el('circle', { r: 2.6, fill: 'currentColor', stroke: 'var(--surface-0, #fff)', 'stroke-width': 1.2 }, lead);
    const tag = document.createElement('div');
    tag.className = 'v3-tag';
    tag.style.cssText = 'position:absolute;left:0;top:0;padding:6px 12px 7px;border-radius:12px;max-width:260px;'
      + 'background:var(--overlay, rgba(240,240,244,0.95));color:var(--text-1, #111);pointer-events:none;'
      + 'box-shadow:0 2px 10px var(--shadow, rgba(0,0,0,0.22));opacity:0;visibility:hidden;will-change:transform,opacity;'
      + 'display:flex;flex-direction:column;gap:1px';
    const name = document.createElement('span');
    name.style.cssText = 'font-size:13px;font-weight:600;line-height:17px;white-space:nowrap;overflow:hidden;'
      + 'text-overflow:ellipsis';
    const info = document.createElement('span');
    info.style.cssText = 'font-size:11.5px;line-height:15px;color:var(--text-3, #777);white-space:nowrap';
    tag.append(name, info);
    v.box.append(lead, tag);
    v.tag = tag; v.lead = lead; v.label = name; v.info = info;
    v.gizmoPlace();
    // the axis under the pointer, as Blender finds it: within the disc, the nearest axis's end (the one pointing
    // at the eye left out while the view looks along it); a click looks along it, a drag turns (on the disc), zooms
    // or moves (the buttons)
    const hover = function (e) {
      const b = svg.getBoundingClientRect();
      const p = [(e.clientX - b.left - v.gz.cx) / R, -(e.clientY - b.top - v.gz.cy) / R];
      const on = p[0] * p[0] + p[1] * p[1] <= 1;
      let best = -1, bd = Infinity;
      if (on && v.gz.dirs) {
        v.gz.dirs.forEach(function (d, i) {
          [-1, 1].forEach(function (s) {
            const x = d.x * s, y = d.y * s;
            if (x * x + y * y < 1e-6 && (d.z > 0) === (s > 0)) return;
            const dd = (x - p[0]) * (x - p[0]) + (y - p[1]) * (y - p[1]);
            if (dd < bd) { bd = dd; best = 2 * i + (s > 0 ? 1 : 0); }
          });
        });
      }
      if (best !== v.gz.hot || on !== v.gz.active) { v.gz.hot = best; v.gz.active = on; v.gizmoDraw(); }
    };
    v.gz.hover = hover;
    v.gz.disc.addEventListener('pointermove', function (e) { if (!v.gzDrag) hover(e); });
    v.gz.disc.addEventListener('pointerleave', function () {
      if (!v.gzDrag) { v.gz.hot = -1; v.gz.active = false; v.gizmoDraw(); }
    });
    svg.addEventListener('pointerdown', function (e) {
      e.preventDefault(); e.stopPropagation();
      const t = v.gz.tools.find(function (x) { return x.g.contains(e.target); });
      v.gzDrag = { id: e.pointerId, x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY,
                   mode: t ? t.mode : 'orbit', tool: t, part: t ? -1 : v.gz.hot, still: true };
      try { svg.setPointerCapture(e.pointerId); } catch (err) { /* a pointer the browser does not track */ }
    });
    svg.addEventListener('pointermove', function (e) {
      const d = v.gzDrag;
      if (!d || d.id !== e.pointerId) return;
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      d.x = e.clientX; d.y = e.clientY;
      if (d.still && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < VIEW3D.CLICK_PX) return;
      d.still = false;
      if (!d.mode) return;
      v.size();
      v.drag(d.mode, dx, dy);
    });
    const end = function (e) {
      const d = v.gzDrag;
      if (!d || d.id !== e.pointerId) return;
      v.gzDrag = null;
      if (d.still && d.tool && d.tool.click) d.tool.click();
      else if (d.still && d.part >= 0) v.axisLook(d.part >> 1, d.part & 1 ? 1 : -1);
      if (!d.tool) hover(e);
    };
    svg.addEventListener('pointerup', end);
    svg.addEventListener('pointercancel', end);
    svg.addEventListener('wheel', function (e) { e.stopPropagation(); }, { passive: true });
  },

  // where the gizmo and its buttons stand, as Blender sets them: its centre GIZMO_OFFSET + half its size from the
  // top and right edges, the buttons in a column under it from 2.2 times that, each MINI + MINI_GAP below the
  // last, on a backdrop as wide as their bounds; on Blender's whole pixels (at the interface's scale)
  gizmoPlace: function () {
    const v = this, gz = v.gz, S = VIEW3D, R = S.GIZMO / 2, s = window.devicePixelRatio || 1;
    const at = function (a) { return -Math.round(-a * s) / s; };       // roundf(edge - a), as px from the edge
    const off = S.GIZMO_OFFSET + R, step = S.MINI + S.MINI_GAP;
    gz.cx = gz.W - at(off); gz.cy = at(off);
    gz.disc.setAttribute('cx', gz.cx); gz.disc.setAttribute('cy', gz.cy);
    const mx = gz.W - at(step * 0.75), my = at(off * 2.2), half = S.MINI / 2;
    gz.tools.forEach(function (t, i) {
      t.x = mx; t.y = my + at(step * i);
      t.hit.setAttribute('cx', t.x); t.hit.setAttribute('cy', t.y);
      // (drawn on 16 px with a margin: brought up to the size of Blender's own, which reach the edges)
      const k = 1.12;
      t.icon.setAttribute('transform', 'translate(' + (t.x - k * S.ICON / 2) + ' ' + (t.y - k * S.ICON / 2) + ') scale(' + k + ')');
    });
    const first = gz.tools[0].y, last = gz.tools[gz.tools.length - 1].y, pad = half * 0.2;
    gz.back.setAttribute('x', mx - half); gz.back.setAttribute('width', 2 * half);
    gz.back.setAttribute('y', first - half - pad); gz.back.setAttribute('height', last - first + 2 * half + 2 * pad);
    gz.back.setAttribute('rx', half);
    gz.scale = s;
    v.gizmoTools();
  },

  // the buttons: lit under the pointer; the perspective one shows the view there is
  gizmoTools: function () {
    const v = this;
    v.gz.tools.forEach(function (t) {
      if (t.name === 'persp' || t.name === 'ortho') {
        const want = v.ortho ? 'ortho' : 'persp';
        if (t.name !== want) { t.name = want; t.icon.innerHTML = VIEW3D.ICONS[want]; }
      }
      const c = t.lit ? '#ffffff' : VIEW3D.ICON_COLOR, a = t.lit ? 1 : 0.6;
      t.icon.setAttribute('stroke', c); t.icon.setAttribute('fill', c);
      t.icon.setAttribute('opacity', a);
    });
  },

  // looked at from the end of an axis of the gizmo, towards the centre (X: from the right, -Y: from the front...);
  // pressed again when the view already looks that way, from the other end, as Blender does
  axisLook: function (axis, sign) {
    const v = this;
    const along = [v.R, v.F, v.U][axis].clone().multiplyScalar(sign);
    if (new THREE.Vector3(0, 0, 1).applyQuaternion(v.goal.q).dot(along) > 0.9998) sign = -sign;
    const A = [v.R, v.F, v.U][axis].clone().multiplyScalar(sign);
    const up = axis === 2 ? v.F.clone().multiplyScalar(sign) : v.U;
    v.goal.q.copy(v.look(A.negate(), up));
    v.user();
  },

  // the gizmo turned as the view is, as Blender draws it: the axes from the farthest to the nearest; an axis's
  // colour mixed with the view's own the more the farther it points (half at the back), its circle 8 % larger at
  // the front than in the middle; a positive axis a line from the centre in its colour (from three quarters of it)
  // and a full circle with its letter in black (white under the pointer); a negative one its ring round a quarter
  // of its colour, faded out as it points away, its letter only under the pointer; looking along an axis, the end
  // pointing away is not drawn and the near one has its letter
  gizmoDraw: function () {
    const v = this, gz = v.gz, S = VIEW3D;
    if (!gz) return;
    if (gz.scale !== (window.devicePixelRatio || 1)) v.gizmoPlace();
    // nothing to do while the view only moves or zooms (its turn the same): the page's work spared
    const sq = v.shown.q, bg0 = v.viewColor ? v.viewColor.getHex() : -1;
    const key = sq.x.toFixed(6) + sq.y.toFixed(6) + sq.z.toFixed(6) + sq.w.toFixed(6) + '|' + gz.hot + gz.active + bg0
      + '|' + gz.scale;
    if (key === gz.key) return;
    gz.key = key;
    const R = S.GIZMO / 2, px = Math.max(1, Math.floor(72 * gz.scale / 64)) / gz.scale;
    const LW = S.GIZMO / 40 * px, RW = S.GIZMO / 60 * px, H = 1 - S.HANDLE;
    const q = v.shown.q.clone().invert();
    const dirs = [v.R, v.F, v.U].map(function (a) { return a.clone().applyQuaternion(q); });
    gz.dirs = dirs;
    let align = -1;
    for (let i = 0; i < 3; i++) if (dirs[i].x * dirs[i].x + dirs[i].y * dirs[i].y < 1e-6) { align = i; break; }
    const bg = v.viewColor || new THREE.Color(0.24, 0.24, 0.24), white = new THREE.Color(1, 1, 1);
    const css = function (c, a) {
      return 'rgba(' + Math.round(c.r * 255) + ',' + Math.round(c.g * 255) + ',' + Math.round(c.b * 255) + ','
        + (a === undefined ? 1 : Math.max(0, Math.min(1, a))).toFixed(3) + ')';
    };
    const f = function (x) { return x.toFixed(3); };
    // an attribute written only when it changes (the page redraws what changed)
    const set = function (e, k, val) {
      const c = e.__a || (e.__a = {});
      if (c[k] !== val) { c[k] = val; e.setAttribute(k, val); }
    };
    set(gz.disc, 'fill-opacity', gz.active ? 1 : 0);
    const order = gz.axes.map(function (a) { return { a: a, depth: dirs[a.axis].z * a.sign }; })
      .sort(function (p, r) { return p.depth - r.depth; });
    order.forEach(function (it) {
      const a = it.a, depth = it.depth, pos = a.sign > 0, d = dirs[a.axis];
      const behind = depth <= S.DEPTH_BIAS * (pos ? -1 : 1);
      const front = align === a.axis && !behind, back = align === a.axis && behind;
      const lit = gz.hot === a.index || (front && gz.hot === a.opposite);
      const col = new THREE.Color(a.color);
      const fading = bg.clone().lerp(col, (depth + 1) * 0.25 + 0.5), middle = bg.clone().lerp(col, 0.75);
      const x = d.x * a.sign, y = -d.y * a.sign, bx = gz.cx + x * R * H, by = gz.cy + y * R * H;
      // the line: from a little behind the centre to under the circle
      const lineOn = (pos || align !== -1) && x * x + y * y > 1e-9;
      set(a.line, 'visibility', lineOn ? 'visible' : 'hidden');
      if (lineOn) {
        const k0 = -H * 0.66 * LW, k1 = R * H * H;
        const x1 = f(gz.cx + x * k0), y1 = f(gz.cy + y * k0), x2 = f(gz.cx + x * k1), y2 = f(gz.cy + y * k1);
        [a.line, a.grad].forEach(function (e) {
          set(e, 'x1', x1); set(e, 'y1', y1); set(e, 'x2', x2); set(e, 'y2', y2);
        });
        set(a.line, 'stroke-width', f(LW));
        set(a.stops[0], 'stop-color', css(middle)); set(a.stops[1], 'stop-color', css(fading));
      }
      // the circle
      set(a.fill, 'visibility', back ? 'hidden' : 'visible');
      set(a.ring, 'visibility', back || pos ? 'hidden' : 'visible');
      if (!back) {
        const rad = R * S.HANDLE * ((depth + 1) * 0.08 + 0.92);
        let inner = css(fading), ring = css(fading);
        if (!pos && front) ring = css(white.clone().lerp(col, 0.5), Math.min(depth + 1, 1));
        else if (!pos) inner = css(bg.clone().lerp(col, 0.25), Math.min(depth + 1, 1));
        [a.fill, a.ring].forEach(function (c) { set(c, 'cx', f(bx)); set(c, 'cy', f(by)); });
        set(a.fill, 'r', f(pos ? rad : rad - RW / 2));
        set(a.fill, 'fill', inner);
        set(a.ring, 'r', f(rad - RW / 2));
        set(a.ring, 'stroke', ring);
        set(a.ring, 'stroke-width', f(RW));
      }
      // the letter
      const textOn = (pos || lit || a.axis === align) && !back;
      set(a.text, 'visibility', textOn ? 'visible' : 'hidden');
      if (textOn) {
        set(a.text, 'transform', 'translate(' + f(bx) + ' ' + f(by) + ')');
        set(a.text, 'fill', lit ? '#ffffff' : '#000000');
        set(a.text, 'fill-opacity', lit || gz.active ? 1 : 0.9);
      }
    });
    // nearest last, put in that order only when it changed
    const ok = order.map(function (it) { return it.a.index; }).join();
    if (ok !== gz.orderKey) { gz.orderKey = ok; order.forEach(function (it) { gz.layer.append(it.a.g); }); }
  },

  // a drag of the view: turned, moved or zoomed by pixels of the box
  drag: function (mode, dx, dy) {
    const v = this;
    if (mode === 'pan') v.pan(dx, dy);
    else if (mode === 'zoom') v.zoom(Math.exp(dy * VIEW3D.DRAG_ZOOM));
    else {
      const k = VIEW3D.ORBIT / Math.max(1, Math.min(v.w, v.h));
      v.turn(dx * k, dy * k);
    }
  },

  // a key of Blender's: true when it was this viewer's
  key: function (e) {
    const v = this, c = e.code, step = VIEW3D.STEP_DEG * Math.PI / 180;
    v.size();
    if (c === 'Numpad1') v.axisView(e.ctrlKey ? 'back' : 'front');
    else if (c === 'Numpad3') v.axisView(e.ctrlKey ? 'left' : 'right');
    else if (c === 'Numpad7') v.axisView(e.ctrlKey ? 'bottom' : 'top');
    else if (c === 'Numpad9') v.opposite();
    else if (c === 'Numpad5') v.setOrtho(!v.ortho);
    else if (c === 'Numpad2' || c === 'Numpad8' || c === 'Numpad4' || c === 'Numpad6') {
      const dir = { Numpad2: [0, -1], Numpad8: [0, 1], Numpad4: [-1, 0], Numpad6: [1, 0] }[c];
      if (e.ctrlKey) v.pan(-dir[0] * v.w * VIEW3D.PAN_STEP, dir[1] * v.h * VIEW3D.PAN_STEP);
      else v.turn(-dir[0] * step, dir[1] * step);         // the eye goes round: 8 up, 2 down, 4 left, 6 right
    } else if (c === 'NumpadAdd') v.zoom(1 / VIEW3D.KEY_ZOOM);
    else if (c === 'NumpadSubtract') v.zoom(VIEW3D.KEY_ZOOM);
    else if (c === 'Numpad0') { if (!v.o.camera) return false; v.o.camera(); }
    else if (c === 'NumpadDecimal') { v.frameSelected(); v.used = performance.now(); }
    else if (c === 'Home') { v.moved = true; v.fit(v.goal.q.clone(), v.bounds); v.used = performance.now(); }
    else if (c === 'KeyA' && e.altKey) v.select(null);
    else if (c === 'Escape' && v.sel) v.select(null);
    else return false;
    if (c !== 'Numpad0' && c !== 'KeyA' && v.o.changed) v.o.changed();    // a view of the keys is the hand's own
    return true;
  },

  // the mouse as Blender's: the middle button turns (Shift: moves, Ctrl: zooms), the left one with Alt does the
  // same (a touchpad), a click of the left one selects; the wheel zooms towards the centre (Ctrl: moves across,
  // Shift: up and down). On the canvas there is now: wired again on a new one
  wire: function () {
    const v = this, c = v.canvas;
    v.press = null;
    c.addEventListener('pointerenter', function () { v.hot = true; });
    c.addEventListener('pointerleave', function () { v.hot = false; v.hoverXY = null; v.hover(null); });
    c.addEventListener('pointerdown', function (e) {
      if (v.press) return;
      const mid = e.button === 1 || (e.button === 0 && e.altKey);
      // a model of one piece (o.pick false: the plate) has nothing to select: the left button turns it too
      const left = v.o.pick === false ? (e.shiftKey ? 'pan' : e.ctrlKey ? 'zoom' : 'orbit') : 'click';
      const mode = mid ? (e.shiftKey ? 'pan' : e.ctrlKey ? 'zoom' : 'orbit') : e.button === 0 ? left : '';
      if (!mode) return;
      v.used = performance.now();
      v.press = { id: e.pointerId, x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, mode: mode };
      if (mode !== 'click') v.hover(null);
      try { c.setPointerCapture(e.pointerId); } catch (err) { /* a pointer the browser does not track */ }
      if (mode !== 'click') c.style.cursor = mode === 'pan' ? 'move' : mode === 'zoom' ? 'ns-resize' : 'grabbing';
    });
    // the middle button moves the view, it does not scroll the page
    c.addEventListener('mousedown', function (e) { if (e.button === 1) e.preventDefault(); });
    c.addEventListener('auxclick', function (e) { if (e.button === 1) e.preventDefault(); });
    c.addEventListener('pointermove', function (e) {
      const d = v.press;
      if (!d) { v.hoverAt(e); return; }
      if (e.pointerId !== d.id || !v.size()) return;
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      d.x = e.clientX; d.y = e.clientY;
      if (d.mode !== 'click') v.drag(d.mode, dx, dy);
    });
    const end = function (e) {
      const d = v.press;
      if (!d || e.pointerId !== d.id) return;
      v.press = null;
      c.style.cursor = '';
      if (d.mode === 'click' && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < VIEW3D.CLICK_PX && v.size()) {
        const b = c.getBoundingClientRect();
        v.select(v.pick(e.clientX - b.left, e.clientY - b.top));
      }
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('contextmenu', function (e) { e.preventDefault(); });
    c.addEventListener('wheel', function (e) {
      e.preventDefault();
      if (!v.size()) return;
      const px = e.deltaY * (e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? v.h : 1);
      if (e.ctrlKey) v.pan(-px * VIEW3D.WHEEL_PAN, 0);
      else if (e.shiftKey) v.pan(0, -px * VIEW3D.WHEEL_PAN);
      else v.zoom(Math.exp(px * VIEW3D.WHEEL));
    }, { passive: false });
  },

  // the part under the pointer, looked for at most every HOVER_MS while the pointer moves over the model and no
  // button is down: a lighter highlight than the selection's
  hoverAt: function (e) {
    const v = this;
    if (v.o.pick === false) return;
    v.hoverXY = [e.clientX, e.clientY];
    if (v.hoverTimer) return;
    const wait = Math.max(0, VIEW3D.HOVER_MS - (performance.now() - (v.hoverLast || 0)));
    v.hoverTimer = setTimeout(function () {
      v.hoverTimer = 0;
      v.hoverLast = performance.now();
      if (v.press || !v.hoverXY || !v.size()) return;
      const b = v.canvas.getBoundingClientRect(), p = v.pick(v.hoverXY[0] - b.left, v.hoverXY[1] - b.top);
      v.hover(p && v.sel && p.name === v.sel.name ? null : p);
    }, wait);
  },

  // the GPU's memory given back for good: geometries, materials, the context
  dispose: function () {
    const v = this;
    v.gone = true;
    if (v.raf) cancelAnimationFrame(v.raf);
    clearTimeout(v.sleeper);
    v.observer.disconnect();
    VIEW3D.free(v.root);
    if (v.maskRT) v.maskRT.dispose();
    if (v.r) v.r.dispose();
    v.r = null;
    // what it laid over its box (the gizmo, a part's tag and its line): a new viewer there makes its own
    [v.gz && v.gz.svg, v.tag, v.lead].forEach(function (e) { if (e) e.remove(); });
    VIEW3D.all = VIEW3D.all.filter(function (x) { return x !== v; });
  },
};

// the visible meshes under a root (not those of the outline or of the marks the page lays over the model)
function v_root_meshes(root, list, visible) {
  root.traverse(function (o) { if (o.isMesh && o.geometry && o.geometry.attributes.position && visible(o)) list.push(o); });
}

VIEW3D.free = function (obj) {
  obj.traverse(function (o) {
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
};

/** The 3D-printed plate, in the board step. */
const PLATE3D = {
  URL: 'ui/models/charuco_plate.glb',
  viewer: null, model: null, failed: false,

  // the viewer and the model made ready, without showing them (at idle, or at the first show)
  prepare: function () {
    if (PLATE3D.viewer || PLATE3D.failed) return !!PLATE3D.viewer;
    // the first view as the plate is held to be measured: from the front, a little to the left, above
    // (the GLB is glTF's Y up: up is +Y, the front view looks along -Z, as Blender's +Y)
    PLATE3D.viewer = VIEW3D.create($('bd3d'), $('bd3dCanvas'), { margin: 1.1, dir: [-0.179, 0.796, 0.579],
                                                                 up: [0, 1, 0], front: [0, 0, -1],
                                                                 failed: PLATE3D.fail,
                                                                 pick: false,     // one piece: nothing to select
                                                                 mm: 1000 });     // the plate's GLB is in metres
    if (!PLATE3D.viewer) { PLATE3D.failed = true; return false; }
    GLB.load(PLATE3D.URL, true, function (res) {
      if (!PLATE3D.viewer) return;
      PLATE3D.model = res.object;
      PLATE3D.viewer.set(res.object);
      if (!PLATE3D.viewer.r) PLATE3D.viewer.warm();      // not on screen yet: its shaders ready for when it is
    }, function (e) { console.warn('plate model: ' + e); });
    window.addEventListener('pagehide', PLATE3D.dispose);
    return true;
  },

  // no GL context to be had (WebGL off, or the GPU's process gone): the box goes, the steps beside it stay
  fail: function () {
    PLATE3D.failed = true;
    $('bd3d').classList.add('hidden');
  },

  show: function () {
    if (!PLATE3D.prepare()) { $('bd3d').classList.add('hidden'); return; }
    PLATE3D.viewer.background();
    PLATE3D.viewer.draw();
  },

  hide: function () { /* nothing runs while it is hidden: it draws only on a change, its context given back */ },

  dispose: function () {
    if (PLATE3D.viewer) PLATE3D.viewer.dispose();
    PLATE3D.viewer = null;
    PLATE3D.failed = true;
  },
};

// made ready a moment after the page is up, while the app is idle: the board step then shows the plate at once
window.addEventListener('load', function () {
  setTimeout(function () {
    (window.requestIdleCallback || function (f) { return setTimeout(f, 0); })(function () { PLATE3D.prepare(); },
                                                                               { timeout: 3000 });
  }, 1200);
});
