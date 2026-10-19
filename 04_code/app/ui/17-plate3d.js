// ImpactGuide · the 3D-printed plate in 3D, in the board step when the board
// is physical: the plate as it comes off a two-colour printer (white plate,
// black print in relief), turned with the mouse (drag), zoomed (wheel), back
// to its first view with a double click. The model is ui/models/charuco_plate.glb
// (tools/build_plate_model.py, from the STL of 02_cad/stl), read here with a
// few lines (a GLB of two meshes, no loader needed); the two lengths the
// caliper reads are drawn on it, the long one in the accent. three.js r128
// (ui/libs), lit as UniGuide lights its models: a soft sky and ground, a key
// light, a fill and a rim, and a studio gradient for the reflections. It draws
// only when something changes.
'use strict';

// The kinds of numbers a glTF accessor holds (componentType), as typed arrays.
const GLB_TYPES = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const GLB_MAGIC = 0x46546C67;     // 'glTF'

/** The 3D-printed plate in 3D (three.js), turned with the mouse, drawn only when something changes. */
const PLATE3D = {
  r: null, scene: null, cam: null, root: null, model: null, lines: null, raf: 0, info: null, failed: false,
  turned: false,    // moved by hand: a new size of its box keeps the view
  short: false,     // the short side's length drawn too (its fold is open)
  view: { az: -0.3, el: 0.92, dist: 0.62 },
  home: { az: -0.3, el: 0.92, dist: 0.62 },

  tok: function (name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#808080'; },

  init: function () {
    if (PLATE3D.r || PLATE3D.failed) return !!PLATE3D.r;
    if (!window.THREE) { PLATE3D.failed = true; return false; }
    const canvas = $('bd3dCanvas');
    try {
      PLATE3D.r = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
    } catch (e) {
      PLATE3D.failed = true;
      return false;
    }
    const r = PLATE3D.r;
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    try { r.outputEncoding = THREE.sRGBEncoding; } catch (e) { /* r128 has it */ }
    r.setClearColor(new THREE.Color(PLATE3D.tok('--surface-2')), 1);
    const sc = new THREE.Scene();
    PLATE3D.scene = sc;
    PLATE3D.cam = new THREE.PerspectiveCamera(30, 1, 0.01, 10);
    const white = new THREE.Color(1, 1, 1);
    sc.add(new THREE.HemisphereLight(white, new THREE.Color(0.25, 0.24, 0.22), 0.55));
    const key = new THREE.DirectionalLight(white, 0.95);
    key.position.set(0.35, 0.8, 0.45);
    sc.add(key);
    const fill = new THREE.DirectionalLight(white, 0.3);
    fill.position.set(-0.6, 0.35, -0.2);
    sc.add(fill);
    const rim = new THREE.DirectionalLight(white, 0.3);
    rim.position.set(-0.2, 0.5, -0.8);
    sc.add(rim);
    PLATE3D.environment();
    PLATE3D.root = new THREE.Group();
    sc.add(PLATE3D.root);
    PLATE3D.wire(canvas);
    new ResizeObserver(function () {
      PLATE3D.size();
      if (!PLATE3D.turned) Object.assign(PLATE3D.view, PLATE3D.home);
      PLATE3D.draw();
    }).observe($('bd3d'));
    // the GPU reset: three.js restores its context by itself; the reflections are made again and the plate drawn
    canvas.addEventListener('webglcontextrestored', function () { PLATE3D.environment(); PLATE3D.draw(); });
    window.addEventListener('pagehide', PLATE3D.dispose);
    PLATE3D.load();
    return true;
  },

  // The soft studio gradient the reflections come from (made on the GPU: again after a reset of its context).
  environment: function () {
    const sc = PLATE3D.scene;
    try {
      const c = document.createElement('canvas');
      c.width = c.height = 64;
      const g = c.getContext('2d');
      const grd = g.createLinearGradient(0, 0, 0, 64);
      grd.addColorStop(0, '#f4f4f4'); grd.addColorStop(0.5, '#9a9a9a'); grd.addColorStop(1, '#2a2a2a');
      g.fillStyle = grd;
      g.fillRect(0, 0, 64, 64);
      const tex = new THREE.CanvasTexture(c);
      tex.mapping = THREE.EquirectangularReflectionMapping;
      if (sc.environment) sc.environment.dispose();
      const pm = new THREE.PMREMGenerator(PLATE3D.r);
      sc.environment = pm.fromEquirectangular(tex).texture;
      pm.dispose();
      tex.dispose();
    } catch (e) { sc.environment = null; }     // lit without reflections
  },

  // the GLB: its JSON, its binary chunk, a BufferGeometry per primitive with its material's colour and roughness
  load: function () {
    const x = new XMLHttpRequest();
    x.open('GET', 'ui/models/charuco_plate.glb', true);
    x.responseType = 'arraybuffer';
    x.onload = function () {
      try { PLATE3D.build(x.response); } catch (e) { console.warn('plate model: ' + e); }
    };
    x.send();
  },

  build: function (buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== GLB_MAGIC) throw new Error('not a GLB');
    const jl = dv.getUint32(12, true);
    const doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jl)));
    const bin = 28 + jl;
    // an accessor's numbers, copied out of the buffer (a copy is aligned whatever its offset), of the kind it says
    const arr = function (i, comps) {
      const a = doc.accessors[i], v = doc.bufferViews[a.bufferView], Type = GLB_TYPES[a.componentType];
      if (!Type || (v.byteStride && v.byteStride !== comps * Type.BYTES_PER_ELEMENT)) throw new Error('accessor ' + i);
      const start = bin + (v.byteOffset || 0) + (a.byteOffset || 0);
      return new Type(buf.slice(start, start + a.count * comps * Type.BYTES_PER_ELEMENT));
    };
    const group = new THREE.Group();
    doc.meshes.forEach(function (m) {
      m.primitives.forEach(function (pr) {
        const mat = doc.materials[pr.material].pbrMetallicRoughness;
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(arr(pr.attributes.POSITION, 3), 3));
        if (pr.indices !== undefined) g.setIndex(new THREE.BufferAttribute(arr(pr.indices, 1), 1));
        g.computeVertexNormals();
        const c = mat.baseColorFactor;
        const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(c[0], c[1], c[2]), roughness: mat.roughnessFactor,
                                                          metalness: 0, flatShading: true, envMapIntensity: 0.35 });
        group.add(new THREE.Mesh(g, material));
      });
    });
    PLATE3D.root.add(group);
    PLATE3D.model = group;
    PLATE3D.measures(((doc.scenes[0] || {}).extras || {}).measure || {});
    PLATE3D.size();
    PLATE3D.draw();
  },

  // the two lengths the caliper reads: a line between the ticks' outer faces with a short bar at each end
  measures: function (m) {
    const accent = new THREE.Color(PLATE3D.tok('--accent')), plain = new THREE.Color(PLATE3D.tok('--text-2'));
    const lines = new THREE.Group();
    ['long', 'short'].forEach(function (k) {
      if (!m[k]) return;
      const a = new THREE.Vector3().fromArray(m[k].from), b = new THREE.Vector3().fromArray(m[k].to);
      const along = b.clone().sub(a).normalize(), up = new THREE.Vector3(0, 1, 0);
      const side = new THREE.Vector3().crossVectors(along, up).multiplyScalar(0.0045);
      const pts = [a, b, a.clone().add(side), a.clone().sub(side), b.clone().add(side), b.clone().sub(side)];
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      g.setIndex([0, 1, 2, 3, 4, 5]);
      const line = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: k === 'long' ? accent : plain,
                                                                          depthTest: false, transparent: true }));
      line.renderOrder = 2;
      line.userData.which = k;
      lines.add(line);
    });
    PLATE3D.lines = lines;
    PLATE3D.root.add(lines);
  },

  size: function () {
    const box = $('bd3d'), r = PLATE3D.r;
    if (!r || !box.clientWidth || !box.clientHeight) return;
    r.setSize(box.clientWidth, box.clientHeight, false);
    PLATE3D.cam.aspect = box.clientWidth / box.clientHeight;
    // the first view holds the whole plate with a margin, in a box of any shape: its half diagonal (0.155 m) within
    // the narrower of the two fields of view
    const half = 15 * Math.PI / 180, wide = Math.atan(Math.tan(half) * PLATE3D.cam.aspect);
    PLATE3D.home.dist = 0.175 / Math.sin(Math.min(half, wide));
    PLATE3D.cam.updateProjectionMatrix();
  },

  draw: function () {
    if (!PLATE3D.r || PLATE3D.raf) return;
    PLATE3D.raf = requestAnimationFrame(function () {
      PLATE3D.raf = 0;
      const v = PLATE3D.view, cam = PLATE3D.cam;
      cam.position.set(v.dist * Math.cos(v.el) * Math.sin(v.az), v.dist * Math.sin(v.el), v.dist * Math.cos(v.el) * Math.cos(v.az));
      cam.lookAt(0, 0, 0);
      if (PLATE3D.lines) {
        PLATE3D.lines.children.forEach(function (l) { l.visible = l.userData.which === 'long' || PLATE3D.short; });
      }
      PLATE3D.r.render(PLATE3D.scene, cam);
    });
  },

  // drag turns it, the wheel brings it closer, a double click puts it back
  wire: function (canvas) {
    let drag = null;
    canvas.addEventListener('pointerdown', function (e) {
      drag = { x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', function (e) {
      if (!drag) return;
      const v = PLATE3D.view;
      PLATE3D.turned = true;
      v.az -= (e.clientX - drag.x) * 0.008;
      v.el = Math.max(0.12, Math.min(1.52, v.el + (e.clientY - drag.y) * 0.008));
      drag = { x: e.clientX, y: e.clientY };
      PLATE3D.draw();
    });
    const end = function () { drag = null; };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault();
      const v = PLATE3D.view;
      PLATE3D.turned = true;
      v.dist = Math.max(0.18, Math.min(1.6, v.dist * Math.exp(e.deltaY * 0.0012)));
      PLATE3D.draw();
    }, { passive: false });
    canvas.addEventListener('dblclick', function () {
      PLATE3D.turned = false;
      Object.assign(PLATE3D.view, PLATE3D.home);
      PLATE3D.draw();
    });
  },

  show: function (info) {
    PLATE3D.info = info;
    PLATE3D.short = $('calBdShort').classList.contains('open');
    const first = !PLATE3D.r;
    if (!PLATE3D.init()) { $('bd3d').classList.add('hidden'); return; }
    PLATE3D.size();
    if (first || !PLATE3D.turned) Object.assign(PLATE3D.view, PLATE3D.home);
    PLATE3D.draw();
  },

  hide: function () { /* nothing runs while it is hidden: it draws only on a change */ },

  // The page goes away: the GPU's memory is given back (geometries, materials, the reflections, the context).
  dispose: function () {
    if (!PLATE3D.r) return;
    if (PLATE3D.raf) cancelAnimationFrame(PLATE3D.raf);
    PLATE3D.root.traverse(function (o) {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    if (PLATE3D.scene.environment) PLATE3D.scene.environment.dispose();
    PLATE3D.r.dispose();
    PLATE3D.r = null;
    PLATE3D.failed = true;
  },
};
