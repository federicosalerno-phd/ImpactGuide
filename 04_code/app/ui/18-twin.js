// ImpactGuide · the twin of the bench: the bench in 3D as a trial has it.
//
// The model is the bench scene as Blender has it, made again by the backend once the scene rests (twin.py,
// twin_export.py): a GLB whose nodes the page moves. It shows the trial in progress (opened from the trial's page)
// or a trial with its scans (Scans and results), step by step (a scroller, or the arrows): the holder mounted (A or
// B), the pendulum at its direction, raised to its hole (hanging on the cap at the moment of the impact), the arm
// where the step is taken (the scanner's view for a scan; for the video its sweep from end stop to end stop, the
// cables made between the scene's own four), the drape while it is on. After the scans (the steps after the
// impact) the marker moves as the scanner measured
// it, done over from 1 to 20 times on a slider (its place before the impact stays as a faint copy), and the seven
// targets of the mandible take the colour of their error on the scale of the figures, blue (small) to yellow orange
// (large), that keeps apart for those who confuse red and green; the face and the cap turn see-through, so that the
// mandible shows. It opens on the whole bench; ready views: the phone at its four positions (its calibrated field of
// view; the keypad's 0 is the phone where the arm is), the scanner, from above; the viewer of 17-plate3d.js is
// navigated as Blender's viewport. It is lit as the scene is in Blender: its suns with soft shadows, the world's
// even light and, made once after the first frame, its reflections for the metals; no tone mapping (the scene's view
// is Standard), the world's colour behind. The panel says when the synthetic dataset and the network were made on
// another version of the bench.
'use strict';

// The sign of the twin: a cube in three faces, on SlantUI's 24 grid and stroke.
ICONS.cube = { s: '<path d="M12 3.2l7.6 4.2v9.2L12 20.8l-7.6-4.2V7.4z"/>'
  + '<path d="M4.4 7.4L12 11.6l7.6-4.2M12 11.6v9.2"/>' };

const TWIN = {
  viewer: null, model: null, extras: null, url: '', failed: false,
  state: null,          // the backend's last answer (twin_state)
  mode: 'trials',       // opened from Trials (the trial in progress) or from Scans and results
  pick: '',             // in Scans and results, the trial chosen
  view: '',             // the ready view on show (phone0..3, scanner, top), '' once turned by hand
  armAz: 40,            // the arm position shown (camera azimuth, degrees)
  step: '',             // the step the trial is at (its key)
  steps: [],            // the keys of the trial's steps
  now: 0,               // the step the trial is at
  trialId: '',
  entries: [],          // what the steps scroll through: [step, pose] (the release twice: raised, then the impact)
  entry: 0,             // the one on show
  sweeping: 0,          // the video's sweep running (its frame request)
  SWEEP_HOLD_S: 0.6,    // the arm a moment at each end of the sweep
  SWEEP_SHADOW_MS: 90,  // its shadows drawn again at most this often
  DECAL: /layout|_ink$|^print_|_print$|_mark$|_logo|engraving/,   // materials of markings laid on a surface
  blend: null,          // the cables between the arm's positions (false: their meshes do not allow it)
  sweepMesh: null,
  shadowKey: '', targetsKey: '',
  warmed: '',           // the model made ready off screen
  gain: 1,              // the marker's motion enlarged this many times
  parts: {},            // node name -> {obj, q0, p0}
  ghost: null, points: null,
  loadMs: 0, openMs: 0, t0: 0,
  pending: null,        // a view asked for before the model was there
  GAIN_MAX: 20,
  RETRY_MS: 2000,       // a GL context refused: asked for again this often
  SHADOW_PX: 4096,      // the shadow map of the brightest sun (the others half as fine)
  SHADOW_NORMAL_MM: 1.0,  // shadows taken off a surface by this much, against acne
  SHADOW_SOFT: 3,       // texels the edge of a shadow is spread over (Blender's suns are 8 degrees wide)
  lights: [],
  retry: 0,
  TRE_TOP: 4,           // mm at the top of the colour scale (beyond: its last colour)
  TARGET_MM: 8,         // a target's disc, across
  GHOST_MM: 0.5,        // the place before the impact shows once the marker is drawn this far from it
  SEE_THROUGH: 0.35,    // the face and the cap after the scans, so that the mandible and its targets show
  SKIN: ['silicone_face', 'fabric_surgical_green'],      // their materials
  // the scale of the targets: the one of the figures (FIG.scale, 12-figures.js: a shortened plasma in OKLCH, deep
  // blue to yellow orange, its lightness rising evenly, apart for eyes that confuse red and green); without it the
  // same colours from plasma's own stops
  STOPS: ['#2c0594', '#5302a3', '#7a02a8', '#9c179e', '#bb3488', '#d35171', '#e66c5c', '#f48849', '#fca636',
          '#fdc02a'],
  fig: null,

  // ── the colour of an error ─────────────────────────────────────────────────
  scale: function (t) {
    if (typeof FIG !== 'undefined' && FIG.SCALE && FIG.scale) {
      if (!TWIN.fig) TWIN.fig = FIG.scale();
      return TWIN.fig(t);
    }
    t = Math.max(0, Math.min(1, isFinite(t) ? t : 0)) * (TWIN.STOPS.length - 1);
    const i = Math.min(TWIN.STOPS.length - 2, Math.floor(t)), f = t - i;
    const a = TWIN.STOPS[i], b = TWIN.STOPS[i + 1];
    const c = function (h, k) { return parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16); };
    return '#' + [0, 1, 2].map(function (k) {
      return ('0' + Math.round(c(a, k) + (c(b, k) - c(a, k)) * f).toString(16)).slice(-2);
    }).join('');
  },
  colour: function (mm) { return TWIN.scale(mm / TWIN.TRE_TOP); },

  // ── opening ────────────────────────────────────────────────────────────────
  showing: function () {
    const v = document.querySelector('.view[data-view="twin"]');
    return !!v && v.classList.contains('show');
  },

  // the viewer made, the model asked for: at idle after the start, or at the first opening
  prepare: function () {
    if (TWIN.viewer || TWIN.failed) return !!TWIN.viewer;
    // Blender's axes (Z up, the front view along +Y); the keypad's 0 is the phone where the arm is; a turn by hand
    // leaves the ready view
    TWIN.viewer = VIEW3D.create($('tw3d'), $('tw3dCanvas'), {
      margin: 1.04, dir: [0.42, -0.78, 0.46], up: [0, 0, 1], front: [0, 1, 0], failed: TWIN.fail,
      name: function (n) { return PARTS.name(n); }, material: function (m) { return PARTS.material(m); }, mm: 1,
      camera: function () { TWIN.look('phone' + TWIN.armIndex(TWIN.armAz)); },
      changed: function () { if (TWIN.view) { TWIN.view = ''; TWIN.marks(); } } });
    if (!TWIN.viewer) { TWIN.failed = true; return false; }
    const v = TWIN.viewer;
    // lit by the scene's own lamps and world (TWIN.light), not by the lights that go with the eye
    v.hemi.intensity = 0;
    v.cam.children.forEach(function (l) { if (l.isLight) l.intensity = 0; });
    // its context kept while it is off screen: the reflections are made once for it
    v.sleep = function () {};
    // a context made (the first, or again after the GPU took it away): shadows, the world's colour behind
    v.ready = function () {
      const had = v.r, ok = VIEW3D.proto.ready.call(v);
      if (ok && v.r !== had) TWIN.setup(v.r);
      return ok;
    };
    v.background = function () { v.viewColor = TWIN.backdrop(); if (v.r) v.r.setClearColor(v.viewColor, 1); v.gizmoDraw(); };
    v.background();
    window.addEventListener('pagehide', function () { if (TWIN.viewer) TWIN.viewer.dispose(); TWIN.viewer = null; });
    return true;
  },

  // no GL context to be had (WebGL off, or taken away for a while after the GPU's process was lost): said, and
  // asked for again every few seconds while the twin is on show
  fail: function () {
    TWIN.message(T('twin.no_gl'));
    clearTimeout(TWIN.retry);
    TWIN.retry = setTimeout(function again() {
      if (!TWIN.viewer || !TWIN.showing()) return;
      TWIN.viewer.draw();
      TWIN.retry = setTimeout(function () {
        if (TWIN.viewer && TWIN.viewer.r) TWIN.message(TWIN.url ? '' : $('twMsg').textContent);
        else TWIN.retry = setTimeout(again, TWIN.RETRY_MS);
      }, 200);
    }, TWIN.RETRY_MS);
  },

  // opened from the trial's page (mode trials) or as the Scans step of Scans and results
  open: function (mode) {
    TWIN.mode = mode === 'results' ? 'results' : 'trials';
    TWIN.t0 = performance.now();
    TWIN.openMs = 0;
    showView('twin');
    showPanel('twin');
    // the way back: to the trial, or to the scans it was opened from (ui/22-scans.js)
    const back = $('twBack').querySelector('.btn-label');
    back.setAttribute('data-t', TWIN.mode === 'results' ? 'twin.back.scans' : 'twin.back');
    back.textContent = T(back.getAttribute('data-t'));
    $('twPick').classList.toggle('hidden', TWIN.mode !== 'results');
    $('hint').textContent = '';
    if (TWIN.prepare()) {
      TWIN.viewer.background();
      TWIN.viewer.draw();
    }
    TWIN.view = '';
    TWIN.refresh(true);
  },

  // the backend's state of the twin; first: choose the view of the trial (the phone where the step is taken)
  refresh: function (first) {
    askQuiet('twin_state', JSON.stringify({ mode: TWIN.mode, trial: TWIN.mode === 'results' ? TWIN.pick : '' }),
      function (s, bad) {
        if (!s) { if (TWIN.showing()) failed(bad); return; }
        TWIN.render(s, first);
      });
  },

  // ── the model ──────────────────────────────────────────────────────────────
  load: function (url, then) {
    if (!TWIN.prepare()) return;
    const t0 = performance.now();
    TWIN.url = url;
    TWIN.fetch(url, function (res) {
      if (!TWIN.viewer || TWIN.url !== url) return;
      TWIN.model = res.object;
      TWIN.extras = res.extras || {};
      TWIN.parts = {};
      res.object.traverse(function (o) {
        if (o.name && TWIN.extras.parts && TWIN.extras.parts[o.name]) {
          TWIN.parts[o.name] = { obj: o, q0: o.quaternion.clone(), p0: o.position.clone() };
        }
      });
      TWIN.ghost = TWIN.makeGhost();
      TWIN.points = new THREE.Group();
      TWIN.points.userData.noPick = true;
      TWIN.points.renderOrder = 10;
      res.object.add(TWIN.ghost, TWIN.points);
      TWIN.blend = null; TWIN.sweepMesh = null; TWIN.shadowKey = ''; TWIN.targetsKey = ''; TWIN.warmed = '';
      TWIN.viewer.moved = false;
      TWIN.viewer.set(res.object);
      TWIN.light();
      if (TWIN.viewer.r) TWIN.setup(TWIN.viewer.r);
      // not on screen yet: made ready off screen when the app is idle, so that the first view does not wait
      if (!TWIN.showing()) TWIN.idle(TWIN.warm);
      TWIN.loadMs = performance.now() - t0;
      TWIN.apply();
      if (then) then();
      else if (TWIN.pending && TWIN.showing()) { TWIN.look(TWIN.pending); TWIN.opened(); }   // opened while it loaded
    }, function (e) {
      console.warn('twin model: ' + e);
      TWIN.url = '';
      TWIN.message(T('twin.model.unreadable'));
    });
  },

  // ── the file: a GLB of quantized meshes (twin_export.py), its binary chunk compressed ─────────────────────────
  fetch: function (url, done, failed) {
    const x = new XMLHttpRequest();
    x.open('GET', url, true);
    x.responseType = 'arraybuffer';
    x.onload = function () {
      let p = null;
      try { p = TWIN.read(x.response); } catch (e) { failed(e); return; }
      p.then(done, failed);
    };
    x.onerror = function () { failed(new Error(url)); };
    x.send();
  },

  // the GLB read into three.js: {object, extras} once its binary chunk is inflated
  read: function (buf) {
    const dv = new DataView(buf);
    if (buf.byteLength < 28 || dv.getUint32(0, true) !== 0x46546C67) throw new Error('not a GLB');
    const jl = dv.getUint32(12, true);
    const doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jl)));
    const info = (doc.asset && doc.asset.extras) || {};
    const packed = new Uint8Array(buf, 28 + jl, info.packedLength || dv.getUint32(20 + jl, true));
    const bin = info.compression === 'zlib'
      ? new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer()
      : Promise.resolve(packed.slice().buffer);
    return bin.then(function (b) { return TWIN.build(doc, b); });
  },

  build: function (doc, bin) {
    const view = function (i, Type, comps) {
      const a = doc.accessors[i], v = doc.bufferViews[a.bufferView];
      const per = v.byteStride ? v.byteStride / Type.BYTES_PER_ELEMENT : comps;
      return { data: new Type(bin, v.byteOffset + (a.byteOffset || 0), a.count * per), per: per, count: a.count };
    };
    const mats = (doc.materials || []).map(function (m) {
      const p = m.pbrMetallicRoughness || {}, c = p.baseColorFactor || [1, 1, 1, 1], e = m.emissiveFactor || [0, 0, 0];
      const st = ((m.extensions || {}).KHR_materials_emissive_strength || {}).emissiveStrength || 1;
      const opts = { name: m.name || '', color: new THREE.Color(c[0], c[1], c[2]),
        opacity: c[3], transparent: c[3] < 1, metalness: p.metallicFactor === undefined ? 0 : p.metallicFactor,
        roughness: p.roughnessFactor === undefined ? 0.5 : p.roughnessFactor, emissive: new THREE.Color(e[0], e[1], e[2]),
        emissiveIntensity: st, side: THREE.DoubleSide };
      // a cloth's sheen (the cap) as Blender's: the physical material, for that one
      const sh = m.extras && m.extras.sheen;
      // the markings printed or laid on a surface (the board's layout, inks, engravings, logos) drawn a little nearer
      // than the surface they lie on: never flickering through it in specks
      if (TWIN.DECAL.test(opts.name)) {
        Object.assign(opts, { polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4 });
      }
      if (sh) return new THREE.MeshPhysicalMaterial(Object.assign(opts, { sheen: new THREE.Color(sh[0], sh[1], sh[2]) }));
      return new THREE.MeshStandardMaterial(opts);
    });
    const TYPES = { 5120: Int8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
    // an attribute of three numbers as floats: quantized ones (16 bits over the box of the primitive, 8 bits for a
    // normal) brought back, plain ones copied
    const floats = function (i, centre, half) {
      const a = doc.accessors[i], v = view(i, TYPES[a.componentType], 3), out = new Float32Array(v.count * 3);
      const q = a.componentType === 5122 ? 32767 : a.componentType === 5120 ? 127 : 1;
      const c = centre || [0, 0, 0], h = half || [1, 1, 1];
      const k = [h[0] / q, h[1] / q, h[2] / q];
      for (let j = 0, s = 0, d = 0; j < v.count; j++, s += v.per, d += 3) {
        out[d] = c[0] + v.data[s] * k[0]; out[d + 1] = c[1] + v.data[s + 1] * k[1]; out[d + 2] = c[2] + v.data[s + 2] * k[2];
      }
      return out;
    };
    const meshes = (doc.meshes || []).map(function (m) {
      return m.primitives.map(function (pr) {
        const ex = pr.extras || {};
        const g = new THREE.BufferGeometry();
        const pa = doc.accessors[pr.attributes.POSITION], na = doc.accessors[pr.attributes.NORMAL];
        // quantized (16 bits, the same step on the three axes; normals of 8 bits): drawn as they are, the mesh scaled
        // and moved to their box (no copy into floats, no work on the page's thread); plain floats as they are
        const quant = pa.componentType === 5122 && ex.half && ex.half[0] === ex.half[1] && ex.half[1] === ex.half[2];
        if (quant) {
          const pv = view(pr.attributes.POSITION, Int16Array, 4), nv = view(pr.attributes.NORMAL, Int8Array, 4);
          g.setAttribute('position', new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(pv.data, pv.per), 3, 0, false));
          g.setAttribute('normal', new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(nv.data, nv.per), 3, 0,
                                                                        na.componentType !== 5126));
        } else {
          g.setAttribute('position', new THREE.BufferAttribute(floats(pr.attributes.POSITION, ex.centre, ex.half), 3));
          g.setAttribute('normal', new THREE.BufferAttribute(floats(pr.attributes.NORMAL), 3));
        }
        const ia = doc.accessors[pr.indices], idx = view(pr.indices, TYPES[ia.componentType], 1);
        g.setIndex(new THREE.BufferAttribute(idx.data, 1));
        const mesh = new THREE.Mesh(g, mats[pr.material]);
        if (quant) { mesh.position.fromArray(ex.centre); mesh.scale.setScalar(ex.half[0] / 32767); }
        mesh.castShadow = mesh.receiveShadow = true;
        if (ex.objects) mesh.userData.objects = ex.objects;      // its parts, for a click to pick one
        return mesh;
      });
    });
    const node = function (i) {
      const d = doc.nodes[i], o = new THREE.Group();
      o.name = d.name || '';
      if (d.matrix) new THREE.Matrix4().fromArray(d.matrix).decompose(o.position, o.quaternion, o.scale);
      if (d.mesh !== undefined) meshes[d.mesh].forEach(function (m) { o.add(m); });
      (d.children || []).forEach(function (k) { o.add(node(k)); });
      return o;
    };
    const sc = doc.scenes[doc.scene || 0], group = new THREE.Group();
    sc.nodes.forEach(function (i) { group.add(node(i)); });
    return { object: group, extras: sc.extras || {} };
  },

  // ── the look: as the scene is lit in Blender ───────────────────────────────────────────────────────────────
  // the scene's suns (Blender's strength in W/m2 is pi times three's intensity), each casting a soft shadow from
  // around the bench; the world's light as an even light all round (the hemisphere until the reflections are made,
  // then the reflections themselves)
  light: function () {
    const v = TWIN.viewer, x = TWIN.extras || {};
    (TWIN.lights || []).forEach(function (l) {
      v.scene.remove(l, l.target);
      if (l.shadow && l.shadow.map) { l.shadow.map.dispose(); l.shadow.map = null; }
    });
    TWIN.lights = [];
    const b = x.bounds || [[-500, -500, -300], [500, 500, 400]];
    const c = new THREE.Vector3().fromArray(b[0]).add(new THREE.Vector3().fromArray(b[1])).multiplyScalar(0.5);
    const R = new THREE.Vector3().fromArray(b[1]).sub(new THREE.Vector3().fromArray(b[0])).length() / 2;
    const suns = (x.lamps || []).filter(function (l) { return l.type === 'SUN'; });
    const top = suns.reduce(function (m, l) { return Math.max(m, l.energy); }, 0);
    suns.forEach(function (l) {
      const d = new THREE.DirectionalLight(new THREE.Color(l.color[0], l.color[1], l.color[2]), l.energy / Math.PI);
      d.position.copy(c).addScaledVector(new THREE.Vector3().fromArray(l.dir).normalize(), -2 * R);
      d.target.position.copy(c);
      d.castShadow = true;
      const s = d.shadow, cam = s.camera;
      cam.left = cam.bottom = -R; cam.right = cam.top = R; cam.near = R * 0.5; cam.far = R * 3.5;
      cam.updateProjectionMatrix();
      const size = l.energy === top ? TWIN.SHADOW_PX : TWIN.SHADOW_PX / 2;
      s.mapSize.set(size, size);
      s.bias = -0.0004;
      s.normalBias = TWIN.SHADOW_NORMAL_MM;
      s.radius = TWIN.SHADOW_SOFT;
      v.scene.add(d, d.target);
      TWIN.lights.push(d);
    });
    const w = (x.world && x.world.light) || [0.3, 0.3, 0.3];
    v.hemi.color.setRGB(w[0], w[1], w[2]);
    v.hemi.groundColor.setRGB(w[0], w[1], w[2]);
    v.hemi.intensity = v.scene.environment ? 0 : 1;
  },

  // what the camera sees behind the bench in Blender (the world for camera rays)
  backdrop: function () {
    const w = (TWIN.extras && TWIN.extras.world && TWIN.extras.world.camera) || [1, 1, 1];
    return new THREE.Color(w[0], w[1], w[2]).convertLinearToSRGB();
  },

  // a renderer as the twin wants it: soft shadows, made again only when something moved; no tone mapping (the
  // scene's view transform is Standard); its reflections made once, after its first frame
  setup: function (r, quiet) {
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap;           // filtered over SHADOW_SOFT texels: a soft edge, no grain
    r.shadowMap.autoUpdate = false;
    r.shadowMap.needsUpdate = true;
    r.toneMapping = THREE.NoToneMapping;
    if (quiet) return;
    const v = TWIN.viewer;
    v.background();
    if (v.scene.environment) { v.scene.environment.dispose(); v.scene.environment = null; }
    v.hemi.intensity = 1;
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { TWIN.idle(function () { TWIN.reflections(r); }); });
    });
  },

  // the reflections of an even world (the world's light all round, as Blender's), for the metals; made once per
  // context
  envOf: function (r) {
    const pm = new THREE.PMREMGenerator(r), sc = new THREE.Scene(), w = (TWIN.extras.world || {}).light || [0.3, 0.3, 0.3];
    // the world as a sphere around the eye in its colour (linear light: the cube is drawn without encoding); a
    // solid background colour would go through an sRGB conversion of three's own and light too brightly
    const ball = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 8),
                                new THREE.MeshBasicMaterial({ color: new THREE.Color(w[0], w[1], w[2]), side: THREE.BackSide }));
    sc.add(ball);
    const t = pm.fromScene(sc, 0, 0.1, 10).texture;
    pm.dispose();
    ball.geometry.dispose();
    ball.material.dispose();
    return t;
  },
  reflections: function (r) {
    const v = TWIN.viewer;
    if (!v || v.r !== r || v.scene.environment || !TWIN.extras) return;
    v.scene.environment = TWIN.envOf(r);
    v.hemi.intensity = 0;
    v.draw();
  },

  // made ready while the app is idle, before it is first shown: its own context (kept while it is off screen) at
  // the size its box will have, the reflections made, the shaders compiled with them and with the shadows, the
  // model sent to the GPU and the shadows drawn, in two short steps; the trial's page then opens it with a frame
  // like any other
  warm: function () {
    const v = TWIN.viewer;
    if (!v || !TWIN.model || TWIN.showing() || TWIN.warmed === TWIN.url || !v.ready()) return;
    const r = v.r, view = v.box.closest('.view'), stage = view && view.parentElement;
    if (!v.w && stage && stage.clientWidth) {
      // the box as it will be: the stage less the view's padding
      const cs = getComputedStyle(view);
      v.w = Math.max(1, stage.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight));
      v.h = Math.max(1, stage.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom));
      r.setSize(v.w, v.h, false);
      v.cam.aspect = v.w / v.h;
      v.cam.updateProjectionMatrix();
    }
    const url = TWIN.url;
    try {
      if (!v.scene.environment) { v.scene.environment = TWIN.envOf(r); v.hemi.intensity = 0; }
      TWIN.cablesReady();             // the cables of the video's sweep, compiled with the rest
      r.compile(v.scene, v.eye);
      r.compile(VIEW3D.kit().quad.scene, VIEW3D.kit().quad.cam);
    } catch (e) { console.warn('twin warm: ' + e); return; }
    TWIN.idle(function () {
      if (TWIN.url !== url || TWIN.showing() || v.r !== r) return;
      try {
        v.place(v.shown);
        r.shadowMap.needsUpdate = true;
        // (the sweep's cables drawn once too, off screen, their shadows' shaders with them)
        const sweep = TWIN.sweepMesh, was = sweep && sweep.visible;
        if (sweep) sweep.visible = true;
        r.render(v.scene, v.eye);
        if (sweep) { sweep.visible = was; r.shadowMap.needsUpdate = true; }
        // the points the first view is framed by, kept, and the framing run a few times (its first runs are slow
        // ones, before the page's engine compiles it)
        const pts = v.pointsFlat();
        for (let k = 0; k < 3; k++) v.fitPoints(v.goal.q, pts, v.centre.clone(), v.goal.dist);
        TWIN.warmed = url;
      } catch (e) { console.warn('twin warm: ' + e); }
    });
  },

  // the marker as it stood before the impact: a faint copy of the parts that move with it
  makeGhost: function () {
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#d6a531';
    const g = new THREE.Group(), m = new THREE.MeshBasicMaterial({ color: new THREE.Color(accent), transparent: true,
                                                                   opacity: 0.3, depthWrite: false });
    Object.keys(TWIN.parts).forEach(function (n) {
      const spec = TWIN.extras.parts[n];
      if (spec.moves !== 'marker') return;
      const holder = new THREE.Group();
      holder.userData.holder = spec.holder || '';
      TWIN.parts[n].obj.children.forEach(function (c) {
        if (!c.isMesh) return;
        const k = new THREE.Mesh(c.geometry, m);
        k.position.copy(c.position); k.quaternion.copy(c.quaternion); k.scale.copy(c.scale);
        holder.add(k);
      });
      g.add(holder);
    });
    g.visible = false;
    g.userData.noPick = true;
    return g;
  },

  message: function (text) {
    $('twMsg').textContent = text || '';
    $('twMsg').classList.toggle('hidden', !text);
  },

  // ── the bench as the trial has it ──────────────────────────────────────────
  armIndex: function (az) {
    const a = TWIN.extras && TWIN.extras.arm;
    if (!a || !a.camera_deg) return 0;
    let k = 0;
    a.camera_deg.forEach(function (d, i) { if (Math.abs(d - az) < Math.abs(a.camera_deg[k] - az)) k = i; });
    return k;
  },

  turn: function (name, deg) {
    const p = TWIN.parts[name], spec = TWIN.extras.parts[name];
    if (!p || !spec) return;
    const ax = new THREE.Vector3().fromArray(spec.axis || [0, 0, 1]).normalize();
    const q = new THREE.Quaternion().setFromAxisAngle(ax, (spec.sign || 1) * deg * Math.PI / 180);
    p.obj.quaternion.copy(p.q0);
    if (spec.frame === 'world') p.obj.quaternion.premultiply(q); else p.obj.quaternion.multiply(q);
  },

  // D of the scans (the marker's motion in the world, 4 x 4 by rows) done k times over, k continuous: the screw
  // motion D^k, that turns k times as far about the same axis and slides k times as far along it. A turn about the
  // A1 ball keeps the ball where it is however large k is.
  motion: function (D, k) {
    const m = new THREE.Matrix4();
    if (!D) return m;
    const R = new THREE.Matrix4().set(D[0][0], D[0][1], D[0][2], 0, D[1][0], D[1][1], D[1][2], 0,
                                      D[2][0], D[2][1], D[2][2], 0, 0, 0, 0, 1);
    const t = new THREE.Vector3(D[0][3], D[1][3], D[2][3]);
    const q = new THREE.Quaternion().setFromRotationMatrix(R);
    if (q.w < 0) { q.x = -q.x; q.y = -q.y; q.z = -q.z; q.w = -q.w; }
    const s = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z), th = 2 * Math.atan2(s, q.w);
    if (s < 1e-9 || th < 1e-9) return m.makeTranslation(k * t.x, k * t.y, k * t.z);
    const u = new THREE.Vector3(q.x / s, q.y / s, q.z / s);
    const along = u.clone().multiplyScalar(u.dot(t)), across = t.clone().sub(along);
    // a point of the axis: (I - R) p = the translation across it
    const p = across.clone().add(u.clone().cross(across).multiplyScalar(1 / Math.tan(th / 2))).multiplyScalar(0.5);
    const qk = new THREE.Quaternion().setFromAxisAngle(u, k * th);
    m.makeRotationFromQuaternion(qk);
    m.setPosition(p.clone().sub(p.clone().applyQuaternion(qk)).addScaledVector(along, k));
    return m;
  },

  // the bench as the step on show has it: the arm's position (a camera azimuth, degrees; null for the video's
  // sweep), the pendulum raised to its hole or hanging on the cap (the moment of the impact), the drape on or off,
  // the marker where the scans found it (any step after the impact)
  pose: function () {
    const s = TWIN.state || {}, t = s.trial || {}, cfg = s.arm || {};
    const keys = TWIN.steps, e = TWIN.entries[TWIN.entry] || [0, ''];
    const key = keys[e[0]] || '', at = function (k) { return keys.indexOf(k); };
    const m = /_az(\d+)$/.exec(key), photo = cfg.photo_az || [40, 55, 70, 85];
    const on = at('drape_on'), off = at('drape_off'), rel = at('release');
    return { key: key, sub: e[1],
             az: m ? Number(m[1]) : /video$/.test(key) ? null : photo[0],
             swing: key === 'release' && e[1] === 'impact' ? 0 : Number(t.angle_deg) || 0,
             drape: !!t.drape && on >= 0 && e[0] >= on && (off < 0 || e[0] < off),
             after: rel < 0 || e[0] > rel,
             scan: /scan_az\d+$/.test(key) };
  },

  apply: function () {
    const s = TWIN.state, x = TWIN.extras;
    if (!TWIN.model || !x || !s) return;
    const t = s.trial || { holder: 'A', angle_deg: 0, direction_deg: 0 };
    const pose = TWIN.pose();
    const scan = pose.after && s.scan && s.scan.state === 'ready' ? s.scan : null;
    const M = TWIN.motion(scan && scan.D, TWIN.gain);
    // what follows the pendulum without being carried: the mesh of its direction (the arm's: TWIN.arm)
    const dirs = Object.keys(x.parts).filter(function (n) { return x.parts[n].pose === 'pendulum'; })
      .map(function (n) { return x.parts[n].value; });
    const dir = dirs.reduce(function (m, d) {
      return Math.abs(d - (Number(t.direction_deg) || 0)) < Math.abs(m - (Number(t.direction_deg) || 0)) ? d : m;
    }, dirs.length ? dirs[0] : 0);
    Object.keys(TWIN.parts).forEach(function (n) {
      const p = TWIN.parts[n], spec = x.parts[n];
      if (spec.pose === 'arm') return;
      p.obj.visible = !spec.holder || spec.holder === t.holder;
      if (spec.pose === 'pendulum') p.obj.visible = spec.value === dir;
      if (spec.drape) p.obj.visible = pose.drape;
      if (spec.moves === 'marker') M.decompose(p.obj.position, p.obj.quaternion, p.obj.scale);
    });
    TWIN.turn('pendulum', Number(t.direction_deg) || 0);
    TWIN.turn('pendulum_swing', pose.swing);
    // the place before the impact, while the marker is seen elsewhere
    const moved = !!scan && scan.shift_mm * TWIN.gain > TWIN.GHOST_MM;
    TWIN.ghost.visible = moved;
    TWIN.ghost.children.forEach(function (g) { g.visible = !g.userData.holder || g.userData.holder === t.holder; });
    const tk = scan ? JSON.stringify(scan.tre || []) : '';
    if (tk !== TWIN.targetsKey) { TWIN.targetsKey = tk; TWIN.marksTargets(scan); }
    TWIN.skin(!!scan);
    TWIN.shadowState = [t.holder, dir, t.direction_deg, pose.swing, pose.drape, TWIN.gain, tk.length, !!scan].join();
    TWIN.arm();
  },

  // the arm at TWIN.armAz (its four positions as the scene has them, any other between them and on beyond the
  // last ones, for the video's sweep), the cables with it; the shadows drawn again only when something moved
  arm: function () {
    const x = TWIN.extras, a = x && x.arm, v = TWIN.viewer;
    if (a && a.arm_deg && a.camera_deg && TWIN.parts.arm) {
      const deg = TWIN.armDeg(TWIN.armAz);
      TWIN.turn('arm', deg - a.shown_deg);
      TWIN.cables(deg);
    }
    // (during the sweep at most every SWEEP_SHADOW_MS: two maps of the whole bench each time)
    const key = (TWIN.shadowState || '') + '|' + TWIN.armAz.toFixed(3), now = performance.now();
    if (key !== TWIN.shadowKey && v.r && (!TWIN.sweeping || now - (TWIN.shadowAt || 0) > TWIN.SWEEP_SHADOW_MS)) {
      TWIN.shadowKey = key; TWIN.shadowAt = now; v.r.shadowMap.needsUpdate = true;
    }
    v.draw();
  },

  armDeg: function (az) {
    const a = TWIN.extras.arm, c = a.camera_deg, d = a.arm_deg;
    if (c.length < 2) return d[0];
    let k = 0;
    while (k < c.length - 2 && az > c[k + 1]) k++;
    return d[k] + (d[k + 1] - d[k]) * (az - c[k]) / (c[k + 1] - c[k]);
  },

  // the cables, which follow the arm without being carried by it: the scene's own mesh at each of the arm's
  // positions; between and beyond them (the sweep) made from the two nearest, each point turned about the arm's
  // axis by its share of the turn between them (its distance from the axis and its height mixed alike), so that
  // the end held on the arm goes round with it and the end held on the bench stays
  cables: function (deg) {
    const x = TWIN.extras;
    const list = Object.keys(TWIN.parts).filter(function (n) { return x.parts[n].pose === 'arm'; })
      .map(function (n) { return { obj: TWIN.parts[n].obj, value: x.parts[n].value }; })
      .sort(function (p, q) { return p.value - q.value; });
    if (!list.length) return;
    let near = list[0];
    list.forEach(function (p) { if (Math.abs(p.value - deg) < Math.abs(near.value - deg)) near = p; });
    const exact = Math.abs(near.value - deg) < 0.05;
    const b = exact || list.length < 2 ? null : TWIN.blendOf(list);
    list.forEach(function (p) { p.obj.visible = !b && p === near; });
    if (TWIN.sweepMesh) TWIN.sweepMesh.visible = !!b;
    if (!b) return;
    let k = 0;
    while (k < list.length - 2 && deg > list[k + 1].value) k++;
    b.fill(k, (deg - list[k].value) / (list[k + 1].value - list[k].value));
  },

  // made once (at idle, before the twin is first shown): for each point of the cables, its distance from the arm's
  // axis, its turn about it and its height at each of the arm's positions, and its normal, as attributes; the GPU
  // mixes them for the arm where it is (a weight for each position), so that the sweep sends four numbers a frame
  blendOf: function (list) {
    if (TWIN.blend !== null) return TWIN.blend || null;
    TWIN.blend = false;
    const meshes = list.map(function (p) { return p.obj.children.filter(function (c) { return c.isMesh; }); });
    const n0 = meshes[0];
    const same = n0.length && meshes.length <= 4 && meshes.every(function (m) {
      return m.length === n0.length && m.every(function (c, j) {
        return c.geometry.attributes.position.count === n0[j].geometry.attributes.position.count
          && c.geometry.index.count === n0[j].geometry.index.count;
      });
    });
    if (!same) return null;
    // the arm's axis in the model: through its origin, along its own axis
    TWIN.model.updateMatrixWorld(true);
    const inv = new THREE.Matrix4().copy(TWIN.model.matrixWorld).invert();
    const am = new THREE.Matrix4().multiplyMatrices(inv, TWIN.parts.arm.obj.matrixWorld);
    const c = new THREE.Vector3().setFromMatrixPosition(am);
    const u = new THREE.Vector3().fromArray(TWIN.extras.parts.arm.axis || [0, 0, 1]).transformDirection(am);
    const e1 = new THREE.Vector3(Math.abs(u.x) < 0.9 ? 1 : 0, Math.abs(u.x) < 0.9 ? 0 : 1, 0);
    e1.addScaledVector(u, -e1.dot(u)).normalize();
    const e2 = new THREE.Vector3().crossVectors(u, e1);
    const U = { poseW: { value: new THREE.Vector4(1, 0, 0, 0) }, swC: { value: c }, swU: { value: u },
                swE1: { value: e1 }, swE2: { value: e2 } };
    const head = 'attribute vec4 th4; attribute vec4 r4; attribute vec4 h4; attribute vec3 n0; attribute vec3 n1;'
      + ' attribute vec3 n2; attribute vec3 n3; uniform vec4 poseW; uniform vec3 swC; uniform vec3 swU;'
      + ' uniform vec3 swE1; uniform vec3 swE2;\n';
    const morph = function (m) {
      m.customProgramCacheKey = function () { return 'twin-sweep'; };
      m.onBeforeCompile = function (s) {
        Object.assign(s.uniforms, U);
        s.vertexShader = head + s.vertexShader
          .replace('#include <beginnormal_vertex>',
                   'vec3 objectNormal = normalize(poseW.x * n0 + poseW.y * n1 + poseW.z * n2 + poseW.w * n3);')
          .replace('#include <begin_vertex>', 'float swTh = dot(poseW, th4), swR = dot(poseW, r4), swH = dot(poseW, h4);'
                   + ' vec3 transformed = swC + swR * cos(swTh) * swE1 + swR * sin(swTh) * swE2 + swH * swU;');
      };
      return m;
    };
    const group = new THREE.Group();
    group.name = 'arm_sweep';
    group.visible = false;
    n0.forEach(function (m0, j) {
      const n = m0.geometry.attributes.position.count, p = new THREE.Vector3(), q = new THREE.Vector3();
      const th4 = new Float32Array(4 * n), r4 = new Float32Array(4 * n), h4 = new Float32Array(4 * n);
      const nk = [0, 1, 2, 3].map(function () { return new Float32Array(3 * n); });
      const pos0 = new Float32Array(3 * n);
      meshes.forEach(function (ms, k) {
        const m = ms[j], M = new THREE.Matrix4().multiplyMatrices(inv, m.matrixWorld);
        const N = new THREE.Matrix3().getNormalMatrix(M);
        const pa = m.geometry.attributes.position, na = m.geometry.attributes.normal;
        const nq = na.normalized && na.isInterleavedBufferAttribute ? 127 : 1;
        for (let i = 0; i < n; i++) {
          p.fromBufferAttribute(pa, i).applyMatrix4(M);
          if (k === 0) { pos0[3 * i] = p.x; pos0[3 * i + 1] = p.y; pos0[3 * i + 2] = p.z; }
          p.sub(c);
          const a = p.dot(e1), b = p.dot(e2);
          let th = Math.atan2(b, a);
          // the turn followed on from the position before (never the long way round)
          if (k) { let d = th - th4[4 * i + k - 1]; d -= 2 * Math.PI * Math.round(d / (2 * Math.PI)); th = th4[4 * i + k - 1] + d; }
          th4[4 * i + k] = th; r4[4 * i + k] = Math.hypot(a, b); h4[4 * i + k] = p.dot(u);
          q.fromBufferAttribute(na, i).multiplyScalar(1 / nq).applyMatrix3(N).normalize();
          nk[k][3 * i] = q.x; nk[k][3 * i + 1] = q.y; nk[k][3 * i + 2] = q.z;
        }
      });
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos0, 3));
      g.setAttribute('normal', new THREE.BufferAttribute(nk[0], 3));
      g.setAttribute('th4', new THREE.BufferAttribute(th4, 4));
      g.setAttribute('r4', new THREE.BufferAttribute(r4, 4));
      g.setAttribute('h4', new THREE.BufferAttribute(h4, 4));
      nk.forEach(function (a, k) { g.setAttribute('n' + k, new THREE.BufferAttribute(a, 3)); });
      g.setIndex(m0.geometry.index);
      const mesh = new THREE.Mesh(g, morph(m0.material.clone()));
      mesh.customDepthMaterial = morph(new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking }));
      mesh.castShadow = mesh.receiveShadow = true;
      mesh.frustumCulled = false;
      mesh.userData.noPick = true;       // where it is is the GPU's: not looked into by a click
      group.add(mesh);
    });
    TWIN.model.add(group);
    TWIN.sweepMesh = group;
    TWIN.blend = {
      // the arm between positions k and k + 1 (w of the way; beyond the ends, w below 0 or above 1)
      fill: function (k, w) {
        const v = [0, 0, 0, 0];
        v[k] = 1 - w; v[k + 1] = w;
        U.poseW.value.set(v[0], v[1], v[2], v[3]);
      },
    };
    return TWIN.blend;
  },

  // the cables between the arm's positions made ready (at idle): their attributes, their shaders with the rest
  cablesReady: function () {
    const x = TWIN.extras;
    if (!x || TWIN.blend !== null) return;
    const list = Object.keys(TWIN.parts).filter(function (n) { return x.parts[n].pose === 'arm'; })
      .map(function (n) { return { obj: TWIN.parts[n].obj, value: x.parts[n].value }; })
      .sort(function (p, q) { return p.value - q.value; });
    if (list.length > 1) TWIN.blendOf(list);
  },

  // the video's sweep: the arm from one end stop to the other and back, at the video's speed, a moment at each end;
  // a view from the phone goes with it. It runs while that step is on show and the twin is on screen
  sweep: function (on) {
    if (TWIN.sweeping) { cancelAnimationFrame(TWIN.sweeping); TWIN.sweeping = 0; }
    if (!on || !TWIN.model) return;
    const cfg = (TWIN.state && TWIN.state.arm) || {}, ends = cfg.video_az || [35, 95];
    const run = Math.abs(ends[1] - ends[0]) / (cfg.video_deg_s || 8), hold = TWIN.SWEEP_HOLD_S, period = 2 * (run + hold);
    const t0 = performance.now();
    const tick = function (now) {
      TWIN.sweeping = 0;
      if (!TWIN.showing() || TWIN.pose().az !== null) return;
      const u = ((now - t0) / 1000) % period;
      const f = u < run ? u / run : u < run + hold ? 1 : u < 2 * run + hold ? 1 - (u - run - hold) / run : 0;
      TWIN.armAz = ends[0] + (ends[1] - ends[0]) * VIEW3D.ease(f);
      TWIN.arm();
      if (/^phone\d$/.test(TWIN.view)) TWIN.look('phone' + TWIN.armIndex(TWIN.armAz), true);
      TWIN.sweeping = requestAnimationFrame(tick);
    };
    TWIN.sweeping = requestAnimationFrame(tick);
  },

  // the seven targets on the mandible, drawn over everything (they are inside the face, which turns see-through),
  // each a disc of the colour of its error with a light rim, TARGET_MM across
  marksTargets: function (scan) {
    const g = TWIN.points;
    g.children.slice().forEach(function (c) {
      g.remove(c);
      if (c.material.map) c.material.map.dispose();
      c.material.dispose();
    });
    if (!scan || !TWIN.extras.targets) return;
    const by = {};
    (scan.tre || []).forEach(function (e) { by[e.id] = e.mm; });
    TWIN.extras.targets.forEach(function (tg) {
      if (by[tg.id] === undefined) return;
      const cv = document.createElement('canvas');
      cv.width = cv.height = 64;
      const c = cv.getContext('2d');
      c.beginPath(); c.arc(32, 32, 31, 0, 2 * Math.PI); c.fillStyle = '#1b1b1f'; c.fill();
      c.beginPath(); c.arc(32, 32, 27, 0, 2 * Math.PI); c.fillStyle = '#ffffff'; c.fill();
      c.beginPath(); c.arc(32, 32, 22, 0, 2 * Math.PI); c.fillStyle = TWIN.colour(by[tg.id]); c.fill();
      const tex = new THREE.CanvasTexture(cv);
      tex.encoding = THREE.sRGBEncoding;
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, depthWrite: false,
                                                             transparent: true }));
      sp.position.fromArray(tg.world);
      sp.scale.set(TWIN.TARGET_MM, TWIN.TARGET_MM, 1);
      sp.renderOrder = 10;
      sp.userData.id = tg.id;
      g.add(sp);
    });
  },

  // the face and the cap see-through (after the scans) or solid
  skin: function (through) {
    TWIN.model.traverse(function (o) {
      const m = o.isMesh && o.material;
      if (!m || TWIN.SKIN.indexOf(m.name) < 0 || m.userData.through === through) return;
      m.userData.through = through;
      m.transparent = through;
      m.opacity = through ? TWIN.SEE_THROUGH : 1;
      m.depthWrite = !through;
      m.needsUpdate = true;
      o.castShadow = !through;
    });
  },

  // ── the ready views ────────────────────────────────────────────────────────
  // name: phone0..phone3 (the arm positions), scanner (from the arm where it stands), top; follow: the phone
  // where the arm is now (the sweep), at once
  look: function (name, follow) {
    const v = TWIN.viewer, x = TWIN.extras;
    if (!v || !TWIN.model || !x) { TWIN.pending = name; return; }
    TWIN.pending = null;
    if (name === 'first') { TWIN.first(); return; }
    v.size();
    const box = v.w && v.h ? v.w / v.h : 1.5;
    let eye, q, fov, dist;
    if (name === 'top') {
      const b = x.bounds || [[-400, -400, -300], [400, 400, 400]];
      const c = new THREE.Vector3((b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2, (b[0][2] + b[1][2]) / 2);
      const half = Math.max((b[1][1] - b[0][1]) / 2, (b[1][0] - b[0][0]) / 2 / box) * 1.04;
      fov = VIEW3D.FOV;
      const h = half / Math.tan(fov / 2 * Math.PI / 180);
      eye = new THREE.Vector3(c.x, c.y, c.z + h);         // the bench fills the view at its middle height
      q = new THREE.Quaternion();       // looking straight down, the back of the bench up the screen
      dist = eye.z - c.z;
    } else {
      const cam = name === 'scanner' ? x.cameras && x.cameras.scanner : x.cameras && x.cameras.phone;
      if (!cam) return;
      if (!follow) {
        if (name !== 'scanner') TWIN.armAz = x.arm.camera_deg[Number(name.slice(5)) || 0];
        TWIN.apply();
      }
      TWIN.model.updateMatrixWorld(true);
      const node = TWIN.parts[cam.node] ? TWIN.parts[cam.node].obj : TWIN.model;
      const C = cam.matrix;
      const Mc = new THREE.Matrix4().set(C[0][0], C[0][1], C[0][2], C[0][3], C[1][0], C[1][1], C[1][2], C[1][3],
                                         C[2][0], C[2][1], C[2][2], C[2][3], 0, 0, 0, 1);
      const W = new THREE.Matrix4().multiplyMatrices(node.matrixWorld, Mc);
      eye = new THREE.Vector3(); q = new THREE.Quaternion();
      W.decompose(eye, q, new THREE.Vector3());
      // the phone's own field (its calibration), else the camera of the scene; all of the picture in the box
      const ph = name === 'scanner' ? null : TWIN.state && TWIN.state.phone;
      const fy = ph ? ph.fov_y : cam.fov_y, aspect = ph ? ph.aspect : cam.aspect || 4 / 3;
      fov = box >= aspect ? fy : 2 * Math.atan(Math.tan(fy / 2 * Math.PI / 180) * aspect / box) * 180 / Math.PI;
      const focus = new THREE.Vector3().fromArray(x.target || x.board_centre || [0, 0, 0]);
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(q);
      dist = Math.max(50, focus.sub(eye).dot(fwd));
    }
    v.lookFrom(eye, q, dist, fov);
    if (follow) { v.shown.q.copy(v.goal.q); v.shown.dist = v.goal.dist; v.shown.at.copy(v.goal.at); }
    if (TWIN.view !== name) { TWIN.view = name; TWIN.marks(); }
  },

  // the view a trial opens with: the whole bench framed, turning about its centre, the bench as the step on show
  // has it
  first: function () {
    const v = TWIN.viewer, p = TWIN.pose();
    TWIN.armAz = TWIN.azOf(p);
    TWIN.apply();
    v.setOrtho(false);
    v.moved = false;
    v.frameAll(true);
    TWIN.view = '';
    TWIN.marks();
    TWIN.stepMarks();
    TWIN.sweep(p.az === null);
  },

  // the arm's position of a pose (a camera azimuth): the video's starts at its first end stop
  azOf: function (p) {
    const cfg = (TWIN.state && TWIN.state.arm) || {};
    return p.az === null ? (cfg.video_az || [35])[0] : p.az;
  },

  // ── the steps of the trial ─────────────────────────────────────────────────
  // as the backend gives them; what the scroller goes through: each step, the release twice (the pendulum raised
  // to its hole, then hanging on the cap at the impact)
  setSteps: function (s) {
    const keys = s.steps || [];
    TWIN.steps = keys;
    TWIN.entries = [];
    keys.forEach(function (k, i) {
      if (k === 'release') TWIN.entries.push([i, 'raised'], [i, 'impact']);
      else TWIN.entries.push([i, '']);
    });
    TWIN.now = Math.max(0, Math.min(keys.length - 1, Number(s.step) || 0));
  },
  entryOf: function (i) {
    return Math.max(0, TWIN.entries.findIndex(function (e) { return e[0] === i; }));
  },

  // a step on show: the bench put as it has it; a view from the phone or the scanner goes to the one of the step
  // (the phone where its photo is taken, the scanner for a scan)
  goEntry: function (i) {
    if (!TWIN.entries.length) return;
    TWIN.entry = Math.max(0, Math.min(TWIN.entries.length - 1, i));
    const p = TWIN.pose();
    TWIN.armAz = TWIN.azOf(p);
    TWIN.stepMarks();
    if (!TWIN.model) return;
    if (/^(phone\d|scanner)$/.test(TWIN.view)) TWIN.look(p.scan ? 'scanner' : 'phone' + TWIN.armIndex(TWIN.armAz));
    else TWIN.apply();
    TWIN.sweep(p.az === null);
  },

  // the scroller: the step on show, of how many, its name (the release: before it, or the impact); the step the
  // trial is at marked; the buttons off at the ends
  stepMarks: function () {
    const has = TWIN.entries.length > 0;
    $('twSteps').classList.toggle('hidden', !has);
    if (!has) return;
    const e = TWIN.entries[TWIN.entry] || TWIN.entries[0], key = TWIN.steps[e[0]] || '';
    let name = typeof TRIALS !== 'undefined' ? TRIALS.stepTitle({ key: key }) : key;
    if (e[1]) name += ' · ' + T('twin.pose.' + e[1]);
    $('twStepN').textContent = T('trials.now.n', e[0] + 1, TWIN.steps.length);
    $('twStepT').textContent = name;
    $('twStepT').title = name;
    const t = TWIN.state && TWIN.state.trial;
    $('twStepNow').classList.toggle('hidden', !(TWIN.mode === 'trials' && t && t.status === 'open' && e[0] === TWIN.now));
    $('twPrev').disabled = TWIN.entry <= 0;
    $('twNext').disabled = TWIN.entry >= TWIN.entries.length - 1;
  },

  // ── the panel ──────────────────────────────────────────────────────────────
  render: function (s, first) {
    const was = [TWIN.trialId, TWIN.steps.join(), TWIN.now].join('|');
    TWIN.state = s;
    const m = s.model || {}, t = s.trial;
    // the steps: on a new trial, or when it went on to another step, the one it is at
    TWIN.setSteps(s);
    TWIN.trialId = t ? t.trial : '';
    const moved = was !== [TWIN.trialId, TWIN.steps.join(), TWIN.now].join('|');
    if (first || moved) TWIN.entry = TWIN.entryOf(TWIN.now);
    // the model: none yet (being made), unreadable, or the one to show
    const fresh = !!m.url && m.url !== TWIN.url;
    if (fresh) {
      TWIN.message('');
      TWIN.load(m.url, function () {
        TWIN.look(TWIN.pending || (first || !TWIN.view ? 'first' : TWIN.view));
        TWIN.opened();
      });
    } else if (!m.url) {
      TWIN.message(T(m.state === 'no_blender' ? 'twin.model.no_blender' : 'twin.model.making'));
    } else {
      TWIN.message('');
    }
    if (!fresh && first && m.url) {
      if (TWIN.model) { TWIN.first(); TWIN.opened(); } else TWIN.pending = 'first';
    } else if (!fresh && !first && moved && TWIN.model) {
      TWIN.goEntry(TWIN.entry);       // the trial went on while the twin is on show: the bench as its new step has it
    }
    if (t) TWIN.step = t.step;
    TWIN.stepMarks();
    // the head: the trial and its setting
    $('twSub').textContent = t ? t.trial : T('twin.trial.none');
    $('where').textContent = t ? [T('trials.tile.holder') + ' ' + t.holder,
                                  T('trials.tile.hole') + ' ' + fmtNum(t.angle_deg, 0) + '°',
                                  T('trials.tile.direction') + ' ' + fmtNum(t.direction_deg, 0) + '°']
      .concat(t.drape ? [T('trials.chip.drape')] : []).join(' · ') : '';
    // dataset and network made on another version of the bench
    const stale = $('twStale'), say = TWIN.staleKey(s.versions || {});
    stale.textContent = say ? T(say) : '';
    stale.classList.toggle('hidden', !say);
    // Scans and results: the trial to look at
    if (TWIN.mode === 'results') {
      const list = s.trials || [];
      setOptions($('twTrial'), list.map(function (r) {
        return r.trial + ' · ' + r.holder + ' · ' + r.angle_deg + '° · ' + r.direction_deg + '°';
      }), list.map(function (r) { return r.trial; }));
      if (t) $('twTrial').value = t.trial;
      $('twPick').classList.toggle('hidden', !list.length);
    }
    TWIN.views();
    TWIN.scanPanel(s.scan);
    TWIN.modelRows(m);
    TWIN.apply();
  },

  // the words of dataset and network made on another version of the bench ('' when both are of this one or none)
  staleKey: function (v) {
    if (!v || !v.stale) return '';
    if (v.dataset === 'old' && v.network === 'old') return 'twin.stale.both';
    return v.dataset === 'old' ? 'twin.stale.data' : 'twin.stale.net';
  },

  // the first frame of an opening drawn: how long it took
  opened: function () {
    if (!TWIN.t0) return;
    const t0 = TWIN.t0;
    TWIN.t0 = 0;
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { TWIN.openMs = performance.now() - t0; });
    });
  },

  views: function () {
    const host = $('twViews'), x = TWIN.extras;
    const az = x && x.arm && x.arm.camera_deg ? x.arm.camera_deg : [40, 55, 70, 85];
    const want = az.map(function (a, i) { return ['phone' + i, T('twin.view.phone', fmtNum(a, 0))]; })
      .concat([['scanner', T('twin.view.scanner')], ['top', T('twin.view.top')]]);
    if (host.children.length !== want.length || host.dataset.lang !== LANG + az.join()) {
      host.replaceChildren();
      host.dataset.lang = LANG + az.join();
      want.forEach(function (w) {
        const b = E('button', 'btn btn-2 btn-sm');
        b.type = 'button';
        b.dataset.view = w[0];
        b.append(E('span', 'btn-label', w[1]));
        b.onclick = function () { TWIN.look(w[0]); };
        host.append(b);
      });
    }
    TWIN.marks();
  },

  marks: function () {
    document.querySelectorAll('#twViews .btn').forEach(function (b) {
      b.classList.toggle('on', b.dataset.view === TWIN.view);
    });
  },

  scanPanel: function (scan) {
    const ready = !!scan && scan.state === 'ready';
    $('twScan').classList.toggle('hidden', !ready);
    const note = $('twScanNote');
    let say = '';
    if (!scan || scan.state === 'none' || scan.state === 'no_model') say = 'twin.scan.none';
    else if (scan.state === 'measuring') say = 'twin.scan.measuring';
    else if (scan.state === 'failed') {
      say = hasT('twin.scan.failed.' + scan.why) ? 'twin.scan.failed.' + scan.why : 'twin.scan.failed';
    }
    note.textContent = say ? T(say) : '';
    note.classList.toggle('hidden', !say);
    $('twSim').classList.toggle('hidden', !(ready && scan.simulated));
    // measured with free radii: the spheres' radii are not calibrated yet (ground_truth.py calibrate)
    $('twNoCal').classList.toggle('hidden', !(ready && !scan.simulated && scan.calibrated === false));
    if (!ready) return;
    $('twScanRows').replaceChildren(
      piRow(T('twin.scan.shift'), withUnit(scan.shift_mm, 2, 'mm') + ' · ' + fmtNum(scan.rot_deg, 2) + '°'),
      piRow(T('twin.scan.tre'), scan.tre_max_mm === null ? '-' : withUnit(scan.tre_max_mm, 2, 'mm')));
    $('twGain').value = String(TWIN.gain);
    TWIN.gainLabel();
    // the scale with its values, then each target in its colour
    const n = 12, stops = [];
    for (let i = 0; i <= n; i++) stops.push(TWIN.scale(i / n) + ' ' + (100 * i / n).toFixed(1) + '%');
    $('twRamp').style.background = 'linear-gradient(90deg, ' + stops.join(', ') + ')';
    const ticks = $('twTicks');
    ticks.replaceChildren();
    for (let mm = 0; mm <= TWIN.TRE_TOP; mm++) {
      ticks.append(E('span', null, mm === TWIN.TRE_TOP ? '≥ ' + mm + ' mm' : fmtNum(mm, 0)));
    }
    const list = $('twTargets');
    list.replaceChildren();
    (scan.tre || []).forEach(function (e) {
      const row = E('div', 'tw-tg');
      const dot = E('span', 'tw-dot');
      dot.style.background = TWIN.colour(e.mm);
      row.append(dot, E('span', 'tw-tg-n', e.id), E('span', 'tw-tg-v', withUnit(e.mm, 2, 'mm')));
      list.append(row);
    });
  },

  gainLabel: function () { $('twGainV').textContent = '× ' + fmtNum(TWIN.gain, TWIN.gain < 10 ? 1 : 0); },

  modelRows: function (m) {
    const rows = $('twModelRows');
    rows.replaceChildren();
    if (m.version) {
      rows.append(piRow(T('twin.adv.model'), 'v' + m.version + ' · ' + String(m.phone_print || '').slice(0, 6)),
                  piRow(T('twin.adv.saved'), fmtTime(m.scene_saved)),
                  piRow(T('twin.adv.size'), withUnit((m.bytes || 0) / 1048576, 1, 'MB')));
      if (TWIN.loadMs) rows.append(piRow(T('twin.adv.load'), withUnit(TWIN.loadMs / 1000, 2, 's')));
    }
    if (m.state === 'exporting') rows.append(E('div', 'note', T('twin.model.updating')));
    $('twUpdate').disabled = m.state === 'exporting' || m.state === 'no_blender';
  },

  // ── events ─────────────────────────────────────────────────────────────────
  onEvent: function (e) {
    if (e.type === 'twin') {
      if (e.action === 'updated') {
        const stale = TWIN.staleKey(e);
        toast(T('twin.updated') + (stale ? ' · ' + T(stale) : ''), stale ? TOAST_LONG_MS : TOAST_SAID_MS);
        if (TWIN.showing() || TWIN.url) TWIN.refresh(false);
      } else if (e.action === 'failed') {
        if (TWIN.showing()) { toast(T('twin.model.failed'), TOAST_ERROR_MS); TWIN.refresh(false); }
      } else if (TWIN.showing()) {
        TWIN.refresh(false);        // exporting, the same model, the scans measured
      }
      return;
    }
    // a file of the trial in progress: the arm, the scans
    if (e.type === 'staged' && TWIN.showing() && TWIN.mode === 'trials') TWIN.refresh(false);
  },

  // what the tests read
  probe: function () {
    const x = TWIN.extras, v = TWIN.viewer, p = function (n) { return TWIN.parts[n] && TWIN.parts[n].obj; };
    const mk = p('marker');
    return {
      loaded: !!TWIN.model, url: TWIN.url, loadMs: Math.round(TWIN.loadMs), openMs: Math.round(TWIN.openMs),
      view: TWIN.view, armAz: TWIN.armAz, gain: TWIN.gain, mode: TWIN.mode,
      holderA: !!(p('holder_a') && p('holder_a').visible), holderB: !!(p('holder_b') && p('holder_b').visible),
      marker: mk ? mk.position.toArray().map(function (a) { return Math.round(a * 1000) / 1000; }) : null,
      ghost: !!(TWIN.ghost && TWIN.ghost.visible), points: TWIN.points ? TWIN.points.children.length : 0,
      eye: v ? v.goal.at.clone().add(new THREE.Vector3(0, 0, v.goal.dist).applyQuaternion(v.goal.q)).toArray()
        .map(Math.round) : null,
      fov: v ? Math.round(v.cam.fov * 10) / 10 : null, nodes: x ? Object.keys(TWIN.parts).sort() : [],
      stale: !$('twStale').classList.contains('hidden'), drawn: !!(v && v.r),
      noGl: !$('twMsg').classList.contains('hidden') && $('twMsg').textContent === T('twin.no_gl'),
      steps: TWIN.steps.length, entries: TWIN.entries.length, entry: TWIN.entry, now: TWIN.now,
      pose: TWIN.pose(), swing: p('pendulum_swing') ? Math.round(2 * Math.atan2(Math.hypot(
        p('pendulum_swing').quaternion.x, p('pendulum_swing').quaternion.y, p('pendulum_swing').quaternion.z),
        Math.abs(p('pendulum_swing').quaternion.w)) * 1800 / Math.PI) / 10 : null,
      drape: !!(p('drape') && p('drape').visible), sweeping: !!TWIN.sweeping,
      sweepCables: !!(TWIN.sweepMesh && TWIN.sweepMesh.visible), stepTitle: $('twStepT').textContent,
    };
  },

  wire: function () {
    $('twBack').onclick = function () { if (TWIN.mode === 'results') go('results', 'scans'); else go('trials', 'guide'); };
    $('twUpdate').onclick = function () {
      ask('twin_update', function (a) { if (a) { toast(T('twin.model.updating')); TWIN.modelRows(a.model || {}); } });
    };
    $('twTrial').onchange = function () { TWIN.pick = $('twTrial').value; TWIN.refresh(true); };
    const g = $('twGain');
    g.max = String(TWIN.GAIN_MAX);
    g.oninput = function () {
      TWIN.gain = Math.max(1, Math.min(TWIN.GAIN_MAX, Number(g.value) || 1));
      TWIN.gainLabel();
      TWIN.apply();
    };
    $('gTwin').onclick = function () { TWIN.open('trials'); };
    $('twPrev').onclick = function () { TWIN.goEntry(TWIN.entry - 1); };
    $('twNext').onclick = function () { TWIN.goEntry(TWIN.entry + 1); };
    // the arrows go through the steps, as Blender's go through the frames
    window.addEventListener('keydown', function (e) {
      if (!TWIN.showing() || e.altKey || e.ctrlKey || e.metaKey || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return;
      const a = document.activeElement;
      if (a && (/^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName) || a.isContentEditable)) return;
      e.preventDefault();
      TWIN.goEntry(TWIN.entry + (e.key === 'ArrowRight' ? 1 : -1));
    });
    if (TWIN.state) {
      TWIN.views(); TWIN.scanPanel(TWIN.state.scan); TWIN.modelRows(TWIN.state.model || {}); TWIN.stepMarks();
    }
  },
};

// A ready view: the eye at a point, turned by q (looking along its -z, its y up), turning about the point dist ahead,
// with a field of view of its own (degrees). Turned by hand afterwards as any view.
VIEW3D.proto.lookFrom = function (eye, q, dist, fov) {
  const v = this;
  v.setOrtho(false);
  if (fov) { v.cam.fov = fov; v.cam.updateProjectionMatrix(); }
  v.goal.q.copy(q);
  v.goal.dist = dist;
  v.goal.at.copy(new THREE.Vector3(0, 0, -dist).applyQuaternion(q).add(eye));
  v.moved = true;
  v.draw();
};

// made ready while the app is idle, well after the plate: the trial's page then opens it at once
TWIN.idle = function (fn) { (window.requestIdleCallback || function (f) { return setTimeout(f, 0); })(fn, { timeout: 4000 }); };
window.addEventListener('load', function () {
  setTimeout(function () {
    TWIN.idle(function () {
      if (TWIN.url || TWIN.failed || !hasSlot('twin_state')) return;
      askQuiet('twin_state', JSON.stringify({ mode: 'trials' }), function (s) {
        if (s && s.model && s.model.url && !TWIN.url) { TWIN.state = s; TWIN.load(s.model.url); }
      });
    });
  }, 6000);
});
