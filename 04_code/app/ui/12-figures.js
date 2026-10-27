// ImpactGuide · the figures of the calibration, drawn as SVG in the palette's
// roles: the guide of each photo (the phone's screen with the board as it
// should look there, the board's top, left and right edges each in its own
// colour, curved arrows when the phone is turned in its plane, and the phone
// seen from above and from the side, at its distance and tilt, with its camera
// where the Poco X7 Pro has it and the board edge on in the colour of the edge
// seen; with a physical board the phone stays still and the board is drawn
// moved and turned instead), the photo just taken with the corners found in it
// and where the board has been in all the photos kept, with the four corners
// of the photo reached or not, the error of each photo (bars in the
// colour of their error, a photo left out in grey, one press puts it in or
// out), where on the board the error is (a dot per corner, larger and lighter
// where the error is larger, blue to yellow orange on the scale of this
// series), and
// the A4 sheet with what to measure. Every arrowhead is SlantUI's one filled
// triangle (head, arrowHead). Plain functions of the data the backend gives; a figure
// never holds state of its own.
'use strict';

/** The calibration's figures, each a function of the backend's data that fills a host element. */
const FIG = {
  NS: 'http://www.w3.org/2000/svg',

  // The Poco X7 Pro (mm): its length and width; its rear camera seen through the phone held across with its top to
  // the left: the module (a pill) from `pill[0]` to `pill[1]` of the length from the top edge and from `pill[2]` to
  // `pill[3]` of the width from the upper long side, the main lens at `lens` (length, width) in the same way.
  POCO: { len: 160.75, wid: 75.24, thick: 8.3, pill: [5.8, 49.1, 5.6, 27.6], lens: [16.4, 16.7] },

  el: function (tag, attrs, parent) {
    const e = document.createElementNS(FIG.NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(e);
    return e;
  },
  svg: function (w, h, cls) {
    return FIG.el('svg', { viewBox: '0 0 ' + w + ' ' + h, class: 'fig-svg' + (cls ? ' ' + cls : ''),
                           preserveAspectRatio: 'xMidYMid meet' });
  },
  text: function (parent, x, y, s, cls, anchor) {
    const t = FIG.el('text', { x: x, y: y, class: cls || 'fig-t', 'text-anchor': anchor || 'start' }, parent);
    t.textContent = s;
    return t;
  },
  title: function (node, s) { FIG.el('title', {}, node).textContent = s; },
  poly: function (pts) { return pts.map(function (p) { return p[0].toFixed(2) + ',' + p[1].toFixed(2); }).join(' '); },
  line: function (parent, a, b, cls) {
    return FIG.el('line', { x1: a[0].toFixed(2), y1: a[1].toFixed(2), x2: b[0].toFixed(2), y2: b[1].toFixed(2), class: cls },
                  parent);
  },

  // ── the arrowhead: SlantUI's one filled, slender isosceles triangle (arrowHead, ARROW_HEAD: 0.9 to 2.6, the shape of
  // the dimension arrow over the board on this monitor) ─
  // tip: its point; dir: the way it points; len: its length. Its base is where the line it ends goes in.
  head: function (parent, tip, dir, len, cls) {
    return FIG.el('polygon', { points: FIG.poly(arrowHead(tip, dir, len)), class: cls }, parent);
  },

  // ── a colour scale from the best to the worst, in OKLCH (a perceptual space) ─
  // Sequential, like a shortened "plasma": deep blue (the smallest error) through violet and magenta to orange and
  // a yellow orange (the largest), the lightness rising evenly from 0.45 to 0.88, so that the worst stands out on the
  // dark board and every step stays apart for eyes that confuse red and green or blue and yellow (checked with the
  // Machado 2009 simulations of protanopia, deuteranopia and tritanopia: the nearest of nine steps 0.039 apart in
  // OKLab, the two ends 0.44 or more). A colour out of sRGB keeps its lightness and hue and loses chroma. scale()
  // gives scale(t), t in [0, 1], as #rrggbb, and scale().light: the page's background is light (the marks then get a
  // thin dark edge, as the yellow end is pale).
  SCALE: { L: [0.45, 0.88], H: [265, 295, 355, 405, 440], C: [0.14, 0.17, 0.19, 0.18, 0.17] },
  oklch: function (hex) {
    const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(String(hex).trim()) || [0, '80', '80', '80'];
    const lin = function (h) { const c = parseInt(h, 16) / 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
    const r = lin(m[1]), g = lin(m[2]), b = lin(m[3]);
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const mm = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    const L = 0.2104542553 * l + 0.7936177850 * mm - 0.0040720468 * s;
    const A = 1.9779984951 * l - 2.4285922050 * mm + 0.4505937099 * s;
    const B = 0.0259040371 * l + 0.7827717662 * mm - 0.8086757660 * s;
    return [L, Math.hypot(A, B), Math.atan2(B, A)];
  },
  // linear sRGB of an OKLCH colour [L, C, hue in radians]
  lrgb: function (lch) {
    const A = lch[1] * Math.cos(lch[2]), B = lch[1] * Math.sin(lch[2]);
    const l = Math.pow(lch[0] + 0.3963377774 * A + 0.2158037573 * B, 3);
    const m = Math.pow(lch[0] - 0.1055613458 * A - 0.0638541728 * B, 3);
    const s = Math.pow(lch[0] - 0.0894841775 * A - 1.2914855480 * B, 3);
    return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
  },
  hex: function (lch) {
    const inside = function (c) { return FIG.lrgb([lch[0], c, lch[2]]).every(function (v) { return v >= -1e-6 && v <= 1 + 1e-6; }); };
    let c = lch[1];
    if (!inside(c)) {
      let lo = 0, hi = c;
      for (let k = 0; k < 18; k++) { const mid = (lo + hi) / 2; if (inside(mid)) lo = mid; else hi = mid; }
      c = lo;
    }
    return '#' + FIG.lrgb([lch[0], c, lch[2]]).map(function (v) {
      v = Math.max(0, Math.min(1, v));
      v = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
      return ('0' + Math.round(v * 255).toString(16)).slice(-2);
    }).join('');
  },
  scale: function () {
    const S = FIG.SCALE, n = S.H.length - 1;
    const f = function (t) {
      t = Math.max(0, Math.min(1, isFinite(t) ? t : 0));
      const x = t * n, i = Math.min(n - 1, Math.floor(x)), u = x - i;
      const h = S.H[i] + (S.H[i + 1] - S.H[i]) * u, c = S.C[i] + (S.C[i + 1] - S.C[i]) * u;
      return FIG.hex([S.L[0] + (S.L[1] - S.L[0]) * t, c, (h % 360) * Math.PI / 180]);
    };
    f.light = FIG.oklch(getComputedStyle(document.documentElement).getPropertyValue('--surface-0'))[0] > 0.6;
    return f;
  },
  // the style of a mark in the colour of t: on a light page with a thin dark edge
  mark: function (col, t) {
    return 'fill:' + col(t) + (col.light ? ';stroke:var(--text-2);stroke-opacity:.55;stroke-width:1px;'
                                           + 'vector-effect:non-scaling-stroke' : '');
  },

  // the convex hull of points (monotone chain), counter-clockwise; fewer than three points come back as they are
  hull: function (pts) {
    const p = pts.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    if (p.length < 3) return p;
    const cross = function (o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); };
    const lo = [], up = [];
    p.forEach(function (q) {
      while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop();
      lo.push(q);
    });
    p.slice().reverse().forEach(function (q) {
      while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop();
      up.push(q);
    });
    return lo.slice(0, -1).concat(up.slice(0, -1));
  },

  // ── the guide of the next photo ─────────────────────────────────────────
  // g: calib_guide.guide (outline and black squares in fractions of the screen, yaw, pitch, roll, distance; move:
  // "board" when the phone stays still on its stand and the board is moved, else "phone")
  // the pose in words, two short lines: where from and how far round (or, the board moved, which of its edges comes
  // nearer the phone and by how much), then the distance and the turn
  caption: function (g) {
    // the bench: the pose and the arm angle, then how the ball joint is set (the turn of the board on a third line)
    if (g.view === 'bench') {
      return [T('cal.ph.bench.step', g.pose, fmtNum(g.arm, 0))].concat(FIG.benchWords(g.face, g.lean, g.turn));
    }
    const v = g.move === 'board' ? (g.view === 'front' ? T('guide.move.front') : T('guide.move.' + g.view, g.tilt))
      : g.view === 'front' ? T('guide.view.front') : T('guide.view.' + g.view) + ' ' + g.tilt + '°';
    const parts = [fmtNum(g.dist_cm, 0) + ' cm'];
    if (Math.abs(g.roll) >= 5) parts.push(T('guide.turned', Math.abs(g.roll)));
    return [v, parts.join(' · ')];
  },
  // the setting of the bench's ball joint in words: the arm angle the board faces and how far it leans back; how far
  // the board is turned in its plane ('' when not: turn > 0 anticlockwise seen from the front)
  benchWords: function (face, lean, turn) {
    const f = fmtNum(face, 0), l = Math.round(Math.abs(lean || 0)), r = Math.round(turn || 0);
    return [l ? T('cal.ph.bench.head', f, l) : T('cal.ph.bench.head.up', f),
            !r ? '' : T(r > 0 ? 'cal.ph.bench.ccw' : 'cal.ph.bench.cw', Math.abs(r))];
  },

  guide: function (host, g, big) {
    host.replaceChildren();
    if (!g) return;
    host.classList.toggle('big', !!big);
    const head = E('div', 'guide-hd'), words = FIG.caption(g);
    // a photo past the plan, for a corner of the photo the board has not reached yet: that corner by name
    head.append(E('span', 'guide-n', g.extra ? T('guide.extra', T('cal.corner.' + g.extra)) : T('guide.n', g.n + 1, g.of)),
                E('span', 'guide-w', words[0]),
                E('span', 'guide-d', words[1]));
    if (words[2]) head.append(E('span', 'guide-d', words[2]));
    // the bench: how to set the ball joint, in full on the pointer
    if (g.view === 'bench') head.title = T('cal.ph.bench.head.hint', fmtNum(g.face, 0));
    host.append(head);
    const row = E('div', 'guide-row');
    row.append(FIG.phone(g), FIG.sides(g));
    host.append(row);
    // the key: the board's edges, each in its colour (on the screen and edge on)
    const leg = E('div', 'guide-leg');
    const sw = function (cls, word) { const s = E('span', 'guide-key'); s.append(E('i', cls), E('span', null, word)); return s; };
    leg.append(E('span', 'guide-leg-l', T('guide.board')), sw('k-top', T('guide.top')), sw('k-left', T('guide.left')),
               sw('k-right', T('guide.right')));
    host.append(leg);
  },

  // The phone held across, its top to the left: a phone's shape (about 21 : 10), its screen as the camera shows it,
  // the camera's strip at the top (left) with the main camera in its corner, where it is behind the screen, the
  // photo's frame (4:3, as wide as the screen's height allows) and the shutter in the part at the bottom, kept narrow
  // (a little over half the screen's height), the board in the frame as it should be there, all its squares, with its
  // top, left and right edges in their colours. Turned in its plane (roll), two curved arrows round its ends show
  // which way: clockwise as the operator sees the screen when roll > 0. The room for them is always kept, so the
  // phone never changes size from one photo to the next. With the board moved and the phone still, the two arrows
  // sit on the board in the photo instead, the way the board is turned: the other way round.
  phone: function (g) {
    const L = 200, D = 96, pad = 1.5, M = 14, W = L + 2 * (pad + M), H = D + 2 * pad, x0 = pad + M, cy = H / 2;
    const s = FIG.svg(W, H, 'guide-phone');
    FIG.el('rect', { x: x0, y: pad, width: L, height: D, rx: 12, class: 'g-body' }, s);
    const b = 3, sx = x0 + b, sy = pad + b, sw = L - 2 * b, sh = D - 2 * b, strip = 22;
    FIG.el('rect', { x: sx, y: sy, width: sw, height: sh, rx: 8, class: 'g-screen' }, s);
    const fh = sh, fw = fh * (g.aspect || 4 / 3), fx = sx + strip, fy = sy;
    const ly = pad + D * FIG.POCO.lens[1] / FIG.POCO.wid;
    FIG.el('circle', { cx: sx + strip / 2, cy: ly, r: 3.8, class: 'g-lens' }, s);
    FIG.el('circle', { cx: sx + strip / 2, cy: ly, r: 1.5, class: 'g-lens-in' }, s);
    const kx = (fx + fw + sx + sw) / 2;
    FIG.el('circle', { cx: kx, cy: cy, r: 7.5, class: 'g-shutter' }, s);
    FIG.el('circle', { cx: kx, cy: cy, r: 5.5, class: 'g-shutter-in' }, s);
    const id = 'gclip' + Math.random().toString(36).slice(2, 8);
    const cp = FIG.el('clipPath', { id: id }, FIG.el('defs', {}, s));
    FIG.el('rect', { x: fx, y: fy, width: fw, height: fh }, cp);
    FIG.el('rect', { x: fx, y: fy, width: fw, height: fh, class: 'g-frame' }, s);
    const inner = FIG.el('g', { 'clip-path': 'url(#' + id + ')' }, s);
    const P = function (q) { return [fx + q[0] * fw, fy + q[1] * fh]; };
    FIG.el('polygon', { points: FIG.poly(g.outline.map(P)), class: 'g-board' }, inner);
    g.cells.forEach(function (c) { FIG.el('polygon', { points: FIG.poly(c.map(P)), class: 'g-cell' }, inner); });
    // the outline runs top left, top right, bottom right, bottom left: the sides first, the top over them
    const o = g.outline.map(P);
    FIG.line(inner, o[3], o[0], 'g-edge e-left');
    FIG.line(inner, o[1], o[2], 'g-edge e-right');
    FIG.line(inner, o[0], o[1], 'g-edge e-top');
    // the middle of the photo, faint: where the board's centre goes is read against it
    const mx = fx + fw / 2;
    FIG.el('path', { d: 'M' + (mx - 4) + ' ' + cy + 'h8M' + mx + ' ' + (cy - 4) + 'v8', class: 'g-mid' }, s);
    if (Math.abs(g.roll || 0) >= 5) {
      if (g.move !== 'board') FIG.turn(s, x0 + L / 2, cy, L / 2 + 10, g.roll > 0);
      else {
        // on the board: about its middle in the photo (kept inside the frame), over it with a dark edge
        const c = o.reduce(function (m, q) { return [m[0] + q[0] / 4, m[1] + q[1] / 4]; }, [0, 0]), r = fh * 0.4;
        FIG.turn(s, Math.max(fx + r + 6, Math.min(fx + fw - r - 6, c[0])), Math.max(fy + 12, Math.min(fy + fh - 12, c[1])),
                 r, g.roll < 0, true);
      }
    }
    return s;
  },

  // Two curved arrows round the ends of the phone, about its centre (cx, cy) at radius R: the turn in its plane,
  // clockwise on screen (cw) or the other way. over: drawn over the board, each with a dark edge under it.
  turn: function (s, cx, cy, R, cw, over) {
    const span = (over ? 28 : 18) * Math.PI / 180, len = 9;
    const grp = FIG.el('g', { class: 'g-turn' + (over ? ' g-turn-over' : '') }, s);
    const sg = cw ? 1 : -1;
    (over ? ['g-turn-u', ''] : ['']).forEach(function (under) {
      [0, Math.PI].forEach(function (base) {
        const from = base - sg * span, to = base + sg * span;
        const p = function (a) { return [cx + R * Math.cos(a), cy + R * Math.sin(a)]; };
        // the head's base on the arc one head's length before its end (a chord of len); the arc runs a little into
        // the head, so that line and head join with no gap
        const at = to - sg * 2 * Math.asin(len / (2 * R)), into = at + sg * 1.2 / R;
        const a = p(from), e = p(into), b = p(at), tip = p(to);
        FIG.el('path', { d: 'M' + a[0].toFixed(2) + ' ' + a[1].toFixed(2) + 'A' + R + ' ' + R + ' 0 0 ' + (cw ? 1 : 0)
                           + ' ' + e[0].toFixed(2) + ' ' + e[1].toFixed(2), class: 'g-turn-arc' + (under ? ' ' + under : '') }, grp);
        FIG.head(grp, tip, [tip[0] - b[0], tip[1] - b[1]], len, 'g-turn-head' + (under ? ' ' + under : ''));
      });
    });
  },

  // The phone seen from above (left and right) and from the side (up and down): the board edge on, as a thin strip
  // in the colour of the edge seen (from above its top edge, its left and right ends in theirs; from the side, the
  // operator's left, its left edge with its top end in the top's colour), the phone at its distance, its main lens
  // on the line of sight to the board's centre (dashed), the camera's bump towards the board where the Poco X7 Pro
  // has it: near the top edge (left, from above) and the upper long side (up, from the side). With the board moved
  // (g.move "board": the phone still on its stand) the same pose drawn from the phone: the phone square and still
  // (below the board from above, right of it from the side), the board in front of it turned by the angle, the edge
  // that comes nearer the phone towards it.
  sides: function (g) {
    const box = E('div', 'guide-sides'), still = g.move === 'board';
    // the distance on the drawing: from the nearest to the farthest the plans use
    const near = 250, far = 600;
    const d = 30 + 24 * Math.max(0, Math.min(1, (g.dist_cm * 10 - near) / (far - near)));
    // the board edge on, its two sides in their own proportion: the longer one 64 long
    const bd = g.board && g.board[0] > 0 && g.board[1] > 0 ? g.board : [4, 3], big = Math.max(bd[0], bd[1]);
    const t = 3.4, bump = 1.3, cap = 6, bw = 3.2;
    const one = function (ang, word, vertical) {
      const W = 124, H = 80, s = FIG.svg(W, H, 'guide-side');
      const a = ang * Math.PI / 180;
      // mm to the drawing: the long side 40 from above; from the side, where only the short side shows, 1.4 times that,
      // and the board a little high, as the phone's body hangs below its lens (every pose of the plans stays inside)
      const k = 40 / FIG.POCO.len * (vertical ? 1.4 : 1);
      const half = 32 * (vertical ? bd[1] : bd[0]) / big;
      let bx, by, px, py, rot, turn = 0;
      if (!vertical) {         // from above: the board along the top, the phone below it, left is left
        if (still) {           // the phone square at the bottom, the board turned in front of it, inside the drawing
          px = 58; py = 71; rot = 0; turn = ang;
          bx = px; by = py - Math.min(d, py - 2 - bw - half * Math.abs(Math.sin(a)));
        } else {
          bx = 58; by = 8;
          px = bx + d * Math.sin(a); py = by + d * Math.cos(a);
          rot = -ang;
        }
      } else {                 // from the side: the board along the left, the phone right of it, above is up
        if (still) {
          px = 108; py = 35; rot = -90; turn = ang;
          bx = px - d; by = py;
        } else {
          bx = 10; by = 35;
          px = bx + d * Math.cos(a); py = by - d * Math.sin(a);
          rot = -90 - ang;
        }
      }
      // the board edge on, about its middle: a strip in the colour of the edge seen, its ends in theirs
      const bg = FIG.el('g', { transform: 'translate(' + bx.toFixed(2) + ' ' + by.toFixed(2) + ') rotate(' + turn.toFixed(1) + ')' }, s);
      if (!vertical) {
        FIG.el('rect', { x: -half, y: -bw / 2, width: 2 * half, height: bw, class: 'e-top-f' }, bg);
        FIG.el('rect', { x: -half, y: -bw / 2, width: cap, height: bw, class: 'e-left-f' }, bg);
        FIG.el('rect', { x: half - cap, y: -bw / 2, width: cap, height: bw, class: 'e-right-f' }, bg);
      } else {
        FIG.el('rect', { x: -bw / 2, y: -half, width: bw, height: 2 * half, class: 'e-left-f' }, bg);
        FIG.el('rect', { x: -bw / 2, y: -half, width: bw, height: cap, class: 'e-top-f' }, bg);
      }
      FIG.el('line', { x1: px, y1: py, x2: bx, y2: by, class: 'g-sight' }, s);
      const ph = FIG.el('g', { transform: 'translate(' + px.toFixed(2) + ' ' + py.toFixed(2) + ') rotate(' + rot.toFixed(1) + ')' }, s);
      // edge on, about the main lens: x along the side seen (to the right from above, up from the side), the back of
      // the phone at y = 0 with the bump in front of it (towards the board), its body behind
      const Pc = FIG.POCO;
      const span = vertical ? [-(Pc.wid - Pc.lens[1]), Pc.lens[1]] : [-Pc.lens[0], Pc.len - Pc.lens[0]];
      const hump = vertical ? [Pc.lens[1] - Pc.pill[3], Pc.lens[1] - Pc.pill[2]] : [Pc.pill[0] - Pc.lens[0], Pc.pill[1] - Pc.lens[0]];
      FIG.el('rect', { x: span[0] * k, y: 0, width: (span[1] - span[0]) * k, height: t, rx: 1.2, class: 'g-slab' }, ph);
      FIG.el('rect', { x: hump[0] * k, y: -bump, width: (hump[1] - hump[0]) * k, height: bump + 0.6, rx: 0.5,
                       class: 'g-bump' }, ph);
      FIG.el('circle', { cx: 0, cy: -bump, r: 0.9, class: 'g-eye' }, ph);
      const cell = E('div', 'guide-side-c');
      cell.append(s, E('span', 'guide-side-l', word));
      return cell;
    };
    box.append(one(g.yaw, T('guide.above'), false), one(g.pitch, T('guide.side'), true));
    return box;
  },

  // ── the photo just taken: its corners over it; where the board has been in the photos kept ─
  // host: the photo's box (an <img> and an overlay); last: the series' last check (url, ok, points)
  shot: function (host, last) {
    const img = host.querySelector('img'), over = host.querySelector('.ph-shot-pts');
    const url = (last && last.url) || '';
    if (img.dataset.src !== url) {
      img.dataset.src = url;
      img.classList.add('hidden');
      img.onload = function () { img.classList.remove('hidden'); };
      img.onerror = function () { img.classList.add('hidden'); };
      if (url) img.src = url; else img.removeAttribute('src');
    }
    over.replaceChildren();
    const W = 100, H = 75, s = FIG.svg(W, H, 'shot-svg');
    s.setAttribute('preserveAspectRatio', 'none');
    // kept: a green dot per corner; to be taken again: an amber cross, so the two differ in shape as well as in colour
    (last && last.points || []).forEach(function (p) {
      const x = p[0] * W, y = p[1] * H, k = 1.5;
      if (last.ok) FIG.el('circle', { cx: x.toFixed(2), cy: y.toFixed(2), r: 1.6, class: 'shot-ok' }, s);
      else {
        const d = 'M' + (x - k).toFixed(2) + ' ' + (y - k).toFixed(2) + 'L' + (x + k).toFixed(2) + ' ' + (y + k).toFixed(2)
          + 'M' + (x - k).toFixed(2) + ' ' + (y + k).toFixed(2) + 'L' + (x + k).toFixed(2) + ' ' + (y - k).toFixed(2);
        FIG.el('path', { d: d, class: 'shot-bad-u' }, s);
        FIG.el('path', { d: d, class: 'shot-bad' }, s);
      }
    });
    over.append(s);
  },

  // points: the corners of each photo kept (a few each); last: the corners of the last photo, outlined when kept;
  // missing: the corners of the photo (top_left ...) the board has not reached yet in the photos kept (as
  // calibrate_charuco.coverage counts them). Each corner of the photo has a quarter disc in it: filled once reached, a
  // dashed outline while missing.
  CORNER_R: 13,
  CORNERS: { top_left: [1, 0, 0], top_right: [-1, 100, 0], bottom_right: [-1, 100, 75], bottom_left: [1, 0, 75] },
  cov: function (host, points, last, ok, missing) {
    host.replaceChildren();
    const W = 100, H = 75, s = FIG.svg(W, H, 'cov');
    s.setAttribute('preserveAspectRatio', 'none');
    FIG.el('rect', { x: 0, y: 0, width: W, height: H, class: 'cov-frame' }, s);
    // thirds, faint: the nine places of the image the plans fill
    FIG.el('path', { d: 'M' + W / 3 + ' 0V' + H + 'M' + 2 * W / 3 + ' 0V' + H + 'M0 ' + H / 3 + 'H' + W + 'M0 ' + 2 * H / 3 + 'H' + W,
                     class: 'cov-grid' }, s);
    if (missing) {
      // the top left one, the others mirrored (the box has the photo's 4:3 shape: its units are square)
      const r = FIG.CORNER_R, d = 'M0 0H' + r + 'A' + r + ' ' + r + ' 0 0 1 0 ' + r + 'Z';
      Object.keys(FIG.CORNERS).forEach(function (name) {
        const m = FIG.CORNERS[name], sy = m[2] ? -1 : 1;
        FIG.el('path', { d: d, transform: 'matrix(' + m[0] + ' 0 0 ' + sy + ' ' + m[1] + ' ' + m[2] + ')',
                         class: 'cov-corner ' + (missing.indexOf(name) >= 0 ? 'miss' : 'hit') }, s);
      });
    }
    const P = function (q) { return [q[0] * W, q[1] * H]; };
    (points || []).forEach(function (pts) {
      if (pts && pts.length >= 3) FIG.el('polygon', { points: FIG.poly(FIG.hull(pts).map(P)), class: 'cov-area' }, s);
    });
    if (ok && last && last.length >= 3) {
      FIG.el('polygon', { points: FIG.poly(FIG.hull(last).map(P)), class: 'cov-last' }, s);
    }
    host.append(s);
  },

  // ── the error of each photo ─────────────────────────────────────────────
  // photos: [{name, err, state: used|dropped|excluded}]; limit: the 3-times-the-median line; toggle(name)
  bars: function (host, photos, limit, toggle) {
    host.replaceChildren();
    const W = 440, H = 200, L = 34, R = 8, Tp = 10, B = 18, n = photos.length;
    if (!n) return;
    const s = FIG.svg(W, H, 'bars');
    const top = Math.max(limit || 0, 0.5, Math.max.apply(null, photos.map(function (p) { return p.err; }))) * 1.12;
    const y = function (v) { return Tp + (H - Tp - B) * (1 - v / top); };
    const step = FIG.nice(top / 3);
    for (let v = 0; v <= top + 1e-9; v += step) {
      FIG.el('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: v ? 'ax-grid' : 'ax-base' }, s);
      FIG.text(s, L - 5, y(v) + 3.5, fmtNum(v, step < 0.1 ? 2 : 1), 'fig-t fig-t-s', 'end');
    }
    FIG.text(s, L - 5, Tp - 1, 'px', 'fig-t fig-t-s', 'end');
    // the bars close together: the gap between two a quarter of a bar's band, a bar never wider than 26
    const band = (W - L - R) / n, bw = Math.max(2, Math.min(26, band * 0.76));
    // a photo used in the colour of its error on the scale of this series (the best blue, the worst yellow orange), one left
    // out in grey
    const used = photos.filter(function (p) { return p.state === 'used'; }).map(function (p) { return p.err; });
    const lo = used.length ? Math.min.apply(null, used) : 0, hi = used.length ? Math.max.apply(null, used) : 0;
    const col = FIG.scale();
    photos.forEach(function (p, i) {
      const x = L + i * band + (band - bw) / 2, h = Math.max(1, y(0) - y(p.err)), r = Math.min(1.6, bw / 2, h);
      const g = FIG.el('g', { class: 'bar ' + (p.state === 'used' ? 'bar-in' : 'bar-out') }, s);
      FIG.el('rect', { x: L + i * band, y: Tp, width: band, height: H - Tp - B, class: 'bar-hit' }, g);
      const v = FIG.el('path', { d: 'M' + x + ' ' + y(0) + 'v' + (-(h - r)) + 'q0 ' + (-r) + ' ' + r + ' ' + (-r) + 'h' + (bw - 2 * r)
                                   + 'q' + r + ' 0 ' + r + ' ' + r + 'v' + (h - r) + 'z', class: 'bar-v' }, g);
      if (p.state === 'used') v.setAttribute('style', FIG.mark(col, hi > lo ? (p.err - lo) / (hi - lo) : 0));
      FIG.title(g, (i + 1) + ' · ' + p.name + ' · ' + fmtNum(p.err, 2) + ' px · ' + T('cal.res.state.' + p.state));
      if (toggle) g.addEventListener('click', function () { toggle(p.name); });
      if (n <= 24 || i % Math.ceil(n / 12) === 0) FIG.text(s, x + bw / 2, H - 5, String(i + 1), 'fig-t fig-t-s', 'middle');
    });
    if (limit) {
      FIG.el('line', { x1: L, x2: W - R, y1: y(limit), y2: y(limit), class: 'ax-limit' }, s);
      FIG.text(s, W - R, y(limit) - 4, T('cal.res.limit'), 'fig-t fig-t-s', 'end');
    }
    host.append(s);
  },

  // a round step for an axis: 1, 2 or 5 times a power of ten
  nice: function (v) {
    const p = Math.pow(10, Math.floor(Math.log10(v || 1)));
    const m = v / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
  },

  // ── where on the board the error is ─────────────────────────────────────
  // res: calibrate_charuco.corner_residuals (rows [i, j, rms px, photos, mean x, mean y]); grid: inner corners.
  // The board as wide as the card allows beside the key (its height follows the board's shape, at most MAP_H), its
  // squares drawn, then its outline over them in one stroke. A dot per corner, larger and lighter where its error is
  // larger: size and colour on the scale of this series (the smallest error blue, the largest yellow orange), so it is
  // worked out again for other photos or with some left out; the key, the colour bar with its two ends in px.
  MAP_H: 260,
  map: function (host, res, grid) {
    host.replaceChildren();
    if (!res || !grid) return null;
    const nx = grid[0] + 1, ny = grid[1] + 1, W = 440, pad = 4, key = 64;
    const sq = Math.min((W - 2 * pad - key) / nx, (FIG.MAP_H - 2 * pad) / ny);
    const H = ny * sq + 2 * pad, ox = pad, oy = pad;
    const s = FIG.svg(W, H, 'rmap');
    FIG.el('rect', { x: ox, y: oy, width: nx * sq, height: ny * sq, class: 'rmap-board' }, s);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if ((i + j) % 2 === 0) FIG.el('rect', { x: ox + i * sq, y: oy + j * sq, width: sq, height: sq, class: 'rmap-sq' }, s);
      }
    }
    FIG.el('rect', { x: ox, y: oy, width: nx * sq, height: ny * sq, class: 'rmap-edge' }, s);
    const vals = res.corners.map(function (c) { return c[2]; });
    const hi = Math.max.apply(null, vals), lo = Math.min.apply(null, vals);
    const rmax = Math.min(7, sq * 0.42), col = FIG.scale();
    const t = function (v) { return hi > lo ? Math.max(0, v - lo) / (hi - lo) : 0; };
    const r = function (v) { return rmax * (0.3 + 0.7 * Math.sqrt(t(v))); };
    res.corners.forEach(function (c) {
      const g = FIG.el('g', {}, s);
      FIG.el('circle', { cx: ox + (c[0] + 1) * sq, cy: oy + (c[1] + 1) * sq, r: sq * 0.5, class: 'rmap-hit' }, g);
      FIG.el('circle', { cx: ox + (c[0] + 1) * sq, cy: oy + (c[1] + 1) * sq, r: r(c[2]).toFixed(2), class: 'rmap-dot',
                         style: FIG.mark(col, t(c[2])) }, g);
      FIG.title(g, fmtNum(c[2], 2) + ' px · ' + T('cal.res.seen', c[3]));
    });
    // the key: the colour bar, the largest error (yellow orange) on top, the smallest (blue) at the bottom
    const id = 'rgrad' + Math.random().toString(36).slice(2, 8), lx = ox + nx * sq + 14, kw = 7;
    const ky = oy + 6, kh = Math.max(40, Math.min(120, ny * sq - 12));
    const grad = FIG.el('linearGradient', { id: id, x1: 0, y1: 1, x2: 0, y2: 0 }, FIG.el('defs', {}, s));
    for (let k = 0; k <= 12; k++) FIG.el('stop', { offset: (k / 12).toFixed(3), 'stop-color': col(k / 12) }, grad);
    const bar = function () { return 'url(#' + id + ')'; };
    bar.light = col.light;
    FIG.el('rect', { x: lx, y: ky, width: kw, height: kh, rx: 1.5, style: FIG.mark(bar, 0) }, s);
    FIG.text(s, lx + kw + 5, ky + 6, fmtNum(hi, 2) + ' px', 'fig-t fig-t-s');
    FIG.text(s, lx + kw + 5, ky + kh, fmtNum(lo, 2) + ' px', 'fig-t fig-t-s');
    host.append(s);
    return { lo: lo, hi: hi };
  },

  // ── the A4 sheet: the board, and the length to measure over its top row ─
  // The page and its board as printed (the white squares with a marker, drawn as a black square with two white
  // cells, a hint of its code), the length over the top row from the left edge of the first square to the right edge
  // of the last one, in the accent, with its value.
  sheet: function (host, info) {
    host.replaceChildren();
    if (!info) return;
    const W = 330, H = 252, pw = 297, ph = 210, x0 = (W - pw) / 2, y0 = 34;
    const nx = info.squares[0], ny = info.squares[1], sq = info.square_mm;
    const bx = x0 + (pw - nx * sq) / 2, by = y0 + (ph - ny * sq) / 2;
    const s = FIG.svg(W, H, 'sheet');
    FIG.el('rect', { x: x0, y: y0, width: pw, height: ph, rx: 2, class: 'sh-page' }, s);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const x = bx + i * sq, y = by + j * sq;
        if ((i + j) % 2 === 0) { FIG.el('rect', { x: x, y: y, width: sq, height: sq, class: 'sh-sq' }, s); continue; }
        const m = sq * 0.75, o = (sq - m) / 2, c = m / 7, k = (i * 7 + j * 3) % 5;
        FIG.el('rect', { x: x + o, y: y + o, width: m, height: m, class: 'sh-sq' }, s);
        FIG.el('rect', { x: x + o + c * (1 + k % 3), y: y + o + c * (1 + (k + 1) % 3), width: 2 * c, height: c,
                         class: 'sh-bit' }, s);
        FIG.el('rect', { x: x + o + c * (4 - k % 2), y: y + o + c * (3 + k % 2), width: c, height: 2 * c,
                         class: 'sh-bit' }, s);
      }
    }
    const ly = y0 - 9, x1 = bx + nx * sq;
    FIG.el('path', { d: 'M' + bx + ' ' + (by - 0.5) + 'V' + (ly - 4) + 'M' + x1 + ' ' + (by - 0.5) + 'V' + (ly - 4),
                     class: 'sh-ext' }, s);
    // the dimension line between two filled heads, running a little into each so that they join with no gap
    const hl = 7;
    FIG.el('path', { d: 'M' + (bx + hl - 1) + ' ' + ly + 'H' + (x1 - hl + 1), class: 'sh-dim' }, s);
    FIG.head(s, [bx, ly], [-1, 0], hl, 'sh-head');
    FIG.head(s, [x1, ly], [1, 0], hl, 'sh-head');
    FIG.text(s, (bx + x1) / 2, ly - 5, fmtNum(info.nominal_mm, 1) + ' mm', 'fig-t sh-len', 'middle');
    host.append(s);
  },
};
