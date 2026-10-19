// ImpactGuide · the figures of the calibration, drawn as SVG in the palette's
// roles: the guide of each photo (the phone's screen with the board as it
// should look there, the phone's top and bottom edges marked, and the phone
// seen from above and from the side, at its distance and tilt), where the
// corners of the photos fell, the error of each photo (bars, a photo left out
// in grey, one press puts it in or out), where on the board the error is (a
// dot per corner, larger and brighter where the error is larger), and the A4
// sheet with what to measure. Plain functions of the data the backend gives;
// a figure never holds state of its own.
'use strict';

/** The calibration's figures, each a function of the backend's data that fills a host element. */
const FIG = {
  NS: 'http://www.w3.org/2000/svg',

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

  // ── the guide of the next photo ─────────────────────────────────────────
  // g: calib_guide.guide (outline and black squares in fractions of the screen, yaw, pitch, roll, distance)
  // the pose in words, two short lines: where from and how far round, then the distance and the turn
  caption: function (g) {
    const v = g.view === 'front' ? T('guide.view.front') : T('guide.view.' + g.view) + ' ' + g.tilt + '°';
    const parts = [fmtNum(g.dist_cm, 0) + ' cm'];
    if (Math.abs(g.roll) >= 5) parts.push(T('guide.turned', Math.abs(g.roll)));
    return [v, parts.join(' · ')];
  },

  guide: function (host, g, big) {
    host.replaceChildren();
    if (!g) return;
    host.classList.toggle('big', !!big);
    const head = E('div', 'guide-hd'), words = FIG.caption(g);
    head.append(E('span', 'guide-n', T('guide.n', g.n + 1, g.of)), E('span', 'guide-w', words[0]),
                E('span', 'guide-d', words[1]));
    host.append(head);
    const row = E('div', 'guide-row');
    row.append(FIG.phone(g), FIG.sides(g));
    host.append(row);
    const leg = E('div', 'guide-leg');
    const sw = function (cls, word) { const s = E('span', 'guide-key'); s.append(E('i', cls), E('span', null, word)); return s; };
    leg.append(sw('k-top', T('guide.top')), sw('k-bottom', T('guide.bottom')));
    host.append(leg);
  },

  // The phone held across, its top to the left: a phone's shape (19.5 : 9), its screen as the camera shows it, the
  // photo's frame (4:3) under the strip at the top and the shutter in the part at the bottom, the board in the frame
  // as it should be there.
  phone: function (g) {
    const L = 195, D = 90, pad = 1.5, W = L + 2 * pad, H = D + 2 * pad, cy = H / 2;
    const s = FIG.svg(W, H, 'guide-phone');
    FIG.el('rect', { x: pad, y: pad, width: L, height: D, rx: 12, class: 'g-body' }, s);
    // the screen inside an even bezel; the camera's strip at the top (left), the photo's frame, the shutter's part
    const b = 4.5, sx = pad + b, sy = pad + b, sw = L - 2 * b, sh = D - 2 * b;
    FIG.el('rect', { x: sx, y: sy, width: sw, height: sh, rx: 8, class: 'g-screen' }, s);
    const fh = sh, fw = fh * (g.aspect || 4 / 3), fx = sx + 20, fy = sy;
    FIG.el('circle', { cx: sx + 8, cy: cy, r: 1.8, class: 'g-lens' }, s);
    const kx = (fx + fw + sx + sw) / 2;
    FIG.el('circle', { cx: kx, cy: cy, r: 7.5, class: 'g-shutter' }, s);
    FIG.el('circle', { cx: kx, cy: cy, r: 5.5, class: 'g-shutter-in' }, s);
    // the top edge (left) and the bottom edge (right), on the bezel
    FIG.el('rect', { x: pad + 1, y: cy - D * 0.3, width: 2.5, height: D * 0.6, rx: 1.25, class: 'g-top' }, s);
    FIG.el('rect', { x: pad + L - 3.5, y: cy - D * 0.2, width: 2.5, height: D * 0.4, rx: 1.25, class: 'g-bottom' }, s);
    const id = 'gclip' + Math.random().toString(36).slice(2, 8);
    const cp = FIG.el('clipPath', { id: id }, FIG.el('defs', {}, s));
    FIG.el('rect', { x: fx, y: fy, width: fw, height: fh }, cp);
    FIG.el('rect', { x: fx, y: fy, width: fw, height: fh, class: 'g-frame' }, s);
    const inner = FIG.el('g', { 'clip-path': 'url(#' + id + ')' }, s);
    const P = function (q) { return [fx + q[0] * fw, fy + q[1] * fh]; };
    FIG.el('polygon', { points: FIG.poly(g.outline.map(P)), class: 'g-board' }, inner);
    g.cells.forEach(function (c) { FIG.el('polygon', { points: FIG.poly(c.map(P)), class: 'g-cell' }, inner); });
    // the middle of the photo, faint: where the board's centre goes is read against it
    const mx = fx + fw / 2;
    FIG.el('path', { d: 'M' + (mx - 4) + ' ' + cy + 'h8M' + mx + ' ' + (cy - 4) + 'v8', class: 'g-mid' }, s);
    return s;
  },

  // The phone seen from above (left and right) and from the side (up and down): the board as a thick line, the
  // phone at its distance, turned to look at the board's centre, the line of sight dashed.
  sides: function (g) {
    const box = E('div', 'guide-sides');
    // the distance on the drawing: from the nearest to the farthest the plans use
    const near = 250, far = 600;
    const d = 30 + 26 * Math.max(0, Math.min(1, (g.dist_cm * 10 - near) / (far - near)));
    const one = function (ang, word, vertical) {
      const W = 120, H = 78, s = FIG.svg(W, H, 'guide-side');
      const a = ang * Math.PI / 180;
      let bx, by, px, py, rot;
      // the board edge on: a bar of black and white squares, so it reads as the board
      const bar = function (x, y, horizontal) {
        const n = 6, L = 64, t = 6;
        for (let i = 0; i < n; i++) {
          const o = -L / 2 + i * L / n;
          FIG.el('rect', horizontal ? { x: x + o, y: y - t / 2, width: L / n, height: t }
            : { x: x - t / 2, y: y + o, width: t, height: L / n }, s).setAttribute('class', i % 2 ? 'g-bar-w' : 'g-bar-k');
        }
      };
      if (!vertical) {         // from above: the board along the top, the phone below it, left is left
        bx = W / 2; by = 8;
        bar(bx, by, true);
        px = bx + d * Math.sin(a); py = by + d * Math.cos(a);
        rot = -ang;
      } else {                 // from the side: the board along the left, the phone right of it, above is up
        bx = 10; by = H / 2;
        bar(bx, by, false);
        px = bx + d * Math.cos(a); py = by - d * Math.sin(a);
        rot = -90 - ang;
      }
      FIG.el('line', { x1: px, y1: py, x2: bx, y2: by, class: 'g-sight' }, s);
      const ph = FIG.el('g', { transform: 'translate(' + px.toFixed(2) + ' ' + py.toFixed(2) + ') rotate(' + rot.toFixed(1) + ')' }, s);
      // edge on, the phone held across: its long side from above, its short side from the side (19.5 : 9), the
      // camera towards the board
      const len = vertical ? 44 * 9 / 19.5 : 44, t = 3.6;
      FIG.el('rect', { x: -len / 2, y: -t / 2, width: len, height: t, rx: 1.2, class: 'g-slab' }, ph);
      FIG.el('rect', { x: -4, y: -t / 2 - 1.8, width: 8, height: 2.2, rx: 0.6, class: 'g-lens' }, ph);
      const cell = E('div', 'guide-side-c');
      cell.append(s, E('span', 'guide-side-l', word));
      return cell;
    };
    box.append(one(g.yaw, T('guide.above'), false), one(g.pitch, T('guide.side'), true));
    return box;
  },

  // ── where the corners of the photos fell, on a frame of the photo's shape ─
  cov: function (host, points, last, ok) {
    host.replaceChildren();
    const W = 80, H = 60, s = FIG.svg(W, H, 'cov');
    FIG.el('rect', { x: 0.5, y: 0.5, width: W - 1, height: H - 1, rx: 3, class: 'cov-frame' }, s);
    (points || []).forEach(function (pts) {
      pts.forEach(function (p) { FIG.el('circle', { cx: p[0] * W, cy: p[1] * H, r: 0.9, class: 'cov-old' }, s); });
    });
    (last || []).forEach(function (p) {
      FIG.el('circle', { cx: p[0] * W, cy: p[1] * H, r: 1.3, class: ok ? 'cov-new' : 'cov-bad' }, s);
    });
    host.append(s);
  },

  // ── the error of each photo ─────────────────────────────────────────────
  // photos: [{name, err, state: used|dropped|excluded}]; limit: the 3-times-the-median line; toggle(name)
  bars: function (host, photos, limit, toggle) {
    host.replaceChildren();
    const W = 440, H = 170, L = 34, R = 8, Tp = 10, B = 18, n = photos.length;
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
    const band = (W - L - R) / n, bw = Math.max(2, Math.min(12, band * 0.45));
    photos.forEach(function (p, i) {
      const x = L + i * band + (band - bw) / 2, h = Math.max(1, y(0) - y(p.err)), r = Math.min(0.8, bw / 2, h);
      const g = FIG.el('g', { class: 'bar ' + (p.state === 'used' ? 'bar-in' : 'bar-out') }, s);
      FIG.el('rect', { x: L + i * band, y: Tp, width: band, height: H - Tp - B, class: 'bar-hit' }, g);
      FIG.el('path', { d: 'M' + x + ' ' + y(0) + 'v' + (-(h - r)) + 'q0 ' + (-r) + ' ' + r + ' ' + (-r) + 'h' + (bw - 2 * r)
                         + 'q' + r + ' 0 ' + r + ' ' + r + 'v' + (h - r) + 'z', class: 'bar-v' }, g);
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
  // res: calibrate_charuco.corner_residuals (rows [i, j, rms px, photos, mean x, mean y]); grid: inner corners
  map: function (host, res, grid) {
    host.replaceChildren();
    if (!res || !grid) return null;
    const nx = grid[0] + 1, ny = grid[1] + 1, W = 440, H = 200, pad = 14;
    const sq = Math.min((W - 2 * pad - 70) / nx, (H - 2 * pad) / ny);
    const ox = pad, oy = (H - ny * sq) / 2;
    const s = FIG.svg(W, H, 'rmap');
    FIG.el('rect', { x: ox, y: oy, width: nx * sq, height: ny * sq, rx: 2, class: 'rmap-board' }, s);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if ((i + j) % 2 === 0) FIG.el('rect', { x: ox + i * sq, y: oy + j * sq, width: sq, height: sq, class: 'rmap-sq' }, s);
      }
    }
    const vals = res.corners.map(function (c) { return c[2]; });
    const hi = Math.max.apply(null, vals), lo = Math.min.apply(null, vals);
    const r = function (v) { return 1.6 + 4.4 * Math.sqrt(Math.max(0, v - lo) / Math.max(1e-9, hi - lo)); };
    const o = function (v) { return 0.35 + 0.65 * Math.max(0, v - lo) / Math.max(1e-9, hi - lo); };
    res.corners.forEach(function (c) {
      const g = FIG.el('g', {}, s);
      FIG.el('circle', { cx: ox + (c[0] + 1) * sq, cy: oy + (c[1] + 1) * sq, r: sq * 0.5, class: 'rmap-hit' }, g);
      FIG.el('circle', { cx: ox + (c[0] + 1) * sq, cy: oy + (c[1] + 1) * sq, r: r(c[2]).toFixed(2),
                         class: 'rmap-dot', 'fill-opacity': o(c[2]).toFixed(2) }, g);
      FIG.title(g, fmtNum(c[2], 2) + ' px · ' + T('cal.res.seen', c[3]));
    });
    // the legend: the smallest and the largest dot, with their errors
    const lx = ox + nx * sq + 22;
    [[lo, oy + 18], [hi, oy + 46]].forEach(function (k) {
      FIG.el('circle', { cx: lx, cy: k[1], r: r(k[0]), class: 'rmap-dot', 'fill-opacity': o(k[0]) }, s);
      FIG.text(s, lx + 10, k[1] + 3.5, fmtNum(k[0], 2) + ' px', 'fig-t fig-t-s');
    });
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
    FIG.el('path', { d: 'M' + bx + ' ' + ly + 'H' + x1 + 'M' + (bx + 5) + ' ' + (ly - 2.5) + 'L' + bx + ' ' + ly + 'L' + (bx + 5)
                       + ' ' + (ly + 2.5) + 'M' + (x1 - 5) + ' ' + (ly - 2.5) + 'L' + x1 + ' ' + ly + 'L' + (x1 - 5) + ' '
                       + (ly + 2.5), class: 'sh-dim' }, s);
    FIG.text(s, (bx + x1) / 2, ly - 5, fmtNum(info.nominal_mm, 1) + ' mm', 'fig-t sh-len', 'middle');
    host.append(s);
  },
};
