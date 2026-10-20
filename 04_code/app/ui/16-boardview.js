// ImpactGuide · the board on this monitor, drawn by the page (its controls and
// the board step are in ui/15-board.js).
//
// One canvas, moved into the work area of the step that shows it (the board
// step, the photos, the live view). Its size in device pixels is what the
// backend lays the board out for (board_area); the image comes back as large
// as the canvas, so it is drawn at (0, 0) one pixel on one pixel. The backend
// makes the image in the background and sends it as a board event (action
// image) with the key of its layout: only the image of the layout asked for
// last is drawn. While the canvas has another size than its image (a resize in
// progress) it is hidden: a stretched board is never on screen.
'use strict';

/** The board on this monitor: one canvas the backend's image is drawn on, one image pixel on one screen pixel. */
const BOARDVIEW = {
  wrap: null, canvas: null, len: null, msg: null, host: null, key: '', dev: null, timer: null, busy: false,
  again: false, lay: null, shown: false,
  want: null,       // {key, a}: the layout whose image the backend is making, and the area it was asked for

  init: function () {
    if (BOARDVIEW.wrap) return;
    const w = E('div', 'bd-canvas');
    const c = document.createElement('canvas');
    c.id = 'bdCanvas';
    const len = E('div', 'bd-len hidden');
    len.id = 'bdLen';
    const msg = E('div', 'bd-msg hidden');
    msg.id = 'bdMsg';
    w.append(c, len, msg);
    // the GPU reset empties the canvas: the image is asked for again (never a blank where the board should be)
    c.addEventListener('contextrestored', function () { BOARDVIEW.key = ''; BOARDVIEW.soon(); });
    BOARDVIEW.wrap = w;
    BOARDVIEW.canvas = c;
    BOARDVIEW.len = len;          // the wrap is out of the document while no step shows it: kept by reference
    BOARDVIEW.msg = msg;
    try {
      new ResizeObserver(function (es) {
        const e = es[es.length - 1];
        const d = e.devicePixelContentBoxSize && e.devicePixelContentBoxSize[0];
        BOARDVIEW.dev = d ? [d.inlineSize, d.blockSize] : null;
        if (BOARDVIEW.dev && (BOARDVIEW.dev[0] !== c.width || BOARDVIEW.dev[1] !== c.height)) c.style.visibility = 'hidden';
        BOARDVIEW.soon();
      }).observe(c, { box: 'device-pixel-content-box' });
    } catch (err) {
      new ResizeObserver(function () { BOARDVIEW.dev = null; BOARDVIEW.soon(); }).observe(c);
    }
    window.addEventListener('resize', BOARDVIEW.soon);
    (function density() {        // the window taken to a monitor with another scaling
      try {
        matchMedia('(resolution: ' + (window.devicePixelRatio || 1) + 'dppx)').addEventListener('change', function () {
          BOARDVIEW.soon(); density();
        }, { once: true });
      } catch (err) { /* nothing to follow */ }
    })();
  },

  // the work area the board goes in now (a host element), or none
  place: function (id) {
    BOARDVIEW.init();
    const host = id ? $(id) : null;
    if (host === BOARDVIEW.host) return;
    BOARDVIEW.host = host;
    if (host) host.append(BOARDVIEW.wrap);
    else BOARDVIEW.wrap.remove();
    BOARDVIEW.soon();
  },

  soon: function () {
    clearTimeout(BOARDVIEW.timer);
    BOARDVIEW.timer = setTimeout(BOARDVIEW.send, BOARD_LAYOUT_DELAY_MS);
  },

  area: function () {
    const c = BOARDVIEW.canvas;
    if (!BOARDVIEW.host || !c || !c.isConnected || !c.getClientRects().length) return { w: 0, h: 0 };
    const dpr = window.devicePixelRatio || 1, r = c.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return { w: 0, h: 0 };
    const d = BOARDVIEW.dev || [Math.round(r.width * dpr), Math.round(r.height * dpr)];
    return { w: d[0], h: d[1], dpr: dpr, x: Math.round(r.left * dpr), y: Math.round(r.top * dpr), iw: window.innerWidth };
  },

  send: function () {
    if (BOARDVIEW.busy) { BOARDVIEW.again = true; return; }
    const a = BOARDVIEW.area();
    if (!a.w && !BOARDVIEW.shown) return;
    BOARDVIEW.busy = true;
    // asked again on every change of the area: a failure is not toasted (the board step says what is wrong)
    askQuiet('board_area', JSON.stringify(Object.assign({ have: BOARDVIEW.key }, a)), function (r) {
      BOARDVIEW.busy = false;
      if (r) BOARDVIEW.answer(r, a);
      if (BOARDVIEW.again) { BOARDVIEW.again = false; BOARDVIEW.soon(); }
    });
  },

  answer: function (r, a) {
    BOARDVIEW.shown = !!r.shown;
    if (!r.shown) { BOARDVIEW.want = null; return; }
    const c = BOARDVIEW.canvas, msg = BOARDVIEW.msg;
    BOARDVIEW.lay = r.state && r.state.layout;
    BOARDVIEW.want = null;
    if (r.png) {
      BOARDVIEW.draw(r.key, r.png, a);              // a backend of an older version sends the image at once
    } else if (r.pending) {
      BOARDVIEW.want = { key: r.key, a: a };         // its image follows as a board event (BOARDVIEW.image)
    } else if (!r.key) {
      BOARDVIEW.key = '';
      c.style.visibility = 'hidden';
    } else if (BOARDVIEW.dev && BOARDVIEW.dev[0] === c.width && BOARDVIEW.dev[1] === c.height) {
      c.style.visibility = '';
    }
    const text = r.problem === 'too_small' ? T('board.err.small') : r.problem === 'no_monitor' ? T('board.err.no_monitor') : '';
    msg.textContent = text;
    msg.classList.toggle('hidden', !text);
    if (r.state) {
      BOARD.render(r.state);
      if (page === 'calibration' && CAL.state) CAL.state.board_ui = r.state;
    }
  },

  // The board event with an image: drawn when it is of the layout asked for last.
  image: function (e) {
    const w = BOARDVIEW.want;
    if (!w || w.key !== e.key || !e.png) return;
    BOARDVIEW.want = null;
    BOARDVIEW.draw(e.key, e.png, w.a);
  },

  // The image of a layout on the canvas, one pixel on one pixel; an image of another size than the area it was
  // asked for is an older one (a newer one is on its way) and is left out.
  draw: function (key, png, a) {
    const c = BOARDVIEW.canvas, img = new Image();
    img.onload = function () {
      if (img.width !== a.w || img.height !== a.h) return;
      c.width = img.width;
      c.height = img.height;
      const g = c.getContext('2d', { alpha: false });
      g.imageSmoothingEnabled = false;
      g.drawImage(img, 0, 0);
      c.style.visibility = BOARDVIEW.dev && (BOARDVIEW.dev[0] !== c.width || BOARDVIEW.dev[1] !== c.height) ? 'hidden' : '';
      const changed = BOARDVIEW.key !== key;
      BOARDVIEW.key = key;
      BOARDVIEW.label();
      // another board: the module asks again for what follows from it (the guide of the next photo, the series)
      if (changed) BOARD.pageRefresh();
    };
    img.src = png;
  },

  // The length between the two thin lines over the board, written on it while the ruler check is open.
  label: function () {
    const el = BOARDVIEW.len, lay = BOARDVIEW.lay;
    if (!el) return;
    const on = !!(lay && page === 'calibration' && STEP.calibration === 'board' && $('calBdRuler').classList.contains('open')
                  && BOARDVIEW.host === $('bdHostBoard'));
    el.classList.toggle('hidden', !on);
    if (!on) return;
    const dpr = window.devicePixelRatio || 1;
    el.textContent = BOARD.mm(lay.width_mm, 1) + ' mm';
    el.style.left = ((lay.origin_px[0] + lay.squares[0] * lay.square_px / 2) / dpr) + 'px';
    el.style.top = ((lay.line_y - 0.5) / dpr) + 'px';
  },
};
