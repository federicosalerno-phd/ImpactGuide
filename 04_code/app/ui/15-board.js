// ImpactGuide · the calibration board, the first step of Calibrate the phone
// and of Live AR (both copies of its controls, ids starting with cal and lv).
//
// First the question: this monitor, or a physical board. On this monitor the
// board is drawn here, in the work area of the step that shows it (BOARDVIEW):
// the page tells the backend the area's size in its device pixels, which are
// the monitor's own, and draws the image it gets back one pixel on one pixel,
// never scaled; the size in mm follows from the monitor's pitch. The window is
// never moved or resized for it: a larger window makes a larger board, and the
// board is laid out again when the area changes or the window goes to another
// monitor. Physical: the 3D-printed plate (recommended, shown in 3D by
// ui/17-plate3d.js) or a printed A4 sheet, each measured three times with a
// caliper; the mean and the spread are shown as the readings are typed.
'use strict';

const BOARD_LAYOUT_DELAY_MS = 60;   // a burst of size changes (a resize being dragged) asks for one layout

// The sign of this monitor with the board on it, on SlantUI's 24 grid: 5 x 3 squares of 2.4, lying across, as
// every board sign of the app (ui/01-shell.js, ICONS.board).
ICONS.screen = { s: '<rect x="3" y="4.5" width="18" height="12" rx="1.5"/><path d="M9 20h6M12 16.5V20"/>'
  + '<path d="M6 6.9h2.4v2.4h-2.4zM10.8 6.9h2.4v2.4h-2.4zM15.6 6.9h2.4v2.4h-2.4zM8.4 9.3h2.4v2.4h-2.4zM13.2 9.3h2.4v2.4'
  + 'h-2.4zM6 11.7h2.4v2.4h-2.4zM10.8 11.7h2.4v2.4h-2.4zM15.6 11.7h2.4v2.4h-2.4z" fill="currentColor" stroke="none"/>' };

/** The board's controls, in Calibrate and in Live AR (ids starting with cal and lv), and the board step's stage. */
const BOARD = {
  state: null,
  copies: ['cal', 'lv'],

  mm: function (v, n) { return fmtNum(v, n === undefined ? 1 : n); },
  physical: function (kind) { return kind === 'plate' || kind === 'sheet'; },
  // a board in plain words: 11 × 8 · 20,00 mm
  line: function (b) { return b.squares[0] + ' × ' + b.squares[1] + ' · ' + BOARD.mm(b.square_mm, 2) + ' mm'; },
  // a monitor by its name and its diagonal. Short: the model alone, without the maker's codes Windows and the EDID
  // put before it (PHI PHL BDM4037U: BDM4037U), and its inches without a decimal when they are whole (40″); else the
  // whole name and the inches to a tenth (PHI PHL BDM4037U · 40.0″)
  monName: function (m, short) {
    const words = String(m.label || '').trim().split(/\s+/).filter(Boolean);
    while (short && words.length > 1 && /^[A-Z]{3}$/.test(words[0])) words.shift();
    const d = Number(m.diag_in), whole = Math.abs(d - Math.round(d)) < 0.05;
    const inch = isFinite(d) && d > 0 ? fmtNum(d, short && whole ? 0 : 1) + '″' : '';
    return [words.join(' '), inch].filter(Boolean).join(' · ') || '-';
  },

  // on this monitor: what needs doing, one short line, or nothing
  warning: function (s) {
    const m = s.mon;
    if (!m) return T('board.err.no_monitor');
    if (s.problem === 'too_small') return T('board.err.small');
    if (s.problem === 'across') return T('board.err.across');
    const lay = s.layout;
    if (lay && (lay.squares[0] - 1) * (lay.squares[1] - 1) < 24) return T('board.warn.small_board');
    if (!m.is_native && m.native) return T('board.warn.native', m.native[0], m.native[1]);
    if (m.source === 'none') return T('board.sure.unknown');
    if (m.source === 'edid_cm') return T('board.sure.rough');
    if (m.agree === false && m.source !== 'ruler') return T('board.warn.disagree', fmtNum(100 * m.spread, 1));
    return '';
  },

  // physical: what is wrong with the readings typed, one short line, or nothing
  readWarning: function (r, nominal) {
    if (!r || !r.problems || !r.problems.length) return '';
    if (r.problems.indexOf('few_readings') >= 0) return T('board.read.few');
    if (r.problems.indexOf('spread') >= 0) return T('board.read.spread', BOARD.mm(r.spread_mm, 2));
    if (r.problems.indexOf('off_nominal') >= 0) return T('board.read.off', BOARD.mm(nominal, 1));
    return '';
  },

  render: function (s) {
    if (!s) return;
    BOARD.state = s;
    BOARD.copies.forEach(function (p) { if ($(p + 'Bd')) BOARD.draw(p, s); });
    if (BOARD.onStage()) BOARD.stage(s);
    BOARDVIEW.label();
  },

  // the board view is on the stage: the first step of Calibrate, or of Live AR on the calibration board
  onStage: function () {
    return (page === 'calibration' && STEP.calibration === 'board')
      || (page === 'live' && STEP.live === 'board' && LIVE.state && LIVE.state.prefs.board === 'sheet');
  },

  draw: function (p, s) {
    const phys = BOARD.physical(s.kind), kind = s.kind;
    document.querySelectorAll('#' + p + 'BdKinds .mitem').forEach(function (it) {
      it.classList.toggle('sel', s.chosen && (it.dataset.kind === 'physical' ? phys : it.dataset.kind === kind));
    });
    document.querySelectorAll('#' + p + 'BdWhich .mitem').forEach(function (it) {
      it.classList.toggle('sel', it.dataset.kind === kind);
    });
    $(p + 'BdScreen').classList.toggle('hidden', !s.chosen || phys);
    $(p + 'BdPhys').classList.toggle('hidden', !s.chosen || !phys);

    // on this monitor: the board and the monitor, what needs doing
    const lay = s.layout, m = s.mon, rows = [];
    if (lay) rows.push(piRow(T('board.f.board'), BOARD.line(lay)));
    if (m) {
      // its short name on the row, the whole one with its pixels and where its size comes from on the hint
      const r = piRow(T('board.f.monitor'), BOARD.monName(m, true));
      r.title = BOARD.monName(m, false) + ' · ' + m.px[0] + ' × ' + m.px[1] + ' · ' + Tcode('board.src.', m.source, 'none');
      rows.push(r);
    }
    $(p + 'BdRows').replaceChildren.apply($(p + 'BdRows'), rows);
    const warn = s.shown || p === 'cal' ? BOARD.warning(s) : '';
    $(p + 'BdWarn').textContent = warn;
    $(p + 'BdWarn').classList.toggle('hidden', !warn);
    if (p === 'cal') {
      // the ruler check, once per monitor: the length between the two thin lines above the board
      const ruler = $('calBdRulerV');
      if (lay) ruler.placeholder = BOARD.mm(lay.width_mm);
      if (document.activeElement !== ruler) ruler.value = '';
      ruler.disabled = !lay;
      $('calBdRulerNote').textContent = !lay || !m ? T('board.ruler.none')
        : m.ruler ? T('board.ruler.done', fmtTime(m.ruler.date), BOARD.mm(m.ruler.measured_mm))
        : T('board.ruler.note', BOARD.mm(lay.width_mm));
      $('calBdForget').classList.toggle('hidden', !(m && m.ruler));
    }

    // physical: the readings and what they give
    const info = phys ? s.physical[kind] : null;
    if (!info) return;
    const rd = info.long, nominal = kind === 'plate' ? info.measure.long.nominal_mm : info.nominal_mm;
    const spec = info.spec, out = [];
    if (rd) {
      out.push(piRow(T('board.read.mean', rd.n), BOARD.mm(rd.mean_mm, 2) + ' mm'),
               piRow(T('board.read.spread.l'), BOARD.mm(rd.spread_mm, 2) + ' mm'));
    }
    out.push(piRow(T('board.f.square'), BOARD.mm(spec.square_mm, 3) + ' mm' + (spec.measured ? '' : ' · ' + T('board.read.nominal'))));
    $(p + 'BdRead').replaceChildren.apply($(p + 'BdRead'), out);
    const rw = p === 'cal' ? BOARD.readWarning(rd, nominal) : (!spec.measured ? T('board.read.todo') : '');
    $(p + 'BdReadWarn').textContent = rw;
    $(p + 'BdReadWarn').classList.toggle('hidden', !rw);
    if (p !== 'cal') return;
    BOARD.fill('calBdLong', (info.readings || {}).long);
    $('calBdLong').querySelectorAll('input').forEach(function (i) { i.placeholder = BOARD.mm(nominal, 1); });
    $('calBdFileL').textContent = T(kind === 'plate' ? 'board.plate.file' : 'board.sheet.file');
    $('calBdFile').title = T(kind === 'plate' ? 'board.plate.file.hint' : 'board.sheet.file.hint');
    $('calBdShort').classList.toggle('hidden', kind !== 'plate');
    if (kind === 'plate') {
      BOARD.fill('calBdShortV', (info.readings || {}).short);
      $('calBdShortV').querySelectorAll('input').forEach(function (i) { i.placeholder = BOARD.mm(info.measure.short.nominal_mm, 1); });
      const sh = info.short, so = [];
      if (sh) so.push(piRow(T('board.read.mean', sh.n), BOARD.mm(sh.mean_mm, 2) + ' mm'));
      if (spec.anisotropy !== undefined) so.push(piRow(T('board.read.xy'), fmtNum(100 * spec.anisotropy, 2) + ' %'));
      $('calBdShortRead').replaceChildren.apply($('calBdShortRead'), so);
    }
  },

  // the readings typed, back in their fields (the one being typed in is left alone)
  fill: function (id, vals) {
    $(id).querySelectorAll('input').forEach(function (inp, i) {
      if (document.activeElement === inp) return;
      const v = vals && vals[i];
      inp.value = v === undefined || v === null ? '' : String(v);
    });
  },
  readings: function (id) {
    const out = [];
    $(id).querySelectorAll('input').forEach(function (inp) {
      const v = parseFloat(String(inp.value).replace(',', '.'));
      if (isFinite(v) && v > 0) out.push(v);
    });
    return out;
  },

  // ── the stage of the board step ───────────────────────────────────────
  stage: function (s) {
    const asking = !s.chosen, phys = !asking && BOARD.physical(s.kind);
    $('bdPatient').classList.add('hidden');
    $('bdAsk').classList.toggle('hidden', !asking);
    $('bdSplit').classList.toggle('hidden', asking || phys);
    $('bdPhys').classList.toggle('hidden', !phys);
    // in Calibrate the board sits where it will be for the photos, beside the guide of the first one
    // (none while no board fits: the column is then empty, never a guide drawn for another board or language)
    const guide = page === 'calibration' && CAL.state && CAL.state.next;
    $('bdGuideCol').classList.toggle('hidden', page !== 'calibration');
    FIG.guide($('bdGuide'), guide || null, false);
    BOARDVIEW.place(!asking && !phys ? 'bdHostBoard' : null);
    if (!phys) { PLATE3D.hide(); return; }
    const plate = s.kind === 'plate' && s.physical.plate;
    $('bd3d').classList.toggle('hidden', !plate);
    $('bdSheetFig').classList.toggle('hidden', !!plate);
    if (plate) PLATE3D.show(s.physical.plate);
    else { PLATE3D.hide(); FIG.sheet($('bdSheetFig'), s.physical.sheet); }
    BOARD.steps(s);
  },

  // what to do with the physical board: three steps, a few words each
  steps: function (s) {
    const plate = s.kind === 'plate', info = s.physical[s.kind];
    const nominal = plate ? info.measure.long.nominal_mm : info.nominal_mm;
    const mm = function (v, n) { return withUnit(v, n, 'mm'); };     // a number never apart from its unit
    const rows = plate ? [
      [T('board.st.print'), T('board.st.print.plate')],
      [T('board.st.measure'), T('board.st.measure.plate', mm(nominal, 1))],
      [T('board.st.type'), T('board.st.type.line', mm(info.spread_max, 2))],
    ] : [
      [T('board.st.print.sheet.t'), T('board.st.print.sheet')],
      [T('board.st.measure'), T('board.st.measure.sheet', mm(nominal, 1))],
      [T('board.st.type'), T('board.st.type.line', mm(info.spread_max, 2))],
    ];
    const host = $('bdSteps');
    host.replaceChildren();
    host.append(E('div', 'bd-steps-hd', T(plate ? 'board.kind.plate' : 'board.kind.sheet')));
    rows.forEach(function (r, i) {
      const row = E('div', 'bd-step');
      const txt = E('div', 'bd-step-t');
      txt.append(E('div', 'bd-step-n', r[0]), E('div', 'bd-step-l', r[1]));
      row.append(E('span', 'tn', i + 1), txt);
      host.append(row);
    });
    host.append(E('div', 'bd-tool', T('board.st.tool')));
  },

  // A change of the board; a refusal is said in this part's own words (board.err.*) when it has them.
  set: function (change) {
    askQuiet('board_set', JSON.stringify(change), function (s, bad) {
      if (bad) { toast(errorText(bad, 'board.err.'), TOAST_LONG_MS, 'error'); return; }
      BOARD.render(s);
      if (change.ruler_mm !== undefined) toast(T('board.said.corrected'), TOAST_SAID_MS);
      BOARD.pageRefresh();
    });
  },

  pageRefresh: function () {
    if (page === 'calibration') CAL.refresh();
    else if (page === 'live') LIVE.refresh();
  },

  onEvent: function (e) {
    if (e.type !== 'board') return;
    if (e.action === 'screen') BOARDVIEW.soon();      // another monitor, or the monitor changed: laid out again
  },

  wire: function () {
    // the first question, on the stage
    document.querySelectorAll('#bdAsk .choice').forEach(function (b) {
      b.onclick = function () {
        const s = BOARD.state;
        BOARD.set({ kind: b.dataset.kind === 'plate' && s && s.kind === 'sheet' ? 'sheet' : b.dataset.kind });
      };
    });
    BOARD.copies.forEach(function (p) {
      if (!$(p + 'Bd')) return;
      document.querySelectorAll('#' + p + 'BdKinds .mitem').forEach(function (it) {
        onActivate(it, function () {
          const s = BOARD.state;
          let k = it.dataset.kind;
          if (k === 'physical') k = s && BOARD.physical(s.kind) ? s.kind : 'plate';
          BOARD.set({ kind: k });
        });
      });
      document.querySelectorAll('#' + p + 'BdWhich .mitem').forEach(function (it) {
        onActivate(it, function () { BOARD.set({ kind: it.dataset.kind }); });
      });
    });
    $('calBdRulerV').onchange = function () {
      const v = parseFloat(String($('calBdRulerV').value).replace(',', '.'));
      if (isFinite(v) && v > 0) BOARD.set({ ruler_mm: v });
    };
    $('calBdForget').onclick = function () { BOARD.set({ forget: true }); };
    // the readings: kept as they are typed (mean and spread follow at once)
    $('calBdLong').querySelectorAll('input').forEach(function (inp) {
      inp.onchange = function () {
        const s = BOARD.state, c = {};
        c[(s && s.kind === 'sheet' ? 'sheet' : 'plate') + '_long'] = BOARD.readings('calBdLong');
        BOARD.set(c);
      };
    });
    $('calBdShortV').querySelectorAll('input').forEach(function (inp) {
      inp.onchange = function () { BOARD.set({ plate_short: BOARD.readings('calBdShortV') }); };
    });
    $('calBdFile').onclick = function () {
      const s = BOARD.state;
      if (s && s.kind === 'sheet') ask('calib_board_pdf');
      else ask('plate_open', '3mf');
    };
    $('calBdFolder').onclick = function () { ask('plate_open', 'folder'); };
    // the ruler's fold opened or closed: the length is written on the board while it is open
    $('calBdRuler').querySelector('.fold-hd').addEventListener('click', function () { setTimeout(BOARDVIEW.label, 0); });
  },
};

// ── the board on this monitor, drawn by the page ─────────────────────────────
// One canvas, moved into the work area of the step that shows it (the board
// step, the photos, the live view). Its size in device pixels is what the
// backend lays the board out for (board_area); the image comes back as large
// as the canvas, so it is drawn at (0, 0) one pixel on one pixel. While the
// canvas has another size than its image (a resize in progress) it is hidden:
// a stretched board is never on screen.
/** The board on this monitor: one canvas the backend's image is drawn on, one image pixel on one screen pixel. */
const BOARDVIEW = {
  wrap: null, canvas: null, len: null, msg: null, host: null, key: '', dev: null, timer: null, busy: false,
  again: false, lay: null, shown: false,

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
    if (!r.shown) return;
    const c = BOARDVIEW.canvas, msg = BOARDVIEW.msg;
    BOARDVIEW.lay = r.state && r.state.layout;
    if (r.png) {
      const img = new Image();
      img.onload = function () {
        if (img.width !== a.w || img.height !== a.h) return;     // an older answer: a newer one is on its way
        c.width = img.width;
        c.height = img.height;
        const g = c.getContext('2d', { alpha: false });
        g.imageSmoothingEnabled = false;
        g.drawImage(img, 0, 0);
        c.style.visibility = BOARDVIEW.dev && (BOARDVIEW.dev[0] !== c.width || BOARDVIEW.dev[1] !== c.height) ? 'hidden' : '';
        const changed = BOARDVIEW.key !== r.key;
        BOARDVIEW.key = r.key;
        BOARDVIEW.label();
        // another board: the module asks again for what follows from it (the guide of the next photo, the series)
        if (changed) BOARD.pageRefresh();
      };
      img.src = r.png;
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
