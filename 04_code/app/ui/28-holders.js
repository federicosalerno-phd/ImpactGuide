// ImpactGuide · Holders, the last step of Trials: the two marker holders measured on their own before the impacts
// (04_code/src/interfaces.py). A: calibrated masses hung from the arm one at a time, each press says whether the
// joint held or slipped; a slip ends the repeat and its torque joins the result, live. B: the marker lifted and let
// go, then scanned, a run of the Scans page (kind reseat) that measures each scan as it comes; the error of each
// seating against the first, live. On the stage both results; in the panel the holder chosen and its few controls.
// Every press writes the same files the analysis reads. A plain script sharing one global scope; it lays its own
// view and panel into the page when it loads.
'use strict';

const HOLDERS_POLL_MS = 1000;       // the state read this often while the run of B is on

/** The Holders step of Trials. */
const HOLDERS = {
  state: null,
  which: 'a',
  timer: null,

  build: function () {
    const view = E('div', 'view narrowable');
    view.dataset.view = 'holders';
    view.innerHTML =                          // the app's own markup: words come from data-t, no data in it
      '<div class="sheet-col an-col"><div class="an-grid" id="hoGrid"></div></div>';
    document.querySelector('.stage').append(view);
    const panel = E('div', 'rp-page');
    panel.dataset.panel = 'holders';
    panel.innerHTML =
      '<div class="rp-hd"><div class="rp-title" data-t="hold.title"></div><div class="rp-sub" data-t="hold.sub"></div></div>'
      + '<div class="rp-scroll stack">'
      + '<div class="mlist" id="hoWhich">'
      + '<div class="mitem" data-which="a" tabindex="0" role="button"><span class="mitem-name" data-t="hold.a"></span></div>'
      + '<div class="mitem" data-which="b" tabindex="0" role="button"><span class="mitem-name" data-t="hold.b"></span></div>'
      + '</div>'
      + '<div class="stack" id="hoA">'
      + '<div class="note" id="hoARep"></div>'
      + '<div class="field" data-t-title="hold.arm.hint"><span class="field-n" data-t="hold.arm"></span>'
      + '<input class="input" type="number" id="hoArm" min="5" max="500" step="0.1"/><span class="input-unit">mm</span></div>'
      + '<div class="field" data-t-title="hold.mass.hint"><span class="field-n" data-t="hold.mass"></span>'
      + '<input class="input" type="number" id="hoMass" min="1" max="20000" step="0.1"/><span class="input-unit">g</span></div>'
      + '<div class="row ho-press">'
      + '<button class="btn btn-2 btn-grow" id="hoHeld" type="button" data-sound="confirm" data-t-title="hold.held.hint">'
      + '<span class="btn-label" data-t="hold.held"></span></button>'
      + '<button class="btn btn-primary btn-grow" id="hoSlip" type="button" data-sound="confirm" data-t-title="hold.slipped.hint">'
      + '<span class="btn-label" data-t="hold.slipped"></span></button>'
      + '</div>'
      + '<div class="fold" id="hoAdv"><button class="fold-hd" type="button"><span class="fold-name" data-t="word.advanced">'
      + '</span></button><div class="fold-body"><div class="stack">'
      + '<button class="btn btn-2 btn-wide" id="hoUndo" type="button"><span data-icon="reset"></span>'
      + '<span class="btn-label" data-t="hold.undo"></span></button>'
      + '</div></div></div>'
      + '</div>'
      + '<div class="stack hidden" id="hoB">'
      + '<div class="note" id="hoBSay"></div>'
      + '<div class="callout callout-warn hidden" id="hoBNoScan" data-t="hold.b.noscanner"></div>'
      + '<button class="btn btn-primary btn-wide" id="hoBRun" type="button" data-t-title="hold.b.start.hint">'
      + '<span data-icon="play"></span><span class="btn-label" id="hoBRunL"></span></button>'
      + '</div>'
      + '</div>';
    const soon = document.querySelector('.rp-page[data-panel="soon"]');
    soon.parentNode.insertBefore(panel, soon);
    Bridge.on('happened', function (raw) {
      let e = null;
      try { e = JSON.parse(raw); } catch (err) { return; }
      if (e && e.type === 'scans' && (e.kind === 'reseat' || e.action === 'captured') && HOLDERS.here()) HOLDERS.refresh();
    });
  },

  here: function () { return page === 'trials' && trialsView === 'holders'; },

  enter: function () {
    showView('holders');
    showPanel('holders');
    HOLDERS.wire();
    HOLDERS.refresh();
  },

  wire: function () {
    document.querySelectorAll('#hoWhich .mitem').forEach(function (it) {
      onActivate(it, function () { HOLDERS.which = it.dataset.which; playSound('select'); HOLDERS.render(); });
    });
    $('hoHeld').onclick = function () { HOLDERS.press(false); };
    $('hoSlip').onclick = function () { HOLDERS.press(true); };
    $('hoUndo').onclick = function () { ask('holders_a_undo', function (s) { if (s) HOLDERS.show(s); }); };
    $('hoBRun').onclick = function () {
      const on = HOLDERS.state && HOLDERS.state.b_on;
      ask('holders_b', on ? 'stop' : 'start', 0, function (s) { if (s) HOLDERS.show(s); });
    };
  },

  refresh: function () {
    askQuiet('holders_state', function (s) { if (s) HOLDERS.show(s); });
  },

  show: function (s) {
    HOLDERS.state = s;
    if (s.b_on && !HOLDERS.timer) {
      HOLDERS.timer = setInterval(function () {
        if (!HOLDERS.here()) { clearInterval(HOLDERS.timer); HOLDERS.timer = null; return; }
        HOLDERS.refresh();
      }, HOLDERS_POLL_MS);
    } else if (!s.b_on && HOLDERS.timer) {
      clearInterval(HOLDERS.timer);
      HOLDERS.timer = null;
    }
    if (HOLDERS.here()) HOLDERS.render();
  },

  // one mass tried: the mass and the arm as typed, held or slipped
  press: function (slipped) {
    const mass = $('hoMass').value, arm = $('hoArm').value;
    if (!(Number(mass) > 0) || !(Number(arm) > 0)) { toast(T('hold.err.value'), TOAST_ERROR_MS, 'error'); return; }
    ask('holders_a', JSON.stringify({ mass_g: mass, arm_mm: arm, slipped: slipped }), function (s) {
      if (s) HOLDERS.show(s);
    });
  },

  render: function () {
    const s = HOLDERS.state;
    if (!s) return;
    document.querySelectorAll('#hoWhich .mitem').forEach(function (it) { it.classList.toggle('sel', it.dataset.which === HOLDERS.which); });
    $('hoA').classList.toggle('hidden', HOLDERS.which !== 'a');
    $('hoB').classList.toggle('hidden', HOLDERS.which !== 'b');
    const a = s.a, b = s.b;
    $('hoARep').textContent = a.complete ? T('hold.a.complete', fmtNum(a.n, 0)) : T('hold.repeat', fmtNum(a.repeat, 0), fmtNum(a.of, 0));
    if (!$('hoArm').value && a.arm_mm) $('hoArm').value = String(a.arm_mm);
    if (!$('hoMass').value && a.last_mass_g) $('hoMass').value = String(a.last_mass_g);
    $('hoUndo').disabled = !a.last_mass_g;
    $('hoBNoScan').classList.toggle('hidden', !!s.scanner);
    $('hoBSay').textContent = T(b.scans ? 'hold.b.do' : 'hold.b.first');
    $('hoBRunL').textContent = T(s.b_on ? 'hold.b.stop' : 'hold.b.start');
    $('hoBRun').classList.toggle('btn-primary', !s.b_on);
    $('hoBRun').classList.toggle('btn-2', !!s.b_on);
    $('hoBRun').disabled = !s.scanner && !s.b_on;
    const grid = $('hoGrid');
    grid.replaceChildren(HOLDERS.cardA(a), HOLDERS.cardB(s));
  },

  cardA: function (a) {
    const rows = a.tried.map(function (t, i) {
      return [T('hold.try', i + 1), withUnit(t.mass_g, 1, 'g'), T(t.slipped ? 'hold.chip.slipped' : 'hold.chip.held')];
    });
    const c = ANALYSIS.card(T('hold.a') + ' · ' + T('hold.repeat', fmtNum(Math.min(a.repeat, a.of), 0), fmtNum(a.of, 0)),
                            ['', T('hold.mass'), ''], rows, rows.length ? '' : T('hold.none'));
    const t = E('div', 'tiles');
    const tile = function (label, value) {
      const x = E('div', 'tile-set');
      x.append(E('div', 'tile-l', label), E('div', 'tile-v', value));
      return x;
    };
    t.append(tile(T('hold.torque'), a.mean_nm === null ? '-' : withUnit(a.mean_nm, 3, 'N·m')),
             tile(T('hold.sd'), a.sd_nm === null ? '-' : fmtNum(a.sd_nm, 3)),
             tile(T('hold.done'), fmtNum(a.n, 0) + ' / ' + fmtNum(a.of, 0)));
    c.insertBefore(t, c.children[1] || null);
    if (a.done.length) {
      c.append(E('div', 'an-note', T('hold.done.list', a.done.map(function (d) { return fmtNum(d.torque_nm, 3); })
        .join(' · '))));
    }
    return c;
  },

  cardB: function (s) {
    const b = s.b;
    const rows = (b.files || []).slice().reverse().map(function (f) {
      return [f.name, T(f.ok ? 'hold.scan.ok' : 'hold.scan.bad')];
    });
    const c = ANALYSIS.card(T('hold.b'), ['', ''], rows, s.measuring ? T('hold.b.measuring') : '');
    const t = E('div', 'tiles');
    const tile = function (label, value) {
      const x = E('div', 'tile-set');
      x.append(E('div', 'tile-l', label), E('div', 'tile-v', value));
      return x;
    };
    t.append(tile(T('hold.b.seatings'), fmtNum(b.n, 0) + ' / ' + fmtNum(b.of, 0)),
             tile(T('hold.b.median'), b.median_mm === null ? '-' : ANALYSIS.mm(b.median_mm)),
             tile(T('hold.b.p95'), b.p95_mm === null ? '-' : ANALYSIS.mm(b.p95_mm)));
    c.insertBefore(t, c.children[1] || null);
    return c;
  },
};

HOLDERS.build();
