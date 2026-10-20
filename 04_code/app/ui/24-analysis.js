// ImpactGuide · Analysis, the second step of Scans and results: what the trials say, made again from the trials
// closed with their scans (04_code/src/analysis.py, in a worker of the backend) every time the step opens or Update
// is pressed. On the stage: the trials counted, the four conditions (A and B without correction, each after the AI
// step), the true displacement of the marker, one breakdown at a time (energy with its hole, direction, drape, the
// seven targets), the paired tests and the mixed model, detection, correction, how far the predicted error can be
// trusted, and the two holders measured on their own. Simulated data are said at the top, in words and in the
// accent of a warning. In the panel: the breakdown, Update; the folder of the tables under Advanced. The tables
// themselves go into the results folder (the project's, or the simulation's), only when they change.
// A plain script sharing one global scope; it lays its own view and panel into the page when it loads.
'use strict';

const ANALYSIS_POLL_MS = 600;       // the analysis asked how far it is, while it runs
const AN_CONDS = ['A', 'A_AI', 'B', 'B_AI'];
const AN_BY = ['energy', 'direction', 'drape', 'target'];

/** The Analysis step: its view, its panel, the last summary of the backend. */
const ANALYSIS = {
  state: null,        // analysis_state's answer
  by: 'energy',       // the breakdown shown
  timer: null,
  failed: '',         // the code of the last failure (no_trials: nothing to analyse yet)

  // ── the markup, laid once into the stage and the panel ─────────────────
  build: function () {
    const view = E('div', 'view narrowable');
    view.dataset.view = 'analysis';
    const col = E('div', 'sheet-col an-col');
    col.id = 'anCol';
    const sim = E('div', 'callout callout-warn hidden');
    sim.id = 'anSim';
    const say = E('div', 'say card hidden');
    say.id = 'anSay';
    const top = E('div', 'say-top');
    const spin = E('span', 'spinner spinner-sm hidden');
    spin.id = 'anSpin';
    const title = E('span', 'say-title');
    title.id = 'anSayT';
    top.append(spin, title);
    const line = E('div', 'say-line');
    line.id = 'anSayL';
    say.append(top, line);
    const tiles = E('div', 'tiles');
    tiles.id = 'anTiles';
    const grid = E('div', 'an-grid');
    grid.id = 'anGrid';
    col.append(sim, say, tiles, grid);
    view.append(col);
    document.querySelector('.stage').append(view);

    const panel = E('div', 'rp-page');
    panel.dataset.panel = 'analysis';
    panel.innerHTML =                         // the app's own markup: words come from data-t, no data in it
      '<div class="rp-hd"><div class="rp-title" data-t="page.analysis.title"></div>'
      + '<div class="rp-sub" data-t="page.analysis.sub"></div></div>'
      + '<div class="rp-scroll stack">'
      + '<div class="row"><span class="chip chip-warn hidden" id="anSimChip" data-t="an.sim.chip"></span></div>'
      + '<div id="anRows"></div>'
      + '<div class="sec-lbl" data-t="an.by.sec"></div>'
      + '<div class="mlist" id="anBy">'
      + AN_BY.map(function (k) {
        return '<div class="mitem" data-by="' + k + '" tabindex="0" role="button"><span class="mitem-name" data-t="an.by.'
          + k + '"></span></div>';
      }).join('')
      + '</div>'
      + '<div class="fold" id="anAdv"><button class="fold-hd" type="button"><span class="fold-name" data-t="word.advanced">'
      + '</span></button><div class="fold-body"><div class="stack">'
      + '<div class="note" id="anWritten"></div>'
      + '<button class="btn btn-2 btn-wide" id="anOpen" type="button" data-t-title="an.open.hint"><span data-icon="open">'
      + '</span><span class="btn-label" data-t="an.open"></span></button>'
      + '</div></div></div>'
      + '</div>'
      + '<div class="rp-foot"><button class="btn btn-primary btn-wide" id="anRun" type="button" data-t-title="an.run.hint">'
      + '<span data-icon="reset"></span><span class="btn-label" data-t="an.run"></span></button></div>';
    const soon = document.querySelector('.rp-page[data-panel="soon"]');
    soon.parentNode.insertBefore(panel, soon);
    Bridge.on('happened', function (raw) {
      let e = null;
      try { e = JSON.parse(raw); } catch (err) { return; }
      if (e && e.type === 'analysis') ANALYSIS.onEvent(e);
    });
  },

  here: function () { return page === 'results' && STEP.results === 'analysis'; },

  enter: function () {
    showView('analysis');
    showPanel('analysis');
    ANALYSIS.wire();
    ANALYSIS.refresh(true);
  },

  wire: function () {
    document.querySelectorAll('#anBy .mitem').forEach(function (it) {
      onActivate(it, function () {
        ANALYSIS.by = it.dataset.by;
        playSound('select');
        ANALYSIS.render();
      });
    });
    $('anRun').setAttribute('data-sound', 'confirm');
    $('anRun').onclick = function () { ANALYSIS.run(); };
    $('anOpen').onclick = function () { ask('results_open', 'results'); };
  },

  /** The state read again; auto: an analysis is started when there is none yet and none runs. */
  refresh: function (auto) {
    askQuiet('analysis_state', function (s) {
      if (!s) return;
      ANALYSIS.state = s;
      if (auto && !s.result && !s.running && !ANALYSIS.failed) { ANALYSIS.run(); return; }
      if (s.running) ANALYSIS.watch();
      if (ANALYSIS.here()) ANALYSIS.render();
    });
  },

  run: function () {
    ANALYSIS.failed = '';
    ask('analysis_run', function (a) {
      if (!a) return;
      workOn('analysis', 'an.pill');
      if (ANALYSIS.state) ANALYSIS.state.running = true;
      ANALYSIS.watch();
      if (ANALYSIS.here()) ANALYSIS.render();
    });
  },

  // how far the analysis is, while it runs (a scan measured for the first time takes seconds)
  watch: function () {
    if (ANALYSIS.timer) return;
    ANALYSIS.timer = setInterval(function () {
      askQuiet('analysis_state', function (s) {
        if (!s) return;
        ANALYSIS.state = s;
        if (!s.running) { clearInterval(ANALYSIS.timer); ANALYSIS.timer = null; }
        if (ANALYSIS.here()) ANALYSIS.render();
      });
    }, ANALYSIS_POLL_MS);
  },

  onEvent: function (e) {
    workOff('analysis');
    if (e.action === 'failed') {
      ANALYSIS.failed = String(e.error || 'analysis_failed');
      if (ANALYSIS.failed !== 'no_trials') toast(errorText(e, 'an.err.'), TOAST_ERROR_MS, 'error');
    } else if (e.action === 'ready') {
      ANALYSIS.failed = '';
      if (ANALYSIS.here()) toast(e.written && e.written.length ? T('an.saved', e.written.length) : T('an.same'));
    }
    ANALYSIS.refresh(false);
    if (typeof CHARTS !== 'undefined') CHARTS.analysed(e);
  },

  // ── numbers as the page writes them ─────────────────────────────────────
  mm: function (v) { return v === null || v === undefined ? '-' : withUnit(v, Math.abs(v) < 10 ? 2 : 1, 'mm'); },
  pct: function (v) { return v === null || v === undefined || v === '' ? '-' : withUnit(100 * v, 0, '%'); },
  p: function (v) {
    if (v === null || v === undefined || !isFinite(v)) return '-';
    return v < 0.001 ? '< ' + fmtNum(0.001, 3) : fmtNum(v, 3);
  },
  cond: function (c) { return c.replace('_AI', ' + AI'); },
  /** An interval: low–high, with spaces around the dash when a bound is negative (−1,07 – −0,04). */
  span: function (lo, hi, n) {
    if (lo === null || lo === undefined || lo === '' || hi === null || hi === undefined || hi === '') return '';
    return fmtNum(lo, n) + (Number(lo) < 0 || Number(hi) < 0 ? ' – ' : '–') + fmtNum(hi, n);
  },

  // ── the stage ───────────────────────────────────────────────────────────
  render: function () {
    const s = ANALYSIS.state || {};
    const r = s.result;
    document.querySelectorAll('#anBy .mitem').forEach(function (it) { it.classList.toggle('sel', it.dataset.by === ANALYSIS.by); });
    const simulated = !!(s.simulate || (r && r.simulated));
    $('anSim').classList.toggle('hidden', !simulated);
    $('anSim').textContent = T('an.sim');
    $('anSimChip').classList.toggle('hidden', !simulated);
    $('anRun').disabled = !!s.running;
    ANALYSIS.rows(s);
    const say = $('anSay');
    const waiting = s.running || (!r && !ANALYSIS.failed);
    say.classList.toggle('hidden', !!r && !s.running);
    $('anSpin').classList.toggle('hidden', !waiting);
    if (waiting) {
      $('anSayT').textContent = T('an.running');
      $('anSayL').textContent = s.progress ? T('an.progress', s.progress[0], s.progress[1]) : '';
    } else if (!r) {
      $('anSayT').textContent = T(ANALYSIS.failed === 'no_trials' ? 'an.none.title' : 'an.failed.title');
      $('anSayL').textContent = T(ANALYSIS.failed === 'no_trials' ? 'an.none.line' : 'err.generic.plain');
    }
    const grid = $('anGrid'), tiles = $('anTiles');
    grid.replaceChildren();
    tiles.replaceChildren();
    if (!r) return;
    ANALYSIS.tiles(r);
    grid.append(ANALYSIS.conditions(r), ANALYSIS.displacement(r), ANALYSIS.breakdown(r), ANALYSIS.tests(r),
                ANALYSIS.detection(r), ANALYSIS.correction(r), ANALYSIS.uncertainty(r), ANALYSIS.holders(r));
  },

  rows: function (s) {
    const host = $('anRows');
    host.replaceChildren();
    const r = s.result;
    host.append(piRow(T('cfg.phase'), Tcode('phase.', s.phase, 'campaign')));
    if (r) {
      host.append(piRow(T('an.row.trials'), fmtNum(r.n, 0)));
      if (r.missing) host.append(piRow(T('an.row.missing'), fmtNum(r.missing, 0), 'warn'));
    }
    const w = $('anWritten');
    w.textContent = r ? (r.written.length ? T('an.written', r.written.join(', ')) : T('an.same')) : '';
  },

  tiles: function (r) {
    const tile = function (label, value, title) {
      const t = E('div', 'tile-set');
      t.append(E('div', 'tile-l', label), E('div', 'tile-v', value));
      if (title) t.title = title;
      return t;
    };
    const n = r.n_holder || {};
    $('anTiles').append(tile(T('an.tile.trials'), fmtNum(r.n, 0)), tile('A', fmtNum(n.A || 0, 0), T('an.cond.A')),
                        tile('B', fmtNum(n.B || 0, 0), T('an.cond.B')), tile(T('an.tile.drape'), fmtNum(r.n_drape, 0)));
  },

  /** A card: a title and a grid of cells, the first column the row's name. head: the column heads. */
  card: function (title, head, rows, note) {
    const c = E('div', 'card col');
    c.append(E('div', 'sec-lbl', title));
    if (rows.length) {
      const t = E('div', 'an-t');
      t.style.gridTemplateColumns = 'minmax(0,1.3fr) repeat(' + (head.length - 1) + ',minmax(0,1fr))';
      head.forEach(function (h) {
        const el = E('div', 'an-h', typeof h === 'string' ? h : h[0]);
        if (typeof h !== 'string') el.title = h[1];
        t.append(el);
      });
      rows.forEach(function (row) {
        row.forEach(function (cell, i) {
          if (i === 0) {
            const l = E('div', 'an-l', typeof cell === 'string' ? cell : cell[0]);
            if (typeof cell !== 'string') l.title = cell[1];
            t.append(l);
            return;
          }
          const v = E('div', 'an-v');
          const parts = Array.isArray(cell) ? cell : [cell];
          v.append(document.createTextNode(parts[0]));
          if (parts[1]) v.append(E('small', null, parts[1]));
          t.append(v);
        });
      });
      c.append(t);
    }
    if (note) c.append(E('div', 'an-note', note));
    return c;
  },

  /** A median with its interquartile range under it. */
  miqr: function (x) {
    if (!x) return ['-'];
    return [ANALYSIS.mm(x.median), ANALYSIS.span(x.q1, x.q3, 2)];
  },

  condHead: function (conds) {
    return conds.map(function (c) { return [ANALYSIS.cond(c), T('an.cond.' + c)]; });
  },

  conditions: function (r) {
    const rows = r.conditions.map(function (x) {
      return [[ANALYSIS.cond(x.key), T('an.cond.' + x.key)], fmtNum(x.n, 0), ANALYSIS.miqr(x), ANALYSIS.pct(x.below_1mm),
              ANALYSIS.pct(x.below_2mm)];
    });
    return ANALYSIS.card(T('an.card.conditions'), ['', 'n', [T('an.h.tre'), T('an.h.tre.hint')], '≤ 1 mm', '≤ 2 mm'],
                         rows, r.ai ? '' : T('an.ai.wait'));
  },

  displacement: function (r) {
    const rows = r.displacement.map(function (d) {
      return [[d.holder, T('an.cond.' + d.holder)], [withUnit(d.rot_median, 2, '°'), T('an.max', withUnit(d.rot_max, 2, '°'))],
              [ANALYSIS.mm(d.shift_median), T('an.max', ANALYSIS.mm(d.shift_max))], ANALYSIS.pct(d.lost)];
    });
    return ANALYSIS.card(T('an.card.displacement'), ['', T('an.h.rot'), T('an.h.shift'), [T('an.h.lost'), T('an.h.lost.hint')]],
                         rows);
  },

  breakdown: function (r) {
    const by = ANALYSIS.by, b = (r.by || {})[by] || [];
    if (by === 'target') {
      const rows = b.map(function (x) {
        return [x.key, ANALYSIS.miqr(x.cells.A), ANALYSIS.miqr(x.cells.B)];
      });
      return ANALYSIS.card(T('an.card.by.target'), ['', ['A', T('an.cond.A')], ['B', T('an.cond.B')]], rows);
    }
    const conds = AN_CONDS.filter(function (c) { return r.conditions.some(function (x) { return x.key === c; }); });
    const name = function (x) {
      if (by === 'energy') {
        return ['E' + x.level + ' · ' + fmtNum(x.angle_deg, 0) + '°', withUnit(x.energy_mj, 1, 'mJ')];
      }
      if (by === 'direction') return fmtNum(x.level, 0) + '°';
      return T(x.level ? 'an.drape.with' : 'an.drape.without');
    };
    const rows = b.map(function (x) {
      return [name(x)].concat(conds.map(function (c) {
        const m = x.cells[c];
        return m ? [ANALYSIS.mm(m.median), 'n ' + fmtNum(m.n, 0)] : '-';
      }));
    });
    return ANALYSIS.card(T('an.card.by.' + by), [''].concat(ANALYSIS.condHead(conds)), rows,
                         by === 'drape' ? T('an.drape.note') : '');
  },

  tests: function (r) {
    const rows = r.tests.map(function (t) {
      const pair = t.key.split('_vs_');
      return [ANALYSIS.cond(pair[0]) + ' · ' + ANALYSIS.cond(pair[1]), fmtNum(t.n, 0), ANALYSIS.mm(t.diff), ANALYSIS.p(t.p_holm)];
    });
    const m = r.model || {};
    const term = function (k) { return (m.terms || []).find(function (t) { return t.key === k; }); };
    const model = [['energy_mj', 'an.m.energy', 'mm/mJ', 4], ['holder_b', 'an.m.holder', 'mm', 2],
                   ['energy_mj_x_holder_b', 'an.m.inter', 'mm/mJ', 4]].map(function (x) {
      const t = term(x[0]);
      return t ? [T(x[1]), '', [fmtNum(t.coef, x[3]) + ' ' + x[2], ANALYSIS.span(t.ci_low, t.ci_high, x[3])],
                  ANALYSIS.p(t.p)] : null;
    }).filter(Boolean);
    const note = m.method ? T(m.method === 'mixedlm' ? 'an.m.mixed' : 'an.m.ols', fmtNum(m.groups, 0)) : '';
    return ANALYSIS.card(T('an.card.tests'), ['', 'n', [T('an.h.diff'), T('an.h.diff.hint')], [T('an.h.p'), T('an.h.p.hint')]],
                         rows.concat(model), note);
  },

  detection: function (r) {
    if (!r.ai) return ANALYSIS.card(T('an.card.detection'), [], [], T('an.ai.wait'));
    const rows = r.detection.map(function (d) {
      return [d.key.indexOf('_drape') > 0 ? T('an.drape.with') : T('an.drape.without'),
              [fmtNum(d.auc, 3), ANALYSIS.span(d.auc_ci[0], d.auc_ci[1], 2)],
              ANALYSIS.pct(d.sensitivity), ANALYSIS.pct(d.specificity)];
    });
    return ANALYSIS.card(T('an.card.detection'), ['', 'AUC', [T('an.h.sens'), T('an.h.sens.hint')],
                                                  [T('an.h.spec'), T('an.h.spec.hint')]], rows);
  },

  correction: function (r) {
    if (!r.ai) return ANALYSIS.card(T('an.card.correction'), [], [], T('an.ai.wait'));
    const rows = r.correction.map(function (c) {
      const h = c.key.split('_')[0], drape = c.key.indexOf('_drape') > 0;
      return [[h + ' + AI' + (drape ? ' · ' + T('an.drape.short') : ''), T('an.cond.' + h + '_AI')],
              ANALYSIS.pct(c.below_2mm), ANALYSIS.pct(c.warnings), ANALYSIS.pct(c.worse), ANALYSIS.pct(c.safe)];
    });
    return ANALYSIS.card(T('an.card.correction'), ['', '≤ 2 mm', [T('an.h.warn'), T('an.h.warn.hint')],
                                                   [T('an.h.worse'), T('an.h.worse.hint')], [T('an.h.safe'), T('an.h.safe.hint')]],
                         rows, T('an.corr.note'));
  },

  uncertainty: function (r) {
    const u = r.uncertainty;
    if (!u) return ANALYSIS.card(T('an.card.uncertainty'), [], [], T(r.ai ? 'an.unc.wait' : 'an.ai.wait'));
    const rows = [];
    [['test', 'an.drape.without'], ['test_drape', 'an.drape.with']].forEach(function (x) {
      const g = u[x[0]];
      if (!g) return;
      rows.push([T(x[1]), fmtNum(g.n, 0), ANALYSIS.pct(g.accepted),
                 [ANALYSIS.pct(g.silent), T(g.within ? 'an.unc.ok' : 'an.unc.over')]]);
    });
    const limit = u.u === null || u.u === undefined ? T('an.unc.none') : ANALYSIS.mm(u.u);
    return ANALYSIS.card(T('an.card.uncertainty'), ['', 'n', [T('an.h.accepted'), T('an.h.accepted.hint')],
                                                    [T('an.h.silent'), T('an.h.silent.hint')]], rows,
                         T('an.unc.note', limit, fmtNum(100 * u.alpha, 0), fmtNum(u.n_validation, 0)));
  },

  holders: function (r) {
    const h = r.interfaces;
    if (!h) return ANALYSIS.card(T('an.card.holders'), [], [], T('an.hold.wait'));
    const rows = [];
    if (h.a) {
      rows.push([[T('an.hold.a'), T('an.hold.a.hint')], fmtNum(h.a.n, 0),
                 [withUnit(h.a.mean, 3, 'N·m'), '± ' + fmtNum(h.a.sd, 3)]]);
    }
    if (h.b) {
      rows.push([[T('an.hold.b'), T('an.hold.b.hint')], fmtNum(h.b.n, 0),
                 [ANALYSIS.mm(h.b.median), T('an.p95', ANALYSIS.mm(h.b.p95))]]);
    }
    const sim = (h.simulated_parts || []).map(function (k) { return k.toUpperCase(); }).join(', ');
    return ANALYSIS.card(T('an.card.holders'), ['', 'n', T('an.h.value')], rows, sim ? T('an.hold.sim', sim) : '');
  },
};

ANALYSIS.build();
