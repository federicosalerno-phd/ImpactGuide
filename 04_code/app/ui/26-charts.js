// ImpactGuide · Charts, the third step of Scans and results: a chart of the tables the analysis wrote, drawn by the
// backend in the shared figure style (04_code/src/chart_builder.py, style.py: Abadi, the shared colours, the page
// widths) and shown as the figure will be, a white sheet at its proportions. In the panel: the chart to start from
// (or one pinned before), its kind, the value, along what, the series, the page width; under Advanced the name, the
// height, the scale, the palette, the limits, the legend and the folders. Export writes PDF and SVG (vector) and PNG
// at 300 dpi; Pin writes the chart's own script into the figures folder and runs it there, so the figure no longer
// needs the app (a script of the same name is replaced only on a second press). Without the analysis's tables the
// step starts the analysis and waits for it. Simulated tables are said at the top and the figure carries the
// placeholder mark. A plain script sharing one global scope; it lays its own view and panel into the page when it
// loads.
'use strict';

const CHART_DEBOUNCE_MS = 140;      // a preview is asked this long after the last change
const CHART_ZOOM_MAX = 2.2;         // the sheet shown at most this many times its size at 96 dpi
const MM_PX = 96 / 25.4;

/** The Charts step: its view, its panel, the spec being made and its last preview. */
const CHARTS = {
  st: null,           // chart_state's answer
  spec: null,         // the chart being made (chart_builder.DEFAULT's keys)
  from: '',           // the entry of the first list it started from: p:<preset> or s:<pinned name>
  seq: 0,             // the id of the last preview asked
  shown: null,        // the last preview drawn
  timer: null,
  waitingAnalysis: false,
  triedAnalysis: false,

  build: function () {
    const view = E('div', 'view ch-view');
    view.dataset.view = 'charts';
    view.innerHTML =                          // the app's own markup: words come from data-t, no data in it
      '<div class="ch-col">'
      + '<div class="callout callout-warn hidden" id="chSim" data-t="ch.sim"></div>'
      + '<div class="say card hidden" id="chSay"><div class="say-top"><span class="spinner spinner-sm hidden" id="chSaySpin">'
      + '</span><span class="say-title" id="chSayT"></span></div><div class="say-line" id="chSayL"></div></div>'
      + '<div class="ch-paper hidden" id="chPaper"><img id="chImg" alt=""/><div class="ch-busy hidden" id="chBusy">'
      + '<span class="spinner spinner-sm"></span></div></div>'
      + '<div class="ch-cap" id="chCap"></div>'
      + '</div>';
    document.querySelector('.stage').append(view);

    const combo = function (id, label, hint) {
      return '<div class="stack ch-ctl" id="' + id + 'Row"><div class="sec-lbl" data-t="' + label + '"'
        + (hint ? ' data-t-title="' + hint + '"' : '') + '></div><div class="combo" id="' + id
        + '" tabindex="0" role="button"><span class="combo-v"></span></div></div>';
    };
    const panel = E('div', 'rp-page');
    panel.dataset.panel = 'charts';
    panel.innerHTML =
      '<div class="rp-hd"><div class="rp-title" data-t="page.charts.title"></div>'
      + '<div class="rp-sub" data-t="page.charts.sub"></div></div>'
      + '<div class="rp-scroll stack">'
      + combo('chFrom', 'ch.from', 'ch.from.hint') + combo('chKind', 'ch.kind') + combo('chY', 'ch.y')
      + combo('chX', 'ch.x') + combo('chSeries', 'ch.series') + combo('chWidth', 'ch.width', 'ch.width.hint')
      + '<div class="fold" id="chAdv"><button class="fold-hd" type="button"><span class="fold-name" data-t="word.advanced">'
      + '</span></button><div class="fold-body"><div class="stack">'
      + '<div class="sec-lbl" data-t="ch.name" data-t-title="ch.name.hint"></div>'
      + '<input class="input" type="text" id="chName" maxlength="60" spellcheck="false" data-t-title="ch.name.hint"/>'
      + '<div class="field" data-t-title="ch.height.hint"><span class="field-n" data-t="ch.height"></span>'
      + '<input class="input" type="number" id="chHeight" min="30" max="230" step="1"/><span class="input-unit">mm</span></div>'
      + combo('chScale', 'ch.scale') + combo('chPalette', 'ch.palette', 'ch.palette.hint')
      + '<div class="sec-lbl" data-t="ch.limits" data-t-title="ch.limits.hint"></div>'
      + '<input class="input" type="text" id="chLimits" maxlength="40" data-t-title="ch.limits.hint"/>'
      + combo('chLegend', 'ch.legend')
      + '<button class="btn btn-2 btn-wide" id="chOpenFig" type="button" data-t-title="ch.open.fig.hint"><span data-icon="open">'
      + '</span><span class="btn-label" data-t="ch.open.fig"></span></button>'
      + '<button class="btn btn-2 btn-wide" id="chOpenPic" type="button"><span data-icon="open"></span>'
      + '<span class="btn-label" data-t="ch.open.pic"></span></button>'
      + '</div></div></div>'
      + '</div>'
      + '<div class="rp-foot"><div class="rp-nav">'
      + '<button class="btn btn-2 btn-grow" id="chExport" type="button" data-t-title="ch.export.hint"><span class="btn-label"'
      + ' data-t="ch.export"></span></button>'
      + '<button class="btn btn-primary btn-grow" id="chPin" type="button" data-t-title="ch.pin.hint"><span class="btn-label"'
      + ' data-t="ch.pin"></span></button>'
      + '</div></div>';
    const soon = document.querySelector('.rp-page[data-panel="soon"]');
    soon.parentNode.insertBefore(panel, soon);
    Bridge.on('happened', function (raw) {
      let e = null;
      try { e = JSON.parse(raw); } catch (err) { return; }
      if (e && e.type === 'chart') CHARTS.onEvent(e);
    });
    window.addEventListener('resize', function () { if (CHARTS.here()) CHARTS.size(); });
  },

  here: function () { return page === 'results' && STEP.results === 'charts'; },

  enter: function () {
    showView('charts');
    showPanel('charts');
    CHARTS.wire();
    askQuiet('chart_state', function (st) {
      if (!st || !CHARTS.here()) return;
      CHARTS.st = st;
      $('chSim').classList.toggle('hidden', !st.simulate);
      // something to draw: a table of one of the charts to start from, or a chart pinned before
      const usable = st.presets.some(function (p) { return st.tables[p.spec.data]; })
        || st.saved.some(function (x) { return st.tables[x.spec.data]; });
      if (!usable) { CHARTS.noTables(st); return; }
      CHARTS.waitingAnalysis = false;
      $('chSay').classList.add('hidden');
      if (!CHARTS.spec || !st.tables[CHARTS.spec.data]) CHARTS.start(CHARTS.firstPreset());
      else CHARTS.fill();
      CHARTS.later(0);
    });
  },

  // no table yet: the analysis is started (once) and the step waits for it
  noTables: function (st) {
    $('chPaper').classList.add('hidden');
    $('chCap').textContent = '';
    $('chSay').classList.remove('hidden');
    const running = st.analysing || CHARTS.waitingAnalysis;
    const tried = st.analysis || CHARTS.triedAnalysis;     // an analysis ended and left nothing to draw
    $('chSaySpin').classList.toggle('hidden', !running);
    $('chSayT').textContent = T('ch.none.title');
    $('chSayL').textContent = T(tried && !running ? 'ch.none.done' : 'ch.none.line');
    ['chExport', 'chPin'].forEach(function (id) { $(id).disabled = true; });
    if (!tried && !running) {
      CHARTS.waitingAnalysis = true;
      $('chSaySpin').classList.remove('hidden');
      ask('analysis_run', function (a) { if (a) workOn('analysis', 'an.pill'); });
    }
  },

  /** The analysis ended (ANALYSIS.onEvent): the tables are read again. */
  analysed: function () {
    CHARTS.waitingAnalysis = false;
    CHARTS.triedAnalysis = true;          // asked once per session: a failed one is not asked again by itself
    if (CHARTS.here()) CHARTS.enter();
  },

  firstPreset: function () {
    const st = CHARTS.st;
    const p = st.presets.find(function (x) { return st.tables[x.spec.data]; });
    if (p) return 'p:' + p.key;
    const s = st.saved.find(function (x) { return st.tables[x.spec.data]; });
    return s ? 's:' + s.name : 'p:' + st.presets[0].key;
  },

  /** A chart to start from: a preset (p:key, with a free name asked for) or a pinned one (s:name, with its own). */
  start: function (from) {
    const st = CHARTS.st;
    CHARTS.from = from;
    if (from.indexOf('s:') === 0) {
      const s = st.saved.find(function (x) { return 's:' + x.name === from; });
      CHARTS.spec = JSON.parse(JSON.stringify(s.spec));
      CHARTS.spec.name = s.name;
    } else {
      const p = st.presets.find(function (x) { return 'p:' + x.key === from; }) || st.presets[0];
      CHARTS.spec = JSON.parse(JSON.stringify(p.spec));
      CHARTS.spec.name = '';
      askQuiet('chart_name', p.key, function (a) {
        if (a && CHARTS.spec && !CHARTS.spec.name && CHARTS.from === from) {
          CHARTS.spec.name = a.name;
          $('chName').value = a.name;
          CHARTS.caption();
        }
      });
    }
    CHARTS.fit();
    CHARTS.fill();
  },

  // the spec made to hold for the tables there are: its table, its value, x and series among their columns
  fit: function () {
    const s = CHARTS.spec, tables = CHARTS.st.tables;
    if (!tables[s.data]) {
      const name = Object.keys(tables).find(function (k) { return tables[k].values.indexOf(s.y) >= 0; })
        || Object.keys(tables)[0];
      s.data = name;
    }
    const t = tables[s.data];
    if (t.values.indexOf(s.y) < 0) s.y = t.values[0] || '';
    const xs = CHARTS.xOptions(s, t);
    if (s.x && xs.indexOf(s.x) < 0) s.x = '';
    if (!s.x && (s.kind === 'summary' || s.kind === 'share')) s.x = t.levels[0] || '';
    if (s.series && t.levels.indexOf(s.series) < 0) s.series = '';
  },

  xOptions: function (s, t) {
    return s.kind === 'points' ? t.levels.concat(t.values.filter(function (v) { return v !== s.y; })) : t.levels;
  },

  colName: function (c) { return c ? (hasT('col.' + c) ? T('col.' + c) : c) : T('ch.none'); },
  tableName: function (t) { const k = 'ch.table.' + t.replace(/\.csv$/, ''); return hasT(k) ? T(k) : t; },

  // ── the panel from the spec ────────────────────────────────────────────
  fill: function () {
    const st = CHARTS.st, s = CHARTS.spec, t = st.tables[s.data];
    const fromV = st.presets.filter(function (p) { return st.tables[p.spec.data]; }).map(function (p) { return 'p:' + p.key; })
      .concat(st.saved.map(function (x) { return 's:' + x.name; }));
    const fromL = fromV.map(function (v) {
      return v.indexOf('p:') === 0 ? T('ch.preset.' + v.slice(2)) || v.slice(2) : v.slice(2);
    });
    if (fromV.indexOf(CHARTS.from) < 0) { fromV.push(CHARTS.from); fromL.push(s.name || CHARTS.from.slice(2)); }
    setOptions($('chFrom'), fromL, fromV);
    $('chFrom').value = CHARTS.from;
    setOptions($('chKind'), st.kinds.map(function (k) { return T('ch.kind.' + k); }), st.kinds);
    $('chKind').value = s.kind;
    const ys = [], yl = [];
    Object.keys(st.tables).forEach(function (name) {
      st.tables[name].values.forEach(function (c) {
        ys.push(name + ':' + c);
        yl.push(CHARTS.colName(c) + ' · ' + CHARTS.tableName(name));
      });
    });
    setOptions($('chY'), yl, ys);
    $('chY').value = s.data + ':' + s.y;
    const xs = CHARTS.xOptions(s, t);
    const needX = s.kind === 'summary' || s.kind === 'share';
    setOptions($('chX'), (needX ? [] : ['']).concat(xs).map(CHARTS.colName), (needX ? [] : ['']).concat(xs));
    $('chX').value = s.x;
    $('chXRow').classList.toggle('hidden', s.kind === 'cumulative');
    setOptions($('chSeries'), [''].concat(t.levels).map(CHARTS.colName), [''].concat(t.levels));
    $('chSeries').value = s.series;
    setOptions($('chWidth'), st.widths.map(function (w) { return T('ch.width.' + w); }), st.widths.map(String));
    $('chWidth').value = String(s.width);
    $('chName').value = s.name || '';
    $('chHeight').value = String(s.height);
    setOptions($('chScale'), st.scales.map(function (k) { return T('ch.scale.' + k); }), st.scales);
    $('chScale').value = s.scale;
    setOptions($('chPalette'), st.palettes.map(function (k) { return T('ch.palette.' + k); }), st.palettes);
    $('chPalette').value = s.palette;
    $('chLimits').value = (s.limits || []).map(function (v) { return fmtNum(v, String(v).split('.')[1] ? String(v).split('.')[1].length : 0); }).join('; ');
    setOptions($('chLegend'), st.legends.map(function (k) { return T('ch.legend.' + k.replace(' ', '_')); }), st.legends);
    $('chLegend').value = s.legend;
    ['chExport', 'chPin'].forEach(function (id) { $(id).disabled = false; });
    CHARTS.caption();
  },

  /** The panel read into the spec after a change of the control `what`. */
  read: function (what) {
    const s = CHARTS.spec, st = CHARTS.st;
    if (what === 'chFrom') { CHARTS.start($('chFrom').value); CHARTS.later(0); return; }
    if (what === 'chKind') s.kind = $('chKind').value;
    if (what === 'chY') {
      const v = $('chY').value.split(':');
      if (v[0] !== s.data) { s.data = v[0]; s.x = ''; s.series = ''; }
      s.y = v[1];
      s.y_ticks = null;
    }
    if (what === 'chX') s.x = $('chX').value;
    if (what === 'chSeries') s.series = $('chSeries').value;
    if (what === 'chWidth') {
      const w = Number($('chWidth').value);
      if (Number(s.height) === st.default_height[String(s.width)]) s.height = st.default_height[String(w)];
      s.width = w;
    }
    if (what === 'chScale') { s.scale = $('chScale').value; s.y_ticks = null; }
    if (what === 'chPalette') s.palette = $('chPalette').value;
    if (what === 'chLegend') s.legend = $('chLegend').value;
    if (what === 'chHeight') {
      const h = Number(String($('chHeight').value).replace(',', '.'));
      if (isFinite(h) && h >= st.heights[0] && h <= st.heights[1]) s.height = Math.round(h * 10) / 10;
    }
    if (what === 'chLimits') {
      const v = String($('chLimits').value).split(/[;\s]+/).filter(Boolean).map(function (x) { return Number(x.replace(',', '.')); });
      if (v.every(isFinite) && v.length <= 4) s.limits = v;
    }
    if (what === 'chName') s.name = String($('chName').value).trim();
    CHARTS.fit();
    CHARTS.fill();
    CHARTS.later(CHART_DEBOUNCE_MS);
  },

  wire: function () {
    ['chFrom', 'chKind', 'chY', 'chX', 'chSeries', 'chWidth', 'chScale', 'chPalette', 'chLegend'].forEach(function (id) {
      $(id).onchange = function () { playSound('select'); CHARTS.read(id); };
    });
    ['chHeight', 'chLimits', 'chName'].forEach(function (id) {
      $(id).onchange = function () { CHARTS.read(id); };
    });
    $('chExport').onclick = function () { CHARTS.exportIt(); };
    $('chPin').setAttribute('data-sound', 'confirm');
    $('chPin').onclick = function () { CHARTS.pin(); };
    $('chOpenFig').onclick = function () { ask('results_open', 'figures'); };
    $('chOpenPic').onclick = function () { ask('results_open', 'pictures'); };
  },

  // ── the preview ─────────────────────────────────────────────────────────
  later: function (ms) {
    clearTimeout(CHARTS.timer);
    CHARTS.timer = setTimeout(CHARTS.preview, ms);
  },

  preview: function () {
    if (!CHARTS.spec || !CHARTS.here()) return;
    const id = ++CHARTS.seq;
    $('chBusy').classList.remove('hidden');
    ask('chart_preview', JSON.stringify({ spec: CHARTS.spec, id: id }));
  },

  onEvent: function (e) {
    if (e.action === 'preview' || (e.action === 'failed' && e.id)) {
      if (e.id !== CHARTS.seq) return;                // an older chart: the newer one is on its way
      $('chBusy').classList.add('hidden');
      if (e.action === 'failed') { toast(errorText(e, 'ch.err.'), TOAST_ERROR_MS, 'error'); return; }
      CHARTS.shown = e;
      $('chImg').src = e.svg;
      $('chPaper').classList.remove('hidden');
      CHARTS.size();
      CHARTS.caption();
      return;
    }
    if (e.action === 'exported') { workOff('chart_export'); toast(T('ch.exported', e.files.join(', ')), TOAST_SAID_MS); return; }
    if (e.action === 'pinned') {
      workOff('chart_pin');
      toast(T(e.replaced ? 'ch.pinned.again' : 'ch.pinned', e.script), TOAST_SAID_MS);
      CHARTS.from = 's:' + e.name;
      askQuiet('chart_state', function (st) { if (st) { CHARTS.st = st; if (CHARTS.here()) CHARTS.fill(); } });
      return;
    }
    if (e.action === 'failed') {
      workOff('chart_export');
      workOff('chart_pin');
      toast(errorText(e, 'ch.err.'), TOAST_ERROR_MS, 'error');
    }
  },

  // the sheet at its proportions, as large as the stage allows up to CHART_ZOOM_MAX times its 96 dpi size
  size: function () {
    const e = CHARTS.shown;
    if (!e) return;
    const col = document.querySelector('.ch-col');
    const room = Math.max(120, col.clientWidth - 2 * 14);
    const w = Math.min(room, e.width_mm * MM_PX * CHART_ZOOM_MAX);
    $('chImg').style.width = Math.round(w) + 'px';
  },

  caption: function () {
    const s = CHARTS.spec;
    if (!s) return;
    $('chCap').textContent = (s.name || '') + (s.name ? ' · ' : '') + fmtNum(s.width, 0) + ' × ' + fmtNum(s.height, 0) + ' mm';
  },

  // ── export and pin ──────────────────────────────────────────────────────
  exportIt: function () {
    if (!CHARTS.spec) return;
    ask('chart_export', JSON.stringify({ spec: CHARTS.spec }), function (a) {
      if (a && a.started) workOn('chart_export', 'ch.pill.export');
    });
  },

  pin: function () {
    const btn = $('chPin'), s = CHARTS.spec;
    if (!s) return;
    if (!s.name) { toast(T('ch.err.bad_spec'), TOAST_ERROR_MS, 'error'); return; }
    const replace = !!btn._armed;
    if (replace) armed(btn);                          // the second press: the button says Pin again
    askQuiet('chart_pin', JSON.stringify({ spec: s, replace: replace }), function (a, bad) {
      if (bad && bad.error === 'script_exists') { armed(btn, T('ch.pin.replace')); return; }
      if (bad) { toast(errorText(bad, 'ch.err.'), TOAST_ERROR_MS, 'error'); return; }
      if (a && a.started) workOn('chart_pin', 'ch.pill.pin');
    });
  },
};

CHARTS.build();
