// ImpactGuide · Dataset, the last step of Scans and results: what the dataset holds (the real trials with their
// photos, videos, scans and ground truth; the synthetic dataset of the twin with its sessions and the scene it was
// rendered on), whether it is whole (Check reads every file and compares its checksum) and the package to publish,
// written into a folder or a zip on this computer (dataset_pack.py). Nothing goes online from here.
// A plain script sharing one global scope. Its view and its panel are laid into the page here, once, before the
// page starts (ui/90-boot.js wires and speaks them with the rest).
'use strict';

const DATASET = {
  POLL_MS: 400,       // the state read this often while a check or an export runs
  NAMES_MAX: 3,       // names in the tooltip of a problem
  state: null,
  timer: null,
  // the checks of every file only Check makes: shown from its last run beside what the reading says
  DEEP: ['changed', 'unlisted', 'unreadable', 'bad_image', 'frames_missing'],

  // ── the markup: keys of words and ids, never data ──────────────────────────
  build: function () {
    const view = E('div', 'view ds-view');
    view.dataset.view = 'dataset';
    view.innerHTML = ''
      + '<div class="sheet-col">'
      + '  <div class="progress-card ds-job hidden" id="dsJob">'
      + '    <div class="progress-title" id="dsJobT"></div>'
      + '    <div class="progress-pct"><span class="progress-num" id="dsJobN">0</span><span class="progress-sym">%</span></div>'
      + '    <div class="progress-track"><div class="progress-fill" id="dsJobF"></div></div>'
      + '    <div class="progress-stage" id="dsJobS"></div>'
      + '  </div>'
      + '  <div class="say card ds-card">'
      + '    <div class="say-top"><span class="say-title grow" data-t="ds.real"></span><div class="stcells line" id="dsRealSt"></div></div>'
      + '    <div class="say-line" id="dsRealNone"></div>'
      + '    <div class="tiles" id="dsRealTiles"></div>'
      + '    <div class="say-list" id="dsRealList"></div>'
      + '  </div>'
      + '  <div class="say card ds-card">'
      + '    <div class="say-top"><span class="say-title grow" data-t="ds.synth"></span><div class="stcells line" id="dsSynthSt"></div></div>'
      + '    <div class="say-line" id="dsSynthNone"></div>'
      + '    <div class="tiles" id="dsSynthTiles"></div>'
      + '    <div class="say-list" id="dsSynthList"></div>'
      + '  </div>'
      + '  <div class="say card ds-card">'
      + '    <div class="say-top"><span class="say-title grow" data-t="ds.pack"></span>'
      + '      <button class="btn btn-2 btn-sm hidden" id="dsOpen" type="button" data-t-title="ds.open.hint"><span data-icon="open"></span><span class="btn-label" data-t="ds.open"></span></button></div>'
      + '    <div class="stcells line" id="dsLast"></div>'
      + '  </div>'
      + '</div>';
    const stage = document.querySelector('.stage');
    if (stage) stage.append(view);
    const panel = E('div', 'rp-page');
    panel.dataset.panel = 'dataset';
    panel.innerHTML = ''
      + '<div class="rp-hd"><div class="rp-title" data-t="page.dataset.title"></div><div class="rp-sub" data-t="ds.sub"></div></div>'
      + '<div class="rp-scroll stack">'
      + '  <button class="btn btn-2 btn-wide" id="dsCheck" type="button" data-t-title="ds.check.hint"><span data-icon="check"></span><span class="btn-label" data-t="ds.check"></span></button>'
      + '  <div class="sec-lbl" data-t="ds.export"></div>'
      + '  <button class="btn btn-primary btn-wide" id="dsFolder" type="button" data-t-title="ds.export.folder.hint"><span data-icon="export"></span><span class="btn-label" data-t="ds.export.folder"></span></button>'
      + '  <button class="btn btn-2 btn-wide" id="dsZip" type="button" data-t-title="ds.export.zip.hint"><span data-icon="save"></span><span class="btn-label" data-t="ds.export.zip"></span></button>'
      + '  <div class="fold" id="dsAdv"><button class="fold-hd" type="button"><span class="fold-name" data-t="word.advanced"></span></button><div class="fold-body"><div class="stack">'
      + '    <div class="sec-lbl" data-t="ds.include"></div>'
      + '    <label class="toggle" data-t-title="ds.inc.real.hint"><input type="checkbox" id="dsIncReal"/><span data-t="ds.real"></span></label>'
      + '    <label class="toggle" data-t-title="ds.inc.synth.hint"><input type="checkbox" id="dsIncSynth"/><span data-t="ds.synth"></span></label>'
      + '    <button class="btn btn-2 btn-wide" id="dsReread" type="button"><span data-icon="reset"></span><span class="btn-label" data-t="ds.reread"></span></button>'
      + '  </div></div></div>'
      + '</div>'
      + '<div class="rp-foot hidden" id="dsFoot"><button class="btn btn-danger btn-wide" id="dsStop" type="button"><span data-icon="stop"></span><span class="btn-label" data-t="ds.stop"></span></button></div>';
    const rp = document.querySelector('.rp');
    if (rp) rp.append(panel);
  },

  // ── the step ───────────────────────────────────────────────────────────────
  showing: function () { return page === 'results' && STEP.results === 'dataset'; },

  enter: function () {
    showView('dataset');
    showPanel('dataset');
    $('where').textContent = '';
    DATASET.ask(true);
  },

  // the state: fresh asks the backend to read the dataset again (in a worker: a `dataset` event when read)
  ask: function (fresh) {
    askQuiet('dataset_state', JSON.stringify({ fresh: !!fresh }), function (s) {
      if (s) DATASET.render(s);
    });
  },

  // read again and again while a check or an export runs, and only while the step is on show
  watch: function (on) {
    if (on && !DATASET.timer) {
      DATASET.timer = setInterval(function () {
        if (!DATASET.showing()) { DATASET.watch(false); return; }
        DATASET.ask(false);
      }, DATASET.POLL_MS);
    } else if (!on && DATASET.timer) {
      clearInterval(DATASET.timer);
      DATASET.timer = null;
    }
  },

  // ── words and numbers ──────────────────────────────────────────────────────
  size: function (b) {
    const n = Number(b) || 0;
    if (n >= 1e9) return withUnit(n / 1e9, 2, 'GB');
    if (n >= 1e6) return withUnit(n / 1e6, n >= 1e8 ? 0 : 1, 'MB');
    return withUnit(Math.max(n / 1e3, n ? 0.1 : 0), n >= 1e4 ? 0 : 1, 'kB');
  },

  // a cell of a quiet line: the dot (its shape says the state too), a label, a value
  cell: function (dot, label, value, hint) {
    const c = E('div', 'stcell');
    c.append(E('span', 'sd' + (dot ? ' ' + dot : '')), E('span', 'stcell-l', label));
    if (value !== undefined && value !== null && value !== '') c.append(E('span', 'stcell-v', value));
    if (hint) c.title = hint;
    return c;
  },

  verdict: function (host, present, lvl, reading) {
    host.replaceChildren();
    if (reading) { host.append(DATASET.cell('busy', T('ds.reading'))); return; }
    if (!present) { host.append(DATASET.cell('', T('ds.state.none'))); return; }
    host.append(DATASET.cell(lvl === 'ok' ? 'ok' : lvl, T('ds.state.' + (lvl === 'ok' ? 'ok' : lvl))));
  },

  tiles: function (host, list) {
    host.replaceChildren();
    list.forEach(function (x) {
      const t = E('div', 'tile-set');
      t.append(E('div', 'tile-l', x[0]), E('div', 'tile-v', x[1]), E('div', 'tile-c', x[2] || ''));
      if (x[3]) t.title = x[3];
      host.append(t);
    });
  },

  // what is wrong: one row each, a cross before it (a mark when it is only to know), the names in its tooltip
  problems: function (host, list) {
    host.replaceChildren();
    list.forEach(function (p, i) {
      const key = 'ds.p.' + p.code;
      const r = E('div', 'say-issue' + (p.level === 'err' ? '' : ' si-advice') + (i ? '' : ' si-first'));
      r.append(E('i'), E('span', null, hasT(key) ? T(key, fmtNum(p.n, 0)) : T('ds.p.other', fmtNum(p.n, 0))));
      const names = (p.names || []).slice(0, DATASET.NAMES_MAX);
      if (names.length) r.title = names.join(', ') + (p.n > names.length ? ' …' : '');
      host.append(r);
    });
  },

  // the problems of a part: what the reading says, and what only the last check could find
  merged: function (part, s) {
    const out = ((s.inventory || {})[part] || {}).problems || [];
    const last = s.check && s.check[part] ? s.check[part].problems || [] : [];
    const have = {};
    out.forEach(function (p) { have[p.code] = true; });
    return out.concat(last.filter(function (p) { return DATASET.DEEP.indexOf(p.code) >= 0 && !have[p.code]; }))
      .sort(function (a, b) { return (a.level === 'err' ? 0 : 1) - (b.level === 'err' ? 0 : 1); });
  },

  worst: function (list) {
    if (list.some(function (p) { return p.level === 'err'; })) return 'err';
    return list.length ? 'warn' : 'ok';
  },

  // ── the page ───────────────────────────────────────────────────────────────
  render: function (s) {
    DATASET.state = s;
    const inv = s.inventory, reading = !inv;
    const real = (inv && inv.real) || {}, syn = (inv && inv.synthetic) || {};
    // the real trials
    const rp = DATASET.merged('real', s);
    DATASET.verdict($('dsRealSt'), real.present, DATASET.worst(rp), reading);
    $('dsRealNone').textContent = !reading && !real.present ? T('ds.none.real') : '';
    $('dsRealTiles').classList.toggle('hidden', !real.present);
    if (real.present) {
      DATASET.tiles($('dsRealTiles'), [
        [T('ds.tile.trials'), fmtNum(real.trials, 0)],
        [T('ds.tile.photos'), fmtNum(real.photos, 0)],
        [T('ds.tile.videos'), fmtNum(real.videos, 0)],
        [T('ds.tile.scans'), fmtNum(real.scans, 0)],
        [T('ds.tile.truth'), fmtNum(real.truth, 0), T('ds.tile.of', fmtNum(real.scanned, 0)), T('ds.tile.truth.hint')],
        [T('ds.tile.size'), DATASET.size(real.bytes)]]);
    }
    DATASET.problems($('dsRealList'), real.present || rp.length ? rp : []);
    // the synthetic dataset
    const sp = DATASET.merged('synthetic', s);
    DATASET.verdict($('dsSynthSt'), syn.present, DATASET.worst(sp), reading);
    $('dsSynthNone').textContent = !reading && !syn.present ? T('ds.none.synth') : '';
    $('dsSynthTiles').classList.toggle('hidden', !syn.present);
    if (syn.present) {
      const sc = syn.scene || {}, split = syn.splits || {};
      const parts = ['train', 'val', 'test'].map(function (k) { return fmtNum((split[k] || {}).sessions || 0, 0); });
      DATASET.tiles($('dsSynthTiles'), [
        [T('ds.tile.sessions'), fmtNum(syn.sessions, 0)],
        [T('ds.tile.samples'), fmtNum(syn.samples, 0)],
        [T('ds.tile.images'), fmtNum(syn.images, 0)],
        [T('ds.tile.split'), parts.join(' · '), '', T('ds.tile.split.hint')],
        [T('ds.tile.scene'), Tcode('ds.scene.', sc.state, 'unknown'), sc.print || '', T('ds.scene.hint')],
        [T('ds.tile.size'), DATASET.size(syn.bytes)]]);
    }
    DATASET.problems($('dsSynthList'), sp);
    // the last check and the last package
    const last = $('dsLast');
    last.replaceChildren();
    const c = s.check, x = s.export;
    if (c) {
      const n = (c.real.problems || []).concat(c.synthetic.problems || []).reduce(function (a, p) { return a + p.n; }, 0);
      last.append(DATASET.cell(c.level === 'ok' ? 'ok' : c.level, T('ds.last.check'),
        (n ? T('ds.last.problems', fmtNum(n, 0)) : T('ds.last.ok')) + ' · ' + fmtTime(c.time)));
    } else {
      last.append(DATASET.cell('', T('ds.last.check'), T('ds.last.never')));
    }
    if (x) {
      last.append(DATASET.cell(x.level === 'err' ? 'err' : 'ok', T('ds.last.export'),
        x.name + ' · ' + DATASET.size(x.bytes) + ' · ' + fmtTime(x.time)));
    } else {
      last.append(DATASET.cell('', T('ds.last.export'), T('ds.last.never')));
    }
    $('dsOpen').classList.toggle('hidden', !(x && x.there));
    // the job on its way
    const j = s.job, running = !!j;
    $('dsJob').classList.toggle('hidden', !running);
    if (running) {
      const pct = Math.round((j.fraction || 0) * 100);
      $('dsJobT').textContent = T(j.kind === 'export' ? 'ds.job.export' : 'ds.job.check');
      $('dsJobN').textContent = fmtNum(pct, 0);
      $('dsJobF').style.transform = 'scaleX(' + (j.fraction || 0) + ')';
      $('dsJobS').textContent = j.stopping ? T('ds.job.stopping')
        : (j.stage === 'truth' ? T('ds.tile.truth')
          : T('ds.job.files', fmtNum(j.files_done, 0), fmtNum(j.files, 0))
            + (j.bytes ? ' · ' + DATASET.size(j.bytes_done) + ' / ' + DATASET.size(j.bytes) : ''));
    }
    // the panel
    const parts = s.parts || {};
    $('dsIncReal').checked = parts.real !== false;
    $('dsIncSynth').checked = parts.synthetic !== false;
    const some = (parts.real !== false && real.present) || (parts.synthetic !== false && syn.present);
    ['dsCheck', 'dsFolder', 'dsZip'].forEach(function (id) { $(id).disabled = running || reading; });
    if (!running && !reading && !some) { $('dsFolder').disabled = true; $('dsZip').disabled = true; }
    $('dsIncReal').disabled = running;
    $('dsIncSynth').disabled = running;
    $('dsFoot').classList.toggle('hidden', !running);
    $('dsStop').disabled = !!(j && j.stopping);
    DATASET.watch(running || (reading && s.reading));
    if (running) workOn('dataset', j.kind === 'export' ? 'ds.job.export' : 'ds.job.check');
    else workOff('dataset');
  },

  // ── what the operator does ─────────────────────────────────────────────────
  start: function (slot, arg) {
    const then = function (a, bad) {
      if (bad || !a || a.cancelled) return;
      DATASET.render(a);
    };
    if (arg === undefined) ask(slot, then); else ask(slot, arg, then);
  },

  setParts: function () {
    ask('dataset_parts', JSON.stringify({ real: $('dsIncReal').checked, synthetic: $('dsIncSynth').checked }),
      function (a) { if (a) DATASET.render(a); });
  },

  onEvent: function (e) {
    if (!e || e.type !== 'dataset') return;
    if (e.action === 'read') { if (DATASET.showing()) DATASET.ask(false); return; }
    workOff('dataset');
    if (e.action === 'checked') {
      const n = e.problems || 0;
      toast(n ? T('ds.done.check.bad', fmtNum(n, 0)) : T('ds.done.check.ok'), TOAST_SAID_MS, n ? 'error' : 'confirm');
    } else if (e.action === 'exported') {
      toast(T('ds.done.export'), TOAST_SAID_MS, 'confirm');
    } else if (e.action === 'stopped') {
      toast(T('ds.done.stopped'), TOAST_MS);
    } else if (e.action === 'failed') {
      toast(errorText(e, 'ds.err.'), TOAST_ERROR_MS, 'error');
    }
    if (DATASET.showing()) DATASET.ask(false);
  },

  wire: function () {
    $('dsCheck').onclick = function () { DATASET.start('dataset_check'); };
    $('dsFolder').onclick = function () { DATASET.start('dataset_export', JSON.stringify({ to: 'folder' })); };
    $('dsZip').onclick = function () { DATASET.start('dataset_export', JSON.stringify({ to: 'zip' })); };
    $('dsStop').onclick = function () { ask('dataset_stop', function (a) { if (a) DATASET.render(a); }); };
    $('dsOpen').onclick = function () { ask('dataset_open'); };
    $('dsReread').onclick = function () { DATASET.ask(true); };
    $('dsIncReal').onchange = DATASET.setParts;
    $('dsIncSynth').onchange = DATASET.setParts;
    if (DATASET.state && DATASET.showing()) DATASET.render(DATASET.state);   // the language changed
  },
};

DATASET.build();
