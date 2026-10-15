// ImpactGuide · the Trials page: the trial to do now, step by step, the log
// with its audit trail, the sequence and its configuration. The protocol is
// trials.py's (sequence, steps, names, log, completeness); the page only shows
// it and asks the backend, which also drives the phone and watches the scanner.

const TRIALS = {
  state: null,
  log: null,
  logSel: '',
  cfg: null,
  extraOpen: false,

  enter: function () {
    TRIALS.setView(trialsView);
    be('phone_refresh');
  },

  setView: function (v) {
    trialsView = v;
    showHome();
    document.querySelectorAll('.view-btn').forEach(function (b) { b.classList.toggle('on', b.dataset.view === v); });
    showView('trials-' + v);
    if (v === 'guide') TRIALS.refresh();
    else if (v === 'log') TRIALS.loadLog();
    else if (v === 'sequence') TRIALS.loadSeq();
    else TRIALS.loadCfg();
    if (v !== 'guide') showPanel('trials-' + v);
  },

  where: function (t) {
    $('where').textContent = T('trials.where', T('phase.' + (TRIALS.state ? TRIALS.state.phase : 'campaign')),
                               t.filled, t.slots) + (t.extra ? ' · ' + T('trials.where.extra', t.extra) : '');
  },

  refresh: function () {
    ask('trials_state', function (s) { if (s) TRIALS.render(s); });
  },

  // ── the guide ──────────────────────────────────────────────────────────
  base: function (key) { return key.replace(/_az\d+$/, ''); },
  az: function (key) { const m = /_az(\d+)$/.exec(key); return m ? m[1] : ''; },

  stepTitle: function (st) {
    const k = 'step.' + TRIALS.base(st.key) + '.title';
    return hasT(k) ? T(k, TRIALS.az(st.key)) : st.key;
  },

  stepText: function (st, s) {
    const t = s.trial, c = s.config, base = TRIALS.base(st.key), az = TRIALS.az(st.key);
    if (base === 'holder') return T('step.holder.text', t.holder);
    if (base === 'reposition') return T('step.reposition.text.' + (t.holder === 'A' ? 'A' : 'B'));
    if (base === 'pendulum') {
      const ch = t.changes || [];
      const part = function (field, key, v) { return T(key, v) + (ch.indexOf(field) >= 0 ? ' ' + T('step.pendulum.changed') : ''); };
      let x = T('step.pendulum.text', part('direction_deg', 'step.pendulum.dir', t.direction_deg),
                part('angle_deg', 'step.pendulum.hole', t.angle_deg),
                part('washers_per_side', 'step.pendulum.washers', t.washers_per_side));
      if (ch.indexOf('direction_deg') >= 0) x += ' ' + T('step.pendulum.turn');
      return x;
    }
    if (base === 'release') return T('step.release.text', t.angle_deg);
    if (base === 'video' || base === 'drape_video') {
      const a = c.video_az[0], b = c.video_az[1];
      return T('step.' + base + '.text', a, b, Math.round((b - a) / 8), c.photo_az[0]);
    }
    const k = 'step.' + base + '.text';
    return hasT(k) ? T(k, az) : st.text;
  },

  render: function (s) {
    TRIALS.state = s;
    TRIALS.where(s.totals);
    if (trialsView !== 'guide') return;
    const done = !!s.complete;
    $('guide').classList.toggle('hidden', done);
    $('guideDone').classList.toggle('hidden', !done);
    if (done) { showPanel('trials-plan'); TRIALS.renderPlan(s); return; }
    const t = s.trial;

    // the head
    $('gId').textContent = t.trial;
    $('gSub').textContent = t.slot ? T('trials.slot', t.slot, s.totals.slots) : T('trials.extra');
    const chips = $('gChips');
    chips.replaceChildren();
    if (t.open) chips.append(chip(T(t.resumed ? 'trials.chip.resumed' : 'trials.chip.open', quando(t.time_start)), 'accent'));
    const reasons = (t.reference || '').split(';').filter(Boolean);
    if (reasons.length) chips.append(chip(T('trials.chip.reference', reasons.map(function (r) { return T('ref.' + r); }).join(', '))));
    if (String(t.drape) === '1') chips.append(chip(T('trials.chip.drape'), 'warn'));

    // the settings
    const tiles = $('gTiles');
    tiles.replaceChildren();
    const ch = t.changes || [];
    [['holder', 'trials.tile.holder', t.holder],
     ['angle_deg', 'trials.tile.hole', t.angle_deg + '°'],
     ['direction_deg', 'trials.tile.direction', t.direction_deg + '°'],
     ['washers_per_side', 'trials.tile.washers', t.washers_per_side],
     ['drape', 'trials.tile.drape', T(String(t.drape) === '1' ? 'word.yes' : 'word.no')]].forEach(function (x) {
      const on = ch.indexOf(x[0]) >= 0 && t.prev;
      const tile = E('div', 'tile-set' + (on ? ' changed' : ''));
      tile.append(E('div', 'tile-l', T(x[1])), E('div', 'tile-v', x[2]), E('div', 'tile-c', on ? T('trials.tile.changed') : ''));
      tiles.append(tile);
    });

    // what to watch out for
    const warn = $('gWarn');
    warn.replaceChildren();
    if (reasons.indexOf('reposition') >= 0) {
      warn.append(E('div', 'callout callout-warn', T('trials.warn.reposition.' + (t.holder === 'A' ? 'A' : 'B'),
        t.prev ? t.prev.trial : '', t.prev && t.prev.cap_shift_mm ? cifre(t.prev.cap_shift_mm, 1) : '?', cifre(s.config.reposition_mm, 1))));
    }
    (t.warnings || []).forEach(function (w) {
      warn.append(E('div', 'callout callout-warn', /unknown/.test(w) && t.prev ? T('trials.warn.capunknown', t.prev.trial) : w));
    });
    if (String(t.drape) === '1') warn.append(E('div', 'callout callout-accent', T('trials.warn.drape')));
    if (!t.open && t.extra) warn.append(E('div', 'callout callout-accent', T('trials.warn.extra')));

    // the step to do now
    const steps = s.steps, i = s.step, st = steps[i];
    const prev = $('gPrev');
    prev.replaceChildren();
    prev.classList.add('hidden');
    if (t.open) {
      $('gNowN').textContent = T('trials.now.n', i + 1, steps.length);
      $('gNowT').textContent = TRIALS.stepTitle(st);
      $('gNowX').textContent = TRIALS.stepText(st, s);
      const staged = s.staged[st.key];
      $('gNowT').title = st.files.join(', ');
      if (staged) TRIALS.preview(prev, staged);
      else if (st.done && st.files.length) {
        prev.append(piRow(T('trials.file.kept'), st.files[0]));
        prev.classList.remove('hidden');
      }
    } else {
      $('gNowN').textContent = T('trials.now.plan');
      $('gNowT').textContent = T('trials.now.plan.t', t.trial);
      $('gNowT').title = '';
      const files = steps.reduce(function (n, x) { return n + x.files.length; }, 0);
      $('gNowX').textContent = T('trials.now.plan.x', steps.length, files);
    }

    // all the steps
    const list = $('gSteps');
    list.replaceChildren();
    steps.forEach(function (x, k) {
      const now = t.open && k === i;
      const it = E('div', 'mitem ' + (now ? 'sel' : (x.done ? 'done' : 'todo')));
      const name = E('span', 'mitem-name');
      name.append(E('span', 'blk-n', k + 1), E('span', null, TRIALS.stepTitle(x)));
      if (x.files.length) it.title = x.files.join(', ');
      it.append(name);
      if (now || x.done) it.append(E('span', 'mbadge', T(now ? 'trials.badge.now' : 'trials.badge.done')));
      if (t.open) it.onclick = function () { ask('step_go', k, function (a) { if (a) TRIALS.render(a); }); };
      list.append(it);
    });
    const sel = list.querySelector('.mitem.sel');
    if (sel && TRIALS._lastKey !== (t.trial + st.key)) {
      TRIALS._lastKey = t.trial + st.key;
      sel.scrollIntoView({ block: 'center' });
    }

    if (t.open) { showPanel('trials-step'); TRIALS.renderStep(s); }
    else { showPanel('trials-plan'); TRIALS.renderPlan(s); }
  },

  preview: function (host, staged) {
    host.classList.remove('hidden');
    if (staged.kind === 'photo') {
      const img = E('img');
      img.src = staged.url + '?t=' + Date.now();
      img.alt = staged.name;
      host.append(img);
    }
    host.append(piRow(T('trials.file.staged'), staged.name),
                piRow(T('trials.file.size'), cifre(staged.size / 1048576, 1) + ' MB'));
    (staged.problems || []).forEach(function (p) { host.append(E('div', 'callout callout-warn', p)); });
  },

  renderPlan: function (s) {
    const t = s.trial;
    if (!t) {
      $('planSub').textContent = '';
      $('planStart').disabled = true;
      $('planStartL').textContent = T('trials.plan.start', '');
      return;
    }
    $('planSub').textContent = t.slot ? '' : T('trials.extra');
    $('planRef').checked = !!s.options.reference;
    if (document.activeElement !== $('planMass')) $('planMass').value = t.swing_mass_g || '';
    $('planMass').placeholder = T('trials.plan.mass.ph');
    $('planStart').disabled = false;
    $('planStartL').textContent = T('trials.plan.start', t.trial);
    $('planScanner').textContent = s.scanner_folder || T('trials.scanner.none');
    $('planScannerBtn').classList.toggle('hidden', !!s.simulate);
    $('extraForm').classList.toggle('hidden', !TRIALS.extraOpen);
    $('extraBtn').classList.toggle('hidden', TRIALS.extraOpen);
    const c = s.config;
    setOptions($('exHolder'), c.holders.map(function (h) { return T('trials.tile.holder') + ' ' + h; }), c.holders);
    setOptions($('exAngle'), c.angles_deg.map(function (a) { return T('trials.tile.hole') + ' ' + a + '°'; }), c.angles_deg.map(String));
    setOptions($('exDir'), c.directions_deg.map(function (d) { return T('trials.tile.direction') + ' ' + d + '°'; }), c.directions_deg.map(String));
    if (!TRIALS.extraOpen) {
      $('exHolder').value = t.holder;
      $('exAngle').value = String(t.angle_deg);
      $('exDir').value = String(t.direction_deg);
      $('exDrape').checked = String(t.drape) === '1';
    }
  },

  options: function (settings) {
    const o = { reference: $('planRef').checked, settings: settings === undefined ? (TRIALS.state.options.settings || null) : settings };
    be('trial_options', JSON.stringify(o), function (raw) {
      const a = JSON.parse(raw);
      if (!failed(a)) TRIALS.render(a);
    });
  },

  start: function () {
    const mass = $('planMass').value.trim();
    be('trial_start', JSON.stringify({ swing_mass_g: mass }), function (raw) {
      const a = JSON.parse(raw);
      if (a && a.error === 'swing_mass') { toast(T('err.swing_mass')); $('planMass').focus(); return; }
      if (failed(a)) return;
      TRIALS.extraOpen = false;
      TRIALS.render(a);
      toast(T('trials.started', a.trial.trial));
    });
  },

  renderStep: function (s) {
    const st = s.steps[s.step], t = s.trial;
    const staged = s.staged[st.key];
    const kind = st.kind;
    $('stTitle').textContent = T('trials.now.n', s.step + 1, s.steps.length);
    $('stSub').textContent = t.trial;
    // in plain view what this step asks for; importing a file by hand, the scanner folder and voiding under Advanced
    const show = [kind].concat(staged ? ['staged'] : []);
    document.querySelectorAll('#stBody [data-kind]').forEach(function (b) {
      b.classList.toggle('hidden', show.indexOf(b.dataset.kind) < 0);
    });
    $('stImport').classList.toggle('hidden', ['photo', 'video', 'scan'].indexOf(kind) < 0 || !!staged || st.done);
    $('stImportHs').disabled = !!staged || st.done;
    const recording = s.recording === st.key;
    $('stRecL').textContent = T(recording ? 'step.ctl.stop' : 'step.ctl.record');
    $('stRec').classList.toggle('on', recording);
    $('stShoot').disabled = !!staged || st.done;
    $('stRec').disabled = (!!staged || st.done) && !recording;
    const fileStep = st.files.length > 0;
    let label = 'step.ok.confirm';
    if (staged) label = 'step.ok.keep';
    else if (kind === 'close') label = 'step.ok.close';
    else if (kind === 'release') label = 'step.ok.release';
    $('stOkL').textContent = T(label);
    $('stOk').disabled = fileStep && !staged && !st.done;
    $('stBack').disabled = s.step === 0;
    $('stScanner').textContent = s.scanner_folder || T('trials.scanner.none');
    $('stScannerBtn').classList.toggle('hidden', !!s.simulate || !!s.scanner_folder);
    $('stScannerBtn2').classList.toggle('hidden', !!s.simulate);
    const waiting = kind === 'scan' && !staged && !st.done;
    $('stWait').classList.toggle('hidden', !waiting || !s.scanner_folder);
    if (waiting && s.scanner_folder && s.watching !== st.key) ask('scan_watch', st.key);
    if (kind === 'spheres' && TRIALS._spheresFor !== t.trial) {
      TRIALS._spheresFor = t.trial;
      $('stSpheres').checked = t.spheres_ok !== '0';
      $('stShift').value = t.cap_shift_mm || '';
    }
    if (kind === 'close' && TRIALS._closeFor !== t.trial) {
      TRIALS._closeFor = t.trial;
      $('stSecond').checked = false;
      $('stCap').checked = false;
      $('stNotes').value = '';
    }
  },

  confirm: function () {
    const s = TRIALS.state, st = s.steps[s.step];
    let values = {};
    if (st.kind === 'spheres') values = { spheres_ok: $('stSpheres').checked ? '1' : '0', cap_shift_mm: $('stShift').value.trim() };
    if (st.kind === 'close') values = { second_hit: $('stSecond').checked ? '1' : '0', cap_replaced: $('stCap').checked ? '1' : '0',
                                        notes: $('stNotes').value.trim() };
    $('stOk').disabled = true;
    be('step_confirm', st.key, JSON.stringify(values), function (raw) {
      const a = JSON.parse(raw);
      if (failed(a)) { TRIALS.refresh(); return; }
      if (a.closed) toast(T('trials.closed', a.closed), 3500);
      TRIALS.render(a);
    });
  },

  onEvent: function (e) {
    if (e.type === 'staged' || e.type === 'recording') {
      work(false);
      if (e.type === 'recording') work(true, T('pill.recording'));
      if (page === 'trials') TRIALS.refresh();
    } else if (e.type === 'acquire_failed' && e.key !== 'calibration') {
      work(false);
      if (page === 'trials') TRIALS.refresh();
    }
  },

  // ── the log ────────────────────────────────────────────────────────────
  colName: function (c) { return hasT('col.' + c) ? T('col.' + c) : c; },

  // A logged value as it reads: yes or no, a time, a unit after the number.
  colValue: function (c, v) {
    if (v === '') return '';
    if (c.unit === '0/1') return T(v === '1' ? 'word.yes' : 'word.no');
    if (c.unit === 'ISO 8601') return v.replace('T', ' ');
    if (c.unit === 'deg') return v + '°';
    if (c.unit === 'mm' || c.unit === 'g') return v + ' ' + c.unit;
    if (c.name === 'status') return T('status.' + v);
    if (c.name === 'phase') return T('phase.' + v);
    if (c.name === 'reference') return v.split(';').filter(Boolean).map(function (r) { return T('ref.' + r); }).join(', ');
    return v;
  },

  loadLog: function () {
    ask('log_rows', function (L) { if (L) TRIALS.renderLog(L); });
  },

  renderLog: function (L) {
    TRIALS.log = L;
    if (TRIALS.state) TRIALS.where(TRIALS.state.totals);
    const list = $('logList');
    list.replaceChildren();
    const edited = {};
    L.edits.forEach(function (e) { edited[e.trial] = (edited[e.trial] || 0) + 1; });
    const rows = L.rows.slice().reverse();
    if (!TRIALS.logSel && rows.length) TRIALS.logSel = rows[0].trial;
    rows.forEach(function (r) {
      const it = E('div', 'mitem' + (r.trial === TRIALS.logSel ? ' sel' : ''));
      const name = E('span', 'mitem-name', r.trial + ' ');
      let det = r.holder + ' · ' + T('seq.cell', r.angle_deg, r.direction_deg) + (r.drape === '1' ? ' · ' + T('trials.chip.drape') : '')
        + ' · ' + quando(r.time_impact || r.time_start);
      if (r.cap_shift_mm) det += ' · ' + T('log.cap', cifre(r.cap_shift_mm, 2));
      if (edited[r.trial]) det += ' · ' + T('log.edited', edited[r.trial]);
      name.append(E('span', 'mitem-det', det));
      it.append(name, E('span', 'mbadge', T('status.' + r.status)));
      it.onclick = function () { TRIALS.logSel = r.trial; TRIALS.renderLog(TRIALS.log); };
      list.append(it);
    });
    $('logNote').textContent = rows.length ? '' : T('log.none');

    // the panel: the selected trial
    const r = L.rows.find(function (x) { return x.trial === TRIALS.logSel; });
    $('lgTitle').textContent = r ? r.trial : T('trials.view.log');
    $('lgSub').textContent = r ? T('status.' + r.status) + ' · ' + r.session : T('log.none');
    const fields = $('lgFields');
    fields.replaceChildren();
    const editable = L.columns.filter(function (c) { return c.name !== 'trial' && c.name !== 'phase'; });
    if (r) {
      L.columns.forEach(function (c) {
        if (r[c.name] !== '') fields.append(piRow(TRIALS.colName(c.name), TRIALS.colValue(c, r[c.name])));
      });
    }
    setOptions($('lgField'), editable.map(function (c) { return TRIALS.colName(c.name); }), editable.map(function (c) { return c.name; }));
    const canVoid = !!r && r.status !== 'void';
    $('lgVoid').classList.toggle('hidden', !canVoid);
    $('lgVoidReason').classList.toggle('hidden', !canVoid);
    $('lgVoidLbl').classList.toggle('hidden', !canVoid);
    ['lgField', 'lgValue', 'lgReason', 'lgSave', 'lgOpen'].forEach(function (id) {
      const el = $(id);
      if (el.tagName === 'BUTTON' || el.tagName === 'INPUT') el.disabled = !r;
    });
    const ed = $('lgEdits');
    ed.replaceChildren();
    L.edits.filter(function (e) { return r && e.trial === r.trial; }).reverse().forEach(function (e) {
      const card = E('div', 'edit-card', TRIALS.colName(e.field) + ': ' + (e.old || '∅') + ' → ' + (e.new || '∅'));
      card.append(E('span', 'aside', quando(e.time) + ' · ' + e.reason));
      ed.append(card);
    });
    if (r && !ed.children.length) ed.append(E('div', 'note', T('log.edits.none')));
  },

  saveEdit: function () {
    const field = $('lgField').value, value = $('lgValue').value.trim(), reason = $('lgReason').value.trim();
    if (!reason) { toast(T('err.reason')); $('lgReason').focus(); return; }
    be('log_edit', TRIALS.logSel, field, value, reason, function (raw) {
      const L = JSON.parse(raw);
      if (failed(L)) return;
      $('lgValue').value = '';
      $('lgReason').value = '';
      TRIALS.renderLog(L);
      toast(T('log.saved', TRIALS.colName(field)));
      ask('trials_state', function (s) { if (s) { TRIALS.state = s; TRIALS.where(s.totals); } });
    });
  },

  // ── the sequence ───────────────────────────────────────────────────────
  loadSeq: function () {
    ask('sequence_rows', function (Q) { if (Q) TRIALS.renderSeq(Q); });
  },

  renderSeq: function (Q) {
    const grid = $('seqList');
    grid.replaceChildren();
    Q.rows.forEach(function (r) {
      const next = String(r.slot) === String(Q.next);
      const cell = E('div', 'seq-cell' + (r.trial ? ' filled' : '') + (next ? ' next' : '') + (r.drape === '1' ? ' drape' : ''));
      cell.append(E('div', 'seq-n', r.slot + ' · ' + r.holder + ' · R' + r.replicate + (r.drape === '1' ? ' · ' + T('trials.chip.drape') : '')),
                  E('div', 'seq-v', T('seq.cell', r.angle_deg, r.direction_deg)),
                  E('div', 'seq-t', r.trial || (next ? T('trials.badge.next') : '')));
      grid.append(cell);
    });
    const t = Q.totals;
    const host = $('seqTotals');
    host.replaceChildren(
      piRow(T('seq.slots'), t.slots), piRow(T('seq.filled'), t.filled), piRow(T('seq.pending'), t.pending),
      piRow(T('seq.void'), t.void), piRow(T('seq.extra'), t.extra));
    if (Q.extra.length) host.append(E('div', 'note', T('seq.extra.list', Q.extra.join(', '))));
    TRIALS.where(t);
  },

  // ── the configuration ──────────────────────────────────────────────────
  loadCfg: function () {
    ask('config_get', function (C) { if (C) TRIALS.renderCfg(C); });
  },

  fillCfg: function (c) {
    $('cfgHolders').value = c.holders.join(', ');
    $('cfgAngles').value = c.angles_deg.join(', ');
    $('cfgDirections').value = c.directions_deg.join(', ');
    $('cfgReps').value = c.repetitions;
    $('cfgDrape').value = c.drape_trials;
    $('cfgOrder').value = c.order;
    $('cfgRun').value = c.direction_run;
    $('cfgSeed').value = c.seed;
    $('cfgWashers').value = c.washers_per_side;
    $('cfgRepos').value = c.reposition_mm;
    $('cfgHigh').checked = !!c.highspeed;
    TRIALS.cfgPreview();
  },

  list: function (id) { return $(id).value.split(/[\s,;]+/).filter(Boolean); },

  readCfg: function () {
    return {
      holders: TRIALS.list('cfgHolders'), angles_deg: TRIALS.list('cfgAngles').map(Number),
      directions_deg: TRIALS.list('cfgDirections').map(Number), repetitions: Number($('cfgReps').value),
      drape_trials: Number($('cfgDrape').value), order: $('cfgOrder').value, direction_run: Number($('cfgRun').value),
      seed: Number($('cfgSeed').value), washers_per_side: Number($('cfgWashers').value),
      reposition_mm: Number($('cfgRepos').value), highspeed: $('cfgHigh').checked,
    };
  },

  cfgPreview: function () {
    const c = TRIALS.readCfg();
    const main = c.holders.length * c.angles_deg.length * c.directions_deg.length * (c.repetitions || 0);
    $('cfgPreview').textContent = T('cfg.preview', main + (c.drape_trials || 0), main, c.drape_trials || 0);
  },

  renderCfg: function (C) {
    TRIALS.cfg = C;
    $('cfgPhase').value = C.phase;
    TRIALS.fillCfg(C.config);
    const t = C.totals;
    $('cfgTotals').replaceChildren(
      piRow(T('cfg.t.phase'), T('phase.' + C.phase)), piRow(T('seq.slots'), t.slots),
      piRow(T('seq.filled'), t.filled), piRow(T('seq.pending'), t.pending),
      piRow(T('cfg.t.written'), quando(C.written)));
    const h = $('cfgHistory');
    h.replaceChildren();
    C.history.slice().reverse().forEach(function (x) {
      const c = x.config;
      const card = E('div', 'edit-card', T('cfg.history.item', c.holders.join('+'), c.angles_deg.length, c.directions_deg.length,
                                             c.repetitions, c.drape_trials, c.seed));
      card.append(E('span', 'aside', T('cfg.history.until', quando(x.written))));
      h.append(card);
    });
    if (!C.history.length) h.append(E('div', 'note', T('cfg.history.none')));
    TRIALS.where(t);
  },

  applyCfg: function () {
    be('config_apply', JSON.stringify(TRIALS.readCfg()), function (raw) {
      const C = JSON.parse(raw);
      if (failed(C)) return;
      TRIALS.renderCfg(C);
      toast(T('cfg.applied', C.totals.slots, C.totals.pending), 3500);
    });
  },

  runCheck: function () {
    ask('check_phase', $('chkDeep').checked, function (R) {
      if (!R) return;
      const lines = [];
      R.trials.forEach(function (t) {
        if (t.problems.length) lines.push(t.trial + ' (' + T('status.' + t.status) + '): '
          + t.problems.map(function (p) { return T('problem.' + p[0]) + ' ' + p[1]; }).join('; '));
      });
      lines.push(T('cfg.check.summary', R.trials.length, lines.length, R.pending, R.extra.join(', ') || '-', R.orphans.join(', ') || '-'));
      $('chkLog').textContent = lines.join('\n');
    });
  },

  wire: function () {
    document.querySelectorAll('.view-btn').forEach(function (b) { b.onclick = function () { TRIALS.setView(b.dataset.view); }; });
    $('planRef').onchange = function () { TRIALS.options(); };
    $('planStart').onclick = TRIALS.start;
    $('extraBtn').onclick = function () { TRIALS.extraOpen = true; TRIALS.renderPlan(TRIALS.state); };
    $('exCancel').onclick = function () { TRIALS.extraOpen = false; TRIALS.options(null); };
    $('exApply').onclick = function () {
      TRIALS.options({ holder: $('exHolder').value, angle_deg: $('exAngle').value, direction_deg: $('exDir').value,
                       drape: $('exDrape').checked ? 1 : 0 });
    };
    const chooseFolder = function () { ask('choose_scanner_folder', function (s) { if (s) TRIALS.render(s); }); };
    $('planScannerBtn').onclick = chooseFolder;
    $('stScannerBtn').onclick = chooseFolder;
    $('stScannerBtn2').onclick = chooseFolder;
    $('stShoot').onclick = function () {
      const st = TRIALS.state.steps[TRIALS.state.step];
      ask('step_shoot', st.key, function (a) { if (a) work(true, T('pill.shooting')); });
    };
    $('stRec').onclick = function () {
      const s = TRIALS.state, st = s.steps[s.step];
      const stop = s.recording === st.key;
      ask('step_video', st.key, stop ? 'stop' : 'start', function (a) {
        if (!a) return;
        if (stop) work(false);  // the recording ends, the transfer starts
        work(true, T(stop ? 'pill.pulling' : 'pill.starting'));
      });
    };
    $('stImport').onclick = function () {
      const st = TRIALS.state.steps[TRIALS.state.step];
      ask('step_import', st.key, function (a) { if (a && a.trial) TRIALS.render(a); });
    };
    $('stImportHs').onclick = $('stImport').onclick;
    $('stDiscard').onclick = function () {
      const st = TRIALS.state.steps[TRIALS.state.step];
      ask('step_discard', st.key, function (a) { if (a) TRIALS.render(a); });
    };
    $('stOk').onclick = TRIALS.confirm;
    $('stBack').onclick = function () { ask('step_go', TRIALS.state.step - 1, function (a) { if (a) TRIALS.render(a); }); };
    $('stVoid').onclick = function () {
      const reason = $('stVoidReason').value.trim();
      if (!reason) { toast(T('err.reason')); $('stVoidReason').focus(); return; }
      const t = TRIALS.state.trial.trial;
      if (!armed($('stVoid'), T('step.ctl.void.sure', t))) return;
      be('trial_void', t, reason, function (raw) {
        const a = JSON.parse(raw);
        if (failed(a)) return;
        $('stVoidReason').value = '';
        toast(T('trials.voided', t), 3500);
        TRIALS.render(a);
      });
    };
    $('lgSave').onclick = TRIALS.saveEdit;
    $('lgVoid').onclick = function () {
      const reason = $('lgVoidReason').value.trim(), t = TRIALS.logSel;
      if (!reason) { toast(T('err.reason')); $('lgVoidReason').focus(); return; }
      if (!armed($('lgVoid'), T('step.ctl.void.sure', t))) return;
      be('trial_void', t, reason, function (raw) {
        const a = JSON.parse(raw);
        if (failed(a)) return;
        $('lgVoidReason').value = '';
        toast(T('trials.voided', t), 3500);
        TRIALS.loadLog();
      });
    };
    $('lgOpen').onclick = function () {
      if (TRIALS.log && TRIALS.logSel) ask('open_path', TRIALS.log.folder + '\\' + TRIALS.logSel);
    };
    setOptions($('cfgPhase'), [T('phase.campaign'), T('phase.pilot')], ['campaign', 'pilot']);
    setOptions($('cfgOrder'), [T('cfg.order.random'), T('cfg.order.ascending')], ['random', 'ascending']);
    $('cfgPhase').onchange = function () {
      ask('set_phase', $('cfgPhase').value, function (s) {
        if (!s) { $('cfgPhase').value = TRIALS.cfg.phase; return; }
        TRIALS.state = s;
        TRIALS.loadCfg();
        toast(T('cfg.phase.set', T('phase.' + s.phase)));
      });
    };
    ['cfgHolders', 'cfgAngles', 'cfgDirections', 'cfgReps', 'cfgDrape'].forEach(function (id) { $(id).oninput = TRIALS.cfgPreview; });
    $('cfgApply').onclick = TRIALS.applyCfg;
    $('cfgReset').onclick = function () { if (TRIALS.cfg) TRIALS.fillCfg(TRIALS.cfg.default); };
    $('chkRun').onclick = TRIALS.runCheck;
  },
};
