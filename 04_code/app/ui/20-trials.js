// ImpactGuide · Trials, four steps on the rail: the trial to do now, step by
// step, the log with its audit trail, the sequence and its setup. The protocol is
// trials.py's (sequence, steps, names, log, completeness); the page only shows
// it and asks the backend, which also drives the phone and watches the scanner.
// Every sentence on screen is the page's own: warnings, file problems and the
// completeness check come as codes, the backend's English never shows.
'use strict';

/** Trials: the guide of the trial to do now, the log, the sequence and the setup. */
const TRIALS = {
  state: null,
  log: null,
  logSel: '',
  cfg: null,
  extraOpen: false,
  starting: false,   // Start was pressed and its answer has not come yet

  enter: function () {
    TRIALS.setView(trialsView);
    be('phone_refresh');
  },

  setView: function (v) {
    trialsView = v;
    rail();
    showView('trials-' + v);
    if (v === 'guide') TRIALS.refresh();
    else if (v === 'log') TRIALS.loadLog();
    else if (v === 'sequence') TRIALS.loadSeq();
    else TRIALS.loadCfg();
    if (v !== 'guide') showPanel('trials-' + v);
  },

  where: function (t) {
    $('where').textContent = T('trials.where', TRIALS.word('phase.', TRIALS.state ? TRIALS.state.phase : 'campaign'),
                               t.filled, t.slots) + (t.extra ? ' · ' + T('trials.where.extra', t.extra) : '');
  },

  refresh: function () {
    ask('trials_state', function (s) { if (s) TRIALS.render(s); });
  },

  // A pre-impact position (a row or a slot with a position number), not an impact.
  pre: function (r) { return !!r && String(r.position === undefined || r.position === null ? '' : r.position) !== ''; },
  // "Set 1 · Position 3 · left off": a position in words (the lights left out when empty)
  where1: function (set, position, light) {
    return T('trials.tile.set') + ' ' + set + ' · ' + T('trials.tile.position') + ' ' + position
      + (light ? ' · ' + TRIALS.word('light.', light) : '');
  },

  // A value of the log in words (a status, a phase, a reason): its string, or the value itself as it was logged.
  word: function (prefix, v) { return hasT(prefix + v) ? T(prefix + v) : String(v === undefined || v === null ? '' : v); },

  // ── the guide ──────────────────────────────────────────────────────────
  base: function (key) { return key.replace(/_az\d+$/, ''); },
  az: function (key) { const m = /_az(\d+)$/.exec(key); return m ? m[1] : ''; },

  stepTitle: function (st) {
    const k = 'step.' + TRIALS.base(st.key) + '.title';
    return hasT(k) ? T(k, TRIALS.az(st.key)) : T('step.unknown.title');
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
    if (base === 'light') return T('step.light.text', TRIALS.word('light.', t.light));
    if (base === 'video' || base === 'drape_video') {
      const a = c.video_az[0], b = c.video_az[1];
      return T('step.' + base + '.text', a, b, Math.round((b - a) / 8), c.photo_az[0]);
    }
    const k = 'step.' + base + '.text';
    return hasT(k) ? T(k, az) : T('step.unknown.text');
  },

  // What to watch out for before this trial, by code; an older backend sends English sentences, and the only one
  // it makes is the unknown cap shift of the trial before.
  warnings: function (t) {
    if (Array.isArray(t.warning_codes)) {
      return t.warning_codes.map(function (w) {
        // the trial it is about (cap_unknown), or the trials (several_open)
        const which = w.trial || (Array.isArray(w.trials) ? w.trials.join(', ') : '') || (t.prev ? t.prev.trial : '');
        return which ? Tcode('trials.warn.', w.code, 'generic', which) : T('trials.warn.generic');
      });
    }
    return (t.warnings || []).map(function (w) {
      console.warn('trial warning', w);
      return t.prev ? T('trials.warn.cap_unknown', t.prev.trial) : T('trials.warn.generic');
    });
  },

  render: function (s) {
    TRIALS.state = s;
    TRIALS.where(s.totals);
    if (trialsView !== 'guide' || (typeof TWIN !== 'undefined' && TWIN.showing())) return;   // the twin in its place
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
    if (t.open) chips.append(chip(T(t.resumed ? 'trials.chip.resumed' : 'trials.chip.open', fmtTime(t.time_start)), 'accent'));
    const reasons = (t.reference || '').split(';').filter(Boolean);
    if (reasons.length) {
      // the chip names it; why the reference scans are due is its tooltip
      const c = chip(T('trials.chip.ref'));
      c.title = T('trials.chip.reference', reasons.map(function (r) { return TRIALS.word('ref.', r); }).join(', '));
      chips.append(c);
    }
    if (String(t.drape) === '1') chips.append(chip(T('trials.chip.drape'), 'warn'));

    // the settings
    const tiles = $('gTiles');
    tiles.replaceChildren();
    const ch = t.changes || [];
    // a pre-impact position: its set, its number and the lights, no pendulum
    (TRIALS.pre(t) ? [['set', 'trials.tile.set', t.set], ['position', 'trials.tile.position', t.position],
                      ['light', 'trials.tile.light', TRIALS.word('light.', t.light)]] :
    [['holder', 'trials.tile.holder', t.holder],
     ['angle_deg', 'trials.tile.hole', t.angle_deg + '°'],
     ['direction_deg', 'trials.tile.direction', t.direction_deg + '°'],
     ['washers_per_side', 'trials.tile.washers', t.washers_per_side],
     ['drape', 'trials.tile.drape', T(String(t.drape) === '1' ? 'word.yes' : 'word.no')]]).forEach(function (x) {
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
        t.prev ? t.prev.trial : '', t.prev && t.prev.cap_shift_mm ? fmtNum(t.prev.cap_shift_mm, 1) : '?', fmtNum(s.config.reposition_mm, 1))));
    }
    TRIALS.warnings(t).forEach(function (w) { warn.append(E('div', 'callout callout-warn', w)); });
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

    TRIALS.stepList(s);
    if (t.open) { showPanel('trials-step'); TRIALS.renderStep(s); }
    else { showPanel('trials-plan'); TRIALS.renderPlan(s); }
  },

  // All the steps of the trial; while it is open a step can be gone back to, with the mouse or the keyboard.
  stepList: function (s) {
    const t = s.trial, i = s.step, st = s.steps[i];
    const list = $('gSteps');
    const focused = list.contains(document.activeElement);
    list.replaceChildren();
    s.steps.forEach(function (x, k) {
      const now = t.open && k === i;
      const it = E('div', 'mitem ' + (now ? 'sel' : (x.done ? 'done' : 'todo')));
      const name = E('span', 'mitem-name');
      name.append(E('span', 'blk-n', k + 1), E('span', null, TRIALS.stepTitle(x)));
      if (x.files.length) it.title = x.files.join(', ');
      it.append(name);
      if (now || x.done) it.append(E('span', 'mbadge', T(now ? 'trials.badge.now' : 'trials.badge.done')));
      if (t.open) {
        it.tabIndex = 0;
        it.setAttribute('role', 'button');
        onActivate(it, function () { ask('step_go', k, function (a) { if (a) TRIALS.render(a); }); });
      }
      list.append(it);
    });
    const sel = list.querySelector('.mitem.sel');
    if (sel && focused) sel.focus();
    if (sel && TRIALS._lastKey !== (t.trial + st.key)) {
      TRIALS._lastKey = t.trial + st.key;
      // the step to do now in the middle of its own column; on a narrow stage the column is the page,
      // which stays where it is so the step's card stays in view
      const side = sel.closest('.split-side');
      if (side && getComputedStyle(side).overflowY !== 'visible') {
        side.scrollTop += sel.getBoundingClientRect().top - side.getBoundingClientRect().top
          - side.clientHeight / 2 + sel.offsetHeight / 2;
      }
    }
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
                piRow(T('trials.file.size'), fmtNum(staged.size / 1048576, 1) + ' MB'));
    TRIALS.fileProblems(staged).forEach(function (p) { host.append(E('div', 'callout callout-warn', p)); });
  },

  // What looks wrong in a file received, by code with its values; an older backend sends English sentences,
  // and then the page says only that the file looks wrong.
  fileProblems: function (staged) {
    if (Array.isArray(staged.problem_codes)) {
      return staged.problem_codes.map(function (p) {
        const got = TRIALS.measure(p.got), want = TRIALS.measure(p.want);
        // codes said without their values: the file unreadable, the photo out of the camera's Pro mode
        if (p.code !== 'unreadable' && p.code !== 'photo_auto' && (!got || !want)) return T('problem.file.generic');
        return Tcode('problem.file.', p.code, 'generic', got, want);
      });
    }
    return (staged.problems || []).length ? [T('problem.file.generic')] : [];
  },

  // A measured value of a file in the language's numbers: a size (two numbers), a duration or a rate.
  measure: function (v) {
    if (Array.isArray(v)) return v.length === 2 && v.every(isFinite) ? fmtNum(v[0], 0) + ' × ' + fmtNum(v[1], 0) : '';
    if (v === '' || v === null || v === undefined || !isFinite(v)) return '';
    return fmtNum(v, Number(v) % 1 ? 1 : 0);
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
    $('planStart').disabled = TRIALS.starting;
    $('planStartL').textContent = T('trials.plan.start', t.trial);
    $('planScanner').textContent = s.scanner_folder || T('trials.scanner.none');
    $('planScannerBtn').classList.toggle('hidden', !!s.simulate);
    // a pre-impact position has no pendulum to weigh and no extra impact
    const pre = TRIALS.pre(t);
    $('planMass').closest('.field').classList.toggle('hidden', pre);
    $('extraForm').previousElementSibling.classList.toggle('hidden', pre);
    $('extraForm').classList.toggle('hidden', !TRIALS.extraOpen || pre);
    $('extraBtn').classList.toggle('hidden', TRIALS.extraOpen || pre);
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
    ask('trial_options', JSON.stringify(o), function (a) { if (a) TRIALS.render(a); });
  },

  start: function () {
    const mass = $('planMass').value.trim();
    TRIALS.starting = true;          // one press opens one trial: the button waits for the answer
    $('planStart').disabled = true;
    ask('trial_start', JSON.stringify({ swing_mass_g: mass }), function (a, bad) {
      TRIALS.starting = false;
      $('planStart').disabled = false;
      if (bad && bad.error === 'swing_mass') { $('planMass').focus(); return; }
      if (!a) return;
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
    // the scanner's folder watched for this step's export: asked once per step (a refusal is said once)
    if (waiting && s.scanner_folder && s.watching !== st.key && TRIALS._watchFor !== t.trial + st.key) {
      TRIALS._watchFor = t.trial + st.key;
      ask('scan_watch', st.key);
    }
    if (kind === 'spheres' && TRIALS._spheresFor !== t.trial) {
      TRIALS._spheresFor = t.trial;
      $('stSpheres').checked = t.spheres_ok !== '0';
      $('stShift').value = t.cap_shift_mm || '';
    }
    $('stSecond').closest('label').classList.toggle('hidden', TRIALS.pre(t));     // no pendulum at a position
    $('stCap').closest('label').classList.toggle('hidden', TRIALS.pre(t));
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
    ask('step_confirm', st.key, JSON.stringify(values), function (a) {
      if (!a) { TRIALS.refresh(); return; }
      if (a.closed) toast(T('trials.closed', a.closed), TOAST_SAID_MS);
      TRIALS.render(a);
    });
  },

  onEvent: function (e) {
    // a file of a step: shot, recorded and copied, or exported by the scanner
    if (e.type === 'staged') workOff('trial_file');
    else if (e.type === 'recording') workOn('trial_file', 'pill.recording');
    else if (e.type === 'acquire_failed' && e.key !== 'calibration') workOff('trial_file');
    else return;
    if (page === 'trials') TRIALS.refresh();
  },

  // ── the log ────────────────────────────────────────────────────────────
  colName: function (c) { return hasT('col.' + c) ? T('col.' + c) : c; },
  // a trial's hole and direction as two signs, each with its name in the tooltip (ICONS.hole, ICONS.direction)
  signs: function (r) {
    return [signed('hole', r.angle_deg + '°', T('col.angle_deg')),
            signed('direction', r.direction_deg + '°', T('col.direction_deg'))];
  },

  // A logged value as it reads: yes or no, a time, a unit after the number.
  colValue: function (c, v) {
    if (v === '') return '';
    if (c.unit === '0/1') return T(v === '1' ? 'word.yes' : 'word.no');
    if (c.unit === 'ISO 8601') return v.replace('T', ' ').replace(/(Z|[+-]\d\d:?\d\d)$/, '');   // local time, as logged
    if (c.unit === 'deg') return v + '°';
    if (c.unit === 'mm' || c.unit === 'g') return v + ' ' + c.unit;
    if (c.name === 'status') return TRIALS.word('status.', v);
    if (c.name === 'phase') return TRIALS.word('phase.', v);
    if (c.name === 'reference') return v.split(';').filter(Boolean).map(function (r) { return TRIALS.word('ref.', r); }).join(', ');
    return v;
  },

  loadLog: function () {
    ask('log_rows', function (L) { if (L) TRIALS.renderLog(L); });
  },

  renderLog: function (L) {
    TRIALS.log = L;
    if (TRIALS.state) TRIALS.where(TRIALS.state.totals);
    const list = $('logList');
    const focused = list.contains(document.activeElement);
    list.replaceChildren();
    const edited = {};
    L.edits.forEach(function (e) { edited[e.trial] = (edited[e.trial] || 0) + 1; });
    const rows = L.rows.slice().reverse();
    if (!TRIALS.logSel && rows.length) TRIALS.logSel = rows[0].trial;
    rows.forEach(function (r) {
      const it = E('div', 'mitem' + (r.trial === TRIALS.logSel ? ' sel' : ''));
      it.tabIndex = 0;
      it.setAttribute('role', 'button');
      const name = E('span', 'mitem-name', r.trial + ' ');
      // the hole and the direction as signs, their names in the tooltips (TRIALS.signs)
      let det = (r.drape === '1' ? ' · ' + T('trials.chip.drape') : '') + ' · ' + fmtTime(r.time_impact || r.time_start);
      if (r.cap_shift_mm) det += ' · ' + T('log.cap', fmtNum(r.cap_shift_mm, 2));
      if (edited[r.trial]) det += ' · ' + T('log.edited', edited[r.trial]);
      const d = E('span', 'mitem-det', TRIALS.pre(r) ? TRIALS.where1(r.set, r.position, r.light) : r.holder + ' · ');
      if (!TRIALS.pre(r)) d.append.apply(d, TRIALS.signs(r));
      d.append(document.createTextNode(det));
      name.append(d);
      it.append(name, E('span', 'mbadge', TRIALS.word('status.', r.status)));
      onActivate(it, function () { TRIALS.logSel = r.trial; TRIALS.renderLog(TRIALS.log); });
      list.append(it);
    });
    const sel = list.querySelector('.mitem.sel');
    if (sel && focused) sel.focus();
    $('logNote').textContent = rows.length ? '' : T('log.none');

    // the panel: the selected trial
    const r = L.rows.find(function (x) { return x.trial === TRIALS.logSel; });
    $('lgTitle').textContent = r ? r.trial : T('trials.view.log');
    $('lgSub').textContent = r ? TRIALS.word('status.', r.status) + ' · ' + r.session : T('log.none');
    const fields = $('lgFields');
    fields.replaceChildren();
    // never edited: the trial, its phase, the version of the configuration it started under
    const fixed = ['trial', 'phase', 'config_version'];
    const editable = L.columns.filter(function (c) { return fixed.indexOf(c.name) < 0; });
    if (r) {
      L.columns.forEach(function (c) {
        if (r[c.name] === '') return;
        // a time as the hour (the day too when it is not today), the whole stamp in the row's tooltip
        const iso = c.unit === 'ISO 8601';
        const row = piRow(TRIALS.colName(c.name), iso ? fmtTime(r[c.name]) : TRIALS.colValue(c, r[c.name]));
        if (iso) row.title = TRIALS.colValue(c, r[c.name]);
        fields.append(row);
      });
    }
    setOptions($('lgField'), editable.map(function (c) { return TRIALS.colName(c.name); }), editable.map(function (c) { return c.name; }));
    const canVoid = !!r && r.status !== 'void';
    $('lgVoid').classList.toggle('hidden', !canVoid);
    $('lgVoidReason').classList.toggle('hidden', !canVoid);
    $('lgVoidLbl').classList.toggle('hidden', !canVoid);
    ['lgValue', 'lgReason', 'lgSave', 'lgOpen'].forEach(function (id) { $(id).disabled = !r; });
    const ed = $('lgEdits');
    ed.replaceChildren();
    L.edits.filter(function (e) { return r && e.trial === r.trial; }).reverse().forEach(function (e) {
      const card = E('div', 'edit-card', TRIALS.colName(e.field) + ': ' + (e.old || '∅') + ' → ' + (e.new || '∅'));
      card.append(E('span', 'aside', fmtTime(e.time) + ' · ' + e.reason));
      ed.append(card);
    });
    if (r && !ed.children.length) ed.append(E('div', 'note', T('log.edits.none')));
  },

  saveEdit: function () {
    const field = $('lgField').value, value = $('lgValue').value.trim(), reason = $('lgReason').value.trim();
    if (!reason) { toast(T('err.reason')); $('lgReason').focus(); return; }
    ask('log_edit', TRIALS.logSel, field, value, reason, function (L) {
      if (!L) return;
      $('lgValue').value = '';
      $('lgReason').value = '';
      TRIALS.renderLog(L);
      toast(T('log.saved', TRIALS.colName(field)));
      ask('trials_state', function (s) { if (s) { TRIALS.state = s; TRIALS.where(s.totals); } });
    });
  },

  // The selected trial's folder: by its name with a newer backend, else by its path.
  openFolder: function () {
    if (!TRIALS.log || !TRIALS.logSel) return;
    if (hasSlot('trial_open_folder')) { ask('trial_open_folder', TRIALS.logSel); return; }
    const folder = String(TRIALS.log.folder || ''), sep = folder.indexOf('\\') >= 0 ? '\\' : '/';
    ask('open_path', folder.replace(/[\\/]+$/, '') + sep + TRIALS.logSel);
  },

  // Voids a trial, with its reason and a second press; then(answer) once it is done.
  voidTrial: function (trial, reasonId, buttonId, then) {
    const reason = $(reasonId).value.trim();
    if (!reason) { toast(T('err.reason')); $(reasonId).focus(); return; }
    if (!armed($(buttonId), T('step.ctl.void.sure', trial))) return;
    ask('trial_void', trial, reason, function (a) {
      if (!a) return;
      $(reasonId).value = '';
      toast(T('trials.voided', trial), TOAST_SAID_MS);
      then(a);
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
      if (TRIALS.pre(r)) {          // a pre-impact slot: its set, its position and the lights
        cell.append(E('div', 'seq-n', r.slot + ' · ' + TRIALS.where1(r.replicate, r.position, '')),
                    E('div', 'seq-v', TRIALS.word('light.', r.light)), E('div', 'seq-t', r.trial || (next ? T('trials.badge.next') : '')));
        grid.append(cell);
        return;
      }
      cell.title = T('seq.cell', r.angle_deg, r.direction_deg);      // the two signs in words
      cell.append(E('div', 'seq-n', r.slot + ' · ' + r.holder + ' · R' + r.replicate + (r.drape === '1' ? ' · ' + T('trials.chip.drape') : '')),
                  (function () { const v = E('div', 'seq-v'); v.append.apply(v, TRIALS.signs(r)); return v; })(),
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
    $('cfgSets').value = c.repetitions;          // pre-impact: the sets of positions
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
    // the pre-impact sets have no matrix of impacts: only their number, under Advanced
    const pre = C.phase === 'pre_impact';
    $('cfgPreAdv').classList.toggle('hidden', !pre);
    ['cfgHolders', 'cfgAngles', 'cfgDirections', 'cfgReps', 'cfgDrape', 'cfgPreview', 'cfgAdv'].forEach(function (id) {
      const el = $(id);
      const box = el.closest('.field') || (el.parentElement.classList.contains('row') ? el.parentElement : el);
      box.classList.toggle('hidden', pre);
      const lbl = box.previousElementSibling;
      if (lbl && lbl.classList.contains('sec-lbl')) lbl.classList.toggle('hidden', pre);
    });
    const t = C.totals;
    $('cfgTotals').replaceChildren(
      piRow(T('cfg.t.phase'), TRIALS.word('phase.', C.phase)), piRow(T('seq.slots'), t.slots),
      piRow(T('seq.filled'), t.filled), piRow(T('seq.pending'), t.pending),
      piRow(T('cfg.t.written'), fmtTime(C.written)));
    const h = $('cfgHistory');
    h.replaceChildren();
    C.history.slice().reverse().forEach(function (x) {
      const c = x.config;
      const card = E('div', 'edit-card', T('cfg.history.item', c.holders.join('+'), c.angles_deg.length, c.directions_deg.length,
                                             c.repetitions, c.drape_trials, c.seed));
      card.append(E('span', 'aside', T('cfg.history.until', fmtTime(x.written))));
      h.append(card);
    });
    if (!C.history.length) h.append(E('div', 'note', T('cfg.history.none')));
    TRIALS.where(t);
  },

  applyCfg: function () {
    const pre = TRIALS.cfg && TRIALS.cfg.phase === 'pre_impact';
    ask('config_apply', JSON.stringify(pre ? { repetitions: Number($('cfgSets').value) } : TRIALS.readCfg()), function (C) {
      if (!C) return;
      TRIALS.renderCfg(C);
      toast(T('cfg.applied', C.totals.slots, C.totals.pending), TOAST_SAID_MS);
    });
  },

  // One problem of the completeness check in words: its kind, then the file or the field it is about. A file's
  // name is shown, a field by its name in the log; the backend's own sentence (a file's content) is not, and
  // nothing is shown beside a kind the page does not know.
  checkProblem: function (p) {
    const kind = Array.isArray(p) ? p[0] : (p.code || p.kind);
    const item = String((Array.isArray(p) ? p[1] : (p.file || p.field)) || '');
    const word = Tcode('problem.', kind, 'content');
    if (kind === 'log' || kind === 'online' || kind === 'bad_field') return word + ' ' + TRIALS.colName(item);
    let name = '';
    if (kind === 'content') name = (/^([\w.-]+):/.exec(item) || [])[1] || '';
    else if (TRIALS.FILE_PROBLEMS.indexOf(kind) >= 0) name = item;
    return name ? word + ' ' + name : word;
  },
  FILE_PROBLEMS: ['missing', 'duplicate', 'empty', 'unexpected', 'unreadable', 'changed'],    // kinds naming files

  runCheck: function () {
    ask('check_phase', $('chkDeep').checked, function (R) {
      if (!R) return;
      const lines = [];
      R.trials.forEach(function (t) {
        if (t.problems.length) lines.push(t.trial + ' (' + TRIALS.word('status.', t.status) + '): '
          + t.problems.map(TRIALS.checkProblem).join('; '));
      });
      lines.push(T('cfg.check.summary', R.trials.length, lines.length, R.pending, R.extra.join(', ') || '-', R.orphans.join(', ') || '-'));
      // trials logged twice, and the open ones (more than one open is a problem)
      if ((R.duplicates || []).length) lines.push(T('cfg.check.duplicates', R.duplicates.join(', ')));
      if ((R.open || []).length) lines.push(T('cfg.check.open', R.open.join(', ')));
      $('chkLog').textContent = lines.join('\n');
    });
  },

  wire: function () {
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
      ask('step_shoot', st.key, function (a) { if (a) workOn('trial_file', 'pill.shooting'); });
    };
    $('stRec').onclick = function () {
      const s = TRIALS.state, st = s.steps[s.step];
      const stop = s.recording === st.key;
      ask('step_video', st.key, stop ? 'stop' : 'start', function (a) {
        // the recording is starting, or it ends and the copy starts: one piece of work either way
        if (a) workOn('trial_file', stop ? 'pill.pulling' : 'pill.starting');
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
      TRIALS.voidTrial(TRIALS.state.trial.trial, 'stVoidReason', 'stVoid', TRIALS.render);
    };
    $('lgSave').onclick = TRIALS.saveEdit;
    $('lgVoid').onclick = function () { TRIALS.voidTrial(TRIALS.logSel, 'lgVoidReason', 'lgVoid', TRIALS.loadLog); };
    $('lgOpen').onclick = TRIALS.openFolder;
    setOptions($('cfgPhase'), [T('phase.campaign'), T('phase.pilot'), T('phase.pre_impact')], ['campaign', 'pilot', 'pre_impact']);
    setOptions($('cfgOrder'), [T('cfg.order.random'), T('cfg.order.ascending')], ['random', 'ascending']);
    $('cfgPhase').onchange = function () {
      ask('set_phase', $('cfgPhase').value, function (s) {
        if (!s) { $('cfgPhase').value = TRIALS.cfg.phase; return; }
        TRIALS.state = s;
        TRIALS.loadCfg();
        toast(T('cfg.phase.set', TRIALS.word('phase.', s.phase)));
      });
    };
    ['cfgHolders', 'cfgAngles', 'cfgDirections', 'cfgReps', 'cfgDrape'].forEach(function (id) { $(id).oninput = TRIALS.cfgPreview; });
    $('cfgApply').onclick = TRIALS.applyCfg;
    $('cfgReset').onclick = function () { if (TRIALS.cfg) TRIALS.fillCfg(TRIALS.cfg.default); };
    $('chkRun').onclick = TRIALS.runCheck;
  },
};
