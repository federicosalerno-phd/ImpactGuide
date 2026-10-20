// ImpactGuide · Scans and results, its first step: the scans of a trial as the scanner exports them and what they
// measure. The backend (scans_page.py, scan_desk.py) watches the scanner's export folder, binds each export to its
// step (before or after the impact, at the arm position it was taken from), measures the six spheres at once, writes
// the trial log's two values and says so in `scans` events; the page shows it: the scans before and after the
// impact, each with its state (waiting, measuring, its spheres found, or missing, with a shape for each), then the
// marker's shift, its turn, the worst target and the cap's shift since the reference scans (over the limit the
// holder goes back with the jig), and the error at each target on the twin's colour scale (blue to yellow orange,
// apart for eyes that confuse red and green). The before and after in 3D is the twin's (18-twin.js), opened on the
// trial. The panel: the folder watched, the trial to look at, the spheres' radii (calibrated once from three scans
// without an impact), the repeatability (scans of the bench left alone, their spread at the targets), the day's
// processing; the folder, the formats and the results under Advanced. A run without an impact fills the stage
// with its count and its scans; its actions are at the foot of the panel.
'use strict';

const SCANS = {
  state: null,
  pick: '',             // the trial chosen in the panel ('' the open one, else the newest)
  timer: 0,
  POLL_MS: 2500,        // read again this often while on show (and at every event)
  RUN_MIN: 3,           // scans of a repeatability run, at least
  repTyped: false,      // their number typed under Advanced

  showing: function () {
    const v = document.querySelector('.view[data-view="scans"]');
    return !!v && v.classList.contains('show');
  },

  enter: function () {
    showView('scans');
    showPanel('scans');
    $('hint').textContent = '';
    SCANS.refresh();
    clearInterval(SCANS.timer);
    SCANS.timer = setInterval(function () {
      if (!SCANS.showing() || page !== 'results') { clearInterval(SCANS.timer); return; }
      SCANS.refresh();
    }, SCANS.POLL_MS);
  },

  refresh: function () {
    askQuiet('scans_state', JSON.stringify({ trial: SCANS.pick }), function (s, bad) {
      if (!s) { if (SCANS.showing()) console.warn('scans state', bad); return; }
      SCANS.state = s;
      if (SCANS.showing()) SCANS.render(s);
    });
  },

  // ── the page ───────────────────────────────────────────────────────────
  render: function (s) {
    const run = s.capture && s.capture.on;
    $('scTrialCol').classList.toggle('hidden', !!run || !s.trial);
    $('scEmpty').classList.toggle('hidden', !!run || !!s.trial);
    $('scRun').classList.toggle('hidden', !run);
    if (run) SCANS.renderRun(s, run);
    else if (s.trial) SCANS.renderTrial(s);
    else $('where').textContent = '';
    SCANS.renderPanel(s);
  },

  // the trial: its head, its scans before and after the impact, the result
  renderTrial: function (s) {
    const t = s.trial;
    $('scId').textContent = t.trial;
    $('scSub').textContent = [T('trials.tile.holder') + ' ' + t.holder, T('trials.tile.hole') + ' ' + fmtNum(t.angle_deg, 0) + '°',
                              T('trials.tile.direction') + ' ' + fmtNum(t.direction_deg, 0) + '°'].join(' · ');
    $('where').textContent = t.trial;
    const chips = $('scChips');
    chips.replaceChildren();
    if (s.open) chips.append(chip(T('scans.chip.open'), 'accent'));
    if (String(t.drape) === '1') chips.append(chip(T('trials.chip.drape'), 'warn'));
    if (!s.calibration.done) {
      const c = chip(T('twin.scan.uncalibrated'), 'warn');
      c.title = T('twin.scan.uncalibrated.hint');
      chips.append(c);
    }
    // the twin shows a trial's motion once its scans are measured and filed
    $('scTwin').disabled = s.result.state !== 'ready' || !s.slots.every(function (x) { return x.filed; });
    // before: the trial's own reference scans, or the scans after the impact of the trial before
    const b = s.before || {};
    const before = s.slots.filter(function (x) { return x.when === 'ref'; });
    const after = s.slots.filter(function (x) { return x.when === 'post'; });
    $('scBefore').replaceChildren.apply($('scBefore'), before.length ? before.map(SCANS.slotRow)
      : [SCANS.fromRow(b)]);
    $('scBeforeSub').textContent = before.length ? T('scans.before.own') : (b.trial ? T('scans.before.from', b.trial) : '');
    $('scAfter').replaceChildren.apply($('scAfter'), after.map(SCANS.slotRow));
    $('scAfterSub').textContent = '';
    SCANS.renderResult(s);
  },

  // a scan of the trial: its arm position, its state as a sign and a word, the spheres found
  slotRow: function (x) {
    const r = E('div', 'sc-slot sc-' + x.state);
    const sign = E('span', 'sc-st');
    if (x.state === 'measuring') sign.append(E('span', 'spinner spinner-sm'));
    const words = E('div', 'sc-slot-t');
    words.append(E('div', 'sc-slot-n', T('scans.arm', x.arm || '-', fmtNum(x.az, 0))),
                 E('div', 'sc-slot-s', T('scans.state.' + x.state)));
    r.append(sign, words);
    if (x.n_spheres !== null && x.n_spheres !== undefined) {
      const v = E('span', 'sc-slot-v', T('scans.spheres', fmtNum(x.n_spheres, 0)));
      v.title = T('scans.spheres.hint');
      r.append(v);
    }
    if (x.arm_off) {
      const c = chip(T('scans.arm.off'), 'warn');
      c.title = T('scans.arm.off.hint', x.arm_seen);
      r.append(c);
    }
    if (x.file) r.title = x.file;
    return r;
  },

  // before the impact, when it is the scans after the impact of the trial before
  fromRow: function (b) {
    const state = { ready: 'ok', partial: 'ok', measuring: 'measuring', failed: 'bad' }[b.state] || 'skipped';
    const r = E('div', 'sc-slot sc-' + state);
    const sign = E('span', 'sc-st');
    if (state === 'measuring') sign.append(E('span', 'spinner spinner-sm'));
    const words = E('div', 'sc-slot-t');
    words.append(E('div', 'sc-slot-n', b.trial ? T('scans.before.after', b.trial) : T('scans.before.none')),
                 E('div', 'sc-slot-s', T('scans.state.' + state)));
    r.append(sign, words);
    return r;
  },

  // what the scans measure: the marker's shift, its turn, the worst target, the cap's shift since the reference;
  // the error at each target in its colour; or why there is nothing yet
  renderResult: function (s) {
    const res = s.result || {}, st = res.state;
    const ready = st === 'ready' || st === 'partial';
    const chips = $('scResChips');
    chips.replaceChildren();
    $('scResBody').classList.toggle('hidden', !ready);
    $('scResSay').classList.toggle('hidden', ready);
    if (!ready) {
      const key = st === 'failed' ? (hasT('scans.res.failed.' + res.why) ? 'scans.res.failed.' + res.why : 'scans.res.failed')
        : st === 'measuring' || s.measuring ? 'scans.res.measuring' : st === 'waiting' ? 'scans.res.waiting' : 'scans.res.none';
      $('scResText').textContent = T(key);
      $('scResSpin').classList.toggle('hidden', key !== 'scans.res.measuring');
      $('scResSay').classList.toggle('sc-bad', st === 'failed');
      return;
    }
    if (st === 'partial') chips.append(chip(T('scans.res.partial', res.views, res.of), 'warn'));
    if (s.measuring) chips.append(chip(T('scans.res.again')));
    if (res.seconds !== null && res.seconds !== undefined) {
      const c = chip(T('scans.res.time', fmtNum(res.seconds, 0)));
      c.title = T('scans.res.time.hint');
      chips.append(c);
    }
    const logged = res.logged && Object.keys(res.logged).length;
    if (logged) {
      const c = chip(T('scans.res.logged'), 'accent');
      c.title = T('scans.res.logged.hint');
      chips.append(c);
    }
    if ((res.hand || []).length) {
      const c = chip(T('scans.res.hand'), 'warn');
      c.title = T('scans.res.hand.hint');
      chips.append(c);
    }
    const over = !!res.reposition;
    const tiles = [
      ['scans.res.shift', withUnit(res.shift_mm, 2, 'mm'), '', 'scans.res.shift.hint'],
      ['scans.res.rot', fmtNum(res.rot_deg, 2) + '°', '', 'scans.res.rot.hint'],
      ['scans.res.worst', withUnit(res.tre_max_mm, 2, 'mm'), T('scans.res.mean', fmtNum(res.tre_mean_mm, 2)), 'scans.res.worst.hint'],
      ['scans.res.cap', res.cap_shift_mm === null || res.cap_shift_mm === undefined ? '-' : withUnit(res.cap_shift_mm, 2, 'mm'),
       over ? T('scans.res.cap.over') : (res.reference ? T('scans.res.cap.since', res.reference) : ''), 'scans.res.cap.hint'],
    ];
    $('scTiles').replaceChildren.apply($('scTiles'), tiles.map(function (x, i) {
      const tile = E('div', 'tile-set' + (i === 3 && over ? ' sc-over' : ''));
      tile.title = T(x[3], fmtNum(res.cap_limit_mm, 0));
      tile.append(E('div', 'tile-l', T(x[0])), E('div', 'tile-v', x[1]), E('div', 'tile-c', x[2]));
      return tile;
    }));
    SCANS.legend($('scRamp'), $('scTicks'));
    $('scTargets').replaceChildren.apply($('scTargets'), (res.tre || []).map(function (e) {
      const row = E('div', 'tw-tg');
      const dot = E('span', 'tw-dot');
      dot.style.background = TWIN.colour(e.mm);
      row.append(dot, E('span', 'tw-tg-n', e.id), E('span', 'tw-tg-v', withUnit(e.mm, 2, 'mm')));
      row.title = e.name;
      return row;
    }));
  },

  // the twin's scale of the error at the targets, with its values
  legend: function (ramp, ticks) {
    const n = 12, stops = [];
    for (let i = 0; i <= n; i++) stops.push(TWIN.scale(i / n) + ' ' + (100 * i / n).toFixed(1) + '%');
    ramp.style.background = 'linear-gradient(90deg, ' + stops.join(', ') + ')';
    ticks.replaceChildren();
    for (let mm = 0; mm <= TWIN.TRE_TOP; mm++) ticks.append(E('span', null, mm === TWIN.TRE_TOP ? '≥ ' + mm + ' mm' : fmtNum(mm, 0)));
  },

  // a run without an impact: what to do, how many scans of how many, each scan, and the repeatability's spread
  renderRun: function (s, kind) {
    const c = s.capture[kind] || {};
    const want = Number(c.want) || 0, good = Number(c.good) || 0, n = Number(c.n) || 0;
    $('where').textContent = T('scans.run.' + kind + '.title');
    $('scRunTitle').textContent = T('scans.run.' + kind + '.title');
    $('scRunLine').textContent = T('scans.run.' + kind + '.line', want);
    $('scRunN').textContent = fmtNum(n, 0);
    $('scRunOf').textContent = T('scans.run.of', want);
    $('scRunBar').style.transform = 'scaleX(' + (want ? Math.min(1, good / want) : 0) + ')';
    const enough = good >= want;
    $('scRunCount').classList.toggle('enough', enough);
    $('scRunSay').textContent = T(enough ? (kind === 'calibration' ? 'scans.run.enough' : 'scans.run.done')
      : (n > good ? 'scans.run.some_bad' : 'scans.run.waiting'));
    $('scRunFiles').replaceChildren.apply($('scRunFiles'), (c.files || []).map(function (f, i) {
      const st = f.ok ? 'ok' : 'bad';
      const tile = E('div', 'sc-cell sc-' + st);
      tile.append(E('span', 'sc-st'), E('span', 'sc-cell-n', String(n - c.files.length + i + 1)),
                  E('span', 'sc-cell-v', T('scans.spheres', fmtNum(f.n_spheres || 0, 0))));
      tile.title = f.name;
      return tile;
    }));
    const r = c.result || {};
    const has = kind === 'repeat' && r.n >= 2 && r.sd_mm;
    $('scRunRes').classList.toggle('hidden', !has);
    if (has) SCANS.renderSpread(r, $('scRunVerdict'), $('scRunTiles'), $('scRunTargets'));
  },

  // the repeatability: its spread at the worst target against the limit, its median, each target's
  renderSpread: function (r, verdict, tiles, list) {
    const ok = !!r.ok;
    const v = E('span', 'sc-verdict ' + (ok ? 'sc-ok' : 'sc-bad'));
    v.append(E('span', 'sc-st'), E('span', null, T(ok ? 'scans.run.limit.ok' : 'scans.run.limit.bad', fmtNum(r.limit_mm, 2))));
    verdict.replaceChildren(v);
    tiles.replaceChildren(
      SCANS.tile(T('scans.run.worst'), withUnit(r.spread_mm, 3, 'mm'), T('scans.run.over', fmtNum(r.n, 0))),
      SCANS.tile(T('scans.run.median'), withUnit(r.median_mm, 3, 'mm'), ''));
    list.replaceChildren.apply(list, Object.keys(r.sd_mm).map(function (k) {
      const row = E('div', 'tw-tg');
      const dot = E('span', 'tw-dot');
      dot.style.background = TWIN.colour(r.sd_mm[k] / r.limit_mm * TWIN.TRE_TOP / 2);
      row.append(dot, E('span', 'tw-tg-n', k), E('span', 'tw-tg-v', withUnit(r.sd_mm[k], 3, 'mm')));
      return row;
    }));
  },

  tile: function (label, value, sub) {
    const t = E('div', 'tile-set');
    t.append(E('div', 'tile-l', label), E('div', 'tile-v', value), E('div', 'tile-c', sub || ''));
    return t;
  },

  // ── the panel ──────────────────────────────────────────────────────────
  renderPanel: function (s) {
    const run = s.capture && s.capture.on;
    $('scPSub').textContent = s.trial ? s.trial.trial : '';
    // the folder: watched for a scan, waiting for nothing now, not chosen, or not found
    const say = !s.folder ? 'none' : !s.folder_ok ? 'missing' : s.watching ? 'on' : 'idle';
    $('scWatchDot').className = 'live-dot ' + ({ on: 'busy', idle: 'ok', none: 'warn', missing: 'warn' }[say]);
    $('scWatchW').textContent = T('scans.watch.' + say);
    $('scWatch').title = s.folder || '';
    $('scFolderBtn').classList.toggle('hidden', !!s.simulate || say !== 'none' && say !== 'missing');
    $('scFolderBtn2').classList.toggle('hidden', !!s.simulate);
    $('scSim').classList.toggle('hidden', !(s.simulate && s.open && !run && s.slots.some(function (x) { return x.state === 'waiting'; })));
    $('scFolder').textContent = s.folder || T('trials.scanner.none');
    $('scFormats').textContent = (s.formats || []).join(' · ');
    // the trial to look at
    const list = s.trials || [];
    setOptions($('scTrial'), list.map(function (r) {
      return r.trial + ' · ' + r.holder + ' · ' + r.angle_deg + '° · ' + r.direction_deg + '°';
    }), list.map(function (r) { return r.trial; }));
    if (s.trial) $('scTrial').value = s.trial.trial;
    $('scPick').classList.toggle('hidden', list.length < 2 || !!run);
    // the radii: calibrated (when, from how many scans) or not yet
    const cal = s.calibration || {};
    const cc = $('scCalChip');
    cc.className = 'chip' + (cal.done ? ' chip-accent' : ' chip-warn');
    cc.textContent = cal.done ? T('scans.cal.done') : T('twin.scan.uncalibrated');
    cc.title = cal.done ? T('scans.cal.done.hint', fmtNum(cal.scans, 0), fmtTime(cal.date)) : T('twin.scan.uncalibrated.hint');
    $('scCalBtn').querySelector('.btn-label').textContent = T(cal.done ? 'scans.cal.again' : 'scans.cal.start');
    $('scCalBtn').disabled = !!run;
    // the repeatability: its last spread against the limit
    const rep = (s.capture || {}).repeat || {}, rr = rep.result || {};
    const note = $('scRepNote');
    note.classList.toggle('hidden', !(rr.n >= 2));
    if (rr.n >= 2) note.textContent = T('scans.rep.last', fmtNum(rr.spread_mm, 3), fmtNum(rr.n, 0))
      + ' · ' + T(rr.ok ? 'scans.run.limit.ok' : 'scans.run.limit.bad', fmtNum(rr.limit_mm, 2));
    $('scRepBtn').disabled = !!run;
    $('scRepBtn').title = T('scans.rep.start.hint', fmtNum(SCANS.repN(), 0));
    if (document.activeElement !== $('scRepN')) $('scRepN').value = String(SCANS.repN());
    // the day's processing: running, or what it said last
    const busy = (s.busy || []).indexOf('process') >= 0;
    const p = s.processed || {};
    const dn = $('scDayNote');
    dn.classList.toggle('hidden', !busy && !p.when);
    dn.textContent = busy ? T('scans.day.running') : p.when ? T('scans.day.last', fmtNum(p.measured, 0), fmtNum(p.trials, 0))
      + (p.missing ? ' · ' + T('scans.day.missing', fmtNum(p.missing, 0)) : '') : '';
    if (p.missing_trials && p.missing_trials.length) dn.title = p.missing_trials.join(', ');
    $('scDayBtn').disabled = busy || !!run;
    $('scDayAll').disabled = busy || !!run;
    // a run's actions at the foot
    $('scFoot').classList.toggle('hidden', !run);
    if (run) {
      const c = s.capture[run] || {};
      const enough = (Number(c.good) || 0) >= (Number(c.want) || 0);
      $('scRunGo').classList.toggle('hidden', run !== 'calibration');
      $('scRunGo').disabled = !enough || (s.busy || []).indexOf('calibrate') >= 0;
      $('scRunStopL').textContent = T(run === 'repeat' && enough ? 'scans.run.close' : 'scans.run.stop');
    }
  },

  // the scans of a repeatability run: as typed under Advanced, else as many as the last run's, else 20
  repN: function () {
    const last = Number((((SCANS.state || {}).capture || {}).repeat || {}).want) || 20;
    const n = SCANS.repTyped ? Number($('scRepN').value) || last : last;
    return Math.max(SCANS.RUN_MIN, Math.min(100, Math.round(n)));
  },

  // ── what the page asks ─────────────────────────────────────────────────
  capture: function (kind, action) {
    ask('scans_capture', kind, action, kind === 'repeat' ? SCANS.repN() : 0, function (s) {
      if (s) { SCANS.state = s; SCANS.render(s); }
    });
  },

  twin: function () {
    const t = SCANS.state && SCANS.state.trial;
    if (!t) return;
    TWIN.pick = t.trial;
    TWIN.open('results');
  },

  // ── events ─────────────────────────────────────────────────────────────
  onEvent: function (e) {
    if (e.type === 'scans') {
      if (e.action === 'measured') {
        const seen = SCANS.showing();
        if (e.state === 'ready') {
          playSound(e.bad ? 'error' : 'confirm');
          if (!seen) toast(T(e.reposition ? 'scans.toast.reposition' : 'scans.toast.measured', e.trial, fmtNum(e.seconds, 0)),
                           e.reposition ? TOAST_LONG_MS : TOAST_SAID_MS);
        } else if (e.bad) playSound('error');
        // the trial's page shows the values the scans wrote in its spheres step
        if (typeof TRIALS !== 'undefined' && (e.logged || []).length) {
          TRIALS._spheresFor = '';
          if (page === 'trials' && trialsView === 'guide') TRIALS.refresh();
        }
      } else if (e.action === 'unused') {
        toast(T('scans.toast.unused', e.file || ''), TOAST_LONG_MS);
      } else if (e.action === 'calibrated') {
        playSound('confirm');
        toast(T('scans.toast.calibrated'), TOAST_SAID_MS);
      } else if (e.action === 'processed') {
        toast(T('scans.toast.processed', fmtNum(e.measured, 0), fmtNum(e.trials, 0))
              + (e.missing ? ' · ' + T('scans.day.missing', fmtNum(e.missing, 0)) : ''), TOAST_LONG_MS);
      } else if (e.action === 'captured') {
        if (!e.ok) playSound('error');
        if (e.done && e.kind === 'repeat') playSound('confirm');
      } else if (e.action === 'failed') {
        toast(errorText(e), TOAST_ERROR_MS, 'error');
      }
      if (SCANS.showing()) SCANS.refresh();
      return;
    }
    if (e.type === 'staged' && SCANS.showing()) SCANS.refresh();
  },

  wire: function () {
    $('scTwin').onclick = SCANS.twin;
    const choose = function () {
      ask('choose_scanner_folder', function () { SCANS.refresh(); });
    };
    $('scFolderBtn').onclick = choose;
    $('scFolderBtn2').onclick = choose;
    $('scSim').onclick = function () { ask('scans_simulate', function () { SCANS.refresh(); }); };
    $('scTrial').onchange = function () { SCANS.pick = $('scTrial').value; SCANS.refresh(); };
    $('scCalBtn').onclick = function () { SCANS.capture('calibration', 'start'); };
    $('scRepBtn').onclick = function () { SCANS.capture('repeat', 'start'); };
    $('scRunStop').onclick = function () {
      const s = SCANS.state, run = s && s.capture && s.capture.on;
      if (run) SCANS.capture(run, 'stop');
    };
    $('scRunGo').onclick = function () {
      $('scRunGo').disabled = true;
      ask('scans_calibrate', function () { SCANS.refresh(); });
    };
    $('scDayBtn').onclick = function () {
      ask('scans_process', JSON.stringify({ day: '', all: false }), function () { SCANS.refresh(); });
    };
    $('scDayAll').onclick = function () {
      ask('scans_process', JSON.stringify({ day: '', all: true }), function () { SCANS.refresh(); });
    };
    $('scOpen').onclick = function () { ask('scans_open'); };
    $('scRepN').onchange = function () {
      SCANS.repTyped = true;
      $('scRepN').value = String(SCANS.repN());
      if (SCANS.state) SCANS.renderPanel(SCANS.state);
    };
    ['scCalBtn', 'scRepBtn', 'scDayBtn', 'scDayAll', 'scSim', 'scRunGo'].forEach(function (id) { $(id).setAttribute('data-sound', 'forward'); });
    if (SCANS.state && SCANS.showing()) SCANS.render(SCANS.state);
  },

  // what the tests read
  probe: function () {
    const s = SCANS.state || {};
    return {
      showing: SCANS.showing(), trial: s.trial ? s.trial.trial : '', run: s.capture ? s.capture.on : '',
      slots: (s.slots || []).map(function (x) { return [x.key, x.state, x.n_spheres, x.arm_seen]; }),
      result: (s.result || {}).state || '', seconds: (s.result || {}).seconds, targets: $('scTargets').children.length,
      tiles: Array.from($('scTiles').querySelectorAll('.tile-v')).map(function (x) { return x.textContent; }),
      watch: $('scWatchW').textContent, calibrated: !!(s.calibration && s.calibration.done),
    };
  },
};
