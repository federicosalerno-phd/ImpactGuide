// ImpactGuide · Calibrate the phone, three steps on the rail: the board (on a
// monitor or printed, 15-board.js), the photos (a counter and the moment they
// are enough), the calibration (its result in one line, then Save). The mode
// (photo or video), the session check, the folders and the files sit under
// Advanced. The work is calibrate_charuco.py's; the page only asks the backend,
// which keeps the photos of today in a series per mode (impactguide_app.py).

const CAL = {
  state: null,
  last: null,      // {action, mode, ok, log, message}: the last calibration or check of this session

  enter: function (step) {
    showView(step === 'board' ? 'board' : 'cal-' + step);
    showPanel('cal-' + step);
    if (step === 'board' && BOARD.state) BOARD.card(BOARD.state);
    CAL.refresh();
    be('phone_refresh');
  },

  refresh: function (then) {
    ask('calib_state', function (s) {
      if (!s) return;
      CAL.render(s);
      if (then) then(s);
    });
  },

  // the series the photos go to now, when its photos are of the board in use
  series: function (s) { return s.series && s.series.same ? s.series : null; },
  count: function (s) { const f = CAL.series(s); return f ? f.n : 0; },

  render: function (s) {
    CAL.state = s;
    BOARD.render(s.board_ui);
    if (page === 'calibration' && STEP.calibration === 'board') BOARD.card(s.board_ui);
    const check = s.kind === 'check', n = CAL.count(s), enough = s.enough, ok = n >= enough;
    const busyNow = s.busy.indexOf('calibration') >= 0;
    const mode = T('cal.mode.' + s.mode);
    $('where').textContent = page === 'calibration'
      ? (check ? T('cal.where.check', mode) : mode) + (s.series ? ' · ' + s.series.name : '') : '';

    // the photos: how many of the board in use, against how many are enough
    $('cpN').textContent = cifre(n, 0);
    $('cpOf').textContent = T('cal.count.of', cifre(enough, 0));
    $('cpBar').style.transform = 'scaleX(' + Math.min(1, n / enough) + ')';
    $('cpCard').classList.toggle('enough', ok);
    $('cpSay').textContent = ok ? T('cal.count.enough') : n ? T('cal.count.more', cifre(enough - n, 0)) : T('cal.count.none');
    $('cpTip').textContent = T(check ? 'cal.tip.check' : 'cal.tip');
    const changed = s.series && !s.series.same && s.series.n > 0;
    $('cpNote').textContent = changed ? T('cal.board.changed') : '';
    $('cpNote').classList.toggle('hidden', !changed);
    const last = CAL.series(s) && CAL.series(s).last;
    $('cpLast').classList.toggle('hidden', !last);
    if (last && $('cpImg').getAttribute('src') !== last) $('cpImg').src = last;
    $('calPhotosSub').textContent = T(check ? 'cal.photos.sub.check' : 'cal.photos.sub');

    // the photos panel: Take photo; the board first when it is the one on a monitor and not on show
    const bs = s.board_ui, needBoard = bs.kind === 'screen' && (!bs.shown || !bs.layout);
    $('calNeedBoard').classList.toggle('hidden', !needBoard);
    $('calTake').disabled = needBoard || busyNow;
    // the clear moment: enough photos lights the way on
    $('calNext').classList.toggle('btn-primary', ok);
    $('calNext').classList.toggle('btn-2', !ok);
    $('calMode').value = s.mode;
    $('calKind').checked = check;
    $('calFolder').textContent = s.series ? s.series.path : T('cal.folder.none');
    $('calOpen').disabled = !s.series;
    // the printed board, under the board's Advanced
    const pr = bs.printed;
    $('calBoard').replaceChildren(
      piRow(T('cal.board.squares'), pr.squares[0] + ' × ' + pr.squares[1]),
      piRow(T('cal.board.marker'), cifre(pr.marker_mm, 2) + ' mm'),
      piRow(T('cal.board.dict'), pr.dictionary));
    $('calBoardPrint').classList.toggle('hidden', bs.kind === 'screen');
    $('calBoardOpen').disabled = !s.board_file;

    CAL.result(s, busyNow);
  },

  // the third step: the calibration, or the session check, and its result in one line
  result: function (s, busyNow) {
    const check = s.kind === 'check', m = s.modes[s.mode], pend = m.pending, st = m.stored;
    const n = CAL.count(s), last = CAL.last && CAL.last.mode === s.mode ? CAL.last : null;
    let title, line, dot = '';
    if (busyNow) {
      title = T(check ? 'cal.res.checking' : 'cal.res.working');
      line = T('cal.res.wait');
    } else if (check && last && last.action === 'check') {
      title = T(last.ok ? 'cal.res.check.ok' : 'cal.res.check.failed');
      line = T(last.ok ? 'cal.res.check.ok.line' : 'cal.res.check.failed.line');
      dot = last.ok ? 'ok' : 'warn';
    } else if (!check && pend) {
      const good = pend.rms_px < s.max_rms;
      title = T(good ? 'cal.res.good' : 'cal.res.high');
      line = T(good ? 'cal.res.good.line' : 'cal.res.high.line', cifre(pend.rms_px, 2), cifre(pend.n_views, 0));
      dot = good ? 'ok' : 'warn';
    } else if (!check && st) {
      title = T('cal.res.stored');
      line = T('cal.res.stored.line', st.date, cifre(st.rms_px, 2), cifre(st.n_views, 0));
      dot = 'ok';
    } else if (check && !st) {
      title = T('cal.res.none');
      line = T('cal.res.check.none');
    } else {
      title = T(check ? 'cal.res.check.ready' : 'cal.res.none');
      line = n >= s.min_views ? T('cal.res.ready', cifre(n, 0)) : T('cal.res.few', cifre(s.min_views, 0));
    }
    $('crTitle').textContent = title;
    $('crLine').textContent = line;
    $('crDot').className = 'live-dot' + (dot ? ' ' + dot : '');
    $('calResTitle').textContent = T(check ? 'cal.check.title' : 'cal.result.title');
    $('calResSub').textContent = T('cal.mode.' + s.mode);
    // the log of the last run, under Advanced
    const showLog = $('calShowLog').checked && last && (last.log || last.message);
    $('calLog').classList.toggle('hidden', !showLog);
    $('calLog').textContent = showLog ? (last.log || last.message) : '';

    // the stored calibrations of the two modes, one row each
    const rows = ['photo', 'video'].map(function (md) {
      const x = s.modes[md].stored, c = s.last_check[md];
      const v = x ? T('cal.stored.row', x.date, cifre(x.rms_px, 2)) : T('cal.stored.no');
      const r = piRow(T('cal.mode.' + md), v);
      if (x) r.title = x.image_size[0] + ' × ' + x.image_size[1] + ' px · fx, fy ' + cifre(x.fx, 1) + ', ' + cifre(x.fy, 1)
        + ' · cx, cy ' + cifre(x.cx, 1) + ', ' + cifre(x.cy, 1) + ' · ' + x.n_views + ' ' + T('cal.f.views')
        + (c ? ' · ' + T(c.ok === '1' ? 'cal.check.ok' : 'cal.check.failed', quando(c.time)) : '');
      return r;
    });
    $('calModes').replaceChildren.apply($('calModes'), rows);

    // the one button: Save when a good calibration waits, else Calibrate (or Check)
    const save = !check && pend && pend.rms_px < s.max_rms;
    const run = $('calRun');
    if (!run._armed) $('calRunL').textContent = T(save ? 'cal.save.action' : check ? 'cal.check.action' : 'cal.run.action');
    run.dataset.what = save ? 'save' : check ? 'check' : 'calibrate';
    run.disabled = busyNow || (!save && (n < s.min_views || (check && !st)));
    run.title = !save && n < s.min_views ? T('cal.run.few', s.min_views) : '';
  },

  set: function (change) {
    ask('calib_set', JSON.stringify(change), function (s) { if (s) CAL.render(s); });
  },

  take: function () {
    ask('calib_take', function (a, bad) {
      if (a) { work(true, T('pill.shooting')); return; }
      if (bad && bad.error === 'board_hidden') CAL.refresh();
    });
  },

  run: function () {
    const s = CAL.state, f = CAL.series(s), what = $('calRun').dataset.what;
    if (what === 'save') {
      const st = s.modes[s.mode].stored;
      if (st && !armed($('calRun'), T('cal.save.replace', st.date))) return;
      ask('calib_save', s.mode, function (r) { if (r) { CAL.render(r); toast(T('cal.save.done', T('cal.mode.' + s.mode))); } });
      return;
    }
    if (!f) return;
    ask(what === 'check' ? 'calib_check' : 'calib_run', f.name, s.mode, function (a) {
      if (!a) return;
      work(true, T(what === 'check' ? 'pill.checking' : 'pill.calibrating'));
      CAL.refresh();
    });
  },

  onEvent: function (e) {
    BOARD.onEvent(e);     // the board's controls live in this module and in Live AR
    if (e.type === 'calib') {
      work(false);
      CAL.last = { action: e.action, mode: e.mode, ok: !!e.ok, log: e.log, message: e.message };
      if (e.error) toast(T('err.generic', e.message || e.error), 5000);
      CAL.refresh();
    } else if (e.type === 'calib_photo') {
      work(false);
      CAL.refresh();
    } else if (e.type === 'acquire_failed' && e.key === 'calibration') {
      work(false);
      CAL.refresh();
    }
  },

  wire: function () {
    setOptions($('calMode'), [T('cal.mode.photo'), T('cal.mode.video')], ['photo', 'video']);
    $('calMode').onchange = function () { CAL.set({ mode: $('calMode').value }); };
    $('calKind').onchange = function () { CAL.set({ kind: $('calKind').checked ? 'check' : 'calibration' }); };
    $('calTake').onclick = CAL.take;
    $('calNeedShow').onclick = BOARD.toggle;
    $('calImport').onclick = function () {
      ask('calib_import', '', function (s) {
        if (!s) return;
        CAL.render(s);
        if (s.added || s.skipped) toast(T('cal.photos.added', s.added, s.skipped));
      });
    };
    $('calOpen').onclick = function () { if (CAL.state && CAL.state.series) ask('open_path', CAL.state.series.path); };
    $('calNew').onclick = function () {
      if (!armed($('calNew'), T('cal.series.new.sure'))) return;
      ask('calib_series_new', function (s) { if (s) { CAL.render(s); toast(T('cal.series.made')); } });
    };
    $('calBoardBtn').onclick = function () { ask('calib_board', function (s) { if (s) CAL.render(s); }); };
    $('calBoardOpen').onclick = function () { ask('open_path', CAL.state.board_dir); };
    $('calRun').onclick = CAL.run;
    $('calShowLog').onchange = function () { if (CAL.state) CAL.render(CAL.state); };
    BOARD.wire();         // its two copies, here and in Live AR
  },
};
