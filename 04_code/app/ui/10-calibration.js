// ImpactGuide · Calibrate the phone, three steps on the rail.
//   Board      this monitor or a physical board (15-board.js).
//   Photos     the camera mode (the phone's photos or the live video), quick or
//              accurate, then one photo after the other as the guide shows: how
//              to hold the phone for the next one (12-figures.js). Each photo is
//              checked as it comes (board found, corners, blur, the same size as
//              the others) and kept, or put aside with the reason: its pose comes
//              again. A counter with the least and the most the plan takes.
//   Calibrate  the calibration and its result: a verdict, the error of each
//              photo (a press on a bar leaves that photo out or puts it back,
//              then Calibrate again), where on the board the error is, the
//              intrinsics in plain units, Save.
// The session check, adding files, the folder and Start again sit under
// Advanced. The work is calibrate_charuco.py's and calib_guide.py's; the page
// shows, chooses and asks.

const CAL = {
  state: null,
  last: null,       // {action, mode, ok, log, message}: the last calibration or check of this session
  exclude: null,    // the photos left out by hand for the next calibration (null: as the result has them)

  enter: function (step) {
    showView(step === 'board' ? 'board' : 'cal-' + step);
    showPanel('cal-' + step);
    if (step !== 'board') BOARDVIEW.place(null);
    if (step === 'board' && BOARD.state) BOARD.stage(BOARD.state);
    CAL.refresh(function (s) { CAL.stream(step === 'photos' && s.mode === 'stream'); });
    be('phone_refresh');
  },

  // The Photos step left: the live video of the stream mode is stopped.
  leave: function () {
    const s = CAL.state;
    if (s && s.live && s.live.purpose === 'calibration' && (s.live.running || s.live.starting)) ask('calib_stream', false);
  },

  stream: function (on) {
    const s = CAL.state;
    const running = s && s.live && s.live.purpose === 'calibration' && (s.live.running || s.live.starting);
    if (on && !running) ask('calib_stream', true, function (r) { if (r) CAL.render(r); });
    else if (!on && running) ask('calib_stream', false, function (r) { if (r) CAL.render(r); });
  },

  refresh: function (then) {
    ask('calib_state', function (s) {
      if (!s) return;
      CAL.render(s);
      if (then) then(s);
    });
  },

  render: function (s) {
    CAL.state = s;
    BOARD.render(s.board_ui);
    $('where').textContent = '';
    if (page !== 'calibration') return;
    CAL.photos(s);
    CAL.result(s);
  },

  busy: function (s) {
    return ['phone', 'photo_check', 'calibration', 'live'].some(function (k) { return s.busy.indexOf(k) >= 0; });
  },

  // ── the photos ───────────────────────────────────────────────────────────
  photos: function (s) {
    const n = s.count, lim = s.limits, check = s.kind === 'check', screen = s.board_ui.kind === 'screen';
    const stream = s.mode === 'stream', step = STEP.calibration;
    document.querySelectorAll('#calModes .mitem').forEach(function (it) { it.classList.toggle('sel', it.dataset.mode === s.mode); });
    document.querySelectorAll('#calPlans .mitem').forEach(function (it) {
      it.classList.toggle('sel', !check && it.dataset.plan === s.plan);
      it.classList.toggle('off', check);
    });
    $('calPlanQ').textContent = cifre(s.plans.quick.n, 0);
    $('calPlanA').textContent = cifre(s.plans.accurate.n, 0);
    $('calPhotosSub').textContent = T(check ? 'cal.photos.sub.check' : stream ? 'cal.photos.sub.stream' : 'cal.photos.sub');

    // the counter: how many kept, the least to calibrate and the most the plan takes
    $('cpN').textContent = cifre(n, 0);
    $('cpOf').textContent = T('cal.count.of', cifre(lim.max, 0));
    $('cpBar').style.transform = 'scaleX(' + Math.min(1, n / lim.max) + ')';
    $('cpMin').style.left = (100 * lim.min / lim.max) + '%';
    $('cpMin').title = T('cal.count.min', lim.min);
    const ok = n >= lim.min, full = n >= lim.max;
    $('cpCard').classList.toggle('enough', ok);
    $('cpSay').textContent = full ? T('cal.count.full') : ok ? T('cal.count.enough', cifre(lim.max - n, 0))
      : n ? T('cal.count.more', cifre(lim.min - n, 0)) : T('cal.count.none', cifre(lim.min, 0));
    const changed = s.series && !s.series.same && s.series.n > 0;
    $('cpNote').textContent = changed ? T('cal.board.changed') : '';
    $('cpNote').classList.toggle('hidden', !changed);

    // the stage: the board (on this monitor) or the live picture, beside the guide
    if (step === 'photos') {
      $('bdHostPhotos').classList.toggle('hidden', !screen);
      BOARDVIEW.place(screen ? 'bdHostPhotos' : null);
      const lv = s.live, on = stream && lv && lv.purpose === 'calibration' && lv.running && !lv.error;
      // the live picture: in the guide's column, above the guide, beside the board on this monitor (the board
      // keeps the room); large, beside the guide, with a physical board
      const pl = $('phLive');
      if (screen && pl.parentElement !== $('phGuide')) $('phGuide').prepend(pl);
      if (!screen && pl.parentElement !== $('phGuide').parentElement) $('phGuide').before(pl);
      pl.classList.toggle('hidden', !stream);
      const img = $('phLiveImg');
      if (on && img.dataset.src !== lv.url) { img.dataset.src = lv.url; img.src = lv.url + '?t=' + Date.now(); }
      if (!on && img.dataset.src) { img.dataset.src = ''; img.removeAttribute('src'); }
      if (lv && lv.display && lv.display[0] > 0) $('phLive').style.setProperty('--ar', String(lv.display[0] / lv.display[1]));
      $('phLiveTag').textContent = !on ? T(lv && lv.starting ? 'live.idle.title.starting' : 'cal.stream.off')
        : T(lv.tracked ? 'cal.stream.seen' : 'cal.stream.none');
      $('phLiveTag').classList.toggle('hidden', false);
      $('phGuide').classList.toggle('wide', !screen && !stream);
      if (s.next) FIG.guide($('phGuideFig'), s.next, !screen && !stream);
      else {
        const why = full ? 'cal.count.full' : screen && !s.board_ui.layout ? 'cal.need.board' : 'cal.guide.none';
        $('phGuideFig').replaceChildren(E('div', 'guide-done', T(why)));
      }
      const last = s.series && s.series.same ? s.series.last : null;
      $('phLast').classList.toggle('hidden', !last);
      if (last) {
        $('phLastDot').className = 'live-dot ' + (last.ok ? 'ok' : 'warn');
        $('phLastT').textContent = last.ok ? T('cal.last.kept', cifre(last.corners, 0))
          : T('cal.last.retake', T('cal.why.' + (last.why || 'none')));
        const kept = (s.series.photos || []).map(function (p, i) { return p.ok ? s.series.points[i] : null; })
          .filter(function (x) { return x; });
        FIG.cov($('phCov'), kept, last.points, last.ok);
      }
    }

    // Take: the next photo; the board first when it is the one on this monitor
    const bs = s.board_ui, needBoard = bs.kind === 'screen' && !bs.layout;
    const take = $('calTake');
    $('calTakeL').textContent = T(stream ? 'cal.take.frame' : 'cal.take');
    const streamOff = stream && !(s.live && s.live.purpose === 'calibration' && s.live.running);
    take.disabled = CAL.busy(s) || needBoard || full || streamOff || !s.board_ui.active;
    take.title = needBoard ? T('cal.need.board') : full ? T('cal.count.full') : '';
    $('calNext').classList.toggle('btn-primary', ok);
    $('calNext').classList.toggle('btn-2', !ok);
    $('calKind').checked = check;
    $('calOpen').disabled = !s.series;
    $('calRetake').disabled = !(s.series && s.series.last && s.series.last.ok) || CAL.busy(s);
  },

  // ── the result ───────────────────────────────────────────────────────────
  result: function (s) {
    const check = s.kind === 'check', r = s.result, n = s.count, calBusy = s.busy.indexOf('calibration') >= 0;
    const last = CAL.last && CAL.last.mode === s.mode ? CAL.last : null;
    let title, line, dot = '';
    if (calBusy) {
      title = T(check ? 'cal.res.checking' : 'cal.res.working');
      line = T('cal.res.wait');
    } else if (check) {
      if (last && last.action === 'check') {
        title = T(last.ok ? 'cal.res.check.ok' : 'cal.res.check.failed');
        line = T(last.ok ? 'cal.res.check.ok.line' : 'cal.res.check.failed.line');
        dot = last.ok ? 'ok' : 'warn';
      } else if (!s.modes[s.mode].stored) {
        title = T('cal.res.none');
        line = T('cal.res.check.none');
      } else {
        title = T('cal.res.check.ready');
        line = n >= s.min_views ? T('cal.res.ready', cifre(n, 0)) : T('cal.res.few', cifre(s.min_views, 0));
      }
    } else if (r) {
      title = T('cal.res.v.' + r.verdict);
      line = T('cal.res.v.line', cifre(r.rms_px, 2), cifre(r.n_views, 0))
        + (r.verdict === 'good' ? '' : ' · ' + T(r.few ? 'cal.res.v.few' : 'cal.res.v.high'));
      dot = r.verdict === 'good' ? 'ok' : 'warn';
    } else {
      title = T('cal.res.none');
      line = n >= s.min_views ? T('cal.res.ready', cifre(n, 0)) : T('cal.res.few', cifre(s.min_views, 0));
    }
    $('crTitle').textContent = title;
    $('crLine').textContent = line;
    $('crDot').className = 'live-dot' + (dot ? ' ' + dot : '');
    $('calResTitle').textContent = T(check ? 'cal.check.title' : 'cal.result.title');
    $('calResSub').textContent = T('cal.mode.' + s.mode);

    // the figures: the error of each photo, and over the board
    const show = !check && r && !calBusy;
    $('crFigs').classList.toggle('hidden', !show);
    let leave = [];
    if (show) {
      const key = r.images + '|' + r.date + '|' + r.rms_px;
      if (CAL.excludeFor !== key) { CAL.exclude = null; CAL.excludeFor = key; }
      const out = CAL.exclude || r.photos.filter(function (p) { return p.state === 'excluded'; }).map(function (p) { return p.name; });
      leave = out;
      const photos = r.photos.map(function (p) {
        return { name: p.name, err: p.err, state: out.indexOf(p.name) >= 0 ? 'excluded' : p.state === 'excluded' ? 'used' : p.state };
      });
      FIG.bars($('crBars'), photos, r.limit_px, function (name) {
        const cur = (CAL.exclude || out).slice(), i = cur.indexOf(name);
        if (i >= 0) cur.splice(i, 1); else cur.push(name);
        CAL.exclude = cur;
        CAL.render(CAL.state);
      });
      const dropped = r.photos.filter(function (p) { return p.state === 'dropped'; }).length;
      $('crBarsNote').textContent = T('cal.res.bars.note') + (dropped ? ' · ' + T('cal.res.dropped', dropped) : '');
      const m = FIG.map($('crMap'), r.residuals, r.grid);
      $('crMapNote').textContent = m ? T('cal.res.map.note') : '';
    }

    // the panel: the intrinsics in plain units
    const rows = [];
    if (r && !check) {
      const pl = r.plain;
      const sgn = function (v) { return (v >= 0 ? '+' : '−') + cifre(Math.abs(v), 1); };
      rows.push(piRow(T('cal.p.focal'), cifre((pl.fx + pl.fy) / 2, 1) + ' px' + (pl.f_std ? ' ± ' + cifre(pl.f_std, 1) : '')),
                piRow(T('cal.p.fov'), cifre(pl.fov_x, 1) + '° × ' + cifre(pl.fov_y, 1) + '°'),
                piRow(T('cal.p.centre'), sgn(pl.cx_off) + ', ' + sgn(pl.cy_off) + ' px'),
                piRow(T('cal.p.bend'), cifre(pl.bend_px, 1) + ' px'),
                piRow(T('cal.p.photos'), T('cal.p.photos.v', r.n_views, r.of)),
                piRow(T('cal.p.size'), pl.size[0] + ' × ' + pl.size[1]));
      rows[0].title = 'fx ' + cifre(pl.fx, 2) + ' · fy ' + cifre(pl.fy, 2);
      rows[2].title = T('cal.p.centre.hint');
      rows[3].title = T('cal.p.bend.hint');
    }
    $('calPlain').replaceChildren.apply($('calPlain'), rows);
    const changedOut = show && JSON.stringify(leave.slice().sort()) !== JSON.stringify(
      r.photos.filter(function (p) { return p.state === 'excluded'; }).map(function (p) { return p.name; }).sort());
    $('calLeft').textContent = show && leave.length ? T('cal.res.left', leave.length) : '';
    $('calLeft').classList.toggle('hidden', !(show && leave.length));

    // Advanced: the stored calibration of each mode (its day, short, and its error), and the log of the last run
    const day = function (iso) {
      const d = new Date(String(iso || '').slice(0, 10) + 'T12:00:00');
      if (isNaN(d)) return String(iso || '');
      const o = { day: 'numeric', month: 'short' };
      if (d.getFullYear() !== new Date().getFullYear()) o.year = 'numeric';
      return d.toLocaleDateString(locale(), o);
    };
    $('calStored').replaceChildren.apply($('calStored'), ['photo', 'stream'].map(function (md) {
      const x = s.modes[md].stored, c = s.last_check[md];
      const row = piRow(T('cal.stored.' + md), x ? T('cal.stored.row', day(x.date), cifre(x.rms_px, 2)) : T('cal.stored.no'));
      if (c) row.title = T(c.ok === '1' ? 'cal.check.ok' : 'cal.check.failed', quando(c.time));
      return row;
    }));
    const showLog = $('calShowLog').checked && last && (last.log || last.message);
    $('calLog').classList.toggle('hidden', !showLog);
    $('calLog').textContent = showLog ? (last.log || last.message) : '';

    // the one button: Calibrate, Calibrate again (photos left out or put back), Save, or Check
    const what = check ? 'check' : changedOut ? 'again' : r && !r.saved ? 'save' : 'calibrate';
    const run = $('calRun');
    if (!run._armed) $('calRunL').textContent = T({ check: 'cal.check.action', again: 'cal.run.again', save: 'cal.save.action',
                                                    calibrate: 'cal.run.action' }[what]);
    run.dataset.what = what;
    run.disabled = calBusy || (what !== 'save' && (n < s.min_views || (check && !s.modes[s.mode].stored)));
    run.title = what !== 'save' && n < s.min_views ? T('cal.run.few', s.min_views) : '';
  },

  set: function (change) {
    ask('calib_set', JSON.stringify(change), function (s) {
      if (!s) return;
      CAL.render(s);
      if (change.mode !== undefined) CAL.stream(STEP.calibration === 'photos' && s.mode === 'stream');
    });
  },

  take: function () {
    ask('calib_take', function (a, bad) {
      if (a) { work(true, T('pill.shooting')); CAL.refresh(); return; }
      if (bad) CAL.refresh();
    });
  },

  run: function () {
    const s = CAL.state, what = $('calRun').dataset.what;
    if (what === 'save') {
      const st = s.modes[s.mode].stored;
      if (st && !armed($('calRun'), T('cal.save.replace', st.date))) return;
      ask('calib_save', function (r) { if (r) { CAL.render(r); toast(T('cal.save.done', T('cal.mode.' + s.mode))); } });
      return;
    }
    if (what === 'check') {
      ask('calib_check', function (a) { if (a) { work(true, T('pill.checking')); CAL.refresh(); } });
      return;
    }
    const out = CAL.exclude || (s.result ? s.result.photos.filter(function (p) { return p.state === 'excluded'; })
      .map(function (p) { return p.name; }) : []);
    ask('calib_run', JSON.stringify(what === 'calibrate' && !CAL.exclude ? [] : out), function (a) {
      if (!a) return;
      work(true, T('pill.calibrating'));
      CAL.refresh();
    });
  },

  onEvent: function (e) {
    BOARD.onEvent(e);     // the board's controls live in this module and in Live AR
    if (e.type === 'calib') {
      work(false);
      CAL.last = { action: e.action, mode: e.mode, ok: !!e.ok, log: e.log, message: e.message };
      CAL.exclude = null;
      if (e.error) toast(sayError(e.error, e.message), 5000);
      CAL.refresh();
    } else if (e.type === 'calib_photo') {
      work(false);
      CAL.refresh();
    } else if (e.type === 'acquire_failed' && e.key === 'calibration') {
      work(false);
      CAL.refresh();
    } else if (e.type === 'live' && page === 'calibration') {
      if (e.action === 'started' || e.action === 'failed') work(false);
      CAL.refresh();
    }
  },

  wire: function () {
    const key = function (it) {
      it.onkeydown = function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); it.onclick(); } };
    };
    document.querySelectorAll('#calModes .mitem').forEach(function (it) {
      it.onclick = function () { CAL.set({ mode: it.dataset.mode }); };
      key(it);
    });
    document.querySelectorAll('#calPlans .mitem').forEach(function (it) {
      it.onclick = function () { if (!it.classList.contains('off')) CAL.set({ plan: it.dataset.plan }); };
      key(it);
    });
    $('calKind').onchange = function () { CAL.set({ kind: $('calKind').checked ? 'check' : 'calibration' }); };
    $('calTake').onclick = CAL.take;
    $('calRetake').onclick = function () { ask('calib_retake', function (s) { if (s) CAL.render(s); }); };
    $('calImport').onclick = function () {
      ask('calib_import', '', function (s) {
        if (!s) return;
        CAL.render(s);
        if (s.added || s.skipped) toast(T('cal.photos.added', s.added, s.skipped));
      });
    };
    $('calOpen').onclick = function () { ask('calib_open'); };
    $('calNew').onclick = function () {
      if (!armed($('calNew'), T('cal.series.new.sure'))) return;
      ask('calib_series_new', function (s) { if (s) { CAL.render(s); toast(T('cal.series.made')); } });
    };
    $('calRun').onclick = CAL.run;
    $('calShowLog').onchange = function () { if (CAL.state) CAL.render(CAL.state); };
    BOARD.wire();         // its two copies, here and in Live AR
  },
};
