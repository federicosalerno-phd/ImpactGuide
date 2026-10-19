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
'use strict';

/** Calibrate the phone: its state from calib_state, the photos and the result. */
const CAL = {
  state: null,
  last: null,       // {action, mode, ok}: the last calibration or check of this session
  exclude: null,    // the photos left out by hand for the next calibration (null: as the result has them)
  excludeFor: '',   // the result `exclude` belongs to
  waiting: false,   // the stream of the streaming mode was asked for and has not started yet

  enter: function (step) {
    showView(step === 'board' ? 'board' : 'cal-' + step);
    showPanel('cal-' + step);
    if (step !== 'board') BOARDVIEW.place(null);
    if (step === 'board' && BOARD.state) BOARD.stage(BOARD.state);
    CAL.refresh(function (s) { CAL.stream(step === 'photos' && s.mode === 'stream'); });
    be('phone_refresh');
  },

  // The stream of a calibration in the streaming mode: Calibrate's own while it lasts.
  streaming: function (s) {
    return !!(s && s.live && s.live.purpose === 'calibration' && (s.live.running || s.live.starting));
  },

  // The Photos step left: the live video of the stream mode is stopped.
  leave: function () {
    if (!CAL.streaming(CAL.state) && !CAL.waiting) return;
    setLiveOwner('live');
    workOff('cal_stream');
    ask('calib_stream', false);
  },

  stream: function (on) {
    const running = CAL.streaming(CAL.state);
    if (on && !running) {
      setLiveOwner('calibration');
      ask('calib_stream', true, function (r) {
        if (!r) return;
        // the stream opens in the background: a 'live' event of purpose calibration says how it went
        if (r.busy && r.busy.indexOf('live') >= 0 && !(r.live && r.live.running)) {
          CAL.waiting = true;
          workOn('cal_stream', 'pill.live.starting');
        }
        CAL.render(r);
      });
    } else if (!on && running) {
      setLiveOwner('live');
      ask('calib_stream', false, function (r) { if (r) CAL.render(r); });
    }
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
    if (s.live && s.live.purpose) setLiveOwner(s.live.purpose);
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
      itemOff(it, check);
    });
    $('calPlanQ').textContent = fmtNum(s.plans.quick.n, 0);
    $('calPlanA').textContent = fmtNum(s.plans.accurate.n, 0);
    $('calPhotosSub').textContent = T(check ? 'cal.photos.sub.check' : stream ? 'cal.photos.sub.stream' : 'cal.photos.sub');

    // the counter: how many kept, the least to calibrate and the most the plan takes
    $('cpN').textContent = fmtNum(n, 0);
    $('cpOf').textContent = T('cal.count.of', fmtNum(lim.max, 0));
    $('cpBar').style.transform = 'scaleX(' + Math.min(1, n / lim.max) + ')';
    $('cpMin').style.left = (100 * lim.min / lim.max) + '%';
    $('cpMin').title = T('cal.count.min', lim.min);
    const ok = n >= lim.min, full = n >= lim.max;
    $('cpCard').classList.toggle('enough', ok);
    $('cpSay').textContent = full ? T('cal.count.full') : ok ? T('cal.count.enough', fmtNum(lim.max - n, 0))
      : n ? T('cal.count.more', fmtNum(lim.min - n, 0)) : T('cal.count.none', fmtNum(lim.min, 0));
    const changed = s.series && !s.series.same && s.series.n > 0;
    $('cpNote').textContent = changed ? T('cal.board.changed') : '';
    $('cpNote').classList.toggle('hidden', !changed);

    // the stage: the board (on this monitor) or the live picture, beside the guide
    if (step === 'photos') CAL.stage(s, screen, stream, full);

    // Take: the next photo; the board first when it is the one on this monitor
    const bs = s.board_ui, needBoard = bs.kind === 'screen' && !bs.layout, busyNow = CAL.busy(s);
    const take = $('calTake');
    $('calTakeL').textContent = T(stream ? 'cal.take.frame' : 'cal.take');
    const streamOff = stream && !(s.live && s.live.purpose === 'calibration' && s.live.running);
    take.disabled = busyNow || needBoard || full || streamOff || !s.board_ui.active;
    take.title = needBoard ? T('cal.need.board') : full ? T('cal.count.full') : '';
    $('calNext').classList.toggle('btn-primary', ok);
    $('calNext').classList.toggle('btn-2', !ok);
    $('calKind').checked = check;
    // while a photo is taken or checked, nothing else is added to the series
    $('calOpen').disabled = !s.series;
    $('calRetake').disabled = !(s.series && s.series.last && s.series.last.ok) || busyNow;
    $('calImport').disabled = busyNow;
    $('calNew').disabled = busyNow;
  },

  // The Photos step's stage: the board on this monitor or the live picture, the guide of the next photo, the last one.
  stage: function (s, screen, stream, full) {
    $('bdHostPhotos').classList.toggle('hidden', !screen);
    BOARDVIEW.place(screen ? 'bdHostPhotos' : null);
    // the picture stays while frames keep coming, as in the Live AR view (one bad frame does not take it away)
    const lv = s.live, on = stream && lv && lv.purpose === 'calibration' && lv.running && (!lv.error || (lv.fps_ar || 0) > 0);
    // the live picture: in the guide's column, above the guide, beside the board on this monitor (the board
    // keeps the room); large, beside the guide, with a physical board
    const pl = $('phLive');
    if (screen && pl.parentElement !== $('phGuide')) $('phGuide').prepend(pl);
    if (!screen && pl.parentElement !== $('phGuide').parentElement) $('phGuide').before(pl);
    pl.classList.toggle('hidden', !stream);
    const img = $('phLiveImg');
    if (on && img.dataset.src !== lv.url) { img.dataset.src = lv.url; img.src = lv.url + '?t=' + Date.now(); }
    if (!on && img.dataset.src) { img.dataset.src = ''; img.removeAttribute('src'); }
    if (lv && lv.display && lv.display[0] > 0) pl.style.setProperty('--ar', String(lv.display[0] / lv.display[1]));
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
      $('phLastT').textContent = last.ok ? T('cal.last.kept', fmtNum(last.corners, 0))
        : T('cal.last.retake', Tcode('cal.why.', last.why, 'none'));
      const kept = (s.series.photos || []).map(function (p, i) { return p.ok ? s.series.points[i] : null; })
        .filter(function (x) { return x; });
      FIG.cov($('phCov'), kept, last.points, last.ok);
    }
  },

  // ── the result ───────────────────────────────────────────────────────────
  // The photos left out by the result itself (by hand, before).
  excludedOf: function (r) {
    return r ? r.photos.filter(function (p) { return p.state === 'excluded'; }).map(function (p) { return p.name; }) : [];
  },

  result: function (s) {
    const check = s.kind === 'check', r = s.result, n = s.count, calBusy = s.busy.indexOf('calibration') >= 0;
    const last = CAL.last && CAL.last.mode === s.mode ? CAL.last : null;
    const verdict = CAL.verdict(s, check, r, n, calBusy, last);
    $('crTitle').textContent = verdict.title;
    $('crLine').textContent = verdict.line;
    $('crDot').className = 'live-dot' + (verdict.dot ? ' ' + verdict.dot : '');
    $('calResTitle').textContent = T(check ? 'cal.check.title' : 'cal.result.title');
    $('calResSub').textContent = T('cal.mode.' + s.mode);

    // the figures: the error of each photo, and over the board
    const show = !check && r && !calBusy;
    $('crFigs').classList.toggle('hidden', !show);
    let leave = [];
    if (show) {
      const key = r.images + '|' + r.date + '|' + r.rms_px;
      if (CAL.excludeFor !== key) { CAL.exclude = null; CAL.excludeFor = key; }
      const out = CAL.exclude || CAL.excludedOf(r);
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
    $('calPlain').replaceChildren.apply($('calPlain'), r && !check ? CAL.plain(r) : []);
    const changedOut = show && JSON.stringify(leave.slice().sort()) !== JSON.stringify(CAL.excludedOf(r).sort());
    $('calLeft').textContent = show && leave.length ? T('cal.res.left', leave.length) : '';
    $('calLeft').classList.toggle('hidden', !(show && leave.length));

    // Advanced: the stored calibration of each mode (its day, short, and its error), and the log of the last run
    $('calStored').replaceChildren.apply($('calStored'), ['photo', 'stream'].map(function (md) {
      const x = s.modes[md].stored, c = s.last_check[md];
      const row = piRow(T('cal.stored.' + md), x ? T('cal.stored.row', fmtDay(x.date), fmtNum(x.rms_px, 2)) : T('cal.stored.no'));
      if (c) row.title = T(c.ok === '1' ? 'cal.check.ok' : 'cal.check.failed', fmtTime(c.time));
      return row;
    }));
    // the log stays in a file: a button opens it, once a calibration or a check has run (a newer backend only)
    $('calLogOpen').classList.toggle('hidden', !(CAL.last && hasSlot('calib_log_open')));

    // the one button: Calibrate, Calibrate again (photos left out or put back), Save, or Check
    const what = check ? 'check' : changedOut ? 'again' : r && !r.saved ? 'save' : 'calibrate';
    const run = $('calRun');
    if (!run._armed) $('calRunL').textContent = T({ check: 'cal.check.action', again: 'cal.run.again', save: 'cal.save.action',
                                                    calibrate: 'cal.run.action' }[what]);
    run.dataset.what = what;
    run.disabled = calBusy || (what !== 'save' && (n < s.min_views || (check && !s.modes[s.mode].stored)));
    run.title = what !== 'save' && n < s.min_views ? T('cal.run.few', s.min_views) : '';
  },

  // The card of the result: working, the check's outcome, the calibration's verdict, or what is still needed.
  verdict: function (s, check, r, n, calBusy, last) {
    const ready = function () { return n >= s.min_views ? T('cal.res.ready', fmtNum(n, 0)) : T('cal.res.few', fmtNum(s.min_views, 0)); };
    if (calBusy) return { title: T(check ? 'cal.res.checking' : 'cal.res.working'), line: T('cal.res.wait'), dot: '' };
    if (check) {
      if (last && last.action === 'check') {
        return { title: T(last.ok ? 'cal.res.check.ok' : 'cal.res.check.failed'),
                 line: T(last.ok ? 'cal.res.check.ok.line' : 'cal.res.check.failed.line'), dot: last.ok ? 'ok' : 'warn' };
      }
      if (!s.modes[s.mode].stored) return { title: T('cal.res.none'), line: T('cal.res.check.none'), dot: '' };
      return { title: T('cal.res.check.ready'), line: ready(), dot: '' };
    }
    if (r) {
      // why it is not good: too few photos, a focal length known too loosely (focal_ok false), else the error
      const why = r.few ? 'cal.res.v.few' : r.focal_ok === false && !(r.rms_px >= r.max_rms) ? 'cal.res.v.focal'
        : 'cal.res.v.high';
      return { title: Tcode('cal.res.v.', r.verdict, 'poor'),
               line: T('cal.res.v.line', fmtNum(r.rms_px, 2), fmtNum(r.n_views, 0))
                 + (r.verdict === 'good' ? '' : ' · ' + T(why)),
               dot: r.verdict === 'good' ? 'ok' : 'warn' };
    }
    return { title: T('cal.res.none'), line: ready(), dot: '' };
  },

  // The intrinsics in plain units, one row each.
  plain: function (r) {
    const pl = r.plain;
    const sgn = function (v) { return (v >= 0 ? '+' : '−') + fmtNum(Math.abs(v), 1); };
    const rows = [piRow(T('cal.p.focal'), fmtNum((pl.fx + pl.fy) / 2, 1) + ' px' + (pl.f_std ? ' ± ' + fmtNum(pl.f_std, 1) : '')),
                  piRow(T('cal.p.fov'), fmtNum(pl.fov_x, 1) + '° × ' + fmtNum(pl.fov_y, 1) + '°'),
                  piRow(T('cal.p.centre'), sgn(pl.cx_off) + ', ' + sgn(pl.cy_off) + ' px'),
                  piRow(T('cal.p.bend'), fmtNum(pl.bend_px, 1) + ' px'),
                  piRow(T('cal.p.photos'), T('cal.p.photos.v', r.n_views, r.of)),
                  piRow(T('cal.p.size'), pl.size[0] + ' × ' + pl.size[1])];
    rows[0].title = 'fx ' + fmtNum(pl.fx, 2) + ' · fy ' + fmtNum(pl.fy, 2);
    rows[2].title = T('cal.p.centre.hint');
    rows[3].title = T('cal.p.bend.hint');
    return rows;
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
      if (a) { workOn('cal_photo', 'pill.shooting'); CAL.refresh(); return; }
      if (bad) CAL.refresh();
    });
  },

  run: function () {
    const s = CAL.state, what = $('calRun').dataset.what;
    if (what === 'save') {
      const st = s.modes[s.mode].stored;
      if (st && !armed($('calRun'), T('cal.save.replace', fmtDay(st.date)))) return;
      ask('calib_save', function (r) { if (r) { CAL.render(r); toast(T('cal.save.done', T('cal.mode.' + s.mode))); } });
      return;
    }
    if (what === 'check') {
      ask('calib_check', function (a) { if (a) { workOn('calib', 'pill.checking'); CAL.refresh(); } });
      return;
    }
    const out = CAL.exclude || CAL.excludedOf(s.result);
    ask('calib_run', JSON.stringify(what === 'calibrate' && !CAL.exclude ? [] : out), function (a) {
      if (!a) return;
      workOn('calib', 'pill.calibrating');
      CAL.refresh();
    });
  },

  onEvent: function (e) {
    BOARD.onEvent(e);     // the board's controls live in this module and in Live AR
    if (e.type === 'calib') {
      workOff('calib');
      CAL.last = { action: e.action, mode: e.mode, ok: !!e.ok };
      CAL.exclude = null;
      if (e.error) toast(errorText(e), TOAST_LONG_MS);
      CAL.refresh();
    } else if (e.type === 'calib_photo' || (e.type === 'acquire_failed' && e.key === 'calibration')) {
      workOff('cal_photo');
      CAL.refresh();
    } else if (e.type === 'live' && livePurpose(e) === 'calibration') {
      // the stream of the streaming mode (the Live AR view never handles these)
      const asked = CAL.waiting;
      if (e.action === 'started' || e.action === 'failed' || (e.action === 'stopped' && e.reason === 'cancelled')) {
        CAL.waiting = false;
        workOff('cal_stream');
      }
      if (e.action === 'failed') toast(liveErrorText(e), TOAST_LONG_MS);
      // the Photos step was left while the stream was opening: it is not wanted any more
      const here = page === 'calibration' && STEP.calibration === 'photos';
      if (e.action === 'started' && asked && !here) ask('calib_stream', false);
      if (page === 'calibration') CAL.refresh();
    }
  },

  wire: function () {
    document.querySelectorAll('#calModes .mitem').forEach(function (it) {
      onActivate(it, function () { CAL.set({ mode: it.dataset.mode }); });
    });
    document.querySelectorAll('#calPlans .mitem').forEach(function (it) {
      onActivate(it, function () { if (!it.classList.contains('off')) CAL.set({ plan: it.dataset.plan }); });
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
    $('calLogOpen').onclick = function () { if (hasSlot('calib_log_open')) ask('calib_log_open'); };
    BOARD.wire();         // its two copies, here and in Live AR
  },
};
