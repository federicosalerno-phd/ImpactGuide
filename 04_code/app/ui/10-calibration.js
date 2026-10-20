// ImpactGuide · Calibrate the phone, three steps on the rail.
//   Board      this monitor or a physical board (15-board.js); one line says
//              which makes clean photos (the plate or a sheet on glass, the
//              phone still on a stand, the board moved).
//   Photos     the camera mode (the phone's photos or the live video), quick or
//              accurate, then one photo after the other as the guide shows: how
//              to hold the phone for the next one (12-figures.js). A photo is
//              taken at once, never after a countdown: with Take, the space bar
//              or Enter, or by itself when the phone holds the guide's pose still
//              (Auto, on unless switched off); a flash and a short sound say so
//              and the guide moves on. Each photo is checked in the background
//              (board found, corners, blur, the same size as the others) and its
//              small picture says later whether it is kept or put aside: its
//              pose comes again; one refused for the camera's settings says
//              which. A counter with the least and the most the plan takes.
//   Calibrate  the calibration and its result: a verdict, never good with a
//              board not flat, a phone that moved or automatic settings, and
//              what spoils it (the worst in full); the error of each
//              photo (a press on a bar leaves that photo out or puts it back,
//              then Calibrate again), where on the board the error is, the
//              intrinsics in plain units, Save; before it, the corners of the
//              photo the board has not reached yet.
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
  keys: false,      // the space bar and Enter take the photo (wired once)

  enter: function (step) {
    showView(step === 'board' ? 'board' : 'cal-' + step);
    showPanel('cal-' + step);
    if (step !== 'board') BOARDVIEW.place(null);
    if (step === 'board' && BOARD.state) BOARD.stage(BOARD.state);
    CAL.refresh(function (s) { CAL.stream(step === 'photos' && s.mode === 'stream'); });
    CAL.watch(step === 'photos');
    be('phone_refresh');
  },

  // The backend told whether the Photos step is open: the photo taken by itself watches the preview meanwhile.
  watch: function (on) {
    if (hasSlot('calib_watch')) askQuiet('calib_watch', !!on, function (a) { if (a) CAL.autoSay(a.auto); });
  },

  // The stream of a calibration in the streaming mode: Calibrate's own while it lasts.
  streaming: function (s) {
    return !!(s && s.live && s.live.purpose === 'calibration' && (s.live.running || s.live.starting));
  },

  // The Photos step left: the photo taken by itself stops watching, the live video of the stream mode is stopped.
  leave: function () {
    CAL.watch(false);
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
    // the board for clean photos, in one line: this monitor only to try (or a flat one)
    const bu = s.board_ui || {};
    $('calBdAdvice').textContent = T(bu.chosen && bu.kind === 'screen' ? 'cal.board.advice.screen' : 'cal.board.advice');
    if (page !== 'calibration') return;
    CAL.photos(s);
    CAL.result(s);
  },

  busy: function (s) {
    return ['phone', 'photo_check', 'calibration', 'live', 'film'].some(function (k) { return s.busy.indexOf(k) >= 0; });
  },

  // What keeps Take from taking: never a photo on its way (taken, copied, checked: a Take then follows at once).
  takeBusy: function (s) {
    return ['calibration', 'live', 'film'].some(function (k) { return s.busy.indexOf(k) >= 0; });
  },

  // ── the photos ───────────────────────────────────────────────────────────
  photos: function (s) {
    const n = s.count, lim = s.limits, check = s.kind === 'check', screen = s.board_ui.kind === 'screen';
    const stream = s.mode === 'stream', film = s.mode === 'video', step = STEP.calibration;
    document.querySelectorAll('#calModes .mitem').forEach(function (it) { it.classList.toggle('sel', it.dataset.mode === s.mode); });
    document.querySelectorAll('#calPlans .mitem').forEach(function (it) {
      it.classList.toggle('sel', !check && it.dataset.plan === s.plan);
      itemOff(it, check);
    });
    $('calPlanQ').textContent = fmtNum(s.plans.quick.n, 0);
    $('calPlanA').textContent = fmtNum(s.plans.accurate.n, 0);
    if (s.plans.bench) $('calPlanB').textContent = fmtNum(s.plans.bench.n, 0);
    const bench = !check && s.plan === 'bench';
    $('calPhotosSub').textContent = T(check ? 'cal.photos.sub.check' : stream ? 'cal.photos.sub.stream'
      : film ? 'cal.photos.sub.video' : 'cal.photos.sub');

    // the counter: how many kept, the least to calibrate and the most the plan takes
    $('cpN').textContent = fmtNum(n, 0);
    $('cpOf').textContent = T('cal.count.of', fmtNum(lim.max, 0));
    $('cpBar').style.transform = 'scaleX(' + Math.min(1, n / lim.max) + ')';
    $('cpMin').style.left = (100 * lim.min / lim.max) + '%';
    $('cpMin').title = T('cal.count.min', lim.min);
    // every pose has its photo but a corner of the photo has no board yet: that corner's pose once more (s.next.extra)
    const extra = s.next && s.next.extra;
    const ok = n >= lim.min, full = n >= lim.max && !extra;
    $('cpCard').classList.toggle('enough', ok);
    // below the plan's least: needed up to the calibration's own least (min_views), advised beyond it
    $('cpSay').textContent = extra ? T('guide.extra', T('cal.corner.' + extra))
      : full ? T('cal.count.full') : ok ? T('cal.count.enough', fmtNum(Math.max(0, lim.max - n), 0))
      : !n ? T('cal.count.none', fmtNum(lim.min, 0)) : n >= s.min_views ? T('cal.count.advised', fmtNum(lim.min - n, 0))
      : T('cal.count.more', fmtNum(s.min_views - n, 0));
    const changed = s.series && !s.series.same && s.series.n > 0;
    $('cpNote').textContent = changed ? T('cal.board.changed') : '';
    $('cpNote').classList.toggle('hidden', !changed);
    // what to change before the next photo: the camera's settings the last one was refused for, else a phone that
    // moved in the last calibration (kept still on a stand); the camera in Pro with its focus set by hand
    const last = s.series && s.series.same ? s.series.last : null;
    // the bench: before the first photo what to set up (the guide says the seat and the arm angle of each stop)
    const warn = last && last.why === 'settings' ? Tcode('cal.set.', (last.problems || [])[0], 'exposure_auto')
      : bench ? (!n ? T('cal.ph.bench') : '')
      : s.moved ? T('cal.ph.stand') : '';
    $('calPhWarn').textContent = warn;
    $('calPhWarn').classList.toggle('hidden', !warn);
    $('calProNote').classList.toggle('hidden', s.mode !== 'photo');

    // the stage: the board (on this monitor) or the live picture, beside the guide
    if (step === 'photos') CAL.stage(s, screen, stream, full);

    // Take: the next photo; the board first when it is the one on this monitor
    const bs = s.board_ui, needBoard = bs.kind === 'screen' && !bs.layout, busyNow = CAL.busy(s);
    const take = $('calTake');
    // a film: the same button starts and stops it (each film is a series of its own, never full)
    const rec = film && s.film && s.film.recording;
    $('calTakeL').textContent = T(rec ? 'cal.take.stop' : film ? 'cal.take.film' : stream ? 'cal.take.frame' : 'cal.take');
    const streamOff = stream && !(s.live && s.live.purpose === 'calibration' && s.live.running);
    // a photo on its way never holds Take back: the next one follows it at once; a film's own work does
    take.disabled = (film ? busyNow : CAL.takeBusy(s))
      || (!rec && (needBoard || (full && !film) || streamOff || !s.board_ui.active));
    take.title = needBoard ? T('cal.need.board') : full ? T('cal.count.full') : film ? '' : T('cal.take.hint');
    CAL.autoSay(s.auto, s);
    CAL.strip(s);
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
      const why = s.mode !== 'video' && full ? 'cal.count.full' : screen && !s.board_ui.layout ? 'cal.need.board'
        : s.mode === 'video' ? (screen ? 'cal.guide.film' : 'cal.guide.film.board') : 'cal.guide.none';
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
      // the last photo with only its own corners, and apart from it where the board has been in the photos kept
      FIG.shot($('phShot'), last);
      FIG.cov($('phCov'), kept, last.points, last.ok, s.series.missing_corners || []);
    }
  },

  // The photo taken by itself: the switch, and in a few words what it waits for (the space bar when it is off).
  autoSay: function (a, s) {
    s = s || CAL.state;
    if (!a || !s) return;
    const on = !!a.on, manual = s.mode === 'video';
    $('calAuto').checked = on;
    $('calAutoRow').classList.toggle('hidden', manual);
    // holding still: on a stand, when the last calibration found the phone moved between its photos
    const hold = s.moved ? 'cal.auto.hold.stand' : 'cal.auto.hold';
    const word = !on ? 'cal.auto.key' : { off: 'cal.auto.off', none: 'cal.auto.none', far: 'cal.auto.far',
      hold: hold, wait: hold, ready: 'cal.auto.ready' }[a.state] || 'cal.auto.none';
    // the bench: the arm to the stop shown, or the seat the board stands in when it is the wrong one
    const bench = on && s.plan === 'bench' && s.kind !== 'check' && a.state === 'far';
    $('calAutoSay').textContent = !bench ? T(word) : a.seat && s.next && s.next.seat
      ? T('cal.auto.bench.seat', a.seat, s.next.seat) : T('cal.auto.bench.far');
    $('calAutoSay').classList.toggle('near', on && (a.state === 'hold' || a.state === 'wait' || a.state === 'ready'));
  },

  // The newest photos of the series, each a small picture with how its check went: checking, kept, retake.
  strip: function (s) {
    const shots = s.series && s.series.same ? s.series.shots || [] : [];
    const el = $('phStrip');
    el.classList.toggle('hidden', !shots.length || s.mode === 'video');
    const key = JSON.stringify(shots);
    if (el.dataset.key === key) return;
    el.dataset.key = key;
    el.replaceChildren.apply(el, shots.map(function (x) {
      const t = E('div', 'ph-tile ' + x.state);
      if (x.thumb) {
        const im = E('img');
        im.alt = '';
        im.src = x.thumb;
        t.append(im);
      }
      t.append(E('span', 'ph-tile-n', x.pose === null || x.pose === undefined ? '' : fmtNum(x.pose + 1, 0)),
               E('span', 'ph-tile-s'));
      t.title = x.state === 'kept' ? T('cal.strip.kept') : x.state === 'retake'
        ? T('cal.last.retake', Tcode('cal.why.', x.why, 'none')) : T('cal.strip.checking');
      return t;
    }));
  },

  // The shutter went: a flash on the guide and the counter (never over the board), and a short sound.
  // A photo taken by hand made its sound on the press (Take, the space bar, Enter); one taken by itself makes it here.
  flash: function (auto) {
    ['phGuide', 'cpCard'].forEach(function (id) {
      const el = $(id);
      el.classList.remove('shot');
      void el.offsetWidth;          // the animation again from its start
      el.classList.add('shot');
    });
    if (auto) playSound('shutter');       // SlantUI's voice of a picture taken (none while the sounds are off)
  },

  // ── the result ───────────────────────────────────────────────────────────
  // The photos left out by the result itself (by hand, before).
  excludedOf: function (r) {
    return r ? r.photos.filter(function (p) { return p.state === 'excluded'; }).map(function (p) { return p.name; }) : [];
  },

  result: function (s) {
    // a Calibrate that waits for the photos on their way counts as calibrating already
    const check = s.kind === 'check', r = s.result, n = s.count;
    const calBusy = s.busy.indexOf('calibration') >= 0 || !!s.calib_waiting;
    const last = CAL.last && CAL.last.mode === s.mode ? CAL.last : null;
    const verdict = CAL.verdict(s, check, r, n, calBusy, last);
    $('crTitle').textContent = verdict.title;
    $('crLine').textContent = verdict.line;
    $('crDot').className = 'live-dot' + (verdict.dot ? ' ' + verdict.dot : '');
    // what spoils it: the most serious in full, the others in a few words, each after a cross (a mark, for advice)
    const rows = verdict.issues || [];
    $('crIssues').replaceChildren.apply($('crIssues'), rows.map(function (x) {
      const row = E('div', 'say-issue' + (x.main ? ' si-first' : '') + (x.advice ? ' si-advice' : ''));
      row.append(E('i'), E('span', null, x.text));
      return row;
    }));
    $('crIssues').classList.toggle('hidden', !rows.length);
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
      $('crMapRel').textContent = m ? T('cal.res.map.rel') : '';
    }

    // the panel: the intrinsics in plain units
    $('calPlain').replaceChildren.apply($('calPlain'), r && !check ? CAL.plain(r) : []);
    const changedOut = show && JSON.stringify(leave.slice().sort()) !== JSON.stringify(CAL.excludedOf(r).sort());
    $('calLeft').textContent = show && leave.length ? T('cal.res.left', leave.length) : '';
    $('calLeft').classList.toggle('hidden', !(show && leave.length));

    // Advanced: what the checks of the calibration measured (the board's bend, the error of a flat board, how far the
    // optical centre moved, how near the corners the board came, the photos with automatic settings)
    const checks = r && !check ? CAL.checks(r) : [];
    $('calChecks').replaceChildren.apply($('calChecks'), checks);
    $('calChecksL').classList.toggle('hidden', !checks.length);
    // Advanced: the stored calibration of each mode (its day, short, and its error), and the log of the last run
    $('calStored').replaceChildren.apply($('calStored'), ['photo', 'video', 'stream'].map(function (md) {
      const x = s.modes[md].stored, c = s.last_check[md];
      const row = piRow(T('cal.stored.' + md), x ? T('cal.stored.row', fmtDay(x.date), fmtNum(x.rms_px, 2)) : T('cal.stored.no'));
      if (c) row.title = T(c.ok === '1' ? 'cal.check.ok' : 'cal.check.failed', fmtTime(c.time));
      return row;
    }));
    // the log stays in a file: a button opens it, once a calibration or a check has run (a newer backend only)
    $('calLogOpen').classList.toggle('hidden', !(CAL.last && hasSlot('calib_log_open')));

    // the one button: Calibrate, Calibrate again (photos left out or put back), Save, or Check
    const what = check ? 'check' : changedOut ? 'again' : r && !r.saved ? 'save' : 'calibrate';
    // always pressable from the calibration's own least (min_views) on, fewer than the plan advises included (the
    // card says so); below it the button says how many it takes
    const run = $('calRun'), few = what !== 'save' && n < s.min_views;
    if (!run._armed) $('calRunL').textContent = few ? T('cal.run.few', fmtNum(s.min_views, 0))
      : T({ check: 'cal.check.action', again: 'cal.run.again', save: 'cal.save.action', calibrate: 'cal.run.action' }[what]);
    run.dataset.what = what;
    run.disabled = calBusy || (what !== 'save' && (few || (check && !s.modes[s.mode].stored)));
    // before a calibration: the corners of the photo the board has not reached yet in the photos kept, above the button
    const miss = !check && what === 'calibrate' && !calBusy && n > 0 && s.series && s.series.same
      ? s.series.missing_corners || [] : [];
    const corners = !miss.length ? '' : miss.length >= 4 ? T('cal.run.corners_all')
      : miss.length === 1 ? T('cal.run.corner', T('cal.corner.' + miss[0])) : T('cal.run.corners', CAL.corners(miss));
    $('calCorners').textContent = corners;
    $('calCorners').classList.toggle('hidden', !corners);
    run.title = corners;
  },

  // corners of the photo by name, in a list as the language writes it
  corners: function (names) {
    return names.map(function (c) { return T('cal.corner.' + c); }).join(T('cal.corner.sep'));
  },

  // What spoils a calibration (r.issues, the most serious first): the first in full with its number, the others in a
  // few words; a corner never reached is advice (the calibration may still be good), the others are not.
  issueRows: function (r) {
    const miss = r.missing_corners || [], all = miss.length >= 4;
    const words = {
      board_not_flat: [T('cal.issue.board_not_flat', fmtNum(r.sag_mm, 1)), T('cal.res.v.flat')],
      phone_not_still: [T('cal.issue.phone_not_still'), T('cal.res.v.moved')],
      camera_settings: [T('cal.issue.camera_settings'), T('cal.res.v.settings')],
      corners_missing: all ? [T('cal.issue.corners_all'), T('cal.res.v.corners_all')]
        : [T('cal.issue.corners_missing', CAL.corners(miss)), T('cal.res.v.corners', CAL.corners(miss))],
    };
    return (r.issues || []).filter(function (c) { return words[c]; }).map(function (c, i) {
      return { code: c, main: i === 0, advice: (r.not_good || []).indexOf(c) < 0, text: words[c][i === 0 ? 0 : 1] };
    });
  },

  // The numbers of those checks, one row each (Advanced): only those the calibration has.
  checks: function (r) {
    const has = function (v) { return v !== null && v !== undefined && isFinite(v); };
    const rows = [];
    if (has(r.sag_mm)) rows.push(piRow(T('cal.adv.bend'), withUnit(r.sag_mm, 2, 'mm')));
    if (has(r.rms_flat_px)) rows.push(piRow(T('cal.adv.flat'), withUnit(r.rms_flat_px, 2, 'px')));
    if (has(r.centre_px)) {
      const row = piRow(T('cal.adv.centre'), withUnit(r.centre_px, 1, 'px'));
      if (has(r.centre_max_px)) row.title = T('cal.adv.centre.max', withUnit(r.centre_max_px, 1, 'px'));
      rows.push(row);
    }
    if (has(r.reach)) rows.push(piRow(T('cal.adv.reach'), fmtNum(100 * r.reach, 0) + ' %'));
    if (r.settings_n) {
      const row = piRow(T('cal.adv.settings'), T('cal.adv.settings.v', fmtNum(r.settings_n, 0)));
      row.title = (r.settings || []).map(function (c) { return Tcode('cal.set.', c, 'exposure_auto'); }).join('\n');
      rows.push(row);
    }
    return rows;
  },

  // The card of the result: working, the check's outcome, the calibration's verdict, or what is still needed.
  verdict: function (s, check, r, n, calBusy, last) {
    const ready = function () {
      return n < s.min_views ? T('cal.res.few', fmtNum(s.min_views, 0))
        : n < s.limits.min ? T('cal.res.fewer', fmtNum(n, 0), fmtNum(s.limits.min, 0)) : T('cal.res.ready', fmtNum(n, 0));
    };
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
      // why it is not good: what spoils it (issues, listed under the line: never good with one of them), else too few
      // photos, a focal length known too loosely (focal_ok false), else the error
      const why = r.few ? 'cal.res.v.few' : r.cropped ? 'cal.res.v.crop'
        : r.focal_ok === false && !(r.rms_px >= r.max_rms) ? 'cal.res.v.focal' : 'cal.res.v.high';
      const spoilt = (r.not_good || []).length > 0;
      return { title: Tcode('cal.res.v.', r.verdict, 'poor'),
               line: T('cal.res.v.line', fmtNum(r.rms_px, 2), fmtNum(r.n_views, 0))
                 + (r.verdict === 'good' || spoilt ? '' : ' · ' + T(why)),
               dot: r.verdict === 'good' ? 'ok' : 'warn', issues: CAL.issueRows(r) };
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
      // a film: filming from its start, choosing frames once stopped (calib_film events end them)
      if (a && a.film) { workOn('cal_film', a.film === 'starting' ? 'pill.filming' : 'pill.film.choosing'); CAL.refresh(); return; }
      if (a) { workOn('cal_photo', 'pill.shooting'); CAL.refresh(); return; }
      if (bad) CAL.refresh();
    });
  },

  run: function () {
    const s = CAL.state, what = $('calRun').dataset.what;
    if (what === 'save') {
      // a second press saves what would replace the stored one, or what is not good enough (said on the button)
      const st = s.modes[s.mode].stored, poor = !!(s.result && s.result.verdict === 'poor');
      const ask2 = poor ? (st ? T('cal.save.poor.replace', fmtDay(st.date)) : T('cal.save.poor'))
        : st ? T('cal.save.replace', fmtDay(st.date)) : '';
      if ((ask2 || $('calRun')._armed) && !armed($('calRun'), ask2)) return;
      playSound('confirm');           // on the press that saves, never after the answer
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
      if (e.error) toast(errorText(e), TOAST_LONG_MS, 'error');
      else if (e.ok) playSound('confirm');       // worked out, seconds after the press: said when it is so
      CAL.refresh();
    } else if (e.type === 'calib_photo' && e.action === 'countdown') {
      toast(T('cal.countdown'), TOAST_LONG_MS);     // the phone's camera counted down all the same: its timer is on
    } else if (e.type === 'calib_photo' && e.action === 'shot') {
      // the shutter went: said at once, and the guide moves to the next pose (the check comes later on its tile)
      workOff('cal_photo');
      if (page === 'calibration' && STEP.calibration === 'photos') CAL.flash(!!e.auto);
      CAL.refresh();
    } else if (e.type === 'calib_auto') {
      if (CAL.state && CAL.state.auto) CAL.autoSay({ on: CAL.state.auto.on, state: e.state, score: e.score });
    } else if (e.type === 'calib_photo' || (e.type === 'acquire_failed' && e.key === 'calibration')) {
      workOff('cal_photo');
      workOff('cal_film');
      CAL.refresh();
    } else if (e.type === 'calib_film') {
      // a film for a calibration: recording, its frames being chosen, then chosen (the calibration follows by itself)
      if (e.action === 'choosing') workOn('cal_film', 'pill.film.choosing');
      if (e.action === 'chosen') {
        workOff('cal_film');
        toast(T('cal.film.chosen', fmtNum(e.n, 0), fmtNum(e.seconds, 0))
          + (e.slower ? ' · ' + T('cal.film.slower', fmtNum(e.slower, 0)) : ''));
      }
      CAL.refresh();
    } else if (e.type === 'live' && livePurpose(e) === 'calibration') {
      // the stream of the streaming mode (the Live AR view never handles these)
      const asked = CAL.waiting;
      if (e.action === 'started' || e.action === 'failed' || (e.action === 'stopped' && e.reason === 'cancelled')) {
        CAL.waiting = false;
        workOff('cal_stream');
      }
      if (e.action === 'failed') toast(liveErrorText(e), TOAST_LONG_MS, 'error');
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
    $('calAuto').onchange = function () { CAL.set({ auto: $('calAuto').checked }); };
    $('calTake').onclick = CAL.take;
    $('calTake').setAttribute('data-sound', 'shutter');     // SlantUI's voice, on the press
    if (!CAL.keys) {
      CAL.keys = true;
      // the space bar or Enter takes the photo on the Photos step (the phone is in the operator's hands), whatever
      // has the focus there, except a field being typed in or the Phone panel; before a focused button acts on it
      const mine = function (ev) {
        if ((ev.key !== ' ' && ev.key !== 'Enter') || ev.ctrlKey || ev.altKey || ev.metaKey) return false;
        if (page !== 'calibration' || STEP.calibration !== 'photos' || LINK.open) return false;
        if (!CAL.state || CAL.state.mode === 'video') return false;     // a film is started and stopped by hand only
        if (typeof sheetIsOpen === 'function' && sheetIsOpen()) return false;
        const t = ev.target, tag = t && t.tagName;
        const typing = (tag === 'INPUT' && t.type !== 'checkbox' && t.type !== 'radio') || tag === 'TEXTAREA'
          || tag === 'SELECT' || (t && t.isContentEditable);
        return !typing;
      };
      window.addEventListener('keydown', function (ev) {
        if (!mine(ev)) return;
        ev.preventDefault();
        ev.stopPropagation();
        if (!ev.repeat && !$('calTake').disabled) { playSound('shutter'); CAL.take(); }
      }, true);
      window.addEventListener('keyup', function (ev) { if (mine(ev)) { ev.preventDefault(); ev.stopPropagation(); } }, true);
    }
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
