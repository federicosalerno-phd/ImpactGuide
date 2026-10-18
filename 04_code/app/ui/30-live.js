// ImpactGuide · Live AR, two steps on the rail: the board the phone follows
// (the patient board, or the calibration board on a monitor or printed, with
// the controls of 15-board.js) and the view: the phone's stream with the board
// pose and the face overlay, the status light, fps and latency, and the
// opacity of what is drawn over the picture. The work is live_ar.py's: the
// backend composes the frames and serves them on 127.0.0.1; the page shows
// them and asks for the rest.

const LIVE = {
  state: null,
  timer: null,
  src: '',          // the address of the frames while the view runs, '' otherwise
  sliding: false,   // the opacity slider is being moved
  sentAt: 0,        // when the opacity was last sent

  enter: function (step) {
    clearInterval(LIVE.timer);
    LIVE.timer = null;
    if (step === 'board') {
      showView('board');
      showPanel('live-board');
      if (LIVE.state) LIVE.renderBoard(LIVE.state);
    } else {
      showView('live');
      showPanel('live');
      LIVE.timer = setInterval(LIVE.refresh, 400);
    }
    LIVE.refresh();
    be('phone_refresh');
  },

  // Leaving the view stops it: the camera is free for whoever needs it next.
  leave: function () {
    clearInterval(LIVE.timer);
    LIVE.timer = null;
    LIVE.show('');
    const s = LIVE.state;
    if (s && (s.running || s.starting)) {
      ask('live_stop', function (a) { if (a) { LIVE.state = a; toast(T('live.said.stopped')); } });
    }
  },

  refresh: function () {
    ask('live_state', function (s) { if (s && page === 'live') LIVE.render(s); });
  },

  // The picture follows the view: a new connection when it starts, none when it stops.
  show: function (url) {
    if (url === LIVE.src) return;
    LIVE.src = url;
    const img = $('lvImg');
    if (url) img.src = url + '?t=' + Date.now();
    else img.removeAttribute('src');
  },

  name: function (path) { return String(path || '').split(/[\\/]/).pop(); },

  // the stream's own error text (often a Java stack from the phone) stays in the log;
  // the operator reads what to do: the link's cause when the link is why
  plainError: function (msg, cause) {
    if (cause && hasT('link.cause.' + cause)) return T('link.cause.' + cause);
    if (/camera|in use|disconnect/i.test(String(msg || ''))) return T('live.err.busy');
    return T(PHONE.link === 'wifi' ? 'live.err.wifi' : 'live.err.generic');
  },

  // the first step: which board; on the stage the calibration board (on this monitor, or physical) or the
  // patient board's card
  renderBoard: function (s) {
    const sheet = s.prefs.board === 'sheet';
    document.querySelectorAll('#lvBoards .mitem').forEach(function (it) {
      it.classList.toggle('sel', it.dataset.board === s.prefs.board);
    });
    $('lvBd').classList.toggle('hidden', !sheet);
    BOARD.render(s.board_ui);
    // the view: the board on this monitor beside the picture (hidden in the picture, so the phone sees it once)
    const beside = sheet && s.board_ui.kind === 'screen' && STEP.live === 'view';
    $('bdHostLive').classList.toggle('hidden', !beside);
    if (STEP.live === 'view') BOARDVIEW.place(beside ? 'bdHostLive' : null);
    if (STEP.live !== 'board') return;
    $('bdPatient').classList.toggle('hidden', sheet);
    if (sheet) { BOARD.stage(s.board_ui); return; }
    ['bdAsk', 'bdSplit', 'bdPhys'].forEach(function (id) { $(id).classList.add('hidden'); });
    BOARDVIEW.place(null);
    const b = s.board;
    $('bdTitle').textContent = T('live.card.patient');
    $('bdLine').textContent = T('live.card.patient.line', b.squares[0], b.squares[1],
                                cifre(b.square_mm, b.square_mm % 1 ? 2 : 0));
    $('bdHint').textContent = T('live.card.patient.hint');
  },

  render: function (s) {
    LIVE.state = s;
    LIVE.renderBoard(s);
    const view = STEP.live === 'view';
    const on = s.running && !s.error;
    const starting = !s.running && (s.starting || s.busy.indexOf('live') >= 0);
    LIVE.show(on && view ? s.url : '');
    // the frame takes the picture's own shape: 4:3 with the phone held across, 3:4 held upright (live_ar turns the
    // picture with the phone, `display`), as large as the stage allows and never stretched
    const d = s.display || s.image || [4, 3];
    if (d[0] > 0 && d[1] > 0) $('lvImg').parentElement.style.setProperty('--ar', String(d[0] / d[1]));
    ['lvImg', 'lvTags', 'lvLights'].forEach(function (id) { $(id).classList.toggle('hidden', !on); });
    $('lvIdle').classList.toggle('hidden', on);
    $('lvIdleWait').classList.toggle('hidden', !starting);
    $('lvIdleTitle').textContent = T(starting ? 'live.idle.title.starting' : 'live.idle.title');
    $('lvIdleText').textContent = s.error && s.running ? LIVE.plainError(s.error, s.cause)
      : T(s.simulate ? 'live.idle.text.sim' : 'live.idle.text');
    const size = s.intrinsics.size;
    $('where').textContent = T(s.synthetic ? 'live.where.sim' : 'live.where', size[0], size[1]);

    // over the picture: a link that dropped, what is not real yet, the light and why, the pace
    const reg = s.registration, tags = [];
    if (s.link_waiting) tags.push([T('live.waiting'), true]);
    if (!s.models.face) tags.push([T('live.tag.nomodel')]);
    else if (reg.source === 'demo') tags.push([T('live.tag.demo.' + reg.demo), true]);
    else tags.push([T('live.tag.file', LIVE.name(reg.path))]);
    if (s.intrinsics.source === 'nominal') tags.push([T('live.tag.nominal')]);
    if (s.synthetic) tags.push([T(s.replay ? 'live.tag.replay' : 'live.tag.synthetic')]);
    $('lvTags').replaceChildren.apply($('lvTags'), tags.map(function (t) { return E('div', t[1] ? 'live-demo' : null, t[0]); }));
    const word = s.tracked ? T('live.light.tracked') : T(s.corners ? 'live.light.weak' : 'live.light.none');
    $('lvDot').className = 'live-dot' + (on && s.tracked ? ' ok' : '');
    $('lvLight').textContent = word;
    let why = '';
    if (on && !s.tracked && s.corners) {
      why = s.corners < s.min_corners ? T('live.why.corners', s.min_corners) : T('live.why.rms', cifre(s.rms_limit_px, 0));
    }
    $('lvWhy').textContent = why;
    $('lvWhy').classList.toggle('hidden', !why);
    const lat = s.latency || {};

    // the pace and the delay are not written on the picture (no technical words in the window): they are the first
    // rows of the panel's Advanced part, with the rest of the measures

    // the panel: how strongly the virtual elements are drawn
    const op = $('lvOpacity');
    if (typeof s.opacity === 'number' && !LIVE.sliding && Date.now() - LIVE.sentAt > 1500) op.value = s.opacity;
    $('lvOpacityV').textContent = cifre(100 * Number(op.value), 0) + ' %';

    // Advanced: the board, the pace and the link while the view runs; the registration; the camera
    const b = s.board, rows = [];
    if (on) {
      const tm = s.timings || {}, ln = s.phone_link || {};
      rows.push(piRow(T('live.f.corners'), T('live.f.corners.v', s.corners || 0, b.corners, s.markers || 0)),
                piRow(T('live.f.rms', cifre(s.rms_limit_px, 0)), s.rms_px === null || s.rms_px === undefined ? '-'
                  : T('live.f.px', cifre(s.rms_px, 2))));
      if (s.distance_mm) rows.push(piRow(T('live.f.distance'), cifre(s.distance_mm, 0) + ' mm'), piRow(T('live.f.tilt'), cifre(s.tilt_deg, 0) + '°'));
      // the video keeps the stream's rate; the board pose is found as often as the computer allows
      rows.push(piRow(T('live.f.stream'), T('live.f.fps', cifre(s.fps_stream, 1))),
                piRow(T('live.f.video'), T('live.f.fps', cifre(s.fps_ar || 0, 1))),
                piRow(T('live.f.poses'), T('live.f.fps', cifre(s.fps_pose || 0, 1))),
                piRow(T('live.f.latency'), T('live.f.ms', cifre(lat.total_ms || 0, 0))),
                piRow(T('live.f.pose'), T('live.f.ms', cifre((tm.detect_ms || 0) + (tm.pose_ms || 0), 0))),
                piRow(T('live.f.drawing'), T('live.f.ms', cifre((tm.render_ms || 0) + (tm.draw_ms || 0), 0))));
      const lw = T('link.btn.' + (ln.mode === 'wifi' ? 'wifi' : 'usb'));
      rows.push(piRow(T('live.f.link'), ln.state === 'ready'
        ? (ln.rtt_ms === null || ln.rtt_ms === undefined ? lw : T('live.f.link.v', lw, cifre(ln.rtt_ms, 0)))
        : lw + ' · ' + T('link.state.' + (hasT('link.state.' + ln.state) ? ln.state : 'unknown'))));
    }
    $('lvStatus').replaceChildren.apply($('lvStatus'), rows);
    $('lvStatusLbl').classList.toggle('hidden', !on);
    const regRow = piRow(T('live.reg.now'), reg.source === 'file' ? LIVE.name(reg.path) : T('live.reg.demo.' + (reg.demo || 'lying')));
    regRow.title = reg.source === 'file' ? reg.path : T('live.reg.demo.hint');
    $('lvReg').replaceChildren(regRow);
    if (reg.error) $('lvReg').append(E('div', 'callout callout-warn', T('live.reg.error', LIVE.name(s.prefs.registration))));
    $('lvRegDemo').disabled = reg.source !== 'file' && !s.prefs.registration;
    const c = s.intrinsics;
    const camRow = piRow(T('live.cam.now'), c.source === 'calibrated' ? T('live.cam.calibrated', c.date || '-') : T('live.cam.nominal'));
    camRow.title = c.source === 'calibrated'
      ? T('live.cam.calibrated.hint', cifre(c.f, 1), c.rms_px === null ? '-' : cifre(c.rms_px, 3)) + ' · ' + c.path
      : T('live.cam.nominal.hint', cifre(c.f, 0), c.size[0], c.size[1]) + ' · ' + s.intrinsics_file;
    $('lvCam').replaceChildren(camRow);

    // the toolbar and the footer
    document.querySelectorAll('.layer-btn').forEach(function (btn) {
      const k = btn.dataset.layer;
      btn.classList.toggle('on', !!s.layers[k]);
      btn.disabled = k !== 'axes' && !s.models[k];
    });
    $('lvStart').classList.toggle('hidden', s.running);
    $('lvStop').classList.toggle('hidden', !s.running);
    $('lvStart').disabled = starting;
  },

  // The opacity of the virtual elements, 0 to 1, as the live setting `opacity`: sent while the slider
  // moves (at most every 150 ms) and when it is let go; the backend keeps it for the next session. Once
  // let go, the slider follows the value the live view reports.
  opacity: function (final) {
    const v = Math.round(Number($('lvOpacity').value) * 100) / 100;
    $('lvOpacityV').textContent = cifre(100 * v, 0) + ' %';
    clearTimeout(LIVE._opTimer);
    const send = function () {
      LIVE.sentAt = Date.now();
      LIVE._opLast = LIVE.sentAt;
      ask('live_settings', JSON.stringify({ opacity: v }), function (s) { if (s && page === 'live') LIVE.render(s); });
    };
    if (final || Date.now() - (LIVE._opLast || 0) > 150) send();
    else LIVE._opTimer = setTimeout(send, 150);
  },

  start: function () {
    ask('live_start', function (s) {
      if (!s) return;
      work(true, T('pill.live.starting'));
      LIVE.render(s);
    });
  },

  stop: function () {
    ask('live_stop', function (s) { if (s) { LIVE.render(s); toast(T('live.said.stopped')); } });
  },

  settings: function (change) {
    ask('live_settings', JSON.stringify(change), function (s) { if (s) LIVE.render(s); });
  },

  onEvent: function (e) {
    // the stream of a calibration in the streaming mode is Calibrate's own (CAL.onEvent): never stopped from here
    if (e.type !== 'live' || e.purpose === 'calibration') return;
    if (e.action === 'started') {
      work(false);
      if (page === 'live' && STEP.live === 'view') toast(T('live.said.started'));
      else ask('live_stop');      // the view was left while the stream was opening
    } else if (e.action === 'failed') {
      work(false);
      toast(LIVE.plainError(e.message || e.error, e.cause), 6000);
    } else if (e.action === 'stopped' && e.reason === 'photo') {
      toast(T('live.said.stopped.photo'), 3500);
    }
    if (page === 'live') LIVE.refresh();
  },

  wire: function () {
    document.querySelectorAll('#lvBoards .mitem').forEach(function (it) {
      it.onclick = function () { LIVE.settings({ board: it.dataset.board }); };
      it.onkeydown = function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); it.onclick(); } };
    });
    document.querySelectorAll('.layer-btn').forEach(function (btn) {
      btn.onclick = function () {
        const k = btn.dataset.layer, change = {};
        change[k] = !(LIVE.state && LIVE.state.layers[k]);
        LIVE.settings(change);
      };
    });
    $('lvStart').onclick = LIVE.start;
    $('lvStop').onclick = LIVE.stop;
    $('lvRegLoad').onclick = function () { ask('live_registration_load', function (s) { if (s) LIVE.render(s); }); };
    $('lvRegDemo').onclick = function () { ask('live_registration_demo', function (s) { if (s) LIVE.render(s); }); };
    $('lvOpacity').oninput = function () { LIVE.sliding = true; LIVE.opacity(false); };
    $('lvOpacity').onchange = function () { LIVE.sliding = false; LIVE.opacity(true); };
  },
};
