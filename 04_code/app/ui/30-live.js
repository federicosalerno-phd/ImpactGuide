// ImpactGuide · Live AR, two steps on the rail: the board the phone follows
// (the patient board, or the calibration board on a monitor or printed, with
// the controls of 15-board.js) and the view: the phone's stream with the board
// pose and the face overlay, the status light, fps and latency. Its panel says
// what is drawn over the picture (a row per element, grouped, with an eye) and
// how solid, and holds the view on the phone's own screen: how it stands and
// the command that opens it. The work is live_ar.py's: the backend composes the
// frames and serves them on 127.0.0.1; the page shows them and asks for the rest.
'use strict';

const LIVE_POLL_MS = 400;        // the state of the view, read while the view is open
const OPACITY_HOLD_MS = 1500;    // the state does not move the opacity slider for this long after the hand let it go
const LIVE_FAILS_SAID = 2;       // reads of the state failed in a row before the view says so
const PHONE_OPENING_MS = 8000;   // after Open on the phone, the phone's page has this long to show up before it is off

// What can be drawn over the picture, in groups: a row each in the panel, with its eye. A new hologram is one more
// name here, with its word live.layer.<name> and its colour in app.css (.lyr[data-layer=...]); the patient's
// elements are models, shown only once the live view has the model. The board's are drawn from its pose.
const LIVE_LAYERS = [
  { group: 'patient', layers: ['face', 'mandible'] },
  { group: 'board', layers: ['axes'] },
];

/** Live AR: the board the phone follows and the view of its stream. */
const LIVE = {
  state: null,
  timer: null,
  src: '',          // the address of the frames while the view runs, '' otherwise
  sliding: false,   // the opacity slider is being moved
  sentAt: 0,        // when the opacity was last sent
  opNext: null,     // the newest opacity of the slider not sent yet (sent at the next frame of the screen)
  opFrame: 0,       // the frame that will send it (requestAnimationFrame), 0 when none is asked for
  phoneAt: 0,       // when Open on the phone was pressed
  waiting: false,   // Start was pressed and the stream has not started yet

  enter: function (step) {
    clearInterval(LIVE.timer);
    LIVE.timer = null;
    if (step === 'board') {
      showView('board');
      showPanel('live-board');
      if (LIVE.state) LIVE.renderBoard(LIVE.state);
      LIVE.refresh();
    } else {
      showView('live');
      showPanel('live');
      LIVE.timer = poll('live_state', LIVE_POLL_MS, LIVE.polled);
    }
    be('phone_refresh');
  },

  // Leaving the view stops it: the camera is free for whoever needs it next.
  leave: function () {
    clearInterval(LIVE.timer);
    LIVE.timer = null;
    LIVE.show('');
    const s = LIVE.state;
    if (LIVE.waiting || (s && (s.running || s.starting))) {
      ask('live_stop', function (a) { if (a) { LIVE.state = a; toast(T('live.said.stopped')); } });
    }
  },

  refresh: function () {
    // while the view polls, a failed read is said in the view, not in a toast
    (LIVE.timer ? askQuiet : ask)('live_state', function (s) { if (s && page === 'live') LIVE.render(s); });
  },

  polled: function (s, bad, fails) {
    if (page !== 'live') return;
    if (s) LIVE.render(s);
    else if (fails >= LIVE_FAILS_SAID) LIVE.unreachable(bad);
  },

  // The state cannot be read: the picture off, and why, in the card of the view at rest.
  unreachable: function (bad) {
    if (STEP.live !== 'view') return;
    LIVE.show('');
    ['lvImg', 'lvTags', 'lvLights'].forEach(function (id) { $(id).classList.add('hidden'); });
    $('lvIdle').classList.remove('hidden');
    $('lvIdleWait').classList.add('hidden');
    $('lvIdleTitle').textContent = T('live.idle.title');
    $('lvIdleText').textContent = errorText(bad);
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
                                fmtNum(b.square_mm, b.square_mm % 1 ? 2 : 0));
    $('bdHint').textContent = T('live.card.patient.hint');
  },

  render: function (s) {
    LIVE.state = s;
    LIVE.renderBoard(s);
    const view = STEP.live === 'view';
    // one bad frame does not take the picture away while frames keep coming
    // (the stream's code is live_error: an `error` in an answer would say the call itself failed)
    const on = s.running && (!s.live_error || (s.fps_ar || 0) > 0);
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
    // why the stream is not there: it gave up, or the last start failed (Start opens a new one)
    $('lvIdleText').textContent = s.live_error && (s.running || s.stream_state === 'failed')
      ? liveErrorText({ error: s.live_error, cause: s.cause })
      : T(s.simulate ? 'live.idle.text.sim' : 'live.idle.text');
    const size = s.intrinsics.size;
    $('where').textContent = T(s.synthetic ? 'live.where.sim' : 'live.where', size[0], size[1]);
    LIVE.overlay(s, on);

    // the panel: what is drawn and how strongly, the phone's screen, Advanced
    LIVE.renderLayers(s);
    const op = $('lvOpacity');
    if (typeof s.opacity === 'number' && !LIVE.sliding && Date.now() - LIVE.sentAt > OPACITY_HOLD_MS) op.value = s.opacity;
    $('lvOpacityV').textContent = fmtNum(100 * Number(op.value), 0) + ' %';
    LIVE.phone(s);
    LIVE.advanced(s, on);

    // the footer
    $('lvStart').classList.toggle('hidden', s.running);
    $('lvStop').classList.toggle('hidden', !s.running);
    $('lvStart').disabled = starting;
  },

  // The rows of what is drawn: a heading per group (SlantUI's metric rows), a row per element with its name and an
  // eye; a press (Enter, Space) shows or hides it. Made again when the language changes.
  buildLayers: function () {
    const out = [];
    LIVE_LAYERS.forEach(function (g) {
      out.push(E('div', 'mg', T('live.group.' + g.group)));
      g.layers.forEach(function (k) {
        const row = E('div', 'mrow lyr');
        row.dataset.layer = k;
        row.dataset.model = g.group === 'patient' ? '1' : '';
        row.setAttribute('role', 'switch');
        row.append(E('span', 'ml', T('live.layer.' + k)), E('span', 'lyr-eye'));
        onActivate(row, function () { LIVE.toggleLayer(row); });
        out.push(row);
      });
    });
    $('lvLayers').replaceChildren.apply($('lvLayers'), out);
    if (LIVE.state) LIVE.renderLayers(LIVE.state);
  },

  // a model the view does not have (or cannot draw) cannot be shown; the board's elements always can
  hasLayer: function (s, row) { return !row.dataset.model || !!(s.models && s.models[row.dataset.layer]); },

  toggleLayer: function (row) {
    const s = LIVE.state, k = row.dataset.layer, change = {};
    if (!s || !LIVE.hasLayer(s, row)) return;
    change[k] = !(s.layers && s.layers[k]);
    LIVE.settings(change);
  },

  renderLayers: function (s) {
    document.querySelectorAll('#lvLayers .lyr').forEach(function (row) {
      const have = LIVE.hasLayer(s, row), on = have && !!(s.layers && s.layers[row.dataset.layer]);
      row.classList.toggle('active', on);
      itemOff(row, !have);
      row.setAttribute('aria-checked', on ? 'true' : 'false');
      row.title = have ? '' : T('live.layer.missing');
      const eye = row.querySelector('.lyr-eye'), sign = on ? 'eye' : 'eye-off';
      if (eye.dataset.sign !== sign) { setIcon(eye, sign); eye.dataset.sign = sign; }
    });
  },

  // The view on the phone's own screen, in a few words: off; opening; open on the phone (its page there, waiting for
  // frames: the view not started); linked to the view (its page painting the frames). The command opens it (again);
  // the Phone panel has the rest.
  phone: function (s) {
    const v = s.phone_view || {}, busy = (s.busy || []).indexOf('view_open') >= 0;
    const ready = (s.phone_link || {}).state === 'ready';
    const there = (v.viewers || 0) > 0;
    let st = 'off', dot = '';
    if (there && (v.frames || 0) > 0) { st = 'linked'; dot = ' ok'; }
    else if (there) { st = 'open'; dot = ' warn'; }
    else if (busy || (ready && Date.now() - LIVE.phoneAt < PHONE_OPENING_MS)) { st = 'opening'; dot = ' busy'; }
    $('lvPhDot').className = 'live-dot' + dot;
    $('lvPhWord').textContent = !ready && !there ? T('link.btn.none') : T('view.st.' + st);
    $('lvPhOpen').disabled = !ready || busy;
  },

  openPhone: function () {
    LIVE.phoneAt = Date.now();
    ask('view_open', function (a) {
      if (!a) LIVE.phoneAt = 0;
      LIVE.refresh();
    });
    if (LIVE.state) LIVE.phone(LIVE.state);
  },

  // Over the picture: a link that dropped, what is not real yet, the light and why. The pace and the delay are
  // not written on the picture (no technical words in the window): they are rows of the panel's Advanced part.
  overlay: function (s, on) {
    const reg = s.registration, tags = [];
    if (s.link_waiting) tags.push([T('live.waiting'), true]);
    if (!s.models.face) tags.push([T('live.tag.nomodel')]);
    else if (reg.source === 'demo') tags.push([Tcode('live.tag.demo.', reg.demo, 'lying'), true]);
    else tags.push([T('live.tag.file', LIVE.name(reg.path))]);
    if (s.intrinsics.source === 'nominal') tags.push([T('live.tag.nominal')]);
    if (s.synthetic) tags.push([T(s.replay ? 'live.tag.replay' : 'live.tag.synthetic')]);
    $('lvTags').replaceChildren.apply($('lvTags'), tags.map(function (t) { return E('div', t[1] ? 'live-demo' : null, t[0]); }));
    const word = s.tracked ? T('live.light.tracked') : T(s.corners ? 'live.light.weak' : 'live.light.none');
    $('lvDot').className = 'live-dot' + (on && s.tracked ? ' ok' : '');
    $('lvLight').textContent = word;
    let why = '';
    if (on && !s.tracked && s.corners) {
      why = s.corners < s.min_corners ? T('live.why.corners', s.min_corners) : T('live.why.rms', fmtNum(s.rms_limit_px, 0));
    }
    $('lvWhy').textContent = why;
    $('lvWhy').classList.toggle('hidden', !why);
  },

  // Advanced: the board, the pace and the link while the view runs; the registration; the camera.
  advanced: function (s, on) {
    const b = s.board, rows = [], lat = s.latency || {}, reg = s.registration;
    const ms = function (v) { return T('live.f.ms', fmtNum(v || 0, 0)); };
    if (on) {
      const tm = s.timings || {}, ln = s.phone_link || {};
      rows.push(piRow(T('live.f.corners'), T('live.f.corners.v', s.corners || 0, b.corners, s.markers || 0)),
                piRow(T('live.f.rms', fmtNum(s.rms_limit_px, 0)), s.rms_px === null || s.rms_px === undefined ? '-'
                  : T('live.f.px', fmtNum(s.rms_px, 2))));
      if (s.distance_mm) rows.push(piRow(T('live.f.distance'), fmtNum(s.distance_mm, 0) + ' mm'), piRow(T('live.f.tilt'), fmtNum(s.tilt_deg, 0) + '°'));
      // how much the drawn face shakes, against the pose as found (px of the phone's picture, the last moments)
      if (typeof s.tremble_px === 'number') {
        const tr = piRow(T('live.f.tremble'), T('live.f.tremble.v', fmtNum(s.tremble_px, 2),
                                                 typeof s.tremble_raw_px === 'number' ? fmtNum(s.tremble_raw_px, 2) : '-'));
        tr.title = T('live.f.tremble.hint');
        rows.push(tr);
      }
      // the video keeps the stream's rate; the board pose is found as often as the computer allows
      rows.push(piRow(T('live.f.stream'), T('live.f.fps', fmtNum(s.fps_stream, 1))),
                piRow(T('live.f.video'), T('live.f.fps', fmtNum(s.fps_ar || 0, 1))),
                piRow(T('live.f.poses'), T('live.f.fps', fmtNum(s.fps_pose || 0, 1))),
                piRow(T('live.f.latency'), ms(lat.total_ms)),
                piRow(T('live.f.pose'), ms((tm.detect_ms || 0) + (tm.pose_ms || 0))),
                piRow(T('live.f.drawing'), ms((tm.render_ms || 0) + (tm.draw_ms || 0))));
      const lw = T('link.btn.' + (ln.mode === 'wifi' ? 'wifi' : 'usb'));
      rows.push(piRow(T('live.f.link'), ln.state === 'ready'
        ? (ln.rtt_ms === null || ln.rtt_ms === undefined ? lw : T('live.f.link.v', lw, fmtNum(ln.rtt_ms, 0)))
        : lw + ' · ' + Tcode('link.state.', ln.state, 'unknown')));
    }
    $('lvStatus').replaceChildren.apply($('lvStatus'), rows);
    $('lvStatusLbl').classList.toggle('hidden', !on);
    const regRow = piRow(T('live.reg.now'), reg.source === 'file' ? LIVE.name(reg.path) : Tcode('live.reg.demo.', reg.demo, 'lying'));
    regRow.title = reg.source === 'file' ? reg.path : T('live.reg.demo.hint');
    $('lvReg').replaceChildren(regRow);
    if (reg.error) $('lvReg').append(E('div', 'callout callout-warn', T('live.reg.error', LIVE.name(s.prefs.registration))));
    $('lvRegDemo').disabled = reg.source !== 'file' && !s.prefs.registration;
    const c = s.intrinsics;
    const camRow = piRow(T('live.cam.now'), c.source === 'calibrated' ? T('live.cam.calibrated', c.date ? fmtDay(c.date) : '-')
      : T('live.cam.nominal'));
    camRow.title = c.source === 'calibrated'
      ? T('live.cam.calibrated.hint', fmtNum(c.f, 1), c.rms_px === null ? '-' : fmtNum(c.rms_px, 3)) + ' · ' + c.path
      : T('live.cam.nominal.hint', fmtNum(c.f, 0), c.size[0], c.size[1]) + ' · ' + s.intrinsics_file;
    $('lvCam').replaceChildren(camRow);
    if (c.error) $('lvCam').append(E('div', 'callout callout-warn', liveErrorText({ error: c.error })));
  },

  // The opacity of the virtual elements, 0 to 1 with no steps, as the live setting `opacity`. While the slider moves
  // (the hand, the wheel, the arrow keys: SlantUI's initFineRanges) the newest value goes to the live view once per
  // frame of the screen (requestAnimationFrame joins the moves of a frame), without waiting for an answer: the call
  // only sets the value (live_opacity) and the next picture composed takes it. Let go, the value is kept for the next
  // session (live_settings) and the slider follows the value the live view reports.
  opacity: function (final) {
    const v = Math.min(1, Math.max(0, Number($('lvOpacity').value) || 0));
    $('lvOpacityV').textContent = fmtNum(100 * v, 0) + ' %';
    LIVE.sentAt = Date.now();
    if (final) {
      LIVE.opNext = null;
      ask('live_settings', JSON.stringify({ opacity: v }), function (s) { if (s && page === 'live') LIVE.render(s); });
      return;
    }
    LIVE.opNext = v;
    if (!LIVE.opFrame) LIVE.opFrame = requestAnimationFrame(LIVE.sendOpacity);
  },

  sendOpacity: function () {
    LIVE.opFrame = 0;
    const v = LIVE.opNext;
    LIVE.opNext = null;
    if (v !== null) be('live_opacity', v);
  },

  start: function () {
    setLiveOwner('live');
    ask('live_start', function (s) {
      if (!s) return;
      // the stream opens in the background: a 'live' event says how it went (none when it was already on)
      if (!s.running) { LIVE.waiting = true; workOn('live', 'pill.live.starting'); }
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
    if (e.type !== 'live' || livePurpose(e) !== 'live') return;
    const inView = page === 'live' && STEP.live === 'view';
    if (e.action === 'started') {
      workOff('live');
      if (LIVE.waiting && !inView) askQuiet('live_stop');      // the view was left while the stream was opening
      else if (inView) toast(T('live.said.started'));
      LIVE.waiting = false;
    } else if (e.action === 'failed') {
      workOff('live');
      LIVE.waiting = false;
      toast(liveErrorText(e), TOAST_LONG_MS, 'error');
    } else if (e.action === 'stopped' && e.reason === 'cancelled') {
      workOff('live');            // a start stopped before its stream came: nothing to say, the view is at rest
      LIVE.waiting = false;
    } else if (e.action === 'stopped' && e.reason === 'photo') {
      toast(T('live.said.stopped.photo'), TOAST_SAID_MS);
    }
    if (page === 'live') LIVE.refresh();
  },

  wire: function () {
    document.querySelectorAll('#lvBoards .mitem').forEach(function (it) {
      onActivate(it, function () { LIVE.settings({ board: it.dataset.board }); });
    });
    LIVE.buildLayers();
    $('lvPhOpen').onclick = LIVE.openPhone;
    $('lvStart').onclick = LIVE.start;
    $('lvStop').onclick = LIVE.stop;
    $('lvStart').setAttribute('data-sound', 'on');       // SlantUI's voices, on the press
    $('lvStop').setAttribute('data-sound', 'off');
    $('lvRegLoad').onclick = function () { ask('live_registration_load', function (s) { if (s) LIVE.render(s); }); };
    $('lvRegDemo').onclick = function () { ask('live_registration_demo', function (s) { if (s) LIVE.render(s); }); };
    $('lvOpacity').oninput = function () { LIVE.sliding = true; LIVE.opacity(false); };
    $('lvOpacity').onchange = function () { LIVE.sliding = false; LIVE.opacity(true); };
    initFineRanges($('lvOpacity').parentElement);      // the wheel and the arrow keys by 0.1 %, by 1 % with Shift
  },
};
