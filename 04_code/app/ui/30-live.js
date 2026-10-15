// ImpactGuide · the Live AR page: the phone's stream with the board pose and
// the face overlay, the status light, fps and latency. The work is live_ar.py's:
// the backend composes the frames and serves them on 127.0.0.1; the page shows
// them and asks for the rest.

const LIVE = {
  state: null,
  timer: null,
  src: '',          // the address of the frames while the view runs, '' otherwise

  enter: function () {
    LIVE.refresh();
    clearInterval(LIVE.timer);
    LIVE.timer = setInterval(LIVE.refresh, 400);
    be('phone_refresh');
  },

  // Leaving the page stops the view: the camera is free for whoever needs it next.
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
  // the operator reads what to do
  plainError: function (msg) {
    return T(/camera|in use|disconnect/i.test(String(msg || '')) ? 'live.err.busy' : 'live.err.generic');
  },

  render: function (s) {
    LIVE.state = s;
    const on = s.running && !s.error;
    const starting = !s.running && (s.starting || s.busy.indexOf('live') >= 0);
    LIVE.show(on ? s.url : '');
    ['lvImg', 'lvTags', 'lvLights', 'lvRate'].forEach(function (id) { $(id).classList.toggle('hidden', !on); });
    $('lvIdle').classList.toggle('hidden', on);
    $('lvIdleWait').classList.toggle('hidden', !starting);
    $('lvIdleTitle').textContent = T(starting ? 'live.idle.title.starting' : 'live.idle.title');
    $('lvIdleText').textContent = s.error && s.running ? LIVE.plainError(s.error)
      : T(s.simulate ? 'live.idle.text.sim' : 'live.idle.text');
    const size = s.intrinsics.size;
    $('where').textContent = T(s.synthetic ? 'live.where.sim' : 'live.where', size[0], size[1]);

    // over the picture: what is not real yet, the light, the pace
    const reg = s.registration, tags = [];
    if (!s.models.face) tags.push(T('live.tag.nomodel'));
    else if (reg.source === 'demo') tags.push(T('live.tag.demo.' + reg.demo));
    else tags.push(T('live.tag.file', LIVE.name(reg.path)));
    if (s.intrinsics.source === 'nominal') tags.push(T('live.tag.nominal'));
    if (s.synthetic) tags.push(T(s.replay ? 'live.tag.replay' : 'live.tag.synthetic'));
    $('lvTags').replaceChildren.apply($('lvTags'), tags.map(function (t, i) {
      return E('div', i === 0 && reg.source === 'demo' ? 'live-demo' : null, t);
    }));
    const word = s.tracked ? T('live.light.tracked') : T(s.corners ? 'live.light.weak' : 'live.light.none');
    $('lvDot').className = 'live-dot' + (on && s.tracked ? ' ok' : '');
    $('lvLight').textContent = word;
    const lat = s.latency || {};
    $('lvRate').textContent = on && s.fps_ar ? T('live.rate', cifre(s.fps_ar, 0), cifre(lat.total_ms || 0, 0)) : '';

    // the panel: the board
    $('lvStDot').className = 'live-dot' + (on && s.tracked ? ' ok' : '');
    $('lvStWord').textContent = on ? word : T('live.idle.title');
    const b = s.board, rows = [];
    if (on) {
      rows.push(piRow(T('live.f.corners'), T('live.f.corners.v', s.corners || 0, b.corners, s.markers || 0)));
      rows.push(piRow(T('live.f.rms', cifre(s.rms_limit_px, 0)), s.rms_px === null || s.rms_px === undefined ? '-'
        : T('live.f.px', cifre(s.rms_px, 2))));
      if (s.distance_mm) rows.push(piRow(T('live.f.distance'), T('live.f.distance.v', cifre(s.distance_mm, 0), cifre(s.tilt_deg, 0))));
    }
    $('lvStatus').replaceChildren.apply($('lvStatus'), rows);
    let why = '';
    if (on && !s.tracked && s.corners) {
      why = s.corners < s.min_corners ? T('live.why.corners', s.min_corners) : T('live.why.rms', cifre(s.rms_limit_px, 0));
    }
    $('lvWhy').textContent = why;
    $('lvWhy').classList.toggle('hidden', !why);

    // the panel: the pace
    const speed = [];
    if (on) {
      const tm = s.timings || {};
      speed.push(piRow(T('live.f.stream'), T('live.f.fps', cifre(s.fps_stream, 1))),
                 piRow(T('live.f.overlay'), T('live.f.fps', cifre(s.fps_ar || 0, 1))),
                 piRow(T('live.f.latency'), T('live.f.ms', cifre(lat.total_ms || 0, 0))),
                 piRow(T('live.f.pose'), T('live.f.ms', cifre((tm.detect_ms || 0) + (tm.pose_ms || 0), 0))),
                 piRow(T('live.f.drawing'), T('live.f.ms', cifre((tm.render_ms || 0) + (tm.draw_ms || 0), 0))));
      $('lvLatency').textContent = s.synthetic ? T('live.f.latency.note.sim', cifre(lat.processing_ms || 0, 0))
        : T('live.f.latency.note', cifre(lat.stream_ms || 0, 0), cifre(lat.processing_ms || 0, 0));
    } else {
      $('lvLatency').textContent = '';
    }
    $('lvSpeed').replaceChildren.apply($('lvSpeed'), speed);
    $('lvLatency').classList.toggle('hidden', !on);
    $('lvSpeedLbl').classList.toggle('hidden', !on);

    // the panel: board, registration, camera
    $('lvBoard').value = s.prefs.board;
    $('lvBoardNote').textContent = T('live.board.note', b.squares[0], b.squares[1], cifre(b.square_mm, b.square_mm % 1 ? 2 : 0),
                                     cifre(b.marker_mm, b.marker_mm % 1 ? 2 : 0), b.dictionary)
      + (b.name === 'sheet' ? ' ' + T('live.board.note.sheet') : '');
    let r = reg.source === 'file' ? T('live.reg.file', reg.path) : T('live.reg.demo.' + (reg.demo || 'lying'));
    if (reg.error) r = T('live.reg.error', LIVE.name(s.prefs.registration)) + ' ' + r;
    $('lvRegNote').textContent = r;
    $('lvRegDemo').disabled = reg.source !== 'file' && !s.prefs.registration;
    const c = s.intrinsics;
    if (c.source === 'calibrated') {
      $('lvCamNote').textContent = T('live.cam.calibrated', c.date || '-', cifre(c.f, 1), c.rms_px === null ? '-' : cifre(c.rms_px, 3));
      $('lvCamPath').textContent = c.path;
    } else {
      $('lvCamNote').textContent = T('live.cam.nominal', cifre(c.f, 0), c.size[0], c.size[1]);
      $('lvCamPath').textContent = s.intrinsics_file;
    }

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
    if (e.type !== 'live') return;
    if (e.action === 'started') {
      work(false);
      if (page === 'live') toast(T('live.said.started'));
      else ask('live_stop');      // the page was left while the stream was opening
    } else if (e.action === 'failed') {
      work(false);
      toast(LIVE.plainError(e.message || e.error), 6000);
    } else if (e.action === 'stopped' && e.reason === 'photo') {
      toast(T('live.said.stopped.photo'), 3500);
    }
    if (page === 'live') LIVE.refresh();
  },

  wire: function () {
    setOptions($('lvBoard'), [T('live.board.patient'), T('live.board.sheet')], ['patient', 'sheet']);
    $('lvBoard').onchange = function () { LIVE.settings({ board: $('lvBoard').value }); };
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
  },
};
