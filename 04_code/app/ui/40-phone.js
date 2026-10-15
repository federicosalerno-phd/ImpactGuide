// ImpactGuide · the phone: the link (cable or Wi-Fi) with its pairing steps,
// its state and its quality, and the live view sent back to the phone's own
// screen. SlantUI's pane docked down the side of the work area (.sheet-side),
// opened from the Phone button on the rail. The work is phone_link.py's and
// phone_view.py's; the pane shows their state and asks.

// The sign of the phone, on SlantUI's 24 grid.
ICONS.phone = { s: '<rect x="7" y="3" width="10" height="18" rx="2"/><path d="M11 17.5h2"/>' };

const LINK = {
  state: null,
  timer: null,
  open: false,
  svg: { lkQr: '', vwQr: '' },   // what each QR tile holds, written only when it changes

  // The rail button says the link in use.
  btn: function () {
    const m = (LINK.state && LINK.state.mode) || PHONE.link || 'usb';
    $('lkBtnL').textContent = T('link.btn.' + (m === 'wifi' ? 'wifi' : 'usb'));
  },

  show: function () {
    LINK.open = true;
    $('lkBtn').classList.add('on');
    showSheet($('lkSheet'));
    LINK.refresh();
    clearInterval(LINK.timer);
    LINK.timer = setInterval(LINK.refresh, 1000);
  },

  hide: function () {
    LINK.open = false;
    $('lkBtn').classList.remove('on');
    hideSheet($('lkSheet'));
    clearInterval(LINK.timer);
    LINK.timer = null;
  },

  toggle: function () { if (LINK.open) LINK.hide(); else LINK.show(); },

  refresh: function () { ask('link_state', function (s) { if (s) LINK.render(s); }); },

  qr: function (id, svg) {
    if (LINK.svg[id] === svg) return;
    LINK.svg[id] = svg;
    $(id).innerHTML = svg || '';     // markup made by the backend from the QR modules, no data in it
  },

  render: function (s) {
    LINK.state = s;
    LINK.btn();
    if (!LINK.open) return;
    const ok = s.state === 'ready', wifi = s.mode === 'wifi', busy = s.busy || [];
    document.querySelectorAll('#lkModes .mitem').forEach(function (it) {
      it.classList.toggle('sel', it.dataset.mode === s.mode);
    });

    // the state, in words
    $('lkDot').className = 'live-dot' + (ok ? ' ok' : '');
    $('lkWord').textContent = ok && s.model ? T('link.state.ready.model', s.model)
      : T('link.state.' + (hasT('link.state.' + s.state) ? s.state : 'unknown'));
    const why = !ok && s.cause && hasT('link.cause.' + s.cause) ? T('link.cause.' + s.cause) : '';
    $('lkWhy').textContent = why;
    $('lkWhy').classList.toggle('hidden', !why);

    // the quality: round trip, the stream while the live view runs, the delay from the camera, the drops
    const rows = [];
    if (ok && wifi && s.address) rows.push(piRow(T('link.f.address'), s.address));
    if (ok) rows.push(piRow(T('link.f.rtt'), s.rtt_ms === null ? '-' : T('live.f.ms', cifre(s.rtt_ms, 0))));
    const st = s.stream || {};
    rows.push(piRow(T('link.f.stream'), st.running
      ? T('link.f.stream.v', cifre(st.fps || 0, 1), st.gap_p95_ms === null || st.gap_p95_ms === undefined ? '-' : cifre(st.gap_p95_ms, 0))
      : T('link.f.stream.off')));
    rows.push(piRow(T('link.f.latency'), s.latency_ms === null || s.latency_ms === undefined
      ? T('link.f.latency.none') : T('link.f.latency.v', cifre(s.latency_ms, 0))));
    if (wifi && (s.drops || s.reconnects)) rows.push(piRow(T('link.f.drops'), T('link.f.drops.v', s.drops, s.reconnects)));
    $('lkRows').replaceChildren.apply($('lkRows'), rows);
    const note = !ok && !wifi ? T('link.usb.note') : ok && wifi ? T('link.ready.note') : '';
    $('lkNote').textContent = note;
    $('lkNote').classList.toggle('hidden', !note);

    // pairing: on Wi-Fi while the phone is not there
    $('lkPair').classList.toggle('hidden', !wifi || ok);
    LINK.qr('lkQr', s.qr || '');
    $('lkQrWait').classList.toggle('hidden', !s.qr);
    $('lkQrBtnL').textContent = T(s.qr ? 'link.pair.qr.new' : 'link.pair.qr.show');
    $('lkPairBtn').disabled = busy.indexOf('pair') >= 0;
    $('lkConnect').disabled = busy.indexOf('connect') >= 0;
    if (s.address && !$('lkAddr').value && document.activeElement !== $('lkAddr')) $('lkAddr').value = s.address;

    // the live view on the phone's screen
    const v = s.view || {}, vr = [];
    vr.push(piRow(T('view.f.viewers'), v.viewers ? cifre(v.viewers, 0) : T('view.none')));
    if (v.viewers && v.added_ms !== null && v.added_ms !== undefined) {
      vr.push(piRow(T('view.f.fps'), T('live.f.fps', cifre(v.fps || 0, 1))),
              piRow(T('view.f.added'), T('view.f.added.v', cifre(v.added_ms, 0), cifre(v.added_p95_ms, 0))));
      if (v.live && v.pc_ms !== null && v.pc_ms !== undefined) {
        const total = (v.camera_ms || 0) + v.pc_ms + v.added_ms;
        vr.push(piRow(T('view.f.total'), v.camera_ms === null || v.camera_ms === undefined
          ? T('view.f.total.pc', cifre(v.pc_ms + v.added_ms, 0)) : T('live.f.ms', cifre(total, 0))));
      }
      vr.push(piRow(T('view.f.data'), T('view.f.data.v', cifre(v.kb || 0, 0), cifre(v.mbit_s || 0, 1))));
    }
    $('vwRows').replaceChildren.apply($('vwRows'), vr);
    LINK.qr('vwQr', v.qr || '');
    $('vwUrl').textContent = v.url_lan || '';
    $('vwOpen').disabled = !ok || busy.indexOf('view_open') >= 0;
  },

  setMode: function (mode) {
    if (LINK.state && LINK.state.mode === mode) return;
    ask('link_mode', mode, function (s) {
      if (!s) return;
      PHONE.link = mode;
      LINK.render(s);
      toast(T('link.said.' + mode));
    });
  },

  onEvent: function (e) {
    if (e.type === 'phone') { LINK.btn(); return; }
    if (e.type === 'link') {
      if (e.action === 'state' && e.back) toast(T('link.said.back'), 3500);
      else if (e.action === 'paired') toast(T('link.said.paired'), 3500);
      else if (e.action === 'connected') toast(T('link.said.connected'), 3500);
      else if (e.action === 'failed') toast(sayError(e.cause || e.error, ''), 6000);
      if (LINK.open) LINK.refresh();
    } else if (e.type === 'view') {
      if (e.action === 'opened') toast(T('view.said.opened'), 3500);
      else if (e.action === 'failed') toast(sayError(e.cause || e.error, ''), 6000);
      if (LINK.open) LINK.refresh();
    } else if (e.type === 'live' && e.action === 'stopped' && e.reason === 'link') {
      toast(T('link.said.live_stopped'), 4000);
    }
  },

  wire: function () {
    $('lkBtn').onclick = LINK.toggle;
    $('lkClose').onclick = LINK.hide;
    document.querySelectorAll('#lkModes .mitem').forEach(function (it) {
      it.onclick = function () { LINK.setMode(it.dataset.mode); };
      it.onkeydown = function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); it.click(); } };
    });
    $('lkQrBtn').onclick = function () { ask('link_qr', function (s) { if (s) LINK.render(s); }); };
    $('lkPairBtn').onclick = function () {
      ask('link_pair', $('lkPairAddr').value.trim(), $('lkPairCode').value.trim(), function (s) {
        if (s) LINK.render(s);
      });
    };
    $('lkConnect').onclick = function () {
      ask('link_connect', $('lkAddr').value.trim(), function (s) { if (s) LINK.render(s); });
    };
    $('vwOpen').onclick = function () { ask('view_open', function (s) { if (s) LINK.render(s); }); };
    LINK.btn();
    if (!LINK.keys) {
      LINK.keys = true;
      window.addEventListener('keydown', function (ev) {
        if (ev.key === 'Escape' && LINK.open && !sheetIsOpen()) { ev.preventDefault(); LINK.hide(); }
      });
    }
  },
};
