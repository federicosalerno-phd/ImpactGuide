// ImpactGuide · the phone: how it is linked, and the live view sent back to its
// own screen. The app finds the phone by itself and keeps watching it
// (phone_link.py, "auto"): the cable when the phone is on it, else the Wi-Fi it
// was paired on; nothing is chosen here. Wi-Fi is paired only when it is asked
// for (Use Wi-Fi), once per PC: with the cable in, the app switches Wireless
// debugging on by itself and one step is left on the phone. The panel takes
// the place of the step's panel down the right of the window (at home too), so
// it is part of the window and never covers the page; it is opened from the
// phone button on the rail or the phone cell of home. The measures, pairing
// with a code, connecting by address and the code for another screen sit under
// Advanced. The work is phone_link.py's and phone_view.py's.

// The sign of the phone, on SlantUI's 24 grid.
ICONS.phone = { s: '<rect x="7" y="3" width="10" height="18" rx="2"/><path d="M11 17.5h2"/>' };

const LINK = {
  state: null,
  timer: null,
  open: false,
  pairing: false,    // Use Wi-Fi was pressed: the pairing steps are on show
  svg: { lkQr: '', vwQr: '' },   // what each QR tile holds, written only when it changes

  // The rail button says how the phone is linked: cable, Wi-Fi, or no phone (the backend's phone events).
  btn: function () {
    $('lkBtnL').textContent = PHONE.ok ? T('link.btn.' + (PHONE.link === 'wifi' ? 'wifi' : 'usb')) : T('link.btn.none');
  },

  show: function () {
    LINK.open = true;
    $('lkBtn').classList.add('on');
    document.querySelector('.app').classList.add('phone-open');
    document.querySelectorAll('.rp-page').forEach(function (p) { p.classList.toggle('show', p.dataset.panel === 'phone'); });
    LINK.refresh();
    clearInterval(LINK.timer);
    LINK.timer = setInterval(LINK.refresh, 1000);
  },

  hide: function () {
    LINK.open = false;
    $('lkBtn').classList.remove('on');
    document.querySelector('.app').classList.remove('phone-open');
    clearInterval(LINK.timer);
    LINK.timer = null;
    showPanel(panelNow);          // the step's own panel comes back
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
    const ok = s.state === 'ready', wifi = s.via === 'wifi', busy = s.busy || [];

    // the state, in words: on the cable, on Wi-Fi, or no phone and why
    $('lkDot').className = 'live-dot' + (ok ? ' ok' : s.state === 'reconnecting' || s.state === 'connecting' ? ' busy' : '');
    $('lkWord').textContent = ok ? T(wifi ? 'link.on.wifi' : 'link.on.usb')
      : T('link.state.' + (hasT('link.state.' + s.state) ? s.state : 'unknown'));
    const why = !ok && s.cause && hasT('link.cause.' + s.cause) ? T('link.cause.' + s.cause) : '';
    $('lkWhy').textContent = why;
    $('lkWhy').classList.toggle('hidden', !why);

    // the phone and its round trip in plain view; the address, the stream, the delay and the drops under Advanced
    const rows = [];
    if (ok && s.model) rows.push(piRow(T('link.f.phone'), s.model));
    if (ok) rows.push(piRow(T('link.f.rtt'), s.rtt_ms === null ? '-' : T('live.f.ms', cifre(s.rtt_ms, 0))));
    if (s.paired && !(ok && wifi)) rows.push(piRow(T('link.f.wifi'), T('link.f.wifi.paired')));
    $('lkRows').replaceChildren.apply($('lkRows'), rows);
    const more = [];
    if (ok && wifi && s.address) more.push(piRow(T('link.f.address'), s.address));
    const st = s.stream || {};
    more.push(piRow(T('link.f.stream'), st.running
      ? T('link.f.stream.v', cifre(st.fps || 0, 1), st.gap_p95_ms === null || st.gap_p95_ms === undefined ? '-' : cifre(st.gap_p95_ms, 0))
      : T('link.f.stream.off')));
    more.push(piRow(T('link.f.latency'), s.latency_ms === null || s.latency_ms === undefined
      ? T('link.f.latency.none') : T('link.f.latency.v', cifre(s.latency_ms, 0))));
    if (wifi && (s.drops || s.reconnects)) more.push(piRow(T('link.f.drops'), T('link.f.drops.v', s.drops, s.reconnects)));
    $('lkMore').replaceChildren.apply($('lkMore'), more);

    // Wi-Fi only when wanted: Use Wi-Fi while it was never paired; its steps once pressed, until it is paired.
    // Paired once, the app goes on Wi-Fi by itself when the cable is out; pairing again is under Advanced.
    if (ok && wifi) LINK.pairing = false;
    $('lkWifi').classList.toggle('hidden', !!s.paired || LINK.pairing);
    $('lkPair').classList.toggle('hidden', !LINK.pairing);
    // with the cable in, the app has switched Wireless debugging on: one step on the phone is left
    const short = !!s.wifi_on;
    $('lkS1').classList.toggle('hidden', short);
    $('lkS2').classList.toggle('hidden', short);
    $('lkS3n').textContent = short ? '1' : '3';
    LINK.qr('lkQr', s.qr || '');
    $('lkQrWait').classList.toggle('hidden', !s.qr);
    $('lkQrBtn').disabled = busy.indexOf('qr') >= 0 && !s.qr;
    $('lkPairBtn').disabled = busy.indexOf('pair') >= 0;
    $('lkConnect').disabled = busy.indexOf('connect') >= 0;
    if (s.address && !$('lkAddr').value && document.activeElement !== $('lkAddr')) $('lkAddr').value = s.address;

    // the live view on the phone's screen: how many screens show it; its measures under Advanced
    const v = s.view || {}, vr = [];
    $('vwRows').replaceChildren(piRow(T('view.f.viewers'), v.viewers ? cifre(v.viewers, 0) : T('view.none')));
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
    $('vwMore').replaceChildren.apply($('vwMore'), vr);
    LINK.qr('vwQr', v.qr || '');
    $('vwUrl').textContent = v.url_lan || '';
    $('vwOpen').disabled = !ok || busy.indexOf('view_open') >= 0;
  },

  // Use Wi-Fi: the pairing steps and the QR code (Wireless debugging switched on from here when on the cable)
  wifi: function () {
    LINK.pairing = true;
    ask('link_wifi', function (s) { if (s) LINK.render(s); });
  },

  onEvent: function (e) {
    if (e.type === 'phone') { LINK.btn(); if (LINK.open) LINK.refresh(); return; }
    if (e.type === 'link') {
      if (e.action === 'state' && e.back) toast(T('link.said.back'), 3500);
      else if (e.action === 'via') toast(T('link.said.' + (e.via === 'wifi' ? 'wifi' : 'usb')), 3500);
      else if (e.action === 'paired') { LINK.pairing = false; toast(T('link.said.paired'), 3500); }
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
    $('lkWifi').onclick = LINK.wifi;
    $('lkWifiAgain').onclick = LINK.wifi;
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
