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
'use strict';

const LINK_POLL_MS = 1000;      // the state of the link, read while the panel is open
const LINK_FAILS_SAID = 2;      // reads of the state failed in a row before the panel says so

// The sign of the phone, on SlantUI's 24 grid.
ICONS.phone = { s: '<rect x="7" y="3" width="10" height="18" rx="2"/><path d="M11 17.5h2"/>' };

/** The Phone panel: the link to the phone, Wi-Fi pairing, and the live view on the phone's own screen. */
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
    clearInterval(LINK.timer);
    LINK.timer = poll('link_state', LINK_POLL_MS, LINK.polled);
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

  // Read once more (an event said something changed); a failure is said by the panel's next poll.
  refresh: function () { askQuiet('link_state', function (s) { if (s) LINK.render(s); }); },

  polled: function (s, bad, fails) {
    if (s) LINK.render(s);
    else if (fails >= LINK_FAILS_SAID && LINK.open) LINK.unreachable(bad);
  },

  // The state cannot be read: the panel says so in place of the link's state.
  unreachable: function (bad) {
    $('lkDot').className = 'live-dot';
    $('lkWord').textContent = T('link.state.unknown');
    $('lkWhy').textContent = errorText(bad);
    $('lkWhy').classList.remove('hidden');
  },

  qr: function (id, svg) {
    if (LINK.svg[id] === svg) return;
    LINK.svg[id] = svg;
    $(id).innerHTML = svg || '';     // SVG made by the app (phone_view.qr_svg) from the QR modules: no data in it
  },

  render: function (s) {
    LINK.state = s;
    LINK.btn();
    if (!LINK.open) return;
    const ok = s.state === 'ready', wifi = s.via === 'wifi', busy = s.busy || [];

    // the state, in words: on the cable, on Wi-Fi, or no phone and why
    $('lkDot').className = 'live-dot' + (ok ? ' ok' : s.state === 'reconnecting' || s.state === 'connecting' ? ' busy' : '');
    $('lkWord').textContent = ok ? T(wifi ? 'link.on.wifi' : 'link.on.usb') : Tcode('link.state.', s.state, 'unknown');
    const why = !ok && s.cause && hasT('link.cause.' + s.cause) ? T('link.cause.' + s.cause) : '';
    $('lkWhy').textContent = why;
    $('lkWhy').classList.toggle('hidden', !why);

    // the phone and its round trip in plain view; the address, the stream, the delay and the drops under Advanced
    const rows = [];
    if (ok && s.model) rows.push(piRow(T('link.f.phone'), s.model));
    if (ok) rows.push(piRow(T('link.f.rtt'), s.rtt_ms === null ? '-' : T('live.f.ms', fmtNum(s.rtt_ms, 0))));
    if (s.paired && !(ok && wifi)) rows.push(piRow(T('link.f.wifi'), T('link.f.wifi.paired')));
    // files copied to the PC whose deletion on the phone failed: how many, their names on hover
    const left = LINK.leftOnPhone();
    if (left.length) {
      const row = piRow(T('link.f.left'), fmtNum(left.length, 0));
      row.title = left.join(', ');
      rows.push(row);
    }
    $('lkRows').replaceChildren.apply($('lkRows'), rows);
    const more = [];
    if (ok && wifi && s.address) more.push(piRow(T('link.f.address'), s.address));
    const st = s.stream || {};
    more.push(piRow(T('link.f.stream'), st.running
      ? T('link.f.stream.v', fmtNum(st.fps || 0, 1), st.gap_p95_ms === null || st.gap_p95_ms === undefined ? '-' : fmtNum(st.gap_p95_ms, 0))
      : T('link.f.stream.off')));
    more.push(piRow(T('link.f.latency'), s.latency_ms === null || s.latency_ms === undefined
      ? T('link.f.latency.none') : T('link.f.latency.v', fmtNum(s.latency_ms, 0))));
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
    LINK.view(s.view || {}, ok, busy);
  },

  // The live view on the phone's screen: how many screens show it; its measures under Advanced.
  view: function (v, ok, busy) {
    const vr = [];
    $('vwRows').replaceChildren(piRow(T('view.f.viewers'), v.viewers ? fmtNum(v.viewers, 0) : T('view.none')));
    if (v.viewers && v.added_ms !== null && v.added_ms !== undefined) {
      vr.push(piRow(T('view.f.fps'), T('live.f.fps', fmtNum(v.fps || 0, 1))),
              piRow(T('view.f.added'), T('view.f.added.v', fmtNum(v.added_ms, 0), fmtNum(v.added_p95_ms, 0))));
      if (v.live && v.pc_ms !== null && v.pc_ms !== undefined) {
        const total = (v.camera_ms || 0) + v.pc_ms + v.added_ms;
        vr.push(piRow(T('view.f.total'), v.camera_ms === null || v.camera_ms === undefined
          ? T('view.f.total.pc', fmtNum(v.pc_ms + v.added_ms, 0)) : T('live.f.ms', fmtNum(total, 0))));
      }
      vr.push(piRow(T('view.f.data'), T('view.f.data.v', fmtNum(v.kb || 0, 0), fmtNum(v.mbit_s || 0, 1))));
    }
    $('vwMore').replaceChildren.apply($('vwMore'), vr);
    LINK.qr('vwQr', v.qr || '');
    $('vwUrl').textContent = v.url_lan || '';
    // another screen on the network: off until the operator asks (view_lan), then the code to scan
    const lan = hasSlot('view_lan');
    ['vwOtherSec', 'vwLan'].forEach(function (id) { $(id).closest('.toggle, .sec-lbl').classList.toggle('hidden', !lan); });
    if (document.activeElement !== $('vwLan')) $('vwLan').checked = !!v.lan;
    ['vwQr', 'vwQrNote'].forEach(function (id) { $(id).classList.toggle('hidden', !v.qr); });
    $('vwOpen').disabled = !ok || busy.indexOf('view_open') >= 0;
  },

  // The names of the files copied from the phone whose deletion there failed (the backend's phone events).
  leftOnPhone: function () {
    return Array.isArray(PHONE.left_on_phone) ? PHONE.left_on_phone.map(String) : [];
  },

  // A phone event with more files left on the phone than before: said once, with the new names.
  sayLeft: function (before) {
    const now = LINK.leftOnPhone();
    if (now.length <= before.length) return;
    const fresh = now.filter(function (n) { return before.indexOf(n) < 0; });
    const names = fresh.slice(0, DETAIL_LIST_MAX).join(', ') + (fresh.length > DETAIL_LIST_MAX ? ' …' : '');
    if (names) toast(T('link.said.left', names), TOAST_LONG_MS);
  },

  // Show the view to other screens of the network, or stop: the answer is the link state with the code, or none.
  lan: function () {
    const on = $('vwLan').checked;
    ask('view_lan', on, function (s) {
      if (s) LINK.render(s);
      else $('vwLan').checked = !on;
    });
  },

  // Use Wi-Fi: the pairing steps and the QR code (Wireless debugging switched on from here when on the cable)
  wifi: function () {
    LINK.pairing = true;
    ask('link_wifi', function (s) { if (s) LINK.render(s); });
  },

  onEvent: function (e) {
    if (e.type === 'phone') { LINK.btn(); if (LINK.open) LINK.refresh(); return; }
    if (e.type === 'link') {
      if (e.action === 'state' && e.back) toast(T('link.said.back'), TOAST_SAID_MS);
      else if (e.action === 'via') toast(T('link.said.' + (e.via === 'wifi' ? 'wifi' : 'usb')), TOAST_SAID_MS);
      else if (e.action === 'paired') { LINK.pairing = false; toast(T('link.said.paired'), TOAST_SAID_MS); }
      else if (e.action === 'connected') toast(T('link.said.connected'), TOAST_SAID_MS);
      else if (e.action === 'failed') toast(errorText(e), TOAST_LONG_MS, 'error');
      if (LINK.open) LINK.refresh();
    } else if (e.type === 'view') {
      if (e.action === 'opened') toast(T('view.said.opened'), TOAST_SAID_MS);
      else if (e.action === 'failed') toast(errorText(e), TOAST_LONG_MS, 'error');
      if (LINK.open) LINK.refresh();
    } else if (e.type === 'live' && e.action === 'stopped' && e.reason === 'link') {
      toast(T('link.said.live_stopped'), TOAST_SAID_MS);
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
    $('vwLan').onchange = LINK.lan;
    LINK.btn();
    if (!LINK.keys) {
      LINK.keys = true;
      window.addEventListener('keydown', function (ev) {
        if (ev.key === 'Escape' && LINK.open && !sheetIsOpen()) { ev.preventDefault(); LINK.hide(); }
      });
    }
  },
};
