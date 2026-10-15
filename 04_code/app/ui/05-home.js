// ImpactGuide · home: the start, as UniGuide's entry. Four large choices, each
// with one short line, and the state of the bench under them in three cells:
// the phone, the calibration, the next trial. The language is chosen here
// (ui/90-boot.js fills the picker). No panel and no toolbar: the rail stays,
// for moving between pages.

const HOME = {
  enter: function () {
    HOME.phone();
    ask('calib_state', function (s) { if (s && page === 'home') HOME.cal(s); });
    ask('trials_state', function (s) { if (s && page === 'home') HOME.trial(s); });
    be('phone_refresh');
  },

  dot: function (id, kind) { $(id).className = 'live-dot' + (kind ? ' ' + kind : ''); },

  // the link chosen and whether the phone is there on it
  phone: function () {
    const how = T(PHONE.link === 'wifi' ? 'link.btn.wifi' : 'link.btn.usb');
    const st = PHONE.ok ? 'ready' : (PHONE.state && hasT('link.state.' + PHONE.state) ? PHONE.state : 'unknown');
    $('hsPhoneV').textContent = how + ' · ' + T('link.state.' + st);
    HOME.dot('hsPhoneDot', PHONE.ok ? 'ok' : 'warn');
  },

  // the intrinsics stored for the two camera modes
  cal: function (s) {
    const p = !!s.modes.photo.stored, v = !!s.modes.video.stored;
    $('hsCalV').textContent = T(p && v ? 'home.cal.both' : p ? 'home.cal.photo' : v ? 'home.cal.video' : 'home.cal.none');
    HOME.dot('hsCalDot', p && v ? 'ok' : 'warn');
  },

  // the trial open now, or the next one of the sequence
  trial: function (s) {
    const t = s.trial;
    if (s.complete || !t) { $('hsTrialV').textContent = T('home.trial.done'); HOME.dot('hsTrialDot', 'ok'); return; }
    $('hsTrialV').textContent = t.open ? T('home.trial.open', t.trial)
      : T('home.trial.next', t.trial, s.totals.filled, s.totals.slots);
    HOME.dot('hsTrialDot', t.open ? 'warn' : '');
  },

  wire: function () {
    document.querySelectorAll('.home-tile[data-go]').forEach(function (b) {
      b.onclick = function () {
        if (b.disabled) return;
        if (b.dataset.go === 'trials') trialsView = 'guide';     // the next trial, step by step
        go(b.dataset.go);
      };
    });
    $('hsPhone').onclick = function () { if (!LINK.open) LINK.show(); };
    $('hsPhone').onkeydown = function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('hsPhone').click(); } };
  },
};
