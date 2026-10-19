// ImpactGuide · home: the start. Four large choices, one per module, each with
// one short line, and the state of the bench under them in three cells
// (SlantUI's .choices and .stcells): the phone, the calibration, the next trial.
// The language is chosen here (ui/90-boot.js fills the picker). No rail, no
// toolbar, no panel: only the Phone panel, from the phone cell.
'use strict';

/** Home: its choices and the three cells of the bench's state. */
const HOME = {
  enter: function () {
    HOME.phone();
    askQuiet('calib_state', function (s) { if (s && page === 'home') HOME.cal(s); });
    askQuiet('trials_state', function (s) { if (s && page === 'home') HOME.trial(s); });
    be('phone_refresh');
  },

  // how the phone is linked, found by itself: cable, Wi-Fi, or no phone
  phone: function () {
    setStcell($('hsPhone'), PHONE.ok ? 'ok' : PHONE.state === 'reconnecting' || PHONE.state === 'connecting' ? 'busy' : 'warn',
              PHONE.ok ? T(PHONE.link === 'wifi' ? 'link.on.wifi' : 'link.on.usb') : Tcode('link.state.', PHONE.state, 'none'));
  },

  // the intrinsics stored for the two camera modes
  cal: function (s) {
    const p = !!s.modes.photo.stored, v = !!s.modes.stream.stored;
    setStcell($('hsCal'), p && v ? 'ok' : 'warn',
              T(p && v ? 'home.cal.both' : p ? 'home.cal.photo' : v ? 'home.cal.stream' : 'home.cal.none'));
  },

  // the trial open now, or the next one of the sequence
  trial: function (s) {
    const t = s.trial;
    if (s.complete || !t) { setStcell($('hsTrial'), 'ok', T('home.trial.done')); return; }
    setStcell($('hsTrial'), t.open ? 'warn' : '', t.open ? T('home.trial.open', t.trial)
      : T('home.trial.next', t.trial, s.totals.filled, s.totals.slots));
  },

  wire: function () {
    document.querySelectorAll('.choice[data-go]').forEach(function (b) {
      b.onclick = function () {
        if (b.disabled) return;
        const m = b.dataset.go;
        go(m, MODULES[m].start);      // a module opens at its first step
      };
    });
    $('hsPhone').onclick = function () { if (!LINK.open) LINK.show(); };
  },
};
