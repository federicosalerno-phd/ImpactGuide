// ImpactGuide · home: the start. Four large choices, one per module, each with
// one short line, and the state of the bench under them as one quiet line
// (SlantUI's .choices and .stcells.line): the phone, the calibration, the next
// trial, each a word and a short value, and each a way in: the phone to its
// panel, the other two to their module. The language is chosen here
// (ui/90-boot.js fills the picker). No rail, no toolbar, no panel: only the
// Phone panel, from the phone's state.
'use strict';

/** Home: its choices and the line of the bench's state. */
const HOME = {
  enter: function () {
    HOME.phone();
    askQuiet('calib_state', function (s) { if (s && page === 'home') HOME.cal(s); });
    askQuiet('trials_state', function (s) { if (s && page === 'home') HOME.trial(s); });
    be('phone_refresh');
  },

  // how the phone is linked, found by itself: cable, Wi-Fi, or no phone
  phone: function () {
    const busy = PHONE.state === 'reconnecting' || PHONE.state === 'connecting';
    const word = PHONE.ok ? (PHONE.link === 'wifi' ? 'home.ph.wifi' : 'home.ph.usb')
      : busy ? 'home.ph.busy' : 'home.ph.off';
    setStcell($('hsPhone'), PHONE.ok ? 'ok' : busy ? 'busy' : 'warn', T(word));
  },

  // the intrinsics stored for the three camera modes: the phone's photos, its videos (the trials'), the live stream;
  // the modes calibrated, and on the hint those still to do
  cal: function (s) {
    const modes = ['photo', 'video', 'stream'];
    const name = function (m) { return T('home.cal.m.' + m); };
    const have = modes.filter(function (m) { return s.modes[m] && s.modes[m].stored; });
    const missing = modes.filter(function (m) { return have.indexOf(m) < 0; });
    setStcell($('hsCal'), missing.length ? 'warn' : 'ok', have.length ? have.map(name).join(' · ') : T('home.cal.none'));
    $('hsCal').title = missing.length ? T('home.cal.missing', missing.map(name).join(', ')) : T('home.cal');
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
      b.setAttribute('data-sound', 'forward');      // SlantUI's voice, on the press
      b.onclick = function () {
        if (b.disabled) return;
        const m = b.dataset.go;
        go(m, MODULES[m].start);      // a module opens at its first step
      };
    });
    $('hsPhone').onclick = function () { if (!LINK.open) LINK.show(); };
    $('hsCal').onclick = function () { go('calibration', MODULES.calibration.start); };
    $('hsTrial').onclick = function () { go('trials', MODULES.trials.start); };
    $('hsCal').setAttribute('data-sound', 'forward');
    $('hsTrial').setAttribute('data-sound', 'forward');
  },
};
