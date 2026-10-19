// ImpactGuide · start up. Loads last: every part is defined by now.
'use strict';

// The work of the page (workOn keys) that each job of the backend ends, for an `error` event naming its job; a job
// without work of the page here ends none, an event without a job ends all.
const JOB_WORK = { phone: ['cal_photo', 'trial_file'], phone_copy: ['cal_photo'], photo_check: ['cal_photo'], calibration: ['calib'],
                   live: ['live', 'cal_stream'], import_file: ['trial_file'] };

// The backend's events, to every part that follows them; what concerns no part in particular is said here.
let phoneHeard = false;     // a phone event came already: the next one that changes the link is a phone come or gone
function onEvent(raw) {
  let e = null;
  try { e = JSON.parse(raw); } catch (err) { console.warn('an event that is not JSON', raw); return; }
  if (!e || typeof e !== 'object') return;
  if (e.type === 'phone') {
    const left = LINK.leftOnPhone(), was = !!PHONE.ok, heard = phoneHeard;
    PHONE = e;
    phoneHeard = true;
    // the phone came, or went; never at the start (the first state is how things stand, nothing that happened)
    if (heard && !!e.ok !== was) playSound(e.ok ? 'on' : 'off');
    LINK.sayLeft(left);
    showPhone();
    LINK.onEvent(e);
    return;
  }
  if (e.type === 'acquire_failed') toast(errorText(e), TOAST_LONG_MS, 'error');
  if (e.type === 'error') {
    // something failed in the backend without the event of its own work: that work is over, the page reads again
    toast(errorText({ error: e.error || 'internal', args: e.args, message: e.message }), TOAST_LONG_MS, 'error');
    if (!e.job) workClear();
    else if (Object.prototype.hasOwnProperty.call(JOB_WORK, e.job)) JOB_WORK[e.job].forEach(function (k) { workOff(k); });
    if (page === 'calibration') CAL.refresh();
    else if (page === 'trials' && trialsView === 'guide') TRIALS.refresh();
  }
  CAL.onEvent(e);
  TRIALS.onEvent(e);
  LIVE.onEvent(e);
  TWIN.onEvent(e);
  LINK.onEvent(e);
}

function boot() {
  Bridge.init();
  initSelects();
  placeIcons(document);
  initTitlebar();
  initFolds(document);
  Bridge.on('happened', onEvent);
  // Everything that holds words: run at start and again when the language changes
  // (the handlers are assigned, not added, so wiring twice changes nothing).
  const speak = function (lang) {
    setLang(lang);
    setWindowHints({ maximise: T('win.maximise.hint'), restore: T('win.restore.hint') });
    setSoundHints({ on: T('win.sound.on.hint'), off: T('win.sound.off.hint') });
    wireSteps();
    HOME.wire();
    CAL.wire();
    TRIALS.wire();
    LIVE.wire();
    LINK.wire();
    TWIN.wire();
    showWork();         // the pill: what the app is doing, or the phone, in the new language
  };
  askQuiet('app_info', function (info, bad) {
    if (!info) {             // no answer to the first call: the page still opens, in English, and says so
      speak('en');
      go('home');
      failed(bad);
      return;
    }
    speak(info.lang);
    // SlantUI's picker with flags, on the home page: flag and tag on the button; flag, tag and the name in the
    // language itself in the list, at most 300 px and then it scrolls. The list comes from the host
    // (impactguide_app.py, LANGUAGES), the flags from ui/flags/ (flag-icons, MIT).
    setPicker($('lang'), info.langs.map(function (l) {
      return { code: l.code, tag: l.code.toUpperCase(), label: l.native, flag: 'ui/flags/' + l.flag + '.svg' };
    }), info.lang, { maxHeight: 300 });
    $('lang').onchange = function () {
      be('set_lang', $('lang').value);
      speak($('lang').value);
      const open = LINK.open;
      go(page);
      if (open) LINK.show();
    };
    $('simChip').classList.toggle('hidden', !info.simulate);
    PHONE.link = info.link;
    LINK.btn();
    // the app opens at home; a module asked for on the command line opens instead
    go(info.page || 'home');
    // SlantUI's sounds, once the start is made (it makes none): the speaker on the band and Ctrl+M, the choice kept
    // in the preferences of this computer
    initSounds({ on: info.sound !== false, keep: function (on) { be('set_sound', on); },
                 hints: { on: T('win.sound.on.hint'), off: T('win.sound.off.hint') } });
  });
}

boot();
