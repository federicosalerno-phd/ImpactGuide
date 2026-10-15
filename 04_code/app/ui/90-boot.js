// ImpactGuide · start up. Loads last: every part is defined by now.

function onEvent(raw) {
  let e = null;
  try { e = JSON.parse(raw); } catch (err) { return; }
  if (!e) return;
  if (e.type === 'phone') { PHONE = e; showPhone(); return; }
  if (e.type === 'acquire_failed') {
    const key = 'err.' + e.error;
    toast(hasT(key) ? T(key, e.message || '') : T('err.generic', e.error + (e.message ? ': ' + e.message : '')), 5000);
  }
  if (e.type === 'error') toast(T('err.generic', e.message || ''), 6000);
  CAL.onEvent(e);
  TRIALS.onEvent(e);
  LIVE.onEvent(e);
}

function inlineMark() {
  try {
    const x = new XMLHttpRequest();
    x.open('GET', 'ui/impactguide-icon.svg', false);
    x.send();
    $('tbLogo').innerHTML = x.responseText;
  } catch (e) { /* the band still has the name */ }
}

function boot() {
  Bridge.init();
  initSelects();
  inlineMark();
  placeIcons(document);
  initTitlebar();
  Bridge.on('happened', onEvent);
  // Everything that holds words: run at start and again when the language changes
  // (the handlers are assigned, not added, so wiring twice changes nothing).
  const speak = function (lang) {
    setLang(lang);
    setWindowHints({ maximise: T('win.maximise.hint'), restore: T('win.restore.hint') });
    wireRail();
    CAL.wire();
    TRIALS.wire();
    LIVE.wire();
    showPhone();
  };
  beJson('app_info', function (info) {
    speak(info.lang);
    // SlantUI's picker with flags: flag and tag on the button; flag, tag and the name in the
    // language itself in the list, at most 300 px and then it scrolls. The list comes from
    // the host (impactguide_app.py, LANGUAGES), the flags from ui/flags/ (flag-icons, MIT).
    setPicker($('lang'), info.langs.map(function (l) {
      return { code: l.code, tag: l.code.toUpperCase(), label: l.native, flag: 'ui/flags/' + l.flag + '.svg' };
    }), info.lang, { maxHeight: 300 });
    $('lang').onchange = function () {
      be('set_lang', $('lang').value);
      speak($('lang').value);
      go(page);
    };
    $('simChip').classList.toggle('hidden', !info.simulate);
    go(info.page);
  });
}

boot();
