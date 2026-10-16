// ImpactGuide · start up. Loads last: every part is defined by now.

function onEvent(raw) {
  let e = null;
  try { e = JSON.parse(raw); } catch (err) { return; }
  if (!e) return;
  if (e.type === 'phone') { PHONE = e; showPhone(); LINK.onEvent(e); return; }
  if (e.type === 'acquire_failed') toast(sayError(e.error, e.message), 5000);
  if (e.type === 'error') toast(sayError('generic', e.message), 6000);
  CAL.onEvent(e);
  TRIALS.onEvent(e);
  LIVE.onEvent(e);
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
    wireSteps();
    HOME.wire();
    CAL.wire();
    TRIALS.wire();
    LIVE.wire();
    LINK.wire();
    showPhone();
  };
  beJson('app_info', function (info) {
    speak(info.lang);
    // SlantUI's picker with flags, on the home page as in UniGuide: flag and tag on the button; flag,
    // tag and the name in the language itself in the list, at most 300 px and then it scrolls. The
    // list comes from the host (impactguide_app.py, LANGUAGES), the flags from ui/flags/ (flag-icons, MIT).
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
  });
}

boot();
