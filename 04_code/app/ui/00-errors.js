// ImpactGuide · the errors of the page's scripts, caught from the first script on (this one is loaded before every
// other): kept in window.__errs, which the window test reads, and written to the app's log through the backend
// (page_error) as soon as it can be reached. Never shown: the operator only ever sees the app's own sentences.
'use strict';

(function () {
  const MAX_KEPT = 50;          // errors kept (and sent) at most: a script failing on every frame never floods the log
  const MAX_CHARS = 500;
  const FLUSH_MS = 2000;        // errors from before the backend could be reached are sent this often
  const errs = window.__errs = window.__errs || [];
  let sent = 0;

  function flush() {
    const b = window.Bridge && Bridge.obj;
    if (!b || typeof b.page_error !== 'function') return;
    while (sent < errs.length) {
      try { b.page_error(errs[sent]); } catch (err) { return; }
      sent++;
    }
  }

  function keep(text) {
    if (errs.length >= MAX_KEPT) return;
    errs.push(String(text).slice(0, MAX_CHARS));
    flush();
  }

  window.addEventListener('error', function (e) {
    const file = e.filename ? String(e.filename).split('/').pop() : '';
    keep((e.message || 'error') + ' line ' + e.lineno + (file ? ' ' + file : ''));
  });
  window.addEventListener('unhandledrejection', function (e) {
    const r = e.reason;
    keep('promise: ' + (r && r.message ? r.message : String(r)));
  });
  setInterval(function () { if (sent < errs.length) flush(); }, FLUSH_MS);
})();
