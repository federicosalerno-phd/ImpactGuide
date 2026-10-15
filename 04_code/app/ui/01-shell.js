// ImpactGuide · the shell: home, the rail, the views on the stage, the panels,
// the Advanced parts, the status pill, the toast and the wait. A plain script
// sharing one global scope.

// The pages of the rail and whether they are built. A page not built yet opens
// and says which part of the work will bring it. Home is not on the rail: it is
// the start, and the mark on the band leads back to it.
const PAGES = ['calibration', 'live', 'trials', 'scans', 'analysis', 'charts', 'dataset'];
const BUILT = { calibration: true, live: true, trials: true };

// The signs this application adds to SlantUI's set, on the same 24 grid.
ICONS.camera = { s: '<path d="M4 8.5A1.5 1.5 0 0 1 5.5 7h2.3l1.4-2h5.6l1.4 2h2.3A1.5 1.5 0 0 1 20 8.5v9'
  + 'a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5z"/><circle cx="12" cy="12.8" r="3.4"/>' };
ICONS.pendulum = { s: '<path d="M5 4h14"/><path d="M12 4 8.9 13.8"/><circle cx="8" cy="16.6" r="2.9"/>'
  + '<path d="M14.6 17.2a13 13 0 0 0 4.3-3.6" stroke-dasharray="1.6 2"/>' };
ICONS.chart = { s: '<path d="M4 20h16"/><path d="M7 16.5V11"/><path d="M12 16.5V6"/><path d="M17 16.5v-3.5"/>' };

let page = 'home';
let trialsView = 'guide';

function $(id) { return document.getElementById(id); }

// An element with a class and a text; never markup from data.
function E(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined && text !== null) el.textContent = String(text);
  return el;
}

function piRow(label, value, cls) {
  const r = E('div', 'pi-row' + (cls ? ' ' + cls : ''));
  r.append(E('span', 'pi-l', label), E('span', null, value));
  return r;
}

function chip(text, kind) { return E('span', 'chip' + (kind ? ' chip-' + kind : ''), text); }

// The signs in the markup: a [data-icon] span becomes the <svg> itself, so the
// button centres sign and word by their ink; a window button gets it inside.
function placeIcons(root) {
  (root || document).querySelectorAll('[data-icon]').forEach(function (el) {
    const name = el.getAttribute('data-icon');
    if (el.tagName === 'SPAN') {
      const holder = document.createElement('span');
      holder.innerHTML = icon(name);
      const svg = holder.firstElementChild;
      if (!svg) return;
      if (el.previousElementSibling) svg.classList.add('r');
      if (el.id) svg.id = el.id;
      el.replaceWith(svg);
    } else {
      setIcon(el, name);
      el.removeAttribute('data-icon');
    }
  });
}

// ── Advanced ─────────────────────────────────────────────────────────────────
// What is rarely needed sits in a part of the panel that stays closed until its
// button is pressed (SlantUI has no folding section: this is a plain button that
// shows the stack after it). Open or closed is kept while the app runs.
function wireAdvanced(root) {
  (root || document).querySelectorAll('.adv-btn[data-adv]').forEach(function (b) {
    const box = $(b.getAttribute('data-adv'));
    b.setAttribute('aria-expanded', 'false');
    b.onclick = function () {
      const open = box.classList.contains('hidden');
      box.classList.toggle('hidden', !open);
      b.classList.toggle('on', open);
      b.setAttribute('aria-expanded', open ? 'true' : 'false');
    };
  });
}

// ── the answers of the backend ─────────────────────────────────────────────
// Every slot answers JSON; a failure carries `error`, a cause the page names.
// A cause of the link to the phone (phone_link.py) has its own plain words and
// never shows adb's text.
// What the operator reads is always a sentence of the app: a detail from the backend is kept only
// when it is short and plain (a file or a trial name), never an exception or a log line.
function plainDetail(m) {
  m = String(m || '').trim();
  if (!m || m.length > 48 || /[|\n\r{}<>]|exception|error|java|android|traceback|adb|server|socket|\bat\s/i.test(m)) return '';
  return m;
}
function sayError(code, message) {
  if (message) console.warn('error', code, message);
  const d = plainDetail(message);
  if (hasT('err.' + code)) return d ? T('err.' + code, d) : T('err.' + code, '').replace(/[\s:]+$/, '');
  if (hasT('link.cause.' + code)) return T('link.cause.' + code);
  return T('err.generic.plain');
}

function failed(a) {
  if (!a || !a.error) return false;
  toast(a.cause && hasT('link.cause.' + a.cause) ? T('link.cause.' + a.cause) : sayError(a.error, a.message), 4200);
  return true;
}

function ask(slot) {
  // ask(slot, args..., fn): the answer parsed, failures said once here
  const args = Array.prototype.slice.call(arguments, 1);
  const fn = typeof args[args.length - 1] === 'function' ? args.pop() : null;
  beJson.apply(null, [slot].concat(args, [function (a) {
    if (failed(a)) { if (fn) fn(null, a); return; }
    if (fn) fn(a);
  }]));
}

// ── the pill, the toast, the wait ──────────────────────────────────────────
let toastTimer = null;
function toast(text, ms) {
  // last guard: a raw line (stack, exception, log) never reaches the screen
  if (/[|]|exception|traceback|java\.|android\.|\bat [a-z]+\./i.test(String(text || ''))) text = T('err.generic.plain');
  const el = $('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function () { el.classList.remove('show'); }, ms || 2600);
}

function setStatus(kind, text) {
  $('sd').className = 'sd' + (kind ? ' ' + kind : '');
  $('st').textContent = text;
}

let PHONE = { ok: false, cause: 'unknown', link: 'usb' };
let working = 0;
// The pill says whether the phone is there; the button beside it says the link.
// A Wi-Fi link coming back pulses, as work does.
function showPhone() {
  if (typeof HOME !== 'undefined') HOME.phone();
  if (working) return;
  if (PHONE.ok) setStatus('ok', T('pill.phone.ok'));
  else setStatus(PHONE.state === 'reconnecting' || PHONE.state === 'connecting' ? 'busy' : 'warn',
                 T('pill.phone.' + (hasT('pill.phone.' + PHONE.cause) ? PHONE.cause : 'none')));
}
function work(on, text) {
  working = Math.max(0, working + (on ? 1 : -1));
  if (working) setStatus('busy', text || T('pill.working'));
  else showPhone();
}

function busy(on, text) {
  $('busyText').textContent = text || '';
  $('busy').classList.toggle('show', !!on);
}

// A button that does something that cannot be taken back asks for a second
// press: the first one writes what it will do on the button, for four seconds.
function armed(btn, text) {
  if (btn._armed) {
    clearTimeout(btn._armed);
    btn._armed = null;
    btn.classList.remove('on');
    const l = btn.querySelector('.btn-label');
    if (l) l.textContent = btn._word;
    return true;
  }
  const l = btn.querySelector('.btn-label');
  btn._word = l ? l.textContent : '';
  if (l) l.textContent = text;
  btn.classList.add('on');
  btn._armed = setTimeout(function () {
    btn._armed = null;
    btn.classList.remove('on');
    if (l) l.textContent = btn._word;
  }, 4000);
  return false;
}

// ── home, the rail, the views, the panels ──────────────────────────────────
function showView(name) {
  document.querySelectorAll('.view').forEach(function (v) { v.classList.toggle('show', v.dataset.view === name); });
}
function showPanel(name) {
  document.querySelectorAll('.rp-page').forEach(function (p) { p.classList.toggle('show', p.dataset.panel === name); });
}

// Home is the start (Federico, 15/10): the app opens there and the mark on the
// band always leads back to it. The way back is SlantUI's: the mark becomes a
// button, raised and pressable on every page, home too.
function goHome() { go('home'); }
function showHome() {
  setBrandAction(goHome, T('win.home.hint'));
}

function go(p) {
  if (p !== 'home' && PAGES.indexOf(p) < 0) p = 'home';
  if (page === 'live' && p !== 'live') LIVE.leave();
  page = p;
  document.querySelector('.app').classList.toggle('at-home', p === 'home');
  document.querySelectorAll('#rail .tab').forEach(function (t) {
    t.classList.toggle('active', t.dataset.page === p);
  });
  document.querySelectorAll('.tool-group').forEach(function (g) { g.classList.toggle('hidden', g.dataset.for !== p); });
  $('where').textContent = '';
  $('hint').textContent = '';
  showHome();
  if (p === 'home') { showView('home'); HOME.enter(); return; }
  be('set_page', p);
  if (p === 'calibration') { showView('calibration'); showPanel('calibration'); CAL.enter(); return; }
  if (p === 'live') { showView('live'); showPanel('live'); LIVE.enter(); return; }
  if (p === 'trials') { TRIALS.enter(); return; }
  showView('soon');
  showPanel('soon');
  $('soonTitle').textContent = T('page.' + p + '.title');
  $('soonPanelTitle').textContent = T('page.' + p + '.title');
  $('soonText').textContent = T('soon.' + p);
}

function wireRail() {
  document.querySelectorAll('#rail .tab').forEach(function (t) {
    t.onclick = function () { go(t.dataset.page); };
    t.title = T('page.' + t.dataset.page + '.sub');
  });
}

// Esc closes nothing here, so it is free; a file dropped on the window must not
// make the browser navigate away from the app.
window.addEventListener('dragover', function (e) { e.preventDefault(); });
window.addEventListener('drop', function (e) { e.preventDefault(); });
