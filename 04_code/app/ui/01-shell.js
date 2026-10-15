// ImpactGuide · the shell: the rail, the views on the stage, the panels, the
// status pill, the toast and the wait. A plain script sharing one global scope.

// The pages of the rail and whether they are built. A page not built yet opens
// and says which part of the work will bring it.
const PAGES = ['calibration', 'live', 'trials', 'scans', 'analysis', 'charts', 'dataset'];
const BUILT = { calibration: true, trials: true };

// The signs this application adds to SlantUI's set, on the same 24 grid.
ICONS.camera = { s: '<path d="M4 8.5A1.5 1.5 0 0 1 5.5 7h2.3l1.4-2h5.6l1.4 2h2.3A1.5 1.5 0 0 1 20 8.5v9'
  + 'a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5z"/><circle cx="12" cy="12.8" r="3.4"/>' };

let page = 'trials';
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

// ── the answers of the backend ─────────────────────────────────────────────
// Every slot answers JSON; a failure carries `error`, a cause the page names.
function failed(a) {
  if (!a || !a.error) return false;
  const key = 'err.' + a.error;
  toast(hasT(key) ? T(key, a.message || '') : T('err.generic', a.error + (a.message ? ': ' + a.message : '')), 4200);
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

let PHONE = { ok: false, cause: 'unknown' };
let working = 0;
function showPhone() {
  if (working) return;
  if (PHONE.ok) setStatus('ok', T('pill.phone.ok'));
  else setStatus('warn', T('pill.phone.' + (hasT('pill.phone.' + PHONE.cause) ? PHONE.cause : 'none')));
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

// ── the rail, the views, the panels ────────────────────────────────────────
function showView(name) {
  document.querySelectorAll('.view').forEach(function (v) { v.classList.toggle('show', v.dataset.view === name); });
}
function showPanel(name) {
  document.querySelectorAll('.rp-page').forEach(function (p) { p.classList.toggle('show', p.dataset.panel === name); });
}

function go(p) {
  if (PAGES.indexOf(p) < 0) p = 'trials';
  page = p;
  document.querySelectorAll('#rail .tab').forEach(function (t) {
    t.classList.toggle('active', t.dataset.page === p);
  });
  document.querySelectorAll('.tool-group').forEach(function (g) { g.classList.toggle('hidden', g.dataset.for !== p); });
  $('where').textContent = '';
  be('set_page', p);
  if (p === 'calibration') { showView('calibration'); showPanel('calibration'); $('hint').textContent = T('cal.hint'); CAL.enter(); return; }
  if (p === 'trials') { TRIALS.enter(); return; }
  showView('soon');
  showPanel('soon');
  $('hint').textContent = '';
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
