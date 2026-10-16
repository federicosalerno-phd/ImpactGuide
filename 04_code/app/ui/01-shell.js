// ImpactGuide · the shell: home and the modules, the rail of the module open,
// the views on the stage, the panels, the Advanced parts, the status pill, the
// toast and the wait. A plain script sharing one global scope.
//
// As UniGuide (ui/19-ingresso.js, ui/20-boot.js): the app opens at home, four
// large choices and no rail; each choice is a MODULE that opens its own view,
// with its own steps on the rail and a panel for each step; the mark on the
// band always leads home. What a module is, is said once, here (MODULES): its
// steps in the order of the work and the step it opens at.

const MODULES = {
  calibration: { steps: ['board', 'photos', 'result'], start: 'board' },
  live: { steps: ['board', 'view'], start: 'view' },
  trials: { steps: ['guide', 'log', 'sequence', 'config'], start: 'guide' },
  results: { steps: ['scans', 'analysis', 'charts', 'dataset'], start: 'scans' },
};
const PAGES = Object.keys(MODULES);
// The word of each step on the rail (its key in ui/strings).
const STEP_WORD = {
  calibration: { board: 'cal.step.board', photos: 'cal.step.photos', result: 'cal.step.result' },
  live: { board: 'live.step.board', view: 'live.step.view' },
  trials: { guide: 'trials.view.guide', log: 'trials.view.log', sequence: 'trials.view.sequence',
            config: 'trials.view.config' },
  results: { scans: 'page.scans.title', analysis: 'page.analysis.title', charts: 'page.charts.title',
             dataset: 'page.dataset.title' },
};

// The signs this application adds to SlantUI's set, on the same 24 grid.
ICONS.camera = { s: '<path d="M4 8.5A1.5 1.5 0 0 1 5.5 7h2.3l1.4-2h5.6l1.4 2h2.3A1.5 1.5 0 0 1 20 8.5v9'
  + 'a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5z"/><circle cx="12" cy="12.8" r="3.4"/>' };
// The four signs of home, on SlantUI's 24 grid and stroke, each drawn so that its box is centred by construction
// (SlantUI centres a sign by its box: no shift, so the squares stay on whole pixels at the choices' 28 px, where one
// pixel is 6/7 of a unit). Calibrate the phone: a 4 x 4 checkerboard of 3 px squares in the outline of its sheet.
ICONS.board = { s: '<rect x="4" y="4" width="16" height="16" rx="1.5"/>'
  + '<path d="M6.857 6.857h2.571v2.571H6.857zM12 6.857h2.571v2.571H12zM9.429 9.429H12V12H9.429zM14.571 9.429h2.571V12'
  + 'h-2.571zM6.857 12h2.571v2.571H6.857zM12 12h2.571v2.571H12zM9.429 14.571H12v2.571H9.429zM14.571 14.571h2.571'
  + 'v2.571h-2.571z" fill="currentColor" stroke="none"/>' };
// Run the trials: the app's logo in line, as much of it as reads at 28 px: the support and its pin, the rod, the
// mass, the flash of the impact and the checkered marker.
ICONS.pendulum = { s: '<path d="M6.5 2.571h7"/><circle cx="10" cy="4.75" r="1.15" fill="currentColor"/>'
  + '<path d="M9.62 6.05 7.682 11.94"/><circle cx="6.73" cy="15.1" r="3.3"/><path d="M16.6 4.4 14.2 8.1h3.1l-2.4 3.7"/>'
  + '<rect x="12" y="12.857" width="8.571" height="8.571" rx="1"/>'
  + '<path d="M13.714 14.571h2.571v2.571h-2.571zM16.286 17.143h2.571v2.571h-2.571z" fill="currentColor" stroke="none"/>' };
// Scans and results: a sphere fitted to the points of a scan, its centre marked. The circle, of radius 7.8792, goes
// through the three points, whose centres are worked out on it (cos and sin of 225, 315 and 45 degrees) and fall on
// pixel centres at 28 px (6.5 px from the middle); the circle stops short of each point, so the points read as points.
ICONS.results = { s: '<path d="M14.9172 19.3193A7.8792 7.8792 0 0 1 4.6807 9.0828M9.0828 4.6807A7.8792 7.8792 0 0 1'
  + ' 14.9172 4.6807M19.3193 9.0828A7.8792 7.8792 0 0 1 19.3193 14.9172"/><path d="M12 10v4M10 12h4"/>'
  + '<circle cx="6.4286" cy="6.4286" r="2" fill="currentColor" stroke="none"/>'
  + '<circle cx="17.5714" cy="6.4286" r="2" fill="currentColor" stroke="none"/>'
  + '<circle cx="17.5714" cy="17.5714" r="2" fill="currentColor" stroke="none"/>' };

let page = 'home';          // the module open, or home
let trialsView = 'guide';   // the step open in Trials (ui/20-trials.js)
const STEP = { calibration: 'board', live: 'view', results: 'scans' };   // and in the others

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
// What is rarely needed sits in SlantUI's fold (.fold, initFolds in ui/90-boot.js),
// closed until its head is pressed; open or closed is kept while the app runs.
// From code: setFold(el, open). This closes every one (a module left).
function closeFolds(root) {
  (root || document).querySelectorAll('.fold.open').forEach(function (f) { setFold(f, false); });
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
  $('pill').title = text;
}

let PHONE = { ok: false, cause: 'unknown', link: 'usb' };
let working = 0;
// The pill says whether the phone is there; the button beside it says how.
function showPhone() {
  if (typeof HOME !== 'undefined') HOME.phone();
  if (typeof LINK !== 'undefined') LINK.btn();
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

// ── the views, the panels ──────────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll('.view').forEach(function (v) {
    const on = v.dataset.view === name;
    if (on && !v.classList.contains('show')) v.scrollTop = 0;     // a view comes back at its top
    v.classList.toggle('show', on);
  });
}
// The panel of the step. While the Phone panel is open it stays in its place,
// and the step's panel comes back when it closes (ui/40-phone.js).
let panelNow = '';
function showPanel(name) {
  panelNow = name;
  if (typeof LINK !== 'undefined' && LINK.open) return;
  document.querySelectorAll('.rp-page').forEach(function (p) { p.classList.toggle('show', p.dataset.panel === name); });
}

// ── home, the modules, the rail ────────────────────────────────────────────
// Home is the start (Federico, 15/10 and 16/10): the app opens there and the
// mark on the band always leads back to it, from every module and step. The
// way back is SlantUI's: the mark becomes a button, raised and pressable.
function goHome() { go('home'); }
function showHome() {
  setBrandAction(goHome, T('win.home.hint'));
}

function stepOf(p) { return p === 'trials' ? trialsView : STEP[p]; }
function stepWord(p, s) { return T(STEP_WORD[p][s]); }

// The rail: the steps of the module open, numbered, the open one lit; nothing at home.
function rail() {
  const host = $('rail');
  host.replaceChildren();
  if (page === 'home' || !MODULES[page]) return;
  const now = stepOf(page);
  MODULES[page].steps.forEach(function (s, i) {
    const t = E('div', 'tab' + (s === now ? ' active' : ''));
    t.dataset.step = s;
    t.title = stepWord(page, s);
    t.append(E('div', 'tn', i + 1), E('span', null, stepWord(page, s)));
    t.onclick = function () { go(page, s); };
    host.append(t);
  });
}

// A module, at a step (its own last one, or the one it opens at the first time).
function go(p, step) {
  if (p !== 'home' && !MODULES[p]) p = 'home';
  const was = page, wasStep = was !== 'home' && MODULES[was] ? stepOf(was) : '';
  if (MODULES[p] && step && MODULES[p].steps.indexOf(step) >= 0) {
    if (p === 'trials') trialsView = step; else STEP[p] = step;
  }
  const s = MODULES[p] ? stepOf(p) : '';
  if (was === 'live' && wasStep === 'view' && !(p === 'live' && s === 'view')) LIVE.leave();
  page = p;
  if (typeof LINK !== 'undefined' && LINK.open) LINK.hide();
  document.querySelector('.app').classList.toggle('at-home', p === 'home');
  document.querySelectorAll('.tool-group').forEach(function (g) { g.classList.toggle('hidden', g.dataset.for !== p + '-' + s); });
  $('where').textContent = '';
  $('hint').textContent = '';
  showHome();
  rail();
  if (p === 'home') { showView('home'); HOME.enter(); return; }
  be('set_page', p);
  if (p === 'calibration') { CAL.enter(s); return; }
  if (p === 'live') { LIVE.enter(s); return; }
  if (p === 'trials') { TRIALS.enter(); return; }
  showView('soon');
  showPanel('soon');
  $('soonTitle').textContent = T('page.' + s + '.title');
  $('soonPanelTitle').textContent = T('page.' + s + '.title');
  $('soonText').textContent = T('soon.' + s);
}

// The buttons of a panel that lead to another step of the same module (Next, Back).
function wireSteps() {
  document.querySelectorAll('.rp-page [data-step]').forEach(function (b) {
    b.onclick = function () { go(page, b.dataset.step); };
  });
}

// Esc closes nothing here, so it is free; a file dropped on the window must not
// make the browser navigate away from the app.
window.addEventListener('dragover', function (e) { e.preventDefault(); });
window.addEventListener('drop', function (e) { e.preventDefault(); });
