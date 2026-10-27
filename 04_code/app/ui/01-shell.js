// ImpactGuide · the shell: home and the modules, the rail of the module open,
// the views on the stage, the panels, the Advanced parts, the sentences of a
// failure, the status pill, the toast and the wait. A plain script sharing one
// global scope.
//
// The app opens at home, four large choices and no rail; each choice is a
// MODULE that opens its own view, with its own steps on the rail and a panel
// for each step; the mark on the band always leads home. What a module is, is
// said once, here (MODULES): its steps in the order of the work and the step it
// opens at.
'use strict';

const MODULES = {
  calibration: { steps: ['board', 'photos', 'result'], start: 'board' },
  live: { steps: ['board', 'view'], start: 'view' },
  trials: { steps: ['guide', 'log', 'sequence', 'config', 'holders'], start: 'guide' },
  results: { steps: ['scans', 'analysis', 'charts', 'dataset'], start: 'scans' },
};
// The word of each step on the rail (its key in ui/strings).
const STEP_WORD = {
  calibration: { board: 'cal.step.board', photos: 'cal.step.photos', result: 'cal.step.result' },
  live: { board: 'live.step.board', view: 'live.step.view' },
  trials: { guide: 'trials.view.guide', log: 'trials.view.log', sequence: 'trials.view.sequence',
            config: 'trials.view.config', holders: 'hold.step' },
  results: { scans: 'page.scans.title', analysis: 'page.analysis.title', charts: 'page.charts.title',
             dataset: 'page.dataset.title' },
};

const TOAST_MS = 2600;          // a toast that says something went as asked
const TOAST_SAID_MS = 3500;     // one that says what happened by itself (a trial closed, the phone on Wi-Fi)
const TOAST_ERROR_MS = 4200;    // one that says why something did not work
const TOAST_LONG_MS = 6000;     // one with something to do (the live view did not start, a pairing failed)
const ARMED_MS = 4000;          // how long a button that cannot be taken back waits for its second press

// The signs this application adds to SlantUI's set, on the same 24 grid.
ICONS.camera = { s: '<path d="M4 8.5A1.5 1.5 0 0 1 5.5 7h2.3l1.4-2h5.6l1.4 2h2.3A1.5 1.5 0 0 1 20 8.5v9'
  + 'a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5z"/><circle cx="12" cy="12.8" r="3.4"/>' };
// The four signs of home, on SlantUI's 24 grid and stroke, each drawn whole inside the grid with room for its stroke
// (nothing cut at the edge) and its ink centred on it (SlantUI centres a sign by its box: no shift). Every board the
// app draws as a sign is a sheet lying across, its long side horizontal, with squares of 2.4 (ICONS.screen too).
// Calibrate the phone: a 6 x 4 checkerboard in the outline of its sheet.
ICONS.board = { s: '<rect x="2.7" y="5.1" width="18.6" height="13.8" rx="1.5"/>'
  + '<path d="M4.8 7.2h2.4v2.4h-2.4zM9.6 7.2h2.4v2.4h-2.4zM14.4 7.2h2.4v2.4h-2.4zM7.2 9.6h2.4v2.4h-2.4zM12 9.6h2.4v2.4'
  + 'h-2.4zM16.8 9.6h2.4v2.4h-2.4zM4.8 12h2.4v2.4h-2.4zM9.6 12h2.4v2.4h-2.4zM14.4 12h2.4v2.4h-2.4zM7.2 14.4h2.4v2.4'
  + 'h-2.4zM12 14.4h2.4v2.4h-2.4zM16.8 14.4h2.4v2.4h-2.4z" fill="currentColor" stroke="none"/>' };
// Run the trials: the round of the trials, an arc of three quarters with the dots of the quarter still to come,
// around the sign that starts the next one.
ICONS.trials = { s: '<path d="M12 3.5A8.5 8.5 0 1 1 3.5 12"/>'
  + '<circle cx="4.119" cy="8.817" r="0.9" fill="currentColor" stroke="none"/>'
  + '<circle cx="5.99" cy="5.99" r="0.9" fill="currentColor" stroke="none"/>'
  + '<circle cx="8.817" cy="4.119" r="0.9" fill="currentColor" stroke="none"/>'
  + '<path d="M10.25 8.75L15.5 12L10.25 15.25z" fill="currentColor"/>' };
// Scans and results: a mesh as a scan gives it, a polyhedron seen along its axis: its outline, the face in front,
// and the edges behind it, half seen.
ICONS.results = { s: '<path d="M12 6.438L4.206 7.5M12 6.438L12 3M12 6.438L19.794 7.5M16.817 14.781L19.794 7.5'
  + 'M16.817 14.781L19.794 16.5M16.817 14.781L12 21M7.183 14.781L12 21M7.183 14.781L4.206 16.5M7.183 14.781L4.206 7.5"'
  + ' opacity="0.5"/><path d="M12 3 19.794 7.5 19.794 16.5 12 21 4.206 16.5 4.206 7.5z"/>'
  + '<path d="M12 6.438 16.817 14.781 7.183 14.781z"/>' };
// The two names a trial is read by in every row and cell, as signs (SlantUI's .ilbl, the name in the tooltip):
// the hole, an angle opened from the rail, and the direction, an arrow that points.
ICONS.hole = { s: '<path d="M4 19h16M4 19 15.5 7.5"/><path d="M11.5 19a7.5 7.5 0 0 0-2.2-5.3"/>' };
ICONS.direction = { s: '<path d="M5.5 18.5 18.5 5.5M9.5 5.5h9v9"/>' };

// A value with a sign for its name: the sign in front of it, the name its tooltip (SlantUI's .ilbl).
function signed(sign, text, name) {
  const s = E('span', 'ilbl');
  s.innerHTML = icon(sign);       // SlantUI's own sign, made by the app: no data in it
  s.append(document.createTextNode(text));
  s.title = name;
  return s;
}

let page = 'home';          // the module open, or home
let trialsView = 'guide';   // the step open in Trials (ui/20-trials.js)
const STEP = { calibration: 'board', live: 'view', results: 'scans' };   // and in the others

// The signs in the markup: a [data-icon] span becomes the <svg> itself, so the
// button centres sign and word by their ink; a window button gets it inside.
function placeIcons(root) {
  (root || document).querySelectorAll('[data-icon]').forEach(function (el) {
    const name = el.getAttribute('data-icon');
    if (el.tagName === 'SPAN') {
      const holder = document.createElement('span');
      holder.innerHTML = icon(name);      // SlantUI's own sign, made by the app: no data in it
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

// ── the sentences of a failure ─────────────────────────────────────────────
// A failure from the backend carries a code (`error`; `cause` for the link to the phone), maybe its values
// (`args`) and a short detail (`message`). The operator always reads a sentence of the app in his language: the
// code's own (err.*, link.cause.*, or a part's own prefix such as board.err.*), else the generic one. Only a
// value of a known kind reaches the screen beside it (a file or trial name, a field, a step, a camera mode, a count,
// a size, a number), read from its values (`args`, or `params` as the library names them) or, for the first one,
// from `message` when that is a single plain word; any other text (an exception, adb's output, a path, a sentence)
// stays in the console.
// code: the values its sentence shows, in the order of {0}, {1}: [kind, the names the value may have]
const ERROR_DETAIL = {
  missing_file: [['names', 'files', 'file']], trial_open: [['name', 'trial']], exists: [['name', 'file', 'trial']],
  not_found: [['name', 'file']], bad_registration: [['name', 'file']], no_intrinsics: [['mode', 'mode']],
  bad_value: [['field', 'field']], bad_field: [['field', 'field']], ruler_bad: [['number', 'value', 'mm']],
  config_corrupt: [['name', 'file']], unknown_trial: [['name', 'trial']], unknown_step: [['step', 'step']],
  cannot_close: [['problems', 'problems']], not_open: [['name', 'trial']], no_step: [['step', 'step']],
  trial_changed: [['name', 'trial']],
  few_photos: [['count', 'n'], ['count', 'need']], intrinsics_size: [['size', 'got'], ['size', 'want']],
};
const PLAIN_NAME = /^\w[\w .,()+-]{0,63}$/;     // a name the backend gives as a value
const PLAIN_WORD = /^\w[\w.()+-]{0,63}$/;       // a name read from `message`: one word, never a sentence
const DETAIL_LIST_MAX = 3;                       // names or problems shown in a list, at most

/** One value of a failure in the language's words, or '' when it is not of its kind. */
function detailValue(kind, v, fromMessage) {
  if (v === undefined || v === null || v === '') return '';
  if (kind === 'names' || kind === 'problems') {
    const all = Array.isArray(v) ? v : kind === 'names' ? String(v).split(/,\s*/) : [];
    const one = function (x) { return kind === 'names' ? detailValue('name', x, fromMessage) : TRIALS.checkProblem(x); };
    const out = all.slice(0, DETAIL_LIST_MAX).map(one);
    if (!out.length || out.some(function (x) { return !x; })) return '';
    return out.join(kind === 'names' ? ', ' : '; ') + (all.length > DETAIL_LIST_MAX ? ' …' : '');
  }
  if (kind === 'size') return Array.isArray(v) && v.length === 2 && v.every(isFinite) ? fmtNum(v[0], 0) + ' × ' + fmtNum(v[1], 0) : '';
  const s = String(v).trim();
  if (kind === 'name') { const b = s.split(/[\\/]/).pop(); return (fromMessage ? PLAIN_WORD : PLAIN_NAME).test(b) ? b : ''; }
  if (kind === 'mode') return hasT('cal.mode.' + s) ? T('cal.mode.' + s) : '';
  if (kind === 'field') return hasT('col.' + s) ? T('col.' + s) : '';
  if (kind === 'step') return /^\w+$/.test(s) && hasT('step.' + TRIALS.base(s) + '.title') ? TRIALS.stepTitle({ key: s }) : '';
  if (kind === 'count') return /^\d+$/.test(s) ? fmtNum(s, 0) : '';
  const m = /^-?\d+(?:[.,](\d+))?$/.exec(s);
  return m ? fmtNum(s.replace(',', '.'), m[1] ? m[1].length : 0) : '';
}

/** The values a failure shows beside its sentence, in the order of its placeholders ('' for one it lacks). */
function errorDetails(code, args, message) {
  return (ERROR_DETAIL[code] || []).map(function (spec, i) {
    const k = args && typeof args === 'object'
      ? spec.slice(1).find(function (x) { return args[x] !== undefined && args[x] !== null; }) : undefined;
    if (k) return detailValue(spec[0], args[k], false);
    return i === 0 ? detailValue(spec[0], message, true) : '';
  });
}

/**
 * A sentence with its {0}, {1} filled; a value it lacks is left out with its colon when it ends the sentence, and
 * the generic sentence is said when it is needed in the middle.
 */
function fillDetail(key, values) {
  const marks = ['\u0000', '\u0001', '\u0002'];
  let s = T.apply(null, [key].concat(marks));
  for (let i = 0; i < marks.length; i++) {
    const at = s.indexOf(marks[i]);
    if (at < 0) continue;
    if (values[i]) { s = s.split(marks[i]).join(values[i]); continue; }
    if (s.slice(at + 1).trim()) return T('err.generic.plain');
    s = s.slice(0, at).replace(/[\s:：,，(（]+$/, '');
  }
  return s;
}

/** The sentence of a code: the part's own (prefix), the app's (err.), the link's (link.cause.), else the generic. */
function sayError(code, args, message, prefix) {
  if (message) console.warn('failure', code, message);
  const key = [prefix ? prefix + code : '', 'err.' + code, 'link.cause.' + code]
    .find(function (k) { return k && hasT(k); });
  return key ? fillDetail(key, errorDetails(code, args, message)) : T('err.generic.plain');
}

/** The sentence of a failed answer or of an event that says a failure. */
function errorText(a, prefix) {
  if (!a) return T('err.generic.plain');
  if (a.cause && hasT('link.cause.' + a.cause)) return T('link.cause.' + a.cause);
  return sayError(String(a.error || a.code || a.cause || 'internal'), a.args || a.params, a.message, prefix);
}

/** Says a failed answer in a toast; false when the answer is not a failure. */
function failed(a) {
  if (!a || !a.error) return false;
  toast(errorText(a), TOAST_ERROR_MS, 'error');
  return true;
}

// ── the pill, the toast, the wait ──────────────────────────────────────────
let toastTimer = null;
// `sound`: the SlantUI voice said with it ('error'), or none; a press makes its own sound (data-sound).
function toast(text, ms, sound) {
  // last guard: a raw line (stack, exception, log) never reaches the screen
  if (/[|]|exception|traceback|java\.|android\.|\bat [a-z]+\./i.test(String(text || ''))) text = T('err.generic.plain');
  const el = $('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function () { el.classList.remove('show'); }, ms || TOAST_MS);
  if (sound) playSound(sound);
}

function setStatus(kind, text) {
  $('sd').className = 'sd' + (kind ? ' ' + kind : '');
  $('st').textContent = text;
  $('pill').title = text;
}

let PHONE = { ok: false, cause: 'unknown', link: 'usb' };

// What the app is doing for the operator, by what started it ('cal_photo', 'calib', 'live'...), each with the key
// of its word: the pill says the newest one, in the language in use. Each is ended by the event that closes it;
// ending one that is not on is harmless.
const WORK = new Map();
function workOn(key, word) {
  WORK.delete(key);
  WORK.set(key, word || 'pill.working');
  showWork();
}
function workOff(key) {
  if (WORK.delete(key)) showWork();
}
/** Everything ended at once: the backend said that something failed without saying which work. */
function workClear() {
  WORK.clear();
  showWork();
}
function showWork() {
  if (!WORK.size) { showPhone(); return; }
  setStatus('busy', T(Array.from(WORK.values()).pop()));
}

// The pill says whether the phone is there; the button beside it says how.
function showPhone() {
  if (typeof HOME !== 'undefined') HOME.phone();
  if (typeof LINK !== 'undefined') LINK.btn();
  if (WORK.size) return;
  if (PHONE.ok) setStatus('ok', T('pill.phone.ok'));
  else setStatus(PHONE.state === 'reconnecting' || PHONE.state === 'connecting' ? 'busy' : 'warn',
                 Tcode('pill.phone.', PHONE.cause, 'none'));
}

function busy(on, text) {
  $('busyText').textContent = text || '';
  $('busy').classList.toggle('show', !!on);
}

// A button that does something that cannot be taken back asks for a second
// press: the first one writes what it will do on the button, for a few seconds.
function armed(btn, text) {
  const l = btn.querySelector('.btn-label');
  if (btn._armed) {
    clearTimeout(btn._armed);
    btn._armed = null;
    btn.classList.remove('on');
    if (l) l.textContent = btn._word;
    return true;
  }
  btn._word = l ? l.textContent : '';
  if (l) l.textContent = text;
  btn.classList.add('on');
  btn._armed = setTimeout(function () {
    btn._armed = null;
    btn.classList.remove('on');
    if (l) l.textContent = btn._word;
  }, ARMED_MS);
  return false;
}

// ── the live stream: whose it is ───────────────────────────────────────────
// One stream serves either the Live AR view or a calibration of the streaming mode. Its events say whose it is
// (`purpose`); an older backend leaves it out, and then the owner is the one the page last asked for (or the one
// the last calibration state names), so that one part never stops or counts the other's stream.
let liveOwner = 'live';
function setLiveOwner(purpose) { liveOwner = purpose === 'calibration' ? 'calibration' : 'live'; }
function livePurpose(e) {
  if (e.purpose === 'calibration' || e.purpose === 'live') return e.purpose;
  return liveOwner;
}

// The stream's codes (live_ar.ERRORS) and the sentence each one says; several share one.
const LIVE_ERRORS = {
  no_adb: 'link.cause.no_adb', camera_busy: 'live.err.busy', adb_failed: 'live.err.no_phone', no_phone: 'live.err.no_phone',
  link_lost: 'live.err.link_lost', start_failed: 'live.err.start_failed', no_stream: 'live.err.start_failed',
  no_frame: 'live.err.start_failed', stopped: 'live.err.stream_failed', server_exited: 'live.err.stream_failed',
  stream_closed: 'live.err.stream_failed', stream_failed: 'live.err.stream_failed', stream_stopped: 'live.err.stream_failed',
  bad_stream: 'live.err.stream_failed', stalled: 'live.err.stalled', behind: 'live.err.stalled',
  frames_failing: 'live.err.frames_failing', replay_unreadable: 'live.err.replay_unreadable', bad_board: 'live.err.bad_board',
  bad_value: 'live.err.bad_value', bad_registration: 'err.bad_registration', closed: 'live.err.closed',
  intrinsics_bad: 'err.intrinsics_bad',
};

/** Why the live stream did not start or stopped, in the app's words: the link's cause, else the stream's code. */
function liveErrorText(x) {
  if (x.cause && hasT('link.cause.' + x.cause)) return T('link.cause.' + x.cause);
  const code = typeof x.error === 'string' ? x.error : '';
  if (LIVE_ERRORS[code]) return fillDetail(LIVE_ERRORS[code], []);
  // an older backend sends the stream's own text (often a Java stack): it only chooses the sentence
  if (/camera|in use|disconnect/i.test(String(x.message || code))) return T('live.err.busy');
  return T(PHONE.link === 'wifi' ? 'live.err.wifi' : 'live.err.generic');
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
// Home is the start: the app opens there and the mark on the band always leads
// back to it, from every module and step. The way back is SlantUI's: the mark
// becomes a button, raised and pressable.
function goHome() { go('home'); }
function showHome() {
  const b = setBrandAction(goHome, T('win.home.hint'));
  if (b) b.setAttribute('data-sound', page === 'home' ? '' : 'back');      // SlantUI's voice, on the press
}

function stepOf(p) { return p === 'trials' ? trialsView : STEP[p]; }
// The voice of a way to step s of module p (SlantUI's data-sound, played on the press): forward to a later step,
// back to an earlier one, none to the step open.
function stepSound(p, s) {
  const steps = MODULES[p] ? MODULES[p].steps : [], now = steps.indexOf(stepOf(p)), to = steps.indexOf(s);
  return to < 0 || to === now ? '' : to > now ? 'forward' : 'back';
}
function stepWord(p, s) { return T(STEP_WORD[p][s]); }

// The rail: the steps of the module open, numbered, the open one lit; nothing at home.
function rail() {
  const host = $('rail');
  const focused = host.contains(document.activeElement);    // a step chosen with the keyboard keeps the focus
  host.replaceChildren();
  if (page === 'home' || !MODULES[page]) return;
  const now = stepOf(page);
  MODULES[page].steps.forEach(function (s, i) {
    const t = E('div', 'tab' + (s === now ? ' active' : ''));
    t.dataset.step = s;
    t.title = stepWord(page, s);
    t.tabIndex = 0;
    t.setAttribute('role', 'button');
    t.setAttribute('data-sound', stepSound(page, s));
    t.append(E('div', 'tn', i + 1), E('span', null, stepWord(page, s)));
    onActivate(t, function () { go(page, s); });
    host.append(t);
    if (focused && s === now) t.focus();
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
  if (was === 'calibration' && wasStep === 'photos' && !(p === 'calibration' && s === 'photos')) CAL.leave();
  // the board on this monitor is drawn only by the steps that show it: they place it again
  if (typeof BOARDVIEW !== 'undefined') BOARDVIEW.place(null);
  page = p;
  // the twin made ready in the modules it opens from, let go a while after the others (ui/18-twin.js)
  if (typeof TWIN !== 'undefined' && TWIN.module) TWIN.module(p);
  if (typeof LINK !== 'undefined' && LINK.open) LINK.hide();
  document.querySelector('.app').classList.toggle('at-home', p === 'home');
  document.querySelectorAll('.tool-group').forEach(function (g) { g.classList.toggle('hidden', g.dataset.for !== p + '-' + s); });
  $('where').textContent = '';
  $('hint').textContent = '';
  showHome();
  rail();
  document.querySelectorAll('.rp-page [data-step]').forEach(function (b) { b.setAttribute('data-sound', stepSound(page, b.dataset.step)); });
  if (p === 'home') { showView('home'); HOME.enter(); return; }
  be('set_page', p);
  if (p === 'calibration') { CAL.enter(s); return; }
  if (p === 'live') { LIVE.enter(s); return; }
  if (p === 'trials' && s === 'holders') { HOLDERS.enter(); return; }     // the holders on their own (28-holders.js)
  if (p === 'trials') { TRIALS.enter(); return; }
  if (p === 'results' && s === 'scans') { SCANS.enter(); return; }     // the scans (22-scans.js; in 3D: the twin)
  if (p === 'results' && s === 'analysis') { ANALYSIS.enter(); return; }   // what the trials say (24-analysis.js)
  if (p === 'results' && s === 'charts') { CHARTS.enter(); return; }       // the figures (26-charts.js)
  if (p === 'results' && s === 'dataset') { DATASET.enter(); return; }     // the dataset and its package (28-dataset.js)
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
