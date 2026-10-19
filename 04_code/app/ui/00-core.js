// ImpactGuide · the core every part of the page shares: elements, numbers and times as the language writes them,
// a key press that works as a click, and the two ways of calling the backend: ask() for one answer (a failure is
// said once, in the app's words) and poll() for a state read again and again (a failure is never toasted).
// A plain script sharing one global scope with the others; the words (T, locale) come from 00-strings.js, the
// sentences of a failure (failed, errorText) from 01-shell.js, both used only when a call runs.
'use strict';

/** How long a polled call may stay unanswered before it is asked again, in periods of its poll. */
const POLL_STALE_PERIODS = 10;
/** After failures a poll asks every 2, 4, 8... periods, at most this many. */
const POLL_BACKOFF_MAX = 8;

function $(id) { return document.getElementById(id); }

/** An element with a class and a text; never markup from data. */
function E(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined && text !== null) el.textContent = String(text);
  return el;
}

/** A row of a panel: a label and its value. */
function piRow(label, value, cls) {
  const r = E('div', 'pi-row' + (cls ? ' ' + cls : ''));
  r.append(E('span', 'pi-l', label), E('span', null, value));
  return r;
}

function chip(text, kind) { return E('span', 'chip' + (kind ? ' chip-' + kind : ''), text); }

/**
 * Makes an element that is not a button act as one: a click, Enter or Space runs fn. Handlers are assigned, not
 * added, so wiring twice (the language changed) changes nothing.
 */
function onActivate(el, fn) {
  el.onclick = fn;
  el.onkeydown = function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn.call(el, e); }
  };
}

/** A list item that can be chosen with the keyboard as with the mouse; `off` takes it out of the way. */
function itemOff(el, off) {
  el.classList.toggle('off', !!off);
  el.tabIndex = off ? -1 : 0;
  el.setAttribute('aria-disabled', off ? 'true' : 'false');
}

// ── numbers and times as the language writes them ──────────────────────────
/** A number with n decimals as the language writes it; '-' when there is no number. */
function fmtNum(v, n) {
  const x = v === '' || v === null || v === undefined ? NaN : Number(v);
  if (!isFinite(x)) return '-';
  return x.toLocaleString(locale(), { minimumFractionDigits: n || 0, maximumFractionDigits: n || 0, useGrouping: false });
}

/** A number and its unit, never apart on two lines: 20,00 mm. */
function withUnit(v, n, unit) { return fmtNum(v, n) + ' ' + unit; }

/** A time stamp of the log (ISO 8601) as hours and minutes, with the day if it is not today. */
function fmtTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d)) return String(iso);
  const today = new Date().toDateString() === d.toDateString();
  const hm = d.toLocaleTimeString(locale(), { hour: '2-digit', minute: '2-digit' });
  return today ? hm : d.toLocaleDateString(locale(), { day: 'numeric', month: 'short' }) + ' ' + hm;
}

/** The day of a time stamp, short, with the year only when it is not this one. */
function fmtDay(iso) {
  const d = new Date(String(iso || '').slice(0, 10) + 'T12:00:00');
  if (isNaN(d)) return String(iso || '');
  const o = { day: 'numeric', month: 'short' };
  if (d.getFullYear() !== new Date().getFullYear()) o.year = 'numeric';
  return d.toLocaleDateString(locale(), o);
}

// ── the backend ─────────────────────────────────────────────────────────────
/** Whether the backend offers a slot (a newer slot may be missing in an older backend). */
function hasSlot(name) { return !!(Bridge.obj && typeof Bridge.obj[name] === 'function'); }

// The call itself: the answer parsed; an answer that is not JSON counts as a failure of the app.
function callSlot(slot, args, then) {
  beJson.apply(null, [slot].concat(args, [function (a) {
    if (a === null || typeof a !== 'object') then(null, { error: 'internal' });
    else if (a.error) then(null, a);
    else then(a);
  }]));
}

// ask(slot, args..., fn) and askQuiet(slot, args..., fn): the slot's arguments, then fn(answer) on success or
// fn(null, failure) on failure; failure is the backend's answer ({error, cause, args, message}).
function splitCall(rest) {
  const args = Array.prototype.slice.call(rest);
  const fn = typeof args[args.length - 1] === 'function' ? args.pop() : null;
  return { args: args, fn: fn };
}

/** One answer; a failure is said once, here, in the app's words (toast), then handed to fn as well. */
function ask(slot) {
  const c = splitCall(Array.prototype.slice.call(arguments, 1));
  callSlot(slot, c.args, function (a, bad) {
    if (bad) failed(bad);
    if (c.fn) c.fn(a, bad);
  });
}

/** One answer; a failure is handed to fn only, for a caller that says it its own way (or not at all). */
function askQuiet(slot) {
  const c = splitCall(Array.prototype.slice.call(arguments, 1));
  callSlot(slot, c.args, function (a, bad) { if (c.fn) c.fn(a, bad); });
}

/**
 * A state read every `ms`: fn(answer) or fn(null, failure, failures in a row), never a toast. It asks again only
 * once the last call has answered (or stayed unanswered for POLL_STALE_PERIODS), and less often after failures.
 * The first call is made at once. Returns the interval's id: clearInterval() stops it.
 */
function poll(slot, ms, fn) {
  const p = { busy: 0, fails: 0, skip: 0 };
  const tick = function () {
    if (p.busy && Date.now() - p.busy < ms * POLL_STALE_PERIODS) return;
    if (p.skip > 0) { p.skip--; return; }
    p.busy = Date.now();
    askQuiet(slot, function (a, bad) {
      p.busy = 0;
      if (bad) {
        p.fails++;
        p.skip = Math.min(Math.pow(2, p.fails), POLL_BACKOFF_MAX) - 1;
      } else {
        p.fails = 0;
        p.skip = 0;
      }
      fn(a, bad, p.fails);
    });
  };
  tick();
  return setInterval(tick, ms);
}
