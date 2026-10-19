// ImpactGuide · the words. Every other part asks them for a string.
// A plain script, not a module: the parts share one global scope.
//
// ui/strings/en.json is the reference; it, fr, es, de and zh are its translations,
// with the same keys and the same {0}, {1}... A key says where the string is
// (`cal.board.action`), not what it says. A key missing in a language falls back
// to English; a key missing everywhere gives an empty string (and a warning in the
// console), never the key itself on screen. {0}, {1}... are the arguments of T().
'use strict';

const STRINGS = {};
let LANG = 'en';
// How each language writes numbers and dates.
const LOCALES = { en: 'en-GB', it: 'it-IT', fr: 'fr-FR', es: 'es-ES', de: 'de-DE', zh: 'zh-CN' };
const MISSING_SAID = {};     // the keys already reported missing, said once each
function locale() { return LOCALES[LANG] || 'en-GB'; }

/** Reads ui/strings/<lang>.json once (a few kB beside the page). */
function loadLang(lang) {
  if (STRINGS[lang]) return true;
  try {
    const x = new XMLHttpRequest();
    x.open('GET', 'ui/strings/' + lang + '.json', false);
    x.send();
    STRINGS[lang] = JSON.parse(x.responseText);
    return true;
  } catch (e) {
    console.warn('ui/strings/' + lang + '.json could not be read: ' + e);
    return false;
  }
}

function hasT(key) {
  return (STRINGS[LANG] && STRINGS[LANG][key] !== undefined) || (STRINGS.en && STRINGS.en[key] !== undefined);
}

/** The string of a key in the language in use, {0}, {1}... replaced by the arguments that follow the key. */
function T(key) {
  let s = STRINGS[LANG] ? STRINGS[LANG][key] : undefined;
  if (s === undefined) s = STRINGS.en ? STRINGS.en[key] : undefined;
  if (s === undefined) {
    if (!MISSING_SAID[key]) { MISSING_SAID[key] = true; console.warn('no string for ' + key); }
    return '';
  }
  const a = arguments;
  return String(s).replace(/\{(\d+)\}/g, function (m, i) {
    const v = a[(+i) + 1];
    return v === undefined ? m : String(v);
  });
}

/**
 * The string of a code under a prefix (`link.state.` + `ready`), or of the fallback code when the code has none:
 * a code the backend learns later never reaches the screen as a key. Arguments after the fallback go to T().
 */
function Tcode(prefix, code, fallback) {
  const key = prefix + (code !== undefined && code !== null && code !== '' && hasT(prefix + code) ? code : fallback);
  return T.apply(null, [key].concat(Array.prototype.slice.call(arguments, 3)));
}

function setLang(lang) {
  if (!loadLang(lang)) return false;
  LANG = lang;
  document.documentElement.lang = lang;
  applyStrings(document);
  return true;
}

// The markup carries keys; this writes their words.
function applyStrings(root) {
  root.querySelectorAll('[data-t]').forEach(function (el) { el.textContent = T(el.getAttribute('data-t')); });
  root.querySelectorAll('[data-t-title]').forEach(function (el) { el.title = T(el.getAttribute('data-t-title')); });
  root.querySelectorAll('[data-t-ph]').forEach(function (el) { el.placeholder = T(el.getAttribute('data-t-ph')); });
}

loadLang('en');
