// ImpactGuide · the words. Loads first: every other part asks it for a string.
// A plain script, not a module: the parts share one global scope.
//
// ui/strings/en.json is the reference, ui/strings/it.json its translation. A key
// says where the string is (`cal.board.action`), not what it says. A key missing
// in a language falls back to English, and a key missing everywhere shows itself.
// {0}, {1}... are the arguments of T().

const STRINGS = {};
let LANG = 'en';

function loadLang(lang) {
  if (STRINGS[lang]) return true;
  try {
    const x = new XMLHttpRequest();
    x.open('GET', 'ui/strings/' + lang + '.json', false);  // a few kB beside the page, read once
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

function T(key) {
  let s = STRINGS[LANG] ? STRINGS[LANG][key] : undefined;
  if (s === undefined) s = STRINGS.en ? STRINGS.en[key] : undefined;
  if (s === undefined) return key;
  const a = arguments;
  return String(s).replace(/\{(\d+)\}/g, function (m, i) {
    const v = a[(+i) + 1];
    return v === undefined ? m : String(v);
  });
}

// A number as the language writes it, with n decimals.
function cifre(v, n) {
  const x = Number(v);
  if (!isFinite(x)) return String(v);
  return x.toLocaleString(LANG === 'it' ? 'it-IT' : 'en-GB',
                          { minimumFractionDigits: n || 0, maximumFractionDigits: n || 0, useGrouping: false });
}

// A time stamp of the log (ISO 8601) as hours and minutes, with the day if it is not today.
function quando(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  const today = new Date().toDateString() === d.toDateString();
  const hm = d.toLocaleTimeString(LANG === 'it' ? 'it-IT' : 'en-GB', { hour: '2-digit', minute: '2-digit' });
  return today ? hm : d.toLocaleDateString(LANG === 'it' ? 'it-IT' : 'en-GB', { day: 'numeric', month: 'short' }) + ' ' + hm;
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
