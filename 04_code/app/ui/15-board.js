// ImpactGuide · the calibration board: on a monitor (the default) or printed.
// The same controls sit on the Calibration page and on the Live AR page (ids
// starting with cal and lv); both pages hand this part the backend's board
// state with their own, and it draws both copies. In plain view: the kind, the
// square, what needs doing and Show board; the monitor and the ruler check sit
// in the page's Advanced part. The work is screen_board.py's and the backend's:
// the page shows, chooses and asks.

// The sign of a monitor, on SlantUI's 24 grid.
ICONS.screen = { s: '<rect x="3" y="4.5" width="18" height="12" rx="1.5"/><path d="M9 20h6M12 16.5V20"/>' };

const BOARD = {
  state: null,
  copies: ['cal', 'lv'],

  // a length as the language writes it: one decimal, or two for a square
  mm: function (v, n) { return cifre(v, n === undefined ? 1 : n); },

  // a monitor in the list: its name and size; the pixels and which one is the main one on hover
  monitorLabel: function (m, i) {
    const parts = [m.label || T('board.monitor.sec') + ' ' + (i + 1)];
    if (m.diag_in) parts.push(cifre(m.diag_in, 1) + '″');
    return parts.join(' · ');
  },
  monitorHint: function (m) {
    return m ? m.px[0] + ' × ' + m.px[1] + (m.primary ? ' · ' + T('board.monitor.main') : '') : '';
  },

  render: function (s) {
    if (!s) return;
    BOARD.state = s;
    BOARD.copies.forEach(function (p) { if ($(p + 'Bd')) BOARD.draw(p, s); });
  },

  draw: function (p, s) {
    const screen = s.kind === 'screen';
    document.querySelectorAll('#' + p + 'BdKinds .mitem').forEach(function (it) {
      it.classList.toggle('sel', it.dataset.kind === s.kind);
    });
    $(p + 'BdPrinted').classList.toggle('hidden', screen);
    $(p + 'BdMore').classList.toggle('hidden', !screen);

    // on a monitor: which one, what the board is there, how sure its size is
    const mons = s.monitors || [];
    setOptions($(p + 'BdMonitor'), mons.length ? mons.map(BOARD.monitorLabel) : [T('board.monitor.none')],
               mons.length ? mons.map(function (m) { return m.key; }) : ['']);
    $(p + 'BdMonitor').value = s.monitor || '';
    $(p + 'BdMonitor').title = BOARD.monitorHint(s.mon);
    const m = s.mon, lay = s.layout, rows = [], pr = s.printed;
    const square = screen ? (lay ? lay.square_mm : null) : pr.square_mm;
    $(p + 'BdSquare').replaceChildren.apply($(p + 'BdSquare'),
      square ? [piRow(T('board.f.square'), BOARD.mm(square, 2) + ' mm')] : []);
    if (lay) {
      rows.push(piRow(T('board.f.board'), T('board.f.board.v', lay.squares[0], lay.squares[1],
                                            BOARD.mm(lay.width_mm), BOARD.mm(lay.height_mm))),
                piRow(T('board.f.pitch'), cifre(lay.pitch_mm, 4) + ' mm'));
      rows[rows.length - 1].title = T('board.src.' + m.source);
    }
    $(p + 'BdRows').replaceChildren.apply($(p + 'BdRows'), rows);
    // how sure the size is, in the Advanced part; what needs doing, in plain view
    let sure = '', warn = '';
    if (m && lay && m.source !== 'none' && m.source !== 'ruler') sure = T('board.sure.data', BOARD.mm(lay.width_mm), BOARD.mm(lay.width_u_mm));
    $(p + 'BdSure').textContent = sure;
    $(p + 'BdSure').classList.toggle('hidden', !sure);
    if (!screen) warn = '';
    else if (!m) warn = T('board.err.no_monitor');
    else if (s.error === 'too_small') warn = T('board.err.small');
    else if (!m.is_native && m.native) warn = T('board.warn.native', m.native[0], m.native[1]);
    else if (m.source === 'none') warn = T('board.sure.unknown');
    else if (m.source === 'edid_cm') warn = T('board.sure.rough');
    else if (m.agree === false && m.source !== 'ruler') warn = T('board.warn.disagree', cifre(100 * m.spread, 1));
    $(p + 'BdWarn').textContent = warn;
    $(p + 'BdWarn').classList.toggle('hidden', !warn);
    const show = $(p + 'BdShow');
    show.querySelector('.btn-label').textContent = T(s.shown ? 'board.hide' : 'board.show');
    show.classList.toggle('on', !!s.shown);
    show.disabled = !lay;
    show.classList.toggle('hidden', !screen);

    // the ruler check, once per monitor
    const ruler = $(p + 'BdRuler');
    if (lay) ruler.placeholder = BOARD.mm(lay.width_mm);
    if (document.activeElement !== ruler) ruler.value = '';
    $(p + 'BdRulerNote').textContent = !lay ? ''
      : m.ruler ? T('board.ruler.done', quando(m.ruler.date), BOARD.mm(m.ruler.measured_mm), BOARD.mm(m.ruler.data_mm))
      : T('board.ruler.note', BOARD.mm(lay.width_mm));
    $(p + 'BdForget').classList.toggle('hidden', !(m && m.ruler));
    ruler.disabled = !lay;

    // printed: the length measured between the arrows of the sheet; the square is worked out from it
    const long = $(p + 'BdLong');
    if (document.activeElement !== long) long.value = Math.round(pr.square_mm * pr.squares[0] * 10) / 10;
  },

  // The backend's answer: the board state, or a cause this part says in its own words.
  answer: function (fn) {
    return function (a) {
      if (a && a.error) {
        toast(hasT('board.err.' + a.error) ? T('board.err.' + a.error, a.message || '') : sayError(a.error, a.message),
              5000);
        return;
      }
      if (fn) fn(a);
    };
  },

  // a change answers the board state; the page asks for its own again
  set: function (change) {
    beJson('board_set', JSON.stringify(change), BOARD.answer(function (s) {
      BOARD.render(s);
      if (change.ruler_mm !== undefined) toast(T('board.said.corrected'), 3500);
      BOARD.pageRefresh();
    }));
  },

  pageRefresh: function () {
    if (page === 'calibration') CAL.refresh();
    else if (page === 'live') LIVE.refresh();
  },

  toggle: function () {
    const on = BOARD.state && BOARD.state.shown;
    beJson(on ? 'board_hide' : 'board_show', BOARD.answer(BOARD.render));
  },

  onEvent: function (e) {
    if (e.type !== 'board') return;
    if (e.action === 'closed' && e.reason === 'changed') toast(T('board.said.changed'), 4000);
    ask('board_state', function (s) { if (s) BOARD.render(s); });
  },

  wire: function () {
    BOARD.copies.forEach(function (p) {
      if (!$(p + 'Bd')) return;
      document.querySelectorAll('#' + p + 'BdKinds .mitem').forEach(function (it) {
        it.onclick = function () { BOARD.set({ kind: it.dataset.kind }); };
        it.onkeydown = function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); it.onclick(); } };
      });
      $(p + 'BdMonitor').onchange = function () {
        if ($(p + 'BdMonitor').value) BOARD.set({ monitor: $(p + 'BdMonitor').value });
      };
      $(p + 'BdShow').onclick = BOARD.toggle;
      $(p + 'BdRuler').onchange = function () {
        const v = parseFloat($(p + 'BdRuler').value);
        if (isFinite(v) && v > 0) BOARD.set({ ruler_mm: v });
      };
      $(p + 'BdForget').onclick = function () { BOARD.set({ forget: true }); };
      $(p + 'BdLong').onchange = function () {
        const v = parseFloat($(p + 'BdLong').value);
        if (isFinite(v) && v > 0) BOARD.set({ printed_mm: v });
      };
    });
  },
};
