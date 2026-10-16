// ImpactGuide · the calibration board: on a monitor (the default) or printed.
// The same controls sit in the first step of Calibrate the phone and of Live AR
// (ids starting with cal and lv); both modules hand this part the backend's
// board state, and it draws both copies and the card on the stage. On a
// monitor the board goes full screen on another monitor when there is one, or
// in a window beside the app, docked, at any split: its size in mm comes from
// the pixels of that window and the pitch of its monitor, and follows the
// window when it is moved or resized (board_window.py). In plain view: the
// kind, the square, what needs doing and Show board; where it goes, the
// monitor and the ruler check sit under Advanced. The page shows, chooses and asks.

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

  // what needs doing, in plain view: one line, or nothing
  warning: function (s) {
    const m = s.mon;
    if (s.kind !== 'screen') return '';
    if (!m) return T('board.err.no_monitor');
    if (s.error === 'too_small') return T(s.shown ? 'board.err.small.window' : 'board.err.small');
    if (!m.is_native && m.native) return T('board.warn.native', m.native[0], m.native[1]);
    if (m.source === 'none') return T('board.sure.unknown');
    if (m.source === 'edid_cm') return T('board.sure.rough');
    if (m.agree === false && m.source !== 'ruler') return T('board.warn.disagree', cifre(100 * m.spread, 1));
    return '';
  },

  draw: function (p, s) {
    const screen = s.kind === 'screen';
    document.querySelectorAll('#' + p + 'BdKinds .mitem').forEach(function (it) {
      it.classList.toggle('sel', it.dataset.kind === s.kind);
    });
    $(p + 'BdPrinted').classList.toggle('hidden', screen);
    $(p + 'BdMore').classList.toggle('hidden', !screen);

    // where it goes: beside the app (one monitor) or full screen (on another one, or on this one)
    document.querySelectorAll('#' + p + 'BdPlaces .mitem').forEach(function (it) {
      it.classList.toggle('sel', it.dataset.place === s.placed);
    });
    $(p + 'BdDock').classList.toggle('hidden', !(s.shown && s.placed === 'beside' && !s.docked));

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
      rows[rows.length - 1].title = T('board.src.' + (m ? m.source : 'none'));
    }
    $(p + 'BdRows').replaceChildren.apply($(p + 'BdRows'), rows);
    const sure = m && lay && m.source !== 'none' && m.source !== 'ruler'
      ? T('board.sure.data', BOARD.mm(lay.width_mm), BOARD.mm(lay.width_u_mm)) : '';
    $(p + 'BdSure').textContent = sure;
    $(p + 'BdSure').classList.toggle('hidden', !sure);
    const warn = BOARD.warning(s);
    $(p + 'BdWarn').textContent = warn;
    $(p + 'BdWarn').classList.toggle('hidden', !warn);
    const show = $(p + 'BdShow');
    show.querySelector('.btn-label').textContent = T(s.shown ? 'board.hide' : 'board.show');
    show.classList.toggle('on', !!s.shown);
    show.disabled = !m;
    show.classList.toggle('hidden', !screen);

    // the ruler check, once per monitor
    const ruler = $(p + 'BdRuler');
    if (lay) ruler.placeholder = BOARD.mm(lay.width_mm);
    if (document.activeElement !== ruler) ruler.value = '';
    $(p + 'BdRulerNote').textContent = !lay || !m ? ''
      : m.ruler ? T('board.ruler.done', quando(m.ruler.date), BOARD.mm(m.ruler.measured_mm), BOARD.mm(m.ruler.data_mm))
      : T('board.ruler.note', BOARD.mm(lay.width_mm));
    $(p + 'BdForget').classList.toggle('hidden', !(m && m.ruler));
    ruler.disabled = !lay;

    // printed: the length measured between the arrows of the sheet; the square is worked out from it
    const long = $(p + 'BdLong');
    if (document.activeElement !== long) long.value = Math.round(pr.square_mm * pr.squares[0] * 10) / 10;
  },

  // The card on the stage of the board step: where the board is, how large, and the one thing to do.
  card: function (s) {
    if (!s) return;
    const screen = s.kind === 'screen', lay = s.layout;
    let title, line = '', hint = '', dot = '';
    if (!screen) {
      const pr = s.printed;
      title = T('board.card.printed');
      line = T('board.card.line', pr.squares[0], pr.squares[1], BOARD.mm(pr.square_mm, 2),
               BOARD.mm(pr.square_mm * pr.squares[0]));
      hint = T('board.card.printed.hint');
      dot = 'ok';
    } else {
      title = T(!s.shown ? 'board.card.off' : s.placed === 'full' ? 'board.card.full'
        : s.docked ? 'board.card.beside' : 'board.card.window');
      line = lay ? T('board.card.line', lay.squares[0], lay.squares[1], BOARD.mm(lay.square_mm, 2), BOARD.mm(lay.width_mm))
        : BOARD.warning(s);
      hint = !s.shown ? T('board.card.off.hint') : s.error === 'too_small' ? T('board.err.small.window')
        : s.docked ? T('board.card.beside.hint') : s.placed === 'full' ? T('board.card.full.hint')
        : T('board.card.window.hint');
      dot = s.shown && lay ? 'ok' : s.shown ? 'warn' : '';
      if (s.held && s.shown) hint = T('board.card.held');
    }
    $('bdTitle').textContent = title;
    $('bdLine').textContent = line;
    $('bdHint').textContent = hint;
    $('bdDot').className = 'live-dot' + (dot ? ' ' + dot : '');
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

  // a change answers the board state; the module asks for its own again
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
    beJson(on ? 'board_hide' : 'board_show', BOARD.answer(function (s) { BOARD.render(s); BOARD.pageRefresh(); }));
  },

  onEvent: function (e) {
    if (e.type !== 'board') return;
    if (e.action === 'closed' && e.reason === 'changed') toast(T('board.said.changed'), 4000);
    BOARD.pageRefresh();
  },

  wire: function () {
    BOARD.copies.forEach(function (p) {
      if (!$(p + 'Bd')) return;
      const key = function (it) {
        it.onkeydown = function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); it.onclick(); } };
      };
      document.querySelectorAll('#' + p + 'BdKinds .mitem').forEach(function (it) {
        it.onclick = function () { BOARD.set({ kind: it.dataset.kind }); };
        key(it);
      });
      document.querySelectorAll('#' + p + 'BdPlaces .mitem').forEach(function (it) {
        it.onclick = function () { BOARD.set({ place: it.dataset.place }); };
        key(it);
      });
      $(p + 'BdDock').onclick = function () { BOARD.set({ dock: true }); };
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
