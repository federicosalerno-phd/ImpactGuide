// ImpactGuide · the Calibration page: the ChArUco board (on a monitor or
// printed, 15-board.js), the photos, the intrinsics of each camera mode and
// the check at the start of a session. The work is calibrate_charuco.py's;
// the page only asks the backend.

const CAL = {
  state: null,
  folder: '',
  last: null,      // {action, mode, ok, log, cause}

  enter: function () {
    CAL.refresh();
    be('phone_refresh');
  },

  refresh: function (then) {
    ask('calib_state', function (s) {
      if (!s) return;
      CAL.render(s);
      if (then) then(s);
    });
  },

  folderOf: function (s) {
    return s.folders.find(function (f) { return f.name === CAL.folder; }) || null;
  },

  render: function (s) {
    CAL.state = s;
    if (!CAL.folderOf(s)) {
      const mine = s.folders.find(function (f) { return f.kind === s.mode || f.kind === 'check_' + s.mode; });
      CAL.folder = mine ? mine.name : (s.folders[0] ? s.folders[0].name : '');
    }
    const f = CAL.folderOf(s);

    // the two camera modes: stored or not, the date, the error and the photos; the rest on hover
    const host = $('calModes');
    host.replaceChildren();
    ['photo', 'video'].forEach(function (mode) {
      const m = s.modes[mode], st = m.stored, chk = s.last_check[mode];
      const card = E('div', 'card col');
      const head = E('div', 'row');
      head.append(E('div', 'sec-lbl grow', T('cal.mode.' + mode)));
      if (chk) head.append(chip(T(chk.ok === '1' ? 'cal.check.ok' : 'cal.check.failed', quando(chk.time)), chk.ok === '1' ? '' : 'err'));
      head.append(chip(st ? T('cal.stored.yes') : T('cal.stored.no'), st ? 'accent' : 'warn'));
      card.append(head);
      if (st) {
        card.append(piRow(T('cal.f.date'), st.date),
                    piRow(T('cal.f.rms'), cifre(st.rms_px, 3) + ' px'),
                    piRow(T('cal.f.views'), st.n_views));
        card.title = st.image_size[0] + ' × ' + st.image_size[1] + ' px · fx, fy ' + cifre(st.fx, 1) + ', ' + cifre(st.fy, 1)
          + ' · cx, cy ' + cifre(st.cx, 1) + ', ' + cifre(st.cy, 1) + ' · ' + st.n_corners + ' ' + T('cal.f.corners');
      }
      if (m.pending) {
        const ok = m.pending.rms_px < s.max_rms;
        card.append(E('div', 'callout ' + (ok ? 'callout-accent' : 'callout-warn'),
                      T(ok ? 'cal.pending.ok' : 'cal.pending.high', cifre(m.pending.rms_px, 3), m.pending.n_views)));
      }
      host.append(card);
    });

    // the board the photos are of: chosen in the panel; the printed one's details under Advanced
    const a = s.active;
    $('calBoard').replaceChildren(
      piRow(T('cal.board.squares'), a.squares[0] + ' × ' + a.squares[1]),
      piRow(T('cal.board.marker'), cifre(a.marker_mm, 2) + ' mm'),
      piRow(T('cal.board.dict'), a.dictionary));
    $('calBoardPrint').classList.toggle('hidden', a.kind === 'screen');
    $('calBoardOpen').disabled = !s.board_file;
    BOARD.render(s.board_ui);

    // the last run, once there is one
    const last = CAL.last;
    $('calLast').classList.toggle('hidden', !last);
    $('calLastHead').replaceChildren();
    if (last) {
      $('calLastHead').append(chip(T('cal.last.' + last.action) + ' · ' + T('cal.mode.' + last.mode),
                                   last.ok ? 'accent' : 'err'));
      $('calLog').textContent = last.log || last.message || '';
    }

    // the folders: the one chosen is where the photos go
    const list = $('calFolders');
    list.replaceChildren();
    s.folders.forEach(function (fo) {
      const it = E('div', 'mitem' + (fo.name === CAL.folder ? ' sel' : ''));
      const name = E('span', 'mitem-name', fo.name + ' ');
      name.append(E('span', 'mitem-det', T('cal.folders.n', fo.n)
        + (fo.board ? ' · ' + T('board.desc.' + (fo.board.kind === 'screen' ? 'screen' : 'printed')) : '')));
      it.append(name, E('span', 'mbadge', hasT('cal.kind.' + fo.kind) ? T('cal.kind.' + fo.kind) : fo.kind.toUpperCase()));
      it.onclick = function () { CAL.folder = fo.name; CAL.render(CAL.state); };
      list.append(it);
    });
    $('calFoldersNote').textContent = s.folders.length ? '' : T('cal.folders.none');

    // the panel
    $('calMode').value = s.mode;
    $('calFolder').textContent = f ? f.path : T('cal.folder.none');
    const busyNow = s.busy.indexOf('calibration') >= 0;
    $('calShoot').disabled = !f;
    $('calImport').disabled = !f;
    $('calOpen').disabled = !f;
    $('calRun').disabled = !f || f.n < s.min_views || busyNow;
    $('calRun').title = f && f.n < s.min_views ? T('cal.run.few', s.min_views) : '';
    $('calCheck').disabled = !f || !s.modes[s.mode].stored || busyNow;
    $('calSave').classList.toggle('hidden', !s.modes[s.mode].pending);
  },

  settings: function () {
    be('calib_settings', $('calMode').value, 0, function (raw) {
      const s = JSON.parse(raw);
      if (!failed(s)) CAL.render(s);
    });
  },

  newFolder: function (kind) {
    ask('calib_folder_new', kind, function (s) {
      if (!s) return;
      CAL.folder = s.folder;
      CAL.render(s);
      toast(T('cal.folder.made', s.folder));
    });
  },

  run: function (action) {
    const s = CAL.state, mode = s.mode;
    ask(action === 'check' ? 'calib_check' : 'calib_run', CAL.folder, mode, function (a) {
      if (!a) return;
      busy(true, T(action === 'check' ? 'cal.busy.check' : 'cal.busy.calibrate', CAL.folder));
      work(true);
    });
  },

  onEvent: function (e) {
    BOARD.onEvent(e);     // the board's controls live on this page and on the Live AR page
    if (e.type === 'calib') {
      busy(false);
      work(false);
      CAL.last = { action: e.action, mode: e.mode, ok: !!e.ok, log: e.log, message: e.message };
      if (e.error) toast(T('err.generic', e.message || e.error), 5000);
      else if (e.action === 'check') toast(T(e.ok ? 'cal.check.said.ok' : 'cal.check.said.failed'), 4000);
      else toast(T(e.ok ? 'cal.run.said.ok' : 'cal.run.said.high', cifre(e.result.rms_px, 3)), 4000);
      CAL.refresh();
    } else if (e.type === 'calib_photo') {
      work(false);
      toast(T('cal.photos.got', e.name));
      CAL.refresh();
    } else if (e.type === 'acquire_failed' && e.key === 'calibration') {
      work(false);
      CAL.refresh();
    }
  },

  wire: function () {
    setOptions($('calMode'), [T('cal.mode.photo'), T('cal.mode.video')], ['photo', 'video']);
    $('calMode').onchange = CAL.settings;
    $('calNewCal').onclick = function () { CAL.newFolder(CAL.state ? CAL.state.mode : 'photo'); };
    $('calNewChk').onclick = function () { CAL.newFolder('check_' + (CAL.state ? CAL.state.mode : 'photo')); };
    $('calShoot').onclick = function () {
      ask('calib_shoot', CAL.folder, function (a) { if (a) work(true, T('pill.shooting')); });
    };
    $('calImport').onclick = function () {
      ask('calib_import', CAL.folder, function (s) {
        if (!s) return;
        CAL.render(s);
        if (s.added || s.skipped) toast(T('cal.photos.added', s.added, s.skipped));
      });
    };
    $('calOpen').onclick = function () {
      const f = CAL.folderOf(CAL.state);
      if (f) ask('open_path', f.path);
    };
    $('calBoardBtn').onclick = function () { ask('calib_board', function (s) { if (s) CAL.render(s); }); };
    $('calBoardOpen').onclick = function () { ask('open_path', CAL.state.board_dir); };
    $('calRun').onclick = function () { CAL.run('calibrate'); };
    $('calCheck').onclick = function () { CAL.run('check'); };
    $('calSave').onclick = function () {
      const mode = CAL.state.mode, st = CAL.state.modes[mode].stored;
      if (st && !armed($('calSave'), T('cal.save.replace', st.date))) return;
      ask('calib_save', mode, function (s) { if (s) { CAL.render(s); toast(T('cal.save.done', T('cal.mode.' + mode))); } });
    };
    BOARD.wire();         // its two copies, here and on the Live AR page
  },
};
