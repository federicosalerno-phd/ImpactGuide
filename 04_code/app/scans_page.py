"""The Scans page's side of ImpactGuide's backend; the work itself is scan_desk.py's.

The scanner's export folder is watched (acquisition.ExportWatch, kept across the steps so that no export between two
of them is missed) for the open trial or for a run of scans without an impact: a new export of the trial is measured
alone, bound to its step (scan_desk.bind: its side of the impact, the arm position it was taken from) and staged as
the trial's page stages one; once staged or filed, the trial's scans are measured (scan_desk.Desk.evaluate) and the
trial log's two online values written. The trial's page may still ask for the scan of the step it shows
(Backend.scan_watch): that watch goes first, and the file it takes is not seen again here. All the work runs in
workers, one job at a time in the order it came (a measure, a run's scan, the calibration, the day's processing);
the page is told by `scans` events: measuring, measured (with the time from the export), unused (an export no step
waits for), captured, calibrated, processed, failed.

In simulation the stand-in scanner writes synthetic scans of the bench (scan_desk.SimScanner) for the step it is
asked for, and for a run without an impact one after the other; IMPACTGUIDE_SIM_SCANS=stand_in keeps the small
stand-in files of acquisition.fake_scan, which hold no spheres (the backend's tests of the trials).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import acquisition
import ground_truth as gt
import scan_desk
import trials

log = logging.getLogger("impactguide")

SIM_SCANS_ENV = "IMPACTGUIDE_SIM_SCANS"     # "stand_in": the small stand-ins in simulation, else synthetic scans
SIM_GAP_S = 1.2                             # in simulation, the scans of a run exported this far apart
SIM_STATE = "sim_scanner.json"              # the simulated bench's state, in the simulation's folder


class ScansPage:
    """What the Scans page asks of the backend `b` (impactguide_app.Backend): its state, the runs without an impact,
    the calibration, the day's processing; and the hooks the backend calls (poll, consumed, arrived, sim_export)."""

    def __init__(self, backend: Any) -> None:
        self.b = backend
        self.watch: acquisition.ExportWatch | None = None
        self.jobs: list = []            # [(what, fn, done)] waiting, run one at a time
        self.running = ""               # what runs now
        self.capture = ""               # the run without an impact that takes the exports now ('' none)
        self.sim_stop = False
        self.processed: dict = {}       # what the last processing said
        self._desks: dict = {}
        self._sim = None

    # ── where things are ─────────────────────────────────────────────
    def store(self) -> Path:
        return self.b.roots.processed / scan_desk.STORE

    def calibration(self) -> Path:
        """The rig calibration: the simulation's own under --simulate, else the project's."""
        if self.b.roots.simulate:
            return self.b.roots.processed / "targets" / gt.CALIBRATION_JSON.name
        return gt.CALIBRATION_JSON

    def info(self) -> dict | None:
        try:
            return self.b.twin.info()
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def desk(self, phase: str | None = None) -> scan_desk.Desk:
        run = self.b.run(phase)
        d = self._desks.get(run.phase)
        if d is None or d.run is not run:
            d = self._desks[run.phase] = scan_desk.Desk(run, self.store(), self.calibration(), self.info)
        return d

    def capture_of(self, kind: str) -> scan_desk.Capture:
        return scan_desk.Capture(kind, self.b.roots.raw, self.store(), self.calibration())

    def synthetic(self) -> bool:
        return bool(self.b.roots.simulate) and os.environ.get(SIM_SCANS_ENV, "synthetic") != "stand_in"

    def sim(self) -> scan_desk.SimScanner:
        if self._sim is None:
            self._sim = scan_desk.SimScanner(self.b.roots.local / SIM_STATE)
        return self._sim

    # ── the open trial ───────────────────────────────────────────────
    def open_row(self, run: trials.Run) -> dict | None:
        return next((r for r in run.log.rows() if r["status"] == trials.OPEN), None)

    def staged(self) -> dict:
        """{step key: path} of the scans staged for the open trial."""
        return {k: v["path"] for k, v in (self.b.cur.get("staged") or {}).items() if v.get("kind") == "scan"}

    def shown_step(self, run: trials.Run, row: dict) -> str:
        """The step the trial's page is at."""
        try:
            steps = run.steps(row["trial"])
            return steps[min(int(self.b.cur.get("step", 0)), len(steps) - 1)].key
        except (LookupError, ValueError, TypeError, trials.TrialsError):
            return ""

    def waits(self) -> bool:
        """Something takes the next export: a run without an impact, or a step of the open trial."""
        if self.capture:
            d = self.capture_of(self.capture).load()
            return not (self.capture == "repeat" and len(d["files"]) >= int(d["want"]))   # a full run takes no more
        run = self.b.run()
        row = self.open_row(run)
        return row is not None and scan_desk.bind(run, row) is not None

    # ── the work, one job at a time ──────────────────────────────────
    def _queue(self, what: str, prepare, done) -> bool:
        """A job: prepare() runs on this (the GUI) thread when its turn comes and gives the function the worker runs;
        done(result, error) runs back here. A job already waiting is not queued twice (it sees what is new)."""
        if any(j[0] == what for j in self.jobs):
            return False
        self.jobs.append((what, prepare, done))
        self._next()
        return True

    def _next(self) -> None:
        if self.running or not self.jobs:
            return
        what, prepare, done = self.jobs.pop(0)
        self.running = what

        def end(result: Any, error: BaseException | None) -> None:
            self.running = ""
            try:
                done(result, error)
            finally:
                self._next()
        try:
            fn = prepare()
        except Exception as e:  # noqa: BLE001  said like a failure of the job
            end(None, e)
            return
        if not self.b._spawn("scans_work", fn, end):
            self.running = ""
            self.jobs.insert(0, (what, prepare, done))

    def busy(self) -> list[str]:
        return [w for w in [self.running] + [j[0] for j in self.jobs] if w]

    # ── the scanner's folder ─────────────────────────────────────────
    def poll(self) -> None:
        """The folder listed in a worker, while the trial's page waits for no step of its own (Backend._poll_scan).
        An export that nothing takes when it comes is said unused, and never goes to a step that comes later."""
        folder = self.b.scanner_folder()
        if folder is None:
            self.watch = None
            return
        if self.watch is None or self.watch.folder != folder:
            if not folder.is_dir():
                if not self.b.roots.simulate:
                    return
                folder.mkdir(parents=True, exist_ok=True)
            self.watch = acquisition.ExportWatch(folder)
        watch = self.watch

        def done(found: Path | None, error: BaseException | None) -> None:
            if error is not None or found is None or watch is not self.watch:
                return
            self.consumed(found)
            if self.waits():
                self.came(found)
            else:           # said, and never given to a step that comes later
                self.b._emit({"type": "scans", "action": "unused", "file": found.name})
        self.b._spawn("scans_poll", watch.poll, done)

    def consumed(self, path: Path) -> None:
        """An export taken (here, or by the trial's page's own watch): never taken again."""
        w = self.watch
        if w is None or Path(path).parent != w.folder:
            return
        try:
            st = Path(path).stat()
            w.before[Path(path).name] = (st.st_size, st.st_mtime_ns)
        except OSError:
            pass
        w.sizes.pop(Path(path).name, None)

    def came(self, found: Path, prefer: str | None = None) -> None:
        """A new export: to the run without an impact that is on, else to a step of the open trial. prefer: the step
        whose own watch took it (the trial's page waits at it): that step, unless the scan was taken at the arm
        position of another step that waits."""
        if self.capture:
            d = self.capture_of(self.capture).load()
            if self.capture in ("repeat", "reseat") and len(d["files"]) >= int(d["want"]):
                self.b._emit({"type": "scans", "action": "unused", "file": found.name})     # the run has them all
            else:
                self.add_capture(self.capture, found)
            return
        run = self.b.run()
        row = self.open_row(run)
        if row is None:
            self.b._emit({"type": "scans", "action": "unused", "file": found.name})
            return
        explicit = prefer is not None
        trial, staged = row["trial"], set(self.staged())
        prefer = prefer or self.shown_step(run, row)
        desk = self.desk()

        def work() -> tuple[dict, str | None]:
            v = desk.view_alone(found)
            arm = (v.get("arm") or {}).get("arm")
            r = run.log.row(trial)
            key = scan_desk.bind(run, r, staged, arm, prefer)
            if explicit:
                after = bool(r.get("time_impact"))
                waiting = {s["key"]: s for s in scan_desk.slots(run, trial)
                           if (s["when"] == "post") == after and scan_desk.filed(run, trial, s) is None}
                if prefer in waiting and (arm is None or waiting[prefer]["arm"] == arm
                                          or not any(s["arm"] == arm for s in waiting.values())):
                    key = prefer
            if key:
                desk.keep_view(trial, v)
            return v, key

        def done(got: Any, error: BaseException | None) -> None:
            if error is not None:
                log.warning("scan %s not bound", found.name, exc_info=(type(error), error, error.__traceback__))
                if explicit:            # the step its own watch was for takes it all the same
                    self.b._staged(trial, prefer, found, temp=False)
                    return
                self.b._emit({"type": "scans", "action": "failed", "trial": trial, "error": "scan_unreadable",
                              "args": {"file": found.name}})
                return
            v, key = got
            if key is None:
                self.b._emit({"type": "scans", "action": "unused", "file": found.name, "trial": trial})
                return
            self.b._staged(trial, key, found, temp=False)      # staged, said, then measured (arrived)
        self._queue(f"bind:{found.name}", lambda: work, done)

    def arrived(self, trial: str, key: str) -> None:
        """A scan of a trial staged or filed (Backend._staged, step_confirm): the trial's scans measured."""
        if scan_desk.SCAN_KEY.match(key or ""):
            self.measure(trial)

    def measure(self, trial: str, phase: str | None = None) -> None:
        desk = self.desk(phase)
        cache = getattr(self.b.twin, "cache", None)

        def prepare():
            staged = self.staged() if self.b.cur.get("trial") == trial else {}

            def work() -> dict:
                st = desk.evaluate(trial, staged, measure=True)
                try:
                    desk.twin_entry(trial, cache, st["result"].get("dT"))
                except (OSError, ValueError, KeyError, trials.TrialsError):
                    log.warning("twin's motion of %s not kept", trial, exc_info=True)
                return st
            return work

        def done(st: dict | None, error: BaseException | None) -> None:
            if error is not None:
                log.warning("scans of %s not measured", trial, exc_info=(type(error), error, error.__traceback__))
                self.b._emit({"type": "scans", "action": "failed", "trial": trial, "error": "scans_failed"})
                return
            res = st["result"]
            said = {}
            if res.get("log"):
                try:
                    said = desk.log_values(trial, res["log"])
                except (trials.TrialsError, OSError, ValueError):
                    log.warning("online values of %s not logged", trial, exc_info=True)
            self.b._emit({"type": "scans", "action": "measured", "trial": trial, "state": res.get("state"),
                          "seconds": res.get("seconds"), "reposition": bool(res.get("reposition")),
                          "bad": any(s["state"] == "bad" for s in st["slots"]),
                          "logged": sorted({**(said.get("written") or {}), **(said.get("changed") or {})})})
        if self._queue(f"measure:{trial}", prepare, done):
            self.b._emit({"type": "scans", "action": "measuring", "trial": trial})

    # ── the page's state ─────────────────────────────────────────────
    def state(self, ask: dict) -> dict:
        run = self.b.run()
        desk = self.desk()
        rows = run.log.rows()
        open_row = next((r for r in rows if r["status"] == trials.OPEN), None)
        measured = set(desk.measured_trials())
        listed = [r for r in reversed(rows) if r["trial"] in measured or r is open_row]
        names = [r["trial"] for r in listed]
        want = str(ask.get("trial") or "")
        trial = want if want in names else (open_row["trial"] if open_row else (names[0] if names else ""))
        is_open = bool(open_row) and trial == open_row["trial"]
        st = desk.evaluate(trial, self.staged() if is_open else None) if trial else \
            {"trial": None, "slots": [], "before": {}, "result": {"state": "none"}, "todo": False}
        if st["todo"] and not any(w == f"measure:{trial}" for w in self.busy()):
            self.measure(trial)
        st["result"].pop("dT", None)
        folder = self.b.scanner_folder()
        out = {**st, "open": is_open, "phase": run.phase, "simulate": bool(self.b.roots.simulate),
               "folder": str(folder or ""), "folder_ok": bool(folder and folder.is_dir()),
               "formats": [s.lstrip(".").upper() for s in acquisition.SCAN_SUFFIXES],
               "watching": bool(folder) and self.waits(), "calibration": desk.calibration_state(),
               "rigs": gt.rig_status(self.calibration()),
               "trials": [{k: r.get(k, "") for k in ("trial", "holder", "angle_deg", "direction_deg", "status")}
                          for r in listed],
               "busy": self.busy(), "measuring": any(w == f"measure:{trial}" for w in self.busy()),
               "capture": self.capture_state(), "processed": self.processed, "today": date.today().isoformat()}
        return gt.jsonable(out)

    def capture_state(self) -> dict:
        out = {"on": self.capture}
        for kind in scan_desk.CAPTURES:
            d = self.capture_of(kind).load()
            files = d.get("files") or []
            out[kind] = {"want": d.get("want"), "n": len(files), "good": sum(1 for f in files if f.get("ok")),
                         "started": d.get("started", ""), "calibrated": d.get("calibrated", ""),
                         "files": [{k: f.get(k) for k in ("name", "ok", "n_spheres", "exported")} for f in files[-24:]],
                         "result": {k: v for k, v in (d.get("result") or {}).items() if k != "deviation_mm"}}
        return out

    # ── runs without an impact ───────────────────────────────────────
    def capture_cmd(self, kind: str, action: str, want: int | None = None) -> dict:
        if kind not in scan_desk.CAPTURES:
            raise ValueError(f"no run of kind {kind}")
        cap = self.capture_of(kind)
        if action == "start":
            if self.capture and self.capture != kind:
                self.capture_of(self.capture).stop()
            cap.start(want)
            self.capture, self.sim_stop = kind, False
            if self.synthetic():
                self.sim_run(kind, cap.load()["want"])
        elif action == "stop":
            cap.stop()
            if self.capture == kind:
                self.capture = ""
            self.sim_stop = True
        else:
            raise ValueError(f"no action {action}")
        return self.state({})

    def add_capture(self, kind: str, found: Path) -> None:
        cap = self.capture_of(kind)

        def done(entry: dict | None, error: BaseException | None) -> None:
            if error is not None:
                log.warning("scan %s of the run not kept", found.name, exc_info=(type(error), error,
                                                                                 error.__traceback__))
                self.b._emit({"type": "scans", "action": "failed", "kind": kind, "error": "copy_failed",
                              "args": {"file": found.name}})
                return
            d = cap.load()
            full = kind in ("repeat", "reseat") and len(d["files"]) >= int(d["want"])
            if full:                # its result stays on show until the run is closed (Done)
                self.sim_stop = True
            self.b._emit({"type": "scans", "action": "captured", "kind": kind, "n": len(d["files"]),
                          "want": d["want"], "ok": bool(entry.get("ok")), "done": full})
        self._queue(f"capture:{found.name}", lambda: (lambda: cap.add(found)), done)

    def calibrate(self) -> dict:
        cap = self.capture_of("calibration")

        def done(cal: dict | None, error: BaseException | None) -> None:
            if error is not None:
                log.warning("rig calibration not made", exc_info=(type(error), error, error.__traceback__))
                self.b._emit({"type": "scans", "action": "failed", "kind": "calibration", "error": "calibration_failed"})
                return
            if self.capture == "calibration":
                self.capture = ""
            self.sim_stop = True
            self.b._emit({"type": "scans", "action": "calibrated", "scans": len((cal or {}).get("scans") or [])})
        if not self._queue("calibrate", lambda: cap.calibrate, done):
            return {"error": "busy"}
        return {"ok": True}

    # ── the day's processing ─────────────────────────────────────────
    def process(self, day: str = "", every: bool = False) -> dict:
        desk = self.desk()
        cache = getattr(self.b.twin, "cache", None)
        day = day or date.today().isoformat()

        def done(out: dict | None, error: BaseException | None) -> None:
            if error is not None:
                log.warning("scans not processed", exc_info=(type(error), error, error.__traceback__))
                self.b._emit({"type": "scans", "action": "failed", "error": "process_failed"})
                return
            self.processed = {"day": out.get("day", ""), "trials": out["trials"], "measured": out["measured"],
                              "failed": len(out["failed"]), "missing": len(out["missing"]),
                              "missing_trials": sorted(out["missing"])[:12], "when": scan_desk.now_iso()}
            self.b._emit({"type": "scans", "action": "processed", **self.processed})
        if not self._queue("process", lambda: (lambda: scan_desk.process(desk, day, every, cache)), done):
            return {"error": "busy"}
        return {"ok": True}

    # ── the stand-in scanner (simulation) ────────────────────────────
    def sim_export(self, trial: str, key: str, folder: Path, name: str) -> None:
        """The scan of a trial's step written into the export folder, in a worker."""
        run = self.b.run()
        slot = scan_desk.slot_of(key, run.config)
        if slot is None:
            return

        def work() -> Path:
            return self.sim().export(folder, name, slot["az"], run.log.row(trial), slot["when"])
        self.b._spawn(f"scans_sim:{key}", work, lambda _r, e: e and log.warning(
            "simulated scan not written", exc_info=(type(e), e, e.__traceback__)))

    def simulate(self) -> dict:
        """In simulation: the scanner exports the scan the open trial waits for (the Scans page's own button)."""
        if not self.b.roots.simulate:
            return {"error": "not_simulated"}
        run = self.b.run()
        row = self.open_row(run)
        folder = self.b.scanner_folder()
        if row is None or folder is None:
            return {"error": "no_trial"}
        key = scan_desk.bind(run, row, set(self.staged()), None, self.shown_step(run, row))
        if key is None:
            return {"error": "no_scan_waits"}
        name = f"export_{row['trial']}_{key}_{time.strftime('%H%M%S')}.ply"
        if self.synthetic():
            self.sim_export(row["trial"], key, folder, name)
        else:
            acquisition.fake_scan(folder, name)
        return {"ok": True, "key": key}

    def sim_run(self, kind: str, want: int) -> None:
        """The scans of a run without an impact, one after the other, until it is stopped or has them all."""
        folder = self.b.scanner_folder()
        az = self.b.run().config.scan_az[0]
        sim = self.sim()
        if folder is None:
            return
        folder.mkdir(parents=True, exist_ok=True)

        def work() -> int:
            n = 0
            stamp = time.strftime("%H%M%S")
            for i in range(int(want)):
                if self.sim_stop:
                    break
                time.sleep(SIM_GAP_S)
                if self.sim_stop:
                    break
                sim.export(folder, f"export_{scan_desk.CAPTURES[kind]}_{stamp}_{i + 1:03d}.ply", az, kind=kind)
                n += 1
            return n
        self.b._spawn("scans_sim_run", work, lambda _r, e: e and log.warning(
            "simulated run not written", exc_info=(type(e), e, e.__traceback__)))
