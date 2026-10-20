"""The Dataset page's side of ImpactGuide's backend; the work itself is dataset_pack.py's.

What the dataset holds is read in a worker each time the step opens (a `dataset` event, action read, once read).
Check reads every file and Export writes the package, one job at a time in a worker: the page reads their progress
(state) while they run and is told by a `dataset` event when they end: checked (with the count of problems),
exported, stopped, failed (with its code). The last check and the last package are kept in dataset.json among the
app's own files, so the page shows them after a restart. The real trials come from the backend's data folders (the
simulation's under --simulate), their ground truth from the Scans page's store, the synthetic dataset from `synth`
(dataset_pack.SYNTH, the work folder's synth, by default). The folder dialogs are the host's (impactguide_app.py):
here the package is written where it was asked for, never over anything.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import dataset_pack

log = logging.getLogger("impactguide")

STATE = "dataset.json"          # the last check and the last package, among the app's own files
PARTS = ("real", "synthetic")


class DatasetError(Exception):
    """A refusal the page says in its words (ds.err.<code>): busy, empty."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DatasetPage:
    """What the Dataset page asks of the backend `b` (impactguide_app.Backend)."""

    def __init__(self, backend: Any) -> None:
        self.b = backend
        self.synth = dataset_pack.SYNTH
        self.inv: dict | None = None    # the dataset as last read
        self.reading = False
        self.job: dataset_pack.Job | None = None
        self.last = self._load()

    # ── where things are ─────────────────────────────────────────────
    def _path(self) -> Path:
        return Path(self.b.roots.local) / STATE

    def _load(self) -> dict:
        try:
            d = json.loads(self._path().read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _keep(self) -> None:
        path = self._path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.last, f, indent=1)
            os.replace(tmp, path)
        except OSError:
            log.warning("the dataset's last check and package not kept", exc_info=True)

    def sources(self) -> dataset_pack.Sources:
        """The data folders of the backend; the rig calibration the Scans page measures with (scans_page.py)."""
        r = self.b.roots
        scans = getattr(self.b, "scans", None)
        return dataset_pack.Sources(r.raw, r.protocol, r.processed, self.synth,
                                    rig=scans.calibration() if scans is not None else None)

    def truth(self) -> dataset_pack.Truth:
        scans = getattr(self.b, "scans", None)
        return dataset_pack.Truth(self.sources(), scans.info if scans is not None else None)

    def model_print(self) -> str:
        cur = self.b.twin.current()
        return str((cur or {}).get("phone_print", ""))

    def parts(self) -> dict:
        kept = self.b.prefs.data.get("dataset_parts")
        kept = kept if isinstance(kept, dict) else {}
        return {k: kept.get(k) is not False for k in PARTS}

    def set_parts(self, ask: dict) -> dict:
        self.b.prefs.set("dataset_parts", {k: bool(ask.get(k, True)) for k in PARTS})
        return self.state({})

    def start_folder(self) -> Path:
        """Where the dialog opens: the folder chosen last, else the dataset's folder of the work folder (the
        simulation's own under --simulate)."""
        kept = self.b.prefs.data.get("dataset_folder") or ""
        if kept and Path(kept).is_dir():
            return Path(kept)
        data = os.environ.get("IMPACTGUIDE_DATA")        # the installed app's data folder, in Documents
        out = (Path(self.b.roots.simulate) / "dataset" if self.b.roots.simulate
               else Path(data) / "dataset" if data else dataset_pack.DEFAULT_OUT)
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError:
            return Path.home()
        return out

    # ── the state ────────────────────────────────────────────────────
    def state(self, ask: dict) -> dict:
        """What the page shows; fresh: the dataset read again (in a worker)."""
        if ask.get("fresh") or self.inv is None:
            self.read()
        x = self.last.get("export")
        if x:
            x = {**x, "there": Path(x.get("path", "")).exists()}
        return {"inventory": self.inv, "reading": self.reading,
                "job": self.job.progress() if self.job else None,
                "check": self.last.get("check"), "export": x, "parts": self.parts(),
                "simulate": bool(self.b.roots.simulate)}

    def read(self) -> None:
        if self.reading:
            return
        src, truth, model = self.sources(), self.truth(), self.model_print()

        def done(inv: dict | None, error: BaseException | None) -> None:
            self.reading = False
            if error is not None:
                log.warning("the dataset could not be read", exc_info=(type(error), error, error.__traceback__))
                self._emit("failed", error="ds_read")
                return
            self.inv = inv
            self._emit("read")
        self.reading = self.b._spawn("dataset_read", lambda: dataset_pack.inventory(src, model, truth), done)

    def _emit(self, action: str, **fields: Any) -> None:
        self.b._emit({"type": "dataset", "action": action, **fields})

    # ── the jobs ─────────────────────────────────────────────────────
    def _wanted(self) -> list[str]:
        p = self.parts()
        return [k for k in PARTS if p[k]]

    def _run(self, kind: str, work) -> dict:
        if self.job is not None:
            raise DatasetError("ds_busy")
        job = dataset_pack.Job(kind)

        def done(res: dict | None, error: BaseException | None) -> None:
            self.job = None
            if isinstance(error, dataset_pack.Stopped):
                self._emit("stopped", kind=kind)
            elif isinstance(error, ValueError) and str(error) == "empty":
                self._emit("failed", kind=kind, error="ds_empty")
            elif error is not None:
                log.warning("dataset %s failed", kind, exc_info=(type(error), error, error.__traceback__))
                self._emit("failed", kind=kind, error="ds_failed")
            else:
                self.last[kind] = res
                self._keep()
                n = sum(p["n"] for part in ("real", "synthetic") for p in (res.get(part) or {}).get("problems", [])) \
                    if kind == "check" else len(res.get("problems") or [])
                self._emit("checked" if kind == "check" else "exported", problems=n, level=res.get("level"))
            self.read()
        if not self.b._spawn("dataset", lambda: work(job), done):
            raise DatasetError("ds_busy")
        self.job = job
        return self.state({})

    def check(self) -> dict:
        src, truth, model, parts = self.sources(), self.truth(), self.model_print(), self._wanted()
        if not parts:
            raise DatasetError("ds_empty")
        return self._run("check", lambda job: dataset_pack.check(src, job, parts, model, truth))

    def export(self, dest: Path, as_zip: bool, made_by: str) -> dict:
        """The package written at dest (the folder of the package, or the zip's path)."""
        src, truth, model, parts = self.sources(), self.truth(), self.model_print(), self._wanted()
        if not parts:
            raise DatasetError("ds_empty")
        sim = bool(self.b.roots.simulate)

        def work(job):
            res = dataset_pack.export(src, dest, job, as_zip=as_zip, parts=parts, model_print=model, truth=truth,
                                      made_by=made_by, simulated=sim)
            return {**res, "name": Path(res["path"]).name}
        return self._run("export", work)

    def target(self, chosen: str, as_zip: bool) -> Path:
        """The package's place from what the dialog gave: a folder (the package's folder inside it, a name of its
        own) or a zip's path (a name of its own beside it when it is taken)."""
        p = Path(chosen)
        if as_zip:
            folder = p.parent
            stem = p.stem if p.suffix.lower() == ".zip" else p.name
            self.b.prefs.set("dataset_folder", str(folder))
            want = folder / f"{stem}.zip"
            return want if not want.exists() and not want.with_name(want.name + dataset_pack.PART).exists() \
                else dataset_pack.free_target(folder, stem, ".zip")
        self.b.prefs.set("dataset_folder", str(p))
        return dataset_pack.free_target(p, dataset_pack.package_name(), "")

    def stop(self) -> dict:
        if self.job is not None:
            self.job.stop()
        return self.state({})

    def opened(self) -> Path | None:
        """What Open shows: the package's folder, or the folder of the zip."""
        x = self.last.get("export") or {}
        p = Path(x.get("path", ""))
        if not x or not p.exists():
            return None
        return p if p.is_dir() else p.parent

    def shutdown(self) -> None:
        if self.job is not None:
            self.job.stop()
