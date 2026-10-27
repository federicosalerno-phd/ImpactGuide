"""Files of a synced drive read once and then answered from memory, looked at again by a thread of this module.

On a synced drive a look at a file takes tens of milliseconds, and half a second or more while the drive is busy. The
window's thread asked several times a second for files that change once a week (the plate's design, the mandible's
frame) and stood still that long each time. Memo(fn, paths) answers fn() from memory: computed the first time it is
asked (or ahead, by warm()), then again only after this module's thread has seen one of the paths change (size, time
or presence), which it looks at no more often than every refresh_s and never on the caller's thread. A failure (the
file missing or unreadable) is kept like a value and raised to every caller until the file changes.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

REFRESH_S = 3.0             # a path looked at again at most this often

_todo: queue.SimpleQueue = queue.SimpleQueue()
_thread: list[threading.Thread] = []
_start_lock = threading.Lock()


def _worker() -> None:
    while True:
        job = _todo.get()
        try:
            job()
        except Exception:   # noqa: BLE001  a memo keeps its own failures; nothing else may stop this thread
            pass


def _later(job: Callable[[], None]) -> None:
    """job() on this module's thread (one for every memo: a look never makes a thread of its own)."""
    with _start_lock:
        if not _thread or not _thread[0].is_alive():
            t = threading.Thread(target=_worker, name="drive-cache", daemon=True)
            t.start()
            _thread[:] = [t]
    _todo.put(job)


def stamp(paths: Iterable[os.PathLike | str]) -> tuple:
    """(size, mtime_ns) of each path, None for one that is missing."""
    out = []
    for p in paths:
        try:
            st = os.stat(p)
            out.append((st.st_size, st.st_mtime_ns))
        except OSError:
            out.append(None)
    return tuple(out)


class Memo:
    """fn() from memory, computed again when one of `paths` changes (see the module)."""

    def __init__(self, fn: Callable[[], Any], paths: Iterable[os.PathLike | str], refresh_s: float = REFRESH_S):
        self.fn = fn
        self.paths = [os.fspath(p) for p in paths]
        self.refresh_s = refresh_s
        self._lock = threading.Lock()
        self._have = False
        self._value: Any = None
        self._error: BaseException | None = None
        self._stamp: tuple | None = None
        self._looked = 0.0
        self._looking = False

    def _compute(self) -> None:
        st = stamp(self.paths)
        try:
            value, error = self.fn(), None
        except Exception as e:  # noqa: BLE001  kept and raised to the callers
            value, error = None, e
        with self._lock:
            self._value, self._error, self._stamp, self._have = value, error, st, True
            self._looked = time.monotonic()

    def _look(self) -> None:
        try:
            if stamp(self.paths) != self._stamp:
                self._compute()
        finally:
            with self._lock:
                self._looking = False
                self._looked = time.monotonic()

    def get(self) -> Any:
        """The value; computed here only the first time (unless warm() did it before)."""
        if not self._have:
            self._compute()
        elif time.monotonic() - self._looked >= self.refresh_s:
            with self._lock:
                go = not self._looking
                self._looking = True
            if go:
                _later(self._look)
        with self._lock:
            value, error = self._value, self._error
        if error is not None:
            raise error.with_traceback(None)
        return value

    def warm(self) -> None:
        """Computed now on this module's thread, so the first get() finds it ready."""
        _later(lambda: None if self._have else self._compute())

    def forget(self) -> None:
        """Computed again at the next get() (a file this program wrote itself)."""
        with self._lock:
            self._have = False
