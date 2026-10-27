"""A watch on the window's thread, for finding what holds it up. Off unless asked for:

    IMPACTGUIDE_WATCHDOG=1      every stall of the window's thread over 100 ms goes to the log, with where it was
    IMPACTGUIDE_WATCHDOG=250    over 250 ms

A timer on the window's thread beats every BEAT_MS. A thread of its own looks every LOOK_MS whether the last beat is
older than the threshold; when it is, it takes the window thread's Python stack then, and again every RESAMPLE_MS while
the stall lasts. When the beats come back the stall is logged once: how long, what the app was doing (tag), and each
different stack taken. A stall the looking thread could not see, because the window's thread kept Python's lock
throughout (a long call into Qt, or into a library that keeps the lock), is still logged from the gap between two
beats, without a stack. stats() gives the count and the longest stall per tag, for measurements.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from typing import Any

ENV = "IMPACTGUIDE_WATCHDOG"
DEFAULT_MS = 100            # IMPACTGUIDE_WATCHDOG=1: stalls over this long
BEAT_MS = 20                # the window's thread says it is alive this often
LOOK_MS = 10                # the watching thread looks this often
RESAMPLE_MS = 250           # a stall that goes on is looked at again this often
STACK_DEPTH = 18            # frames of each stack in the log
MAX_STACKS = 4              # different stacks kept for one stall

log = logging.getLogger("impactguide.watchdog")


def threshold_from_env(value: str | None) -> float | None:
    """The threshold in ms the environment asks for, or None (off)."""
    value = (value or "").strip()
    if not value or value in ("0", "off", "no", "false"):
        return None
    try:
        n = float(value)
    except ValueError:
        return DEFAULT_MS
    return DEFAULT_MS if n == 1 else max(n, 2 * BEAT_MS)


class Watchdog:
    """The watch itself (start() makes one when the environment asks for it)."""

    def __init__(self, threshold_ms: float, timer_factory: Any, sink: logging.Logger = log) -> None:
        self.threshold = threshold_ms / 1000.0
        self.sink = sink
        self.main_id = threading.get_ident()
        self.beat = time.perf_counter()
        self.tag_now = "start"
        self._lock = threading.Lock()
        self._stacks: list[str] = []
        self._seen_at = 0.0
        self._stats: dict[str, list[float]] = {}
        self._stop = threading.Event()
        self.timer = timer_factory()
        self.timer.setInterval(BEAT_MS)
        self.timer.timeout.connect(self._on_beat)
        self.timer.start()
        self.thread = threading.Thread(target=self._watch, name="ui-watchdog", daemon=True)
        self.thread.start()

    def tag(self, name: str) -> None:
        """What the app is doing now (a page, a flow): stalls are counted under it."""
        self.tag_now = str(name)

    def stats(self) -> dict[str, tuple[int, float]]:
        """{tag: (stalls, longest in ms)} since the start or the last reset()."""
        with self._lock:
            return {k: (len(v), max(v)) for k, v in self._stats.items()}

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()

    def stop(self) -> None:
        self._stop.set()
        self.timer.stop()

    # ── the window's thread ───────────────────────────────────────────────
    def _on_beat(self) -> None:
        now = time.perf_counter()
        gap = now - self.beat
        self.beat = now
        if gap - BEAT_MS / 1000.0 < self.threshold:
            return
        with self._lock:
            stacks, self._stacks, self._seen_at = self._stacks, [], 0.0
            self._stats.setdefault(self.tag_now, []).append(gap * 1000.0)
        where = "".join(f"\n  -- stack {i + 1}\n{s}" for i, s in enumerate(stacks)) or \
            "\n  (no stack: the window's thread kept Python's lock, a long call into Qt or a library)"
        self.sink.warning("window thread stalled %.0f ms during %s%s", gap * 1000.0, self.tag_now, where)

    # ── the watching thread ───────────────────────────────────────────────
    def _watch(self) -> None:
        while not self._stop.wait(LOOK_MS / 1000.0):
            now = time.perf_counter()
            late = now - self.beat - BEAT_MS / 1000.0
            if late < self.threshold:
                continue
            with self._lock:
                if self._seen_at and now - self._seen_at < RESAMPLE_MS / 1000.0:
                    continue
                self._seen_at = now
            frame = sys._current_frames().get(self.main_id)
            if frame is None:
                continue
            text = "".join(traceback.format_stack(frame, limit=STACK_DEPTH))
            with self._lock:
                if text not in self._stacks and len(self._stacks) < MAX_STACKS:
                    self._stacks.append(text)


def start(timer_factory: Any, env: dict[str, str] | None = None) -> Watchdog | None:
    """A watch on the calling thread (the window's), if IMPACTGUIDE_WATCHDOG asks for one; else None.

    Args:
        timer_factory: makes the QTimer of the beats (the app passes QTimer).
        env: the environment to read (os.environ).
    """
    ms = threshold_from_env((env if env is not None else os.environ).get(ENV))
    if ms is None:
        return None
    dog = Watchdog(ms, timer_factory)
    log.warning("watching the window thread: stalls over %.0f ms are logged", ms)
    return dog
