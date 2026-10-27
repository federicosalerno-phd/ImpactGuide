"""The app's own folders answered from one listing each (impactguide_app.py installs it first thing).

To import a module Python asks each folder of sys.path, in turn, whether it changed since its last look, and once it
has found the source it asks for the file's time and size, to know whether the bytecode it keeps is still good. On a
synced drive each of those questions is a round trip to the drive's process: tens of milliseconds, seconds when the
computer is busy (that process is then starved too), and the app asked about three hundred of them at its start.
Here each of the app's folders (04_code/app and 04_code/src, last on sys.path) is listed once, and a listing carries
the time and size of every file in it: whether a module is there, and whether its bytecode is still good, is answered
from that. Everything else is Python's own: the order of sys.path, the bytecode cache (sys.pycache_prefix), compiling
a source that changed, the other folders. importlib.invalidate_caches() lists the folders again, as it would.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys


class _Loader(importlib.machinery.SourceFileLoader):
    """A source loader that takes the source's time and size from the listing instead of asking the drive again."""

    def __init__(self, fullname: str, path: str, st: os.stat_result):
        super().__init__(fullname, path)
        self._st = st

    def path_stats(self, path):
        if path == self.path:
            return {"mtime": self._st.st_mtime, "size": self._st.st_size}
        return super().path_stats(path)


class FolderFinder:
    """What sys.path_importer_cache keeps for one folder: its modules (name.py), from one listing."""

    def __init__(self, folder: str):
        self.folder = folder
        self.modules: dict[str, tuple[str, os.stat_result]] = {}
        self.invalidate_caches()

    def invalidate_caches(self) -> None:
        found = {}
        with os.scandir(self.folder) as entries:
            for e in entries:
                if e.name.endswith(".py") and e.name[:-3].isidentifier() and e.is_file():
                    found[e.name[:-3]] = (e.path, e.stat())
        self.modules = found

    def find_spec(self, fullname, target=None):
        hit = None if "." in fullname else self.modules.get(fullname)
        if hit is None:
            return None
        return importlib.util.spec_from_file_location(fullname, hit[0], loader=_Loader(fullname, hit[0], hit[1]))


def install(folders) -> list[FolderFinder]:
    """These folders (as written in sys.path) answered from a listing each; a folder that cannot be listed is left to
    Python's own finder."""
    out = []
    for folder in map(str, folders):
        current = sys.path_importer_cache.get(folder)
        if isinstance(current, FolderFinder):
            out.append(current)
            continue
        try:
            finder = FolderFinder(folder)
        except OSError:
            continue
        sys.path_importer_cache[folder] = finder
        out.append(finder)
    return out
