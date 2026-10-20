"""Checks a copy of the ImpactGuide bench dataset: every file of files.csv there, with its size and SHA-256.

It travels in each package as verify.py (dataset_pack.py puts it there) and needs Python 3 and nothing else.

    python verify.py [FOLDER]      FOLDER: the package, by default the folder of this file

Prints one line per file that is missing or differs, and a last line with the count; exit code 1 when any does.
Files not listed in files.csv (files.csv itself, files added later) are said but do not count.
"""
import csv
import hashlib
import sys
from pathlib import Path

BLOCK = 1 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def verify(folder):
    folder = Path(folder)
    listed = folder / "files.csv"
    if not listed.is_file():
        print(f"{listed} not found")
        return 1
    with open(listed, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    bad = 0
    seen = {"files.csv"}
    for i, r in enumerate(rows, 1):
        rel = r["path"]
        seen.add(rel)
        p = folder / rel
        if not p.is_file():
            print(f"missing   {rel}")
            bad += 1
        elif p.stat().st_size != int(r["bytes"]):
            print(f"size      {rel}")
            bad += 1
        elif sha256(p) != r["sha256"].lower():
            print(f"changed   {rel}")
            bad += 1
        if i % 1000 == 0:
            print(f"... {i} of {len(rows)}", flush=True)
    extra = [p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()]
    extra = [e for e in extra if e not in seen and not e.endswith("__pycache__") and "/__pycache__/" not in e
             and Path(e).name.lower() != "desktop.ini"]
    for e in extra[:20]:
        print(f"unlisted  {e}")
    print(f"{len(rows) - bad} of {len(rows)} files as listed" + (f", {bad} not" if bad else "")
          + (f"; {len(extra)} files not in the list" if extra else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(verify(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent))
