"""The words of the page in its six languages (ui/strings/<lang>.json), checked against each other and the page.

    python check_strings.py [--strict]

One line per check, "ok" or "FAIL" (run_tests.py counts them); exit code 1 on a failure:
- every file reads as JSON, an object of texts;
- every language has the keys of English, and none more;
- every text has the same placeholders ({0}, {1} ...) as its English one;
- every key the page names whole (data-t, data-t-title, data-t-ph and data-t-aria in ui.html; T, hasT and Tcode with a
  whole key, a string literal, in ui/*.js) is in English.
Keys no part of the app names, whole or by one of the prefixes the page builds keys from (a literal that ends with a
dot, as T('home.cal.m.' + mode)), are listed as "warn": with --strict they fail.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
STRINGS = APP / "ui" / "strings"
LANGS = ("en", "it", "fr", "es", "de", "zh")
PLACEHOLDER = re.compile(r"\{(\d+)\}")
HTML_KEY = re.compile(r"data-t(?:-title|-ph|-aria)?=\"([^\"]+)\"")
JS_CALL = re.compile(r"\b(?:T|hasT|Tcode)\(\s*'([^']+)'\s*([,)+])")
MARKUP_PREFIX = re.compile(r"data-t(?:-title|-ph|-aria)?=\"([a-z][a-z0-9_.]*\.)'")    # markup built in a script
LITERAL = re.compile(r"""'([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*\.?)'|"([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*\.?)\"""")
CODE_FILES = ("ui/*.js", "ui.html", "*.py")     # where the app names its keys (the backend: the phone's page, the
#                                                 loading screen)
SOURCES = (APP.parent / "src" / "phone_view.py",)


def load(fails: list[str]) -> dict[str, dict[str, str]]:
    out = {}
    for lang in LANGS:
        p = STRINGS / f"{lang}.json"
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            fails.append(f"{p.name}: not JSON ({e})")
            continue
        if not isinstance(d, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in d.items()):
            fails.append(f"{p.name}: not an object of texts")
            continue
        out[lang] = d
    return out


def code_texts() -> list[tuple[str, str]]:
    """(file name, text) of every file of the app that may name a key."""
    files = sorted({p for pattern in CODE_FILES for p in APP.glob(pattern)} | {p for p in SOURCES if p.is_file()})
    return [(p.name, p.read_text(encoding="utf-8", errors="replace")) for p in files if p.name != Path(__file__).name]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strict", action="store_true", help="keys no part of the app names fail too")
    args = ap.parse_args(argv)
    results = []

    def check(ok: bool, what: str) -> None:
        results.append(ok)
        print(("ok    " if ok else "FAIL  ") + what, flush=True)

    bad: list[str] = []
    words = load(bad)
    check(not bad and set(words) == set(LANGS), "six string files, each an object of texts" + (": " + "; ".join(bad)
                                                                                                  if bad else ""))
    en = words.get("en", {})
    for lang, d in words.items():
        if lang == "en":
            continue
        missing, extra = sorted(set(en) - set(d)), sorted(set(d) - set(en))
        check(not missing and not extra, f"{lang}: the keys of English ({len(missing)} missing, {len(extra)} more)"
              + (f": missing {missing[:8]}" if missing else "") + (f" more {extra[:8]}" if extra else ""))
        wrong = sorted(k for k in set(en) & set(d)
                       if sorted(set(PLACEHOLDER.findall(en[k]))) != sorted(set(PLACEHOLDER.findall(d[k]))))
        check(not wrong, f"{lang}: the placeholders of English in every text ({len(wrong)} differ)"
              + (f": {wrong[:8]}" if wrong else ""))
    texts = code_texts()
    named, prefixes, literals = set(), set(), set()
    for name, text in texts:
        if name.endswith(".html"):
            named |= set(HTML_KEY.findall(text))
        if name.endswith(".js"):
            prefixes |= set(MARKUP_PREFIX.findall(text))
            for key, after in JS_CALL.findall(text):
                if after == "+" or key.endswith("."):
                    prefixes.add(key)
                else:
                    named.add(key)
        for a, b in LITERAL.findall(text):
            lit = a or b
            (prefixes if lit.endswith(".") else literals).add(lit)
    absent = sorted(k for k in named if k not in en)
    check(not absent, f"every key the page names whole is in English ({len(named)} keys)"
          + (f": not there {absent[:10]}" if absent else ""))
    used = named | literals
    unused = sorted(k for k in en if k not in used and not any(k.startswith(p) for p in prefixes))
    if unused:
        line = f"{len(unused)} keys no part of the app names: {unused[:20]}"
        if args.strict:
            check(False, line)
        else:
            print("warn  " + line, flush=True)
    else:
        check(True, "every key named by some part of the app")
    print(f"\n{results.count(False)} failures" if not all(results) else "\nall checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
