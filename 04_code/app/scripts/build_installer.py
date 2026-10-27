"""The Windows installer of ImpactGuide: a Python of its own with the packages the app runs on, the app, the models
it needs, a shortcut on the desktop and in the Start menu with the app's icon, and an uninstaller.

    python build_installer.py [--out DIR] [--slantui CHECKOUT] [--ai] [--iscc ISCC.exe] [--no-compile]

Free tools only. uv (https://docs.astral.sh/uv/) fetches a standalone CPython 3.12 (python-build-standalone, made to
be moved) and installs into it the packages of 04_code/requirements.txt that the app runs on; Inno Setup 6
(https://jrsoftware.org/isinfo.php: ISCC.exe on the PATH, in its usual folders, or --iscc) packs it with
impactguide.iss. In OUT:

    stage/ImpactGuide/          what the installer puts into the install folder
        python/                 the interpreter with the packages
        04_code/app, 04_code/src, 04_code/figures/style.py, 04_code/requirements.txt
        02_cad/stl/             the plate's measures (charuco_plate_*.json)
        03_data/processed/      the protocol and the targets, when found here: the start copies them into the
                                user's data folder, Documents\\ImpactGuide, where they are missing
        work/                   the models of the live view (face, occluders, mandible), when found in the work folder
        twin/                   the twin's model in use, copied into the user's cache at the first start
        scrcpy/                 scrcpy's server and adb (Apache 2.0, its LICENSE beside it), for the phone's stream
        impactguide_start.pyw   the start (this folder's), impactguide.ico and first.json (the first window's,
                                launcher_assets.py), LICENSE.txt
    ImpactGuide-<version>-setup.exe

Left out: PyTorch, SAM 2 and the network's weights (only the correction of the registration needs them: --ai adds
PyTorch for the CPU and SAM 2; the weights are downloaded apart), the tools that build the bench and the plate
(manifold3d, zstandard). --slantui installs SlantUI from a checkout instead of the commit requirements.txt pins.
After the build, the stage's Python opens the app's module offscreen (a check that every package is there).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent
CODE = APP.parent
ROOT = CODE.parent
WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
LOCAL = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ImpactGuide"
PYTHON = "3.12"
# packages of requirements.txt the app does not run on: the network and its segmenter (--ai), the build tools
LEFT_OUT = {"torch", "torchvision", "sam-2", "manifold3d", "zstandard"}
AI = ["--extra-index-url https://download.pytorch.org/whl/cpu", "torch==2.11.0+cpu", "torchvision==0.26.0+cpu"]
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}
SKIP_FILES = {"desktop.ini", "thumbs.db"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".tmp", ".bak", ".blend1"}
# the models of the live view: the work folder's own (live_ar.FACE_MESH, DESIGN, MANDIBLE_MESH)
WORK_MODELS = ("synth/meshes/face_phantom_r20k.npz", "synth/meshes/occluders.npz", "synth/meshes/meshes.json",
               "targets/mandible.npz")
DATA = ("03_data/processed/protocol", "03_data/processed/targets")
SCRCPY_FILES = ("scrcpy-server", "adb.exe", "AdbWinApi.dll", "AdbWinUsbApi.dll", "LICENSE.txt")
ISCC_PLACES = (Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
               Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
               Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe")
NOTICE = """ImpactGuide
Copyright (c) {year} Federico Salerno
https://github.com/federicosalerno-phd/ImpactGuide

The code of ImpactGuide is licensed under the MIT License (below). The models
it carries (the bench, the skull, mandible and face, the printable parts) are
licensed under the Creative Commons Attribution 4.0 International License,
https://creativecommons.org/licenses/by/4.0/

The Python interpreter (PSF License) and the packages in the python folder
keep their own licenses, found beside each package. scrcpy (in the scrcpy
folder, when present) is licensed under the Apache License 2.0, see its
LICENSE.txt.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def say(text):
    print(text, flush=True)


def run(cmd, **kw):
    say("  > " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def app_version():
    text = (APP / "impactguide_app.py").read_text(encoding="utf-8")
    m = re.search(r'^APP_VERSION = "([^"]+)"', text, re.M)
    return m.group(1) if m else "0.0.0"


def requirements(slantui, ai):
    """The lines of requirements.txt the app runs on (see LEFT_OUT), with SlantUI from a checkout when asked."""
    out = []
    for line in (CODE / "requirements.txt").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("--extra-index-url"):
            continue                        # the CUDA wheels of PyTorch: never in the installer
        name = re.split(r"[\s=<>~!@\[;]", s, 1)[0].lower()
        if name in LEFT_OUT and not (ai and name == "sam-2"):
            continue
        if name == "slantui" and slantui:
            s = f"slantui[shell] @ {Path(slantui).resolve().as_uri()}"
        out.append(s)
    return (AI if ai else []) + out


def copy_tree(src, dst, keep=lambda p: True):
    """src into dst, without caches, Drive's desktop.ini and what keep() refuses."""
    for p in sorted(Path(src).rglob("*")):
        rel = p.relative_to(src)
        if any(part in SKIP_DIRS for part in rel.parts) or p.name.lower() in SKIP_FILES:
            continue
        if p.suffix.lower() in SKIP_SUFFIXES or not keep(p):
            continue
        q = Path(dst) / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def python_into(stage, cache):
    """A standalone CPython of PYTHON, copied into stage/python, without its mark of a managed install (the packages
    go straight into it)."""
    run(["uv", "python", "install", PYTHON, "--install-dir", cache, "--no-bin"])
    found = sorted(Path(cache).glob(f"cpython-{PYTHON}.*-windows-x86_64-none"))
    if not found:
        sys.exit(f"no CPython {PYTHON} in {cache}")
    dst = stage / "python"
    shutil.copytree(found[-1], dst)
    for marker in dst.rglob("EXTERNALLY-MANAGED"):
        marker.unlink()
    return dst / "python.exe"


def packages(py, reqs, out):
    req = out / "requirements_app.txt"
    req.write_text("\n".join(reqs) + "\n", encoding="utf-8")
    env = dict(os.environ, SAM2_BUILD_CUDA="0")
    run(["uv", "pip", "install", "--python", py, "-r", req, "--index-strategy", "unsafe-best-match",
         "--link-mode", "copy", "--compile-bytecode"], env=env)


def the_app(stage):
    copy_tree(APP, stage / "04_code" / "app")
    src = CODE / "src"
    copy_tree(src, stage / "04_code" / "src", keep=lambda p: p.is_dir() or p.suffix == ".py")
    (stage / "04_code" / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CODE / "figures" / "style.py", stage / "04_code" / "figures" / "style.py")
    shutil.copy2(CODE / "requirements.txt", stage / "04_code" / "requirements.txt")
    stl = stage / "02_cad" / "stl"
    stl.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "02_cad" / "stl").glob("charuco_plate_*.json"):
        shutil.copy2(p, stl / p.name)
    for rel in DATA:
        if (ROOT / rel).is_dir():
            copy_tree(ROOT / rel, stage / rel, keep=lambda p: not p.name.endswith((".lock", ".part")))
    shutil.copy2(HERE / "impactguide_start.pyw", stage / "impactguide_start.pyw")


def the_models(stage):
    """The live view's models of the work folder, the twin's model in use, scrcpy: each when found here."""
    said = []
    for rel in WORK_MODELS:
        p = WORK / rel
        if p.is_file():
            q = stage / "work" / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
            said.append(rel)
    cur = LOCAL / "twin" / "current.json"
    if cur.is_file():
        import json
        c = json.loads(cur.read_text(encoding="utf-8"))
        files = [LOCAL / "twin" / c["model"], LOCAL / "twin" / c["info"]]
        if all(f.is_file() for f in files):
            (stage / "twin").mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, stage / "twin" / f.name)
            c["scene"] = ""                                    # no local path in the installer
            (stage / "twin" / "current.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
            said.append(f"twin {c['model']}")
    sys.path.insert(0, str(CODE / "src"))
    try:
        import phone_link
        scr = phone_link.scrcpy_dir()
    except Exception:  # noqa: BLE001  no scrcpy here: the installed app asks for it when the stream starts
        scr = None
    if scr:
        (stage / "scrcpy").mkdir(parents=True, exist_ok=True)
        for name in SCRCPY_FILES:
            if (Path(scr) / name).is_file():
                shutil.copy2(Path(scr) / name, stage / "scrcpy" / name)
        said.append(f"scrcpy {Path(scr).name}")
    return said


def icon_and_notice(stage, py):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    run([py, "-E", "-s", stage / "04_code" / "app" / "scripts" / "launcher_assets.py", stage], env=env)
    (stage / "icon.ico").replace(stage / "impactguide.ico")
    (stage / "theme.json").unlink(missing_ok=True)
    import datetime
    (stage / "LICENSE.txt").write_text(NOTICE.format(year=datetime.date.today().year), encoding="utf-8")


def smoke(stage, py):
    """The app's module opened by the stage's Python alone (no user packages, no variables of this computer)."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("PYTHON", "VIRTUAL_ENV", "CONDA"))}
    env["QT_QPA_PLATFORM"] = "offscreen"
    code = ("import sys; sys.path.insert(0, r'%s'); import impactguide_app, slantui, cv2, av, scipy, matplotlib, "
            "dataset_pack, scan_desk; print('app module', impactguide_app.APP_VERSION, 'slantui', "
            "slantui.__file__)" % (stage / "04_code" / "app"))
    run([py, "-E", "-s", "-c", code], env=env)


def find_iscc(given):
    for p in ([Path(given)] if given else []) + [Path(p) for p in [shutil.which("ISCC") or ""] if p] + list(ISCC_PLACES):
        if p and p.is_file():
            return p
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "impactguide_installer")
    p.add_argument("--slantui", type=Path, help="install SlantUI from this checkout (default: requirements.txt's pin)")
    p.add_argument("--ai", action="store_true", help="also PyTorch for the CPU and SAM 2 (the correction)")
    p.add_argument("--iscc", type=Path, help="Inno Setup's compiler, ISCC.exe")
    p.add_argument("--no-compile", action="store_true", help="lay out the stage only")
    a = p.parse_args()
    out = a.out.resolve()
    stage = out / "stage" / "ImpactGuide"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    version = app_version()
    say(f"ImpactGuide {version}: stage in {stage}")
    py = python_into(stage, out / "pythons")
    packages(py, requirements(a.slantui, a.ai), out)
    the_app(stage)
    for line in the_models(stage):
        say(f"  with {line}")
    icon_and_notice(stage, py)
    smoke(stage, py)
    size = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
    say(f"stage: {size / 1e6:.0f} MB")
    if a.no_compile:
        return 0
    iscc = find_iscc(a.iscc)
    if iscc is None:
        sys.exit("Inno Setup 6 not found: install it (https://jrsoftware.org/isinfo.php) or give --iscc")
    name = f"ImpactGuide-{version}-setup"
    run([iscc, "/Q", f"/DStage={stage}", f"/DOutDir={out}", f"/DOutName={name}", f"/DAppVersion={version}",
         HERE / "impactguide.iss"])
    setup = out / f"{name}.exe"
    say(f"installer: {setup} ({setup.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
