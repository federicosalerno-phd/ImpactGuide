# ImpactGuide

A patient marker fixed to the teeth of a phantom head is struck by a pendulum,
and the augmented reality (AR) registration that relies on that marker has to
be checked and recovered. ImpactGuide runs this experiment: it guides each
trial step by step and collects what the trial produces, from the phone camera
calibration to the live AR overlay, the scans that measure where the marker
went, the analysis and the charts.

**Status: early.** The desktop app, built on
[SlantUI](https://github.com/federicosalerno-phd/SlantUI), is in `04_code/app`:
today it guides the phone calibration and the impact trials; live AR, scans,
analysis, charts and the dataset are the next pages. The code the app runs is
in `04_code/src` and also works on its own. The Blender model of the test bench
is in `02_cad`.

## The bench

- Phantom: printed skull and mandible, mandible locked in occlusion, cast
  silicone face.
- Marker: a 50 x 50 mm ChArUco board on an arm fixed to a dental splint through
  an interchangeable marker mount. A replaceable cap on the mount takes the
  blow, never the board.
- Pendulum: strikes the cap from a chosen direction and drop angle.
- Structured light scanner (EinScan SP V2) and smartphone: both ride on one arm
  that turns around the pendulum pivot, so they see the phantom from several
  angles without being moved by hand.

## Contents

| Path | What it holds |
|---|---|
| `04_code/app` | The ImpactGuide desktop app: the window, the bridge to the code in `04_code/src`, the page (`ui.html`, `ui/`), the words in English and Italian (`ui/strings`), the desktop launcher (`scripts`) and a test that runs whole trials with stand-in files |
| `04_code/src` | Phone camera calibration with a ChArUco board, marker pose from one image, silhouette and edges of a mesh for the AR overlay, synthetic phone views of the bench with their ground truth, evaluation of the marker pose against it, mandible segmentation with SAM 2.1 |
| `04_code/figures/style.py` | Shared matplotlib style for the charts: Abadi, column widths of 90, 140 and 190 mm, vector PDF with embedded fonts and 300 dpi PNG |
| `04_code/tools/build_test_bench.py` | Builds the test bench in Blender from the dimensions in its `P` dictionary (mm) |
| `04_code/tools/build_silicone_face.py` | Builds the silicone face from the cavity of its mould (the mould model is not in this repository yet) |
| `02_cad/blender/test_bench.blend` | The current bench scene, with the alternatives still under evaluation as separate collections (Git LFS) |

`build_test_bench.py` builds the base of the bench; the scene also contains
later modelling steps whose scripts are not published yet. Printable parts will
go in `02_cad/stl` (Git LFS).

## Setup

Python 3.12 and Blender 5.2. With [uv](https://docs.astral.sh/uv/):

```
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r 04_code\requirements.txt --index-strategy unsafe-best-match
```

PyTorch is pinned to the CUDA 12.8 wheels; see the comments in
`04_code/requirements.txt`. Clone with Git LFS installed to get the scene.

A few entry points:

```
python 04_code/src/calibrate_charuco.py board
python 04_code/src/calibrate_charuco.py calibrate PHOTOS --out intrinsics.json
blender --background --python 04_code/tools/build_test_bench.py -- --save
blender --background --factory-startup --python 04_code/src/render_phone_views.py
python 04_code/src/eval_marker_pose.py
```

Each script explains its options in its docstring. Intermediate files go to
`C:\work\marker-impact` by default; the paths are constants at the top of each
script.

## The app

```
python 04_code/app/impactguide_app.py
python 04_code/app/impactguide_app.py --simulate SOME_EMPTY_FOLDER
python 04_code/app/test_app.py [--ui PICTURES_FOLDER]
powershell -ExecutionPolicy Bypass -File 04_code/app/scripts/install_shortcut.ps1
```

The first line opens the app on the project's data (`03_data`); `--simulate`
keeps everything in the given folder and stands in for the phone and the
scanner, so the app can be tried without the bench. `test_app.py` runs whole
trials through the app's backend, and with `--ui` through the real window,
pressing the page's buttons. The last line puts an icon on the desktop.

The trial page follows the seeded sequence of `04_code/src/trials.py`: each
step is confirmed with a button, photos and videos are shot from the PC over
USB, exported scans are picked up from the scanner's export folder, every file
gets its name from the trial and the step, and the log row is written at the
end. Logged values can be corrected later, with a reason, in an audit trail;
raw files are never overwritten. After a closed app the open trial resumes
where it stopped. The phone needs USB debugging on and the camera app open,
with the volume key set as the shutter; `adb` comes with the `adbutils`
package.

## License

Code: MIT, see `LICENSE`. Bench scene, skull, mandible and face models and
printable parts: CC BY 4.0, see `02_cad/LICENSE`.

Copyright (c) 2026 Federico Salerno, 3D Lab, Politecnico di Torino.
