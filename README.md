# ImpactGuide

A patient marker fixed to the teeth of a phantom head is struck by a pendulum,
and the augmented reality (AR) registration that relies on that marker has to
be checked and recovered. ImpactGuide runs this experiment: it guides each
trial step by step and collects what the trial produces, from the phone camera
calibration to the live AR overlay, the scans that measure where the marker
went, the analysis and the charts.

**Status: early.** The desktop app, built on
[SlantUI](https://github.com/federicosalerno-phd/SlantUI), is being written and
will live in `04_code/app`. What is here today is the code the app runs, which
also works on its own, and the Blender model of the test bench.

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

## License

Code: MIT, see `LICENSE`. Bench scene, skull, mandible and face models and
printable parts: CC BY 4.0, see `02_cad/LICENSE`.

Copyright (c) 2026 Federico Salerno, 3D Lab, Politecnico di Torino.
