"""SAM 2.1 segmentation of the mandible from a box prompt, and GPU benchmark.

The image predictor of SAM 2.1 segments the mandible in a smartphone image,
prompted with a box around the mandible. Checkpoints live in the work folder
(WEIGHTS_DIR: models/sam2 under the environment variable IMPACTGUIDE_WORK),
downloaded from
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_<model>.pt
with <model> in tiny, small, base_plus, large.

Run as a script, it benchmarks the SAM 2.1 models on the phone views rendered
from the Blender bench (02_cad/renders/check_2*_phone_view_*.png): time to
encode an image and to decode one box prompt, peak GPU memory, and agreement of
the masks with those of the large model in fp32 (mean IoU over the views, one
column per prompt). The GPU also drives the display: other programs using it
(Blender, SolidWorks) slow the timings down, so close them first. It writes
05_results/sam2_runtime.csv and, in SHEETS_DIR (sam2 in the work folder), one
sheet per model with the masks and the prompt boxes on the four views. Without
a CUDA device only the CPU run of --cpu is possible.

Usage: python sam2_segment.py [--models tiny small ...] [--repeats N] [--cpu]
"""

import argparse
import csv
import gc
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).resolve().parents[2]
RENDERS_DIR = ROOT / "02_cad" / "renders"
OUTPUT_CSV = ROOT / "05_results" / "sam2_runtime.csv"
WORK_ROOT = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
WEIGHTS_DIR = WORK_ROOT / "models" / "sam2"
SHEETS_DIR = WORK_ROOT / "sam2"

MODELS = {
    "tiny": "sam2.1_hiera_t.yaml",
    "small": "sam2.1_hiera_s.yaml",
    "base_plus": "sam2.1_hiera_b+.yaml",
    "large": "sam2.1_hiera_l.yaml",
}

# Boxes (x0, y0, x1, y1 in pixels of the 1600 x 1200 renders) drawn by hand on
# the phone views: "mandible" around the whole mandible (chin, inferior border,
# ramus), "window" around the bone exposed by the surgical window. In the
# pipeline the box comes from the mandible rendered from the marker pose.
BOXES = {
    1: {"mandible": (765, 420, 1155, 705), "window": (850, 585, 1010, 690)},
    2: {"mandible": (815, 420, 1250, 740), "window": (850, 615, 1070, 700)},
    3: {"mandible": (850, 440, 1335, 780), "window": (920, 650, 1100, 740)},
    4: {"mandible": (860, 470, 1395, 830), "window": (930, 690, 1100, 790)},
}
# Margins added to the boxes: 30% as in the design of the pipeline, 50% as the
# alternative to compare.
PROMPTS = {
    "mandible": ("mandible", 1.0),
    "mandible_30": ("mandible", 1.3),
    "window": ("window", 1.0),
    "window_30": ("window", 1.3),
    "window_50": ("window", 1.5),
}
PROMPT_COLOURS = {
    "mandible": (0, 0, 220), "mandible_30": (0, 140, 255),
    "window": (0, 170, 0), "window_30": (170, 120, 0), "window_50": (170, 0, 170),
}


def enlarge_box(box, factor, shape):
    """Scale a box about its centre, clipped to an image of the given shape."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * factor / 2, (y1 - y0) * factor / 2
    h, w = shape[:2]
    return (max(0, cx - hw), max(0, cy - hh), min(w - 1, cx + hw), min(h - 1, cy + hh))


def prompt_boxes(view, shape):
    return {prompt: enlarge_box(BOXES[view][box], factor, shape) for prompt, (box, factor) in PROMPTS.items()}


def load_predictor(model="tiny", device="cuda"):
    """SAM 2.1 image predictor with the checkpoint of the given model."""
    checkpoint = WEIGHTS_DIR / f"sam2.1_hiera_{model}.pt"
    sam = build_sam2(f"configs/sam2.1/{MODELS[model]}", str(checkpoint), device=device)
    return SAM2ImagePredictor(sam)


def autocast(device, dtype):
    return torch.autocast(device, dtype=torch.float16, enabled=dtype == "fp16")


def segment(predictor, image, box, device="cuda", dtype="fp32"):
    """Mask (bool, H x W) and predicted IoU of the object in the box; image is RGB."""
    with torch.inference_mode(), autocast(device, dtype):
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(box=np.asarray(box, float), multimask_output=False)
    return masks[0] > 0, float(scores[0])


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, repeats):
    """Median time of fn in ms over the given number of repeats, and its last result."""
    times = []
    for _ in range(repeats):
        sync(device)
        t = time.perf_counter()
        result = fn()
        sync(device)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times)), result


def read_views():
    views = {}
    for path in sorted(RENDERS_DIR.glob("check_2*_phone_view_*.png")):
        view = int(path.stem.rsplit("_", 1)[1])
        views[view] = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return views


def benchmark(model, device, dtype, views, repeats):
    """Timing, memory and masks of one model on all views and prompts."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    predictor = load_predictor(model, device)
    load_s = time.perf_counter() - t
    params_m = sum(p.numel() for p in predictor.model.parameters()) / 1e6

    encode_ms, decode_ms, masks, scores = [], [], {}, {}
    with torch.inference_mode(), autocast(device, dtype):
        first = next(iter(views.values()))
        for _ in range(2):  # warm-up
            predictor.set_image(first)
            predictor.predict(box=np.asarray(BOXES[1]["mandible"], float), multimask_output=False)
        for view, image in views.items():
            ms, _ = timed(lambda: predictor.set_image(image), device, repeats)
            encode_ms.append(ms)
            for prompt, box in prompt_boxes(view, image.shape).items():
                ms, (m, s, _) = timed(
                    lambda: predictor.predict(box=np.asarray(box, float), multimask_output=False),
                    device, repeats)
                decode_ms.append(ms)
                masks[view, prompt] = m[0] > 0
                scores[view, prompt] = float(s[0])

    row = {
        "model": model,
        "device": device,
        "dtype": dtype,
        "params_m": round(params_m, 1),
        "load_s": round(load_s, 2),
        "encode_ms": round(float(np.median(encode_ms)), 1),
        "decode_ms": round(float(np.median(decode_ms)), 1),
    }
    row["image_ms"] = round(row["encode_ms"] + row["decode_ms"], 1)
    row["images_per_s"] = round(1000 / row["image_ms"], 2)
    if device == "cuda":
        row["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        row["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
    del predictor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return row, masks, scores


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def save_sheet(name, views, masks, scores):
    """Masks and boxes for every prompt (rows) and view (columns), cropped on the mandible."""
    x0, y0, x1, y1 = 600, 300, 1600, 1000
    scale = 0.45
    rows = []
    for prompt in PROMPTS:
        tiles = []
        for view, image in views.items():
            tile = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mask = masks[view, prompt]
            colour = np.array(PROMPT_COLOURS[prompt], np.uint8)
            tile[mask] = (0.5 * tile[mask] + 0.5 * colour).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(tile, contours, -1, PROMPT_COLOURS[prompt], 2)
            bx0, by0, bx1, by1 = (int(round(v)) for v in prompt_boxes(view, image.shape)[prompt])
            cv2.rectangle(tile, (bx0, by0), (bx1, by1), (0, 0, 0), 3)
            tile = cv2.resize(tile[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            label = f"view {view}  {prompt}  score {scores[view, prompt]:.2f}"
            cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SHEETS_DIR / f"sheet_{name}.png"), np.vstack(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cpu", action="store_true", help="also time the tiny model on the CPU")
    args = parser.parse_args()

    views = read_views()
    if not views:
        raise SystemExit(f"no phone views check_2*_phone_view_*.png in {RENDERS_DIR}")
    cuda = torch.cuda.is_available()
    if cuda:
        free, total = torch.cuda.mem_get_info()
        print(f"{torch.cuda.get_device_name()}: {free / 2**20:.0f} of {total / 2**20:.0f} MB free")
    elif not args.cpu:
        raise SystemExit("no CUDA device: run with --cpu for the CPU timing of the tiny model")
    else:
        free = None
        print("no CUDA device: CPU run only")
    runs = [(m, "cuda", d) for m in args.models for d in ("fp32", "fp16")] if cuda else []
    if args.cpu:
        runs.append(("tiny", "cpu", "fp32"))

    rows, all_masks = [], {}
    for model, device, dtype in runs:
        repeats = args.repeats if device == "cuda" else 2
        try:
            row, masks, scores = benchmark(model, device, dtype, views, repeats)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"{model} {dtype}: out of GPU memory")
            continue
        row["gpu_free_mb_before"] = round(free / 2**20) if free is not None else ""
        rows.append(row)
        all_masks[model, device, dtype] = masks
        save_sheet(f"{model}_{device}_{dtype}", views, masks, scores)
        print(row)

    reference = all_masks.get(("large", "cuda", "fp32"))
    for row in rows:
        masks = all_masks[row["model"], row["device"], row["dtype"]]
        for prompt in PROMPTS:
            row[f"iou_{prompt}"] = round(float(np.mean(
                [iou(masks[view, prompt], reference[view, prompt]) for view in views]
            )), 3) if reference else ""
    if not rows:
        raise SystemExit("no model ran: nothing written")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(max(rows, key=len)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written {OUTPUT_CSV} and sheets in {SHEETS_DIR}")


if __name__ == "__main__":
    main()
