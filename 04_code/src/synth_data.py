"""Reader, augmentations and checks of the digital twin dataset written by synth_twin.py (T13.1, module 3a).

    python 04_code/src/synth_data.py index  [--root DIR]
        OUT/index.csv: one row per image (session, split, face, interface, state, view, mode, motion, pixel counts)
    python 04_code/src/synth_data.py check  [--root DIR] [--sessions s0900] [--renderer numpy|gl|both]
        renderer of module 1 (mesh_overlay.py, and gl_overlay.py) against the exact masks of Blender: the decimated face
        rendered at the true pose T_C_B against <image>_model.png (undeformed face alone, same camera); IoU, outline
        distance in px and mm, time of a ~200 mm crop at 256 x 256 (with edges, and for gl also mask and buffer
        only) and of the whole image. Writes
        05_results/twin_overlay.csv (one row per renderer) and OUT/check_overlay.csv (one row per image).

Library: sessions(root), frames(session_dir), load(session_dir, frame) -> dict with rgb (BGR uint8), labels (uint8),
masks (face, marker, drape: bool), depth (mm, NaN where nothing is hit) and the poses as arrays; augment(img, rng) for
the photometric augmentations of module 3 (exposure and white balance, defocus and motion blur up to 3 px, noise,
JPEG); crop_K(K, centre, side_px, out_px) for the ~200 mm crops.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]
OUT = Path(r"C:\work\marker-impact\synth")
RESULTS = ROOT / "05_results"
LABELS = {"background": 0, "face": 1, "marker": 2, "drape": 3, "interface": 4, "head_spheres": 5, "head": 6,
          "pendulum": 7, "bench": 8, "cap": 9}
CROP_MM = 200.0          # side of the crop on the face (ai_design.md, T12.0 proposals)
CROP_PX = 256


def sessions(root=OUT):
    return sorted(p for p in (Path(root) / "sessions").glob("s[0-9][0-9][0-9][0-9]") if (p / "frames.jsonl").exists())


def frames(session_dir):
    return [json.loads(line) for line in (Path(session_dir) / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line]


def load(session_dir, frame):
    d = Path(session_dir)
    stem = d / frame["image"]
    lab = cv2.imread(str(stem) + "_label.png", cv2.IMREAD_GRAYSCALE)
    out = {"rgb": cv2.imread(str(stem) + ".jpg", cv2.IMREAD_COLOR), "labels": lab,
           "masks": {k: lab == LABELS[k] for k in ("face", "marker", "drape")}}
    dp = Path(str(stem) + "_depth.png")
    if dp.exists():
        z = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.float32) / 16.0
        z[z >= 65535 / 16.0 - 1e-3] = np.nan
        out["depth"] = z
    for k in ("K", "T_C_B", "T_C_M", "T_M_B", "T_M_B0", "dT"):
        out[k] = np.array(frame[k], np.float64)
    out["dist"] = np.array(frame["dist"], np.float64)
    return out


def augment(img, rng, max_blur_px=3.0):
    """Photometric augmentation of a BGR uint8 image (module 3): exposure, white balance, defocus or motion blur,
    sensor noise, JPEG. Geometry is untouched, so masks and poses stay valid."""
    x = img.astype(np.float32) / 255.0
    x = np.clip(x ** rng.uniform(0.8, 1.25) * rng.uniform(0.7, 1.3) * rng.uniform(0.92, 1.08, 3), 0, 1)
    b = rng.uniform(0.0, max_blur_px)
    if b > 0.3:
        if rng.random() < 0.5:
            x = cv2.GaussianBlur(x, (0, 0), b / 2)
        else:
            n = max(int(round(b)) * 2 + 1, 3)
            k = np.zeros((n, n), np.float32)
            a = rng.uniform(0, np.pi)
            c = n // 2
            for t in np.linspace(-b, b, 4 * n):
                k[int(round(c + t * np.sin(a))), int(round(c + t * np.cos(a)))] = 1
            x = cv2.filter2D(x, -1, k / k.sum())
    x = x + rng.normal(0, rng.uniform(0.002, 0.02), x.shape).astype(np.float32)
    x = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def crop_K(K, centre, side_px, out_px=CROP_PX):
    """Intrinsics of a square crop of side side_px centred at centre (px), resampled to out_px (pixel centres kept)."""
    s = out_px / side_px
    x0, y0 = centre[0] - side_px / 2, centre[1] - side_px / 2
    Kc = np.array(K, np.float64).copy()
    Kc[0, 0] *= s
    Kc[1, 1] *= s
    Kc[0, 2] = (K[0, 2] - x0 + 0.5) * s - 0.5
    Kc[1, 2] = (K[1, 2] - y0 + 0.5) * s - 0.5
    return Kc


def cmd_index(args):
    rows = []
    for sd in sessions(args.root):
        for f in frames(sd):
            rows.append({"session": f["session"], "split": f["split"], "face": f["face"], "interface": f["interface"],
                         "image": f"{sd.name}/{f['image']}", "registration": f["registration"], "state": f["state"],
                         "view": f["view"], "mode": f["mode"], "motion": f["motion"]["kind"],
                         "deg": round(f["motion"]["deg"], 3), "mm": round(f["motion"]["mm"], 3),
                         "target_mm": None if f.get("target_shift_mm") is None else f["target_shift_mm"]["mean"],
                         "deform_mm": f["deformation"]["amp_mm"], "drape": f["drape"] or "",
                         **{f"px_{k}": v for k, v in f["pixels"].items()}})
    if not rows:
        sys.exit("no sessions")
    with (Path(args.root) / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"images": len(rows), "sessions": len({r["session"] for r in rows}),
                      "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}}))


def outline_distance(a, b):
    """Symmetric distances (px) between the outlines of two binary masks."""
    def outline(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0
    oa, ob = outline(a), outline(b)
    if not oa.any() or not ob.any():
        return np.array([np.inf])
    da = cv2.distanceTransform((~oa).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~ob).astype(np.uint8), cv2.DIST_L2, 5)
    return np.r_[db[oa], da[ob]]


def timed(fn, n):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def cmd_check(args):
    sys.path.insert(0, str(SRC))
    import mesh_overlay
    root = Path(args.root)
    z = np.load(root / "meshes" / "face_phantom_r20k.npz")
    v, f = z["vertices"].astype(np.float64), z["faces"].astype(np.int64)
    renderers = {}
    if args.renderer in ("numpy", "both"):
        renderers["numpy"] = mesh_overlay.MeshRenderer(v, f)
    if args.renderer in ("gl", "both"):
        import gl_overlay
        renderers["gl"] = gl_overlay.GLMeshRenderer(v, f)
    rows = []
    for name in args.sessions:
        sd = root / "sessions" / name
        for fr in frames(sd):
            model = Path(sd / (fr["image"] + "_model.png"))
            if not model.exists():
                continue
            ref = cv2.imread(str(model), cv2.IMREAD_GRAYSCALE) == LABELS["face"]
            K, T, size = np.array(fr["K"]), np.array(fr["T_C_B"]), tuple(fr["size"])
            depth = float(np.median((v @ T[:3, :3].T + T[:3, 3])[:, 2]))
            mm_per_px = depth / K[0, 0]
            ys, xs = np.nonzero(ref)
            centre = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
            side = CROP_MM / mm_per_px
            Kc = crop_K(K, centre, side)
            # the reference crop: the full-resolution reference resampled by nearest pixel centre
            for rname, rnd in renderers.items():
                r = rnd.render(K, None, T, size)
                m = r.mask > 0
                inter, union = (m & ref).sum(), (m | ref).sum()
                d = outline_distance(m, ref)
                ms_crop = timed(lambda: rnd.render(Kc, None, T, (CROP_PX, CROP_PX)), args.repeat)
                ms_full = timed(lambda: rnd.render(K, None, T, size), max(args.repeat // 5, 2))
                ms_mask = (timed(lambda: rnd.render(Kc, None, T, (CROP_PX, CROP_PX), edges=False), args.repeat)
                           if rname == "gl" else ms_crop)      # mask and buffer only (correction)
                ys2, xs2 = np.nonzero(m)
                rows.append({"renderer": rname, "image": f"{name}/{fr['image']}", "mode": fr["mode"],
                             "width": size[0], "height": size[1], "iou": inter / union, "xor_px": int((m ^ ref).sum()),
                             "outline_mean_px": float(d.mean()), "outline_p95_px": float(np.percentile(d, 95)),
                             "outline_max_px": float(d.max()), "mm_per_px": mm_per_px,
                             "centroid_dx_px": float(xs2.mean() - xs.mean()), "centroid_dy_px": float(ys2.mean() - ys.mean()),
                             "crop_ms": ms_crop, "crop_mask_ms": ms_mask, "full_ms": ms_full})
                print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rows[-1].items()}), flush=True)
    if not rows:
        sys.exit("no images with _model.png: render with --model-mask")
    with (root / "check_overlay.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for rname in renderers:
        rr = [r for r in rows if r["renderer"] == rname]
        summary.append({"renderer": rname, "images": len(rr), "triangles": len(f),
                        "iou_min": round(min(r["iou"] for r in rr), 5),
                        "outline_mean_px": round(statistics.mean(r["outline_mean_px"] for r in rr), 3),
                        "outline_p95_px": round(max(r["outline_p95_px"] for r in rr), 3),
                        "outline_max_px": round(max(r["outline_max_px"] for r in rr), 3),
                        "outline_p95_mm": round(max(r["outline_p95_px"] * r["mm_per_px"] for r in rr), 4),
                        "centroid_shift_px": round(max(abs(r["centroid_dx_px"]) + abs(r["centroid_dy_px"]) for r in rr), 3),
                        "crop256_ms": round(statistics.median(r["crop_ms"] for r in rr), 2),
                        "crop256_mask_ms": round(statistics.median(r["crop_mask_ms"] for r in rr), 2),
                        "full_ms": round(statistics.median(r["full_ms"] for r in rr), 1)})
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "twin_overlay.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("index", "check"))
    p.add_argument("--root", default=str(OUT))
    p.add_argument("--sessions", nargs="+", default=["s0900"])
    p.add_argument("--renderer", choices=("numpy", "gl", "both"), default="numpy")
    p.add_argument("--repeat", type=int, default=20)
    args = p.parse_args()
    {"index": cmd_index, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    main()
