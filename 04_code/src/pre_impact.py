"""Pre-impact sessions: marker A moved by hand to new positions, each one scanned and photographed, without an impact;
the real images that train and validate the correction, with their true registration.

Sets and positions. A set is one session of POSITIONS positions of the marker (8 at least): the first from the jig,
every other one moved by hand to somewhere new and let go (the joint holds it). The panels change from one position
to the next (LIGHTS, all four in every set). SETS sets at least (the dataset is split by set: two for training, one
for validation; trials.Config.repetitions holds the number of sets). Position ids m001, m002... in execution order.

Files, in 03_data/raw/pre_impact/ (the layout and the writing of trials.py: log, audit trail, manifest, lock):
    trial_log.csv            one row per position: set, position, light, session, times, status, notes
    m001/
        m001_ref_scan_az40.ply   the scans of the position (arm positions 1 and 2, panels off): its registration
        m001_ref_scan_az55.ply
        m001_photo_az40.jpg      photos from the 4 arm positions, panels as the position's light
        ... m001_photo_az85.jpg
        m001_video.mp4           video from the arm, end stop to end stop
Configuration and sequence in 03_data/processed/protocol (config_pre_impact.json, sequence_pre_impact.csv).

ImpactGuide drives a set like the impact trials (trials.Run): trials.Run("pre_impact") is a PreImpactRun, whose steps
are position_steps (no pendulum, no impact); the page Trials shows the set, the position and the light.

Processing (process). For each position closed: its registration from its two scans (ground_truth.measure: T_M_mandible
and T_M_B), its turn from the set's first position; the marker's pose in each photo and in every video_step-th video
frame (marker_pose.py: ChArUco and PnP with the intrinsics of the mode), the reprojection error (the board's check,
REPROJ_PX), the camera in the head frame (T_C_B = T_C_M T_M_B) and the reference mask of the face rendered there
(mesh_overlay.py, the face in B). Written to 03_data/processed/pre_impact: positions.csv (one row per position),
images.csv (one row per image: pose, error, valid or why not, mask), sets.csv (per set: positions, turns, images),
and <position>/<image>_mask.png.

Usage:
    python pre_impact.py sequence [--sets N]        configuration and sequence of the sets
    python pre_impact.py check [--deep]             what is missing for each position
    python pre_impact.py process [--video-step N] [--face NPZ] [--no-masks]
    python pre_impact.py test [--work DIR]          a simulated set through the log and the processing; exit 1 on failure
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trials  # noqa: E402

PHASE = trials.PRE_IMPACT
ROOT = trials.ROOT
PROCESSED = ROOT / "03_data" / "processed"
OUT = PROCESSED / "pre_impact"
CALIBRATION = PROCESSED / "calibration"
WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
FACE_NPZ = WORK / "synth" / "meshes" / "face_phantom_r20k.npz"     # the face in B (synth_twin.py meshes)
POSITIONS = 8                   # positions of the marker per set
SETS = 3                        # sets at least: two for training, one for validation
LIGHTS = trials.LIGHTS          # full, half, left_off, right_off
VIDEO_STEP = 10                 # every 10th frame of the video (3 a second at 30 fps)
REPROJ_PX = 1.5                 # above: the board is not as printed (bent, scratched) or the pose is wrong
MIN_CORNERS = 8
PRE = trials.Config(phase=PHASE, holders=("A",), angles_deg=(), directions_deg=(), repetitions=SETS, drape_trials=0,
                    order="ascending", direction_run=1, seed=0, washers_per_side=0, highspeed=False)
SEQUENCE_FIELDS = ("slot", "holder", "replicate", "position", "light", "change")
REQUIRED = ("session", "holder", "set", "position", "light", "time_start", "time_end")


# ── the sets ────────────────────────────────────────────────────────────────
def make_sequence(cfg: trials.Config) -> list[dict]:
    """The slots: replicate = set, POSITIONS positions each, the lights in turn (each set starts one further on)."""
    rows, k = [], 0
    for s in range(1, cfg.repetitions + 1):
        for p in range(1, POSITIONS + 1):
            k += 1
            rows.append({"slot": k, "holder": cfg.holders[0], "replicate": s, "position": p,
                         "light": LIGHTS[(p - 1 + s - 1) % len(LIGHTS)],
                         "change": "reposition" if p == 1 else "position;light"})
    return rows


def position_steps(cfg: trials.Config, row: dict, prev: dict | None = None) -> list[trials.Step]:
    """Steps of one position: the marker placed, the panels set, the two scans, the four photos, the video, close."""
    t = row["trial"]
    first = str(row.get("position", "")) in ("", "1")
    s = []
    if first:
        s.append(trials.Step("reposition", "Put marker A in its start position with the jig, then take the jig off."))
    else:
        s.append(trials.Step("marker_move", "Move marker A by hand to a new position, unlike the ones before, "
                                            "and let go: the joint holds it."))
    s.append(trials.Step("light", f"Set the panels: {row.get('light') or LIGHTS[0]}."))
    for az in cfg.scan_az:
        s.append(trials.Step(f"ref_scan_az{az}", f"Scan: arm at {az}°, panels off. Scan and export.",
                             (trials.file_name(t, "scan", az, ref=True),)))
    for az in cfg.photo_az:
        s.append(trials.Step(f"photo_az{az}", f"Photo: arm at {az}°, panels on. Shoot from the PC.",
                             (trials.file_name(t, "photo", az),)))
    a, b = cfg.video_az
    s.append(trials.Step("video", f"Video: lift the index plunger and sweep the arm from end stop to end stop "
                                  f"({a}° to {b}°) in about {round((b - a) / trials.VIDEO_SPEED)} s, then back to "
                                  f"{cfg.photo_az[0]}°.", (trials.file_name(t, "video"),)))
    s.append(trials.Step("close", "Notes, then close the position.", fields=("time_end",)))
    return s


class PreImpactRun(trials.Run):
    """The pre-impact sets driven as trials.Run drives the impacts: one log row and one folder per position."""

    pendulum = False                # no swing mass to weigh, no hole, no direction
    required = REQUIRED
    online = ()

    def __init__(self, phase: str = PHASE, raw: Path | str = trials.RAW, protocol: Path | str = trials.PROTOCOL,
                 calibration: Path | str | None = None):
        super().__init__(PHASE, raw, protocol, calibration)

    sequence_fields = SEQUENCE_FIELDS
    sequence_numeric = ("slot", "replicate", "position")

    def _make_sequence(self, cfg: trials.Config, done) -> list[dict]:
        return make_sequence(cfg)

    def next(self, today: date | None = None, reference: str | None = None, settings: dict | None = None) -> dict | None:
        """The open position (resume) or a plan for the next pending slot; None when every set is done."""
        rows = self.log.rows()
        open_rows = [r for r in rows if r["status"] == trials.OPEN]
        if open_rows:
            r = open_rows[0]
            return {**r, "resume": True, "step": self.resume_step(r["trial"]), "changes": [], "warnings": [],
                    "warning_codes": [{"code": "several_open", "trials": [x["trial"] for x in open_rows]}]
                    if len(open_rows) > 1 else []}
        pending = self.pending()
        if not pending:
            return None
        s = pending[0]
        trial = self.new_id()
        plan = {"trial": trial, "phase": PHASE, "session": (today or date.today()).isoformat(), "holder": s["holder"],
                "set": str(s["replicate"]), "position": str(s["position"]), "light": s["light"], "drape": "0",
                "angle_deg": "", "direction_deg": "", "washers_per_side": "", "swing_mass_g": "", "slot": s["slot"],
                "reference": "", "pre_scan": f"{trial}_ref"}
        prev = rows[-1] if rows else None
        plan.update(resume=False, step=0, changes=[c for c in ("set", "position", "light")
                                                   if prev is None or str(prev.get(c)) != plan[c]],
                    warnings=[], warning_codes=[])
        return plan

    def confirm(self, trial: str, key: str, when: datetime | None = None, **values: object) -> None:
        """As trials.Run.confirm; a position has no second hit and no cap to replace."""
        if key == "close":
            values = {k: v for k, v in values.items() if k not in ("second_hit", "cap_replaced")}
            values.update(second_hit="", cap_replaced="")
        super().confirm(trial, key, when, **values)


# ── processing ──────────────────────────────────────────────────────────────
def load_intrinsics(mode: str, folder: Path = CALIBRATION):
    """K, dist and image size of a mode (photo, video) from calibrate_charuco.py's file; None when there is none."""
    p = Path(folder) / f"intrinsics_{mode}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return np.asarray(d["K"], np.float64), np.asarray(d.get("dist") or np.zeros(5), np.float64).ravel(), \
        tuple(d.get("image_size") or d.get("size") or ())


def scale_K(K: np.ndarray, size_from, size_to) -> np.ndarray:
    """Intrinsics of a calibration at another image size (the same sensor crop)."""
    if not size_from or tuple(size_from) == tuple(size_to):
        return K
    sx, sy = size_to[0] / size_from[0], size_to[1] / size_from[1]
    K = K.copy()
    K[0] *= sx
    K[1] *= sy
    return K


def board_pose(image, board, detector, K, dist):
    """The board's pose in one image: {T_C_M, rms_px, corners, ambiguity} or a reason it has none."""
    import marker_pose as mp
    corners, ids, n_markers = mp.detect(detector, image)
    if len(ids) < MIN_CORNERS:
        return {"reason": "board", "corners": int(len(ids))}
    est = mp.estimate_pose(board, corners, ids, K, dist, min_corners=MIN_CORNERS)
    if est is None:
        return {"reason": "pose", "corners": int(len(ids))}
    out = {"T_C_M": est["T"], "rms_px": est["rms_px"], "corners": int(est["n"]), "ambiguity": est["ambiguity"]}
    if est["rms_px"] > REPROJ_PX:
        out["reason"] = "reprojection"
    return out


def flat(T) -> str:
    return " ".join(f"{x:.6f}" for x in np.asarray(T).ravel()) if T is not None else ""


def process(raw: Path = trials.RAW, protocol: Path = trials.PROTOCOL, out: Path = OUT, face: Path = FACE_NPZ,
            intrinsics: dict | None = None, video_step: int = VIDEO_STEP, masks: bool = True, log=print) -> dict:
    """Every closed position: registration, poses, errors, masks; the three tables in out (see the module's doc).
    intrinsics: {mode: (K, dist, size)} in place of the calibration's files."""
    import cv2
    import ground_truth as gt
    import marker_pose as mp
    run = trials.Run(PHASE, raw, protocol)
    rows = [r for r in run.log.rows() if r["status"] == trials.OK]
    board = mp.make_board()
    renderer = None
    if masks:
        if not Path(face).exists():
            log(f"{face} not found: no masks")
            masks = False
        else:
            import mesh_overlay
            m = np.load(face)
            renderer = mesh_overlay.MeshRenderer(m["vertices"].astype(np.float64), m["faces"].astype(np.int64))
    intr = dict(intrinsics or {})
    for mode in ("photo", "video"):
        if mode not in intr:
            v = load_intrinsics(mode)
            if v is not None:
                intr[mode] = v
    out.mkdir(parents=True, exist_ok=True)
    pos_rows, img_rows, firsts = [], [], {}
    for r in rows:
        t = r["trial"]
        folder = run.folder / t
        scans = [p for az in run.config.scan_az for p in run.find(t, trials.file_name(t, "scan", az, ref=True))]
        pr = {"position": t, "set": r["set"], "number": r["position"], "light": r["light"], "session": r["session"]}
        T_MB = None
        try:
            res = gt.measure(scans)
            T_MB, T_Mm = res["T_M_B"], res["T_M_mandible"]
            pr.update(registration="ok", scans=len(res["views_used"]), calibrated=int(res["calibrated"]),
                      warnings="; ".join(res["warnings"]), T_M_mandible=flat(T_Mm), T_M_B=flat(T_MB))
            first = firsts.setdefault(r["set"], T_Mm)
            d = gt.displacement(first, T_Mm)
            pr.update(turn_deg=f"{d['rot_deg']:.3f}", tre_from_first_mm=f"{d['tre_mean_mm']:.3f}")
        except (ValueError, OSError) as e:
            pr.update(registration="failed", warnings=str(e)[:300])
        images = []
        for az in run.config.photo_az:
            for p in run.find(t, trials.file_name(t, "photo", az)):
                images.append(("photo", p, None, cv2.imread(str(p))))
        for p in run.find(t, trials.file_name(t, "video")):
            cap = cv2.VideoCapture(str(p))
            k = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if k % max(video_step, 1) == 0:
                    images.append(("video", p, k, img))
                k += 1
            cap.release()
        n_valid = 0
        for kind, p, k, img in images:
            name = p.stem + ("" if k is None else f"_f{k:04d}")
            ir = {"position": t, "set": r["set"], "light": r["light"], "image": name, "file": p.name,
                  "frame": "" if k is None else k, "kind": kind}
            if img is None:
                img_rows.append({**ir, "valid": 0, "reason": "unreadable"})
                continue
            h, w = img.shape[:2]
            ir.update(width=w, height=h)
            if kind not in intr:
                img_rows.append({**ir, "valid": 0, "reason": "intrinsics"})
                continue
            K0, dist, size0 = intr[kind]
            K = scale_K(K0, size0, (w, h))
            det = mp.make_detector(board, K, dist)
            bp = board_pose(img, board, det, K, dist)
            ir.update(corners=bp.get("corners", ""), rms_px=f"{bp['rms_px']:.3f}" if "rms_px" in bp else "",
                      T_C_M=flat(bp.get("T_C_M")))
            if T_MB is None and "reason" not in bp:
                bp["reason"] = "registration"
            if "reason" in bp:
                img_rows.append({**ir, "valid": 0, "reason": bp["reason"]})
                continue
            T_CB = bp["T_C_M"] @ T_MB
            ir.update(T_C_B=flat(T_CB), valid=1, reason="")
            if renderer is not None:
                mask = renderer.render(K, dist, T_CB, (w, h)).mask
                mp_path = out / t / f"{name}_mask.png"
                mp_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mp_path), mask)
                ir.update(mask=f"{t}/{mp_path.name}", mask_px=int(np.count_nonzero(mask)))
            n_valid += 1
            img_rows.append(ir)
        pr.update(images=len(images), valid=n_valid)
        pos_rows.append(pr)
        log(f"{t}: set {r['set']}, position {r['position']}, {pr['registration']}, {n_valid}/{len(images)} images")
    set_rows = []
    for s in sorted({p["set"] for p in pos_rows}, key=lambda x: int(float(x))):
        ps = [p for p in pos_rows if p["set"] == s]
        turns = [float(p["turn_deg"]) for p in ps if p.get("turn_deg")]
        set_rows.append({"key": f"set_{s}", "positions": len(ps), "registered": sum(p["registration"] == "ok" for p in ps),
                         "lights": len({p["light"] for p in ps}), "turn_max_deg": f"{max(turns):.2f}" if turns else "",
                         "turn_median_deg": f"{np.median(turns[1:]):.2f}" if len(turns) > 1 else "",
                         "images": sum(p["images"] for p in ps), "valid": sum(p["valid"] for p in ps),
                         "enough": int(len(ps) >= POSITIONS)})
    pcols = ["position", "set", "number", "light", "session", "registration", "scans", "calibrated", "turn_deg",
             "tre_from_first_mm", "images", "valid", "T_M_mandible", "T_M_B", "warnings"]
    icols = ["position", "set", "light", "image", "file", "frame", "kind", "width", "height", "corners", "rms_px",
             "valid", "reason", "T_C_M", "T_C_B", "mask", "mask_px"]
    scols = ["key", "positions", "registered", "lights", "turn_max_deg", "turn_median_deg", "images", "valid", "enough"]
    trials.write_csv(out / "positions.csv", pos_rows, pcols)
    trials.write_csv(out / "images.csv", img_rows, icols)
    trials.write_csv(out / "sets.csv", set_rows, scols)
    return {"positions": pos_rows, "images": img_rows, "sets": set_rows}


# ── checks ──────────────────────────────────────────────────────────────────
def synth_set(work: Path, sessions: Path, regs=(0, 1, 2), log=print) -> dict:
    """A set of len(regs) positions in work from the twin's images (synth_twin.py render): each registration of the
    first session becomes a position with its four photos, a video of its frames and two simulated scans."""
    import cv2
    import ground_truth as gt
    sdir = sorted(p for p in sessions.iterdir() if (p / "frames.jsonl").exists())[0]
    frames = [json.loads(x) for x in (sdir / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    raw, protocol = work / "raw", work / "protocol"
    run = trials.Run(PHASE, raw, protocol)
    T_W_mandible, _ = gt.scene_poses()
    T_B_m = gt.load_frame()
    truth, clock = {}, datetime(2026, 10, 27, 9, 0)
    for i, reg in enumerate(regs):
        fr = [f for f in frames if f["registration"] == reg and f["state"] == 0]
        plan = run.next(today=clock.date())
        t = plan["trial"]
        run.start(plan, when=clock)
        steps = run.steps(t)
        T_MB = np.asarray(fr[0]["T_M_B"], np.float64)
        T_Mm = T_MB @ T_B_m
        T_W_M = T_W_mandible @ np.linalg.inv(T_Mm)
        clouds, _ = gt.synthetic_scan(T_W_M, T_W_mandible, np.random.default_rng(10 + i), merged=False)
        photos = [f for f in fr if f["mode"] == "photo"]
        video = [f for f in fr if f["mode"] == "video"]
        tmp = work / "tmp" / t
        tmp.mkdir(parents=True, exist_ok=True)
        vpath = tmp / "video.mp4"
        size = tuple(video[0]["size"])
        wr = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
        for f in video:
            wr.write(cv2.imread(str(sdir / f"{f['image']}.jpg")))
        wr.release()
        sources = {}
        for k, az in enumerate(run.config.scan_az):
            p = tmp / f"scan_{az}.ply"
            gt.write_ply(p, clouds[k % len(clouds)])
            sources[f"ref_scan_az{az}"] = p
        for k, az in enumerate(run.config.photo_az):
            sources[f"photo_az{az}"] = sdir / f"{photos[k]['image']}.jpg"
        sources["video"] = vpath
        for st in steps:
            if st.files:
                run.store(t, st.files[0], sources[st.key])
            run.confirm(t, st.key, when=clock)
        truth[t] = {"T_M_B": T_MB, "photos": photos, "video": video, "dir": sdir}
        clock = clock.replace(minute=clock.minute + 5)
    intr = {"photo": (np.asarray(photos[0]["K"]), np.zeros(5), tuple(photos[0]["size"])),
            "video": (np.asarray(video[0]["K"]), np.zeros(5), tuple(video[0]["size"]))}
    return {"run": run, "raw": raw, "protocol": protocol, "truth": truth, "intrinsics": intr}


def run_test(work: Path) -> int:
    """A set through the log (steps, files, close, the next plan, the sequence, the completeness check) and, with the
    twin's images, through the processing (registration, poses, masks against the twin's truth); exit 1 on failure."""
    import cv2
    import ground_truth as gt
    if work.exists():
        if any(work.iterdir()) and not (work / trials.TEST_MARKER).exists():
            raise SystemExit(f"{work} is not empty and not a folder of an earlier test")
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / trials.TEST_MARKER).write_text("pre_impact\n", encoding="utf-8")
    fails = []

    def expect(ok, what):
        print(("  ok   " if ok else "  FAIL ") + what, flush=True)
        if not ok:
            fails.append(what)

    # the log and the steps, with fake files
    raw, protocol = work / "log" / "raw", work / "log" / "protocol"
    run = trials.Run(PHASE, raw, protocol)
    expect(type(run).__name__ == "PreImpactRun" and type(trials.Run("campaign", raw, protocol)) is trials.Run,
           "trials.Run('pre_impact') is a PreImpactRun, the impacts keep trials.Run")
    expect(len(run.sequence) == SETS * POSITIONS and run.sequence[0]["light"] == LIGHTS[0]
           and {r["light"] for r in run.sequence[:POSITIONS]} == set(LIGHTS),
           f"{SETS} sets of {POSITIONS} positions, every light in a set")
    plan = run.next(today=date(2026, 10, 27))
    expect(plan["trial"] == "m001" and plan["set"] == "1" and plan["position"] == "1" and str(plan["slot"]) == "1",
           f"first plan: {plan['trial']}, set {plan['set']}, position {plan['position']}")
    keys = [s.key for s in trials.trial_steps(run.config, {k: plan.get(k, "") for k in trials.LOG_FIELDS})]
    want = ["reposition", "light", "ref_scan_az40", "ref_scan_az55", "photo_az40", "photo_az55", "photo_az70",
            "photo_az85", "video", "close"]
    expect(keys == want, f"steps of a first position: {', '.join(keys)}")
    run.start(plan)
    fake = work / "fake"
    fake.mkdir()
    for n, data in (("a.ply", b"ply\n"), ("a.jpg", b"\xff\xd8jpg"), ("a.mp4", b"mp4")):
        (fake / n).write_bytes(data)
    for st in run.steps("m001"):
        if st.files:
            run.store("m001", st.files[0], fake / ("a" + Path(st.files[0]).suffix))
        run.confirm("m001", st.key)
    row = run.log.row("m001")
    names = sorted(p.name for p in (raw / PHASE / "m001").iterdir())
    expect(row["status"] == trials.OK and row["second_hit"] == "" and row["light"] == LIGHTS[0] and len(names) == 7,
           f"m001 closed with its 7 files in its folder ({names[0]} ...), no pendulum values")
    p2 = run.next(today=date(2026, 10, 27))
    k2 = [s.key for s in trials.trial_steps(run.config, {k: p2.get(k, "") for k in trials.LOG_FIELDS})]
    expect(p2["trial"] == "m002" and p2["position"] == "2" and p2["light"] == LIGHTS[1] and k2[0] == "marker_move",
           f"next: {p2['trial']}, position {p2['position']}, light {p2['light']}, first step {k2[0]}")
    filled, extra = trials.match(run.sequence, run.log.rows())
    rep = run.check()
    expect({int(k): v for k, v in filled.items()} == {1: "m001"} and not extra and not rep["trials"][0]["problems"] and len(rep["pending"]) == 23,
           f"sequence: slot 1 filled by m001, 23 pending, nothing missing")
    run.configure(trials.replace(run.config, repetitions=4))
    expect(len(run.sequence) == 4 * POSITIONS and list(trials.match(run.sequence, run.log.rows())[0].values()) == ["m001"],
           "four sets: 32 slots, m001 still in slot 1")
    camp = trials.Run("campaign", raw, protocol)
    expect(len(camp.sequence) == 128 and camp.next()["trial"] == "t001", "the campaign is untouched (128 slots)")

    # processing on the twin's images
    sessions = WORK / "synth" / "sessions"
    if not FACE_NPZ.exists() or not sessions.exists():
        print(f"  skip processing: no twin dataset in {sessions}")
    else:
        sim = synth_set(work / "proc", sessions, log=lambda *_: None)
        res = process(sim["raw"], sim["protocol"], work / "proc" / "out", intrinsics=sim["intrinsics"], video_step=5,
                      log=lambda *_: None)
        errs_reg, errs_pose, ious = [], [], []
        for pr in res["positions"]:
            T = np.array([float(x) for x in pr["T_M_B"].split()]).reshape(4, 4)
            tr = sim["truth"][pr["position"]]
            d = gt.displacement(tr["T_M_B"] @ gt.load_frame(), T @ gt.load_frame())
            errs_reg.append(d["tre_max_mm"])
        for ir in res["images"]:
            if not ir.get("valid"):
                continue
            tr = sim["truth"][ir["position"]]
            if ir["kind"] == "photo":
                az = int(ir["file"].split("_az")[1].split(".")[0])
                rec = tr["photos"][list(sim["run"].config.photo_az).index(az)]
            else:
                rec = tr["video"][int(ir["frame"])]
            T = np.array([float(x) for x in ir["T_C_M"].split()]).reshape(4, 4)
            errs_pose.append(np.linalg.norm(T[:3, 3] - np.asarray(rec["T_C_M"])[:3, 3]))
            lab = cv2.imread(str(tr["dir"] / f"{rec['image']}_label.png"), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(work / "proc" / "out" / ir["mask"]), cv2.IMREAD_GRAYSCALE) > 0
            face = lab == 1
            ious.append(np.logical_and(mask, face).sum() / max(np.logical_or(mask, face).sum(), 1))
        n_valid = sum(int(ir.get("valid") or 0) for ir in res["images"])
        expect(len(res["positions"]) == 3 and all(p["registration"] == "ok" for p in res["positions"])
               and max(errs_reg) < 0.1, f"registration of 3 positions from their scans: worst error at the targets "
                                        f"{max(errs_reg):.3f} mm")
        expect(n_valid >= 0.8 * len(res["images"]) and np.median(errs_pose) < 1.5 and max(errs_pose) < 5.0,
               f"marker pose in {n_valid} of {len(res['images'])} images: {np.median(errs_pose):.2f} mm from the truth "
               f"(median), worst {max(errs_pose):.2f}")
        expect(np.median(ious) > 0.85, f"face masks from the registration against the twin's labels: IoU median "
                                       f"{np.median(ious):.3f}, worst {min(ious):.3f} (the label is the face as seen, "
                                       f"under its cover and occluders)")
        expect(res["sets"][0]["positions"] == 3 and res["sets"][0]["enough"] == 0,
               f"sets.csv: set 1 with 3 positions, said too few (need {POSITIONS})")
    print(f"{'FAIL' if fails else 'ok'}: pre_impact, {len(fails)} failed", flush=True)
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sq = sub.add_parser("sequence")
    sq.add_argument("--sets", type=int, default=SETS)
    ck = sub.add_parser("check")
    ck.add_argument("--deep", action="store_true")
    pr = sub.add_parser("process")
    pr.add_argument("--video-step", type=int, default=VIDEO_STEP)
    pr.add_argument("--face", type=Path, default=FACE_NPZ)
    pr.add_argument("--no-masks", action="store_true")
    t = sub.add_parser("test")
    t.add_argument("--work", type=Path, default=WORK / "pre_impact_test")
    a = ap.parse_args(argv)
    if a.cmd == "test":
        return run_test(a.work)
    run = trials.Run(PHASE)
    if a.cmd == "sequence":
        run.configure(trials.replace(run.config, repetitions=a.sets))
        print(f"{len(run.sequence)} positions in {a.sets} sets: {run.sequence_path}")
        return 0
    if a.cmd == "check":
        rep = run.check(a.deep)
        bad = 0
        for r in rep["trials"]:
            if r["problems"]:
                bad += 1
                print(f"{r['trial']} ({r['status']}): " + "; ".join(f"{k} {v}" for k, v in r["problems"]))
        print(f"{len(rep['trials'])} positions, {bad} with problems, {len(rep['pending'])} to do")
        return 1 if bad else 0
    res = process(video_step=a.video_step, face=a.face, masks=not a.no_masks)
    for s in res["sets"]:
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
