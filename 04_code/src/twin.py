"""The digital twin of the bench for ImpactGuide: the 3D model kept in step with the bench scene, the setting of a
trial for it, the marker's motion as the scanner measures it, and whether the synthetic dataset and the network were
made on the scene as it stands.

Model. The bench scene (scene_path()) is the only source: nothing goes back to it. Once it has not been saved for
QUIET_S seconds and differs from the scene of the model in use, Watcher copies it, and a Blender in the background
(twin_export.py, at low priority) makes the model from the copy: CACHE/bench_<hash>.glb with its extras in
bench_<hash>.json, and current.json says which one is in use, with its version number (one more at each new model),
the fingerprint of what the phone sees (phone_print) and the scene it came from. The KEEP newest models stay.

Versions. synth_data.py index writes the fingerprint of the scene the dataset was rendered on (index.json,
scene_print; synth_twin.py render records it per session) and refiner.py train the one of the dataset it learned from
(refiner.json beside the weights). versions() compares them with the model's: current, old (another fingerprint, or
made before fingerprints were written) or none (nothing made yet).

Trial. setting() reads a row of the trial log (trials.py): holder (A or B), hole and direction of the pendulum, the arm
position of a step (its _azNN, else the first photo position). scan_result() gives the marker's motion of a trial from
its post-impact scans and the scans of before (the log's pre_scan), measured with ground_truth.measure (kept in
CACHE/scans), as a world motion D of the marker (the head stays: D = T_W_mandible dT^-1 T_W_mandible^-1) and the TRE
at each target; in simulation, when the stand-in scans hold no spheres, a motion drawn from the trial (A turns about
its ball, more for a higher hole; B comes back within a tenth of a millimetre), said as simulated.

    python twin.py test          checks of the watcher, versions, setting and scan results (and of twin_export.py in
                                 Blender when Blender is found); exit code 1 on failure
    python twin.py export        exports the scene now, into the app's cache
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]
if str(SRC) not in sys.path:        # the app keeps it last, away from the libraries' imports (impactguide_app.py)
    sys.path.insert(0, str(SRC))
WORK = Path(os.environ.get("IMPACTGUIDE_WORK", r"C:\work\marker-impact"))
LOCAL = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "ImpactGuide"
CACHE = LOCAL / "twin"
EXPORTER = SRC / "twin_export.py"
DATASET_INDEX = WORK / "synth" / "index.json"                 # synth_data.py index
NETWORK_LOG = WORK / "models" / "refiner" / "refiner.json"    # refiner.py train, beside refiner.pt
# the scanner at each arm position as the scene has it, written at each export (read by ground_truth.py's simulated
# scans)
SCANNER_POSES = ROOT / "03_data" / "processed" / "protocol" / "scanner_poses.json"
RIG_DESIGN_NAME = "rig_design.json"     # beside it: the reference spheres and the board's frame (ground_truth.py)
QUIET_S = 120           # the scene is exported once it has not been saved for this long
RETRY_S = 600           # an export that failed is tried again on the same scene after this long
EXPORT_TIMEOUT_S = 900
KEEP = 3                # models kept in the cache
CURRENT = "current.json"
# the phone's photos when no calibration has been saved: Poco X7 Pro, 4080 x 3072, f about 3070 px (synth_twin.NOMINAL)
NOMINAL_PHOTO = ((4080, 3072), 3070.0)
A_DEG_PER_HOLE = (1.0 / 50, 3.5 / 50)   # simulated A: turn about the ball, degrees per degree of the hole (min, max)
B_MM, B_DEG = (0.02, 0.12), (0.02, 0.10)  # simulated B: back in its seats within this much
STAND_IN_BYTES = 4096                   # a scan smaller than this is the simulated scanner's stand-in
BELOW_NORMAL = 0x00004000               # Windows: the export runs below the app's priority
NO_WINDOW = 0x08000000


# ── where things are ─────────────────────────────────────────────────────────
def scene_path() -> Path:
    """The bench scene: IMPACTGUIDE_BENCH, else the scene being worked on (WORK/test_bench_draft.blend), else the one
    of the repository (02_cad/blender/test_bench.blend)."""
    env = os.environ.get("IMPACTGUIDE_BENCH")
    if env:
        return Path(env)
    draft = WORK / "test_bench_draft.blend"
    return draft if draft.exists() else ROOT / "02_cad" / "blender" / "test_bench.blend"


def blender_path() -> Path | None:
    """Blender: IMPACTGUIDE_BLENDER, else the newest under Program Files, else the one on the PATH."""
    env = os.environ.get("IMPACTGUIDE_BLENDER")
    if env:
        return Path(env) if Path(env).exists() else None
    base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"
    found = sorted(base.glob("Blender */blender.exe"),
                   key=lambda p: [int(x) if x.isdigit() else 0 for x in p.parent.name.split()[-1].split(".")])
    if found:
        return found[-1]
    which = shutil.which("blender")
    return Path(which) if which else None


def read_json(path: Path) -> dict | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def write_json(path: Path, data: dict) -> None:
    """Written whole beside its place and put there in one step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


# ── the model ────────────────────────────────────────────────────────────────
class Busy(Exception):
    """The scene was being saved while it was copied: tried again later."""


def run_blender(blender: Path, copy: Path, out: Path, timeout: float = EXPORT_TIMEOUT_S,
                started: callable = None) -> None:
    """twin_export.py on the copy, in a Blender of its own in the background, below the app's priority."""
    env = dict(os.environ)
    env.setdefault("PYTHONPYCACHEPREFIX", str(LOCAL / "pycache"))
    flags = (BELOW_NORMAL | NO_WINDOW) if os.name == "nt" else 0
    cmd = [str(blender), "--background", "--factory-startup", str(copy), "--python-exit-code", "1",
           "--python", str(EXPORTER), "--", "export", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                         creationflags=flags)
    if started is not None:
        started(p)
    try:
        text, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        raise RuntimeError(f"export took more than {timeout:.0f} s") from None
    if p.returncode != 0 or not (out / "bench.glb").exists():
        tail = (text or b"").decode("utf-8", "replace").strip().splitlines()[-12:]
        raise RuntimeError(f"export failed ({p.returncode}): " + " | ".join(tail))


class Watcher:
    """The model kept in step with the scene. due() and run() are asked by the app (run() in a worker thread);
    current() says the model in use.

    Args:
        cache: where the models are kept.
        scene: the bench scene (scene_path() by default).
        blender: Blender's executable (blender_path() by default); None: no export, the models of the cache only.
        quiet_s: how long the scene must stay unsaved before it is exported.
        clock: the time now, in seconds (tests move it by hand).
        export: export(copy, out) writes out/bench.glb and out/bench.json (by default Blender with twin_export.py).
        seed: a cache whose newest model is taken when this one has none for the same scene (the app's own, for a
            simulation: no second export of the same scene).
        poses: where an export's scanner poses are written (SCANNER_POSES; None: nowhere).
    """

    def __init__(self, cache: Path | str = CACHE, scene: Path | str | None = None, blender: Path | None | bool = True,
                 quiet_s: float = QUIET_S, clock=time.time, export=None, seed: Path | str | None = None,
                 poses: Path | str | None = SCANNER_POSES) -> None:
        self.cache = Path(cache)
        self.poses = Path(poses) if poses is not None else None
        self.scene = Path(scene) if scene is not None else scene_path()
        self.blender = blender_path() if blender is True else blender
        self.quiet_s, self.clock = quiet_s, clock
        self._export = export
        self._seen = None           # (mtime, size) at the last look
        self._failed = {}           # (mtime, size) -> time of the failure
        self._proc = None
        self._lock = threading.Lock()
        self.running = False
        self.last_error = ""
        if seed is not None:
            self.adopt_from(Path(seed))

    # -- the scene and the model in use
    def stat(self) -> tuple[float, int] | None:
        try:
            st = self.scene.stat()
        except OSError:
            return None
        return st.st_mtime, st.st_size

    def current(self) -> dict | None:
        """The model in use (current.json, with its files present), or None."""
        cur = read_json(self.cache / CURRENT)
        if not cur or not (self.cache / str(cur.get("model", ""))).is_file():
            return None
        return cur

    def info(self, cur: dict | None = None) -> dict | None:
        """The extras of the model in use (bench_<hash>.json): parts, cameras, targets, fingerprint."""
        cur = cur or self.current()
        return read_json(self.cache / cur["info"]) if cur else None

    def can_export(self) -> bool:
        return bool(self._export or (self.blender and Path(self.blender).exists()))

    def due(self) -> bool:
        """Whether the scene should be exported now: it exists, differs from the scene of the model in use, has not
        been saved for quiet_s and has not changed since the last look (a scene being saved is never copied)."""
        st = self.stat()
        seen, self._seen = self._seen, st
        if st is None or self.running or not self.can_export():
            return False
        cur = self.current()
        if cur and [cur.get("scene_mtime"), cur.get("scene_size")] == [st[0], st[1]]:
            return False
        if self.clock() - st[0] < self.quiet_s or seen != st:
            return False
        failed = self._failed.get(st)
        return failed is None or self.clock() - failed >= RETRY_S

    # -- the export
    def run(self) -> dict:
        """Copies the scene, makes the model from the copy and puts it in use; returns current.json (the same model
        when the scene changed in nothing the model shows). Raises Busy when the scene was saved meanwhile."""
        with self._lock:
            self.running = True
        st = self.stat()
        t0 = time.monotonic()
        work = Path(tempfile.mkdtemp(prefix="export_", dir=self._work_dir()))
        try:
            copy = work / "scene.blend"
            shutil.copy2(self.scene, copy)
            if self.stat() != st:
                raise Busy("the scene was saved while it was copied")
            out = work / "out"
            if self._export is not None:
                self._export(copy, out)
            else:
                run_blender(self.blender, copy, out, started=lambda p: setattr(self, "_proc", p))
            cur = self._adopt(out, st, time.monotonic() - t0)
            self.last_error = ""
            return cur
        except Busy:
            raise
        except Exception as e:
            self._failed[st] = self.clock()
            self.last_error = str(e)[:400]
            raise
        finally:
            self._proc = None
            self.running = False
            shutil.rmtree(work, ignore_errors=True)
            try:
                work.parent.rmdir()         # the copies' folder, once empty
            except OSError:
                pass

    def _work_dir(self) -> Path:
        d = self.cache / "work"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _adopt(self, out: Path, st: tuple[float, int], seconds: float) -> dict:
        glb = (out / "bench.glb").read_bytes()
        info = read_json(out / "bench.json") or {}
        h = hashlib.sha256(glb).hexdigest()[:12]
        cur = self.current()
        name = f"bench_{h}"
        if not (self.cache / f"{name}.glb").exists():
            tmp = self.cache / f"{name}.glb.part"
            tmp.write_bytes(glb)
            os.replace(tmp, self.cache / f"{name}.glb")
        info.update(scene=str(self.scene), scene_mtime=st[0], scene_size=st[1])
        write_json(self.cache / f"{name}.json", info)
        same = bool(cur and cur.get("hash") == h)
        new = {"version": (cur or {}).get("version", 0) + (0 if same else 1), "hash": h, "model": f"{name}.glb",
               "info": f"{name}.json", "phone_print": info.get("phone_print", ""), "scene": str(self.scene),
               "scene_mtime": st[0], "scene_size": st[1],
               "scene_saved": datetime.fromtimestamp(st[0]).astimezone().isoformat(timespec="seconds"),
               "exported": datetime.now().astimezone().isoformat(timespec="seconds"),
               "export_s": round(seconds, 1), "bytes": len(glb), "triangles": info.get("triangles")}
        if same:
            new["exported"] = cur.get("exported", new["exported"])
        write_json(self.cache / CURRENT, new)
        if self.poses is not None and info.get("scanner_poses"):
            write_json(self.poses, {
                "scene": self.scene.name, "scene_saved": new["scene_saved"], "version": new["version"], "hash": h,
                "written": datetime.now().astimezone().isoformat(timespec="seconds"),
                "frame": "the bench's world, mm: x towards the patient's left, -y anterior, z up",
                "centroid": info.get("spheres_centroid"), "positions": info["scanner_poses"]})
        if self.poses is not None and info.get("rig_design"):
            # the six spheres and the board's frame as the scene has them: ground_truth.py's rig templates
            write_json(self.poses.with_name(RIG_DESIGN_NAME), {
                "scene": self.scene.name, "scene_path": str(self.scene), "scene_saved": new["scene_saved"],
                "scene_mtime": st[0],
                "version": new["version"], "hash": h, "written": datetime.now().astimezone().isoformat(timespec="seconds"),
                "frame": "the bench's world, mm", "T_W_mandible": info.get("T_W_mandible"), **info["rig_design"]})
        self._prune(new)
        return new

    def _prune(self, cur: dict) -> None:
        models = sorted(self.cache.glob("bench_*.glb"), key=lambda p: p.stat().st_mtime, reverse=True)
        keep = {cur["model"]} | {p.name for p in models[:KEEP]}
        for p in models:
            if p.name not in keep:
                for f in (p, p.with_suffix(".json")):
                    try:
                        f.unlink()
                    except OSError:
                        pass

    def adopt_from(self, other: Path) -> bool:
        """The model in use in another cache taken as this one's when this one has none (or one of another scene);
        True when one was taken."""
        theirs = read_json(other / CURRENT)
        if not theirs or other.resolve() == self.cache.resolve():
            return False
        mine = self.current()
        if mine and mine.get("scene_mtime") == theirs.get("scene_mtime"):
            return False
        try:
            self.cache.mkdir(parents=True, exist_ok=True)
            for k in ("model", "info"):
                shutil.copy2(other / theirs[k], self.cache / theirs[k])
        except (OSError, KeyError):
            return False
        write_json(self.cache / CURRENT, theirs)
        return True

    def cancel(self) -> None:
        """A running export stopped (the app closing)."""
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass


# ── versions ─────────────────────────────────────────────────────────────────
def made_on(path: Path, marker: Path | None = None) -> str | None:
    """The fingerprint a dataset index or a network log records; '' when the file exists without one (made before
    fingerprints were written), None when there is nothing (no file, nor the marker file of what it describes)."""
    d = read_json(path)
    if d is None:
        return "" if marker is not None and marker.exists() else None
    p = d.get("scene_print")
    return p if isinstance(p, str) else ""


def versions(phone_print: str, dataset: Path | None = None, network: Path | None = None) -> dict:
    """Whether the dataset and the network were made on the scene of the model: current, old or none each, and
    stale (one of them old). By default the dataset of DATASET_INDEX and the network of NETWORK_LOG."""
    dataset, network = dataset or DATASET_INDEX, network or NETWORK_LOG
    out = {}
    for key, path, marker in (("dataset", dataset, None), ("network", network, Path(network).with_suffix(".pt"))):
        p = made_on(Path(path), marker)
        out[key] = "none" if p is None else ("current" if p and p == phone_print else "old")
        out[key + "_print"] = p or ""
    out["stale"] = bool(phone_print) and "old" in (out["dataset"], out["network"])
    return out


# ── the phone's field of view ──────────────────────────────────────────────
def lens_of(d: dict | None) -> dict | None:
    """A saved calibration as the twin's phone view bends its picture with it: the image size, K (3 x 3) and the
    distortion (k1 k2 p1 p2 k3, and k4 k5 k6 of the rational model when calibrated, as OpenCV has them), its
    vertical field of view (degrees) and aspect; None when it is not a usable calibration."""
    try:
        (w, h), K = d["image_size"], [[float(x) for x in row] for row in d["K"]]
        dist = [float(x) for x in np.ravel(np.asarray(d.get("dist") or [], dtype=float))]
        if not (w > 0 and h > 0 and K[0][0] > 0 and K[1][1] > 0) or not all(map(math.isfinite, dist + K[0] + K[1])):
            return None
    except (TypeError, KeyError, IndexError, ValueError, AttributeError):
        return None
    dist = (dist + [0.0] * 8)[:8]
    return {"size": [int(w), int(h)], "K": K, "dist": dist,
            "fov_y": round(math.degrees(2 * math.atan(h / 2 / K[1][1])), 3), "aspect": round(w / h, 4)}


def phone_view(calib: Path | None) -> dict:
    """Vertical field of view (degrees) and aspect of the phone's photos: the saved calibration, else nominal; and
    the lens of each mode the twin shows (photo, video: lens_of, None without its calibration)."""
    lens = {m: lens_of(read_json(Path(calib) / f"intrinsics_{m}.json") if calib else None) for m in ("photo", "video")}
    if lens["photo"]:
        (w, h), fy, src = lens["photo"]["size"], lens["photo"]["K"][1][1], "calibration"
    else:
        (w, h), fy, src = NOMINAL_PHOTO[0], NOMINAL_PHOTO[1], "nominal"
    return {"fov_y": round(math.degrees(2 * math.atan(h / 2 / fy)), 3), "aspect": round(w / h, 4), "source": src,
            "lens": lens}


# ── the setting of a trial ──────────────────────────────────────────────────
def arm_of(step_key: str | None, photo_az) -> int:
    """The arm position (camera azimuth, degrees) a step is taken at; the first photo position for any other."""
    import re
    m = re.search(r"_az(\d+)$", step_key or "")
    return int(m.group(1)) if m else int(photo_az[0])


def setting(row: dict, step_key: str | None = None, photo_az=(40, 55, 70, 85)) -> dict:
    """What the twin shows of a trial: holder, hole, direction, arm position, drape."""
    def num(v, default=0.0):
        try:
            return float(str(v).replace(",", "."))
        except ValueError:
            return default
    return {"trial": row.get("trial", ""), "holder": str(row.get("holder") or "A"),
            "angle_deg": num(row.get("angle_deg")), "direction_deg": num(row.get("direction_deg")),
            "drape": str(row.get("drape", "")) == "1", "arm_az": arm_of(step_key, photo_az), "step": step_key or "",
            "status": row.get("status", "")}


# ── the marker's motion from the scans ──────────────────────────────────────
def rot(axis, deg):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def inverse(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def scans_of(run, trial: str, ref: bool = False) -> list[Path]:
    import trials
    out = []
    try:        # the arm positions of the trial's own settings (config_version), not today's
        cfg = run.config_for(run.log.row(trial))
    except (trials.NotFound, AttributeError):       # a trial not logged, or a stand-in run without a log
        cfg = run.config
    for az in cfg.scan_az:
        found = run.find(trial, trials.file_name(trial, "scan", az, ref=ref))
        if not found:
            return []
        out.append(found[0])
    return out


def reference_scans(run, row: dict) -> list[Path]:
    """The scans of before the impact: the reference of the trial (pre_scan <trial>_ref) or the post-impact scans
    of the trial it names."""
    pre = str(row.get("pre_scan") or "")
    if pre.endswith("_ref"):
        return scans_of(run, pre[:-4], ref=True)
    return scans_of(run, pre) if pre else []


def file_key(paths: list[Path]) -> list:
    return [[p.name, p.stat().st_size, round(p.stat().st_mtime, 3)] for p in paths]


def scan_key(post: list[Path], ref: list[Path], rig: Path | None = None) -> dict:
    """What a measured result is kept by: the scans it comes from and the rig calibration it was measured with
    (ground_truth.CALIBRATION_JSON when rig is None; none yet: an empty list), so that the scans are measured again
    with the calibrated radii once the calibration is made or made again."""
    if rig is None:
        import ground_truth as gt
        rig = gt.CALIBRATION_JSON
    rig = Path(rig)
    return {"post": file_key(post), "ref": file_key(ref), "rig": file_key([rig]) if rig.is_file() else []}


def world_motion(dT: np.ndarray, T_W_mandible: np.ndarray) -> np.ndarray:
    """The marker's motion in the world from dT (mandible frame): the head stays, the marker moves by dT^-1."""
    return T_W_mandible @ inverse(dT) @ inverse(T_W_mandible)


def summary(D: np.ndarray, dT: np.ndarray, info: dict, simulated: bool, extra: dict) -> dict:
    targets = info.get("targets") or []
    P = np.array([t["mandible"] for t in targets], np.float64).reshape(-1, 3)
    tre = np.linalg.norm(P @ dT[:3, :3].T + dT[:3, 3] - P, axis=1) if len(P) else np.zeros(0)
    c = np.asarray(info.get("board_centre") or [0, 0, 0], np.float64)
    from scipy.spatial.transform import Rotation
    out = {"state": "ready", "simulated": simulated, "D": D.round(6).tolist(),
           "rot_deg": round(float(np.degrees(Rotation.from_matrix(D[:3, :3]).magnitude())), 4),
           "shift_mm": round(float(np.linalg.norm(D[:3, :3] @ c + D[:3, 3] - c)), 4),
           "tre": [{"id": t["id"], "name": t.get("name", t["id"]), "mm": round(float(e), 4)}
                   for t, e in zip(targets, tre)],
           "tre_max_mm": round(float(tre.max()), 4) if len(tre) else None,
           "tre_mean_mm": round(float(tre.mean()), 4) if len(tre) else None}
    out.update(extra)
    return out


def simulated_motion(row: dict, info: dict) -> np.ndarray:
    """A motion of the marker drawn from the trial (the same trial, the same motion): A turns about its ball by up
    to 3.5 degrees at the highest hole, B comes back within a tenth of a millimetre."""
    seed = int(hashlib.sha256(str(row.get("trial", "")).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis[2] *= 0.4                                   # mostly about axes across the impact
    hole = abs(float(str(row.get("angle_deg") or 30).replace(",", ".")))
    if str(row.get("holder", "A")).upper() == "B":
        c = np.asarray(info.get("board_centre") or [0, 0, 0], np.float64)
        R = rot(axis, rng.uniform(*B_DEG))
        s = rng.normal(size=3)
        s *= rng.uniform(*B_MM) / np.linalg.norm(s)
        return rigid(R, c - R @ c + s)
    c = np.asarray(info.get("ball_centre") or info.get("board_centre") or [0, 0, 0], np.float64)
    R = rot(axis, hole * rng.uniform(*A_DEG_PER_HOLE))
    return rigid(R, c - R @ c)


def scan_result(run, row: dict, info: dict | None, cache: Path, simulate: bool = False, measure: bool = True) -> dict:
    """The marker's motion of a trial after its scans: {"state": "none"} before them; "ready" with D (world motion
    of the marker, 4 x 4), rot_deg, shift_mm (of the board's centre), tre (mm at each target), tre_max_mm; "failed"
    with why (no_reference, no_spheres); "measuring" when the scans are still to be measured and measure is False
    (the app measures them in a worker). Measured results are kept in cache (scans/<phase>_<trial>.json) by the
    files they come from and the rig calibration (scan_key); calibrated says whether the spheres had their
    calibrated radii."""
    if info is None:
        return {"state": "no_model"}
    trial = row.get("trial", "")
    post = scans_of(run, trial)
    if not post:
        return {"state": "none"}
    ref = reference_scans(run, row)
    T = np.asarray(info.get("T_W_mandible") or np.eye(4), np.float64)
    stand_in = all(p.stat().st_size < STAND_IN_BYTES for p in post + ref)
    if simulate and stand_in:
        D = simulated_motion(row, info)
        dT = inverse(T) @ inverse(D) @ T          # D = T dT^-1 T^-1
        return summary(D, dT, info, True, {"scans": [p.name for p in post], "reference": row.get("pre_scan", "")})
    if not ref:
        return {"state": "failed", "why": "no_reference"}
    key = scan_key(post, ref)
    kept = Path(cache) / "scans" / f"{run.phase}_{trial}.json"
    old = read_json(kept)
    if old and old.get("key") == key:
        dT = np.asarray(old["dT"], np.float64)
        calibrated = bool(old.get("calibrated", key["rig"]))
    elif not measure:
        return {"state": "measuring"}
    else:
        import ground_truth as gt
        try:
            before = gt.measure([str(p) for p in ref])
            after = gt.measure([str(p) for p in post], reference=before)
        except ValueError:
            return {"state": "failed", "why": "no_spheres"}
        dT = np.asarray(after["displacement"]["dT"], np.float64)
        calibrated = bool(after.get("calibrated"))
        write_json(kept, {"key": key, "dT": dT.tolist(), "warnings": after.get("warnings", []),
                          "calibrated": calibrated,
                          "measured": datetime.now().astimezone().isoformat(timespec="seconds")})
    # calibrated: measured with the calibrated radii of the spheres (else free radii, about twice the error)
    return summary(world_motion(dT, T), dT, info, False, {"scans": [p.name for p in post],
                                                           "reference": row.get("pre_scan", ""),
                                                           "calibrated": calibrated})


def scanned_trials(run) -> list[dict]:
    """The trials of a phase with their post-impact scans, newest first."""
    out = []
    for r in reversed(run.log.rows()):
        if r.get("status") in ("ok", "open") and scans_of(run, r["trial"]):
            out.append({"trial": r["trial"], "holder": r.get("holder", ""), "angle_deg": r.get("angle_deg", ""),
                        "direction_deg": r.get("direction_deg", ""), "status": r.get("status", "")})
    return out


# ── tests ────────────────────────────────────────────────────────────────────
def selftest(work: Path | None = None) -> int:
    fails = []

    def expect(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    work = Path(work or tempfile.mkdtemp(prefix="twin_test_"))
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    scene = work / "bench.blend"
    scene.write_bytes(b"scene 1")
    now = [time.time()]
    os.utime(scene, (now[0], now[0]))
    made = []

    def fake_export(copy, out):
        out.mkdir(parents=True, exist_ok=True)
        body = copy.read_bytes()
        (out / "bench.glb").write_bytes(b"glTF" + body)
        (out / "bench.json").write_text(json.dumps({"phone_print": hashlib.sha256(body).hexdigest()[:16],
                                                    "triangles": 12}), encoding="utf-8")
        made.append(body)

    w = Watcher(work / "cache", scene, blender=None, clock=lambda: now[0], export=fake_export)
    expect(not w.due() and not w.due(), "a scene saved just now: not exported")
    now[0] += QUIET_S - 1
    expect(not w.due(), f"{QUIET_S - 1:.0f} s after the save: not yet")
    now[0] += 2
    expect(w.due(), f"{QUIET_S + 1:.0f} s without a save: exported")
    c1 = w.run()
    expect(c1["version"] == 1 and (w.cache / c1["model"]).is_file() and c1["phone_print"],
           f"the first model in use, version 1 ({c1['model']})")
    expect(not w.due() and not w.due(), "the same scene: not exported again")
    scene.write_bytes(b"scene 2")
    os.utime(scene, (now[0], now[0]))
    w.due()
    now[0] += 60
    expect(not w.due(), "saved again: waits for the quiet")
    now[0] += 61
    expect(w.due(), "quiet again: exported")
    c2 = w.run()
    expect(c2["version"] == 2 and c2["model"] != c1["model"] and c2["phone_print"] != c1["phone_print"],
           f"a new model, version 2, a new fingerprint ({c2['phone_print']})")
    os.utime(scene, (now[0], now[0]))         # saved unchanged: a new time, the same content
    w.due()
    now[0] += QUIET_S + 1
    w.due()
    c3 = w.run()
    expect(c3["version"] == 2 and c3["model"] == c2["model"], "saved without a change: the same model, version 2")

    def broken(copy, out):
        raise RuntimeError("no Blender")
    w2 = Watcher(work / "cache2", scene, blender=None, clock=lambda: now[0], export=broken)
    w2.due()
    try:
        w2.run()
        expect(False, "a failed export raises")
    except RuntimeError:
        expect(not w2.due() and w2.current() is None and "no Blender" in w2.last_error,
               "a failed export is not tried again at once, and leaves no model")
    w3 = Watcher(work / "cache3", scene, blender=None, seed=work / "cache")
    expect(w3.current() and w3.current()["model"] == c2["model"], "a simulation's cache takes the app's model")

    # an export with the scanner's poses: written for ground_truth.py's simulated scans, with the scene's version
    def posed_export(copy, out):
        fake_export(copy, out)
        info = json.loads((out / "bench.json").read_text(encoding="utf-8"))
        info.update(centroid=[0.0, 0.0, 0.0], scanner_poses=[
            {"camera": f"cam_scanner_{k}", "position": k, "camera_deg": 25.0 + 15 * k, "azimuth_deg": 15.0 * k,
             "elevation_deg": -10.0, "distance_mm": 400.0 - 10 * k} for k in (1, 2, 3, 4)])
        (out / "bench.json").write_text(json.dumps(info), encoding="utf-8")
    posed = work / "scanner_poses.json"
    w4 = Watcher(work / "cache4", scene, blender=None, clock=lambda: now[0] + QUIET_S * 3, export=posed_export,
                 poses=posed)
    c4 = w4.run()
    sp = read_json(posed) or {}
    expect(len(sp.get("positions", [])) == 4 and sp.get("version") == c4["version"] and sp.get("hash") == c4["hash"]
           and sp.get("scene") == scene.name and not (work / "cache" / "scanner_poses.json").exists(),
           "an export's scanner poses written beside the protocol, with the scene's version; none without them")

    ds, net = work / "index.json", work / "refiner.json"
    v = versions(c2["phone_print"], ds, net)
    expect(v["dataset"] == "none" and v["network"] == "none" and not v["stale"], "nothing made yet: no warning")
    ds.write_text(json.dumps({"samples": 300}), encoding="utf-8")
    v = versions(c2["phone_print"], ds, net)
    expect(v["dataset"] == "old" and v["stale"], "a dataset without a fingerprint: old")
    ds.write_text(json.dumps({"scene_print": c2["phone_print"]}), encoding="utf-8")
    net.write_text(json.dumps({"scene_print": c1["phone_print"]}), encoding="utf-8")
    v = versions(c2["phone_print"], ds, net)
    expect(v["dataset"] == "current" and v["network"] == "old" and v["stale"], "dataset current, network old")
    net.write_text(json.dumps({"scene_print": c2["phone_print"]}), encoding="utf-8")
    expect(not versions(c2["phone_print"], ds, net)["stale"], "both made on this scene: no warning")

    st = setting({"trial": "t003", "holder": "B", "angle_deg": "35", "direction_deg": "45", "drape": "1"},
                 "photo_az70")
    expect(st["holder"] == "B" and st["angle_deg"] == 35 and st["direction_deg"] == 45 and st["arm_az"] == 70
           and st["drape"], f"setting of a trial at a photo step ({st})")
    expect(setting({"trial": "t003"}, "pendulum")["arm_az"] == 40, "another step: the arm at the first photo position")
    pv = phone_view(None)
    expect(abs(pv["fov_y"] - 53.2) < 0.1 and abs(pv["aspect"] - 4080 / 3072) < 1e-3
           and pv["lens"] == {"photo": None, "video": None}, f"the phone's nominal field ({pv})")
    cal = work / "calibration"
    cal.mkdir(parents=True, exist_ok=True)
    write_json(cal / "intrinsics_photo.json", {"image_size": [4080, 3072], "K": [[3050, 0, 2051], [0, 3046, 1529],
                                                                               [0, 0, 1]],
                                               "dist": [[0.11], [-0.32], [0.001], [-0.0007], [0.29]]})
    write_json(cal / "intrinsics_video.json", {"image_size": [1920, 1080], "K": "broken"})
    pv = phone_view(cal)
    ph = pv["lens"]["photo"]
    expect(pv["source"] == "calibration" and ph and ph["dist"] == [0.11, -0.32, 0.001, -0.0007, 0.29, 0, 0, 0]
           and ph["size"] == [4080, 3072] and ph["K"][0][2] == 2051 and pv["lens"]["video"] is None
           and abs(ph["fov_y"] - pv["fov_y"]) < 1e-9,
           "the lens of each mode from its calibration (the photo's; a broken one for the video left out)")

    info = {"targets": [{"id": "Pog", "mandible": [0, 87.6, -88.6]}, {"id": "Co_L", "mandible": [52.5, 0, 0]}],
            "T_W_mandible": rigid(rot([1, 0, 0], -92.3745), [0, 96.36, 16.66]).tolist(),
            "board_centre": [0, -91.2, 8.7], "ball_centre": [0, -51.2, -19.3]}
    T = np.asarray(info["T_W_mandible"])
    D = simulated_motion({"trial": "t001", "holder": "A", "angle_deg": "50"}, info)
    dT = inverse(T) @ inverse(D) @ T
    rA = summary(D, dT, info, True, {})
    c = np.asarray(info["ball_centre"])
    expect(np.allclose(D[:3, :3] @ c + D[:3, 3], c, atol=1e-9) and 1.0 <= rA["rot_deg"] <= 3.5
           and rA["tre_max_mm"] > 0.5, f"simulated A: a turn about its ball ({rA['rot_deg']:.2f} deg, TRE max "
                                        f"{rA['tre_max_mm']:.2f} mm)")
    D2 = simulated_motion({"trial": "t001", "holder": "A", "angle_deg": "50"}, info)
    expect(np.allclose(D, D2), "the same trial, the same simulated motion")
    DB = simulated_motion({"trial": "t002", "holder": "B", "angle_deg": "50"}, info)
    rB = summary(DB, inverse(T) @ inverse(DB) @ T, info, True, {})
    expect(rB["shift_mm"] < 0.4 and rB["rot_deg"] < 0.11, f"simulated B: back in its seats ({rB['shift_mm']:.3f} mm)")
    # the TRE |p - dT p| is how far the marker's motion D carries a target drawn from the registration of before
    p = np.array(info["targets"][1]["mandible"])
    pw = T[:3, :3] @ p + T[:3, 3]
    moved = D[:3, :3] @ pw + D[:3, 3]
    expect(abs(np.linalg.norm(moved - pw) - rA["tre"][1]["mm"]) < 1e-3,
           "the TRE at a target is how far the marker's motion carries it")
    # a measured result is kept by its scans and by the rig calibration: one made (or made again) later measures the
    # scans again, with the calibrated radii
    (work / "post.ply").write_bytes(b"post")
    (work / "ref.ply").write_bytes(b"ref")
    rig = work / "rig_calibration.json"
    k0 = scan_key([work / "post.ply"], [work / "ref.ply"], rig)
    rig.write_text("{}", encoding="utf-8")
    k1 = scan_key([work / "post.ply"], [work / "ref.ply"], rig)
    expect(k0["rig"] == [] and k1["rig"] and k1 != k0 and k1 == scan_key([work / "post.ply"], [work / "ref.ply"], rig),
           "the scans' cache: kept by the scans and the rig calibration (a calibration made later measures again)")

    bl = blender_path()
    if bl is None:
        print("skip  twin_export.py in Blender: no Blender found", flush=True)
    else:
        env = dict(os.environ, TEMP=str(work), TMP=str(work))
        r = subprocess.run([str(bl), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                            str(EXPORTER), "--", "test"], capture_output=True, text=True, env=env, timeout=600,
                           creationflags=NO_WINDOW if os.name == "nt" else 0)
        lines = [s for s in r.stdout.splitlines() if s.startswith(("ok ", "FAIL"))]
        for s in lines:
            print(s.replace("ok    ", "ok    blender: ", 1).replace("FAIL  ", "FAIL  blender: ", 1), flush=True)
            if s.startswith("FAIL"):
                fails.append(s)
        expect(r.returncode == 0 and lines, f"twin_export.py test in {bl.parent.name} ({len(lines)} checks)")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(fails)} failures" if fails else "\nall checks passed", flush=True)
    return 1 if fails else 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("test", "export"))
    p.add_argument("--work", type=Path)
    p.add_argument("--cache", type=Path, default=CACHE)
    a = p.parse_args(argv)
    if a.command == "test":
        return selftest(a.work)
    w = Watcher(a.cache)
    if not w.can_export():
        print("no Blender found", flush=True)
        return 1
    print(json.dumps(w.run(), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
