"""Face shapes and skin deformations of the digital twin (T13.1, module 3a of ai_design.md). NumPy only: it runs inside
Blender (synth_twin.py) and outside.

Face bank. A face identity is the phantom face (`silicone_face` of the bench scene) plus a displacement of its vertices,
so mouth on the splint, cut of the mask and seat on the skull stay where they are and only the shape changes:
    OUT/faces/<face_id>.npz
        displacement   N x 3 float32, mm, frame B (bench scene), one row per vertex of silicone_face
        vertex_count   N, checked against the scene mesh (a remodelled face invalidates the bank)
        split          "train", "val" or "test" (identities never cross splits)
        source         free text, e.g. the model and its coefficients
The phantom itself ("phantom") has no file. Faces of the parametric model (FLAME 2023 Open, ai_design.md) enter the
dataset by writing files of this form with save_face(); the generator needs nothing else.

Skin deformation. A smooth random field along the vertex normals: a Gaussian process with squared exponential kernel
approximated by random Fourier features (length scale 20-40 mm), scaled so that its largest value on the weighted
vertices is the sampled amplitude (0-4 mm). Weights go to zero at the lips on the splint, at the cut border of the mask
and on the inner skin that rests on the skull. The registration stays the rigid one of the head: the deformation is a
disturbance, and the modules always render the face without it.
"""
from pathlib import Path

import numpy as np

AMP_MM = (0.0, 4.0)        # largest displacement of the skin, uniform
SCALE_MM = (20.0, 40.0)    # length scale of the field, uniform
FEATURES = 96              # random Fourier features
MOUTH_MM = (3.0, 15.0)     # weight 0 within 3 mm of the splint contact, 1 beyond 3 + 15 mm
BORDER_MM = (2.0, 12.0)    # same from the cut border of the mask


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ramp(d, start_ramp):
    start, width = start_ramp
    return smoothstep((np.asarray(d) - start) / width)


def weights(points, normals, inside_point, d_mouth, d_border):
    """Deformation weight per vertex: outer skin only (normal away from inside_point), damped near mouth and border."""
    outer = np.sum(normals * (points - np.asarray(inside_point)), axis=1) > 0.0
    return (outer * ramp(d_mouth, MOUTH_MM) * ramp(d_border, BORDER_MM)).astype(np.float32)


def rff_field(points, rng, scale_mm, n=FEATURES):
    """One draw of a zero-mean Gaussian process with squared exponential kernel of length scale scale_mm."""
    omega = rng.normal(0.0, 1.0 / scale_mm, (n, 3))
    phase = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.sqrt(2.0 / n) * np.cos(points @ omega.T + phase) @ rng.normal(0.0, 1.0, n)


def deformation(points, normals, w, rng, amp_mm=None, scale_mm=None):
    """Displacement N x 3 (mm) of a random skin deformation and its parameters."""
    amp = rng.uniform(*AMP_MM) if amp_mm is None else amp_mm
    scale = rng.uniform(*SCALE_MM) if scale_mm is None else scale_mm
    f = rff_field(points, rng, scale) * w
    peak = np.abs(f).max()
    f = f * (amp / peak) if peak > 0 else f
    return (f[:, None] * normals).astype(np.float32), {"amp_mm": round(float(amp), 3), "scale_mm": round(float(scale), 2)}


def save_face(folder, face_id, displacement, split, source):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    assert split in ("train", "val", "test")
    np.savez_compressed(folder / f"{face_id}.npz", displacement=np.asarray(displacement, np.float32),
                        vertex_count=len(displacement), split=split, source=source)


def load_bank(folder, vertex_count):
    """Faces of the bank that fit the current phantom mesh: {face_id: {"displacement", "split", "source"}}."""
    bank = {}
    for p in sorted(Path(folder).glob("*.npz")):
        z = np.load(p)
        if int(z["vertex_count"]) != vertex_count:
            raise ValueError(f"{p.name}: made for {int(z['vertex_count'])} vertices, the face has {vertex_count}")
        bank[p.stem] = {"displacement": z["displacement"], "split": str(z["split"]), "source": str(z["source"])}
    return bank
