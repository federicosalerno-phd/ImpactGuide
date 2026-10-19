"""The 3D-printed calibration plate as ImpactGuide's page shows it: its two STL (02_cad/stl, build_charuco_plate.py)
as one binary glTF, 04_code/app/ui/models/charuco_plate.glb.

Two meshes, the white plate and the black print in relief, each with its colour as printed (PLA white and black,
matte), vertices welded, in metres as glTF wants them, Y up (the plate's face looking up, +Y), centred on the plate.
The scene's extras say where the caliper reads (the two lengths of the plate's JSON, between the outer faces of the
ticks, at the middle of the ticks and just above the face): `check` holds them to the nominal lengths (the page draws
nothing over the model: the plate's own printed marks show both lengths).
The page (ui/17-plate3d.js, VIEW3D) reads it with three.js and draws it flat shaded: a printed part is all planes, so
the file carries no normals. It is made once, and again whenever the plate is rebuilt.

Usage:
    python build_plate_model.py            write the GLB from the STL of 02_cad/stl
    python build_plate_model.py check      read the GLB back: two meshes, closed, the same triangles as the STL
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STL = ROOT / "02_cad" / "stl"
PLATE = STL / "charuco_plate_11x8_20mm.json"
OUT = ROOT / "04_code" / "app" / "ui" / "models" / "charuco_plate.glb"
# printed colours, linear RGB for glTF (sRGB #F2F1EC and #1A1A1A): the white and black of a matte PLA
COLOURS = {"white": ((0.888, 0.879, 0.838, 1.0), 0.62), "black": ((0.010, 0.010, 0.010, 1.0), 0.48)}


def read_stl(path):
    """Triangles (N x 3 x 3, mm) of a binary STL."""
    data = Path(path).read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * n:
        raise ValueError(f"{path.name}: not a binary STL")
    rec = np.frombuffer(data, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
    return rec["v"].astype(np.float64)


def weld(tris):
    """(vertices, faces) with the corners that are the same point joined: the same float32 coordinates, as the STL
    writes a shared corner (joining near ones instead would fuse the black pieces that the plate keeps 0.1 mm apart
    at the chessboard's corners)."""
    v = tris.reshape(-1, 3).astype(np.float32)
    uniq, inv = np.unique(v, axis=0, return_inverse=True)
    return uniq.astype(np.float64), inv.reshape(-1, 3).astype(np.uint32)


def glb(meshes, extras=None):
    """A binary glTF of meshes [(name, vertices in m, faces, (rgba, roughness))], one node each."""
    bin_ = bytearray()
    views, accessors, gmeshes, materials, nodes = [], [], [], [], []

    def add(arr, target):
        while len(bin_) % 4:
            bin_.append(0)
        views.append({"buffer": 0, "byteOffset": len(bin_), "byteLength": arr.nbytes, "target": target})
        bin_.extend(arr.tobytes())
        return len(views) - 1
    for i, (name, v, f, (rgba, rough)) in enumerate(meshes):
        v = v.astype(np.float32)
        pv = add(v, 34962)
        accessors.append({"bufferView": pv, "componentType": 5126, "count": len(v), "type": "VEC3",
                          "min": v.min(0).tolist(), "max": v.max(0).tolist()})
        fi = add(f.astype(np.uint32).ravel(), 34963)
        accessors.append({"bufferView": fi, "componentType": 5125, "count": int(f.size), "type": "SCALAR"})
        materials.append({"name": name, "pbrMetallicRoughness": {"baseColorFactor": list(rgba), "metallicFactor": 0.0,
                                                                  "roughnessFactor": rough}})
        gmeshes.append({"name": name, "primitives": [{"attributes": {"POSITION": 2 * i}, "indices": 2 * i + 1,
                                                      "material": i}]})
        nodes.append({"name": name, "mesh": i})
    while len(bin_) % 4:
        bin_.append(0)
    doc = {"asset": {"version": "2.0", "generator": "ImpactGuide build_plate_model.py"}, "scene": 0,
           "scenes": [{"nodes": list(range(len(nodes))), "extras": extras or {}}], "nodes": nodes, "meshes": gmeshes,
           "materials": materials, "accessors": accessors, "bufferViews": views,
           "buffers": [{"byteLength": len(bin_)}]}
    js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(bin_), 0x004E4942) + bytes(bin_)
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def read_glb(path):
    """(document, binary chunk) of a GLB."""
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2 or length != len(data):
        raise ValueError("not a glTF 2 binary")
    jl = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + jl])
    bl = struct.unpack_from("<I", data, 20 + jl)[0]
    return doc, data[28 + jl:28 + jl + bl]


def measures(plate):
    """Where the caliper reads, in plate coordinates (mm): the long length above the board, the short one right of
    it, each a segment between the outer faces of its ticks (build_charuco_plate.py: TICK_GAP, TICK_L)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_charuco_plate as bcp
    ox, oy, top = plate["plate"]["board_origin_mm"]
    L, H = plate["board"]["size_mm"]
    mid = bcp.TICK_GAP + bcp.TICK_L / 2
    z = top + 0.6
    return {"long": [[ox, oy + mid, z], [ox + L, oy + mid, z]],
            "short": [[ox + L + mid, oy, z], [ox + L + mid, oy - H, z]]}


def build():
    plate = json.loads(PLATE.read_text(encoding="utf-8"))
    w, h, t = plate["plate"]["size_mm"]

    def to_gltf(v):
        v = np.asarray(v, np.float64)
        return np.column_stack([v[:, 0] - w / 2, v[:, 2] - t / 2, -(v[:, 1] - h / 2)]) / 1000.0
    meshes = []
    for part in ("white", "black"):
        tris = read_stl(STL / plate["files"][part])
        v, f = weld(tris)
        # plate coordinates (X right, Y up the plate, Z out of its face, mm) to glTF (X right, Y up, Z towards the
        # viewer, m): the plate lies flat, its face up, its top edge away from the viewer, centred
        meshes.append((part, to_gltf(v), f, COLOURS[part]))
        print(f"{part}: {len(tris)} triangles, {len(v)} vertices")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    extras = {"measure": {k: {"from": to_gltf(seg)[0].round(6).tolist(), "to": to_gltf(seg)[1].round(6).tolist(),
                              "nominal_mm": plate["measure"][k]["nominal_mm"]} for k, seg in measures(plate).items()},
              "size_mm": plate["plate"]["size_mm"]}
    OUT.write_bytes(glb(meshes, extras))
    print(f"{OUT} ({OUT.stat().st_size / 1024:.0f} kB)")
    return check()


def check():
    doc, bin_ = read_glb(OUT)
    plate = json.loads(PLATE.read_text(encoding="utf-8"))
    ok = len(doc["meshes"]) == 2
    m = doc["scenes"][0].get("extras", {}).get("measure", {})
    for k in ("long", "short"):
        a, b = np.array(m[k]["from"]), np.array(m[k]["to"])
        ln = 1000 * np.linalg.norm(a - b)
        print(f"caliper, {k}: {ln:.2f} mm between the ticks' outer faces (nominal {m[k]['nominal_mm']:.1f})")
        ok &= abs(ln - m[k]["nominal_mm"]) < 1e-3
    for i, part in enumerate(("white", "black")):
        acc = doc["accessors"][2 * i + 1]
        view = doc["bufferViews"][acc["bufferView"]]
        f = np.frombuffer(bin_, np.uint32, acc["count"], view["byteOffset"]).reshape(-1, 3)
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        closed = bool(np.all(counts == 2))
        same = len(f) == len(read_stl(STL / plate["files"][part]))
        lo, hi = doc["accessors"][2 * i]["min"], doc["accessors"][2 * i]["max"]
        print(f"{part}: {len(f)} triangles ({'as the STL' if same else 'NOT as the STL'}), "
              f"{'closed' if closed else 'NOT closed'}, {1000 * (hi[0] - lo[0]):.1f} x {1000 * (hi[2] - lo[2]):.1f} "
              f"x {1000 * (hi[1] - lo[1]):.2f} mm")
        ok &= closed and same
    print("check: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["build", "check"], default="build")
    a = ap.parse_args()
    return build() if a.command == "build" else check()


if __name__ == "__main__":
    sys.exit(main())
