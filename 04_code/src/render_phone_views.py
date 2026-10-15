"""Render the smartphone photo views of the test bench with the ground truth of the marker pose (T5.4).

Run with Blender 5.2 in the background; the scene is rebuilt from 04_code/tools/build_test_bench.py and nothing is saved:
    blender --background --factory-startup --python 04_code/src/render_phone_views.py -- [--out DIR] [--size W H]
        [--samples N] [--variant a1]
Writes in DIR (default OUT), for each photo position k = 1..4:
    view_k.png           what the phone sees (silicone on, pendulum latched up, phone body hidden)
    mandible_mask_k.png  alpha of the mandible alone, to check the overlay renderer
and gt.json (OpenCV intrinsics, camera from world, world from board, board parameters) and mandible_board.npz (mandible
mesh in the board frame: vertices in mm, triangles).

Frames as in OpenCV. Camera: x right, y down, z forward; pixel centres at integer coordinates. Board as
cv2.aruco.CharucoBoard: origin at the outer corner of the first square (top left of the printed image), x along the
printed columns, y along the rows, z into the board.
"""
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "04_code" / "tools" / "build_test_bench.py"
OUT = Path(r"C:\work\marker-impact\synthetic_pose")
SIZE = (4080, 3072)   # Poco X7 Pro main camera, 12.5 MP binned photo (4:3)
CV_FROM_BLENDER = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))   # Blender cameras look along -z with y up


def load_bench():
    ns = {"__file__": str(BENCH), "__name__": "build_test_bench"}
    exec(compile(BENCH.read_text(encoding="utf-8"), str(BENCH), "exec"), ns)
    return ns


def as_list(m):
    return [list(row) for row in m]


def intrinsics(cam, w, h):
    d = cam.data
    assert d.type == "PERSP" and d.sensor_fit == "HORIZONTAL" and d.shift_x == 0.0 and d.shift_y == 0.0
    f = d.lens / d.sensor_width * w
    # Blender's image plane spans [0, w] x [0, h]; OpenCV puts the centre of pixel i at i
    return [[f, 0.0, w / 2 - 0.5], [0.0, f, h / 2 - 0.5], [0.0, 0.0, 1.0]]


def world_from_board(ob, charuco, size):
    """Board frame from the UV corners of the printed square: UV (0, 1) is the top left of the image."""
    me, mw = ob.data, ob.matrix_world
    uv = me.uv_layers.active.data
    corner = {}
    for lp in me.loops:                                           # the board is a grid of cells: keep the four corners
        u, v = uv[lp.index].uv
        if max(abs(u - round(u)), abs(v - round(v))) < 1e-6:
            corner[(round(u), round(v))] = mw @ me.vertices[lp.vertex_index].co
    top_left, ex, ey = corner[(0, 1)], corner[(1, 1)] - corner[(0, 1)], corner[(0, 0)] - corner[(0, 1)]
    assert abs(ex.length - size) < 1e-3 and abs(ey.length - size) < 1e-3 and abs(ex.dot(ey)) < 1e-3
    x, y = ex.normalized(), ey.normalized()
    origin = top_left + (x + y) * (size - charuco["squares"] * charuco["square"]) / 2
    m = Matrix.Identity(4)
    m.col[0][:3], m.col[1][:3], m.col[2][:3], m.col[3][:3] = x, y, x.cross(y), origin
    return m


def mesh_arrays(ob):
    """World vertices and triangles of the evaluated mesh, duplicate vertices merged."""
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    tri = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    ev.to_mesh_clear()
    mw = np.array(ob.matrix_world)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    _, first, inverse = np.unique(np.round(co, 6), axis=0, return_index=True, return_inverse=True)
    tri = inverse.ravel()[tri.reshape(-1, 3)]
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    return co[first], tri[keep].astype(np.int32)


def render(sc, path, transparent, samples):
    sc.render.film_transparent = transparent
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    sc.eevee.taa_render_samples = samples
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main(argv):
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    opt = lambda name, n, default: args[args.index(name) + 1:args.index(name) + 1 + n] if name in args else default
    out = Path(opt("--out", 1, [OUT])[0])
    w, h = (int(v) for v in opt("--size", 2, SIZE))
    samples = int(opt("--samples", 1, [64])[0])
    variant = opt("--variant", 1, ["a1"])[0]
    out.mkdir(parents=True, exist_ok=True)

    bench = load_bench()
    info = bench["build"]()
    P = bench["P"]
    sc, objs = bpy.context.scene, bpy.data.objects
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    bench["show"](variant=variant, silicone=True)
    bench["set_direction"](0.0)   # pendulum latched up, as during photos

    wb = world_from_board(objs["marker_charuco"], P["charuco"], P["marker_size"])
    mandible = objs["phantom_mandible"]
    verts, faces = mesh_arrays(mandible)
    bw = np.array(wb.inverted())
    np.savez_compressed(out / "mandible_board.npz", vertices=verts @ bw[:3, :3].T + bw[:3, 3], faces=faces)

    sa = sc["sensor_arm"]
    phone_body = {o.name for o in objs if o.name.startswith(("smartphone", "phone_cradle"))}
    views = []
    for k, phi in enumerate(sa["arm_deg"]):
        bench["set_arm"](phi)
        cam = objs[f"cam_phone_{k + 1}"]
        sc.camera = cam
        for ob in objs:
            ob.hide_render = ob.name in phone_body
        render(sc, out / f"view_{k + 1}.png", False, samples)
        for ob in objs:
            ob.hide_render = ob is not mandible
        render(sc, out / f"mandible_mask_{k + 1}.png", True, 16)
        cw = CV_FROM_BLENDER @ cam.matrix_world.normalized().inverted()
        cb = cw @ wb
        views.append({"k": k + 1, "arm_deg": phi, "camera_deg": sa["camera_deg"][k], "K": intrinsics(cam, w, h),
                      "camera_from_world": as_list(cw), "camera_from_board": as_list(cb),
                      "distance_mm": cb.translation.length})
    for ob in objs:
        ob.hide_render = False
    bench["set_arm"]()

    c = P["charuco"]
    gt = {
        "image_size": [w, h],
        "dist": [0.0] * 5,
        "board": {"squares": c["squares"], "square_mm": c["square"], "marker_mm": c["marker"],
                  "dictionary": c["dictionary"], "print_mm": P["marker_size"]},
        "variant": variant,
        "world_from_board": as_list(wb),
        "views": views,
        "render": {"engine": sc.render.engine, "samples": samples, "filter_px": sc.render.filter_size,
                   "blender": bpy.app.version_string,
                   "bench_sha256": hashlib.sha256(BENCH.read_bytes()).hexdigest()[:12]},
        "bench": {key: info[key] for key in ("pad_origin", "target", "arm_deg", "phone_mm")},
        "mandible": {"vertices": len(verts), "faces": len(faces)},
    }
    (out / "gt.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out), "views": [round(v["distance_mm"], 1) for v in views], **gt["mandible"]}))


if __name__ == "__main__":
    main(sys.argv)
