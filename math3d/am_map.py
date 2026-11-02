"""AM MAP: the map AM builds of a video, ready for the map UI.
  EYES: Apple Vision tracks every object (video_eyes): each box in every frame.
  LIFT, SHARED MEMORY, AM'S CORE: am3d.solve: the 3D lift through the stated lens with metric depth, the camera track
    and exposure from background patches, AM's spine (its hives and number law, with proof, or CANNOT) and sweep.
  SCENE: a coarse point cloud from metric depth at keyframes, so the map shows the room around the objects.
    build(clip, outdir, kind="indoor", progress=print) -> outdir/map.json (and outdir/clip.mp4, for the app)
    python3 am_map.py CLIP OUTDIR [indoor|outdoor]      then see it in the AM Map app (python3 ../map_app.py)"""
import json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am3d                                                    # noqa: E402

D, A = am3d.D, am3d.A
GRID = (96, 54)                                                # scene points per keyframe
NKEYS = 6
FAR = 600                                                      # cm: farther scene points are left out


def tracks(clip, out):
    """Apple Vision's boxes for every object in every frame -> out/eyes.json (what am3d.lift reads)"""
    import av
    with av.open(clip) as c:
        st = c.streams.video[0]
        fps, size = float(st.average_rate or 30), [st.width, st.height]
    w0 = time.time()
    tr = subprocess.run([os.path.join(D.APP, "video_eyes"), clip], capture_output=True, text=True, check=True)
    rows = [json.loads(l) for l in tr.stdout.splitlines() if l.strip()][:D.C.MAX_FRAMES]
    eyes = {}
    for row in rows:
        for tid, x, y, w, h, conf in row["tracks"]:
            eyes.setdefault(str(int(tid)), {})[str(row["t"])] = dict(x=int(round(x)), y=int(round(y)), w=int(round(w)),
                                                                     h=int(round(h)))
    data = dict(clip=os.path.basename(clip), fps=fps, size=size, frames=len(rows),
                stages=dict(tracking=dict(wall=round(time.time() - w0, 2))), eyes=eyes)
    json.dump(data, open(os.path.join(out, "eyes.json"), "w"))
    return data


def scene_points(clip, lens, kind, n):
    """the room: metric depth at NKEYS keyframes, sampled on a grid and lifted through the same lens"""
    import av
    from PIL import Image
    keys = sorted({round(i * (n - 1) / (NKEYS - 1)) for i in range(NKEYS)})
    pipe = am3d.depth_model(kind)
    f, cx, cy = lens["f"], lens["cx"], lens["cy"]
    out = {}
    with av.open(clip) as c:
        for t, fr in enumerate(c.decode(video=0)):
            if t > keys[-1]:
                break
            if t not in keys:
                continue
            img = fr.to_image().convert("RGB")
            W, H = img.size
            dm = pipe(img)["predicted_depth"].squeeze().detach().cpu().numpy()
            rgb = np.asarray(img)
            pts = []
            for gy in range(GRID[1]):
                for gx in range(GRID[0]):
                    u, v = (gx + 0.5) * W / GRID[0], (gy + 0.5) * H / GRID[1]
                    Z = 100.0 * float(dm[int(v * dm.shape[0] / H), int(u * dm.shape[1] / W)])
                    if not 0 < Z < FAR:
                        continue
                    r, g, b = rgb[int(v), int(u)]
                    pts += [round((u - cx) * Z / f, 1), round((v - cy) * Z / f, 1), round(Z, 1), int(r), int(g), int(b)]
            out[str(t)] = pts
    return out


def patch_tracks(clip, marks):
    """the shared memory's six patches in every frame, followed by the same call am3d.lift makes (for the eyes view)"""
    tr = subprocess.run([os.path.join(HERE, "track_from"), clip,
                         json.dumps([[i + 1, x, y, am3d.PATCH, am3d.PATCH] for i, (x, y) in enumerate(marks)])],
                        capture_output=True, text=True, check=True)
    rows = [json.loads(l) for l in tr.stdout.splitlines() if l.strip()]
    return [[[int(tid), int(round(x)), int(round(y))] for tid, x, y, w, h, conf in r["tracks"]] for r in rows]


def am_values(rule, first, last):
    """AM's own value for one number at every frame of the object, from its rule written down (None: no value)"""
    v = D.evaluate(rule, A.N)
    return [None if t >= len(v) or v[t] is None else int(v[t]) for t in range(first, last + 1)]


def camera_verdict(r, n):
    v = dict(rule=r["rule"], band=r["band"], measured=r["measured"])
    if r["rule"] != "CANNOT":
        v.update(formula=r.get("formula", ""), kept=r["kept"], broken=r["broken_frames"], am=am_values(r, 0, n - 1))
    return v


def build(clip, out, kind="indoor", progress=print):
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(os.path.join(out, "clip.mp4")):      # the app plays the map's own copy
        shutil.copyfile(clip, os.path.join(out, "clip.mp4"))
    w0 = time.time()
    progress("eyes")
    data = tracks(clip, out)
    progress("lift")
    pz = am3d.solve(clip, out, kind)
    progress("scene")
    w1 = time.time()
    points = scene_points(clip, pz["lens"], kind, pz["frames"])
    patch_xy = patch_tracks(clip, pz["markers"])
    t_scene = round(time.time() - w1, 2)
    objects = []
    for o in pz["objects"]:
        tid = str(o["id"])
        lifted = pz["lifted"][tid]
        rows = []
        for t in sorted(int(t) for t in o["own"]):
            e, L, own = data["eyes"][tid][str(t)], lifted[str(t)], o["own"][str(t)]
            rows.append([t, e["x"], e["y"], e["w"], e["h"], L["X"], L["Y"], L["Z"], L["W"], L["H"], L["B"],
                         own["rX"], own["rY"], own["rZ"], own["oB"]])
        verdict = {}
        for k, r in o["numbers"].items():
            v = dict(rule=r["rule"], band=r["band"], measured=r["measured"])
            if r["rule"] != "CANNOT":
                v.update(formula=r.get("formula", ""), kept=r["kept"], broken=r["broken_frames"],
                         am=am_values(r, o["first"], o["last"]))
            verdict[k] = v
        swept = {run: {k: g for k, g in got.items() if g is not None} for run, got in o["swept"].items()}
        objects.append(dict(id=o["id"], first=o["first"], last=o["last"], rows=rows, numbers=verdict, swept=swept))
    stages = dict(pz["stages"])
    stages["tracking"] = data["stages"]["tracking"]
    stages["scene"] = dict(wall=t_scene)
    m = dict(clip=data["clip"], fps=data["fps"], size=data["size"], frames=pz["frames"], lens=pz["lens"],
             depth_model=pz["depth_model"], camera={k: camera_verdict(r, pz["frames"]) for k, r in pz["camera"].items()},
             marker=pz["marker"], grey=pz["grey"], patches=pz["markers"], patch_xy=patch_xy, patch_size=am3d.PATCH,
             objects=objects, points=points,
             stages=stages, total_wall=round(time.time() - w0, 1),
             cost_usd=round((time.time() - w0) / 3600 * 0.03, 5))
    tmp = os.path.join(out, "map.json.tmp")                    # whole or nothing: the app may be serving the old one
    json.dump(m, open(tmp, "w"))
    os.replace(tmp, os.path.join(out, "map.json"))
    progress("done")
    return m


if __name__ == "__main__":
    os.makedirs(sys.argv[2], exist_ok=True)
    if not os.path.exists(os.path.join(sys.argv[2], "name.txt")):           # the name the app lists it under
        open(os.path.join(sys.argv[2], "name.txt"), "w").write(os.path.basename(sys.argv[1]))
    build(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "indoor")
