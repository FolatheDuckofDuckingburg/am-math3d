"""AM 3D: a real clip lifted into a 3D geometric puzzle, solved by AM's passes, with proof for every answer.
  THE LENS: a pinhole with a 60-degree horizontal field of view (f = W / (2 tan 30 deg)), centred: an assumption, so
    every 3D number is in centimetres under it.
  THE EYES: Apple Vision's tracks (AM-Video's video_eyes) give each object's box; Depth Anything V2 Metric (indoor or
    outdoor, small) gives metric depth.  Each object's camera-space X, Y, Z (cm) come from its box centre and the
    median depth of the box's inner 60%; its width and height in cm; its brightness, the inner box's mean luminance.
  PASS 1, THE GLOBAL MAP (shared memory): six textured background patches, away from every object's boxes and the
    corner watermark, are followed from frame 0 (track_from) and lifted the same way.  Their median displacement is
    the camera's track (the marker); their median brightness change is the exposure (the grey card).  Cancelling both
    leaves each object's own numbers: rX, rY, rZ (relative to the marker), W, H, oB.
  THE SPINE (the solved puzzle): every own number's rule over the clip, from AM's hives and number law, sealed
    (anchor_am, as in the trap), with the band from the number's own noise (tau = 3 sigma + 0.5, AM-Video's).
  PASS 2, THE SWEEP: every hidden or broken stretch is asked of the five hives with the 3D test's law (w3d_am, sealed).
  PROOF: each rule carries the hive the law chose, its formula, its band, the frames that keep it and the frames that
    break it; CANNOT where AM cannot tell.
    python3 am3d.py eyes CLIP OUTDIR
    python3 am3d.py solve CLIP OUTDIR indoor|outdoor"""
import json, os, subprocess, sys, time, resource
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am_dict as D                                           # noqa: E402  (AM-Video, the hives, rules written down)

C, A = D.C, D.A
sys.path.insert(0, os.path.join(HERE, "..", "am"))
import w3d_am as M                                            # noqa: E402  (the five hives, the 3D test's law)

FOV = 60.0
OWN = ("rX", "rY", "rZ", "W", "H", "oB")
CAM = ("mX", "mY", "mZ")
PATCH, NMARK = 48, 6
_DEPTH = {}


def lens(W, H):
    f = (W / 2) / np.tan(np.radians(FOV / 2))
    return dict(f=float(f), cx=W / 2, cy=H / 2, fov=FOV)


def depth_model(kind):
    if kind not in _DEPTH:
        import torch
        from transformers import pipeline
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _DEPTH[kind] = pipeline("depth-estimation", device=dev,
                                model="depth-anything/Depth-Anything-V2-Metric-%s-Small-hf" % kind.capitalize())
    return _DEPTH[kind]


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def pick_markers(frame, boxes_all, W, H):
    """six textured background patches at frame 0, away from every object box (any frame) and the watermark corner"""
    g = frame.astype(float).mean(2)
    gx, gy = np.abs(np.diff(g, axis=1))[:-1, :], np.abs(np.diff(g, axis=0))[:, :-1]
    tex = gx + gy
    bad = np.zeros((H, W), bool)
    for x, y, w, h in boxes_all:
        x0, x1 = int(max(0, x - w / 2 - 16)), int(min(W, x + w / 2 + 16))
        y0, y1 = int(max(0, y - h / 2 - 16)), int(min(H, y + h / 2 + 16))
        bad[y0:y1, x0:x1] = True
    bad[int(0.75 * H):, int(0.8 * W):] = True
    cands = []
    for y in range(PATCH // 2, H - PATCH // 2, PATCH // 2):
        for x in range(PATCH // 2, W - PATCH // 2, PATCH // 2):
            y0, x0 = y - PATCH // 2, x - PATCH // 2
            if bad[y0:y0 + PATCH, x0:x0 + PATCH].any():
                continue
            cands.append((float(tex[y0:min(y0 + PATCH, H - 1), x0:min(x0 + PATCH, W - 1)].mean()), x, y))
    cands.sort(reverse=True)
    picked = []
    for s, x, y in cands:
        if all((x - a) ** 2 + (y - b) ** 2 >= 100 ** 2 for _, a, b in picked):
            picked.append((s, x, y))
        if len(picked) == NMARK:
            break
    return [(x, y) for _, x, y in picked]


def lift(clip, out, kind):
    """the eyes in 3D: {objects: {id: {t: X, Y, Z, W, H, B}}, marker: {t: X, Y, Z}, grey: {t: g}}"""
    import av
    eyes = json.load(open(os.path.join(out, "eyes.json")))
    Wd, Hd = eyes["size"]
    L = lens(Wd, Hd)
    with av.open(clip) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)][:eyes["frames"]]
    objs = {tid: {int(t): e for t, e in fr.items()} for tid, fr in eyes["eyes"].items()
            if len(fr) >= C.MIN_TRACK}
    allboxes = [(e["x"], e["y"], e["w"], e["h"]) for fr in eyes["eyes"].values() for e in fr.values()]
    marks = pick_markers(frames[0], allboxes, Wd, Hd)
    w0 = time.time()
    tr = subprocess.run([os.path.join(HERE, "track_from"), clip,
                         json.dumps([[i + 1, x, y, PATCH, PATCH] for i, (x, y) in enumerate(marks)])],
                        capture_output=True, text=True, check=True)
    mrows = [json.loads(l) for l in tr.stdout.splitlines() if l.strip()]
    t_track = time.time() - w0
    w0, c0 = time.time(), _cpu()
    pipe = depth_model(kind)
    f, cx, cy = L["f"], L["cx"], L["cy"]
    lum = lambda im: 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
    o3d, mk, grey = {tid: {} for tid in objs}, {}, {}
    m0 = None
    from PIL import Image
    for t, fr in enumerate(frames):
        dm = pipe(Image.fromarray(fr))["predicted_depth"]
        dm = dm.squeeze().detach().cpu().numpy() if hasattr(dm, "detach") else np.array(dm, dtype=float)
        sy, sx = dm.shape[0] / Hd, dm.shape[1] / Wd
        lm = lum(fr.astype(float))

        def inner(x, y, w, h, a=0.3):
            x0, x1 = int(max(0, x - a * w)), int(min(Wd, x + a * w))
            y0, y1 = int(max(0, y - a * h)), int(min(Hd, y + a * h))
            if x1 <= x0 or y1 <= y0:
                return None, None
            d = dm[int(y0 * sy):max(int(y1 * sy), int(y0 * sy) + 1), int(x0 * sx):max(int(x1 * sx), int(x0 * sx) + 1)]
            return 100.0 * float(np.median(d)), float(lm[y0:y1, x0:x1].mean())

        for tid, fr_o in objs.items():
            e = fr_o.get(t)
            if e is None:
                continue
            Z, B = inner(e["x"], e["y"], e["w"], e["h"])
            if Z is None or Z <= 0:
                continue
            o3d[tid][t] = dict(X=int(round((e["x"] - cx) * Z / f)), Y=int(round((e["y"] - cy) * Z / f)),
                               Z=int(round(Z)), W=int(round(e["w"] * Z / f)), H=int(round(e["h"] * Z / f)),
                               B=int(round(B)))
        pts, ls = [], []
        for tid, x, y, w, h, conf in mrows[t]["tracks"] if t < len(mrows) else []:
            Z, Lm = inner(x, y, w, h, a=0.5)
            if Z is None or Z <= 0:
                continue
            pts.append((int(tid), (x - cx) * Z / f, (y - cy) * Z / f, Z))
            ls.append((int(tid), Lm))
        if t == 0:
            m0 = {i: (X, Y, Z) for i, X, Y, Z in pts}
            l0 = dict(ls)
        dis = [(X - m0[i][0], Y - m0[i][1], Z - m0[i][2]) for i, X, Y, Z in pts if i in m0]
        dl = [Lm - l0[i] for i, Lm in ls if i in l0]
        if dis:
            ref = np.median(np.array(list(m0.values())), 0)
            d = np.median(np.array(dis), 0)
            mk[t] = dict(X=int(round(ref[0] + d[0])), Y=int(round(ref[1] + d[1])), Z=int(round(ref[2] + d[2])))
            grey[t] = int(round(128 + float(np.median(dl))))
    stages = dict(markers_tracking=round(t_track, 2), depth_and_lift=dict(wall=round(time.time() - w0, 2),
                                                                           cpu=round(_cpu() - c0, 2)))
    return dict(lens=L, depth_model=kind, markers=marks, objects=o3d, marker=mk, grey=grey, stages=stages,
                frames=len(frames), size=[Wd, Hd], fps=eyes["fps"])


def spine(ts, y):
    """AM's spine for one number: the law's hive and its rule written down, or CANNOT; with its proof"""
    tau = 3 * C.sigma(y) + 0.5
    reps = A.reports(ts, y, tau)
    act = A.law(A.state(reps, tau, len(ts)))
    if not act or reps[act - 1] is None:
        return dict(rule="CANNOT", band=round(tau, 2), measured=len(ts)), None
    rule = D.write_rule(act, ts, y, tau, reps[act - 1])
    vals = D.evaluate(rule, A.N)
    off = [t for t, v in zip(ts, y) if vals[t] is not None and abs(int(v) - vals[t]) > tau]
    rule.update(band=round(tau, 2), measured=len(ts), kept=len(ts) - len(off), broken=len(off), broken_frames=off)
    return rule, vals


def solve(clip, out, kind):
    l3 = lift(clip, out, kind)
    n = l3["frames"]
    C._length(n)
    w0, c0 = time.time(), _cpu()
    cam_ts = sorted(int(t) for t in l3["marker"])
    camera = {}
    for key, k in zip(CAM, "XYZ"):
        rule, _ = spine(cam_ts, np.array([l3["marker"][t][k] for t in cam_ts], dtype=np.int64))
        camera[key] = rule
    objects = []
    for tid, fr in sorted(l3["objects"].items(), key=lambda kv: int(kv[0])):
        meas = sorted(t for t in fr if t in l3["marker"] and t in l3["grey"])
        if len(meas) < C.MIN_TRACK:
            continue
        own = {t: dict(rX=fr[t]["X"] - l3["marker"][t]["X"], rY=fr[t]["Y"] - l3["marker"][t]["Y"],
                       rZ=fr[t]["Z"] - l3["marker"][t]["Z"], W=fr[t]["W"], H=fr[t]["H"],
                       oB=fr[t]["B"] - (l3["grey"][t] - 128)) for t in meas}
        nums, vals, broken = {}, {}, set()
        for key in OWN:
            rule, v = spine(meas, np.array([own[t][key] for t in meas], dtype=np.int64))
            nums[key], vals[key] = rule, v
            broken |= set(rule.get("broken_frames", []))
        clean = [t for t in meas if t not in broken]
        ask = sorted(set(range(min(meas), max(meas) + 1)) - set(clean))
        swept = {}
        for run in M.runs(ask) if ask and len(clean) >= 5 else []:
            got = {key: M.answers({t: own[t][key] for t in clean}, run, laws=("w3d",))["w3d"] for key in OWN}
            swept["%d-%d" % (run[0], run[-1])] = {key: (None if g is None else {str(t): int(g[t]) for t in run})
                                                  for key, g in got.items()}
        objects.append(dict(id=int(tid), first=min(meas), last=max(meas), measured=len(meas), numbers=nums,
                            broken=sorted(broken), swept=swept, own={str(t): own[t] for t in meas}))
    l3["stages"]["am"] = dict(wall=round(time.time() - w0, 2), cpu=round(_cpu() - c0, 2), tokens=0)
    puzzle = dict(clip=os.path.basename(clip), lens=l3["lens"], depth_model=l3["depth_model"], size=l3["size"],
                  fps=l3["fps"], frames=n, markers=l3["markers"], camera=camera,
                  marker={str(t): v for t, v in l3["marker"].items()}, grey={str(t): v for t, v in l3["grey"].items()},
                  objects=objects, stages=l3["stages"],
                  lifted={tid: {str(t): v for t, v in fr.items()} for tid, fr in l3["objects"].items()})
    json.dump(puzzle, open(os.path.join(out, "puzzle3d.json"), "w"), indent=1)
    return puzzle


def proof(puzzle):
    lines = ["%s: %d frames, lens f=%.0f px (%.0f-degree view, assumed), depth: %s metric" % (
        puzzle["clip"], puzzle["frames"], puzzle["lens"]["f"], puzzle["lens"]["fov"], puzzle["depth_model"])]
    for key, r in puzzle["camera"].items():
        lines.append("  camera %s: %s" % (key, _say(r)))
    for o in puzzle["objects"]:
        lines.append("  object %d (frames %d-%d, %d measured):" % (o["id"], o["first"], o["last"], o["measured"]))
        for key, r in o["numbers"].items():
            lines.append("    %-3s %s" % (key, _say(r)))
        if o["swept"]:
            lines.append("    sweep: " + "; ".join("%s %s" % (run, ",".join(k for k, v in got.items() if v is not None)
                                                                 or "CANNOT") for run, got in o["swept"].items()))
    return "\n".join(lines)


def _say(r):
    if r["rule"] == "CANNOT":
        return "CANNOT (band %s, %d frames)" % (r["band"], r["measured"])
    return "%s %s  band %s  keeps %d of %d frames" % (r["rule"], r.get("formula", ""), r["band"], r["kept"], r["measured"])


if __name__ == "__main__":
    if sys.argv[1] == "eyes":                                  # Apple Vision's tracks, cached in OUTDIR/eyes.json
        os.makedirs(sys.argv[3], exist_ok=True)
        D.measure(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "solve":
        pz = solve(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "indoor")
        print(proof(pz))
        print(json.dumps(pz["stages"], indent=1))
