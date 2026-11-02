"""AM-VIDEO's core, run locally.  Two modes.
VIDEO MODE (any mp4, zero-shot): the eyes (Apple Vision tracking, video_eyes; Depth Anything V2 Small for each object's
  depth) measure every tracked object's x, y, w, h (px) and depth in every frame; AM (the sealed hives and number law,
  AM's eyes in C) finds the rule each number follows in each 150-frame window, within that number's own noise band
  (measured from the clip), and flags the frames that break it, with the numbers that broke.  AM sees motion and
  geometry, not content: a creature that morphs but keeps its box is not a break it can see.
3D-PASSES MODE (render passes, the trap's format): AM's three passes (global map, spine, five-hive sweep) restore the
  asset's state in every frame that is hidden or broken; where it cannot, it says so.
    video(path, out_mp4) -> report;  passes(data) -> report"""
import json, os, resource, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                      # AM's modules and laws live next to this file
import anchor_am as A                                         # noqa: E402
import scene_am as S                                          # noqa: E402
import am_eyes_c                                              # noqa: E402

am_eyes_c.use()                                               # AM's eyes in C: identical rows, 6-8x faster
KEYS = ("x", "y", "w", "h", "d")
NAMES = dict(x="position x", y="position y", w="width", h="height", d="depth")
WINDOW = 150
MIN_TRACK = 20
MAX_FRAMES = int(os.environ.get("AMV_MAX_FRAMES", "600"))
_DEPTH = None


def _cpu(who):
    r = resource.getrusage(who)
    return r.ru_utime + r.ru_stime


def depth_model():
    global _DEPTH
    if _DEPTH is None:
        import torch
        from transformers import pipeline
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _DEPTH = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)
    return _DEPTH


def sigma(ys):
    ys = np.asarray(ys, dtype=float)
    if len(ys) < 5:
        return 1.0
    d2 = ys[2:] - 2 * ys[1:-1] + ys[:-2]
    return max(0.5, 1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / np.sqrt(6))


def _length(n, cache={}):
    if n not in cache:
        cache.clear()
        cache[n] = [(d, Y) for d, Y in S.levels(list(range(n)))]
    A.N, A._EYES = n, cache[n]


def video(path, out_mp4):
    import av
    from PIL import Image, ImageDraw
    rep = dict(stages={})
    with av.open(path) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate or 30)
        rep.update(fps=fps, size="%dx%d" % (st.width, st.height))
    # STAGE A: Apple Vision tracking
    w0, c0 = time.time(), _cpu(resource.RUSAGE_CHILDREN)
    tr = subprocess.run([os.path.join(HERE, "video_eyes"), path], capture_output=True, text=True, check=True)
    rows = [json.loads(l) for l in tr.stdout.splitlines() if l.strip()][:MAX_FRAMES]
    rep["stages"]["tracking"] = dict(wall=round(time.time() - w0, 2), cpu=round(_cpu(resource.RUSAGE_CHILDREN) - c0, 2))
    n = len(rows)
    rep.update(frames=n, seconds=round(n / fps, 2), truncated=len(tr.stdout.splitlines()) > MAX_FRAMES)
    # STAGE B: depth per object per frame
    w0, c0 = time.time(), _cpu(resource.RUSAGE_SELF)
    pipe = depth_model()
    eyes, images = {}, []
    with av.open(path) as c:
        for t, frame in enumerate(c.decode(video=0)):
            if t >= n:
                break
            img = frame.to_image()
            images.append(img)
            pdm = pipe(img)["predicted_depth"]
            dm = pdm.squeeze().detach().cpu().numpy() if hasattr(pdm, "detach") else np.array(pdm, dtype=float)
            H, W = dm.shape
            sx, sy = W / img.width, H / img.height
            for tid, x, y, w, h, conf in rows[t]["tracks"]:
                x0, x1 = int(max(0, (x - w / 2) * sx)), int(min(W, (x + w / 2) * sx))
                y0, y1 = int(max(0, (y - h / 2) * sy)), int(min(H, (y + h / 2) * sy))
                if x1 <= x0 or y1 <= y0:
                    continue
                eyes.setdefault(int(tid), {})[t] = dict(x=int(round(x)), y=int(round(y)), w=int(round(w)),
                                                       h=int(round(h)), d=int(round(100 * float(np.median(dm[y0:y1, x0:x1])))))
    rep["stages"]["depth"] = dict(wall=round(time.time() - w0, 2), cpu=round(_cpu(resource.RUSAGE_SELF) - c0, 2),
                                  note="mostly on this Mac's GPU")
    # STAGE C: AM over the eyes, window by window
    w0, c0 = time.time(), _cpu(resource.RUSAGE_SELF)
    flags = {}
    tracks = []
    for tid, fr in sorted(eyes.items()):
        ts_all = sorted(fr)
        if len(ts_all) < MIN_TRACK:
            continue
        info = dict(id=tid, frames=len(ts_all), first=ts_all[0], last=ts_all[-1], flagged=[], numbers={}, cannot=[])
        for a in range(0, n, WINDOW):
            ts = [t for t in ts_all if a <= t < a + WINDOW]
            if len(ts) < MIN_TRACK:
                continue
            m = min(WINDOW, n - a)
            _length(m)
            local = [t - a for t in ts]
            for key in KEYS:
                y = np.array([fr[t][key] for t in ts], dtype=np.int64)
                tau = 3 * sigma(y) + 0.5
                reps = A.reports(local, y, tau)
                act = A.law(A.state(reps, tau, len(local)))
                if not act or reps[act - 1] is None:
                    info["cannot"].append("%s (frames %d-%d)" % (NAMES[key], a, a + m - 1))
                    continue
                anc = reps[act - 1]["anc"]
                for t, lt, v in zip(ts, local, y):
                    if anc[lt] is not None and abs(int(v) - anc[lt]) > tau:
                        flags.setdefault((tid, t), []).append(dict(key=key, shown=int(v), rule=int(anc[lt]),
                                                                   far=abs(int(v) - anc[lt]) > 3 * tau))
                        info["numbers"][NAMES[key]] = info["numbers"].get(NAMES[key], 0) + 1
        info["flagged"] = sorted(t for (k, t) in flags if k == tid)
        tracks.append(info)
    rep["stages"]["am"] = dict(wall=round(time.time() - w0, 2), cpu=round(_cpu(resource.RUSAGE_SELF) - c0, 2), tokens=0)
    # the annotated video
    w0 = time.time()
    with tempfile.TemporaryDirectory(prefix="amv_") as tmp:
        for t, img in enumerate(images):
            im = img.convert("RGB")
            d = ImageDraw.Draw(im)
            for tid, x, y, w, h, conf in rows[t]["tracks"]:
                if not any(tk["id"] == int(tid) for tk in tracks):
                    continue
                f = flags.get((int(tid), t))
                col = (230, 40, 40) if f else (40, 200, 90)
                d.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline=col, width=3)
                label = "#%d" % int(tid) + (" broke: " + ", ".join(sorted({NAMES[q["key"]] for q in f})) if f else "")
                d.text((x - w / 2 + 4, y - h / 2 + 4), label, fill=col)
            d.text((8, 8), "AM-Video  frame %d / %d" % (t, n - 1), fill=(255, 255, 255))
            im.save(os.path.join(tmp, "f_%05d.png" % t))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "f_%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_mp4], check=True)
    rep["stages"]["annotate"] = dict(wall=round(time.time() - w0, 2))
    of = sum(t["frames"] for t in tracks)
    rep.update(tracks=tracks, object_frames=of, flagged=len(flags))
    secs = rep["seconds"] or 1
    machine = sum(v.get("wall", 0) for k, v in rep["stages"].items() if k != "annotate")
    rep["cost"] = dict(machine_seconds_per_minute=round(60 * machine / secs, 1),
                       owned_mac_usd_per_minute=round(60 * machine / secs / 3600 * 0.03, 4),
                       note="owned Apple-silicon Mac at about $0.03 per hour (price amortised over 3 years plus power); "
                            "0 tokens")
    return rep


def passes(data):
    """3D-passes mode: data = {"frames": [{"asset": [X, Y, Z, yaw, width, brightness] or null, "marker": [X, Y, Z],
    "grey": v}, ...]} in the trap's camera-space units"""
    import trap_world as T
    import trap_am as AMS
    fr = data["frames"]
    n = len(fr)
    T.N = n
    v = dict(eyes={}, marker={}, grey={})
    for t, f in enumerate(fr):
        if f.get("asset") is not None:
            X, Y, Z, Aa, W, B = f["asset"]
            v["eyes"][t] = dict(X=X, Y=Y, Z=Z, A=Aa, W=W, B=B)
        mx, my, mz = f["marker"]
        v["marker"][t] = dict(X=mx, Y=my, Z=mz)
        v["grey"][t] = f["grey"]
    w0, c0 = time.time(), _cpu(resource.RUSAGE_SELF)
    out, info = AMS.stabilize(v)
    restored = {t: s for t, s in out.items() if s is not None}
    withheld = sorted(t for t, s in out.items() if s is None)
    return dict(frames=n, hidden=sorted(set(range(n)) - set(v["eyes"])), broken=info["broken"],
                restored={str(t): [round(s[k], 1) for k in ("X", "Y", "Z", "A", "W", "B")] for t, s in sorted(restored.items())},
                withheld=withheld, spine=info["acts"],
                stages=dict(am=dict(wall=round(time.time() - w0, 2), cpu=round(_cpu(resource.RUSAGE_SELF) - c0, 2), tokens=0)))
