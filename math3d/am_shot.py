"""THE SHOT: an intent asked on AM's puzzle, worked out as numbers for every frame, locked before anything is generated,
drawn as a guide video for a video model; then the take that comes back, measured against the lock frame by frame.
  THE LOCK: the plan (the question, AM's answer for every frame, and the rule the take will be held to) is written
    once, and its SHA-256 is recorded before any video is generated.  Nothing rewrites it afterwards.
  THE MEASUREMENT: Apple Vision's tracker follows the moving piece from its box in the first frame, the same tracker
    with the same start on AM's guide and on the take, both at the puzzle frame's size, so the tracker's own bias
    cancels.  The distance between the two at the same moment is the score.  Six textured background patches,
    away from the piece's path and every other piece, are tracked the same way: their shift between guide and take
    is how far the camera or the background moved.
  THE RULE, fixed before generation: within RULE_PX px of AM's guide at every frame of AM's plan, ending within
    RULE_PX px; frames after the plan ends are measured and shown, not judged.
    plan(mapdir, q) -> mapdir/shots/<sid>/: plan.json, plan.sha256, guide.mp4, guide_end.png, prompt.txt, shot.json
    score(mapdir, sid, take_mp4) -> mapdir/shots/<sid>/: take.mp4, score.json, labelled.mp4"""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am_intent                                               # noqa: E402

TRACK = os.path.join(HERE, "track_from")
RULE_PX = 5.0
LOST_SHARE = 0.10                                              # more frames lost than this: the take is not measured
MOVER, PATCH, NPATCH = 100, 48, 6


def shot_id(mapdir, q):
    pz = json.load(open(os.path.join(mapdir, "puzzle.json")))
    key = dict(q, frame=pz["frame"], clip=pz.get("clip"))
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def prompt(plan, piece):
    q, a = plan["question"], plan["answer"]
    name = (q.get("name") or "").strip() or "the piece that moves in this video"
    t0, t1 = a["take_off"] / a["fps"], (a["take_off"] + a["air_frames"]) / a["fps"]
    how = "slides along the surface, without lifting," if q["mode"] == "slide" else "jumps in one arc"
    look = ("Make it look naturally alive as it moves." if piece.get("kind") == "subject"
            else "Make it look real, with its shadow moving with it.")
    return ("Edit this video. Keep every frame exactly as it is: the camera, the background and every object stay where "
            "they are. Only one thing moves: %s. It %s exactly as in this video, starting at %.1f seconds and stopping "
            "at %.1f seconds, then stays there. Follow its position and its size in every frame of this video exactly. "
            "Nothing else touches it. %s Where it stood at the start, the soft blurred patch is only the background behind "
            "it: paint it as that background, sharp and natural. Do not add, move or remove anything else." % (name, how, t0, t1, look))


def plan(mapdir, q, progress=print):
    """q: mover, onto (a piece) or across/forward (a spot on the stage), mode (slide or jump), height (cm, a jump),
    duration (s, a slide), start (s), length (s), name (for the prompt)"""
    w0 = time.time()
    sid = shot_id(mapdir, q)
    d = os.path.join(mapdir, "shots", sid)
    os.makedirs(d, exist_ok=True)
    pz = json.load(open(os.path.join(mapdir, "puzzle.json")))
    piece = next(p for p in pz["pieces"] if p["id"] == int(q["mover"]))
    target = int(q["onto"]) if q.get("onto") not in (None, "") else dict(across=float(q["across"]), forward=float(q["forward"]))
    progress("shot-solve")
    p = am_intent.solve(mapdir, int(q["mover"]), target, height=q.get("height"), wait=float(q["start"]),
                        seconds=float(q["length"]), mode=q["mode"], duration=q.get("duration"), write=False)
    p["question"]["name"] = q.get("name") or ""
    p["rule"] = dict(px=RULE_PX, at=p["size"], text="within %g px of AM's guide at every frame of this plan, ending "
                     "within %g px, measured at %dx%d with the same tracker on the guide and the take"
                     % (RULE_PX, RULE_PX, p["size"][0], p["size"][1]))
    p["clip"], p["shot"] = pz.get("clip"), sid
    blob = json.dumps(p, indent=1).encode()
    open(os.path.join(d, "plan.json"), "wb").write(blob)
    seal = hashlib.sha256(blob).hexdigest()
    open(os.path.join(d, "plan.sha256"), "w").write(seal + "  plan.json\n")
    shot = dict(id=sid, question=q, verdict=p["verdict"], why=p["why"], checks=p["checks"], sha256=seal,
                fps=p["answer"]["fps"], frames=p["answer"]["frames"], take_off=p["answer"]["take_off"],
                moving_frames=p["answer"]["air_frames"], equation=p["answer"]["equation"], size=p["size"],
                solve_seconds=round(time.time() - w0, 2), tokens=0, made=time.strftime("%Y-%m-%d %H:%M:%S"))
    if p["verdict"] == "SOLVED":
        progress("shot-guide")
        w1 = time.time()
        am_intent.render(mapdir, out=os.path.join(d, "guide.mp4"), plan=p, end=os.path.join(d, "guide_end.png"),
                         height=720, fill=q.get("fill", "smooth"), occlude=True)
        shot["guide_seconds"] = round(time.time() - w1, 2)
        shot["prompt"] = prompt(p, piece)
        open(os.path.join(d, "prompt.txt"), "w").write(shot["prompt"] + "\n")
    json.dump(shot, open(os.path.join(d, "shot.json"), "w"), indent=1)
    return shot


def _norm(src, dst, W, H):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", "scale=%d:%d:flags=lanczos" % (W, H), "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", dst], check=True)
    import av
    with av.open(dst) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate)
        first = None
        n = 0
        for f in c.decode(video=0):
            if first is None:
                first = f.to_ndarray(format="rgb24")
            n += 1
    return fps, n, first


def _track(video, boxes):
    r = subprocess.run([TRACK, video, json.dumps(boxes)], capture_output=True, text=True, check=True)
    out = {}
    for line in r.stdout.splitlines():
        if line.strip():
            row = json.loads(line)
            for tr in row["tracks"]:
                out.setdefault(int(tr[0]), {})[row["t"]] = tr[1:]
    return out


def _am_box(p, r):
    """AM's box for the piece at row r: its box in the puzzle frame, scaled by perspective and carried by the foot"""
    x0, y0, bw, bh = p["mover_box"]
    r0 = p["rows"][0]
    s = r["h"] / r0["h"]
    left = r["foot"][0] + (x0 - r0["foot"][0]) * s
    top = r["foot"][1] + (y0 - r0["foot"][1]) * s
    return left + bw * s / 2, top + bh * s / 2, bw * s, bh * s


def score(mapdir, sid, take, progress=print):
    w0 = time.time()
    d = os.path.join(mapdir, "shots", sid)
    if os.path.abspath(take) != os.path.abspath(os.path.join(d, "take.mp4")):
        shutil.copyfile(take, os.path.join(d, "take.mp4"))
    blob = open(os.path.join(d, "plan.json"), "rb").read()
    seal = open(os.path.join(d, "plan.sha256")).read().split()[0]
    lock_ok = hashlib.sha256(blob).hexdigest() == seal
    p = json.loads(blob)
    W, H = p["size"]
    rows, fps_plan = p["rows"], p["answer"]["fps"]
    plan_end = rows[-1]["t"]
    pz = json.load(open(os.path.join(mapdir, "puzzle.json")))
    with tempfile.TemporaryDirectory() as tmp:
        progress("take-normalise")
        gn, tn = os.path.join(tmp, "guide.mp4"), os.path.join(tmp, "take.mp4")
        gfps, gcount, g0 = _norm(os.path.join(d, "guide.mp4"), gn, W, H)
        tfps, tcount, _ = _norm(os.path.join(d, "take.mp4"), tn, W, H)
        x0, y0, bw, bh = p["mover_box"]
        start = [MOVER, x0 + bw / 2, y0 + bh / 2, bw, bh]
        avoid = [_am_box(p, r) for r in rows] + [(q["box"][0] + q["box"][2] / 2, q["box"][1] + q["box"][3] / 2,
                                                  q["box"][2], q["box"][3]) for q in pz["pieces"]]
        import am3d
        patches = am3d.pick_markers(g0, avoid, W, H)
        boxes = [start] + [[i + 1, x, y, PATCH, PATCH] for i, (x, y) in enumerate(patches)]
        progress("take-track")
        G, T = _track(gn, boxes), _track(tn, boxes)
        frames, drift = [], []
        for k in range(tcount):
            t = k / tfps
            r = min(rows, key=lambda q: abs(q["t"] - t))
            j = min(gcount - 1, int(round(t * gfps)))
            am = _am_box(p, r)
            g, tk = G.get(MOVER, {}).get(j), T.get(MOVER, {}).get(k)
            shifts = [np.subtract(T[i][k][:2], G[i][j][:2]) for i in range(1, len(patches) + 1)
                      if k in T.get(i, {}) and j in G.get(i, {})]
            dv = np.median(shifts, axis=0) if shifts else None
            drift.append(None if dv is None else round(float(np.hypot(*dv)), 2))
            frames.append(dict(k=k, t=round(t, 3), in_plan=t <= plan_end + 0.5 / tfps, am=[round(v, 1) for v in am],
                               guide=None if g is None else [round(v, 1) for v in g[:4]],
                               take=None if tk is None else [round(v, 1) for v in tk[:4]],
                               off=None if (g is None or tk is None) else round(float(np.hypot(tk[0] - g[0], tk[1] - g[1])), 2),
                               off_am=None if tk is None else round(float(np.hypot(tk[0] - am[0], tk[1] - am[1])), 2),
                               bias=None if g is None else round(float(np.hypot(g[0] - am[0], g[1] - am[1])), 2),
                               drift=drift[-1]))
        progress("take-label")
        label(os.path.join(d, "take.mp4"), os.path.join(d, "labelled.mp4"), p, frames, W, H)
    inp = [f for f in frames if f["in_plan"]]
    got = [f["off"] for f in inp if f["off"] is not None]
    lost = len(inp) - len(got)
    first_lost = next((f["t"] for f in inp if f["off"] is None), None)
    end = next((f["off"] for f in reversed(inp) if f["off"] is not None), None)
    worst = max(got) if got else None
    worst_t = next((f["t"] for f in inp if f["off"] == worst), None) if got else None
    dr = [f["drift"] for f in inp if f["drift"] is not None]
    bias = [f["bias"] for f in inp if f["bias"] is not None]
    if not lock_ok:
        verdict, why = "LOCK BROKEN", "plan.json no longer matches the SHA-256 recorded before generation"
    elif not got or lost > LOST_SHARE * len(inp):
        verdict, why = "NOT MEASURED", ("the tracker lost the piece from %.2f s; %d of %d frames of AM's plan unmeasured"
                                        % (first_lost or 0.0, lost, len(inp)))
    elif worst <= RULE_PX and end is not None and end <= RULE_PX:
        verdict, why = "PASS", "within %g px of AM's guide at every measured frame of the plan, and at the end" % RULE_PX
    else:
        over = sum(v > RULE_PX for v in got)
        verdict, why = "FAIL", ("%d of %d measured frames are more than %g px from AM's guide; the worst is %.1f px at %.2f s"
                                % (over, len(got), RULE_PX, worst, worst_t))
    res = dict(shot=sid, lock=seal, lock_ok=lock_ok, rule=p["rule"]["text"], verdict=verdict, why=why,
               take=dict(fps=round(tfps, 3), frames=tcount, seconds=round(tcount / tfps, 3)),
               guide=dict(fps=round(gfps, 3), frames=gcount),
               plan_seconds=plan_end, measured=len(got), in_plan=len(inp), lost=lost,
               median_px=None if not got else round(float(np.median(got)), 2), worst_px=worst, worst_t=worst_t, end_px=end,
               camera_drift_max_px=max(dr) if dr else None, tracker_bias_median_px=round(float(np.median(bias)), 2) if bias else None,
               patches=len(patches), seconds=round(time.time() - w0, 2), tokens=0, frames=frames)
    json.dump(res, open(os.path.join(d, "score.json"), "w"), indent=1)
    return res


def label(take, out, p, frames, W, H):
    """the take with AM's box (green) and the tracked piece (amber) at every frame, and the distance under it"""
    import av
    from PIL import Image, ImageDraw, ImageFont
    OW, OH = 1280, int(round(1280 * H / W / 2)) * 2
    S = OW / W
    font = lambda n: ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", n)
    fb, fs = font(24), font(18)
    GREEN, AMBER, WHITE, GREY = (70, 220, 140), (255, 190, 70), (240, 244, 242), (150, 160, 158)
    path = [(r["foot"][0] * S, r["foot"][1] * S) for r in p["rows"]]
    with tempfile.TemporaryDirectory() as tmp:
        with av.open(take) as c:
            fps = float(c.streams.video[0].average_rate)
            for k, fr in enumerate(c.decode(video=0)):
                if k >= len(frames):
                    break
                f = frames[k]
                canvas = Image.new("RGB", (OW, OH + 120), (14, 20, 18))
                canvas.paste(fr.to_image().convert("RGB").resize((OW, OH), Image.LANCZOS), (0, 0))
                dr = ImageDraw.Draw(canvas)
                dr.line(path, fill=GREEN, width=2)
                cx, cy, w, h = [v * S for v in f["am"]]
                dr.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=GREEN, width=3)
                if f["take"]:
                    tx, ty, tw, th = [v * S for v in f["take"]]
                    dr.rectangle([tx - tw / 2, ty - th / 2, tx + tw / 2, ty + th / 2], outline=AMBER, width=2)
                    dr.line([(tx - 10, ty), (tx + 10, ty)], fill=AMBER, width=3)
                    dr.line([(tx, ty - 10), (tx, ty + 10)], fill=AMBER, width=3)
                state = "in AM's plan" if f["in_plan"] else "after AM's plan"
                dr.text((20, OH + 14), "t %.2f s   %s" % (f["t"], state), fill=WHITE, font=fb)
                dr.text((20, OH + 52), "green: AM's locked box and path    amber: the piece in this take", fill=GREY, font=fs)
                off = ("off AM's guide %.1f px at %dx%d" % (f["off"], W, H)) if f["off"] is not None else "the tracker lost the piece here"
                dr.text((20, OH + 82), off + ("" if f["drift"] is None else "    background shift %.1f px" % f["drift"]),
                        fill=AMBER, font=fs)
                canvas.save(os.path.join(tmp, "f_%04d.png" % k))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "%.3f" % fps, "-i", os.path.join(tmp, "f_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        s = plan(sys.argv[2], json.loads(sys.argv[3]))
        print(json.dumps({k: v for k, v in s.items() if k != "prompt"}, indent=1))
    else:
        r = score(sys.argv[2], sys.argv[3], sys.argv[4])
        print(json.dumps({k: v for k, v in r.items() if k != "frames"}, indent=1))
