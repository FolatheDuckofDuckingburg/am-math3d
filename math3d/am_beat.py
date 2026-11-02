"""AM FINDS THE BEAT, AND THE NEW FRAMES FOLLOW IT: a clip as an ongoing math problem, extended by the law AM's core
reasoner proved on it, with every step of AM's process timed.
  THE CLIP: Sora 130, drummers dancing in a field, 150 frames at 30 fps.
  WHAT AM'S CORE PROVED: AM's eyes followed the dancers through frames 0-54; for each dancer's own up-and-down number
    (rY, the camera taken away) AM's core reasoner wrote a rule with its proof, or CANNOT.  Five dancers: REPEAT with a
    period of 14 frames, all in one phase.  Over the clip's last 55 frames (95-149, mapped again) it proved REPEAT with
    a period of 13 for the dancers it could follow: the beat sped up, and AM's core caught it.
  THE PIXELS, WITHOUT AM: the clip matches itself best when shifted by 14 frames in frames 0-54, and by 13 in 55-149.
  THE NEW FRAMES: AM's law at the end of the clip is value(t) = value(t - P), P = 13.  So frame t of the extension is
    the clip's own frame t - 13k, the latest one; the guide is the clip's last frame and then that law, phase-locked.
    Among the phase-equal cycles, the one whose start best follows the clip's last frame is used, and its last frames
    are cross-faded into the frames just before its start, so the cycle joins itself without a jump; the clip's last
    frame dissolves into the law over the first new frames, since the dance is not exactly periodic there.
  THE LOCK: the plan (the law, its proofs, every guide frame's source, and the rule the take will be held to) is
    written and its SHA-256 recorded before anything is generated.
  THE TAKE, measured two ways (rule fixed in the plan): the take's own pixels repeat best at P +- 1 frames (at 30 fps),
    and AM's eyes and core, run on the take, report the periods they prove.
    build(clip_map, window_map, out) -> out/: am_beat_guide_1280x720.mp4, prompt.txt, plan.json, plan.sha256,
        process.json, am_process.mp4
    score(out, take) -> out/score.json"""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAGS = range(4, 29)
BAND_TOP = 30                                                  # rows (of 120) below the sky: where the dancers are


def _frames(clip, size=None, gray=False):
    import av
    from PIL import Image
    out = []
    with av.open(clip) as c:
        fps = float(c.streams.video[0].average_rate)
        for f in c.decode(video=0):
            im = f.to_image().convert("L" if gray else "RGB")
            if size:
                im = im.resize(size, Image.BILINEAR)
            out.append(np.asarray(im).astype(np.float32))
    return fps, np.array(out)


def lag_curve(G, lo, hi):
    """mean |frame(t) - frame(t - p)| over t in [lo, hi), on the dancers' band: low means the video repeats every p"""
    band = G[:, BAND_TOP:, :]
    out = {}
    for p in LAGS:
        ts = [t for t in range(max(lo, p), hi)]
        if ts:
            out[p] = round(float(np.mean([np.abs(band[t] - band[t - p]).mean() for t in ts])), 3)
    return out


def repeats(mp, key="rY"):
    """the REPEAT rules AM's core proved for one number, per object: period, kept, measured, and its table"""
    m = json.load(open(mp))
    out = []
    for o in m["objects"]:
        v = o["numbers"][key]
        if v["rule"] == "REPEAT":
            P = int(v["formula"].split("mod ")[1].rstrip("]"))
            ts = [r[0] for r in o["rows"]]
            tab = [None] * P
            for t, a in zip(range(ts[0], ts[0] + len(v["am"])), v["am"]):
                if a is not None and tab[t % P] is None:
                    tab[t % P] = a
            out.append(dict(object=o["id"], period=P, kept=v["kept"], measured=v["measured"], band=v["band"], table=tab,
                            frames=[ts[0], ts[-1]]))
    return out, m


def build(clip_map, window_map, out, window_start=95, seconds=3.2, fade=4, lead=3):
    from PIL import Image
    os.makedirs(out, exist_ok=True)
    log, w0 = [], time.time()

    def step(name, t0, **info):
        log.append(dict(step=name, at=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 3), **info))

    # 1-2. what AM's core proved on the clip, from the two maps (built by am_map.py; their own times are recorded there)
    t0 = time.time()
    early, M = repeats(os.path.join(clip_map, "map.json"))
    late, Mw = repeats(os.path.join(window_map, "map.json"))
    step("AM maps the whole clip: eyes, metric depth, shared memory, AM's core reasoner (built earlier)", t0,
         clip_seconds_measured=M["total_wall"], stages=M["stages"], tokens=0,
         found="dancers' up-and-down: " + "; ".join("object %d REPEAT period %d, kept %d of %d" % (r["object"], r["period"], r["kept"], r["measured"]) for r in early))
    step("AM maps the clip's last 55 frames (%d-%d) and reasons over them again" % (window_start, window_start + 54), t0,
         clip_seconds_measured=Mw["total_wall"], stages=Mw["stages"], tokens=0,
         found="dancers' up-and-down: " + "; ".join("object %d REPEAT period %d, kept %d of %d" % (r["object"], r["period"], r["kept"], r["measured"]) for r in late))
    periods = [r["period"] for r in late]
    P = max(set(periods), key=lambda p: (periods.count(p), -p)) if periods else None
    if P is None:
        raise SystemExit("AM's core proved no REPEAT on the last stretch: CANNOT extend by a beat")
    # 3. the pixels, without AM
    t0 = time.time()
    clip = os.path.join(clip_map, "clip.mp4")
    fps, G = _frames(clip, size=(214, 120), gray=True)
    n = len(G)
    curves = {"0-54": lag_curve(G, 0, 55), "55-%d" % (n - 1): lag_curve(G, 55, n), "%d-%d" % (n - 50, n - 1): lag_curve(G, n - 50, n)}
    best = {k: sorted(c, key=c.get)[:3] for k, c in curves.items()}
    step("The pixels, without AM: how well the clip matches itself shifted by p frames", t0, best_lags=best, curves=curves)
    # 4. AM's law for the new frames, and the cycle that carries it
    t0 = time.time()
    last = n - 1
    cands = [s for s in range(last + 1 - P, P - 1, -P) if s - fade >= 0][:4]
    err = lambda a, b: float(np.abs(G[a, BAND_TOP:] - G[b, BAND_TOP:]).mean())
    scored = sorted(((err(last, s - 1) + err(s + P - 1, s - 1), s) for s in cands))
    s = scored[0][1]
    step("AM's law for the new frames: frame t = frame t - %d (its period at the end of the clip)" % P, t0,
         period_frames=P, period_seconds=round(P / fps, 4), beats_per_minute=round(60 * fps / P, 1),
         cycles_tried=[dict(start=c, frames=[c, c + P - 1], join_error=round(e, 2)) for e, c in scored], cycle=[s, s + P - 1])
    # 5. the guide: the clip's last frame, then the law, phase-locked, the cycle joined to itself by a cross-fade
    t0 = time.time()
    _, F = _frames(clip)
    K = int(round(seconds * fps))
    C, src = [], []
    for i in range(P):
        if i < P - fade:
            C.append(F[s + i]); src.append([[s + i, 1.0]])
        else:
            w = (i - (P - fade) + 1) / (fade + 1)
            C.append((1 - w) * F[s + i] + w * F[s + i - P]); src.append([[s + i, round(1 - w, 3)], [s + i - P, round(w, 3)]])
    guide_src = [[[last, 1.0]]] + [src[(j - 1) % P] for j in range(1, K)]
    lead_w = {j: j / (lead + 1) for j in range(1, lead + 1)}         # the clip's last frame dissolves into the law
    for j, v in lead_w.items():
        guide_src[j] = [[last, round(1 - v, 3)]] + [[a, round(b * v, 3)] for a, b in guide_src[j]]
    gpath = os.path.join(out, "am_beat_guide_1280x720.mp4")
    with tempfile.TemporaryDirectory() as tmp:
        for j in range(K):
            im = F[last] if j == 0 else C[(j - 1) % P]
            if j in lead_w:
                im = (1 - lead_w[j]) * F[last] + lead_w[j] * im
            Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).resize((1280, 720), Image.LANCZOS).save(os.path.join(tmp, "f_%04d.png" % j))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(int(fps)), "-i", os.path.join(tmp, "f_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", gpath], check=True)
    step("The guide drawn: %d frames at %d fps, 1280x720; every frame's source recorded" % (K, fps), t0, path=os.path.basename(gpath))
    # 6. the prompt
    prompt = ("Edit this video. Keep the camera, the field, the sky and every drummer exactly where they are in each frame. "
              "The drummers keep dancing and drumming exactly to the rhythm of this video: one bounce every %.2f seconds, on "
              "the same beat, the whole time. Make every movement smooth, natural and alive, with no jumps and no loop you can "
              "see. Keep their faces, bodies, clothes, drums and shadows as they are. Do not add, remove or move anyone or "
              "anything else." % (P / fps))
    open(os.path.join(out, "prompt.txt"), "w").write(prompt + "\n")
    # 7. the lock
    t0 = time.time()
    plan = dict(question="extend the drummers' dance past the clip's last frame, on the beat AM's core proved",
                clip=M["clip"], clip_frames=n, fps=fps,
                am_core=dict(frames_0_54=early, last_55=late),
                pixels_without_am=dict(best_lags=best),
                law=dict(text="value(t) = value(t - P): frame t of the extension is the clip's frame t - %dk, the latest" % P,
                         P=P, seconds=round(P / fps, 4), proved_by=[r["object"] for r in late if r["period"] == P]),
                guide=dict(frames=K, fps=fps, size=[1280, 720], cycle=[s, s + P - 1], fade_frames=fade, lead_frames=lead,
                           sources=guide_src),
                rule=dict(text="the take, resampled to %d fps, repeats best (dancers' band, lags %d-%d) at %d +- 1 frames; "
                               "AM's eyes and core run on the take, and the periods they prove are reported" % (fps, LAGS[0], LAGS[-1], P),
                          period=P, tolerance=1),
                prompt=prompt)
    blob = json.dumps(plan, indent=1).encode()
    open(os.path.join(out, "plan.json"), "wb").write(blob)
    seal = hashlib.sha256(blob).hexdigest()
    open(os.path.join(out, "plan.sha256"), "w").write(seal + "  plan.json\n")
    step("The lock: plan.json written, SHA-256 recorded before anything is generated", t0, sha256=seal)
    json.dump(dict(steps=log, total_seconds_this_run=round(time.time() - w0, 2), tokens=0), open(os.path.join(out, "process.json"), "w"), indent=1)
    return plan, seal, log


def score(out, take):
    """the take measured two ways, as the plan says: its pixels' best repeat, and what AM's core proves on it"""
    import am_map
    w0 = time.time()
    plan = json.load(open(os.path.join(out, "plan.json")))
    seal = open(os.path.join(out, "plan.sha256")).read().split()[0]
    lock_ok = hashlib.sha256(open(os.path.join(out, "plan.json"), "rb").read()).hexdigest() == seal
    fps, P = plan["fps"], plan["law"]["P"]
    t30 = os.path.join(out, "take_30fps_854x480.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", take, "-vf", "fps=%d,scale=854:480:flags=lanczos" % fps, "-an",
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", t30], check=True)
    _, G = _frames(t30, size=(214, 120), gray=True)
    curve = lag_curve(G, 0, len(G))
    best = sorted(curve, key=curve.get)
    pix_ok = abs(best[0] - P) <= plan["rule"]["tolerance"]
    md = os.path.join(out, "map_take")
    if os.path.exists(md):
        shutil.rmtree(md)
    am_map.build(t30, md, "outdoor")
    found, _ = repeats(os.path.join(md, "map.json"))
    core_hits = [r for r in found if abs(r["period"] - P) <= plan["rule"]["tolerance"]]
    res = dict(lock=seal, lock_ok=lock_ok, rule=plan["rule"]["text"], take_frames_at_30fps=len(G),
               pixels=dict(best_lags=best[:4], curve=curve, at_P=curve.get(P), keeps_the_beat=pix_ok),
               am_core_on_take=dict(repeats_up_down=found, at_P_within_1=[r["object"] for r in core_hits]),
               verdict=("KEEPS AM'S BEAT" if pix_ok else "OFF AM'S BEAT") if lock_ok else "LOCK BROKEN",
               seconds=round(time.time() - w0, 2), tokens=0)
    json.dump(res, open(os.path.join(out, "score.json"), "w"), indent=1)
    return res


def process_video(out, clip_map, window_map, window_start=95):
    """AM's process, recorded: its eyes on the dancers with what its core proved, frame by frame, the pixels' own
    answer, the second look at the last 55 frames, the law and the lock, then the guide; each part with its time"""
    from PIL import Image, ImageDraw, ImageFont
    plan = json.load(open(os.path.join(out, "plan.json")))
    seal = open(os.path.join(out, "plan.sha256")).read().split()[0]
    proc = json.load(open(os.path.join(out, "process.json")))["steps"]
    font = lambda n: ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", n)
    f1, f2, f3 = font(30), font(20), font(16)
    BG, INK, MUT, GRN, AMB = (14, 20, 18), (236, 242, 239), (150, 162, 157), (70, 220, 140), (255, 190, 70)
    COL = [(255, 99, 99), (76, 195, 255), (255, 196, 0), (180, 120, 255), (70, 226, 138), (255, 122, 200)]
    W, H = 1280, 720
    frames = []

    def card(lines, n):
        im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); y = 210
        for txt, f, c in lines:
            d.text((80, y), txt, fill=c, font=f); y += f.size + 18
        frames.extend([im] * n)

    def section(clip, mp, start, P, caption, slow=2):
        import av
        m = json.load(open(os.path.join(mp, "map.json")))
        objs = [o for o in m["objects"] if o["numbers"]["rY"]["rule"] == "REPEAT" and
                int(o["numbers"]["rY"]["formula"].split("mod ")[1].rstrip("]")) == P]
        with av.open(clip) as c:
            vid = [f.to_image().convert("RGB") for f in c.decode(video=0)]
        n = 55
        for k in range(n):
            im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
            fr = vid[start + k].resize((853, 480)); fd = ImageDraw.Draw(fr)
            for i, o in enumerate(objs):
                r = next((r for r in o["rows"] if r[0] == k), None)
                if r:
                    x, y, w, h = r[1:5]; cc = COL[i % len(COL)]
                    fd.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline=cc, width=3)
            im.paste(fr, (0, 0))
            # each dancer's up-and-down (rY): measured dots, AM's REPEAT rule as a line with its band
            sh = 480 // max(1, len(objs))
            for i, o in enumerate(objs):
                v, cc, top = o["numbers"]["rY"], COL[i % len(COL)], i * sh
                ys = [r[12] for r in o["rows"]]; am = [a for a in v["am"]]
                lo, hi = min(ys + [a for a in am if a is not None]) - v["band"], max(ys + [a for a in am if a is not None]) + v["band"]
                X = lambda t: 870 + 390 * t / (n - 1)
                Y = lambda val: top + 8 + (sh - 28) * (1 - (val - lo) / (hi - lo + 1e-9))
                pts = [(X(t), Y(a)) for t, a in enumerate(am) if a is not None]
                d.polygon([(x, Y(a + v["band"])) for (x, _), a in zip(pts, [a for a in am if a is not None])] +
                          [(x, Y(a - v["band"])) for (x, _), a in reversed(list(zip(pts, [a for a in am if a is not None])))],
                          fill=(26, 48, 36))
                d.line(pts, fill=GRN, width=2)
                for r in o["rows"]:
                    if r[0] <= k:
                        d.ellipse([X(r[0]) - 2.5, Y(r[12]) - 2.5, X(r[0]) + 2.5, Y(r[12]) + 2.5], fill=cc)
                d.line([(X(k), top + 4), (X(k), top + sh - 18)], fill=MUT, width=1)
                d.text((872, top + sh - 18), "dancer %d  rY = table[t mod %d]  %d/%d" % (o["id"], P, v["kept"], v["measured"]), fill=cc, font=f3)
            d.text((24, 500), caption[0], fill=INK, font=f2)
            for j, line in enumerate(caption[1:]):
                d.text((24, 540 + 30 * j), line, fill=MUT, font=f3)
            d.text((24, 680), "frame %d   beat phase t mod %d = %d" % (start + k, P, (start + k) % P), fill=AMB, font=f2)
            frames.extend([im] * slow)

    clip = os.path.join(clip_map, "clip.mp4")
    st = {s["step"][:30]: s for s in proc}
    s1 = next(s for s in proc if s["step"].startswith("AM maps the whole clip"))
    s2 = next(s for s in proc if s["step"].startswith("AM maps the clip's last 55"))
    s3 = next(s for s in proc if s["step"].startswith("The pixels"))
    early, late, P = plan["am_core"]["frames_0_54"], plan["am_core"]["last_55"], plan["law"]["P"]
    card([("AM finds the beat", f1, INK), ("Sora 130: drummers dancing in a field, 150 frames at 30 fps", f2, MUT),
          ("AM's eyes measure every dancer; AM's core reasoner proves the rule each number follows, or says CANNOT.", f3, MUT),
          ("The new frames are that rule, continued. Every step below is timed; no tokens anywhere.", f3, MUT)], 75)
    section(clip, clip_map, 0, early[0]["period"],
            ["AM's core reasoner, frames 0-54: %d dancers move on one REPEAT, period %d frames" % (len(early), early[0]["period"]),
             "Nobody told AM there is music. It proved the rhythm from the numbers: kept " + ", ".join(str(r["kept"]) for r in early) + " of 55 frames.",
             "Mapping the clip took %.1f s on the Mac (AM's reasoning %.1f s), 0 tokens." % (s1["clip_seconds_measured"], s1["stages"]["am"]["wall"])])
    card([("Without AM: the pixels' own answer", f1, INK),
          ("Shift the video by p frames and compare it with itself; the best match is the period.", f2, MUT),
          ("frames 0-54:   best p = %s   (AM's core: %d)" % (s3["best_lags"]["0-54"][0], early[0]["period"]), f2, GRN),
          ("frames 55-149: best p = %s   (AM's core never saw these frames)" % s3["best_lags"]["55-149"][0], f2, AMB),
          ("The beat speeds up near the end. AM looks again at the last 55 frames.", f3, MUT)], 75)
    section(os.path.join(window_map, "clip.mp4"), window_map, 0, P,
            ["AM's core reasoner again, frames %d-%d: REPEAT, period %d frames" % (window_start, window_start + 54, P),
             "It caught the faster beat on its own: " + ", ".join("dancer %d kept %d of 55" % (r["object"], r["kept"]) for r in late if r["period"] == P) + ". The pixels say %d too." % s3["best_lags"]["55-149"][0],
             "This look took %.1f s (AM's reasoning %.1f s), 0 tokens." % (s2["clip_seconds_measured"], s2["stages"]["am"]["wall"])])
    card([("AM's law for the new frames", f1, INK),
          ("frame(t) = frame(t - %d):  one beat every %.3f s, %.1f beats a minute" % (P, P / plan["fps"], 60 * plan["fps"] / P), f2, GRN),
          ("phase-locked to the clip; the cycle joins itself by a cross-fade; every frame's source is in the plan", f3, MUT),
          ("Locked before generation:  SHA-256 %s" % seal[:32], f2, AMB), ("                            %s" % seal[32:], f2, AMB)], 75)
    import av
    with av.open(os.path.join(out, "am_beat_guide_1280x720.mp4")) as c:
        for k, f in enumerate(c.decode(video=0)):
            im = f.to_image().convert("RGB"); d = ImageDraw.Draw(im)
            d.rectangle([0, H - 64, W, H], fill=BG)
            d.text((24, H - 50), "AM's guide: new frame %d   beat phase %d of %d   (the clip's last frame, then AM's law)" % (k, (k + 149) % P, P), fill=AMB, font=f2)
            frames.extend([im, im])
    path = os.path.join(out, "am_process.mp4")
    with tempfile.TemporaryDirectory() as tmp:
        for i, im in enumerate(frames):
            im.save(os.path.join(tmp, "f_%05d.png" % i))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", os.path.join(tmp, "f_%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path], check=True)
    return path


if __name__ == "__main__":
    if sys.argv[1] == "video":
        print(process_video(sys.argv[2], sys.argv[3], sys.argv[4]))
        sys.exit()
    if sys.argv[1] == "build":
        plan, seal, log = build(sys.argv[2], sys.argv[3], sys.argv[4])
        for s in log:
            print("%-100s %7.2f s" % (s["step"][:100], s["seconds"]))
        print("lock", seal)
    else:
        r = score(sys.argv[2], sys.argv[3])
        print(json.dumps({k: v for k, v in r.items() if k != "pixels"}, indent=1), r["pixels"]["best_lags"])
