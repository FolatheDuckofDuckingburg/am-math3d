"""AM DICT: a real clip's scene written down as integer rules over the frame number, and drawn back from them alone.
  THE EYES (AM-Video's video mode, unchanged): Apple Vision tracks every object (video_eyes); Depth Anything V2 Small
    gives each object's depth.  Each object's x, y, w, h (px) and d (depth x 100) in every frame; cached in eyes.json
    with what each stage cost.
  AM (AM-Video's sealed hives and number law, unchanged): for each number of each object in each window, the law
    names the hive whose rule holds the number, or CANNOT.  The band is AM-Video's: tau = 3 sigma + 0.5, sigma from
    the number's own second differences.
  THE DICT writes each named rule down as mathematics over t, the frame counted from the window's start:
    EXACT   s(t): a chain of at most 3 steps from AM's shelf
    SHAPE   a*s(t) + b, or a constant
    REPEAT  table[t mod p]
    LOCAL   a straight line through the frames near each frame: values kept per frame, not a global rule
    CANNOT  no rule: the dict holds nothing for that number
  When a hive's best fits tie, AM anchors at their median; the dict keeps the tied fit closest to that median and
  records how far from it the kept fit ever goes.
  THE NULL: the same AM on each number's measurements shuffled in time, with the same band.  Rules found as often on
    shuffled numbers would be the band's doing, not the clip's.
  THE 2D PROJECTION: the dict evaluated at every frame, drawn beside the clip.
    python3 am_dict.py CLIP OUTDIR [WINDOW ...]"""
import json, os, re, resource, subprocess, sys, tempfile, time
import numpy as np

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "am")   # AM's core, next to this folder
sys.path.insert(0, APP)
import am_video_core as C                                     # noqa: E402  (AM-Video: eyes, band, hives, law)

A, S = C.A, C.S
KEYS = C.KEYS
HIVE = {1: "EXACT", 2: "SHAPE", 3: "LOCAL", 4: "REPEAT"}
MAC_USD_PER_HOUR = 0.03                                       # owned Apple-silicon Mac: price over 3 years plus power
_STEP = {name: f for name, f in S.SHELF}


def _cpu(who):
    r = resource.getrusage(who)
    return r.ru_utime + r.ru_stime


def bits(v):
    return 0 if v is None else abs(int(v)).bit_length() + 1


# ---------------------------------------------------------------- the eyes
def measure(clip, out):
    path = os.path.join(out, "eyes.json")
    if os.path.exists(path):
        return json.load(open(path))
    import av
    with av.open(clip) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate or 30)
        size = [st.width, st.height]
    w0, c0 = time.time(), _cpu(resource.RUSAGE_CHILDREN)
    tr = subprocess.run([os.path.join(APP, "video_eyes"), clip], capture_output=True, text=True, check=True)
    rows = [json.loads(l) for l in tr.stdout.splitlines() if l.strip()][:C.MAX_FRAMES]
    stages = dict(tracking=dict(wall=round(time.time() - w0, 2), cpu=round(_cpu(resource.RUSAGE_CHILDREN) - c0, 2)))
    n = len(rows)
    w0, c0 = time.time(), _cpu(resource.RUSAGE_SELF)
    pipe = C.depth_model()
    load = time.time() - w0
    eyes = {}
    with av.open(clip) as c:
        for t, frame in enumerate(c.decode(video=0)):
            if t >= n:
                break
            img = frame.to_image()
            pdm = pipe(img)["predicted_depth"]
            dm = pdm.squeeze().detach().cpu().numpy() if hasattr(pdm, "detach") else np.array(pdm, dtype=float)
            H, W = dm.shape
            sx, sy = W / img.width, H / img.height
            for tid, x, y, w, h, conf in rows[t]["tracks"]:
                x0, x1 = int(max(0, (x - w / 2) * sx)), int(min(W, (x + w / 2) * sx))
                y0, y1 = int(max(0, (y - h / 2) * sy)), int(min(H, (y + h / 2) * sy))
                if x1 <= x0 or y1 <= y0:
                    continue
                eyes.setdefault(str(int(tid)), {})[str(t)] = dict(
                    x=int(round(x)), y=int(round(y)), w=int(round(w)), h=int(round(h)),
                    d=int(round(100 * float(np.median(dm[y0:y1, x0:x1])))))
    stages["depth"] = dict(wall=round(time.time() - w0, 2), cpu=round(_cpu(resource.RUSAGE_SELF) - c0, 2),
                           model_load=round(load, 2), note="mostly on this Mac's GPU")
    data = dict(clip=os.path.basename(clip), fps=fps, size=size, frames=n, stages=stages, eyes=eyes)
    json.dump(data, open(path, "w"))
    return data


# ---------------------------------------------------------------- rules written down
def chain_of(target, n):
    """the steps from AM's shelf that turn t = 0..n-1 into the target behaviour exactly, shallowest first"""
    lv = S.levels(list(range(n)))
    if np.array_equal(target, lv[0][1][0]):
        return []
    for d in range(1, len(lv)):
        prev = lv[d - 1][1]
        for name, f in S.SHELF:
            with np.errstate(all="ignore"):
                y = f(prev)
            hit = np.nonzero(np.all(y == target, axis=1) & np.all(np.abs(y) < S.look.LIM, axis=1))[0]
            if len(hit):
                return chain_of(prev[hit[0]], n) + [name]
    return None


def apply_chain(chain, t):
    v = np.array(t, dtype=np.int64)
    for name in chain:
        with np.errstate(all="ignore"):
            v = _STEP[name](v)
    return v


def expr_of(chain):
    e = "t"
    for name in chain:
        e = re.sub(r"\bx\b", lambda m, e=e: e if e == "t" else "(" + e + ")", name)
    return e


def _search_ids(fit, shape):
    """A._search, keeping each tied best fit's values, depth, a, b and base behaviour"""
    best, cands = -1, []
    for d, Y in A.eyes():
        dmax, dc = -1, []
        for c in range(0, len(Y), A.CHUNK):
            B = Y[c:c + A.CHUNK]
            if shape:
                inl, ai, bi = fit(B)
            else:
                inl = fit(B)
            mx = int(inl.max()) if len(inl) else -1
            if mx < 0 or mx < dmax:
                continue
            sel = np.nonzero(inl == mx)[0][:2000]
            if shape:
                rows = [(B[i] * ai[i] + bi[i], d, int(ai[i]), int(bi[i]), B[i]) for i in sel]
            else:
                rows = [(B[i], d, 1, 0, B[i]) for i in sel]
            if mx > dmax:
                dmax, dc = mx, rows
            else:
                dc += rows[:2000 - len(dc)]
        if dmax > best:
            best, cands = dmax, dc
    return best, cands


def _fit_exact(ts, y, tau):
    return lambda B: (np.abs(B[:, ts] - y) <= tau).sum(1)


def _fit_shape(ts, y, tau):
    yf = y.astype(float)

    def fit(B):                                                # anchor_am.shape's fit, line for line
        Sx = B[:, ts].astype(float)
        sm = Sx.mean(1)
        ds = Sx - sm[:, None]
        var = (ds * ds).sum(1)
        ok = var > 0
        a = np.where(ok, (ds * (yf - yf.mean())).sum(1) / np.where(ok, var, 1.0), 0.0)
        b = np.median(yf - a[:, None] * Sx, axis=1)
        w = np.abs(yf - (a[:, None] * Sx + b[:, None])) <= 3 * tau
        nw = np.maximum(w.sum(1), 1)
        sw = (Sx * w).sum(1) / nw
        yw = (yf * w).sum(1) / nw
        dsw = (Sx - sw[:, None]) * w
        varw = (dsw * (Sx - sw[:, None])).sum(1)
        a2 = np.where(varw > 0, (dsw * (yf - yw[:, None])).sum(1) / np.where(varw > 0, varw, 1.0), 0.0)
        ai = np.rint(a2)
        ok &= ai != 0
        bi = np.rint(np.median(yf - ai[:, None] * Sx, axis=1))
        inl = (np.abs(yf - (ai[:, None] * Sx + bi[:, None])) <= tau).sum(1)
        inl[~ok] = -1
        return inl, ai, bi
    return fit


def write_rule(act, ts, y, tau, rep):
    """the rule AM's law named, written down; its values over the window equal AM's anchors or sit within the band"""
    N, anc = A.N, np.array([np.nan if v is None else v for v in rep["anc"]], dtype=float)
    if act in (1, 2):
        best, cands = _search_ids((_fit_shape if act == 2 else _fit_exact)(ts, y, tau), act == 2)
        if act == 2:
            yf = y.astype(float)
            med = int(round(float(np.median(yf))))
            if int((np.abs(yf - med) <= tau).sum()) >= best:
                cands = [(np.full(N, med, dtype=float), 0, 0, med, None)]
        vals = np.array([c[0] for c in cands], dtype=float)
        assert np.array_equal(np.rint(np.median(vals, 0)), anc), "the rule search differs from AM's"
        dev = np.abs(vals - anc).max(1)
        v, d, a, b, base = cands[int(np.argmin(dev))]
        if base is None:
            return dict(rule="SHAPE", kind="constant", formula="%d" % b, a=0, b=b, depth=0, chain=[],
                        tied=1, off_anchor_max=0)
        chain = chain_of(np.array(base, dtype=np.int64), N)
        assert chain is not None and np.array_equal(apply_chain(chain, np.arange(N)), base), "no chain"
        s = expr_of(chain)
        formula = s if act == 1 else ("%s%s" % ("" if a == 1 else "-" if a == -1 else "%d*" % a,
                                                s if a in (1, -1) and s == "t" else "(%s)" % s)
                                      + ("" if b == 0 else " + %d" % b if b > 0 else " - %d" % -b))
        return dict(rule=HIVE[act], kind="formula", formula=formula, a=a, b=b, depth=d, chain=chain,
                    tied=len(cands), off_anchor_max=int(dev.min()))
    if act == 4:
        val = dict(zip(ts, (int(v) for v in y)))
        best = None
        for p in range(2, 25):                                 # anchor_am.repeat, keeping the period
            anc_p = []
            for t in range(N):
                vs = [val[u] for u in range(t % p, N, p) if u in val]
                anc_p.append(int(round(float(np.median(vs)))) if vs else None)
            out = sum(abs(val[t] - anc_p[t]) > tau for t in ts)
            if any(sum(1 for u in range(r, N, p) if u in val) >= 2 for r in range(p)) and \
                    (best is None or out < best[0]):
                best = (out, p, anc_p)
        out, p, anc_p = best
        assert anc_p == rep["anc"], "the repeat search differs from AM's"
        return dict(rule="REPEAT", kind="table", formula="table[t mod %d]" % p, period=p, table=anc_p[:p],
                    tied=1, off_anchor_max=0)
    return dict(rule="LOCAL", kind="values", formula="a straight line through the frames within 5 of t",
                values=rep["anc"], tied=1, off_anchor_max=0)


def evaluate(rule, n):
    if rule["rule"] == "CANNOT":
        return [None] * n
    if rule["kind"] == "constant":
        return [rule["b"]] * n
    if rule["kind"] == "formula":
        return [int(v) for v in rule["a"] * apply_chain(rule["chain"], np.arange(n)) + rule["b"]]
    if rule["kind"] == "table":
        tb = rule["table"]
        return [tb[t % rule["period"]] for t in range(n)]
    return list(rule["values"])


def rule_bits(rule):
    k = rule.get("kind")
    head = 3 + 8                                               # which hive, and the band
    if rule["rule"] == "CANNOT":
        return 3
    if k == "constant":
        return head + bits(rule["b"])
    if k == "formula":
        return head + 2 + 8 * len(rule["chain"]) + bits(rule["a"]) + bits(rule["b"])
    if k == "table":
        return head + 5 + sum(bits(v) for v in rule["table"])
    return head + sum(bits(v) for v in rule["values"])


# ---------------------------------------------------------------- AM over the clip
def build(data, window, null_seed=None, tail=False):
    """the dict for one window size (tail: only the clip's last WINDOW frames, the motion an extension continues);
    with null_seed, AM on each number shuffled in time instead (no dict kept)"""
    n = data["frames"]
    rng = np.random.default_rng(null_seed) if null_seed is not None else None
    objects, t_am, t_dict, ruled, total, null_ruled = [], 0.0, 0.0, 0, 0, 0
    for tid, fr in sorted(data["eyes"].items(), key=lambda kv: int(kv[0])):
        ts_all = sorted(int(t) for t in fr)
        if len(ts_all) < C.MIN_TRACK:
            continue
        obj = dict(id=int(tid), first=ts_all[0], last=ts_all[-1], measured=len(ts_all), windows=[])
        for a in ([max(0, n - window)] if tail else range(0, n, window)):
            ts = [t for t in ts_all if a <= t < a + window]
            if len(ts) < C.MIN_TRACK:
                continue
            m = min(window, n - a)
            C._length(m)
            local = [t - a for t in ts]
            win = dict(start=a, end=a + m - 1, measured=len(ts), numbers={})
            for key in KEYS:
                y = np.array([fr[str(t)][key] for t in ts], dtype=np.int64)
                tau = 3 * C.sigma(y) + 0.5
                if rng is not None:
                    y = rng.permutation(y)
                w0, c0 = time.time(), _cpu(resource.RUSAGE_SELF)
                reps = A.reports(local, y, tau)
                act = A.law(A.state(reps, tau, len(local)))
                t_am += _cpu(resource.RUSAGE_SELF) - c0
                total += 1
                if not act or reps[act - 1] is None:
                    win["numbers"][key] = dict(rule="CANNOT", band=round(tau, 2))
                    continue
                ruled += 1
                if rng is not None:
                    null_ruled += 1
                    continue
                c0 = _cpu(resource.RUSAGE_SELF)
                rule = write_rule(act, local, y, tau, reps[act - 1])
                t_dict += _cpu(resource.RUSAGE_SELF) - c0
                rule["band"] = round(tau, 2)
                vals = evaluate(rule, m)
                off = [t for t, lt, v in zip(ts, local, y) if vals[lt] is not None and abs(int(v) - vals[lt]) > tau]
                rule.update(kept=len(ts) - len(off), broken=len(off), broken_frames=off)
                win["numbers"][key] = rule
            obj["windows"].append(win)
        objects.append(obj)
    return dict(window=window, objects=objects, numbers=total, ruled=ruled, am_cpu=round(t_am, 2),
                dict_cpu=round(t_dict, 2))


def project(data, dct):
    """the dict evaluated at every frame: {object id: {t: {key: value or None}}}"""
    n = data["frames"]
    out = {}
    for obj in dct["objects"]:
        per = {}
        for win in obj["windows"]:
            m = win["end"] - win["start"] + 1
            vals = {k: evaluate(win["numbers"][k], m) for k in KEYS}
            for lt in range(m):
                t = win["start"] + lt
                if obj["first"] <= t <= obj["last"]:
                    per[t] = {k: vals[k][lt] for k in KEYS}
        out[obj["id"]] = per
    return out


def summary(data, dct, proj):
    n, fps = data["frames"], data["fps"]
    broken = set()
    hives, by_key = {}, {k: [0, 0] for k in KEYS}
    dict_bits = raw_bits = 0                                   # over the numbers the dict holds a rule for
    for obj in dct["objects"]:
        fr = data["eyes"][str(obj["id"])]
        for win in obj["windows"]:
            for k, r in win["numbers"].items():
                by_key[k][1] += 1
                if r["rule"] != "CANNOT":
                    by_key[k][0] += 1
                    hives[r["rule"]] = hives.get(r["rule"], 0) + 1
                    broken |= set(r.get("broken_frames", []))
                    dict_bits += rule_bits(r)
                    raw_bits += sum(bits(e[k]) for t, e in fr.items() if win["start"] <= int(t) <= win["end"])
    boxes = sum(1 for per in proj.values() for v in per.values() if all(v[k] is not None for k in "xywh"))
    obj_frames = sum(len(per) for per in proj.values())
    seg = int(round(4 * fps))
    return dict(numbers=dct["numbers"], ruled=dct["ruled"], by_hive=hives,
                ruled_by_number={k: "%d of %d" % tuple(v) for k, v in by_key.items()},
                object_frames=obj_frames, boxes_drawable_from_rules=boxes,
                frames_with_a_break=len(broken), seconds_with_a_break=len({t // int(round(fps)) for t in broken}),
                four_second_segments_with_a_break="%d of %d" % (len({t // seg for t in broken}), -(-n // seg)),
                dict_bits=dict_bits, raw_measurement_bits=raw_bits)


# ---------------------------------------------------------------- the 2D projection, drawn
PALETTE = [(80, 160, 255), (255, 170, 60), (190, 110, 255), (60, 210, 190), (255, 110, 150), (200, 200, 80),
           (140, 200, 255)]


def _font(size):
    from PIL import ImageFont
    for p in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def render(clip, data, dct, proj, out_mp4, peek_png):
    import av
    from PIL import Image, ImageDraw
    n, fps = data["frames"], data["fps"]
    W, H = data["size"]
    bar = 72
    f_big, f_small = _font(17), _font(14)
    rules = {(obj["id"], t): win["numbers"] for obj in dct["objects"] for win in obj["windows"]
             for t in range(win["start"], win["end"] + 1)}
    colour = {obj["id"]: PALETTE[i % len(PALETTE)] for i, obj in enumerate(dct["objects"])}
    peek_at = n // 2
    with tempfile.TemporaryDirectory(prefix="amdict_") as tmp, av.open(clip) as c:
        for t, frame in enumerate(c.decode(video=0)):
            if t >= n:
                break
            canvas = Image.new("RGB", (2 * W, H + bar), (14, 16, 20))
            left = frame.to_image().convert("RGB")
            dl = ImageDraw.Draw(left)
            for tid, fr in data["eyes"].items():
                e = fr.get(str(t))
                if e is None or int(tid) not in proj:
                    continue
                nums = rules.get((int(tid), t), {})
                pv = proj[int(tid)].get(t, {})
                ruled = [k for k in KEYS if pv.get(k) is not None]
                brk = [k for k in ruled if abs(e[k] - pv[k]) > nums[k]["band"]]
                col = (230, 50, 50) if brk else (50, 205, 95) if ruled else (150, 150, 150)
                dl.rectangle([e["x"] - e["w"] / 2, e["y"] - e["h"] / 2, e["x"] + e["w"] / 2, e["y"] + e["h"] / 2],
                             outline=col, width=3)
                dl.text((e["x"] - e["w"] / 2 + 5, e["y"] - e["h"] / 2 + 3), "#%s" % tid, fill=col, font=f_small)
            dl.text((10, 8), "Sora clip, measured", fill=(255, 255, 255), font=f_big)
            right = Image.new("RGB", (W, H), (22, 25, 31))
            dr = ImageDraw.Draw(right)
            for x in range(0, W, 61):
                dr.line([(x, 0), (x, H)], fill=(32, 36, 44))
            for y in range(0, H, 61):
                dr.line([(0, y), (W, y)], fill=(32, 36, 44))
            drawn = 0
            for oid, per in proj.items():
                v = per.get(t)
                if v is None:
                    continue
                col = colour[oid]
                if all(v[k] is not None for k in "xywh"):
                    dr.rectangle([v["x"] - v["w"] / 2, v["y"] - v["h"] / 2, v["x"] + v["w"] / 2, v["y"] + v["h"] / 2],
                                 outline=col, width=3)
                    dr.text((v["x"] - v["w"] / 2 + 5, v["y"] - v["h"] / 2 + 3), "#%d" % oid, fill=col, font=f_small)
                    drawn += 1
                elif v["x"] is not None and v["y"] is not None:
                    dr.line([(v["x"] - 12, v["y"]), (v["x"] + 12, v["y"])], fill=col, width=3)
                    dr.line([(v["x"], v["y"] - 12), (v["x"], v["y"] + 12)], fill=col, width=3)
                    dr.text((v["x"] + 14, v["y"] - 8), "#%d" % oid, fill=col, font=f_small)
                    drawn += 1
                elif v["x"] is not None:
                    dr.line([(v["x"], 0), (v["x"], H)], fill=col, width=2)
                    dr.text((v["x"] + 6, H - 24), "#%d x" % oid, fill=col, font=f_small)
                    drawn += 1
            dr.text((10, 8), "Drawn from the AM dict alone", fill=(255, 255, 255), font=f_big)
            canvas.paste(left, (0, 0))
            canvas.paste(right, (W, 0))
            db = ImageDraw.Draw(canvas)
            db.text((12, H + 10), "Left: green = the object keeps its rules here, red = it breaks one, grey = AM found "
                    "no rule (CANNOT).", fill=(220, 220, 220), font=f_small)
            db.text((12, H + 34), "Right: only what the rules say: a box needs x, y, w and h; a cross means x and y; "
                    "a line means x alone.", fill=(220, 220, 220), font=f_small)
            db.text((2 * W - 190, H + 22), "frame %d / %d   window %s" % (t, n - 1, dct["window"]),
                    fill=(170, 170, 170), font=f_small)
            canvas.save(os.path.join(tmp, "f_%05d.png" % t))
            if t == peek_at:
                canvas.save(peek_png)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "f_%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_mp4], check=True)


# ---------------------------------------------------------------- the run
def main():
    clip, out = sys.argv[1], sys.argv[2]
    windows = sys.argv[3:] or ["150"]                          # "48" every 48 frames; "t48" the last 48 only
    os.makedirs(out, exist_ok=True)
    data = measure(clip, out)
    rpath = os.path.join(out, "report.json")
    report = json.load(open(rpath)) if os.path.exists(rpath) else dict(
        clip=data["clip"], frames=data["frames"], fps=data["fps"], size=data["size"], eyes_stages=data["stages"],
        windows={})
    for name in windows:
        tail, window = name.startswith("t"), int(name.lstrip("t"))
        w0 = time.time()
        dct = build(data, window, tail=tail)
        dct["window"] = name
        build_wall = time.time() - w0
        w0, c0 = time.time(), _cpu(resource.RUSAGE_SELF)
        proj = project(data, dct)
        proj_cpu = _cpu(resource.RUSAGE_SELF) - c0
        stats = summary(data, dct, proj)
        w0n = time.time()
        null = build(data, window, null_seed=20261102, tail=tail)
        null_wall = time.time() - w0n
        json.dump(dict(clip=data["clip"], fps=data["fps"], frames=data["frames"], size=data["size"], **dct),
                  open(os.path.join(out, "dict_%s.json" % name), "w"), indent=1)
        w0r = time.time()
        render(clip, data, dct, proj, os.path.join(out, "projection_%s.mp4" % name),
               os.path.join(out, "peek_%s.png" % name))
        report["windows"][name] = dict(
            stats=stats, null=dict(ruled=null["ruled"], numbers=null["numbers"], wall=round(null_wall, 2)),
            am=dict(cpu=dct["am_cpu"], tokens=0), dict=dict(cpu=dct["dict_cpu"]),
            build_wall=round(build_wall, 2), projection=dict(cpu=round(proj_cpu, 4)),
            render_wall=round(time.time() - w0r, 2))
        print(json.dumps(report["windows"][name], indent=1), flush=True)
    json.dump(report, open(rpath, "w"), indent=1)


if __name__ == "__main__":
    main()
