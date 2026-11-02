"""AI MOVIES: the 3D world (world3d: plates moving by their own rules under a perspective camera) as a video generator
would draw it, and as depth and tracking estimators would measure it.
THE TRUTH: every object's 3D state at every frame, from its rules (world3d.state); its BOX on screen, centre x, y and
  width w, height h in pixels (world3d.project).
THE GENERATOR'S FAULTS (what AI video does wrong), applied to the 3D state it draws, 0-3 per video:
  DRIFT   an object slides off its path in X or Y: the offset grows to 10-40 cm over 4-10 frames, holds 0-6, snaps back
  WOBBLE  an object breathes in depth: Z off by up to 5-15%, rising and falling over 3-6 frames
  SWAP    two objects swap identities where their paths come closest in frames 30-42, to the end of the video
  GLITCH  for 1-2 frames an object jumps: 15-60 cm in X or Y, or 10-30% in Z
  MORPH   for 2-5 frames an object turns 20-60 degrees off its yaw
  POP     for 1-3 frames an object is not drawn at all
THE EYES (the estimators' 3D eyes): for every object drawn and not hidden (a nearer object covering 60% or more of its
  box), X and Y in cm with noise sigma 1 cm, Z with noise sigma 1% of Z, yaw with sigma 1 degree, rounded to whole
  numbers.  The noise level is the eyes' specification, known to every arm.
A frame is FAULTED for an object when it is measured and what the generator drew is more than TOL = 2 px away from the
truth in x, y, w or h.  An object's MAINTENANCE frames are its faulted, popped and hidden frames: where the video is
wrong or missing.
    video(seed) -> dict(objs, true, drawn, eyes, faults, faulted, popped, hidden, maint, kind_of); the arms read only
    eyes and sizes (true, drawn and the rest are for the judge and the design)"""
import math, os, random, sys

sys.path.insert(0, os.path.expanduser("~/neooos/world3d"))
import world3d as W                                             # noqa: E402

N = 48
TOL = 2.0
KEYS = ("X", "Y", "Z", "A")
KINDS = ("DRIFT", "WOBBLE", "SWAP", "GLITCH", "MORPH", "POP")


def sigma(key, z=None):
    """the eyes' noise, sigma: 1 cm, 1% of the depth, 1 degree"""
    return 0.01 * z if key == "Z" else 1.0


def box(s):
    x, y, ww, hh = W.project(s)
    return (x, y, ww, hh)


def off(a, b):
    return max(abs(u - v) for u, v in zip(box(a), box(b)))


def video(seed, n=N):
    sc = W.scene(seed, n)
    objs = [o["k"] for o in sc["objs"]]
    true = {t: {o["k"]: W.state(o, t) for o in sc["objs"]} for t in range(n)}
    drawn = {t: {k: dict(v) for k, v in true[t].items()} for t in range(n)}
    rng = random.Random(seed * 104729 + 7 * n)
    popped, faults, touched = set(), [], {}
    for _ in range(rng.choice([0, 1, 1, 2, 2, 3])):
        kind, k = rng.choice(KINDS), rng.choice(objs)
        if kind == "DRIFT":
            key, d = rng.choice("XY"), rng.choice([-1, 1]) * rng.randint(10, 40)
            L, hold, t0 = rng.randint(4, 10), rng.randint(0, 6), rng.randint(1, n - 3)
            for i, t in enumerate(range(t0, min(n, t0 + L + hold))):
                drawn[t][k][key] += int(round(d * min(i + 1, L) / L))
                touched[(k, t)] = kind
            what = "object %d slides %+d cm in %s over %d frames from frame %d, holds %d, snaps back" % (k, d, key, L, t0, hold)
        elif kind == "WOBBLE":
            amp, L, t0 = rng.choice([-1, 1]) * rng.uniform(0.05, 0.15), rng.randint(3, 6), rng.randint(1, n - 3)
            for i, t in enumerate(range(t0, min(n, t0 + L))):
                drawn[t][k]["Z"] = int(round(drawn[t][k]["Z"] * (1 + amp * math.sin(math.pi * (i + 1) / (L + 1)))))
                touched[(k, t)] = kind
            what = "object %d breathes %+.0f%% in depth over %d frames from frame %d" % (k, 100 * amp, L, t0)
        elif kind == "SWAP":
            if len(objs) < 2:
                continue
            j = rng.choice([o for o in objs if o != k])
            tc = min(range(30, n - 5), key=lambda t: (box(true[t][k])[0] - box(true[t][j])[0]) ** 2 +
                                                      (box(true[t][k])[1] - box(true[t][j])[1]) ** 2)
            for t in range(tc, n):
                drawn[t][k], drawn[t][j] = drawn[t][j], drawn[t][k]
                touched[(k, t)] = touched[(j, t)] = kind
            what = "objects %d and %d swap identities from frame %d" % (k, j, tc)
        elif kind == "GLITCH":
            key, L, t0 = rng.choice("XYZ"), rng.randint(1, 2), rng.randint(1, n - 3)
            d = rng.choice([-1, 1]) * (rng.uniform(0.1, 0.3) if key == "Z" else rng.randint(15, 60))
            for t in range(t0, min(n, t0 + L)):
                drawn[t][k][key] = int(round(drawn[t][k][key] * (1 + d))) if key == "Z" else drawn[t][k][key] + d
                touched[(k, t)] = kind
            what = "object %d jumps %s in %s for %d frame(s) from frame %d" % (
                k, ("%+.0f%%" % (100 * d)) if key == "Z" else ("%+d cm" % d), key, L, t0)
        elif kind == "MORPH":
            L, t0, d = rng.randint(2, 5), rng.randint(1, n - 3), rng.choice([-1, 1]) * rng.randint(20, 60)
            for t in range(t0, min(n, t0 + L)):
                drawn[t][k]["A"] += d
                touched[(k, t)] = kind
            what = "object %d turns %+d degrees off for %d frames from frame %d" % (k, d, L, t0)
        else:
            L, t0 = rng.randint(1, 3), rng.randint(1, n - 3)
            popped |= {(k, t) for t in range(t0, min(n, t0 + L))}
            touched.update({(k, t): kind for t in range(t0, min(n, t0 + L))})
            what = "object %d not drawn for %d frame(s) from frame %d" % (k, L, t0)
        faults.append(kind + ": " + what)
    eyes, hidden, faulted = {}, set(), set()
    for t in range(n):
        boxes = {}
        for k, s in drawn[t].items():
            x, y, ww, hh = box(s)
            boxes[k] = (x - max(ww, 2) / 2, y - hh / 2, x + max(ww, 2) / 2, y + hh / 2, s["Z"])
        for k, s in drawn[t].items():
            if (k, t) in popped:
                continue
            x0, y0, x1, y1, z = boxes[k]
            area = (x1 - x0) * (y1 - y0)
            cover = any(j != k and (j, t) not in popped and zj < z and
                        max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0)) >= 0.6 * area
                        for j, (a0, b0, a1, b1, zj) in boxes.items())
            if cover:
                hidden.add((k, t))
                continue
            nr = random.Random(seed * 1000003 + t * 101 + k)
            eyes[(k, t)] = {key: int(round(s[key] + nr.gauss(0, sigma(key, s["Z"])))) for key in KEYS}
            if off(s, true[t][k]) > TOL:
                faulted.add((k, t))
    return dict(seed=seed, objs=objs, sizes={o["k"]: (o["w"], o["h"]) for o in sc["objs"]}, true=true, eyes=eyes,
                drawn=drawn, faults=faults, faulted=faulted, popped=popped, hidden=hidden, maint=faulted | popped | hidden,
                kind_of={kt: touched.get(kt, "?") for kt in faulted | popped})


def judge(v, flags, anchors):
    """flags: set of (k, t); anchors: {(k, t): dict X, Y, Z, A}.  JOB 1 per measured object-frame; JOB 2 per
    maintenance object-frame: the arm's state there is its anchor where it flagged the frame or nothing was measured,
    else the measurement it let stand; RIGHT when its box is within TOL of the truth, SILENT when it gave none."""
    j1 = dict(caught=0, missed=0, fa=0)
    for kt in v["eyes"]:
        if kt in v["faulted"]:
            j1["caught" if kt in flags else "missed"] += 1
        elif kt in flags:
            j1["fa"] += 1
    j2 = dict(right=0, wrong=0, silent=0)
    by = {}
    for kt in v["maint"]:
        k, t = kt
        if kt in v["eyes"] and kt not in flags:
            mine = v["eyes"][kt]
        else:
            mine = anchors.get(kt)
        if mine is None or mine.get("Z", 0) <= 0:
            res = "silent"
        else:
            w, h = v["sizes"][k]
            res = "right" if off(dict(mine, w=w, h=h), v["true"][t][k]) <= TOL else "wrong"
        j2[res] += 1
        by[kt] = res
    return j1, j2, by
