"""WORLD 3D: true 3D motion under a perspective camera, with rotation and objects passing behind one another.
A scene: K = 3-5 flat square plates, each with its own rules over the frame number t (the common animation curves of
scene_gen: hold, linear, ease-in, stepped, cycle, ping-pong, clamped) for its world position X, Y, Z (cm) and its yaw
angle (degrees; a plate turns about its vertical axis).  The camera is a pinhole at the origin looking down +Z, focal
length 400 px, image 640 x 480, centre (320, 240).
The renderer gives, for every frame and every object that is visible:
  2D EYES (what any tracker sees on the screen): screen x = round(320 + 400 X / Z), screen y = round(240 + 400 Y / Z),
           apparent width = round(400 w |cos yaw| / Z)
  3D EYES (a renderer's position, depth and orientation passes): X, Y, Z exactly, and the yaw angle, unwrapped
An object is HIDDEN in a frame when a nearer object covers 60% or more of its screen box: then neither eye sees it.
FORGERIES edit an object's 3D state in a frame before it is drawn, so both eyes see the edit as the scene would show it:
  MOVE_Z   pushed 20-60 cm deeper or nearer            MOVE_X   moved 8-30 cm sideways
  TURN     turned 8-25 degrees off                     SWAP     given its state from another frame
  BLEND    every number the average of the frames either side (an in-between frame)
    scene(seed, n), frames(scene, n, edits) -> dict t -> {k: dict(eyes2d, eyes3d) or None};  fakes(seed, n)"""
import math, random, sys, os
sys.path.insert(0, os.path.expanduser("~/neooos/scene_dna"))
import scene_gen as G                                        # noqa: E402  (the curve families and their evaluator)

F, CX, CY, W, H = 400, 320, 240, 640, 480


def draw(rng, what):
    f = rng.choice(["const", "lin", "ease", "step", "cycle", "pingpong", "clamp"])
    s = rng.choice([-1, 1])
    if what == "Z":
        a, b = s * rng.randint(2, 12), rng.randint(450, 1100)
        p = {"ease": rng.choice([4, 6, 8]), "step": rng.choice([3, 4, 6]), "cycle": rng.choice([5, 6, 8, 10]),
             "pingpong": rng.choice([5, 8, 10, 12]), "clamp": rng.randint(80, 300)}.get(f, 1)
    elif what == "A":
        a, b = s * rng.choice([2, 3, 4, 5, 6]), rng.randint(0, 80)
        p = {"ease": rng.choice([8, 12, 16]), "step": rng.choice([3, 4, 6]), "cycle": rng.choice([6, 8, 10]),
             "pingpong": rng.choice([6, 8, 10]), "clamp": rng.randint(30, 120)}.get(f, 1)
    else:
        a, b = s * rng.randint(1, 6), rng.randint(-90, 90) if what == "X" else rng.randint(-50, 50)
        p = {"ease": rng.choice([8, 12, 16]), "step": rng.choice([3, 4, 6]), "cycle": rng.choice([5, 6, 8, 10]),
             "pingpong": rng.choice([5, 8, 10, 12]), "clamp": rng.randint(20, 120)}.get(f, 1)
    if f == "const":
        return dict(f=f, b=b)
    return dict(f=f, a=a, b=b, p=p)


def state(o, t):
    return dict(X=G.ev(o["X"], t), Y=G.ev(o["Y"], t), Z=G.ev(o["Z"], t), A=G.ev(o["A"], t), w=o["w"], h=o["h"])


def project(s):
    x = CX + F * s["X"] / s["Z"]
    y = CY + F * s["Y"] / s["Z"]
    ww = F * s["w"] * abs(math.cos(math.radians(s["A"]))) / s["Z"]
    hh = F * s["h"] / s["Z"]
    return x, y, ww, hh


def scene(seed, n, min_hidden=0):
    rng = random.Random(seed * 50021 + n)
    for _ in range(100000):
        objs = [dict(k=k, X=draw(rng, "X"), Y=draw(rng, "Y"), Z=draw(rng, "Z"), A=draw(rng, "A"),
                     w=rng.randint(60, 160), h=rng.randint(60, 160)) for k in range(1, rng.randint(3, 5) + 1)]
        ok = True
        for o in objs:
            for t in range(n):
                s = state(o, t)
                if not (300 <= s["Z"] <= 2000):
                    ok = False
                    break
                x, y, ww, hh = project(s)
                if not (40 <= x <= W - 40 and 40 <= y <= H - 40):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            sc = dict(seed=seed, objs=objs)
            if min_hidden and sum(v is None for f in frames(sc, n).values() for v in f.values()) < min_hidden:
                continue
            return sc
    raise RuntimeError("no scene")


def frames(sc, n, edits=None):
    edits = edits or {}
    out = {}
    for t in range(n):
        st = {o["k"]: state(o, t) for o in sc["objs"]}
        if t in edits:
            st = edits[t](st)
        boxes = {}
        for k, s in st.items():
            x, y, ww, hh = project(s)
            boxes[k] = (x - max(ww, 2) / 2, y - hh / 2, x + max(ww, 2) / 2, y + hh / 2, s["Z"])
        out[t] = {}
        for k, s in st.items():
            x0, y0, x1, y1, z = boxes[k]
            area = (x1 - x0) * (y1 - y0)
            hidden = False
            for j, (a0, b0, a1, b1, zj) in boxes.items():
                if j != k and zj < z:
                    ix, iy = min(x1, a1) - max(x0, a0), min(y1, b1) - max(y0, b0)
                    if ix > 0 and iy > 0 and ix * iy >= 0.6 * area:
                        hidden = True
            if hidden:
                out[t][k] = None
                continue
            x, y, ww, hh = project(s)
            out[t][k] = dict(eyes2d=dict(x=int(round(x)), y=int(round(y)), w=int(round(ww))),
                             eyes3d=dict(X=s["X"], Y=s["Y"], Z=s["Z"], A=s["A"]))
    return out


KINDS = ["MOVE_Z", "MOVE_X", "TURN", "SWAP", "BLEND"]


def fakes(seed, n):
    sc = scene(seed, n)
    rng = random.Random(seed * 7907 + n)
    k_ed = rng.choice([0, 1, 2, 2, 3, 4])
    edits, what = {}, {}
    for t in rng.sample(range(1, n - 1), k_ed):
        kind = rng.choice(KINDS)
        obj = rng.choice([o["k"] for o in sc["objs"]])
        if kind == "MOVE_Z":
            d = rng.choice([-1, 1]) * rng.randint(20, 60)
            fn = (lambda d, obj: lambda st: {**st, obj: {**st[obj], "Z": st[obj]["Z"] + d}})(d, obj)
            desc = "object %d pushed %+d cm in depth" % (obj, d)
        elif kind == "MOVE_X":
            d = rng.choice([-1, 1]) * rng.randint(8, 30)
            fn = (lambda d, obj: lambda st: {**st, obj: {**st[obj], "X": st[obj]["X"] + d}})(d, obj)
            desc = "object %d moved %+d cm sideways" % (obj, d)
        elif kind == "TURN":
            d = rng.choice([-1, 1]) * rng.randint(8, 25)
            fn = (lambda d, obj: lambda st: {**st, obj: {**st[obj], "A": st[obj]["A"] + d}})(d, obj)
            desc = "object %d turned %+d degrees off" % (obj, d)
        elif kind == "SWAP":
            t2 = rng.choice([u for u in range(n) if abs(u - t) >= 3])
            o = [o for o in sc["objs"] if o["k"] == obj][0]
            fn = (lambda o, t2, obj: lambda st: {**st, obj: state(o, t2)})(o, t2, obj)
            desc = "object %d given its frame-%d state" % (obj, t2)
        else:
            fn = (lambda sc, t: lambda st: {o["k"]: {key: (state(o, t - 1)[key] + state(o, t + 1)[key]) // 2
                                                      if key in ("X", "Y", "Z", "A") else state(o, t)[key]
                                                      for key in state(o, t)} for o in sc["objs"]})(sc, t)
            desc = "in-between of frames %d and %d" % (t - 1, t + 1)
        # an edit nobody could see is not an edit
        before, after = frames(sc, n), None
        after = frames(sc, n, {t: fn})
        if after[t] != before[t]:
            edits[t], what[t] = fn, kind + ": " + desc
    return sc, frames(sc, n, edits), what


if __name__ == "__main__":
    import time
    t0 = time.time()
    for s in range(1, 4):
        sc, fr, ed = fakes(s, 48)
        hidden = sum(1 for t in fr for k, v in fr[t].items() if v is None)
        print(s, "objects", len(sc["objs"]), "| hidden object-frames", hidden, "| edits", ed, "| %.1fs" % (time.time() - t0))
        o = sc["objs"][0]
        print("    object 1 rules:", {a: G.expr(o[a]) for a in ("X", "Y", "Z", "A")})
        print("    frame 0 eyes:", fr[0][1])
