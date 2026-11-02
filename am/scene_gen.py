"""SCENE DNA, the world (v1).  A scene is flat rectangles on a background, under a panning camera and one exposure
that lights every pixel.  Every number in it comes from a short integer rule over the frame number t: that set of
rules is the scene's DNA.  The renderer is exact integer code, so the same DNA always gives the same pixels.
    make_scene(seed) -> scene (dict of rules);  render(scene, t) -> beauty, id pass, u pass, v pass (H x W int arrays)
Passes, as a VFX renderer writes them: id = which object covers the pixel (0 = background); u, v = the pixel's
position inside that object (-1 on background).  Draw order: objects by id (3, 4, ...), then the marker (1), then the
grey card (2) on top.  Pixel = clamp(object value + exposure, 0, 255); the generator keeps every value inside 1..254."""
import random
import numpy as np

W, H = 32, 24
SEEN = list(range(12))
HELD = list(range(12, 18))
T_ALL = SEEN + HELD
MARKER_V, CARD_V = 200, 128                          # the marker's and the grey card's own values (known to every arm)
CARD_XY = (0, H - 2)                                 # the grey card: 2 x 2, fixed on screen, bottom left


def ev(r, t):
    f, a, b, p = r["f"], r.get("a", 0), r.get("b", 0), r.get("p", 1)
    if f == "const":
        return b
    if f == "lin":
        return a * t + b
    if f == "ease":
        return a * ((t * t) // p) + b
    if f == "step":
        return (t // p) * a + b
    if f == "cycle":
        return (t % p) * a + b
    if f == "pingpong":
        return abs(t % (2 * p) - p) * a + b
    if f == "clamp":
        return (min(a * t, p) if a > 0 else max(a * t, -p)) + b
    raise ValueError(f)


def expr(r):
    f, a, b, p = r["f"], r.get("a", 0), r.get("b", 0), r.get("p", 1)
    return {"const": f"{b}", "lin": f"{a}*t + {b}", "ease": f"{a}*((t*t)//{p}) + {b}", "step": f"(t//{p})*{a} + {b}",
            "cycle": f"(t%{p})*{a} + {b}", "pingpong": f"abs(t%{2 * p} - {p})*{a} + {b}",
            "clamp": (f"min({a}*t, {p}) + {b}" if a > 0 else f"max({a}*t, -{p}) + {b}")}[f]


def draw(rng, kind):
    """one rule from the common animation curves: hold, linear, ease-in, stepped, cycle, ping-pong, clamped"""
    if kind == "light" and rng.random() < 1 / 3 or kind == "cam" and rng.random() < 1 / 3:
        return dict(f="const", b=0)
    f = rng.choice(["const", "lin", "ease", "step", "cycle", "pingpong", "clamp"])
    s = rng.choice([-1, 1])
    if kind in ("pos", "cam"):
        A = [1, 2] if kind == "cam" else [1, 2, 3]
        a = s * rng.choice(A)
        b = 0 if kind == "cam" else rng.randint(-4, W)
        p = {"ease": rng.choice([6, 8, 12, 16]), "step": rng.choice([2, 3, 4]), "cycle": rng.choice([3, 4, 5, 6]),
             "pingpong": rng.choice([2, 3, 4, 5]), "clamp": rng.randint(4, 12)}.get(f, 1)
    elif kind == "val":
        a = s * rng.randint(1, 8)
        b = rng.randint(30, 225)
        p = {"ease": rng.choice([2, 3, 4, 6, 8]), "step": rng.choice([2, 3, 4]), "cycle": rng.choice([3, 4, 5, 6]),
             "pingpong": rng.choice([2, 3, 4, 5]), "clamp": rng.randint(8, 60)}.get(f, 1)
    else:                                                  # light: one exposure for every pixel
        a = s * rng.randint(1, 4)
        b = rng.randint(-20, 20)
        p = {"ease": rng.choice([4, 6, 8]), "step": rng.choice([2, 3, 4]), "cycle": rng.choice([3, 4, 5, 6]),
             "pingpong": rng.choice([2, 3, 4, 5]), "clamp": rng.randint(8, 40)}.get(f, 1)
    if f == "const":
        return dict(f=f, b=b if kind != "val" else rng.randint(30, 225))
    return dict(f=f, a=a, b=b, p=p)


def _rect_on_screen(sx, sy, w, h):
    return sx >= 0 and sy >= 0 and sx + w <= W and sy + h <= H


def make_scene(seed):
    rng = random.Random(seed)
    for _ in range(100000):
        K = rng.randint(3, 6)
        cam = [draw(rng, "cam"), draw(rng, "cam")]
        light = draw(rng, "light")
        Ls = [ev(light, t) for t in T_ALL]
        if not (-60 <= min(Ls) and max(Ls) <= 54):
            continue
        bg = rng.randint(20, 235)
        if not all(1 <= bg + L <= 254 for L in Ls):
            continue
        cxs = [ev(cam[0], t) for t in T_ALL]
        cys = [ev(cam[1], t) for t in T_ALL]
        # the marker: world-static, 2 x 2, on screen and clear of the grey card in every frame
        ok_m = []
        for mx in range(-10, W + 10):
            for my in range(-10, H + 10):
                if all(_rect_on_screen(mx - cx, my - cy, 2, 2) and not (mx - cx < 2 and my - cy + 2 > H - 2)
                       for cx, cy in zip(cxs, cys)):
                    ok_m.append((mx, my))
        if not ok_m:
            continue
        mx, my = rng.choice(ok_m)
        objs = []
        for k in range(K):
            o = dict(id=3 + k, w=rng.randint(3, 8), h=rng.randint(3, 8), x=draw(rng, "pos"), y=draw(rng, "pos"),
                     v=draw(rng, "val"))
            o["x"]["b"] = o["x"].get("b", 0) + (cxs[0] if o["x"]["f"] != "const" else 0)
            if not all(1 <= ev(o["v"], t) + L <= 254 for t, L in zip(T_ALL, Ls)):
                break
            xs = [ev(o["x"], t) for t in T_ALL]
            ys = [ev(o["y"], t) for t in T_ALL]
            if max(abs(v) for v in xs + ys) > 60:
                break
            objs.append(o)
        if len(objs) < K:
            continue
        scene = dict(seed=seed, cam=cam, light=light, bg=bg, marker=(mx, my), objs=objs)
        # every object fully visible (on screen, nothing drawn over it) in at least one seen frame
        good = True
        for o in objs:
            vis = False
            for t in SEEN:
                _, idp, _, _ = render(scene, t)
                if (idp == o["id"]).sum() == o["w"] * o["h"]:
                    vis = True
                    break
            if not vis:
                good = False
                break
        if good:
            return scene
    raise RuntimeError("no scene for seed %d" % seed)


def render(scene, t):
    cx, cy = ev(scene["cam"][0], t), ev(scene["cam"][1], t)
    L = ev(scene["light"], t)
    beauty = np.full((H, W), scene["bg"] + L, np.int64)
    idp = np.zeros((H, W), np.int64)
    up = np.full((H, W), -1, np.int64)
    vp = np.full((H, W), -1, np.int64)
    mx, my = scene["marker"]
    layers = [(o["id"], ev(o["x"], t) - cx, ev(o["y"], t) - cy, o["w"], o["h"], ev(o["v"], t)) for o in scene["objs"]]
    layers += [(1, mx - cx, my - cy, 2, 2, MARKER_V), (2, CARD_XY[0], CARD_XY[1], 2, 2, CARD_V)]
    for i, sx, sy, w, h, v in layers:
        x0, x1, y0, y1 = max(sx, 0), min(sx + w, W), max(sy, 0), min(sy + h, H)
        if x0 >= x1 or y0 >= y1:
            continue
        beauty[y0:y1, x0:x1] = min(max(v + L, 0), 255)
        idp[y0:y1, x0:x1] = i
        up[y0:y1, x0:x1] = np.arange(x0, x1)[None, :] - sx
        vp[y0:y1, x0:x1] = np.arange(y0, y1)[:, None] - sy
    return beauty, idp, up, vp


def dna(scene):
    """the scene's DNA as text: every rule"""
    lines = ["camera x: " + expr(scene["cam"][0]), "camera y: " + expr(scene["cam"][1]),
             "exposure: " + expr(scene["light"]), "background: %d" % scene["bg"], "marker (world): %d, %d" % scene["marker"]]
    for o in scene["objs"]:
        lines.append("object %d (%dx%d): x %s | y %s | value %s" % (o["id"], o["w"], o["h"], expr(o["x"]), expr(o["y"]),
                                                                   expr(o["v"])))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    s = make_scene(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    print(dna(s))
    b, i, u, v = render(s, 0)
    print(b)
    print(i)
    print("%.2fs" % (time.time() - t0))
