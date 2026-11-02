"""INTENT 3D: an action said in words, set exactly in the clip's 3D puzzle, projected to exact 2D data for a model.
  THE START: the subject's box at the clip's last frame (AM-Video's eyes) lifted with metric depth through the
    puzzle's lens: its camera-space X, Y, Z and its width and height in cm.  The landing spot is lifted the same way.
  THE ACTION: stand still, then jump at 1.0 s and land, then stand still; camera held, everything else still.  Over the
    extension frame k (24 fps) with s = (k - k0) / AIR in [0, 1]:
      X = X0 + (X1 - X0) s,  Z = Z0 + (Z1 - Z0) s,  Y = Y0 + (Y1 - Y0) s - 4 Hap s (1 - s)
    a ballistic arc: the second difference of Y is the same in every airborne frame, and Hap = g T^2 / 8 with
    g = 981 cm/s^2 makes it Earth's gravity exactly.  Its width and height in cm never change, so its box shrinks
    or grows only by perspective.
  THE DATA: every frame's 3D state, projected through the lens to the box the model must draw.
  THE ARMS: A, a normal optimised prompt; B, the same words plus AM's exact data (and an end frame when the end
    state is exactly the start frame).
  INTENT ACCURACY: the subject is followed through each extension from its start box (track_from).
    python3 intent3d.py plan OUTDIR VARIANT [AIR]    (VARIANT: pan | plate | plate_feasible; AIR: frames in the air)
    python3 intent3d.py score OUTDIR VARIANT ARM EXTENSION.mp4"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
G = 981.0
FPS, SECS = 24, 4
K0, AIR = 24, 8                                               # takeoff at 1.0 s; 8 frames (1/3 s) in the air
GEN_W, GEN_H = 1280, 720
CLIP = os.path.join(HERE, "..", "samples", "sora_mouse_chef.mp4")
SCENE = "Mouse in chef hat cooking cat dinner in fancy restaurant"
MOUSE = (425, 298, 64, 104)                                   # the mouse's body at frame 149 (AM-Video's eyes, #8)
PLATE = (738, 298, 124, 28)                                   # the plate with the tomato at frame 149


def measure(out):
    """the mouse and the plate at the clip's last frame, in 3D (metric depth, the puzzle's lens)"""
    path = os.path.join(out, "jump", "scene3d.json")
    if os.path.exists(path):
        return json.load(open(path))
    import av
    import am3d
    pz = json.load(open(os.path.join(out, "puzzle3d.json")))
    L = pz["lens"]
    with av.open(CLIP) as c:
        fr = [f.to_image() for f in c.decode(video=0)][pz["frames"] - 1]
    dm = am3d.depth_model(pz["depth_model"])(fr)["predicted_depth"].squeeze().cpu().numpy()
    Wd, Hd = pz["size"]
    sy, sx = dm.shape[0] / Hd, dm.shape[1] / Wd
    res = dict(lens=L, size=pz["size"])
    for name, (x, y, w, h) in (("mouse", MOUSE), ("plate", PLATE)):
        a = 0.3
        Z = 100 * float(np.median(dm[int((y - a * h) * sy):int((y + a * h) * sy) + 1,
                                     int((x - a * w) * sx):int((x + a * w) * sx) + 1]))
        res[name] = dict(box=[x, y, w, h], Z=Z, X=(x - L["cx"]) * Z / L["f"], Y=(y - L["cy"]) * Z / L["f"],
                         W=w * Z / L["f"], H=h * Z / L["f"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"), indent=1)
    return res


def plan(out, variant, air=None):
    from PIL import Image
    global AIR
    AIR = air or AIR
    sc = measure(out)
    L, (W, H) = sc["lens"], sc["size"]
    f, cx, cy = L["f"], L["cx"], L["cy"]
    m = sc["mouse"]
    X0, Y0, Z0 = m["X"], m["Y"], m["Z"]
    if variant.startswith("pan"):
        X1, Y1, Z1 = X0, Y0, Z0
    else:
        p = sc["plate"]
        top = p["Y"] - p["H"] / 2                              # the plate's top, as the eyes see it
        X1, Y1, Z1 = p["X"], top - m["H"] / 2, p["Z"]         # the mouse stands on it
    T = AIR / FPS
    hap = G * T * T / 8                                        # apex above the chord, cm: gravity exactly
    k1 = K0 + AIR
    rows = []
    for k in range(SECS * FPS + 1):
        s = min(1.0, max(0.0, (k - K0) / AIR))
        X, Z = X0 + (X1 - X0) * s, Z0 + (Z1 - Z0) * s
        Y = Y0 + (Y1 - Y0) * s - 4 * hap * s * (1 - s)
        rows.append(dict(k=k, t=k / FPS, X=X, Y=Y, Z=Z, x=cx + f * X / Z, y=cy + f * Y / Z, w=f * m["W"] / Z,
                         h=f * m["H"] / Z))
    d2 = [rows[k + 1]["Y"] - 2 * rows[k]["Y"] + rows[k - 1]["Y"] for k in range(K0 + 1, k1)]
    g_check = -float(np.mean(d2)) * FPS * FPS
    pct = lambda r: "centre %.1f%% across, %.1f%% down, %.1f%% wide, %.1f%% tall" % (
        100 * r["x"] / W, 100 * r["y"] / H, 100 * r["w"] / W, 100 * r["h"] / H)
    apex = rows[K0 + AIR // 2]
    if variant.startswith("pan"):
        words = ("Continuing from this frame, with the camera completely still: the mouse chef stands in the frying "
                 "pan for one second, then jumps straight up about 14 centimetres, reaching the top of the jump at "
                 "about 1.17 seconds and landing back in the pan on the same spot at about 1.33 seconds, then stands "
                 "still until the end. The mouse keeps the same size and look; nothing else in the scene moves.")
        where = "lands exactly where it started"
    else:
        words = ("Continuing from this frame, with the camera completely still: the mouse chef stands in the frying "
                 "pan for one second, then leaps out of the pan onto the white plate with the tomato on the right of "
                 "the counter, about 14 centimetres high at the top of the leap, landing on the plate at about 1.33 "
                 "seconds, then stands still on the plate until the end. The mouse keeps the same look; nothing else "
                 "in the scene moves.")
        where = "lands on the plate, %.0f cm right and %.0f cm further from the camera" % (X1 - X0, Z1 - Z0)
    data = ("Exact motion, solved from the video in 3D (lens: 60-degree view; the mouse is %.0f cm from the camera, "
            "%.0f cm wide and %.0f cm tall): 0 to 1.00 s: still, %s. 1.00 s: takes off. %.2f s: top of the jump, "
            "%.1f cm above the straight line from takeoff to landing, %s. %.2f s: %s, %s. Then still to 4.00 s. "
            "Path: X, Z move at constant speed; height follows a parabola with gravity 9.81 m/s^2. Its real size never "
            "changes (its picture shrinks only with distance). Camera fixed. Every other object holds still."
            % (Z0, m["W"], m["H"], pct(rows[0]), apex["t"], hap, pct(apex), rows[k1]["t"], where, pct(rows[k1])))
    prompt_a = "%s. %s" % (SCENE, words)
    prompt_b = "%s %s" % (prompt_a, data)
    import av
    with av.open(CLIP) as c:
        frames = [fr.to_image() for fr in c.decode(video=0)]
    d = os.path.join(out, "jump", variant)
    os.makedirs(d, exist_ok=True)
    start = frames[-1].resize((GEN_W, GEN_H), Image.LANCZOS)
    start.save(os.path.join(d, "start_frame.png"))
    if variant.startswith("pan"):
        start.save(os.path.join(d, "end_frame_am.png"))                # the jump ends exactly where it began
    sx, sy = GEN_W / W, GEN_H / H
    res = dict(variant=variant, apex_cm=hap, takeoff_k=K0, land_k=k1, fps=FPS, secs=SECS, gravity_check=g_check,
               lens=L, size=[W, H], gen=[GEN_W, GEN_H], start=dict(X=X0, Y=Y0, Z=Z0), land=dict(X=X1, Y=Y1, Z=Z1),
               start_box=dict(x=MOUSE[0] * sx, y=MOUSE[1] * sy, w=MOUSE[2] * sx, h=MOUSE[3] * sy),
               rows=rows, prompt_a=prompt_a, prompt_b=prompt_b)
    json.dump(res, open(os.path.join(d, "plan.json"), "w"), indent=1)
    print("%s: apex %.2f cm, gravity from the path %.1f cm/s^2, landing at k=%d\n\nARM A (%d chars):\n%s\n\n"
          "ARM B (%d chars):\n%s" % (variant, hap, g_check, k1, len(prompt_a), prompt_a, len(prompt_b), prompt_b))


def score(out, variant, arm, mp4):
    import av
    d = os.path.join(out, "jump", variant)
    p = json.load(open(os.path.join(d, "plan.json")))
    W, H = p["size"]
    with av.open(mp4) as c:
        st = c.streams.video[0]
        gw, gh, gfps = st.width, st.height, float(st.average_rate or 24)
    sx, sy = gw / W, gh / H
    b, k = p["start_box"], gw / p["gen"][0]
    tr = subprocess.run([os.path.join(HERE, "track_from"), mp4,
                         json.dumps([[1, b["x"] * k, b["y"] * k, b["w"] * k, b["h"] * k]])],
                        capture_output=True, text=True, check=True)
    rows = [json.loads(l) for l in tr.stdout.splitlines() if l.strip()]
    pairs = []
    for r in p["rows"]:
        t = int(round(r["t"] * gfps))
        tk = rows[t]["tracks"] if t < len(rows) else []
        g = None if not tk else dict(x=tk[0][1] / sx, y=tk[0][2] / sy, w=tk[0][3] / sx, h=tk[0][4] / sy)
        pairs.append((r, g))
    seen = [(r, g) for r, g in pairs if g is not None]
    err = [np.hypot(g["x"] - r["x"], g["y"] - r["y"]) for r, g in seen]
    base = [g["y"] for r, g in seen if r["k"] < p["takeoff_k"]]
    air = [(r, g) for r, g in seen if p["takeoff_k"] <= r["k"] <= p["land_k"]]
    after = [(r, g) for r, g in seen if r["k"] >= p["land_k"] + 4]
    res = dict(variant=variant, arm=arm, video=os.path.basename(mp4), fps=gfps, size=[gw, gh],
               tracked="%d of %d frames" % (len(seen), len(pairs)),
               mean_centre_error_px=round(float(np.mean(err)), 1) if err else None,
               max_centre_error_px=round(float(np.max(err)), 1) if err else None)
    if base and air:
        y0 = float(np.median(base))
        top = min(air, key=lambda rg: rg[1]["y"])
        want_top = min(p["rows"][p["takeoff_k"]:p["land_k"] + 1], key=lambda r: r["y"])
        res.update(rise_px=round(y0 - top[1]["y"], 1), want_rise_px=round(p["rows"][0]["y"] - want_top["y"], 1),
                   top_s=round(top[0]["t"], 3), want_top_s=round(want_top["t"], 3))
    if after:
        lx = float(np.median([g["x"] for r, g in after]))
        ly = float(np.median([g["y"] for r, g in after]))
        lw = float(np.median([g["w"] for r, g in after]))
        want = p["rows"][-1]
        res.update(landing_error_px=round(float(np.hypot(lx - want["x"], ly - want["y"])), 1),
                   landing_width_ratio=round(lw / want["w"], 2))
    want = p["rows"][-1]                                       # the end, seen afresh: every object Vision finds there
    ve = subprocess.run([os.path.join(HERE, "..", "am", "video_eyes"), mp4], capture_output=True,
                        text=True, check=True)
    fin = [json.loads(l) for l in ve.stdout.splitlines() if l.strip()]
    found = [dict(x=r[1] / sx, y=r[2] / sy, w=r[3] / sx, h=r[4] / sy) for r in (fin[-1]["tracks"] if fin else [])]
    if found:
        near = min(found, key=lambda g: np.hypot(g["x"] - want["x"], g["y"] - want["y"]))
        res.update(end_nearest_object_px=round(float(np.hypot(near["x"] - want["x"], near["y"] - want["y"])), 1),
                   end_objects=[[round(g["x"]), round(g["y"]), round(g["w"]), round(g["h"])] for g in found])
    res["per_quarter_second"] = [dict(t=round(r["t"], 2), want=[round(r["x"]), round(r["y"])],
                                      got=None if g is None else [round(g["x"]), round(g["y"])])
                                 for r, g in pairs if r["k"] % 6 == 0]
    json.dump(res, open(os.path.join(d, "score_%s.json" % arm), "w"), indent=1)
    print(json.dumps({q: v for q, v in res.items() if q != "per_quarter_second"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        plan(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
