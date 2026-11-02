"""AM WRITES THE AD: one product photo, an intent, and AM's equation for every frame of a 15-second camera move, sealed
before anything is generated; the take is then measured against it.
  THE SCENE IN 3D: AM's eyes and puzzle find the product (a piece) and the surface it stands on; metric depth through
    AM's stated lens (60 degrees wide) puts every pixel in AM's centimetres, P = ((u - cx) Z / f, (v - cy) Z / f, Z).
    On a product photo taken with a longer lens those centimetres are scaled wrong; the shot below uses only angles,
    fractions of distance and fractions of the frame, so that scale cancels.
  THE INTENT: the camera arcs around the product to reveal its side, eases in and out, and pushes in until the
    product fills a chosen share of the frame, centred.
  THE EQUATION, for frame k at time t = k / fps, s = clamp((t - t0) / T, 0, 1), e = 3 s^2 - 2 s^3:
      theta(t) = theta_end e                 the arc, about the stage's up axis n through the product's centre B
      d(t) = d0 + (d_end - d0) e             the camera's distance from B; d_end solved so the product's projected
                                             height is the chosen share of the frame
      C(t) = B + R_n(theta) (C0 - B) d / d0  the camera's centre (C0 = 0, where the photo was taken)
      L(t) = (1 - e) L0 + e B                where it looks (L0 straight ahead, at B's depth)
      x = cx + f q_x / q_z, y = cy + f q_y / q_z, q = R(t) (P - C(t))   every point of the scene, every frame
  THE GUIDE: each frame is the photo re-projected through C(t), R(t): every target pixel finds the photo pixel that
    lands on it (a fixed-point search through the depth map); where the photo never saw a region, it stretches, and
    the video model is told to paint clean detail there.
  THE RULE, fixed in the plan: the product tracked by the same tracker on the guide and on the take stays within
    RULE_PX px at every frame of the plan and at the end, and the background (textured patches, tracked the same way)
    follows AM's camera: median within RULE_PX px on at least 90% of the frames where three or more are tracked.
    plan(mapdir, out) -> out/: plan.json, plan.sha256, guide.mp4, prompt.txt, process.json
    score(out, take) -> out/: score.json, labelled.mp4"""
import hashlib, json, os, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RULE_PX = 8.0
FPS, SECONDS, T0, T1 = 30, 15.0, 1.5, 13.5                    # hold, move with ease in and out, hold
THETA_END, SHARE = -8.0, 0.85                                 # degrees of arc (negative: to the camera's left, toward the
                                                              # spine); the product's share of the frame height


def _unit(v):
    return v / np.linalg.norm(v)


def _rot(axis, deg):
    a, th = _unit(np.asarray(axis, float)), np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def camera(B, n, d_end, e, theta_end=THETA_END):
    """the camera at ease e: its centre C and its rotation R (rows: x right, y down, z forward in the photo's frame)"""
    d0 = np.linalg.norm(B)
    C = B + _rot(n, theta_end * e) @ (-B) * ((d0 + (d_end - d0) * e) / d0)
    L = (1 - e) * np.array([0.0, 0.0, d0]) + e * B
    z = _unit(L - C)
    x = _unit(np.cross([0.0, 1.0, 0.0], z))
    return C, np.stack([x, np.cross(z, x), z])


def project(Pts, C, R, L):
    q = (Pts - C) @ R.T
    return np.stack([L["cx"] + L["f"] * q[:, 0] / q[:, 2], L["cy"] + L["f"] * q[:, 1] / q[:, 2]], -1)


def plan(mapdir, out, progress=print):
    from PIL import Image
    import am3d
    os.makedirs(out, exist_ok=True)
    log, w0 = [], time.time()

    def step(name, t0, **info):
        log.append(dict(step=name, at=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 3), **info))

    m = json.load(open(os.path.join(mapdir, "map.json")))
    pz = json.load(open(os.path.join(mapdir, "puzzle.json")))
    step("AM's eyes and map of the photo (built by am_map.py)", time.time(), seconds_measured=m["total_wall"], tokens=0,
         objects=len(m["objects"]))
    step("AM's puzzle: the surface and the pieces (built by am_puzzle.py)", time.time(), seconds_measured=pz["seconds"],
         camera_above_stage_cm=pz["camera"]["height"], tilt_down_deg=pz["camera"]["tilt_down"], stage_rms_cm=pz["stage"]["rms_cm"],
         pieces=[dict(id=p["id"], kind=p["kind"], box=p["box"]) for p in pz["pieces"]])
    # the scene in 3D
    t0 = time.time()
    img = Image.open(os.path.join(mapdir, "puzzle_frame.png")).convert("RGB")
    W, H = img.size
    L = pz["lens"]
    dm = am3d.depth_model(m["depth_model"])(img)["predicted_depth"].squeeze().detach().cpu().numpy()
    Z = np.array(Image.fromarray(dm.astype(np.float32)).resize((W, H), Image.BILINEAR)) * 100.0
    np.save(os.path.join(out, "depth_cm.npy"), Z.astype(np.float32))
    v, u = np.mgrid[0:H, 0:W]
    P = np.stack([(u - L["cx"]) * Z / L["f"], (v - L["cy"]) * Z / L["f"], Z], -1)
    ids = np.round(np.array(Image.open(os.path.join(mapdir, "puzzle_labels.png"))) / 20.0).astype(int)
    prod = max(pz["pieces"], key=lambda p: p["box"][2] * p["box"][3])          # the product: the largest piece
    mask = ids == prod["id"]
    B = np.median(P[mask], axis=0)
    n = np.asarray(pz["stage"]["normal"], float)
    rng = np.random.default_rng(0)
    pts = P[mask][rng.choice(int(mask.sum()), size=min(4000, int(mask.sum())), replace=False)]
    step("The scene in AM's centimetres: metric depth through the stated lens", t0, product_piece=prod["id"],
         product_centre_cm=[round(float(x), 1) for x in B], product_distance_cm=round(float(np.linalg.norm(B)), 1),
         depth_cm=dict(min=round(float(Z.min()), 1), median=round(float(np.median(Z)), 1), max=round(float(Z.max()), 1)),
         note="AM's stated lens is 60 degrees; on this photo its centimetres are scaled wrong, and the shot uses only angles and fractions")
    # the end distance, solved: the product's projected height is SHARE of the frame at e = 1
    t0 = time.time()
    height_at = lambda d: float(np.ptp(project(pts, *camera(B, n, d, 1.0), L)[:, 1]))
    d0 = float(np.linalg.norm(B))
    lo, hi = 0.2 * d0, 3.0 * d0
    for _ in range(60):                                        # height falls as the distance grows
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if height_at(mid) > SHARE * H else (lo, mid)
    d_end = (lo + hi) / 2
    K = int(round(SECONDS * FPS))
    rows = []
    for k in range(K):
        t = k / FPS
        s = min(1.0, max(0.0, (t - T0) / (T1 - T0)))
        e = 3 * s * s - 2 * s ** 3
        C, R = camera(B, n, d_end, e)
        pr = project(pts, C, R, L)
        bc = project(B[None], C, R, L)[0]
        rows.append(dict(k=k, t=round(t, 4), e=round(e, 6), theta=round(THETA_END * e, 4),
                         d=round(float(np.linalg.norm(B - C)), 3), C=[round(float(x), 3) for x in C],
                         R=[[round(float(x), 6) for x in r] for r in R], product_centre=[round(float(x), 2) for x in bc],
                         product_box=[round(float(np.percentile(pr[:, 0], 1)), 1), round(float(np.percentile(pr[:, 1], 1)), 1),
                                      round(float(np.percentile(pr[:, 0], 99)), 1), round(float(np.percentile(pr[:, 1], 99)), 1)]))
    step("The equation solved: arc %.0f degrees to the left, end distance %.3f of the start so the product fills %.0f%% "
         "of the height" % (abs(THETA_END), d_end / d0, 100 * SHARE), t0, d0=round(d0, 2), d_end=round(d_end, 2), frames=K, fps=FPS)
    # the patches the background is held to: textured, away from the product, lifted to 3D at frame 0
    pick = am3d.pick_markers(np.asarray(img), [((prod["box"][0] + prod["box"][2] / 2), (prod["box"][1] + prod["box"][3] / 2),
                                                prod["box"][2], prod["box"][3])], W, H)
    patches = [dict(id=i + 1, px=[int(x), int(y)], P=[round(float(c), 3) for c in P[int(y), int(x)]]) for i, (x, y) in enumerate(pick)]
    for r in rows:
        C, R = np.array(r["C"]), np.array(r["R"])
        r["patches"] = [[round(float(c), 1) for c in p] for p in project(np.array([q["P"] for q in patches]), C, R, L)]
    prompt = ("Edit this video into a premium 15-second advertisement for this Vazram notebook. Keep the camera move exactly "
              "as in this video: a slow, smooth arc to the left around the notebook with a gentle push in, easing in and "
              "easing out, the notebook centred at the end. Keep the notebook exactly where it is in every frame and exactly "
              "the size shown, with its cover art, the words LIFE IS THE BEST JOURNEY, the Vazram logo and the green spine "
              "unchanged and sharp. Keep the pen, its cap, the desk, the plant and the books in place. Make it look like a "
              "real high-end product film: crisp detail, soft warm natural light, gentle depth of field, the plant's leaves "
              "moving very slightly. Where any edge looks smeared or stretched, paint clean natural detail. Do not add text, "
              "hands, people or new objects, and do not change the notebook's cover.")
    open(os.path.join(out, "prompt.txt"), "w").write(prompt + "\n")
    t0 = time.time()
    still_hash = hashlib.sha256(open(os.path.join(mapdir, "puzzle_frame.png"), "rb").read()).hexdigest()
    depth_hash = hashlib.sha256(open(os.path.join(out, "depth_cm.npy"), "rb").read()).hexdigest()
    plan_doc = dict(intent="a 15-second hero ad: the camera arcs %.0f degrees to the left around the notebook, revealing "
                           "its spine, eases in and out, and pushes in until the notebook fills %.0f%% of the frame height, "
                           "centred" % (abs(THETA_END), 100 * SHARE),
                    inputs=dict(photo_sha256=still_hash, depth_sha256=depth_hash, size=[W, H], lens=L,
                                stage_normal=[round(float(x), 6) for x in n], product_piece=prod["id"], product_box=prod["box"]),
                    equation=dict(ease="s = clamp((t - %.1f) / %.1f, 0, 1), e = 3 s^2 - 2 s^3" % (T0, T1 - T0),
                                  arc="theta = %.1f e degrees about n through B (negative: to the left)" % THETA_END,
                                  distance="d = %.3f + (%.3f - %.3f) e" % (d0, d_end, d0),
                                  camera="C = B + R_n(theta) (0 - B) d / d0; looks at (1 - e) L0 + e B",
                                  B=[round(float(x), 3) for x in B], d0=round(d0, 3), d_end=round(d_end, 3)),
                    fps=FPS, frames=K, rows=rows, patches=patches,
                    rule=dict(px=RULE_PX, text="the product, tracked with the same tracker on the guide and the take at %dx%d, "
                              "within %g px of the guide at every frame of the plan and at the end; the background patches' "
                              "median within %g px of the guide on at least 90%% of the frames where three or more are tracked"
                              % (W, H, RULE_PX, RULE_PX)),
                    prompt=prompt, model="Kling 4.0 Edit, one pass")
    blob = json.dumps(plan_doc, indent=1).encode()
    open(os.path.join(out, "plan.json"), "wb").write(blob)
    seal = hashlib.sha256(blob).hexdigest()
    open(os.path.join(out, "plan.sha256"), "w").write(seal + "  plan.json\n")
    step("The lock: plan.json written, SHA-256 recorded before anything is generated", t0, sha256=seal)
    # the guide, drawn from the sealed rows: the photo re-projected through AM's camera at every frame
    t0 = time.time()
    from scipy import ndimage
    arr = np.asarray(img).astype(np.float32)
    gpath = os.path.join(out, "guide.mp4")
    unseen = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in rows:
            C, R = np.array(r["C"]), np.array(r["R"])
            su, sv = u.astype(np.float32).copy(), v.astype(np.float32).copy()
            for _ in range(16):                                # find the photo pixel that lands on each target pixel
                Zs = ndimage.map_coordinates(Z, [sv, su], order=1, mode="nearest")
                X = np.stack([(su - L["cx"]) * Zs / L["f"], (sv - L["cy"]) * Zs / L["f"], Zs], -1) - C
                q = X @ R.T
                pu, pv = L["cx"] + L["f"] * q[..., 0] / q[..., 2], L["cy"] + L["f"] * q[..., 1] / q[..., 2]
                su, sv = su + (u - pu), sv + (v - pv)
            res = np.hypot(u - pu, v - pv)
            out_of = (su < 0) | (su > W - 1) | (sv < 0) | (sv > H - 1)
            unseen.append(round(float(((res > 1.0) | out_of).mean()), 4))
            frame = np.stack([ndimage.map_coordinates(arr[..., c], [sv, su], order=1, mode="nearest") for c in range(3)], -1)
            Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).save(os.path.join(tmp, "f_%04d.png" % r["k"]))
            if r["k"] % 50 == 0:
                progress("guide %d/%d" % (r["k"], K))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(tmp, "f_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", gpath], check=True)
    step("The guide drawn: the photo re-projected through AM's camera, %d frames" % K, t0,
         unseen_share=dict(max=max(unseen), median=float(np.median(unseen))))
    json.dump(dict(steps=log, total_seconds_this_run=round(time.time() - w0, 2), tokens=0),
              open(os.path.join(out, "process.json"), "w"), indent=1)
    return plan_doc, seal, log


def score(out, take, progress=print):
    """the take against the sealed plan: the product and the background, with the same tracker on the guide and the take"""
    import am_shot
    w0 = time.time()
    blob = open(os.path.join(out, "plan.json"), "rb").read()
    seal = open(os.path.join(out, "plan.sha256")).read().split()[0]
    lock_ok = hashlib.sha256(blob).hexdigest() == seal
    p = json.loads(blob)
    W, H = p["inputs"]["size"]
    rows, end_t = p["rows"], p["rows"][-1]["t"]
    with tempfile.TemporaryDirectory() as tmp:
        gn, tn = os.path.join(tmp, "g.mp4"), os.path.join(tmp, "t.mp4")
        gfps, gcount, _ = am_shot._norm(os.path.join(out, "guide.mp4"), gn, W, H)
        tfps, tcount, _ = am_shot._norm(take, tn, W, H)
        x0, y0, bw, bh = p["inputs"]["product_box"]
        boxes = [[100, x0 + bw / 2, y0 + bh / 2, bw, bh]] + [[q["id"], q["px"][0], q["px"][1], 48, 48] for q in p["patches"]]
        G, T = am_shot._track(gn, boxes), am_shot._track(tn, boxes)
    frames = []
    for k in range(tcount):
        t = k / tfps
        j = min(gcount - 1, int(round(t * gfps)))
        g, tk = G.get(100, {}).get(j), T.get(100, {}).get(k)
        sh = [float(np.hypot(*np.subtract(T[i][k][:2], G[i][j][:2]))) for i in range(1, len(p["patches"]) + 1)
              if k in T.get(i, {}) and j in G.get(i, {})]
        frames.append(dict(k=k, t=round(t, 3), in_plan=t <= end_t + 0.5 / tfps,
                           off=None if (g is None or tk is None) else round(float(np.hypot(tk[0] - g[0], tk[1] - g[1])), 2),
                           guide=None if g is None else [round(x, 1) for x in g[:4]], take=None if tk is None else [round(x, 1) for x in tk[:4]],
                           bg=round(float(np.median(sh)), 2) if len(sh) >= 3 else None, bg_n=len(sh)))
    inp = [f for f in frames if f["in_plan"]]
    got = [f["off"] for f in inp if f["off"] is not None]
    bgs = [f["bg"] for f in inp if f["bg"] is not None]
    end = next((f["off"] for f in reversed(inp) if f["off"] is not None), None)
    prod_ok = bool(got and len(got) >= 0.9 * len(inp) and max(got) <= RULE_PX and end is not None and end <= RULE_PX)
    bg_ok = bool(bgs and len(bgs) >= 0.5 * len(inp) and np.mean(np.array(bgs) <= RULE_PX) >= 0.9)
    verdict = "LOCK BROKEN" if not lock_ok else ("PASS" if prod_ok and bg_ok else "FAIL")
    res = dict(lock=seal, lock_ok=lock_ok, rule=p["rule"]["text"], verdict=verdict,
               take=dict(fps=round(tfps, 3), frames=tcount, seconds=round(tcount / tfps, 3)), plan_seconds=end_t,
               product=dict(measured=len(got), in_plan=len(inp), median_px=round(float(np.median(got)), 2) if got else None,
                            worst_px=max(got) if got else None, end_px=end, within_rule=prod_ok),
               background=dict(measured=len(bgs), median_px=round(float(np.median(bgs)), 2) if bgs else None,
                               share_within=round(float(np.mean(np.array(bgs) <= RULE_PX)), 3) if bgs else None, within_rule=bg_ok),
               seconds=round(time.time() - w0, 2), tokens=0, frames=frames)
    json.dump(res, open(os.path.join(out, "score.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        doc, seal, log = plan(sys.argv[2], sys.argv[3])
        for s in log:
            print("%-110s %7.2f s" % (s["step"][:110], s["seconds"]))
        print("lock", seal)
    else:
        r = score(sys.argv[2], sys.argv[3])
        print(json.dumps({k: v for k, v in r.items() if k != "frames"}, indent=1))
