"""AM'S GEOMETRIC PUZZLE: one frame of a clip (by default its last) as exact 3D geometry, the board an intent is solved on.
  THE CAMERA: the stated lens (60 degrees wide), at the origin; the frame's edges bound what can be seen at each depth.
  THE STAGE: the support surface things stand on (a counter, a floor), fitted as a plane to the frame's metric-depth
    points (RANSAC, then least squares on its inliers); the camera's height above it and its tilt come with it.
    Stage coordinates: origin under the camera, `across` to the right, `forward` away along the stage, `up` off it.
  THE PIECES: the subject Apple Vision lifts out of the frame (foreground instances), and everything else that stands
    up off the stage over the stage's own footprint, cut apart where depth jumps.  Parts that float and touch a piece
    belong to it; what reaches the top of the frame or rises past 60 cm is scenery (walls, cupboards), not a piece.
    Each piece: where it stands (stage coordinates of its foot), its size (across x forward-as-seen x up, cm), what it
    rests on (the stage, another piece, or nothing: floating), its box on screen, and its pixels (for moving it later).
  AM'S INVARIANTS: each piece is matched to the object AM's eyes tracked through the clip (box overlap at this frame),
    and carries AM's verdict on that object's numbers over the whole clip: a rule with its proof, or CANNOT.
    build(mapdir, frame=None) -> mapdir/puzzle.json, puzzle_frame.png, puzzle_labels.png (grey = 20 x piece)"""
import json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am3d                                                    # noqa: E402

TOL = 1.5                                                      # cm: on the stage
CELL = 2.0                                                     # cm: the stage's footprint grid
SCENERY_H = 60.0                                               # cm: taller than this is the room, not a piece
MIN_AREA = 0.0015                                              # of the frame
REST = 3.0                                                     # cm: a foot this close to a surface rests on it


def _frame(clip, t):
    import av
    with av.open(clip) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i == t:
                return fr.to_image().convert("RGB")
    raise ValueError("frame %d not in the clip" % t)


def _instances(png, labels_png):
    out = subprocess.run([os.path.join(HERE, "instances"), png, labels_png], capture_output=True, text=True, check=True)
    return [json.loads(l) for l in out.stdout.splitlines() if l.strip()]


def stage(P, H):
    """the support plane: normal n (unit, pointing to the camera's side) and offset d, with n . p + d = height"""
    rng = np.random.default_rng(0)
    cand = P[int(H * 0.4):, :][::3, ::3].reshape(-1, 3)
    best = (0, None)
    for _ in range(1500):
        a, b, c = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        if n[1] > 0:
            n = -n                                             # camera y points down; the stage's up is -y
        if -n[1] < 0.5:
            continue                                           # a wall, not something to stand on
        d = -float(n @ a)
        k = int((np.abs(cand @ n + d) < TOL).sum())
        if k > best[0]:
            best = (k, (n, d))
    if best[1] is None:
        return None
    n, d = best[1]
    inl = cand[np.abs(cand @ n + d) < TOL]
    c0 = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c0, full_matrices=False)    # least squares on the inliers
    n2 = vt[-1] if vt[-1][1] < 0 else -vt[-1]
    d2 = -float(n2 @ c0)
    res = float(np.sqrt(np.mean((inl @ n2 + d2) ** 2)))
    return dict(n=n2, d=d2, inliers=float(len(inl) / len(cand)), rms=res)


def build(mapdir, frame=None, progress=print):
    w0 = time.time()
    m = json.load(open(os.path.join(mapdir, "map.json")))
    clip = os.path.join(mapdir, "clip.mp4")
    t = m["frames"] - 1 if frame is None else int(frame)
    img = _frame(clip, t)
    W, H = img.size
    L = m["lens"]
    f, cx, cy = L["f"], L["cx"], L["cy"]
    png = os.path.join(mapdir, "puzzle_frame.png")
    img.save(png)
    progress("eyes")
    from PIL import Image
    from scipy import ndimage
    inst = _instances(png, os.path.join(mapdir, "puzzle_subjects.png"))
    subj_lab = np.array(Image.open(os.path.join(mapdir, "puzzle_subjects.png")))
    dm = am3d.depth_model(m["depth_model"])(img)["predicted_depth"].squeeze().detach().cpu().numpy()
    Z = np.array(Image.fromarray(dm.astype(np.float32)).resize((W, H), Image.BILINEAR)) * 100.0
    v, u = np.mgrid[0:H, 0:W]
    P = np.stack([(u - cx) * Z / f, (v - cy) * Z / f, Z], -1)
    progress("stage")
    st = stage(P, H)
    if st is None:
        raise RuntimeError("no surface to stand on was found in this frame")
    n, d = st["n"], st["d"]
    h = P @ n + d                                              # every pixel's height above the stage, cm
    e1 = np.array([1.0, 0, 0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)                                       # forward, away from the camera along the stage
    O = -d * n                                                 # the point on the stage under the camera
    G = P - h[..., None] * n
    ga, gb = (G - O) @ e1, (G - O) @ e2                        # each pixel's foot, in stage coordinates
    lo_a, lo_b = np.percentile(ga, 1) - 20, np.percentile(gb, 1) - 20
    ia = np.clip(((ga - lo_a) / CELL).astype(int), 0, 1499)
    ib = np.clip(((gb - lo_b) / CELL).astype(int), 0, 1499)
    occ = np.zeros((1500, 1500), bool)
    flat = np.abs(h) < TOL
    occ[ia[flat], ib[flat]] = True
    occ = ndimage.binary_dilation(ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=4)), iterations=3)
    above = (h > TOL) & occ[ia, ib]
    jump = np.hypot(*np.gradient(Z)) / np.maximum(Z, 1) > 0.04
    progress("pieces")
    masks = []
    near = (h > TOL)                                           # Vision's subjects first, grown into everything joined
    for s in inst:                                             # to them at their own depth (a hat Vision's mask leaves)
        mk = subj_lab == s["instance"]
        if mk.sum():
            zs, hs = float(np.median(Z[mk])), float(np.percentile(h[mk], 20))
            same, _ = ndimage.label(near & (np.abs(Z - zs) < 0.2 * zs) & (h > hs))   # above its own lower body
            touch = set(np.unique(same[ndimage.binary_dilation(mk, iterations=2)])) - {0}
            if touch:
                mk = ndimage.binary_fill_holes(mk | np.isin(same, list(touch)))
            masks.append(["subject", mk])
            subj_lab[mk & (subj_lab == 0)] = s["instance"]
    subj = subj_lab > 0
    lab, k = ndimage.label(ndimage.binary_opening(above & ~jump & ~subj, iterations=1))
    for i in range(1, k + 1):
        mk = lab == i
        if mk.sum() >= MIN_AREA * W * H:
            masks.append(["thing", ndimage.binary_closing(mk, iterations=2)])

    def geom(mk):
        hs = h[mk]
        foot, top = float(np.percentile(hs, 5)), float(np.percentile(hs, 95))
        low = mk & (h <= foot + 3)
        a, b = ga[low if low.sum() >= 20 else mk], gb[low if low.sum() >= 20 else mk]
        ys, xs = np.nonzero(mk)
        return dict(foot=foot, top=top, a=float(np.median(a)), b=float(np.median(b)),
                    a_ext=(float(np.percentile(ga[mk], 5)), float(np.percentile(ga[mk], 95))),
                    b_ext=(float(np.percentile(gb[mk], 15)), float(np.percentile(gb[mk], 85))),   # edges bleed depth
                    Z=float(np.median(Z[mk])), box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])

    # scenery out; floating parts that touch a piece belong to it
    kept, scenery = [], []
    for kind, mk in masks:
        g = geom(mk)
        if kind == "thing" and (g["box"][1] <= 0.02 * H or g["top"] > SCENERY_H):
            scenery.append(mk)
        else:
            kept.append([kind, mk, g])
    changed = True
    while changed:
        changed = False
        for i, (kind, mk, g) in enumerate(kept):
            if kind != "thing" or g["foot"] <= 10:
                continue
            grown = ndimage.binary_dilation(mk, iterations=12)        # a part within 12 px of a piece, or inside its
            b = g["box"]                                               # box, at its depth, is part of it
            for j, (k2, m2, g2) in enumerate(kept):
                b2 = g2["box"]
                boxes = b[0] <= b2[2] + 10 and b2[0] <= b[2] + 10 and b[1] <= b2[3] + 10 and b2[1] <= b[3] + 10
                if j != i and ((grown & m2).any() or boxes) and abs(g2["Z"] - g["Z"]) < 0.15 * g["Z"]:
                    kept[j][1] = m2 | mk
                    kept[j][2] = geom(kept[j][1])
                    kept.pop(i)
                    changed = True
                    break
            if changed:
                break
    progress("relations")
    # what each piece rests on
    pieces = []
    for i, (kind, mk, g) in enumerate(kept):
        pieces.append(dict(id=i + 1, kind=kind, mask=mk, **g))
    for p in pieces:
        if p["foot"] <= REST:
            p["rests_on"] = "stage"
            continue
        under = None
        for q in pieces:
            if q is p:
                continue
            inside = q["a_ext"][0] - 2 <= p["a"] <= q["a_ext"][1] + 2 and q["b_ext"][0] - 2 <= p["b"] <= q["b_ext"][1] + 2
            if inside and q["foot"] - REST <= p["foot"] <= q["top"] + REST:
                if under is None or q["top"] > under["top"]:
                    under = q
        p["rests_on"] = under["id"] if under else "nothing"
    # AM's invariants: the tracked object under each piece at this frame, and AM's verdict on its numbers
    tracked = []
    for o in m["objects"]:
        r = next((r for r in o["rows"] if r[0] == t), None)
        if r:
            tracked.append((o, [r[1] - r[3] / 2, r[2] - r[4] / 2, r[1] + r[3] / 2, r[2] + r[4] / 2]))

    def iou(a, b):
        x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    labels = np.zeros((H, W), np.uint8)
    out = []
    for p in pieces:
        best = max(((iou(p["box"], bx), o) for o, bx in tracked), key=lambda x: x[0], default=(0, None))
        inv = None
        if best[0] >= 0.3:
            o = best[1]
            inv = dict(object=o["id"], overlap=round(best[0], 2), frames=len(o["rows"]),
                       numbers={k: (dict(rule=v["rule"], formula=v.get("formula", ""), kept=v.get("kept"),
                                         measured=v["measured"], band=v["band"]) if v["rule"] != "CANNOT"
                                    else dict(rule="CANNOT", band=v["band"], measured=v["measured"]))
                                for k, v in o["numbers"].items()})
        labels[p["mask"]] = 20 * p["id"]                            # grey level 20 x the piece's number
        x0, y0, x1, y1 = p["box"]
        out.append(dict(id=p["id"], kind=p["kind"] if p["rests_on"] != "nothing" else "floating",
                        stands=dict(across=round(p["a"], 1), forward=round(p["b"], 1), up=round(p["foot"], 1)),
                        size=dict(across=round(p["a_ext"][1] - p["a_ext"][0], 1), forward_seen=round(p["b_ext"][1] - p["b_ext"][0], 1),
                                  up=round(p["top"] - max(0.0, p["foot"]), 1)),
                        top=round(p["top"], 1), rests_on=p["rests_on"], distance_cm=round(p["Z"], 1),
                        box=[x0, y0, x1 - x0 + 1, y1 - y0 + 1], area=int(p["mask"].sum()), am=inv))
    rel = []
    for i, p in enumerate(out):
        for q in out[i + 1:]:
            da = q["stands"]["across"] - p["stands"]["across"]
            db = q["stands"]["forward"] - p["stands"]["forward"]
            du = q["stands"]["up"] - p["stands"]["up"]
            rel.append(dict(a=p["id"], b=q["id"], across=round(da, 1), forward=round(db, 1), up=round(du, 1),
                            apart=round(float(np.sqrt(da * da + db * db + du * du)), 1)))
    Image.fromarray(labels).save(os.path.join(mapdir, "puzzle_labels.png"))
    tilt = float(np.degrees(np.arcsin(min(1.0, abs(n[2])))))
    puzzle = dict(clip=m["clip"], frame=t, time=round(t / m["fps"], 3), size=[W, H], lens=L, depth_model=m["depth_model"],
                  camera=dict(height=round(d, 1), tilt_down=round(tilt, 1)),
                  stage=dict(normal=[round(float(x), 4) for x in n], offset=round(d, 2), inliers=round(st["inliers"], 3),
                             rms_cm=round(st["rms"], 2), axes=dict(across=[round(float(x), 4) for x in e1],
                                                                   forward=[round(float(x), 4) for x in e2],
                                                                   origin=[round(float(x), 2) for x in O])),
                  pieces=out, relations=rel, scenery=len(scenery), seconds=round(time.time() - w0, 1),
                  cost_usd=round((time.time() - w0) / 3600 * 0.03, 6), tokens=0)
    tmp = os.path.join(mapdir, "puzzle.json.tmp")
    json.dump(puzzle, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(mapdir, "puzzle.json"))
    progress("done")
    return puzzle


def say(pz):
    c = pz["camera"]
    print("frame %d (%.2f s): camera %.1f cm above the stage, tilted %.1f deg down; stage fit %.0f%% of the lower frame, rms %.2f cm"
          % (pz["frame"], pz["time"], c["height"], c["tilt_down"], 100 * pz["stage"]["inliers"], pz["stage"]["rms_cm"]))
    for p in pz["pieces"]:
        am = p["am"]
        ruled = [k for k, v in am["numbers"].items() if v["rule"] != "CANNOT"] if am else []
        print("  P%d %-8s stands across %6.1f forward %6.1f up %5.1f | size %5.1f x %5.1f x %5.1f cm | rests on %-7s | AM: %s"
              % (p["id"], p["kind"], p["stands"]["across"], p["stands"]["forward"], p["stands"]["up"], p["size"]["across"],
                 p["size"]["forward_seen"], p["size"]["up"], p["rests_on"],
                 ("object #%d, rules for %s" % (am["object"], ", ".join(ruled) or "nothing")) if am else "not tracked"))
    print("  scenery parts left out: %d; %.1f s, %d tokens" % (pz["scenery"], pz["seconds"], pz["tokens"]))


if __name__ == "__main__":
    say(build(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None))
