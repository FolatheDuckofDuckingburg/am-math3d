"""AM SOLVES AN INTENT ON ITS PUZZLE: the question is which piece goes where, and how high or how fast; the answer is the
piece's exact place at every new frame, on the board AM built from the clip's last frame (am_puzzle.py).
  THE QUESTION: the mover (a piece); its target, the top of another piece or a spot on the stage (across, forward);
    a jump, with the arc's height OR its time in the air (gravity gives the other), or a slide along the surface, with
    its time; the wait before it moves; the length.
  THE ANSWER: in stage coordinates, s = (k - k0) / AIR.  A jump moves across and forward in a straight line with s, and
    up(s) = up0 + (up1 - up0) s + 4 Hap s (1 - s), Hap = g T^2 / 8 with g = 981 cm/s^2 and T = AIR / fps, so every
    airborne frame has the same second difference: gravity, exactly.  A slide moves with e = 3 s^2 - 2 s^3, starting
    and stopping at rest, and keeps its height.  Each stage point goes to the camera, p = O + across e1 + forward e2 +
    up n, then through the lens, x = cx + f X / Z, y = cy + f Y / Z, and the piece's real size gives its size on
    screen, w = f W / Z, h = f H / Z.
  WHAT THE SCENE ALLOWS: AM works out the shot only where the scene allows it: the piece stays inside the frame, passes
    over (a jump) or around (a slide) every other piece, and a slide cannot climb onto a higher piece.  Where the scene
    does not allow it, AM's answer is CANNOT, and why.
  THE GUIDE: the piece's own pixels from the last frame, moved and scaled to AM's numbers at every frame, over the frame
    with the piece lifted out: a picture of the math for a video model.  The hole is filled from the nearest pixels,
    or (fill="stage") from the surface beside it, carried along the stage so the perspective holds; pieces nearer the
    camera can be drawn back over the mover (occlude=True).
    solve(mapdir, mover, target, height=None, flight=None, wait=0.5, seconds=4.0, mode="jump", duration=None)
        -> mapdir/intent.json
    render(mapdir) -> mapdir/intent_guide.mp4, mapdir/intent_end.png"""
import json, math, os, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G = 981.0                                                      # cm/s^2


def _vec(v):
    return np.array(v, dtype=float)


def solve(mapdir, mover, target, height=None, flight=None, wait=0.5, seconds=4.0, mode="jump", duration=None, write=True):
    w0 = time.time()
    pz = json.load(open(os.path.join(mapdir, "puzzle.json")))
    fps = json.load(open(os.path.join(mapdir, "map.json")))["fps"]
    pieces = {p["id"]: p for p in pz["pieces"]}
    P = pieces[int(mover)]
    start = dict(P["stands"])
    slide = mode == "slide"
    if isinstance(target, dict):                               # a spot on the stage; a slide keeps its own height
        Q, land = None, dict(across=float(target["across"]), forward=float(target["forward"]),
                             up=start["up"] if slide else 0.0)
    else:
        Q = pieces[int(target)]
        land = dict(across=Q["stands"]["across"], forward=Q["stands"]["forward"], up=Q["top"])
    if slide:
        flight = float(duration if duration is not None else 1.0)
        AIR = max(2, int(round(flight * fps)))
        T, Hap = AIR / fps, 0.0
    else:
        if flight is None:
            flight = math.sqrt(8 * float(height if height is not None else 14.0) / G)
        AIR = max(2, int(round(float(flight) * fps)))
        T = AIR / fps
        Hap = G * T * T / 8                                    # the arc's height above the straight line, cm
    K = max(AIR + 2, int(round(seconds * fps)))
    k0 = min(K - AIR - 1, max(0, int(round(wait * fps))))
    L, st = pz["lens"], pz["stage"]
    f, cx, cy = L["f"], L["cx"], L["cy"]
    n, e1, e2, O = _vec(st["normal"]), _vec(st["axes"]["across"]), _vec(st["axes"]["forward"]), _vec(st["axes"]["origin"])
    W, Hgt = P["size"]["across"], P["size"]["up"]

    def lens(a, b, u):
        X, Y, Z = O + a * e1 + b * e2 + u * n
        return X, Y, Z, cx + f * X / Z, cy + f * Y / Z

    rows = []
    moving = "move" if slide else "air"
    for k in range(K):
        s = 0.0 if k <= k0 else (1.0 if k >= k0 + AIR else (k - k0) / AIR)
        e = 3 * s * s - 2 * s ** 3 if slide else s            # a slide starts and stops at rest
        a = start["across"] + (land["across"] - start["across"]) * e
        b = start["forward"] + (land["forward"] - start["forward"]) * e
        u = start["up"] + (land["up"] - start["up"]) * e + 4 * Hap * s * (1 - s)
        X, Y, Z, xf, yf = lens(a, b, u)                        # the foot
        _, _, Zt, xt, yt = lens(a, b, u + Hgt)                  # the top
        rows.append(dict(k=k, t=round(k / fps, 4), phase="wait" if k <= k0 else (moving if k < k0 + AIR else "landed"),
                         across=round(a, 2), forward=round(b, 2), up=round(u, 2), X=round(X, 2), Y=round(Y, 2),
                         Z=round(Z, 2), foot=[round(xf, 2), round(yf, 2)], h=round(yf - yt, 2), w=round(f * W / Z, 2)))
    # the checks
    Wd, Hd = pz["size"]
    h0 = rows[0]["h"]
    x0, y0, bw, bh = P["box"]
    out_of_frame = []
    for r in rows:
        sc = r["h"] / h0
        left = r["foot"][0] - (rows[0]["foot"][0] - x0) * sc
        top = r["foot"][1] - (rows[0]["foot"][1] - y0) * sc
        if left < 0 or top < 0 or left + bw * sc > Wd or top + bh * sc > Hd:
            out_of_frame.append(r["k"])
    hits = []
    for R in pz["pieces"]:
        if R["id"] in (P["id"], Q["id"] if Q else None) or P.get("rests_on") == R["id"]:
            continue
        for r in rows:
            if r["phase"] != moving:                           # standing where the frame shows it, or landed
                continue
            over = abs(r["across"] - R["stands"]["across"]) < (W + R["size"]["across"]) / 2 and \
                abs(r["forward"] - R["stands"]["forward"]) < (P["size"]["forward_seen"] + R["size"]["forward_seen"]) / 2
            if over and r["up"] < R["top"]:
                hits.append(dict(piece=R["id"], k=r["k"]))
                break
    climbs = bool(slide and Q and Q["top"] > start["up"] + 2.0)
    checks = dict(in_frame=not out_of_frame, frames_out=out_of_frame[:20], clears=not hits, hits=hits,
                  lands_on=("P%d" % Q["id"]) if Q else "the stage", climbs=climbs)
    why = []
    if out_of_frame:
        why.append("the piece leaves the frame from %.2f s" % (out_of_frame[0] / fps))
    for h in hits:
        why.append("it runs into P%d at %.2f s" % (h["piece"], h["k"] / fps))
    if climbs:
        why.append("a slide stays on its surface; P%d stands %.1f cm higher, so this move has to be a jump"
                   % (Q["id"], Q["top"] - start["up"]))
    gx = [r["up"] for r in rows if r["phase"] == "air"]
    d2 = [round(gx[i + 1] - 2 * gx[i] + gx[i - 1], 4) for i in range(1, len(gx) - 1)]
    if slide:
        equation = ("e = 3 s^2 - 2 s^3, s = (k - %d) / %d; across = %.1f + (%.1f - %.1f) e; forward = %.1f + (%.1f - %.1f) e;"
                    " up = %.1f" % (k0, AIR, start["across"], land["across"], start["across"], start["forward"],
                                     land["forward"], start["forward"], start["up"]))
    else:
        equation = ("up(s) = %.1f + (%.1f - %.1f) s + 4 x %.2f s (1 - s), s = (k - %d) / %d"
                    % (start["up"], land["up"], start["up"], Hap, k0, AIR))
    plan = dict(question=dict(mover=P["id"], target=(Q["id"] if Q else target), mode="slide" if slide else "jump",
                              height=height, flight=flight, duration=duration, wait=wait, seconds=seconds),
                answer=dict(fps=fps, frames=K, take_off=k0, air_frames=AIR, air_seconds=round(T, 4), arc_height=round(Hap, 2),
                            start=start, land=land, gravity_per_frame2=round(-G / fps / fps, 4),
                            second_difference=sorted(set(d2))[:3], equation=equation),
                verdict="SOLVED" if not why else "CANNOT", why=why, checks=checks, rows=rows,
                size=pz["size"], mover_box=P["box"], frame=pz["frame"], seconds_taken=round(time.time() - w0, 3), tokens=0)
    if write:
        json.dump(plan, open(os.path.join(mapdir, "intent.json"), "w"), indent=1)
    return plan


def plate_from(mapdir, mover, src_mp4, frame, out_png):
    """the last frame with the mover lifted out, its hole filled from a frame of the same view where that spot is empty
    (a take generated from the same start frame), colour-matched to the ring around the hole and feathered in; drawn at
    the source's size"""
    import av
    from PIL import Image, ImageFilter
    with av.open(src_mp4) as c:
        for k, x in enumerate(c.decode(video=0)):
            if k == frame:
                src_im = x.to_image().convert("RGB")
                break
    size = src_im.size
    start = np.asarray(Image.open(os.path.join(mapdir, "puzzle_frame.png")).convert("RGB").resize(size, Image.LANCZOS)).astype(float)
    lab = np.round(np.asarray(Image.open(os.path.join(mapdir, "puzzle_labels.png"))) / 20.0).astype(int)
    m8 = Image.fromarray(((lab == int(mover)) * 255).astype(np.uint8)).resize(size, Image.NEAREST)
    hole = np.asarray(m8.filter(ImageFilter.MaxFilter(15))) > 127
    alpha = np.asarray(Image.fromarray((hole * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))).astype(float) / 255
    ring = (np.asarray(m8.filter(ImageFilter.MaxFilter(61))) > 127) & ~hole
    src = np.asarray(src_im).astype(float).copy()
    for ch in range(3):
        a, b = src[..., ch][ring], start[..., ch][ring]
        src[..., ch] = (src[..., ch] - a.mean()) / (a.std() + 1e-6) * b.std() + b.mean()
    plate = start * (1 - alpha[..., None]) + np.clip(src, 0, 255) * alpha[..., None]
    Image.fromarray(plate.astype(np.uint8)).save(out_png)
    return out_png


def stage_point(pz, x, y):
    """the stage point under pixel (x, y): the lens ray meets the stage plane n.p + offset = 0; None above the horizon"""
    L, st = pz["lens"], pz["stage"]
    d = np.array([(x - L["cx"]) / L["f"], (y - L["cy"]) / L["f"], 1.0])
    n = _vec(st["normal"])
    nd = float(n @ d)
    if abs(nd) < 1e-9 or -st["offset"] / nd <= 0:
        return None
    q = d * (-st["offset"] / nd) - _vec(st["axes"]["origin"])
    return float(q @ _vec(st["axes"]["across"])), float(q @ _vec(st["axes"]["forward"]))


def _stage_fill(arr, hole, others, pz, P):
    """fill the hole from beside it: a pixel on the surface in front of or under the piece is copied from the surface
    one piece-width across on the stage (so the surface's perspective and texture carry on); a pixel behind the piece
    (wall, background) from one hole-width across on the same row; whatever is left from the nearest pixel, softened"""
    from PIL import Image, ImageFilter
    from scipy import ndimage
    Hh, Ww = hole.shape
    L, st = pz["lens"], pz["stage"]
    O, e1, e2, n = (_vec(st["axes"]["origin"]), _vec(st["axes"]["across"]), _vec(st["axes"]["forward"]),
                    _vec(st["normal"]))
    shift_cm = P["size"]["across"] + 2.0
    far = P["stands"]["forward"] + P["size"]["forward_seen"]
    ys, xs = np.nonzero(hole)
    bw = int(xs.max() - xs.min()) + 4
    blocked = hole | others
    out = arr.copy()
    rest = np.zeros(hole.shape, bool)
    for y, x in zip(ys, xs):
        cands = []
        sp = stage_point(pz, x, y)
        if sp is not None and sp[1] <= far:
            for k in (1, -1, 2, -2):
                p = O + (sp[0] + k * shift_cm) * e1 + sp[1] * e2
                if p[2] > 1e-6:
                    cands.append((L["cx"] + L["f"] * p[0] / p[2], L["cy"] + L["f"] * p[1] / p[2]))
        else:
            cands = [(x + k * bw, y) for k in (1, -1, 2, -2)]
        for qx, qy in cands:
            qx, qy = int(round(qx)), int(round(qy))
            if 0 <= qx < Ww and 0 <= qy < Hh and not blocked[qy, qx]:
                out[y, x] = arr[qy, qx]
                break
        else:
            rest[y, x] = True
    if rest.any():
        idx = ndimage.distance_transform_edt(rest, return_distances=False, return_indices=True)
        out[rest] = out[idx[0][rest], idx[1][rest]]
    soft = np.asarray(Image.fromarray(out.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))).astype(float)
    out[hole] = soft[hole]                                      # a light blur hides the seams of the copied patches
    return out


def _smooth_fill(arr, hole):
    """fill the hole smoothly from its border: each hole pixel is the average of its four neighbours (Laplace's
    equation, solved exactly), so no edge or copied shape is left for a video model to keep"""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    Hh, Ww = hole.shape
    ys, xs = np.nonzero(hole)
    idx = -np.ones(hole.shape, int)
    idx[ys, xs] = np.arange(len(ys))
    A = sparse.lil_matrix((len(ys), len(ys)))
    b = np.zeros((len(ys), arr.shape[2]))
    for i, (y, x) in enumerate(zip(ys, xs)):
        n = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < Hh and 0 <= xx < Ww:
                n += 1
                if hole[yy, xx]:
                    A[i, idx[yy, xx]] = -1
                else:
                    b[i] += arr[yy, xx]
        A[i, i] = n
    A = A.tocsr()
    out = arr.copy()
    for ch in range(arr.shape[2]):
        out[ys, xs, ch] = spsolve(A, b[:, ch])
    return np.clip(out, 0, 255)


def render(mapdir, plate=None, out=None, plan=None, end=None, height=None, fill="nearest", occlude=False):
    """the guide: the mover's own pixels at AM's place and size in every frame, over the frame with it lifted out.
    plate: an image of the same view with the mover already lifted out cleanly (any size; the guide is drawn at its
    size); without one, the hole is filled from the nearest pixels, which leaves a smear, or with fill="stage" from
    the surface beside it.  plan: a plan already worked out (by default mapdir/intent.json, which then records the
    render time); occlude: pieces nearer the camera than the mover are drawn back over it; height: the guide's height
    in pixels (the frame is scaled up to it, for video models that want 720p)"""
    from PIL import Image, ImageFilter
    from scipy import ndimage
    w0 = time.time()
    own = plan is None
    if own:
        plan = json.load(open(os.path.join(mapdir, "intent.json")))
    pz = json.load(open(os.path.join(mapdir, "puzzle.json")))
    frame = Image.open(os.path.join(mapdir, "puzzle_frame.png")).convert("RGB")
    ids = np.round(np.array(Image.open(os.path.join(mapdir, "puzzle_labels.png"))) / 20.0).astype(int)
    mover = plan["question"]["mover"]
    mask = ids == mover
    hole = ndimage.binary_dilation(mask, iterations=4)
    arr = np.array(frame).astype(float)
    if plate:
        plate = Image.open(plate).convert("RGB")
    elif fill == "stage":
        P = next(p for p in pz["pieces"] if p["id"] == mover)
        plate = Image.fromarray(_stage_fill(arr, hole, (ids > 0) & ~mask, pz, P).astype(np.uint8))
    elif fill == "smooth":
        plate = Image.fromarray(_smooth_fill(arr, hole).astype(np.uint8))
    else:
        idx = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
        filled = arr[idx[0], idx[1]]                           # each hole pixel takes the nearest pixel outside
        soft = np.array(Image.fromarray(filled.astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))).astype(float)
        plate = Image.fromarray(np.where(hole[..., None], soft, arr).astype(np.uint8))
    sx, sy = plate.size[0] / frame.size[0], plate.size[1] / frame.size[1]
    ys, xs = np.nonzero(mask)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    sprite = frame.crop((bx0, by0, bx1, by1))
    alpha = Image.fromarray((mask[by0:by1, bx0:bx1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    nearer = []
    if occlude:                                                # (forward, the piece's own pixels) for every other piece
        big = lambda m: np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(plate.size, Image.NEAREST)) > 127
        src = np.asarray(frame.resize(plate.size, Image.LANCZOS))
        nearer = [(R["stands"]["forward"], big(ids == R["id"])) for R in pz["pieces"] if R["id"] != mover]
    rows = plan["rows"]
    fx0, fy0, h0 = rows[0]["foot"][0], rows[0]["foot"][1], rows[0]["h"]
    out = out or os.path.join(mapdir, "intent_guide.mp4")
    size = plate.size
    if height and plate.size[1] < height:
        size = (int(round(plate.size[0] * height / plate.size[1] / 2)) * 2, int(height))
    with tempfile.TemporaryDirectory() as tmp:
        for r in rows:
            s = r["h"] / h0                                    # AM's perspective: the real size never changes
            nw, nh = max(1, round((bx1 - bx0) * s * sx)), max(1, round((by1 - by0) * s * sy))
            left = round((r["foot"][0] + (bx0 - fx0) * s) * sx)
            top = round((r["foot"][1] + (by0 - fy0) * s) * sy)
            im = plate.copy()
            im.paste(sprite.resize((nw, nh), Image.LANCZOS), (left, top), alpha.resize((nw, nh), Image.LANCZOS))
            if nearer:
                a = np.asarray(im).copy()
                for fwd, m in nearer:
                    if fwd < r["forward"]:
                        a[m] = src[m]
                im = Image.fromarray(a)
            if size != im.size:
                im = im.resize(size, Image.LANCZOS)
            im.save(os.path.join(tmp, "f_%04d.png" % r["k"]))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(plan["answer"]["fps"]), "-i",
                        os.path.join(tmp, "f_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
                       check=True)
        Image.open(os.path.join(tmp, "f_%04d.png" % rows[-1]["k"])).save(end or os.path.join(mapdir, "intent_end.png"))
    if own:
        plan["render_seconds"] = round(time.time() - w0, 2)
        json.dump(plan, open(os.path.join(mapdir, "intent.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    mapdir, mover, target = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    height = float(sys.argv[4]) if len(sys.argv) > 4 else 14.0
    p = solve(mapdir, mover, target, height=height)
    a = p["answer"]
    print("%s: P%d onto P%s, %d frames in the air (%.3f s), arc %.2f cm, take-off at frame %d of %d; checks %s"
          % (p["verdict"], mover, target, a["air_frames"], a["air_seconds"], a["arc_height"], a["take_off"], a["frames"],
             {k: v for k, v in p["checks"].items() if k in ("in_frame", "clears", "lands_on", "hits")}))
    print("  ", a["equation"], "| gravity per frame^2", a["gravity_per_frame2"], "| second differences", a["second_difference"])
    print("  guide:", render(mapdir), "in", p.get("render_seconds", json.load(open(os.path.join(mapdir, "intent.json")))["render_seconds"]), "s")
