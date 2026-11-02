"""AM'S GRID SURFACE: a flat surface with a two-colour square pattern (a chessboard, tiles), fitted exactly.
  THE COLOURS: the two coloured classes of the surface are learned from the frame: 6-means on the saturated pixels of
    the lower part of the frame, and the pair of classes that border each other most (greys and near-blacks are left
    out: pieces, shadows).
  THE GRID: the outer frame of the lighter class gives a first quadrilateral; then for every grid size N = 3..10 the four
    corners are refined so the predicted pattern (light where i + j is odd, or even) agrees with the frame at 25 sample
    points per square; the size and parity with the best agreement win.  The homography H maps board squares (u, v)
    to pixels; with the stated lens K it gives the surface in 3D: H ~ K [r1 r2 T], r3 = r1 x r2 (u across, v away
    from the camera's side, w up off the surface; one unit = one square).
    fit(frame_png) -> dict(N, H, K, r1, r2, r3, T, corners, agreement, samples)"""
import itertools, json, sys
import numpy as np
from PIL import Image


def _classes(im):
    """the surface's two colour classes, learned: 6-means on the saturated pixels of the lower 65% of the frame, then
    the pair of classes that touch each other most often (a square pattern is where two colours meet everywhere)"""
    h = im.shape[0]
    low = im[int(h * 0.35):]
    flat = low.reshape(-1, 3)
    sat = flat.max(1) - flat.min(1)
    ok = (sat > 45) & (flat.max(1) > 40)
    rng = np.random.default_rng(0)
    px = flat[ok][rng.choice(int(ok.sum()), min(int(ok.sum()), 20000), replace=False)].astype(float)
    k = 6
    c = px[rng.choice(len(px), k, replace=False)]
    for _ in range(30):
        lab = np.argmin(((px[:, None, :] - c[None]) ** 2).sum(-1), 1)
        c = np.array([px[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    full = np.argmin(((low[..., None, :] - c[None, None]) ** 2).sum(-1), -1)
    full[~ok.reshape(low.shape[:2])] = -1
    best = None
    for a, b in itertools.combinations(range(k), 2):
        A, B = full == a, full == b
        touch = int((A[:, :-1] & B[:, 1:]).sum() + (B[:, :-1] & A[:, 1:]).sum() + (A[:-1] & B[1:]).sum() + (B[:-1] & A[1:]).sum())
        if best is None or touch > best[0]:
            best = (touch, a, b)
    pair = c[[best[1], best[2]]]
    return pair[np.argsort(pair.sum(1))]                       # dark class, light class


def fit(frame_png, sizes=range(3, 11)):
    im = np.asarray(Image.open(frame_png).convert("RGB")).astype(float)
    Hh, Ww = im.shape[:2]
    dark_c, light_c = _classes(im)
    d2 = lambda c: ((im - c) ** 2).sum(-1)
    sat = im.max(-1) - im.min(-1)
    tol = (0.35 * np.linalg.norm(light_c - dark_c)) ** 2
    light = (d2(light_c) < tol) & (sat > 30)
    dark = (d2(dark_c) < tol) & (sat > 30)
    light[: int(Hh * 0.35)] = False
    dark[: int(Hh * 0.35)] = False
    # the first quadrilateral: where the two classes meet (the pattern's edges), its largest connected region, and that
    # region's outer frame (leftmost / rightmost per row, top / bottom per column)
    from scipy import ndimage
    E = np.zeros_like(light)
    E[:, :-1] |= (light[:, :-1] & dark[:, 1:]) | (dark[:, :-1] & light[:, 1:])
    E[:-1] |= (light[:-1] & dark[1:]) | (dark[:-1] & light[1:])
    lab, k = ndimage.label(ndimage.binary_dilation(E, iterations=6))
    if k:
        big = 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, k + 1))))
        region = ndimage.binary_fill_holes(lab == big)         # the squares inside the edges, whole
        light = light & region
        dark = dark & region
    rows = [y for y in range(Hh) if light[y].sum() > 3]
    cols = [x for x in range(Ww) if light[:, x].sum() > 3]
    ys, xs = np.array(rows, float), np.array(cols, float)

    def robust(a, b):
        p = np.polyfit(a, b, 1)
        for _ in range(3):
            r = np.abs(b - np.polyval(p, a))
            keep = r <= np.percentile(r, 75)
            p = np.polyfit(a[keep], b[keep], 1)
        return p
    pl = robust(ys, np.array([np.nonzero(light[int(y)])[0].min() for y in ys], float))
    pr = robust(ys, np.array([np.nonzero(light[int(y)])[0].max() for y in ys], float))
    pb = robust(xs, np.array([np.nonzero(light[:, int(x)])[0].max() for x in xs], float))
    pt = robust(xs, np.array([np.nonzero(light[:, int(x)])[0].min() for x in xs], float))

    def meet(pxy, pyx):                                        # x = a y + b meets y = c x + d
        a, b = pxy
        c, d = pyx
        y = (c * b + d) / (1 - c * a)
        return [a * y + b, y]
    quad0 = np.array([meet(pl, pb), meet(pr, pb), meet(pr, pt), meet(pl, pt)])

    def homog(c, n):
        src = np.array([[0, 0], [n, 0], [n, n], [0, n]], float)
        A = []
        for (u, v), (x, y) in zip(src, c):
            A += [[u, v, 1, 0, 0, 0, -x * u, -x * v, -x], [0, 0, 0, u, v, 1, -y * u, -y * v, -y]]
        Hm = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
        return Hm / Hm[2, 2]
    S = [(a, b) for a in (0.2, 0.35, 0.5, 0.65, 0.8) for b in (0.2, 0.35, 0.5, 0.65, 0.8)]

    def score(c, n, parity):
        Hm = homog(c, n)
        good = tot = 0
        for i in range(n):
            for j in range(n):
                want = (i + j) % 2 == parity
                for a, b in S:
                    p = Hm @ np.array([i + a, j + b, 1.0])
                    x, y = int(round(p[0] / p[2])), int(round(p[1] / p[2]))
                    if 0 <= x < Ww and 0 <= y < Hh and (light[y, x] or dark[y, x]):
                        tot += 1
                        good += int(light[y, x] == want)
        return good / max(tot, 1), tot
    best = None
    for n in sizes:
        for parity in (0, 1):
            c = quad0.copy()
            s = score(c, n, parity)
            for step in (6, 3, 1.5, 0.75):
                better = True
                while better:
                    better = False
                    for k, ax, dd in itertools.product(range(4), range(2), (-step, step)):
                        c2 = c.copy()
                        c2[k, ax] += dd
                        s2 = score(c2, n, parity)
                        if s2[0] > s[0] + 1e-9:
                            c, s, better = c2, s2, True
            if s[1] >= 8 * n * n and (best is None or s[0] > best[0][0]):
                best = (s, n, parity, c)
    (agree, samples), n, parity, c = best
    Hm = homog(c, n)
    f, cx, cy = (Ww / 2) / np.tan(np.radians(30)), Ww / 2, Hh / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    Mm = np.linalg.inv(K) @ Hm
    lam = 1 / np.linalg.norm(Mm[:, 0])
    r1, r2, T = lam * Mm[:, 0], lam * Mm[:, 1], lam * Mm[:, 2]
    if T[2] < 0:
        r1, r2, T = -r1, -r2, -T
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3)
    if r3[1] > 0:
        r3 = -r3
    return dict(N=n, parity=parity, H=Hm.tolist(), K=K.tolist(), r1=r1.tolist(), r2=r2.tolist(), r3=r3.tolist(),
                T=T.tolist(), corners=c.tolist(), agreement=round(agree, 4), samples=samples,
                colours=dict(dark=dark_c.round(1).tolist(), light=light_c.round(1).tolist()))


def to_px(g, u, v, w=0.0):
    K, r1, r2, r3, T = (np.array(g[k]) for k in ("K", "r1", "r2", "r3", "T"))
    p = K @ (u * r1 + v * r2 + w * r3 + T)
    return p[:2] / p[2]


def to_board(g, x, y):
    p = np.linalg.inv(np.array(g["H"])) @ np.array([x, y, 1.0])
    return p[:2] / p[2]


if __name__ == "__main__":
    g = fit(sys.argv[1])
    print(json.dumps({k: g[k] for k in ("N", "agreement", "samples", "corners", "colours")}))
