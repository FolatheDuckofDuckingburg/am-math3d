# AM's first lock, am_plan.json: fits the board, finds the dark pawn on 3,4, writes the move for every frame and its SHA-256.
# As run on 2 Nov 2026. Only file locations changed: run it from the am-math3d folder; it writes to
# results/chess_test/rerun/ (or the folder you pass), never over the locked files.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
dst = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "results", "chess_test", "rerun")
os.makedirs(dst, exist_ok=True)
os.chdir(ROOT)
import hashlib, json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
sys.path.insert(0, "math3d")
import am_grid
out = "results/chess_test"
g = am_grid.fit(os.path.join(out, "start_frame.png"))
frame = Image.open(os.path.join(out, "start_frame.png")).convert("RGB")
im = np.asarray(frame).astype(float); Hh, Ww = im.shape[:2]
R, G, B = im[..., 0], im[..., 1], im[..., 2]
blk = (R < 70) & (G < 45) & (B < 45)
lab, k = ndimage.label(ndimage.binary_dilation(blk, iterations=2))
best = None
for i in range(1, k + 1):                                   # the dark piece standing on square 3,4
    m = (lab == i) & blk
    if m.sum() < 60: continue
    yy, xx = np.nonzero(m); fy = yy.max(); fx = xx[yy >= fy - 2].mean()
    u, v = am_grid.to_board(g, fx, fy)
    if 3 <= u < 4 and 4 <= v < 5.3 and (best is None or m.sum() > best[0]):
        best = (int(m.sum()), float(fx), int(fy), float(u), float(v), int(fy - yy.min()))
_, fx0, fy0, u0, v0, tall = best
print("dark pawn on 3,4: foot px (%.1f, %d), board (%.3f, %.3f), %d px tall" % (fx0, fy0, u0, v0, tall))
hs = np.linspace(0.05, 3, 600)
hp = float(min(hs, key=lambda h: abs((am_grid.to_px(g, u0, v0, 0)[1] - am_grid.to_px(g, u0, v0, h)[1]) - tall)))
fps, t0, T, lift, n = 30, 0.5, 1.0, 0.6, 1
rows = []
for kk in range(int(2.5 * fps)):
    t = kk / fps; s = min(1.0, max(0.0, (t - t0) / T)); e = 3 * s * s - 2 * s ** 3
    u, v, w = u0, v0 - n * e, lift * np.sin(np.pi * s)
    fx, fy = am_grid.to_px(g, u, v, w); tx, ty = am_grid.to_px(g, u, v, w + hp)
    rows.append(dict(k=kk, t=round(t, 4), u=round(u, 4), v=round(v, 4), w=round(float(w), 4), foot_px=[round(float(fx), 2), round(float(fy), 2)], height_px=round(float(fy - ty), 2)))
plan = dict(intent="the grey cat moves the dark pawn on square 3,4 one square toward the camera, to 3,3",
            board=dict(N=g["N"], agreement=g["agreement"], samples=g["samples"], corners=g["corners"], K=g["K"], r1=g["r1"], r2=g["r2"], r3=g["r3"], T=g["T"], H=g["H"]),
            pawn=dict(foot_px=[fx0, fy0], at=[round(u0, 4), round(v0, 4)], height_squares=round(hp, 4), height_px=tall),
            target=dict(u=round(u0, 4), v=round(v0 - n, 4), square=[int(u0), int(v0 - n)]),
            law="s = clamp((t - 0.5)/1.0, 0, 1); e = 3 s^2 - 2 s^3; u = u0; v = v0 - e; w = 0.6 sin(pi s); [x, y, 1] ~ K (u r1 + v r2 + w r3 + T)",
            fps=fps, frames=len(rows), rows=rows, code=dict(am_grid=hashlib.sha256(open("math3d/am_grid.py", "rb").read()).hexdigest()))
blob = json.dumps(plan, indent=1).encode()
open(os.path.join(dst, "am_plan.json"), "wb").write(blob)
seal = hashlib.sha256(blob).hexdigest()
open(os.path.join(dst, "am_plan.sha256"), "w").write(seal + "  am_plan.json\n")
print("target", plan["target"], "| foot px", rows[0]["foot_px"], "->", rows[-1]["foot_px"], "| SEALED sha256", seal)
