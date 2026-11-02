# The final score, score_slide_final.json: the same head detector, in the same window, on AM's guide and on Kling's take, frame by frame.
# As run on 2 Nov 2026. Only file locations changed: run it from the am-math3d folder; it writes to
# results/chess_test/rerun/ (or the folder you pass), never over the locked files.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
dst = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "results", "chess_test", "rerun")
os.makedirs(dst, exist_ok=True)
os.chdir(os.path.join(ROOT, "results", "chess_test"))
import json
import av, numpy as np
from PIL import Image
from scipy import ndimage
plan = json.load(open("am_plan_slide.json"))
def load(p):
    with av.open(p) as c:
        fps = float(c.streams.video[0].average_rate)
        return fps, [f.to_image().resize((854, 480), Image.LANCZOS) for f in c.decode(video=0)]
gf, guide = load("chess_slide_guide_1280x720.mp4")
kf, kling = load("output_slide.mp4")
def head(img, row):
    fx, fy = row["foot_px"]; h = row["height_px"]
    a = np.asarray(img).astype(float)
    m = (np.abs(a - np.array([35., 4., 1.])).max(-1) < 40) & (a.max(-1) < 120)
    win = np.zeros_like(m); win[int(fy - h - 18):int(fy - h * 0.55), int(fx - 18):int(fx + 19)] = True   # the pawn's upper half, wide margin
    lab, n = ndimage.label(m & win)
    if not n: return None
    i = 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))))
    yy, xx = np.nonzero(lab == i)
    return int(yy.min()), float(xx[yy <= yy.min() + 3].mean())
res = []
for k, img in enumerate(kling):
    t = k / kf
    row = min(plan["rows"], key=lambda q: abs(q["t"] - t))
    j = min(len(guide) - 1, int(round(t * gf)))
    a, b = head(guide[j], row), head(img, row)
    if a and b:
        res.append((k, round(t, 3), a, b, float(np.hypot(a[0] - b[0], a[1] - b[1]))))
e = np.array([r[4] for r in res])
print("Kling vs AM's guide (identical detector and window, %d of %d frames): median %.1f px, worst %.1f px (t=%.2f s), last %.1f px, over 5 px: %d"
      % (len(res), len(kling), np.median(e), e.max(), res[int(e.argmax())][1], e[-1], int((e > 5).sum())))
for r in res[::8] + [res[-1]]:
    print("  t=%.2f s: AM %s | Kling %s -> %.1f px" % (r[1], r[2], (r[3][0], round(r[3][1], 1)), r[4]))
json.dump(res, open(os.path.join(dst, "score_slide_final.json"), "w"))
