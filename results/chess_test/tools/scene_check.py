# The still-pawn check, score_slide_scene.json: the grey pawns that never move, matched between the start frame and the take (scale and shift). Its second half re-scores the take with an earlier detector that was dropped.
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
orig = Image.open("start_frame.png").convert("RGB")
with av.open("output_slide.mp4") as c:
    fps = float(c.streams.video[0].average_rate)
    fr = [f.to_image().resize((854, 480), Image.LANCZOS) for f in c.decode(video=0)]
def grey_heads(img):
    a = np.asarray(img).astype(float)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    m = (lum > 200) & (np.abs(a[..., 0] - a[..., 1]) < 16) & (np.abs(a[..., 1] - a[..., 2]) < 16)
    m[:250] = False; m[440:] = False; m[:, :200] = False; m[:, 640:] = False
    lab, n = ndimage.label(ndimage.binary_opening(m, iterations=1))
    return [tuple(np.array(np.nonzero(lab == i))[::-1].mean(1)) for i in range(1, n + 1) if (lab == i).sum() >= 20]
ref = grey_heads(orig)
def scene_shift(img):
    cur = grey_heads(img); pairs = []
    for p in ref:
        q = min(cur, key=lambda q: np.hypot(q[0] - p[0], q[1] - p[1])) if cur else None
        if q and np.hypot(q[0] - p[0], q[1] - p[1]) < 20:
            pairs.append((p, q))
    P = np.array([p for p, _ in pairs]); Q = np.array([q for _, q in pairs])
    mp, mq = P.mean(0), Q.mean(0)
    s = np.sqrt(((Q - mq) ** 2).sum() / ((P - mp) ** 2).sum())    # similarity: scale and shift
    return s, mp, mq, len(pairs), np.abs((P - mp) * s + mq - Q).max()
for k in (0, len(fr) // 2, len(fr) - 1):
    s, mp, mq, n, res = scene_shift(fr[k])
    print("frame %2d: %d still grey pawns matched; scene scale %.4f, shift (%.1f, %.1f) px; worst residual %.1f px" % (k, n, s, *(mq - mp), res))
rows = json.load(open("score_slide_frames.json"))
top = lambda r: r["foot_px"][1] - r["height_px"]
s, mp, mq, n, res = scene_shift(fr[0])
errs = []
for k, t, b, ay, ax, d in rows:
    if not b: continue
    am = min(plan["rows"], key=lambda q: abs(q["t"] - t))
    p = (np.array([am["foot_px"][0], top(am)]) - mp) * s + mq      # AM's sealed head, carried into Kling's scene
    errs.append((k, t, float(np.hypot(b[1] - p[0], b[0] - p[1]))))
e = [x[2] for x in errs]
print("against the scene: median %.1f px, worst %.1f px (frame %d, %.2f s), last %.1f px; frames over 5 px: %d of %d"
      % (np.median(e), max(e), max(errs, key=lambda x: x[2])[0], max(errs, key=lambda x: x[2])[1], e[-1], sum(x > 5 for x in e), len(e)))
for x in errs[::8] + [errs[-1]]:
    print("  t=%.2f s: %.1f px" % (x[1], x[2]))
json.dump(dict(scene_scale=float(s), scene_shift=[float(v) for v in (mq - mp)], errors=errs), open(os.path.join(dst, "score_slide_scene.json"), "w"))
