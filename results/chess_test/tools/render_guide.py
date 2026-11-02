# AM's guide video: AM's locked numbers drawn. The pawn's own pixels go to AM's foot point at AM's height in every frame.
# As run on 2 Nov 2026 (only the mask's location changed). From the am-math3d folder:
#   python3 results/chess_test/tools/render_guide.py results/chess_test/am_plan_slide.json results/chess_test/rerun/guide.mp4
# pawn34_mask.png is Apple Vision's subject mask (math3d/subject_mask) of the pawn on a 4x crop, pawn34_crop.png.
import json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
sys.path.insert(0, "math3d")
import am_grid
plan_path, guide_path = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(os.path.abspath(guide_path)), exist_ok=True)
out = "results/chess_test"
plan = json.load(open(plan_path)); g = plan["board"]; rows = plan["rows"]
fx0, fy0 = plan["pawn"]["foot_px"]
frame = Image.open(os.path.join(out, "start_frame.png")).convert("RGB")
im = np.asarray(frame).astype(float); Hh, Ww = im.shape[:2]
R, G, B = im[..., 0], im[..., 1], im[..., 2]
blk = (R < 70) & (G < 45) & (B < 45)
lab, k = ndimage.label(ndimage.binary_dilation(blk, iterations=2))
X0, Y0, X1, Y1 = int(fx0) - 55, int(fy0) - 85, int(fx0) + 55, int(fy0) + 20
mk = np.asarray(Image.open(os.path.join(out, "pawn34_mask.png")).convert("L").resize((X1 - X0, Y1 - Y0), Image.BILINEAR)) > 110
mask = np.zeros((Hh, Ww), bool); mask[Y0:Y1, X0:X1] = mk
mask[int(fy0) + 2:] = False
lid = lab[int(fy0) - 1, int(round(fx0))]
if lid: mask &= ndimage.binary_dilation(lab == lid, iterations=4)
L2, k2 = ndimage.label(mask)
if k2 > 1: mask = L2 == (1 + int(np.argmax(ndimage.sum(np.ones_like(L2), L2, range(1, k2 + 1)))))
ys, xs = np.nonzero(mask); bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
grey_all = (np.abs(R - G) < 18) & (np.abs(G - B) < 18) & (R > 95)
glab, gk = ndimage.label(ndimage.binary_opening(grey_all, iterations=1))
others = np.zeros((Hh, Ww), bool); near = []
for i in range(1, gk + 1):
    m = glab == i
    if m.sum() < 80 or (m & mask).sum() > 0.5 * m.sum(): continue
    others |= m
    yy, xx = np.nonzero(m); fy = yy.max(); fx = xx[yy >= fy - 2].mean()
    uu, vv = am_grid.to_board(g, fx, fy)
    if -0.3 <= uu <= g["N"] + 0.3 and -0.3 <= vv <= g["N"] + 0.3:
        near.append((float(vv), ndimage.binary_dilation(m, iterations=1)))
others = ndimage.binary_dilation(others, iterations=1)
hole = ndimage.binary_dilation(mask, iterations=3) & ~others
pieces_px = ndimage.binary_dilation(grey_all | blk, iterations=3) & ~hole
Hm = np.array(g["H"]); Hi = np.linalg.inv(Hm)
plate = im.copy(); rest = np.zeros((Hh, Ww), bool)
for y, x in zip(*np.nonzero(hole)):
    p = Hi @ np.array([x, y, 1.0]); uu, vv = p[:2] / p[2]; done = False
    if 0 <= uu <= g["N"] and 0 <= vv <= g["N"]:
        for du in (2, -2, 4, -4):
            if 0 <= uu + du <= g["N"]:
                q = Hm @ np.array([uu + du, vv, 1.0]); qx, qy = int(round(q[0] / q[2])), int(round(q[1] / q[2]))
                if 0 <= qx < Ww and 0 <= qy < Hh and not pieces_px[qy, qx] and not hole[qy, qx]:
                    plate[y, x] = im[qy, qx]; done = True; break
    if not done: rest[y, x] = True
idx = ndimage.distance_transform_edt(rest | pieces_px | hole, return_distances=False, return_indices=True)
plate[rest] = im[idx[0][rest], idx[1][rest]]
blur = np.asarray(Image.fromarray(plate.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))).astype(float); plate[rest] = blur[rest]
plate_img = Image.fromarray(plate.astype(np.uint8))
sprite = frame.crop((bx0, by0, bx1, by1))
alpha = Image.fromarray((mask[by0:by1, bx0:bx1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))
h0 = rows[0]["height_px"]
with tempfile.TemporaryDirectory() as tmp:
    for row in rows:
        s = row["height_px"] / h0; fx, fy = row["foot_px"]
        nw, nh = max(1, round((bx1 - bx0) * s)), max(1, round((by1 - by0) * s))
        left, top = round(fx + (bx0 - fx0) * s), round(fy + (by0 - fy0) * s)
        f = plate_img.copy(); f.paste(sprite.resize((nw, nh), Image.LANCZOS), (left, top), alpha.resize((nw, nh), Image.LANCZOS))
        fa = np.asarray(f).copy()
        for vv, m in near:
            if vv < row["v"]: fa[m] = im[m].astype(np.uint8)
        Image.fromarray(fa).resize((1280, 720), Image.LANCZOS).save(os.path.join(tmp, "f_%04d.png" % row["k"]))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(plan["fps"]), "-i", os.path.join(tmp, "f_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", guide_path], check=True)
import av
with av.open(guide_path) as c:
    n = sum(1 for _ in c.decode(video=0))
print(guide_path, n, "frames,", round(n / plan["fps"], 3), "s")
