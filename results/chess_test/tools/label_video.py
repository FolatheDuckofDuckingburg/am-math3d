# The website's labelled videos: AM's board, locked box and path drawn on AM's guide and on Kling's take, with the
# distance at every frame. As run on 2 Nov 2026, from the am-math3d folder; writes site/media/chess_label_*.mp4.
import json, os, subprocess, sys, tempfile
import av, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "math3d")
import am_grid
out = "results/chess_test"
plan = json.load(open(os.path.join(out, "am_plan_slide.json"))); g = plan["board"]
score = json.load(open(os.path.join(out, "score_slide_final.json")))      # Kling frame -> (t, AM's head, Kling's head, px)
S = 1280 / 854.0
F = lambda n: ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", n)
f_big, f_mid, f_small = F(26), F(21), F(17)
CYAN, GREEN, MAG, WHITE, GREY, AMBER = (40, 200, 220), (70, 220, 140), (255, 90, 200), (240, 244, 242), (150, 160, 158), (255, 190, 70)
P = lambda u, v, w=0.0: tuple(np.array(am_grid.to_px(g, u, v, w)) * S)
u0, v0 = plan["pawn"]["at"]; v1 = plan["target"]["v"]
def board(dr):
    for i in range(6):
        dr.line([P(i, 0), P(i, 5)], fill=CYAN + (110,), width=1); dr.line([P(0, i), P(5, i)], fill=CYAN + (110,), width=1)
    path = [P(u0, v0 - e) for e in np.linspace(0, 1, 30)]
    dr.line(path, fill=GREEN + (200,), width=3)
    for (u, v), col, name in (((u0, v0), MAG, "start 3,4"), ((u0, v1), GREEN, "target 3,3")):
        x, y = P(u, v); dr.ellipse([x - 7, y - 7, x + 7, y + 7], outline=col + (255,), width=3)
        dr.text((x + 12, y - 4), name, fill=col + (255,), font=f_small)
def phase(t):
    return "waiting" if t < 0.5 else ("sliding" if t < 1.5 else "at rest")
def row_at(t):
    return min(plan["rows"], key=lambda q: abs(q["t"] - t))
def render(src, dst, kind):
    with av.open(src) as c:
        fps = float(c.streams.video[0].average_rate)
        frames = [f.to_image().convert("RGB").resize((1280, 720), Image.LANCZOS) for f in c.decode(video=0)]
    with tempfile.TemporaryDirectory() as tmp:
        for k, fr in enumerate(frames):
            t = k / fps; r = row_at(t)
            canvas = Image.new("RGB", (1280, 916), (14, 20, 18)); canvas.paste(fr, (0, 0))
            ov = Image.new("RGBA", (1280, 720), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
            board(dr)
            fx, fy = np.array(r["foot_px"]) * S; h = r["height_px"] * S
            dr.rectangle([fx - 0.32 * h, fy - h, fx + 0.32 * h, fy], outline=GREEN + (255,), width=3)     # AM's box at this frame
            dr.line([(fx - 10, fy), (fx + 10, fy)], fill=GREEN + (255,), width=3)
            if kind == "kling" and k < len(score):
                _, _, am_head, kl_head, d = score[k]
                ky, kx = kl_head[0] * S, kl_head[1] * S
                dr.line([(kx - 12, ky), (kx + 12, ky)], fill=AMBER + (255,), width=3); dr.line([(kx, ky - 12), (kx, ky + 12)], fill=AMBER + (255,), width=3)
            canvas.paste(Image.alpha_composite(fr.convert("RGBA"), ov).convert("RGB"), (0, 0))
            pd = ImageDraw.Draw(canvas)
            pd.text((24, 738), "t %.2f s   %s" % (t, phase(t)), fill=WHITE, font=f_big)
            pd.text((24, 782), "AM (locked):  u %.2f  v %.2f  square 3,%d   s %.2f   e = 3s²−2s³ = %.2f"
                    % (r["u"], r["v"], int(r["v"]), min(1, max(0, (t - 0.5))), 4.42 - r["v"] if t >= 0.5 else 0.0), fill=GREEN, font=f_mid)
            pd.text((24, 816), "lens:  foot (%.0f, %.0f) px   height %.0f px   size = f·H/Z" % (fx, fy, h), fill=GREEN, font=f_mid)
            if kind == "kling" and k < len(score):
                d = score[k][4]
                note = "   detector artifact: the pawn did not move after 3.0 s" if t >= 3.5 else ""
                pd.text((24, 856), "Kling's pawn head (%.0f, %.0f)   AM's (%.0f, %.0f)   off %.1f px = %.1f px at 854×480%s"
                        % (score[k][3][1] * S, score[k][3][0] * S, score[k][2][1] * S, score[k][2][0] * S, d * S, d, note), fill=AMBER, font=f_small)
            else:
                pd.text((24, 856), "green box: AM's pawn at this frame   green line: its locked path   cyan: AM's 5×5 board", fill=GREY, font=f_small)
            canvas.save(os.path.join(tmp, "f_%04d.png" % k))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "f_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", dst], check=True)
        canvas.save(dst.replace(".mp4", "_last.jpg"), quality=86)
render(os.path.join(out, "chess_slide_guide_1280x720.mp4"), "site/media/chess_label_guide.mp4", "guide")
render(os.path.join(out, "output_slide.mp4"), "site/media/chess_label_kling.mp4", "kling")
print("ok")
