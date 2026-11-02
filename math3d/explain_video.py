"""EXPLAIN VIDEO: a take with AM's math drawn on it, frame by frame, so a viewer sees how the shot is computed.
  On the picture: the planned path (solid where the mouse has been, dotted ahead), AM's box for this frame, the
  take-off, top and landing points.  Under it: the time and phase, the mouse's 3D position, the gravity formula with
  this frame's height, and the perspective formula with this frame's size on screen.
    python3 explain_video.py VARIANT INPUT.mp4 OUTPUT.mp4 "TITLE"    (VARIANT: pan | plate)"""
import json, os, subprocess, sys, tempfile
import av
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
GREEN, WHITE, GREY, AMBER = (60, 220, 120), (240, 242, 240), (165, 172, 168), (255, 190, 90)
PANEL = 196


def font(size, mono=False):
    for p in (("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc") if mono else
              ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc")):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main(variant, src, out, title):
    p = json.load(open(os.path.join(HERE, "out", "jump", variant, "plan.json")))
    W, H = p["size"]
    f854 = p["lens"]["f"]
    k0, k1, fps = p["takeoff_k"], p["land_k"], p["fps"]
    hap = p["apex_cm"]
    real_h = p["rows"][0]["h"] * p["rows"][0]["Z"] / f854         # the mouse's real height, cm, as AM measured it
    with av.open(src) as c:
        frames = [x.to_image().convert("RGB") for x in c.decode(video=0)]
    OW, OH = 1280, 720
    sx, sy = OW / W, OH / H
    f = f854 * sx                                              # the lens in this frame's pixels
    pts = [(r["x"] * sx, r["y"] * sy) for r in p["rows"]]
    d2 = 8 * hap / (k1 - k0) ** 2                              # cm per frame^2, the same in every airborne frame
    f_title, f_lab, f_mono, f_small = font(26), font(20), font(21, True), font(17)
    apex_k = (k0 + k1) // 2
    with tempfile.TemporaryDirectory() as tmp:
        for t, img in enumerate(frames):
            k = min(t, len(p["rows"]) - 1)
            r = p["rows"][k]
            canvas = Image.new("RGB", (OW, OH + PANEL), (14, 16, 18))
            canvas.paste(img.resize((OW, OH), Image.LANCZOS), (0, 0))
            d = ImageDraw.Draw(canvas)
            for j in range(1, len(pts)):                       # the path: solid behind, dotted ahead
                if j <= k:
                    d.line([pts[j - 1], pts[j]], fill=GREEN, width=4)
                elif j % 2 == 0:
                    x, y = pts[j]
                    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=GREEN)
            for kk, lab in ((k0, "take-off 1.00 s"), (apex_k, "top: %.1f cm up, %.2f s" % (hap, apex_k / fps)),
                            (k1, "lands %.2f s, %.0f cm away" % (k1 / fps, p["land"]["Z"]))):
                x, y = pts[kk]
                d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=WHITE, width=3)
                tw = d.textlength(lab, font=f_small)
                lx = min(max(8, x + 12), OW - tw - 8)
                ly = y - 30 if kk != k0 else y + 14
                d.rectangle([lx - 4, ly - 2, lx + tw + 4, ly + 21], fill=(0, 0, 0))
                d.text((lx, ly), lab, fill=WHITE, font=f_small)
            bw, bh = r["w"] * sx, r["h"] * sy
            cx, cy = pts[k]
            d.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], outline=GREEN, width=4)
            d.rectangle([0, 0, OW, 40], fill=(0, 0, 0))
            d.text((14, 6), title, fill=WHITE, font=f_title)
            d.text((OW - 120, 8), "%.2f s" % (t / fps), fill=GREY, font=f_lab)
            # the math, this frame
            s = min(1.0, max(0.0, (k - k0) / (k1 - k0)))
            h = 4 * hap * s * (1 - s)
            phase = "still" if k < k0 else ("in the air" if k < k1 else "landed")
            size_px = f * real_h / r["Z"]
            y0 = OH + 14
            lines = [
                ("t = %.2f s   frame %d of %d   %s" % (t / fps, t, len(frames) - 1, phase), WHITE),
                ("3D position (cm, camera at 0):  X %5.1f   Y %5.1f   Z (distance) %5.1f" % (r["X"], r["Y"], r["Z"]), WHITE),
                ("height  h(t) = %.1f x (1 - ((t - %.3f) / %.3f)^2) = %4.1f cm" % (hap, (k0 + k1) / 2 / fps, (k1 - k0) / 2 / fps, h),
                 AMBER if 0 < s < 1 else GREY),
                ("gravity: h drops by %.2f cm more each frame = %.2f m/s^2 at %d fps" % (d2, d2 * fps * fps / 100, fps), GREY),
                ("perspective: size on screen = f x real height / Z = %.0f x %.1f cm / %.1f cm = %3.0f px" % (f, real_h, r["Z"], size_px),
                 WHITE),
                ("AM's box: centre (%4.0f, %4.0f) px, %3.0f x %3.0f px in this %dx%d frame" % (cx, cy, bw, bh, OW, OH), GREEN),
            ]
            for i, (txt, col) in enumerate(lines):
                d.text((18, y0 + i * 29), txt, fill=col, font=f_mono)
            canvas.save(os.path.join(tmp, "f_%04d.png" % t))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "f_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", out], check=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
