"""MAKE PLATE: the two pictures AM's render needs, built the way the test built them.
  MASK: Apple Vision lifts the mouse out of the start frame (subject_mask, the point is the mouse's box centre).
  BACKGROUND: the start frame with the mouse lifted out.  No frame of the Sora clip shows the pan empty, so the hole is
    filled from another frame where it is empty (in the test: frame 53 of a generated take, same camera), colour-matched
    to the ring around the mouse and feathered in.  Nothing in that patch moves.
    python3 make_plate.py START.png MASK_X MASK_Y BACKGROUND_SOURCE.mp4 FRAME OUTDIR"""
import os, subprocess, sys
import av
import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))


def main(start_png, mx, my, src_mp4, frame, out):
    os.makedirs(out, exist_ok=True)
    mask_png = os.path.join(out, "mouse_mask.png")
    subprocess.run([os.path.join(HERE, "subject_mask"), start_png, mask_png, str(mx), str(my)], check=True)
    start = np.asarray(Image.open(start_png).convert("RGB")).astype(float)
    m8 = Image.open(mask_png).convert("L")
    hole = np.asarray(m8.filter(ImageFilter.MaxFilter(15))) > 127
    alpha = np.asarray(Image.fromarray((hole * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))).astype(float) / 255
    ring = (np.asarray(m8.filter(ImageFilter.MaxFilter(61))) > 127) & ~hole
    with av.open(src_mp4) as c:
        for k, x in enumerate(c.decode(video=0)):
            if k == frame:
                src = np.asarray(x.to_image().convert("RGB").resize((start.shape[1], start.shape[0]))).astype(float)
                break
    for ch in range(3):
        s, t = src[..., ch][ring], start[..., ch][ring]
        src[..., ch] = (src[..., ch] - s.mean()) / (s.std() + 1e-6) * t.std() + t.mean()
    plate = start * (1 - alpha[..., None]) + np.clip(src, 0, 255) * alpha[..., None]
    Image.fromarray(plate.astype(np.uint8)).save(os.path.join(out, "clean_plate.png"))
    print("wrote", mask_png, "and", os.path.join(out, "clean_plate.png"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6])
