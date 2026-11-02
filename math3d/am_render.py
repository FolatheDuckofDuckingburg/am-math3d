"""AM RENDER: the extension drawn by AM alone, from its solved 3D puzzle; no video model.
  Every frame k: the mouse, lifted from the Sora clip's last frame by Apple Vision's subject mask, is placed at the box
  AM's math gives for k (its 3D state through the lens): moved so its body box sits on AM's box and scaled by AM's
  perspective (its real size never changes).  Behind it, the Sora frame with the mouse lifted out; the hidden patch of
  empty pan behind the mouse is a colour-matched patch borrowed from a generated frame, and nothing in it moves.
    python3 am_render.py VARIANT OUT.mp4    (VARIANT: pan | plate | plate_feasible)"""
import json, os, subprocess, sys, tempfile, time, resource
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def render(variant, out_mp4):
    w0, c0 = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    p = json.load(open(os.path.join(HERE, "out", "jump", variant, "plan.json")))
    W, H = p["size"]
    gw, gh = p["gen"]
    sx, sy = gw / W, gh / H
    frame = Image.open(os.path.join(HERE, "out", "jump", "pan", "start_frame.png")).convert("RGB")
    plate = Image.open(os.path.join(HERE, "out", "jump", "clean_plate.png")).convert("RGB")
    mask = Image.open(os.path.join(HERE, "out", "jump", "mouse_mask.png")).convert("L")
    bx0, by0, bx1, by1 = mask.getbbox()
    sprite = frame.crop((bx0, by0, bx1, by1))
    alpha = mask.crop((bx0, by0, bx1, by1))
    b = p["start_box"]                                            # the mouse's body box in the start frame (gen pixels)
    with tempfile.TemporaryDirectory() as tmp:
        for r in p["rows"]:
            s = (r["w"] * sx) / b["w"]                            # perspective scale from AM's box
            cx, cy = r["x"] * sx, r["y"] * sy
            nw, nh = max(1, round((bx1 - bx0) * s)), max(1, round((by1 - by0) * s))
            left = round(cx + (bx0 - b["x"]) * s)
            top = round(cy + (by0 - b["y"]) * s)
            im = plate.copy()
            im.paste(sprite.resize((nw, nh), Image.LANCZOS), (left, top), alpha.resize((nw, nh), Image.LANCZOS))
            im.save(os.path.join(tmp, "f_%04d.png" % r["k"]))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(p["fps"]), "-i", os.path.join(tmp, "f_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_mp4], check=True)
    c1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime)
    print(json.dumps(dict(variant=variant, frames=len(p["rows"]), wall=round(time.time() - w0, 2), cpu=round(cpu, 2))))


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
