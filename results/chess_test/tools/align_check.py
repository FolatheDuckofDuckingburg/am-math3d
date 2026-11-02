# The pixel-alignment check: after 3.5 s the head detector reads 8.1 px on Kling's take. This lines the resting pawn's
# upper body at 3.5 s and 4.0 s up against 3.0 s, pixel for pixel. As run on 2 Nov 2026; only the map's location changed.
# Run it from the am-math3d folder.
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(os.path.join(ROOT, "results", "chess_test"))
import json
import av, numpy as np
from PIL import Image
with av.open("output_slide.mp4") as c:
    fr = [np.asarray(f.to_image().resize((854, 480), Image.LANCZOS).convert("L")).astype(float) for f in c.decode(video=0)]
y0, y1, x0, x1 = 235, 300, 448, 486                             # the resting pawn's upper body, above the grey pawn in front
ref = fr[72][y0:y1, x0:x1]
for k in (84, 96):
    best = min(((dy, dx) for dy in range(-10, 11) for dx in range(-10, 11)),
               key=lambda d: ((fr[k][y0 + d[0]:y1 + d[0], x0 + d[1]:x1 + d[1]] - ref) ** 2).mean())
    print("frame %d vs frame 72 (3.0 s): the pawn is shifted by %s px" % (k, best))
m = json.load(open(os.path.join(ROOT, "maps", "sora_206", "map.json")))
print("AM's map of the clip:", m["total_wall"], "s on this Mac")
