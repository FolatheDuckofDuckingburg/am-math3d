# AM's final lock, am_plan_slide.json: the dark pawn slides from 3,4 to 3,3 between 0.5 and 1.5 s, 91 frames.
# As run on 2 Nov 2026. Only file locations changed: run it from the am-math3d folder; it writes to
# results/chess_test/rerun/ (or the folder you pass), never over the locked files.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
dst = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "results", "chess_test", "rerun")
os.makedirs(dst, exist_ok=True)
os.chdir(ROOT)
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, "math3d")
import am_grid
out = "results/chess_test"
base = json.load(open(os.path.join(out, "am_plan.json")))
g = base["board"]; u0, v0 = base["pawn"]["at"]; hp = base["pawn"]["height_squares"]
fps, t0, T, K = 30, 0.5, 1.0, 91
rows = []
for k in range(K):
    t = k / fps; s = min(1.0, max(0.0, (t - t0) / T)); e = 3 * s * s - 2 * s ** 3
    u, v, w = u0, v0 - e, 0.0                                     # a slide: it never leaves the board
    fx, fy = am_grid.to_px(g, u, v, w); tx, ty = am_grid.to_px(g, u, v, w + hp)
    rows.append(dict(k=k, t=round(t, 4), u=round(u, 4), v=round(v, 4), w=0.0, foot_px=[round(float(fx), 2), round(float(fy), 2)], height_px=round(float(fy - ty), 2)))
plan = dict(base); plan["rows"] = rows; plan["frames"] = K
plan["intent"] = "the dark pawn on square 3,4 slides by itself one square toward the camera, to 3,3 (no cat, no lift)"
plan["law"] = "s = clamp((t - 0.5)/1.0, 0, 1); e = 3 s^2 - 2 s^3; u = u0; v = v0 - e; w = 0; [x, y, 1] ~ K (u r1 + v r2 + w r3 + T)"
blob = json.dumps(plan, indent=1).encode()
open(os.path.join(dst, "am_plan_slide.json"), "wb").write(blob)
seal = hashlib.sha256(blob).hexdigest()
open(os.path.join(dst, "am_plan_slide.sha256"), "w").write(seal + "  am_plan_slide.json\n")
print("SEALED am_plan_slide.json", seal, "| foot", rows[0]["foot_px"], "->", rows[-1]["foot_px"], "| height px", rows[0]["height_px"], "->", rows[-1]["height_px"])
