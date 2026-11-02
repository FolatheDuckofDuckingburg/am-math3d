# AM's 3-second lock, am_plan_3s.json: the first lock's 75 rows unchanged, then the pawn at rest on 3,3 to 91 frames (Kling needs 3 s).
# As run on 2 Nov 2026. Only file locations changed: run it from the am-math3d folder; it writes to
# results/chess_test/rerun/ (or the folder you pass), never over the locked files.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
dst = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "results", "chess_test", "rerun")
os.makedirs(dst, exist_ok=True)
os.chdir(ROOT)
import hashlib, json, os
out = "results/chess_test"
old = json.load(open(os.path.join(out, "am_plan.json")))
assert hashlib.sha256(open(os.path.join(out, "am_plan.json"), "rb").read()).hexdigest().startswith("14de6e99")
rows = list(old["rows"])
last = rows[-1]
assert last["t"] >= 1.5 and last["w"] == 0.0               # the law: after 1.5 s the pawn rests on its target
for k in range(len(rows), 91):
    r = dict(last); r["k"] = k; r["t"] = round(k / old["fps"], 4); rows.append(r)
plan = dict(old); plan["rows"] = rows; plan["frames"] = len(rows)
plan["note"] = "frames 0-74 are byte-for-byte the rows of seal 14de6e99...0893; frames 75-90 are the sealed law's rest (s = 1): the pawn on 3,3"
blob = json.dumps(plan, indent=1).encode()
open(os.path.join(dst, "am_plan_3s.json"), "wb").write(blob)
seal = hashlib.sha256(blob).hexdigest()
open(os.path.join(dst, "am_plan_3s.sha256"), "w").write(seal + "  am_plan_3s.json\n")
print("frames", len(rows), "| first 75 identical:", rows[:75] == old["rows"], "| NEW SEAL", seal)
