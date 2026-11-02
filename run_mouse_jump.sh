#!/bin/sh
# The Mouse Jump Test, end to end on this Mac: AM solves the Sora clip's last frame in 3D, writes both jumps as exact
# math, renders them, and draws the math on the videos.  The painter step (a video-edit model) is manual; see README.
set -e
cd "$(dirname "$0")/math3d"
CLIP=../samples/sora_mouse_chef.mp4
python3 am3d.py eyes "$CLIP" out                      # Apple Vision's tracks -> out/eyes.json
python3 am3d.py solve "$CLIP" out indoor              # lift, shared memory, spine, sweep -> out/puzzle3d.json + proof
python3 intent3d.py plan out pan                      # jump straight up, back into the pan (14 cm, as asked)
python3 intent3d.py plan out plate                    # leap onto the tomato plate (14 cm, as asked)
python3 intent3d.py plan out plate_feasible 12        # AM's feasible leap: highest arc that keeps the mouse in frame
mkdir -p out/jump
[ -f out/jump/mouse_mask.png ] || cp ../samples/mouse_mask.png out/jump/
[ -f out/jump/clean_plate.png ] || cp ../samples/clean_plate.png out/jump/
python3 am_render.py pan out/pan_AM.mp4               # AM draws its own math
python3 am_render.py plate out/plate_AM.mp4
python3 explain_video.py pan out/pan_AM.mp4 out/explain_pan_AM.mp4 "AM's math, drawn by AM: jump back into the pan"
python3 explain_video.py plate out/plate_AM.mp4 out/explain_plate_AM.mp4 "AM's math, drawn by AM: leap onto the tomato plate"
echo "done: math3d/out/ (puzzle3d.json, jump/*/plan.json and prompts, AM renders, labelled videos)"
