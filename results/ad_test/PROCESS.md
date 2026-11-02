# Vazram ad: AM's sealed record

One product photo, one intent, and AM's equation for every frame of a 15-second camera move, sealed before anything is generated. No tokens anywhere.

**The intent:** a 15-second hero ad: the camera arcs 8 degrees to the left around the notebook, revealing its spine, eases in and out, and pushes in until the notebook fills 85% of the frame height, centred.

| Step | Measured | What came out |
|---|---|---|
| AM's eyes and map of the photo | 19.1 s on the Mac | the notebook found as a piece |
| AM's puzzle | 2.5 s | the desk fitted as a plane (0.87 cm), the camera 23.9° above it, the notebook P1 |
| Metric depth through AM's stated lens | 4.9 s | desk 86, pen 108, notebook 114, books 169, wall 225 (AM's units; the absolute scale is off on a product photo, the equation uses angles and fractions) |
| The equation solved | 0.09 s | arc 8° to the left; end distance 0.896 of the start, so the notebook fills 85% of the frame height, centred |
| The lock | 0.01 s | SHA-256 `8b9b09135f34d2e69ae9d9ab34ef2a2c8b4821754676063edaf8d968c98b2e66` |
| The guide drawn from the sealed rows | 228.0 s | 450 frames, 720×1280 at 30 fps; unseen by the photo: median 4.2%, at most 10.2% of a frame |

**The equation:**
```
s = clamp((t - 1.5) / 12.0, 0, 1), e = 3 s^2 - 2 s^3
theta = -8.0 e degrees about n through B (negative: to the left)
d = 113.602 + (101.784 - 113.602) e
C = B + R_n(theta) (0 - B) d / d0; looks at (1 - e) L0 + e B
x = cx + f q_x / q_z,  y = cy + f q_y / q_z,  q = R(t) (P - C(t))   for every point P of the scene
```
**The rule the take is held to:** the product, tracked with the same tracker on the guide and the take at 720x1280, within 8 px of the guide at every frame of the plan and at the end; the background patches' median within 8 px of the guide on at least 90% of the frames where three or more are tracked.

Files: `guide.mp4` (give it to Kling 4.0 Edit), `prompt.txt`, `plan.json` + `plan.sha256` (the lock: 450 rows of camera, notebook centre and box, background patches), `depth_cm.npy` + `still_720x1280.png` (the hashed inputs), `process.json`.

## The take (Kling 4.0 Edit, one pass) — verdict: FAIL

| Measure | Result | Rule |
|---|---|---|
| Lock | plan.json still matches `8b9b0913…2e66` | — |
| Take length | **4.04 s** (1080×1920, 24 fps) of AM's 15.0 s plan | the plan covers 15 s |
| Notebook vs AM's guide, same moment, same tracker | median 6.9 px, worst 15.9 px, at the take's end 13.9 px | ≤ 8 px every frame: fails |
| Background vs AM's camera | median 20.2 px; within 8 px on 3% of frames | ≥ 90%: fails |

Not part of the rule, measured afterwards to explain it:
- Kling ran AM's move about 2x fast. Its 4 s reaches the guide's 7.3 s, about 48% of AM's move, and then the video ends.
- Matched to the guide moment it looks most like, the notebook is close to AM's path (median 3.1 px, max 9.0), but the background is still 17 px off (max 27.7). So Kling kept the notebook roughly on AM's path, and kept neither AM's clock nor AM's camera geometry behind it.
