# AM Math 3D

AM is a core reasoner. It turns a video into exact 3D numbers, works out the shot you ask for as numbers for every frame, and locks those numbers before anything is generated. No video model we can use accepts numbers, so AM renders them as a guide video, and a video model repaints that guide into the shot. Every frame it paints is measured against AM's locked numbers.

This package holds:
- the whole pipeline;
- both tests (the Mouse Jump and Cats Playing Chess), with every take, plan, score and price;
- the tools that reproduce each proof byte for byte;
- the AM Map app;
- the website's source.

| Where | Link |
|---|---|
| Website (public, GitHub Pages) | https://devkancheti4-design.github.io/am-exact-video/ (repo `devkancheti4-design/am-exact-video`) |

The website's repo holds the pages, the videos and the proof files; this repo holds the code behind them.

## What it proved

**Final test: Cats Playing Chess. PASS.**
- AM locked the pawn's place and size for all 91 frames (SHA-256 `e9c256d1…83ab`) before anything was generated.
- One Kling 4.0 Edit take followed AM within a median of 1.7 px, worst 3.4 px, over the 73 frames AM's plan covers.
- The pawn came to rest 1.5 px from AM, on square 3,3.
- The rule was fixed before generation: within 5 px at every frame and ending on 3,3.
- AM's part took 166 s on this Mac, 0 tokens and about $0.0014. The take cost about $1.05.

**First test: the Mouse Jump.**
- AM's guide plus one FLUX 3 Edit take landed the mouse 3 px from AM's spot at 1.29 s (1.33 s asked), with a median of 3.7 px from AM's path over all 97 frames.
- Prompts alone took 6 tries at the plate leap and $0.87, and never landed it.

## Why AM renders a guide video

```
AM's core      →  raw numbers for every frame: where, when, how big
                  ↓  no video model we can use accepts numbers
AM renders     →  a guide video: the object's own pixels placed at AM's numbers, frame by frame
                  ↓
a generator    →  repaints the guide into a natural video
                  ↓
measurement    →  every painted frame against AM's locked numbers
```

1. **AM's core outputs raw numbers.** These are rows of the final lock, `results/chess_test/am_plan_slide.json`. `u`, `v` and `w` are the pawn's place in board squares; `foot_px` and `height_px` are the same place on screen, through AM's lens.

   ```
   {"k": 0,  "t": 0.0, "u": 3.2367, "v": 4.4161, "w": 0.0, "foot_px": [460.34, 303.0],  "height_px": 60.92}
   {"k": 30, "t": 1.0, "u": 3.2367, "v": 3.9161, "w": 0.0, "foot_px": [462.02, 313.04], "height_px": 63.36}
   {"k": 45, "t": 1.5, "u": 3.2367, "v": 3.4161, "w": 0.0, "foot_px": [463.85, 323.97], "height_px": 66.0}
   ```

2. **No video model we can use today accepts those numbers.** Veo, FLUX and Kling take text, pictures and video. Numbers can only go in as words in a prompt, and as words they are ignored:
   - In the mouse jump, 6 prompt tries at the plate leap ($0.87) never landed on the spot.
   - The try whose prompt carried AM's raw numbers landed 221 px off, at 2.96 s instead of 1.33 s.
   - Every mouse given text stayed in its own depth plane, 9 to 16 cm deeper instead of 69 cm.
3. **So AM renders its numbers as a guide video.** The object's own pixels are placed at AM's numbers in every frame (`results/chess_test/tools/render_guide.py`; for the mouse, `math3d/am_render.py`, within 1.3 to 1.8 px of the math).
4. **A generator repaints the guide.** A video-edit model turns the cut-out into a natural video, told to keep every frame where it is. Pictures get through:
   - mouse: one FLUX 3 Edit take, 3 px from AM's spot;
   - chess: one Kling 4.0 Edit take, median 1.7 px at every frame.
5. **Precise results need a model that takes the numbers directly.** The repaint is where the remaining error comes in:
   - the generator re-draws the object from a picture instead of reading its numbers;
   - when the ask clashes with the scene, it improvises (chess takes 1 and 2);
   - the measurement only sees what was painted, to within about 2 to 3 px.

   A model that takes AM's numbers themselves as input (every frame's position, size and timing) would have nothing to reinterpret. Until one is available, the render is the bridge.

## The method, exactly

### 1. Measure

- Apple Vision tracks every object's box in every frame: centre x, y and size w, h in pixels (`am/video_eyes`).
- Depth Anything V2 Metric (small; indoor or outdoor) gives metric depth for every pixel.

### 2. Lift into 3D

The lens is assumed and stated: a pinhole with a 60-degree horizontal view, centred on the image.

```
f = (W / 2) / tan(30°)            740 px for an 854-px-wide frame
Z = 100 × median depth over the box's inner 60%          cm
X = (x − cx) · Z / f      Y = (y − cy) · Z / f
W = w · Z / f             H = h · Z / f                  cm
B = mean luminance of the inner box
```

### 3. Shared memory: cancel the camera and the light

- Six textured 48-px background patches are followed from frame 0 (`math3d/track_from`) and lifted the same way. They sit away from every object's boxes and the watermark corner.
- Camera track: `m(t) = reference + median over patches of (P(t) − P(0))`.
- Exposure: `g(t) = 128 + median luminance change of the patches`.
- Each object's own numbers: `rX = X − mX`, `rY = Y − mY`, `rZ = Z − mZ`, `W`, `H`, `oB = B − (g − 128)`.

### 4. The core reasoner: the spine and the sweep

- For each own number over the clip, AM's hives each report a candidate rule:
  - exact behaviour;
  - shape, `a·s(t) + b`;
  - a local line;
  - a repeating period.

  AM's number law picks one or answers CANNOT.
- The band is `τ = 3σ + 0.5`, with `σ = 1.4826 · median|Δ²y − median(Δ²y)| / √6`.
- Every chosen rule is written down as a formula: a chain of up to three integer steps, `a·s(t) + b`, a period table, or local values.
- Each rule comes with its proof: the frames that keep it and the frames that break it.
- Pass 2, the sweep, asks five hives (exact, shape, latest, velocity, repeat) about every hidden or broken stretch, under AM's 3D law.
- A CANNOT is part of the answer: it marks what in the clip follows no law AM can stand behind.

The 8 rules AM proved on the chess clip (`maps/sora_206/map.json`):

| Object (frames) | Number | Rule | Formula | Band | Kept / measured |
|---|---|---|---|---|---|
| 2 (0–149) | H | REPEAT | `table[t mod 18]` | 2.32 | 95 / 150 |
| 4 (10–149) | rZ | EXACT | `12*((0)-12)` | 28.65 | 88 / 140 |
| 4 (10–149) | W | REPEAT | `table[t mod 20]` | 4.13 | 87 / 140 |
| 5 (30–149) | rY | LOCAL | a straight line through the frames within 5 of t | 4.13 | 119 / 120 |
| 5 (30–149) | rZ | LOCAL | a straight line through the frames within 5 of t | 18.66 | 118 / 120 |
| 6 (90–149) | rX | EXACT | `((0)-11)<<4` | 8.67 | 38 / 60 |
| 6 (90–149) | rZ | EXACT | `((-t)//8)^12` | 12.3 | 47 / 60 |
| 6 (90–149) | oB | EXACT | `5*(min((t//4),24))` | 5.95 | 59 / 60 |

The other 28 own numbers and the camera's mX, mY and mZ are CANNOT: Sora's frames wobble and morph.

### 5. AM's geometric puzzle

The clip's last frame is built into exact geometry: the board an intent is solved on (`math3d/am_puzzle.py`).
- **The stage:** the surface things stand on, a RANSAC plane fitted to the frame's metric-depth points. It gives the camera's height above the stage and its tilt.
- **The pieces:** the subjects Apple Vision lifts out of the frame (`math3d/instances.swift`), grown into same-depth parts above their lower body. Then everything else that stands up off the stage over its own footprint, cut apart where the depth jumps. Anything taller than 60 cm or touching the top of the frame is scenery.
- **For each piece:**
  - where it stands in stage coordinates (across, forward, up);
  - its real size;
  - what it rests on: the stage, another piece, or nothing;
  - what AM proved about it over the whole clip.

### 6. The board (`math3d/am_grid.py`)

For a chessboard, `am_grid.fit(frame_png)`:
1. learns the two square colours: 6-means on the frame, then the pair of clusters that touch most;
2. seeds the board from that pair's edge map (the largest component, holes filled);
3. tries every grid size N = 3 to 10, in both parities;
4. refines the four corners until the pattern fits, scored on 25 samples per square;
5. recovers the pose through the lens: `H ∝ K [r1 r2 T]`, so `[x, y, 1] ∝ K·(u·r1 + v·r2 + w·r3 + T)`.

`to_px(g, u, v, w)` and `to_board(g, x, y)` convert between board squares and pixels.

On the chess clip: N = 5, and 98.7% of 303 samples agree. The camera sees the board from 13.1° above its surface.

### 7. The intent, solved as numbers

The intent is the question. AM answers with numbers for every frame, or with CANNOT.
- On a board: the piece's square (u, v, w) at every frame, its foot point and its height on screen.
- On a stage (`math3d/am_intent.py`, `solve(mapdir, mover, target, height, flight, wait, seconds)`): a gravity arc with `Hap = g·T²/8`, with collisions checked only in the airborne frames.

### 8. The lock

The plan JSON is written and its SHA-256 recorded before any generation. The plan also carries the SHA-256 of `math3d/am_grid.py` that made it (`93d87375…`, unchanged since). After a take comes back, `shasum -a 256 -c` checks the lock.

### 9. AM draws its own numbers: the guide

- Apple Vision's subject mask lifts the object out of the start frame (`math3d/subject_mask`). For the pawn this ran on a 4× crop (`results/chess_test/pawn34_crop.png` → `pawn34_mask.png`), clipped at its foot.
- The background has the object removed. The hole is filled along the board's own pattern, two or four squares across, then from the nearest clean pixels.
- In every frame the object's own pixels go to AM's foot point at AM's height. Pieces nearer the camera are pasted back over it.
- The mouse's guide is checked directly against the math: 1.3 px (pan jump) and 1.8 px (plate leap) at the mouse's centre.

### 10. Giving the math to a video model

Text doesn't reach a video model's geometry; pictures do. Give a video-edit model AM's guide video with an instruction to keep every frame where it is and only make the scene look real. AM's math sets where, when and how big. Looks, costume and obedience to the instruction belong to the painter.

## The final test: Cats Playing Chess

### The request, and AM's answer

The ask was "The grey cat moves a pawn two steps."
- AM's answer was CANNOT: every pawn's second square ahead is taken.
- The move became one square: the dark pawn on 3,4 goes to 3,3, toward the camera.
- The final ask: the pawn slides by itself, between 0.5 and 1.5 s.

```
s = clamp((t − 0.5) / 1.0, 0, 1)
e = 3s² − 2s³                       starts and stops at rest
u = 3.2367   v = 4.4161 − e   w = 0
[x, y, 1] ∝ K·(u·r₁ + v·r₂ + w·r₃ + T)      K: f = 739.6 px, centre (427, 240)
foot (460.3, 303.0) → (463.9, 324.0) px;  height 60.9 → 66.0 px
```

### Locks and takes

| Lock | SHA-256 | Move | Take | Result |
|---|---|---|---|---|
| `am_plan.json` | `14de6e998d9e32da6295429e4064c8e06a17ac9eca5b2de386479ad3c52a0893` | grey cat moves 3,4 → 3,3, lifted 0.6 square, 75 frames | 1 · FLUX 3 Edit, `output_flux.mp4` | FAIL: moved to another column in frame 0, pushed 2.1 squares, ended 2.24 squares off (`score.json`) |
| `am_plan_3s.json` | `29fad5aa52dbc55f611426e62d60654cf8036c0fa2b91d7fa2e3830df9501099` | the same, 91 frames; first 75 rows unchanged | 2 · Kling 4.0 Edit, `output_kling.mp4` | FAIL: the grey cat moved a piece on its own side, 3.4 squares off (`score_kling.json`) |
| `am_plan_3s.json` | same | orange-cat prompt | 3 · Kling 4.0 Edit, `output_orange.mp4` | FOLLOWS, NOT EXACT: median 5.9 px, worst 27 px in the lift, ends 2.9 px on 3,3 (`score_orange.json`) |
| `am_plan_slide.json` | `e9c256d12ac8d464a3ddda02525f28871180ffb5a954e94284432656543083ab` | the pawn slides by itself, 91 frames | 4 · Kling 4.0 Edit, `output_slide.mp4` | **PASS**: median 1.7 px, worst 3.4 px in the plan, at rest 1.5 px on 3,3 (`score_slide_final.json`, `score_final.json`) |

One earlier plan was voided before any generation: it moved the crowned piece, picked by height instead of shape.

### How the final take was measured

- **Identical detector and window.** `score_final.py` finds the pawn's head (a dark blob with a round top) in AM's guide and in Kling's take. The window is placed from AM's locked foot point and height at that moment. Each row of `score_slide_final.json` is: frame, time, the head in the guide (y, x), the head in the take (y, x), and the distance in px at 854×480.
- **Over the plan, 0 to 3.0 s (73 frames):** median 1.69 px, worst 3.43 px, last 1.49 px.
- **From 3.5 s on, the detector reads 8.1 px.** `align_check.py` lines up the resting pawn at 3.5 s and at 4.0 s against 3.0 s and finds a shift of (0, 0) px: a highlight, not a move.
- **The scene stayed put.** `scene_check.py` matched 16 still grey pawns between the start frame and the take:
  - scale 1.0009, 0.9996 and 0.9997 at take frames 0, 48 and 96;
  - shift (0.8, −0.6 to −0.7) px;
  - worst residual 1.2, 1.0 and 0.9 px.

  The second half of `scene_check.py` re-scores with an earlier detector that was dropped, because it compared the head top against AM's computed top. That is its `errors` list.
- **Detectors dropped earlier:** one locked onto dark squares (orange take), and one compared different points of the pawn.

### Reproduce every proof

The tools in `results/chess_test/tools/` are the code as run on 2 November 2026; only file locations changed. Run them from this folder. They write to `results/chess_test/rerun/` (or the folder you pass), never over the locked files.

```bash
python3 results/chess_test/tools/seal_plan.py
```

```bash
python3 results/chess_test/tools/seal_3s.py
```

```bash
python3 results/chess_test/tools/seal_slide.py
```

```bash
cd results/chess_test && for f in am_plan am_plan_3s am_plan_slide; do cmp rerun/$f.json $f.json && echo "$f identical"; done
```

```bash
python3 results/chess_test/tools/render_guide.py results/chess_test/am_plan_slide.json results/chess_test/rerun/guide.mp4
```

```bash
python3 results/chess_test/tools/score_final.py
```

```bash
python3 results/chess_test/tools/scene_check.py
```

```bash
python3 results/chess_test/tools/align_check.py
```

Checked on 2 November 2026:

| Tool | What it rebuilds | Result |
|---|---|---|
| `seal_plan.py` | `am_plan.json`, from `start_frame.png` alone | identical, `14de6e99…0893` |
| `seal_3s.py` | `am_plan_3s.json` | identical, `29fad5aa…1099` |
| `seal_slide.py` | `am_plan_slide.json` | identical, `e9c256d1…83ab` |
| `render_guide.py` | `chess_slide_guide_1280x720.mp4` | identical, `827fefc5…7afe` |
| `score_final.py` | `score_slide_final.json` | identical |
| `scene_check.py` | `score_slide_scene.json` | identical; 16 still pawns |
| `align_check.py` | the 3.5 s and 4.0 s alignment | (0, 0) px |

`label_video.py` draws the website's labelled videos (`site/media/chess_label_*.mp4`): AM's board, AM's locked box and path, and the distance at every frame.

The website's videos are byte-identical to the measured files. For example, `site/media/chess_kling_slide.mp4` = `output_slide.mp4` (`38846c8c…`).

### Why two takes went their own way

The pawn stands next to the orange cat, far from the grey cat. Asked to have the grey cat move it, FLUX and Kling made the grey cat move a piece within its own reach. When the ask fit the scene, Kling followed AM, except where it invented the paw's lift. With nothing to invent, it followed every frame.

## The first test: the Mouse Jump

### The last frame in 3D

| | Box (px, 854 × 480) | 3D |
|---|---|---|
| Mouse body | (425, 298, 64, 104) | 104 cm away, X 0, Y 8; 9 × 14.6 cm |
| Tomato plate | (738, 298, 124, 28) | 174 cm away, X 72, Y 13 |

The camera is 27.7 cm above the counter, tilted 3.8° down, and the stage fits to 0.63 cm. AM's core ruled 14 of 30 numbers. The camera track and the mouse's past motion are CANNOT, so the jump starts from the measured last frame.

### The action as exact math

Frames `k` are at 24 fps. Take-off is at `k0 = 24` (1.00 s), with `AIR` frames in the air and `s = (k − k0) / AIR`, held between 0 and 1.

```
X(k) = X0 + (X1 − X0)·s          Z(k) = Z0 + (Z1 − Z0)·s
Y(k) = Y0 + (Y1 − Y0)·s − 4·Hap·s·(1 − s)
Hap  = g·T² / 8,   T = AIR / 24 s,   g = 981 cm/s²
```

- The height's second difference is the same in every airborne frame, `8·Hap / AIR²` cm per frame², which is gravity exactly.
- The mouse's real size never changes. On screen: `x = cx + f·X/Z`, `y = cy + f·Y/Z`, `w = f·W/Z`, `h = f·H/Z`.
- As asked (14 cm): `Hap` 13.6 cm, `AIR` 8 frames (0.333 s), landing at 1.33 s.
- AM's feasible leap is the highest arc that keeps the mouse and its hat in frame: `AIR` 12 frames, `Hap` 30.7 cm, 2.0 m/s, landing at 1.50 s.
- This jump equation and the guide drawing were written during the test from AM's measurements. AM's core reasoner answered CANNOT on the clip's own motion, so none of its rules went into the jump.

### Results

Same start frame. The Veo takes are Veo 3.1 Lite through ElevenLabs, 4 s at 720p, 727 credits per take.

**Plate leap (AM's spot: x 738 in the 854-px frame, landing at 1.33 s):**

| Take | Off the spot | Landed | Notes |
|---|---|---|---|
| **AM renders its own math** | 1.8 px (direct check) | 1.33 s | the same Sora mouse, shrinks to 60% exactly |
| **FLUX 3 Edit repaints AM's render** | **3 px** | **1.29 s** | median 3.7 px from AM's path (mean 8.1, max 51.5); 0.95× AM's height; natural motion; 735 credits |
| Veo + AM's end frame + AM's numbers | 4 px | 2.33 s | three hops |
| Veo, optimised words | 53 px | about 2.8 s | flew out of the frame, turned its back |
| Veo, AM's numbers, AM's feasible leap | 79 px | about 1.5 s | in front of the plate |
| Veo, AM's numbers, labelled | 149 px | never settled | |
| Veo, words + AM's numbers | 186 px | 1.42 s | a nearer plate |
| Veo, normal words | 201 px | 1.79 s | a nearer plate |
| Veo, AM's raw numbers | 221 px | 2.96 s | |

Every Veo mouse given text stayed in its own depth plane: 9 to 16 cm deeper instead of 69 cm.

**Pan jump (13.6 cm up, top at 1.17 s, back on the spot at 1.33 s):**
- AM is within 1.3 px of its math.
- Veo with AM's end frame landed 3.7 px off, at the same size.
- The text-only takes landed 11.7 to 43 px off, at 75% to 123% of the height and with the wrong timing.

**Notes on the record files:**
- `results/data/plate/score_G_final.json` says `size_ratio 1.45`. That compared mixed boxes, one of them with the hat. Apple Vision's box at the end on both AM's guide and the take gives height 0.95 (width 0.61).
- The chef's hat floating in that take came from the guide's background, which still held the hat because Apple Vision's mask had missed its top. AM's puzzle of the take showed the hat 36.6 cm above the counter, resting on nothing.
- A cleaner guide, with the whole mouse and hat moving, was blocked three times by FLUX's filter (not charged).

## Cost and price

Machine time is priced at about $0.03 an hour: an owned Mac, its price spread over three years, plus power. Credits are priced at $6 per 30,000 ($0.0002 each). Kling credits use its listed rate of 1,730 a second.

| The chess shot | Measured | Cost |
|---|---|---|
| AM maps the clip (eyes, depth, core reasoner) | 149.5 s (AM's reasoning 116.3 s) | $0.0012 |
| Board, lock and guide | about 16 s | $0.0001 |
| Tokens | 0 | $0 |
| One Kling 4.0 Edit take (3.03 s guide) | about 5,245 credits | about $1.05 |
| **One exact shot** | **1 take** | **about $1.05** |

| For comparison | Takes | Cost | Exact? |
|---|---|---|---|
| Chess, before the ask and the scene agreed | 3 | about $2.19 | No |
| Mouse jump, prompts only | 6 | $0.87 | Never |
| Mouse jump, AM's guide + FLUX 3 Edit | 1 | about $0.15 | 3 px at landing |

| The mouse shot | Measured | Cost |
|---|---|---|
| Solving the clip's 3D puzzle, once | 281 s | about $0.0023 |
| AM's render, 4 s | 4.6 s | about $0.00004 |
| AM's end frame | 4.6 s | about $0.00004 |
| Labelled video, 4 s | 6.2 s | about $0.00005 |
| One video-model take, 4 s | 727–735 credits | about $0.15 |

## Run it

```bash
pip install -r requirements.txt
```

```bash
./build.sh
```

```bash
./run_mouse_jump.sh
```

The mouse run writes everything to `math3d/out/`:
- `eyes.json`, the tracks;
- `puzzle3d.json`, AM's solved puzzle with its proof;
- `jump/*/plan.json`, every frame's 3D state and box, plus both prompts;
- AM's renders and the labelled videos.

The depth models download from Hugging Face on first use, about 100 MB each.

**The painter step, by hand:**
1. In ElevenLabs Image & Video, choose FLUX 3 Edit or Kling 4.0 Edit.
2. Drop the guide video into Video:
   - mouse: `math3d/out/plate_AM.mp4`, or `results/media/guide_am_plate_leap.mp4`;
   - chess: `results/chess_test/chess_slide_guide_1280x720.mp4`.
3. Paste the matching prompt (`results/chess_test/prompt_slide.txt` for chess).

To score a mouse take:

```bash
python3 math3d/intent3d.py score math3d/out plate G downloaded_video.mp4
```

## The AM Map app

A local page that shows how AM maps a video, step by step:
- AM's eyes on the video;
- the 3D map: the room from metric depth, every object's place and size, the camera and its track;
- AM's verdict on every number, with the proof behind each rule;
- the last frame as AM's geometric puzzle.

```bash
python3 map_app.py
```

Then open http://127.0.0.1:8792. Choose **Map a video** to map your own clip; it never leaves this Mac. To build a map from the command line instead:

```bash
python3 math3d/am_map.py samples/sora_mouse_chef.mp4 maps/sample indoor
```

```bash
python3 math3d/am_puzzle.py maps/sample
```

What a map holds (`maps/<id>/map.json`):
- every object's box, 3D place and size, and own numbers, in every frame;
- AM's verdict on each own number: the rule it wrote, its band, its value at every frame, and the frames that keep it and break it; or CANNOT;
- the camera's track, and AM's verdict on it;
- the sweep's answers on broken or hidden stretches;
- the room, as coloured points from metric depth at six keyframes;
- the time and cost of every step.

On every map built so far, AM's value at every frame reproduces its proof exactly: 14 of 14 ruled numbers on the Sora mouse, 11 of 11 on the FLUX take.

| Map | Frames | Numbers ruled | Camera | Time | Cost |
|---|---|---|---|---|---|
| Sora mouse chef (`maps/sample`) | 150 at 30 fps | 14 of 30 | CANNOT on all three | 112.5 s | about $0.0009 |
| FLUX repaint take (plate_G) | 97 at 24 fps | 11 of 18 | up/down and depth ruled | 62.9 s | about $0.0005 |
| Sora cats chess (`maps/sora_206`) | 150 at 30 fps | 8 of 36 | CANNOT on all three | 149.5 s | about $0.0012 |

Predicting the clip's next frames without an intent was tried and dropped. Over 2,790 number-frames hidden from AM:
- continuing the anchor was right 1,308 times and wrong 686;
- simply holding the last value was right 1,903 times.

That is not the job anyway: the intent sets the question, and the puzzle is what it is asked on. `math3d/am_extend.py` is kept only as the record.

## Files

| Path | What it is |
|---|---|
| `am/` | AM's core, unchanged from AM-Video: eyes, hives, laws (machine-authored by integer program synthesis; `lanes_*.json`), the C eyes |
| `math3d/am3d.py` | measure, lift, shared memory, spine and sweep, proof |
| `math3d/am_dict.py` | rules written down as formulas and evaluated |
| `math3d/am_map.py` | builds a map: eyes, 3D solve, room points, AM's values for every rule |
| `math3d/am_puzzle.py`, `math3d/instances.swift` | AM's geometric puzzle of a frame: stage, camera, pieces, relations |
| `math3d/am_grid.py` | the board: colours, grid size, corners, pose through the lens |
| `math3d/am_intent.py` | an intent solved on the puzzle: the gravity arc, the clean background, the render |
| `math3d/intent3d.py` | the mouse jump as exact math, prompts, scoring |
| `math3d/am_render.py`, `math3d/make_plate.py` | AM draws its own math; the mouse mask and the empty-pan background |
| `math3d/explain_video.py` | AM's math drawn on any take, frame by frame |
| `math3d/am_extend.py` | prediction without an intent: the record of why it was dropped |
| `math3d/track_from.swift`, `subject_mask.swift` | Apple Vision helpers |
| `map_app.py`, `ui/map.html`, `ui/vendor/three.min.js` | the AM Map app (three.js r128, MIT, kept here so it works offline) |
| `maps/sample`, `maps/sora_206` | AM's maps of the Sora mouse and the Sora chess clip |
| `samples/` | the Sora clip (Rapidata, Apache-2.0), the mask and the background |
| `results/chess_test/` | the final test: start frame, the three locks and hashes, guides, every take, prompts, scores, the pawn's mask |
| `results/chess_test/tools/` | the code that made and checks each chess proof |
| `results/data/`, `results/media/`, `results/ask_am/` | the mouse test: plans, scores, takes, last frames, labelled videos |
| `site/` | the website's source; the public copy is the `am-exact-video` repo |

## Notes

- The lens is assumed (a 60-degree view), and depth comes from a model, not a measured camera.
- The empty pan behind AM's mouse is a colour-matched patch from a generated frame.
- Each prompt style was run once. Video models vary from run to run.
- **Corrections made on 2 November 2026 after re-checking against the locked pose:**
  - The website first gave the board's tilt as 13.7°. That came from an earlier board fit; the locked board gives 13.1°.
  - The website first said a square was "about 90 px wide". That is true only at the near edge; at the pawn a square is about 63 px wide, and the move is 21 px down the screen.
