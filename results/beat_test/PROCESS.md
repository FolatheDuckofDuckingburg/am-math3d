# AM finds the beat: the process record

Sora 130 (drummers in a field, 150 frames at 30 fps). Every step AM took, with its measured time. No tokens anywhere.

| Step | Measured | What came out |
|---|---|---|
| AM maps the whole clip: eyes, metric depth, shared memory, AM's core reasoner (built earlier) | 131.2 s on the Mac (AM's reasoning 96.0 s) | dancers' up-and-down: object 1 REPEAT period 14, kept 51 of 55; object 2 REPEAT period 14, kept 46 of 55; object 4 REPEAT period 14, kept 50 of 55; object 5 REPEAT period 14, kept 50 of 55; object 6 REPEAT period 14, kept 44 of 55 |
| AM maps the clip's last 55 frames (95-149) and reasons over them again | 101.1 s on the Mac (AM's reasoning 81.8 s) | dancers' up-and-down: object 1 REPEAT period 13, kept 52 of 55; object 2 REPEAT period 24, kept 51 of 55; object 5 REPEAT period 13, kept 50 of 55 |
| The pixels, without AM: how well the clip matches itself shifted by p frames | 0.86 s | best shift: frames 0-54 → 14; frames 55-149 → 13 |
| AM's law for the new frames: frame t = frame t - 13 (its period at the end of the clip) | 0.00 s | period 13 frames = 0.433 s, 138.5 beats a minute; cycle [137, 149] |
| The guide drawn: 96 frames at 30 fps, 1280x720; every frame's source recorded | 6.28 s | am_beat_guide_1280x720.mp4 |
| The lock: plan.json written, SHA-256 recorded before anything is generated | 0.00 s | b5ce4f668399b336a0396e39ac09e31346dbfadef18804fd38e0f3c7221a6cfa |

**The law for the new frames:** value(t) = value(t - P): frame t of the extension is the clip's frame t - 13k, the latest.
**Locked before generation:** SHA-256 `b5ce4f668399b336a0396e39ac09e31346dbfadef18804fd38e0f3c7221a6cfa`.
**The rule the take is held to:** the take, resampled to 30 fps, repeats best (dancers' band, lags 4-28) at 13 +- 1 frames; AM's eyes and core run on the take, and the periods they prove are reported.

Files: `am_beat_guide_1280x720.mp4` (the guide), `prompt.txt`, `plan.json` + `plan.sha256` (the lock), `process.json` (this record), `am_process.mp4` (AM's process, recorded).

## The take (Kling 4.0 Edit, one take, measured after generation)

| Measure | Result |
|---|---|
| Lock | plan.json still matches `b5ce4f66…6cfa` |
| Take | 5.04 s at 24 fps; AM's plan covers the first 3.2 s, Kling made up the rest |
| The take's pixels: best repeat (rule: 13 ± 1 frames) | **14 frames: keeps AM's beat** |
| Exact beat from the pixels | 0.456 s inside AM's plan, 0.465 s after it; AM's law 0.433 s (Kling 5–7% slow) |
| Control: AM's guide through the same 24 fps conversion | 13 frames, 0.433 s (the conversion does not shift the beat) |
| AM's eyes and core on the take, on their own | 4 of 6 dancers REPEAT period 14 (0.467 s), kept 132, 133, 135, 136 of 151 frames; 1 SHAPE; 1 CANNOT |
| AM's time on the take | 342.6 s on the Mac (AM's reasoning 307.6 s), 0 tokens |
