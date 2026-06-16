# tennis_cut

Automatically removes the dead time between points in a **fixed-camera, behind-the-baseline** recreational tennis video.

It's deliberately lightweight: **no trained model, no GPU.** Audio ball-strike detection is the primary signal; cheap frame-difference motion in a court ROI confirms and extends it; a hysteresis state machine plus generous padding turns the noisy scores into clean cuts.

```
video ─┬─ ffmpeg → wav ─ spectral-flux onsets ─ rally clustering ─→ audio score ─┐
       │                                                                          ├─ fuse → hysteresis → merge → pad → segments → ffmpeg cut
       └─ frames @ ~5fps ─ frame-diff energy in ROI ─ robust-normalise ─→ motion score ─┘
```

## Why this design

A fixed camera means the on-screen *view never changes*, so "is this the live shot?" classification (the usual broadcast trick) is useless here — every frame is the court. What discriminates live from dead is **what you hear** (the rhythmic ball "pock" during a rally) and **how much is moving** (a rally produces far more inter-frame change than someone walking to the baseline). Audio leads because recreational footage has no commentary or music to pollute the onset signal; motion supports it for windy/quiet stretches.

## Install

```bash
pip install -r requirements.txt      # numpy, scipy, opencv-python
# ffmpeg must also be installed and on your PATH
```

## Usage

CLI:

```bash
# Detect and render the cut
python -m tennis_cut.cli match.mp4 -o cut.mp4

# Just see what it would do (segments + the ffmpeg command), no render
python -m tennis_cut.cli match.mp4 --dry-run -o cut.mp4

# Restrict motion analysis to the court (strongly recommended if another court is in frame)
python -m tennis_cut.cli match.mp4 --select-roi -o cut.mp4          # drag a box on frame 1
python -m tennis_cut.cli match.mp4 --roi 120 80 1600 700 -o cut.mp4 # or pass it directly

# Dump segments to JSON for use in your own editor
python -m tennis_cut.cli match.mp4 --segments-json points.json
```

Library:

```python
from tennis_cut import Config, segment_video, render

cfg = Config(roi=(120, 80, 1600, 700))     # x, y, w, h in original pixels
result = segment_video("match.mp4", cfg)

print(f"{result.removed_seconds:.0f}s of dead time removed")
for start, end in result.segments:          # spans to KEEP
    print(f"  point: {start:7.1f} → {end:7.1f}")

render("match.mp4", result.segments, "cut.mp4")
```

`result` also exposes `audio_score`, `motion_score`, `fused_score`, `grid`, and `onsets` — plot these against time to debug or tune.

## Tuning knobs (all in `tennis_cut/config.py`)

These are the accuracy ⇄ compute and accuracy ⇄ "feel" dials.

| Knob | Default | Effect |
|---|---|---|
| `motion_fps` | 5.0 | **Main compute lever.** Lower = faster, coarser boundaries. Audio is dense and cheap regardless. |
| `motion_width` | 480 | Frame downscale width. Lower = faster. |
| `rally_min_strikes` | 3 | Higher = fewer false points (ignores a couple of stray bounces); too high drops short points. |
| `rally_max_gap` | 2.5 s | Max silence between strikes still counted as one rally. |
| `onset_delta` | 0.06 | Onset sensitivity. Raise if a noisy court invents strikes; lower if quiet audio misses them. |
| `enter_thresh` / `exit_thresh` | 0.50 / 0.30 | Hysteresis. `enter > exit` keeps you live through brief audio dropouts. |
| `merge_gap` | 1.2 s | Bridges short pauses inside a rally so one point isn't split in two. |
| `min_rally_len` | 1.0 s | Drops sub-second live blips. |
| `lead_in` / `lead_out` | 1.5 / 1.2 s | **Generous on purpose.** Clipping a serve feels far worse than leaving a second of walking. Raise if starts feel clipped. |
| `roi` | None | Court box. Set it if an adjacent court is visible — biggest single accuracy win in that case. |

## Tuning recipe

1. Run with `--dry-run` and check the kept/removed split and segment count against reality.
2. **Too many false points?** raise `rally_min_strikes` or `onset_delta`.
3. **Missing real points?** lower `onset_delta`; check the ROI isn't excluding play.
4. **One point split in two?** raise `merge_gap`.
5. **Serves/returns clipped?** raise `lead_in`/`lead_out`.
6. **Too slow?** lower `motion_fps` to 2–3 (audio quality is unaffected).

## Known limitations

- Built for a **fixed** camera. A moving/handheld camera breaks the frame-diff assumption — you'd add background-motion compensation or fall back to audio-only.
- Can't distinguish a *practice* rally from a *scored* point — by design it keeps any sustained rally and cuts the standing-around, which is what you want for a watchable cut.
- A loud adjacent court can inject false onsets; the ROI mask only gates the *visual* signal. If audio bleed is severe, raise `rally_min_strikes`/`onset_delta`, or gate audio onsets by simultaneous in-ROI motion (a natural extension point in `fusion.fuse`).
- For matches with hundreds of points, the single-pass `filter_complex` cut works but the command gets long; switch to per-segment extraction + the ffmpeg `concat` demuxer if you hit command-length limits.

## Swapping in background subtraction

`MotionClassifier` uses frame differencing because players are always foreground here, so *amount of change* discriminates better than *presence of foreground*. To try MOG2 instead, replace the `absdiff` step with `cv2.createBackgroundSubtractorMOG2()` and measure the foreground ratio — the rest of the pipeline is unchanged.

## Test

```bash
python -m tests.selftest
```

Validates onset detection + rally clustering, the fusion state machine, and a full synthetic end-to-end run (motion + click track → detection → ffmpeg cut) — no real footage required.
