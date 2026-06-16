"""Self-test. Validates the algorithms on synthetic data — no real footage needed.

Run:  python -m tests.selftest    (from the package root)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennis_cut.audio_classifier import (cluster_rallies, detect_onsets, onset_envelope)
from tennis_cut.config import Config
from tennis_cut.fusion import build_segments, segments_from_score
from tennis_cut.segmenter import segment_video

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, cond):
    _results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


# --------------------------------------------------------------------------
# 1. Audio: synthetic click trains -> onsets -> rally clustering
# --------------------------------------------------------------------------
def test_audio():
    print("Audio onset + rally clustering")
    cfg = Config()
    sr = cfg.audio_sr
    dur = 16.0
    y = np.random.randn(int(sr * dur)).astype(np.float32) * 0.002  # quiet noise floor

    def click(t):
        i = int(t * sr)
        n = int(0.01 * sr)
        burst = np.random.randn(n).astype(np.float32) * np.hanning(n)
        y[i:i + n] += burst * 3.0

    rally1 = [1.0, 1.6, 2.3, 2.9, 3.5]      # 5 strikes, gaps < max -> one rally
    rally2 = [10.0, 10.7, 11.5, 12.1, 12.8] # second rally
    for t in rally1 + rally2:
        click(t)

    times, env = onset_envelope(y, cfg)
    onsets = detect_onsets(times, env, cfg)
    bursts = cluster_rallies(onsets, cfg)

    check(f"detected >= 8 onsets (got {len(onsets)})", len(onsets) >= 8)
    check(f"found exactly 2 rallies (got {len(bursts)})", len(bursts) == 2)
    if len(bursts) == 2:
        (s1, e1), (s2, e2) = bursts
        check("rally 1 near 1.0–3.5s", abs(s1 - 1.0) < 0.4 and abs(e1 - 3.5) < 0.4)
        check("rally 2 near 10.0–12.8s", abs(s2 - 10.0) < 0.4 and abs(e2 - 12.8) < 0.4)


# --------------------------------------------------------------------------
# 2. Fusion state machine: merge dropouts, drop blips, pad
# --------------------------------------------------------------------------
def test_fusion():
    print("Fusion state machine (hysteresis / merge / min-len / pad)")
    cfg = Config(lead_in=0.0, lead_out=0.0)  # isolate the machine from padding here
    grid = np.arange(0, 20, cfg.grid_hop)
    score = np.zeros_like(grid)

    score[(grid >= 2) & (grid < 5)] = 0.8          # rally A
    score[(grid >= 3.4) & (grid < 3.7)] = 0.1      # brief dropout inside A (< merge_gap)
    score[(grid >= 8) & (grid < 8.3)] = 0.8        # blip (< min_rally_len) -> dropped
    score[(grid >= 12) & (grid < 16)] = 0.8        # rally B

    segs = segments_from_score(score, grid, cfg)
    check(f"two rallies survive, blip dropped (got {len(segs)})", len(segs) == 2)
    if len(segs) == 2:
        check("rally A bridged across dropout", segs[0][0] < 2.2 and segs[0][1] > 4.8)
        check("rally B intact", abs(segs[1][0] - 12) < 0.2 and abs(segs[1][1] - 16) < 0.2)

    # Padding extends spans generously.
    cfg2 = Config(lead_in=1.5, lead_out=1.0)
    segs2, _ = build_segments(np.where(score > 0.5, 1.0, 0.0),
                              np.zeros_like(score), grid, cfg2, duration=20.0)
    if segs2:
        check("padding applied (rally A starts earlier)", segs2[0][0] <= 0.6)


# --------------------------------------------------------------------------
# 3. End-to-end: synthesize video (motion + clicks) and run the full pipeline
# --------------------------------------------------------------------------
def _synthesize_video(path, cfg):
    fps, W, H, dur = 10, 320, 180, 30.0
    # Realistic spacing: long dead time between points, as on a real court.
    rallies = [(4.0, 8.0), (20.0, 24.0)]  # ground-truth play windows

    tmp_silent = path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(tmp_silent, fourcc, fps, (W, H))
    n_frames = int(dur * fps)
    rng = np.random.default_rng(0)
    for f in range(n_frames):
        t = f / fps
        frame = np.zeros((H, W, 3), np.uint8)
        live = any(s <= t < e for s, e in rallies)
        if live:                              # fast jumping box = high frame-diff
            x = int(rng.integers(0, W - 40)); y = int(rng.integers(0, H - 40))
            frame[y:y + 40, x:x + 40] = 255
        else:                                  # slow drift = low frame-diff ("walking")
            x = int((t * 6) % (W - 40))
            frame[70:110, x:x + 40] = 90
        vw.write(frame)
    vw.release()

    # Click track aligned to the rally windows.
    sr = cfg.audio_sr
    y = (rng.standard_normal(int(sr * dur)).astype(np.float32) * 0.002)
    for s, e in rallies:
        t = s + 0.2
        while t < e - 0.2:
            i = int(t * sr); nlen = int(0.01 * sr)
            y[i:i + nlen] += rng.standard_normal(nlen).astype(np.float32) * np.hanning(nlen) * 3.0
            t += 0.6
    wav_path = path + ".wav"
    wavfile.write(wav_path, sr, (y / (np.abs(y).max() + 1e-9) * 0.9 * 32767).astype(np.int16))

    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-i", tmp_silent, "-i", wav_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", path], check=True)
    os.remove(tmp_silent); os.remove(wav_path)
    return rallies


def test_end_to_end():
    print("End-to-end pipeline on synthetic video")
    cfg = Config()
    with tempfile.TemporaryDirectory() as d:
        vid = os.path.join(d, "synthetic.mp4")
        rallies = _synthesize_video(vid, cfg)

        result = segment_video(vid, cfg)
        check(f"~30s duration (got {result.duration:.1f})", abs(result.duration - 30) < 1.0)
        check(f"detected 2 play segments (got {len(result.segments)})", len(result.segments) == 2)

        # Each detected segment should overlap a ground-truth rally window.
        def overlaps(seg, gts):
            return any(min(seg[1], e) - max(seg[0], s) > 0 for s, e in gts)
        if result.segments:
            check("all detected segments overlap a real rally",
                  all(overlaps(s, rallies) for s in result.segments))
            check("removed > 30% of footage", result.removed_seconds > 0.3 * result.duration)

        # Actually render the cut to prove the ffmpeg path works.
        out = os.path.join(d, "cut.mp4")
        from tennis_cut.cut_video import render
        render(vid, result.segments, out)
        check("rendered cut.mp4 exists and is non-empty", os.path.getsize(out) > 0)


if __name__ == "__main__":
    test_audio()
    test_fusion()
    test_end_to_end()
    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    sys.exit(0 if all(_results) else 1)
