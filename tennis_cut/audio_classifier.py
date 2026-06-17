"""Audio classifier — the *primary* signal.

Pipeline:  video --ffmpeg--> mono wav
           wav   --STFT--->   spectral-flux onset envelope
           env   --peaks-->   ball-strike onset times
           onsets --cluster-> rally bursts (>= N strikes, small gaps)
           bursts --> per-grid-time "live" score in {0, 1}

Rationale: on recreational footage there's no commentary/music, so the ball
"pock" is a clean, near-periodic transient train during a point and absent
between points. This runs many times faster than real time on CPU.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import List, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter
from scipy.signal import stft

from .config import Config

_EPS = 1e-9


def extract_audio(video_path: str, sr: int) -> np.ndarray:
    """Decode `video_path` to a mono float32 waveform in [-1, 1] via ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", video_path,
            "-ac", "1", "-ar", str(sr), "-f", "wav", wav_path,
        ]
        subprocess.run(cmd, check=True)
        rate, data = wavfile.read(wav_path)

    if data.ndim > 1:                       # safety: collapse to mono
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    # Normalise integer PCM to [-1, 1].
    if np.issubdtype(np.dtype(data.dtype), np.integer) or data.max() > 1.5:
        peak = np.max(np.abs(data)) + _EPS
        data = data / peak
    return data


def onset_envelope(y: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Band-limited spectral-flux onset envelope. Returns (times, env).

    Summing flux over the full spectrum lets low-frequency rumble (passing
    trains, traffic, wind) dominate the envelope and bury the ball-strike
    transient. Restricting to `onset_band_lo_hz`-`onset_band_hi_hz` keeps the
    strike's broadband click, which is concentrated in the mid/high range,
    while excluding the band rumble lives in.
    """
    noverlap = cfg.n_fft - cfg.audio_hop
    f, t, Z = stft(
        y, fs=cfg.audio_sr, nperseg=cfg.n_fft, noverlap=noverlap,
        boundary=None, padded=False,
    )
    band = (f >= cfg.onset_band_lo_hz) & (f <= cfg.onset_band_hi_hz)
    mag = np.abs(Z[band])
    # Positive first-difference of magnitude, summed over frequency = spectral flux.
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    return t[1:], flux


def detect_onsets(times: np.ndarray, env: np.ndarray, cfg: Config) -> np.ndarray:
    """Adaptive peak-pick the envelope into ball-strike onset times."""
    if env.size == 0:
        return np.empty(0)
    env = env / (env.max() + _EPS)

    frame_dt = float(np.median(np.diff(times))) if times.size > 1 else cfg.grid_hop
    win = max(1, int(round(cfg.onset_mean_win / frame_dt))) | 1  # force odd for median_filter
    # Median (not mean) baseline: a burst of noise (train clatter, a passing
    # car) pulls a mean filter way up, raising the threshold and burying real
    # strikes; the median shrugs off short bursts and tracks the steady floor.
    local_mean = median_filter(env, size=win, mode="nearest")
    thresh = local_mean + cfg.onset_delta

    # Local maxima strictly above the adaptive threshold.
    is_peak = np.zeros_like(env, dtype=bool)
    is_peak[1:-1] = (
        (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > thresh[1:-1])
    )
    idx = np.flatnonzero(is_peak)
    if idx.size == 0:
        return np.empty(0)

    # Enforce minimum separation, keeping the stronger of competing peaks.
    keep: List[int] = []
    last_t = -np.inf
    for i in sorted(idx, key=lambda j: -env[j]):  # strongest first
        keep.append(i)
    keep.sort()
    out, last_t = [], -np.inf
    for i in keep:
        if times[i] - last_t >= cfg.min_onset_sep:
            out.append(times[i])
            last_t = times[i]
    return np.asarray(out)


def cluster_rallies(onsets: np.ndarray, cfg: Config) -> List[Tuple[float, float]]:
    """Group onsets into rally bursts: runs with small inter-onset gaps."""
    if onsets.size == 0:
        return []
    bursts: List[Tuple[float, float]] = []
    run = [float(onsets[0])]
    for o in onsets[1:]:
        if o - run[-1] <= cfg.rally_max_gap:
            run.append(float(o))
        else:
            if len(run) >= cfg.rally_min_strikes:
                bursts.append((run[0], run[-1]))
            run = [float(o)]
    if len(run) >= cfg.rally_min_strikes:
        bursts.append((run[0], run[-1]))
    return bursts


class AudioRallyClassifier:
    """Bundles the audio pipeline; `.score()` returns a value per grid time."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.last_onsets: np.ndarray = np.empty(0)
        self.last_bursts: List[Tuple[float, float]] = []

    def score(self, video_path: str, grid_times: np.ndarray) -> np.ndarray:
        y = extract_audio(video_path, self.cfg.audio_sr)
        times, env = onset_envelope(y, self.cfg)
        onsets = detect_onsets(times, env, self.cfg)
        bursts = cluster_rallies(onsets, self.cfg)

        self.last_onsets, self.last_bursts = onsets, bursts

        out = np.zeros_like(grid_times, dtype=np.float32)
        for s, e in bursts:
            out[(grid_times >= s) & (grid_times <= e)] = 1.0
        return out
