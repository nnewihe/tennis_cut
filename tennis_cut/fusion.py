"""Fusion + temporal post-processing.

This is where most of the *perceived* quality comes from. The raw scores are
noisy; smoothing, hysteresis, segment merging and generous padding turn them
into cuts that feel clean.

Design choices baked in:
  * Weighted sum with audio weighted so it alone clears `enter_thresh`, motion
    so it alone does not -> audio drives, motion supports/extends.
  * Asymmetric thresholds (enter > exit) keep us live through brief audio
    dropouts inside a rally.
  * Padding is intentionally generous: clipping the start of a serve is far
    worse than leaving a second of walking in.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.ndimage import median_filter

from .config import Config

Segment = Tuple[float, float]


def fuse(audio: np.ndarray, motion: np.ndarray, cfg: Config) -> np.ndarray:
    """Combine the two per-grid-time scores into one in [0, 1]."""
    combined = cfg.w_audio * audio + cfg.w_motion * motion
    combined = np.clip(combined, 0.0, 1.0)
    k = max(1, int(round(cfg.smooth_win / cfg.grid_hop)))
    if k > 1:
        combined = median_filter(combined, size=k, mode="nearest")
    return combined


def _hysteresis(score: np.ndarray, cfg: Config) -> np.ndarray:
    """Boolean live-mask with separate enter/exit thresholds."""
    live = np.zeros_like(score, dtype=bool)
    state = False
    for i, s in enumerate(score):
        if not state and s >= cfg.enter_thresh:
            state = True
        elif state and s <= cfg.exit_thresh:
            state = False
        live[i] = state
    return live


def _mask_to_segments(mask: np.ndarray, grid: np.ndarray) -> List[Segment]:
    if not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return [(float(grid[s]), float(grid[min(e, len(grid) - 1)])) for s, e in zip(starts, ends)]


def _merge_close(segs: List[Segment], max_gap: float) -> List[Segment]:
    if not segs:
        return []
    out = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = out[-1]
        if s - pe <= max_gap:
            out[-1] = (ps, e)
        else:
            out.append((s, e))
    return out


def segments_from_score(score: np.ndarray, grid: np.ndarray, cfg: Config) -> List[Segment]:
    """Score -> clean live segments (before padding)."""
    mask = _hysteresis(score, cfg)
    segs = _mask_to_segments(mask, grid)
    segs = _merge_close(segs, cfg.merge_gap)              # bridge in-rally pauses
    segs = [(s, e) for s, e in segs if e - s >= cfg.min_rally_len]  # drop blips
    return segs


def apply_padding(segs: List[Segment], cfg: Config, duration: float) -> List[Segment]:
    """Add generous lead-in/out, clip to bounds, merge resulting overlaps."""
    padded = [
        (max(0.0, s - cfg.lead_in), min(duration, e + cfg.lead_out))
        for s, e in segs
    ]
    return _merge_close(padded, 0.0)  # only merge true overlaps/touches


def build_segments(audio: np.ndarray, motion: np.ndarray, grid: np.ndarray,
                   cfg: Config, duration: float) -> Tuple[List[Segment], np.ndarray]:
    """Full fusion path. Returns (padded_live_segments, fused_score)."""
    fused = fuse(audio, motion, cfg)
    segs = segments_from_score(fused, grid, cfg)
    segs = apply_padding(segs, cfg, duration)
    return segs, fused
