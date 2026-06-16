"""Top-level orchestrator: video path -> list of live segments to keep."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .audio_classifier import AudioRallyClassifier
from .config import Config
from .fusion import build_segments
from .motion_classifier import MotionClassifier, video_duration

Segment = Tuple[float, float]


@dataclass
class SegmentResult:
    segments: List[Segment]          # padded live spans to KEEP (seconds)
    duration: float
    grid: np.ndarray = field(repr=False)
    audio_score: np.ndarray = field(repr=False)
    motion_score: np.ndarray = field(repr=False)
    fused_score: np.ndarray = field(repr=False)
    onsets: np.ndarray = field(repr=False)

    @property
    def kept_seconds(self) -> float:
        return float(sum(e - s for s, e in self.segments))

    @property
    def removed_seconds(self) -> float:
        return self.duration - self.kept_seconds


def segment_video(video_path: str, cfg: Config | None = None) -> SegmentResult:
    cfg = cfg or Config()

    duration = video_duration(video_path)
    if duration <= 0:
        raise RuntimeError(f"Could not read duration from {video_path!r}.")
    grid = np.arange(0.0, duration, cfg.grid_hop)

    audio_clf = AudioRallyClassifier(cfg)
    motion_clf = MotionClassifier(cfg)

    audio_score = audio_clf.score(video_path, grid)
    motion_score = motion_clf.score(video_path, grid)

    segs, fused = build_segments(audio_score, motion_score, grid, cfg, duration)

    return SegmentResult(
        segments=segs, duration=duration, grid=grid,
        audio_score=audio_score, motion_score=motion_score,
        fused_score=fused, onsets=audio_clf.last_onsets,
    )
