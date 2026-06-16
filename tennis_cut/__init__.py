"""tennis_cut — automatic dead-time removal for fixed-camera tennis video.

Quick start
-----------
    from tennis_cut import Config, segment_video, render

    result = segment_video("match.mp4")          # detect points
    print(result.segments)                        # [(start, end), ...] to keep
    render("match.mp4", result.segments, "cut.mp4")
"""
from .config import Config
from .segmenter import SegmentResult, segment_video
from .cut_video import render, build_ffmpeg_cmd, ffmpeg_cmd_string, select_roi
from .audio_classifier import AudioRallyClassifier
from .motion_classifier import MotionClassifier

__all__ = [
    "Config", "SegmentResult", "segment_video",
    "render", "build_ffmpeg_cmd", "ffmpeg_cmd_string", "select_roi",
    "AudioRallyClassifier", "MotionClassifier",
]
