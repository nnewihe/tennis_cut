"""Rendering (ffmpeg) and an interactive ROI picker.

Cutting strategy: a single-pass filter_complex that trims each kept segment and
concatenates them, re-encoding so cuts are frame-accurate (stream-copy would
snap to keyframes and is imprecise for short points). For very long matches with
hundreds of points, see the per-segment + concat-demuxer note in the README.
"""
from __future__ import annotations

import shlex
import subprocess
from typing import List, Optional, Tuple

Segment = Tuple[float, float]


def build_ffmpeg_cmd(video_path: str, segments: List[Segment], out_path: str) -> List[str]:
    if not segments:
        raise ValueError("No segments to render.")
    parts, labels = [], []
    for i, (s, e) in enumerate(segments):
        parts.append(
            f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )
        labels.append(f"[v{i}][a{i}]")
    concat = "".join(labels) + f"concat=n={len(segments)}:v=1:a=1[outv][outa]"
    filter_complex = ";".join(parts) + ";" + concat
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac",
        out_path,
    ]


def render(video_path: str, segments: List[Segment], out_path: str) -> None:
    subprocess.run(build_ffmpeg_cmd(video_path, segments, out_path), check=True)


def ffmpeg_cmd_string(video_path: str, segments: List[Segment], out_path: str) -> str:
    return " ".join(shlex.quote(a) for a in build_ffmpeg_cmd(video_path, segments, out_path))


def select_roi(video_path: str) -> Optional[Tuple[int, int, int, int]]:
    """Open the first frame and let the user drag a court ROI. Needs a display."""
    import cv2  # local import so headless installs don't choke on highgui

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame for ROI selection.")
    r = cv2.selectROI("Drag court ROI, then ENTER", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    x, y, w, h = (int(v) for v in r)
    return None if w == 0 or h == 0 else (x, y, w, h)
