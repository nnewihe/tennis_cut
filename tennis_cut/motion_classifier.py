"""Motion classifier — the *confirmation* signal.

The camera is fixed, so frame differencing inside a court ROI is a cheap, strong
activity measure: a rally (fast ball + lunging players) produces far more
inter-frame change than someone walking to the baseline. We sample at a low fps
(the main compute knob), normalise robustly so a rally reads ~1.0 and walking
reads low, and resample onto the master grid.

Frame differencing is used rather than MOG2 background subtraction because here
the *players are always foreground* whether live or dead — it's the amount of
change that discriminates, not the presence of a foreground object. (MOG2 is
easy to swap in; see README.)
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import Config

_EPS = 1e-9


def video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and n > 0:
            return n / fps
        return 0.0
    finally:
        cap.release()


def _roi_slices(roi: Optional[Tuple[int, int, int, int]], scale: float, shape):
    """Translate an original-pixel ROI into slices on the downscaled frame."""
    if roi is None:
        return slice(None), slice(None)
    x, y, w, h = roi
    H, W = shape[:2]
    x0 = max(0, int(x * scale)); y0 = max(0, int(y * scale))
    x1 = min(W, int((x + w) * scale)); y1 = min(H, int((y + h) * scale))
    if x1 <= x0 or y1 <= y0:
        return slice(None), slice(None)
    return slice(y0, y1), slice(x0, x1)


class MotionClassifier:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def score(self, video_path: str, grid_times: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / cfg.motion_fps)))

        times, energy = [], []
        prev = None
        ys = xs = None
        scale = 1.0
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % step == 0:
                if scale == 1.0 and frame.shape[1] > cfg.motion_width:
                    scale = cfg.motion_width / frame.shape[1]
                if scale != 1.0:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if ys is None:
                    ys, xs = _roi_slices(cfg.roi, scale, gray.shape)
                crop = gray[ys, xs]
                if prev is not None:
                    diff = cv2.absdiff(crop, prev)
                    energy.append(float(diff.mean()))
                    times.append(i / fps)
                prev = crop
            i += 1
        cap.release()

        if not energy:
            return np.zeros_like(grid_times, dtype=np.float32)

        energy = np.asarray(energy, dtype=np.float32)
        times = np.asarray(times, dtype=np.float32)

        lo = np.percentile(energy, cfg.motion_lo_pct)
        hi = np.percentile(energy, cfg.motion_hi_pct)
        norm = np.clip((energy - lo) / (hi - lo + _EPS), 0.0, 1.0)

        # Resample onto the master grid (0 outside the measured range).
        return np.interp(grid_times, times, norm, left=0.0, right=0.0).astype(np.float32)
