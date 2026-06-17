"""Central configuration for the tennis dead-time remover.

Every tunable lives here so the accuracy/compute trade-offs we discussed are in
one place. Defaults are tuned for outdoor clay courts with adjacent courts in
frame (tighter audio thresholds, reduced padding, lower motion weight).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Config:
    # ---- master timeline -------------------------------------------------
    grid_hop: float = 0.1            # s. Resolution at which scores live & cuts snap.

    # ---- audio: extraction ----------------------------------------------
    audio_sr: int = 22_050          # Hz. Plenty for ball-strike transients.
    n_fft: int = 1024               # STFT window for the onset envelope.
    audio_hop: int = 256            # STFT hop -> ~86 envelope frames/sec.

    # ---- audio: onset (ball-strike) detection ---------------------------
    onset_mean_win: float = 0.25    # s. Window for the adaptive (median) threshold baseline.
    onset_delta: float = 0.11       # Height above local median to count as an onset.
    min_onset_sep: float = 0.10     # s. Two strikes can't be closer than this.
    onset_band_lo_hz: float = 1000.0  # Hz. Exclude rumble (trains, traffic, wind) below this.
    onset_band_hi_hz: float = 9000.0  # Hz. Upper edge of the ball-strike flux band.

    # ---- audio: rally clustering ----------------------------------------
    rally_max_gap: float = 1.8      # s. Max silence between strikes inside one rally.
    rally_min_strikes: int = 2      # A "rally" needs at least this many strikes.

    # ---- motion classifier ----------------------------------------------
    motion_fps: float = 5.0         # Sample rate for the visual pass (the main compute knob).
    motion_width: int = 480         # Downscale width before differencing.
    motion_lo_pct: float = 10.0     # Robust-normalisation low percentile.
    motion_hi_pct: float = 90.0     # Robust-normalisation high percentile (~rally level).
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in *original* px, or None=full frame.
    roi_poly: Optional[Tuple[Tuple[int, int], ...]] = None  # N (x,y) points in original px, clockwise, defining the active-play region (e.g. an 8-point shape that excludes a near-camera conversation area). Overrides roi.

    # ---- fusion ----------------------------------------------------------
    w_audio: float = 0.70           # Weight so audio alone clears `enter_thresh`.
    w_motion: float = 0.22          # Reduced further so non-ball motion can't extend live state.
    smooth_win: float = 0.5         # s. Median filter on the fused score.

    # ---- state machine (hysteresis = tighter for clay/adjacent courts) ---
    enter_thresh: float = 0.60      # Go live above this...
    exit_thresh: float = 0.40       # ...stay live until below this (enter > exit).
    merge_gap: float = 4.0          # s. Bridge live spans separated by less than this.
    min_rally_len: float = 1.0      # s. Drop live blips shorter than this.

    # ---- padding ---------------------------------------------------------
    lead_in: float = 1.0            # s. Keep before each point starts.
    lead_out: float = 0.8           # s. Keep after the last strike.

    def __post_init__(self) -> None:
        if self.enter_thresh <= self.exit_thresh:
            raise ValueError("enter_thresh must be > exit_thresh (hysteresis).")
        if self.grid_hop <= 0:
            raise ValueError("grid_hop must be positive.")
        if self.roi is not None and self.roi_poly is not None:
            raise ValueError("Specify roi or roi_poly, not both.")
