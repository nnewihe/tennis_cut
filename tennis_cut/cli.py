"""Command-line interface.

    python -m tennis_cut.cli match.mp4 -o cut.mp4
    python -m tennis_cut.cli match.mp4 --dry-run                            # just print segments + ffmpeg cmd
    python -m tennis_cut.cli match.mp4 --select-roi -o cut.mp4             # drag a rectangle
    python -m tennis_cut.cli match.mp4 --roi 120 80 1600 700 -o cut.mp4   # rectangle by coords
    python -m tennis_cut.cli match.mp4 --select-roi-poly -o cut.mp4        # click up to 8 boundary points
    python -m tennis_cut.cli match.mp4 --roi-poly 200 80 900 60 1500 80 1700 500 1700 900 900 950 100 900 60 500 -o cut.mp4
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .cut_video import ffmpeg_cmd_string, render, select_roi, select_roi_poly
from .segmenter import segment_video


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Remove dead time between points in a fixed-camera tennis video.")
    p.add_argument("video", help="Input video path.")
    p.add_argument("-o", "--output", help="Output video path.")
    p.add_argument("--dry-run", action="store_true", help="Print segments + ffmpeg command, don't render.")
    p.add_argument("--segments-json", help="Write detected keep-segments to this JSON file.")

    roi = p.add_mutually_exclusive_group()
    roi.add_argument("--select-roi", action="store_true",
                     help="Interactively drag a rectangular court ROI.")
    roi.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                     help="Rectangular court ROI in original pixels.")
    roi.add_argument("--select-roi-poly", action="store_true",
                     help="Click up to 8 boundary points (right-click=undo, c=close early) "
                          "for a polygon ROI — use a concave shape to exclude a near-camera area.")
    roi.add_argument("--roi-poly", nargs="+", type=int, metavar="X Y ...",
                     help="Polygon ROI: 3-8 boundary points as flat x y pairs, in original pixels.")

    # A few of the most-tuned knobs exposed directly; everything else lives in Config.
    p.add_argument("--motion-fps", type=float, help="Visual sampling rate (main compute knob).")
    p.add_argument("--rally-min-strikes", type=int, help="Min ball strikes to count as a rally.")
    p.add_argument("--lead-in", type=float, help="Seconds kept before each point.")
    p.add_argument("--lead-out", type=float, help="Seconds kept after each point.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg = Config()
    if args.roi_poly:
        if len(args.roi_poly) % 2 != 0 or not (6 <= len(args.roi_poly) <= 16):
            print("--roi-poly needs 3 to 8 points as flat x y pairs (6-16 ints).", file=sys.stderr)
            return 2
        cfg.roi_poly = tuple(zip(args.roi_poly[::2], args.roi_poly[1::2]))  # type: ignore[assignment]
    elif args.select_roi_poly:
        cfg.roi_poly = select_roi_poly(args.video)
    elif args.roi:
        cfg.roi = tuple(args.roi)            # type: ignore[assignment]
    elif args.select_roi:
        cfg.roi = select_roi(args.video)
    for attr in ("motion_fps", "rally_min_strikes", "lead_in", "lead_out"):
        val = getattr(args, attr)
        if val is not None:
            setattr(cfg, attr, val)

    result = segment_video(args.video, cfg)

    print(f"Duration:        {result.duration:7.1f} s")
    print(f"Kept (play):     {result.kept_seconds:7.1f} s  ({len(result.segments)} segments)")
    print(f"Removed (dead):  {result.removed_seconds:7.1f} s  "
          f"({100 * result.removed_seconds / max(result.duration, 1e-9):.0f}%)")

    if args.segments_json:
        with open(args.segments_json, "w") as f:
            json.dump({"segments": result.segments, "duration": result.duration}, f, indent=2)
        print(f"Wrote segments -> {args.segments_json}")

    if args.dry_run or not args.output:
        if result.segments and args.output:
            print("\nffmpeg command:\n" + ffmpeg_cmd_string(args.video, result.segments, args.output))
        elif not args.output:
            print("\n(no --output given; pass -o to render or --dry-run with -o to see the ffmpeg command)")
        return 0

    if not result.segments:
        print("No play detected — nothing to render.", file=sys.stderr)
        return 1

    print(f"\nRendering -> {args.output} ...")
    render(args.video, result.segments, args.output)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
