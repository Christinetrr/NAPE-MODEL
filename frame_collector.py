#!/usr/bin/env python3
"""
Extract a single frame from a video at a given time (seconds) and save as PNG under ./frames/.

Example (project videos live in ./videos/):

    python frame_collector.py videos/vid1.mp4 12.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path, help="Input video file")
    p.add_argument(
        "time_sec",
        type=float,
        help="Time in seconds (e.g. 12.5)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("frames"),
        help='Output directory (default: "frames")',
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output PNG filename (default: derived from time and video stem)",
    )
    args = p.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    frame_idx = int(round(args.time_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise SystemExit(
            f"Could not read frame at ~{args.time_sec}s (frame index {frame_idx}). "
            "Try a slightly smaller time or check video duration."
        )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.name:
        out_name = args.name if args.name.lower().endswith(".png") else f"{args.name}.png"
    else:
        safe_t = str(args.time_sec).replace(".", "p")
        out_name = f"{video.stem}_t{safe_t}s.png"

    out_path = out_dir / out_name
    if not cv2.imwrite(str(out_path), frame):
        raise SystemExit(f"Failed to write: {out_path}")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
