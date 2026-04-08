#!/usr/bin/env python3
"""
cursor_tracker.py  –  Scratch-specific cursor locator for 1920×1080 screen recordings.

For each frame, finds the best-matching macOS cursor (pointer / hand / type)
and records its approximate hotspot position.

Usage
-----
    python cursor_tracker.py video.mp4                      # → cursor_track.csv
    python cursor_tracker.py video.mp4 --csv out.csv
    python cursor_tracker.py video.mp4 --preview out.mp4   # annotated debug video
    python cursor_tracker.py video.mp4 --start-sec 5 --end-sec 15

Output CSV columns:  frame, time_sec, cx, cy, score, template, scale
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Template directory (relative to this file)
TEMPLATE_DIR = Path(__file__).parent / "system_cursors copy"

# Cursor templates to use: (filename, hotspot_x, hotspot_y)
# Hotspot is the cursor's active point in the @4x image coordinates.
TEMPLATES = [
    ("pointer@4x.png",  4,  4),
    ("hand@4x.png",    28,  4),
    ("type@4x.png",    13,  0),
]

# Scales to try (fraction of the @4x template size).
# At 1920×1080 Scratch recordings the cursor is typically 0.20–0.34× the @4x size.
SCALES = [0.20, 0.24, 0.28, 0.32, 0.36]

# Downscale factor applied to every video frame before matching (for speed).
FRAME_SCALE = 0.6

# Skip any template variant whose smallest side is below this many pixels.
# Tiny templates (< ~10 px) match Scratch UI noise rather than the real cursor.
MIN_TEMPLATE_DIM = 10

# Minimum match score to report a detection (otherwise cx/cy left blank).
MIN_SCORE = 0.50


# ---------------------------------------------------------------------------
# Template preparation
# ---------------------------------------------------------------------------

@dataclass
class PreparedTemplate:
    gray:      np.ndarray   # grey-filled grayscale patch
    scale:     float        # fraction of @4x native size
    native_w:  int          # @4x template width
    native_h:  int          # @4x template height
    hotspot_x: float        # hotspot x at this scale (full-res pixels)
    hotspot_y: float        # hotspot y at this scale (full-res pixels)
    template_stem: str     # e.g. hand@4x (for CSV / logging)


def _prepare_templates(frame_w: int, frame_h: int) -> list[PreparedTemplate]:
    """Load cursor templates and build all scale variants."""
    sw = round(frame_w * FRAME_SCALE)
    sh = round(frame_h * FRAME_SCALE)

    prepared: list[PreparedTemplate] = []
    for fname, hx4, hy4 in TEMPLATES:
        stem = Path(fname).stem
        img = cv2.imread(str(TEMPLATE_DIR / fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Template not found: {TEMPLATE_DIR / fname}")
        if img.shape[2] != 4:
            raise ValueError(f"Expected RGBA template: {fname}")

        native_h, native_w = img.shape[:2]
        alpha = img[:, :, 3]
        bgr   = img[:, :, :3]
        gray_src = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        alpha_f = alpha.astype(np.float32) / 255.0
        filled  = (gray_src * alpha_f + 128.0 * (1.0 - alpha_f)).astype(np.uint8)

        for scale in SCALES:
            tw_s = max(4, round(native_w * scale * sw / frame_w))
            th_s = max(4, round(native_h * scale * sh / frame_h))
            if min(tw_s, th_s) < MIN_TEMPLATE_DIM:
                continue

            tpl_gray = cv2.resize(filled, (tw_s, th_s), interpolation=cv2.INTER_AREA)

            # hx4/hy4 are in @4x image pixel coordinates.
            # Divide by 4 → logical (1×) display pixels, then multiply by scale
            # → offset in full-res video pixels at this cursor render size.
            hs_x = (hx4 / 4.0) * scale
            hs_y = (hy4 / 4.0) * scale

            prepared.append(PreparedTemplate(
                gray=tpl_gray,
                scale=scale,
                native_w=native_w,
                native_h=native_h,
                hotspot_x=hs_x,
                hotspot_y=hs_y,
                template_stem=stem,
            ))

    return prepared

# ---------------------------------------------------------------------------
# Per-frame detection
# ---------------------------------------------------------------------------

def _detect(frame_gray_small: np.ndarray,
            templates: list[PreparedTemplate],
            frame_w: int, frame_h: int,
            ) -> tuple[float, float, float, str, float] | None:
    """
    Return (cx, cy, score, template_stem, scale) in full-res pixels, or None if
    no match passes MIN_SCORE.
    """
    sw = frame_gray_small.shape[1]
    sh = frame_gray_small.shape[0]

    best_score = -1.0
    best_cx = best_cy = 0.0
    best_stem = ""
    best_scale = 0.0

    for pt in templates:
        th, tw = pt.gray.shape[:2]
        if tw >= sw or th >= sh:
            continue  # template larger than frame (shouldn't happen)

        result = cv2.matchTemplate(frame_gray_small, pt.gray, cv2.TM_CCOEFF_NORMED)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        _, score, _, (px, py) = cv2.minMaxLoc(result)
        score = float(score)
        if score <= best_score:
            continue

        # Convert small-frame top-left → full-res hotspot
        tl_x_full = float(px) * frame_w / sw
        tl_y_full = float(py) * frame_h / sh
        cx = tl_x_full + pt.hotspot_x
        cy = tl_y_full + pt.hotspot_y

        best_score = score
        best_cx, best_cy = cx, cy
        best_stem = pt.template_stem
        best_scale = pt.scale

    if best_score < MIN_SCORE:
        return None
    return best_cx, best_cy, best_score, best_stem, best_scale


# ---------------------------------------------------------------------------
# Main video processing
# ---------------------------------------------------------------------------

def track_video(
    video_path: Path,
    out_csv:    Path,
    out_video:  Path | None = None,
    start_sec:  float | None = None,
    end_sec:    float | None = None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    templates = _prepare_templates(fw, fh)

    if start_sec and start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_sec * fps)))

    writer: cv2.VideoWriter | None = None
    if out_video:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sw = round(fw * FRAME_SCALE)
    sh = round(fh * FRAME_SCALE)

    rows: list[dict] = []
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    while True:
        if end_sec is not None and fps > 0 and frame_idx / fps > end_sec + 1e-6:
            break

        ok, bgr = cap.read()
        if not ok:
            break

        tsec = frame_idx / fps if fps > 0 else 0.0
        gray_small = cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (sw, sh),
            interpolation=cv2.INTER_AREA,
        )

        det = _detect(gray_small, templates, fw, fh)

        if det is not None:
            cx, cy, score, tmpl, sc = det
            rows.append(
                {
                    "frame": frame_idx,
                    "time_sec": f"{tsec:.6f}",
                    "cx": f"{cx:.2f}",
                    "cy": f"{cy:.2f}",
                    "score": f"{score:.4f}",
                    "template": tmpl,
                    "scale": f"{sc:.4f}",
                }
            )
        else:
            rows.append(
                {
                    "frame": frame_idx,
                    "time_sec": f"{tsec:.6f}",
                    "cx": "",
                    "cy": "",
                    "score": "",
                    "template": "",
                    "scale": "",
                }
            )

        if writer is not None:
            if det is not None:
                cx, cy, score, tmpl, sc = det
                cv2.drawMarker(
                    bgr,
                    (int(round(cx)), int(round(cy))),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    30,
                    2,
                )
                label = f"{tmpl} {score:.3f} sc={sc:.2f}"
                cv2.putText(
                    bgr,
                    label,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(bgr)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    fields = ["frame", "time_sec", "cx", "cy", "score", "template", "scale"]
    with open(out_csv, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"Written → {out_csv}  ({len(rows)} frames)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Locate macOS cursor in Scratch screen recordings.")
    ap.add_argument("video",        type=Path, help="Input MP4 path")
    ap.add_argument("--csv",        type=Path, default=None)
    ap.add_argument("--preview",    type=Path, default=None, help="Annotated output video")
    ap.add_argument("--start-sec",  type=float, default=None)
    ap.add_argument("--end-sec",    type=float, default=None)
    args = ap.parse_args()

    out_csv = args.csv or args.video.with_suffix(".csv")
    track_video(args.video, out_csv, args.preview, args.start_sec, args.end_sec)


if __name__ == "__main__":
    main()
