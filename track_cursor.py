#!/usr/bin/env python3
"""
Scratch-specific cursor locator for 1920×1080 screen recordings.

For each frame, finds the best-matching macOS cursor (pointer / hand / type).
**Pointer** positions feed a **dwell-based** stable hotspot: the cursor must sit
within a small pixel neighborhood for at least a minimum **time** before that
spot is **committed**; `cx`/`cy` use the last committed position so brief false
matches do not move the track. **Hand** matches lock `cx`/`cy` to the last committed stable hotspot at the
moment of hand-cursor transition; the position stays frozen until the cursor
returns to pointer mode (so noisy hand↔pointer flicker cannot shift the output).
**Type** rows use the detected position.

Usage
-----
    python track_cursor.py video.mp4                      # → cursor_track.csv
    python track_cursor.py video.mp4 --csv out.csv
    python track_cursor.py video.mp4 --preview out.mp4   # annotated debug video
    python track_cursor.py video.mp4 --start-sec 5 --end-sec 15

Output CSV columns:  frame, time_sec, cx, cy, score, template, scale

Use :func:`click_happened_in_segment` for a coarse **bool** over a time window:
near-cursor pixel change vs. the previous frame (see docstring).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Configuration

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
# Tiny templates match Scratch UI noise rather than the real cursor.
MIN_TEMPLATE_DIM = 10

# Ignore matches whose top-left is within this many full-res pixels of the top
# of the frame. The macOS browser chrome (traffic lights, tab bar) lives here
# and consistently produces false positives with the small hand template.
TOP_MARGIN_PX = 160

# Ignore matches whose top-left is within this many full-res pixels of the left
# of the frame. The Scratch block palette lives here and its rounded pill inputs
# produce false positives with the hand template.
LEFT_MARGIN_PX = 200

# Minimum match score to report a detection (otherwise cx/cy left blank).
MIN_SCORE = 0.58

# Dwell-based pointer stability (full-res pixels, seconds): a reading must stay
# within ``STABLE_CLUSTER_JITTER_PX`` of the running cluster center for at least
# ``MIN_STABLE_DWELL_SEC`` before it is **committed** as the stable hotspot.
# Output uses the last committed position until a new cluster earns dwell time
# (ignoring one-frame or short spikes).
STABLE_CLUSTER_JITTER_PX = 18.0
MIN_STABLE_DWELL_SEC = 0.30  # ~9 frames @ 30 fps before a spot can commit
STABLE_RUN_EMA_ALPHA = 0.15  # slower cluster drift — outliers pull the center less

# Minimum score required for a detection to BREAK the current pointer cluster and
# start a new one at a far-away position.  Detections within jitter distance are
# always accepted regardless of score (smooth cursor motion).  Large jumps to
# low-score UI false-positives (e.g. Scratch chrome that resembles the pointer
# glyph) are suppressed; the tracker keeps the current cluster instead.
SCORE_MIN_FOR_RUN_BREAK = 0.67

# Heuristic UI “click” detection: compare patches under the cursor between frames.
# Large outer radius catches nearby labels / button chrome; keep inner small so
# under-cursor UI (pressed state, recolor) still contributes — a big inner disk
# zeros out exactly where those pixels often land.
CLICK_ROI_HALF_PX = 120  # full-res; patch is (2*half) × (2*half)
CLICK_INNER_IGNORE_PX = 12  # disk around hotspot de-weighted (cursor glyph only)
CLICK_DIFF_MEAN_THRESHOLD = 11.0  # mean absdiff in 0–255 gray; raise if noisier video


# ---------------------------------------------------------------------------
# Template preparation
# ---------------------------------------------------------------------------

@dataclass
class PreparedTemplate:
    gray:      np.ndarray   # grey-filled grayscale patch
    grad_x:    np.ndarray   # Sobel X of the grey-filled patch
    grad_y:    np.ndarray   # Sobel Y of the grey-filled patch
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
            tpl_f    = tpl_gray.astype(np.float32)

            # hx4/hy4 are in @4x pixel coordinates.
            # Multiply by scale to get the hotspot offset in full-res video pixels.
            hs_x = hx4 * scale
            hs_y = hy4 * scale

            prepared.append(PreparedTemplate(
                gray=tpl_gray,
                grad_x=cv2.Sobel(tpl_f, cv2.CV_32F, 1, 0, ksize=3),
                grad_y=cv2.Sobel(tpl_f, cv2.CV_32F, 0, 1, ksize=3),
                scale=scale,
                native_w=native_w,
                native_h=native_h,
                hotspot_x=hs_x,
                hotspot_y=hs_y,
                template_stem=stem,
            ))

    return prepared

# Per-frame detection

def _detect(frame_gray_small: np.ndarray,
            templates: list[PreparedTemplate],
            frame_w: int, frame_h: int,
            ) -> tuple[float, float, float, str, float] | None:
    """
    Return (cx, cy, score, template_stem, scale) in full-res pixels, or None if
    no match passes MIN_SCORE.

    Combines appearance (TM_CCOEFF_NORMED) with Sobel X/Y gradient correlation
    to discriminate real cursors (sharp aliased edges) from Scratch sprites
    (soft anti-aliased rendering) that look similar at small scales.
    """
    sw = frame_gray_small.shape[1]
    sh = frame_gray_small.shape[0]
    top_margin_s  = round(TOP_MARGIN_PX  * sh / frame_h)
    left_margin_s = round(LEFT_MARGIN_PX * sw / frame_w)

    # Precompute frame gradients once (reused across all templates).
    frame_f = frame_gray_small.astype(np.float32)
    frame_gx = cv2.Sobel(frame_f, cv2.CV_32F, 1, 0, ksize=3)
    frame_gy = cv2.Sobel(frame_f, cv2.CV_32F, 0, 1, ksize=3)

    best_score = -1.0
    best_cx = best_cy = 0.0
    best_stem = ""
    best_scale = 0.0

    for pt in templates:
        th, tw = pt.gray.shape[:2]
        if tw >= sw or th >= sh:
            continue

        app = cv2.matchTemplate(frame_gray_small, pt.gray,   cv2.TM_CCOEFF_NORMED)
        gx  = cv2.matchTemplate(frame_gx,         pt.grad_x, cv2.TM_CCOEFF_NORMED)
        gy  = cv2.matchTemplate(frame_gy,         pt.grad_y, cv2.TM_CCOEFF_NORMED)
        result = 0.5 * app + 0.25 * gx + 0.25 * gy
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        # Zero out the top margin (macOS browser chrome / menu bar).
        if top_margin_s > 0 and top_margin_s < result.shape[0]:
            result[:top_margin_s, :] = 0.0
        # Zero out the left margin (Scratch block palette).
        if left_margin_s > 0 and left_margin_s < result.shape[1]:
            result[:, :left_margin_s] = 0.0

        _, score, _, (px, py) = cv2.minMaxLoc(result)
        score = float(score)
        if score <= best_score:
            continue

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


@dataclass
class StableCursorState:
    """Pointer/hotspot state: ``stable_xy`` is last dwell-qualified position.
    ``hand_locked_xy`` is frozen at the moment the cursor first transitions to
    hand so that noisy hand/pointer back-and-forth cannot shift the output."""

    stable_xy: tuple[float, float] | None = None
    run_xy: tuple[float, float] | None = None
    run_start_sec: float | None = None
    hand_locked_xy: tuple[float, float] | None = None


def _stable_cursor_pointer_cluster_update(
    state: StableCursorState,
    cx: float,
    cy: float,
    t_sec: float,
    score: float,
    *,
    jitter_px: float = STABLE_CLUSTER_JITTER_PX,
    min_dwell_sec: float = MIN_STABLE_DWELL_SEC,
) -> None:
    """Absorb a **pointer** detection into the running cluster / commit stable.

    Large jumps (> ``jitter_px``) are only accepted when ``score`` meets
    ``SCORE_MIN_FOR_RUN_BREAK``; low-confidence detections at a distant position
    (e.g. persistent Scratch-UI false positives) are silently ignored so the
    current cluster is preserved.  Detections within jitter distance are always
    accepted regardless of score.
    """
    if state.run_xy is None:
        state.run_xy = (cx, cy)
        state.run_start_sec = t_sec
    else:
        rx, ry = state.run_xy
        if math.hypot(cx - rx, cy - ry) <= jitter_px:
            a = STABLE_RUN_EMA_ALPHA
            state.run_xy = ((1.0 - a) * rx + a * cx, (1.0 - a) * ry + a * cy)
        else:
            # Large jump: suppress low-confidence detections that are likely
            # false positives (e.g. static UI chrome that resembles the cursor).
            if score < SCORE_MIN_FOR_RUN_BREAK:
                return
            if (
                state.run_start_sec is not None
                and (t_sec - state.run_start_sec) >= min_dwell_sec
            ):
                state.stable_xy = state.run_xy
            state.run_xy = (cx, cy)
            state.run_start_sec = t_sec
            return

    if (
        state.run_start_sec is not None
        and (t_sec - state.run_start_sec) >= min_dwell_sec
    ):
        state.stable_xy = state.run_xy


def _stable_cursor_track_step(
    state: StableCursorState,
    cx: float,
    cy: float,
    score: float,
    tmpl: str,
    t_sec: float,
) -> tuple[float, float]:
    """
    Update dwell-based stable pointer (for ``pointer`` only) and return ``(out_x,
    out_y)`` for CSV / preview / ROI. ``hand`` locks to the last committed stable
    position at the moment of hand-cursor transition and stays frozen until the
    cursor returns to pointer mode (prevents noisy hand/pointer flicker from
    shifting the output). ``type`` uses raw ``(cx, cy)``.
    """
    if tmpl.startswith("pointer"):
        state.hand_locked_xy = None  # returning to pointer; release hand lock
        _stable_cursor_pointer_cluster_update(state, cx, cy, t_sec, score)
        if state.stable_xy is not None:
            return state.stable_xy
        if state.run_xy is not None:
            return state.run_xy
        return cx, cy
    if tmpl.startswith("hand"):
        if state.hand_locked_xy is None:
            # Lock at last stable position; fall back to run_xy only if no
            # stable position has been committed yet (e.g. very start of video).
            state.hand_locked_xy = (
                state.stable_xy if state.stable_xy is not None else state.run_xy
            )
        if state.hand_locked_xy is not None:
            return state.hand_locked_xy
        return cx, cy
    return cx, cy


def _square_roi_padded(gray: np.ndarray, cx: float, cy: float, half: int) -> np.ndarray:
    """Extract a (2*half)×(2*half) gray ROI centered on (cx,cy), replicating border."""
    h, w = gray.shape[:2]
    xi, yi = int(round(cx)), int(round(cy))
    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_REPLICATE)
    cy2, cx2 = yi + half, xi + half
    return padded[cy2 - half : cy2 + half, cx2 - half : cx2 + half]


def _annulus_weights(size: int, inner_r: float) -> np.ndarray:
    """Float weights for pixels with distance >= inner_r from patch center (disk crop)."""
    c = (size - 1) * 0.5
    yy, xx = np.ogrid[:size, :size]
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2).astype(np.float64)
    return (d >= inner_r).astype(np.float64)


def _annulus_diff_stat(
    diff: np.ndarray, mask_w: np.ndarray, mask_sum: float, agg: str
) -> float:
    """Reduce per-pixel absolute difference using annulus ``mask_w`` (0/1)."""
    if agg == "mean":
        return float((diff * mask_w).sum() / mask_sum)
    if agg == "max":
        sel = mask_w > 0
        if not np.any(sel):
            return 0.0
        return float(diff[sel].max())
    raise ValueError("diff_agg must be 'mean' or 'max'")


def click_happened_in_segment(
    video_path: Path | str,
    start_sec: float | None = None,
    end_sec: float | None = None,
    *,
    roi_half_px: int = CLICK_ROI_HALF_PX,
    inner_ignore_px: float = CLICK_INNER_IGNORE_PX,
    diff_mean_threshold: float = CLICK_DIFF_MEAN_THRESHOLD,
    diff_agg: str = "mean",
    debug: bool = False,
) -> bool:
    """
    Heuristic: **True** if any **substantial pixel change** occurs near the cursor
    hotspot between consecutive frames in [``start_sec``, ``end_sec``].

    For each frame (after the first), a grayscale patch centered on the **current**
    frame’s cursor position is cut from the **previous** and **current** full-res
    frames and compared. The inner disk (``inner_ignore_px``) around the hotspot
    is down-weighted so the cursor glyph is less likely to dominate the diff.

    This is a **proxy** for “UI changed under the cursor” (e.g. button state), not
    proof of a real click. Compression noise and scrolling can cause false
    positives; tune ``diff_mean_threshold`` if needed.

    ``diff_agg``:

    - ``\"mean\"`` — average absolute gray change in the annulus (default). With a
      **large** ``roi_half_px``, small UI updates can dilute this value.
    - ``\"max\"`` — strongest single-pixel change in the annulus; better when a
      **small** control (text, colored button) changes inside a broad patch.

    Threshold ``diff_mean_threshold`` applies to whichever statistic ``diff_agg``
    selects (despite the parameter name).

    If ``debug`` is True, log each comparable frame’s statistic and a short summary
    to **stderr**.

    ``roi_half_px`` / ``inner_ignore_px`` trade off coverage vs. ignoring the
    cursor bitmap: widen ``roi_half_px`` for text or buttons beside the hotspot;
    shrink ``inner_ignore_px`` if the visible change is mostly *under* the
    cursor tip (otherwise those pixels are excluded from the diff).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    templates = _prepare_templates(fw, fh)

    if start_sec is not None and start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_sec * fps)))

    sw = round(fw * FRAME_SCALE)
    sh = round(fh * FRAME_SCALE)

    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    stable_cursor_state = StableCursorState()
    prev_gray_full: np.ndarray | None = None
    n_compared = 0
    max_stat = 0.0
    agg = diff_agg.strip().lower()
    if agg not in ("mean", "max"):
        cap.release()
        raise ValueError("diff_agg must be 'mean' or 'max'")

    size = 2 * roi_half_px
    mask_w = _annulus_weights(size, inner_ignore_px)
    mask_sum = float(mask_w.sum())
    if mask_sum < 1.0:
        cap.release()
        raise ValueError("inner_ignore_px too large for roi_half_px; no pixels to compare")

    while True:
        if end_sec is not None and fps > 0 and frame_idx / fps > end_sec + 1e-6:
            break

        ok, bgr = cap.read()
        if not ok:
            break

        tsec = frame_idx / fps if fps > 0 else 0.0
        _ = tsec  # time available if caller wants logging later
        gray_small = cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
            (sw, sh),
            interpolation=cv2.INTER_AREA,
        )
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        det = _detect(gray_small, templates, fw, fh)

        out_x = out_y = 0.0
        has_cursor = False
        if det is not None:
            cx, cy, score, tmpl, _ = det
            out_x, out_y = _stable_cursor_track_step(
                stable_cursor_state, cx, cy, score, tmpl, tsec
            )
            has_cursor = True

        if (
            prev_gray_full is not None
            and has_cursor
            and roi_half_px > 0
        ):
            roi_prev = _square_roi_padded(prev_gray_full, out_x, out_y, roi_half_px)
            roi_cur = _square_roi_padded(gray_full, out_x, out_y, roi_half_px)
            diff = cv2.absdiff(roi_prev, roi_cur).astype(np.float64)
            stat = _annulus_diff_stat(diff, mask_w, mask_sum, agg)
            n_compared += 1
            max_stat = max(max_stat, stat)
            hit = stat >= diff_mean_threshold
            if debug:
                print(
                    f"click_debug frame={frame_idx} t={tsec:.6f} "
                    f"agg={agg} stat={stat:.3f} threshold={diff_mean_threshold} "
                    f"hit={hit}",
                    file=sys.stderr,
                )
            if hit:
                if debug:
                    print("click_debug result=True (first hit)", file=sys.stderr)
                cap.release()
                return True

        prev_gray_full = gray_full
        frame_idx += 1

    cap.release()
    if debug:
        print(
            f"click_debug summary compared={n_compared} agg={agg} "
            f"max_stat={max_stat:.3f} threshold={diff_mean_threshold} "
            f"result=False",
            file=sys.stderr,
        )
    return False


# Main video processing

def track_segment(
    video_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
    out_video: Path | None = None,
) -> list[dict]:
    """
    Run the tracker on [start_sec, end_sec] and return per-frame rows (same
    fields as the CSV: frame, time_sec, cx, cy, score, template, scale).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    templates = _prepare_templates(fw, fh)

    if start_sec and start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_sec * fps)))

    writer: cv2.VideoWriter | None = None
    if out_video:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)
        )

    sw = round(fw * FRAME_SCALE)
    sh = round(fh * FRAME_SCALE)

    rows: list[dict] = []
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    stable_cursor_state = StableCursorState()

    while True:
        if end_sec is not None and fps > 0 and frame_idx / fps > end_sec + 1e-6:
            break

        ok, bgr = cap.read()
        if not ok:
            break

        tsec = frame_idx / fps if fps > 0 else 0.0
        gray_small = cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
            (sw, sh),
            interpolation=cv2.INTER_AREA,
        )

        det = _detect(gray_small, templates, fw, fh)

        if det is not None:
            cx, cy, score, tmpl, sc = det
            out_x, out_y = _stable_cursor_track_step(
                stable_cursor_state, cx, cy, score, tmpl, tsec
            )
            rows.append(
                {
                    "frame": frame_idx,
                    "time_sec": f"{tsec:.6f}",
                    "cx": f"{out_x:.2f}",
                    "cy": f"{out_y:.2f}",
                    "score": f"{score:.4f}",
                    "template": tmpl,
                    "scale": f"{sc:.4f}",
                }
            )
            if writer is not None:
                cv2.drawMarker(
                    bgr,
                    (int(round(out_x)), int(round(out_y))),
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
            writer.write(bgr)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    return rows


def track_video(
    video_path: Path,
    out_csv: Path,
    out_video: Path | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> None:
    rows = track_segment(video_path, start_sec, end_sec, out_video)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
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
