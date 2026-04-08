#!/usr/bin/env python3
"""
Load labeled segments from JSON and run track_cursor.track_segment on each
[t_start, t_end] window to collect cursor positions per frame.

Expected JSON: a list of objects (or a single object) with:
  - video_id: filename or path to the recording
  - t_start / t_end: time segment in seconds (also accepts start_sec / end_sec)

Python / JSON output is a **list of objects**, one per frame:
``[{"frame": 0, "time_sec": 0.0, "cx": 1.2, "cy": 3.4}, ...]``.
Missing detections use ``null`` for ``cx`` / ``cy``. Multiple segments → list
of such lists.

Use :func:`cursor_segment_samples` for full tracks; :func:`cursor_start_xy` /
:func:`cursor_end_xy` / :func:`cursor_avg_xy` / :func:`cursor_max_speed` /
:func:`cursor_avg_speed` / :func:`cursor_total_distance` / :func:`cursor_dwell_time`
aggregate motion (pass ``samples=`` to avoid re-running the tracker).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from track_cursor import track_segment

__all__ = [
    "load_segments_from_json",
    "segment_time_range",
    "resolve_video_path",
    "rows_to_track_samples",
    "FrameTrack",
    "cursor_segment_samples",
    "cursor_start_xy",
    "cursor_end_xy",
    "cursor_avg_xy",
    "cursor_max_speed",
    "cursor_avg_speed",
    "cursor_total_distance",
    "cursor_dwell_time",
    "get_cursors_pos_from_segment",
]

_DEFAULT_VIDEOS_DIR = Path(__file__).resolve().parent / "videos"


def load_segments_from_json(path: Path | str) -> list[dict[str, Any]]:
    """Parse manifest JSON into a list of segment dicts."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    raise ValueError("JSON root must be an object or array of objects")


def segment_time_range(entry: dict[str, Any]) -> tuple[float, float]:
    t0 = entry.get("t_start", entry.get("start_sec"))
    t1 = entry.get("t_end", entry.get("end_sec"))
    if t0 is None or t1 is None:
        raise ValueError(
            f"Segment missing t_start/t_end (or start_sec/end_sec): {entry!r}"
        )
    return float(t0), float(t1)


def resolve_video_path(video_id: str, videos_dir: Path) -> Path:
    """Resolve video_id to an existing file path."""
    p = Path(video_id)
    if p.is_file():
        return p.resolve()
    cand = videos_dir / video_id
    if cand.is_file():
        return cand.resolve()
    cand = videos_dir / Path(video_id).name
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(
        f"Video not found for video_id={video_id!r} under {videos_dir}. "
        "Pass videos_dir or a full path in video_id."
    )


# One segment: list of {frame, time_sec, cx, cy}; cx/cy null when no hit
FrameTrack = list[dict[str, Any]]


def rows_to_track_samples(rows: list[dict]) -> FrameTrack:
    """``[{frame, time_sec, cx, cy}, ...]`` — JSON-ready dicts per frame."""
    out: FrameTrack = []
    for r in rows:
        cx_s, cy_s = r.get("cx", ""), r.get("cy", "")
        if cx_s == "" or cy_s == "":
            cx, cy = None, None
        else:
            cx, cy = float(cx_s), float(cy_s)
        out.append(
            {
                "frame": int(r["frame"]),
                "time_sec": float(r["time_sec"]),
                "cx": cx,
                "cy": cy,
            }
        )
    return out


def cursor_segment_samples(
    video_id: str,
    t_start: float,
    t_end: float,
    *,
    videos_dir: Path | str | None = None,
    preview_path: Path | str | None = None,
) -> FrameTrack:
    """
    Run the tracker on ``video_id`` for ``[t_start, t_end]`` (seconds) and
    return per-frame ``{frame, time_sec, cx, cy}`` samples.
    """
    vdir = Path(videos_dir) if videos_dir is not None else _DEFAULT_VIDEOS_DIR
    path = resolve_video_path(str(video_id), vdir)
    out_v = Path(preview_path) if preview_path else None
    raw = track_segment(path, start_sec=t_start, end_sec=t_end, out_video=out_v)
    return rows_to_track_samples(raw)


def cursor_start_xy(
    video_id: str,
    t_start: float,
    t_end: float,
    *,
    videos_dir: Path | str | None = None,
    samples: FrameTrack | None = None,
) -> tuple[float, float]:
    """
    First detected cursor position in the segment: ``(cursor_start_x, cursor_start_y)``.
    """
    track = (
        samples
        if samples is not None
        else cursor_segment_samples(video_id, t_start, t_end, videos_dir=videos_dir)
    )
    for s in track:
        cx, cy = s.get("cx"), s.get("cy")
        if cx is not None and cy is not None:
            return (float(cx), float(cy))
    raise ValueError(
        "No cursor detection in segment start: no frame with cx/cy in the window"
    )


def cursor_end_xy(
    video_id: str,
    t_start: float,
    t_end: float,
    *,
    videos_dir: Path | str | None = None,
    samples: FrameTrack | None = None,
) -> tuple[float, float]:
    """
    Last detected cursor position in the segment: ``(cursor_end_x, cursor_end_y)``.
    """
    track = (
        samples
        if samples is not None
        else cursor_segment_samples(video_id, t_start, t_end, videos_dir=videos_dir)
    )
    for s in reversed(track):
        cx, cy = s.get("cx"), s.get("cy")
        if cx is not None and cy is not None:
            return (float(cx), float(cy))
    raise ValueError(
        "No cursor detection in segment end: no frame with cx/cy in the window"
    )


def cursor_avg_xy(
    video_id: str,
    t_start: float,
    t_end: float,
    *,
    videos_dir: Path | str | None = None,
    samples: FrameTrack | None = None,
) -> tuple[float, float]:
    """
    Arithmetic mean of cursor positions over the segment: ``(cursor_avg_x,
    cursor_avg_y)``. Only frames with both ``cx`` and ``cy`` set contribute.
    """
    track = (
        samples
        if samples is not None
        else cursor_segment_samples(video_id, t_start, t_end, videos_dir=videos_dir)
    )
    xs: list[float] = []
    ys: list[float] = []
    for s in track:
        cx, cy = s.get("cx"), s.get("cy")
        if cx is not None and cy is not None:
            xs.append(float(cx))
            ys.append(float(cy))
    if not xs:
        raise ValueError(
            "No cursor detections in segment: cannot compute cursor_avg_x/cursor_avg_y"
        )
    n = len(xs)
    return (sum(xs) / n, sum(ys) / n)


def _pairwise_motion_intervals(track: FrameTrack) -> list[tuple[float, float]]:
    """
    For each consecutive pair with ``cx``/``cy`` on both frames and positive
    ``Δtime_sec``, append ``(distance_px, dt_sec)``.
    """
    intervals: list[tuple[float, float]] = []
    for a, b in zip(track, track[1:]):
        cxa, cya = a.get("cx"), a.get("cy")
        cxb, cyb = b.get("cx"), b.get("cy")
        if cxa is None or cya is None or cxb is None or cyb is None:
            continue
        dt = float(b["time_sec"]) - float(a["time_sec"])
        if dt <= 1e-9:
            continue
        dist = math.hypot(float(cxb) - float(cxa), float(cyb) - float(cya))
        intervals.append((dist, dt))
    return intervals


def _pairwise_instantaneous_speeds_px_s(track: FrameTrack) -> list[float]:
    """Euclidean speed (px/s) for each valid consecutive motion interval."""
    return [dist / dt for dist, dt in _pairwise_motion_intervals(track)]


def _any_cursor_detection(track: FrameTrack) -> bool:
    for s in track:
        cx, cy = s.get("cx"), s.get("cy")
        if cx is not None and cy is not None:
            return True
    return False


def cursor_max_speed(
    video_id: str,
    t_start: float,
    t_end: float,
    *,
    videos_dir: Path | str | None = None,
    samples: FrameTrack | None = None,
) -> float:
    """
    Peak cursor speed in **pixels per second** over adjacent frames: for each
    consecutive pair of samples that both have ``cx``/``cy``, compute Euclidean
    distance divided by ``Δ time_sec``; return the maximum. Pairs with
    non-positive ``Δt`` are skipped.
    """
    track = (
        samples
        if samples is not None
        else cursor_segment_samples(video_id, t_start, t_end, videos_dir=videos_dir)
    )
    speeds = _pairwise_instantaneous_speeds_px_s(track)
    if not speeds:
        raise ValueError(
            "No consecutive cursor detections with positive Δt in segment: "
            "cannot compute cursor_max_speed"
        )
    return max(speeds)


def cursor_avg_speed(
    video_id: str,
    t_start: float,
    t_end: float,
    *,
    videos_dir: Path | str | None = None,
    samples: FrameTrack | None = None,
) -> float:
    """
    Mean cursor speed in **pixels per second**: arithmetic average of the same
    pairwise instantaneous speeds as :func:`cursor_max_speed`.
    """
    track = (
        samples
        if samples is not None
        else cursor_segment_samples(video_id, t_start, t_end, videos_dir=videos_dir)
    )
    speeds = _pairwise_instantaneous_speeds_px_s(track)
    if not speeds:
        raise ValueError(
            "No consecutive cursor detections with positive Δt in segment: "
            "cannot compute cursor_avg_speed"
        )
    return sum(speeds) / len(speeds)


def cursor_total_distance(
    video_id: str,
    t_start: float,
    t_end: float,
    *,
    videos_dir: Path | str | None = None,
    samples: FrameTrack | None = None,
) -> float:
    """
    Total cursor **path length in pixels**: sum of Euclidean step lengths for the
    same consecutive pairs as :func:`cursor_max_speed` (both frames detected,
    positive ``Δt``). A single detection contributes **0** length. Raises if
    there are no detections in the segment at all.
    """
    track = (
        samples
        if samples is not None
        else cursor_segment_samples(video_id, t_start, t_end, videos_dir=videos_dir)
    )
    if not _any_cursor_detection(track):
        raise ValueError(
            "No cursor detections in segment: cannot compute cursor_total_distance"
        )
    return sum(dist for dist, _ in _pairwise_motion_intervals(track))


def cursor_dwell_time(
    video_id: str,
    t_start: float,
    t_end: float,
    *,
    videos_dir: Path | str | None = None,
    samples: FrameTrack | None = None,
    speed_threshold_px_s: float = 10.0,
) -> float:
    """
    **Low-motion dwell** in seconds: sum of ``Δt`` over consecutive detected
    pairs (same pairing as speed metrics) whose instantaneous speed is at most
    ``speed_threshold_px_s`` (pixels per second). Requires at least one
    detection in the segment; returns **0** when there are no valid intervals.
    """
    if speed_threshold_px_s < 0:
        raise ValueError("speed_threshold_px_s must be non-negative")
    track = (
        samples
        if samples is not None
        else cursor_segment_samples(video_id, t_start, t_end, videos_dir=videos_dir)
    )
    if not _any_cursor_detection(track):
        raise ValueError(
            "No cursor detections in segment: cannot compute cursor_dwell_time"
        )
    dwell = 0.0
    for dist, dt in _pairwise_motion_intervals(track):
        if dist / dt <= speed_threshold_px_s:
            dwell += dt
    return dwell


def get_cursors_pos_from_segment(
    manifest_path: Path | str,
    *,
    videos_dir: Path | str | None = None,
    segment_index: int | None = None,
    preview_path: Path | str | None = None,
    on_segment_start: Callable[[int, Path, float, float], None] | None = None,
) -> FrameTrack | list[FrameTrack]:
    """
    For each segment in the manifest, run the cursor tracker on
    ``[t_start, t_end]``.

    Parameters
    ----------
    manifest_path
        JSON file (object or array) with ``video_id``, ``t_start``, ``t_end``.
    videos_dir
        Folder containing ``video_id`` files. Defaults to ``./videos`` next to
        this module.
    segment_index
        If set, only the *i* th segment (0-based) is processed.
    preview_path
        If set, writes one MP4; only when the effective manifest has a single
        segment (after ``segment_index`` filtering) or one total entry.
    on_segment_start
        Optional ``callback(i, video_path, t_start, t_end)`` before each run.

    Returns
    -------
    If exactly one segment is processed: a list of
    ``{"frame": int, "time_sec": float, "cx": float | null, "cy": float | null}``.

    If multiple segments are processed: a list of those lists (one per segment).
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    vdir = Path(videos_dir) if videos_dir is not None else _DEFAULT_VIDEOS_DIR
    segments = load_segments_from_json(manifest_path)

    if segment_index is not None:
        if segment_index < 0 or segment_index >= len(segments):
            raise IndexError(
                f"segment_index {segment_index} out of range (0..{len(segments) - 1})"
            )
        segments = [segments[segment_index]]

    if preview_path is not None and len(segments) > 1:
        raise ValueError(
            "preview_path is only allowed for a single segment "
            "(set segment_index or use a one-entry manifest)"
        )

    preview = Path(preview_path) if preview_path is not None else None

    tracks: list[FrameTrack] = []
    for i, seg in enumerate(segments):
        video_id = seg.get("video_id")
        if not video_id:
            raise ValueError(f"Segment missing video_id: {seg!r}")
        t0, t1 = segment_time_range(seg)
        video_file = resolve_video_path(str(video_id), vdir)

        use_preview = preview if len(segments) == 1 else None
        if on_segment_start is not None:
            on_segment_start(i, video_file, t0, t1)

        raw_rows = track_segment(
            video_file, start_sec=t0, end_sec=t1, out_video=use_preview
        )
        tracks.append(rows_to_track_samples(raw_rows))

    if len(tracks) == 1:
        return tracks[0]
    return tracks

