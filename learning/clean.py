#!/usr/bin/env python3
"""Normalize training rows: video filename, seconds, labels, bbox as [x1,y1,x2,y2] or null."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROW_KEYS = ("video_id", "t_start", "t_end", "intervention", "bbox")
ALLOWED_LABELS = frozenset({"none", "zoom_in", "bounding_box"})


def video_filename(value: str) -> str:
    """Bare filename only (matches on-disk name), e.g. videos/a.mp4 -> a.mp4."""
    return Path(str(value).replace("\\", "/")).name


def seconds(value: object) -> float:
    """Timeline in seconds (float)."""
    return float(value)


def intervention_label(raw: str) -> str:
    """One canonical spelling per logical label."""
    s = raw.strip().lower().replace("-", "_")
    for ch in " \t":
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    synonyms = {"zoomin": "zoom_in", "boundingbox": "bounding_box"}
    s = synonyms.get(s, s)
    if s not in ALLOWED_LABELS:
        allowed = ", ".join(sorted(ALLOWED_LABELS))
        raise ValueError(f"unknown intervention {raw!r}; use one of: {allowed}")
    return s


def bbox_xyxy(raw: object) -> list[int]:
    """[x1, y1, x2, y2] pixel ints."""
    if isinstance(raw, dict):
        for k in ("x1", "y1", "x2", "y2"):
            if k not in raw:
                raise ValueError("bbox dict needs keys x1, y1, x2, y2")
        return [int(raw["x1"]), int(raw["y1"]), int(raw["x2"]), int(raw["y2"])]
    if isinstance(raw, list):
        if len(raw) != 4:
            raise ValueError("bbox list must have length 4 [x1, y1, x2, y2]")
        return [int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])]
    raise ValueError("bbox must be a dict, a list of 4 numbers, or null")


def clean_row(
    raw: dict,
    index: int,
    *,
    videos_dir: Path | None,
) -> dict:
    p = f"example {index}: "

    extra = set(raw) - set(ROW_KEYS)
    if extra:
        raise ValueError(p + f"unknown keys {sorted(extra)}")
    missing = [k for k in ROW_KEYS if k not in raw]
    if missing:
        raise ValueError(p + f"missing keys {missing}")

    name = video_filename(raw["video_id"])
    if videos_dir is not None and videos_dir.is_dir():
        path = videos_dir / name
        if not path.is_file():
            raise ValueError(
                p + f"no file {path} (video_id must match a filename in {videos_dir})"
            )

    t0, t1 = seconds(raw["t_start"]), seconds(raw["t_end"])
    if t0 >= t1:
        raise ValueError(p + f"need t_start < t_end, got {t0} and {t1}")

    label = intervention_label(str(raw["intervention"]))
    box = raw["bbox"]

    if label == "none":
        if box is not None:
            raise ValueError(p + '"none" rows must have bbox null')
        out_box = None
    else:
        if box is None:
            raise ValueError(p + f"intervention {label!r} needs a bbox")
        out_box = bbox_xyxy(box)

    return {
        "video_id": name,
        "t_start": t0,
        "t_end": t1,
        "intervention": label,
        "bbox": out_box,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="input JSON array (default: training_data_structure.json or labeled_interventions.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write here (default: overwrite input)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=repo / "videos",
        help="if this directory exists, require each video_id file to be present (default: <repo>/videos)",
    )
    parser.add_argument(
        "--skip-video-check",
        action="store_true",
        help="do not require files under --videos-dir",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate only; do not write",
    )
    args = parser.parse_args()

    in_path = args.path
    if in_path is None:
        for candidate in (repo / "training_data_structure.json", repo / "labeled_interventions.json"):
            if candidate.is_file():
                in_path = candidate
                break
        if in_path is None:
            raise SystemExit(
                "pass a JSON path, or add training_data_structure.json or labeled_interventions.json"
            )
    elif not in_path.is_file():
        raise SystemExit(f"not a file: {in_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("root JSON value must be an array")

    vdir = None if args.skip_video_check else args.videos_dir

    cleaned = [clean_row(row, i, videos_dir=vdir) for i, row in enumerate(data)]

    text = json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n"
    if not args.check:
        out = args.output or in_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(f"OK: {len(cleaned)} examples")


if __name__ == "__main__":
    main()
