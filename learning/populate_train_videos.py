#!/usr/bin/env python3
"""Split repo ``videos/`` into train (80%) and val (20%) by sorted filename."""

from pathlib import Path

learning_dir = Path(__file__).resolve().parent
videos_dir = learning_dir.parent / "videos"
video_exts = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}


def videos_train_val_split(train_frac: float = 0.8) -> tuple[list[str], list[str]]:
    if not videos_dir.is_dir():
        return [], []
    names = sorted(
        p.name for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in video_exts
    )
    n = len(names)
    if n <= 1:
        return (names, []) if n == 1 else ([], [])
    n_train = max(1, min(round(train_frac * n), n - 1))
    return names[:n_train], names[n_train:]


if __name__ == "__main__":
    import json

    tr, va = videos_train_val_split()
    print("train:", json.dumps(tr, indent=2))
    print("val:", json.dumps(va, indent=2))
