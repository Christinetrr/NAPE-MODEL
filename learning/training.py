from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from populate_train_videos import videos_train_val_split

learning_dir = Path(__file__).resolve().parent
videos_dir = learning_dir.parent / "videos"

label_map = {
    "none": 0,
    "zoom_in": 1,
    "bounding_box": 2,
}

#split data to training and validation
train_videos, val_videos = videos_train_val_split()
train_ids = set(train_videos)
val_ids = set(val_videos)

labeled_rows = json.loads((learning_dir / "labeled_interventions.json").read_text(encoding="utf-8"))
train_data = [row for row in labeled_rows if row["video_id"] in train_ids]
val_data = [row for row in labeled_rows if row["video_id"] in val_ids]

#convert interventions to machine code
for row in train_data:
    row["label"] = label_map[row["intervention"]]

for row in val_data:
    row["label"] = label_map[row["intervention"]]

# extract frames given segment and video id; returns RGB uint8 arrays shaped (224, 224, 3)
def extract_frames(video_id, t_start, t_end, num_frames: int = 4):
    if t_start >= t_end or num_frames < 1:
        raise ValueError("need t_start < t_end and num_frames >= 1")
    path = videos_dir / video_id
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        last_i = None if n_frames <= 0 else n_frames - 1
        frames = []
        for t in np.linspace(t_start, t_end, num_frames, endpoint=False):
            idx = int(round(t * fps))
            if last_i is not None:
                idx = min(max(idx, 0), last_i)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if (not ok or frame is None) and last_i is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, last_i)
                ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"no frame at {t:.3f}s in {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        return frames
    finally:
        cap.release()

def normalize_bbox(bbox, width=1920, height=1080):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [x1/width, y1/height, x2/width, y2/height]

for row in train_data:
    row["bbox_norm"] = normalize_bbox(row["bbox"])

for row in val_data:
    row["bbox_norm"] = normalize_bbox(row["bbox"])

def build_dataset(rows):
    dataset = []
    for row in rows:
        frames = extract_frames(row["video_id"], row["t_start"], row["t_end"])
        frames = np.stack(frames, axis=0)
        frames = np.transpose(frames, (0, 3, 1, 2))
        frames = frames.astype(np.float32) / 255.0

        sample = {
            "frames": frames,
            "label": row["label"],
            "bbox_norm": row["bbox_norm"],
            "video_id": row["video_id"],
            "t_start": row["t_start"],
            "t_end": row["t_end"],
        }
        dataset.append(sample)
    return dataset


