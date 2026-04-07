import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from training import build_dataset, train_data, val_data
from collections import Counter


train_dataset = build_dataset(train_data)
val_dataset = build_dataset(val_data)

print("num train samples:", len(train_dataset))
print("num val samples:", len(val_dataset))
print("train sample shape:", train_dataset[0]["frames"].shape)
print("train sample label:", train_dataset[0]["label"])
print("train sample bbox:", train_dataset[0]["bbox_norm"])
print("train class counts:", Counter(row["label"] for row in train_data))
print("val class counts:", Counter(row["label"] for row in val_data))

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def bbox_to_tensor(bbox_norm):
    """Return a 5-d vector: [has_bbox, x1, y1, x2, y2]. Zeros when bbox is absent."""
    if bbox_norm is None:
        return torch.zeros(5)
    return torch.tensor([1.0] + list(bbox_norm), dtype=torch.float32)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # bbox branch: 5-dim input (has_bbox flag + 4 normalised coords)
        self.bbox_branch = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64 + 16, 3)

    def forward(self, x, bbox):
        x = x.mean(dim=1)              # (B, 4, 3, 224, 224) -> (B, 3, 224, 224)
        x = self.features(x)           # (B, 64, 1, 1)
        x = x.reshape(x.size(0), -1)   # (B, 64)
        b = self.bbox_branch(bbox)     # (B, 16)
        return self.classifier(torch.cat([x, b], dim=1))


model = SimpleCNN()

batch_frames = torch.tensor(
    np.stack([train_dataset[0]["frames"], train_dataset[1]["frames"]])
)
batch_bbox = torch.stack([
    bbox_to_tensor(train_dataset[0]["bbox_norm"]),
    bbox_to_tensor(train_dataset[1]["bbox_norm"]),
])

outputs = model(batch_frames, batch_bbox)

print("batch shape:", batch_frames.shape)
print("output shape:", outputs.shape)
print(outputs)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

indices = list(range(len(train_dataset)))

for epoch in range(30):
    model.train()
    total_loss = 0.0
    random.shuffle(indices)

    for i in indices:
        sample = train_dataset[i]
        x = torch.tensor(sample["frames"]).unsqueeze(0)   # (1, 4, 3, 224, 224)
        bbox = bbox_to_tensor(sample["bbox_norm"]).unsqueeze(0)  # (1, 5)
        y = torch.tensor([sample["label"]], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(x, bbox)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataset)
    print(f"epoch {epoch+1}, train loss: {avg_loss:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for sample in val_dataset:
        x = torch.tensor(sample["frames"]).unsqueeze(0)
        bbox = bbox_to_tensor(sample["bbox_norm"]).unsqueeze(0)
        y = sample["label"]

        logits = model(x, bbox)
        pred = torch.argmax(logits, dim=1).item()

        if pred == y:
            correct += 1
        total += 1

print("val accuracy:", correct / total)

model.eval()
pred_counts = {0: 0, 1: 0, 2: 0}

with torch.no_grad():
    for sample in val_dataset:
        x = torch.tensor(sample["frames"]).unsqueeze(0)
        bbox = bbox_to_tensor(sample["bbox_norm"]).unsqueeze(0)
        pred = torch.argmax(model(x, bbox), dim=1).item()
        pred_counts[pred] += 1

print("val predicted class counts:", pred_counts)