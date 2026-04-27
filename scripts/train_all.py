"""
train_all.py - Train all model variants on FaceForensics++ video subset

Models: spatial, frequency, fusion, attention_fusion, vgg16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from video_dataset import FaceForensicsVideoDataset
from model import SimpleCNN
from fusion_model import FusionCNN
from fusion_attention_model import AttentionFusionCNN
from fft import to_fft

import matplotlib.pyplot as plt
from torchvision.models import vgg16, VGG16_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

DEFAULT_DATA = PROJECT_ROOT / "data"
OUT_FIG_LOSS = PROJECT_ROOT / "outputs" / "figures" / "training_curves"
OUT_FIG_CM = PROJECT_ROOT / "outputs" / "figures" / "confusion_matrices"
OUT_METRICS = PROJECT_ROOT / "outputs" / "metrics"
OUT_CKPT = PROJECT_ROOT / "outputs" / "checkpoints"

ALL_MODELS = ["spatial", "frequency", "fusion", "attention_fusion", "vgg16"]


class VGG16TransferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for p in backbone.parameters():
            p.requires_grad = False
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.avgpool(self.features(x)))


def make_model(name: str) -> nn.Module:
    if name == "spatial":
        return SimpleCNN(in_channels=3).to(DEVICE)
    if name == "frequency":
        return SimpleCNN(in_channels=1).to(DEVICE)
    if name == "fusion":
        return FusionCNN().to(DEVICE)
    if name == "attention_fusion":
        return AttentionFusionCNN().to(DEVICE)
    if name == "vgg16":
        return VGG16TransferModel().to(DEVICE)
    raise ValueError(f"Unknown model: {name}")


def get_logits(model: nn.Module, images: torch.Tensor, fft_images: torch.Tensor, name: str) -> torch.Tensor:
    if name in ("fusion", "attention_fusion"):
        return model(images, fft_images)
    if name == "frequency":
        return model(to_fft(images))
    if name == "vgg16":
        x = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return model((x - mean) / std)
    return model(images)


def train_one_epoch(model, loader, optimizer, criterion, name) -> float:
    model.train()
    total = 0.0
    for images, fft_images, labels in loader:
        images = images.to(DEVICE)
        fft_images = fft_images.to(DEVICE)
        labels = labels.float().view(-1, 1).to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(get_logits(model, images, fft_images, name), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, name) -> dict:
    model.eval()
    total = 0.0
    preds: list[int] = []
    probs_all: list[float] = []
    labels_all: list[float] = []
    for images, fft_images, labels in loader:
        images = images.to(DEVICE)
        fft_images = fft_images.to(DEVICE)
        labels_t = labels.float().view(-1, 1).to(DEVICE)
        logits = get_logits(model, images, fft_images, name)
        total += criterion(logits, labels_t).item()
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds.extend((probs > 0.5).astype(int).tolist())
        probs_all.extend(probs.tolist())
        labels_all.extend(labels.cpu().numpy().tolist())
    try:
        auc = (
            roc_auc_score(labels_all, probs_all)
            if len(set(labels_all)) > 1
            else float("nan")
        )
    except Exception:
        auc = float("nan")
    return {
        "loss": total / max(len(loader), 1),
        "accuracy": float(accuracy_score(labels_all, preds)),
        "precision": float(precision_score(labels_all, preds, zero_division=0)),
        "recall": float(recall_score(labels_all, preds, zero_division=0)),
        "f1": float(f1_score(labels_all, preds, zero_division=0)),
        "roc_auc": float(auc) if not np.isnan(auc) else None,
        "confusion_matrix": confusion_matrix(labels_all, preds).tolist(),
    }


def plot_losses(train_losses, val_losses, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()


def plot_confusion(cm, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues"); plt.title(title); plt.colorbar()
    plt.xticks([0, 1], ["Real", "Fake"]); plt.yticks([0, 1], ["Real", "Fake"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    vmax = max(map(max, cm)) if cm else 1
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > vmax / 2 else "black")
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train all multi-domain detectors")
    p.add_argument("--data-root", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-real", type=int, default=60)
    p.add_argument("--max-fake-per-csv", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)

    for d in (OUT_FIG_LOSS, OUT_FIG_CM, OUT_METRICS, OUT_CKPT):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    dataset = FaceForensicsVideoDataset(
        dataset_root=args.data_root,
        max_real=args.max_real,
        max_fake_per_csv=args.max_fake_per_csv,
        frame_mode="middle",
        image_size=args.image_size,
    )
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    g = torch.Generator().manual_seed(SEED)
    tr, va = random_split(dataset, [n_train, n_val], generator=g)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Total samples: {n} (train={n_train}, val={n_val})")

    summary = []
    for name in args.models:
        print(f"\n=== {name} ===")
        model = make_model(name)
        params = filter(lambda p: p.requires_grad, model.parameters())
        lr = args.lr if name != "vgg16" else min(args.lr, 1e-4)
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        train_losses, val_losses = [], []
        for epoch in range(args.epochs):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, name)
            metrics = evaluate(model, val_loader, criterion, name)
            train_losses.append(tr_loss); val_losses.append(metrics["loss"])
            print(
                f"[{name}] Ep {epoch+1}/{args.epochs} "
                f"tr={tr_loss:.4f} val={metrics['loss']:.4f} "
                f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}"
            )

        final = evaluate(model, val_loader, criterion, name)
        plot_losses(train_losses, val_losses, OUT_FIG_LOSS / f"{name}_loss.png", f"{name} Loss")
        plot_confusion(final["confusion_matrix"], OUT_FIG_CM / f"{name}_cm.png",
                       f"{name} Confusion Matrix")
        with open(OUT_METRICS / f"{name}_metrics.json", "w") as fp:
            json.dump(final, fp, indent=2)
        torch.save(model.state_dict(), OUT_CKPT / f"{name}.pt")

        summary.append({"model": name, **{k: final[k] for k in
                        ("accuracy", "precision", "recall", "f1", "roc_auc", "loss")}})

    with open(OUT_METRICS / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    import csv
    with open(OUT_METRICS / "summary.csv", "w", newline="") as fp:
        if summary:
            w = csv.DictWriter(fp, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)
    print("\nSummary:")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
