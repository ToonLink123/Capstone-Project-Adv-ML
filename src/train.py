import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from video_dataset import FaceForensicsVideoDataset
from model import SimpleCNN
from fusion_model import FusionCNN
from fft import to_fft
from fusion_attention_model import AttentionFusionCNN

DATASET_ROOT = r"D:\OpenCV\Capstone-Project-Adv-ML\data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion, model_name="spatial"):
    model.train()
    total_loss = 0.0

    for images, fft_images, labels in loader:
        images = images.to(DEVICE)
        fft_images = fft_images.to(DEVICE)
        labels = labels.float().view(-1, 1).to(DEVICE)

        optimizer.zero_grad()

        if model_name in ["fusion", "attention_fusion"]:
            logits = model(images, fft_images)
        elif model_name == "frequency":
            logits = model(to_fft(images))
        else:
            logits = model(images)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, model_name="spatial"):
    model.eval()
    total_loss = 0.0
    preds = []
    labels_all = []

    for images, fft_images, labels in loader:
        images = images.to(DEVICE)
        fft_images = fft_images.to(DEVICE)
        labels_tensor = labels.float().view(-1, 1).to(DEVICE)

        if model_name in ["fusion", "attention_fusion"]:
            logits = model(images, fft_images)
        elif model_name == "frequency":
            logits = model(to_fft(images))
        else:
            logits = model(images)

        loss = criterion(logits, labels_tensor)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        batch_preds = (probs > 0.5).int().cpu().numpy().flatten()

        preds.extend(batch_preds.tolist())
        labels_all.extend(labels.cpu().numpy().tolist())

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(labels_all, preds),
        "precision": precision_score(labels_all, preds, zero_division=0),
        "recall": recall_score(labels_all, preds, zero_division=0),
        "f1": f1_score(labels_all, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels_all, preds).tolist()
    }


def plot_losses(train_losses, val_losses, out_path, title):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_experiment(model_name="spatial", epochs=5, batch_size=8):
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    dataset = FaceForensicsVideoDataset(
        dataset_root=DATASET_ROOT,
        max_real=20,
        max_fake_per_csv=8,
        frame_mode="middle",
        image_size=128,
    )

    print(f"Total samples: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    if model_name == "fusion":
        model = FusionCNN().to(DEVICE)
    elif model_name == "attention_fusion":
        model = AttentionFusionCNN().to(DEVICE)
    elif model_name == "frequency":
        model = SimpleCNN(in_channels=1).to(DEVICE)
    else:
        model = SimpleCNN(in_channels=3).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, model_name=model_name)
        val_metrics = evaluate(model, val_loader, criterion, model_name=model_name)

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])

        print(
            f"[{model_name}] Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"f1={val_metrics['f1']:.4f}"
        )

    plot_losses(
        train_losses,
        val_losses,
        f"outputs/figures/{model_name}_loss.png",
        f"{model_name.capitalize()} Loss"
    )

    final_metrics = evaluate(model, val_loader, criterion, model_name=model_name)

    with open(f"outputs/metrics/{model_name}_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    return final_metrics


if __name__ == "__main__":
    spatial = run_experiment("spatial", epochs=3, batch_size=4)
    frequency = run_experiment("frequency", epochs=3, batch_size=4)
    fusion = run_experiment("fusion", epochs=3, batch_size=4)
    attention_fusion = run_experiment("attention_fusion", epochs=5, batch_size=4)

    print("\nSpatial metrics:", spatial)
    print("Frequency metrics:", frequency)
    print("Fusion metrics:", fusion)
    print("Attention Fusion metrics:", attention_fusion)
