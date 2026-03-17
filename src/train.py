import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from video_dataset import FaceForensicsVideoDataset
from model import SimpleCNN
from fft import to_fft


DATASET_ROOT = r"D:\OpenCV\Capstone-Project-Adv-ML\data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion, use_fft=False):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.float().unsqueeze(1).to(DEVICE)

        if use_fft:
            x = to_fft(x)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, use_fft=False):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []

    for x, y in loader:
        x = x.to(DEVICE)
        y_tensor = y.float().unsqueeze(1).to(DEVICE)

        if use_fft:
            x = to_fft(x)

        logits = model(x)
        loss = criterion(logits, y_tensor)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        batch_preds = (probs > 0.5).int().cpu().numpy().flatten()

        preds.extend(batch_preds.tolist())
        labels.extend(y.numpy().tolist())

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
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

    use_fft = model_name == "frequency"
    in_channels = 1 if use_fft else 3

    model = SimpleCNN(in_channels=in_channels).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, use_fft=use_fft)
        val_metrics = evaluate(model, val_loader, criterion, use_fft=use_fft)

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

    final_metrics = evaluate(model, val_loader, criterion, use_fft=use_fft)

    with open(f"outputs/metrics/{model_name}_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    return final_metrics


if __name__ == "__main__":
    spatial = run_experiment("spatial", epochs=3, batch_size=4)
    frequency = run_experiment("frequency", epochs=3, batch_size=4)

    print("\nSpatial metrics:", spatial)
    print("Frequency metrics:", frequency)