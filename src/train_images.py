import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from image_dataset import RealFakeImageDataset
from model import SimpleCNN
from fusion_model import FusionCNN
from fusion_attention_model import AttentionFusionCNN
from fft import to_fft


DATA_ROOT = r"D:\OpenCV\Capstone-Project-Adv-ML\data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_logits(model, images, fft_images, model_name):
    if model_name in ["fusion", "attention_fusion"]:
        return model(images, fft_images)
    elif model_name == "frequency":
        return model(to_fft(images))
    else:
        return model(images)


def train_one_epoch(model, loader, optimizer, criterion, model_name):
    model.train()
    total_loss = 0.0

    for images, fft_images, labels in loader:
        images = images.to(DEVICE)
        fft_images = fft_images.to(DEVICE)
        labels = labels.float().view(-1, 1).to(DEVICE)

        optimizer.zero_grad()
        logits = get_logits(model, images, fft_images, model_name)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, model_name):
    model.eval()
    total_loss = 0.0
    preds = []
    labels_all = []

    for images, fft_images, labels in loader:
        images = images.to(DEVICE)
        fft_images = fft_images.to(DEVICE)
        labels_tensor = labels.float().view(-1, 1).to(DEVICE)

        logits = get_logits(model, images, fft_images, model_name)
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


def make_model(model_name):
    if model_name == "fusion":
        return FusionCNN().to(DEVICE)
    elif model_name == "attention_fusion":
        return AttentionFusionCNN().to(DEVICE)
    elif model_name == "frequency":
        return SimpleCNN(in_channels=1).to(DEVICE)
    else:
        return SimpleCNN(in_channels=3).to(DEVICE)


def plot_losses(train_losses, test_losses, out_path, title):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_experiment(model_name, epochs=10, batch_size=32, image_size=128, max_per_class=None):
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    train_dataset = RealFakeImageDataset(
        os.path.join(DATA_ROOT, "train"),
        image_size=image_size,
        max_per_class=max_per_class
    )

    test_dataset = RealFakeImageDataset(
        os.path.join(DATA_ROOT, "test"),
        image_size=image_size,
        max_per_class=max_per_class
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = make_model(model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    test_losses = []

    best_f1 = -1

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, model_name)
        test_metrics = evaluate(model, test_loader, criterion, model_name)

        train_losses.append(train_loss)
        test_losses.append(test_metrics["loss"])

        print(
            f"[{model_name}] Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"acc={test_metrics['accuracy']:.4f} | "
            f"f1={test_metrics['f1']:.4f}"
        )

        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            torch.save(model.state_dict(), f"outputs/checkpoints/{model_name}_best.pt")

    plot_losses(
        train_losses,
        test_losses,
        f"outputs/figures/{model_name}_image_loss.png",
        f"{model_name} Image Dataset Loss"
    )

    final_metrics = evaluate(model, test_loader, criterion, model_name)

    with open(f"outputs/metrics/{model_name}_image_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    return final_metrics


if __name__ == "__main__":
    print("Using device:", DEVICE)

    # Start smaller to confirm everything works
    spatial = run_experiment("spatial", epochs=5, batch_size=32, max_per_class=2000)
    frequency = run_experiment("frequency", epochs=5, batch_size=32, max_per_class=2000)
    fusion = run_experiment("fusion", epochs=5, batch_size=32, max_per_class=2000)
    attention_fusion = run_experiment("attention_fusion", epochs=5, batch_size=32, max_per_class=2000)

    print("\nSpatial:", spatial)
    print("Frequency:", frequency)
    print("Fusion:", fusion)
    print("Attention Fusion:", attention_fusion)