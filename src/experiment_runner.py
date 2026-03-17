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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FIGURES = os.path.join(PROJECT_ROOT, "outputs", "figures")
OUTPUT_METRICS = os.path.join(PROJECT_ROOT, "outputs", "metrics")

os.makedirs(OUTPUT_FIGURES, exist_ok=True)
os.makedirs(OUTPUT_METRICS, exist_ok=True)


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


def plot_confusion_matrix(cm, title, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Real", "Fake"])
    plt.yticks([0, 1], ["Real", "Fake"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_experiment(
    experiment_name,
    model_name="spatial",
    epochs=3,
    batch_size=4,
    max_real=20,
    max_fake_per_csv=8,
    frame_mode="middle",
    image_size=128,
):
    print(f"\n=== Running Experiment: {experiment_name} ===")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Max real: {max_real}, Max fake per csv: {max_fake_per_csv}")
    print(f"Frame mode: {frame_mode}, Image size: {image_size}")

    dataset = FaceForensicsVideoDataset(
        dataset_root=DATASET_ROOT,
        max_real=max_real,
        max_fake_per_csv=max_fake_per_csv,
        frame_mode=frame_mode,
        image_size=image_size,
    )

    print(f"Total samples: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    use_fft = model_name.lower() == "frequency"
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
            f"[{experiment_name}] Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"f1={val_metrics['f1']:.4f}"
        )

    final_metrics = evaluate(model, val_loader, criterion, use_fft=use_fft)

    loss_path = os.path.join(OUTPUT_FIGURES, f"{experiment_name}_loss.png")
    metrics_path = os.path.join(OUTPUT_METRICS, f"{experiment_name}_metrics.json")
    cm_path = os.path.join(OUTPUT_FIGURES, f"{experiment_name}_confusion_matrix.png")

    plot_losses(
        train_losses,
        val_losses,
        loss_path,
        f"{experiment_name} Loss"
    )

    plot_confusion_matrix(
        final_metrics["confusion_matrix"],
        f"{experiment_name} Confusion Matrix",
        cm_path
    )

    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved loss plot to: {loss_path}")
    print(f"Saved confusion matrix to: {cm_path}")
    print("Final metrics:")
    print(json.dumps(final_metrics, indent=2))

    return final_metrics


def print_menu():
    print("\nChoose an experiment:")
    print("1. Spatial baseline")
    print("2. Frequency baseline")
    print("3. Run both baselines")
    print("4. Custom single experiment")
    print("5. Automatic sweep (multiple tests)")
    print("6. Quit")


def get_int_input(prompt, default):
    raw = input(f"{prompt} [default={default}]: ").strip()
    if raw == "":
        return default
    return int(raw)


def get_str_input(prompt, default):
    raw = input(f"{prompt} [default={default}]: ").strip()
    if raw == "":
        return default
    return raw


def run_spatial_default():
    run_experiment(
        experiment_name="spatial_baseline",
        model_name="spatial",
        epochs=3,
        batch_size=4,
        max_real=20,
        max_fake_per_csv=8,
    )


def run_frequency_default():
    run_experiment(
        experiment_name="frequency_baseline",
        model_name="frequency",
        epochs=3,
        batch_size=4,
        max_real=20,
        max_fake_per_csv=8,
    )


def run_both_defaults():
    run_spatial_default()
    run_frequency_default()


def run_custom():
    model_name = get_str_input("Model type (spatial/frequency)", "spatial").lower()
    experiment_name = get_str_input("Experiment name", f"{model_name}_custom")
    epochs = get_int_input("Epochs", 3)
    batch_size = get_int_input("Batch size", 4)
    max_real = get_int_input("Max real videos", 20)
    max_fake_per_csv = get_int_input("Max fake videos per CSV", 8)
    frame_mode = get_str_input("Frame mode (middle/random)", "middle")
    image_size = get_int_input("Image size", 128)

    run_experiment(
        experiment_name=experiment_name,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        max_real=max_real,
        max_fake_per_csv=max_fake_per_csv,
        frame_mode=frame_mode,
        image_size=image_size,
    )


def run_sweep():
    print("\nRunning a small automatic sweep...")

    experiments = [
        {
            "experiment_name": "spatial_e3_b4",
            "model_name": "spatial",
            "epochs": 3,
            "batch_size": 4,
            "max_real": 20,
            "max_fake_per_csv": 8,
        },
        {
            "experiment_name": "frequency_e3_b4",
            "model_name": "frequency",
            "epochs": 3,
            "batch_size": 4,
            "max_real": 20,
            "max_fake_per_csv": 8,
        },
        {
            "experiment_name": "spatial_e5_b4",
            "model_name": "spatial",
            "epochs": 5,
            "batch_size": 4,
            "max_real": 20,
            "max_fake_per_csv": 8,
        },
        {
            "experiment_name": "frequency_e5_b4",
            "model_name": "frequency",
            "epochs": 5,
            "batch_size": 4,
            "max_real": 20,
            "max_fake_per_csv": 8,
        },
        {
            "experiment_name": "spatial_balancedish",
            "model_name": "spatial",
            "epochs": 3,
            "batch_size": 4,
            "max_real": 20,
            "max_fake_per_csv": 3,
        },
        {
            "experiment_name": "frequency_balancedish",
            "model_name": "frequency",
            "epochs": 3,
            "batch_size": 4,
            "max_real": 20,
            "max_fake_per_csv": 3,
        },
    ]

    summary = []

    for config in experiments:
        metrics = run_experiment(**config)
        summary.append({
            "experiment": config["experiment_name"],
            "model": config["model_name"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "max_real": config["max_real"],
            "max_fake_per_csv": config["max_fake_per_csv"],
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

    summary_path = os.path.join(OUTPUT_METRICS, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved experiment summary to: {summary_path}")
    print("\nSummary:")
    for row in summary:
        print(row)


def main():
    while True:
        print_menu()
        choice = input("Enter choice: ").strip()

        if choice == "1":
            run_spatial_default()
        elif choice == "2":
            run_frequency_default()
        elif choice == "3":
            run_both_defaults()
        elif choice == "4":
            run_custom()
        elif choice == "5":
            run_sweep()
        elif choice == "6":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()