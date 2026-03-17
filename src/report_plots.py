import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


DATASET_ROOT = r"D:\OpenCV\Capstone-Project-Adv-ML\data"
CSV_DIR = os.path.join(DATASET_ROOT, "csv")
OUTPUT_DIR = r"D:\OpenCV\Capstone-Project-Adv-ML\outputs\figures"
METRICS_DIR = r"D:\OpenCV\Capstone-Project-Adv-ML\outputs\metrics"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_confusion_matrix(cm, labels, title, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_metrics(filename):
    path = os.path.join(METRICS_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)


def plot_metric_bar_chart(spatial_metrics, frequency_metrics, out_path):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    spatial_vals = [spatial_metrics[m] for m in metric_names]
    freq_vals = [frequency_metrics[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, spatial_vals, width, label="Spatial CNN")
    plt.bar(x + width/2, freq_vals, width, label="Frequency FFT-CNN")

    plt.xticks(x, [m.capitalize() for m in metric_names])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Baseline Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_class_distribution():
    valid_csv_names = [
        "DeepFakeDetection.csv",
        "Deepfakes.csv",
        "Face2Face.csv",
        "FaceShifter.csv",
        "FaceSwap.csv",
        "NeuralTextures.csv",
        "original.csv",
    ]

    real_count = 0
    fake_count = 0

    for csv_name in valid_csv_names:
        csv_path = os.path.join(CSV_DIR, csv_name)
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

        if "Label" in df.columns:
            labels = df["Label"].astype(str).str.upper().str.strip()
            real_count += (labels == "REAL").sum()
            fake_count += (labels == "FAKE").sum()

    plt.figure(figsize=(6, 4))
    plt.bar(["Real", "Fake"], [real_count, fake_count])
    plt.ylabel("Number of Videos")
    plt.title("Class Distribution from CSV Metadata")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    plt.close()

    print(f"Real count: {real_count}, Fake count: {fake_count}")


def read_middle_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read video: {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    middle = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read middle frame: {video_path}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def fft_mag(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256))
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_shift) + 1)
    return gray, mag


def plot_fft_comparison():
    real_video = None
    fake_video = None

    original_dir = Path(DATASET_ROOT) / "original"
    for ext in ("*.mp4", "*.avi", "*.mov"):
        vids = list(original_dir.rglob(ext))
        if vids:
            real_video = vids[0]
            break

    fake_csv = Path(CSV_DIR) / "DeepFakeDetection.csv"
    df = pd.read_csv(fake_csv)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    fake_rel = df[df["Label"].astype(str).str.upper().str.strip() == "FAKE"].iloc[0]["File Path"]
    fake_video = Path(DATASET_ROOT) / str(fake_rel).replace("/", os.sep)

    if real_video is None or not fake_video.exists():
        print("Could not find real/fake example videos for FFT comparison.")
        return

    real_frame = read_middle_frame(real_video)
    fake_frame = read_middle_frame(fake_video)

    real_gray, real_fft = fft_mag(real_frame)
    fake_gray, fake_fft = fft_mag(fake_frame)

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(real_gray, cmap="gray")
    plt.title("Real Frame")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(real_fft, cmap="gray")
    plt.title("Real FFT Magnitude")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(fake_gray, cmap="gray")
    plt.title("Fake Frame")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(fake_fft, cmap="gray")
    plt.title("Fake FFT Magnitude")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fft_comparison.png"))
    plt.close()


def main():
    ensure_dirs()

    spatial_metrics = load_metrics("spatial_metrics.json")
    frequency_metrics = load_metrics("frequency_metrics.json")

    plot_confusion_matrix(
        spatial_metrics["confusion_matrix"],
        ["Real", "Fake"],
        "Spatial CNN Confusion Matrix",
        os.path.join(OUTPUT_DIR, "spatial_confusion_matrix.png")
    )

    plot_confusion_matrix(
        frequency_metrics["confusion_matrix"],
        ["Real", "Fake"],
        "Frequency FFT-CNN Confusion Matrix",
        os.path.join(OUTPUT_DIR, "frequency_confusion_matrix.png")
    )

    plot_metric_bar_chart(
        spatial_metrics,
        frequency_metrics,
        os.path.join(OUTPUT_DIR, "baseline_comparison.png")
    )

    plot_class_distribution()
    plot_fft_comparison()

    print("Saved all report plots to outputs/figures")


if __name__ == "__main__":
    main()