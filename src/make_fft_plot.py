import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "data")
CSV_DIR = os.path.join(DATASET_ROOT, "csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fft_comparison.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_middle_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Could not read video: {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    middle = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Could not read middle frame: {video_path}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def fft_mag(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256))
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_shift) + 1)
    return gray, mag


def get_real_video():
    original_dir = Path(DATASET_ROOT) / "original"
    for ext in ("*.mp4", "*.avi", "*.mov"):
        vids = list(original_dir.rglob(ext))
        if vids:
            return vids[0]
    raise FileNotFoundError("No real video found in original/")


def get_fake_video():
    fake_csv = Path(CSV_DIR) / "DeepFakeDetection.csv"
    if not fake_csv.exists():
        raise FileNotFoundError("DeepFakeDetection.csv not found")

    df = pd.read_csv(fake_csv)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    df = df[df["Label"].astype(str).str.upper().str.strip() == "FAKE"]

    if len(df) == 0:
        raise ValueError("No FAKE rows found in DeepFakeDetection.csv")

    fake_rel = str(df.iloc[0]["File Path"]).replace("/", os.sep).strip()
    fake_video = Path(DATASET_ROOT) / fake_rel

    if not fake_video.exists():
        raise FileNotFoundError(f"Fake video not found: {fake_video}")

    return fake_video


def main():
    real_video = get_real_video()
    fake_video = get_fake_video()

    print(f"Using real video: {real_video}")
    print(f"Using fake video: {fake_video}")

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
    plt.savefig(OUTPUT_PATH)
    plt.close()

    print(f"Saved FFT plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()