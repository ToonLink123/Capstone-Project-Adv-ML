import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_ROOT = r"D:\OpenCV\Capstone-Project-Adv-ML\data"
CSV_DIR = os.path.join(DATASET_ROOT, "csv")
OUTPUT_DIR = r"D:\OpenCV\Capstone-Project-Adv-ML\outputs\figures"


def clean_columns(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def load_csv(name):
    path = os.path.join(CSV_DIR, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return clean_columns(df)


def plot_fake_by_method():
    methods = [
        "DeepFakeDetection.csv",
        "Deepfakes.csv",
        "Face2Face.csv",
        "FaceShifter.csv",
        "FaceSwap.csv",
        "NeuralTextures.csv"
    ]

    names = []
    counts = []

    for m in methods:
        df = load_csv(m)
        if df is None:
            continue
        count = (df["Label"].astype(str).str.upper().str.strip() == "FAKE").sum()
        names.append(m.replace(".csv", ""))
        counts.append(count)

    plt.figure(figsize=(8, 5))
    plt.bar(names, counts)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Number of Fake Videos")
    plt.title("Fake Video Count by Manipulation Method")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fake_by_method.png"))
    plt.close()


def plot_frame_count_distribution():
    methods = [
        "DeepFakeDetection.csv",
        "Deepfakes.csv",
        "Face2Face.csv",
        "FaceShifter.csv",
        "FaceSwap.csv",
        "NeuralTextures.csv",
        "original.csv"
    ]

    frame_counts = []

    for m in methods:
        df = load_csv(m)
        if df is None or "Frame Count" not in df.columns:
            continue
        frame_counts.extend(df["Frame Count"].dropna().tolist())

    plt.figure(figsize=(7, 4))
    plt.hist(frame_counts, bins=20)
    plt.xlabel("Frame Count")
    plt.ylabel("Number of Videos")
    plt.title("Distribution of Video Frame Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "frame_count_distribution.png"))
    plt.close()


def plot_file_size_distribution():
    methods = [
        "DeepFakeDetection.csv",
        "Deepfakes.csv",
        "Face2Face.csv",
        "FaceShifter.csv",
        "FaceSwap.csv",
        "NeuralTextures.csv",
        "original.csv"
    ]

    sizes = []

    for m in methods:
        df = load_csv(m)
        if df is None or "File Size(MB)" not in df.columns:
            continue
        sizes.extend(df["File Size(MB)"].dropna().tolist())

    plt.figure(figsize=(7, 4))
    plt.hist(sizes, bins=20)
    plt.xlabel("File Size (MB)")
    plt.ylabel("Number of Videos")
    plt.title("Distribution of Video File Sizes")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "file_size_distribution.png"))
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_fake_by_method()
    plot_frame_count_distribution()
    plot_file_size_distribution()
    print("Saved EDA plots to outputs/figures")


if __name__ == "__main__":
    main() 