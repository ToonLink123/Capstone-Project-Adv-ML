import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_ROOT = r"D:\OpenCV\Capstone-Project-Adv-ML\data"
CSV_DIR = os.path.join(DATASET_ROOT, "csv")
OUTPUT_PATH = os.path.join("..", "outputs", "figures", "class_distribution.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

valid_csvs = [
    "DeepFakeDetection.csv",
    "Deepfakes.csv",
    "Face2Face.csv",
    "FaceShifter.csv",
    "FaceSwap.csv",
    "NeuralTextures.csv"
]

fake_count = 0

for csv_name in valid_csvs:
    path = os.path.join(CSV_DIR, csv_name)
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    fake_count += (df["Label"].astype(str).str.upper().str.strip() == "FAKE").sum()

real_count = 20

plt.figure(figsize=(6, 4))
plt.bar(["Real", "Fake"], [real_count, fake_count])
plt.ylabel("Number of Samples")
plt.title("Class Distribution in Training Subset")

plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.close()

print("Saved class_distribution.png")
print(f"Real: {real_count}, Fake: {fake_count}")