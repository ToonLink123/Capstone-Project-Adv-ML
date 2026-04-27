import os
import random
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def make_fft_image(image_tensor):
    gray = image_tensor.mean(dim=0, keepdim=True)

    fft = torch.fft.fft2(gray)
    fft = torch.fft.fftshift(fft)

    magnitude = torch.log(torch.abs(fft) + 1e-8)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

    return magnitude.float()

class FaceForensicsVideoDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        max_real=40,
        max_fake_per_csv=20,
        frame_mode="middle",
        image_size=128,
    ):
        self.dataset_root = Path(dataset_root)
        self.csv_dir = self.dataset_root / "csv"
        self.samples = []
        self.frame_mode = frame_mode

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # Real videos
        real_videos = []
        for ext in ("*.mp4", "*.avi", "*.mov"):
            real_videos.extend((self.dataset_root / "original").rglob(ext))
        real_videos = sorted(real_videos)[:max_real]

        for path in real_videos:
            self.samples.append((str(path), 0))

        # Fake videos
        for csv_file in sorted(self.csv_dir.glob("*.csv")):
            if csv_file.stem.lower() == "original":
                continue

            df = pd.read_csv(csv_file)

            df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

            print(f"\nLoaded CSV: {csv_file.name}")
            print("Columns:", df.columns.tolist())

            path_col = None
            label_col = None

            for col in df.columns:
                col_lower = col.lower()
                if "file" in col_lower and "path" in col_lower:
                    path_col = col
                if "label" in col_lower:
                    label_col = col

            if path_col is None:
                print(f"Skipping {csv_file.name}: no file path column")
                continue

            if label_col is None:
                print(f"Skipping {csv_file.name}: no label column")
                continue

            df = df[df[label_col].astype(str).str.upper().str.strip() == "FAKE"]

            if len(df) == 0:
                print(f"Skipping {csv_file.name}: no FAKE rows")
                continue

            df = df.head(max_fake_per_csv)

            for _, row in df.iterrows():
                rel_path = str(row[path_col]).replace("/", os.sep).strip()
                full_path = self.dataset_root / rel_path

                if full_path.exists():
                    self.samples.append((str(full_path), 1))
                else:
                    print(f"Missing fake video: {full_path}")

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _read_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Invalid frame count: {video_path}")

        if self.frame_mode == "random":
            target = random.randint(0, max(frame_count - 1, 0))
        else:
            target = frame_count // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise RuntimeError(f"Could not read frame from: {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        try:
            frame = self._read_frame(video_path)
        except Exception:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise RuntimeError(f"Failed to load video entirely: {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.transform(frame)
        fft_image = make_fft_image(image)
        return image, fft_image, torch.tensor(label, dtype=torch.float32)