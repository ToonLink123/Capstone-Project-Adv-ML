"""
preprocess.py - Preprocess FaceForensics++ videos for model training

Extracts middle frames, computes FFT, and creates manifest CSV
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data"

FAKE_METHODS = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frames + FFT spectra for FF++ subset")
    p.add_argument("--data-root", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_DATA / "preprocessed"))
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-real", type=int, default=60)
    p.add_argument("--max-fake-per-method", type=int, default=12)
    return p.parse_args()


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def read_middle_frame(video_path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        # fallback: try first frame
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def fft_spectrum(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    f = np.fft.fftshift(f)
    mag = np.log(np.abs(f) + 1e-8)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return (mag * 255).astype(np.uint8)


def collect_real_videos(data_root: Path, max_real: int) -> list[Path]:
    real_dir = data_root / "original"
    videos: list[Path] = []
    for ext in ("*.mp4", "*.avi", "*.mov"):
        videos.extend(real_dir.rglob(ext))
    return sorted(videos)[:max_real]


def collect_fake_videos(data_root: Path, max_fake_per_method: int) -> list[tuple[Path, str]]:
    csv_dir = data_root / "csv"
    items: list[tuple[Path, str]] = []
    for method in FAKE_METHODS:
        csv_path = csv_dir / f"{method}.csv"
        if not csv_path.exists():
            continue
        df = _clean_columns(pd.read_csv(csv_path))
        path_col = next((c for c in df.columns if "file" in c.lower() and "path" in c.lower()), None)
        label_col = next((c for c in df.columns if "label" in c.lower()), None)
        if not path_col or not label_col:
            continue
        df = df[df[label_col].astype(str).str.upper().str.strip() == "FAKE"].head(max_fake_per_method)
        for _, row in df.iterrows():
            rel = str(row[path_col]).replace("\\", "/").strip()
            full = data_root / rel
            if full.exists():
                items.append((full, method))
    return items


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    frames_real = out_dir / "frames" / "REAL"
    frames_fake = out_dir / "frames" / "FAKE"
    fft_real = out_dir / "fft" / "REAL"
    fft_fake = out_dir / "fft" / "FAKE"
    for d in (frames_real, frames_fake, fft_real, fft_fake):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    size = (args.image_size, args.image_size)

    real_videos = collect_real_videos(data_root, args.max_real)
    print(f"Found {len(real_videos)} real videos")
    for vp in tqdm(real_videos, desc="REAL"):
        frame = read_middle_frame(vp)
        if frame is None:
            continue
        frame = cv2.resize(frame, size)
        spec = fft_spectrum(frame)
        f_path = frames_real / f"{vp.stem}.png"
        s_path = fft_real / f"{vp.stem}.png"
        cv2.imwrite(str(f_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(s_path), spec)
        rows.append({
            "frame_path": str(f_path),
            "fft_path": str(s_path),
            "label": 0,
            "method": "original",
        })

    fake_videos = collect_fake_videos(data_root, args.max_fake_per_method)
    print(f"Found {len(fake_videos)} fake videos across {len(FAKE_METHODS)} methods")
    for vp, method in tqdm(fake_videos, desc="FAKE"):
        frame = read_middle_frame(vp)
        if frame is None:
            continue
        frame = cv2.resize(frame, size)
        spec = fft_spectrum(frame)
        stem = f"{method}__{vp.stem}"
        f_path = frames_fake / f"{stem}.png"
        s_path = fft_fake / f"{stem}.png"
        cv2.imwrite(str(f_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(s_path), spec)
        rows.append({
            "frame_path": str(f_path),
            "fft_path": str(s_path),
            "label": 1,
            "method": method,
        })

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest with {len(manifest)} rows -> {manifest_path}")


if __name__ == "__main__":
    main()
