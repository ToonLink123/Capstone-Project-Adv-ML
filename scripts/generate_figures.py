"""
generate_figures.py - Generate labeled image outputs for the project
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

DATA_ROOT = PROJECT_ROOT / "data"
CSV_DIR = DATA_ROOT / "csv"
FIG_ROOT = PROJECT_ROOT / "outputs" / "figures"
METRIC_DIR = PROJECT_ROOT / "outputs" / "metrics"
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

EDA_DIR = FIG_ROOT / "eda"
SAMPLES_DIR = FIG_ROOT / "samples"
FFT_DIR = FIG_ROOT / "fft"
CM_DIR = FIG_ROOT / "confusion_matrices"
CMP_DIR = FIG_ROOT / "comparison"
GRADCAM_DIR = FIG_ROOT / "gradcam"

FAKE_METHODS = [
    "DeepFakeDetection", "Deepfakes", "Face2Face",
    "FaceShifter", "FaceSwap", "NeuralTextures",
]


def _ensure_dirs() -> None:
    for d in (EDA_DIR, SAMPLES_DIR, FFT_DIR, CM_DIR, CMP_DIR, GRADCAM_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def _read_csv(name: str) -> Optional[pd.DataFrame]:
    p = CSV_DIR / name
    if not p.exists():
        return None
    return _clean(pd.read_csv(p))


# --------------------------------------------------------------------------- #
# EDA figures
# --------------------------------------------------------------------------- #

def figure_class_distribution() -> None:
    real_df = _read_csv("original.csv")
    real_count = 0 if real_df is None else len(real_df)
    fake_count = 0
    for m in FAKE_METHODS:
        df = _read_csv(f"{m}.csv")
        if df is None or "Label" not in df.columns:
            continue
        fake_count += (df["Label"].astype(str).str.upper().str.strip() == "FAKE").sum()

    plt.figure(figsize=(6, 4))
    plt.bar(["Real", "Fake"], [real_count, fake_count], color=["steelblue", "indianred"])
    for i, v in enumerate([real_count, fake_count]):
        plt.text(i, v, str(int(v)), ha="center", va="bottom")
    plt.ylabel("Number of Videos"); plt.title("Class Distribution (Real vs Fake)")
    plt.tight_layout(); plt.savefig(EDA_DIR / "class_distribution.png", dpi=120); plt.close()


def figure_method_breakdown() -> None:
    counts = {}
    for m in FAKE_METHODS:
        df = _read_csv(f"{m}.csv")
        if df is None or "Label" not in df.columns:
            continue
        counts[m] = int((df["Label"].astype(str).str.upper().str.strip() == "FAKE").sum())
    plt.figure(figsize=(8, 4.5))
    plt.bar(list(counts.keys()), list(counts.values()), color="indianred")
    plt.xticks(rotation=25, ha="right")
    for i, (k, v) in enumerate(counts.items()):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.ylabel("Fake Videos"); plt.title("Fake Video Count by Manipulation Method")
    plt.tight_layout(); plt.savefig(EDA_DIR / "method_breakdown.png", dpi=120); plt.close()


def _hist_from_csv(column: str, title: str, xlabel: str, out: Path, color: str) -> None:
    values: list[float] = []
    for m in FAKE_METHODS + ["original"]:
        df = _read_csv(f"{m}.csv")
        if df is None or column not in df.columns:
            continue
        values.extend(df[column].dropna().tolist())
    if not values:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=20, color=color, edgecolor="black", alpha=0.85)
    plt.xlabel(xlabel); plt.ylabel("Number of Videos"); plt.title(title)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def figure_metadata_distributions() -> None:
    _hist_from_csv("Frame Count", "Distribution of Video Frame Counts",
                   "Frame Count", EDA_DIR / "frame_count_distribution.png", "steelblue")
    _hist_from_csv("File Size(MB)", "Distribution of Video File Sizes (MB)",
                   "File Size (MB)", EDA_DIR / "file_size_distribution.png", "seagreen")


# --------------------------------------------------------------------------- #
# Sample frames + FFT figures
# --------------------------------------------------------------------------- #

def _read_middle(video_path: Path, size: int = 256) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ret, frame = cap.read(); cap.release()
    if not ret or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (size, size))


def _fft_log_mag(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(g))
    return np.log(np.abs(f) + 1e-8)


def _first_video(directory: Path) -> Optional[Path]:
    for ext in ("*.mp4", "*.avi", "*.mov"):
        for v in sorted(directory.rglob(ext)):
            return v
    return None


def figure_sample_grid() -> None:
    real_v = _first_video(DATA_ROOT / "original")
    fake_videos = []
    for m in FAKE_METHODS:
        v = _first_video(DATA_ROOT / m)
        if v is not None:
            fake_videos.append((m, v))
    if real_v is None or not fake_videos:
        return

    cols = max(3, len(fake_videos))
    fig, axes = plt.subplots(2, cols, figsize=(2.5 * cols, 5))
    real_frame = _read_middle(real_v)
    if real_frame is not None:
        axes[0, 0].imshow(real_frame); axes[0, 0].set_title("REAL\n(original)")
    for ax in axes[0, 1:]:
        ax.axis("off")
    axes[0, 0].axis("off")

    for j, (m, v) in enumerate(fake_videos):
        if j >= cols:
            break
        f = _read_middle(v)
        if f is None:
            axes[1, j].axis("off"); continue
        axes[1, j].imshow(f); axes[1, j].set_title(f"FAKE\n{m}")
        axes[1, j].axis("off")
    for j in range(len(fake_videos), cols):
        axes[1, j].axis("off")
    plt.tight_layout()
    plt.savefig(SAMPLES_DIR / "real_vs_fake_grid.png", dpi=120); plt.close()


def figure_fft_comparison() -> None:
    real_v = _first_video(DATA_ROOT / "original")
    fake_v = _first_video(DATA_ROOT / "Deepfakes") or _first_video(DATA_ROOT / "DeepFakeDetection")
    if real_v is None or fake_v is None:
        return
    real_f = _read_middle(real_v); fake_f = _read_middle(fake_v)
    if real_f is None or fake_f is None:
        return

    real_mag = _fft_log_mag(real_f); fake_mag = _fft_log_mag(fake_f)
    plt.imsave(FFT_DIR / "fft_real.png", real_mag, cmap="gray")
    plt.imsave(FFT_DIR / "fft_fake.png", fake_mag, cmap="gray")

    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    ax[0, 0].imshow(real_f); ax[0, 0].set_title("Real Frame"); ax[0, 0].axis("off")
    ax[0, 1].imshow(real_mag, cmap="gray"); ax[0, 1].set_title("Real FFT Magnitude"); ax[0, 1].axis("off")
    ax[1, 0].imshow(fake_f); ax[1, 0].set_title("Fake Frame"); ax[1, 0].axis("off")
    ax[1, 1].imshow(fake_mag, cmap="gray"); ax[1, 1].set_title("Fake FFT Magnitude"); ax[1, 1].axis("off")
    plt.tight_layout(); plt.savefig(FFT_DIR / "fft_comparison.png", dpi=120); plt.close()


# --------------------------------------------------------------------------- #
# Confusion matrix + comparison figures
# --------------------------------------------------------------------------- #

def _load_metrics() -> dict[str, dict]:
    metrics = {}
    if not METRIC_DIR.exists():
        return metrics
    for p in METRIC_DIR.glob("*_metrics.json"):
        name = p.stem.replace("_metrics", "")
        with open(p) as fp:
            metrics[name] = json.load(fp)
    return metrics


def figure_confusion_matrices() -> None:
    metrics = _load_metrics()
    if not metrics:
        return
    for name, m in metrics.items():
        cm = m.get("confusion_matrix")
        if cm is None:
            continue
        plt.figure(figsize=(4.5, 4))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"{name} Confusion Matrix"); plt.colorbar()
        plt.xticks([0, 1], ["Real", "Fake"]); plt.yticks([0, 1], ["Real", "Fake"])
        plt.xlabel("Predicted"); plt.ylabel("True")
        vmax = max(map(max, cm)) if cm else 1
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                plt.text(j, i, str(cm[i][j]), ha="center", va="center",
                         color="white" if cm[i][j] > vmax / 2 else "black")
        plt.tight_layout(); plt.savefig(CM_DIR / f"{name}_cm.png", dpi=120); plt.close()


def figure_comparison() -> None:
    metrics = _load_metrics()
    if not metrics:
        return
    # Stable model order
    order = [m for m in ("spatial", "frequency", "fusion", "attention_fusion", "vgg16") if m in metrics]
    order += sorted(set(metrics) - set(order))

    metric_names = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(order)); width = 0.2

    plt.figure(figsize=(max(7, 1.4 * len(order)), 4.5))
    for i, mn in enumerate(metric_names):
        vals = [metrics[m].get(mn, 0.0) for m in order]
        plt.bar(x + i * width, vals, width, label=mn.capitalize())
    plt.xticks(x + 1.5 * width, order, rotation=15)
    plt.ylim(0, 1.05); plt.ylabel("Score"); plt.title("Model Performance Comparison")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(CMP_DIR / "model_comparison.png", dpi=120); plt.close()

    # Table image
    rows = []
    for m in order:
        r = metrics[m]
        rows.append([m] + [f"{r.get(mn, 0):.3f}" for mn in metric_names + ["roc_auc", "loss"]])
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Loss"]
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.45 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.4)
    plt.title("Per-Model Metrics", pad=10)
    plt.savefig(CMP_DIR / "metric_table.png", dpi=120, bbox_inches="tight"); plt.close()


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #

def figure_gradcam() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split

    from video_dataset import FaceForensicsVideoDataset
    from fusion_attention_model import AttentionFusionCNN

    ckpt = CKPT_DIR / "attention_fusion.pt"
    if not ckpt.exists():
        # Fall back to legacy checkpoint location used by training script.
        legacy = SRC_DIR / "outputs" / "checkpoints" / "attention_fusion_best.pt"
        if legacy.exists():
            ckpt = legacy
        else:
            print("Skipping Grad-CAM: no attention_fusion checkpoint found.")
            return

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = AttentionFusionCNN().to(device)
    state = __import__("torch").load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = FaceForensicsVideoDataset(
        dataset_root=str(DATA_ROOT), max_real=20, max_fake_per_csv=5,
        frame_mode="middle", image_size=128,
    )
    if len(ds) == 0:
        print("Skipping Grad-CAM: empty dataset.")
        return
    n_train = int(0.8 * len(ds)); n_val = len(ds) - n_train
    g = __import__("torch").Generator().manual_seed(0)
    _, val = random_split(ds, [n_train, n_val], generator=g)
    loader = DataLoader(val, batch_size=1, shuffle=True)

    last_conv = None
    for module in model.spatial_branch.features:
        if isinstance(module, nn.Conv2d):
            last_conv = module
    activations: dict = {}; gradients: dict = {}

    def fwd(_, __, output): activations["v"] = output.detach()
    def bwd(_, grad_in, grad_out): gradients["v"] = grad_out[0].detach()
    last_conv.register_forward_hook(fwd)
    last_conv.register_full_backward_hook(bwd)

    def overlay_cam(rgb01: np.ndarray, cam: np.ndarray) -> np.ndarray:
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.clip(0.55 * rgb01 + 0.45 * heat, 0, 1)

    n_each = 3
    real_done = fake_done = 0
    rows = []
    for image, fft_image, label in loader:
        if real_done >= n_each and fake_done >= n_each:
            break
        lbl = int(label.item())
        if lbl == 0 and real_done >= n_each: continue
        if lbl == 1 and fake_done >= n_each: continue

        img = image.to(device).clone().detach().requires_grad_(True)
        ff = fft_image.to(device)
        model.zero_grad()
        logit = model(img, ff)
        prob = float(__import__("torch").sigmoid(logit).item())
        logit.sum().backward()
        weights = gradients["v"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations["v"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        rgb01 = img.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        rgb01 = (rgb01 - rgb01.min()) / (rgb01.max() - rgb01.min() + 1e-8)
        overlay = overlay_cam(rgb01, cam)

        idx = real_done if lbl == 0 else fake_done
        tag = "real" if lbl == 0 else "fake"
        plt.imsave(GRADCAM_DIR / f"gradcam_{tag}_{idx}.png", overlay)

        rows.append((tag, idx, rgb01, overlay, prob))
        if lbl == 0: real_done += 1
        else: fake_done += 1

    if rows:
        n_show = max(real_done, fake_done)
        fig, axes = plt.subplots(2, 2 * n_show, figsize=(2.5 * 2 * n_show, 5))
        # row 0: REAL frames + overlays; row 1: FAKE frames + overlays
        for tag, idx, rgb01, overlay, prob in rows:
            row = 0 if tag == "real" else 1
            base_col = idx * 2
            axes[row, base_col].imshow(rgb01); axes[row, base_col].set_title(f"{tag.upper()} #{idx}")
            axes[row, base_col].axis("off")
            axes[row, base_col + 1].imshow(overlay); axes[row, base_col + 1].set_title(f"Grad-CAM p(fake)={prob:.2f}")
            axes[row, base_col + 1].axis("off")
        for r in (0, 1):
            for c in range(2 * n_show):
                axes[r, c].set_facecolor("white")
        plt.tight_layout(); plt.savefig(GRADCAM_DIR / "gradcam_grid.png", dpi=120); plt.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate labeled image outputs")
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--skip-samples", action="store_true")
    p.add_argument("--skip-fft", action="store_true")
    p.add_argument("--skip-cm", action="store_true")
    p.add_argument("--skip-comparison", action="store_true")
    p.add_argument("--skip-gradcam", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dirs()

    if not args.skip_eda:
        print("[1/6] EDA figures...")
        figure_class_distribution()
        figure_method_breakdown()
        figure_metadata_distributions()
    if not args.skip_samples:
        print("[2/6] Sample frame grid...")
        figure_sample_grid()
    if not args.skip_fft:
        print("[3/6] FFT comparison...")
        figure_fft_comparison()
    if not args.skip_cm:
        print("[4/6] Confusion matrices...")
        figure_confusion_matrices()
    if not args.skip_comparison:
        print("[5/6] Model comparison...")
        figure_comparison()
    if not args.skip_gradcam:
        print("[6/6] Grad-CAM (skip with --skip-gradcam if no checkpoints)...")
        try:
            figure_gradcam()
        except Exception as e:
            print(f"Grad-CAM skipped: {e}")

    print(f"\nAll figures written under: {FIG_ROOT}")


if __name__ == "__main__":
    main()
