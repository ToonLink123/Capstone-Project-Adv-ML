#!/usr/bin/env python
# coding: utf-8

# Multi-domain Synthetic Image Detection - Final Notebook
# Authors: Tanvir Mahmud Prince, Eshan Agarwal
# Dataset: FaceForensics++ subset + CIFAKE-style image split (100K train, 20K test)

# Imports, Config & GPU Setup

# In[1]:


import json, random, warnings, sys, os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
DATA_ROOT = PROJECT_ROOT / 'data'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
FIG_DIR = OUTPUT_ROOT / 'figures'
METRIC_DIR = OUTPUT_ROOT / 'metrics'
CKPT_DIR = OUTPUT_ROOT / 'checkpoints'
for d in [FIG_DIR, METRIC_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Seeding
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# GPU detection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    torch.backends.cudnn.benchmark = True

scaler = GradScaler(enabled=(DEVICE == 'cuda'))
print(f'Mixed precision: {"ENABLED" if DEVICE == "cuda" else "DISABLED"}')


# Load Full Dataset
# Using CIFAKE-style image split: 100K training images (50K real + 50K fake) and 20K test images

# In[ ]:


def to_fft_batch(images):
    """Batch FFT: (B,3,H,W) -> (B,1,H,W) log-magnitude."""
    g = images.mean(dim=1, keepdim=True)
    f = torch.fft.fft2(g)
    return torch.log(torch.abs(torch.fft.fftshift(f)) + 1e-8)


class CIFAKEImageDataset(Dataset):
    """Loads REAL/FAKE images from data/train and data/test."""
    def __init__(self, root_dir, max_per_class=None, image_size=128):
        self.root = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.samples = []
        for cls_name, label in [('REAL', 0), ('FAKE', 1)]:
            d = self.root / cls_name
            paths = sorted([p for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.webp') for p in d.rglob(ext)])
            if max_per_class is not None:
                paths = paths[:max_per_class]
            self.samples.extend([(str(p), label) for p in paths])
        print(f'Loaded {len(self.samples)} samples from {root_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        # On-the-fly FFT
        gray = image.mean(dim=0, keepdim=True)
        f = torch.fft.fft2(gray)
        mag = torch.log(torch.abs(torch.fft.fftshift(f)) + 1e-8)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        return image, mag.float(), torch.tensor(label, dtype=torch.float32)


# In[ ]:


MAX_PER_CLASS = 10000  # 10K real + 10K fake = 20K training, 20K test total
IMAGE_SIZE = 128

train_dataset = CIFAKEImageDataset(
    DATA_ROOT / 'train',
    max_per_class=MAX_PER_CLASS,
    image_size=IMAGE_SIZE
)
test_dataset = CIFAKEImageDataset(
    DATA_ROOT / 'test',
    max_per_class=MAX_PER_CLASS,
    image_size=IMAGE_SIZE
)

print(f'Train: {len(train_dataset)} | Test: {len(test_dataset)}')


# In[ ]:


BATCH_SIZE = 128
EPOCHS = 10
NUM_WORKERS = 0  # Windows + multiprocessing = broken pipe; use 0 for reliable execution

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

print(f'Train batches: {len(train_loader)} | Test batches: {len(test_loader)}')
print(f'Samples per epoch: {len(train_dataset)}')
print(f'Total samples in one epoch: {len(train_dataset)}')


# Model Definitions
# Four variants: spatial, frequency, fusion, attention_fusion

# In[ ]:


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class _SmallBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    def forward(self, x):
        return self.features(x).view(x.size(0), -1)

class FusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_branch = _SmallBranch(3)
        self.frequency_branch = _SmallBranch(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, img, fft):
        return self.classifier(torch.cat(
            (self.spatial_branch(img), self.frequency_branch(fft)), dim=1))


class _AttnBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    def forward(self, x):
        return self.features(x).view(x.size(0), -1)

class AttentionFusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_branch = _AttnBranch(3)
        self.frequency_branch = _AttnBranch(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, img, fft):
        s = self.spatial_branch(img)
        f = self.frequency_branch(fft)
        w = self.attention(torch.cat([s, f], dim=1))
        return self.classifier(
            w[:, 0].unsqueeze(1) * s + w[:, 1].unsqueeze(1) * f)


# Training & Evaluation Utilities (GPU-optimized)

# In[ ]:


def get_logits(model, images, fft_images, model_name):
    if model_name in ('fusion', 'attention_fusion'):
        return model(images, fft_images)
    if model_name == 'frequency':
        return model(to_fft_batch(images))
    return model(images)

def make_model(model_name):
    if model_name == 'fusion':           return FusionCNN().to(DEVICE)
    if model_name == 'attention_fusion': return AttentionFusionCNN().to(DEVICE)
    if model_name == 'frequency':        return SimpleCNN(in_channels=1).to(DEVICE)
    return SimpleCNN(in_channels=3).to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion, model_name):
    model.train()
    total = 0.0
    for images, fft_images, labels in tqdm(loader, desc='Train', leave=False):
        images = images.to(DEVICE, non_blocking=True)
        fft_images = fft_images.to(DEVICE, non_blocking=True)
        labels = labels.float().view(-1, 1).to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with autocast(enabled=(DEVICE == 'cuda')):
            logits = get_logits(model, images, fft_images, model_name)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, model_name):
    model.eval()
    total = 0.0
    preds, probs_all, labels_all = [], [], []
    for images, fft_images, labels in tqdm(loader, desc='Eval', leave=False):
        images = images.to(DEVICE, non_blocking=True)
        fft_images = fft_images.to(DEVICE, non_blocking=True)
        labels_t = labels.float().view(-1, 1).to(DEVICE, non_blocking=True)
        with autocast(enabled=(DEVICE == 'cuda')):
            logits = get_logits(model, images, fft_images, model_name)
            total += criterion(logits, labels_t).item()
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds.extend((probs > 0.5).astype(int).tolist())
        probs_all.extend(probs.tolist())
        labels_all.extend(labels.cpu().numpy().tolist())
    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else float('nan')
    return {
        'loss': total / len(loader),
        'accuracy': float(accuracy_score(labels_all, preds)),
        'precision': float(precision_score(labels_all, preds, zero_division=0)),
        'recall': float(recall_score(labels_all, preds, zero_division=0)),
        'f1': float(f1_score(labels_all, preds, zero_division=0)),
        'roc_auc': float(auc) if not np.isnan(auc) else None,
        'confusion_matrix': confusion_matrix(labels_all, preds).tolist(),
    }


def plot_losses(train_losses, val_losses, out_path, title):
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Test')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.show(); plt.close()


def plot_confusion(cm, title, out_path):
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title); plt.colorbar()
    plt.xticks([0, 1], ['Real', 'Fake'])
    plt.yticks([0, 1], ['Real', 'Fake'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    vmax = max(map(max, cm)) if cm else 1
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center',
                     color='white' if cm[i][j] > vmax/2 else 'black')
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.show(); plt.close()


# ## 5. Train All Four Models
# 
# Each model trains for 10 epochs on 20K balanced training samples.
# Expect ~70-85% accuracy range; anything near 100% or 50% would indicate a bug.
# On a T4 GPU, each epoch takes about 30-60 seconds.

# In[ ]:


def run_experiment(model_name, epochs=EPOCHS, lr=1e-3):
    print(f'\n{"="*60}\nTraining: {model_name.upper()}\n{"="*60}')
    model = make_model(model_name)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {param_count:,}')

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    # Simple step decay scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    test_losses = []
    best_f1 = 0.0

    for epoch in range(epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, model_name)
        metrics = evaluate(model, test_loader, criterion, model_name)
        scheduler.step()

        train_losses.append(tr_loss)
        test_losses.append(metrics['loss'])

        # Sanity check: flag suspicious results
        if metrics['accuracy'] >= 0.98:
            print('  *** WARNING: Accuracy >= 98% - check for data leak or overfitting ***')
        if metrics['accuracy'] <= 0.52:
            print('  *** WARNING: Accuracy near chance level - check training ***')

        print(f'  Ep {epoch+1}/{epochs} | '
              f'tr={tr_loss:.4f} te={metrics["loss"]:.4f} | '
              f'acc={metrics["accuracy"]:.4f} prec={metrics["precision"]:.4f} '
              f'rec={metrics["recall"]:.4f} f1={metrics["f1"]:.4f} '
              f'auc={metrics["roc_auc"]}')

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), CKPT_DIR / f'{model_name}_best.pt')

    # Final evaluation
    final = evaluate(model, test_loader, criterion, model_name)

    # Save loss plot
    plot_losses(train_losses, test_losses,
                FIG_DIR / f'{model_name}_loss.png',
                f'{model_name} - Train/Test Loss')

    # Save confusion matrix
    if final.get('confusion_matrix'):
        plot_confusion(final['confusion_matrix'],
                       FIG_DIR / f'{model_name}_cm.png',
                       f'{model_name} Confusion Matrix')

    # Save metrics
    with open(METRIC_DIR / f'{model_name}_metrics.json', 'w') as fp:
        json.dump(final, fp, indent=2)

    print(f'\nFinal {model_name}:')
    print(f'  Acc={final["accuracy"]:.4f} | F1={final["f1"]:.4f} | AUC={final["roc_auc"]}')

    # Free GPU memory
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return final


# In[ ]:


if __name__ == '__main__':
    results = {}
    for name in ['spatial', 'frequency', 'fusion', 'attention_fusion']:
        metrics = run_experiment(name, epochs=EPOCHS)
        results[name] = metrics
    print('\nAll models trained successfully.')

    # Results table
    rows = []
    model_order = [('Spatial CNN', 'spatial'), ('Frequency FFT-CNN', 'frequency'),
                   ('Fusion (concat)', 'fusion'), ('Attention Fusion', 'attention_fusion')]
    for display_name, key in model_order:
        m = results[key]
        rows.append({'Model': display_name, 'Accuracy': round(m['accuracy'], 4),
                     'Precision': round(m['precision'], 4), 'Recall': round(m['recall'], 4),
                     'F1': round(m['f1'], 4),
                     'ROC-AUC': round(m['roc_auc'], 4) if m['roc_auc'] else 'N/A',
                     'Loss': round(m['loss'], 4)})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(METRIC_DIR / 'summary.csv', index=False)
    print('\n=== FINAL COMPARISON TABLE ===')
    print(summary_df.to_string(index=False))
    print('\nSanity check:')
    for _, r in summary_df.iterrows():
        if r['Accuracy'] >= 0.98:
            print(f'  WARNING: {r["Model"]} suspiciously high ({r["Accuracy"]})')
        elif r['Accuracy'] <= 0.52:
            print(f'  WARNING: {r["Model"]} near chance ({r["Accuracy"]})')
        else:
            print(f'  OK: {r["Model"]} = {r["Accuracy"]}')

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(summary_df)); width = 0.2
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
    for i, mname in enumerate(metrics_to_plot):
        ax.bar(x + i*width, summary_df[mname], width, label=mname, color=colors[i])
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(summary_df['Model'], rotation=15, fontsize=11)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison (20K test samples)', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.savefig(FIG_DIR / 'model_comparison.png', dpi=120); plt.close()

    # Grad-CAM
    best_ckpt = CKPT_DIR / 'attention_fusion_best.pt'
    if not best_ckpt.exists():
        print('No attention_fusion checkpoint, skipping Grad-CAM.')
    else:
        print(f'Loading {best_ckpt} for Grad-CAM...')
        attn_model = AttentionFusionCNN().to(DEVICE)
        attn_model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        attn_model.eval()
        last_conv = None
        for mod in attn_model.spatial_branch.features:
            if isinstance(mod, nn.Conv2d):
                last_conv = mod
        class _GradCAM:
            def __init__(self, model, target_layer):
                self.model = model.eval()
                self.activations = self.gradients = None
                target_layer.register_forward_hook(lambda _,__,o: setattr(self,'activations',o.detach()))
                target_layer.register_full_backward_hook(lambda _,__,go: setattr(self,'gradients',go[0].detach()))
            def __call__(self, image, fft_image):
                self.model.zero_grad()
                self.model(image, fft_image).sum().backward()
                w = self.gradients.mean(dim=(2,3), keepdim=True)
                cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=image.shape[-2:], mode='bilinear', align_corners=False)
                cam = cam.squeeze().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                return cam
        cam_engine = _GradCAM(attn_model, last_conv)
        def overlay(rgb01, cam):
            h = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
            h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return np.clip(0.55 * rgb01 + 0.45 * h, 0, 1)
        n_show = 4; n_per = n_show // 2; sr = sf = 0
        fig2, axes = plt.subplots(2, n_show, figsize=(3*n_show, 6))
        for images, fft_images, labels in test_loader:
            if sr >= n_per and sf >= n_per: break
            for i in range(len(labels)):
                if sr >= n_per and sf >= n_per: break
                lbl = int(labels[i].item())
                if lbl == 0 and sr >= n_per: continue
                if lbl == 1 and sf >= n_per: continue
                img = images[i].unsqueeze(0).to(DEVICE).clone().detach().requires_grad_(True)
                ff = fft_images[i].unsqueeze(0).to(DEVICE)
                cam = cam_engine(img, ff)
                prob = float(torch.sigmoid(attn_model(img, ff)).item())
                rgb = img.detach().squeeze().permute(1,2,0).cpu().numpy()
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                ov = overlay(rgb, cam)
                col = sr if lbl == 0 else (n_per + sf)
                axes[0,col].imshow(rgb); axes[0,col].axis('off')
                axes[0,col].set_title(f'{"REAL" if lbl==0 else "FAKE"}\\np={prob:.3f}', fontsize=10)
                axes[1,col].imshow(ov); axes[1,col].axis('off'); axes[1,col].set_title('Grad-CAM', fontsize=10)
                if lbl == 0: sr += 1
                else: sf += 1
        plt.tight_layout(); plt.savefig(FIG_DIR / 'gradcam_examples.png', dpi=120); plt.close()

    print(f'\nDone. Results saved to: {FIG_DIR}, {METRIC_DIR}, {CKPT_DIR}')
