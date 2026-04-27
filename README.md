# Multi-domain, Artifact-aware, and Explanation-integrated Universal Synthetic Image Detection

**Capstone project, Advanced Machine Learning **
**Authors:** Tanvir Mahmud Prince and Eshan Agarwal

Sorts real photographs / video frames from
AI-generated content. The system uses spatial (RGB) features with
frequency-domain (FFT) features through both a simple concatenation head
and a learned attention gate, then explains its decisions with Grad-CAM
artifact localization heatmaps.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Dataset](#dataset)
4. [Models](#models)
5. [Reproducible Pipeline](#reproduce)
6. [Outputs](#outputs)
7. [Notebook (Google Colab)](#notebook-google-colab)

---

## Project Overview

Modern generative models (GANs, diffusion, face-swap pipelines) produce media
that fools human observers. Existing detectors typically:

1. Generalize poorly to unseen generators.
2. Latch onto generator specific fingerprints rather than universal artifacts.
3. Provide no explanation beyond a binary label.

This project implements a hybrid multi-domain detection architecture

Five model variants are trained and compared on the FaceForensics++ subset:
`spatial`, `frequency`, `fusion`, `attention_fusion`, and `vgg16` (transfer
learning). All five share the same evaluation so their accuracy /
precision / recall / F1 / ROC-AUC / confusion matrices are directly comparable

## Quick Start

```bash
git clone https://github.com/ToonLink123/Capstone-Project-Adv-ML.git
cd Capstone-Project-Adv-ML
python -m venv .venv && source .venv/bin/activate   # or: conda env create -f environment.yml
pip install -r requirements.txt
python scripts/download_data.py
python scripts/preprocess.py --max-real 60 --max-fake-per-method 12
python scripts/train_all.py --epochs 5 --batch-size 8
python scripts/generate_figures.py
```

## Dataset

We host a curated FaceForensics++ subset on Hugging Face:

> https://huggingface.co/datasets/Toonlink/Capstone

## Models

| Model              | Module                              | Inputs              | Notes                                    |
|--------------------|-------------------------------------|---------------------|------------------------------------------|
| `spatial`          | `src/model.py:SimpleCNN`            | RGB (3-ch)          | 3-block conv -> MLP head.                |
| `frequency`        | `src/model.py:SimpleCNN`            | FFT log-mag (1-ch)  | Same backbone, single-channel input.     |
| `fusion`           | `src/fusion_model.py:FusionCNN`     | RGB + FFT           | Concatenation of two pooled branches.    |
| `attention_fusion` | `src/fusion_attention_model.py`     | RGB + FFT           | Softmax-weighted fusion (learned alpha). |
| `vgg16`            | `scripts/train_all.py`              | RGB resized to 224  | Frozen ImageNet backbone + MLP head.     |

All models output a single logit and are trained with `BCEWithLogitsLoss`

## Reproduce

| Step                | Command                                                              | Outputs                                     |
|---------------------|----------------------------------------------------------------------|---------------------------------------------|
| Download dataset    | `python scripts/download_data.py`                                    | `data/`                                     |
| Preprocess          | `python scripts/preprocess.py`                                       | `data/preprocessed/{frames,fft}/...`        |
| Train all models    | `python scripts/train_all.py`                                        | `outputs/{checkpoints,metrics,figures}/...` |
| Generate figures    | `python scripts/generate_figures.py`                                 | `outputs/figures/**`                        |

## Outputs
- [`outputs/figures/eda/`](outputs/figures/eda/README.md)
- [`outputs/figures/samples/`](outputs/figures/samples/README.md)
- [`outputs/figures/fft/`](outputs/figures/fft/README.md)
- [`outputs/figures/training_curves/`](outputs/figures/training_curves/README.md)
- [`outputs/figures/confusion_matrices/`](outputs/figures/confusion_matrices/README.md)
- [`outputs/figures/comparison/`](outputs/figures/comparison/README.md)
- [`outputs/figures/gradcam/`](outputs/figures/gradcam/README.md)
- [`outputs/metrics/`](outputs/metrics/README.md)
- [`outputs/checkpoints/`](outputs/checkpoints/README.md)

## Notebook

`notebooks/Capstone_Final_Universal_Synthetic_Image_Detection.ipynb`

1. Sets up GPU configuration and downloads data from HuggingFace
2. Loads and prepares the FaceForensics++ dataset from HuggingFace
3. Defines dataset classes and FFT helpers inline
4. Defines all model variants (SimpleCNN, FusionCNN, AttentionFusionCNN, VGG16)
5. Implements GPU-optimized training and evaluation loops
6. Trains all model variants and persists checkpoints / metrics / figures
7. Produces confusion matrices, loss curves, and Grad-CAM explainability overlays
8. Saves results to `outputs/` directory