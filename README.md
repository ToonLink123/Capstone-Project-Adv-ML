# Capstone Milestone: Universal Synthetic Image Detection

## Overview
This project investigates synthetic image and video frame detection using a multi-domain approach. The milestone focuses on two baseline classifiers:

- **Spatial CNN** operating on RGB frames
- **Frequency FFT-CNN** operating on FFT magnitude representations

The long-term goal is to build a unified artifact aware and explanation integrated detector that generalizes across manipulation methods.

## Dataset
Due to access issues with the GenImage dataset, this milestone uses **FaceForensics++**. The dataset contains:

- `original/` videos used as real samples
- manipulation folders such as:
  - `DeepFakeDetection/`
  - `Deepfakes/`
  - `Face2Face/`
  - `FaceShifter/`
  - `FaceSwap/`
  - `NeuralTextures/`
- CSV metadata files in `data/csv/` containing file paths and labels

The pipeline reads video paths directly from the CSV metadata rather than manually restructuring the dataset.

## Milestone Goals
This milestone implements the proposal stages corresponding to:

1. Dataset preprocessing
2. Spatial baseline classifier
3. Frequency-domain baseline classifier
4. Preliminary evaluation and analysis