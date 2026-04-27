# `figures/eda/` - Exploratory Data Analysis

Quick visual sanity checks on the FaceForensics++ subset *before* any training.
Produced by `scripts/generate_figures.py` (steps reading `data/csv/*.csv`).

| File                              | What it shows                                                                 |
|-----------------------------------|--------------------------------------------------------------------------------|
| `class_distribution.png`          | Total REAL vs FAKE video count aggregated across every CSV.                    |
| `method_breakdown.png`            | FAKE video count broken out per manipulation method (DeepFakeDetection, ...). |
| `frame_count_distribution.png`    | Histogram of frame counts across all videos (real + fake).                    |
| `file_size_distribution.png`      | Histogram of video file sizes in MB.                                          |

These plots correspond to "Section 3. Dataset Analysis" of the
*Final Capstone Milestone Report* in `references/`.
