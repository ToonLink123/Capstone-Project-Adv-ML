# `figures/comparison/` - Cross-Model Comparison

Side-by-side comparison of every model's classification metrics, computed on
the same validation split (seed = 42).

| File                    | What it shows                                                        |
|-------------------------|-----------------------------------------------------------------------|
| `model_comparison.png`  | Grouped bar chart over Accuracy / Precision / Recall / F1 per model.  |
| `metric_table.png`      | Rendered table image of model x {Accuracy, Precision, Recall, F1, ROC-AUC, Loss}. |

Generated from the per-model JSON files in `outputs/metrics/` by
`scripts/generate_figures.py`.
