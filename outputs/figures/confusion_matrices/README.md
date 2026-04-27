# `figures/confusion_matrices/` - 2x2 Confusion Matrices

Each model is evaluated on the held-out validation split. The confusion matrix
is always laid out as:

```
            Predicted
              Real  Fake
True Real      TN    FP
     Fake      FN    TP
```

Produced by `scripts/train_all.py` and re-rendered by `scripts/generate_figures.py`
from the JSONs in `outputs/metrics/`.

| File                          | Source model                       |
|-------------------------------|-------------------------------------|
| `spatial_cm.png`              | RGB-only `SimpleCNN`.              |
| `frequency_cm.png`            | FFT-only `SimpleCNN`.              |
| `fusion_cm.png`               | `FusionCNN` (concat fusion).       |
| `attention_fusion_cm.png`     | `AttentionFusionCNN`.              |
| `vgg16_cm.png`                | `VGG16TransferModel`.              |
