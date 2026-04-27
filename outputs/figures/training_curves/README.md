# `figures/training_curves/` - Per-Model Loss Curves

Train vs validation BCE-with-logits loss as a function of epoch.
Produced by `scripts/train_all.py`.

| File                          | Model                                                       |
|-------------------------------|-------------------------------------------------------------|
| `spatial_loss.png`            | RGB-only `SimpleCNN` (3 input channels).                    |
| `frequency_loss.png`          | FFT-only `SimpleCNN` (1 input channel, log-magnitude FFT).  |
| `fusion_loss.png`             | `FusionCNN` - concatenation of spatial + frequency branches.|
| `attention_fusion_loss.png`   | `AttentionFusionCNN` - learned softmax weighting of branches.|
| `vgg16_loss.png`              | `VGG16TransferModel` - frozen ImageNet backbone + MLP head. |

Each curve is saved at 120 dpi PNG with `Train` / `Val` legend.
