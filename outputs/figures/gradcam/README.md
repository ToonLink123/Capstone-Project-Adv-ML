# `figures/gradcam/` - Grad-CAM Artifact Localization

Grad-CAM heatmaps over the final conv layer of the `AttentionFusionCNN` spatial
branch, satisfying the proposal's third research goal:
**"How can artifact localization be integrated to provide interpretable explanations?"**

| File                       | What it shows                                                    |
|----------------------------|------------------------------------------------------------------|
| `gradcam_real_<i>.png`     | Single Grad-CAM overlay for a REAL example (i = 0, 1, 2).        |
| `gradcam_fake_<i>.png`     | Single Grad-CAM overlay for a FAKE example (i = 0, 1, 2).        |
| `gradcam_grid.png`         | Combined grid: top row REAL (frame + heatmap pairs), bottom row FAKE. Each title shows `p(fake)` as predicted by the attention model. |

Heatmaps use `cv2.COLORMAP_JET` blended with the original frame at 0.45 opacity.

> Requires a trained checkpoint at
> `outputs/checkpoints/attention_fusion.pt`. If absent, the script will fall
> back to `src/outputs/checkpoints/attention_fusion_best.pt` (legacy location).
