# `figures/samples/` - Sample Frame Grids

Visual references for what a "real" vs a "fake" video frame looks like in
this subset. Useful in slide decks and during qualitative inspection.

| File                       | What it shows                                                            |
|----------------------------|---------------------------------------------------------------------------|
| `real_vs_fake_grid.png`    | Top row: middle frame from one REAL video. Bottom row: middle frame from one FAKE video per manipulation method (DeepFakeDetection, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures). |

Frames are extracted with `cv2.VideoCapture` at `frame_count // 2`.
