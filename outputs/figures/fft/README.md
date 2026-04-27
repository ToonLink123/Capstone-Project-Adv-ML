# `figures/fft/` - Frequency-Domain Visualizations

Generative models leave signatures in the frequency domain (cross-shaped energy
spikes, periodic peaks, unnatural high-frequency content). These figures show
those signatures qualitatively.

| File                  | What it shows                                                            |
|-----------------------|---------------------------------------------------------------------------|
| `fft_real.png`        | Log-magnitude FFT of a single REAL middle frame (256x256, gray colormap).|
| `fft_fake.png`        | Log-magnitude FFT of a single FAKE middle frame (256x256, gray colormap).|
| `fft_comparison.png`  | 2x2 grid: [Real frame | Real FFT] / [Fake frame | Fake FFT].             |

Backed by `np.fft.fftshift(np.fft.fft2(gray)) -> log(|.| + eps)`.
