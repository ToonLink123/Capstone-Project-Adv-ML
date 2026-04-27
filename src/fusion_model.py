import torch
import torch.nn as nn


class SmallCNNBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class FusionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial_branch = SmallCNNBranch(in_channels=3)
        self.frequency_branch = SmallCNNBranch(in_channels=1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, image, fft_image):
        spatial_features = self.spatial_branch(image)
        frequency_features = self.frequency_branch(fft_image)

        combined = torch.cat((spatial_features, frequency_features), dim=1)
        return self.classifier(combined)