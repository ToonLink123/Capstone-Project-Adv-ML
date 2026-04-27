import torch
import torch.nn as nn

class CNNBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class AttentionFusionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial_branch = CNNBranch(in_channels=3)
        self.frequency_branch = CNNBranch(in_channels=1)

        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, image, fft_image):
        spatial_features = self.spatial_branch(image)
        frequency_features = self.frequency_branch(fft_image)

        combined = torch.cat([spatial_features, frequency_features], dim=1)

        weights = self.attention(combined)

        spatial_weight = weights[:, 0].unsqueeze(1)
        frequency_weight = weights[:, 1].unsqueeze(1)

        fused_features = (
            spatial_weight * spatial_features
            + frequency_weight * frequency_features
        )

        logits = self.classifier(fused_features)
        return logits