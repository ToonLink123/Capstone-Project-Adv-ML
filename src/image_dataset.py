from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def make_fft_image(image_tensor):
    gray = image_tensor.mean(dim=0, keepdim=True)

    fft = torch.fft.fft2(gray)
    fft = torch.fft.fftshift(fft)

    magnitude = torch.log(torch.abs(fft) + 1e-8)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

    return magnitude.float()


class RealFakeImageDataset(Dataset):
    def __init__(self, root_dir, image_size=128, max_per_class=None):
        self.root_dir = Path(root_dir)
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        real_dir = self.root_dir / "REAL"
        fake_dir = self.root_dir / "FAKE"

        real_images = []
        fake_images = []

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            real_images.extend(real_dir.rglob(ext))
            fake_images.extend(fake_dir.rglob(ext))

        real_images = sorted(real_images)
        fake_images = sorted(fake_images)

        if max_per_class is not None:
            real_images = real_images[:max_per_class]
            fake_images = fake_images[:max_per_class]

        for path in real_images:
            self.samples.append((str(path), 0))

        for path in fake_images:
            self.samples.append((str(path), 1))

        print(f"Loaded image dataset from {self.root_dir}")
        print(f"REAL images: {len(real_images)}")
        print(f"FAKE images: {len(fake_images)}")
        print(f"Total images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        fft_image = make_fft_image(image)

        return image, fft_image, torch.tensor(label, dtype=torch.float32)