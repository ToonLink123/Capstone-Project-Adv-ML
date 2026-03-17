import torch


def to_fft(images):
    gray = images.mean(dim=1, keepdim=True)
    fft = torch.fft.fft2(gray)
    fft = torch.fft.fftshift(fft)
    mag = torch.log(torch.abs(fft) + 1e-8)
    return mag