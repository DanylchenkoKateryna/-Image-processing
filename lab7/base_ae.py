import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_loaders(batch_size=128, root="data"):
    t = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root, train=True,  download=False, transform=t)
    test_ds  = datasets.CIFAR10(root, train=False, download=False, transform=t)
    return (DataLoader(train_ds, batch_size, shuffle=True,  num_workers=0),
            DataLoader(test_ds,  batch_size, shuffle=False, num_workers=0))


class ConvAE(nn.Module):
    """Convolutional autoencoder for 32x32 images."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 8x8
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),   # 16x16
            nn.ConvTranspose2d(32, 3,  2, stride=2), nn.Sigmoid(), # 32x32
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class ConvAE128(nn.Module):
    """Super-resolution autoencoder: 32x32 -> 128x128."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),   # 64x64
            nn.ConvTranspose2d(64, 32, 2, stride=2),  nn.ReLU(),   # 128x128
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def psnr(out, target):
    mse = torch.mean((out - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * math.log10(mse)


def add_gaussian_noise(x, sigma=0.1):
    return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)


def add_blur(x, kernel_size=5):
    pad = kernel_size // 2
    return F.avg_pool2d(F.pad(x, [pad] * 4, mode="reflect"), kernel_size, stride=1)


def add_mask(x, mask_size=8):
    x = x.clone()
    B, _, H, W = x.shape
    for i in range(B):
        y  = random.randint(0, H - mask_size)
        xp = random.randint(0, W - mask_size)
        x[i, :, y:y + mask_size, xp:xp + mask_size] = 0.0
    return x


def train_epoch(model, loader, optimizer, criterion, device, corrupt_fn, target_fn=None):
    model.train()
    total = 0.0
    for x, _ in loader:
        x   = x.to(device)
        inp = corrupt_fn(x)
        tgt = target_fn(x) if target_fn else x
        out = model(inp)
        loss = criterion(out, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, device, corrupt_fn, target_fn=None):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x   = x.to(device)
            inp = corrupt_fn(x)
            tgt = target_fn(x) if target_fn else x
            out = model(inp)
            total += psnr(out, tgt)
            n += 1
    return total / n


def show(inp, out, target, path, n=4):
    """Save grid: [noisy/low-res | restored | original], n samples per row."""
    grid = torch.cat([inp[:n].cpu(), out[:n].cpu(), target[:n].cpu()], dim=0)
    vutils.save_image(grid, path, nrow=n, normalize=False)
    print(f"Saved: {path}")
