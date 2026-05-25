import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128


def get_cat_loaders(batch_size=8, train_size=500, test_size=50):
    t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_ds = Subset(datasets.ImageFolder("dataset/train", transform=t), range(train_size))
    test_ds  = Subset(datasets.ImageFolder("dataset/test",  transform=t), range(test_size))
    return (DataLoader(train_ds, batch_size, shuffle=True,  num_workers=0),
            DataLoader(test_ds,  batch_size, shuffle=False, num_workers=0))


class _Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class GeneratorUNet(nn.Module):
    """Conditional U-Net generator for 128x128 images (denoising / inpainting)."""
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1   = _Block(in_ch, 32)
        self.e2   = _Block(32, 64)
        self.e3   = _Block(64, 128)
        self.bot  = _Block(128, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3   = _Block(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2   = _Block(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1   = _Block(64, 32)
        self.out  = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b  = self.bot(self.pool(e3))
        d3 = self.d3(torch.cat([self.up3(b),  e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch * 2, 64,  4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,       128,  4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,      256,  4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,        1,  4, padding=1),
        )

    def forward(self, cond, img):
        return self.model(torch.cat([cond, img], 1))


class _ResBlock(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
    def forward(self, x):
        return x + self.conv(x)


class SRGenerator(nn.Module):
    """4x super-resolution generator: 32x32 -> 128x128 (ESRGAN-lite)."""
    def __init__(self, in_ch=3, out_ch=3, n_res=6):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.res  = nn.Sequential(*[_ResBlock(64) for _ in range(n_res)])
        self.up   = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(True),
        )
        self.tail = nn.Sequential(nn.Conv2d(64, out_ch, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        h = self.head(x)
        return self.tail(self.up(self.res(h) + h))


def psnr(out, tgt):
    mse = torch.mean((out - tgt) ** 2).item()
    return -10.0 * math.log10(mse) if mse > 1e-10 else 100.0


def add_gaussian_noise(x, sigma=0.1):
    return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)


def add_mask(x, mask_size=32):
    x = x.clone()
    B, _, H, W = x.shape
    for i in range(B):
        y  = random.randint(0, H - mask_size)
        xp = random.randint(0, W - mask_size)
        x[i, :, y:y + mask_size, xp:xp + mask_size] = 0.0
    return x


def train_gan_epoch(G, D, loader, opt_G, opt_D, device, corrupt_fn, target_fn=None, lambda_l1=100):
    G.train(); D.train()
    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()
    g_total = d_total = 0.0
    for x, _ in loader:
        x    = x.to(device)
        cond = corrupt_fn(x)
        real = target_fn(x) if target_fn else x
        fake = G(cond)
        # --- Discriminator ---
        r_pred = D(cond, real)
        f_pred = D(cond, fake.detach())
        loss_D = 0.5 * (bce(r_pred, torch.ones_like(r_pred)) +
                        bce(f_pred, torch.zeros_like(f_pred)))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()
        # --- Generator ---
        f_pred = D(cond, fake)
        loss_G = bce(f_pred, torch.ones_like(f_pred)) + lambda_l1 * l1(fake, real)
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()
        g_total += loss_G.item()
        d_total += loss_D.item()
    return g_total / len(loader), d_total / len(loader)
