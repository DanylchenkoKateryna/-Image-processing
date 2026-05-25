"""
DnCNN-S-15 pretrained denoising (grayscale, sigma=15).
The .pth file stores the full model (torch.save(model, ...)).
"""
import os
import math
import warnings
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

WEIGHTS  = "DnCNN-S-15.pth"
SIGMA    = 15 / 255.0
OUT_DIR  = "2_2_dncnn"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)


class DnCNN(nn.Module):
    """DnCNN-S (grayscale) — must match the saved model's class definition."""
    def __init__(self, channels=1, num_of_layers=17):
        super().__init__()
        features = 64
        padding  = 1
        layers   = [nn.Conv2d(channels, features, 3, padding=padding, bias=True),
                    nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers += [nn.Conv2d(features, features, 3, padding=padding, bias=False),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, channels, 3, padding=padding, bias=False)]
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)   # residual: subtract predicted noise


# Load full model (weights_only=False required for legacy torch.save(model))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
model.eval()
print("DnCNN-S-15 loaded.")

# Load test image as grayscale
img_path = "dataset/test/cat/0000.png"
img      = Image.open(img_path).convert("L")
img_np   = np.array(img, dtype=np.float32) / 255.0

# Add Gaussian noise (sigma=15/255)
np.random.seed(42)
noisy_np = np.clip(img_np + np.random.normal(0, SIGMA, img_np.shape), 0, 1).astype(np.float32)

# Tensor (1, 1, H, W)
x_t     = torch.from_numpy(img_np[None, None]).to(DEVICE)
noisy_t = torch.from_numpy(noisy_np[None, None]).to(DEVICE)

with torch.no_grad():
    denoised_t = model(noisy_t).clamp(0, 1)

def save_gray(t, path):
    arr = (t.squeeze().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

save_gray(x_t,        os.path.join(OUT_DIR, "original.png"))
save_gray(noisy_t,    os.path.join(OUT_DIR, "noisy.png"))
save_gray(denoised_t, os.path.join(OUT_DIR, "dncnn_denoised.png"))

mse_n = float(torch.mean((noisy_t    - x_t) ** 2))
mse_d = float(torch.mean((denoised_t - x_t) ** 2))
psnr_n = -10 * math.log10(mse_n) if mse_n > 1e-10 else 100.0
psnr_d = -10 * math.log10(mse_d) if mse_d > 1e-10 else 100.0
print(f"Noisy    PSNR: {psnr_n:.2f} dB")
print(f"Denoised PSNR: {psnr_d:.2f} dB")
print(f"Improvement: +{psnr_d - psnr_n:.2f} dB")
print("Saved: original.png, noisy.png, dncnn_denoised.png ->", OUT_DIR)
