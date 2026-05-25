"""
Section 3.2 — Fréchet Inception Distance (FID).
Compares a set of real images to generated (or noisy) images.
Uses torchmetrics.image.FrechetInceptionDistance.
"""
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

REAL_DIR  = "fid_test_images"
OUT_DIR   = "3_2_fid"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = (299, 299)
os.makedirs(OUT_DIR, exist_ok=True)


def load_images_as_tensor(folder: str, size=IMG_SIZE) -> torch.Tensor:
    """Load all images in folder → uint8 tensor (N, 3, H, W)."""
    paths = sorted(glob.glob(os.path.join(folder, "*.png")) +
                   glob.glob(os.path.join(folder, "*.jpg")))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(size, Image.BICUBIC)
        imgs.append(np.array(img, dtype=np.uint8))
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)  # (N,3,H,W)


def add_noise_batch(t: torch.Tensor, std: float) -> torch.Tensor:
    """Add Gaussian noise to uint8 tensor (returns uint8 clamped)."""
    noisy = t.float() + torch.randn_like(t.float()) * std * 255
    return noisy.clamp(0, 255).to(torch.uint8)


print("Loading real images ...")
real_imgs = load_images_as_tensor(REAL_DIR)
print(f"  {real_imgs.shape[0]} real images loaded")

# If generated images from 2_1_basic_sd exist, use those; otherwise use noisy copies
gen_dir = "2_1_basic_sd"
if os.path.isdir(gen_dir) and glob.glob(os.path.join(gen_dir, "*.png")):
    print("Loading generated images from", gen_dir)
    gen_imgs = load_images_as_tensor(gen_dir)
else:
    print("No generated images found — using noisy copies of real images for demo")
    gen_imgs = add_noise_batch(real_imgs, std=0.15)

print(f"  {gen_imgs.shape[0]} generated images")

# ── FID computation ──
# Note: FID needs ≥2 images per split in each distribution.
# With very few images the value is not meaningful — this is a demo.
fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

real_f = real_imgs.to(DEVICE)
gen_f  = gen_imgs.to(DEVICE)

fid_metric.update(real_f, real=True)
fid_metric.update(gen_f,  real=False)
fid_value = float(fid_metric.compute())

print(f"\nFID score: {fid_value:.4f}")
print("(Lower is better; 0 = identical distributions)")

# Save result
with open(os.path.join(OUT_DIR, "fid_result.txt"), "w") as f:
    f.write(f"Real images:     {real_imgs.shape[0]} from {REAL_DIR}\n")
    f.write(f"Generated images:{gen_imgs.shape[0]}\n")
    f.write(f"FID:             {fid_value:.4f}\n")
    f.write("Note: meaningful FID requires 10,000+ images.\n")

print("Saved fid_result.txt")
print("Done. Results in:", OUT_DIR)
