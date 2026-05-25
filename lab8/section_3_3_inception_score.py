"""
Section 3.3 — Inception Score (IS).
Uses torchmetrics.image.InceptionScore.
Higher IS = more diverse & recognisable images.
"""
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchmetrics.image.inception import InceptionScore

OUT_DIR  = "3_3_inception_score"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (299, 299)
os.makedirs(OUT_DIR, exist_ok=True)


def load_images_as_tensor(folder: str, size=IMG_SIZE) -> torch.Tensor:
    paths = sorted(glob.glob(os.path.join(folder, "*.png")) +
                   glob.glob(os.path.join(folder, "*.jpg")))
    if not paths:
        return None
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(size, Image.BICUBIC)
        imgs.append(np.array(img, dtype=np.uint8))
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)


# Prefer generated images; fall back to real test images
sources = ["2_1_basic_sd", "2_4_styles", "fid_test_images", "controlnet_test"]
imgs = None
used_dir = None
for d in sources:
    if os.path.isdir(d):
        t = load_images_as_tensor(d)
        if t is not None and t.shape[0] > 0:
            imgs = t
            used_dir = d
            break

if imgs is None:
    raise FileNotFoundError("No images found in any expected directory.")

print(f"Loaded {imgs.shape[0]} images from '{used_dir}'")
print(f"Computing Inception Score on {DEVICE} ...")

is_metric = InceptionScore(normalize=True).to(DEVICE)
is_metric.update(imgs.to(DEVICE))
is_mean, is_std = is_metric.compute()
is_mean = float(is_mean)
is_std  = float(is_std)

print(f"\nInception Score: {is_mean:.4f} ± {is_std:.4f}")
print("(IS for ImageNet ~300; typical diffusion outputs ~3-15 for few images)")

with open(os.path.join(OUT_DIR, "inception_score.txt"), "w") as f:
    f.write(f"Images:          {imgs.shape[0]} from {used_dir}\n")
    f.write(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}\n")
    f.write("Note: IS is meaningful only with diverse sets of 10,000+ images.\n")

print("Saved inception_score.txt")
print("Done. Results in:", OUT_DIR)
