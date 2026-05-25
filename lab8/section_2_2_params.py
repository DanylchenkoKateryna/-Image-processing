"""
Section 2.2 — Hyperparameter grid: guidance_scale × num_inference_steps.
Generates a 3×3 grid of images varying these two parameters.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline

SD_MODEL = "runwayml/stable-diffusion-v1-5"
OUT_DIR  = "2_2_params"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if torch.cuda.is_available() else torch.float32
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading pipeline on {DEVICE} ...")
pipe = StableDiffusionPipeline.from_pretrained(
    SD_MODEL, torch_dtype=DTYPE, safety_checker=None)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)

PROMPT          = "a serene mountain lake at sunset, photorealistic"
GUIDANCE_SCALES = [3.0, 7.5, 15.0]
INFERENCE_STEPS = [10, 30, 50]

images = []
labels = []

for steps in INFERENCE_STEPS:
    for gs in GUIDANCE_SCALES:
        print(f"  steps={steps}, guidance={gs}")
        gen = torch.Generator(DEVICE).manual_seed(42)
        img = pipe(PROMPT, num_inference_steps=steps,
                   guidance_scale=gs, generator=gen).images[0]
        img.save(os.path.join(OUT_DIR, f"gs{gs}_steps{steps}.png"))
        images.append(img)
        labels.append(f"gs={gs}\nsteps={steps}")

# Build comparison grid
fig, axes = plt.subplots(len(INFERENCE_STEPS), len(GUIDANCE_SCALES),
                         figsize=(12, 12))
for idx, (ax, img, lbl) in enumerate(zip(axes.flat, images, labels)):
    ax.imshow(np.array(img))
    ax.set_title(lbl, fontsize=9)
    ax.axis("off")
plt.suptitle(f'Prompt: "{PROMPT}"', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "params_grid.png"), dpi=100)
plt.close()
print("Saved params_grid.png")
print("Done. Results in:", OUT_DIR)
