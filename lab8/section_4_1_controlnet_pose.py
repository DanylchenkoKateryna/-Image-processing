"""
Section 4.1 — ControlNet with Canny edge conditioning.
Uses the local controlnet/ (lllyasviel/sd-controlnet-canny) weights
and runwayml/stable-diffusion-v1-5 as the base model.
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline,
                       ControlNetModel, UniPCMultistepScheduler)

CONTROLNET_PATH = "controlnet"
SD_MODEL        = "runwayml/stable-diffusion-v1-5"
INPUT_IMAGE     = "controlnet_test/input.png"
OUT_DIR         = "4_1_controlnet_canny"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE           = torch.float16 if torch.cuda.is_available() else torch.float32
os.makedirs(OUT_DIR, exist_ok=True)


def make_canny(image: Image.Image, lo=100, hi=200) -> Image.Image:
    """Extract Canny edges and convert to 3-channel PIL image."""
    arr   = np.array(image.convert("RGB"))
    edges = cv2.Canny(arr, lo, hi)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)


# Load input image
print("Loading input image:", INPUT_IMAGE)
img_pil = Image.open(INPUT_IMAGE).convert("RGB").resize((512, 512), Image.BICUBIC)
canny   = make_canny(img_pil)
canny.save(os.path.join(OUT_DIR, "canny_input.png"))
print("Canny edge map saved.")

# Load models
print(f"Loading ControlNet from {CONTROLNET_PATH} ...")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=DTYPE)

print(f"Loading SD pipeline from {SD_MODEL} ...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    SD_MODEL,
    controlnet=controlnet,
    torch_dtype=DTYPE,
    safety_checker=None,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=False)

# Generate
PROMPTS = [
    "a portrait of a woman with detailed features, oil painting style, masterpiece",
    "a cyberpunk warrior, neon lights, highly detailed, 8k",
    "a sketch portrait, pencil drawing, black and white",
]

generator = torch.Generator(DEVICE).manual_seed(42)
for i, prompt in enumerate(PROMPTS):
    print(f"\nGenerating ({i+1}/{len(PROMPTS)}): {prompt[:50]}")
    out = pipe(
        prompt,
        image=canny,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        generator=generator,
    ).images[0]
    fname = os.path.join(OUT_DIR, f"controlnet_out_{i+1:02d}.png")
    out.save(fname)
    print(f"  Saved: {fname}")

# Save side-by-side comparison for first prompt
fig_imgs = [img_pil, canny]
titles   = ["Original input", "Canny edges"]
for i in range(min(2, len(PROMPTS))):
    p = os.path.join(OUT_DIR, f"controlnet_out_{i+1:02d}.png")
    if os.path.exists(p):
        fig_imgs.append(Image.open(p))
        titles.append(f"Generated {i+1}")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(fig_imgs), figsize=(5 * len(fig_imgs), 5))
for ax, img, title in zip(axes, fig_imgs, titles):
    ax.imshow(np.array(img))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "controlnet_comparison.png"), dpi=100)
plt.close()
print("\nSaved controlnet_comparison.png")
print("Done. Results in:", OUT_DIR)
