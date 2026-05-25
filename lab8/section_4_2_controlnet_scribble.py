"""
Section 4.2 — ControlNet with scribble-style conditioning.
Creates a simplified scribble map from the input image by dilating
strong Canny edges, then uses the same canny ControlNet to generate
a realistic image from that coarse "scribble" guide.
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
INPUT_IMAGE     = "controlnet_test/input_image_vermeer.png"
OUT_DIR         = "4_2_controlnet_scribble"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE           = torch.float16 if torch.cuda.is_available() else torch.float32
os.makedirs(OUT_DIR, exist_ok=True)


def make_scribble(image: Image.Image) -> Image.Image:
    """
    Create a scribble-like edge map:
    1. Detect coarse Canny edges (low thresholds)
    2. Dilate to simulate thick hand-drawn strokes
    """
    arr    = np.array(image.convert("RGB"))
    gray   = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thick  = cv2.dilate(edges, kernel, iterations=2)
    scrib  = np.stack([thick] * 3, axis=-1)
    return Image.fromarray(scrib)


# Load & prepare input
print("Loading input image:", INPUT_IMAGE)
if not os.path.exists(INPUT_IMAGE):
    # Fall back to the other test image
    INPUT_IMAGE = "controlnet_test/input.png"
    print("  Falling back to:", INPUT_IMAGE)

img_pil = Image.open(INPUT_IMAGE).convert("RGB").resize((512, 512), Image.BICUBIC)
scribble = make_scribble(img_pil)
scribble.save(os.path.join(OUT_DIR, "scribble_input.png"))
img_pil.save(os.path.join(OUT_DIR, "original.png"))
print("Scribble map saved.")

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

PROMPTS = [
    "a beautiful realistic portrait painting, Girl with a Pearl Earring style, "
    "Vermeer painting, oil on canvas, museum quality",
    "a watercolor portrait of a young woman, soft colors, detailed",
]

generator = torch.Generator(DEVICE).manual_seed(123)
outputs = []
for i, prompt in enumerate(PROMPTS):
    print(f"\nGenerating ({i+1}/{len(PROMPTS)}): {prompt[:55]}")
    out = pipe(
        prompt,
        image=scribble,
        num_inference_steps=30,
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.8,
        generator=generator,
    ).images[0]
    fname = os.path.join(OUT_DIR, f"scribble_out_{i+1:02d}.png")
    out.save(fname)
    outputs.append(out)
    print(f"  Saved: {fname}")

# Comparison figure
import matplotlib.pyplot as plt
all_imgs   = [img_pil, scribble] + outputs
all_titles = ["Original", "Scribble map"] + [f"Output {i+1}" for i in range(len(outputs))]
fig, axes = plt.subplots(1, len(all_imgs), figsize=(5 * len(all_imgs), 5))
for ax, img, title in zip(axes, all_imgs, all_titles):
    ax.imshow(np.array(img))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scribble_comparison.png"), dpi=100)
plt.close()
print("\nSaved scribble_comparison.png")
print("Done. Results in:", OUT_DIR)
