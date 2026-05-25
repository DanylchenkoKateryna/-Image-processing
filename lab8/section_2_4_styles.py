"""
Section 2.4 — Style variations for the same subject.
Generates 4 artistic styles for the same scene.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

SD_MODEL = "runwayml/stable-diffusion-v1-5"
OUT_DIR  = "2_4_styles"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if torch.cuda.is_available() else torch.float32
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading pipeline on {DEVICE} ...")
pipe = StableDiffusionPipeline.from_pretrained(
    SD_MODEL, torch_dtype=DTYPE, safety_checker=None)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)

SUBJECT = "a cat sitting on a windowsill looking outside"

STYLES = [
    ("Photorealistic",
     f"{SUBJECT}, photorealistic, 8k, DSLR, sharp focus"),
    ("Oil painting",
     f"{SUBJECT}, oil painting, impressionist, thick brushstrokes, "
     "museum quality, Monet style"),
    ("Anime",
     f"{SUBJECT}, anime style, Studio Ghibli, soft colors, detailed, "
     "hand-drawn"),
    ("Pencil sketch",
     f"{SUBJECT}, pencil sketch, black and white, crosshatching, "
     "detailed drawing"),
]

NEGATIVE = "blurry, low quality, deformed, watermark, text"

fig, axes = plt.subplots(1, len(STYLES), figsize=(16, 5))
for i, (style_name, prompt) in enumerate(STYLES):
    print(f"  Generating: {style_name}")
    gen = torch.Generator(DEVICE).manual_seed(42)
    img = pipe(prompt, negative_prompt=NEGATIVE,
               num_inference_steps=35, guidance_scale=8.0,
               generator=gen).images[0]
    img.save(os.path.join(OUT_DIR, f"style_{i+1}_{style_name.lower().replace(' ', '_')}.png"))
    axes[i].imshow(np.array(img))
    axes[i].set_title(style_name, fontsize=11)
    axes[i].axis("off")

plt.suptitle(f'Subject: "{SUBJECT}"', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "styles_comparison.png"), dpi=100)
plt.close()
print("Saved styles_comparison.png")
print("Done. Results in:", OUT_DIR)
