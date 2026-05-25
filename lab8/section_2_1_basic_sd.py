"""
Section 2.1 — Basic Stable Diffusion text-to-image generation.
Uses runwayml/stable-diffusion-v1-5 (downloads ~4 GB on first run, then cached).
"""
import os
import torch
from diffusers import StableDiffusionPipeline

SD_MODEL = "runwayml/stable-diffusion-v1-5"
OUT_DIR  = "2_1_basic_sd"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if torch.cuda.is_available() else torch.float32
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading SD pipeline ({SD_MODEL}) on {DEVICE} ...")
pipe = StableDiffusionPipeline.from_pretrained(
    SD_MODEL,
    torch_dtype=DTYPE,
    safety_checker=None,
)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=False)

prompts = [
    "a beautiful mountain landscape with a lake, photorealistic",
    "a futuristic city at night with neon lights",
    "a cozy cottage in the forest, warm light, autumn",
]

generator = torch.Generator(DEVICE).manual_seed(42)

for i, prompt in enumerate(prompts):
    print(f"\nGenerating ({i+1}/{len(prompts)}): {prompt}")
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    fname = os.path.join(OUT_DIR, f"generated_{i+1:02d}.png")
    image.save(fname)
    print(f"  Saved: {fname}")

print("\nDone. Results in:", OUT_DIR)
