"""
Section 2.3 — Generation with Ukrainian prompts and negative prompts.
Compares results with and without negative prompts.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

SD_MODEL = "runwayml/stable-diffusion-v1-5"
OUT_DIR  = "2_3_negative_prompt"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if torch.cuda.is_available() else torch.float32
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading pipeline on {DEVICE} ...")
pipe = StableDiffusionPipeline.from_pretrained(
    SD_MODEL, torch_dtype=DTYPE, safety_checker=None)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)

# Ukrainian subject translated to English for SD (SD uses English prompts best)
PROMPTS = [
    ("Карпатські гори на світанку, фотореалістично",
     "Carpathian mountains at dawn, photorealistic, highly detailed, 8k"),
    ("Замок на скелі над річкою, середньовіччя",
     "Castle on a cliff above a river, medieval, cinematic lighting"),
    ("Соняшникове поле влітку, Україна",
     "Sunflower field in summer, Ukraine, golden hour"),
]

NEGATIVE = ("blurry, low quality, deformed, ugly, bad anatomy, "
            "watermark, text, oversaturated, out of focus")

fig, axes = plt.subplots(len(PROMPTS), 2, figsize=(10, 5 * len(PROMPTS)))

for i, (ua_prompt, en_prompt) in enumerate(PROMPTS):
    gen = torch.Generator(DEVICE).manual_seed(7)

    # Without negative prompt
    img_pos = pipe(en_prompt, num_inference_steps=30, guidance_scale=7.5,
                   generator=gen).images[0]

    gen = torch.Generator(DEVICE).manual_seed(7)

    # With negative prompt
    img_neg = pipe(en_prompt, negative_prompt=NEGATIVE,
                   num_inference_steps=30, guidance_scale=7.5,
                   generator=gen).images[0]

    img_pos.save(os.path.join(OUT_DIR, f"no_neg_{i+1}.png"))
    img_neg.save(os.path.join(OUT_DIR, f"with_neg_{i+1}.png"))

    axes[i][0].imshow(np.array(img_pos))
    axes[i][0].set_title(f"No negative\n{ua_prompt}", fontsize=8)
    axes[i][0].axis("off")
    axes[i][1].imshow(np.array(img_neg))
    axes[i][1].set_title(f"With negative prompt\n{ua_prompt}", fontsize=8)
    axes[i][1].axis("off")
    print(f"  Done: {ua_prompt[:40]}")

plt.suptitle("Effect of negative prompts on image quality", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "negative_comparison.png"), dpi=100)
plt.close()
print("Saved negative_comparison.png")
print("Done. Results in:", OUT_DIR)
