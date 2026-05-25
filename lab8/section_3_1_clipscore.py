"""
Section 3.1 — CLIP-based text-image similarity (CLIPScore).
Measures cosine similarity between CLIP text and image embeddings.
"""
import os
import glob
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

CLIP_MODEL = "openai/clip-vit-base-patch32"
OUT_DIR    = "3_1_clipscore"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading CLIP model ({CLIP_MODEL}) ...")
model     = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
model.eval()


def clip_score(image: Image.Image, text: str) -> float:
    """Cosine similarity between CLIP image and text embeddings (0-100 scale)."""
    inputs = processor(text=[text], images=image,
                       return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    # Works across all transformers versions
    img_emb = outputs.image_embeds
    txt_emb = outputs.text_embeds
    if not isinstance(img_emb, torch.Tensor):
        img_emb = img_emb.pooler_output
    if not isinstance(txt_emb, torch.Tensor):
        txt_emb = txt_emb.pooler_output
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return float((img_emb * txt_emb).sum()) * 100.0


# Collect images from 2_1_basic_sd if they exist, else use controlnet_test
PAIRS = []
search_dirs = ["2_1_basic_sd", "2_4_styles", "controlnet_test"]
for d in search_dirs:
    for f in sorted(glob.glob(os.path.join(d, "*.png")) +
                    glob.glob(os.path.join(d, "*.jpg")))[:2]:
        PAIRS.append(f)

# Define prompts for known images
PROMPT_MAP = {
    "2_1_basic_sd": "a beautiful mountain landscape with a lake",
    "2_4_styles":   "a cat sitting on a windowsill",
    "controlnet_test": "a portrait of a person",
}

# Also add some explicit pairs to demonstrate matching vs non-matching
TEST_IMAGE = "controlnet_test/input.png"
TEST_PAIRS = [
    (TEST_IMAGE, "a portrait painting of a person"),
    (TEST_IMAGE, "a dog running in a park"),
    (TEST_IMAGE, "a landscape with mountains"),
]

print("\n" + "=" * 60)
print(f"{'Image':<35} {'Prompt':<20} {'Score':>6}")
print("=" * 60)
results = []
for img_path, prompt in TEST_PAIRS:
    if not os.path.exists(img_path):
        continue
    img   = Image.open(img_path).convert("RGB")
    score = clip_score(img, prompt)
    results.append((img_path, prompt, score))
    print(f"{os.path.basename(img_path):<35} {prompt[:20]:<20} {score:>6.2f}")

# Extra: evaluate generated images from 2_1_basic_sd if available
prompts_2_1 = [
    "a beautiful mountain landscape with a lake, photorealistic",
    "a futuristic city at night with neon lights",
    "a cozy cottage in the forest, warm light, autumn",
]
for i, prompt in enumerate(prompts_2_1):
    img_path = os.path.join("2_1_basic_sd", f"generated_{i+1:02d}.png")
    if not os.path.exists(img_path):
        continue
    img   = Image.open(img_path).convert("RGB")
    score = clip_score(img, prompt)
    results.append((img_path, prompt, score))
    print(f"{os.path.basename(img_path):<35} {prompt[:20]:<20} {score:>6.2f}")
print("=" * 60)

if results:
    avg = np.mean([r[2] for r in results])
    print(f"\nAverage CLIPScore: {avg:.2f}")

# Save summary
with open(os.path.join(OUT_DIR, "clipscore_results.txt"), "w") as f:
    f.write("CLIPScore Results\n")
    f.write("=" * 60 + "\n")
    for img_path, prompt, score in results:
        f.write(f"{img_path}\n  prompt: {prompt}\n  score:  {score:.2f}\n\n")
    if results:
        f.write(f"Average: {np.mean([r[2] for r in results]):.2f}\n")

print("Saved clipscore_results.txt")
print("Done. Results in:", OUT_DIR)
