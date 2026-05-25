"""
Section 1.1 — CLIP: пошук зображення за текстовим запитом.
Порівнює зображення та текстові описи у спільному векторному просторі CLIP.
"""
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

print(f"Завантаження моделі {model_name} на {device} ...")
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

image_paths = [
    "scan_images/scan_mobile.jpg",
    "scan_images/scan_receipt_1.jpg",
    "scan_images/foto_terminal.png",
    "scan_images/scan_table.jpg",
]

texts = [
    "касовий чек з магазину",
    "сканована сторінка книжки",
    "термінал з командами Linux",
    "сканована сторінка з таблицями",
]

print("Завантаження зображень ...")
images = [Image.open(p).convert("RGB") for p in image_paths]

inputs = processor(
    text=texts,
    images=images,
    return_tensors="pt",
    padding=True
).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds  = outputs.text_embeds

# Нормалізація векторів
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds  = text_embeds  / text_embeds.norm(p=2, dim=-1, keepdim=True)

# Косинусна подібність (text × image)
similarity = text_embeds @ image_embeds.T

print("\n" + "=" * 60)
for i, text in enumerate(texts):
    sims = similarity[i]
    best_img_idx = sims.argmax().item()
    print(f"Текст: {text}")
    print(f"Найбільш подібне зображення: {image_paths[best_img_idx]}")
    print(f"Симілярність: {sims[best_img_idx]:.3f}")
    print("-" * 40)

print("\nМатриця подібності (рядки=тексти, стовпці=зображення):")
print("          ", "  ".join([p.split("/")[-1][:12] for p in image_paths]))
for i, text in enumerate(texts):
    row = "  ".join([f"{similarity[i][j]:.3f}" for j in range(len(image_paths))])
    print(f"{text[:18]:<20} {row}")
